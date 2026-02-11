import torch
from fastapi import FastAPI, UploadFile, File, Form
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import requests  # <--- Using Request to bypass Qdrant Client bugs
from ultralytics import YOLO
from PIL import Image
import io
import os
import uvicorn

app = FastAPI()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))

print("🚀 STARTING GOD MODE SERVER (Vision + Memory + LangChain)...")

# ==========================================
# 1. LOAD VISION (The Eyes)
# ==========================================
yolo_path = os.path.join(MODELS_DIR, "yolo_vision.pt")
if os.path.exists(yolo_path):
    yolo = YOLO(yolo_path)
    print("✅ Vision Model Loaded")
else:
    print("⚠️ Warning: using default YOLO (Not trained on Supermarket)")
    yolo = YOLO("yolov8n-cls.pt")

# ==========================================
# 2. LOAD BRAIN (The LLM)
# ==========================================
print("🧠 Loading TinyLlama Brain...")
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Load Fine-Tuned Adapter
adapter_path = os.path.join(MODELS_DIR, "tinyllama_finetune")
if os.path.exists(adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Adapter Loaded")
else:
    model = base_model

# Create the LangChain Pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=128,  # Keep answers concise
    temperature=0.1,     # Low creativity = High accuracy
    do_sample=True,
    top_k=50,
    top_p=0.95
)
llm = HuggingFacePipeline(pipeline=pipe)

# ==========================================
# 3. SETUP MEMORY (Embeddings Only)
# ==========================================
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 4. DEFINE LANGCHAIN PROMPT
# ==========================================
# This tells the AI exactly how to behave
template = """
<|system|>
You are an intelligent Supermarket Assistant.
Your goal is to answer the user's question based on the visual input and the database memory.
If the Memory contains a price, you MUST state it clearly.
If the item is not found, apologize politely.

VISUAL CONTEXT (What you see): {vision}
DATABASE MEMORY (Facts): {memory}

<|user|>
{question}
<|assistant|>
"""
prompt = PromptTemplate(template=template, input_variables=["vision", "memory", "question"])
chain = prompt | llm

# ==========================================
# 5. THE CHAT ENDPOINT
# ==========================================
@app.post("/chat")
async def chat(text: str = Form(...), file: UploadFile = File(None)):
    vision_label = "Nothing"
    memory_context = "No database information available."
    
    # --- A. VISION PHASE ---
    if file:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        results = yolo(img)
        # Get the top prediction name
        vision_label = results[0].names[results[0].probs.top1]
        print(f"👁️ SEEN: {vision_label}")

        # --- B. MEMORY PHASE (Direct API) ---
        try:
            # 1. Turn Label into Vector
            vector = embedder.encode(vision_label).tolist()
            
            # 2. Ask Qdrant (Raw HTTP Request)
            url = "http://localhost:6333/collections/supermarket_db/points/search"
            payload = {
                "vector": vector,
                "limit": 3,
                "with_payload": True
            }
            response = requests.post(url, json=payload)
            result = response.json().get("result", [])
            
            if result:
                # We found a match!
                best = result[0]['payload']
                price = best['price']
                desc = best['description']
                
                # Truncate description to first sentence for cleaner output
                short_desc = desc.split('.')[0] + "."
                
                print(f"💾 DATABASE MATCH: {best['product_name']} - ${price}")
                
                # Create the Fact Sheet for the AI
                memory_context = f"Product Name: {best['product_name']}. Price: ${price}. Description: {short_desc}"
            else:
                print(f"⚠️ NO MATCH for: {vision_label}")
                memory_context = "The product was seen, but it is not listed in the database."

        except Exception as e:
            print(f"❌ DATABASE ERROR: {e}")
            memory_context = "Database connection failed."

    # --- C. BRAIN PHASE (LangChain) ---
    print("🧠 Generating Answer...")
    
    # Run the Chain
    response = chain.invoke({
        "vision": vision_label,
        "memory": memory_context,
        "question": text
    })

    # Clean up the output (remove the prompt part)
    final_answer = response.split("<|assistant|>")[-1].strip()

    return {
        "reply": final_answer, 
        "seen": vision_label,
        "memory_used": memory_context
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)