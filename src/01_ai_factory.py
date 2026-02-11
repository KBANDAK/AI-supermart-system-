import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from transformers import pipeline

# 1. SETUP FOLDERS
DATA_DIR = "../data"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 2. LOAD MODELS IN SAFE MODE (GTX 1650 Optimized)
print("⏳ Loading AI Models...")

try:
    # A. Load Stable Diffusion (Standard Precision to prevent Black Images)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        safety_checker=None
    )
    
    # B. Enable 'Low VRAM' Mode (Critical for 4GB Cards)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    # C. Load GPT-2 for descriptions
    text_gen = pipeline("text-generation", model="gpt2", device=0)
    print("✅ Models Ready!")

except Exception as e:
    print(f"❌ MODEL CRASH: {e}")
    exit()

# 3. DEFINE THE SUPERMARKET INVENTORY
# We added 30 categories for a full store experience
categories = [
    # Fruits & Veg
    "Red Apple", "Yellow Banana", "Green Avocado", "Orange Citrus", 
    "Fresh Strawberry", "Green Grapes", "Red Tomato", "Carrot Vegetable",
    "Green Broccoli", "Potato", "Red Bell Pepper", "Cucumber",
    
    # Bakery & Breakfast
    "French Croissant", "Baguette Bread", "Glazed Donut", "Chocolate Cake",
    "Sourdough Loaf", "Cereal Box",
    
    # Dairy & Drinks
    "Milk Bottle", "Orange Juice Bottle", "Swiss Cheese Block", "Strawberry Yogurt",
    "Soda Can", "Water Bottle", "Coffee Cup",
    
    # Pantry & Snacks
    "Pasta Box", "Tomato Ketchup Bottle", "Jar of Honey", "Chocolate Bar", "Potato Chips Bag"
]

data = []

print(f"🚀 STARTING MASS GENERATION ({len(categories) * 3} Images)...")
print("☕ This will take about 20-30 minutes. Time for a break!")

for cat in categories:
    print(f"   🎨 Drawing {cat}...")
    
    # GENERATE 3 ITEMS PER CLASS
    for i in range(3): 
        filename = f"{cat.replace(' ', '')}_{i}.jpg"
        path = os.path.join(IMG_DIR, filename)
        
        try:
            # Generate Image (Stable Diffusion)
            # We use a specific prompt to get clean 'Product Photography'
            prompt = f"professional product photo of {cat}, studio lighting, white background, 4k, realistic"
            image = pipe(prompt, num_inference_steps=20).images[0]
            image.save(path)
            
            # Generate Description (GPT-2)
            # We guide GPT-2 to write a sales pitch
            desc_prompt = f"Delicious {cat} for sale. It is"
            desc = text_gen(desc_prompt, max_length=40, num_return_sequences=1)[0]['generated_text']
            desc = desc.replace("\n", " ") # Clean up newlines
            
            # Random Price logic
            import random
            price = round(random.uniform(1.50, 12.50), 2)
            
            data.append({
                "filename": filename, 
                "class": cat, 
                "description": desc, 
                "price": price, 
                "path": path
            })
            print(f"      ✨ Created {filename}")
            
        except Exception as e:
            print(f"      ⚠️ Error creating {filename}: {e}")

# 4. SAVE CSV
if len(data) > 0:
    pd.DataFrame(data).to_csv(os.path.join(DATA_DIR, "ai_data.csv"), index=False)
    print("✅ SUPERMARKET GENERATED! Check 'data/ai_data.csv'.")
else:
    print("❌ No data generated.")