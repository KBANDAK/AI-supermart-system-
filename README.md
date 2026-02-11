🛒 AI Supermarket System: A Synthetic Data Ecosystem
An End-to-End AI Project that Generates, Trains, and Sells its own Products.

The Problem: Data Scarcity (The Cold Start)
In the world of Artificial Intelligence, the biggest hurdle is often the lack of high-quality data. We wanted to build a futuristic supermarket scanner, but we faced a common challenge: We didn't have a dataset. We had no product photos, no descriptions, and no price lists.

The Solution: A Self-Generating AI Pipeline
Instead of manually collecting data, we engineered a Synthetic Data Pipeline. We used Generative AI to create the entire supermarket inventory from scratch, effectively teaching our discriminative models (Computer Vision) to "see" products that don't even exist in the real world.

How It Works :
The system operates in three distinct phases:

1. Creation Phase (Generative AI)

Visuals: We used Stable Diffusion to generate photorealistic images of unique products (e.g., "Neon Apples," "Cyberpunk Soda").

Metadata: We used GPT-2 to hallucinate creative product names, descriptions, and pricing strategies for these synthetic items.

2. Learning Phase (Training)

Vision: We trained a YOLOv8 model on this purely synthetic dataset. The AI learned to recognize the features of our AI-generated products with high precision.

Memory: We embedded the synthetic descriptions and prices into Qdrant, a Vector Database, creating a "semantic memory" for the store.

3. Execution Phase (Inference)

The Kiosk: When a user scans one of these AI-generated images at the Streamlit kiosk, the trained YOLO model identifies it.

The Brain: TinyLlama (via LangChain) retrieves the specific details from Qdrant (RAG) and acts as a store assistant, explaining the product and its price.

The Manager: n8n automates the entire flow, ensuring the user gets their receipt instantly.

Why This Matters
This project demonstrates a complete Closed-Loop AI Ecosystem. It proves that Synthetic Data can effectively solve the "Cold Start" problem, allowing engineers to build and train sophisticated Computer Vision and RAG systems even without access to real-world data.

Tech Stack
Generative AI: Stable Diffusion, GPT-2

Computer Vision: YOLOv8

LLM & RAG: TinyLlama, LangChain, Qdrant

Automation: n8n

Frontend: Streamlit
