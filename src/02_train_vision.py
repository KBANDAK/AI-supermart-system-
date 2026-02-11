from ultralytics import YOLO
import pandas as pd
import os
import shutil
import multiprocessing

# --- WINDOWS SAFE GUARD ---
def main():
    print("🟢 STARTING ROBUST VISION TRAINING...")

    # 1. SETUP ABSOLUTE PATHS
    # Get the directory where this script lives (src)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define key folders relative to src
    BASE_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))
    MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))
    
    CSV_PATH = os.path.join(BASE_DATA_DIR, "ai_data.csv")
    TRAIN_DIR = os.path.join(BASE_DATA_DIR, "yolo_train")
    IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")

    print(f"📂 Data Source: {BASE_DATA_DIR}")
    print(f"💾 Model Destination: {MODELS_DIR}")

    # 2. VERIFY CSV EXISTS
    if not os.path.exists(CSV_PATH):
        print(f"❌ CRITICAL ERROR: Could not find {CSV_PATH}")
        print("   Did you run '01_ai_factory.py' fully?")
        return

    # 3. ORGANIZE DATA
    print("📂 Organizing Data for YOLO...")
    
    # Clean old training data if exists
    if os.path.exists(TRAIN_DIR):
        try:
            shutil.rmtree(TRAIN_DIR)
        except PermissionError:
            print("⚠️ Warning: Could not delete old folder (file in use). Continuing...")
    
    try:
        df = pd.read_csv(CSV_PATH)
        valid_count = 0
        
        for idx, row in df.iterrows():
            filename = os.path.basename(row['path'])
            src = os.path.join(IMAGES_DIR, filename)
            
            if os.path.exists(src):
                # Clean class name
                cls_name = str(row['class']).replace(" ", "")
                cls_dir = os.path.join(TRAIN_DIR, cls_name)
                os.makedirs(cls_dir, exist_ok=True)
                
                # Copy
                dst = os.path.join(cls_dir, filename)
                shutil.copy(src, dst)
                valid_count += 1
        
        if valid_count == 0:
            print("❌ ERROR: No valid images found!")
            return

        print(f"✅ Organized {valid_count} images.")

    except Exception as e:
        print(f"❌ ORGANIZATION ERROR: {e}")
        return

    # 4. TRAIN YOLO
    print("👁️ Training Vision Model (Relax, this takes ~5 mins)...")
    
    try:
        # Load smallest model
        model = YOLO('yolov8n-cls.pt') 
        
        # Train
        results = model.train(data=TRAIN_DIR, epochs=50, imgsz=224, workers=2)
        
        # --- NEW: SAVE TO MODELS FOLDER ---
        # 1. Ensure models folder exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # 2. Find the 'best.pt' file created by YOLO
        # results.save_dir is usually 'runs/classify/trainX'
        best_model_src = os.path.join(str(results.save_dir), "weights", "best.pt")
        
        # 3. Define our target path
        final_model_path = os.path.join(MODELS_DIR, "yolo_vision.pt")
        
        # 4. Copy it over
        print(f"📦 Copying model from {best_model_src}...")
        shutil.copy(best_model_src, final_model_path)
        
        print("\n------------------------------------------------")
        print("🎉 VISION TRAINING COMPLETE!")
        print(f"✅ Model saved safely at: {final_model_path}")
        print("------------------------------------------------\n")

    except Exception as e:
        print(f"❌ TRAINING CRASHED: {e}")

# --- ENTRY POINT (REQUIRED FOR WINDOWS) ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()