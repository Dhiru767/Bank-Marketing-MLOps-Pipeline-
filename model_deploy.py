import os
import shutil

def run():
    model_path = "/home/dhiraj-bohara/projects/MLops/model/bank_model.pkl"
    api_folder = "/home/dhiraj-bohara/projects/MLops/api"
    deployed_model_path = os.path.join(api_folder, "bank_model.pkl")
   
    os.makedirs(api_folder, exist_ok=True)
    
    if os.path.exists(model_path):
        shutil.copy(model_path, deployed_model_path)
        print(f"Model copied to {deployed_model_path} for deployment.")
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure training task succeeded.")
    