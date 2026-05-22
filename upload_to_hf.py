import os
from huggingface_hub import HfApi, create_repo, login

def upload_model_to_hf(repo_name: str, local_model_dir: str, local_tokenizer_dir: str, hf_token: str = None):
    """
    Programmatically logs in to Hugging Face, creates a new model repository,
    and uploads the trained state dictionary weights and tokenizer files.
    """
    # 1. Handle Hugging Face Access Token authentication
    if hf_token:
        print("Authenticating with provided token...")
        login(token=hf_token)
    else:
        print("No token provided. Ensure you are already logged in via 'huggingface-cli login'.")

    api = HfApi()
    
    # 2. Extract active Hugging Face username
    user_info = api.whoami()
    username = user_info["name"]
    full_repo_id = f"{username}/{repo_name}"
    
    print(f"\nTarget Repository: https://huggingface.co/{full_repo_id}")
    
    # 3. Create the remote model repository
    print(f"Creating repository '{full_repo_id}' on Hugging Face Hub...")
    try:
        create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True)
        print("✔ Repository initialized successfully.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
        
    # 4. Define files to upload
    files_to_upload = {}
    
    # Locate model state dict weights
    weights_path = os.path.join(local_model_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        files_to_upload["pytorch_model.bin"] = weights_path
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Train the model first!")
        
    # Locate tokenizer configuration files
    if os.path.exists(local_tokenizer_dir):
        for filename in os.listdir(local_tokenizer_dir):
            file_path = os.path.join(local_tokenizer_dir, filename)
            if os.path.isfile(file_path):
                files_to_upload[filename] = file_path
                
    # 5. Upload files to the remote repository
    print("\nStarting upload of model and tokenizer assets to Hugging Face Hub...")
    for remote_path, local_path in files_to_upload.items():
        print(f"  Uploading {remote_path} ({os.path.getsize(local_path) / (1024*1024):.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=full_repo_id,
                repo_type="model"
            )
            print(f"  ✔ {remote_path} uploaded.")
        except Exception as e:
            print(f"  ❌ Error uploading {remote_path}: {e}")
            
    print(f"\n🎉 Success! Your model is now hosted live at: https://huggingface.co/{full_repo_id}")
    print(f"You can load this model in your Streamlit app by setting: repo_id='{full_repo_id}'")

if __name__ == "__main__":
    # Interactive CLI execution
    print("--- Hugging Face Model Publisher ---")
    token = input("Enter your Hugging Face Access Token (with WRITE permissions): ").strip()
    repo = input("Enter desired repository name (e.g. news-topic-classifier-bert): ").strip()
    
    if not repo:
        repo = "news-topic-classifier-bert"
        
    upload_model_to_hf(
        repo_name=repo,
        local_model_dir="saved_models/bert-agnews",
        local_tokenizer_dir="saved_models/tokenizer",
        hf_token=token
    )
