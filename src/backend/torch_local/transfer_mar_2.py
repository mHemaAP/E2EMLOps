import os
import shutil
import boto3
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Load AWS credentials from .env
load_dotenv()


def upload_folder_to_s3(local_folder, bucket_name, s3_prefix):
    """Recursively uploads a folder to S3."""
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)

    print("✅ Upload complete.")


all_projects = ["vegfruits", "sports"]
s3_bucket = "emlo-project"

for project in all_projects:
    handler_path = f"src/backend/torchserve_app/{project}_handler.py"
    onnx_path = Path("checkpoints/onnxs") / f"{project}.onnx"
    index_to_name_path = Path("checkpoints/model_stores") / project / "index_to_name.json"
    model_store_dir = Path("checkpoints/model_stores") / project
    model_store_dir.mkdir(parents=True, exist_ok=True)

    # Temporary directory for mar creation
    deploy_dir = Path("checkpoints/deployment") / project
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Copy required files for mar creation
    shutil.copy(onnx_path, deploy_dir / f"{project}.onnx")
    shutil.copy(index_to_name_path, deploy_dir / "index_to_name.json")

    # Generate .mar file
    subprocess.run([
        "torch-model-archiver",
        "--model-name", f"{project}-classifier",
        "--version", "1.0",
        "--serialized-file", str(deploy_dir / f"{project}.onnx"),
        "--handler", handler_path,
        "--extra-files", str(deploy_dir / "index_to_name.json"),
        "--export-path", str(deploy_dir),
        "--force"
    ], check=True)

    print(f"✅ {project}.mar generated.") 

    # Move the generated .mar file to model_store_dir
    shutil.move(str(deploy_dir / f"{project}-classifier.mar"), str(model_store_dir / f"{project}-classifier.mar"))

    print(f"✅ {project}.mar generated and moved to model_stores.")

# ✅ Upload ENTIRE checkpoints directory to S3
upload_folder_to_s3("checkpoints", s3_bucket, "checkpoints")

print("✅ Entire 'checkpoints' folder uploaded to s3://emlo-project/checkpoints")
