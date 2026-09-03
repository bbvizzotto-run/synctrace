import torch
import os
import sys
import subprocess
import argparse

def check_gpu():
    print("=== GPU Diagnostic ===")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Check your NVIDIA drivers and PyTorch installation.")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} GPU(s).")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    return True

def check_software():
    print("\n=== Software Diagnostic ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    
    try:
        ffmpeg_ver = subprocess.check_output(["ffmpeg", "-version"]).decode().split('\n')[0]
        print(f"FFmpeg: {ffmpeg_ver}")
    except Exception:
        print("[ERROR] FFmpeg not found in PATH.")

def check_datasets(fakeavceleb_path, avtimit_path):
    print("\n=== Dataset Diagnostic ===")
    
    def validate_dir(name, path):
        if not path:
            print(f"[SKIP] {name} path not provided.")
            return
        if os.path.exists(path):
            print(f"[OK] {name} found at: {path}")
            # Check for common subdirs
            contents = os.listdir(path)
            print(f"  Contents (sample): {contents[:5]}")
        else:
            print(f"[ERROR] {name} NOT found at: {path}")

    validate_dir("FakeAVCeleb", fakeavceleb_path)
    validate_dir("AV-LipSync-TIMIT", avtimit_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SyncTrace Local Environment Diagnostic")
    parser.add_argument("--fakeavceleb", type=str, help="Path to FakeAVCeleb dataset root")
    parser.add_argument("--avtimit", type=str, help="Path to AV-LipSync-TIMIT dataset root")
    args = parser.parse_args()

    check_gpu()
    check_software()
    check_datasets(args.fakeavceleb, args.avtimit)
    
    print("\nDiagnostic complete. If all [OK], you are ready to start the campaign.")
