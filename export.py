import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
    parser.add_argument('--model_path', type=str, 
                       default="bunch_model.pt",
                       help='Path to the YOLO model file')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for export (default: 640)')
    
    args = parser.parse_args()
    
    # Load the YOLO model
    model = YOLO(args.model_path)
    
    # Export the model to ONNX format
    export_path = model.export(format="onnx", imgsz=args.imgsz)
    
    print(f"Model exported to {export_path}")

if __name__ == "__main__":
    main()