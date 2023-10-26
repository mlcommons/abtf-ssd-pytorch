import os
import argparse
import torch
import onnx
from src.model import SSD, ResNet

def get_args():
    parser = argparse.ArgumentParser("Script to export pyTorch model to ONNX")
    parser.add_argument('--weights_file', type=str, help='Path to pre-trained model weights')
    parser.add_argument('--output_dir', type=str, default=os.getcwd(), help='Path to save the exported onnx')
    args = parser.parse_args()
    return args

def export_pyTorch_to_onnx(opt):

    # Sanity checks
    if not opt.weights_file or not os.path.isfile(opt.weights_file):
        print(f'Invalid weights file: \'{opt.weights_file}\'. Exiting.')
        exit(0)

    if not os.path.exists(opt.output_dir):
       os.makedirs(opt.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create pyTorch model from base definition
    model = SSD(backbone=ResNet())
    # Load weights
    checkpoint = torch.load(opt.weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device)
    model.eval()
    # Create dummy input required for ONNX export
    dummy_input = torch.randn((1,3,300,300)).to(device)
    onnx_model_path = os.path.join(opt.output_dir, 'Resnet50_SSD.onnx')
    torch.onnx.export(model,(dummy_input), onnx_model_path,opset_version=11,verbose=True)
    # Optionally, rewrite exported ONNX file with shape markers for convenience
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_model_path)), onnx_model_path)
    print(f'Export to ONNX sucessful. Model saved at {onnx_model_path}')


if __name__ == "__main__":
    opt = get_args()
    export_pyTorch_to_onnx(opt)
