import os
import onnx
import torch
import argparse
import torchreid
from torch.autograd import Variable

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='osnet_x1_0', help="ReID model name")
    parser.add_argument('--nc', type=int, default=1000, help="Number of training identities")
    parser.add_argument('--weights', type=str, default='', help="Path to pre-trained weights")
    parser.add_argument('--img_h', type=int, default=256, help="image height")
    parser.add_argument('--img_w', type=int, default=128, help="image width")

    args = parser.parse_args()
    
    return args
    
    
def main(args):
    model = torchreid.models.build_model(name=args.name, num_classes=args.nc)
    if args.weights:
        torchreid.utils.load_pretrained_weights(model, args.weights) 

    input_name  = ['input']
    output_name = ['output']
    save_folder = 'models/onnx'
    os.makedirs(save_folder, exist_ok=True)
    save_path = f'{save_folder}/{args.name}.onnx'
    input_var = Variable(torch.randn(1, 3, args.img_h, args.img_w))
    torch.onnx.export(
        model, input_var, save_path, input_names=input_name, output_names=output_name, verbose=True, export_params=True
    )

    print('Checking converted ONNX model...')
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print('Model was converted successfully.')


if __name__ == "__main__":
    args = get_parser()
    main(args)
