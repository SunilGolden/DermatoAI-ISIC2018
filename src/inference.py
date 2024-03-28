import json
import argparse
from utils import get_device, load_model, get_data_transforms, preprocess_image, do_inference


def main(args):
    with open(args.config_filepath) as config_file:
        config_file = json.load(config_file)

    device = get_device()

    model = load_model(args.architecture, num_classes=config_file['num_classes'], model_path=args.checkpoint_filename, device=device)
    model.eval()
    
    transform = get_data_transforms()['test']
    img = preprocess_image(args.img_path, transform).to(device)

    predicted_class, confidence = do_inference(model, img, device, config_file['class_names'])
    print(f"Predicted class: {predicted_class} with confidence {confidence*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--architecture', type=str, default='resnet50', choices=['vit', 'resnet50'], help='Architecture to use (vit or resnet50) (default: resnet50)')
    parser.add_argument('--img_path', type=str, default='./data/ISIC2018_Task3_Test_Input/ISIC_0034524.jpg', help='Path to image')
    parser.add_argument('--config_filepath', type=str, default='./config/config.json', help='Path to configuration file (default: ./config/config.json)')
    parser.add_argument('--checkpoint_filename', type=str, default='./weights/best_model.pth', help='Filename to save the best model (default: ./weights/best_model.pth)')
    
    args = parser.parse_args()

    main(args)