import torchinfo
import torch
from model import YOLO26MultiTask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO26MultiTask(scale="n").to(device)
    # Load the best model weights from stage 2 training
    model.load_state_dict(torch.load("outputs/stage_2/train_2/best.pt", map_location=device))
    # Print a detailed summary of the model architecture and parameter counts
    torchinfo.summary(model, input_size=(1, 3, 480, 640))