import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader

from dataloader import HandGestureDataset, check_image_mask_resolutions

class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 59 * 79, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # print(f"Feature map shape before pooling: {x.shape}")
        x = self.pool(x)
        # print(f"Feature map shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Flattened feature vector shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model_1().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


dataset = HandGestureDataset(dataset_path="/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset", transform=None)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

for epoch in tqdm(range(10)):
    model.train()
    total_loss = 0.0
    for images, _masks, class_ids in dataloader:
        images = images.to(device)
        class_ids = class_ids.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, class_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")