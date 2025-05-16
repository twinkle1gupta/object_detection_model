import os
import ast
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import pynvml

def get_gpu_stats(device_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem.used / (1024 ** 2)  
    mem_total = mem.total / (1024 ** 2)  
    mem_free = mem.free / (1024 ** 2)  
    gpu_util = util.gpu  
    pynvml.nvmlShutdown()
    return gpu_util, mem_used, mem_total, mem_free


class YOLODataset(Dataset):
    def __init__(self, csv_file, S=7, B=2, C=20, img_height=480, img_width=640):
        self.annotations = pd.read_csv(csv_file)
        self.S = S
        self.B = B
        self.C = C
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row['img_path']
        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            return None, None
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None, None
        img = cv2.resize(img, (self.img_width, self.img_height)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW

        
        bboxes = ast.literal_eval(row['img_gt_bbox_coords'])
        labels = ast.literal_eval(row['img_gt_class_labels'])
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C), dtype=torch.float32)

        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            x_center = (xmin + xmax) / 2 / self.img_width
            y_center = (ymin + ymax) / 2 / self.img_height
            w = (xmax - xmin) / self.img_width
            h = (ymax - ymin) / self.img_height

            grid_x = min(int(x_center * self.S), self.S - 1)
            grid_y = min(int(y_center * self.S), self.S - 1)

            x_cell = x_center * self.S - grid_x
            y_cell = y_center * self.S - grid_y

            class_vector = torch.zeros(self.C)
            class_vector[label] = 1.0

            target[grid_y, grid_x, :4] = torch.tensor([x_cell, y_cell, w, h])
            target[grid_y, grid_x, 4] = 1.0  # Objectness
            target[grid_y, grid_x, 5:5+self.C] = class_vector

        return img, target

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class YOLOVGG16(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.conv = nn.Conv2d(512, 30, kernel_size=(9, 14))  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))
        self.sigmoid = nn.Sigmoid()
        self.S, self.B, self.C = S, B, C

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


def train_model(csv_path, batch_size=4, num_epochs=10, S=7, B=2, C=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))
    dataset = YOLODataset(csv_path, S=S, B=B, C=C)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = YOLOVGG16(S=S, B=B, C=C).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, targets) in progress_bar:
            if images is None or targets is None:
                continue
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            
            if device.type == 'cuda':
                gpu_util, mem_used, mem_total, mem_free = get_gpu_stats()
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "GPU%": f"{gpu_util}%",
                    "VRAM": f"{mem_used:.0f}/{mem_total:.0f} MB"
                })
            else:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(dataloader):.4f}')
    return model


if __name__ == "__main__":
    csv_path = 'training_data.csv'  
    trained_model = train_model(csv_path, batch_size=4, num_epochs=10)
    torch.save(trained_model.state_dict(), 'yolo_vgg16.pth')
