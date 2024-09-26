import torch
import numpy as np
import pickle
import os
from torch.utils.tensorboard import SummaryWriter


# load Data(Cifar-10)
def unpickle(file):
    """Function to load batch data"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_dir = r"cifar-10-batches-py"

# Load training data
train_data = []
train_labels = []
for i in range(1, 6):
    batch_file = os.path.join(data_dir, f"data_batch_{i}")
    batch_data = unpickle(batch_file)
    train_data.append(batch_data[b'data'])
    train_labels += batch_data[b'labels']
train_images = np.concatenate(train_data, axis=0)  # 合併為一個單一的NumPy數組
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
train_labels = np.array(train_labels).reshape(-1, 1)

# Load test data
test_file = os.path.join(data_dir, "test_batch")
test_data = unpickle(test_file)
test_images = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
test_labels = np.array(test_data[b'labels']).reshape(-1, 1)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Convert to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)


# Create TensorDataset
train_data_torch = torch.utils.data.TensorDataset(train_images, train_labels)
test_data_torch = torch.utils.data.TensorDataset(test_images, test_labels)
# 創建 DataLoader
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_data_torch, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data_torch, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomViT(torch.nn.Module):
    def __init__(self, num_classes, input_shape, patch_size=4, d_model=64):
        super(CustomViT, self).__init__()
        self.input_shape = input_shape  # 32*32*3
        self.num_classes = num_classes
        self.patch_size = patch_size
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        # self.patch_embedding = torch.nn.Conv2d(3, num_patches, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding = torch.nn.Linear(patch_size * patch_size * 3, d_model)
        self.position_embedding = torch.nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer Encoder Layer
        # batch_first=True 代表輸入的形狀應該是 (batch_size, seq_len, embedding_dim)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.output_layer = torch.nn.Linear(64, num_classes)


    def forward(self, inputs):
        batch_size, height, width, channels = inputs.shape
        # 將圖像拆分為小塊（patches）
        x = inputs.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)
        # (batch_size, num_patches, patch_size*patch_size*channels)

        x = self.patch_embedding(x)  # (batch_size, num_patches, d_model)
        x = x + self.position_embedding  # (batch_size, num_patches, d_model)
        x = self.transformer_encoder(x)

        # 取最後一個時間步的輸出進行分類
        x = x.mean(dim=1)  # (batch_size, d_model)
        outputs = self.output_layer(x)
        return outputs


model = CustomViT(num_classes=10, input_shape=(32, 32, 3)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter('runs/experiment_1')

num_epochs = 300
training_accuracies = []
training_losses = []

testing_accuracies = []
testing_losses = []
for epoch in range(num_epochs):
    model.train()  # 設置為訓練模式
    running_loss = 0.0
    training_correct = 0
    training_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 將梯度歸零
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)  # Flatten the output for linear layer
        loss = criterion(outputs, labels.squeeze())

        # 反向傳播和優化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 計算訓練準確率
        _, predicted = torch.max(outputs.data, 1)
        training_total += labels.size(0)
        training_correct += (predicted == labels.squeeze()).sum().item()

    training_accuracy = training_correct / training_total
    training_loss = running_loss / len(train_loader)

    training_accuracies.append(training_accuracy)
    training_losses.append(training_loss)

    writer.add_scalar('Training Accuracy', training_accuracy, epoch)
    writer.add_scalar('Training Loss', training_loss, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.4f}")

    # 驗證
    model.eval()  # 設置為評估模式
    testing_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.squeeze())
            testing_loss += loss.item()
            test_total += labels.size(0)
            test_correct += (predicted == labels.squeeze()).sum().item()

    testing_accuracy = test_correct / test_total
    testing_loss = testing_loss / len(test_loader)

    testing_accuracies.append(testing_accuracy)
    testing_losses.append(testing_loss)
    writer.add_scalar('Testing Accuracy', testing_accuracy, epoch)
    writer.add_scalar('Testing Loss', testing_loss, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {testing_loss:.4f}, Test Accuracy: {testing_accuracy:.4f}")

writer.close()


# 保存整個模型
torch.save(model, "model.pth")
print("Model saved")

# # 將 losses 儲存到檔案中
# with open('test_acc.pkl', 'wb') as f:
#     pickle.dump(test_acc, f)

# # 從檔案中載入 losses
# with open("losses.pkl", "rb") as f:
#     losses = pickle.load(f)





