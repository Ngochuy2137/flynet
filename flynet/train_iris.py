import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dữ liệu
iris = load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target  # shape: (150,)
print(f"X shape: {X.shape}, y shape: {y.shape}"); input()
y_unique = set(y)
print(f"y unique: {y_unique}"); input()

# Tiền xử lý dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển về Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Định nghĩa model MLP đơn giản
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)

model = IrisNet()

# Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    outputs = model(X_train)
    print('x_train.shape: ', X_train.shape); # torch.Size([120, 4])
    print('outputs.shape: ', outputs.shape); # torch.Size([120, 3])
    print('y_train.shape: ', y_train.shape); # torch.Size([120])
    input('press enter to continue...')
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Đánh giá mô hình
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    acc = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f"\n🎯 Accuracy on test set: {acc:.4f}")

# Vẽ biểu đồ loss
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
