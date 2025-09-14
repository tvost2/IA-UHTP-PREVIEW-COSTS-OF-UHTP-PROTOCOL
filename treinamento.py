import torch
import torch.nn as nn
import torch.optim as optim
import random

CAMADAS = 15
QUANTIDADE_IPS = 200

class NetMindNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def gerar_ips_fake():
    ips, x, y = [], 0, 1
    for _ in range(QUANTIDADE_IPS):
        if y > 254:
            x += 1
            y = 1
        ips.append(f"10.66.{x}.{y}")
        y += 1
    return ips

def gerar_circuito(ips):
    return random.sample(ips, CAMADAS)

def codificar_circuito(circuito, todos_ips):
    return [todos_ips.index(ip) / len(todos_ips) for ip in circuito]

def simular_custo(circuito):
    rtt = [random.uniform(20, 100) for _ in circuito]
    perdas = sum([1 for _ in circuito if random.random() < 0.1])
    overhead = random.uniform(0, 1)
    return sum(rtt)/len(circuito) + perdas * 50 + overhead * 20

def gerar_dataset(n):
    X, y = [], []
    ips = gerar_ips_fake()
    for _ in range(n):
        c = gerar_circuito(ips)
        X.append(codificar_circuito(c, ips))
        y.append([simular_custo(c)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def treinar_modelo():
    X, y = gerar_dataset(1000)
    model = NetMindNN(CAMADAS)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "netmind_model.pt")
    print("Modelo salvo como netmind_model.pt")

if __name__ == "__main__":
    treinar_modelo()
