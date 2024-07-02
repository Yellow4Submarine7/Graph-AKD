import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

class GraphAKD:
    def __init__(self, teacher, student, alpha1, alpha2, temperature):
        self.teacher = teacher
        self.student = student
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.temperature = temperature

    def kl_loss(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.temperature, dim=1)
        p_t = F.softmax(y_t / self.temperature, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature ** 2)

    def lsp_loss(self, h_s, h_t, edge_index):
        def structural_distance(h, edge_index):
            row, col = edge_index
            diff = h[row] - h[col]
            dist = torch.norm(diff, p=2, dim=1)
            exp_dist = torch.exp(dist)
            return exp_dist / exp_dist.sum()

        l_s = structural_distance(h_s, edge_index)
        l_t = structural_distance(h_t, edge_index)
        return F.kl_div(torch.log(l_s), l_t, reduction='batchmean')

    def train_step(self, x, edge_index, y, quiz_mask):
        # Temporary update
        self.student.train()
        self.teacher.eval()
        
        with torch.no_grad():
            y_t = self.teacher(x, edge_index)
        
        y_s = self.student(x, edge_index)
        
        loss_ce = F.cross_entropy(y_s[quiz_mask], y[quiz_mask])
        loss_kl = self.kl_loss(y_s[quiz_mask], y_t[quiz_mask])
        loss_lsp = self.lsp_loss(y_s, y_t, edge_index)
        
        loss = loss_ce + self.alpha1 * loss_kl + self.alpha2 * loss_lsp
        
        # Update student
        self.student.optimizer.zero_grad()
        loss.backward()
        self.student.optimizer.step()
        
        # Feedback update
        self.teacher.train()
        y_t = self.teacher(x, edge_index)
        y_s = self.student(x, edge_index).detach()
        
        loss_t = self.kl_loss(y_s[quiz_mask], y_t[quiz_mask])
        
        self.teacher.optimizer.zero_grad()
        loss_t.backward()
        self.teacher.optimizer.step()

        return loss.item()

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# Set up models
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
teacher_layers = 3
student_layers = 2

teacher = GNN(input_dim, hidden_dim, output_dim, teacher_layers)
student = GNN(input_dim, hidden_dim // 2, output_dim, student_layers)

# Set up optimizers
teacher.optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
student.optimizer = torch.optim.Adam(student.parameters(), lr=0.01)

# Initialize GraphAKD
graphakd = GraphAKD(teacher, student, alpha1=0.5, alpha2=0.1, temperature=2.0)

# Training loop
num_epochs = 200
quiz_ratio = 0.1
num_nodes = data.x.size(0)
quiz_size = int(num_nodes * quiz_ratio)

for epoch in range(num_epochs):
    # Randomly select quiz nodes
    perm = torch.randperm(num_nodes)
    quiz_mask = torch.zeros(num_nodes, dtype=torch.bool)
    quiz_mask[perm[:quiz_size]] = True

    # Train step
    loss = graphakd.train_step(data.x, data.edge_index, data.y, quiz_mask)
    
    # Evaluation
    if (epoch + 1) % 10 == 0:
        student.eval()
        with torch.no_grad():
            pred = student(data.x, data.edge_index).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

print("Training completed!")

# Final evaluation
student.eval()
with torch.no_grad():
    pred = student(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
print(f'Final Test Accuracy: {acc:.4f}')
