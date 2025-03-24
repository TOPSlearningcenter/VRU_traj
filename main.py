# TODO.1 导入必要的库
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# TODO.2 定义参数
# 数据相关的参数
T_in = 5  # 输入的时间帧
n_veh = 5  # 一帧最大的车辆数目
feat = 8  # 预测采用的特征数目，一共有10个特征，此项目只用8个特征
T_out = 6  # 要预测的时间帧
num_modes = 3  # 预测的车辆模态

seed = 42  # 随机种子
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备

continue_train = False  # 是否继续训练

# TODO.3 导入数据集、预处理、制作数据加载器
# 数据加载
data_x = np.load("train_data_x.npy")
data_y = np.load("train_data_y.npy")
# 不使用急动度jerk_x与jerk_y这两个特征，新 shape: (样本数, T_in=5, n_veh=5, feat=8)
data_x = np.delete(data_x, [6, 7], axis=-1)
# 按 8:2 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)


class TrajectoryDataset(Dataset):
    def __init__(self, x, y, dev=torch.device('cpu')):
        self.data_x = torch.tensor(x, dtype=torch.float32).to(dev)
        self.data_y = torch.tensor(y, dtype=torch.float32).to(dev)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

batch_size = 64  # 批次大小
train_dataset = TrajectoryDataset(train_x, train_y, device)
test_dataset = TrajectoryDataset(test_x, test_y, device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练加载器数据打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试加载器数据不打乱


# TODO.4 定义模型

def build_fully_connected_graph(num_nodes, b_s, mask):
    """
    构建批量全连接图，仅连接每个样本中有效车辆之间的边
    """
    edges = []
    for b in range(b_s):
        valid_nodes = torch.where(mask[b])[0]
        for i in valid_nodes:
            for j in valid_nodes:
                if i != j:
                    edges.append([i + b * num_nodes, j + b * num_nodes])
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=mask.device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=mask.device).t().contiguous()
    return edge_index


class Model(nn.Module):
    def __init__(self, feature_size=10, hidden_dim=64, num_modes=5, T_in=5, T_out=6, n_veh=5, dropout_rate=0.2):
        """
        feature_size: 输入每辆车的特征数 —— (x, y, v_x, v_y, a_x, a_y, theta, type)
        hidden_dim: 隐藏层维度（例如 32）
        num_modes: 多模态轨迹数（例如 5）
        T_in: 历史时间步数（例如 5）
        T_out: 未来预测时间步数（例如 6）
        n_veh: 每帧最大车辆数（例如 5）
        dropout_rate: dropout 比例
        """
        super(Model, self).__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.n_veh = n_veh
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Encoder 部分
        self.embed = nn.Linear(feature_size, hidden_dim)
        # GNN 交互层，仅在某一时刻使用
        self.gnn = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate)
        # GRU 编码器（对每辆车时序编码）
        self.gru_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Decoder 部分：使用 GRUCell 逐步解码
        self.gru_decoder = nn.GRUCell(hidden_dim, hidden_dim)
        # MLP 将隐藏状态映射为预测输出：每个时间步输出 2*num_modes 数值
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * num_modes)
        )
        # 投影层：将 (x,y) 转换回 hidden_dim 用作下一时间步的输入
        self.proj_input = nn.Linear(2, hidden_dim)
        # 置信度分支：预测每辆车每个模态的置信度
        self.conf_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )

    def forward(self, x):
        """
        x: 输入张量，形状 (B, T_in, n_veh, feature_size)
        返回：
            traj: 预测轨迹，形状 (B, num_modes, n_veh, T_out, 2)
            conf: 置信度，形状 (B, num_modes, n_veh)
        """
        B, T, N, _ = x.shape
        # 有效性掩码：根据最后一帧判断，若该车坐标x, y, v_x, v_y, a_x, a_y之和为0，则为无效
        mask = (x[:, -1, :, 0:6].sum(dim=-1) != 0)  # (B, N)
        # 特征编码
        x_emb = self.embed(x)  # (B, T_in, N, hidden_dim)
        # GRU 编码器：对每辆车的历史轨迹进行编码
        x_enc = x_emb.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)  # (B*N, T_in, hidden_dim)
        _, h_enc = self.gru_encoder(x_enc)  # h_enc: (1, B*N, hidden_dim)
        h_enc = h_enc.squeeze(0).view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # GNN 交互：仅对最后时刻的特征使用 GNN，构建全连接图（仅对有效车辆）
        x_gnn = x_emb[:, -1, :, :]  # (B, N, hidden_dim)
        edge_index = build_fully_connected_graph(N, B, mask)  # (2, num_edges)
        x_gnn_flat = x_gnn.reshape(B * N, self.hidden_dim)
        x_gnn_out = self.gnn(x_gnn_flat, edge_index)  # (B*N, hidden_dim)
        x_gnn_out = x_gnn_out.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # 融合 GRU 编码和 GNN 交互：作为每辆车的最终编码状态
        h_final = h_enc + x_gnn_out  # (B, N, hidden_dim)

        # Decoder：顺序解码 T_out 时间步
        h_dec = h_final.reshape(B * N, self.hidden_dim)  # 初始化隐藏状态：h_final 重塑为 (B*N, hidden_dim)
        decoder_input = self.proj_input(x[:, -1, :, 0:2])  # (B, N, 2) -> (B, N, hidden_dim)
        decoder_input = decoder_input.reshape(B * N, self.hidden_dim)
        outputs = []
        for t in range(self.T_out):
            h_dec = self.gru_decoder(decoder_input, h_dec)  # 更新隐藏状态，形状 (B*N, hidden_dim)
            # 从隐藏状态预测当前时间步输出： (B*N, 2*num_modes)
            out_t = self.output_mlp(h_dec)
            out_t = out_t.reshape(B, N, 2, self.num_modes)  # (B, N, 2, num_modes)
            outputs.append(out_t)
            # 作为下一步输入：取每辆车的第一模态 (B, N, 2)
            next_input = out_t[:, :, :, 0]
            # 投影到 hidden_dim
            decoder_input = self.proj_input(next_input)  # (B, N, hidden_dim)
            decoder_input = decoder_input.reshape(B * N, self.hidden_dim)

        # 堆叠所有时间步：得到 (B, T_out, N, 2, num_modes)
        traj = torch.stack(outputs, dim=1)
        output_traj = traj.permute(0, 4, 2, 1, 3)  # 转化成【场景，模态，个体，时间，特征】

        # 置信度分支：从编码器 h_final 预测每辆车每个模态的置信度
        conf = self.conf_mlp(h_final)  # (B, N, num_modes)
        conf = F.softmax(conf, dim=-1)  # (B, N, num_modes)
        output_conf = conf.permute(0, 2, 1)  # 转化成【场景，模态，个体】

        return output_traj, output_conf

H_dim = 32
D_rate = 0.2
model = Model(feature_size=feat, hidden_dim=H_dim, num_modes=num_modes, T_in=T_in, T_out=T_out, n_veh=n_veh,
              dropout_rate=D_rate)
model.to(device)
if continue_train:
     model.load_state_dict(torch.load("best_model.pth", map_location=device))


# TODO.5 模型训练
num_epochs = 30  # 训练的回合数
best_val_loss = float('inf')
fre_val = 5  # 多少个回合测试一下，并保存好的权重
learning_rate = 0.0005  # 学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_direct_loss(preds, targets, conf):
    # 将targets扩展到与preds相同的维度
    targets_expanded = targets.permute(0, 2, 1, 3).unsqueeze(1).expand_as(preds)
    diff = preds - targets_expanded
    error_per_mode = torch.mean(torch.norm(diff, dim=-1), dim=-1)
    min_error = torch.sum(conf * error_per_mode, dim=1)  # (B, n_veh, T_out)
    loss = torch.mean(min_error)  # 批次内平均 ADE
    return loss


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch_x, batch_y in train_loader_tqdm:
        optimizer.zero_grad()
        out, conf = model(batch_x)
        gt_xy = batch_y[:, :, :, 0:2]
        loss = compute_direct_loss(out, gt_xy, conf)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        train_loader_tqdm.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

    if (epoch+1) % fre_val == 0:
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out, conf = model(batch_x)
                gt_xy = batch_y[:, :, :, 0:2]
                loss = compute_direct_loss(out, gt_xy, conf)
                total_val_loss += loss.item() * batch_x.size(0)
        avg_val_loss = total_val_loss / len(test_dataset)
        print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}")

        # 记录最佳模型与最新模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
        torch.save(model.state_dict(), "last_model.pth")

# TODO.6 结果可视化
import matplotlib
from matplotlib import pyplot as plt
# 设置全局字体为支持中文的字体，SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


vid = 10  # 准备可视化的id
model = Model(feature_size=feat, hidden_dim=H_dim, num_modes=num_modes, T_in=T_in, T_out=T_out, n_veh=n_veh,
              dropout_rate=D_rate)
model.to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
with torch.no_grad():
    test_vid_x = torch.tensor(test_x[vid], dtype=torch.float32).to(device)
    pred_vid_y, conf = model(test_vid_x.unsqueeze(0))

# 结果处理
test_vid_x = test_x[vid]
test_vid_y = test_y[vid]
pred_vid_y = pred_vid_y.squeeze(0).cpu().detach().numpy()
conf = conf.squeeze(0).cpu().detach().numpy()
mean_conf = conf.mean(axis=1)
for i in range(num_modes):
    print(f"第{i+1}个模态所有车辆的平均置信度为{np.mean(conf[i], axis=0)}")

# 开始画图
fig = plt.figure(figsize=(18, 6))
for i in range(num_modes):
    ax = fig.add_subplot(1, num_modes, i+1)
    for n in range(n_veh):
        ax.plot(test_vid_x[:, n, 0], test_vid_x[:, n, 1], 'go', label='历史轨迹')
        ax.plot(test_vid_y[:,n,0], test_vid_y[:,n,1], 'ko', label='未来真值')
        ax.plot(pred_vid_y[i, :, n, 0], pred_vid_y[i, :, n, 1], 'r*', label='预测轨迹')
    ax.set_title(f'第{i+1}个模态的预测结果')
plt.tight_layout()
plt.show()