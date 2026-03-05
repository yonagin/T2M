import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ChoiceGAN 核心组件
# ==========================================

class CodeDiscriminator(nn.Module):
    def __init__(self, num_embeddings: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p).squeeze(-1)

def get_soft_p(h: torch.Tensor, codebook: torch.Tensor, temperature: float) -> torch.Tensor:
    # 核心算法思想：h被detach，说明 GAN 的对抗梯度只优化 Codebook 向量
    h_norm = (h.detach() ** 2).sum(dim=1, keepdim=True)        
    c_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)   
    logits = -(h_norm + c_norm - 2 * h.detach() @ codebook.t()) / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)

# ==========================================
# 2. Toy 实验构建与 SGD 优化循环
# ==========================================

def run_experiment(method="vanilla", steps=80, lr_e=0.08, lr_q=0.08, beta=0.25):
    # 固定目标位置与初始起点
    T = torch.tensor([[1.0, 1.0]]) 
    z_e_init = torch.tensor([[-1.0, -1.0]])
    
    # 构建包含3个码字的Codebook：我们追踪第一个(C[0])，其余两个作为 ChoiceGAN 的负样本环境
    C_init = torch.tensor([
        [1.0, -1.0],  # Active z_q (被匹配追踪)
        [0.0, 1.5],   # Distractor 1
        [-1.5, 0.5],   # Distractor 2
        [-0.5,1.0]
    ])
    
    z_e = nn.Parameter(z_e_init.clone())
    C = nn.Parameter(C_init.clone())
    
    # 优化器设计：为ChoiceGAN中的 z_q 引入动量，以模拟对抗学习中常见的螺旋/振荡动力学
    opt_main = optim.SGD([z_e], lr=lr_e)
    opt_cb = optim.SGD([C], lr=lr_q, momentum=0.6 if method == "choicegan" else 0.0)
    
    if method == "choicegan":
        D = CodeDiscriminator(num_embeddings=C_init.size(0))
        opt_d = optim.Adam(D.parameters(), lr=0.01)
        alpha=0.1
        tau = 1.0
        adv_weight = 6e-5

    traj_e, traj_q = [], []

    for step in range(steps):
        # 记录当前轨迹
        traj_e.append(z_e.detach().numpy()[0].copy())
        traj_q.append(C.detach().numpy()[0].copy())
        
        # 在Toy设定下，默认 z_e 一直与 C[0] 匹配以展示连贯轨迹
        z_q = C[0].unsqueeze(0)
        
        # 1. 任务损失 Task Loss (采用 Straight-Through Estimator)
        z_q_ste = z_e + (z_q - z_e).detach()
        loss_task = 0.5 * torch.sum((z_q_ste - T) ** 2)
        
        # 2. Vanilla VQ 的基础损失
        loss_commit = 0.5 * torch.sum((z_e - z_q.detach()) ** 2)
        loss_cb = 0.5 * torch.sum((z_e.detach() - z_q) ** 2)
        
        loss_g = loss_task + loss_cb + beta * loss_commit

        # 3. ChoiceGAN 对抗损失计算
        if method == "choicegan":
            p_fake = get_soft_p(z_e, C, tau)
            alpha_tensor = torch.full((C_init.size(0),), alpha)
            p_real = torch.distributions.Dirichlet(alpha_tensor).sample((1,))
            
            # --- 更新判别器 D ---
            d_real = D(p_real)
            d_fake_d = D(p_fake.detach())
            loss_d = 0.5 * (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + 
                            F.binary_cross_entropy_with_logits(d_fake_d, torch.zeros_like(d_fake_d)))
            
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            
            # --- 更新生成器 G (即带有对抗梯度的码本) ---
            p_fake_g = get_soft_p(z_e, C, tau)
            d_fake_g = D(p_fake_g)
            loss_adv = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))
            
            # 融合对抗 Loss 
            loss_g += adv_weight * loss_adv

        # 梯度回传与步进
        opt_main.zero_grad()
        opt_cb.zero_grad()
        loss_g.backward()
        
        opt_main.step()
        opt_cb.step()

    return traj_e, traj_q, T.numpy()[0]

# ==========================================
# 3. 结果可视化 (ICLR Paper 风格)
# ==========================================

# 运行两组实验
traj_e_v, traj_q_v, target = run_experiment("vanilla")
traj_e_c, traj_q_c, _ = run_experiment("choicegan")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# ========== 关键修改1：统一坐标范围 ==========
# 收集所有轨迹数据，计算统一的 x/y 范围
all_x, all_y = [], []
for t_e, t_q in zip([traj_e_v, traj_e_c], [traj_q_v, traj_q_c]):
    ex, ey = zip(*t_e)
    qx, qy = zip(*t_q)
    all_x.extend(ex); all_x.extend(qx)
    all_y.extend(ey); all_y.extend(qy)
all_x.append(target[0]); all_y.append(target[1])

margin = 0.5  # 留白
xlim = (min(all_x) - margin, max(all_x) + margin)
ylim = (min(all_y) - margin, max(all_y) + margin)

for ax, t_e, t_q, title in zip(axes, [traj_e_v, traj_e_c], [traj_q_v, traj_q_c], ["Vanilla VQ", "C VQ"]):
    t_e_x, t_e_y = zip(*t_e)
    t_q_x, t_q_y = zip(*t_q)

    ax.plot(t_e_x, t_e_y, color='#6A0572', linestyle='-', marker='.', linewidth=2, markersize=7, label=r'$z_e$')   # 深莓紫
    ax.plot(t_q_x, t_q_y, color='#3DA35D', linestyle='-', marker='.', linewidth=2, markersize=7, label=r'$z_q$')  # 森林绿

    ax.scatter(target[0], target[1], color='red', marker='$\u2665$', s=100, zorder=5, label='Target')
    ax.scatter(t_e_x[0], t_e_y[0], color='black', marker='o', s=80, zorder=5, label='Start')
    ax.scatter(t_q_x[0], t_q_y[0], color='black', marker='o', s=80, zorder=5)

    ax.set_title(title, fontsize=16, pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)

    # ========== 关键修改2：统一坐标范围 ==========
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')  # 加 adjustable='box' 保持子图框架大小一致

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title == "C VQ":
        # ========== 关键修改3：缩小图例 ==========
        ax.legend(
            loc='lower right',
            fontsize=8,                # 字体缩小
            framealpha=0.9,
            markerscale=0.6,           # marker 缩小
            handlelength=2.0,          # 图例中线段长度缩短
            handletextpad=0.4,         # 线段与文字间距缩小
            borderpad=0.3,             # 图例内边距缩小
            labelspacing=0.3,          # 条目间距缩小
        )

plt.tight_layout()
plt.savefig('vq_trajectories.pdf', bbox_inches='tight')
plt.show()