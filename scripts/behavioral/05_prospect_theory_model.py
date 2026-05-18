import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 准备数据：1号被试
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
sub001 = df[df['subject'] == 'sub-001'].copy()

# 提取我们需要的三列，转成numpy数组
# numpy数组是Python里的"数字列表"，PyMC需要这种格式
# .values 把pandas的列转成numpy数组
gain = sub001['gain'].values       # 每个trial的潜在收益
loss = sub001['loss'].values       # 每个trial的潜在损失
choice = sub001['accepted'].values # 每个trial的选择（1=接受, 0=拒绝）

n_trials = len(choice)
print(f"被试 sub-001: {n_trials} 个trial")
print(f"接受次数: {choice.sum()}, 接受率: {choice.mean():.1%}")

# ============================================================
# 构建前景理论贝叶斯模型
# ============================================================

with pm.Model() as pt_model:

    # ----------------------------------------------------------
    # 第一步：定义 Priors（先验）
    # ----------------------------------------------------------

    # λ (lambda): 损失厌恶系数
    # LogNormal(0.5, 0.5) 的意思是：
    #   取对数后服从正态分布 Normal(0.5, 0.5)
    #   所以λ本身一定是正数（对数正态分布只产生正数）
    #   中心大约在 exp(0.5) ≈ 1.6 附近
    #   大部分值落在 0.8 到 3.5 之间
    lam = pm.LogNormal('lambda', mu=0.5, sigma=0.5)

    # α (alpha): 价值函数曲率
    # Beta(2, 2) 是一个在0到1之间的钟形分布
    #   中心在0.5，大部分值在0.2到0.8之间
    #   这符合文献中α通常小于1的发现
    alpha = pm.Beta('alpha', alpha=2, beta=2)

    # β (beta): 选择一致性 / 逆温度
    # HalfNormal(sigma=1) 的意思是：
    #   只取正态分布的正数部分
    #   大部分值在0到2之间，峰值在0
    beta = pm.HalfNormal('beta', sigma=1)

    # ----------------------------------------------------------
    # 第二步：计算每个trial的主观价值（Subjective Value）
    # ----------------------------------------------------------

    # 前景理论公式：SV = gain^α - λ * loss^α
    #
    # gain 和 loss 是我们前面准备好的数组（每个trial一个值）
    # alpha, lambda, beta 是要估计的参数（标量）
    #
    # PyMC会自动把标量参数"广播"到每个trial上
    # 就像Excel里一个公式拖下去应用到每一行
    sv = gain ** alpha - lam * (loss ** alpha)

    # ----------------------------------------------------------
    # 第三步：把主观价值转化为接受概率
    # ----------------------------------------------------------

    # logistic函数：P(accept) = 1 / (1 + exp(-β * SV))
    #
    # pm.math.sigmoid 就是logistic函数的PyMC版本
    # 它和 1/(1+exp(-x)) 完全等价，只是写法更简洁
    p_accept = pm.math.sigmoid(beta * sv)

    # 为了避免数值问题（概率太接近0或1会让计算出错）
    # 把概率限制在 [0.01, 0.99] 之间
    # pm.math.clip 就像Excel里的 MAX(0.01, MIN(0.99, x))
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)

    # ----------------------------------------------------------
    # 第四步：定义 Likelihood（似然）
    # ----------------------------------------------------------

    # Bernoulli分布描述的是"一次试验，成功或失败"
    # 每个trial就是一次独立的"赌不赌"的决定
    # p = 接受概率（由上面的前景理论公式计算得到）
    # observed = 实际选择（1或0）
    #
    # 这行代码告诉PyMC：
    # "请找到一组(λ, α, β)的值，使得模型预测的接受概率
    #  能最好地解释被试实际做出的这些选择"
    y = pm.Bernoulli('y', p=p_accept, observed=choice)

    # ----------------------------------------------------------
    # 第五步：运行MCMC采样
    # ----------------------------------------------------------

    print("\n开始MCMC采样，请稍等（大约1-3分钟）...")

    # target_accept=0.9 让采样器更"谨慎"，提高采样质量
    # 对于这种非线性模型，稍高的target_accept有助于避免采样问题
    trace = pm.sample(
        draws=2000,
        chains=4,
        random_seed=42,
        target_accept=0.9
    )

# ============================================================
# 查看结果
# ============================================================

# 打印参数估计的摘要表
print("\n=== Posterior 摘要 ===")
summary = az.summary(trace, var_names=['lambda', 'alpha', 'beta'])
print(summary)

# 画出三个参数的posterior分布
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# 1, 3 表示一行三列（三张子图并排）

az.plot_posterior(trace, var_names=['lambda'], ax=axes[0])
axes[0].set_title('λ (loss aversion)')

az.plot_posterior(trace, var_names=['alpha'], ax=axes[1])
axes[1].set_title('α (curvature)')

az.plot_posterior(trace, var_names=['beta'], ax=axes[2])
axes[2].set_title('β (consistency)')

plt.tight_layout()
plt.savefig('prospect_theory_sub001.png', dpi=150)
print("\n三个参数的posterior分布图已保存为 prospect_theory_sub001.png")

# ============================================================
# 用posterior的均值来画预测的"决策地图"
# ============================================================

# 提取每个参数的posterior均值
lam_est = float(trace.posterior['lambda'].mean())
alpha_est = float(trace.posterior['alpha'].mean())
beta_est = float(trace.posterior['beta'].mean())

print(f"\n参数估计（posterior均值）：")
print(f"  λ = {lam_est:.2f} （损失厌恶）")
print(f"  α = {alpha_est:.2f} （曲率）")
print(f"  β = {beta_est:.2f} （选择一致性）")

# 用这组参数，对每种gain/loss组合预测接受概率
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：实际数据
ax = axes[0]
acc = sub001[sub001['accepted'] == 1]
rej = sub001[sub001['accepted'] == 0]
ax.scatter(acc['gain'], acc['loss'], color='#1D9E75', alpha=0.6, s=40, label='Accepted')
ax.scatter(rej['gain'], rej['loss'], color='#D85A30', alpha=0.6, s=40, label='Rejected')
ax.set_xlabel('Potential gain')
ax.set_ylabel('Potential loss')
ax.set_title('Actual choices (sub-001)')
ax.legend()

# 右图：模型预测
ax = axes[1]
# np.linspace(10, 40, 30) 生成从10到40之间均匀分布的30个点
gain_grid = np.linspace(10, 40, 30)
loss_grid = np.linspace(5, 20, 30)
# np.meshgrid 把两个一维数组变成二维网格
# 就像在Excel里横着写gain值、竖着写loss值，生成一个完整的表格
G, L = np.meshgrid(gain_grid, loss_grid)

# 对网格上的每个点计算接受概率
SV = G ** alpha_est - lam_est * (L ** alpha_est)
P = 1 / (1 + np.exp(-beta_est * SV))

# contourf 画等高线填充图（就是热力图的一种）
# cmap='RdYlGn' 是颜色方案：红-黄-绿
c = ax.contourf(G, L, P, levels=20, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(c, ax=ax, label='P(accept)')
ax.set_xlabel('Potential gain')
ax.set_ylabel('Potential loss')
ax.set_title('Model prediction (sub-001)')

plt.tight_layout()
plt.savefig('model_vs_data_sub001.png', dpi=150)
print("模型预测 vs 实际数据的对比图已保存为 model_vs_data_sub001.png")