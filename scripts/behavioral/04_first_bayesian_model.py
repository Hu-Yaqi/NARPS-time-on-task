import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ============================================================
# 准备数据：只用1号被试
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')

# 筛选1号被试的数据
sub001 = df[df['subject'] == 'sub-001']

# 统计：256个trial中接受了多少次
n_trials = len(sub001)
n_accept = sub001['accepted'].sum()

print(f"被试 sub-001:")
print(f"  总trial数：{n_trials}")
print(f"  接受次数：{n_accept}")
print(f"  接受率（直接计算）：{n_accept / n_trials:.1%}")

# ============================================================
# 构建贝叶斯模型
# ============================================================

# pm.Model() 创建一个"模型容器"
# with 语句的意思是：接下来的所有模型定义都属于这个容器
with pm.Model() as simple_model:

    # 第一步：定义 Prior（先验）
    # Beta(1, 1) 就是"0到1之间均匀分布"
    # 意思是：在看数据之前，我认为接受率可以是0%到100%之间的任何值
    # Beta分布是专门用来描述"概率"这种0到1之间的量的
    theta = pm.Beta('theta', alpha=1, beta=1)

    # 第二步：定义 Likelihood（似然）
    # Binomial 是"二项分布"——描述"n次试验中成功k次"的概率
    # n=总trial数，p=接受率（就是我们要估的theta），observed=实际观察到的接受次数
    # observed 这个参数告诉PyMC："这是真实数据，不是要估计的参数"
    y = pm.Binomial('y', n=n_trials, p=theta, observed=n_accept)

    # 第三步：运行MCMC采样
    # draws=2000 表示采样2000次（爬山走2000步）
    # chains=4 表示同时派4个"探险队"从不同起点出发
    #   多个chain的好处是可以检查它们是否都走到了同一个区域
    #   如果是，说明结果可靠；如果不是，说明模型可能有问题
    # random_seed=42 是随机种子，保证每次运行结果一样（方便复现）
    trace = pm.sample(draws=2000, chains=4, random_seed=42)
    # "trace"是采样的记录——4个探险队各走2000步，共8000个θ值
    # 这8000个值的分布就近似于posterior分布

# ============================================================
# 查看结果
# ============================================================

# az.summary 给出posterior的数字摘要
print("\n=== Posterior 摘要 ===")
print(az.summary(trace, var_names=['theta']))

# az.plot_posterior 画出posterior分布图
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_posterior(trace, var_names=['theta'], ax=ax)
ax.set_title('Posterior distribution of accept rate (sub-001)')
plt.tight_layout()
plt.savefig('posterior_theta_sub001.png', dpi=150)
print("\nPosterior分布图已保存为 posterior_theta_sub001.png")