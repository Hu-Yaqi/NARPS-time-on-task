import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# 准备数据
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())
sub_to_idx = {s: i for i, s in enumerate(subjects)}
df['sub_idx'] = df['subject'].map(sub_to_idx)
n_subjects = len(subjects)

gain = df['gain'].values.astype('float64')
loss = df['loss'].values.astype('float64')
choice = df['accepted'].values.astype('float64')
sub_idx = df['sub_idx'].values
log_gain = np.log(gain)
log_loss = np.log(loss)

print(f"被试: {n_subjects}, Trial数: {len(choice)}")

# ============================================================
# 模型
# ============================================================

with pm.Model() as model:

    mu_lam = pm.Normal('mu_lam', mu=0.2, sigma=0.3)
    sigma_lam = pm.Exponential('sigma_lam', lam=5)
    mu_alpha = pm.Normal('mu_alpha', mu=0.5, sigma=0.5)
    sigma_alpha = pm.Exponential('sigma_alpha', lam=5)
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.Exponential('sigma_beta', lam=5)

    lam_offset = pm.Normal('lam_offset', mu=0, sigma=1, shape=n_subjects)
    lam = pm.math.exp(mu_lam + sigma_lam * lam_offset)

    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects)
    alpha = pm.math.sigmoid(mu_alpha + sigma_alpha * alpha_offset)

    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
    beta = pm.math.exp(mu_beta + sigma_beta * beta_offset)

    alpha_i = alpha[sub_idx]
    sv = pm.math.exp(alpha_i * log_gain) - lam[sub_idx] * pm.math.exp(alpha_i * log_loss)
    p_accept = pm.math.sigmoid(beta[sub_idx] * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)

    y = pm.Bernoulli('y', p=p_accept, observed=choice)

    print("\n开始过夜采样...")
    print("预计需要2-3小时，可以放着去睡觉")
    print(f"开始时间: {time.strftime('%H:%M:%S')}\n")
    start = time.time()

    trace = pm.sample(
        draws=2000,
        tune=3000,       # 充分热身
        chains=4,
        random_seed=42,
        target_accept=0.95,
        cores=1,          # 单核，一条chain一条chain跑，最稳定
        max_treedepth=12,
    )

    elapsed = time.time() - start
    print(f"\n采样完成！耗时 {elapsed/60:.1f} 分钟")
    print(f"结束时间: {time.strftime('%H:%M:%S')}")

# 保存
trace.to_netcdf('hierarchical_trace_final.nc')

# ============================================================
# 收敛检查
# ============================================================

print("\n=== 群体层面参数 ===")
hyper_vars = ['mu_lam', 'sigma_lam', 'mu_alpha', 'sigma_alpha',
              'mu_beta', 'sigma_beta']
hyper_summary = az.summary(trace, var_names=hyper_vars)
print(hyper_summary)

rhat_ok = (hyper_summary['r_hat'] <= 1.01).all()
ess_ok = (hyper_summary['ess_bulk'] >= 400).all()
print(f"\n收敛检查：")
print(f"  所有r_hat <= 1.01? {'通过' if rhat_ok else '未通过'}")
print(f"  所有ess_bulk >= 400? {'通过' if ess_ok else '未通过'}")

# ============================================================
# 提取个体参数
# ============================================================

post = trace.posterior
mu_l = post['mu_lam'].values
sig_l = post['sigma_lam'].values
off_l = post['lam_offset'].values
lam_est = np.exp(mu_l[:,:,np.newaxis] + sig_l[:,:,np.newaxis] * off_l).mean(axis=(0,1))

mu_a = post['mu_alpha'].values
sig_a = post['sigma_alpha'].values
off_a = post['alpha_offset'].values
alpha_est = (1/(1+np.exp(-(mu_a[:,:,np.newaxis] + sig_a[:,:,np.newaxis] * off_a)))).mean(axis=(0,1))

mu_b = post['mu_beta'].values
sig_b = post['sigma_beta'].values
off_b = post['beta_offset'].values
beta_est = np.exp(mu_b[:,:,np.newaxis] + sig_b[:,:,np.newaxis] * off_b).mean(axis=(0,1))

params_df = pd.DataFrame({
    'subject': subjects,
    'lambda': lam_est,
    'alpha': alpha_est,
    'beta': beta_est
})
params_df.to_csv('individual_parameters_final.csv', index=False)

print(f"\n=== 108个被试的参数统计 ===")
print(params_df[['lambda', 'alpha', 'beta']].describe().round(3))

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(params_df['lambda'], bins=25, color='#534AB7', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('λ (loss aversion)')
axes[0].set_title(f'Loss aversion across 108 subjects\nmedian={params_df["lambda"].median():.2f}')
axes[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='neutral')
axes[0].legend(fontsize=9)

axes[1].hist(params_df['alpha'], bins=25, color='#1D9E75', alpha=0.7, edgecolor='white')
axes[1].set_xlabel('α (curvature)')
axes[1].set_title(f'Value curvature across 108 subjects\nmedian={params_df["alpha"].median():.2f}')

axes[2].hist(params_df['beta'], bins=25, color='#D85A30', alpha=0.7, edgecolor='white')
axes[2].set_xlabel('β (consistency)')
axes[2].set_title(f'Choice consistency across 108 subjects\nmedian={params_df["beta"].median():.2f}')

plt.tight_layout()
plt.savefig('hierarchical_params_final.png', dpi=150)
print("\n参数分布图已保存为 hierarchical_params_final.png")
print("\n全部完成！结果保存在：")
print("  - hierarchical_trace_final.nc（采样trace）")
print("  - individual_parameters_final.csv（个体参数）")
print("  - hierarchical_params_final.png（分布图）")