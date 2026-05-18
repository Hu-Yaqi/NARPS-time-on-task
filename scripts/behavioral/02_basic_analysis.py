import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 第一部分：把4个run的数据合并成一张大表
# ============================================================

# 创建一个空列表，用来收集每个run的数据
all_runs = []

# range(1, 5) 会生成 1, 2, 3, 4 四个数字，对应4个run
for run in range(1, 5):
    # f"..." 是格式化字符串，{run:02d} 的意思是把数字补成两位（1变成01）
    path = f'data/sub-001/func/sub-001_task-MGT_run-{run:02d}_events.tsv'
    run_df = pd.read_csv(path, sep='\t')

    # 给这个表加一列，记录它来自哪个run
    run_df['run'] = run

    # 把这个run的数据放进列表里
    all_runs.append(run_df)

# pd.concat 把列表里的4张小表纵向拼接成一张大表
# ignore_index=True 让行号从0重新编号
df = pd.concat(all_runs, ignore_index=True)

print(f"合并后：{df.shape[0]} 行（trials），{df.shape[1]} 列")

# ============================================================
# 第二部分：基本统计
# ============================================================

# 把选择简化成"接受"或"拒绝"两类
# .str.contains('accept') 检查每个选择里是否包含"accept"这个词
# 结果是一列 True/False，True=接受，False=拒绝
df['accepted'] = df['participant_response'].str.contains('accept')

# .mean() 对 True/False 列求平均值——True算1，False算0
# 所以结果就是接受赌博的比例
accept_rate = df['accepted'].mean()
print(f"\n这个被试接受了 {accept_rate:.1%} 的赌博")
# :.1% 是格式化语法，意思是"显示为百分比，保留1位小数"

# 排除没有响应的trial（RT为0的那些）
valid_df = df[df['RT'] > 0]
print(f"平均反应时：{valid_df['RT'].mean():.2f} 秒")

# ============================================================
# 第三部分：画一张图——这个人的"决策地图"
# ============================================================

# 创建一张图，figsize=(8, 6) 设置图的宽8英寸、高6英寸
fig, ax = plt.subplots(figsize=(8, 6))

# 把数据分成"接受"和"拒绝"两组，分别画散点
accepted = df[df['accepted'] == True]
rejected = df[df['accepted'] == False]

# 画散点图：x轴是gain，y轴是loss
# alpha=0.6 让点半透明，这样重叠的点能看出密度
# s=40 设置点的大小
ax.scatter(accepted['gain'], accepted['loss'],
           color='#1D9E75', label='Accepted', alpha=0.6, s=40)
ax.scatter(rejected['gain'], rejected['loss'],
           color='#D85A30', label='Rejected', alpha=0.6, s=40)

# 设置轴标签和标题
ax.set_xlabel('Potential gain', fontsize=13)
ax.set_ylabel('Potential loss', fontsize=13)
ax.set_title('Sub-001: Decision map', fontsize=15)

# 添加图例（告诉读者绿色和橙色分别代表什么）
ax.legend(fontsize=12)

# 保存图片到项目文件夹
plt.tight_layout()
plt.savefig('sub001_decision_map.png', dpi=150)
print("\n图片已保存为 sub001_decision_map.png")
print("在PyCharm左侧文件列表里双击它就能看到")
