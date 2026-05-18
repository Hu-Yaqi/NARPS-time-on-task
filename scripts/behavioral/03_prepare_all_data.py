import pandas as pd
import os

# ============================================================
# 把108个被试 × 4个run的数据全部合并成一张大表
# ============================================================

all_data = []  # 空列表，用来收集所有数据

# os.listdir 列出data文件夹里的所有文件/文件夹名
# sorted 让它们按字母顺序排列（sub-001, sub-002, ...）
data_dir = 'data'
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
# 这行代码的意思是：
# 1. os.listdir(data_dir) 列出data文件夹里所有东西
# 2. [s for s in ... if s.startswith('sub-')] 只保留以"sub-"开头的
#    这叫"列表推导式"，是Python里筛选列表的简洁写法
# 3. sorted() 按字母顺序排序

print(f"找到 {len(subjects)} 个被试")

for sub in subjects:
    func_dir = os.path.join(data_dir, sub, 'func')
    # os.path.join 把多段路径拼接起来，比如 'data' + 'sub-001' + 'func'
    # 变成 'data/sub-001/func'

    # 找到这个被试的所有events.tsv文件
    event_files = sorted([f for f in os.listdir(func_dir) if f.endswith('events.tsv')])

    for f in event_files:
        filepath = os.path.join(func_dir, f)
        run_df = pd.read_csv(filepath, sep='\t')

        # 从文件名中提取被试编号和run编号
        # 例如文件名 sub-001_task-MGT_run-01_events.tsv
        # split('_') 按下划线切开 → ['sub-001', 'task-MGT', 'run-01', 'events.tsv']
        parts = f.split('_')
        run_df['subject'] = parts[0]       # 'sub-001'
        run_df['run'] = int(parts[2].split('-')[1])  # 'run-01' → 1

        all_data.append(run_df)

# 合并成一张大表
df = pd.concat(all_data, ignore_index=True)

# ============================================================
# 数据清洗
# ============================================================

# 创建二分变量：接受=1，拒绝=0
# 后面建模需要的是0和1，不是文字
df['accepted'] = df['participant_response'].str.contains('accept').astype(int)
# .str.contains('accept') → True/False
# .astype(int) → 把True变成1，False变成0

# 标记无效trial（没有响应的）
df['valid'] = df['participant_response'] != 'NoResp'

# 统计一下
n_total = len(df)
n_valid = df['valid'].sum()
n_subjects = df['subject'].nunique()  # nunique = number of unique（有多少个不同的值）

print(f"\n数据汇总：")
print(f"  被试数量：{n_subjects}")
print(f"  总trial数：{n_total}")
print(f"  有效trial数：{n_valid} ({n_valid/n_total:.1%})")
print(f"  无响应trial数：{n_total - n_valid} ({(n_total-n_valid)/n_total:.1%})")

# 看看每个被试的接受率分布
# groupby('subject') 意思是"按被试分组"
# 然后对每组的'accepted'列求平均值
accept_rates = df[df['valid']].groupby('subject')['accepted'].mean()
print(f"\n接受率统计（across subjects）：")
print(f"  最低：{accept_rates.min():.1%}")
print(f"  最高：{accept_rates.max():.1%}")
print(f"  中位数：{accept_rates.median():.1%}")
print(f"  平均：{accept_rates.mean():.1%}")

# ============================================================
# 保存清洗后的数据，方便后面直接使用
# ============================================================

# 只保存有效trial
clean_df = df[df['valid']].copy()
# .copy() 创建一个独立副本，避免后续操作时pandas报警告

clean_df.to_csv('all_subjects_behavior.csv', index=False)
# .to_csv 把表格保存为csv文件
# index=False 表示不要把行号写进文件

print(f"\n已保存清洗后的数据到 all_subjects_behavior.csv")
print(f"文件包含 {len(clean_df)} 行 × {len(clean_df.columns)} 列")
print(f"列名：{clean_df.columns.tolist()}")