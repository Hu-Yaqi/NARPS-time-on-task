import pandas as pd

# 读取1号被试第1个run的行为数据
# sep='\t' 告诉pandas这个文件用tab分隔（tsv格式）
df = pd.read_csv('data/sub-001/func/sub-001_task-MGT_run-01_events.tsv', sep='\t')

# 查看表格的大小：(行数, 列数)
print("表格大小:", df.shape)

# 查看所有列的名字
print("所有列名:", df.columns.tolist())

# 查看前5行数据
print("\n前5行数据:")
print(df.head())