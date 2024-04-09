# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('data/co-KSH-BJ-3.csv', header=None)
#
# # 合并第1列和第2列，中间用'_'连接
# df['merged'] = df[0].astype(str) + '_' + df[1].astype(str)
#
# # 按照合并后的值对第4列进行汇总
# merge_data = df.groupby('merged')[2].agg(lambda x: ','.join(x.astype(str))).reset_index()
#
# lines = []
# for index, row in merge_data.iterrows():
#     line = []
#     for j in row['merged'].split('_'):
#         line.append(j)
#
#     for j in row[2].split(','):
#         if (not line.__contains__(j)):
#             line.append(j)
#     lines.append(line)
#
# # 将结果转换为DataFrame
# final_df = pd.DataFrame(lines)
#
# # 保存为CSV文件（保存到当前工作目录）
# final_df.to_csv('co-KSH_HB-BJ-3.csv', header=None, index=None)


import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/co-KSH-BJ-3.csv', header=None)

# 对前两列进行排序
df[[0, 1]] = df.apply(lambda x: sorted([x[0], x[1]], reverse=True), axis=1, result_type='expand')

# 合并第1列和第2列，中间用'_'连接
df['merged1'] = df[0].astype(str) + '_' + df[1].astype(str)

# 按照合并后的值对第4列进行汇总
merge_data = df.groupby('merged1')[2].agg(lambda x: ','.join(x.astype(str))).reset_index()

lines = []
for index, row in merge_data.iterrows():
    line = []
    for j in row['merged1'].split('_'):
        line.append(j)

    for j in row[2].split(','):
        if (not line.__contains__(j)):
            line.append(j)
    lines.append(line)

# 将结果转换为DataFrame
final_df = pd.DataFrame(lines)

# 保存为CSV文件（保存到当前工作目录）
final_df.to_csv('co-KSH_HB-BJ-3_2.csv', header=None, index=None)
