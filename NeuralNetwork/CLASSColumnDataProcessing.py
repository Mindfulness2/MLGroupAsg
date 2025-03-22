import pandas as pd

data = pd.read_csv('Processed_Dataset_of_Diabetes_Version2.csv')

data['CLASS'] = data['CLASS'].str.strip()

# 计算每列的缺失值数量
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# 查看目标变量 'CLASS' 的唯一值
unique_classes = data['CLASS'].unique()
print("Unique classes:", unique_classes)

# 计算唯一类别的数量
num_classes = len(unique_classes)

# 计算目标变量中 'N' 的出现次数
n_count = (data['CLASS'] == 'N').sum()
print("Count of 'N':", n_count)

n_records = data[data['CLASS'] == 'N']
print(n_records)
