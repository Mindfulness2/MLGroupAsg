import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
data = pd.read_csv('Processed_Dataset_of_Diabetes_Version3.csv')

X = data.drop('CLASS', axis=1)  # Features
y = data['CLASS']                # Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.dtypes)
print(X_train['Gender'].unique())
print(X_train.isnull().sum())

# 计算 'Gender' 列中值为 'f' 的行数
f_count = (X_train['Gender'] == 'f').sum()
print("Number of rows with 'f' in Gender column:", f_count)


# # 过滤 X_train 和 y_train
# mask = X_train['Gender'].isin(['0', '1'])
# X_train = X_train[mask]
# y_train = y_train[mask]
#
# # 将 'Gender' 列从对象类型转换为整数类型
# X_train['Gender'] = X_train['Gender'].astype(int)


# 创建梯度提升树分类器
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
gb_model.fit(X_train, y_train)

# 进行预测
y_pred = gb_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印分类报告
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

import matplotlib.pyplot as plt

feature_importance = gb_model.feature_importances_
plt.barh(range(len(feature_importance)), feature_importance)
plt.yticks(range(len(feature_importance)), X.columns)
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Gradient Boosting Model')
plt.show()

dump(gb_model,'GBT3.joblib')