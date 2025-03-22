import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# 加载数据集
df = pd.read_csv('Processed_Dataset_of_Diabetes_Version3.csv')

# 选择特征和目标变量
X = df[['BMI', 'Chol', 'HbA1c', 'AGE']]
y = df['CLASS']

# 确保类别是数值型
y = y.astype('category').cat.codes

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建 FNN 模型
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')  # 假设 CLASS 有 3 个类别
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，并保存训练历史
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 评估模型
y_pred = np.argmax(model.predict(X_test), axis=1)

# 计算和打印指标
print("Classification Report:")
print(classification_report(y_test, y_pred))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 计算 AUC-ROC
y_prob = model.predict(X_test)  # 获取所有类别的概率
auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"AUC-ROC: {auc_roc:.4f}")

# **绘制 Loss 曲线**
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('FNN Training and Validation Loss')
plt.legend()
plt.savefig("FNN_loss_curve.png")
plt.show()

# 保存模型
dump(model, 'FNN3.joblib')
