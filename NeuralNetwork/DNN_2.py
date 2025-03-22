import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# 加载数据
df = pd.read_csv('Processed_Dataset_of_Diabetes_Version3.csv')

# 确保类别从 0 开始
df['CLASS'] = df['CLASS'] - df['CLASS'].min()

# 选择特征
X = df[['BMI', 'Chol', 'HbA1c', 'AGE']]
y = tf.keras.utils.to_categorical(df['CLASS'], num_classes=3)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换数据类型
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# 构建 DNN 模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 评估模型
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 计算 AUC-ROC
y_prob = model.predict(X_test)
auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
print(f"AUC-ROC: {auc_roc:.4f}")

# 绘制 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 保存模型
model.save("DNN.h5")
dump({"model": model, "scaler": scaler}, "DNN.joblib")
