import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 读取数据集
df = pd.read_csv("Processed_Dataset_of_Diabetes_Version3.csv")

# 可选模型列表
model_paths = {
    "DNN": "DNN.h5",
    "FNN": "FNN.h5",
    "GBT": "GBT.joblib",
    "RF": "RF.joblib",
}

# 各模型对应的特征
feature_dict = {
    "DNN": ['BMI', 'Chol', 'HbA1c', 'AGE'],
    "FNN": ['BMI', 'Chol', 'HbA1c', 'AGE'],
    "GBT": ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI'],
    "RF": ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI'],
}

# 选择数据集分割比例
split_ratios = {"8:2": 0.2, "7:3": 0.3}

def evaluate_model(model_name, split_ratio):
    """ 加载模型并评估 """
    
    # 选择特征
    features = feature_dict[model_name]
    X = df[features]
    y = df["CLASS"]

    # 训练测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratios[split_ratio], random_state=42)

    # 标准化（仅 DNN 和 FNN 需要）
    if model_name in ["DNN", "FNN"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 加载模型
    if model_name in ["DNN", "FNN"]:
        model = load_model(model_paths[model_name])
        y_test_one_hot = np.eye(3)[y_test]  # One-hot 编码
        history = model.fit(X_train, y_test_one_hot, epochs=1, batch_size=10, validation_split=0.2, verbose=0)

        # 计算指标
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        loss = history.history["loss"][0]  # 取最后一次 loss
    else:
        model = load(model_paths[model_name])
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        loss = None  # 传统 ML 模型无 loss

    # 画 ACC、F1、Loss 曲线
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # ACC 柱状图
    sns.barplot(x=["Accuracy"], y=[acc], ax=ax[0])
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Accuracy")

    # F1 柱状图
    sns.barplot(x=["F1-Score"], y=[f1], ax=ax[1])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("F1 Score")

    # Loss 曲线（仅 DNN/FNN）
    if loss is not None:
        ax[2].plot(history.history["loss"], label="Training Loss")
        ax[2].plot(history.history["val_loss"], label="Validation Loss")
        ax[2].legend()
        ax[2].set_title("Loss Curve")
    else:
        ax[2].text(0.5, 0.5, "No Loss Available", ha="center", va="center", fontsize=12)
        ax[2].set_xticks([])
        ax[2].set_yticks([])

    plt.tight_layout()
    return fig

# 创建 Gradio 界面
demo = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.Dropdown(choices=list(model_paths.keys()), label="选择模型"),
        gr.Radio(choices=list(split_ratios.keys()), label="数据集分割"),
    ],
    outputs=gr.Plot(label="评估结果"),
    title="模型评估",
    description="选择模型和数据集分割方式，查看 Accuracy、F1-Score、Loss 曲线",
)

# 运行 Gradio
if __name__ == "__main__":
    demo.launch()
