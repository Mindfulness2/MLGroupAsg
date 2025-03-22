import os
import gradio as gr
import joblib
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tensorflow.keras.models import load_model

# 读取数据集
df = pd.read_csv("Processed_Dataset_of_Diabetes_Version3.csv")
X = df.iloc[:, :-1]  # 特征列
y = df.iloc[:, -1]   # 标签列
MODELS_DIR = "./models"
# 可选的模型列表
model_options = {
    "DNN": "DNN.joblib",
    "FNN": "FNN.joblib",
    "GBT": "GBT.joblib",
    "RandomForest": "RandomForest.joblib",
    "LightGBM": "LightGBM.joblib",
}



def evaluate_model(model_name, split_ratio):
    # 加载模型
    #model_path = model_options[model_name]
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if split_ratio != "8:2":
        model_path = os.path.join(MODELS_DIR, f"{model_name}3.joblib")
    # 训练损失和验证损失图片路径
    if model_name in ["DNN", "FNN"]:
        if split_ratio == "8:2":
            LOSS_IMG = f"{model_name}_loss_curve.png"
        else:
            LOSS_IMG = f"{model_name}3_loss_curve.png"
    elif model_name in ["LightGBM"]:
        if split_ratio == "8:2":
            LOSS_IMG = f"{model_name}_loss_curve.png"
        else:
            LOSS_IMG = f"{model_name}3_loss_curve.png"
    else:
        LOSS_IMG = None

    model = joblib.load(model_path)
    print(model)
    
    if model_name in ["DNN", "FNN"]:
        X_selected = X[['BMI', 'Chol', 'HbA1c', 'AGE']]
    else:
        X_selected = X  # 其他模型用全部特征

    
    # 划分数据集
    test_size = 0.2 if split_ratio == "8:2" else 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # 如果 y_pred 是概率分布或 one-hot 编码，转换成类别索引
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)
    
    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_proba) if y_proba is not None else np.nan
    
    # 生成图像
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 绘制 F1 Score
    axes[0].bar(["F1 Score"], [f1], color='blue')
    axes[0].set_ylim(0, 1)
    axes[0].set_title("F1 Score")
    
    # 绘制 Accuracy
    axes[1].bar(["Accuracy"], [acc], color='green')
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    
    # 绘制 Loss
    axes[2].bar(["Loss"], [loss], color='red')
    axes[2].set_ylim(0, max(loss, 1) if not np.isnan(loss) else 1)
    axes[2].set_title("Loss")
    
    plt.tight_layout()
    
    return acc, f1, loss, fig, LOSS_IMG

# # Gradio 界面
# demo = gr.Interface(
#     fn=evaluate_model,
#     inputs=[
#         gr.Dropdown(list(model_options.keys()), label="选择模型"),
#         gr.Radio(["8:2", "7:3"], label="数据集分割比例")
#     ],
#     outputs=[
#         gr.Textbox(label="Accuracy"),
#         gr.Textbox(label="F1 Score"),
#         gr.Textbox(label="Loss"),
#         gr.Plot(label="评估图像"),
#         gr.Image(label="Training Loss 曲线"),
#     ],
#     title="机器学习模型评估",
#     description="选择一个模型并设置数据集分割比例，查看评估结果"
# )

with gr.Blocks(title="Diabetes Model Evaluation") as demo:
      
    with gr.Row():
        # 左侧面板：模型选择和评估指标
        with gr.Column(scale=1):
            gr.Markdown("# Model Evaluation")
            gr.Markdown("Select a model and set the dataset split ratio to view the evaluation results.")

            model = gr.Dropdown(list(model_options.keys()), label="Select Model")
            split_ratio = gr.Radio(["8:2", "7:3"], label="Dataset split ratio")
            
            # 评估按钮
            eval_button = gr.Button("Evaluate")
            
            # 评估指标 - 创建为输出组件
            accuracy_output = gr.Textbox(label="Accuracy")
            f1_score_output = gr.Textbox(label="F1 Score")
            loss_output = gr.Textbox(label="Loss")
        
        # 右侧面板：图表显示
        with gr.Column(scale=2):
            eval_plot = gr.Plot(label="Evaluate")
            train_loss_curve = gr.Image(label="Training Loss Curve")
    
    # 按钮点击事件
    eval_button.click(
        fn=evaluate_model,
        inputs=[model, split_ratio],
        outputs=[accuracy_output, f1_score_output, loss_output, eval_plot, train_loss_curve]
    )
demo.launch()
