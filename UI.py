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
    "XGBoost": "XGBoost.joblib",
}

# 特征名称列表（根据你的示例）
FEATURES = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# 模型对应的特征子集
MODEL_FEATURES = {
    "DNN": ['BMI', 'Chol', 'HbA1c', 'AGE'],
    "FNN": ['BMI', 'Chol', 'HbA1c', 'AGE'],
    "XGBoost": ['Urea', 'Cr', 'HbA1c', 'BMI'],
    "GBT": FEATURES,  # 假设这些模型使用所有特征
    "RandomForest": FEATURES,
    "LightGBM": FEATURES
}

def evaluate_model(model_name, split_ratio):
    # 原有的evaluate_model函数保持不变
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if split_ratio != "8:2":
        model_path = os.path.join(MODELS_DIR, f"{model_name}3.joblib")
    
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
    
    if model_name in ["DNN", "FNN"]:
        X_selected = X[['BMI', 'Chol', 'HbA1c', 'AGE']]
    elif model_name in ["XGBoost"]:
        X_selected = X[['Urea', 'Cr', 'HbA1c', 'BMI']]
    else:
        X_selected = X
    
    test_size = 0.2 if split_ratio == "8:2" else 0.3
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_proba) if y_proba is not None else np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].bar(["F1 Score"], [f1], color='blue')
    axes[0].set_ylim(0, 1)
    axes[0].set_title("F1 Score")
    
    axes[1].bar(["Accuracy"], [acc], color='green')
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    
    axes[2].bar(["Loss"], [loss], color='red')
    axes[2].set_ylim(0, max(loss, 1) if not np.isnan(loss) else 1)
    axes[2].set_title("Loss")
    
    plt.tight_layout()
    
    return acc, f1, loss, fig, LOSS_IMG

def predict_class(model_name, gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi):
    # 加载模型
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    model = joblib.load(model_path)
    
    # 将输入数据组织成字典
    input_data = {
        'Gender': gender, 'AGE': age, 'Urea': urea, 'Cr': cr, 'HbA1c': hba1c,
        'Chol': chol, 'TG': tg, 'HDL': hdl, 'LDL': ldl, 'VLDL': vldl, 'BMI': bmi
    }
    
    # 根据模型选择相关特征
    selected_features = MODEL_FEATURES[model_name]
    input_array = np.array([[input_data[feat] for feat in selected_features]])
    
    # 进行预测
    prediction = model.predict(input_array)
    
    # 处理预测结果（如果是多维输出，取argmax）
    if len(prediction.shape) > 1:
        prediction = prediction.argmax(axis=1)
    
    return f"Predicted CLASS: {prediction[0]}"

with gr.Blocks(title="Diabetes Model Evaluation") as demo:
    with gr.Tabs() as tabs:

        
        # 预测页面Tab
        with gr.TabItem("Prediction", id="predict"):
            gr.Markdown("# Diabetes Prediction System")
            gr.Markdown("Enter the feature values and select a model to predict diabetes risk classification.")
            
            with gr.Row():
                # Left side: Model selection and prediction button area
                with gr.Column(scale=1):
                    gr.Markdown("### Model Selection")
                    pred_model = gr.Dropdown(
                        list(model_options.keys()), 
                        label="Select Prediction Model",
                        value=list(model_options.keys())[0],  # Set default value
                        container=True
                    )
                    
                    gr.Markdown("### Prediction Results", elem_id="result_header")
                    prediction_output = gr.Textbox(
                        label="Prediction Result",
                        placeholder="Click 'Predict' button to see results...",
                        elem_id="prediction_result"
                    )
                    
                    predict_button = gr.Button(
                        "Predict", 
                        variant="primary",
                        size="lg"
                    )
                
                # Right side: All input features
                with gr.Column(scale=1):
                    gr.Markdown("### Patient Features")
                    
                    with gr.Group():
                        gr.Markdown("#### Basic Information")
                        with gr.Row():
                            gender = gr.Radio(
                                choices=[0, 1], 
                                label="Gender (0-Female, 1-Male)", 
                                value=1
                            )
                            age = gr.Slider(
                                minimum=18, 
                                maximum=90, 
                                value=50, 
                                label="Age"
                            )
                            bmi = gr.Slider(
                                minimum=15, 
                                maximum=45, 
                                value=24, 
                                label="BMI"
                            )
                    
                    with gr.Group():
                        gr.Markdown("#### Blood Test Indicators")
                        with gr.Row():
                            urea = gr.Number(label="Urea", value=4.7)
                            cr = gr.Number(label="Creatinine (Cr)", value=46)
                        
                        with gr.Row():
                            hba1c = gr.Number(label="Glycated Hemoglobin (HbA1c)", value=4.9)
                        
                        gr.Markdown("#### Lipid Profile")
                        with gr.Row():
                            chol = gr.Number(label="Total Cholesterol (Chol)", value=4.2)
                            tg = gr.Number(label="Triglycerides (TG)", value=0.9)
                        
                        with gr.Row():
                            hdl = gr.Number(label="High-Density Lipoprotein (HDL)", value=2.4)
                            ldl = gr.Number(label="Low-Density Lipoprotein (LDL)", value=1.4)
                            vldl = gr.Number(label="Very Low-Density Lipoprotein (VLDL)", value=0.5)
        # 主页面Tab
        with gr.TabItem("Model Evaluation", id="main"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# Model Evaluation")
                    gr.Markdown("Select a model and set the dataset split ratio to view the evaluation results.")

                    model = gr.Dropdown(list(model_options.keys()), label="Select Model")
                    split_ratio = gr.Radio(["8:2", "7:3"], label="Dataset split ratio")
                    
                    eval_button = gr.Button("Evaluate")
                    
                    accuracy_output = gr.Textbox(label="Accuracy")
                    f1_score_output = gr.Textbox(label="F1 Score")
                    loss_output = gr.Textbox(label="Loss")
                
                with gr.Column(scale=2):
                    eval_plot = gr.Plot(label="Evaluate")
                    train_loss_curve = gr.Image(label="Training Loss Curve")
    
    # 绑定评估按钮事件
    eval_button.click(
        fn=evaluate_model,
        inputs=[model, split_ratio],
        outputs=[accuracy_output, f1_score_output, loss_output, eval_plot, train_loss_curve]
    )
    
    # 绑定预测按钮事件
    predict_button.click(
        fn=predict_class,
        inputs=[pred_model, gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi],
        outputs=prediction_output
    )

demo.launch()