import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='devadidev123', repo_name='MLops', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/devadidev123/MLops.mlflow')

# remote connecting server using dagshub so that everyone can see mlflow of your

 
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
 
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
 
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
 
# Define the params for RF model
max_depth = 10
n_estimators = 5
 
# Mention your experiment below
mlflow.set_experiment('YT-MLOPS-Exp2')# it makes code ml code to run in particular amed mlflow project table
 
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion-matrix.png")

    # Log artifacts
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)  # log your script

    # Tags
    mlflow.set_tags({"Author": 'Aditya', "Project": "Wine Classification"})

    # ✅ Fix: log model in a DagsHub-compatible way
    mlflow.sklearn.log_model(rf, artifact_path="Random-Forest-Model")
    # or:
    # mlflow.sklearn.save_model(rf, "Random-Forest-Model")
    # mlflow.log_artifact("Random-Forest-Model")
    #pip install mlflow==2.9.2
#https://dagshub.com/devadidev123/MLops.mlflow/#/experiments/1?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D


    print(accuracy)
