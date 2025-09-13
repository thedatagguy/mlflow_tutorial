import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

mlflow.set_experiment('mlflow_exp1') 

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("Accuracy",accuracy)
    mlflow.log_metric("Max_depth",max_depth)
    mlflow.log_metric("n_estimaators",n_estimators)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix') 


    plt.savefig("Confusion_matrix.png")

    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({"Author":"Manoj","project":"nkn ley"})
    
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    print(accuracy)
