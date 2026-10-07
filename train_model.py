from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib

# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=10
)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# 5. Train model
knn.fit(X_train_scaled, y_train)

# 6. Test model
y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=iris.target_names))

# 7. Save model and scaler
joblib.dump(knn, "iris_knn_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")

print("\nModel saved as: iris_knn_model.pkl")
print("Scaler saved as: iris_scaler.pkl")


