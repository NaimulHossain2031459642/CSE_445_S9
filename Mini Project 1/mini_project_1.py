import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. ডেটা লোড
df = pd.read_excel("stroop_test_data.xlsx")

# 2. K-means দিয়ে লেবেল বানানো
features = df[['Accuracy(%)', 'ReactionTime']]
kmeans = KMeans(n_clusters=2, random_state=42)
df['ClusterLabel'] = kmeans.fit_predict(features)

# ✅ 2.1 K-means ভিজ্যুয়ালাইজেশন (Scatter Plot)
plt.figure(figsize=(8,6))
plt.scatter(df['Accuracy(%)'], df['ReactionTime'], c=df['ClusterLabel'], cmap='viridis', s=80, alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel("Accuracy (%)")
plt.ylabel("Reaction Time")
plt.title("K-means Clustering Result")
plt.legend()
plt.show()

# ✅ 2.2 Cluster Bar Chart (Count)
cluster_counts = df['ClusterLabel'].value_counts().sort_index()
plt.figure(figsize=(6,5))
cluster_counts.plot(kind='bar', color=['skyblue','orange'])
plt.xlabel("Cluster")
plt.ylabel("Number of Participants")
plt.title("Number of Participants in Each Cluster")
plt.xticks([0,1], ["Cluster 0", "Cluster 1"], rotation=0)
plt.show()

# ✅ 2.3 Cluster অনুযায়ী গড় Accuracy এবং Reaction Time (Mean Values)
cluster_means = df.groupby('ClusterLabel')[['Accuracy(%)', 'ReactionTime']].mean()
cluster_means.plot(kind='bar', figsize=(8,6))
plt.title("Average Accuracy and Reaction Time per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.xticks([0,1], ["Cluster 0", "Cluster 1"], rotation=0)
plt.legend(title="Metrics")
plt.show()

# 3. ডেটা ভাগ করা
X_train, X_test, y_train, y_test = train_test_split(features, df['ClusterLabel'], test_size=0.3, random_state=42)

# 4. SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 5. Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 6. Confusion Matrix SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(cm_svm).plot()
plt.title("SVM Confusion Matrix")
plt.show()

# 7. Confusion Matrix Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

# 8. Decision Tree ভিজ্যুয়াল
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=['Accuracy','ReactionTime'], class_names=['Cluster 0','Cluster 1'], filled=True)
plt.show()
