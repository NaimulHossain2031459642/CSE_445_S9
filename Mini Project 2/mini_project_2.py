import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. ডেটা লোড
df = pd.read_excel("stroop_test_data.xlsx")

# Column নাম ক্লিন
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())

# Accuracy মান চেক
print(df['Accuracy(%)'].describe())

# 2. Threshold অটো-সেট (মিডিয়ান + সামান্য মার্জিন)
auto_threshold = df['Accuracy(%)'].median()
df['ManualLabel'] = np.where(df['Accuracy(%)'] > auto_threshold, 1, 0)

# যদি এখনো এক ক্লাস থাকে, fallback threshold ব্যবহার করো
if df['ManualLabel'].nunique() < 2:
    fallback_threshold = df['Accuracy(%)'].mean()
    df['ManualLabel'] = np.where(df['Accuracy(%)'] > fallback_threshold, 1, 0)

print("Class distribution:\n", df['ManualLabel'].value_counts())

# ✅ 2.1 Scatter Plot (Manual Labels)
plt.figure(figsize=(8,6))
plt.scatter(df['Accuracy(%)'], df['ReactionTime'], c=df['ManualLabel'], cmap='viridis', s=80, alpha=0.7, edgecolors='k')
plt.xlabel("Accuracy (%)")
plt.ylabel("Reaction Time")
plt.title("Scatter Plot: Accuracy vs Reaction Time (Manual Labels)")
plt.show()

# ✅ 2.2 Bar Chart (Class distribution)
class_counts = df['ManualLabel'].value_counts().sort_index()
plt.figure(figsize=(6,5))
class_counts.plot(kind='bar', color=['skyblue','orange'])
plt.xlabel("Class")
plt.ylabel("Number of Participants")
plt.title("Number of Participants per Class")
plt.xticks([0,1], ["Class 0", "Class 1"], rotation=0)
plt.show()

# ✅ 2.3 Mean Accuracy & ReactionTime per Class
class_means = df.groupby('ManualLabel')[['Accuracy(%)', 'ReactionTime']].mean()
class_means.plot(kind='bar', figsize=(8,6))
plt.title("Average Accuracy and Reaction Time per Class")
plt.xlabel("Class")
plt.ylabel("Average Value")
plt.xticks([0,1], ["Class 0", "Class 1"], rotation=0)
plt.legend(title="Metrics")
plt.show()

# 3. Statistical Features
df['Accuracy_mean'] = df['Accuracy(%)'].mean()
df['Accuracy_std'] = df['Accuracy(%)'].std()
df['RT_mean'] = df['ReactionTime'].mean()
df['RT_std'] = df['ReactionTime'].std()

features = df[['Accuracy(%)', 'ReactionTime', 'Accuracy_mean', 'Accuracy_std', 'RT_mean', 'RT_std']]
target = df['ManualLabel']

# 4. Train-Test Split (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# 5. Neural Network
nn = MLPClassifier(max_iter=1000)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)

# 6. Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 7. XGBoost
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 8. Confusion Matrix প্লট
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)
    plt.show()

plot_cm(y_test, y_pred_nn, "Neural Network")
plot_cm(y_test, y_pred_rf, "Random Forest")
plot_cm(y_test, y_pred_xgb, "XGBoost")
