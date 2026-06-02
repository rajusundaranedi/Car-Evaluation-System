# ============================================================
# 10. MULTI-MODEL COMPARISON
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import joblib

print("\n" + "=" * 60)
print(" ADVANCED MODEL COMPARISON ")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": best_svm
}

results = {}

print("\nModel Performance Comparison")
print("-" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    cv = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5,
        scoring="accuracy"
    )

    results[name] = {
        "accuracy": acc,
        "cv_mean": cv.mean(),
        "cv_std": cv.std()
    }

    print(
        f"{name:<22}"
        f" Accuracy={acc:.4f}"
        f" | CV={cv.mean():.4f}"
    )

# ============================================================
# 11. MODEL RANKING
# ============================================================

ranking = pd.DataFrame({
    "Model": results.keys(),
    "Accuracy": [
        results[m]["accuracy"]
        for m in results
    ],
    "CV Mean": [
        results[m]["cv_mean"]
        for m in results
    ]
})

ranking = ranking.sort_values(
    "Accuracy",
    ascending=False
)

print("\nModel Ranking")
print(ranking)

plt.figure(figsize=(10,5))

plt.bar(
    ranking["Model"],
    ranking["Accuracy"]
)

plt.xticks(rotation=15)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "model_comparison.png",
    dpi=150
)
plt.show()

# ============================================================
# 12. PCA VISUALIZATION
# ============================================================

print("\nGenerating PCA Visualization...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap="viridis",
    alpha=0.7
)

plt.title("PCA Projection of Car Evaluation Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(scatter)

plt.tight_layout()
plt.savefig(
    "pca_visualization.png",
    dpi=150
)

plt.show()

# ============================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(8,5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig(
    "feature_importance.png",
    dpi=150
)

plt.show()

# ============================================================
# 14. LEARNING CURVE
# ============================================================

print("\nGenerating Learning Curve...")

sizes, train_scores, test_scores = learning_curve(
    best_svm,
    X_scaled,
    y,
    cv=5,
    train_sizes=np.linspace(0.1,1.0,10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))

plt.plot(
    sizes,
    train_mean,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    sizes,
    test_mean,
    marker="s",
    label="Validation Accuracy"
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "learning_curve.png",
    dpi=150
)

plt.show()

# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

best_model_name = ranking.iloc[0]["Model"]

best_model = models[best_model_name]

joblib.dump(
    best_model,
    "best_car_model.pkl"
)

joblib.dump(
    scaler,
    "car_scaler.pkl"
)

print(
    f"\nBest model saved: "
    f"{best_model_name}"
)

# ============================================================
# 16. PREDICTION WITH PROBABILITIES
# ============================================================

def predict_car_proba(
    buying,
    maint,
    doors,
    persons,
    lug_boot,
    safety
):

    row = [[
        category_maps["buying"][buying],
        category_maps["maint"][maint],
        category_maps["doors"][doors],
        category_maps["persons"][persons],
        category_maps["lug_boot"][lug_boot],
        category_maps["safety"][safety]
    ]]

    row_scaled = scaler.transform(row)

    pred = best_model.predict(row_scaled)[0]

    print("\nPrediction:")

    print(
        "Class:",
        class_map_inv[pred]
    )

    if hasattr(best_model, "predict_proba"):

        probs = best_model.predict_proba(
            row_scaled
        )[0]

        print("\nProbabilities:")

        for i, p in enumerate(probs):
            print(
                f"{class_map_inv[i]} : {p:.4f}"
            )

# Example

predict_car_proba(
    buying="low",
    maint="low",
    doors="4",
    persons="more",
    lug_boot="big",
    safety="high"
)

print("\nAdvanced pipeline completed.")
print("Generated Files:")
print("- model_comparison.png")
print("- pca_visualization.png")
print("- feature_importance.png")
print("- learning_curve.png")
print("- best_car_model.pkl")
print("- car_scaler.pkl")
