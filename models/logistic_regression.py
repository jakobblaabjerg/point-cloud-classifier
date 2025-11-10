



def train_logistic_regression(dataset):

    df = dataset.copy()

    # Features and target
    X = df[["energy_total", "hits_total", "energy_hcal/ecal", "hits_hcal/ecal"]]
    y = df["label"]

    # Encode target as numeric
    y = y.map({"proton": 0, "piM": 1})

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


    df = pd.DataFrame(X_test, columns=["energy_total", "hits_total"]).copy()
    df["label"] = y_test
    plot_data(df)

    df["label"] = y_pred
    plot_data(df)



