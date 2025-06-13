# PCA-with-Logistic-Regression
### appley PCA with Logistic Regression

Here’s a step-by-step breakdown of what this notebook does:

---

## 1. Dataset

* **Source & Format**
  The data come from a local CSV file (`data-.csv`), which follows the familiar structure of the Wisconsin Breast Cancer dataset (as found on Kaggle).

* **Columns**

  1. **id**: a unique identifier (dropped early)
  2. **Unnamed: 32**: an empty column (also dropped)
  3. **diagnosis**: categorical label—“M” for malignant tumors and “B” for benign
  4. **30 numeric features**: measurements computed from digitized images of fine-needle aspirate (FNA) of breast masses (e.g. radius, texture, perimeter, area, smoothness, compactness, concavity, etc.).

* **Final shape**
  After dropping `id` and `Unnamed: 32`, you have 569 samples × 31 columns (1 label + 30 features).

---

## 2. Exploratory Data Analysis (EDA)

1. **Data loading & peek**

   ```python
   df = pd.read_csv('data-.csv')
   df.head()
   ```

2. **Cleaning**

   ```python
   df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
   df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
   ```

3. **Structure & completeness**

   * `df.info()` shows data types (all numerics after mapping)
   * `df.isnull().sum()` & `df.isna().sum()` confirm there are no missing values

4. **Basic statistics**

   ```python
   df.describe()
   ```

   gives mean, std, min/max, quartiles for each feature.

---

## 3. Preparing Features & Labels

* **Splitting X and y**

  ```python
  X = df.iloc[:, 1:].values   # all 30 features
  y = df.iloc[:, 0].values    # 0 = malignant, 1 = benign
  ```

* **Scaling**
  Most algorithms (and PCA) work better when features are on the same scale:

  ```python
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

  After this, each feature has mean 0 and variance 1.

---

## 4. Principal Component Analysis (PCA)

* **What it is**: an unsupervised method that finds orthogonal directions (principal components) capturing the most variance in your data.

* **Here**:

  ```python
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X_scaled)
  ```

  * **`explained_variance_`**: the raw variances of the first two components
  * **`explained_variance_ratio_`**: fraction of total dataset variance captured by each PC
  * **Total explained variance**: typically around 70–80% for PC1+PC2 on this dataset.

* **Why**:

  * **Visualization**: project 30-D → 2-D to see if malignant vs. benign points separate
  * **Dimensionality reduction**: use just these two PCs as inputs into a classifier

---

## 5. Visualization

* **Original vs. PCA space**
  A 2×2 grid of scatter plots:

  1. Original data plotted on its first two *scaled* features (for comparison)
  2. PCA-transformed data on PC1 vs. PC2

  * Points are colored by diagnosis (malignant/benign), with colorbars.

* **Insight**:
  You should see better clustering/separation in the PCA plot than in the raw-feature plot.

---

## 6. Classification with Logistic Regression

1. **Train/test split**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X_pca, y, test_size=0.2, random_state=42
   )
   ```

2. **Modeling**

   ```python
   lr = LogisticRegression()
   lr.fit(X_train, y_train)
   y_pred = lr.predict(X_test)
   ```

3. **Evaluation**

   * **Confusion matrix** & **classification report** (precision, recall, F1) printed
   * **Heatmap** of the confusion matrix (via Seaborn)

This shows how well a simple linear classifier can separate malignant from benign once you’ve reduced to two dimensions.

---

## 7. PCA Reconstruction Loss

* **What**: measures how much information PCA dropped when you reduced from 30 → 2 dimensions.
* **How**:

  ```python
  X_reconstructed = pca.inverse_transform(X_pca)
  loss = mean((X_scaled - X_reconstructed)**2)
  print(f"Reconstruction Loss: {loss:.4f}")
  ```
* **Interpretation**: lower loss means the two principal components capture most of the original data’s structure.

---

## Main Idea

1. **Load & clean** a real-world medical dataset.
2. **Explore** it (structure, missingness, basic stats).
3. **Preprocess** via standardization.
4. **Reduce dimensionality** with PCA for both visualization and computational efficiency.
5. **Classify** the reduced data with logistic regression.
6. **Evaluate** both the classification performance and the information loss from PCA.

Together, this workflow illustrates a common pattern in machine learning:

1. EDA & cleaning
2. Preprocessing
3. Unsupervised feature reduction
4. Supervised modeling
5. Visualization & quantitative evaluation.
