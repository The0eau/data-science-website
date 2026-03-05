# Unsupervised Learning Algorithms Overview

This module covers the fundamental algorithms used to find hidden patterns in unlabeled data.

---

## 1. Clustering
*Goal: Group similar data points together based on specific features.*

| Algorithm | Category | Best For... |
| :--- | :--- | :--- |
| **K-Means** | Centroid-based | Large datasets with spherical clusters. |
| **DBSCAN** | Density-based | Finding clusters of arbitrary shapes and identifying noise/outliers. |
| **Hierarchical (AHC)** | Connectivity-based | Understanding the taxonomy or relationship between groups (Dendrogram). |
| **GMM** | Distribution-based | When clusters have different sizes and elliptical shapes (probabilistic). |



---

## 2. Dimensionality Reduction
*Goal: Reduce the number of random variables under consideration by obtaining a set of principal variables.*

### Linear Methods
* **PCA (Principal Component Analysis)**: Rotates data to capture the directions of maximum variance. Essential for noise reduction and speeding up other ML algos.

### Non-Linear Methods (Manifold Learning)
* **t-SNE**: Specialized in visualizing high-dimensional data in 2D or 3D. Great for seeing local clusters.
* **UMAP**: Faster than t-SNE while preserving more of the global data structure.



---

## 3. Anomaly Detection
*Goal: Identify rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.*

* **Isolation Forest**: Isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are easier to isolate (shorter paths in the tree).
* **Local Outlier Factor (LOF)**: Measures the local density deviation of a given data point with respect to its neighbors.
* **One-Class SVM**: Learns a decision boundary that encloses the majority of the data points.

---

## 4. Association Rule Learning
*Goal: Discover interesting relations between variables in large databases.*

* **Apriori Algorithm**: The classic "Market Basket Analysis" tool. It identifies frequent individual items and extends them to larger itemsets.
* **Eclat**: A more efficient version of Apriori that uses a depth-first search approach on a vertical data format.

---

## Summary Table for Your Site

| Project Type | Recommended Algorithm |
| :--- | :--- |
| **Customer Segmentation** | K-Means / GMM |
| **Fraud Detection** | Isolation Forest |
| **Visualizing Genomics** | t-SNE / UMAP |
| **Product Recommendation** | Apriori |


1. Le Clustering (Regroupement)
C'est la catégorie la plus connue. L'objectif est de diviser les données en groupes homogènes (clusters).

Partitionnement (Centroïdes) :

K-Means (Celui que tu codes !) : Basé sur la distance à un centre.

K-Medoids : Plus robuste aux valeurs aberrantes que K-Means.

Approches Hiérarchiques :

CAH (Clustering Ascendant Hiérarchique) : Crée un dendrogramme (arbre de décision) pour visualiser les fusions de groupes.

Approches basées sur la Densité :

DBSCAN : Idéal pour trouver des formes complexes et identifier le "bruit" (points isolés).

OPTICS : Une amélioration de DBSCAN pour les densités variables.

Modèles de Distribution :

GMM (Gaussian Mixture Models) : Utilise des probabilités (courbes de Gauss) pour définir l'appartenance à un groupe.

2. Réduction de Dimension (Dimensionality Reduction)
L'objectif est de réduire le nombre de variables (colonnes) tout en gardant l'essentiel de l'information. Crucial pour la visualisation 2D/3D.

Linéaire :

PCA (Principal Component Analysis) : Projette les données sur les axes de plus grande variance.

LDA (Linear Discriminant Analysis) : Souvent utilisé en pré-traitement.

Non-linéaire (Manifold Learning) :

t-SNE : Excellent pour visualiser des clusters complexes en 2D.

UMAP : Plus rapide que t-SNE et préserve mieux la structure globale des données.

Compression :

Autoencoders : Réseaux de neurones qui apprennent à compresser puis décompresser la donnée.

3. Détection d'Anomalies (Anomaly Detection)
Identifier les points qui ne ressemblent pas aux autres (fraude, erreur capteur, etc.).

Isolation Forest : Basé sur des arbres de décision (isole les points atypiques plus rapidement).

Local Outlier Factor (LOF) : Compare la densité locale d'un point par rapport à ses voisins.

One-Class SVM : Apprend la "frontière" des données normales.

4. Règles d'Association (Association Rules)
Découvrir des relations entre des variables dans de grandes bases de données.

Apriori : Utilisé pour l'analyse du panier de la ménagère (ex: "ceux qui achètent des bières achètent aussi des chips").

Eclat : Une version plus rapide de l'algorithme Apriori.