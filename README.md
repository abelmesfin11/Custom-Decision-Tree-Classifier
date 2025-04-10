# 🌳 Custom-Decision-Tree-Classifier

Tech stack: Python, Pandas, NumPy, Object-Oriented Programming (OOP)

An interpretable, from-scratch implementation of a binary decision tree classifier. Built using entropy minimization and recursive tree construction without relying on external ML libraries. Designed to work on both synthetic and real-world clinical datasets.

✨ Features:

Entropy-Based Splitting: Implemented core decision tree logic using information gain (reduction in entropy) to find optimal splits across all features and thresholds.

Recursive Tree Construction: Leveraged object-oriented principles to build tree nodes (Vertex class), recursively splitting data to form a complete decision tree with customizable depth.

Prediction via Depth-First Search: Developed custom traversal logic to classify samples by walking them through the tree from root to leaf based on learned thresholds.

Model Visualization: Included a print_tree() utility to display the structure of the learned tree for interpretability and debugging.

Hyperparameter Tuning: Performed cross-validation to determine optimal tree depth on real-world biomedical data, significantly improving prediction accuracy.

📊 Datasets:

Simulated Data: Used for initial debugging, feature splitting validation, and accuracy testing.

Bone Marrow Transplant Dataset: Real-world clinical dataset used to predict patient survival based on features like donor/recipient age, cell counts, and disease severity. Custom preprocessing included handling categorical variables, dropping missing values, and mapping disease labels to numerical values.

🧪 Results:

Tuned the maximum depth of the decision tree via cross-validation, achieving strong test accuracy on unseen transplant data.

Designed from the ground up without Scikit-learn, gaining deep understanding of entropy, recursive algorithms, and classification decision boundaries.

