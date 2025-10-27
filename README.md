# DirichletForest_distributed  

This repository contains an implementation of a **parallel Dirichlet Random Forest**, designed for modeling **compositional (Dirichlet-distributed) data**.  

⚠️ **Note**: This project is still in progress. For a simpler and more stable version, see my [DirichletRandom](https://github.com/Xaleed/DirichletForest.git) repository.  




## 📦 Installation  

### Option 1: Clone this repository and install locally in R:  
```r
devtools::install_github("Xaleed/DirichletForestParallel")
```
### Option 2: Install pre-built binary (no Rtools required - Windows only)

Download and install the latest binary release:
```r
# Replace v0.1.0 with the latest release version
install.packages("https://github.com/Xaleed/DirichletForestParallel/releases/download/v0.1.0/DirichletForestParallel_0.0.0.9000.zip", 
                 repos = NULL, type = "win.binary")
```

Or manually:
1. Go to [Releases](https://github.com/Xaleed/DirichletForestParallel/releases)
2. Download the `.zip` file from the latest release
3. In R: `install.packages("path/to/downloaded/file.zip", repos = NULL, type = "win.binary")`

---

## 🚀 Quick Start
```r
library(DirichletForestParallel)

# Generate predictors
n <- 500
p <- 4
X <- matrix(rnorm(n * p), n, p)

# Generate Dirichlet responses
if (!requireNamespace("MCMCpack", quietly = TRUE)) {
  install.packages("MCMCpack")
}
alpha <- c(2, 3, 4)
Y <- MCMCpack::rdirichlet(n, alpha)

# Fit a distributed Dirichlet Forest with 50 trees using 3 cores
df_par <- DirichletForest_distributed(X, Y, B = 50, n_cores = 3)

# Predict on new data
X_test <- matrix(rnorm(10 * p), 10, p)
pred <- predict_distributed_forest(df_par, X_test)

# Access predictions
print(pred$mean_predictions)      # Mean predictions
print(pred$alpha_predictions)     # Dirichlet parameters

# Clean up cluster resources (important for Windows)
cleanup_distributed_forest(df_par)
```

---

## 🔧 Key Features

### **Parallel Processing**
- **Automatic detection**: Uses fork-based parallelization on Unix/Mac and cluster-based on Windows
- **Flexible cores**: Set `n_cores = -1` to use all available cores minus one, or specify exact number
- **Sequential fallback**: Small forests automatically run sequentially for efficiency

### **Two Prediction Modes**

#### default: `store_samples = FALSE`
Pre-computes predictions at training time for faster inference:
```r
df_fast <- DirichletForest_distributed(X, Y, B = 100, store_samples = FALSE)
pred_fast <- predict_distributed_forest(df_fast, X_test)
```

#### Weight-Based Mode (`store_samples = TRUE`)
Stores sample indices for distributional predictions and weight analysis:
```r
df_weights <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE)

# Get prediction weights for a single test sample
weights <- get_sample_weights_distributed(df_weights, X_test[1, ])
print(weights$sample_indices)  # Which training samples influenced prediction
print(weights$weights)         # How much weight each sample received
print(weights$Y_values)        # Compositional values of weighted samples
```

### **Parameter Estimation**
Choose between Method of Moments (`method = "mom"`, default) or Maximum Likelihood Estimation (`method = "mle"`):
```r
df_mle <- DirichletForest_distributed(X, Y, method = "mle")
```

---

## 📊 Example: Understanding Predictions
```r
# Train with weight-based mode
df <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE)

# Get prediction for a test sample
test_point <- X_test[1, ]
weights <- get_sample_weights_distributed(df, test_point)

# Analyze which training samples matter most
top_indices <- weights$sample_indices[order(weights$weights, decreasing = TRUE)[1:5]]
print("Top 5 most influential training samples:")
print(top_indices)

# Make predictions
pred <- predict_distributed_forest(df, matrix(test_point, nrow = 1))
print("Predicted composition:")
print(pred$mean_predictions)

# Cleanup
cleanup_distributed_forest(df)
```

---

## ⚙️ Function Reference

### `DirichletForest_distributed()`
Main function to build a distributed forest.

**Parameters:**
- `X`: Predictor matrix (n × p)
- `Y`: Compositional response matrix (n × k), rows sum to 1
- `B`: Number of trees (default: 100)
- `d_max`: Maximum tree depth (default: 10)
- `n_min`: Minimum samples per leaf (default: 5)
- `m_try`: Features to try at each split, -1 for sqrt(p) (default: -1)
- `seed`: Random seed (default: 123)
- `method`: Parameter estimation, "mom" or "mle" (default: "mom")
- `store_samples`: Enable weight-based predictions (default: FALSE)
- `n_cores`: Number of cores, -1 for auto-detect (default: -1)

### `predict_distributed_forest()`
Make predictions with a trained forest.

**Parameters:**
- `distributed_forest`: Trained forest object
- `X_new`: New predictor matrix
- `method`: Parameter estimation method (default: "mom")

**Returns:** List with `alpha_predictions` and `mean_predictions`

### `get_sample_weights_distributed()`
Get sample weights for a test observation (requires `store_samples = TRUE`).

**Parameters:**
- `distributed_forest`: Trained forest object
- `test_sample`: Single test observation (vector)

**Returns:** List with `sample_indices`, `weights`, and `Y_values`

### `cleanup_distributed_forest()`
Clean up cluster resources (essential on Windows).

---

## 💡 Tips

1. **Windows users**: Always call `cleanup_distributed_forest()` when done to properly close worker processes
2. **Small forests**: For B < 10 trees, sequential processing is automatically used
3. **Memory**: Weight-based mode (`store_samples = TRUE`) uses more memory but enables deeper analysis.

---

## 📝 License

This project is open source and available under standard licensing terms.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
