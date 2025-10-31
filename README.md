# DirichletForest_distributed  

This repository contains an implementation of a **parallel Dirichlet Random Forest**, designed for modeling **compositional (Dirichlet-distributed) data**.  

⚠️ **Note**: This project is still in progress. For a simpler and more stable version, see my [DirichletRandom](https://github.com/Xaleed/DirichletForest.git) repository.  

---

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
print(pred$mean_predictions)      # Mean-based predictions
print(pred$alpha_predictions)     # Estimated Dirichlet parameters

# Access fitted values
print(df_par$fitted$alpha_hat)      # Estimated parameters (α̂)
print(df_par$fitted$mean_based)     # Fitted values from sample means
print(df_par$fitted$param_based)    # Fitted values from normalized parameters

# Access residuals
print(df_par$residuals$mean_based)   # Residuals for mean-based predictions
print(df_par$residuals$param_based)  # Residuals for parameter-based predictions

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

#### Fast Mode: `store_samples = FALSE` (Default)
Pre-computes predictions at training time for faster inference:
```r
df_fast <- DirichletForest_distributed(X, Y, B = 100, store_samples = FALSE)
pred_fast <- predict_distributed_forest(df_fast, X_test)
```

#### Weight-Based Mode: `store_samples = TRUE`
Stores sample indices for distributional predictions and weight analysis:
```r
df_weights <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE)

# Option 1: Use fast leaf predictions (default for fitted values)
pred_fast <- predict_distributed_forest(df_weights, X_test, use_leaf_predictions = TRUE)

# Option 2: Use weight-based predictions for deeper analysis
pred_weights <- predict_distributed_forest(df_weights, X_test, use_leaf_predictions = FALSE)

# Get weight matrix for all test samples
weights_matrix <- get_weight_matrix_distributed(df_weights, X_test)
print(dim(weights_matrix$weight_matrix))  # test_samples x training_samples

# Verify predictions using weight matrix
manual_pred <- weights_matrix$weight_matrix %*% weights_matrix$Y_values
```

### **Parameter Estimation**
Choose between Method of Moments (`method = "mom"`, default) or Maximum Likelihood Estimation (`method = "mle"`):
```r
df_mle <- DirichletForest_distributed(X, Y, method = "mle")
```

### **Multiple Fitted Values and Residuals**
The model provides three types of fitted values:
- **`alpha_hat`**: Estimated Dirichlet concentration parameters (α̂)
- **`mean_based`**: Predictions computed from sample means in terminal nodes
- **`param_based`**: Predictions computed from normalized estimated parameters

```r
# Access different types of fitted values
alpha_estimates <- df_par$fitted$alpha_hat      # Parameter estimates
mean_fitted <- df_par$fitted$mean_based         # Mean-based fitted values
param_fitted <- df_par$fitted$param_based       # Parameter-based fitted values

# Compare residual performance
rmse_mean <- sqrt(mean(df_par$residuals$mean_based^2))
rmse_param <- sqrt(mean(df_par$residuals$param_based^2))
print(paste("RMSE (mean-based):", round(rmse_mean, 4)))
print(paste("RMSE (param-based):", round(rmse_param, 4)))
```

**Note**: When `store_samples = TRUE`, the fitted values are computed using pre-computed leaf predictions by default (`use_leaf_predictions = TRUE`). Set `use_leaf_predictions = FALSE` in the model fitting function to use weight-based predictions for fitted values instead:
```r
# Use weight-based predictions for fitted values
df_weights <- DirichletForest_distributed(X, Y, B = 100, 
                                          store_samples = TRUE, 
                                          use_leaf_predictions = FALSE)
```

---

## 📊 Example: Working with Weight Matrices

```r
# Train with weight-based mode
df <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE)

# Get weight matrix for multiple test samples
X_test <- matrix(rnorm(20 * p), 20, p)
weights <- get_weight_matrix_distributed(df, X_test)

# Examine weight matrix structure
dim(weights$weight_matrix)  # 20 test samples x n training samples
cat("Sparsity:", sum(weights$weight_matrix > 1e-10) / length(weights$weight_matrix), "\n")

# Find most influential training samples for first test sample
top_5_idx <- order(weights$weight_matrix[1, ], decreasing = TRUE)[1:5]
print("Top 5 most influential training samples:")
print(data.frame(
  train_index = top_5_idx,
  weight = round(weights$weight_matrix[1, top_5_idx], 4)
))

# Verify predictions match weight-based computation
pred <- predict_distributed_forest(df, X_test, use_leaf_predictions = FALSE)
manual_pred <- weights$weight_matrix %*% weights$Y_values
cat("Max prediction difference:", max(abs(pred$mean_predictions - manual_pred)), "\n")

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
- `use_leaf_predictions`: If TRUE, uses pre-computed leaf predictions for fitted values even when `store_samples = TRUE` (default: TRUE)

**Returns:** A list containing:
- `fitted`: List with `alpha_hat` (parameter estimates), `mean_based` (mean-based fitted values), `param_based` (parameter-based fitted values)
- `residuals`: List with `mean_based` and `param_based` residuals
- `type`: Forest type ("sequential", "fork", or "cluster")
- Additional components depending on forest type

### `predict_distributed_forest()`
Make predictions with a trained forest.

**Parameters:**
- `distributed_forest`: Trained forest object
- `X_new`: New predictor matrix (or vector for single sample)
- `method`: Parameter estimation method (default: "mom")
- `use_leaf_predictions`: If TRUE, uses pre-computed leaf predictions (faster). If FALSE, uses weight-based predictions (default: TRUE)

**Returns:** List with `alpha_predictions` (estimated Dirichlet parameters) and `mean_predictions` (mean-based predictions)

### `get_weight_matrix_distributed()`
Get weight matrix for multiple test observations (requires `store_samples = TRUE`).

**Parameters:**
- `distributed_forest`: Trained forest object with `store_samples = TRUE`
- `X_test`: Test predictor matrix (m × p) or vector

**Returns:** List with:
- `weight_matrix`: Matrix (m × n) where entry [i,j] is the weight of training sample j for test sample i
- `sample_indices`: Integer vector 1:n (all training indices)
- `Y_values`: Matrix of training Y values (n × k)

### `get_leaf_predictions_distributed()`
Convenience wrapper for getting pre-computed leaf predictions when `store_samples = TRUE`.

**Parameters:**
- `distributed_forest`: Trained forest object
- `X_new`: New predictor matrix

**Returns:** List with `alpha_predictions` and `mean_predictions` from leaf nodes

### `cleanup_distributed_forest()`
Clean up cluster resources (essential on Windows).

---

## 💡 Tips

1. **Windows users**: Always call `cleanup_distributed_forest()` when done to properly close worker processes
2. **Small forests**: For B < 10 trees, sequential processing is automatically used
3. **Memory considerations**: 
   - `store_samples = FALSE`: Most memory efficient, fastest predictions
   - `store_samples = TRUE` with `use_leaf_predictions = TRUE`: Moderate memory, fast predictions, enables weight analysis
   - `store_samples = TRUE` with `use_leaf_predictions = FALSE`: Higher memory usage, slower predictions but more flexible
4. **Fitted values**: When `store_samples = TRUE`, fitted values are computed using pre-computed leaf predictions by default for efficiency. Set `use_leaf_predictions = FALSE` if you need weight-based fitted values
5. **Weight matrices**: Use `get_weight_matrix_distributed()` to analyze which training samples influence predictions most

---

## 🔍 Understanding Prediction Modes

| Mode | `store_samples` | `use_leaf_predictions` | Speed | Memory | Use Case |
|------|----------------|----------------------|-------|--------|----------|
| **Fast** | FALSE | N/A | Fastest | Lowest | Standard predictions |
| **Leaf-based** | TRUE | TRUE | Fast | Medium | Predictions + weight analysis |
| **Weight-based** | TRUE | FALSE | Slower | Higher | Full distributional analysis |

---

## 📄 License

This project is open source and available under standard licensing terms.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.