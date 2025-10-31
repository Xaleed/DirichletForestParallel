
library(Rcpp)
library(microbenchmark)

remove.packages("DirichletForestParallel")

setwd("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/")
devtools::document()
devtools::check()
devtools::install()


# Test everything works

#devtools::install_github("https://github.com/Xaleed/DirichletForestParallel.git")
library(DirichletForestParallel)

# Source your local code
sourceCpp("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/src/dirichlet_forest.cpp")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/dirichlet_forest.R")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/parallel_utils.R")


# Users only see and use the public functions:

# Setup
n <- 15
X <- matrix(rnorm(n * 4), n, 4)
Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
X_test <- matrix(rnorm(10 * 4), 10, 4)

# Build forest
df <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE, n_cores = 3)
df$residuals

# Get weights - works for single or multiple test samples
weights <- get_weight_matrix_distributed(df, X_test)

# Output:
# $weight_matrix: 10 x 100 (all test samples x all training samples)
# $sample_indices: 1:100 (in order)
# $Y_values: 100 x 3 (complete training data)

# Access weights for specific test sample
weights$weight_matrix[1, ]  # Weights for first test sample (all 100 training samples)

# Find influential samples
top_5 <- order(weights$weight_matrix[1, ], decreasing = TRUE)[1:5]
print(data.frame(
  train_idx = top_5,
  weight = weights$weight_matrix[1, top_5]
))

# Verify predictions
pred <- predict_distributed_forest(df, X_test)
manual <- weights$weight_matrix %*% weights$Y_values
max(abs(pred$mean_predictions - manual))  # Should be ~0

cleanup_distributed_forest(df)




# Setup
n <- 100
p <- 4
X <- matrix(rnorm(n * p), n, p)
Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
X_test <- matrix(rnorm(10 * p), 10, p)

# Build distributed forest
df <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE, n_cores = 3)

# ============================================
# OPTION 1: Single test sample (your original use case)
# ============================================
weights_single <- get_sample_weights_distributed(df, X_test[1, ])

# Output structure:
# $sample_indices: [1, 2, 3, ..., 100]
# $weights: [0.333, 0, 0, 0.333, 0, ..., 0.333, 0]  # All samples, many zeros
# $Y_values: Full training matrix (100 x 3)

# ============================================
# OPTION 2: Multiple test samples (NEW!)
# ============================================
weights_matrix <- get_weight_matrix_distributed(df, X_test)
weights_matrix$weight_matrix
# Output structure:
# $weight_matrix: 10 x 100 matrix
#   - Row 1: weights for X_test[1, ]
#   - Row 2: weights for X_test[2, ]
#   - ...
#   - Row 10: weights for X_test[10, ]
# $sample_indices: [1, 2, 3, ..., 100]
# $Y_values: Full training matrix (100 x 3)

# Example: Examine weights for first 3 test samples
print(weights_matrix$weight_matrix[1:3, 1:10])  # First 3 test samples, first 10 training samples

# Verify row sums to 1
rowSums(weights_matrix$weight_matrix)  # Should all be 1.0

# Find most influential training samples for test sample 5
test_idx <- 5
influential <- order(weights_matrix$weight_matrix[test_idx, ], decreasing = TRUE)[1:5]
cat("Most influential training samples for test sample", test_idx, ":\n")
print(data.frame(
  train_index = influential,
  weight = weights_matrix$weight_matrix[test_idx, influential]
))

# Verify predictions match
pred <- predict_distributed_forest(df, X_test)
manual_pred <- weights_matrix$weight_matrix %*% weights_matrix$Y_values
max(abs(pred$mean_predictions - manual_pred))  # Should be ~ 0

# Clean up
cleanup_distributed_forest(df)


















# Test: Weights should be identical regardless of n_cores (same seed)
set.seed(123)
n <- 100
X <- matrix(rnorm(n * 4), n, 4)
Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
X_test <- matrix(rnorm(3 * 4), 3, 4)

# Sequential
f1 <- DirichletForest_distributed(X, Y, B = 50, m_try = 1, seed = 999, n_cores = 1, 
                                  store_samples = TRUE)
pr <- predict_distributed_forest(f1, X_test[1, , drop = FALSE])
pr$mean_predictions

w1 <- get_sample_weights_distributed(f1,  X_test[1, , drop = FALSE])


# 1. Weight-based (distributional) - default behavior
pred1 <- predict_distributed_forest(f1, X_test)

# 2. Pre-computed leaf predictions
pred2 <- predict_distributed_forest(f1, X_test, use_leaf_predictions = FALSE)

# 3. Convenience function for leaf predictions
pred3 <- get_leaf_predictions_distributed(df, X_test)

# Parallel
f2 <- DirichletForest_distributed(X, Y, B = 200, m_try = 2, seed = 123, n_cores = 3, 
                                  store_samples = TRUE)
w2 <- get_sample_weights_distributed(f2, X_test[1, , drop = FALSE])

# Compare
cat("Weight difference:", max(abs(w1$weights - w2$weights)), "\n")
cat("Identical?", all.equal(w1$weights, w2$weights, tolerance = 1e-10), "\n")

cleanup_distributed_forest(f1)
cleanup_distributed_forest(f2)

#compare with julia
set.seed(123)
n <- 200
X <- matrix(rnorm(n * 4), n, 4)
Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
X_test <- matrix(rnorm(3 * 1), 1, 4)

# Sequential
f1 <- DirichletForest_distributed(X, Y, B = 20, m_try = 4, seed = 132, n_cores = 1, 
                                  store_samples = TRUE)
pr <- predict_distributed_forest(f1, X_test[1,])
pr$mean_predictions

cat("\n=== Julia Implementation ===\n")
library(JuliaCall)
julia_source("C:\\Users\\29827094\\Documents\\GitHub\\Dirichlet_RF_clean_code\\Julia\\dirichlet_decision_tree.jl")

time_julia <- system.time({
  # Assign data
  julia_assign("X_train", X)
  julia_assign("Y_train", Y)
  julia_assign("X_test", X_test)
  
  # Train and predict
  julia_eval('begin
    x_tr, y_tr, x_te = process_matrix_data(X_train, Y_train, X_test)
    forest = DirichletForest(20)
    fit_dirichlet_forest!(forest, x_tr, y_tr, 3000, 10, 5,4, estimate_parameters_mom)
    predictions = predict_dirichlet_forest(forest, x_te)
  end')
  
  pred_julia <- julia_eval("predictions")
})

pred_julia
pr$mean_predictions

















# Fast mode (pre-computed predictions)
forest_fast <- DirichletForest_distributed(X, Y, B = 100, store_samples = FALSE)
pred <- predict_distributed_forest(forest_fast, X_test)
# Distributional mode (weight-based predictions)
forest_dist <- DirichletForest_distributed(X, Y, B = 20, m_try = 2,seed = 999, n_cores = 1,method = "mom" , store_samples = TRUE)
pred <- predict_distributed_forest(forest_dist, X_test)
weights <- get_sample_weights_distributed(forest_dist, X_test[1,])
weights1
# Accuracy comparison
f_local <- DirichletForest_distributed(X, Y, B = 20, m_try = 4,seed = 999, n_cores = 5,method = "mom" , store_samples = TRUE)
f_pkg <- DirichletForestParallel::DirichletForest_distributed(X, Y, B = 20, m_try = 4,seed = 999, n_cores = 5)

p_local <- predict_distributed_forest(f_local, X_test, method = "mom")
p_pkg <- DirichletForestParallel::predict_distributed_forest(f_pkg, X_test)

# RMSE on test set
mse <- function(pred, actual) (mean((pred - actual)^2))
cat("\nRMSE Local:",mse(p_local$mean_predictions, Y_test))
cat("\nRMSE Package:", mse(p_pkg$mean_predictions, Y_test))
cat("\nPrediction difference:", max(abs(p_local$mean_predictions - p_pkg$mean_predictions)))

DirichletForestParallel::cleanup_distributed_forest(f_pkg)

