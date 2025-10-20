
library(Rcpp)
library(microbenchmark)
#remove.packages("DirichletForestParallel")
#devtools::install_github("https://github.com/Xaleed/DirichletForestParallel.git")
library(DirichletForestParallel)

# Source your local code
sourceCpp("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/src/dirichlet_forest.cpp")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/dirichlet_forest.R")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/parallel_utils.R")

# Generate data
set.seed(123)
n <- 200
p <- 4
X <- matrix(rnorm(n * p), n, p)
alpha <- c(2, 3, 4)
Y <- MCMCpack::rdirichlet(n, alpha)
X_test <- matrix(rnorm(3 * p), 3, p)
Y_test <- MCMCpack::rdirichlet(3, alpha)
forest_dist <- DirichletForest_distributed(X, Y, B = 20, m_try = 4,seed = 999, n_cores = 2,method = "mom" , store_samples = TRUE)
pred <- predict_distributed_forest(forest_dist, X_test)

weights <- get_sample_weights_distributed(forest_dist, X_test[1,])

cleanup_distributed_forest(forest_dist)  # Only call this when completely done
# Benchmark
results <- microbenchmark(
  Local_1 = {
    f <- DirichletForest_distributed(X, Y, B = 500, seed = 999, method = "mom", store_samples = FALSE, n_cores = 3)
    p <- predict_distributed_forest(f, X_test, method = "mom")
  },
  Local_2 = {
    f <- DirichletForest_distributed(X, Y, B = 500, seed = 999,  method = "mom", store_samples = TRUE, n_cores = 3)
    p <- predict_distributed_forest(f, X_test, method = "mom")
  },
  Package_Par = {
    f <- DirichletForestParallel::DirichletForest_distributed(X, Y, B = 500, seed = 999,  method = "mom", n_cores = 3)
    p <- DirichletForestParallel::predict_distributed_forest(f, X_test)
    DirichletForestParallel::cleanup_distributed_forest(f)
  },
  times = 4
)

print(results)
# Fast mode (pre-computed predictions)
forest_fast <- DirichletForest_distributed(X, Y, B = 100, store_samples = FALSE)
pred <- predict_distributed_forest(forest_fast, X_test)
# Distributional mode (weight-based predictions)
forest_dist <- DirichletForest_distributed(X, Y, B = 20, m_try = 4,seed = 999, n_cores = 4,method = "mom" , store_samples = TRUE)
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




# Compare Direct C++ calls, Sequential R wrapper, and Parallel R wrapper for Dirichlet Forests
library(Rcpp)
sourceCpp("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/src/dirichlet_forest.cpp")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/dirichlet_forest.R")
source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/parallel_utils.R")
# Generate data


# Data
set.seed(123)
n <- 500
X <- matrix(rnorm(n * 3), n, 3)
Y <- matrix(runif(n * 3), n, 3)
Y <- Y / rowSums(Y)
X_new <- matrix(rnorm(5 * 3), 5, 3)

# Test 1: Direct C++ vs R wrapper (sequential)
f1 <- DirichletForest(X, Y, B = 10, seed = 999)
f2 <- DirichletForest_distributed(X, Y, B = 10, seed = 999, n_cores = 1)

p1 <- PredictDirichletForest(f1, X_new, method = "mom")
p2 <- predict_distributed_forest(f2, X_new, method = "mom")

cat("Direct C++ vs Sequential wrapper:\n")
print(all.equal(p1$mean_predictions, p2$mean_predictions))

# Test 2: Sequential vs Parallel
f3 <- DirichletForest_distributed(X, Y, B = 20,d_max = 5, n_min = 5, 
                                  m_try = 5, seed = 111, n_cores = 1)
f4 <- DirichletForest_distributed(X, Y, B = 20,d_max = 5, n_min = 5, 
                                  m_try = 5, seed = 111, n_cores = 2)

p3 <- predict_distributed_forest(f3, X_new, method = "mom")
p4 <- predict_distributed_forest(f4, X_new, method = "mom")

cat("\nSequential vs Parallel:\n")
print(all.equal(p3$mean_predictions, p4$mean_predictions, tolerance = 1e-10))
cat("Max difference:", max(abs(p3$mean_predictions - p4$mean_predictions)), "\n")





# ============================================
# Test 1: Direct Rcpp vs Sequential R wrapper
# ============================================
cat("========== TEST 1: Direct vs Sequential ==========\n")

# Direct C++ call
set.seed(456)
forest_cpp <- DirichletForest(X, Y, B = 20, seed = 999, method = "mom")
pred_cpp <- PredictDirichletForest(forest_cpp, X_new, method = "mom")

# R wrapper (sequential, n_cores = 1)
set.seed(456)
forest_seq <- DirichletForest_distributed(X, Y, B = 20, seed = 999, 
                                          method = "mom", n_cores = 1)
pred_seq <- predict_distributed_forest(forest_seq, X_new, method = "mom")

cat("\nDirect C++ mean predictions (first 3 samples):\n")
print(pred_cpp$mean_predictions[1:3, ])

cat("\nSequential R wrapper mean predictions (first 3 samples):\n")
print(pred_seq$mean_predictions[1:3, ])

# ============================================
# Test 2: Sequential vs Parallel (same total trees)
# ============================================
cat("\n========== TEST 2: Sequential vs Parallel ==========\n")

# Sequential
set.seed(789)
forest_seq2 <- DirichletForest_distributed(X, Y, B = 40,d_max = 5, n_min = 5, 
                                           m_try = 2, seed = 111, 
                                           method = "mom", n_cores = 1)
pred_seq2 <- predict_distributed_forest(forest_seq2, X_new, method = "mom")

# Parallel (2 cores, 20 trees each)
set.seed(789)
forest_par <- DirichletForest_distributed(X, Y, B = 40,d_max = 5, n_min = 5, 
                                          m_try = 2, seed = 111, 
                                          method = "mom", n_cores = 2)
pred_par <- predict_distributed_forest(forest_par, X_new, method = "mom")

cat("\nSequential mean predictions (first 3 samples):\n")
print(pred_seq2$mean_predictions[1:3, ])

cat("\nParallel mean predictions (first 3 samples):\n")
print(pred_par$mean_predictions[1:3, ])

cat("\nAre they close? (tolerance 1e-10)\n")
print(all.equal(pred_seq2$mean_predictions, pred_par$mean_predictions, 
                tolerance = 1e-10))

cat("\nMax absolute difference:", 
    max(abs(pred_seq2$mean_predictions - pred_par$mean_predictions)), "\n")

# ============================================
# Test 3: Weight-based predictions
# ============================================
cat("\n========== TEST 3: Regular vs Weight-based ==========\n")

forest_test <- DirichletForest(X, Y, B = 20, seed = 222, method = "mom")

pred_regular <- PredictDirichletForest(forest_test, X_new, method = "mom")
pred_weighted <- PredictDirichletForestWeightBased(forest_test, X_new, method = "mom")

cat("\nRegular prediction (first sample):\n")
print(pred_regular$mean_predictions[1, ])

cat("\nWeight-based prediction (first sample):\n")
print(pred_weighted$mean_predictions[1, ])

cat("\nAre they close? (tolerance 0.01)\n")
print(all.equal(pred_regular$mean_predictions, pred_weighted$mean_predictions, 
                tolerance = 0.01))

cat("\nMax absolute difference:", 
    max(abs(pred_regular$mean_predictions - pred_weighted$mean_predictions)), "\n")

# ============================================
# Test 4: Verify weights work correctly
# ============================================
cat("\n========== TEST 4: Manual Weight Verification ==========\n")

# Get weights for first test sample
weights <- get_sample_weights(forest_test, X_new[1,])

# Calculate weighted mean manually
manual_pred <- colSums(weights$Y_values * weights$weights)

# Compare with weight-based prediction
auto_pred <- pred_weighted$mean_predictions[1, ]

cat("\nManual weighted calculation:\n")
print(manual_pred)

cat("\nAutomatic weight-based prediction:\n")
print(auto_pred)

cat("\nAre they identical?\n")
print(all.equal(manual_pred, auto_pred))

cat("\nDifference:", max(abs(manual_pred - auto_pred)), "\n")

# ============================================
# Summary
# ============================================
cat("\n========== SUMMARY ==========\n")
cat("✓ Direct C++ and Sequential wrapper should be IDENTICAL\n")
cat("✓ Sequential and Parallel should be VERY CLOSE (same seed, same logic)\n")
cat("✓ Regular and Weight-based predictions may differ slightly (different algorithms)\n")
cat("✓ Manual weight calculation should MATCH weight-based prediction exactly\n")

# Generate test data
set.seed(123)
n <- 50
X <- matrix(rnorm(n * 3), n, 3)
Y <- matrix(runif(n * 3), n, 3)
Y <- Y / rowSums(Y)

# Build forest
forest <- DirichletForest(X, Y, B = 10)

# Predict
X_new <- matrix(rnorm(5 * 3), 5, 3)

# Get weights for first test sample
weights <- get_sample_weights(forest, X_new[1,])

# Now you have everything:
cat("Number of training samples with weights:", length(weights$weights), "\n")
cat("Total weight:", sum(weights$weights), "\n\n")

# Show details
df <- data.frame(
  train_index = weights$sample_indices,
  weight = round(weights$weights, 4),
  Y1 = round(weights$Y_values[,1], 3),
  Y2 = round(weights$Y_values[,2], 3),
  Y3 = round(weights$Y_values[,3], 3)
)

cat("Top 10 most important training samples:\n")
print(head(df[order(-df$weight), ], 10))

# Calculate weighted mean manually (verify it matches prediction)
weighted_mean <- colSums(weights$Y_values * weights$weights)
cat("\nWeighted mean prediction:\n")
print(weighted_mean)
cat("Sum:", sum(weighted_mean), "\n")

# Compare with forest prediction
pred <- PredictDirichletForestWeightBased(forest, matrix(X_new[1,], nrow=1), method="mom")
cat("\nForest prediction:\n")
print(pred$mean_predictions[1,])
cat("Match:", all.equal(weighted_mean, pred$mean_predictions[1,]), "\n")








# ============================================
# Example Usage
# ============================================

# Generate test data
set.seed(123)
n <- 200
p <- 5
k <- 3

X <- matrix(rnorm(n * p), n, p)
Y <- matrix(runif(n * k), n, k)
Y <- Y / rowSums(Y)  # Normalize to sum to 1

# ============================================
# Option 1: Simple sequential forest
# ============================================
cat("Building simple forest...\n")
forest_simple <- DirichletForest(X, Y, B = 20, method = "mom")

# Predict with simple forest
X_new <- matrix(rnorm(10 * p), 10, p)
pred_simple <- PredictDirichletForest(forest_simple, X_new, method = "mom")

cat("\nSimple predictions:\n")
print(head(pred_simple$mean_predictions))
print(rowSums(pred_simple$mean_predictions))

# Get sample weights for first test sample
weights <- get_sample_weights(forest_simple, X_new[1,])
cat("\nNumber of training samples with weights:", length(weights$weights), "\n")
cat("Total weight:", sum(weights$weights), "\n")

# ============================================
# Option 2: Distributed forest (parallel)
# ============================================
cat("\n\nBuilding distributed forest...\n")
forest_dist <- DirichletForest_distributed(X, Y, B = 50, n_cores = 2, method = "mom")

# Predict with distributed forest
pred_dist <- predict_distributed_forest(forest_dist, X_new, method = "mom")

cat("\nDistributed predictions:\n")
print(head(pred_dist$mean_predictions))
print(rowSums(pred_dist$mean_predictions))

# Clean up
cleanup_distributed_forest(forest_dist)

# ============================================
# Option 3: Weight-based predictions
# ============================================
cat("\n\nTesting weight-based predictions...\n")
forest_simple2 <- DirichletForest(X, Y, B = 20, method = "mom")

pred_weighted <- PredictDirichletForestWeightBased(forest_simple2, X_new, method = "mom")

cat("\nWeight-based predictions:\n")
print(head(pred_weighted$mean_predictions))
print(rowSums(pred_weighted$mean_predictions))

cat("\nAlpha predictions (normalized):\n")
print(pred_weighted$alpha_predictions / rowSums(pred_weighted$alpha_predictions))



library(DirichletForestParallel)

# Generate tiny test data
set.seed(123)
n <- 500
X <- matrix(rnorm(n * 3), n, 3)
Y <- matrix(runif(n * 3), n, 3)
Y <- Y / rowSums(Y)  # Normalize to sum to 1
X_new <- matrix(rnorm(5 * 3), 5, 3)

# Build forest
forest <- DirichletForest_distributed(X, Y, B = 10,m_try = 3, n_cores =1)

# Predict
pred <- predict_distributed_forest(forest, X_new)

# Check results
print(pred$mean_predictions)
print(rowSums(pred$mean_predictions))  # Should be close to 1

pred_weighted <- predict_distributed_forest_weighted(forest, X_new)
print(pred_weighted$mean_predictions)
print(pred_weighted$alpha_predictions/rowSums(pred_weighted$alpha_predictions))

print(rowSums(pred_weighted$mean_predictions)) 

# Extract the actual forest from the distributed wrapper
actual_forest <- forest_simple$forest

# Now get weights
weights <- get_sample_weights(actual_forest, X_new)

print(paste("Number of samples with weights:", length(weights$weights)))
print(paste("Total weight:", sum(weights$weights)))
print(head(data.frame(index = weights$sample_indices, weight = weights$weights)))
