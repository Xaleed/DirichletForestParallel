
library(Rcpp)
library(microbenchmark)
#remove.packages("DirichletForestParallel")
#devtools::install_github("https://github.com/Xaleed/DirichletForestParallel.git")
library(DirichletForestParallel)

# Source your local code
#sourceCpp("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/src/dirichlet_forest.cpp")
#source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/dirichlet_forest.R")
#source("C:/Users/29827094/Documents/GitHub/DirichletForestParallel/R/parallel_utils.R")


# Test: Weights should be identical regardless of n_cores (same seed)
set.seed(123)
n <- 200
X <- matrix(rnorm(n * 4), n, 4)
Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
X_test <- matrix(rnorm(3 * 4), 3, 4)

# Sequential
f1 <- DirichletForest_distributed(X, Y, B = 20, m_try = 1, seed = 999, n_cores = 1, 
                                  store_samples = TRUE)
pr <- predict_distributed_forest(f1, X_test[1, , drop = FALSE])
pr$mean_predictions

w1 <- get_sample_weights_distributed(f1,  X_test[1, , drop = FALSE])

# Parallel
f2 <- DirichletForest_distributed(X, Y, B = 20, m_try = 4, seed = 999, n_cores = 3, 
                                  store_samples = TRUE)
w2 <- get_sample_weights_distributed(f2, X_test[1,])

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
f1 <- DirichletForest_distributed(X, Y, B = 20, m_try = 4, seed = 999, n_cores = 1, 
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
    forest = DirichletForest(1)
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

