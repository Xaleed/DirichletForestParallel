#' Build Distributed Dirichlet Forest

#'

#' Builds a Dirichlet regression forest using parallel processing when available.

#' Supports both fork-based (Unix/Mac) and cluster-based (Windows) parallelization.

#'

#' @param X Numeric matrix of predictors (n x p)

#' @param Y Numeric matrix of compositional response variables (n x k), 

#'        each row should sum to 1

#' @param B Integer, number of trees in the forest (default: 100)

#' @param d_max Integer, maximum depth of trees (default: 10)

#' @param n_min Integer, minimum samples per leaf node (default: 5)

#' @param m_try Integer, number of features to try at each split. 

#'        If -1, uses sqrt(p) (default: -1)

#' @param seed Integer, random seed for reproducibility (default: 123)

#' @param method Character, parameter estimation method: "mle" or "mom" (default: "mom")

#' @param store_samples Logical, if TRUE stores sample indices for weight-based predictions,

#'        if FALSE pre-computes predictions (default: FALSE)

#' @param n_cores Integer, number of cores to use. If -1, uses all available cores minus 1.

#'        If 1, uses sequential processing (default: -1)
#' @param use_leaf_predictions Logical, If TRUE, uses pre-computed leaf predictions
#'        for fitted values even when store_samples = TRUE. If FALSE, 
#'        gathers all related training samples across all trees and estimates parameters
#'        from this pooled set.
#'        This affects the fitted values and residuals returned by the function (default: TRUE)
    
#'

#' @return A list containing the distributed forest model with fitted values and residuals

#' @examples
#' \donttest{
#' # Generate sample data
#' library(DirichletForestParallel)
#' n <- 500
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' 
#' # Generate Dirichlet responses
#' if (!requireNamespace("MCMCpack", quietly = TRUE)) {
#'   install.packages("MCMCpack")
#' }
#' alpha <- c(2, 3, 4)
#' Y <- MCMCpack::rdirichlet(n, alpha)
#' 
#' # ========================================
#' # FITTING MODELS
#' # ========================================
#' 
#' # Example 1: Basic distributed forest with parallel processing
#' df_par <- DirichletForest_distributed(X, Y, B = 50, n_cores = 3)
#' 
#' # Example 2: pre-computed predictions, no sample storage
#' df <- DirichletForest_distributed(X, Y, B = 100, store_samples = FALSE)
#' 
#' # Example 3: Weight-based mode (stores samples for distributional analysis)
#' df_weights <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE)
#' 
#' # Example 4: Using MLE instead of method of moments
#' df_mle <- DirichletForest_distributed(X, Y, B = 50, method = "mle")
#' 
#' # Example 5: Sequential processing (no parallelization)
#' df_seq <- DirichletForest_distributed(X, Y, B = 50, n_cores = 1)
#' 
#' # ========================================
#' # ACCESSING FITTED VALUES AND RESIDUALS
#' # ========================================
#' 
#' # Three types of fitted values
#' alpha_hat <- df_par$fitted$alpha_hat        # Estimated Dirichlet parameters
#' mean_fit <- df_par$fitted$mean_based        # Predictions from sample means
#' param_fit <- df_par$fitted$param_based      # Predictions from normalized parameters
#' 
#' # Compare predictions
#' print("Mean-based vs Parameter-based predictions:")
#' print(head(cbind(mean_fit, param_fit)))
#' 
#' # Corresponding residuals
#' resid_mean <- df_par$residuals$mean_based
#' resid_param <- df_par$residuals$param_based
#' 
#' # Compare residual performance
#' rmse_mean <- sqrt(mean(resid_mean^2))
#' rmse_param <- sqrt(mean(resid_param^2))
#' print(paste("RMSE (mean-based):", round(rmse_mean, 4)))
#' print(paste("RMSE (param-based):", round(rmse_param, 4)))
#' 
#' # ========================================
#' # MAKING PREDICTIONS
#' # ========================================
#' 
#' # Create test data
#' X_test <- matrix(rnorm(10 * p), 10, p)
#' 
#' # Make predictions
#' pred <- predict_distributed_forest(df_par, X_test)
#' 
#' # Access different prediction types
#' print(pred$mean_predictions)      # Direct mean-based predictions
#' print(pred$alpha_predictions)     # Estimated Dirichlet parameters
#' 
#' # Parameter-based predictions (normalized alphas)
#' param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' print(param_pred)
#' 
#' # Predict on single observation
#' single_pred <- predict_distributed_forest(df_par, X_test[1, , drop = FALSE])
#' 
#' # ========================================
#' # ANALYZING SAMPLE WEIGHTS
#' # ========================================
#' 
#' # Get weights for a single test sample (requires store_samples = TRUE)
#' test_point <- X_test[1, ]
#' weights <- get_sample_weights_distributed(df_weights, test_point)
#' 
#' # Examine results
#' print(weights$sample_indices)  # Which training samples influenced prediction
#' print(weights$weights)         # How much weight each sample received
#' print(weights$Y_values)        # Compositional values of weighted samples
#' 
#' # Find most influential training samples
#' top_5 <- order(weights$weights, decreasing = TRUE)[1:5]
#' print("Top 5 most influential training samples:")
#' print(weights$sample_indices[top_5])
#' print("Their weights:")
#' print(weights$weights[top_5])
#' 
#' # Verify weights sum to 1
#' print(sum(weights$weights))  # Should be 1.0
#' 
#' # Compare with actual prediction
#' pred_single <- predict_distributed_forest(df_weights, matrix(test_point, nrow = 1))
#' print("Predicted composition:")
#' print(pred_single$mean_predictions)
#' 
#' # Manual weighted prediction (should match pred_single$mean_predictions)
#' manual_pred <- colSums(weights$weights * weights$Y_values)
#' print("Manual weighted average:")
#' print(manual_pred)
#' 
#' # ========================================
#' # CLEANUP
#' # ========================================
#' 
#' # Always clean up at the end, especially important on Windows
#' cleanup_distributed_forest(df_par)
#' cleanup_distributed_forest(df)
#' cleanup_distributed_forest(df_weights)
#' cleanup_distributed_forest(df_mle)
#' cleanup_distributed_forest(df_seq)
#' }
#'
#' @export

DirichletForest_distributed <- function(X, Y, B = 100, d_max = 10, n_min = 5, 
                                        m_try = -1, seed = 123, method = "mom",
                                        store_samples = FALSE, n_cores = -1,
                                        use_leaf_predictions = TRUE) {

  # Input validation

  if (!is.matrix(X) || !is.matrix(Y)) {

    stop("X and Y must be matrices")

  }


  if (nrow(X) != nrow(Y)) {

    stop("X and Y must have the same number of rows")

  }

  

  # Handle parallel package dependency

  if (n_cores != 1) {

    if (!requireNamespace("parallel", quietly = TRUE)) {

      stop("Package 'parallel' is required for distributed computing but not available.\n",
           "Please install it with: install.packages('parallel')\n",
           "Or set n_cores = 1 to use sequential processing.")
    }
  }

  # Force sequential if n_cores = 1

  if (n_cores == 1) {
    forest_seq <- DirichletForest(X, Y, B, d_max, n_min, m_try, seed, method, store_samples)
    result <- list(

      type = "sequential",

      forest = forest_seq,

      n_cores = 1,

      trees_per_worker = B,

      store_samples = store_samples

    )

    

    # Compute fitted values and residuals

    cat("Computing fitted values and residuals...\n")

    fitted_preds <- predict_distributed_forest(result, X, method = method, 
                                           use_leaf_predictions = use_leaf_predictions)
    alpha_means <- fitted_preds$alpha_predictions / 

                   rowSums(fitted_preds$alpha_predictions)
    result$fitted <- list(
      alpha_hat = fitted_preds$alpha_predictions,      # Estimated parameters
      mean_based = fitted_preds$mean_predictions,      # Mean-based predictions
      param_based = alpha_means                         # Parameter-based predictions
    )
    result$residuals <- list(
      mean_based = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )
    class(result) <- c("dirichlet_forest", "list")

    return(result)

  }

  

  # Determine cores for parallel processing

  if (n_cores == -1) {

    n_cores <- max(1, parallel::detectCores() - 1)

  }
  n_cores <- max(1, min(n_cores, B))
  # For small forests, use sequential
  if (B < max(4, n_cores)) {
    forest_seq <- DirichletForest(X, Y, B, d_max, n_min, m_try, seed, method, store_samples)

    result <- list(

      type = "sequential", 

      forest = forest_seq,

      n_cores = 1,

      trees_per_worker = B,

      store_samples = store_samples

    )

    

    # Compute fitted values and residuals

    cat("Computing fitted values and residuals...\n")

    fitted_preds <- predict_distributed_forest(result, X, method = method, 
                                           use_leaf_predictions = use_leaf_predictions)

    

    alpha_means <- fitted_preds$alpha_predictions / 

                   rowSums(fitted_preds$alpha_predictions)

    

    result$fitted <- list(
      alpha_hat = fitted_preds$alpha_predictions,      # Estimated parameters
      mean_based = fitted_preds$mean_predictions,      # Mean-based predictions
      param_based = alpha_means                         # Parameter-based predictions
    )

    result$residuals <- list(
      mean_based = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )

    class(result) <- c("dirichlet_forest", "list")

    return(result)

  }

  

  cat("Building distributed forest with", n_cores, "workers for", B, "trees\n")

  cat("Store samples mode:", store_samples, "\n")

  

  # Distribute trees across workers

  trees_per_core <- rep(B %/% n_cores, n_cores)

  remainder <- B %% n_cores

  if (remainder > 0) {

    trees_per_core[1:remainder] <- trees_per_core[1:remainder] + 1

  }

  

  # Create seeds for each worker

  worker_seeds <- seq(seed, seed + n_cores * 99991, length.out = n_cores)

  cat("Tree distribution:", paste(trees_per_core, collapse = ", "), "\n")

  if (.Platform$OS.type != "windows") {
    # Unix/Mac: fork-based

    cat("Using fork-based parallelization\n")

    worker_forests <- parallel::mclapply(seq_len(n_cores), function(i) {

      DirichletForest(X, Y, B = trees_per_core[i], d_max = d_max,

                      n_min = n_min, m_try = m_try, 

                      seed = worker_seeds[i], method = method,

                      store_samples = store_samples)

    }, mc.cores = n_cores)

    

    result <- list(

      type = "fork",

      worker_forests = worker_forests,

      n_cores = n_cores,

      trees_per_worker = trees_per_core,

      total_trees = sum(trees_per_core),

      store_samples = store_samples

    )

    # Compute fitted values and residuals
    cat("Computing fitted values and residuals...\n")
    fitted_preds <- predict_distributed_forest(result, X, method = method, 
                                           use_leaf_predictions = use_leaf_predictions)

    alpha_means <- fitted_preds$alpha_predictions / 
                   rowSums(fitted_preds$alpha_predictions)

    result$fitted <- list(
      alpha_hat = fitted_preds$alpha_predictions,      # Estimated parameters
      mean_based = fitted_preds$mean_predictions,      # Mean-based predictions
      param_based = alpha_means                         # Parameter-based predictions
    )

    result$residuals <- list(
      mean_based = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )


    class(result) <- c("dirichlet_forest", "list")

    return(result)

    

  } else {

    # Windows: cluster-based - keep workers alive for predictions

    cat("Using persistent cluster (Windows)\n")

    

    cl <- parallel::makeCluster(n_cores, type = "PSOCK")

    

    # Setup workers with Rcpp functions
    # for package:
    setup_cluster_workers_installed(cl)
    # for test:
    #setup_cluster_workers(cl)

    

    # Export variables to workers

    parallel::clusterExport(cl, c("X", "Y", "d_max", "n_min", "m_try", "method", 

                                "trees_per_core", "worker_seeds", "store_samples"), 

                          envir = environment())

    

    # Build forests in each worker

    parallel::clusterApply(cl, seq_len(n_cores), function(worker_id) {

      # Build and store forest in worker's environment

      worker_forest <- DirichletForest(X, Y, B = trees_per_core[worker_id],

                                    d_max = d_max, n_min = n_min, m_try = m_try,

                                    seed = worker_seeds[worker_id], method = method,

                                    store_samples = store_samples)

      # Store forest and training data in worker's global environment

      assign("worker_forest", worker_forest, envir = .GlobalEnv)

      assign("Y_train", Y, envir = .GlobalEnv)

      return(worker_forest$n_trees)

    })

    

    result <- list(

      type = "cluster",

      cluster = cl,

      n_cores = n_cores, 

      trees_per_worker = trees_per_core,

      total_trees = sum(trees_per_core),

      store_samples = store_samples,

      Y_train = Y

    )

    

    # Compute fitted values and residuals

    cat("Computing fitted values and residuals...\n")

    fitted_preds <- predict_distributed_forest(result, X, method = method, 
                                           use_leaf_predictions = use_leaf_predictions)

    

    alpha_means <- fitted_preds$alpha_predictions / 

                   rowSums(fitted_preds$alpha_predictions)

    

    result$fitted <- list(
      alpha_hat = fitted_preds$alpha_predictions,      # Estimated parameters
      mean_based = fitted_preds$mean_predictions,      # Mean-based predictions
      param_based = alpha_means                         # Parameter-based predictions
    )

    result$residuals <- list(
      mean_based = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
)


    class(result) <- c("dirichlet_forest", "list")

    return(result)

  }

}


#' Custom Print Method for dirichlet_forest objects
#'
#' Suppresses the display of large data matrices (Y_train, fitted, residuals)
#' when the object is printed, while keeping them accessible via $.
#'
#' @param x A dirichlet_forest object
#' @param ... Further arguments passed to or from other methods
#' @export
print.dirichlet_forest <- function(x, ...) {
  
  cat("============================================\n")
  cat(" Dirichlet Forest Model (Distributed)\n")
  cat("============================================\n")
  
  # 1. Print Model Metadata
  cat(" Type:", x$type, "\n")
  cat(" Total Trees:", x$total_trees, "\n")
  cat(" Cores Used:", x$n_cores, "\n")
  cat(" Store Samples Mode:", x$store_samples, "\n")
  
  # 2. Print Specific Worker Details
  if (length(x$trees_per_worker) > 1 && x$n_cores > 1) {
    cat(" Trees per Worker:", paste(x$trees_per_worker, collapse = ", "), "\n")
  }

  # 3. Print Cluster/Worker Status
  if (x$type == "cluster" && !is.null(x$cluster)) {
    cat(" Cluster Status: Active (", length(x$cluster), " workers)\n", sep="")
  } else if (x$type == "fork") {
    cat(" Worker Status: Forked (", x$n_cores, " workers)\n", sep="")
  } else if (x$type == "sequential") {
    cat(" Worker Status: Sequential\n")
  }
  
  # 4. Print Data Dimensions
  N <- NULL
  K <- NULL
  if (!is.null(x$Y_train)) {
    N <- nrow(x$Y_train)
    K <- ncol(x$Y_train)
  } else if (x$type == "sequential" && !is.null(x$forest$Y_train)) {
    N <- nrow(x$forest$Y_train)
    K <- ncol(x$forest$Y_train)
  }
  
  if (!is.null(N) && !is.null(K)) {
    cat(paste0(" Training Data Size: ", N, " observations (n) x ", K, " components (k)\n"))
  }


  cat("--------------------------------------------\n")
  cat(" Note: Large data structures (fitted values, residuals) are suppressed.\n")
  cat("\n Access data via the following components:\n")
  cat(" - Training Data (Y):                   $Y_train\n")
  cat(" - Fitted Values:\n")
  cat("   - Estimated Alpha Parameters:        $fitted$alpha_hat\n")
  cat("   - Mean-based Predictions:            $fitted$mean_based\n")
  cat("   - Parameter-based Predictions:       $fitted$param_based\n")
  cat(" - Residuals:\n")
  cat("   - Mean-based Residuals:              $residuals$mean_based\n")
  cat("   - Parameter-based Residuals:         $residuals$param_based\n")
  cat("============================================\n")

  invisible(x)
}

#' Clean Up Distributed Forest

#'

#' Properly cleans up resources used by distributed forest.

#'

#' @param distributed_forest A distributed forest object

#'
#' @examples
#' \donttest{
#' # Setup
#' X <- matrix(rnorm(100 * 4), 100, 4)
#' Y <- MCMCpack::rdirichlet(100, c(2, 3, 4))
#' 
#' # Build forest
#' df <- DirichletForest_distributed(X, Y, B = 50)
#' 
#' # Use the forest...
#' pred <- predict_distributed_forest(df, X)
#' 
#' # Always clean up at the end, especially on Windows
#' cleanup_distributed_forest(df)
#' 
#' # After cleanup, cluster is no longer available
#' # This would fail: predict_distributed_forest(df, X)
#' }
#' @export

cleanup_distributed_forest <- function(distributed_forest) {

  if (distributed_forest$type == "cluster" && !is.null(distributed_forest$cluster)) {

    cat("Stopping cluster workers\n")

    parallel::stopCluster(distributed_forest$cluster)

    distributed_forest$cluster <- NULL

  }

}

#' Predict with Distributed Dirichlet Forest
#'
#' Makes predictions using a distributed Dirichlet forest model.
#' Automatically uses the appropriate prediction mode based on store_samples setting.
#'
#' @param distributed_forest A distributed forest object
#' @param X_new Numeric matrix of new predictors
#' @param method Character, parameter estimation method: "mle" or "mom" (default: "mom")
#' @param use_leaf_predictions Logical, if TRUE uses pre-computed leaf predictions 
#'        even when store_samples = TRUE. If FALSE (default), uses weight-based 
#'        predictions when store_samples = TRUE (default: TRUE)
#'
#' @return A list with alpha_predictions and mean_predictions
#'
#' @examples
#' \donttest{
#' # Setup
#' n <- 500
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
#' 
#' # Fit model with store_samples = TRUE
#' df <- DirichletForest_distributed(X, Y, B = 50, store_samples = TRUE)
#' 
#' # Create test data
#' X_test <- matrix(rnorm(10 * p), 10, p)
#' 
#' # Option 1: Weight-based predictions (default when store_samples = TRUE)
#' pred_weights <- predict_distributed_forest(df, X_test)
#' 
#' # Option 2: uses pre-computed values
#' pred <- predict_distributed_forest(df, X_test, use_leaf_predictions = TRUE)
#' 
#' # Compare the two approaches
#' print("Weight-based mean predictions:")
#' print(head(pred_weights$mean_predictions))
#' print("leaf mean predictions:")
#' print(head(pred$mean_predictions))
#' 
#' # Clean up
#' cleanup_distributed_forest(df)
#' }
#' @export
predict_distributed_forest <- function(distributed_forest, X_new, method = "mom",
                                       use_leaf_predictions = TRUE) {
  
  # Input validation and coercion
  if (!is.matrix(X_new)) {
    if (is.data.frame(X_new)) {
      X_new <- as.matrix(X_new)
    } else if (is.vector(X_new) || is.numeric(X_new)) {
      X_new <- matrix(X_new, nrow = 1)
      warning("Input was a vector. Converting to 1-row matrix. ",
              "Consider using X_new[i, , drop = FALSE] when subsetting matrices.")
    } else {
      stop("X_new must be a matrix, data frame, or numeric vector")
    }
  }
  
  if (!is.numeric(X_new)) {
    stop("X_new must contain numeric values")
  }
  
  n_samples <- nrow(X_new)
  store_samples <- distributed_forest$store_samples
  
  # Determine prediction mode
  if (!store_samples) {
    pred_mode <- " (pre-computed)"
  } else if (use_leaf_predictions) {
    pred_mode <- " (pre-computed leaf values)"
  } else {
    pred_mode <- "Weight-based (distributional)"
  }
  
  cat("Prediction mode:", pred_mode, "\n")
  
  if (distributed_forest$type == "sequential") {
    return(PredictDirichletForest(distributed_forest$forest, X_new, 
                                   method = method, 
                                   use_leaf_predictions = use_leaf_predictions))
  }
  
  if (distributed_forest$type == "fork") {
    cat("Predicting with", distributed_forest$n_cores, "fork workers\n")
    
    worker_predictions <- parallel::mclapply(seq_len(distributed_forest$n_cores), function(i) {
      worker_forest <- distributed_forest$worker_forests[[i]]
      if (worker_forest$n_trees > 0) {
        pred_result <- PredictDirichletForest(worker_forest, X_new, 
                                              method = method,
                                              use_leaf_predictions = use_leaf_predictions)
        if (is.list(pred_result) && 
            !is.null(pred_result$alpha_predictions) && 
            !is.null(pred_result$mean_predictions)) {
          return(pred_result)
        }
      }
      return(NULL)
    }, mc.cores = distributed_forest$n_cores, mc.preschedule = FALSE)
    
    valid_predictions <- Filter(function(p) {
      !is.null(p) && is.list(p) && !is.null(p$alpha_predictions)
    }, worker_predictions)
    
  } else if (distributed_forest$type == "cluster") {
    cat("Predicting with", distributed_forest$n_cores, "cluster workers\n")
    
    cl <- distributed_forest$cluster
    parallel::clusterExport(cl, c("X_new", "method", "use_leaf_predictions"), 
                           envir = environment())
    
    worker_predictions <- parallel::clusterApply(cl, seq_len(distributed_forest$n_cores), 
      function(worker_id) {
        if (exists("worker_forest", envir = .GlobalEnv)) {
          forest <- get("worker_forest", envir = .GlobalEnv)
          if (forest$n_trees > 0) {
            pred_result <- PredictDirichletForest(forest, X_new, 
                                                  method = method,
                                                  use_leaf_predictions = use_leaf_predictions)
            if (is.list(pred_result) && !is.null(pred_result$alpha_predictions)) {
              return(pred_result)
            }
          }
        }
        return(NULL)
      })
    
    valid_predictions <- Filter(function(p) {
      !is.null(p) && is.list(p) && !is.null(p$alpha_predictions)
    }, worker_predictions)
  }
  
  if (length(valid_predictions) == 0) {
    stop("No valid predictions from workers")
  }
  
  cat("Combining predictions from", length(valid_predictions), "workers\n")
  
  first_pred <- valid_predictions[[1]]
  n_classes <- ncol(first_pred$alpha_predictions)
  
  combined_alpha <- array(0, dim = c(n_samples, n_classes))
  combined_mean <- array(0, dim = c(n_samples, n_classes))
  
  total_trees <- sum(distributed_forest$trees_per_worker[seq_along(valid_predictions)])
  
  for (i in seq_along(valid_predictions)) {
    pred <- valid_predictions[[i]]
    weight <- distributed_forest$trees_per_worker[i] / total_trees
    
    combined_alpha <- combined_alpha + weight * pred$alpha_predictions
    combined_mean <- combined_mean + weight * pred$mean_predictions
  }
  
  return(list(
    alpha_predictions = combined_alpha,
    mean_predictions = combined_mean
  ))
}




#' Get Leaf-Based Predictions from Distributed Forest
#'
#' Extracts pre-computed leaf predictions when store_samples = TRUE.
#' This is a convenience wrapper for predict_distributed_forest with use_leaf_predictions = TRUE.
#'
#' @param distributed_forest A distributed forest object with store_samples = TRUE
#' @param X_new Numeric matrix of new predictors
#'
#' @return A list with alpha_predictions and mean_predictions from leaf nodes
#'
#' @examples
#' \donttest{
#' # Setup
#' n <- 500
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
#' 
#' # Fit model with store_samples = TRUE
#' df <- DirichletForest_distributed(X, Y, B = 50, store_samples = TRUE)
#' 
#' # Create test data
#' X_test <- matrix(rnorm(10 * p), 10, p)
#' 
#' # Get leaf predictions
#' leaf_pred <- get_leaf_predictions_distributed(df, X_test)
#' 
#' # Compare with weight-based predictions
#' weight_pred <- predict_distributed_forest(df, X_test)
#' 
#' print("Difference in mean predictions:")
#' print(head(abs(leaf_pred$mean_predictions - weight_pred$mean_predictions)))
#' 
#' # Clean up
#' cleanup_distributed_forest(df)
#' }
#' @export
get_leaf_predictions_distributed <- function(distributed_forest, X_new) {
  
  if (!is.null(distributed_forest$store_samples) && !distributed_forest$store_samples) {
    message("Note: This function can be used with store_samples = FALSE or TRUE.\n",
            "When store_samples = FALSE, it's equivalent to regular prediction.")
  }
  
  return(predict_distributed_forest(distributed_forest, X_new, 
                                    use_leaf_predictions = TRUE))
}

# ============================================================================
# INTERNAL HELPER FUNCTIONS (NOT EXPORTED)
# ============================================================================

# Internal function: Get sample weights for a single test sample
# Used by get_weight_matrix()
get_sample_weights <- function(forest_model, test_sample) {
  
  # Input validation and coercion
  if (is.matrix(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation (vector or 1-row matrix)")
    }
    test_sample <- as.vector(test_sample)
  } else if (is.data.frame(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation")
    }
    test_sample <- as.numeric(test_sample[1, ])
  }
  
  # Ensure it's a plain numeric vector
  test_sample <- as.vector(test_sample)
  
  if (!is.numeric(test_sample)) {
    stop("test_sample must contain numeric values")
  }
  
  # Get weights from C++ (returns only non-zero weights)
  result <- GetSampleWeights(forest_model, test_sample)
  
  # Convert 0-indexed C++ indices to 1-indexed R
  sparse_indices <- result$sample_indices + 1
  sparse_weights <- result$weights
  
  # Get total number of training samples
  n_train <- nrow(forest_model$Y_train)
  
  # Create full weight vector with zeros
  full_weights <- numeric(n_train)
  full_weights[sparse_indices] <- sparse_weights
  
  # Return all indices in order with full Y_train
  return(list(
    sample_indices = 1:n_train,
    weights = full_weights,
    Y_values = forest_model$Y_train
  ))
}


# Internal function: Get sample weights for a single test sample (distributed)
# Used by get_weight_matrix_distributed()
get_sample_weights_distributed <- function(distributed_forest, test_sample) {
  
  # Input validation and coercion
  if (is.matrix(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation (vector or 1-row matrix)")
    }
    test_sample <- as.vector(test_sample)
  } else if (is.data.frame(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation")
    }
    test_sample <- as.numeric(test_sample[1, ])
  }
  
  # Ensure it's a plain numeric vector
  test_sample <- as.vector(test_sample)
  
  if (!is.numeric(test_sample)) {
    stop("test_sample must contain numeric values")
  }
  
  if (distributed_forest$type == "sequential") {
    return(get_sample_weights(distributed_forest$forest, test_sample))
  }
  
  # For distributed forests, combine weights from all workers
  if (distributed_forest$type == "fork") {
    worker_weights <- parallel::mclapply(seq_len(distributed_forest$n_cores), function(i) {
      worker_forest <- distributed_forest$worker_forests[[i]]
      if (worker_forest$n_trees > 0) {
        return(GetSampleWeights(worker_forest, test_sample))
      }
      return(NULL)
    }, mc.cores = distributed_forest$n_cores)
    
    # Get Y values from the first worker
    Y_train <- distributed_forest$worker_forests[[1]]$Y_train
    
  } else if (distributed_forest$type == "cluster") {
    # Check if cluster is still valid
    cl <- distributed_forest$cluster
    if (!inherits(cl, "cluster")) {
      stop("Cluster is no longer valid. The cluster may have been stopped or disconnected.\n",
           "Please rebuild the forest or ensure the cluster remains active.")
    }
    
    # Export test sample to workers
    parallel::clusterExport(cl, "test_sample", envir = environment())
    
    # Get weights from workers
    worker_weights <- tryCatch({
      parallel::clusterApply(cl, seq_len(distributed_forest$n_cores), 
        function(worker_id) {
          if (exists("worker_forest", envir = .GlobalEnv)) {
            forest <- get("worker_forest", envir = .GlobalEnv)
            if (forest$n_trees > 0) {
              return(GetSampleWeights(forest, test_sample))
            }
          }
          return(NULL)
        })
    }, error = function(e) {
      stop("Failed to get weights from workers.\nError: ", e$message)
    })
    
    # Use Y_train stored in the distributed_forest object
    Y_train <- distributed_forest$Y_train
    if (is.null(Y_train)) {
      stop("Training data not found in the distributed forest object.")
    }
  }
  
  # Combine weights from all workers
  valid_weights <- Filter(Negate(is.null), worker_weights)
  
  if (length(valid_weights) == 0) {
    stop("No valid weights from workers")
  }
  
  # Get total number of training samples
  n_train <- nrow(Y_train)
  
  # Initialize full weight vector with zeros
  full_weights <- numeric(n_train)
  
  # Aggregate weights from all workers
  for (i in seq_along(valid_weights)) {
    w <- valid_weights[[i]]
    r_indices <- w$sample_indices + 1  # Convert to 1-indexed
    weight_scale <- distributed_forest$trees_per_worker[i] / distributed_forest$total_trees
    
    # Add scaled weights to the appropriate indices
    full_weights[r_indices] <- full_weights[r_indices] + (w$weights * weight_scale)
  }
  
  # Normalize (should already sum to 1, but ensure it)
  full_weights <- full_weights / sum(full_weights)
  
  # Return all indices in order
  return(list(
    sample_indices = 1:n_train,
    weights = full_weights,
    Y_values = Y_train
  ))
}


# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

#' Get Weight Matrix for Multiple Test Samples
#'
#' Computes sample weights for multiple test observations at once.
#' Each row of the output matrix contains weights for one test sample,
#' showing how much each training sample contributed to that prediction.
#'
#' @param forest_model A forest model created by \code{\link{DirichletForest}} with store_samples = TRUE
#' @param X_test Numeric matrix of test samples (m x p), where m is number of test samples.
#'        Can also be a single vector or 1-row matrix for a single test sample.
#'
#' @return A list with:
#' \describe{
#'   \item{weight_matrix}{Numeric matrix (m x n) where entry [i,j] is the weight of 
#'         training sample j for test sample i. Each row sums to 1.0}
#'   \item{sample_indices}{Integer vector 1:n (all training sample indices in order)}
#'   \item{Y_values}{Matrix of Y values for ALL training samples (n x k)}
#' }
#'
#' @details
#' The weight matrix shows which training samples influenced each prediction.
#' Most weights will be zero; non-zero weights indicate training samples that
#' fell into the same leaf nodes as the test sample across the forest.
#' 
#' The predictions can be verified using matrix multiplication:
#' \code{predicted_Y = weight_matrix \%*\% Y_values}
#'
#' @examples
#' \donttest{
#' # Setup
#' library(DirichletForestParallel)
#' n <- 100
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
#' X_test <- matrix(rnorm(10 * p), 10, p)
#' 
#' # Build forest with store_samples = TRUE
#' f <- DirichletForest(X, Y, B = 50, store_samples = TRUE)
#' 
#' # Get weight matrix for all test samples
#' weights <- get_weight_matrix(f, X_test)
#' 
#' # Examine structure
#' dim(weights$weight_matrix)  # 10 x 100 (10 test samples, 100 training samples)
#' 
#' # Weights for first test sample
#' cat("Weights for test sample 1:\n")
#' print(head(weights$weight_matrix[1, ]))
#' 
#' # Count non-zero weights per test sample
#' non_zero <- rowSums(weights$weight_matrix > 1e-10)
#' cat("\nNon-zero weights per test sample:\n")
#' print(non_zero)
#' 
#' # Find most influential training samples for test sample 1
#' top_5 <- order(weights$weight_matrix[1, ], decreasing = TRUE)[1:5]
#' cat("\nTop 5 influential training samples for test sample 1:\n")
#' print(data.frame(
#'   train_index = top_5,
#'   weight = round(weights$weight_matrix[1, top_5], 4),
#'   Y_comp1 = round(weights$Y_values[top_5, 1], 3)
#' ))
#' 
#' # Verify predictions match
#' pred <- PredictDirichletForest(f, X_test)
#' manual_pred <- weights$weight_matrix %*% weights$Y_values
#' cat("\nMax prediction difference:", 
#'     max(abs(pred$mean_predictions - manual_pred)), "\n")
#' 
#' # Single test sample also works
#' weights_single <- get_weight_matrix(f, X_test[1, , drop = FALSE])
#' dim(weights_single$weight_matrix)  # 1 x 100
#' }
#' @export
get_weight_matrix <- function(forest_model, X_test) {
  
  # Check if store_samples was enabled
  if (!is.null(forest_model$store_samples) && !forest_model$store_samples) {
    stop("Sample weights are only available when store_samples = TRUE.\n",
         "Please rebuild your forest with store_samples = TRUE.")
  }
  
  # Input validation
  if (!is.matrix(X_test)) {
    if (is.data.frame(X_test)) {
      X_test <- as.matrix(X_test)
    } else if (is.vector(X_test) || is.numeric(X_test)) {
      X_test <- matrix(X_test, nrow = 1)
    } else {
      stop("X_test must be a matrix, data frame, or numeric vector")
    }
  }
  
  if (!is.numeric(X_test)) {
    stop("X_test must contain numeric values")
  }
  
  n_test <- nrow(X_test)
  n_train <- nrow(forest_model$Y_train)
  
  # Initialize weight matrix
  weight_matrix <- matrix(0, nrow = n_test, ncol = n_train)
  
  # Progress indicator for large datasets
  if (n_test > 50) {
    cat("Computing weights for", n_test, "test samples...\n")
  }
  
  # Compute weights for each test sample
  for (i in 1:n_test) {
    if (n_test > 100 && i %% 20 == 0) {
      cat("  Progress:", i, "/", n_test, "\n")
    }
    
    test_sample <- X_test[i, ]
    result <- get_sample_weights(forest_model, test_sample)
    weight_matrix[i, ] <- result$weights
  }
  
  return(list(
    weight_matrix = weight_matrix,
    sample_indices = 1:n_train,
    Y_values = forest_model$Y_train
  ))
}


#' Get Weight Matrix for Distributed Forest
#'
#' Computes sample weights for multiple test observations using a distributed forest.
#' Each row of the output matrix contains weights for one test sample,
#' showing how much each training sample contributed to that prediction.
#'
#' @param distributed_forest A distributed forest object with store_samples = TRUE
#' @param X_test Numeric matrix of test samples (m x p), where m is number of test samples.
#'        Can also be a single vector or 1-row matrix for a single test sample.
#'
#' @return A list with:
#' \describe{
#'   \item{weight_matrix}{Numeric matrix (m x n) where entry [i,j] is the weight of 
#'         training sample j for test sample i. Each row sums to 1.0}
#'   \item{sample_indices}{Integer vector 1:n (all training sample indices in order)}
#'   \item{Y_values}{Matrix of Y values for ALL training samples (n x k)}
#' }
#'
#' @details
#' The weight matrix shows which training samples influenced each prediction.
#' Most weights will be zero; non-zero weights indicate training samples that
#' fell into the same leaf nodes as the test sample across the forest.
#' 
#' The predictions can be verified using matrix multiplication:
#' \code{predicted_Y = weight_matrix \%*\% Y_values}
#'
#' @examples
#' \donttest{
#' # Setup
#' library(DirichletForestParallel)
#' n <- 100
#' p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- MCMCpack::rdirichlet(n, c(2, 3, 4))
#' X_test <- matrix(rnorm(10 * p), 10, p)
#' 
#' # Build distributed forest with store_samples = TRUE
#' df <- DirichletForest_distributed(X, Y, B = 100, store_samples = TRUE, n_cores = 4)
#' 
#' # Get weight matrix for all test samples
#' weights <- get_weight_matrix_distributed(df, X_test)
#' 
#' # Examine structure
#' dim(weights$weight_matrix)  # 10 x 100 (10 test samples, 100 training samples)
#' cat("Weight matrix dimensions:", dim(weights$weight_matrix), "\n")
#' 
#' # Analyze weight sparsity
#' sparsity <- sum(weights$weight_matrix > 1e-10) / length(weights$weight_matrix)
#' cat("Proportion of non-zero weights:", round(sparsity, 3), "\n")
#' 
#' # Find most influential training samples for each test sample
#' for (i in 1:3) {
#'   cat("\n--- Test sample", i, "---\n")
#'   top_5 <- order(weights$weight_matrix[i, ], decreasing = TRUE)[1:5]
#'   print(data.frame(
#'     train_idx = top_5,
#'     weight = round(weights$weight_matrix[i, top_5], 4)
#'   ))
#' }
#' 
#' # Verify predictions match
#' pred <- predict_distributed_forest(df, X_test)
#' manual_pred <- weights$weight_matrix %*% weights$Y_values
#' cat("\nMax prediction difference:", 
#'     max(abs(pred$mean_predictions - manual_pred)), "\n")
#' 
#' # Verify each row sums to 1
#' row_sums <- rowSums(weights$weight_matrix)
#' cat("All row sums equal 1?", all(abs(row_sums - 1) < 1e-10), "\n")
#' 
#' # Analyze which training samples are most influential overall
#' col_sums <- colSums(weights$weight_matrix)
#' most_influential <- order(col_sums, decreasing = TRUE)[1:5]
#' cat("\nMost influential training samples overall:\n")
#' print(data.frame(
#'   train_idx = most_influential,
#'   total_weight = round(col_sums[most_influential], 2)
#' ))
#' 
#' # Clean up
#' cleanup_distributed_forest(df)
#' }
#' @export
get_weight_matrix_distributed <- function(distributed_forest, X_test) {
  
  # Check if store_samples was enabled
  if (!is.null(distributed_forest$store_samples) && !distributed_forest$store_samples) {
    stop("Sample weights are only available when store_samples = TRUE.\n",
         "Please rebuild your forest with store_samples = TRUE.")
  }
  
  # Input validation
  if (!is.matrix(X_test)) {
    if (is.data.frame(X_test)) {
      X_test <- as.matrix(X_test)
    } else if (is.vector(X_test) || is.numeric(X_test)) {
      X_test <- matrix(X_test, nrow = 1)
    } else {
      stop("X_test must be a matrix, data frame, or numeric vector")
    }
  }
  
  if (!is.numeric(X_test)) {
    stop("X_test must contain numeric values")
  }
  
  n_test <- nrow(X_test)
  
  # Get Y_train based on forest type
  if (distributed_forest$type == "sequential") {
    n_train <- nrow(distributed_forest$forest$Y_train)
  } else if (distributed_forest$type == "fork") {
    n_train <- nrow(distributed_forest$worker_forests[[1]]$Y_train)
  } else if (distributed_forest$type == "cluster") {
    n_train <- nrow(distributed_forest$Y_train)
  }
  
  # Initialize weight matrix
  weight_matrix <- matrix(0, nrow = n_test, ncol = n_train)
  
  # Progress indicator for large datasets
  if (n_test > 50) {
    cat("Computing weights for", n_test, "test samples...\n")
  }
  
  # Compute weights for each test sample
  for (i in 1:n_test) {
    if (n_test > 100 && i %% 20 == 0) {
      cat("  Progress:", i, "/", n_test, "\n")
    }
    
    test_sample <- X_test[i, ]
    result <- get_sample_weights_distributed(distributed_forest, test_sample)
    weight_matrix[i, ] <- result$weights
  }
  
  # Get Y_values
  if (distributed_forest$type == "sequential") {
    Y_train <- distributed_forest$forest$Y_train
  } else if (distributed_forest$type == "fork") {
    Y_train <- distributed_forest$worker_forests[[1]]$Y_train
  } else {
    Y_train <- distributed_forest$Y_train
  }
  
  return(list(
    weight_matrix = weight_matrix,
    sample_indices = 1:n_train,
    Y_values = Y_train
  ))
}