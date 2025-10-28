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
#'        if FALSE pre-computes predictions for faster inference (default: FALSE)
#' @param n_cores Integer, number of cores to use. If -1, uses all available cores minus 1.
#'        If 1, uses sequential processing (default: -1)
#'
#' @return A list containing the distributed forest model with fitted values and residuals
#'
#' @export
DirichletForest_distributed <- function(X, Y, B = 100, d_max = 10, n_min = 5, 
                                        m_try = -1, seed = 123, method = "mom",
                                        store_samples = FALSE, n_cores = -1) {
  
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
    fitted_preds <- predict_distributed_forest(result, X, method = method)
    
    alpha_means <- fitted_preds$alpha_predictions / 
                   rowSums(fitted_preds$alpha_predictions)
    
    result$fitted <- list(
      alpha = fitted_preds$alpha_predictions,
      mean = fitted_preds$mean_predictions,
      alpha_mean = alpha_means
    )
    
    result$residuals <- list(
      mean = Y - fitted_preds$mean_predictions,
      alpha = Y - alpha_means
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
    fitted_preds <- predict_distributed_forest(result, X, method = method)
    
    alpha_means <- fitted_preds$alpha_predictions / 
                   rowSums(fitted_preds$alpha_predictions)
    
    result$fitted <- list(
      alpha = fitted_preds$alpha_predictions,
      mean = fitted_preds$mean_predictions,
      alpha_mean = alpha_means
    )
    
    result$residuals <- list(
      mean = Y - fitted_preds$mean_predictions,
      alpha = Y - alpha_means
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
    fitted_preds <- predict_distributed_forest(result, X, method = method)
    
    alpha_means <- fitted_preds$alpha_predictions / 
                   rowSums(fitted_preds$alpha_predictions)
    
    result$fitted <- list(
      alpha = fitted_preds$alpha_predictions,
      mean = fitted_preds$mean_predictions,
      alpha_mean = alpha_means
    )
    
    result$residuals <- list(
      mean = Y - fitted_preds$mean_predictions,
      alpha = Y - alpha_means
    )
    

    
    class(result) <- c("dirichlet_forest", "list")
    return(result)
    
  } else {
    # Windows: cluster-based - keep workers alive for predictions
    cat("Using persistent cluster (Windows)\n")
    
    cl <- parallel::makeCluster(n_cores, type = "PSOCK")
    
    # Setup workers with Rcpp functions
    setup_cluster_workers(cl)
    
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
    fitted_preds <- predict_distributed_forest(result, X, method = method)
    
    alpha_means <- fitted_preds$alpha_predictions / 
                   rowSums(fitted_preds$alpha_predictions)
    
    result$fitted <- list(
      alpha = fitted_preds$alpha_predictions,
      mean = fitted_preds$mean_predictions,
      alpha_mean = alpha_means
    )
    
    result$residuals <- list(
      mean = Y - fitted_preds$mean_predictions,
      alpha = Y - alpha_means
    )
    

    
    class(result) <- c("dirichlet_forest", "list")
    return(result)
  }
}

#' Clean Up Distributed Forest
#'
#' Properly cleans up resources used by distributed forest.
#'
#' @param distributed_forest A distributed forest object
#'
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
#'
#' @return A list with alpha_predictions and mean_predictions
#'
#' @export
predict_distributed_forest <- function(distributed_forest, X_new, method = "mom") {
  
  # Input validation and coercion
  if (!is.matrix(X_new)) {
    if (is.data.frame(X_new)) {
      X_new <- as.matrix(X_new)
    } else if (is.vector(X_new) || is.numeric(X_new)) {
      # Handle vector input - convert to 1-row matrix
      X_new <- matrix(X_new, nrow = 1)
      warning("Input was a vector. Converting to 1-row matrix. ",
              "Consider using X_new[i, , drop = FALSE] when subsetting matrices.")
    } else {
      stop("X_new must be a matrix, data frame, or numeric vector")
    }
  }
  
  # Ensure it's numeric
  if (!is.numeric(X_new)) {
    stop("X_new must contain numeric values")
  }
  
  n_samples <- nrow(X_new)
  store_samples <- distributed_forest$store_samples
  
  cat("Prediction mode:", ifelse(store_samples, "Weight-based (distributional)", "Fast (pre-computed)"), "\n")
  
  if (distributed_forest$type == "sequential") {
    return(PredictDirichletForest(distributed_forest$forest, X_new, method = method))
  }
  
  if (distributed_forest$type == "fork") {
    cat("Predicting with", distributed_forest$n_cores, "fork workers\n")
    
    worker_predictions <- parallel::mclapply(seq_len(distributed_forest$n_cores), function(i) {
      worker_forest <- distributed_forest$worker_forests[[i]]
      if (worker_forest$n_trees > 0) {
        pred_result <- PredictDirichletForest(worker_forest, X_new, method = method)
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
    parallel::clusterExport(cl, c("X_new", "method"), envir = environment())
    
    worker_predictions <- parallel::clusterApply(cl, seq_len(distributed_forest$n_cores), 
      function(worker_id) {
        if (exists("worker_forest", envir = .GlobalEnv)) {
          forest <- get("worker_forest", envir = .GlobalEnv)
          if (forest$n_trees > 0) {
            pred_result <- PredictDirichletForest(forest, X_new, method = method)
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

#' Get Sample Weights for a Test Sample
#'
#' Computes the weight assigned to each training sample by the DRF algorithm
#' for a given test sample. Useful for understanding model predictions.
#' Only works when store_samples = TRUE.
#'
#' @param forest_model A forest model created by \code{\link{DirichletForest}} with store_samples = TRUE
#' @param test_sample Numeric vector of length p (must match training features)
#'
#' @return A list with:
#' \describe{
#'   \item{sample_indices}{Integer vector of training sample indices (1-indexed for R)}
#'   \item{weights}{Numeric vector of corresponding weights (sum to 1.0)}
#'   \item{Y_values}{Matrix of Y values for the weighted samples (n_weighted x k)}
#' }
#'
#' @export
get_sample_weights <- function(forest_model, test_sample) {
  
  # Check if store_samples was enabled
  if (!is.null(forest_model$store_samples) && !forest_model$store_samples) {
    stop("Sample weights are only available when store_samples = TRUE.\n",
         "Please rebuild your forest with store_samples = TRUE.")
  }
  
  # Input validation and coercion
  if (is.matrix(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation (vector or 1-row matrix)")
    }
    test_sample <- as.vector(test_sample)
    warning("test_sample was a matrix. Converting to vector. ",
            "Consider using test_sample[1, , drop = FALSE] then converting to vector with as.vector().")
  } else if (is.data.frame(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation")
    }
    test_sample <- as.numeric(test_sample[1, ])
    warning("test_sample was a data frame. Converting to numeric vector.")
  }
  
  if (!is.vector(test_sample) && !is.numeric(test_sample)) {
    stop("test_sample must be a numeric vector, 1-row matrix, or single-row data frame")
  }
  
  # Ensure it's a plain numeric vector
  test_sample <- as.vector(test_sample)
  
  if (!is.numeric(test_sample)) {
    stop("test_sample must contain numeric values")
  }
  
  result <- GetSampleWeights(forest_model, test_sample)
  
  # Convert 0-indexed C++ indices to 1-indexed R
  result$sample_indices <- result$sample_indices + 1
  
  return(result)
}

#' Get Sample Weights for Distributed Forest
#'
#' Computes sample weights using a distributed forest model.
#' Only works when store_samples = TRUE.
#'
#' @param distributed_forest A distributed forest object with store_samples = TRUE
#' @param test_sample Numeric vector of length p (must match training features)
#'
#' @return A list with sample_indices, weights, and Y_values
#'
#' @export
get_sample_weights_distributed <- function(distributed_forest, test_sample) {
  
  # Check if store_samples was enabled
  if (!is.null(distributed_forest$store_samples) && !distributed_forest$store_samples) {
    stop("Sample weights are only available when store_samples = TRUE.\n",
         "Please rebuild your forest with store_samples = TRUE.")
  }
  
  # Input validation and coercion
  if (is.matrix(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation (vector or 1-row matrix)")
    }
    test_sample <- as.vector(test_sample)
    warning("test_sample was a matrix. Converting to vector. ",
            "Consider using test_sample[1, , drop = FALSE] for consistency, ",
            "then pass as vector or use as.vector().")
  } else if (is.data.frame(test_sample)) {
    if (nrow(test_sample) != 1) {
      stop("test_sample must be a single observation")
    }
    test_sample <- as.numeric(test_sample[1, ])
    warning("test_sample was a data frame. Converting to numeric vector.")
  }
  
  if (!is.vector(test_sample) && !is.numeric(test_sample)) {
    stop("test_sample must be a numeric vector, 1-row matrix, or single-row data frame")
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
  
  # Merge all weights
  all_indices <- c()
  all_weights <- c()
  
  for (i in seq_along(valid_weights)) {
    w <- valid_weights[[i]]
    all_indices <- c(all_indices, w$sample_indices + 1)  # Convert to 1-indexed
    weight_scale <- distributed_forest$trees_per_worker[i] / distributed_forest$total_trees
    all_weights <- c(all_weights, w$weights * weight_scale)
  }
  
  # Aggregate duplicate indices
  unique_indices <- unique(all_indices)
  aggregated_weights <- sapply(unique_indices, function(idx) {
    sum(all_weights[all_indices == idx])
  })
  
  # Normalize
  aggregated_weights <- aggregated_weights / sum(aggregated_weights)
  
  Y_values <- Y_train[unique_indices, , drop = FALSE]
  
  return(list(
    sample_indices = unique_indices,
    weights = aggregated_weights,
    Y_values = Y_values
  ))
}
