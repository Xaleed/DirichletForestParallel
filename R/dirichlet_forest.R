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
#' @param n_cores Integer, number of cores to use. If -1, uses all available cores minus 1.
#'        If 1, uses sequential processing (default: -1)
#'
#' @return A list containing the distributed forest model with the following components:
#' \describe{
#'   \item{type}{Character, type of parallelization used ("sequential", "fork", or "cluster")}
#'   \item{forest}{For sequential: the forest object}
#'   \item{worker_forests}{For fork: list of worker forests}
#'   \item{cluster}{For cluster: the cluster object}
#'   \item{n_cores}{Integer, number of cores used}
#'   \item{trees_per_worker}{Integer vector, number of trees per worker}
#'   \item{total_trees}{Integer, total number of trees}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate sample data
#' n <- 100
#' p <- 5
#' k <- 3
#' X <- matrix(rnorm(n * p), n, p)
#' 
#' # Generate compositional response
#' alpha <- matrix(runif(k), 1, k) + 0.5
#' Y <- matrix(0, n, k)
#' for(i in 1:n) {
#'   Y[i, ] <- MCMCpack::rdirichlet(1, alpha)
#' }
#' 
#' # Build forest
#' forest <- DirichletForest_distributed(X, Y, B = 50, n_cores = 2)
#' 
#' # Clean up (important for cluster-based)
#' cleanup_distributed_forest(forest)
#' }
#'
#' @export

DirichletForest_distributed <- function(X, Y, B = 100, d_max = 10, n_min = 5, 
                                        m_try = -1, seed = 123, method = "mom", 
                                        n_cores = -1) {
  
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
    forest_seq <- DirichletForest(X, Y, B, d_max, n_min, m_try, seed, method)
    return(list(
      type = "sequential",
      forest = forest_seq,
      n_cores = 1,
      trees_per_worker = B
    ))
  }
  
  # Determine cores for parallel processing
  if (n_cores == -1) {
    n_cores <- max(1, parallel::detectCores() - 1)
  }
  n_cores <- max(1, min(n_cores, B))
  
  # For small forests, use sequential
  if (B < max(4, n_cores)) {
    forest_seq <- DirichletForest(X, Y, B, d_max, n_min, m_try, seed, method)
    return(list(
      type = "sequential", 
      forest = forest_seq,
      n_cores = 1,
      trees_per_worker = B
    ))
  }
  
  cat("Building distributed forest with", n_cores, "workers for", B, "trees\n")
  
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
                      seed = worker_seeds[i], method = method)
    }, mc.cores = n_cores)
    
    return(list(
      type = "fork",
      worker_forests = worker_forests,
      n_cores = n_cores,
      trees_per_worker = trees_per_core,
      total_trees = sum(trees_per_core)
    ))
    
  } else {
    # Windows: cluster-based - keep workers alive for predictions
    cat("Using persistent cluster (Windows)\n")
    
    cl <- parallel::makeCluster(n_cores, type = "PSOCK")
    
    # Setup workers with Rcpp functions
    setup_cluster_workers(cl)
    
    # Export variables to workers
    parallel::clusterExport(cl, c("X", "Y", "d_max", "n_min", "m_try", "method", 
                                  "trees_per_core", "worker_seeds"), envir = environment())
    
    # Build forests in each worker
    parallel::clusterApply(cl, seq_len(n_cores), function(worker_id) {
      # Build and store forest in worker's environment
      worker_forest <- DirichletForest(X, Y, B = trees_per_core[worker_id],
                                       d_max = d_max, n_min = n_min, m_try = m_try,
                                       seed = worker_seeds[worker_id], method = method)
      # Store in worker's global environment
      assign("worker_forest", worker_forest, envir = .GlobalEnv)
      return(worker_forest$n_trees)  # Just return confirmation
    })
    
    return(list(
      type = "cluster",
      cluster = cl,
      n_cores = n_cores, 
      trees_per_worker = trees_per_core,
      total_trees = sum(trees_per_core)
    ))
  }
}

#' Clean Up Distributed Forest
#'
#' Properly cleans up resources used by distributed forest, especially
#' important for cluster-based parallelization on Windows.
#'
#' @param distributed_forest A distributed forest object created by 
#'        \code{\link{DirichletForest_distributed}}
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
#' Handles all types of parallelization transparently.
#'
#' @param distributed_forest A distributed forest object created by 
#'        \code{\link{DirichletForest_distributed}}
#' @param X_new Numeric matrix of new predictors to predict on (n_new x p)
#'
#' @return A list containing predictions:
#' \describe{
#'   \item{alpha_predictions}{Matrix of predicted Dirichlet alpha parameters (n_new x k)}
#'   \item{mean_predictions}{Matrix of predicted mean compositions (n_new x k)}
#' }
#'
#' @examples
#' \dontrun{
#' # Using the forest from the previous example
#' X_new <- matrix(rnorm(20 * 5), 20, 5)
#' predictions <- predict_distributed_forest(forest, X_new)
#' 
#' # Access predictions
#' alpha_pred <- predictions$alpha_predictions
#' mean_pred <- predictions$mean_predictions
#' }
#'
#' @export
predict_distributed_forest <- function(distributed_forest, X_new) {
  
  if (!is.matrix(X_new)) {
    X_new <- as.matrix(X_new)
  }
  
  n_samples <- nrow(X_new)
  
  if (distributed_forest$type == "sequential") {
    # Simple case - just use regular prediction
    return(PredictDirichletForest(distributed_forest$forest, X_new))
  }
  
  if (distributed_forest$type == "fork") {
    # Fork-based: workers are done, forests are in memory
    cat("Predicting with", distributed_forest$n_cores, "fork workers\n")
    
    # Get predictions from each worker's forest
    worker_predictions <- parallel::mclapply(seq_len(distributed_forest$n_cores), function(i) {
      worker_forest <- distributed_forest$worker_forests[[i]]
      if (worker_forest$n_trees > 0) {
        # Force proper return structure
        pred_result <- PredictDirichletForest(worker_forest, X_new)
        # Ensure it's a proper list structure
        if (is.list(pred_result) && 
            !is.null(pred_result$alpha_predictions) && 
            !is.null(pred_result$mean_predictions)) {
          return(pred_result)
        } else {
          return(NULL)
        }
      } else {
        return(NULL)
      }
    }, mc.cores = distributed_forest$n_cores, mc.preschedule = FALSE)
    
    # Filter out null results and validate structure
    valid_predictions <- list()
    for (i in seq_along(worker_predictions)) {
      pred <- worker_predictions[[i]]
      if (!is.null(pred) && is.list(pred) && 
          !is.null(pred$alpha_predictions) && !is.null(pred$mean_predictions)) {
        valid_predictions <- append(valid_predictions, list(pred))
      }
    }
    
  } else if (distributed_forest$type == "cluster") {
    # Cluster-based: send prediction job to each worker
    cat("Predicting with", distributed_forest$n_cores, "cluster workers\n")
    
    cl <- distributed_forest$cluster
    
    # Export test data to workers
    parallel::clusterExport(cl, "X_new", envir = environment())
    
    # Each worker predicts with its forest
    worker_predictions <- parallel::clusterApply(cl, seq_len(distributed_forest$n_cores), function(worker_id) {
      # Use the forest stored in this worker's environment
      if (exists("worker_forest", envir = .GlobalEnv)) {
        forest <- get("worker_forest", envir = .GlobalEnv)
        if (forest$n_trees > 0) {
          pred_result <- PredictDirichletForest(forest, X_new)
          # Ensure proper structure
          if (is.list(pred_result) && 
              !is.null(pred_result$alpha_predictions) && 
              !is.null(pred_result$mean_predictions)) {
            return(pred_result)
          }
        }
      }
      return(NULL)
    })
    
    # Filter out null results and validate structure
    valid_predictions <- list()
    for (i in seq_along(worker_predictions)) {
      pred <- worker_predictions[[i]]
      if (!is.null(pred) && is.list(pred) && 
          !is.null(pred$alpha_predictions) && !is.null(pred$mean_predictions)) {
        valid_predictions <- append(valid_predictions, list(pred))
      }
    }
  }
  
  # Combine predictions from all workers
  if (length(valid_predictions) == 0) {
    stop("No valid predictions from workers")
  }
  
  cat("Combining predictions from", length(valid_predictions), "workers\n")
  
  # Extract dimensions from first valid prediction
  first_pred <- valid_predictions[[1]]
  
  # Defensive programming: check structure before accessing
  if (!is.list(first_pred) || is.null(first_pred$alpha_predictions)) {
    stop("Invalid prediction structure from workers")
  }
  
  n_classes <- ncol(first_pred$alpha_predictions)
  
  # Initialize combined results
  combined_alpha <- array(0, dim = c(n_samples, n_classes))
  combined_mean <- array(0, dim = c(n_samples, n_classes))
  
  # Weight each worker's contribution by number of trees
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
