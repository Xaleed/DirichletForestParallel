# DirichletForest_distributed  

This repository contains an implementation of a **parallel Dirichlet Random Forest**, designed for modeling **compositional (Dirichlet-distributed) data**.  

⚠️ **Note**: This project is still in progress. For a simpler and more stable version, see my [DirichletRandom](https://github.com/Xaleed/DirichletForest.git) repository.  

---

## 📦 Installation  

Clone this repository and install locally in R:  

```r
devtools::install_github("https://github.com/Xaleed/DirichletForestParallel.git")
 
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
df_par3 <- DirichletForest_distributed(X, Y, B = 50, n_cores = 3)

# Predict on new data (here we reuse X for illustration)
pred3 <- predict_distributed_forest(df_par3, X)

# Clean up cluster resources
cleanup_distributed_forest(df_par3)
