# Simple and reliable approach for cluster workers
setup_cluster_workers <- function(cl, pkg_name = "DirichletForestParallel") {
  
  # Load Rcpp on workers
  parallel::clusterEvalQ(cl, library(Rcpp))
  
  # Load the package's shared library on each worker
  parallel::clusterCall(cl, function(pkg_name) {
    # Find the installed package
    lib_path <- find.package(pkg_name)
    
    # Load the compiled DLL
    dll_file <- file.path(lib_path, "libs", 
                          paste0(pkg_name, .Platform$dynlib.ext))
    
    if (file.exists(dll_file)) {
      library.dynam(pkg_name, pkg_name, lib_path)
      return("DLL loaded successfully")
    } else {
      return("DLL not found")
    }
  }, pkg_name)
  
  # Verify functions are available
  parallel::clusterCall(cl, function() {
    if (exists("DirichletForest") && exists("PredictDirichletForest")) {
      return("Functions available")
    } else {
      return("Functions not found")
    }
  })
}