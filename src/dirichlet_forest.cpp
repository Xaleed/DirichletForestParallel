#include <Rcpp.h>

#include <vector>

#include <random>

#include <algorithm>

#include <cmath>

#include <limits>

#include <unordered_map>



using namespace Rcpp;


// ============================================================================
// NODE STRUCTURE
// ============================================================================
struct Node {
    int feature_index;
    double split_value;
    bool is_leaf;
    Node* left;
    Node* right;
    
    // For store_samples = FALSE (fast mode)
    NumericVector mean_prediction;
    NumericVector alpha_prediction;
    
    // For store_samples = TRUE (distributional mode)
    std::vector<int> leaf_samples;
    
    Node() : feature_index(-1), split_value(0.0), is_leaf(false), 
             left(nullptr), right(nullptr) {}
    
    ~Node() {
        delete left;
        delete right;
    }
};

// Helper function to calculate Dirichlet log-likelihood



double log_likelihood_dirichlet_rcpp(const NumericMatrix& Y, const NumericVector& alpha) {



    int n = Y.nrow();



    int k = Y.ncol();



    double loglik = 0.0;



    double alpha_sum = 0.0;



    



    // Calculate alpha sum



    for (int j = 0; j < k; j++) {



        alpha_sum += alpha[j];



    }



    



    double log_gamma_alpha_sum = R::lgammafn(alpha_sum);



    



    // Pre-compute log_gamma for alpha values



    std::vector<double> log_gamma_alpha(k);



    for (int j = 0; j < k; j++) {



        log_gamma_alpha[j] = R::lgammafn(alpha[j]);



    }



    



    for (int i = 0; i < n; i++) {



        double sum_y = 0.0;



        double row_contrib = 0.0;



        



        // Check validity and calculate contribution



        for (int j = 0; j < k; j++) {



            double y_val = Y(i, j);



            if (y_val <= 0 || y_val >= 1) {



                return -1e18; // Invalid values



            }



            sum_y += y_val;



            row_contrib += (alpha[j] - 1) * std::log(y_val);



        }



        



        if (std::abs(sum_y - 1.0) > 1e-6) {



            return -1e18; // Doesn't sum to 1



        }



        



        loglik += log_gamma_alpha_sum;



        for (int j = 0; j < k; j++) {



            loglik -= log_gamma_alpha[j];



        }



        loglik += row_contrib;



    }



    



    return loglik;



}







// =============================================================================

// METHOD OF MOMENTS ESTIMATION - IMPLEMENTATION

// =============================================================================



NumericVector estimate_parameters_mom_rcpp(const NumericMatrix& Y) {



    const int n = Y.nrow();



    const int k = Y.ncol();



    



    // Handle empty matrix case



    if (n == 0) {



        return NumericVector(k, 1.0);



    }



    



    // Handle single sample case



    if (n == 1) {



        NumericVector result(k);



        for (int j = 0; j < k; j++) {



            result[j] = std::max(0.1, std::min(1000.0, Y(0, j)));



        }



        return result;



    }



    



    // Calculate means efficiently



    NumericVector means(k, 0.0);



    for (int i = 0; i < n; i++) {



        for (int j = 0; j < k; j++) {



            means[j] += Y(i, j);



        }



    }



    



    const double inv_n = 1.0 / n;



    for (int j = 0; j < k; j++) {



        means[j] *= inv_n;



    }



    



    // Calculate sample variances with Bessel's correction (n-1 denominator)



    NumericVector variances(k, 0.0);



    for (int i = 0; i < n; i++) {



        for (int j = 0; j < k; j++) {



            const double diff = Y(i, j) - means[j];



            variances[j] += diff * diff;



        }



    }



    



    // Use n-1 for sample variance (Bessel's correction)



    const double inv_n_minus_1 = 1.0 / (n - 1);



    for (int j = 0; j < k; j++) {



        variances[j] *= inv_n_minus_1;



    }



    



    // Ensure first variance is not too close to zero (numerical stability)



    const double min_var = 1e-8;



    // --- MODIFICATION HERE: Apply min_var to all variances, not just the first ---



    for (int j = 0; j < k; j++) {



        if (variances[j] < min_var) {



            variances[j] = min_var;



        }



    }



    // -----------------------------------------------------------------------------



    



    // Estimate concentration parameter using first category



    // v = (μ₁(1-μ₁))/σ₁² - 1



    const double numerator = means[0] * (1.0 - means[0]);



    double v_val = numerator / variances[0] - 1.0;



    



    // --- MODIFICATION HERE: Match Julia's max(v, 0.1) for v_val ---



    if (v_val <= 0.0) { // If it's zero or negative



        v_val = 0.1;   // Use 0.1 as the lower bound, matching Julia



    }



    // -----------------------------------------------------------------



    



    // Calculate alpha parameters: αⱼ = v * μⱼ



    NumericVector alpha(k);



    for (int j = 0; j < k; j++) {



        alpha[j] = v_val * means[j];



    }



    



    // Apply bounds to ensure numerical stability



    // Match Julia's clamp.(alpha, 0.1, 1000.0)



    for (int j = 0; j < k; j++) {



        if (alpha[j] < 0.1) {



            alpha[j] = 0.1;



        } else if (alpha[j] > 1000.0) {



            alpha[j] = 1000.0;



        }



    }



    



    return alpha;



}





// R export wrapper - calls the C++ function above

// [[Rcpp::export]]

NumericVector estimate_dirichlet_mom(const NumericMatrix& Y) {

    return estimate_parameters_mom_rcpp(Y);

}



// =============================================================================

// MLE ESTIMATION - IMPLEMENTATION

// =============================================================================

bool lu_solve(std::vector<std::vector<double>>& A, 

              const std::vector<double>& b, 

              std::vector<double>& x, 

              int n) {

    

    // Create permutation vector

    std::vector<int> perm(n);

    for (int i = 0; i < n; i++) perm[i] = i;

    

    // LU decomposition with partial pivoting

    for (int k = 0; k < n; k++) {

        // Find pivot

        int max_row = k;

        double max_val = std::abs(A[k][k]);

        for (int i = k + 1; i < n; i++) {

            double val = std::abs(A[i][k]);

            if (val > max_val) {

                max_val = val;

                max_row = i;

            }

        }

        

        if (max_val < 1e-12) return false; // Singular matrix

        

        // Swap rows

        if (max_row != k) {

            std::swap(A[k], A[max_row]);

            std::swap(perm[k], perm[max_row]);

        }

        

        // Eliminate

        for (int i = k + 1; i < n; i++) {

            A[i][k] /= A[k][k];

            for (int j = k + 1; j < n; j++) {

                A[i][j] -= A[i][k] * A[k][j];

            }

        }

    }

    

    // Apply permutation to b

    std::vector<double> b_perm(n);

    for (int i = 0; i < n; i++) {

        b_perm[i] = b[perm[i]];

    }

    

    // Forward substitution (solve L * y = b_perm)

    std::vector<double> y(n);

    for (int i = 0; i < n; i++) {

        y[i] = -b_perm[i];  // Note: we want -grad

        for (int j = 0; j < i; j++) {

            y[i] -= A[i][j] * y[j];

        }

    }

    

    // Back substitution (solve U * x = y)

    for (int i = n - 1; i >= 0; i--) {

        x[i] = y[i];

        for (int j = i + 1; j < n; j++) {

            x[i] -= A[i][j] * x[j];

        }

        x[i] /= A[i][i];

    }

    

    return true;

}

NumericVector estimate_parameters_mle_newton_rcpp(const NumericMatrix& Y, int max_iter = 1000, double tol = 1e-6, double lambda = 1e-6) {

    int n = Y.nrow();



    int k = Y.ncol();



    



    if (n == 0) {



        return NumericVector(k, 1.0);



    }



    



    // Initialize with method of moments



    NumericVector alpha = estimate_parameters_mom_rcpp(Y);



    



    // Pre-calculate log Y values



    std::vector<std::vector<double>> log_Y(n, std::vector<double>(k));



    std::vector<std::vector<bool>> valid_log(n, std::vector<bool>(k, false));



    



    for (int i = 0; i < n; i++) {



        for (int j = 0; j < k; j++) {



            if (Y(i, j) > 0) {



                log_Y[i][j] = std::log(Y(i, j));



                valid_log[i][j] = true;



            }



        }



    }



    



    for (int iter = 0; iter < max_iter; iter++) {



        double alpha_sum = 0.0;



        for (int j = 0; j < k; j++) {



            alpha_sum += alpha[j];



        }



        



        double digamma_alpha_sum = R::digamma(alpha_sum);



        double trigamma_alpha_sum = R::trigamma(alpha_sum);



        



        // Calculate gradient



        std::vector<double> grad(k, 0.0);



        for (int j = 0; j < k; j++) {



            grad[j] = n * (digamma_alpha_sum - R::digamma(alpha[j]));



            for (int i = 0; i < n; i++) {



                if (valid_log[i][j]) {



                    grad[j] += log_Y[i][j];



                }



            }



        }



        



        // Calculate Hessian



        std::vector<std::vector<double>> H(k, std::vector<double>(k));



        for (int j = 0; j < k; j++) {



            for (int l = 0; l < k; l++) {



                if (j == l) {



                    H[j][l] = n * (trigamma_alpha_sum - R::trigamma(alpha[j])) + lambda;



                } else {



                    H[j][l] = n * trigamma_alpha_sum;



                }



            }



        }



        



        // Solve H * delta = -grad using LU decomposition (more stable)

        std::vector<double> delta(k);



        // Make a copy of H since LU decomposition will modify it

        std::vector<std::vector<double>> H_copy = H;



        // Try to solve using LU decomposition

        bool success = lu_solve(H_copy, grad, delta, k);



        if (!success) {

            // Fallback to diagonal approximation if LU fails

            for (int j = 0; j < k; j++) {

                double diag_val = n * (trigamma_alpha_sum - R::trigamma(alpha[j])) + lambda;

                delta[j] = -grad[j] / diag_val;

            }

        }



        



        // Check convergence



        double norm_delta_sq = 0.0;



        for (int j = 0; j < k; j++) {



            norm_delta_sq += delta[j] * delta[j];



        }



        



        if (norm_delta_sq < tol * tol) {



            break;



        }



        



        // Line search



        double step_size = 1.0;



        bool valid_step = false;



        



        for (int ls = 0; ls < 10; ls++) {



            bool all_valid = true;



            for (int j = 0; j < k; j++) {



                double new_alpha = alpha[j] + step_size * delta[j];



                if (new_alpha < 0.1 || new_alpha > 1000.0) {



                    all_valid = false;



                    break;



                }



            }



            



            if (all_valid) {



                for (int j = 0; j < k; j++) {



                    alpha[j] += step_size * delta[j];



                }



                valid_step = true;



                break;



            }



            



            step_size *= 0.5;



        }



        



        if (!valid_step) {



            break;



        }



    }



    



    return alpha;



}





// R export wrapper - calls the C++ function above

// [[Rcpp::export]]

NumericVector estimate_dirichlet_mle(const NumericMatrix& Y, int max_iter = 10000, double tol = 1e-6, double lambda = 1e-6) {

    return estimate_parameters_mle_newton_rcpp(Y, max_iter, tol, lambda);

}





// ============================================================================

// LEAF NODE: Calculate Predictions

// ============================================================================



// Calculate mean of compositional data in leaf

NumericVector calculate_mean_prediction(const NumericMatrix& Y, 

                                       const IntegerVector& indices) {

  int k = Y.ncol();

  NumericVector means(k, 0.0);

  

  if (indices.size() == 0) {

    // Return uniform distribution if no samples

    for (int j = 0; j < k; j++) {

      means[j] = 1.0 / k;

    }

    return means;

  }

  

  // Average the compositional values

  for (int j = 0; j < k; j++) {

    double sum = 0.0;

    for (int i = 0; i < indices.size(); i++) {

      sum += Y(indices[i], j);

    }

    means[j] = sum / indices.size();

  }

  

  return means;

}



// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
List predict_sample_tree_fast(Node* node, const NumericVector& x);

// ============================================================================
// FIT TERMINAL NODE - DUAL MODE (MODIFIED)
// ============================================================================
void FitTerminalNode(Node* node, const NumericMatrix& Y,
                     const IntegerVector& sample_indices,
                     const std::string& method,
                     bool store_samples) {
    
    node->is_leaf = true;
    
    // ALWAYS compute mean and alpha predictions (for both modes)
    if (sample_indices.size() == 0) {
        int k = Y.ncol();
        node->alpha_prediction = NumericVector(k, 1.0);
        node->mean_prediction = NumericVector(k, 1.0/k);
    } else {
        // Calculate mean
        node->mean_prediction = NumericVector(Y.ncol());
        for (int j = 0; j < Y.ncol(); j++) {
            double sum = 0.0;
            for (int i = 0; i < sample_indices.size(); i++) {
                sum += Y(sample_indices[i], j);
            }
            node->mean_prediction[j] = sum / sample_indices.size();
        }
        
        // Estimate alpha parameters
        NumericMatrix Y_subset(sample_indices.size(), Y.ncol());
        for (int i = 0; i < sample_indices.size(); i++) {
            for (int j = 0; j < Y.ncol(); j++) {
                Y_subset(i, j) = Y(sample_indices[i], j);
            }
        }
        
        if (method == "mle") {
            node->alpha_prediction = estimate_parameters_mle_newton_rcpp(Y_subset);
        } else {
            node->alpha_prediction = estimate_parameters_mom_rcpp(Y_subset);
        }
    }
    
    // Additionally store sample indices if requested
    if (store_samples) {
        node->leaf_samples.clear();
        node->leaf_samples.reserve(sample_indices.size());
        for (int i = 0; i < sample_indices.size(); i++) {
            node->leaf_samples.push_back(sample_indices[i]);
        }
    }
}

// ============================================================================

// FIT TERMINAL NODE - Store sample indices only

// ============================================================================

void FitTerminalNode(Node* node, const IntegerVector& sample_indices) {

    node->is_leaf = true;

    node->leaf_samples.clear();

    node->leaf_samples.reserve(sample_indices.size());

    

    for (int i = 0; i < sample_indices.size(); i++) {

        node->leaf_samples.push_back(sample_indices[i]);

    }

}



// ============================================================================

// SPLITTING: Find Best Split Point

// ============================================================================

List FindBestSplit(const NumericMatrix& X, const NumericMatrix& Y,

                   const IntegerVector& sample_indices,

                   const IntegerVector& feature_subset,

                   int n_min,

                   const std::string& method) {

    

    double best_gain = -std::numeric_limits<double>::infinity();

    int best_feature = -1;

    double best_split_value = 0.0;

    IntegerVector best_left_indices, best_right_indices;

    

    int n_samples = sample_indices.size();

    

    // Calculate parent node's log-likelihood

    NumericMatrix Y_parent(n_samples, Y.ncol());

    for (int i = 0; i < n_samples; i++) {

        for (int j = 0; j < Y.ncol(); j++) {

            Y_parent(i, j) = Y(sample_indices[i], j);

        }

    }

    

    NumericVector parent_alpha;

    if (method == "mle") {

        parent_alpha = estimate_parameters_mle_newton_rcpp(Y_parent);

    } else {

        parent_alpha = estimate_parameters_mom_rcpp(Y_parent);

    }

    double parent_loglik = log_likelihood_dirichlet_rcpp(Y_parent, parent_alpha);

    

    // Try each feature

    int n_features = feature_subset.size();

    for (int f = 0; f < n_features; f++) {

        int feature = feature_subset[f];

        

        // Get sorted unique values

        std::vector<double> values;

        values.reserve(n_samples);

        for (int i = 0; i < n_samples; i++) {

            values.push_back(X(sample_indices[i], feature));

        }

        std::sort(values.begin(), values.end());

        values.erase(std::unique(values.begin(), values.end()), values.end());

        

        int n_values = static_cast<int>(values.size());

        if (n_values <= 1) continue;

        

        // Try split points

        for (int k = 1; k < n_values; k++) {

            double split_val = (values[k-1] + values[k]) / 2.0;  // Uses midpoints split_val = values[k];
            

            

            std::vector<int> left_idx, right_idx;

            left_idx.reserve(n_samples);

            right_idx.reserve(n_samples);

            

            for (int i = 0; i < n_samples; i++) {

                int idx = sample_indices[i];

                if (X(idx, feature) <= split_val) {

                    left_idx.push_back(idx);

                } else {

                    right_idx.push_back(idx);

                }

            }

            

            int n_left = static_cast<int>(left_idx.size());

            int n_right = static_cast<int>(right_idx.size());

            

            if (n_left < 2 || n_right < 2) continue;

            

            // Create subsets

            NumericMatrix Y_left(n_left, Y.ncol());

            for (int i = 0; i < n_left; i++) {

                for (int j = 0; j < Y.ncol(); j++) {

                    Y_left(i, j) = Y(left_idx[i], j);

                }

            }

            

            NumericMatrix Y_right(n_right, Y.ncol());

            for (int i = 0; i < n_right; i++) {

                for (int j = 0; j < Y.ncol(); j++) {

                    Y_right(i, j) = Y(right_idx[i], j);

                }

            }

            

            // Estimate parameters

            NumericVector left_alpha = (method == "mle") ? 

                estimate_parameters_mle_newton_rcpp(Y_left) : 

                estimate_parameters_mom_rcpp(Y_left);

            

            NumericVector right_alpha = (method == "mle") ? 

                estimate_parameters_mle_newton_rcpp(Y_right) : 

                estimate_parameters_mom_rcpp(Y_right);

            

            // Calculate gain

            double left_loglik = log_likelihood_dirichlet_rcpp(Y_left, left_alpha);

            double right_loglik = log_likelihood_dirichlet_rcpp(Y_right, right_alpha);

            double gain = (left_loglik + right_loglik) - parent_loglik;

            

            if (gain > best_gain) {

                best_gain = gain;

                best_feature = feature;

                best_split_value = split_val;

                best_left_indices = IntegerVector(left_idx.begin(), left_idx.end());

                best_right_indices = IntegerVector(right_idx.begin(), right_idx.end());

            }

        }

    }

    

    return List::create(

        Named("gain") = best_gain,

        Named("feature") = best_feature,

        Named("split_value") = best_split_value,

        Named("left_indices") = best_left_indices,

        Named("right_indices") = best_right_indices

    );

}



// ============================================================================
// GROW TREE - WITH store_samples PARAMETER
// ============================================================================
Node* GrowTree(const NumericMatrix& X, const NumericMatrix& Y,
               const IntegerVector& sample_indices,
               int current_depth, int d_max, int n_min, int m_try,
               std::mt19937& gen, const std::string& method,
               bool store_samples) {
    
    Node* node = new Node();
    
    // Check termination conditions
    if (sample_indices.size() < n_min || current_depth >= d_max || 
        sample_indices.size() == 0) {
        FitTerminalNode(node, Y, sample_indices, method, store_samples);
        return node;
    }
    
    // Feature subset selection
    int n_features = X.ncol();
    IntegerVector all_features = seq(0, n_features - 1);
    std::shuffle(all_features.begin(), all_features.end(), gen);
    IntegerVector feature_subset(all_features.begin(), 
                                 all_features.begin() + std::min(m_try, n_features));
    
    // Find best split
    List split_result = FindBestSplit(X, Y, sample_indices, feature_subset, n_min, method);
    double gain = as<double>(split_result["gain"]);
    
    if (gain <= 0 || as<int>(split_result["feature"]) == -1) {
        FitTerminalNode(node, Y, sample_indices, method, store_samples);
        return node;
    }
    
    // Set node properties
    node->feature_index = as<int>(split_result["feature"]);
    node->split_value = as<double>(split_result["split_value"]);
    node->is_leaf = false;
    
    // Grow children
    IntegerVector left_indices = as<IntegerVector>(split_result["left_indices"]);
    IntegerVector right_indices = as<IntegerVector>(split_result["right_indices"]);
    
    node->left = GrowTree(X, Y, left_indices, current_depth + 1, d_max, n_min, 
                          m_try, gen, method, store_samples);
    node->right = GrowTree(X, Y, right_indices, current_depth + 1, d_max, n_min, 
                           m_try, gen, method, store_samples);
    
    return node;
}



// ============================================================================
// TREE TRAVERSAL - Find leaf node
// ============================================================================
Node* FindLeafNode(Node* node, const NumericVector& x) {
    if (node->is_leaf) {
        return node;
    }
    
    if (x[node->feature_index] <= node->split_value) {
        return FindLeafNode(node->left, x);
    } else {
        return FindLeafNode(node->right, x);
    }
}


// ============================================================================
// FAST PREDICTION (for store_samples = FALSE)
// ============================================================================
List predict_sample_tree_fast(Node* node, const NumericVector& x) {
    if (node->is_leaf) {
        return List::create(
            Named("alpha_prediction") = node->alpha_prediction,
            Named("mean_prediction") = node->mean_prediction
        );
    }
    
    if (x[node->feature_index] <= node->split_value) {
        return predict_sample_tree_fast(node->left, x);
    } else {
        return predict_sample_tree_fast(node->right, x);
    }
}




// ============================================================================
// COMPUTE WEIGHTS (for store_samples = TRUE)
// ============================================================================
std::unordered_map<int, double> ComputeWeights(const NumericVector& sample,
                                                const List& forest_ptrs,
                                                int n_trees) {
    std::unordered_map<int, double> weights_by_sample;
    
    for (int t = 0; t < n_trees; t++) {
        XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[t]));
        Node* leaf = FindLeafNode(tree_ptr, sample);
        
        const std::vector<int>& leaf_samples = leaf->leaf_samples;
        if (leaf_samples.empty()) continue;
        
        double sample_weight = 1.0 / leaf_samples.size();
        
        for (int sample_idx : leaf_samples) {
            weights_by_sample[sample_idx] += sample_weight;
        }
    }
    
    // Normalize
    double total_weight = 0.0;
    for (const auto& entry : weights_by_sample) {
        total_weight += entry.second;
    }
    
    if (total_weight > 0) {
        for (auto& entry : weights_by_sample) {
            entry.second /= total_weight;
        }
    }
    
    return weights_by_sample;
}



// ============================================================================
// BUILD DIRICHLET FOREST
// ============================================================================
// [[Rcpp::export]]
List DirichletForest(NumericMatrix X, NumericMatrix Y, int B = 100,
                     int d_max = 10, int n_min = 5, int m_try = -1,
                     int seed = 123, std::string method = "mom",
                     bool store_samples = false) {
    
    int n_samples = X.nrow();
    int n_features = X.ncol();
    
    if (m_try <= 0) {
        m_try = std::max(1, (int)std::sqrt(n_features));
    }
    
    std::mt19937 gen(seed);
    std::vector<Node*> forest(B);
    
    for (int b = 0; b < B; b++) {
        // Bootstrap sampling without replacement (shuffle)
        IntegerVector all_indices = seq(0, n_samples - 1);
        std::shuffle(all_indices.begin(), all_indices.end(), gen);
        
        IntegerVector bootstrap_indices(n_samples);
        for (int i = 0; i < n_samples; i++) {
            bootstrap_indices[i] = all_indices[i];
        }
        
        // Grow tree with store_samples flag
        forest[b] = GrowTree(X, Y, bootstrap_indices, 0, d_max, n_min, 
                             m_try, gen, method, store_samples);
    }
    
    // Convert to external pointers
    List forest_ptrs(B);
    for (int i = 0; i < B; i++) {
        forest_ptrs[i] = XPtr<Node>(forest[i]);
    }
    
    List result = List::create(
        Named("forest") = forest_ptrs,
        Named("n_trees") = B,
        Named("n_features") = n_features,
        Named("n_classes") = Y.ncol(),
        Named("store_samples") = store_samples
    );
    
    // Add training data only if store_samples = TRUE
    if (store_samples) {
        result["X_train"] = X;
        result["Y_train"] = Y;
    }
    
    return result;
}

// ============================================================================
// WEIGHT-BASED PREDICTION (for store_samples = TRUE)
// ============================================================================
// [[Rcpp::export]]
List PredictDirichletForestWeightBased(List forest_model, NumericMatrix X_new, 
                                       std::string method = "mom") {
    
    List forest_ptrs = forest_model["forest"];
    int n_trees = forest_model["n_trees"];
    int n_classes = forest_model["n_classes"];
    int n_samples = X_new.nrow();
    
    NumericMatrix X_train = forest_model["X_train"];
    NumericMatrix Y_train = forest_model["Y_train"];
    
    NumericMatrix alpha_predictions(n_samples, n_classes);
    NumericMatrix mean_predictions(n_samples, n_classes);
    
    for (int i = 0; i < n_samples; i++) {
        NumericVector sample = X_new(i, _);
        
        std::unordered_map<int, double> weights = ComputeWeights(sample, forest_ptrs, n_trees);
        
        std::vector<int> weighted_indices;
        std::vector<double> sample_weights;
        
        for (const auto& entry : weights) {
            if (entry.second > 1e-10) {
                weighted_indices.push_back(entry.first);
                sample_weights.push_back(entry.second);
            }
        }
        
        if (weighted_indices.empty()) {
            for (int j = 0; j < n_classes; j++) {
                alpha_predictions(i, j) = 1.0;
                mean_predictions(i, j) = 1.0 / n_classes;
            }
            continue;
        }
        
        // Calculate weighted mean
        NumericVector mean_pred(n_classes, 0.0);
        for (size_t k = 0; k < weighted_indices.size(); k++) {
            for (int j = 0; j < n_classes; j++) {
                mean_pred[j] += sample_weights[k] * Y_train(weighted_indices[k], j);
            }
        }
        
        // Create replicated dataset for alpha estimation
        std::vector<int> replicated_indices;
        int replication_factor = 100;
        
        for (size_t k = 0; k < weighted_indices.size(); k++) {
            int n_reps = std::max(1, (int)(sample_weights[k] * replication_factor));
            for (int r = 0; r < n_reps; r++) {
                replicated_indices.push_back(weighted_indices[k]);
            }
        }
        
        NumericMatrix Y_weighted(replicated_indices.size(), n_classes);
        for (size_t k = 0; k < replicated_indices.size(); k++) {
            for (int j = 0; j < n_classes; j++) {
                Y_weighted(k, j) = Y_train(replicated_indices[k], j);
            }
        }
        
        NumericVector alpha_pred;
        if (method == "mle") {
            alpha_pred = estimate_parameters_mle_newton_rcpp(Y_weighted);
        } else {
            alpha_pred = estimate_parameters_mom_rcpp(Y_weighted);
        }
        
        for (int j = 0; j < n_classes; j++) {
            alpha_predictions(i, j) = alpha_pred[j];
            mean_predictions(i, j) = mean_pred[j];
        }
    }
    
    return List::create(
        Named("alpha_predictions") = alpha_predictions,
        Named("mean_predictions") = mean_predictions
    );
}

// ============================================================================
// GET LEAF PREDICTIONS ( store_samples = TRUE)
// ============================================================================
// [[Rcpp::export]]
List GetLeafPredictions(List forest_model, NumericMatrix X_new) {
    
    List forest_ptrs = forest_model["forest"];
    int n_trees = forest_model["n_trees"];
    int n_classes = forest_model["n_classes"];
    int n_samples = X_new.nrow();
    
    NumericMatrix alpha_predictions(n_samples, n_classes);
    NumericMatrix mean_predictions(n_samples, n_classes);
    
    for (int i = 0; i < n_samples; i++) {
        NumericVector sample = X_new(i, _);
        
        NumericVector alpha_sum(n_classes, 0.0);
        NumericVector mean_sum(n_classes, 0.0);
        
        for (int t = 0; t < n_trees; t++) {
            XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[t]));
            List tree_pred = predict_sample_tree_fast(tree_ptr, sample);
            
            NumericVector alpha_pred = tree_pred["alpha_prediction"];
            NumericVector mean_pred = tree_pred["mean_prediction"];
            
            for (int j = 0; j < n_classes; j++) {
                alpha_sum[j] += alpha_pred[j];
                mean_sum[j] += mean_pred[j];
            }
        }
        
        for (int j = 0; j < n_classes; j++) {
            alpha_predictions(i, j) = alpha_sum[j] / n_trees;
            mean_predictions(i, j) = mean_sum[j] / n_trees;
        }
    }
    
    return List::create(
        Named("alpha_predictions") = alpha_predictions,
        Named("mean_predictions") = mean_predictions
    );
}
// ============================================================================
// UNIFIED PREDICTION FUNCTION (MODIFIED)
// ============================================================================
// [[Rcpp::export]]
List PredictDirichletForest(List forest_model, NumericMatrix X_new, 
                            std::string method = "mom",
                            bool use_leaf_predictions = true) {
    
    bool store_samples = as<bool>(forest_model["store_samples"]);
    
    if (!store_samples || use_leaf_predictions) {
        // FAST MODE: Use pre-computed leaf predictions
        return GetLeafPredictions(forest_model, X_new);
        
    } else {
        // DISTRIBUTIONAL MODE: Use weight-based approach
        return PredictDirichletForestWeightBased(forest_model, X_new, method);
    }
}
// ============================================================================
// GET SAMPLE WEIGHTS (for analysis - store_samples = TRUE only)
// ============================================================================
// [[Rcpp::export]]
List GetSampleWeights(List forest_model, NumericVector test_sample) {
    List forest_ptrs = forest_model["forest"];
    int n_trees = forest_model["n_trees"];
    NumericMatrix Y_train = forest_model["Y_train"];
    int n_classes = Y_train.ncol();
    
    std::unordered_map<int, double> weights = ComputeWeights(test_sample, forest_ptrs, n_trees);
    
    std::vector<int> sample_indices;
    std::vector<double> sample_weights;
    
    for (const auto& entry : weights) {
        sample_indices.push_back(entry.first);
        sample_weights.push_back(entry.second);
    }
    
    int n_weighted = sample_indices.size();
    NumericMatrix Y_weighted(n_weighted, n_classes);
    
    for (int i = 0; i < n_weighted; i++) {
        for (int j = 0; j < n_classes; j++) {
            Y_weighted(i, j) = Y_train(sample_indices[i], j);
        }
    }
    
    return List::create(
        Named("sample_indices") = sample_indices,
        Named("weights") = sample_weights,
        Named("Y_values") = Y_weighted
    );
}

// ============================================================================
// CLEANUP
// ============================================================================
// [[Rcpp::export]]
void delete_dirichlet_forest_rcpp(List forest_model) {
    List forest_ptrs = forest_model["forest"];
    int n_trees = forest_model["n_trees"];
    
    for (int i = 0; i < n_trees; i++) {
        XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[i]));
        Node* raw_ptr = tree_ptr.get();
        if (raw_ptr != nullptr) {
            delete raw_ptr;
            tree_ptr.release();
        }
    }
}