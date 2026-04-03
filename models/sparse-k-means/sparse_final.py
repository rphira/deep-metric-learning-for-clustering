import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score
from scipy.optimize import bisect
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score
from scipy.optimize import bisect

def soft_threshold(x, delta):
    """Soft-thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0)

def compute_cer(true_labels, cluster_labels):
    """Compute classification error rate"""
    cm = confusion_matrix(true_labels, cluster_labels)
    return 1 - np.trace(cm) / np.sum(cm)

def sparse_kmeans(X, K, s, true_labels=None, max_iter=1000, tol=1e-6, patience=50, min_iter=10, random_state=42):
    """
    Sparse K-means clustering following Witten & Tibshirani (2010)
    
    Parameters:
    - X: numpy array of shape (n, p)
    - K: int, number of clusters
    - s: float, L1 bound on weights (1 <= s <= sqrt(p))
    - true_labels: optional true labels for evaluation
    - max_iter: int, maximum iterations
    - tol: float, convergence tolerance
    - random_state: random seed
    
    Returns:
    - cluster_labels: array of cluster assignments
    - weights: array of feature weights
    - metrics: dictionary of performance metrics
    """
    n, p = X.shape
    
    # validate parameters
    if s < 1 or s > np.sqrt(p):
        raise ValueError(f"s must be between 1 and sqrt(p)={np.sqrt(p):.2f}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # weight initialization
    weights = np.ones(p) / np.sqrt(p)
    objective_history = []
    weight_history = [weights.copy()]

    # Early stopping variables
    best_weights = weights.copy()
    best_objective = -np.inf
    patience_counter = 0
    converged = False
    print(f"{n} samples, {p} features, s={s}")
    
    for iteration in range(max_iter):
        # Weighted K-means clustering
        weighted_X = X_scaled * np.sqrt(weights)
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(weighted_X)
        
        # Compute BCSS
        BCSS = np.zeros(p)
        for j in range(p):
            # Total sum of squares for feature j
            overall_mean = np.mean(X_scaled[:, j])
            total_ss = np.sum((X_scaled[:, j] - overall_mean) ** 2)
            
            # Within cluster sum of squares for feature j
            within_ss = 0
            for k in range(K):
                cluster_indices = (cluster_labels == k)
                n_k = np.sum(cluster_indices)
                if n_k > 0:  # Avoid empty clusters
                    cluster_mean = np.mean(X_scaled[cluster_indices, j])
                    within_ss += np.sum((X_scaled[cluster_indices, j] - cluster_mean) ** 2)
            
            # BCSS = Total SS - Within SS
            BCSS[j] = (total_ss - within_ss)
        
        # Update weights with L1 constraint enforcement
        a = BCSS
        
        # Scale BCSS to unit norm for numerical stability
        a = a / (np.linalg.norm(a, 2) + 1e-10)
        
        # Binary search for delta to satisfy ||w||₁ = s
        def l1_norm(delta):
            w_temp = soft_threshold(a, delta)
            norm_w = np.linalg.norm(w_temp, 2)
            if norm_w > 0:
                w_temp = w_temp / norm_w
            return np.sum(np.abs(w_temp)) - s
        
        # Check if constraint is already satisfied
        current_l1 = np.sum(np.abs(a / (np.linalg.norm(a, 2) + 1e-10)))
        
        if current_l1 <= s:
            delta = 0
            w_new = a
        else:
            # Binary search for optimal delta
            delta_low, delta_high = 0, np.max(np.abs(a))
            delta = bisect(l1_norm, delta_low, delta_high, xtol=1e-8, maxiter=50)
            w_new = soft_threshold(a, delta)
        
        # Normalize final weights
        norm_w = np.linalg.norm(w_new, 2)
        if norm_w > 0:
            w_new = w_new / norm_w
        else:
            w_new = np.ones(p) / np.sqrt(p)
        
        # Calculate objective
        current_objective = np.sum(w_new * BCSS)
        objective_history.append(current_objective)
        weight_history.append(w_new.copy())

        # Check for improvement
        if current_objective > best_objective:
            best_objective = current_objective
            best_weights = w_new.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Check convergence
        weight_change = np.linalg.norm(w_new - weights, 2)
        n_nonzero = np.sum(w_new > 1e-6)
        
        print(f"Iteration {iteration + 1}:")
        print(f"  - BCSS range: [{BCSS.min():.6f}, {BCSS.max():.6f}]")
        print(f"  - Weight change: {weight_change:.6f}")
        print(f"  - Objective: {current_objective:.6f}")
        print(f"  - Non-zero features: {n_nonzero}/{p} ({n_nonzero/p*100:.1f}%)")
        
        # Early stopping conditions
        if weight_change < tol and iteration >= min_iter:
            print(f"converged after {iteration + 1} iterations (weight change < {tol})")
            converged = True
            break
        elif patience_counter >= patience and iteration >= min_iter:
            print(f"early stopping after {iteration + 1} iterations (no improvement for {patience} iterations)")
            converged = True
            break
            
        weights = w_new
    if not converged:
            print(f"reached maximum iterations ({max_iter})")
    
    # Use best weights found
    final_weights = best_weights
    final_n_nonzero = np.sum(final_weights > 1e-6)
    
    print(f"Final: {final_n_nonzero} non-zero features, Best objective: {best_objective:.6f}")
    
    return cluster_labels, final_weights, objective_history
    
    # Compute final metrics
    # metrics = {}
    # if true_labels is not None:
    #     metrics['CER'] = compute_cer(true_labels, cluster_labels)
    #     metrics['ARI'] = adjusted_rand_score(true_labels, cluster_labels)
    
    # metrics['Silhouette'] = silhouette_score(X_scaled, cluster_labels)
    # metrics['Nonzero_Features'] = np.sum(weights > 1e-6)
    # metrics['Sparsity_Ratio'] = np.sum(weights > 1e-6) / p
    # metrics['Iterations'] = len(objective_history)
    # metrics['Final_Objective'] = current_objective
    
    # print(f"\nFINAL RESULTS")
    # print(f"Selected {metrics['Nonzero_Features']} out of {p} features")
    # if true_labels is not None:
    #     print(f"ARI: {metrics['ARI']:.3f}, CER: {metrics['CER']:.3f}")
    # print(f"Silhouette: {metrics['Silhouette']:.3f}")
    
def find_optimal_sparsity(X, K, true_labels=None, sparsity_range=None, 
                         max_iter=50, random_state=42, metric='ari'):
    """
    Find optimal sparsity parameter s for sparse K-means
    
    Parameters:
    - X: feature matrix
    - K: number of clusters
    - true_labels: true labels for evaluation (optional)
    - sparsity_range: range of s values to try
    - max_iter: maximum iterations per run
    - random_state: random seed
    - metric: evaluation metric ('silhouette', 'ari', or 'objective')
    
    Returns:
    - best_s: optimal sparsity parameter
    - best_score: best score achieved
    - results: dictionary with all results
    """
    n, p = X.shape
    
    # set default sparsity range if not provided
    if sparsity_range is None:
        sparsity_range = np.linspace(1.0, min(10.0, np.sqrt(p)), 8)
    
    results = {}
    best_score = -np.inf
    best_s = None
    best_labels = None
    best_weights = None
    
    print(f"Searching for optimal sparsity in range: {sparsity_range}")
    print(f"Using metric: {metric}")
    
    for s in sparsity_range:
        print(f"\ntesting s={s:.2f}")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                labels, weights, objectives = sparse_kmeans(
                    X=X, K=K, s=s, true_labels=true_labels,
                    max_iter=max_iter, random_state=random_state,
                    patience=50, min_iter=5
                )
            
            # Calculate evaluation metric
            if metric == 'silhouette':
                from sklearn.metrics import silhouette_score
                X_scaled = StandardScaler().fit_transform(X)
                score = silhouette_score(X_scaled, labels)
            elif metric == 'ari' and true_labels is not None:
                score = adjusted_rand_score(true_labels, labels)
            elif metric == 'objective':
                score = objectives[-1] if objectives else -np.inf
            else:
                # Default to silhouette
                X_scaled = StandardScaler().fit_transform(X)
                score = silhouette_score(X_scaled, labels)
            
            n_nonzero = np.sum(weights > 1e-6)
            results[s] = {
                'score': score,
                'n_nonzero': n_nonzero,
                'weights': weights,
                'labels': labels,
                'objectives': objectives
            }
            
            print(f"s={s:.2f}: {metric}={score:.4f}, non-zero features={n_nonzero}")
            
            if score > best_score:
                best_score = score
                best_s = s
                best_labels = labels
                best_weights = weights
                
        except Exception as e:
            print(f"failed for s={s:.2f}: {e}")
            results[s] = None
    
    print(f"\ns={best_s:.2f}, {metric}={best_score:.4f}")
    print(f"selected {np.sum(best_weights > 1e-6)} non-zero features")
    
    return best_s, best_score, best_labels, best_weights, results