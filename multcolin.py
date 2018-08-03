import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr_matrix(in_corr_matrix):
    """
    Plots the correlation matrix as a heatmap using seaborn
    
    """
    sns.set(style = 'white')
    f, ax = plt.subplots(figsize = (12, 9))
    
    # Generate a mask for the upper triangle
    corr_mask = np.zeros_like(in_corr_matrix, dtype = np.bool)
    corr_mask[np.triu_indices_from(corr_mask)] = True
    
    ax.set_title('CORRELATION MATRIX (%s FEATURES)' %len(in_corr_matrix),
                 fontsize = 14, fontweight = 'bold')
    
    sns.heatmap(in_corr_matrix, mask = corr_mask,
                cmap = sns.diverging_palette(220, 10, as_cmap=True),
                square = True, ax = ax, 
                vmin = -1, vmax = 1)

def plot_feature_reduction_results(indf):
    """
    Plots the feature reduction results to visualize the discarded features
    
    """
    
    sns.set(style = 'whitegrid')
    f, ax = plt.subplots(figsize = (12, 9))
    
    # TODO: Show x labels if they are visible (i.e., not too many)
    ax = sns.barplot(x = 'index', y = 'corr_with_y', hue = 'Survived', 
                     palette = sns.color_palette('Paired'), data = indf)
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_title('FEATURE REDUCTION RESULTS', fontsize = 14, 
                 fontweight = 'bold')
    
    plt.xlabel('Features', fontsize = 14)
    plt.ylabel('Correlation with Target', fontsize = 14)
    #plt.legend(fontsize = 14)
    plt.show()
    
def multcolin(indf, y_vals, min_vars_to_keep, corr_tol, 
              condition_ind_tol,
              verbose,
              export_csv):
    """
    Performs feature reduction on *numeric* features using:
        1. Pairwise correlation analysis, followed by
        2. Multi-collinearity analysis
        
    min_vars_to_keep:   Stop further feature reduction if this threshold is met
    corr_tol:           If the absolute correlation between two variables is 
                        higher, one of them will be dropped (the one that has 
                        high corr with target)
    condition_ind_tol:  Used to detect high levels of multicollinearity
    
    To disable the pairwise correlation step, set corr_tol to 1.0 

    """
    
    # If empty dataframe, raise an error
    if indf.shape[0] == 0:
        raise RuntimeError('The input dataframe is empty!')
        
    # Number of input features (original)
    in_col_ct = len(indf.columns)

    # Discard categorical vars, if any
    indf = indf.loc[:, indf.dtypes != object]
    num_col_ct = len(indf.columns)
    
    # If one or less (valid) column, print message and return
    if num_col_ct <= 1:
        print ('The input dataframe contains one or less column! \
               Exiting function without executing.')
        return 

    # If number of columns <= min_vars_to_keep, raise an error
    if num_col_ct <= min_vars_to_keep:
        raise RuntimeError('The number of valid features = min_vars_to_keep. \
                            Choose a higher value for min_vars_to_keep.') 
    
    
    # If y_vals are not numeric, raise warning and return
    if type(y[0]) == 'object':
        raise ValueError('The target/output vector is non-numeric.') 
        
    ## Everything looks good -- let's proceed!
    
    # Print the number of discarded non-numeric features (if any) 
    if in_col_ct != num_col_ct and verbose == True:
        print ("%s non-numeric feature(s) discarded" %(in_col_ct - num_col_ct))
    
    # Correlation matrix for all independent vars
    corr_matrix = indf.corr()
    num_features = len(corr_matrix)
    
    # Export the initial correlation matrix for all input features
    if export_csv == True:
        corr_matrix.to_csv('initial_corr_matrix.csv')
        
    if verbose == 1:
        print (f'# of input vars = {num_features}', '\n')
    
    # Plot the initial correlation matrix
    plot_corr_matrix(corr_matrix)
    
    # Correlations with the target/output vector
    corr_with_y = {}
    for var in indf.columns:
        corr_with_y[var] = y_vals.corr(indf[var])
    
    # Save those in a dataframe
    orig_vars_df = pd.DataFrame.from_dict([corr_with_y]).T
    orig_vars_df.columns = ['corr_with_y']
        
    # For each column in the corr matrix
    print ('Running Pairwise Correlation Analysis')
    for col in corr_matrix:
        if col in corr_matrix.keys():
            this_col, these_vars = [], []
            
            for i in range(len(corr_matrix)):
                
                this_var = corr_matrix.keys()[i]
                this_corr = corr_matrix[col][i]
                
                if abs(this_corr) == 1.0 and col != this_var:
                    highly_corr = 0
                else:
                    highly_corr = (1 if abs(this_corr) > corr_tol 
                                   else -1) * abs(corr_with_y[this_var])
                
                this_col.append(highly_corr)
                these_vars.append(corr_matrix.keys()[i])
            
            # Initialize the mask
            mask = np.ones(len(this_col), dtype = bool) 
            
            # To keep track of the number of columns deleted
            del_col_ct = 0
            
            for n, j in enumerate(this_col):
                # Delete if (a) a var is correlated with others and do not have
                # the best corr with dep, or (b) completely corr with the 'col'
                mask[n] = not (j != max(this_col) and j >= 0)
                
                if j != max(this_col) and j >= 0:
                    
                    if verbose == 1:
                        print ('    Dropping %s {Corr with %s=%.5f}' 
                               %(these_vars[n], this_var, corr_matrix[col][n]))
                        
                    # Delete the column from corr matrix
                    corr_matrix.pop('%s' %these_vars[n])
                    corr_with_y.pop('%s' %these_vars[n])
                    del_col_ct += 1
                    
            # Delete the corresponding row(s) from the corr matrix
            corr_matrix = corr_matrix[mask]
    
    if verbose == 1 and corr_tol != 1:
        print ('\n# of vars after eliminating high pairwise correlations =', 
               len(corr_matrix), '\n')
    
    # Multicollinearity
    if num_features > min_vars_to_keep:
        print ('Running Multi-collinearity Analysis')
    
        while True:
            num_features -= 1
            
            # Update the list of columns
            cols = corr_matrix.keys() 
            
            # Eigen values and vectors
            eigen_vals, eigen_vectors = np.linalg.eig(corr_matrix) 
            
            # Calculate the max of all conditinon indices
            c_ind = max((max(eigen_vals) / eigen_vals) ** 0.5)
            
            # If the condition index <= 30 then multicolin is not an issue
            if c_ind <= condition_ind_tol or num_features == min_vars_to_keep:
                break
            
            for i, val in enumerate(eigen_vals):
                if val == min(eigen_vals):   # Min value, close to zero
                    # Look into that vector
                    this_eigen_vector = eigen_vectors[:, i]
                    max_w = max(abs(this_eigen_vector))
                    
                    for j, vec in enumerate(this_eigen_vector):
                        # Var that has the max weight on that vector
                        if abs(vec) == max_w:
                            # Initialize
                            mask = np.ones(len(corr_matrix), dtype = bool)
                            for n, col in enumerate(corr_matrix.keys()):
                                mask[n] = n != j
                                
                            #TODO: Also print the set of features
                            # with which this var is correlated
                            if verbose == 1:
                                print ('    Dropping %s {Weight=%.2f}' 
                                       %(corr_matrix.keys()[j], max_w))
                                
                            # Delete row
                            corr_matrix = corr_matrix[mask]  
                            # Delete column
                            corr_matrix.pop(cols[j])
                                                     
        
        if verbose == 1:
            print ('\n # of vars after multicolinearity analyis =', 
                   len(corr_matrix), '\n')
    
    # Export the final correlation matrix for the survivors
    if export_csv == True:
        corr_matrix.to_csv('final_corr_matrix.csv')
        
    # Survivors
    surv_vars = {k:v for (k,v) in corr_with_y.items() 
                if k in corr_matrix.keys()}
    
    # Create a dataframe (to be exported)
    surv_vars_df = pd.DataFrame.from_dict([surv_vars]).T
    surv_vars_df.columns = ['Survived']
    
    orig_vars_df = orig_vars_df.merge(surv_vars_df, left_index = True, 
                                      right_index = True, how = 'left')
    
    orig_vars_df.loc[pd.isnull(orig_vars_df['Survived']) == False, 
                     'Survived'] = 'Y'
                     
    orig_vars_df['Survived'].fillna('N', inplace = True)
    
    orig_vars_df = orig_vars_df.reset_index().sort_values(by = 'corr_with_y', 
                                           ascending = False)

    if export_csv == True:
        orig_vars_df.to_csv('multcolin_results.csv')
    
    # Plot the final correlation matrix
    plot_corr_matrix(corr_matrix)
     
    # Plot the feature reduction results
    plot_feature_reduction_results(orig_vars_df)
    
    return surv_vars


if __name__ == "__main__":
    
    # For the test, we will use the boson housing data
    from sklearn.datasets import load_boston
    boston = load_boston()
    print(boston.data.shape)
    # 506, 13
    
    # the X values must be a dataframe with column names
    # y must be a series
    # TODO: Accept other formats (i.e., numpy arrays)
    X, y = pd.DataFrame(boston['data']), pd.Series(boston['target'])
    X.columns = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B1000', 'LSTAT', 'MEDV']
    
    # This data doesn't suffer from multi-collinearity
    # So let's introduce it
    X['EXTRA'] = .01*X['RAD'] + .02*X['ZN'] - .003*X['CRIM']    \
        + np.random.uniform()
    
    vars_to_keep = multcolin(X, y, 
                             min_vars_to_keep = 5, 
                             corr_tol = .95, 
                             condition_ind_tol = 5,
                             verbose = 1, 
                             export_csv = 1)
    
    # WARNING: I've used condition_ind_tol = 5 just for this demonstration
    # This is TOO LOW! In practice, you should use 30 

