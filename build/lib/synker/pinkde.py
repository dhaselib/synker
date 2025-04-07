import numpy as np
import pandas as pd



def pinkde(X, Y, hx, hy, bandwidth_method, grid_x, grid_y, res, min_val, max_val):
    """
    Computes the probability kernel density estimation (PKDE) for given data and selects the optimal bandwidth.

    Parameters:
    -----------
    X : array-like
        Input data for the first variable.
    Y : array-like
        Input data for the second variable.
    hx : float or None
        Bandwidth for the first variable. If None, it will be computed using the specified bandwidth method.
    hy : float or None
        Bandwidth for the second variable. If None, it will be computed using the specified bandwidth method.
    bandwidth_method : str or None
        Method for bandwidth selection. Choose from:
        - "Scott" : Uses Scott’s rule to determine bandwidth.
        - "Silverman" : Uses Silverman’s rule to determine bandwidth.
        If None, the user must provide hx and hy.
    grid_x : array-like or None
        Grid points for the first variable. If None, it is generated using `np.linspace(min(X), max(X), res)`.
    grid_y : array-like or None
        Grid points for the second variable. If None, it is generated using `np.linspace(min(Y), max(Y), res)`.
    res : int
        Number of points in the grid along each axis.
    min_val : float
        Minimum threshold for normalized PKDE values when filtering data.
    max_val : float
        Maximum threshold for normalized PKDE values when filtering data.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing:
        - 'X': Filtered X values based on the specified PKDE range.
        - 'Y': Corresponding Y values.
        - 'index': Original indices of the filtered data points in the input dataset.

    The function filters data points whose normalized PKDE values lie within the range [min_val, max_val].
    """

    # Bandwidth selection
    if bandwidth_method is not None:
        if bandwidth_method.lower() == "scott":
            hx = Scott(X)
            hy = Scott(Y)
        elif bandwidth_method.lower() == "silverman":
            hx = Silverman(X)
            hy = Silverman(Y)
        else:
            raise ValueError("Invalid bandwidth_method. Choose 'Scott' or 'Silverman'.")
    else:
        if hx is None or hy is None:
            raise ValueError("Provide hx and hy or set bandwidth_method.")
    
    # Grid generation if not provided
    if grid_x is None:
        grid_x = np.linspace(min(X), max(X), res)
    if grid_y is None:
        grid_y = np.linspace(min(Y), max(Y), res)

    pkde = kde(X, Y, X, Y, hx, hy)
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'pkde': np.diag(pkde), 'index': np.arange(len(X))})
    sorted_df = df.sort_values(by='pkde').reset_index(drop=True)
    sorted_df['normalized_pkde'] = (sorted_df['pkde'] - sorted_df['pkde'].min()) / (sorted_df['pkde'].max() - sorted_df['pkde'].min())

    def query_data(min_val, max_val):
        filtered_df = sorted_df[(sorted_df['normalized_pkde'] >= min_val) & (sorted_df['normalized_pkde'] <= max_val)]
        return filtered_df[['X', 'Y', 'index']]

    result = query_data(min_val, max_val)
    return result
