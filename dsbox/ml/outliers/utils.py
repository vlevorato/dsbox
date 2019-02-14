import numpy as np


def mad(x):
    """
    Median Absolute Deviation
    """
    return np.median(np.abs(x - np.median(x)))


def mad_outliers(x, cutoff=2):
    """
    Median Absolute Deviation outliers method.
    
    It returns values as outliers if their residuals relative to the median exceeds 'n' times the ratio
    to the MAD. The cutoff parameter represents this 'n' times quantity.
    
    Main code: x_mad = np.abs(x - np.median(x)) / mad_value
    if x_mad contains a value above the cutoff parameter, it is considered as an outlier.
    
    Parameters
    ----------
    x : array-like
    
    cutoff : int, optional (default=2)
        amount of times residuals relative to the median exceed the ratio to the MAD
    
    Returns
    -------
    Boolean array with outlier tag

    """
    mad_value = mad(x)
    if mad_value == 0:
        raise ZeroDivisionError("Median Absolute Deviation is equal to zero.")

    x_mad = np.abs(x - np.median(x)) / mad_value
    return x_mad > cutoff


def fft_outliers(x, freq_cut_index=None, outlier_proportion=0.1):
    """
    Fast Fourier Transformation outliers method.
    
    The idea is to isolate high frequencies in frequency domain, i.e. fast variations. Isolating such frequencies
    allows to identify abrupt change in numerical series, which are considered here as outliers.
    
    Parameters
    ----------
    x : array-like
    
    freq_cut_index : int, optional (default=None)
        set the index in the FFT numpy array beyond which the frequency is considered to be filtered. The highest,
        the less filtering. By default, it sets this index to 0.9 of x length
        
    outlier_proportion : float, optional (default=0.1)
        set the proportion of outliers to return

    Returns
    -------
    Boolean array with outlier tag
    
    """
    if freq_cut_index is None:
        freq_cut_index = int(0.9 * float(len(x)))

    fft_x = np.fft.fft(x)
    for freq_index in range(0, len(fft_x)):
        if freq_index >= freq_cut_index:
            fft_x[freq_index] = 0

    ifft_x = np.real(np.fft.ifft(fft_x).round())
    diff_fft_x = np.abs(x - ifft_x)
    outlier_amount = int(len(x) * outlier_proportion)
    outliers = np.zeros(len(x), dtype='bool')
    for i in range(0, outlier_amount):
        position_outlier = np.argmax(diff_fft_x)
        outliers[position_outlier] = True
        diff_fft_x[position_outlier] = -1

    return outliers
