import numpy as np

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def mean_absolute_deviation(x):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.abs(x - np.mean(x)))


def median_absolute_deviation(x):
    """
    Median Absolute Deviation
    """
    return np.median(np.abs(x - np.median(x)))


def double_median_absolute_deviation(x):
    """
    If the underlying distribution of the data is unsymmetric, one should use the double median 
    absolute deviation, instead of the classic mad.
    
    Parameters
    ----------
    x : array-like

    Returns
    -------
    Couple of values: left mad, right mad

    """
    median = np.median(x)
    absolute_deviation = np.abs(x - median)

    left_mad = np.median(absolute_deviation[absolute_deviation <= median])
    right_mad = np.median(absolute_deviation[absolute_deviation >= median])

    return left_mad, right_mad


def mad_outliers(x, cutoff=2, z_score_coeff=0.6745):
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
        
    z_score_coeff : float, optional (default=0.6745)
        0.6745 is the 0.75th quartile of the standard normal distribution, to which the MAD converges to.
    
    Returns
    -------
    Boolean array with outlier tag

    """
    mad_value = median_absolute_deviation(x)
    if mad_value == 0:
        return x * np.nan

    x_mad = (z_score_coeff * np.abs(x - np.median(x))) / mad_value
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
