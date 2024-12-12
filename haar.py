import cv2
import numpy as np
import pywt
import pywt.data

def haar_wavelet_decomposition(image):
    """
    Decompose an image using Haar wavelet transform.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        tuple: Approximation and detail coefficients (LL, LH, HL, HH).
    """
    # Apply 2D Haar Wavelet transform (one level)
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    return LL, LH, HL, HH

def compute_wavelet_energy(coeffs):
    """
    Compute the energy in each wavelet sub-band.

    Parameters:
        coeffs (tuple): The Haar wavelet coefficients (LL, LH, HL, HH).

    Returns:
        tuple: Energy in each sub-band (LL, LH, HL, HH).
    """
    LL, LH, HL, HH = coeffs
    energy_LL = np.sum(LL**2)
    energy_LH = np.sum(LH**2)
    energy_HL = np.sum(HL**2)
    energy_HH = np.sum(HH**2)
    
    return energy_LL, energy_LH, energy_HL, energy_HH

def compute_wavelet_entropy(coeffs):
    """
    Compute the entropy of each wavelet sub-band.

    Parameters:
        coeffs (tuple): The Haar wavelet coefficients (LL, LH, HL, HH).

    Returns:
        tuple: Entropy of each sub-band (LL, LH, HL, HH).
    """
    LL, LH, HL, HH = coeffs
    entropy_LL = -np.sum(LL * np.log2(LL + 1e-10))
    entropy_LH = -np.sum(LH * np.log2(LH + 1e-10))
    entropy_HL = -np.sum(HL * np.log2(HL + 1e-10))
    entropy_HH = -np.sum(HH * np.log2(HH + 1e-10))

    return entropy_LL, entropy_LH, entropy_HL, entropy_HH

def compute_haar_wavelet_metrics(image_path):
    """
    Compute Haar wavelet-based metrics for an input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        dict: Dictionary containing energy and entropy values.
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error reading image from {image_path}")
    
    # Decompose image using Haar wavelet transform
    coeffs = haar_wavelet_decomposition(image)
    
    # Compute energy in each wavelet sub-band
    energy_values = compute_wavelet_energy(coeffs)
    
    # Compute entropy in each wavelet sub-band
    entropy_values = compute_wavelet_entropy(coeffs)
    
    # Return the results as a dictionary
    metrics = {
        'Energy (LL)': energy_values[0],
        'Energy (LH)': energy_values[1],
        'Energy (HL)': energy_values[2],
        'Energy (HH)': energy_values[3],
        'Entropy (LL)': entropy_values[0],
        'Entropy (LH)': entropy_values[1],
        'Entropy (HL)': entropy_values[2],
        'Entropy (HH)': entropy_values[3]
    }
    
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Haar wavelet-based metrics for an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    try:
        # Compute Haar wavelet metrics
        metrics = compute_haar_wavelet_metrics(args.image_path)
        
        # Print the computed metrics
        print("Haar Wavelet-Based Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
