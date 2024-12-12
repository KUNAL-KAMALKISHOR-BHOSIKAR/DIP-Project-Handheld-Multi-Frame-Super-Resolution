import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows
import scipy.io

def compute_niqe(image, patch_size=6):
    """
    Compute the NIQE score for an input image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        patch_size (int): Size of the patches for NIQE computation.

    Returns:
        float: NIQE score.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0

    # Apply Gaussian filter to compute local mean and variance
    mu = gaussian_filter(image, sigma=7/6, truncate=3)
    sigma_squared = gaussian_filter(image**2, sigma=7/6, truncate=3) - mu**2
    sigma = np.sqrt(np.maximum(sigma_squared, 0))

    # Normalize the image
    normalized = (image - mu) / (sigma + 1e-8)

    # Extract patches
    patches = view_as_windows(normalized, (patch_size, patch_size), step=patch_size).reshape(-1, patch_size**2)

    # Compute mean and covariance of patches
    patch_mean = np.mean(patches, axis=0)
    patch_cov = np.cov(patches, rowvar=False)

    model_mat = scipy.io.loadmat('modelparameters.mat')
    model_mu = model_mat['mu_prisparam'].flatten()
    model_cov = model_mat['cov_prisparam']

    # Pre-trained NIQE model parameters (modify as per actual model values)
    # model_mu = np.zeros_like(patch_mean)  # Mean of pristine patches
    # model_cov = np.eye(patch_size**2)    # Covariance of pristine patches

    # Compute Mahalanobis distance
    diff = patch_mean - model_mu
    inv_model_cov = np.linalg.inv(model_cov + np.eye(patch_size**2) * 1e-8)
    niqe_score = np.sqrt(np.dot(diff.T, np.dot(inv_model_cov, diff)))

    return niqe_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute NIQE score for an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    # Read input image
    input_image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        print("Error: Could not read image.")
        exit()

    # Compute NIQE score
    niqe_score = compute_niqe(input_image)
    print(f"NIQE Score: {niqe_score:.4f}")
