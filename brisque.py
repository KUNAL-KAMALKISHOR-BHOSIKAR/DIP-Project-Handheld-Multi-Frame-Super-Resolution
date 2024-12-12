import cv2
import argparse

def compute_brisque(image_path):
    """
    Compute the BRISQUE score for an input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        float: BRISQUE score.
    """
    # Read input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Error: Could not read image.")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Load pre-trained BRISQUE model
    # model = cv2.quality.QualityBRISQUE_create()

    # # Compute BRISQUE score
    # brisque_score = model.compute(gray_image)[0]

    brisque_score = cv2.quality.QualityBRISQUE_compute(image, "models/brisque_model_live.yml", "models/brisque_range_live.yml")

    return brisque_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BRISQUE score for an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    try:
        # Compute BRISQUE score
        brisque_score = compute_brisque(args.image_path)
        print(f"BRISQUE Score: {brisque_score:.4f}")
    except Exception as e:
        print(f"Error: {e}")
