import rawpy
import imageio
import os

def dng_to_png(dng_path, png_path=None):
    """
    Convert a DNG image to a PNG image.

    Parameters:
        dng_path (str): Path to the input DNG file.
        png_path (str): Path to save the output PNG file. 
                        If None, the PNG will be saved in the same directory as the DNG.

    Returns:
        str: Path to the saved PNG file.
    """
    # Open and process the DNG file
    try:
        with rawpy.imread(dng_path) as raw:
            # Process the raw image
            rgb_image = raw.postprocess()

        # Define the PNG output path
        if png_path is None:
            png_path = os.path.splitext(dng_path)[0] + ".png"

        # Save as PNG
        imageio.imwrite(png_path, rgb_image)
        print(f"Converted DNG to PNG: {png_path}")
        return png_path

    except Exception as e:
        print(f"Error converting DNG to PNG: {e}")
        return None


# Example usage
dng_file = "test_burst/lamp/G0080118.dng"  # Replace with your .dng file path
output_png = "original_lamp.png"  # Replace with your desired .png output path
dng_to_png(dng_file, output_png)