import os
from wand.image import Image

def convert_eps_to_png():
    # Get all .eps files in the current working directory
    eps_files = [f for f in os.listdir('.') if f.lower().endswith('.eps')]
    
    if not eps_files:
        print("No .eps files found in the current directory.")
        return

    print(f"Found {len(eps_files)} .eps file(s). Starting conversion...")

    for eps_file in eps_files:
        # Define the output filename by replacing .eps with .png
        png_file = os.path.splitext(eps_file)[0] + '.png'
        
        try:
            # Open the EPS file with high resolution (300 DPI) so it stays sharp
            with Image(filename=eps_file, resolution=300) as img:
                # Convert format to PNG
                img.format = 'png'
                # Save the image
                img.save(filename=png_file)
                print(f"✓ Converted: {eps_file} -> {png_file}")
        except Exception as e:
            print(f"✗ Failed to convert {eps_file}. Error: {e}")

    print("\nConversion process complete!")

if __name__ == "__main__":
    convert_eps_to_png()