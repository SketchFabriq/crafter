import replicate
import base64
import re
from PIL import Image, ImageDraw


def call_molmo(file_path, prompt):
    """
    Call the Molmo model with the given image and prompt.
    """
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_str = f"data:application/octet-stream;base64,{encoded_image}"

    input = {
        "text": prompt,
        "image": image_str
    }
    print("Input prepared, Sending request to Replicate API molmo")
    output = replicate.run(
        "zsxkib/molmo-7b:76ebd700864218a4ca97ac1ccff068be7222272859f9ea2ae1dd4ac073fa8de8",
        input=input
    )
    print("Response received")
    return output


def draw_point_on_image(image_path, out, x, y):
    """
    Draw a point on the image at the specified coordinates.
    """

    # Open the image
    image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    width, height = image.size
    x = int(x * width / 100)
    y = int(y * height / 100)
    # Draw the point
    # Adjust the size of the point as needed
    point_size = 5
    draw.ellipse((x - point_size, y - point_size, x + point_size, y + 
                  point_size), fill="black")
    # Save the modified image
    image.save(out)


def draw_multiple_points_on_image(image_path, out, points):
    """
    Draw multiple points on the image at the specified coordinates.
    
    Args:
        image_path: Path to the input image
        out: Path to save the output image
        points: List of (x, y) coordinate tuples (values between 0 and 100)
    """
    # Open the image
    image = Image.open(image_path)
    
    # Create a draw object
    draw = ImageDraw.Draw(image)
    
    width, height = image.size
    
    # Draw each point
    point_size = 5
    for x, y in points:
        # Convert percentage coordinates to absolute coordinates
        abs_x = int(x * width / 100)
        abs_y = int(y * height / 100)
        
        # Draw the point
        draw.ellipse(
            (abs_x - point_size, abs_y - point_size, 
             abs_x + point_size, abs_y + point_size), 
            fill="black"
        )
    
    # Save the modified image
    image.save(out)


if __name__ == "__main__":
    # Example usage
    file_path = "resources/test1.jpg"
    prompt = "Point to all the blocks in the left part of the image "
    result = call_molmo(file_path, prompt)

    matches = re.findall(r'x\d+="([\d.]+)" y\d+="([\d.]+)"', result)
    # Convert to list of tuples of floats
    coordinates = [(float(x), float(y)) for x, y in matches]

    print(coordinates)
    print(result)
    draw_multiple_points_on_image(file_path, "out.png", coordinates)
    # draw_point_on_image(file_path, 15.7, 56.1)
    # print(result)