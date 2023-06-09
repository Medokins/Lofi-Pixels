from PIL import Image
import os
import shutil


def pixelate(name, block_size):
    image_path = os.path.join("data", "to_convert", f"{name}.jpg")
    # like a 2nd bin, to move images after processing them but not deleting
    temp_image_path = os.path.join("data", "temp", f"{name}.jpg")
    input_image = Image.open(image_path)
    # so it fits GAN model
    input_image = input_image.resize((512, 512))
    width, height = input_image.size

    num_blocks_x = width // block_size
    num_blocks_y = height // block_size

    output_image = Image.new('RGB', (width, height))

    for block_x in range(num_blocks_x):
        for block_y in range(num_blocks_y):
            x_start = block_x * block_size
            y_start = block_y * block_size
            x_end = x_start + block_size
            y_end = y_start + block_size

            block_pixels = []
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    pixel = input_image.getpixel((x, y))
                    block_pixels.append(pixel)

            average_color = tuple(int(sum(channel) / len(block_pixels)) for channel in zip(*block_pixels))

            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    output_image.putpixel((x, y), average_color)

    output_image.save(os.path.join("data", "converted", f'pixelated_{name}_{block_size}.png'))
    shutil.move(image_path, temp_image_path)


def convert_all(block_size):
    data_dir_path = os.path.join("data", "to_convert")
    for file_name in os.listdir(data_dir_path):
        if file_name.endswith(".jpg"):
            name = os.path.splitext(file_name)[0]
            pixelate(name, block_size)
