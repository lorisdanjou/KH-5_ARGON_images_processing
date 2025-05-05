import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, tile="00", fraction=3, title=None, cmap='gray', rescale=False):
    '''
    Plot a tile of an image with a given fraction and title.
    Input:
    - image: 2D numpy array representing the image to be plotted.
    - tile: string representing the tile to be plotted. Default is "00".
    - fraction: integer representing the fraction (of a side) of the image to be plotted. Default is 3.
    - title: string representing the title of the plot. Default is None.
    - cmap: string representing the colormap to be used. Default is 'gray'.
    Output:
    - offset_x: integer representing the x offset of the tile in pixels.
    - offset_y: integer representing the y offset of the tile in pixels.
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    
    shy, shx = image.shape
    offset_x, offset_y = int(tile[0]) * shx // fraction, int(tile[1]) * shy // fraction
    limit_x, limit_y = (int(tile[0]) + 1) * shx // fraction, (int(tile[1]) + 1) * shy // fraction
    print("(offset_x, offset_y) : ", offset_x, offset_y)
    
    image_tile = image[offset_y:limit_y, offset_x:limit_x]
    
    if rescale:
        image_tile = (image_tile - np.min(image_tile)) / (np.max(image_tile) - np.min(image_tile))

    
    ax.imshow(image_tile, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("X - offset_x (pixels)")
    ax.set_ylabel("Y - offset_y (pixels)")
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    
    return offset_x, offset_y