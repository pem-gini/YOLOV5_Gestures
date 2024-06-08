import math

def find_closest_rectangle(pixel, rectangles):
    """
    Find the rectangle whose center is closest to the given pixel.

    :param pixel: A tuple (x, y) representing the pixel coordinates.
    :param rectangles: A list of rectangles, where each rectangle is represented as a tuple (x1, y1, x2, y2)
                       with (x1, y1) being the top-left corner and (x2, y2) being the bottom-right corner.
    :return: The rectangle whose center is closest to the given pixel.
    """
    def center_of_rectangle(rect):
        x1, y1, x2, y2 = rect
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)

    def distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt( (x2 - x1) ** 2 + (y2 - y1) ** 2)

    pixel_x, pixel_y = pixel
    closest_rectangle = None
    min_distance = float('inf')
    index = None
    for i, stuff in enumerate(rectangles):
        rect, obj = stuff
        center = center_of_rectangle(rect)
        dist = distance((pixel_x, pixel_y), center)
        if dist < min_distance:
            min_distance = dist
            closest_rectangle = obj
            index = i

    return index, min_distance, closest_rectangle

if __name__ == "__main__":
    # Example usage
    pixel = (5, 5)
    rectangles = [(1, 1, 4, 4), ((2, 2, 8, 8), 6, 6, 10, 10)]
    closest_rect = find_closest_rectangle(pixel, rectangles)
    print("Closest rectangle:", closest_rect)