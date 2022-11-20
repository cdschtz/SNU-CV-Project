# TODO:
# - Save box (coordinate) and label predictions of each frame to separate (e.g., .txt) file
# - Use frame and info.txt files as input for tracker class
# - Output (text as PoC):
#   - Independent trajectory id, label, timestamp, and coordinate for each object
#   - Verify label structure from weights of detector model
#   - Important labels: car, truck, bus, person?, motorcycle

# Detection dataset file structure example (taken from MOT Challenge):
# 1, -1, 794.2, 47.5, 71.2, 174.8, 67.5, -1, -1
# ...
# -> 1: frame number
# -> 2: object id (belonging to trajectory, set to -1 in a detection file as no ID assigned yet)
# -> 3-6: position of the bounding box (top-left corner, width, height)
# -> 7: confidence score
# -> 8: Ground Truth: indicates the type of object annotated
# -> 9: Ground Truth: indicates the visibility of the object (number between 0 and 1)

# Idea differentiate stationary vs. moving objects