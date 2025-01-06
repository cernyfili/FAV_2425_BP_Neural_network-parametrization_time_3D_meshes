class ClusteredCenterPointsAllFrames:
    """
    Class to store clustered center points for all frames.
    points_allframes: np.ndarray of shape (num_time_steps, num_points_in_file, 3)
    labels_frame: np.ndarray of shape (num_points_in_file,) labels for each index of centers in one frame which same
        for all frames
    """
    def __init__(self, points_allframes, labels_frame):
        self.points_allframes = points_allframes
        self.labels_frame = labels_frame
        # check if shapes are the same
        if points_allframes.shape[1] != labels_frame.shape[0]:
            raise ValueError("Shapes of points and labels do not match.")

    def get_points_from_time_step(self, time_step):
        return self.points_allframes[time_step].reshape(-1, 3)
