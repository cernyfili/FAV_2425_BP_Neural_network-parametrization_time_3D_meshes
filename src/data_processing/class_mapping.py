from typing import List

import numpy as np


class SurfacePointsFrameList:
    def __init__(self, surface_data_list: list):
        if not isinstance(surface_data_list, list) or not all(
                isinstance(item, SurfacePointsFrame) for item in surface_data_list):
            raise TypeError("All items in surface_data_list must be instances of SurfaceData")
        self.list = surface_data_list
        # Initialize unique_clusters at creation
        self.max_time_index = None

    def assign_time_to_all_elements(self):
        """Assign a time index to each surface data item if not already assigned."""
        for i, surface_data in enumerate(self.list):
            if surface_data.time is not None:
                raise Exception("Time already added to surface data.")

            surface_data.time = TimeFrame(i, i)

    def get_all_points(self):
        """
        Return all points in the list.
        """
        all_points = []
        for surface_data in self.list:
            points_list = surface_data.points_list
            # add to all_points all elements of points_list
            all_points.extend(points_list)
        return all_points

    def normalize_all_elements(self):
        # todo test if it is working
        """
        Normalize the surface points for all objects in the list.
        Normalize the data to the range [0, 1] for each axis and shift it to the origin (0, 0, 0).
        :return: normalized_surface_points:
        """
        import numpy as np

        def normalize_time(surface_data_list):
            total_length = len(surface_data_list.list) - 1
            for surface_data in surface_data_list.list:
                surface_data.time.value /= total_length

        def compute_shift_and_scale(surface_data_list : SurfacePointsFrameList):
            # Combine all points for faster computation
            all_points_list = surface_data_list.get_all_points()
            all_points = np.vstack([all_points_list])
            min_corner = np.min(all_points, axis=0)
            max_corner = np.max(all_points, axis=0)
            shift_vector = (min_corner + max_corner) / 2
            max_norm = np.linalg.norm(all_points - shift_vector, axis=1).max()
            return shift_vector, max_norm

        def shift_and_scale_points(surface_data_list, shift_vector, max_norm):
            for surface_data in surface_data_list.list:
                value = (surface_data.points_list - shift_vector) / max_norm
                surface_data.points_list = value

        # Normalize time for each object
        normalize_time(self)

        # Compute the shift vector and max norm
        shift_vector, max_norm = compute_shift_and_scale(self)

        # Shift points to origin and scale
        shift_and_scale_points(self, shift_vector, max_norm)

        return self

    def get_unique_times(self):
        """
        Return the set of unique times.
        """

        return {surface_data.time.value for surface_data in self.list}



    def get_element_by_time_index(self, time_index):
        """
        Find the element in the list with the specified time index.
        """

        def filter_by_time_index(self, time_index : int):
            """
            :param time_index:
            :return:
            """
            filtered_data = []
            for surface_data in self.list:
                if surface_data.time.index == time_index:
                    filtered_data.append(surface_data)

            return SurfacePointsFrameList(filtered_data)

        filtered_list = filter_by_time_index(self, time_index)
        if len(filtered_list.list) == 0:
            return None
        if len(filtered_list.list) > 1:
            raise ValueError("Multiple elements found with the same time index.")

        return filtered_list.list[0]

    def get_element_by_time_value(self, time_index):
        """
        Find the element in the list with the specified time index.
        """
        def filter_by_time_value(self, time_value):
            """
            :param time_value:
            :return:
            """
            filtered_data = []
            for surface_data in self.list:
                if surface_data.time.value == time_value:
                    filtered_data.append(surface_data)

            return SurfacePointsFrameList(filtered_data)

        filtered_list = filter_by_time_value(self, time_index)
        if len(filtered_list.list) == 0:
            return None
        if len(filtered_list.list) > 1:
            raise ValueError("Multiple elements found with the same time index.")

        return filtered_list.list[0]

    def get_cluster_labels(self):
        """
        Return the list of cluster labels.
        """
        return [label for surface_data in self.list for label in surface_data.labels_list]



    def get_unique_clusters(self):
        """
        Return the set of unique clusters.
        """


        def compute_unique_clusters(self : SurfacePointsFrameList):
            """
            Private method to compute unique clusters from the surface data list.
            """
            unique_clusters = set()
            for surface_data in self.list:
                if surface_data.labels_list is None or not surface_data.labels_list or surface_data.labels_list[0] == None:
                    raise ValueError("Labels list is empty.")

                unique_clusters.update(
                    int(label) for label in surface_data.labels_list)  # Convert each sub-array to a tuple
            return unique_clusters

        self.unique_clusters = compute_unique_clusters(self)

        if self.unique_clusters is None or not self.unique_clusters:
            raise Exception("Unique clusters is Empty")

        return self.unique_clusters

    def append(self, surface_data):
        """
        Append a SurfaceData object to the list and update unique clusters.
        """
        if not isinstance(surface_data, SurfacePointsFrame):
            raise TypeError("surface_data must be an instance of SurfaceData")
        self.list.append(surface_data)
        if surface_data.time is not None:
            self.max_time_index = max(self.max_time_index, surface_data.time.index)  # Update unique clusters

    def filter_by_label(self, label_index):
        """
        Filter the  by the given label index, keeping only the corresponding surface points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        -  instance containing only the SurfaceData objects with the specified label index in both
          surface_labels_list and surface_points_list
        """
        filtered_data = []

        for surface_data in self.list:
            # Convert labels_list to a numpy array for efficient filtering
            labels_array = np.array(surface_data.labels_list)
            points_array = np.array(surface_data.points_list)

            # Find indices where the label matches the specified label index
            matching_indices = np.where(labels_array == label_index)[0]
            #
            # if matching_indices.size == 0:
            #     raise ValueError("No points with the specified label index found.")

            # Use the indices to filter points and labels
            filtered_points = points_array[matching_indices].tolist()
            filtered_labels = labels_array[matching_indices].tolist()

            # Create a new SurfaceData instance with the filtered points and labels
            filtered_data.append(SurfacePointsFrame(filtered_points, filtered_labels, surface_data.time))

        return SurfacePointsFrameList(filtered_data)

    def slice_arrays(self, num_of_elements):
        self.list = [surface_data.slice_arrays(num_of_elements) for surface_data in self.list]
        return self

    def select_random_points(self, num_points):
        """
        Select a random subset of points from each SurfaceData object in the list.

        Parameters:
        - num_points: int, the number of points to select

        Returns:
        -  instance containing only the selected points
        """
        if num_points <= 0:
            raise ValueError("Number of points must be greater than 0.")
        if num_points >= len(self.list[0].points_list):
            raise ValueError("Number of points must be less than the total number of points.")

        selected_data = []

        for surface_data in self.list:
            # Randomly select a subset of points
            selected_indices = np.random.choice(len(surface_data.points_list), num_points, replace=False)
            selected_points = [surface_data.points_list[i] for i in selected_indices]
            selected_labels = [surface_data.labels_list[i] for i in selected_indices]

            # Create a new SurfaceData instance with the selected points
            selected_data.append(SurfacePointsFrame(selected_points, selected_labels, surface_data.time))

        return SurfacePointsFrameList(selected_data)

    def get_time_list(self):
        """
        Return the list of time frames.
        """
        return [surface_data.time for surface_data in self.list]


class TimeFrame:
    """
    Class represents time for one frame from sequence
    """

    def __init__(self, index: int, value: float = None):
        self.index = index
        self.value = value

    def __repr__(self):
        return f"TimeFrame(index={self.index}, value={self.value})"


def time_frame_list_find_closest_element_index(time_frame_list: List[TimeFrame], time_value: float) -> int:
    if time_value == 0.0:
        return 0

    epsilon = (1 / len(time_frame_list)) / 4
    closest_list = []
    for time_frame in time_frame_list:
        if abs(time_frame.value - time_value) < epsilon:
            return time_frame.index

    if len(closest_list) != 1:
        raise ValueError("Multiple elements found with the same time index.")

    return closest_list[0].index

class LabeledPoint:
    """
    Class to represents point in object for one cluster and for a single time step.
    """

    def __init__(self, point : list, label : int = None):
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")

        self.point = point
        self.label = label

    def __repr__(self):
        return f"LabeledPoint(point={self.point}, label={self.label})"


class LabeledPointsList:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, labeled_points_list: list):
        self.list = labeled_points_list

    def append(self, labeled_point):
        """
        Append a LabeledPoint object to the list.
        """
        if not isinstance(labeled_point, LabeledPoint):
            raise TypeError("labeled_point must be an instance of LabeledPoint")
        self.list.append(labeled_point)

    def filter_by_label(self, label_index):
        """
        Filter the LabeledPointsList by the given label index, keeping only the corresponding points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        - LabeledPointsList instance containing only the LabeledPoint objects with the specified label index
        """
        filtered_data = []

        for labeled_point in self.list:
            if labeled_point.label == label_index:
                filtered_data.append(labeled_point)

        return LabeledPointsList(filtered_data)

    def get_points(self):
        """
        Return the list of points.
        """
        points_list = []
        for labeled_point in self.list:
            points_list.append(labeled_point.point)
        return points_list

    def get_labels(self):
        """
        Return the list of labels.
        """
        return [labeled_point.label for labeled_point in self.list]

    def __repr__(self):
        return f"LabeledPointsList(list={self.list})"


class SurfacePointsFrame:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, surface_points, surface_labels=None, time: TimeFrame = None):


        if surface_labels is not None and len(surface_labels) != len(surface_points):
            raise ValueError("Number of labels must match the number of points.")

        len_labeled_points_list = len(surface_points)
        self.labeled_points_list = LabeledPointsList([])
        for i in range(len_labeled_points_list):
            if surface_labels is None:
                self.labeled_points_list.append(LabeledPoint(surface_points[i]))
            else:
                self.labeled_points_list.append(LabeledPoint(surface_points[i], surface_labels[i]))

        self.time = time

    def slice_arrays(self, id):
        if id >= len(self.labeled_points_list.list):
            raise ValueError("Index out of range.")

        self.labeled_points_list = LabeledPointsList(self.labeled_points_list.list[id:])
        return self

    # set time value
    def set_time_value(self, value):
        self.time.value = value
        return self

    def filter_by_label(self, label_index):
        """
        Filter the SurfaceData by the given label index, keeping only the corresponding points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        - SurfaceData instance containing only the points with the specified label index
        """
        filtered_data = self.labeled_points_list.filter_by_label(label_index)
        return SurfacePointsFrame(filtered_data.get_points(), filtered_data.get_labels(), self.time)

    @property
    def points_list(self):
        return self.labeled_points_list.get_points()

    #setter points_list
    @points_list.setter
    def points_list(self, points_list):
        len_labeled_points_list = len(points_list)
        surface_labels = self.labeled_points_list.get_labels()
        self.labeled_points_list = LabeledPointsList([])

        for i in range(len_labeled_points_list):
            if surface_labels is None or not surface_labels:
                self.labeled_points_list.append(LabeledPoint(points_list[i]))
            else:
                self.labeled_points_list.append(LabeledPoint(points_list[i], surface_labels[i]))
        return self.points_list


    @property
    def labels_list(self):
        return self.labeled_points_list.get_labels()

    # function which represents object in debug value view
    def __repr__(self):
        return f"SurfacePointsFrame(labeled_points_list={self.labeled_points_list}, time={self.time})"
