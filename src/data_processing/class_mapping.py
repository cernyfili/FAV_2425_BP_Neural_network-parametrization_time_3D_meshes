from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import trimesh
from numpy import ndarray
from scipy.spatial import KDTree
from trimesh import Trimesh

from utils.constants import CDataPreprocessing


@dataclass
class NormalizedSetttings:
    is_normalized: bool
    shift_vector: np.ndarray | None
    max_norm: float | None


class MeshList(list):

    # override append
    def append(self, __object):
        if not isinstance(__object, tuple):
            raise ValueError("Object must be a tuple.")
        if len(__object) != 2:
            raise ValueError("Tuple must have 2 elements.")
        if not isinstance(__object[0], int):
            raise ValueError("First element of tuple must be an integer.")
        if not isinstance(__object[1], Trimesh):
            raise ValueError("Second element of tuple must be a Trimesh object.")
        super().append(__object)

    # get mesh by time index
    def get_mesh_by_time_index(self, time_index: int) -> Trimesh:
        element = self[time_index]

        if element[0] == time_index:
            return element[1]

        for i, element in enumerate(self):
            if element[0] == time_index:
                return element[1]

        raise ValueError("Time index not found.")


class TimeFrame:
    """
    Class represents time for one frame from sequence
    """

    def __init__(self, index: int, value: float = None):
        self.index = index
        self.value = value

    def __repr__(self):
        return f"TimeFrame(index={self.index}, value={self.value})"


class SurfacePointsFrame:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, surface_points, surface_labels, time: TimeFrame, mesh: Trimesh,
                 centers_points: np.array):
        """

        :param surface_points:
        :param surface_labels:
        :param time:
        :param mesh:
        :param centers_points: np.array of shape (num_points_in_file, 3) (x,y,z)
        """

        if surface_labels is not None and len(surface_labels) != len(surface_points):
            raise ValueError("Number of labels must match the number of points.")

        closest_centers_to_points = None
        # if centers_points is not None:
        #     closest_centers_to_points = compute_closest_centers(surface_points, centers_points)

        # region Compute closest centers

        # endregion

        self.labeled_points_list = LabeledPointsList([])
        len_labeled_points_list = len(surface_points)
        for i in range(len_labeled_points_list):
            labeled_point = LabeledPoint(surface_points[i])
            if surface_labels is not None:
                labeled_point.label = surface_labels[i]
            if closest_centers_to_points is not None:
                labeled_point.closest_centers = closest_centers_to_points[i]
            self.labeled_points_list.append(labeled_point)

        self.time = time
        self._mesh: Trimesh = mesh
        self._centers_points: np.array = centers_points

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
        # return SurfacePointsFrame(filtered_data.get_points(), filtered_data.get_labels(), self.time)
        points_frame = SurfacePointsFrame(surface_points=filtered_data.get_points(),
                                          surface_labels=filtered_data.get_labels(), time=self.time,
                                          mesh=self.mesh, centers_points=self.centers_points)
        return points_frame

    @property
    def points_list(self):
        return self.labeled_points_list.get_points()

    @property
    def closest_centers_list(self):
        closest_centers_list = []
        for labeled_point in self.labeled_points_list.list:
            closest_centers_list.append(labeled_point.closest_centers)
        return closest_centers_list

    # setter points_list
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
        # return self.points_list

    @property
    def labels_list(self):
        return self.labeled_points_list.get_labels()

    @property
    def mesh(self):
        if self._mesh is None:
            raise ValueError("Mesh is not loaded.")
        return self._mesh

    @property
    def centers_points(self) -> np.array:
        if self._centers_points is None:
            raise ValueError("Centers points are not loaded.")
        return self._centers_points

    @mesh.setter
    def mesh(self, mesh: Trimesh):
        if not mesh:
            raise ValueError("Mesh is empty.")
        self._mesh = mesh

    # function which represents object in debug value view
    def __repr__(self):
        return f"SurfacePointsFrame(labeled_points_list={self.labeled_points_list}, time={self.time})"

def compute_closest_centers_tensor(points: torch.Tensor, centers_points: torch.Tensor) -> torch.Tensor:
    """

    :param points:
    :param centers_points:
    :return: list of ClosestCentersList with index coreesponding to index of points
    """
    # region SANITY CHECKS
    if points is None:
        raise AssertionError("Points are empty.")
    # check if points elements shape is (x,y,z) and it is float numbers
    if points.shape[1] != 3:
        raise AssertionError("Points must have 3 coordinates.")

    if centers_points is None:
        raise AssertionError("Centers points are empty.")
    # check if centers_points elements shape is (x,y,z) and it is float numbers
    if centers_points.shape[1] != 3:
        raise AssertionError("Centers points must have 3 coordinates.")

    # endregion

    # start points tensor so i can backpropagate
    points.requires_grad_(True)

    # region LOGIC
    num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

    # Build a KDTree for the clustered points
    centers_points_np = centers_points.detach().numpy()
    kdtree = KDTree(centers_points_np)
    # Find num_closest_centers closest centers to each point
    points_np = points.detach().numpy()
    distances, indices = kdtree.query(points_np, k=num_closest_centers)

    # Convert distances to a tensor
    distances_tensor = torch.tensor(distances, dtype=torch.float32, device=points.device)
    distances_tensor.requires_grad_(True)
    # distances_tensor from shape (x,3) to (x)
    distances_tensor = distances_tensor.view(-1)

    return distances_tensor

def compute_closest_centers(points: torch.Tensor, centers_points: np.array) -> list:
    """

    :param points:
    :param centers_points:
    :return: list of ClosestCentersList with index coreesponding to index of points
    """
    # region SANITY CHECKS
    if points is None:
        raise AssertionError("Points are empty.")
    # check if points elements shape is (x,y,z) and it is float numbers
    if points.shape[1] != 3:
        raise AssertionError("Points must have 3 coordinates.")

    if centers_points is None:
        raise AssertionError("Centers points are empty.")
    # check if centers_points elements shape is (x,y,z) and it is float numbers
    if centers_points.shape[1] != 3:
        raise AssertionError("Centers points must have 3 coordinates.")
    # endregion

    # region LOGIC
    num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

    # Build a KDTree for the clustered points
    kdtree = KDTree(centers_points)
    # Find num_closest_centers closest centers to each point

    points_np = points.detach().numpy()
    distances, indices = kdtree.query(points_np, k=num_closest_centers)

    points_closest_centers = []
    for i, point in enumerate(points):
        closest_centers = ClosestCentersList([])
        closest_centers_indices = indices[i]
        closest_centers_distances = distances[i]
        for j in range(num_closest_centers):
            index = closest_centers_indices[j]
            # index to int
            index = int(index)

            distance = closest_centers_distances[j]
            # distance to float
            distance = float(distance)
            center_point = centers_points[index]
            closest_centers.append(CenterPoint(index,center_point, distance))
        points_closest_centers.append(closest_centers)
    # endregion

    return points_closest_centers

def compute_distances_to_original_points_centers(points: torch.Tensor, original_points_centers: list) -> np.array:
    """

    :param points:
    :return: list of ClosestCentersList with index coreesponding to index of points
    """
    # region SANITY CHECKS
    if points is None:
        raise AssertionError("Points are empty.")
    # check if points elements shape is (x,y,z) and it is float numbers
    if points.shape[1] != 3:
        raise AssertionError("Points must have 3 coordinates.")
    # endregion

    # region LOGIC
    num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

    # Assuming original_points_centers is a list of ClosestCentersList
    original_points_centers_points = []
    for closest_centers_list in original_points_centers:
        original_points_centers_points.extend([center_point.point for center_point in closest_centers_list])
    original_points_centers_points = torch.tensor(original_points_centers_points)
    original_points_centers_points.requires_grad_(True)
    """ list of center cordinates in a row - one point has num_closest_centers rows """

    points_duplicated = torch.repeat_interleave(points, num_closest_centers, dim=0)
    points_duplicated.requires_grad_(True)
    """ duplicate points to match the shape of original_points_centers_points_np """

    if points_duplicated.shape != original_points_centers_points.shape:
        raise ValueError("Shapes of points and original_points_centers_points_np do not match.")

    distances = torch.norm(points_duplicated - original_points_centers_points, dim=1)
    distances.requires_grad_(True)

    return distances






class SurfacePointsFrameList:
    def __init__(self, surface_data_list: list):
        self._normalized_settings = NormalizedSetttings(False, None, None)
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

        def __normalize_time(surface_data_list):
            total_length = len(surface_data_list.list) - 1
            for surface_data in surface_data_list.list:
                surface_data.time.value /= total_length

        def __compute_shift_and_scale(surface_data_list: SurfacePointsFrameList):
            # Combine all points for faster computation
            all_points_list = surface_data_list.get_all_points()
            all_points = np.vstack([all_points_list])
            min_corner = np.min(all_points, axis=0)
            max_corner = np.max(all_points, axis=0)
            shift_vector = (min_corner + max_corner) / 2
            max_norm = np.linalg.norm(all_points - shift_vector, axis=1).max()
            return shift_vector, max_norm

        def __create_normalized_mesh(shift_vector, max_norm, mesh: Trimesh) -> Trimesh:
            # normalize mesh
            normalized_vertices = (mesh.vertices - shift_vector) / max_norm
            normalized_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=mesh.faces)
            return normalized_mesh

        def __shift_and_scale_points(shift_vector, max_norm):
            for surface_data in self.list:
                # normalize surface points
                normalized_points = (surface_data.points_list - shift_vector) / max_norm
                surface_data.points_list = normalized_points

                # normalize mesh
                mesh = surface_data.mesh
                normalized_mesh = __create_normalized_mesh(shift_vector, max_norm, mesh)
                surface_data.mesh = normalized_mesh

        # Normalize time for each object
        __normalize_time(self)

        # Compute the shift vector and max norm
        shift_vector, max_norm = __compute_shift_and_scale(self)

        # Shift points to origin and scale
        __shift_and_scale_points(shift_vector, max_norm)

        self._normalized_settings = NormalizedSetttings(True, shift_vector, max_norm)

        return self

    @property
    def is_normalized(self):
        return self._normalized_settings.is_normalized

    def get_unique_times(self):
        """
        Return the set of unique times.
        """

        return {surface_data.time.value for surface_data in self.list}

    def get_element_by_time_index(self, time_index) -> SurfacePointsFrame:
        """
        Find the element in the list with the specified time index.
        """

        def filter_by_time_index(self, time_index: int):
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

    def get_element_by_time_value(self, time_index) -> SurfacePointsFrame:
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

        def compute_unique_clusters(self: SurfacePointsFrameList):
            """
            Private method to compute unique clusters from the surface data list.
            """
            unique_clusters = set()
            for surface_data in self.list:
                if surface_data.labels_list is None or not surface_data.labels_list or surface_data.labels_list[
                    0] == None:
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
            filtered_points = points_array[matching_indices]
            filtered_labels = labels_array[matching_indices]

            # Create a new SurfaceData instance with the filtered points and labels
            points_frame = SurfacePointsFrame(surface_points=filtered_points, surface_labels=filtered_labels,
                                              time=surface_data.time, mesh=surface_data.mesh,
                                              centers_points=surface_data.centers_points)
            filtered_data.append(points_frame)


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
        return [surface_data_element.time for surface_data_element in self.list]

    def get_meshes_list(self) -> MeshList:
        """
        Return the list of meshes.
        if some none mesh is found, raise ValueError
        and check if index of output list of mesh is same as time index
        """
        meshes_list = MeshList()
        for i, surface_data in enumerate(self.list):
            if surface_data.mesh is None:
                raise ValueError("Mesh is not loaded.")
            meshes_list.append((surface_data.time.index, surface_data.mesh))

        return meshes_list


def time_frame_list_find_closest_element_index(time_frame_list: List[TimeFrame], time_value: float) -> int:
    if time_value == 0.0:
        return 0

    epsilon = (1 / len(time_frame_list)) / 4
    closest_list = []
    for time_frame in time_frame_list:
        if abs(time_frame.value - time_value) < epsilon:
            return time_frame.index

    raise ValueError("Index not found")


class CenterPoint:
    """
    Data class for saving distance to specifed point
    """

    def __init__(self, center_point_index: int, point : ndarray, distance: float):
        # region SANITY CHECK
        if center_point_index is None:
            raise AssertionError("Point index is empty.")
        if center_point_index < 0:
            raise AssertionError("Point index must be greater than 0.")

        if point is None:
            raise AssertionError("Point is empty.")
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")

        if distance is None:
            raise AssertionError("Distance is empty.")
        if distance < 0:
            raise AssertionError("Distance must be greater than 0.")
        # endregion

        self.point_index: int = center_point_index
        self.distance: float = distance
        self.point: ndarray = point


class ClosestCentersList(list):
    """
    Data class for saving distance to specifed point
    """

    _ELEMENT_DATA_TYPE = CenterPoint

    def __init__(self, elements):
        if not all(isinstance(element, self._ELEMENT_DATA_TYPE) for element in elements):
            raise ValueError("All elements must be integers")
        super().__init__(elements)

    def append(self, __object):
        if not isinstance(__object, self._ELEMENT_DATA_TYPE):
            raise ValueError("Object must be an instance of CenterPoint.")
        super().append(__object)


class LabeledPoint:
    """
    Class to represents point in object for one cluster and for a single time step.
    """

    def __init__(self, point: list, label: int = None, closest_centers: ClosestCentersList = None):
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")

        self.point = point
        self.label = label
        self.closest_centers = closest_centers

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
