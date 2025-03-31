from dataclasses import dataclass
from typing import List, TypeAlias

import numpy as np
import torch
import trimesh
from numpy import ndarray
from scipy.spatial import KDTree
from trimesh import Trimesh

from utils.constants import CDataPreprocessing


# region CLASES - Centers info
class CenterPoint:
    """
    Data class for saving distance to specifed point
    """

    def __init__(self, center_point_index: int, point : ndarray, distance: float | None):
        # region SANITY CHECK
        if center_point_index is None:
            raise AssertionError("Point index is empty.")
        if center_point_index < 0:
            raise AssertionError("Point index must be greater than 0.")

        if point is None:
            raise AssertionError("Point is empty.")
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")

        if distance is not None:
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

    def get_centers_indices(self):
        return [element.point_index for element in self]


class LabeledPoint:
    """
    Class to represents point in object for one cluster and for a single time step.
    """

    def __init__(self, point: list, label: int = None, closest_centers: ClosestCentersList = None):
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")

        self.point = point
        self.label = label
        self.closest_centers : ClosestCentersList = closest_centers

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

    def filter_by_points_indices(self, points_indices):
        """
        Filter the LabeledPointsList by the given points indices, keeping only the corresponding points.

        Parameters:
        - points_indices: list, the points indices to filter by

        Returns:
        - LabeledPointsList instance containing only the LabeledPoint objects with the specified points indices
        """
        filtered_data = []

        for point_index in points_indices:
            filtered_data.append(self.list[point_index])

        return LabeledPointsList(filtered_data)

    def get_points(self):
        """
        Return the list of points.
        """
        return [labeled_point.point for labeled_point in self.list]

    def get_labels(self):
        """
        Return the list of labels.
        """
        return [labeled_point.label for labeled_point in self.list]

    def get_closest_centers(self) -> list[ClosestCentersList]:
        """
        Return the list of closest centers.
        """
        return [labeled_point.closest_centers for labeled_point in self.list]

    def __repr__(self):
        return f"LabeledPointsList(list={self.list})"
# endregion

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


class CentersInfo:

    def __init__(self, points: np.ndarray):
        self.points : ndarray = points
        self.kd_tree : KDTree = KDTree(points)

class SurfacePointsFrame:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, surface_points : np.ndarray, surface_labels : np.ndarray, time: TimeFrame | None, mesh: Trimesh,
                 centers_points: np.ndarray, center_info : CentersInfo = None):
        """
        :param surface_points: np.array of shape (num_points_in_file, 3) (x,y,z)
        :param surface_labels: np.array of shape (num_points_in_file, 1) (label)
        :param time:
        :param mesh:
        :param centers_points: np.array of shape (num_points_in_file, 3) (x,y,z)
        """

        # region SANITY CHECK
        if surface_labels is not None and len(surface_labels) != len(surface_points):
            raise ValueError("Number of labels must match the number of points.")

        # check surface_points
        if surface_points is None:
            raise AssertionError("Surface points are empty.")
        if surface_points.shape[1] != 3:
            raise AssertionError("Surface points must have 3 coordinates.")
        # check if it is float numbers inside
        if not np.issubdtype(surface_points.dtype, np.floating):
            raise AssertionError("Surface points must be float numbers.")

        # check surface_labels
        if surface_labels is not None:
            if not np.issubdtype(surface_labels.dtype, np.integer):
                raise AssertionError("Surface labels must be integers.")
            if len(surface_labels) != len(surface_points):
                raise AssertionError("Number of labels must match the number of points.")


        # endregion



        # region self._centers_info
        self._centers_info: CentersInfo | None = None
        if center_info is not None:
            self._centers_info = center_info
        elif centers_points is not None:
            self._centers_info = CentersInfo(points=centers_points)
        # endregion

        # region self_labeled_points_list

        # compute closest centers to points
        closest_centers_to_points = None
        if self._centers_info is not None:
            closest_centers_to_points = SurfacePointsFrame.compute_closest_centers(points=surface_points, centers_info=self._centers_info)

        labeled_points_list : LabeledPointsList = LabeledPointsList([])
        len_labeled_points_list = len(surface_points)
        for i in range(len_labeled_points_list):
            points = surface_points[i].tolist()
            label = None
            closest_centers = None
            if surface_labels is not None:
                label = surface_labels[i]
            if closest_centers_to_points is not None:
                closest_centers = closest_centers_to_points[i]

            labeled_point = LabeledPoint(points, label, closest_centers)
            labeled_points_list.append(labeled_point)

        self._labeled_points_list  : LabeledPointsList = labeled_points_list
        # endregion

        self.time : TimeFrame = time
        self._mesh: Trimesh = mesh
        self._original_mesh = None
        if mesh is not None:
            self.original_mesh = mesh

    @staticmethod
    def compute_closest_centers(points : np.ndarray, centers_info: CentersInfo) -> list[ClosestCentersList]:
        """
        Compute the closest centers to each point in the surface points list.
        """
        num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

        # Find num_closest_centers closest centers to each point
        distances, indices = centers_info.kd_tree.query(points, k=num_closest_centers)

        # create list of closest centers
        all_closest_centers_list : list[ClosestCentersList] = []
        for i in range(len(points)):
            points_closest_centers = ClosestCentersList([])
            for j in range(num_closest_centers):
                center_point_index = int(indices[i][j])
                distance = float(distances[i][j])
                center_point = CenterPoint(center_point_index=center_point_index, point=centers_info.points[center_point_index], distance=distance)
                points_closest_centers.append(center_point)
            all_closest_centers_list.append(points_closest_centers)
        return all_closest_centers_list

    @classmethod
    def create_instance(cls, surface_points : np.ndarray, surface_labels : np.ndarray, time: TimeFrame, mesh: Trimesh, centers_points: np.ndarray):
        return cls(surface_points, surface_labels, time, mesh, centers_points)

    @classmethod
    def duplicate_instances(cls, surface_points, surface_labels, time: TimeFrame, mesh: Trimesh, centers_info: CentersInfo):
        return cls(surface_points, surface_labels, time, mesh, None, centers_info)

    def slice_arrays(self, id):
        if id >= len(self._labeled_points_list.list):
            raise ValueError("Index out of range.")

        self._labeled_points_list = LabeledPointsList(self._labeled_points_list.list[id:])
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
        filtered_data = self._labeled_points_list.filter_by_label(label_index)
        points_np = np.array(filtered_data.get_points())
        labels_np = np.array(filtered_data.get_labels())

        points_frame = SurfacePointsFrame.duplicate_instances(surface_points=points_np,
                                                              surface_labels=labels_np, time=self.time,
                                                              mesh=self.mesh, centers_info=self.centers_info)
        return points_frame


    @property
    def points_list(self):
        return self._labeled_points_list.get_points()


    @property
    def labeled_points_list(self) -> LabeledPointsList:
        return self._labeled_points_list

    @property
    def closest_centers_list(self):
        closest_centers_list = []
        for labeled_point in self._labeled_points_list.list:
            closest_centers_list.append(labeled_point.closest_centers)
        return closest_centers_list

    # setter points_list
    @points_list.setter
    def points_list(self, points_list):
        len_labeled_points_list = len(points_list)
        surface_labels = self._labeled_points_list.get_labels()
        self._labeled_points_list = LabeledPointsList([])

        for i in range(len_labeled_points_list):
            if surface_labels is None or not surface_labels:
                self._labeled_points_list.append(LabeledPoint(points_list[i]))
            else:
                self._labeled_points_list.append(LabeledPoint(points_list[i], surface_labels[i]))
        # return self.points_list

    @property
    def labels_list(self):
        return self._labeled_points_list.get_labels()

    @property
    def mesh(self):
        if self._mesh is None:
            raise ValueError("Mesh is not loaded.")
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Trimesh):
        if not mesh:
            raise ValueError("Mesh is empty.")
        self._mesh = mesh

    @property
    def original_mesh(self):
        if self._original_mesh is None:
            raise ValueError("Mesh is not loaded.")
        return self._original_mesh

    @original_mesh.setter
    def original_mesh(self, mesh: Trimesh):
        if mesh is None:
            raise ValueError("Mesh is empty.")
        if self._original_mesh is not None:
            raise ValueError("Original mesh is already set.")
        self._original_mesh = mesh

    @property
    def centers_info(self) -> CentersInfo | None:
        if self._centers_info is None:
            return None
        return self._centers_info

    @centers_info.setter
    def centers_info(self, center_info: CentersInfo):
        if center_info is None:
            raise ValueError("Center info is empty.")

        self._centers_info = center_info

        self._compute_closest_centers_to_points()


    def _compute_closest_centers_to_points(self):
        surface_points = self.points_list
        surface_points = np.array(surface_points)

        closest_centers_to_points = None
        if self._centers_info is not None:
            closest_centers_to_points = SurfacePointsFrame.compute_closest_centers(points=surface_points, centers_info=self._centers_info)

        if self._labeled_points_list is None:
            raise ValueError("Labeled points list is empty.")

        if len(self._labeled_points_list.list) != len(closest_centers_to_points):
            raise ValueError("Number of points must match the number of closest centers.")

        # set closest centers to labeled points
        for i, labeled_point in enumerate(self._labeled_points_list.list):
            labeled_point.closest_centers = closest_centers_to_points[i]


    # function which represents object in debug value view
    def __repr__(self):
        return f"SurfacePointsFrame(labeled_points_list={self._labeled_points_list}, time={self.time})"

def find_closest_centers(points: torch.Tensor, centers_points: torch.Tensor, num_closest_centers : int, kdtree : KDTree ) -> torch.Tensor:
    """

    :param points:
    :param centers_points:
    :return:
     distance_tensor: tensor of distances in a row so the size is "len(points) * num_closest_points"
     centers_points_tensor: tensor of closest centers points to points in a row so the size is "len(points) * num_closest_points"
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

    if kdtree is None:
        raise AssertionError("KDTree is empty.")
    # endregion

    # region LOGIC


    # Find num_closest_centers closest centers to each point
    points_np = points.cpu().detach().numpy()
    _, indices = kdtree.query(points_np, k=num_closest_centers)

    # select centers points
    selected_centers_points = centers_points[indices]

    return selected_centers_points

def compute_distances_from_centers(points: torch.Tensor, closest_centers_points: torch.Tensor, num_closest_centers : int ) -> torch.Tensor:
    # region SANITY CHECKS
    if points is None:
        raise AssertionError("Points are empty.")
    # check if points elements shape is (x,y,z) and it is float numbers
    if points.shape[1] != 3:
        raise AssertionError("Points must have 3 coordinates.")

    if closest_centers_points is None:
        raise AssertionError("Centers points are empty.")
    # check if centers_points elements shape is (x,y,z) and it is float numbers
    if closest_centers_points.shape[1] != num_closest_centers:
        raise AssertionError("Centers points must specified closest centers.")
    if closest_centers_points.shape[2] != 3:
        raise AssertionError("Centers points must have 3 coordinates.")

    if closest_centers_points.shape[0] != points.shape[0]:
        raise AssertionError("Centers points must have the same number of points")

    # num_closest_centers
    if num_closest_centers is None:
        raise AssertionError("Number of closest centers is empty.")
    if num_closest_centers < 1:
        raise AssertionError("Number of closest centers must be greater than 0.")
    # endregion


    # compute distances
    points_in_row = points.unsqueeze(1).repeat(1, num_closest_centers, 1).view(-1, 3)
    closest_centers_points_in_row = closest_centers_points.view(-1, 3)
    distances_tensor = torch.norm(points_in_row - closest_centers_points_in_row, dim=1)

    return distances_tensor

@dataclass
class NormalizeValues:
    shift_vector: np.ndarray
    max_norm: float

class SurfacePointsFrameList:
    def __init__(self, surface_data_list: list):
        self._normalized_settings = NormalizedSetttings(False, None, None)
        if not isinstance(surface_data_list, list) or not all(
                isinstance(item, SurfacePointsFrame) for item in surface_data_list):
            raise TypeError("All items in surface_data_list must be instances of SurfaceData")
        self.list = surface_data_list

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

    @staticmethod
    def denormalize_points(normalize_values : NormalizeValues, points: ndarray) -> ndarray:
        denormalized_points = []
        for point in points:
            denormalized_point = (point * normalize_values.max_norm) + normalize_values.shift_vector
            denormalized_points.append(denormalized_point)
        #convert to numpy array
        return np.array(denormalized_points)

    def normalize_labeled_points_by_values(self, normalize_values : NormalizeValues):
        if self.is_normalized:
            raise ValueError("Data is already normalized.")
        if normalize_values is None:
            raise ValueError("Normalize values are empty.")

        # normalize Labeled points
        for surface_data in self.list:
            for labeled_point in surface_data.labeled_points_list.list:
                original_point = labeled_point.point
                normalized_point = (original_point - normalize_values.shift_vector) / normalize_values.max_norm
                labeled_point.point = normalized_point


    def normalize_all_elements(self):
        # todo test if it is working
        """
        Normalize the surface points for all objects in the list.
        Normalize the data to the range [0, 1] for each axis and shift it to the origin (0, 0, 0).
        :return: normalized_surface_points:
        """
        import numpy as np

        def __normalize_time(surface_data_list):
            total_length = len(surface_data_list.list)
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

                try:
                    mesh = surface_data.mesh
                except ValueError:
                    mesh = None
                if mesh is not None:
                    normalized_mesh = __create_normalized_mesh(shift_vector, max_norm, mesh)
                    surface_data.mesh = normalized_mesh

        def __shift_and_scale_centers(shift_vector, max_norm):
            for surface_data in self.list:
                surface_data : SurfacePointsFrame = surface_data
                # normalize centers
                points = surface_data.centers_info.points
                normalized_centers = (points - shift_vector) / max_norm
                surface_data.centers_info = CentersInfo(points=normalized_centers)

        # Normalize time for each object
        __normalize_time(self)

        # Compute the shift vector and max norm
        shift_vector, max_norm = __compute_shift_and_scale(self)

        self._normalized_settings = NormalizedSetttings(True, shift_vector, max_norm)

        # Shift points to origin and scale
        __shift_and_scale_points(shift_vector, max_norm)

        __shift_and_scale_centers(shift_vector, max_norm)

        return self

    @property
    def is_normalized(self):
        return self._normalized_settings.is_normalized

    @property
    def normalize_values(self) -> NormalizeValues:
        return NormalizeValues(self._normalized_settings.shift_vector, self._normalized_settings.max_norm)


    def get_unique_times(self):
        """
        Return the set of unique times.
        """

        return {surface_data.time.value for surface_data in self.list}

    def get_element_by_time_index(self, time_index : int) -> SurfacePointsFrame | None:
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
            points_frame = SurfacePointsFrame.duplicate_instances(surface_points=filtered_points, surface_labels=filtered_labels,
                                              time=surface_data.time, mesh=surface_data.mesh,
                                              centers_info=surface_data.centers_info)
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

    def get_original_meshes_list(self) -> MeshList:
        """
        Return the list of original meshes.
        if some none mesh is found, raise ValueError
        and check if index of output list of mesh is same as time index
        """
        meshes_list = MeshList()
        for i, surface_data in enumerate(self.list):
            if surface_data.original_mesh is None:
                raise ValueError("Original mesh is not loaded.")
            meshes_list.append((surface_data.time.index, surface_data.original_mesh))

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

