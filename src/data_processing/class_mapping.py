from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from numpy import ndarray
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.linear_model.tests.test_perceptron import indices
from sklearn.preprocessing import StandardScaler
from trimesh import Trimesh

from utils.constants import CDataPreprocessing

@dataclass
class NormalizedSetttings:
    is_normalized: bool
    shift_vector: np.ndarray | None
    max_norm: float | None


# region CLASES - Centers info
class CenterPoint:
    """
    Data class for saving distance to specifed point
    """

    def __init__(self, center_point_index: int):
        # region SANITY CHECK
        if center_point_index is None:
            raise AssertionError("Point index is empty.")
        if center_point_index < 0:
            raise AssertionError("Point index must be greater than 0.")

        # endregion

        self.point_index: int = center_point_index


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

    def __init__(self, point: list, index : int, label: int = None, closest_centers: ClosestCentersList = None):
        if len(point) != 3:
            raise ValueError("Point must have 3 coordinates.")
        if index < 0:
            raise ValueError("Index must be greater than 0.")

        self.point : list = point
        self.index : int = index
        self.label : int = label
        self.closest_centers : ClosestCentersList = closest_centers

    def normalize(self, normalized_settings : NormalizedSetttings):
        """
        Normalize the point using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        # normalize Labeled points
        original_point = self.point
        normalized_point = (original_point - normalized_settings.shift_vector) / normalized_settings.max_norm
        self.point = normalized_point

    def __repr__(self):
        return f"LabeledPoint(point={self.point}, label={self.label})"


class LabeledPointsList:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, labeled_points_list: list[LabeledPoint]):
        self.list = labeled_points_list

    def append(self, labeled_point):
        """
        Append a LabeledPoint object to the list.
        """
        if not isinstance(labeled_point, LabeledPoint):
            raise TypeError("labeled_point must be an instance of LabeledPoint")
        self.list.append(labeled_point)

    def normalize(self, normalized_settings: NormalizedSetttings):
        """
        Normalize the labeled points using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        # normalize Labeled points
        for labeled_point in self.list:
            labeled_point.normalize(normalized_settings)

    def filter_by_label(self, label_index):
        """
        Filter the LabeledPointsList by the given label index, keeping only the corresponding points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        - LabeledPointsList instance containing only the LabeledPoint objects with the specified label index
        """
        new_labeled_points_list = deepcopy(self)

        filtered_list_data = []
        for labeled_point in new_labeled_points_list.list:
            if labeled_point.label == label_index:
                filtered_list_data.append(labeled_point)

        new_labeled_points_list.list = filtered_list_data

        return new_labeled_points_list

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

    def get_points_indices(self):
        """
        Return the list of points indices.
        """
        return [labeled_point.index for labeled_point in self.list]

    def get_labels(self):
        """
        Return the list of labels.
        """
        return [labeled_point.label for labeled_point in self.list]

    def create_closest_centers_indicies_list(self) -> list[list[int]]:
        """
        Creates a list of lists of closest centers indices for each labeled point.

        """

        all_points_closest_centers = []
        for labeled_point in self.list:
            if labeled_point.closest_centers is None:
                raise ValueError("Closest centers are not set for all points.")

            point_closest_centers_indices = labeled_point.closest_centers.get_centers_indices()
            all_points_closest_centers.append(point_closest_centers_indices)


        return all_points_closest_centers



    def get_closest_centers(self) -> list[ClosestCentersList]:
        """
        Return the list of closest centers.
        """
        return [labeled_point.closest_centers for labeled_point in self.list]

    def __repr__(self):
        return f"LabeledPointsList(list={self.list})"
# endregion

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
        return f"{self.index} : {self.value}"


class CentersInfo:

    def __init__(self, points: np.ndarray):
        self._points : ndarray = points
        self._kd_tree : KDTree = KDTree(points)

    @property
    def points(self):
        if self._points is None:
            raise ValueError("Points are not loaded.")
        return self._points

    @points.setter
    def points(self, points: np.ndarray):
        if points is None:
            raise ValueError("Points are empty.")
        if len(points) == 0:
            raise ValueError("Points are empty.")
        if points.shape[1] != 3:
            raise ValueError("Points must have 3 coordinates.")
        if not np.issubdtype(points.dtype, np.floating):
            raise ValueError("Points must be float numbers.")
        self._points = points
        self._kd_tree : KDTree = KDTree(points)

    @property
    def kd_tree(self):
        if self._kd_tree is None:
            raise ValueError("KDTree is not loaded.")
        return self._kd_tree

class SurfacePointsFrame:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, surface_points: np.ndarray, surface_labels: np.ndarray, mesh: Trimesh,
                 centers_points: np.ndarray):
        """
        :param surface_points: np.array of shape (num_points_in_file, 3) (x,y,z)
        :param surface_labels: np.array of shape (num_points_in_file, 1) (label)
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
        if surface_labels is None:
            raise AssertionError("Surface labels are empty.")
        if not np.issubdtype(surface_labels.dtype, np.integer):
            raise AssertionError("Surface labels must be integers.")
        if len(surface_labels) != len(surface_points):
            raise AssertionError("Number of labels must match the number of points.")

        # check mesh
        if mesh is None:
            raise AssertionError("Mesh is empty.")
        if not isinstance(mesh, trimesh.Trimesh):
            raise AssertionError("Mesh must be trimesh object.")

        # check centers_points
        if centers_points is None:
            raise AssertionError("Centers points are empty.")
        if centers_points.shape[1] != 3:
            raise AssertionError("Centers points must have 3 coordinates.")
        # check if it is float numbers inside
        if not np.issubdtype(centers_points.dtype, np.floating):
            raise AssertionError("Centers points must be float numbers.")

        # endregion

        # region create values which are set afterwards
        self._time : TimeFrame | None = None # created in set time
        # endregion

        self._normalized_centers_info : CentersInfo | None = None
        self._original_centers_info : CentersInfo = CentersInfo(points=centers_points)

        self._normalized_mesh: Trimesh | None = None
        self._original_mesh : Trimesh = mesh

        # region self._labeled_points_list
        closest_centers_to_points = self.compute_closest_centers(points=surface_points, centers_info=self._original_centers_info)

        labeled_points_list : LabeledPointsList = LabeledPointsList([])

        if len(surface_points) != len(surface_labels) and len(surface_points) != len(closest_centers_to_points):
            raise ValueError("Number of points must match the number of labels and closest centers.")

        len_labeled_points_list = len(surface_points)


        for i in range(len_labeled_points_list):
            points = surface_points[i].tolist()
            label = int(surface_labels[i])
            closest_centers = closest_centers_to_points[i]

            labeled_point = LabeledPoint(points, i, label, closest_centers)

            labeled_points_list.append(labeled_point)

        # endregion
        self._normalized_labeled_points_list  : LabeledPointsList | None = None

        self._original_labeled_points_list : LabeledPointsList = labeled_points_list

    def normalize_time(self, length):
        """
        Normalize the time value to the range [0, 1].
        """
        new_time_value = deepcopy(self.time)

        if length <= 0:
            raise ValueError("Length must be greater than 0.")

        new_time_value.value = self.time.index / length

        self._time = new_time_value

    def normalize_labeled_points(self, normalized_settings : NormalizedSetttings):
        """
        Normalize the labeled points using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        self._normalized_labeled_points_list = deepcopy(self._original_labeled_points_list)
        # normalize Labeled points
        self._normalized_labeled_points_list.normalize(normalized_settings)

    def _normalize_mesh(self, normalized_settings : NormalizedSetttings):
        """
        Normalize the mesh using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        # normalize mesh
        original_mesh = self._original_mesh

        normalized_vertices = (original_mesh.vertices - normalized_settings.shift_vector) / normalized_settings.max_norm

        normalized_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=original_mesh.faces)
        self._normalized_mesh = normalized_mesh

    def _normalize_centers_info(self, normalized_settings : NormalizedSetttings):
        """
        Normalize the centers info using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        # normalize centers
        original_centers_info = self._original_centers_info

        normalized_centers_points = (original_centers_info.points - normalized_settings.shift_vector) / normalized_settings.max_norm

        self._normalized_centers_info = CentersInfo(points=normalized_centers_points)

    def normalize(self, normalized_settings : NormalizedSetttings):
        """
        Normalize the surface points using the provided normalization settings.
        """
        if normalized_settings is None:
            raise ValueError("Normalized settings are empty.")

        self.normalize_labeled_points(normalized_settings)
        self._normalize_mesh(normalized_settings)
        self._normalize_centers_info(normalized_settings)


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
                center_point = CenterPoint(center_point_index=center_point_index)
                points_closest_centers.append(center_point)
            all_closest_centers_list.append(points_closest_centers)
        return all_closest_centers_list

    @classmethod
    def create_instance(cls, surface_points: np.ndarray, surface_labels: np.ndarray, mesh: Trimesh,
                        centers_points: np.ndarray):
        instance = cls(surface_points, surface_labels, mesh, centers_points)
        return instance

    # def _compute_pca_centers_aproximation(self, centers_info: CentersInfo):
    #     # region SANITY CHECK
    #     if centers_info is None:
    #         raise ValueError("Centers info is empty.")
    #     # endregion

    # def slice_arrays(self, id):
    #     if id >= len(self._labeled_points_list.list):
    #         raise ValueError("Index out of range.")
    #
    #     self._labeled_points_list = LabeledPointsList(self._labeled_points_list.list[id:])
    #     return self

    # set time value

    # def set_time_value(self, value):
    #     self.time.value = value
    #     return self

    # region SETTERS, GETTERS

    # region BOTH
    @property
    def original_points_list(self):
        # is only normalized used
        points = self._original_labeled_points_list.get_points()
        if points is None:
            raise ValueError("Points are not")
        return points

    @property
    def normalized_points_list(self):
        #is only normalized used
        points = self.normalized_labeled_points_list.get_points()
        if points is None:
            raise ValueError("Points are not normalized.")
        return points

    # @normalized_points_list.setter
    # def normalized_points_list(self, points_list):
    #     # is normalized
    #     len_labeled_points_list = len(points_list)
    #     surface_labels = self.normalized_labeled_points_list.get_labels()
    #     new_labeled_points_list = LabeledPointsList([])
    # 
    #     for i in range(len_labeled_points_list):
    #         if surface_labels is None or not surface_labels:
    #             new_labeled_points_list.append(LabeledPoint(points_list[i]))
    #         else:
    #             new_labeled_points_list.append(LabeledPoint(points_list[i], surface_labels[i]))
    # 
    #     self.normalized_labeled_points_list = new_labeled_points_list
    #     # return self.points_list


    @property
    def normalized_mesh(self):
        # normalized maybe used wrong in chamfer loss function
        if self._normalized_mesh is None:
            raise ValueError("Mesh is not loaded.")
        return self._normalized_mesh

    # @normalized_mesh.setter
    # def normalized_mesh(self, mesh: Trimesh):
    #     # is normalized
    #     if not mesh:
    #         raise ValueError("Mesh is empty.")
    #     self._normalized_mesh = mesh


    @property
    def normalized_centers_info(self) -> CentersInfo | None:
        # normalized - might be used wrong for compute closest centers
        if self._normalized_centers_info is None:
            return None
        return self._normalized_centers_info

    # @normalized_centers_info.setter
    # def normalized_centers_info(self, center_info: CentersInfo):
    #     if center_info is None:
    #         raise ValueError("Center info is empty.")
    #
    #     self._normalized_centers_info = center_info
    #
    #     self._compute_closest_centers_to_points()


    @property
    def time(self) -> TimeFrame:
        if self._time is None:
            raise ValueError("Time is not set.")
        return self._time

    @time.setter
    def time(self, time: TimeFrame):
        if time is None:
            raise ValueError("Time is empty.")
        if not isinstance(time, TimeFrame):
            raise ValueError("Time must be an instance of TimeFrame.")
        self._time = time
    # endregion

    # region GETTERS

    @property
    def original_labeled_points_list(self) -> LabeledPointsList:
       raise NotImplementedError("Cant be used its not filtered in filter by label")

    @property
    def normalized_labeled_points_list(self) -> LabeledPointsList:
        return self._normalized_labeled_points_list

    @normalized_labeled_points_list.setter
    def normalized_labeled_points_list(self, labeled_points_list):
        if not isinstance(labeled_points_list, LabeledPointsList):
            raise ValueError("labeled_points_list must be an instance of LabeledPointsList")
        self._normalized_labeled_points_list = labeled_points_list

    # @property
    # def closest_centers_list(self):
    #     closest_centers_list = []
    #     for labeled_point in self.labeled_points_list.list:
    #         closest_centers_list.append(labeled_point.closest_centers)
    #     return closest_centers_list

    # setter points_list
    @property
    def labels_list(self):
        return self.normalized_labeled_points_list.get_labels()


    @property
    def original_mesh(self):
        if self._original_mesh is None:
            raise ValueError("Mesh is not loaded.")
        return self._original_mesh


    # endregion

    # endregion


    def filter_by_cluster_label(self, label_index):
        """
        Filter the SurfaceData by the given label index, keeping only the corresponding points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        - SurfaceData instance containing only the points with the specified label index
        """

        # deep copy
        new_surface_points_frame = deepcopy(self)

        new_normalized_labeled_points_list = new_surface_points_frame.normalized_labeled_points_list.filter_by_label(label_index)
        new_surface_points_frame.normalized_labeled_points_list = new_normalized_labeled_points_list


        return new_surface_points_frame

    # def _compute_closest_centers_to_points(self):
    #     surface_points = self.normalized_points_list
    #     surface_points = np.array(surface_points)
    #
    #     closest_centers_to_points = None
    #     if self.normalized_centers_info is not None:
    #         closest_centers_to_points = SurfacePointsFrame.compute_closest_centers(points=surface_points, centers_info=self.normalized_centers_info)
    #
    #     if self.normalized_labeled_points_list is None:
    #         raise ValueError("Labeled points list is empty.")
    #
    #     if len(self.normalized_labeled_points_list.list) != len(closest_centers_to_points):
    #         raise ValueError("Number of points must match the number of closest centers.")
    #
    #     # set closest centers to labeled points
    #     for i, labeled_point in enumerate(self.normalized_labeled_points_list.list):
    #         labeled_point.closest_centers = closest_centers_to_points[i]


    # function which represents object in debug value view
    def __repr__(self):
        return f"SurfacePointsFrame(labeled_points_list={self._normalized_labeled_points_list}, time={self.time})"

# def find_closest_centers(points: torch.Tensor, centers_points: torch.Tensor, num_closest_centers : int, kdtree : KDTree ) -> torch.Tensor:
#     """
#
#     :param points:
#     :param centers_points:
#     :return:
#      distance_tensor: tensor of distances in a row so the size is "len(points) * num_closest_points"
#      centers_points_tensor: tensor of closest centers points to points in a row so the size is "len(points) * num_closest_points"
#     """
#     # region SANITY CHECKS
#     if points is None:
#         raise AssertionError("Points are empty.")
#     # check if points elements shape is (x,y,z) and it is float numbers
#     if points.shape[1] != 3:
#         raise AssertionError("Points must have 3 coordinates.")
#
#     if centers_points is None:
#         raise AssertionError("Centers points are empty.")
#     # check if centers_points elements shape is (x,y,z) and it is float numbers
#     if centers_points.shape[1] != 3:
#         raise AssertionError("Centers points must have 3 coordinates.")
#
#     if kdtree is None:
#         raise AssertionError("KDTree is empty.")
#     # endregion
#
#     # region LOGIC
#
#
#     # Find num_closest_centers closest centers to each point
#     points_np = points.cpu().detach().numpy()
#     _, indices = kdtree.query(points_np, k=num_closest_centers)
#
#     # select centers points
#     selected_centers_points = centers_points[indices]
#
#     return selected_centers_points

@dataclass
class NormalizeValues:
    shift_vector: np.ndarray
    max_norm: float

class PointsList(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape[1] != 3:
            raise ValueError("Each point must have 3 coordinates.")
        if not np.issubdtype(obj.dtype, np.floating):
            raise TypeError("Points must be of float type.")
        return obj

    def append(self, point):
        if not isinstance(point, np.ndarray):
            raise TypeError("Point must be a numpy ndarray.")
        if point.shape != (3,):
            raise ValueError("Point must have 3 coordinates.")
        if not np.issubdtype(point.dtype, np.floating):
            raise TypeError("Point must be of float type.")
        self = np.vstack([self, point])
        return self

    def __array_finalize__(self, obj):
        if obj is None: return
        if obj.shape[1] != 3:
            raise ValueError("Each point must have 3 coordinates.")
        if not np.issubdtype(obj.dtype, np.floating):
            raise TypeError("Points must be of float type.")

@dataclass
class CentersPointsWithTime:
    centers_points: PointsList
    time: TimeFrame

# class which represents the list of CentersPointsElements
class CentersPointsList(list):
    _ELEMENT_DATA_TYPE = CentersPointsWithTime

    def __init__(self, elements):
        if not all(isinstance(element, self._ELEMENT_DATA_TYPE) for element in elements):
            raise ValueError("All elements must be integers")
        super().__init__(elements)

    def append(self, __object):
        if not isinstance(__object, self._ELEMENT_DATA_TYPE):
            raise ValueError("Object must be an instance of CenterPoint.")
        super().append(__object)

    def convert_to_nparray(self) -> np.ndarray:
        """
        Converts CentersPointsList into a tensor matrix where:
        - Columns represent center points.
        - Rows represent the same centers at different time frames.

        :param centers_points_list: List of CentersPointsWithTime objects
        :return: torch.Tensor of shape (num_centers, num_time_frames, 3)
        """

        centers_points_list = self

        if not centers_points_list:
            raise ValueError("centers_points_list cannot be empty.")

        # Extract the number of center points (assume the first entry defines it)
        num_centers = len(centers_points_list[0].centers_points)

        # Ensure all time frames have the same number of centers
        for element in centers_points_list:
            if len(element.centers_points) != num_centers:
                raise ValueError("Mismatch in number of center points across time frames.")

        # Stack along time first (shape: num_time_frames, num_centers, 3)
        stacked_points = np.stack([element.centers_points for element in centers_points_list])

        # Transpose to match the desired format: (num_centers, num_time_frames, 3)
        transposed_points = np.transpose(stacked_points, (1, 0, 2))

        # Convert to PyTorch tensor
        return transposed_points




class SurfacePointsFrameList:
    def __init__(self, surface_data_list: list[SurfacePointsFrame]):
        self._normalized_settings = NormalizedSetttings(False, None, None)
        if not isinstance(surface_data_list, list) or not all(
                isinstance(item, SurfacePointsFrame) for item in surface_data_list):
            raise TypeError("All items in surface_data_list must be instances of SurfaceData")
        self._list : list[SurfacePointsFrame] = surface_data_list

    def assign_time_to_all_elements(self):
        """Assign a time index to each surface data item if not already assigned."""
        for i, surface_data in enumerate(self._list):
            surface_data : SurfacePointsFrame = surface_data
            surface_data.time = TimeFrame(i, i)

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
        for surface_data in self._list:
            normalize_settings = NormalizedSetttings(True, normalize_values.shift_vector, normalize_values.max_norm)
            surface_data.normalize_labeled_points(normalize_settings)

    def normalize_all_elements(self):
        """
        Normalize the surface points for all objects in the list.
        Normalize the data to the range [0, 1] for each axis and shift it to the origin (0, 0, 0).
        :return: normalized_surface_points:
        """
        import numpy as np

        def __compute_shift_and_scale(surface_data_list: SurfacePointsFrameList):
            # Combine all points for faster computation
            all_points_list = surface_data_list._get_all_points()

            all_points = np.vstack([all_points_list])
            min_corner = np.min(all_points, axis=0)
            max_corner = np.max(all_points, axis=0)
            shift_vector = (min_corner + max_corner) / 2
            max_norm = np.linalg.norm(all_points - shift_vector, axis=1).max()
            return shift_vector, max_norm

        # Compute the shift vector and max norm
        shift_vector, max_norm = __compute_shift_and_scale(self)

        self._normalized_settings = NormalizedSetttings(True, shift_vector, max_norm)
        total_length = len(self._list)

        for surface_data in self._list:
            surface_data : SurfacePointsFrame = surface_data
            # normalize labeled points
            surface_data.normalize(self._normalized_settings)
            surface_data.normalize_time(total_length)

        return self

    # region GETTERS
    @property
    def is_normalized(self):
        return self._normalized_settings.is_normalized

    @property
    def normalize_values(self) -> NormalizeValues:
        return NormalizeValues(self._normalized_settings.shift_vector, self._normalized_settings.max_norm)

    @property
    def _centers_points_list(self) -> CentersPointsList:
        centers_points_list = CentersPointsList([])
        centers_length = None
        for surface_data in self._list:
            surface_data : SurfacePointsFrame = surface_data
            if surface_data.normalized_centers_info is None:
                raise ValueError("Centers info is not loaded.")
            centers_points = PointsList(surface_data.normalized_centers_info.points)
            if centers_length is None:
                centers_length = len(centers_points)

            if len(centers_points) != centers_length:
                raise ValueError("Centers points length mismatch.")
            time = surface_data.time
            centers_points_list.append(CentersPointsWithTime(centers_points, time))

        return centers_points_list

    @property
    def public_list(self) -> list[SurfacePointsFrame]:
        return self._list
    # endregion

    def append(self, surface_data):
        """
        Append a SurfaceData object to the list and update unique clusters.
        """
        if not isinstance(surface_data, SurfacePointsFrame):
            raise TypeError("surface_data must be an instance of SurfaceData")
        self._list.append(surface_data)

    def filter_by_label(self, label_index):
        """
        Filter the  by the given label index, keeping only the corresponding surface points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        -  instance containing only the SurfaceData objects with the specified label index in both
          surface_labels_list and surface_points_list
        """
        new_surface_points_frame_list = deepcopy(self)

        filtered_data = []

        for surface_data_frame in new_surface_points_frame_list._list:
            filtered_surface_data_frame = surface_data_frame.filter_by_cluster_label(label_index)

            filtered_data.append(filtered_surface_data_frame)

        new_surface_points_frame_list._list = filtered_data

        return new_surface_points_frame_list

    def only_filter_by_label(self, label_index) -> None:
        raise NotImplementedError("Cant be used its not filtered in filter by label")



    # def slice_arrays(self, num_of_elements):
    #     self.list = [surface_data.slice_arrays(num_of_elements) for surface_data in self.list]
    #     return self

    # region SPECIAL GETTERS
    def _get_all_points(self):
        """
        Return all points in the list.
        """
        all_points = []
        for surface_data in self._list:
            points_list = surface_data.original_points_list
            # add to all_points all elements of points_list
            all_points.extend(points_list)
        return all_points

    # def get_unique_times(self):
    #     """
    #     Return the set of unique times.
    #     """
    #
    #     return {surface_data.time.value for surface_data in self._list}

    def create_all_frames_all_points_closest_centers_indices(self) -> list:
        """
        Create a list of lists of closest centers indices for each labeled point.
        in format (num_frames, num_points, num_closest_centers) int
        """

        all_frames_closest_centers = []
        for surface_data in self._list:
            if surface_data.normalized_labeled_points_list is None:
                raise ValueError("Labeled points list is empty.")
            all_frames_closest_centers.append(surface_data.normalized_labeled_points_list.create_closest_centers_indicies_list())

        return all_frames_closest_centers

    def get_element_by_time_index(self, time_index : int) -> SurfacePointsFrame | None:
        """
        Find the element in the list with the specified time index.
        """

        def filter_by_time_index(self : SurfacePointsFrameList, time_index: int):
            """
            :param time_index:
            :return:
            """
            filtered_data = []
            for surface_data in self._list:
                if surface_data.time.index == time_index:
                    filtered_data.append(surface_data)

            return SurfacePointsFrameList(filtered_data)

        filtered_list = filter_by_time_index(self, time_index)
        if len(filtered_list._list) == 0:
            return None
        if len(filtered_list._list) > 1:
            raise ValueError("Multiple elements found with the same time index.")

        return filtered_list._list[0]

    # def get_element_by_time_value(self, time_index) -> SurfacePointsFrame:
    #     """
    #     Find the element in the list with the specified time index.
    #     """
    #
    #     def filter_by_time_value(self : SurfacePointsFrameList, time_value):
    #         """
    #         :param time_value:
    #         :return:
    #         """
    #         filtered_data = []
    #         for surface_data in self._list:
    #             if surface_data.time.value == time_value:
    #                 filtered_data.append(surface_data)
    #
    #         return SurfacePointsFrameList(filtered_data)
    #
    #     filtered_list = filter_by_time_value(self, time_index)
    #     if len(filtered_list._list) == 0:
    #         return None
    #     if len(filtered_list._list) > 1:
    #         raise ValueError("Multiple elements found with the same time index.")
    #
    #     return filtered_list._list[0]

    # def get_cluster_labels(self):
    #     """
    #     Return the list of cluster labels.
    #     """
    #     return [label for surface_data in self._list for label in surface_data.labels_list]

    def get_unique_clusters_indexes(self) -> list[int]:
        """
        Return the set of unique clusters.
        """

        def compute_unique_clusters(self: SurfacePointsFrameList):
            """
            Private method to compute unique clusters from the surface data list.
            """
            unique_clusters = set()
            for surface_data in self._list:
                if surface_data.labels_list is None or not surface_data.labels_list or surface_data.labels_list[0] == None:
                    raise ValueError("Labels list is empty.")

                unique_clusters.update(
                    int(label) for label in surface_data.labels_list)  # Convert each sub-array to a tuple
            return unique_clusters

        unique_clusters = compute_unique_clusters(self)

        if unique_clusters is None or not unique_clusters:
            raise Exception("Unique clusters is Empty")

        unique_clusters = sorted(unique_clusters)

        return unique_clusters

    # def select_random_points(self, num_points):
    #     """
    #     Select a random subset of points from each SurfaceData object in the list.
    #
    #     Parameters:
    #     - num_points: int, the number of points to select
    #
    #     Returns:
    #     -  instance containing only the selected points
    #     """
    #     raise NotImplementedError("This method is not implemented yet.")
    #     #
    #     # if num_points <= 0:
    #     #     raise ValueError("Number of points must be greater than 0.")
    #     # if num_points >= len(self.list[0].points_list):
    #     #     raise ValueError("Number of points must be less than the total number of points.")
    #     #
    #     # selected_data = []
    #     #
    #     # for surface_data in self.list:
    #     #     # Randomly select a subset of points
    #     #     selected_indices = np.random.choice(len(surface_data.points_list), num_points, replace=False)
    #     #     selected_points = [surface_data.points_list[i] for i in selected_indices]
    #     #     selected_labels = [surface_data.labels_list[i] for i in selected_indices]
    #     #
    #     #     # Create a new SurfaceData instance with the selected points
    #     #     selected_data.append(SurfacePointsFrame(selected_points, selected_labels, surface_data.time))
    #     #
    #     # return SurfacePointsFrameList(selected_data)

    def get_time_list(self):
        """
        Return the list of time frames.
        """
        return [surface_data_element.time for surface_data_element in self._list]

    def get_normalized_meshes_list(self) -> MeshList:
        """
        Return the list of meshes.
        if some none mesh is found, raise ValueError
        and check if index of output list of mesh is same as time index
        """
        meshes_list = MeshList()
        for i, surface_data in enumerate(self._list):
            surface_data : SurfacePointsFrame = surface_data
            if surface_data.normalized_mesh is None:
                raise ValueError("Mesh is not loaded.")
            meshes_list.append((surface_data.time.index, surface_data.normalized_mesh))

        return meshes_list

    def get_original_meshes_list(self) -> MeshList:
        """
        Return the list of original meshes.
        if some none mesh is found, raise ValueError
        and check if index of output list of mesh is same as time index
        """
        meshes_list = MeshList()
        for i, surface_data in enumerate(self._list):
            if surface_data.original_mesh is None:
                raise ValueError("Original mesh is not loaded.")
            meshes_list.append((surface_data.time.index, surface_data.original_mesh))

        return meshes_list
    # endregion



    def compute_centers_pca_aproximations(self):
        """
        Compute the PCA centers approximation for each object in the list.
        """
        centers_points_list = self._centers_points_list

        # matrix with samples (rows, size: center indexs) - center index; features (columns, size: time_indexes) - position of the centers (with index) in all times
        centers_points_matrix = centers_points_list.convert_to_nparray()

        # Compute PCA approximation
        data_std = StandardScaler().fit_transform(centers_points_matrix)

        # Apply PCA
        pca = PCA(n_components=N)  # N = number of components you want
        centers_feature_vectors = pca.fit_transform(data_std)



def surfacepointsframelist_group_by_label(data : SurfacePointsFrameList) -> dict[int, SurfacePointsFrameList]:
    """
    Group the SurfaceData objects by their labels.

    Returns:
    - A dictionary where the keys are label indices and the values are lists of SurfaceData objects with that label.
    """
    raise NotImplementedError("This function is not implemented yet.")
    # grouped_data = {}
    #
    # # find in data.public_list[0] unique labels from label list and add to grouped_data deep copy of data
    # labels = data.public_list[0].labels_list
    # if labels is None or not labels:
    #     raise ValueError("Labels list is empty.")
    # # find unique labels
    # unique_labels = set(labels)
    # for label in unique_labels:
    #     grouped_data[label] = deepcopy(data)
    #
    # # filter points by label
    # for label, surface_data in grouped_data.items():
    #     surface_data.only_filter_by_label(label)
    #
    # return grouped_data




# def time_frame_list_find_closest_element_index(time_frame_list: List[TimeFrame], time_value: float) -> int:
#     if time_value == 0.0:
#         return 0
#
#     epsilon = (1 / len(time_frame_list)) / 4
#     closest_list = []
#     for time_frame in time_frame_list:
#         if abs(time_frame.value - time_value) < epsilon:
#             return time_frame.index
#
#     raise ValueError("Index not found")

@dataclass
class LossFunctionInfo:
    """Class to hold information about a loss function."""
    meshes_list: MeshList | None = None
    device: any = None
    time_list: list[TimeFrame] | None = None
    data_cluster: SurfacePointsFrameList | None = None
    data : SurfacePointsFrameList | None = None
    closest_centers_indicies_all_frames : np.ndarray | None = None
    """in the shape of (number_of_frames, number_of_points, number_of_closest_centers) int"""
