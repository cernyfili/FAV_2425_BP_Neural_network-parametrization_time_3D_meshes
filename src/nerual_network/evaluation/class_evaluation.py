class PairPointCenterPoint:
    """
    Class representing a pair of one original point and its one closest center point.
    """

    # point, center, list of EvaluationElement
    def __init__(self, point, center_point, distance, id, time, point_label):
        self.points = point
        self.centers = center_point
        self.time = time
        self.distance = distance
        self.id = id
        self.label = point_label


class PairPointCenterPointList:
    def __init__(self, evaluation_points_list):
        self.list = evaluation_points_list

    def filter_by_point_clusterlabel(self, label):
        return PairPointCenterPointList([point for point in self.list if point.label == label])

    def get_points_list(self):
        return [point.points_allframes for point in self.list]

    def append(self, evaluation_point: PairPointCenterPoint):
        self.list.append(evaluation_point)


class DecoderElement:
    def __init__(self, pair_processed_center: PairPointCenterPointList, decoder_time: int):
        self.pair_processed_center = pair_processed_center
        self.time = decoder_time


class DecoderPairList:
    def __init__(self, decoder_pair_list):
        self.list = decoder_pair_list

    def append(self, decoder_element: DecoderElement):
        self.list.append(decoder_element)

    def get_decoder_element_by_id(self, id):
        list = PairPointCenterPointList([])
        for element in self.list:
            for pair in element.pair_processed_center._list:
                if pair.id == id:
                    list.append(pair)
        return list

    def get_unique_ids(self):
        return list(set([pair.id for element in self.list for pair in element.pair_processed_center._list]))


class EvaluationResult:
    # pair original, encoder time, list of elements which are (pair processed, decoder time)
    def __init__(self, pair_original_center: PairPointCenterPointList, encoder_time: int,
                 decoder_pair_list: DecoderPairList):
        self.pair_original_center = pair_original_center
        self.encoder_time = encoder_time
        self.decoder_pair_list = decoder_pair_list


class EvaluationResultList:
    def __init__(self, evaluation_result_list):
        self.list = evaluation_result_list

    def append(self, evaluation_result: EvaluationResult):
        self.list.append(evaluation_result)
