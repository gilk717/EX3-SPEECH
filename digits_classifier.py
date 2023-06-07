import os
import typing as tp
from abc import abstractmethod
from dataclasses import dataclass

import librosa
import numpy
import torch


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with
    default values (so run won't break when we test this).
    """

    # we will use this to give an absolute path to the data, make sure you read the data using this argument.
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./train_files"
    path_to_test_data_dir: str = "./test_files"

    # you may add other args here


class DigitClassifier:
    """
    You should Implement your classifier object here
    """

    word_to_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.path_to_test_data = args.path_to_test_data_dir
        self.train_data: tp.List[tp.List[torch.Tensor]] = []

    def load_train_data(self):
        """
        function to load train data
        """
        for train_folder_path in sorted(
            {"one", "two", "three", "four", "five"},
            key=lambda x: self.word_to_number[x],
        ):
            cur_list = []
            for train_file_path in os.listdir(
                os.path.join(self.path_to_training_data, train_folder_path)
            ):
                train_paths_file = os.path.join(
                    self.path_to_training_data, train_folder_path, train_file_path
                )
                audio, sr = librosa.load(train_paths_file)
                cur_list.append(self.extract_mfccs(audio, sr))
            self.train_data.append(cur_list)

    @staticmethod
    def load_test_data(paths_to_test_data_dir: tp.List[str]) -> torch.Tensor:
        """
        function to load data for testing
        """
        test_data = torch.tensor([])
        for test_file_path in paths_to_test_data_dir:
            audio, sr = librosa.load(test_file_path)
            test_data = torch.cat((test_data, DigitClassifier.extract_mfccs(audio, sr)))
        return test_data

    @staticmethod
    def extract_mfccs_from_tensor(audio: torch.Tensor) -> torch.Tensor:
        """
        function to extract mfccs from a given audio
        audio: a tensor of shape [Channels, Time]
        return: a tensor of shape [MFCCs, Time]
        """
        numpy_audio = audio.numpy()
        mfcc = librosa.feature.mfcc(y=numpy_audio, n_mfcc=20)
        return torch.tensor(mfcc)

    @staticmethod
    def extract_mfccs(audio: numpy.ndarray, sr) -> torch.Tensor:
        """
        function to extract mfccs from a given audio
        audio: a numpy array of shape [Time]
        return: a tensor of shape [MFCCs, Time]
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        return torch.tensor(mfcc).unsqueeze(0)

    @abstractmethod
    def classify_using_eucledian_distance(
            self, audio_files: tp.Union[tp.List[str], torch.Tensor]
    ) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        return self.classify_digits(audio_files, True)

    @abstractmethod
    def classify_digits(
            self, audio_files: tp.Union[tp.List[str], torch.Tensor], use_euclidean
    ) -> tp.List[int]:
        if isinstance(audio_files, list):
            # load audio files from paths
            test_data = DigitClassifier.load_test_data(audio_files)
        else:
            audio_files = audio_files.squeeze(1)
            test_data = torch.tensor([])
            for test_sample in audio_files:
                cur_test_feats = DigitClassifier.extract_mfccs_from_tensor(test_sample)
                test_data = torch.cat((test_data, cur_test_feats.unsqueeze(0)))
        results = []
        for test_sample in test_data:
            predicted_label = self.classify_sample_using_provided_dist(test_sample, use_euclidean)
            results.append(predicted_label)
        return results

    def classify_sample_using_provided_dist(
            self, test_sample: torch.Tensor, use_euclidean=True
    ) -> int:
        """
        function to classify a given audio using auclidean distance
        test_sample: a tensor of shape [Channels, Time]
        return: predicted label
        """
        distances = []
        for train_data in self.train_data:
            cur_distances = []
            for train_sample in train_data:
                if use_euclidean:
                    dist = torch.sum(torch.pairwise_distance(train_sample.squeeze(0), test_sample, p=2))
                else:
                    dist = self.calculate_dtw_distance(
                        train_sample.squeeze(0), test_sample
                    )
                cur_distances.append(dist)
            distances.append(min(cur_distances))
        # return predicted labels
        return distances.index(min(distances)) + 1

    @abstractmethod
    def classify_using_DTW_distance(
            self, audio_files: tp.Union[tp.List[str], torch.Tensor]
    ) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        return self.classify_digits(audio_files, False)

    def calculate_dtw_distance(self, train_data: torch.Tensor, test_data: torch.Tensor):
        """
        function to calculate the DTW distance between two samples
        return: DTW distance
        """
        # transpose the data since we are indexing it the other way
        m = test_data.shape[0]
        results = [[numpy.inf for _ in range(m)] for _ in range(m)]
        results[0][0] = torch.sum(torch.pairwise_distance(test_data[0], train_data[0]))
        for i in range(0, m):
            for j in range(0, m):
                if i == 0 and j != 0:
                    results[i][j] = torch.sum(
                        torch.pairwise_distance(test_data[i], train_data[j])) + results[i][j - 1]
                elif i != 0 and j == 0:
                    results[i][j] = torch.sum(
                        torch.pairwise_distance(test_data[i], train_data[j])) + results[i - 1][j]
                elif i != 0 and j != 0:
                    results[i][j] = torch.sum(torch.pairwise_distance(
                        test_data[i], train_data[j]
                    )) + min(
                        results[i][j - 1], results[i - 1][j - 1], results[i - 1][j]
                    )
        return results[m - 1][m - 1]

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        predict_dtw = self.classify_using_DTW_distance(audio_files)
        predict_euc = self.classify_using_eucledian_distance(audio_files)
        return [str(os.path.basename(filepath)) + " - " + str(euc) + " - " + str(dtw) for filepath, euc, dtw in
                zip(audio_files, predict_euc, predict_dtw)]


class ClassifierHandler:
    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        model = DigitClassifier(ClassifierArgs())
        model.load_train_data()
        return model


# model = DigitClassifier(ClassifierArgs())
# model.load_train_data()
# results = model.classify([
#         os.path.join(model.path_to_test_data, name)
#         for name in os.listdir(model.path_to_test_data)
#     ])
# # write results to file
# with open("output.txt", "w") as f:
#     f.write("\n".join(results))
