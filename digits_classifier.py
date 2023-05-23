import os
from abc import abstractmethod

import librosa
import numpy
import torch
import typing as tp
from dataclasses import dataclass


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


class DigitClassifier():
    """
    You should Implement your classifier object here
    """
    word_to_number = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5
    }

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.path_to_test_data = args.path_to_test_data_dir
        self.train_data: tp.List[tp.List[torch.Tensor]] = []

    def load_train_data(self):
        """
        function to load train data
        """
        for train_folder_path in sorted(os.listdir(self.path_to_training_data), key=lambda x: self.word_to_number[x]):
            cur_list = []
            for train_file_path in os.listdir(os.path.join(self.path_to_training_data, train_folder_path)):
                train_paths_file = os.path.join(self.path_to_training_data, train_folder_path, train_file_path)
                audio, sr = librosa.load(train_paths_file)
                cur_list.append(self.extract_mfccs(audio, sr))
            self.train_data.append(cur_list)
        pass

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
        mfcc = librosa.feature.mfcc(y=audio, n_mfcc=20)
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
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        if isinstance(audio_files, list):
            # load audio files from paths
            test_data = DigitClassifier.load_test_data(audio_files)
        else:
            ## TODO: test this
            audio_files = audio_files.squeeze(1)
            test_data = torch.tensor([])
            for test_sample in audio_files:
                cur_test_feats = DigitClassifier.extract_mfccs_from_tensor(test_sample)
                test_data = torch.cat((test_data, cur_test_feats.unsqueeze(0)))
        results = []
        for test_sample in test_data:
            predicted_label = self.classify_sample_using_euclidean(test_sample)
            results.append(predicted_label)
        return results
        # calculate distance

    def classify_sample_using_euclidean(self, test_sample: torch.Tensor) -> int:
        """
        function to classify a given audio using auclidean distance
        test_sample: a tensor of shape [Channels, Time]
        return: predicted label
        """
        distances = []
        for train_data in self.train_data:
            cur_distances = []
            for train_sample in train_data:
                cur_distances.append(torch.dist(train_sample, test_sample).item())
            distances.append(min(cur_distances))
        # return predicted labels
        return distances.index(min(distances)) + 1

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        raise NotImplementedError("function is not implemented")

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        raise NotImplementedError("function is not implemented")


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        raise NotImplementedError("function is not implemented")


model = DigitClassifier(ClassifierArgs())
model.load_train_data()
test_paths = [path for path in
              [os.path.join(model.path_to_test_data, name) for name in os.listdir(model.path_to_test_data)]]
print(model.classify_using_eucledian_distance(test_paths))
test_real_results = [2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 3, 2, 4, 2, 1, 5, 4, 5, 4, 1, 4, 3]
print(test_real_results)
