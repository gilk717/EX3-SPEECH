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
                    dist = torch.dist(train_sample.squeeze(0), test_sample).item()
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
        m = test_data.shape[1]
        results = [[numpy.inf for _ in range(m)] for _ in range(m)]
        results[0][0] = torch.dist(test_data[:, 0], train_data[:, 0]).item()
        for i in range(0, m):
            for j in range(0, m):
                if i == 0 and j != 0:
                    results[i][j] = (
                        torch.dist(test_data[:, i], train_data[:, j]).item()
                        + results[i][j - 1]
                    )
                elif i != 0 and j == 0:
                    results[i][j] = (
                        torch.dist(test_data[:, i], train_data[:, j]).item()
                        + results[i - 1][j]
                    )
                elif i != 0 and j != 0:
                    results[i][j] = torch.dist(
                        test_data[:, i], train_data[:, j]
                    ).item() + min(
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
        return [filename + " - " + str(euc) + " - " + str(dtw) for filename, euc, dtw in zip(audio_files, predict_euc, predict_dtw) ]


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
test_paths = [
    path
    for path in [
        os.path.join(model.path_to_test_data, name)
        for name in sorted(os.listdir(model.path_to_test_data))
    ]
]
# print(model.classify(test_paths))
true_labels = [4, 2, 2, 1, 5, 4, 2, 1, 5, 4, 4,  2, 3, 2, 2, 1, 5, 4, 3 , 4, 4, 4, 4 , 1, 2, 2, 2, 3]
pred = model.classify_using_DTW_distance(test_paths[:len(true_labels)])
e_pred = model.classify_using_eucledian_distance(test_paths[:len(true_labels)])
pred_acc = sum([1 if p == t else 0 for p, t in zip(pred, true_labels)]) / len(pred)
e_pred_acc = sum([1 if p == t else 0 for p, t in zip(e_pred, true_labels)]) / len(e_pred)
print(pred_acc, e_pred_acc)

# print('"' + '",\n "'.join(sorted(test_paths, reverse=True)) + '"')
# test_real_results = [3, 1, 4, 4, 4, 2, 1, 2, 1, 5, 5, 1, 2, 4, 2, 1, 2, 5 , 3, 1]
#
# print(model.classify_using_eucledian_distance([
# "./test_files/fdb56edc-2842-4b10-be22-2bb0bcb800a4.wav",
#  "./test_files/fdb31f76-2f16-4f29-80cb-d24f30e94020.wav",
#  "./test_files/fc627939-9957-4cfc-a6d7-24773953f60e.wav",
#  "./test_files/fbb2e629-b1b0-4b93-893e-55b3b5dd7304.wav",
#  "./test_files/fba325b5-404d-4650-a6c6-7f1d3533091e.wav",
#  "./test_files/fb95fddb-c204-4a9f-925c-280f6c332673.wav",
#  "./test_files/f5032534-cc6a-4fbe-ad96-1379fdedb0cd.wav",
#  "./test_files/f492d24b-31b9-4835-a092-d67c09f83380.wav",
#     "./test_files/f44ea05b-177b-4b65-8c9e-cf77c65de82a.wav",
#     "./test_files/f4ea3cb0-b622-43ab-9cdf-b76d3cb74efd.wav",
#     "./test_files/eedcf0f2-c14f-45d6-bb18-2cf9e14c38c8.wav",
#     "./test_files/ed971c32-4a3e-4f8a-86fb-f5463e10ddb1.wav",
#     "./test_files/ecf82db6-ac99-4b2d-81bf-6ee58fd88d95.wav",
#     "./test_files/ebdc6a9b-328f-47b8-a369-d779a5bf7bb9.wav",
#     "./test_files/eb172156-e393-43e9-8f30-0d629b4f900f.wav",
#     "./test_files/eb2f7016-cd34-4b99-b17b-355683780305.wav",
#     "./test_files/ea0949d7-737c-4714-a634-ccffa8b586f8.wav",
#     "./test_files/ea37a038-e95f-4326-a888-9948d29bd785.wav",
#     "./test_files/ea0e7d98-cb6f-45a7-b6ea-92e0000a2a44.wav",
#     "./test_files/e80018b3-23d0-4964-be92-a616e0243ce0.wav",
#     ]))
#
# print(model.classify_using_DTW_distance([
# "./test_files/fdb56edc-2842-4b10-be22-2bb0bcb800a4.wav",
#  "./test_files/fdb31f76-2f16-4f29-80cb-d24f30e94020.wav",
#  "./test_files/fc627939-9957-4cfc-a6d7-24773953f60e.wav",
#  "./test_files/fbb2e629-b1b0-4b93-893e-55b3b5dd7304.wav",
#  "./test_files/fba325b5-404d-4650-a6c6-7f1d3533091e.wav",
#  "./test_files/fb95fddb-c204-4a9f-925c-280f6c332673.wav",
#  "./test_files/f5032534-cc6a-4fbe-ad96-1379fdedb0cd.wav",
#  "./test_files/f492d24b-31b9-4835-a092-d67c09f83380.wav",
#     "./test_files/f44ea05b-177b-4b65-8c9e-cf77c65de82a.wav",
#     "./test_files/f4ea3cb0-b622-43ab-9cdf-b76d3cb74efd.wav",
#     "./test_files/eedcf0f2-c14f-45d6-bb18-2cf9e14c38c8.wav",
#     "./test_files/ed971c32-4a3e-4f8a-86fb-f5463e10ddb1.wav",
#     "./test_files/ecf82db6-ac99-4b2d-81bf-6ee58fd88d95.wav",
#     "./test_files/ebdc6a9b-328f-47b8-a369-d779a5bf7bb9.wav",
#     "./test_files/eb172156-e393-43e9-8f30-0d629b4f900f.wav",
#     "./test_files/eb2f7016-cd34-4b99-b17b-355683780305.wav",
#     "./test_files/ea0949d7-737c-4714-a634-ccffa8b586f8.wav",
#     "./test_files/ea37a038-e95f-4326-a888-9948d29bd785.wav",
#     "./test_files/ea0e7d98-cb6f-45a7-b6ea-92e0000a2a44.wav",
#     "./test_files/e80018b3-23d0-4964-be92-a616e0243ce0.wav",
#     ]))
# print(test_real_results)
