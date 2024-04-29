import os
import cv2
import typing
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(False)

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from . import Image

class ImageReader:
    """Read image with cv2 from path and return image and label"""
    def __init__(self, method: int = cv2.IMREAD_COLOR, log_level: int = logging.INFO) -> None:
        self._method = method
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def __call__(self, image_path: typing.Union[str, np.ndarray], label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Read image with cv2 from path and return image and label
        
        Args:
            image_path (typing.Union[str, np.ndarray]): Path to image or numpy array
            label (Any): Label of image

        Returns:
            Image: Image object
            Any: Label of image
        """
        if isinstance(image_path, str):
            # check whether image_path exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = Image(image = image_path, method = self._method)
        if image.image is None:
            image = None
            self.logger.warning(f"Image {image_path} could not be read, returning None.")

        return image, label

class WavReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """
    def __init__(
        self, 
        frame_length: int=256, 
        frame_step: int=160,
        fft_length: int=384,
        *args, **kwargs
        ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = librosa.load(wav_path) 

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram

    @staticmethod
    def plot_raw_audio(wav_path: str, title:str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
        """
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = librosa.load(wav_path, sr=sr)

        duration = len(audio) / orig_sr

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title('Audio Plot')
        plt.ylabel('signal wave')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T
        
        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.title(f'Spectrogram: {title}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length), label