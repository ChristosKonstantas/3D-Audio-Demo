import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple, Union

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import resample_poly

import sofa


LOGGER = logging.getLogger("binaural_renderer")


class BinauralRenderError(Exception):
    """Raise when binaural rendering cannot be completed."""


@dataclass(frozen=True)
class RenderRequest:
    input_wav: Path
    output_wav: Path
    sofa_path: Path
    azimuth: float
    elevation: float = 0.0
    target_sr: Union[int, None] = None
    normalize: bool = True


@dataclass(frozen=True)
class Direction:
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class HrirSelection:
    left: np.ndarray
    right: np.ndarray
    sample_rate: int
    matched_azimuth: float
    matched_elevation: float
    index: int


class HrirProvider(Protocol):
    def get_hrir(self, direction: Direction, target_sr: Union[int, None] = None) -> HrirSelection:
        ...


class AudioIO:
    """Utility methods for reading, writing, and preparing audio."""

    @staticmethod
    def read_mono(path: Path) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(str(path), always_2d=False)
        samples = np.asarray(data, dtype=np.float32)

        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)

        if samples.ndim != 1:
            raise BinauralRenderError(f"Unsupported audio shape for '{path}': {samples.shape}")

        return samples, sample_rate

    @staticmethod
    def write_stereo(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise BinauralRenderError("Output audio must be a stereo array of shape (N, 2).")
        sf.write(str(path), audio, sample_rate)

    @staticmethod
    def resample_1d(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return samples.astype(np.float32, copy=False)

        LOGGER.info("Resampling from %d Hz to %d Hz", orig_sr, target_sr)
        return resample_poly(samples, up=target_sr, down=orig_sr).astype(np.float32, copy=False)

    @staticmethod
    def normalize_peak(stereo_audio: np.ndarray, peak: float = 0.999) -> np.ndarray:
        max_abs = float(np.max(np.abs(stereo_audio)))
        
        if max_abs == 0.0:
            return stereo_audio.astype(np.float32, copy=False)
        
        return (stereo_audio / max_abs * peak).astype(np.float32, copy=False)


class SofaHrirProvider:
    """SOFA-backed HRIR provider.

   Positions are available in spherical coordinates where:
      - column 0 = azimuth
      - column 1 = elevation
      - column 2 = distance (optional for matching)
    """

    def __init__(self, sofa_path: Path) -> None:
        self.sofa_path = sofa_path
        self.database = sofa.Database.open(str(sofa_path))
        self.positions = np.asarray(self.database.Source.Position.get_values(system="spherical"), dtype=np.float32)
        self.sample_rate = int(self.database.Data.SamplingRate.get_values()[0])

        if self.positions.ndim != 2 or self.positions.shape[1] < 2:
            raise BinauralRenderError(f"Unexpected SOFA source positions shape: {self.positions.shape}")

    def get_hrir(self, direction: Direction, target_sr: Union[int, None] = None) -> HrirSelection:
        idx = self._find_best_measurement(direction)

        left = np.asarray(self.database.Data.IR.get_values(indices={"M": idx, "R": 0, "E": 0}), dtype=np.float32)
        right = np.asarray(self.database.Data.IR.get_values(indices={"M": idx, "R": 1, "E": 0}), dtype=np.float32)

        sample_rate = self.sample_rate
        if target_sr is not None and target_sr != sample_rate:
            left = AudioIO.resample_1d(left, sample_rate, target_sr)
            right = AudioIO.resample_1d(right, sample_rate, target_sr)
            sample_rate = target_sr

        matched = self.positions[idx]
        
        return HrirSelection(left=left, right=right, sample_rate=sample_rate, matched_azimuth=float(matched[0]), matched_elevation=float(matched[1]), index=idx)

    def _find_best_measurement(self, direction: Direction) -> int:
        target_az = self._wrap_azimuth(direction.azimuth)
        target_el = float(direction.elevation)

        azimuths = np.array([self._wrap_azimuth(v) for v in self.positions[:, 0]], dtype=np.float32)
        elevations = self.positions[:, 1].astype(np.float32)

        azimuth_distance = np.array([self._angular_distance_deg(target_az, az) for az in azimuths], dtype=np.float32)
        elevation_distance = np.abs(elevations - target_el)

        combined_distance = azimuth_distance + 0.5 * elevation_distance
        return int(np.argmin(combined_distance))

    @staticmethod
    def _wrap_azimuth(angle_deg: float) -> float:
        return float(angle_deg % 360.0)

    @staticmethod
    def _angular_distance_deg(a: float, b: float) -> float:
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)


class BinauralRenderer:
    """Renders mono source audio to stereo binaural output using HRIR convolution."""

    def __init__(self, hrir_provider: HrirProvider) -> None:
        self.hrir_provider = hrir_provider

    def render(self, request: RenderRequest) -> Tuple[np.ndarray, int, HrirSelection]:
        source, source_sr = AudioIO.read_mono(request.input_wav)

        target_sr = request.target_sr or source_sr
        if source_sr != target_sr:
            source = AudioIO.resample_1d(source, source_sr, target_sr)
            source_sr = target_sr

        direction = Direction(azimuth=request.azimuth, elevation=request.elevation)
        hrir = self.hrir_provider.get_hrir(direction=direction, target_sr=target_sr)

        left = signal.fftconvolve(source, hrir.left, mode="full")
        right = signal.fftconvolve(source, hrir.right, mode="full")
        stereo = np.column_stack((left, right)).astype(np.float32, copy=False)

        if request.normalize:
            stereo = AudioIO.normalize_peak(stereo)

        return stereo, source_sr, hrir

    def render_to_file(self, request: RenderRequest) -> HrirSelection:
        stereo, sample_rate, hrir = self.render(request)
        request.output_wav.parent.mkdir(parents=True, exist_ok=True)
        AudioIO.write_stereo(request.output_wav, stereo, sample_rate)

        return hrir