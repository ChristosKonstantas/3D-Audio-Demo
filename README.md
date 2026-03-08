# About

This project provides a Python command-line tool for applying binaural rendering to PCM WAV audio using HRTF data stored in SOFA files.

It takes an input WAV file, selects the nearest available HRTF for a user-defined direction (azimuth and elevation), convolves the signal with the corresponding left and right HRIRs, and writes the result as a stereo binaural WAV file.

The tool is designed to be:
- simple to use from the command line
- cleanly structured and class-based
- easy to extend with new HRTF backends or rendering options

This tool is intended for headphone playback, where a dry audio source is spatialized so it appears to come from a specified direction around the listener. The rendering pipeline converts the source to mono when needed, resamples audio to a common sample rate, applies HRIR convolution per ear, and exports the final stereo binaural signal.

## Main features

- Input: mono or stereo WAV audio
- Output: stereo binaural WAV audio
- Direction control from the CLI with azimuth and elevation
- Automatic resampling when source and HRTF sample rates differ
- Nearest-direction lookup inside the SOFA dataset
- FFT-based convolution for efficient rendering
- Optional output normalization to avoid clipping

# Instructions

```
git clone git@github.com:ChristosKonstantas/3D-Audio-Demo.git
```

## Windows

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python binaural_audio_mix.py
```

## Unix/Linux/Mac

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 binaural_audio_mix.py 
```

## SOFA dataset used

[SOFA Far-Field](https://zenodo.org/records/3928400)

## Renderer CLI

```bash
python render.py --help
```

```text
usage: binaural-cli [-h] -i INPUT -o OUTPUT -s SOFA -a AZIMUTH [-e ELEVATION] [--sample-rate SAMPLE_RATE] [--no-normalize] [-v]

Render a WAV file to binaural stereo using a SOFA HRIR dataset.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input WAV file.
  -o OUTPUT, --output OUTPUT
                        Path to the output WAV file.
  -s SOFA, --sofa SOFA  Path to the SOFA HRIR file.
  -a AZIMUTH, --azimuth AZIMUTH
                        Target azimuth in degrees. Example: 0=front, 90=left, 180=back, 270=right.
  -e ELEVATION, --elevation ELEVATION
                        Target elevation in degrees. Default: 0.
  --sample-rate SAMPLE_RATE
                        Optional output sample rate. Default: keep input sample rate.
  --no-normalize        Disable peak normalization on output.
  -v, --verbose         Enable verbose logging.
```

### Examples

#### Front
```bash
python render.py -i input/dry_input.wav -o output/front.wav -s path/to/example.sofa -a 0 -e 0
```
#### Left
```bash
python render.py -i input/dry_input.wav -o output/left.wav -s path/to/example.sofa -a 90 -e 0
```

#### Back-left with elevation

```bash
python render.py -i input/dry_input.wav -o output/back_left_up.wav -s data/hrtf/your_dataset.sofa -a 120 -e 45
```

#### Direction convention for azimuth (in degrees)

- `0` = front
- `90` = left
- `180` = back
- `270` = right
- negative azimuths are wrapped modulo 360, so `-90` is equivalent to `270`

#### Direction convention for elevation (in degrees)

- positive values = above ear level
- negative values = below ear level
- `0` = ear level

**Note:** the final matched direction depends on the coordinate convention used by the selected SOFA dataset.