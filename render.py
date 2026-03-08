import argparse
import logging
from pathlib import Path
from typing import Union
from binaural_renderer import BinauralRenderer, RenderRequest, SofaHrirProvider


LOGGER = logging.getLogger("binaural_cli")


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(prog="binaural-cli", description="Render a WAV file to binaural stereo using a SOFA HRIR dataset.")

    parser.add_argument("-i", "--input", required=True, help="Path to the input WAV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output WAV file.")
    parser.add_argument("-s", "--sofa", required=True, help="Path to the SOFA HRIR file.")
    parser.add_argument("-a","--azimuth", required=True,type=float, help="Target azimuth in degrees. Example: 0=front, 90=left, 180=back, 270=right.")
    parser.add_argument("-e", "--elevation", type=float, default=0.0, help="Target elevation in degrees. Default: 0.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional output sample rate. Default: keep input sample rate.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable peak normalization on output.")
    parser.add_argument("-v", "--verbose",action="store_true",help="Enable verbose logging.")
    
    return parser


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def validate_paths(input_wav: Path, sofa_path: Path, sample_rate: Union[int, None] = None)-> None:
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")
    if not sofa_path.exists():
        raise FileNotFoundError(f"SOFA file not found: {sofa_path}")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("--sample-rate must be a positive integer.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    input_wav = Path(args.input).expanduser().resolve()
    output_wav = Path(args.output).expanduser().resolve()
    sofa_path = Path(args.sofa).expanduser().resolve()

    validate_paths(input_wav, sofa_path, args.sample_rate)

    request = RenderRequest(input_wav=input_wav, output_wav=output_wav, sofa_path=sofa_path, 
                            azimuth=args.azimuth, elevation=args.elevation, target_sr=args.sample_rate, 
                            normalize=not args.no_normalize)

    provider = SofaHrirProvider(request.sofa_path)
    renderer = BinauralRenderer(provider)
    hrir = renderer.render_to_file(request)

    LOGGER.info("Input         : %s", request.input_wav)
    LOGGER.info("Output        : %s", request.output_wav)
    LOGGER.info("SOFA dataset  : %s", request.sofa_path)
    LOGGER.info("Requested dir : azimuth=%.2f degrees, elevation=%.2f degrees", request.azimuth, request.elevation)
    LOGGER.info("Matched dir   : azimuth=%.2f degrees, elevation=%.2f degrees (measurement index %d)", hrir.matched_azimuth, hrir.matched_elevation, hrir.index)
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())