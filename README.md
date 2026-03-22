# Video Pipeline

A production-grade video and audio translation pipeline that processes video frames through OCR, subtitle detection, translation, color remapping, face fusion, and audio dubbing stages.

**Video Pipeline v3.2.0** | **Audio Pipeline v4.2.1** | **Python 3.11.8+**

## Features

- **End-to-end video translation** — OCR extraction, subtitle tracking, translation, and re-rendering onto video frames
- **Audio dubbing pipeline** — Source separation, voice analysis, TTS generation, duration alignment, RMS normalization, and final mixing
- **Face fusion engine** — Configurable face detection and fusion with single/multi-face modes
- **Color remapping** — HSV/LAB engine modes with configurable hue mapping ranges
- **Checkpoint & resume** — Stage-level checkpoint persistence allows interrupted runs to resume from the last completed stage
- **Observability** — Stage timing, structured metrics collection, failure capture middleware, and debug report generation
- **Immutable DTOs** — All data transfer objects use frozen dataclasses with post-init validation
- **Schema validation** — Registry-based validation for all audio pipeline contracts (49 validated fields)
- **Run isolation** — Each pipeline run gets a dedicated folder structure for artifacts, checkpoints, and logs

## Architecture

```
CLI (main.py)
 │
 ├── ConfigLoader ──→ ResolvedConfig (frozen)
 │
 ├── RunFolderManager ──→ run_<hash>/
 │   ├── artifacts/
 │   ├── checkpoints/
 │   └── logs/
 │
 ├── CheckpointRunner
 │   ├── VideoPipeline.run()
 │   │   ├── ROI Detection (M4)
 │   │   ├── OCR (M5)
 │   │   ├── Subtitle Tracking (M6)
 │   │   ├── Translation Buffer/Cache/Adapter (M7-M9)
 │   │   ├── Recolor Engine (M11)
 │   │   ├── Face Fusion Engine (M12)
 │   │   ├── Subtitle Renderer (M13)
 │   │   ├── FPS Normalizer (M14)
 │   │   └── Threaded backbone: Reader → Processor → Encoder (M15-M18)
 │   │
 │   └── AudioPipeline.run()
 │       ├── Audio Extraction (M20.1)
 │       ├── Source Separation (M20.2)
 │       ├── Voice Analysis / ASR (M20.3)
 │       ├── Subtitle Alignment (M20.4)
 │       ├── TTS Generation (M20.5)
 │       ├── Duration Alignment (M20.6)
 │       ├── RMS Processing (M20.7)
 │       └── Audio Mixing (M20.8)
 │
 └── FailureCaptureMiddleware ──→ DebugReport (JSON)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd video-pipeline

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic run

```bash
python main.py --input video.mp4 --output results/
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `""` | Input video file path |
| `--output` | `outputs` | Output directory |
| `--config` | `None` | JSON configuration file |
| `--enable-video` | `true` | Enable video pipeline |
| `--enable-audio` | `true` | Enable audio pipeline |
| `--log-level` | `INFO` | Logging level |
| `--fps` | `30.0` | Target FPS |
| `--frame-count` | `10` | Number of frames to process |
| `--face-fusion-enabled` | `false` | Enable face fusion |
| `--face-fusion-mode` | `single` | Face fusion mode (`single`, `multi`, `all`, etc.) |
| `--face-fusion-strength` | `1.0` | Fusion strength `[0.0, 1.0]` |
| `--face-fusion-min-confidence` | `0.5` | Minimum face detection confidence |

### JSON configuration

```bash
python main.py --config config.json
```

Configuration is resolved with priority: **CLI args > JSON file > defaults**.

## Project Structure

```
video-pipeline/
├── main.py                  # CLI entry point & pipeline orchestrator
├── video_pipeline.py        # Video pipeline: DTOs, modules M2-M19
├── audio_pipeline.py        # Audio pipeline: DTOs (ADM-01→15), modules M20.1-M20.8
├── config_loader.py         # Configuration, logging, observability, versioning
├── requirements.txt         # Python dependencies
├── core/
│   ├── runtime/
│   │   ├── run_folder_manager.py    # Run directory structure management
│   │   └── checkpoint_runner.py     # Stage checkpoint persistence & resume
│   └── observability/
│       ├── failure_middleware.py     # Failure capture & debug report trigger
│       ├── debug_report.py          # Structured JSON debug reports
│       └── metrics_instrumentation.py  # Stage metrics decorator
└── docs/
    ├── README.md             # Technical architecture documentation
    ├── architecture/
    │   └── video_pipeline.md # Video Pipeline v3.2.0 freeze spec
    └── freeze/
        ├── video_freeze.md   # Video pipeline validation & architecture
        └── audio_freeze.md   # Audio Pipeline v4.2.1 freeze spec
```

## License

See repository for license details.
