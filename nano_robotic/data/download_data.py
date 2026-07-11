import glob
from pathlib import Path
import argparse
import datetime
import os
import random
import uuid
import draccus
import ray
import tempfile
from nano_robotic.data.lerobot import LeRobotConverter
from nano_robotic.utils.file_utils import check_directory_has_files_with_substring


DROID_CAMERAS = [
"observation.images.exterior_image_1_left",
"observation.images.exterior_image_2_left",
"observation.images.wrist_image_left",
]
COMPAT_ROOT = Path("asset/droid_100_minimal/lerobot_compat").resolve()
PREPROC_ROOT = Path("asset/droid_100_minimal/preprocessed").resolve()
PREPROC_MANIFEST = PREPROC_ROOT / "shards/manifest.jsonl"
PREPROC_STATS = PREPROC_ROOT / "shards/stats.json"
parser = argparse.ArgumentParser(description="Pretrain base model")
parser.add_argument("--source_episodes", type=str, default="['{COMPAT_ROOT}']", help="")
parser.add_argument("--output_dir", type=str, default=str(PREPROC_ROOT), help="")
parser.add_argument("--camera_names", type=str, default=str(DROID_CAMERAS), help="")
parser.add_argument("--observation_keys", type=str, default="['observation.state']", help="")
parser.add_argument("--action_keys", type=str, default="['action']", help="")
parser.add_argument("--samples_per_shard", type=int, default=32, help="")
parser.add_argument("--max_episodes_to_process", type=int, default=2, help="")

args = parser.parse_args()


@ray.remote(memory=2 * 1024 * 1024 * 1024)  # 2GB per episode worker
def streaming_episode_worker(episode_path: str, converter, statistics_ray_actor, logger_actor):
    return converter.process_episode(episode_path, statistics_ray_actor, logger_actor)


def download_droid_robotics():
    #TODO implementation
    
def preprocess():
    cfg = args

    # Safety check: ensure output directory doesn't have existing preprocessing outputs
    frames_dir = os.path.join(cfg.output_dir, "frames")
    existing_episode_files = check_directory_has_files_with_substring(frames_dir, "_frame_")
    if existing_episode_files:
        error_msg = (
            f"❌ ERROR: Output directory is not empty!\n"
            f"The output directory contains {len(existing_episode_files)} existing episode files':\n"
            f"  Output directory: {cfg.output_dir}\n"
            f"  Example files: {', '.join(existing_episode_files[:5])}"
            f"{'...' if len(existing_episode_files) > 5 else ''}\n"
            f"Pre-processing in a non-empty output directory is unsafe"
        )
        raise RuntimeError(error_msg)

    # Initialize Ray - forward AWS credentials to workers when needed for S3 I/O
    runtime_env = {"env_vars": {"MPLBACKEND": "agg"}}

    # Capture the user who launched the job (head node user, not worker node user)
    if os.environ.get("USER"):
        runtime_env["env_vars"]["VLA_LAUNCHED_BY"] = os.environ["USER"]

    ray.init(
        num_cpus=cfg.ray_num_cpus,
        runtime_env=runtime_env
        | {
            "excludes": [
                ".git",
                "*.pt",
                "*.pyc",
                "__pycache__",
                ".pytest_cache",
                "/data/",
                "/gitui",
                "/tests/essential/test_assets/",
                "/worktrees/",
            ]
        },
    )
    print(f"Started new local Ray cluster with num_cpus={cfg.ray_num_cpus}")

    # Create converter
    converter = LeRobotConverter(cfg)

    # Create the derived output subdirectory that may update the full output path
    output_subdir = cfg.output_dir.rstrip("/")

    # Point the converter's output to the subdirectory so process_episode
    # writes frames to the same location that create_shard reads from
    converter.output_dir = output_subdir

    # Discover episodes
    print("🔍 Discovering episodes...")
    episodes = converter.discover_episodes(cfg.source_episodes, cfg.max_episodes_to_process)
    print(f"Found {len(episodes)} episodes")
    if len(episodes) == 0:
        print("❌ No episodes found!")
        return

    # Create initial processing metadata
    metadata = create_processing_metadata(cfg, episodes)
    metadata["processing"]["timestamp_start"] = datetime.datetime.now().isoformat()

    # Ray Phase 1: Process frame individually and upload to S3
    print(f"🚀 Processing {len(episodes)} episodes and uploading to S3...")
    statistics_ray_actor = None
    logger_actor = LoggerActor.remote()

    futures = [
        streaming_episode_worker.remote(episode, converter, statistics_ray_actor, logger_actor) for episode in episodes
    ]
    results = ray.get(futures)
    results = [result for result in results if result is not None]  # Remove None results
    results = [i for result in results for i in result]  # Result is a list of lists, flatten it
    print("✅ Upload phase complete! Starting sharding phase...")

    # Ray Phase 2: Shuffle and group files into shards in parallel
    random.shuffle(results)
    shards = [results[i : i + cfg.samples_per_shard] for i in range(0, len(results), cfg.samples_per_shard)]
    print(f"Creating {len(shards)} shards with up to {cfg.samples_per_shard} samples each")
    shard_futures = [create_shard.remote(shard_files, i, output_subdir) for i, shard_files in enumerate(shards)]
    shard_results = ray.get(shard_futures)
    print(f"✅ Created {len(shard_results)} shards.")

    # Ray Phase 3: Group files by episode and create episode-based shards
    episode_groups = {}
    for filename in results:
        # filename format: {unique_id}_{episode_id}_frame_{frame_idx}.tar
        episode_key = filename.rsplit("_frame_", 1)[0]
        episode_groups.setdefault(episode_key, []).append(filename)
    print(f"Creating {len(episode_groups)} episode shards")
    episode_shard_futures = [
        create_episode_shard.remote(files, episode_key, output_subdir) for episode_key, files in episode_groups.items()
    ]
    episode_shard_results = ray.get(episode_shard_futures)
    print(f"✅ Created {len(episode_shard_results)} episode shards.")

    # Upload episode manifest to S3 in the episodes/ directory
    episode_manifest_lines = []
    for shard_name, num_sequences in episode_shard_results:
        episode_manifest_lines.append({"shard": shard_name, "num_sequences": num_sequences})
    save_and_upload_dict(episode_manifest_lines, f"{output_subdir}/episodes", "manifest.jsonl")

    # Upload shards manifest to S3 in the shards/ directory
    manifest_lines = []
    for shard_name, num_sequences in shard_results:
        manifest_entry = {"shard": shard_name, "num_sequences": num_sequences}
        manifest_lines.append(manifest_entry)
    save_and_upload_dict(manifest_lines, f"{output_subdir}/shards", "manifest.jsonl")

    # Upload statistics to S3 in the shards/ directory and the episodes/ directory
    # if cfg.compute_statistics:
    #     statistics_state = statistics_ray_actor.get_statistics.remote()
    #     statistics_state = ray.get(statistics_state)
    #     save_and_upload_dict(statistics_state, f"{output_subdir}/shards", "stats.json")
    #     save_and_upload_dict(statistics_state, f"{output_subdir}/episodes", "stats.json")

    # Update and save processing metadata with final statistics
    metadata["processing"]["total_samples_created"] = sum(num_sequences for _, num_sequences in shard_results)
    metadata["processing"]["timestamp_end"] = datetime.datetime.now().isoformat()
    metadata["processing"]["sample_counts"] = ray.get(logger_actor.get_values.remote())
    print("Sample counts:", metadata["processing"]["sample_counts"])
    save_and_upload_dict(metadata, f"{output_subdir}/shards", "processing_metadata.json")
    preprocessing_config_dict = vars(cfg).copy()
    save_and_upload_config(preprocessing_config_dict, f"{output_subdir}/shards", "preprocessing_config.yaml")

    # Make a copy of the output directory when source/destination backends match
    dataset_uuid = str(uuid.uuid4())
    fixed_path = f"{cfg.output_dir_fixed_path.rstrip('/')}/{dataset_uuid}"

    ray.shutdown()
    print("🎉 Complete! All samples uploaded and sharded.")
   

def create_processing_metadata(args: PreprocessParams, episodes: list[str]) -> dict[str, Any]:
    """Create comprehensive metadata about the processing run."""

    # Get command line information
    command_line = {
        "script_name": sys.argv[0],
        "full_command": " ".join(sys.argv),
        "arguments": _draccus_encoding.encode(args),
    }

    # Get environment information
    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "timestamp_captured": datetime.datetime.now().isoformat(),
    }

    # Get git information (skip if testing flag is set)
    if args.skip_git_tagging:
        git_info = {"skip_git_tagging": True, "commit_hash": "test", "branch": "test"}
    else:
        git_info = get_git_info(auto_tag=args.auto_tag)

    # Get source data information
    source_data_info = get_source_data_info(args.source_episodes, episodes)

    # Package versions (try to get key dependencies)
    try:
        # Use modern importlib.metadata instead of deprecated pkg_resources
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:
            # Fallback for Python < 3.8
            from importlib_metadata import PackageNotFoundError, version

        key_packages = ["numpy", "fsspec", "PIL", "tqdm", "boto3", "webdataset"]
        package_versions = {}
        for pkg in key_packages:
            try:
                # Handle special case for PIL package name
                pkg_name = "Pillow" if pkg == "PIL" else pkg
                package_versions[pkg] = version(pkg_name)
            except PackageNotFoundError:
                package_versions[pkg] = "not_found"
            except Exception:
                package_versions[pkg] = "unknown"
        environment["package_versions"] = package_versions
    except ImportError:
        # If importlib.metadata is not available, fall back gracefully
        environment["package_versions"] = "unavailable_importlib_metadata_missing"
    except Exception:
        environment["package_versions"] = "unavailable"

    # Create reproducibility instructions based on git state
    reproducibility_notes = []

    if git_info.get("preprocessing_tag"):
        # If we created a tag, use that for reproduction
        reproducibility_notes.extend(
            [
                f"EXACT REPRODUCTION: Use git tag '{git_info['preprocessing_tag']}'",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info['preprocessing_tag']}",
                f"  {command_line['full_command']}",
                "",
                "This tag captures the exact code state including uncommitted changes used for this dataset.",
            ]
        )
    elif git_info.get("has_uncommitted_changes"):
        # If there are uncommitted changes but no tag was created
        reproducibility_notes.extend(
            [
                f"WARNING: Dataset created with uncommitted changes to commit {git_info.get('commit_hash', 'unknown')}",
                "For exact reproduction, the following files had uncommitted changes:",
            ]
        )
        for file in git_info.get("preprocessing_related_files", []):
            reproducibility_notes.append(f"  - {file}")
        reproducibility_notes.extend(
            [
                "",
                "Basic reproduction (may differ due to uncommitted changes):",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )
    else:
        # Clean state - straightforward reproduction
        reproducibility_notes.extend(
            [
                "CLEAN REPRODUCTION: No uncommitted changes",
                "Commands to reproduce:",
                f"  git clone {git_info.get('remote_url', 'REPO_URL')}",
                f"  git checkout {git_info.get('commit_hash', 'COMMIT_HASH')}",
                f"  {command_line['full_command']}",
            ]
        )

    reproducibility_notes.extend(
        [
            "",
            "Additional requirements:",
            "- Ensure the source data at the specified paths is unchanged (check episode_list_hash)",
            "- Use the same package versions if possible for identical results",
            "- Use the same hardware/OS for identical performance characteristics",
        ]
    )

    # Combine all metadata
    metadata = {
        "metadata_version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "command_line": command_line,
        "environment": environment,
        "git_info": git_info,
        "source_data": source_data_info,
        "processing": {},
        "reproducibility_notes": reproducibility_notes,
    }

    return metadata

    
@ray.remote
def create_episode_shard(shard_files: list[str], episode_key: str, output_dir: str) -> str:
    """Download/read tar files and create an episode-based shard. Supports both S3 and local filesystem."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key
        download_tar = partial(_download_tar_from_s3, s3_client=s3_client, bucket_name=bucket_name, s3_prefix=s3_prefix)
    else:
        frames_dir = Path(output_dir) / "frames"

        def download_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Download/read all tars in parallel
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_tar, s3_key) for s3_key in shard_files]
        for future in as_completed(futures):
            s3_key, obj_buffer = future.result()
            downloaded_tars[s3_key] = obj_buffer

    # Sort files by frame index to maintain temporal order within episode
    def get_frame_idx(filename):
        # filename format: {unique_id}_{episode_id}_frame_{frame_idx}.tar
        return int(filename.rsplit("_frame_", 1)[1].replace(".tar", ""))

    sorted_files = sorted(shard_files, key=get_frame_idx)

    # Create shard by combining all downloaded tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        for s3_key in sorted_files:
            obj_buffer = downloaded_tars[s3_key]
            obj_buffer.seek(0)

            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Save shard
    shard_buffer.seek(0)
    shard_key = f"episode_{episode_key}.tar"

    if is_s3:
        s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/episodes/{shard_key}")
        print(f"Uploaded episode shard {shard_key} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/episodes/{shard_key}")
    else:
        episodes_dir = Path(output_dir) / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        shard_path = episodes_dir / shard_key
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved episode shard {shard_key} to {shard_path}")

    return (shard_key.rstrip(".tar"), len(shard_files))


@ray.remote
def create_shard(shard_files: list[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard. OPTIMIZED with parallel downloads."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key

        def read_tar(tar_key):
            """Download a single tar file from S3."""
            obj_buffer = io.BytesIO()
            full_key = f"{s3_prefix.rstrip('/')}/frames/{tar_key}"
            s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
            obj_buffer.seek(0)
            return (tar_key, obj_buffer)
    else:
        frames_dir = Path(output_dir) / "frames"

        def read_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Read all tars in parallel (use 5 threads. reduced concurrency to avoid S3 throttling)
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(read_tar, tar_key) for tar_key in shard_files]
        for future in as_completed(futures):
            tar_key, obj_buffer = future.result()
            downloaded_tars[tar_key] = obj_buffer

    # Create shard by combining all tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        # Process in original order for consistency
        for tar_key in shard_files:
            obj_buffer = downloaded_tars[tar_key]
            obj_buffer.seek(0)

            # Extract contents and add to shard
            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Save shard
    shard_buffer.seek(0)
    shard_name = f"shard_{shard_idx:06d}.tar"

    if is_s3:
        upload_fileobj_to_s3(
            shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_name}", s3_client=s3_client
        )
        print(f"Uploaded shard {shard_name} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/shards/{shard_name}")
    else:
        shard_path = Path(output_dir) / "shards" / shard_name
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved shard {shard_name} to {shard_path}")

    return (shard_name.rstrip(".tar"), len(shard_files))
    
def save_and_upload_dict(dict_data: dict, output_path: str, file_name: str):
    # Used to upload manifest.jsonl and stats.json (or save locally)
    body = "\n".join(json.dumps(record) for record in dict_data) if "jsonl" in file_name else json.dumps(dict_data)
    # Save to local filesystem
    local_path = Path(output_path) / file_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Saved {file_name} to {local_path}")

def save_and_upload_config(config, output_path: str, file_name: str):
    # Draccus dump to temp file then upload to s3 (or save locally)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
        draccus.dump(config, temp_file)
        temp_path = temp_file.name

    # Save to local filesystem
    local_path = Path(output_path) / file_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(temp_path, local_path)
    print(f"Saved {file_name} to {local_path}")
    

import json
from typing import Any

import numpy as np
import ray
from tdigest_rs import TDigest


@ray.remote
class LoggerActor:
    def __init__(self):
        self.total_potential_samples = 0
        self.still_samples_filtered = 0
        self.padding_samples_filtered = 0

    def get_values(self):
        return {
            "total_potential_samples": self.total_potential_samples,
            "still_samples_filtered": self.still_samples_filtered,
            "padding_samples_filtered": self.padding_samples_filtered,
        }

    def increment_total_potential_samples(self):
        self.total_potential_samples += 1

    def increment_still_samples_filtered(self):
        self.still_samples_filtered += 1

    def increment_padding_samples_filtered(self):
        self.padding_samples_filtered += 1    

def main():
    download_droid_robotics()
    preprocess()

    
if __name__ == "__main__":
    main()