#!/usr/bin/env python3
"""
Image Regeneration Pipeline for VisMem-C
==========================================

Generates images for events in events.json and uploads directly to S3.
Fills in img.url with public S3 URLs before downstream processing.

Usage:
  python scripts/regenerate_images.py \
    --out-dir data/metadata \
    --backend imagen \
    --bucket vismem-c-images-aadhi

  python scripts/regenerate_images.py \
    --out-dir data/metadata \
    --backend imagen \
    --bucket vismem-c-images-aadhi \
    --dry-run

  python scripts/regenerate_images.py \
    --out-dir data/metadata \
    --backend imagen \
    --bucket vismem-c-images-aadhi \
    --start-event 0 --num-events 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional 
import backends

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# S3 imports
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("[WARN] boto3 not installed: pip install boto3")

# Image backend
try:
    from backends import get_backend
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    logger.warning("[WARN] backends module not found. See README for setup.")


class ImageRegenerationPipeline:
    """Pipeline for image generation and S3 upload."""

    def __init__(
        self,
        out_dir: str,
        backend_name: str,
        bucket: str,
        dry_run: bool = False,
        replace_external: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.backend_name = backend_name
        self.bucket = bucket
        self.dry_run = dry_run
        self.replace_external = replace_external
        self.s3_bucket_url = f"https://{self.bucket}.s3.amazonaws.com"

        if not self.out_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.out_dir}")

        self.events_path = self.out_dir / "events.json"
        if not self.events_path.exists():
            raise FileNotFoundError(f"File not found: {self.events_path}")

        self.s3_client = None
        if not dry_run and BOTO3_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3')
                logger.info("[✓] S3 client initialized")
            except Exception as e:
                logger.error(f"[✗] Failed to initialize S3: {e}")
                raise

        self.backend = None
        if BACKENDS_AVAILABLE:
            try:
                self.backend = get_backend(backend_name, config={})
                logger.info(f"[✓] Backend '{backend_name}' initialized")
            except Exception as e:
                logger.error(f"[✗] Failed to initialize backend: {e}")
                raise
        else:
            raise RuntimeError("Backends module not available")

        self.stats = {
            'total_events': 0,
            'events_with_images': 0,
            'images_generated': 0,
            'images_uploaded': 0,
            'skipped': 0,
            'errors': 0,
        }

    def load_json(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    def save_json(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[✓] Saved: {path}")
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
            raise

    def get_image_prompt(self, event: Dict[str, Any]) -> Optional[str]:
        if not isinstance(event.get("img"), dict):
            return None

        img_meta = event["img"]

        for field in ['prompt', 'caption', 'description']:
            if field in img_meta and img_meta[field]:
                return str(img_meta[field]).strip()

        if event.get("text"):
            return str(event["text"]).strip()

        return None

    def generate_image(self, prompt: str, event_idx: int) -> Optional[str]:
        """Generate image using backend."""
        if not self.backend:
            logger.error("Backend not initialized")
            return None

        try:
            logger.info(f"[{event_idx}] Generating: {prompt[:70]}...")
            output_dir = str(self.out_dir / "generated_images")
            generated_files = self.backend.generate(
                prompt=prompt,
                count=1,
                output_dir=output_dir
            )

            if generated_files and len(generated_files) > 0:
                filepath = generated_files[0]['filepath']
                gen_time = generated_files[0].get('generation_time', 0.0)
                logger.info(f"[✓] Generated: {Path(filepath).name} ({gen_time:.2f}s)")
                return filepath
            else:
                logger.error(f"[✗] Backend returned no image for event {event_idx}")
                logger.error(f"[DEBUG] Response: {generated_files}")
                return None

        except Exception as e:
            logger.error(f"[✗] Generation failed: {e}")
            logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None

    def upload_to_s3(
        self,
        local_path: str,
        event_idx: int,
        dia_id: Optional[str] = None,
    ) -> Optional[str]:
        if self.dry_run:
            safe_name = Path(local_path).stem
            url = f"https://{self.bucket}.s3.amazonaws.com/vismen/{safe_name}.png"
            logger.info(f"[dry-run] Would upload to: {url}")
            return url

        if not self.s3_client:
            logger.error("[✗] S3 client not available")
            return None

        try:
            if dia_id:
                safe_dia_id = dia_id.replace(':', '_').replace('/', '_').replace('\\', '_')
                s3_key = f"vismen/{safe_dia_id}.png"
            else:
                local_file = Path(local_path)
                s3_key = f"vismen/event_{event_idx}_{local_file.stem}.png"

            logger.info(f"[{event_idx}] Uploading to S3: s3://{self.bucket}/{s3_key}")

            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket,
                Key=s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )

            s3_url = f"https://{self.bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"[✓] Uploaded: {s3_url}")
            return s3_url

        except ClientError as e:
            logger.error(f"[✗] S3 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[✗] Upload error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_events(
        self,
        start_event: int = 0,
        num_events: Optional[int] = None,
    ) -> None:
        events_data = self.load_json(self.events_path)
        events = events_data.get('events', [])

        if not events:
            logger.warning("[!] No events found in events.json")
            return

        self.stats['total_events'] = len(events)

        end_event = len(events)
        if num_events:
            end_event = min(start_event + num_events, end_event)

        logger.info(f"[*] Processing events {start_event}–{end_event-1} (total: {len(events)})")

        for idx in range(start_event, end_event):
            event = events[idx]

            if not isinstance(event.get("img"), dict):
                logger.debug(f"[{idx}] Skipping: no img metadata")
                self.stats['skipped'] += 1
                continue

            current_url = event["img"].get("url")
            is_s3_url = current_url and self.s3_bucket_url in current_url if current_url else False
            
            if is_s3_url:
                logger.debug(f"[{idx}] Skipping: S3 URL already set")
                self.stats['skipped'] += 1
                continue
            
            if current_url and not is_s3_url and not self.replace_external:
                logger.debug(f"[{idx}] Skipping: external URL exists (use --replace-external to override)")
                self.stats['skipped'] += 1
                continue

            self.stats['events_with_images'] += 1

            prompt = self.get_image_prompt(event)
            if not prompt:
                logger.warning(f"[{idx}] No prompt found")
                self.stats['errors'] += 1
                continue

            local_path = self.generate_image(prompt, idx)
            if not local_path:
                self.stats['errors'] += 1
                continue

            self.stats['images_generated'] += 1

            dia_id = event.get('dia_id')
            s3_url = self.upload_to_s3(local_path, idx, dia_id)

            if s3_url:
                event['img']['url'] = s3_url
                self.stats['images_uploaded'] += 1
                logger.info(f"[✓] Event {idx} updated with S3 URL")
            else:
                logger.error(f"[✗] Failed to upload event {idx}")
                self.stats['errors'] += 1

        if not self.dry_run:
            self.save_json(self.events_path, events_data)
        else:
            logger.info("[dry-run] Would save updated events.json")

    def print_summary(self) -> None:
        logger.info("\n" + "="*70)
        logger.info("IMAGE REGENERATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total events:         {self.stats['total_events']}")
        logger.info(f"Events with images:   {self.stats['events_with_images']}")
        logger.info(f"Images generated:     {self.stats['images_generated']}")
        logger.info(f"Images uploaded:      {self.stats['images_uploaded']}")
        logger.info(f"Events skipped:       {self.stats['skipped']}")
        logger.info(f"Errors:               {self.stats['errors']}")
        logger.info("="*70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--out-dir', required=True, type=str, help='Path to data/metadata directory containing events.json')
    parser.add_argument('--backend', default='imagen', type=str, help='Image generation backend (imagen, openai, etc.)')
    parser.add_argument('--bucket', required=True, type=str, help='AWS S3 bucket name for image uploads')
    parser.add_argument('--start-event', default=0, type=int, help='Start processing from this event index (0-based)')
    parser.add_argument('--num-events', default=None, type=int, help='Limit processing to this many events')
    parser.add_argument('--dry-run', action='store_true', help='Simulate run without uploading to S3 or modifying files')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--replace-external', action='store_true', help='Replace external (non-S3) URLs with S3 URLs generated from prompts')

    return parser.parse_args()


def main():
    args = parse_args()
    logger.setLevel(args.log_level)

    logger.info("="*70)
    logger.info("VisMem-C: IMAGE REGENERATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Output dir: {args.out_dir}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"S3 Bucket: {args.bucket}")
    if args.dry_run:
        logger.warning("[!] DRY RUN MODE - No changes will be saved")
    if args.replace_external:
        logger.info("[*] Mode: Replace external URLs with S3 URLs")
    logger.info("="*70 + "\n")

    try:
        pipeline = ImageRegenerationPipeline(
            out_dir=args.out_dir,
            backend_name=args.backend,
            bucket=args.bucket,
            dry_run=args.dry_run,
            replace_external=args.replace_external
        )

        pipeline.process_events(
            start_event=args.start_event,
            num_events=args.num_events
        )

        pipeline.print_summary()
        logger.info("[✓] Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"[✗] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
