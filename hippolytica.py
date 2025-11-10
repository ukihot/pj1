#!/usr/bin/env python3
"""
エントリーポイント
"""
import sys
import os
import logging
import argparse
from video_downloader import VideoDownloader
from video_processor import VideoProcessor, ProcessingConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Application:
    """アプリケーションのメインクラス"""

    def __init__(self):
        self.videos_dir = "videos"
        self.output_dir = "output"
        self._ensure_directories()

    def __repr__(self):
        """String representation of Application"""
        return (
            f"Application(videos_dir={self.videos_dir}, output_dir={self.output_dir})"
        )

    def _ensure_directories(self):
        """必要なディレクトリの作成"""
        try:
            os.makedirs(self.videos_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create directories: %s", e)
            sys.exit(1)

    def run(self, args):
        """アプリケーション実行"""
        video_path = self._get_video_path(args.input)

        if not os.path.isfile(video_path):
            logger.error("Video file not found: %s", video_path)
            sys.exit(1)

        output_path = self._get_output_path(video_path)
        config = self._create_config(args)

        try:
            logger.info("Starting video analysis: %s", video_path)
            logger.info("Config: %s", config)

            processor = VideoProcessor(config)
            processor.process(video_path, output_path)

            logger.info("Analysis complete. CSV saved to %s", output_path)
        except (RuntimeError, IOError, ValueError) as e:
            logger.error("Failed to process video: %s", e, exc_info=True)
            sys.exit(1)

    def _get_video_path(self, input_arg: str) -> str:
        """入力から動画パスを取得"""
        if os.path.isfile(input_arg):
            logger.info("Using local video file: %s", input_arg)
            return input_arg

        try:
            downloader = VideoDownloader(self.videos_dir)
            logger.info("Downloading video from: %s", input_arg)
            video_path = downloader.download(input_arg)
            logger.info("Downloaded video to %s", video_path)
            return video_path
        except ValueError as e:
            logger.error("Invalid URL: %s", e)
            sys.exit(1)
        except RuntimeError as e:
            logger.error("Failed to download video: %s", e)
            logger.info("Tip: Check if the URL is correct and accessible")
            sys.exit(1)

    def _get_output_path(self, video_path: str) -> str:
        """出力パスの生成"""
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        return os.path.join(self.output_dir, f"{base_name}.csv")

    @staticmethod
    def _create_config(args) -> ProcessingConfig:
        """設定オブジェクトの作成"""
        return ProcessingConfig(
            skip_frames=args.skip_frames,
            batch_size=1 if args.no_batch else args.batch_size,
            use_async=not args.no_async,
            use_interpolation=not args.no_interpolation,
        )


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Kabaddi 動画解析システム")
    parser.add_argument("input", help="YouTube URLまたはローカル動画パス")
    parser.add_argument(
        "--skip-frames", type=int, default=3, help="フレーム間引き数（デフォルト: 3）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="バッチサイズ（デフォルト: 4）"
    )
    parser.add_argument(
        "--no-interpolation", action="store_true", help="線形補間を無効化"
    )
    parser.add_argument("--no-batch", action="store_true", help="バッチ推論を無効化")
    parser.add_argument(
        "--no-async", action="store_true", help="非同期読み込みを無効化"
    )

    args = parser.parse_args()

    app = Application()
    app.run(args)


if __name__ == "__main__":
    main()
