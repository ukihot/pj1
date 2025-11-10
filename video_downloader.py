"""
動画ダウンローダー
"""

import os
import re
import logging
import subprocess

logger = logging.getLogger(__name__)


class VideoDownloader:
    """YouTube動画のダウンロード管理"""

    def __init__(self, output_dir: str = "videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download(self, url: str) -> str:
        """
        YouTube URLから動画をダウンロード

        Args:
            url: YouTube URL

        Returns:
            ダウンロードされた動画のパス
        """
        self._validate_url(url)

        logger.info("Fetching video from: %s", url)

        output_template = os.path.join(self.output_dir, "%(title)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f",
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            "--no-playlist",
            "--quiet",
            "--progress",
            url,
        ]

        try:
            logger.info("Downloading with yt-dlp...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            downloaded_files = [
                f for f in os.listdir(self.output_dir) if f.endswith(".mp4")
            ]

            if not downloaded_files:
                raise FileNotFoundError("Downloaded file not found")

            downloaded_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)),
                reverse=True,
            )
            output_path = os.path.join(self.output_dir, downloaded_files[0])

            logger.info("Download complete: %s", output_path)
            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"yt-dlp error: {error_msg}") from e
        except FileNotFoundError as e:
            raise RuntimeError(f"Could not download video: {e}") from e

    @staticmethod
    def _validate_url(url: str):
        """URL検証"""
        if not url:
            raise ValueError("URL cannot be empty")

        if not any(
            domain in url.lower() for domain in ["youtube.com", "youtu.be", "youtube"]
        ):
            raise ValueError("Invalid YouTube URL format")

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """ファイル名のサニタイズ"""
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        if len(filename) > 200:
            filename = filename[:200]
        return filename
