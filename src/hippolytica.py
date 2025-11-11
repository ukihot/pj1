#!/usr/bin/env python3
"""
Supervision を使った動画解析のハローワールド
"""
import sys
import os
import logging
import argparse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from video_downloader import VideoDownloader

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
        
        # YOLOv8モデルの初期化（人間検出用）
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")  # nanoモデルで高速化
        
        # Supervisionのアノテーター初期化
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
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

        try:
            logger.info("Starting video analysis: %s", video_path)
            self._process_video(video_path, limit_seconds=args.limit_seconds)
            logger.info("Analysis complete")
        except Exception as e:
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
            sys.exit(1)

    def _detect_white_lines(self, frame: np.ndarray) -> list:
        """白線コートの検出（Canny + Hough変換）"""
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny エッジ検出
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Hough変換で直線検出
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # 白線らしい線をフィルタリング（水平・垂直に近い線）
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 線の角度を計算
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 水平（0度, 180度）または垂直（90度）に近い線のみ
            if angle < 15 or angle > 165 or (85 < angle < 95):
                filtered_lines.append(line[0])
        
        return filtered_lines

    def _process_video(self, video_path: str, limit_seconds: float = None):
        """Supervisionを使った動画処理"""
        # 動画を開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # 動画情報を取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 処理フレーム数を制限
        if limit_seconds is not None:
            max_frames = int(fps * limit_seconds)
            total_frames = min(total_frames, max_frames)
            logger.info(
                "Limiting processing to %.1f seconds (%d frames)",
                limit_seconds,
                total_frames,
            )

        logger.info(
            "Video info: %dx%d, %.2f fps, %d frames",
            width,
            height,
            fps,
            total_frames,
        )

        # 出力動画の設定
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        output_video_path = os.path.join(self.output_dir, f"{base_name}_annotated.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        logger.info("Saving annotated video to: %s", output_video_path)

        frame_count = 0
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 100フレームごとにログ出力
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(
                        "Processing frame %d/%d (%.1f%%)",
                        frame_count,
                        total_frames,
                        progress,
                    )

                annotated_frame = frame.copy()

                # 1. 白線コートの検出と描画
                lines = self._detect_white_lines(frame)
                for line in lines:
                    x1, y1, x2, y2 = line
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # 2. YOLOv8で人間検出
                results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
                
                # Supervisionの検出形式に変換
                detections = sv.Detections.from_ultralytics(results[0])
                
                # ラベル作成
                labels = [
                    f"Person {i+1} {confidence:.2f}"
                    for i, confidence in enumerate(detections.confidence)
                ]
                
                # Boxアノテーション描画
                annotated_frame = self.box_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )

                # フレーム情報をオーバーレイ
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}/{total_frames} | Lines: {len(lines)} | People: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # 動画ファイルに書き込み
                out.write(annotated_frame)

        finally:
            cap.release()
            out.release()

        logger.info("Processed %d frames", frame_count)
        logger.info("Annotated video saved to: %s", output_video_path)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Supervision動画解析")
    parser.add_argument("input", help="YouTube URLまたはローカル動画パス")
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=None,
        help="処理する秒数を制限（例: 30で最初の30秒のみ処理）",
    )

    args = parser.parse_args()

    app = Application()
    app.run(args)


if __name__ == "__main__":
    main()
