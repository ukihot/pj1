"""
Kabaddi 動画解析
"""

import logging
import os
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """処理設定"""

    skip_frames: int = 3
    batch_size: int = 4
    use_async: bool = True
    use_interpolation: bool = True
    conf_threshold: float = 0.4
    target_size: Tuple[int, int] = (640, 384)  # YOLOv8 stride=32の倍数に変更
    max_gap_seconds: float = 2.0
    min_persons: int = 2
    max_persons: int = 20


@dataclass
class VideoInfo:
    """動画情報"""

    total_frames: int
    fps: float
    width: int
    height: int


@dataclass
class ProcessingStats:
    """処理統計"""

    total_frames: int = 0
    in_play_frames: int = 0
    out_of_play_frames: int = 0
    scene_changes: int = 0
    processed_frames: int = 0
    unique_players: int = 0
    total_positions: int = 0


class ModelManager:
    """YOLOとDeepSORTモデルの管理"""

    def __init__(self, use_half: bool = True):
        self.use_half = use_half
        self.yolo_model: Optional[YOLO] = None
        self.tracker: Optional[DeepSort] = None
        self._initialized = False

    def initialize(self):
        """モデルの初期化"""
        if self._initialized:
            return

        logger.info("Loading YOLOv8 model...")
        self.yolo_model = YOLO("yolov8n.pt")

        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            self.yolo_model.to("cuda")
            # Note: half precision is handled by ultralytics internally via 'half' parameter
        else:
            logger.warning("CUDA not available, using CPU (slower)")

        logger.info("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=50,
            n_init=2,
            nms_max_overlap=0.7,
            max_cosine_distance=0.4,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            embedder_gpu=torch.cuda.is_available(),
        )

        self._initialized = True
        logger.info("Models initialized successfully")

    def detect_single(
        self, frame: np.ndarray, config: ProcessingConfig
    ) -> List[List[float]]:
        """単一フレームの検出"""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if w > config.target_size[0] or h > config.target_size[1]:
            frame_resized = cv2.resize(
                frame, config.target_size, interpolation=cv2.INTER_LINEAR
            )
            scale_x = w / config.target_size[0]
            scale_y = h / config.target_size[1]
        else:
            frame_resized = frame
            scale_x = scale_y = 1.0

        results = self.yolo_model(
            frame_resized,
            verbose=False,
            imgsz=config.target_size[1],
            half=self.use_half and torch.cuda.is_available(),
        )
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if int(box.cls[0]) == 0:  # 人物
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    if conf > config.conf_threshold:
                        detections.append(
                            [
                                x1 * scale_x,
                                y1 * scale_y,
                                x2 * scale_x,
                                y2 * scale_y,
                                conf,
                            ]
                        )

        return detections

    def detect_batch(
        self, frames: List[np.ndarray], config: ProcessingConfig
    ) -> List[List[List[float]]]:
        """バッチ検出"""
        if not frames:
            return []

        resized_frames = []
        scales = []

        for frame in frames:
            if frame is None or frame.size == 0:
                resized_frames.append(None)
                scales.append((1.0, 1.0))
                continue

            h, w = frame.shape[:2]
            if w > config.target_size[0] or h > config.target_size[1]:
                frame_resized = cv2.resize(
                    frame, config.target_size, interpolation=cv2.INTER_LINEAR
                )
                scale_x = w / config.target_size[0]
                scale_y = h / config.target_size[1]
            else:
                frame_resized = frame
                scale_x = scale_y = 1.0

            resized_frames.append(frame_resized)
            scales.append((scale_x, scale_y))

        valid_frames = [f for f in resized_frames if f is not None]
        if not valid_frames:
            return [[] for _ in frames]

        results = self.yolo_model(
            valid_frames,
            verbose=False,
            imgsz=config.target_size[1],
            half=self.use_half and torch.cuda.is_available(),
        )
        all_detections = []
        result_idx = 0

        for i, frame in enumerate(resized_frames):
            if frame is None:
                all_detections.append([])
                continue

            result = results[result_idx]
            result_idx += 1
            detections = []

            if result.boxes is not None:
                scale_x, scale_y = scales[i]

                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # 人物
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])

                        if conf > config.conf_threshold:
                            detections.append(
                                [
                                    x1 * scale_x,
                                    y1 * scale_y,
                                    x2 * scale_x,
                                    y2 * scale_y,
                                    conf,
                                ]
                            )

            all_detections.append(detections)

        return all_detections


class FrameAnalyzer:
    """フレーム分析（インプレー判定、シーン切り替え検出）"""

    @staticmethod
    def is_in_play(
        frame: np.ndarray, num_persons: int, min_persons: int = 2, max_persons: int = 20
    ) -> bool:
        """インプレー判定"""
        if frame is None or num_persons < min_persons or num_persons > max_persons:
            return False

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            s_std = np.std(s)
            v_mean = np.mean(v)
            v_std = np.std(v)

            if s_std < 30 and (v_mean > 200 or v_mean < 50):
                return False

            if v_std < 40 and v_mean > 180:
                return False

            return True
        except cv2.error:
            return True

    @staticmethod
    def detect_scene_change(
        frame: np.ndarray, prev_frame: np.ndarray, threshold: float = 30.0
    ) -> bool:
        """シーン切り替え検出"""
        if frame is None or prev_frame is None:
            return False

        try:
            hist_curr = cv2.calcHist(
                [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            hist_prev = cv2.calcHist(
                [prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )

            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()

            correlation = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)

            if correlation < 0.7:
                return True

            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_prev, gray_curr)
            mean_diff = np.mean(diff)

            return mean_diff > threshold
        except cv2.error:
            return False


class AsyncFrameReader(Thread):
    """非同期フレーム読み込み"""

    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 128):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        """フレーム読み込みスレッドのメインループ"""
        try:
            while not self.stopped:
                if not self.queue.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stopped = True
                        # 終了シグナルをキューに追加
                        self.queue.put((False, None))
                        break
                    self.queue.put((ret, frame))
        except Exception as e:
            logger.error("AsyncFrameReader error: %s", e)
            self.stopped = True
            self.queue.put((False, None))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """キューからフレームを取得"""
        if self.stopped and self.queue.empty():
            return False, None
        return self.queue.get(timeout=1.0)

    def stop(self):
        """スレッドの停止"""
        self.stopped = True
        # キューをクリアして即座に終了できるようにする
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

    def more(self) -> bool:
        """まだフレームが残っているか確認"""
        return not self.stopped or not self.queue.empty()


class TrackInterpolator:
    """トラッキング結果の補間"""

    @staticmethod
    def interpolate(
        results: List[List],
        skip_frames: int,
        fps: float,
        max_gap_seconds: float = 2.0,  # pylint: disable=unused-argument
    ) -> List[List]:
        """線形補間"""
        if not results:
            return results

        max_gap_frames = int(fps * max_gap_seconds)
        df = pd.DataFrame(results, columns=["frame", "player_id", "x_coord", "y_coord"])
        interpolated = []
        skipped_gaps = 0

        for player_id in df["player_id"].unique():
            player_data = df[df["player_id"] == player_id].sort_values("frame")

            if len(player_data) < 2:
                interpolated.extend(player_data.values.tolist())
                continue

            frames = player_data["frame"].values
            x_coords = player_data["x_coord"].values
            y_coords = player_data["y_coord"].values

            for i in range(len(frames) - 1):
                start_frame = frames[i]
                end_frame = frames[i + 1]
                gap = end_frame - start_frame

                interpolated.append([start_frame, player_id, x_coords[i], y_coords[i]])

                if gap > max_gap_frames:
                    skipped_gaps += 1
                    continue

                if gap > 1:
                    for f in range(start_frame + 1, end_frame):
                        alpha = (f - start_frame) / gap
                        x_interp = x_coords[i] + alpha * (x_coords[i + 1] - x_coords[i])
                        y_interp = y_coords[i] + alpha * (y_coords[i + 1] - y_coords[i])
                        interpolated.append([f, player_id, x_interp, y_interp])

            interpolated.append([frames[-1], player_id, x_coords[-1], y_coords[-1]])

        if skipped_gaps > 0:
            logger.info("Skipped %d large gaps (>%ss)", skipped_gaps, max_gap_seconds)

        return interpolated


class VideoProcessor:
    """動画処理のメインクラス"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.frame_analyzer = FrameAnalyzer()
        self.interpolator = TrackInterpolator()
        self.stats = ProcessingStats()

    def process(self, input_path: str, output_path: str):
        """動画処理のメインメソッド"""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        self.model_manager.initialize()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        video_info = self._get_video_info(cap)
        logger.info(
            "Video: %d frames, %.2f FPS", video_info.total_frames, video_info.fps
        )

        try:
            if self.config.batch_size > 1:
                results = self._process_batch(cap, video_info)
            else:
                results = self._process_sequential(cap, video_info)

            self._save_results(results, output_path, video_info.fps)
        finally:
            cap.release()

    def _get_video_info(self, cap: cv2.VideoCapture) -> VideoInfo:
        """動画情報取得"""
        return VideoInfo(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _process_sequential(
        self, cap: cv2.VideoCapture, video_info: VideoInfo
    ) -> List[List]:
        """逐次処理"""
        frame_reader = AsyncFrameReader(cap) if self.config.use_async else None
        if frame_reader:
            frame_reader.start()

        results = []
        prev_frame = None
        frame_idx = 0

        try:
            while True:
                try:
                    ret, frame = frame_reader.read() if frame_reader else cap.read()
                except Exception as e:
                    logger.warning("Frame read error at frame %d: %s", frame_idx, e)
                    break

                if not ret or frame is None:
                    break

                if frame_idx % 100 == 0:
                    progress = 100 * frame_idx / video_info.total_frames
                    logger.info(
                        "Processing: %d/%d (%.1f%%)",
                        frame_idx,
                        video_info.total_frames,
                        progress,
                    )

                if prev_frame is not None and self.frame_analyzer.detect_scene_change(
                    frame, prev_frame
                ):
                    self.stats.scene_changes += 1
                    prev_frame = frame
                    frame_idx += 1
                    continue

                prev_frame = frame

                if frame_idx % self.config.skip_frames != 0:
                    frame_idx += 1
                    continue

                detections = self.model_manager.detect_single(frame, self.config)

                if not self.frame_analyzer.is_in_play(
                    frame,
                    len(detections),
                    self.config.min_persons,
                    self.config.max_persons,
                ):
                    self.stats.out_of_play_frames += 1
                    frame_idx += 1
                    continue

                self.stats.in_play_frames += 1

                if detections:
                    tracks = self._update_tracker(detections, frame)
                    results.extend(self._extract_positions(tracks, frame_idx))

                frame_idx += 1
                self.stats.processed_frames += 1

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            if frame_reader:
                frame_reader.stop()
                # スレッドの終了を待つ（最大2秒）
                frame_reader.join(timeout=2.0)
                if frame_reader.is_alive():
                    logger.warning("AsyncFrameReader thread did not terminate cleanly")

        self.stats.total_frames = frame_idx
        return results

    def _process_batch(
        self, cap: cv2.VideoCapture, video_info: VideoInfo
    ) -> List[List]:
        """バッチ処理"""
        results = []
        prev_frame = None
        frame_idx = 0
        frame_buffer = []
        frame_indices = []

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_buffer:
                    results.extend(
                        self._process_frame_batch(frame_buffer, frame_indices)
                    )
                break

            if frame_idx % 100 == 0:
                progress = 100 * frame_idx / video_info.total_frames
                logger.info(
                    "Processing: %d/%d (%.1f%%)",
                    frame_idx,
                    video_info.total_frames,
                    progress,
                )

            if prev_frame is not None and self.frame_analyzer.detect_scene_change(
                frame, prev_frame
            ):
                self.stats.scene_changes += 1
                prev_frame = frame
                frame_idx += 1
                continue

            prev_frame = frame

            if frame_idx % self.config.skip_frames != 0:
                frame_idx += 1
                continue

            frame_buffer.append(frame.copy())
            frame_indices.append(frame_idx)

            if len(frame_buffer) >= self.config.batch_size:
                results.extend(self._process_frame_batch(frame_buffer, frame_indices))
                frame_buffer = []
                frame_indices = []

            frame_idx += 1

        self.stats.total_frames = frame_idx
        return results

    def _process_frame_batch(
        self, frames: List[np.ndarray], indices: List[int]
    ) -> List[List]:
        """フレームバッチの処理"""
        batch_detections = self.model_manager.detect_batch(frames, self.config)
        results = []

        for i, (detections, frame_idx) in enumerate(zip(batch_detections, indices)):
            if not self.frame_analyzer.is_in_play(
                frames[i],
                len(detections),
                self.config.min_persons,
                self.config.max_persons,
            ):
                self.stats.out_of_play_frames += 1
                continue

            self.stats.in_play_frames += 1

            if detections:
                tracks = self._update_tracker(detections, frames[i])
                results.extend(self._extract_positions(tracks, frame_idx))

            self.stats.processed_frames += 1

        return results

    def _update_tracker(self, detections: List[List[float]], frame: np.ndarray):
        """トラッカー更新"""
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1

            if w > 0 and (y2 - y1) > 0:
                deepsort_detections.append(([x1, y1, w, y2 - y1], conf, "person"))

        return self.model_manager.tracker.update_tracks(
            deepsort_detections, frame=frame
        )

    def _extract_positions(self, tracks, frame_idx: int) -> List[List]:
        """トラックから位置情報を抽出"""
        positions = []
        for track in tracks:
            if track.is_confirmed():
                ltrb = track.to_ltrb()
                bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                x, y = self._project_to_court(bbox)
                positions.append([frame_idx, track.track_id, x, y])
        return positions

    @staticmethod
    def _project_to_court(bbox: List[float], homography=None) -> Tuple[float, float]:
        """
        コート座標への射影

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            homography: Optional homography matrix for perspective transform

        Returns:
            Tuple of (x, y) coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if homography is None:
            return cx, cy

        pt = np.array([[[cx, cy]]], dtype="float32")
        dst = cv2.perspectiveTransform(pt, homography)
        return dst[0][0][0], dst[0][0][1]

    def _save_results(self, results: List[List], output_path: str, fps: float):
        """
        結果の保存

        Args:
            results: Tracking results list
            output_path: Output CSV file path
            fps: Video frames per second
        """
        if self.config.use_interpolation and results:
            logger.info("Applying interpolation...")
            results = self.interpolator.interpolate(
                results, self.config.skip_frames, fps, self.config.max_gap_seconds
            )

        if results:
            self.stats.unique_players = len(set(r[1] for r in results))
            self.stats.total_positions = len(results)
            df = pd.DataFrame(
                results, columns=["frame", "player_id", "x_coord", "y_coord"]
            )
        else:
            logger.warning("No tracking results generated")
            df = pd.DataFrame(columns=["frame", "player_id", "x_coord", "y_coord"])

        df.to_csv(output_path, index=False)
        self._log_stats()
        logger.info("Results saved to %s", output_path)

    def _log_stats(self):
        """統計情報のログ出力"""
        logger.info("Statistics:")
        logger.info("  Total frames: %d", self.stats.total_frames)
        logger.info("  In-play: %d", self.stats.in_play_frames)
        logger.info("  Out-of-play: %d", self.stats.out_of_play_frames)
        logger.info("  Scene changes: %d", self.stats.scene_changes)
        logger.info("  Processed: %d", self.stats.processed_frames)
        logger.info("  Unique players: %d", self.stats.unique_players)
        logger.info("  Total positions: %d", self.stats.total_positions)
