# インプレー判定システム

## 概要
このドキュメントでは、動画解析における「インプレー中」と「インプレー外」の判定ロジックについて説明します。研究の質を高めるため、戦術的な「動かない」選択と、広告・トーナメント表などの非プレー映像を正確に区別します。

## 問題点

### 従来の問題
1. **動き検出の誤判定**: 選手が戦術的に静止している場面を「不要フレーム」として除外
2. **シーン切り替えの未検出**: 広告、リプレイ、客席映像への切り替えを検出できない
3. **過度な線形補間**: 2秒以上のギャップでも補間し、測定エラーを生む

### 研究への影響
- **時空間分析の精度低下**: 選手の「動かない」という戦術的選択が重要なデータとして失われる
- **データの信頼性低下**: インプレー外の映像を誤って解析し、ノイズが混入
- **補間エラー**: シーン切り替え時の補間により、物理的にありえない移動が記録される

## 改善された判定システム

### 1. インプレー判定（`detect_in_play_frame`）

#### 判定基準
1. **人物数チェック**
   - 最小人数: 2人（選手が少なくとも映っている）
   - 最大人数: 20人（客席・広告を除外）

2. **色分布分析**
   - **彩度の標準偏差**: 広告は彩度が高く均一（`s_std < 30`）
   - **明度の平均**: トーナメント表は白背景で明るい（`v_mean > 200`）
   - **明度の標準偏差**: 均一な背景を検出（`v_std < 40`）

#### 除外される映像
- 広告画面（彩度が高く均一）
- トーナメント表（白背景で均一）
- 客席映像（人物数が多すぎる）
- 解説席（人物数が少なすぎる）

#### コード例
```python
def detect_in_play_frame(frame, num_persons=0, min_persons=2, max_persons=20):
    # 人物数チェック
    if num_persons < min_persons or num_persons > max_persons:
        return False
    
    # HSV色空間で分析
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s_std = np.std(s)  # 彩度の標準偏差
    v_mean = np.mean(v)  # 明度の平均
    v_std = np.std(v)  # 明度の標準偏差
    
    # 広告・トーナメント表の特徴
    if s_std < 30 and (v_mean > 200 or v_mean < 50):
        return False
    
    if v_std < 40 and v_mean > 180:
        return False
    
    return True
```

### 2. シーン切り替え検出（`detect_scene_change`）

#### 判定基準
1. **ヒストグラム相関**
   - 前フレームとのヒストグラム相関を計算
   - 相関 < 0.7 の場合、シーン切り替えと判定

2. **フレーム差分**
   - 急激な画面変化を検出
   - 平均差分 > 30.0 の場合、シーン切り替えと判定

#### 検出される切り替え
- カメラアングル変更
- リプレイ映像への切り替え
- 広告挿入
- 客席・解説席への切り替え

#### コード例
```python
def detect_scene_change(frame, prev_frame, change_thresh=30.0):
    # ヒストグラム相関
    hist_curr = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_prev = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    correlation = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
    
    if correlation < 0.7:
        return True
    
    # フレーム差分
    diff = cv2.absdiff(prev_frame, frame)
    if np.mean(diff) > change_thresh:
        return True
    
    return False
```

### 3. ギャップ制限付き線形補間（`interpolate_tracks`）

#### 改善点
- **最大ギャップ制限**: デフォルト2秒（60フレーム@30fps）
- **ギャップ超過時の処理**: 補間せず、別のトラッキングセグメントとして扱う
- **測定エラー防止**: シーン切り替え時の不自然な補間を回避

#### パラメータ
```python
def interpolate_tracks(results, skip_frames=3, fps=30.0, max_gap_seconds=2.0):
    max_gap_frames = int(fps * max_gap_seconds)  # 60フレーム
    
    for player_id in unique_players:
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            
            # ギャップが大きすぎる場合は補間しない
            if gap > max_gap_frames:
                logger.debug(f"Skipping interpolation: gap={gap} frames")
                continue
            
            # 線形補間
            # ...
```

## 処理フロー

### 通常モード（`process_match_video`）
```
1. フレーム読み込み
2. シーン切り替え検出 → 検出時はスキップ
3. フレーム間引き判定
4. 選手検出（YOLO）
5. インプレー判定 → インプレー外はスキップ
6. トラッキング（DeepSORT）
7. 座標記録
```

### バッチモード（`process_match_video_batch`）
```
1. フレーム読み込み
2. シーン切り替え検出 → 検出時はスキップ
3. フレーム間引き判定
4. エッジ密度による簡易判定 → 広告・トーナメント表を事前除外
5. バッファに蓄積
6. バッチ推論（YOLO）
7. インプレー判定（各フレーム）
8. トラッキング（DeepSORT）
9. 座標記録
```

## 統計情報の出力

処理完了時に以下の統計情報が出力されます:

```
Frame statistics:
  Total frames: 10000
  In-play frames: 6500
  Out-of-play frames: 2800
  Scene changes detected: 45
  Detection frames: 2167
```

### 統計の意味
- **Total frames**: 動画の総フレーム数
- **In-play frames**: インプレー中と判定されたフレーム数
- **Out-of-play frames**: 広告・トーナメント表などと判定されたフレーム数
- **Scene changes detected**: シーン切り替えが検出された回数
- **Detection frames**: 実際にYOLO検出を実行したフレーム数

## 研究への貢献

### 1. 戦術的「静止」の保存
- 選手が意図的に動かない場面を正確に記録
- 防御時の待機姿勢、攻撃前の静止などの戦術的選択を分析可能

### 2. データの信頼性向上
- 広告・トーナメント表などのノイズを除外
- インプレー中のデータのみを解析対象とする

### 3. 測定エラーの削減
- シーン切り替え時の不自然な補間を防止
- 2秒以上のギャップは別セグメントとして扱う

### 4. 時空間分析の精度向上
- 連続したインプレー映像のみを解析
- ラウンド間の連鎖効果を正確に評価可能

## パラメータ調整

### インプレー判定の調整
```python
# より厳格な判定（研究用）
detect_in_play_frame(frame, num_persons, min_persons=4, max_persons=15)

# より緩い判定（一次解析用）
detect_in_play_frame(frame, num_persons, min_persons=1, max_persons=25)
```

### シーン切り替え感度の調整
```python
# より敏感（頻繁に検出）
detect_scene_change(frame, prev_frame, change_thresh=20.0)

# より鈍感（明確な切り替えのみ）
detect_scene_change(frame, prev_frame, change_thresh=40.0)
```

### 補間ギャップの調整
```python
# より短いギャップ（厳格）
interpolate_tracks(results, skip_frames, fps, max_gap_seconds=1.0)

# より長いギャップ（緩い）
interpolate_tracks(results, skip_frames, fps, max_gap_seconds=3.0)
```

## トラブルシューティング

### 問題: インプレー中なのに除外される
**原因**: 人物数が閾値外、または色分布が特殊
**解決策**: `min_persons`を減らす、または色分布閾値を調整

### 問題: 広告が除外されない
**原因**: 広告の色分布が通常映像に近い
**解決策**: `s_std`や`v_std`の閾値を調整

### 問題: シーン切り替えが検出されすぎる
**原因**: `change_thresh`が低すぎる
**解決策**: 閾値を30.0から40.0に上げる

### 問題: 補間ギャップが多すぎる
**原因**: `max_gap_seconds`が短すぎる
**解決策**: 2.0秒から3.0秒に延長

## まとめ

このシステムにより、以下が実現されます:

1. **戦術的「動かない」の保存**: 選手の静止を戦術的選択として正確に記録
2. **インプレー外の除外**: 広告・トーナメント表などのノイズを自動除外
3. **測定エラーの防止**: シーン切り替え時の不自然な補間を回避
4. **研究の質向上**: 信頼性の高いデータで時空間分析を実施

これにより、カバディ競技における勝利要因の定量的解析の精度が大幅に向上します。
