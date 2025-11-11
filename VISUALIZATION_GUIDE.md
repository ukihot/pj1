# 可視化機能ガイド

## インストール

可視化機能を使用するには、Supervisionライブラリをインストールしてください：

```bash
pip install supervision
```

## 使い方

### 基本的な使用方法

`--visualize` フラグを追加するだけで、リアルタイム可視化が有効になります：

```bash
python src/hippolytica.py video.mp4 --visualize
```

### コート四隅を表示する場合

`--court-corners` オプションと組み合わせて使用すると、コートの枠線も表示されます：

```bash
python src/hippolytica.py video.mp4 --visualize --court-corners '{"top_left": [100, 100], "top_right": [500, 100], "bottom_right": [500, 400], "bottom_left": [100, 400]}'
```

### バッチ処理を無効化（推奨）

可視化モードでは、バッチ処理を無効化することをおすすめします：

```bash
python src/hippolytica.py video.mp4 --visualize --no-batch
```

## 表示内容

可視化ウィンドウには以下の情報が表示されます：

### 1. バウンディングボックス + ID
- 検出された各プレイヤーに緑色の枠が表示されます
- 枠の上に「ID: X」というラベルが表示されます

### 2. コート四隅
- `--court-corners` を指定した場合、青色の枠線でコートが表示されます
- 各コーナーに「TL」「TR」「BR」「BL」のラベルが表示されます

### 3. ステータス情報（画面上部）
- **Status**: インプレー（緑）/ アウトオブプレー（赤）
- **Raid ID**: 現在のレイドID
- **Frame**: フレーム番号と動画時間（MM:SS形式）
- **Tracks**: 検出されているトラック数

## 操作方法

- **'q' キー**: 可視化を終了して処理を停止
- **'ESC' キー**: 可視化を終了して処理を停止

## 注意事項

1. **パフォーマンス**: 可視化モードは処理速度が遅くなります
2. **バッチ処理**: バッチ処理モードでは可視化が制限されます（`--no-batch` 推奨）
3. **Supervisionが必要**: `pip install supervision` を実行してください

## トラブルシューティング

### Supervisionがインストールされていない場合

```
WARNING - Supervision not available. Install with: pip install supervision
```

→ `pip install supervision` を実行してください

### ウィンドウが表示されない場合

- GUIが利用できない環境（SSHなど）では可視化モードは使用できません
- X11フォワーディングを有効にするか、ローカル環境で実行してください

## 例

### 例1: シンプルな可視化

```bash
python src/hippolytica.py sample.mp4 --visualize --no-batch
```

### 例2: コート表示付き

```bash
python src/hippolytica.py sample.mp4 --visualize --no-batch \
  --court-corners '{"top_left": [200, 150], "top_right": [1000, 150], "bottom_right": [1000, 600], "bottom_left": [200, 600]}'
```

### 例3: フレーム間引きを減らして滑らかに

```bash
python src/hippolytica.py sample.mp4 --visualize --no-batch --skip-frames 1
```
