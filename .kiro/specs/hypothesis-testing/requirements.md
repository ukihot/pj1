# Requirements Document

## Introduction

本システムは、カバディ競技における5つの研究仮説を統計的に検証するための機能を提供する。既存のHippolyticaシステムが抽出した動作データ（速度、加速度、配置など）を用いて、レイダーの成功要因、防御フォーメーションの効果、ラウンド間連鎖、個人×チーム交互作用、リアルタイム指標の有用性を定量的に分析する。

## Glossary

- **System**: Hypothesis Testing Module（仮説検証モジュール）
- **Raider**: レイド（攻撃）を行う選手
- **Defense Density**: 防御選手間の平均距離（メートル単位）
- **Touch Success Rate**: タッチ成功率（成功タッチ数 / レイド試行数）
- **Raid Blocking Rate**: レイド阻止率（阻止成功数 / レイド試行数）
- **Acceleration**: 加速度（m/s²）、特に0-3m区間の初速加速
- **Formation Entropy**: フォーメーション配置のシャノンエントロピー（予測困難性の指標）
- **Round**: 1回のレイド試行単位
- **Interaction Effect**: 交互作用効果（個人スキル×チーム戦術の相乗効果）
- **Real-time Indicator**: リアルタイム指標（加速度、配置密度など観戦・育成用の即時表示可能な指標）
- **User**: 研究者、コーチ、解説者、育成担当者

## Requirements

### Requirement 1: レイダー加速度とタッチ成功率の相関分析

**User Story:** As a 研究者, I want レイダーの0-3m加速度とタッチ成功率の相関を統計的に検証する機能, so that 初速加速が成功要因であるかを科学的に証明できる

#### Acceptance Criteria

1. WHEN User が仮説1の分析を実行する, THE System SHALL 全レイダーの0-3m区間加速度データとタッチ成功率データを抽出する
2. THE System SHALL ピアソン相関係数とスピアマン順位相関係数を計算し、p値を算出する
3. THE System SHALL 効果量（Cohen's d）を計算し、実用的意義を評価する
4. THE System SHALL 散布図と回帰直線を含む可視化レポートを生成する
5. IF p < 0.05 かつ Cohen's d > 0.5, THEN THE System SHALL 「有意な正の相関あり」と結論を出力する

### Requirement 2: 防御密度とレイド阻止率の関係分析

**User Story:** As a コーチ, I want 防御選手間距離とレイド阻止率の関係を定量化する機能, so that 最適な防御配置を科学的根拠に基づいて設計できる

#### Acceptance Criteria

1. WHEN User が仮説2の分析を実行する, THE System SHALL 各ラウンドの防御密度（選手間平均距離）とレイド阻止率を計算する
2. THE System SHALL 防御密度を0.5m刻みの区間に分割し、各区間の阻止率を集計する
3. THE System SHALL ANOVA検定により区間間の阻止率差の有意性を検証する
4. THE System SHALL 最適密度範囲（阻止率が最大となる区間）を特定する
5. THE System SHALL 防御密度と阻止率のヒートマップを生成する

### Requirement 3: ラウンド間連鎖効果の検証

**User Story:** As a 研究者, I want 前ラウンドの成功/失敗が次ラウンドの成功率に与える影響を分析する機能, so that モーメンタム効果の存在を実証できる

#### Acceptance Criteria

1. WHEN User が仮説3の分析を実行する, THE System SHALL 各レイダーのラウンドn成功/失敗と、ラウンドn+1成功率を時系列データとして抽出する
2. THE System SHALL 条件付き確率 p(成功|前回成功) と p(成功|前回失敗) を計算する
3. THE System SHALL カイ二乗検定により両確率の差の有意性を検証する
4. THE System SHALL 転移エントロピーを計算し、因果的連鎖の強度を定量化する
5. THE System SHALL 連続成功回数と成功率の関係を折れ線グラフで可視化する

### Requirement 4: 個人スキル×チーム戦術の交互作用分析

**User Story:** As a 研究者, I want 個人スキル特徴量とチーム戦術特徴量の交互作用が勝敗に与える影響を評価する機能, so that 単独要因では説明できない相乗効果を発見できる

#### Acceptance Criteria

1. WHEN User が仮説4の分析を実行する, THE System SHALL 個人スキル特徴量（最高速度、加速度、方向転換頻度）とチーム戦術特徴量（配置エントロピー、防御密度）を抽出する
2. THE System SHALL ロジスティック回帰モデルを3種類構築する（個人のみ、チームのみ、個人×チーム交互作用項含む）
3. THE System SHALL 各モデルのAUC、精度、F1スコアを算出し比較する
4. THE System SHALL 相互情報量 I(個人;勝敗)、I(チーム;勝敗)、I(個人×チーム;勝敗) を計算する
5. IF 交互作用モデルのAUCが単独モデルより0.05以上高い, THEN THE System SHALL 「交互作用が勝敗説明力を有意に向上させる」と結論を出力する

### Requirement 5: リアルタイム指標の有用性評価

**User Story:** As a 解説者, I want リアルタイム指標（加速度、配置密度）が観戦理解や育成に有用であることを示す機能, so that 実用化の根拠を提示できる

#### Acceptance Criteria

1. WHEN User が仮説5の分析を実行する, THE System SHALL リアルタイム計算可能な指標（加速度、速度、防御密度、配置エントロピー）のリストを生成する
2. THE System SHALL 各指標と試合結果（タッチ成功/失敗、勝敗）の相互情報量を計算する
3. THE System SHALL 相互情報量が0.1以上の指標を「有用な指標」として抽出する
4. THE System SHALL 各指標の計算コスト（処理時間ms）を測定する
5. THE System SHALL 有用性（相互情報量）と計算コストの散布図を生成し、実用的な指標を推薦する

### Requirement 6: 統合レポート生成

**User Story:** As a User, I want 5つの仮説検証結果を統合したレポートを自動生成する機能, so that 研究成果を論文や発表資料として活用できる

#### Acceptance Criteria

1. WHEN User が統合レポート生成を実行する, THE System SHALL 5つの仮説それぞれの検証結果（統計量、p値、効果量、結論）をまとめる
2. THE System SHALL 各仮説の可視化図表（散布図、ヒートマップ、棒グラフ）をPNG形式で保存する
3. THE System SHALL Markdown形式のレポートファイルを生成する
4. THE System SHALL 生データをCSV形式でエクスポートする
5. THE System SHALL レポート生成完了時にファイルパスをログ出力する

### Requirement 7: データ品質検証

**User Story:** As a 研究者, I want 解析に使用するデータの品質を事前検証する機能, so that 信頼性の低いデータによる誤った結論を防げる

#### Acceptance Criteria

1. WHEN User が解析を開始する前, THE System SHALL 入力CSVファイルの必須カラム（frame_id, player_id, x, y, velocity, acceleration）の存在を確認する
2. THE System SHALL 欠損値の割合を計算し、20%を超える場合は警告を出力する
3. THE System SHALL 異常値（速度>15m/s、加速度>10m/s²）の件数を報告する
4. THE System SHALL サンプルサイズ（ラウンド数、選手数）が統計的検出力を満たすか評価する
5. IF サンプルサイズ < 30ラウンド, THEN THE System SHALL 「統計的検出力が不足している可能性」と警告する

### Requirement 8: 設定可能な解析パラメータ

**User Story:** As a 研究者, I want 有意水準や効果量閾値などの解析パラメータを設定できる機能, so that 研究目的に応じた柔軟な分析ができる

#### Acceptance Criteria

1. THE System SHALL 有意水準α（デフォルト0.05）を設定可能にする
2. THE System SHALL 効果量閾値（デフォルト Cohen's d = 0.5）を設定可能にする
3. THE System SHALL 相互情報量閾値（デフォルト0.1）を設定可能にする
4. THE System SHALL 防御密度の区間幅（デフォルト0.5m）を設定可能にする
5. WHEN User が設定ファイル（YAML形式）を提供する, THE System SHALL その設定を読み込んで解析に適用する
