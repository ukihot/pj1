# Design Document

## Overview

Hypothesis Testing Moduleは、既存のHippolyticaシステムが生成したCSVデータ（選手の位置・速度・加速度）を入力として、5つの研究仮説を統計的に検証する独立したPythonモジュールである。本モジュールは、データ前処理、特徴量計算、統計解析、可視化、レポート生成の5つのコンポーネントで構成される。

設計の核心原則：
1. **モジュール独立性**: 既存のvideo_processor.pyに依存せず、CSVファイルのみを入力とする
2. **拡張性**: 新しい仮説や統計手法を容易に追加できる設計
3. **再現性**: 全ての解析パラメータを設定ファイルで管理し、同一条件での再実行を保証
4. **可視化重視**: 統計量だけでなく、直感的な図表を自動生成

## Architecture

### システム構成図

```
Input: tracking_data.csv (from video_processor.py)
         ↓
┌─────────────────────────────────────────┐
│   HypothesisTestingPipeline (main)      │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   DataPreprocessor                      │
│   - 欠損値処理                           │
│   - 異常値除外                           │
│   - データ品質検証                       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   FeatureExtractor                      │
│   - 速度・加速度計算                     │
│   - 防御密度計算                         │
│   - フォーメーションエントロピー計算     │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   HypothesisTester (5 modules)          │
│   - H1: AccelerationCorrelationTest     │
│   - H2: DefenseDensityAnalysis          │
│   - H3: RoundChainAnalysis              │
│   - H4: InteractionEffectTest           │
│   - H5: RealtimeIndicatorEvaluation     │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   Visualizer                            │
│   - 散布図、ヒートマップ生成             │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   ReportGenerator                       │
│   - Markdownレポート生成                 │
│   - CSV結果エクスポート                  │
└─────────────────────────────────────────┘
         ↓
Output: hypothesis_report.md, figures/*.png, results.csv
```


## Components and Interfaces

### 1. DataPreprocessor

**責務**: 入力CSVデータの品質検証と前処理

**主要メソッド**:
```python
class DataPreprocessor:
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """CSVファイルを読み込み、必須カラムの存在を確認"""
        
    def validate_quality(self, df: pd.DataFrame) -> QualityReport:
        """欠損値、異常値、サンプルサイズを検証"""
        
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR法とIsolation Forestで異常値を除外"""
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を線形補間または除外"""
```

**入力**: tracking_data.csv (columns: frame, player_id, x_coord, y_coord)
**出力**: 前処理済みDataFrame + QualityReport

**データ品質基準**:
- 欠損値率 < 20%
- 速度異常値: > 15 m/s
- 加速度異常値: > 10 m/s²
- 最小サンプルサイズ: 30ラウンド

### 2. FeatureExtractor

**責務**: 物理量と戦術指標の計算

**主要メソッド**:
```python
class FeatureExtractor:
    def compute_velocity(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """速度を計算 (m/s)"""
        
    def compute_acceleration(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """加速度を計算 (m/s²)、特に0-3m区間"""
        
    def compute_defense_density(self, df: pd.DataFrame, team_id: str) -> pd.Series:
        """防御選手間の平均距離を計算"""
        
    def compute_formation_entropy(self, df: pd.DataFrame) -> pd.Series:
        """配置のシャノンエントロピーを計算"""
        
    def extract_round_data(self, df: pd.DataFrame) -> List[RoundData]:
        """ラウンド単位でデータを分割"""
```

**特徴量定義**:
- **速度**: v = √((x_{t+1} - x_t)² + (y_{t+1} - y_t)²) / Δt
- **加速度**: a = (v_{t+1} - v_t) / Δt
- **0-3m加速度**: 移動開始から3m到達までの平均加速度
- **防御密度**: Σ dist(player_i, player_j) / n(n-1)/2
- **配置エントロピー**: H = -Σ p_i log(p_i)、p_i = area_i / total_area

### 3. HypothesisTester (5つのサブモジュール)

#### 3.1 AccelerationCorrelationTest (仮説1)

**責務**: レイダー加速度とタッチ成功率の相関検証

**主要メソッド**:
```python
class AccelerationCorrelationTest:
    def test(self, df: pd.DataFrame, config: TestConfig) -> TestResult:
        """相関検定を実行"""
        
    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """ピアソン相関係数とp値を計算"""
        
    def _compute_effect_size(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cohen's dを計算"""
```

**統計手法**:
- ピアソン相関係数 (正規分布を仮定)
- スピアマン順位相関 (非正規分布に対応)
- 効果量: Cohen's d = (μ₁ - μ₂) / σ_pooled

**出力**:
- 相関係数 r
- p値
- 効果量 d
- 散布図 + 回帰直線

#### 3.2 DefenseDensityAnalysis (仮説2)

**責務**: 防御密度とレイド阻止率の関係分析

**主要メソッド**:
```python
class DefenseDensityAnalysis:
    def test(self, df: pd.DataFrame, config: TestConfig) -> TestResult:
        """ANOVA検定を実行"""
        
    def _bin_density(self, density: pd.Series, bin_width: float) -> pd.Series:
        """密度を区間に分割"""
        
    def _find_optimal_range(self, binned_data: pd.DataFrame) -> Tuple[float, float]:
        """阻止率が最大となる密度範囲を特定"""
```

**統計手法**:
- 一元配置分散分析 (ANOVA)
- 多重比較 (Tukey HSD)
- 効果量: η² (eta squared)

**出力**:
- F統計量、p値
- 最適密度範囲
- ヒートマップ (密度 vs 阻止率)

#### 3.3 RoundChainAnalysis (仮説3)

**責務**: ラウンド間連鎖効果の検証

**主要メソッド**:
```python
class RoundChainAnalysis:
    def test(self, df: pd.DataFrame, config: TestConfig) -> TestResult:
        """連鎖効果を検定"""
        
    def _compute_conditional_prob(self, rounds: List[RoundData]) -> Dict[str, float]:
        """条件付き確率を計算"""
        
    def _compute_transfer_entropy(self, rounds: List[RoundData]) -> float:
        """転移エントロピーを計算"""
```

**統計手法**:
- カイ二乗検定 (独立性の検定)
- 転移エントロピー: TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
- 時系列相関分析

**出力**:
- p(成功|前回成功) vs p(成功|前回失敗)
- χ²統計量、p値
- 転移エントロピー値
- 連続成功回数と成功率の折れ線グラフ

#### 3.4 InteractionEffectTest (仮説4)

**責務**: 個人×チーム交互作用の評価

**主要メソッド**:
```python
class InteractionEffectTest:
    def test(self, df: pd.DataFrame, config: TestConfig) -> TestResult:
        """交互作用モデルを構築・評価"""
        
    def _build_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, LogisticRegression]:
        """3種類のモデルを構築"""
        
    def _compute_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """相互情報量を計算"""
```

**統計手法**:
- ロジスティック回帰 (3モデル比較)
  - Model 1: 個人スキルのみ
  - Model 2: チーム戦術のみ
  - Model 3: 個人 + チーム + 交互作用項
- 相互情報量: I(X;Y) = H(Y) - H(Y|X)
- モデル評価: AUC, 精度, F1スコア

**出力**:
- 各モデルのAUC、精度、F1
- 相互情報量 (個人、チーム、交互作用)
- 特徴量重要度の棒グラフ

#### 3.5 RealtimeIndicatorEvaluation (仮説5)

**責務**: リアルタイム指標の有用性評価

**主要メソッド**:
```python
class RealtimeIndicatorEvaluation:
    def test(self, df: pd.DataFrame, config: TestConfig) -> TestResult:
        """指標の有用性を評価"""
        
    def _measure_computation_cost(self, indicator_func: Callable) -> float:
        """計算コストを測定 (ms)"""
        
    def _compute_usefulness(self, indicator: pd.Series, outcome: pd.Series) -> float:
        """相互情報量で有用性を定量化"""
```

**評価基準**:
- 有用性: 相互情報量 I(指標; 結果) > 0.1
- 計算コスト: < 100 ms (リアルタイム性の要件)
- 実用性スコア: usefulness / log(cost + 1)

**出力**:
- 指標ごとの相互情報量
- 計算コスト (ms)
- 有用性 vs コストの散布図
- 推薦指標リスト

### 4. Visualizer

**責務**: 統計結果の可視化

**主要メソッド**:
```python
class Visualizer:
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, title: str) -> str:
        """散布図 + 回帰直線を生成"""
        
    def plot_heatmap(self, data: pd.DataFrame, title: str) -> str:
        """ヒートマップを生成"""
        
    def plot_bar(self, data: Dict[str, float], title: str) -> str:
        """棒グラフを生成"""
        
    def plot_line(self, x: np.ndarray, y: np.ndarray, title: str) -> str:
        """折れ線グラフを生成"""
```

**可視化ライブラリ**: matplotlib + seaborn
**出力形式**: PNG (300 DPI)
**保存先**: output/figures/

### 5. ReportGenerator

**責務**: 統合レポートの生成

**主要メソッド**:
```python
class ReportGenerator:
    def generate_markdown(self, results: List[TestResult]) -> str:
        """Markdownレポートを生成"""
        
    def export_csv(self, results: List[TestResult], path: str):
        """生データをCSVエクスポート"""
```

**レポート構成**:
```markdown
# 仮説検証レポート

## 概要
- 解析日時
- 入力データ
- サンプルサイズ

## 仮説1: レイダー加速度とタッチ成功率
- 統計量: r=0.65, p<0.001, d=0.82
- 結論: 有意な正の相関あり
- 図表: ![scatter](figures/h1_scatter.png)

[以下、仮説2-5も同様]

## 統合的考察
- 最も重要な要因
- 実用的示唆
```


## Data Models

### 入力データモデル

```python
@dataclass
class TrackingData:
    """video_processor.pyが出力するCSVデータ"""
    frame: int
    player_id: int
    x_coord: float  # ピクセル座標 (後でメートルに変換)
    y_coord: float
```

### 中間データモデル

```python
@dataclass
class EnrichedTrackingData:
    """特徴量を追加したデータ"""
    frame: int
    player_id: int
    x_coord: float
    y_coord: float
    velocity: float  # m/s
    acceleration: float  # m/s²
    team_id: str  # "raider" or "defender"
    
@dataclass
class RoundData:
    """ラウンド単位のデータ"""
    round_id: int
    raider_id: int
    raider_acceleration_0_3m: float
    raider_max_velocity: float
    defense_density: float  # 防御選手間平均距離
    formation_entropy: float
    touch_success: bool
    raid_blocked: bool
    
@dataclass
class MatchData:
    """試合全体のデータ"""
    match_id: str
    rounds: List[RoundData]
    winner: str
```

### 出力データモデル

```python
@dataclass
class TestResult:
    """仮説検定の結果"""
    hypothesis_id: str  # "H1", "H2", etc.
    statistic_name: str  # "correlation", "F-statistic", etc.
    statistic_value: float
    p_value: float
    effect_size: float
    conclusion: str  # "有意な相関あり" など
    figure_paths: List[str]  # 生成された図表のパス
    raw_data: pd.DataFrame  # 生データ
    
@dataclass
class QualityReport:
    """データ品質レポート"""
    total_frames: int
    missing_rate: float
    outlier_count: int
    sample_size: int  # ラウンド数
    warnings: List[str]
```

### 設定データモデル

```python
@dataclass
class TestConfig:
    """解析パラメータ"""
    alpha: float = 0.05  # 有意水準
    effect_size_threshold: float = 0.5  # Cohen's d閾値
    mutual_info_threshold: float = 0.1  # 相互情報量閾値
    density_bin_width: float = 0.5  # 防御密度の区間幅 (m)
    fps: float = 30.0  # 動画のFPS
    court_length: float = 13.0  # コート長さ (m)
    court_width: float = 10.0  # コート幅 (m)
```

## Error Handling

### エラー分類と対処

#### 1. データ品質エラー

**エラー**: 欠損値率が20%を超える
```python
class InsufficientDataQualityError(Exception):
    """データ品質が基準を満たさない"""
```
**対処**: 警告を出力し、欠損値を除外して解析を継続。ただし結果の信頼性が低いことを明記。

**エラー**: サンプルサイズ不足 (< 30ラウンド)
```python
class InsufficientSampleSizeError(Exception):
    """統計的検出力が不足"""
```
**対処**: 警告を出力し、「統計的検出力が限定的」とレポートに記載。

#### 2. 計算エラー

**エラー**: 相関係数の計算で分散がゼロ
```python
class ZeroVarianceError(Exception):
    """変数の分散がゼロ"""
```
**対処**: 該当変数をスキップし、ログに記録。

**エラー**: 行列の特異性 (ロジスティック回帰)
```python
class SingularMatrixError(Exception):
    """行列が特異で逆行列が計算できない"""
```
**対処**: 正則化パラメータを追加 (Ridge回帰に切り替え)。

#### 3. ファイルI/Oエラー

**エラー**: 入力CSVファイルが存在しない
```python
class FileNotFoundError(Exception):
    """入力ファイルが見つからない"""
```
**対処**: エラーメッセージを表示し、処理を中断。

**エラー**: 出力ディレクトリへの書き込み権限がない
```python
class PermissionError(Exception):
    """書き込み権限がない"""
```
**対処**: 代替ディレクトリ (temp) への保存を試行。

### ロギング戦略

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ログレベルの使い分け
# DEBUG: 詳細な計算過程 (開発時のみ)
# INFO: 処理の進捗状況
# WARNING: データ品質の問題、統計的検出力の不足
# ERROR: 計算エラー、ファイルI/O失敗
# CRITICAL: システム全体の停止を要するエラー
```

## Testing Strategy

### 1. ユニットテスト

**対象**: 各コンポーネントの個別メソッド

**テストケース例**:
```python
# test_feature_extractor.py
def test_compute_velocity():
    """速度計算の正確性を検証"""
    df = pd.DataFrame({
        'frame': [0, 1, 2],
        'player_id': [1, 1, 1],
        'x_coord': [0, 3, 6],
        'y_coord': [0, 0, 0]
    })
    result = FeatureExtractor().compute_velocity(df, fps=30)
    expected_velocity = 3.0 * 30  # 3m/frame * 30fps = 90 m/s
    assert np.isclose(result['velocity'].iloc[1], expected_velocity)

def test_compute_acceleration():
    """加速度計算の正確性を検証"""
    # 等加速度運動のテストケース
    
def test_defense_density():
    """防御密度計算の正確性を検証"""
    # 既知の配置パターンでテスト
```

**ツール**: pytest
**カバレッジ目標**: 80%以上

### 2. 統合テスト

**対象**: コンポーネント間の連携

**テストケース例**:
```python
# test_pipeline.py
def test_end_to_end_pipeline():
    """CSVファイルからレポート生成までの全体フロー"""
    pipeline = HypothesisTestingPipeline(config)
    results = pipeline.run('test_data.csv')
    assert len(results) == 5  # 5つの仮説全て実行
    assert os.path.exists('output/hypothesis_report.md')
```

### 3. 統計的妥当性テスト

**対象**: 統計手法の正確性

**テストケース例**:
```python
# test_statistics.py
def test_correlation_with_known_data():
    """既知の相関係数を持つデータで検証"""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # 完全な正の相関
    r, p = AccelerationCorrelationTest()._compute_correlation(x, y)
    assert np.isclose(r, 1.0)
    assert p < 0.001

def test_anova_with_synthetic_data():
    """合成データでANOVAの正確性を検証"""
    # 群間差が既知のデータを生成
```

### 4. データ品質テスト

**対象**: 異常値検出、欠損値処理

**テストケース例**:
```python
# test_data_quality.py
def test_outlier_detection():
    """異常値が正しく検出されるか"""
    df = pd.DataFrame({
        'velocity': [3, 4, 5, 100, 3]  # 100は異常値
    })
    cleaned = DataPreprocessor().remove_outliers(df)
    assert 100 not in cleaned['velocity'].values
```

### 5. 回帰テスト

**目的**: 既存機能の破壊を防ぐ

**方法**: 
- 基準データセットを用意
- 各リリース前に全テストを実行
- 統計量の変化を監視 (許容誤差: ±1%)

