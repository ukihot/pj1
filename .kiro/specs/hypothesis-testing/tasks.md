# Implementation Plan

- [ ] 1. プロジェクト構造とコア設定の作成
  - hypothesis_testing/ディレクトリ構造を作成
  - config.py: TestConfig、QualityReportなどのデータクラスを定義
  - __init__.pyでモジュールをエクスポート
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 2. データ前処理モジュールの実装
- [ ] 2.1 DataPreprocessorクラスの基本実装
  - load_data(): CSVファイル読み込みと必須カラム検証
  - validate_quality(): 欠損値率、異常値、サンプルサイズの検証
  - QualityReportの生成
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 2.2 異常値除外と欠損値処理の実装
  - remove_outliers(): IQR法とIsolation Forestによる異常値除外
  - handle_missing_values(): 線形補間または除外
  - 速度>15m/s、加速度>10m/s²の検出
  - _Requirements: 7.3_

- [ ] 3. 特徴量抽出モジュールの実装
- [ ] 3.1 FeatureExtractorクラスの基本実装
  - compute_velocity(): 速度計算 (m/s)
  - compute_acceleration(): 加速度計算 (m/s²)
  - ピクセル座標からメートル座標への変換ロジック
  - _Requirements: 1.1_

- [ ] 3.2 0-3m区間加速度の計算
  - レイダーの移動開始点を検出
  - 3m到達までの平均加速度を計算
  - _Requirements: 1.1_

- [ ] 3.3 防御密度とフォーメーションエントロピーの計算
  - compute_defense_density(): 選手間平均距離の計算
  - compute_formation_entropy(): シャノンエントロピーの計算
  - チーム識別ロジック (raider vs defender)
  - _Requirements: 2.1, 4.1_

- [ ] 3.4 ラウンド単位データの抽出
  - extract_round_data(): フレームデータをラウンド単位に分割
  - RoundDataオブジェクトの生成
  - タッチ成功/失敗の判定ロジック
  - _Requirements: 3.1_

- [ ] 4. 仮説1検証モジュールの実装
- [ ] 4.1 AccelerationCorrelationTestクラスの実装
  - test(): 相関検定のメインロジック
  - _compute_correlation(): ピアソン・スピアマン相関係数の計算
  - _compute_effect_size(): Cohen's dの計算
  - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [ ] 4.2 仮説1の可視化
  - 散布図 + 回帰直線の生成
  - 相関係数とp値の図表への表示
  - _Requirements: 1.4_

- [ ] 5. 仮説2検証モジュールの実装
- [ ] 5.1 DefenseDensityAnalysisクラスの実装
  - test(): ANOVA検定のメインロジック
  - _bin_density(): 防御密度を0.5m刻みの区間に分割
  - _find_optimal_range(): 阻止率が最大となる密度範囲の特定
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 5.2 仮説2の可視化
  - ヒートマップ (密度 vs 阻止率) の生成
  - 最適密度範囲の強調表示
  - _Requirements: 2.5_

- [ ] 6. 仮説3検証モジュールの実装
- [ ] 6.1 RoundChainAnalysisクラスの実装
  - test(): 連鎖効果検定のメインロジック
  - _compute_conditional_prob(): p(成功|前回成功) と p(成功|前回失敗) の計算
  - _compute_transfer_entropy(): 転移エントロピーの計算
  - カイ二乗検定の実装
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 6.2 仮説3の可視化
  - 連続成功回数と成功率の折れ線グラフ
  - 条件付き確率の棒グラフ
  - _Requirements: 3.5_

- [ ] 7. 仮説4検証モジュールの実装
- [ ] 7.1 InteractionEffectTestクラスの実装
  - test(): 交互作用検定のメインロジック
  - _build_models(): 3種類のロジスティック回帰モデル構築
  - _compute_mutual_information(): 相互情報量の計算
  - モデル評価 (AUC, 精度, F1スコア)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 7.2 仮説4の可視化
  - モデル性能比較の棒グラフ
  - 特徴量重要度の可視化
  - 相互情報量の比較図
  - _Requirements: 4.4_

- [ ] 8. 仮説5検証モジュールの実装
- [ ] 8.1 RealtimeIndicatorEvaluationクラスの実装
  - test(): 指標有用性評価のメインロジック
  - _measure_computation_cost(): 計算コストの測定 (ms)
  - _compute_usefulness(): 相互情報量による有用性定量化
  - 実用性スコアの計算
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 8.2 仮説5の可視化
  - 有用性 vs 計算コストの散布図
  - 推薦指標リストの生成
  - _Requirements: 5.5_

- [ ] 9. 可視化モジュールの実装
- [ ] 9.1 Visualizerクラスの実装
  - plot_scatter(): 散布図 + 回帰直線
  - plot_heatmap(): ヒートマップ
  - plot_bar(): 棒グラフ
  - plot_line(): 折れ線グラフ
  - 共通スタイル設定 (フォント、色、DPI)
  - _Requirements: 1.4, 2.5, 3.5, 4.4, 5.5_

- [ ] 10. レポート生成モジュールの実装
- [ ] 10.1 ReportGeneratorクラスの実装
  - generate_markdown(): Markdownレポートの生成
  - 5つの仮説結果の統合
  - 図表の埋め込み
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 10.2 CSV結果エクスポート
  - export_csv(): 生データのCSV出力
  - 各仮説の統計量をまとめたサマリーCSV
  - _Requirements: 6.4_

- [ ] 11. メインパイプラインの実装
- [ ] 11.1 HypothesisTestingPipelineクラスの実装
  - run(): 全体フローの制御
  - 各モジュールの呼び出しと結果の集約
  - エラーハンドリングとロギング
  - _Requirements: 全要件_

- [ ] 11.2 コマンドラインインターフェースの実装
  - argparseによる引数解析
  - 設定ファイル (YAML) の読み込み
  - 進捗表示とログ出力
  - _Requirements: 8.5_

- [ ] 12. エントリーポイントの作成
  - hypothesis_tester.py: メインスクリプト
  - 使用例のドキュメント化
  - README.mdへの追記
  - _Requirements: 全要件_
