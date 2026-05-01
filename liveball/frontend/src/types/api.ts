export interface AuthUser {
  id: string;
  username: string;
  role: "admin" | "user";
}

export interface ManagedUser {
  id: string;
  username: string;
  role: string;
  status: string;
  created_at: string | null;
}

export interface Artifact {
  artifact_id: string;
  artifact_type: "xgboost_sim" | "xgboost_live" | "baserunning" | "pitcher_exit" | "win_expectancy" | "feature_manifest" | "n_lookup" | "stopping_thresholds" | "gamma_schedule" | "horizon_weights";
  run_id: string;
  s3_path: string;
  created_at: string;
  metrics: Record<string, unknown>;
  is_prod?: boolean;
  is_test?: boolean;
}

export interface XGBoostMetrics {
  brier_score_test: number;
  naive_brier_test: number;
  brier_skill: number;
  brier_ci_lower: number;
  brier_ci_upper: number;
  inference_ms_per_row: number;
  sweep_mode: string | null;
  sweep_params: Record<string, unknown> | null;
  total_pas: number;
  train_pas: number;
  test_pas: number;
  seasons: number[];
  classes: string[];
  n_features: number;
  best_iteration: number;
  best_score: number;
  xgboost_version: string;
  date_range: { start: string; end: string };
}

export interface BaserunningMetrics {
  n_pas: number;
  n_keys: number;
  n_transition_entries: number;
  source: string;
}

export interface PitcherExitMetrics {
  n_training_rows: number;
  model_metrics: { auc: number; brier: number; log_loss: number; n_trees: number; pos_rate: number };
  source: string;
}

export interface FeatureManifestMetrics {
  n_features: number;
  n_sim_features: number;
  method: string;
  features: string[];
  sim_features: string[];
}

export interface WinExpectancyMetrics {
  n_pas: number;
  n_games: number;
  n_full_keys: number;
  n_no_bases_keys: number;
  n_coarse_keys: number;
  run_diff_range: [number, number];
  source_events: string;
  source_games: string;
}

export interface ConvergenceConfigMetrics {
  eval_id: string;
  n_states: number;
  target_se: number;
  n_min: number;
  n_max: number;
}

export interface HorizonCutoffMetrics {
  eval_id: string;
  n_states: number;
  imp_threshold: number;
  h_min: number;
  h_max: number;
  h_median: number;
}

export interface CalibrationArtifactStatus {
  artifact_id: string;
  is_prod: boolean;
  is_test: boolean;
}

export interface ArtifactVersion {
  artifact_id: string;
  run_id: string;
  s3_path: string;
  is_prod: boolean;
  is_test: boolean;
  created_at: string;
  metrics: Record<string, unknown>;
}

export interface CalibrationRun {
  eval_id: string;
  created_at: string;
  n_games: number;
  n_sims: number;
  accuracy: number;
  mean_mc_time: number;
  n_per_inning: number;
  total_time: number;
  artifacts: {
    n_lookup: CalibrationArtifactStatus | null;
    stopping_thresholds: CalibrationArtifactStatus | null;
    gamma_schedule: CalibrationArtifactStatus | null;
    horizon_weights: CalibrationArtifactStatus | null;
  };
  is_promoted: boolean;
}

export interface ValidationCheck {
  name: string;
  passed: boolean;
  detail?: string;
}

export interface ValidationResult {
  valid: boolean;
  checks: ValidationCheck[];
}

export interface CurrentABPitch {
  num: number;
  type_code: string | null;
  type_name: string | null;
  speed: number | null;
  px: number | null;
  pz: number | null;
  result: string | null;
  is_strike: boolean;
  is_ball: boolean;
  is_in_play: boolean;
  balls_after: number | null;
  strikes_after: number | null;
}

export interface CurrentAB {
  batter_id: number | null;
  batter_name: string | null;
  pitcher_id: number | null;
  pitcher_name: string | null;
  bat_side: string | null;
  sz_top: number | null;
  sz_bottom: number | null;
  pitches: CurrentABPitch[];
}

export interface Game {
  game_pk: number;
  game_date: string;
  game_datetime: string;
  game_type: string;
  status: string;
  abstract_game_state: string;
  away_team_id: number;
  away_team_name: string;
  home_team_id: number;
  home_team_name: string;
  away_score: number | null;
  home_score: number | null;
  venue_name: string | null;
  // live fields (null until tracker activates)
  inning: number | null;
  inning_half: string | null;
  outs: number | null;
  balls: number | null;
  strikes: number | null;
  runners: string | null;
  current_batter_id: number | null;
  current_batter_name: string | null;
  current_pitcher_id: number | null;
  current_pitcher_name: string | null;
  last_play: string | null;
  current_ab: CurrentAB | null;
  live_updated_at: string | null;
  // sim results
  sim_p_home_win: number | null;
  sim_p_home_win_se: number | null;
  sim_mean_home_score: number | null;
  sim_mean_away_score: number | null;
  sim_n_sims: number | null;
  sim_updated_at: string | null;
  sim_duration_ms: number | null;
  sim_we_baseline: number | null;
}

export interface PlayerGameStats {
  ab: number;
  h: number;
  r: number;
  rbi: number;
  bb: number;
  so: number;
  hr: number;
  avg: string;
}

export interface PitcherGameStats {
  ip: string;
  h: number;
  r: number;
  er: number;
  bb: number;
  so: number;
  pitches: number;
  era: string;
}

export interface LineupPlayer {
  id: number;
  name: string;
  position: string;
  jersey: string;
  batting_order: number;
  stats: PlayerGameStats;
}

export interface PitcherInfo {
  id: number;
  name: string;
  jersey: string;
  stats: PitcherGameStats;
}

export interface BullpenPitcher {
  id: number;
  name: string;
  jersey: string;
}

export interface BenchPlayer {
  id: number;
  name: string;
  jersey: string;
  position: string;
  stats: { ab: number; h: number; rbi: number };
}

export interface TeamDetail {
  team_name: string;
  abbreviation: string;
  lineup: LineupPlayer[];
  pitcher: PitcherInfo;
  pitchers_used: PitcherInfo[];
  bullpen: BullpenPitcher[];
  bench: BenchPlayer[];
}

export interface PlayEntry {
  inning: number;
  half: string;
  event: string;
  description: string;
  is_scoring: boolean;
}

export interface LinescoreInning {
  num: number;
  away: number | null;
  home: number | null;
}

export interface LinescoreTotals {
  runs: number;
  hits: number;
  errors: number;
}

export interface Linescore {
  innings: LinescoreInning[];
  away: LinescoreTotals;
  home: LinescoreTotals;
}

export interface GameDetailData {
  home: TeamDetail;
  away: TeamDetail;
  plays: PlayEntry[];
  linescore: Linescore;
}

export type GameStateEventTrigger =
  | "inning_start"
  | "half_start"
  | "out_recorded"
  | "pa_start";

export interface GameStateEvent {
  ts: string;
  trigger: GameStateEventTrigger;
  inning: number | null;
  inning_half: string | null;
  outs: number | null;
  balls: number | null;
  strikes: number | null;
  current_batter_id: number | null;
  current_pitcher_id: number | null;
}

export interface SimResult {
  ts: string;
  p_home_win: number;
  p_home_win_se: number;
  we_baseline: number | null;
  n_sims: number | null;
  duration_ms: number | null;
  mean_home_score: number | null;
  mean_away_score: number | null;
}

export interface PollEntry {
  ts: number;
  response_ms: number;
  changed: boolean;
  error: string | null;
  event_type: string | null;
}

export interface GamePollMetrics {
  game_pk: number;
  abstract_game_state: string;
  poll_count: number;
  success_count: number;
  error_count: number;
  changed_count: number;
  last_error: string | null;
  last_poll_at: number | null;
  last_change_at: number | null;
  avg_response_ms: number;
  max_response_ms: number;
  poll_interval_s: number;
  started_at: number;
  history: PollEntry[];
}

export interface HorizonDiagnostic {
  horizon: string;
  n: number;
  pred_mae: number | null;
  entry_mae: number | null;
  improvement: number;
  move_mae: number | null;
  move_bias: number | null;
  bss: number | null;
  reliability: number | null;
}

export interface SimEval {
  eval_id: string;
  config_id: string;
  created_at: string;
  n_games: number;
  n_sims: number;
  accuracy: number;
  mean_p_home: number;
  mean_mc_time: number;
  score_mae: number | null;
  prune_rate: number | null;
  estimator: string | null;
  n_per_inning: number | null;
}

export interface BrierBin {
  bin_lo: number;
  bin_hi: number;
  n: number;
  mean_pred: number;
  actual_rate: number;
  gap: number;
  reliability_contrib: number;
}

export interface SimEvalDetail extends SimEval {
  total_time: number | null;
  setup_time: number | null;
  score_error_home: number | null;
  score_error_away: number | null;
  seed: number | null;
  artifact_path: string | null;
  accuracy_by_inning: Record<string, { n: number; correct: number; accuracy: number }>;
  accuracy_by_phase: Record<string, { n: number; correct: number; accuracy: number }>;
  win_probability: {
    brier: number;
    reliability: number;
    resolution: number;
    uncertainty: number;
    brier_skill: number;
    bins: BrierBin[];
  } | null;
  scores: { mean_error_home: number; mean_error_away: number; mean_abs_error: number } | null;
  level_diagnostics: Record<string, {
    n: number; entry_mae: number | null; pred_mae: number | null;
    improvement: number | null; pred_we_std: number | null;
    delta_std: number | null; brier: number | null; brier_skill: number | null;
    reliability: number | null; resolution: number | null; uncertainty: number | null;
  }>;
  games: {
    game_pk: number; game_date: string | null;
    entry_inning: number; entry_phase: string | null;
    p_home_win: number; actual_home_win: boolean; correct: boolean;
  }[];
}

export interface FeedHealthSummary {
  active_trackers: number;
  total_trackers: number;
  total_polls: number;
  error_rate_pct: number;
}

export interface FeedAlert {
  game_pk: number;
  level: "error" | "warn";
  message: string;
}

export interface FeedHealth {
  summary: FeedHealthSummary;
  alerts: FeedAlert[];
}

export interface PollHistoryGame {
  game_pk: number;
  game_date: string;
}

export interface PollHistoryDetail {
  game_pk: number;
  game_date: string;
  total_polls: number;
  change_rate: number;
  wasted_poll_pct: number;
  error_rate: number;
  avg_response_ms: number;
  polls_per_change_by_event: Record<string, number>;
  entries: PollEntry[];
}

export interface ModelingOverviewData {
  prod: { artifacts: Record<string, Artifact>; validation: ValidationResult };
  test: { artifacts: Record<string, Artifact>; validation: ValidationResult };
}

export interface BatterStats {
  batter_id: number;
  full_name: string | null;
  team_name: string | null;
  position: string | null;
  bats: string | null;
  game_date: string;
  season_pa: number;
  season_ba: number | null;
  season_obp: number | null;
  season_slg: number | null;
  season_ops: number | null;
  season_woba: number | null;
  season_k_pct: number | null;
  season_bb_pct: number | null;
  season_avg_ev: number | null;
  season_barrel_pct: number | null;
  career_pa: number;
  career_ba: number | null;
  career_obp: number | null;
  career_slg: number | null;
  career_ops: number | null;
  career_woba: number | null;
  career_k_pct: number | null;
  career_bb_pct: number | null;
  career_avg_ev: number | null;
  career_barrel_pct: number | null;
}

export interface PitcherStats {
  pitcher_id: number;
  full_name: string | null;
  team_name: string | null;
  position: string | null;
  throws: string | null;
  game_date: string;
  season_bf: number;
  season_ip: number | null;
  season_whip: number | null;
  season_k_pct: number | null;
  season_bb_pct: number | null;
  season_hr9: number | null;
  season_woba: number | null;
  career_bf: number;
  career_ip: number | null;
  career_whip: number | null;
  career_k_pct: number | null;
  career_bb_pct: number | null;
  career_hr9: number | null;
  career_woba: number | null;
}
