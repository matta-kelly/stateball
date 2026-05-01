import type {
  Artifact,
  BatterStats,
  FeedHealth,
  Game,
  GameDetailData,
  GamePollMetrics,
  ModelingOverviewData,
  PitcherStats,
  PollHistoryDetail,
  PollHistoryGame,
  SimEval,
  SimEvalDetail,
  GameStateEvent,
  SimResult,
  CalibrationRun,
  HorizonDiagnostic,
  ArtifactVersion,
} from "@/types/api";

const BASE = "/api/v1";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    signal: init?.signal ?? AbortSignal.timeout(10_000),
    ...init,
  });
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("Not authenticated");
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  return res.json();
}

export async function getModelingOverview(): Promise<ModelingOverviewData> {
  return fetchJson<ModelingOverviewData>(`${BASE}/modeling/overview`);
}

export async function getArtifacts(type?: string): Promise<Artifact[]> {
  const params = type ? `?type=${encodeURIComponent(type)}` : "";
  return fetchJson<Artifact[]>(`${BASE}/artifacts${params}`);
}

export async function setArtifactSlot(
  artifactId: string,
  slot: string,
  active: boolean = true,
): Promise<void> {
  const params = `?slot=${encodeURIComponent(slot)}&active=${active}`;
  await fetchJson(`${BASE}/artifacts/${encodeURIComponent(artifactId)}/slot${params}`, {
    method: "PATCH",
  });
}

export async function deleteArtifact(artifactId: string): Promise<void> {
  await fetchJson(`${BASE}/artifacts/${encodeURIComponent(artifactId)}`, {
    method: "DELETE",
  });
}

export async function getGames(date?: string, gameType?: string): Promise<Game[]> {
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (gameType) params.set("game_type", gameType);
  const qs = params.toString();
  return fetchJson<Game[]>(`${BASE}/games${qs ? `?${qs}` : ""}`);
}

export async function getSimEvals(): Promise<SimEval[]> {
  return fetchJson<SimEval[]>(`${BASE}/evals`);
}

export async function getSimEval(evalId: string): Promise<SimEvalDetail> {
  return fetchJson<SimEvalDetail>(`${BASE}/evals/${evalId}`);
}

export async function getEvalDiagnostics(evalId: string, inning?: string): Promise<{ horizons: HorizonDiagnostic[] }> {
  const params = inning && inning !== "all" ? `?inning=${inning}` : "";
  return fetchJson(`${BASE}/evals/${evalId}/diagnostics${params}`);
}

export async function getCalibrations(): Promise<CalibrationRun[]> {
  return fetchJson<CalibrationRun[]>(`${BASE}/calibrations`);
}

export async function getArtifactsByType(artifactType: string): Promise<ArtifactVersion[]> {
  return fetchJson<ArtifactVersion[]>(`${BASE}/artifacts/${artifactType}`);
}

export async function promoteArtifact(artifactId: string, slot: string = "prod"): Promise<{ promoted: string; slot: string }> {
  return fetchJson(`${BASE}/artifacts/${artifactId}/promote?slot=${slot}`, { method: "POST" });
}

export async function promoteCalibration(evalId: string, slot: string = "prod"): Promise<{ promoted: string[]; slot: string }> {
  return fetchJson(`${BASE}/calibrations/${evalId}/promote?slot=${slot}`, { method: "POST" });
}

export async function getFeedMetrics(): Promise<GamePollMetrics[]> {
  return fetchJson<GamePollMetrics[]>(`${BASE}/feed/metrics`);
}

export async function getFeedHealth(): Promise<FeedHealth> {
  return fetchJson<FeedHealth>(`${BASE}/feed/health`);
}

export async function getFeedHistory(): Promise<PollHistoryGame[]> {
  return fetchJson<PollHistoryGame[]>(`${BASE}/feed/history`);
}

export async function getFeedGameHistory(
  gamePk: number,
  gameDate: string,
): Promise<PollHistoryDetail> {
  return fetchJson<PollHistoryDetail>(
    `${BASE}/feed/history/${gamePk}?game_date=${encodeURIComponent(gameDate)}`,
  );
}

export async function getGame(gamePk: number): Promise<Game> {
  return fetchJson<Game>(`${BASE}/games/${gamePk}`);
}

export async function getGameDetail(gamePk: number): Promise<GameDetailData> {
  return fetchJson<GameDetailData>(`${BASE}/games/${gamePk}/detail`);
}

export async function getGameStateEvents(gamePk: number): Promise<GameStateEvent[]> {
  return fetchJson<GameStateEvent[]>(`${BASE}/games/${gamePk}/game-state-events`);
}

export async function getSimResults(gamePk: number): Promise<SimResult[]> {
  return fetchJson<SimResult[]>(`${BASE}/games/${gamePk}/sim-results`);
}

export async function getBatterStats(season?: number, offset = 0, limit = 50): Promise<BatterStats[]> {
  const p = new URLSearchParams();
  if (season) p.set("season", String(season));
  p.set("offset", String(offset));
  p.set("limit", String(limit));
  return fetchJson<BatterStats[]>(`${BASE}/stats/batters?${p}`, {
    signal: AbortSignal.timeout(60_000),
  });
}

export async function getPitcherStats(season?: number, offset = 0, limit = 50): Promise<PitcherStats[]> {
  const p = new URLSearchParams();
  if (season) p.set("season", String(season));
  p.set("offset", String(offset));
  p.set("limit", String(limit));
  return fetchJson<PitcherStats[]>(`${BASE}/stats/pitchers?${p}`, {
    signal: AbortSignal.timeout(60_000),
  });
}

export async function getStatsSeasons(): Promise<number[]> {
  return fetchJson<number[]>(`${BASE}/stats/seasons`);
}

