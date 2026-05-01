import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "@/lib/api";

// ── Query Keys ──

export const queryKeys = {
  games: (date?: string) => ["games", date] as const,
  game: (gamePk: number) => ["game", gamePk] as const,
  gameDetail: (gamePk: number) => ["gameDetail", gamePk] as const,
  gameStateEvents: (gamePk: number) => ["gameStateEvents", gamePk] as const,
  simResults: (gamePk: number) => ["simResults", gamePk] as const,
  feedHealth: () => ["feedHealth"] as const,
  feedMetrics: () => ["feedMetrics"] as const,
  feedHistory: () => ["feedHistory"] as const,
  feedGameHistory: (gamePk: number, gameDate: string) =>
    ["feedGameHistory", gamePk, gameDate] as const,
  modelingOverview: () => ["modelingOverview"] as const,
  artifacts: (type?: string) => ["artifacts", type] as const,
  artifactsByType: (type: string) => ["artifactsByType", type] as const,
  simEvals: () => ["simEvals"] as const,
  simEval: (id: string) => ["simEval", id] as const,
  evalDiagnostics: (evalId: string, inning?: string) =>
    ["evalDiagnostics", evalId, inning] as const,
  calibrations: () => ["calibrations"] as const,
  statsBatters: (season?: number, offset?: number, limit?: number) => ["statsBatters", season, offset, limit] as const,
  statsPitchers: (season?: number, offset?: number, limit?: number) => ["statsPitchers", season, offset, limit] as const,
  statsSeasons: () => ["statsSeasons"] as const,
} as const;

// ── Query Hooks ──

export function useGames(date?: string, opts?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: queryKeys.games(date),
    queryFn: () => api.getGames(date),
    refetchInterval: opts?.refetchInterval,
  });
}

export function useGame(gamePk: number) {
  return useQuery({
    queryKey: queryKeys.game(gamePk),
    queryFn: () => api.getGame(gamePk),
    enabled: !!gamePk,
  });
}

export function useGameDetail(gamePk: number) {
  return useQuery({
    queryKey: queryKeys.gameDetail(gamePk),
    queryFn: () => api.getGameDetail(gamePk),
    enabled: !!gamePk,
    retry: false,
  });
}

export function useGameStateEvents(gamePk: number, opts?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: queryKeys.gameStateEvents(gamePk),
    queryFn: () => api.getGameStateEvents(gamePk),
    enabled: !!gamePk,
    retry: false,
    refetchInterval: opts?.refetchInterval,
  });
}

export function useSimResults(gamePk: number, opts?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: queryKeys.simResults(gamePk),
    queryFn: () => api.getSimResults(gamePk),
    enabled: !!gamePk,
    retry: false,
    refetchInterval: opts?.refetchInterval,
  });
}

export function useModelingOverview() {
  return useQuery({
    queryKey: queryKeys.modelingOverview(),
    queryFn: api.getModelingOverview,
  });
}

export function useArtifacts(type?: string) {
  return useQuery({
    queryKey: queryKeys.artifacts(type),
    queryFn: () => api.getArtifacts(type),
  });
}

export function useArtifactsByType(type: string) {
  return useQuery({
    queryKey: queryKeys.artifactsByType(type),
    queryFn: () => api.getArtifactsByType(type),
  });
}

export function useSimEvals() {
  return useQuery({
    queryKey: queryKeys.simEvals(),
    queryFn: api.getSimEvals,
  });
}

export function useSimEval(evalId: string) {
  return useQuery({
    queryKey: queryKeys.simEval(evalId),
    queryFn: () => api.getSimEval(evalId),
    enabled: !!evalId,
  });
}

export function useEvalDiagnostics(evalId: string, inning?: string) {
  return useQuery({
    queryKey: queryKeys.evalDiagnostics(evalId, inning),
    queryFn: () => api.getEvalDiagnostics(evalId, inning),
    enabled: !!evalId,
  });
}

export function useCalibrations() {
  return useQuery({
    queryKey: queryKeys.calibrations(),
    queryFn: api.getCalibrations,
  });
}

export function useFeedHealth() {
  return useQuery({
    queryKey: queryKeys.feedHealth(),
    queryFn: api.getFeedHealth,
    refetchInterval: 10_000,
  });
}

export function useFeedMetrics() {
  return useQuery({
    queryKey: queryKeys.feedMetrics(),
    queryFn: api.getFeedMetrics,
  });
}

export function useFeedHistory() {
  return useQuery({
    queryKey: queryKeys.feedHistory(),
    queryFn: api.getFeedHistory,
  });
}

export function useFeedGameHistory(gamePk: number, gameDate: string) {
  return useQuery({
    queryKey: queryKeys.feedGameHistory(gamePk, gameDate),
    queryFn: () => api.getFeedGameHistory(gamePk, gameDate),
    enabled: !!gamePk && !!gameDate,
  });
}

export function useStatsBatters(season?: number, offset = 0, limit = 50) {
  return useQuery({
    queryKey: queryKeys.statsBatters(season, offset, limit),
    queryFn: () => api.getBatterStats(season, offset, limit),
  });
}

export function useStatsPitchers(season?: number, offset = 0, limit = 50) {
  return useQuery({
    queryKey: queryKeys.statsPitchers(season, offset, limit),
    queryFn: () => api.getPitcherStats(season, offset, limit),
  });
}

export function useStatsSeasons() {
  return useQuery({
    queryKey: queryKeys.statsSeasons(),
    queryFn: api.getStatsSeasons,
  });
}

// ── Mutation Hooks ──

export function useSetArtifactSlot() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ artifactId, slot, active = true }: { artifactId: string; slot: string; active?: boolean }) =>
      api.setArtifactSlot(artifactId, slot, active),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["artifacts"] });
      qc.invalidateQueries({ queryKey: ["artifactsByType"] });
      qc.invalidateQueries({ queryKey: queryKeys.modelingOverview() });
    },
  });
}

export function useDeleteArtifact() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (artifactId: string) => api.deleteArtifact(artifactId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["artifacts"] });
      qc.invalidateQueries({ queryKey: ["artifactsByType"] });
      qc.invalidateQueries({ queryKey: queryKeys.modelingOverview() });
    },
  });
}

export function usePromoteArtifact() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ artifactId, slot = "prod" }: { artifactId: string; slot?: string }) =>
      api.promoteArtifact(artifactId, slot),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["artifactsByType"] });
      qc.invalidateQueries({ queryKey: queryKeys.calibrations() });
    },
  });
}

export function usePromoteCalibration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ evalId, slot = "prod" }: { evalId: string; slot?: string }) =>
      api.promoteCalibration(evalId, slot),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.calibrations() });
      qc.invalidateQueries({ queryKey: ["artifactsByType"] });
    },
  });
}
