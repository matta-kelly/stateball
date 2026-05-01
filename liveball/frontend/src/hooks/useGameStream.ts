import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { Game } from "@/types/api";
import { toDateString } from "@/lib/date";
import { queryKeys } from "./queries";

export type StreamStatus = "connected" | "reconnecting" | "stale" | "disconnected";

interface GameStreamState {
  games: Game[];
  status: StreamStatus;
  lastEventAt: number | null;
}

const STALE_THRESHOLD_MS = 45_000; // 3 missed heartbeats (15s server interval)
const INITIAL_RETRY_MS = 1_000;
const MAX_RETRY_MS = 30_000;

// ---------------------------------------------------------------------------
// SSE parser — handles event blocks and heartbeat comments
// ---------------------------------------------------------------------------

interface ParsedEvent {
  type: string | null; // "heartbeat" for comments, event name, or null
  id: string | null;
  data: string;
}

function parseSSEBuffer(buffer: string): {
  parsed: ParsedEvent[];
  remaining: string;
} {
  const parsed: ParsedEvent[] = [];
  const blocks = buffer.split("\n\n");
  const remaining = blocks.pop() ?? "";

  for (const block of blocks) {
    if (!block.trim()) continue;

    let eventType: string | null = null;
    let eventId: string | null = null;
    const dataLines: string[] = [];
    let isComment = false;

    for (const line of block.split("\n")) {
      if (line.startsWith(":")) {
        isComment = true;
      } else if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("id:")) {
        eventId = line.slice(3).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (isComment && dataLines.length === 0) {
      parsed.push({ type: "heartbeat", id: null, data: "" });
    } else if (dataLines.length > 0) {
      parsed.push({ type: eventType, id: eventId, data: dataLines.join("\n") });
    }
  }

  return { parsed, remaining };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useGameStream(enabled: boolean): GameStreamState {
  const [games, setGames] = useState<Game[]>([]);
  const [status, setStatus] = useState<StreamStatus>("disconnected");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const queryClient = useQueryClient();

  const abortRef = useRef<AbortController | null>(null);
  const retryDelayRef = useRef(INITIAL_RETRY_MS);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const staleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  useEffect(() => {
    if (!enabled) {
      abortRef.current?.abort();
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (staleTimerRef.current) clearTimeout(staleTimerRef.current);
      setStatus("disconnected");
      setGames([]);
      setLastEventAt(null);
      return;
    }

    // ------ helpers (closed over refs) ------

    function resetStaleTimer() {
      if (staleTimerRef.current) clearTimeout(staleTimerRef.current);
      staleTimerRef.current = setTimeout(() => {
        setStatus("stale");
      }, STALE_THRESHOLD_MS);
    }

    function handleEvent(evt: ParsedEvent) {
      setLastEventAt(Date.now());
      try {
        if (evt.type === "game_update") {
          const game: Game = JSON.parse(evt.data);
          setGames((prev) => {
            const idx = prev.findIndex((g) => g.game_pk === game.game_pk);
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = game;
              return next;
            }
            return [...prev, game];
          });
          queryClient.setQueryData(queryKeys.game(game.game_pk), game);
        } else if (evt.type === "schedule_sync") {
          const parsed: Game[] = JSON.parse(evt.data);
          setGames(parsed);
          queryClient.setQueryData(
            queryKeys.games(toDateString(new Date())),
            parsed,
          );
        }
      } catch {
        // skip malformed
      }
    }

    function scheduleReconnect() {
      if (!enabledRef.current) return;
      setStatus("reconnecting");
      const delay = retryDelayRef.current;
      retryDelayRef.current = Math.min(delay * 2, MAX_RETRY_MS);
      retryTimerRef.current = setTimeout(connect, delay);
    }

    async function connect() {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      // Fetch baseline state before opening stream
      try {
        const baseResp = await fetch(
          `/api/v1/games?date=${toDateString(new Date())}`,
          { signal: controller.signal },
        );
        if (baseResp.ok) {
          const baseline: Game[] = await baseResp.json();
          setGames(baseline);
        }
      } catch {
        // Non-fatal — proceed with existing games
      }

      if (controller.signal.aborted) return;

      try {
        const resp = await fetch("/api/v1/games/stream", {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });

        if (!resp.ok || !resp.body) {
          throw new Error(`Stream failed: ${resp.status}`);
        }

        setStatus("connected");
        retryDelayRef.current = INITIAL_RETRY_MS;
        resetStaleTimer();

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = parseSSEBuffer(buffer);
          buffer = events.remaining;

          for (const evt of events.parsed) {
            if (evt.type === "heartbeat") {
              resetStaleTimer();
              continue;
            }
            resetStaleTimer();
            setStatus("connected");
            handleEvent(evt);
          }
        }
      } catch (err) {
        if (controller.signal.aborted) return;
      }

      // Stream ended or errored — reconnect
      if (enabledRef.current) {
        scheduleReconnect();
      }
    }

    connect();

    return () => {
      abortRef.current?.abort();
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (staleTimerRef.current) clearTimeout(staleTimerRef.current);
    };
  }, [enabled]);

  return { games, status, lastEventAt };
}
