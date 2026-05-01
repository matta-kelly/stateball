import { useState } from "react";
import { cn } from "@/lib/utils";
import WorkshopCards from "./WorkshopCards";
import WorkshopEvals from "./WorkshopEvals";

type Tab = "cards" | "evals";

export default function Workshop() {
  const [tab, setTab] = useState<Tab>("cards");

  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 p-4 sm:p-6">
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-semibold">Workshop</h2>
        <div className="flex items-center gap-0.5 rounded-md bg-muted p-0.5">
          <button
            onClick={() => setTab("cards")}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              tab === "cards"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            Models & Tables
          </button>
          <button
            onClick={() => setTab("evals")}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              tab === "evals"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            Evals
          </button>
        </div>
      </div>

      {tab === "cards" ? <WorkshopCards /> : <WorkshopEvals />}
    </div>
  );
}
