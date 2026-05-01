import { Badge } from "@/components/ui/badge";

export function SlotBadges({ isProd, isTest }: { isProd: boolean; isTest: boolean }) {
  return (
    <div className="flex gap-1">
      {isProd && (
        <Badge className="bg-emerald-500/15 text-emerald-500 border-transparent">
          prod
        </Badge>
      )}
      {isTest && (
        <Badge className="bg-info/15 text-info border-transparent">
          test
        </Badge>
      )}
    </div>
  );
}
