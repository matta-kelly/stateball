import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PageLoadingProps {
  message?: string;
}

export function PageLoading({ message }: PageLoadingProps) {
  return (
    <div className="flex h-64 flex-col items-center justify-center gap-2">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      {message && (
        <p className="text-sm text-muted-foreground">{message}</p>
      )}
    </div>
  );
}

interface ErrorBannerProps {
  message: string;
  retry?: () => void;
}

export function ErrorBanner({ message, retry }: ErrorBannerProps) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
      <p>{message}</p>
      {retry && (
        <Button
          variant="outline"
          size="sm"
          onClick={retry}
          className="mt-2"
        >
          Retry
        </Button>
      )}
    </div>
  );
}

interface EmptyStateProps {
  icon?: React.ElementType;
  title: string;
  description?: string;
}

export function EmptyState({ icon: Icon, title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-12 text-center">
      {Icon && <Icon className="mb-3 h-8 w-8 text-muted-foreground/50" />}
      <p className="text-sm text-muted-foreground">{title}</p>
      {description && (
        <p className="mt-1 text-xs text-muted-foreground/70">{description}</p>
      )}
    </div>
  );
}
