interface ComingSoonProps {
  title: string;
  description?: string;
}

export default function ComingSoon({ title, description }: ComingSoonProps) {
  return (
    <div className="flex flex-1 items-center justify-center">
      <div className="rounded-lg border border-border bg-card p-12 text-center shadow-sm">
        <h2 className="text-2xl font-semibold text-card-foreground">{title}</h2>
        <p className="mt-2 text-muted-foreground">
          {description ?? "Coming soon."}
        </p>
      </div>
    </div>
  );
}
