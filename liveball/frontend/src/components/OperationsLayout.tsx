import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Menu, X } from "lucide-react";
import ModeToggle from "./ModeToggle";
import UserMenu from "./UserMenu";

export default function OperationsLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Top bar */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-card px-3 sm:px-4">
        {/* Left: brand + mode toggle */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-foreground">Stateball</span>
          <ModeToggle className="hidden sm:flex" />
        </div>

        {/* Right: user menu (desktop) / hamburger (mobile) */}
        <div className="hidden sm:flex">
          <UserMenu />
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="rounded-md p-1.5 text-muted-foreground sm:hidden"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>

      {/* Mobile dropdown */}
      {mobileOpen && (
        <div className="border-b border-border bg-card p-3 sm:hidden">
          <div className="flex items-center justify-between">
            <ModeToggle />
            <UserMenu />
          </div>
        </div>
      )}

      {/* Main content — full bleed */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
