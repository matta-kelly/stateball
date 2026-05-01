import { useCallback, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { ArrowLeft, Check, Copy, Trash2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import type { ManagedUser } from "@/types/api";

export default function Admin() {
  const { user } = useAuth();
  if (user?.role !== "admin") return <Navigate to="/" replace />;
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [inviteUrls, setInviteUrls] = useState<Record<string, string>>({});
  const [copied, setCopied] = useState<string | null>(null);

  const fetchUsers = useCallback(async () => {
    const res = await fetch("/api/auth/admin/users");
    if (res.ok) setUsers(await res.json());
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  async function approve(userId: string) {
    const res = await fetch(`/api/auth/admin/approve/${userId}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      setInviteUrls((prev) => ({ ...prev, [userId]: data.invite_url }));
      fetchUsers();
    }
  }

  async function deny(userId: string) {
    const res = await fetch(`/api/auth/admin/deny/${userId}`, { method: "POST" });
    if (res.ok) fetchUsers();
  }

  async function deleteUser(userId: string, username: string) {
    if (!confirm(`Delete user "${username}"? This cannot be undone.`)) return;
    const res = await fetch(`/api/auth/admin/users/${userId}`, { method: "DELETE" });
    if (res.ok) fetchUsers();
  }

  async function copyUrl(userId: string) {
    const url = inviteUrls[userId];
    if (url) {
      await navigator.clipboard.writeText(url);
      setCopied(userId);
      setTimeout(() => setCopied(null), 2000);
    }
  }

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading...</div>;
  }

  const pending = users.filter((u) => u.status === "pending");
  const approved = users.filter((u) => u.status === "approved");
  const active = users.filter((u) => u.status === "active");

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 p-4 sm:p-6">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to games
      </Link>
      <h1 className="text-xl font-semibold sm:text-2xl">Users</h1>

      {/* Pending requests */}
      <section className="space-y-3">
        <h2 className="text-lg font-medium">Pending requests</h2>
        {pending.length === 0 ? (
          <p className="text-sm text-muted-foreground">No pending requests</p>
        ) : (
          <div className="space-y-2">
            {pending.map((u) => (
              <div
                key={u.id}
                className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3"
              >
                <div>
                  <span className="text-sm font-medium">{u.username}</span>
                  {u.created_at && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      {new Date(u.created_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => approve(u.id)}
                    className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => deny(u.id)}
                    className="rounded-md border border-border px-3 py-1 text-xs font-medium text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  >
                    Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Approved (invite sent, waiting for password) */}
      {approved.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-medium">Awaiting password setup</h2>
          <div className="space-y-2">
            {approved.map((u) => (
              <div
                key={u.id}
                className="space-y-2 rounded-md border border-border bg-card px-4 py-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{u.username}</span>
                  <span className="text-xs text-muted-foreground">Invite sent</span>
                </div>
                {inviteUrls[u.id] && (
                  <div className="flex items-center gap-2">
                    <code className="flex-1 truncate rounded bg-background px-2 py-1 text-xs">
                      {inviteUrls[u.id]}
                    </code>
                    <button
                      onClick={() => copyUrl(u.id)}
                      className="text-muted-foreground hover:text-foreground"
                      title="Copy invite URL"
                    >
                      {copied === u.id ? (
                        <Check className="h-4 w-4 text-success" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                )}
                {!inviteUrls[u.id] && (
                  <button
                    onClick={() => approve(u.id)}
                    className="text-xs text-muted-foreground underline hover:text-foreground"
                  >
                    Regenerate invite link
                  </button>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Active users */}
      <section className="space-y-3">
        <h2 className="text-lg font-medium">Active users</h2>
        {active.length === 0 ? (
          <p className="text-sm text-muted-foreground">No active users</p>
        ) : (
          <div className="space-y-2">
            {active.map((u) => (
              <div
                key={u.id}
                className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3"
              >
                <span className="text-sm font-medium">{u.username}</span>
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-card-accent px-2 py-0.5 text-xs text-muted-foreground">
                    {u.role}
                  </span>
                  {u.role !== "admin" && u.id !== user?.id && (
                    <button
                      onClick={() => deleteUser(u.id, u.username)}
                      className="rounded-md p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                      title={`Delete ${u.username}`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
