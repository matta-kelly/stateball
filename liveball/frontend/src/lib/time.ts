/**
 * Format a game datetime as dual timezone: "5:05 PM PT / 8:05 PM ET"
 * Shows local (PT) and ET since MLB operates on Eastern.
 */
export function formatGameTime(datetime: string): string {
  try {
    const d = new Date(datetime);

    const pt = d.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      timeZone: "America/Los_Angeles",
    });

    const et = d.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      timeZone: "America/New_York",
    });

    return `${pt} PT / ${et} ET`;
  } catch {
    return "";
  }
}
