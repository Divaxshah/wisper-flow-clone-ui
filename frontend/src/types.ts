export type ServerEvent =
  | {
      type: "ready" | "started" | "ended";
      status?: string;
      error?: string | null;
      device?: string | null;
      languages?: { label: string; value: string }[];
      profiles?: string[];
      default_profile?: string;
      language?: string;
      profile?: string;
    }
  | { type: "partial"; full: string; live: string; detected_lang: string }
  | {
      type: "commit";
      id: number;
      raw: string;
      cleanup_status?: "pending" | "skipped";
    }
  | {
      type: "cleaned";
      id: number;
      raw: string;
      cleaned: string;
      cleanup_status?: "applied" | "skipped" | "failed";
    }
  | { type: "error" | "warning"; message: string };

export type StatusPayload = {
  cleanup_available: boolean;
  status: string;
  error: string | null;
  device: string | null;
  languages: { label: string; value: string }[];
  profiles: string[];
  default_profile: string;
};
