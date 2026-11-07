/**
 * OMP (Oh My Pi) model roles, mirroring the Roles view of OMP's own model hub.
 *
 * A model picked for a role is saved to OMP's ``modelRoles`` config on the host
 * when the session first launches; "this session only" changes nothing saved.
 */

/** Role id meaning "use this model for this session only; save nothing". */
export const OMP_SESSION_ONLY = "";

/** Session labels the runner reads to save a role assignment before OMP starts. */
export const OMP_ASSIGN_ROLE_LABEL_KEY = "omnigent.omp.assign_role";
export const OMP_ASSIGN_MODEL_LABEL_KEY = "omnigent.omp.assign_model";

export interface OmpRole {
  id: string;
  label: string;
  description: string;
  /** Chat roles take a thinking-effort suffix; model-kind roles do not. */
  group: "chat" | "kind";
  /** Catalog kinds this role accepts. */
  kinds: readonly string[];
}

const CHAT = ["chat"] as const;
const CHAT_OR_TINY = ["chat", "tiny"] as const;

export const OMP_ROLES: readonly OmpRole[] = [
  {
    id: "default",
    label: "Default",
    description: "Main conversation model",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "smol",
    label: "Smol",
    description: "Fast model for lightweight tasks",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "slow",
    label: "Slow",
    description: "Reasoning model for thorough analysis",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "vision",
    label: "Vision",
    description: "Answers questions about images",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "plan",
    label: "Plan",
    description: "Architectural planning in plan mode",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "commit",
    label: "Commit",
    description: "Commit messages and changelogs",
    group: "chat",
    kinds: CHAT,
  },
  { id: "task", label: "Task", description: "Subagent tasks", group: "chat", kinds: CHAT },
  { id: "advisor", label: "Advisor", description: "Advisor agent", group: "chat", kinds: CHAT },
  {
    id: "widget",
    label: "Widget",
    description: "Background job widgets",
    group: "chat",
    kinds: CHAT,
  },
  {
    id: "tiny",
    label: "Tiny",
    description: "Background work such as session titles",
    group: "chat",
    kinds: CHAT_OR_TINY,
  },
  { id: "memory", label: "Memory", description: "Memory work", group: "chat", kinds: CHAT_OR_TINY },
  {
    id: "judge",
    label: "Judge",
    description: "Judgment calls",
    group: "kind",
    kinds: ["judge", "tiny", "chat"],
  },
  { id: "image", label: "Image", description: "Image generation", group: "kind", kinds: ["image"] },
  { id: "web", label: "Web", description: "Web search", group: "kind", kinds: ["search"] },
  { id: "speech", label: "Speech", description: "Text-to-speech", group: "kind", kinds: ["tts"] },
  {
    id: "dictation",
    label: "Dictation",
    description: "Speech-to-text",
    group: "kind",
    kinds: ["stt"],
  },
];

const ROLES_BY_ID = new Map(OMP_ROLES.map((role) => [role.id, role]));

/** Catalog kinds offered for a role; "this session only" runs the chat model. */
function kindsForRole(roleId: string): readonly string[] {
  return ROLES_BY_ID.get(roleId)?.kinds ?? CHAT;
}

/** Models a role accepts, in catalog order. Rows without a kind count as chat. */
export function ompModelsForRole<T extends { kind?: string }>(
  models: readonly T[],
  roleId: string,
): T[] {
  const kinds = kindsForRole(roleId);
  return models.filter((model) => kinds.includes(model.kind ?? "chat"));
}

/** Whether a role takes a thinking-effort suffix (session-only and chat roles). */
export function ompRoleUsesEffort(roleId: string): boolean {
  return roleId === OMP_SESSION_ONLY || ROLES_BY_ID.get(roleId)?.group === "chat";
}

/** Efforts for a model: ``off``, its own ladder, then ``auto``; none if it can't think. */
export function ompEffortOptions(model: { thinking?: readonly string[] } | undefined): string[] {
  const ladder = model?.thinking ?? [];
  return ladder.length > 0 ? ["off", ...ladder, "auto"] : [];
}

/**
 * Create-call fields for an OMP launch.
 *
 * Session only → ``modelOverride`` runs this session on ``model[:effort]``.
 * A role → labels the runner saves to ``modelRoles``; ``default`` also runs it now.
 */
export function ompLaunchFields(
  roleId: string,
  modelId: string,
  effort: string,
): { modelOverride?: string; labels: Record<string, string> } {
  if (!modelId) return { labels: {} };
  const selector = effort && ompRoleUsesEffort(roleId) ? `${modelId}:${effort}` : modelId;
  if (roleId === OMP_SESSION_ONLY) return { modelOverride: selector, labels: {} };
  const labels = {
    [OMP_ASSIGN_ROLE_LABEL_KEY]: roleId,
    [OMP_ASSIGN_MODEL_LABEL_KEY]: selector,
  };
  return roleId === "default" ? { modelOverride: selector, labels } : { labels };
}

const EFFORT_LABELS: Record<string, string> = {
  off: "Off",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra high",
  max: "Max",
  auto: "Auto",
};

/** User-facing label for an OMP thinking level, e.g. ``"xhigh"`` → ``"Extra high"``. */
export function ompEffortLabel(effort: string): string {
  return EFFORT_LABELS[effort] ?? effort;
}
