import { describe, expect, it } from "vitest";

import {
  OMP_ASSIGN_MODEL_LABEL_KEY,
  OMP_ASSIGN_ROLE_LABEL_KEY,
  OMP_ROLES,
  OMP_SESSION_ONLY,
  ompEffortOptions,
  ompLaunchFields,
  ompModelsForRole,
  ompRoleUsesEffort,
} from "./ompRoles";

const MODELS = [
  {
    id: "ollama/glm-5.1:cloud",
    displayName: "glm",
    kind: "chat",
    thinking: ["low", "high", "max"],
  },
  { id: "ollama/gemma4", displayName: "gemma", kind: "chat", thinking: [] },
  { id: "local/lfm2-1.2b", displayName: "lfm", kind: "tiny" },
  { id: "openai-codex/gpt-image-2", displayName: "img", kind: "image" },
  { id: "web/brave", displayName: "brave", kind: "search" },
  { id: "local/kokoro", displayName: "kokoro", kind: "tts" },
  { id: "local/whisper-base", displayName: "whisper", kind: "stt" },
  { id: "openrouter/~liquid/d1-latest", displayName: "d1", kind: "judge" },
];
const ids = (role: string) => ompModelsForRole(MODELS, role).map((m) => m.id);

describe("OMP roles", () => {
  it("lists every chat and model-kind role OMP has", () => {
    expect(OMP_ROLES.map((r) => r.id)).toEqual([
      "default",
      "smol",
      "slow",
      "vision",
      "plan",
      "commit",
      "task",
      "advisor",
      "widget",
      "tiny",
      "memory",
      "judge",
      "image",
      "web",
      "speech",
      "dictation",
    ]);
  });

  it("filters the model list to the kinds each role accepts", () => {
    expect(ids(OMP_SESSION_ONLY)).toEqual(["ollama/glm-5.1:cloud", "ollama/gemma4"]);
    expect(ids("slow")).toEqual(["ollama/glm-5.1:cloud", "ollama/gemma4"]);
    expect(ids("tiny")).toEqual(["ollama/glm-5.1:cloud", "ollama/gemma4", "local/lfm2-1.2b"]);
    expect(ids("judge")).toEqual([
      "ollama/glm-5.1:cloud",
      "ollama/gemma4",
      "local/lfm2-1.2b",
      "openrouter/~liquid/d1-latest",
    ]);
    expect(ids("image")).toEqual(["openai-codex/gpt-image-2"]);
    expect(ids("web")).toEqual(["web/brave"]);
    expect(ids("speech")).toEqual(["local/kokoro"]);
    expect(ids("dictation")).toEqual(["local/whisper-base"]);
  });

  it("treats rows without a kind as chat models", () => {
    const rows: { id: string; kind?: string }[] = [{ id: "pi/x" }];
    expect(ompModelsForRole(rows, "default")).toHaveLength(1);
  });

  it("offers off, the model's own ladder, and auto as efforts", () => {
    expect(ompEffortOptions(MODELS[0])).toEqual(["off", "low", "high", "max", "auto"]);
    expect(ompEffortOptions(MODELS[1])).toEqual([]);
    expect(ompEffortOptions(undefined)).toEqual([]);
  });

  it("only uses effort for session-only and chat-type roles", () => {
    expect(ompRoleUsesEffort(OMP_SESSION_ONLY)).toBe(true);
    expect(ompRoleUsesEffort("slow")).toBe(true);
    expect(ompRoleUsesEffort("tiny")).toBe(true);
    for (const role of ["judge", "image", "web", "speech", "dictation"]) {
      expect(ompRoleUsesEffort(role)).toBe(false);
    }
  });
});

describe("ompLaunchFields", () => {
  it("session only: launches this session on model:effort and saves nothing", () => {
    expect(ompLaunchFields(OMP_SESSION_ONLY, "ollama/glm-5.1:cloud", "high")).toEqual({
      modelOverride: "ollama/glm-5.1:cloud:high",
      labels: {},
    });
    expect(ompLaunchFields(OMP_SESSION_ONLY, "", "high")).toEqual({ labels: {} });
  });

  it("a non-default role is saved via labels and leaves this session's model alone", () => {
    expect(ompLaunchFields("slow", "ollama/glm-5.1:cloud", "max")).toEqual({
      labels: {
        [OMP_ASSIGN_ROLE_LABEL_KEY]: "slow",
        [OMP_ASSIGN_MODEL_LABEL_KEY]: "ollama/glm-5.1:cloud:max",
      },
    });
  });

  it("the default role is saved and also used for this session", () => {
    expect(ompLaunchFields("default", "ollama/glm-5.1:cloud", "")).toEqual({
      modelOverride: "ollama/glm-5.1:cloud",
      labels: {
        [OMP_ASSIGN_ROLE_LABEL_KEY]: "default",
        [OMP_ASSIGN_MODEL_LABEL_KEY]: "ollama/glm-5.1:cloud",
      },
    });
  });

  it("model-kind roles never get an effort suffix", () => {
    expect(ompLaunchFields("image", "openai-codex/gpt-image-2", "high").labels).toEqual({
      [OMP_ASSIGN_ROLE_LABEL_KEY]: "image",
      [OMP_ASSIGN_MODEL_LABEL_KEY]: "openai-codex/gpt-image-2",
    });
  });

  it("a role without a model assigns nothing", () => {
    expect(ompLaunchFields("slow", "", "high")).toEqual({ labels: {} });
  });
});
