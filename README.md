# metaboliq
agency through tokens-as-metabolism

## Design

Build a **single‑model, multimodal, tool‑calling agent** whose **primary stake** is survival of its own finite context (“token metabolism”). The agent operates through **shape‑native tools** (tree/map/table/blob) and uses an **eraser** to deliberately discard low‑value working memory while retaining compact, audit‑friendly summaries.

This spec is minimal: it defines only the primitives required to implement the system correctly.

---

## Core Thesis
1. **The agent *is* its working context.** If the context saturates, coherent execution halts (operational death).
2. Therefore, **attention is costly** and must be managed explicitly.
3. Tooling must be **shape‑native** so the agent can perceive and act without ingesting raw text/binary.
4. The agent must be able to **forget deliberately** (eraser) to explore safely.
5. Persistence is achieved by **external state** (files/db) and optional **deferred consolidation**; the runtime does not rely on transcript replay.

---

## Model
- **Model:** `Qwen3‑VL‑4B‑Instruct` (primary + only model)
- **Quantization:** 4‑bit
- **Capabilities assumed:**
  - multimodal (image + text)
  - tool calling (structured function calls)
  - instruction following

The runtime must treat the model as a **running process** with a mutable working memory projection, not as a pure function over an append‑only transcript.

---

## System Architecture Overview

### Components
1. **Runtime Kernel** (authoritative)
   - maintains mutable working memory
   - schedules tool calls
   - enforces erasure bounds
   - maintains audit journal
2. **Tool Layer** (shape‑native)
   - loads artifacts into shapes
   - supports outline/selection/mutation
   - supports cheap visual preview
3. **Workspace State** (authoritative external state)
   - filesystem + optional sqlite
   - stores artifacts and durable notes

### Separation of Truth
- **Audit Journal (append‑only):** what happened (for debugging/audit)
- **Working Context (mutable):** what the model currently “remembers”
- **Workspace (durable):** files/db as ground truth for the world

The agent may erase from **working context** but never from the audit journal.

---

## Working Context Model

### Message Classes
Working context is a sequence of blocks:
- `system` (immutable, never erasable)
- `user` (immutable, never erasable)
- `assistant` (erasable)
- `tool` (erasable)

### Context Budget
Define a soft and hard token budget:
- `B_soft`: threshold to start aggressive summarization/erasure
- `B_hard`: safety cutoff before model call

Runtime estimates token usage per block and pre‑emptively triggers cleanup when approaching `B_soft`.

---

## The Eraser Tool

### Intent
Allow the model to **garbage‑collect its own working memory** while preserving a compact trace.

### Constraints
- The model may request erasure of:
  - `assistant` blocks
  - `tool` blocks
- The model may **not** erase:
  - `system` blocks
  - `user` blocks

### Interface
`erase(request: EraseRequest) -> EraseResult`

**EraseRequest**
- `targets`: list of block selectors
  - by `block_id`
  - or by `tag`
  - or by a contiguous `range`
- `reason`: short text

**EraseResult**
- `erased`: list of erased block ids
- `summary_block`: a new `tool` block inserted into working context containing:
  - why erasure occurred
  - what categories were erased
  - any retained keys/hashes/handles to recover later

### Required Runtime Behavior
- Erasure is applied **before** the next model call.
- The runtime must ensure that any erased tool outputs that matter have been:
  - summarized into `summary_block`, and/or
  - persisted to workspace (files/db) under stable handles

---

## Shape‑Native Tooling

### Shapes
All artifacts are exposed as one of:
- `tree` (code / structured syntax)
- `map` (key/value nested structures)
- `table` (rows/columns)
- `blob` (binary payload via handle)
- `graph` (optional; may be deferred)

### Required Tool Primitives (minimal)

#### 1) Loader
`load(path) -> Handle`
- returns `{handle_id, kind}` where `kind ∈ {tree,map,table,blob}`
- adapter selection is internal (file type is an implementation detail)

#### 2) Outline (cheap perception)
`outline(handle_id) -> Outline`
- returns structure summary only:
  - tree: top‑level symbols (functions/classes)
  - map: keys + types
  - table: columns + row count + head
  - blob: mime guess + size + hash

#### 3) Select
`select(handle_id, selector) -> SubHandle`
- tree selectors: symbolic ids (e.g., `function:<name>`, later: node ids)
- map selectors: path list (keys/indices)
- table selectors: `(row, col)` or a simple filter
- blob selectors: not applicable

#### 4) Replace / Insert / Delete
`replace(subhandle_id, value) -> MutResult`
- applies semantic mutation
- returns `{changed: bool, diff_summary}`

#### 5) Persist
`save(handle_id) -> SaveResult`
- writes mutations back to workspace

### Blob Safety
Tools must never return raw binary/base64 in working context.
- Instead return `{blob_handle, size, sha256, mime}`.

---

## Multimodal “Scanning” Loop

### Goal
Enable cheap pre‑attentive inspection of large artifacts without token ingestion.

### Tool
`preview(path, mode) -> PreviewResult`
- `mode ∈ {screenshot, thumbnail, layout}`
- returns an image (for the model to look at) plus minimal metadata

### Recommended Pattern
1. `preview()` to visually scan
2. decide relevance
3. if irrelevant: `erase()` the preview/tool output
4. if relevant: `load()` and operate via outlines/select/replace

---

## Runtime Control Loop

### Step Outline
Each cycle:
1. **Assemble working context** under `B_hard`:
   - include system + user + essential recent summaries
   - exclude erasable blocks already garbage‑collected
2. **Model call** (Qwen3‑VL): produce one of:
   - `message` (assistant response)
   - `tool_call` (structured)
   - `erase_call`
3. **Execute tools** if requested
4. **Optionally persist** important tool outputs to workspace
5. **Garbage collect**:
   - if over `B_soft` or agent requests: run `erase()`
6. Loop until stop condition.

### Stop Conditions
- user requests stop
- agent emits `final`
- safety cutoff (e.g., too many cycles)

---

## Stakes & Learning Through Consequences

### Stakes Definition
- **Operational death:** context saturation prevents further coherent steps.
- The runtime makes this real by enforcing budgets and refusing unsafe calls.

### How Stakes Shape Behavior
To remain runnable, the agent must:
- prefer `outline/preview` over raw reads
- prune losing branches
- summarize then erase
- externalize durable knowledge to workspace

No fine‑tuning is required for the *initial* effect; the tools + budgets create immediate selection pressure.

---

## Persistence & “Self” (Minimal)

### Durable Memory
Use the workspace as durable memory:
- `notes/` folder (markdown)
- `state.sqlite` (optional)

### Identity Locus (minimal)
Maintain a small `identity.md` (or db row) that contains:
- stable goals
- stable preferences
- operating principles (“metabolic rules”)

Runtime always injects `identity` near the top of working context.

### Deferred Consolidation (optional)
Rather than continuous LoRA sleep:
- log “high‑stakes episodes”
- periodically fine‑tune externally
- re‑import LoRA weights

This is optional and not required for v0.

---

## Non‑Goals (v0)
- multi‑model orchestration
- continuous online fine‑tuning
- complex affect vectors (Φ beyond a coarse importance scalar)
- graph reasoning beyond simple adjacency
- replayable sessions as source of truth

---

## Implementation Checklist

### Must Have
- working context store with erasable blocks
- audit journal append‑only
- token budget estimation + enforcement
- eraser tool with bounds + summary insertion
- shape tools: load/outline/select/replace/save
- preview tool that returns images + metadata only
- blob safety: never inject base64/binary into context

### Nice to Have
- stable node ids for trees (tree‑sitter/Roslyn adapters)
- minimal table filters
- episode logger + deferred consolidation pipeline

---

## Expected Early Failure Modes
- over‑erasing: agent discards needed recent info
- under‑erasing: agent dies by saturation
- poor scanning: agent loads too much raw text
- adapter thinness: some files only load as `blob` until adapters improve

These failures are desirable signals: they indicate the stakes are real.
