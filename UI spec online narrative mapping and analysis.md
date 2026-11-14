# UI spec: online narrative mapping

Visual and interaction spec for the website. The data and endpoints come from section 9 of the [project spec](./Coordinated-behavior%20detector%20project%20spec.md). This doc decides how it looks and feels. Terse on purpose, for the team and for coding agents.

## 1. Direction

- **Feel:** super minimalist. Group balls fill the page. No banners, no toolbars, no boxed controls, no panels that are always open.
- **Pages:** a menu and a unified Live analysis view with streaming groups and chat. Legacy Explore remains available at its direct URL.
- **Unit of display:** one ball per group of similar posts. Ball area represents post count; colour represents group concern.

**Copy rules (honesty is judged)**

- Ball colour means **group concern**, not proven maliciousness or falsity. Uploaded material has an independent content category and concern.
- The internal review-priority score is a triage heuristic, never a probability; show concern levels prominently instead of calling that score concern.
- Describe what was observed, not who is guilty. Never label an account.
- No 2D/3D controls, projected points, dimensional copy, rotation or semantic distances. Ball positions are only a layout.
- Propaganda-style persuasion can be identified from observable rhetoric without establishing truth, intent or coordination. Do not equate any political viewpoint with propaganda.
- The live test streams a historical dataset, not real-time social traffic, and the page says so.

## 2. Design tokens

All colours and sizes are CSS variables in `:root` in `styles.css`. No other colours in the code.

| Token | Value | Use |
| --- | --- | --- |
| `--bg` | `#1a1a1c` | Page background |
| `--surface` | `#232326` | Chat box, cards (`--glass` is this at 86 percent) |
| `--border` | `#34343a` | Hairlines |
| `--text` | `#ececea` | Main text |
| `--text-2` | `#a3a3a6` | Secondary text |
| `--text-3` | `#8a8a8e` | Hints and corner text |
| `--concern-none` / `-low` / `-medium` / `-high` | `#6c7078` / `#9a9468` / `#d2a044` / `#d65a4c` | Concern colours |
| `--radius` / `--radius-input` | `18px` / `22px` | Cards and bubbles / the chat box |

**Typography:** JetBrains Mono, self-hosted woff2 in `web/fonts/` (400 and 600). Title 23 px, body 16 px, small 14 px. No CDN assets.

## 3. Menu (`/`)

Left-aligned column, at most 620 px wide: title, tagline, two plain text rows with an arrow (Open live analysis, Browse datasets), one status line from `/api/live/chat/status`, and the honesty notes. No buttons.

**Browse datasets (`/data`):** text tabs for the allowed datasets (the Live source is marked "live"), a one-line description with licence and missing metadata, previous/next/random paging, and 25 rounded row cards showing the post text with the other columns in small text. Rows are proxied read-only from the Hugging Face rows API through `GET /api/datasets/{id}/rows`; only datasets listed by `GET /api/datasets` can be read, and failures are shown explicitly.

## 4. Group balls (shared by Explore and Live test, `map.js`)

- One circular, focusable element per non-discarded group. Radius is proportional to the square root of post count, so area is proportional to count.
- A deterministic non-overlapping pack uses final group sizes. Live balls grow in stable slots; positions have no semantic meaning.
- New balls fade and scale in, and existing balls move smoothly when the layout changes. Existing elements are retained across snapshots, so updates do not replay their entrance; reduced-motion disables animations.
- Fill and outline use the group's concern colour. Each ball contains the count and, when large enough, a wrapped label. Full label, count and concern are available through the accessible name and tooltip.
- **Discarded groups** (assessed, below `MIN_REVIEW_PRIORITY`) and ungrouped noise are not drawn. Failed and pending groups remain visible.
- **Selection:** click or keyboard-activate a ball. Its outline brightens and other balls dim. Click empty space or press Esc to clear.
- **Corner text:** menu link, total posts, groups shown, groups dropped, and "size = posts · colour = concern".
- Explore loads counts directly from `/api/clusters?include_hidden=true`; rendering does not require projection artifacts.

## 5. Explore (`/explore`)

- Groups from the 10,000-post run are loaded on arrival.
- **Group card:** shared by Explore and Live. Selecting a ball opens a rounded right-hand card with the full label and prominent concern, followed by divided Overview, Framing, cited evidence, Spread and Example posts. Show count, confidence, summary, explanation, relevance/techniques, spread statistics and up to five sample posts. Disclose mixed remainders and overlapping cross-batch repeated-text candidates. Fetch examples from the posts endpoint for groups without an analysis sample. Esc or the close control closes it.
- Put behaviour signal lines before Overview when present. Spread has a
  value/p50/p90 table with above-p90 marks, baseline bucket/fallback disclosure,
  duplicate-author count, largest repeated group, top domain/hashtag and missing
  image metadata shown as unknown. Unreliable burst timing is explicitly excluded
  from flags and priority. Do not equate aggregate anomalies with bot identity.
- If a saved model assessment predates the current stats, prompt or sample count, show a refresh
  warning in Overview and the page count. Keep stale groups visible until
  re-analysis. Tooltips include the first signal line when one exists.
- **Chat box:** one rounded input (22 px radius), nothing else. `＋` attaches files or articles (also drag and drop, or paste), Enter sends, Shift+Enter adds a line. Attached files show as small chips with a `×`.
- **Replies** are rounded bubbles with separate Summary, Content assessment, Evidence, Spread review and Related group sections. An unavailable spread judge never erases a successful content assessment. Each unavailable assessment is explicitly labelled with the reason.
- **Match:** show a closest-group link and highlight that ball. Explain that its concern describes the existing group, not the submission, and that matching alone does not establish coordination. No similarity debug numbers, fake coordinates or connection lines. An unmatched item is labelled as such; hidden-group matches are disclosed.
- **Layout:** from 1000 px wide the right-hand dock (420 px) holds the group details or chat thread above the input; the bubbles are centered in the remaining area. Selecting a group hides the thread without deleting it. Focusing the input, closing the card or pressing Esc restores the chat and clears the selection. On narrow screens the dock uses the available width at the bottom; details remain scrollable.
- **States:** loading, no artifacts (with the command to run), no groups above threshold, chat unavailable (reason replaces placeholder and input is disabled), request errors (bubble shows the server's reason). Show a spinner and status text while a chat request is running.

## 6. Live test (`/live`)

- The ball view starts a shared dataset-stream processor via `GET /api/live/processed`. New batches are embedded, grouped and assessed independently. Separate exact-text candidates spanning at least two batches and three accounts are assessed across the rolling window; they overlap batch groups and are not proof of coordination.
- The right-hand dock contains Super chat and live group details, with the thread and group details mutually exclusive. Chat matches against a snapshot of the retained live posts, including ungrouped posts. A related group is linked only while it is still visible; a match may expire while the request runs.
- The rolling window holds at most 50,000 eligible posts by default and evicts whole batches. At most 60 highest-priority groups are shown. Live clustering defaults to a minimum of 10 posts and 3 density samples; batch-local groups below the review threshold remain hidden.
- **Top centre:** retained count and last-batch speed, sized like the menu link. Speed colour interpolates from red at 0 to green at 20 posts/s and is not a performance target; missing speeds remain neutral. **Bottom:** processed post count and configured batch size in small text. An optional Details disclosure shows the dataset, eligible and grouped posts, source records, elapsed time, scanned records and group counts. Analysis failures remain visible even when details are closed. Batch timing covers reading through checkpointing; model startup is excluded, and the last batch metrics reset on restart.
- When no bubbles are visible, distinguish between no groups found and all groups being below the review threshold.
- Show the dataset and source record count resumed from in Details. A persisted cursor resumes after the last completed batch, but the rolling window is in memory only. If a completed run is restarted, explain why its previous groups are no longer displayed.
- The menu and Details identify the historical source. The card displays live batch statistics and examples. Errors are explicit; restart the server to retry after fixing the cause.

## 7. Accessibility and motion

- Text contrast at least 4.5:1. Colour is never the only signal: labels carry the score and the cards spell the level out.
- Everything works from the keyboard: balls and links are focusable; Enter or Space selects a ball, Esc closes details, the chat box takes Enter.
- `prefers-reduced-motion`: no blinking cursor, spinner rotation or transitions; the working status text remains visible.

## 8. Implementation rules

- No framework, no build step, no third-party scripts. ES modules only.
- All text from the API or the models is written with `textContent`. Never build HTML from data.
- Keep every state: loading, empty, error, unavailable.
