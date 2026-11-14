# UI spec: online narrative mapping

Visual and interaction spec for the website. The data and endpoints come from section 9 of the [project spec](./Coordinated-behavior%20detector%20project%20spec.md). This doc decides how it looks and feels. Terse on purpose, for the team and for coding agents.

## 1. Direction

- **Feel:** super minimalist. Group balls fill the page. No banners, no toolbars, no boxed controls, no panels that are always open.
- **Three pages:** a menu that explains the demo, **Explore** (map plus chat) and **Live test**.
- **Unit of display:** one ball per group of similar posts. Ball area represents post count; colour represents group concern.

**Copy rules (honesty is judged)**

- Ball colour means **group concern**, not proven maliciousness or falsity. Uploaded material has an independent content category and concern.
- Review priority is a triage score from 0 to 100, never a probability.
- Describe what was observed, not who is guilty. Never label an account.
- No 2D/3D controls, projected points, dimensional copy, rotation or semantic distances. Ball positions are only a layout.
- Propaganda-style persuasion can be identified from observable rhetoric without establishing truth, intent or coordination. Do not equate any political viewpoint with propaganda.
- The live test is a replay of stored posts, and the page says so.

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

**Typography:** JetBrains Mono, self-hosted woff2 in `web/fonts/` (400 and 600). Title 22 px, body 15 px, small 13 px. No CDN assets, so the demo works offline.

## 3. Menu (`/`)

Left-aligned column, at most 620 px wide: title, tagline, four numbered sentences on how it works, then the two modes as plain text rows with an arrow (Explore groups, Live test), one status line from `/api/meta` (posts, groups, shown, chat model), and the honesty notes. No buttons.

## 4. Group balls (shared by Explore and Live test, `map.js`)

- One circular, focusable element per non-discarded group. Radius is proportional to the square root of post count, so area is proportional to count.
- A deterministic non-overlapping pack uses final group sizes. Live balls grow in stable slots; positions have no semantic meaning.
- Fill and outline use the group's concern colour. Each ball contains the count and, when large enough, a wrapped label. Full label, count, concern and review priority are available through the accessible name and tooltip.
- **Discarded groups** (assessed, below `MIN_REVIEW_PRIORITY`) and ungrouped noise are not drawn. Failed and pending groups remain visible.
- **Selection:** click or keyboard-activate a ball. Its outline brightens and other balls dim. Click empty space or press Esc to clear.
- **Corner text:** menu link, total posts, groups shown, groups dropped, and "size = posts · colour = concern".
- Explore loads counts directly from `/api/clusters?include_hidden=true`; rendering does not require projection artifacts.

## 5. Explore (`/explore`)

- Groups from the 10,000-post run are loaded on arrival.
- **Group card:** shared by Explore and Live. Selecting a ball opens a rounded right-hand card with thin dividing lines between Overview, Framing, Spread and Example posts. Show count, priority, concern/confidence, summary, explanation, relevance/techniques, spread statistics and up to five sample posts. Fetch examples from the posts endpoint for groups without an analysis sample. Esc or the close control closes it.
- Put behaviour signal lines before Overview when present. Spread has a
  value/p50/p90 table with above-p90 marks, baseline bucket/fallback disclosure,
  duplicate-author count, largest repeated group, top domain/hashtag and missing
  image metadata shown as unknown. Unreliable burst timing is explicitly excluded
  from flags and priority. Do not equate aggregate anomalies with bot identity.
- If a saved model assessment predates the current stats, prompt or sample count, show a refresh
  warning in Overview and the page count. Keep stale groups visible until
  re-analysis. Tooltips include the first signal line when one exists.
- **Chat box:** one rounded input (22 px radius), nothing else. `＋` attaches files or articles (also drag and drop, or paste), Enter sends, Shift+Enter adds a line. Attached files show as small chips with a `×`.
- **Replies** are rounded bubbles. Lead with the original content category (including propaganda-style persuasion where supported), content concern/confidence, techniques and quoted or visual evidence. A thin divider separates contextual spread concern. An unavailable spread judge never erases a successful content assessment. Each unavailable assessment is explicitly labelled with the reason.
- **Match:** show a closest-group link and highlight that ball. Explain that its review priority describes the existing group, not the submission, and that matching alone does not establish coordination. No similarity debug numbers, fake coordinates or connection lines. An unmatched item is labelled as such; hidden-group matches are disclosed.
- **Layout:** from 1000 px wide the chat docks to the bottom left (420 px), while selected group details open on the right. Below that chat sits at the bottom centre.
- **States:** loading, no artifacts (with the command to run), no groups above threshold, chat unavailable (reason replaces placeholder and input is disabled), request errors (bubble shows the server's reason). Show a spinner and status text while a chat request is running.

## 6. Live test (`/live`)

- The same ball view, empty at first. `GET /api/live/stream` delivers batches (120 posts per second by default).
- A ball appears at its first arriving post and grows in area with count.
- **Bottom left:** the six clouds that matter most so far, re-sorted as posts arrive: review priority, label, posts collected.
- **Bottom right:** posts arrived out of the total, the replay speed and the timestamp of the latest post. When the replay ends: "replay complete" and a text link to replay.
- **Corner text** states that this is simulated traffic replayed through the server and that cloud membership comes from the offline run.
- The card shows the full offline group's statistics and examples, not a new inference about replay arrivals. If the stream breaks, the page says so and does not silently restart.

## 7. Accessibility and motion

- Text contrast at least 4.5:1. Colour is never the only signal: labels carry the score and the cards spell the level out.
- Everything works from the keyboard: balls and links are focusable; Enter or Space selects a ball, Esc closes details, the chat box takes Enter.
- `prefers-reduced-motion`: no blinking cursor, spinner rotation or transitions; the working status text remains visible.

## 8. Implementation rules

- No framework, no build step, no third-party scripts. ES modules only.
- All text from the API or the models is written with `textContent`. Never build HTML from data.
- Keep every state: loading, empty, error, unavailable.
