# MNH26SVR
Neuromorphic singing voice recognition

NeuroTUM Hackathon 2026, Innatera × Sofilab challenge. **Goal:** you sing any song, in any key, at any tempo, and a small recurrent spiking neural network on the Innatera Pulsar chip plays a fitting violin accompaniment in real time.

## Layout
```
MNH26SVR/
├── README.md          ← you are here
├── scripts/           ← helper scripts (paper downloader, …); code goes in the repo root
├── jan/               ← Jan's code: scripts/ (analysis & experiments), prep/ (data prep),
│                        singalong/ (SNN app on Innatera's Talamo pipeline: devcontainer, run.sh, train/sim checks),
│                        train/ (run visualiser) — see Hackathon/07 Experiments/2026-10-07 Chord SNN training.md
└── Hackathon/         ← the team Obsidian vault (all notes, decisions, research, songs)
```

## Quick start (new teammate)
1. **Get access** to `nc-fortiss/MNH26SVR` (ask Jan) and add your SSH key to GitHub.
2. **Clone:**
   ```bash
   git clone git@github.com:nc-fortiss/MNH26SVR.git
   cd MNH26SVR
   git config user.name "Firstname Lastname"     # your commits are your signature
   git config user.email "you@example.com"
   python3 scripts/download_papers.py            # fetch the reference PDFs (not in git)
   ```
3. **Open the vault:** install [Obsidian](https://obsidian.md), choose **Open folder as vault**, and select `MNH26SVR/Hackathon`. When asked, click **Trust author and enable plugins**. This turns on *Obsidian Git* (auto-sync, see below).
4. **Introduce yourself:** create your note in `03 Team/` from the *Person* template (see [Templates](#templates)). Pick a handle such as `@anna` and add yourself to the table in `03 Team/README.md`.
5. **Read in this order:** `Home.md` → `01 Project/Overview.md` → `02 Ideas/SNN Violin Accompaniment.md` → `09 Decisions/Decisions.md`.

## How the vault is organised
| Folder | What goes there |
|---|---|
| `Home.md` | Start page: current status and links to everything |
| `00 Inbox/` | Quick unsorted captures; sort them later |
| `01 Project/` | `Overview.md`: challenge, rules, deadlines, **decisions log** |
| `02 Ideas/` | One note per idea. Main idea: `SNN Violin Accompaniment` |
| `03 Team/` | One note per person, plus the handle table |
| `04 Meetings/` | `YYYY-MM-DD Topic.md`: notes, decisions, action items |
| `05 Tasks/` | `Tasks.md`: the single shared task list |
| `06 Research/` | Hardware/SDK notes, `Lit - …` literature surveys, `Paper - …` summaries, `Papers.md` (paper library) |
| `07 Experiments/` | `YYYY-MM-DD Name.md`: one note per experiment (setup, results, commit) |
| `08 Songs/` | The challenge song set (audio, MIDI, scores). **Read-only reference data; check licences** |
| `09 Decisions/` | **One page per design decision** (`D01 …`–`D14 …`) plus the `Decisions.md` index |
| `Templates/` | Note templates: Decision, Experiment, Meeting, Idea, Person |
| `99 Archive/` | Dropped or finished material. **Move here, never delete** |
| `Attachments/` | Images, plots, PDFs. `Attachments/Papers/` is local only (not in git) |

## Working rules
Full version: `09 Decisions/D00 Collaboration conventions.md`.

- **Sign your work.**
  - Every new note gets `author: "[[Firstname Lastname]]"` in its frontmatter (the templates already include it).
  - Comment in someone else's note by appending a signed line: `- @anna 2026-10-06: I think …`. **Never rewrite someone else's text.**
- **One thing per note.** Each idea, paper, experiment or decision gets its own note, connected to the others with `[[wikilinks]]`.
- **Decisions:**
  - Every open design question has a page in `09 Decisions/`. Status goes 🟡 `proposed` → 🟢 `accepted` / 🔴 `rejected` → ⚪ `superseded`.
  - To weigh in, add a signed comment under *Discussion*.
  - The **owner** accepts a decision by filling in the *Decision* section (what, who, when, link to the meeting or experiment). The owner then updates the table in `Decisions.md` and the log in `Overview.md`.
  - New decision: use the *Decision* template, give it the next free number, and add a row to `Decisions.md`.
- **Tasks:** go in `05 Tasks/Tasks.md` as `- [ ] what — @handle 📅 YYYY-MM-DD`, with a link to the note it comes from. Tick them off with `[x]`.
- **Experiments:** use the *Experiment* template, link the decision it informs, and record the code commit (`git rev-parse --short HEAD`). Then add the result to the decision page.
- **Uncertain facts:** mark them with a `> [!question]` callout instead of guessing.
- **New notes:** link them from `Home.md` so others can find them.

## Templates
Settings → Core plugins → **Templates** is already set to the `Templates/` folder.

To make a note from a template:
1. Create the note (`Ctrl/Cmd + N`) and give it a title.
2. Open the command palette (`Ctrl/Cmd + P`), run **Templates: Insert template**, and pick one.

The template fills in the date and title.

## Syncing (Obsidian Git)
The *Obsidian Git* plugin is included and preconfigured:
- **Pulls** on startup and every 10 minutes.
- **Commits and pushes** your changes automatically 15 minutes after you stop editing.
- Manual sync: command palette → **Obsidian Git: Commit-and-sync** (or the git icon in the sidebar).
- Before editing busy notes (`Tasks.md`, `Decisions.md`, `Home.md`), pull first. That avoids merge conflicts.
- If a conflict happens: open the file, keep both versions (delete only the `<<<<<<<`, `=======` and `>>>>>>>` lines), save and sync again.

> **The plugin only syncs the vault (`Hackathon/`).** Code, `scripts/` and this README live outside it, so commit and push them yourself with the commands below.

Without the plugin, or for code, use the terminal:
```bash
git pull --rebase            # before you start
git add -A && git commit -m "what you did" && git push
```

## Papers
The literature behind each decision is in the paper library:
- **List:** `scripts/papers.tsv`, one row per paper with the decisions it informs.
- **Index:** `Hackathon/06 Research/Papers.md`.
- **Per decision:** each decision page ends with a "Papers" section.

The PDFs are not in git (copyright). Get them with:
```bash
python3 scripts/download_papers.py
```
To add a paper, add a row to `papers.tsv` and run the script again. It downloads the PDF and regenerates `Papers.md` and the decision sections. **Don't edit those generated sections by hand.**

## Using Claude in the vault
`Hackathon/CLAUDE.md` holds the vault rules for AI assistants: structure, signatures, and the decision workflow. Claude (Claude Code or the Claudian plugin) reads it automatically. Notes written by Claude are signed `author: "Claude (for [[Your Name]])"`.

## Don't commit
- **Secrets:** SDK tokens, Docker passwords, `.env` files. Share them in Discord, never in the vault. `.secrets/` and `*.token` are gitignored.
- Large data or model files over about 50 MB: keep them on the Sofilab Nextcloud and link them instead.
- Personal Obsidian layout (`.obsidian/workspace*.json`), which is already gitignored.
