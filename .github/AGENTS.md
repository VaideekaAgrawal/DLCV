# AI Agent Git Rules

> **MANDATORY READING** — Every AI agent (GitHub Copilot, Cursor, Claude, etc.) MUST read this file in full before performing any git operation in this repository.

---

## Core Rules (Non-Negotiable)

### 1. NEVER push directly to `main` or `develop`
```
❌ git push origin main
❌ git push origin develop
```
Always work on a feature or experiment branch.

### 2. ALWAYS create a new branch for new work
```bash
# For new features / modules:
git checkout develop && git pull origin develop
git checkout -b feature/<short-description>

# For experiment code:
git checkout develop && git pull origin develop
git checkout -b exp/<experiment-name>
```

### 3. ALWAYS update docs before committing
Before any `git commit`, you MUST update:
- `docs/changelog.md` — describe code changes (what was added/modified/deleted)
- `docs/sessionlogs.md` — describe the AI session (date, goal, what was done)

### 4. ALWAYS use Conventional Commits
```
feat(scope): description    ← new capability
exp(scope): description     ← experiment code
fix(scope): description     ← bug fix
docs(scope): description    ← documentation
refactor(scope): description ← restructuring
```

### 5. NEVER commit data files, model weights, or caches
```
❌ data/
❌ checkpoints/
❌ cache/
❌ *.h5, *.hdf5, *.pth (except tiny test fixtures < 10 KB)
```

### 6. NEVER force-push (`--force` or `-f`)
```
❌ git push --force
❌ git push -f
```

### 7. NEVER amend commits that have been pushed
```
❌ git commit --amend   (if already pushed)
❌ git rebase -i HEAD~N (if commits are already on remote)
```

### 8. NEVER bypass pre-commit hooks
```
❌ git commit --no-verify
```

### 9. ALWAYS run a sanity check before committing
```bash
# Verify no large files are staged
git diff --cached --stat | awk '{print $1, $NF}' | sort -rn | head -10

# Verify no secrets or credentials
git diff --cached | grep -i "password\|secret\|api_key\|token" && echo "STOP: credentials detected"

# Verify the branch is NOT main or develop
git branch --show-current | grep -E "^(main|develop)$" && echo "STOP: on protected branch"
```

---

## Permitted Git Operations (Agent Can Do Without Confirmation)

| Operation | Condition |
|---|---|
| `git status` | Always |
| `git log` | Always |
| `git diff` | Always |
| `git add <file>` | On feature/* or exp/* branches only |
| `git commit -m "..."` | On feature/* or exp/* branches only, after updating docs |
| `git push origin <branch>` | On feature/* or exp/* branches only |
| `git checkout -b <branch>` | Following naming conventions above |
| `git pull origin develop` | To sync with develop |

## Operations That Require Human Confirmation

| Operation | Why |
|---|---|
| `git push origin develop` | Protected branch |
| `git push origin main` | Protected branch |
| `git merge` | Risk of conflicts |
| `git rebase` | Rewrites history |
| `git reset --hard` | Destructive |
| `git clean -fd` | Destructive |
| Opening/merging a PR | Human decision |
| Tagging a release | Human decision |

---

## Agent Commit Template

When writing a commit message, use this template:

```
<type>(<scope>): <50-char summary>

Changes made:
- <bullet 1>
- <bullet 2>

Files modified:
- src/xxx.py
- docs/changelog.md
- docs/sessionlogs.md

Session: <date YYYY-MM-DD>
Agent: <agent name>
```

---

## Branch Lifecycle

```
1. Create branch from develop
   git checkout develop && git pull && git checkout -b exp/my-experiment

2. Do work, update docs/changelog.md and docs/sessionlogs.md

3. Stage and commit
   git add src/ configs/ docs/
   git commit -m "exp(week1): ..."

4. Push to remote
   git push origin exp/my-experiment

5. ← STOP HERE. Human opens PR. Human merges.
```

---

## File Ownership Matrix

| Path | Agent may create | Agent may edit | Agent may delete |
|---|---|---|---|
| `src/` | ✅ | ✅ | ✅ (with changelog entry) |
| `configs/` | ✅ | ✅ | ✅ |
| `scripts/` | ✅ | ✅ | ✅ |
| `docs/sessionlogs.md` | ✅ | ✅ | ❌ |
| `docs/changelog.md` | ✅ | ✅ | ❌ |
| `plan.md` | ❌ | ✅ (append only, flag changes) | ❌ |
| `README.md` | ❌ | ✅ | ❌ |
| `CONTRIBUTING.md` | ❌ | ❌ (human only) | ❌ |
| `.github/AGENTS.md` | ❌ | ❌ (human only) | ❌ |
| `Exp*.ipynb` | ✅ | ✅ | ❌ |
| `data/` | ❌ | ❌ | ❌ |
| `checkpoints/` | ❌ | ❌ | ❌ |
| `glue-factory/` | ❌ | ❌ (fork first) | ❌ |
| `LightGlue/` | ❌ | ❌ (fork first) | ❌ |

---

## Emergency Stop

If at any point during a git operation something seems wrong (unexpected diff, unexpected branch, file size > 50 MB staged), **STOP IMMEDIATELY** and report to the human user before continuing.
