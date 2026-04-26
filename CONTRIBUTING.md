# Contributing Guide & Git Workflow

> **All contributors (human and AI agents) must read this file before making any git operations.**

---

## Branching Strategy

We use a **Git Flow–inspired** model optimized for a research project with multiple contributors.

```
main ─────────────────────────────────── production / paper submission
  └── develop ──────────────────────────── integration branch
        ├── feature/dinov2-extractor       feature work
        ├── feature/projection-mlp
        ├── exp/week1-poc                  experiment branches
        ├── exp/week2-training
        ├── exp/ablation-proj-type
        └── hotfix/fix-sampling-bug        critical fixes
```

### Branch Types

| Branch | Naming Pattern | Purpose | Merge Target | Direct Push? |
|---|---|---|---|---|
| `main` | — | Paper-ready code only | — | ❌ **Never** |
| `develop` | — | Latest integrated work | `main` via PR | ❌ Only via PR |
| `feature/*` | `feature/<short-desc>` | New modules, utilities | `develop` | ✅ (author only) |
| `exp/*` | `exp/<experiment-name>` | Experiment code + results | `develop` | ✅ (author only) |
| `hotfix/*` | `hotfix/<desc>` | Critical bug fixes | `develop` and `main` | ✅ (author only) |
| `docs/*` | `docs/<desc>` | Documentation only | `develop` | ✅ |

---

## Workflow for Human Contributors

### Starting new work
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name   # or exp/your-experiment
```

### Committing
Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

**Types:**
- `feat` — new feature or model component
- `exp` — experiment code, results, analysis
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code restructuring (no behavior change)
- `test` — adding/fixing tests
- `chore` — build scripts, configs, CI

**Examples:**
```bash
git commit -m "feat(extractor): add DINOv2-B feature extractor with bilinear sampling"
git commit -m "exp(week1): add zero-shot descriptor quality results on MegaDepth-1500"
git commit -m "fix(sampling): correct grid_sample normalization for non-square images"
git commit -m "docs(plan): update disk budget after MegaDepth subset decision"
```

### Opening a Pull Request
- Always target `develop` (never `main` directly)
- PR title = commit style: `feat(matcher): integrate DINOv2 into LightGlue pipeline`
- Assign at least one reviewer
- Include experiment results in PR description if applicable

### Merging to `main`
Only merge `develop → main` at **Go/No-Go checkpoint milestones** (see `plan.md §14`):
- After Phase 1 (Day 7): Go/No-Go gate passed
- After Phase 2 (Day 15): Core training complete
- After Phase 3 (Day 20): Full evaluation done
- Final: Paper ready

---

## What Goes in Git vs. What Doesn't

### ✅ Commit to git
- All Python source files (`src/`, `configs/`, `scripts/`)
- Notebooks (`Exp1.ipynb`, etc.)
- Documentation (`docs/`, `README.md`, `plan.md`)
- Config YAML files
- Small result JSON/CSV files (< 1 MB per file)

### ❌ Never commit (enforced by `.gitignore`)
- `data/` — datasets (too large, not reproducible via git)
- `checkpoints/` — model weights
- `cache/` — HDF5 feature caches
- `.venv/` — virtual environment
- `__pycache__/`, `*.pyc`
- Wandb run files (`wandb/`)
- Tensorboard logs (`runs/`)

---

## .gitignore Additions

Add to `.gitignore`:
```
# Project-specific
data/
checkpoints/
cache/
wandb/
runs/
*.h5
*.hdf5
experiments/**/features/
experiments/**/*.pth

# Python
.venv/
__pycache__/
*.pyc
*.egg-info/
.ipynb_checkpoints/
```

---

## Code Review Checklist

Before approving any PR:
- [ ] Code runs without error on RTX 3060 (12 GB VRAM)
- [ ] VRAM usage is within budget (see `plan.md §9`)
- [ ] No hardcoded absolute paths (use `pathlib.Path` and config files)
- [ ] Experiment results are logged to `docs/changelog.md`
- [ ] New modules have docstrings on public functions
- [ ] `sessionlogs.md` updated if AI agent made the changes

---

## Conflict Resolution

- `plan.md` conflicts → discuss in PR comments before resolving
- Experiment result conflicts → keep both sets, rename files with `_v2` suffix
- Never force-push to `develop` or `main`

---

## Emergency / Hotfix Procedure

```bash
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix-name
# ... make fix ...
git commit -m "fix(critical): <description>"
git push origin hotfix/critical-fix-name
# Open PR targeting BOTH develop and main
```
