#!/usr/bin/env bash
# One-shot rename: "ramp" -> "$1" across this repo. Run from the repo root.
# Caveats:
#   - Substring replace, so words containing "ramp" (e.g. "trampoline") would also change. None known in this repo.
#   - .venv / caches are skipped; re-run `uv sync` after.
#   - Repo-root directory is not renamed by this script (a process can't rename its own cwd safely); see final hint.

set -euo pipefail

if [ $# -ne 1 ] || [ -z "$1" ]; then
  echo "usage: $0 <new-name>" >&2
  exit 2
fi

OLD=ramp
NEW=$1

[ -d .git ] || { echo "Run from the repo root." >&2; exit 1; }

# 1) Replace content in tracked files that contain OLD.
git ls-files -z | while IFS= read -r -d '' f; do
  [ -L "$f" ] && continue  # skip symlinks; the target file is edited when iterated
  if grep -Iq "$OLD" "$f" 2>/dev/null; then
    sed -i '' "s/$OLD/$NEW/g" "$f"
  fi
done

# 2) Rename paths, deepest first so renames don't invalidate parent paths mid-loop.
find . \
  \( -name .git -o -name .venv -o -name node_modules -o -name target \
     -o -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ \) -prune -o \
  -name "*${OLD}*" -print \
  | awk -F/ '{print NF "\t" $0}' | sort -rn | cut -f2- \
  | while IFS= read -r p; do
      np="$(dirname "$p")/$(basename "$p" | sed "s/$OLD/$NEW/g")"
      mv "$p" "$np"
    done

root="$(basename "$PWD")"
new_root="$(printf '%s' "$root" | sed "s/$OLD/$NEW/g")"
if [ "$root" != "$new_root" ]; then
  echo "Done. To rename the repo root directory itself, run from one level up:"
  echo "  cd .. && mv $root $new_root"
else
  echo "Done."
fi
