#!/bin/bash
# Shared helper for enforcing --run-dir, RUN.md existence, and commands.sh logging
# (SafeMergeBench Implementation Spec v2, Section 1.4).

# Extracts --run-dir <DIR> (or --run-dir=<DIR>) from "$@" into RUN_DIR,
# leaving all remaining arguments in REMAINING_ARGS array.
extract_run_dir_arg() {
  RUN_DIR=""
  REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --run-dir requires a directory argument." >&2
          return 2
        fi
        RUN_DIR="$2"
        shift 2
        ;;
      --run-dir=*)
        RUN_DIR="${1#--run-dir=}"
        shift
        ;;
      *)
        REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: --run-dir <RUN_DIR> is required (see Implementation Spec v2, Section 1)." >&2
    echo "Create one first with: python3 safemerge/new_run.py --name <name> --why \"...\" --what \"...\"" >&2
    return 2
  fi
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: --run-dir does not exist: $RUN_DIR" >&2
    return 2
  fi
  if [[ ! -f "$RUN_DIR/RUN.md" ]]; then
    echo "ERROR: $RUN_DIR/RUN.md not found. Refusing to run without a recorded RUN.md." >&2
    return 2
  fi
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  mkdir -p "$RUN_DIR/config" "$RUN_DIR/logs" "$RUN_DIR/outputs"
  return 0
}

# Appends an executable command line with timestamp comment to $RUN_DIR/commands.sh.
record_run_command() {
  local run_dir="$1"
  shift
  local cmd_file="$run_dir/commands.sh"
  if [[ ! -f "$cmd_file" ]]; then
    printf '#!/bin/bash\nset -euo pipefail\n\n' > "$cmd_file"
    chmod +x "$cmd_file"
  fi
  local ts
  ts="$(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST')"
  {
    printf '# %s (cwd=%s)\n' "$ts" "$(pwd)"
    printf '%q ' "$@"
    printf '\n'
  } >> "$cmd_file"
}
