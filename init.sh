#!/usr/bin/env bash
# Usage: source init.sh

# -------- Initialization Flag: BandPO_isInit --------
# If already "true" (true/1/yes/on), skip execution.
if [[ -n "${BandPO_isInit:-}" ]]; then
  flag_lc="${BandPO_isInit,,}"
  if [[ "$flag_lc" == "true" || "$flag_lc" == "1" || "$flag_lc" == "yes" || "$flag_lc" == "on" ]]; then
    echo "Detected BandPO_isInit=${BandPO_isInit}, already initialized. Skipping initialization."
    exit 0
  fi
fi

# -------- BandPODir --------
if [[ -n "${BandPODir:-}" ]]; then
  echo "BandPODir is already set to: $BandPODir"
else
  VAL="$(pwd)"
  echo "BandPODir is not set. Setting to current directory: $VAL"
  export BandPODir="$VAL"
  echo "export BandPODir=\"$VAL\"" >> "$HOME/.bashrc"
fi

# -------- BandPODir_LargeData --------
if [[ -n "${BandPODir_LargeData:-}" ]]; then
  echo "BandPODir_LargeData is already set to: $BandPODir_LargeData"
else
  local_data="$BandPODir/data"
  echo "BandPODir_LargeData is not set. Setting to: $local_data"
  export BandPODir_LargeData="$local_data"
  _persist_export "BandPODir_LargeData" "$local_data"
fi

# -------- TMPDIR / TEMP / TMP Unification Rule --------
# Target directory: use $HOME/tmp if empty
TMP_TARGET="$HOME/tmp"
TARGET="${TMP_TARGET:-"$HOME/tmp"}"
echo "Current: TMPDIR='${TMPDIR-}'  TEMP='${TEMP-}'  TMP='${TMP-}'"
# If any is empty (undefined or empty string) => unify; otherwise if they are not equal => unify
any_empty=false
for v in TMPDIR TEMP TMP; do
  if [[ ! -v $v || -z "${!v}" ]]; then
    any_empty=true
    break
  fi
done
need_unify=false
if $any_empty; then
  need_unify=true
else
  [[ "$TMPDIR" != "$TEMP" || "$TMPDIR" != "$TMP" ]] && need_unify=true
fi
if $need_unify; then
  echo "Unifying TMPDIR/TEMP/TMP => $TARGET"
  mkdir -p "$TARGET" 2>/dev/null
  export TMPDIR="$TARGET"
  export TEMP="$TARGET"
  export TMP="$TARGET"
  {
    echo "export TMPDIR=\"$TARGET\""
    echo "export TEMP=\"$TARGET\""
    echo "export TMP=\"$TARGET\""
  } >> "$HOME/.bashrc"
else
  echo "All three variables are set and consistent. No changes needed."
fi

# -------- Check and Set Hugging Face Username --------
if [[ -n "${HUGGING_FACE_USERNAME:-}" ]]; then
  echo "HUGGING_FACE_USERNAME is already set to: $HUGGING_FACE_USERNAME"
else
  # Input username in plain text
  read -r -p "Please enter Hugging Face Username (plain text): " hf_username
  if [[ -z "$hf_username" ]]; then
    echo "No username entered. Exiting."
    exit 1
  fi
  export HUGGING_FACE_USERNAME="$hf_username"
  # Write to bashrc
  echo "export HUGGING_FACE_USERNAME=\"$hf_username\"" >> "$HOME/.bashrc"
  echo "HUGGING_FACE_USERNAME saved."
fi

# -------- huggingface key --------
# If exists, prompt and skip (do not print token)
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" || -n "${HF_TOKEN:-}" ]]; then
  echo "Detected Hugging Face token:"
  [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && echo " - HUGGING_FACE_HUB_TOKEN is set"
  [[ -n "${HF_TOKEN:-}" ]] && echo " - HF_TOKEN is set"
else
  # If not set, prompt for input (hidden input)
  read -rsp "Token not found. Please enter Hugging Face token (input invisible): " token
  echo
  if [[ -z "$token" ]]; then
    echo "No token entered. Exiting."
    exit 1
  fi
  # Export to current process and write to ~/.bashrc (no deduplication, keeping it simple)
  export HUGGING_FACE_HUB_TOKEN="$token"
  export HF_TOKEN="$token"
  {
    echo
    echo "# added on $(date '+%Y-%m-%d %H:%M:%S')"
    echo "export HUGGING_FACE_HUB_TOKEN=\"$token\""
    echo "export HF_TOKEN=\"$token\""
  } >> "$HOME/.bashrc"
  echo "Written to ~/.bashrc: HUGGING_FACE_HUB_TOKEN / HF_TOKEN"
  echo "To take effect immediately in current session, run: . ~/.bashrc"
fi

# -------- wandb key --------
# If exists, prompt and skip (do not print token)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "Detected W&B API Key:"
  echo " - WANDB_API_KEY is set"
else
  # If not set, prompt for input (hidden input)
  read -rsp "W&B API Key not found. Please enter (input invisible): " token_wandb
  echo
  if [[ -z "$token_wandb" ]]; then
    echo "No W&B API Key entered. Exiting."
    exit 1
  fi
  # Export to current process and write to ~/.bashrc (no deduplication, keeping it simple)
  export WANDB_API_KEY="$token_wandb"
  {
    echo
    echo "# added on $(date '+%Y-%m-%d %H:%M:%S')"
    echo "export WANDB_API_KEY=\"$token_wandb\""
  } >> "$HOME/.bashrc"
  echo "Written to ~/.bashrc: WANDB_API_KEY"
  echo "To take effect immediately in current session, run: . ~/.bashrc"
fi
# Login and verification (non-intrusive: warning only on failure, no exit)
if command -v wandb >/dev/null 2>&1; then
  echo "Logging in and verifying W&B..."
  # If you have a private deployment, pre-export WANDB_BASE_URL=http://your-host:port
  extra=()
  [[ -n "${WANDB_BASE_URL:-}" ]] && extra=(--host "$WANDB_BASE_URL")

  # Ensure online mode, then login with existing API Key
  wandb online >/dev/null 2>&1 || true
  if wandb login "${extra[@]}" "$WANDB_API_KEY" >/dev/null 2>&1; then
    # New version has whoami; fallback to status for older versions
    if wandb whoami >/dev/null 2>&1; then
      echo "W&B Login Successful: $(wandb whoami || true)"
    else
      wandb status || true
      echo "W&B Login Successful (online mode enabled)."
    fi
  else
    echo "[Warning] W&B Login failed. Please check API Key/Network/Proxy or WANDB_BASE_URL."
  fi
else
  echo "[Info] 'wandb' command not found, skipping login verification. You can install it via: python3 -m pip install --upgrade wandb"
fi

# ---------------- Initialize Datasets and Base Model ----------------
bash $BandPODir/utils/init/init_datasets_and_models.sh

# ---------------- Write Initialization Flag to ~/.bashrc ----------------
export BandPO_isInit=True
echo 'export BandPO_isInit=True' >> "$HOME/.bashrc"
echo "Written to ~/.bashrc: export BandPO_isInit=True"

source ~/.bashrc
echo "Done."