"""
manage_api_keys.py  —  Inject or scrub the OpenWeather API key across the workspace.
======================================================================================

Usage:
    python manage_api_keys.py inject <API_KEY>    # Replace placeholder with real key
    python manage_api_keys.py scrub               # Replace real key with placeholder

The script never stores the API key in any persistent file.

Primary files are checked first for speed; a full workspace walk catches anything
that was missed.
"""
import os
import sys

PLACEHOLDER = "YOUR_OPENWEATHER_API_KEY"
SCRIPT_NAMES = {"manage_api_keys.py"}

# Files to check first before doing a full recursive walk
PRIMARY_FILES = [
    "backend/src/main/resources/application.properties",
    "backend/src/main/java/com/weatherapp/backend/service/WeatherService.java",
    "docker-compose.yml",
    "k8s/backend.yaml",
]

# Directories to skip during the full workspace walk
EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", ".venv", ".idea",
    "target", "pretrained", "build", "dist", "__pycache__",
}

# File extensions to check during the full walk (text-based only)
INCLUDE_EXTENSIONS = {
    ".properties", ".yaml", ".yml", ".java", ".py",
    ".sh", ".ps1", ".xml", ".json", ".env", ".txt", ".md",
}


def _replace_in_file(filepath: str, src: str, dst: str) -> bool:
    """Replace all occurrences of *src* with *dst* in *filepath*."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        if src not in content:
            return False
        label = "placeholder" if src == PLACEHOLDER else "API key"
        print(f"  [{label} found] {filepath}")
        new_content = content.replace(src, dst)
        with open(filepath, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_content)
        return True
    except Exception as exc:
        print(f"  [ERROR] {filepath}: {exc}")
        return False


def _run(mode: str, api_key: str) -> int:
    """
    Run a single-pass inject or scrub across the workspace.

    mode    : 'inject'  →  replace PLACEHOLDER with api_key
              'scrub'   →  replace api_key with PLACEHOLDER
    api_key : the actual API key (required for both directions)
    """
    if not api_key:
        print("ERROR: API key must not be empty.")
        return 1

    if mode == "inject":
        src, dst = PLACEHOLDER, api_key
        action_label = "Injecting API key"
    elif mode == "scrub":
        src, dst = api_key, PLACEHOLDER
        action_label = "Scrubbing API key"
    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'inject' or 'scrub'.")
        return 1

    print(f"\n{action_label} — checking files...\n")
    modified = set()

    # 1. Fast-path: primary files
    print("--- Primary files ---")
    base = os.getcwd()
    for rel in PRIMARY_FILES:
        fpath = os.path.join(base, rel.replace("/", os.sep))
        if os.path.isfile(fpath):
            if _replace_in_file(fpath, src, dst):
                modified.add(fpath)
        else:
            print(f"  [not found] {rel}")

    # 2. Full workspace walk for anything else
    print("\n--- Full workspace scan ---")
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if name in SCRIPT_NAMES:
                continue
            _, ext = os.path.splitext(name)
            if ext not in INCLUDE_EXTENSIONS:
                continue
            fpath = os.path.join(root, name)
            if fpath in modified:
                continue
            if _replace_in_file(fpath, src, dst):
                modified.add(fpath)

    print(f"\nDone. Modified {len(modified)} file(s).")
    return 0


def _prompt_for_key(prompt_text: str) -> str:
    """Prompt the user interactively for the API key (no echo)."""
    import getpass
    try:
        key = getpass.getpass(prompt_text)
    except Exception:
        key = input(prompt_text)
    return key.strip()


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    mode = args[0].lower()

    if mode == "inject":
        if len(args) >= 2:
            api_key = args[1]
        else:
            api_key = _prompt_for_key("Enter OpenWeather API Key to inject: ")
        if not api_key:
            print("ERROR: No API key provided. Aborting.")
            sys.exit(1)
        sys.exit(_run("inject", api_key))

    elif mode == "scrub":
        if len(args) >= 2:
            api_key = args[1]
        else:
            api_key = _prompt_for_key(
                "Enter the API key to scrub (so we know what to replace): "
            )
        if not api_key:
            print("ERROR: No API key provided. Aborting.")
            sys.exit(1)
        sys.exit(_run("scrub", api_key))

    else:
        print(f"ERROR: Unknown command '{mode}'. Use 'inject' or 'scrub'.")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
