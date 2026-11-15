import argparse
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

# Actions to ignore (e.g., local composite actions or internal workflows)
IGNORED_PREFIXES = ("./", "../")

API_CACHE = {}

def github_api_get(url):
    if url in API_CACHE:
        return API_CACHE[url]

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Autonomous-Action-Updater",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
        API_CACHE[url] = data
        return data

def get_latest_release_and_sha(action_name):
    """Fetches the latest release tag and resolves its underlying commit SHA."""
    parts = action_name.split("/")
    if len(parts) < 2:
        return None, None
    repo = f"{parts[0]}/{parts[1]}"

    try:
        release = github_api_get(f"https://github.com{repo}/releases/latest")
        latest_tag = release.get("tag_name")
        if not latest_tag:
            return None, None

        ref_data = github_api_get(f"https://github.com{repo}/git/ref/tags/{latest_tag}")
        obj = ref_data["object"]

        # Dereference annotated tags to get the actual commit SHA
        if obj["type"] == "tag":
            tag_obj = github_api_get(obj["url"])
            obj = tag_obj["object"]

        return latest_tag, obj["sha"]
    except urllib.error.HTTPError as e:
        print(f"  [Skip] {repo}: HTTP {e.code} (No public releases or rate-limited)")
        return None, None
    except Exception as e:
        print(f"  [Warning] {repo}: {e}")
        return None, None

def process_workflows(workflows_dir=".github/workflows", dry_run=False):
    path = pathlib.Path(workflows_dir)
    if not path.exists():
        print(f"Directory not found: {workflows_dir}")
        return 0

    pattern = re.compile(
        r"(uses:\s*['\"]?)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+)@"
        r"([0-9a-fA-F]{40}|[a-zA-Z0-9_.-]+)"
        r"(['\"]?(?:\s*#\s*(?:tag=)?([a-zA-Z0-9_.-]+))?)"
    )

    files_modified = 0

    for file_path in sorted(list(path.glob("*.yml")) + list(path.glob("*.yaml"))):
        original_content = file_path.read_text(encoding="utf-8")

        def replace_match(match):
            prefix, action_name, current_ref, suffix, comment_tag = match.groups()

            if action_name.startswith(IGNORED_PREFIXES):
                return match.group(0)

            latest_tag, latest_sha = get_latest_release_and_sha(action_name)
            if not latest_tag or not latest_sha:
                return match.group(0)

            if current_ref.lower() == latest_sha.lower() and comment_tag == latest_tag:
                return match.group(0)

            print(f"  [{file_path.name}] {action_name}: {current_ref[:12]} -> {latest_sha[:12]} ({latest_tag})")
            # Preserve any trailing quote cleanly before adding the version comment
            quote = suffix[0] if suffix and suffix[0] in ("'", '"') else ""
            return f"{prefix}{action_name}@{latest_sha}{quote} # {latest_tag}"

        updated_content = pattern.sub(replace_match, original_content)

        if updated_content != original_content:
            files_modified += 1
            if not dry_run:
                file_path.write_text(updated_content, encoding="utf-8")

    mode_label = "would be updated (dry-run)" if dry_run else "updated"
    print(f"\nCompleted: {files_modified} workflow file(s) {mode_label}.")
    return files_modified

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically pin GitHub Actions to latest Node 24+ SHAs.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files.")
    args = parser.parse_args()
    process_workflows(dry_run=args.dry_run)
  
