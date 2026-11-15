import json
import os
import pathlib
import re
import urllib.request

# Expanded map of GitHub Actions to their Node.js 24 compatible major versions
ACTION_UPDATES = {
    # Core Actions
    "actions/checkout": "v6",
    "actions/download-artifact": "v8",
    "actions/upload-artifact": "v7",
    "actions/setup-node": "v7",
    "actions/setup-python": "v6",
    "actions/cache": "v5",
    "actions/github-script": "v8",
    "actions/attest-build-provenance": "v3",
    "actions/configure-pages": "v5",
    "actions/deploy-pages": "v4",
    # Security & CodeQL
    "github/codeql-action/init": "v4",
    "github/codeql-action/analyze": "v4",
    "github/codeql-action/autobuild": "v4",
    "github/codeql-action/upload-sarif": "v4",
    # Python, PyPI & Build Ecosystem (Common in ROOT / wheel builds)
    "pypa/gh-action-pypi-publish": "release/v1",
    "pypa/cibuildwheel": "v3.0.0",
    "docker/setup-buildx-action": "v4",
    "docker/setup-qemu-action": "v4",
    "docker/build-push-action": "v7",
    "docker/login-action": "v4",
    "softprops/action-gh-release": "v2",
    "codecov/codecov-action": "v5",
}

SHA_CACHE = {}

def resolve_tag_to_sha(action_name, tag):
    """Fetches the commit SHA for a given repository tag via the GitHub API."""
    # Extract base 'owner/repo' even if action has a subpath (e.g., github/codeql-action/init)
    parts = action_name.split("/")
    repo = f"{parts[0]}/{parts[1]}"
    cache_key = f"{repo}@{tag}"

    if cache_key in SHA_CACHE:
        return SHA_CACHE[cache_key]

    url = f"https://github.com{repo}/git/ref/tags/{tag}"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Action-Updater-Script"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            obj = data["object"]
            # If it's an annotated tag, dereference it to get the underlying commit SHA
            if obj["type"] == "tag":
                tag_req = urllib.request.Request(obj["url"], headers=headers)
                with urllib.request.urlopen(tag_req, timeout=10) as tag_res:
                    obj = json.loads(tag_res.read().decode())["object"]
            sha = obj["sha"]
            SHA_CACHE[cache_key] = sha
            return sha
    except Exception as e:
        print(f"  [Warning] Could not resolve SHA for {repo}@{tag}: {e}")
        return None

def update_workflow_files(workflows_dir=".github/workflows"):
    path = pathlib.Path(workflows_dir)
    if not path.exists():
        print(f"Directory not found: {workflows_dir}")
        return

    # Matches both standard tags (@v4) and 40-char pinned SHAs (@abcdef... # v4)
    pattern = re.compile(
        r"(uses:\s*['\"]?)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+)@"
        r"([0-9a-fA-F]{40}|v\d+(?:\.\d+\.\d+)?|release/v\d+)"
        r"(['\"]?(?:\s*#\s*(?:tag=)?(v\d+(?:\.\d+\.\d+)?|release/v\d+))?)"
    )

    files_modified = 0

    for file_path in list(path.glob("*.yml")) + list(path.glob("*.yaml")):
        original_content = file_path.read_text(encoding="utf-8")

        def replace_action(match):
            prefix, action_name, ref, suffix, comment_tag = match.groups()
            if action_name not in ACTION_UPDATES:
                return match.group(0)

            target_tag = ACTION_UPDATES[action_name]
            is_sha = len(ref) == 40 and all(c in "0123456789abcdefABCDEF" for c in ref)

            if is_sha:
                target_sha = resolve_tag_to_sha(action_name, target_tag)
                if target_sha and ref.lower() != target_sha.lower():
                    print(f"  [{file_path.name}] {action_name}: {ref[:7]} -> {target_sha[:7]} ({target_tag})")
                    return f"{prefix}{action_name}@{target_sha} # {target_tag}"
            else:
                if ref != target_tag:
                    print(f"  [{file_path.name}] {action_name}: {ref} -> {target_tag}")
                    return f"{prefix}{action_name}@{target_tag}{suffix}"

            return match.group(0)

        updated_content = pattern.sub(replace_action, original_content)

        if updated_content != original_content:
            file_path.write_text(updated_content, encoding="utf-8")
            files_modified += 1

    print(f"\nDone! Updated {files_modified} workflow file(s).")

if __name__ == "__main__":
    update_workflow_files()
    
