import pathlib
import re

# Map of common GitHub Actions to their Node.js 24 compatible major versions
ACTION_UPDATES = {
    "actions/checkout": "v6",
    "actions/download-artifact": "v8",
    "actions/upload-artifact": "v7",
    "actions/setup-node": "v7",
    "actions/setup-python": "v6",
    "actions/cache": "v5",
    "github/codeql-action/init": "v4",
    "github/codeql-action/analyze": "v4",
    "github/codeql-action/autobuild": "v4",
}

def update_workflow_files(workflows_dir=".github/workflows"):
    path = pathlib.Path(workflows_dir)
    if not path.exists():
        print(f"Directory not found: {workflows_dir}")
        return

    # Matches 'uses: owner/repo@vX' or 'uses: owner/repo/subpath@vX' (ignores SHA hashes)
    pattern = re.compile(r"(uses:\s*['\"]?)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+)@(v\d+(?:\.\d+\.\d+)?)(['\"]?)")

    files_modified = 0

    for file_path in list(path.glob("*.yml")) + list(path.glob("*.yaml")):
        original_content = file_path.read_text(encoding="utf-8")
        
        def replace_tag(match):
            prefix, action_name, current_version, suffix = match.groups()
            if action_name in ACTION_UPDATES:
                target_version = ACTION_UPDATES[action_name]
                if current_version != target_version:
                    print(f"  [{file_path.name}] {action_name}: {current_version} -> {target_version}")
                    return f"{prefix}{action_name}@{target_version}{suffix}"
            return match.group(0)

        updated_content = pattern.sub(replace_tag, original_content)

        if updated_content != original_content:
            file_path.write_text(updated_content, encoding="utf-8")
            files_modified += 1

    print(f"\nDone! Updated {files_modified} workflow file(s).")

if __name__ == "__main__":
    update_workflow_files()
                  
