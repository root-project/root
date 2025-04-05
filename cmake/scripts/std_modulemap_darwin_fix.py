#!/usr/bin/env python3
import os
import re
import subprocess
import sys


def get_sdk_path():
    result = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Could not resolve SDK path")
    return os.path.realpath(result.stdout.strip())


def remove_ctype_module(content):
    # Break cyclic module dependencies
    # See: https://github.com/root-project/root/commit/8045591a17125b49c1007787c586868dea764479
    pattern = re.compile(r"module\s+std_ctype_h\s+\[system\]\s*\{.*?\}", re.DOTALL)
    return pattern.sub("", content)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: std_modulemap_darwin_fix.py <output_path>")
    output_path = sys.argv[1]
    sdk = get_sdk_path()
    cpp_modulemap = os.path.join(sdk, "usr/include/c++/v1/module.modulemap")

    if not os.path.exists(cpp_modulemap):
        raise FileNotFoundError(f"Cannot find libc++ modulemap at {cpp_modulemap}")

    with open(cpp_modulemap, "r") as f:
        original_content = f.read()

    cleaned_content = remove_ctype_module(original_content)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("// Auto-generated flat modulemap for Cling\n")
        f.write("module std [system] {\n")
        f.write("export *\n\n")
        f.write("// Entire modulemap wrapped\n")
        f.write(cleaned_content)
        f.write("\n}\n")


if __name__ == "__main__":
    main()
