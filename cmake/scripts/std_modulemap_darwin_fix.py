#!/usr/bin/env python3
import os
import re
import sys


def resolve_cyclic_module_dependencies(content):
    for h in ["ctype.h", "stdint.h"]:
        content = re.sub(f'  header "{h}"', f'  textual header "{h}"', content)
    return content


def main():
    if len(sys.argv) < 3:
        raise ValueError("Usage: std_modulemap_darwin_fix.py <sdk_path> <output_path> [<libcxx_include_dir>]")
    sdk_path = sys.argv[1]
    output_path = sys.argv[2]
    # The libc++ include directory of the compiler, if CMake found one in the
    # implicit include directories. It takes precedence over the libc++ in the
    # SDK, because it is the one that cling uses at runtime. This also covers
    # toolchains whose SDK is stripped of libc++ (conda-forge, Nix): there, the
    # compiler ships its own libc++ next to itself, which clang searches before
    # the SDK, so it is in the implicit include directories.
    libcxx_include_dir = sys.argv[3] if len(sys.argv) > 3 else None
    cpp_modulemap = os.path.join(sdk_path, "usr/include/c++/v1/module.modulemap")
    if libcxx_include_dir and os.path.exists(os.path.join(libcxx_include_dir, "module.modulemap")):
        cpp_modulemap = os.path.join(libcxx_include_dir, "module.modulemap")

    if not os.path.exists(cpp_modulemap):
        raise FileNotFoundError(f"Cannot find libc++ modulemap at {cpp_modulemap}")

    with open(cpp_modulemap, "r") as f:
        original_content = f.read()

    cleaned_content = resolve_cyclic_module_dependencies(original_content)

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
