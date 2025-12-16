#!/usr/bin/env python3
"""
Script to check for unused library dependencies in ROOT shared libraries.
Uses 'ldd -u' to detect libraries that are linked but not actually used. 

Usage: python3 check_unused_dependencies.py <build_directory>
Example: python3 check_unused_dependencies.py build/
"""

import subprocess
import sys
import os
from pathlib import Path
from collections import defaultdict


def find_shared_libraries(build_dir):
    """Find all . so files in the build directory."""
    build_path = Path(build_dir)
    if not build_path.exists():
        print(f"Error: Build directory '{build_dir}' does not exist.")
        sys.exit(1)
    
    # Find all .so files
    so_files = list(build_path.rglob("*.so"))
    
    # Filter out symlinks, keep only real files
    so_files = [f for f in so_files if not f.is_symlink()]
    
    return sorted(so_files)


def check_unused_dependencies(so_file):
    """Run ldd -u on a shared library and return unused dependencies."""
    try:
        # Run ldd -u on the library
        result = subprocess. run(
            ['ldd', '-u', str(so_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse output - ldd -u shows "Unused direct dependencies:" followed by library paths
        output = result.stdout + result.stderr
        
        if "Unused direct dependencies:" not in output:
            return []
        
        # Extract unused dependencies
        lines = output.split('\n')
        unused_deps = []
        capture = False
        
        for line in lines:
            if "Unused direct dependencies:" in line:
                capture = True
                continue
            if capture and line. strip():
                # Lines with dependencies are typically indented
                if line.startswith('\t') or line.startswith(' '):
                    unused_deps.append(line.strip())
        
        return unused_deps
    
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout checking {so_file. name}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Error checking {so_file. name}: {e}", file=sys.stderr)
        return []


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_unused_dependencies.py <build_directory>")
        print("Example: python3 check_unused_dependencies.py build/")
        sys.exit(1)
    
    build_dir = sys.argv[1]
    
    print(f"Scanning for shared libraries in '{build_dir}'...")
    so_files = find_shared_libraries(build_dir)
    
    if not so_files: 
        print(f"No shared libraries (. so files) found in '{build_dir}'")
        sys.exit(0)
    
    print(f"Found {len(so_files)} shared libraries. Checking for unused dependencies.. .\n")
    
    # Dictionary to store results
    libraries_with_unused = {}
    total_unused_count = 0
    
    # Check each library
    for so_file in so_files:
        unused = check_unused_dependencies(so_file)
        if unused:
            libraries_with_unused[so_file] = unused
            total_unused_count += len(unused)
    
    # Print results
    if not libraries_with_unused:
        print("✓ No unused dependencies found!")
        return
    
    print(f"Found {len(libraries_with_unused)} libraries with unused dependencies")
    print(f"Total unused dependencies: {total_unused_count}\n")
    print("=" * 80)
    
    for so_file, unused_deps in libraries_with_unused. items():
        print(f"\n{so_file. name}:")
        print("Unused direct dependencies:")
        for dep in unused_deps:
            print(f"\t{dep}")
    
    print("\n" + "=" * 80)
    print(f"\nSummary: {len(libraries_with_unused)} libraries with {total_unused_count} unused dependencies")


if __name__ == "__main__":
    main()
