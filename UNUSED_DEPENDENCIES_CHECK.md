# Checking for Unused Library Dependencies in ROOT

This document describes how to identify and fix unused library dependencies in ROOT's CMake build system. 

## Problem

ROOT's shared libraries sometimes link to other libraries they don't actually use. This happens when:
1. Libraries are explicitly linked but not needed
2. Dependencies are incorrectly marked as `PUBLIC` instead of `PRIVATE`, causing transitive linking

This wastes resources and increases build times.

## Solution

### Step 1: Detect Unused Dependencies

Use the provided Python script to scan all shared libraries in your ROOT build: 

```bash
python3 check_unused_dependencies.py /path/to/root/build
