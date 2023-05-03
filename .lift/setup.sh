#!/usr/bin/env bash
wget https://raw.githubusercontent.com/root-project/root-ci-images/main/ubuntu22/packages.txt
sudo apt update && sudo apt install -y ninja-build `cat packages.txt`
# Generate a compilation database next
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DALLOW_IN_SOURCE=On -G Ninja . || cat CMakeFiles/CMakeOutput.log
ninja
