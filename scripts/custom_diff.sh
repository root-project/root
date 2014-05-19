#!/bin/bash

FILEOUT="$1"
FILECMP="$2"

set +o posix
diff -u -w <(grep -v Processing "$FILEOUT") <(grep -v Processing "$FILECMP")
