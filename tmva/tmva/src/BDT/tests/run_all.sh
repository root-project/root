#!/bin/bash

for f in *.exe; do  # or wget-*.exe instead of *.exe
  ./"$f" -H
done
