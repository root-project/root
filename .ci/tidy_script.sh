#!/usr/bin/env bash

set -ex

CLANG_TIDY_CHECKS='-*'
if [[ $TOOL == clang-tidy-analyzer ]]; then
  CLANG_TIDY_CHECKS+=',clang-analyzer-*,-clang-analyzer-alpha*,bugprone*'
elif [[ $TOOL == clang-tidy-modernize ]]; then
  CLANG_TIDY_CHECKS+=',modernize*'
fi

echo "Running clang-tidy only against the changes in branch $TRAVIS_BRANCH."

cd ../root/

# Workaround for travis issue: travis-ci/travis-ci#6069
git remote set-branches --add origin master
git fetch

# clang-tidy-diff.py not installed on travis
wget https://raw.githubusercontent.com/llvm-mirror/clang-tools-extra/release_50/clang-tidy/tool/clang-tidy-diff.py

RESULT_OUTPUT="$(git diff -U0 origin/master | python clang-tidy-diff.py -p1 -clang-tidy-binary $(which clang-tidy) \
                 -checks=$CLANG_TIDY_CHECKS)"
if [[ $? -eq 0 ]]; then
  echo "$TOOL passed."
  exit 0
else
  echo "To reproduce it locally please run"
  echo -e "\tgit checkout $TRAVIS_BRANCH"
  echo -e "Command: git diff -U0 $TRAVIS_BRANCH..origin/master | clang-tidy-diff.py -p1 -clang-tidy-binary \$(which clang-tidy) -checks=$CLANG_TIDY_CHECKS"
  echo "$RESULT_OUTPUT"
  exit 1
fi
