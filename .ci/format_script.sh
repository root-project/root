#!/usr/bin/env bash

set -ex

echo "Running clang-format against branch $TRAVIS_BRANCH, with hash $BASE_COMMIT"
COMMIT_FILES=$(git diff --name-status $BASE_COMMIT | grep -i -v '.mjs$' | grep -i -v LinkDef | grep -v -E '^D +' | sed -E 's,^.[[:space:]]+,,')

RESULT_OUTPUT="no modified files to format"
if [ ! -z "$COMMIT_FILES" ]; then
  # If COMMIT_FILES is empty, git-clang-format will take all the modified files
  # Therefore, we only actually run git-clang-format if there is anything to check
  RESULT_OUTPUT="$(git-clang-format --commit $BASE_COMMIT --diff --binary `which clang-format` $COMMIT_FILES)"
fi

if [ "$RESULT_OUTPUT" == "no modified files to format" ] \
  || [ "$RESULT_OUTPUT" == "clang-format did not modify any files" ] ; then

  echo "clang-format passed."
  exit 0
else
  echo "clang-format failed."
  echo "To reproduce it locally please run"
  echo -e "\tgit checkout $TRAVIS_PULL_REQUEST_BRANCH"
  echo -e "\tgit-clang-format --commit $BASE_COMMIT --diff --binary $(which clang-format)"
  echo "$RESULT_OUTPUT"

  echo -e "\n\nPlease apply the code formatting changes without bloating the history."
  echo -e "\tConsider running:"
  echo -e "\t\tgit checkout $TRAVIS_PULL_REQUEST_BRANCH"
  echo -e "\t\tgit rebase -i -x \"git-clang-format-7 master && git commit -a --allow-empty --fixup=HEAD\" --strategy-option=theirs origin/master"
  echo -e "\t Then inspect the results with git log --oneline"
  echo -e "\t Then squash without poluting the history with: git rebase --autosquash -i master"

  exit 1
fi
