#!/usr/bin/env bash

if [ -z "$MERGE_BASE" ]; then
  echo "Warning: merging base was not found!  Using the tip of $TRAVIS_BRANCH instead."
  MERGE_BASE=$BASE_COMMIT
fi

echo "Running clang-format from $TRAVIS_PULL_REQUEST_BRANCH against branch $TRAVIS_BRANCH, with hash $MERGE_BASE"
clang-format --version
COMMIT_FILES=$(git diff --name-status $MERGE_BASE | grep -i -v '.mjs$' | grep -i -v LinkDef | grep -v -E '^D[[:space:]]+' | sed -E 's,^.[[:space:]]+,,')

RESULT_OUTPUT="no modified files to format"
if [ ! -z "$COMMIT_FILES" ]; then
  # If COMMIT_FILES is empty, git-clang-format will take all the modified files
  # Therefore, we only actually run git-clang-format if there is anything to check
  RESULT_OUTPUT="$(git-clang-format --commit $MERGE_BASE --diff --binary `which clang-format` -- $COMMIT_FILES)"
fi

if [ "$RESULT_OUTPUT" == "no modified files to format" ] \
  || [ "$RESULT_OUTPUT" == "clang-format did not modify any files" ] ; then

  echo "clang-format passed."
  exit 0
else
  echo  -e "clang-format failed with the following diff:\n"
  echo "$RESULT_OUTPUT"

  echo -e "\n\nTo reproduce it locally please run (skip the first 2 steps if already have the branch checked out):"
  echo -e "\tgit checkout -b \$USER-$TRAVIS_PULL_REQUEST_BRANCH master"
  echo -e "\tgit pull $TRAVIS_PULL_REQUEST_REPO $TRAVIS_PULL_REQUEST_BRANCH"
  echo -e "\tgit-clang-format --commit $MERGE_BASE --diff --binary $(which clang-format) # adjust to point to the local clang-format"

  echo -e "\nConsider running the following to apply the code formatting changes without bloating the history."
  echo -e "\t\tgit rebase -i -x \"git-clang-format master && git commit -a --allow-empty --fixup=HEAD\" --strategy-option=theirs origin/master"
  echo -e "\t Then inspect the results with 'git log --oneline' and 'git show'"
  echo -e "\t Then squash without poluting the history with: git rebase --autosquash -i master"

  exit 1
fi
