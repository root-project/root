#!/usr/bin/env bash

set -ex

BASE_COMMIT=$(git rev-parse $TRAVIS_BRANCH)
echo "Running clang-format against branch $TRAVIS_BRANCH, with hash $BASE_COMMIT"
COMMIT_FILES=$(git diff --name-only $BASE_COMMIT | grep -i -v LinkDef)
RESULT_OUTPUT="$(git-clang-format --commit $BASE_COMMIT --diff --binary `which clang-format` $COMMIT_FILES)"

if [ "$RESULT_OUTPUT" == "no modified files to format" ] \
  || [ "$RESULT_OUTPUT" == "clang-format did not modify any files" ] ; then

  echo "clang-format passed."
  exit 0
else
  echo "clang-format failed."
  echo "To reproduce it locally please run"
  echo -e "\tgit checkout $TRAVIS_BRANCH"
  echo -e "\tgit-clang-format --commit $BASE_COMMIT --diff --binary $(which clang-format)"
  echo "$RESULT_OUTPUT"
  exit 1
fi
