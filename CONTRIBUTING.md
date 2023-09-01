
How to Contribute Code to ROOT
==============================

Thank you for your interest in contributing to ROOT!  We strongly welcome and appreciate such contributions!
This short guide tries to make contributing as quick and painless as possible.

Your Pull Request
-----------------------

The source code for ROOT is kept in [GitHub](https://github.com/root-project/root).
Changes go through pull requests ("PRs").
The primary branch for development is `master`.
Visit [this page](https://root.cern/for_developers/creating_pr) for the mechanics on how to
create pull requests.

By providing code, you agree to transfer your copyright on the code to the "ROOT project".
Of course you will be duly credited: for sizable contributions your name will appear in the
[CREDITS](https://raw.githubusercontent.com/root-project/root/master/README/CREDITS){:target="_blank"}
file shipped with every binary and source distribution.
The copyright transfer helps us with effectively defending the project in case of litigation.

:warning: We require PRs to cleanly apply to master without a merge commit, i.e. through "fast-forward".
Please follow the [coding conventions](https://root.cern.ch/coding-conventions),
as this is a simple item for reviewers to otherwise get stuck on.

Once a PR is created, a member of the ROOT team will review it as quickly as possible.  If you are familiar with the
ROOT community, it may be beneficial to add a suggested reviewer to the PR in order to get quicker attention.
Please ping people :wave: should you not get timely feedback, for instance with `@root-project/core ping!`

Tests
-----

As you contribute code, this code will likely fix an issue or add a feature.
Whatever it is: this requires you to add a new test, or to extend an existing test.
We have concise unittests in the `test/` subdirectory of each part of ROOT;
see for instance [`tree/dataframe/test`](https://github.com/root-project/root/tree/master/tree/dataframe/test).
These tests are generally based on [Google Test](https://github.com/google/googletest) and easily extended.

For more involved tests, such as tests requiring custom dictionaries or data
files, we have [roottest](https://github.com/root-project/roottest.git).
Suppose for your PR you create a branch on `root.git`.
Our CI infrastructure automatically picks up a branch with the same name in your fork of `roottest.git`
and use that for testing your PR.


Continuous Integration
----------------------

To prevent bad surprises and make a better first impression, we
strongly encourage new developers to [run the tests](https://root.cern/for_developers/run_the_tests/)
_before_ submitting a pull request.

ROOT has automated CI tests :cop: that are used for pull requests:
- *Build and test*: a [Jenkins-based CI workflow](https://github.com/phsft-bot/build-configuration/blob/master/README.md)
    tests PRs automatically; a project member might need to initiate this build.
    The results are posted to the pull request.
    Compared to ROOT's nightly builds, PRs are tested with less tests, on less platforms.
- *Formatting check*: `clang-format` automatically checks that a PR
    [follows](https://github.com/root-project/root/blob/master/.clang-format) ROOT's
    [coding conventions](https://root.cern/contribute/coding_conventions/).
    If coding violations are found, it provides you with a `patch` output that you likely want to apply to your PR.
- *Simple Static Analysis*: PRs are analyzed using [`clang-tidy`](https://clang.llvm.org/extra/clang-tidy/).

Typically, PRs must pass all these tests; we will ask you to fix any issues that may arise.
Some tests are run only outside the PR testing system:
we might come back to you with additional reports after your contribution was merged.

Thank you for reading this; and even more: thank you :bouquet: for considering to contribute!
