# How to Contribute Code to ROOT

Thank you for your interest in contributing to ROOT!  We strongly welcome and appreciate such contributions!
This short guide tries to make contributing as quick and painless as possible.

> [!NOTE]
> These guidelines should be applicable to most contributes. At the same time, these are not 'one-size-fits-all' rules,
> and there might be cases where diverging from these guidelines is warranted. If you are unsure about how to structure
> your contribution, don't hesitate to reach out! We are always happy to provide help and feedback.

## Your Code Contribution

The source code for ROOT is kept in [GitHub](https://github.com/root-project/root).
Changes go through pull requests ("PRs").
The primary branch for development is `master`.

> [!IMPORTANT]
> We require PRs to cleanly apply to master without a merge commit, i.e. through "fast-forward".
> Please follow the [coding conventions](https://root.cern/contribute/coding_conventions/), as this is a simple item for
> reviewers to otherwise get stuck on.
> To make your (and our own) life easier, we provide a
> [`clang-format` configuration file](https://github.com/root-project/root/blob/master/.clang-format) as well
> as a [`ruff` configuration file](https://github.com/root-project/root/blob/master/ruff.toml)

By providing code, you agree to transfer your copyright on the code to the "ROOT project".
Of course you will be duly credited: for sizable contributions your name will appear in the
[CREDITS](https://raw.githubusercontent.com/root-project/root/master/README/CREDITS)
file shipped with every binary and source distribution.
The copyright transfer helps us with effectively defending the project in case of litigation.

## Your Commit

Each commit is a self-contained, _atomic_ change. This means that:
1. **Each commit should be able to successfully build ROOT.**
Doing so makes traveling through the git history, for example during a `git bisect` much easier.
Ideally, the commit also should not depend on other commits to _run_ ROOT.
2. **Each commit does not contain more than one independent change.**
This allows us to revert changes when needed, without affecting anything else.

> [!TIP]
> During a code review, it may be useful to make smaller commits to track intermediate changes, and rebase after the PR
> is approved to ensure the above points are met and to reduce clutter.

> [!TIP]
> Enable the CMake build option `dev=ON` to enable extra checks that are normally off. Most notably, this will turn
> compiler warnings into errors, preventing you from accidentally push code that causes warnings.

### Your Commit Message

The commit summary (i.e. the first line of the commit message) should be preceded by the a tag indicating the scope of
ROOT that is affected by your commit, in square brackets. Most tags are self-describing (e.g., `[tree]` indicates a
change to TTree, `[RF]` indicates a change to RooFit). If you are unsure about which scope tags to use, we are happy to
point you in the right direction! See also the [commit log](https://github.com/root-project/root/commits/master/) for
examples. The summary itself should not exceed 50 characters (excluding the scope tag), be meaningful (i.e., it
describes the change) and should be written in the
[present imperative mood](https://git.kernel.org/pub/scm/git/git.git/tree/Documentation/SubmittingPatches?id=HEAD#n239)
(e.g. `Add this awesome feature` instead of `Adds this awesome feature` or `Added this awesome feature`).

The commit message that follow the summary can be used to provide more context to the change.
It should describe the **why**, rather than the **what** and **how** (we can gather this from the commit summary and the
change diff, respectively).
The commit message should be wrapped at 72 characters.

> [!TIP]
> We provide a commit message template to help with following the above guidelines. It can be found in the root of this
> repository as [`.git-commit-template`](https://github.com/root-project/root/blob/master/.git-commit-template),
> and can be set to automatically be used for every commit with the following command:
> ```sh
> $ git config commit.template .git-commit-template
> ```

## Your Pull Request

> [!NOTE]
> For the mechanics on how to create pull requests, please visit
> [this page](https://root.cern/for_developers/creating_pr).

The title of your PR follows the same principle as the commit summary. If your PR only involves one commit, you can
reuse this summary. For non-functional changes (e.g. to the documentation) or changes for which you want to
**temporarily** prevent Jenkins from being triggered (e.g., for a draft PR), use `[skip-CI]` as the first tag.
Note that for functional changes this tag needs to be removed and it has to pass the CI before merging to ensure
the change does not break anything.

The PR description describes (and in case of multiple commits, summarizes) the change in more detail.
Again, try to describe the **why** (and in this case, to a lesser extent the **what**), rather than the **how**.

If your PR is related to an open [issue](https://github.com/root-project/root/issues), make sure to link it.
This will be done automatically if you add
[closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
to the PR description.

Once a PR is created, a member of the ROOT team will review it as quickly as possible.  If you are familiar with the
ROOT community, it may be beneficial to add a suggested reviewer to the PR in order to get quicker attention.
Please ping people :wave: should you not get timely feedback, for instance with `@root-project/core ping!`

## Tests

As you contribute code, this code will likely fix an issue or add a feature.
Whatever it is: this requires you to add a new test, or to extend an existing test. Depending on the size and complexity
of this test, it exists either in the `test/` subdirectory of each part of ROOT (see for instance
[`tree/dataframe/test`](https://github.com/root-project/root/tree/master/tree/dataframe/test)), or in the
[roottest](https://github.com/root-project/root/tree/master/roottest) directory.
Tests in `test/` subdirectories are unit tests, mostly based on
[Google Test](https://github.com/google/googletest) and easily extended. Tests in
[roottest](https://github.com/root-project/root/tree/master/roottest) are more involved (e.g., tests requiring custom dictionaries or
data files).

## Continuous Integration

To prevent bad surprises and make a better first impression, we
strongly encourage new developers to [run the tests](https://root.cern/for_developers/run_the_tests/)
_before_ submitting a pull request.

ROOT has automated CI tests :cop: that are used for pull requests:
- *Build and test*: a [Jenkins-based CI workflow](https://github.com/phsft-bot/build-configuration/blob/master/README.md)
    as well as a GitHub Actions CI workflow tests PRs automatically; only a
    [project member](https://github.com/orgs/root-project/people) is allowed to initiate this build.
    The results are posted to the pull request.
    Compared to ROOT's nightly builds, PRs are tested with less tests, on less platforms.
- *Linting check*: `ruff` automatically checks that a PR adheres to the project's
    [style guide](https://github.com/root-project/root/blob/master/ruff.toml).
    If any linting violations are found, it provides you with a detailed report that you should address in your PR.
- *Formatting check*: 
    - `clang-format`: automatically checks that a PR
    [follows](https://github.com/root-project/root/blob/master/.clang-format) ROOT's
    [coding conventions](https://root.cern/contribute/coding_conventions/).
    If coding violations are found, it provides you with a `patch` output that you likely want to apply to your PR.
    - `ruff`: ensures Python code in the PR adheres to the project's [style guidelines](https://github.com/root-project/root/blob/master/ruff.toml).
- *Simple Static Analysis*: PRs are analyzed using [`clang-tidy`](https://clang.llvm.org/extra/clang-tidy/).

Typically, PRs must pass all these tests; we will ask you to fix any issues that may arise.
Some tests are run only outside the PR testing system:
we might come back to you with additional reports after your contribution was merged.

Thank you for reading this; and even more: thank you :bouquet: for considering to contribute!
