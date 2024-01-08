
How to Contribute Code to ROOT
==============================

Thank you for your interest in contributing to ROOT!  We warmly welcome and appreciate such contributions!
This short guide tries to make contributing as quick and painless as possible.

Any questions? Contact the devs!
-----------------------

First of all, feel free to discuss your plans / improvements with us at https://root-forum.cern.ch - that's where you can find the developers.

What to contribute
-----------------------

You can fix one of the [good first issues](https://github.com/root-project/root/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22), or one of the bugs that you hit yourself.
Or you can contribute a new feature - if that's your plan then please first discuss with the developers whether ROOT is interested in incorporating it.

How to contribute
-----------------------

We use the following procedure:
1. Code following the conventions
2. Writing tests for your code
3. You build ROOT with your changes
4. You run the test suite
5. You create a pull request
6. Your code gets tested by our continuous integration system

We explain these steps below.

Coding conventions
-----------------------

Please follow the [coding conventions](https://root.cern.ch/coding-conventions).

Writing tests for your code
-----------------------

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

Building ROOT with your changes
-----------------------

The source code for ROOT is kept in [GitHub](https://github.com/root-project/root).
In general, with the sources in `../src`, you build ROOT with `cmake -Droottest=On -Dtesting=On -Dall=On ../src && cmake --build`.
If possible, configure with ninja, passing `-G Ninja` to the first `cmake` invocation.
If any of this doesn't work for some reason then you can find more details [here](https://root.cern/install/build_from_source/).

Testing ROOT with your changes
-----------------------

To prevent bad surprises and make a better first impression, we
strongly encourage new developers to [run the tests](https://root.cern/for_developers/run_the_tests/)
_before_ submitting a pull request.

To run ROOT's test suite you need to execute `ctest -j 12`. The number (12 in this example) depends on how many cores you have.
This takes more than 30 minutes; it should not show any failing tests.
Sometimes we don't manage to cover every distribution / setup, and yoou get a test failure.
Please [check with us](https://root-forum.cern.ch)  whether the failure is introduced by your change or not.
You can also run the test suite with and without your chages; it should show the same set of failures (ideally none).

Create a pull request
-----------------------

Changes are reviewed by the developers using the common pull request ("PR") approach.
The primary branch for development is `master`.
Visit [this page](https://root.cern/for_developers/creating_pr) for how to create pull requests.

By providing code, you agree to transfer your copyright of your code to the "ROOT project".
Of course you will be duly credited: for sizable contributions your name will appear in the
[CREDITS](https://raw.githubusercontent.com/root-project/root/master/README/CREDITS){:target="_blank"}
file shipped with every binary and source distribution.
The copyright transfer helps us with effectively defending the project in case of litigation.

:warning: We require PRs to cleanly apply to master without a merge commit, i.e. through "fast-forward".

Once a PR is created, a member of the ROOT team will review it as quickly as possible.  If you are familiar with the
ROOT community, it may be beneficial to add a suggested reviewer to the PR in order to get quicker attention.
Please ping people :wave: should you not get timely feedback, for instance with `@root-project/core ping!`


Continuous Integration
----------------------

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
