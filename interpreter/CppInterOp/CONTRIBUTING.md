# Overview  

Thank you for investing your time in contributing to our project! There are
numbers of ways to contribute to the project and we appreciate all of them. If
you like the project please give CppInterOp a star.

Any contribution to open source makes a difference!

## Are you new to open source, git or GitHub?

To get an overview of the project, read the [README](README.md). Here are some
resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Are you a contributor looking for a challenging summer project?

Various opportunities such as information about google summer of code is
generally published on the [Compiler Research Open Projects page](https://compiler-research.org/open_projects).
If you have used CppInterOp and you have particular project proposal please reach out.

## Ways to contribute

### Submit a bug report

If something does not seem right [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments)  
in [CppInterOps issue tracker](https://github.com/compiler-research/CppInterOp/issues). If a related issue doesn't exist, you can open a  
new issue using a relevant [issue form](https://github.com/compiler-research/CppInterOp/issues/new).

### Good first issues

Some issues have been marked as ["good first issues"](https://github.com/compiler-research/CppInterOp/labels/good%20first%20issue).
These are intended to be a good place to start contributing.

### Write documentation

Documentation is critical for any open source project, especially for complex
projects such as CppInterOp. We have our documentation in the repository which is then
rendered in the [CppInterOp.readthedocs](https://cppinterop.readthedocs.io/en/latest/) website.
Documentation modifications happen by proposing a pull request.

## Creating a successfull pull request

To propose a code modification we use the pull requests. Pull requests which
review quickly and successfully share several common traits:

- Sharp -- intends to fix a concrete problem. Usually the pull request addresses
  an already opened issue;
- Atomic -- has one or more commits that can be reverted without any unwanted
  side effects or regressions, aside from what you’d expect based on its
  message. [More on atomic commits in git](https://www.aleksandrhovhannisyan.com/blog/atomic-git-commits/).
- Descriptive -- has a good description in what is being solved. This
  information is usually published as part of the pull request description and
  as part of the commit message. Writing good commit messages are critical. More
  [here](https://github.blog/2022-06-30-write-better-commits-build-better-projects/)
  and [here](https://cbea.ms/git-commit/). If your pull request fixes an existing
  issue from the bug tracker make sure that the commit log and the pull request
  description mentions `Fixes: #<ISSUE_NUMBER>`. That will link both and will
  close the issue automatically upon merging.
- Tested -- has a set of tests making sure that the issue will not resurface
  without a notice. Usually the codecov bots annotate the code paths that are
  not tested in the pull request after being run.
- Documented -- has good amount of code comment. The test cases are also a good
  source of documentation. [Here](https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/)
  is a guideline about how write good code comments. [Here](https://stackoverflow.com/questions/184618/what-is-the-best-comment-in-source-code-you-have-ever-encountered)
  are examples of what *not* to write as a code comment.

### Developer Documentation  

We have documented several useful hints that usually help when addressing issues
as they come during developement time in our [developer documentation](https://cppinterop.readthedocs.io/en/latest/InstallationAndUsage.html).  
