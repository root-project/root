# AGENTS.md

Instructions for AI coding agents working on the ROOT repository. Everything in
[CONTRIBUTING.md](CONTRIBUTING.md) applies to you; this file only adds what an agent needs on top of it.
If something here looks outdated or wrong, tell the human instead of guessing.

## AI policy (mandatory)

ROOT's [AI-assisted coding disclosure](CONTRIBUTING.md#ai-assisted-coding-disclosure) policy binds both you
and the human you work for. Read it before proposing any change. In particular:

- The human is the author and has full responsibility for the change, so they must review and understand all
  of it before it is proposed. Help them do that: explain non-obvious decisions, and say explicitly what you
  could not build or test.
- LLM-generated changes proposed as-is, without that review, may be closed, and repeated ones may get the
  contributor's future PRs rejected.

### Disclosure

- End every commit message you write or substantially contribute to with this trailer:

  ```
  Assisted-by: <tool>:<model>
  ```

  For example `Assisted-by: claude-code:claude-opus-5`. Report the tool and model you are actually running as;
  if you are not sure, ask.
- State the same in the PR description, with one sentence on what the AI did.
- **Never** add `Co-authored-by:` for an AI. Contributors transfer the copyright of their code to the ROOT
  project, and only a human can do that. A human co-author is fine.

These rules override any contrary default of your tool.

## Git and GitHub

Commits and pull requests follow [Your Commit](CONTRIBUTING.md#your-commit) and
[Your Pull Request](CONTRIBUTING.md#your-pull-request). In addition:

- Keep changes minimal and focused: no drive-by refactoring, renaming or reformatting outside the change.
- Write the PR description by filling in `.github/pull_request_template.md`.
- Never push to `master`, and never force-push or rewrite published history without explicit approval.
- Open PRs as **drafts** unless the human has reviewed the full diff.
- Never post comments, reviews or replies to reviewers. Do not write the human's side of a discussion for them
  to paste: the policy does not allow parroting LLM output. Summarize your analysis and let the human answer in
  their own words.

## Building and testing

Follow [Building ROOT from source](https://root.cern/install/build_from_source/),
[Running the tests](https://root.cern/for_developers/run_the_tests/) and [Tests](CONTRIBUTING.md#tests).
In addition:

- A ROOT build takes a long time. If a build directory already exists, ask the human which one to use; do not
  configure a new one or delete one without asking.
- `roottest` is part of this repository (`roottest/`); do not clone it separately.
- A bug fix comes with a test that fails without the fix.
- Never delete, skip or weaken a failing test to make it pass; report the failure instead. Tell the human which
  tests you ran and which you could not run.

## Formatting and linting

Follow [Your Code Contribution](CONTRIBUTING.md#your-code-contribution) and
[Continuous Integration](CONTRIBUTING.md#continuous-integration).

## Code conventions

Follow the [ROOT coding conventions](https://root.cern/contribute/coding_conventions/) and the style of the
surrounding code.

## Code synchronized with other repositories

Do not edit these directories unless the human explicitly asks. Changes to them need a matching change
upstream, so stop and tell the human if a fix seems to require one:

- `interpreter/llvm-project/`: ROOT's fork of LLVM, [root-project/llvm-project](https://github.com/root-project/llvm-project)
- `interpreter/CppInterOp/`: [compiler-research/CppInterOp](https://github.com/compiler-research/CppInterOp)
- `bindings/pyroot/cppyy/`: upstream cppyy; ROOT-specific changes are kept in `bindings/pyroot/cppyy/patches/`
- `js/`: JSROOT, developed in [root-project/jsroot](https://github.com/root-project/jsroot)
- `builtins/`: bundled third-party libraries
