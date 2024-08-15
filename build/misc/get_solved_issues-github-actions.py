#! /usr/bin/env python
#
# Harvest the solved issues for a certain tag and print them out in a format
# which is ready to be pasted in the release notes.
#
# Copyright (c) 2024 Rene Brun and Fons Rademakers
# Author: Enrico Guiraud, Axel Naumann, Danilo Piparo

from github3 import GitHub
from argparse import ArgumentParser
import sys, os, re

token = os.environ["GITHUB_TOKEN"]

def parse_args():
    p = ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--project-name",
        type=str,
        dest="name",
        help="GitHub project name, or part of it",
    )
    return p.parse_args()


def format_solved_issue(number: int, title: str, url: str) -> str:
    return f"* [[#{number}]({url})] - {title}"


def print_fmtted_issues():
    repo = GitHub(token=token).repository("root-project", "root")
    args = parse_args()
    name_pattern_str = args.name
    if name_pattern_str[0] == 'v': name_pattern_str = name_pattern_str[1:]
    # Don't care about "." or "/" or "-" delimiting version number parts:
    name_pattern = re.compile(r''.join([r'\W' if not char.isalnum() else char for char in name_pattern_str]))
    def matches(project):
        return name_pattern.search(project.name)
    
    repo_projects_str = " ".join(repo.projects())
    print(f"List of projects in the repository: {repo_projects_str}")

    pro = [p for p in repo.projects() if matches(p)]
    if len(pro) != 1:
        print(
            "Could not identify a unique GitHub project in root-project/root with "
            f"name containing {name_pattern_str}",
            file=sys.stderr,
        )
        sys.exit(1)
    pro = pro[0]

    col = list(pro.columns())
    if len(col) != 1:
        print(f"Project '{pro.name}' has more than one column", file=sys.stderr)
        sys.exit(2)
    col = col[0]

    issues_and_prs = [card.retrieve_issue_from_content() for card in col.cards()]
    issues = [i for i in issues_and_prs if i.pull_request() is None]
    sored_issues = sorted(issues, key = lambda i: -1 * i.number)
    fmtted_issues = [format_solved_issue(i.number, i.title, i.html_url) for i in sored_issues]
    print(pro.name)
    print('### Bugs and Issues fixed in this release\n')
    for i in fmtted_issues:
        print(i)

    prs = [format_solved_issue(i.number, i.title, i.html_url) for i in issues if i.pull_request() is not None]
    # check we did not miss anything and we did not count anything twice
    assert(len(issues) == len(fmtted_issues) + len(prs))
    if len(prs) > 0:
        print("<!--")
        print("Also found the following pull requests in this project (they should not be there):")
        for pr in prs:
            print(pr)
        print("-->")


if __name__ == "__main__":
    print_fmtted_issues()
