#!/usr/bin/env python3

from common import printError, printWarning, printInfo, execCommand

import argparse
import json
import os
import shutil
import sys

BOT_REPO_DIR_NAME = "root_bot"
# This is needed for adding the remote branch to apply the patch deriving from the PR
OFFICIAL_ROOT_REPO = 'https://github.com/root-project/root'
OFFICIAL_REPO_DIR_NAME = "root_official"

def validateTargetBranches(brs):
    '''
    Verify the branches are floating point numbers
    
    :param brs: The branches as passed to the tool
    '''
    for br in brs:
        try:
            brf = float(br)
        except ValueError:
            raise Exception(f'Invalid branch name {br}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull',                type=int, help='The PR number to backport')
    parser.add_argument('--to', action='append', type=str, help='The target branches')
    parser.add_argument('--comment',             type=str, help='Comment that contains the info, e.g "/backport to 6.36, 6.42"')
    parser.add_argument('--requestor',           type=str, help='The requestor of the backport"')

    args = parser.parse_args()

    # We give precedence to the comment:
    targetBranches = []
    pullN = 0
    comm = args.comment
    if comm:
        commBeginning = '/backport to '
        for line in comm.splitlines():
            # the comment is of the type "/backport to 6.32, 6.40..."
            if not line.startswith(commBeginning):
                continue
            branchesStr = line[len(commBeginning):]
            targetBranches = [s.strip() for s in branchesStr.split(',')]
        if [] == targetBranches:
            raise Exception(f'Could not extract target branches from the comment in input:\n***\n{args.comment}\n***')
    else:
        targetBranches = args.to
    
    pullN = args.pull

    if not pullN:
        raise Exception('No vaild PR specified!')
    if not targetBranches or 0 == len(targetBranches):
        raise Exception('No target branch specified for the backport!')
    
    validateTargetBranches(targetBranches)

    return (args.requestor, pullN, targetBranches)

def printFirstMessage(requestor, pullNumber, targetBranches):
    '''
    Prints a message useful for debugging, declaring what the script will do 
    for what branches
    
    :param requestor: The user who would like to prepare a backport
    :param pullNumber: The number of the PR to backport
    :param targetBranches: The branches onto which the backport has to be prepared
    '''
    nBranches = len(targetBranches)
    targetBranchesStr = ''
    plural = ''
    if nBranches == 1:
        targetBranchesStr = targetBranches[0]
    else:
        targetBranchesStr = ', '.join(targetBranches[:-1])
        targetBranchesStr += f' and {targetBranches[-1]}' 
        plural = 'es'
    requestorInfo = f' requested by {requestor}' if requestor else ''
    printInfo(f'Preparing to backport PR #{pullNumber} to branch{plural} {targetBranchesStr} {requestorInfo}')

def execCommandOfficialRepo(command):
    '''
    Execute a shell command in the official ROOT repo
    
    :param command: The command to execute
    '''
    return execCommand(command, OFFICIAL_REPO_DIR_NAME)

def execCommandBotRepo(command):
    '''
    Execute a shell command in the bot ROOT repo
    
    :param command: The command to execute
    '''
    return execCommand(command, BOT_REPO_DIR_NAME)

def shortBranchToRealBranch(shortBranchName):
    '''
    Translate branches of the form X.YZ to the names of the branches in the root repository.
    For example, 6.40 will become v6-40-00-patches
    
    :param shortBranchName: The branch name in the form X.YZ, e.g. 6.38
    '''
    major, minor = shortBranchName.split('.')
    return f'v{major}-{minor}-00-patches'

class OfficialROOTRepoPR:
    '''
    A class that represents a PR to the Official ROOT repository 
    '''
    def __init__(self, pullNumber):
        self.pullNumber = pullNumber
        PRJsonStr = execCommandOfficialRepo(f'gh pr view {pullNumber} --json labels,assignees,title,commits,mergeCommit,baseRefName,author')
        self.PRJsonObject = json.loads(PRJsonStr)

    def _parseJson(self, level1Label, level2Label=''):
        '''
        Obtain information from a json of the type returned by the GitHub interface as specified
        by the level 1 and 2 labels.
        The result is a comma separated list of strings.

        A typical json returned has the following structure:
        ```
        {
        "labels": [
        {
        "id": "MDU6TGFiZWwyMzM2MDQ5MDQw",
        "name": "affects:master",
        "description": "",
        "color": "93e0ea"
        },
        {
        "id": "LA_kwDOAKfCqc8AAAACEWlFOQ",
        "name" ...
        ```

        :param jsonStr: The json as string
        :param level1Label: The label of the first level of the json
        :param level2Label: The label of the second level of the json
        '''
        if level2Label != '':
            l1 = self.PRJsonObject[level1Label]
            if isinstance(l1, dict):
                return l1[level2Label]
            else:
                return ','.join([ l[level2Label] for l in  self.PRJsonObject[level1Label] ])
        else:
            return self.PRJsonObject[level1Label]

    def getMergeCommit(self):
        # We have a handle on the most recent commit merged. Its
        # hash is the one in the target branch.
        if self._parseJson('mergeCommit'):
            return self._parseJson('mergeCommit', 'oid')
        else:
            raise Exception(f'PR {self.pullNumber} does not seem to be merged.')

    def getBaseRefName(self):
        return self._parseJson('baseRefName')

    def getNCommits(self):
        return len(self._parseJson('commits'))

    def getTitle(self):
        '''
        Obtain the name of a PR to the official ROOT repo by its number.
        '''
        return self._parseJson('title')
    
    def getLabels(self):
        '''
        Get the labels of a PR to the ROOT official repository as a comma separated list.

        '''
        labels = self._parseJson('labels', 'name')
        labels = f'pr:backport,{labels}'
        printInfo(f'The label(s) of PR {self.pullNumber} is(are) {labels}')
        return labels

    def getAssignees(self):
        '''
        Get the assignees of a PR to the ROOT official repository as a comma separated list.
        '''
        assignees = self._parseJson('assignees', 'login')
        printInfo(f'The assignee(s) of PR {self.pullNumber} is(are) {assignees}')
        return assignees

    def getAuthor(self):
        '''
        Get the author of a PR to the ROOT official repository.
        '''
        author = self._parseJson('author', 'login')
        printInfo(f'The author of PR {self.pullNumber} is {author}')
        return author

    def postCommentAfterBP(self, bpPRUrlBranch):
        '''
        Post a clear message summarising what backports have been created

        :param bpPRUrlBranch: The list of backport PRs numbers and branch names
        '''
        # We now prepare a clear message to post on the PR for which backports have been created...
        prComment = 'This PR has been backported to'
        if len(bpPRUrlBranch) == 1:
            prUrl, brName = bpPRUrlBranch[0]
            prComment += f' branch {brName}: {prUrl}'
        else:
            for prUrl, brName in bpPRUrlBranch:
                prComment += f'\n   - Branch {brName}:#{prUrl}'
        # ...and post it
        execCommandOfficialRepo(f'gh pr comment {self.pullNumber} --body "{prComment}"')
        return 0

def principal():
    
    # We first obtain the information from the parser
    requestor, pullNumber, targetBranches = parse_args()
    
    # We declare what we are about to do, for clarity and debugging purposes
    printFirstMessage(requestor, pullNumber, targetBranches)

    # We get some information about the PR
    thePR = OfficialROOTRepoPR(pullNumber)
    assignees = thePR.getAssignees()
    labels = thePR.getLabels()
    prTitle = thePR.getTitle()
    mergeCommit = thePR.getMergeCommit()
    nCommits = thePR.getNCommits()
    baseRefName = thePR.getBaseRefName()
    originalPRAuthor = thePR.getAuthor()
    
    requestorInfo = f', requested by @{requestor}' if requestor else ''
    if originalPRAuthor!=requestor:
        requestorInfo += f'\nFor your information @{originalPRAuthor}'

    bpPRUrlBranch = []

    labelSwitch = '' if labels == '' else f'--label "{labels}"'
    assigneesSwitch = '' if assignees == '' else f'--assignee "{assignees}"'

    execCommandBotRepo(f'git config user.email "{requestor}@no-reply.github.com"')
    execCommandBotRepo(f'git config user.name "{requestor}"')

    # Before looping on the target branches to prepare the backport PRs, we 
    # need to add the ROOT repo as remote to the bot repo.
    if 'root_upstream' in execCommandBotRepo('git remote'):
        execCommandBotRepo(f'git remote remove root_upstream')
    execCommandBotRepo(f'git remote add root_upstream {OFFICIAL_ROOT_REPO}')

    # We need to fetch the branch onto which the original PR was merged to have the
    # hashes at our disposal.
    execCommandBotRepo(f'git fetch --depth=8192 root_upstream {baseRefName}')

    # Now we loop on the target branches to create one PR for each of them
    for targetBranch in targetBranches:
        printInfo(f'--------- Backporting PR {pullNumber} to branch {targetBranch}')
        realTargetBranch = shortBranchToRealBranch(targetBranch)
        bpBranchName = f'BP_{targetBranch}_pull_{pullNumber}'
        execCommandBotRepo(f'git fetch root_upstream {realTargetBranch}')
        execCommandBotRepo(f'git checkout {realTargetBranch}')
        if bpBranchName in execCommandBotRepo('git branch'):
            execCommandBotRepo(f'git branch -D {bpBranchName}')
        if bpBranchName in execCommandBotRepo('git ls-remote origin'):
            execCommandBotRepo(f'git push -d origin {bpBranchName}')
        execCommandBotRepo(f'git checkout -b {bpBranchName}')
        execCommandBotRepo(f'git cherry-pick -x {mergeCommit}~{nCommits}..{mergeCommit}')
        execCommandBotRepo(f'git push --set-upstream origin {bpBranchName}')
        prUrl = execCommandBotRepo('gh pr create --repo root-project/root ' \
                                               f'--base {realTargetBranch} '\
                                               f'--head root-project-bot:{bpBranchName} ' \
                                               f"--title '[{targetBranch}] {prTitle}'  "\
                                               f"--body 'Backport of #{pullNumber}{requestorInfo}' "\
                                            #    f'{labelSwitch} {assigneesSwitch} ' \
                                               '-d')
        bpPRUrlBranch.append((prUrl,targetBranch))       

    if bpPRUrlBranch == []:
        Exception('No backport succeeded!')

    return thePR.postCommentAfterBP(bpPRUrlBranch)

if __name__ == "__main__":
    sys.exit(principal())