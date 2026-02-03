#!/usr/bin/env python3

from common import printError, printWarning, printInfo, execCommand

import argparse
import json
import os
import shutil
import sys

PUBLIC_BOT_ROOT_REPO = f'https:/github.com/root-project-bot/root.git'
OFFICIAL_ROOT_REPO = 'https://github.com/root-project/root'

def validateTargetBranches(brs):
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
    parser.add_argument('--push-token',          type=str, help='The secret to push')
    parser.add_argument('--pr-token',            type=str, help='The secret to open a PR')

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

    return (args.push_token, args.pr_token, args.requestor, pullN, targetBranches)

def printFirstMessage(requestor, pullNumber, targetBranches):
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

def shortBranchToRealBranch(shortBranchName):
    major, minor = shortBranchName.split('.')
    return f'v{major}-{minor}-00-patches'

def getPRTitle(pullNumber):
    out = execCommand(f'gh pr view {pullNumber} --repo root-project/root')
    titleLine = out.split('\n')[0]
    _, title = titleLine.split('\t')
    return title

def createWorkdir(workdir):
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

def fetchPatch(pullNumber):
    outname = f'../{pullNumber}.patch'
    out = execCommand(f'gh pr diff {pullNumber} --patch > {outname}')
    return os.path.abspath(outname)

def parseJson(jsonStr, level1, level2):
    '''
    A json of the type
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
    is returned.
    '''
    json_object = json.loads(jsonStr)
    return ','.join([ l[level2] for l in  json_object[level1] ])

def getPRLabels(pullNumber):
    jsonStr = execCommand(f'gh pr view {pullNumber} --json labels')
    labels = parseJson(jsonStr, 'labels', 'name')
    labels = f'pr:backport,{labels}'
    printInfo(f'The label(s) of PR {pullNumber} is(are) {labels}')
    return labels

def getPRAssignees(pullNumber):
    jsonStr = execCommand(f'gh pr view {pullNumber} --json assignees')
    assignees = parseJson(jsonStr, 'assignees', 'login')
    printInfo(f'The assignee(s) of PR {pullNumber} is(are) {assignees}')
    return assignees

def authenticate(token):
    execCommand(f'gh auth login --with-token', theInput=token)

def principal():
    push_token, pr_token, requestor, pullNumber, targetBranches = parse_args()
    printFirstMessage(requestor, pullNumber, targetBranches)

    BOT_ROOT_REPO = f'https://x-access-token:{push_token}@github.com/root-project-bot/root.git'

    # Get some information about the PR
    authenticate(push_token)
    assignees = getPRAssignees(pullNumber)
    labels = getPRLabels(pullNumber)
    patchName = fetchPatch(pullNumber)
    prTitle = getPRTitle(pullNumber)
    
    # We start the automated procedure, assuming to be in a clean root repo
    os.chdir('../')
    workdir = f'./workdir_{pullNumber}'
    
    createWorkdir(workdir)
    os.chdir(workdir)
    
    cloneCmd = 'git clone --depth 1'

    execCommand(cmd=f'{cloneCmd} {BOT_ROOT_REPO}', replace=push_token)
    
    os.chdir('root')
    execCommand(f'git remote add root_upstream {OFFICIAL_ROOT_REPO}')

    requestorInfo = f', requested by @{requestor}' if requestor else ''
    bpPRUrlBranch = []

    labelSwitch = '' if labels == '' else f'--label "{labels}"'
    assigneesSwitch = '' if assignees == '' else f'--assignee "{assignees}"'

    execCommand(f'git config user.email "{requestor}@no-reply.github.com"')
    execCommand(f'git config user.name "{requestor}"')

    # Now we loop on the target branches to create one PR for each of them
    for targetBranch in targetBranches:
       printInfo(f'--------- Backporting PR {pullNumber} to branch {targetBranch}')
       realTargetBranch = shortBranchToRealBranch(targetBranch)
       bpBranchName = f'BP_{targetBranch}_pull_{pullNumber}'
       execCommand(f'git fetch root_upstream {realTargetBranch}')
       execCommand(f'git checkout {realTargetBranch}')
       execCommand(f'git checkout -b {bpBranchName}')
       execCommand(f'git apply --check {patchName}')
       execCommand(f'git am --keep-cr --signoff < {patchName}')
       execCommand(f'git push --set-upstream {bpBranchName}')
       authenticate(pr_token)
       prUrl = execCommand('gh pr create --repo root-project/root ' \
                                        f'--base {realTargetBranch} '\
                                        f'--head root-project-bot:{bpBranchName} ' \
                                        f'--title "[{targetBranch}] {prTitle}"  '\
                                        f'--body "Backport of #{pullNumber}{requestorInfo}" '\
                                        f'{labelSwitch} {assigneesSwitch} ' \
                                        '-d')
       bpPRUrlBranch.append((prUrl,targetBranch))       

    if bpPRUrlBranch == []:
        Exception('No backport succeeded!')

    prComment = 'This PR has been backported to'
    if len(bpPRUrlBranch) == 1:
        prUrl, brName = bpPRUrlBranch[0]
        prComment += f' branch {brName}: {prUrl}'
    else:
        for prUrl, brName in bpPRUrlBranch:
            prComment += f'\n   - Branch {brName}:#{prUrl}'

    authenticate(pr_token)
    os.chdir('../../root') # we go back to the original root repo dir
    execCommand(f'gh pr comment {pullNumber} --body "{prComment}"')

if __name__ == "__main__":
    sys.exit(principal())