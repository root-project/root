#!/usr/bin/env python3

import subprocess
import sys

# Parameters
CLING_TAG_ROOT_HASH_PREFIX='__internal-root-'
# Arbitrarily choose a commit hash which is "old enough"
# In this case, we choose the first commit after the latest
# cling release at the time of writing, 1.2, i.e. 5ea7949 (in cling)
# 08f123f8e7 (in ROOT)
# See Cling https://github.com/root-project/cling/commit/5ea7949
# ROOT https://github.com/root-project/root/commit/08f123f8e7
DEFAULT_STARTING_ROOT_HASH = '08f123f8e7'
CLING_REPO_DIR_NAME = 'cling'
ROOT_REPO_DIR_NAME = 'root'
INTERP_DIR_NAME = 'interpreter/cling'
DEST_INTERP_DIR_NAME = ''
TEXTINPUT_DIR_NAME = 'core/textinput/src/textinput'
DEST_TEXTINPUT_DIR_NAME='lib/UserInterface/textinput'

def printError(msg):
    print(f'*** Error: {msg}')

def printWarning(msg):
    print(f'*** Warning: {msg}')

def printInfo(msg):
    print(f'Info: {msg}')

def execCommand(cmd, thisCwd = './', theInput = None, desc=""):
    '''
    Execute a command and return the output. For logging reasons, the command
    is also printed.
    If "desc" is specificed, the command is not printed but "desc".
    '''
    if '' == desc:
        printInfo(f'In directory {thisCwd} *** {cmd} {"with std input" if theInput else ""}')
    else:
        print(desc)
    compProc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=thisCwd, input=theInput, encoding='latin1')
    if 0 != compProc.returncode:
        print(f"Error:\n {compProc.stderr.strip()}")
        raise ValueError(f'Command "{cmd}" failed ({compProc.returncode})')
    out = compProc.stdout.strip()
    return out

def getAllClingTags():
    execCommand(f'git fetch --tags', CLING_REPO_DIR_NAME)

def getRootSyncTag():
    # We try to get the tags, to get the latest ROOT commit that was synchronized
    getAllClingTags()
    tagsStr = execCommand(f'git tag', CLING_REPO_DIR_NAME)
    tags = tagsStr.split('\n')
    if tags == ['']:
        printInfo(f'No tags found locally. Looking in the source repository.')
        tagsStr = execCommand(f'git ls-remote --tags origin', CLING_REPO_DIR_NAME)
        tags = tagsStr.split('\n')
        print(tags)
        tags = list(map(lambda t: t.split('/')[-1] if '/' in t else t, tags))
        print(tags)

    printInfo(f'Tags found: {str(tags)}')

    rootTags = list(filter(lambda tag: tag.startswith(CLING_TAG_ROOT_HASH_PREFIX), tags))
    if not rootTags:
        raise ValueError(f'No sync tags starting with {CLING_TAG_ROOT_HASH_PREFIX} were found!')
    if len(rootTags) > 1:
        raise ValueError(f'More than one sync tag were found: {str(rootTags)}!')
    return rootTags[0]

def getStartingRootHash(rootTag):
    printInfo('Getting the starting ROOT Hash from the tag')
    prefixLen = len(CLING_TAG_ROOT_HASH_PREFIX)
    defHashLen = len(DEFAULT_STARTING_ROOT_HASH)
    hash = rootTag[prefixLen:prefixLen+defHashLen]
    return hash

def getHashes(repoDirName, startingHash, dirInRepoName=''):
    out = execCommand(f'git log --oneline {startingHash}..HEAD {dirInRepoName}', thisCwd=repoDirName)
    hashes = []
    if not out:
        return hashes
    # skip the first line since it's '\n'
    hashes = [line.split(' ', 1)[0] for line in out.split('\n')]
    return hashes

def createPatches(rootHashes, interpHashes, textinputHashes):
    patches = []

    # We'll need a few sets to quickly check what to do with ROOT hashes
    interpHashesSet = set(interpHashes)
    textInputHashesSet = set(textinputHashes)
    allHashesSet = interpHashesSet | textInputHashesSet

    # We filter the ROOT hashes that we do not want to sync
    rootHashesToSync = list(filter(lambda hash: hash in allHashesSet, rootHashes))

    # We loop on ROOT hashes to sync, from oldest to newest
    # to return a list of triples [label, hash, patchAsSting], where label allows us
    # to disinguish between textinput and interpreter patches and hash is there
    # for debugging purposes.
    # One commit can be visible in both directories, ergo 2 patches per hash are possible.
    for rootHashtoSync in reversed(rootHashesToSync):
        keys = []
        if rootHashtoSync in interpHashesSet: keys.append(INTERP_DIR_NAME)
        if rootHashtoSync in textInputHashesSet: keys.append(TEXTINPUT_DIR_NAME)
        for key in keys:
            patchAsStr = execCommand(f"git format-patch -1 {rootHashtoSync} {key} --stdout", ROOT_REPO_DIR_NAME)
            patches.append([key, rootHashtoSync, patchAsStr])
    return patches

def applyPatches(patches):
    for dirInRepo, hash, patchAsStr in patches:
        ignorePathLevel = dirInRepo.count('/') + 2
        destDirName = DEST_INTERP_DIR_NAME if dirInRepo == INTERP_DIR_NAME else DEST_TEXTINPUT_DIR_NAME
        directoryOption = f'--directory {destDirName}' if destDirName else ''
        printInfo(f'Applying {hash} restricted to {dirInRepo} to repository {CLING_REPO_DIR_NAME}')
        execCommand(f'git am -p {ignorePathLevel} {directoryOption}', CLING_REPO_DIR_NAME, patchAsStr)

def syncTagAndPush(oldSyncTag, rootSyncHash):
    '''
    Replace the tag mentioning the previous root commit to which the repo was synchronized.
    Push the changes and tags upstream.
    '''
    # We fetch the remote and local tags to make the following operations more resilient
    remoteTags = execCommand(f'git ls-remote --tags origin', CLING_REPO_DIR_NAME)
    localTags = execCommand(f'git tag', CLING_REPO_DIR_NAME)

    # Clean the old tag
    printInfo(f'Found a sync tag ({oldSyncTag}): deleting it.')
    if oldSyncTag in localTags:
        execCommand(f'git tag -d {oldSyncTag}', CLING_REPO_DIR_NAME)
    if oldSyncTag in remoteTags:
        execCommand(f'git push --delete origin  {oldSyncTag}', CLING_REPO_DIR_NAME)
    else:
        printWarning(f'Tag {oldSyncTag} was not found in the upstream repository.')

    # Time to push the sync commits!
    execCommand(f'git push', CLING_REPO_DIR_NAME)

    # And finally we tag and push the tag
    newTag = CLING_TAG_ROOT_HASH_PREFIX+rootSyncHash
    printInfo(f'Creating a new tag ({newTag}).')
    execCommand(f'git tag {newTag}', CLING_REPO_DIR_NAME)
    execCommand(f'git push origin tag {newTag}', CLING_REPO_DIR_NAME)

def principal():

    # We want a starting hash not to deal with the entire commit history
    # of ROOT
    rootSyncTag = getRootSyncTag()
    startingRootHash = getStartingRootHash(rootSyncTag)

    # We now get all recent ROOT hashes, as well as the ones relative
    # to commits in the directories we are interested in for the sync
    rootHashes = getHashes(ROOT_REPO_DIR_NAME, startingRootHash)
    interpHashes = getHashes(ROOT_REPO_DIR_NAME, startingRootHash, INTERP_DIR_NAME)
    textinputHashes = getHashes(ROOT_REPO_DIR_NAME, startingRootHash, TEXTINPUT_DIR_NAME)

    # If we have no commits to sync, we quit.
    if not interpHashes and not textinputHashes:
        # nothing to do, we have no commits to sync
        printInfo('No commit to sync. Exiting now.')
        return 0

    printInfo(f'We found:\n - {len(interpHashes)} patches from the directory {INTERP_DIR_NAME}\n - {len(textinputHashes)} patches from the directory {TEXTINPUT_DIR_NAME}')

    # We now create the patches we want to apply to the cling repo
    patches = createPatches(rootHashes, interpHashes, textinputHashes)

    # We now apply the patches
    if not patches:
        printError('No patch was distilled: this status should not be reachable')
        return 1

    # We now need to apply patches, update the tag that mentions the ROOT commit
    # to which the cling repo was synchronised and push everything.
    # First of all we need to acquire an identity
    printInfo('Acquiring an identity in preparation to the upstreaming of the changes')
    execCommand('git config user.email "root@persona.com"', CLING_REPO_DIR_NAME)
    execCommand('git config user.name "Root Persona"', CLING_REPO_DIR_NAME)

    applyPatches(patches)

    syncTagAndPush(rootSyncTag, rootHashes[0])

if __name__ == '__main__':
    sys.exit(principal())
