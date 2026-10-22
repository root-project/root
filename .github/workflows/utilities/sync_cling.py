#!/usr/bin/env python3

import subprocess
import sys

from common import printWarning, printInfo, execCommand

# Parameters
CLING_TAG_ROOT_HASH_PREFIX = "__internal-root-"
# Arbitrarily choose a commit hash which is "old enough"
# In this case, we choose the first commit after the latest
# cling release at the time of writing, 1.2, i.e. 5ea7949 (in cling)
# 08f123f8e7 (in ROOT)
# See Cling https://github.com/root-project/cling/commit/5ea7949
# ROOT https://github.com/root-project/root/commit/08f123f8e7
DEFAULT_STARTING_ROOT_HASH = "08f123f8e7"
CLING_REPO_DIR_NAME = "cling"
ROOT_REPO_DIR_NAME = "root"
INTERP_DIR_NAME = "interpreter/cling"
DEST_INTERP_DIR_NAME = ""

def getAllClingTags():
    execCommand("git fetch --tags", CLING_REPO_DIR_NAME)


def getRootSyncTag():
    # We try to get the tags, to get the latest ROOT commit that was synchronized
    getAllClingTags()
    tagsStr = execCommand("git tag", CLING_REPO_DIR_NAME)
    tags = tagsStr.split("\n")
    if tags == [""]:
        printInfo("No tags found locally. Looking in the source repository.")
        tagsStr = execCommand("git ls-remote --tags origin", CLING_REPO_DIR_NAME)
        tags = tagsStr.split("\n")
        print(tags)
        tags = list(map(lambda t: t.split("/")[-1] if "/" in t else t, tags))
        print(tags)

    printInfo(f"Tags found: {str(tags)}")

    rootTags = list(filter(lambda tag: tag.startswith(CLING_TAG_ROOT_HASH_PREFIX), tags))
    if not rootTags:
        raise ValueError(f"No sync tags starting with {CLING_TAG_ROOT_HASH_PREFIX} were found!")
    if len(rootTags) > 1:
        raise ValueError(f"More than one sync tag were found: {str(rootTags)}!")
    return rootTags[0]


def getStartingRootHash(rootTag):
    printInfo("Getting the starting ROOT Hash from the tag")
    prefixLen = len(CLING_TAG_ROOT_HASH_PREFIX)
    defHashLen = len(DEFAULT_STARTING_ROOT_HASH)
    hash = rootTag[prefixLen : prefixLen + defHashLen]
    # A tag made from a local ROOT branch names a commit that is not in ROOT's history.
    # Then start from the ROOT commit that cling's last commit came from.
    if subprocess.run(["git", "merge-base", "--is-ancestor", hash, "HEAD"], cwd=ROOT_REPO_DIR_NAME).returncode != 0:
        printWarning(f"The sync tag names {hash}, which is not in ROOT's history")
        hash = findRootCommit(execCommand("git log -1 --format=%ae%x09%ad --date=raw HEAD", CLING_REPO_DIR_NAME))
    return hash


def findRootCommit(identity):
    """
    Find the ROOT commit with the given author identity. git am keeps the author
    and the author date, so they identify a ROOT commit in cling.
    """
    out = execCommand(f"git log --format=%h%x09%ae%x09%ad --date=raw HEAD -- {INTERP_DIR_NAME}", ROOT_REPO_DIR_NAME)
    for line in out.split("\n"):
        hash, commitIdentity = line.split("\t", 1)
        if commitIdentity == identity:
            printInfo(f"Cling's last commit comes from ROOT's {hash}")
            return hash
    raise ValueError(f"No ROOT commit matches cling's last commit ({identity})")


def getHashes(repoDirName, startingHash, dirInRepoName=""):
    out = execCommand(f"git log --oneline {startingHash}..HEAD -- {dirInRepoName}", thisCwd=repoDirName)
    hashes = []
    if not out:
        return hashes
    # skip the first line since it's '\n'
    hashes = [line.split(" ", 1)[0] for line in out.split("\n")]
    return hashes


def createPatches(rootHashes, interpHashes):
    patches = []

    # We'll need a few sets to quickly check what to do with ROOT hashes
    interpHashesSet = set(interpHashes)

    # We filter the ROOT hashes that we do not want to sync
    rootHashesToSync = list(filter(lambda hash: hash in interpHashesSet, rootHashes))

    # We loop on ROOT hashes to sync, from oldest to newest
    # to return a list of triples [dirInRepo, hash, patchAsSting], where dirInRepo is the
    # directory and hash is there for debugging purposes.
    # Commits that cling already has, e.g. because the sync tag names an older ROOT
    # commit than the one cling was synchronized to, are skipped.
    syncedIdentities = set(execCommand("git log -n 1000 --format=%ae%x09%ad --date=raw HEAD", CLING_REPO_DIR_NAME).split("\n"))
    for rootHashtoSync in reversed(rootHashesToSync):
        identity = execCommand(f"git log -1 --format=%ae%x09%ad --date=raw {rootHashtoSync}", ROOT_REPO_DIR_NAME)
        if identity in syncedIdentities:
            printWarning(f"Skipping {rootHashtoSync}: cling already has it")
            continue
        keys = []
        if rootHashtoSync in interpHashesSet:
            keys.append(INTERP_DIR_NAME)
        for key in keys:
            patchAsStr = execCommand(f"git format-patch -1 {rootHashtoSync} --stdout -- {key}", ROOT_REPO_DIR_NAME)
            patches.append([key, rootHashtoSync, patchAsStr])
    return patches


def applyPatches(patches):
    for dirInRepo, hash, patchAsStr in patches:
        ignorePathLevel = dirInRepo.count("/") + 2
        destDirName = DEST_INTERP_DIR_NAME
        directoryOption = f"--directory {destDirName}" if destDirName else ""
        printInfo(f"Applying {hash} restricted to {dirInRepo} to repository {CLING_REPO_DIR_NAME}")
        execCommand(f"git am -p {ignorePathLevel} {directoryOption}", CLING_REPO_DIR_NAME, patchAsStr)


def syncTagAndPush(oldSyncTag, rootSyncHash):
    """
    Replace the tag mentioning the previous root commit to which the repo was synchronized.
    Push the changes and tags upstream.
    """
    # Push the commits and the new tag together, so that cling never has the
    # commits without a tag that names them. Only then remove the old tag.
    newTag = CLING_TAG_ROOT_HASH_PREFIX + rootSyncHash
    printInfo(f"Creating a new tag ({newTag}).")
    execCommand(f"git tag {newTag}", CLING_REPO_DIR_NAME)
    execCommand(f"git push --atomic origin HEAD tag {newTag}", CLING_REPO_DIR_NAME)

    printInfo(f"Deleting the previous sync tag ({oldSyncTag}).")
    execCommand(f"git tag -d {oldSyncTag}", CLING_REPO_DIR_NAME)
    remoteTags = execCommand("git ls-remote --tags origin", CLING_REPO_DIR_NAME)
    if f"refs/tags/{oldSyncTag}" in remoteTags:
        execCommand(f"git push --delete origin tag {oldSyncTag}", CLING_REPO_DIR_NAME)
    else:
        printWarning(f"Tag {oldSyncTag} was not found in the upstream repository.")


def principal():
    # We want a starting hash not to deal with the entire commit history
    # of ROOT
    rootSyncTag = getRootSyncTag()
    startingRootHash = getStartingRootHash(rootSyncTag)

    # We now get all recent ROOT hashes, as well as the ones relative
    # to commits in the directories we are interested in for the sync
    rootHashes = getHashes(ROOT_REPO_DIR_NAME, startingRootHash)
    interpHashes = getHashes(ROOT_REPO_DIR_NAME, startingRootHash, INTERP_DIR_NAME)

    # If we have no commits to sync, we quit.
    if not interpHashes:
        # nothing to do, we have no commits to sync
        printInfo("No commit to sync. Exiting now.")
        return 0

    printInfo(f"We found:\n - {len(interpHashes)} patches from the directory {INTERP_DIR_NAME}")

    # We now create the patches we want to apply to the cling repo
    patches = createPatches(rootHashes, interpHashes)

    # We now apply the patches
    if not patches:
        printInfo("Cling already has all these commits: only moving the sync tag.")

    # We now need to apply patches, update the tag that mentions the ROOT commit
    # to which the cling repo was synchronised and push everything.
    # First of all we need to acquire an identity
    printInfo("Acquiring an identity in preparation to the upstreaming of the changes")
    execCommand('git config user.email "root@persona.com"', CLING_REPO_DIR_NAME)
    execCommand('git config user.name "Root Persona"', CLING_REPO_DIR_NAME)

    applyPatches(patches)

    syncTagAndPush(rootSyncTag, rootHashes[0])


if __name__ == "__main__":
    sys.exit(principal())
