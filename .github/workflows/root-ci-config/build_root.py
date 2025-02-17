#!/usr/bin/env python3

# pylint: disable=broad-except,missing-function-docstring,line-too-long

"""This mainly functions as a shell script, but python is used for its
   superior control flow. An important requirement of the CI is easily
   reproducible builds, therefore a wrapper is made for running shell
   commands so that they are also logged.

   The log is printed when build fails/succeeds and needs to perfectly
   reproduce the build when pasted into a shell. Therefore all file system
   modifying code not executed from shell needs a shell equivalent
   explicitly appended to the shell log.
      e.g. `os.chdir(x)` requires `cd x` to be appended to the shell log  """

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

import openstack

from build_utils import (
    die,
    github_log_group,
    load_config,
    calc_options_hash,
    subprocess_with_log,
    subprocess_with_capture,
    upload_file,
    is_macos
)
import build_utils

S3CONTAINER = 'ROOT-build-artifacts'  # Used for uploads
S3URL = 'https://s3.cern.ch/swift/v1/' + S3CONTAINER  # Used for downloads

try:
    CONNECTION = openstack.connect(cloud='envvars')
except Exception as exc:
    print("Failed to open the S3 connection:", exc, file=sys.stderr)
    CONNECTION = None

WINDOWS = (os.name == 'nt')
WORKDIR = (os.environ['HOME'] + '/ROOT-CI') if not WINDOWS else 'C:/ROOT-CI'
COMPRESSIONLEVEL = 6 if not WINDOWS else 1


def main():
    # openstack.enable_logging(debug=True)

    # used when uploading artifacts, calculate early since build times are inconsistent
    yyyy_mm_dd = datetime.datetime.today().strftime('%Y-%m-%d')

    args = parse_args()

    build_utils.log = build_utils.Tracer(args.platform, args.dockeropts)

    pull_request = args.head_ref and args.head_ref != args.base_ref

    if not pull_request:
        build_utils.print_info("head_ref same as base_ref, assuming non-PR build")

    cleanup_previous_build()

    # Load CMake options from .github/workflows/root-ci-config/buildconfig/[platform].txt
    this_script_dir = os.path.dirname(os.path.abspath(__file__))

    options_dict = {
        **load_config(f'{this_script_dir}/buildconfig/global.txt'),
        # file below overwrites values from above
        **load_config(f'{this_script_dir}/buildconfig/{args.platform}.txt')
    }

    options = build_utils.cmake_options_from_dict(options_dict)

    if WINDOWS:
        options = "-Thost=x64 " + options

        if args.architecture == 'x86':
            options = "-AWin32 " + options

    # The hash of the build option string is used to find existing artifacts
    # with matching build options on s3 storage.
    options_hash = calc_options_hash(options)

    # Differentiate between macos versions: it's possible to have the same label
    # for different macos versions, especially different minor versions.
    macos_version_prefix = ''
    if is_macos():
        macos_version_tuple = platform.mac_ver()
        macos_version = macos_version_tuple[0]
        macos_version_prefix = f'{macos_version}/'
    platform_machine = platform.machine()

    obj_prefix = f'{args.platform}/{macos_version_prefix}{args.base_ref}/{args.buildtype}_{platform_machine}/{options_hash}'

    # Make testing of CI in forks not impact artifacts
    if 'root-project/root' not in args.repository:
        obj_prefix = f"ci-testing/{args.repository.split('/')[-2]}/" + obj_prefix
        print("Attempting to download")

    if args.incremental:
        try:
            download_artifacts(obj_prefix)
        except Exception as err:
            build_utils.print_warning(f'Failed to download: {err}')
            args.incremental = False

    git_pull("src", args.repository, args.base_ref)

    if pull_request:
      base_head_sha = get_base_head_sha("src", args.repository, args.sha, args.head_sha)

      head_ref_src, _, head_ref_dst = args.head_ref.partition(":")
      head_ref_dst = head_ref_dst or "__tmp"

      rebase("src", "origin", base_head_sha, head_ref_dst, args.head_sha)

    testing: bool = options_dict['testing'].lower() == "on" and options_dict['roottest'].lower() == "on"

    if testing:
      # Where to put the roottest directory
      if os.path.exists(os.path.join(WORKDIR, "src", "roottest", ".git")):
         roottest_dir = "src/roottest"
      else:
         roottest_dir = "roottest"

      # Where to find the target branch
      roottest_origin_repository = re.sub( "/root(.git)*$", "/roottest.git", args.repository)

      # Where to find the incoming branch
      roottest_repository, roottest_head_ref = relatedrepo_GetClosestMatch("roottest", args.pull_repository, args.repository)

      git_pull(roottest_dir, roottest_origin_repository, args.base_ref)

      if pull_request:
        rebase(roottest_dir, roottest_repository, args.base_ref, roottest_head_ref, roottest_head_ref)

    if not WINDOWS:
        show_node_state()

    build(options, args.buildtype)

    # Build artifacts should only be uploaded for full builds, and only for
    # "official" branches (master, v?-??-??-patches), i.e. not for pull_request
    # We also want to upload any successful build, even if it fails testing
    # later on.
    if not pull_request and not args.incremental and not args.coverage:
        archive_and_upload(yyyy_mm_dd, obj_prefix)

    if args.binaries:
        create_binaries(args.buildtype)

    if testing:
        extra_ctest_flags = ""
        if WINDOWS:
            extra_ctest_flags += "--repeat until-pass:5 "
            extra_ctest_flags += "--build-config " + args.buildtype

        ctest_returncode = run_ctest(extra_ctest_flags)

    if args.coverage:
        create_coverage_xml()

    if testing and ctest_returncode != 0:
        handle_test_failure(ctest_returncode)

    print_trace()

def handle_test_failure(ctest_returncode):
    logloc = os.path.join(WORKDIR, "build", "Testing", "Temporary", "LastTestsFailed.log")
    if os.path.isfile(logloc):
        with open(logloc, 'r') as logf:
            print("TEST FAILURES:")
            print(logf.read())
    else:
        print(f'Internal error: cannot find {logloc}\nAdding some debug output:')
        subprocess.run(f'ls -l {WORKDIR}/build', shell=True, check=False, stderr=subprocess.STDOUT)
        subprocess.run(f'ls -l {WORKDIR}/build/Testing', shell=True, check=False, stderr=subprocess.STDOUT)
        subprocess.run(f'ls -l {WORKDIR}/build/Testing/Temporary', shell=True, check=False, stderr=subprocess.STDOUT)

    die(msg=f"TEST FAILURE: ctest exited with code {ctest_returncode}")


def parse_args():
    # it is difficult to use boolean flags from github actions, use strings to convey
    # true/false for boolean arguments instead.
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform",                           help="Platform to build on")
    parser.add_argument("--dockeropts",      default=None,      help="Extra docker options, if any")
    parser.add_argument("--incremental",     default="false",   help="Do incremental build")
    parser.add_argument("--buildtype",       default="Release", help="Release|Debug|RelWithDebInfo")
    parser.add_argument("--coverage",        default="false",   help="Create Coverage report in XML")
    parser.add_argument("--sha",             default=None,      help="sha that triggered the event")
    parser.add_argument("--base_ref",        default=None,      help="Ref to target branch")
    parser.add_argument("--pull_repository", default="",        help="Url to the pull request incoming repository")
    parser.add_argument("--head_ref",        default=None,      help="Ref to feature branch; it may contain a :<dst> part")
    parser.add_argument("--head_sha",        default=None,      help="Sha of commit that triggered the event")
    parser.add_argument("--binaries",        default="false",   help="Whether to create binary artifacts")
    parser.add_argument("--architecture",    default=None,      help="Windows only, target arch")
    parser.add_argument("--repository",      default="https://github.com/root-project/root.git",
                        help="url to repository")

    args = parser.parse_args()

    # Set argument to True if matched
    args.incremental = args.incremental.lower() in ('yes', 'true', '1', 'on')
    args.coverage = args.coverage.lower() in ('yes', 'true', '1', 'on')
    args.binaries = args.binaries.lower() in ('yes', 'true', '1', 'on')

    if not args.base_ref:
        die(os.EX_USAGE, "base_ref not specified")

    return args


def print_trace():
    build_utils.log.print()

@github_log_group("Clean up from previous runs")
def cleanup_previous_build():
    # runners should never have root permissions but be on the safe side
    if WORKDIR in ("", "/"):
        die(1, "WORKDIR not set")

    if WINDOWS:
        # windows
        os.environ['COMSPEC'] = 'powershell.exe'
        result = subprocess_with_log(f"""
            $ErrorActionPreference = 'Stop'
            if (Test-Path {WORKDIR}) {{
                Remove-Item -Recurse -Force -Path {WORKDIR}
            }}
            New-Item -Force -Type directory -Path {WORKDIR}
        """)
    else:
        # mac/linux/POSIX
        result = subprocess_with_log(f"""
            rm -rf {WORKDIR}
            mkdir -p {WORKDIR}
        """)

    if result != 0:
        die(result, "Failed to clean up previous artifacts")


@github_log_group("Pull/clone branch")
def git_pull(directory: str, repository: str, branch: str):
    returncode = 1

    max_attempts = 6
    sleep_time_unit = 3
    for attempt in range(1, max_attempts+1):
        targetdir = os.path.join(WORKDIR, directory)
        if os.path.exists(os.path.join(targetdir, ".git")):
            returncode = subprocess_with_log(f"""
                cd '{targetdir}'
                git checkout {branch}
                git fetch
                git reset --hard @{{u}}
            """)
        else:
            returncode = subprocess_with_log(f"""
                git clone --branch {branch} --single-branch {repository} "{targetdir}"
            """)
        
        if returncode == 0:
            return

        sleep_time = sleep_time_unit * attempt
        build_utils.print_warning(f"""Attempt {attempt}: failed to pull/clone branch. Retrying in {sleep_time} seconds...""")
        time.sleep(sleep_time)

    if returncode != 0:
        die(returncode, f"Failed to pull {branch}")


@github_log_group("Download previous build artifacts")
def download_artifacts(obj_prefix: str):
    try:
        tar_path = build_utils.download_latest(S3URL, obj_prefix, WORKDIR)

        print(f"\nExtracting archive {tar_path}")

        with tarfile.open(tar_path) as tar:
            tar.extractall(WORKDIR)

        build_utils.log.add(f'\ncd {WORKDIR} && tar -xf {tar_path}\n')

    except Exception as err:
        build_utils.print_warning("failed to download/extract:", err)
        shutil.rmtree(os.path.join(WORKDIR, "src"), ignore_errors=True)
        shutil.rmtree(os.path.join(WORKDIR, "build"), ignore_errors=True)
        raise err


@github_log_group("Node state")
def show_node_state() -> None:
    result = subprocess_with_log("""
        which cmake
        cmake --version
        which c++ || true
        c++ --version || true
        uname -a || true
        cat /etc/os-release || true
        sw_vers || true
        uptime || true
        df || true
    """)

    if result != 0:
        build_utils.print_warning("Failed to extract node state")

@github_log_group("Run tests")
def run_ctest(extra_ctest_flags: str) -> int:
    """
    Just return the exit code in case of test failures instead of `die()`-ing; report test
    failures in main().
    """
    builddir = os.path.join(WORKDIR, "build")
    ctest_result = subprocess_with_log(f"""
        cd '{builddir}'
        ctest --output-on-failure --parallel {os.cpu_count()} --output-junit TestResults.xml {extra_ctest_flags}
    """)

    return ctest_result


@github_log_group("Archive and upload")
def archive_and_upload(archive_name, prefix):
    new_archive = f"{archive_name}.tar.gz"

    os.chdir(WORKDIR)

    with tarfile.open(f"{WORKDIR}/{new_archive}", "x:gz", compresslevel=COMPRESSIONLEVEL) as targz:
        targz.add("src")
        targz.add("build")

    upload_file(
        connection=CONNECTION,
        container=S3CONTAINER,
        dest_object=f"{prefix}/{new_archive}",
        src_file=f"{WORKDIR}/{new_archive}"
    )


@github_log_group("Configure")
def cmake_configure(options, buildtype):
    srcdir = os.path.join(WORKDIR, "src")
    builddir = os.path.join(WORKDIR, "build")
    result = subprocess_with_log(f"""
        cmake -S '{srcdir}' -B '{builddir}' -DCMAKE_BUILD_TYPE={buildtype} {options}
    """)

    if result != 0:
        die(result, "Failed cmake generation step")


@github_log_group("Dump existing configuration")
def cmake_dump_config():
    # Print CMake cached config
    srcdir = os.path.join(WORKDIR, "src")
    builddir = os.path.join(WORKDIR, "build")
    result = subprocess_with_log(f"""
        cmake -S '{srcdir}' -B '{builddir}' -N -L
    """)

    if result != 0:
        die(result, "Failed cmake cache print step")


@github_log_group("Dump requested build configuration")
def dump_requested_config(options):
    print(f"\nBUILD OPTIONS: {options}")


@github_log_group("Build")
def cmake_build(buildtype):
    generator_flags = "-- '-verbosity:minimal'" if WINDOWS else ""
    parallel_jobs = "4" if WINDOWS else str(os.cpu_count())

    builddir = os.path.join(WORKDIR, "build")
    result = subprocess_with_log(f"""
        cmake --build '{builddir}' --config '{buildtype}' --parallel '{parallel_jobs}' {generator_flags}
    """)

    if result != 0:
        die(result, "Failed to build")


def build(options, buildtype):
    if not os.path.isdir(os.path.join(WORKDIR, "build")):
        builddir = os.path.join(WORKDIR, "build")
        result = subprocess_with_log(f"mkdir {builddir}")

        if result != 0:
            die(result, "Failed to create build directory")

    if not os.path.exists(os.path.join(WORKDIR, "build", "CMakeCache.txt")):
        cmake_configure(options, buildtype)
    else:
        cmake_dump_config()

    dump_requested_config(options)

    cmake_build(buildtype)


@github_log_group("Create binary packages")
def create_binaries(buildtype):
    builddir = os.path.join(WORKDIR, "build")
    packagedir = os.path.join(WORKDIR, "packages")
    os.makedirs(packagedir, exist_ok=True)
    result = subprocess_with_log(f"""
        cd '{builddir}'
        cpack -B {packagedir} --verbose -C {buildtype}
    """)

    if result != 0:
        die(result, "Failed to generate binary package")


@github_log_group("Rebase")
def rebase(directory: str, repository:str, base_ref: str, head_ref: str, head_sha: str) -> None:
    # rebase fails unless user.email and user.name is set
    targetdir = os.path.join(WORKDIR, directory)
    if (head_sha and head_ref):
      branch = f"{head_sha}:{head_ref}"
    else:
      branch = ""

    result = subprocess_with_log(f"""
        cd '{targetdir}'

        git config user.email "rootci@root.cern"
        git config user.name 'ROOT Continous Integration'

        git fetch {repository} {branch}
        git checkout {head_ref}
        git rebase {base_ref}
    """)

    if result != 0:
        die(result, "Rebase failed")

def get_stdout_subprocess(command: str, error_message: str) -> str:
  """
  get_stdout_subprocess
  execute and log a command.
  capture the stdout, strip white space and return it
  die in case of failed execution unless the error_message is empty.
  """
  result  = subprocess_with_capture(command)
  if result.returncode != 0:
    if error_message != "":
      die(result, error_message)
    else:
      print("\033[90m", end='')
      print(result.stdout)
      print(result.stderr)
      print("\033[0m", end='')
      return ""
  if result.stderr != "":
    print("\033[90m", end='')
    print(result.stdout)
    print(result.stderr)
    print("\033[0m", end='')
  string_result = result.stdout
  string_result = string_result.strip()
  return string_result


@github_log_group("Rebase")
def get_base_head_sha(directory: str, repository: str, merge_sha: str, head_sha: str) -> str:
  """
  get_base_head_sha

  Given a pull request merge commit and the incoming commit return
  the commit corresponding to the head of the branch we are merging into.
  """
  targetdir = os.path.join(WORKDIR, directory)
  command = f"""
      cd '{targetdir}'
      git fetch {repository} {merge_sha}
      """
  result = subprocess_with_log(command)
  if result != 0:
      die("Failed to fetch {merge_sha} from {repository}")
  command = f"""
      cd '{targetdir}'
      git rev-list --parents -1 {merge_sha}
      """
  result = get_stdout_subprocess(command, "Failed to find the base branch head for this pull request")

  for s in result.split(' '):
    if (s != merge_sha and s != head_sha):
      return s

  return ""

@github_log_group("Pull/clone roottest branch")
def relatedrepo_GetClosestMatch(repo_name: str, origin: str, upstream: str):
  """
  relatedrepo_GetClosestMatch(REPO_NAME <repo> ORIGIN_PREFIX <originp> UPSTREAM_PREFIX <upstreamp>
                              FETCHURL_VARIABLE <output_url> FETCHREF_VARIABLE <output_ref>)
  Return the clone URL and head/tag of the closest match for `repo` (e.g. roottest), based on the
  current head name.

  See relatedrepo_GetClosestMatch in toplevel CMakeLists.txt
  """

  # Alternatively, we could use: re.sub( "/root(.git)*$", "", varname)
  origin_prefix = origin[:origin.rfind('/')]
  upstream_prefix = upstream[:upstream.rfind('/')]

  fetch_url = upstream_prefix + "/" + repo_name

  gitdir = os.path.join(WORKDIR, "src", ".git")
  current_head = get_stdout_subprocess(f"""
      git --git-dir={gitdir} rev-parse --abbrev-ref HEAD
      """, "Failed capture of current branch name")

  # `current_head` is a well-known branch, e.g. master, or v6-28-00-patches.  Use the matching branch
  # upstream as the fork repository may be out-of-sync
  branch_regex = re.compile("^(master|latest-stable|v[0-9]+-[0-9]+-[0-9]+(-patches)?)$")
  known_head = branch_regex.match(current_head)

  if known_head:
    if current_head == "latest-stable":
      # Resolve the 'latest-stable' branch to the latest merged head/tag
      current_head = get_stdout_subprocess(f"""
           git --git-dir={gitdir} for-each-ref --points-at=latest-stable^2 --format=%\\(refname:short\\)
           """, "Failed capture of lastest-stable underlying branch name")
      return fetch_url, current_head

  # Otherwise, try to use a branch that matches `current_head` in the fork repository
  matching_refs = get_stdout_subprocess(f"""
       git ls-remote --heads --tags {origin_prefix}/{repo_name} {current_head}
       """, "")
  if matching_refs != "":
    fetch_url = origin_prefix + "/" + repo_name
    return fetch_url, current_head

  # Finally, try upstream using the closest head/tag below the parent commit of the current head
  closest_ref = get_stdout_subprocess(f"""
       git --git-dir={gitdir} describe --all --abbrev=0 HEAD^
       """, "") # Empty error means, ignore errors.
  candidate_head = re.sub("^(heads|tags)/", "", closest_ref)

  matching_refs = get_stdout_subprocess(f"""
       git ls-remote --heads --tags {upstream_prefix}/{repo_name} {candidate_head}
       """, "")
  if matching_refs != "":
    return fetch_url, candidate_head
  return "", ""


@github_log_group("Create Test Coverage in XML")
def create_coverage_xml() -> None:
    builddir = os.path.join(WORKDIR, "build")
    result = subprocess_with_log(f"""
        cd '{builddir}'
        gcovr --output=cobertura-cov.xml --cobertura-pretty --gcov-ignore-errors=no_working_dir_found --merge-mode-functions=merge-use-line-min --exclude-unreachable-branches --exclude-directories="roottest|runtutorials|interpreter" --exclude='.*/G__.*' --exclude='.*/(roottest|runtutorials|externals|ginclude|googletest-prefix|macosx|winnt|geombuilder|cocoa|quartz|win32gdk|x11|x11ttf|eve|fitpanel|ged|gui|guibuilder|guihtml|qtgsi|qtroot|recorder|sessionviewer|tmvagui|treeviewer|geocad|fitsio|gviz|qt|gviz3d|x3d|spectrum|spectrumpainter|dcache|hdfs|foam|genetic|mlp|quadp|splot|memstat|rpdutils|proof|odbc|llvm|test|interpreter)/.*' --gcov-exclude='.*_ACLiC_dict[.].*' '--exclude=.*_ACLiC_dict[.].*' -v -r ../src ../build
    """)

    if result != 0:
        die(result, "Failed to create test coverage")


if __name__ == "__main__":
    main()
