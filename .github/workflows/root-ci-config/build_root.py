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
import shutil
import tarfile
from hashlib import sha1

import openstack

from build_utils import (
    cmake_options_from_dict,
    die,
    download_latest,
    github_log_group,
    print_info,
    load_config,
    print_shell_log,
    subprocess_with_log,
    upload_file,
    print_warning,
)

S3CONTAINER = 'ROOT-build-artifacts'  # Used for uploads
S3URL = 'https://s3.cern.ch/swift/v1/' + S3CONTAINER  # Used for downloads

try:
    CONNECTION = openstack.connect(cloud='envvars')
except:
    CONNECTION = None

WINDOWS = (os.name == 'nt')
WORKDIR = '/tmp/workspace' if not WINDOWS else 'C:/ROOT-CI'
COMPRESSIONLEVEL = 6 if not WINDOWS else 1


def main():
    # openstack.enable_logging(debug=True)

    # accumulates commands executed so they can be displayed as a script on build failure
    shell_log = ""

    # used when uploading artifacts, calculate early since build times are inconsistent
    yyyy_mm_dd = datetime.datetime.today().strftime('%Y-%m-%d')

    # it is difficult to use boolean flags from github actions, use strings to convey
    # true/false for boolean arguments instead.
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform",     default="centos8", help="Platform to build on")
    parser.add_argument("--incremental",  default="false",   help="Do incremental build")
    parser.add_argument("--buildtype",    default="Release", help="Release|Debug|RelWithDebInfo")
    parser.add_argument("--base_ref",     default=None,      help="Ref to target branch")
    parser.add_argument("--head_ref",     default=None,      help="Ref to feature branch; it may contain a :<dst> part")
    parser.add_argument("--architecture", default=None,      help="Windows only, target arch")
    parser.add_argument("--repository",   default="https://github.com/root-project/root.git",
                        help="url to repository")

    args = parser.parse_args()

    args.incremental = args.incremental.lower() in ('yes', 'true', '1', 'on')

    if not args.base_ref:
        die(os.EX_USAGE, "base_ref not specified")

    pull_request = args.head_ref and args.head_ref != args.base_ref

    if not pull_request:
        print_info("head_ref same as base_ref, assuming non-PR build")

    shell_log = cleanup_previous_build(shell_log)

    # Load CMake options from .github/workflows/root-ci-config/buildconfig/[platform].txt
    this_script_dir = os.path.dirname(os.path.abspath(__file__))

    options_dict = {
        **load_config(f'{this_script_dir}/buildconfig/global.txt'),
        # file below overwrites values from above
        **load_config(f'{this_script_dir}/buildconfig/{args.platform}.txt')
    }

    options = cmake_options_from_dict(options_dict)

    if WINDOWS:
        options = "-Thost=x64 " + options

        if args.architecture == 'x86':
            options = "-AWin32 " + options

    # The sha1 of the build option string is used to find existing artifacts
    # with matching build options on s3 storage.
    option_hash = sha1(options.encode('utf-8')).hexdigest()
    obj_prefix = f'{args.platform}/{args.base_ref}/{args.buildtype}/{option_hash}'

    # Make testing of CI in forks not impact artifacts
    if 'root-project/root' not in args.repository:
        obj_prefix = f"ci-testing/{args.repository.split('/')[-2]}/" + obj_prefix
        print("Attempting to download")

    if args.incremental:
        try:
            shell_log += download_artifacts(obj_prefix, shell_log)
        except Exception as err:
            print_warning(f'Failed to download: {err}')
            args.incremental = False

    shell_log = git_pull(args.repository, args.base_ref, shell_log)

    if pull_request:
        shell_log = rebase(args.base_ref, args.head_ref, shell_log)

    if not WINDOWS:
        shell_log = show_node_state(shell_log, options)

    shell_log = build(options, args.buildtype, shell_log)

    # Build artifacts should only be uploaded for full builds, and only for
    # "official" branches (master, v?-??-??-patches), i.e. not for pull_request
    # We also want to upload any successful build, even if it fails testing
    # later on.
    try:
        if not pull_request and not args.incremental:
            archive_and_upload(yyyy_mm_dd, obj_prefix)
    except:
        testing: bool = options_dict['testing'].lower() == "on" and options_dict['roottest'].lower() == "on"

        if testing:
            extra_ctest_flags = ""

            if WINDOWS:
                extra_ctest_flags += "--repeat until-pass:3 "
                extra_ctest_flags += "--build-config " + args.buildtype

            shell_log = run_ctest(shell_log, extra_ctest_flags)
        shell_log = create_coverage(shell_log)

        print_shell_log(shell_log)


@github_log_group("Clean up from previous runs")
def cleanup_previous_build(shell_log):
    # runners should never have root permissions but be on the safe side
    if WORKDIR == "" or WORKDIR == "/":
        die(1, "WORKDIR not set", "")

    if WINDOWS:
        # windows
        os.environ['COMSPEC'] = 'powershell.exe'
        result, shell_log = subprocess_with_log(f"""
            $ErrorActionPreference = 'Stop'
            if (Test-Path {WORKDIR}) {{
                Remove-Item -Recurse -Force -Path {WORKDIR}
            }}
            New-Item -Force -Type directory -Path {WORKDIR}
        """, shell_log)
    else:
        # mac/linux/POSIX
        result, shell_log = subprocess_with_log(f"""
            rm -rf {WORKDIR}
            mkdir -p {WORKDIR}
        """, shell_log)

    if result != 0:
        die(result, "Failed to clean up previous artifacts", shell_log)

    return shell_log


@github_log_group("Pull/clone branch")
def git_pull(repository: str, branch: str, shell_log: str):
    returncode = 1

    for attempts in range(5):
        if returncode == 0:
            break

        if os.path.exists(f"{WORKDIR}/src/.git"):
            returncode, shell_log = subprocess_with_log(f"""
                cd '{WORKDIR}/src'
                git checkout {branch}
                git fetch
                git reset --hard @{{u}}
            """, shell_log)
        else:
            returncode, shell_log = subprocess_with_log(f"""
                git clone --branch {branch} --single-branch {repository} "{WORKDIR}/src"
            """, shell_log)

    if returncode != 0:
        die(returncode, f"Failed to pull {branch}", shell_log)

    return shell_log


@github_log_group("Download previous build artifacts")
def download_artifacts(obj_prefix: str, shell_log: str):
    try:
        tar_path, shell_log = download_latest(S3URL, obj_prefix, WORKDIR, shell_log)

        print(f"\nExtracting archive {tar_path}")

        with tarfile.open(tar_path) as tar:
            tar.extractall(WORKDIR)

        shell_log += f'\ntar -xf {tar_path}\n'

    except Exception as err:
        print_warning("failed to download/extract:", err)
        shutil.rmtree(f'{WORKDIR}/src', ignore_errors=True)
        shutil.rmtree(f'{WORKDIR}/build', ignore_errors=True)
        raise err

    return shell_log


@github_log_group("Node state")
def show_node_state(shell_log: str, options: str) -> str:
    result, shell_log = subprocess_with_log("""
        which cmake
        cmake --version
        which c++ || true
        c++ --version || true
        uname -a || true
        cat /etc/os-release || true
        sw_vers || true
        uptime || true
        df || true
    """, shell_log)

    if result != 0:
        print_warning("Failed to extract node state")

    return shell_log

#Even in case of some failed tests I am still making it run
@github_log_group("Run tests")
def run_ctest(shell_log: str, extra_ctest_flags: str) -> str:
    result, shell_log = subprocess_with_log(f"""
        cd '{WORKDIR}/build'
        ctest --output-on-failure --parallel {os.cpu_count()} --output-junit TestResults.xml {extra_ctest_flags}
    """, shell_log)

    # if result != 0:
    #     die(result, "Some tests failed", shell_log)
    print(result)

    return shell_log

# @github_log_group("Create Test Coverage")
# def create_coverage():
#     directory = f"{WORKDIR}/builddir/interpreter/llvm-project/llvm/lib/ProfileData/Coverage/CmakeFiles"
#     contents = os.listdir(directory)
#     return contents

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


@github_log_group("Build")
def build(options, buildtype, shell_log):
    generator_flags = "-- '-verbosity:minimal'" if WINDOWS else ""

    if not os.path.isdir(f'{WORKDIR}/build'):
        result, shell_log = subprocess_with_log(f"mkdir {WORKDIR}/build", shell_log)

        if result != 0:
            die(result, "Failed to create build directory", shell_log)

    if not os.path.exists(f'{WORKDIR}/build/CMakeCache.txt'):
        result, shell_log = subprocess_with_log(f"""
            cmake -S '{WORKDIR}/src' -B '{WORKDIR}/build' {options} -DCMAKE_BUILD_TYPE={buildtype}
        """, shell_log)

        if result != 0:
            die(result, "Failed cmake generation step", shell_log)
    else:
        # Print CMake cached config
        result, shell_log = subprocess_with_log(f"""
            cmake -S '{WORKDIR}/src' -B '{WORKDIR}/build' -N -L
        """, shell_log)

        if result != 0:
            die(result, "Failed cmake cache print step", shell_log)

    shell_log += f"\nBUILD OPTIONS: {options}"

    result, shell_log = subprocess_with_log(f"""
        cmake --build '{WORKDIR}/build' --config '{buildtype}' --parallel '{os.cpu_count()}' {generator_flags}
    """, shell_log)

    if result != 0:
        die(result, "Failed to build", shell_log)

    return shell_log


@github_log_group("Rebase")
def rebase(base_ref, head_ref, shell_log) -> str:
    head_ref_src, _, head_ref_dst = head_ref.partition(":")
    head_ref_dst = head_ref_dst or "__tmp"
    # rebase fails unless user.email and user.name is set
    result, shell_log = subprocess_with_log(f"""
        cd '{WORKDIR}/src'

        git config user.email "rootci@root.cern"
        git config user.name 'ROOT Continous Integration'

        git fetch origin {head_ref_src}:{head_ref_dst}
        git checkout {head_ref_dst}
        git rebase {base_ref}
    """, shell_log)

    if result != 0:
        die(result, "Rebase failed", shell_log)

    return shell_log


@github_log_group("Create Test Coverage")
def create_coverage(shell_log: str) -> str:
    directory = f"{WORKDIR}/build/interpreter/llvm-project/llvm/lib/ProfileData/Coverage/CMakeFiles"
    #
    #directory = f"../root-ci-config/"
    #directory = f"builddir/interpreter/llvm-project/llvm/lib/ProfileData/Coverage/CMakeFiles"
    result, shell_log = subprocess_with_log(f"""
        cd '{directory}'
        lcov --directory . --capture --output-file coverage.info
        genhtml coverage.info --output-directory coverage_report
        cd coverage_report
        firefox index.html
    """, shell_log)
    if directory == "":
        print("No content")
    #contents = os.listdir(directory)
    print(result)
    print("---------")
    print(shell_log)
    return shell_log


if __name__ == "__main__":
    main()
