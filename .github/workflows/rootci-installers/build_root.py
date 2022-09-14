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

import datetime
from hashlib import sha1
import os
import re
import argparse
import shutil
import sys
import tarfile
import openstack

from build_utils import (
    cmake_options_from_dict,
    die,
    download_latest,
    load_config,
    print_shell_log,
    warning,
    subprocess_with_log,
    upload_file,
)

S3CONTAINER = 'ROOT-build-artifacts'

def main():
    # openstack.enable_logging(debug=True)
    shell_log = ''
    yyyy_mm_dd = datetime.datetime.today().strftime('%Y-%m-%d')
    python_script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform",    default="centos8", help="Platform to build on")
    parser.add_argument("--incremental", default=False,     help="Do incremental build")
    parser.add_argument("--buildtype",   default="Release", help="Release, Debug or RelWithDebInfo")
    parser.add_argument("--base_ref",    default=None,      help="Ref to target branch")
    parser.add_argument("--head_ref",    default=None,      help="Ref to feature branch")
    parser.add_argument("--repository",  default="https://root-project/root",
                        help="url to repository")

    args = parser.parse_args()

    platform    = args.platform
    incremental = args.incremental.lower() in ('yes', 'true', '1', 'on')
    buildtype   = args.buildtype
    base_ref    = args.base_ref
    head_ref    = args.head_ref
    repository  = args.repository

    if not base_ref:
        die(1, "base_ref not specified")

    if not head_ref or (head_ref == base_ref):
        warning("head_ref not specified or same as base_ref, building base_ref only")
        rebase = False
        head_ref = base_ref
    else:
        rebase = True

    if os.name == 'nt':
        # windows
        compressionlevel = 1
        workdir = 'C:/ROOT-CI'
        os.environ['COMSPEC'] = 'powershell.exe'
        result, shell_log = subprocess_with_log(f"""
            Remove-Item -Recurse -Force -Path {workdir}
            New-Item -Force -Type directory -Path {workdir}
            Set-Location -LiteralPath {workdir}
        """, shell_log)
    else:
        # mac/linux/POSIX
        compressionlevel = 6
        workdir = '/tmp/workspace'
        result, shell_log = subprocess_with_log(f"""
            mkdir -p {workdir}
            rm -rf {workdir}/*
            cd {workdir}
        """, shell_log)

    if result != 0:
        die(result, "Failed to clean up previous artifacts", shell_log)

    os.chdir(workdir)
    shell_log += f"\ncd {workdir}\n"

    # Load CMake options from file
    options_dict = {
        **load_config(f'{python_script_dir}/buildconfig/global.txt'),
        # below has precedence
        **load_config(f'{python_script_dir}/buildconfig/{platform}.txt')
    }
    options = cmake_options_from_dict(options_dict)

    option_hash = sha1(options.encode('utf-8')).hexdigest()
    s3_prefix = f'{platform}/{base_ref}/{buildtype}/{option_hash}'

    print("\nEstablishing s3 connection")
    connection = openstack.connect('envvars')
    # without openstack we can't run test workflow, might as well give up here ¯\_(ツ)_/¯
    if not connection:
        die(msg="Could not connect to OpenStack")

    # Download and extract previous build artifacts
    if incremental:
        print("Attempting incremental build")

        try:
            print("\nDownloading")
            tar_path, shell_log = download_latest(connection, S3CONTAINER, s3_prefix, workdir, shell_log)

            print("\nExtracting archive")
            with tarfile.open(tar_path) as tar:
                tar.extractall()
            shell_log += f'\ntar -xf {tar_path}\n'
        except Exception as err:
            warning("failed to download/extract:", err)
            shutil.rmtree(f'{workdir}/src', ignore_errors=True)
            shutil.rmtree(f'{workdir}/build', ignore_errors=True)

            incremental = False


    # Add remote on non incremental builds
    if not incremental:
        print("Doing non-incremental build")

        result, shell_log = subprocess_with_log(f"""
            git clone --branch {base_ref} --single-branch {repository} "{workdir}/src"
        """, shell_log)

        if result != 0:
            die(result, "Failed to pull", shell_log)


    # First: fetch, build and upload base branch. Skipped if existing artifacts
    # are up to date with upstream
    #
    # Makes some builds marginally slower but populates the artifact storage
    # which makes subsequent builds much much faster
    result, shell_log = subprocess_with_log(f"""
        cd '{workdir}/src' || exit 1
        
        git checkout {base_ref} || exit 2
        
        git fetch
        
        echo "$(git rev-parse HEAD)" = "$(git rev-parse '@{{u}}')"
        
        if [ "$(git rev-parse HEAD)" = "$(git rev-parse '@{{u}}')" ]; then
            exit 123
        fi
        
        git reset --hard @{{u}} || exit 4
    """, shell_log)

    if result not in (0, 123):
        die(result, f"Failed to pull {base_ref}", shell_log)

    if not incremental or result != 123:
        shell_log = build_base(base_ref, incremental, workdir, options, buildtype, shell_log)

        release_branches = r'master|latest-stable|v.+?-.+?-.+?-patches'

        if not re.match(release_branches, base_ref):
            warning("{base_ref} is not a release branch, skipping artifact upload")
        else:
            try:
                print(f"Archiving build artifacts of {base_ref}")
                archive_and_upload(yyyy_mm_dd, workdir, connection, compressionlevel, s3_prefix)
            except Exception as err:
                warning("failed to archive/upload artifacts: ", err)
        print(f"Successfully built branch {base_ref}")

    if not rebase:
        print(f"Successfully built {base_ref}!")
        print_shell_log(shell_log)
        sys.exit(0)

    shell_log = rebase_and_build(base_ref, head_ref, buildtype, workdir, shell_log)

    try:
        print("Archiving build artifacts to run tests in a new workflow")
        if "pull" in head_ref:
            name = "pr-" + head_ref.split('/')[-2]
        else:
            name = head_ref
        test_prefix = f'to-test/{name}/{platform}/{buildtype}/{option_hash}'
        archive_and_upload(f"test{yyyy_mm_dd}", workdir, connection, compressionlevel, test_prefix)
    except Exception as err:
        warning("failed to archive/upload artifacts: ", err)

    print_shell_log(shell_log)


def archive_and_upload(archive_name, workdir, connection, compressionlevel, prefix):
    new_archive = f"{archive_name}.tar.gz"

    with tarfile.open(f"{workdir}/{new_archive}", "x:gz", compresslevel=compressionlevel) as targz:
        targz.add("src")
        targz.add("build")

    upload_file(
        connection=connection,
        container=S3CONTAINER,
        name=f"{prefix}/{new_archive}",
        path=f"{workdir}/{new_archive}"
    )


def build_base(base_ref, incremental, workdir, options, buildtype, shell_log) -> str:
    if not incremental:
        result, shell_log = subprocess_with_log(f"""
            mkdir -p '{workdir}/build'
            cmake -S '{workdir}/src' -B '{workdir}/build' {options}
        """, shell_log)

        if result != 0:
            die(result, "Failed cmake generation step", shell_log)

    result, shell_log = subprocess_with_log(f"""
        mkdir -p {workdir}/build
        cmake --build '{workdir}/build' --config '{buildtype}' --parallel '{os.cpu_count()}'
    """, shell_log)

    if result != 0:
        die(result, f"Failed to build {base_ref}", shell_log)

    return shell_log


def rebase_and_build(base_ref, head_ref, buildtype, workdir, shell_log) -> str:
    """rebases and builds head_ref on base_ref, returns shell log"""

    print(f"Rebasing {head_ref} onto {base_ref}...")

    result, shell_log = subprocess_with_log(f"""
        cd '{workdir}/src' || exit 1
            
        git config user.email "rootci@root.cern"
        git config user.name 'ROOT Continous Integration'
        
        git fetch origin {head_ref}:head || exit 2
        git checkout head || exit 3
        
        git rebase {base_ref} || exit 5
    """, shell_log)

    if result != 0:
        die(result, "Rebase failed", shell_log)

    print("Building changes...")
    result, shell_log = subprocess_with_log(f"""
        mkdir -p {workdir}/build
        cmake --build '{workdir}/build' --config '{buildtype}' --parallel '{os.cpu_count()}'
    """, shell_log)

    if result != 0:
        die(result, "Build step after rebase failed", shell_log)

    print(f"Rebase and build of {head_ref} onto {base_ref} successful!")

    return shell_log


if __name__ == "__main__":
    main()
