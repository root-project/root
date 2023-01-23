#!/usr/bin/env -S python3 -u

"""This mainly functions as a shell script, but python is used for its
   superior control flow. An important requirement of the CI is easily
   reproducible builds, therefore a wrapper is made for running shell
   commands so that they are also logged.

   The log is printed when build fails/succeeds and needs to perfectly
   reproduce the build when pasted into a shell. Therefore all file system
   modifying code not executed from shell needs a shell equivalent
   explicitly appended to the shell log.
      e.g. `os.chdir(x)` requires `cd x` to be appended to the shell log

   Writing a similar wrapper in bash is difficult because variables are
   expanded before being sent to the log wrapper in hard to predict ways. """

import datetime
import getopt
from hashlib import sha1
import os
import shutil
import sys
import tarfile
import openstack

from build_utils import (
    cmake_options_from_dict,
    die,
    download_latest,
    load_config,
    print_fancy,
    print_warning,
    shortspaced,
    subprocess_with_log,
    upload_file,
)


CONTAINER = 'ROOT-build-artifacts'
DEFAULT_BUILDTYPE = 'Release'


def main():
    shell_log = ''
    yyyy_mm_dd = datetime.datetime.today().strftime('%Y-%m-%d')

    # CLI arguments with defaults
    repository       = 'https://github.com/root-project/root'
    force_generation = False
    platform         = "centos8"
    incremental      = False
    buildtype        = "Release"
    head_ref         = None
    base_ref         = None

    options, _ = getopt.getopt(
        args = sys.argv[1:],
        shortopts = '',
        longopts = ["alwaysgenerate=", "platform=", "incremental=", "buildtype=",
                    "head_ref=", "base_ref=", "repository="]
    )

    for opt, val in options:
        if opt == "--alwaysgenerate":
            force_generation = val in ('true', '1', 'yes', 'on')
        elif opt == "--platform":
            platform = val
        elif opt == "--incremental":
            incremental = val in ('true', '1', 'yes', 'on')
        elif opt == "--buildtype":
            buildtype = val
        elif opt == "--head_ref":
            head_ref = val
        elif opt == "--base_ref":
            base_ref = val
        elif opt == "--repository":
            repository = val
    
    if not base_ref or not head_ref:
        sys.exit(1)

    windows = 'windows' in platform

    if windows:
        workdir = 'C:/ROOT-CI'
    else:
        workdir = '/tmp/workspace'


    # Load CMake options from file
    python_script_dir = os.path.dirname(os.path.abspath(__file__))

    options_dict = {
        **load_config(f'{python_script_dir}/buildconfig/global.txt'),
        # below has precedence
        **load_config(f'{python_script_dir}/buildconfig/{platform}.txt')
    }
    options = cmake_options_from_dict(options_dict)


    # Clean up previous builds
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    os.makedirs(workdir)
    os.chdir(workdir)

    if windows:
        shell_log += shortspaced(f"""
            Remove-Item -Recurse -Force -Path {workdir}
            New-Item -Force -Type directory -Path {workdir}
            Set-Location -LiteralPath {workdir}
        """)
    else:
        shell_log += shortspaced("""
            rm -rf {WORKDIR}
            mkdir -p {WORKDIR}
            cd {WORKDIR}
        """)


    # Attempt openstack connection even on non-incremental builds to upload later
    print("\nEstablishing s3 connection")
    # openstack.enable_logging(debug=True)
    connection = openstack.connect('envvars')


    if incremental:
        print("Attempting incremental build")

        # Download and extract previous build artifacts
        try:
            print("\nDownloading")
            option_hash = sha1(options.encode('utf-8')).hexdigest()
            prefix = f'{platform}/{head_ref}-to-{base_ref}/{buildtype}/{option_hash}'
            tar_path = download_latest(connection, CONTAINER, prefix, workdir)

            print("\nExtracting archive")
            with tarfile.open(tar_path) as tar:
                tar.extractall()
        except Exception as err:
            print_warning(f"failed: {err}")
            incremental = False
        else:
            if windows:
                shell_log += f"(new-object System.Net.WebClient).DownloadFile('https://s3.cern.ch/swift/v1/{CONTAINER}/{tar_path}','{workdir}')"
            else:
                shell_log += f"wget https://s3.cern.ch/swift/v1/{CONTAINER}/{tar_path} -x -nH --cut-dirs 3"

    if incremental:
        # Do git pull and check if build is needed
        result, shell_log = subprocess_with_log(f"""
            cd "{workdir}/src" || exit 3

            git fetch || exit 1

            test "$(git rev-parse HEAD)" = "$(git rev-parse '@{{u}}')" && exit 2

            git merge FETCH_HEAD || exit 1
            
            git rebase {base_ref} {head_ref}
        """, shell_log)

        if result == 1:
            print_warning("failed to git pull")
            incremental = False
        elif result == 2:
            print("Files are unchanged since last build, exiting")
            sys.exit(0)
        elif result == 3:
            print_warning(f"could not cd {workdir}/src")
            incremental = False

    # Clone and run generation step on non-incremental builds
    if not incremental:
        print("Doing non-incremental build")

        if windows:
            result, shell_log = subprocess_with_log(f"""
                New-Item -Force -Type directory -Path {workdir}/build
                New-Item -Force -Type directory -Path {workdir}/install
            """, shell_log)
        else:
            result, shell_log = subprocess_with_log(f"""
                mkdir -p '{workdir}/build'
                mkdir -p '{workdir}/install'
            """, shell_log)

        result, shell_log = subprocess_with_log(f"""
            git clone --depth 1 \
                      {repository} \
                      {workdir}/src
            
            git checkout {base_ref}
            git rebase {base_ref} {head_ref}
        """, shell_log)

        if result != 0:
            die(result, "Could not clone from git", shell_log)


    if force_generation or not incremental:
        result, shell_log = subprocess_with_log(f"""
            cmake -S {workdir}/src \
                  -B {workdir}/build \
                  -DCMAKE_INSTALL_PREFIX={workdir}/install \
                    {options}
        """, shell_log)

        if result != 0:
            die(result, "Failed cmake generation step", shell_log)

    # Build
    cpus = os.cpu_count()

    result, shell_log = subprocess_with_log(f"""
        cmake --build {workdir}/build \
              -- -j"{cpus}"
    """, shell_log)

    if result != 0:
        die(result, "Build step failed", shell_log)

    # Upload and archive
    if connection:
        print("Archiving build artifacts")
        new_archive = f"{yyyy_mm_dd}.tar.gz"
        try:
            with tarfile.open(f"{workdir}/{new_archive}", "x:gz", compresslevel=4) as targz:
                targz.add("src")
                targz.add("install")
                targz.add("build")

            upload_file(
                connection=connection,
                container=CONTAINER,
                name=f"{prefix}/{new_archive}",
                path=f"{workdir}/{new_archive}"
            )
        except tarfile.TarError as err:
            print_warning(f"could not tar artifacts: {err}")
        except Exception as err:
            print_warning(err)

    print_fancy("Script to replicate log:\n")
    print(shell_log)


if __name__ == "__main__":
    main()
