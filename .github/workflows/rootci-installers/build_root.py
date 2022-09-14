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
import re
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


WORKDIR = '/tmp/workspace'
CONTAINER = 'ROOT-build-artifacts'
DEFAULT_BUILDTYPE = 'Release'


def main():
    # openstack.enable_logging(debug=True)
    
    force_generation = False
    platform         = "centos8"
    branch           = "master"
    incremental      = False
    buildtype        = "Release"

    options, _ = getopt.getopt(
        args = sys.argv[1:],
        shortopts = '',
        longopts = ["alwaysgenerate=", "platform=", "branch=", "incremental=", "buildtype="]
    )

    for opt, val in options:
        if opt == "--alwaysgenerate":
            force_generation = val in ('true', '1', 'yes', 'on')
        elif opt == "--platform":
            platform = val
        elif opt == "--branch":
            branch = val
        elif opt == "--incremental":
            incremental = val in ('true', '1', 'yes', 'on')
        elif opt == "--buildtype":
            buildtype = val

    python_script_dir = os.path.dirname(os.path.abspath(__file__))
    yyyymmdd = datetime.datetime.today().strftime('%Y-%m-%d')

    shell_log = ""

    options_dict = {
        **load_config(f'{python_script_dir}/buildconfig/global.txt'),
        # below has precedence
        **load_config(f'{python_script_dir}/buildconfig/{platform}.txt')
    }
    options = cmake_options_from_dict(options_dict)

    option_hash = sha1(options.encode('utf-8')).hexdigest()
    prefix = f'{platform}/{branch}/{buildtype}/{option_hash}'

    # Clean up previous builds
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR)
    os.chdir(WORKDIR)
    shell_log += shortspaced("""
        rm -rf {WORKDIR}
        mkdir -p {WORKDIR}
        cd {WORKDIR}
    """)

    print("\nEstablishing s3 connection")
    connection = openstack.connect('envvars')

    if incremental:
        print("Attempting incremental build")

        # Download and extract previous build artifacts
        try:

            print("\nDownloading")
            tar_path = download_latest(connection, CONTAINER, prefix, WORKDIR)

            print("\nExtracting archive")
            with tarfile.open(tar_path) as tar:
                tar.extractall()
        except Exception as err:
            print_warning(f"failed: {err}")
            incremental = False
        else:
            shell_log += f"""
                wget https://s3.cern.ch/swift/v1/{CONTAINER}/{tar_path} -x -nH --cut-dirs 3
            """

    if incremental:
        # Do git pull and check if build is needed
        result, shell_log = subprocess_with_log(f"""
            cd "{WORKDIR}/src" || exit 3

            git fetch || exit 1

            test "$(git rev-parse HEAD)" = "$(git rev-parse '@{{u}}')" && exit 2

            git merge FETCH_HEAD || exit 1
        """, shell_log)

        if result == 1:
            print_warning("failed to git pull")
            incremental = False
        elif result == 2:
            print("Files are unchanged since last build, exiting")
            exit(0)
        elif result == 3:
            print_warning(f"could not cd {WORKDIR}/src")
            incremental = False

    # Clone and run generation step on non-incremental builds
    if not incremental:
        print("Doing non-incremental build")

        result, shell_log = subprocess_with_log(f"""
            mkdir -p '{WORKDIR}/build'
            mkdir -p '{WORKDIR}/install'

            git clone -b {branch} \
                      --single-branch \
                      --depth 1 \
                      https://github.com/root-project/root.git \
                      {WORKDIR}/src
        """, shell_log)

        if result != 0:
            die(result, "Could not clone from git", shell_log)


    if force_generation or not incremental:
        result, shell_log = subprocess_with_log(f"""
            cmake -S {WORKDIR}/src \
                  -B {WORKDIR}/build \
                  -DCMAKE_INSTALL_PREFIX={WORKDIR}/install \
                    {options}
        """, shell_log)

        if result != 0:
            die(result, "Failed cmake generation step", shell_log)

    # Build
    result, shell_log = subprocess_with_log(f"""
        cmake --build {WORKDIR}/build \
              -- -j"$(getconf _NPROCESSORS_ONLN)"
    """, shell_log)

    if result != 0:
        die(result, "Build step failed", shell_log)

    # Upload and archive
    if connection:
        print("Archiving build artifacts")
        new_archive = f"{yyyymmdd}.tar.gz"
        try:
            with tarfile.open(f"{WORKDIR}/{new_archive}", "x:gz", compresslevel=4) as targz:
                targz.add("src")
                targz.add("install")
                targz.add("build")

            upload_file(
                connection=connection,
                container=CONTAINER,
                name=f"{prefix}/{new_archive}",
                path=f"{WORKDIR}/{new_archive}"
            )
        except tarfile.TarError as err:
            print_warning(f"could not tar artifacts: {err}")
        except Exception as err:
            print_warning(err)

    print_fancy("Script to replicate log:\n")
    print(shell_log)


if __name__ == "__main__":
    main()
