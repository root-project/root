#!/usr/bin/env false
# avoid running on unix^^

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
from hashlib import sha1
import os
import shutil
import tarfile
import openstack

from build_utils import (
    die,
    download_latest,
    load_config,
    cmake_options_from_dict,
    print_fancy,
    print_warning,
    subprocess_with_log,
    upload_file
)


WORKDIR = "C:/ROOT-CI"
CONTAINER = "ROOT-build-artifacts"
DEFAULT_BUILDTYPE = 'Release'


def main():
    # openstack.enable_logging(debug=True)
    python_script_dir = os.path.dirname(os.path.abspath(__file__))
    yyyymmdd = datetime.datetime.today().strftime('%Y-%m-%d')

    shell_log = ""

    platform = os.environ['PLATFORM']
    branch = os.environ['BRANCH']
    incremental = os.environ['INCREMENTAL'].lower() in ['true', 'yes', 'on']

    options_dict = {
        **load_config(f'{python_script_dir}/buildconfig/global.txt'),
        # has precedence
        **load_config(f'{python_script_dir}/buildconfig/{platform}.txt')
    }
    buildtype = options_dict.get('CMAKE_BUILD_TYPE', DEFAULT_BUILDTYPE)
    options = cmake_options_from_dict(options_dict)

    option_hash = sha1(options.encode('utf-8')).hexdigest()
    prefix = f'{platform}/{branch}/{buildtype}/{option_hash}'

    # Clean up previous builds
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR)
    os.chdir(WORKDIR)
    shell_log += f"""
        Remove-Item -Recurse -Force -Path {WORKDIR}
        New-Item -Force -Type directory -Path {WORKDIR}
        Set-Location -LiteralPath {WORKDIR}
    """

    connection = None

    if incremental:
        print("Attempting incremental build")

        # Download and extract previous build artifacts
        try:
            print("\nEstablishing s3 connection")
            connection = openstack.connect('envvars')

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
                (new-object System.Net.WebClient).DownloadFile('https://s3.cern.ch/swift/v1/{CONTAINER}/{tar_path}','{WORKDIR}')
            """

    if incremental:
        # Do git pull and check if build is needed
        result, shell_log = subprocess_with_log(f"""
            cd {WORKDIR}/src || return 3

            git fetch || return 1

            if( "$(git rev-parse HEAD)" -eq "$(git rev-parse '@{{u}}')" ){{
                return 2
            }}

            git merge FETCH_HEAD || return 1
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
            New-Item -Force -Type directory -Path {WORKDIR}/build
            New-Item -Force -Type directory -Path {WORKDIR}/install

            git clone -b {branch} \
                      --single-branch \
                      --depth 1 \
                      https://github.com/root-project/root.git \
                      {WORKDIR}/src
        """, shell_log)

        if result != 0:
            die(result, "Could not clone from git", shell_log)

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
              --target install \
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
