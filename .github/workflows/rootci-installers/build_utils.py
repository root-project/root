#!/usr/bin/env false

import datetime
import re
import textwrap
from typing import Dict, Tuple
import os
import subprocess
import sys
import time

def shortspaced(string) -> str:
    """Replaces multiple spaces with a single space"""
    return re.sub(' +',' ', string)

def print_fancy(*values, sgr=1, **kwargs) -> None:
    """prints message using select graphic rendition, defaults to bold text
       https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters"""

    print(f"\033[{sgr}m", end='', **kwargs)
    print(*values, end='', **kwargs)
    print("\033[0m", **kwargs)


def print_warning(*values, **kwargs):
    print_fancy("Warning: ", *values, sgr=33, file=sys.stderr, **kwargs)


def print_error(*values, **kwargs):
    print_fancy("Fatal error: ", *values, sgr=31, file=sys.stderr, **kwargs)


def subprocess_with_log(command: str, log="", debug=True) -> Tuple[int, str]:
    """Runs <command> in shell and appends <command> to log"""

    start: float = 0.0
    if debug:
        print_fancy(command)
        start = time.time()

    print("\033[0m", end='')
    result = subprocess.run(command, shell=True, check=False)

    if debug:
        elapsed = datetime.timedelta(seconds=time.time() - start)
        print_fancy(f"\nFinished expression in {elapsed}\n", sgr=3)

    return (result.returncode,
            log + '\n(\n' + shortspaced(command) + '\n)')


def die(code: int, msg: str, log: str = "") -> None:
    """prints error code, message and exits"""
    print_error(f"({code}) {msg}")

    if log != "":
        error_msg = textwrap.dedent(f"""
            ######################################
            #    To replicate build locally     #
            ######################################
            
            {log}
        """)

        print_error(error_msg)

        try:
            with open("/etc/motd", "w", encoding="ascii") as f:
                f.write(error_msg)
        except Exception:
            pass

    sys.exit(code)


def load_config(filename) -> dict:
    """Loads cmake options from a file to a dictionary"""

    options = {}

    try:
        file = open(filename, 'r', encoding='utf-8')
    except OSError as err:
        print_warning(f"couldn't load {filename}: {err.strerror}")
    else:
        with file:
            for line in file:
                if '=' not in line:
                    continue

                key, val = line.rstrip().split('=')

                if val.lower() in ["on", "off"]:
                    val = val.lower()

                options[key] = val

    return options


def cmake_options_from_dict(config: Dict[str, str]) -> str:
    """Converts a dictionary of build options to string.
       The output is sorted alphanumerically.

       example: {"builtin_xrootd"="on", "alien"="on"}
                 -> '"-Dalien=on" -Dbuiltin_xrootd=on"'
    """

    if not config:
        return ''

    output = []

    for key, value in config.items():
        output.append(f'"-D{key}={value}"')

    output.sort()

    return ' '.join(output)


def upload_file(connection, container: str, name: str, path: str) -> None:
    """Uploads file to s3 object storage."""

    print(f"Attempting to upload {path} to {name}")

    if not os.path.exists(path):
        raise Exception(f"No such file: {path}")

    gigabyte = 1073741824
    # week_in_seconds = 604800

    connection.create_object(
        container,
        name,
        path,
        segment_size=2*gigabyte
        # **{'X-Delete-After':week_in_seconds}
    )

    print(f"Successfully uploaded to {name}")


def download_file(connection, container: str, name: str, destination: str) -> None:
    """Downloads a file from s3 object storage"""

    print(f"\nAttempting to download {name} to {destination}")

    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    with open(destination, 'wb') as file:
        connection.get_object(container, name, outfile=file)


def download_latest(connection, container: str, prefix: str, destination: str) -> str:
    """Downloads latest build artifact tar starting with <prefix>
       and returns its path.

       Outputs a link to the file to stdout"""

    objects = connection.list_objects(container, prefix=prefix)

    if not objects:
        raise Exception(f"No object found with prefix: {prefix}")

    artifacts = [obj.name for obj in objects]
    latest = max(artifacts)
    file = latest.split(".tar.gz")[0] + ".tar.gz"  # < ugly fix because files
                                                   # are sometimes segmented in s3
                                                   # so that they end in *.tar.gz/001

    download_file(connection, container, file, f"{destination}/{file}")

    return f"{destination}/{file}"
