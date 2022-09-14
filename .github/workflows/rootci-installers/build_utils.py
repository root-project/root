#!/usr/bin/env false

import textwrap
from typing import Dict, Tuple
import os
import subprocess
import sys

def print_fancy(*values, sgr=1, **kwargs) -> None:
    """prints message using select graphic rendition, defaults to bold text
       https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters"""

    print(f"\033[{sgr}m", end='', **kwargs)
    print(*values, end='', **kwargs)
    print("\033[0m", **kwargs)


def warning(*values, **kwargs):
    print_fancy("Warning: ", *values, sgr=33, file=sys.stderr, **kwargs)


def error(*values, **kwargs):
    print_fancy("Fatal error: ", *values, sgr=31, file=sys.stderr, **kwargs)


def subprocess_with_log(command: str, log="") -> Tuple[int, str]:
    """Runs <command> in shell and appends <command> to log"""

    print_fancy(command, sgr=90)

    print("\033[0m", end='')
    print("\033[90m", end='')

    result = subprocess.run(command, shell=True, check=False)


    return (result.returncode,
            log + '\n(\n' + textwrap.dedent(command.strip()) + '\n)')


def die(code: int = 1, msg: str = "", log: str = "") -> None:
    error(f"({code}) {msg}")

    print_shell_log(log)

    sys.exit(code)


def print_shell_log(log: str) -> None:
    if log != "":
        shell_log = textwrap.dedent(f"""\
            ######################################
            #    To replicate build locally     #
            ######################################
            
            {log}
        """)

        print(shell_log)



def load_config(filename) -> dict:
    """Loads cmake options from a file to a dictionary"""

    options = {}

    try:
        file = open(filename, 'r', encoding='utf-8')
    except OSError as err:
        warning(f"couldn't load {filename}: {err.strerror}")
        return {}

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
                        ->
                 '"-Dalien=on" -Dbuiltin_xrootd=on"'
    """

    if not config:
        return ''

    output = []

    for key, value in config.items():
        output.append(f'"-D{key}={value}"')

    output.sort()

    return ' '.join(output)


def upload_file(connection, container: str, name: str, path: str) -> None:
    print(f"Attempting to upload {path} to {name}", file=sys.stderr)

    if not os.path.exists(path):
        raise Exception(f"No such file: {path}")

    gigabyte = 1024*1024*1024
    week_in_seconds = 60*60*24*7

    connection.create_object(
        container,
        name,
        path,
        segment_size=5*gigabyte,
        **{
            'X-Delete-After':str(2*week_in_seconds)
        }
    )

    print(f"Successfully uploaded to {name}")


def download_file(connection, container: str, name: str, destination: str) -> None:
    print(f"\nAttempting to download {name} to {destination}", file=sys.stderr)

    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    with open(destination, 'wb') as file:
        connection.get_object(container, name, outfile=file)


def download_latest(connection, container: str, prefix: str, destination: str, shell_log: str) -> str:
    """Downloads latest build artifact starting with <prefix>,
       and returns the file path to the downloaded file."""

    objects = connection.list_objects(container, prefix=prefix)

    if not objects:
        raise Exception(f"No object found with prefix: {prefix}")

    artifacts = [obj.name for obj in objects]
    latest = max(artifacts)
    file = latest.split(".tar.gz")[0] + ".tar.gz"  # < ugly fix because files
                                                   # are sometimes segmented in openstack s3
                                                   # so that they end in *.tar.gz/001, *.tar.gz/002 etc.

    download_file(connection, container, file, f"{destination}/{file}")

    if os.name == 'nt':
        shell_log += f"\n(new-object System.Net.WebClient).DownloadFile('https://s3.cern.ch/swift/v1/{container}/{file}','{destination}')\n"
    else:
        shell_log += f"\nwget https://s3.cern.ch/swift/v1/{container}/{file} -x -nH --cut-dirs 3\n"

    return f"{destination}/{file}", shell_log
