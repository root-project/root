#!/usr/bin/env false

import json
import os
import subprocess
import sys
import textwrap
from functools import wraps
from http import HTTPStatus
from typing import Callable, Dict
from collections import namedtuple

from openstack.connection import Connection
from requests import get


def github_log_group(title: str):
    """ decorator that places function's stdout/stderr output in a
        dropdown group when running on github workflows """
    def group(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n::group::{title}\n")

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                print("\n::endgroup::\n")
                raise exc

            print("\n::endgroup::\n")

            return result

        return wrapper if os.getenv("GITHUB_ACTIONS") else func

    return group


class Tracer:
    """
    Trace command invocations and print them to reproduce builds.
    """

    image = ""
    docker_opts = []
    trace = ""

    def __init__(self, image: str, docker_opts: str):
        self.image = image
        if docker_opts:
            self.docker_opts = docker_opts.split(' ')
        if '--rm' in self.docker_opts:
            self.docker_opts.remove('--rm')

    def add(self, command: str) -> None:
        self.trace += '\n(\n' + textwrap.dedent(command.strip()) + '\n)'

    @github_log_group("To replicate this build locally")
    def print(self) -> None:
        if self.trace != "":
            if self.image:
                print(f"""\
Grab the image:
$ docker run {' '.join(self.docker_opts)} -it {self.image}
Then:
""")
            print(self.trace)


log = Tracer("", "")


def print_fancy(*values, sgr=1, **kwargs) -> None:
    """prints message using select graphic rendition, defaults to bold text
       https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters"""

    print(f"\033[{sgr}m", end='', **kwargs)
    print(*values, end='', **kwargs)
    print("\033[0m", **kwargs)


def print_info(*values, **kwargs):
    print_fancy("Info: ", *values, sgr=90, **kwargs)


def print_warning(*values, **kwargs):
    print_fancy("Warning: ", *values, sgr=33, **kwargs)


def print_error(*values, **kwargs):
    print_fancy("Fatal error: ", *values, sgr=31, **kwargs)


def subprocess_with_log(command: str) -> int:
    """Runs <command> in shell and appends <command> to log"""

    print_fancy(textwrap.dedent(command), sgr=1)

    print("\033[90m", end='')

    if os.name == 'nt':
        command = "$env:comspec = 'cmd.exe'; " + command

    result = subprocess.run(command, shell=True, check=False, stderr=subprocess.STDOUT)

    print("\033[0m", end='')

    log.add(command)

    return result.returncode

def subprocess_with_capture(command: str):
    """Runs <command> in shell, capture output and appends <command> to log"""

    print_fancy(textwrap.dedent(command), sgr=1)

    print("\033[90m", end='')

    if os.name == 'nt':
        command = "$env:comspec = 'cmd.exe'; " + command

    result = subprocess.run(command, capture_output=True, text=True, shell=True, check=False)

    print(result.stdout)
    print(result.stderr)
    print("\033[0m", end='')

    # Since we are capturing the result and using it in other command later,
    # we don't need it for the reproducing steps.
    # So no call to: log.add(command)

    return result


def die(code: int = 1, msg: str = "") -> None:
    log.print()

    print_error(f"({code}) {msg}")

    sys.exit(code)


def load_config(filename) -> dict:
    """Loads cmake options from a file to a dictionary"""

    options = {}

    try:
        file = open(filename, 'r', encoding='utf-8')
    except OSError as err:
        print_warning(f"couldn't load {filename}: {err.strerror}")
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


def upload_file(connection: Connection, container: str, dest_object: str, src_file: str) -> None:
    print(f"Attempting to upload {src_file} to {dest_object}")

    if not os.path.exists(src_file):
        raise Exception(f"No such file: {src_file}")

    gigabyte = 1024**3
    week_in_seconds = 60*60*24*7

    connection.create_object(
        container=container,
        name=dest_object,
        filename=src_file,
        segment_size=5*gigabyte,
        **{
            'X-Delete-After': str(2*week_in_seconds)
        }
    )

    print(f"Successfully uploaded to {dest_object}")


def download_file(url: str, dest: str) -> None:
    print(f"\nAttempting to download {url} to {dest}")

    parent_dir = os.path.dirname(dest)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(dest, 'wb') as fout, get(url, timeout=300) as req:
        fout.write(req.content)


def download_latest(url: str, prefix: str, destination: str) -> str:
    """Downloads latest build artifact starting with <prefix>,
       and returns the file path to the downloaded file and shell_log."""

    # https://docs.openstack.org/api-ref/object-store/#show-container-details-and-list-objects
    with get(f"{url}/?prefix={prefix}&format=json", timeout=20) as req:
        if req.status_code == HTTPStatus.NO_CONTENT or req.content == b'[]':
            raise Exception(f"No object found with prefix: {prefix}")

        result = json.loads(req.content)
        artifacts = [x['name'] for x in result if 'content_type' in x]

    latest = max(artifacts)

    download_file(f"{url}/{latest}", f"{destination}/artifacts.tar.gz")

    if os.name == 'nt':
        log.add(f"\nInvoke-WebRequest {url}/{latest} -OutFile {destination}\\artifacts.tar.gz")
    else:
        log.add(f"\ncurl --output {destination}/artifacts.tar.gz {url}/{latest}\n")

    return f"{destination}/artifacts.tar.gz"
