from __future__ import print_function
import os, sys, subprocess

MYHOME = os.path.dirname(__file__)

def main():
    if len(sys.argv) == 2:
        options = sys.argv[1]

        if options == '--cmake':
            print(os.path.join(MYHOME, "cmake"))
            return 0

        if options == '--cppflags':
            options = '--cflags'

        if options != '--help':
            try:
                cli_arg = subprocess.check_output(
                    [os.path.join(MYHOME, 'bin', 'root-config'), options],
                    stderr=subprocess.STDOUT)
                print(cli_arg.decode("utf-8").strip())
                return 0
            except subprocess.CalledProcessError:
                pass

    print('Usage: cling-config [--cflags] [--cppflags] [--cmake]')
    return 1
