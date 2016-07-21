# Pretend to ROOT that its input is a tty, to enable tab completion.
from __future__ import print_function
import pty, os, subprocess, sys
master, slave = pty.openpty()
cmd = [os.environ['ROOTSYS'] + '/bin/root.exe', '-b', '-l']
proc = subprocess.Popen(cmd,
                        stdin=slave,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        close_fds=True
                        )
master_stdin = os.fdopen(master, 'wb', 0)
while True:
    data = os.read(0, 1024);
    if len(data) == 0:
        break
    ret = master_stdin.write(data)
proc.wait()
out, err = proc.communicate()
sys.stdout.write(out)
sys.stderr.write(err)
