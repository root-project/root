# Pretend to ROOT that its input is a tty, to enable tab completion.
from __future__ import print_function
import pty, os, subprocess, sys

# Create a tty: master is our side, slave is what we pass as Popen's stdin.
# This makes ROOT believe that its stdin is a tty.
master, slave = pty.openpty()
cmd = [os.environ['ROOTSYS'] + '/bin/root.exe', '-b', '-l']
proc = subprocess.Popen(cmd,
                        stdin=slave,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        close_fds=True
                        )

# Now we need to pass what's on our stdin to ROOT's stdin
master_stdin = os.fdopen(master, 'wb', 0)
while True:
    data = os.read(0, 1024);
    if len(data) == 0:
        break
    ret = master_stdin.write(data)

# Wait until the subprocess has exited...
proc.wait()
# ...and pick up its output.
out, err = proc.communicate()

# Then write the subprocess's output to our output streams.
sys.stdout.write(out.decode('utf-8'))
sys.stderr.write(err.decode('utf-8'))

exit(proc.returncode)
