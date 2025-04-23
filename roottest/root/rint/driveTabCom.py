# Pretend to ROOT that its input is a tty, to enable tab completion.
from __future__ import print_function
import pty, os, subprocess, sys, select

# Create a tty: master is our side, slave is what we pass as Popen's stdin.
# This makes ROOT believe that its stdin is a tty.
master, slave = pty.openpty()
cmd = [os.environ['ROOTSYS'] + '/bin/root.exe', '-b', '-l']
proc = subprocess.Popen(cmd,
                        stdin = slave,
                        stdout = slave,
                        stderr = subprocess.PIPE,
                        close_fds=True
                        )

# Now we need to pass what's on our stdin to ROOT's stdin
# and read what's on the slave's stdout.
while True:
    data = os.read(0, 1024);
    if len(data) == 0:
        break
    #ret = master_stdin.write(data)
    os.write(master, data)
    while select.select([master],[],[],1.0)[0]:
        os.write(1, os.read(master, 1))

# Wait until the subprocess has exited...
proc.wait()

# ...and pick up its output.
out, err = proc.communicate()

# Then write the subprocess's output to our output streams.
#sys.stdout.write(out.decode('utf-8'))
sys.stderr.write(err.decode('utf-8'))

# No need to wait; the subprocess is dead.
while select.select([master],[],[],0.0)[0]:
  sys.stdout.write(os.read(master, 1).decode('utf-8'))

exit(proc.returncode)

