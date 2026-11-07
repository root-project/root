# Test for issue 7626: the ".> file" meta command must redirect the output of
# the executed code to the file, while the interactive prompt, the echo of the
# typed characters and the line editing (e.g. recalling a previous command with
# the up arrow) stay on the terminal and keep working - none of this must leak
# into the file.
#
# ROOT is driven through a pseudo terminal (like driveTabCom.py) so that it
# believes its stdin/stdout are a real, interactive terminal. Using pty.fork()
# also makes that pty the controlling terminal of the ROOT process, so that
# "/dev/tty" resolves to it even on headless machines (e.g. in the CI).
#
# The emulated input (read from our stdin) redirects to a file, runs two
# commands, then recalls the first one with the up arrow and re-executes it, and
# finally un-redirects and quits.
#
# We then print to our stdout, for comparison against the reference:
#  * the content of the redirected file: it must contain only the output of the
#    executed code, including the extra line produced by the command recalled
#    from the history. Before the fix the prompt and the terminal escape
#    sequences would leak into the file.
#  * whether the terminal was left in "cooked" mode while stdout was redirected:
#    in that case pressing the up arrow is echoed verbatim by the terminal as
#    the raw escape sequence "\033[A" instead of being interpreted (and the line
#    redrawn) by ROOT. We report whether that raw escape leaked to the terminal.
#
# Each emulated line is sent only once ROOT has displayed the next prompt, i.e.
# once ROOT has switched the terminal to raw mode and is waiting for input. This
# is essential: if we typed while ROOT is still starting up or busy executing a
# command, the terminal is momentarily in cooked mode and the line discipline
# would echo our keystrokes (including the up arrow) regardless of the bug,
# yielding a false positive.

import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

OUTFILE = "redirect_output.txt"
PROMPT = b"root ["  # start of the interactive prompt "root [N] "

try:
    os.remove(OUTFILE)
except OSError:
    pass

pid, master = pty.fork()
if pid == 0:
    # Child: turn into ROOT. Its stdin/stdout/stderr are the pty slave, which
    # is now its controlling terminal.
    os.execvp("root.exe", ["root.exe", "-b", "-l"])
    os._exit(1)

# Pin the terminal size so that the (short) recalled command is redrawn on a
# single line, making the output deterministic across machines.
fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

captured = bytearray()


def pump(timeout):
    # Read whatever ROOT rendered on the terminal (prompt, echo, line editing),
    # if anything is available within `timeout`. Returns False on EOF/error.
    if not select.select([master], [], [], timeout)[0]:
        return True
    try:
        chunk = os.read(master, 4096)
    except OSError:
        return False
    if not chunk:
        return False
    captured.extend(chunk)
    return True


def wait_for_prompts(count, overall_timeout=60.0):
    # Read until at least `count` prompts have been shown (ROOT is idle, waiting
    # for input, i.e. the terminal is in raw mode), or until the timeout.
    deadline = time.time() + overall_timeout
    while captured.count(PROMPT) < count and time.time() < deadline:
        if not pump(1.0):
            break


# Read the whole emulated input from our stdin.
emulated_input = b""
while True:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    emulated_input += chunk

lines = emulated_input.splitlines(keepends=True)

# Wait for the first prompt so ROOT is in raw mode before we type anything.
seen = 1
wait_for_prompts(seen)

for index, line in enumerate(lines):
    try:
        os.write(master, line)
    except OSError:
        break
    if index == len(lines) - 1:
        break  # last line is ".q": ROOT quits, no further prompt
    # Wait for the prompt that follows this command before typing the next line.
    seen += 1
    wait_for_prompts(seen)

# Let ROOT finish and close the pty.
while pump(1.0):
    pass

os.close(master)
_, status = os.waitpid(pid, 0)

# The content of the redirected file.
with open(OUTFILE) as f:
    sys.stdout.write(f.read())

# Whether pressing the up arrow leaked the escape sequence to the terminal
# (i.e. the terminal was stuck in cooked mode while stdout was redirected, so
# the arrow was echoed by the terminal instead of being interpreted by ROOT).
# A cooked terminal echoes the ESC either verbatim ("\033[A") or, more commonly,
# in caret notation ("^[[A"); in raw mode neither is echoed.
leaked = b"^[[A" in captured or b"\033[A" in captured
sys.stdout.write("---\n")
sys.stdout.write("up-arrow-leaked-raw-escape: {}\n".format("yes" if leaked else "no"))

sys.exit(0 if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else 1)
