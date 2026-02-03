import subprocess

def printError(msg):
    print(f"*** Error: {msg}")

def printWarning(msg):
    print(f"*** Warning: {msg}")

def printInfo(msg):
    print(f"Info: {msg}")

def execCommand(cmd, thisCwd="./", theInput=None, desc="", replace=''):
    """
    Execute a command and return the output. For logging reasons, the command
    is also printed.
    If "desc" is specificed, the command is not printed but "desc".
    """
    if "" == desc:
        echoCmd = f"In directory {thisCwd} *** {cmd} {'with std input' if theInput else ''}"
        if "" != replace:
            echoCmd = echoCmd.replace(replace, '***')
        printInfo(echoCmd)
    else:
        print(desc)
    compProc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=thisCwd, input=theInput, encoding="latin1"
    )
    if 0 != compProc.returncode:
        printError(f"{compProc.stderr.strip()}")
        raise ValueError(f'Command "{cmd}" failed ({compProc.returncode})')
    out = compProc.stdout.strip()
    return out