#!/usr/bin/env python

# ROOT command line tools: rootls
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to dump ROOT files contents to terminal"""

import sys
import ROOT
import cmdLineUtils

# Help strings
COMMAND_HELP = """Display ROOT files contents in the terminal."""

ONE_HELP = "Print content in one column"
LONG_PRINT_HELP = "use a long listing format."
TREE_PRINT_HELP = "print tree recursively and use a long listing format."

EPILOG = """Examples:
- rools example.root
  Display contents of the ROOT file 'example.root'.

- rools example.root:dir
  Display contents of the directory 'dir' from the ROOT file 'example.root'.

- rools example.root:*
  Display contents of the ROOT file 'example.root' and his subdirectories.

- rools file1.root file2.root
  Display contents of ROOT files 'file1.root' and 'file2.root'.

- rools *.root
  Display contents of ROOT files whose name ends with '.root'.

- rools -1 example.root
  Display contents of the ROOT file 'example.root' in one column.

- rools -l example.root
  Display contents of the ROOT file 'example.root' and use a long listing format.

- rools -t example.root
  Display contents of the ROOT file 'example.root', use a long listing format and print trees recursively.
"""

# Ansi characters
ANSI_BOLD = "\x1B[1m"
ANSI_BLUE = "\x1B[34m"
ANSI_GREEN = "\x1B[32m"
ANSI_END = "\x1B[0m"

# Needed for column width calculation
ANSI_BOLD_LENGTH = len(ANSI_BOLD+ANSI_END)
ANSI_BLUE_LENGTH = len(ANSI_BLUE+ANSI_END)
ANSI_GREEN_LENGTH = len(ANSI_GREEN+ANSI_END)

# Terminal and platform booleans
IS_TERMINAL = sys.stdout.isatty()
IS_WIN32 = sys.platform == 'win32'

def isSpecial(ansiCode,string):
    """Use ansi code on 'string' if the output is the
    terminal of a not Windows platform"""
    if IS_TERMINAL and not IS_WIN32: return ansiCode+string+ANSI_END
    else: return string

def write(string,indent=0,end=""):
    """Use sys.stdout.write to write the string with an indentation
    equal to indent and specifying the end character"""
    sys.stdout.write(" "*indent+string+end)

TREE_TEMPLATE = "{0:{nameWidth}}"+"{1:{titleWidth}}{2:{memoryWidth}}"

def recursifTreePrinter(tree,indent):
    """Print recursively tree informations"""
    listOfBranches = tree.GetListOfBranches()
    if len(listOfBranches) > 0: # Width informations
        maxCharName = max([len(branch.GetName()) \
            for branch in listOfBranches])
        maxCharTitle = max([len(branch.GetTitle()) \
            for branch in listOfBranches])
        dic = { \
            "nameWidth":maxCharName+2, \
            "titleWidth":maxCharTitle+4, \
            "memoryWidth":1}
    for branch in listOfBranches: # Print loop
        rec = \
            [branch.GetName(), \
            "\""+branch.GetTitle()+"\"", \
            str(branch.GetTotBytes())]
        write(TREE_TEMPLATE.format(*rec,**dic),indent,end="\n")
        recursifTreePrinter(branch,indent+2)

def prepareTime(time):
    """Get time in the proper shape
    ex : 174512 for 17h 45m 12s
    ex : 094023 for 09h 40m 23s"""
    time = str(time)
    time = '000000'+time
    time = time[len(time)-6:]
    return time

MONTH = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun', \
         7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
LONG_TEMPLATE = \
    isSpecial(ANSI_BOLD,"{0:{classWidth}}")+"{1:{timeWidth}}" + \
    "{2:{nameWidth}}{3:{titleWidth}}"

def roolsPrintLongLs(keyList,optDict,indent):
    """Print a list of Tkey in columns
    pattern : classname, datetime, name and title"""
    if len(keyList) > 0: # Width informations
        maxCharClass = max([len(key.GetClassName()) for key in keyList])
        maxCharTime = 12
        maxCharName = max([len(key.GetName()) for key in keyList])
        dic = { \
            "classWidth":maxCharClass+2, \
            "timeWidth":maxCharTime+2, \
            "nameWidth":maxCharName+2, \
            "titleWidth":1}
    date = ROOT.Long(0)
    for key in keyList:
        time = ROOT.Long(0)
        datime = key.GetDatime()
        datime.GetDateTime(datime.Get(),date,time)
        time = prepareTime(time)
        rec = \
            [key.GetClassName(), \
            MONTH[int(str(date)[4:6])]+" " +str(date)[6:]+ \
            " "+time[:2]+":"+time[2:4], \
            key.GetName(), \
            "\""+key.GetTitle()+"\""]
        write(LONG_TEMPLATE.format(*rec,**dic),indent,end="\n")
        if optDict['tree'] and cmdLineUtils.isTreeKey(key):
            tree = key.ReadObj()
            recursifTreePrinter(tree,indent+2)

##
# The code of the getTerminalSize function can be found here :
# https://gist.github.com/jtriley/1108174
# Thanks jtriley !!

import os
import shlex
import struct
import platform
import subprocess

def getTerminalSize():
    """ getTerminalSize()
     - get width and height of console
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python"""
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        #print "default"
        #_get_terminal_size_windows() or _get_terminal_size_tput don't work
        tuple_xy = (80, 25)      # default value
    return tuple_xy

def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass

def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass

def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])

# End of getTerminalSize code
##

def roolsPrintSimpleLs(keyList,indent,oneColumn):
    """Print list of strings in columns
    - blue for directories
    - green for trees"""
    # This code is adaptated from the pprint_list function here :
    # http://stackoverflow.com/questions/25026556/output-list-like-ls
    # Thanks hawkjo !!
    if len(keyList) == 0: return
    (term_width, term_height) = getTerminalSize()
    term_width = term_width - indent
    min_chars_between = 2
    min_element_width = min( len(key.GetName()) for key in keyList ) \
                        + min_chars_between
    max_element_width = max( len(key.GetName()) for key in keyList ) \
                        + min_chars_between
    if max_element_width >= term_width: ncol,col_widths = 1,[1]
    else:
        # Start with max possible number of columns and reduce until it fits
        ncol = 1 if oneColumn else min( len(keyList), term_width / min_element_width  )
        while True:
            col_widths = \
                [ max( len(key.GetName()) + min_chars_between \
                for j, key in enumerate(keyList) if j % ncol == i ) \
                for i in range(ncol) ]
            if sum( col_widths ) <= term_width: break
            else: ncol -= 1

    for i, key in enumerate(keyList):
        if i%ncol == 0: write("",indent) # indentation
        # Don't add spaces after the last element of the line or of the list
        if (i+1)%ncol != 0 and i != len(keyList)-1:
            if not IS_TERMINAL: write( \
                key.GetName().ljust(col_widths[i%ncol]))
            elif cmdLineUtils.isDirectoryKey(keyList[i]): write( \
                isSpecial(ANSI_BLUE,key.GetName()).ljust( \
                    col_widths[i%ncol] + ANSI_BLUE_LENGTH))
            elif cmdLineUtils.isTreeKey(keyList[i]): write( \
                isSpecial(ANSI_GREEN,key.GetName()).ljust( \
                    col_widths[i%ncol] + ANSI_GREEN_LENGTH))
            else: write(key.GetName().ljust(col_widths[i%ncol]))
        else: # No spaces after the last element of the line or of the list
            if not IS_TERMINAL: write(key.GetName())
            elif cmdLineUtils.isDirectoryKey(keyList[i]):
                write(isSpecial(ANSI_BLUE, key.GetName()))
            elif cmdLineUtils.isTreeKey(keyList[i]):
                write(isSpecial(ANSI_GREEN, key.GetName()))
            else: write(key.GetName())
            write('\n')

def roolsPrint(keyList,optDict,indent=0):
    """Print informations given by keyList with a rools
    style choosen with optDict"""
    if optDict['long'] or optDict['tree']: \
       roolsPrintLongLs(keyList,optDict,indent)
    else:
       oneColumn = True if optDict['one'] else False
       roolsPrintSimpleLs(keyList,indent, oneColumn)

def processFile(fileName, pathSplitList, optDict, manySources, indent):
    retcode = 0
    rootFile = cmdLineUtils.openROOTFile(fileName)
    if not rootFile: return 1

    keyList,dirList = cmdLineUtils.keyClassSpliter(rootFile,pathSplitList)
    if manySources: write("{0} :".format(fileName)+"\n")
    roolsPrint(keyList,optDict,indent)

    # Loop on the directories
    manyPathSplits = len(pathSplitList) > 1
    indentDir = 2 if manyPathSplits else 0
    for pathSplit in dirList:
        keyList = cmdLineUtils.getKeyList(rootFile,pathSplit)
        cmdLineUtils.keyListSort(keyList)
        if manyPathSplits: write("{0} :".format("/".join(pathSplit)),indent,end="\n")
        roolsPrint(keyList,optDict,indent+indentDir)

    rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-1", "--one", help=ONE_HELP, action="store_true")
    parser.add_argument("-l", "--long", help=LONG_PRINT_HELP, action="store_true")
    parser.add_argument("-t", "--tree", help=TREE_PRINT_HELP, action="store_true")

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)
    if sourceList == []: return 1
    cmdLineUtils.tupleListSort(sourceList)

    # Loop on the ROOT files
    retcode = 0
    manySources = len(sourceList) > 1
    indent = 2 if manySources else 0
    for fileName, pathSplitList in sourceList:
        retcode += processFile(fileName, pathSplitList, optDict, manySources, indent)
    return retcode

sys.exit(execute())
