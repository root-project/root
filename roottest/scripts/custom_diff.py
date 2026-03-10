"""Clever (with filters) compare command using difflib.py providing diffs in four formats:

* ndiff:    lists every line and highlights interline changes.
* context:  highlights clusters of changes in a before/after format.
* unified:  highlights clusters of changes in an inline format.
* html:     generates side by side comparison with change highlights.

"""
import difflib
import optparse
import os
import re
import sys
import time


#---------------------------------------------------------------------------------------------------------------------------
#---Filter and substitutions------------------------------------------------------------------------------------------------
class LineFilter:
    """A line filter to suppress lines in the diff.
    self.skip_regexes contains patterns to skip entirely.
    self.substitutions contains patterns to replace with the given strings.
    These patterns only need to be compiled once for all the incoming lines.
    """

    def __init__(self):
        # Skip these lines
        self.skip_regexes = [
            re.compile(pattern)
            for pattern in [
                r"^Processing ",  # Interpreted macros
                r"^Info in <\w+::ACLiC>: creating shared library",  # Compiled macros
                r"^In file included from input_line",  # Wrapper input line
                r"^[:space:]*$",  # Lines which are empty apart from spaces
                r"^Info in <TROOT>: Object auto registration",  # ROOT 7 mode
            ]
        ]

        # Replace these patterns in all lines
        self.substitutions = [
            (re.compile(r"[ ]@0x[a-fA-F0-9]+"), ""),  # Remove pointers from output
            (re.compile(r"std::__[0-9]+::"), "std::"),  # Canonicalise standard namespaces
            (
                re.compile(r"^(\S*/|)([^:/]+)[-:0-9]*(?=: error:)"),
                r"\2",
            ),  # Trim file paths and line numbers from lines with ": error:"
            (re.compile(r"(input_line_|ROOT_prompt_)[0-9]+"), r"\1--"),  # Canonicalise input_line_123, ROOT_prompt_345
        ]

    def filter(self, lines, ignoreWhiteSpace=False):
        outlines = []
        for line in lines:
            if sys.platform == "win32":
                if "Creating library " in line:
                    continue
                if "_ACLiC_dict" in line:
                    continue
                if "Warning in <TInterpreter::ReadRootmapFile>:" in line:
                    continue
                if "Warning in <TClassTable::Add>:" in line:
                    continue
                if "Error: Removing " in line:
                    continue
                if " -nologo -TP -c -nologo -I" in line:
                    continue
                if "rootcling -v1 -f " in line:
                    continue
                if "No precompiled header available" in line:
                    continue

            # ---Skip all lines matching predefined expressions---------------------------
            if any(pattern.match(line) is not None for pattern in self.skip_regexes):
                continue

            # ---Apply predefined replacements--------------------------------------------
            for pattern, replacement in self.substitutions:
                line = pattern.sub(replacement, line)

            # ---Remove white spaces------------------------------------------------------
            if ignoreWhiteSpace:
                line = re.sub(r"[ ]", "", line)
            outlines.append(line)
        return outlines

#-----------------------------------------------------------------------------------------------------------------------------
def main():
  usage = "usage: %prog [options] fromfile tofile"
  parser = optparse.OptionParser(usage)
  parser.add_option("-c", action="store_true", default=False, help='Produce a context format diff (default)')
  parser.add_option("-u", action="store_true", default=True, help='Produce a unified format diff')
  parser.add_option("-m", action="store_true", default=False, help='Produce HTML side by side diff (can use -c and -l in conjunction)')
  parser.add_option("-n", action="store_true", default=False, help='Produce a ndiff format diff')
  parser.add_option("-l", "--lines", type="int", default=3, help='Set number of context lines (default 3)')
  (options, args) = parser.parse_args()

  if len(args) == 0:
    parser.print_help()
    sys.exit(1)
  if len(args) != 2:
    parser.error("need to specify both a fromfile and tofile")

  n = options.lines
  fromfile, tofile = args

  fromdate = time.ctime(os.stat(fromfile).st_mtime)
  todate = time.ctime(os.stat(tofile).st_mtime)
  if sys.platform == 'win32':
    fromlines = open(fromfile).readlines()
    tolines = open(tofile).readlines()
  else:
    fromlines = open(fromfile, 'r' if sys.version_info >= (3, 4) else 'U').readlines()
    tolines = open(tofile, 'r' if sys.version_info >= (3, 4) else 'U').readlines()

  lineFilter = LineFilter()
  nows_fromlines = lineFilter.filter(fromlines, True)
  nows_tolines = lineFilter.filter(tolines, True)

  check = difflib.context_diff(nows_fromlines, nows_tolines)
  try:
    _ = next(check)
  except StopIteration:
    sys.exit(0)

  fromlines = lineFilter.filter(fromlines, False)
  tolines = lineFilter.filter(tolines, False)

  if options.u:
    diff = difflib.unified_diff(fromlines, tolines, fromfile, tofile, fromdate, todate, n=n)
  elif options.n:
    diff = difflib.ndiff(fromlines, tolines)
  elif options.m:
    diff = difflib.HtmlDiff().make_file(fromlines,tolines,fromfile,tofile,context=options.c,numlines=n)
  else:
    diff = difflib.context_diff(fromlines, tolines, fromfile, tofile, fromdate, todate, n=n)

  difflines = [line for line in diff]
  sys.stdout.writelines(difflines)

  if difflines:
    sys.exit(1)
  else:
    sys.exit(0)

if __name__ == '__main__':
  main()

