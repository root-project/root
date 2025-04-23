from __future__ import print_function

import sys

def print_flushed(*args):
   if sys.hexversion < 0x3000000:
      return print(*args)
   else:
      return print(*args, flush=True)
