# File: roottest/python/basic/importErrorHandling.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 03/17/05
# Last: 03/17/05

"""Test import error handling."""


if __name__ == '__main__':
   try:
      from ROOT import GatenKaas
   except ImportError:
      pass                         # this is what we want
