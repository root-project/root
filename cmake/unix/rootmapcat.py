#! /usr/bin/env python

'''
An utility to smartly "cat" rootmap files.
'''
from __future__ import print_function
import argparse
import sys

#-------------------------------------------------------------------------------
def getParser():
    parser = argparse.ArgumentParser(description='Get input rootmaps and output rootmap.')
    parser.add_argument("rootmaps", nargs='+', help='The name of the rootmaps separated by a space.')
    parser.add_argument("-o", "--output", dest='output',
                        default="all.rootmap", help='The output rootmap name.')
    return parser

#-------------------------------------------------------------------------------
class Rootmap(object):
    def __init__(self):
        self.fwdDecls = []
        self.sections = {}

    def ParseAndAddMany(self,rootmapnames):
        for rootmapname in rootmapnames:
            self.ParseAndAdd(rootmapname)

    def ParseAndAdd(self,rootmapname):
        ifile = open(rootmapname)
        rootmapLines = ifile.readlines()
        ifile.close()
        fwdDeclsSet = set()
        fwdDeclsSection = False
        keysSection = True
        for line in rootmapLines:
            if line.startswith("{ decls }"):
                fwdDeclsSection = True
                keysSection = False
                continue
            if line.startswith("[ "):
                fwdDeclsSection = False
                keysSection = True
                secName = line
            if line == "\n": continue
            if fwdDeclsSection:
                fwdDeclsSet.add(line)
            if keysSection:
                if self.sections.has_key(secName):
                    self.sections[secName].append(line)
                else:
                    self.sections[secName] = []
        self.fwdDecls.extend(fwdDeclsSet)
        
    def Print(self,outrootmapname):
        # Now we reduce the fwd declarations
        self.fwdDecls = sorted(list(set(self.fwdDecls)))
        ofile = file(outrootmapname, "w")
        if len(self.fwdDecls) != 0:
            ofile.write("{ decls }\n")
            for fwdDecl in self.fwdDecls:
                ofile.write(fwdDecl)
            ofile.write("\n")

        for libname, keylines in self.sections.items():
            ofile.write(libname)
            for keyline in keylines:
                ofile.write(keyline)
            ofile.write("\n")
        ofile.close()

#-------------------------------------------------------------------------------
def merge(rmapsnames, outrootmapname):
    rm = Rootmap()
    rm.ParseAndAddMany(rmapsnames)
    rm.Print(outrootmapname)
    return 0

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    rmapsnames = args.rootmaps
    outrootmapname = args.output
    sys.exit(merge(rmapsnames, outrootmapname))
