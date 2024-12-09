# \file
# \ingroup tutorial_io
# \notebook -nodraw
# Illustrates how to merge two files using TFileMerger from Python.
# \author Giacomo Parolini

import ROOT
import os
import random

# abridged from hsimple.py
def CreateInputFile(fname): 
   with ROOT.TFile.Open( fname, "RECREATE", "Demo ROOT file with histograms" ) as hfile:
      # Create some histograms, a profile histogram and an ntuple
      hpx    = ROOT.TH1F( "hpx", "This is the px distribution", 100, -4, 4 )
      hpxpy  = ROOT.TH2F( "hpxpy", "py vs px", 40, -4, 4, 40, -4, 4 )
      hprof  = ROOT.TProfile( "hprof", "Profile of pz versus px", 100, -4, 4, 0, 20 )
      ntuple = ROOT.TNtuple( "ntuple", "Demo ntuple", "px:py:pz:random:i" )

      # Fill histograms randomly.
      for i in range( 2000 ):
         px = random.randrange(0, 1)
         py = random.randrange(0, 1) 
         pz = px*px + py*py
         r = random.randrange(0, 1)

       # Fill histograms.
         hpx.Fill( px )
         hpxpy.Fill( px, py )
         hprof.Fill( px, pz )
         ntuple.Fill( px, py, pz, r, i )
      hfile.Write()


def MergeFiles(files_to_cleanup, nfiles):
   # NOTE: when the TFileMerger is used in a `with` statement, it will automatically
   # close its output file when going out of scope.
   with ROOT.TFileMerger(False) as fm:
      fm.OutputFile("merged.root")
      files_to_cleanup.append(fm.GetOutputFile().GetName())
      for i in range(0, nfiles):
         fm.AddFile(f"tomerge{i}.root")

      # New merging flags must be bitwise OR-ed on top of the default ones.
      # Here, as an example, we are doing an incremental merging, meaning we want to merge the new
      # files with the current content of the output file.
      # See TFileMerger docs for all the flags available: 
      # https://root.cern/doc/master/classTFileMerger.html#a8ea43dc0722ce413c7332584d8c3ef0f
      mode = ROOT.TFileMerger.kAll | ROOT.TFileMerger.kIncremental
      fm.PartialMerge(mode)
      fm.Reset()


if __name__ == '__main__':
   nfiles = 2
   files_to_cleanup = []
   try:
      # Create the files to be merged
      for i in range(0, nfiles):
         fname = f"tomerge{i}.root"
         CreateInputFile(fname)
         files_to_cleanup.append(fname)

      MergeFiles(files_to_cleanup, nfiles)

   finally:
      # Cleanup initial files
      for filename in files_to_cleanup:
         os.remove(filename)

