# \file
# \ingroup tutorial_pyroot
# \notebook -nodraw
#
# This tutorial demonstrates the usage of the TFile class as a Python context
# manager.
#
# \macro_code
# \macro_output
#
# \date March 2022
# \author Vincenzo Eduardo Padulano CERN/UPV
import os

import ROOT
from ROOT import TFile

# By default, objects of some ROOT types such as `TH1` and its derived types
# are automatically attached to a ROOT.TDirectory when they are created.
# Specifically, at any given point of a ROOT application, the ROOT.gDirectory
# object tells which is the current directory where objects will be attached to.
# The next line will print 'PyROOT' as the name of the current directory.
# That is the global directory created when using ROOT from Python, which is
# the ROOT.gROOT object.
print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))

# We can check to which directory a newly created histogram is attached.
histo_1 = ROOT.TH1F("histo_1", "histo_1", 10, 0, 10)
print("Histogram '{}' is attached to: '{}'.\n".format(histo_1.GetName(), histo_1.GetDirectory().GetName()))

# For quick saving and forgetting of objects into ROOT files, it is possible to
# open a TFile as a Python context manager. In the context, objects can be
# created, modified and finally written to the file. At the end of the context,
# the file will be automatically closed.
with TFile.Open("pyroot005_file_1.root", "recreate") as f:
    histo_2 = ROOT.TH1F("histo_2", "histo_2", 10, 0, 10)
    # Inside the context, the current directory is the open file
    print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))
    # And the created histogram is automatically attached to the file
    print("Histogram '{}' is attached to: '{}'.\n".format(histo_2.GetName(), histo_2.GetDirectory().GetName()))
    # Before exiting the context, objects can be written to the file
    f.WriteObject(histo_2, "my_histogram")

# When the TFile.Close method is called, the current directory is automatically
# set again to ROOT.gROOT. Objects that were attached to the file inside the
# context are automatically deleted and made 'None' when the file is closed.
print("Status after the first TFile context manager:")
print(" Current directory: '{}'.".format(ROOT.gDirectory.GetName()))
print(" Accessing 'histo_2' gives: '{}'.\n".format(histo_2))

# Also reading data from a TFile can be done in a context manager. Information
# stored in the objects of the file can be queried and used inside the context.
# After the context, the objects are not usable anymore because the file is
# automatically closed. This means you should use this pattern as a quick way
# to get information or modify objects from a certain file, without needing to
# keep the histograms alive afterwards.
with TFile.Open("pyroot005_file_1.root", "read") as f:
    # Retrieve histogram using the name given to f.WriteObject in the previous
    # with statement
    histo_2_fromfile = f.my_histogram
    print("Retrieved '{}' histogram from file '{}'.\n".format(histo_2_fromfile.GetName(), f.GetName()))

# Cleanup the file created for this tutorial
os.remove("pyroot005_file_1.root")
