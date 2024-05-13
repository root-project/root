## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
##
## This tutorial demonstrates the usage of the TContext class as a Python context
## manager. This functionality is related with how TFile works, so it is
## suggested to also take a look at the pyroot005 tutorial.
##
## \macro_code
## \macro_output
##
## \date March 2022
## \author Vincenzo Eduardo Padulano CERN/UPV
import os

import ROOT
from ROOT import TDirectory, TFile

# Sometimes it is useful to have multiple open files at once. In such cases,
# the current directory will always be the file that was open last.
file_1 = TFile("pyroot006_file_1.root", "recreate")
file_2 = TFile("pyroot006_file_2.root", "recreate")
print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))
# Changing directory into another file can be safely done through a TContext
# context manager.
with TDirectory.TContext(file_1):
    # Inside the statement, the current directory is file_1
    print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))
    histo_1 = ROOT.TH1F("histo_1", "histo_1", 10, 0, 10)
    file_1.WriteObject(histo_1, "my_histogram")

# After the context, the current directory is restored back to file_2. Also, the
# two files are kept open. This means that objects read, written or modified
# inside the context are still available afterwards.
print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))
if file_1.IsOpen() and file_2.IsOpen():
    print("'{}' and '{}' are open.\n".format(file_1.GetName(), file_2.GetName()))

# TContext and TFile context managers can also be used in conjunction, allowing
# for safely:
# - Opening a file, creating, modifying, writing and reading objects in it.
# - Closing the file, storing it on disk.
# - Restoring the previous value of gDirectory to the latest file opened before
#   this context, rather than to the global ROOT.gROOT
# Remember that the TContext must be initialized before the TFile, otherwise the
# current directory would already be set to the file opened for this context.
with TDirectory.TContext(), TFile("pyroot006_file_3.root", "recreate") as f:
    print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))
    histo_2 = ROOT.TH1F("histo_2", "histo_2", 10, 0, 10)
    f.WriteObject(histo_2, "another_histogram")

print("Current directory: '{}'.\n".format(ROOT.gDirectory.GetName()))

# Cleanup the files created for this tutorial
file_1.Close();
file_2.Close();
for i in range(1, 4):
    os.remove("pyroot006_file_{}.root".format(i))
