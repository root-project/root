# \file
# \ingroup ROOT7 tutorial_io
# Demonstrate the basic usage of RFile.
#
# \author Giacomo Parolini <giacomo.parolini@cern.ch>
# \date 2025-11-06
# \macro_code
# \macro_output
# \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
# is welcome!

import ROOT

import os

def write_hist_to_rfile(fileName):
    # Create a histogram to write to the file
    hist = ROOT.TH1D("hist", "hist", 10, 0, 100)
    hist.FillRandom("gaus", 1000)

    # Create a new ROOT file for writing
    with ROOT.Experimental.RFile.Recreate(fileName) as file:
        # Put objects into the file (in this case we write the same object multiple times
        # under different paths). Note that the ownership of `hist` is untouched by `file.Put`.
        file.Put(hist.GetName(), hist)
        file.Put(f"a/{hist.GetName()}", hist)
        file.Put(f"a/b/{hist.GetName()}", hist)

    # When the `with` statement is exited the file will write itself to disk and close itself.
    # To manually write the file without closing it, one can use `file.Flush()`.
        

def read_hist_from_rfile(fileName):
    # Open an existing ROOT file for reading (will raise an exception if `fileName` cannot be read).
    with ROOT.Experimental.RFile.Open(fileName) as file:
        # Iterate all keys of all objects in the file (this excludes directories by default - see the documentation of
        # ListKeys() for all the options).
        for key in file.ListKeys():
            # Retrieve the objects from the file. `file.Get` will return an object of the proper type or None if
            # the object isn't there.
            # Once an object is retrieved, it is fully owned by the application, so it survives even if `file` is closed.
            hist = file.Get(key.GetPath())
            if hist is not None:
                continue
            print(f"{key.GetClassName()} at {key.GetPath()};{key.GetCycle()}:")
            print(f"  entries: {hist.GetEntries()}")


fileName = "rfile_basics_py.root"
try:
    write_hist_to_rfile(fileName)
    read_hist_from_rfile(fileName)
    os.remove(fileName)
except FileNotFoundError:
    pass
