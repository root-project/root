# \file
# \ingroup ROOT7 tutorial_io
# Demonstrate how to read/write an RNTuple with RFile
#
# \macro_code
# \macro_output
#
# \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
# is welcome!
#
# \date 2026-10-12
# \author Giacomo Parolini <giacomo.parolini@cern.ch>

import ROOT

import os


def write_rntuple_to_rfile(fileName):
    # Create a new ROOT file for writing
    with ROOT.Experimental.RFile.Recreate(fileName) as file:
        # Define the RNTuple schema
        model = ROOT.RNTupleModel.Create()
        model.MakeField["int"]("intField")
        # Create an RNTupleWriter to append to the RFile
        with ROOT.RNTupleWriter.Append(model, "ntuple", file) as writer:
            entry = writer.CreateEntry()

            # Fill the RNTuple with data
            for i in range(10):
                entry["intField"] = i
                writer.Fill(entry)

            # `writer` will write the data to file upon exiting the with statement....
        # ...then `file` will write itself to disk in turn when exiting the outer with statement.


def read_rntuple_from_rfile(fileName):
    # Open an existing ROOT file for reading (will throw an exception if `fileName` cannot be read).
    with ROOT.Experimental.RFile.Open(fileName) as file:
        # Get back the RNTuple anchor from the file
        rntuple = file.Get("ntuple")

        # Create an RNTupleReader from the anchor
        with ROOT.RNTupleReader.Open(rntuple) as reader:
            entry = reader.CreateEntry()
            for i in reader.GetEntryRange():
                reader.LoadEntry(i, entry)
                print(f"value[{i}] = {entry['intField']}")


fileName = "rfile_rntuple_py.root"
try:
    write_rntuple_to_rfile(fileName)
    read_rntuple_from_rfile(fileName)
    os.remove(fileName)
except FileNotFoundError:
    pass
