import unittest

import ROOT

RNTupleModel = ROOT.Experimental.RNTupleModel
RNTupleReader = ROOT.Experimental.RNTupleReader
RNTupleWriter = ROOT.Experimental.RNTupleWriter


class RNTupleBasics(unittest.TestCase):
    """Basic tests of using RNTuple from Python"""

    def test_write_read(self):
        """Can write and read a basic RNTuple."""

        model = RNTupleModel.Create()
        model.MakeField["int"]("f")
        model.MakeField["std::string"]("mystr")

        with RNTupleWriter.Recreate(model, "ntpl", "test_ntuple_py_write_read.root") as writer:
            entry = writer.CreateEntry()
            entry["f"] = 42
            entry["mystr"] = "string stored in RNTuple"
            writer.Fill(entry)

        reader = RNTupleReader.Open("ntpl", "test_ntuple_py_write_read.root")
        self.assertEqual(reader.GetNEntries(), 1)
        entry = reader.GetModel().CreateEntry()
        reader.LoadEntry(0, entry)
        self.assertEqual(entry["f"], 42)
        self.assertEqual(entry["mystr"], "string stored in RNTuple")

    def test_forbid_writing_wrong_type(self):
        """Forbid writing the wrong type into an RNTuple field."""

        model = RNTupleModel.Create()
        model.MakeField["std::string"]("mystr")

        class WrongClass:
            ...

        with RNTupleWriter.Recreate(model, "ntpl", "test_ntuple_py_test_forbid_writing_wrong_type.root") as writer:
            entry = writer.CreateEntry()
            with self.assertRaises(TypeError):
                entry["mystr"] = WrongClass()
