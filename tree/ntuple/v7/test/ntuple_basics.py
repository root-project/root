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
        # The model should not have been destroyed (a clone has been used).
        self.assertFalse(model.IsFrozen())

        reader = RNTupleReader.Open("ntpl", "test_ntuple_py_write_read.root")
        self.assertEqual(reader.GetNEntries(), 1)
        entry = reader.CreateEntry()
        reader.LoadEntry(0, entry)
        self.assertEqual(entry["f"], 42)
        self.assertEqual(entry["mystr"], "string stored in RNTuple")

    def test_write_fields(self):
        """Can create writer with on-the-fly model"""

        # FIXME: This should work without make_pair...
        fields = [ROOT.std.make_pair("int", "f")]
        with RNTupleWriter.Recreate(fields, "ntpl", "test_ntuple_py_write_fields.root") as writer:
            entry = writer.CreateEntry()
            entry["f"] = 42
            writer.Fill(entry)

        reader = RNTupleReader.Open("ntpl", "test_ntuple_py_write_fields.root")
        self.assertEqual(reader.GetNEntries(), 1)
        entry = reader.CreateEntry()
        reader.LoadEntry(0, entry)
        self.assertEqual(entry["f"], 42)

    def test_append_open(self):
        """Can append to existing TFile and open from RNTuple key."""

        model = RNTupleModel.Create()
        model.MakeField["int"]("f")

        with ROOT.TFile.Open("test_ntuple_py_append.root", "RECREATE") as f:
            with RNTupleWriter.Append(model, "ntpl", f) as writer:
                entry = writer.CreateEntry()
                entry["f"] = 42
                writer.Fill(entry)
        # The model should not have been destroyed (a clone has been used).
        self.assertFalse(model.IsFrozen())

        with ROOT.TFile.Open("test_ntuple_py_append.root") as f:
            reader = RNTupleReader.Open(f["ntpl"])
            self.assertEqual(reader.GetNEntries(), 1)
            entry = reader.CreateEntry()
            reader.LoadEntry(0, entry)
            self.assertEqual(entry["f"], 42)

    def test_read_model(self):
        """Can impose a model when reading."""

        write_model = RNTupleModel.Create()
        write_model.MakeField["int"]("f1")
        write_model.MakeField["int"]("f2")

        with RNTupleWriter.Recreate(write_model, "ntpl", "test_ntuple_py_read_model.root") as writer:
            entry = writer.CreateEntry()
            writer.Fill(entry)

        read_model = RNTupleModel.Create()
        read_model.MakeField["int"]("f1")

        reader = RNTupleReader.Open(read_model, "ntpl", "test_ntuple_py_read_model.root")
        entry = reader.CreateEntry()
        with self.assertRaises(Exception):
            # Field f2 does not exist in imposed model
            entry["f2"] = 42

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
