import platform
import unittest

import ROOT


class RNTupleBasics(unittest.TestCase):
    """Basic tests of using RNTuple from Python"""

    def test_write_read(self):
        """Can write and read a basic RNTuple."""

        model = ROOT.RNTupleModel.Create()
        model.MakeField["int"]("f")
        model.MakeField["std::string"]("mystr")

        nentries = 2
        with ROOT.RNTupleWriter.Recreate(model, "ntpl", "test_ntuple_py_write_read.root") as writer:
            entry = writer.CreateEntry()
            for i in range(nentries):
                entry["f"] = i
                entry["mystr"] = f"{i} string stored in RNTuple"
                writer.Fill(entry)
        self.assertFalse(model.IsFrozen(),
                         msg="The model should not have been destroyed (a clone has been used).")

        with self.assertRaisesRegex(ReferenceError, "attempt to access a null-pointer",
                                    msg="Upon exiting the context, the writer is destructed."):
            writer.GetNEntries()

        with ROOT.RNTupleReader.Open("ntpl", "test_ntuple_py_write_read.root") as reader:
            self.assertEqual(reader.GetNEntries(), nentries)
            entry = reader.CreateEntry()
            for i in reader:
                reader.LoadEntry(i, entry)
                with self.subTest(i=i):
                    self.assertEqual(entry["f"], i)
                    self.assertEqual(entry["mystr"], f"{i} string stored in RNTuple")

        with self.assertRaisesRegex(ReferenceError, "attempt to access a null-pointer",
                                    msg="Upon exiting the context, the reader is destructed."):
            reader.GetNEntries()

        msg = "Last entry values are still accessible after the reader is destructed."
        self.assertEqual(entry["f"], nentries - 1, msg=msg)
        self.assertEqual(entry["mystr"], f"{nentries - 1} string stored in RNTuple", msg=msg)

    def test_write_fields(self):
        """Can create writer with on-the-fly model"""

        # FIXME: This should work without make_pair...
        fields = [ROOT.std.make_pair("int", "f")]
        with ROOT.RNTupleWriter.Recreate(fields, "ntpl", "test_ntuple_py_write_fields.root") as writer:
            entry = writer.CreateEntry()
            entry["f"] = 42
            writer.Fill(entry)

        with ROOT.RNTupleReader.Open("ntpl", "test_ntuple_py_write_fields.root") as reader:
            self.assertEqual(reader.GetNEntries(), 1)
            entry = reader.CreateEntry()
            reader.LoadEntry(0, entry)
            self.assertEqual(entry["f"], 42)

    def test_append_open(self):
        """Can append to existing TFile and open from RNTuple key."""

        model = ROOT.RNTupleModel.Create()
        model.MakeField["int"]("f")

        with ROOT.TFile.Open("test_ntuple_py_append.root", "RECREATE") as f:
            with ROOT.RNTupleWriter.Append(model, "ntpl", f) as writer:
                entry = writer.CreateEntry()
                entry["f"] = 42
                writer.Fill(entry)

        self.assertFalse(model.IsFrozen(),
                         msg="The model should not have been destroyed (a clone has been used).")

        with ROOT.TFile.Open("test_ntuple_py_append.root") as f:
            with ROOT.RNTupleReader.Open(f["ntpl"]) as reader:
                self.assertEqual(reader.GetNEntries(), 1)
                entry = reader.CreateEntry()
                reader.LoadEntry(0, entry)
                self.assertEqual(entry["f"], 42)

        with self.subTest(repr(reader)):
            self.assertFalse(reader, "RNTupleReader destructed")
        self.assertEqual(entry["f"], 42, "Entry values still accessible")

    def test_read_model(self):
        """Can impose a model when reading."""

        write_model = ROOT.RNTupleModel.Create()
        write_model.MakeField["int"]("f1")
        write_model.MakeField["int"]("f2")

        with ROOT.RNTupleWriter.Recreate(write_model, "ntpl", "test_ntuple_py_read_model.root") as writer:
            entry = writer.CreateEntry()
            writer.Fill(entry)

        read_model = ROOT.RNTupleModel.Create()
        read_model.MakeField["int"]("f1")

        with ROOT.RNTupleReader.Open(read_model, "ntpl", "test_ntuple_py_read_model.root") as reader:
            entry = reader.CreateEntry()
            if not platform.system() == "Windows":
                # TODO: re-enable it on Windows once the exception handling is fixed
                with self.assertRaises(ROOT.RException, msg="Field f2 does not exist in imposed model"):
                    entry["f2"] = 42

    def test_forbid_writing_wrong_type(self):
        """Forbid writing the wrong type into an RNTuple field."""

        model = ROOT.RNTupleModel.Create()
        model.MakeField["std::string"]("mystr")

        class WrongClass: ...

        with ROOT.RNTupleWriter.Recreate(model, "ntpl", "test_ntuple_py_test_forbid_writing_wrong_type.root") as writer:
            entry = writer.CreateEntry()
            with self.assertRaises(TypeError):
                entry["mystr"] = WrongClass()

    def test_singleuse_ctxmanager(self):
        """RNTupleReader/RNTupleWriter context managers are single use context managers.

        Upon exiting the context, they are destructed.
        They are not reentrant - cannot be used in nested 'with' statements,
        or are not reusable - cannot be used multiple times."""

        try:
            fileName = "test_singleuse_ctxmanager_py.root"
            model = ROOT.RNTupleModel.Create()
            model.MakeField["int"]("f")
            writer = ROOT.RNTupleWriter.Recreate(model, "ntpl", fileName)
            with writer as w1:
                entry1 = w1.CreateEntry()
                entry1["f"] = 2
                with writer as w2:
                    entry2 = w2.CreateEntry()
                    entry2["f"] = 4
                    w2.Fill(entry2)
                with self.assertRaisesRegex((ReferenceError, TypeError), "attempt to access a null-pointer"):
                    w1.Fill(entry1)

            with self.assertRaisesRegex(ValueError, "I/O operation on destructed 'RNTupleWriter'"):
                with writer as w:
                    entry = w.CreateEntry()
                    entry["f"] = 8
                    w.Fill(entry)

            reader = ROOT.RNTupleReader.Open("ntpl", fileName)
            with reader as r1:
                with reader as r2:
                    print(r2.GetNEntries())
                with self.assertRaisesRegex(ReferenceError, "attempt to access a null-pointer"):
                    print(r1.GetNEntries())

            with self.assertRaisesRegex(ValueError, "I/O operation on destructed 'RNTupleReader'"):
                with reader as r:
                    print(r.GetNEntries())

        finally:
            import os
            os.remove(fileName)
