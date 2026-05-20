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

        with ROOT.RNTupleWriter.Recreate(model, "ntpl", "test_ntuple_py_write_read.root") as writer:
            entry = writer.CreateEntry()
            entry["f"] = 42
            entry["mystr"] = "string stored in RNTuple"
            writer.Fill(entry)
        # The model should not have been destroyed (a clone has been used).
        self.assertFalse(model.IsFrozen())

        # Accessing the writer after the context manager is an error
        with self.assertRaisesRegex(RuntimeError, "cannot access RNTupleWriter after"):
            writer.GetNEntries()

        with ROOT.RNTupleReader.Open("ntpl", "test_ntuple_py_write_read.root") as reader:
            self.assertEqual(reader.GetNEntries(), 1)
            entry = reader.CreateEntry()
            reader.LoadEntry(0, entry)
            self.assertEqual(entry["f"], 42)
            self.assertEqual(entry["mystr"], "string stored in RNTuple")

        # Entry values are still accessible after the reader is gone
        self.assertEqual(entry["f"], 42)
        self.assertEqual(entry["mystr"], "string stored in RNTuple")

        # Accessing the reader after the context manager is an error
        with self.assertRaisesRegex(RuntimeError, "cannot access RNTupleReader after"):
            reader.GetNEntries()

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

        # The model should not have been destroyed (a clone has been used).
        self.assertFalse(model.IsFrozen())

        with ROOT.TFile.Open("test_ntuple_py_append.root") as f:
            reader = ROOT.RNTupleReader.Open(f["ntpl"])
            self.assertEqual(reader.GetNEntries(), 1)
            entry = reader.CreateEntry()
            reader.LoadEntry(0, entry)
            self.assertEqual(entry["f"], 42)

        # Entry values are still accessible after the reader is gone
        self.assertEqual(entry["f"], 42)

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
                with self.assertRaises(Exception):
                    # Field f2 does not exist in imposed model
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

    def test_nested_ctxmanager(self):
        """Nesting context managers of the same object is an error"""

        try:
            fileName = "test_ntuple_nested_ctxmanager_py.root"
            model = ROOT.RNTupleModel.Create()
            model.MakeField["int"]("f")
            writer = ROOT.RNTupleWriter.Recreate(model, "ntpl", fileName)
            with writer as w1:
                entry1 = w1.CreateEntry()
                with self.assertRaisesRegex(RuntimeError, "cannot nest `with`"):
                    with writer as w2:
                        entry2 = w2.CreateEntry()
                        entry1["f"] = 2
                        entry2["f"] = 4
                        w2.Fill(entry2)
                w1.Fill(entry1)

            reader = ROOT.RNTupleReader.Open("ntpl", fileName)
            with reader as r1:
                with self.assertRaisesRegex(RuntimeError, "cannot nest `with`"):
                    with reader as r2:
                        print(r2.GetNEntries())
                print(r1.GetNEntries())

        finally:
            import os
            os.remove(fileName)

    def test_weird_ctxmanager(self):
        """Using an existing object with a context manager"""

        try:
            fileName = "test_ntuple_weird_ctxmanager_py.root"
            model = ROOT.RNTupleModel.Create()
            model.MakeField["int"]("f")
            writer = ROOT.RNTupleWriter.Recreate(model, "ntpl", fileName)
            entry = writer.CreateEntry()
            with writer as w1:
                w1.Fill(entry)

            with self.assertRaisesRegex(RuntimeError, "after the `with` statement"):
                writer.Fill(entry)

            reader = ROOT.RNTupleReader.Open("ntpl", fileName)
            with reader as r1:
                with self.assertRaisesRegex(RuntimeError, "cannot nest `with`"):
                    with reader as r2:
                        print(r2.GetNEntries())
                print(r1.GetNEntries())

        finally:
            import os
            os.remove(fileName)
                    
            
