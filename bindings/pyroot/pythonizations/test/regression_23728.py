import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


class Regression23728(unittest.TestCase):
    """
    Deleting an object of a class without compiled dictionary during
    TROOT::EndOfProcessCleanups() must not crash, even if the interpreter
    already JIT-compiled a destructor earlier in the process. The cache of
    compiled destructors used to be a function-local static, destroyed before
    the end-of-process cleanup ran because its first use came after ROOT
    registered its atexit handlers.

    See https://github.com/root-project/root/issues/23728
    """

    def test_exit_cleanup_after_interpreted_destructor(self):
        import ROOT

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "vv.root")

            # vector<vector<int>> has no compiled dictionary, so its destructor
            # has to be JIT-compiled when the branch object is deleted.
            with ROOT.TFile(fname, "RECREATE") as f:
                t = ROOT.TTree("t", "t")
                v = ROOT.std.vector[ROOT.std.vector[int]]()
                # Use the overload taking the class name and the address: the
                # templated Branch() refuses collections without a compiled
                # CollectionProxy, which would silently leave the tree without
                # the branch and the test without its purpose.
                t.Branch("vv", "vector<vector<int> >", ROOT.AddressOf(v))
                v.push_back(ROOT.std.vector[int]([1, 2]))
                t.Fill()
                f.Write()

            script = textwrap.dedent(
                f"""
                import ROOT

                f = ROOT.TFile({fname!r})
                # Let TROOT::EndOfProcessCleanups() delete the file, and with it
                # the tree and its branch object, from the C atexit handler.
                ROOT.SetOwnership(f, False)
                t = f.Get("t")
                t.GetEntry(0)

                # That only happens if the file is registered with gROOT and owns
                # the tree. Check it, so that the test fails loudly instead of
                # silently losing its purpose if these defaults change.
                if not ROOT.gROOT.GetListOfFiles().FindObject(f):
                    raise RuntimeError("the file is not in gROOT->GetListOfFiles()")
                if t.GetDirectory() != f:
                    raise RuntimeError("the tree is not owned by the file")

                # Make the interpreter JIT-compile the destructor of the branch
                # class before exit. This creates the destructor cache before the
                # end-of-process cleanup needs it (and, before the fix, destroyed
                # it again before that). Using the very same class keeps the
                # exit-time destructor call a pure cache lookup: compiling code
                # at that point is not possible on Windows, where the atexit
                # handler of libCore runs only when libCling's statics have
                # already been torn down by its DLL_PROCESS_DETACH.
                ROOT.std.vector[ROOT.std.vector[int]]()

                print("end of script")
                """
            )

            # Poison freed memory, so that a use of destroyed objects (like the
            # dead destructor cache of the issue) crashes deterministically,
            # whatever the heap layout and the interpreter internals.
            env = dict(os.environ, MALLOC_PERTURB_="85")

            # Without the fix, ROOT's crash handler can hang on the way out
            # of the segfault instead of terminating the process: fail
            # instead of waiting for the ctest timeout.
            res = subprocess.run(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=300,
            )

            self.assertEqual(res.returncode, 0, msg=res.stdout)
            self.assertIn("end of script", res.stdout)


if __name__ == "__main__":
    unittest.main()
