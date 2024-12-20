import unittest


class TPython(unittest.TestCase):
    """
    Testing features of TPython from Python, to see if they still work when the
    Python interpreter was not initialized by TPython on the C++ side.
    """

    def test_exec(self):
        """
        Test TPython::Exec.
        """
        import ROOT

        ROOT.gInterpreter.Declare(
            """

            // Test TPython::Exec from multiple threads.
            int testTPythonExec(int nIn)
            {
               std::any out;
               std::stringstream cmd;
               cmd << "_anyresult = ROOT.std.make_any['int'](" << nIn << ")";
               TPython::Exec(cmd.str().c_str(), &out);
               return std::any_cast<int>(out);
            }
        """
        )

        n_in = 100
        n_out = ROOT.testTPythonExec(n_in)
        self.assertEqual(n_out, n_in)


if __name__ == "__main__":
    unittest.main()
