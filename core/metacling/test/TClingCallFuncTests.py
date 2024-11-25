import unittest

import ROOT

class TClingCallFuncTest(unittest.TestCase):
    """Tests related to TClingCallFunc usage from Python"""

    def test_GH_14425(self):
        """Can call a function with non-copyable argument."""

        ROOT.gInterpreter.Declare(r"""
struct GH_14425 {
   int fMember;
   GH_14425(int m = 1) : fMember(m) {}
   GH_14425(const GH_14425&) = delete;
   GH_14425(GH_14425&&) = default;
};
int GH_14425_f(GH_14425 p = GH_14425()) { return p.fMember; }
int GH_14425_g(GH_14425 p) { return p.fMember; }
struct GH_14425_Default {
   int fMember;
   GH_14425_Default(GH_14425 p = GH_14425()) : fMember(p.fMember) {}
};
struct GH_14425_Required {
   int fMember;
   GH_14425_Required(GH_14425 p) : fMember(p.fMember) {}
};
""")
        self.assertEqual(ROOT.GH_14425_f(), 1)
        self.assertEqual(ROOT.GH_14425_f(ROOT.GH_14425(2)), 2)
        self.assertEqual(ROOT.GH_14425_g(ROOT.GH_14425(3)), 3)
        self.assertEqual(ROOT.GH_14425_Default().fMember, 1)
        self.assertEqual(ROOT.GH_14425_Default(ROOT.GH_14425(2)).fMember, 2)
        self.assertEqual(ROOT.GH_14425_Required(ROOT.GH_14425(3)).fMember, 3)
