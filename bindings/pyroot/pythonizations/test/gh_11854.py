import unittest

import ROOT


class GH11854(unittest.TestCase):
    """Regression test for https://github.com/root-project/root/issues/11854"""

    def test_gh_11854(self):

        ROOT.gInterpreter.Declare(r"""
        #ifndef GH11854_HELPER
        #define GH11854_HELPER

        template <typename T>
        struct GH11854Helper {

            std::size_t operator() () const {
                const std::size_t res = 0;
                res = T{0, 0}.size();
                return res;
            }

        };

        template <typename H>
        std::size_t call_gh11854_helper(const H &helper) {
            return helper();
        }

        #endif // GH11854_HELPER
        """)

        helper = ROOT.GH11854Helper[ROOT.std.vector["double"]]()
        with self.assertRaisesRegex(TypeError, "cannot assign to variable 'res' with const-qualified type 'const std::size_t'"):
            ROOT.call_gh11854_helper(helper)


if __name__ == '__main__':
    unittest.main()
