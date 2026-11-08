import unittest
import warnings

import ROOT


class TemplateInstantiationDiagnostics(unittest.TestCase):
    """Errors and warnings from the instantiation of templates called from Python"""

    def test_gh_11854(self):
        """Regression test for https://github.com/root-project/root/issues/11854: the compiler
        error in a nested instantiation is reported, rather than a missing symbol"""

        ROOT.gInterpreter.Declare(r"""
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
        """)

        helper = ROOT.GH11854Helper[ROOT.std.vector["double"]]()
        with self.assertRaisesRegex(TypeError, "cannot assign to variable 'res' with const-qualified type"):
            ROOT.call_gh11854_helper(helper)

    def test_static_assert_deduced(self):
        """An instantiation with a failing static_assert is rejected (rather than run), with its message"""

        ROOT.gInterpreter.Declare(r"""
        template <typename T>
        T diag_only_floating(T x) {
            static_assert(std::is_floating_point_v<T>, "diag_only_floating requires a floating point type");
            return x;
        }
        """)

        self.assertEqual(ROOT.diag_only_floating(1.5), 1.5)
        with self.assertRaisesRegex(TypeError, "diag_only_floating requires a floating point type"):
            ROOT.diag_only_floating(ROOT.std.string("abc"))

    def test_static_assert_explicit(self):
        """Same for explicitly given template arguments"""

        ROOT.gInterpreter.Declare(r"""
        template <typename T>
        T diag_only_floating_explicit(T x) {
            static_assert(std::is_floating_point_v<T>, "diag_only_floating_explicit requires a floating point type");
            return x;
        }
        """)

        with self.assertRaisesRegex(TypeError, "diag_only_floating_explicit requires a floating point type"):
            ROOT.diag_only_floating_explicit["int"](3)
        # also when trying again
        with self.assertRaises(TypeError):
            ROOT.diag_only_floating_explicit["int"](3)

    def test_error_in_body(self):
        """The compiler error in the body is reported"""

        ROOT.gInterpreter.Declare(r"""
        template <typename T>
        auto diag_call_foo(const T &x) { return x.foo(); }
        """)

        with self.assertRaisesRegex(TypeError, "no member named 'foo'"):
            ROOT.diag_call_foo(ROOT.std.vector["int"]())

    def test_warning(self):
        """Compiler warnings from a successful instantiation are reported as a python warning"""

        ROOT.gInterpreter.Declare(r"""
        template <typename T>
        T diag_shift_big(T x) { return x << 40; }
        """)

        with self.assertWarnsRegex(UserWarning, "shift count >= width of type"):
            ROOT.diag_shift_big(1)

        # no warning when the cached instantiation is used again, or for an instantiation without
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ROOT.diag_shift_big(1)
            self.assertEqual(ROOT.diag_shift_big["long"](1), 1 << 40)

if __name__ == '__main__':
    unittest.main()
