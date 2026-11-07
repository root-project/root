import unittest

import numba  # noqa: F401
import numpy as np
import ROOT

# numba is not used directly, but tests can crash when ROOT is built with
# builtin_llvm=OFF and numba is not imported at the beginning

# C++ type names, used as type annotations of the Python callables passed to Define
DOUBLE = "double"
ULONG64 = "ULong64_t"
RVEC_ULONG64 = "ROOT::VecOps::RVec<ULong64_t>"


class PyDefine(unittest.TestCase):
    """
    Testing Pythonized Define of RDF
    """

    def test_with_dtypes(self):
        """
        Tests the pythonized define with all the numba declare datatypes and
        """
        numba_declare_dtypes = ["float", "double", "int", "unsigned int", "long", "unsigned long", "bool"]
        rdf = ROOT.RDataFrame(10)
        for type in numba_declare_dtypes:
            col_name = "col_" + type.replace(" ", "")
            rdf = rdf.Define(col_name, f"({type}) rdfentry_")

            def to_array(col: type) -> f"ROOT::VecOps::RVec<{type}>":
                return [col, col]

            rdf = rdf.Define(col_name + "_arr", to_array, [col_name])
            arr = np.arange(0, 10)
            if type == "bool":
                arr = np.array(arr, dtype="bool")
            flag1 = np.array_equal(rdf.AsNumpy()[col_name], arr)
            flag2 = True
            for idx, entry in enumerate(rdf.AsNumpy()[col_name + "_arr"]):
                if not (entry[0] == arr[idx] and entry[1] == arr[idx]):
                    flag2 = False
            self.assertTrue(flag1 and flag2)

    def test_define_overload1(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")

        def square(y: ULONG64) -> ULONG64:
            return y * y

        rdf = rdf.Define("x2", square, ["x"])
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x2"], arr * arr)
        self.assertTrue(flag)

    def test_define_overload2(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")

        def square(x: ULONG64) -> ULONG64:
            return x * x

        rdf = rdf.Define("x2", square)
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x2"], arr * arr)
        self.assertTrue(flag)

    def test_define_extra_args(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")

        @ROOT.Numba.Declare(["unsigned long", "double"], "double")
        def x_y(x, y):
            return x * y

        rdf = rdf.Define("x_y", x_y, extra_args={"y": 0.5})
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x_y"], arr * 0.5)
        self.assertTrue(flag)

    def test_python_callable_extra_args(self):
        """
        Test that extra_args are rejected for a Python callable not decorated with ROOT.Numba.Declare.
        """
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")

        def x_y(x: ULONG64, y: DOUBLE) -> DOUBLE:
            return x * y

        with self.assertRaisesRegex(TypeError, "extra_args"):
            rdf.Define("x_y", x_y, ["x"], extra_args={"y": 0.5})

    def test_capture_from_scope(self):
        rdf = ROOT.RDataFrame(10).Define("x", "rdfentry_")
        y = 0.5

        def x_times_y(x: ULONG64) -> DOUBLE:
            return x * y

        rdf = rdf.Define("x_y", x_times_y)
        arr = np.arange(0, 10)
        flag = np.array_equal(rdf.AsNumpy()["x_y"], arr * 0.5)
        self.assertTrue(flag)

    def test_arrays(self):
        rdf = ROOT.RDataFrame(5).Define("x", "rdfentry_")

        def to_array(x: ULONG64) -> RVEC_ULONG64:
            return np.array([x, x])

        rdf = rdf.Define("x_arr", to_array)

        def norm(x_arr: RVEC_ULONG64) -> DOUBLE:
            return np.sqrt(x_arr[0] ** 2 + x_arr[1] ** 2)

        rdf = rdf.Define("mag", norm)
        arr = np.arange(0, 5)
        arr = np.sqrt(arr * arr + arr * arr)
        flag = np.array_equal(rdf.AsNumpy()["mag"], arr)
        self.assertTrue(flag)

    def test_cpp_functor(self):
        """
        Test that a C++ functor can be passed as a callable argument of a
        Define operation.
        """

        ROOT.gInterpreter.Declare("""
        struct MyFunctor
        {
            ULong64_t operator()(ULong64_t l) { return l*l; };
        };
        """)
        f = ROOT.MyFunctor()

        rdf = ROOT.RDataFrame(5)
        rdf2 = rdf.Define("x", f, ["rdfentry_"])

        for x, y in zip(rdf2.Take["ULong64_t"]("rdfentry_"), rdf2.Take["ULong64_t"]("x")):
            self.assertEqual(x * x, y)

    def test_std_function(self):
        """
        Test that an std::function can be passed as a callable argument of a
        Define operation.
        """

        ROOT.gInterpreter.Declare("""
        std::function<ULong64_t(ULong64_t)> myfun = [](ULong64_t l) { return l*l; };
        """)

        rdf = ROOT.RDataFrame(5)
        rdf2 = rdf.Define("x", ROOT.myfun, ["rdfentry_"])

        for x, y in zip(rdf2.Take["ULong64_t"]("rdfentry_"), rdf2.Take["ULong64_t"]("x")):
            self.assertEqual(x * x, y)

    def test_cpp_free_function(self):
        """
        Test that a C++ free function can be passed as a callable argument of a
        Define operation.
        """

        test_cases = [
            # Free function with arguments
            {
                "name": "input_ULong64_t",
                "decl": "ULong64_t my_free_function(ULong64_t l) { return l; }",
                "coltype": "ULong64_t",
                "define_args": ["rdfentry_"],
                "callable": lambda: ROOT.my_free_function,
                "extract_fn": lambda x: x,
                "expected_fn": lambda i: i,
            },
            # Free function with user defined struct
            {
                "name": "input_user_defined_struct",
                "decl": """
                    struct MyStruct {
                        ULong64_t value;
                    };
                    MyStruct my_free_function_struct(ULong64_t x) {
                        MyStruct s; s.value = x; return s;
                    }
                """,
                "coltype": "MyStruct",
                "define_args": ["rdfentry_"],
                "callable": lambda: ROOT.my_free_function_struct,
                "extract_fn": lambda s: s.value,
                "expected_fn": lambda i: i,
            },
            # Free function with no arguments
            {
                "name": "no_input",
                "decl": "ULong64_t my_free_function_none() { return 42; }",
                "coltype": "ULong64_t",
                "define_args": [],
                "callable": lambda: ROOT.my_free_function_none,
                "extract_fn": lambda x: x,
                "expected_fn": lambda _: 42,
            },
            # Free function with more than one argument
            {
                "name": "two_inputs",
                "decl": """
                    struct MyStruct2 {
                        int value;
                    };
                    MyStruct2 my_free_function_two_args(MyStruct2 s, int x) {
                        s.value = x; return s;
                    }
                """,
                "coltype": "MyStruct2",
                "define_args": ["s_col", "int_col"],
                "setup_columns": {"s_col": "MyStruct2()", "int_col": "(int)rdfentry_"},
                "callable": lambda: ROOT.my_free_function_two_args,
                "extract_fn": lambda s: s.value,
                "expected_fn": lambda i: i,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                ROOT.gInterpreter.Declare(case["decl"])
                rdf = ROOT.RDataFrame(5)

                if "setup_columns" in case:
                    for colname, gen_fn in case["setup_columns"].items():
                        rdf = rdf.Define(colname, gen_fn)

                rdf = rdf.Define("new_col", case["callable"](), case.get("define_args", []))

                outputs = rdf.Take[case["coltype"]]("new_col")
                for i, out in enumerate(outputs):
                    expected = case["expected_fn"](i)
                    actual = case["extract_fn"](out)
                    self.assertEqual(actual, expected)

    def test_cpp_free_function_overload(self):
        """
        Test that an overload of a C++ free function can be passed as a callable argument of a
        Define operation with overloads.
        """

        ROOT.gInterpreter.Declare("""
            ULong64_t my_free_function_overload(ULong64_t l) { return l; }
            ULong64_t my_free_function_overload(ULong64_t l, ULong64_t m) { return l * m; }
        """)

        rdf = ROOT.RDataFrame(5)
        rdf = rdf.Define("new_col", ROOT.my_free_function_overload, ["rdfentry_"])

        for x, y in zip(rdf.Take["ULong64_t"]("rdfentry_"), rdf.Take["ULong64_t"]("new_col")):
            self.assertEqual(x, y)

        rdf = rdf.Define("new_col_overload", ROOT.my_free_function_overload, ["rdfentry_", "rdfentry_"])
        for x, y in zip(rdf.Take["ULong64_t"]("rdfentry_"), rdf.Take["ULong64_t"]("new_col_overload")):
            self.assertEqual(x * x, y)

    def test_cpp_free_function_template(self):
        """
        Test that a templated C++ free function can be passed as a callable argument of a
        Define operation.
        """

        ROOT.gInterpreter.Declare("""
            template <typename T>
            T my_free_function_template(T l) { return l; }
        """)

        rdf = ROOT.RDataFrame(5)
        rdf = rdf.Define("new_col", ROOT.my_free_function_template["ULong64_t"], ["rdfentry_"])

        for x, y in zip(rdf.Take["ULong64_t"]("rdfentry_"), rdf.Take["ULong64_t"]("new_col")):
            self.assertEqual(x, y)

    def test_python_callables_different_signatures(self):
        """
        Test that each Python callable is instantiated with the signature of its own annotations,
        not with the one of a previous Python callable.
        """

        def times_two(x: float) -> float:
            return 2 * x

        def times_three(i: int) -> int:
            return 3 * i

        def is_positive(x: float, i: int) -> bool:
            return x > 0 and i > 0

        def constant() -> float:
            return 7.0

        rdf = (
            ROOT
            .RDataFrame(3)
            .Define("x", "(double) rdfentry_")
            .Define("i", "(int) rdfentry_")
            .Define("x2", times_two, ["x"])
            .Define("i3", times_three, ["i"])
            .Define("pos", is_positive, ["x", "i"])
            .Define("c", constant)
        )
        res = rdf.AsNumpy(["x2", "i3", "pos", "c"])

        self.assertEqual(rdf.GetColumnType("i3"), "int")
        self.assertEqual(rdf.GetColumnType("pos"), "bool")
        self.assertTrue(np.array_equal(res["x2"], [0.0, 2.0, 4.0]))
        self.assertTrue(np.array_equal(res["i3"], [0, 3, 6]))
        self.assertTrue(np.array_equal(res["pos"], [False, True, True]))
        self.assertTrue(np.array_equal(res["c"], [7.0, 7.0, 7.0]))

    def test_python_callable_root_class(self):
        """
        Test that a ROOT class can be used as type annotation of a Python callable.
        """

        def get_mass(v: ROOT.Math.PtEtaPhiMVector) -> float:
            return v.M()

        rdf = (
            ROOT
            .RDataFrame(5)
            .Define("v", "ROOT::Math::PtEtaPhiMVector(1., 0., 0., (double) rdfentry_)")
            .Define("mass", get_mass, ["v"])
        )

        self.assertTrue(np.allclose(rdf.AsNumpy(["mass"])["mass"], np.arange(5)))

    def test_python_callable_missing_annotations(self):
        """
        Test that a Python callable without type annotations is rejected.
        """
        rdf = ROOT.RDataFrame(3).Define("x", "(double) rdfentry_")
        with self.assertRaisesRegex(TypeError, "needs type annotations"):
            rdf.Define("y", lambda x: 2 * x, ["x"])


if __name__ == "__main__":
    unittest.main()
