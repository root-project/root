import unittest

import numpy as np
import ROOT


class NumpyArrayView(unittest.TestCase):
    """
    Test the conversion of interpreter-defined C++ arrays into numpy views
    """

    # typemaps based on https://numpy.org/doc/stable/reference/arrays.scalars.html
    cpp_dtypes = [
        "char",
        "unsigned char",
        "int",
        "unsigned int",
        "short",
        "unsigned short",
        "float",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
    ]

    np_dtypes = [
        np.byte,
        np.ubyte,
        np.intc,
        np.uintc,
        np.short,
        np.ushort,
        np.float32,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
    ]

    typemap = zip(np_dtypes, cpp_dtypes)

    bounds = {
        "char": (-128, 127),
        "unsigned char": (0, 255),
        "int": (-(2**31), 2**31 - 1),
        "unsigned int": (0, 2**32 - 1),
        "short": (-(2**15), 2**15 - 1),
        "unsigned short": (0, 2**16 - 1),
        # FIXME : low level views for 64 bit types (long and double) do not work, upstream interprets the converter with dims = 0,
        # which somehow makes this work, however this needs to be investigated further.
        "long": (-(2**31), 2**31 - 1),
        "long long": (-(2**62), 2**62 - 1),
        "unsigned long": (0, 2**32 - 1),
        "unsigned long long": (0, 2**64 - 1),
        "float": (-3.4e38, 3.4e38),
        "double": (-1.7e308, 1.7e308),
        "int8_t": (-128, 127),
        "uint8_t": (0, 255),
        "int16_t": (-(2**15), 2**15 - 1),
        "uint16_t": (0, 2**16 - 1),
        "int32_t": (-(2**31), 2**31 - 1),
        "uint32_t": (0, 2**32 - 1),
    }

    def generate_cpp_arrays(self, dtype_cpp):
        mn, mx = self.bounds[dtype_cpp]
        # sanitize a name for the struct so there's no spaces
        tag = dtype_cpp.replace(" ", "_").replace("unsigned_", "u")

        cpp = f"""
            struct Foo_{tag} {{
                {dtype_cpp} bar[11][2] = {{}};
            }};
            Foo_{tag} foo_{tag};
            foo_{tag}.bar[0][0] = {mn};
            foo_{tag}.bar[1][1] = {mx};
            foo_{tag}.bar[2][0] = {mn};
            foo_{tag}.bar[3][1] = {mx};
            foo_{tag}.bar[4][0] = {mn};
            foo_{tag}.bar[5][1] = {mx};
            foo_{tag}.bar[6][0] = {mn};
            foo_{tag}.bar[7][1] = {mx};
            foo_{tag}.bar[8][0] = {mn};
            foo_{tag}.bar[9][1] = {mx};
            foo_{tag}.bar[10][0] = {mn};
            """
        ROOT.gInterpreter.ProcessLine(cpp)
        return getattr(ROOT, f"foo_{tag}").bar

    def check_shape(self, cpp_arr, np_obj):
        self.assertEqual(cpp_arr.shape, np_obj.shape)

    def validate_numpy_view(self, np_obj, dtype):
        # obtain bounds for C++ builtins
        mn, mx = self.bounds[dtype[1]]
        kind = dtype[0]

        if issubclass(kind, np.integer):
            cast = int
        elif issubclass(kind, np.floating):

            def cast(v):
                return float(f"{v:.2e}")

        for i, row in enumerate(np_obj[:11]):
            # we check col 0 for even i, 1 for odd i, as the array was filled that way
            col = i & 1
            val = cast(row[col])

            # the expected bound is min for col 0 and max for col 1
            expected = mn if col == 0 else mx
            self.assertEqual(val, expected)

    def test_2DArray_NumpyView(self):
        """
        Test correct numpy view for different C++ builtin-type 2D arrays
        """
        for dtype in self.typemap:
            cpp_arr = self.generate_cpp_arrays(dtype[1])
            np_obj = np.frombuffer(cpp_arr, dtype[0], count=11 * 2).reshape(11, 2)
            self.check_shape(cpp_arr, np_obj)
            self.validate_numpy_view(np_obj, dtype)


if __name__ == "__main__":
    unittest.main()
