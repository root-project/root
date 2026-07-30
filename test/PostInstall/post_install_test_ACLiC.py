import ctypes

import cppyy.ll
import ROOT


def main():
    err = ctypes.c_int(-1)  # A value purposely different than any of the EErrorCode values
    err_ptr = cppyy.ll.reinterpret_cast["TInterpreter::EErrorCode*"](ctypes.addressof(err))
    ROOT.gInterpreter.ProcessLine(".L post_install_test_ACLiC.C+", err_ptr)
    if err.value != ROOT.TInterpreter.kNoError:
        raise RuntimeError(f"Failed to compile post_install_test_ACLiC.C, error code: {err.value}")

    val = ROOT.gInterpreter.MakeInterpreterValue()
    ROOT.gInterpreter.Evaluate("post_install_test_ACLiC()", val)

    return 0 if val.GetAsUnsignedLong() == 42 else 1


if __name__ == "__main__":
    raise SystemExit(main())
