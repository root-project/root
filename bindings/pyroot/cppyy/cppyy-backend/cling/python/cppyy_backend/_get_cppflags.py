import os

MYHOME = os.path.dirname(__file__)

def get_cppflags():
  # add flags from allCppflags.txt (mainly for C++ standard choice)
    extra_flags = None
    ct_flags = os.path.join(MYHOME, 'etc', 'dictpch', 'allCppflags.txt')
    try:
        with open(ct_flags) as f:
            all_flags = f.readlines()
        if "std" in os.environ.get("EXTRA_CLING_ARGS", ""):
            keep = []
            for flag in all_flags:
                if not "std" in flag:
                    keep.append(flag)
            all_flags = keep
        extra_flags = " ".join(map(lambda line: line[:-1], all_flags))
    except (OSError, IOError):
        pass

    return extra_flags

def get_cppversion():
  # requested C++ version based on flags or environment variables
    try:
        return os.environ['STDCXX']
    except KeyError:
        pass

    pstd = -1
    try:
        flags = os.environ['EXTRA_CLING_ARGS']
        pstd = flags.find("std")
    except KeyError:
        pass

    if pstd < 0:
        flags = get_cppflags()
        pstd = flags.find("std")

    if 0 < pstd:
      # syntax is "-std=c++XY" on Linux/Mac and "/std:c++NM" on Windows
        return flags[pstd+7:pstd+9]

    return "20"    # default but should never happen anyway

