# Regression test for https://github.com/root-project/root/issues/24176
#
# The names are only known from the rootmap, and finding them requires parsing
# their header. The first lookup parses the whole header, so each kind of name
# is checked in its own process.

import sys

kind = sys.argv[-1]

if kind == "enum":
    from ROOT.gh_24176.ns import EnumInNamespace

    assert EnumInNamespace.kSecond == 2
elif kind == "variable":
    from ROOT.gh_24176.ns import kConstant

    assert kConstant == 3
else:
    raise ValueError(f"unknown kind {kind!r}")
