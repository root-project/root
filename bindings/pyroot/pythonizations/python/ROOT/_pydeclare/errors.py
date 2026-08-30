# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""Errors raised by the Python-to-C++ transpiler backends.

The guiding principle of this package is that anything outside the supported
subset must fail *loudly*, at declaration time, with a message that points at
the offending line of Python source.  Silently approximating Python semantics
in C++ would mean silently wrong physics results, which is far worse than an
error.
"""

import textwrap


class PyDeclareError(Exception):
    """Raised when a Python callable cannot be translated to C++.

    The message carries the location in the user's Python source whenever the
    backend was able to determine it.
    """


def _source_context(func, lineno, col_offset=None):
    """Return a '  File "...", line N' block quoting the offending source line."""
    import inspect

    try:
        lines, first = inspect.getsourcelines(func)
        filename = inspect.getsourcefile(func) or "<unknown>"
    except (OSError, TypeError):
        return ""

    # inspect gives 1-based file line numbers; AST line numbers are relative to
    # the start of the (dedented) function source.
    if lineno is None:
        return '  File "{}", line {}\n'.format(filename, first)

    abs_lineno = first + lineno - 1
    idx = lineno - 1
    text = lines[idx].rstrip("\n") if 0 <= idx < len(lines) else ""
    stripped = text.strip()
    out = '  File "{}", line {}\n    {}\n'.format(filename, abs_lineno, stripped)
    if col_offset is not None and text:
        caret_col = col_offset - (len(text) - len(text.lstrip()))
        if caret_col >= 0:
            out += "    " + " " * caret_col + "^\n"
    return out


def raise_unsupported(func, node, what, hint=None):
    """Raise a PyDeclareError for an unsupported construct at an AST node."""
    lineno = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    msg = "{}\n{}".format(what, _source_context(func, lineno, col))
    if hint:
        msg += textwrap.indent("\n" + hint.strip() + "\n", "  ")
    raise PyDeclareError(msg.rstrip() + "\n")
