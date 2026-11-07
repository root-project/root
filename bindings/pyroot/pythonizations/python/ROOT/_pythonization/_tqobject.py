# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\pythondoc TQObject

The `Connect` method of TQObject (and thus of any class deriving from it, like
the GUI widgets) directly accepts a Python callable as the slot:

\code{.py}
import ROOT

def on_clicked():
    print("button clicked")

button = ROOT.TGTextButton(parent, "&Draw", 10)
button.Connect("Clicked()", on_clicked)
\endcode

The arguments emitted by the signal are forwarded to the callable, as far as
its signature accepts them. The connection keeps the callable alive; it can be
undone with `Disconnect`, passing the same signal and callable:

\code{.py}
button.Disconnect("Clicked()", on_clicked)
\endcode

The C++ signature `Connect(signal, receiver_class, receiver, slot)` remains
available.

\endpythondoc
*/
"""

import inspect
import traceback

from . import pythonization

# Dispatcher instances of active connections, keyed by (sender address, signal,
# callable). TQObject stores the receiver as a raw pointer, so the dispatcher
# that wraps the Python callable must be kept alive as long as it is connected.
_dispatchers = {}

# Dispatcher classes, keyed by the C++ prototype of the signal arguments that
# their Dispatch method forwards.
_dispatcher_classes = {}


def _dispatcher_class(arg_types):
    """Get or create a dispatcher class whose Dispatch(<arg_types>) method
    forwards to a wrapped Python callable. The class is generated in the
    interpreter, which is what allows connecting a TQObject signal to it by
    class name; it holds the callable as a std::function, whose conversion
    from a Python callable and calling back into Python are done by cppyy."""
    import cppyy

    proto = ", ".join(arg_types)
    klass = _dispatcher_classes.get(proto)
    if klass is None:
        name = "TPyDispatcher_{}".format(len(_dispatcher_classes))
        params = ", ".join("{} a{}".format(t, i) for i, t in enumerate(arg_types))
        forwarded = ", ".join("a{}".format(i) for i in range(len(arg_types)))
        cppyy.cppdef(
            """
#include <functional>
#include <utility>

namespace PyROOT {{

class {name} {{
public:
   {name}(std::function<void({proto})> callable) : fCallable(std::move(callable)) {{}}
   void Dispatch({params}) {{ fCallable({forwarded}); }}

private:
   std::function<void({proto})> fCallable;
}};

}} // namespace PyROOT
""".format(name=name, proto=proto, params=params, forwarded=forwarded)
        )
        klass = getattr(cppyy.gbl.PyROOT, name)
        _dispatcher_classes[proto] = klass
    return klass


def _connection_key(sender, signal, callable_):
    import cppyy

    return (cppyy.addressof(sender), signal.replace(" ", ""), callable_)


def _max_accepted_args(callable_):
    """Maximum number of positional arguments the callable accepts, or None if
    unbounded or undeterminable."""
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return None
    nmax = 0
    for par in signature.parameters.values():
        if par.kind in (par.POSITIONAL_ONLY, par.POSITIONAL_OR_KEYWORD):
            nmax += 1
        elif par.kind == par.VAR_POSITIONAL:
            return None
    return nmax


def _signal_arg_types(signal, callable_):
    """The argument types the signal emits, truncated to what the callable
    accepts."""
    lpar = signal.find("(")
    rpar = signal.rfind(")")
    if lpar < 0 or rpar < lpar:
        raise ValueError('signal "{}" lacks the argument list, e.g. "Clicked()"'.format(signal))
    proto = signal[lpar + 1 : rpar].strip()
    arg_types = [a.strip() for a in proto.split(",")] if proto else []
    nmax = _max_accepted_args(callable_)
    return arg_types if nmax is None else arg_types[:nmax]


def _print_errors(callable_):
    """Print exceptions from the callable instead of letting them escape: the
    slot is invoked from C++ signal emission, which exceptions cannot safely
    propagate through."""

    def wrapper(*args):
        try:
            callable_(*args)
        except Exception:
            traceback.print_exc()

    return wrapper


def _is_new_style_args(args, kwargs):
    return not kwargs and len(args) == 2 and not isinstance(args[1], str) and callable(args[1])


def _TQObject_Connect(self, *args, **kwargs):
    if isinstance(self, str):
        # Static overload connecting by sender class name, called unbound,
        # e.g. TQObject.Connect("TGButton", "Clicked()", ...)
        import cppyy

        return cppyy.gbl.TQObject._OriginalConnect(self, *args, **kwargs)
    if _is_new_style_args(args, kwargs):
        signal, callable_ = args
        arg_types = _signal_arg_types(signal, callable_)
        wrapper = _print_errors(callable_)
        dispatcher = _dispatcher_class(arg_types)(wrapper)
        # The std::function holds only a borrowed reference to the Python
        # callable, so tie the wrapper's lifetime to the dispatcher
        dispatcher._callable = wrapper
        result = self._OriginalConnect(
            signal, type(dispatcher).__cpp_name__, dispatcher, "Dispatch({})".format(", ".join(arg_types))
        )
        if result:
            _dispatchers.setdefault(_connection_key(self, signal, callable_), []).append(dispatcher)
        return result
    return self._OriginalConnect(*args, **kwargs)


def _TQObject_Disconnect(self, *args, **kwargs):
    if isinstance(self, str):
        import cppyy

        return cppyy.gbl.TQObject._OriginalDisconnect(self, *args, **kwargs)
    if _is_new_style_args(args, kwargs):
        signal, callable_ = args
        dispatchers = _dispatchers.pop(_connection_key(self, signal, callable_), None)
        if not dispatchers:
            return False
        result = False
        for dispatcher in dispatchers:
            result = self._OriginalDisconnect(signal, dispatcher) or result
        return result
    return self._OriginalDisconnect(*args, **kwargs)


@pythonization("TQObject")
def pythonize_tqobject(klass):
    klass._OriginalConnect = klass.Connect
    klass.Connect = _TQObject_Connect
    klass._OriginalDisconnect = klass.Disconnect
    klass.Disconnect = _TQObject_Disconnect
