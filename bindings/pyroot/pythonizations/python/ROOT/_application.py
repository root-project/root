# Author: Enric Tejedor CERN  04/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import ctypes
import os
import sys
import time
import warnings


def _warning_handler(location, msg):
    # Turns ROOT warnings into Python warnings, see the call to
    # ROOT::Internal::SetWarningHandler()
    warnings.warn_explicit(msg, RuntimeWarning, location, 0, module="ROOT")


class PyROOTApplication(object):
    """
    Application class for PyROOT.
    Configures the interactive usage of ROOT from Python.

    It is created while the ROOT module is set up, so it reaches C++ through
    cppyy.gbl: attribute lookups on the ROOT module would start the setup
    again.
    """

    # The PyOS_InputHook, kept alive as long as Python may call it
    _input_hook = None

    def __init__(self, config, is_ipython):
        import cppyy

        if not cppyy.gbl.gApplication:
            self._create_application(config.IgnoreCommandLineOptions)
            self._init_root_globals()
            cppyy.gbl.ROOT.Internal.SetWarningHandler(_warning_handler)

        self._is_ipython = is_ipython

    @staticmethod
    def _create_application(ignore_cmd_line_opts):
        """
        Create the TApplication. Unless ignore_cmd_line_opts is set, it gets
        the command line options in sys.argv, up to a "-" or "--" that
        separates them from the options for the Python script. For example,
        to enable batch mode from the command line:

            python script_name.py -b -- user_arg1 ... user_argn

        or, if the script takes no options:

            python script_name.py -b
        """
        import cppyy

        gbl = cppyy.gbl

        args = ["python"]
        if not ignore_cmd_line_opts:
            for arg in getattr(sys, "argv", [])[1:]:
                if arg in ("-", "--"):
                    break
                args.append(arg)

        # The arguments as they were given, and argv[argc] null like for main()
        args = [os.fsencode(arg) for arg in args]
        argc = ctypes.c_int(len(args))
        argv = (ctypes.c_char_p * (len(args) + 1))(*args, None)
        app = gbl.TApplication("PyROOT", argc, ctypes.cast(argv, ctypes.POINTER(ctypes.c_char_p)))
        # It is gApplication from now on
        cppyy._backend.SetOwnership(app, False)

        # Prevent crashes on accessing history
        gbl.Gl_histinit("-")

        # Prevent ROOT from exiting Python
        app.SetReturnFromRun(True)

    @staticmethod
    def _init_root_globals():
        """Set up the basic ROOT globals, where not set already"""
        import cppyy

        gbl = cppyy.gbl

        # The globals own their objects, so Python must not delete them
        if not gbl.gBenchmark:
            benchmark = gbl.TBenchmark()
            cppyy._backend.SetOwnership(benchmark, False)
            gbl.gBenchmark = benchmark
        if not gbl.gStyle:
            style = gbl.TStyle()
            cppyy._backend.SetOwnership(style, False)
            gbl.gStyle = style
        # Should have been set by TApplication
        if not gbl.gProgName:
            gbl.gSystem.SetProgname("python")

    @staticmethod
    def _ipython_config():
        # Integrate IPython >= 5 with ROOT's event loop
        # Check for new GUI events until there is some user input to process

        from IPython import get_ipython
        from IPython.terminal import pt_inputhooks
        from IPython.terminal.interactiveshell import TerminalInteractiveShell

        def inputhook(context):
            import ROOT

            while not context.input_is_ready():
                ROOT.gSystem.ProcessEvents()
                time.sleep(0.01)

        pt_inputhooks.register("ROOT", inputhook)

        ipy = get_ipython()

        # Only the TerminalInteractiveShell will use the input hooks that are
        # registered via terminal.pt_inputhooks.
        if ipy and isinstance(ipy, TerminalInteractiveShell):
            get_ipython().run_line_magic("gui", "ROOT")

    @classmethod
    def _inputhook_config(cls):
        # PyOS_InputHook-based mechanism
        # Point to a function which will be called when Python's interpreter prompt
        # is about to become idle and wait for user input from the terminal
        if cls._input_hook is not None:
            return

        hook_type = ctypes.CFUNCTYPE(ctypes.c_int)
        hook_ptr = ctypes.c_void_p.in_dll(ctypes.pythonapi, "PyOS_InputHook")
        previous_hook = hook_type(hook_ptr.value) if hook_ptr.value else None

        import cppyy

        gbl = cppyy.gbl

        def process_gui_events():
            # Being a ctypes callback, this runs with the GIL held like any
            # other Python code
            pad = gbl.TVirtualPad.Pad()
            if pad and pad.IsWeb():
                pad.UpdateAsync()
            gbl.gSystem.ProcessEvents()

            return previous_hook() if previous_hook else 0

        cls._input_hook = hook_type(process_gui_events)
        hook_ptr.value = ctypes.cast(cls._input_hook, ctypes.c_void_p).value

    @staticmethod
    def _set_display_hook():
        # Set the display hook

        orig_dhook = sys.displayhook

        def displayhook(v):
            # sys.displayhook is called on the result of evaluating an expression entered
            # in an interactive Python session.
            # Therefore, this function will call EndOfLineAction after each interactive
            # command (to update display etc.)
            import ROOT

            ROOT.gInterpreter.EndOfLineAction()
            return orig_dhook(v)

        sys.displayhook = displayhook

    def init_graphics(self, gEnv, gSystem):
        """Configure ROOT graphics to be used interactively"""

        # Note that we only end up in this function if gROOT.IsBatch() is false
        import __main__

        if self._is_ipython and "IPython" in sys.modules and sys.modules["IPython"].version_info[0] >= 5:
            # ipython and notebooks, register our event processing with their hooks
            self._ipython_config()
        elif (sys.flags.interactive == 1 or not hasattr(__main__, "__file__")) and not gSystem.InheritsFrom(
            "TWinNTSystem"
        ):
            # Python in interactive mode, use the PyOS_InputHook to call our event processing
            # - sys.flags.interactive checks for the -i flags passed to python
            # - __main__ does not have the attribute __file__ if the Python prompt is started directly
            # - does not work properly on Windows
            self._inputhook_config()
            gEnv.SetValue("WebGui.ExternalProcessEvents", "yes")
        else:
            # Python in script mode, instead of separate thread methods like canvas.Update should run events

            # indicate that ProcessEvents called in different thread, let ignore thread id checks in RWebWindow
            gEnv.SetValue("WebGui.ExternalProcessEvents", "yes")

        self._set_display_hook()
