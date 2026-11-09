# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------
#  Authors: Danilo Piparo
#           Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
# -----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import codecs
import contextvars
import ctypes
import os
import queue
import sys
import threading
import time
from threading import Thread
from time import sleep as timeSleep

from ROOT._jupyroot import helpers

# C stdio, whose buffers have to be flushed for output to reach the pipes
if sys.platform == "win32":
    _crt = ctypes.cdll.ucrtbase
else:
    _crt = ctypes.CDLL(None)
_crt.fflush.argtypes = [ctypes.c_void_p]

# How long to wait at the end of a capture for the rest of the output. It only
# takes this long when a process started during the capture still holds the
# pipe, so that its end is never reached.
_END_CAPTURE_TIMEOUT = 1.0


class _FileDescriptorCapture(object):
    """Redirects a file descriptor into a pipe and collects what is written
    to it, also by C and C++ code.

    A thread reads the pipe for as long as the capture runs, so that writers
    never wait on a full pipe. A writer of C stdio does so holding the lock of
    the stream, and flushing the stream to get its output would then wait for
    that lock forever.
    """

    def __init__(self, fd):
        self._fd = fd
        self._saved_fd = None
        self._write_fd = None
        self._reader = None
        self._lock = threading.Lock()
        self._text = ""

    @property
    def text(self):
        with self._lock:
            return self._text

    def clear(self):
        with self._lock:
            self._text = ""

    def start(self):
        self._saved_fd = os.dup(self._fd)
        read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, self._fd)
        self._reader = threading.Thread(target=self._read, args=(read_fd,), name="JupyROOT capture", daemon=True)
        self._reader.start()

    def _read(self, read_fd):
        # Decode incrementally: a read can end in the middle of a character
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                text = decoder.decode(chunk, final=not chunk)
                with self._lock:
                    # After a timeout in stop(), the output belongs to no
                    # capture anymore
                    if self._reader is threading.current_thread():
                        self._text += text
                if not chunk:
                    break
        finally:
            os.close(read_fd)

    def stop(self):
        """Restore the file descriptor; call wait() for the rest of the output"""
        os.dup2(self._saved_fd, self._fd)
        os.close(self._saved_fd)
        os.close(self._write_fd)
        self._saved_fd = self._write_fd = None

    def wait(self, deadline):
        """Wait until the output written before stop() is collected, or until
        the deadline (from time.monotonic()) has passed"""
        self._reader.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._reader = None


class IOHandler(object):
    r"""Class used to capture output from C/C++ libraries.
    >>> import sys
    >>> h = IOHandler()
    >>> h.GetStdout()
    ''
    >>> h.GetStderr()
    ''
    >>> h.GetStreamsDicts()
    (None, None)
    >>> del h
    """

    def __init__(self):
        import ROOT

        # Fixes for ROOT-7999
        ROOT.SetErrorHandler(ROOT.DefaultErrorHandler)

        self._stdout = _FileDescriptorCapture(1)
        self._stderr = _FileDescriptorCapture(2)
        self._capturing = False

    def __del__(self):
        self.EndCapture()

    def Clear(self):
        self._stdout.clear()
        self._stderr.clear()

    def Poll(self):
        if self._capturing:
            # The reader threads collect what reaches the pipes
            _crt.fflush(None)

    def InitCapture(self):
        if not self._capturing:
            self._stdout.start()
            self._stderr.start()
            self._capturing = True

    def EndCapture(self):
        if self._capturing:
            _crt.fflush(None)
            self._stdout.stop()
            self._stderr.stop()
            deadline = time.monotonic() + _END_CAPTURE_TIMEOUT
            self._stdout.wait(deadline)
            self._stderr.wait(deadline)
            self._capturing = False

    def GetStdout(self):
        return self._stdout.text

    def GetStderr(self):
        return self._stderr.text

    def GetStreamsDicts(self):
        out = self.GetStdout()
        err = self.GetStderr()
        outDict = {"name": "stdout", "text": out} if out != "" else None
        errDict = {"name": "stderr", "text": err} if err != "" else None
        return outDict, errDict


class Poller(Thread):
    def __init__(self):
        # Run in a fresh context instead of a copy of the creator's, which is
        # the default since Python 3.14 in free-threaded builds. The ipykernel
        # output streams keep the parent message header in a context variable,
        # and a copy taken at kernel startup would pin it to an empty header
        # for the lifetime of this thread. Output written from here would then
        # not be associated with the executing cell, and therefore be lost.
        kwargs = {"context": contextvars.Context()} if sys.version_info >= (3, 14) else {}
        Thread.__init__(self, group=None, target=None, name="JupyROOT Poller Thread", **kwargs)
        self.daemon = True
        self.poll = True
        self.is_running = False
        self.queue = queue.Queue()

    def run(self):
        while self.poll:
            work_item = self.queue.get()
            if work_item is not None:
                function, argument = work_item
                self.is_running = True
                function(argument)
                self.is_running = False
            else:
                self.poll = False

    def Stop(self):
        if self.is_alive():
            self.queue.put(None)
            self.join()


class Runner(object):
    """Asynchrously run functions
    >>> import time
    >>> def f(code):
    ...    print(code)
    >>> p = Poller(); p.start()
    >>> r= Runner(f, p)
    >>> r.Run("ss")
    ss
    >>> r.AsyncRun("ss");time.sleep(1)
    ss
    >>> def g(msg):
    ...    time.sleep(.25)
    ...    print(msg)
    >>> r= Runner(g, p)
    >>> r.AsyncRun("Asynchronous");print("Synchronous");time.sleep(1)
    Synchronous
    Asynchronous
    >>> r.AsyncRun("Asynchronous"); print(r.HasFinished())
    False
    >>> time.sleep(1)
    Asynchronous
    >>> print(r.HasFinished())
    True
    >>> p.Stop()
    """

    def __init__(self, function, poller):
        self.function = function
        self.poller = poller

    def Run(self, argument):
        return self.function(argument)

    def AsyncRun(self, argument):
        self.poller.is_running = True
        self.poller.queue.put((self.Run, argument))

    def Wait(self):
        while self.poller.is_running:
            timeSleep(0.1)

    def HasFinished(self):
        return not self.poller.is_running


def _report_exception(location, e):
    # Report exceptions that escape the interpreted code like TRint does at
    # the ROOT prompt (see TRint::HandleTermInput), instead of swallowing them
    # silently (ROOT-10589)
    import ROOT

    if isinstance(e, ROOT.std.exception):
        message = "{} caught: {}".format(type(e).__cpp_name__, e.what())
        ROOT.Error(location, message.replace("%", "%%"))
    else:
        ROOT.Error(location, "Exception caught!")


def _jupyroot_execute(code):
    import ROOT

    status = False
    try:
        err = ctypes.c_uint(ROOT.TInterpreter.kNoError)
        if ROOT.gInterpreter.ProcessLine(code, err):
            status = True
        if err.value == ROOT.TInterpreter.kProcessing:
            ROOT.gInterpreter.ProcessLine(".@")
            ROOT.gInterpreter.ProcessLine('cerr << "Unbalanced braces. This cell was not processed." << endl;')
    except Exception as e:
        _report_exception("JupyROOTExecutor", e)
    return int(status)


def _jupyroot_declare(code):
    import ROOT

    status = False
    try:
        status = bool(ROOT.gInterpreter.Declare(code))
    except Exception as e:
        _report_exception("JupyROOTDeclarer", e)
    return int(status)


class JupyROOTDeclarer(Runner):
    """Asynchrously execute declarations
    >>> import ROOT
    >>> p = Poller(); p.start()
    >>> d = JupyROOTDeclarer(p)
    >>> d.Run("int f(){return 3;}")
    1
    >>> ROOT.f()
    3
    >>> p.Stop()
    """

    def __init__(self, poller):
        super(JupyROOTDeclarer, self).__init__(_jupyroot_declare, poller)


class JupyROOTExecutor(Runner):
    r"""Asynchrously execute process lines
    >>> import ROOT
    >>> p = Poller(); p.start()
    >>> d = JupyROOTExecutor(p)
    >>> d.Run('cout << "Here am I" << endl;')
    1
    >>> p.Stop()
    """

    def __init__(self, poller):
        super(JupyROOTExecutor, self).__init__(_jupyroot_execute, poller)


def display_drawables(displayFunction):
    drawers = helpers.utils.GetDrawers()
    for drawer in drawers:
        drawer.Draw(displayFunction)


class JupyROOTDisplayer(Runner):
    """Display all canvases"""

    def __init__(self, poller):
        super(JupyROOTDisplayer, self).__init__(display_drawables, poller)


def RunAsyncAndPrint(executor, code, ioHandler, printFunction, displayFunction, silent=False, timeout=0.1):
    ioHandler.Clear()
    ioHandler.InitCapture()
    executor.AsyncRun(code)
    while not executor.HasFinished():
        ioHandler.Poll()
        if not silent:
            printFunction(ioHandler)
            ioHandler.Clear()
        if executor.HasFinished():
            break
        timeSleep(0.1)
    executor.Wait()
    ioHandler.EndCapture()


def Display(displayer, displayFunction):
    displayer.AsyncRun(displayFunction)
    displayer.Wait()
