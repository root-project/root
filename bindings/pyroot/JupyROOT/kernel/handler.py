# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2015, ROOT Team.
#  Authors: Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#  website: http://oproject.org/ROOT+Jupyter+Kernel (information only for ROOT kernel)
#  Distributed under the terms of the Modified LGPLv3 License.
#
#  The full license is in the file COPYING.rst, distributed with this software.
#-----------------------------------------------------------------------------
from ctypes import CDLL, c_char_p
from threading import Thread

_lib = CDLL("libJupyROOT.so")

def _GetStream(getter):
   out_raw = getter()
   if 0 != out_raw:
      return c_char_p(out_raw).value
   return ""

class IOHandler(object):
    def __init__(self):
        _lib.JupyROOTExecutorHandler_Ctor()

    def __del__(self):
        _lib.JupyROOTExecutorHandler_Dtor()

    def Clear(self):
        _lib.JupyROOTExecutorHandler_Clear()

    def Poll(self):
        _lib.JupyROOTExecutorHandler_Poll()

    def InitCapture(self):
        _lib.JupyROOTExecutorHandler_InitCapture()

    def EndCapture(self):
        _lib.JupyROOTExecutorHandler_EndCapture()

    def GetStdout(self):
       return _GetStream(_lib.JupyROOTExecutorHandler_GetStdout)

    def GetStderr(self):
       return _GetStream(_lib.JupyROOTExecutorHandler_GetStderr)

class Runner(object):
    def __init__(self, function):
        self.function = function

    def Run(self, argument):
        return self.function(argument)

    def AsyncRun(self, argument):
        self.thread = threading.Thread(target=self.Run, args =(argument,))
        self.thread.start()

    def HasFinished(self):
        finished = False
        if self.thread:
           finished = not self.thread.is_alive()

        if finished:
           self.thread.join()
           self.thread = None

        return finished

class JupyROOTDeclarer(Runner):
    def __init__(self):
       super(JupyROOTDeclarer, self).__init__(_lib.JupyROOTDeclarer)

class JupyROOTExecutor(Runner):
    def __init__(self):
       super(JupyROOTExecutor, self).__init__(_lib.JupyROOTExecutor)

