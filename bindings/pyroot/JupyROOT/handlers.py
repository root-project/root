   # -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2015, ROOT Team.
#  Authors: Danilo Piparo
#           Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#  Distributed under the terms of the Modified LGPLv3 License.
#
#  The full license is in the file COPYING.rst, distributed with this software.
#-----------------------------------------------------------------------------
from ctypes import CDLL, c_char_p
from threading import Thread
from time import sleep as timeSleep

_lib = CDLL("libJupyROOT.so")

def _GetStream(getter):
   out_raw = getter()
   if 0 != out_raw:
      return c_char_p(out_raw).value
   return ""

class IOHandler(object):
    '''Class used to capture output from C/C++ libraries.

    >>> h = IOHandler()
    >>> h.InitCapture()
    >>> h.GetStdout()
    ''
    >>> h.GetStderr()
    ''
    >>> h.Poll()
    >>> del h
    '''

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

    def GetStreamsDicts(self):
       out = self.GetStdout()
       err = self.GetStderr()
       outDict = {'name': 'stdout', 'text': out} if out != "" else None
       errDict = {'name': 'stderr', 'text': err} if err != "" else None
       return outDict,errDict

class Runner(object):
    def __init__(self, function):
        self.function = function
        self.thread = None

    def Run(self, argument):
        return self.function(argument)

    def AsyncRun(self, argument):
        self.thread = Thread(target=self.Run, args =(argument,))
        self.thread.start()

    def Wait(self):
        if not self.thread: return
        self.thread.join()

    def HasFinished(self):
        if not self.thread: return True

        finished = not self.thread.is_alive()
        if not finished: return False

        self.thread.join()
        self.thread = None

        return True


class JupyROOTDeclarer(Runner):
    def __init__(self):
       super(JupyROOTDeclarer, self).__init__(_lib.JupyROOTDeclarer)

class JupyROOTExecutor(Runner):
    def __init__(self):
       super(JupyROOTExecutor, self).__init__(_lib.JupyROOTExecutor)

def RunAsyncAndPrint(executor, code, ioHandler, printFunction, silent = False, timeout = 0.1):
   ioHandler.Clear()
   ioHandler.InitCapture()
   executor.AsyncRun(code)
   while not executor.HasFinished():
         ioHandler.Clear()
         ioHandler.Poll()
         if not silent:
            printFunction(ioHandler)
         if executor.HasFinished(): break
         timeSleep(.1)
   executor.Wait()
   ioHandler.EndCapture()
