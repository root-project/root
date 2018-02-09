#! /usr//bin/env python

usage = '''Usage: watch.py .2 -- root -e "sleep(7)"'''

import os
import signal
import subprocess
import sys
import time
import threading

timeoutOffset = 5 # to give gdb the time to fire up

class AsyncExecutor(threading.Thread):
   def __init__(self, command):
      threading.Thread.__init__(self)
      self.command = command.split()
      self.canPoll = True
      self.proc = subprocess.Popen(self.command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   preexec_fn=os.setsid)
   def _OutputGenerator(self):
      while True:
         self.canPoll = False
         lineo = self.proc.stdout.readline().rstrip()
         linee = self.proc.stderr.readline().rstrip()
         if not linee and not lineo:
            break
         self.canPoll = True
         yield linee, lineo

   def Print(self):
      for lines in self._OutputGenerator():
         for line in lines:
            if line != "":
               print (line)

   def run(self):
      self.Print()

   def GetProc(self):
      return self.proc

   def Poll(self):
      return self.proc.poll() if self.canPoll else None

def launchAndSendSignal(command, sig, timeout):
   start = time.clock()
   ae = AsyncExecutor(command)
   ae.start()
   proc = ae.GetProc()

   while True:
      rc = ae.Poll()
      if rc is not None:
         ae.join()
         return rc
      if timeout > 0 and (time.clock() - start) > timeout:
         print ('Timeout reached: sending %s signal to process %s' %(sig, proc.pid))
         # Here it is *fundamental* to use killpg to reach all the processes
         # in the process group. This covers cases where for example root is invoked
         # and it launches root.exe -splash
         pgid = os.getpgid(proc.pid)
         os.killpg(pgid, sig)
         # give the time to GDB to fire up
         time.sleep(timeoutOffset)
         # here we tap again on the process group to allow the printing on screen of the
         # full stack trace built with gdb. This is a bit of black magic.
         # It is not yet clear why to flush the buffers the process group needs
         # to be terminated by hand and it does not terminate by itself.
         os.killpg(pgid, signal.SIGKILL)
         ae.join()
         return 1

   return 0

def checkArgs(timeout, command):
   if "" == command:
      print ('No command to watch specified.\n%s' %usage)
      sys.exit(1)

def getArgs():
   # the rules are quite strict: %prog [timeout] -- my-command -and all --its "options until the" -end of --the --line
   args = sys.argv
   timeouts = args[1]
   if '--' != args[2]:
      print ('Second argument must be "--".\n%s' %usage)
      sys.exit(1)
   command = " ".join(args[3:])
   checkArgs(timeouts, command)
   timeout = float(timeouts)
   if timeout != -1:
      timeout += timeoutOffset
      print ('Adding to the timeout a safety margin of %s seconds to allow gdb to fire up: total timeout is %s' %( timeoutOffset, timeout))
   return timeout, command

if __name__ == "__main__":
   timeout, command = getArgs()
   sig = signal.SIGUSR2
   ret = launchAndSendSignal(command, sig, timeout)
   sys.exit(ret)
