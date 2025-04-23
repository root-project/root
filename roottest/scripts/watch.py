usage = '''Usage: watch.py .2 -- root -e "sleep(7)"'''

import os
import signal
import subprocess
import sys
import time
import threading

timeoutOffset = 5 # to give gdb the time to fire up

class AsyncExecutor(threading.Thread):
   def __init__(self, commandArgs):
      threading.Thread.__init__(self)
      self.command = commandArgs
      self.proc = subprocess.Popen(self.command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   preexec_fn=os.setsid)
   def _OutputGenerator(self):
      while True:
         lineo = self.proc.stdout.readline().rstrip()
         linee = self.proc.stderr.readline().rstrip()
         if not linee and not lineo:
            break
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
      return self.proc.poll()

def launchAndSendSignal(commandArgs, sig, timeout):
   start = time.clock()
   ae = AsyncExecutor(commandArgs)
   ae.start()
   proc = ae.GetProc()

   while True:
      rc = ae.Poll()
      if rc is not None:
         ae.join()
         return rc
      if timeout > 0 and (time.clock() - start) > timeout:
         # Here it is *fundamental* to use killpg to reach all the processes
         # in the process group. This covers cases where for example root is invoked
         # and it launches root.exe -splash
         try:
            pgid = os.getpgid(proc.pid)
         except:
            pgid = 0
         if 0 == pgid:
            rc = ae.Poll()
            ae.join()
            return rc
         print ('Timeout reached: sending %s signal to process %s' %(sig, proc.pid))
         os.killpg(pgid, sig)
         # give the time to GDB to fire up
         time.sleep(timeoutOffset)
         # here we tap again on the process group to allow the printing on screen of the
         # full stack trace built with gdb. This is a bit of black magic.
         # It is not yet clear why to flush the buffers the process group needs
         # to be terminated by hand and it does not terminate by itself.
         try:
            os.killpg(pgid, signal.SIGKILL)
         except:
            pass
         ae.join()
         return 1

   return 0

def checkArgs(timeout, commandArgs):
   if 0 == len(commandArgs):
      print ('No command to watch specified.\n%s' %usage)
      sys.exit(1)

def getArgs():
   # the rules are quite strict: %prog [timeout] -- my-command -and all --its "options until the" -end of --the --line
   args = sys.argv
   timeouts = args[1]
   if '--' != args[2]:
      print ('Second argument must be "--".\n%s' %usage)
      sys.exit(1)
   commandArgs = args[3:]
   checkArgs(timeouts, commandArgs)
   timeout = float(timeouts)
   if timeout != -1:
      timeout += timeoutOffset
      print ('Adding to the timeout a safety margin of %s seconds to allow gdb to fire up: total timeout is %s s' %( timeoutOffset, timeout))
   return timeout, commandArgs

if __name__ == "__main__":
   timeout, commandArgs = getArgs()
   sig = signal.SIGUSR2
   ret = launchAndSendSignal(commandArgs, sig, timeout)
   sys.exit(ret)
