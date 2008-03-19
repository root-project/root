demo/ipc/README.txt

 This directory contains IPC programming example using CINT.
Cint supports IPC library in lib/ipc directory. You need to build
include/sys/ipc.dll in order to run IPC demo program.

FILES: 
  README.txt   : This file
  common.cxx   : IPC example common routine
  proc1.cxx    : IPC example program 1
  proc2.cxx    : IPC example program 2
  keyfile      : keyfile for ftok (file to key)

RUNNING THE DEMO:

 One terminal
  $  cint proc1.cxx

 Another terminal
  $  cint proc2.cxx


