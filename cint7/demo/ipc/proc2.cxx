/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// proc2.cxx

#include "common.cxx"

int main() {
  initrecv();

  // resume proc1
  sendmsg("message from proc2 a");
  sendmsg("message from proc2 b");

  // wait for proc1 message
  recvmsg2();
  recv();

  // set shared memory and send message to proc1
  send(5);
  sendmsg("message from proc2 c");

  recvmsg2();
  sendmsg("message from proc2 d");
  sleep(1);

  finishrecv();

  return(0);
}
