/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// proc1.cxx

#include "common.cxx"

int main() {
  initsend();

  // wait for proc2 to start
  recvmsg();
  recvmsg();

  // set shared memory and send message to proc2
  send(2);
  sendmsg2("message from proc1 a");

  // wait for proc2 message
  recvmsg();
  recv();

  sendmsg2("message from proc1 b");
  recvmsg();

  finishsend();

  return(0);
}
