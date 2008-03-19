/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

void test1() {
  int cmd_no=-1;
  while (cmd_no!=-3) {
    printf("Cmd_no (start)=%d\n",cmd_no);
    switch(cmd_no)  {
    case 0: cmd_no=-3;continue;
    case -1: cmd_no=0;continue;
    }
    // should never get to here....
    printf("Cmd_no (end) =%d\n",cmd_no);
    /* if(1) continue;
       printf("Cmd_no (end) =%d\n",cmd_no); */
  }
}

void test2() {
  int cmd_no=-1;
  while (cmd_no!=-3) {
    printf("Cmd_no (start)=%d\n",cmd_no);
    switch(cmd_no)  {
    case 0: cmd_no=-3;break;
    case -1: cmd_no=0;break;
    }
    // should never get to here....
    printf("Cmd_no (end) =%d\n",cmd_no);
    /* if(1) continue;
       printf("Cmd_no (end) =%d\n",cmd_no); */
  }
}

int main() {
  test1();
  test2();
  return 0;
}




