/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// File t02a.C
//#include "TROOT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int induce() {
   int *p;
   char line[200];
   sprintf(line,
  "int** R__Event_fTracks1 = (int**)(%ld); *R__Event_fTracks1 = new int(3);",
           &p);
   G__exec_text(line);
   //gROOT->ProcessLine(line); // eventually call G__process_cmd
   return *p;
}

class myclass {
public:
   void run() {
      int a = induce();
      printf("created an int of value %d\n",a);

   }
   operator int() { return induce(); };
   myclass() { } //run(); }
};
