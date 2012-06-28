/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

class SystemScript {
 public:
  SystemScript(const char* cname,const char* objname) {}
};

class System {
 public:
  static void Print(const char* x) {
    printf("%s\n",x);
  }
};

class EditorScript {
 public:
  EditorScript(const char* cname,const char* objname) { }
  void BeginTerrainPaint() { printf("BeginTerrainPaint()\n"); }
};


int x=0;
class Math {
 public:
  static int rand() { return(x++); }
};
