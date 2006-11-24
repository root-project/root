/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
 * cxx2cpp.c
 *  Script for changing C++ file extension to cpp. Bash on Windows did
 *  not work well for cxx2cpp script. So, I implemented this portion
 *  in C. This script is specifically made for Borland C++ Builder debugging,
 *  hence BC++ does not accept .cxx as C++ source. You also need to change
 *  platform/borland/libcint.cpp which is done in cxx2cpp shell script.
 ************************************************************************/
#include <stdio.h>
#include <string.h>

int main(int argc,char** argv) {
  char *p;
  char buf[100];
  for(int i=1;i<argc;i++) {
    if((p=strstr(argv[i],".cxx"))) {
      strcpy(buf,argv[i]);
      p=strstr(buf,".cxx");
      strcpy(p,".cpp");
      rename(argv[i],buf);
    }
  }
  return(0);
}

