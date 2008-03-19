/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*****************************************************************************
* ReadFile.h
*
*
*
*****************************************************************************/

#ifndef READFILE_H
#define READFILE_H

#define G__READFILESL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef G__NEVER
#include <regex.h>
#include "Common.h"
#endif


#define MAX_LINE_LENGTH  500
#define MAX_TOKEN        100
#define MAX_SEPARATOR    20
#define MAX_ENDOFLINE    10

class ReadFile {
 public:
  int argc;               // number of arguments in one line
  char *argv[MAX_TOKEN];  // argument buffer
  int line;               // line number

  ReadFile(const char *filename);
  ReadFile(FILE *fpin);
  ~ReadFile();

  int read();
  int readword();
#ifdef G__NEVER
  int regex(char *pattern,char *string=(char*)NULL);
#endif
  void setseparator(const char *separatorin); 
  void setendofline(const char *endoflinein); 

  int isvalid() { if(fp) return(1); else return(0); }
  void disp();

 private:
  FILE *fp;
  int openflag;

  char buf[MAX_LINE_LENGTH];
  char argbuf[MAX_LINE_LENGTH];

  char separator[MAX_SEPARATOR];
  int lenseparator;
  char endofline[MAX_ENDOFLINE];
  int lenendofline;

  void separatearg(void);
  void initialize();
  int isseparator(int c);
  int isendofline(int c);
};


#endif

