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


#define MAX_LINE_LENGTH  8048
#define MAX_TOKEN        1024
#define MAX_SEPARATOR    32
#define MAX_ENDOFLINE    10
#ifndef G__OLDIMPLEMENTATION3000
#define MAX_QUOTATION    10
#endif

class ReadFile {
 public:
  int argc;               // number of arguments in one line
  char *argv[MAX_TOKEN];  // argument buffer
  int line;               // line number

  ReadFile(const char *filename);
  ReadFile(FILE *fpin);
  ~ReadFile();

  void parse(const char* s) {
    strcpy(buf,s);
    separatearg();
  }

  int read();
  int readword();
#ifdef G__NEVER
  int regex(char *pattern,char *string=(char*)NULL);
#endif
  void setseparator(const char *separatorin); 
#ifndef G__OLDIMPLEMENTATION1960
  void setdelimitor(const char *delimitorin); 
#endif
  void setendofline(const char *endoflinein); 
#ifndef G__OLDIMPLEMENTATION3000
  void setquotation(const char *quotationin); 
#endif

  int isvalid() { if(fp) return(1); else return(0); }
  void disp();

 private:
  FILE *fp;
  int openflag;

  char buf[MAX_LINE_LENGTH];
  char argbuf[MAX_LINE_LENGTH];

  char separator[MAX_SEPARATOR];
  int lenseparator;
#ifndef G__OLDIMPLEMENTATION1960
  char delimitor[MAX_SEPARATOR];
  int lendelimitor;
#endif
  char endofline[MAX_ENDOFLINE];
  int lenendofline;
#ifndef G__OLDIMPLEMENTATION3000
  char quotation[MAX_SEPARATOR];
  int lenquotation;
#endif

  void separatearg(void);
  void initialize();
  int isseparator(int c);
#ifndef G__OLDIMPLEMENTATION1960
  int isdelimitor(int c);
#endif
  int isendofline(int c);
#ifndef G__OLDIMPLEMENTATION3000
  int isquotation(int c);
#endif
};


#endif

