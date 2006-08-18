/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*****************************************************************************
* RegExp.h
*
*
*****************************************************************************/

#ifndef REGEXP_H
#define REGEXP_H

#define G__REGEXPSL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>


class RegExp {
 public:
  RegExp(char *pattern) { regcomp(&re,pattern,REG_EXTENDED|REG_NOSUB); }
  ~RegExp() { regfree(&re); }
  int match(char *string) {
    return(!regexec(&re,string,(size_t)0,(regmatch_t*)NULL,0));
  }
 private:
  regex_t re;
};

int matchregex(char *pattern,char *string);
int operator==(RegExp& ex,char *string) ;
int operator!=(RegExp& ex,char *string) ;
int operator==(char *string,RegExp& ex) ;
int operator!=(char *string,RegExp& ex) ;


#endif

