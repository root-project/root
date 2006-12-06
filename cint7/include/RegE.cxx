/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*****************************************************************************
* RegExp.C
*
*
*****************************************************************************/
#include "RegE.h"

/**************************************************************************
* matchtregex()
**************************************************************************/
int matchregex(char *pattern,char *string)
{
  int i;
  regex_t re;
  i=regcomp(&re,pattern,REG_EXTENDED|REG_NOSUB);
  if(i!=0) return(0); 
  i=regexec(&re,string,(size_t)0,(regmatch_t*)NULL,0);
  regfree(&re);
  if(i!=0) return(0); 
  return(1); /* match */
}


int operator==(RegExp& ex,char *string) 
{
  return(ex.match(string));
}

int operator!=(RegExp& ex,char *string) 
{
  return(!ex.match(string));
}

int operator==(char *string,RegExp& ex) 
{
  return(ex.match(string));
}

int operator!=(char *string,RegExp& ex) 
{
  return(!ex.match(string));
}

