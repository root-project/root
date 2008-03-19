/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__REGEX_H
#define G__REGEX_H
/*
*  regex.h dummy file
*/

class regex_t {
  regex_t() {fprintf(stderr,"Limitation: regex not supported\n");}
};

int regcomp()
{
}

int regexec()
{
}

void regfree()
{
}

#endif
