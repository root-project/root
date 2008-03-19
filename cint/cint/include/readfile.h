/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* readfile.h
*
**************************************************************************/
#ifndef G__READFILE_H
#define G__READFILE_H


# ifndef G__READFILESL

#  ifdef G__SHAREDLIB
#pragma include_noerr <ReadF.dll>
#   ifndef READFILE_H
#include <ReadF.C>
#   endif
#  else
#   ifndef READFILE_H
#include <ReadF.C>
#   endif
#  endif

# endif

#endif

