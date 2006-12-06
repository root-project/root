/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* exten.h
*  Extention for UNIX/WIN32 common API interface.
*  This source applies to UNIX.
***********************************************************************/
#ifndef G__EXTEN_H
#define G__EXTEN_H

#include "posix.h"
int isDirectory(struct dirent* pd);

#endif
