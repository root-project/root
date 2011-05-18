/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* exten.c
*  Extention for UNIX/WIN32 common API interface.
*  This source applies to UNIX.
***********************************************************************/
#include <unistd.h>
#include "exten.h"

int isDirectory(struct dirent* pd) {
  struct stat st;
  if(stat(pd->d_name,&st)) return(0);
  else return(S_ISDIR(st.st_mode));
}

