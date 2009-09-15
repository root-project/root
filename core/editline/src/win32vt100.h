// @(#)root/editline:$Id$
// Author: Axel Naumann, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#define ERR 0
typedef int (*PutcFunc_t)(int);

char* tigetstr(const char*);
int tputs(const char* what, int, PutcFunc_t putc);
char* tparm(const char* termstr, ...);
int setupterm(const char* /*term*/, int fd, int* errcode);
