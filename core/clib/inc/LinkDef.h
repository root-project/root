/* @(#)root/clib:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifdef __CINT__

#pragma link C++ enum EGetLineMode;

#pragma link C++ function Getline(char*);
#pragma link C++ function Getlinem(EGetLineMode,char*);
#pragma link C++ function Gl_histadd(char*);

#pragma link C++ function strlcpy(char *, const char *, size_t);
#pragma link C++ function strlcat(char *, const char *, size_t);
#pragma link C++ function snprintf(char *, size_t, const char *, ...);

// Over-ride the CINT hand coded dictionary to allow for full
// parameter conversion resolution.
char *strtok(char *str, const char *delim);
#pragma link C++ function strtok;

#endif
