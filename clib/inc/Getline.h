/* @(#)root/clib:$Name:  $:$Id: Getline.h,v 1.4 2001/06/22 16:10:16 rdm Exp $ */
/* Author: */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef _GETLINE_
#define _GETLINE_

#ifndef __CINT__
#ifdef __cplusplus
extern "C" {
#endif
#endif

typedef enum { kInit = -1, kLine1, kOneChar, kCleanUp } EGetLineMode;

char *Getline(const char *prompt);
char *Getlinem(EGetLineMode mode, const char *prompt);
void Gl_config(const char *which, int value);
void Gl_setwidth(int width);
void Gl_windowchanged();
void Gl_histinit(char *file);
void Gl_histadd(char *buf);
int  Gl_eof();

R__EXTERN int (*Gl_in_hook)(char *buf);
R__EXTERN int (*Gl_out_hook)(char *buf);
R__EXTERN int (*Gl_tab_hook)(char *buf, int prompt_width, int *cursor_loc);

#ifndef __CINT__
#ifdef __cplusplus
}
#endif
#endif

#endif   /* _GETLINE_ */
