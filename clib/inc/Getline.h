/* @(#)root/clib:$Name:  $:$Id: Getline.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $ */
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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { kInit = -1, kLine1, kOneChar, kCleanUp } EGetLineMode;

char *Getline(char *prompt);
char *Getlinem(EGetLineMode mode, char *prompt);
void Gl_config(const char *which, int value);
void Gl_setwidth(int width);
void Gl_windowchanged();
void Gl_histinit(char *file);
void Gl_histadd(char *buf);
int  Gl_eof();

char *strip(char *line);

R__EXTERN int (*gl_in_hook)(char *buf);
R__EXTERN int (*gl_out_hook)(char *buf);
R__EXTERN int (*gl_tab_hook)(char *buf, int prompt_width, int *cursor_loc);

#ifdef __cplusplus
}
#endif

#endif   /* _GETLINE_ */
