// @(#)root/editline:$Id$
// Author: Axel Naumann, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __sun
# include R__CURSESHDR
extern "C" {
   // cannot #include term.h because it #defines move() etc
   char *tparm(char*, long, long, long, long, long, long, long, long, long);
   char *tigetstr(char*);
   int tigetnum(char*);
   char *tgoto(char*, int, int);
   int   tputs(char*, int, int (*)(int));
   int tgetflag(char*);
   int tgetnum(char*);
   char* tgetstr(char*, char**);
   int tgetent(char*, const char*);
}
#else
extern "C" {
# include R__CURSESHDR
   // some curses.h include a curses-version of termcap.h which
   // conflicts with the system one:
# ifndef _TERMCAP_H
#  include <termcap.h>
# endif
int setupterm(const char* term, int fd, int* perrcode);
}
#endif

// un-be-lievable: termcap.h / term.h often #define these.
# ifdef erase
#  undef erase
# endif
# ifdef move
#  undef move
# endif
# ifdef clear
#  undef clear
# endif
# ifdef del
#  undef del
# endif
# ifdef key_end
#  undef key_end
# endif
# ifdef key_clear
#  undef key_clear
# endif
# ifdef key_print
#  undef key_print
# endif
