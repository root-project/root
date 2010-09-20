// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: history.c,v 1.17 2001/03/20 00:08:31 christos Exp $	*/

/*-
 * Copyright (c) 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Christos Zoulas of Cornell University.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "compat.h"

/*
 * hist.c: History access functions
 */
#include "sys.h"

#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#ifdef HAVE_VIS_H
# include <vis.h>
#endif
#include <sys/stat.h>

#include <string>
#include <fstream>

#include "histedit.h"

typedef int (*HistoryGFun_t)(ptr_t, HistEvent_t*);
typedef int (*HistoryEFun_t)(ptr_t, HistEvent_t*, const char*);
typedef void (*HistoryVFun_t)(ptr_t, HistEvent_t*);
typedef int (*HistorySFun_t)(ptr_t, HistEvent_t*, const int);

struct HistoryFcns_t {
   ptr_t fRef;                 /* Argument for history fcns	 */
   int fEnt;                   /* Last entry point for history	 */
   HistoryGFun_t fFirst;      /* Get the first element	 */
   HistoryGFun_t fNext;       /* Get the next element		 */
   HistoryGFun_t fLast;       /* Get the last element		 */
   HistoryGFun_t fPrev;       /* Get the previous element	 */
   HistoryGFun_t fCurr;       /* Get the current element	 */
   HistorySFun_t fSet;        /* Set the current element	 */
   HistoryVFun_t fClear;      /* Clear the history list	 */
   HistoryEFun_t fEnter;      /* Add an element		 */
   HistoryEFun_t fAdd;        /* Append to an element		 */
};
#define HNEXT(h, ev) (*(h)->fNext)((h)->fRef, ev)
#define HFIRST(h, ev) (*(h)->fFirst)((h)->fRef, ev)
#define HPREV(h, ev) (*(h)->fPrev)((h)->fRef, ev)
#define HLAST(h, ev) (*(h)->fLast)((h)->fRef, ev)
#define HCURR(h, ev) (*(h)->fCurr)((h)->fRef, ev)
#define HSET(h, ev, n) (*(h)->fSet)((h)->fRef, ev, n)
#define HCLEAR(h, ev) (*(h)->fClear)((h)->fRef, ev)
#define HENTER(h, ev, str) (*(h)->fEnter)((h)->fRef, ev, str)
#define HADD(h, ev, str) (*(h)->fAdd)((h)->fRef, ev, str)

#define h_malloc(a) malloc(a)
#define h_realloc(a, b) realloc((a), (b))
#define h_free(a) free(a)


el_private int history_setsize(HistoryFcns_t*, HistEvent_t*, int);
el_private int history_getsize(HistoryFcns_t*, HistEvent_t*);
el_private int history_set_fun(HistoryFcns_t*, HistoryFcns_t*);
el_private int history_load(HistoryFcns_t*, const char*);
el_private int history_save(HistoryFcns_t*, const char*);
el_private int history_prev_event(HistoryFcns_t*, HistEvent_t*, int);
el_private int history_next_event(HistoryFcns_t*, HistEvent_t*, int);
el_private int history_next_string(HistoryFcns_t*, HistEvent_t*, const char*);
el_private int history_prev_string(HistoryFcns_t*, HistEvent_t*, const char*);


/***********************************************************************/

/*
 * Builtin- history implementation
 */
struct HEntry_t {
   HistEvent_t fEv;                /* What we return		 */
   struct HEntry_t* fNext;       /* Next entry			 */
   struct HEntry_t* fPrev;       /* Previous entry		 */
};

struct History_t {
   HEntry_t fList;               /* Fake list header element	 */
   HEntry_t* fCursor;            /* Current element in the list	 */
   int fMax;                     /* Maximum number of events	 */
   int fCur;                     /* Current number of events	 */
   int fEventId;                 /* For generation of unique event id	 */
};

el_private int history_def_first(ptr_t, HistEvent_t*);
el_private int history_def_last(ptr_t, HistEvent_t*);
el_private int history_def_next(ptr_t, HistEvent_t*);
el_private int history_def_prev(ptr_t, HistEvent_t*);
el_private int history_def_curr(ptr_t, HistEvent_t*);
el_private int history_def_set(ptr_t, HistEvent_t *, const int n);
el_private int history_def_enter(ptr_t, HistEvent_t *, const char*);
el_private int history_def_add(ptr_t, HistEvent_t *, const char*);
el_private void history_def_init(ptr_t*, HistEvent_t*, int);
el_private void history_def_clear(ptr_t, HistEvent_t*);
el_private int history_def_insert(History_t*, HistEvent_t*, const char*);
el_private void history_def_delete(History_t*, HistEvent_t*, HEntry_t*);

#define history_def_setsize(p, num) (void) (((History_t*) p)->fMax = (num))
#define history_def_getsize(p) (((History_t*) p)->fCur)

#define he_strerror(code) he_errlist[code]
#define he_seterrev(evp, code) { \
   evp->fNum = code; \
   evp->fStr = he_strerror(code); \
}

/* error messages */
static const char* const he_errlist[] = {
   "OK",
   "unknown error",
   "malloc() failed",
   "first event not found",
   "last event not found",
   "empty list",
   "no next event",
   "no previous event",
   "current event is invalid",
   "event not found",
   "can't read history from file",
   "can't write history",
   "required parameter(s) not supplied",
   "history size negative",
   "function not allowed with other history-functions-set the default",
   "bad parameters"
};
/* error codes */
#define _HE_OK 0
#define _HE_UNKNOWN 1
#define _HE_MALLOC_FAILED 2
#define _HE_FIRST_NOTFOUND 3
#define _HE_LAST_NOTFOUND 4
#define _HE_EMPTY_LIST 5
#define _HE_END_REACHED 6
#define _HE_START_REACHED 7
#define _HE_CURR_INVALID 8
#define _HE_NOT_FOUND 9
#define _HE_HIST_READ 10
#define _HE_HIST_WRITE 11
#define _HE_PARAM_MISSING 12
#define _HE_SIZE_NEGATIVE 13
#define _HE_NOT_ALLOWED 14
#define _HE_BAD_PARAM 15

/* history_def_first():
 *	Default function to return the first event in the history.
 */
el_private int
history_def_first(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   h->fCursor = h->fList.fNext;

   if (h->fCursor != &h->fList) {
      *ev = h->fCursor->fEv;
   } else {
      he_seterrev(ev, _HE_FIRST_NOTFOUND);
      return -1;
   }

   return 0;
}


/* history_def_last():
 *	Default function to return the last event in the history.
 */
el_private int
history_def_last(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   h->fCursor = h->fList.fPrev;

   if (h->fCursor != &h->fList) {
      *ev = h->fCursor->fEv;
   } else {
      he_seterrev(ev, _HE_LAST_NOTFOUND);
      return -1;
   }

   return 0;
}


/* history_def_next():
 *	Default function to return the next event in the history.
 */
el_private int
history_def_next(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   if (h->fCursor != &h->fList) {
      h->fCursor = h->fCursor->fNext;
   } else {
      he_seterrev(ev, _HE_EMPTY_LIST);
      return -1;
   }

   if (h->fCursor != &h->fList) {
      *ev = h->fCursor->fEv;
   } else {
      he_seterrev(ev, _HE_END_REACHED);
      return -1;
   }

   return 0;
} // history_def_next


/* history_def_prev():
 *	Default function to return the previous event in the history.
 */
el_private int
history_def_prev(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   if (h->fCursor != &h->fList) {
      h->fCursor = h->fCursor->fPrev;
   } else {
      he_seterrev(ev,
                  (h->fCur > 0) ? _HE_END_REACHED : _HE_EMPTY_LIST);
      return -1;
   }

   if (h->fCursor != &h->fList) {
      *ev = h->fCursor->fEv;
   } else {
      he_seterrev(ev, _HE_START_REACHED);
      return -1;
   }

   return 0;
} // history_def_prev


/* history_def_curr():
 *	Default function to return the current event in the history.
 */
el_private int
history_def_curr(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   if (h->fCursor != &h->fList) {
      *ev = h->fCursor->fEv;
   } else {
      he_seterrev(ev,
                  (h->fCur > 0) ? _HE_CURR_INVALID : _HE_EMPTY_LIST);
      return -1;
   }

   return 0;
}


/* history_def_set():
 *	Default function to set the current event in the history to the
 *	given one.
 */
el_private int
history_def_set(ptr_t p, HistEvent_t* ev, const int n) {
   History_t* h = (History_t*) p;

   if (h->fCur == 0) {
      he_seterrev(ev, _HE_EMPTY_LIST);
      return -1;
   }

   if (h->fCursor == &h->fList || h->fCursor->fEv.fNum != n) {
      for (h->fCursor = h->fList.fNext; h->fCursor != &h->fList;
           h->fCursor = h->fCursor->fNext) {
         if (h->fCursor->fEv.fNum == n) {
            break;
         }
      }
   }

   if (h->fCursor == &h->fList) {
      he_seterrev(ev, _HE_NOT_FOUND);
      return -1;
   }
   return 0;
} // history_def_set


/* history_def_add():
 *	Append string to element
 */
el_private int
history_def_add(ptr_t p, HistEvent_t* ev, const char* str) {
   History_t* h = (History_t*) p;
   size_t len;
   char* s;

   if (h->fCursor == &h->fList) {
      return history_def_enter(p, ev, str);
   }
   len = strlen(h->fCursor->fEv.fStr) + strlen(str) + 1;
   s = (char*) h_malloc(len);

   if (!s) {
      he_seterrev(ev, _HE_MALLOC_FAILED);
      return -1;
   }
   (void) strlcpy(s, h->fCursor->fEv.fStr, len);
   (void) strlcat(s, str, len);
   /* LINTED const cast */
   h_free((ptr_t) h->fCursor->fEv.fStr);
   h->fCursor->fEv.fStr = s;
   *ev = h->fCursor->fEv;
   return 0;
} // history_def_add


/* history_def_delete():
 *	Delete element hp of the h list
 */
/* ARGSUSED */
el_private void
history_def_delete(History_t* h, HistEvent_t* /*ev*/, HEntry_t* hp) {
   if (hp == &h->fList) {
      abort();
   }
   hp->fPrev->fNext = hp->fNext;
   hp->fNext->fPrev = hp->fPrev;
   /* LINTED const cast */
   h_free((ptr_t) hp->fEv.fStr);
   h_free(hp);
   h->fCur--;
}


/* history_def_insert():
 *	Insert element with string str in the h list
 */
el_private int
history_def_insert(History_t* h, HistEvent_t* ev, const char* str) {
   h->fCursor = (HEntry_t*) h_malloc(sizeof(HEntry_t));

   if (h->fCursor) {
      h->fCursor->fEv.fStr = strdup(str);
   }

   if (!h->fCursor || !h->fCursor->fEv.fStr) {
      he_seterrev(ev, _HE_MALLOC_FAILED);
      return -1;
   }
   h->fCursor->fEv.fNum = ++h->fEventId;
   h->fCursor->fNext = h->fList.fNext;
   h->fCursor->fPrev = &h->fList;
   h->fList.fNext->fPrev = h->fCursor;
   h->fList.fNext = h->fCursor;
   h->fCur++;

   *ev = h->fCursor->fEv;
   return 0;
} // history_def_insert


/* history_def_enter():
 *	Default function to enter an item in the history
 */
el_private int
history_def_enter(ptr_t p, HistEvent_t* ev, const char* str) {
   History_t* h = (History_t*) p;

   if (history_def_insert(h, ev, str) == -1) {
      return -1;                /* error, keep error message */

   }

   /*
    * Always keep at least one entry.
    * This way we don't have to check for the empty list.
    */
   while (h->fCur - 1 > h->fMax)
      history_def_delete(h, ev, h->fList.fPrev);

   return 0;
}


/* history_def_init():
 *	Default history initialization function
 */
/* ARGSUSED */
el_private void
history_def_init(ptr_t* p, HistEvent_t* /*ev*/, int n) {
   History_t* h = (History_t*) h_malloc(sizeof(History_t));

   if (n <= 0) {
      n = 0;
   }
   h->fEventId = 0;
   h->fCur = 0;
   h->fMax = n;
   h->fList.fNext = h->fList.fPrev = &h->fList;
   h->fList.fEv.fStr = NULL;
   h->fList.fEv.fNum = 0;
   h->fCursor = &h->fList;
   *p = (ptr_t) h;
}


/* history_def_clear():
 *	Default history cleanup function
 */
el_private void
history_def_clear(ptr_t p, HistEvent_t* ev) {
   History_t* h = (History_t*) p;

   while (h->fList.fPrev != &h->fList)
      history_def_delete(h, ev, h->fList.fPrev);
   h->fEventId = 0;
   h->fCur = 0;
}


/************************************************************************/

/* history_init():
 *	Initialization function.
 */
el_public HistoryFcns_t*
history_init(void) {
   HistoryFcns_t* h = (HistoryFcns_t*) h_malloc(sizeof(HistoryFcns_t));
   HistEvent_t ev;

   history_def_init(&h->fRef, &ev, 0);
   h->fEnt = -1;
   h->fNext = history_def_next;
   h->fFirst = history_def_first;
   h->fLast = history_def_last;
   h->fPrev = history_def_prev;
   h->fCurr = history_def_curr;
   h->fSet = history_def_set;
   h->fClear = history_def_clear;
   h->fEnter = history_def_enter;
   h->fAdd = history_def_add;

   return h;
} // history_init


/* history_end():
 *	clean up history;
 */
el_public void
history_end(HistoryFcns_t* h) {
   HistEvent_t ev;

   if (h->fNext == history_def_next) {
      history_def_clear(h->fRef, &ev);
   }
}


/* history_setsize():
 *	Set history number of events
 */
el_private int
history_setsize(HistoryFcns_t* h, HistEvent_t* ev, int num) {
   if (h->fNext != history_def_next) {
      he_seterrev(ev, _HE_NOT_ALLOWED);
      return -1;
   }

   if (num < 0) {
      he_seterrev(ev, _HE_BAD_PARAM);
      return -1;
   }
   history_def_setsize(h->fRef, num);
   return 0;
}


/* history_getsize():
 *      Get number of events currently in history
 */
el_private int
history_getsize(HistoryFcns_t* h, HistEvent_t* ev) {
   int retval = 0;

   if (h->fNext != history_def_next) {
      he_seterrev(ev, _HE_NOT_ALLOWED);
      return -1;
   }
   retval = history_def_getsize(h->fRef);

   if (retval < -1) {
      he_seterrev(ev, _HE_SIZE_NEGATIVE);
      return -1;
   }
   ev->fNum = retval;
   return 0;
}


/* history_set_fun():
 *	Set history functions
 */
el_private int
history_set_fun(HistoryFcns_t* h, HistoryFcns_t* nh) {
   HistEvent_t ev;

   if (nh->fFirst == NULL || nh->fNext == NULL || nh->fLast == NULL ||
       nh->fPrev == NULL || nh->fCurr == NULL || nh->fSet == NULL ||
       nh->fEnter == NULL || nh->fAdd == NULL || nh->fClear == NULL ||
       nh->fRef == NULL) {
      if (h->fNext != history_def_next) {
         history_def_init(&h->fRef, &ev, 0);
         h->fFirst = history_def_first;
         h->fNext = history_def_next;
         h->fLast = history_def_last;
         h->fPrev = history_def_prev;
         h->fCurr = history_def_curr;
         h->fSet = history_def_set;
         h->fClear = history_def_clear;
         h->fEnter = history_def_enter;
         h->fAdd = history_def_add;
      }
      return -1;
   }

   if (h->fNext == history_def_next) {
      history_def_clear(h->fRef, &ev);
   }

   h->fEnt = -1;
   h->fFirst = nh->fFirst;
   h->fNext = nh->fNext;
   h->fLast = nh->fLast;
   h->fPrev = nh->fPrev;
   h->fCurr = nh->fCurr;
   h->fSet = nh->fSet;
   h->fClear = nh->fClear;
   h->fEnter = nh->fEnter;
   h->fAdd = nh->fAdd;

   return 0;
} // history_set_fun


/* history_load():
 *	History_t load function
 */
el_private int
history_load(HistoryFcns_t* h, const char* fname) {
   HistEvent_t ev;

   std::ifstream in(fname);
   if (!in) {
      return -1;
   }

   std::string line;
   int i = 0;
   for (; in && std::getline(in, line); i++) {
      HENTER(h, &ev, line.c_str());
   }
   return i;
} // history_load


/* history_save():
 *	HistoryFcns_t save function
 *  mod for ROOT: only append the most recent line!
 */
el_private int
history_save(HistoryFcns_t* h, const char* fname) {
   FILE* fp;
   HistEvent_t ev;
   int i = 0;

   if ((fp = fopen(fname, "a")) == NULL) {
      return -1;
   }

   (void) fchmod(fileno(fp), S_IRUSR | S_IWUSR);

   /*
   for (retval = HLAST(h, &ev);
        retval != -1;
        retval = HPREV(h, &ev), i++) {
      (void) fprintf(fp, "%s\n", ev.fStr);
   }
   */
   HFIRST(h, &ev);
   fprintf(fp, "%s\n", ev.fStr);

   (void) fclose(fp);
   return i;
} // history_save


/* history_prev_event():
 *	Find the previous event, with number given
 */
el_private int
history_prev_event(HistoryFcns_t* h, HistEvent_t* ev, int num) {
   int retval;

   for (retval = HCURR(h, ev); retval != -1; retval = HPREV(h, ev)) {
      if (ev->fNum == num) {
         return 0;
      }
   }

   he_seterrev(ev, _HE_NOT_FOUND);
   return -1;
}


/* history_next_event():
 *	Find the next event, with number given
 */
el_private int
history_next_event(HistoryFcns_t* h, HistEvent_t* ev, int num) {
   int retval;

   for (retval = HCURR(h, ev); retval != -1; retval = HNEXT(h, ev)) {
      if (ev->fNum == num) {
         return 0;
      }
   }

   he_seterrev(ev, _HE_NOT_FOUND);
   return -1;
}


/* history_prev_string():
 *	Find the previous event beginning with string
 */
el_private int
history_prev_string(HistoryFcns_t* h, HistEvent_t* ev, const char* str) {
   size_t len = strlen(str);
   int retval;

   for (retval = HCURR(h, ev); retval != -1; retval = HNEXT(h, ev)) {
      if (strncmp(str, ev->fStr, len) == 0) {
         return 0;
      }
   }

   he_seterrev(ev, _HE_NOT_FOUND);
   return -1;
}


/* history_next_string():
 *	Find the next event beginning with string
 */
el_private int
history_next_string(HistoryFcns_t* h, HistEvent_t* ev, const char* str) {
   size_t len = strlen(str);
   int retval;

   for (retval = HCURR(h, ev); retval != -1; retval = HPREV(h, ev)) {
      if (strncmp(str, ev->fStr, len) == 0) {
         return 0;
      }
   }

   he_seterrev(ev, _HE_NOT_FOUND);
   return -1;
}


/* history():
 *	User interface to history functions.
 */
int
history(HistoryFcns_t* h, HistEvent_t* ev, int fun, ...) {
   va_list va;
   const char* str;
   int retval;

   va_start(va, fun);

   he_seterrev(ev, _HE_OK);

   switch (fun) {
   case H_GETSIZE:
      retval = history_getsize(h, ev);
      break;

   case H_SETSIZE:
      retval = history_setsize(h, ev, va_arg(va, int));
      break;

   case H_ADD:
      str = va_arg(va, const char*);
      retval = HADD(h, ev, str);
      break;

   case H_ENTER:
      str = va_arg(va, const char*);

      if ((retval = HENTER(h, ev, str)) != -1) {
         h->fEnt = ev->fNum;
      }
      break;

   case H_APPEND:
      str = va_arg(va, const char*);

      if ((retval = HSET(h, ev, h->fEnt)) != -1) {
         retval = HADD(h, ev, str);
      }
      break;

   case H_FIRST:
      retval = HFIRST(h, ev);
      break;

   case H_NEXT:
      retval = HNEXT(h, ev);
      break;

   case H_LAST:
      retval = HLAST(h, ev);
      break;

   case H_PREV:
      retval = HPREV(h, ev);
      break;

   case H_CURR:
      retval = HCURR(h, ev);
      break;

   case H_SET:
      retval = HSET(h, ev, va_arg(va, int));
      break;

   case H_CLEAR:
      HCLEAR(h, ev);
      retval = 0;
      break;

   case H_LOAD:
      retval = history_load(h, va_arg(va, const char*));

      if (retval == -1) {
         he_seterrev(ev, _HE_HIST_READ);
      }
      break;

   case H_SAVE:
      retval = history_save(h, va_arg(va, const char*));

      if (retval == -1) {
         he_seterrev(ev, _HE_HIST_WRITE);
      }
      break;

   case H_PREV_EVENT:
      retval = history_prev_event(h, ev, va_arg(va, int));
      break;

   case H_NEXT_EVENT:
      retval = history_next_event(h, ev, va_arg(va, int));
      break;

   case H_PREV_STR:
      retval = history_prev_string(h, ev, va_arg(va, const char*));
      break;

   case H_NEXT_STR:
      retval = history_next_string(h, ev, va_arg(va, const char*));
      break;

   case H_FUNC:
      {
         HistoryFcns_t hf;

         hf.fRef = va_arg(va, ptr_t);
         h->fEnt = -1;
         hf.fFirst = va_arg(va, HistoryGFun_t);
         hf.fNext = va_arg(va, HistoryGFun_t);
         hf.fLast = va_arg(va, HistoryGFun_t);
         hf.fPrev = va_arg(va, HistoryGFun_t);
         hf.fCurr = va_arg(va, HistoryGFun_t);
         hf.fSet = va_arg(va, HistorySFun_t);
         hf.fClear = va_arg(va, HistoryVFun_t);
         hf.fEnter = va_arg(va, HistoryEFun_t);
         hf.fAdd = va_arg(va, HistoryEFun_t);

         if ((retval = history_set_fun(h, &hf)) == -1) {
            he_seterrev(ev, _HE_PARAM_MISSING);
         }
         break;
      }

   case H_END:
      history_end(h);
      retval = 0;
      break;

   default:
      retval = -1;
      he_seterrev(ev, _HE_UNKNOWN);
      break;
   } // switch
   va_end(va);
   return retval;
} // history
