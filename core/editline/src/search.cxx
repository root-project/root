// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: search.c,v 1.11 2001/01/23 15:55:31 jdolecek Exp $	*/

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
 * search.c: History_t and character search functions
 */
#include "sys.h"
#include <stdlib.h>
#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#if defined(REGEX)
# include <regex.h>
#elif defined(REGEXP)
# include <regexp.h>
#endif
#include "el.h"
#include "enhance.h"

#define ANCHOR_SEARCHES 1
// ^^^ don't undef ANCHOR_SEARCHES or set it to 0! It breaks ^R searches! (stephan)

/*
 * Adjust cursor in vi mode to include the character under it
 */
#define EL_CURSOR(el) \
   ((el)->fLine.fCursor + (((el)->fMap.fType == MAP_VI) && \
                            ((el)->fMap.fCurrent == (el)->fMap.fAlt)))

/* search_init():
 *	Initialize the search stuff
 */
el_protected int
search_init(EditLine_t* el) {
   el->fSearch.fPatBuf = (char*) el_malloc(EL_BUFSIZ);

   if (el->fSearch.fPatBuf == NULL) {
      return -1;
   }
   el->fSearch.fPatLen = 0;
   el->fSearch.fPatDir = -1;
   el->fSearch.fChaCha = '\0';
   el->fSearch.fChaDir = -1;
   return 0;
}


/* search_end():
 *	Initialize the search stuff
 */
el_protected void
search_end(EditLine_t* el) {
   el_free((ptr_t) el->fSearch.fPatBuf);
   el->fSearch.fPatBuf = NULL;
}


#ifdef REGEXP

/* regerror():
 *	Handle regular expression errors
 */
el_public void
/*ARGSUSED*/
regerror(const char* msg) {
}


#endif


/* el_match():
 *	Return if string matches pattern
 */
el_protected int
el_match(const char* str, const char* pat) {
#if defined(REGEX)
   regex_t re;
   int rv;
#elif defined(REGEXP)
   regexp* rp;
   int rv;
#else
   extern char* re_comp(const char*);
   extern int re_exec(const char*);
#endif

   if (strstr(str, pat) != NULL) {
      return 1;
   }

#if defined(REGEX)

   if (regcomp(&re, pat, 0) == 0) {
      rv = regexec(&re, str, 0, NULL, 0) == 0;
      regfree(&re);
   } else {
      rv = 0;
   }
   return rv;
#elif defined(REGEXP)

   if ((re = regcomp(pat)) != NULL) {
      rv = regexec(re, str);
      free((ptr_t) re);
   } else {
      rv = 0;
   }
   return rv;
#else

   if (re_comp(pat) != NULL) {
      return 0;
   } else {
      return re_exec(str) == 1;
   }
#endif
} // el_match


/* c_hmatch():
 *	 return True if the pattern matches the prefix
 */
el_protected int
c_hmatch(EditLine_t* el, const char* str) {
#ifdef SDEBUG
      (void) fprintf(el->fErrFile, "match `%s' with `%s'\n",
                     el->fSearch.fPatBuf, str);
#endif /* SDEBUG */

   return el_match(str, el->fSearch.fPatBuf);
}


/* c_setpat():
 *	Set the history seatch pattern
 */
el_protected void
c_setpat(EditLine_t* el) {
   if (el->fState.fLastCmd != ED_SEARCH_PREV_HISTORY &&
       el->fState.fLastCmd != ED_SEARCH_NEXT_HISTORY) {
      el->fSearch.fPatLen = EL_CURSOR(el) - el->fLine.fBuffer;

      if (el->fSearch.fPatLen >= EL_BUFSIZ) {
         el->fSearch.fPatLen = EL_BUFSIZ - 1;
      }

      if (el->fSearch.fPatLen != 0) {
         (void) strncpy(el->fSearch.fPatBuf, el->fLine.fBuffer,
                        el->fSearch.fPatLen);
         el->fSearch.fPatBuf[el->fSearch.fPatLen] = '\0';
      } else {
         el->fSearch.fPatLen = strlen(el->fSearch.fPatBuf);
      }
   }
#ifdef SDEBUG
      (void) fprintf(el->fErrFile, "\neventno = %d\n",
                     el->fHistory.fEventNo);
   (void) fprintf(el->fErrFile, "patlen = %d\n", el->fSearch.fPatLen);
   (void) fprintf(el->fErrFile, "patbuf = \"%s\"\n",
                  el->fSearch.fPatBuf);
   (void) fprintf(el->fErrFile, "cursor %d lastchar %d\n",
                  EL_CURSOR(el) - el->fLine.fBuffer,
                  el->fLine.fLastChar - el->fLine.fBuffer);
#endif
} // c_setpat


/* ce_inc_search():
 *	Emacs incremental search
 */
el_protected ElAction_t
ce_inc_search(EditLine_t* el, int dir) {
   static const char strfwd[] = { 'f', 'w', 'd', '\0' },
                     strbck[] = { 'b', 'c', 'k', '\0' };
   static char pchar = ':';     /* ':' = normal, '?' = failed */
   static char endcmd[2] = { '\0', '\0' };
   char ch, * ocursor = el->fLine.fCursor, oldpchar = pchar;
   const char* cp;

   ElAction_t ret = CC_NORM;

   int ohisteventno = el->fHistory.fEventNo;
   int oldpatlen = el->fSearch.fPatLen;
   int newdir = dir;
   int done, redo;

   if (el->fLine.fLastChar + sizeof(strfwd) / sizeof(char) + 2 +
       el->fSearch.fPatLen >= el->fLine.fLimit) {
      return CC_ERROR;
   }

   for ( ; ;) {
      if (el->fSearch.fPatLen == 0) {                  /* first round */
         pchar = ':';
#if ANCHOR_SEARCHES
         el->fSearch.fPatBuf[el->fSearch.fPatLen++] = '.';
         el->fSearch.fPatBuf[el->fSearch.fPatLen++] = '*';
#endif
      }
      done = redo = 0;
      el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
      *el->fLine.fLastChar++ = '\n';

      for (cp = (newdir == ED_SEARCH_PREV_HISTORY) ? strbck : strfwd;
           *cp; *el->fLine.fLastChar++ = *cp++) {
         el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
      }
      el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
      *el->fLine.fLastChar++ = pchar;
#if ANCHOR_SEARCHES
      cp = &el->fSearch.fPatBuf[2];
#else
      cp = &el->fSearch.fPatBuf[1];
#endif

      for ( ;
            cp < &el->fSearch.fPatBuf[el->fSearch.fPatLen];
            *el->fLine.fLastChar++ = *cp++) {
         el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
         continue;
      }
      el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
      *el->fLine.fLastChar = '\0';

      // Would love to highlight, but el_dispcolor isn't set up yet...
      //highlightKeywords(el);
      re_refresh(el);

      if (el_getc(el, &ch) != 1) {
         return ed_end_of_file(el, 0);
      }

      // Coverity is complaining that the value of ch comes from the user
      // and nowhere do we check its value. But that's fine: it's 0<=ch<255,
      // and fCurrent has 256 entries.
      // coverity[tainted_data]
      switch (el->fMap.fCurrent[(unsigned char) ch]) {
      case ED_INSERT:
      case ED_DIGIT:

         if (el->fSearch.fPatLen > EL_BUFSIZ - 3) {
            term_beep(el);
         } else {
            el->fSearch.fPatBuf[el->fSearch.fPatLen++] =
               ch;
            el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
            *el->fLine.fLastChar++ = ch;
            el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
            *el->fLine.fLastChar = '\0';
            re_refresh(el);
         }
         break;

      case EM_INC_SEARCH_NEXT:
         newdir = ED_SEARCH_NEXT_HISTORY;
         redo++;
         break;

      case EM_INC_SEARCH_PREV:
         newdir = ED_SEARCH_PREV_HISTORY;
         redo++;
         break;

      case ED_DELETE_PREV_CHAR:

         if (el->fSearch.fPatLen > 1) {
            done++;
         } else {
            term_beep(el);
         }
         break;

      default:

         switch (ch) {
         case 0007:                     /* ^G: Abort */
            ret = CC_ERROR;
            done++;
            break;

         case 0027:                     /* ^W: Append word */

            /* No can do if globbing characters in pattern */
            for (cp = &el->fSearch.fPatBuf[1]; ; cp++) {
               if (cp >= &el->fSearch.fPatBuf[el->fSearch.fPatLen]) {
                  el->fLine.fCursor +=
                     el->fSearch.fPatLen - 1;
                  cp = c__next_word(el->fLine.fCursor,
                                    el->fLine.fLastChar, 1,
                                    ce__isword);

                  while (el->fLine.fCursor < cp &&
                         *el->fLine.fCursor != '\n') {
                     if (el->fSearch.fPatLen >
                         EL_BUFSIZ - 3) {
                        term_beep(el);
                        break;
                     }
                     el->fSearch.fPatBuf[el->fSearch.fPatLen++] =
                        *el->fLine.fCursor;
                     el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] =
                        el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer];
                     *el->fLine.fLastChar++ =
                        *el->fLine.fCursor++;
                  }
                  el->fLine.fCursor = ocursor;
                  el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
                  *el->fLine.fLastChar = '\0';
                  re_refresh(el);
                  break;
               } else if (isglob(*cp)) {
                  term_beep(el);
                  break;
               }
            }
            break;

         default:                       /* Terminate and execute cmd */
            endcmd[0] = ch;
            el_push(el, endcmd);
            ret = CC_REFRESH;
            done++;
            break;
         } // switch
         break;
      } // switch

      while (el->fLine.fLastChar > el->fLine.fBuffer &&
             *el->fLine.fLastChar != '\n') {
         el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
         *el->fLine.fLastChar-- = '\0';
      }
      el->fLine.fBufColor[el->fLine.fLastChar - el->fLine.fBuffer] = -1;
      *el->fLine.fLastChar = '\0';

      if (!done) {
         /* Can't search if unmatched '[' */
         ch = ']';
         for (cp = &el->fSearch.fPatBuf[el->fSearch.fPatLen - 1];
              cp > el->fSearch.fPatBuf; cp--) {
            if (*cp == '[' || *cp == ']') {
               ch = *cp;
               break;
            }
         }

         if (el->fSearch.fPatLen > 1 && ch != '[') {
            if (redo && newdir == dir) {
               if (pchar == '?') {                          /* wrap around */
                  el->fHistory.fEventNo =
                     newdir == ED_SEARCH_PREV_HISTORY ? 0 : 0x7fffffff;

                  if (hist_get(el) == CC_ERROR) {
                     /* el->fHistory.event
                      * no was fixed by
                      * first call */
                     (void) hist_get(el);
                  }
                  el->fLine.fCursor = newdir ==
                                       ED_SEARCH_PREV_HISTORY ?
                                       el->fLine.fLastChar :
                                       el->fLine.fBuffer;
               } else {
                  el->fLine.fCursor +=
                     newdir ==
                     ED_SEARCH_PREV_HISTORY ?
                     -1 : 1;
               }
            }
#if ANCHOR_SEARCHES
            el->fSearch.fPatBuf[el->fSearch.fPatLen++] =
               '.';
            el->fSearch.fPatBuf[el->fSearch.fPatLen++] =
               '*';
#endif
            el->fSearch.fPatBuf[el->fSearch.fPatLen] =
               '\0';

            if (el->fLine.fCursor < el->fLine.fBuffer ||
                el->fLine.fCursor > el->fLine.fLastChar ||
                (ret = ce_search_line(el, &el->fSearch.fPatBuf[1], newdir)) == CC_ERROR) {
               /* avoid c_setpat */
               el->fState.fLastCmd =
                  (ElAction_t) newdir;
               ret = newdir == ED_SEARCH_PREV_HISTORY ?
                     ed_search_prev_history(el, 0) :
                     ed_search_next_history(el, 0);

               if (ret != CC_ERROR) {
                  el->fLine.fCursor = newdir ==
                                       ED_SEARCH_PREV_HISTORY ?
                                       el->fLine.fLastChar :
                                       el->fLine.fBuffer;
                  (void) ce_search_line(el,
                                        &el->fSearch.fPatBuf[1],
                                        newdir);
               }
            }
#if ANCHOR_SEARCHES
            el->fSearch.fPatLen -= 2;
            el->fSearch.fPatBuf[el->fSearch.fPatLen] = 0;
#else
            el->fSearch.fPatBuf[--el->fSearch.fPatLen] = 0;
#endif

            if (ret == CC_ERROR) {
               term_beep(el);

               if (el->fHistory.fEventNo !=
                   ohisteventno) {
                  el->fHistory.fEventNo =
                     ohisteventno;

                  if (hist_get(el) == CC_ERROR) {
                     return CC_ERROR;
                  }
               }
               el->fLine.fCursor = ocursor;
               pchar = '?';
            } else {
               pchar = ':';
            }
         }
         ret = ce_inc_search(el, newdir);

         if (ret == CC_ERROR && pchar == '?' && oldpchar == ':') {
            /*
             * break abort of failed search at last
             * non-failed
             */
            ret = CC_NORM;
         }

      }

      if (ret == CC_NORM || (ret == CC_ERROR && oldpatlen == 0)) {
         /* restore on normal return or error exit */
         pchar = oldpchar;
         el->fSearch.fPatLen = oldpatlen;

         if (el->fHistory.fEventNo != ohisteventno) {
            el->fHistory.fEventNo = ohisteventno;

            if (hist_get(el) == CC_ERROR) {
               return CC_ERROR;
            }
         }
         el->fLine.fCursor = ocursor;

         if (ret == CC_ERROR) {
            re_refresh(el);
         }
      }

      if (done || ret != CC_NORM) {
         return ret;
      }
   }
} // ce_inc_search


/* cv_search():
 *	Vi search.
 */
el_protected ElAction_t
cv_search(EditLine_t* el, int dir) {
   char ch;
   char tmpbuf[EL_BUFSIZ];
   int tmplen;

   tmplen = 0;
#if ANCHOR_SEARCHES
   tmpbuf[tmplen++] = '.';
   tmpbuf[tmplen++] = '*';
#endif

   el->fLine.fBuffer[0] = '\0';
   el->fLine.fLastChar = el->fLine.fBuffer;
   el->fLine.fCursor = el->fLine.fBuffer;
   el->fSearch.fPatDir = dir;

   c_insert(el, 2);             /* prompt + '\n' */
   *el->fLine.fCursor++ = '\n';
   *el->fLine.fCursor++ = dir == ED_SEARCH_PREV_HISTORY ? '/' : '?';
   re_refresh(el);

#if ANCHOR_SEARCHES
# define LEN 2
#else
# define LEN 0
#endif

   tmplen = c_gets(el, &tmpbuf[LEN]) + LEN;
   ch = tmpbuf[tmplen];
   tmpbuf[tmplen] = '\0';

   if (tmplen == LEN) {
      /*
       * Use the old pattern, but wild-card it.
       */
      if (el->fSearch.fPatLen == 0) {
         el->fLine.fBuffer[0] = '\0';
         el->fLine.fLastChar = el->fLine.fBuffer;
         el->fLine.fCursor = el->fLine.fBuffer;
         re_refresh(el);
         return CC_ERROR;
      }
#if ANCHOR_SEARCHES

      if (el->fSearch.fPatBuf[0] != '.' &&
          el->fSearch.fPatBuf[0] != '*') {
         (void) strncpy(tmpbuf, el->fSearch.fPatBuf,
                        sizeof(tmpbuf) - 1);
         el->fSearch.fPatBuf[0] = '.';
         el->fSearch.fPatBuf[1] = '*';
         (void) strncpy(&el->fSearch.fPatBuf[2], tmpbuf,
                        EL_BUFSIZ - 3);
         el->fSearch.fPatLen++;
         el->fSearch.fPatBuf[el->fSearch.fPatLen++] = '.';
         el->fSearch.fPatBuf[el->fSearch.fPatLen++] = '*';
         el->fSearch.fPatBuf[el->fSearch.fPatLen] = '\0';
      }
#endif
   } else {
#if ANCHOR_SEARCHES
      tmpbuf[tmplen++] = '.';
      tmpbuf[tmplen++] = '*';
#endif
      tmpbuf[tmplen] = '\0';
      (void) strncpy(el->fSearch.fPatBuf, tmpbuf, EL_BUFSIZ - 1);
      el->fSearch.fPatLen = tmplen;
   }
   el->fState.fLastCmd = (ElAction_t) dir;            /* avoid c_setpat */
   el->fLine.fCursor = el->fLine.fLastChar = el->fLine.fBuffer;

   if ((dir == ED_SEARCH_PREV_HISTORY ? ed_search_prev_history(el, 0) :
        ed_search_next_history(el, 0)) == CC_ERROR) {
      re_refresh(el);
      return CC_ERROR;
   } else {
      if (ch == 0033) {
         re_refresh(el);
         *el->fLine.fLastChar++ = '\n';
         *el->fLine.fLastChar = '\0';
         re_goto_bottom(el);
         return CC_NEWLINE;
      } else {
         return CC_REFRESH;
      }
   }
} // cv_search


/* ce_search_line():
 *	Look for a pattern inside a line
 */
el_protected ElAction_t
ce_search_line(EditLine_t* el, char* pattern, int dir) {
   char* cp;

   if (dir == ED_SEARCH_PREV_HISTORY) {
      for (cp = el->fLine.fCursor; cp >= el->fLine.fBuffer; cp--) {
         if (el_match(cp, pattern)) {
            el->fLine.fCursor = cp;
            return CC_NORM;
         }
      }
      return CC_ERROR;
   } else {
      for (cp = el->fLine.fCursor; *cp != '\0' &&
           cp < el->fLine.fLimit; cp++) {
         if (el_match(cp, pattern)) {
            el->fLine.fCursor = cp;
            return CC_NORM;
         }
      }
      return CC_ERROR;
   }
} // ce_search_line


/* cv_repeat_srch():
 *	Vi repeat search
 */
el_protected ElAction_t
cv_repeat_srch(EditLine_t* el, int c) {
#ifdef SDEBUG
      (void) fprintf(el->fErrFile, "dir %d patlen %d patbuf %s\n",
                     c, el->fSearch.fPatLen, el->fSearch.fPatBuf);
#endif

   el->fState.fLastCmd = (ElAction_t) c;      /* Hack to stop c_setpat */
   el->fLine.fLastChar = el->fLine.fBuffer;

   switch (c) {
   case ED_SEARCH_NEXT_HISTORY:
      return ed_search_next_history(el, 0);
   case ED_SEARCH_PREV_HISTORY:
      return ed_search_prev_history(el, 0);
   default:
      return CC_ERROR;
   }
} // cv_repeat_srch


/* cv_csearch_back():
 *	Vi character search reverse
 */
el_protected ElAction_t
cv_csearch_back(EditLine_t* el, int ch, int count, int tflag) {
   char* cp;

   cp = el->fLine.fCursor;

   while (count--) {
      if (*cp == ch) {
         cp--;
      }

      while (cp > el->fLine.fBuffer && *cp != ch)
         cp--;
   }

   if (cp < el->fLine.fBuffer || (cp == el->fLine.fBuffer && *cp != ch)) {
      return CC_ERROR;
   }

   if (*cp == ch && tflag) {
      cp++;
   }

   el->fLine.fCursor = cp;

   if (el->fCharEd.fVCmd.fAction & DELETE) {
      el->fLine.fCursor++;
      cv_delfini(el);
      return CC_REFRESH;
   }
   re_refresh_cursor(el);
   return CC_NORM;
} // cv_csearch_back


/* cv_csearch_fwd():
 *	Vi character search forward
 */
el_protected ElAction_t
cv_csearch_fwd(EditLine_t* el, int ch, int count, int tflag) {
   char* cp;

   cp = el->fLine.fCursor;

   while (count--) {
      if (*cp == ch) {
         cp++;
      }

      while (cp < el->fLine.fLastChar && *cp != ch)
         cp++;
   }

   if (cp >= el->fLine.fLastChar) {
      return CC_ERROR;
   }

   if (*cp == ch && tflag) {
      cp--;
   }

   el->fLine.fCursor = cp;

   if (el->fCharEd.fVCmd.fAction & DELETE) {
      el->fLine.fCursor++;
      cv_delfini(el);
      return CC_REFRESH;
   }
   re_refresh_cursor(el);
   return CC_NORM;
} // cv_csearch_fwd
