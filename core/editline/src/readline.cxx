// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: readline.c,v 1.19 2001/01/10 08:10:45 jdolecek Exp $	*/

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


#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <pwd.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#ifndef __FreeBSD__
#include <alloca.h>
#endif // __FreeBSD__
#include "histedit.h"
// #include "readline/readline.h"
#include "editline.h"
#include "el.h"
#include "compat.h"
#include "TTermManip.h"
#include "enhance.h"

/* for rl_complete() */
#define TAB '\r'

/* see comment at the #ifdef for sense of this */
#define GDB_411_HACK

/* readline compatibility stuff - look at readline sources/documentation */
/* to see what these variables mean */
const char* rl_library_version = "EditLine_t wrapper";
const char* rl_readline_name = "";
FILE* rl_instream = NULL;
FILE* rl_outstream = NULL;
int rl_point = 0;
int rl_end = 0;
char* rl_line_buffer = NULL;

int history_base = 1;           /* probably never subject to change */
int history_length = 0;
int max_input_history = 0;
char history_expansion_char = '!';
char history_subst_char = '^';
const char* history_no_expand_chars = " \t\n=(";
Function* history_inhibit_expansion_function = NULL;

int rl_inhibit_completion = 0;
int rl_attempted_completion_over = 0;
const char* rl_basic_word_break_characters = " \t\n\"\\'`@$><=;|&{(";
char* rl_completer_word_break_characters = NULL;
char* rl_completer_quote_characters = NULL;
CPFunction* rl_completion_entry_function = NULL;
CPPFunction* rl_attempted_completion_function = NULL;
int tab_color = 5;
El_tab_hook_t rl_tab_hook = NULL;
El_in_key_hook_t rl_in_key_hook = NULL;

/*
 * This is set to character indicating type of completion being done by
 * rl_complete_internal(); this is available for application completion
 * functions.
 */
int rl_completion_type = 0;

/*
 * If more than this number of items results from query for possible
 * completions, we ask user if they are sure to really display the list.
 */
int rl_completion_query_items = 100;

/*
 * List of characters which are word break characters, but should be left
 * in the parsed text when it is passed to the completion function.
 * Shell uses this to help determine what kind of completing to do.
 */
char* rl_special_prefixes = (char*) NULL;

/*
 * This is the character appended to the completed words if at the end of
 * the line. Default is ' ' (a space).
 */
int rl_completion_append_character = ' ';

/* stuff below is used internally by libedit for readline emulation */

/* if not zero, non-unique completions always show list of possible matches */
static int grl_complete_show_all = 0;

static HistoryFcns_t* gHistory = NULL;
static EditLine_t* gEditLine = NULL;
static int gel_rl_complete_cmdnum = 0;

/* internal functions */
static unsigned char _el_rl_complete(EditLine_t*, int);
static char* _get_prompt(EditLine_t*);
static HIST_ENTRY* _move_history(int);
static int _history_search_gen(const char*, int, int);
#ifdef EL_HISTORY_EXPAND
static int _history_expand_command(const char*, size_t, char**);
static char* _rl_compat_sub(const char*, const char*,
                            const char*, int);
#endif
static int rl_complete_internal(int);
static int _rl_qsort_string_compare(const void*, const void*);

/*
 * needed for prompt switching in readline()
 */
static char* gel_rl_prompt = NULL;

/* ARGSUSED */
static char*
_get_prompt(EditLine_t* /*el*/) {
   return gel_rl_prompt;
}


/*
 * generic function for moving around history
 */
static HIST_ENTRY*
_move_history(int op) {
   HistEvent_t ev;
   static HIST_ENTRY rl_he;

   if (history(gHistory, &ev, op) != 0) {
      return (HIST_ENTRY*) NULL;
   }

   rl_he.line = ev.fStr;
   rl_he.data = "";

   return &rl_he;
}


bool
rl_isinitialized() {
   return gEditLine != NULL;
}

/*
 * READLINE compatibility stuff
 */

/*
 * initialize rl compat stuff
 */
int
rl_initialize(void) {
   HistEvent_t ev;
   const LineInfo_t* li;
   int i;
   int editmode = 1;
   struct termios t;

   if (gEditLine != NULL) {
      el_end(gEditLine);
   }

   if (gHistory != NULL) {
      history_end(gHistory);
   }

   if (!rl_instream) {
      rl_instream = stdin;
   }

   if (!rl_outstream) {
      rl_outstream = stdout;
   }

   /*
    * See if we don't really want to run the editor
    */
   if (tcgetattr(fileno(rl_instream), &t) != -1 && (t.c_lflag & ECHO) == 0) {
      editmode = 0;
   }

   gEditLine = el_init(rl_readline_name, rl_instream, rl_outstream, stderr);

   if (!editmode) {
      el_set(gEditLine, EL_EDITMODE, 0);
   }

   gHistory = history_init();

   if (!gEditLine || !gHistory) {
      return -1;
   }

   history(gHistory, &ev, H_SETSIZE, INT_MAX);         /* unlimited */
   history_length = 0;
   max_input_history = INT_MAX;
   el_set(gEditLine, EL_HIST, history, gHistory);

   /* for proper prompt printing in readline() */
   gel_rl_prompt = strdup("");
   el_set(gEditLine, EL_PROMPT, _get_prompt);
   el_set(gEditLine, EL_SIGNAL, 1);


   /* set default mode to "emacs"-style and read setting afterwards */
   /* so this can be overriden */
   // NO vi at the ROOT prompt, please!
   char* editor = 0; // getenv("EDITOR");
   // coverity[dead_error_line] - yes, this is always "emacs".
   el_set(gEditLine, EL_EDITOR, editor ? editor : "emacs");

   /*
    * Word completition - this has to go AFTER rebinding keys
    * to emacs-style.
    */
   el_set(gEditLine, EL_ADDFN, "rl_complete",
          "ReadLine compatible completition function",
          _el_rl_complete);
   el_set(gEditLine, EL_BIND, "^I", "rl_complete", NULL);

   /** added by stephan: default emacs bindings... */
   el_set(gEditLine, EL_BIND, "^R", "em-inc-search-prev", NULL);
   el_set(gEditLine, EL_BIND, "^S", "em-inc-search-next", NULL);
   /** end stephan's personal preferences. */

   /*
    * Find out where the rl_complete function was added; this is
    * used later to detect that lastcmd was also rl_complete.
    */
   for (i = EL_NUM_FCNS; i < gEditLine->fMap.fNFunc; i++) {
      if (gEditLine->fMap.fFunc[i] == _el_rl_complete) {
         gel_rl_complete_cmdnum = i;
         break;
      }
   }


   /* read settings from configuration file */
   el_source(gEditLine, NULL);

   /*
    * Unfortunately, some applications really do use rl_point
    * and rl_line_buffer directly.
    */
   li = el_line(gEditLine);
   /* LINTED const cast */
   rl_line_buffer = (char*) li->fBuffer;
   rl_point = rl_end = 0;

   return 0;
} // rl_initialize


/*
 * Ends readline/editline mode. Intended to be called
 * from signal handlers.
 *
 * Submitted to editline by Alexey Zakhlestin, of Milk Farm Software.
 */
void
rl_cleanup_after_signal() {
   el_end(gEditLine);
   history_end(gHistory);
}


/*
 * read one line from input stream and return it, chomping
 * trailing newline (if there is any)
 *
 * Calls to libeditline builtin functions are consumed here, and those
 * lines are not passed back to the client (who doesn't need
 * them). They ARE added to add_history().
 */
#include <iostream>
char*
readline(const char* prompt, bool newline) {
   HistEvent_t ev;
   int count;
   const char* ret;

   if (gEditLine == NULL || gHistory == NULL) {
      rl_initialize();
   }

   /* update prompt accordingly to what has been passed; */
   /* also disable prompt if nobody can type at it, or if nobody sees it */
   if (!prompt || !isatty(0) || !isatty(1)) {
      prompt = "";
   }

   if (strcmp(gel_rl_prompt, prompt) != 0) {
      free(gel_rl_prompt);
      gel_rl_prompt = strdup(prompt);
   }

   /* get one line from input stream */
   if (newline) {
      tty_rawmode(gEditLine);
      ret = el_gets_newline(gEditLine, &count);
   } else {
      ret = el_gets(gEditLine, &count);

      if (rl_in_key_hook && count > 0 && gEditLine->fLine.fLastChar > gEditLine->fLine.fBuffer) {
         char* key = gEditLine->fLine.fLastChar;

         if (*key == '\a') {
            --key;
         }
         (*rl_in_key_hook)(*key);
      }
   }

   // std::cerr <<  "readline(): el_gets(gEditLine,"<<count<<") got ["<<(ret?ret:"NULL")<<"]\n";
   //if( 0 == ret ) return NULL;

   // reminder: gEditLine owns the string, actually,
   // which is why we duplicate it here:
   if (ret) {
      if (count <= 0) {
         ret = NULL;
      }
   }

   history(gHistory, &ev, H_GETSIZE);
   history_length = ev.fNum;
   if (ret && (!*ret ||
               (!strchr(ret, '\a') && ret[strlen(ret) - 1] == '\n'))) {
      tty_cookedmode(gEditLine);
   }
   /* LINTED const cast */
   return (char*) ret;
} // readline


void
setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
          const char* colorBadBracket, const char* colorPrompt) {

   tab_color = term__atocolor(colorTabComp);
   prompt_setcolor(term__atocolor(colorPrompt));

   setKeywordColors(term__atocolor(colorTab),
                    term__atocolor(colorBracket),
                    term__atocolor(colorBadBracket));
}


void
termResize(void) {
   el_resize(gEditLine);  // this is called by SIGWINCH when term is resized - detects term width itself
}


void
setEcho(bool echo) {
   if (echo) {
      tty_noquotemode(gEditLine);
   } else {
      tty_quotemode(gEditLine);
   }
}


/*
 * history functions
 */

/*
 * is normally called before application starts to use
 * history expansion functions
 */
void
using_history(void) {
   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }
}


#ifdef EL_HISTORY_EXPAND
/*
 * substitute ``what'' with ``with'', returning resulting string; if
 * globally == 1, substitutes all occurences of what, otherwise only the
 * first one
 */
static char*
_rl_compat_sub(const char* str, const char* what, const char* with,
               int globally) {
   char* result;
   const char* temp, * newp;
   size_t len, add;
   int with_len, what_len;
   size_t size, i;

   result = (char*) malloc((size = 16));
   temp = str;
   with_len = strlen(with);
   what_len = strlen(what);
   len = 0;

   do {
      newp = strstr(temp, what);

      if (newp) {
         i = newp - temp;
         add = i + with_len;

         if (i + add + 1 >= size) {
            size += add + 1;
            result = (char*) realloc(result, size);
         }
         (void) strncpy(&result[len], temp, i);
         len += i;
         (void) strcpy(&result[len], with);                     /* safe */
         len += with_len;
         temp = newp + what_len;
      } else {
         add = strlen(temp);

         if (len + add + 1 >= size) {
            size += add + 1;
            result = (char*) realloc(result, size);
         }
         (void) strcpy(&result[len], temp);                     /* safe */
         len += add;
         temp = NULL;
      }
   }
   while (temp && globally);
   result[len] = '\0';

   return result;
} // _rl_compat_sub
#endif


#ifdef EL_HISTORY_EXPAND
/*
 * the real function doing history expansion - takes as argument command
 * to do and data upon which the command should be executed
 * does expansion the way I've understood readline documentation
 * word designator ``%'' isn't supported (yet ?)
 *
 * returns 0 if data was not modified, 1 if it was and 2 if the string
 * should be only printed and not executed; in case of error,
 * returns -1 and *result points to NULL
 * it's callers responsibility to free() string returned in *result
 */
static int
_history_expand_command(const char* command, size_t cmdlen, char** result) {
   char** arr, * tempcmd, * line, * search = NULL, * cmd;
   const char* event_data = NULL;
   static char* from = NULL, * to = NULL;
   int start = -1, end = -1, max, i, idx;
   int gHistory_on = 0, t_on = 0, r_on = 0, gEditLine_on = 0, p_on = 0, g_on = 0;
   int event_num = 0, retval;
   size_t cmdsize;

   *result = NULL;

   cmd = (char*) alloca(cmdlen + 1);
   (void) strncpy(cmd, command, cmdlen);
   cmd[cmdlen] = 0;

   idx = 1;

   /* find out which event to take */
   if (cmd[idx] == history_expansion_char) {
      event_num = history_length;
      idx++;
   } else {
      int off, num;
      size_t len;
      off = idx;

      while (cmd[off] && !strchr(":^$*-%", cmd[off]))
         off++;
      num = atoi(&cmd[idx]);

      if (num != 0) {
         event_num = num;

         if (num < 0) {
            event_num += history_length + 1;
         }
      } else {
         int prefix = 1, curr_num;
         HistEvent_t ev;

         len = off - idx;

         if (cmd[idx] == '?') {
            idx++, len--;

            if (cmd[off - 1] == '?') {
               len--;
            } else if (cmd[off] != '\n' && cmd[off] != '\0') {
               return -1;
            }
            prefix = 0;
         }
         search = (char*) alloca(len + 1);
         (void) strncpy(search, &cmd[idx], len);
         search[len] = '\0';

         if (history(gHistory, &ev, H_CURR) != 0) {
            return -1;
         }
         curr_num = ev.fNum;

         if (prefix) {
            retval = history_search_prefix(search, -1);
         } else {
            retval = history_search(search, -1);
         }

         if (retval == -1) {
            fprintf(rl_outstream, "%s: Event not found\n",
                    search);
            return -1;
         }

         if (history(gHistory, &ev, H_CURR) != 0) {
            return -1;
         }
         event_data = ev.fStr;

         /* roll back to original position */
         history(gHistory, &ev, H_NEXT_EVENT, curr_num);
      }
      idx = off;
   }

   if (!event_data && event_num >= 0) {
      HIST_ENTRY* rl_he;
      rl_he = history_get(event_num);

      if (!rl_he) {
         return 0;
      }
      event_data = rl_he->line;
   } else {
      return -1;
   }

   if (cmd[idx] != ':') {
      return -1;
   }
   cmd += idx + 1;

   /* recognize cmd */
   if (*cmd == '^') {
      start = end = 1, cmd++;
   } else if (*cmd == '$') {
      start = end = -1, cmd++;
   } else if (*cmd == '*') {
      start = 1, end = -1, cmd++;
   } else if (isdigit((unsigned char) *cmd)) {
      const char* temp;
      int shifted = 0;

      start = atoi(cmd);
      temp = cmd;

      for ( ; isdigit((unsigned char) *cmd); cmd++) {
      }

      if (temp != cmd) {
         shifted = 1;
      }

      if (shifted && *cmd == '-') {
         if (!isdigit((unsigned char) *(cmd + 1))) {
            end = -2;
         } else {
            end = atoi(cmd + 1);

            for ( ; isdigit((unsigned char) *cmd); cmd++) {
            }
         }
      } else if (shifted && *cmd == '*') {
         end = -1, cmd++;
      } else if (shifted) {
         end = start;
      }
   }

   if (*cmd == ':') {
      cmd++;
   }

   line = strdup(event_data);

   for ( ; *cmd; cmd++) {
      if (*cmd == ':') {
         continue;
      } else if (*cmd == 'h') {
         gHistory_on = 1 | g_on, g_on = 0;
      } else if (*cmd == 't') {
         t_on = 1 | g_on, g_on = 0;
      } else if (*cmd == 'r') {
         r_on = 1 | g_on, g_on = 0;
      } else if (*cmd == 'e') {
         gEditLine_on = 1 | g_on, g_on = 0;
      } else if (*cmd == 'p') {
         p_on = 1 | g_on, g_on = 0;
      } else if (*cmd == 'g') {
         g_on = 2;
      } else if (*cmd == 's' || *cmd == '&') {
         char* what, * with, delim;
         size_t len, from_len;
         size_t size;

         if (*cmd == '&' && (from == NULL || to == NULL)) {
            continue;
         } else if (*cmd == 's') {
            delim = *(++cmd), cmd++;
            size = 16;
            what = (char*) realloc(from, size);
            len = 0;

            for ( ; *cmd && *cmd != delim; cmd++) {
               if (*cmd == '\\'
                   && *(cmd + 1) == delim) {
                  cmd++;
               }

               if (len >= size) {
                  what = (char*) realloc(what,
                                         (size <<= 1));
               }
               what[len++] = *cmd;
            }
            what[len] = '\0';
            from = what;

            if (*what == '\0') {
               free(what);

               if (search) {
                  from = strdup(search);
               } else {
                  from = NULL;
                  free(line);
                  return -1;
               }
            }
            cmd++;                      /* shift after delim */

            if (!*cmd) {
               continue;
            }

            size = 16;
            with = (char*) realloc(to, size);
            len = 0;
            from_len = strlen(from);

            for ( ; *cmd && *cmd != delim; cmd++) {
               if (len + from_len + 1 >= size) {
                  size += from_len + 1;
                  with = (char*) realloc(with, size);
               }

               if (*cmd == '&') {
                  /* safe */
                  (void) strcpy(&with[len], from);
                  len += from_len;
                  continue;
               }

               if (*cmd == '\\'
                   && (*(cmd + 1) == delim
                       || *(cmd + 1) == '&')) {
                  cmd++;
               }
               with[len++] = *cmd;
            }
            with[len] = '\0';
            to = with;

            tempcmd = _rl_compat_sub(line, from, to,
                                     (g_on) ? 1 : 0);
            free(line);
            line = tempcmd;
            g_on = 0;
         }
      }
   }

   arr = history_tokenize(line);
   free(line);                  /* no more needed */

   if (arr && *arr == NULL) {
      free(arr), arr = NULL;
   }

   if (!arr) {
      return -1;
   }

   /* find out max valid idx to array of array */
   max = 0;

   for (i = 0; arr[i]; i++) {
      max++;
   }
   max--;

   /* set boundaries to something relevant */
   if (start < 0) {
      start = 1;
   }

   if (end < 0) {
      end = max - ((end < -1) ? 1 : 0);
   }

   /* check boundaries ... */
   if (start > max || end > max || start > end) {
      free(arr);
      return -1;
   }

   for (i = 0; i <= max; i++) {
      char* temp;

      if (gHistory_on && (i == 1 || gHistory_on > 1) &&
          (temp = strrchr(arr[i], '/'))) {
         *(temp + 1) = '\0';
      }

      if (t_on && (i == 1 || t_on > 1) &&
          (temp = strrchr(arr[i], '/'))) {
         (void) strcpy(arr[i], temp + 1);
      }

      if (r_on && (i == 1 || r_on > 1) &&
          (temp = strrchr(arr[i], '.'))) {
         *temp = '\0';
      }

      if (gEditLine_on && (i == 1 || gEditLine_on > 1) &&
          (temp = strrchr(arr[i], '.'))) {
         (void) strcpy(arr[i], temp);
      }
   }

   cmdsize = 1, cmdlen = 0;
   tempcmd = (char*) malloc(cmdsize);

   for (i = start; start <= i && i <= end; i++) {
      int arr_len;

      arr_len = strlen(arr[i]);

      if (cmdlen + arr_len + 1 >= cmdsize) {
         cmdsize += arr_len + 1;
         tempcmd = (char*) realloc(tempcmd, cmdsize);
      }
      (void) strcpy(&tempcmd[cmdlen], arr[i]);                  /* safe */
      cmdlen += arr_len;
      tempcmd[cmdlen++] = ' ';                  /* add a space */
   }

   while (cmdlen > 0 && isspace((unsigned char) tempcmd[cmdlen - 1]))
      cmdlen--;
   tempcmd[cmdlen] = '\0';

   *result = tempcmd;

   for (i = 0; i <= max; i++) {
      free(arr[i]);
   }
   free(arr), arr = (char**) NULL;
   return (p_on) ? 2 : 1;
} // _history_expand_command
#endif

/*
 * csh-style history expansion
 */
#ifdef EL_HISTORY_EXPAND
int
history_expand(char* str, char** output) {
   int i, retval = 0, idx;
   size_t size;
   char* temp, * result;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   *output = strdup(str);       /* do it early */

   if (str[0] == history_subst_char) {
      /* ^foo^foo2^ is equivalent to !!:s^foo^foo2^ */
      temp = (char*) alloca(4 + strlen(str) + 1);
      temp[0] = temp[1] = history_expansion_char;
      temp[2] = ':';
      temp[3] = 's';
      (void) strcpy(temp + 4, str);
      str = temp;
   }
#define ADD_STRING(what, len) \
   { \
      if (idx + len + 1 > size) { \
         result = (char*) realloc(result, (size += len + 1)); } \
                                           (void) strncpy(&result[idx], what, len); \
         idx += len; \
         result[idx] = '\0'; \
      }

   result = NULL;
   size = idx = 0;

   for (i = 0; str[i];) {
      int start, j, loop_again;
      size_t len;

      loop_again = 1;
      start = j = i;
loop:

      for ( ; str[j]; j++) {
         if (str[j] == '\\' &&
             str[j + 1] == history_expansion_char) {
            (void) strcpy(&str[j], &str[j + 1]);
            continue;
         }

         if (!loop_again) {
            if (str[j] == '?') {
               while (str[j] && str[++j] != '?')
                  ;

               if (str[j] == '?') {
                  j++;
               }
            } else if (isspace((unsigned char) str[j])) {
               break;
            }
         }

         if (str[j] == history_expansion_char
             && !strchr(history_no_expand_chars, str[j + 1])
             && (!history_inhibit_expansion_function ||
                 (*history_inhibit_expansion_function)(str, j) == 0)) {
            break;
         }
      }

      if (str[j] && str[j + 1] != '#' && loop_again) {
         i = j;
         j++;

         if (str[j] == history_expansion_char) {
            j++;
         }
         loop_again = 0;
         goto loop;
      }
      len = i - start;
      temp = &str[start];
      ADD_STRING(temp, len);

      if (str[i] == '\0' || str[i] != history_expansion_char
          || str[i + 1] == '#') {
         len = j - i;
         temp = &str[i];
         ADD_STRING(temp, len);

         if (start == 0) {
            retval = 0;
         } else {
            retval = 1;
         }
         break;
      }
      retval = _history_expand_command(&str[i], (size_t) (j - i),
                                       &temp);

      if (retval != -1) {
         len = strlen(temp);
         ADD_STRING(temp, len);
      }
      i = j;
   }                            /* for(i ...) */

   if (retval == 2) {
      add_history(temp);
#ifdef GDB_411_HACK
      /* gdb 4.11 has been shipped with readline, where */
      /* history_expand() returned -1 when the line	  */
      /* should not be executed; in readline 2.1+	  */
      /* it should return 2 in such a case		  */
      retval = -1;
#endif
   }
   free(*output);
   *output = result;

   return retval;
} // history_expand
#endif

/*
 * Parse the string into individual tokens, similarily to how shell would do it.
 */
char**
history_tokenize(const char* str) {
   int size = 1, result_idx = 0, i, start;
   size_t len;
   char** result = NULL, * temp, delim = '\0';

   for (i = 0; str[i]; i++) {
      while (isspace((unsigned char) str[i]))
         i++;
      start = i;

      for ( ; str[i]; i++) {
         if (str[i] == '\\') {
            if (str[i + 1] != '\0') {
               i++;
            }
         } else if (str[i] == delim) {
            delim = '\0';
         } else if (!delim &&
                    (isspace((unsigned char) str[i]) ||
                     strchr("()<>;&|$", str[i]))) {
            break;
         } else if (!delim && strchr("'`\"", str[i])) {
            delim = str[i];
         }
      }

      if (result_idx + 2 >= size) {
         size <<= 1;
         result = (char**) realloc(result, size * sizeof(char*));
      }
      len = i - start;
      temp = (char*) malloc(len + 1);
      (void) strncpy(temp, &str[start], len);
      temp[len] = '\0';
      result[result_idx++] = temp;
      result[result_idx] = NULL;
   }

   return result;
} // history_tokenize


/*
 * limit size of history record to ``max'' events
 */
void
stifle_history(int max) {
   HistEvent_t ev;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   if (history(gHistory, &ev, H_SETSIZE, max) == 0) {
      max_input_history = max;
   }
}


/*
 * "unlimit" size of history - set the limit to maximum allowed int value
 */
int
unstifle_history(void) {
   HistEvent_t ev;
   int omax;

   history(gHistory, &ev, H_SETSIZE, INT_MAX);
   omax = max_input_history;
   max_input_history = INT_MAX;
   return omax;                 /* some value _must_ be returned */
}


int
history_is_stifled(void) {
   /* cannot return true answer */
   return max_input_history != INT_MAX;
}


/*
 * read history from a file given
 */
int
read_history(const char* filename) {
   HistEvent_t ev;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }
   return history(gHistory, &ev, H_LOAD, filename);
}


/*
 * write history to a file given
 */
int
write_history(const char* filename) {
   HistEvent_t ev;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }
   return history(gHistory, &ev, H_SAVE, filename);
}


/*
 * returns history ``num''th event
 *
 * returned pointer points to static variable
 */
HIST_ENTRY*
history_get(int num) {
   static HIST_ENTRY she;
   HistEvent_t ev;
   int i = 1, curr_num;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   /* rewind to beginning */
   if (history(gHistory, &ev, H_CURR) != 0) {
      return NULL;
   }
   curr_num = ev.fNum;

   if (history(gHistory, &ev, H_LAST) != 0) {
      return NULL;              /* error */
   }

   while (i < num && history(gHistory, &ev, H_PREV) == 0)
      i++;

   if (i != num) {
      return NULL;              /* not so many entries */

   }
   she.line = ev.fStr;
   she.data = NULL;

   /* rewind history to the same event it was before */
   (void) history(gHistory, &ev, H_FIRST);
   (void) history(gHistory, &ev, H_NEXT_EVENT, curr_num);

   return &she;
} // history_get


/*
 * add the line to history table
 */
int
add_history(char* line) {
   HistEvent_t ev;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   size_t len = strlen(line);
   char oldlast = line[len - 1];

   if (oldlast == '\n') {
      // remove trailing newline; it would add a second, empty history entry
      line[len - 1] = 0;
   }

   if (line[0]) {
      // no empty lines in history, please
      (void) history(gHistory, &ev, H_ENTER, line);

      if (history(gHistory, &ev, H_GETSIZE) == 0) {
         history_length = ev.fNum;
      }
   }
   line[len - 1] = oldlast;

   return !(history_length > 0);        /* return 0 if all is okay */
} // add_history


/*
 * clear the history list - delete all entries
 */
void
clear_history(void) {
   HistEvent_t ev;

   history(gHistory, &ev, H_CLEAR);
}


/*
 * returns offset of the current history event
 */
int
where_history(void) {
   HistEvent_t ev;
   int curr_num, off;

   if (history(gHistory, &ev, H_CURR) != 0) {
      return 0;
   }
   curr_num = ev.fNum;

   history(gHistory, &ev, H_FIRST);
   off = 1;

   while (ev.fNum != curr_num && history(gHistory, &ev, H_NEXT) == 0)
      off++;

   return off;
} // where_history


/*
 * returns current history event or NULL if there is no such event
 */
HIST_ENTRY*
current_history(void) {
   return _move_history(H_CURR);
}


/*
 * returns total number of bytes history events' data are using
 */
int
history_total_bytes(void) {
   HistEvent_t ev;
   int curr_num, size;

   if (history(gHistory, &ev, H_CURR) != 0) {
      return -1;
   }
   curr_num = ev.fNum;

   history(gHistory, &ev, H_FIRST);
   size = 0;

   do {
      size += strlen(ev.fStr);
   }
   while (history(gHistory, &ev, H_NEXT) == 0);

   /* get to the same position as before */
   history(gHistory, &ev, H_PREV_EVENT, curr_num);

   return size;
} // history_total_bytes


/*
 * sets the position in the history list to ``pos''
 */
int
history_set_pos(int pos) {
   HistEvent_t ev;
   int off, curr_num;

   if (pos > history_length || pos < 0) {
      return -1;
   }

   history(gHistory, &ev, H_CURR);
   curr_num = ev.fNum;
   history(gHistory, &ev, H_FIRST);
   off = 0;

   while (off < pos && history(gHistory, &ev, H_NEXT) == 0)
      off++;

   if (off != pos) {            /* do a rollback in case of error */
      history(gHistory, &ev, H_FIRST);
      history(gHistory, &ev, H_NEXT_EVENT, curr_num);
      return -1;
   }
   return 0;
} // history_set_pos


/*
 * returns previous event in history and shifts pointer accordingly
 */
HIST_ENTRY*
previous_history(void) {
   return _move_history(H_PREV);
}


/*
 * returns next event in history and shifts pointer accordingly
 */
HIST_ENTRY*
next_history(void) {
   return _move_history(H_NEXT);
}


/*
 * generic history search function
 */
static int
_history_search_gen(const char* str, int direction, int pos) {
   HistEvent_t ev;
   const char* strp;
   int curr_num;

   if (history(gHistory, &ev, H_CURR) != 0) {
      return -1;
   }
   curr_num = ev.fNum;

   for ( ; ;) {
      strp = strstr(ev.fStr, str);

      if (strp && (pos < 0 || &ev.fStr[pos] == strp)) {
         return (int) (strp - ev.fStr);
      }

      if (history(gHistory, &ev, direction < 0 ? H_PREV : H_NEXT) != 0) {
         break;
      }
   }

   history(gHistory, &ev, direction < 0 ? H_NEXT_EVENT : H_PREV_EVENT, curr_num);

   return -1;
} // _history_search_gen


/*
 * searches for first history event containing the str
 */
int
history_search(const char* str, int direction) {
   return _history_search_gen(str, direction, -1);
}


/*
 * searches for first history event beginning with str
 */
int
history_search_prefix(const char* str, int direction) {
   return _history_search_gen(str, direction, 0);
}


/*
 * search for event in history containing str, starting at offset
 * abs(pos); continue backward, if pos<0, forward otherwise
 */
/* ARGSUSED */
int
history_search_pos(const char* str, int /*direction*/, int pos) {
   HistEvent_t ev;
   int curr_num, off;

   off = (pos > 0) ? pos : -pos;
   pos = (pos > 0) ? 1 : -1;

   if (history(gHistory, &ev, H_CURR) != 0) {
      return -1;
   }
   curr_num = ev.fNum;

   if (history_set_pos(off) != 0 || history(gHistory, &ev, H_CURR) != 0) {
      return -1;
   }

   for ( ; ;) {
      if (strstr(ev.fStr, str)) {
         return off;
      }

      if (history(gHistory, &ev, (pos < 0) ? H_PREV : H_NEXT) != 0) {
         break;
      }
   }

   /* set "current" pointer back to previous state */
   history(gHistory, &ev, (pos < 0) ? H_NEXT_EVENT : H_PREV_EVENT, curr_num);

   return -1;
} // history_search_pos


/********************************/
/* completition functions	*/

/*
 * does tilde expansion of strings of type ``~user/foo''
 * if ``user'' isn't valid user name or ``txt'' doesn't start
 * w/ '~', returns pointer to strdup()ed copy of ``txt''
 *
 * it's callers's responsibility to free() returned string
 */
char*
tilde_expand(char* txt) {
   struct passwd* pass;
   char* temp;
   size_t len = 0;

   if (txt[0] != '~') {
      return strdup(txt);
   }

   temp = strchr(txt + 1, '/');

   if (temp == NULL) {
      temp = strdup(txt + 1);
   } else {
      len = temp - txt + 1;             /* text until string after slash */
      temp = (char*) malloc(len);
      (void) strncpy(temp, txt + 1, len - 2);
      temp[len - 2] = '\0';
   }
   pass = getpwnam(temp);
   free(temp);                  /* value no more needed */

   if (pass == NULL) {
      return strdup(txt);
   }

   /* update pointer txt to point at string immedially following */
   /* first slash */
   txt += len;

   temp = (char*) malloc(strlen(pass->pw_dir) + 1 + strlen(txt) + 1);
   (void) sprintf(temp, "%s/%s", pass->pw_dir, txt);

   return temp;
} // tilde_expand


/*
 * return first found file name starting by the ``text'' or NULL if no
 * such file can be found
 * value of ``state'' is ignored
 *
 * it's caller's responsibility to free returned string
 */
char*
filename_completion_function(const char* text, int state) {
   static DIR* dir = NULL;
   static char* filename = NULL, * dirname = NULL;
   static size_t filename_len = 0;
   struct dirent* entry;
   char* temp;
   size_t len;

   if (state == 0 || dir == NULL) {
      if (dir != NULL) {
         closedir(dir);
         dir = NULL;
      }
      temp = strrchr((char*)text, '/');

      if (temp) {
         temp++;
         filename = (char*) realloc(filename, strlen(temp) + 1);
         (void) strcpy(filename, temp);
         len = temp - text;                     /* including last slash */
         dirname = (char*) realloc(dirname, len + 1);
         (void) strncpy(dirname, text, len);
         dirname[len] = '\0';
      } else {
         filename = strdup(text);
         dirname = NULL;
      }

      /* support for ``~user'' syntax */
      if (dirname && *dirname == '~') {
         temp = tilde_expand(dirname);
         dirname = (char*) realloc(dirname, strlen(temp) + 1);
         (void) strcpy(dirname, temp);                  /* safe */
         free(temp);                    /* no longer needed */
      }
      /* will be used in cycle */
      filename_len = strlen(filename);

      if (filename_len == 0) {
         return NULL;                   /* no expansion possible */

      }
      dir = opendir(dirname ? dirname : ".");

      if (!dir) {
         return NULL;                   /* cannot open the directory */
      }
   }

   /* find the match */
   while ((entry = readdir(dir)) != NULL) {
      /* otherwise, get first entry where first */
      /* filename_len characters are equal	  */
      if (entry->d_name[0] == filename[0]
#if defined(__SVR4) || defined(__linux__) || defined(__CYGWIN__)
          && strlen(entry->d_name) >= filename_len
#else
          && entry->d_namlen >= filename_len
#endif
          && strncmp(entry->d_name, filename,
                     filename_len) == 0) {
         break;
      }
   }

   if (entry) {                 /* match found */
      struct stat stbuf;
#if defined(__SVR4) || defined(__linux__) || defined(__CYGWIN__)
      len = strlen(entry->d_name) +
#else
      len = entry->d_namlen +
#endif
            ((dirname) ? strlen(dirname) : 0) + 1 + 1;
      temp = (char*) malloc(len);
      (void) sprintf(temp, "%s%s",
                     dirname ? dirname : "", entry->d_name);    /* safe */

      /* test, if it's directory */
      if (stat(temp, &stbuf) == 0 && S_ISDIR(stbuf.st_mode)) {
         strcat(temp, "/");                     /* safe */
      }
   } else {
      temp = NULL;
   }

   return temp;
} // filename_completion_function


/*
 * a completion generator for usernames; returns _first_ username
 * which starts with supplied text
 * text contains a partial username preceded by random character
 * (usually '~'); state is ignored
 * it's callers responsibility to free returned value
 */
char*
username_completion_function(const char* text, int state) {
   struct passwd* pwd;

   if (text[0] == '\0') {
      return NULL;
   }

   if (*text == '~') {
      text++;
   }

   if (state == 0) {
      setpwent();
   }

   while ((pwd = getpwent()) && text[0] == pwd->pw_name[0]
          && strcmp(text, pwd->pw_name) == 0)
      ;

   if (pwd == NULL) {
      endpwent();
      return NULL;
   }
   return strdup(pwd->pw_name);
} // username_completion_function


/*
 * el-compatible wrapper around rl_complete; needed for key binding
 */
/* ARGSUSED */
static unsigned char
_el_rl_complete(EditLine_t* /*el*/, int ch) {
   return (unsigned char) rl_complete(0, ch);
}


/*
 * returns list of completitions for text given
 */
char**
completion_matches(const char* text, CPFunction* genfunc) {
   char** match_list = NULL, * retstr, * prevstr;
   size_t match_list_len, max_equal, which, i;
   size_t matches;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   matches = 0;
   match_list_len = 1;

   while ((retstr = (*genfunc)(text, matches)) != NULL) {
      if (matches + 1 >= match_list_len) {
         match_list_len <<= 1;
         match_list = (char**) realloc(match_list,
                                       match_list_len * sizeof(char*));
      }
      match_list[++matches] = retstr;
   }

   if (!match_list) {
      return (char**) NULL;             /* nothing found */

   }
   /* find least denominator and insert it to match_list[0] */
   which = 2;
   prevstr = match_list[1];
   max_equal = strlen(prevstr);

   for ( ; which <= matches; which++) {
      for (i = 0; i < max_equal &&
           prevstr[i] == match_list[which][i]; i++) {
         continue;
      }
      max_equal = i;
   }

   retstr = (char*) malloc(max_equal + 1);
   (void) strncpy(retstr, match_list[1], max_equal);
   retstr[max_equal] = '\0';
   match_list[0] = retstr;

   /* add NULL as last pointer to the array */
   if (matches + 1 >= match_list_len) {
      match_list = (char**) realloc(match_list,
                                    (match_list_len + 1) * sizeof(char*));
   }
   match_list[matches + 1] = (char*) NULL;

   return match_list;
} // completion_matches


/*
 * Sort function for qsort(). Just wrapper around strcasecmp().
 */
static int
_rl_qsort_string_compare(const void* i1, const void* i2) {
   /*LINTED const castaway*/
   const char* s1 = ((const char**) i1)[0];
   /*LINTED const castaway*/
   const char* s2 = ((const char**) i2)[0];

   return strcasecmp(s1, s2);
}


/*
 * Display list of strings in columnar format on readline's output stream.
 * 'matches' is list of strings, 'len' is number of strings in 'matches',
 * 'max' is maximum length of string in 'matches'.
 */
void
rl_display_match_list(char** matches, int len, int max) {
   int i, idx, limit, count;
   int screenwidth = gEditLine->fTerm.fSize.fH;

   /*
    * Find out how many entries can be put on one line, count
    * with two spaces between strings.
    */
   limit = screenwidth / (max + 2);

   if (limit == 0) {
      limit = 1;
   }

   /* how many lines of output */
   count = len / limit;

   if (count * limit < len) {
      count++;
   }

   /* Sort the items if they are not already sorted. */
   qsort(&matches[1], (size_t) (len - 1), sizeof(char*),
         _rl_qsort_string_compare);

   idx = 1;

   for ( ; count > 0; count--) {
      for (i = 0; i < limit && matches[idx]; i++, idx++) {
         fprintf(gEditLine->fOutFile, "%-*s  ", max, matches[idx]);
      }
      fprintf(gEditLine->fOutFile, "\n");
   }
} // rl_display_match_list


/*
 * Complete the word at or before point, called by rl_complete()
 * 'what_to_do' says what to do with the completion.
 * `?' means list the possible completions.
 * TAB means do standard completion.
 * `*' means insert all of the possible completions.
 * `!' means to do standard completion, and list all possible completions if
 * there is more than one.
 *
 * Note: '*' support is not implemented
 */
static int
rl_complete_internal(int what_to_do) {
   CPFunction* complet_func;
   const LineInfo_t* li;
   char* temp, ** matches;
   const char* ctemp;
   size_t len;

   if (rl_tab_hook) {
      int cursorIdx = gEditLine->fLine.fCursor - gEditLine->fLine.fBuffer;
      char old = *gEditLine->fLine.fCursor;      // presumably \a
      *gEditLine->fLine.fCursor = 0;
      term__setcolor(tab_color);
      int loc = rl_tab_hook(gEditLine->fLine.fBuffer, 0, &cursorIdx);
      term__resetcolor();

      if (loc >= 0 || loc == -2 /* new line */ || cursorIdx != gEditLine->fLine.fCursor - gEditLine->fLine.fBuffer) {
         size_t lenbuf = strlen(gEditLine->fLine.fBuffer);
         gEditLine->fLine.fBuffer[lenbuf] = old;
         gEditLine->fLine.fBuffer[lenbuf + 1] = 0;

         for (int i = gEditLine->fLine.fCursor - gEditLine->fLine.fBuffer; i < cursorIdx; ++i) {
            gEditLine->fLine.fBufColor[i] = -1;
         }
         gEditLine->fLine.fCursor = gEditLine->fLine.fBuffer + cursorIdx;
         gEditLine->fLine.fLastChar = gEditLine->fLine.fCursor;

         if (loc == -2) {
            // spit out several lines; redraw prompt!
            re_clear_display(gEditLine);
         }
         re_refresh(gEditLine);
      } else {
         *gEditLine->fLine.fCursor = old;
      }
      return CC_NORM;
   }

   rl_completion_type = what_to_do;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   complet_func = rl_completion_entry_function;

   if (!complet_func) {
      complet_func = filename_completion_function;
   }

   /* We now look backwards for the start of a filename/variable word */
   li = el_line(gEditLine);
   ctemp = (const char*) li->fCursor;

   while (ctemp > li->fBuffer
          && !strchr(rl_basic_word_break_characters, ctemp[-1])
          && (!rl_special_prefixes
              || !strchr(rl_special_prefixes, ctemp[-1])))
      ctemp--;

   len = li->fCursor - ctemp;
   temp = (char*) alloca(len + 1);
   (void) strncpy(temp, ctemp, len);
   temp[len] = '\0';

   /* these can be used by function called in completion_matches() */
   /* or (*rl_attempted_completion_function)() */
   rl_point = li->fCursor - li->fBuffer;
   rl_end = li->fLastChar - li->fBuffer;

   if (!rl_attempted_completion_function) {
      matches = completion_matches(temp, complet_func);
   } else {
      int end = li->fCursor - li->fBuffer;
      matches = (*rl_attempted_completion_function)(temp, (int)
                                                    (end - len), end);
   }

   if (matches) {
      int i, retval = CC_REFRESH;
      int matches_num, maxlen, match_len, match_display = 1;

      /*
       * Only replace the completed string with common part of
       * possible matches if there is possible completion.
       */
      if (matches[0][0] != '\0') {
         el_deletestr(gEditLine, (int) len);
         el_insertstr(gEditLine, matches[0]);
      }

      if (what_to_do == '?') {
         goto display_matches;
      }

      if (matches[2] == NULL && strcmp(matches[0], matches[1]) == 0) {
         /*
          * We found exact match. Add a space after
          * it, unless we do filename completition and the
          * object is a directory.
          */
         size_t alen = strlen(matches[0]);

         if ((complet_func != filename_completion_function
              || (alen > 0 && (matches[0])[alen - 1] != '/'))
             && rl_completion_append_character) {
            char buf[2];
            buf[0] = rl_completion_append_character;
            buf[1] = '\0';
            el_insertstr(gEditLine, buf);
         }
      } else if (what_to_do == '!') {
display_matches:

         /*
          * More than one match and requested to list possible
          * matches.
          */

         for (i = 1, maxlen = 0; matches[i]; i++) {
            match_len = strlen(matches[i]);

            if (match_len > maxlen) {
               maxlen = match_len;
            }
         }
         matches_num = i - 1;

         /* newline to get on next line from command line */
         fprintf(gEditLine->fOutFile, "\n");

         /*
          * If there are too many items, ask user for display
          * confirmation.
          */
         if (matches_num > rl_completion_query_items) {
            fprintf(gEditLine->fOutFile,
                    "Display all %d possibilities? (y or n) ",
                    matches_num);
            fflush(gEditLine->fOutFile);

            if (getc(stdin) != 'y') {
               match_display = 0;
            }
            fprintf(gEditLine->fOutFile, "\n");
         }

         if (match_display) {
            rl_display_match_list(matches, matches_num,
                                  maxlen);
         }
         retval = CC_REDISPLAY;
      } else if (matches[0][0]) {
         /*
          * There was some common match, but the name was
          * not complete enough. Next tab will print possible
          * completions.
          */
         el_beep(gEditLine);
      } else {
         /* lcd is not a valid object - further specification */
         /* is needed */
         el_beep(gEditLine);
         retval = CC_NORM;
      }

      /* free elements of array and the array itself */
      for (i = 0; matches[i]; i++) {
         free(matches[i]);
      }
      free(matches), matches = NULL;

      return retval;
   }
   return CC_NORM;
} // rl_complete_internal


/*
 * complete word at current point
 */
int
rl_complete(int ignore, int invoking_key) {
   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   if (rl_inhibit_completion) {
      rl_insert(ignore, invoking_key);
      return CC_REFRESH;
   } else if (gEditLine->fState.fLastCmd == gel_rl_complete_cmdnum) {
      return rl_complete_internal('?');
   } else if (grl_complete_show_all) {
      return rl_complete_internal('!');
   } else {
      return rl_complete_internal(TAB);
   }
}


/*
 * misc other functions
 */

/*
 * bind key c to readline-type function func
 */
int
rl_bind_key(int c, int func(int, int)) {
   int retval = -1;

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   if (func == rl_insert) {
      /* XXX notice there is no range checking of ``c'' */
      gEditLine->fMap.fKey[c] = ED_INSERT;
      retval = 0;
   }
   return retval;
}


/*
 * read one key from input - handles chars pushed back
 * to input stream also
 */
int
rl_read_key(void) {
   char fooarr[2 * sizeof(int)];

   if (gEditLine == NULL || gHistory == NULL) {
      rl_initialize();
   }

   return el_getc(gEditLine, fooarr);
}


/*
 * reset the terminal
 */
/* ARGSUSED */
void
rl_reset_terminal(void) {
   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }
   el_reset(gEditLine);
}


/*
 * insert character ``c'' back into input stream, ``count'' times
 */
int
rl_insert(int count, int c) {
   char arr[2];

   if (gHistory == NULL || gEditLine == NULL) {
      rl_initialize();
   }

   /* XXX - int -> char conversion can lose on multichars */
   arr[0] = c;
   arr[1] = '\0';

   for ( ; count > 0; count--) {
      el_push(gEditLine, arr);
   }

   return 0;
} // rl_insert


EditLine_t*
el_readline_el() {
   if (NULL == gEditLine) {
      rl_initialize();
   }
   return gEditLine;
}


int
rl_eof() {
   return el_eof(gEditLine);
}
