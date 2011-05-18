// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: readline.fH,v 1.1 2001/01/05 21:15:50 jdolecek Exp $	*/

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

#ifndef _READLINE_H_
#define _READLINE_H_

#include <stdio.h>
#include <sys/types.h>
#if HAVE_SYS_CDEFS_H
# include <sys/cdefs.h>
#else
/*
 #ifndef __BEGIN_DECLS
 #if defined(__cplusplus)
 #define __BEGIN_DECLS   extern "C" {
 #define __END_DECLS     }
 #else
 #define __BEGIN_DECLS
 #define __END_DECLS
 #endif
 #endif
*/
#endif

// Generic helper definitions for shared library support hiding readline symbols
// See http://gcc.gnu.org/wiki/Visibility
#if defined(__GNUC__) && (__GNUC__ >= 4)
# define R_EL__LOCAL  __attribute__ ((visibility("hidden")))
#else
# define R_EL__LOCAL
#endif

/* list of readline stuff supported by SEditLine_t library's readline wrapper */

/* typedefs */
typedef int Function (const char*, int);
typedef void VFunction (void);
typedef char* CPFunction (const char*, int);
typedef char** CPPFunction (const char*, int, int);
typedef int (*El_tab_hook_t)(char* buf, int prompt_width, int* cursor_loc);
typedef int (*El_in_key_hook_t)(int ch);

typedef struct R_EL__LOCAL _hist_entry {
   const char* line;
   const char* data;
} HIST_ENTRY;

/* global variables used by readline enabled applications */
extern R_EL__LOCAL const char* rl_library_version;
extern R_EL__LOCAL const char* rl_readline_name;
extern R_EL__LOCAL FILE* rl_instream;
extern R_EL__LOCAL FILE* rl_outstream;
extern R_EL__LOCAL char* rl_line_buffer;
extern R_EL__LOCAL int rl_point, rl_end;
extern R_EL__LOCAL int history_base, history_length;
extern R_EL__LOCAL int max_input_history;
extern R_EL__LOCAL const char* rl_basic_word_break_characters;
extern R_EL__LOCAL char* rl_completer_word_break_characters;
extern R_EL__LOCAL char* rl_completer_quote_characters;
extern R_EL__LOCAL CPFunction* rl_completion_entry_function;
extern R_EL__LOCAL CPPFunction* rl_attempted_completion_function;
extern R_EL__LOCAL int rl_completion_type;
extern R_EL__LOCAL int rl_completion_query_items;
extern R_EL__LOCAL char* rl_special_prefixes;
extern R_EL__LOCAL int rl_completion_append_character;
extern R_EL__LOCAL El_tab_hook_t rl_tab_hook;
extern R_EL__LOCAL El_tab_hook_t rl_tab_hook;
extern R_EL__LOCAL El_in_key_hook_t rl_in_key_hook;

/* supported functions */
R_EL__LOCAL char* readline(const char*, bool newline);
R_EL__LOCAL int rl_initialize(void);
R_EL__LOCAL bool rl_isinitialized();

R_EL__LOCAL void setEcho(bool echo);
R_EL__LOCAL void termResize(void);
R_EL__LOCAL void setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
               const char* colorBadBracket, const char* colorPrompt);

R_EL__LOCAL void using_history(void);
R_EL__LOCAL int add_history(char*);
R_EL__LOCAL void clear_history(void);
R_EL__LOCAL void stifle_history(int);
R_EL__LOCAL int unstifle_history(void);
R_EL__LOCAL int history_is_stifled(void);
R_EL__LOCAL int where_history(void);
R_EL__LOCAL HIST_ENTRY* current_history(void);
R_EL__LOCAL HIST_ENTRY* history_get(int);
R_EL__LOCAL int history_total_bytes(void);
R_EL__LOCAL int history_set_pos(int);
R_EL__LOCAL HIST_ENTRY* previous_history(void);
R_EL__LOCAL HIST_ENTRY* next_history(void);
R_EL__LOCAL int history_search(const char*, int);
R_EL__LOCAL int history_search_prefix(const char*, int);
R_EL__LOCAL int history_search_pos(const char*, int, int);
R_EL__LOCAL int read_history(const char*);
R_EL__LOCAL int write_history(const char*);
#ifdef EL_HISTORY_EXPAND
R_EL__LOCAL int history_expand(char*, char**);
#endif
R_EL__LOCAL char** history_tokenize(const char*);

R_EL__LOCAL char* tilde_expand(char*);
R_EL__LOCAL char* filename_completion_function(const char*, int);
R_EL__LOCAL char* username_completion_function(const char*, int);
R_EL__LOCAL int rl_complete(int, int);
R_EL__LOCAL int rl_read_key(void);
R_EL__LOCAL char** completion_matches(const char*, CPFunction*);
R_EL__LOCAL void rl_display_match_list(char**, int, int);

R_EL__LOCAL int rl_insert(int, int);
R_EL__LOCAL void rl_reset_terminal(void);
R_EL__LOCAL int rl_bind_key(int, int(*) (int, int));
R_EL__LOCAL int rl_eof(void);

R_EL__LOCAL void rl_cleanup_after_signal();

#endif /* _READLINE_H_ */
