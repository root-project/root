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

/* list of readline stuff supported by SEditLine_t library's readline wrapper */

/* typedefs */
typedef int Function (const char*, int);
typedef void VFunction (void);
typedef char* CPFunction (const char*, int);
typedef char** CPPFunction (const char*, int, int);
typedef int (*El_tab_hook_t)(char* buf, int prompt_width, int* cursor_loc);
typedef int (*El_in_key_hook_t)(int ch);

typedef struct _hist_entry {
   const char* line;
   const char* data;
} HIST_ENTRY;

/* global variables used by readline enabled applications */
extern const char* rl_library_version;
extern const char* rl_readline_name;
extern FILE* rl_instream;
extern FILE* rl_outstream;
extern char* rl_line_buffer;
extern int rl_point, rl_end;
extern int history_base, history_length;
extern int max_input_history;
extern const char* rl_basic_word_break_characters;
extern char* rl_completer_word_break_characters;
extern char* rl_completer_quote_characters;
extern CPFunction* rl_completion_entry_function;
extern CPPFunction* rl_attempted_completion_function;
extern int rl_completion_type;
extern int rl_completion_query_items;
extern char* rl_special_prefixes;
extern int rl_completion_append_character;
extern El_tab_hook_t rl_tab_hook;
extern El_tab_hook_t rl_tab_hook;
extern El_in_key_hook_t rl_in_key_hook;

/* supported functions */
char* readline(const char*, bool newline);
int rl_initialize(void);
bool rl_isinitialized();

void setEcho(bool echo);
void termResize(void);
void setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
               const char* colorBadBracket, const char* colorPrompt);

void using_history(void);
int add_history(char*);
void clear_history(void);
void stifle_history(int);
int unstifle_history(void);
int history_is_stifled(void);
int where_history(void);
HIST_ENTRY* current_history(void);
HIST_ENTRY* history_get(int);
int history_total_bytes(void);
int history_set_pos(int);
HIST_ENTRY* previous_history(void);
HIST_ENTRY* next_history(void);
int history_search(const char*, int);
int history_search_prefix(const char*, int);
int history_search_pos(const char*, int, int);
int read_history(const char*);
int write_history(const char*);
int history_expand(char*, char**);
char** history_tokenize(const char*);

char* tilde_expand(char*);
char* filename_completion_function(const char*, int);
char* username_completion_function(const char*, int);
int rl_complete(int, int);
int rl_read_key(void);
char** completion_matches(const char*, CPFunction*);
void rl_display_match_list(char**, int, int);

int rl_insert(int, int);
void rl_reset_terminal(void);
int rl_bind_key(int, int(*) (int, int));
int rl_eof(void);

void rl_cleanup_after_signal();

#endif /* _READLINE_H_ */
