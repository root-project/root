// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef _h_common_c
#define _h_common_c
el_protected ElAction_t ed_end_of_file(EditLine_t*, int);
el_protected ElAction_t ed_insert(EditLine_t*, int);
el_protected ElAction_t ed_delete_prev_word(EditLine_t*, int);
el_protected ElAction_t ed_delete_next_char(EditLine_t*, int);
el_protected ElAction_t ed_kill_line(EditLine_t*, int);
el_protected ElAction_t ed_move_to_end(EditLine_t*, int);
el_protected ElAction_t ed_move_to_beg(EditLine_t*, int);
el_protected ElAction_t ed_transpose_chars(EditLine_t*, int);
el_protected ElAction_t ed_next_char(EditLine_t*, int);
el_protected ElAction_t ed_prev_word(EditLine_t*, int);
el_protected ElAction_t ed_prev_char(EditLine_t*, int);
el_protected ElAction_t ed_quoted_insert(EditLine_t*, int);
el_protected ElAction_t ed_digit(EditLine_t*, int);
el_protected ElAction_t ed_argument_digit(EditLine_t*, int);
el_protected ElAction_t ed_unassigned(EditLine_t*, int);
el_protected ElAction_t ed_tty_sigint(EditLine_t*, int);
el_protected ElAction_t ed_tty_dsusp(EditLine_t*, int);
el_protected ElAction_t ed_tty_flush_output(EditLine_t*, int);
el_protected ElAction_t ed_tty_sigquit(EditLine_t*, int);
el_protected ElAction_t ed_tty_sigtstp(EditLine_t*, int);
el_protected ElAction_t ed_tty_stop_output(EditLine_t*, int);
el_protected ElAction_t ed_tty_start_output(EditLine_t*, int);
el_protected ElAction_t ed_newline(EditLine_t*, int);
el_protected ElAction_t ed_delete_prev_char(EditLine_t*, int);
el_protected ElAction_t ed_clear_screen(EditLine_t*, int);
el_protected ElAction_t ed_redisplay(EditLine_t*, int);
el_protected ElAction_t ed_start_over(EditLine_t*, int);
el_protected ElAction_t ed_sequence_lead_in(EditLine_t*, int);
el_protected ElAction_t ed_prev_history(EditLine_t*, int);
el_protected ElAction_t ed_next_history(EditLine_t*, int);
el_protected ElAction_t ed_search_prev_history(EditLine_t*, int);
el_protected ElAction_t ed_search_next_history(EditLine_t*, int);
el_protected ElAction_t ed_prev_line(EditLine_t*, int);
el_protected ElAction_t ed_next_line(EditLine_t*, int);
el_protected ElAction_t ed_command(EditLine_t*, int);
el_protected ElAction_t ed_replay_hist(EditLine_t*, int);
#endif /* _h_common_c */
