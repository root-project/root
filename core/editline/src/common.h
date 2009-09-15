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
el_protected el_action_t ed_end_of_file(EditLine*, int);
el_protected el_action_t ed_insert(EditLine*, int);
el_protected el_action_t ed_delete_prev_word(EditLine*, int);
el_protected el_action_t ed_delete_next_char(EditLine*, int);
el_protected el_action_t ed_kill_line(EditLine*, int);
el_protected el_action_t ed_move_to_end(EditLine*, int);
el_protected el_action_t ed_move_to_beg(EditLine*, int);
el_protected el_action_t ed_transpose_chars(EditLine*, int);
el_protected el_action_t ed_next_char(EditLine*, int);
el_protected el_action_t ed_prev_word(EditLine*, int);
el_protected el_action_t ed_prev_char(EditLine*, int);
el_protected el_action_t ed_quoted_insert(EditLine*, int);
el_protected el_action_t ed_digit(EditLine*, int);
el_protected el_action_t ed_argument_digit(EditLine*, int);
el_protected el_action_t ed_unassigned(EditLine*, int);
el_protected el_action_t ed_tty_sigint(EditLine*, int);
el_protected el_action_t ed_tty_dsusp(EditLine*, int);
el_protected el_action_t ed_tty_flush_output(EditLine*, int);
el_protected el_action_t ed_tty_sigquit(EditLine*, int);
el_protected el_action_t ed_tty_sigtstp(EditLine*, int);
el_protected el_action_t ed_tty_stop_output(EditLine*, int);
el_protected el_action_t ed_tty_start_output(EditLine*, int);
el_protected el_action_t ed_newline(EditLine*, int);
el_protected el_action_t ed_delete_prev_char(EditLine*, int);
el_protected el_action_t ed_clear_screen(EditLine*, int);
el_protected el_action_t ed_redisplay(EditLine*, int);
el_protected el_action_t ed_start_over(EditLine*, int);
el_protected el_action_t ed_sequence_lead_in(EditLine*, int);
el_protected el_action_t ed_prev_history(EditLine*, int);
el_protected el_action_t ed_next_history(EditLine*, int);
el_protected el_action_t ed_search_prev_history(EditLine*, int);
el_protected el_action_t ed_search_next_history(EditLine*, int);
el_protected el_action_t ed_prev_line(EditLine*, int);
el_protected el_action_t ed_next_line(EditLine*, int);
el_protected el_action_t ed_command(EditLine*, int);
#endif /* _h_common_c */
