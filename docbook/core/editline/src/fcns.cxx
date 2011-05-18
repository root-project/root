// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "sys.h"
#include "el.h"
el_private const ElFunc_t el_func[] = {
   ed_argument_digit, ed_clear_screen,
   ed_command, ed_delete_next_char,
   ed_delete_prev_char, ed_delete_prev_word,
   ed_digit, ed_end_of_file,
   ed_insert, ed_kill_line,
   ed_move_to_beg, ed_move_to_end,
   ed_newline, ed_next_char,
   ed_next_history, ed_next_line,
   ed_prev_char, ed_prev_history,
   ed_prev_line, ed_prev_word,
   ed_quoted_insert, ed_redisplay,
   ed_search_next_history, ed_search_prev_history,
   ed_sequence_lead_in, ed_start_over,
   ed_transpose_chars, ed_tty_dsusp,
   ed_tty_flush_output, ed_tty_sigint,
   ed_tty_sigquit, ed_tty_sigtstp,
   ed_tty_start_output, ed_tty_stop_output,
   ed_unassigned, em_capitol_case,
   em_copy_prev_word, em_copy_region,
   em_delete_next_word, em_delete_or_list,
   em_exchange_mark, em_gosmacs_traspose,
   em_inc_search_next, em_inc_search_prev,
   em_kill_line, em_kill_region,
   em_lower_case, em_meta_next,
   em_next_word, em_set_mark,
   em_toggle_overwrite, em_universal_argument,
   em_upper_case, em_yank,
#ifdef EL_USE_VI
   vi_add, vi_add_at_eol,
   vi_change_case, vi_change_meta,
   vi_change_to_eol, vi_command_mode,
   vi_delete_meta, vi_delete_prev_char,
   vi_end_word, vi_insert,
   vi_insert_at_bol, vi_kill_line_prev,
   vi_list_or_eof, vi_next_char,
   vi_next_space_word, vi_next_word,
   vi_paste_next, vi_paste_prev,
   vi_prev_char, vi_prev_space_word,
   vi_prev_word, vi_repeat_next_char,
   vi_repeat_prev_char, vi_repeat_search_next,
   vi_repeat_search_prev, vi_replace_char,
   vi_replace_mode, vi_search_next,
   vi_search_prev, vi_substitute_char,
   vi_substitute_line, vi_to_end_word,
   vi_to_next_char, vi_to_prev_char,
   vi_undo, vi_zero,
#else
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
   0, 0,
#endif
   em_undo, ed_replay_hist,
};

el_protected const ElFunc_t*
func__get() { return el_func; }
