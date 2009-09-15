// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef _h_emacs_c
#define _h_emacs_c
el_protected el_action_t em_delete_or_list(EditLine*, int);
el_protected el_action_t em_delete_next_word(EditLine*, int);
el_protected el_action_t em_yank(EditLine*, int);
el_protected el_action_t em_kill_line(EditLine*, int);
el_protected el_action_t em_kill_region(EditLine*, int);
el_protected el_action_t em_copy_region(EditLine*, int);
el_protected el_action_t em_gosmacs_traspose(EditLine*, int);
el_protected el_action_t em_next_word(EditLine*, int);
el_protected el_action_t em_upper_case(EditLine*, int);
el_protected el_action_t em_capitol_case(EditLine*, int);
el_protected el_action_t em_lower_case(EditLine*, int);
el_protected el_action_t em_set_mark(EditLine*, int);
el_protected el_action_t em_exchange_mark(EditLine*, int);
el_protected el_action_t em_universal_argument(EditLine*, int);
el_protected el_action_t em_meta_next(EditLine*, int);
el_protected el_action_t em_toggle_overwrite(EditLine*, int);
el_protected el_action_t em_copy_prev_word(EditLine*, int);
el_protected el_action_t em_inc_search_next(EditLine*, int);
el_protected el_action_t em_inc_search_prev(EditLine*, int);
el_protected el_action_t em_undo(EditLine*, int);
#endif /* _h_emacs_c */
