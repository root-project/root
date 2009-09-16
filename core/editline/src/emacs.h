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
el_protected ElAction_t em_delete_or_list(EditLine_t*, int);
el_protected ElAction_t em_delete_next_word(EditLine_t*, int);
el_protected ElAction_t em_yank(EditLine_t*, int);
el_protected ElAction_t em_kill_line(EditLine_t*, int);
el_protected ElAction_t em_kill_region(EditLine_t*, int);
el_protected ElAction_t em_copy_region(EditLine_t*, int);
el_protected ElAction_t em_gosmacs_traspose(EditLine_t*, int);
el_protected ElAction_t em_next_word(EditLine_t*, int);
el_protected ElAction_t em_upper_case(EditLine_t*, int);
el_protected ElAction_t em_capitol_case(EditLine_t*, int);
el_protected ElAction_t em_lower_case(EditLine_t*, int);
el_protected ElAction_t em_set_mark(EditLine_t*, int);
el_protected ElAction_t em_exchange_mark(EditLine_t*, int);
el_protected ElAction_t em_universal_argument(EditLine_t*, int);
el_protected ElAction_t em_meta_next(EditLine_t*, int);
el_protected ElAction_t em_toggle_overwrite(EditLine_t*, int);
el_protected ElAction_t em_copy_prev_word(EditLine_t*, int);
el_protected ElAction_t em_inc_search_next(EditLine_t*, int);
el_protected ElAction_t em_inc_search_prev(EditLine_t*, int);
el_protected ElAction_t em_undo(EditLine_t*, int);
#endif /* _h_emacs_c */
