// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef _h_fcns_c
#define _h_fcns_c
#define ED_ARGUMENT_DIGIT 0
#define ED_CLEAR_SCREEN 1
#define ED_COMMAND 2
#define ED_DELETE_NEXT_CHAR 3
#define ED_DELETE_PREV_CHAR 4
#define ED_DELETE_PREV_WORD 5
#define ED_DIGIT 6
#define ED_END_OF_FILE 7
#define ED_INSERT 8
#define ED_KILL_LINE 9
#define ED_MOVE_TO_BEG 10
#define ED_MOVE_TO_END 11
#define ED_NEWLINE 12
#define ED_NEXT_CHAR 13
#define ED_NEXT_HISTORY 14
#define ED_NEXT_LINE 15
#define ED_PREV_CHAR 16
#define ED_PREV_HISTORY 17
#define ED_PREV_LINE 18
#define ED_PREV_WORD 19
#define ED_QUOTED_INSERT 20
#define ED_REDISPLAY 21
#define ED_SEARCH_NEXT_HISTORY 22
#define ED_SEARCH_PREV_HISTORY 23
#define ED_SEQUENCE_LEAD_IN 24
#define ED_START_OVER 25
#define ED_TRANSPOSE_CHARS 26
#define ED_TTY_DSUSP 27
#define ED_TTY_FLUSH_OUTPUT 28
#define ED_TTY_SIGINT 29
#define ED_TTY_SIGQUIT 30
#define ED_TTY_SIGTSTP 31
#define ED_TTY_START_OUTPUT 32
#define ED_TTY_STOP_OUTPUT 33
#define ED_UNASSIGNED 34
#define EM_CAPITOL_CASE 35
#define EM_COPY_PREV_WORD 36
#define EM_COPY_REGION 37
#define EM_DELETE_NEXT_WORD 38
#define EM_DELETE_OR_LIST 39
#define EM_EXCHANGE_MARK 40
#define EM_GOSMACS_TRASPOSE 41
#define EM_INC_SEARCH_NEXT 42
#define EM_INC_SEARCH_PREV 43
#define EM_KILL_LINE 44
#define EM_KILL_REGION 45
#define EM_LOWER_CASE 46
#define EM_META_NEXT 47
#define EM_NEXT_WORD 48
#define EM_SET_MARK 49
#define EM_TOGGLE_OVERWRITE 50
#define EM_UNIVERSAL_ARGUMENT 51
#define EM_UPPER_CASE 52
#define EM_YANK 53
// EM_UNDO at the end
#ifdef EL_USE_VI
# define VI_ADD 54
# define VI_ADD_AT_EOL 55
# define VI_CHANGE_CASE 56
# define VI_CHANGE_META 57
# define VI_CHANGE_TO_EOL 58
# define VI_COMMAND_MODE 59
# define VI_DELETE_META 60
# define VI_DELETE_PREV_CHAR 61
# define VI_END_WORD 62
# define VI_INSERT 63
# define VI_INSERT_AT_BOL 64
# define VI_KILL_LINE_PREV 65
# define VI_LIST_OR_EOF 66
# define VI_NEXT_CHAR 67
# define VI_NEXT_SPACE_WORD 68
# define VI_NEXT_WORD 69
# define VI_PASTE_NEXT 70
# define VI_PASTE_PREV 71
# define VI_PREV_CHAR 72
# define VI_PREV_SPACE_WORD 73
# define VI_PREV_WORD 74
# define VI_REPEAT_NEXT_CHAR 75
# define VI_REPEAT_PREV_CHAR 76
# define VI_REPEAT_SEARCH_NEXT 77
# define VI_REPEAT_SEARCH_PREV 78
# define VI_REPLACE_CHAR 79
# define VI_REPLACE_MODE 80
# define VI_SEARCH_NEXT 81
# define VI_SEARCH_PREV 82
# define VI_SUBSTITUTE_CHAR 83
# define VI_SUBSTITUTE_LINE 84
# define VI_TO_END_WORD 85
# define VI_TO_NEXT_CHAR 86
# define VI_TO_PREV_CHAR 87
# define VI_UNDO 88
# define VI_ZERO 89
#else
# define VI_ADD ED_UNASSIGNED
# define VI_ADD_AT_EOL ED_UNASSIGNED
# define VI_CHANGE_CASE ED_UNASSIGNED
# define VI_CHANGE_META ED_UNASSIGNED
# define VI_CHANGE_TO_EOL ED_UNASSIGNED
# define VI_COMMAND_MODE ED_UNASSIGNED
# define VI_DELETE_META ED_UNASSIGNED
# define VI_DELETE_PREV_CHAR ED_UNASSIGNED
# define VI_END_WORD ED_UNASSIGNED
# define VI_INSERT ED_UNASSIGNED
# define VI_INSERT_AT_BOL ED_UNASSIGNED
# define VI_KILL_LINE_PREV ED_UNASSIGNED
# define VI_LIST_OR_EOF ED_UNASSIGNED
# define VI_NEXT_CHAR ED_UNASSIGNED
# define VI_NEXT_SPACE_WORD ED_UNASSIGNED
# define VI_NEXT_WORD ED_UNASSIGNED
# define VI_PASTE_NEXT ED_UNASSIGNED
# define VI_PASTE_PREV ED_UNASSIGNED
# define VI_PREV_CHAR ED_UNASSIGNED
# define VI_PREV_SPACE_WORD ED_UNASSIGNED
# define VI_PREV_WORD ED_UNASSIGNED
# define VI_REPEAT_NEXT_CHAR ED_UNASSIGNED
# define VI_REPEAT_PREV_CHAR ED_UNASSIGNED
# define VI_REPEAT_SEARCH_NEXT ED_UNASSIGNED
# define VI_REPEAT_SEARCH_PREV ED_UNASSIGNED
# define VI_REPLACE_CHAR ED_UNASSIGNED
# define VI_REPLACE_MODE ED_UNASSIGNED
# define VI_SEARCH_NEXT ED_UNASSIGNED
# define VI_SEARCH_PREV ED_UNASSIGNED
# define VI_SUBSTITUTE_CHAR ED_UNASSIGNED
# define VI_SUBSTITUTE_LINE ED_UNASSIGNED
# define VI_TO_END_WORD ED_UNASSIGNED
# define VI_TO_NEXT_CHAR ED_UNASSIGNED
# define VI_TO_PREV_CHAR ED_UNASSIGNED
# define VI_UNDO ED_UNASSIGNED
# define VI_ZERO ED_UNASSIGNED
#endif
#define EM_UNDO 90
#define ED_REPLAY_HIST 91
#define EL_NUM_FCNS 92
typedef ElAction_t (*ElFunc_t)(EditLine_t*, int);
el_protected const ElFunc_t* func__get(void);
#endif /* _h_fcns_c */
