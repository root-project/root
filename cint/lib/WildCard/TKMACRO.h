/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * parameter information file tkmacro.h
 ************************************************************************
 * Description:
 *  Constant macro and function macro to be exposed to C/C++ interpreter.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef _TK
#define _TK

#define TK_MAJOR_VERSION 4
#define TK_MINOR_VERSION 0

#define TK_ARGV_CONSTANT		15
#define TK_ARGV_INT			16
#define TK_ARGV_STRING			17
#define TK_ARGV_UID			18
#define TK_ARGV_REST			19
#define TK_ARGV_FLOAT			20
#define TK_ARGV_FUNC			21
#define TK_ARGV_GENFUNC			22
#define TK_ARGV_HELP			23
#define TK_ARGV_CONST_OPTION		24
#define TK_ARGV_OPTION_VALUE		25
#define TK_ARGV_OPTION_NAME_VALUE	26
#define TK_ARGV_END			27

#define TK_ARGV_NO_DEFAULTS		0x1
#define TK_ARGV_NO_LEFTOVERS		0x2
#define TK_ARGV_NO_ABBREV		0x4
#define TK_ARGV_DONT_SKIP_FIRST_ARG	0x8

#define TK_CONFIG_BOOLEAN	1
#define TK_CONFIG_INT		2
#define TK_CONFIG_DOUBLE	3
#define TK_CONFIG_STRING	4
#define TK_CONFIG_UID		5
#define TK_CONFIG_COLOR		6
#define TK_CONFIG_FONT		7
#define TK_CONFIG_BITMAP	8
#define TK_CONFIG_BORDER	9
#define TK_CONFIG_RELIEF	10
#define TK_CONFIG_CURSOR	11
#define TK_CONFIG_ACTIVE_CURSOR	12
#define TK_CONFIG_JUSTIFY	13
#define TK_CONFIG_ANCHOR	14
#define TK_CONFIG_SYNONYM	15
#define TK_CONFIG_CAP_STYLE	16
#define TK_CONFIG_JOIN_STYLE	17
#define TK_CONFIG_PIXELS	18
#define TK_CONFIG_MM		19
#define TK_CONFIG_WINDOW	20
#define TK_CONFIG_CUSTOM	21
#define TK_CONFIG_END		22

/*  NO WAY TO HANDLE THIS  MACRO
#define Tk_Offset(type, field) ((int) offsetof(type, field))
*/

#define TK_CONFIG_ARGV_ONLY	1

#define TK_CONFIG_COLOR_ONLY		1
#define TK_CONFIG_MONO_ONLY		2
#define TK_CONFIG_NULL_OK		4
#define TK_CONFIG_DONT_SET_DEFAULT	8
#define TK_CONFIG_OPTION_SPECIFIED	0x10
#define TK_CONFIG_USER_BIT		0x100

#define TK_READABLE	1
#define TK_WRITABLE	2
#define TK_EXCEPTION	4

#define TK_FILE_HANDLED -1

#define TK_DONT_WAIT		1
#define TK_X_EVENTS		2
#define TK_FILE_EVENTS		4
#define TK_TIMER_EVENTS		8
#define TK_IDLE_EVENTS		0x10
#define TK_ALL_EVENTS		0x1e

#define TK_WIDGET_DEFAULT_PRIO	20
#define TK_STARTUP_FILE_PRIO	40
#define TK_USER_DEFAULT_PRIO	60
#define TK_INTERACTIVE_PRIO	80
#define TK_MAX_PRIO		100

#define TK_RELIEF_RAISED	1
#define TK_RELIEF_FLAT		2
#define TK_RELIEF_SUNKEN	4
#define TK_RELIEF_GROOVE	8
#define TK_RELIEF_RIDGE		16

#define TK_3D_FLAT_GC		1
#define TK_3D_LIGHT_GC		2
#define TK_3D_DARK_GC		3

#define TK_NOTIFY_SHARE		20

#define TK_SCROLL_MOVETO	1
#define TK_SCROLL_PAGES		2
#define TK_SCROLL_UNITS		3
#define TK_SCROLL_ERROR		4

Display* Tk_Display(Tk_Window tkwin);
int Tk_ScreenNumber(Tk_Window tkwin);
int Tk_Screen(Tk_Window tkwin);
int Tk_Depth(Tk_Window tkwin);
Visual* Tk_Visual(Tk_Window tkwin);
Window Tk_WindowId(Tk_Window tkwin);
char* Tk_PathName(Tk_Window tkwin);
Tk_Uid Tk_Name(Tk_Window tkwin);
Tk_Uid Tk_Class(Tk_Window tkwin);
int Tk_X(Tk_Window tkwin);
int Tk_Y(Tk_Window tkwin);
int Tk_Width(Tk_Window tkwin);
int Tk_Height(Tk_Window tkwin);
XWindowChanges* Tk_Changes(Tk_Window tkwin);
XSetWindowAttributes* Tk_Attributes(Tk_Window tkwin);
int Tk_IsMapped(Tk_Window tkwin);
int Tk_IsTopLevel(Tk_Window tkwin);
int Tk_ReqWidth(Tk_Window tkwin);
int Tk_ReqHeight(Tk_Window tkwin);
int Tk_InternalBorderWidth(Tk_Window tkwin);
Tk_Window Tk_Parent(Tk_Window tkwin);
Colormap Tk_Colormap(Tk_Window tkwin);

#define TK_MAPPED		1
#define TK_TOP_LEVEL		2
#define TK_ALREADY_DEAD		4
#define TK_NEED_CONFIG_NOTIFY	8
#define TK_GRAB_FLAG		0x10
#define TK_CHECKED_IC		0x20
#define TK_PARENT_DESTROYED	0x40

#define TK_TAG_SPACE 3

#endif /* _TK */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
