/* @(#)root/clib:$Id$ */
/* Author: */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Defs for interface to demanglers.
   Copyright 1992 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */


#if !defined (DEMANGLE_H)
#define DEMANGLE_H

/* Options passed to cplus_demangle (in 2nd parameter). */

#define DMGL_NO_OPTS 0   /* For readability... */
#define DMGL_PARAMS  (1 << 0)  /* Include function args */
#define DMGL_ANSI  (1 << 1)  /* Include const, volatile, etc */

#define DMGL_AUTO  (1 << 8)
#define DMGL_GNU  (1 << 9)
#define DMGL_LUCID  (1 << 10)
#define DMGL_ARM  (1 << 11)
/* If none of these are set, use 'current_demangling_style' as the default. */
#define DMGL_STYLE_MASK (DMGL_AUTO|DMGL_GNU|DMGL_LUCID|DMGL_ARM)

/* Define string names for the various demangling styles. */

#define AUTO_DEMANGLING_STYLE_STRING  "auto"
#define GNU_DEMANGLING_STYLE_STRING  "gnu"
#define LUCID_DEMANGLING_STYLE_STRING  "lucid"
#define ARM_DEMANGLING_STYLE_STRING  "arm"

/* Some macros to test what demangling style is active. */

#define CURRENT_DEMANGLING_STYLE current_demangling_style
#define AUTO_DEMANGLING (((int) CURRENT_DEMANGLING_STYLE) & DMGL_AUTO)
#define GNU_DEMANGLING (((int) CURRENT_DEMANGLING_STYLE) & DMGL_GNU)
#define LUCID_DEMANGLING (((int) CURRENT_DEMANGLING_STYLE) & DMGL_LUCID)
#define ARM_DEMANGLING (CURRENT_DEMANGLING_STYLE & DMGL_ARM)

#ifdef __cplusplus
extern "C" {
#endif

/* Enumeration of possible demangling styles.

   Lucid and ARM styles are still kept logically distinct, even though
   they now both behave identically.  The resulting style is actual the
   union of both.  I.E. either style recognizes both "__pt__" and "__rf__"
   for operator "->", even though the first is lucid style and the second
   is ARM style. (FIXME?) */

extern enum demangling_styles
{
  unknown_demangling = 0,
  auto_demangling = DMGL_AUTO,
  gnu_demangling = DMGL_GNU,
  lucid_demangling = DMGL_LUCID,
  arm_demangling = DMGL_ARM
} current_demangling_style;


extern char *
cplus_demangle (const char *mangled, int options);

extern int
cplus_demangle_opname (char *opname, char *result, int options);

/* Note: This sets global state.  FIXME if you care about multi-threading. */

extern void
set_cplus_marker_for_demangling (int ch);

#ifdef __cplusplus
}
#endif

#endif  /* DEMANGLE_H */
