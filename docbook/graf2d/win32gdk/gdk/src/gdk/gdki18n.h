/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

#ifndef __GDK_I18N_H__
#define __GDK_I18N_H__

/* GDK uses "glib". (And so does GTK).
 */
#include <glib.h>
#include <gdkconfig.h>

/* international string support */

#include <stdlib.h>

#if !defined(GDK_HAVE_BROKEN_WCTYPE) && (defined(GDK_HAVE_WCTYPE_H) || defined(GDK_HAVE_WCHAR_H)) && !defined(X_LOCALE)
#  ifdef GDK_HAVE_WCTYPE_H
#    include <wctype.h>
#  else
#    ifdef GDK_HAVE_WCHAR_H
#      include <wchar.h>
#    endif
#  endif
#  define gdk_iswalnum(c) iswalnum(c)
#  define gdk_iswspace(c) iswspace(c)
#else
#  include <ctype.h>
#  define gdk_iswalnum(c) ((wchar_t)(c) <= 0xFF && isalnum(c))
#  define gdk_iswspace(c) ((wchar_t)(c) <= 0xFF && isspace(c))
#endif

#endif                          /* __GDK_I18N_H__ */
