/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

#ifndef __G_WIN32_H__
#define __G_WIN32_H__

#include <glib/gtypes.h>

#ifdef G_PLATFORM_WIN32

/* Windows emulation stubs for common Unix functions
 */

G_BEGIN_DECLS

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#ifdef _MSC_VER
typedef int pid_t;
#endif

#ifdef G_OS_WIN32

/*
 * To get prototypes for the following POSIXish functions, you have to
 * include the indicated non-POSIX headers. The functions are defined
 * in OLDNAMES.LIB (MSVC) or -lmoldname-msvc (mingw32).
 *
 * getcwd: <direct.h> (MSVC), <io.h> (mingw32)
 * getpid: <process.h>
 * access: <io.h>
 * unlink: <stdio.h> or <io.h>
 * open, read, write, lseek, close: <io.h>
 * rmdir: <direct.h>
 * pipe: <direct.h>
 */

/* pipe is not in OLDNAMES.LIB or -lmoldname-msvc. */
#define pipe(phandles)	_pipe (phandles, 4096, _O_BINARY)

/* For some POSIX functions that are not provided by the MS runtime,
 * we provide emulators in glib, which are prefixed with g_win32_.
 */
#    define ftruncate(fd, size)	g_win32_ftruncate (fd, size)

/* -lmingw32 also has emulations for these, but we need our own
 * for MSVC anyhow, so we might aswell use them always.
 */
#    define opendir		g_win32_opendir
#    define readdir		g_win32_readdir
#    define rewinddir		g_win32_rewinddir
#    define closedir		g_win32_closedir
#    define NAME_MAX 255

struct dirent
{
  gchar  d_name[NAME_MAX + 1];
};

struct DIR
{
  gchar        *dir_name;
  gboolean 	just_opened;
  gulong    	find_file_handle;
  gpointer 	find_file_data;
  struct dirent readdir_result;
};
typedef struct DIR DIR;

/* emulation functions */
gint		g_win32_ftruncate	(gint		 f,
					 guint		 size);
DIR*		g_win32_opendir		(const gchar	*dirname);
struct dirent*	g_win32_readdir  	(DIR		*dir);
void		g_win32_rewinddir 	(DIR		*dir);
gint		g_win32_closedir  	(DIR		*dir);

#endif /* G_OS_WIN32 */

/* The MS setlocale uses locale names of the form "English_United
 * States.1252" etc. We want the Unixish standard form "en", "zh_TW"
 * etc. This function gets the current thread locale from Windows and
 * returns it as a string of the above form for use in forming file
 * names etc. The returned string should be deallocated with g_free().
 */
gchar* 		g_win32_getlocale  (void);

/* Translate a Win32 error code (as returned by GetLastError()) into
 * the corresponding message. The returned string should be deallocated
 * with g_free().
 */
gchar*          g_win32_error_message (gint error);

gchar*          g_win32_get_package_installation_directory (gchar *package,
							    gchar *dll_name);

gchar*          g_win32_get_package_installation_subdirectory (gchar *package,
							       gchar *dll_name,
							       gchar *subdir);

G_END_DECLS

#endif	 /* G_PLATFORM_WIN32 */

#endif /* __G_WIN32_H__ */
