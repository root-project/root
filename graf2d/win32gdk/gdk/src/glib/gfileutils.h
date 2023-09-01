/* gfileutils.h - File utility functions
 *
 *  Copyright 2000 Red Hat, Inc.
 *
 * GLib is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * GLib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GLib; see the file COPYING.LIB.  If not,
 * write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 *   Boston, MA 02111-1307, USA.
 */

#ifndef __G_FILEUTILS_H__
#define __G_FILEUTILS_H__

#include <glib/gerror.h>

G_BEGIN_DECLS

#define G_FILE_ERROR g_file_error_quark ()

typedef enum
{
  G_FILE_ERROR_EXIST,
  G_FILE_ERROR_ISDIR,
  G_FILE_ERROR_ACCES,
  G_FILE_ERROR_NAMETOOLONG,
  G_FILE_ERROR_NOENT,
  G_FILE_ERROR_NOTDIR,
  G_FILE_ERROR_NXIO,
  G_FILE_ERROR_NODEV,
  G_FILE_ERROR_ROFS,
  G_FILE_ERROR_TXTBSY,
  G_FILE_ERROR_FAULT,
  G_FILE_ERROR_LOOP,
  G_FILE_ERROR_NOSPC,
  G_FILE_ERROR_NOMEM,
  G_FILE_ERROR_MFILE,
  G_FILE_ERROR_NFILE,
  G_FILE_ERROR_BADF,
  G_FILE_ERROR_INVAL,
  G_FILE_ERROR_PIPE,
  G_FILE_ERROR_AGAIN,
  G_FILE_ERROR_INTR,
  G_FILE_ERROR_IO,
  G_FILE_ERROR_PERM,
  G_FILE_ERROR_FAILED
} GFileError;

/* For backward-compat reasons, these are synced to an old 
 * anonymous enum in libgnome. But don't use that enum
 * in new code.
 */
typedef enum
{
  G_FILE_TEST_IS_REGULAR    = 1 << 0,
  G_FILE_TEST_IS_SYMLINK    = 1 << 1,
  G_FILE_TEST_IS_DIR        = 1 << 2,
  G_FILE_TEST_IS_EXECUTABLE = 1 << 3,
  G_FILE_TEST_EXISTS        = 1 << 4
} GFileTest;

GQuark     g_file_error_quark      (void);
/* So other code can generate a GFileError */
GFileError g_file_error_from_errno (gint err_no);

gboolean g_file_test         (const gchar  *filename,
                              GFileTest     test);
gboolean g_file_get_contents (const gchar  *filename,
                              gchar       **contents,
                              gsize        *length,    
                              GError      **error);


/* Wrapper / workalike for mkstemp() */
int     g_mkstemp            (char         *tmpl);

/* Wrapper for g_mkstemp */
int     g_file_open_tmp      (const char   *tmpl,
			      char        **name_used,
			      GError      **error);

gchar *g_build_path     (const gchar *separator,
			 const gchar *first_element,
			 ...);
gchar *g_build_filename (const gchar *first_element,
			 ...);

G_END_DECLS

#endif /* __G_FILEUTILS_H__ */


