/* gfileutils.c - File utility functions
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

#include "config.h"

#include "glib.h"

#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

#ifdef G_OS_WIN32
#include <io.h>
#ifndef F_OK
#define	F_OK 0
#define	X_OK 1
#define	W_OK 2
#define	R_OK 4
#endif /* !F_OK */

#ifndef S_ISREG
#define S_ISREG(mode) ((mode)&_S_IFREG)
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) ((mode)&_S_IFDIR)
#endif

#endif /* G_OS_WIN32 */

#ifndef S_ISLNK
#define S_ISLNK(x) 0
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#include "glibintl.h"

/**
 * g_file_test:
 * @filename: a filename to test
 * @test: bitfield of #GFileTest flags
 * 
 * Returns TRUE if any of the tests in the bitfield @test are
 * TRUE. For example, (G_FILE_TEST_EXISTS | G_FILE_TEST_IS_DIR)
 * will return TRUE if the file exists; the check whether it's
 * a directory doesn't matter since the existence test is TRUE.
 * With the current set of available tests, there's no point
 * passing in more than one test at a time.
 *
 * Return value: whether a test was TRUE
 **/
gboolean
g_file_test (const gchar *filename,
             GFileTest    test)
{
  if (test & G_FILE_TEST_EXISTS)
    return (access (filename, F_OK) == 0);
  else if (test & G_FILE_TEST_IS_EXECUTABLE)
    return (access (filename, X_OK) == 0);
  else
    {
      struct stat s;
      
      if (stat (filename, &s) < 0)
        return FALSE;

      if ((test & G_FILE_TEST_IS_REGULAR) &&
          S_ISREG (s.st_mode))
        return TRUE;
      else if ((test & G_FILE_TEST_IS_DIR) &&
               S_ISDIR (s.st_mode))
        return TRUE;
      else if ((test & G_FILE_TEST_IS_SYMLINK) &&
               S_ISLNK (s.st_mode))
        return TRUE;
      else
        return FALSE;
    }
}

GQuark
g_file_error_quark (void)
{
  static GQuark q = 0;
  if (q == 0)
    q = g_quark_from_static_string ("g-file-error-quark");

  return q;
}

/**
 * g_file_error_from_errno:
 * @err_no: an "errno" value
 * 
 * Gets a #GFileError constant based on the passed-in errno.
 * For example, if you pass in EEXIST this function returns
 * #G_FILE_ERROR_EXIST. Unlike errno values, you can portably
 * assume that all #GFileError values will exist.
 *
 * Normally a #GFileError value goes into a #GError returned
 * from a function that manipulates files. So you would use
 * g_file_error_from_errno() when constructing a #GError.
 * 
 * Return value: #GFileError corresponding to the given errno
 **/
GFileError
g_file_error_from_errno (gint err_no)
{
  switch (err_no)
    {
#ifdef EEXIST
    case EEXIST:
      return G_FILE_ERROR_EXIST;
      break;
#endif

#ifdef EISDIR
    case EISDIR:
      return G_FILE_ERROR_ISDIR;
      break;
#endif

#ifdef EACCES
    case EACCES:
      return G_FILE_ERROR_ACCES;
      break;
#endif

#ifdef ENAMETOOLONG
    case ENAMETOOLONG:
      return G_FILE_ERROR_NAMETOOLONG;
      break;
#endif

#ifdef ENOENT
    case ENOENT:
      return G_FILE_ERROR_NOENT;
      break;
#endif

#ifdef ENOTDIR
    case ENOTDIR:
      return G_FILE_ERROR_NOTDIR;
      break;
#endif

#ifdef ENXIO
    case ENXIO:
      return G_FILE_ERROR_NXIO;
      break;
#endif

#ifdef ENODEV
    case ENODEV:
      return G_FILE_ERROR_NODEV;
      break;
#endif

#ifdef EROFS
    case EROFS:
      return G_FILE_ERROR_ROFS;
      break;
#endif

#ifdef ETXTBSY
    case ETXTBSY:
      return G_FILE_ERROR_TXTBSY;
      break;
#endif

#ifdef EFAULT
    case EFAULT:
      return G_FILE_ERROR_FAULT;
      break;
#endif

#ifdef ELOOP
    case ELOOP:
      return G_FILE_ERROR_LOOP;
      break;
#endif

#ifdef ENOSPC
    case ENOSPC:
      return G_FILE_ERROR_NOSPC;
      break;
#endif

#ifdef ENOMEM
    case ENOMEM:
      return G_FILE_ERROR_NOMEM;
      break;
#endif

#ifdef EMFILE
    case EMFILE:
      return G_FILE_ERROR_MFILE;
      break;
#endif

#ifdef ENFILE
    case ENFILE:
      return G_FILE_ERROR_NFILE;
      break;
#endif

#ifdef EBADF
    case EBADF:
      return G_FILE_ERROR_BADF;
      break;
#endif

#ifdef EINVAL
    case EINVAL:
      return G_FILE_ERROR_INVAL;
      break;
#endif

#ifdef EPIPE
    case EPIPE:
      return G_FILE_ERROR_PIPE;
      break;
#endif

#ifdef EAGAIN
    case EAGAIN:
      return G_FILE_ERROR_AGAIN;
      break;
#endif

#ifdef EINTR
    case EINTR:
      return G_FILE_ERROR_INTR;
      break;
#endif

#ifdef EIO
    case EIO:
      return G_FILE_ERROR_IO;
      break;
#endif

#ifdef EPERM
    case EPERM:
      return G_FILE_ERROR_PERM;
      break;
#endif
      
    default:
      return G_FILE_ERROR_FAILED;
      break;
    }
}

static gboolean
get_contents_stdio (const gchar *filename,
                    FILE        *f,
                    gchar      **contents,
                    gsize       *length, 
                    GError     **error)
{
  gchar buf[2048];
  size_t bytes;
  GString *str;

  g_assert (f != NULL);
  
  str = g_string_new ("");
  
  while (!feof (f))
    {
      bytes = fread (buf, 1, 2048, f);
      
      if (ferror (f))
        {
          g_set_error (error,
                       G_FILE_ERROR,
                       g_file_error_from_errno (errno),
                       _("Error reading file '%s': %s"),
                       filename, strerror (errno));

          g_string_free (str, TRUE);
	  fclose (f);
          
          return FALSE;
        }

      g_string_append_len (str, buf, bytes);
    }

  fclose (f);

  if (length)
    *length = str->len;
  
  *contents = g_string_free (str, FALSE);

  return TRUE;  
}

#ifndef G_OS_WIN32

static gboolean
get_contents_regfile (const gchar *filename,
                      struct stat *stat_buf,
                      gint         fd,
                      gchar      **contents,
                      gsize       *length,
                      GError     **error)
{
  gchar *buf;
  size_t bytes_read;
  size_t size;
      
  size = stat_buf->st_size;

  buf = g_new (gchar, size + 1);
      
  bytes_read = 0;
  while (bytes_read < size)
    {
      gssize rc;
          
      rc = read (fd, buf + bytes_read, size - bytes_read);

      if (rc < 0)
        {
          if (errno != EINTR) 
            {
              close (fd);

              g_free (buf);
                  
              g_set_error (error,
                           G_FILE_ERROR,
                           g_file_error_from_errno (errno),
                           _("Failed to read from file '%s': %s"),
                           filename, strerror (errno));

              return FALSE;
            }
        }
      else if (rc == 0)
        break;
      else
        bytes_read += rc;
    }
      
  buf[bytes_read] = '\0';

  if (length)
    *length = bytes_read;
  
  *contents = buf;

  return TRUE;
}

static gboolean
get_contents_posix (const gchar *filename,
                    gchar      **contents,
                    gsize       *length,
                    GError     **error)
{
  struct stat stat_buf;
  gint fd;
  
  /* O_BINARY useful on Cygwin */
  fd = open (filename, O_RDONLY|O_BINARY);

  if (fd < 0)
    {
      g_set_error (error,
                   G_FILE_ERROR,
                   g_file_error_from_errno (errno),
                   _("Failed to open file '%s': %s"),
                   filename, strerror (errno));

      return FALSE;
    }

  /* I don't think this will ever fail, aside from ENOMEM, but. */
  if (fstat (fd, &stat_buf) < 0)
    {
      close (fd);
      
      g_set_error (error,
                   G_FILE_ERROR,
                   g_file_error_from_errno (errno),
                   _("Failed to get attributes of file '%s': fstat() failed: %s"),
                   filename, strerror (errno));

      return FALSE;
    }

  if (stat_buf.st_size > 0 && S_ISREG (stat_buf.st_mode))
    {
      return get_contents_regfile (filename,
                                   &stat_buf,
                                   fd,
                                   contents,
                                   length,
                                   error);
    }
  else
    {
      FILE *f;

      f = fdopen (fd, "r");
      
      if (f == NULL)
        {
          g_set_error (error,
                       G_FILE_ERROR,
                       g_file_error_from_errno (errno),
                       _("Failed to open file '%s': fdopen() failed: %s"),
                       filename, strerror (errno));
          
          return FALSE;
        }
  
      return get_contents_stdio (filename, f, contents, length, error);
    }
}

#else  /* G_OS_WIN32 */

static gboolean
get_contents_win32 (const gchar *filename,
                    gchar      **contents,
                    gsize       *length,
                    GError     **error)
{
  FILE *f;

  /* I guess you want binary mode; maybe you want text sometimes? */
  f = fopen (filename, "rb");

  if (f == NULL)
    {
      g_set_error (error,
                   G_FILE_ERROR,
                   g_file_error_from_errno (errno),
                   _("Failed to open file '%s': %s"),
                   filename, strerror (errno));
      
      return FALSE;
    }
  
  return get_contents_stdio (filename, f, contents, length, error);
}

#endif

/**
 * g_file_get_contents:
 * @filename: a file to read contents from
 * @contents: location to store an allocated string
 * @length: location to store length in bytes of the contents
 * @error: return location for a #GError
 * 
 * Reads an entire file into allocated memory, with good error
 * checking. If @error is set, FALSE is returned, and @contents is set
 * to NULL. If TRUE is returned, @error will not be set, and @contents
 * will be set to the file contents.  The string stored in @contents
 * will be nul-terminated, so for text files you can pass NULL for the
 * @length argument.  The error domain is #G_FILE_ERROR. Possible
 * error codes are those in the #GFileError enumeration.
 *
 * FIXME currently crashes if the file is too big to fit in memory;
 * should probably use g_try_malloc() when we have that function.
 * 
 * Return value: TRUE on success, FALSE if error is set
 **/
gboolean
g_file_get_contents (const gchar *filename,
                     gchar      **contents,
                     gsize       *length,
                     GError     **error)
{  
  g_return_val_if_fail (filename != NULL, FALSE);
  g_return_val_if_fail (contents != NULL, FALSE);

  *contents = NULL;
  if (length)
    *length = 0;

#ifdef G_OS_WIN32
  return get_contents_win32 (filename, contents, length, error);
#else
  return get_contents_posix (filename, contents, length, error);
#endif
}

/*
 * mkstemp() implementation is from the GNU C library.
 * Copyright (C) 1991,92,93,94,95,96,97,98,99 Free Software Foundation, Inc.
 */
/**
 * g_mkstemp:
 * @tmpl: template filename
 *
 * Open a temporary file. See "man mkstemp" on most UNIX-like systems.
 * This is a portability wrapper, which simply calls mkstemp() on systems
 * that have it, and implements it in GLib otherwise.
 *
 * The parameter is a string that should match the rules for mktemp, i.e.
 * end in "XXXXXX". The X string will be modified to form the name
 * of a file that didn't exist.
 *
 * Return value: A file handle (as from open()) to the file
 * opened for reading and writing. The file is opened in binary mode
 * on platforms where there is a difference. The file handle should be
 * closed with close(). In case of errors, -1 is returned.
 */
int
g_mkstemp (char *tmpl)
{
#ifdef HAVE_MKSTEMP
  return mkstemp (tmpl);
#else
  int len;
  char *XXXXXX;
  int count, fd;
  static const char letters[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  static const int NLETTERS = sizeof (letters) - 1;
  glong value;
  GTimeVal tv;
  static int counter = 0;

  len = strlen (tmpl);
  if (len < 6 || strcmp (&tmpl[len - 6], "XXXXXX"))
    return -1;

  /* This is where the Xs start.  */
  XXXXXX = &tmpl[len - 6];

  /* Get some more or less random data.  */
  g_get_current_time (&tv);
  value = (tv.tv_usec ^ tv.tv_sec) + counter++;

  for (count = 0; count < 100; value += 7777, ++count)
    {
      glong v = value;

      /* Fill in the random bits.  */
      XXXXXX[0] = letters[v % NLETTERS];
      v /= NLETTERS;
      XXXXXX[1] = letters[v % NLETTERS];
      v /= NLETTERS;
      XXXXXX[2] = letters[v % NLETTERS];
      v /= NLETTERS;
      XXXXXX[3] = letters[v % NLETTERS];
      v /= NLETTERS;
      XXXXXX[4] = letters[v % NLETTERS];
      v /= NLETTERS;
      XXXXXX[5] = letters[v % NLETTERS];

      fd = open (tmpl, O_RDWR | O_CREAT | O_EXCL | O_BINARY, 0600);

      if (fd >= 0)
	return fd;
      else if (errno != EEXIST)
	/* Any other error will apply also to other names we might
	 *  try, and there are 2^32 or so of them, so give up now.
	 */
	return -1;
    }

  /* We got out of the loop because we ran out of combinations to try.  */
  return -1;
#endif
}

/**
 * g_file_open_tmp:
 * @tmpl: Template for file name, as in g_mkstemp, basename only
 * @name_used: location to store actual name used
 * @error: return location for a #GError
 *
 * Opens a file for writing in the preferred directory for temporary
 * files (as returned by g_get_tmp_dir()). 
 *
 * @tmpl should be a string ending with six 'X' characters, as the
 * parameter to g_mkstemp() (or mkstemp()). However, unlike these
 * functions, the template should only be a basename, no directory
 * components are allowed. If template is NULL, a default template is
 * used.
 *
 * Note that in contrast to g_mkstemp() (and mkstemp()) @tmpl is not
 * modified, and might thus be a read-only literal string.
 *
 * The actual name used is returned in @name_used if non-NULL. This
 * string should be freed with g_free when not needed any longer.
 *
 * Return value: A file handle (as from open()) to the file
 * opened for reading and writing. The file is opened in binary mode
 * on platforms where there is a difference. The file handle should be
 * closed with close(). In case of errors, -1 is returned and
 * @error will be set.
 **/
int
g_file_open_tmp (const char *tmpl,
		 char      **name_used,
		 GError    **error)
{
  int retval;
  const char *tmpdir;
  char *sep;
  char *fulltemplate;

  if (tmpl == NULL)
    tmpl = ".XXXXXX";

  if (strchr (tmpl, G_DIR_SEPARATOR)
#ifdef G_OS_WIN32
      || strchr (tmpl, '/')
#endif
				    )
    {
      g_set_error (error,
		   G_FILE_ERROR,
		   G_FILE_ERROR_FAILED,
		   _("Template '%s' invalid, should not contain a '%s'"),
		   tmpl, G_DIR_SEPARATOR_S);

      return -1;
    }
  
  if (strlen (tmpl) < 6 ||
      strcmp (tmpl + strlen (tmpl) - 6, "XXXXXX") != 0)
    {
      g_set_error (error,
		   G_FILE_ERROR,
		   G_FILE_ERROR_FAILED,
		   _("Template '%s' doesn't end with XXXXXX"),
		   tmpl);
      return -1;
    }

  tmpdir = g_get_tmp_dir ();

  if (tmpdir [strlen (tmpdir) - 1] == G_DIR_SEPARATOR)
    sep = "";
  else
    sep = G_DIR_SEPARATOR_S;

  fulltemplate = g_strconcat (tmpdir, sep, tmpl, NULL);

  retval = g_mkstemp (fulltemplate);

  if (retval == -1)
    {
      g_set_error (error,
		   G_FILE_ERROR,
		   g_file_error_from_errno (errno),
		   _("Failed to create file '%s': %s"),
		   fulltemplate, strerror (errno));
      g_free (fulltemplate);
      return -1;
    }

  if (name_used)
    *name_used = fulltemplate;
  else
    g_free (fulltemplate);

  return retval;
}

static gchar *
g_build_pathv (const gchar *separator,
	       const gchar *first_element,
	       va_list      args)
{
  GString *result;
  gint separator_len = strlen (separator);
  gboolean is_first = TRUE;
  const gchar *next_element;

  result = g_string_new (NULL);

  next_element = first_element;

  while (TRUE)
    {
      const gchar *element;
      const gchar *start;
      const gchar *end;

      if (next_element)
	{
	  element = next_element;
	  next_element = va_arg (args, gchar *);
	}
      else
	break;

      start = element;
      
      if (is_first)
	is_first = FALSE;
      else if (separator_len)
	{
	  while (start &&
		 strncmp (start, separator, separator_len) == 0)
	    start += separator_len;
	}

      end = start + strlen (start);
      
      if (next_element && separator_len)
	{
	  while (end > start + separator_len &&
		 strncmp (end - separator_len, separator, separator_len) == 0)
	    end -= separator_len;
	}

      if (end > start)
	{
	  if (result->len > 0)
	    g_string_append (result, separator);

	  g_string_append_len (result, start, end - start);
	}
    }
  
  return g_string_free (result, FALSE);
}

/**
 * g_build_path:
 * @separator: a string used to separator the elements of the path.
 * @first_element: the first element in the path
 * @Varargs: remaining elements in path
 * 
 * Create a path from a series of elements using @separator as the
 * separator between elements. At the boundary between two elements,
 * any trailing occurrences of separator in the first element, or
 * leading occurrences of separator in the second element are removed
 * and exactly one copy of the separator is inserted.
 * 
 * Return value: a newly allocated string that must be freed with g_free().
 **/
gchar *
g_build_path (const gchar *separator,
	      const gchar *first_element,
	      ...)
{
  gchar *str;
  va_list args;

  g_return_val_if_fail (separator != NULL, NULL);

  va_start (args, first_element);
  str = g_build_pathv (separator, first_element, args);
  va_end (args);

  return str;
}

/**
 * g_build_filename:
 * @first_element: the first element in the path
 * @Varargs: remaining elements in path
 * 
 * Create a filename from a series of elements using the correct
 * separator for filenames. This function behaves identically
 * to g_build_path (G_DIR_SEPARATOR_S, first_element, ....)
 *
 * No attempt is made to force the resulting filename to be an absolute
 * path. If the first element is a relative path, the result will
 * be a relative path. 
 * 
 * Return value: a newly allocated string that must be freed with g_free().
 **/
gchar *
g_build_filename (const gchar *first_element, 
		  ...)
{
  gchar *str;
  va_list args;

  va_start (args, first_element);
  str = g_build_pathv (G_DIR_SEPARATOR_S, first_element, args);
  va_end (args);

  return str;
}
