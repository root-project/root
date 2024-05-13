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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
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

/*
 * MT safe
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define _GNU_SOURCE		/* For stpcpy */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <ctype.h>		/* For tolower() */
#if !defined (HAVE_STRSIGNAL) || !defined(NO_SYS_SIGLIST_DECL)
#include <signal.h>
#endif
#include "glib.h"

#ifdef G_OS_WIN32
#include <windows.h>
#endif

/* do not include <unistd.h> in this place since it
 * inteferes with g_strsignal() on some OSes
 */

static const guint16 ascii_table_data[256] = {
  0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004,
  0x004, 0x104, 0x104, 0x004, 0x104, 0x104, 0x004, 0x004,
  0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004,
  0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004, 0x004,
  0x140, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0,
  0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0,
  0x459, 0x459, 0x459, 0x459, 0x459, 0x459, 0x459, 0x459,
  0x459, 0x459, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0,
  0x0d0, 0x653, 0x653, 0x653, 0x653, 0x653, 0x653, 0x253,
  0x253, 0x253, 0x253, 0x253, 0x253, 0x253, 0x253, 0x253,
  0x253, 0x253, 0x253, 0x253, 0x253, 0x253, 0x253, 0x253,
  0x253, 0x253, 0x253, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x0d0,
  0x0d0, 0x473, 0x473, 0x473, 0x473, 0x473, 0x473, 0x073,
  0x073, 0x073, 0x073, 0x073, 0x073, 0x073, 0x073, 0x073,
  0x073, 0x073, 0x073, 0x073, 0x073, 0x073, 0x073, 0x073,
  0x073, 0x073, 0x073, 0x0d0, 0x0d0, 0x0d0, 0x0d0, 0x004
  /* the upper 128 are all zeroes */
};

#if defined(G_PLATFORM_WIN32) && defined(__GNUC__)
__declspec(dllexport)
#endif
const guint16 * const g_ascii_table = ascii_table_data;

gchar*
g_strdup (const gchar *str)
{
  gchar *new_str;

  if (str)
    {
      new_str = g_new (char, strlen (str) + 1);
      strcpy (new_str, str);
    }
  else
    new_str = NULL;

  return new_str;
}

gpointer
g_memdup (gconstpointer mem,
	  guint         byte_size)
{
  gpointer new_mem;

  if (mem)
    {
      new_mem = g_malloc (byte_size);
      memcpy (new_mem, mem, byte_size);
    }
  else
    new_mem = NULL;

  return new_mem;
}

gchar*
g_strndup (const gchar *str,
	   gsize        n)    
{
  gchar *new_str;

  if (str)
    {
      new_str = g_new (gchar, n + 1);
      strncpy (new_str, str, n);
      new_str[n] = '\0';
    }
  else
    new_str = NULL;

  return new_str;
}

gchar*
g_strnfill (gsize length,     
	    gchar fill_char)
{
  register gchar *str, *s, *end;

  str = g_new (gchar, length + 1);
  s = str;
  end = str + length;
  while (s < end)
    *(s++) = fill_char;
  *s = 0;

  return str;
}

/**
 * g_stpcpy:
 * @dest: destination buffer
 * @src: source string
 * 
 * Copies a nul-terminated string into the dest buffer, include the
 * trailing nul, and return a pointer to the trailing nul byte.
 * This is useful for concatenating multiple strings together
 * without having to repeatedly scan for the end.
 * 
 * Return value: a pointer to trailing nul byte.
 **/
gchar *
g_stpcpy (gchar       *dest,
          const gchar *src)
{
#ifdef HAVE_STPCPY
  g_return_val_if_fail (dest != NULL, NULL);
  g_return_val_if_fail (src != NULL, NULL);
  return stpcpy (dest, src);
#else
  register gchar *d = dest;
  register const gchar *s = src;

  g_return_val_if_fail (dest != NULL, NULL);
  g_return_val_if_fail (src != NULL, NULL);
  do
    *d++ = *s;
  while (*s++ != '\0');

  return d - 1;
#endif
}

gchar*
g_strdup_vprintf (const gchar *format,
		  va_list      args1)
{
  gchar *buffer;
#ifdef HAVE_VASPRINTF
  vasprintf (&buffer, format, args1);
  if (g_mem_is_system_malloc ()) 
    {
      gchar *buffer1 = g_strdup (buffer);
      free (buffer);
      buffer = buffer1;
    }
#else
  va_list args2;

  G_VA_COPY (args2, args1);

  buffer = g_new (gchar, g_printf_string_upper_bound (format, args1));

  vsprintf (buffer, format, args2);
  va_end (args2);
#endif
  return buffer;
}

gchar*
g_strdup_printf (const gchar *format,
		 ...)
{
  gchar *buffer;
  va_list args;

  va_start (args, format);
  buffer = g_strdup_vprintf (format, args);
  va_end (args);

  return buffer;
}

gchar*
g_strconcat (const gchar *string1, ...)
{
  gsize	  l;     
  va_list args;
  gchar	  *s;
  gchar	  *concat;
  gchar   *ptr;

  g_return_val_if_fail (string1 != NULL, NULL);

  l = 1 + strlen (string1);
  va_start (args, string1);
  s = va_arg (args, gchar*);
  while (s)
    {
      l += strlen (s);
      s = va_arg (args, gchar*);
    }
  va_end (args);

  concat = g_new (gchar, l);
  ptr = concat;

  ptr = g_stpcpy (ptr, string1);
  va_start (args, string1);
  s = va_arg (args, gchar*);
  while (s)
    {
      ptr = g_stpcpy (ptr, s);
      s = va_arg (args, gchar*);
    }
  va_end (args);

  return concat;
}

gdouble
g_strtod (const gchar *nptr,
	  gchar **endptr)
{
  gchar *fail_pos_1;
  gchar *fail_pos_2;
  gdouble val_1;
  gdouble val_2 = 0;

  g_return_val_if_fail (nptr != NULL, 0);

  fail_pos_1 = NULL;
  fail_pos_2 = NULL;

  val_1 = strtod (nptr, &fail_pos_1);

  if (fail_pos_1 && fail_pos_1[0] != 0)
    {
      gchar *old_locale;

      old_locale = g_strdup (setlocale (LC_NUMERIC, NULL));
      setlocale (LC_NUMERIC, "C");
      val_2 = strtod (nptr, &fail_pos_2);
      setlocale (LC_NUMERIC, old_locale);
      g_free (old_locale);
    }

  if (!fail_pos_1 || fail_pos_1[0] == 0 || fail_pos_1 >= fail_pos_2)
    {
      if (endptr)
	*endptr = fail_pos_1;
      return val_1;
    }
  else
    {
      if (endptr)
	*endptr = fail_pos_2;
      return val_2;
    }
}

G_CONST_RETURN gchar*
g_strerror (gint errnum)
{
  static GStaticPrivate msg_private = G_STATIC_PRIVATE_INIT;
  char *msg;

  //
  // Removed a lot fo code as errors are not common on windows that actually would mean anythign here...
  //

  msg = g_static_private_get (&msg_private);
  if (!msg)
    {
      msg = g_new (gchar, 64);
      g_static_private_set (&msg_private, msg, g_free);
    }

  sprintf (msg, "unknown error (%d)", errnum);

  return msg;
}

G_CONST_RETURN gchar*
g_strsignal (gint signum)
{
  static GStaticPrivate msg_private = G_STATIC_PRIVATE_INIT;
  char *msg;

  //
  // A lot of code removed by GTW - this will be building only on win32, and signals are... not really the thing
  // here!
  //

  msg = g_static_private_get (&msg_private);
  if (!msg)
    {
      msg = g_new (gchar, 64);
      g_static_private_set (&msg_private, msg, g_free);
    }

  sprintf (msg, "unknown signal (%d)", signum);
  
  return msg;
}

/* Functions g_strlcpy and g_strlcat were originally developed by
 * Todd C. Miller <Todd.Miller@courtesan.com> to simplify writing secure code.
 * See ftp://ftp.openbsd.org/pub/OpenBSD/src/lib/libc/string/strlcpy.3
 * for more information.
 */

#ifdef HAVE_STRLCPY
/* Use the native ones, if available; they might be implemented in assembly */
gsize
g_strlcpy (gchar       *dest,
	   const gchar *src,
	   gsize        dest_size)
{
  g_return_val_if_fail (dest != NULL, 0);
  g_return_val_if_fail (src  != NULL, 0);
  
  return strlcpy (dest, src, dest_size);
}

gsize
g_strlcat (gchar       *dest,
	   const gchar *src,
	   gsize        dest_size)
{
  g_return_val_if_fail (dest != NULL, 0);
  g_return_val_if_fail (src  != NULL, 0);
  
  return strlcat (dest, src, dest_size);
}

#else /* ! HAVE_STRLCPY */
/* g_strlcpy
 *
 * Copy string src to buffer dest (of buffer size dest_size).  At most
 * dest_size-1 characters will be copied.  Always NUL terminates
 * (unless dest_size == 0).  This function does NOT allocate memory.
 * Unlike strncpy, this function doesn't pad dest (so it's often faster).
 * Returns size of attempted result, strlen(src),
 * so if retval >= dest_size, truncation occurred.
 */
gsize
g_strlcpy (gchar       *dest,
           const gchar *src,
           gsize        dest_size)
{
  register gchar *d = dest;
  register const gchar *s = src;
  register gsize n = dest_size;
  
  g_return_val_if_fail (dest != NULL, 0);
  g_return_val_if_fail (src  != NULL, 0);
  
  /* Copy as many bytes as will fit */
  if (n != 0 && --n != 0)
    do
      {
	register gchar c = *s++;
	
	*d++ = c;
	if (c == 0)
	  break;
      }
    while (--n != 0);
  
  /* If not enough room in dest, add NUL and traverse rest of src */
  if (n == 0)
    {
      if (dest_size != 0)
	*d = 0;
      while (*s++)
	;
    }
  
  return s - src - 1;  /* count does not include NUL */
}

/* g_strlcat
 *
 * Appends string src to buffer dest (of buffer size dest_size).
 * At most dest_size-1 characters will be copied.
 * Unlike strncat, dest_size is the full size of dest, not the space left over.
 * This function does NOT allocate memory.
 * This always NUL terminates (unless siz == 0 or there were no NUL characters
 * in the dest_size characters of dest to start with).
 * Returns size of attempted result, which is
 * MIN (dest_size, strlen (original dest)) + strlen (src),
 * so if retval >= dest_size, truncation occurred.
 */
gsize
g_strlcat (gchar       *dest,
           const gchar *src,
           gsize        dest_size)
{
  register gchar *d = dest;
  register const gchar *s = src;
  register gsize bytes_left = dest_size;
  gsize dlength;  /* Logically, MIN (strlen (d), dest_size) */
  
  g_return_val_if_fail (dest != NULL, 0);
  g_return_val_if_fail (src  != NULL, 0);
  
  /* Find the end of dst and adjust bytes left but don't go past end */
  while (*d != 0 && bytes_left-- != 0)
    d++;
  dlength = d - dest;
  bytes_left = dest_size - dlength;
  
  if (bytes_left == 0)
    return dlength + strlen (s);
  
  while (*s != 0)
    {
      if (bytes_left != 1)
	{
	  *d++ = *s;
	  bytes_left--;
	}
      s++;
    }
  *d = 0;
  
  return dlength + (s - src);  /* count does not include NUL */
}
#endif /* ! HAVE_STRLCPY */

/**
 * g_ascii_strdown:
 * @str: a string
 * @len: length of @str in bytes, or -1 if @str is nul-terminated.
 * 
 * Converts all upper case ASCII letters to lower case ASCII letters.
 * 
 * Return value: a newly allocated string, with all the upper case
 *               characters in @str converted to lower case, with
 *               semantics that exactly match g_ascii_tolower. (Note
 *               that this is unlike the old g_strdown, which modified
 *               the string in place.)
 **/
gchar*
g_ascii_strdown (const gchar *str,
		 gssize         len)
{
  gchar *result, *s;
  
  g_return_val_if_fail (str != NULL, NULL);

  if (len < 0)
    len = strlen (str);

  result = g_strndup (str, len);
  for (s = result; *s; s++)
    *s = g_ascii_tolower (*s);
  
  return result;
}

/**
 * g_ascii_strup:
 * @str: a string
 * @len: length of @str in bytes, or -1 if @str is nul-terminated.
 * 
 * Converts all lower case ASCII letters to upper case ASCII letters.
 * 
 * Return value: a newly allocated string, with all the lower case
 *               characters in @str converted to upper case, with
 *               semantics that exactly match g_ascii_toupper. (Note
 *               that this is unlike the old g_strup, which modified
 *               the string in place.)
 **/
gchar*
g_ascii_strup (const gchar *str,
	       gssize         len)
{
  gchar *result, *s;

  g_return_val_if_fail (str != NULL, NULL);

  if (len < 0)
    len = strlen (str);

  result = g_strndup (str, len);
  for (s = result; *s; s++)
    *s = g_ascii_toupper (*s);

  return result;
}

gchar*
g_strdown (gchar *string)
{
  register guchar *s;
  
  g_return_val_if_fail (string != NULL, NULL);
  
  s = (guchar *) string;
  
  while (*s)
    {
      if (isupper (*s))
	*s = tolower (*s);
      s++;
    }
  
  return (gchar *) string;
}

gchar*
g_strup (gchar *string)
{
  register guchar *s;

  g_return_val_if_fail (string != NULL, NULL);

  s = (guchar *) string;

  while (*s)
    {
      if (islower (*s))
	*s = toupper (*s);
      s++;
    }

  return (gchar *) string;
}

gchar*
g_strreverse (gchar *string)
{
  g_return_val_if_fail (string != NULL, NULL);

  if (*string)
    {
      register gchar *h, *t;

      h = string;
      t = string + strlen (string) - 1;

      while (h < t)
	{
	  register gchar c;

	  c = *h;
	  *h = *t;
	  h++;
	  *t = c;
	  t--;
	}
    }

  return string;
}

/**
 * g_ascii_tolower:
 * @c: any character
 * 
 * Convert a character to ASCII lower case.
 *
 * Unlike the standard C library tolower function, this only
 * recognizes standard ASCII letters and ignores the locale, returning
 * all non-ASCII characters unchanged, even if they are lower case
 * letters in a particular character set. Also unlike the standard
 * library function, this takes and returns a char, not an int, so
 * don't call it on EOF but no need to worry about casting to guchar
 * before passing a possibly non-ASCII character in.
 * 
 * Return value: the result of converting @c to lower case.
 *               If @c is not an ASCII upper case letter,
 *               @c is returned unchanged.
 **/
gchar
g_ascii_tolower (gchar c)
{
  return g_ascii_isupper (c) ? c - 'A' + 'a' : c;
}

/**
 * g_ascii_toupper:
 * @c: any character
 * 
 * Convert a character to ASCII upper case.
 *
 * Unlike the standard C library toupper function, this only
 * recognizes standard ASCII letters and ignores the locale, returning
 * all non-ASCII characters unchanged, even if they are upper case
 * letters in a particular character set. Also unlike the standard
 * library function, this takes and returns a char, not an int, so
 * don't call it on EOF but no need to worry about casting to guchar
 * before passing a possibly non-ASCII character in.
 * 
 * Return value: the result of converting @c to upper case.
 *               If @c is not an ASCII lower case letter,
 *               @c is returned unchanged.
 **/
gchar
g_ascii_toupper (gchar c)
{
  return g_ascii_islower (c) ? c - 'a' + 'A' : c;
}

/**
 * g_ascii_digit_value:
 * @c: an ASCII character
 *
 * Determines the numeric value of a character as a decimal
 * digit. Differs from g_unichar_digit_value because it takes
 * a char, so there's no worry about sign extension if characters
 * are signed.
 *
 * Return value: If @c is a decimal digit (according to
 * `g_ascii_isdigit'), its numeric value. Otherwise, -1.
 **/
int
g_ascii_digit_value (gchar c)
{
  if (g_ascii_isdigit (c))
    return c - '0';
  return -1;
}

/**
 * g_ascii_xdigit_value:
 * @c: an ASCII character
 *
 * Determines the numeric value of a character as a hexidecimal
 * digit. Differs from g_unichar_xdigit_value because it takes
 * a char, so there's no worry about sign extension if characters
 * are signed.
 *
 * Return value: If @c is a hex digit (according to
 * `g_ascii_isxdigit'), its numeric value. Otherwise, -1.
 **/
int
g_ascii_xdigit_value (gchar c)
{
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  return g_ascii_digit_value (c);
}

/**
 * g_ascii_strcasecmp:
 * @s1: string to compare with @s2
 * @s2: string to compare with @s1
 * 
 * Compare two strings, ignoring the case of ASCII characters.
 *
 * Unlike the BSD strcasecmp function, this only recognizes standard
 * ASCII letters and ignores the locale, treating all non-ASCII
 * characters as if they are not letters.
 * 
 * Return value: an integer less than, equal to, or greater than
 *               zero if @s1 is found, respectively, to be less than,
 *               to match, or to be greater than @s2.
 **/
gint
g_ascii_strcasecmp (const gchar *s1,
		    const gchar *s2)
{
  gint c1, c2;

  g_return_val_if_fail (s1 != NULL, 0);
  g_return_val_if_fail (s2 != NULL, 0);

  while (*s1 && *s2)
    {
      c1 = (gint)(guchar) g_ascii_tolower (*s1);
      c2 = (gint)(guchar) g_ascii_tolower (*s2);
      if (c1 != c2)
	return (c1 - c2);
      s1++; s2++;
    }

  return (((gint)(guchar) *s1) - ((gint)(guchar) *s2));
}

/**
 * g_ascii_strncasecmp:
 * @s1: string to compare with @s2
 * @s2: string to compare with @s1
 * @n:  number of characters to compare
 * 
 * Compare @s1 and @s2, ignoring the case of ASCII characters and any
 * characters after the first @n in each string.
 *
 * Unlike the BSD strcasecmp function, this only recognizes standard
 * ASCII letters and ignores the locale, treating all non-ASCII
 * characters as if they are not letters.
 * 
 * Return value: an integer less than, equal to, or greater than zero
 *               if the first @n bytes of @s1 is found, respectively,
 *               to be less than, to match, or to be greater than the
 *               first @n bytes of @s2.
 **/
gint
g_ascii_strncasecmp (const gchar *s1,
		     const gchar *s2,
		     guint n)
{
  gint c1, c2;

  g_return_val_if_fail (s1 != NULL, 0);
  g_return_val_if_fail (s2 != NULL, 0);

  while (n && *s1 && *s2)
    {
      n -= 1;
      c1 = (gint)(guchar) g_ascii_tolower (*s1);
      c2 = (gint)(guchar) g_ascii_tolower (*s2);
      if (c1 != c2)
	return (c1 - c2);
      s1++; s2++;
    }

  if (n)
    return (((gint) (guchar) *s1) - ((gint) (guchar) *s2));
  else
    return 0;
}

gint
g_strcasecmp (const gchar *s1,
	      const gchar *s2)
{
#ifdef HAVE_STRCASECMP
  g_return_val_if_fail (s1 != NULL, 0);
  g_return_val_if_fail (s2 != NULL, 0);

  return strcasecmp (s1, s2);
#else
  gint c1, c2;

  g_return_val_if_fail (s1 != NULL, 0);
  g_return_val_if_fail (s2 != NULL, 0);

  while (*s1 && *s2)
    {
      /* According to A. Cox, some platforms have islower's that
       * don't work right on non-uppercase
       */
      c1 = isupper ((guchar)*s1) ? tolower ((guchar)*s1) : *s1;
      c2 = isupper ((guchar)*s2) ? tolower ((guchar)*s2) : *s2;
      if (c1 != c2)
	return (c1 - c2);
      s1++; s2++;
    }

  return (((gint)(guchar) *s1) - ((gint)(guchar) *s2));
#endif
}

gint
g_strncasecmp (const gchar *s1,
	       const gchar *s2,
	       gssize n)     
{
#ifdef HAVE_STRNCASECMP
  return strncasecmp (s1, s2, n);
#else
  gint c1, c2;

  g_return_val_if_fail (s1 != NULL, 0);
  g_return_val_if_fail (s2 != NULL, 0);

  while (n && *s1 && *s2)
    {
      n -= 1;
      /* According to A. Cox, some platforms have islower's that
       * don't work right on non-uppercase
       */
      c1 = isupper ((guchar)*s1) ? tolower ((guchar)*s1) : *s1;
      c2 = isupper ((guchar)*s2) ? tolower ((guchar)*s2) : *s2;
      if (c1 != c2)
	return (c1 - c2);
      s1++; s2++;
    }

  if (n)
    return (((gint) (guchar) *s1) - ((gint) (guchar) *s2));
  else
    return 0;
#endif
}

gchar*
g_strdelimit (gchar	  *string,
	      const gchar *delimiters,
	      gchar	   new_delim)
{
  register gchar *c;

  g_return_val_if_fail (string != NULL, NULL);

  if (!delimiters)
    delimiters = G_STR_DELIMITERS;

  for (c = string; *c; c++)
    {
      if (strchr (delimiters, *c))
	*c = new_delim;
    }

  return string;
}

gchar*
g_strcanon (gchar       *string,
	    const gchar *valid_chars,
	    gchar        substitutor)
{
  register gchar *c;

  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (valid_chars != NULL, NULL);

  for (c = string; *c; c++)
    {
      if (!strchr (valid_chars, *c))
	*c = substitutor;
    }

  return string;
}

gchar*
g_strcompress (const gchar *source)
{
  const gchar *p = source, *octal;
  gchar *dest = g_malloc (strlen (source) + 1);
  gchar *q = dest;
  
  while (*p)
    {
      if (*p == '\\')
	{
	  p++;
	  switch (*p)
	    {
	    case '0':  case '1':  case '2':  case '3':  case '4':
	    case '5':  case '6':  case '7':
	      *q = 0;
	      octal = p;
	      while ((p < octal + 3) && (*p >= '0') && (*p <= '7'))
		{
		  *q = (*q * 8) + (*p - '0');
		  p++;
		}
	      q++;
	      p--;
	      break;
	    case 'b':
	      *q++ = '\b';
	      break;
	    case 'f':
	      *q++ = '\f';
	      break;
	    case 'n':
	      *q++ = '\n';
	      break;
	    case 'r':
	      *q++ = '\r';
	      break;
	    case 't':
	      *q++ = '\t';
	      break;
	    default:		/* Also handles \" and \\ */
	      *q++ = *p;
	      break;
	    }
	}
      else
	*q++ = *p;
      p++;
    }
  *q = 0;
  
  return dest;
}

gchar *
g_strescape (const gchar *source,
	     const gchar *exceptions)
{
  const guchar *p;
  gchar *dest;
  gchar *q;
  guchar excmap[256];
  
  g_return_val_if_fail (source != NULL, NULL);

  p = (guchar *) source;
  /* Each source byte needs maximally four destination chars (\777) */
  q = dest = g_malloc (strlen (source) * 4 + 1);

  memset (excmap, 0, 256);
  if (exceptions)
    {
      guchar *e = (guchar *) exceptions;

      while (*e)
	{
	  excmap[*e] = 1;
	  e++;
	}
    }

  while (*p)
    {
      if (excmap[*p])
	*q++ = *p;
      else
	{
	  switch (*p)
	    {
	    case '\b':
	      *q++ = '\\';
	      *q++ = 'b';
	      break;
	    case '\f':
	      *q++ = '\\';
	      *q++ = 'f';
	      break;
	    case '\n':
	      *q++ = '\\';
	      *q++ = 'n';
	      break;
	    case '\r':
	      *q++ = '\\';
	      *q++ = 'r';
	      break;
	    case '\t':
	      *q++ = '\\';
	      *q++ = 't';
	      break;
	    case '\\':
	      *q++ = '\\';
	      *q++ = '\\';
	      break;
	    case '"':
	      *q++ = '\\';
	      *q++ = '"';
	      break;
	    default:
	      if ((*p < ' ') || (*p >= 0177))
		{
		  *q++ = '\\';
		  *q++ = '0' + (((*p) >> 6) & 07);
		  *q++ = '0' + (((*p) >> 3) & 07);
		  *q++ = '0' + ((*p) & 07);
		}
	      else
		*q++ = *p;
	      break;
	    }
	}
      p++;
    }
  *q = 0;
  return dest;
}

gchar*
g_strchug (gchar *string)
{
  guchar *start;

  g_return_val_if_fail (string != NULL, NULL);

  for (start = (guchar*) string; *start && g_ascii_isspace (*start); start++)
    ;

  g_memmove (string, start, strlen ((gchar *) start) + 1);

  return string;
}

gchar*
g_strchomp (gchar *string)
{
  gchar *s;

  g_return_val_if_fail (string != NULL, NULL);

  if (!*string)
    return string;

  for (s = string + strlen (string) - 1; s >= string && g_ascii_isspace ((guchar)*s); 
       s--)
    *s = '\0';

  return string;
}

/**
 * g_strsplit:
 * @string: a string to split.
 * @delimiter: a string which specifies the places at which to split the string.
 *     The delimiter is not included in any of the resulting strings, unless
 *     max_tokens is reached.
 * @max_tokens: the maximum number of pieces to split @string into. If this is
 *              less than 1, the string is split completely.
 * 
 * Splits a string into a maximum of @max_tokens pieces, using the given
 * @delimiter. If @max_tokens is reached, the remainder of @string is appended
 * to the last token. 
 *
 * As a special case, the result of splitting the empty string "" is an empty
 * vector, not a vector containing a single string. The reason for this
 * special case is that being able to represent a empty vector is typically
 * more useful than consistent handling of empty elements. If you do need
 * to represent empty elements, you'll need to check for the empty string
 * before calling g_strsplit().
 * 
 * Return value: a newly-allocated %NULL-terminated array of strings. Use g_strfreev()
 *    to free it.
 **/
gchar**
g_strsplit (const gchar *string,
	    const gchar *delimiter,
	    gint         max_tokens)
{
  GSList *string_list = NULL, *slist;
  gchar **str_array, *s;
  guint n = 0;
  const gchar *remainder;

  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (delimiter != NULL, NULL);
  g_return_val_if_fail (delimiter[0] != '\0', NULL);

  if (max_tokens < 1)
    max_tokens = G_MAXINT;
  else
    --max_tokens;

  remainder = string;
  s = strstr (remainder, delimiter);
  if (s)
    {
      gsize delimiter_len = strlen (delimiter);   

      do
	{
	  gsize len;     
	  gchar *new_string;

	  len = s - remainder;
	  new_string = g_new (gchar, len + 1);
	  strncpy (new_string, remainder, len);
	  new_string[len] = 0;
	  string_list = g_slist_prepend (string_list, new_string);
	  n++;
	  remainder = s + delimiter_len;
	  s = strstr (remainder, delimiter);
	}
      while (--max_tokens && s);
    }
  if (*string)
    {
      n++;
      string_list = g_slist_prepend (string_list, g_strdup (remainder));
    }

  str_array = g_new (gchar*, n + 1);

  str_array[n--] = NULL;
  for (slist = string_list; slist; slist = slist->next)
    str_array[n--] = slist->data;

  g_slist_free (string_list);

  return str_array;
}

void
g_strfreev (gchar **str_array)
{
  if (str_array)
    {
      int i;

      for(i = 0; str_array[i] != NULL; i++)
	g_free(str_array[i]);

      g_free (str_array);
    }
}

/**
 * g_strdupv:
 * @str_array: %NULL-terminated array of strings
 * 
 * Copies %NULL-terminated array of strings. The copy is a deep copy;
 * the new array should be freed by first freeing each string, then
 * the array itself. g_strfreev() does this for you. If called
 * on a %NULL value, g_strdupv() simply returns %NULL.
 * 
 * Return value: a new %NULL-terminated array of strings
 **/
gchar**
g_strdupv (gchar **str_array)
{
  if (str_array)
    {
      gint i;
      gchar **retval;

      i = 0;
      while (str_array[i])
        ++i;
          
      retval = g_new (gchar*, i + 1);

      i = 0;
      while (str_array[i])
        {
          retval[i] = g_strdup (str_array[i]);
          ++i;
        }
      retval[i] = NULL;

      return retval;
    }
  else
    return NULL;
}

gchar*
g_strjoinv (const gchar  *separator,
	    gchar       **str_array)
{
  gchar *string;
  gchar *ptr;

  g_return_val_if_fail (str_array != NULL, NULL);

  if (separator == NULL)
    separator = "";

  if (*str_array)
    {
      gint i;
      gsize len;
      gsize separator_len;     

      separator_len = strlen (separator);
      /* First part, getting length */
      len = 1 + strlen (str_array[0]);
      for (i = 1; str_array[i] != NULL; i++)
        len += strlen (str_array[i]);
      len += separator_len * (i - 1);

      /* Second part, building string */
      string = g_new (gchar, len);
      ptr = g_stpcpy (string, *str_array);
      for (i = 1; str_array[i] != NULL; i++)
	{
          ptr = g_stpcpy (ptr, separator);
          ptr = g_stpcpy (ptr, str_array[i]);
	}
      }
  else
    string = g_strdup ("");

  return string;
}

gchar*
g_strjoin (const gchar  *separator,
	   ...)
{
  gchar *string, *s;
  va_list args;
  gsize len;               
  gsize separator_len;     
  gchar *ptr;

  if (separator == NULL)
    separator = "";

  separator_len = strlen (separator);

  va_start (args, separator);

  s = va_arg (args, gchar*);

  if (s)
    {
      /* First part, getting length */
      len = 1 + strlen (s);

      s = va_arg (args, gchar*);
      while (s)
	{
	  len += separator_len + strlen (s);
	  s = va_arg (args, gchar*);
	}
      va_end (args);

      /* Second part, building string */
      string = g_new (gchar, len);

      va_start (args, separator);

      s = va_arg (args, gchar*);
      ptr = g_stpcpy (string, s);

      s = va_arg (args, gchar*);
      while (s)
	{
	  ptr = g_stpcpy (ptr, separator);
          ptr = g_stpcpy (ptr, s);
	  s = va_arg (args, gchar*);
	}
    }
  else
    string = g_strdup ("");

  va_end (args);

  return string;
}


/**
 * g_strstr_len:
 * @haystack: a string
 * @haystack_len: The maximum length of haystack
 * @needle: The string to search for.
 *
 * Searches the string haystack for the first occurrence
 * of the string needle, limiting the length of the search
 * to haystack_len. 
 *
 * Return value: A pointer to the found occurrence, or
 * NULL if not found.
 **/
gchar *
g_strstr_len (const gchar *haystack,
	      gssize       haystack_len,
	      const gchar *needle)
{
  g_return_val_if_fail (haystack != NULL, NULL);
  g_return_val_if_fail (needle != NULL, NULL);
  
  if (haystack_len < 0)
    return strstr (haystack, needle);
  else
    {
      const gchar *p = haystack;
      gsize needle_len = strlen (needle);
      const gchar *end;
      gsize i;

      if (needle_len == 0)
	return (gchar *)haystack;

      if (haystack_len < needle_len)
	return NULL;
      
      end = haystack + haystack_len - needle_len;
      
      while (*p && p <= end)
	{
	  for (i = 0; i < needle_len; i++)
	    if (p[i] != needle[i])
	      goto next;
	  
	  return (gchar *)p;
	  
	next:
	  p++;
	}
      
      return NULL;
    }
}

/**
 * g_strrstr_len:
 * @haystack: a nul-terminated string
 * @needle: The nul-terminated string to search for.
 *
 * Searches the string haystack for the last occurrence
 * of the string needle.
 *
 * Return value: A pointer to the found occurrence, or
 * NULL if not found.
 **/
gchar *
g_strrstr (const gchar *haystack,
	   const gchar *needle)
{
  gsize i;
  gsize needle_len;
  gsize haystack_len;
  const gchar *p;
      
  g_return_val_if_fail (haystack != NULL, NULL);
  g_return_val_if_fail (needle != NULL, NULL);

  needle_len = strlen (needle);
  haystack_len = strlen (haystack);

  if (needle_len == 0)
    return (gchar *)haystack;

  if (haystack_len < needle_len)
    return NULL;
  
  p = haystack + haystack_len - needle_len;

  while (p >= haystack)
    {
      for (i = 0; i < needle_len; i++)
	if (p[i] != needle[i])
	  goto next;
      
      return (gchar *)p;
      
    next:
      p--;
    }
  
  return NULL;
}

/**
 * g_strrstr_len:
 * @haystack: a nul-terminated string
 * @haystack_len: The maximum length of haystack
 * @needle: The nul-terminated string to search for.
 *
 * Searches the string haystack for the last occurrence
 * of the string needle, limiting the length of the search
 * to haystack_len. 
 *
 * Return value: A pointer to the found occurrence, or
 * NULL if not found.
 **/
gchar *
g_strrstr_len (const gchar *haystack,
	       gssize        haystack_len,
	       const gchar *needle)
{
  g_return_val_if_fail (haystack != NULL, NULL);
  g_return_val_if_fail (needle != NULL, NULL);
  
  if (haystack_len < 0)
    return g_strrstr (haystack, needle);
  else
    {
      gsize needle_len = strlen (needle);
      const gchar *haystack_max = haystack + haystack_len;
      const gchar *p = haystack;
      gsize i;

      while (p < haystack_max && *p)
	p++;

      if (p < haystack + needle_len)
	return NULL;
	
      p -= needle_len;

      while (p >= haystack)
	{
	  for (i = 0; i < needle_len; i++)
	    if (p[i] != needle[i])
	      goto next;
	  
	  return (gchar *)p;
	  
	next:
	  p--;
	}

      return NULL;
    }
}


