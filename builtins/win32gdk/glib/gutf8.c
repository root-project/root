/* gutf8.c - Operations on UTF-8 strings.
 *
 * Copyright (C) 1999 Tom Tromey
 * Copyright (C) 2000 Red Hat, Inc.
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

#include <config.h>

#include <stdlib.h>
#ifdef HAVE_CODESET
#include <langinfo.h>
#endif
#include <string.h>

#include "glib.h"

#ifdef G_PLATFORM_WIN32
#include <stdio.h>
#define STRICT
#include <windows.h>
#undef STRICT
#endif

#include "glibintl.h"

#define UTF8_COMPUTE(Char, Mask, Len)					      \
  if (Char < 128)							      \
    {									      \
      Len = 1;								      \
      Mask = 0x7f;							      \
    }									      \
  else if ((Char & 0xe0) == 0xc0)					      \
    {									      \
      Len = 2;								      \
      Mask = 0x1f;							      \
    }									      \
  else if ((Char & 0xf0) == 0xe0)					      \
    {									      \
      Len = 3;								      \
      Mask = 0x0f;							      \
    }									      \
  else if ((Char & 0xf8) == 0xf0)					      \
    {									      \
      Len = 4;								      \
      Mask = 0x07;							      \
    }									      \
  else if ((Char & 0xfc) == 0xf8)					      \
    {									      \
      Len = 5;								      \
      Mask = 0x03;							      \
    }									      \
  else if ((Char & 0xfe) == 0xfc)					      \
    {									      \
      Len = 6;								      \
      Mask = 0x01;							      \
    }									      \
  else									      \
    Len = -1;

#define UTF8_LENGTH(Char)              \
  ((Char) < 0x80 ? 1 :                 \
   ((Char) < 0x800 ? 2 :               \
    ((Char) < 0x10000 ? 3 :            \
     ((Char) < 0x200000 ? 4 :          \
      ((Char) < 0x4000000 ? 5 : 6)))))
   

#define UTF8_GET(Result, Chars, Count, Mask, Len)			      \
  (Result) = (Chars)[0] & (Mask);					      \
  for ((Count) = 1; (Count) < (Len); ++(Count))				      \
    {									      \
      if (((Chars)[(Count)] & 0xc0) != 0x80)				      \
	{								      \
	  (Result) = -1;						      \
	  break;							      \
	}								      \
      (Result) <<= 6;							      \
      (Result) |= ((Chars)[(Count)] & 0x3f);				      \
    }

#define UNICODE_VALID(Char)                   \
    ((Char) < 0x110000 &&                     \
     ((Char) < 0xD800 || (Char) >= 0xE000) && \
     (Char) != 0xFFFE && (Char) != 0xFFFF)
   
     
static const gchar utf8_skip_data[256] = {
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,6,6,0,0
};

const gchar * const g_utf8_skip = utf8_skip_data;

/**
 * g_utf8_find_prev_char:
 * @str: pointer to the beginning of a UTF-8 encoded string
 * @p: pointer to some position within @str
 * 
 * Given a position @p with a UTF-8 encoded string @str, find the start
 * of the previous UTF-8 character starting before @p. Returns %NULL if no
 * UTF-8 characters are present in @p before @str.
 *
 * @p does not have to be at the beginning of a UTF-8 character. No check
 * is made to see if the character found is actually valid other than
 * it starts with an appropriate byte.
 *
 * Return value: a pointer to the found character or %NULL.
 **/
gchar *
g_utf8_find_prev_char (const char *str,
		       const char *p)
{
  for (--p; p >= str; --p)
    {
      if ((*p & 0xc0) != 0x80)
	return (gchar *)p;
    }
  return NULL;
}

/**
 * g_utf8_find_next_char:
 * @p: a pointer to a position within a UTF-8 encoded string
 * @end: a pointer to the end of the string, or %NULL to indicate
 *        that the string is nul-terminated, in which case
 *        the returned value will be 
 *
 * Finds the start of the next UTF-8 character in the string after @p.
 *
 * @p does not have to be at the beginning of a UTF-8 character. No check
 * is made to see if the character found is actually valid other than
 * it starts with an appropriate byte.
 * 
 * Return value: a pointer to the found character or %NULL
 **/
gchar *
g_utf8_find_next_char (const gchar *p,
		       const gchar *end)
{
  if (*p)
    {
      if (end)
	for (++p; p < end && (*p & 0xc0) == 0x80; ++p)
	  ;
      else
	for (++p; (*p & 0xc0) == 0x80; ++p)
	  ;
    }
  return (p == end) ? NULL : (gchar *)p;
}

/**
 * g_utf8_prev_char:
 * @p: a pointer to a position within a UTF-8 encoded string
 *
 * Finds the previous UTF-8 character in the string before @p.
 *
 * @p does not have to be at the beginning of a UTF-8 character. No check
 * is made to see if the character found is actually valid other than
 * it starts with an appropriate byte. If @p might be the first
 * character of the string, you must use g_utf8_find_prev_char() instead.
 * 
 * Return value: a pointer to the found character.
 **/
gchar *
g_utf8_prev_char (const gchar *p)
{
  while (TRUE)
    {
      p--;
      if ((*p & 0xc0) != 0x80)
	return (gchar *)p;
    }
}

/**
 * g_utf8_strlen:
 * @p: pointer to the start of a UTF-8 encoded string.
 * @max: the maximum number of bytes to examine. If @max
 *       is less than 0, then the string is assumed to be
 *       nul-terminated.
 * 
 * Returns the length of the string in characters.
 *
 * Return value: the length of the string in characters
 **/
glong
g_utf8_strlen (const gchar *p,
               gssize       max)
{
  glong len = 0;
  const gchar *start = p;

  if (max < 0)
    {
      while (*p)
        {
          p = g_utf8_next_char (p);
          ++len;
        }
    }
  else
    {
      if (max == 0 || !*p)
        return 0;
      
      p = g_utf8_next_char (p);          

      while (p - start < max && *p)
        {
          ++len;
          p = g_utf8_next_char (p);          
        }

      /* only do the last len increment if we got a complete
       * char (don't count partial chars)
       */
      if (p - start == max)
        ++len;
    }

  return len;
}

/**
 * g_utf8_get_char:
 * @p: a pointer to Unicode character encoded as UTF-8
 * 
 * Converts a sequence of bytes encoded as UTF-8 to a Unicode character.
 * If @p does not point to a valid UTF-8 encoded character, results are
 * undefined. If you are not sure that the bytes are complete
 * valid Unicode characters, you should use g_utf8_get_char_validated()
 * instead.
 * 
 * Return value: the resulting character
 **/
gunichar
g_utf8_get_char (const gchar *p)
{
  int i, mask = 0, len;
  gunichar result;
  unsigned char c = (unsigned char) *p;

  UTF8_COMPUTE (c, mask, len);
  if (len == -1)
    return (gunichar)-1;
  UTF8_GET (result, p, i, mask, len);

  return result;
}

/**
 * g_utf8_offset_to_pointer:
 * @str: a UTF-8 encoded string
 * @offset: a character offset within @str
 * 
 * Converts from an integer character offset to a pointer to a position
 * within the string.
 * 
 * Return value: the resulting pointer
 **/
gchar *
g_utf8_offset_to_pointer  (const gchar *str,
			   glong        offset)    
{
  const gchar *s = str;
  while (offset--)
    s = g_utf8_next_char (s);
  
  return (gchar *)s;
}

/**
 * g_utf8_pointer_to_offset:
 * @str: a UTF-8 encoded string
 * @pos: a pointer to a position within @str
 * 
 * Converts from a pointer to position within a string to a integer
 * character offset.
 * 
 * Return value: the resulting character offset
 **/
glong    
g_utf8_pointer_to_offset (const gchar *str,
			  const gchar *pos)
{
  const gchar *s = str;
  glong offset = 0;    
  
  while (s < pos)
    {
      s = g_utf8_next_char (s);
      offset++;
    }

  return offset;
}


/**
 * g_utf8_strncpy:
 * @dest: buffer to fill with characters from @src
 * @src: UTF-8 encoded string
 * @n: character count
 * 
 * Like the standard C <function>strncpy()</function> function, but 
 * copies a given number of characters instead of a given number of 
 * bytes. The @src string must be valid UTF-8 encoded text. 
 * (Use g_utf8_validate() on all text before trying to use UTF-8 
 * utility functions with it.)
 * 
 * Return value: @dest
 **/
gchar *
g_utf8_strncpy (gchar       *dest,
		const gchar *src,
		gsize        n)
{
  const gchar *s = src;
  while (n && *s)
    {
      s = g_utf8_next_char(s);
      n--;
    }
  strncpy(dest, src, s - src);
  dest[s - src] = 0;
  return dest;
}

static gboolean
g_utf8_get_charset_internal (char **a)
{
  char *charset = getenv("CHARSET");

  if (charset && a && ! *a)
    *a = charset;

  if (charset && strstr (charset, "UTF-8"))
      return TRUE;

#ifdef HAVE_CODESET
  charset = nl_langinfo(CODESET);
  if (charset)
    {
      if (a && ! *a)
	*a = charset;
      if (strcmp (charset, "UTF-8") == 0)
	return TRUE;
    }
#endif
  
#if 0 /* #ifdef _NL_CTYPE_CODESET_NAME */
  charset = nl_langinfo (_NL_CTYPE_CODESET_NAME);
  if (charset)
    {
      if (a && ! *a)
	*a = charset;
      if (strcmp (charset, "UTF-8") == 0)
	return TRUE;
    }
#endif

#ifdef G_PLATFORM_WIN32
  if (a && ! *a)
    {
      static char codepage[10];
      
      sprintf (codepage, "CP%d", GetACP ());
      *a = codepage;
      /* What about codepage 1200? Is that UTF-8? */
      return FALSE;
    }
#else
  if (a && ! *a) 
    *a = "US-ASCII";
#endif

  /* Assume this for compatibility at present.  */
  return FALSE;
}

static int utf8_locale_cache = -1;
static char *utf8_charset_cache = NULL;

/**
 * g_get_charset:
 * @charset: return location for character set name
 * 
 * Obtains the character set for the current locale; you might use
 * this character set as an argument to g_convert(), to convert from
 * the current locale's encoding to some other encoding. (Frequently
 * g_locale_to_utf8() and g_locale_from_utf8() are nice shortcuts,
 * though.)
 *
 * The return value is %TRUE if the locale's encoding is UTF-8, in that
 * case you can perhaps avoid calling g_convert().
 *
 * The string returned in @charset is not allocated, and should not be
 * freed.
 * 
 * Return value: %TRUE if the returned charset is UTF-8
 **/
gboolean
g_get_charset (G_CONST_RETURN char **charset) 
{
  if (utf8_locale_cache != -1)
    {
      if (charset)
	*charset = utf8_charset_cache;
      return utf8_locale_cache;
    }
  utf8_locale_cache = g_utf8_get_charset_internal (&utf8_charset_cache);
  if (charset) 
    *charset = utf8_charset_cache;
  return utf8_locale_cache;
}

/* unicode_strchr */

/**
 * g_unichar_to_utf8:
 * @c: a ISO10646 character code
 * @outbuf: output buffer, must have at least 6 bytes of space.
 *       If %NULL, the length will be computed and returned
 *       and nothing will be written to @outbuf.
 * 
 * Converts a single character to UTF-8.
 * 
 * Return value: number of bytes written
 **/
int
g_unichar_to_utf8 (gunichar c,
		   gchar   *outbuf)
{
  guint len = 0;    
  int first;
  int i;

  if (c < 0x80)
    {
      first = 0;
      len = 1;
    }
  else if (c < 0x800)
    {
      first = 0xc0;
      len = 2;
    }
  else if (c < 0x10000)
    {
      first = 0xe0;
      len = 3;
    }
   else if (c < 0x200000)
    {
      first = 0xf0;
      len = 4;
    }
  else if (c < 0x4000000)
    {
      first = 0xf8;
      len = 5;
    }
  else
    {
      first = 0xfc;
      len = 6;
    }

  if (outbuf)
    {
      for (i = len - 1; i > 0; --i)
	{
	  outbuf[i] = (c & 0x3f) | 0x80;
	  c >>= 6;
	}
      outbuf[0] = c | first;
    }

  return len;
}

/**
 * g_utf8_strchr:
 * @p: a nul-terminated UTF-8 encoded string
 * @len: the maximum length of @p
 * @c: a ISO10646 character
 * 
 * Finds the leftmost occurrence of the given ISO10646 character
 * in a UTF-8 encoded string, while limiting the search to @len bytes.
 * If @len is -1, allow unbounded search.
 * 
 * Return value: %NULL if the string does not contain the character, 
 *   otherwise, a pointer to the start of the leftmost occurrence of 
 *   the character in the string.
 **/
gchar *
g_utf8_strchr (const char *p,
	       gssize      len,
	       gunichar    c)
{
  gchar ch[10];

  gint charlen = g_unichar_to_utf8 (c, ch);
  ch[charlen] = '\0';
  
  return g_strstr_len (p, len, ch);
}


/**
 * g_utf8_strrchr:
 * @p: a nul-terminated UTF-8 encoded string
 * @len: the maximum length of @p
 * @c: a ISO10646 character
 * 
 * Find the rightmost occurrence of the given ISO10646 character
 * in a UTF-8 encoded string, while limiting the search to @len bytes.
 * If @len is -1, allow unbounded search.
 * 
 * Return value: %NULL if the string does not contain the character, 
 *   otherwise, a pointer to the start of the rightmost occurrence of the 
 *   character in the string.
 **/
gchar *
g_utf8_strrchr (const char *p,
		gssize      len,
		gunichar    c)
{
  gchar ch[10];

  gint charlen = g_unichar_to_utf8 (c, ch);
  ch[charlen] = '\0';
  
  return g_strrstr_len (p, len, ch);
}


/* Like g_utf8_get_char, but take a maximum length
 * and return (gunichar)-2 on incomplete trailing character
 */
static inline gunichar
g_utf8_get_char_extended (const  gchar *p,
			  gssize max_len)  
{
  guint i, len;
  gunichar wc = (guchar) *p;

  if (wc < 0x80)
    {
      return wc;
    }
  else if (wc < 0xc0)
    {
      return (gunichar)-1;
    }
  else if (wc < 0xe0)
    {
      len = 2;
      wc &= 0x1f;
    }
  else if (wc < 0xf0)
    {
      len = 3;
      wc &= 0x0f;
    }
  else if (wc < 0xf8)
    {
      len = 4;
      wc &= 0x07;
    }
  else if (wc < 0xfc)
    {
      len = 5;
      wc &= 0x03;
    }
  else if (wc < 0xfe)
    {
      len = 6;
      wc &= 0x01;
    }
  else
    {
      return (gunichar)-1;
    }
  
  if (max_len >= 0 && len > max_len)
    {
      for (i = 1; i < max_len; i++)
	{
	  if ((((guchar *)p)[i] & 0xc0) != 0x80)
	    return (gunichar)-1;
	}
      return (gunichar)-2;
    }

  for (i = 1; i < len; ++i)
    {
      gunichar ch = ((guchar *)p)[i];
      
      if ((ch & 0xc0) != 0x80)
	{
	  if (ch)
	    return (gunichar)-1;
	  else
	    return (gunichar)-2;
	}

      wc <<= 6;
      wc |= (ch & 0x3f);
    }

  if (UTF8_LENGTH(wc) != len)
    return (gunichar)-1;
  
  return wc;
}

/**
 * g_utf8_get_char_validated:
 * @p: a pointer to Unicode character encoded as UTF-8
 * @max_len: the maximum number of bytes to read, or -1, for no maximum.
 * 
 * Convert a sequence of bytes encoded as UTF-8 to a Unicode character.
 * This function checks for incomplete characters, for invalid characters
 * such as characters that are out of the range of Unicode, and for
 * overlong encodings of valid characters.
 * 
 * Return value: the resulting character. If @p points to a partial
 *    sequence at the end of a string that could begin a valid character,
 *    returns (gunichar)-2; otherwise, if @p does not point to a valid
 *    UTF-8 encoded Unicode character, returns (gunichar)-1.
 **/
gunichar
g_utf8_get_char_validated (const  gchar *p,
			   gssize max_len)
{
  gunichar result = g_utf8_get_char_extended (p, max_len);

  if (result & 0x80000000)
    return result;
  else if (!UNICODE_VALID (result))
    return (gunichar)-1;
  else
    return result;
}

/**
 * g_utf8_to_ucs4_fast:
 * @str: a UTF-8 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is nul-terminated.
 * @items_written: location to store the number of characters in the
 *                 result, or %NULL.
 *
 * Convert a string from UTF-8 to a 32-bit fixed width
 * representation as UCS-4, assuming valid UTF-8 input.
 * This function is roughly twice as fast as g_utf8_to_ucs4()
 * but does no error checking on the input.
 * 
 * Return value: a pointer to a newly allocated UCS-4 string.
 *               This value must be freed with g_free().
 **/
gunichar *
g_utf8_to_ucs4_fast (const gchar *str,
		     glong        len,              
		     glong       *items_written)    
{
  gint j, charlen;
  gunichar *result;
  gint n_chars, i;
  const gchar *p;

  g_return_val_if_fail (str != NULL, NULL);

  p = str;
  n_chars = 0;
  if (len < 0)
    {
      while (*p)
	{
	  p = g_utf8_next_char (p);
	  ++n_chars;
	}
    }
  else
    {
      while (p < str + len && *p)
	{
	  p = g_utf8_next_char (p);
	  ++n_chars;
	}
    }
  
  result = g_new (gunichar, n_chars + 1);
  
  p = str;
  for (i=0; i < n_chars; i++)
    {
      gunichar wc = ((unsigned char *)p)[0];

      if (wc < 0x80)
	{
	  result[i] = wc;
	  p++;
	}
      else
	{ 
	  if (wc < 0xe0)
	    {
	      charlen = 2;
	      wc &= 0x1f;
	    }
	  else if (wc < 0xf0)
	    {
	      charlen = 3;
	      wc &= 0x0f;
	    }
	  else if (wc < 0xf8)
	    {
	      charlen = 4;
	      wc &= 0x07;
	    }
	  else if (wc < 0xfc)
	    {
	      charlen = 5;
	      wc &= 0x03;
	    }
	  else
	    {
	      charlen = 6;
	      wc &= 0x01;
	    }

	  for (j = 1; j < charlen; j++)
	    {
	      wc <<= 6;
	      wc |= ((unsigned char *)p)[j] & 0x3f;
	    }

	  result[i] = wc;
	  p += charlen;
	}
    }
  result[i] = 0;

  if (items_written)
    *items_written = i;

  return result;
}

/**
 * g_utf8_to_ucs4:
 * @str: a UTF-8 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is nul-terminated.
 * @items_read: location to store number of bytes read, or %NULL.
 *              If %NULL, then %G_CONVERT_ERROR_PARTIAL_INPUT will be
 *              returned in case @str contains a trailing partial
 *              character. If an error occurs then the index of the
 *              invalid input is stored here.
 * @items_written: location to store number of characters written or %NULL.
 *                 The value here stored does not include the trailing 0
 *                 character. 
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from UTF-8 to a 32-bit fixed width
 * representation as UCS-4. A trailing 0 will be added to the
 * string after the converted text.
 * 
 * Return value: a pointer to a newly allocated UCS-4 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gunichar *
g_utf8_to_ucs4 (const gchar *str,
		glong        len,             
		glong       *items_read,      
		glong       *items_written,   
		GError     **error)
{
  gunichar *result = NULL;
  gint n_chars, i;
  const gchar *in;
  
  in = str;
  n_chars = 0;
  while ((len < 0 || str + len - in > 0) && *in)
    {
      gunichar wc = g_utf8_get_char_extended (in, str + len - in);
      if (wc & 0x80000000)
	{
	  if (wc == (gunichar)-2)
	    {
	      if (items_read)
		break;
	      else
		g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_PARTIAL_INPUT,
			     _("Partial character sequence at end of input"));
	    }
	  else
	    g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			 _("Invalid byte sequence in conversion input"));

	  goto err_out;
	}

      n_chars++;

      in = g_utf8_next_char (in);
    }

  result = g_new (gunichar, n_chars + 1);
  
  in = str;
  for (i=0; i < n_chars; i++)
    {
      result[i] = g_utf8_get_char (in);
      in = g_utf8_next_char (in);
    }
  result[i] = 0;

  if (items_written)
    *items_written = n_chars;

 err_out:
  if (items_read)
    *items_read = in - str;

  return result;
}

/**
 * g_ucs4_to_utf8:
 * @str: a UCS-4 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is terminated with a 0 character.
 * @items_read: location to store number of characters read read, or %NULL.
 * @items_written: location to store number of bytes written or %NULL.
 *                 The value here stored does not include the trailing 0
 *                 byte. 
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from a 32-bit fixed width representation as UCS-4.
 * to UTF-8. The result will be terminated with a 0 byte.
 * 
 * Return value: a pointer to a newly allocated UTF-8 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gchar *
g_ucs4_to_utf8 (const gunichar *str,
		glong           len,              
		glong          *items_read,       
		glong          *items_written,    
		GError        **error)
{
  gint result_length;
  gchar *result = NULL;
  gchar *p;
  gint i;

  result_length = 0;
  for (i = 0; len < 0 || i < len ; i++)
    {
      if (!str[i])
	break;

      if (str[i] >= 0x80000000)
	{
	  if (items_read)
	    *items_read = i;
	  
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Character out of range for UTF-8"));
	  goto err_out;
	}
      
      result_length += UTF8_LENGTH (str[i]);
    }

  result = g_malloc (result_length + 1);
  p = result;

  i = 0;
  while (p < result + result_length)
    p += g_unichar_to_utf8 (str[i++], p);
  
  *p = '\0';

  if (items_written)
    *items_written = p - result;

 err_out:
  if (items_read)
    *items_read = i;

  return result;
}

#define SURROGATE_VALUE(h,l) (((h) - 0xd800) * 0x400 + (l) - 0xdc00 + 0x10000)

/**
 * g_utf16_to_utf8:
 * @str: a UTF-16 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is terminated with a 0 character.
 * @items_read: location to store number of words read, or %NULL.
 *              If %NULL, then %G_CONVERT_ERROR_PARTIAL_INPUT will be
 *              returned in case @str contains a trailing partial
 *              character. If an error occurs then the index of the
 *              invalid input is stored here.
 * @items_written: location to store number of bytes written, or %NULL.
 *                 The value stored here does not include the trailing
 *                 0 byte.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from UTF-16 to UTF-8. The result will be
 * terminated with a 0 byte.
 * 
 * Return value: a pointer to a newly allocated UTF-8 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gchar *
g_utf16_to_utf8 (const gunichar2  *str,
		 glong             len,              
		 glong            *items_read,       
		 glong            *items_written,    
		 GError          **error)
{
  /* This function and g_utf16_to_ucs4 are almost exactly identical - The lines that differ
   * are marked.
   */
  const gunichar2 *in;
  gchar *out;
  gchar *result = NULL;
  gint n_bytes;
  gunichar high_surrogate;

  g_return_val_if_fail (str != 0, NULL);

  n_bytes = 0;
  in = str;
  high_surrogate = 0;
  while ((len < 0 || in - str < len) && *in)
    {
      gunichar2 c = *in;
      gunichar wc;

      if (c >= 0xdc00 && c < 0xe000) /* low surrogate */
	{
	  if (high_surrogate)
	    {
	      wc = SURROGATE_VALUE (high_surrogate, c);
	      high_surrogate = 0;
	    }
	  else
	    {
	      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			   _("Invalid sequence in conversion input"));
	      goto err_out;
	    }
	}
      else
	{
	  if (high_surrogate)
	    {
	      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			   _("Invalid sequence in conversion input"));
	      goto err_out;
	    }

	  if (c >= 0xd800 && c < 0xdc00) /* high surrogate */
	    {
	      high_surrogate = c;
	      goto next1;
	    }
	  else
	    wc = c;
	}

      /********** DIFFERENT for UTF8/UCS4 **********/
      n_bytes += UTF8_LENGTH (wc);

    next1:
      in++;
    }

  if (high_surrogate && !items_read)
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_PARTIAL_INPUT,
		   _("Partial character sequence at end of input"));
      goto err_out;
    }
  
  /* At this point, everything is valid, and we just need to convert
   */
  /********** DIFFERENT for UTF8/UCS4 **********/
  result = g_malloc (n_bytes + 1);
  
  high_surrogate = 0;
  out = result;
  in = str;
  while (out < result + n_bytes)
    {
      gunichar2 c = *in;
      gunichar wc;

      if (c >= 0xdc00 && c < 0xe000) /* low surrogate */
	{
	  wc = SURROGATE_VALUE (high_surrogate, c);
	  high_surrogate = 0;
	}
      else if (c >= 0xd800 && c < 0xdc00) /* high surrogate */
	{
	  high_surrogate = c;
	  goto next2;
	}
      else
	wc = c;

      /********** DIFFERENT for UTF8/UCS4 **********/
      out += g_unichar_to_utf8 (wc, out);

    next2:
      in++;
    }
  
  /********** DIFFERENT for UTF8/UCS4 **********/
  *out = '\0';

  if (items_written)
    /********** DIFFERENT for UTF8/UCS4 **********/
    *items_written = out - result;

 err_out:
  if (items_read)
    *items_read = in - str;

  return result;
}

/**
 * g_utf16_to_ucs4:
 * @str: a UTF-16 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is terminated with a 0 character.
 * @items_read: location to store number of words read, or %NULL.
 *              If %NULL, then %G_CONVERT_ERROR_PARTIAL_INPUT will be
 *              returned in case @str contains a trailing partial
 *              character. If an error occurs then the index of the
 *              invalid input is stored here.
 * @items_written: location to store number of characters written, or %NULL.
 *                 The value stored here does not include the trailing
 *                 0 character.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from UTF-16 to UCS-4. The result will be
 * terminated with a 0 character.
 * 
 * Return value: a pointer to a newly allocated UCS-4 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gunichar *
g_utf16_to_ucs4 (const gunichar2  *str,
		 glong             len,              
		 glong            *items_read,       
		 glong            *items_written,    
		 GError          **error)
{
  const gunichar2 *in;
  gchar *out;
  gchar *result = NULL;
  gint n_bytes;
  gunichar high_surrogate;

  g_return_val_if_fail (str != 0, NULL);

  n_bytes = 0;
  in = str;
  high_surrogate = 0;
  while ((len < 0 || in - str < len) && *in)
    {
      gunichar2 c = *in;
      gunichar wc;

      if (c >= 0xdc00 && c < 0xe000) /* low surrogate */
	{
	  if (high_surrogate)
	    {
	      wc = SURROGATE_VALUE (high_surrogate, c);
	      high_surrogate = 0;
	    }
	  else
	    {
	      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			   _("Invalid sequence in conversion input"));
	      goto err_out;
	    }
	}
      else
	{
	  if (high_surrogate)
	    {
	      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			   _("Invalid sequence in conversion input"));
	      goto err_out;
	    }

	  if (c >= 0xd800 && c < 0xdc00) /* high surrogate */
	    {
	      high_surrogate = c;
	      goto next1;
	    }
	  else
	    wc = c;
	}

      /********** DIFFERENT for UTF8/UCS4 **********/
      n_bytes += sizeof (gunichar);

    next1:
      in++;
    }

  if (high_surrogate && !items_read)
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_PARTIAL_INPUT,
		   _("Partial character sequence at end of input"));
      goto err_out;
    }
  
  /* At this point, everything is valid, and we just need to convert
   */
  /********** DIFFERENT for UTF8/UCS4 **********/
  result = g_malloc (n_bytes + 4);
  
  high_surrogate = 0;
  out = result;
  in = str;
  while (out < result + n_bytes)
    {
      gunichar2 c = *in;
      gunichar wc;

      if (c >= 0xdc00 && c < 0xe000) /* low surrogate */
	{
	  wc = SURROGATE_VALUE (high_surrogate, c);
	  high_surrogate = 0;
	}
      else if (c >= 0xd800 && c < 0xdc00) /* high surrogate */
	{
	  high_surrogate = c;
	  goto next2;
	}
      else
	wc = c;

      /********** DIFFERENT for UTF8/UCS4 **********/
      *(gunichar *)out = wc;
      out += sizeof (gunichar);

    next2:
      in++;
    }

  /********** DIFFERENT for UTF8/UCS4 **********/
  *(gunichar *)out = 0;

  if (items_written)
    /********** DIFFERENT for UTF8/UCS4 **********/
    *items_written = (out - result) / sizeof (gunichar);

 err_out:
  if (items_read)
    *items_read = in - str;

  return (gunichar *)result;
}

/**
 * g_utf8_to_utf16:
 * @str: a UTF-8 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is nul-terminated.
 
 * @items_read: location to store number of bytes read, or %NULL.
 *              If %NULL, then %G_CONVERT_ERROR_PARTIAL_INPUT will be
 *              returned in case @str contains a trailing partial
 *              character. If an error occurs then the index of the
 *              invalid input is stored here.
 * @items_written: location to store number of words written, or %NULL.
 *                 The value stored here does not include the trailing
 *                 0 word.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from UTF-8 to UTF-16. A 0 word will be
 * added to the result after the converted text.
 * 
 * Return value: a pointer to a newly allocated UTF-16 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gunichar2 *
g_utf8_to_utf16 (const gchar *str,
		 glong        len,              
		 glong       *items_read,       
		 glong       *items_written,    
		 GError     **error)
{
  gunichar2 *result = NULL;
  gint n16;
  const gchar *in;
  gint i;

  g_return_val_if_fail (str != NULL, NULL);

  in = str;
  n16 = 0;
  while ((len < 0 || str + len - in > 0) && *in)
    {
      gunichar wc = g_utf8_get_char_extended (in, str + len - in);
      if (wc & 0x80000000)
	{
	  if (wc == (gunichar)-2)
	    {
	      if (items_read)
		break;
	      else
		g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_PARTIAL_INPUT,
			     _("Partial character sequence at end of input"));
	    }
	  else
	    g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			 _("Invalid byte sequence in conversion input"));

	  goto err_out;
	}

      if (wc < 0xd800)
	n16 += 1;
      else if (wc < 0xe000)
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Invalid sequence in conversion input"));

	  goto err_out;
	}
      else if (wc < 0x10000)
	n16 += 1;
      else if (wc < 0x110000)
	n16 += 2;
      else
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Character out of range for UTF-16"));

	  goto err_out;
	}
      
      in = g_utf8_next_char (in);
    }

  result = g_new (gunichar2, n16 + 1);
  
  in = str;
  for (i = 0; i < n16;)
    {
      gunichar wc = g_utf8_get_char (in);

      if (wc < 0x10000)
	{
	  result[i++] = wc;
	}
      else
	{
	  result[i++] = (wc - 0x10000) / 0x400 + 0xd800;
	  result[i++] = (wc - 0x10000) % 0x400 + 0xdc00;
	}
      
      in = g_utf8_next_char (in);
    }

  result[i] = 0;

  if (items_written)
    *items_written = n16;

 err_out:
  if (items_read)
    *items_read = in - str;
  
  return result;
}

/**
 * g_ucs4_to_utf16:
 * @str: a UCS-4 encoded string
 * @len: the maximum length of @str to use. If @len < 0, then
 *       the string is terminated with a 0 character.
 * @items_read: location to store number of bytes read, or %NULL.
 *              If an error occurs then the index of the invalid input
 *              is stored here.
 * @items_written: location to store number of words written, or %NULL.
 *                 The value stored here does not include the trailing
 *                 0 word.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError other than
 *         %G_CONVERT_ERROR_NO_CONVERSION may occur.
 *
 * Convert a string from UCS-4 to UTF-16. A 0 word will be
 * added to the result after the converted text.
 * 
 * Return value: a pointer to a newly allocated UTF-16 string.
 *               This value must be freed with g_free(). If an
 *               error occurs, %NULL will be returned and
 *               @error set.
 **/
gunichar2 *
g_ucs4_to_utf16 (const gunichar  *str,
		 glong            len,              
		 glong           *items_read,       
		 glong           *items_written,    
		 GError         **error)
{
  gunichar2 *result = NULL;
  gint n16;
  gint i, j;

  n16 = 0;
  i = 0;
  while ((len < 0 || i < len) && str[i])
    {
      gunichar wc = str[i];

      if (wc < 0xd800)
	n16 += 1;
      else if (wc < 0xe000)
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Invalid sequence in conversion input"));

	  goto err_out;
	}
      else if (wc < 0x10000)
	n16 += 1;
      else if (wc < 0x110000)
	n16 += 2;
      else
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Character out of range for UTF-16"));

	  goto err_out;
	}

      i++;
    }
  
  result = g_new (gunichar2, n16 + 1);
  
  for (i = 0, j = 0; j < n16; i++)
    {
      gunichar wc = str[i];

      if (wc < 0x10000)
	{
	  result[j++] = wc;
	}
      else
	{
	  result[j++] = (wc - 0x10000) / 0x400 + 0xd800;
	  result[j++] = (wc - 0x10000) % 0x400 + 0xdc00;
	}
    }
  result[j] = 0;

  if (items_written)
    *items_written = n16;
  
 err_out:
  if (items_read)
    *items_read = i;
  
  return result;
}

/**
 * g_utf8_validate:
 * @str: a pointer to character data
 * @max_len: max bytes to validate, or -1 to go until nul
 * @end: return location for end of valid data
 * 
 * Validates UTF-8 encoded text. @str is the text to validate;
 * if @str is nul-terminated, then @max_len can be -1, otherwise
 * @max_len should be the number of bytes to validate.
 * If @end is non-%NULL, then the end of the valid range
 * will be stored there (i.e. the address of the first invalid byte
 * if some bytes were invalid, or the end of the text being validated
 * otherwise).
 *
 * Returns %TRUE if all of @str was valid. Many GLib and GTK+
 * routines <emphasis>require</emphasis> valid UTF-8 as input;
 * so data read from a file or the network should be checked
 * with g_utf8_validate() before doing anything else with it.
 * 
 * Return value: %TRUE if the text was valid UTF-8
 **/
gboolean
g_utf8_validate (const gchar  *str,
                 gssize        max_len,    
                 const gchar **end)
{

  const gchar *p;

  g_return_val_if_fail (str != NULL, FALSE);
  
  if (end)
    *end = str;
  
  p = str;
  
  while ((max_len < 0 || (p - str) < max_len) && *p)
    {
      int i, mask = 0, len;
      gunichar result;
      unsigned char c = (unsigned char) *p;
      
      UTF8_COMPUTE (c, mask, len);

      if (len == -1)
        break;

      /* check that the expected number of bytes exists in str */
      if (max_len >= 0 &&
          ((max_len - (p - str)) < len))
        break;
        
      UTF8_GET (result, p, i, mask, len);

      if (UTF8_LENGTH (result) != len) /* Check for overlong UTF-8 */
	break;

      if (result == (gunichar)-1)
        break;

      if (!UNICODE_VALID (result))
	break;
      
      p += len;
    }

  if (end)
    *end = p;

  /* See that we covered the entire length if a length was
   * passed in, or that we ended on a nul if not
   */
  if (max_len >= 0 &&
      p != (str + max_len))
    return FALSE;
  else if (max_len < 0 &&
           *p != '\0')
    return FALSE;
  else
    return TRUE;
}

/**
 * g_unichar_validate:
 * @ch: a Unicode character
 * 
 * Checks whether @ch is a valid Unicode character. Some possible
 * integer values of @ch will not be valid. 0 is considered a valid
 * character, though it's normally a string terminator.
 * 
 * Return value: %TRUE if @ch is a valid Unicode character
 **/
gboolean
g_unichar_validate (gunichar ch)
{
  return UNICODE_VALID (ch);
}
