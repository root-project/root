/* gunicode.h - Unicode manipulation functions
 *
 *  Copyright (C) 1999, 2000 Tom Tromey
 *  Copyright 2000 Red Hat, Inc.
 *
 * The Gnome Library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * The Gnome Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with the Gnome Library; see the file COPYING.LIB.  If not,
 * write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 *   Boston, MA 02111-1307, USA.
 */

#ifndef __G_UNICODE_H__
#define __G_UNICODE_H__

#include <glib/gerror.h>
#include <glib/gtypes.h>

G_BEGIN_DECLS

typedef guint32 gunichar;
typedef guint16 gunichar2;

/* These are the possible character classifications.
 * See http://www.unicode.org/Public/UNIDATA/UnicodeData.html
 */
typedef enum
{
  G_UNICODE_CONTROL,
  G_UNICODE_FORMAT,
  G_UNICODE_UNASSIGNED,
  G_UNICODE_PRIVATE_USE,
  G_UNICODE_SURROGATE,
  G_UNICODE_LOWERCASE_LETTER,
  G_UNICODE_MODIFIER_LETTER,
  G_UNICODE_OTHER_LETTER,
  G_UNICODE_TITLECASE_LETTER,
  G_UNICODE_UPPERCASE_LETTER,
  G_UNICODE_COMBINING_MARK,
  G_UNICODE_ENCLOSING_MARK,
  G_UNICODE_NON_SPACING_MARK,
  G_UNICODE_DECIMAL_NUMBER,
  G_UNICODE_LETTER_NUMBER,
  G_UNICODE_OTHER_NUMBER,
  G_UNICODE_CONNECT_PUNCTUATION,
  G_UNICODE_DASH_PUNCTUATION,
  G_UNICODE_CLOSE_PUNCTUATION,
  G_UNICODE_FINAL_PUNCTUATION,
  G_UNICODE_INITIAL_PUNCTUATION,
  G_UNICODE_OTHER_PUNCTUATION,
  G_UNICODE_OPEN_PUNCTUATION,
  G_UNICODE_CURRENCY_SYMBOL,
  G_UNICODE_MODIFIER_SYMBOL,
  G_UNICODE_MATH_SYMBOL,
  G_UNICODE_OTHER_SYMBOL,
  G_UNICODE_LINE_SEPARATOR,
  G_UNICODE_PARAGRAPH_SEPARATOR,
  G_UNICODE_SPACE_SEPARATOR
} GUnicodeType;

/* These are the possible line break classifications.
 * See http://www.unicode.org/unicode/reports/tr14/
 */
typedef enum
{
  G_UNICODE_BREAK_MANDATORY,
  G_UNICODE_BREAK_CARRIAGE_RETURN,
  G_UNICODE_BREAK_LINE_FEED,
  G_UNICODE_BREAK_COMBINING_MARK,
  G_UNICODE_BREAK_SURROGATE,
  G_UNICODE_BREAK_ZERO_WIDTH_SPACE,
  G_UNICODE_BREAK_INSEPARABLE,
  G_UNICODE_BREAK_NON_BREAKING_GLUE,
  G_UNICODE_BREAK_CONTINGENT,
  G_UNICODE_BREAK_SPACE,
  G_UNICODE_BREAK_AFTER,
  G_UNICODE_BREAK_BEFORE,
  G_UNICODE_BREAK_BEFORE_AND_AFTER,
  G_UNICODE_BREAK_HYPHEN,
  G_UNICODE_BREAK_NON_STARTER,
  G_UNICODE_BREAK_OPEN_PUNCTUATION,
  G_UNICODE_BREAK_CLOSE_PUNCTUATION,
  G_UNICODE_BREAK_QUOTATION,
  G_UNICODE_BREAK_EXCLAMATION,
  G_UNICODE_BREAK_IDEOGRAPHIC,
  G_UNICODE_BREAK_NUMERIC,
  G_UNICODE_BREAK_INFIX_SEPARATOR,
  G_UNICODE_BREAK_SYMBOL,
  G_UNICODE_BREAK_ALPHABETIC,
  G_UNICODE_BREAK_PREFIX,
  G_UNICODE_BREAK_POSTFIX,
  G_UNICODE_BREAK_COMPLEX_CONTEXT,
  G_UNICODE_BREAK_AMBIGUOUS,
  G_UNICODE_BREAK_UNKNOWN
} GUnicodeBreakType;

/* Returns TRUE if current locale uses UTF-8 charset.  If CHARSET is
 * not null, sets *CHARSET to the name of the current locale's
 * charset.  This value is statically allocated, and should be copied
 * in case the locale's charset will be changed later using setlocale()
 * or in some other way.
 */
gboolean g_get_charset (G_CONST_RETURN char **charset);

/* These are all analogs of the <ctype.h> functions.
 */
gboolean g_unichar_isalnum   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isalpha   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_iscntrl   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isdigit   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isgraph   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_islower   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isprint   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_ispunct   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isspace   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isupper   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isxdigit  (gunichar c) G_GNUC_CONST;
gboolean g_unichar_istitle   (gunichar c) G_GNUC_CONST;
gboolean g_unichar_isdefined (gunichar c) G_GNUC_CONST;
gboolean g_unichar_iswide    (gunichar c) G_GNUC_CONST;

/* More <ctype.h> functions.  These convert between the three cases.
 * See the Unicode book to understand title case.  */
gunichar g_unichar_toupper (gunichar c) G_GNUC_CONST;
gunichar g_unichar_tolower (gunichar c) G_GNUC_CONST;
gunichar g_unichar_totitle (gunichar c) G_GNUC_CONST;

/* If C is a digit (according to `g_unichar_isdigit'), then return its
   numeric value.  Otherwise return -1.  */
gint g_unichar_digit_value (gunichar c) G_GNUC_CONST;

gint g_unichar_xdigit_value (gunichar c) G_GNUC_CONST;

/* Return the Unicode character type of a given character.  */
GUnicodeType g_unichar_type (gunichar c) G_GNUC_CONST;

/* Return the line break property for a given character */
GUnicodeBreakType g_unichar_break_type (gunichar c) G_GNUC_CONST;


/* Compute canonical ordering of a string in-place.  This rearranges
   decomposed characters in the string according to their combining
   classes.  See the Unicode manual for more information.  */
void g_unicode_canonical_ordering (gunichar *string,
				   gsize     len);

/* Compute canonical decomposition of a character.  Returns g_malloc()d
   string of Unicode characters.  RESULT_LEN is set to the resulting
   length of the string.  */
gunichar *g_unicode_canonical_decomposition (gunichar  ch,
					     gsize    *result_len);

/* Array of skip-bytes-per-initial character.
 */
GLIB_VAR const gchar * const g_utf8_skip;

#define g_utf8_next_char(p) (char *)((p) + g_utf8_skip[*(guchar *)(p)])

gunichar g_utf8_get_char           (const gchar  *p);
gunichar g_utf8_get_char_validated (const  gchar *p,
				    gssize        max_len);

gchar*   g_utf8_offset_to_pointer (const gchar *str,
                                   glong        offset);  
glong    g_utf8_pointer_to_offset (const gchar *str,      
				   const gchar *pos);
gchar*   g_utf8_prev_char         (const gchar *p);
gchar*   g_utf8_find_next_char    (const gchar *p,
				   const gchar *end);
gchar*   g_utf8_find_prev_char    (const gchar *str,
				   const gchar *p);

glong g_utf8_strlen (const gchar *p,  
		     gssize       max);        

/* Copies n characters from src to dest */
gchar* g_utf8_strncpy (gchar       *dest,
		       const gchar *src,
		       gsize        n);

/* Find the UTF-8 character corresponding to ch, in string p. These
   functions are equivalants to strchr and strrchr */
gchar* g_utf8_strchr  (const gchar *p,
		       gssize       len,
		       gunichar     c);
gchar* g_utf8_strrchr (const gchar *p,
		       gssize       len,
		       gunichar     c);

gunichar2 *g_utf8_to_utf16     (const gchar      *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);
gunichar * g_utf8_to_ucs4      (const gchar      *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);
gunichar * g_utf8_to_ucs4_fast (const gchar      *str,
				glong             len,            
				glong            *items_written); 
gunichar * g_utf16_to_ucs4     (const gunichar2  *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);
gchar*     g_utf16_to_utf8     (const gunichar2  *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);
gunichar2 *g_ucs4_to_utf16     (const gunichar   *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);
gchar*     g_ucs4_to_utf8      (const gunichar   *str,
				glong             len,            
				glong            *items_read,     
				glong            *items_written,  
				GError          **error);

/* Convert a single character into UTF-8. outbuf must have at
 * least 6 bytes of space. Returns the number of bytes in the
 * result.
 */
gint      g_unichar_to_utf8 (gunichar    c,
			     gchar      *outbuf);

/* Validate a UTF8 string, return TRUE if valid, put pointer to
 * first invalid char in **end
 */

gboolean g_utf8_validate (const gchar  *str,
                          gssize        max_len,  
                          const gchar **end);

/* Validate a Unicode character */
gboolean g_unichar_validate (gunichar ch);

gchar *g_utf8_strup   (const gchar *str,
		       gssize       len);
gchar *g_utf8_strdown (const gchar *str,
		       gssize       len);
gchar *g_utf8_casefold (const gchar *str,
			gssize       len);

typedef enum {
  G_NORMALIZE_DEFAULT,
  G_NORMALIZE_NFD = G_NORMALIZE_DEFAULT,
  G_NORMALIZE_DEFAULT_COMPOSE,
  G_NORMALIZE_NFC = G_NORMALIZE_DEFAULT_COMPOSE,
  G_NORMALIZE_ALL,
  G_NORMALIZE_NFKD = G_NORMALIZE_ALL,
  G_NORMALIZE_ALL_COMPOSE,
  G_NORMALIZE_NFKC = G_NORMALIZE_ALL_COMPOSE
} GNormalizeMode;

gchar *g_utf8_normalize (const gchar   *str,
			 gssize         len,
			 GNormalizeMode mode);

gint   g_utf8_collate     (const gchar *str1,
			   const gchar *str2);
gchar *g_utf8_collate_key (const gchar *str,
			   gssize       len);

G_END_DECLS

#endif /* __G_UNICODE_H__ */
