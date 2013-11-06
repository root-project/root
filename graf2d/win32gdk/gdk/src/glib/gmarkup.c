/* gmarkup.c - Simple XML-like parser
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

#include "glib.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "glibintl.h"

GQuark
g_markup_error_quark ()
{
  static GQuark error_quark = 0;

  if (error_quark == 0)
    error_quark = g_quark_from_static_string ("g-markup-error-quark");

  return error_quark;
}

typedef enum
{
  STATE_START,
  STATE_AFTER_OPEN_ANGLE,
  STATE_AFTER_CLOSE_ANGLE,
  STATE_AFTER_ELISION_SLASH, /* the slash that obviates need for end element */
  STATE_INSIDE_OPEN_TAG_NAME,
  STATE_INSIDE_ATTRIBUTE_NAME,
  STATE_BETWEEN_ATTRIBUTES,
  STATE_AFTER_ATTRIBUTE_EQUALS_SIGN,
  STATE_INSIDE_ATTRIBUTE_VALUE,
  STATE_INSIDE_TEXT,
  STATE_AFTER_CLOSE_TAG_SLASH,
  STATE_INSIDE_CLOSE_TAG_NAME,
  STATE_INSIDE_PASSTHROUGH,
  STATE_ERROR
} GMarkupParseState;

struct _GMarkupParseContext
{
  const GMarkupParser *parser;

  GMarkupParseFlags flags;

  gint line_number;
  gint char_number;

  gpointer user_data;
  GDestroyNotify dnotify;

  /* A piece of character data or an element that
   * hasn't "ended" yet so we haven't yet called
   * the callback for it.
   */
  GString *partial_chunk;

  GMarkupParseState state;
  GSList *tag_stack;
  gchar **attr_names;
  gchar **attr_values;
  gint cur_attr;
  gint alloc_attrs;

  const gchar *current_text;
  gssize       current_text_len;      
  const gchar *current_text_end;

  GString *leftover_char_portion;

  /* used to save the start of the last interesting thingy */
  const gchar *start;

  const gchar *iter;

  guint document_empty : 1;
  guint parsing : 1;
};

/**
 * g_markup_parse_context_new:
 * @parser: a #GMarkupParser
 * @flags: one or more #GMarkupParseFlags
 * @user_data: user data to pass to #GMarkupParser functions
 * @user_data_dnotify: user data destroy notifier called when the parse context is freed
 * 
 * Creates a new parse context. A parse context is used to parse
 * marked-up documents. You can feed any number of documents into
 * a context, as long as no errors occur; once an error occurs,
 * the parse context can't continue to parse text (you have to free it
 * and create a new parse context).
 * 
 * Return value: a new #GMarkupParseContext
 **/
GMarkupParseContext *
g_markup_parse_context_new (const GMarkupParser *parser,
                            GMarkupParseFlags    flags,
                            gpointer             user_data,
                            GDestroyNotify       user_data_dnotify)
{
  GMarkupParseContext *context;

  g_return_val_if_fail (parser != NULL, NULL);

  context = g_new (GMarkupParseContext, 1);

  context->parser = parser;
  context->flags = flags;
  context->user_data = user_data;
  context->dnotify = user_data_dnotify;

  context->line_number = 1;
  context->char_number = 1;

  context->partial_chunk = NULL;

  context->state = STATE_START;
  context->tag_stack = NULL;
  context->attr_names = NULL;
  context->attr_values = NULL;
  context->cur_attr = -1;
  context->alloc_attrs = 0;

  context->current_text = NULL;
  context->current_text_len = -1;
  context->current_text_end = NULL;
  context->leftover_char_portion = NULL;

  context->start = NULL;
  context->iter = NULL;

  context->document_empty = TRUE;
  context->parsing = FALSE;

  return context;
}

/**
 * g_markup_parse_context_free:
 * @context: a #GMarkupParseContext
 * 
 * Frees a #GMarkupParseContext. Can't be called from inside
 * one of the #GMarkupParser functions.
 * 
 **/
void
g_markup_parse_context_free (GMarkupParseContext *context)
{
  g_return_if_fail (context != NULL);
  g_return_if_fail (!context->parsing);

  if (context->dnotify)
    (* context->dnotify) (context->user_data);

  g_strfreev (context->attr_names);
  g_strfreev (context->attr_values);

  g_slist_foreach (context->tag_stack, (GFunc)g_free, NULL);
  g_slist_free (context->tag_stack);

  if (context->partial_chunk)
    g_string_free (context->partial_chunk, TRUE);

  if (context->leftover_char_portion)
    g_string_free (context->leftover_char_portion, TRUE);

  g_free (context);
}

static void
mark_error (GMarkupParseContext *context,
            GError              *error)
{
  context->state = STATE_ERROR;

  if (context->parser->error)
    (*context->parser->error) (context, error, context->user_data);
}

static void
set_error (GMarkupParseContext *context,
           GError             **error,
           GMarkupError         code,
           const gchar         *format,
           ...)
{
  GError *tmp_error;
  gchar *s;
  va_list args;

  va_start (args, format);
  s = g_strdup_vprintf (format, args);
  va_end (args);

  tmp_error = g_error_new (G_MARKUP_ERROR,
                           code,
                           _("Error on line %d char %d: %s"),
                           context->line_number,
                           context->char_number,
                           s);

  g_free (s);

  mark_error (context, tmp_error);

  g_propagate_error (error, tmp_error);
}

static gboolean
is_name_start_char (gunichar c)
{
  if (g_unichar_isalpha (c) ||
      c == '_' ||
      c == ':')
    return TRUE;
  else
    return FALSE;
}

static gboolean
is_name_char (gunichar c)
{
  if (g_unichar_isalnum (c) ||
      c == '.' ||
      c == '-' ||
      c == '_' ||
      c == ':')
    return TRUE;
  else
    return FALSE;
}


static gchar*
char_str (gunichar c,
          gchar   *buf)
{
  memset (buf, 0, 7);
  g_unichar_to_utf8 (c, buf);
  return buf;
}

static gchar*
utf8_str (const gchar *utf8,
          gchar       *buf)
{
  char_str (g_utf8_get_char (utf8), buf);
  return buf;
}

static void
set_unescape_error (GMarkupParseContext *context,
                    GError             **error,
                    const gchar         *remaining_text,
                    const gchar         *remaining_text_end,
                    GMarkupError         code,
                    const gchar         *format,
                    ...)
{
  GError *tmp_error;
  gchar *s;
  va_list args;
  gint remaining_newlines;
  const gchar *p;

  remaining_newlines = 0;
  p = remaining_text;
  while (p != remaining_text_end)
    {
      if (*p == '\n')
        ++remaining_newlines;
      ++p;
    }

  va_start (args, format);
  s = g_strdup_vprintf (format, args);
  va_end (args);

  tmp_error = g_error_new (G_MARKUP_ERROR,
                           code,
                           _("Error on line %d: %s"),
                           context->line_number - remaining_newlines,
                           s);

  g_free (s);

  mark_error (context, tmp_error);

  g_propagate_error (error, tmp_error);
}

typedef enum
{
  USTATE_INSIDE_TEXT,
  USTATE_AFTER_AMPERSAND,
  USTATE_INSIDE_ENTITY_NAME,
  USTATE_AFTER_CHARREF_HASH
} UnescapeState;

static gboolean
unescape_text (GMarkupParseContext *context,
               const gchar         *text,
               const gchar         *text_end,
               gchar              **unescaped,
               GError             **error)
{
#define MAX_ENT_LEN 5
  GString *str;
  const gchar *p;
  UnescapeState state;
  const gchar *start;

  str = g_string_new ("");

  state = USTATE_INSIDE_TEXT;
  p = text;
  start = p;
  while (p != text_end && context->state != STATE_ERROR)
    {
      g_assert (p < text_end);
      
      switch (state)
        {
        case USTATE_INSIDE_TEXT:
          {
            while (p != text_end && *p != '&')
              p = g_utf8_next_char (p);

            if (p != start)
              {
                g_string_append_len (str, start, p - start);

                start = NULL;
              }
            
            if (p != text_end && *p == '&')
              {
                p = g_utf8_next_char (p);
                state = USTATE_AFTER_AMPERSAND;
              }
          }
          break;

        case USTATE_AFTER_AMPERSAND:
          {
            if (*p == '#')
              {
                p = g_utf8_next_char (p);

                start = p;
                state = USTATE_AFTER_CHARREF_HASH;
              }
            else if (!is_name_start_char (g_utf8_get_char (p)))
              {
                if (*p == ';')
                  {
                    set_unescape_error (context, error,
                                        p, text_end,
                                        G_MARKUP_ERROR_PARSE,
                                        _("Empty entity '&;' seen; valid "
                                          "entities are: &amp; &quot; &lt; &gt; &apos;"));
                  }
                else
                  {
                    gchar buf[7];

                    set_unescape_error (context, error,
                                        p, text_end,
                                        G_MARKUP_ERROR_PARSE,
                                        _("Character '%s' is not valid at "
                                          "the start of an entity name; "
                                          "the & character begins an entity; "
                                          "if this ampersand isn't supposed "
                                          "to be an entity, escape it as "
                                          "&amp;"),
                                        utf8_str (p, buf));
                  }
              }
            else
              {
                start = p;
                state = USTATE_INSIDE_ENTITY_NAME;
              }
          }
          break;


        case USTATE_INSIDE_ENTITY_NAME:
          {
            gchar buf[MAX_ENT_LEN+1] = {
              '\0', '\0', '\0', '\0', '\0', '\0'
            };
            gchar *dest;

            while (p != text_end)
              {
                if (*p == ';')
                  break;
                else if (!is_name_char (*p))
                  {
                    gchar ubuf[7];

                    set_unescape_error (context, error,
                                        p, text_end,
                                        G_MARKUP_ERROR_PARSE,
                                        _("Character '%s' is not valid "
                                          "inside an entity name"),
                                        utf8_str (p, ubuf));
                    break;
                  }

                p = g_utf8_next_char (p);
              }

            if (context->state != STATE_ERROR)
              {
                if (p != text_end)
                  {
                    const gchar *src;
                
                    src = start;
                    dest = buf;
                    while (src != p)
                      {
                        *dest = *src;
                        ++dest;
                        ++src;
                      }

                    /* move to after semicolon */
                    p = g_utf8_next_char (p);
                    start = p;
                    state = USTATE_INSIDE_TEXT;

                    if (strcmp (buf, "lt") == 0)
                      g_string_append_c (str, '<');
                    else if (strcmp (buf, "gt") == 0)
                      g_string_append_c (str, '>');
                    else if (strcmp (buf, "amp") == 0)
                      g_string_append_c (str, '&');
                    else if (strcmp (buf, "quot") == 0)
                      g_string_append_c (str, '"');
                    else if (strcmp (buf, "apos") == 0)
                      g_string_append_c (str, '\'');
                    else
                      {
                        set_unescape_error (context, error,
                                            p, text_end,
                                            G_MARKUP_ERROR_PARSE,
                                            _("Entity name '%s' is not known"),
                                            buf);
                      }
                  }
                else
                  {
                    set_unescape_error (context, error,
                                        /* give line number of the & */
                                        start, text_end,
                                        G_MARKUP_ERROR_PARSE,
                                        _("Entity did not end with a semicolon; "
                                          "most likely you used an ampersand "
                                          "character without intending to start "
                                          "an entity - escape ampersand as &amp;"));
                  }
              }
          }
          break;

        case USTATE_AFTER_CHARREF_HASH:
          {
            gboolean is_hex = FALSE;
            if (*p == 'x')
              {
                is_hex = TRUE;
                p = g_utf8_next_char (p);
                start = p;
              }

            while (p != text_end && *p != ';')
              p = g_utf8_next_char (p);

            if (p != text_end)
              {
                g_assert (*p == ';');

                /* digit is between start and p */

                if (start != p)
                  {
                    gchar *digit = g_strndup (start, p - start);
                    gulong l;
                    gchar *end = NULL;
                    gchar *digit_end = digit + (p - start);
                    
                    errno = 0;
                    if (is_hex)
                      l = strtoul (digit, &end, 16);
                    else
                      l = strtoul (digit, &end, 10);

                    if (end != digit_end || errno != 0)
                      {
                        set_unescape_error (context, error,
                                            start, text_end,
                                            G_MARKUP_ERROR_PARSE,
                                            _("Failed to parse '%s', which "
                                              "should have been a digit "
                                              "inside a character reference "
                                              "(&#234; for example) - perhaps "
                                              "the digit is too large"),
                                            digit);
                      }
                    else
                      {
                        /* characters XML permits */
                        if (l == 0x9 ||
                            l == 0xA ||
                            l == 0xD ||
                            (l >= 0x20 && l <= 0xD7FF) ||
                            (l >= 0xE000 && l <= 0xFFFD) ||
                            (l >= 0x10000 && l <= 0x10FFFF))
                          {
                            gchar buf[7];
                            g_string_append (str, char_str (l, buf));
                          }
                        else
                          {
                            set_unescape_error (context, error,
                                                start, text_end,
                                                G_MARKUP_ERROR_PARSE,
                                                _("Character reference '%s' does not encode a permitted character"),
                                                digit);
                          }
                      }

                    g_free (digit);

                    /* Move to next state */
                    p = g_utf8_next_char (p); /* past semicolon */
                    start = p;
                    state = USTATE_INSIDE_TEXT;
                  }
                else
                  {
                    set_unescape_error (context, error,
                                        start, text_end,
                                        G_MARKUP_ERROR_PARSE,
                                        _("Empty character reference; "
                                          "should include a digit such as "
                                          "&#454;"));
                  }
              }
            else
              {
                set_unescape_error (context, error,
                                    start, text_end,
                                    G_MARKUP_ERROR_PARSE,
                                    _("Character reference did not end with a "
                                      "semicolon; "
                                      "most likely you used an ampersand "
                                      "character without intending to start "
                                      "an entity - escape ampersand as &amp;"));
              }
          }
          break;

        default:
          g_assert_not_reached ();
          break;
        }
    }

  /* If no errors, we should have returned to USTATE_INSIDE_TEXT */
  g_assert (context->state == STATE_ERROR ||
            state == USTATE_INSIDE_TEXT);

  if (context->state == STATE_ERROR)
    {
      g_string_free (str, TRUE);
      *unescaped = NULL;
      return FALSE;
    }
  else
    {
      *unescaped = g_string_free (str, FALSE);
      return TRUE;
    }

#undef MAX_ENT_LEN
}

static gboolean
advance_char (GMarkupParseContext *context)
{

  context->iter = g_utf8_next_char (context->iter);
  context->char_number += 1;
  if (*context->iter == '\n')
    {
      context->line_number += 1;
      context->char_number = 1;
    }

  return context->iter != context->current_text_end;
}

static void
skip_spaces (GMarkupParseContext *context)
{
  do
    {
      if (!g_unichar_isspace (g_utf8_get_char (context->iter)))
        return;
    }
  while (advance_char (context));
}

static void
advance_to_name_end (GMarkupParseContext *context)
{
  do
    {
      if (!is_name_char (g_utf8_get_char (context->iter)))
        return;
    }
  while (advance_char (context));
}

static void
add_to_partial (GMarkupParseContext *context,
                const gchar         *text_start,
                const gchar         *text_end)
{
  if (context->partial_chunk == NULL)
    context->partial_chunk = g_string_new ("");

  if (text_start != text_end)
    g_string_append_len (context->partial_chunk, text_start,
                         text_end - text_start);

  /* Invariant here that partial_chunk exists */
}

static void
truncate_partial (GMarkupParseContext *context)
{
  if (context->partial_chunk != NULL)
    {
      context->partial_chunk = g_string_truncate (context->partial_chunk, 0);
    }
}

static const gchar*
current_element (GMarkupParseContext *context)
{
  return context->tag_stack->data;
}

static const gchar*
current_attribute (GMarkupParseContext *context)
{
  g_assert (context->cur_attr >= 0);
  return context->attr_names[context->cur_attr];
}

static void
find_current_text_end (GMarkupParseContext *context)
{
  /* This function must be safe (non-segfaulting) on invalid UTF8 */
  const gchar *end = context->current_text + context->current_text_len;
  const gchar *p;
  const gchar *next;

  g_assert (context->current_text_len > 0);

  p = context->current_text;
  next = g_utf8_find_next_char (p, end);

  while (next)
    {
      p = next;
      next = g_utf8_find_next_char (p, end);
    }

  /* p is now the start of the last character or character portion. */
  g_assert (p != end);
  next = g_utf8_next_char (p); /* this only touches *p, nothing beyond */

  if (next == end)
    {
      /* whole character */
      context->current_text_end = end;
    }
  else
    {
      /* portion */
      context->leftover_char_portion = g_string_new_len (p, end - p);
      context->current_text_len -= (end - p);
      context->current_text_end = p;
    }
}

static void
add_attribute (GMarkupParseContext *context, char *name)
{
  if (context->cur_attr + 2 >= context->alloc_attrs)
    {
      context->alloc_attrs += 5; /* silly magic number */
      context->attr_names = g_realloc (context->attr_names, sizeof(char*)*context->alloc_attrs);
      context->attr_values = g_realloc (context->attr_values, sizeof(char*)*context->alloc_attrs);
    }
  context->cur_attr++;
  context->attr_names[context->cur_attr] = name;
  context->attr_values[context->cur_attr] = NULL;
  context->attr_names[context->cur_attr+1] = NULL;
}

/**
 * g_markup_parse_context_parse:
 * @context: a #GMarkupParseContext
 * @text: chunk of text to parse
 * @text_len: length of @text in bytes
 * @error: return location for a #GError
 * 
 * Feed some data to the #GMarkupParseContext. The data need not
 * be valid UTF-8; an error will be signaled if it's invalid.
 * The data need not be an entire document; you can feed a document
 * into the parser incrementally, via multiple calls to this function.
 * Typically, as you receive data from a network connection or file,
 * you feed each received chunk of data into this function, aborting
 * the process if an error occurs. Once an error is reported, no further
 * data may be fed to the #GMarkupParseContext; all errors are fatal.
 * 
 * Return value: %FALSE if an error occurred, %TRUE on success
 **/
gboolean
g_markup_parse_context_parse (GMarkupParseContext *context,
                              const gchar         *text,
                              gssize               text_len,
                              GError             **error)
{
  const gchar *first_invalid;
  
  g_return_val_if_fail (context != NULL, FALSE);
  g_return_val_if_fail (text != NULL, FALSE);
  g_return_val_if_fail (context->state != STATE_ERROR, FALSE);
  g_return_val_if_fail (!context->parsing, FALSE);
  
  if (text_len < 0)
    text_len = strlen (text);

  if (text_len == 0)
    return TRUE;
  
  context->parsing = TRUE;
  
  if (context->leftover_char_portion)
    {
      const gchar *first_char;

      if ((*text & 0xc0) != 0x80)
        first_char = text;
      else
        first_char = g_utf8_find_next_char (text, text + text_len);

      if (first_char)
        {
          /* leftover_char_portion was completed. Parse it. */
          GString *portion = context->leftover_char_portion;
          
          g_string_append_len (context->leftover_char_portion,
                               text, first_char - text);

          /* hacks to allow recursion */
          context->parsing = FALSE;
          context->leftover_char_portion = NULL;
          
          if (!g_markup_parse_context_parse (context,
                                             portion->str, portion->len,
                                             error))
            {
              g_assert (context->state == STATE_ERROR);
            }
          
          g_string_free (portion, TRUE);
          context->parsing = TRUE;

          /* Skip the fraction of char that was in this text */
          text_len -= (first_char - text);
          text = first_char;
        }
      else
        {
          /* another little chunk of the leftover char; geez
           * someone is inefficient.
           */
          g_string_append_len (context->leftover_char_portion,
                               text, text_len);

          if (context->leftover_char_portion->len > 7)
            {
              /* The leftover char portion is too big to be
               * a UTF-8 character
               */
              set_error (context,
                         error,
                         G_MARKUP_ERROR_BAD_UTF8,
                         _("Invalid UTF-8 encoded text"));
            }
          
          goto finished;
        }
    }

  context->current_text = text;
  context->current_text_len = text_len;
  context->iter = context->current_text;
  context->start = context->iter;

  /* Nothing left after finishing the leftover char, or nothing
   * passed in to begin with.
   */
  if (context->current_text_len == 0)
    goto finished;

  /* find_current_text_end () assumes the string starts at
   * a character start, so we need to validate at least
   * that much. It doesn't assume any following bytes
   * are valid.
   */
  if ((*context->current_text & 0xc0) == 0x80) /* not a char start */
    {
      set_error (context,
                 error,
                 G_MARKUP_ERROR_BAD_UTF8,
                 _("Invalid UTF-8 encoded text"));
      goto finished;
    }

  /* Initialize context->current_text_end, possibly adjusting
   * current_text_len, and add any leftover char portion
   */
  find_current_text_end (context);

  /* Validate UTF8 (must be done after we find the end, since
   * we could have a trailing incomplete char)
   */
  if (!g_utf8_validate (context->current_text,
                        context->current_text_len,
                        &first_invalid))
    {
      gint newlines = 0;
      const gchar *p;
      p = context->current_text;
      while (p != context->current_text_end)
        {
          if (*p == '\n')
            ++newlines;
          ++p;
        }

      context->line_number += newlines;

      set_error (context,
                 error,
                 G_MARKUP_ERROR_BAD_UTF8,
                 _("Invalid UTF-8 encoded text"));
      goto finished;
    }

  while (context->iter != context->current_text_end)
    {
      switch (context->state)
        {
        case STATE_START:
          /* Possible next state: AFTER_OPEN_ANGLE */

          g_assert (context->tag_stack == NULL);

          /* whitespace is ignored outside of any elements */
          skip_spaces (context);

          if (context->iter != context->current_text_end)
            {
              if (*context->iter == '<')
                {
                  /* Move after the open angle */
                  advance_char (context);

                  context->state = STATE_AFTER_OPEN_ANGLE;

                  /* this could start a passthrough */
                  context->start = context->iter;

                  /* document is now non-empty */
                  context->document_empty = FALSE;
                }
              else
                {
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("Document must begin with an element (e.g. <book>)"));
                }
            }
          break;

        case STATE_AFTER_OPEN_ANGLE:
          /* Possible next states: INSIDE_OPEN_TAG_NAME,
           *  AFTER_CLOSE_TAG_SLASH, INSIDE_PASSTHROUGH
           */
          if (*context->iter == '?' ||
              *context->iter == '!')
            {
              /* include < in the passthrough */
              const gchar *openangle = "<";
              add_to_partial (context, openangle, openangle + 1);
              context->start = context->iter;
              context->state = STATE_INSIDE_PASSTHROUGH;
            }
          else if (*context->iter == '/')
            {
              /* move after it */
              advance_char (context);

              context->state = STATE_AFTER_CLOSE_TAG_SLASH;
            }
          else if (is_name_start_char (g_utf8_get_char (context->iter)))
            {
              context->state = STATE_INSIDE_OPEN_TAG_NAME;

              /* start of tag name */
              context->start = context->iter;
            }
          else
            {
              gchar buf[7];
              set_error (context,
                         error,
                         G_MARKUP_ERROR_PARSE,
                         _("'%s' is not a valid character following "
                           "a '<' character; it may not begin an "
                           "element name"),
                         utf8_str (context->iter, buf));
            }
          break;

          /* The AFTER_CLOSE_ANGLE state is actually sort of
           * broken, because it doesn't correspond to a range
           * of characters in the input stream as the others do,
           * and thus makes things harder to conceptualize
           */
        case STATE_AFTER_CLOSE_ANGLE:
          /* Possible next states: INSIDE_TEXT, STATE_START */
          if (context->tag_stack == NULL)
            {
              context->start = NULL;
              context->state = STATE_START;
            }
          else
            {
              context->start = context->iter;
              context->state = STATE_INSIDE_TEXT;
            }
          break;

        case STATE_AFTER_ELISION_SLASH:
          /* Possible next state: AFTER_CLOSE_ANGLE */

          {
            /* We need to pop the tag stack and call the end_element
             * function, since this is the close tag
             */
            GError *tmp_error = NULL;
          
            g_assert (context->tag_stack != NULL);

            tmp_error = NULL;
            if (context->parser->end_element)
              (* context->parser->end_element) (context,
                                                context->tag_stack->data,
                                                context->user_data,
                                                &tmp_error);
          
            if (tmp_error)
              {
                mark_error (context, tmp_error);
                g_propagate_error (error, tmp_error);
              }          
            else
              {
                if (*context->iter == '>')
                  {
                    /* move after the close angle */
                    advance_char (context);
                    context->state = STATE_AFTER_CLOSE_ANGLE;
                  }
                else
                  {
                    gchar buf[7];
                    set_error (context,
                               error,
                               G_MARKUP_ERROR_PARSE,
                               _("Odd character '%s', expected a '>' character "
                                 "to end the start tag of element '%s'"),
                               utf8_str (context->iter, buf),
                               current_element (context));
                  }
              }

            g_free (context->tag_stack->data);
            context->tag_stack = g_slist_delete_link (context->tag_stack,
                                                      context->tag_stack);
          }
          break;

        case STATE_INSIDE_OPEN_TAG_NAME:
          /* Possible next states: BETWEEN_ATTRIBUTES */

          /* if there's a partial chunk then it's the first part of the
           * tag name. If there's a context->start then it's the start
           * of the tag name in current_text, the partial chunk goes
           * before that start though.
           */
          advance_to_name_end (context);

          if (context->iter == context->current_text_end)
            {
              /* The name hasn't necessarily ended. Merge with
               * partial chunk, leave state unchanged.
               */
              add_to_partial (context, context->start, context->iter);
            }
          else
            {
              /* The name has ended. Combine it with the partial chunk
               * if any; push it on the stack; enter next state.
               */
              add_to_partial (context, context->start, context->iter);
              context->tag_stack =
                g_slist_prepend (context->tag_stack,
                                 g_string_free (context->partial_chunk,
                                                FALSE));

              context->partial_chunk = NULL;

              context->state = STATE_BETWEEN_ATTRIBUTES;
              context->start = NULL;
            }
          break;

        case STATE_INSIDE_ATTRIBUTE_NAME:
          /* Possible next states: AFTER_ATTRIBUTE_EQUALS_SIGN */

          /* read the full name, if we enter the equals sign state
           * then add the attribute to the list (without the value),
           * otherwise store a partial chunk to be prepended later.
           */
          advance_to_name_end (context);

          if (context->iter == context->current_text_end)
            {
              /* The name hasn't necessarily ended. Merge with
               * partial chunk, leave state unchanged.
               */
              add_to_partial (context, context->start, context->iter);
            }
          else
            {
              /* The name has ended. Combine it with the partial chunk
               * if any; push it on the stack; enter next state.
               */
              add_to_partial (context, context->start, context->iter);

              add_attribute (context, g_string_free (context->partial_chunk, FALSE));

              context->partial_chunk = NULL;
              context->start = NULL;

              if (*context->iter == '=')
                {
                  advance_char (context);
                  context->state = STATE_AFTER_ATTRIBUTE_EQUALS_SIGN;
                }
              else
                {
                  gchar buf[7];
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("Odd character '%s', expected a '=' after "
                               "attribute name '%s' of element '%s'"),
                             utf8_str (context->iter, buf),
                             current_attribute (context),
                             current_element (context));

                }
            }
          break;

        case STATE_BETWEEN_ATTRIBUTES:
          /* Possible next states: AFTER_CLOSE_ANGLE,
           * AFTER_ELISION_SLASH, INSIDE_ATTRIBUTE_NAME
           */
          skip_spaces (context);

          if (context->iter != context->current_text_end)
            {
              if (*context->iter == '/')
                {
                  advance_char (context);
                  context->state = STATE_AFTER_ELISION_SLASH;
                }
              else if (*context->iter == '>')
                {

                  advance_char (context);
                  context->state = STATE_AFTER_CLOSE_ANGLE;
                }
              else if (is_name_start_char (g_utf8_get_char (context->iter)))
                {
                  context->state = STATE_INSIDE_ATTRIBUTE_NAME;
                  /* start of attribute name */
                  context->start = context->iter;
                }
              else
                {
                  gchar buf[7];
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("Odd character '%s', expected a '>' or '/' "
                               "character to end the start tag of "
                               "element '%s', or optionally an attribute; "
                               "perhaps you used an invalid character in "
                               "an attribute name"),
                             utf8_str (context->iter, buf),
                             current_element (context));
                }

              /* If we're done with attributes, invoke
               * the start_element callback
               */
              if (context->state == STATE_AFTER_ELISION_SLASH ||
                  context->state == STATE_AFTER_CLOSE_ANGLE)
                {
                  const gchar *start_name;
		  /* Ugly, but the current code expects an empty array instead of NULL */
		  const gchar *empty = NULL;
                  const gchar **attr_names =  &empty;
                  const gchar **attr_values = &empty;
                  GError *tmp_error;

                  /* Call user callback for element start */
                  start_name = current_element (context);

		  if (context->cur_attr >= 0)
		    {
		      attr_names = (const gchar**)context->attr_names;
		      attr_values = (const gchar**)context->attr_values;
		    }

                  tmp_error = NULL;
                  if (context->parser->start_element)
                    (* context->parser->start_element) (context,
                                                        start_name,
                                                        (const gchar **)attr_names,
                                                        (const gchar **)attr_values,
                                                        context->user_data,
                                                        &tmp_error);

                  /* Go ahead and free the attributes. */
		  for (; context->cur_attr >= 0; context->cur_attr--)
		    {
		      int pos = context->cur_attr;
		      g_free (context->attr_names[pos]);
		      g_free (context->attr_values[pos]);
		      context->attr_names[pos] = context->attr_values[pos] = NULL;
		    }
                  context->cur_attr = -1;

                  if (tmp_error != NULL)
                    {
                      mark_error (context, tmp_error);
                      g_propagate_error (error, tmp_error);
                    }
                }
            }
          break;

        case STATE_AFTER_ATTRIBUTE_EQUALS_SIGN:
          /* Possible next state: INSIDE_ATTRIBUTE_VALUE */
          if (*context->iter == '"')
            {
              advance_char (context);
              context->state = STATE_INSIDE_ATTRIBUTE_VALUE;
              context->start = context->iter;
            }
          else
            {
              gchar buf[7];
              set_error (context,
                         error,
                         G_MARKUP_ERROR_PARSE,
                         _("Odd character '%s', expected an open quote mark "
                           "after the equals sign when giving value for "
                           "attribute '%s' of element '%s'"),
                         utf8_str (context->iter, buf),
                         current_attribute (context),
                         current_element (context));
            }
          break;

        case STATE_INSIDE_ATTRIBUTE_VALUE:
          /* Possible next states: BETWEEN_ATTRIBUTES */
          do
            {
              if (*context->iter == '"')
                break;
            }
          while (advance_char (context));

          if (context->iter == context->current_text_end)
            {
              /* The value hasn't necessarily ended. Merge with
               * partial chunk, leave state unchanged.
               */
              add_to_partial (context, context->start, context->iter);
            }
          else
            {
              /* The value has ended at the quote mark. Combine it
               * with the partial chunk if any; set it for the current
               * attribute.
               */
              add_to_partial (context, context->start, context->iter);

              g_assert (context->cur_attr >= 0);
              
              if (unescape_text (context,
                                 context->partial_chunk->str,
                                 context->partial_chunk->str +
                                 context->partial_chunk->len,
                                 &context->attr_values[context->cur_attr],
                                 error))
                {
                  /* success, advance past quote and set state. */
                  advance_char (context);
                  context->state = STATE_BETWEEN_ATTRIBUTES;
                  context->start = NULL;
                }
              
              truncate_partial (context);
            }
          break;

        case STATE_INSIDE_TEXT:
          /* Possible next states: AFTER_OPEN_ANGLE */
          do
            {
              if (*context->iter == '<')
                break;
            }
          while (advance_char (context));

          /* The text hasn't necessarily ended. Merge with
           * partial chunk, leave state unchanged.
           */

          add_to_partial (context, context->start, context->iter);

          if (context->iter != context->current_text_end)
            {
              gchar *unescaped = NULL;

              /* The text has ended at the open angle. Call the text
               * callback.
               */
              
              if (unescape_text (context,
                                 context->partial_chunk->str,
                                 context->partial_chunk->str +
                                 context->partial_chunk->len,
                                 &unescaped,
                                 error))
                {
                  GError *tmp_error = NULL;

                  if (context->parser->text)
                    (*context->parser->text) (context,
                                              unescaped,
                                              strlen (unescaped),
                                              context->user_data,
                                              &tmp_error);
                  
                  g_free (unescaped);

                  if (tmp_error == NULL)
                    {
                      /* advance past open angle and set state. */
                      advance_char (context);
                      context->state = STATE_AFTER_OPEN_ANGLE;
                      /* could begin a passthrough */
                      context->start = context->iter;
                    }
                  else
                    {
                      mark_error (context, tmp_error);
                      g_propagate_error (error, tmp_error);
                    }
                }

              truncate_partial (context);
            }
          break;

        case STATE_AFTER_CLOSE_TAG_SLASH:
          /* Possible next state: INSIDE_CLOSE_TAG_NAME */
          if (is_name_start_char (g_utf8_get_char (context->iter)))
            {
              context->state = STATE_INSIDE_CLOSE_TAG_NAME;

              /* start of tag name */
              context->start = context->iter;
            }
          else
            {
              gchar buf[7];
              set_error (context,
                         error,
                         G_MARKUP_ERROR_PARSE,
                         _("'%s' is not a valid character following "
                           "the characters '</'; '%s' may not begin an "
                           "element name"),
                         utf8_str (context->iter, buf),
                         utf8_str (context->iter, buf));
            }
          break;

        case STATE_INSIDE_CLOSE_TAG_NAME:
          /* Possible next state: AFTER_CLOSE_ANGLE */
          advance_to_name_end (context);

          if (context->iter == context->current_text_end)
            {
              /* The name hasn't necessarily ended. Merge with
               * partial chunk, leave state unchanged.
               */
              add_to_partial (context, context->start, context->iter);
            }
          else
            {
              /* The name has ended. Combine it with the partial chunk
               * if any; check that it matches stack top and pop
               * stack; invoke proper callback; enter next state.
               */
              gchar *close_name;

              add_to_partial (context, context->start, context->iter);

              close_name = g_string_free (context->partial_chunk, FALSE);
              context->partial_chunk = NULL;
              
              if (context->tag_stack == NULL)
                {
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("Element '%s' was closed, no element "
                               "is currently open"),
                             close_name);
                }
              else if (strcmp (close_name, current_element (context)) != 0)
                {
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("Element '%s' was closed, but the currently "
                               "open element is '%s'"),
                             close_name,
                             current_element (context));
                }
              else if (*context->iter != '>')
                {
                  gchar buf[7];
                  set_error (context,
                             error,
                             G_MARKUP_ERROR_PARSE,
                             _("'%s' is not a valid character following "
                               "the close element name '%s'; the allowed "
                               "character is '>'"),
                             utf8_str (context->iter, buf),
                             close_name);
                }
              else
                {
                  GError *tmp_error;
                  advance_char (context);
                  context->state = STATE_AFTER_CLOSE_ANGLE;
                  context->start = NULL;

                  /* call the end_element callback */
                  tmp_error = NULL;
                  if (context->parser->end_element)
                    (* context->parser->end_element) (context,
                                                      close_name,
                                                      context->user_data,
                                                      &tmp_error);

                  
                  /* Pop the tag stack */
                  g_free (context->tag_stack->data);
                  context->tag_stack = g_slist_delete_link (context->tag_stack,
                                                            context->tag_stack);
                  
                  if (tmp_error)
                    {
                      mark_error (context, tmp_error);
                      g_propagate_error (error, tmp_error);
                    }
                }

              g_free (close_name);
            }
          break;

        case STATE_INSIDE_PASSTHROUGH:
          /* Possible next state: AFTER_CLOSE_ANGLE */
          do
            {
              if (*context->iter == '>')
                break;
            }
          while (advance_char (context));

          if (context->iter == context->current_text_end)
            {
              /* The passthrough hasn't necessarily ended. Merge with
               * partial chunk, leave state unchanged.
               */
              add_to_partial (context, context->start, context->iter);
            }
          else
            {
              /* The passthrough has ended at the close angle. Combine
               * it with the partial chunk if any. Call the passthrough
               * callback. Note that the open/close angles are
               * included in the text of the passthrough.
               */
              GError *tmp_error = NULL;

              advance_char (context); /* advance past close angle */
              add_to_partial (context, context->start, context->iter);

              if (context->parser->passthrough)
                (*context->parser->passthrough) (context,
                                                 context->partial_chunk->str,
                                                 context->partial_chunk->len,
                                                 context->user_data,
                                                 &tmp_error);
                  
              truncate_partial (context);

              if (tmp_error == NULL)
                {
                  context->state = STATE_AFTER_CLOSE_ANGLE;
                  context->start = context->iter; /* could begin text */
                }
              else
                {
                  mark_error (context, tmp_error);
                  g_propagate_error (error, tmp_error);
                }
            }
          break;

        case STATE_ERROR:
          goto finished;
          break;

        default:
          g_assert_not_reached ();
          break;
        }
    }

 finished:
  context->parsing = FALSE;

  return context->state != STATE_ERROR;
}

/**
 * g_markup_parse_context_end_parse:
 * @context: a #GMarkupParseContext
 * @error: return location for a #GError
 * 
 * Signals to the #GMarkupParseContext that all data has been
 * fed into the parse context with g_markup_parse_context_parse().
 * This function reports an error if the document isn't complete,
 * for example if elements are still open.
 * 
 * Return value: %TRUE on success, %FALSE if an error was set
 **/
gboolean
g_markup_parse_context_end_parse (GMarkupParseContext *context,
                                  GError             **error)
{
  g_return_val_if_fail (context != NULL, FALSE);
  g_return_val_if_fail (!context->parsing, FALSE);
  g_return_val_if_fail (context->state != STATE_ERROR, FALSE);

  if (context->partial_chunk != NULL)
    {
      g_string_free (context->partial_chunk, TRUE);
      context->partial_chunk = NULL;
    }

  if (context->document_empty)
    {
      set_error (context, error, G_MARKUP_ERROR_EMPTY,
                 _("Document was empty or contained only whitespace"));
      return FALSE;
    }
  
  context->parsing = TRUE;
  
  switch (context->state)
    {
    case STATE_START:
      /* Nothing to do */
      break;

    case STATE_AFTER_OPEN_ANGLE:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly just after an open angle bracket '<'"));
      break;

    case STATE_AFTER_CLOSE_ANGLE:
      if (context->tag_stack != NULL)
        {
          /* Error message the same as for INSIDE_TEXT */
          set_error (context, error, G_MARKUP_ERROR_PARSE,
                     _("Document ended unexpectedly with elements still open - "
                       "'%s' was the last element opened"),
                     current_element (context));
        }
      break;
      
    case STATE_AFTER_ELISION_SLASH:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly, expected to see a close angle "
                   "bracket ending the tag <%s/>"), current_element (context));
      break;

    case STATE_INSIDE_OPEN_TAG_NAME:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly inside an element name"));
      break;

    case STATE_INSIDE_ATTRIBUTE_NAME:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly inside an attribute name"));
      break;

    case STATE_BETWEEN_ATTRIBUTES:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly inside an element-opening "
                   "tag."));
      break;

    case STATE_AFTER_ATTRIBUTE_EQUALS_SIGN:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly after the equals sign "
                   "following an attribute name; no attribute value"));
      break;

    case STATE_INSIDE_ATTRIBUTE_VALUE:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly while inside an attribute "
                   "value"));
      break;

    case STATE_INSIDE_TEXT:
      g_assert (context->tag_stack != NULL);
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly with elements still open - "
                   "'%s' was the last element opened"),
                 current_element (context));
      break;

    case STATE_AFTER_CLOSE_TAG_SLASH:
    case STATE_INSIDE_CLOSE_TAG_NAME:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly inside the close tag for "
                   "element '%s'"), current_element);
      break;

    case STATE_INSIDE_PASSTHROUGH:
      set_error (context, error, G_MARKUP_ERROR_PARSE,
                 _("Document ended unexpectedly inside a comment or "
                   "processing instruction"));
      break;

    case STATE_ERROR:
    default:
      g_assert_not_reached ();
      break;
    }

  context->parsing = FALSE;

  return context->state != STATE_ERROR;
}

/**
 * g_markup_parse_context_get_position:
 * @context: a #GMarkupParseContext
 * @line_number: return location for a line number, or %NULL
 * @char_number: return location for a char-on-line number, or %NULL
 *
 * Retrieves the current line number and the number of the character on
 * that line. Intended for use in error messages; there are no strict
 * semantics for what constitutes the "current" line number other than
 * "the best number we could come up with for error messages."
 * 
 **/
void
g_markup_parse_context_get_position (GMarkupParseContext *context,
                                     gint                *line_number,
                                     gint                *char_number)
{
  g_return_if_fail (context != NULL);

  if (line_number)
    *line_number = context->line_number;

  if (char_number)
    *char_number = context->char_number;
}

static void
append_escaped_text (GString     *str,
                     const gchar *text,
                     gssize       length)    
{
  const gchar *p;
  const gchar *end;

  p = text;
  end = text + length;

  while (p != end)
    {
      const gchar *next;
      next = g_utf8_next_char (p);

      switch (*p)
        {
        case '&':
          g_string_append (str, "&amp;");
          break;

        case '<':
          g_string_append (str, "&lt;");
          break;

        case '>':
          g_string_append (str, "&gt;");
          break;

        case '\'':
          g_string_append (str, "&apos;");
          break;

        case '"':
          g_string_append (str, "&quot;");
          break;

        default:
          g_string_append_len (str, p, next - p);
          break;
        }

      p = next;
    }
}

/**
 * g_markup_escape_text:
 * @text: some valid UTF-8 text
 * @length: length of @text in bytes
 * 
 * Escapes text so that the markup parser will parse it verbatim.
 * Less than, greater than, ampersand, etc. are replaced with the
 * corresponding entities. This function would typically be used
 * when writing out a file to be parsed with the markup parser.
 * 
 * Return value: escaped text
 **/
gchar*
g_markup_escape_text (const gchar *text,
                      gssize       length)  
{
  GString *str;

  g_return_val_if_fail (text != NULL, NULL);

  if (length < 0)
    length = strlen (text);

  str = g_string_new ("");
  append_escaped_text (str, text, length);

  return g_string_free (str, FALSE);
}
