/* gmarkup.h - Simple XML-like string parser/writer
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

#ifndef __G_MARKUP_H__
#define __G_MARKUP_H__

#include <gerror.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
  G_MARKUP_ERROR_BAD_UTF8,
  G_MARKUP_ERROR_EMPTY,
  G_MARKUP_ERROR_PARSE,
  /* These three are primarily intended for specific GMarkupParser
   * implementations to set.
   */
  G_MARKUP_ERROR_UNKNOWN_ELEMENT,
  G_MARKUP_ERROR_UNKNOWN_ATTRIBUTE,
  G_MARKUP_ERROR_INVALID_CONTENT
} GMarkupError;

#define G_MARKUP_ERROR g_markup_error_quark ()

GQuark g_markup_error_quark ();

typedef enum
{
  /* Hmm, can't think of any at the moment */
  G_MARKUP_DO_NOT_USE_THIS_UNSUPPORTED_FLAG = 1 << 0
  
} GMarkupParseFlags;

typedef struct _GMarkupParseContext GMarkupParseContext;
typedef struct _GMarkupParser GMarkupParser;

struct _GMarkupParser
{
  /* Called for open tags <foo bar="baz"> */
  void (*start_element)  (GMarkupParseContext *context,
                          const gchar         *element_name,
                          const gchar        **attribute_names,
                          const gchar        **attribute_values,
                          gpointer             user_data,
                          GError             **error);

  /* Called for close tags </foo> */
  void (*end_element)    (GMarkupParseContext *context,
                          const gchar         *element_name,
                          gpointer             user_data,
                          GError             **error);

  /* Called for character data */
  /* text is not nul-terminated */
  void (*text)           (GMarkupParseContext *context,
                          const gchar         *text,
                          gint                 text_len,
                          gpointer             user_data,
                          GError             **error);

  /* Called for strings that should be re-saved verbatim in this same
   * position, but are not otherwise interpretable.  At the moment
   * this includes comments and processing instructions.
   */
  /* text is not nul-terminated. */
  void (*passthrough)    (GMarkupParseContext *context,
                          const gchar         *passthrough_text,
                          gint                 text_len,
                          gpointer             user_data,
                          GError             **error);

  /* Called on error, including one set by other
   * methods in the vtable. The GError should not be freed.
   */
  void (*error)          (GMarkupParseContext *context,
                          GError              *error,
                          gpointer             user_data);
};

GMarkupParseContext *g_markup_parse_context_new   (const GMarkupParser *parser,
                                                   GMarkupParseFlags    flags,
                                                   gpointer             user_data,
                                                   GDestroyNotify       user_data_dnotify);
void                 g_markup_parse_context_free  (GMarkupParseContext *context);
gboolean             g_markup_parse_context_parse (GMarkupParseContext *context,
                                                   const gchar         *text,
                                                   gint                 text_len,
                                                   GError             **error);
                                                   
gboolean             g_markup_parse_context_end_parse (GMarkupParseContext *context,
                                                       GError             **error);

/* For user-constructed error messages, has no precise semantics */
void                 g_markup_parse_context_get_position (GMarkupParseContext *context,
                                                          gint                *line_number,
                                                          gint                *char_number);

/* useful when saving */
gchar* g_markup_escape_text (const gchar *text,
                             gint         length);


#ifdef __cplusplus
}
#endif

#endif /* __G_MARKUP_H__ */

