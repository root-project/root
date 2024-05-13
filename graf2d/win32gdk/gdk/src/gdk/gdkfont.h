#ifndef __GDK_FONT_H__
#define __GDK_FONT_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* Types of font.
 *   GDK_FONT_FONT: the font is an XFontStruct.
 *   GDK_FONT_FONTSET: the font is an XFontSet used for I18N.
 */
   typedef enum {
      GDK_FONT_FONT,
      GDK_FONT_FONTSET
   } GdkFontType;

   struct _GdkFont {
      GdkFontType type;
      gint ascent;
      gint descent;
   };

   gchar **gdk_font_list_new(const gchar * font_pattern,
                             gint * n_returned);
   void gdk_font_list_free(gchar ** font_list);
//-----------------------------------------------------------------------------
   GdkFont *gdk_font_load(const gchar * font_name);
   GdkFont *gdk_fontset_load(const gchar * fontset_name);
   GdkFont *gdk_font_ref(GdkFont * font);
   void gdk_font_unref(GdkFont * font);
   glong gdk_font_id(const GdkFont * font);
   gboolean gdk_font_equal(const GdkFont * fonta, const GdkFont * fontb);

   gint gdk_string_width(GdkFont * font, const gchar * string);
   gint gdk_text_width(GdkFont * font,
                       const gchar * text, gint text_length);
   gint gdk_text_width_wc(GdkFont * font,
                          const GdkWChar * text, gint text_length);
   gint gdk_char_width(GdkFont * font, gchar character);
   gint gdk_char_width_wc(GdkFont * font, GdkWChar character);
   gint gdk_string_measure(GdkFont * font, const gchar * string);
   gint gdk_text_measure(GdkFont * font,
                         const gchar * text, gint text_length);
   gint gdk_char_measure(GdkFont * font, gchar character);
   gint gdk_string_height(GdkFont * font, const gchar * string);
   gint gdk_text_height(GdkFont * font,
                        const gchar * text, gint text_length);
   gint gdk_char_height(GdkFont * font, gchar character);

   void gdk_text_extents(GdkFont * font,
                         const gchar * text,
                         gint text_length,
                         gint * lbearing,
                         gint * rbearing,
                         gint * width, gint * ascent, gint * descent);
   void gdk_text_extents_wc(GdkFont * font,
                            const GdkWChar * text,
                            gint text_length,
                            gint * lbearing,
                            gint * rbearing,
                            gint * width, gint * ascent, gint * descent);
   void gdk_string_extents(GdkFont * font,
                           const gchar * string,
                           gint * lbearing,
                           gint * rbearing,
                           gint * width, gint * ascent, gint * descent);

   gchar *gdk_font_full_name_get(GdkFont * font);
   void gdk_font_full_name_free(gchar * name);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_FONT_H__ */
