#ifndef __GDK_PROPERTY_H__
#define __GDK_PROPERTY_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef enum {
      GDK_PROP_MODE_REPLACE,
      GDK_PROP_MODE_PREPEND,
      GDK_PROP_MODE_APPEND
   } GdkPropMode;

   GdkAtom gdk_atom_intern(const gchar * atom_name,
                           gboolean only_if_exists);
   gchar *gdk_atom_name(GdkAtom atom);

   gboolean gdk_property_get(GdkWindow * window,
                             GdkAtom property,
                             GdkAtom type,
                             gulong offset,
                             gulong length,
                             gint pdelete,
                             GdkAtom * actual_property_type,
                             gint * actual_format,
                             gint * actual_length, guchar ** data);
   void gdk_property_change(GdkWindow * window,
                            GdkAtom property,
                            GdkAtom type,
                            gint format,
                            GdkPropMode mode,
                            const guchar * data, gint nelements);
   void gdk_property_delete(GdkWindow * window, GdkAtom property);

   gint gdk_text_property_to_text_list(GdkAtom encoding,
                                       gint format,
                                       const guchar * text,
                                       gint length, gchar *** list);
   void gdk_free_text_list(gchar ** list);
   gint gdk_string_to_compound_text(const gchar * str,
                                    GdkAtom * encoding,
                                    gint * format,
                                    guchar ** ctext, gint * length);
   void gdk_free_compound_text(guchar * ctext);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_PROPERTY_H__ */
