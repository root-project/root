#ifndef __GDK_SELECTION_H__
#define __GDK_SELECTION_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* The next three types define enums for predefined atoms relating
   to selections. In general, one will need to use gdk_intern_atom */

   typedef enum {
      GDK_SELECTION_PRIMARY = 1,
      GDK_SELECTION_SECONDARY = 2
   } GdkSelection;

   typedef enum {
      GDK_TARGET_BITMAP = 5,
      GDK_TARGET_COLORMAP = 7,
      GDK_TARGET_DRAWABLE = 17,
      GDK_TARGET_PIXMAP = 20,
      GDK_TARGET_STRING = 31
   } GdkTarget;

   typedef enum {
      GDK_SELECTION_TYPE_ATOM = 4,
      GDK_SELECTION_TYPE_BITMAP = 5,
      GDK_SELECTION_TYPE_COLORMAP = 7,
      GDK_SELECTION_TYPE_DRAWABLE = 17,
      GDK_SELECTION_TYPE_INTEGER = 19,
      GDK_SELECTION_TYPE_PIXMAP = 20,
      GDK_SELECTION_TYPE_WINDOW = 33,
      GDK_SELECTION_TYPE_STRING = 31
   } GdkSelectionType;

/* Selections
 */
   gboolean gdk_selection_owner_set(GdkWindow * owner,
                                    GdkAtom selection,
                                    guint32 time, gboolean send_event);
   GdkWindow *gdk_selection_owner_get(GdkAtom selection);
   void gdk_selection_convert(GdkWindow * requestor,
                              GdkAtom selection,
                              GdkAtom target, guint32 time);
   gboolean gdk_selection_property_get(GdkWindow * requestor,
                                       guchar ** data,
                                       GdkAtom * prop_type,
                                       gint * prop_format);
   void gdk_selection_send_notify(gulong requestor,
                                  GdkAtom selection,
                                  GdkAtom target,
                                  GdkAtom property, guint32 time);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_SELECTION_H__ */
