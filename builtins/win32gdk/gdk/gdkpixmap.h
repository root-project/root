#ifndef __GDK_PIXMAP_H__
#define __GDK_PIXMAP_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* Pixmaps
 */
   GdkPixmap *gdk_pixmap_new(GdkWindow * window,
                             gint width, gint height, gint depth);
#ifdef GDK_WINDOWING_WIN32
   GdkPixmap *gdk_pixmap_create_on_shared_image
       (GdkImage ** image_return,
        GdkWindow * window,
        GdkVisual * visual, gint width, gint height, gint depth);
#endif
   GdkBitmap *gdk_bitmap_create_from_data(GdkWindow * window,
                                          const gchar * data,
                                          gint width, gint height);
   GdkPixmap *gdk_pixmap_create_from_data(GdkWindow * window,
                                          const gchar * data,
                                          gint width,
                                          gint height,
                                          gint depth,
                                          GdkColor * fg, GdkColor * bg);
   GdkPixmap *gdk_pixmap_create_from_xpm(GdkWindow * window,
                                         GdkBitmap ** mask,
                                         GdkColor * transparent_color,
                                         const gchar * filename);
   GdkPixmap *gdk_pixmap_colormap_create_from_xpm
       (GdkWindow * window,
        GdkColormap * colormap,
        GdkBitmap ** mask,
        GdkColor * transparent_color, const gchar * filename);
   GdkPixmap *gdk_pixmap_create_from_xpm_d(GdkWindow * window,
                                           GdkBitmap ** mask,
                                           GdkColor * transparent_color,
                                           gchar ** data);
   GdkPixmap *gdk_pixmap_colormap_create_from_xpm_d
       (GdkWindow * window,
        GdkColormap * colormap,
        GdkBitmap ** mask, GdkColor * transparent_color, gchar ** data);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_PIXMAP_H__ */
