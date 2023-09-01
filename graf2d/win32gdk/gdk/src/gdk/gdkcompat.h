#ifndef __GDK_COMPAT_H__
#define __GDK_COMPAT_H__


#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* use -DGDK_DISABLE_COMPAT_H to compile your code and asure that it
 * works with future GTK+ versions as well.
 */
#ifndef	GDK_DISABLE_COMPAT_H

#define GdkWindowType                  GdkDrawableType

#define gdk_draw_pixmap                gdk_draw_drawable
#define gdk_draw_bitmap                gdk_draw_drawable

#define gdk_window_get_size            gdk_drawable_get_size
#define gdk_window_get_type            gdk_drawable_get_type
#define gdk_window_get_colormap        gdk_drawable_get_colormap
#define gdk_window_set_colormap        gdk_drawable_set_colormap
#define gdk_window_get_visual          gdk_drawable_get_visual

#define gdk_window_ref                 gdk_drawable_ref
#define gdk_window_unref               gdk_drawable_unref
#define gdk_bitmap_ref                 gdk_drawable_ref
#define gdk_bitmap_unref               gdk_drawable_unref
#define gdk_pixmap_ref                 gdk_drawable_ref
#define gdk_pixmap_unref               gdk_drawable_unref

#define gdk_window_copy_area(drawable,gc,x,y,source_drawable,source_x,source_y,width,height) \
   gdk_draw_pixmap(drawable,gc,source_drawable,source_x,source_y,x,y,width,height)

#define gdk_gc_destroy                 gdk_gc_unref
#define gdk_image_destroy              gdk_image_unref
#define gdk_cursor_destroy             gdk_cursor_unref


#define GDK_WINDOW_PIXMAP GDK_DRAWABLE_PIXMAP

#endif                          /* GDK_DISABLE_COMPAT_H */

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_COMPAT_H__ */
