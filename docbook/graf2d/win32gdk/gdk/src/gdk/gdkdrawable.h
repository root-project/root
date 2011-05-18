#ifndef __GDK_DRAWABLE_H__
#define __GDK_DRAWABLE_H__

#include <gdk/gdktypes.h>
#include <gdk/gdkgc.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkDrawableClass GdkDrawableClass;

/* Types of windows.
 *   Root: There is only 1 root window and it is initialized
 *	   at startup. Creating a window of type GDK_WINDOW_ROOT
 *	   is an error.
 *   Toplevel: Windows which interact with the window manager.
 *   Child: Windows which are children of some other type of window.
 *	    (Any other type of window). Most windows are child windows.
 *   Dialog: A special kind of toplevel window which interacts with
 *	     the window manager slightly differently than a regular
 *	     toplevel window. Dialog windows should be used for any
 *	     transient window.
 *   Pixmap: Pixmaps are really just another kind of window which
 *	     doesn't actually appear on the screen. It can't have
 *	     children, either and is really just a convenience so
 *	     that the drawing functions can work on both windows
 *	     and pixmaps transparently. (ie. You shouldn't pass a
 *	     pixmap to any procedure which accepts a window with the
 *	     exception of the drawing functions).
 *   Foreign: A window that actually belongs to another application
 */
   typedef enum {
      GDK_WINDOW_ROOT,
      GDK_WINDOW_TOPLEVEL,
      GDK_WINDOW_CHILD,
      GDK_WINDOW_DIALOG,
      GDK_WINDOW_TEMP,
      GDK_DRAWABLE_PIXMAP,
      GDK_WINDOW_FOREIGN
   } GdkDrawableType;

   struct _GdkDrawable {
      gpointer user_data;
   };

   struct _GdkDrawableClass {
      void (*destroy) (GdkDrawable * drawable);
      GdkGC *(*create_gc) (GdkDrawable * drawable,
                           GdkGCValues * values, GdkGCValuesMask mask);
      void (*draw_rectangle) (GdkDrawable * drawable,
                              GdkGC * gc,
                              gint filled,
                              gint x, gint y, gint width, gint height);
      void (*draw_arc) (GdkDrawable * drawable,
                        GdkGC * gc,
                        gint filled,
                        gint x,
                        gint y,
                        gint width, gint height, gint angle1, gint angle2);
      void (*draw_polygon) (GdkDrawable * drawable,
                            GdkGC * gc,
                            gint filled, GdkPoint * points, gint npoints);
      void (*draw_text) (GdkDrawable * drawable,
                         GdkFont * font,
                         GdkGC * gc,
                         gint x,
                         gint y, const gchar * text, gint text_length);
      void (*draw_text_wc) (GdkDrawable * drawable,
                            GdkFont * font,
                            GdkGC * gc,
                            gint x,
                            gint y,
                            const GdkWChar * text, gint text_length);
      void (*draw_drawable) (GdkDrawable * drawable,
                             GdkGC * gc,
                             GdkDrawable * src,
                             gint xsrc,
                             gint ysrc,
                             gint xdest,
                             gint ydest, gint width, gint height);
      void (*draw_points) (GdkDrawable * drawable,
                           GdkGC * gc, GdkPoint * points, gint npoints);
      void (*draw_segments) (GdkDrawable * drawable,
                             GdkGC * gc, GdkSegment * segs, gint nsegs);
      void (*draw_lines) (GdkDrawable * drawable,
                          GdkGC * gc, GdkPoint * points, gint npoints);
   };

/* Manipulation of drawables
 */
   GdkDrawable *gdk_drawable_alloc(void);

   GdkDrawableType gdk_drawable_get_type(GdkDrawable * window);

   void gdk_drawable_set_data(GdkDrawable * drawable,
                              const gchar * key,
                              gpointer data, GDestroyNotify destroy_func);
   void gdk_drawable_get_data(GdkDrawable * drawable, const gchar * key);

   void gdk_drawable_get_size(GdkWindow * drawable,
                              gint * width, gint * height);
   void gdk_drawable_set_colormap(GdkDrawable * drawable,
                                  GdkColormap * colormap);
   GdkColormap *gdk_drawable_get_colormap(GdkDrawable * drawable);
   GdkVisual *gdk_drawable_get_visual(GdkDrawable * drawable);
   GdkDrawable *gdk_drawable_ref(GdkDrawable * drawable);
   void gdk_drawable_unref(GdkDrawable * drawable);

/* Drawing
 */
   void gdk_draw_point(GdkDrawable * drawable, GdkGC * gc, gint x, gint y);
   void gdk_draw_line(GdkDrawable * drawable,
                      GdkGC * gc, gint x1, gint y1, gint x2, gint y2);
   void gdk_draw_rectangle(GdkDrawable * drawable,
                           GdkGC * gc,
                           gint filled,
                           gint x, gint y, gint width, gint height);
   void gdk_draw_arc(GdkDrawable * drawable,
                     GdkGC * gc,
                     gint filled,
                     gint x,
                     gint y,
                     gint width, gint height, gint angle1, gint angle2);
   void gdk_draw_polygon(GdkDrawable * drawable,
                         GdkGC * gc,
                         gint filled, GdkPoint * points, gint npoints);
   void gdk_draw_string(GdkDrawable * drawable,
                        GdkFont * font,
                        GdkGC * gc, gint x, gint y, const gchar * string);
   void gdk_draw_text(GdkDrawable * drawable,
                      GdkFont * font,
                      GdkGC * gc,
                      gint x,
                      gint y, const gchar * text, gint text_length);
   void gdk_draw_text_wc(GdkDrawable * drawable,
                         GdkFont * font,
                         GdkGC * gc,
                         gint x,
                         gint y, const GdkWChar * text, gint text_length);
   void gdk_draw_drawable(GdkDrawable * drawable,
                          GdkGC * gc,
                          GdkDrawable * src,
                          gint xsrc,
                          gint ysrc,
                          gint xdest, gint ydest, gint width, gint height);
   void gdk_draw_image(GdkDrawable * drawable,
                       GdkGC * gc,
                       GdkImage * image,
                       gint xsrc,
                       gint ysrc,
                       gint xdest, gint ydest, gint width, gint height);
   void gdk_draw_points(GdkDrawable * drawable,
                        GdkGC * gc, GdkPoint * points, gint npoints);
   void gdk_draw_segments(GdkDrawable * drawable,
                          GdkGC * gc, GdkSegment * segs, gint nsegs);
   void gdk_draw_lines(GdkDrawable * drawable,
                       GdkGC * gc, GdkPoint * points, gint npoints);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_DRAWABLE_H__ */
