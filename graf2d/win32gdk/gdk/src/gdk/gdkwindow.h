#ifndef __GDK_WINDOW_H__
#define __GDK_WINDOW_H__

#include <gdk/gdkdrawable.h>
#include <gdk/gdktypes.h>
#include <gdk/gdkwindow.h>
#include <gdk/gdkevents.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkGeometry GdkGeometry;
   typedef struct _GdkWindowAttr GdkWindowAttr;

/* Classes of windows.
 *   InputOutput: Almost every window should be of this type. Such windows
 *		  receive events and are also displayed on screen.
 *   InputOnly: Used only in special circumstances when events need to be
 *		stolen from another window or windows. Input only windows
 *		have no visible output, so they are handy for placing over
 *		top of a group of windows in order to grab the events (or
 *		filter the events) from those windows.
 */
   typedef enum {
      GDK_INPUT_OUTPUT,
      GDK_INPUT_ONLY
   } GdkWindowClass;


/* Window attribute mask values.
 *   GDK_WA_TITLE: The "title" field is valid.
 *   GDK_WA_X: The "x" field is valid.
 *   GDK_WA_Y: The "y" field is valid.
 *   GDK_WA_CURSOR: The "cursor" field is valid.
 *   GDK_WA_COLORMAP: The "colormap" field is valid.
 *   GDK_WA_VISUAL: The "visual" field is valid.
 */
   typedef enum {
      GDK_WA_TITLE = 1 << 1,
      GDK_WA_X = 1 << 2,
      GDK_WA_Y = 1 << 3,
      GDK_WA_CURSOR = 1 << 4,
      GDK_WA_COLORMAP = 1 << 5,
      GDK_WA_VISUAL = 1 << 6,
      GDK_WA_WMCLASS = 1 << 7,
      GDK_WA_NOREDIR = 1 << 8
   } GdkWindowAttributesType;

/* Size restriction enumeration.
 */
   typedef enum {
      GDK_HINT_POS = 1 << 0,
      GDK_HINT_MIN_SIZE = 1 << 1,
      GDK_HINT_MAX_SIZE = 1 << 2,
      GDK_HINT_BASE_SIZE = 1 << 3,
      GDK_HINT_ASPECT = 1 << 4,
      GDK_HINT_RESIZE_INC = 1 << 5
   } GdkWindowHints;

/* The next two enumeration values current match the
 * Motif constants. If this is changed, the implementation
 * of gdk_window_set_decorations/gdk_window_set_functions
 * will need to change as well.
 */
   typedef enum {
      GDK_DECOR_ALL = 1 << 0,
      GDK_DECOR_BORDER = 1 << 1,
      GDK_DECOR_RESIZEH = 1 << 2,
      GDK_DECOR_TITLE = 1 << 3,
      GDK_DECOR_MENU = 1 << 4,
      GDK_DECOR_MINIMIZE = 1 << 5,
      GDK_DECOR_MAXIMIZE = 1 << 6
   } GdkWMDecoration;

   typedef enum {
      GDK_FUNC_ALL = 1 << 0,
      GDK_FUNC_RESIZE = 1 << 1,
      GDK_FUNC_MOVE = 1 << 2,
      GDK_FUNC_MINIMIZE = 1 << 3,
      GDK_FUNC_MAXIMIZE = 1 << 4,
      GDK_FUNC_CLOSE = 1 << 5
   } GdkWMFunction;

   struct _GdkWindowAttr {
      gchar *title;
      gint event_mask;
      gint16 x, y;
      gint16 width;
      gint16 height;
      GdkWindowClass wclass;
      GdkVisual *visual;
      GdkColormap *colormap;
      GdkDrawableType window_type;
      GdkCursor *cursor;
      gchar *wmclass_name;
      gchar *wmclass_class;
      gboolean override_redirect;
   };

   struct _GdkGeometry {
      gint min_width;
      gint min_height;
      gint max_width;
      gint max_height;
      gint base_width;
      gint base_height;
      gint width_inc;
      gint height_inc;
      gdouble min_aspect;
      gdouble max_aspect;
      /* GdkGravity gravity; */
   };

/* Windows
 */
   GdkWindow *gdk_window_new(GdkWindow * parent,
                             GdkWindowAttr * attributes,
                             gint attributes_mask);

   void gdk_window_destroy(GdkWindow * window, gboolean xdestroy);

   GdkWindow *gdk_window_at_pointer(gint * win_x, gint * win_y);
   void gdk_window_show(GdkWindow * window);
   void gdk_window_hide(GdkWindow * window);
   void gdk_window_withdraw(GdkWindow * window);
   void gdk_window_move(GdkWindow * window, gint x, gint y);
   void gdk_window_resize(GdkWindow * window, gint width, gint height);
   void gdk_window_move_resize(GdkWindow * window,
                               gint x, gint y, gint width, gint height);
   void gdk_window_reparent(GdkWindow * window,
                            GdkWindow * new_parent, gint x, gint y);
   void gdk_window_clear(GdkWindow * window);
   void gdk_window_clear_area(GdkWindow * window,
                              gint x, gint y, gint width, gint height);
   void gdk_window_clear_area_e(GdkWindow * window,
                                gint x, gint y, gint width, gint height);
   void gdk_window_raise(GdkWindow * window);
   void gdk_window_lower(GdkWindow * window);

   void gdk_window_set_user_data(GdkWindow * window, gpointer user_data);
   void gdk_window_set_override_redirect(GdkWindow * window,
                                         gboolean override_redirect);

   void gdk_window_add_filter(GdkWindow * window,
                              GdkFilterFunc function, gpointer data);
   void gdk_window_remove_filter(GdkWindow * window,
                                 GdkFilterFunc function, gpointer data);

/* 
 * This allows for making shaped (partially transparent) windows
 * - cool feature, needed for Drag and Drag for example.
 *  The shape_mask can be the mask
 *  from gdk_pixmap_create_from_xpm.   Stefan Wille
 */
   void gdk_window_shape_combine_mask(GdkWindow * window,
                                      GdkBitmap * shape_mask,
                                      gint offset_x, gint offset_y);
/*
 * This routine allows you to quickly take the shapes of all the child windows
 * of a window and use their shapes as the shape mask for this window - useful
 * for container windows that dont want to look like a big box
 * 
 * - Raster
 */
   void gdk_window_set_child_shapes(GdkWindow * window);

/*
 * This routine allows you to merge (ie ADD) child shapes to your
 * own window's shape keeping its current shape and ADDING the child
 * shapes to it.
 * 
 * - Raster
 */
   void gdk_window_merge_child_shapes(GdkWindow * window);

/*
 * Check if a window has been shown, and whether all its
 * parents up to a toplevel have been shown, respectively.
 * Note that a window that is_viewable below is not necessarily
 * viewable in the X sense.
 */
   gboolean gdk_window_is_visible(GdkWindow * window);
   gboolean gdk_window_is_viewable(GdkWindow * window);

/* Set static bit gravity on the parent, and static
 * window gravity on all children.
 */
   gboolean gdk_window_set_static_gravities(GdkWindow * window,
                                            gboolean use_static);

/* GdkWindow */

   void gdk_window_set_hints(GdkWindow * window,
                             gint x,
                             gint y,
                             gint min_width,
                             gint min_height,
                             gint max_width, gint max_height, gint flags);
   void gdk_window_set_geometry_hints(GdkWindow * window,
                                      GdkGeometry * geometry,
                                      GdkWindowHints flags);
   void gdk_set_sm_client_id(const gchar * sm_client_id);


   void gdk_window_set_title(GdkWindow * window, const gchar * title);
   void gdk_window_set_role(GdkWindow * window, const gchar * role);
   void gdk_window_set_transient_for(GdkWindow * window,
                                     GdkWindow * leader);
   void gdk_window_set_background(GdkWindow * window, GdkColor * color);
   void gdk_window_set_back_pixmap(GdkWindow * window,
                                   GdkPixmap * pixmap,
                                   gboolean parent_relative);
   void gdk_window_set_cursor(GdkWindow * window, GdkCursor * cursor);
   void gdk_window_get_user_data(GdkWindow * window, gpointer * data);
   void gdk_window_get_geometry(GdkWindow * window,
                                gint * x,
                                gint * y,
                                gint * width, gint * height, gint * depth);
   void gdk_window_get_position(GdkWindow * window, gint * x, gint * y);
   gint gdk_window_get_origin(GdkWindow * window, gint * x, gint * y);
   gboolean gdk_window_get_deskrelative_origin(GdkWindow * window,
                                               gint * x, gint * y);
   void gdk_window_get_root_origin(GdkWindow * window, gint * x, gint * y);
   GdkWindow *gdk_window_get_pointer(GdkWindow * window,
                                     gint * x,
                                     gint * y, GdkModifierType * mask);
   GdkWindow *gdk_window_get_parent(GdkWindow * window);
   GdkWindow *gdk_window_get_toplevel(GdkWindow * window);
   GList *gdk_window_get_children(GdkWindow * window);
   GdkEventMask gdk_window_get_events(GdkWindow * window);
   void gdk_window_set_events(GdkWindow * window, GdkEventMask event_mask);

   void gdk_window_set_icon(GdkWindow * window,
                            GdkWindow * icon_window,
                            GdkPixmap * pixmap, GdkBitmap * mask);
   void gdk_window_set_icon_name(GdkWindow * window, const gchar * name);
   void gdk_window_set_group(GdkWindow * window, GdkWindow * leader);
   void gdk_window_set_decorations(GdkWindow * window,
                                   GdkWMDecoration decorations);
   void gdk_window_set_functions(GdkWindow * window,
                                 GdkWMFunction functions);
   GList *gdk_window_get_toplevels(void);

   void gdk_window_register_dnd(GdkWindow * window);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_WINDOW_H__ */
