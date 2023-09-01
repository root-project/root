/* International Input Method Support Functions
 */

#ifndef __GDK_IM_H__
#define __GDK_IM_H__

#include <gdk/gdkcolor.h>
#include <gdk/gdkevents.h>
#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkIC GdkIC;
   typedef struct _GdkICAttr GdkICAttr;

   typedef enum {               /*< flags > */
          GDK_IM_PREEDIT_AREA = 0x0001,
      GDK_IM_PREEDIT_CALLBACKS = 0x0002,
      GDK_IM_PREEDIT_POSITION = 0x0004,
      GDK_IM_PREEDIT_NOTHING = 0x0008,
      GDK_IM_PREEDIT_NONE = 0x0010,
      GDK_IM_PREEDIT_MASK = 0x001f,

      GDK_IM_STATUS_AREA = 0x0100,
      GDK_IM_STATUS_CALLBACKS = 0x0200,
      GDK_IM_STATUS_NOTHING = 0x0400,
      GDK_IM_STATUS_NONE = 0x0800,
      GDK_IM_STATUS_MASK = 0x0f00
   } GdkIMStyle;

   typedef enum {
      GDK_IC_STYLE = 1 << 0,
      GDK_IC_CLIENT_WINDOW = 1 << 1,
      GDK_IC_FOCUS_WINDOW = 1 << 2,
      GDK_IC_FILTER_EVENTS = 1 << 3,
      GDK_IC_SPOT_LOCATION = 1 << 4,
      GDK_IC_LINE_SPACING = 1 << 5,
      GDK_IC_CURSOR = 1 << 6,

      GDK_IC_PREEDIT_FONTSET = 1 << 10,
      GDK_IC_PREEDIT_AREA = 1 << 11,
      GDK_IC_PREEDIT_AREA_NEEDED = 1 << 12,
      GDK_IC_PREEDIT_FOREGROUND = 1 << 13,
      GDK_IC_PREEDIT_BACKGROUND = 1 << 14,
      GDK_IC_PREEDIT_PIXMAP = 1 << 15,
      GDK_IC_PREEDIT_COLORMAP = 1 << 16,

      GDK_IC_STATUS_FONTSET = 1 << 21,
      GDK_IC_STATUS_AREA = 1 << 22,
      GDK_IC_STATUS_AREA_NEEDED = 1 << 23,
      GDK_IC_STATUS_FOREGROUND = 1 << 24,
      GDK_IC_STATUS_BACKGROUND = 1 << 25,
      GDK_IC_STATUS_PIXMAP = 1 << 26,
      GDK_IC_STATUS_COLORMAP = 1 << 27,

      GDK_IC_ALL_REQ = GDK_IC_STYLE | GDK_IC_CLIENT_WINDOW,

      GDK_IC_PREEDIT_AREA_REQ = GDK_IC_PREEDIT_AREA |
          GDK_IC_PREEDIT_FONTSET,
      GDK_IC_PREEDIT_POSITION_REQ =
          GDK_IC_PREEDIT_AREA | GDK_IC_SPOT_LOCATION |
          GDK_IC_PREEDIT_FONTSET,

      GDK_IC_STATUS_AREA_REQ = GDK_IC_STATUS_AREA | GDK_IC_STATUS_FONTSET
   } GdkICAttributesType;

   struct _GdkICAttr {
      GdkIMStyle style;
      GdkWindow *client_window;
      GdkWindow *focus_window;
      GdkEventMask filter_events;
      GdkPoint spot_location;
      gint line_spacing;
      GdkCursor *cursor;

      GdkFont *preedit_fontset;
      GdkRectangle preedit_area;
      GdkRectangle preedit_area_needed;
      GdkColor preedit_foreground;
      GdkColor preedit_background;
      GdkPixmap *preedit_pixmap;
      GdkColormap *preedit_colormap;

      GdkFont *status_fontset;
      GdkRectangle status_area;
      GdkRectangle status_area_needed;
      GdkColor status_foreground;
      GdkColor status_background;
      GdkPixmap *status_pixmap;
      GdkColormap *status_colormap;
   };

   gboolean gdk_im_ready(void);

   void gdk_im_begin(GdkIC * ic, GdkWindow * window);
   void gdk_im_end(void);
   GdkIMStyle gdk_im_decide_style(GdkIMStyle supported_style);
   GdkIMStyle gdk_im_set_best_style(GdkIMStyle best_allowed_style);

   GdkIC *gdk_ic_new(GdkICAttr * attr, GdkICAttributesType mask);
   void gdk_ic_destroy(GdkIC * ic);
   GdkIMStyle gdk_ic_get_style(GdkIC * ic);
   GdkEventMask gdk_ic_get_events(GdkIC * ic);

   GdkICAttr *gdk_ic_attr_new(void);
   void gdk_ic_attr_destroy(GdkICAttr * attr);

   GdkICAttributesType gdk_ic_set_attr(GdkIC * ic,
                                       GdkICAttr * attr,
                                       GdkICAttributesType mask);
   GdkICAttributesType gdk_ic_get_attr(GdkIC * ic,
                                       GdkICAttr * attr,
                                       GdkICAttributesType mask);
#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_IM_H__ */
