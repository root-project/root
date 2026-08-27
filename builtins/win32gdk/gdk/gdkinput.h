#ifndef __GDK_INPUT_H__
#define __GDK_INPUT_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkDeviceKey GdkDeviceKey;
   typedef struct _GdkDeviceInfo GdkDeviceInfo;
   typedef struct _GdkTimeCoord GdkTimeCoord;

   typedef enum {
      GDK_EXTENSION_EVENTS_NONE,
      GDK_EXTENSION_EVENTS_ALL,
      GDK_EXTENSION_EVENTS_CURSOR
   } GdkExtensionMode;

   typedef enum {
      GDK_SOURCE_MOUSE,
      GDK_SOURCE_PEN,
      GDK_SOURCE_ERASER,
      GDK_SOURCE_CURSOR
   } GdkInputSource;

   typedef enum {
      GDK_MODE_DISABLED,
      GDK_MODE_SCREEN,
      GDK_MODE_WINDOW
   } GdkInputMode;

   typedef enum {
      GDK_AXIS_IGNORE,
      GDK_AXIS_X,
      GDK_AXIS_Y,
      GDK_AXIS_PRESSURE,
      GDK_AXIS_XTILT,
      GDK_AXIS_YTILT,
      GDK_AXIS_LAST
   } GdkAxisUse;

   struct _GdkDeviceInfo {
      guint32 deviceid;
      gchar *name;
      GdkInputSource source;
      GdkInputMode mode;
      gint has_cursor;          /* TRUE if the X pointer follows device motion */
      gint num_axes;
      GdkAxisUse *axes;         /* Specifies use for each axis */
      gint num_keys;
      GdkDeviceKey *keys;
   };

   struct _GdkDeviceKey {
      guint keyval;
      GdkModifierType modifiers;
   };

   struct _GdkTimeCoord {
      guint32 time;
      gdouble x;
      gdouble y;
      gdouble pressure;
      gdouble xtilt;
      gdouble ytilt;
   };

   GList *gdk_input_list_devices(void);
   void gdk_input_set_extension_events(GdkWindow * window,
                                       gint mask, GdkExtensionMode mode);
   void gdk_input_set_source(guint32 deviceid, GdkInputSource source);
   gboolean gdk_input_set_mode(guint32 deviceid, GdkInputMode mode);
   void gdk_input_set_axes(guint32 deviceid, GdkAxisUse * axes);
   void gdk_input_set_key(guint32 deviceid,
                          guint index,
                          guint keyval, GdkModifierType modifiers);
   void gdk_input_window_get_pointer(GdkWindow * window,
                                     guint32 deviceid,
                                     gdouble * x,
                                     gdouble * y,
                                     gdouble * pressure,
                                     gdouble * xtilt,
                                     gdouble * ytilt,
                                     GdkModifierType * mask);
   GdkTimeCoord *gdk_input_motion_events(GdkWindow * window,
                                         guint32 deviceid,
                                         guint32 start,
                                         guint32 stop,
                                         gint * nevents_return);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_INPUT_H__ */
