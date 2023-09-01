#ifndef __GDK_GC_H__
#define __GDK_GC_H__

#include <gdk/gdkcolor.h>
#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkGCValues GdkGCValues;
   typedef struct _GdkGCClass GdkGCClass;

/* GC cap styles
 *  CapNotLast:
 *  CapButt:
 *  CapRound:
 *  CapProjecting:
 */
   typedef enum {
      GDK_CAP_NOT_LAST,
      GDK_CAP_BUTT,
      GDK_CAP_ROUND,
      GDK_CAP_PROJECTING
   } GdkCapStyle;

/* GC fill types.
 *  Solid:
 *  Tiled:
 *  Stippled:
 *  OpaqueStippled:
 */
   typedef enum {
      GDK_SOLID,
      GDK_TILED,
      GDK_STIPPLED,
      GDK_OPAQUE_STIPPLED
   } GdkFill;

/* GC function types.
 *   Copy: Overwrites destination pixels with the source pixels.
 *   Invert: Inverts the destination pixels.
 *   Xor: Xor's the destination pixels with the source pixels.
 *   Clear: set pixels to 0
 *   And: source AND destination
 *   And Reverse: source AND (NOT destination)
 *   And Invert: (NOT source) AND destination
 *   Noop: destination
 *   Or: source OR destination
 *   Nor: (NOT source) AND (NOT destination)
 *   Equiv: (NOT source) XOR destination
 *   Xor Reverse: source OR (NOT destination)
 *   Copy Inverted: NOT source
 *   Xor Inverted: (NOT source) OR destination
 *   Nand: (NOT source) OR (NOT destination)
 *   Set: set pixels to 1
 */
   typedef enum {
      GDK_COPY,
      GDK_INVERT,
      GDK_XOR,
      GDK_CLEAR,
      GDK_AND,
      GDK_AND_REVERSE,
      GDK_AND_INVERT,
      GDK_NOOP,
      GDK_OR,
      GDK_EQUIV,
      GDK_OR_REVERSE,
      GDK_COPY_INVERT,
      GDK_OR_INVERT,
      GDK_NAND,
      GDK_SET
   } GdkFunction;

/* GC join styles
 *  JoinMiter:
 *  JoinRound:
 *  JoinBevel:
 */
   typedef enum {
      GDK_JOIN_MITER,
      GDK_JOIN_ROUND,
      GDK_JOIN_BEVEL
   } GdkJoinStyle;

/* GC line styles
 *  Solid:
 *  OnOffDash:
 *  DoubleDash:
 */
   typedef enum {
      GDK_LINE_SOLID,
      GDK_LINE_ON_OFF_DASH,
      GDK_LINE_DOUBLE_DASH
   } GdkLineStyle;

   typedef enum {
      GDK_CLIP_BY_CHILDREN = 0,
      GDK_INCLUDE_INFERIORS = 1
   } GdkSubwindowMode;

   typedef enum {
      GDK_GC_FOREGROUND = 1 << 0,
      GDK_GC_BACKGROUND = 1 << 1,
      GDK_GC_FONT = 1 << 2,
      GDK_GC_FUNCTION = 1 << 3,
      GDK_GC_FILL = 1 << 4,
      GDK_GC_TILE = 1 << 5,
      GDK_GC_STIPPLE = 1 << 6,
      GDK_GC_CLIP_MASK = 1 << 7,
      GDK_GC_SUBWINDOW = 1 << 8,
      GDK_GC_TS_X_ORIGIN = 1 << 9,
      GDK_GC_TS_Y_ORIGIN = 1 << 10,
      GDK_GC_CLIP_X_ORIGIN = 1 << 11,
      GDK_GC_CLIP_Y_ORIGIN = 1 << 12,
      GDK_GC_EXPOSURES = 1 << 13,
      GDK_GC_LINE_WIDTH = 1 << 14,
      GDK_GC_LINE_STYLE = 1 << 15,
      GDK_GC_CAP_STYLE = 1 << 16,
      GDK_GC_JOIN_STYLE = 1 << 17
   } GdkGCValuesMask;

   struct _GdkGCValues {
      GdkColor foreground;
      GdkColor background;
      GdkFont *font;
      GdkFunction function;
      GdkFill fill;
      GdkPixmap *tile;
      GdkPixmap *stipple;
      GdkPixmap *clip_mask;
      GdkSubwindowMode subwindow_mode;
      gint ts_x_origin;
      gint ts_y_origin;
      gint clip_x_origin;
      gint clip_y_origin;
      gint graphics_exposures;
      gint line_width;
      GdkLineStyle line_style;
      GdkCapStyle cap_style;
      GdkJoinStyle join_style;
   };

   struct _GdkGC {
      gint dummy_var;
   };

   struct _GdkGCClass {
      void (*destroy) (GdkGC * gc);
      void (*get_values) (GdkGC * gc, GdkGCValues * values);
      void (*set_values) (GdkGC * gc,
                          GdkGCValues * values, GdkGCValuesMask mask);
      void (*set_dashes) (GdkGC * gc,
                          gint dash_offset, gchar dash_list[], gint n);
   };


   GdkGC *gdk_gc_new(GdkDrawable * drawable);
   GdkGC *gdk_gc_alloc(void);

   GdkGC *gdk_gc_new_with_values(GdkDrawable * drawable,
                                 GdkGCValues * values,
                                 GdkGCValuesMask values_mask);
   void gdk_gc_set_values(GdkGC * gc,
                                 GdkGCValues * values,
                                 GdkGCValuesMask values_mask);
   void gdk_gc_init(GdkGC * gc, GdkGCClass * klass);
   GdkGC *gdk_gc_ref(GdkGC * gc);
   void gdk_gc_unref(GdkGC * gc);
   void gdk_gc_get_values(GdkGC * gc, GdkGCValues * values);
   void gdk_gc_set_foreground(GdkGC * gc, GdkColor * color);
   void gdk_gc_set_background(GdkGC * gc, GdkColor * color);
   void gdk_gc_set_font(GdkGC * gc, GdkFont * font);
   void gdk_gc_set_function(GdkGC * gc, GdkFunction function);
   void gdk_gc_set_fill(GdkGC * gc, GdkFill fill);
   void gdk_gc_set_tile(GdkGC * gc, GdkPixmap * tile);
   void gdk_gc_set_stipple(GdkGC * gc, GdkPixmap * stipple);
   void gdk_gc_set_ts_origin(GdkGC * gc, gint x, gint y);
   void gdk_gc_set_clip_origin(GdkGC * gc, gint x, gint y);
   void gdk_gc_set_clip_mask(GdkGC * gc, GdkBitmap * mask);
   void gdk_gc_set_clip_rectangle(GdkGC * gc, GdkRectangle * rectangle);
   void gdk_gc_set_clip_region(GdkGC * gc, GdkRegion * region);
   void gdk_gc_set_subwindow(GdkGC * gc, GdkSubwindowMode mode);
   void gdk_gc_set_exposures(GdkGC * gc, gboolean exposures);
   void gdk_gc_set_line_attributes(GdkGC * gc,
                                   gint line_width,
                                   GdkLineStyle line_style,
                                   GdkCapStyle cap_style,
                                   GdkJoinStyle join_style);
   void gdk_gc_set_dashes(GdkGC * gc,
                          gint dash_offset, gint8 dash_list[], gint n);
   void gdk_gc_copy(GdkGC * dst_gc, GdkGC * src_gc);
   guint gdk_gc_set_text_align(GdkGC * gc, guint tAlign);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_DRAWABLE_H__ */
