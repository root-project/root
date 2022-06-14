#ifndef __GDK_VISUAL_H__
#define __GDK_VISUAL_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* Types of visuals.
 *   StaticGray:
 *   Grayscale:
 *   StaticColor:
 *   PseudoColor:
 *   TrueColor:
 *   DirectColor:
 */
   typedef enum {
      GDK_VISUAL_STATIC_GRAY,
      GDK_VISUAL_GRAYSCALE,
      GDK_VISUAL_STATIC_COLOR,
      GDK_VISUAL_PSEUDO_COLOR,
      GDK_VISUAL_TRUE_COLOR,
      GDK_VISUAL_DIRECT_COLOR
   } GdkVisualType;

/* The visual type.
 *   "type" is the type of visual this is (PseudoColor, TrueColor, etc).
 *   "depth" is the bit depth of this visual.
 *   "colormap_size" is the size of a colormap for this visual.
 *   "bits_per_rgb" is the number of significant bits per red, green and blue.
 *  The red, green and blue masks, shifts and precisions refer
 *   to value needed to calculate pixel values in TrueColor and DirectColor
 *   visuals. The "mask" is the significant bits within the pixel. The
 *   "shift" is the number of bits left we must shift a primary for it
 *   to be in position (according to the "mask"). "prec" refers to how
 *   much precision the pixel value contains for a particular primary.
 */
   struct _GdkVisual {
      GdkVisualType type;
      gint depth;
      GdkByteOrder byte_order;
      gint colormap_size;
      gint bits_per_rgb;

      guint32 red_mask;
      gint red_shift;
      gint red_prec;

      guint32 green_mask;
      gint green_shift;
      gint green_prec;

      guint32 blue_mask;
      gint blue_shift;
      gint blue_prec;
   };

   gint gdk_visual_get_best_depth(void);
   GdkVisualType gdk_visual_get_best_type(void);
   GdkVisual *gdk_visual_get_system(void);
   GdkVisual *gdk_visual_get_best(void);
   GdkVisual *gdk_visual_get_best_with_depth(gint depth);
   GdkVisual *gdk_visual_get_best_with_type(GdkVisualType visual_type);
   GdkVisual *gdk_visual_get_best_with_both(gint depth,
                                            GdkVisualType visual_type);

/* Actually, these are no-ops... */
   GdkVisual *gdk_visual_ref(GdkVisual * visual);
   void gdk_visual_unref(GdkVisual * visual);

   void gdk_query_depths(gint ** depths, gint * count);
   void gdk_query_visual_types(GdkVisualType ** visual_types,
                               gint * count);

   GList *gdk_list_visuals(void);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_VISUAL_H__ */
