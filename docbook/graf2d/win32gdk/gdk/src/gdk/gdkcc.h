#ifndef __GDK_CC_H__
#define __GDK_CC_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkColorContextDither GdkColorContextDither;
   typedef struct _GdkColorContext GdkColorContext;


/* Color Context modes.
 *
 * GDK_CC_MODE_UNDEFINED - unknown
 * GDK_CC_MODE_BW	 - default B/W
 * GDK_CC_MODE_STD_CMAP	 - has a standard colormap
 * GDK_CC_MODE_TRUE	 - is a TrueColor/DirectColor visual
 * GDK_CC_MODE_MY_GRAY	 - my grayramp
 * GDK_CC_MODE_PALETTE	 - has a pre-allocated palette
 */

   typedef enum {
      GDK_CC_MODE_UNDEFINED,
      GDK_CC_MODE_BW,
      GDK_CC_MODE_STD_CMAP,
      GDK_CC_MODE_TRUE,
      GDK_CC_MODE_MY_GRAY,
      GDK_CC_MODE_PALETTE
   } GdkColorContextMode;

   struct _GdkColorContextDither {
      gint fast_rgb[32][32][32];	/* quick look-up table for faster rendering */
      gint fast_err[32][32][32];	/* internal RGB error information */
      gint fast_erg[32][32][32];
      gint fast_erb[32][32][32];
   };

   struct _GdkColorContext {
      GdkVisual *visual;
      GdkColormap *colormap;

      gint num_colors;          /* available no. of colors in colormap */
      gint max_colors;          /* maximum no. of colors */
      gint num_allocated;       /* no. of allocated colors */

      GdkColorContextMode mode;
      gint need_to_free_colormap;
      GdkAtom std_cmap_atom;

      gulong *clut;             /* color look-up table */
      GdkColor *cmap;           /* colormap */

      GHashTable *color_hash;   /* hash table of allocated colors */
      GdkColor *palette;        /* preallocated palette */
      gint num_palette;         /* size of palette */

      GdkColorContextDither *fast_dither;	/* fast dither matrix */

      struct {
         gint red;
         gint green;
         gint blue;
      } shifts;

      struct {
         gulong red;
         gulong green;
         gulong blue;
      } masks;

      struct {
         gint red;
         gint green;
         gint blue;
      } bits;

      gulong max_entry;

      gulong black_pixel;
      gulong white_pixel;
   };

   GdkColorContext *gdk_color_context_new(GdkVisual * visual,
                                          GdkColormap * colormap);

   GdkColorContext *gdk_color_context_new_mono(GdkVisual * visual,
                                               GdkColormap * colormap);

   void gdk_color_context_free(GdkColorContext * cc);

   gulong gdk_color_context_get_pixel(GdkColorContext * cc,
                                      gushort red,
                                      gushort green,
                                      gushort blue, gint * failed);
   void gdk_color_context_get_pixels(GdkColorContext * cc,
                                     gushort * reds,
                                     gushort * greens,
                                     gushort * blues,
                                     gint ncolors,
                                     gulong * colors, gint * nallocated);
   void gdk_color_context_get_pixels_incremental(GdkColorContext * cc,
                                                 gushort * reds,
                                                 gushort * greens,
                                                 gushort * blues,
                                                 gint ncolors,
                                                 gint * used,
                                                 gulong * colors,
                                                 gint * nallocated);

   gint gdk_color_context_query_color(GdkColorContext * cc,
                                      GdkColor * color);
   gint gdk_color_context_query_colors(GdkColorContext * cc,
                                       GdkColor * colors, gint num_colors);

   gint gdk_color_context_add_palette(GdkColorContext * cc,
                                      GdkColor * palette,
                                      gint num_palette);

   void gdk_color_context_init_dither(GdkColorContext * cc);
   void gdk_color_context_free_dither(GdkColorContext * cc);

   gulong gdk_color_context_get_pixel_from_palette(GdkColorContext * cc,
                                                   gushort * red,
                                                   gushort * green,
                                                   gushort * blue,
                                                   gint * failed);
   guchar gdk_color_context_get_index_from_palette(GdkColorContext * cc,
                                                   gint * red,
                                                   gint * green,
                                                   gint * blue,
                                                   gint * failed);


#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_CC_H__ */
