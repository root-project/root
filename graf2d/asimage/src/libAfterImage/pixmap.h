#ifndef LIBAFTERIMAGE_PIXMAP_H_HEADER_FILE_INCLUDED
#define LIBAFTERIMAGE_PIXMAP_H_HEADER_FILE_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif


typedef struct ShadingInfo
{
    XColor tintColor;
    unsigned int shading;
}
ShadingInfo;

#define NO_NEED_TO_SHADE(o) ((o).shading==100 && (o).tintColor.red==0xFFFF && (o).tintColor.green==0xFFFF && (o).tintColor.blue == 0xFFFF)
int FillPixmapWithTile (Pixmap pixmap, Pixmap tile, int x, int y, int width, int height, int tile_x, int tile_y);
Pixmap GetRootPixmap (Atom id);
Pixmap ValidatePixmap (Pixmap p, int bSetHandler, int bTransparent, unsigned int *pWidth, unsigned int *pHeight);
int GetRootDimensions (int *width, int *height);
int GetWinPosition (Window w, int *x, int *y);
ARGB32 shading2tint32(ShadingInfo * shading);
Pixmap scale_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint);
Pixmap ScalePixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading);
void copyshade_drawable_area( ASVisual *asv, Drawable src, Pixmap trg,
				  		 int x, int y, int w, int h,
				  		 int trg_x, int trg_y,
				  		 GC gc, ARGB32 tint);
void CopyAndShadeArea ( Drawable src, Pixmap trg,
				   int x, int y, int w, int h,
				   int trg_x, int trg_y,
				   GC gc, ShadingInfo * shading);
void tile_pixmap (ASVisual *asv, Pixmap src, Pixmap trg, int src_w, int src_h, int x, int y, int w, int h, GC gc, ARGB32 tint);
void ShadeTiledPixmap (Pixmap src, Pixmap trg, int src_w, int src_h, int x, int y, int w, int h, GC gc, ShadingInfo * shading);
Pixmap shade_pixmap (ASVisual *asv, Pixmap src, int x, int y, int width, int height, GC gc, ARGB32 tint);
Pixmap ShadePixmap (Pixmap src, int x, int y, int width, int height, GC gc, ShadingInfo * shading);
Pixmap center_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint);
Pixmap CenterPixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading);
Pixmap grow_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint );
Pixmap GrowPixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading);
Pixmap cut_win_pixmap ( ASVisual *asv, Window win, Drawable src, int src_w, int src_h, int width,
					    int height, GC gc, ARGB32 tint);
Pixmap CutWinPixmap ( Window win, Drawable src, int src_w, int src_h, int width,
				      int height, GC gc, ShadingInfo * shading);
int fill_with_darkened_background (ASVisual *asv, Pixmap * pixmap, ARGB32 tint, int x, int y, int width, int height, int root_x, int root_y, int bDiscardOriginal, ASImage *root_im);
int fill_with_pixmapped_background (ASVisual *asv, Pixmap * pixmap, ASImage *image, int x, int y, int width, int height, int root_x, int root_y, int bDiscardOriginal, ASImage *root_im);
/************************************************/

#ifdef __cplusplus
}
#endif


#endif /* #ifndef LIBAFTERIMAGE_PIXMAP_H_HEADER_FILE_INCLUDED */

