#ifndef LIBAFTERIMAGE_XIMAGE_HEADER_FILE_INCLUDED
#define LIBAFTERIMAGE_XIMAGE_HEADER_FILE_INCLUDED

#include "asvisual.h"
#include "blender.h"
#include "asimage.h"

#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/ximage.h
 * NAME
 * ximage - Defines conversion to and from XImages and Pixmaps.
 * DESCRIPTION
 * ximage2asimage()	- convert XImage structure into ASImage
 * pixmap2asimage()	- convert X11 pixmap into ASImage
 * asimage2ximage()	- convert ASImage into XImage
 * asimage2mask_ximage() - convert alpha channel of ASImage into XImage
 * asimage2pixmap()	- convert ASImage into Pixmap ( possibly using
 * 					  precreated XImage )
 * asimage2mask() 	- convert alpha channel of ASImage into 1 bit
 * 				  	  mask Pixmap.
 * SEE ALSO
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 *******/

/****f* libAfterImage/picture_ximage2asimage()
 * NAME
 * picture_ximage2asimage()
 * SYNOPSIS
 * ASImage *picture_ximage2asimage ( struct ASVisual *asv,
 *                                   XImage * xim, XImage *alpha_xim,
 *                                   unsigned int compression );
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * xim           - source XImage
 * alpha_xim     - source XImage for Alpha channel
 * compression   - degree of compression of resulting ASImage.
 * RETURN VALUE
 * pointer to newly allocated ASImage, containing encoded data, on
 * success. NULL on failure.
 * DESCRIPTION
 * picture_ximage2asimage will attempt to create new ASImage with the same
 * dimensions as supplied XImage. If both XImages are supplied - they must
 * have same dimentions. XImage will be decoded based on
 * supplied ASVisual, and resulting scanlines will be encoded into
 * ASImage.
 *********/
/****f* libAfterImage/ximage2asimage()
 * NAME
 * ximage2asimage() - same as picture_ximage2asimage with alpha_ximage 
 * set to NULL. Supplied for compatibility with older versions and for 
 * convinience.
 * SYNOPSIS
 * ASImage *ximage2asimage ( struct ASVisual *asv, XImage * xim,
 *                           unsigned int compression );
 * INPUTS
 * asv  		 - pointer to valid ASVisual structure
 * xim  		 - source XImage
 * compression - degree of compression of resulting ASImage.
 * RETURN VALUE
 * pointer to newly allocated ASImage, containing encoded data, on
 * success. NULL on failure.
 * DESCRIPTION
 *********/
/****f* libAfterImage/pixmap2asimage()
 * NAME
 * pixmap2asimage()
 * SYNOPSIS
 * ASImage *pixmap2ximage( ASVisual *asv, Pixmap p, int x, int y,
 *                         unsigned int width, unsigned int height,
 *						   unsigned long plane_mask,
 *                         unsigned int compression);
 * INPUTS
 * asv  		  - pointer to valid ASVisual structure
 * p    		  - source Pixmap
 * x, y,
 * width, height- rectangle on Pixmap to be encoded into ASImage.
 * plane_mask   - limits color planes to be copied from Pixmap.
 * keep_cache   - indicates if we should keep XImage, used to copy
 *                image data from the X server, and attached it to 
 * 				  ximage member of resulting ASImage.
 * compression  - degree of compression of resulting ASImage.
 * RETURN VALUE
 * pointer to newly allocated ASImage, containing data in XImage format, 
 * on success. NULL on failure.
 * DESCRIPTION
 * pixmap2ximage will obtain XImage of the requested area of the
 * X Pixmap, and it will attach it to newly created ASImage using 
 * alt.ximage member. After that newly created ASImage could be used 
 * in any transformations.
 *********/
/****f* libAfterImage/picture2asimage()
 * NAME
 * picture2asimage()
 * SYNOPSIS
 * ASImage *picture2asimage (struct ASVisual *asv,
 *                           Pixmap rgb, Pixmap a,
 *                           int x, int y,
 *                           unsigned int width,
 *                           unsigned int height,
 *                           unsigned long plane_mask,
 *                           Bool keep_cache,
 *                           unsigned int compression );
 * INPUTS
 * asv  		  - pointer to valid ASVisual structure
 * rgb    		  - source Pixmap for red, green and blue channels
 * a    		  - source Pixmap for the alpha channel
 * x, y,
 * width, height- rectangle on Pixmap to be encoded into ASImage.
 * plane_mask   - limits color planes to be copied from Pixmap.
 * keep_cache   - indicates if we should keep XImage, used to copy
 *                image data from the X server, and attached it to 
 * 				  ximage member of resulting ASImage.
 * compression  - degree of compression of resulting ASImage.
 * RETURN VALUE
 * pointer to newly allocated ASImage, containing encoded data, on
 * success. NULL on failure.
 * DESCRIPTION
 * picture2asimage will obtain XImage of the requested area of the
 * X Pixmap, If alpha channel pixmap is supplied - it will be used to 
 * encode ASImage's alpha channel. Alpha channel pixmap must be either
 * 8 or 1 bit deep, and it must have the same dimentions as main Pixmap.
 *********/
/****f* libAfterImage/pixmap2asimage()
 * NAME
 * pixmap2asimage()
 * SYNOPSIS
 * ASImage *pixmap2asimage ( struct ASVisual *asv, Pixmap p,
 *                           int x, int y,
 *                           unsigned int width,
 *                           unsigned int height,
 *                           unsigned long plane_mask,
 *                           Bool keep_cache,
 *                           unsigned int compression );
 * INPUTS
 * asv  		  - pointer to valid ASVisual structure
 * p    		  - source Pixmap
 * x, y,
 * width, height- rectangle on Pixmap to be encoded into ASImage.
 * plane_mask   - limits color planes to be copied from Pixmap.
 * keep_cache   - indicates if we should keep XImage, used to copy
 *                image data from the X server, and attached it to 
 *                ximage member of resulting ASImage.
 * compression  - degree of compression of resulting ASImage.
 * RETURN VALUE
 * pointer to newly allocated ASImage, containing encoded data, on
 * success. NULL on failure.
 * DESCRIPTION
 * same as picture2asimage() with alpha pixmap set to None. Supplied for
 * compatibility and convinience.
 *********/
ASImage *picture_ximage2asimage (ASVisual *asv, XImage *xim, XImage *alpha_xim, unsigned int compression);
ASImage *ximage2asimage (struct ASVisual *asv, XImage * xim, unsigned int compression);
ASImage *pixmap2ximage( ASVisual *asv, Pixmap p, int x, int y,
                        unsigned int width, unsigned int height,
						unsigned long plane_mask, unsigned int compression);
ASImage *picture2asimage(ASVisual *asv, Pixmap rgb, Pixmap a , int x, int y,
                         unsigned int width, unsigned int height,
						 unsigned long plane_mask, Bool keep_cache, unsigned int compression);
ASImage *pixmap2asimage (struct ASVisual *asv, Pixmap p, int x, int y,
	                     unsigned int width, unsigned int height,
		  				 unsigned long plane_mask, Bool keep_cache, unsigned int compression);

/****f* libAfterImage/asimage2ximage()
 * NAME
 * asimage2ximage()
 * SYNOPSIS
 * XImage  *asimage2ximage  (struct ASVisual *asv, ASImage *im);
 * INPUTS
 * asv  		- pointer to valid ASVisual structure
 * im    		- source ASImage
 * RETURN VALUE
 * On success returns newly created and encoded XImage of the same
 * colordepth as the supplied ASVisual. NULL on failure.
 * DESCRIPTION
 * asimage2ximage() creates new XImage of the exact same size as
 * supplied ASImage, and depth of supplied ASVisual. REd, Green and
 * Blue channels of ASImage then gets decoded, and encoded into XImage.
 * Missing scanlines get filled with black color.
 * NOTES
 * Returned pointer to XImage will also be stored in im->alt.ximage,
 * and It will be destroyed when XImage is destroyed, or reused in any
 * subsequent calls to asimage2ximage(). If any other behaviour is
 * desired - make sure you set im->alt.ximage to NULL, to dissociate
 * XImage object from ASImage.
 * SEE ALSO
 * create_visual_ximage()
 *********/
/****f* libAfterImage/asimage2alpha_ximage()
 * NAME
 * asimage2alpha_ximage()
 * SYNOPSIS
 * XImage  *asimage2alpha_ximage (struct ASVisual *asv, 
 *                                ASImage *im, Bool bitmap);
 * INPUTS
 * asv   		- pointer to valid ASVisual structure
 * im    		- source ASImage
 * bitmap       - if True resulting XImage will have depth of 1 bit -
 *                traditional X mask; otherwise it will have depth of 8
 *                (usefull for XFree86 RENDER extension)
 * RETURN VALUE
 * On success returns newly created and encoded XImage of the depth 1 or 8.
 * NULL on failure.
 * DESCRIPTION
 * asimage2alpha_ximage() creates new XImage of the exact same size as
 * supplied ASImage, and depth 1 or 8. Alpha channels of ASImage then gets
 * decoded, and encoded into XImage. In case requested depth is 1 then
 * alpha channel is interpreted like so: 127 or greater is encoded as 1,
 * otherwise as 0.
 * Missing scanlines get filled with 1s as they signify absence of mask.
 * NOTES
 * Returned pointer to XImage will also be stored in im->alt.mask_ximage,
 * and It will be destroyed when XImage is destroyed, or reused in any
 * subsequent calls to asimage2mask_ximage(). If any other behaviour is
 * desired - make sure you set im->alt.mask_ximage to NULL, to dissociate
 * XImage object from ASImage.
 *********/
/****f* libAfterImage/asimage2mask_ximage()
 * NAME
 * asimage2mask_ximage() - same as asimage2alpha_ximage(). Supplied for 
 * convinience and compatibility with older versions.
 * SYNOPSIS
 * XImage  *asimage2mask_ximage (struct ASVisual *asv, ASImage *im);
 * INPUTS
 * asv   		- pointer to valid ASVisual structure
 * im    		- source ASImage
 * RETURN VALUE
 * On success returns newly created and encoded XImage of the depth 1.
 * NULL on failure.
 *********/
/****f* libAfterImage/asimage2pixmap()
 * NAME
 * asimage2pixmap()
 * SYNOPSIS
 * Bool	 asimage2drawable( struct ASVisual *asv, Drawable d, ASImage *im,
 *                         GC gc,
 *        			       int src_x, int src_y, int dest_x, int dest_y,
 *       		  		   unsigned int width, unsigned int height,
 *				  		   Bool use_cached);
 * INPUTS
 * asv  		- pointer to valid ASVisual structure
 * d  			- destination drawable - Pixmap or Window
 * im    		- source ASImage
 * gc   		- precreated GC to use for XImage transfer. If NULL,
 *  			  asimage2drawable() will use DefaultGC.
 * src_x        - Specifies the offset in X from the left edge of the image
 *                defined by the ASImage structure.
 * src_y        - Specifies the offset in Y from the top edge of the image
 *                defined by the ASImage structure.
 * dest_x,dest_y- Specify the x and y coordinates, which are relative to
 *                the origin of the drawable and are the coordinates of
 *                the subimage.
 * width,height - Specify the width and height of the subimage, which
 *                define the dimensions of the rectangle.
 * use_cached	- If True will make asimage2pixmap() to use XImage
 *  			  attached to ASImage, instead of creating new one. Only
 *  			  works if ASImage->ximage data member is not NULL.
 * RETURN VALUE
 * On success returns True.
 * DESCRIPTION
 * asimage2drawable() creates will copy portion of ASImage onto the X
 * Drawable. It checks if it needs to encode XImage
 * from ASImage data, and calls asimage2ximage() if yes, it has to.
 * It then supplied gc or DefaultGC of the screen to transfer
 * XImage to the server.
 * Missing scanlines get filled with black color.
 * SEE ALSO
 * asimage2ximage()
 * asimage2pixmap()
 * create_visual_pixmap()
 *********/

/****f* libAfterImage/asimage2pixmap()
 * NAME
 * asimage2pixmap()
 * SYNOPSIS
 * Pixmap   asimage2pixmap  ( struct ASVisual *asv, Window root,
 *                            ASImage *im, GC gc, Bool use_cached);
 * INPUTS
 * asv  		- pointer to valid ASVisual structure
 * root 		- root window of destination screen
 * im    		- source ASImage
 * gc   		- precreated GC to use for XImage transfer. If NULL,
 *  			  asimage2pixmap() will use DefaultGC.
 * use_cached	- If True will make asimage2pixmap() to use XImage
 *  			  attached to ASImage, instead of creating new one. Only
 *  			  works if ASImage->ximage data member is not NULL.
 * RETURN VALUE
 * On success returns newly pixmap of the same colordepth as ASVisual.
 * None on failure.
 * DESCRIPTION
 * asimage2pixmap() creates new pixmap of exactly same size as
 * supplied ASImage. It then calls asimage2drawable to copy entire content
 * of the ASImage onto that created pixmap.
 * EXAMPLE
 * asview.c: ASView.5
 * SEE ALSO
 * asimage2ximage()
 * asimage2drawable()
 * create_visual_pixmap()
 *********/
/****f* libAfterImage/asimage2mask()
 * NAME
 * asimage2mask()
 * SYNOPSIS
 * Pixmap   asimage2mask ( struct ASVisual *asv, Window root,
 *                         ASImage *im, GC gc, Bool use_cached);
 * asv        - pointer to valid ASVisual structure
 * root       - root window of destination screen
 * im         - source ASImage
 * gc         - precreated GC for 1 bit deep drawables to use for
 *              XImage transfer. If NULL, asimage2mask() will create one.
 * use_cached - If True will make asimage2mask() to use mask XImage
 *  			attached to ASImage, instead of creating new one. Only
 *  			works if ASImage->alt.mask_ximage data member is not NULL.
 * RETURN VALUE
 * On success returns newly created pixmap of the colordepth 1.
 * None on failure.
 * DESCRIPTION
 * asimage2mask() creates new pixmap of exactly same size as
 * supplied ASImage. It then calls asimage2mask_ximage().
 * It then uses supplied gc, or creates new gc, to transfer
 * XImage to the server and put it on Pixmap.
 * Missing scanlines get filled with 1s.
 * SEE ALSO
 * asimage2mask_ximage()
 **********/
XImage  *asimage2ximage  (struct ASVisual *asv, ASImage *im);
Bool     subimage2ximage (struct ASVisual *asv, ASImage *im, int x, int y, XImage* xim);
Bool     put_ximage( ASVisual *asv, XImage *xim, Drawable d, GC gc,
                     int src_x, int src_y, int dest_x, int dest_y,
  		             unsigned int width, unsigned int height );


XImage  *asimage2alpha_ximage (ASVisual *asv, ASImage *im, Bool bitmap );
XImage  *asimage2mask_ximage (struct ASVisual *asv, ASImage *im);
Bool	 asimage2drawable( struct ASVisual *asv, Drawable d, ASImage *im, GC gc,
         			       int src_x, int src_y, int dest_x, int dest_y,
        		  		   unsigned int width, unsigned int height,
				  		   Bool use_cached);
/* these will do the same, but will use OpenGL API where available */
Bool asimage2drawable_gl(	ASVisual *asv, Drawable d, ASImage *im,
                  		int src_x, int src_y, int dest_x, int dest_y,
        		  		int width, int height, int d_width, int d_height, 
						Bool force_direct );

Bool	 asimage2alpha_drawable( ASVisual *asv, Drawable d, ASImage *im, GC gc,
      	    		  	   int src_x, int src_y, int dest_x, int dest_y,
        				   unsigned int width, unsigned int height,
						   Bool use_cached);
Pixmap   asimage2pixmap  (struct ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached);
Pixmap	 asimage2alpha   (struct ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached, Bool bitmap);
Pixmap   asimage2mask    (struct ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached);

#ifdef __cplusplus
}
#endif

#endif
