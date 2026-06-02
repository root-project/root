#ifndef TRANSFORM_HEADER_FILE_INCLUDED
#define TRANSFORM_HEADER_FILE_INCLUDED

#include "asvisual.h"
#include "blender.h"
#include "asimage.h"


#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/transform.h
 * NAME
 * transform
 * SYNOPSIS
 * Defines transformations that could be performed on ASImage.
 * DESCRIPTION
 *
 * Transformations can be performed with different degree of quality.
 * Internal engine uses 24.8 bits per channel per pixel. As the result
 * there are no precision loss, while performing complex calculations.
 * Error diffusion algorithms could be used to transform it back into 8
 * bit without quality loss.
 *
 * Any Transformation could be performed with the result written directly
 * into XImage, so that it could be displayed faster.
 *
 * Complex interpolation algorithms are used to perform scaling
 * operations, thus yielding very good quality. All the transformations
 * are performed in integer math, with the result of greater speeds.
 * Optional MMX inline assembly has been incorporated into some
 * procedures, and allows to achieve considerably better performance on
 * compatible CPUs.
 *
 * SEE ALSO
 *  Transformations :
 *          scale_asimage(), tile_asimage(), merge_layers(), 
 * 			make_gradient(), flip_asimage(), mirror_asimage(), 
 * 			pad_asimage(), blur_asimage_gauss(), fill_asimage(), 
 * 			adjust_asimage_hsv()
 *
 *  Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 *******/
/****f* libAfterImage/transform/scale_asimage()
 * NAME
 * scale_asimage() - scales source ASImage into new image of requested 
 * dimensions. 
 * SYNOPSIS
 * ASImage *scale_asimage( struct ASVisual *asv,
 *                         ASImage *src,
 *                         unsigned int to_width,
 *                         unsigned int to_height,
 *                         ASAltImFormats out_format,
 *                         unsigned int compression_out, int quality );
 * INPUTS
 * asv  		- pointer to valid ASVisual structure
 * src   		- source ASImage
 * to_width 	- desired width of the resulting image
 * to_height	- desired height of the resulting image
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out- compression level of resulting image in range 0-100.
 * quality  	- output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * If size has to be reduced - then several neighboring pixels will be 
 * averaged into single pixel. If size has to be increased then new 
 * pixels will be interpolated based on values of four neighboring pixels.
 * EXAMPLE
 * ASScale
 *********/
/****f* libAfterImage/transform/tile_asimage()
 * NAME
 * tile_asimage() - tiles/crops ASImage to desired size, while optionaly 
 * tinting it at the same time.
 * SYNOPSIS
 * ASImage *tile_asimage ( struct ASVisual *asv,
 *                         ASImage *src,
 *                         int offset_x,
 *                         int offset_y,
 *                         unsigned int to_width,
 *                         unsigned int to_height,
 *                         ARGB32 tint,
 *                         ASAltImFormats out_format,
 *                         unsigned int compression_out, int quality );
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * src          - source ASImage
 * offset_x     - left clip margin
 * offset_y     - right clip margin
 * to_width     - desired width of the resulting image
 * to_height    - desired height of the resulting image
 * tint         - ARGB32 value describing tinting color.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out- compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * Offset_x and offset_y define origin on source image from which
 * tiling will start. If offset_x or offset_y is outside of the image
 * boundaries, then it will be reduced by whole number of image sizes to
 * fit inside the image. At the time of tiling image will be tinted
 * unless tint == 0.
 * EXAMPLE
 * ASTile
 *********/
/****f* libAfterImage/transform/merge_layers()
 * NAME
 * merge_layers()
 * SYNOPSIS
 * ASImage *merge_layers  ( struct ASVisual *asv,
 *                          ASImageLayer *layers, int count,
 *                          unsigned int dst_width,
 *                          unsigned int dst_height,
 *                          ASAltImFormats out_format,
 *                          unsigned int compression_out, int quality);
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * layers       - array of ASImageLayer structures that will be rendered
 *                one on top of another. First element corresponds to
 *                the bottommost layer.
 * dst_width    - desired width of the resulting image
 * dst_height   - desired height of the resulting image
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out - compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * merge_layers() will create new ASImage of requested size. It will then
 * go through all the layers, and fill image with composition.
 * Bottommost layer will be used unchanged and above layers will be
 * superimposed on it, using algorithm specified in ASImageLayer
 * structure of the overlaying layer. Layers may have smaller size
 * then destination image, and maybe placed in arbitrary locations. Each
 * layer will be padded to fit width of the destination image with all 0
 * effectively making it transparent.
 *********/
/****f* libAfterImage/transform/make_gradient()
 * NAME
 * make_gradient() - renders linear gradient into new ASImage
 * SYNOPSIS
 * ASImage *make_gradient ( struct ASVisual *asv,
 *                          struct ASGradient *grad,
 *                          unsigned int width,
 *                          unsigned int height,
 *                          ASFlagType filter,
 *                          ASAltImFormats out_format,
 *                          unsigned int compression_out, int quality);
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * grad         - ASGradient structure defining how gradient should be
 *                drawn
 * width        - desired width of the resulting image
 * height       - desired height of the resulting image
 * filter       - only channels corresponding to set bits will be
 *                rendered.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out- compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * make_gradient() will create new image of requested size and it will
 * fill it with gradient, described in structure pointed to by grad.
 * Different dithering techniques will be applied to produce nicer
 * looking gradients.
 *********/
/****f* libAfterImage/transform/flip_asimage()
 * NAME
 * flip_asimage() - rotates ASImage in 90 degree increments
 * SYNOPSIS
 * ASImage *flip_asimage ( struct ASVisual *asv,
 *                         ASImage *src,
 *                         int offset_x, int offset_y,
 *                         unsigned int to_width,
 *                         unsigned int to_height,
 *                         int flip, ASAltImFormats out_format,
 *                         unsigned int compression_out, int quality );
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * src          - source ASImage
 * offset_x     - left clip margin
 * offset_y     - right clip margin
 * to_width     - desired width of the resulting image
 * to_height    - desired height of the resulting image
 * flip         - flip flags determining degree of rotation.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out - compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * flip_asimage() will create new image of requested size, it will then
 * tile source image based on offset_x, offset_y, and destination size,
 * and it will rotate it then based on flip value. Three rotation angles
 * supported 90, 180 and 270 degrees.
 *********/
/****f* libAfterImage/transform/mirror_asimage()
 * NAME
 * mirror_asimage()
 * SYNOPSIS
 * ASImage *mirror_asimage ( struct ASVisual *asv,
 *                           ASImage *src,
 *                           int offset_x, int offset_y,
 *                           unsigned int to_width,
 *                           unsigned int to_height,
 *                           Bool vertical, ASAltImFormats out_format,
 *                           unsigned int compression_out, int quality );
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * src          - source ASImage
 * offset_x     - left clip margin
 * offset_y     - right clip margin
 * to_width     - desired width of the resulting image
 * to_height    - desired height of the resulting image
 * vertical     - mirror in vertical direction.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out - compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * mirror_asimage() will create new image of requested size, it will then
 * tile source image based on offset_x, offset_y, and destination size,
 * and it will mirror it in vertical or horizontal direction.
 *********/
/****f* libAfterImage/transform/pad_asimage()
 * NAME 
 * pad_asimage() enlarges ASImage, padding it with specified color on 
 * each side in accordance with requested geometry.
 * SYNOPSIS
 * ASImage *pad_asimage( ASVisual *asv, ASImage *src,
 *                      int dst_x, int dst_y,
 *                      unsigned int to_width,
 *                      unsigned int to_height,
 *                      ARGB32 color,
 *                      ASAltImFormats out_format,
 *                      unsigned int compression_out, int quality );
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * src          - source ASImage
 * dst_x, dst_y - placement of the source image relative to the origin of
 *                destination image
 * to_width     - width of the destination image
 * to_height    - height of the destination image
 * color        - ARGB32 color value to pad with.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out - compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 *********/
/****f* libAfterImage/transform/blur_asimage_gauss()
 * NAME
 * blur_asimage_gauss() Performs Gaussian blurr of the image 
 * ( usefull for drop shadows and the likes ).
 * SYNOPSIS
 * ASImage* blur_asimage_gauss( ASVisual* asv, ASImage* src,
 *                              double horz, double vert,
 *                              ASAltImFormats out_format,
 *                              unsigned int compression_out, 
 * 								int quality );
 * INPUTS
 * asv          - pointer to valid ASVisual structure
 * src          - source ASImage
 * horz         - horizontal radius of the blurr
 * vert         - vertical radius of the blurr
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out - compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 *********/
/****f* libAfterImage/transform/fill_asimage()
 * NAME
 * fill_asimage() - Fills rectangle within the existing ASImage with 
 * specified color.
 * SYNOPSIS
 * Bool fill_asimage( ASVisual *asv, ASImage *im,
 *                    int x, int y, int width, int height,
 *                    ARGB32 color );
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * im            - ASImage to fill with the color
 * x, y          - left-top corner of the rectangle to fill.
 * width, height - size of the rectangle to fill.
 * color         - ARGB32 color value to fill rectangle with.
 * RETURN VALUE
 * True on success, False on failure.
 *********/
/****f* libAfterImage/transform/adjust_asimage_hsv()
 * NAME
 * adjust_asimage_hsv() - adjusts image color properties in HSV colorspace
 * SYNOPSIS
 * ASImage *adjust_asimage_hsv( ASVisual *asv, ASImage *src,
 *                              int offset_x, int offset_y,
 *                              unsigned int to_width,
 *                              unsigned int to_height,
 *                              unsigned int affected_hue,
 *                              unsigned int affected_radius,
 *                              int hue_offset, int saturation_offset,
 *                              int value_offset,
 *                              ASAltImFormats out_format,
 *                              unsigned int compression_out, int quality);
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * src           - ASImage to adjust colors of.
 * offset_x,
 * offset_y      - position on infinite surface tiled with original image,
 *                 of the left-top corner of the area to be used for new
 *                 image.
 * to_width,
 * to_height     - size of the area of the original image to be used
 *                 for new image.
 * affected_hue  - hue in degrees in range 0-360. This allows to limit
 *                 impact of color adjustment to affect only limited
 *                 range of hues.
 * affected_radius Sets the diapason of the range of affected hues.
 * hue_offset    - value by which to change hues in affected range.
 * saturation_offset -
 *                 value by which to change saturation of the pixels in
 *                 affected hue range.
 * value_offset  - value by which to change Value(brightness) of pixels
 *                 in affected hue range.
 * out_format 	- optionally describes alternative ASImage format that
 *                should be produced as the result - XImage, ARGB32, etc.
 * compression_out- compression level of resulting image in range 0-100.
 * quality      - output quality
 * RETURN VALUE
 * returns newly created and encoded ASImage on success, NULL of failure.
 * DESCRIPTION
 * This function will tile original image to specified size with offsets
 * requested, and then it will go though it and adjust hue, saturation and
 * value of those pixels that have specific hue, set by affected_hue/
 * affected_radius parameters. When affected_radius is greater then 180
 * entire image will be adjusted. Note that since grayscale colors have
 * no hue - the will not get adjusted. Only saturation and value will be
 * adjusted in gray pixels.
 * Hue is measured as an angle on a 360 degree circle, The following is
 * relationship of hue values to regular color names :
 * red      - 0
 * yellow   - 60
 * green    - 120
 * cyan     - 180
 * blue     - 240
 * magenta  - 300
 * red      - 360
 *
 * All the hue values in parameters will be adjusted to fall withing
 * 0-360 range.
 *********/

ASImage *scale_asimage( struct ASVisual *asv, ASImage *src,
						int to_width, int to_height,
						ASAltImFormats out_format,
						unsigned int compression_out, int quality );
ASImage *scale_asimage2( ASVisual *asv, ASImage *src, 
		 				int clip_x, int clip_y, 
						int clip_width, int clip_height, 
						int to_width, int to_height,
			   			ASAltImFormats out_format, unsigned int compression_out, int quality );

ASImage *tile_asimage ( struct ASVisual *asv, ASImage *src,
						int offset_x, int offset_y,
  					    int to_width,  int to_height, ARGB32 tint,
						ASAltImFormats out_format,
						unsigned int compression_out, int quality );
ASImage *merge_layers ( struct ASVisual *asv, ASImageLayer *layers, int count,
			  		    int dst_width, int dst_height,
			  		    ASAltImFormats out_format,
						unsigned int compression_out, int quality );
ASImage *make_gradient( struct ASVisual *asv, struct ASGradient *grad,
               			int width, int height, ASFlagType filter,
  			   			ASAltImFormats out_format,
						unsigned int compression_out, int quality  );
ASImage *flip_asimage( struct ASVisual *asv, ASImage *src,
		 		       int offset_x, int offset_y,
			  		   int to_width, int to_height,
					   int flip, ASAltImFormats out_format,
					   unsigned int compression_out, int quality );
ASImage *mirror_asimage( ASVisual *asv, ASImage *src,
				         int offset_x, int offset_y,
						 int to_width,
			             int to_height,
			             Bool vertical, ASAltImFormats out_format,
						 unsigned int compression_out, int quality );
ASImage *pad_asimage(   ASVisual *asv, ASImage *src,
		      			int dst_x, int dst_y,
			  			int to_width,
			  			int to_height,
			  			ARGB32 color,
			  			ASAltImFormats out_format,
			  			unsigned int compression_out, int quality );
ASImage* blur_asimage_gauss( ASVisual* asv, ASImage* src,
	                         double horz, double vert,
                             ASFlagType filter,
                             ASAltImFormats out_format,
							 unsigned int compression_out, int quality);

Bool fill_asimage( ASVisual *asv, ASImage *im,
               	   int x, int y, int width, int height,
				   ARGB32 color );

ASImage* adjust_asimage_hsv( ASVisual *asv, ASImage *src,
				    int offset_x, int offset_y,
	  			    int to_width, int to_height,
					int affected_hue, int affected_radius,
					int hue_offset, int saturation_offset, int value_offset,
					ASAltImFormats out_format,
					unsigned int compression_out, int quality );

/****f* libAfterImage/transform/colorize_asimage_vector()
 * NAME
 * colorize_asimage_vector() creates ASImage from double precision indexed 
 * image data - usefull for scientific visualisation.
 * SYNOPSIS
 * Bool colorize_asimage_vector( ASVisual *asv, ASImage *im,
 * 		             	         ASVectorPalette *palette,
 *                               ASAltImFormats out_format,
 *                               int quality );
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * im            - ASImage to update.
 * palette       - palette to be used in conversion of double precision
 *                 values into colors.
 * out_format 	 - optionally describes alternative ASImage format that
 *                 should be produced as the result - XImage, ARGB32, etc.
 * quality       - output quality
 * RETURN VALUE
 * True on success, False on failure.
 * DESCRIPTION
 * This function will try to convert double precision indexed image data
 * into actuall color image using palette. Original data should be 
 * attached to ASImage using vector member. Operation is relatively fast 
 * and allows representation of scientific data as color image with 
 * dynamically changing palette.
 *********/
/****f* libAfterImage/transform/create_asimage_from_vector()
 * NAME
 * create_asimage_from_vector() - convinience function allowing to 
 * create new ASImage, set its vector data and colorize it using 
 * palette - all in one step.
 * SYNOPSIS
 * ASImage *create_asimage_from_vector( ASVisual *asv, double *vector,
 *                                      unsigned int width,
 *                                      unsigned int height,
 *                                      ASVectorPalette *palette,
 *                                      ASAltImFormats out_format,
 *                                      unsigned int compression,
 *                                      int quality );
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * vector        - data to be attached to new ASImage and used to generate
 *                 RGB image
 * width, height - size of the new image.
 * palette       - palette to be used in conversion of double precision
 *                 values into colors.
 * out_format 	 - optionally describes alternative ASImage format that
 *                 should be produced as the result - XImage, ARGB32, etc.
 * compression_out- compression level of resulting image in range 0-100.
 * quality       - output quality
 * RETURN VALUE
 * New ASImage  on success, NULL on failure.
 * SEE ALSO
 * colorize_asimage_vector(), create_asimage(), set_asimage_vector()
 *********/
/****f* libAfterImage/transform/slice_asimage2()
 * NAME
 * slice_asimage2() - slice ASImage leaving its corners intact, and scaling 
 * the middle part.
 * SYNOPSIS
 * ASImage*
 * slice_asimage2( ASVisual *asv, ASImage *src,
 *             int slice_x_start, int slice_x_end,
 *             int slice_y_start, int slice_y_end,
 *             int to_width,
 *             int to_height,
 *             Bool scaled,
 *             ASAltImFormats out_format,
 *             unsigned int compression_out, int quality );
 * INPUTS
 * asv           - pointer to valid ASVisual structure
 * src           - source ASImage.
 * slice_x_start - ending of the left corners
 * slice_x_end   - begining of the right corners
 * slice_y_start - ending of the top corners
 * slice_y_end   - begining of the bottom corners
 * to_width      - width of the generated image;
 * to_height     - height of the generated image;
 * scaled        - if True - middle part of the image will be scaled, 
 *                 otherwise - tiled;
 * out_format 	 - optionally describes alternative ASImage format that
 *                 should be produced as the result - XImage, ARGB32, etc.;
 * compression_out- compression level of resulting image in range 0-100;
 * quality       - output quality.
 * RETURN VALUE
 * New ASImage  on success, NULL on failure.
 * SEE ALSO
 * scale_asimage(), tile_asimage()
 *********/

Bool
colorize_asimage_vector( ASVisual *asv, ASImage *im,
						 ASVectorPalette *palette,
						 ASAltImFormats out_format,
						 int quality );
ASImage *
create_asimage_from_vector( ASVisual *asv, double *vector,
							int width, int height,
							ASVectorPalette *palette,
							ASAltImFormats out_format,
							unsigned int compression, int quality );
ASImage*
slice_asimage2( ASVisual *asv, ASImage *src,
			   int slice_x_start, int slice_x_end,
			   int slice_y_start, int slice_y_end,
			   int to_width,
			   int to_height,
			   Bool scaled,    /* middle portion */
			   ASAltImFormats out_format,
			   unsigned int compression_out, int quality );

/* same as above with scale = 0 */
ASImage*
slice_asimage (ASVisual *asv, ASImage *src,
			   int slice_x_start, int slice_x_end,
			   int slice_y_start, int slice_y_end,
			   int to_width,
			   int to_height,
			   ASAltImFormats out_format,
			   unsigned int compression_out, int quality);

ASImage *
pixelize_asimage (ASVisual *asv, ASImage *src,
			      int clip_x, int clip_y, int clip_width, int clip_height,
				  int pixel_width, int pixel_height,
				  ASAltImFormats out_format, unsigned int compression_out, int quality );
ASImage *
color2alpha_asimage (ASVisual *asv, ASImage *src,
			         int clip_x, int clip_y, int clip_width, int clip_height,
				     ARGB32 color,
				     ASAltImFormats out_format, unsigned int compression_out, int quality);

#ifdef __cplusplus
}
#endif

#endif
