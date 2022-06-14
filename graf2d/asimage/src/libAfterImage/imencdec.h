#ifndef IMENCDEC_HEADER_FILE_INCLUDED
#define IMENCDEC_HEADER_FILE_INCLUDED

#include "asvisual.h"
#include "blender.h"
#include "scanline.h"
/*#define TRACK_ASIMAGES*/
#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/imencdec.h
 * NAME
 * imencdec defines main structures and function for image storing,
 * extraction and conversion to/from usable formats.
 * DESCRIPTION
 * this header defines structures and functions to be used by outside 
 * applications for reading and writing into ASImages. ASImage pixel 
 * data maybe stored in sevral different formats, and should not be 
 * accessed directly, but only through encoder/decoder facility.
 *
 * SEE ALSO
 * Structures :
 *          ASImageBevel
 *          ASImageDecoder
 *          ASImageOutput
 *
 * Functions :
 *   Encoding :
 *          asimage_add_line(),	asimage_add_line_mono(),
 *          asimage_print_line(), get_asimage_chanmask(),
 *          move_asimage_channel(), copy_asimage_channel(),
 *          copy_asimage_lines()
 *
 *   Decoding
 *          start_image_decoding(), stop_image_decoding(),
 *          asimage_decode_line (), set_decoder_shift(),
 *          set_decoder_back_color()
 *
 *   Output :
 *          start_image_output(), set_image_output_back_color(),
 *          toggle_image_output_direction(), stop_image_output()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******/

struct ASVisual;
struct ASImage;

/****s* libAfterImage/ASImageBevel
 * NAME
 * ASImageBevel describes bevel to be drawn around the image.
 * DESCRIPTION
 * Bevel is used to create 3D effect while drawing buttons, or any other
 * image that needs to be framed. Bevel is drawn using 2 primary colors:
 * one for top and left sides - hi color, and another for bottom and
 * right sides - low color. There are additionally 3 auxiliary colors:
 * hihi is used for the edge of top-left corner, hilo is used for the
 * edge of top-right and bottom-left corners, and lolo is used for the
 * edge of bottom-right corner. Colors are specified as ARGB and contain
 * alpha component, thus allowing for semitransparent bevels.
 *
 * Bevel consists of outline and inline. Outline is drawn outside of the
 * image boundaries and its size adds to image size as the result. Alpha
 * component of the outline is constant. Inline is drawn on top of the
 * image and its alpha component is fading towards the center of the
 * image, thus creating illusion of smooth disappearing edge.
 * SOURCE
 */

typedef struct ASImageBevel
{
#define BEVEL_SOLID_INLINE	(0x01<<0)
	ASFlagType type ;	             /* reserved for future use */

	/* primary bevel colors */
	ARGB32	hi_color ;		/* top and left side color */
	ARGB32	lo_color ;		/* bottom and right side color */		 

	/* these will be placed in the corners */
	ARGB32	hihi_color ;	/* color of the top-left corner */
	ARGB32	hilo_color ;	/* color of the top-right and 
							 * bottom-left corners */
	ARGB32	lolo_color ;	/* color of the bottom-right corner */

	/* outlines define size of the line drawn around the image */
	unsigned short left_outline ; 
	unsigned short top_outline ;
	unsigned short right_outline ; 
	unsigned short bottom_outline ;
	/* inlines define size of the semitransparent line drawn 
	 * inside the image */
	unsigned short left_inline ;
	unsigned short top_inline ;
	unsigned short right_inline ;
	unsigned short bottom_inline ;
}ASImageBevel;
/*******/

/****s* libAfterImage/ASImageDecoder
 * NAME
 * ASImageDecoder describes the status of reading any particular ASImage,
 * as well as providing detail on how it should be done.
 * DESCRIPTION
 * ASImageDecoder works as an abstraction layer and as the way to
 * automate several operations. Most of the transformations in
 * libAfterImage are performed as operations on ASScanline data
 * structure, that holds all or some of the channels of single image
 * scanline. In order to automate data extraction from ASImage into
 * ASScanline ASImageDecoder has been designed.
 *
 * It has following features :
 * 1) All missing scanlines, or channels of scanlines will be filled with
 * supplied back_color
 * 2) It is possible to leave out some channels of the image, extracting
 * only subset of channels. It is done by setting only needed flags in
 * filter member.
 * 3) It is possible to extract sub-image of the image by setting offset_x
 * and offset_y to top-left corner of sub-image, out_width - to width of
 * the sub-image and calling decode_image_scanline method as many times
 * as height of the sub-image.
 * 4) It is possible to apply bevel to extracted sub-image, by setting
 * bevel member to specific ASImageBevel structure.
 *
 * Extracted Scanlines will be stored in buffer and it will be updated
 * after each call to decode_image_scanline().
 * SOURCE
 */

/* low level driver (what data to use - native, XImage or ARGB): */
typedef void (*decode_asscanline_func)( struct ASImageDecoder *imdec, 
										unsigned int skip, int y );
/* high level driver (bevel or not bevel): */
typedef void (*decode_image_scanline_func)
				(struct ASImageDecoder *imdec);

typedef struct ASImageDecoder
{
	struct ASVisual *asv;
	struct ASImage 	*im ;
	ASFlagType 		filter;		 /* flags that mask set of 
								  * channels to be extracted 
								  * from the image */

	ARGB32	 		back_color;  /* we fill missing scanlines 
								  * with this default - black*/
	unsigned int    offset_x,    /* left margin on source image 
								  * before which we skip everything */
					out_width;   /* actual length of the output 
								  * scanline */
	unsigned int 	offset_y,	 /* top margin */
                    out_height;
	ASImageBevel	*bevel;      /* bevel to wrap everything 
								  * around with */

	/* offsets of the drawn bevel baseline on resulting image : */
	int            bevel_left, bevel_top, 
					bevel_right, bevel_bottom ;

	/* scanline buffer containing current scanline */
	struct ASScanline buffer; /* matches the out_width */

	/* internal data : */
	unsigned short    bevel_h_addon, bevel_v_addon ;
	int 			  next_line ;

    struct ASScanline   *xim_buffer; /* matches the size of the 
							   * original XImage */

	decode_asscanline_func     decode_asscanline ;
	decode_image_scanline_func decode_image_scanline ;
}ASImageDecoder;
/********/

/****d* libAfterImage/asimage/quality
 * FUNCTION
 * Defines level of output quality/speed ratio
 * NAME
 * ASIMAGE_QUALITY_POOR there will be no dithering and interpolation used 
 * while transforming 
 * NAME
 * ASIMAGE_QUALITY_FAST there will be no dithering and used while 
 * transforming but interpolation will be used.
 * NAME
 * ASIMAGE_QUALITY_GOOD simplified dithering is performed in addition to 
 * interpolation.
 * NAME
 * ASIMAGE_QUALITY_TOP full dithering and interpolation.
 * NAME
 * ASIMAGE_QUALITY_DEFAULT requests current default setting  - typically
 * same as ASIMAGE_QUALITY_GOOD.
 * NAME
 * MAX_GRADIENT_DITHER_LINES defines number of lines to use for dithering,
 * while rendering gradients, in order to create smooth effect. Higher 
 * number will slow things down, but will create better gradients.
 * SOURCE
 */
#define ASIMAGE_QUALITY_POOR	0
#define ASIMAGE_QUALITY_FAST	1
#define ASIMAGE_QUALITY_GOOD	2
#define ASIMAGE_QUALITY_TOP		3
#define ASIMAGE_QUALITY_DEFAULT	-1

#define MAX_GRADIENT_DITHER_LINES 	ASIMAGE_QUALITY_TOP+1
/*******/


/****s* libAfterImage/asimage/ASImageOutput
 * NAME
 * ASImageOutput describes the output state of the transformation result.
 * It is used to transparently write results into ASImage or XImage with
 * different levels of quality.
 * DESCRIPTION
 * libAfterImage allows for transformation result to be stored in both
 * ASImage ( useful for long term storage and subsequent processing )
 * and XImage ( useful for transfer of the result onto the X Server).
 * At the same time there are 4 different quality levels of output
 * implemented. They differ in the way special technics, like error
 * diffusion and interpolation are applyed, and allow for fine grained
 * selection of quality/speed ratio. ASIMAGE_QUALITY_GOOD should be good
 * enough for most applications.
 * The following additional output features are implemented :
 * 1) Filling of the missing channels with supplied values.
 * 2) Error diffusion to improve quality while converting from internal
 * 	  24.8 format to 8 bit format.
 * 3) Tiling of the output. If tiling_step is greater then 0, then each
 * 	  scanlines will be copied into lines found tiling_step one from
 * 	  another, upto the edge of the image.
 * 4) Reverse order of output. Output image will be mirrored along y
 * 	  axis if bottom_to_top is set to True.
 * NOTES
 * The output_image_scanline method should be called for each scanline
 * to be stored. Convenience functions listed below should be used to
 * safely alter state of the output instead of direct manipulation of
 * the data members. (makes you pity you don't write in C++ doesn't it ?)
 *
 * Also There is a trick in the way how output_image_scanline handles
 * empty scanlines while writing ASImage. If back_color of empty scanline
 * matches back_color of ASImageOutput - then particular line is erased!
 * If back_colors are same - then particular line of ASImage gets filled
 * with the back_color of ASScanline. First approach is usefull when
 * resulting image will be used in subsequent call to merge_layers - in
 * such case knowing back_color of image is good enough and we don't need
 * to store lines with the same color. In case where ASImage will be
 * converted into Pixmap/XImage - second approach is preferable, since
 * that conversion does not take into consideration image's back color -
 * we may want to change it in the future.
 *
 * SEE ALSO
 * start_image_output()
 * set_image_output_back_color()
 * toggle_image_output_direction()
 * stop_image_output()
 * SOURCE
 */
typedef void (*encode_image_scanline_func)( struct ASImageOutput *imout,
											struct ASScanline *to_store );
typedef void (*output_image_scanline_func)( struct ASImageOutput *,
											struct ASScanline *, int );

typedef struct ASImageOutput
{
	struct ASVisual 		*asv;
	struct ASImage  		*im ;
	ASAltImFormats   out_format ;
	CARD32 			 chan_fill[4];
	int 			 buffer_shift;  /* -1 means - buffer is empty,
									 * 0 - no shift,
									 * 8 - use 8 bit precision */
	int 			 next_line ;    /* next scanline to be written */
	unsigned int	 tiling_step;   /* each line written will be 
									 * repeated with this step until 
									 * we exceed image size */
	unsigned int 	 tiling_range;  /* Limits region in which we need 
									 * to tile. If set to 0 then image 
									 * height is used */
	int 	    	 bottom_to_top; /* -1 if we should output in
									 * bottom to top order, 
									 * +1 otherwise*/

	int     		 quality ;		/* see above */

	output_image_scanline_func
		output_image_scanline ;  /* high level interface - division,
								  * error diffusion as well 
								  * as encoding */
	encode_image_scanline_func
		encode_image_scanline ;  /* low level interface - 
								  * encoding only */

	/* internal data members : */
	struct ASScanline 		 buffer[2], *used, *available;
}ASImageOutput;
/********/
/****f* libAfterImage/asimage/start_image_decoding()
 * NAME
 * start_image_decoding()   - allocates and initializes decoder structure.
 * SYNOPSIS
 * ASImageDecoder *start_image_decoding( ASVisual *asv,ASImage *im,
 *                                       ASFlagType filter,
 *                                       int offset_x, int offset_y,
 *                                       unsigned int out_width,
 *                                       unsigned int out_height,
 *                                       ASImageBevel *bevel );
 * INPUTS
 * asv      - pointer to valid ASVisual structure ( needed mostly
 * 			to see if we are in BGR mode or not );
 * im       - ASImage we are going to decode;
 * filter   - bitmask where set bits mark channels that has to be
 * 			decoded.
 * offset_x - left margin inside im, from which we should start
 * 			reading pixel data, effectively clipping source image.
 * offset_y - top margin inside im, from which we should start
 * 			reading scanlines, effectively clipping source image.
 * 			Note that when edge of the image is reached,
 * 			subsequent requests for scanlines will wrap around to
 * 			the top of the image, and not offset_y.
 * out_width- width of the scanline needed. If it is larger then
 * 			source image - then image data will be tiled in it.
 * 			If it is smaller - then image data will be clipped.
 * out_height - height of the output drawable. -1 means that same as
 *          image height. if out_height is greater then image height,
 *          then image will be tiled.
 * bevel    - NULL or pointer to valid ASImageBevel structure if
 * 			decoded data should be overlayed with bevel at the
 * 			time of decoding.
 * RETURN VALUE
 * start_image_decoding() returns pointer to newly allocated
 * ASImageDecoder structure on success, NULL on failure.
 * DESCRIPTION
 * Normal process of reading image data from ASImage consists of
 * 3 steps :
 * 1) start decoding by calling start_image_decoding.
 * 2) call decode_image_scanline() method of returned structure, for
 * each scanline upto desired height of the target image. Decoded data
 * will be returned in buffer member of the ASImageDecoder structure.
 * 3) finish decoding and deallocated all the used memory by calling
 * stop_image_decoding()
 *********/
/****f* libAfterImage/asimage/set_decoder_bevel_geom()
 * NAME
 * set_decoder_bevel_geom() - changes default placement of the bevel on 
 * decoded image. 
 * SYNOPSIS
 * void set_decoder_bevel_geom( ASImageDecoder *imdec, int x, int y,
 *                              unsigned int width, unsigned int height );
 * INPUTS
 * imdec   - pointer to pointer to structure, previously created
 *           by start_image_decoding.
 * x,y     - left top position of the inner border of the Bevel outline
 *           as related to the origin of subimage being decoded.
 * width,
 * height  - widtha and height of the inner border of the bevel outline.
 * DESCRIPTION
 * For example if you only need to render small part of the button, that 
 * is being rendered from transparency image.
 * NOTE
 * This call modifies bevel_h_addon and bevel_v_addon of
 * ASImageDecoder structure.
 *******/
/****f* libAfterImage/asimage/set_decoder_shift()
 * NAME
 * set_decoder_shift() - changes the shift value of decoder - 8 or 0.
 * SYNOPSIS
 * void set_decoder_shift( ASImageDecoder *imdec, int shift );
 * INPUTS
 * imdec   - pointer to pointer to structure, previously created
 *            by start_image_decoding.
 * shift   - new value to be used as the shift while decoding image.
 *           valid values are 8 and 0.
 * DESCRIPTION
 * This function should be used instead of directly modifyeing value of
 * shift memebr of ASImageDecoder structure.
 *******/
/****f* libAfterImage/asimage/set_decoder_back_color()
 * NAME
 * set_decoder_back_color() - changes the back color to be used while
 * decoding the image.
 * SYNOPSIS
 * void set_decoder_back_color( ASImageDecoder *imdec, ARGB32 back_color );
 * INPUTS
 * imdec      - pointer to pointer to structure, previously created
 *              by start_image_decoding.
 * back_color - ARGB32 color value to be used as the background color to
 *              fill empty spaces in decoded ASImage.
 * DESCRIPTION
 * This function should be used instead of directly modifyeing value of
 * back_color memebr of ASImageDecoder structure.
 *******/
/****f* libAfterImage/asimage/stop_image_decoding()
 * NAME
 * stop_image_decoding()    - finishes decoding, frees all allocated
 * memory.
 * SYNOPSIS
 * void stop_image_decoding( ASImageDecoder **pimdec );
 * INPUTS
 * pimdec   - pointer to pointer to structure, previously created
 * 			by start_image_decoding.
 * RETURN VALUE
 * pimdec	- pointer to ASImageDecoder will be reset to NULL.
 * SEE ALSO
 * start_image_decoding()
 *******/

ASImageDecoder *start_image_decoding( struct ASVisual *asv, struct ASImage *im, 
									  ASFlagType filter,
									  int offset_x, int offset_y,
									  unsigned int out_width,
									  unsigned int out_height,
									  ASImageBevel *bevel );
void set_decoder_bevel_geom( ASImageDecoder *imdec, int x, int y,
     	                     unsigned int width, unsigned int height );
void set_decoder_shift( ASImageDecoder *imdec, int shift );
void set_decoder_back_color( ASImageDecoder *imdec, ARGB32 back_color );
void stop_image_decoding( ASImageDecoder **pimdec );

/****f* libAfterImage/asimage/start_image_output()
 * NAME
 * start_image_output() - initializes output structure
 * SYNOPSIS
 * ASImageOutput *start_image_output ( struct ASVisual *asv,
 *                                     ASImage *im,
 *                                     ASAltImFormats format,
 *                                     int shift, int quality );
 * INPUTS
 * asv      - pointer to valid ASVisual structure
 * im       - destination ASImage
 * format   - indicates that output should be written into alternative
 *            format, such as supplied XImage, ARGB32 array etc.
 * shift    - precision of scanline data. Supported values are 0 - no
 *            precision, and 8 - 24.8 precision. Value of that argument
 *            defines by how much scanline data is shifted rightwards.
 * quality  - what algorithms should be used while writing data out, i.e.
 *            full error diffusion, fast error diffusion, no error
 *            diffusion.
 * DESCRIPTION
 * start_image_output() creates and initializes new ASImageOutput
 * structure based on supplied parameters. Created structure can be
 * subsequently used to write scanlines into destination image.
 * It is effectively hiding differences of XImage and ASImage and other
 * available output formats.
 * outpt_image_scanline() method of the structure can be used to write
 * out single scanline. Each written scanlines moves internal pointer to
 * the next image line, and possibly writes several scanlines at once if
 * tiling_step member is not 0.
 **********/
/****f* libAfterImage/asimage/set_image_output_back_color()
 * NAME
 * set_image_output_back_color() - changes background color of output
 * SYNOPSIS
 * void set_image_output_back_color ( ASImageOutput *imout,
 *                                    ARGB32 back_color );
 * INPUTS
 * imout		- ASImageOutput structure, previously created with
 *  			  start_image_output();
 * back_color	- new background color value in ARGB format. This color
 *  			  will be used to fill empty parts of outgoing scanlines.
 *********/
/****f* libAfterImage/asimage/toggle_image_output_direction()
 * NAME
 * toggle_image_output_direction() - reverses vertical direction of output
 * SYNOPSIS
 * void toggle_image_output_direction( ASImageOutput *imout );
 * INPUTS
 * imout		- ASImageOutput structure, previously created with
 *  			  start_image_output();
 * DESCRIPTION
 * reverses vertical direction output. If previously scanlines has
 * been written from top to bottom, for example, after this function is
 * called they will be written in opposite direction. Current line does
 * not change, unless it points to the very first or the very last
 * image line. In this last case it will be moved to the opposing end of
 * the image.
 *********/
/****f* libAfterImage/asimage/stop_image_output()
 * NAME
 * stop_image_output() - finishes output, frees all the allocated memory.
 * SYNOPSIS
 * void stop_image_output( ASImageOutput **pimout );
 * INPUTS
 * pimout		- pointer to pointer to ASImageOutput structure,
 *  			  previously created with call to	start_image_output().
 * RETURN VALUE
 * pimout		- pointer to ASImageOutput will be reset to NULL.
 * DESCRIPTION
 * Completes image output process. Flushes all the internal buffers.
 * Deallocates all the allocated memory. Resets pointer to NULL to
 * avoid dereferencing invalid pointers.
 *********/
ASImageOutput *start_image_output( struct ASVisual *asv, ASImage *im, ASAltImFormats format, int shift, int quality );
void set_image_output_back_color( ASImageOutput *imout, ARGB32 back_color );
void toggle_image_output_direction( ASImageOutput *imout );
void stop_image_output( ASImageOutput **pimout );

#ifdef __cplusplus
}
#endif

#endif
