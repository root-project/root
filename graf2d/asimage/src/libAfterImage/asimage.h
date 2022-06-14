#ifndef ASIMAGE_HEADER_FILE_INCLUDED
#define ASIMAGE_HEADER_FILE_INCLUDED

#include "asvisual.h"
#include "blender.h"
#include "asstorage.h"
#undef TRACK_ASIMAGES
#ifdef __cplusplus
extern "C" {
#endif

struct ASImageBevel;
struct ASImageDecoder;
struct ASImageOutput;
struct ASScanline;

/****h* libAfterImage/asimage.h
 * NAME
 * asimage defines main structures and function for image manipulation.
 * DESCRIPTION
 * libAfterImage provides powerful functionality to load, store
 * and transform images. It allows for smaller memory utilization by
 * utilizing run-length encoding of the image data. There could be
 * different levels of compression selected, allowing to choose best
 * speed/memory ratio.
 *
 * SEE ALSO
 * Structures :
 *          ASImage
 *          ASImageManager
 *          ASImageBevel
 *          ASImageDecoder
 *          ASImageOutput
 *          ASImageLayer
 *          ASGradient
 *
 * Functions :
 *          asimage_init(), asimage_start(), create_asimage(),
 *          clone_asimage(), destroy_asimage()
 *
 *   ImageManager Reference counting and managing :
 *          create_image_manager(), destroy_image_manager(),
 *          store_asimage(), fetch_asimage(), query_asimage(),
 *          dup_asimage(), release_asimage(),
 *          release_asimage_by_name(), forget_asimage(),
 *          safe_asimage_destroy()
 *
 *   Gradients helper functions :
 *          flip_gradient(), destroy_asgradient()
 *
 *   Layers helper functions :
 *          init_image_layers(), create_image_layers(),
 *          destroy_image_layers()
 *
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

#define ASIMAGE_PATH_ENVVAR		"IMAGE_PATH"
#define ASFONT_PATH_ENVVAR		"FONT_PATH"

/****d* libAfterImage/ASAltImFormats
 * NAME 
 * ASAltImFormats identifies what output format should be used for storing 
 * the transformation result. Also identifies what data is currently stored
 * in alt member of ASImage structure.
 * SOURCE
 */
typedef enum {
	ASA_ASImage = 0,
    ASA_XImage,
	ASA_MaskXImage,
	/* temporary XImages to be allocated from static pool of memory :*/
    ASA_ScratchXImage,  
	ASA_ScratchMaskXImage,
	
	ASA_ScratchXImageAndAlpha,

	ASA_ARGB32,
	ASA_Vector,       /* This cannot be used for transformation's result
					   * format */
	ASA_Formats
}ASAltImFormats;
/*******/
/****s* libAfterImage/ASImage
 * NAME
 * ASImage is the main structure to hold image data.
 * DESCRIPTION
 * Images are stored internally split into ARGB channels, each split
 * into scanline. Actuall data is stored using ASStorage container. Inside
 * ASImage data structure we only store IDs pointing to data in ASStorage
 * ASStorage implements reference counting, data compression, 
 * automatic memory defragmentation and other nice things.
 * SEE ALSO
 *  asimage_init()
 *  asimage_start()
 *  create_asimage()
 *  destroy_asimage()
 * SOURCE
 */

struct ASImageAlternative;
struct ASImageManager;

/* magic number identifying ASFont data structure */
#define MAGIC_ASIMAGE            0xA3A314AE

typedef struct ASImage
{

  unsigned long magic ;

  unsigned int width, height;       /* size of the image in pixels */

  /* arrays of storage ids of stored scanlines of particular channel: */
  ASStorageID *alpha,
  			  *red,
			  *green,
			  *blue;
  
  ASStorageID *channels[IC_NUM_CHANNELS]; 
  									/* merely a shortcut so we can
									 * somewhat simplify code in loops */

  ARGB32 back_color ;               /* background color of the image, so
									 * we could discard everything that
									 * matches it, and then restore it
									 * back. */

  struct ASImageAlternative
  {  /* alternative forms of ASImage storage : */
   	XImage *ximage ;                /* pointer to XImage created as the
					 				 * result of transformations whenever
									 * we request it to output into
									 * XImage ( see to_xim parameter ) */
	XImage *mask_ximage ;           /* XImage of depth 1 that could be
									 * used to store mask of the image */
	ARGB32 *argb32 ;                /* array of widthxheight ARGB32
									 * values */
	double *vector ;			    /* scientific data that should be used
									 * in conjunction with
									 * ASScientificPalette to produce
									 * actuall ARGB data */
  }alt;

  struct ASImageManager *imageman;  /* if not null - then image could be
									 * referenced by some other code */
  int                    ref_count ;/* this will tell us what us how many
									 * times */

	
  char                  *name ;     /* readonly copy of image name 
  									 * this name is a hash value used to 
									 * store image in the image-man's hash,
									 * and gets freed automagically on image 
									 * removal from hash */

#define ASIM_DATA_NOT_USEFUL	(0x01<<0)
#define ASIM_VECTOR_TOP2BOTTOM	(0x01<<1)
#define ASIM_XIMAGE_8BIT_MASK	(0x01<<2)
#define ASIM_NO_COMPRESSION		(0x01<<3) /* Do not use compression to 
										   * save some computation time
										   */
#define ASIM_ALPHA_IS_BITMAP	(0x01<<4) 
#define ASIM_RGB_IS_BITMAP		(0x01<<5) 
#define ASIM_XIMAGE_NOT_USEFUL	(0x01<<6)
#define ASIM_NAME_IS_FILENAME	(0x01<<7)

  ASFlagType			 flags ;    /* combination of the above flags */
  
} ASImage;
/*******/

/****d* libAfterImage/LIMITS
 * NAME
 * MAX_IMPORT_IMAGE_SIZE	effectively limits size of the allowed
 *							images to be loaded from files. That is
 * 							needed to be able to filter out corrupt files.
 * NAME
 * MAX_BEVEL_OUTLINE		Limit on bevel outline to be drawn around
 * 							the image.
 * NAME
 * MAX_SEARCH_PATHS		Number of search paths to be used while loading 
 * 							images from files.
 */
#define MAX_IMPORT_IMAGE_SIZE 	8000
#define MAX_BEVEL_OUTLINE 		100
#define MAX_SEARCH_PATHS		8      /* prudently limiting ourselfs */
/******/

/****s* libAfterImage/ASImageManager
 * NAME
 * ASImageManager structure to be used to maintain list of loaded images 
 * for given set of search paths and gamma. Images are named and reference 
 * counted.
 * SOURCE
 */
typedef struct ASImageManager
{
	ASHashTable  *image_hash ;
	/* misc stuff that may come handy : */
	char 	     *search_path[MAX_SEARCH_PATHS+1];
	double 		  gamma ;
}ASImageManager;
/*************/


/* Auxiliary data structures : */
/****s* libAfterImage/ASVectorPalette
 * NAME
 * ASVectorPalette contains pallette allowing us to map double values 
 * in vector image data into actuall ARGB values.
 * SOURCE
 */
typedef struct ASVectorPalette
{
	unsigned int npoints ;
	double *points ;
	CARD16 *channels[IC_NUM_CHANNELS] ;   /* ARGB data for key points. */
	ARGB32  default_color;
}ASVectorPalette;
/*************/

/****s* libAfterImage/asimage/ASImageLayer
 * NAME
 * ASImageLayer specifies parameters of the image superimposition.
 * DESCRIPTION
 * libAfterImage allows for simultaneous superimposition (overlaying) of
 * arbitrary number of images. To facilitate this ASImageLayer structure
 * has been created in order to specify parameters of each image
 * participating in overlaying operation. Images need not to be exact
 * same size. For each image its position on destination is specified
 * via dst_x and dst_y data members. Each image maybe tiled and clipped
 * to fit into rectangle specified by clip_x, clip_y, clip_width,
 * clip_height ( in image coordinates - not destination ). If image is
 * missing, then area specified by dst_x, dst_y, clip_width, clip_height
 * will be filled with solid_color.
 * Entire image will be tinted using tint parameter prior to overlaying.
 * Bevel specified by bevel member will be drawn over image prior to
 * overlaying. Specific overlay method has to be specified.
 * merge_scanlines method is pointer to a function,
 * that accepts 2 ASScanlines as arguments and performs overlaying of
 * first one with the second one.
 * There are 15 different merge_scanline methods implemented in
 * libAfterImage, including alpha-blending, tinting, averaging,
 * HSV and HSL colorspace operations, etc.
 * NOTES
 * ASImageLayer s could be organized into chains using next pointers.
 * Since there could be a need to rearrange layers and maybe bypass some
 * layers - we need to provide for flexibility, while at the same time
 * allowing for simplicity of arrays. As the result next pointers could
 * be used to link together continuous arrays of layer, like so :
 * array1: [layer1(next==NULL)][layer2(next!=NULL)]
 *          ____________________________|
 *          V
 * array2: [layer3(next==NULL)][layer4(next==NULL)][layer5(next!=NULL)]
 *          ________________________________________________|
 *          V
 * array3: [layer6(next==NULL)][layer7(next==layer7)]
 *                                ^______|
 *
 * While iterating throught such a list we check for two conditions -
 * exceeding count of layers and layer pointing to self. When any of
 * that is met - we stopping iteration.
 * SEE ALSO
 * merge_layers()
 * blender.h
 * SOURCE
 */

typedef struct ASImageLayer
{
	ASImage *im;
	ARGB32   solid_color ;              /* If im == NULL, then fill
								  		 * the area with this color. */

	int dst_x, dst_y;			  		/* placement in overall
								  		 * composition */

	/* clip area could be partially outside of the image -
	 * image gets tiled in it */
	int clip_x, clip_y;
	unsigned int clip_width, clip_height;

	ARGB32 tint ;                  		/* if 0 - no tint */
	struct ASImageBevel *bevel ;  		/* border to wrap layer with
								  		 * (for buttons, etc.)*/

	/* if image is clipped then we need to specify offsets of bevel as
	 * related to clipped rectangle. Normally it should be :
	 * 0, 0, im->width, im->height. And if width/height left 0 - it will
	 * default to this values. Note that clipped image MUST be entirely
	 * inside the bevel rectangle. !!!*/
	int bevel_x, bevel_y;
	unsigned int bevel_width, bevel_height;

	int merge_mode ;                     	/* reserved for future use */
	merge_scanlines_func merge_scanlines ;	/* overlay method */
	struct ASImageLayer *next;              /* optional pointer to next
											 * layer. If it points to
											 * itself - then end of the
											 * chain.*/
	void *data;                           	/* hook to hung data on */
}ASImageLayer;
/********/

/****d* libAfterImage/asimage/GRADIENT_TYPE_flags
 * FUNCTION
 * Combination of this flags defines the way gradient is rendered.
 * NAME
 * GRADIENT_TYPE_DIAG when set it will cause gradient's direction to be 
 * rotated by 45 degrees
 * NAME
 * GRADIENT_TYPE_ORIENTATION will cause gradient direction to be rotated 
 * by 90 degrees. When combined with GRADIENT_TYPE_DIAG - rotates gradient 
 * direction by 135 degrees.
 * SOURCE
 */
#define GRADIENT_TYPE_DIAG          (0x01<<0)
#define GRADIENT_TYPE_ORIENTATION   (0x01<<1)
#define GRADIENT_TYPE_MASK          (GRADIENT_TYPE_ORIENTATION| \
									 GRADIENT_TYPE_DIAG)
/********/

/****d* libAfterImage/asimage/GRADIENT_TYPE
 * FUNCTION
 * This are named combinations of above flags to define type of gradient.
 * NAME 
 * GRADIENT_Left2Right normal left-to-right gradient.
 * NAME 
 * GRADIENT_TopLeft2BottomRight diagonal top-left to bottom-right.
 * NAME 
 * GRADIENT_Top2Bottom vertical top to bottom gradient.
 * NAME 
 * GRADIENT_BottomLeft2TopRight diagonal bottom-left to top-right.
 * SOURCE
 */
#define GRADIENT_Left2Right        		0
#define GRADIENT_TopLeft2BottomRight	GRADIENT_TYPE_DIAG
#define GRADIENT_Top2Bottom				GRADIENT_TYPE_ORIENTATION
#define GRADIENT_BottomLeft2TopRight    (GRADIENT_TYPE_DIAG| \
 									 	 GRADIENT_TYPE_ORIENTATION)
/********/

/****s* libAfterImage/ASGradient
 * NAME
 * ASGradient describes how gradient is to be drawn.
 * DESCRIPTION
 * libAfterImage includes functionality to draw multipoint gradients in
 * 4 different directions left->right, top->bottom and diagonal
 * lefttop->rightbottom and bottomleft->topright. Each gradient described
 * by type, number of colors (or anchor points), ARGB values for each
 * color and offsets of each point from the beginning of gradient in
 * fractions of entire length. There should be at least 2 anchor points.
 * very first point should have offset of 0. and last point should have
 * offset of 1. Gradients are drawn in ARGB colorspace, so it is possible
 * to have semitransparent gradients.
 * SEE ALSO
 * make_gradient()
 * SOURCE
 */

typedef struct ASGradient
{
	int			type;     /* see GRADIENT_TYPE above */
	
	int         npoints;  /* number of anchor points */
	ARGB32     *color;    /* ARGB color values for each anchor point*/
	double     *offset;   /* offset of each point from the beginning in
						   * fractions of entire length */
}ASGradient;
/********/

/****d* libAfterImage/asimage/flip
 * FUNCTION
 * This are flags that define rotation angle.
 * NAME
 * FLIP_VERTICAL defines rotation of 90 degrees counterclockwise.
 * NAME
 * FLIP_UPSIDEDOWN defines rotation of 180 degrees counterclockwise.
 * combined they define rotation of 270 degrees counterclockwise.
 * SOURCE
 */
#define FLIP_VERTICAL       (0x01<<0)
#define FLIP_UPSIDEDOWN		(0x01<<1)
#define FLIP_MASK			(FLIP_UPSIDEDOWN|FLIP_VERTICAL)
/********/
/****d* libAfterImage/asimage/tint
 * FUNCTION
 * We use 32 bit ARGB values to define how tinting should be done.
 * The formula for tinting particular channel data goes like that:
 * tinted_data = (image_data * tint)/128
 * So if tint channel value is greater then 127 - same channel will be
 * brighter in destination image; if it is lower then 127 - same channel
 * will be darker in destination image. Tint channel value of 127
 * ( or 0x7F hex ) does not change anything.
 * Alpha channel is tinted as well, allowing for creation of
 * semitransparent images. Calculations are performed in 24.8 format -
 * with 8 bit precision. Result is saturated to avoid overflow, and
 * precision is carried over to next pixel ( error diffusion ), when con
 * verting 24.8 to 8 bit format.
 * NAME
 * TINT_NONE special value that disables tinting
 * NAME
 * TINT_LEAVE_SAME also disables tinting.
 * SOURCE
 */
#define TINT_NONE			0
#define TINT_LEAVE_SAME     (0x7F7F7F7F)
#define TINT_HALF_DARKER	(0x3F3F3F3F)
#define TINT_HALF_BRIGHTER	(0xCFCFCFCF)
#define TINT_RED			(0x7F7F0000)
#define TINT_GREEN			(0x7F007F00)
#define TINT_BLUE			(0x7F00007F)
/********/
/****d* libAfterImage/asimage/compression
 * FUNCTION
 * Defines the level of compression to attempt on ASImage scanlines.
 * NAME 
 * ASIM_COMPRESSION_NONE defined as 0 - disables compression.
 * NAME 
 * ASIM_COMPRESSION_FULL defined as 100 - highest compression level.
 * Anything in between 0 and 100 will cause only part of the scanline to 
 * be compressed. 
 * This is obsolete. Now all images are compressed if possible.
 ********/
#define ASIM_COMPRESSION_NONE       0
#define ASIM_COMPRESSION_FULL	   100

extern Bool asimage_use_mmx ;

/****f* libAfterImage/asimage/asimage_init()
 * NAME 
 * asimage_init() frees datamembers of the supplied ASImage structure, and
 * 	initializes it to all 0.
 * SYNOPSIS
 * void asimage_init (ASImage * im, Bool free_resources);
 * INPUTS
 * im             - pointer to valid ASImage structure
 * free_resources - if True will make function attempt to free
 *                  all non-NULL pointers.
 *********/
/****f* libAfterImage/asimage/flush_asimage_cache()
 * NAME
 * flush_asimage_cache() destroys XImage and mask XImage kept from previous 
 * conversions to/from X Pixmap.
 * SYNOPSIS
 * void flush_asimage_cache (ASImage * im );
 * INPUTS
 * im             - pointer to valid ASImage structure
 *********/
/****f* libAfterImage/asimage/asimage_start()
 * NAME
 * asimage_start() Allocates memory needed to store scanline of the image 
 * of supplied size. Assigns all the data members valid values. Makes sure 
 * that ASImage structure is ready to store image data.
 * SYNOPSIS
 * void asimage_start (ASImage * im, unsigned int width,
 *                                   unsigned int height,
 *                                   unsigned int compression);
 * INPUTS
 * im          - pointer to valid ASImage structure
 * width       - width of the image
 * height      - height of the image
 * compression - level of compression to perform on image data.
 *               compression has to be in range of 0-100 with 100
 *               signifying highest level of compression.
 * NOTES
 * In order to resize ASImage structure after asimage_start() has been
 * called, asimage_init() must be invoked to free all the memory, and
 * then asimage_start() has to be called with new dimensions.
 *********/
/****f* libAfterImage/asimage/create_asimage()
 * NAME
 * create_asimage() Performs memory allocation for the new ASImage 
 * structure, as well as initialization of allocated structure based on 
 * supplied parameters.
 * SYNOPSIS
 * ASImage *create_asimage( unsigned int width,
 *                          unsigned int height,
 *                          unsigned int compression);
 * INPUTS
 * width       - desired image width
 * height      - desired image height
 * compression - compression level in new ASImage( see asimage_start()
 *               for more ).
 * RETURN VALUE
 * Pointer to newly allocated and initialized ASImage structure on
 * Success. NULL in case of any kind of error - that should never happen.
 *********/
/****f* libAfterImage/asimage/clone_asimage()
 * NAME 
 * clone_asimage()
 * SYNOPSIS
 * ASImage *clone_asimage(ASImage *src, ASFlagType filter );
 * INPUTS
 * src      - original ASImage.
 * filter   - bitmask of channels to be copied from one image to another.
 * RETURN VALUE
 * New ASImage, as a copy of original image.
 * DESCRIPTION
 * Creates exact clone of the original ASImage, with same compression,
 * back_color and rest of the attributes. Only ASImage data will be
 * carried over. Any attached alternative forms of images (XImages, etc.)
 * will not be copied. Any channel with unset bit in filter will not be
 * copied. Image name, ASImageManager and ref_count will not be copied -
 * use store_asimage() afterwards and make sure you use different name,
 * to avoid clashes with original image.
 *********/
/****f* libAfterImage/asimage/destroy_asimage()
 * NAME
 * destroy_asimage() frees all the memory allocated for specified ASImage. 
 * SYNOPSIS
 * void destroy_asimage( ASImage **im );
 * INPUTS
 * im				- pointer to valid ASImage structure.
 * NOTES
 * If there was XImage attached to it - it will be deallocated as well.
 * EXAMPLE
 * asview.c: ASView.5
 *********/
/****f* libAfterImage/asimage/asimage_replace()
 * NAME
 * asimage_replace() will replace ASImage's data using data from 
 * another ASImage
 * SYNOPSIS
 * Bool asimage_replace (ASImage *im, ASImage *from);
 * INPUTS
 * im				- pointer to valid ASImage structure.
 * from				- pointer to ASImage from which to take the data.
 * NOTES
 * this function updates image without reallocating structure itself, which 
 * means that all pointers to it will still be valid. If that function 
 * succeeds - [from] ASImage will become unusable and should be deallocated 
 * using free() call.
 *********/
void asimage_init (ASImage * im, Bool free_resources);
void flush_asimage_cache( ASImage *im );
void asimage_start (ASImage * im, unsigned int width, unsigned int height, unsigned int compression);
ASImage *create_asimage( unsigned int width, unsigned int height, unsigned int compression);
ASImage *create_static_asimage( unsigned int width, unsigned int height, unsigned int compression);
ASImage *clone_asimage( ASImage *src, ASFlagType filter );
void destroy_asimage( ASImage **im );
Bool asimage_replace (ASImage *im, ASImage *from);
/****f* libAfterImage/asimage/set_asimage_vector()
 * NAME
 * set_asimage_vector() This function replaces contents of the vector 
 * member of ASImage structure with new double precision data.
 * SYNOPSIS
 * set_asimage_vector( ASImage *im, register double *vector );
 * INPUTS
 * im				- pointer to valid ASImage structure.
 * vector           - scientific data to attach to the image.
 * NOTES
 * Data must have size of width*height ahere width and height are size of 
 * the ASImage.
 *********/
Bool set_asimage_vector( ASImage *im, register double *vector );
/****f* libAfterImage/asimage/vectorize_asimage()
 * NAME
 * vectorize_asimage() This function replaces contents of the vector 
 * member of ASImage structure with new double precision data, generated 
 * from native ARGB32 image contents. Color palette is generated by 
 * indexing color values using max_colors, dither and opaque_threshold 
 * parameters.
 * SYNOPSIS
 * ASVectorPalette* vectorize_asimage( ASImage *im, 
 *                                     unsigned int max_colors, 
 *                                     unsigned int dither,  
 *                                     int opaque_threshold );
 * INPUTS
 * im				- pointer to valid ASImage structure.
 * max_colors       - maximum size of the colormap.
 * dither           - number of bits to strip off the color data ( 0...7 )
 * opaque_threshold - alpha channel threshold at which pixel should be
 *                    treated as opaque
 * RETURN VALUE
 * pointer to the ASVectorPalette structure that could be used for 
 * reverse conversion from double values to ARGB32. 
 * NOTES
 * alt.vector member of the supplied ASImage will be replaced and will 
 * contain WIDTHxHEIGHT double values representing generated scientific 
 * data.
 *********/
ASVectorPalette* vectorize_asimage( ASImage *im, unsigned int max_colors, 
						unsigned int dither,  int opaque_threshold );


/****f* libAfterImage/asimage/create_image_manager()
 * NAME
 * create_image_manager()  create ASImage management and reference 
 * counting object.
 * SYNOPSIS
 * ASImageManager *create_image_manager( ASImageManager *reusable_memory,
 *                                       double gamma, ... );
 * INPUTS
 * reusable_memory - optional pointer to a block of memory to be used to
 *                   store ASImageManager object.
 * double gamma    - value of gamma correction to be used while loading
 *                   images from files.
 * ...             - NULL terminated list of up to 8 PATH strings to list
 *                   locations at which images could be found.
 * DESCRIPTION
 * Creates ASImageManager object in memory and initializes it with
 * requested gamma value and PATH list. This Object will contain a hash
 * table referencing all the loaded images. When such object is used while
 * loading images from the file - gamma and PATH values will be used, so
 * that all the loaded and referenced images will have same parameters.
 * File name will be used as the image name, and if same file is attempted
 * to be loaded again - instead reference will be incremented, and
 * previously loaded image will be retyrned. All the images stored in
 * ASImageManager's table will contain a back pointer to it, and they must
 * be deallocated only by calling release_asimage(). destroy_asimage() will
 * refuse to deallocate such an image.
 *********/
/****f* libAfterImage/asimage/destroy_image_manager()
 * NAME 
 * destroy_image_manager() destroy management obejct.
 * SYNOPSIS
 * void destroy_image_manager( struct ASImageManager *imman, 
 * 							   Bool reusable );
 * INPUTS
 * imman           - pointer to ASImageManager object to be deallocated
 * reusable        - if True, then memory that holds object itself will
 *                   not be deallocated. Usefull when object is created
 *                   on stack.
 * DESCRIPTION
 * Destroys all the referenced images, PATH values and if reusable is False,
 * also deallocates object's memory.
 *********/
ASImageManager *create_image_manager( struct ASImageManager *reusable_memory, double gamma, ... );
void     destroy_image_manager( struct ASImageManager *imman, Bool reusable );

/****f* libAfterImage/asimage/store_asimage()
 * NAME
 * store_asimage()  add ASImage to the reference.
 * SYNOPSIS
 * Bool store_asimage( ASImageManager* imageman, ASImage *im, 
 * 					   const char *name );
 * INPUTS
 * imageman        - pointer to valid ASImageManager object.
 * im              - pointer to the image to be stored.
 * name            - unique name of the image.
 * DESCRIPTION
 * Adds specifyed image to the ASImageManager's list of referenced images.
 * Stored ASImage could be deallocated only by release_asimage(), or when
 * ASImageManager object itself is destroyed.
 *********/
/****f* libAfterImage/asimage/relocate_asimage()
 * NAME
 * relocate_asimage()  relocate ASImage into a different image manager.
 * SYNOPSIS
 * void	 relocate_asimage( ASImageManager* to_imageman, ASImage *im );
 * INPUTS
 * to_imageman        - pointer to valid ASImageManager object.
 * im              - pointer to the image to be stored.
 * DESCRIPTION
 * Moves image from one ASImageManager's list of referenced images into 
 * another ASImageManager. Reference count will be kept the same.
 *********/
Bool     store_asimage( ASImageManager* imageman, ASImage *im, const char *name );
void	 relocate_asimage( ASImageManager* to_imageman, ASImage *im );

/****f* libAfterImage/asimage/fetch_asimage()
 * NAME
 * fetch_asimage()
 * NAME
 * query_asimage() 
 * SYNOPSIS
 * ASImage *fetch_asimage( ASImageManager* imageman, const char *name );
 * ASImage *query_asimage( ASImageManager* imageman, const char *name );
 * INPUTS
 * imageman        - pointer to valid ASImageManager object.
 * name            - unique name of the image.
 * DESCRIPTION
 * Looks for image with the name in ASImageManager's list and if found,
 * returns pointer to it. Note that query_asimage() does not increment 
 * reference count, while fetch_asimage() does. Therefore if fetch_asimage()
 * is used - release_asimage() should be called , when image is no longer 
 * in use.
 *********/
ASImage *fetch_asimage( ASImageManager* imageman, const char *name );
ASImage *query_asimage( ASImageManager* imageman, const char *name );

/****f* libAfterImage/asimage/dup_asimage()
 * NAME
 * dup_asimage() increment reference count of stored ASImage.
 * SYNOPSIS
 * ASImage *dup_asimage( ASImage* im );
 * INPUTS
 * im              - pointer to already referenced image.
 *********/
ASImage *dup_asimage  ( ASImage* im );         /* increment ref countif applicable */

/****f* libAfterImage/asimage/release_asimage()
 * NAME
 * release_asimage() decrement reference count for given ASImage. 
 * NAME
 * release_asimage_by_name() decrement reference count for ASImage 
 * identifyed by its name. 
 * SYNOPSIS
 * int	release_asimage( ASImage *im );
 * int release_asimage_by_name( ASImageManager *imman, char *name );
 * INPUTS
 * im              - pointer to already referenced image.
 * imageman        - pointer to valid ASImageManager object.
 * name            - unique name of the image.
 * DESCRIPTION
 * Decrements reference count on the ASImage object and destroys it if
 * reference count is below zero.
 *********/
int      release_asimage( ASImage *im );
int		 release_asimage_by_name( ASImageManager *imman, char *name );

/****f* libAfterImage/asimage/forget_asimage()
 * NAME
 * forget_asimage() remove ASImage from ASImageManager's hash by pointer.
 * NAME
 * forget_asimage_name() remove ASImage from ASImageManager's hash by its 
 * name.
 * SYNOPSIS
 * void	 forget_asimage( ASImage *im );
 * void  forget_asimage_name( ASImageManager *imman, const char *name );
 * INPUTS
 * im       pointer to already referenced image.
 * imageman pointer to valid ASImageManager object.
 * name     unique name of the image.
 *********/
void	 forget_asimage( ASImage *im );
void     forget_asimage_name( ASImageManager *imman, const char *name );

/****f* libAfterImage/safe_asimage_destroy()
 * NAME
 * safe_asimage_destroy() either release or destroy asimage, checking
 * if it is attached to ASImageManager.
 * SYNOPSIS
 * int		 safe_asimage_destroy( ASImage *im );
 * INPUTS
 * im  pointer to and ASImage structure.
 *********/
int		 safe_asimage_destroy( ASImage *im );

/****f* libAfterImage/print_asimage_manager()
 * NAME
 * print_asimage_manager() prints list of images referenced in given 
 * ASImageManager structure.
 *********/
void     print_asimage_manager(ASImageManager *imageman);

/****f* libAfterImage/asimage/destroy_asgradient()
 * NAME
 * destroy_asgradient() - destroy ASGradient structure, deallocating all
 * 						  associated memory
 *********/
void  destroy_asgradient( ASGradient **pgrad );

/****f* libAfterImage/asimage/flip_gradient()
 * NAME 
 * flip_gradient()    - rotates gradient in 90 degree increments.
 * SYNOPSIS
 * ASGradient *flip_gradient( ASGradient *orig, int flip );
 * INPUTS
 * orig       - pointer to original ASGradient structure to be rotated.
 * flip       - value defining desired rotation.
 * RETURN VALUE
 * Same as original gradient if flip is 0. New gradient structure in any
 * other case.
 * DESCRIPTION
 * Rotates ( flips ) gradient data in 90 degree increments. When needed
 * order of points is reversed.
 *********/
ASGradient *flip_gradient( ASGradient *orig, int flip );
/****f* libAfterImage/asimage/init_image_layers()
 * NAME 
 * init_image_layers()    - initialize set of ASImageLayer structures.
 * SYNOPSIS
 * void init_image_layers( register ASImageLayer *l, int count );
 * INPUTS
 * l              - pointer to valid ASImageLayer structure.
 * count          - number of elements to initialize.
 * DESCRIPTION
 * Initializes array on ASImageLayer structures to sensible defaults.
 * Basically - all zeros and merge_scanlines == alphablend_scanlines.
 *********/
void init_image_layers( register ASImageLayer *l, int count );
/****f* libAfterImage/asimage/create_image_layers()
 * NAME 
 * create_image_layers()  - allocate and initialize set of ASImageLayer's.
 * SYNOPSIS
 * ASImageLayer *create_image_layers( int count );
 * INPUTS
 * count       - number of ASImageLayer structures in allocated array.
 * RETURN VALUE
 * Pointer to newly allocated and initialized array of ASImageLayer
 * structures on Success. NULL in case of any kind of error - that
 * should never happen.
 * DESCRIPTION
 * Performs memory allocation for the new array of ASImageLayer
 * structures, as well as initialization of allocated structure to
 * sensible defaults - merge_func will be set to alphablend_scanlines.
 *********/
ASImageLayer *create_image_layers( int count );
/****f* libAfterImage/asimage/destroy_image_layers()
 * NAME 
 * destroy_image_layers() - destroy set of ASImageLayer structures.
 * SYNOPSIS
 * void destroy_image_layers( register ASImageLayer *l,
 *                            int count,
 *                            Bool reusable );
 * INPUTS
 * l			- pointer to pointer to valid array of ASImageLayer
 *                structures.
 * count        - number of structures in array.
 * reusable     - if True - then array itself will not be deallocates -
 *                    which is usable when it was allocated on stack.
 * DESCRIPTION
 * frees all the memory allocated for specified array of ASImageLayer s.
 * If there was ASImage and/or ASImageBevel attached to it - it will be
 * deallocated as well.
 *********/
void destroy_image_layers( register ASImageLayer *l, int count, Bool reusable );

/****f* libAfterImage/asimage/asimage_add_line()
 * NAME
 * asimage_add_line()
 * SYNOPSIS
 * size_t asimage_add_line ( ASImage * im, ColorPart color,
 *                           CARD32 * data, unsigned int y);
 * INPUTS
 * im      - pointer to valid ASImage structure
 * color   - color channel's number
 * data    - raw channel data of 32 bits per pixel - only lowest 8 bits
 *           gets encoded.
 * y       - image row starting with 0
 * RETURN VALUE
 * asimage_add_line() return size of the encoded channel scanline in
 * bytes. On failure it will return 0.
 * DESCRIPTION
 * Encodes raw data of the single channel into ASImage channel scanline.
 * based on compression level selected for this ASImage all or part of
 * the scanline will be RLE encoded.
 *********/
/****f* libAfterImage/asimage/asimage_add_line_mono()
 * NAME
 * asimage_add_line_mono()
 * SYNOPSIS
 * size_t asimage_add_line_mono ( ASImage * im, ColorPart color,
 *                                CARD8 value, unsigned int y);
 * INPUTS
 * im				- pointer to valid ASImage structure
 * color			- color channel's number
 * value			- value for the channel
 * y 				- image row starting with 0
 * RETURN VALUE
 * asimage_add_line_mono() return size of the encoded channel scanline
 * in bytes. On failure it will return 0.
 * DESCRIPTION
 * encodes ASImage channel scanline to have same color components
 * value in every pixel. Useful for vertical gradients for example.
 *********/
/****f* libAfterImage/asimage/get_asimage_chanmask()
 * NAME
 * get_asimage_chanmask()
 * SYNOPSIS
 * ASFlagType get_asimage_chanmask( ASImage *im);
 * INPUTS
 * im         - valid ASImage object.
 * DESCRIPTION
 * goes throu all the scanlines of the ASImage and toggles bits 
 * representing those components that have at least some data.
 *********/
/****f* libAfterImage/asimage/move_asimage_channel()
 * NAME
 * move_asimage_channel()
 * SYNOPSIS
 * void move_asimage_channel( ASImage *dst, int channel_dst,
 *                            ASImage *src, int channel_src );
 * INPUTS
 * dst         - ASImage which will have its channel substituted;
 * channel_dst - what channel to move data to;
 * src         - ASImage which will donate its channel to dst;
 * channel_src - what source image channel to move data from.
 * DESCRIPTION
 * MOves channel data from one ASImage to another, while discarding
 * what was already in destination's channel.
 * NOTES
 * Source image (donor) will loose its channel data, as it will be
 * moved to destination ASImage. Also there is a condition that both
 * images must be of the same width - otherwise function returns
 * without doing anything. If height is different - the minimum of
 * two will be used.
 *********/
/****f* libAfterImage/asimage/copy_asimage_channel()
 * NAME
 * copy_asimage_channel()
 * SYNOPSIS
 * void copy_asimage_channel( ASImage *dst, int channel_dst,
 *                            ASImage *src, int channel_src );
 * INPUTS
 * dst         - ASImage which will have its channel substituted;
 * channel_dst - what channel to copy data to;
 * src         - ASImage which will donate its channel to dst;
 * channel_src - what source image channel to copy data from.
 * DESCRIPTION
 * Same as move_asimage_channel() but makes copy of channel's data
 * instead of simply moving it from one image to another.
 *********/
/****f* libAfterImage/asimage/copy_asimage_lines()
 * NAME
 * copy_asimage_lines()
 * SYNOPSIS
 * void copy_asimage_lines( ASImage *dst, unsigned int offset_dst,
 *                          ASImage *src, unsigned int offset_src,
 *                          unsigned int nlines, ASFlagType filter );
 * INPUTS
 * dst         - ASImage which will have its channel substituted;
 * offset_dst  - scanline in destination image to copy to;
 * src         - ASImage which will donate its channel to dst;
 * offset_src  - scanline in source image to copy data from;
 * nlines      - number of scanlines to be copied;
 * filter      - specifies what channels should be copied.
 * DESCRIPTION
 * Makes copy of scanline data for continuos set of scanlines, affecting
 * only those channels marked in filter.
 * NOTE
 * Images must be of the same width.
 *********/
size_t asimage_add_line (ASImage * im, ColorPart color, CARD32 * data, unsigned int y);
size_t asimage_add_line_mono (ASImage * im, ColorPart color, CARD8 value, unsigned int y);
size_t asimage_add_line_bgra (ASImage * im, register CARD32 * data, unsigned int y);

ASFlagType get_asimage_chanmask( ASImage *im);
int check_asimage_alpha (ASVisual *asv, ASImage *im );
int asimage_decode_line (ASImage * im, ColorPart color, CARD32 * to_buf, unsigned int y, unsigned int skip, unsigned int out_width);
void move_asimage_channel( ASImage *dst, int channel_dst, ASImage *src, int channel_src );
void copy_asimage_channel( ASImage *dst, int channel_dst, ASImage *src, int channel_src );
void copy_asimage_lines( ASImage *dst, unsigned int offset_dst,
                    	 ASImage *src, unsigned int offset_src,
						 unsigned int nlines, ASFlagType filter );
/****d* libAfterImage/asimage/verbosity
 * FUNCTION
 * This are flags that define what should be printed by
 * asimage_print_line():
 * 	VRB_LINE_SUMMARY	- print only summary for each scanline
 * 	VRB_LINE_CONTENT 	- print summary and data for each scanline
 * 	VRB_CTRL_EXPLAIN 	- print summary, data and control codes for each
 * 						  scanline
 * SOURCE
 */
#define VRB_LINE_SUMMARY 	(0x01<<0)
#define VRB_LINE_CONTENT 	(0x01<<1)
#define VRB_CTRL_EXPLAIN 	(0x01<<2)
#define VRB_EVERYTHING		(VRB_LINE_SUMMARY|VRB_CTRL_EXPLAIN| \
							 VRB_LINE_CONTENT)
/*********/
/****f* libAfterImage/asimage/asimage_print_line()
 * NAME
 * asimage_print_line()
 * SYNOPSIS
 * 	unsigned int asimage_print_line ( ASImage * im, ColorPart color,
 *									  unsigned int y,
 * 									  unsigned long verbosity);
 * INPUTS
 * im				- pointer to valid ASImage structure
 * color			- color channel's number
 * y 				- image row starting with 0
 * verbosity		- verbosity level - any combination of flags is
 *                  allowed
 * RETURN VALUE
 * amount of memory used by this particular channel of specified
 * scanline.
 * DESCRIPTION
 * asimage_print_line() prints data stored in specified image scanline
 * channel. That may include simple summary of how much memory is used,
 * actual visible data, and/or RLE control codes. That helps to see
 * how effectively data is encoded.
 *
 * Useful mostly for debugging purposes.
 *********/
unsigned int asimage_print_line (ASImage * im, ColorPart color,
				 unsigned int y, unsigned long verbosity);
void print_asimage( ASImage *im, int flags, char * func, int line );

void print_asimage_func (ASHashableValue value);
#define print_asimage_ptr (ptr)  print_asimage_func(AS_HASHABLE(ptr))
void print_asimage_registry();                 /* TRACK_ASIMAGES must be defined for this to work */
void purge_asimage_registry();


/* the following 5 macros will in fact unfold into huge but fast piece of code : */
/* we make poor compiler work overtime unfolding all this macroses but I bet it  */
/* is still better then C++ templates :)									     */

#define ENCODE_SCANLINE(im,src,y) \
do{	asimage_add_line((im), IC_RED,   (src).red,   (y)); \
   	asimage_add_line((im), IC_GREEN, (src).green, (y)); \
   	asimage_add_line((im), IC_BLUE,  (src).blue,  (y)); \
	if( get_flags((src).flags,SCL_DO_ALPHA))asimage_add_line((im), IC_ALPHA, (src).alpha, (y)); \
  }while(0)

#define SCANLINE_FUNC(f,src,dst,scales,len) \
do{	if( (src).offset_x > 0 || (dst).offset_x > 0 ) \
		LOCAL_DEBUG_OUT( "(src).offset_x = %d. (dst).offset_x = %d", (src).offset_x, (dst).offset_x ); \
	f((src).red+(src).offset_x,  (dst).red+(dst).offset_x,  (scales),(len));		\
	f((src).green+(src).offset_x,(dst).green+(dst).offset_x,(scales),(len)); 		\
	f((src).blue+(src).offset_x, (dst).blue+(dst).offset_x, (scales),(len));   	\
	if(get_flags((src).flags,SCL_DO_ALPHA)) f((src).alpha+(src).offset_x,(dst).alpha+(dst).offset_x,(scales),(len)); \
  }while(0)

#define SCANLINE_FUNC_FILTERED(f,src,dst,scales,len) \
do{	if( (src).offset_x > 0 || (dst).offset_x > 0 ) \
		LOCAL_DEBUG_OUT( "(src).offset_x = %d. (dst).offset_x = %d", (src).offset_x, (dst).offset_x ); \
    if(get_flags((src).flags,SCL_DO_RED)) f((src).red+(src).offset_x,  (dst).red+(dst).offset_x,  (scales),(len));        \
    if(get_flags((src).flags,SCL_DO_GREEN)) f((src).green+(src).offset_x,(dst).green+(dst).offset_x,(scales),(len));        \
    if(get_flags((src).flags,SCL_DO_BLUE)) f((src).blue+(src).offset_x, (dst).blue+(dst).offset_x, (scales),(len));    \
	if(get_flags((src).flags,SCL_DO_ALPHA)) f((src).alpha+(src).offset_x,(dst).alpha+(dst).offset_x,(scales),(len)); \
  }while(0)

#define CHOOSE_SCANLINE_FUNC(r,src,dst,scales,len) \
 switch(r)                                              							\
 {  case 0:	SCANLINE_FUNC(shrink_component11,(src),(dst),(scales),(len));break;   	\
	case 1: SCANLINE_FUNC(shrink_component, (src),(dst),(scales),(len));	break;  \
	case 2: SCANLINE_FUNC(enlarge_component_dumb,(src),(dst),(scales),(len));break ;\
	case 3:	SCANLINE_FUNC(enlarge_component12,(src),(dst),(scales),(len));break ; 	\
	case 4:	SCANLINE_FUNC(enlarge_component23,(src),(dst),(scales),(len));break;  	\
	default:SCANLINE_FUNC(enlarge_component,  (src),(dst),(scales),(len));        	\
 }

#define SCANLINE_MOD(f,src,p,len) \
do{	f((src).red+(src).offset_x,(p),(len));		\
	f((src).green+(src).offset_x,(p),(len));		\
	f((src).blue+(src).offset_x,(p),(len));		\
	if(get_flags((src).flags,SCL_DO_ALPHA)) f((src).alpha+(src).offset_x,(p),(len));\
  }while(0)

#define SCANLINE_MOD_FILTERED(f,src,p,len) \
do{ if(get_flags((src).flags,SCL_DO_RED)) f((src).red+(src).offset_x,(p),(len));      \
    if(get_flags((src).flags,SCL_DO_GREEN)) f((src).green+(src).offset_x,(p),(len));        \
    if(get_flags((src).flags,SCL_DO_BLUE)) f((src).blue+(src).offset_x,(p),(len));     \
	if(get_flags((src).flags,SCL_DO_ALPHA)) f((src).alpha+(src).offset_x,(p),(len));\
  }while(0)

#define SCANLINE_COMBINE_slow(f,c1,c2,c3,c4,o1,o2,p,len)						   \
do{	f((c1).red,(c2).red,(c3).red,(c4).red,(o1).red,(o2).red,(p),(len));		\
	f((c1).green,(c2).green,(c3).green,(c4).green,(o1).green,(o2).green,(p),(len));	\
	f((c1).blue,(c2).blue,(c3).blue,(c4).blue,(o1).blue,(o2).blue,(p),(len));		\
	if(get_flags((c1).flags,SCL_DO_ALPHA)) f((c1).alpha,(c2).alpha,(c3).alpha,(c4).alpha,(o1).alpha,(o2).alpha,(p),(len));	\
  }while(0)

#define SCANLINE_COMBINE(f,c1,c2,c3,c4,o1,o2,p,len)						   \
do{	f((c1).red,(c2).red,(c3).red,(c4).red,(o1).red,(o2).red,(p),(len+(len&0x01))*3);		\
	if(get_flags((c1).flags,SCL_DO_ALPHA)) f((c1).alpha,(c2).alpha,(c3).alpha,(c4).alpha,(o1).alpha,(o2).alpha,(p),(len));	\
  }while(0)


/* note that we shift values by 8 to keep quanitzation error in   */
/* lower 1 byte for subsequent dithering 	:					  */
#define QUANT_ERR_BITS  	8
#define QUANT_ERR_MASK  	0x000000FF

void copy_component( register CARD32 *src, register CARD32 *dst, int *unused, int len );

#ifdef X_DISPLAY_MISSING
typedef struct XRectangle
{
	short x, y;
	unsigned short width, height ;
}XRectangle ;
#endif

/****f* libAfterImage/asimage/get_asimage_channel_rects()
 * NAME
 * get_asimage_channel_rects() - translate image into a 
 * list of rectangles.
 * SYNOPSIS
 * XRectangle* 
 *     get_asimage_channel_rects( ASImage *src, int channel, 
 *                                unsigned int threshold, 
 *                                unsigned int *rects_count_ret ); 
 * INPUTS
 * src         - ASImage which will donate its channel to dst;
 * channel     - what source image channel to copy data from;
 * threshold   - threshold to compare channel values against;
 * rects_count_ret - returns count of generated rectangles.
 * DESCRIPTION
 * This function will translate contents of selected channel 
 * (usualy alpha) into a list of rectangles, ecompasing regions 
 * with values above the threshold. This is usefull to generate shape
 * of the window to be used with X Shape extention.
 *********/
XRectangle*
get_asimage_channel_rects( ASImage *src, int channel, unsigned int threshold, unsigned int *rects_count_ret );

void
raw2scanline( register CARD8 *row, struct ASScanline *buf, CARD8 *gamma_table, unsigned int width, Bool grayscale, Bool do_alpha );

#ifdef __cplusplus
}
#endif


#endif
