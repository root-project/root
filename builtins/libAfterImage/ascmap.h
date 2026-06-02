#ifndef ASCMAP_H_HEADER_ICLUDED
#define ASCMAP_H_HEADER_ICLUDED

#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/ascmap.h
 * NAME
 * ascmap - Defines main structures and function for image quantization.
 * DESCRIPTION
 * Image quantization is needed primarily in order to be able to export
 * images into file, with colormap format, such as GIF and XPM.
 * libAfterImage attempts to allocate colorcells to the most used colors,
 * and then approximate remaining colors with the closest colorcell.
 *
 * Since quality of quantization is in reverse proportion to the number
 * of colors in original image, libAfterImage allows to set arbitrary
 * level of downsampling of the color spectrum in the range of 8 bit per
 * channel to 1 bit per channel. Downsampling is performed by simple
 * dropping of less significant bits off of color values.
 *
 * In order to be able to determine closeness of colors, 3-channel RGB
 * values are converted into flat 24bit (or less if downsampling is used)
 * index. That is done by intermixing bits from different channels, like
 * so : R8G8B8R7G7B7...R1G1B1. That flat index is used to arrange colors
 * in ascending order, and later on to be able to find closest mapped
 * color. Simple hashing technique is used to speed up the
 * sorting/searching, as it allows to limit linked lists traversals.
 *
 * SEE ALSO
 * Structures :
 *          ASColormapEntry
 *          ASColormap
 *
 * Functions :
 *          colormap_asimage(), destroy_colormap()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 *******/

/***********************************************************************************/
/* reduced colormap building code :                                                */
/***********************************************************************************/
typedef struct ASMappedColor
{
	CARD8  alpha, red, green, blue;
	CARD32 indexed;
	unsigned int count ;
	int cmap_idx ;
	struct ASMappedColor *next ;
}ASMappedColor;

typedef struct ASSortedColorBucket
{
	unsigned int count ;
	ASMappedColor *head, *tail ;			/* pointers to first and last
											 * mapped colors in the stack */
	int good_offset ;                       /* skip to closest stack that
											 * has mapped colors */
}ASSortedColorBucket;

#define MAKE_INDEXED_COLOR3(red,green,blue) \
                   ((((green&0x200)|(blue&0x100)|(red&0x80))<<14))

#define MAKE_INDEXED_COLOR6(red,green,blue) \
				   (MAKE_INDEXED_COLOR3(red,green,blue)| \
		            (((green&0x100)|(blue&0x80) |(red&0x40))<<12))

#define MAKE_INDEXED_COLOR9(red,green,blue) \
                   (MAKE_INDEXED_COLOR6(red,green,blue)| \
		            (((green&0x80) |(blue&0x40) |(red&0x20))<<10))

#define MAKE_INDEXED_COLOR12(red,green,blue) \
                   (MAKE_INDEXED_COLOR9(red,green,blue)| \
					(((green&0x40) |(blue&0x20) |(red&0x10))<<8 ))

#define MAKE_INDEXED_COLOR15(red,green,blue) \
                   (MAKE_INDEXED_COLOR12(red,green,blue)| \
					(((green&0x20) |(blue&0x10) |(red&0x08))<<6 ))

#define MAKE_INDEXED_COLOR18(red,green,blue) \
                   (MAKE_INDEXED_COLOR15(red,green,blue)| \
					(((green&0x10) |(blue&0x08) |(red&0x04))<<4 ))

#define MAKE_INDEXED_COLOR21(red,green,blue) \
                   (MAKE_INDEXED_COLOR18(red,green,blue)| \
					(((green&0x08) |(blue&0x04) |(red&0x02))<<2 ))

#define MAKE_INDEXED_COLOR24(red,green,blue) \
                   (MAKE_INDEXED_COLOR21(red,green,blue)| \
					 ((green&0x04) |(blue&0x02) |(red&0x01)))

#define INDEX_SHIFT_RED(r)    (r)
#define INDEX_SHIFT_GREEN(g) ((g)<<2)
#define INDEX_SHIFT_BLUE(b) ((b)<<1)

#define INDEX_UNSHIFT_RED(r)    (r)
#define INDEX_UNSHIFT_GREEN(g)  ((g)>>2)
#define INDEX_UNSHIFT_BLUE(b)   ((b)>>1)

#define INDEX_UNESHIFT_RED(r,e)   ((r)>>(e))
#define INDEX_UNESHIFT_GREEN(g,e) ((g)>>(2+(e)))
#define INDEX_UNESHIFT_BLUE(b,e)  ((b)>>(1+(e)))


#define SLOTS_OFFSET24 15
#define SLOTS_MASK24   0x1FF
#define SLOTS_OFFSET21 12
#define SLOTS_MASK21   0x1FF

#define MAKE_INDEXED_COLOR MAKE_INDEXED_COLOR21
#define SLOTS_OFFSET	9
#define SLOTS_MASK		0xFFF
#define MAX_COLOR_BUCKETS		  4096


typedef struct ASSortedColorHash
{
	unsigned int count_unique ;
	ASSortedColorBucket *buckets ;
	int buckets_num ;
	CARD32  last_found ;
	int     last_idx ;
}ASSortedColorHash;

/****s* libAfterImage/ASColormapEntry
 * NAME
 * ASColormapEntry - ASColormapEntry represents single colorcell in the colormap.
 * SOURCE
 */

typedef struct ASColormapEntry
{
	CARD8 red, green, blue;
}ASColormapEntry;
/*******/
/****s* libAfterImage/ASColormap
 * NAME
 * ASColormap - ASColormap represents entire colormap generated for the image.
 * SOURCE
 */
typedef struct ASColormap
{
	ASColormapEntry *entries ;  /* array of colorcells */
	unsigned int count ;        /* number of used colorcells */
	ASSortedColorHash *hash ;   /* internal data */
	Bool has_opaque ;           /* If True then Image has opaque pixels */
}ASColormap;
/*******/

void         add_index_color   ( ASSortedColorHash *index,
	                             CARD32 indexed, unsigned int slot,
							     CARD32 red, CARD32 green, CARD32 blue );
void         destroy_colorhash ( ASSortedColorHash *index, Bool reusable );
unsigned int add_colormap_items( ASSortedColorHash *index,
	                             unsigned int start, unsigned int stop,
								 unsigned int quota, unsigned int base,
								 ASColormapEntry *entries );

void        fix_colorindex_shortcuts( ASSortedColorHash *index );
int         get_color_index         ( ASSortedColorHash *index,
	                                  CARD32 indexed, unsigned int slot );
ASColormap *color_hash2colormap     ( ASColormap *cmap,
	                                  unsigned int max_colors );

/****f* libAfterImage/colormap_asimage()
 * NAME
 * colormap_asimage()
 * SYNOPSIS
 * int *colormap_asimage( ASImage *im, ASColormap *cmap,
 *                        unsigned int max_colors, unsigned int dither,
 *                        int opaque_threshold );
 * INPUTS
 * im				- pointer to valid ASImage structure.
 * cmap             - preallocated structure to store colormap in.
 * max_colors       - maximum size of the colormap.
 * dither           - number of bits to strip off the color data ( 0...7 )
 * opaque_threshold - alpha channel threshold at which pixel should be
 *                    treated as opaque
 * RETURN VALUE
 * pointer to the array of indexes representing pixel's colorcells. This
 * array has size of WIDTHxHEIGHT where WIDTH and HEIGHT are size of the
 * source image.
 * DESCRIPTION
 * This function is all that is needed to quantize the ASImage. In order
 * to obtain colorcell of the pixel at (x,y) from result, the following
 * code could be used :
 * cmap->entries[res[y*width+x]]
 * where res is returned pointer.
 * Recommended value for dither parameter is 4 while quantizing photos to
 * 256 colors, and it could be less , if original has limited number of
 * colors.
 *
 *********/
/****f* libAfterImage/destroy_colormap()
 * NAME
 * destroy_colormap()
 * SYNOPSIS
 * void destroy_colormap( ASColormap *cmap, Bool reusable );
 * INPUTS
 * cmap				- pointer to valid ASColormap structure.
 * reusable         - if True, then the memory pointed to by cmap will
 *                    not be deallocated, as if it was allocated on stack
 * DESCRIPTION
 * Destroys ASColormap object created using colormap_asimage.
 *********/
int *colormap_asimage( ASImage *im, ASColormap *cmap,
	                   unsigned int max_colors, unsigned int dither,
					   int opaque_threshold );
void destroy_colormap( ASColormap *cmap, Bool reusable );

#ifdef __cplusplus
}
#endif

#endif
