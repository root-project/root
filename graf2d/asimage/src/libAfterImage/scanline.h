#ifndef _SCANLINE_H_HEADER_INCLUDED
#define _SCANLINE_H_HEADER_INCLUDED

#include "asvisual.h"

#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/scanline.h
 * NAME
 * scanline - Structures and functions for manipulation of image data 
 * in blocks of uncompressed scanlines. Each scanline has 4 32 bit channels.
 * Data in scanline could be both 8bit or 16 bit, with automated 
 * dithering of 16 bit data into standard 8-bit image.
 * SEE ALSO
 * Structures:
 *  	    ASScanline
 *
 * Functions :
 *   ASScanline handling:
 *  	    prepare_scanline(), free_scanline()
 *
 * Other libAfterImage modules :
 *          asvisual.h imencdec.h asimage.h blender.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******************/

/****s* libAfterImage/ASScanline
 * NAME
 * ASScanline - structure to hold contents of the single scanline.
 * DESCRIPTION
 * ASScanline holds data for the single scanline, split into channels
 * with 32 bits per pixel per channel. All the memory is allocated at
 * once, and then split in between channels. There are three ways to
 * access channel data :
 * 1) using blue, green, red, alpha pointers.
 * 2) using channels[] array of pointers - convenient in loops
 * 4) using xc3, xc2, xc1 pointers. These are different from red, green,
 * blue in the way that xc3 will point to blue when BGR mode is specified
 * at the time of creation, otherwise it will point to red channel.
 * Likewise xc1 will point to red in BGR mode and blue otherwise.
 * xc2 always points to green channel's data. This is convenient while
 * writing XImages and when channels in source and destination has to be
 * reversed, while reading images from files.
 * Channel data is always aligned by 8 byte boundary allowing for
 * utilization of MMX, floating point and other 64bit registers for
 * transfer and processing.
 * SEE ALSO
 * ASImage
 * SOURCE
 */
typedef struct ASScanline
{
#define SCL_DO_BLUE         (0x01<<ARGB32_BLUE_CHAN )
#define SCL_DO_GREEN        (0x01<<ARGB32_GREEN_CHAN)
#define SCL_DO_RED          (0x01<<ARGB32_RED_CHAN  )
#define SCL_DO_ALPHA		(0x01<<ARGB32_ALPHA_CHAN)
#define SCL_DO_COLOR		(SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE)
#define SCL_DO_ALL			(SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE| \
                             SCL_DO_ALPHA)
#define SCL_RESERVED_MASK	0x0000FFFF
#define SCL_CUSTOM_MASK		0xFFFF0000
#define SCL_CUSTOM_OFFSET	16
							 
	CARD32	 	   flags ;   /* combination of  the above values */
	CARD32        *buffer ;
	CARD32        *blue, *green, *red, *alpha ;
	CARD32	      *channels[IC_NUM_CHANNELS];
	CARD32        *xc3, *xc2, *xc1; /* since some servers require
									 * BGR mode here we store what
									 * goes into what color component
									 * in XImage */
	ARGB32         back_color;
	unsigned int   width, shift;
	unsigned int   offset_x ;
}ASScanline;
/*******************/

#define ASIM_DEMOSAIC_DEFAULT_STRIP_SIZE 	5

typedef struct ASIMStrip
{
#define ASIM_SCL_InterpolatedH 		(0x01<<SCL_CUSTOM_OFFSET)
#define ASIM_SCL_InterpolatedV 		(0x01<<(SCL_CUSTOM_OFFSET+ARGB32_CHANNELS))
#define ASIM_SCL_InterpolatedAll 	(ASIM_SCL_InterpolatedV|ASIM_SCL_InterpolatedH)

#define ASIM_SCL_RGDiffCalculated 	(0x01<<(SCL_CUSTOM_OFFSET+ARGB32_CHANNELS*2))
#define ASIM_SCL_BGDiffCalculated 	(0x01<<(SCL_CUSTOM_OFFSET+ARGB32_CHANNELS*2+1))

	int 		 size, width;
	ASScanline 	**lines;
	int 		 start_line;
	
	void 	   **aux_data;

#define ASIM_SCL_MissingValue	0xF0000000
#define ASIM_IsMissingValue(v)		((v)&0xF0000000)

#define ASIM_IsStripLineLoaded(sptr,l)   		((sptr)->lines[l]->flags & SCL_DO_COLOR)
#define ASIM_IsStripLineInterpolated(sptr,l)   	((sptr)->lines[l]->flags & ASIM_SCL_Interpolated)
}ASIMStrip;

typedef void (*ASIMStripLoader)(ASScanline *scl, CARD8 *data, int data_size);



/****f* libAfterImage/prepare_scanline()
 * NAME
 * prepare_scanline()
 * SYNOPSIS
 * ASScanline *prepare_scanline ( unsigned int width,
 *                                unsigned int shift,
 *                                ASScanline *reusable_memory,
 *                                Bool BGR_mode);
 * INPUTS
 * width           - width of the scanline.
 * shift           - format of contained data. 0 means - 32bit unshifted
 *                   8 means - 24.8bit ( 8 bit left shifted ).
 * reusable_memory - preallocated object.
 * BGR_mode        - if True will cause xc3 to point to Blue and xc1 to
 *                   point to red.
 * DESCRIPTION
 * This function allocates memory ( if reusable_memory is NULL ) for
 * the new ASScanline structure. Structures buffers gets allocated to
 * hold scanline data of at least width pixel wide. Buffers are adjusted
 * to start on 8 byte boundary.
 *********/
/****f* libAfterImage/free_scanline()
 * NAME
 * free_scanline()
 * SYNOPSIS
 * void       free_scanline ( ASScanline *sl, Bool reusable );
 * INPUTS
 * sl       - pointer to previously allocated ASScanline structure to be
 *            deallocated.
 * reusable - if true then ASScanline object itself will not be
 *            deallocated.
 * DESCRIPTION
 * free_scanline() frees all the buffer memory allocated for ASScanline.
 * If reusable is false then object itself in not freed. That is usable
 * for declaring ASScanline on stack.
 *********/
ASScanline* prepare_scanline( unsigned int width, unsigned int shift,
	                          ASScanline *reusable_memory, Bool BGR_mode);
void       free_scanline( ASScanline *sl, Bool reusable );

/* Scanline strips */
void destroy_asim_strip (ASIMStrip **pstrip);
ASIMStrip *create_asim_strip(unsigned int size, unsigned int width, int shift, int bgr);
void advance_asim_strip (ASIMStrip *strip);

/* demosaicing */
/* returns number of lines processed from the data */
int load_asim_strip (ASIMStrip *strip, CARD8 *data, int data_size, int data_start_line, int data_row_size, 
					 ASIMStripLoader *line_loaders, int line_loaders_num);
void decode_BG_12_be (ASScanline *scl, CARD8 *data, int data_size);
void decode_GR_12_be (ASScanline *scl, CARD8 *data, int data_size);
void decode_RG_12_be (ASScanline *scl, CARD8 *data, int data_size);
void decode_GB_12_be (ASScanline *scl, CARD8 *data, int data_size);

void interpolate_asim_strip_custom_rggb2 (ASIMStrip *strip, ASFlagType filter, Bool force_all);

#ifdef __cplusplus
}
#endif

#endif /* _SCANLINE_H_HEADER_INCLUDED */
