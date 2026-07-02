#ifndef XCF_H_INCLUDED
#define XCF_H_INCLUDED

/* GIMP's XCF file properties/structures : */

#include "asvisual.h"
#include "scanline.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XCF_MAX_CHANNELS     4

#define XCF_GRAY_PIX         0
#define XCF_ALPHA_G_PIX      1
#define XCF_RED_PIX          0
#define XCF_GREEN_PIX        1
#define XCF_BLUE_PIX         2
#define XCF_ALPHA_PIX        3
#define XCF_INDEXED_PIX      0
#define XCF_ALPHA_I_PIX      1

#define XCF_COLORMAP_SIZE    768

typedef enum
{
  XCF_PROP_END = 0,
  XCF_PROP_COLORMAP = 1,
  XCF_PROP_ACTIVE_LAYER = 2,
  XCF_PROP_ACTIVE_CHANNEL = 3,
  XCF_PROP_SELECTION = 4,
  XCF_PROP_FLOATING_SELECTION = 5,
  XCF_PROP_OPACITY = 6,
  XCF_PROP_MODE = 7,
  XCF_PROP_VISIBLE = 8,
  XCF_PROP_LINKED = 9,
  XCF_PROP_PRESERVE_TRANSPARENCY = 10,
  XCF_PROP_APPLY_MASK = 11,
  XCF_PROP_EDIT_MASK = 12,
  XCF_PROP_SHOW_MASK = 13,
  XCF_PROP_SHOW_MASKED = 14,
  XCF_PROP_OFFSETS = 15,
  XCF_PROP_COLOR = 16,
  XCF_PROP_COMPRESSION = 17,
  XCF_PROP_GUIDES = 18,
  XCF_PROP_RESOLUTION = 19,
  XCF_PROP_TATTOO = 20,
  XCF_PROP_PARASITES = 21,
  XCF_PROP_UNIT = 22,
  XCF_PROP_PATHS = 23,
  XCF_PROP_USER_UNIT = 24,
  XCF_PROP_Total = 25
} XcfPropType;

typedef enum
{
  XCF_COMPRESS_NONE = 0,
  XCF_COMPRESS_RLE = 1,
  XCF_COMPRESS_ZLIB = 2,
  XCF_COMPRESS_FRACTAL = 3  /* Unused. */
} XcfCompressionType;

typedef enum
{
  XCF_RED_CHANNEL,
  XCF_GREEN_CHANNEL,
  XCF_BLUE_CHANNEL,
  XCF_GRAY_CHANNEL,
  XCF_INDEXED_CHANNEL,
  XCF_ALPHA_CHANNEL,
  XCF_AUXILLARY_CHANNEL
} XcfChannelType;

typedef enum
{
  XCF_EXPAND_AS_NECESSARY,
  XCF_CLIP_TO_IMAGE,
  XCF_CLIP_TO_BOTTOM_LAYER,
  XCF_FLATTEN_IMAGE
} XcfMergeType;

#define XCF_SIGNATURE      		"gimp xcf"
#define XCF_SIGNATURE_LEN  		8              /* use in strncmp() */
#define XCF_SIGNATURE_FULL 		"gimp xcf file"
#define XCF_SIGNATURE_FULL_LEN 	14             /* use in seek() */

#define XCF_TILE_WIDTH			64
#define XCF_TILE_HEIGHT			64

struct XcfProperty;
struct XcfLayer;
struct XcfChannel;
struct XcfHierarchy;
struct XcfLevel;
struct XcfTile;


typedef struct XcfImage
{
	int 		version;
	CARD32 		width;
	CARD32 		height;
	CARD32 		type ;

	CARD8 		compression ;
	CARD32      num_cols ;
	CARD8      *colormap ;

	struct XcfProperty   *properties ;
	struct XcfLayer		 *layers;
	struct XcfChannel	 *channels;

	struct XcfLayer		 *floating_selection;
	struct XcfChannel	 *selection;

	ASScanline 			  scanline_buf[XCF_TILE_HEIGHT];
	CARD8 				  tile_buf[XCF_TILE_WIDTH*XCF_TILE_HEIGHT*6];
}XcfImage;

typedef struct XcfProperty
{
	CARD32 	   		  	  id ;
	CARD32				  len;
	CARD8	 		     *data;
/* most properties will fit in here - save on memory allocation */
	CARD32				  buffer[20] ;

	struct XcfProperty   *next;
}XcfProperty;

typedef struct XcfLayer
{
	struct XcfLayer 	 *next;
	CARD32 	  		      offset ;
	/* layer data goes here */
	CARD32	    		  width ;
	CARD32	    		  height ;
	CARD32	    		  type ;
	/* we don't give a damn about layer's name - skip it */
	struct XcfProperty   *properties ;
	CARD32 				  opacity ;
	Bool 				  visible ;
	Bool				  preserve_transparency ;
	CARD32				  mode ;
	CARD32				  offset_x, offset_y ;

	CARD32				  hierarchy_offset;
	CARD32      		  mask_offset ;
	struct XcfHierarchy	 *hierarchy ;
	struct XcfChannel	 *mask ;

}XcfLayer;

typedef struct XcfChannel
{
	struct XcfChannel *next;
	CARD32 		offset ;
	/* Channel data goes here */
	CARD32	    width ;
	CARD32	    height ;
	/* we don't give a damn about layer's name - skip it */
	struct XcfProperty   *properties ;
	CARD32 				  opacity ;
	Bool 				  visible ;
	ARGB32				  color ;

	CARD32 		hierarchy_offset;
	struct XcfHierarchy	 *hierarchy ;

}XcfChannel;

typedef struct XcfHierarchy
{
	/* layer data goes here */
	CARD32	    width ;
	CARD32	    height ;
	CARD32		bpp ;

	/* we don't give a damn about layer's name - skip it */
	struct XcfLevel	 	 *levels ;

	ASImage 			 *image ;
}XcfHierarchy;

typedef struct XcfLevel
{
	struct XcfLevel *next ;
	CARD32 		offset ;
	CARD32	    width ;
	CARD32	    height ;

	struct XcfTile *tiles ;
}XcfLevel;

typedef struct XcfTile
{
	struct XcfTile *next ;
	CARD32 		offset ;
	CARD32	    estimated_size ;

	CARD8	   *data;
}XcfTile;

union XcfListElem;

typedef struct XcfAnyListElem
{
	union XcfListElem *next;
	CARD32 offset ;
}XcfAnyListElem;

typedef union XcfListElem{
	XcfAnyListElem  any;
	XcfLayer		layer;
	XcfChannel		channel;
	XcfLevel		level;
	XcfTile			tile;
}XcfListElem ;


XcfImage   *read_xcf_image( FILE *fp );
void 		print_xcf_image( XcfImage *xcf_im );
void		free_xcf_image( XcfImage *xcf_im );

#ifdef __cplusplus
}
#endif

#endif /* #ifndef XCF_H_INCLUDED */
