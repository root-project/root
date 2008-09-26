#ifndef _ASVISUAL_H_HEADER_INCLUDED
#define _ASVISUAL_H_HEADER_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/asvisual.h
 * NAME
 * asvisual - Defines abstraction layer on top of X Visuals, as well as 
 * several fundamental color datatypes.
 * SEE ALSO
 * Structures:
 *  	    ColorPair
 *  	    ASVisual
 *
 * Functions :
 *   ASVisual initialization :
 *  	    query_screen_visual(), setup_truecolor_visual(),
 *  	    setup_pseudo_visual(), setup_as_colormap(),create_asvisual(),
 *  	    destroy_asvisual()
 *
 *   ASVisual encoding/decoding :
 *  	    visual2visual_prop(), visual_prop2visual()
 *
 *   ASVisual convenience functions :
 *  	    create_visual_window(), create_visual_pixmap(),
 *  	    create_visual_ximage()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******************/

/****d* libAfterImage/alpha
 * FUNCTION
 * Alpha channel adds visibility parameter to color value.
 * Alpha channel's value of 0xFF signifies complete visibility, while 0
 * makes pixel completely transparent.
 * SOURCE
 */
#define ALPHA_TRANSPARENT      	0x00
#define ALPHA_SEMI_TRANSPARENT 	0x7F
#define ALPHA_SOLID            	0xFF
/*******************/
/****d* libAfterImage/ARGB32
 * NAME
 * ARGB32 - main color datatype
 * FUNCTION
 * ARGB32 is fundamental datatype that hold 32bit value corresponding to
 * pixels color and transparency value (alpha channel) in ARGB
 * colorspace. It is encoded as follows :
 * Lowermost 8 bits - Blue channel
 * bits 8 to 15     - Green channel
 * bits 16 to 23    - Red channel
 * bits 24 to 31    - Alpha channel
 * EXAMPLE
 * ASTile.1
 * SOURCE
 */
typedef CARD32 ARGB32;
#define ARGB32_White    		0xFFFFFFFF
#define ARGB32_Black    		0xFF000000
/* default background color is #FF000000 : */
#define ARGB32_DEFAULT_BACK_COLOR	ARGB32_Black

#define ARGB32_ALPHA_CHAN		3
#define ARGB32_RED_CHAN			2
#define ARGB32_GREEN_CHAN		1
#define ARGB32_BLUE_CHAN		0
#define ARGB32_CHANNELS			4

#define MAKE_ARGB32(a,r,g,b)	((( (CARD32)a)        <<24)| \
								 ((((CARD32)r)&0x00FF)<<16)| \
                                 ((((CARD32)g)&0x00FF)<<8 )| \
								 (( (CARD32)b)&0x00FF))

#define MAKE_ARGB32_GREY8(a,l)	(((a)<<24)|(((l)&0x00FF)<<16)| \
                                 (((l)&0x00FF)<<8)|((l)&0x00FF))
#define ARGB32_ALPHA8(c)		(((c)>>24)&0x00FF)
#define ARGB32_RED8(c)			(((c)>>16)&0x00FF)
#define ARGB32_GREEN8(c)	 	(((c)>>8 )&0x00FF)
#define ARGB32_BLUE8(c)			( (c)     &0x00FF)
#define ARGB32_CHAN8(c,i)		(((c)>>((i)<<3))&0x00FF)
#define MAKE_ARGB32_CHAN8(v,i)	(((v)&0x0000FF)<<((i)<<3))

#ifdef __GNUC__
#define ARGB32_ALPHA16(c)		({ CARD32 __c = ARGB32_ALPHA8(c); __c | (__c<<8);})
#define ARGB32_RED16(c)			({ CARD32 __c = ARGB32_RED8(c); __c | (__c<<8);})
#define ARGB32_GREEN16(c)	 	({ CARD32 __c = ARGB32_GREEN8(c); __c | (__c<<8);})
#define ARGB32_BLUE16(c)		({ CARD32 __c = ARGB32_BLUE8(c); __c | (__c<<8);})
#define ARGB32_CHAN16(c,i)		({ CARD32 __c = ARGB32_CHAN8(c,i); __c | (__c<<8);})
#else
#define ARGB32_ALPHA16(c)		((((c)>>16)&0x00FF00)|(((c)>>24)&0x0000FF))
#define ARGB32_RED16(c)			((((c)>>8 )&0x00FF00)|(((c)>>16)&0x0000FF))
#define ARGB32_GREEN16(c)	 	(( (c)     &0x00FF00)|(((c)>>8 )&0x0000FF))
#define ARGB32_BLUE16(c)		((((c)<<8) &0x00FF00)|(((c)    )&0x0000FF))
#define ARGB32_CHAN16(c,i)		((ARGB32_CHAN8(c,i)<<8)|ARGB32_CHAN8(c,i))
#endif

#define MAKE_ARGB32_CHAN16(v,i)	((((v)&0x00FF00)>>8)<<((i)<<3))
/*******************/


struct ASScanline;

/****d* libAfterImage/ColorPart
 * NAME
 * IC_RED - red channel
 * NAME
 * IC_GREEN - green channel
 * NAME 
 * IC_BLUE - blue channel
 * NAME
 * IC_ALPHA - alpha channel
 * NAME
 * IC_NUM_CHANNELS - number of supported channels
 * FUNCTION
 * Ids of the channels. These are basically synonyms to related ARGB32
 * channel numbers
 * SOURCE
 */
typedef enum
{
  IC_BLUE	= ARGB32_BLUE_CHAN ,
  IC_GREEN	= ARGB32_GREEN_CHAN,
  IC_RED 	= ARGB32_RED_CHAN  ,
  IC_ALPHA  = ARGB32_ALPHA_CHAN,
  IC_NUM_CHANNELS = ARGB32_CHANNELS
}
ColorPart;
/*******************/
/****s* libAfterImage/ColorPair
 * NAME
 * ColorPair - convenient structure to hold pair of colors.
 * SOURCE
 */
typedef struct ColorPair
{
  ARGB32 fore;
  ARGB32 back;
}ColorPair;
/*******************/
/****f* libAfterImage/ARGB32_manhattan_distance()
 * NAME 
 * ARGB32_manhattan_distance() - This function can be used to evaluate closeness of 
 * two colors.
 * SYNOPSIS
 * long ARGB32_manhattan_distance (long a, long b);
 * INPUTS
 * a, b - ARGB32 color values to calculate Manhattan distance in between
 * RETURN VALUE
 * returns calculated Manhattan distance.
 *********/
long ARGB32_manhattan_distance (long a, long b);

/****s* libAfterImage/ASVisual
 * NAME
 * ASVisual - an abstraction layer on top of X Server Visual.
 * DESCRIPTION
 * This structure has been introduced in order to compensate for the
 * fact that X may have so many different types of Visuals. It provides
 * shortcuts to most Visual data, compensated for differences in Visuals.
 * For PseudoColor visual it also contains preallocated set of colors.
 * This colormap allows us to write XImages very fast and without
 * exhausting available X colors. This colormap consist of 8, 64, or 4096
 * colors and constitutes fraction of colors available in particular
 * colordepth. This colors are allocated to be evenly spread around RGB
 * spectrum. Thus when converting from internal presentation - all we
 * need to do is to discard unused bits, and use rest of them bits as
 * an index in our colormap. Opposite conversion is much trickier and we
 * engage into nasty business of having hash table mapping pixel values
 * into colors, or straight table doing same in lower colordepths.
 * Idea is that we do all internal processing in 32bit colordepth, and
 * ASVisual provides us with means to convert it to actual X display
 * format. Respectively ASVisual has methods to write out XImage lines
 * and read XImage lines.
 * ASVisual creation is a tricky process. Basically first we have to go
 * through the list of available Visuals and choose the best suitable.
 * Then based on the type of this Visual we have to setup our data
 * members and method hooks. Several functions provided for that :
 *  query_screen_visual()    - will lookup best suitable visual
 *  setup_truecolor_visual() - will setup hooks if visual is TrueColor
 *  setup_pseudo_visual()	 - will setup hooks and data if Visual is
 *                             PseudoColor.
 *  setup_as_colormap()      - will preallocate colors for PseudoColor.
 * Alternative to the above is :
 *  create_asvisual()        - it encapsulates all of the above
 *                             functionality, and returns completely set
 *                             up ASVisual object.
 * Since Visual selected for ASVisual may differ from default
 * ( we choose the best suitable ), all the window creation function
 * must provide colormap and some other parameters, like border color
 * for example. Thus we created some convenience functions.
 * These should be used instead of standard Xlib calls :
 *  create_visual_window() - to create window
 *  create_visual_pixmap() - to create pixmap
 *  create_visual_ximage() - to create XImage
 * ASVisual could be dealolocated and its resources freed with :
 *  destroy_asvisual()
 * EXAMPLE
 * asview.c: ASView
 * SOURCE
 */
typedef struct ASVisual
{
	Display      *dpy;

	/* This envvar will be used to determine what X Visual 
	 * (in hex) to use. If unset then best possible will 
	 * be selected automagically : */
#define ASVISUAL_ID_ENVVAR "AFTERIMAGE_VISUAL_ID"

	XVisualInfo	  visual_info;
	/* this things are calculated based on Visual : */
	unsigned long rshift, gshift, bshift;
	unsigned long rbits,  gbits,  bbits;
	unsigned long true_depth;	/* could be 15 when X reports 16 */
	Bool          BGR_mode;
	Bool 		  msb_first;
	/* we must have colormap so that we can safely create windows
	 * with different visuals even if we are in TrueColor mode : */
	Colormap 	  colormap;
	Bool          own_colormap; /* tells us to free colormap when we
								 * done */
	unsigned long black_pixel, white_pixel;
	/* for PseudoColor mode we need some more stuff : */
	enum {
		ACM_None = 0,
		ACM_3BPP,
		ACM_6BPP,
		ACM_12BPP
	} as_colormap_type ;    /* there can only be 64 or 4096 entries
							 * so far ( 6 or 12 bpp) */
	unsigned long *as_colormap; /* array of preallocated colors for
								 * PseudoColor mode */
	union                       /* reverse color lookup tables : */
	{
		ARGB32 		  		*xref;
		struct ASHashTable  *hash;
	}as_colormap_reverse ;

	/* different useful callbacks : */
	CARD32 (*color2pixel_func) 	  ( struct ASVisual *asv,
		                            CARD32 encoded_color,
									unsigned long *pixel);
	void   (*pixel2color_func)    ( struct ASVisual *asv,
		                            unsigned long pixel,
									CARD32 *red, CARD32 *green,
									CARD32 *blue);
	void   (*ximage2scanline_func)( struct ASVisual *asv, 
									XImage *xim,
		                            struct ASScanline *sl, int y,
								    unsigned char *xim_data );
	void   (*scanline2ximage_func)( struct ASVisual *asv, 
									XImage *xim,
									struct ASScanline *sl, int y,
									unsigned char *xim_data );

#define ASGLX_Unavailable			0
#define ASGLX_Available				(0x01<<0)
#define ASGLX_DoubleBuffer			(0x01<<1)
#define ASGLX_RGBA					(0x01<<2)
#define ASGLX_UseForImageTx			(0x01<<3)	
	ASFlagType glx_support ;    /* one of the above flags */

	void *glx_scratch_gc_indirect ; /* (GLXContext) */
	void *glx_scratch_gc_direct ;	/* (GLXContext) */

	Window scratch_window;

#ifndef X_DISPLAY_MISSING
#define ARGB2PIXEL(asv,argb,pixel) 		   \
	(asv)->color2pixel_func((asv),(argb),(pixel))
#define GET_SCANLINE(asv,xim,sl,y,xim_data) \
	(asv)->ximage2scanline_func((asv),(xim),(sl),(y),(xim_data))
#define PUT_SCANLINE(asv,xim,sl,y,xim_data) \
	(asv)->scanline2ximage_func((asv),(xim),(sl),(y),(xim_data))
#else
#define ARGB2PIXEL(asv,argb,pixel) 		   \
	do{ break; }while(0)
#define GET_SCANLINE(asv,xim,sl,y,xim_data) \
	do{ break; }while(0)
#define PUT_SCANLINE(asv,xim,sl,y,xim_data) \
	do{ break; }while(0)
#endif
}ASVisual;
/*******************/
CARD32 color2pixel32bgr(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel32rgb(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel24bgr(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel24rgb(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel16bgr(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel16rgb(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel15bgr(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel15rgb(ASVisual *asv, CARD32 encoded_color, unsigned long *pixel);
CARD32 color2pixel_pseudo3bpp( ASVisual *asv, CARD32 encoded_color, unsigned long *pixel );
CARD32 color2pixel_pseudo6bpp( ASVisual *asv, CARD32 encoded_color, unsigned long *pixel );
CARD32 color2pixel_pseudo12bpp( ASVisual *asv, CARD32 encoded_color, unsigned long *pixel );

void pixel2color32rgb(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color32bgr(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color24rgb(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color24bgr(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color16rgb(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color16bgr(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color15rgb(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);
void pixel2color15bgr(ASVisual *asv, unsigned long pixel, CARD32 *red, CARD32 *green, CARD32 *blue);

void ximage2scanline32( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void ximage2scanline16( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void ximage2scanline15( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void ximage2scanline_pseudo3bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void ximage2scanline_pseudo6bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void ximage2scanline_pseudo12bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );

void scanline2ximage32( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void scanline2ximage16( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void scanline2ximage15( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void scanline2ximage_pseudo3bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void scanline2ximage_pseudo6bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );
void scanline2ximage_pseudo12bpp( ASVisual *asv, XImage *xim, struct ASScanline *sl, int y,  register unsigned char *xim_data );

/****f* libAfterImage/query_screen_visual()
 * NAME
 * query_screen_visual_id()
 * NAME
 * query_screen_visual()
 * SYNOPSIS
 * Bool query_screen_visual_id( ASVisual *asv, Display *dpy, int screen,
 *                           Window root, int default_depth,
 *							 VisualID visual_id, Colormap cmap );
 * Bool query_screen_visual( ASVisual *asv, Display *dpy, int screen,
 *                           Window root, int default_depth );
 * INPUTS
 * asv  		- preallocated ASVisual structure.
 * dpy  		- valid pointer to opened X display.
 * screen   	- screen number on which to query visuals.
 * root     	- root window on that screen.
 * default_depth- default colordepth of the screen.
 * visual_id    - optional ID of prefered Visual.
 * cmap         - optional colormap to be used.
 * RETURN VALUE
 * True on success, False on failure
 * ASVisual structure pointed by asv will have the following data
 * members set on success :
 * dpy, visual_info, colormap, own_colormap, black_pixel, white_pixel.
 * DESCRIPTION
 * query_screen_visual_id() will go though prioritized list of possible
 * Visuals and attempt to match those to what is available on the
 * specified screen. If all items from list fail, then it goes about
 * querying default visual.
 * query_screen_visual is identical to query_screen_visual_id with
 * visual_id and cmap set to 0.
 * Once X Visual has been identified, we create X colormap and allocate
 * white and black pixels from it.
 *********/
/****f* libAfterImage/setup_truecolor_visual()
 * NAME
 * setup_truecolor_visual()
 * SYNOPSIS
 * Bool setup_truecolor_visual( ASVisual *asv );
 * INPUTS
 * asv  		- preallocated ASVisual structure.
 * RETURN VALUE
 * True on success, False if visual is not TrueColor.
 * DESCRIPTION
 * setup_truecolor_visual() checks if Visual is indeed TrueColor and if
 * so it goes about querying color masks, deducing real XImage
 * colordepth, and whether we work in BGR mode. It then goes about
 * setting up correct hooks to X IO functions.
 *********/
/****f* libAfterImage/setup_pseudo_visual()
 * NAME
 * setup_pseudo_visual()
 * SYNOPSIS
 * void setup_pseudo_visual( ASVisual *asv  );
 * INPUTS
 * asv  		- preallocated ASVisual structure.
 * DESCRIPTION
 * setup_pseudo_visual() assumes that Visual is PseudoColor. It then
 * tries to decide as to how many colors preallocate, and goes about
 * setting up correct X IO hooks and possibly initialization of reverse
 * colormap in case ASVisual already has colormap preallocated.
 *********/
/****f* libAfterImage/setup_as_colormap()
 * NAME
 * setup_as_colormap()
 * SYNOPSIS
 * void setup_as_colormap( ASVisual *asv );
 * INPUTS
 * asv  		- preallocated ASVisual structure.
 * DESCRIPTION
 * That has to be called in order to pre-allocate sufficient number of
 * colors. It uses colormap size identification supplied in ASVisual
 * structure. If colors where preallocated successfully - it will also
 * create reverse lookup colormap.
 *********/

Bool query_screen_visual_id( ASVisual *asv, Display *dpy, int screen,
	                      	 Window root, int default_depth,
							 VisualID visual_id, Colormap cmap );
#define query_screen_visual(a,d,s,r,dd) query_screen_visual_id((a),(d),(s),(r),(dd),0,0)

Bool setup_truecolor_visual( ASVisual *asv );
void setup_pseudo_visual( ASVisual *asv  );
void setup_as_colormap( ASVisual *asv );
/****f* libAfterImage/create_asvisual_for_id()
 * NAME
 * create_asvisual_for_id()
 * SYNOPSIS
 * ASVisual *create_asvisual_for_id( Display *dpy, int screen,
 *                                   int default_depth,
 *                                   VisualID visual_id, Colormap cmap,
 *                                   ASVisual *reusable_memory );
 * INPUTS
 * dpy  		- valid pointer to opened X display.
 * screen   	- screen number on which to query visuals.
 * root     	- root window on that screen.
 * default_depth- default colordepth of the screen.
 * visual_id    - ID of X visual to use.
 * cmap         - optional ID of the colormap to be used.
 * reusable_memory - pointer to preallocated ASVisual structure.
 * RETURN VALUE
 * Pointer to ASVisual structure initialized with enough information
 * to be able to deal with current X Visual.
 * DESCRIPTION
 * This function calls all the needed functions in order to setup new
 * ASVisual structure for the specified screen and visual. If
 * reusable_memory is not null - it will not allocate new ASVisual
 * structure, but instead will use supplied one. Useful for allocating
 * ASVisual on stack.
 * This particular function will not do any autodetection and will use
 * Visual ID supplied. That is usefull when libAfterImage is used with
 * an app that has its own approach to Visual handling, and since Visuals
 * on all Windows, Pixmaps and colormaps must match, there is a need to
 * synchronise visuals used by an app and libAfterImage.
 *********/
/****f* libAfterImage/create_asvisual()
 * NAME
 * create_asvisual()
 * SYNOPSIS
 * ASVisual *create_asvisual( Display *dpy, int screen,
 *                            int default_depth,
 *                            ASVisual *reusable_memory );
 * INPUTS
 * dpy  		- valid pointer to opened X display.
 * screen   	- screen number on which to query visuals.
 * root     	- root window on that screen.
 * default_depth- default colordepth of the screen.
 * reusable_memory - pointer to preallocated ASVisual structure.
 * RETURN VALUE
 * Pointer to ASVisual structure initialized with enough information
 * to be able to deal with current X Visual.
 * DESCRIPTION
 * This function calls all the needed functions in order to setup new
 * ASVisual structure for the specified screen. If reusable_memory is
 * not null - it will not allocate new ASVisual structure, but instead
 * will use supplied one. Useful for allocating ASVisual on stack.
 * It is different from create_asvisualfor_id() in that it will attempt
 * to autodetect best possible visual for the screen. For example on some
 * SUN Solaris X servers there will be both 8bpp pseudocolor and 24bpp
 * truecolor, and default will be 8bpp. In this scenario libAfterImage
 * will detect and use 24bpp true color visual, thus producing much better
 * results.
 *********/

/****f* libAfterImage/destroy_asvisual()
 * NAME
 * destroy_asvisual()
 * SYNOPSIS
 * void destroy_asvisual( ASVisual *asv, Bool reusable );
 * INPUTS
 * asv      - valid ASVisual structure.
 * reusable - if True it will cause function to not free object
 *            itself.
 * DESCRIPTION
 * Cleanup function. Frees all the memory and deallocates all the
 * resources. If reusable is False it will also free the object, pointed
 * to by asv.
 * EXAMPLE
 * asview.c: ASView.2
 *********/
ASVisual *create_asvisual_for_id( Display *dpy, int screen, int default_depth,
	                              VisualID visual_id, Colormap cmap,
								  ASVisual *reusable_memory );
ASVisual *create_asvisual( Display *dpy, int screen, int default_depth,
	                       ASVisual *reusable_memory );
ASVisual *get_default_asvisual();
void destroy_asvisual( ASVisual *asv, Bool reusable );
/****f* libAfterImage/visual2visual_prop()
 * NAME
 * visual2visual_prop()
 * SYNOPSIS
 * Bool visual2visual_prop( ASVisual *asv, size_t *size,
 *                          unsigned long *version, unsigned long **data );
 * INPUTS
 * asv      	- valid ASVisual structure.
 * RETURN VALUE
 * size         - size of the encoded memory block.
 * version      - version of the encoding
 * data         - actual encoded memory block
 * True on success, False on failure
 * DESCRIPTION
 * This function will encode ASVisual structure into memory block of
 * 32 bit values, suitable for storing in X property.
 *********/
/****f* libAfterImage/visual_prop2visual()
 * NAME
 * visual_prop2visual()
 * SYNOPSIS
 * Bool visual_prop2visual( ASVisual *asv, Display *dpy, int screen,
 *                          size_t size,
 *                          unsigned long version, unsigned long *data );
 * INPUTS
 * asv       - valid ASVisual structure.
 * dpy       - valid pointer to open X display.
 * screen    - screen number.
 * size      - encoded memory block's size.
 * version   - version of encoding.
 * data      - actual encoded memory block.
 * RETURN VALUE
 * True on success, False on failure
 * DESCRIPTION
 * visual_prop2visual() will read ASVisual data from the memory block
 * encoded by visual2visual_prop(). It could be used to read data from
 * X property and convert it into usable information - such as colormap,
 * visual info, etc.
 * Note: setup_truecolor_visual() or setup_pseudo_visual() has to be
 * invoked in order to complete ASVisual setup.
 *********/
Bool visual2visual_prop( ASVisual *asv, size_t *size,
	                     unsigned long *version, unsigned long **data );
Bool visual_prop2visual( ASVisual *asv, Display *dpy, int screen,
						 size_t size,
						 unsigned long version, unsigned long *data );
/* handy utility functions for creation of windows/pixmaps/XImages : */
/* this is from xc/programs/xserver/dix/window.h */
#define INPUTONLY_LEGAL_MASK (CWWinGravity | CWEventMask | \
		   				      CWDontPropagate | CWOverrideRedirect | \
							  CWCursor )
/****f* libAfterImage/create_visual_window()
 * NAME
 * create_visual_window()
 * SYNOPSIS
 * Window  create_visual_window( ASVisual *asv, Window parent,
 *                               int x, int y,
 *                               unsigned int width, unsigned int height,
 *                               unsigned int border_width,
 *                               unsigned int wclass,
 *                               unsigned long mask,
 *                               XSetWindowAttributes *attributes );
 * INPUTS
 * asv           - pointer to the valid ASVisual structure.
 * parent        - Window ID of the parent the window.
 * x, y          - initial position of the new window.
 * width, height - initial size of the new window.
 * border_width  - initial border width of the new window.
 * wclass         - Window class  - InputOnly or InputOutput.
 * mask          - defines what attributes are set.
 * attributes    - different window attributes.
 * RETURN VALUE
 * ID of the newly created window on success. None on failure.
 * DESCRIPTION
 * create_visual_window() will do sanity checks on passed parameters,
 * it will then add mandatory attributes if needed, and attempt to
 * create window for the specified ASVisual.
 *********/
/****f* libAfterImage/create_visual_gc()
 * NAME
 * create_visual_gc()
 * SYNOPSIS
 * GC      create_visual_gc( ASVisual *asv, Window root,
 *                           unsigned long mask, XGCValues *gcvalues );
 * INPUTS
 * asv            - pointer to the valid ASVisual structure.
 * root           - Window ID of the root window of destination screen
 * mask, gcvalues - values for creation of new GC - see XCreateGC() for
 *                  details.
 * RETURN VALUE
 * New GC created for regular window on success. NULL on failure.
 * DESCRIPTION
 * create_visual_gc() will create temporary window for the ASVisual
 * specific depth and Visual and it will then create GC for such window.
 * Obtained GC should be good to be used for manipulation of windows and
 * Pixmaps created for the same ASVisual.
 *********/
/****f* libAfterImage/create_visual_pixmap()
 * NAME
 * create_visual_pixmap()
 * SYNOPSIS
 * Pixmap  create_visual_pixmap( ASVisual *asv, Window root,
 *                               unsigned int width, unsigned int height,
 *                               unsigned int depth );
 * INPUTS
 * asv            - pointer to the valid ASVisual structure.
 * root           - Window ID of the root window of destination screen
 * width, height  - size of the pixmap to create.
 * depth          - depth of the pixmap to create. If 0 asv->true_depth
 *                  will be used.
 * RETURN VALUE
 * ID of the newly created pixmap on success. None on failure.
 * DESCRIPTION
 * create_visual_pixmap() will perform sanity checks on passed
 * parameters, and attempt to create pixmap for the specified ASVisual,
 * root and depth.
 *********/
/****f* libAfterImage/create_visual_ximage()
 * NAME
 * create_visual_ximage()
 * SYNOPSIS
 * XImage* create_visual_ximage( ASVisual *asv,
 *                               unsigned int width, unsigned int height,
 *                               unsigned int depth );
 * INPUTS
 * asv            - pointer to the valid ASVisual structure.
 * width, height  - size of the XImage to create.
 * depth          - depth of the XImage to create. If 0 asv->true_depth
 *                  will be used.
 * RETURN VALUE
 * pointer to newly created XImage on success. NULL on failure.
 * DESCRIPTION
 * create_visual_ximage() will perform sanity checks on passed
 * parameters, and it will attempt to create XImage of sufficient size,
 * and specified colordepth. It will also setup hooks for XImage
 * deallocation to be handled by custom function.
 *********/
Window  create_visual_window( ASVisual *asv, Window parent,
							  int x, int y,
							  unsigned int width, unsigned int height,
							  unsigned int border_width,
							  unsigned int wclass,
 					  		  unsigned long mask,
							  XSetWindowAttributes *attributes );
GC      create_visual_gc( ASVisual *asv, Window root,
	                          unsigned long mask, XGCValues *gcvalues );
Pixmap  create_visual_pixmap( ASVisual *asv, Window root,
	                          unsigned int width, unsigned int height,
							  unsigned int depth );
void destroy_visual_pixmap( ASVisual *asv, Pixmap *ppmap );

int get_dpy_drawable_size (Display *drawable_dpy, Drawable d, unsigned int *ret_w, unsigned int *ret_h);
Bool get_dpy_window_position (Display *window_dpy, Window root, Window w, int *px, int *py, int *transparency_x, int *transparency_y);


XImage* create_visual_ximage( ASVisual *asv,
	                          unsigned int width, unsigned int height,
							  unsigned int depth );
XImage* create_visual_scratch_ximage( ASVisual *asv,
	                          unsigned int width, unsigned int height,
							  unsigned int depth );

#define ASSHM_SAVED_MAX	(256*1024)

#ifdef XSHMIMAGE
Bool destroy_xshm_segment( unsigned long );
unsigned long ximage2shmseg( XImage *xim );
void flush_shm_cache();
#endif
Bool enable_shmem_images ();
void disable_shmem_images();
Bool check_shmem_images_enabled();

void* check_XImage_shared( XImage *xim );
Bool ASPutXImage( ASVisual *asv, Drawable d, GC gc, XImage *xim,
                  int src_x, int src_y, int dest_x, int dest_y,
				  unsigned int width, unsigned int height );
XImage * ASGetXImage( ASVisual *asv, Drawable d,
                  int x, int y, unsigned int width, unsigned int height,
				  unsigned long plane_mask );


#ifdef __cplusplus
}
#endif

#endif /* _SCREEN_ */
