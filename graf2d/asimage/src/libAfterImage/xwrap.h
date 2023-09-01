#ifndef XWRAP_H_HEADER_INCLUDED
#define XWRAP_H_HEADER_INCLUDED

#if !defined(X_DISPLAY_MISSING)
# include <X11/Xlib.h>
# include <X11/Xutil.h>
# include <X11/Xmd.h>
# include <X11/Xatom.h>
# include <X11/Xproto.h>
# include <X11/Xresource.h>

#else

#ifdef __cplusplus
extern "C" {
#endif

# define Display  void
# ifndef Bool
#  define Bool int
# endif
# ifndef True
#  define True 1
# endif
# ifndef False
#  define False 0
# endif

#ifdef HAVE_STDINT_H
# include <stdint.h>
# ifndef CARD32
#  define CARD32 uint32_t
# endif
# ifndef CARD16
#  define CARD16 uint16_t
# endif
# ifndef CARD8
#  define CARD8  uint8_t
# endif
#endif

# ifndef CARD32
#  define CARD32 unsigned int
# endif
# ifndef CARD16
#  define CARD16 unsigned short
# endif
# ifndef CARD8
#  define CARD8 unsigned char
# endif

# ifndef XID
#  define XID XID
   typedef CARD32 XID;
# endif

# ifndef Drawable
#  define Drawable   Drawable
   typedef XID Drawable;
# endif
# ifndef Atom
#  define Atom       Atom
   typedef XID Atom;
# endif
# ifndef Window
#  define Window     Window
   typedef XID Window;
# endif
# ifndef Pixmap
#  define Pixmap     Pixmap
   typedef XID Pixmap;
# endif
# ifndef Font
#  define Font       Font
   typedef XID Font;
# endif
# ifndef Colormap
#  define Colormap   Colormap
   typedef XID Colormap;
# endif
# ifndef Cursor
#  define Cursor     Cursor
   typedef XID Cursor;
# endif
# ifndef VisualID
#  define VisualID   VisualID
   typedef XID VisualID;
# endif

# ifndef GC
#  define GC GC
   typedef void* GC;
# endif

# ifndef None
#  define None 0
# endif

typedef struct {
	int function;		/* logical operation */
	unsigned long plane_mask;/* plane mask */
	unsigned long foreground;/* foreground pixel */
	unsigned long background;/* background pixel */
	int line_width;		/* line width */
	int line_style;	 	/* LineSolid, LineOnOffDash, LineDoubleDash */
	int cap_style;	  	/* CapNotLast, CapButt,
				   CapRound, CapProjecting */
	int join_style;	 	/* JoinMiter, JoinRound, JoinBevel */
	int fill_style;	 	/* FillSolid, FillTiled,
				   FillStippled, FillOpaeueStippled */
	int fill_rule;	  	/* EvenOddRule, WindingRule */
	int arc_mode;		/* ArcChord, ArcPieSlice */
	Pixmap tile;		/* tile pixmap for tiling operations */
	Pixmap stipple;		/* stipple 1 plane pixmap for stipping */
	int ts_x_origin;	/* offset for tile or stipple operations */
	int ts_y_origin;
        Font font;	        /* default text font for text operations */
	int subwindow_mode;     /* ClipByChildren, IncludeInferiors */
	Bool graphics_exposures;/* boolean, should exposures be generated */
	int clip_x_origin;	/* origin for clipping */
	int clip_y_origin;
	Pixmap clip_mask;	/* bitmap clipping; other calls for rects */
	int dash_offset;	/* patterned/dashed line information */
	char dashes;
} XGCValues;

typedef struct {
    unsigned char *value;		/* same as Property routines */
    Atom encoding;			/* prop type */
    int format;				/* prop data format: 8, 16, or 32 */
    unsigned long nitems;		/* number of data items in value */
} XTextProperty;

typedef struct {
	void *ext_data;	/* hook for extension to hang data */
	XID visualid;	/* visual id of this visual */
# if defined(__cplusplus) || defined(c_plusplus)
	int c_class;		/* C++ class of screen (monochrome, etc.) */
# else
	int class;		/* class of screen (monochrome, etc.) */
# endif
	unsigned long red_mask, green_mask, blue_mask;	/* mask values */
	int bits_per_rgb;	/* log base 2 of distinct color values */
	int map_entries;	/* color map entries */
} Visual;

typedef struct {
  Visual *visual;
  VisualID visualid;
  int screen;
  int depth;
# if defined(__cplusplus) || defined(c_plusplus)
  int c_class;					/* C++ */
# else
  int class;
# endif
  unsigned long red_mask;
  unsigned long green_mask;
  unsigned long blue_mask;
  int colormap_size;
  int bits_per_rgb;
} XVisualInfo;

typedef struct _XImage {
    int width, height;		/* size of image */
    int xoffset;		/* number of pixels offset in X direction */
    int format;			/* XYBitmap, XYPixmap, ZPixmap */
    char *data;			/* pointer to image data */
    int byte_order;		/* data byte order, LSBFirst, MSBFirst */
    int bitmap_unit;		/* quant. of scanline 8, 16, 32 */
    int bitmap_bit_order;	/* LSBFirst, MSBFirst */
    int bitmap_pad;		/* 8, 16, 32 either XY or ZPixmap */
    int depth;			/* depth of image */
    int bytes_per_line;		/* accelarator to next line */
    int bits_per_pixel;		/* bits per pixel (ZPixmap) */
    unsigned long red_mask;	/* bits in z arrangment */
    unsigned long green_mask;
    unsigned long blue_mask;
    void *obdata;		/* hook for the object routines to hang on */
    struct funcs {		/* image manipulation routines */
# if NeedFunctionPrototypes
	struct _XImage *(*create_image)(
		void* /* display */,
		void*		/* visual */,
		unsigned int	/* depth */,
		int		/* format */,
		int		/* offset */,
		char*		/* data */,
		unsigned int	/* width */,
		unsigned int	/* height */,
		int		/* bitmap_pad */,
		int		/* bytes_per_line */);
	int (*destroy_image)        (struct _XImage *);
	unsigned long (*get_pixel)  (struct _XImage *, int, int);
	int (*put_pixel)            (struct _XImage *, int, int, unsigned long);
	struct _XImage *(*sub_image)(struct _XImage *, int, int, unsigned int, unsigned int);
	int (*add_pixel)            (struct _XImage *, long);
# else
	struct _XImage *(*create_image)();
	int (*destroy_image)();
	unsigned long (*get_pixel)();
	int (*put_pixel)();
	struct _XImage *(*sub_image)();
	int (*add_pixel)();
# endif
	} f;
} XImage;

typedef struct {
    Pixmap background_pixmap;	/* background or None or ParentRelative */
    unsigned long background_pixel;	/* background pixel */
    Pixmap border_pixmap;	/* border of the window */
    unsigned long border_pixel;	/* border pixel value */
    int bit_gravity;		/* one of bit gravity values */
    int win_gravity;		/* one of the window gravity values */
    int backing_store;		/* NotUseful, WhenMapped, Always */
    unsigned long backing_planes;/* planes to be preseved if possible */
    unsigned long backing_pixel;/* value to use in restoring planes */
    Bool save_under;		/* should bits under be saved? (popups) */
    long event_mask;		/* set of events that should be saved */
    long do_not_propagate_mask;	/* set of events that should not propagate */
    Bool override_redirect;	/* boolean value for override-redirect */
    Colormap colormap;		/* color map to be associated with window */
    Cursor cursor;		/* cursor to be displayed (or None) */
} XSetWindowAttributes;

typedef struct {
	unsigned long pixel;
	unsigned short red, green, blue;
	char flags;  /* do_red, do_green, do_blue */
	char pad;
} XColor;

int XParseGeometry (  char *string,int *x,int *y,
                      unsigned int *width,    /* RETURN */
					  unsigned int *height);    /* RETURN */

/* needed by above function : */
# define NoValue		0x0000
# define XValue  	0x0001
# define YValue		0x0002
# define WidthValue  	0x0004
# define HeightValue  	0x0008
# define AllValues 	0x000F
# define XNegative 	0x0010
# define YNegative 	0x0020

# ifdef __cplusplus
}
# endif

#endif                         /* X_DISPLAY_MISSING */

# ifdef __cplusplus
extern "C" {
# endif

# if defined(ASIM_AFTERBASE_H_HEADER_INCLUDED)

int asim_get_drawable_size (Drawable d, unsigned int *ret_w, unsigned int *ret_h);
#   define get_drawable_size(d,w,h) asim_get_drawable_size((d),(w),(h))

# else

int grab_server();
int ungrab_server();
Bool is_server_grabbed();


Bool     get_drawable_size (Drawable d, unsigned int *ret_w, unsigned int *ret_h);
Drawable validate_drawable (Drawable d, unsigned int *pwidth, unsigned int *pheight);
void 	 backtrace_window ( const char *file, int line, Window w );

Window get_parent_window( Window w );
Window get_topmost_parent( Window w, Window *desktop_w );

# endif

# ifdef __cplusplus
}
# endif

#endif
