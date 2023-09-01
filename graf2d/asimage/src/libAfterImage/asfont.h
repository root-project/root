#ifndef ASFONT_HEADER_ICLUDED
#define ASFONT_HEADER_ICLUDED


#include "asvisual.h"
#include "char2uni.h"

#ifdef __cplusplus
extern "C" {
#endif


/****h* libAfterImage/asfont.h
 * NAME
 * asfont
 * DESCRIPTION
 * Text drawing functionality.
 * Text is drawn as an ASImage with only alpha channel. Since alpha
 * channel is 8 bit widths that allows for 256 shades to be used in
 * rendered glyphs. That in turn allows for smoothing and antialiasing
 * of the drawn text. Such an approcah allows for easy manipulation of
 * the drawn text, such as changing color, making it transparent,
 * texturizing, rotation, etc.
 *
 * libAfterImage supports two types of fonts :
 * Fonts that could be rendered using standard Xlib functionality, and
 * fonts rendered by FreeType 2 library. That may include TrueType
 * fonts. When fonts are obtained via Xlib special processing is
 * performed in order to smooth its shape and leverage 256 shades
 * palette available.
 *
 * Any font being used is has to be opened first. At that time its
 * properties are analysed and glyphs are cached in clients memory.
 * Special RLE compression method is used for font glyphs, significantly
 * reducing memory utilization without any effect on performance.
 *
 * Font management and drawing functionality has been designed with
 * internatiolization in mind, althou support for locales is not
 * complete yet.
 * SEE ALSO
 * Structures :
 *          ASFontManager
 *          ASFont
 *          ASGlyph
 *          ASGlyphRange
 *
 * Functions :
 *          create_font_manager(), destroy_font_manager(),
 *          open_freetype_font(), open_X11_font(), get_asfont(),
 *          destroy_font(), print_asfont(), print_asglyph(),
 *          draw_text(),
 *          get_asfont_glyph_spacing(), set_asfont_glyph_spacing()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******************/

/****d* libAfterImage/MAX_GLYPHS_PER_FONT
 * NAME
 * MAX_GLYPHS_PER_FONT - Max value of glyphs per font allowed. 
 * We need that so we can detect and avoid broken fonts somehow.
 * SOURCE
 */
#define MAX_GLYPHS_PER_FONT  2048
/*************/

/* magic number identifying ASFont data structure */
#define MAGIC_ASFONT            0xA3A3F098

/****d* libAfterImage/ASFontType
 * NAME
 * ASFontType - Supported types of fonts - Xlib or FreeType 2
 * ASF_GuessWho will enable autodetection of the font type.
 * It is attempted to be opened as FreeType font first, and if that
 * fails - it will be opened as Xlib font.
 * SOURCE
 */
typedef enum
{
	ASF_X11 	 	= 0,
	ASF_Freetype 	= (0x01<<0),
	ASF_GuessWho 	= (0x01<<1),
#define	ASF_TypeMask  (ASF_GuessWho|ASF_Freetype)

	/* flags below should be combined with one of the above values */
	ASF_Monospaced 	= (0x01<<2),
	ASF_RightToLeft = (0x01<<3),  /* direction of the text flow */
	ASF_HasKerning 	= (0x01<<4)
}ASFontType;
/*************/

struct ASFontManager;
struct ASFont;

/****s* libAfterImage/ASGlyph
 * NAME
 * ASGlyph
 * DESCRIPTION
 * Stores glyph's image, as well as width, height and other
 * characteristics of the glyph.
 * SOURCE
 */
typedef struct ASGlyph
{
	CARD8 		   *pixmap ;        /* glyph's RLE encoded pixmap */
	short 	width, height ;	/* meaningfull width and height
									 * of the glyphs pixmap */
	short  lead, step ;			    /* distance pen position to glyph
									 * beginning and to the next glyph */
									/* in XRender it should be used as so:
									 * x = -lead, xOff = step
									 */
	short  ascend, descend ;        /* distance of the top of the
									 * glyph from the baseline */
									/* in XRender it should be used as so:
									 * y = -ascend, yOff = 0
									 */
	unsigned int font_gid ;		    /* index of the glyph inside the font( TTF only ) */
	long 		 xrender_gid ;	    /* Used only with XRender  - gid of the glyph in GlyphSet */	    
}ASGlyph;
/*************/

/****s* libAfterImage/ASGlyphRange
 * NAME
 * ASGlyphRange
 * DESCRIPTION
 * Organizes glyphs that belongs to the continuos range of char codes.
 * ASGlyphRange structures could be tied together to cover entire
 * codeset supported by the font.
 * SOURCE
 */
typedef struct ASGlyphRange
{
	unsigned long   min_char, max_char; /* Code range.
										 * for some locales that would
										 * be sufficient to simply set
										 * range of characteres
										 * supported by font */
	ASGlyph *glyphs;                    /* array of glyphs belonging to
										 * that code range */
	struct ASGlyphRange *below, *above;
}ASGlyphRange;
/*************/

#define MAX_SPECIAL_GLYPH	((ASGlyph*)0x00000003)
#define GLYPH_TAB	((ASGlyph*)0x00000003)
#define GLYPH_SPACE	((ASGlyph*)0x00000002)
#define GLYPH_EOL	((ASGlyph*)0x00000001)
#define GLYPH_EOT	((ASGlyph*)0x00000000)

/****s* libAfterImage/ASFont
 * NAME
 * ASFont
 * DESCRIPTION
 * Structure to contain all the font characteristics, as well as
 * set of glyph images. Such structure has to be created/opened prior to
 * being able to draw characters with any font.
 * SOURCE
 */
typedef struct ASFont
{
	unsigned long 	magic ;
	int 			ref_count ;

	struct ASFontManager *fontman;  /* our owner */
	char 				 *name;

	ASFontType  	type ;
	ASFlagType  	flags ;

	ASGlyphRange   *codemap; /* linked list of glyphsets, each
							  * representing continuos range of
							  * available codes - used for ASCII 
							  * codes */
	ASHashTable    *locale_glyphs;  /* hash of locale specific glyphs */

	ASGlyph         default_glyph;  /* valid glyph to be drawn when
									 * code is not valid */

	int 			max_height;  /* maximiu height of the character
								  * glyph */
	int	            max_ascend,  /* maximum distance from the baseline
								  * to the top of the character glyph */
					max_descend; /* need both descend and ascend to be 
								    able to dynamically recalculate font 
								    height while adding new characters */
	int				space_size;  /* fixed width value to be used when
								  * rendering spaces and tabs */
	int 			spacing_x, spacing_y;
	/* If these are set to anything above 0 then all the glyphs has to be 
	 * padded ( if its smaller then the cell ) or scaled down 
	 * ( if its bigger then the cell )
	 */								   
	int    cell_width, cell_height ;

#ifdef HAVE_FREETYPE
	FT_Face  		ft_face;        /* free type font handle */
#else
	CARD32         *pad;
#endif
	
	unsigned long	xrender_glyphset ;  /* GlyphSet is the actuall datatype, 
										 * but for easier compilation - 
										 * we use generic which is the same */ 
}ASFont;
/*************/
/****s* libAfterImage/ASFontManager
 * NAME
 * ASFontManager
 * DESCRIPTION
 * Global data identifying connection to external libraries, as well as
 * fonts location paths.
 * This structure has to be created/initialized prior to any font being
 * loaded.
 * It also holds list of fonts that are currently open, allowing for
 * easy access to fonts.
 * When ASFontManagaer object is destroyd it automagically closes all
 * the open fonts.
 * SOURCE
 */
typedef struct ASFontManager
{
	Display    *dpy;
	char 	   *font_path ;

	ASHashTable *fonts_hash ;

	size_t 		 unicode_used;
	CARD32		*local_unicode; /* list of unicodes from current locale
							     * - we use it to limit number of glyphs
							     * we load */
	Bool 		ft_ok ;
#ifdef HAVE_FREETYPE
	FT_Library  ft_library;     /* free type library handle */
#else
	void       *pad ;
#endif
}ASFontManager;
/*************/

/****d* libAfterImage/ASText3DType
 * NAME
 * ASText3DType - Available types of 3D text to be drawn.
 * SOURCE
 */
typedef enum ASText3DType{
	AST_Plain =0,	                           /* regular 2D text */
	AST_Embossed,
	AST_Sunken,
	AST_ShadeAbove,
	AST_ShadeBelow,
	AST_EmbossedThick,
	AST_SunkenThick,
	AST_OutlineAbove,
	AST_OutlineBelow,
	AST_OutlineFull,
	AST_3DTypes
}ASText3DType;
/*************/
typedef enum {
  ASCT_UTF8 = 0,
  ASCT_Char	= 1,
  ASCT_Unicode=sizeof(UNICODE_CHAR)
}ASCharType;


/****s* libAfterImage/ASTextAttributes
 * NAME
 * ASTextAttributes - Attributes for text rendering
 * SOURCE
 */
typedef struct ASTextAttributes
{
#define ASTA_UseTabStops	(0x01<<0)	
	unsigned int version ;     
	                /* structure version, so we can implement future 
	  				 * extensions without breaking binary apps */             
	ASFlagType   rendition_flags ;
	ASText3DType type;
	ASCharType 	 char_type;
 	unsigned int tab_size ;   /* tab size in chars  - defaults to 8 */
	unsigned int origin ;	  /* distance from the left margin 
							   * (in pixels) */
	unsigned int *tab_stops ; /* tab stops in pixels where 
							   * left margin is 0 */				   
	unsigned int tab_stops_num ;

	ARGB32		fore_color ;   /* used with 3D type of Outlined */
  unsigned int width;
#define ASTA_VERSION_1	1

#define ASTA_VERSION_INTERNAL	ASTA_VERSION_1
}ASTextAttributes;
/*************/

/****f* libAfterImage/asfont/create_font_manager()
 * NAME
 * create_font_manager()
 * SYNOPSIS
 * ASFontManager *create_font_manager( Display *dpy,
 *                                     const char *font_path,
 *                                     ASFontManager *reusable_memory );
 * INPUTS
 * dpy             - pointer to valid and opened Display.
 * font_path       - string, representing colon separated list of
 *                   directories to search for FreeType fonts.
 * reusable_memory - optional preallocated memory for the ASFontMagaer
 *                   object
 * RETURN VALUE
 * Pointer to Initialized ASFontManager object on success.
 * NULL otherwise.
 * DESCRIPTION
 * create_font_manager() will create new ASFontManager structure if
 * needed. It wioll then store copy of font_path and supplied pointer to
 * Display in it. At that time Hash table of loaded fonts is initialized,
 * and if needed FreeType library is initialized as well.
 * ASFontManager object returned by this functions has to be open at all
 * times untill text drawing is no longer needed.
 *********/
/****f* libAfterImage/asfont/destroy_font_manager()
 * NAME
 * destroy_font_manager()
 * SYNOPSIS
 * void destroy_font_manager( ASFontManager *fontman,
 *                            Bool reusable );
 * INPUTS
 * fontman  - pointer to valid ASFontManager object to be deallocated.
 * reusable - If True, then memory holding object itself will not be
 *            freed - only resources will be deallocated. That is
 *            usefull for structures created on stack.
 * DESCRIPTION
 * destroy_font_manager() closes all the fonts open with this
 * ASFontManager. It will also close connection to FreeType library, and
 * deallocate all cached data. If reusable is False - then memory used
 * for object itself will not be freed.
 *********/
struct ASFontManager *create_font_manager( Display *dpy, const char * font_path, struct ASFontManager *reusable_memory );
void    destroy_font_manager( struct ASFontManager *fontman, Bool reusable );

/****f* libAfterImage/asfont/open_freetype_font()
 * NAME
 * open_freetype_font()
 * SYNOPSIS
 * ASFont *open_freetype_font( ASFontManager *fontman,
 *                             const char *font_string,
 *                             int face_no,
 *                             int size, Bool verbose);
 * INPUTS
 * fontman     - pointer to previously created ASFontManager. Needed for
 *               connection to FreeType library, as well as path to
 *               search fonts in.
 * font_string - filename of the file containing font's data.
 * face_no     - number of face within the font file
 * size        - font size in points. Applicable only to scalable fonts,
 *               such as TrueType.
 * verbose     - if True, extensive error messages will be printed if
 *               problems encountered.
 * RETURN VALUE
 * pointer to Opened ASFont structure, containing all the glyphs of the
 * font, as well as other relevant info. On failure returns NULL.
 * DESCRIPTION
 * open_freetype_font() will attempt to find font file in any of the
 * directories specified in ASFontManager's font_path. If it fails to do
 * so - then it will check if filename has alldigit extentions. It will
 * then try to interpret that extention as a face number, and try and
 * find the file with extention stripped off.
 * If file was found function will atempt to read it using FreeType
 * library. If requested face is not available in the font - face 0 will
 * be used.
 * On success all the font's glyphs will be rendered and cached, and
 * needed font geometry info collected.
 * When FreeType Library is not available that function does nothing.
 *********/
/****f* libAfterImage/asfont/open_X11_font()
 * NAME
 * open_X11_font()
 * SYNOPSIS
 * ASFont *open_X11_font( ASFontManager *fontman,
 *                        const char *font_string);
 * INPUTS
 * fontman     - pointer to previously created ASFontManager. Needed for
 *               connection X Server.
 * font_string - name of the font as recognized by Xlib.
 * RETURN VALUE
 * pointer to Opened ASFont structure, containing all the glyphs of the
 * font, as well as other relevant info. On failure returns NULL.
 * DESCRIPTION
 * open_X11_font() attempts to load and query font using Xlib calls.
 * On success it goes thgroughthe codemap of the font and renders all
 * the glyphs available. Glyphs then gets transfered to the client's
 * memory and encoded using RLE compression. At this time smoothing
 * filters are applied on glyph pixmaps, if its size exceeds threshold.
 * TODO
 * implement proper XFontSet support, when used with I18N enabled.
 *********/
/****f* libAfterImage/asfont/get_asfont()
 * NAME
 * get_asfont()
 * SYNOPSIS
 * ASFont *get_asfont( ASFontManager *fontman,
 *                     const char *font_string,
 *                     int face_no, int size,
 *                     ASFontType type );
 * INPUTS
 * fontman     - pointer to previously created ASFontManager. Needed for
 *               connection to FreeType library, path to search fonts
 *               in, and X Server connection.
 * font_string - font name or filename of the file containing font's data.
 * face_no     - number of face within the font file
 * size        - font size in points. Applicable only to scalable fonts,
 *               such as TrueType.
 * type        - specifies the type of the font, or GuessWho for
 *               autodetection.
 * RETURN VALUE
 * pointer to Opened ASFont structure, containing all the glyphs of the
 * font, as well as other relevant info. On failure returns NULL.
 * DESCRIPTION
 * This function provides unified interface to font loading. It performs
 * search in ASFontManager's list to see if this specific font has been
 * loaded already, and if so - returns pointer to relevant structure.
 * Otherwise it tryes to load font as FreeType font first, and then
 * Xlib font, unless exact font type is specifyed.
 *********/
/****f* libAfterImage/asfont/release_font()
 * NAME
 * release_font()
 * SYNOPSIS
 * void release_font( ASFont *font );
 * INPUTS
 * font - pointer to the valid ASFont structure containing loaded font.
 * RETURN VALUE
 * returns current reference count. -1 means that object has been
 * destroyed.
 * DESCRIPTION
 * This function will decrement reference count on loaded font and if
 * reference count will be less then 0 ( meaning that release_font() has
 * been called more times then get_asfont() ) it will close the font,
 * remove it from ASFontManager's list, destroy all the glyphs and
 * generally free everything else used by ASFont.
 * Otherwise font will remain in memory for faster reuse.
 *********/
struct ASFont *open_freetype_font( struct ASFontManager *fontman, const char *font_string, int face_no, int size, Bool verbose);
struct ASFont *open_X11_font( struct ASFontManager *fontman, const char *font_string);
struct ASFont *get_asfont( struct ASFontManager *fontman, const char *font_string, int face_no, int size, ASFontType type_and_flags );
struct ASFont *dup_asfont( ASFont *font );
int            release_font( struct ASFont *font );

/****f* libAfterImage/asfont/print_asfont()
 * NAME
 * print_asfont()
 * SYNOPSIS
 * void    print_asfont( FILE* stream,
 *                       ASFont* font);
 * INPUTS
 * stream - output file pointer
 * font   - pointer to ASFont structure to print.
 * DESCRIPTION
 * prints all the geometry information about font.
 *********/
/****f* libAfterImage/asfont/print_asglyph()
 * NAME
 * print_asglyph()
 * SYNOPSIS
 * void 	print_asglyph( FILE* stream,
 *                         ASFont* font, unsigned long c);
 * INPUTS
 * stream - output file pointer
 * font   - pointer to ASFont structure to print.
 * c      - character code to print glyph for
 * DESCRIPTION
 * prints out contents of the cached glyph for specifyed character code.
 *********/
void    print_asfont( FILE* stream, struct ASFont* font);
void 	print_asglyph( FILE* stream, struct ASFont* font, unsigned long c);

ASGlyph** get_text_glyph_list (const char *text, ASFont *font, ASCharType char_type, int length);

/****f* libAfterImage/asfont/draw_text()
 * NAME
 * draw_text()
 * NAME
 * draw_fancy_text()
 * NAME
 * get_text_size()
 * NAME
 * get_fancy_text_size();
 * SYNOPSIS
 * ASImage *draw_text( const char *text,
 *                     ASFont *font, ASText3DType type,
 *                     int compression );
 * ASImage *draw_fancy_text( const void *text,
 *  						 struct ASFont *font, ASTextAttributes *attr,
 *							 int compression, 
 *							 int length );
 * Bool get_text_size( const char *text,
 *                     ASFont *font, ASText3DType type,
 *                     unsigned int *width, unsigned int *height );
 * Bool get_fancy_text_size( const void *text,
 *	                		 struct ASFont *font, ASTextAttributes *attr,
 *                   		 unsigned int *width, unsigned int *height,
 *							 int length,
 *							 int *x_positions );        
 *
 * INPUTS
 * text 		- actuall text to render
 * font 		- pointer to ASFont to render text with
 * type         - one of the few available types of 3D text.
 * compression  - compression level to attempt on resulting ASImage.
 * width        - pointer to value to be set to text width.
 * height       - pointer to value to be set to text height.
 * RETURN VALUE
 * Pointer to new ASImage containing rendered text on success.
 * NULL on failure.
 * DESCRIPTION
 * draw_text() creates new ASImage of the size big enough to contain
 * entire text. It then renders the text using supplied font as an alpha
 * channel of ASImage.
 * get_text_size() can be used to determine the size of the text about
 * to be drawn, so that appropriate drawable can be prepared.
 *********/
struct ASImage *draw_text( const char *text,
	                       struct ASFont *font, ASText3DType type,
						   int compression );
struct ASImage *draw_utf8_text( const char *text,
	                       struct ASFont *font, ASText3DType type,
						   int compression );
struct ASImage *draw_unicode_text( const CARD32 *text,
	                       struct ASFont *font, ASText3DType type,
						   int compression );
/* same as above only allows specifying the length of the text, instead of 
 * using zero-terminated strings :
 * Note: if length <= 0 then call is equivalent to above. 
 */
struct ASImage *draw_fancy_text( const void *text,
							struct ASFont *font, ASTextAttributes *attr,
							int compression, 
							int length );

Bool get_text_size( const char *text,
	                struct ASFont *font, ASText3DType type,
                    unsigned int *width, unsigned int *height );
Bool get_utf8_text_size( const char *text,
	                struct ASFont *font, ASText3DType type,
                    unsigned int *width, unsigned int *height );
Bool get_unicode_text_size( const CARD32 *text,
	                struct ASFont *font, ASText3DType type,
                    unsigned int *width, unsigned int *height );

Bool get_fancy_text_size( const void *text,
	                struct ASFont *font, ASTextAttributes *attr,
                    unsigned int *width, unsigned int *height,
					int length,
					int *x_positions );        
					/* array of length elements or NULL */
/****f* libAfterImage/asfont/get_asfont_glyph_spacing()
 * NAME
 * get_asfont_glyph_spacing()
 * SYNOPSIS
 * Bool get_asfont_glyph_spacing( ASFont* font, int *x, int *y );
 * INPUTS
 * font    - Loaded ASFont structure.
 * x       - pointer to the variable to receive horizontal spacing value.
 * y       - pointer to the variable to receive vertical spacing value.
 * RETURN VALUE
 * True if meaningfull information has been returned.
 * DESCRIPTION
 * Returns inter-glyph spacing of specified font.
 *********/
/****f* libAfterImage/asfont/set_asfont_glyph_spacing()
 * NAME
 * set_asfont_glyph_spacing()
 * SYNOPSIS
 * Bool set_asfont_glyph_spacing( ASFont* font, int x, int y );
 * INPUTS
 * font    - Loaded ASFont structure.
 * x       - new horizontal spacing value.
 * y       - new vertical spacing value.
 * RETURN VALUE
 * TRue on success.
 * DESCRIPTION
 * Changes inter-glyph spacing of the specified font.
 *********/

Bool get_asfont_glyph_spacing( ASFont* font, int *x, int *y );
Bool set_asfont_glyph_spacing( ASFont* font, int x, int y );

#ifdef __cplusplus
}
#endif


#endif
