#ifndef AFTERSTEP_XPM_H_HEADER_INCLUDED
#define AFTERSTEP_XPM_H_HEADER_INCLUDED

/* our own Xpm handling code : */

#include "scanline.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ASColormap;

typedef enum
{
	XPM_Outside = 0,
	XPM_InFile,
	XPM_InImage,
	XPM_InComments,
	XPM_InString
}ASXpmParseState;

#define MAX_XPM_BPP	16

typedef struct ASXpmFile
{
	int 	 fd ;
	char   **data;                             /* preparsed and preloaded data */

#define AS_XPM_BUFFER_UNDO		8
#define AS_XPM_BUFFER_SIZE		8192

#ifdef HAVE_LIBXPM
	XpmImage xpmImage;
#else
	char 	 *buffer;
	size_t   bytes_in;
	size_t   curr_byte;
#endif

	int 	 curr_img;
	int 	 curr_img_line;

	ASXpmParseState parse_state;

	char 	*str_buf ;
	size_t 	 str_buf_size ;

	unsigned short width, height, bpp;
	size_t     cmap_size;
	ASScanline scl ;

	ARGB32		*cmap, **cmap2;
	ASHashTable *cmap_name_xref;

	Bool do_alpha, full_alpha ;
}ASXpmFile;

typedef enum {
	XPM_Error = -2,
	XPM_EndOfFile = -1,
	XPM_EndOfImage = 0,
	XPM_Success = 1
}ASXpmStatus;

typedef struct ASXpmCharmap
{
	unsigned int count ;
	unsigned int cpp ;
	char *char_code ;
}ASXpmCharmap;

/*************************************************************************
 * High level xpm reading interface ;
 *************************************************************************/
void 		close_xpm_file( ASXpmFile **xpm_file );
ASXpmFile  *open_xpm_file( const char *realfilename );
Bool		parse_xpm_header( ASXpmFile *xpm_file );
ASXpmStatus get_xpm_string( ASXpmFile *xpm_file );
ASImage    *create_xpm_image( ASXpmFile *xpm_file, int compression );
Bool 		build_xpm_colormap( ASXpmFile *xpm_file );
Bool 		convert_xpm_scanline( ASXpmFile *xpm_file, unsigned int line );

ASXpmCharmap *build_xpm_charmap( struct ASColormap *cmap, Bool has_alpha,
	                             ASXpmCharmap *reusable_memory );
void destroy_xpm_charmap( ASXpmCharmap *xpm_cmap, Bool reusable );

#ifdef __cplusplus
}
#endif

#endif /* AFTERSTEP_XPM_H_HEADER_INCLUDED */
