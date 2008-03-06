#ifndef IMPORT_H_HEADER_INCLUDED
#define IMPORT_H_HEADER_INCLUDED

#include "asimage.h"
#include "xcf.h"
#include "xpm.h"

#ifdef __cplusplus
extern "C" {
#endif


/****h* libAfterImage/import.h
 * NAME
 * import - Image file format autodetection, reading and decoding 
 * routines.
 * SEE ALSO
 * Functions :
 *  		file2ASImage(), get_asimage(), file2pixmap()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******************/

/****d* libAfterImage/gamma
 * NAME
 * SCREEN_GAMMA - Defines default value for screen gamma correction.
 * SOURCE
 */
#define SCREEN_GAMMA 2.2
#define DEFAULT_PNG_IMAGE_GAMMA     0.45455
/*************/
/****s* libAfterImage/ASImageFileTypes
 * NAME
 * ASImageFileTypes - List of known image file formats.
 * SOURCE
 */
typedef enum
{
	ASIT_Xpm = 0,
	ASIT_ZCompressedXpm,
	ASIT_GZCompressedXpm,
	ASIT_Png,
	ASIT_Jpeg,
	ASIT_Xcf,
	ASIT_Ppm,
	ASIT_Pnm,
	ASIT_Bmp,
	ASIT_Ico,
	ASIT_Cur,
	ASIT_Gif,
	ASIT_Tiff,
	ASIT_XMLScript,
	ASIT_SVG,
	/* reserved for future implementation : */
	ASIT_Xbm,
	ASIT_Targa,
	ASIT_Supported = ASIT_Targa,
	ASIT_Pcx,
	ASIT_HTML,
	ASIT_XML,
	ASIT_Unknown
}ASImageFileTypes;
/*************/

/****s* libAfterImage/ASImageListEntry
 * NAME
 * ASImageListEntry - entry in linked list of images loaded from single 
 * directory.
 * SOURCE
 */
typedef struct ASImageListEntryBuffer
{
#define ASILEB_Dirty		(0x01<<0)		   
#define ASILEB_Binary		(0x01<<1)		   

	ASFlagType flags ; 

	size_t size ; 
	char  *data ; 
	
}ASImageListEntryBuffer;

typedef struct ASImageListEntry
{
#define MAGIC_ASIMAGE_LIST_ENTRY            0xA3A311E4
#define IS_ASIMAGE_LIST_ENTRY(e)  (((e)!=NULL)&&((e)->magic==MAGIC_ASIMAGE_LIST_ENTRY)&&((e)->ref_count>0)) 

	unsigned long magic ; 
	struct ASImageListEntry *prev, *next ;
	char   *name ;
	char   *fullfilename ;

	ASImageFileTypes 	type;
	ASImage 		   *preview;

#ifndef _WIN32
	mode_t d_mode;
#endif
    time_t d_mtime;
	off_t  d_size;		/* total size, in bytes */

	ASImageListEntryBuffer *buffer ; 

	int ref_count;
}ASImageListEntry;
/*************/

#define AS_IMPORT_ORIGINAL		0
#define AS_IMPORT_RESIZE_H		(0x01<<1)
#define AS_IMPORT_RESIZE_V		(0x01<<2)
#define AS_IMPORT_RESIZED		(AS_IMPORT_RESIZE_V|AS_IMPORT_RESIZE_H)
#define AS_IMPORT_SCALED_H		(0x01<<3)      /* if unset - then tile */
#define AS_IMPORT_SCALED_V		(0x01<<4)      /* if unset - then tile */
#define AS_IMPORT_SCALED_BOTH	(AS_IMPORT_SCALED_H|AS_IMPORT_SCALED_V)
#define AS_IMPORT_FAST			(0x01<<5)      /* can sacrifice quality for speed */


typedef struct ASImageImportParams
{
	ASFlagType 		flags ;                         /* see above */
	int 			width, height ;
	ASFlagType 		filter ; 					   /* what channels to load */
	double 			gamma ;
	CARD8 		   *gamma_table ;
	int 			subimage ;
	ASAltImFormats 	format ;
	unsigned int 	compression ;
	char 		  **search_path ;                 /* NULL terminated list  */
	
	int 			return_animation_delay ;
	int 			return_animation_repeats ;
}ASImageImportParams;

typedef ASImage* (*as_image_loader_func)( const char * path, ASImageImportParams *params );
extern as_image_loader_func as_image_file_loaders[ASIT_Unknown];

ASImage *xpm2ASImage ( const char * path, ASImageImportParams *params );
ASImage *xpm_data2ASImage( const char **data, ASImageImportParams *params );
ASImage *xpmRawBuff2ASImage( const char *data, ASImageImportParams *params );
ASImage *png2ASImage ( const char * path, ASImageImportParams *params );
ASImage *jpeg2ASImage( const char * path, ASImageImportParams *params );
ASImage *xcf2ASImage ( const char * path, ASImageImportParams *params );
ASImage *ppm2ASImage ( const char * path, ASImageImportParams *params );
ASImage *bmp2ASImage ( const char * path, ASImageImportParams *params );
ASImage *ico2ASImage ( const char * path, ASImageImportParams *params );
ASImage *gif2ASImage ( const char * path, ASImageImportParams *params );
ASImage *tiff2ASImage( const char * path, ASImageImportParams *params );
ASImage *xml2ASImage ( const char * path, ASImageImportParams *params );
ASImage *tga2ASImage ( const char * path, ASImageImportParams *params );
ASImage *svg2ASImage ( const char * path, ASImageImportParams *params );
ASImage *convert_argb2ASImage( ASVisual *asv, int width, int height, ARGB32 *argb, CARD8 *gamma_table );
ASImage *argb2ASImage( const char *path, ASImageImportParams *params );


/****f* libAfterImage/import/file2ASImage()
 * NAME
 * file2ASImage() - load ASImage from file.
 * SYNOPSIS
 * ASImage *file2ASImage( const char *file, ASFlagType what,
 *                        double gamma,
 *                        unsigned int compression, ... );
 * INPUTS
 * file         - file name with or without directory name
 * what         - reserved for future use
 * gamma        - gamma value to be used to correct image
 * compression  - compression level of the resulting ASImage
 * ...          - NULL terminated list of strings, representing
 *                arbitrary number of directories to be searched each.
 * RETURN VALUE
 * Pointer to ASImage structure holding image data on success.
 * NULL on failure
 * DESCRIPTION
 * file2ASImage will attempt to interpret filename in the following way:
 * 1)It will try to find file using unmodified filename in all the
 * provided search paths.
 * 2)It will attempt to append .gz and then .Z to the filename and
 * find such file in all the provided search paths.
 * 3)If filename ends with extension consisting of digits only - it will
 * attempt to find file with this extension stripped off. On success
 * this extension will be used to load subimage from the file with that
 * number. Subimages are supported only for XCF, GIF, BMP, ICO and CUR
 * files.
 * After the file is found file2ASImage() attempts to detect file format,
 * and if it is known it will load it into new ASImage structure.
 * EXAMPLE
 * asview.c: ASView.2
 *********/
/****f* libAfterImage/import/get_asimage()
 * NAME
 * get_asimage() - increment reference counter if file is already loaded,
 * or load image from file.
 * SYNOPSIS
 * ASImage *get_asimage( ASImageManager* imageman, const char *file,
 *                       ASFlagType what, unsigned int compression );
 * INPUTS
 * imageman     - pointer to valid ASVisual structure.
 * file         - root window ID for the destination screen.
 * what         - full image file's name with path.
 * compression  -
 * RETURN VALUE
 * Pointer to ASImage structure holding image data on success.
 * NULL on failure
 * DESCRIPTION
 * get_asimage will attempt check with the ASImageManager's list of load
 * images, and if image with requested filename already exists - it will
 * increment its reference count and return its pointer.
 * Otherwise it will call file2ASImage() to load image from file. It will
 * use PATH and gamma values from the ASImageManager to pass to
 * file2ASImage(). If image is successfully loaded - it will be added to
 * the ASImageManager's list and its pointer will be returned.
 * SEE ALSO
 * file2ASImage()
 *********/
ASImage *file2ASImage( const char *file, ASFlagType what, double gamma, unsigned int compression, ... );
ASImage *file2ASImage_extra( const char *file, ASImageImportParams *params );
ASImage *get_asimage( ASImageManager* imageman, const char *file, ASFlagType what, unsigned int compression );
ASImage *get_asimage_extra( ASImageManager* imageman, const char *file, ASImageImportParams *params );
ASImageFileTypes get_asimage_file_type( ASImageManager* imageman, const char *file );

#define AS_THUMBNAIL_PROPORTIONAL 		(0x01<<0)
#define AS_THUMBNAIL_DONT_ENLARGE 		(0x01<<1)
#define AS_THUMBNAIL_DONT_REDUCE		(0x01<<2)
ASImage *get_thumbnail_asimage( ASImageManager* imageman, const char *file, int thumb_width, int thumb_height, ASFlagType flags /* see above */);

ASImageFileTypes check_asimage_file_type( const char *realfilename );


Bool reload_asimage_manager( ASImageManager *imman );

#define NO_PREVIEW		 	0
#define LOAD_PREVIEW   		AS_IMPORT_RESIZED
#define SCALE_PREVIEW_H		AS_IMPORT_SCALED_H
#define SCALE_PREVIEW_V		AS_IMPORT_SCALED_V

ASImageListEntry *get_asimage_list( struct ASVisual *asv, const char *dir,
	                                ASFlagType preview_type, double gamma,
									unsigned int preview_width, unsigned int preview_height,
									unsigned int preview_compression,
									unsigned int *count_ret,
									int (*select) (const char *) );

ASImageListEntry *ref_asimage_list_entry( ASImageListEntry *entry );
ASImageListEntry *unref_asimage_list_entry( ASImageListEntry *entry );
ASImageListEntry *create_asimage_list_entry();
void destroy_asimage_list_entry_buffer( ASImageListEntryBuffer **pbuffer );
void destroy_asimage_list( ASImageListEntry **plist );
char *format_asimage_list_entry_details( ASImageListEntry *entry, Bool vertical );
Bool load_asimage_list_entry_data( ASImageListEntry *entry, size_t max_bytes );


/****f* libAfterImage/import/file2pixmap()
 * NAME
 * file2pixmap() - convinience function to load file into X Pixmap.
 * SYNOPSIS
 * Pixmap file2pixmap( struct ASVisual *asv, Window root,
 *                     const char *realfilename,
 *                     Pixmap *mask_out);
 * INPUTS
 * asv          - pointer to valid ASVisual structure.
 * root         - root window ID for the destination screen.
 * realfilename - full image file's name with path.
 * RETURN VALUE
 * Pixmap ID of the X Pixmap filled with loaded image. If mask_out is
 * not NULL it will point to image mask Pixmap ID if there is an alpha
 * channel in image, None otherwise.
 * On failure None will be returned.
 * DESCRIPTION
 * file2pixmap() will attempt to open specified file and autodetect its
 * format. If format is known it will load it into ASImage first, and
 * then convert it into X Pixmap. In case image has alpha channel -
 * mask Pixmap will be produced if mask_out is not NULL.
 *********/
Pixmap file2pixmap(struct ASVisual *asv, Window root, const char *realfilename, Pixmap *mask_out);

/**** VO ******/
ASImage *PNGBuff2ASimage( CARD8 *buffer, ASImageImportParams *params );

#ifdef __cplusplus
}
#endif

#endif

