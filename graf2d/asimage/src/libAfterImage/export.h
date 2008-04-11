#ifndef EXPORT_H_HEADER_INCLUDED
#define EXPORT_H_HEADER_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/****h* libAfterImage/export.h
 * NAME
 * export - Image output into different file formats.
 * SEE ALSO
 * Structures :
 *          ASXpmExportParams
 *          ASPngExportParams
 *          ASJpegExportParams
 *          ASGifExportParams
 *          ASImageExportParams
 *
 * Functions :
 *  		ASImage2file()
 *
 * Other libAfterImage modules :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 ******************/

/****d* libAfterImage/ExportFlags
 * NAME
 * EXPORT_GRAYSCALE - save image as grayscale.
 * NAME
 * EXPORT_ALPHA - save alpha channel if format permits
 * NAME
 * EXPORT_APPEND - if format allows multiple images - image will be 
 * appended
 * FUNCTION
 * Some common flags that could be used while writing images into
 * different file formats.
 * SOURCE
 */
#define EXPORT_GRAYSCALE			(0x01<<0)
#define EXPORT_ALPHA				(0x01<<1)
#define EXPORT_APPEND				(0x01<<3)  /* adds subimage  */
#define EXPORT_ANIMATION_REPEATS	(0x01<<4)  /* number of loops to repeat GIF animation */
/*****/

/****s* libAfterImage/ASXpmExportParams
 * NAME
 * ASXpmExportParams - parameters for export into XPM file.
 * SOURCE
 */
typedef struct
{
	ASImageFileTypes type;
	ASFlagType flags ;
	int dither ;
	int opaque_threshold ;
	int max_colors ;
}ASXpmExportParams ;
/*******/
/****s* libAfterImage/ASPngExportParams
 * NAME
 * ASPngExportParams - parameters for export into PNG file.
 * SOURCE
 */
typedef struct
{
	ASImageFileTypes type;
	ASFlagType flags ;
	int compression ;
}ASPngExportParams ;
/*******/
/****s* libAfterImage/ASJpegExportParams
 * NAME
 * ASJpegExportParams - parameters for export into JPEG file.
 * SOURCE
 */
typedef struct
{
	ASImageFileTypes type;
	ASFlagType flags ;
	int quality ;
}ASJpegExportParams ;
/*******/
/****s* libAfterImage/ASGifExportParams
 * NAME
 * ASGifExportParams - parameters for export into GIF file.
 * SOURCE
 */
typedef struct
{
	ASImageFileTypes type;
	ASFlagType flags ;
	int dither ;
	int opaque_threshold ;
	unsigned short animate_delay ;
	unsigned short animate_repeats ;
}ASGifExportParams ;
/*******/
/****s* libAfterImage/ASTiffExportParams
 * NAME
 * ASTiffExportParams - parameters for export into TIFF file.
 * SOURCE
 */
typedef struct
{
	ASImageFileTypes type;
	ASFlagType flags ;
	CARD32 rows_per_strip ;

/* these are suitable compressions : */
#define TIFF_COMPRESSION_NONE 		1
#define	TIFF_COMPRESSION_OJPEG		6	/* !6.0 JPEG */
#define	TIFF_COMPRESSION_JPEG		7
#define	TIFF_COMPRESSION_PACKBITS	32773	/* Macintosh RLE */
#define	TIFF_COMPRESSION_DEFLATE  	32946	/* Deflate compression */
	/* you should be able to use other values from tiff.h as well */
	CARD32 compression_type ;
	int jpeg_quality ;

	int opaque_threshold ;
}ASTiffExportParams ;
/*******/
/****s* libAfterImage/ASImageExportParams
 * NAME
 * ASImageExportParams - union of structures holding parameters for
 *   export into different file formats.
 * DESCRIPTION
 * Treatment of this union depends on what type of export was requested.
 * SEE ALSO
 * ASImageFileTypes
 * SOURCE
 */
typedef union ASImageExportParams
{
	ASImageFileTypes   type;
	ASXpmExportParams  xpm;
	ASPngExportParams  png;
	ASJpegExportParams jpeg;
	ASGifExportParams  gif;
	ASTiffExportParams tiff;
}ASImageExportParams;
/******/

typedef Bool (*as_image_writer_func)( ASImage *im, const char *path,
									  ASImageExportParams *params );
extern as_image_writer_func as_image_file_writers[ASIT_Unknown];


/****f* libAfterImage/export/ASImage2file()
 * NAME
 * ASImage2file()
 * SYNOPSIS
 * Bool ASImage2file( ASImage *im, const char *dir, const char *file,
					  ASImageFileTypes type, ASImageExportParams *params );
 * INPUTS
 * im			- Image to write out.
 * dir          - directory name to write file into (optional,
 *                could be NULL)
 * file         - file name with or without directory name.
 * type         - output file format. ( see ASImageFileTypes )
 * params       - pointer to ASImageExportParams union's member for the
 *                above type, with additional export parameters, such as
 *                quality, compression, etc. If NULL then all defaults
 *                will be used.
 * RETURN VALUE
 * True on success. False - failure.
 * DESCRIPTION
 * ASImage2file will construct filename out of dir and file components
 * and then will call specific filter to write out file in requested
 * format.
 * NOTES
 * Some formats support compression, others support lossy compression,
 * yet others allows you to limit number of colors and colordepth.
 * Each specific filter will try to interpret those parameters in its
 * own way.
 * EXAMPLE
 * asmerge.c: ASMerge.3
 *********/

Bool
ASImage2file( ASImage *im, const char *dir, const char *file,
			  ASImageFileTypes type, ASImageExportParams *params );


Bool
ASImage2PNGBuff( ASImage *im, CARD8 **buffer, int *size, ASImageExportParams *params );
Bool
ASImage2xpmRawBuff( ASImage *im, CARD8 **buffer, int *size, ASImageExportParams *params );


Bool ASImage2xpm ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2png ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2jpeg( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2xcf ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2ppm ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2bmp ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2ico ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2gif ( ASImage *im, const char *path, ASImageExportParams *params );
Bool ASImage2tiff( ASImage *im, const char *path, ASImageExportParams *params );

#ifdef __cplusplus
}
#endif


#endif
