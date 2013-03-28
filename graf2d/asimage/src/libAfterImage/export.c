/* This file contains code for unified image writing into many file formats */
/********************************************************************/
/* Copyright (c) 2001 Sasha Vasko <sasha at aftercode.net>           */
/********************************************************************/
/*
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#undef LOCAL_DEBUG
#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

#ifdef HAVE_PNG
/* Include file for users of png library. */
# ifdef HAVE_BUILTIN_PNG
#  include "libpng/png.h"
# else
#  include <png.h>
# endif
#else
#include <setjmp.h>
# ifdef HAVE_JPEG
#   ifdef HAVE_UNISTD_H
#     include <unistd.h>
#   endif
#   include <stdio.h>
# endif
#endif
#ifdef HAVE_JPEG
/* Include file for users of jpg library. */
# undef HAVE_STDLIB_H
# ifndef X_DISPLAY_MISSING
#  include <X11/Xmd.h>
# endif
# ifdef HAVE_BUILTIN_JPEG
#  include "libjpeg/jpeglib.h"
# else
#  include <jpeglib.h>
# endif
#endif
/*#define DO_CLOCKING*/

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <string.h>
#include <ctype.h>
/* <setjmp.h> is used for the optional error recovery mechanism */

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif

#ifdef HAVE_GIF
# ifdef HAVE_BUILTIN_UNGIF
#  include "libungif/gif_lib.h"
# else
#  include <gif_lib.h>
# endif
#endif

#ifdef HAVE_TIFF
#include <tiff.h>
#include <tiffio.h>
#endif
#ifdef HAVE_LIBXPM
#ifdef HAVE_LIBXPM_X11
#include <X11/xpm.h>
#else
#include <xpm.h>
#endif
#endif

#include "asimage.h"
#include "imencdec.h"
#include "xcf.h"
#include "xpm.h"
#include "ungif.h"
#include "import.h"
#include "export.h"
#include "ascmap.h"
//#include "bmp.h"

#ifdef jmpbuf
#undef jmpbuf
#endif

/***********************************************************************************/
/* High level interface : 														   */
as_image_writer_func as_image_file_writers[ASIT_Unknown] =
{
	ASImage2xpm ,
	ASImage2xpm ,
	ASImage2xpm ,
	ASImage2png ,
	ASImage2jpeg,
	ASImage2xcf ,
	ASImage2ppm ,
	ASImage2ppm ,
	ASImage2bmp ,
	ASImage2ico ,
	ASImage2ico ,
	ASImage2gif ,
	ASImage2tiff,
	NULL,
	NULL,
	NULL
};

Bool
ASImage2file( ASImage *im, const char *dir, const char *file,
			  ASImageFileTypes type, ASImageExportParams *params )
{
	int   filename_len, dirname_len = 0 ;
	char *realfilename = NULL ;
	Bool  res = False ;
   int typei = (int) type;

	if( im == NULL ) return False;

	if( file )
	{
  		filename_len = strlen(file);
		if( dir != NULL )
			dirname_len = strlen(dir)+1;
		realfilename = safemalloc( dirname_len+filename_len+1 );
		if( dir != NULL )
		{
			strcpy( realfilename, dir );
			realfilename[dirname_len-1] = '/' ;
		}
		strcpy( realfilename+dirname_len, file );
#ifdef _WIN32
		unix_path2dos_path( realfilename );
#endif
	}
	if( type >= ASIT_Unknown || typei < 0 )
		show_error( "Hmm, I don't seem to know anything about format you trying to write file \"%s\" in\n.\tPlease check the manual", realfilename );
   	else if( as_image_file_writers[type] )
   		res = as_image_file_writers[type](im, realfilename, params);
   	else
   		show_error( "Support for the format of image file \"%s\" has not been implemented yet.", realfilename );

	free( realfilename );
	return res;
}

/* hmm do we need pixmap2file ???? */

/***********************************************************************************/
/* Some helper functions :                                                         */

FILE*
open_writable_image_file( const char *path )
{
	FILE *fp = NULL;
	if ( path )
	{
		if ((fp = fopen (path, "wb")) == NULL)
			show_error("cannot open image file \"%s\" for writing. Please check permissions.", path);
	}else
		fp = stdout ;
	return fp ;
}

void
scanline2raw( register CARD8 *row, ASScanline *buf, CARD8 *gamma_table, unsigned int width, Bool grayscale, Bool do_alpha )
{
	register int x = width;

	if( grayscale )
		row += do_alpha? width<<1 : width ;
	else
		row += width*(do_alpha?4:3) ;

	if( gamma_table )
	{
		if( !grayscale )
		{
			while ( --x >= 0 )
			{
				row -= 3 ;
				if( do_alpha )
				{
					--row;
					buf->alpha[x] = row[3];
				}
				buf->xc1[x]  = gamma_table[row[0]];
				buf->xc2[x]= gamma_table[row[1]];
				buf->xc3[x] = gamma_table[row[2]];
			}
		}else /* greyscale */
			while ( --x >= 0 )
			{
				if( do_alpha )
					buf->alpha[x] = *(--row);
				buf->xc1 [x] = buf->xc2[x] = buf->xc3[x]  = gamma_table[*(--row)];
			}
	}else
	{
		if( !grayscale )
		{
			while ( --x >= 0 )
			{
				row -= 3 ;
				if( do_alpha )
				{
					--row;
					buf->alpha[x] = row[3];
				}
				buf->xc1[x]  = row[0];
				buf->xc2[x]= row[1];
				buf->xc3[x] = row[2];
			}
		}else /* greyscale */
			while ( --x >= 0 )
			{
				if( do_alpha )
					buf->alpha[x] = *(--row);
				buf->xc1 [x] = buf->xc2[x] = buf->xc3[x]  = *(--row);
			}
	}
}

/***********************************************************************************/
#define SHOW_PENDING_IMPLEMENTATION_NOTE(f) \
	show_error( "I'm sorry, but " f " image writing is pending implementation. Appreciate your patience" )
#define SHOW_UNSUPPORTED_NOTE(f,path) \
	show_error( "unable to write file \"%s\" - " f " image format is not supported.\n", (path) )


#ifdef HAVE_XPM      /* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */

#ifdef LOCAL_DEBUG
Bool print_component( CARD32*, int, unsigned int );
#endif

Bool
ASImage2xpm ( ASImage *im, const char *path, ASImageExportParams *params )
{
	FILE *outfile;
	unsigned int y, x ;
	int *mapped_im, *row_pointer ;
	ASColormap         cmap = {0};
	ASXpmCharmap       xpm_cmap = {0};
	int transp_idx = 0;
	START_TIME(started);
	static const ASXpmExportParams defaultsXPM = { ASIT_Xpm, EXPORT_ALPHA, 4, 127, 512 };
	ASImageExportParams defaults;
	register char *ptr ;

	LOCAL_DEBUG_CALLER_OUT ("(\"%s\")", path);

	if( params == NULL ) {
           defaults.type = defaultsXPM.type;
           defaults.xpm = defaultsXPM;
           params = &defaults ;
        }

	if ((outfile = open_writable_image_file( path )) == NULL)
		return False;

    mapped_im = colormap_asimage( im, &cmap, params->xpm.max_colors, params->xpm.dither, params->xpm.opaque_threshold );
	if( !get_flags( params->xpm.flags, EXPORT_ALPHA) )
		cmap.has_opaque = False ;
	else
		transp_idx = cmap.count ;

LOCAL_DEBUG_OUT("building charmap%s","");
	build_xpm_charmap( &cmap, cmap.has_opaque, &xpm_cmap );
	SHOW_TIME("charmap calculation",started);

LOCAL_DEBUG_OUT("writing file%s","");
	fprintf( outfile, "/* XPM */\nstatic char *asxpm[] = {\n/* columns rows colors chars-per-pixel */\n"
					  "\"%d %d %d %d\",\n", im->width, im->height, xpm_cmap.count,  xpm_cmap.cpp );
    ptr = &(xpm_cmap.char_code[0]);
	for( y = 0 ; y < cmap.count ; y++ )
	{
		fprintf( outfile, "\"%s c #%2.2X%2.2X%2.2X\",\n", ptr, cmap.entries[y].red, cmap.entries[y].green, cmap.entries[y].blue );
		ptr += xpm_cmap.cpp+1 ;
	}
	if( cmap.has_opaque && y < xpm_cmap.count )
		fprintf( outfile, "\"%s c None\",\n", ptr );
	SHOW_TIME("image header writing",started);

	row_pointer = mapped_im ;
	for( y = 0 ; y < im->height ; y++ )
	{
		fputc( '"', outfile );
		for( x = 0; x < im->width ; x++ )
		{
			register int idx = (row_pointer[x] >= 0)? row_pointer[x] : transp_idx ;
			register char *ptr = &(xpm_cmap.char_code[idx*(xpm_cmap.cpp+1)]) ;
LOCAL_DEBUG_OUT( "(%d,%d)->%d (row_pointer %d )", x, y, idx, row_pointer[x] );
            if( idx > (int)cmap.count )
                show_error("bad XPM color index :(%d,%d) -> %d, %d: %s", x, y, idx, row_pointer[x], ptr );
			while( *ptr )
				fputc( *(ptr++), outfile );
		}
		row_pointer += im->width ;
		fputc( '"', outfile );
		if( y < im->height-1 )
			fputc( ',', outfile );
		fputc( '\n', outfile );
	}
	fprintf( outfile, "};\n" );
	if (outfile != stdout)
		fclose( outfile );

	SHOW_TIME("image writing",started);
	destroy_xpm_charmap( &xpm_cmap, True );
	free( mapped_im );
	destroy_colormap( &cmap, True );

	SHOW_TIME("total",started);
	return True;
}


/****** VO ******/
Bool
ASImage2xpmRawBuff ( ASImage *im, CARD8 **buffer, int *size, ASImageExportParams *params )
{
	unsigned int y, x ;
	int *mapped_im, *row_pointer ;
	ASColormap         cmap = {0};
	ASXpmCharmap       xpm_cmap = {0} ;
	int transp_idx = 0;
	START_TIME(started);
	static const ASXpmExportParams defaultsXPM = { ASIT_Xpm, EXPORT_ALPHA, 4, 127, 512 };
        ASImageExportParams defaults;
	register char *ptr ;
   char *curr;

   if( params == NULL ) {
      defaults.type = defaultsXPM.type;
      defaults.xpm = defaultsXPM;
      params = &defaults ;
   }

    mapped_im = colormap_asimage( im, &cmap, params->xpm.max_colors, params->xpm.dither, params->xpm.opaque_threshold );
	if (mapped_im == NULL)
		return False;
	if( !get_flags( params->xpm.flags, EXPORT_ALPHA) )
		cmap.has_opaque = False ;
	else
		transp_idx = cmap.count ;


LOCAL_DEBUG_OUT("building charmap%s","");
	build_xpm_charmap( &cmap, cmap.has_opaque, &xpm_cmap );
	SHOW_TIME("charmap calculation",started);

   *size = 0;
   *buffer = 0;

   /* crazy check against buffer overflow */
   if ((im->width > 100000) || (im->height > 1000000) || 
       (xpm_cmap.count > 100000) || (xpm_cmap.cpp > 100000)) {
		destroy_xpm_charmap( &xpm_cmap, True );
		free( mapped_im );
		destroy_colormap( &cmap, True );
      return False;
   }

   /* estimate size*/
   *size =  (im->width + 4)*im->height*xpm_cmap.cpp;
   *size += cmap.count*(20 + xpm_cmap.cpp);
   *size += 200;

   curr = calloc(*size, 1);
   *buffer = (CARD8*)curr;

	sprintf(curr, "/* XPM */\nstatic char *asxpm[] = {\n/* columns rows colors chars-per-pixel */\n"
					  "\"%d %d %d %d\",\n", im->width, im->height, xpm_cmap.count,  xpm_cmap.cpp );

   curr += strlen(curr);

    ptr = &(xpm_cmap.char_code[0]);
	for( y = 0 ; y < cmap.count ; y++ )
	{
		sprintf(curr, "\"%s c #%2.2X%2.2X%2.2X\",\n", ptr, cmap.entries[y].red, cmap.entries[y].green, cmap.entries[y].blue );
		ptr += xpm_cmap.cpp+1 ;
      curr += strlen(curr);
	}
	if( cmap.has_opaque && y < xpm_cmap.count ) {
		sprintf(curr, "\"%s c None\",\n", ptr );
      curr += strlen(curr);
   }
	SHOW_TIME("image header writing",started);

	row_pointer = mapped_im ;
	for( y = 0 ; y < im->height ; y++ )
	{
		*curr = '"';
      curr++;

		for( x = 0; x < im->width ; x++ )
		{
			register int idx = (row_pointer[x] >= 0)? row_pointer[x] : transp_idx ;
			register char *ptr = &(xpm_cmap.char_code[idx*(xpm_cmap.cpp+1)]) ;
         int len = strlen(ptr);

            if( idx > (int)cmap.count )
                show_error("bad XPM color index :(%d,%d) -> %d, %d: %s", x, y, idx, row_pointer[x], ptr );

         memcpy(curr, ptr, len);
         curr += len;
		}
		row_pointer += im->width ;
		*curr = '"';
      curr++;
		if( y < im->height-1 ) {
		   *curr = ',';
         curr++;
      }
		*curr = '\n';
      curr++;
	}
	sprintf( curr, "};\n" );

	destroy_xpm_charmap( &xpm_cmap, True );
	free( mapped_im );
	destroy_colormap( &cmap, True );
   *size = strlen((char*)*buffer);

	SHOW_TIME("total",started);
	return True;
}



#else  			/* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */

Bool
ASImage2xpm ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	SHOW_UNSUPPORTED_NOTE("XPM",path);
	return False ;
}

#endif 			/* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */
/***********************************************************************************/

/***********************************************************************************/
#ifdef HAVE_PNG		/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
static Bool
ASImage2png_int ( ASImage *im, void *data, png_rw_ptr write_fn, png_flush_ptr flush_fn, register ASImageExportParams *params )
{
	png_structp png_ptr  = NULL;
	png_infop   info_ptr = NULL;
	png_byte *row_pointer;
	int y ;
	Bool has_alpha;
	Bool grayscale;
	int compression;
	ASImageDecoder *imdec ;
	CARD32 *r, *g, *b, *a ;
	png_color_16 back_color ;

	START_TIME(started);
	static const ASPngExportParams defaults = { ASIT_Png, EXPORT_ALPHA, -1 };

	png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
    if ( png_ptr != NULL )
    	if( (info_ptr = png_create_info_struct(png_ptr)) != NULL )
			if( setjmp(png_jmpbuf(png_ptr)) )
			{
				png_destroy_info_struct(png_ptr, (png_infopp) &info_ptr);
				info_ptr = NULL ;
    		}


	if( params == NULL )
	{
		compression = defaults.compression ;
		grayscale = get_flags(defaults.flags, EXPORT_GRAYSCALE );
		has_alpha = get_flags(defaults.flags, EXPORT_ALPHA );
	}else
	{
		compression = params->png.compression ;
		grayscale = get_flags(params->png.flags, EXPORT_GRAYSCALE );
		has_alpha = get_flags(params->png.flags, EXPORT_ALPHA );
	}

	/* lets see if we have alpha channel indeed : */
	if( has_alpha )
	{
		if( !get_flags( get_asimage_chanmask(im), SCL_DO_ALPHA) )
			has_alpha = False ;
	}

	if((imdec = start_image_decoding( NULL /* default visual */ , im,
		                              has_alpha?SCL_DO_ALL:(SCL_DO_GREEN|SCL_DO_BLUE|SCL_DO_RED),
									  0, 0, im->width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return False;
	}

	if( !info_ptr)
	{
		if( png_ptr )
    		png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
		stop_image_decoding( &imdec );
    	return False;
    }

	if( write_fn == NULL && flush_fn == NULL ) 
	{	
		png_init_io(png_ptr, (FILE*)data);
	}else
	{
	    png_set_write_fn(png_ptr,data,(png_rw_ptr) write_fn, flush_fn );	
	}	 

	if( compression > 0 )
		png_set_compression_level(png_ptr,MIN(compression,99)/10);

	png_set_IHDR(png_ptr, info_ptr, im->width, im->height, 8,
		         grayscale ? (has_alpha?PNG_COLOR_TYPE_GRAY_ALPHA:PNG_COLOR_TYPE_GRAY):
		                     (has_alpha?PNG_COLOR_TYPE_RGB_ALPHA:PNG_COLOR_TYPE_RGB),
				 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
				 PNG_FILTER_TYPE_DEFAULT );
	/* better set background color as some web browsers can't seem to work without it ( IE in particular ) */
	memset( &back_color, 0x00, sizeof(png_color_16));
	back_color.red = ARGB32_RED16( im->back_color );
	back_color.green = ARGB32_GREEN16( im->back_color );
	back_color.blue = ARGB32_BLUE16( im->back_color );
	png_set_bKGD(png_ptr, info_ptr, &back_color);
	/* PNG treats alpha s alevel of opacity,
	 * and so do we - there is no need to reverse it : */
	/*	png_set_invert_alpha(png_ptr); */

	/* starting writing the file : writing info first */
	png_write_info(png_ptr, info_ptr);

	r = imdec->buffer.red ;
	g = imdec->buffer.green ;
	b = imdec->buffer.blue ;
	a = imdec->buffer.alpha ;

	if( grayscale )
	{
		row_pointer = safemalloc( im->width*(has_alpha?2:1));
		for ( y = 0 ; y < (int)im->height ; y++ )
		{
			register int i = im->width;
			CARD8   *ptr = (CARD8*)row_pointer;

			imdec->decode_image_scanline( imdec );
			if( has_alpha )
			{
				while( --i >= 0 ) /* normalized graylevel computing :  */
				{
                    ptr[(i<<1)] = (57*r[i]+181*g[i]+18*b[i])/256 ;
					ptr[(i<<1)+1] = a[i] ;
				}
			}else
				while( --i >= 0 ) /* normalized graylevel computing :  */
                    ptr[i] = (57*r[i]+181*g[i]+18*b[i])/256 ;
			png_write_rows(png_ptr, &row_pointer, 1);
		}
	}else
	{
/*		fprintf( stderr, "saving : %s\n", path );*/
		row_pointer = safecalloc( im->width * (has_alpha?4:3), 1 );
		for (y = 0; y < (int)im->height; y++)
		{
			register int i = im->width;
			CARD8   *ptr = (CARD8*)(row_pointer+(i-1)*(has_alpha?4:3)) ;
			imdec->decode_image_scanline( imdec );
			if( has_alpha )
			{
				while( --i >= 0 )
				{
					/* 0 is red, 1 is green, 2 is blue, 3 is alpha */
		            ptr[0] = r[i] ;
					ptr[1] = g[i] ;
					ptr[2] = b[i] ;
					ptr[3] = a[i] ;
					ptr-=4;
					/*fprintf( stderr, "#%2.2X%2.2X%2.2X%2.2X ", imbuf.alpha[i], imbuf.red[i], imbuf.green[i], imbuf.blue[i] );*/
				}
			}else
				while( --i >= 0 )
				{
					ptr[0] = r[i] ;
					ptr[1] = g[i] ;
					ptr[2] = b[i] ;
					ptr-=3;
					/*fprintf( stderr, "#%FFX%2.2X%2.2X%2.2X ", imbuf.red[i], imbuf.green[i], imbuf.blue[i] );*/
				}
			/*fprintf( stderr, "\n");*/
			png_write_rows(png_ptr, &row_pointer, 1);
		}
	}

	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	free( row_pointer );
	stop_image_decoding( &imdec );

	SHOW_TIME("image writing", started);
	return True ;
}

Bool
ASImage2png ( ASImage *im, const char *path, register ASImageExportParams *params )
{
	FILE *outfile;
	Bool res ;
	
	if( im == NULL )
		return False;
	
	if ((outfile = open_writable_image_file( path )) == NULL)
		return False;

	res = ASImage2png_int ( im, outfile, NULL, NULL, params );
	
	if (outfile != stdout)
		fclose(outfile);
	return res;
}

typedef struct ASImPNGBuffer
{
	CARD8 *buffer ; 
	int used_size, allocated_size ;
		 
}ASImPNGBuffer;

void asim_png_write_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
	ASImPNGBuffer *buff = (ASImPNGBuffer*) png_get_io_ptr(png_ptr); 
	if( buff && length > 0 )
	{
		if( buff->used_size + length > (unsigned int)buff->allocated_size ) 
		{                      /* allocating in 2048 byte increements : */
			buff->allocated_size = (buff->used_size + length + 2048)&0xFFFFF800 ; 
			buff->buffer = realloc( buff->buffer, buff->allocated_size );
		}	 
		memcpy( &(buff->buffer[buff->used_size]), data, length );
		buff->used_size += length ;
	}	 
}
	 
void asim_png_flush_data(png_structp png_ptr)
{
 	/* nothing to do really, but PNG requires it */	
}	 


Bool
ASImage2PNGBuff( ASImage *im, CARD8 **buffer, int *size, ASImageExportParams *params )
{
	ASImPNGBuffer int_buff  ;

	if( im == NULL || buffer == NULL || size == NULL ) 
		return False;
	
	memset( &int_buff, 0x00, sizeof(ASImPNGBuffer) );

 	if( ASImage2png_int ( im, &int_buff, (png_rw_ptr)asim_png_write_data, (png_flush_ptr)asim_png_flush_data, params ) )
	{
		*buffer	= int_buff.buffer ; 
		*size = int_buff.used_size ; 		   
		return True;
	}

	if( int_buff.buffer ) 
		free( int_buff.buffer );
	
	*buffer = NULL ; 
	*size = 0 ;
	return False;
}


#else 			/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
Bool
ASImage2png ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	SHOW_UNSUPPORTED_NOTE( "PNG", path );
	return False;
}

Bool
ASImage2PNGBuff( ASImage *im, CARD8 **buffer, int *size, ASImageExportParams *params )
{
	if( buffer ) 
		*buffer = NULL ; 
	if( size ) 
		*size = 0 ;
	return False;
}


#endif 			/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
/***********************************************************************************/


/***********************************************************************************/
#ifdef HAVE_JPEG     /* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */
Bool
ASImage2jpeg( ASImage *im, const char *path,  ASImageExportParams *params )
{
	/* This struct contains the JPEG decompression parameters and pointers to
	 * working space (which is allocated as needed by the JPEG library).
	 */
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	/* More stuff */
	FILE 		 *outfile;		/* target file */
    JSAMPROW      row_pointer[1];/* pointer to JSAMPLE row[s] */
	int 		  y;
	static const ASJpegExportParams defaultsJpeg = { ASIT_Jpeg, 0, -1 };
	ASImageExportParams defaults;
	Bool grayscale;
	ASImageDecoder *imdec ;
	CARD32 *r, *g, *b ;
	START_TIME(started);

	if( im == NULL )
		return False;

	if( params == NULL ) {
           defaults.type = defaultsJpeg.type;
           defaults.jpeg = defaultsJpeg;
           params = &defaults ;
        }

	if ((outfile = open_writable_image_file( path )) == NULL)
		return False;

	if((imdec = start_image_decoding( NULL /* default visual */ , im,
		                              (SCL_DO_GREEN|SCL_DO_BLUE|SCL_DO_RED),
									  0, 0, im->width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		if (outfile != stdout)
			fclose(outfile);
		return False;
	}


	grayscale = get_flags(params->jpeg.flags, EXPORT_GRAYSCALE );

	/* Step 1: allocate and initialize JPEG compression object */
	/* We have to set up the error handler first, in case the initialization
	* step fails.  (Unlikely, but it could happen if you are out of memory.)
	* This routine fills in the contents of struct jerr, and returns jerr's
	* address which we place into the link field in cinfo.
	*/
	cinfo.err = jpeg_std_error(&jerr);
	/* Now we can initialize the JPEG compression object. */
	jpeg_create_compress(&cinfo);

	/* Step 2: specify data destination (eg, a file) */
	/* Note: steps 2 and 3 can be done in either order. */
	/* Here we use the library-supplied code to send compressed data to a
	* stdio stream.  You can also write your own code to do something else.
	* VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
	* requires it in order to write binary files.
	*/
	jpeg_stdio_dest(&cinfo, outfile);

	/* Step 3: set parameters for compression */
	cinfo.image_width  = im->width; 	/* image width and height, in pixels */
	cinfo.image_height = im->height;
	cinfo.input_components = (grayscale)?1:3;		    /* # of color components per pixel */
	cinfo.in_color_space   = (grayscale)?JCS_GRAYSCALE:JCS_RGB; 	/* colorspace of input image */
	/* Now use the library's routine to set default compression parameters.
	* (You must set at least cinfo.in_color_space before calling this)*/
	jpeg_set_defaults(&cinfo);
	if( params->jpeg.quality > 0 )
		jpeg_set_quality(&cinfo, MIN(params->jpeg.quality,100), TRUE /* limit to baseline-JPEG values */);

	/* Step 4: Start compressor */
	/* TRUE ensures that we will write a complete interchange-JPEG file.*/
	jpeg_start_compress(&cinfo, TRUE);

	/* Step 5: while (scan lines remain to be written) */
	/*           jpeg_write_scanlines(...); */

	/* Here we use the library's state variable cinfo.next_scanline as the
	* loop counter, so that we don't have to keep track ourselves.
	* To keep things simple, we pass one scanline per call; you can pass
	* more if you wish, though.
	*/
	r = imdec->buffer.red ;
	g = imdec->buffer.green ;
	b = imdec->buffer.blue ;

	if( grayscale )
	{
		row_pointer[0] = safemalloc( im->width );
		for (y = 0; y < (int)im->height; y++)
		{
			register int i = im->width;
			CARD8   *ptr = (CARD8*)row_pointer[0];
			imdec->decode_image_scanline( imdec );
			while( --i >= 0 ) /* normalized graylevel computing :  */
				ptr[i] = (54*r[i]+183*g[i]+19*b[i])/256 ;
			(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
		}
	}else
	{
		row_pointer[0] = safemalloc( im->width * 3 );
		for (y = 0; y < (int)im->height; y++)
		{
			register int i = (int)im->width;
			CARD8   *ptr = (CARD8*)(row_pointer[0]+(i-1)*3) ;
LOCAL_DEBUG_OUT( "decoding  row %d", y );
			imdec->decode_image_scanline( imdec );
LOCAL_DEBUG_OUT( "building  row %d", y );
			while( --i >= 0 )
			{
				ptr[0] = r[i] ;
				ptr[1] = g[i] ;
				ptr[2] = b[i] ;
				ptr-=3;
			}
LOCAL_DEBUG_OUT( "writing  row %d", y );
			(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
		}
	}
LOCAL_DEBUG_OUT( "done writing image%s","" );
/*	free(buffer); */

	/* Step 6: Finish compression and release JPEG compression object*/
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	free( row_pointer[0] );
	
	stop_image_decoding( &imdec );
	if (outfile != stdout)
		fclose(outfile);

	SHOW_TIME("image export",started);
	LOCAL_DEBUG_OUT("done writing JPEG image \"%s\"", path);
	return True ;
}
#else 			/* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */

Bool
ASImage2jpeg( ASImage *im, const char *path,  ASImageExportParams *params )
{
	SHOW_UNSUPPORTED_NOTE( "JPEG", path );
	return False;
}

#endif 			/* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */
/***********************************************************************************/

/***********************************************************************************/
/* XCF - GIMP's native file format : 											   */

Bool
ASImage2xcf ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	/* More stuff */
	XcfImage  *xcf_im = NULL;
	START_TIME(started);

	SHOW_PENDING_IMPLEMENTATION_NOTE("XCF");
	if( xcf_im == NULL )
		return False;

#ifdef LOCAL_DEBUG
	print_xcf_image( xcf_im );
#endif
	/* Make a one-row-high sample array that will go away when done with image */
	SHOW_TIME("write initialization",started);

	free_xcf_image(xcf_im);
	SHOW_TIME("image export",started);
	return False ;
}

/***********************************************************************************/
/* PPM/PNM file format : 											   				   */
Bool
ASImage2ppm ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	START_TIME(started);
	SHOW_PENDING_IMPLEMENTATION_NOTE("PPM");
	SHOW_TIME("image export",started);
	return False;
}

/***********************************************************************************/
/* Windows BMP file format :   	see bmp.c								   				   */
/***********************************************************************************/
/* Windows ICO/CUR file format :   									   			   */
Bool
ASImage2ico ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	START_TIME(started);
	SHOW_PENDING_IMPLEMENTATION_NOTE("ICO");
	SHOW_TIME("image export",started);
	return False;
}

/***********************************************************************************/
#ifdef HAVE_GIF		/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

Bool ASImage2gif( ASImage *im, const char *path,  ASImageExportParams *params )
{
	FILE *outfile = NULL, *infile = NULL;
	GifFileType *gif = NULL ;
	ColorMapObject *gif_cmap ;
	Bool dont_save_cmap = False ;
	static const ASGifExportParams defaultsGif = { ASIT_Gif,EXPORT_ALPHA|EXPORT_APPEND, 3, 127, 10 };
        ASImageExportParams defaults;
	ASColormap         cmap;
	int *mapped_im ;
	int y ;
	GifPixelType *row_pointer ;
	Bool new_image = True ;
	START_TIME(started);
	int cmap_size = 1;
#define GIF_NETSCAPE_EXT_BYTES 3
	unsigned char netscape_ext_bytes[GIF_NETSCAPE_EXT_BYTES] = { 0x1, 0x0, 0x0};
#define GIF_GCE_BYTES 4	
	unsigned char gce_bytes[GIF_GCE_BYTES] = {0x01, 0x0, 0x0, 0x0 }; /* Graphic Control Extension bytes :
	                                                           		* first byte - flags (0x01 for transparency )
															   		* second and third bytes - animation delay
															   		* forth byte - transoparent pixel value.
															   		*/
	LOCAL_DEBUG_CALLER_OUT ("(\"%s\")", path);

	if( params == NULL ) {
           defaults.type = defaultsGif.type;
           defaults.gif = defaultsGif;
           params = &defaults ;
        }

	mapped_im = colormap_asimage( im, &cmap, 255, params->gif.dither, params->gif.opaque_threshold );

	if( get_flags( params->gif.flags, EXPORT_ALPHA) &&
		get_flags( get_asimage_chanmask(im), SCL_DO_ALPHA) )
		gce_bytes[GIF_GCE_TRANSPARENCY_BYTE] = cmap.count ;
	else
		gce_bytes[0] = 0 ;
#ifdef DEBUG_TRANSP_GIF
	fprintf( stderr, "***> cmap.count = %d, transp_byte = %X, flags = %d, chanmask = %d\n", cmap.count, gce_bytes[GIF_GCE_TRANSPARENCY_BYTE],
		     get_flags( params->gif.flags, EXPORT_ALPHA), get_flags( get_asimage_chanmask(im), SCL_DO_ALPHA) );
#endif
 	gce_bytes[GIF_GCE_DELAY_BYTE_HIGH] = (params->gif.animate_delay>>8)&0x00FF;
	gce_bytes[GIF_GCE_DELAY_BYTE_LOW] =  params->gif.animate_delay&0x00FF;

	if( get_flags( params->gif.flags, EXPORT_ANIMATION_REPEATS ) )
	{
		netscape_ext_bytes[GIF_NETSCAPE_REPEAT_BYTE_HIGH] = (params->gif.animate_repeats>>8)&0x00FF;
		netscape_ext_bytes[GIF_NETSCAPE_REPEAT_BYTE_LOW] = params->gif.animate_repeats&0x00FF;
	}		

	while( cmap_size < 256 && cmap_size < (int)cmap.count+(gce_bytes[0]&0x01) )
		cmap_size = cmap_size<<1 ;
	if( (gif_cmap = MakeMapObject(cmap_size, NULL )) == NULL )
	{
		free( mapped_im );
		ASIM_PrintGifError();
		return False;
	}
	memcpy( &(gif_cmap->Colors[0]), &(cmap.entries[0]), MIN(cmap.count,(unsigned int)cmap_size)*3 );

	if( get_flags(params->gif.flags, EXPORT_APPEND) && path != NULL)
		infile = fopen( path, "rb" );
	if( infile != NULL )
	{
		SavedImage *images = NULL ;
		int count = 0 ;
		/* TODO: do something about multiimage files !!! */
		gif = open_gif_read(infile);
		if( gif == NULL || get_gif_saved_images(gif, -1, &images, &count) == GIF_ERROR)
		{
			ASIM_PrintGifError();
			if( gif )
			{
				DGifCloseFile(gif);
				gif = NULL ;
			}
			if (infile)
			{
				fclose( infile );
				infile = NULL;
			}
		}else
		{
			GifFileType gif_src ;

			new_image = False ;
			gif_src = *gif ;
			gif->SColorMap = NULL ;
			gif->Image.ColorMap = NULL ;
			DGifCloseFile(gif);
			gif = NULL;
			fclose (infile);
			infile = NULL;
			outfile = open_writable_image_file( path );

			if (outfile)
				gif = EGifOpenFileHandle(fileno(outfile));
				
			if (gif)
			{
				int status;
				if( ( status = EGifPutScreenDesc(gif, gif_src.SWidth, gif_src.SHeight,
				                       gif_src.SColorResolution,
									   gif_src.SBackGroundColor,
									   gif_src.SColorMap )) == GIF_OK )
					status = write_gif_saved_images( gif, images, count );
				if( status != GIF_OK )
					ASIM_PrintGifError();
			}
			if (gif_src.SColorMap)
			{  /* we only want to save private colormap if it is any different from
			    * screen colormap ( saves us  768 bytes per image ) */
				if( gif_cmap->ColorCount == gif_src.SColorMap->ColorCount )
					dont_save_cmap = ( memcmp( gif_cmap->Colors, gif_src.SColorMap->Colors, gif_cmap->ColorCount*sizeof(GifColorType)) == 0 );
				FreeMapObject(gif_src.SColorMap);
			}
			if (gif)
			{
				EGifPutExtension(gif, GRAPHICS_EXT_FUNC_CODE, GIF_GCE_BYTES, &(gce_bytes[0]));
				if( get_flags( params->gif.flags, EXPORT_ANIMATION_REPEATS ) )
				{
					EGifPutExtensionFirst(gif, APPLICATION_EXT_FUNC_CODE, 11, "NETSCAPE2.0");
					EGifPutExtensionLast(gif, 0, GIF_NETSCAPE_EXT_BYTES, &(netscape_ext_bytes[0]));
				}
				
				if( EGifPutImageDesc(gif, 0, 0, im->width, im->height, FALSE, (dont_save_cmap)?NULL:gif_cmap ) == GIF_ERROR )
					ASIM_PrintGifError();
			}
		}
		free_gif_saved_images( images, count );
	}

	if (gif == NULL)
	{
		if (outfile == NULL)
			outfile = open_writable_image_file(path);
			
		if (outfile)
			if ((gif = EGifOpenFileHandle(fileno(outfile))) == NULL)
				ASIM_PrintGifError();
	}

	if( new_image && gif )
	{
		if( EGifPutScreenDesc(gif, im->width, im->height, cmap_size, 0, gif_cmap ) == GIF_ERROR )
			ASIM_PrintGifError();
	
		EGifPutExtension(gif, 0xf9, GIF_GCE_BYTES, &(gce_bytes[0]));
	
		if( EGifPutImageDesc(gif, 0, 0, im->width, im->height, FALSE, NULL ) == GIF_ERROR )
			ASIM_PrintGifError();
	}

	if( gif_cmap )
	{
		FreeMapObject(gif_cmap);
		gif_cmap = NULL ;
	}
	if( gif )
	{
		row_pointer = safemalloc( im->width*sizeof(GifPixelType));

		/* it appears to be much faster to write image out in line by line fashion */
		for( y = 0 ; y < (int)im->height ; y++ )
		{
			register int x = im->width ;
			register int *src = mapped_im + x*y;
	  	    while( --x >= 0 )
	  			row_pointer[x] = src[x] ;
			if( EGifPutLine(gif, row_pointer, im->width)  == GIF_ERROR)
				ASIM_PrintGifError();
		}
		free( row_pointer );
		if (EGifCloseFile(gif) == GIF_ERROR)
			ASIM_PrintGifError();
		gif = NULL;
	}
	free( mapped_im );
	destroy_colormap( &cmap, True );
	
	if (outfile 
#ifdef NO_DOUBLE_FCLOSE_AFTER_FDOPEN
		&& gif  /* can't do double fclose in MS CRT after VC2005 */
#endif
		&& outfile != stdout)
	{
		fclose (outfile);
	}

	SHOW_TIME("image export",started);
	return True ;
}
#else 			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */
Bool
ASImage2gif( ASImage *im, const char *path, ASImageExportParams *params )
{
	SHOW_UNSUPPORTED_NOTE("GIF",path);
	return False ;
}
#endif			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

#ifdef HAVE_TIFF/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */
Bool
ASImage2tiff( ASImage *im, const char *path, ASImageExportParams *params)
{
	TIFF *out;
	static const ASTiffExportParams defaultsTiff = { ASIT_Tiff, 0, (CARD32)-1, TIFF_COMPRESSION_NONE, 100, 0 };
        ASImageExportParams defaults;
	uint16 photometric = PHOTOMETRIC_RGB;
	tsize_t linebytes, scanline;
	ASImageDecoder *imdec ;
	CARD32 *r, *g, *b, *a ;
	unsigned char* buf;
	CARD32  row ;
	Bool has_alpha ;
	int nsamples = 3 ;
	START_TIME(started);

	if( params == NULL ) {
           defaults.type = defaultsTiff.type;
           defaults.tiff = defaultsTiff;
           params = &defaults ;
        }

	if( path == NULL )
	{
		SHOW_UNSUPPORTED_NOTE("TIFF streamed into stdout",path);
		return False ;
	}
	out = TIFFOpen(path, "w");
	if (out == NULL)
		return False;
	/* I don't really know why by grayscale images in Tiff does not work :(
	 * still here is the code :*/
	if( get_flags( params->tiff.flags, EXPORT_GRAYSCALE ) )
		nsamples = 1 ;
	has_alpha = get_flags( params->tiff.flags, EXPORT_ALPHA );
	if( has_alpha )
	{
		if( !get_flags( get_asimage_chanmask(im), SCL_DO_ALPHA) )
			has_alpha = False ;
		else
			++nsamples ;
	}

	if((imdec = start_image_decoding( NULL /* default visual */ , im,
		                              has_alpha?SCL_DO_ALL:(SCL_DO_GREEN|SCL_DO_BLUE|SCL_DO_RED),
									  0, 0, im->width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		TIFFClose(out);
		return False;
	}

	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, (uint32) im->width);
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, (uint32) im->height);
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, nsamples);
	if (has_alpha)
	{
	    uint16 v[1];
	    v[0] = EXTRASAMPLE_UNASSALPHA;
	    TIFFSetField(out, TIFFTAG_EXTRASAMPLES, 1, v);
	}

	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,   8);
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	if( params->tiff.compression_type == -1  )
		params->tiff.compression_type = defaultsTiff.compression_type ;
	TIFFSetField(out, TIFFTAG_COMPRESSION,  params->tiff.compression_type);
	switch (params->tiff.compression_type )
	{
		case COMPRESSION_JPEG:
			photometric = PHOTOMETRIC_YCBCR;
			if( params->tiff.jpeg_quality > 0 )
				TIFFSetField(out, TIFFTAG_JPEGQUALITY, params->tiff.jpeg_quality );
			TIFFSetField( out, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB );
			break;
	}
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, photometric);

	linebytes = im->width*nsamples;
	scanline = TIFFScanlineSize(out);
	if (scanline > linebytes)
	{
		buf = (unsigned char *)_TIFFmalloc(scanline);
		_TIFFmemset(buf+linebytes, 0, scanline-linebytes);
	} else
		buf = (unsigned char *)_TIFFmalloc(linebytes);
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP,
				 TIFFDefaultStripSize(out, params->tiff.rows_per_strip));

	r = imdec->buffer.red ;
	g = imdec->buffer.green ;
	b = imdec->buffer.blue ;
	a = imdec->buffer.alpha ;

	for (row = 0; row < im->height; ++row)
	{
		register int i = im->width, k = (im->width-1)*nsamples ;
		imdec->decode_image_scanline( imdec );

		if( has_alpha )
		{
			if( nsamples == 2 )
				while ( --i >= 0 )
				{
					buf[k+1] = a[i] ;
					buf[k] = (54*r[i]+183*g[i]+19*b[i])/256 ;
					k-= 2;
				}
			else
				while ( --i >= 0 )
				{
					buf[k+3] = a[i] ;
					buf[k+2] = b[i] ;
					buf[k+1] = g[i] ;
					buf[k] = r[i] ;
					k-= 4;
				}
		}else if( nsamples == 1 )
			while ( --i >= 0 )
				buf[k--] = (54*r[i]+183*g[i]+19*b[i])/256 ;
		else
			while ( --i >= 0 )
			{
				buf[k+2] = b[i] ;
				buf[k+1] = g[i] ;
				buf[k] = r[i] ;
				k-= 3;
			}

		if (TIFFWriteScanline(out, buf, row, 0) < 0)
			break;
	}
	stop_image_decoding( &imdec );
	TIFFClose(out);
	SHOW_TIME("image export",started);
	return True;
}
#else 			/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */

Bool
ASImage2tiff( ASImage *im, const char *path, ASImageExportParams *params )
{
	SHOW_UNSUPPORTED_NOTE("TIFF",path);
	return False ;
}
#endif			/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */
