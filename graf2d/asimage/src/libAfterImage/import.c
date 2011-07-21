/* This file contains code for unified image loading from many file formats */
/********************************************************************/
/* Copyright (c) 2001,2004 Sasha Vasko <sasha at aftercode.net>     */
/* Copyright (c) 2004 Maxim Nikulin <nikulin at gorodok.net>        */
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
/*#undef NO_DEBUG_OUTPUT*/
#ifndef NO_DEBUG_OUTPUT
#define DEBUG_TIFF
#endif

#undef LOCAL_DEBUG
#undef DO_CLOCKING
#undef DEBUG_TRANSP_GIF

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
# include <setjmp.h>
# ifdef HAVE_JPEG
#   ifdef HAVE_UNISTD_H
#     include <unistd.h>
#   endif
#   include <stdio.h>
# endif
#endif
#ifdef HAVE_JPEG
/* Include file for users of png library. */
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
#include <stdlib.h>
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <string.h>
#include <ctype.h>
/* <setjmp.h> is used for the optional error recovery mechanism */

#ifdef const
#undef const
#endif
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
#ifdef HAVE_SVG
#include <librsvg/rsvg.h>
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
#include "scanline.h"
#include "ximage.h"
#include "xcf.h"
#include "xpm.h"
#include "ungif.h"
#include "import.h"
#include "asimagexml.h"
#include "transform.h"

#ifdef jmpbuf
#undef jmpbuf
#endif


/***********************************************************************************/
/* High level interface : 														   */
static char *locate_image_file( const char *file, char **paths );
static ASImageFileTypes	check_image_type( const char *realfilename );

as_image_loader_func as_image_file_loaders[ASIT_Unknown] =
{
	xpm2ASImage ,
	xpm2ASImage ,
	xpm2ASImage ,
	png2ASImage ,
	jpeg2ASImage,
	xcf2ASImage ,
	ppm2ASImage ,
	ppm2ASImage ,
	bmp2ASImage ,
	ico2ASImage ,
	ico2ASImage ,
	gif2ASImage ,
	tiff2ASImage,
	xml2ASImage ,
	svg2ASImage ,
	NULL,
	tga2ASImage,
	NULL,
	NULL,
	NULL
};

const char *as_image_file_type_names[ASIT_Unknown+1] =
{
	"XPM" ,
	"Z-compressed XPM" ,
	"GZ-compressed XPM" ,
	"PNG" ,
	"JPEG",
	"GIMP Xcf" ,
	"PPM" ,
	"PNM" ,
	"MS Windows Bitmap" ,
	"MS Windows Icon" ,
	"MS Windows Cursor" ,
	"GIF" ,
	"TIFF",
	"AfterStep XML" ,
	"Scalable Vector Graphics (SVG)" ,
	"XBM",
	"Targa",
	"PCX",
	"HTML",
	"XML",
	"Unknown"
};

char *locate_image_file_in_path( const char *file, ASImageImportParams *iparams ) 
{
	int 		  filename_len ;
	char 		 *realfilename = NULL, *tmp = NULL ;
	register int i;
	ASImageImportParams dummy_iparams = {0};

	if( iparams == NULL )
		iparams = &dummy_iparams ;

	if( file )
	{
		filename_len = strlen(file);
#ifdef _WIN32
		for( i = 0 ; iparams->search_path[i] != NULL ; ++i ) 
			unix_path2dos_path( iparams->search_path[i] );
#endif

		/* first lets try to find file as it is */
		if( (realfilename = locate_image_file(file, iparams->search_path)) == NULL )
		{
			tmp = safemalloc( filename_len+3+1);
			strcpy(tmp, file);
		}
		if( realfilename == NULL )
		{ /* let's try and see if appending .gz will make any difference */
			strcpy(&(tmp[filename_len]), ".gz");
			realfilename = locate_image_file(tmp,iparams->search_path);
		}
		if( realfilename == NULL )
		{ /* let's try and see if appending .Z will make any difference */
			strcpy(&(tmp[filename_len]), ".Z");
			realfilename = locate_image_file(tmp,iparams->search_path);
		}
		if( realfilename == NULL )
		{ /* let's try and see if we have subimage number appended */
			for( i = filename_len-1 ; i > 0; i-- )
				if( !isdigit( (int)tmp[i] ) )
					break;
			if( i < filename_len-1 && i > 0 )
				if( tmp[i] == '.' )                 /* we have possible subimage number */
				{
					iparams->subimage = atoi( &tmp[i+1] );
					tmp[i] = '\0';
					filename_len = i ;
					realfilename = locate_image_file(tmp,iparams->search_path);
					if( realfilename == NULL )
					{ /* let's try and see if appending .gz will make any difference */
						strcpy(&(tmp[filename_len]), ".gz");
						realfilename = locate_image_file(tmp,iparams->search_path);
					}
					if( realfilename == NULL )
					{ /* let's try and see if appending .Z will make any difference */
						strcpy(&(tmp[filename_len]), ".Z");
						realfilename = locate_image_file(tmp,iparams->search_path);
					}
				}
		}
		if( tmp != realfilename && tmp != NULL )
			free( tmp );
	}
	if( realfilename == file )
		realfilename = mystrdup(file);
	return realfilename ;
}
ASImage *
file2ASImage_extra( const char *file, ASImageImportParams *iparams )
{
	char *realfilename ;
	ASImage *im = NULL;
	ASImageImportParams dummy_iparams = {0};
	
	if( iparams == NULL )
		iparams = &dummy_iparams ;

	realfilename = locate_image_file_in_path( file, iparams ); 
	
	if( realfilename != NULL ) 
	{
		ASImageFileTypes file_type = check_image_type( realfilename );

		if( file_type == ASIT_Unknown )
			show_error( "Hmm, I don't seem to know anything about format of the image file \"%s\"\n.\tPlease check the manual", realfilename );
		else if( as_image_file_loaders[file_type] )
		{
			char *g_var = getenv( "SCREEN_GAMMA" );
			if( g_var != NULL )
				iparams->gamma = atof(g_var);
			im = as_image_file_loaders[file_type](realfilename, iparams);
		}else
			show_error( "Support for the format of image file \"%s\" has not been implemented yet.", realfilename );
		/* returned image must not be tracked by any ImageManager yet !!! */
		if( im != NULL && im->imageman != NULL ) 
		{
			if( im->ref_count == 1 ) 
			{
				forget_asimage( im );
			}else
			{
				ASImage *tmp = clone_asimage( im , 0xFFFFFFFF); 
				if( tmp ) 
				{
					release_asimage( im );
					im = tmp ;
				}
			}
		}

#ifndef NO_DEBUG_OUTPUT
		if( im != NULL ) 
			show_progress( "image loaded from \"%s\"", realfilename );
#endif
		free( realfilename );
	}else
		show_error( "I'm terribly sorry, but image file \"%s\" is nowhere to be found.", file );

	return im;
}

void init_asimage_import_params( ASImageImportParams *iparams ) 
{
	if( iparams ) 
	{
		iparams->flags = 0 ;
		iparams->width = 0 ;
		iparams->height = 0 ;
		iparams->filter = SCL_DO_ALL ;
		iparams->gamma = 0. ;
		iparams->gamma_table = NULL ;
		iparams->compression = 100 ;
		iparams->format = ASA_ASImage ;
		iparams->search_path = NULL;
		iparams->subimage = 0 ;
	}
}

ASImage *
file2ASImage( const char *file, ASFlagType what, double gamma, unsigned int compression, ... )
{
	int i ;
	char 		 *paths[MAX_SEARCH_PATHS+1] ;
	ASImageImportParams iparams ;
	va_list       ap;

	init_asimage_import_params( &iparams );
	iparams.gamma = gamma ;
	iparams.compression = compression ;
	iparams.search_path = &(paths[0]);
#if 0
	iparams.width = 1024 ; 
	iparams.height = -1 ; 	
	iparams.flags |= AS_IMPORT_SCALED_H|AS_IMPORT_SCALED_V ;
#endif
	va_start (ap, compression);
	for( i = 0 ; i < MAX_SEARCH_PATHS ; i++ )
	{	
		if( (paths[i] = va_arg(ap,char*)) == NULL )
			break;
	}		   
	paths[MAX_SEARCH_PATHS] = NULL ;
	va_end (ap);

	return file2ASImage_extra( file, &iparams );

}

Pixmap
file2pixmap(ASVisual *asv, Window root, const char *realfilename, Pixmap *mask_out)
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	Pixmap mask = None ;
	if( asv && realfilename )
	{
		double gamma = SCREEN_GAMMA;
		char  *gamma_str;
		ASImage *im = NULL;

		if ((gamma_str = getenv ("SCREEN_GAMMA")) != NULL)
		{
			gamma = atof (gamma_str);
			if (gamma == 0.0)
				gamma = SCREEN_GAMMA;
		}

		im = file2ASImage( realfilename, 0xFFFFFFFF, gamma, 0, NULL );

		if( im != NULL )
		{
			trg = asimage2pixmap( asv, root, im, NULL, False );
			if( mask_out )
				if( get_flags( get_asimage_chanmask(im), SCL_DO_ALPHA ) )
					mask = asimage2mask( asv, root, im, NULL, False );
			destroy_asimage( &im );
		}
	}
	if( mask_out )
	{
		if( *mask_out && asv )
			XFreePixmap( asv->dpy, *mask_out );
		*mask_out = mask ;
	}
#endif
	return trg ;
}

static ASImage *
load_image_from_path( const char *file, char **path, double gamma)
{
	ASImageImportParams iparams ;

	init_asimage_import_params( &iparams );
	iparams.gamma = gamma ;
	iparams.search_path = path;

	return file2ASImage_extra( file, &iparams );
}

ASImageFileTypes
get_asimage_file_type( ASImageManager* imageman, const char *file )
{
	ASImageFileTypes file_type = ASIT_Unknown ;
	if( file )
	{
		ASImageImportParams iparams ;
		char *realfilename ;
	
		init_asimage_import_params( &iparams );
		iparams.search_path = imageman?&(imageman->search_path[0]):NULL;
		realfilename = locate_image_file_in_path( file, &iparams ); 
	
		if( realfilename != NULL ) 
		{
			file_type = check_image_type( realfilename );
			free( realfilename );
		}
	}
	return file_type;
}


ASImage *
get_asimage( ASImageManager* imageman, const char *file, ASFlagType what, unsigned int compression )
{
	ASImage *im = NULL ;
	if( imageman && file )
		if( (im = fetch_asimage(imageman, file )) == NULL )
		{
			im = load_image_from_path( file, &(imageman->search_path[0]), imageman->gamma);
			if( im )
			{
				store_asimage( imageman, im, file );
				set_flags( im->flags, ASIM_NAME_IS_FILENAME );
			}
				
		}
	return im;
}

void 
calculate_proportions( int src_w, int src_h, int *pdst_w, int *pdst_h ) 
{
	int dst_w = pdst_w?*pdst_w:0 ; 
	int dst_h = pdst_h?*pdst_h:0 ; 
	
	if( src_w > 0 && src_w >= src_h && (dst_w > 0 || dst_h <= 0)) 
		dst_h = (src_h*dst_w)/src_w ; 
	else if( src_h > 0 ) 
		dst_w = (src_w*dst_h)/src_h ; 

	if( pdst_w ) *pdst_w = dst_w ; 	
	if( pdst_h ) *pdst_h = dst_h ; 
}

void print_asimage_func (ASHashableValue value);
ASImage *
get_thumbnail_asimage( ASImageManager* imageman, const char *file, int thumb_width, int thumb_height, ASFlagType flags )
{
	ASImage *im = NULL ;
	
	if( imageman && file )
	{
#define AS_THUMBNAIL_NAME_FORMAT	"%s_scaled_to_%dx%d"
		char *thumbnail_name = safemalloc( strlen(file)+sizeof(AS_THUMBNAIL_NAME_FORMAT)+32 );
		ASImage *original_im = query_asimage(imageman, file );

		if( thumb_width <= 0 && thumb_height <= 0 ) 
		{
			thumb_width = 48 ;
			thumb_height = 48 ;
		}

		if( get_flags(flags, AS_THUMBNAIL_PROPORTIONAL ) ) 
		{
			if( original_im != NULL )		
				calculate_proportions( original_im->width, original_im->height, &thumb_width, &thumb_height );
		}else
		{
			if( thumb_width == 0 ) 
				thumb_width = thumb_height ; 
			if( thumb_height == 0 ) 
				thumb_height = thumb_width ; 
		}

		if( thumb_width > 0 && thumb_height > 0 ) 
		{
			sprintf( thumbnail_name, AS_THUMBNAIL_NAME_FORMAT, file, thumb_width, thumb_height ) ;
			im = fetch_asimage(imageman, thumbnail_name );
			if( im == NULL )
			{
				if( original_im != NULL ) /* simply scale it down to a thumbnail size */
				{
					if( (( (int)original_im->width > thumb_width || (int)original_im->height > thumb_height ) && !get_flags( flags, AS_THUMBNAIL_DONT_REDUCE ) ) ||
						(( (int)original_im->width < thumb_width || (int)original_im->height < thumb_height ) && !get_flags( flags, AS_THUMBNAIL_DONT_ENLARGE ) ) )
					{
						im = scale_asimage( NULL, original_im, thumb_width, thumb_height, ASA_ASImage, 100, ASIMAGE_QUALITY_FAST );
						if( im != NULL ) 
							store_asimage( imageman, im, thumbnail_name );
					}else
						im = dup_asimage( original_im );
				}
			}
		}
		
		if( im == NULL ) 	
		{
			ASImage *tmp ; 
			ASImageImportParams iparams ;

			init_asimage_import_params( &iparams );
			iparams.gamma = imageman->gamma ;
			iparams.search_path = &(imageman->search_path[0]);
			
			iparams.width = thumb_width ; 
			iparams.height = thumb_height ; 
			if( !get_flags( flags, AS_THUMBNAIL_DONT_ENLARGE|AS_THUMBNAIL_DONT_REDUCE ) )
				iparams.flags |= AS_IMPORT_RESIZED|AS_IMPORT_SCALED_BOTH ; 
			
			if( get_flags( flags, AS_THUMBNAIL_DONT_ENLARGE ) )
				iparams.flags |= AS_IMPORT_FAST ; 
			
			tmp = file2ASImage_extra( file, &iparams );
			if( tmp ) 
			{
				im = tmp ; 
				if( (int)tmp->width != thumb_width || (int)tmp->height != thumb_height ) 
				{
					if( get_flags(flags, AS_THUMBNAIL_PROPORTIONAL ) ) 
					{
						calculate_proportions( tmp->width, tmp->height, &thumb_width, &thumb_height );
						sprintf( thumbnail_name, AS_THUMBNAIL_NAME_FORMAT, file, thumb_width, thumb_height );
						if( (im = query_asimage( imageman, thumbnail_name )) == NULL ) 
							im = tmp ; 
					}
					if( im == tmp )
					{
						if( (( (int)tmp->width > thumb_width || (int)tmp->height > thumb_height ) && !get_flags( flags, AS_THUMBNAIL_DONT_REDUCE ) ) ||
							(( (int)tmp->width < thumb_width || (int)tmp->height < thumb_height ) && !get_flags( flags, AS_THUMBNAIL_DONT_ENLARGE ) ) )
						{
							im = scale_asimage( NULL, tmp, thumb_width, thumb_height, ASA_ASImage, 100, ASIMAGE_QUALITY_FAST );
							if( im == NULL ) 
								im = tmp ;
						}
					}
				}			

				if( im != NULL )
				{
					if( im->imageman == NULL )
						store_asimage( imageman, im, thumbnail_name );
					else
						dup_asimage( im );
				}
				
				if( im != tmp ) 
					destroy_asimage( &tmp );				
			}
		
		}
								 
		if( thumbnail_name ) 
			free( thumbnail_name );
	}
	return im;
}


Bool
reload_asimage_manager( ASImageManager *imman )
{
#if (HAVE_AFTERBASE_FLAG==1)
	if( imman != NULL ) 
	{
		ASHashIterator iter ;
		if( start_hash_iteration (imman->image_hash, &iter) )
		{
			do
			{
				ASImage *im = curr_hash_data( &iter );
/*fprintf( stderr, "im = %p. flags = 0x%lX\n", im, im->flags );		*/
				if( get_flags( im->flags, ASIM_NAME_IS_FILENAME ) )
				{
/*fprintf( stderr, "reloading image \"%s\" ...", im->name );*/
					ASImage *reloaded_im = load_image_from_path( im->name, &(imman->search_path[0]), imman->gamma);
/*fprintf( stderr, "Done. reloaded_im = %p.\n", reloaded_im );*/					
					if( reloaded_im ) 
					{
						if( asimage_replace (im, reloaded_im) ) 
							free( reloaded_im );
						else
							destroy_asimage( &reloaded_im );
					}				
				}
			}while( next_hash_item( &iter ) );
			return True;		
		}
	}
#endif
	return False;
}


ASImageListEntry * 
ref_asimage_list_entry( ASImageListEntry *entry )
{
	if( entry ) 
	{
		if( IS_ASIMAGE_LIST_ENTRY(entry) )
			++(entry->ref_count);
		else
			entry = NULL ; 
	}
	return entry;
}
	 
ASImageListEntry *
unref_asimage_list_entry( ASImageListEntry *entry )
{
	if( entry ) 
	{	
		if( IS_ASIMAGE_LIST_ENTRY(entry) )
		{
			--(entry->ref_count);
			if( entry->ref_count  <= 0 )
			{
				ASImageListEntry *prev = entry->prev ; 
				ASImageListEntry *next = entry->next ; 
				if( !IS_ASIMAGE_LIST_ENTRY(prev) )
					prev = NULL ; 
				if( !IS_ASIMAGE_LIST_ENTRY(next) )
					next = NULL ; 
				if( prev ) 
					prev->next = next ; 
				if( next ) 
					next->prev = prev ; 

				if( entry->preview ) 
					safe_asimage_destroy( entry->preview );
				if( entry->name )
					free( entry->name );
				if( entry->fullfilename )
					free( entry->fullfilename );
				if( entry->buffer ) 
					destroy_asimage_list_entry_buffer( &(entry->buffer) );
				memset( entry, 0x00, sizeof(ASImageListEntry));
				free( entry );
				entry = NULL ; 
			}	 
		}else
			entry = NULL ;
	}
	return entry;
}	 

ASImageListEntry *
create_asimage_list_entry()
{
	ASImageListEntry *entry = safecalloc( 1, sizeof(ASImageListEntry));
	entry->ref_count = 1 ; 
	entry->magic = MAGIC_ASIMAGE_LIST_ENTRY ; 
	return entry;
}

void
destroy_asimage_list( ASImageListEntry **plist )
{
	if( plist )
	{		   
		ASImageListEntry *curr = *plist ;
		while( IS_ASIMAGE_LIST_ENTRY(curr) )
		{	
			ASImageListEntry *to_delete = curr ; 
			curr = curr->next ;
		 	unref_asimage_list_entry( to_delete );
		}
		*plist = NULL ;
	}
}

void destroy_asimage_list_entry_buffer( ASImageListEntryBuffer **pbuffer )
{
	if( pbuffer && *pbuffer ) 
	{		 
		if( (*pbuffer)->data ) 
			free( (*pbuffer)->data ) ;
		free( *pbuffer );
		*pbuffer = NULL ;
	}
}	 

struct ASImageListAuxData
{
	ASImageListEntry **pcurr;
	ASImageListEntry *last ;
	ASFlagType preview_type ;
	unsigned int preview_width ;
	unsigned int preview_height ;
	unsigned int preview_compression ;
	ASVisual *asv;
};

#ifndef _WIN32
Bool 
direntry2ASImageListEntry( const char *fname, const char *fullname, 
						   struct stat *stat_info, void *aux_data)
{
	struct ASImageListAuxData *data = (struct ASImageListAuxData*)aux_data;
	ASImageFileTypes file_type ;
	ASImageListEntry *curr ;
	   	
	if (S_ISDIR (stat_info->st_mode))
		return False;
	
	file_type = check_image_type( fullname );
	if( file_type != ASIT_Unknown && as_image_file_loaders[file_type] == NULL )
		file_type = ASIT_Unknown ;

	curr = create_asimage_list_entry();
	*(data->pcurr) = curr ; 
	if( data->last )
		data->last->next = curr ;
	curr->prev = data->last ;
	data->last = curr ;
	data->pcurr = &(data->last->next);

	curr->name = mystrdup( fname );
	curr->fullfilename = mystrdup(fullname);
	curr->type = file_type ;
   	curr->d_mode = stat_info->st_mode;
	curr->d_mtime = stat_info->st_mtime;
	curr->d_size  = stat_info->st_size;

	if( curr->type != ASIT_Unknown && data->preview_type != 0 )
	{
		ASImageImportParams iparams = {0} ;
		ASImage *im = as_image_file_loaders[file_type](fullname, &iparams);
		if( im )
		{
			int scale_width = im->width ;
			int scale_height = im->height ;
			int tile_width = im->width ;
			int tile_height = im->height ;

			if( data->preview_width > 0 )
			{
				if( get_flags( data->preview_type, SCALE_PREVIEW_H ) )
					scale_width = data->preview_width ;
				else
					tile_width = data->preview_width ;
			}
			if( data->preview_height > 0 )
			{
				if( get_flags( data->preview_type, SCALE_PREVIEW_V ) )
					scale_height = data->preview_height ;
				else
					tile_height = data->preview_height ;
			}
			if( scale_width != im->width || scale_height != im->height )
			{
				ASImage *tmp = scale_asimage( data->asv, im, scale_width, scale_height, ASA_ASImage, data->preview_compression, ASIMAGE_QUALITY_DEFAULT );
				if( tmp != NULL )
				{
					destroy_asimage( &im );
					im = tmp ;
				}
			}
			if( tile_width != im->width || tile_height != im->height )
			{
				ASImage *tmp = tile_asimage( data->asv, im, 0, 0, tile_width, tile_height, TINT_NONE, ASA_ASImage, data->preview_compression, ASIMAGE_QUALITY_DEFAULT );
				if( tmp != NULL )
				{
					destroy_asimage( &im );
					im = tmp ;
				}
			}
		}

		curr->preview = im ;
	}
	return True;
}
#endif

ASImageListEntry *
get_asimage_list( ASVisual *asv, const char *dir,
	              ASFlagType preview_type, double gamma,
				  unsigned int preview_width, unsigned int preview_height,
				  unsigned int preview_compression,
				  unsigned int *count_ret,
				  int (*select) (const char *) )
{
	ASImageListEntry *im_list = NULL ;
#ifndef _WIN32
	struct ASImageListAuxData aux_data ; 
	int count ; 
	
	aux_data.pcurr = &im_list;
	aux_data.last = NULL;
	aux_data.preview_type = preview_type;
	aux_data.preview_width = preview_width;
	aux_data.preview_height = preview_height;
	aux_data.preview_compression  = preview_compression;
	aux_data.asv = asv ; 
	
	
	if( asv == NULL || dir == NULL )
		return NULL ;

	count = my_scandir_ext ((char*)dir, select, direntry2ASImageListEntry, &aux_data);

	if( count_ret )
		*count_ret = count ;
#endif
	return im_list;
}

char *format_asimage_list_entry_details( ASImageListEntry *entry, Bool vertical )
{
	char *details_text ;

	if( entry ) 
	{	
		int type = (entry->type>ASIT_Unknown)?ASIT_Unknown:entry->type ; 
		details_text = safemalloc(128);
		if( entry->preview ) 
			sprintf( details_text, vertical?"File type: %s\nSize %dx%d":"File type: %s; Size %dx%d", as_image_file_type_names[type], entry->preview->width, entry->preview->height ); 	  
		else 
			sprintf( details_text, "File type: %s", as_image_file_type_names[type]);
	}else
		details_text = mystrdup("");		   
	return details_text;
}	 

Bool 
load_asimage_list_entry_data( ASImageListEntry *entry, size_t max_bytes )
{
	char * new_buffer ; 
	size_t new_buffer_size ;
	FILE *fp;
	Bool binary = False ; 
	if( entry == NULL ) 
		return False;
	if( entry->buffer == NULL ) 
		entry->buffer = safecalloc( 1, sizeof(ASImageListEntryBuffer) );
	if( (int)entry->buffer->size == entry->d_size || entry->buffer->size >= max_bytes )
		return True;
	new_buffer_size = min( max_bytes, (size_t)entry->d_size ); 
	new_buffer = malloc( new_buffer_size );
	if( new_buffer == NULL ) 
		return False ;
	if( entry->buffer->size > 0 ) 
	{	
		memcpy( new_buffer, entry->buffer->data, entry->buffer->size ) ;
		free( entry->buffer->data );
	}
	entry->buffer->data = new_buffer ; 
	/* TODO read new_buffer_size - entry->buffer_size bytes into the end of the buffer */
	fp = fopen(entry->fullfilename, "rb");
	if ( fp != NULL ) 
	{
		int len = new_buffer_size - entry->buffer->size ;
		if( entry->buffer->size > 0 ) 
			fseek( fp, entry->buffer->size, SEEK_SET );
		len = fread(entry->buffer->data, 1, len, fp);
		if( len > 0 ) 
			entry->buffer->size += len ;
		fclose(fp);
	}

	if( entry->type == ASIT_Unknown ) 
	{
		int i = entry->buffer->size ; 
		register char *ptr = entry->buffer->data ;
		while ( --i >= 0 )	
			if( !isprint(ptr[i]) && ptr[i] != '\n'&& ptr[i] != '\r'&& ptr[i] != '\t' )	
				break;
		binary = (i >= 0);				
	}else
		binary = (entry->type != ASIT_Xpm  && entry->type != ASIT_XMLScript &&
			  	  entry->type != ASIT_HTML && entry->type != ASIT_XML ); 
	if( binary ) 
		set_flags( entry->buffer->flags, ASILEB_Binary );
   	else
		clear_flags( entry->buffer->flags, ASILEB_Binary );
	 


	return True;
}

/***********************************************************************************/
/* Some helper functions :                                                         */

static char *
locate_image_file( const char *file, char **paths )
{
	char *realfilename = NULL;
	if( file != NULL )
	{
		realfilename = mystrdup( file );
#ifdef _WIN32
		unix_path2dos_path( realfilename );
#endif
		
		if( CheckFile( realfilename ) != 0 )
		{
			free( realfilename ) ;
			realfilename = NULL ;
			if( paths != NULL )
			{	/* now lets try and find the file in any of the optional paths :*/
				register int i = 0;
				do
				{
					if( i > 0 ) 
					{	
						show_progress( "looking for image \"%s\" in path [%s]", file, paths[i] );
					}		
					realfilename = find_file( file, paths[i], R_OK );
				}while( realfilename == NULL && paths[i++] != NULL );
			}
		}
	}
	return realfilename;
}

FILE*
open_image_file( const char *path )
{
	FILE *fp = NULL;
	if ( path )
	{
		if ((fp = fopen (path, "rb")) == NULL)
			show_error("cannot open image file \"%s\" for reading. Please check permissions.", path);
	}else
		fp = stdin ;
	return fp ;
}

static ASImageFileTypes
check_image_type( const char *realfilename )
{
	ASImageFileTypes type = ASIT_Unknown ;
	int filename_len = strlen( realfilename );
	FILE *fp ;
#define FILE_HEADER_SIZE	512

	/* lets check if we have compressed xpm file : */
	if( filename_len > 5 && (mystrncasecmp( realfilename+filename_len-5, ".html", 5 ) == 0 || 
							 mystrncasecmp( realfilename+filename_len-4, ".htm", 4 ) == 0 ))
		type = ASIT_HTML;
	else if( filename_len > 7 && mystrncasecmp( realfilename+filename_len-7, ".xpm.gz", 7 ) == 0 )
		type = ASIT_GZCompressedXpm;
	else if( filename_len > 6 && mystrncasecmp( realfilename+filename_len-6, ".xpm.Z", 6 ) == 0 )
		type = ASIT_ZCompressedXpm ;
	else if( (fp = open_image_file( realfilename )) != NULL )
	{
		char head[FILE_HEADER_SIZE+1] ;
		int bytes_in = 0 ;
		memset(&head[0], 0x00, sizeof(head));
		bytes_in = fread( &(head[0]), sizeof(char), FILE_HEADER_SIZE, fp );
		DEBUG_OUT("%s: head[0]=0x%2.2X(%d),head[2]=0x%2.2X(%d)\n", realfilename+filename_len-4, head[0], head[0], head[2], head[2] );
/*		fprintf( stderr, " IMAGE FILE HEADER READS : [%s][%c%c%c%c%c%c%c%c][%s], bytes_in = %d\n", (char*)&(head[0]),
						head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7], strstr ((char *)&(head[0]), "XPM"),bytes_in );
 */
		if( bytes_in > 3 )
		{
			if( (CARD8)head[0] == 0xff && (CARD8)head[1] == 0xd8 && (CARD8)head[2] == 0xff)
				type = ASIT_Jpeg;
			else if (strstr ((char *)&(head[0]), "XPM") != NULL)
				type =  ASIT_Xpm;
			else if (head[1] == 'P' && head[2] == 'N' && head[3] == 'G')
				type = ASIT_Png;
			else if (head[0] == 'G' && head[1] == 'I' && head[2] == 'F')
				type = ASIT_Gif;
			else if (head[0] == head[1] && (head[0] == 'I' || head[0] == 'M'))
				type = ASIT_Tiff;
			else if (head[0] == 'P' && isdigit(head[1]))
				type = (head[1]!='5' && head[1]!='6')?ASIT_Pnm:ASIT_Ppm;
			else if (head[0] == 0xa && head[1] <= 5 && head[2] == 1)
				type = ASIT_Pcx;
			else if (head[0] == 'B' && head[1] == 'M')
				type = ASIT_Bmp;
			else if (head[0] == 0 && head[2] == 1 && mystrncasecmp(realfilename+filename_len-4, ".ICO", 4)==0 )
				type = ASIT_Ico;
			else if (head[0] == 0 && head[2] == 2 &&
						(mystrncasecmp(realfilename+filename_len-4, ".CUR", 4)==0 ||
						 mystrncasecmp(realfilename+filename_len-4, ".ICO", 4)==0) )
				type = ASIT_Cur;
		}
		if( type == ASIT_Unknown && bytes_in  > 6 )
		{
			if( mystrncasecmp( head, "<HTML>", 6 ) == 0 )
				type = ASIT_HTML;	
		}	 
		if( type == ASIT_Unknown && bytes_in  > 8 )
		{
			if( strncmp(&(head[0]), XCF_SIGNATURE, (size_t) XCF_SIGNATURE_LEN) == 0)
				type = ASIT_Xcf;
	   		else if (head[0] == 0 && head[1] == 0 &&
			    	 head[2] == 2 && head[3] == 0 && head[4] == 0 && head[5] == 0 && head[6] == 0 && head[7] == 0)
				type = ASIT_Targa;
			else if (strncmp (&(head[0]), "#define", (size_t) 7) == 0)
				type = ASIT_Xbm;
			else if( mystrncasecmp(realfilename+filename_len-4, ".SVG", 4)==0 )
				type = ASIT_SVG ;
			else
			{/* the nastiest check - for XML files : */
				int i ;

				type = ASIT_XMLScript ;
				for( i = 0 ; i < bytes_in ; ++i ) if( !isspace(head[i]) ) break;
				while( bytes_in > 0 && type == ASIT_XMLScript )
				{
					if( i >= bytes_in )
					{	
						bytes_in = fread( &(head[0]), sizeof(CARD8), FILE_HEADER_SIZE, fp );
						for( i = 0 ; i < bytes_in ; ++i ) if( !isspace(head[i]) ) break;
					}
					else if( head[i] != '<' )
						type = ASIT_Unknown ;
					else if( mystrncasecmp( &(head[i]), "<svg", 4 ) == 0 ) 
					{
						type = ASIT_SVG ;
					}else if( mystrncasecmp( &(head[i]), "<!DOCTYPE ", 10 ) == 0 ) 
					{	
						type = ASIT_XML ;
						for( i += 9 ; i < bytes_in ; ++i ) if( !isspace(head[i]) ) break;
						if( i < bytes_in ) 
						{
					 		if( mystrncasecmp( &(head[i]), "afterstep-image-xml", 19 ) == 0 ) 			
							{
								i += 19 ;	  
								type = ASIT_XMLScript ;
							}
						}	 
					}else
					{
						while( bytes_in > 0 && type == ASIT_XMLScript )
						{
							while( ++i < bytes_in )
								if( !isspace(head[i]) )
								{
									if( !isprint(head[i]) )
									{
										type = ASIT_Unknown ;
										break ;
									}else if( head[i] == '>' )
										break ;
								}

							if( i >= bytes_in )
							{	
								bytes_in = fread( &(head[0]), sizeof(CARD8), FILE_HEADER_SIZE, fp );
								i = 0 ; 
							}else
								break ;
						}
						break;
					}	
				}
			}
		}
		fclose( fp );
	}
	return type;
}


ASImageFileTypes
check_asimage_file_type( const char *realfilename )
{
	if( realfilename == NULL ) 
		return ASIT_Unknown;
	return check_image_type( realfilename );
}

/***********************************************************************************/
#ifdef HAVE_XPM      /* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */

#ifdef LOCAL_DEBUG
Bool print_component( CARD32*, int, unsigned int );
#endif

static ASImage *
xpm_file2ASImage( ASXpmFile *xpm_file, unsigned int compression )
{
	ASImage *im = NULL ;
	int line = 0;

	LOCAL_DEBUG_OUT( "do_alpha is %d. im->height = %d, im->width = %d", xpm_file->do_alpha, xpm_file->height, xpm_file->width );
	if( build_xpm_colormap( xpm_file ) )
		if( (im = create_xpm_image( xpm_file, compression )) != NULL )
		{
			int bytes_count = im->width*4 ;
			ASFlagType rgb_flags = ASStorage_RLEDiffCompress|ASStorage_32Bit ;
			ASFlagType alpha_flags = ASStorage_RLEDiffCompress|ASStorage_32Bit ;
			int old_storage_block_size = set_asstorage_block_size( NULL, xpm_file->width*xpm_file->height*3/2 );

			if( !xpm_file->full_alpha ) 
				alpha_flags |= ASStorage_Bitmap ;
			for( line = 0 ; line < xpm_file->height ; ++line )
			{
				if( !convert_xpm_scanline( xpm_file, line ) )
					break;
				im->channels[IC_RED][line]   = store_data( NULL, (CARD8*)xpm_file->scl.red, bytes_count, rgb_flags, 0);
				im->channels[IC_GREEN][line] = store_data( NULL, (CARD8*)xpm_file->scl.green, bytes_count, rgb_flags, 0);
				im->channels[IC_BLUE][line]  = store_data( NULL, (CARD8*)xpm_file->scl.blue, bytes_count, rgb_flags, 0);
				if( xpm_file->do_alpha )
					im->channels[IC_ALPHA][line]  = store_data( NULL, (CARD8*)xpm_file->scl.alpha, bytes_count, alpha_flags, 0);
#ifdef LOCAL_DEBUG
				printf( "%d: \"%s\"\n",  line, xpm_file->str_buf );
				print_component( xpm_file->scl.red, 0, xpm_file->width );
				print_component( xpm_file->scl.green, 0, xpm_file->width );
				print_component( xpm_file->scl.blue, 0, xpm_file->width );
#endif
			}
			set_asstorage_block_size( NULL, old_storage_block_size);
		}
	return im ;
}

ASImage *
xpm2ASImage( const char * path, ASImageImportParams *params )
{
	ASXpmFile *xpm_file = NULL;
	ASImage *im = NULL ;
	START_TIME(started);

	LOCAL_DEBUG_CALLER_OUT ("(\"%s\", 0x%lX)", path, params->flags);
	if( (xpm_file=open_xpm_file(path)) == NULL )
	{
		show_error("cannot open image file \"%s\" for reading. Please check permissions.", path);
		return NULL;
	}

	im = xpm_file2ASImage( xpm_file, params->compression );
	close_xpm_file( &xpm_file );

	SHOW_TIME("image loading",started);
	return im;
}

ASXpmFile *open_xpm_data(const char **data);
ASXpmFile *open_xpm_raw_data(const char *data);

ASImage *
xpm_data2ASImage( const char **data, ASImageImportParams *params )
{
	ASXpmFile *xpm_file = NULL;
	ASImage *im = NULL ;
	START_TIME(started);

    LOCAL_DEBUG_CALLER_OUT ("(\"%s\", 0x%lX)", (char*)data, params->flags);
	if( (xpm_file=open_xpm_data(data)) == NULL )
	{
		show_error("cannot read XPM data.");
		return NULL;
	}

	im = xpm_file2ASImage( xpm_file, params->compression );
	close_xpm_file( &xpm_file );

	SHOW_TIME("image loading",started);
	return im;
}

ASImage *
xpmRawBuff2ASImage( const char *data, ASImageImportParams *params )
{
	ASXpmFile *xpm_file = NULL;
	ASImage *im = NULL ;
	START_TIME(started);

    LOCAL_DEBUG_CALLER_OUT ("(\"%s\", 0x%lX)", (char*)data, params->flags);
	if( (xpm_file=open_xpm_raw_data(data)) == NULL )
	{
		show_error("cannot read XPM data.");
		return NULL;
	}

	im = xpm_file2ASImage( xpm_file, params->compression );
	close_xpm_file( &xpm_file );

	SHOW_TIME("image loading",started);
	return im;
}

#else  			/* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */

ASImage *
xpm2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - XPM image format is not supported.\n", path );
	return NULL ;
}

#endif 			/* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */
/***********************************************************************************/

static inline void
apply_gamma( register CARD8* raw, register CARD8 *gamma_table, unsigned int width )
{
	if( gamma_table )
	{	
		register unsigned int i ;
		for( i = 0 ; i < width ; ++i )
			raw[i] = gamma_table[raw[i]] ;
	}
}

/***********************************************************************************/
#ifdef HAVE_PNG		/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
ASImage *
png2ASImage_int( void *data, png_rw_ptr read_fn, ASImageImportParams *params )
{

   double        image_gamma = DEFAULT_PNG_IMAGE_GAMMA;
	png_structp   png_ptr;
	png_infop     info_ptr;
	png_uint_32   width, height;
	int           bit_depth, color_type, interlace_type;
	int           intent;
	ASScanline    buf;
	CARD8         *upscaled_gray = NULL;
	Bool 	      do_alpha = False, grayscale = False ;
	png_bytep     *row_pointers, row;
	unsigned int  y;
	size_t		  row_bytes, offset ;
	static ASImage 	 *im = NULL ;
	int old_storage_block_size;
	START_TIME(started);

	/* Create and initialize the png_struct with the desired error handler
	 * functions.  If you want to use the default stderr and longjump method,
	 * you can supply NULL for the last three parameters.  We also supply the
	 * the compiler header file version, so that we know if the application
	 * was compiled with a compatible version of the library.  REQUIRED
	 */
	if((png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL)) != NULL )
	{
		/* Allocate/initialize the memory for image information.  REQUIRED. */
		if( (info_ptr = png_create_info_struct (png_ptr)) != NULL )
		{
		  	/* Set error handling if you are using the setjmp/longjmp method (this is
			 * the normal method of doing things with libpng).  REQUIRED unless you
			 * set up your own error handlers in the png_create_read_struct() earlier.
			 */
			if ( !setjmp(png_jmpbuf(png_ptr)) )
			{
				ASFlagType rgb_flags = ASStorage_RLEDiffCompress|ASStorage_32Bit ;

	         if(read_fn == NULL ) 
	         {	
		         png_init_io(png_ptr, (FILE*)data);
	         }else
	         {
	            png_set_read_fn(png_ptr, (void*)data, (png_rw_ptr) read_fn);
	         }	 

		    	png_read_info (png_ptr, info_ptr);
				png_get_IHDR (png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, NULL, NULL);

/*fprintf( stderr, "bit_depth = %d, color_type = %d, width = %d, height = %d\n", 
         bit_depth, color_type, width, height); 
*/
				if (bit_depth < 8)
				{/* Extract multiple pixels with bit depths of 1, 2, and 4 from a single
				  * byte into separate bytes (useful for paletted and grayscale images).
				  */
					if( bit_depth == 1 ) 
					{
						set_flags( rgb_flags, ASStorage_Bitmap );
						png_set_packing (png_ptr);
					}else
					{
						/* even though 2 and 4 bit values get expanded into a whole bytes the 
						   values don't get scaled accordingly !!! 
						   WE will have to take care of it ourselves :
						*/	
						upscaled_gray = safemalloc(width+8);
					}
				}else if (bit_depth == 16)
				{/* tell libpng to strip 16 bit/color files down to 8 bits/color */
					png_set_strip_16 (png_ptr);
				}

				/* Expand paletted colors into true RGB triplets */
				if (color_type == PNG_COLOR_TYPE_PALETTE)
				{
					png_set_expand (png_ptr);
					color_type = PNG_COLOR_TYPE_RGB;
				}

				/* Expand paletted or RGB images with transparency to full alpha channels
				 * so the data will be available as RGBA quartets.
		 		 */
   				if( color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY )
   				{
				   	if( png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
					{
						png_set_expand(png_ptr);
						color_type |= PNG_COLOR_MASK_ALPHA;
					}
   				}else
				{
					png_set_filler( png_ptr, 0xFF, PNG_FILLER_AFTER );
					color_type |= PNG_COLOR_MASK_ALPHA;
				}

/*				if( color_type == PNG_COLOR_TYPE_RGB )
					color_type = PNG_COLOR_TYPE_RGB_ALPHA ;
   				else
					color_type = PNG_COLOR_TYPE_GRAY_ALPHA ;
  */
				if (png_get_sRGB (png_ptr, info_ptr, &intent))
				{
                    png_set_gamma (png_ptr, params->gamma, DEFAULT_PNG_IMAGE_GAMMA);
				}else if (png_get_gAMA (png_ptr, info_ptr, &image_gamma) && bit_depth >= 8)
				{/* don't gamma-correct 1, 2, 4 bpp grays as we loose data this way */
					png_set_gamma (png_ptr, params->gamma, image_gamma);
				}else
				{
                    png_set_gamma (png_ptr, params->gamma, DEFAULT_PNG_IMAGE_GAMMA);
				}

				/* Optional call to gamma correct and add the background to the palette
				 * and update info structure.  REQUIRED if you are expecting libpng to
				 * update the palette for you (ie you selected such a transform above).
				 */

				png_read_update_info (png_ptr, info_ptr);

				png_get_IHDR (png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, NULL, NULL);

				im = create_asimage( width, height, params->compression );
				do_alpha = ((color_type & PNG_COLOR_MASK_ALPHA) != 0 );
				grayscale = ( color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
				              color_type == PNG_COLOR_TYPE_GRAY) ;

/* fprintf( stderr, "do_alpha = %d, grayscale = %d, bit_depth = %d, color_type = %d, width = %d, height = %d\n", 
         do_alpha, grayscale, bit_depth, color_type, width, height); */

				if( !do_alpha && grayscale ) 
					clear_flags( rgb_flags, ASStorage_32Bit );
				else
					prepare_scanline( im->width, 0, &buf, False );

				row_bytes = png_get_rowbytes (png_ptr, info_ptr);
				/* allocating big chunk of memory at once, to enable mmap
				 * that will release memory to system right after free() */
				row_pointers = safemalloc( height * sizeof( png_bytep ) + row_bytes * height );
				row = (png_bytep)(row_pointers + height) ;
				for (offset = 0, y = 0; y < height; y++, offset += row_bytes)
					row_pointers[y] = row + offset;

				/* The easiest way to read the image: */
				png_read_image (png_ptr, row_pointers);

				old_storage_block_size = set_asstorage_block_size( NULL, width*height*3/2 );
				for (y = 0; y < height; y++)
				{
					if( do_alpha || !grayscale ) 
					{	
						raw2scanline( row_pointers[y], &buf, NULL, buf.width, grayscale, do_alpha );
						im->channels[IC_RED][y] = store_data( NULL, (CARD8*)buf.red, buf.width*4, rgb_flags, 0);
					}else
					{
						if ( bit_depth == 2 )
						{
							int i, pixel_i = -1;
							static CARD8  gray2bit_translation[4] = {0,85,170,255};
							for ( i = 0 ; i < row_bytes ; ++i )
							{
								CARD8 b = row_pointers[y][i];
								upscaled_gray[++pixel_i] = gray2bit_translation[b&0x03];
								upscaled_gray[++pixel_i] = gray2bit_translation[(b&0xC)>>2];
								upscaled_gray[++pixel_i] = gray2bit_translation[(b&0x30)>>4];
								upscaled_gray[++pixel_i] = gray2bit_translation[(b&0xC0)>>6];
							}
							im->channels[IC_RED][y] = store_data( NULL, upscaled_gray, width, rgb_flags, 0);
						}else if ( bit_depth == 4 )
						{
							int i, pixel_i = -1;
							static CARD8  gray4bit_translation[16] = {0,17,34,51,  68,85,102,119, 136,153,170,187, 204,221,238,255};
							for ( i = 0 ; i < row_bytes ; ++i )
							{
								CARD8 b = row_pointers[y][i];
								upscaled_gray[++pixel_i] = gray4bit_translation[b&0x0F];
								upscaled_gray[++pixel_i] = gray4bit_translation[(b&0xF0)>>4];
							}
							im->channels[IC_RED][y] = store_data( NULL, upscaled_gray, width, rgb_flags, 0);
						}else
							im->channels[IC_RED][y] = store_data( NULL, row_pointers[y], row_bytes, rgb_flags, 1);
					}
					
					if( grayscale ) 
					{	
						im->channels[IC_GREEN][y] = dup_data( NULL, im->channels[IC_RED][y] );
						im->channels[IC_BLUE][y]  = dup_data( NULL, im->channels[IC_RED][y] );
					}else
					{
						im->channels[IC_GREEN][y] = store_data( NULL, (CARD8*)buf.green, buf.width*4, rgb_flags, 0);	
						im->channels[IC_BLUE][y] = store_data( NULL, (CARD8*)buf.blue, buf.width*4, rgb_flags, 0);
					}	 

					if( do_alpha )
					{
						int has_zero = False, has_nozero = False ;
						register unsigned int i;
						for ( i = 0 ; i < buf.width ; ++i)
						{
							if( buf.alpha[i] != 0x00FF )
							{	
								if( buf.alpha[i] == 0 )
									has_zero = True ;
								else
								{	
									has_nozero = True ;
									break;
								}
							}		
						}
						if( has_zero || has_nozero ) 
						{
							ASFlagType alpha_flags = ASStorage_32Bit|ASStorage_RLEDiffCompress ;
							if( !has_nozero ) 
								set_flags( alpha_flags, ASStorage_Bitmap );
							im->channels[IC_ALPHA][y] = store_data( NULL, (CARD8*)buf.alpha, buf.width*4, alpha_flags, 0);
						}
					}
				}
				set_asstorage_block_size( NULL, old_storage_block_size );
				if (upscaled_gray)
					free(upscaled_gray);
				free (row_pointers);
				if( do_alpha || !grayscale ) 
					free_scanline(&buf, True);
				/* read rest of file, and get additional chunks in info_ptr - REQUIRED */
				png_read_end (png_ptr, info_ptr);
		  	}
		}
		/* clean up after the read, and free any memory allocated - REQUIRED */
		png_destroy_read_struct (&png_ptr, &info_ptr, (png_infopp) NULL);
		if (info_ptr)
			free (info_ptr);
	}

#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
print_asimage( im, ASFLAGS_EVERYTHING, __FUNCTION__, __LINE__ );
#endif
	SHOW_TIME("image loading",started);
	return im ;
}


/****** VO ******/
typedef struct ASImPNGReadBuffer
{
	CARD8 *buffer ; 
		 
} ASImPNGReadBuffer;

static void asim_png_read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
   ASImPNGReadBuffer *buf = (ASImPNGReadBuffer *)png_get_io_ptr(png_ptr);
   memcpy(data, buf->buffer, length);
   buf->buffer += length;
}

ASImage *
PNGBuff2ASimage(CARD8 *buffer, ASImageImportParams *params)
{
   static ASImage *im = NULL;
   ASImPNGReadBuffer buf;
   buf.buffer = buffer;
   im = png2ASImage_int((void*)&buf,(png_rw_ptr)asim_png_read_data, params);
   return im;
}


ASImage *
png2ASImage( const char * path, ASImageImportParams *params )
{
   FILE *fp ;
	static ASImage *im = NULL ;

	if ((fp = open_image_file(path)) == NULL)
		return NULL;

   im = png2ASImage_int((void*)fp, NULL, params);

	fclose(fp);
	return im;
}
#else 			/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
ASImage *
png2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - PNG image format is not supported.\n", path );
	return NULL ;
}

ASImage *
PNGBuff2ASimage(CARD8 *buffer, ASImageImportParams *params)
{
   return NULL;
}

#endif 			/* PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG PNG */
/***********************************************************************************/


/***********************************************************************************/
#ifdef HAVE_JPEG     /* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */
struct my_error_mgr
{
	struct jpeg_error_mgr pub;				   /* "public" fields */
	jmp_buf       setjmp_buffer;			   /* for return to caller */
};
typedef struct my_error_mgr *my_error_ptr;

METHODDEF (void)
my_error_exit (j_common_ptr cinfo)
{
	/* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
	my_error_ptr  myerr = (my_error_ptr) cinfo->err;
	/* Always display the message. */
	/* We could postpone this until after returning, if we chose. */
	(*cinfo->err->output_message) (cinfo);
	/* Return control to the setjmp point */
	longjmp (myerr->setjmp_buffer, 1);
}

ASImage *
jpeg2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im ;
	int old_storage_block_size ;
	/* This struct contains the JPEG decompression parameters and pointers to
	 * working space (which is allocated as needed by the JPEG library).
	 */
	struct jpeg_decompress_struct cinfo;
	void *temp_cinfo = NULL;
	/* We use our private extension JPEG error handler.
	 * Note that this struct must live as long as the main JPEG parameter
	 * struct, to avoid dangling-pointer problems.
	 */
	struct my_error_mgr jerr;
	/* More stuff */
	FILE         *infile;					   /* source file */
	JSAMPARRAY    buffer;					   /* Output row buffer */
	ASScanline    buf;
	int y;
	START_TIME(started);
 /*	register int i ;*/

	/* we want to open the input file before doing anything else,
	 * so that the setjmp() error recovery below can assume the file is open.
	 * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
	 * requires it in order to read binary files.
	 */

	if ((infile = open_image_file(path)) == NULL)
		return NULL;

	/* Step 1: allocate and initialize JPEG decompression object */
	/* We set up the normal JPEG error routines, then override error_exit. */
	cinfo.err = jpeg_std_error (&jerr.pub);
	jerr.pub.error_exit = my_error_exit;
	/* Establish the setjmp return context for my_error_exit to use. */
	if (setjmp (jerr.setjmp_buffer))
	{
		/* If we get here, the JPEG code has signaled an error.
		   * We need to clean up the JPEG object, close the input file, and return.
		 */
		jpeg_destroy_decompress (&cinfo);
		fclose (infile);
		return NULL;
	}
	/* Now we can initialize the JPEG decompression object. */
	jpeg_create_decompress (&cinfo);
	/* Step 2: specify data source (eg, a file) */
	jpeg_stdio_src (&cinfo, infile);
	/* Step 3: read file parameters with jpeg_read_header() */
	(void)jpeg_read_header (&cinfo, TRUE);
	/* We can ignore the return value from jpeg_read_header since
	 *   (a) suspension is not possible with the stdio data source, and
	 *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
	 * See libjpeg.doc for more info.
	 */

	/* Step 4: set parameters for decompression */
	/* Adjust default decompression parameters */
	cinfo.quantize_colors = FALSE;		       /* we don't want no stinking colormaps ! */
	cinfo.output_gamma = params->gamma;
	
	if( get_flags( params->flags, AS_IMPORT_SCALED_BOTH ) == AS_IMPORT_SCALED_BOTH )
	{
		int w = params->width ; 
		int h = params->height ;
		int ratio ; 

		if( w == 0 )
		{
			if( h == 0 ) 
			{
				w = cinfo.image_width ; 
				h = cinfo.image_height ; 
			}else
				w = (cinfo.image_width * h)/cinfo.image_height ;
		}else if( h == 0 )
			h = (cinfo.image_height * w)/cinfo.image_width ;
		
		ratio = cinfo.image_height/h ; 
		if( ratio > (int)cinfo.image_width/w )
			ratio = cinfo.image_width/w ; 
		
		cinfo.scale_num = 1 ; 
		/* only supported values are 1, 2, 4, and 8 */
		cinfo.scale_denom = 1 ; 
		if( ratio >= 2 ) 
		{
			if( ratio >= 4 ) 
			{
				if( ratio >= 8 ) 
					cinfo.scale_denom = 8 ; 
				else
					cinfo.scale_denom = 4 ; 
			}else
				cinfo.scale_denom = 2 ; 
		}
	}
	
	if( get_flags( params->flags, AS_IMPORT_FAST ) )
	{/* this does not really makes much of a difference */
		cinfo.do_fancy_upsampling = FALSE ; 
		cinfo.do_block_smoothing = FALSE ; 
		cinfo.dct_method = JDCT_IFAST ; 
	}
	
	/* Step 5: Start decompressor */
	(void)jpeg_start_decompress (&cinfo);
	LOCAL_DEBUG_OUT("stored image size %dx%d", cinfo.output_width,  cinfo.output_height);

	im = create_asimage( cinfo.output_width,  cinfo.output_height, params->compression );
	
	if( cinfo.output_components != 1 ) 
		prepare_scanline( im->width, 0, &buf, False );

	/* Make a one-row-high sample array that will go away when done with image */
	temp_cinfo = &cinfo;
	buffer = cinfo.mem->alloc_sarray((j_common_ptr) temp_cinfo, JPOOL_IMAGE,
									cinfo.output_width * cinfo.output_components, 1);

	/* Step 6: while (scan lines remain to be read) */
	SHOW_TIME("loading initialization",started);
	y = -1 ;
	/*cinfo.output_scanline*/
/*	for( i = 0 ; i < im->width ; i++ )	fprintf( stderr, "%3.3d    ", i );
	fprintf( stderr, "\n");
 */
	old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3/2 );

 	while ( ++y < (int)cinfo.output_height )
	{
		/* jpeg_read_scanlines expects an array of pointers to scanlines.
		 * Here the array is only one element long, but you could ask for
		 * more than one scanline at a time if that's more convenient.
		 */
		(void)jpeg_read_scanlines (&cinfo, buffer, 1);
		if( cinfo.output_components==1 ) 
		{	
			apply_gamma( (CARD8*)buffer[0], params->gamma_table, im->width );
			im->channels[IC_RED][y] = store_data( NULL, (CARD8*)buffer[0], im->width, ASStorage_RLEDiffCompress, 0);
			im->channels[IC_GREEN][y] = dup_data( NULL, im->channels[IC_RED][y] );
			im->channels[IC_BLUE][y]  = dup_data( NULL, im->channels[IC_RED][y] );
		}else
		{		   
			raw2scanline( (CARD8*)buffer[0], &buf, params->gamma_table, im->width, (cinfo.output_components==1), False);
			im->channels[IC_RED][y] = store_data( NULL, (CARD8*)buf.red, buf.width*4, ASStorage_32BitRLE, 0);
			im->channels[IC_GREEN][y] = store_data( NULL, (CARD8*)buf.green, buf.width*4, ASStorage_32BitRLE, 0);
			im->channels[IC_BLUE][y] = store_data( NULL, (CARD8*)buf.blue, buf.width*4, ASStorage_32BitRLE, 0);
		}
/*		fprintf( stderr, "src:");
		for( i = 0 ; i < im->width ; i++ )
			fprintf( stderr, "%2.2X%2.2X%2.2X ", buffer[0][i*3], buffer[0][i*3+1], buffer[0][i*3+2] );
		fprintf( stderr, "\ndst:");
		for( i = 0 ; i < im->width ; i++ )
			fprintf( stderr, "%2.2X%2.2X%2.2X ", buf.red[i], buf.green[i], buf.blue[i] );
		fprintf( stderr, "\n");
 */
	}
	set_asstorage_block_size( NULL, old_storage_block_size );
	if( cinfo.output_components != 1 ) 
		free_scanline(&buf, True);
	SHOW_TIME("read",started);

	/* Step 7: Finish decompression */
	/* we must abort the decompress if not all lines were read */
	if (cinfo.output_scanline < cinfo.output_height)
		jpeg_abort_decompress (&cinfo);
	else
		(void)jpeg_finish_decompress (&cinfo);
	/* We can ignore the return value since suspension is not possible
	 * with the stdio data source.
	 */
	/* Step 8: Release JPEG decompression object */
	/* This is an important step since it will release a good deal of memory. */
	jpeg_destroy_decompress (&cinfo);
	/* After finish_decompress, we can close the input file.
	 * Here we postpone it until after no more JPEG errors are possible,
	 * so as to simplify the setjmp error logic above.  (Actually, I don't
	 * think that jpeg_destroy can do an error exit, but why assume anything...)
	 */
	fclose (infile);
	/* At this point you may want to check to see whether any corrupt-data
	 * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
	 */
	SHOW_TIME("image loading",started);
	LOCAL_DEBUG_OUT("done loading JPEG image \"%s\"", path);
	return im ;
}
#else 			/* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */
ASImage *
jpeg2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - JPEG image format is not supported.\n", path );
	return NULL ;
}

#endif 			/* JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG JPEG */
/***********************************************************************************/

/***********************************************************************************/
/* XCF - GIMP's native file format : 											   */

ASImage *
xcf2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im = NULL ;
	/* More stuff */
	FILE         *infile;					   /* source file */
	XcfImage  *xcf_im;
	START_TIME(started);

	/* we want to open the input file before doing anything else,
	 * so that the setjmp() error recovery below can assume the file is open.
	 * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
	 * requires it in order to read binary files.
	 */
	if ((infile = open_image_file(path)) == NULL)
		return NULL;

	xcf_im = read_xcf_image( infile );
	fclose( infile );

	if( xcf_im == NULL )
		return NULL;

	LOCAL_DEBUG_OUT("stored image size %ldx%ld", xcf_im->width,  xcf_im->height);
#ifdef LOCAL_DEBUG
	print_xcf_image( xcf_im );
#endif
	{/* TODO : temporary workaround untill we implement layers merging */
		XcfLayer *layer = xcf_im->layers ;
		while ( layer )
		{
			if( layer->hierarchy )
				if( layer->hierarchy->image )
					if( layer->hierarchy->width == xcf_im->width &&
						layer->hierarchy->height == xcf_im->height )
					{
						im = layer->hierarchy->image ;
						layer->hierarchy->image = NULL ;
					}
			layer = layer->next ;
		}
	}
 	free_xcf_image(xcf_im);

	SHOW_TIME("image loading",started);
	return im ;
}

/***********************************************************************************/
/* PPM/PNM file format : 											   				   */
ASImage *
ppm2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im = NULL ;
	/* More stuff */
	FILE         *infile;					   /* source file */
	ASScanline    buf;
	int y;
	unsigned int type = 0, width = 0, height = 0, colors = 0;
#define PPM_BUFFER_SIZE 71                     /* Sun says that no line should be longer then this */
	char buffer[PPM_BUFFER_SIZE];
	START_TIME(started);

	if ((infile = open_image_file(path)) == NULL)
		return NULL;

	if( fgets( &(buffer[0]), PPM_BUFFER_SIZE, infile ) )
	{
		if( buffer[0] == 'P' )
			switch( buffer[1] )
			{    /* we only support RAWBITS formats : */
					case '5' : 	type= 5 ; break ;
					case '6' : 	type= 6 ; break ;
					case '8' : 	type= 8 ; break ;
				default:
					show_error( "invalid or unsupported PPM/PNM file format in image file \"%s\"", path );
			}
		if( type > 0 )
		{
			while ( fgets( &(buffer[0]), PPM_BUFFER_SIZE, infile ) )
			{
				if( buffer[0] != '#' )
				{
					register int i = 0;
					if( width > 0 )
					{
						colors = atoi(&(buffer[i]));
						break;
					}
					width = atoi( &(buffer[i]) );
					while ( buffer[i] != '\0' && !isspace((int)buffer[i]) ) ++i;
					while ( isspace((int)buffer[i]) ) ++i;
					if( buffer[i] != '\0')
						height = atoi(&(buffer[i]));
				}
			}
		}
	}

	if( type > 0 && colors <= 255 &&
		width > 0 && width < MAX_IMPORT_IMAGE_SIZE &&
		height > 0 && height < MAX_IMPORT_IMAGE_SIZE )
	{
		CARD8 *data ;
		size_t row_size = width * ((type==6)?3:((type==8)?4:1));

		data = safemalloc( row_size );

		LOCAL_DEBUG_OUT("stored image size %dx%d", width,  height);
		im = create_asimage( width,  height, params->compression );
		prepare_scanline( im->width, 0, &buf, False );
		y = -1 ;
		/*cinfo.output_scanline*/
		while ( ++y < (int)height )
		{
			if( fread( data, sizeof (char), row_size, infile ) < row_size )
				break;

			raw2scanline( data, &buf, params->gamma_table, im->width, (type==5), (type==8));

			asimage_add_line (im, IC_RED,   buf.red  , y);
			asimage_add_line (im, IC_GREEN, buf.green, y);
			asimage_add_line (im, IC_BLUE,  buf.blue , y);
			if( type == 8 )
				asimage_add_line (im, IC_ALPHA,   buf.alpha  , y);
		}
		free_scanline(&buf, True);
		free( data );
	}
	fclose( infile );
	SHOW_TIME("image loading",started);
	return im ;
}

/***********************************************************************************/
#ifdef HAVE_GIF		/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

int
gif_interlaced2y(int line /* 0 -- (height - 1) */, int height)
{
   	int passed_lines = 0;
   	int lines_in_current_pass;
   	/* pass 1 */
   	lines_in_current_pass = height / 8 + (height%8?1:0);
   	if (line < lines_in_current_pass) 
    	return line * 8;
   
   	passed_lines = lines_in_current_pass;
   	/* pass 2 */
   	if (height > 4) 
   	{
      	lines_in_current_pass = (height - 4) / 8 + ((height - 4)%8 ? 1 : 0);
      	if (line < lines_in_current_pass + passed_lines) 
         	return 4 + 8*(line - passed_lines);
      	passed_lines += lines_in_current_pass;
   	}
   	/* pass 3 */
   	if (height > 2) 
   	{
      	lines_in_current_pass = (height - 2) / 4 + ((height - 2)%4 ? 1 : 0);
      	if (line < lines_in_current_pass + passed_lines) 
        	return 2 + 4*(line - passed_lines);
    	passed_lines += lines_in_current_pass;
   	}
	return 1 + 2*(line - passed_lines);
}


ASImage *
gif2ASImage( const char * path, ASImageImportParams *params )
{
	FILE			   *fp ;
	int					status = GIF_ERROR;
	GifFileType        *gif;
	ASImage 	 	   *im = NULL ;
	int  		transparent = -1 ;
	unsigned int  		y;
	unsigned int		width = 0, height = 0;
	ColorMapObject     *cmap = NULL ;

	START_TIME(started);

	params->return_animation_delay = 0 ; 
	
	if ((fp = open_image_file(path)) == NULL)
		return NULL;
	if( (gif = open_gif_read(fp)) != NULL )
	{
		SavedImage	*sp = NULL ;
		int count = 0 ;
		
		status = get_gif_saved_images(gif, params->subimage, &sp, &count );
		if( status == GIF_OK && sp != NULL && count > 0 )
		{
			GifPixelType *row_pointer ;
#ifdef DEBUG_TRANSP_GIF
			fprintf( stderr, "Ext block = %p, count = %d\n", sp->ExtensionBlocks, sp->ExtensionBlockCount );
#endif
			if( sp->ExtensionBlocks )
				for ( y = 0; y < (unsigned int)sp->ExtensionBlockCount; y++)
				{
#ifdef DEBUG_TRANSP_GIF
					fprintf( stderr, "%d: func = %X, bytes[0] = 0x%X\n", y, sp->ExtensionBlocks[y].Function, sp->ExtensionBlocks[y].Bytes[0]);
#endif
					if( sp->ExtensionBlocks[y].Function == GRAPHICS_EXT_FUNC_CODE ) 
					{
						if( sp->ExtensionBlocks[y].Bytes[0]&0x01 )
						{
			   		 		transparent = ((unsigned int) sp->ExtensionBlocks[y].Bytes[GIF_GCE_TRANSPARENCY_BYTE])&0x00FF;
#ifdef DEBUG_TRANSP_GIF
							fprintf( stderr, "transp = %u\n", transparent );
#endif
						}
		   		 		params->return_animation_delay = (((unsigned int) sp->ExtensionBlocks[y].Bytes[GIF_GCE_DELAY_BYTE_LOW])&0x00FF) + 
												   		((((unsigned int) sp->ExtensionBlocks[y].Bytes[GIF_GCE_DELAY_BYTE_HIGH])<<8)&0x00FF00);
					}else if(  sp->ExtensionBlocks[y].Function == APPLICATION_EXT_FUNC_CODE && sp->ExtensionBlocks[y].ByteCount == 11 ) /* application extension */
					{
						if( strncmp(&(sp->ExtensionBlocks[y].Bytes[0]), "NETSCAPE2.0", 11 ) == 0 ) 
						{
							++y ;
							if( y < (unsigned int)sp->ExtensionBlockCount && sp->ExtensionBlocks[y].ByteCount == 3 )
							{
				   		 		params->return_animation_repeats = (((unsigned int) sp->ExtensionBlocks[y].Bytes[GIF_NETSCAPE_REPEAT_BYTE_LOW])&0x00FF) + 
														   		((((unsigned int) sp->ExtensionBlocks[y].Bytes[GIF_NETSCAPE_REPEAT_BYTE_HIGH])<<8)&0x00FF00);

#ifdef DEBUG_TRANSP_GIF
								fprintf( stderr, "animation_repeats = %d\n", params->return_animation_repeats );
#endif
							}
						}
					}
				}
			cmap = gif->SColorMap ;

			cmap = (sp->ImageDesc.ColorMap == NULL)?gif->SColorMap:sp->ImageDesc.ColorMap;
		    width = sp->ImageDesc.Width;
		    height = sp->ImageDesc.Height;

			if( cmap != NULL && (row_pointer = (unsigned char*)sp->RasterBits) != NULL &&
			    width < MAX_IMPORT_IMAGE_SIZE && height < MAX_IMPORT_IMAGE_SIZE )
			{
				int bg_color =   gif->SBackGroundColor ;
                int interlaced = sp->ImageDesc.Interlace;
                int image_y;
				CARD8 		 *r = NULL, *g = NULL, *b = NULL, *a = NULL ;
				int 	old_storage_block_size ;
				r = safemalloc( width );	   
				g = safemalloc( width );	   
				b = safemalloc( width );	   
				a = safemalloc( width );

				im = create_asimage( width, height, params->compression );
				old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3/2 );

				for (y = 0; y < height; ++y)
				{
					unsigned int x ;
					Bool do_alpha = False ;
                    image_y = interlaced ? gif_interlaced2y(y, height):y;
					for (x = 0; x < width; ++x)
					{
						int c = row_pointer[x];
      					if ( c == transparent)
						{
							c = bg_color ;
							do_alpha = True ;
							a[x] = 0 ;
						}else
							a[x] = 0x00FF ;
						
						r[x] = cmap->Colors[c].Red;
		        		g[x] = cmap->Colors[c].Green;
						b[x] = cmap->Colors[c].Blue;
	        		}
					row_pointer += x ;
					im->channels[IC_RED][image_y]  = store_data( NULL, r, width, ASStorage_RLEDiffCompress, 0);
				 	im->channels[IC_GREEN][image_y] = store_data( NULL, g, width, ASStorage_RLEDiffCompress, 0);	
					im->channels[IC_BLUE][image_y]  = store_data( NULL, b, width, ASStorage_RLEDiffCompress, 0);
					if( do_alpha )
						im->channels[IC_ALPHA][image_y]  = store_data( NULL, a, im->width, ASStorage_RLEDiffCompress|ASStorage_Bitmap, 0);
				}
				set_asstorage_block_size( NULL, old_storage_block_size );
				free(a);
				free(b);
				free(g);
				free(r);
			}
			free_gif_saved_images( sp, count );
		}else if( status != GIF_OK ) 
			ASIM_PrintGifError();
		else if( params->subimage == -1 )
			show_error( "Image file \"%s\" does not have any valid image information.", path );
		else
			show_error( "Image file \"%s\" does not have subimage %d.", path, params->subimage );

		DGifCloseFile(gif);
		fclose( fp );
	}
	SHOW_TIME("image loading",started);
	return im ;
}
#else 			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */
ASImage *
gif2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - missing GIF image format libraries.\n", path );
	return NULL ;
}
#endif			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

#ifdef HAVE_TIFF/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */


ASImage *
tiff2ASImage( const char * path, ASImageImportParams *params )
{
	TIFF 		 *tif ;

	static ASImage 	 *im = NULL ;
	CARD32 *data;
	int data_size;
	CARD32 width = 1, height = 1;
	CARD16 depth = 4 ;
	CARD16 bits = 0 ;
	CARD32 rows_per_strip =0 ;
	CARD32 tile_width = 0, tile_length = 0 ;
	CARD32 planar_config = 0 ;
	CARD16 photo = 0;
	START_TIME(started);

	if ((tif = TIFFOpen(path,"r")) == NULL)
	{
		show_error("cannot open image file \"%s\" for reading. Please check permissions.", path);
		return NULL;
	}

#ifdef DEBUG_TIFF
	{;}
#endif
	if( params->subimage > 0 )
		if( !TIFFSetDirectory(tif, params->subimage))
		{
			TIFFClose(tif);
			show_error("Image file \"%s\" does not contain subimage %d.", path, params->subimage);
			return NULL ;		
		}

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
	if( !TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &depth) )
		depth = 3 ;
	if( !TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits) )
		bits = 8 ;
	if( !TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip ) )
		rows_per_strip = height ;	
	if( !TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo) )
		photo = 0 ;
		
#ifndef PHOTOMETRIC_CFA
#define PHOTOMETRIC_CFA 32803		
#endif
		
	TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar_config);
	
	if( TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width) ||
		TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_length) )
	{
		show_error( "Tiled TIFF image format is not supported yet." );
		TIFFClose(tif);
		return NULL;   
	}		


	if( rows_per_strip == 0 || rows_per_strip > height ) 
		rows_per_strip = height ;
	if( depth <= 0 ) 
		depth = 4 ;
	if( depth <= 2 && get_flags( photo, PHOTOMETRIC_RGB) )
		depth += 2 ;
	LOCAL_DEBUG_OUT ("size = %ldx%ld, depth = %d, bits = %d, rps = %ld, photo = %d, tile_size = %dx%d, config = %d", 
					 width, height, depth, bits, rows_per_strip, photo, tile_width, tile_length, planar_config);
	if( width < MAX_IMPORT_IMAGE_SIZE && height < MAX_IMPORT_IMAGE_SIZE )
	{
		data_size = width*rows_per_strip*sizeof(CARD32);
		data = (CARD32*) _TIFFmalloc(data_size);
		if (data != NULL)
		{
			CARD8 		 *r = NULL, *g = NULL, *b = NULL, *a = NULL ;
			ASFlagType store_flags = ASStorage_RLEDiffCompress	;
			int first_row = 0 ;
			int old_storage_block_size;
			if( bits == 1 ) 
				set_flags( store_flags, ASStorage_Bitmap );
			
			im = create_asimage( width, height, params->compression );
			old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3/2 );
			
			if( depth == 2 || depth == 4 ) 
				a = safemalloc( width );
			r = safemalloc( width );	   
			if( depth > 2 ) 
			{
				g = safemalloc( width );	   
				b = safemalloc( width );	   
			}	 
			if (photo == PHOTOMETRIC_CFA)
			{/* need alternative - more complicated method */
				Bool success = False;

				ASIMStrip *strip = create_asim_strip(10, im->width, 8, True);
				ASImageOutput *imout = start_image_output( NULL, im, ASA_ASImage, 8, ASIMAGE_QUALITY_DEFAULT);

				LOCAL_DEBUG_OUT( "custom CFA TIFF reading...");

				if (strip && imout)
				{
					int cfa_type = 0;
					ASIMStripLoader line_loaders[2][2] = 
						{	{decode_RG_12_be, decode_GB_12_be},
	 						{decode_BG_12_be, decode_GR_12_be}
						};
					int line_loaders_num[2] = {2, 2};

					int bytes_per_row = (bits * width + 7)/8;
					int loaded_data_size = 0;

					if ( 1/* striped image */)
					{
						int strip_no;
						uint32* bc;
						TIFFGetField(tif, TIFFTAG_STRIPBYTECOUNTS, &bc);
						int all_strip_size = 0;
						for (strip_no = 0; strip_no < TIFFNumberOfStrips(tif); ++strip_no)
							all_strip_size += bc[strip_no];
						/* create one large buffer for the image data : */
						if (data_size < all_strip_size)
						{
							data_size = all_strip_size;
							_TIFFfree(data);
							data = _TIFFmalloc(data_size);
						}

						if (planar_config == PLANARCONFIG_CONTIG) 
						{
							for (strip_no = 0; strip_no < TIFFNumberOfStrips(tif); strip_no++)
							{
								int bytes_in;
								if (bits == 12) /* can't use libTIFF's function - it can't handle 12bit data ! */
								{
									/* PENTAX cameras claim that data is compressed as runlength packbits - 
									   it is not in fact run-length, which confuses libTIFF 
									 */
									bytes_in = TIFFReadRawStrip(tif, strip_no, data+loaded_data_size, data_size-loaded_data_size);
								}else
									bytes_in = TIFFReadEncodedStrip(tif, strip_no, data+loaded_data_size, data_size-loaded_data_size);

LOCAL_DEBUG_OUT( "strip size = %d, bytes_in = %d, bytes_per_row = %d", bc[strip_no], bytes_in, bytes_per_row);
								if (bytes_in >= 0)
									loaded_data_size += bytes_in;
								else 
								{
									LOCAL_DEBUG_OUT( "failed reading strip %d", strip_no);
								}
							}	
						} else if (planar_config == PLANARCONFIG_SEPARATE) 
						{
							/* TODO: do something with split channels */
						}
					}else
					{
						/* TODO: implement support for tiled images */
					}

					if (loaded_data_size > 0)
					{
						int offset;
						int data_row = 0;
						do
						{
							offset = data_row * bytes_per_row;
							int loaded_rows = load_asim_strip (strip, (CARD8*)data + offset, loaded_data_size-offset, 
																data_row, bytes_per_row, 
																line_loaders[cfa_type], line_loaders_num[cfa_type]);

							if (loaded_rows == 0)
							{ /* need to write out some rows to free up space */
								interpolate_asim_strip_custom_rggb2 (strip, SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE, False);
#if 0
								if (!get_flags (strip->lines[0]->flags, SCL_DO_RED))
								{
									int x;
									for (x = 0; x < width; ++x)
									{
										strip->lines[0]->red[x] = strip->lines[1]->red[x];
										strip->lines[1]->blue[x] = strip->lines[0]->blue[x];
									}
									set_flags (strip->lines[0]->flags, SCL_DO_RED);
									set_flags (strip->lines[1]->flags, SCL_DO_BLUE);
								}
#endif								
//clear_flags (strip->lines[0]->flags, SCL_DO_GREEN|SCL_DO_BLUE);
								imout->output_image_scanline( imout, strip->lines[0], 1);
								
								advance_asim_strip (strip);

							}	
							data_row += loaded_rows;
						}while (offset < loaded_data_size);
						success = True;
					}
				}
				destroy_asim_strip (&strip);
				stop_image_output( &imout );					

				if (!success)
					destroy_asimage (&im);
			}else
			{
				TIFFReadRGBAStrip(tif, first_row, (void*)data);
				do
				{
					register CARD32 *row = data ;
					int y = first_row + rows_per_strip ;
					if( y > height ) 
						y = height ;
					while( --y >= first_row )
					{
						int x ;
						for( x = 0 ; x < width ; ++x )
						{
							CARD32 c = row[x] ;
							if( depth == 4 || depth == 2 ) 
								a[x] = TIFFGetA(c);
							r[x]   = TIFFGetR(c);
							if( depth > 2 ) 
							{
								g[x] = TIFFGetG(c);
								b[x]  = TIFFGetB(c);
							}
						}
						im->channels[IC_RED][y]  = store_data( NULL, r, width, store_flags, 0);
						if( depth > 2 ) 
						{
					 		im->channels[IC_GREEN][y] = store_data( NULL, g, width, store_flags, 0);	
							im->channels[IC_BLUE][y]  = store_data( NULL, b, width, store_flags, 0);
						}else
						{
					 		im->channels[IC_GREEN][y] = dup_data( NULL, im->channels[IC_RED][y]);	  
							im->channels[IC_BLUE][y]  = dup_data( NULL, im->channels[IC_RED][y]);
						}		 

						if( depth == 4 || depth == 2 ) 
							im->channels[IC_ALPHA][y]  = store_data( NULL, a, width, store_flags, 0);
						row += width ;
					}
					/* move onto the next strip now : */
					do
					{
						first_row += rows_per_strip ;
					}while (first_row < height && !TIFFReadRGBAStrip(tif, first_row, (void*)data));

				}while (first_row < height);
		    }
			set_asstorage_block_size( NULL, old_storage_block_size );

			if( b ) free( b );
			if( g ) free( g );
			if( r ) free( r );
			if( a ) free( a );
			_TIFFfree(data);
		}
	}
	/* close the file */
	TIFFClose(tif);
	SHOW_TIME("image loading",started);

	return im ;
}
#else 			/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */

ASImage *
tiff2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - missing TIFF image format libraries.\n", path );
	return NULL ;
}
#endif			/* TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF TIFF */


static ASImage *
load_xml2ASImage( ASImageManager *imman, const char *path, unsigned int compression, int width, int height )
{
	ASVisual fake_asv ;
	char *slash, *curr_path = NULL ;
	char *doc_str = NULL ;
	ASImage *im = NULL ;

	memset( &fake_asv, 0x00, sizeof(ASVisual) );
	if( (slash = strrchr( path, '/' )) != NULL )
		curr_path = mystrndup( path, slash-path );

	if((doc_str = load_file(path)) == NULL )
		show_error( "unable to load file \"%s\" file is either too big or is not readable.\n", path );
	else
	{
		im = compose_asimage_xml_at_size(&fake_asv, imman, NULL, doc_str, 0, 0, None, curr_path, width, height);
		free( doc_str );
	}

	if( curr_path )
		free( curr_path );
	return im ;
}


ASImage *
xml2ASImage( const char *path, ASImageImportParams *params )
{
	int width = -1, height = -1 ; 
	static ASImage 	 *im = NULL ;
	START_TIME(started);

 	if( get_flags( params->flags, AS_IMPORT_SCALED_H ) )
		width = (params->width <= 0)?((params->height<=0)?-1:params->height):params->width ;
	
 	if( get_flags( params->flags, AS_IMPORT_SCALED_V ) )
		height = (params->height <= 0)?((params->width <= 0)?-1:params->width):params->height ;
		
	im = load_xml2ASImage( NULL, path, params->compression, width, height );

	SHOW_TIME("image loading",started);
	return im ;
}

#ifdef HAVE_SVG/* SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG */
ASImage *
svg2ASImage( const char * path, ASImageImportParams *params )
{
   	static int gType_inited = 0;
   
   	ASImage *im = NULL;
   	GdkPixbuf *pixbuf;
	int channels ;
	Bool do_alpha ; 
	int width = -1, height = -1 ; 
 
	START_TIME(started);
#if 1
	/* Damn gtk mess... must init once atleast.. can we just init
	   several times or do we bork then? */
	if (gType_inited == 0) 
	{
	   g_type_init();
	   gType_inited = 1;
	}
 
 	if( get_flags( params->flags, AS_IMPORT_SCALED_H ) )
		width = (params->width <= 0)?((params->height<=0)?-1:params->height):params->width ;
	
 	if( get_flags( params->flags, AS_IMPORT_SCALED_V ) )
		height = (params->height <= 0)?((params->width <= 0)?-1:params->width):params->height ;
		
	if( (pixbuf = rsvg_pixbuf_from_file_at_size( path, width, height, NULL)) == NULL )
		return NULL ;
	
	channels = gdk_pixbuf_get_n_channels(pixbuf) ;
	do_alpha = gdk_pixbuf_get_has_alpha(pixbuf) ;
	if ( ((channels == 4 && do_alpha) ||(channels == 3 && !do_alpha)) &&
		gdk_pixbuf_get_bits_per_sample(pixbuf) == 8 ) 
	{
	   	int width, height;
		register CARD8 *row = gdk_pixbuf_get_pixels(pixbuf);
		int y;
		CARD8 		 *r = NULL, *g = NULL, *b = NULL, *a = NULL ;
		int old_storage_block_size;

		width = gdk_pixbuf_get_width(pixbuf);
		height = gdk_pixbuf_get_height(pixbuf);

		r = safemalloc( width );	   
		g = safemalloc( width );	   
		b = safemalloc( width );	   
		if( do_alpha )
			a = safemalloc( width );


		im = create_asimage(width, height, params->compression );
		old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3/2 );
		for (y = 0; y < height; ++y) 
		{
			int x, i = 0 ;
			for( x = 0 ; x < width ; ++x )
			{
				r[x] = row[i++];
				g[x] = row[i++];
				b[x] = row[i++];
				if( do_alpha ) 
					a[x] = row[i++];
			}
			im->channels[IC_RED][y]  = store_data( NULL, r, width, ASStorage_RLEDiffCompress, 0);
		 	im->channels[IC_GREEN][y] = store_data( NULL, g, width, ASStorage_RLEDiffCompress, 0);	
			im->channels[IC_BLUE][y]  = store_data( NULL, b, width, ASStorage_RLEDiffCompress, 0);

			if( do_alpha )
				for( x = 0 ; x < width ; ++x )
					if( a[x] != 0x00FF )
					{
						im->channels[IC_ALPHA][y]  = store_data( NULL, a, width, ASStorage_RLEDiffCompress, 0);
						break;
					}
			row += channels*width ;
		}
		set_asstorage_block_size( NULL, old_storage_block_size );
		free(r);
		free(g);
		free(b);
		if( a )
			free(a);
	}
	
	if (pixbuf)
		gdk_pixbuf_unref(pixbuf);
#endif	
	SHOW_TIME("image loading",started);

	return im ;
}
#else 			/* SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG */

ASImage *
svg2ASImage( const char * path, ASImageImportParams *params )
{
	show_error( "unable to load file \"%s\" - missing SVG image format libraries.\n", path );
	return NULL ;
}
#endif			/* SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG SVG */


/*************************************************************************/
/* Targa Image format - some stuff borrowed from the GIMP.
 *************************************************************************/
typedef struct ASTGAHeader
{
	CARD8 IDLength ;
	CARD8 ColorMapType;
#define TGA_NoImageData			0
#define TGA_ColormappedImage	1
#define TGA_TrueColorImage		2
#define TGA_BWImage				3
#define TGA_RLEColormappedImage		9
#define TGA_RLETrueColorImage		10
#define TGA_RLEBWImage				11
	CARD8 ImageType;
	struct 
	{
		CARD16 FirstEntryIndex ;
		CARD16 ColorMapLength ;  /* number of entries */ 
		CARD8  ColorMapEntrySize ;  /* number of bits per entry */ 
	}ColormapSpec;
	struct
	{		
		CARD16 XOrigin;
		CARD16 YOrigin;
		CARD16 Width;
		CARD16 Height;
		CARD8  Depth;
#define TGA_LeftToRight		(0x01<<4)
#define TGA_TopToBottom		(0x01<<5)
		CARD8  Descriptor;
	}ImageSpec;

}ASTGAHeader;

typedef struct ASTGAColorMap
{
	int bytes_per_entry;
	int bytes_total ; 
	CARD8 *data ; 
}ASTGAColorMap;

typedef struct ASTGAImageData
{
	int bytes_per_pixel;
	int image_size;
	int bytes_total ; 
	CARD8 *data ; 
}ASTGAImageData;

static Bool load_tga_colormapped(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
		
	return True;
}

static Bool load_tga_truecolor(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
	CARD32 *a = buf->alpha ;
	CARD32 *r = buf->red ;
	CARD32 *g = buf->green ;
	CARD32 *b = buf->blue ;
	int bpp = (tga->ImageSpec.Depth+7)/8;
	int bpl = buf->width*bpp;
	if( fread( read_buf, 1, bpl, infile ) != (unsigned int)bpl ) 		   
		return False;
	if( bpp == 3 ) 
	{	
		unsigned int i;
		if( gamma_table )
			for( i = 0 ; i < buf->width ; ++i ) 
			{
				b[i] = gamma_table[*(read_buf++)];	
				g[i] = gamma_table[*(read_buf++)];	  
				r[i] = gamma_table[*(read_buf++)];	  
			}	 
		else
			for( i = 0 ; i < buf->width ; ++i ) 
			{
				b[i] = *(read_buf++);	
				g[i] = *(read_buf++);	  
				r[i] = *(read_buf++);	  
			}	 
		set_flags( buf->flags, SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE );
	}else if( bpp == 4 )
	{
		unsigned int i;
		for( i = 0 ; i < buf->width ; ++i ) 
		{
			b[i] = *(read_buf++);	
			g[i] = *(read_buf++);	  
			r[i] = *(read_buf++);	  
			a[i] = *(read_buf++);	  
		}	 
		set_flags( buf->flags, SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE|SCL_DO_ALPHA );
	}	 

	return True;
}

static Bool load_tga_bw(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
		
	return True;
}

static Bool load_tga_rle_colormapped(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
		
	return True;
}

static Bool load_tga_rle_truecolor(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
		
	return True;
}

static Bool load_tga_rle_bw(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table )
{
		
	return True;
}



ASImage *
tga2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im = NULL ;
	/* More stuff */
	FILE         *infile;					   /* source file */
	ASTGAHeader   tga;
	ASTGAColorMap *cmap = NULL ;
	int width = 1, height = 1;
	START_TIME(started);


	if ((infile = open_image_file(path)) == NULL)
		return NULL;
	if( fread( &tga, 1, 3, infile ) == 3 ) 
	if( fread( &tga.ColormapSpec, 1, 5, infile ) == 5 ) 
	if( fread( &tga.ImageSpec, 1, 10, infile ) == 10 ) 
	{
		Bool success = True ;
		Bool (*load_row_func)(FILE *infile, ASTGAHeader *tga, ASTGAColorMap *cmap, ASScanline *buf, CARD8 *read_buf, CARD8 *gamma_table );

		if( tga.IDLength > 0 ) 
			success = (fseek( infile, tga.IDLength, SEEK_CUR )==0);
		if( success && tga.ColorMapType != 0 ) 
		{
			cmap = safecalloc( 1, sizeof(ASTGAColorMap));
			cmap->bytes_per_entry = (tga.ColormapSpec.ColorMapEntrySize+7)/8;
			cmap->bytes_total = cmap->bytes_per_entry*tga.ColormapSpec.ColorMapLength; 
			cmap->data = safemalloc( cmap->bytes_total);
			success = ( fread( cmap->data, 1, cmap->bytes_total, infile ) == (unsigned int)cmap->bytes_total );
		}else if( tga.ImageSpec.Depth != 24 && tga.ImageSpec.Depth != 32 )
			success = False ;
	 
		if( success ) 
		{
			success = False;
			if( tga.ImageType != TGA_NoImageData )
			{	
				width = tga.ImageSpec.Width ; 
				height = tga.ImageSpec.Height ; 
				if( width < MAX_IMPORT_IMAGE_SIZE && height < MAX_IMPORT_IMAGE_SIZE )
					success = True;
			}
		}
		switch( tga.ImageType ) 
		{
			case TGA_ColormappedImage	:load_row_func = load_tga_colormapped ; break ;
			case TGA_TrueColorImage		:load_row_func = load_tga_truecolor ; break ;
			case TGA_BWImage			:load_row_func = load_tga_bw ; break ;
			case TGA_RLEColormappedImage:load_row_func = load_tga_rle_colormapped ; break ;
			case TGA_RLETrueColorImage	:load_row_func = load_tga_rle_truecolor ; break ;
			case TGA_RLEBWImage			:load_row_func = load_tga_rle_bw ; break ;
			default:
				load_row_func = NULL ;
		}	 
		
		if( success && load_row_func != NULL ) 
		{	
			ASImageOutput  *imout ;
			int old_storage_block_size;
			im = create_asimage( width, height, params->compression );
			old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3/2 );

			if((imout = start_image_output( NULL, im, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT)) == NULL )
			{
        		destroy_asimage( &im );
				success = False;
			}else
			{	
				ASScanline    buf;
				int y ;
				CARD8 *read_buf = safemalloc( width*4*2 ); 
				prepare_scanline( im->width, 0, &buf, True );
				if( !get_flags( tga.ImageSpec.Descriptor, TGA_TopToBottom ) )			
					toggle_image_output_direction( imout );
				for( y = 0 ; y < height ; ++y ) 
				{	
					if( !load_row_func( infile, &tga, cmap, &buf, read_buf, params->gamma_table ) )
						break;
					imout->output_image_scanline( imout, &buf, 1);
				}
				stop_image_output( &imout );
				free_scanline( &buf, True );
				free( read_buf );
			}   
			set_asstorage_block_size( NULL, old_storage_block_size );

		}	  
	}	 
	if( im == NULL )
		show_error( "invalid or unsupported TGA format in image file \"%s\"", path );

	if (cmap) free (cmap);
	fclose( infile );
	SHOW_TIME("image loading",started);
	return im ;
}
/*************************************************************************/
/* ARGB 																 */
/*************************************************************************/
ASImage *
convert_argb2ASImage( ASVisual *asv, int width, int height, ARGB32 *argb, CARD8 *gamma_table )
{
	ASImage *im = NULL ;
	ASImageOutput  *imout ;
	im = create_asimage( width, height, 100 );
	if((imout = start_image_output( NULL, im, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT)) == NULL )
	{
   		destroy_asimage( &im );
		return NULL;
	}else
	{	
		ASScanline    buf;
		int y ;
		int old_storage_block_size = set_asstorage_block_size( NULL, im->width*im->height*3 );

		prepare_scanline( im->width, 0, &buf, True );
		for( y = 0 ; y < height ; ++y ) 
		{	  
			int x ;
			for( x = 0 ; x < width ; ++x ) 
			{
				ARGB32 c = argb[x];
				buf.alpha[x] 	= ARGB32_ALPHA8(c);	
				buf.red[x] 	= ARGB32_RED8(c);	  
				buf.green[x] 	= ARGB32_GREEN8(c);	  
				buf.blue[x] 	= ARGB32_BLUE8(c);	  
			}	 
			argb += width ;			
			set_flags( buf.flags, SCL_DO_RED|SCL_DO_GREEN|SCL_DO_BLUE|SCL_DO_ALPHA );
			imout->output_image_scanline( imout, &buf, 1);
		}
		set_asstorage_block_size( NULL, old_storage_block_size );
		stop_image_output( &imout );
		free_scanline( &buf, True );
	}   
						
	return im ;	
}


ASImage *
argb2ASImage( const char *path, ASImageImportParams *params )
{
	ASVisual fake_asv ;
	long argb_data_len = -1; 
	char *argb_data = NULL ;
	ASImage *im = NULL ;

	memset( &fake_asv, 0x00, sizeof(ASVisual) );

	argb_data = load_binary_file(path, &argb_data_len);
	if(argb_data == NULL || argb_data_len < 8 )
		show_error( "unable to load file \"%s\" file is either too big or is not readable.\n", path );
	else
	{
		int width = ((CARD32*)argb_data)[0] ;
		int height = ((CARD32*)argb_data)[1] ;
		if( 2 + width*height > (int)(argb_data_len/sizeof(CARD32)))
		{
			show_error( "file \"%s\" is too small for specified image size of %dx%d.\n", path, width, height );
		}else
			im = convert_argb2ASImage( &fake_asv, width, height, (ARGB32*)argb_data+2, params->gamma_table );
	}
	if( argb_data ) 
		free( argb_data );
	
	return im ;
}

