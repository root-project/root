/* This file contains code for unified image loading from many
 * uncompressed  GIFs */
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

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

/*#define LOCAL_DEBUG */
/*#define DO_CLOCKING */

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

#include "asimage.h"
#include "ascmap.h"
#include "ungif.h"

#ifdef HAVE_GIF		/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

void
free_gif_saved_image( SavedImage *sp, Bool reusable )
{
	if( sp )
	{
		if (sp->ImageDesc.ColorMap)
#if (GIFLIB_MAJOR>=5)
	    	GifFreeMapObject(sp->ImageDesc.ColorMap);
#else
	    	FreeMapObject(sp->ImageDesc.ColorMap);
#endif

		if (sp->RasterBits)
		    free((char *)sp->RasterBits);

		if (sp->ExtensionBlocks)
#if (GIFLIB_MAJOR>=5)
		    GifFreeExtensions(&sp->ExtensionBlockCount, &sp->ExtensionBlocks);
#else
		    FreeExtension(sp);
#endif

		if( !reusable )
			free( sp );
	}
}

void
free_gif_saved_images( SavedImage *images, int count )
{
	if( images )
	{
		while ( --count >= 0 )
			free_gif_saved_image( &(images[count]), True );
		free( images );
	}
}

static void
append_gif_saved_image( SavedImage *src, SavedImage **ret, int *ret_images )
{
	*ret = realloc( *ret, sizeof( SavedImage )*((*ret_images)+1));
	memcpy( &((*ret)[*ret_images]), src, sizeof(SavedImage) );
	memset( src, 0x00, sizeof( SavedImage ) );
	++(*ret_images) ;
}

/* It appears that nobody has bloody debugged giflib with multiimage files where
 * subimages have private colormaps. So it bloody fucks everything up.
 * So we bloody have to cludge it to get something usefull.
 * Oh The pain! The pain!
 */
int fread_gif( GifFileType *gif, GifByteType* buf, int len )
{
	int ret = fread( buf, 1, len, gif->UserData );
	return ret;
}

#if (GIFLIB_MAJOR>=5)
GifFileType*
open_gif_read( FILE *in_stream, int *errcode )
{
	return DGifOpen(in_stream, fread_gif, errcode);
}
#else
GifFileType*
open_gif_read( FILE *in_stream )
{
	return DGifOpen(in_stream, fread_gif);
}
#endif

int
get_gif_image_desc( GifFileType *gif, SavedImage *im )
{
	long start_pos, end_pos ;
	int status;

	start_pos = ftell(gif->UserData);
	status = DGifGetImageDesc( gif );
	end_pos = ftell(gif->UserData);
	if( status == GIF_OK )
	{
		int ext_count = im->ExtensionBlockCount ;
		ExtensionBlock	*ext_ptr = im->ExtensionBlocks ;

		im->ExtensionBlocks = NULL ;
		im->ExtensionBlockCount = 0 ;

		free_gif_saved_image( im, True );
		memset( im, 0x00, sizeof(SavedImage));

		im->ExtensionBlocks = ext_ptr ;
		im->ExtensionBlockCount = ext_count ;

		memcpy( &(im->ImageDesc), &(gif->Image), sizeof(GifImageDesc));
		if( gif->Image.ColorMap )
		{
#if (GIFLIB_MAJOR>=5)
			im->ImageDesc.ColorMap = GifMakeMapObject(gif->Image.ColorMap->ColorCount, NULL);
#else
			im->ImageDesc.ColorMap = MakeMapObject(gif->Image.ColorMap->ColorCount, NULL);
#endif
			fseek( gif->UserData, start_pos+9, SEEK_SET );
			if(fread( im->ImageDesc.ColorMap->Colors, 1, gif->Image.ColorMap->ColorCount*3, gif->UserData)){;};
			fseek( gif->UserData, end_pos, SEEK_SET );
			gif->Image.ColorMap = NULL ;
 		}
	}
	return status;
}

int
get_gif_saved_images( GifFileType *gif, int subimage, SavedImage **ret, int *ret_images  )
{
    GifRecordType RecordType;
    GifByteType *ExtData;
#if (GIFLIB_MAJOR>=5)
    int ExtCode;
    size_t Len;
#endif
    SavedImage temp_save;
	int curr_image = 0, ret_count = *ret_images ;
	int status = GIF_OK;

	memset( &temp_save, 0x00, sizeof( temp_save ) );
	do
	{
		if ( (status = DGifGetRecordType(gif, &RecordType)) == GIF_ERROR)
		{
			break;
		}
		switch (RecordType)
		{
	    	case IMAGE_DESC_RECORD_TYPE:
				if ((status = get_gif_image_desc(gif, &temp_save)) == GIF_OK)
				{
					int size = temp_save.ImageDesc.Width*temp_save.ImageDesc.Height ;
					temp_save.RasterBits = realloc( temp_save.RasterBits, size );
					status = DGifGetLine(gif, (unsigned char*)temp_save.RasterBits, size);
					if (status == GIF_OK)
					{
						if( curr_image == subimage || subimage < 0 )
						{
							append_gif_saved_image( &temp_save, ret, &(ret_count));
						}
					}
					++curr_image ;
				}
				break;

	    	case EXTENSION_RECORD_TYPE:
#if (GIFLIB_MAJOR>=5)
				status = DGifGetExtension(gif,&ExtCode,&ExtData);
#else
				status = DGifGetExtension(gif,&temp_save.Function,&ExtData);
#endif
				while (ExtData != NULL && status == GIF_OK )
				{
            		/* Create an extension block with our data */
#if (GIFLIB_MAJOR>=5)
				      Len = EGifGCBToExtension(gif, ExtData);
            		if ((status = GifAddExtensionBlock(&temp_save.ExtensionBlockCount, &temp_save.ExtensionBlocks,
                            ExtCode, Len, ExtData)) == GIF_OK)
                    status = DGifGetExtension(gif,&ExtCode,&ExtData);
#else
            		if ((status = AddExtensionBlock(&temp_save, ExtData[0], (char*)&(ExtData[1]))) == GIF_OK)
				    	status = DGifGetExtensionNext(gif, &ExtData);
            		temp_save.Function = 0;
#endif
				}
				break;

	    	case TERMINATE_RECORD_TYPE:
				break;

	    	default:	/* Should be trapped by DGifGetRecordType */
				break;
		}
    }while( status == GIF_OK && RecordType != TERMINATE_RECORD_TYPE);

/*	if( status == GIF_OK && *ret == NULL )
		append_gif_saved_image( &temp_save, ret, &(ret_count));
	else
*/
	free_gif_saved_image( &temp_save, True );

	*ret_images = ret_count ;
    return status;
}

int
write_gif_saved_images( GifFileType *gif, SavedImage *images, unsigned int count )
{
	int status = GIF_OK;
	unsigned int i ;

	for( i = 0 ; i < count && status == GIF_OK; ++i )
	{
		register SavedImage	*sp = &images[i];
		int		SavedHeight = sp->ImageDesc.Height;
		int		SavedWidth = sp->ImageDesc.Width;
		ExtensionBlock	*ep;
		int y ;

#if 1
		if (sp->ExtensionBlocks)
        	for ( y = 0; y < sp->ExtensionBlockCount && status == GIF_OK; y++)
			{
            	ep = &sp->ExtensionBlocks[y];
            	status = EGifPutExtension(gif, (ep->Function != 0) ? ep->Function : '\0',
               							  ep->ByteCount, ep->Bytes);
			}
#endif
		if( status == GIF_OK )
		{
			status = EGifPutImageDesc(gif, sp->ImageDesc.Left, sp->ImageDesc.Top,
									  SavedWidth, SavedHeight, sp->ImageDesc.Interlace,
									  sp->ImageDesc.ColorMap );
			for (y = 0; y < SavedHeight && status == GIF_OK ; y++)
	    		status = EGifPutLine(gif, (unsigned char*)sp->RasterBits + y * SavedWidth, SavedWidth);
		}
	}
	return status;
}

#endif			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

