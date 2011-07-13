/* This file contains code for unified image loading from many file formats */
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
#undef DO_CLOCKING
#undef DEBUG_TRANSP_GIF

#ifdef _WIN32
# include "win32/config.h"
# include <windows.h>
# include "win32/afterbase.h"
#else
# include "config.h"
# include <string.h>
# include "afterbase.h"
#endif

#include "asimage.h"
#include "imencdec.h"
#include "import.h"
#include "export.h"
#include "bmp.h"

/* from import.c : */
FILE* open_image_file( const char *path );
/* from export.c : */
FILE* open_writeable_image_file( const char *path );

void 
dib_data_to_scanline( ASScanline *buf, 
                      BITMAPINFOHEADER *bmp_info, CARD8 *gamma_table, 
					  CARD8 *data, CARD8 *cmap, int cmap_entry_size) 
{	
	int x ; 
	switch( bmp_info->biBitCount )
	{
		case 1 :
			for( x = 0 ; x < bmp_info->biWidth ; x++ )
			{
				int entry = (data[x>>3]&(1<<(x&0x07)))?cmap_entry_size:0 ;
				buf->red[x] = cmap[entry+2];
				buf->green[x] = cmap[entry+1];
				buf->blue[x] = cmap[entry];
			}
			break ;
		case 4 :
			for( x = 0 ; x < (int)bmp_info->biWidth ; x++ )
			{
				int entry = data[x>>1];
				if(x&0x01)
					entry = ((entry>>4)&0x0F)*cmap_entry_size ;
				else
					entry = (entry&0x0F)*cmap_entry_size ;
				buf->red[x] = cmap[entry+2];
				buf->green[x] = cmap[entry+1];
				buf->blue[x] = cmap[entry];
			}
			break ;
		case 8 :
			for( x = 0 ; x < (int)bmp_info->biWidth ; x++ )
			{
				int entry = data[x]*cmap_entry_size ;
				buf->red[x] = cmap[entry+2];
				buf->green[x] = cmap[entry+1];
				buf->blue[x] = cmap[entry];
			}
			break ;
		case 16 :
			for( x = 0 ; x < (int)bmp_info->biWidth ; ++x )
			{
				CARD8 c1 = data[x] ;
				CARD8 c2 = data[++x];
				buf->blue[x] =    c1&0x1F;
				buf->green[x] = ((c1>>5)&0x07)|((c2<<3)&0x18);
				buf->red[x] =   ((c2>>2)&0x1F);
			}
			break ;
		default:
			raw2scanline( data, buf, gamma_table, buf->width, False, (bmp_info->biBitCount==32));
	}
}

BITMAPINFO *
ASImage2DIB( ASVisual *asv, ASImage *im, 
		     int offset_x, int offset_y,
			 unsigned int to_width,
			 unsigned int to_height,
  			 void **pBits, int mask )
{
	BITMAPINFO *bmp_info = NULL;
	CARD8 *bits = NULL, *curr ;
	int line_size, pad ; 
	ASImageDecoder *imdec ;
	int y, max_y = to_height;	
	int tiling_step = 0 ;
	CARD32 *a = 0;
   CARD32 *r = 0;
   CARD32 *g = 0;
   CARD32 *b = 0;	
	START_TIME(started);

LOCAL_DEBUG_CALLER_OUT( "src = %p, offset_x = %d, offset_y = %d, to_width = %d, to_height = %d", im, offset_x, offset_y, to_width, to_height );
	if( im== NULL || (imdec = start_image_decoding(asv, im, mask ? SCL_DO_ALPHA : SCL_DO_ALL, offset_x, offset_y, to_width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		return NULL;
	}
	
	if( to_height > im->height )
	{
		tiling_step = im->height ;
		max_y = im->height ;
	}
	/* create bmp_info struct */
	bmp_info = (BITMAPINFO *)safecalloc( 1, sizeof(BITMAPINFO) );
	bmp_info->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmp_info->bmiHeader.biWidth = to_width ;
	bmp_info->bmiHeader.biHeight = to_height ;
	bmp_info->bmiHeader.biPlanes = 1 ;
	bmp_info->bmiHeader.biBitCount = mask ? 1 : 24 ;
	bmp_info->bmiHeader.biCompression = BI_RGB ;
	bmp_info->bmiHeader.biSizeImage = 0 ;
	bmp_info->bmiHeader.biClrUsed = 0 ;
	bmp_info->bmiHeader.biClrImportant = 0 ;
	/* allocate DIB bits : */
	line_size = mask ? to_width : ((to_width*3+3)/4)*4;          /* DWORD aligned */
	pad = line_size-(to_width*(mask ? 1 : 3)) ;
	bits = (CARD8 *)safemalloc(line_size * to_height);
	curr = bits + line_size * to_height ;

   if (mask) {
	   a = imdec->buffer.alpha ;
   } else {
	   r = imdec->buffer.red ;
	   g = imdec->buffer.green ;
	   b = imdec->buffer.blue ;
   }

	for( y = 0 ; y < max_y ; y++  )
	{
		register int x = to_width;
		imdec->decode_image_scanline( imdec );
		/* convert to DIB bits : */
		curr -= pad ;
		while( --x >= 0 ) 
		{
			curr -= (mask ? 1 : 3) ; 
         if (mask) {
            curr[0] = a[x]==0 ? 0 : 1 ;
         } else {
			   curr[0] = b[x] ; 	
			   curr[1] = g[x] ; 
			   curr[2] = r[x] ;
         }
		}	 
		if( tiling_step > 0 ) 
		{
			CARD8 *tile ;
			int offset = tiling_step ; 
			while( y + offset < (int)to_height ) 
			{	 	 
				tile = curr - offset*line_size ; 
				memcpy( tile, curr, line_size );
				offset += tiling_step ;
			}
		}	 
	}
	
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	*pBits = bits ;
	return bmp_info;
}

/* stupid typo !!!!! and now we are stuck with it :( */
#undef ASImage2DBI
BITMAPINFO *
ASImage2DBI( ASVisual *asv, ASImage *im, 
		     int offset_x, int offset_y,
			 unsigned int to_width,
			 unsigned int to_height,
  			 void **pBits, int mask )
{
	return ASImage2DIB(asv, im, offset_x, offset_y, to_width, to_height, pBits, mask );
}



ASImage *
DIB2ASImage(BITMAPINFO *bmp_info, int compression)
{
  int width = bmp_info->bmiHeader.biWidth;
  int height = bmp_info->bmiHeader.biHeight;
  ASImage *im = NULL;
  ASScanline buf;
  int y;
	CARD8 *data ;
	CARD8 *cmap = NULL ;
	int direction = -1 ;
	int cmap_entries = 0, cmap_entry_size = 4, row_size ;

  if (width <= 0 || height == 0 )
    return NULL;

	if( height < 0 )
    {
		  direction = 1 ;
      height = -height;
    }

	if( bmp_info->bmiHeader.biBitCount < 16 )
		cmap_entries = 0x01<<bmp_info->bmiHeader.biBitCount ;

	if( bmp_info->bmiHeader.biSize != 40 )
		cmap_entry_size = 3;

	if( cmap_entries )
    {
		  cmap = (CARD8*)&(bmp_info->bmiColors[0]);
    	data = cmap + cmap_entries*cmap_entry_size;
    }
  else
    data = (CARD8*)&(bmp_info->bmiColors[0]);

	row_size = (width*bmp_info->bmiHeader.biBitCount)>>3 ;
	if( row_size == 0 )
		row_size = 1 ;
	else
		row_size = (row_size+3)/4 ;            /* everything is aligned by 32 bits */
	row_size *= 4 ;                            /* in bytes  */

  im = create_asimage(width, height, compression);

	/* Window BMP files are little endian  - we need to swap Red and Blue */
	prepare_scanline( width, 0, &buf, True );

	y =( direction == 1 )?0:height-1 ;
	while( y >= 0 && y < (int)height)
	{
 		dib_data_to_scanline(&buf, &(bmp_info->bmiHeader), NULL, data, cmap, cmap_entry_size);
		asimage_add_line (im, IC_RED,   buf.red  , y);
		asimage_add_line (im, IC_GREEN, buf.green, y);
		asimage_add_line (im, IC_BLUE,  buf.blue , y);
		y += direction ;
    data += row_size;
	}

  free_scanline( &buf, True );

  return im;
}


ASImage      *
bitmap2asimage (unsigned char *xim, int width, int height, unsigned int compression, 
                unsigned char *mask)
{
	ASImage      *im = NULL;
	int           i, bpl, x;
	ASScanline    xim_buf;

    if( xim == NULL )
		return NULL ;

	im = create_asimage( width, height, compression);
	prepare_scanline( width, 0, &xim_buf, True );

	if( xim )
	{
	    bpl = (width*32)>>3 ;
	    if( bpl == 0 )
		    bpl = 1 ;
	    else
		    bpl = (bpl+3)/4;
	    bpl *= 4;
		for (i = 0; i < height; i++) {
            if (mask) {
               for (x = 0; x < width<<2; x += 4) {
                  xim[3 + x] = mask[x] == 0 ? 0 : 255;
               }
            }
			   raw2scanline( xim, &xim_buf, 0, width, False, True);
            if (mask) asimage_add_line (im, IC_ALPHA, xim_buf.alpha, i);
   		   asimage_add_line (im, IC_RED,   xim_buf.red, i);
		      asimage_add_line (im, IC_GREEN, xim_buf.green, i);
		      asimage_add_line (im, IC_BLUE,  xim_buf.blue, i);
			   xim += bpl;
            if (mask) mask += bpl;
		}
	}
	free_scanline(&xim_buf, True);

	return im;
}

static size_t
bmp_write32 (FILE *fp, CARD32 *data, int count)
{
  	size_t total = count;
	if( count > 0 )
	{
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		CARD8 *raw = (CARD8*)data ;
#endif
		count = 0 ;
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		while( count < total )
		{
			data[count] = (raw[0]<<24)|(raw[1]<<16)|(raw[2]<<8)|raw[3];
			++count ;
			raw += 4 ;
		}
#endif
		total = fwrite((char*) data, sizeof (CARD8), total<<2, fp)>>2;
	}
	return total;
}

static size_t
bmp_write16 (FILE *fp, CARD16 *data, int count)
{
  	size_t total = count;
	if( count > 0 )
	{
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		CARD8 *raw = (CARD8*)data ;
#endif
		count = 0 ;
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		while( count < total )
		{
			data[count] = (raw[0]<<16)|raw[1];
			++count ;
			raw += 2 ;
		}
#endif
		total = fwrite((char*) data, sizeof (CARD8), total<<1, fp)>>1;
	}
	return total;
}

Bool
ASImage2bmp ( ASImage *im, const char *path,  ASImageExportParams *params )
{
	Bool success = False;
	FILE *outfile = NULL ;
	START_TIME(started);

	if ((outfile = open_writeable_image_file( path )) != NULL)
	{
		void *bmbits ;
		BITMAPINFO *bmi = ASImage2DBI( get_default_asvisual(), im, 0, 0, im->width, im->height, &bmbits, 0 );
		if( bmi != NULL && bmbits != NULL ) 
		{
			BITMAPFILEHEADER bmh ;
			int bits_size = (((bmi->bmiHeader.biWidth*3+3)/4)*4)*bmi->bmiHeader.biHeight;          /* DWORD aligned */

			bmh.bfType = BMP_SIGNATURE;
		    bmh.bfSize = 14+bmi->bmiHeader.biSize+bits_size; /* Specifies the size, in bytes, of the bitmap file */
		    bmh.bfReserved1 = 0;
			bmh.bfReserved2 = 0;
		    bmh.bfOffBits = 14+bmi->bmiHeader.biSize; /* Specifies the offset, in bytes, 
							   * from the BITMAPFILEHEADER structure to the bitmap bits */
			/* writing off the header */
			bmp_write16( outfile, &bmh.bfType, 1 );
			bmp_write32( outfile, &bmh.bfSize, 3 );
			/* writing off the bitmapinfo : */
			bmp_write32( outfile, &bmi->bmiHeader.biSize, 1 );
			bmp_write32( outfile, (CARD32*)&bmi->bmiHeader.biWidth, 2 );
			bmp_write16( outfile, &bmi->bmiHeader.biPlanes, 2 );
			/* bmi->bmiHeader.biCompression = 0 ; */
			bmp_write32( outfile, &bmi->bmiHeader.biCompression, 6 );

			/* writing off the bitmapbits */
			if (fwrite( bmbits, sizeof(CARD8), bits_size, outfile ) == bits_size)
				success = True;
				
			free( bmbits );
			free( bmi );
			
		}
		if (outfile != stdout)
			fclose(outfile);
	}
	SHOW_TIME("image export",started);
	return success;
}

/***********************************************************************************/
/* Windows BMP file format :   									   				   */
static size_t
bmp_read32 (FILE *fp, CARD32 *data, int count)
{
  	size_t total = count;
	if( count > 0 )
	{
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		CARD8 *raw = (CARD8*)data ;
#endif
		total = fread((char*) data, sizeof (CARD8), count<<2, fp)>>2;
		count = 0 ;
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		while( count < total )
		{
			data[count] = (raw[0]<<24)|(raw[1]<<16)|(raw[2]<<8)|raw[3];
			++count ;
			raw += 4 ;
		}
#endif
	}
	return total;
}

static size_t
bmp_read16 (FILE *fp, CARD16 *data, int count)
{
  	size_t total = count;
	if( count > 0 )
	{
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		CARD8 *raw = (CARD8*)data ;
#endif
		total = fread((char*) data, sizeof (CARD8), count<<1, fp)>>1;
		count = 0 ;
#ifdef WORDS_BIGENDIAN                         /* BMPs are encoded as Little Endian */
		while( count < total )
		{
			data[count] = (raw[0]<<16)|raw[1];
			++count ;
			raw += 2 ;
		}
#endif
	}
	return total;
}


ASImage *
read_bmp_image( FILE *infile, size_t data_offset, BITMAPINFOHEADER *bmp_info,
				ASScanline *buf, CARD8 *gamma_table,
				unsigned int width, unsigned int height,
				Bool add_colormap, unsigned int compression )
{
	Bool success = False ;
	CARD8 *cmap = NULL ;
	int cmap_entries = 0, cmap_entry_size = 4, row_size ;
	int y;
	ASImage *im = NULL ;
	CARD8 *data ;
	int direction = -1 ;

	if( bmp_read32( infile, &bmp_info->biSize, 1 ) )
	{
		if( bmp_info->biSize == 40 )
		{/* long header */
			bmp_read32( infile, (CARD32*)&bmp_info->biWidth, 2 );
			bmp_read16( infile, &bmp_info->biPlanes, 2 );
			bmp_info->biCompression = 1 ;
			success = (bmp_read32( infile, &bmp_info->biCompression, 6 )==6);
		}else
		{
			CARD16 dumm[2] ;
			bmp_read16( infile, &dumm[0], 2 );
			bmp_info->biWidth = dumm[0] ;
			bmp_info->biHeight = dumm[1] ;
			success = ( bmp_read16( infile, &bmp_info->biPlanes, 2 ) == 2 );
			bmp_info->biCompression = 0 ;
		}
	}
#ifdef LOCAL_DEBUG
	fprintf( stderr, "bmp.info.biSize = %ld(0x%lX)\n", bmp_info->biSize, bmp_info->biSize );
	fprintf( stderr, "bmp.info.biWidth = %ld\nbmp.info.biHeight = %ld\n",  bmp_info->biWidth,  bmp_info->biHeight );
	fprintf( stderr, "bmp.info.biPlanes = %d\nbmp.info.biBitCount = %d\n", bmp_info->biPlanes, bmp_info->biBitCount );
	fprintf( stderr, "bmp.info.biCompression = %ld\n", bmp_info->biCompression );
	fprintf( stderr, "bmp.info.biSizeImage = %ld\n", bmp_info->biSizeImage );
#endif
	if( ((int)(bmp_info->biHeight)) < 0 )
		direction = 1 ;
	if( height == 0 )
		height  = direction == 1 ? -((long)(bmp_info->biHeight)):bmp_info->biHeight ;
	if( width == 0 )
		width = bmp_info->biWidth ;

	if( !success || bmp_info->biCompression != 0 ||
		width > MAX_IMPORT_IMAGE_SIZE ||
		height > MAX_IMPORT_IMAGE_SIZE )
	{
		return NULL;
	}
	if( bmp_info->biBitCount < 16 )
		cmap_entries = 0x01<<bmp_info->biBitCount ;

	if( bmp_info->biSize != 40 )
		cmap_entry_size = 3;
	if( cmap_entries )
	{
		cmap = safemalloc( cmap_entries * cmap_entry_size );
		fread(cmap, sizeof (CARD8), cmap_entries * cmap_entry_size, infile);
	}

	if( add_colormap )
		data_offset += cmap_entries*cmap_entry_size ;

	fseek( infile, data_offset, SEEK_SET );
	row_size = (width*bmp_info->biBitCount)>>3 ;
	if( row_size == 0 )
		row_size = 1 ;
	else
		row_size = (row_size+3)/4 ;            /* everything is aligned by 32 bits */
	row_size *= 4 ;                            /* in bytes  */
	data = safemalloc( row_size );

	im = create_asimage( width,  height, compression );
	/* Window BMP files are little endian  - we need to swap Red and Blue */
	prepare_scanline( im->width, 0, buf, True );

	y =( direction == 1 )?0:height-1 ;
	while( y >= 0 && y < (int)height)
	{
		if( fread( data, sizeof (char), row_size, infile ) < (unsigned int)row_size )
			break;
 		dib_data_to_scanline(buf, bmp_info, gamma_table, data, cmap, cmap_entry_size); 
		asimage_add_line (im, IC_RED,   buf->red  , y);
		asimage_add_line (im, IC_GREEN, buf->green, y);
		asimage_add_line (im, IC_BLUE,  buf->blue , y);
		y += direction ;
	}
	free( data );
	if( cmap )
		free( cmap );
	return im ;
}

ASImage *
bmp2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im = NULL ;
	/* More stuff */
	FILE         *infile;					   /* source file */
	ASScanline    buf;
	BITMAPFILEHEADER  bmp_header ;
	BITMAPINFOHEADER  bmp_info;
	START_TIME(started);


	if ((infile = open_image_file(path)) == NULL)
		return NULL;

	bmp_header.bfType = 0 ;
	if( bmp_read16( infile, &bmp_header.bfType, 1 ) )
		if( bmp_header.bfType == BMP_SIGNATURE )
			if( bmp_read32( infile, &bmp_header.bfSize, 3 ) == 3 )
				im = read_bmp_image( infile, bmp_header.bfOffBits, &bmp_info, &buf, params->gamma_table, 0, 0, False, params->compression );
#ifdef LOCAL_DEBUG
	fprintf( stderr, "bmp.header.bfType = 0x%X\nbmp.header.bfSize = %ld\nbmp.header.bfOffBits = %ld(0x%lX)\n",
					  bmp_header.bfType, bmp_header.bfSize, bmp_header.bfOffBits, bmp_header.bfOffBits );
#endif
	if( im != NULL )
		free_scanline( &buf, True );
	else
		show_error( "invalid or unsupported BMP format in image file \"%s\"", path );

	fclose( infile );
	SHOW_TIME("image loading",started);
	return im ;
}

/***********************************************************************************/
/* Windows ICO/CUR file format :   									   			   */

ASImage *
ico2ASImage( const char * path, ASImageImportParams *params )
{
	ASImage *im = NULL ;
	/* More stuff */
	FILE         *infile;					   /* source file */
	ASScanline    buf;
	int y, mask_bytes;
    CARD8  *and_mask;
	START_TIME(started);
	struct IconDirectoryEntry {
    	CARD8  bWidth;
    	CARD8  bHeight;
    	CARD8  bColorCount;
    	CARD8  bReserved;
    	CARD16  wPlanes;
    	CARD16  wBitCount;
    	CARD32 dwBytesInRes;
    	CARD32 dwImageOffset;
	};
	struct ICONDIR {
    	CARD16          idReserved;
    	CARD16          idType;
    	CARD16          idCount;
	} icon_dir;
   	struct IconDirectoryEntry  icon;
	BITMAPINFOHEADER bmp_info;

	if ((infile = open_image_file(path)) == NULL)
		return NULL;

	icon_dir.idType = 0 ;
	if( bmp_read16( infile, &icon_dir.idReserved, 3 ) == 3)
		if( icon_dir.idType == 1 || icon_dir.idType == 2)
		{
			fread( &(icon.bWidth), sizeof(CARD8),4,infile );
			bmp_read16( infile, &(icon.wPlanes), 2 );
			if( bmp_read32( infile, &(icon.dwBytesInRes), 2 ) == 2 )
			{
				fseek( infile, icon.dwImageOffset, SEEK_SET );
				im = read_bmp_image( infile, icon.dwImageOffset+40+(icon.bColorCount*4), &bmp_info, &buf, params->gamma_table,
					                 icon.bWidth, icon.bHeight, (icon.bColorCount==0), params->compression );
			}
		}
#ifdef LOCAL_DEBUG
	fprintf( stderr, "icon.dir.idType = 0x%X\nicon.dir.idCount = %d\n",  icon_dir.idType, icon_dir.idCount );
	fprintf( stderr, "icon[1].bWidth = %d(0x%X)\n",  icon.bWidth,  icon.bWidth );
	fprintf( stderr, "icon[1].bHeight = %d(0x%X)\n",  icon.bHeight,  icon.bHeight );
	fprintf( stderr, "icon[1].bColorCount = %d\n",  icon.bColorCount );
	fprintf( stderr, "icon[1].dwImageOffset = %ld(0x%lX)\n",  icon.dwImageOffset,  icon.dwImageOffset );
    fprintf( stderr, "icon[1].bmp_size = %ld\n",  icon.dwBytesInRes );
    fprintf( stderr, "icon[1].dwBytesInRes = %ld\n",  icon.dwBytesInRes );
#endif
	if( im != NULL )
	{
        mask_bytes = ((icon.bWidth>>3)+3)/4 ;    /* everything is aligned by 32 bits */
        mask_bytes *= 4 ;                      /* in bytes  */
        and_mask = safemalloc( mask_bytes );
        for( y = icon.bHeight-1 ; y >= 0 ; y-- )
		{
			int x ;
            if( fread( and_mask, sizeof (CARD8), mask_bytes, infile ) < (unsigned int)mask_bytes )
				break;
			for( x = 0 ; x < icon.bWidth ; ++x )
            {
				buf.alpha[x] = (and_mask[x>>3]&(0x80>>(x&0x7)))? 0x0000 : 0x00FF ;
            }
			im->channels[IC_ALPHA][y]  = store_data( NULL, (CARD8*)buf.alpha, im->width*4, 
													 ASStorage_32BitRLE|ASStorage_Bitmap, 0);
		}
        free( and_mask );
		free_scanline( &buf, True );
	}else
		show_error( "invalid or unsupported ICO format in image file \"%s\"", path );

	fclose( infile );
	SHOW_TIME("image loading",started);
	return im ;
}

