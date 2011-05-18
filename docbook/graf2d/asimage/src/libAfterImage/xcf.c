/* This file contains code for unified image loading from XCF file  */
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

/*#define LOCAL_DEBUG*/
/*#define DO_CLOCKING*/

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <string.h>
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

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asimage.h"
#include "xcf.h"

static XcfProperty *read_xcf_props( FILE *fp );
static XcfListElem *read_xcf_list_offsets( FILE *fp, size_t elem_size );
static void 		read_xcf_layers( XcfImage *xcf_im, FILE *fp, XcfLayer *head );
static void			read_xcf_channels( XcfImage *xcf_im, FILE *fp, XcfChannel *head );
static XcfHierarchy*read_xcf_hierarchy( XcfImage *xcf_im, FILE *fp, CARD8 opacity, ARGB32 colormask );
static void 		read_xcf_levels( XcfImage *xcf_im, FILE *fp, XcfLevel *head );
static void 		read_xcf_tiles( XcfImage *xcf_im, FILE *fp, XcfTile *head );
static void 		read_xcf_tiles_rle( XcfImage *xcf_im, FILE *fp, XcfTile *head );

static size_t
xcf_read8 (FILE *fp, CARD8 *data, int count)
{
  	size_t total = count;

  	while (count > 0)
    {
	  	int bytes = fread ((char*) data, sizeof (char), count, fp);
      	if( bytes <= 0 )
			break;
      	count -= bytes;
      	data += bytes;
    }
	return total;
}

static size_t
xcf_read32 (FILE *fp, CARD32 *data, int count)
{
  	size_t total = count;
	if( count > 0 )
	{
		CARD8 *raw = (CARD8*)data ;
		total = xcf_read8( fp, raw, count<<2 )>>2;
		count = 0 ;
#ifndef WORDS_BIGENDIAN
		while( count < (int)total )
		{
			data[count] = (raw[0]<<24)|(raw[1]<<16)|(raw[2]<<8)|raw[3];
			++count ;
			raw += 4 ;
		}
#endif
	}
	return total;
}

static void
xcf_skip_string (FILE *fp)
{
	CARD32 size = 0;
	if( xcf_read32 (fp, &size, 1)< 1 )
		return;
	if( size > 0 )
	{
		fseek(fp, size, SEEK_CUR );
	}
}
void print_xcf_layers( char* prompt, XcfLayer *head );

XcfImage *
read_xcf_image( FILE *fp )
{
	XcfImage *xcf_im = NULL ;
	XcfProperty *prop ;

	if( fp )
	{
		int i ;
		char sig[XCF_SIGNATURE_FULL_LEN+1] ;
		if( xcf_read8( fp, (unsigned char*)&(sig[0]),XCF_SIGNATURE_FULL_LEN ) >= XCF_SIGNATURE_FULL_LEN )
		{
			if( mystrncasecmp( sig, XCF_SIGNATURE, XCF_SIGNATURE_LEN) == 0 )
			{
				xcf_im = safecalloc( 1, sizeof(XcfImage));
				if( mystrncasecmp( &(sig[XCF_SIGNATURE_LEN+1]), "file", 4 ) == 0 )
					xcf_im->version = 0 ;
				else
					xcf_im->version = atoi(&(sig[XCF_SIGNATURE_LEN+1]));
				if( xcf_read32( fp, &(xcf_im->width), 3 ) < 3 )
				{
					free( xcf_im );
					xcf_im = NULL ;
				}
			}
		}
		if( xcf_im == NULL )
		{
			show_error( "invalid .xcf file format - not enough data to read" );
			return NULL ;
		}

		xcf_im->properties = read_xcf_props( fp );
		for( prop = xcf_im->properties ; prop != NULL ; prop = prop->next )
		{
			if( prop->id == XCF_PROP_COLORMAP )
			{
				register int i ;
				CARD32 n = *((CARD32*)(prop->data)) ;
				n = as_ntohl(n);
				xcf_im->num_cols = n ;
				xcf_im->colormap = safemalloc( MAX(n*3,(CARD32)XCF_COLORMAP_SIZE));
				if( xcf_im->version == 0 )
				{
					for( i = 0 ; i < (int)n ; i++ )
					{
						xcf_im->colormap[i*3] = i ;
						xcf_im->colormap[i*3+1] = i ;
						xcf_im->colormap[i*3+2] = i ;
					}
				}else
					memcpy( xcf_im->colormap, prop->data+4, MIN(prop->len-4,n));
			}else if( prop->id == XCF_PROP_COMPRESSION )
				xcf_im->compression = *(prop->data);
		}
		xcf_im->layers = 	(XcfLayer*)  read_xcf_list_offsets( fp, sizeof(XcfLayer)  );
		xcf_im->channels = 	(XcfChannel*)read_xcf_list_offsets( fp, sizeof(XcfChannel));
		for( i = 0 ; i < XCF_TILE_HEIGHT ; i++ )
			prepare_scanline(xcf_im->width,0,&(xcf_im->scanline_buf[i]), False );

		if( xcf_im->layers )
			read_xcf_layers( xcf_im, fp, xcf_im->layers );
		if( xcf_im->channels )
			read_xcf_channels( xcf_im, fp, xcf_im->channels );
	}
	return xcf_im;
}

/*******************************************************************************/
/* printing functions :     												   */
void
print_xcf_properties( char* prompt, XcfProperty *prop )
{
	register int i = 0 ;
	while( prop )
	{
		fprintf( stderr, "%s.properties[%d] = %p\n", prompt, i, prop );
		fprintf( stderr, "%s.properties[%d].id = %ld\n", prompt, i, (long)prop->id );
		fprintf( stderr, "%s.properties[%d].size = %ld\n", prompt, i, (long)prop->len );
		if( prop->len > 0 )
		{
			register unsigned int k ;
			fprintf( stderr, "%s.properties[%d].data = ", prompt, i );
			for( k = 0 ; k < prop->len ; k++ )
				fprintf( stderr, "%2.2X ", prop->data[k] );
			fprintf( stderr, "\n" );
		}
		prop = prop->next ;
		++i ;
	}
}

void
print_xcf_hierarchy( char* prompt, XcfHierarchy *h )
{
	if( h )
	{
		XcfLevel *level = h->levels ;
		int i = 0 ;

		fprintf( stderr, "%s.hierarchy.width = %ld\n", prompt, (long)h->width );
		fprintf( stderr, "%s.hierarchy.height = %ld\n", prompt,(long) h->height );
		fprintf( stderr, "%s.hierarchy.bpp = %ld\n", prompt, (long)h->bpp );
		while( level )
		{
			XcfTile *tile = level->tiles ;
			int k = 0 ;
			fprintf( stderr, "%s.hierarchy.level[%d].offset = %ld\n", prompt, i, (long)level->offset );
			fprintf( stderr, "%s.hierarchy.level[%d].width = %ld\n", prompt, i, (long)level->width );
			fprintf( stderr, "%s.hierarchy.level[%d].height = %ld\n", prompt, i, (long)level->height );
			while ( tile )
			{
				fprintf( stderr, "%s.hierarchy.level[%d].tile[%d].offset = %ld\n", prompt, i, k, (long)tile->offset );
				fprintf( stderr, "%s.hierarchy.level[%d].tile[%d].estimated_size = %ld\n", prompt, i, k, (long)tile->estimated_size );
				tile = tile->next ;
				++k ;
			}
			level = level->next ;
			++i ;
		}
	}
}

void
print_xcf_channels( char* prompt, XcfChannel *head, Bool mask )
{
	register int i = 0 ;
	char p[256] ;
	while( head )
	{
		if( mask )
			sprintf( p, "%s.mask", prompt );
		else
			sprintf( p, "%s.channel[%d]", prompt, i );

		if( head->offset > 0 )
			fprintf( stderr, "%s.offset = %ld\n", p, (long)head->offset );
		fprintf( stderr, "%s.width = %ld\n" , p,(long) head->width );
		fprintf( stderr, "%s.height = %ld\n", p, (long)head->height );
		print_xcf_properties( p, head->properties );
		fprintf( stderr, "%s.opacity = %ld\n", p, (long)head->opacity );
		fprintf( stderr, "%s.visible = %d\n" , p, head->visible );
		fprintf( stderr, "%s.color = #%lX\n" , p, (long)head->color );
		fprintf( stderr, "%s.hierarchy_offset = %ld\n", p, (long)head->hierarchy_offset );
		print_xcf_hierarchy( p, head->hierarchy );

		head = head->next ;
		++i ;
	}
}

void
print_xcf_layers( char* prompt, XcfLayer *head )
{
	register int i = 0 ;
	char p[256] ;
	while( head )
	{
		fprintf( stderr, "%s.layer[%d] = %p\n", prompt, i, head );
		fprintf( stderr, "%s.layer[%d].offset = %ld\n", prompt, i, (long)head->offset );
		fprintf( stderr, "%s.layer[%d].width = %ld\n", prompt, i, (long)head->width );
		fprintf( stderr, "%s.layer[%d].height = %ld\n", prompt, i, (long)head->height );
		fprintf( stderr, "%s.layer[%d].type = %ld\n", prompt, i, (long)head->type );
		sprintf( p, "%s.layer[%d]", prompt, i );
		print_xcf_properties( p, head->properties );
		fprintf( stderr, "%s.layer[%d].opacity = %ld\n", prompt, i, (long)head->opacity );
		fprintf( stderr, "%s.layer[%d].visible = %d\n", prompt, i, head->visible );
		fprintf( stderr, "%s.layer[%d].preserve_transparency = %d\n", prompt, i,  head->preserve_transparency );
		fprintf( stderr, "%s.layer[%d].mode = %ld\n"    , prompt, i, (long)head->mode 				   );
		fprintf( stderr, "%s.layer[%d].offset_x = %ld\n", prompt, i, (long)head->offset_x 			   );
		fprintf( stderr, "%s.layer[%d].offset_y = %ld\n", prompt, i, (long)head->offset_y 			   );

		fprintf( stderr, "%s.layer[%d].hierarchy_offset = %ld\n", prompt, i, (long)head->hierarchy_offset );
		print_xcf_hierarchy( p, head->hierarchy );
		fprintf( stderr, "%s.layer[%d].mask_offset = %ld\n", prompt, i, (long)head->mask_offset );
		print_xcf_channels( p, head->mask, True );

		head = head->next ;
		++i ;
	}
}

void
print_xcf_image( XcfImage *xcf_im )
{
	if( xcf_im )
	{
		fprintf( stderr, "XcfImage.version = %d\n", xcf_im->version );
		fprintf( stderr, "XcfImage.width = %ld\nXcfImage.height = %ld\nXcfImage.type = %ld\n",
		   				  (long)xcf_im->width, (long)xcf_im->height, (long)xcf_im->type );
		fprintf( stderr, "XcfImage.num_cols = %ld\n", (long)xcf_im->num_cols );
		fprintf( stderr, "XcfImage.compression = %d\n", xcf_im->compression );
		print_xcf_properties( "XcfImage", xcf_im->properties );
		print_xcf_layers( "XcfImage", xcf_im->layers );
		print_xcf_channels( "XcfImage", xcf_im->channels, False );
	}
}

/*******************************************************************************/
/* deallocation functions : 												   */

void
free_xcf_properties( XcfProperty *head )
{
	while( head )
	{
		XcfProperty *next = head->next ;
		if( head->len > 0  && head->data && head->data != (CARD8*)&(head->buffer[0]))
			free( head->data );
		free( head );
		head = next ;
	}
}

void
free_xcf_hierarchy( XcfHierarchy *hierarchy )
{
	if( hierarchy )
	{
		register XcfLevel *level = hierarchy->levels;
		while( level )
		{
			XcfLevel *next = level->next ;
			while( level->tiles )
			{
				XcfTile *next = level->tiles->next ;
				if( level->tiles->data )
					free( level->tiles->data );
				free( level->tiles );
				level->tiles = next ;
			}
			free( level );
			level = next ;
		}
		if( hierarchy->image )
            destroy_asimage( &hierarchy->image );
		free( hierarchy );
	}
}

void
free_xcf_channels( XcfChannel *head )
{
	while( head )
	{
		XcfChannel *next = head->next ;
		if( head->properties )
			free_xcf_properties( head->properties );
		if( head->hierarchy )
			free_xcf_hierarchy( head->hierarchy );
		free( head );
		head = next ;
	}
}

void
free_xcf_layers( XcfLayer *head )
{
	while( head )
	{
		XcfLayer *next = head->next ;
		if( head->properties )
			free_xcf_properties( head->properties );
		if( head->hierarchy )
			free_xcf_hierarchy( head->hierarchy );
		free_xcf_channels( head->mask );
		free( head );
		head = next ;
	}
}

void
free_xcf_image( XcfImage *xcf_im )
{
	if( xcf_im )
	{
		int i ;

		if( xcf_im->properties )
			free_xcf_properties( xcf_im->properties );
		if( xcf_im->colormap )
			free( xcf_im->colormap );
		if( xcf_im->layers )
			free_xcf_layers( xcf_im->layers );
		if( xcf_im->channels )
			free_xcf_channels( xcf_im->channels );

		for( i = 0 ; i < XCF_TILE_HEIGHT ; i++ )
			free_scanline( &(xcf_im->scanline_buf[i]), True );
	}
}


/*******************************************************************************/
/* detail loading functions : 												   */

static XcfProperty *
read_xcf_props( FILE *fp )
{
	XcfProperty *head = NULL;
	XcfProperty **tail = &head;
	CARD32 prop_vals[2] ;

	do
	{
		if( xcf_read32( fp, &(prop_vals[0]), 2 ) < 2 )
			break;
		if( prop_vals[0] != 0 )
		{
			*tail = safecalloc( 1, sizeof(XcfProperty));
			(*tail)->id  = prop_vals[0] ;
			(*tail)->len = prop_vals[1] ;
			if( (*tail)->len > 0 )
			{
				if( (*tail)->len <= 8 )
					(*tail)->data = (CARD8*)&((*tail)->buffer[0]) ;
				else
					(*tail)->data = safemalloc( (*tail)->len );
				xcf_read8( fp, (*tail)->data, (*tail)->len );
			}
			tail = &((*tail)->next);
		}
	}while( prop_vals[0] != 0 );
	return head;
}

static XcfListElem *
read_xcf_list_offsets( FILE *fp, size_t elem_size )
{
	XcfListElem *head = NULL ;
	XcfListElem **tail = &head ;
	CARD32 offset ;

	do
	{
		if( xcf_read32( fp, &offset, 1 ) < 1 )
			break;
		if( offset != 0 )
		{
			*tail = safecalloc( 1, elem_size);
			(*tail)->any.offset  = offset ;
			tail = (XcfListElem**)&((*tail)->any.next);
		}
	}while( offset != 0 );
	return head;
}

static void
read_xcf_layers( XcfImage *xcf_im, FILE *fp, XcfLayer *head )
{
	XcfProperty *prop ;
	while( head )
	{
		fseek( fp, head->offset, SEEK_SET );
		if( xcf_read32( fp, &(head->width), 3 ) < 3 )
		{
			head->width = 0 ;
			head->height = 0 ;
			head->type = 0 ;
			continue;                          /* not enough data */
		}
		xcf_skip_string(fp);
		head->properties = read_xcf_props( fp );
		for( prop = head->properties ; prop != NULL ; prop = prop->next )
		{
			CARD32 *pd = (CARD32*)(prop->data) ;
			if( prop->id ==  XCF_PROP_FLOATING_SELECTION )
			{
				xcf_im->floating_selection = head;
			}else if( prop->id ==  XCF_PROP_OPACITY && pd)
			{
				head->opacity = as_ntohl(*pd);
			}else if( prop->id ==  XCF_PROP_VISIBLE && pd)
			{
				head->visible = ( *pd !=0);
			}else if( prop->id ==  XCF_PROP_PRESERVE_TRANSPARENCY && pd)
			{
				head->preserve_transparency = (*pd!=0);
			}else if( prop->id == XCF_PROP_MODE && pd)
			{
				head->mode = as_ntohl(*pd);
			}else if( prop->id == XCF_PROP_OFFSETS && pd)
			{
				head->offset_x = as_ntohl(pd[0]);
				head->offset_y = as_ntohl(pd[1]);
			}
		}

		if( xcf_im->floating_selection != head && head->visible )
		{ /* we absolutely do not want to load floating selection or invisible layers :*/
			if( xcf_read32( fp, &(head->hierarchy_offset), 2 ) < 2 )
			{
				head->hierarchy_offset = 0 ;
				head->mask_offset = 0 ;
			}
			if( head->hierarchy_offset > 0 )
			{
				fseek( fp, head->hierarchy_offset, SEEK_SET );
				head->hierarchy = read_xcf_hierarchy( xcf_im, fp, (CARD8)head->opacity, 0xFFFFFFFF );
			}
			if( head->mask_offset > 0 )
			{
				head->mask = safecalloc( 1, sizeof(XcfChannel) );
				head->mask->offset = head->mask_offset;
				read_xcf_channels( xcf_im, fp, head->mask );
			}
		}

		head = head->next ;
	}
}

static void
read_xcf_channels( XcfImage *xcf_im, FILE *fp, XcfChannel *head )
{
	XcfProperty *prop ;
	while( head )
	{
		fseek( fp, head->offset, SEEK_SET );
		if( xcf_read32( fp, &(head->width), 2 ) < 2 )
		{
			head->width = 0 ;
			head->height = 0 ;
			continue;                          /* not enough data */
		}
		xcf_skip_string(fp);
		head->properties = read_xcf_props( fp );
		for( prop = head->properties ; prop != NULL ; prop = prop->next )
		{
			CARD32 *pd = (CARD32*)(prop->data) ;
			if( prop->id ==  XCF_PROP_OPACITY )
			{
				head->opacity = as_ntohl(*pd);
			}else if( prop->id ==  XCF_PROP_VISIBLE )
			{
				head->visible = ( *pd !=0);
			}else if( prop->id ==  XCF_PROP_COLOR )
			{
				head->color = MAKE_ARGB32(0xFF,prop->data[0],prop->data[1],prop->data[2]);
			}
		}

		if( head->visible )
		{  	/* onli visible channels we need : */
			if( xcf_read32( fp, &(head->hierarchy_offset), 1 ) < 1 )
				head->hierarchy_offset = 0 ;

			if( head->hierarchy_offset > 0 )
			{
				fseek( fp, head->hierarchy_offset, SEEK_SET );
				head->hierarchy = read_xcf_hierarchy( xcf_im, fp, (CARD8)head->opacity, head->color );
			}
		}
		head = head->next ;
	}
}

typedef void (*decode_xcf_tile_func)( FILE *fp, XcfTile *tile, int bpp,
		  				  			  ASScanline *buf, CARD8* tile_buf, int offset_x, int offset_y, int width, int height);


void decode_xcf_tile( FILE *fp, XcfTile *tile, int bpp,
					 ASScanline *buf, CARD8* tile_buf, int offset_x, int offset_y, int width, int height);
void decode_xcf_tile_rle( FILE *fp, XcfTile *tile, int bpp,
					 ASScanline *buf, CARD8* tile_buf, int offset_x, int offset_y, int width, int height);
Bool fix_xcf_image_line( ASScanline *buf, int bpp, unsigned int width, CARD8 *cmap,
	 	  				 CARD8 opacity, ARGB32 color );


static XcfHierarchy*
read_xcf_hierarchy( XcfImage *xcf_im, FILE *fp, CARD8 opacity, ARGB32 colormask )
{
	XcfHierarchy *h = NULL ;
	CARD32 h_props[3] ;

	if( xcf_read32( fp, &(h_props[0]), 3 ) < 3 )
		return NULL;
	h = safecalloc(1, sizeof(XcfHierarchy));

	h->width  = h_props[0] ;
	h->height = h_props[1] ;
	h->bpp	  = h_props[2] ;

	h->levels = (XcfLevel*)read_xcf_list_offsets( fp, sizeof(XcfLevel));
	if( h->levels )
	{
		read_xcf_levels( xcf_im, fp, h->levels );

		/* now we want to try and merge all the tiles into single ASImage */
		if( h->levels->width == h->width && h->levels->height == h->height )
		{ /* only first level is interesting for us : */
		  /* do not know why, but GIMP (at least up to v1.3) has been writing only
		   * one level, and faking the rest - future extensibility ? */
			int height_left = h->height ;
			ASScanline 	*buf = &(xcf_im->scanline_buf[0]) ;
			XcfTile 		*tile = h->levels->tiles ;
			decode_xcf_tile_func decode_func = decode_xcf_tile ;
			CARD8 			*tile_buf = &(xcf_im->tile_buf[0]);
			int i;

			if( xcf_im->compression == XCF_COMPRESS_RLE )
				decode_func = decode_xcf_tile_rle ;
			else if( xcf_im->compression != XCF_COMPRESS_NONE )
			{
				show_error( "XCF image contains information compressed with usupported method." );
				return h;
			}

			if (XCF_TILE_WIDTH < h->width)
				tile_buf = safemalloc (h->width*XCF_TILE_HEIGHT*6);
				
			if (xcf_im->width < h->width)
				for( i = 0 ; i < XCF_TILE_HEIGHT ; i++ )
				{
					free_scanline (&(xcf_im->scanline_buf[i]), True); 
					prepare_scanline (h->width,0,&(xcf_im->scanline_buf[i]), False );
				}

			h->image = create_asimage(  h->width, h->height, 0/* no compression */ );
			while( height_left > 0 && tile )
			{
				int width_left = h->width ;
				int max_i, y ;
				/* first - lets collect our data : */
				while( width_left > 0 && tile )
				{
					fseek( fp, tile->offset, SEEK_SET );
					decode_func(fp, tile, h->bpp, buf, tile_buf,
							    h->width-width_left, h->height-height_left,  /* really don't need this one */
								MIN(width_left,XCF_TILE_WIDTH), MIN(height_left,XCF_TILE_HEIGHT));

					width_left -= XCF_TILE_WIDTH ;
					tile = tile->next ;
				}

				/* now lets encode it into ASImage : */
				max_i = MIN(height_left,XCF_TILE_HEIGHT);
				y = h->height - height_left ;
				for( i = 0 ; i < max_i ; i++ )
				{
					Bool do_alpha = fix_xcf_image_line( &(buf[i]), h->bpp, h->width, xcf_im->colormap, opacity, colormask );
					if( h->bpp > 1 || xcf_im->colormap != NULL )
					{
						asimage_add_line (h->image, IC_RED,   buf[i].red  , y+i);
						asimage_add_line (h->image, IC_GREEN, buf[i].green, y+i);
						asimage_add_line (h->image, IC_BLUE,  buf[i].blue , y+i);
					}
					if( do_alpha ) /* we don't want to store alpha component - if its all FF */
						asimage_add_line (h->image, IC_ALPHA, buf[i].alpha, y+i);
				}
				/* continue on to the next row : */
				height_left -= XCF_TILE_HEIGHT ;
			}
			if (tile_buf != &(xcf_im->tile_buf[0]))
				free (tile_buf);
		}
	}
	return h;
}

static void
read_xcf_levels( XcfImage *xcf_im, FILE *fp, XcfLevel *head )
{
	while( head )
	{
		fseek( fp, head->offset, SEEK_SET );
		if( xcf_read32( fp, &(head->width), 2 ) < 2 )
		{
			head->width = 0 ;
			head->height = 0 ;
			continue;                          /* not enough data */
		}

		head->tiles = (XcfTile*)read_xcf_list_offsets( fp, sizeof(XcfTile));
		if( head->tiles )
		{
			if( xcf_im->compression == XCF_COMPRESS_NONE )
				read_xcf_tiles( xcf_im, fp, head->tiles );
			else if( xcf_im->compression == XCF_COMPRESS_RLE )
				read_xcf_tiles_rle( xcf_im, fp, head->tiles );
		}
		head = head->next ;
	}
}

static void
read_xcf_tiles( XcfImage *xcf_im, FILE *fp, XcfTile *head )
{
	while( head )
	{
		head->estimated_size = XCF_TILE_WIDTH*XCF_TILE_HEIGHT*4 ;
		head = head->next ;
	}
}


static void
read_xcf_tiles_rle( XcfImage *xcf_im, FILE *fp, XcfTile *head )
{
	while( head )
	{
		if( head->next )
			head->estimated_size = head->next->offset - head->offset ;
		else
			head->estimated_size = (XCF_TILE_WIDTH*XCF_TILE_HEIGHT)*6 ;
		head = head->next ;
	}
}

/* now the fun part of actually decoding ARGB values : */

static inline void
store_colors( CARD8 *data, ASScanline *curr_buf, int bpp, int comp, int offset_x, int width )
{
	register int i ;
	CARD32   *out = NULL;
	if( comp+1 < bpp || bpp == 3 )
	{
		switch( comp )
		{
			case 0 : out = &(curr_buf->red[offset_x]); break ;
			case 1 : out = &(curr_buf->green[offset_x]); break ;
			case 2 : out = &(curr_buf->blue[offset_x]); break ;
		}
	}else
		out = &(curr_buf->alpha[offset_x]);

	if( out ) 
  		for( i = 0 ; i < width ; i++ )
			out[i] = data[i] ;
}

void
decode_xcf_tile( FILE *fp, XcfTile *tile, int bpp,
			   	 ASScanline *buf, CARD8* tile_buf, int offset_x, int offset_y, int width, int height)
{
	int bytes_in, available = width*height ;
	int y = 0;
	int comp = 0 ;

	bytes_in = xcf_read8( fp, tile_buf, available*6 );
	while( comp < bpp && bytes_in >= 2 )
	{
		while ( y < height )
		{
			store_colors( tile_buf, &(buf[y]), bpp, comp, offset_x, MIN(width,bytes_in));
			tile_buf += width ;
			bytes_in -= width ;
			++y ;
		}
		++comp;
		y = 0 ;
	}
}


void
decode_xcf_tile_rle( FILE *fp, XcfTile *tile, int bpp,
					 ASScanline *buf, CARD8* tile_buf, int offset_x, int offset_y, int width, int height)
{
	int bytes_in, available = width*height ;
	int x = 0, y = 0;
	CARD8	tmp[XCF_TILE_WIDTH] ;
	int comp = 0 ;

	bytes_in = xcf_read8( fp, tile_buf, available*6 );
	while( comp < bpp && bytes_in >= 2 )
	{
		while ( y < height )
		{
			int len = *tile_buf ;
			register int i ;
			++tile_buf;
			--bytes_in;
			if( len >= 128 )
			{									   /* direct data  */
				if( len == 128 )
				{
					len = (((int)tile_buf[0])<<8)+tile_buf[1] ;
					tile_buf += 2 ; bytes_in -= 2 ;
				}else
					len = 255 - (len-1);
				if( len > bytes_in )
					break;
				for( i = 0 ; i < len ; ++i )
				{
					tmp[x] = tile_buf[i] ;
					if( ++x >= width )
					{
						store_colors( &(tmp[0]), &(buf[y]), bpp, comp, offset_x, width );
 						x = 0 ;
						++y ;
						if( y >= height )
							i = len ;
					}
				}
				tile_buf += len ;
				bytes_in -= len ;
			}else
			{                                      /* repeating data */
				CARD8 v ;
				++len ;
				if( len == 128 )
				{
					len = (((int)tile_buf[0])<<8)+tile_buf[1] ;
					tile_buf += 2 ; bytes_in -= 2 ;
				}
				if( len >= bytes_in )
					len = bytes_in-1 ;
				v = tile_buf[0] ;
				for( i = 0 ; i < len ; ++i)
				{
					tmp[x] = v ;
					if( ++x >= width )
					{
						store_colors( &(tmp[0]), &(buf[y]), bpp, comp, offset_x, width );
						x = 0 ;
						++y ;
						if( y >= height )
							i = len ;
					}
				}
				++tile_buf;
				--bytes_in;
			}
		}
		++comp;
		x = 0 ;
		y = 0 ;
	}
}

Bool
fix_xcf_image_line( ASScanline *buf, int bpp, unsigned int width, CARD8 *cmap,
					CARD8 opacity, ARGB32 color )
{
	register unsigned int i ;
	Bool do_alpha = False ;
	if( bpp == 1 )
	{
		if( cmap )
		{
			for( i = 0 ; i < width ; i++ )
			{
				int cmap_idx = ((int)(buf->alpha[i]))*3 ;
				buf->red[i]   = cmap[cmap_idx];
				buf->blue[i]  = cmap[cmap_idx+1];
				buf->green[i] = cmap[cmap_idx+2];
				buf->alpha[i] = opacity;
			}
		}if ( (color&0x00FFFFFF) == 0x00FFFFFF )
			for( i = 0 ; i < width ; i++ )
			{
				buf->red[i]   = buf->alpha[i];
			   	buf->blue[i]  = buf->alpha[i];
			   	buf->green[i] = buf->alpha[i];
				buf->alpha[i] = opacity;
			}
		else
			for( i = 0 ; i < width ; i++ )
				buf->alpha[i] = ((int)(buf->alpha[i])*opacity)>>8;
	}if( bpp == 2 )
	{
		for( i = 0 ; i < width ; i++ )
		{
			if( cmap )
			{
				int cmap_idx = ((int)(buf->red[i]))*3 ;
				buf->red[i]   = cmap[cmap_idx];
				buf->blue[i]  = cmap[cmap_idx+1];
				buf->green[i] = cmap[cmap_idx+2];
			}else
				buf->blue[i] = buf->green[i] = buf->red[i] ;

			buf->alpha[i] = ((int)(buf->alpha[i])*opacity)>>8;
			if( (buf->alpha[i]&0x00FF) != 0x00FF )
				do_alpha = True ;
		}
	}else
	{
		for( i = 0 ; i < width ; i++ )
		{
			buf->alpha[i] = ((int)(buf->alpha[i])*opacity)>>8;
			if( (buf->alpha[i]&0x00FF) != 0x00FF )
				do_alpha = True ;
		}
	}
	return do_alpha;
}
