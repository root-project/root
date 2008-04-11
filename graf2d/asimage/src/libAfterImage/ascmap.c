/* This file contains code colormapping of the image                */
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
 *
 */

#undef LOCAL_DEBUG
#undef DO_CLOCKING

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
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
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <ctype.h>

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif

#include "asimage.h"
#include "import.h"
#include "export.h"
#include "imencdec.h"
#include "ascmap.h"

/***********************************************************************************/
/* reduced colormap building code :                                                */
/***********************************************************************************/
static inline ASMappedColor *new_mapped_color( CARD32 red, CARD32 green, CARD32 blue, CARD32 indexed )
{
	register ASMappedColor *pnew = malloc( sizeof( ASMappedColor ));
	if( pnew != NULL )
	{
		pnew->red   = INDEX_UNSHIFT_RED  (red) ;
		pnew->green = INDEX_UNSHIFT_GREEN(green) ;
		pnew->blue  = INDEX_UNSHIFT_BLUE (blue) ;
		pnew->indexed = indexed ;
		pnew->count = 1 ;
		pnew->cmap_idx = -1 ;
		pnew->next = NULL ;
/*LOCAL_DEBUG_OUT( "indexed color added: 0x%X(%d): #%2.2X%2.2X%2.2X", indexed, indexed, red, green, blue );*/
	}
	return pnew;
}

void
add_index_color( ASSortedColorHash *index, CARD32 indexed, unsigned int slot, CARD32 red, CARD32 green, CARD32 blue )
{
	ASSortedColorBucket *stack ;
	ASMappedColor **pnext ;

	stack = &(index->buckets[slot]);
	pnext = &(stack->head);

	++(stack->count);

	if( stack->tail )
	{
		if( indexed == stack->tail->indexed )
		{
			++(stack->tail->count);
			return;
		}else if( indexed > stack->tail->indexed )
			pnext = &(stack->tail);
	}
	while( *pnext )
	{
		register ASMappedColor *pelem = *pnext ;/* to avoid double redirection */
		if( pelem->indexed == indexed )
		{
			++(pelem->count);
			return ;
		}else if( pelem->indexed > indexed )
		{
			register ASMappedColor *pnew = new_mapped_color( red, green, blue, indexed );
			if( pnew )
			{
				++(index->count_unique);
				pnew->next = pelem ;
				*pnext = pnew ;
				return;
			}
		}
		pnext = &(pelem->next);
	}
	/* we want to avoid memory overflow : */
	if( (*pnext = new_mapped_color( red, green, blue, indexed )) != NULL )
	{
		stack->tail = (*pnext);
		++(index->count_unique);
	}
}

void destroy_colorhash( ASSortedColorHash *index, Bool reusable )
{
	if( index )
	{
		int i ;
		for( i = 0 ; i < index->buckets_num ; i++ )
			while( index->buckets[i].head )
			{
				ASMappedColor *to_free = index->buckets[i].head;
				index->buckets[i].head = to_free->next ;
				free( to_free );
			}
		if( !reusable )
		{
			free( index->buckets );
			free( index );
		}
	}
}

#ifdef LOCAL_DEBUG
void
check_colorindex_counts( ASSortedColorHash *index )
{
	int i ;
	int count_unique = 0;

	for( i = 0 ; i < index->buckets_num ; i++ )
	{
		register ASMappedColor *pelem = index->buckets[i].head ;
		int row_count = 0 ;
		while( pelem != NULL )
		{
			count_unique++ ;
			if( pelem->cmap_idx < 0 )
				row_count += pelem->count ;
			pelem = pelem->next ;
		}
		if( row_count != index->buckets[i].count )
			fprintf( stderr, "bucket %d counts-> %d : %d\n", i, row_count, index->buckets[i].count );
	}
	fprintf( stderr, "total unique-> %d : %d\n", count_unique, index->count_unique );

}
#endif

void
fix_colorindex_shortcuts( ASSortedColorHash *index )
{
	int i ;
	int last_good = -1, next_good = -1;

	index->last_found = -1 ;

	for( i = 0 ; i < index->buckets_num ; i++ )
	{
		register ASMappedColor **pelem = &(index->buckets[i].head) ;
		register ASMappedColor **tail = &(index->buckets[i].tail) ;
		while( *pelem != NULL )
		{
			if( (*pelem)->cmap_idx < 0 )
			{
				ASMappedColor *to_free = *pelem ;
				*pelem = (*pelem)->next ;
				free( to_free );
			}else
			{
				*tail = *pelem ;
				pelem = &((*pelem)->next);
			}
		}
	}
	for( i = 0 ; i < index->buckets_num ; i++ )
	{
		if( next_good < 0 )
		{
			for( next_good = i ; next_good < index->buckets_num ; next_good++ )
				if( index->buckets[next_good].head )
					break;
			if( next_good >= index->buckets_num )
				next_good = last_good ;
		}
		if( index->buckets[i].head )
		{
			last_good = i;
			next_good = -1;
		}else
		{
			if( last_good < 0 || ( i-last_good >= next_good-i && i < next_good ) )
				index->buckets[i].good_offset = next_good-i ;
			else
				index->buckets[i].good_offset = last_good-i ;
		}
	}
}



static inline void
add_colormap_item( register ASColormapEntry *pentry, ASMappedColor *pelem, int cmap_idx )
{
	pentry->red   = pelem->red ;
	pentry->green = pelem->green ;
	pentry->blue  = pelem->blue ;
	pelem->cmap_idx = cmap_idx ;
LOCAL_DEBUG_OUT( "colormap entry added: %d: #%2.2X%2.2X%2.2X",cmap_idx, pelem->red, pelem->green, pelem->blue );
}

unsigned int
add_colormap_items( ASSortedColorHash *index, unsigned int start, unsigned int stop, unsigned int quota, unsigned int base, ASColormapEntry *entries )
{
	int cmap_idx = 0 ;
	unsigned int i ;
	if( quota >= index->count_unique )
	{
		for( i = start ; i < stop ; i++ )
		{
			register ASMappedColor *pelem = index->buckets[i].head ;
			while ( pelem != NULL )
			{
				add_colormap_item( &(entries[cmap_idx]), pelem, base++ );
				index->buckets[i].count -= pelem->count ;
				++cmap_idx ;
				pelem = pelem->next ;
			}
		}
	}else
	{
		int total = 0 ;
		int subcount = 0 ;
		ASMappedColor *best = NULL ;
		int best_slot = start;
		for( i = start ; i <= stop ; i++ )
			total += index->buckets[i].count ;

		for( i = start ; i <= stop ; i++ )
		{
			register ASMappedColor *pelem = index->buckets[i].head ;
			while ( pelem != NULL /*&& cmap_idx < quota*/ )
			{
				if( pelem->cmap_idx < 0 )
				{
					if( best == NULL )
					{
						best = pelem ;
						best_slot = i ;
					}else if( best->count < pelem->count )
					{
						best = pelem ;
						best_slot = i ;
					}
					else if( best->count == pelem->count &&
						     subcount >= (total>>2) && subcount <= (total>>1)*3 )
					{
						best = pelem ;
						best_slot = i ;
					}
					subcount += pelem->count*quota ;
LOCAL_DEBUG_OUT( "count = %d subtotal = %d, quota = %d, idx = %d, i = %d, total = %d", pelem->count, subcount, quota, cmap_idx, i, total );
					if( subcount >= total )
					{
						add_colormap_item( &(entries[cmap_idx]), best, base++ );
						index->buckets[best_slot].count -= best->count ;
						++cmap_idx ;
						subcount -= total ;
						best = NULL ;
					}
				}
				pelem = pelem->next ;
			}
		}
	}
	return cmap_idx ;
}

ASColormap *
color_hash2colormap( ASColormap *cmap, unsigned int max_colors )
{
	unsigned int cmap_idx = 0 ;
	int i ;
	ASSortedColorHash *index = NULL ;

	if( cmap == NULL || cmap->hash == NULL )
		return NULL;

	index = cmap->hash ;
	cmap->count = MIN(max_colors,index->count_unique);
	cmap->entries = safemalloc( cmap->count*sizeof( ASColormapEntry) );
	/* now lets go ahead and populate colormap : */
	if( index->count_unique <= max_colors )
	{
		add_colormap_items( index, 0, index->buckets_num, index->count_unique, 0, cmap->entries);
	}else
	while( cmap_idx < max_colors )
	{
		long total = 0 ;
		long subcount = 0 ;
		int start_slot = 0 ;
		int quota = max_colors-cmap_idx ;

		for( i = 0 ; i < index->buckets_num ; i++ )
			total += index->buckets[i].count ;

		for( i = 0 ; i < index->buckets_num ; i++ )
		{
			subcount += index->buckets[i].count*quota ;
LOCAL_DEBUG_OUT( "count = %d, subtotal = %ld, to_add = %ld, idx = %d, i = %d, total = %ld", index->buckets[i].count, subcount, subcount/total, cmap_idx, i, total );
			if( subcount >= total )
			{	/* we need to add subcount/index->count items */
				int to_add = subcount/total ;
				if( i == index->buckets_num-1 && to_add < (int)max_colors-(int)cmap_idx )
					to_add = max_colors-cmap_idx;
				cmap_idx += add_colormap_items( index, start_slot, i, to_add, cmap_idx, &(cmap->entries[cmap_idx]));
				subcount %= total;
				start_slot = i+1;
			}
		}
		if( quota == (int)max_colors-(int)cmap_idx )
			break;
	}
	fix_colorindex_shortcuts( index );
	return cmap;
}

void destroy_colormap( ASColormap *cmap, Bool reusable )
{
	if( cmap )
	{
		if( cmap->entries )
			free( cmap->entries );
		if( cmap->hash )
			destroy_colorhash( cmap->hash, False );
		if( !reusable )
			free( cmap );
	}
}

int
get_color_index( ASSortedColorHash *index, CARD32 indexed, unsigned int slot )
{
	ASSortedColorBucket *stack ;
	ASMappedColor *pnext, *lesser ;
	int offset ;

	if( index->last_found == indexed )
		return index->last_idx;
	index->last_found = indexed ;

LOCAL_DEBUG_OUT( "index = %lX(%ld), slot = %d, offset = %d", indexed, indexed, slot, index->buckets[slot].good_offset );
	if( (offset = index->buckets[slot].good_offset) != 0 )
		slot += offset ;
	stack = &(index->buckets[slot]);
LOCAL_DEBUG_OUT( "first_good = %lX(%ld), last_good = %lX(%ld)", stack->head->indexed, stack->head->indexed, stack->tail->indexed, stack->tail->indexed );
	if( offset < 0 || stack->tail->indexed <= indexed )
		return (index->last_idx=stack->tail->cmap_idx);
	if( offset > 0 || stack->head->indexed >= indexed )
		return (index->last_idx=stack->head->cmap_idx);

	lesser = stack->head ;
	for( pnext = lesser; pnext != NULL ; pnext = pnext->next )
	{
LOCAL_DEBUG_OUT( "lesser = %lX(%ld), pnext = %lX(%ld)", lesser->indexed, lesser->indexed, pnext->indexed, pnext->indexed );
			if( pnext->indexed >= indexed )
			{
				index->last_idx = ( pnext->indexed-indexed > indexed-lesser->indexed )?
						lesser->cmap_idx : pnext->cmap_idx ;
				return index->last_idx;
			}
			lesser = pnext ;
	}
	return stack->tail->cmap_idx;
}

int *
colormap_asimage( ASImage *im, ASColormap *cmap, unsigned int max_colors, unsigned int dither, int opaque_threshold )
{
	int *mapped_im = NULL;
	int buckets_num  = MAX_COLOR_BUCKETS;
	ASImageDecoder *imdec ;
	CARD32 *a, *r, *g, *b ;
	START_TIME(started);

	int *dst ;
	unsigned int y ;
	register int x ;

	if( im == NULL || cmap == NULL || im->width == 0 )
		return NULL;

	if((imdec = start_image_decoding( NULL /* default visual */ , im,
		                              SCL_DO_ALL, 0, 0, im->width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		return NULL;
	}

#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
print_asimage( im, ASFLAGS_EVERYTHING, __FUNCTION__, __LINE__ );
#endif

    if( max_colors == 0 )
		max_colors = 256 ;
	if( dither == -1 )
		dither = 4 ;
	else if( dither >= 8 )
		dither = 7 ;
	switch( dither )
	{
		case 0 :
		case 1 :
		case 2 : buckets_num = 4096 ;
		    break ;
		case 3 :
		case 4 : buckets_num = 1024 ;
		    break ;
		case 5 :
		case 6 : buckets_num = 64 ;
		    break ;
		case 7 : buckets_num = 8 ;
		    break ;
	}

	dst = mapped_im = safemalloc( im->width*im->height*sizeof(int));
	memset(cmap, 0x00, sizeof(ASColormap));
	cmap->hash = safecalloc( 1, sizeof(ASSortedColorHash) );
	cmap->hash->buckets = safecalloc( buckets_num, sizeof( ASSortedColorBucket ) );
	cmap->hash->buckets_num = buckets_num ;

	a = imdec->buffer.alpha ;
	r = imdec->buffer.red ;
	g = imdec->buffer.green ;
	b = imdec->buffer.blue ;

	for( y = 0 ; y < im->height ; y++ )
	{
		int red = 0, green = 0, blue = 0;
		int im_width = im->width ;
		imdec->decode_image_scanline( imdec );
		if( opaque_threshold > 0 && !cmap->has_opaque)
		{
			x = im->width ;
			while( --x >= 0  )
			  	if( a[x] != 0x00FF )
				{
					cmap->has_opaque = True;
					break;
				}
		}
		switch( dither )
		{
			case 0 :
				for( x = 0; x < im_width ; x++ )
				{
					if( (int)a[x] < opaque_threshold )	dst[x] = -1 ;
					else
					{
						red   = INDEX_SHIFT_RED(r[x]);
						green = INDEX_SHIFT_GREEN(g[x]);
						blue  = INDEX_SHIFT_BLUE(b[x]);
						dst[x] = MAKE_INDEXED_COLOR24(red,green,blue);
						add_index_color( cmap->hash, dst[x], ((dst[x]>>12)&0x0FFF), red, green, blue);
					}
				}
			    break ;
			case 1 :
				for( x = 0; x < im_width ; x++ )
				{
					if( (int)a[x] < opaque_threshold )	dst[x] = -1 ;
					else
					{
						red   = INDEX_SHIFT_RED(r[x]);
						green = INDEX_SHIFT_GREEN(g[x]);
						blue  = INDEX_SHIFT_BLUE(b[x]);
						dst[x] = MAKE_INDEXED_COLOR21(red,green,blue);
						add_index_color( cmap->hash, dst[x], ((dst[x]>>12)&0x0FFF), red, green, blue);
					}
				}
			    break ;
			case 2 :                           /* 666 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR18(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>12)&0x0FFF), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)&0x01 ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x01 ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x01 ;
					}
				}
			    break ;
			case 3 :                           /* 555 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
                        LOCAL_DEBUG_OUT( "alpha(%d,%d) = %ld, threshold = %d", x, y, a[x], opaque_threshold );
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR15(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>14)&0x03FF), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)    &0x03 ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x03 ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x03 ;
					}
				}
			    break ;
			case 4 :                           /* 444 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR12(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>14)&0x3FF), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)    &0x07 ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x07 ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x07 ;
					}
				}
			    break ;
			case 5 :                           /* 333 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR9(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>18)&0x03F), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)&0x0F ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x0F ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x0F ;
					}
				}
			    break ;
			case 6 :                           /* 222 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR6(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>18)&0x03F), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)&0x01F ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x01F ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x01F ;
					}
				}
			    break ;
			case 7 :                           /* 111 */
				{
					for( x = 0 ; x < im_width ; ++x )
					{
						red   = INDEX_SHIFT_RED  ((red  +r[x]>255)?255:red+r[x]);
						green = INDEX_SHIFT_GREEN((green+g[x]>255)?255:green+g[x]);
						blue  = INDEX_SHIFT_BLUE ((blue +b[x]>255)?255:blue+b[x]);
						if( (int)a[x] < opaque_threshold )
							dst[x] = -1 ;
						else
						{
							dst[x] = MAKE_INDEXED_COLOR3(red,green,blue);
							add_index_color( cmap->hash, dst[x], ((dst[x]>>21)&0x07), red, green, blue);
						}
						red   = INDEX_UNESHIFT_RED(red,1)&0x03F ;
						green = INDEX_UNESHIFT_GREEN(green,1)&0x03F ;
						blue  = INDEX_UNESHIFT_BLUE(blue,1)  &0x03F ;
					}
				}
			    break ;
		}
		dst += im_width ;
	}
	stop_image_decoding( &imdec );
	SHOW_TIME("color indexing",started);

#ifdef LOCAL_DEBUG
check_colorindex_counts( cmap->hash );
#endif

	LOCAL_DEBUG_OUT("building colormap%s","");
	color_hash2colormap( cmap, max_colors );
	SHOW_TIME("colormap calculation",started);

#ifdef LOCAL_DEBUG
check_colorindex_counts( cmap->hash );
#endif

	dst = mapped_im ;
	for( y = 0 ; y < im->height ; ++y )
	{
		switch( dither )
		{
			case 0 :
			case 1 :
			case 2 :
				for( x = 0 ; x < (int)im->width ; ++x )
                {
                    LOCAL_DEBUG_OUT( "Mapping color at (%dx%d) indexed = %X", x, y, dst[x] );
					if( dst[x] >= 0 )
						dst[x] = get_color_index( cmap->hash, dst[x], ((dst[x]>>12)&0x0FFF));
					else
						dst[x] = cmap->count ;
                }
				break;
			case 3 :
			case 4 :
				for( x = 0 ; x < (int)im->width ; ++x )
				{
                    LOCAL_DEBUG_OUT( "Mapping color at (%dx%d) indexed = %X", x, y, dst[x] );
                    if( dst[x] >= 0 )
						dst[x] = get_color_index( cmap->hash, dst[x], ((dst[x]>>14)&0x03FF));
					else
						dst[x] = cmap->count ;
				}
				break;
			case 5 :
			case 6 :
				for( x = 0 ; x < (int)im->width ; ++x )
					if( dst[x] >= 0 )
						dst[x] = get_color_index( cmap->hash, dst[x], ((dst[x]>>18)&0x03F));
					else
						dst[x] = cmap->count ;
			    break ;
			case 7 :
				for( x = 0 ; x < (int)im->width ; ++x )
					if( dst[x] >= 0 )
						dst[x] = get_color_index( cmap->hash, dst[x], ((dst[x]>>21)&0x007));
					else
						dst[x] = cmap->count ;
			    break ;
		}
		dst += im->width ;
	}

	return mapped_im ;
}

