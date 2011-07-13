/*
 * Copyright (c) 2004 Valeriy Onuchin <Valeri dot Onoutchine at cern dot ch>
 * Copyright (c) 2000-2004 Sasha Vasko <sasha at aftercode.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License.
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

#define LOCAL_DEBUG
#undef DO_CLOCKING
#ifndef NO_DEBUG_OUTPUT
#undef DEBUG_RECTS
#undef DEBUG_RECTS2
#endif

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif


#define USE_64BIT_FPU

#include <string.h>
#ifdef DO_CLOCKING
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

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asvisual.h"
#include "scanline.h"
#include "blender.h"
#include "asimage.h"
#include "ascmap.h"

static ASVisual __as_dummy_asvisual = {0};
static ASVisual *__as_default_asvisual = &__as_dummy_asvisual ;

/* these are internal library things - user should not mess with it ! */
ASVisual *_set_default_asvisual( ASVisual *new_v )
{
	ASVisual *old_v = __as_default_asvisual ;
	__as_default_asvisual = new_v?new_v:&__as_dummy_asvisual ;
/* This should be done in application - we should not meddle with other lib's stuff! 
#if HAVE_AFTERBASE_FLAG	
	if (new_v && new_v->dpy && !get_current_X_display (new_v->dpy))
		set_current_X_display (new_v->dpy);
#endif
*/
	return old_v;
}

ASVisual *get_default_asvisual()
{
	return __as_default_asvisual?__as_default_asvisual:&__as_dummy_asvisual;
}

/* internal buffer used for compression/decompression */

static CARD8 *__as_compression_buffer = NULL ;
static size_t __as_compression_buffer_len = 0;   /* allocated size */

static inline CARD8* get_compression_buffer( size_t size )
{
	if( size > __as_compression_buffer_len )
 		__as_compression_buffer_len = (size+1023)&(~0x03FF) ;
	return (__as_compression_buffer = realloc( __as_compression_buffer, __as_compression_buffer_len ));
}

static inline void release_compression_buffer( CARD8 *ptr )
{
	/* do nothing so far */
}



#ifdef TRACK_ASIMAGES
static ASHashTable *__as_image_registry = NULL ;
#endif

#ifdef HAVE_MMX
Bool asimage_use_mmx = True;
#else
Bool asimage_use_mmx = False;
#endif

/* *********************   ASImage  ************************************/
void
asimage_init (ASImage * im, Bool free_resources)
{
	if (im != NULL)
	{
		if (free_resources)
		{
			register int i ;
			for( i = im->height*4-1 ; i>= 0 ; --i )
				if( im->red[i] != 0 )
					forget_data( NULL, im->red[i] );
			if( im->red )
				free(im->red);
#ifndef X_DISPLAY_MISSING
			if( im->alt.ximage )
				XDestroyImage( im->alt.ximage );
			if( im->alt.mask_ximage )
				XDestroyImage( im->alt.mask_ximage );
#endif
			if( im->alt.argb32 )
				free( im->alt.argb32 );
			if( im->alt.vector )
				free( im->alt.vector );
			if( im->name ) 
				free( im->name );
		}
		memset (im, 0x00, sizeof (ASImage));
		im->magic = MAGIC_ASIMAGE ;
		im->back_color = ARGB32_DEFAULT_BACK_COLOR ;
	}
}

void
flush_asimage_cache( ASImage *im )
{
#ifndef X_DISPLAY_MISSING
	if( im->alt.ximage )
    {
		XDestroyImage( im->alt.ximage );
        im->alt.ximage = NULL ;
    }
	if( im->alt.mask_ximage )
    {
		XDestroyImage( im->alt.mask_ximage );
        im->alt.mask_ximage = NULL ;
    }
#endif
}

static void
alloc_asimage_channels ( ASImage *im )
{
	/* we want result to be 32bit aligned and padded */
	im->red = safecalloc (1, sizeof (ASStorageID) * im->height * 4);
	LOCAL_DEBUG_OUT( "allocated %p for channels of the image %p", im->red, im );
	if( im->red == NULL )
	{
		show_error( "Insufficient memory to create image %dx%d!", im->width, im->height );
		return ;
	}

	im->green = im->red+im->height;
	im->blue = 	im->red+(im->height*2);
	im->alpha = im->red+(im->height*3);
	im->channels[IC_RED] = im->red ;
	im->channels[IC_GREEN] = im->green ;
	im->channels[IC_BLUE] = im->blue ;
	im->channels[IC_ALPHA] = im->alpha ;
}

void
asimage_start (ASImage * im, unsigned int width, unsigned int height, unsigned int compression)
{
	if (im)
	{
		asimage_init (im, True);
		im->height = height;
		im->width = width;
		
		alloc_asimage_channels( im );

		if( compression == 0 ) 
			set_flags( im->flags, ASIM_NO_COMPRESSION );
	}
}

Bool
asimage_replace (ASImage *im, ASImage *from)
{
	if ( im && from && im != from )
		if( im->magic == MAGIC_ASIMAGE && from->magic == MAGIC_ASIMAGE && from->imageman == NULL )
		{
			int ref_count = im->ref_count ;
			ASImageManager *imageman = im->imageman ;
			char *name = im->name ;
			ASFlagType  saved_flags = im->flags & (ASIM_NAME_IS_FILENAME|ASIM_NO_COMPRESSION) ;

			im->name = NULL ; 
			asimage_init (im, True);

			memcpy( im, from, sizeof(ASImage) );
			/* Assume : from->name == NULL as from->imageman == NULL (see above ) */
			memset( from, 0x00, sizeof(ASImage) );		

			im->ref_count = ref_count ; 
			im->imageman = imageman ;
			im->name = name ;
			set_flags( im->flags, saved_flags );

			return True ;
		}
	return False;
}


static ASImage* 
check_created_asimage( ASImage *im, unsigned int width, unsigned int height )
{
	if( im->width == 0 || im->height == 0 )
	{
		free( im );
		im = NULL ;
#ifdef TRACK_ASIMAGES
        show_error( "failed to create ASImage of size %dx%d", width, height );
#endif
    }else
    {
#ifdef TRACK_ASIMAGES
        show_progress( "created ASImage %p of size %dx%d (%s compressed )", im,  
						width, height, (get_flags(im->flags, ASIM_NO_COMPRESSION)?" no":"") );
        if( __as_image_registry == NULL )
            __as_image_registry = create_ashash( 0, pointer_hash_value, NULL, NULL );
        add_hash_item( __as_image_registry, AS_HASHABLE(im), im );
#endif
    }
	return im ;
}

ASImage *
create_asimage( unsigned int width, unsigned int height, unsigned int compression)
{
	ASImage *im = safecalloc( 1, sizeof(ASImage) );
	asimage_start( im, width, height, compression );
	return check_created_asimage( im, width, height );
}

void
destroy_asimage( ASImage **im )
{
	if( im )
	{

		if( *im && !AS_ASSERT_NOTVAL((*im)->imageman,NULL))
		{
#ifdef TRACK_ASIMAGES
            show_progress( "destroying ASImage %p of size %dx%d", *im, (*im)->width, (*im)->height );
            remove_hash_item( __as_image_registry, AS_HASHABLE(*im), NULL, False );
#endif
			asimage_init( *im, True );
			(*im)->magic = 0;
			free( *im );
			*im = NULL ;
		}else if( *im )
		{
	        show_error( "Failed to destroy ASImage %p:", *im );
			print_asimage_func (AS_HASHABLE(*im));
		}

	}
}

void print_asimage_func (ASHashableValue value)
{
    ASImage *im = (ASImage*)value ;
    if( im && im->magic == MAGIC_ASIMAGE )
    {
        unsigned int k;
        unsigned int red_mem = 0, green_mem = 0, blue_mem = 0, alpha_mem = 0;
        unsigned int red_count = 0, green_count = 0, blue_count = 0, alpha_count = 0;
		ASStorageSlot slot ;

        fprintf( stderr,"\n\tASImage[%p].size = %dx%d;\n",  im, im->width, im->height );
        fprintf( stderr,"\tASImage[%p].back_color = 0x%lX;\n", im, (long)im->back_color );
        fprintf( stderr,"\t\tASImage[%p].alt.ximage = %p;\n", im, im->alt.ximage );
        if( im->alt.ximage )
        {
            fprintf( stderr,"\t\t\tASImage[%p].alt.ximage.bytes_per_line = %d;\n", im, im->alt.ximage->bytes_per_line);
            fprintf( stderr,"\t\t\tASImage[%p].alt.ximage.size = %dx%d;\n", im, im->alt.ximage->width, im->alt.ximage->height);
        }
        fprintf( stderr,"\t\tASImage[%p].alt.mask_ximage = %p;\n", im, im->alt.mask_ximage);
        if( im->alt.mask_ximage )
        {
            fprintf( stderr,"\t\t\tASImage[%p].alt.mask_ximage.bytes_per_line = %d;\n", im, im->alt.mask_ximage->bytes_per_line);
            fprintf( stderr,"\t\t\tASImage[%p].alt.mask_ximage.size = %dx%d;\n", im, im->alt.mask_ximage->width, im->alt.mask_ximage->height);
        }
        fprintf( stderr,"\t\tASImage[%p].alt.argb32 = %p;\n", im, im->alt.argb32 );
        fprintf( stderr,"\t\tASImage[%p].alt.vector = %p;\n", im, im->alt.vector );
        fprintf( stderr,"\tASImage[%p].imageman = %p;\n", im, im->imageman );
        fprintf( stderr,"\tASImage[%p].ref_count = %d;\n", im, im->ref_count );
        fprintf( stderr,"\tASImage[%p].name = \"%s\";\n", im, im->name );
        fprintf( stderr,"\tASImage[%p].flags = 0x%lX;\n", im, im->flags );

        for( k = 0 ; k < im->height ; k++ )
    	{
			if( im->red[k] ) 
				if( query_storage_slot(NULL, im->red[k], &slot ) )
				{	
			 		++red_count;	
					red_mem += slot.size ;
				}
			if( im->green[k] ) 
				if( query_storage_slot(NULL, im->green[k], &slot ) )
				{	
			 		++green_count;	
					green_mem += slot.size ;
				}
			if( im->blue[k] ) 
				if( query_storage_slot(NULL, im->blue[k], &slot ) )
				{	
			 		++blue_count;	
					blue_mem += slot.size ;
				}
			if( im->alpha[k] ) 
				if( query_storage_slot(NULL, im->alpha[k], &slot ) )
				{	
			 		++alpha_count;	
					alpha_mem += slot.size ;
				}
        }

        fprintf( stderr,"\tASImage[%p].uncompressed_size = %d;\n", im, im->width*red_count +
                                                                    im->width*green_count +
                                                                    im->width*blue_count +
                                                                    im->width*alpha_count );
        fprintf( stderr,"\tASImage[%p].compressed_size = %d;\n",   im, red_mem + green_mem +blue_mem + alpha_mem );
        fprintf( stderr,"\t\tASImage[%p].channel[red].lines_count = %d;\n", im, red_count );
        fprintf( stderr,"\t\tASImage[%p].channel[red].memory_used = %d;\n", im, red_mem );
        fprintf( stderr,"\t\tASImage[%p].channel[green].lines_count = %d;\n", im, green_count );
        fprintf( stderr,"\t\tASImage[%p].channel[green].memory_used = %d;\n", im, green_mem );
        fprintf( stderr,"\t\tASImage[%p].channel[blue].lines_count = %d;\n", im, blue_count );
        fprintf( stderr,"\t\tASImage[%p].channel[blue].memory_used = %d;\n", im, blue_mem );
        fprintf( stderr,"\t\tASImage[%p].channel[alpha].lines_count = %d;\n", im, alpha_count );
        fprintf( stderr,"\t\tASImage[%p].channel[alpha].memory_used = %d;\n", im, alpha_mem );
    }
}

void
print_asimage_registry()
{
#ifdef TRACK_ASIMAGES
    print_ashash( __as_image_registry, print_asimage_func );
#endif
}

void
purge_asimage_registry()
{
#ifdef TRACK_ASIMAGES
    if( __as_image_registry )
        destroy_ashash( &__as_image_registry );
#endif
}

/* ******************** ASImageManager ****************************/
static void
asimage_destroy (ASHashableValue value, void *data)
{
	if( data )
	{
		ASImage *im = (ASImage*)data ;
		if( im != NULL )
		{
			if( AS_ASSERT_NOTVAL(im->magic, MAGIC_ASIMAGE) )
				im = NULL ;
			else
				im->imageman = NULL ;
		}
		if( im == NULL || (char*)value != im->name ) 
			free( (char*)value );/* name */
		destroy_asimage( &im );
	}
}

ASImageManager *create_image_manager( struct ASImageManager *reusable_memory, double gamma, ... )
{
	ASImageManager *imman = reusable_memory ;
	int i ;
	va_list ap;

	if( imman == NULL )
		imman = safecalloc( 1, sizeof(ASImageManager));
	else
		memset( imman, 0x00, sizeof(ASImageManager));

	va_start (ap, gamma);
	for( i = 0 ; i < MAX_SEARCH_PATHS ; i++ )
	{
		char *path = va_arg(ap,char*);
		if( path == NULL )
			break;
		imman->search_path[i] = mystrdup( path );
	}
	va_end (ap);

	imman->search_path[MAX_SEARCH_PATHS] = NULL ;
	imman->gamma = gamma ;

	imman->image_hash = create_ashash( 7, string_hash_value, string_compare, asimage_destroy );

	return imman;
}

void
destroy_image_manager( struct ASImageManager *imman, Bool reusable )
{
	if( imman )
	{
		int i = MAX_SEARCH_PATHS;
		destroy_ashash( &(imman->image_hash) );
		while( --i >= 0 )
			if(imman->search_path[i])
				free( imman->search_path[i] );

		if( !reusable )
			free( imman );
		else
			memset( imman, 0x00, sizeof(ASImageManager));
	}
}

Bool
store_asimage( ASImageManager* imageman, ASImage *im, const char *name )
{
	Bool res = False ;
	if( !AS_ASSERT(im) )
		if( AS_ASSERT_NOTVAL(im->magic, MAGIC_ASIMAGE) )
			im = NULL ;
	if( !AS_ASSERT(imageman) && !AS_ASSERT(im) && !AS_ASSERT((char*)name) )
	{
		if( im->imageman == NULL )
		{
			int hash_res ;
			char *stored_name = mystrdup( name );
			if( im->name ) 
				free( im->name );
			im->name = stored_name ;
			hash_res = add_hash_item( imageman->image_hash, AS_HASHABLE(im->name), im);
			res = ( hash_res == ASH_Success);
			if( !res )
			{
				free( im->name );
				im->name = NULL ;
			}else
			{
				im->imageman = imageman ;
				im->ref_count = 1 ;
			}
		}
	}
	return res ;
}

inline ASImage *
query_asimage( ASImageManager* imageman, const char *name )
{
	ASImage *im = NULL ;
	if( !AS_ASSERT(imageman) && !AS_ASSERT(name) )
	{
		ASHashData hdata = {0} ;
		if( get_hash_item( imageman->image_hash, AS_HASHABLE((char*)name), &hdata.vptr) == ASH_Success )
		{
			im = hdata.vptr ;
			if( im->magic != MAGIC_ASIMAGE )
				im = NULL ;
        }
	}
	return im;
}

ASImage *
fetch_asimage( ASImageManager* imageman, const char *name )
{
    ASImage *im = query_asimage( imageman, name );
    if( im )
	{
        im->ref_count++ ;
	}
	return im;
}


ASImage *
dup_asimage( ASImage* im )
{
	if( !AS_ASSERT(im) )
		if( AS_ASSERT_NOTVAL(im->magic,MAGIC_ASIMAGE) )
		{
			im = NULL ;
			show_error( "ASImage %p has invalid magic number - discarding!", im );
		}

	if( !AS_ASSERT(im) && !AS_ASSERT(im->imageman) )
	{
/*		fprintf( stderr, __FUNCTION__" on image %p ref_count = %d\n", im, im->ref_count ); */
		im->ref_count++ ;
		return im;
	}else if( im ) 
	{
		show_debug( __FILE__, "dup_asimage", __LINE__, "Attempt to duplicate ASImage %p that is not tracked by any image manager!", im );
	}
	return NULL ;
}

inline int
release_asimage( ASImage *im )
{
	int res = -1 ;
	if( !AS_ASSERT(im) )
	{
		if( im->magic == MAGIC_ASIMAGE )
		{
			if( --(im->ref_count) <= 0 )
			{
				ASImageManager *imman = im->imageman ;
				if( !AS_ASSERT(imman) )
                    if( remove_hash_item(imman->image_hash, (ASHashableValue)(char*)im->name, NULL, True) != ASH_Success )
                        destroy_asimage( &im );
			}else
				res = im->ref_count ;
		}
	}
	return res ;
}

void
forget_asimage( ASImage *im )
{
	if( !AS_ASSERT(im) )
	{
		if( im->magic == MAGIC_ASIMAGE )
		{
			ASImageManager *imman = im->imageman ;
			if( !AS_ASSERT(imman) )
				remove_hash_item(imman->image_hash, (ASHashableValue)(char*)im->name, NULL, False);
            im->ref_count = 0;
            im->imageman = NULL;
		}
	}
}

void
relocate_asimage( ASImageManager* to_imageman, ASImage *im )
{
	if( !AS_ASSERT(im) )
	{
		if( im->magic == MAGIC_ASIMAGE )
		{
			ASImageManager *imman = im->imageman ;
			int ref_count = im->ref_count ; 
			if( imman != NULL )
			{
				remove_hash_item(imman->image_hash, (ASHashableValue)(char*)im->name, NULL, False);
	            im->ref_count = 0;
    	        im->imageman = NULL;
			}
			if( to_imageman != NULL ) 
			{
				if( add_hash_item( to_imageman->image_hash, AS_HASHABLE(im->name), im) == ASH_Success ) 
				{
		            im->ref_count = ref_count < 1 ? 1: ref_count;
    		        im->imageman = to_imageman ; 
				}
			}
		}
	}
}

void
forget_asimage_name( ASImageManager *imman, const char *name )
{
    if( !AS_ASSERT(imman) && name != NULL )
	{
        remove_hash_item(imman->image_hash, AS_HASHABLE((char*)name), NULL, False);
    }
}

inline int
safe_asimage_destroy( ASImage *im )
{
	int res = -1 ;
	if( !AS_ASSERT(im) )
	{
		if( im->magic == MAGIC_ASIMAGE )
		{
			ASImageManager *imman = im->imageman ;
			if( imman != NULL )
			{
                res = --(im->ref_count) ;
                if( im->ref_count <= 0 )
					remove_hash_item(imman->image_hash, (ASHashableValue)(char*)im->name, NULL, True);
            }else
			{
				destroy_asimage( &im );
				res = -1 ;
			}
		}
	}
	return res ;
}

int
release_asimage_by_name( ASImageManager *imageman, char *name )
{
	int res = -1 ;
	ASImage *im = NULL ;
	if( !AS_ASSERT(imageman) && !AS_ASSERT(name) )
	{
		ASHashData hdata ;
		if( get_hash_item( imageman->image_hash, AS_HASHABLE((char*)name), &hdata.vptr) == ASH_Success )
		{
			im = hdata.vptr ;
			res = release_asimage( im );
		}
	}
	return res ;
}

void
print_asimage_manager(ASImageManager *imageman)
{
#ifdef TRACK_ASIMAGES
    print_ashash( imageman->image_hash, string_print );
#endif    
}

/* ******************** ASGradient ****************************/

void
destroy_asgradient( ASGradient **pgrad )
{
	if( pgrad && *pgrad )
	{
		if( (*pgrad)->color )
		{
			free( (*pgrad)->color );
			(*pgrad)->color = NULL ;
		}
		if( (*pgrad)->offset )
		{
			free( (*pgrad)->offset );
			(*pgrad)->offset = NULL ;
		}
		(*pgrad)->npoints = 0 ;
		free( *pgrad );
		*pgrad = NULL ;
	}

}

ASGradient *
flip_gradient( ASGradient *orig, int flip )
{
	ASGradient *grad ;
	int npoints ;
	int type ;
	Bool inverse_points = False ;

	flip &= FLIP_MASK ;
	if( orig == NULL || flip == 0 )
		return orig;

	grad = safecalloc( 1, sizeof(ASGradient));

	grad->npoints = npoints = orig->npoints ;
	type = orig->type ;
    grad->color = safemalloc( npoints*sizeof(ARGB32) );
    grad->offset = safemalloc( npoints*sizeof(double) );

	if( get_flags(flip, FLIP_VERTICAL) )
	{
		Bool upsidedown = get_flags(flip, FLIP_UPSIDEDOWN) ;
		switch(type)
		{
			case GRADIENT_Left2Right  :
				type = GRADIENT_Top2Bottom ; inverse_points = !upsidedown ;
				break;
			case GRADIENT_TopLeft2BottomRight :
				type = GRADIENT_BottomLeft2TopRight ; inverse_points = upsidedown ;
				break;
			case GRADIENT_Top2Bottom  :
				type = GRADIENT_Left2Right ; inverse_points = upsidedown ;
				break;
			case GRADIENT_BottomLeft2TopRight :
				type = GRADIENT_TopLeft2BottomRight ; inverse_points = !upsidedown ;
				break;
		}
	}else if( flip == FLIP_UPSIDEDOWN )
	{
		inverse_points = True ;
	}

	grad->type = type ;
	if( inverse_points )
    {
        register int i = 0, k = npoints;
        while( --k >= 0 )
        {
            grad->color[i] = orig->color[k] ;
            grad->offset[i] = 1.0 - orig->offset[k] ;
			++i ;
        }
    }else
	{
        register int i = npoints ;
        while( --i >= 0 )
        {
            grad->color[i] = orig->color[i] ;
            grad->offset[i] = orig->offset[i] ;
        }
    }
	return grad;
}

/* ******************** ASImageLayer ****************************/

void
init_image_layers( register ASImageLayer *l, int count )
{
	memset( l, 0x00, sizeof(ASImageLayer)*count );
	while( --count >= 0 )
	{
		l[count].merge_scanlines = alphablend_scanlines ;
/*		l[count].solid_color = ARGB32_DEFAULT_BACK_COLOR ; */
	}
}

ASImageLayer *
create_image_layers( int count )
{
	ASImageLayer *l = NULL;

	if( count > 0 )
	{
		l = safecalloc( count, sizeof(ASImageLayer) );
		init_image_layers( l, count );
	}
	return l;
}

void
destroy_image_layers( register ASImageLayer *l, int count, Bool reusable )
{
	if( l )
	{
		register int i = count;
		while( --i >= 0 )
		{
			if( l[i].im )
			{
				if( l[i].im->imageman )
					release_asimage( l[i].im );
				else
					destroy_asimage( &(l[i].im) );
			}
			if( l[i].bevel )
				free( l[i].bevel );
		}
		if( !reusable )
			free( l );
		else
			memset( l, 0x00, sizeof(ASImageLayer)*count );
	}
}



/* **********************************************************************/
/*  Compression/decompression 										   */
/* **********************************************************************/
size_t
asimage_add_line_mono (ASImage * im, ColorPart color, CARD8 value, unsigned int y)
{
   int colint = (int) color;
	if (AS_ASSERT(im) || colint <0 || color >= IC_NUM_CHANNELS )
		return 0;
	if (y >= im->height)
		return 0;
	
	if( im->channels[color][y] ) 
		forget_data( NULL, im->channels[color][y] ); 
	im->channels[color][y] = store_data( NULL, &value, 1, 0, 0);
	return im->width;
}

size_t
asimage_add_line (ASImage * im, ColorPart color, register CARD32 * data, unsigned int y)
{
   int colint = (int) color;
	if (AS_ASSERT(im) || colint <0 || color >= IC_NUM_CHANNELS )
		return 0;
	if (y >= im->height)
		return 0;
	if( im->channels[color][y] ) 
		forget_data( NULL, im->channels[color][y] ); 
	im->channels[color][y] = store_data( NULL, (CARD8*)data, im->width*4, ASStorage_RLEDiffCompress|ASStorage_32Bit, 0);
	return im->width;
}

size_t
asimage_add_line_bgra (ASImage * im, register CARD32 * data, unsigned int y)
{
	if (AS_ASSERT(im) )
		return 0;
	if (y >= im->height)
		return 0;
	if( im->channels[IC_ALPHA][y] ) 
		forget_data( NULL, im->channels[IC_ALPHA][y] ); 
	im->channels[IC_ALPHA][y] = store_data( NULL, (CARD8*)data, im->width*4, 
	                                        ASStorage_24BitShift|ASStorage_Masked|
											ASStorage_RLEDiffCompress|ASStorage_32Bit, 0);
	if( im->channels[IC_RED][y] ) 
		forget_data( NULL, im->channels[IC_RED][y] ); 
	im->channels[IC_RED][y] = store_data( NULL, (CARD8*)data, im->width*4, 
	                                        ASStorage_16BitShift|ASStorage_Masked|
											ASStorage_RLEDiffCompress|ASStorage_32Bit, 0);
	if( im->channels[IC_GREEN][y] ) 
		forget_data( NULL, im->channels[IC_GREEN][y] ); 
	im->channels[IC_GREEN][y] = store_data( NULL, (CARD8*)data, im->width*4, 
	                                        ASStorage_8BitShift|ASStorage_Masked|
											ASStorage_RLEDiffCompress|ASStorage_32Bit, 0);
	if( im->channels[IC_BLUE][y] ) 
		forget_data( NULL, im->channels[IC_BLUE][y] ); 
	im->channels[IC_BLUE][y] = store_data( NULL, (CARD8*)data, im->width*4, 
	                                        ASStorage_Masked|
											ASStorage_RLEDiffCompress|ASStorage_32Bit, 0);
	return im->width;
}

unsigned int
asimage_print_line (ASImage * im, ColorPart color, unsigned int y, unsigned long verbosity)
{
   int colint = (int) color;
	if (AS_ASSERT(im) || colint < 0 || color >= IC_NUM_CHANNELS )
		return 0;
	if (y >= im->height)
		return 0;
	
	return print_storage_slot(NULL, im->channels[color][y]);
}

void print_asimage( ASImage *im, int flags, char * func, int line )
{
	if( im )
	{
		register unsigned int k ;
		int total_mem = 0 ;
		fprintf( stderr, "%s:%d> printing ASImage %p.\n", func, line, im);
		for( k = 0 ; k < im->height ; k++ )
    	{
 			fprintf( stderr, "%s:%d> ******* %d *******\n", func, line, k );
			total_mem+=asimage_print_line( im, IC_RED  , k, flags );
			total_mem+=asimage_print_line( im, IC_GREEN, k, flags );
			total_mem+=asimage_print_line( im, IC_BLUE , k, flags );
            total_mem+=asimage_print_line( im, IC_ALPHA , k, flags );
        }
    	fprintf( stderr, "%s:%d> Total memory : %u - image size %dx%d ratio %d%%\n", func, line, total_mem, im->width, im->height, (total_mem*100)/(im->width*im->height*3) );
	}else
		fprintf( stderr, "%s:%d> Attempted to print NULL ASImage.\n", func, line);
}

void print_component( register CARD32 *data, int nonsense, int len );

int
asimage_decode_line (ASImage * im, ColorPart color, CARD32 * to_buf, unsigned int y, unsigned int skip, unsigned int out_width)
{
	ASStorageID id = im->channels[color][y];
	register int i = 0;
	/* that thing below is supposedly highly optimized : */
LOCAL_DEBUG_CALLER_OUT( "im->width = %d, color = %d, y = %d, skip = %d, out_width = %d", im->width, color, y, skip, out_width );

	if( id )
	{
		i = fetch_data32( NULL, id, to_buf, skip, out_width, 0, NULL);
        LOCAL_DEBUG_OUT( "decoded %d pixels", i );
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
        {
            int z = -1 ;
            while( ++z < i )
                fprintf( stderr, "%lX ", (unsigned long)to_buf[z] );
            fprintf( stderr, "\n");

        }
#endif
		return i;
	}
	return 0;
}

void
move_asimage_channel( ASImage *dst, int channel_dst, ASImage *src, int channel_src )
{
	if( !AS_ASSERT(dst) && !AS_ASSERT(src) && channel_src >= 0 && channel_src < IC_NUM_CHANNELS &&
		channel_dst >= 0 && channel_dst < IC_NUM_CHANNELS )
	{
		register int i = MIN(dst->height, src->height);
		register ASStorageID *dst_rows = dst->channels[channel_dst] ;
		register ASStorageID *src_rows = src->channels[channel_src] ;
		while( --i >= 0 )
		{
			if( dst_rows[i] )
				forget_data( NULL, dst_rows[i] );
			dst_rows[i] = src_rows[i] ;
			src_rows[i] = 0 ;
		}
	}
}


void
copy_asimage_channel( ASImage *dst, int channel_dst, ASImage *src, int channel_src )
{
	if( !AS_ASSERT(dst) && !AS_ASSERT(src) && channel_src >= 0 && channel_src < IC_NUM_CHANNELS &&
		channel_dst >= 0 && channel_dst < IC_NUM_CHANNELS )
	{
		register int i = MIN(dst->height, src->height);
		register ASStorageID *dst_rows = dst->channels[channel_dst] ;
		register ASStorageID *src_rows = src->channels[channel_src] ;
		LOCAL_DEBUG_OUT( "src = %p, dst = %p, dst->width = %d, src->width = %d", src, dst, dst->width, src->width );
		while( --i >= 0 )
		{
			if( dst_rows[i] )
				forget_data( NULL, dst_rows[i] );
			dst_rows[i] = dup_data( NULL, src_rows[i] );
		}
	}
}

void
copy_asimage_lines( ASImage *dst, unsigned int offset_dst,
                    ASImage *src, unsigned int offset_src,
					unsigned int nlines, ASFlagType filter )
{
	if( !AS_ASSERT(dst) && !AS_ASSERT(src) &&
		offset_src < src->height && offset_dst < dst->height )
	{
		int chan;

		if( offset_src+nlines > src->height )
			nlines = src->height - offset_src ;
		if( offset_dst+nlines > dst->height )
			nlines = dst->height - offset_dst ;

		for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
			if( get_flags( filter, 0x01<<chan ) )
			{
				register int i = -1;
				register ASStorageID *dst_rows = &(dst->channels[chan][offset_dst]) ;
				register ASStorageID *src_rows = &(src->channels[chan][offset_src]) ;
LOCAL_DEBUG_OUT( "copying %d lines of channel %d...", nlines, chan );
				while( ++i < (int)nlines )
				{
					if( dst_rows[i] )
						forget_data( NULL, dst_rows[i] );
					dst_rows[i] = dup_data( NULL, src_rows[i] );
				}
			}
#if 0
		for( i = 0 ; i < nlines ; ++i )
		{
			asimage_print_line( src, IC_ALPHA, i, (i==4)?VRB_EVERYTHING:VRB_LINE_SUMMARY );
			asimage_print_line( dst, IC_ALPHA, i, (i==4)?VRB_EVERYTHING:VRB_LINE_SUMMARY );
		}
#endif
	}
}

Bool
asimage_compare_line (ASImage *im, ColorPart color, CARD32 *to_buf, CARD32 *tmp, unsigned int y, Bool verbose)
{
	register unsigned int i;
	asimage_decode_line( im, color, tmp, y, 0, im->width );
	for( i = 0 ; i < im->width ; i++ )
		if( tmp[i] != to_buf[i] )
		{
			if( verbose )
				show_error( "line %d, component %d differ at offset %d ( 0x%lX(compresed) != 0x%lX(orig) )\n", y, color, i, (unsigned long)tmp[i], (unsigned long)to_buf[i] );
			return False ;
		}
	return True;
}

ASFlagType
get_asimage_chanmask( ASImage *im)
{
    ASFlagType mask = 0 ;
	int color ;

	if( !AS_ASSERT(im) )
		for( color = 0; color < IC_NUM_CHANNELS ; color++ )
		{
			register ASStorageID *chan = im->channels[color];
			register int y, height = im->height ;
			for( y = 0 ; y < height ; y++ )
				if( chan[y] )
				{
					set_flags( mask, 0x01<<color );
					break;
				}
		}
    return mask ;
}

int
check_asimage_alpha (ASVisual *asv, ASImage *im )
{
	int recomended_depth = 0 ;
	unsigned int            i;
	ASScanline     buf;

	if( asv == NULL )
		asv = get_default_asvisual();

	if (im == NULL)
		return 0;

	prepare_scanline( im->width, 0, &buf, asv->BGR_mode );
	buf.flags = SCL_DO_ALPHA ;
	for (i = 0; i < im->height; i++)
	{
		int count = asimage_decode_line (im, IC_ALPHA, buf.alpha, i, 0, buf.width);
		if( count < (int)buf.width )
		{
			if( ARGB32_ALPHA8(im->back_color) == 0 )
			{
				if( recomended_depth == 0 )
					recomended_depth = 1 ;
			}else if( ARGB32_ALPHA8(im->back_color) != 0xFF )
			{
				recomended_depth = 8 ;
				break ;
			}
		}
		while( --count >= 0 )
			if( buf.alpha[count] == 0  )
			{
				if( recomended_depth == 0 )
					recomended_depth = 1 ;
			}else if( (buf.alpha[count]&0xFF) != 0xFF  )
			{
				recomended_depth = 8 ;
				break ;
			}
		if( recomended_depth == 8 )
			break;
	}
	free_scanline(&buf, True);

	return recomended_depth;
}



/* ********************************************************************************/
/* Vector -> ASImage functions :                                                  */
/* ********************************************************************************/
Bool
set_asimage_vector( ASImage *im, register double *vector )
{
	if( vector == NULL || im == NULL )
		return False;

	if( im->alt.vector == NULL )
		im->alt.vector = safemalloc( im->width*im->height*sizeof(double));

	{
		register double *dst = im->alt.vector ;
		register int i = im->width*im->height;
		while( --i >= 0 )
			dst[i] = vector[i] ;
	}

	return True;
}

ASVectorPalette*
vectorize_asimage( ASImage *im, unsigned int max_colors, unsigned int dither,
				   int opaque_threshold	)
{
	ASVectorPalette* pal ;
	double *vec ;
	ASColormap cmap;
    unsigned int r, g, b, v;
	unsigned int x, y, j ;

	if( im->alt.vector == NULL )
		im->alt.vector = safemalloc( im->width*im->height*sizeof(double));
	vec = im->alt.vector ;

	/* contributed by Valeriy Onuchin from Root project at cern.ch */   

 	dither = dither > 7 ? 7 : dither;
	{
 		int *res = colormap_asimage(im, &cmap, max_colors, dither, opaque_threshold);
 
    	for ( y = 0; y < im->height; y++) 
		{
       		for ( x = 0; x < im->width; x++) 
			{
          		int i = y*im->width + x;
          		g = INDEX_SHIFT_GREEN(cmap.entries[res[i]].green);
          		b = INDEX_SHIFT_BLUE(cmap.entries[res[i]].blue);
          		r = INDEX_SHIFT_RED(cmap.entries[res[i]].red);
          		v = MAKE_INDEXED_COLOR24(r,g,b);
		        v = (v>>12)&0x0FFF;
          		vec[(im->height - y - 1)*im->width + x] = ((double)v)/0x0FFF;
       		}
    	}
	
		free (res);
	}
    pal = safecalloc( 1, sizeof(ASVectorPalette));

	pal->npoints = cmap.count ;	
	pal->points = safemalloc( sizeof(double)*cmap.count);
	pal->channels[IC_RED] = safemalloc( sizeof(CARD16)*cmap.count);
	pal->channels[IC_GREEN] = safemalloc( sizeof(CARD16)*cmap.count);
	pal->channels[IC_BLUE] = safemalloc( sizeof(CARD16)*cmap.count);
	pal->channels[IC_ALPHA] = safemalloc( sizeof(CARD16)*cmap.count);
 
    for ( j = 0; j < cmap.count; j++) {
       g = INDEX_SHIFT_GREEN(cmap.entries[j].green);
       b = INDEX_SHIFT_BLUE(cmap.entries[j].blue);
       r = INDEX_SHIFT_RED(cmap.entries[j].red);
       v = MAKE_INDEXED_COLOR24(r,g,b);
 
       v = (v>>12)&0x0FFF;
       pal->points[j] = ((double)v)/0x0FFF;
 		/* palette uses 16 bit color values for greater precision */
       pal->channels[IC_RED][j] = cmap.entries[j].red<<QUANT_ERR_BITS;
       pal->channels[IC_GREEN][j] = cmap.entries[j].green<<QUANT_ERR_BITS;
       pal->channels[IC_BLUE][j] = cmap.entries[j].blue<<QUANT_ERR_BITS;
       pal->channels[IC_ALPHA][j] = 0xFFFF;
    }
 
    destroy_colormap(&cmap, True);

	return pal;
}

/* ********************************************************************************/
/* Convinience function - very fast image cloning :                               */
/* ********************************************************************************/
ASImage*
clone_asimage( ASImage *src, ASFlagType filter )
{
	ASImage *dst = NULL ;
	START_TIME(started);

	if( !AS_ASSERT(src) )
	{
		int chan ;
		dst = create_asimage(src->width, src->height, 100);
		if( get_flags( src->flags, ASIM_DATA_NOT_USEFUL ) )
			set_flags( dst->flags, ASIM_DATA_NOT_USEFUL );
		dst->back_color = src->back_color ;
		for( chan = 0 ; chan < IC_NUM_CHANNELS;  chan++ )
			if( get_flags( filter, 0x01<<chan) )
			{
				register int i = dst->height;
				register ASStorageID *dst_rows = dst->channels[chan] ;
				register ASStorageID *src_rows = src->channels[chan] ;
				while( --i >= 0 )
					dst_rows[i] = dup_data( NULL, src_rows[i] );
			}
	}
	SHOW_TIME("", started);
	return dst;
}

/* ********************************************************************************/
/* Convinience function
 * 		- generate rectangles list for channel values exceeding threshold:        */
/* ********************************************************************************/
XRectangle*
get_asimage_channel_rects( ASImage *src, int channel, unsigned int threshold, unsigned int *rects_count_ret )
{
	XRectangle *rects = NULL ;
	int rects_count = 0, rects_allocated = 0 ;

	START_TIME(started);

	if( !AS_ASSERT(src) && channel < IC_NUM_CHANNELS )
	{
		int i = src->height;
		ASStorageID  *src_rows = src->channels[channel] ;
		unsigned int *height = safemalloc( (src->width+1)*2 * sizeof(unsigned int) );
		unsigned int *prev_runs = NULL ;
		int prev_runs_count = 0 ;
		unsigned int *runs = safemalloc( (src->width+1)*2 * sizeof(unsigned int) );
		unsigned int *tmp_runs = safemalloc( (src->width+1)*2 * sizeof(unsigned int) );
		unsigned int *tmp_height = safemalloc( (src->width+1)*2 * sizeof(unsigned int) );
		Bool count_empty = (ARGB32_CHAN8(src->back_color,channel)>= threshold);

#ifdef DEBUG_RECTS
		fprintf( stderr, "%d:back_color = #%8.8lX,  count_empty = %d, thershold = %d\n", __LINE__, src->back_color, count_empty, threshold );
#endif
		while( --i >= -1 )
		{
			int runs_count = 0 ;
#ifdef DEBUG_RECTS
			fprintf( stderr, "%d: LINE %d **********************\n", __LINE__, i );
#ifdef DEBUG_RECTS2
			asimage_print_line (src, channel, i, 0xFFFFFFFF);
#else
			asimage_print_line (src, channel, i, VRB_LINE_CONTENT);
#endif
#endif
			if( i >= 0 )
			{
				if( src_rows[i] )
				{
					runs_count = threshold_stored_data(NULL, src_rows[i], runs, src->width, threshold);
				}else if( count_empty )
				{
					runs_count = 2 ;
					runs[0] = 0 ;
					runs[1] = src->width ;
				}
			}
#ifdef DEBUG_RECTS
			fprintf( stderr, "runs_count = %d\n", runs_count );
#endif
			if( runs_count > 0 && (runs_count &0x0001) != 0 )
			{                                  /* allways wants to have even number of runs */
				runs[runs_count] = 0 ;
				++runs_count ;
			}

			if( prev_runs_count > 0 )
			{ /* here we need to merge runs and add all the detached rectangles to the rects list */
				int k = 0, l = 0, last_k = 0 ;
				int tmp_count = 0 ;
				unsigned int *tmp ;
				if( runs_count == 0 )
				{
					runs[0] = src->width ;
					runs[1] = src->width ;
					runs_count = 2 ;
				}
				tmp_runs[0] = 0 ;
				tmp_runs[1] = src->width ;
				/* two passes : in first pass we go through old runs and try and see if they are continued
				 * in this line. If not - we add them to the list of rectangles. At
				 * the same time we subtract them from new line's runs : */
				for( l = 0 ; l < prev_runs_count ; ++l, ++l )
				{
					int start = prev_runs[l], end = prev_runs[l+1] ;
					int matching_runs = 0 ;
#ifdef DEBUG_RECTS
					fprintf( stderr, "%d: prev run %d : start = %d, end = %d, last_k = %d, height = %d\n", __LINE__, l, start, end, last_k, height[l] );
#endif
					for( k = last_k ; k < runs_count ; ++k, ++k )
					{
#ifdef DEBUG_RECTS
						fprintf( stderr, "*%d: new run %d : start = %d, end = %d\n", __LINE__, k, runs[k], runs[k+1] );
#endif
						if( (int)runs[k] > end )
						{	/* add entire run to rectangles list */
							if( rects_count >= rects_allocated )
							{
								rects_allocated = rects_count + 8 + (rects_count>>3);
								rects = realloc( rects, rects_allocated*sizeof(XRectangle));
							}
							rects[rects_count].x = start ;
							rects[rects_count].y = i+1 ;
							rects[rects_count].width = (end-start)+1 ;
							rects[rects_count].height = height[l] ;
#ifdef DEBUG_RECTS
							fprintf( stderr, "*%d: added rectangle at y = %d\n", __LINE__, rects[rects_count].y );
#endif
							++rects_count ;
							++matching_runs;
							break;
						}else if( (int)runs[k+1] >= start  )
						{
							if( start < (int)runs[k] )
							{	/* add rectangle start, , runs[k]-start, height[l] */
								if( rects_count >= rects_allocated )
								{
									rects_allocated = rects_count + 8 + (rects_count>>3);
									rects = realloc( rects, rects_allocated*sizeof(XRectangle));
								}
								rects[rects_count].x = start ;
								rects[rects_count].y = i+1 ;
								rects[rects_count].width = runs[k]-start ;
								rects[rects_count].height = height[l] ;
#ifdef DEBUG_RECTS
								fprintf( stderr, "*%d: added rectangle at y = %d\n", __LINE__, rects[rects_count].y );
#endif
								++rects_count ;
								start = runs[k] ;
							}else if( start > (int)runs[k] )
							{
								tmp_runs[tmp_count] = runs[k] ;
								tmp_runs[tmp_count+1] = start-1 ;
								tmp_height[tmp_count] = 1 ;
#ifdef DEBUG_RECTS
								fprintf( stderr, "*%d: tmp_run %d added : %d ... %d, height = %d\n", __LINE__, tmp_count, runs[k], start-1, 1 );
#endif
								++tmp_count ; ++tmp_count ;
								runs[k] = start ;
							}
							/* at that point both runs start at the same point */
							if( end < (int)runs[k+1] )
							{
								runs[k] = end+1 ;
							}else 
							{   
								if( end > (int)runs[k+1] )
								{	
									/* add rectangle runs[k+1]+1, , end - runs[k+1], height[l] */
									if( rects_count >= rects_allocated )
									{
										rects_allocated = rects_count + 8 + (rects_count>>3);
										rects = realloc( rects, rects_allocated*sizeof(XRectangle));
									}
									rects[rects_count].x = runs[k+1]+1 ;
									rects[rects_count].y = i+1 ;
									rects[rects_count].width = end - runs[k+1] ;
									rects[rects_count].height = height[l] ;
#ifdef DEBUG_RECTS
									fprintf( stderr, "*%d: added rectangle at y = %d\n", __LINE__, rects[rects_count].y );
#endif
									++rects_count ;
									end = runs[k+1] ;
								
								} 
								/* eliminating new run - it was all used up :) */
								runs[k] = src->width ;
								runs[k+1] = src->width ;
#ifdef DEBUG_RECTS
								fprintf( stderr, "*%d: eliminating new run %d\n", __LINE__, k );
#endif
								++k ; ++k ;
							}
							tmp_runs[tmp_count] = start ;
							tmp_runs[tmp_count+1] = end ;
							tmp_height[tmp_count] = height[l]+1 ;
#ifdef DEBUG_RECTS
							fprintf( stderr, "*%d: tmp_run %d added : %d ... %d, height = %d\n", __LINE__, tmp_count, start, end, height[l]+1 );
#endif
							++tmp_count ; ++tmp_count ;
							last_k = k ;
							++matching_runs;
							break;
						}
					}
					if( matching_runs == 0 ) 
					{  /* no new runs for this prev run - add rectangle */
#ifdef DEBUG_RECTS
						fprintf( stderr, "%d: NO MATCHING NEW RUNS : start = %d, end = %d, height = %d\n", __LINE__, start, end, height[l] );
#endif
						if( rects_count >= rects_allocated )
						{
							rects_allocated = rects_count + 8 + (rects_count>>3);
							rects = realloc( rects, rects_allocated*sizeof(XRectangle));
						}
						rects[rects_count].x = start ;
						rects[rects_count].y = i+1 ;
						rects[rects_count].width = (end-start)+1 ;
						rects[rects_count].height = height[l] ;
#ifdef DEBUG_RECTS
						fprintf( stderr, "*%d: added rectangle at y = %d\n", __LINE__, rects[rects_count].y );
#endif
						++rects_count ;
					}	 
				}
				/* second pass: we need to pick up remaining new runs */
				/* I think these should be inserted in oredrly manner so that we have runs list arranged in ascending order */
				for( k = 0 ; k < runs_count ; ++k, ++k )
					if( runs[k] < src->width )
					{
						int ii = tmp_count ; 
						while( ii > 0 && tmp_runs[ii-1] > runs[k] )
						{
							tmp_runs[ii] = tmp_runs[ii-2] ;
							tmp_runs[ii+1] = tmp_runs[ii-1] ;
							tmp_height[ii] = tmp_height[ii-2] ;
							--ii ; --ii ;
						}
						tmp_runs[ii] = runs[k] ;
						tmp_runs[ii+1] = runs[k+1] ;
						tmp_height[ii] = 1 ;
#ifdef DEBUG_RECTS
						fprintf( stderr, "*%d: tmp_run %d added : %d ... %d, height = %d\n", __LINE__, ii, runs[k], runs[k+1], 1 );
#endif
						++tmp_count, ++tmp_count;
					}
				tmp = prev_runs ;
				prev_runs = tmp_runs ;
				tmp_runs = tmp ;
				tmp = height ;
				height = tmp_height ;
				tmp_height = tmp ;
				prev_runs_count = tmp_count ;
			}else if( runs_count > 0 )
			{
				int k = runs_count;
				prev_runs_count = runs_count ;
				prev_runs = runs ;
				runs = safemalloc( (src->width+1)*2 * sizeof(unsigned int) );
				while( --k >= 0 )
					height[k] = 1 ;
			}
		}
		free( runs );
		if( prev_runs )
			free( prev_runs );
		free( tmp_runs );
		free( tmp_height );
		free( height );
	}
	SHOW_TIME("", started);

	if( rects_count_ret )
		*rects_count_ret = rects_count ;

	return rects;
}

/***********************************************************************************/
void
raw2scanline( register CARD8 *row, ASScanline *buf, CARD8 *gamma_table, unsigned int width, Bool grayscale, Bool do_alpha )
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
				buf->red [x] = gamma_table[*(--row)];
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
				buf->red [x] = *(--row);
			}
	}
}

/* ********************************************************************************/
/* The end !!!! 																 */
/* ********************************************************************************/

