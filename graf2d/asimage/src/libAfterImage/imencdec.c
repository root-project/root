/*
 * Copyright (c) 2000,2001,2004 Sasha Vasko <sasha at aftercode.net>
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


/*#undef NO_DEBUG_OUTPUT*/
#undef LOCAL_DEBUG
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

#ifdef HAVE_MMX
#include <mmintrin.h>
#endif

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asvisual.h"
#include "blender.h"
#include "asimage.h"
#include "imencdec.h"

static void decode_asscanline_native( ASImageDecoder *imdec, unsigned int skip, int y );
static void decode_asscanline_ximage( ASImageDecoder *imdec, unsigned int skip, int y );
static void decode_asscanline_argb32( ASImageDecoder *imdec, unsigned int skip, int y );

void decode_image_scanline_normal( ASImageDecoder *imdec );
void decode_image_scanline_beveled( ASImageDecoder *imdec );
void decode_image_scl_bevel_solid( ASImageDecoder *imdec );


Bool create_image_xim( ASVisual *asv, ASImage *im, ASAltImFormats format );
Bool create_image_argb32( ASVisual *asv, ASImage *im, ASAltImFormats format );

void encode_image_scanline_asim( ASImageOutput *imout, ASScanline *to_store );
void encode_image_scanline_xim( ASImageOutput *imout, ASScanline *to_store );
void encode_image_scanline_mask_xim( ASImageOutput *imout, ASScanline *to_store );
void encode_image_scanline_argb32( ASImageOutput *imout, ASScanline *to_store );

static struct ASImageFormatHandlers
{
	Bool (*check_create_asim_format)( ASVisual *asv, ASImage *im, ASAltImFormats format );
	void (*encode_image_scanline)( ASImageOutput *imout, ASScanline *to_store );
}asimage_format_handlers[ASA_Formats] =
{
	{ NULL, encode_image_scanline_asim },
	{ create_image_xim, encode_image_scanline_xim },
	{ create_image_xim, encode_image_scanline_mask_xim },
	{ create_image_xim, encode_image_scanline_xim },
	{ create_image_xim, encode_image_scanline_mask_xim },
	{ create_image_xim, encode_image_scanline_xim },
	{ create_image_argb32, encode_image_scanline_argb32 },
	{ NULL, NULL }                             /* vector of doubles */
};



void output_image_line_top( ASImageOutput *, ASScanline *, int );
void output_image_line_fine( ASImageOutput *, ASScanline *, int );
void output_image_line_fast( ASImageOutput *, ASScanline *, int );
void output_image_line_direct( ASImageOutput *, ASScanline *, int );


/* *********************************************************************
 * quality control: we support several levels of quality to allow for
 * smooth work on older computers.
 * *********************************************************************/
static int asimage_quality_level = ASIMAGE_QUALITY_GOOD;


Bool create_image_xim( ASVisual *asv, ASImage *im, ASAltImFormats format )
{
	Bool scratch = False, do_alpha = False ; 
	XImage **dst ;
	if( format == ASA_ScratchXImageAndAlpha ) 
	{	
		format = ASA_ScratchXImage ;
		do_alpha = True ;
	}

	if( format == ASA_ScratchXImage || format == ASA_ScratchMaskXImage ) 
	{	
		scratch = True ;
		format = (format - ASA_ScratchXImage ) + ASA_XImage ;
	}		
	dst = (format == ASA_MaskXImage )? &(im->alt.mask_ximage):&(im->alt.ximage);
	if( *dst == NULL )
	{
		int depth = 0 ;
		if( format == ASA_MaskXImage )
			depth = get_flags(im->flags, ASIM_XIMAGE_8BIT_MASK )? 8: 1;
		if( scratch )
			*dst = create_visual_scratch_ximage( asv, im->width, im->height, depth );
		else
			*dst = create_visual_ximage( asv, im->width, im->height, depth );
		if( *dst == NULL )
			show_error( "Unable to create %sXImage for the visual %d",
				        (format == ASA_MaskXImage )?"mask ":"",
						asv->visual_info.visualid );
	}
	return ( *dst != NULL );
}

Bool create_image_argb32( ASVisual *asv, ASImage *im, ASAltImFormats format )
{
	if( im->alt.argb32 == NULL )
		im->alt.argb32 = safemalloc( im->width*im->height*sizeof(ARGB32) );
	return True;
}

/*************************************************************************/
/* low level routines ****************************************************/
static void
asimage_dup_line (ASImage * im, ColorPart color, unsigned int y1, unsigned int y2, unsigned int length)
{
	ASStorageID *part = im->channels[color];
	if (part[y2] != 0)
	{	
		forget_data(NULL, part[y2]);
		part[y2] = 0 ;
	}
	if( part[y1] )
	 	part[y2] = dup_data(NULL, part[y1] );
}

void
asimage_erase_line( ASImage * im, ColorPart color, unsigned int y )
{
	if( !AS_ASSERT(im) )
	{
		ASStorageID *part = im->channels[color];
		if( color < IC_NUM_CHANNELS )
		{
			if( part[y] )
			{
				forget_data( NULL, part[y] );
				part[y] = 0;
			}
		}else
		{
			int c ;
			for( c = 0 ; c < IC_NUM_CHANNELS ; c++ )
			{
				part = im->channels[c];
				if( part[y] )
					forget_data( NULL, part[y] );
				part[y] = 0;
			}
		}
	}
}


/* for consistency sake : */
void
copy_component( register CARD32 *src, register CARD32 *dst, int *unused, int len )
{
#if 1
#ifdef CARD64
	CARD64 *dsrc = (CARD64*)src;
	CARD64 *ddst = (CARD64*)dst;
#else
	double *dsrc = (double*)src;
	double *ddst = (double*)dst;
#endif
	register int i = 0;

	len += len&0x01;
	len = len>>1 ;
	do
	{
		ddst[i] = dsrc[i];
	}while(++i < len );
#else
	register int i = 0;

	len += len&0x01;
	do
	{
		dst[i] = src[i];
	}while(++i < len );
#endif
}

static inline int
set_component( register CARD32 *src, register CARD32 value, int offset, int len )
{
	register int i ;
	for( i = offset ; i < len ; ++i )
		src[i] = value;
	return len-offset;
}



static inline void
divide_component( register CARD32 *src, register CARD32 *dst, CARD16 ratio, int len )
{
	register int i = 0;
	len += len&0x00000001;                     /* we are 8byte aligned/padded anyways */
	if( ratio == 2 )
	{
#ifdef HAVE_MMX
		if( asimage_use_mmx )
		{
#if 1
			__m64  *vdst = (__m64*)&(dst[0]);
			__m64  *vsrc = (__m64*)&(src[0]);
			len = len>>1;
			do{
        		vdst[i] = _mm_srli_pi32(vsrc[i],1);  /* psrld */
			}while( ++i < len );
 			_mm_empty();
#else
			double *ddst = (double*)&(dst[0]);
			double *dsrc = (double*)&(src[0]);
			len = len>>1;
			do{
				asm volatile
       		    (
            		"movq %1, %%mm0  \n\t" // load 8 bytes from src[i] into MM0
            		"psrld $1, %%mm0 \n\t" // MM0=src[i]>>1
            		"movq %%mm0, %0  \n\t" // store the result in dest
					: "=m" (ddst[i]) // %0
					: "m"  (dsrc[i]) // %1
	            );
			}while( ++i < len );
#endif		
		}else
#endif
			do{
				dst[i] = src[i] >> 1;
				dst[i+1] = src[i+1]>> 1;
				i += 2 ;
			}while( i < len );
	}else
	{
		do{
			register int c1 = src[i];
			register int c2 = src[i+1];
			dst[i] = c1/ratio;
			dst[i+1] = c2/ratio;
			i+=2;
		}while( i < len );
	}
}


/* ******************** ASImageDecoder ****************************/
ASImageDecoder *
start_image_decoding( ASVisual *asv,ASImage *im, ASFlagType filter,
					  int offset_x, int offset_y,
					  unsigned int out_width,
					  unsigned int out_height,
					  ASImageBevel *bevel )
{
	ASImageDecoder *imdec = NULL;

	if( asv == NULL )
		asv = get_default_asvisual();

 	if( AS_ASSERT(filter) || AS_ASSERT(asv))
		return NULL;
	if( im != NULL )
		if( im->magic != MAGIC_ASIMAGE )
		{
#ifdef LOCAL_DEBUG
			ASImage **tmp = NULL;
			*tmp = im ;                        /* segfault !!!!! */
#endif
			im = NULL ;
		}

	if( im == NULL )
	{
		offset_x = offset_y = 0 ;
		if( AS_ASSERT(out_width)|| AS_ASSERT(out_height))
			return NULL ;
	}else
	{
		if( offset_x < 0 )
			offset_x = (int)im->width + (offset_x%(int)im->width);
		else
			offset_x %= im->width ;
		if( offset_y < 0 )
			offset_y = (int)im->height + (offset_y%(int)im->height);
		else
			offset_y %= im->height ;
		if( out_width == 0 )
			out_width = im->width ;
		if( out_height == 0 )
			out_height = im->height ;

	}

	imdec = safecalloc( 1, sizeof(ASImageDecoder));
	imdec->asv = asv ;
	imdec->im = im ;
	imdec->filter = filter ;
	imdec->offset_x = offset_x ;
	imdec->out_width = out_width;
	imdec->offset_y = offset_y ;
	imdec->out_height = out_height;
	imdec->next_line = offset_y ;
	imdec->back_color = (im != NULL)?im->back_color:ARGB32_DEFAULT_BACK_COLOR ;
    imdec->bevel = bevel ;
  	if( bevel )
	{
		if( bevel->left_outline > MAX_BEVEL_OUTLINE )
			bevel->left_outline = MAX_BEVEL_OUTLINE ;
		if( bevel->top_outline > MAX_BEVEL_OUTLINE )
			bevel->top_outline = MAX_BEVEL_OUTLINE ;
		if( bevel->right_outline > MAX_BEVEL_OUTLINE )
			bevel->right_outline = MAX_BEVEL_OUTLINE ;
		if( bevel->bottom_outline > MAX_BEVEL_OUTLINE )
			bevel->bottom_outline = MAX_BEVEL_OUTLINE ;
		if( bevel->left_inline > out_width )
			bevel->left_inline = MAX((int)out_width,0) ;
		if( bevel->top_inline > out_height )
			bevel->top_inline = MAX((int)out_height,0) ;
		if( bevel->left_inline+bevel->right_inline > (int)out_width )
			bevel->right_inline = MAX((int)out_width-(int)bevel->left_inline,0) ;
		if( bevel->top_inline+bevel->bottom_inline > (int)out_height )
			bevel->bottom_inline = MAX((int)out_height-(int)bevel->top_inline,0) ;

		if( bevel->left_outline == 0 && bevel->right_outline == 0 &&
			bevel->top_outline == 0 && bevel->bottom_outline == 0 &&
			bevel->left_inline == 0 && bevel->right_inline == 0 &&
			bevel->top_inline == 0 && bevel->bottom_inline == 0 )
			imdec->bevel = bevel = NULL ;
	}
	if( bevel )
	{
		imdec->bevel_left   = bevel->left_outline ;
		imdec->bevel_top    = bevel->top_outline ;
		imdec->bevel_right  = imdec->bevel_left + (int)out_width ;
		imdec->bevel_bottom = imdec->bevel_top  + (int)out_height;
		imdec->bevel_h_addon  = bevel->left_outline+ bevel->right_outline;
		imdec->bevel_v_addon  = bevel->top_outline + bevel->bottom_outline;

		imdec->decode_image_scanline = decode_image_scanline_beveled ;
	}else
		imdec->decode_image_scanline = decode_image_scanline_normal ;

	prepare_scanline(out_width+imdec->bevel_h_addon, 0, &(imdec->buffer), asv->BGR_mode );
    imdec->buffer.back_color = (im != NULL)?im->back_color:ARGB32_DEFAULT_BACK_COLOR ;
	imdec->buffer.flags = filter;

	imdec->decode_asscanline = decode_asscanline_native;
	if( im != NULL )
	{	
		if( get_flags( im->flags, ASIM_DATA_NOT_USEFUL ) )
		{
			if( im->alt.ximage != NULL && !get_flags( im->flags, ASIM_XIMAGE_NOT_USEFUL) )
			{
				imdec->decode_asscanline = decode_asscanline_ximage;
				imdec->xim_buffer = safecalloc(1, sizeof(ASScanline));
				prepare_scanline(im->alt.ximage->width, 0, imdec->xim_buffer, asv->BGR_mode );
			}else if( im->alt.argb32 != NULL )
			{
				imdec->decode_asscanline = decode_asscanline_argb32;
			}	 
		}
	}

	return imdec;
}

void
set_decoder_bevel_geom( ASImageDecoder *imdec, int x, int y,
                        unsigned int width, unsigned int height )
{
	if( imdec && imdec->bevel )
	{
		ASImageBevel *bevel = imdec->bevel ;
		int tmp ;
		if( imdec->im )
		{
			if( width == 0 )
				width = imdec->im->width ;
			if( height == 0 )
				height= imdec->im->height;
		}else
		{
			if( width == 0 )
				width = MAX( (int)imdec->out_width - x,0) ;
			if( height == 0 )
				height= MAX( (int)imdec->out_height - y,0) ;
		}
		/* Bevel should completely encompas output region */
		x = MIN(x,0);
		y = MIN(y,0);
		if( x+width < imdec->out_width )
			width += (int)imdec->out_width - x ;
		if( y+height < imdec->out_height )
			height += (int)imdec->out_height - y ;

		imdec->bevel_left = x ;
		imdec->bevel_top  = y ;
		imdec->bevel_right = x+(int)width ;
		imdec->bevel_bottom = y+(int)height ;


		imdec->bevel_h_addon  = MAX(imdec->bevel_left+(int)bevel->left_outline, 0) ;
		tmp = MAX(0, (int)imdec->out_width - imdec->bevel_right );
		imdec->bevel_h_addon += MIN( tmp, (int)bevel->right_outline);

		imdec->bevel_v_addon  = MAX(imdec->bevel_top+(int)bevel->top_outline, 0) ;
		tmp = MAX(0, (int)imdec->out_height - imdec->bevel_bottom );
		imdec->bevel_v_addon += MIN( tmp, (int)bevel->bottom_outline);
	}
}

void
set_decoder_shift( ASImageDecoder *imdec, int shift )
{
	if( shift != 0 )
		shift = 8 ;

	if( imdec )
		imdec->buffer.shift = shift ;
}

void set_decoder_back_color( ASImageDecoder *imdec, ARGB32 back_color )
{
	if( imdec )
	{
		imdec->back_color = back_color ;
		imdec->buffer.back_color = back_color ;
	}
}


void
stop_image_decoding( ASImageDecoder **pimdec )
{
	if( pimdec )
		if( *pimdec )
		{
			free_scanline( &((*pimdec)->buffer), True );
			if( (*pimdec)->xim_buffer )
			{
				free_scanline( (*pimdec)->xim_buffer, True );
				free( (*pimdec)->xim_buffer );
			}

			free( *pimdec );
			*pimdec = NULL;
		}
}

/* ******************** ASImageOutput ****************************/

ASImageOutput *
start_image_output( ASVisual *asv, ASImage *im, ASAltImFormats format,
                    int shift, int quality )
{
	register ASImageOutput *imout= NULL;
   int formati = (int) format;

	if( im != NULL )
		if( im->magic != MAGIC_ASIMAGE )
		{
			im = NULL ;
		}

	if( asv == NULL )
		asv = get_default_asvisual();

	if( AS_ASSERT(im) || AS_ASSERT(asv) )
		return imout;

   if( formati < 0 || format == ASA_Vector || format >= ASA_Formats)
		return NULL;
	if( asimage_format_handlers[format].check_create_asim_format )
		if( !asimage_format_handlers[format].check_create_asim_format(asv, im, format) )
			return NULL;

	imout = safecalloc( 1, sizeof(ASImageOutput));
	imout->asv = asv;
	imout->im = im ;

	imout->out_format = format ;
	imout->encode_image_scanline = asimage_format_handlers[format].encode_image_scanline;

	prepare_scanline( im->width, 0, &(imout->buffer[0]), asv->BGR_mode);
	prepare_scanline( im->width, 0, &(imout->buffer[1]), asv->BGR_mode);

	imout->chan_fill[IC_RED]   = ARGB32_RED8(im->back_color);
	imout->chan_fill[IC_GREEN] = ARGB32_GREEN8(im->back_color);
	imout->chan_fill[IC_BLUE]  = ARGB32_BLUE8(im->back_color);
	imout->chan_fill[IC_ALPHA] = ARGB32_ALPHA8(im->back_color);

	imout->available = &(imout->buffer[0]);
	imout->used 	 = NULL;
	imout->buffer_shift = shift;
	imout->next_line = 0 ;
	imout->bottom_to_top = 1 ;
	if( quality > ASIMAGE_QUALITY_TOP || quality < ASIMAGE_QUALITY_POOR )
		quality = asimage_quality_level;

	imout->quality = quality ;
	if( shift > 0 )
	{/* choose what kind of error diffusion we'll use : */
		switch( quality )
		{
			case ASIMAGE_QUALITY_POOR :
			case ASIMAGE_QUALITY_FAST :
				imout->output_image_scanline = output_image_line_fast ;
				break;
			case ASIMAGE_QUALITY_GOOD :
				imout->output_image_scanline = output_image_line_fine ;
				break;
			case ASIMAGE_QUALITY_TOP  :
				imout->output_image_scanline = output_image_line_top ;
				break;
		}
	}else /* no quanitzation - no error diffusion */
		imout->output_image_scanline = output_image_line_direct ;

	return imout;
}

void set_image_output_back_color( ASImageOutput *imout, ARGB32 back_color )
{
	if( imout )
	{
		imout->chan_fill[IC_RED]   = ARGB32_RED8  (back_color);
		imout->chan_fill[IC_GREEN] = ARGB32_GREEN8(back_color);
		imout->chan_fill[IC_BLUE]  = ARGB32_BLUE8 (back_color);
		imout->chan_fill[IC_ALPHA] = ARGB32_ALPHA8(back_color);
	}
}

void toggle_image_output_direction( ASImageOutput *imout )
{
	if( imout )
	{
		if( imout->bottom_to_top < 0 )
		{
			if( imout->next_line >= (int)imout->im->height-1 )
				imout->next_line = 0 ;
			imout->bottom_to_top = 1 ;
		}else if( imout->next_line <= 0 )
		{
		 	imout->next_line = (int)imout->im->height-1 ;
			imout->bottom_to_top = -1 ;
		}
	}
}


void
stop_image_output( ASImageOutput **pimout )
{
	if( pimout )
	{
		register ASImageOutput *imout = *pimout;
		if( imout )
		{
			if( imout->used )
				imout->output_image_scanline( imout, NULL, 1);
			free_scanline(&(imout->buffer[0]), True);
			free_scanline(&(imout->buffer[1]), True);
			free( imout );
			*pimout = NULL;
		}
	}
}

/* diffusingly combine src onto self and dst, and rightbitshift src by quantization shift */
static inline void
best_output_filter( register CARD32 *line1, register CARD32 *line2, int unused, int len )
{/* we carry half of the quantization error onto the surrounding pixels : */
 /*        X    7/16 */
 /* 3/16  5/16  1/16 */
	register int i ;
	register CARD32 errp = 0, err = 0, c;
	c = line1[0];
	if( (c&0xFFFF0000)!= 0 )
		c = ( c&0x7F000000 )?0:0x0000FFFF;
	errp = c&QUANT_ERR_MASK;
	line1[0] = c>>QUANT_ERR_BITS ;
	line2[0] += (errp*5)>>4 ;

	for( i = 1 ; i < len ; ++i )
	{
		c = line1[i];
		if( (c&0xFFFF0000)!= 0 )
			c = (c&0x7F000000)?0:0x0000FFFF;
		c += ((errp*7)>>4) ;
		err = c&QUANT_ERR_MASK;
		line1[i] = (c&0x7FFF0000)?0x000000FF:(c>>QUANT_ERR_BITS);
		line2[i-1] += (err*3)>>4 ;
		line2[i] += ((err*5)>>4)+(errp>>4);
		errp = err ;
	}
}

static inline void
fine_output_filter( register CARD32 *src, register CARD32 *dst, short ratio, int len )
{/* we carry half of the quantization error onto the following pixel and store it in dst: */
	register int i = 0;
	if( ratio <= 1 )
	{
		register int c = src[0];
  	    do
		{
			if( (c&0xFFFF0000)!= 0 )
				c = ( c&0x7F000000 )?0:0x0000FFFF;
			dst[i] = c>>(QUANT_ERR_BITS) ;
			if( ++i >= len )
				break;
			c = ((c&QUANT_ERR_MASK)>>1)+src[i];
		}while(1);
	}else if( ratio == 2 )
	{
		register CARD32 c = src[0];
  	    do
		{
			c = c>>1 ;
			if( (c&0xFFFF0000) != 0 )
				c = ( c&0x7F000000 )?0:0x0000FFFF;
			dst[i] = c>>(QUANT_ERR_BITS) ;
			if( ++i >= len )
				break;
			c = ((c&QUANT_ERR_MASK)>>1)+src[i];
		}while( 1 );
	}else
	{
		register CARD32 c = src[0];
  	    do
		{
			c = c/ratio ;
			if( c&0xFFFF0000 )
				c = ( c&0x7F000000 )?0:0x0000FFFF;
			dst[i] = c>>(QUANT_ERR_BITS) ;
			if( ++i >= len )
				break;
			c = ((c&QUANT_ERR_MASK)>>1)+src[i];
		}while(1);
	}
}

static inline void
fast_output_filter( register CARD32 *src, register CARD32 *dst, short ratio, int len )
{/*  no error diffusion whatsoever: */
	register int i = 0;
	if( ratio <= 1 )
	{
		for( ; i < len ; ++i )
		{
			register CARD32 c = src[i];
			if( (c&0xFFFF0000) != 0 )
				dst[i] = ( c&0x7F000000 )?0:0x000000FF;
			else
				dst[i] = c>>(QUANT_ERR_BITS) ;
		}
	}else if( ratio == 2 )
	{
		for( ; i < len ; ++i )
		{
			register CARD32 c = src[i]>>1;
			if( (c&0xFFFF0000) != 0 )
				dst[i] = ( c&0x7F000000 )?0:0x000000FF;
			else
				dst[i] = c>>(QUANT_ERR_BITS) ;
		}
	}else
	{
		for( ; i < len ; ++i )
		{
			register CARD32 c = src[i]/ratio;
			if( (c&0xFFFF0000) != 0 )
				dst[i] = ( c&0x7F000000 )?0:0x000000FF;
			else
				dst[i] = c>>(QUANT_ERR_BITS) ;
		}
	}
}

static inline void
fine_output_filter_mod( register CARD32 *data, int unused, int len )
{/* we carry half of the quantization error onto the following pixel : */
	register int i ;
	register CARD32 err = 0, c;
	for( i = 0 ; i < len ; ++i )
	{
		c = data[i];
		if( (c&0xFFFF0000) != 0 )
			c = ( c&0x7E000000 )?0:0x0000FFFF;
		c += err;
		err = (c&QUANT_ERR_MASK)>>1 ;
		data[i] = (c&0x00FF0000)?0x000000FF:c>>QUANT_ERR_BITS ;
	}
}

/* *********************************************************************/
/*					    	 DECODER : 	   							  */
/* Low level drivers :                                                */
static void
decode_asscanline_native( ASImageDecoder *imdec, unsigned int skip, int y )
{
	int i ;
	ASScanline *scl = &(imdec->buffer);
	int count, width = scl->width-skip ;
	for( i = 0 ; i < IC_NUM_CHANNELS ; i++ )
		if( get_flags(imdec->filter, 0x01<<i) )
		{
			register CARD32 *chan = scl->channels[i]+skip;
			if( imdec->im )
				count = fetch_data32( NULL, imdec->im->channels[i][y], chan, imdec->offset_x, width, 0, NULL);
			else
				count = 0 ;
			if( scl->shift )
			{
				register int k  = 0;
				for(; k < count ; k++ )
					chan[k] = chan[k]<<8;
			}
			if( count < width )
				set_component( chan, ARGB32_CHAN8(imdec->back_color, i)<<scl->shift, count, width );
		}
	clear_flags( scl->flags, SCL_DO_ALL);
	set_flags( scl->flags, imdec->filter);
}

static void
decode_asscanline_argb32( ASImageDecoder *imdec, unsigned int skip, int y )
{
	ASScanline *scl = &(imdec->buffer);
	int count, width = scl->width-skip ;
	ARGB32 *row = imdec->im->alt.argb32 + y*imdec->im->width ;
	CARD32 *a = scl->alpha+skip;
	CARD32 *r = scl->red+skip;
	CARD32 *g = scl->green+skip;
	CARD32 *b = scl->blue+skip;
	int max_x = imdec->im->width ;

	if( get_flags( imdec->filter, SCL_DO_ALPHA ) )
	{
		int x = imdec->offset_x ; 
		for( count = 0 ; count < width ; ++count) 
		{	
			a[count] = ARGB32_ALPHA8(row[x])<<scl->shift ;	
			if( ++x >= max_x ) 	x = 0;
		}	 
	}	 
		
	if( get_flags( imdec->filter, SCL_DO_RED ) )
	{
		int x = imdec->offset_x ; 
		for( count = 0 ; count < width ; ++count) 
		{	
			r[count] = ARGB32_RED8(row[x])<<scl->shift ;	
			if( ++x >= max_x ) 	x = 0;
		}	 
	}	 
		
	if( get_flags( imdec->filter, SCL_DO_GREEN ) )
	{
		int x = imdec->offset_x ; 
		for( count = 0 ; count < width ; ++count) 
		{	
			g[count] = ARGB32_GREEN8(row[x])<<scl->shift ;	
			if( ++x >= max_x ) 	x = 0;
		}	 
	}	 
	if( get_flags( imdec->filter, SCL_DO_BLUE ) )
	{
		int x = imdec->offset_x ; 
		for( count = 0 ; count < width ; ++count) 
		{	
			b[count] = ARGB32_BLUE8(row[x])<<scl->shift ;	
			if( ++x >= max_x ) 	x = 0;
		}	 
	}	 

	clear_flags( scl->flags, SCL_DO_ALL);
	set_flags( scl->flags, imdec->filter);
}


static void
decode_asscanline_ximage( ASImageDecoder *imdec, unsigned int skip, int y )
{
	int i ;
	ASScanline *scl = &(imdec->buffer);
	XImage *xim = imdec->im->alt.ximage ;
	int count, width = scl->width-skip, xim_width = xim->width ;
	ASFlagType filter = imdec->filter ;
#if 1
	if( width > xim_width || imdec->offset_x > 0 )
	{/* need to tile :( */
		ASScanline *xim_scl = imdec->xim_buffer;
		int offset_x = imdec->offset_x%xim_width ;
/*fprintf( stderr, __FILE__ ":" __FUNCTION__ ": width=%d, xim_width=%d, skip = %d, offset_x = %d - tiling\n", width, xim->width, skip, imdec->offset_x );	*/

		GET_SCANLINE(imdec->asv,xim,xim_scl,y,(unsigned char*)xim->data+xim->bytes_per_line*y);
		/* We also need to decode mask if we have one :*/
		if( (xim = imdec->im->alt.mask_ximage ) != NULL )
		{
#ifndef X_DISPLAY_MISSING
			CARD32 *dst = xim_scl->alpha ;
			register int x = MIN((int)xim_scl->width,xim->width);
			if( xim->depth == 8 )
			{
				CARD8  *src = (CARD8*)xim->data+xim->bytes_per_line*y ;
				while(--x >= 0 ) dst[x] = (CARD32)(src[x]);
			}else
			{
				while(--x >= 0 ) dst[x] = (XGetPixel(xim, x, y) == 0)?0x00:0xFF;
			}
#endif
		}
		for( i = 0 ; i < IC_NUM_CHANNELS ; i++ )
			if( get_flags(filter, 0x01<<i) )
			{
				register CARD32 *src = xim_scl->channels[i]+offset_x ;
				register CARD32 *dst = scl->channels[i]+skip;
				register int k  = 0;
				count = xim_width-offset_x ;
				if( count > width )
					count = width ;

#define COPY_TILE_CHAN(op) \
		for(; k < count ; k++ )	dst[k] = op; \
		while( k < width ) \
		{	src = xim_scl->channels[i]-k ; \
			count = MIN(xim_width+k,width); \
			for(; k < count ; k++ ) dst[k] = op; \
		}

				if( scl->shift )
				{
					COPY_TILE_CHAN(src[k]<<8)
				}else
				{
					COPY_TILE_CHAN(src[k])
				}
				count += k ;
				if( count < width )
					set_component( src, ARGB32_CHAN8(imdec->back_color, i)<<scl->shift, count, width );
			}
	}else
#endif
	{/* cool we can put data directly into buffer : */
/*fprintf( stderr, __FILE__ ":" __FUNCTION__ ":direct\n" );	*/
		int old_offset = scl->offset_x ;
		scl->offset_x = skip ;
		GET_SCANLINE(imdec->asv,xim,scl,y,(unsigned char *)xim->data+xim->bytes_per_line*y);
		/* We also need to decode mask if we have one :*/
		if( (xim = imdec->im->alt.mask_ximage ) != NULL )
		{
#ifndef X_DISPLAY_MISSING
			CARD32 *dst = scl->alpha+skip ;
			register int x = MIN(width,xim_width);
			if( xim->depth == 8 )
			{
				CARD8  *src = (CARD8*)xim->data+xim->bytes_per_line*y ;
				while(--x >= 0 ) dst[x] = (CARD32)(src[x]);
			}else
			{
				while(--x >= 0 ) dst[x] = (XGetPixel(xim, x, y) == 0)?0x00:0xFF;
			}
#endif
		}
		count = MIN(width,xim_width);
		scl->offset_x = old_offset ;
		for( i = 0 ; i < IC_NUM_CHANNELS ; i++ )
			if( get_flags(filter, 0x01<<i) )
			{
				register CARD32 *chan = scl->channels[i]+skip;
				if( scl->shift )
				{
					register int k  = 0;
					for(; k < count ; k++ )
						chan[k] = chan[k]<<8;
				}
				if( count < width )
					set_component( chan, ARGB32_CHAN8(imdec->back_color, i)<<scl->shift, count, width );
			}
	}
	clear_flags( scl->flags,SCL_DO_ALL);
	set_flags( scl->flags,imdec->filter);
}

/***********************************************************************/
/* High level drivers :                                                */
void                                           /* normal (unbeveled) */
decode_image_scanline_normal( ASImageDecoder *imdec )
{
	int 	 			 y = imdec->next_line;
	if( y - imdec->offset_y >= imdec->out_height )
	{
		imdec->buffer.flags = 0 ;
		imdec->buffer.back_color = imdec->back_color ;
		return ;
	}

	if( imdec->im )
		y %= imdec->im->height;
	imdec->decode_asscanline( imdec, 0, y );
	++(imdec->next_line);
}

static inline void
draw_solid_bevel_line( register ASScanline *scl, int alt_left, int hi_end, int lo_start, int alt_right,
					   ARGB32 bevel_color, ARGB32 shade_color, ARGB32 hi_corner, ARGB32 lo_corner )
{
	int channel ;
	for( channel = 0 ; channel < ARGB32_CHANNELS ; ++channel )
		if( get_flags(scl->flags, (0x01<<channel)) )
		{
			if( hi_end > 0 )
			{
				set_component( scl->channels[channel],
						        ARGB32_CHAN8(bevel_color,channel)<<scl->shift,
								0, hi_end );
				if( alt_left > 0 )
					scl->channels[channel][alt_left-1] =
						        ARGB32_CHAN8(hi_corner,channel)<<scl->shift ;
			}
			if( lo_start < (int)scl->width )
			{
				set_component( scl->channels[channel],
						        ARGB32_CHAN8(shade_color,channel)<<scl->shift,
						        lo_start, scl->width );
				if( alt_right < (int)scl->width && alt_right > 0 )
					scl->channels[channel][scl->width - alt_right] =
					            ARGB32_CHAN8(lo_corner,channel)<<scl->shift ;
			}
		}
}
static inline void
draw_fading_bevel_sides( ASImageDecoder *imdec,
					     int left_margin, int left_delta,
					     int right_delta, int right_margin )
{
	register ASScanline *scl = &(imdec->buffer);
	ASImageBevel *bevel = imdec->bevel ;
	CARD32 ha_bevel = ARGB32_ALPHA8(bevel->hi_color);
	CARD32 ha_shade = ARGB32_ALPHA8(bevel->lo_color);
    CARD32 hda_bevel = (ha_bevel<<8)/(bevel->left_inline+1) ;
    CARD32 hda_shade = (ha_shade<<8)/(bevel->right_inline+1);
	int channel ;

	for( channel = 0 ; channel < ARGB32_CHANNELS ; ++channel )
		if( get_flags(scl->flags, (0x01<<channel)) )
		{
			CARD32 chan_col = ARGB32_CHAN8(bevel->hi_color,channel)<<scl->shift ;
			register CARD32 ca = hda_bevel*(left_delta+1) ;
			register int i = MIN((int)scl->width, imdec->bevel_left+(int)bevel->left_inline-left_delta);
			CARD32 *chan_img_start = scl->channels[channel] ;

			while( --i >= left_margin )
			{
				chan_img_start[i] = (chan_img_start[i]*(255-(ca>>8))+chan_col*(ca>>8))>>8 ;
				ca += hda_bevel ;
			}
			ca = hda_shade*(right_delta+1) ;
			i =  MAX( left_margin, imdec->bevel_right + right_delta - (int)bevel->right_inline);
			chan_col = ARGB32_CHAN8(bevel->lo_color,channel)<<scl->shift ;
			while( ++i < right_margin )
			{
				chan_img_start[i] = (chan_img_start[i]*(255-(ca>>8))+chan_col*(ca>>8))>>8 ;
				ca += hda_shade ;
			}
		}
}

static inline void
draw_transp_bevel_sides( ASImageDecoder *imdec,
					     int left_margin, int left_delta,
					     int right_delta, int right_margin )
{
	register ASScanline *scl = &(imdec->buffer);
	ASImageBevel *bevel = imdec->bevel ;
	CARD32 ha_bevel = ARGB32_ALPHA8(bevel->hi_color)>>1;
	CARD32 ha_shade = ARGB32_ALPHA8(bevel->lo_color)>>1;
	int channel ;

	for( channel = 0 ; channel < ARGB32_CHANNELS ; ++channel )
		if( get_flags(scl->flags, (0x01<<channel)) )
		{
			CARD32 chan_col = (ARGB32_CHAN8(bevel->hi_color,channel)<<scl->shift)*ha_bevel ;
			register CARD32 ca = 255-ha_bevel ;
			register int i = imdec->bevel_left+(int)bevel->left_inline-left_delta;
			CARD32 *chan_img_start = scl->channels[channel] ;

			while( --i >= left_margin )
				chan_img_start[i] = (chan_img_start[i]*ca+chan_col)>>8 ;

			ca = 255-ha_shade ;
			i =  MAX( left_margin, imdec->bevel_right + right_delta - (int)bevel->right_inline);
			chan_col = (ARGB32_CHAN8(bevel->lo_color,channel)<<scl->shift)*ha_shade ;
			while( ++i < right_margin )
				chan_img_start[i] = (chan_img_start[i]*ca+chan_col)>>8 ;
		}
}


static inline void
draw_transp_bevel_line ( ASImageDecoder *imdec,
					     int left_delta, int right_delta,
						 CARD32 ca,
						 ARGB32 left_color, ARGB32 color, ARGB32 right_color )
{
	register ASScanline *scl = &(imdec->buffer);
	ASImageBevel *bevel = imdec->bevel ;
	int start_point = imdec->bevel_left+(int)bevel->left_inline-left_delta;
	int end_point   = imdec->bevel_right + right_delta - (int)bevel->right_inline;
	int channel ;
	CARD32 rev_ca = (255-(ca>>8));
	if( start_point < (int)scl->width && end_point > 0 )
	{
		for( channel = 0 ; channel < ARGB32_CHANNELS ; ++channel )
			if( get_flags(scl->flags, (0x01<<channel)) )
			{
				CARD32 chan_col = (ARGB32_CHAN8(color,channel)<<scl->shift)*(ca>>8) ;
				CARD32 *chan_img_start = scl->channels[channel] ;
				register int i ;
				int end_i;

				if( start_point < 0 )
					i = -1 ;
				else
				{
					i = start_point-1 ;
					if( i < (int)scl->width && i >= 0 )
						chan_img_start[i] = (chan_img_start[i]*rev_ca + ARGB32_CHAN8(left_color,channel)*(ca>>8))>>8 ;
				}
				if( end_point >= (int)scl->width )
					end_i = scl->width ;
				else
				{
					end_i = end_point ;
					if( end_i >= 0 ) 
						chan_img_start[end_i] = (chan_img_start[end_i]*rev_ca + ARGB32_CHAN8(right_color,channel)*(ca>>8))>>8 ;
				}
				while( ++i < end_i )
					chan_img_start[i] = (chan_img_start[i]*rev_ca+chan_col)>>8;
			}
	}
}

void
decode_image_scanline_beveled( ASImageDecoder *imdec )
{
	register ASScanline *scl = &(imdec->buffer);
	int 	 			 y_out = imdec->next_line- (int)imdec->offset_y;
	register ASImageBevel *bevel = imdec->bevel ;
	ARGB32 bevel_color = bevel->hi_color, shade_color = bevel->lo_color;
	int offset_shade = 0;

	scl->flags = 0 ;
	if( y_out < 0 || y_out > (int)imdec->out_height+imdec->bevel_v_addon )
	{
		scl->back_color = imdec->back_color ;
		return ;
	}


	set_flags( scl->flags, imdec->filter );
	if( y_out < imdec->bevel_top )
	{
		if( bevel->top_outline > 0 )
		{
			register int line = y_out - (imdec->bevel_top - (int)bevel->top_outline);
			int alt_left  = (line*bevel->left_outline/bevel->top_outline)+1 ;
			int alt_right = (line*bevel->right_outline/bevel->top_outline)+1 ;

			alt_left += MAX(imdec->bevel_left-(int)bevel->left_outline,0) ;
			offset_shade = MAX(imdec->bevel_right+(int)bevel->right_outline-alt_right,0);

/*		fprintf( stderr, __FUNCTION__ " %d: y_out = %d, alt_left = %d, offset_shade = %d, alt_right = %d, scl->width = %d, out_width = %d\n",
					 	__LINE__, y_out, alt_left, offset_shade, alt_right, scl->width, imdec->out_width );
  */
			if( (int)scl->width < imdec->bevel_right )
				alt_right -= imdec->bevel_right-(int)scl->width ;
			if( offset_shade > (int)scl->width )
				offset_shade = scl->width ;
			draw_solid_bevel_line( scl, alt_left, offset_shade, offset_shade, alt_right,
							   	bevel->hi_color, bevel->lo_color, bevel->hihi_color, bevel->hilo_color );
		}
	}else if( y_out >= imdec->bevel_bottom )
	{
		if( bevel->bottom_outline > 0 )
		{
			register int line = bevel->bottom_outline - (y_out - imdec->bevel_bottom);
			int alt_left  = (line*bevel->left_outline/bevel->bottom_outline)+1 ;
			int alt_right = (line*bevel->right_outline/bevel->bottom_outline)+1 ;

			alt_left += MAX(imdec->bevel_left-(int)bevel->left_outline,0) ;
			offset_shade = MIN(alt_left, (int)scl->width );

			if( (int)scl->width < imdec->bevel_right )
				alt_right -= imdec->bevel_right-(int)scl->width ;

/*	fprintf( stderr, __FUNCTION__ " %d: y_out = %d, alt_left = %d, offset_shade = %d, alt_right = %d, scl->width = %d, out_width = %d\n",
					 __LINE__, y_out, alt_left, offset_shade, alt_right, scl->width, imdec->out_width );
  */
			set_flags( scl->flags, imdec->filter );
			draw_solid_bevel_line( scl, alt_left, alt_left, alt_left, alt_right,
							   	bevel->hi_color, bevel->lo_color,
							   	bevel->hilo_color, bevel->lolo_color );
		}
	}else
	{
		int left_margin = MAX(0, imdec->bevel_left);
		int right_margin = MIN((int)scl->width, imdec->bevel_right);
		int y = imdec->next_line-bevel->top_outline ;
		if( imdec->im )
			y %= imdec->im->height ;

		if( left_margin < (int)scl->width )
			imdec->decode_asscanline( imdec, left_margin, y );

		draw_solid_bevel_line( scl, -1, left_margin, right_margin, scl->width,
							   bevel->hi_color, bevel->lo_color,
							   bevel->hilo_color, bevel->lolo_color );
		if( left_margin < (int)scl->width )
		{
			if( get_flags( bevel->type, BEVEL_SOLID_INLINE ) )
			{
				if( y_out < imdec->bevel_top+(int)bevel->top_inline)
				{
					register int line = y_out - imdec->bevel_top;
					int left_delta  = bevel->left_inline-((line*bevel->left_inline/bevel->top_inline)) ;
					int right_delta = bevel->right_inline-((line*bevel->right_inline/bevel->top_inline)-1) ;

					draw_transp_bevel_sides( imdec, left_margin, left_delta,
									 	 	right_delta, right_margin );
					draw_transp_bevel_line ( imdec, left_delta-1, right_delta-1,
						 			 		ARGB32_ALPHA8(bevel_color)<<7,
									 		bevel->hihi_color, bevel->hi_color, bevel->hilo_color );

				}else if( y_out >= imdec->bevel_bottom - bevel->bottom_inline)
				{
					register int line = y_out - (imdec->bevel_bottom - bevel->bottom_inline);
					int left_delta  = (line*bevel->left_inline/bevel->bottom_inline)+1 ;
					int right_delta = (line*bevel->right_inline/bevel->bottom_inline)-1 ;

					draw_transp_bevel_sides( imdec,	left_margin, left_delta,
									 		right_delta, right_margin );
					draw_transp_bevel_line ( imdec, left_delta-1, right_delta,
						 			 		ARGB32_ALPHA8(shade_color)<<7,
									 		bevel->hilo_color, bevel->lo_color, bevel->lolo_color );

				}else
				{
					draw_transp_bevel_sides( imdec, left_margin, 0, 0, right_margin );
				}

			}
			else
			{
/* fprintf( stderr, __FUNCTION__ ":%d: y_out = %d, imdec->bevel_top = %d, bevel->top_inline = %d\n",
				__LINE__,  y_out, imdec->bevel_top, bevel->top_inline);
 */
 
				if( y_out < imdec->bevel_top+bevel->top_inline)
				{
					register int line = y_out - imdec->bevel_top;
					int left_delta  = bevel->left_inline-((line*bevel->left_inline/bevel->top_inline)) ;
					int right_delta = bevel->right_inline-((line*bevel->right_inline/bevel->top_inline)-1) ;
	    			CARD32 hda_bevel = (ARGB32_ALPHA8(bevel_color)<<8)/(bevel->left_inline+1) ;

					draw_fading_bevel_sides( imdec,	left_margin, left_delta,
									 	 	right_delta, right_margin );
/* fprintf( stderr, __FUNCTION__ ":%d: left_delta = %d, right_delta = %d, left_inline = %d, right_inline = %d, bevel_left = %d, bevel_right = %d\n",
				__LINE__,  left_delta, right_delta, bevel->left_inline, bevel->right_inline, imdec->bevel_left, imdec->bevel_right);
 */ 

					draw_transp_bevel_line ( imdec, left_delta-1, right_delta-1,
						 			 	 	hda_bevel*(left_delta+1),
									 	 	bevel->hihi_color, bevel->hi_color, bevel->hilo_color );

				}else if( y_out >= imdec->bevel_bottom - bevel->bottom_inline)
				{
					register int line = y_out - (imdec->bevel_bottom - bevel->bottom_inline);
					int left_delta  = (line*bevel->left_inline/bevel->bottom_inline)+1 ;
					int right_delta = (line*bevel->right_inline/bevel->bottom_inline)-1 ;
	    			CARD32 hda_shade = (ARGB32_ALPHA8(shade_color)<<8)/(bevel->right_inline+1) ;

					draw_fading_bevel_sides( imdec, left_margin, left_delta,
									 	 	right_delta, right_margin );

					draw_transp_bevel_line ( imdec, left_delta-1, right_delta,
						 			 	 	hda_shade*(right_delta+1),
									 	 	bevel->hilo_color, bevel->lo_color, bevel->lolo_color );
				}else
				{
					draw_fading_bevel_sides( imdec, left_margin, 0, 0, right_margin );
				}
			}
		}
	}
	++(imdec->next_line);
}

/* *********************************************************************/
/*						  ENCODER : 								  */
inline static void
tile_ximage_line( XImage *xim, unsigned int line, int step, int range )
{
	register int i ;
	int xim_step = step*xim->bytes_per_line ;
	char *src_line = xim->data+xim->bytes_per_line*line ;
	char *dst_line = src_line+xim_step ;
	int max_i = MIN((int)xim->height,(int)line+range), min_i = MAX(0,(int)line-range) ;
	for( i = line+step ; i < max_i && i >= min_i ; i+=step )
	{
		memcpy( dst_line, src_line, xim->bytes_per_line );
		dst_line += xim_step ;
	}
}

void
encode_image_scanline_mask_xim( ASImageOutput *imout, ASScanline *to_store )
{
#ifndef X_DISPLAY_MISSING
	ASImage *im = imout->im ;
	register XImage *xim = im->alt.mask_ximage ;
	if( imout->next_line < xim->height && imout->next_line >= 0 )
	{
		if( get_flags(to_store->flags, SCL_DO_ALPHA) )
		{
			CARD32 *a = to_store->alpha ;
			register int x = MIN((unsigned int)(xim->width), to_store->width);
			if( xim->depth == 8 )
			{
				CARD8 *dst = (CARD8*)xim->data+xim->bytes_per_line*imout->next_line ;
				while( --x >= 0 )
					dst[x] = (CARD8)(a[x]);
			}else
			{
				unsigned int nl = imout->next_line ;
				while( --x >= 0 )
					XPutPixel( xim, x, nl, (a[x] >= 0x7F)?1:0 );
			}
		}
		if( imout->tiling_step > 0 )
			tile_ximage_line( xim, imout->next_line,
			                  imout->bottom_to_top*imout->tiling_step,
							  (imout->tiling_range ? imout->tiling_range:imout->im->height) );
		imout->next_line += imout->bottom_to_top;
	}
#endif
}

void
encode_image_scanline_xim( ASImageOutput *imout, ASScanline *to_store )
{
#ifndef X_DISPLAY_MISSING
	register XImage *xim = imout->im->alt.ximage ;
	if( imout->next_line < xim->height && imout->next_line >= 0 )
	{
		unsigned char *dst = (unsigned char*)xim->data+imout->next_line*xim->bytes_per_line ;
		if( !get_flags(to_store->flags, SCL_DO_RED) )
			set_component( to_store->red, ARGB32_RED8(to_store->back_color), 0, to_store->width );
		if( !get_flags(to_store->flags, SCL_DO_GREEN) )
			set_component( to_store->green, ARGB32_GREEN8(to_store->back_color), 0, to_store->width );
		if( !get_flags(to_store->flags, SCL_DO_BLUE) )
			set_component( to_store->blue , ARGB32_BLUE8(to_store->back_color), 0, to_store->width );
		if( !get_flags(to_store->flags, SCL_DO_ALPHA) && (xim->depth == 24 || xim->depth == 32 ))
			set_component( to_store->alpha , ARGB32_ALPHA8(to_store->back_color), 0, to_store->width );
		if( xim->depth == imout->asv->visual_info.depth ) 
			PUT_SCANLINE(imout->asv, xim,to_store,imout->next_line, dst );
		else if( xim->depth == 16 ) 
			scanline2ximage16( imout->asv, xim, to_store,imout->next_line, dst);
		else if( xim->depth == 24 || xim->depth == 32 ) 
			scanline2ximage32( imout->asv, xim, to_store,imout->next_line, dst);
		else if( xim->depth == 15 ) 
			scanline2ximage15( imout->asv, xim, to_store,imout->next_line, dst);
		

		if( imout->tiling_step > 0 )
			tile_ximage_line( imout->im->alt.ximage, imout->next_line,
			                  imout->bottom_to_top*imout->tiling_step,
							  (imout->tiling_range ? imout->tiling_range:imout->im->height) );
		LOCAL_DEBUG_OUT( "flags = %lX", to_store->flags );
#if 1
		if( imout->out_format == ASA_ScratchXImageAndAlpha )
		{	
			if( get_flags(to_store->flags, SCL_DO_ALPHA) && get_flags( imout->im->flags, ASIM_DATA_NOT_USEFUL ))
			{
				int bytes_count, i ;
				int line = imout->next_line ;
				bytes_count = asimage_add_line(imout->im, IC_ALPHA, to_store->channels[IC_ALPHA]+to_store->offset_x, line);
				if( imout->tiling_step > 0 )
				{
					int range = (imout->tiling_range ? imout->tiling_range:imout->im->height);
					int max_i = MIN((int)imout->im->height,line+range), min_i = MAX(0,line-range) ;
					int step =  imout->bottom_to_top*imout->tiling_step;
					for( i = line+step ; i < max_i && i >= min_i ; i+=step )
					{
	/*						fprintf( stderr, "copy-encoding color %d, from lline %d to %d, %d bytes\n", color, imout->next_line, i, bytes_count );*/
						asimage_dup_line( imout->im, IC_ALPHA, line, i, bytes_count );
					}
		   		}
			}	 
		}
#endif
		imout->next_line += imout->bottom_to_top;
	}
#endif
}

void
encode_image_scanline_asim( ASImageOutput *imout, ASScanline *to_store )
{
LOCAL_DEBUG_CALLER_OUT( "imout->next_line = %d, imout->im->height = %d", imout->next_line, imout->im->height );
	if( imout->next_line < (int)imout->im->height && imout->next_line >= 0 )
	{
		CARD8 chan_fill[4];
		chan_fill[IC_RED]   = ARGB32_RED8  (to_store->back_color);
		chan_fill[IC_GREEN] = ARGB32_GREEN8(to_store->back_color);
		chan_fill[IC_BLUE]  = ARGB32_BLUE8 (to_store->back_color);
		chan_fill[IC_ALPHA] = ARGB32_ALPHA8(to_store->back_color);
		if( imout->tiling_step > 0 )
		{
			int bytes_count ;
			register int i, color ;
			int line = imout->next_line ;
			int range = (imout->tiling_range ? imout->tiling_range:imout->im->height);
			int max_i = MIN((int)imout->im->height,line+range), min_i = MAX(0,line-range) ;
			int step =  imout->bottom_to_top*imout->tiling_step;

			for( color = 0 ; color < IC_NUM_CHANNELS ; color++ )
			{
				if( get_flags(to_store->flags,0x01<<color))
					bytes_count = asimage_add_line(imout->im, color, to_store->channels[color]+to_store->offset_x, line);
				else if( chan_fill[color] != imout->chan_fill[color] )
					bytes_count = asimage_add_line_mono( imout->im, color, (CARD8)chan_fill[color], line);
				else
				{
					asimage_erase_line( imout->im, color, line );
					for( i = line+step ; i < max_i && i >= min_i ; i+=step )
						asimage_erase_line( imout->im, color, i );
					continue;
				}
				for( i = line+step ; i < max_i && i >= min_i ; i+=step )
				{
/*						fprintf( stderr, "copy-encoding color %d, from lline %d to %d, %d bytes\n", color, imout->next_line, i, bytes_count );*/
					asimage_dup_line( imout->im, color, line, i, bytes_count );
				}
			}
		}else
		{
			register int color ;
			for( color = 0 ; color < IC_NUM_CHANNELS ; color++ )
			{
				if( get_flags(to_store->flags,0x01<<color))
				{
					LOCAL_DEBUG_OUT( "encoding line %d for component %d offset = %d ", imout->next_line, color, to_store->offset_x );
					asimage_add_line(imout->im, color, to_store->channels[color]+to_store->offset_x, imout->next_line);
				}else if( chan_fill[color] != imout->chan_fill[color] )
				{
					LOCAL_DEBUG_OUT( "filling line %d for component %d with value %X", imout->next_line, color, chan_fill[color] );
					asimage_add_line_mono( imout->im, color, chan_fill[color], imout->next_line);
				}else
				{
					LOCAL_DEBUG_OUT( "erasing line %d for component %d", imout->next_line, color );
					asimage_erase_line( imout->im, color, imout->next_line );
				}
			}
		}
	}
	imout->next_line += imout->bottom_to_top;
}

inline static void
tile_argb32_line( ARGB32 *data, unsigned int line, int step, unsigned int width, unsigned int height, int range )
{
	register int i ;
	ARGB32 *src_line = data+line*width ;
	ARGB32 *dst_line = src_line+step*width ;
	int max_i = MIN((int)height,(int)line+range), min_i = MAX(0,(int)line-range) ;

	for( i = line+step ; i < max_i && i >= min_i ; i+=step )
	{
		memcpy( dst_line, src_line, width*sizeof(ARGB32));
		dst_line += step*width;
	}
}

void
encode_image_scanline_argb32( ASImageOutput *imout, ASScanline *to_store )
{
	register ARGB32 *data = imout->im->alt.argb32 ;
	if( imout->next_line < (int)imout->im->height && imout->next_line >= 0 )
	{
		register int x = imout->im->width;
		register CARD32 *alpha = to_store->alpha ;
		register CARD32 *red = to_store->red ;
		register CARD32 *green = to_store->green ;
		register CARD32 *blue = to_store->blue ;
		if( !get_flags(to_store->flags, SCL_DO_RED) )
			set_component( red, ARGB32_RED8(to_store->back_color), 0, to_store->width );
		if( !get_flags(to_store->flags, SCL_DO_GREEN) )
			set_component( green, ARGB32_GREEN8(to_store->back_color), 0, to_store->width );
		if( !get_flags(to_store->flags, SCL_DO_BLUE) )
			set_component( blue , ARGB32_BLUE8(to_store->back_color), 0, to_store->width );

		data += x*imout->next_line ;
		if( !get_flags(to_store->flags, SCL_DO_ALPHA) )
			while( --x >= 0 )
				data[x] = MAKE_ARGB32( 0xFF, red[x], green[x], blue[x] );
		else
			while( --x >= 0 )
				data[x] = MAKE_ARGB32( alpha[x], red[x], green[x], blue[x] );

		if( imout->tiling_step > 0 )
			tile_argb32_line( imout->im->alt.argb32, imout->next_line,
			                  imout->bottom_to_top*imout->tiling_step,
							  imout->im->width, imout->im->height,
							  (imout->tiling_range ? imout->tiling_range:imout->im->height));
		imout->next_line += imout->bottom_to_top;
	}
}


void
output_image_line_top( ASImageOutput *imout, ASScanline *new_line, int ratio )
{
	ASScanline *to_store = NULL ;
	/* caching and preprocessing line into our buffer : */
	if( new_line )
	{
		if( ratio > 1 )
            SCANLINE_FUNC_FILTERED(divide_component,*(new_line),*(imout->available),(CARD8)ratio,imout->available->width);
		else
            SCANLINE_FUNC_FILTERED(copy_component,*(new_line),*(imout->available),NULL,imout->available->width);
		imout->available->flags = new_line->flags ;
		imout->available->back_color = new_line->back_color ;
	}
	/* copying/encoding previously cahced line into destination image : */
	if( imout->used != NULL )
	{
		if( new_line != NULL )
            SCANLINE_FUNC_FILTERED(best_output_filter,*(imout->used),*(imout->available),0,imout->available->width);
		else
            SCANLINE_MOD_FILTERED(fine_output_filter_mod,*(imout->used),0,imout->used->width);
		to_store = imout->used ;
	}
	if( to_store )
    {
		imout->encode_image_scanline( imout, to_store );
    }
	/* rotating the buffers : */
	if( imout->buffer_shift > 0 )
	{
		if( new_line == NULL )
			imout->used = NULL ;
		else
			imout->used = imout->available ;
		imout->available = &(imout->buffer[0]);
		if( imout->available == imout->used )
			imout->available = &(imout->buffer[1]);
	}
}

void
output_image_line_fine( ASImageOutput *imout, ASScanline *new_line, int ratio )
{
	/* caching and preprocessing line into our buffer : */
    if( new_line )
	{
        SCANLINE_FUNC_FILTERED(fine_output_filter, *(new_line),*(imout->available),(CARD8)ratio,imout->available->width);
		imout->available->flags = new_line->flags ;
		imout->available->back_color = new_line->back_color ;
/*      SCANLINE_MOD(print_component,*(imout->available),0, new_line->width ); */
		/* copying/encoding previously cached line into destination image : */
		imout->encode_image_scanline( imout, imout->available );
	}
}

void
output_image_line_fast( ASImageOutput *imout, ASScanline *new_line, int ratio )
{
	/* caching and preprocessing line into our buffer : */
	if( new_line )
	{
        SCANLINE_FUNC_FILTERED(fast_output_filter,*(new_line),*(imout->available),(CARD8)ratio,imout->available->width);
		imout->available->flags = new_line->flags ;
		imout->available->back_color = new_line->back_color ;
		imout->encode_image_scanline( imout, imout->available );
	}
}

void
output_image_line_direct( ASImageOutput *imout, ASScanline *new_line, int ratio )
{
	/* caching and preprocessing line into our buffer : */
	if( new_line )
	{
        if( ratio > 1)
		{
            SCANLINE_FUNC_FILTERED(divide_component,*(new_line),*(imout->available),(CARD8)ratio,imout->available->width);
			imout->available->flags = new_line->flags ;
			imout->available->back_color = new_line->back_color ;
			imout->encode_image_scanline( imout, imout->available );
		}else
			imout->encode_image_scanline( imout, new_line );
	}
}

