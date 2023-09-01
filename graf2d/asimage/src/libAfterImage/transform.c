/*
 * Copyright (c) 2000,2001 Sasha Vasko <sasha at aftercode.net>
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
/*#undef NO_DEBUG_OUTPUT */
#undef USE_STUPID_GIMP_WAY_DESTROYING_COLORS
#undef LOCAL_DEBUG
#undef DO_CLOCKING
#undef DEBUG_HSV_ADJUSTMENT
#define USE_64BIT_FPU
#undef NEED_RBITSHIFT_FUNCS

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif
//#undef HAVE_MMX

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
#include <math.h>
#include <string.h>

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
#include "transform.h"

ASVisual __transform_fake_asv = {0};


/* ******************************************************************************/
/* below goes all kinds of funky stuff we can do with scanlines : 			   */
/* ******************************************************************************/
/* this will enlarge array based on count of items in dst per PAIR of src item with smoothing/scatter/dither */
/* the following formulas use linear approximation to calculate   */
/* color values for new pixels : 				  				  */
/* for scale factor of 2 we use this formula :    */
/* C = (-C1+3*C2+3*C3-C4)/4 					  */
/* or better :				 					  */
/* C = (-C1+5*C2+5*C3-C4)/8 					  */
#define INTERPOLATE_COLOR1(c) 			   	((c)<<QUANT_ERR_BITS)  /* nothing really to interpolate here */
#define INTERPOLATE_COLOR2(c1,c2,c3,c4)    	((((c2)<<2)+(c2)+((c3)<<2)+(c3)-(c1)-(c4))<<(QUANT_ERR_BITS-3))
#define INTERPOLATE_COLOR2_V(c1,c2,c3,c4)    	((((c2)<<2)+(c2)+((c3)<<2)+(c3)-(c1)-(c4))>>3)
/* for scale factor of 3 we use these formulas :  */
/* Ca = (-2C1+8*C2+5*C3-2C4)/9 		  			  */
/* Cb = (-2C1+5*C2+8*C3-2C4)/9 		  			  */
/* or better : 									  */
/* Ca = (-C1+5*C2+3*C3-C4)/6 		  			  */
/* Cb = (-C1+3*C2+5*C3-C4)/6 		  			  */
#define INTERPOLATE_A_COLOR3(c1,c2,c3,c4)  	(((((c2)<<2)+(c2)+((c3)<<1)+(c3)-(c1)-(c4))<<QUANT_ERR_BITS)/6)
#define INTERPOLATE_B_COLOR3(c1,c2,c3,c4)  	(((((c2)<<1)+(c2)+((c3)<<2)+(c3)-(c1)-(c4))<<QUANT_ERR_BITS)/6)
#define INTERPOLATE_A_COLOR3_V(c1,c2,c3,c4)  	((((c2)<<2)+(c2)+((c3)<<1)+(c3)-(c1)-(c4))/6)
#define INTERPOLATE_B_COLOR3_V(c1,c2,c3,c4)  	((((c2)<<1)+(c2)+((c3)<<2)+(c3)-(c1)-(c4))/6)
/* just a hypotesus, but it looks good for scale factors S > 3: */
/* Cn = (-C1+(2*(S-n)+1)*C2+(2*n+1)*C3-C4)/2S  	  			   */
/* or :
 * Cn = (-C1+(2*S+1)*C2+C3-C4+n*(2*C3-2*C2)/2S  			   */
/*       [ T                   [C2s]  [C3s]]   			       */
#define INTERPOLATION_Cs(c)	 		 	    ((c)<<1)
/*#define INTERPOLATION_TOTAL_START(c1,c2,c3,c4,S) 	(((S)<<1)*(c2)+((c3)<<1)+(c3)-c2-c1-c4)*/
#define INTERPOLATION_TOTAL_START(c1,c2,c3,c4,S) 	((((S)<<1)+1)*(c2)+(c3)-(c1)-(c4))
#define INTERPOLATION_TOTAL_STEP(c2,c3)  	((c3<<1)-(c2<<1))
#define INTERPOLATE_N_COLOR(T,S)		  	(((T)<<(QUANT_ERR_BITS-1))/(S))

#define AVERAGE_COLOR1(c) 					((c)<<QUANT_ERR_BITS)
#define AVERAGE_COLOR2(c1,c2)				(((c1)+(c2))<<(QUANT_ERR_BITS-1))
#define AVERAGE_COLORN(T,N)					(((T)<<QUANT_ERR_BITS)/N)

static inline void
enlarge_component12( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{/* expected len >= 2  */
	register int i = 0, k = 0;
	register int c1 = src[0], c4;
	--len; --len ;
	while( i < len )
	{
		c4 = src[i+2];
		/* that's right we can do that PRIOR as we calculate nothing */
		dst[k] = INTERPOLATE_COLOR1(src[i]) ;
		if( scales[i] == 2 )
		{
			register int c2 = src[i], c3 = src[i+1] ;
			c3 = INTERPOLATE_COLOR2(c1,c2,c3,c4);
			dst[++k] = (c3&0xFF000000 )?0:c3;
		}
		c1 = src[i];
		++k;
		++i;
	}

	/* to avoid one more if() in loop we moved tail part out of the loop : */
	if( scales[i] == 1 )
		dst[k] = INTERPOLATE_COLOR1(src[i]);
	else
	{
		register int c2 = src[i], c3 = src[i+1] ;
		c2 = INTERPOLATE_COLOR2(c1,c2,c3,c3);
		dst[k] = (c2&0xFF000000 )?0:c2;
	}
	dst[k+1] = INTERPOLATE_COLOR1(src[i+1]);
}

static inline void
enlarge_component23( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{/* expected len >= 2  */
	register int i = 0, k = 0;
	register int c1 = src[0], c4 = src[1];
	if( scales[0] == 1 )
	{/* special processing for first element - it can be 1 - others can only be 2 or 3 */
		dst[k] = INTERPOLATE_COLOR1(src[0]) ;
		++k;
		++i;
	}
	--len; --len ;
	while( i < len )
	{
		register int c2 = src[i], c3 = src[i+1] ;
		c4 = src[i+2];
		dst[k] = INTERPOLATE_COLOR1(c2) ;
		if( scales[i] == 2 )
		{
			c3 = INTERPOLATE_COLOR2(c1,c2,c3,c3);
			dst[++k] = (c3&0x7F000000 )?0:c3;
		}else
		{
			dst[++k] = INTERPOLATE_A_COLOR3(c1,c2,c3,c4);
			if( dst[k]&0x7F000000 )
				dst[k] = 0 ;
			c3 = INTERPOLATE_B_COLOR3(c1,c2,c3,c3);
			dst[++k] = (c3&0x7F000000 )?0:c3;
		}
		c1 = c2 ;
		++k;
		++i;
	}
	/* to avoid one more if() in loop we moved tail part out of the loop : */
	{
		register int c2 = src[i], c3 = src[i+1] ;
		dst[k] = INTERPOLATE_COLOR1(c2) ;
		if( scales[i] == 2 )
		{
			c2 = INTERPOLATE_COLOR2(c1,c2,c3,c3);
			dst[k+1] = (c2&0x7F000000 )?0:c2;
		}else
		{
			if( scales[i] == 1 )
				--k;
			else
			{
				dst[++k] = INTERPOLATE_A_COLOR3(c1,c2,c3,c3);
				if( dst[k]&0x7F000000 )
					dst[k] = 0 ;
				c2 = INTERPOLATE_B_COLOR3(c1,c2,c3,c3);
  				dst[k+1] = (c2&0x7F000000 )?0:c2;
			}
		}
	}
 	dst[k+2] = INTERPOLATE_COLOR1(src[i+1]) ;
}

/* this case is more complex since we cannot really hardcode coefficients
 * visible artifacts on smooth gradient-like images
 */
static inline void
enlarge_component( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{/* we skip all checks as it is static function and we want to optimize it
  * as much as possible */
	int i = 0;
	int c1 = src[0];
	register int T ;
	--len ;
	if( len < 1 )
	{
		CARD32 c = INTERPOLATE_COLOR1(c1) ;
		for( i = 0 ; i < scales[0] ; ++i )
			dst[i] = c;
		return;
	}
	do
	{
		register short S = scales[i];
		register int step = INTERPOLATION_TOTAL_STEP(src[i],src[i+1]);

		if( i+1 == len )
			T = INTERPOLATION_TOTAL_START(c1,src[i],src[i+1],src[i+1],S);
		else
			T = INTERPOLATION_TOTAL_START(c1,src[i],src[i+1],src[i+2],S);

/*		LOCAL_DEBUG_OUT( "pixel %d, S = %d, step = %d", i, S, step );*/
		if( step )
		{
			register int n = 0 ;
			do
			{
				dst[n] = (T&0x7F000000)?0:INTERPOLATE_N_COLOR(T,S);
				if( ++n >= S ) break;
				T = (int)T + (int)step;
			}while(1);
			dst += n ;
		}else
		{
			register CARD32 c = (T&0x7F000000)?0:INTERPOLATE_N_COLOR(T,S);
			while(--S >= 0){	dst[S] = c;	}
			dst += scales[i] ;
		}
		c1 = src[i];
	}while(++i < len );
	*dst = INTERPOLATE_COLOR1(src[i]) ;
/*LOCAL_DEBUG_OUT( "%d pixels written", k );*/
}

static inline void
enlarge_component_dumb( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{/* we skip all checks as it is static function and we want to optimize it
  * as much as possible */
	int i = 0, k = 0;
	do
	{
		register CARD32 c = INTERPOLATE_COLOR1(src[i]);
		int max_k = k+scales[i];
		do
		{
			dst[k] = c ;
		}while( ++k < max_k );
	}while( ++i < len );
}

/* this will shrink array based on count of items in src per one dst item with averaging */
static inline void
shrink_component( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{/* we skip all checks as it is static function and we want to optimize it
  * as much as possible */
	register int i = -1, k = -1;
	while( ++k < len )
	{
		register int reps = scales[k] ;
		register int c1 = src[++i];
/*LOCAL_DEBUG_OUT( "pixel = %d, scale[k] = %d", k, reps );*/
		if( reps == 1 )
			dst[k] = AVERAGE_COLOR1(c1);
		else if( reps == 2 )
		{
			++i;
			dst[k] = AVERAGE_COLOR2(c1,src[i]);
		}else
		{
			reps += i-1;
			while( reps > i )
			{
				++i ;
				c1 += src[i];
			}
			{
				register short S = scales[k];
				dst[k] = AVERAGE_COLORN(c1,S);
			}
		}
	}
}
static inline void
shrink_component11( register CARD32 *src, register CARD32 *dst, int *scales, int len )
{
	register int i ;
	for( i = 0 ; i < len ; ++i )
		dst[i] = AVERAGE_COLOR1(src[i]);
}


static inline void
reverse_component( register CARD32 *src, register CARD32 *dst, int *unused, int len )
{
	register int i = 0;
	src += len-1 ;
	do
	{
		dst[i] = src[-i];
	}while(++i < len );
}

static inline void
add_component( CARD32 *src, CARD32 *incr, int *scales, int len )
{
	len += len&0x01;
#ifdef HAVE_MMX   
#if 1
	if( asimage_use_mmx )
	{
		int i = 0;
		__m64  *vdst = (__m64*)&(src[0]);
		__m64  *vinc = (__m64*)&(incr[0]);
		len = len>>1;
		do{
			vdst[i] = _mm_add_pi32(vdst[i],vinc[i]);  /* paddd */
		}while( ++i  < len );
		_mm_empty();
	}else
#else
	if( asimage_use_mmx )
	{
		double *ddst = (double*)&(src[0]);
		double *dinc = (double*)&(incr[0]);
		len = len>>1;
		do{
			asm volatile
       		(
            	"movq %0, %%mm0  \n\t" /* load 8 bytes from src[i] into MM0 */
            	"paddd %1, %%mm0 \n\t" /* MM0=src[i]>>1              */
            	"movq %%mm0, %0  \n\t" /* store the result in dest */
				: "=m" (ddst[i])       /* %0 */
				:  "m"  (dinc[i])       /* %2 */
	        );
		}while( ++i < len );
	}else
#endif
#endif
	{
		register int c1, c2;
		int i = 0;
		do{
			c1 = (int)src[i] + (int)incr[i] ;
			c2 = (int)src[i+1] + (int)incr[i+1] ;
			src[i] = c1;
			src[i+1] = c2;
			i += 2 ;
		}while( i < len );
	}
}

#ifdef NEED_RBITSHIFT_FUNCS
static inline void
rbitshift_component( register CARD32 *src, register CARD32 *dst, int shift, int len )
{
	register int i ;
	for( i = 0 ; i < len ; ++i )
		dst[i] = src[i]>>shift;
}
#endif

static inline void
start_component_interpolation( CARD32 *c1, CARD32 *c2, CARD32 *c3, CARD32 *c4, register CARD32 *T, register CARD32 *step, int S, int len)
{
	register int i;
	for( i = 0 ; i < len ; i++ )
	{
		register int rc2 = c2[i], rc3 = c3[i] ;
		T[i] = INTERPOLATION_TOTAL_START(c1[i],rc2,rc3,c4[i],S)/(S<<1);
		step[i] = INTERPOLATION_TOTAL_STEP(rc2,rc3)/(S<<1);
	}
}

static inline void
component_interpolation_hardcoded( CARD32 *c1, CARD32 *c2, CARD32 *c3, CARD32 *c4, register CARD32 *T, CARD32 *unused, CARD16 kind, int len)
{
	register int i;
	if( kind == 1 )
	{
		for( i = 0 ; i < len ; i++ )
		{
#if 1
			/* its seems that this simple formula is completely sufficient
			   and even better then more complicated one : */
			T[i] = (c2[i]+c3[i])>>1 ;
#else
    		register int minus = c1[i]+c4[i] ;
			register int plus  = (c2[i]<<1)+c2[i]+(c3[i]<<1)+c3[i];

			T[i] = ( (plus>>1) < minus )?(c2[i]+c3[i])>>1 :
								   		 (plus-minus)>>2;
#endif
		}
	}else if( kind == 2 )
	{
		for( i = 0 ; i < len ; i++ )
		{
    		register int rc1 = c1[i], rc2 = c2[i], rc3 = c3[i] ;
			T[i] = INTERPOLATE_A_COLOR3_V(rc1,rc2,rc3,c4[i]);
		}
	}else
		for( i = 0 ; i < len ; i++ )
		{
    		register int rc1 = c1[i], rc2 = c2[i], rc3 = c3[i] ;
			T[i] = INTERPOLATE_B_COLOR3_V(rc1,rc2,rc3,c4[i]);
		}
}

#ifdef NEED_RBITSHIFT_FUNCS
static inline void
divide_component_mod( register CARD32 *data, CARD16 ratio, int len )
{
	register int i ;
	for( i = 0 ; i < len ; ++i )
		data[i] /= ratio;
}

static inline void
rbitshift_component_mod( register CARD32 *data, int bits, int len )
{
	register int i ;
	for( i = 0 ; i < len ; ++i )
		data[i] = data[i]>>bits;
}
#endif
void
print_component( register CARD32 *data, int nonsense, int len )
{
	register int i ;
	for( i = 0 ; i < len ; ++i )
		fprintf( stderr, " %8.8lX", (long)data[i] );
	fprintf( stderr, "\n");
}

static inline void
tint_component_mod( register CARD32 *data, CARD16 ratio, int len )
{
	register int i ;
	if( ratio == 255 )
		for( i = 0 ; i < len ; ++i )
			data[i] = data[i]<<8;
	else if( ratio == 128 )
		for( i = 0 ; i < len ; ++i )
			data[i] = data[i]<<7;
	else if( ratio == 0 )
		for( i = 0 ; i < len ; ++i )
			data[i] = 0;
	else
		for( i = 0 ; i < len ; ++i )
			data[i] = data[i]*ratio;
}

static inline void
make_component_gradient16( register CARD32 *data, CARD16 from, CARD16 to, CARD8 seed, int len )
{
	register int i ;
	long incr = (((long)to<<8)-((long)from<<8))/len ;

	if( incr == 0 )
		for( i = 0 ; i < len ; ++i )
			data[i] = from;
	else
	{
		long curr = from<<8;
		curr += ((long)(((CARD32)seed)<<8) > incr)?incr:((CARD32)seed)<<8 ;
		for( i = 0 ; i < len ; ++i )
		{/* we make calculations in 24bit per chan, then convert it back to 16 and
		  * carry over half of the quantization error onto the next pixel */
			data[i] = curr>>8;
			curr += ((curr&0x00FF)>>1)+incr ;
		}
	}
}


static inline void
copytintpad_scanline( ASScanline *src, ASScanline *dst, int offset, ARGB32 tint )
{
	register int i ;
	CARD32 chan_tint[4], chan_fill[4] ;
	int color ;
	int copy_width = src->width, dst_offset = 0, src_offset = 0;

	if( offset+(int)src->width < 0 || offset > (int)dst->width )
		return;
	chan_tint[IC_RED]   = ARGB32_RED8  (tint)<<1;
	chan_tint[IC_GREEN] = ARGB32_GREEN8(tint)<<1;
	chan_tint[IC_BLUE]  = ARGB32_BLUE8 (tint)<<1;
	chan_tint[IC_ALPHA] = ARGB32_ALPHA8(tint)<<1;
	chan_fill[IC_RED]   = ARGB32_RED8  (dst->back_color)<<dst->shift;
	chan_fill[IC_GREEN] = ARGB32_GREEN8(dst->back_color)<<dst->shift;
	chan_fill[IC_BLUE]  = ARGB32_BLUE8 (dst->back_color)<<dst->shift;
	chan_fill[IC_ALPHA] = ARGB32_ALPHA8(dst->back_color)<<dst->shift;
	if( offset < 0 )
		src_offset = -offset ;
	else
		dst_offset = offset ;
	copy_width = MIN( src->width-src_offset, dst->width-dst_offset );

	dst->flags = src->flags ;
	for( color = 0 ; color < IC_NUM_CHANNELS ; ++color )
	{
		register CARD32 *psrc = src->channels[color]+src_offset;
		register CARD32 *pdst = dst->channels[color];
		int ratio = chan_tint[color];
/*	fprintf( stderr, "channel %d, tint is %d(%X), src_width = %d, src_offset = %d, dst_width = %d, dst_offset = %d psrc = %p, pdst = %p\n", color, ratio, ratio, src->width, src_offset, dst->width, dst_offset, psrc, pdst );
*/
		{
/*			register CARD32 fill = chan_fill[color]; */
			for( i = 0 ; i < dst_offset ; ++i )
				pdst[i] = 0;
			pdst += dst_offset ;
		}

		if( get_flags(src->flags, 0x01<<color) )
		{
			if( ratio >= 254 )
				for( i = 0 ; i < copy_width ; ++i )
					pdst[i] = psrc[i]<<8;
			else if( ratio == 128 )
				for( i = 0 ; i < copy_width ; ++i )
					pdst[i] = psrc[i]<<7;
			else if( ratio == 0 )
				for( i = 0 ; i < copy_width ; ++i )
					pdst[i] = 0;
			else
				for( i = 0 ; i < copy_width ; ++i )
					pdst[i] = psrc[i]*ratio;
		}else
		{
		    ratio = ratio*chan_fill[color];
			for( i = 0 ; i < copy_width ; ++i )
				pdst[i] = ratio;
			set_flags( dst->flags, (0x01<<color));
		}
		{
/*			register CARD32 fill = chan_fill[color]; */
			for( ; i < (int)dst->width-dst_offset ; ++i )
				pdst[i] = 0;
/*				print_component(pdst, 0, dst->width ); */
		}
	}
}

/* **********************************************************************************************/
/* drawing gradient on scanline :  															   */
/* **********************************************************************************************/
void
make_gradient_scanline( ASScanline *scl, ASGradient *grad, ASFlagType filter, ARGB32 seed )
{
	if( scl && grad && filter != 0 )
	{
		int offset = 0, step, i, max_i = grad->npoints - 1 ;
		ARGB32 last_color = ARGB32_Black ;
		int last_idx = 0;
		double last_offset = 0., *offsets = grad->offset ;
		int *used = safecalloc(max_i+1, sizeof(int));
		/* lets find the color of the very first point : */
		for( i = 0 ; i <= max_i ; ++i )
			if( offsets[i] <= 0. )
			{
				last_color = grad->color[i] ;
				last_idx = i ;
				used[i] = 1 ;
				break;
			}

		for( i = 0  ; i <= max_i ; i++ )
		{
			register int k ;
			int new_idx = -1 ;
			/* now lets find the next point  : */
			for( k = 0 ; k <= max_i ; ++k )
			{
				if( used[k]==0 && offsets[k] >= last_offset )
				{
					if( new_idx < 0 )
						new_idx = k ;
					else if( offsets[new_idx] > offsets[k] )
						new_idx = k ;
					else
					{
						register int d1 = new_idx-last_idx ;
						register int d2 = k - last_idx ;
						if( d1*d1 > d2*d2 )
							new_idx = k ;
					}
				}
			}
			if( new_idx < 0 )
				break;
			used[new_idx] = 1 ;
			step = (int)((grad->offset[new_idx] * (double)scl->width) - (double)offset) ;
/*			fprintf( stderr, __FUNCTION__":%d>last_offset = %f, last_color = %8.8X, new_idx = %d, max_i = %d, new_offset = %f, new_color = %8.8X, step = %d, offset = %d\n", __LINE__, last_offset, last_color, new_idx, max_i, offsets[new_idx], grad->color[new_idx], step, offset ); */
			if( step > (int)scl->width-offset )
				step = (int)scl->width-offset ;
			if( step > 0 )
			{
				int color ;
				for( color = 0 ; color < IC_NUM_CHANNELS ; ++color )
					if( get_flags( filter, 0x01<<color ) )
					{
						LOCAL_DEBUG_OUT("channel %d from #%4.4lX to #%4.4lX, ofset = %d, step = %d",
	 	 									color, ARGB32_CHAN8(last_color,color)<<8, ARGB32_CHAN8(grad->color[new_idx],color)<<8, offset, step );
						make_component_gradient16( scl->channels[color]+offset,
												   (CARD16)(ARGB32_CHAN8(last_color,color)<<8),
												   (CARD16)(ARGB32_CHAN8(grad->color[new_idx],color)<<8),
												   (CARD8)ARGB32_CHAN8(seed,color),
												   step);
					}
				offset += step ;
			}
			last_offset = offsets[new_idx];
			last_color = grad->color[new_idx];
			last_idx = new_idx ;
		}
		scl->flags = filter ;
		free( used );
	}
}

/* **********************************************************************************************/
/* Scaling code ; 																			   */
/* **********************************************************************************************/
Bool
check_scale_parameters( ASImage *src, int src_width, int src_height, int *to_width, int *to_height )
{
	if( src == NULL )
		return False;

	if( *to_width == 0 )
		*to_width = src_width ;
	else if( *to_width < 2 )
		*to_width = 2 ;
	if( *to_height == 0 )
		*to_height = src_height ;
	else if( *to_height < 2 )
		*to_height = 2 ;
	return True;
}

int *
make_scales( int from_size, int to_size, int tail )
{
	int *scales ;
    int smaller = MIN(from_size,to_size);
    int bigger  = MAX(from_size,to_size);
	register int i = 0, k = 0;
	int eps;
    LOCAL_DEBUG_OUT( "from %d to %d tail %d", from_size, to_size, tail );
	scales = safecalloc( smaller+tail, sizeof(int));
	if( smaller <= 1 ) 
	{
		scales[0] = bigger ; 
		return scales;
	}
#if 1
	else if( smaller == bigger )
	{
		for ( i = 0 ; i < smaller ; i++ )
			scales[i] = 1 ; 
		return scales;	
	}
#endif
	if( from_size >= to_size )
		tail = 0 ;
	if( tail != 0 )
    {
        bigger-=tail ;
        if( (smaller-=tail) == 1 ) 
		{
			scales[0] = bigger ; 
			return scales;
		}	
    }else if( smaller == 2 ) 
	{
		scales[1] = bigger/2 ; 
		scales[0] = bigger - scales[1] ; 
		return scales ;
	}

    eps = -bigger/2;
    LOCAL_DEBUG_OUT( "smaller %d, bigger %d, eps %d", smaller, bigger, eps );
    /* now using Bresengham algoritm to fiill the scales :
	 * since scaling is merely transformation
	 * from 0:bigger space (x) to 0:smaller space(y)*/
	for ( i = 0 ; i < bigger ; i++ )
	{
		++scales[k];
		eps += smaller;
        LOCAL_DEBUG_OUT( "scales[%d] = %d, i = %d, k = %d, eps %d", k, scales[k], i, k, eps );
        if( eps+eps >= bigger )
		{
			++k ;
			eps -= bigger ;
		}
	}

	return scales;
}

/* *******************************************************************/
void
scale_image_down( ASImageDecoder *imdec, ASImageOutput *imout, int h_ratio, int *scales_h, int* scales_v)
{
	ASScanline dst_line, total ;
	int k = -1;
	int max_k 	 = imout->im->height,
		line_len = MIN(imout->im->width, imdec->out_width);

	prepare_scanline( imout->im->width, QUANT_ERR_BITS, &dst_line, imout->asv->BGR_mode );
	prepare_scanline( imout->im->width, QUANT_ERR_BITS, &total, imout->asv->BGR_mode );
	while( ++k < max_k )
	{
		int reps = scales_v[k] ;
		imdec->decode_image_scanline( imdec );
		total.flags = imdec->buffer.flags ;
		CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,total,scales_h,line_len);

		while( --reps > 0 )
		{
			imdec->decode_image_scanline( imdec );
			total.flags = imdec->buffer.flags ;
			CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,dst_line,scales_h,line_len);
			SCANLINE_FUNC(add_component,total,dst_line,NULL,total.width);
		}

		imout->output_image_scanline( imout, &total, scales_v[k] );
	}
	free_scanline(&dst_line, True);
	free_scanline(&total, True);
}

void
scale_image_up( ASImageDecoder *imdec, ASImageOutput *imout, int h_ratio, int *scales_h, int* scales_v)
{
	ASScanline src_lines[4], *c1, *c2, *c3, *c4 = NULL;
	int i = 0, max_i,
		line_len = MIN(imout->im->width, imdec->out_width),
		out_width = imout->im->width;
	ASScanline step ;

	prepare_scanline( out_width, 0, &(src_lines[0]), imout->asv->BGR_mode);
	prepare_scanline( out_width, 0, &(src_lines[1]), imout->asv->BGR_mode);
	prepare_scanline( out_width, 0, &(src_lines[2]), imout->asv->BGR_mode);
	prepare_scanline( out_width, 0, &(src_lines[3]), imout->asv->BGR_mode);
	prepare_scanline( out_width, QUANT_ERR_BITS, &step, imout->asv->BGR_mode );

/*	set_component(src_lines[0].red,0x00000000,0,out_width*3); */
	imdec->decode_image_scanline( imdec );
	src_lines[1].flags = imdec->buffer.flags ;
	CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,src_lines[1],scales_h,line_len);

	step.flags = src_lines[0].flags = src_lines[1].flags ;

	SCANLINE_FUNC(copy_component,src_lines[1],src_lines[0],0,out_width);

	imdec->decode_image_scanline( imdec );
	src_lines[2].flags = imdec->buffer.flags ;
	CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,src_lines[2],scales_h,line_len);

	i = 0 ;
	max_i = imdec->out_height-1 ;
	LOCAL_DEBUG_OUT( "i = %d, max_i = %d", i, max_i );
	do
	{
		int S = scales_v[i] ;
		c1 = &(src_lines[i&0x03]);
		c2 = &(src_lines[(i+1)&0x03]);
		c3 = &(src_lines[(i+2)&0x03]);
		c4 = &(src_lines[(i+3)&0x03]);

		if( i+1 < max_i )
		{
			imdec->decode_image_scanline( imdec );
			c4->flags = imdec->buffer.flags ;
			CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,*c4,scales_h,line_len);
		}
		/* now we'll prepare total and step : */
        if( S > 0 )
        {
            imout->output_image_scanline( imout, c2, 1);
            if( S > 1 )
            {
                if( S == 2 )
                {
                    SCANLINE_COMBINE(component_interpolation_hardcoded,*c1,*c2,*c3,*c4,*c1,*c1,1,out_width);
                    imout->output_image_scanline( imout, c1, 1);
                }else if( S == 3 )
                {
                    SCANLINE_COMBINE(component_interpolation_hardcoded,*c1,*c2,*c3,*c4,*c1,*c1,2,out_width);
                    imout->output_image_scanline( imout, c1, 1);
                    SCANLINE_COMBINE(component_interpolation_hardcoded,*c1,*c2,*c3,*c4,*c1,*c1,3,out_width);
                    imout->output_image_scanline( imout, c1, 1);
                }else
                {
                    SCANLINE_COMBINE(start_component_interpolation,*c1,*c2,*c3,*c4,*c1,step,S,out_width);
                    do
                    {
                        imout->output_image_scanline( imout, c1, 1);
                        if((--S)<=1)
                            break;
                        SCANLINE_FUNC(add_component,*c1,step,NULL,out_width );
                    }while(1);
                }
            }
        }
	}while( ++i < max_i );
    imout->output_image_scanline( imout, c3, 1);
	free_scanline(&step, True);
	free_scanline(&(src_lines[3]), True);
	free_scanline(&(src_lines[2]), True);
	free_scanline(&(src_lines[1]), True);
	free_scanline(&(src_lines[0]), True);
}

void
scale_image_up_dumb( ASImageDecoder *imdec, ASImageOutput *imout, int h_ratio, int *scales_h, int* scales_v)
{
	ASScanline src_line;
	int	line_len = MIN(imout->im->width, imdec->out_width);
	int	out_width = imout->im->width;
	int y = 0 ;

	prepare_scanline( out_width, QUANT_ERR_BITS, &src_line, imout->asv->BGR_mode );

	imout->tiling_step = 1 ;
	LOCAL_DEBUG_OUT( "imdec->next_line = %d, imdec->out_height = %d", imdec->next_line, imdec->out_height );
	while( y < (int)imdec->out_height )
	{
		imdec->decode_image_scanline( imdec );
		src_line.flags = imdec->buffer.flags ;
		CHOOSE_SCANLINE_FUNC(h_ratio,imdec->buffer,src_line,scales_h,line_len);
		imout->tiling_range = scales_v[y];
		LOCAL_DEBUG_OUT( "y = %d, tiling_range = %d", y, imout->tiling_range );
		imout->output_image_scanline( imout, &src_line, 1);
		imout->next_line += scales_v[y]-1;
		++y;
	}
	free_scanline(&src_line, True);
}


static inline ASImage *
create_destination_image( unsigned int width, unsigned int height, ASAltImFormats format, 
						  unsigned int compression, ARGB32 back_color )
{
	ASImage *dst = create_asimage(width, height, compression);
	if( dst )
	{
		if( format != ASA_ASImage )
			set_flags( dst->flags, ASIM_DATA_NOT_USEFUL );
	
		dst->back_color = back_color ;
	}
	return dst ;
}
						  

/* *****************************************************************************/
/* ASImage transformations : 												  */
/* *****************************************************************************/
ASImage *
scale_asimage( ASVisual *asv, ASImage *src, int to_width, int to_height,
			   ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageOutput  *imout ;
	ASImageDecoder *imdec;
	int h_ratio ;
	int *scales_h = NULL, *scales_v = NULL;
	START_TIME(started);
	
	if( asv == NULL ) 	asv = &__transform_fake_asv ;
	
	if( !check_scale_parameters(src,src->width, src->height,&to_width,&to_height) )
		return NULL;
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, 0, 0, 0, 0, NULL)) == NULL )
		return NULL;

	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color );

	if( to_width == src->width )
		h_ratio = 0;
	else if( to_width < src->width )
		h_ratio = 1;
	else
	{
		if ( quality == ASIMAGE_QUALITY_POOR )
			h_ratio = 1 ;
		else if( src->width > 1 )
		{
			h_ratio = (to_width/(src->width-1))+1;
			if( h_ratio*(src->width-1) < to_width )
				++h_ratio ;
		}else
			h_ratio = to_width ;
		++h_ratio ;
	}
	scales_h = make_scales( src->width, to_width, ( quality == ASIMAGE_QUALITY_POOR )?0:1 );
	scales_v = make_scales( src->height, to_height, ( quality == ASIMAGE_QUALITY_POOR  || src->height <= 3)?0:1 );
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
	{
	  register int i ;
	  for( i = 0 ; i < MIN(src->width, to_width) ; i++ )
		fprintf( stderr, " %d", scales_h[i] );
	  fprintf( stderr, "\n" );
	  for( i = 0 ; i < MIN(src->height, to_height) ; i++ )
		fprintf( stderr, " %d", scales_v[i] );
	  fprintf( stderr, "\n" );
	}
#endif
	if((imout = start_image_output( asv, dst, out_format, QUANT_ERR_BITS, quality )) == NULL )
	{
        destroy_asimage( &dst );
	}else
	{
		if( to_height <= src->height ) 					   /* scaling down */
			scale_image_down( imdec, imout, h_ratio, scales_h, scales_v );
		else if( quality == ASIMAGE_QUALITY_POOR || src->height <= 3 ) 
			scale_image_up_dumb( imdec, imout, h_ratio, scales_h, scales_v );
		else
			scale_image_up( imdec, imout, h_ratio, scales_h, scales_v );
		stop_image_output( &imout );
	}
	free( scales_h );
	free( scales_v );
	stop_image_decoding( &imdec );
	SHOW_TIME("", started);
	return dst;
}

ASImage *
scale_asimage2( ASVisual *asv, ASImage *src, 
					int clip_x, int clip_y, 
					int clip_width, int clip_height, 
					int to_width, int to_height,
			   		ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageOutput  *imout ;
	ASImageDecoder *imdec;
	int h_ratio ;
	int *scales_h = NULL, *scales_v = NULL;
	START_TIME(started);

	if( src == NULL ) 
		return NULL;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if( clip_width == 0 )
		clip_width = src->width ;
	if( clip_height == 0 )
		clip_height = src->height ;
	if( !check_scale_parameters(src, clip_width, clip_height, &to_width, &to_height) )
		return NULL;
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, clip_x, clip_y, clip_width, clip_height, NULL)) == NULL )
		return NULL;

	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color );

	if( to_width == clip_width )
		h_ratio = 0;
	else if( to_width < clip_width )
		h_ratio = 1;
	else
	{
		if ( quality == ASIMAGE_QUALITY_POOR )
			h_ratio = 1 ;
		else if( clip_width > 1 )
		{
			h_ratio = (to_width/(clip_width-1))+1;
			if( h_ratio*(clip_width-1) < to_width )
				++h_ratio ;
		}else
			h_ratio = to_width ;
		++h_ratio ;
	}
	scales_h = make_scales( clip_width, to_width, ( quality == ASIMAGE_QUALITY_POOR )?0:1 );
	scales_v = make_scales( clip_height, to_height, ( quality == ASIMAGE_QUALITY_POOR  || clip_height <= 3)?0:1 );
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
	{
	  register int i ;
	  for( i = 0 ; i < MIN(clip_width, to_width) ; i++ )
		fprintf( stderr, " %d", scales_h[i] );
	  fprintf( stderr, "\n" );
	  for( i = 0 ; i < MIN(clip_height, to_height) ; i++ )
		fprintf( stderr, " %d", scales_v[i] );
	  fprintf( stderr, "\n" );
	}
#endif
	if((imout = start_image_output( asv, dst, out_format, QUANT_ERR_BITS, quality )) == NULL )
	{
        destroy_asimage( &dst );
	}else
	{
		if( to_height <= clip_height ) 					   /* scaling down */
			scale_image_down( imdec, imout, h_ratio, scales_h, scales_v );
		else if( quality == ASIMAGE_QUALITY_POOR || clip_height <= 3 ) 
			scale_image_up_dumb( imdec, imout, h_ratio, scales_h, scales_v );
		else
			scale_image_up( imdec, imout, h_ratio, scales_h, scales_v );
		stop_image_output( &imout );
	}
	free( scales_h );
	free( scales_v );
	stop_image_decoding( &imdec );
	SHOW_TIME("", started);
	return dst;
}

ASImage *
tile_asimage( ASVisual *asv, ASImage *src,
		      int offset_x, int offset_y,
			  int to_width,
			  int to_height,
			  ARGB32 tint,
			  ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder *imdec ;
	ASImageOutput  *imout ;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

LOCAL_DEBUG_CALLER_OUT( "src = %p, offset_x = %d, offset_y = %d, to_width = %d, to_height = %d, tint = #%8.8lX", src, offset_x, offset_y, to_width, to_height, tint );
	if( src== NULL || (imdec = start_image_decoding(asv, src, SCL_DO_ALL, offset_x, offset_y, to_width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		return NULL;
	}

	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color );

	if((imout = start_image_output( asv, dst, out_format, (tint!=0)?8:0, quality)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image output%s", "");
        destroy_asimage( &dst );
    }else
	{
		int y, max_y = to_height;
LOCAL_DEBUG_OUT("tiling actually...%s", "");
		if( to_height > src->height )
		{
			imout->tiling_step = src->height ;
			max_y = src->height ;
		}
		if( tint != 0 )
		{
			for( y = 0 ; y < max_y ; y++  )
			{
				imdec->decode_image_scanline( imdec );
				tint_component_mod( imdec->buffer.red, (CARD16)(ARGB32_RED8(tint)<<1), to_width );
				tint_component_mod( imdec->buffer.green, (CARD16)(ARGB32_GREEN8(tint)<<1), to_width );
  				tint_component_mod( imdec->buffer.blue, (CARD16)(ARGB32_BLUE8(tint)<<1), to_width );
				tint_component_mod( imdec->buffer.alpha, (CARD16)(ARGB32_ALPHA8(tint)<<1), to_width );
				imout->output_image_scanline( imout, &(imdec->buffer), 1);
			}
		}else
			for( y = 0 ; y < max_y ; y++  )
			{
				imdec->decode_image_scanline( imdec );
				imout->output_image_scanline( imout, &(imdec->buffer), 1);
			}
		stop_image_output( &imout );
	}
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	return dst;
}

ASImage *
merge_layers( ASVisual *asv,
				ASImageLayer *layers, int count,
			  	int dst_width,
			  	int dst_height,
			  	ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder **imdecs ;
	ASImageOutput  *imout ;
	ASImageLayer *pcurr = layers;
	int i ;
	ASScanline dst_line ;
	START_TIME(started);

LOCAL_DEBUG_CALLER_OUT( "dst_width = %d, dst_height = %d", dst_width, dst_height );
	
	dst = create_destination_image( dst_width, dst_height, out_format, compression_out, ARGB32_DEFAULT_BACK_COLOR );
	if( dst == NULL )
		return NULL;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	prepare_scanline( dst_width, QUANT_ERR_BITS, &dst_line, asv->BGR_mode );
	dst_line.flags = SCL_DO_ALL ;

	imdecs = safecalloc( count+20, sizeof(ASImageDecoder*));

	for( i = 0 ; i < count ; i++ )
	{
		/* all laayers but first must have valid image or solid_color ! */
		if( (pcurr->im != NULL || pcurr->solid_color != 0 || i == 0) &&
			pcurr->dst_x < (int)dst_width && pcurr->dst_x+(int)pcurr->clip_width > 0 )
		{
			imdecs[i] = start_image_decoding(asv, pcurr->im, SCL_DO_ALL,
				                             pcurr->clip_x, pcurr->clip_y,
											 pcurr->clip_width, pcurr->clip_height,
											 pcurr->bevel);
			if( pcurr->bevel_width != 0 && pcurr->bevel_height != 0 )
				set_decoder_bevel_geom( imdecs[i],
				                        pcurr->bevel_x, pcurr->bevel_y,
										pcurr->bevel_width, pcurr->bevel_height );
  			if( pcurr->tint == 0 && i != 0 )
				set_decoder_shift( imdecs[i], 8 );
			if( pcurr->im == NULL )
				set_decoder_back_color( imdecs[i], pcurr->solid_color );
		}
		if( pcurr->next == pcurr )
			break;
		else
			pcurr = (pcurr->next!=NULL)?pcurr->next:pcurr+1 ;
	}
	if( i < count )
		count = i+1 ;

	if(imdecs[0] == NULL || (imout = start_image_output( asv, dst, out_format, QUANT_ERR_BITS, quality)) == NULL )
	{
		for( i = 0 ; i < count ; i++ )
			if( imdecs[i] )
				stop_image_decoding( &(imdecs[i]) );

        destroy_asimage( &dst );
		free_scanline( &dst_line, True );
    }else
	{
		int y, max_y = 0;
		int min_y = dst_height;
		int bg_tint = (layers[0].tint==0)?0x7F7F7F7F:layers[0].tint ;
		int bg_bottom = layers[0].dst_y+layers[0].clip_height+imdecs[0]->bevel_v_addon ;
LOCAL_DEBUG_OUT("blending actually...%s", "");
		pcurr = layers ;
		for( i = 0 ; i < count ; i++ )
		{
			if( imdecs[i] )
			{
				int layer_bottom = pcurr->dst_y+pcurr->clip_height ;
				if( pcurr->dst_y < min_y )
					min_y = pcurr->dst_y;
				layer_bottom += imdecs[i]->bevel_v_addon ;
				if( (int)layer_bottom > max_y )
					max_y = layer_bottom;
			}
			pcurr = (pcurr->next!=NULL)?pcurr->next:pcurr+1 ;
		}
		if( min_y < 0 )
			min_y = 0 ;
		else if( min_y >= (int)dst_height )
			min_y = dst_height ;
			
		if( max_y >= (int)dst_height )
			max_y = dst_height ;
		else
			imout->tiling_step = max_y ;

LOCAL_DEBUG_OUT( "min_y = %d, max_y = %d", min_y, max_y );
		dst_line.back_color = imdecs[0]->back_color ;
		dst_line.flags = 0 ;
		for( y = 0 ; y < min_y ; ++y  )
			imout->output_image_scanline( imout, &dst_line, 1);
		dst_line.flags = SCL_DO_ALL ;
		pcurr = layers ;
		for( i = 0 ; i < count ; ++i )
		{
			if( imdecs[i] && pcurr->dst_y < min_y  )
				imdecs[i]->next_line = min_y - pcurr->dst_y ;
			pcurr = (pcurr->next!=NULL)?pcurr->next:pcurr+1 ;
		}
		for( ; y < max_y ; ++y  )
		{
			if( layers[0].dst_y <= y && bg_bottom > y )
				imdecs[0]->decode_image_scanline( imdecs[0] );
			else
			{
				imdecs[0]->buffer.back_color = imdecs[0]->back_color ;
				imdecs[0]->buffer.flags = 0 ;
			}
			copytintpad_scanline( &(imdecs[0]->buffer), &dst_line, layers[0].dst_x, bg_tint );
			pcurr = layers[0].next?layers[0].next:&(layers[1]) ;
			for( i = 1 ; i < count ; i++ )
			{
				if( imdecs[i] && pcurr->dst_y <= y &&
					pcurr->dst_y+(int)pcurr->clip_height+(int)imdecs[i]->bevel_v_addon > y )
				{
					register ASScanline *b = &(imdecs[i]->buffer);
					CARD32 tint = pcurr->tint ;
					imdecs[i]->decode_image_scanline( imdecs[i] );
					if( tint != 0 )
					{
						tint_component_mod( b->red,   (CARD16)(ARGB32_RED8(tint)<<1),   b->width );
						tint_component_mod( b->green, (CARD16)(ARGB32_GREEN8(tint)<<1), b->width );
  					   	tint_component_mod( b->blue,  (CARD16)(ARGB32_BLUE8(tint)<<1),  b->width );
					  	tint_component_mod( b->alpha, (CARD16)(ARGB32_ALPHA8(tint)<<1), b->width );
					}
					pcurr->merge_scanlines( &dst_line, b, pcurr->dst_x );
				}
				pcurr = (pcurr->next!=NULL)?pcurr->next:pcurr+1 ;
			}
			imout->output_image_scanline( imout, &dst_line, 1);
		}
		dst_line.back_color = imdecs[0]->back_color ;
		dst_line.flags = 0 ;
		for( ; y < (int)dst_height ; y++  )
			imout->output_image_scanline( imout, &dst_line, 1);
		stop_image_output( &imout );
	}
	for( i = 0 ; i < count ; i++ )
		if( imdecs[i] != NULL )
		{
			stop_image_decoding( &(imdecs[i]) );
		}
	free( imdecs );
	free_scanline( &dst_line, True );
	SHOW_TIME("", started);
	return dst;
}

/* **************************************************************************************/
/* GRADIENT drawing : 																   */
/* **************************************************************************************/
static void
make_gradient_left2right( ASImageOutput *imout, ASScanline *dither_lines, int dither_lines_num, ASFlagType filter )
{
	int line ;

	imout->tiling_step = dither_lines_num;
	for( line = 0 ; line < dither_lines_num ; line++ )
		imout->output_image_scanline( imout, &(dither_lines[line]), 1);
}

static void
make_gradient_top2bottom( ASImageOutput *imout, ASScanline *dither_lines, int dither_lines_num, ASFlagType filter )
{
	int y, height = imout->im->height, width = imout->im->width ;
	int line ;
	ASScanline result;
	CARD32 chan_data[MAX_GRADIENT_DITHER_LINES] = {0,0,0,0};
LOCAL_DEBUG_CALLER_OUT( "width = %d, height = %d, filetr = 0x%lX, dither_count = %d\n", width, height, filter, dither_lines_num );
	prepare_scanline( width, QUANT_ERR_BITS, &result, imout->asv->BGR_mode );
	for( y = 0 ; y < height ; y++ )
	{
		int color ;

		result.flags = 0 ;
		result.back_color = ARGB32_DEFAULT_BACK_COLOR ;
		LOCAL_DEBUG_OUT( "line: %d", y );
		for( color = 0 ; color < IC_NUM_CHANNELS ; color++ )
			if( get_flags( filter, 0x01<<color ) )
			{
				Bool dithered = False ;
				for( line = 0 ; line < dither_lines_num ; line++ )
				{
					/* we want to do error diffusion here since in other places it only works
						* in horisontal direction : */
					CARD32 c = dither_lines[line].channels[color][y] ; 
					if( y+1 < height )
					{
						c += ((dither_lines[line].channels[color][y+1]&0xFF)>>1);
						if( (c&0xFFFF0000) != 0 )
							c = ( c&0x7F000000 )?0:0x0000FF00;
					}
					chan_data[line] = c ;

					if( chan_data[line] != chan_data[0] )
						dithered = True;
				}
				LOCAL_DEBUG_OUT( "channel: %d. Dithered ? %d", color, dithered );

				if( !dithered )
				{
					result.back_color = (result.back_color&(~MAKE_ARGB32_CHAN8(0xFF,color)))|
										MAKE_ARGB32_CHAN16(chan_data[0],color);
					LOCAL_DEBUG_OUT( "back_color = %8.8lX", result.back_color);
				}else
				{
					register CARD32  *dst = result.channels[color] ;
					for( line = 0 ; line  < dither_lines_num ; line++ )
					{
						register int x ;
						register CARD32 d = chan_data[line] ;
						for( x = line ; x < width ; x+=dither_lines_num )
						{
							dst[x] = d ;
						}
					}
					set_flags(result.flags, 0x01<<color);
				}
			}
		imout->output_image_scanline( imout, &result, 1);
	}
	free_scanline( &result, True );
}

static void
make_gradient_diag_width( ASImageOutput *imout, ASScanline *dither_lines, int dither_lines_num, ASFlagType filter, Bool from_bottom )
{
	int line = 0;
	/* using bresengham algorithm again to trigger horizontal shift : */
	short smaller = imout->im->height;
	short bigger  = imout->im->width;
	register int i = 0;
	int eps;
LOCAL_DEBUG_CALLER_OUT( "width = %d, height = %d, filetr = 0x%lX, dither_count = %d, dither width = %d\n", bigger, smaller, filter, dither_lines_num, dither_lines[0].width );

	if( from_bottom )
		toggle_image_output_direction( imout );
	eps = -(bigger>>1);
	for ( i = 0 ; i < bigger ; i++ )
	{
		eps += smaller;
		if( (eps << 1) >= bigger )
		{
			/* put scanline with the same x offset */
			dither_lines[line].offset_x = i ;
			imout->output_image_scanline( imout, &(dither_lines[line]), 1);
			if( ++line >= dither_lines_num )
				line = 0;
			eps -= bigger ;
		}
	}
}

static void
make_gradient_diag_height( ASImageOutput *imout, ASScanline *dither_lines, int dither_lines_num, ASFlagType filter, Bool from_bottom )
{
	int line = 0;
	unsigned short width = imout->im->width, height = imout->im->height ;
	/* using bresengham algorithm again to trigger horizontal shift : */
	unsigned short smaller = width;
	unsigned short bigger  = height;
	register int i = 0, k =0;
	int eps;
	ASScanline result;
	int *offsets ;

	prepare_scanline( width, QUANT_ERR_BITS, &result, imout->asv->BGR_mode );
	offsets = safemalloc( sizeof(int)*width );
	offsets[0] = 0 ;

	eps = -(bigger>>1);
	for ( i = 0 ; i < bigger ; i++ )
	{
		++offsets[k];
		eps += smaller;
		if( (eps << 1) >= bigger )
		{
			if( ++k >= width )
				break;
			offsets[k] = offsets[k-1] ; /* seeding the offset */
			eps -= bigger ;
		}
	}

	if( from_bottom )
		toggle_image_output_direction( imout );

	result.flags = (filter&SCL_DO_ALL);
	if( (filter&SCL_DO_ALL) == SCL_DO_ALL )
	{
		for( i = 0 ; i < height ; i++ )
		{
			for( k = 0 ; k < width ; k++ )
			{
				int offset = i+offsets[k] ;
				CARD32 **src_chan = &(dither_lines[line].channels[0]) ;
				result.alpha[k] = src_chan[IC_ALPHA][offset] ;
				result.red  [k] = src_chan[IC_RED]  [offset] ;
				result.green[k] = src_chan[IC_GREEN][offset] ;
				result.blue [k] = src_chan[IC_BLUE] [offset] ;
				if( ++line >= dither_lines_num )
					line = 0 ;
			}
			imout->output_image_scanline( imout, &result, 1);
		}
	}else
	{
		for( i = 0 ; i < height ; i++ )
		{
			for( k = 0 ; k < width ; k++ )
			{
				int offset = i+offsets[k] ;
				CARD32 **src_chan = &(dither_lines[line].channels[0]) ;
				if( get_flags(filter, SCL_DO_ALPHA) )
					result.alpha[k] = src_chan[IC_ALPHA][offset] ;
				if( get_flags(filter, SCL_DO_RED) )
					result.red[k]   = src_chan[IC_RED]  [offset] ;
				if( get_flags(filter, SCL_DO_GREEN) )
					result.green[k] = src_chan[IC_GREEN][offset] ;
				if( get_flags(filter, SCL_DO_BLUE) )
					result.blue[k]  = src_chan[IC_BLUE] [offset] ;
				if( ++line >= dither_lines_num )
					line = 0 ;
			}
			imout->output_image_scanline( imout, &result, 1);
		}
	}

	free( offsets );
	free_scanline( &result, True );
}

static ARGB32
get_best_grad_back_color( ASGradient *grad )
{
	ARGB32 back_color = 0 ;
	int chan ;
	for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
	{
		CARD8 best = 0;
		unsigned int best_size = 0;
		register int i = grad->npoints;
		while( --i > 0 )
		{ /* very crude algorithm, detecting biggest spans of the same color :*/
			CARD8 c = ARGB32_CHAN8(grad->color[i], chan );
			unsigned int span = grad->color[i]*20000;
			if( c == ARGB32_CHAN8(grad->color[i-1], chan ) )
			{
				span -= grad->color[i-1]*2000;
				if( c == best )
					best_size += span ;
				else if( span > best_size )
				{
					best_size = span ;
					best = c ;
				}
			}
		}
		back_color |= MAKE_ARGB32_CHAN8(best,chan);
	}
	return back_color;
}

ASImage*
make_gradient( ASVisual *asv, ASGradient *grad,
               int width, int height, ASFlagType filter,
  			   ASAltImFormats out_format, unsigned int compression_out, int quality  )
{
	ASImage *im = NULL ;
	ASImageOutput *imout;
	int line_len = width;
	START_TIME(started);
LOCAL_DEBUG_CALLER_OUT( "type = 0x%X, width=%d, height = %d, filter = 0x%lX", grad->type, width, height, filter );
	if( grad == NULL )
		return NULL;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if( width == 0 )
		width = 2;
 	if( height == 0 )
		height = 2;

	im = create_destination_image( width, height, out_format, compression_out, get_best_grad_back_color( grad ) );

	if( get_flags(grad->type,GRADIENT_TYPE_ORIENTATION) )
		line_len = height ;
	if( get_flags(grad->type,GRADIENT_TYPE_DIAG) )
		line_len = MAX(width,height)<<1 ;
	if((imout = start_image_output( asv, im, out_format, QUANT_ERR_BITS, quality)) == NULL )
	{
        destroy_asimage( &im );
    }else
	{
		int dither_lines = MIN(imout->quality+1, MAX_GRADIENT_DITHER_LINES) ;
		ASScanline *lines;
		int line;
		static ARGB32 dither_seeds[MAX_GRADIENT_DITHER_LINES] = { 0, 0xFFFFFFFF, 0x7F0F7F0F, 0x0F7F0F7F };

		if( dither_lines > (int)im->height || dither_lines > (int)im->width )
			dither_lines = MIN(im->height, im->width) ;

		lines = safecalloc( dither_lines, sizeof(ASScanline));
		for( line = 0 ; line < dither_lines ; line++ )
		{
			prepare_scanline( line_len, QUANT_ERR_BITS, &(lines[line]), asv->BGR_mode );
			make_gradient_scanline( &(lines[line]), grad, filter, dither_seeds[line] );
		}
		switch( get_flags(grad->type,GRADIENT_TYPE_MASK) )
		{
			case GRADIENT_Left2Right :
				make_gradient_left2right( imout, lines, dither_lines, filter );
  	    		break ;
			case GRADIENT_Top2Bottom :
				make_gradient_top2bottom( imout, lines, dither_lines, filter );
				break ;
			case GRADIENT_TopLeft2BottomRight :
			case GRADIENT_BottomLeft2TopRight :
				if( width >= height )
					make_gradient_diag_width( imout, lines, dither_lines, filter,
											 (grad->type==GRADIENT_BottomLeft2TopRight));
				else
					make_gradient_diag_height( imout, lines, dither_lines, filter,
											  (grad->type==GRADIENT_BottomLeft2TopRight));
				break ;
			default:
				break;
		}
		stop_image_output( &imout );
		for( line = 0 ; line < dither_lines ; line++ )
			free_scanline( &(lines[line]), True );
		free( lines );
	}
	SHOW_TIME("", started);
	return im;
}

/* ***************************************************************************/
/* Image flipping(rotation)													*/
/* ***************************************************************************/
ASImage *
flip_asimage( ASVisual *asv, ASImage *src,
		      int offset_x, int offset_y,
			  int to_width,
			  int to_height,
			  int flip,
			  ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageOutput  *imout ;
	ASFlagType filter = SCL_DO_ALL;
	START_TIME(started);

LOCAL_DEBUG_CALLER_OUT( "offset_x = %d, offset_y = %d, to_width = %d, to_height = %d", offset_x, offset_y, to_width, to_height );
	if( src == NULL )
		return NULL ;
	
	filter = get_asimage_chanmask(src);
	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
        destroy_asimage( &dst );
    }else
	{
		ASImageDecoder *imdec ;
		ASScanline result ;
		int y;
LOCAL_DEBUG_OUT("flip-flopping actually...%s", "");
		prepare_scanline( to_width, 0, &result, asv->BGR_mode );
		if( (imdec = start_image_decoding(asv, src, filter, offset_x, offset_y,
		                                  get_flags( flip, FLIP_VERTICAL )?to_height:to_width,
										  get_flags( flip, FLIP_VERTICAL )?to_width:to_height, NULL)) != NULL )
		{
            if( get_flags( flip, FLIP_VERTICAL ) )
			{
				CARD32 *chan_data ;
				size_t  pos = 0;
				int x ;
				CARD32 *a = imdec->buffer.alpha ;
				CARD32 *r = imdec->buffer.red ;
				CARD32 *g = imdec->buffer.green ;
				CARD32 *b = imdec->buffer.blue;

				chan_data = safemalloc( to_width*to_height*sizeof(CARD32));
                result.back_color = src->back_color;
				result.flags = filter ;
/*				memset( a, 0x00, to_height*sizeof(CARD32));
				memset( r, 0x00, to_height*sizeof(CARD32));
				memset( g, 0x00, to_height*sizeof(CARD32));
				memset( b, 0x00, to_height*sizeof(CARD32));
  */			for( y = 0 ; y < (int)to_width ; y++ )
				{
					imdec->decode_image_scanline( imdec );
					for( x = 0; x < (int)to_height ; x++ )
					{
						chan_data[pos++] = MAKE_ARGB32( a[x],r[x],g[x],b[x] );
					}
				}

				if( get_flags( flip, FLIP_UPSIDEDOWN ) )
				{
					for( y = 0 ; y < (int)to_height ; ++y )
					{
						pos = y + (int)(to_width-1)*(to_height) ;
						for( x = 0 ; x < (int)to_width ; ++x )
						{
							result.alpha[x] = ARGB32_ALPHA8(chan_data[pos]);
							result.red  [x] = ARGB32_RED8(chan_data[pos]);
							result.green[x] = ARGB32_GREEN8(chan_data[pos]);
							result.blue [x] = ARGB32_BLUE8(chan_data[pos]);
							pos -= to_height ;
						}
						imout->output_image_scanline( imout, &result, 1);
					}
				}else
				{
					for( y = to_height-1 ; y >= 0 ; --y )
					{
						pos = y ;
						for( x = 0 ; x < (int)to_width ; ++x )
						{
							result.alpha[x] = ARGB32_ALPHA8(chan_data[pos]);
							result.red  [x] = ARGB32_RED8(chan_data[pos]);
							result.green[x] = ARGB32_GREEN8(chan_data[pos]);
							result.blue [x] = ARGB32_BLUE8(chan_data[pos]);
							pos += to_height ;
						}
						imout->output_image_scanline( imout, &result, 1);
					}
				}
				free( chan_data );
			}else
			{
				toggle_image_output_direction( imout );
/*                fprintf( stderr, __FUNCTION__":chanmask = 0x%lX", filter ); */
				for( y = 0 ; y < (int)to_height ; y++  )
				{
					imdec->decode_image_scanline( imdec );
                    result.flags = imdec->buffer.flags = imdec->buffer.flags & filter ;
                    result.back_color = imdec->buffer.back_color ;
                    SCANLINE_FUNC_FILTERED(reverse_component,imdec->buffer,result,0,to_width);
					imout->output_image_scanline( imout, &result, 1);
				}
			}
			stop_image_decoding( &imdec );
		}
		free_scanline( &result, True );
		stop_image_output( &imout );
	}
	SHOW_TIME("", started);
	return dst;
}

ASImage *
mirror_asimage( ASVisual *asv, ASImage *src,
		      int offset_x, int offset_y,
			  int to_width,
			  int to_height,
			  Bool vertical, ASAltImFormats out_format,
			  unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageOutput  *imout ;
	START_TIME(started);

	LOCAL_DEBUG_CALLER_OUT( "offset_x = %d, offset_y = %d, to_width = %d, to_height = %d", offset_x, offset_y, to_width, to_height );
	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
        destroy_asimage( &dst );
    }else
	{
		ASImageDecoder *imdec ;
		ASScanline result ;
		int y;
		if( !vertical )
			prepare_scanline( to_width, 0, &result, asv->BGR_mode );
LOCAL_DEBUG_OUT("miroring actually...%s", "");
		if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, offset_x, offset_y,
		                                  to_width, to_height, NULL)) != NULL )
		{
			if( vertical )
			{
				toggle_image_output_direction( imout );
				for( y = 0 ; y < (int)to_height ; y++  )
				{
					imdec->decode_image_scanline( imdec );
					imout->output_image_scanline( imout, &(imdec->buffer), 1);
				}
			}else
			{
				for( y = 0 ; y < (int)to_height ; y++  )
				{
					imdec->decode_image_scanline( imdec );
					result.flags = imdec->buffer.flags ;
					result.back_color = imdec->buffer.back_color ;
					SCANLINE_FUNC(reverse_component,imdec->buffer,result,0,to_width);
					imout->output_image_scanline( imout, &result, 1);
				}
			}
			stop_image_decoding( &imdec );
		}
		if( !vertical )
			free_scanline( &result, True );
		stop_image_output( &imout );
	}
	SHOW_TIME("", started);
	return dst;
}

ASImage *
pad_asimage(  ASVisual *asv, ASImage *src,
		      int dst_x, int dst_y,
			  int to_width,
			  int to_height,
			  ARGB32 color,
			  ASAltImFormats out_format,
			  unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageOutput  *imout ;
	int clip_width, clip_height ;
	START_TIME(started);

LOCAL_DEBUG_CALLER_OUT( "dst_x = %d, dst_y = %d, to_width = %d, to_height = %d", dst_x, dst_y, to_width, to_height );
	if( src == NULL )
		return NULL ;

	if( to_width == src->width && to_height == src->height && dst_x == 0 && dst_y == 0 )
		return clone_asimage( src, SCL_DO_ALL );

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color);

	clip_width = src->width ;
	clip_height = src->height ;
	if( dst_x < 0 )
		clip_width = MIN( (int)to_width, dst_x+clip_width );
	else
		clip_width = MIN( (int)to_width-dst_x, clip_width );
    if( dst_y < 0 )
		clip_height = MIN( (int)to_height, dst_y+clip_height);
	else
		clip_height = MIN( (int)to_height-dst_y, clip_height);
	if( (clip_width <= 0 || clip_height <= 0) )
	{                              /* we are completely outside !!! */
		dst->back_color = color ;
		return dst ;
	}

	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
        destroy_asimage( &dst );
    }else
	{
		ASImageDecoder *imdec = NULL;
		ASScanline result ;
		int y;
		int start_x = (dst_x < 0)? 0: dst_x;
		int start_y = (dst_y < 0)? 0: dst_y;

		if( (int)to_width != clip_width || clip_width != (int)src->width )
		{
			prepare_scanline( to_width, 0, &result, asv->BGR_mode );
			imdec = start_image_decoding(  asv, src, SCL_DO_ALL,
			                               (dst_x<0)? -dst_x:0,
										   (dst_y<0)? -dst_y:0,
		                                    clip_width, clip_height, NULL);
		}

		result.back_color = color ;
		result.flags = 0 ;
LOCAL_DEBUG_OUT( "filling %d lines with %8.8lX", start_y, color );
		for( y = 0 ; y < start_y ; y++  )
			imout->output_image_scanline( imout, &result, 1);

		if( imdec )
			result.back_color = imdec->buffer.back_color ;
		if( (int)to_width == clip_width )
		{
			if( imdec == NULL )
			{
LOCAL_DEBUG_OUT( "copiing %d lines", clip_height );
				copy_asimage_lines( dst, start_y, src, (dst_y < 0 )? -dst_y: 0, clip_height, SCL_DO_ALL );
				imout->next_line += clip_height ;
			}else
				for( y = 0 ; y < clip_height ; y++  )
				{
					imdec->decode_image_scanline( imdec );
					imout->output_image_scanline( imout, &(imdec->buffer), 1);
				}
		}else if( imdec )
		{
			for( y = 0 ; y < clip_height ; y++  )
			{
				int chan ;

				imdec->decode_image_scanline( imdec );
				result.flags = imdec->buffer.flags ;
				for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
				{
	   				register CARD32 *chan_data = result.channels[chan] ;
	   				register CARD32 *src_chan_data = imdec->buffer.channels[chan]+((dst_x<0)? -dst_x : 0) ;
					CARD32 chan_val = ARGB32_CHAN8(color, chan);
					register int k = -1;
					for( k = 0 ; k < start_x ; ++k )
						chan_data[k] = chan_val ;
					chan_data += k ;
					for( k = 0 ; k < clip_width ; ++k )
						chan_data[k] = src_chan_data[k];
					chan_data += k ;
					k = to_width-(start_x+clip_width) ;
					while( --k >= 0 )
						chan_data[k] = chan_val ;
				}
				imout->output_image_scanline( imout, &result, 1);
			}
		}
		result.back_color = color ;
		result.flags = 0 ;
LOCAL_DEBUG_OUT( "filling %d lines with %8.8lX at the end", to_height-(start_y+clip_height), color );
		for( y = start_y+clip_height ; y < (int)to_height ; y++  )
			imout->output_image_scanline( imout, &result, 1);

		if( imdec )
		{
			stop_image_decoding( &imdec );
			free_scanline( &result, True );
		}
		stop_image_output( &imout );
	}
	SHOW_TIME("", started);
	return dst;
}


/**********************************************************************/

Bool fill_asimage( ASVisual *asv, ASImage *im,
               	   int x, int y, int width, int height,
				   ARGB32 color )
{
	ASImageOutput *imout;
	ASImageDecoder *imdec;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if( im == NULL )
		return False;
	if( x < 0 )
	{	width += x ; x = 0 ; }
	if( y < 0 )
	{	height += y ; y = 0 ; }

	if( width <= 0 || height <= 0 || x >= (int)im->width || y >= (int)im->height )
		return False;
	if( x+width > (int)im->width )
		width = (int)im->width-x ;
	if( y+height > (int)im->height )
		height = (int)im->height-y ;

	if((imout = start_image_output( asv, im, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT)) == NULL )
		return False ;
	else
	{
		int i ;
		imout->next_line = y ;
		if( x == 0 && width == (int)im->width )
		{
			ASScanline result ;
			result.flags = 0 ;
			result.back_color = color ;
			for( i = 0 ; i < height ; i++ )
				imout->output_image_scanline( imout, &result, 1);
		}else if ((imdec = start_image_decoding(asv, im, SCL_DO_ALL, 0, y, im->width, height, NULL)) != NULL )
		{
			CARD32 alpha = ARGB32_ALPHA8(color), red = ARGB32_RED8(color),
				   green = ARGB32_GREEN8(color), blue = ARGB32_BLUE8(color);
			CARD32 	*a = imdec->buffer.alpha + x ; 
			CARD32 	*r = imdec->buffer.red + x ;
			CARD32 	*g = imdec->buffer.green + x ;
			CARD32 	*b = imdec->buffer.blue + x  ;
			for( i = 0 ; i < height ; i++ )
			{
				register int k ;
				imdec->decode_image_scanline( imdec );
				for( k = 0 ; k < width ; ++k )
				{
					a[k] = alpha ;
					r[k] = red ;
					g[k] = green ;
					b[k] = blue ;
				}
				imout->output_image_scanline( imout, &(imdec->buffer), 1);
			}
			stop_image_decoding( &imdec );
		}
	}
	stop_image_output( &imout );
	SHOW_TIME("", started);
	return True;
}

/* ********************************************************************************/
/* Vector -> ASImage functions :                                                  */
/* ********************************************************************************/
Bool
colorize_asimage_vector( ASVisual *asv, ASImage *im,
						 ASVectorPalette *palette,
						 ASAltImFormats out_format,
						 int quality )
{
	ASImageOutput  *imout = NULL ;
	ASScanline buf ;
	int x, y, curr_point, last_point ;
    register double *vector ;
	double *points ;
	double *multipliers[IC_NUM_CHANNELS] ;
	START_TIME(started);

	if( im == NULL || palette == NULL || out_format == ASA_Vector )
		return False;

	if( im->alt.vector == NULL )
		return False;
	vector = im->alt.vector ;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if((imout = start_image_output( asv, im, out_format, QUANT_ERR_BITS, quality)) == NULL )
		return False;
	/* as per ROOT ppl request double data goes from bottom to top,
	 * instead of from top to bottom : */
	if( !get_flags( im->flags, ASIM_VECTOR_TOP2BOTTOM) )
		toggle_image_output_direction(imout);

	prepare_scanline( im->width, QUANT_ERR_BITS, &buf, asv->BGR_mode );
	curr_point = palette->npoints/2 ;
	points = palette->points ;
	last_point = palette->npoints-1 ;
	buf.flags = 0 ;
	for( y = 0 ; y < IC_NUM_CHANNELS ; ++y )
	{
		if( palette->channels[y] )
		{
			multipliers[y] = safemalloc( last_point*sizeof(double));
			for( x = 0 ; x < last_point ; ++x )
			{
				if (points[x+1] == points[x])
      				multipliers[y][x] = 1;
				else
					multipliers[y][x] = (double)(palette->channels[y][x+1] - palette->channels[y][x])/
				                 	        (points[x+1]-points[x]);
/*				fprintf( stderr, "%e-%e/%e-%e=%e ", (double)palette->channels[y][x+1], (double)palette->channels[y][x],
				                 	        points[x+1], points[x], multipliers[y][x] );
 */
			}
/*			fputc( '\n', stderr ); */
			set_flags(buf.flags, (0x01<<y));
		}else
			multipliers[y] = NULL ;
	}
	for( y = 0 ; y < (int)im->height ; ++y )
	{
		for( x = 0 ; x < (int)im->width ;)
		{
			register int i = IC_NUM_CHANNELS ;
			double d ;

			if( points[curr_point] > vector[x] )
			{
				while( --curr_point >= 0 )
					if( points[curr_point] < vector[x] )
						break;
				if( curr_point < 0 )
					++curr_point ;
			}else
			{
				while( points[curr_point+1] < vector[x] )
					if( ++curr_point >= last_point )
					{
						curr_point = last_point-1 ;
						break;
					}
			}
			d = vector[x]-points[curr_point];
/*			fprintf( stderr, "%f|%d|%f*%f=%d(%f)+%d=", vector[x], curr_point, d, multipliers[0][curr_point], (int)(d*multipliers[0][curr_point]),(d*multipliers[0][curr_point]) , palette->channels[0][curr_point] ); */
			while( --i >= 0 )
				if( multipliers[i] )
				{/* the following calculation is the most expensive part of the algorithm : */
					buf.channels[i][x] = (int)(d*multipliers[i][curr_point])+palette->channels[i][curr_point] ;
/*					fprintf( stderr, "%2.2X.", buf.channels[i][x] ); */
				}
/*			fputc( ' ', stderr ); */
#if 1
			while( ++x < (int)im->width )
				if( vector[x] == vector[x-1] )
				{
					buf.red[x] = buf.red[x-1] ;
					buf.green[x] = buf.green[x-1] ;
					buf.blue[x] = buf.blue[x-1] ;
					buf.alpha[x] = buf.alpha[x-1] ;
				}else
					break;
#else
			++x ;
#endif
		}
/*		fputc( '\n', stderr ); */
		imout->output_image_scanline( imout, &buf, 1);
		vector += im->width ;
	}
	for( y = 0 ; y < IC_NUM_CHANNELS ; ++y )
		if( multipliers[y] )
			free(multipliers[y]);

	stop_image_output( &imout );
	free_scanline( &buf, True );
	SHOW_TIME("", started);
	return True;
}

ASImage *
create_asimage_from_vector( ASVisual *asv, double *vector,
							int width, int height,
							ASVectorPalette *palette,
							ASAltImFormats out_format,
							unsigned int compression, int quality )
{
	ASImage *im = NULL;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if( vector != NULL )
	{
		im = create_destination_image( width, height, out_format, compression, ARGB32_DEFAULT_BACK_COLOR);

		if( im != NULL )
		{
			if( set_asimage_vector( im, vector ) )
				if( palette )
					colorize_asimage_vector( asv, im, palette, out_format, quality );
		}
	}
	return im ;
}


/***********************************************************************
 * Gaussian blur code.
 **********************************************************************/

#undef PI
#define PI 3.141592526

#if 0
static inline void
gauss_component(CARD32 *src, CARD32 *dst, int radius, double* gauss, int len)
{
	int x, j, r = radius - 1;
	for (x = 0 ; x < len ; x++) {
		register double v = 0.0;
		for (j = x - r ; j <= 0 ; j++) v += src[0] * gauss[x - j];
		for ( ; j < x ; j++) v += src[j] * gauss[x - j];
		v += src[x] * gauss[0];
		for (j = x + r ; j >= len ; j--) v += src[len - 1] * gauss[j - x];
		for ( ; j > x ; j--) v += src[j] * gauss[j - x];
		dst[x] = (CARD32)v;
	}
}
#endif

#define GAUSS_COEFF_TYPE int
/* static void calc_gauss_double(double radius, double* gauss); */
static void calc_gauss_int(int radius, GAUSS_COEFF_TYPE* gauss, GAUSS_COEFF_TYPE* gauss_sums);

#define gauss_data_t CARD32
#define gauss_var_t int

static inline void
gauss_component_int(gauss_data_t *s1, gauss_data_t *d1, int radius, GAUSS_COEFF_TYPE* gauss, GAUSS_COEFF_TYPE* gauss_sums, int len)
{
#define DEFINE_GAUS_TMP_VAR		CARD32 *xs1 = &s1[x]; CARD32 v1 = xs1[0]*gauss[0]
	if( len < radius + radius )
	{
		int x = 0, j;
		while( x < len )
		{
			int tail = len - 1 - x;
			int gauss_sum = gauss[0];
			DEFINE_GAUS_TMP_VAR;
			for (j = 1 ; j <= x ; ++j)
			{
				v1 += xs1[-j]*gauss[j];
				gauss_sum += gauss[j];
			}
			for (j = 1 ; j <= tail ; ++j)
			{
				v1 += xs1[j]*gauss[j];
				gauss_sum += gauss[j];
			}
			d1[x] = (v1<<10)/gauss_sum;
			++x;
		}
		return;
	}

#define MIDDLE_STRETCH_GAUSS(j_check)	\
	do{ for( j = 1 ; j j_check ; ++j ) v1 += (xs1[-j]*gauss[j]+xs1[j]*gauss[j]); }while(0)

	/* left stretch [0, r-2] */
	{
		int x = 0 ;
		for( ; x < radius-1 ; ++x )
		{
			int j ;
			gauss_data_t *xs1 = &s1[x]; 
			gauss_var_t v1 = xs1[0]*gauss[0];
			for( j = 1 ; j <= x ; ++j )
				v1 += (xs1[-j]*gauss[j]+xs1[j]*gauss[j]);

			for( ; j < radius ; ++j ) 
				v1 += xs1[j]*gauss[j];
			d1[x] = (v1<<10)/gauss_sums[x];
		}	
	}

	/* middle stretch : [r-1, l-r] */
	if (radius-1 == len - radius)
	{
		gauss_data_t *xs1 = &s1[radius-1]; 
		gauss_var_t v1 = xs1[0]*gauss[0];
		int j = 1;
		for( ; j < radius ; ++j ) 
			v1 += (xs1[-j]*gauss[j]+xs1[j]*gauss[j]);
		d1[radius] = v1 ;
	}else
	{
		int x = radius;
		for(; x <= len - radius + 1; x+=3)
		{
			gauss_data_t *xs1 = &s1[x]; 
			gauss_var_t v1 = xs1[-1]*gauss[0];
			gauss_var_t v2 = xs1[0]*gauss[0];
			gauss_var_t v3 = xs1[1]*gauss[0];
			int j = 1;
			for( ; j < radius ; ++j ) 
			{
				int g = gauss[j];
				v1 += xs1[-j-1]*g+xs1[j-1]*g;
				v2 += xs1[-j]*g+xs1[j]*g;
				v3 += xs1[-j+1]*g+xs1[j+1]*g;
			}
			d1[x-1] = v1 ; 
			d1[x] = v2 ;
			d1[x+1] = v3 ;
		}
	}
	{
		int x = 0;
		gauss_data_t *td1 = &d1[len-1];
		for( ; x < radius-1; ++x )
		{
			int j;
			gauss_data_t *xs1 = &s1[len-1-x]; 
			gauss_var_t v1 = xs1[0]*gauss[0];
			for( j = 1 ; j <= x ; ++j ) 
				v1 += (xs1[-j]*gauss[j]+xs1[j]*gauss[j]);			
				
			for( ; j <radius ; ++j )
				v1 += xs1[-j]*gauss[j];
			td1[-x] = (v1<<10)/gauss_sums[x];
		}
	}
#undef 	MIDDLE_STRETCH_GAUSS
#undef DEFINE_GAUS_TMP_VAR
}

/*#define 	USE_PARALLEL_OPTIMIZATION */

#ifdef USE_PARALLEL_OPTIMIZATION
/* this ain't worth a crap it seems. The code below seems to perform 20% slower then 
   plain and simple one component at a time 
 */
static inline void
gauss_component_int2(CARD32 *s1, CARD32 *d1, CARD32 *s2, CARD32 *d2, int radius, GAUSS_COEFF_TYPE* gauss, GAUSS_COEFF_TYPE* gauss_sums, int len)
{
#define MIDDLE_STRETCH_GAUSS do{GAUSS_COEFF_TYPE g = gauss[j]; \
								v1 += (xs1[-j]+xs1[j])*g; \
								v2 += (xs2[-j]+xs2[j])*g; }while(0)

	int x, j;
	int tail = radius;
	GAUSS_COEFF_TYPE g0 = gauss[0];
	for( x = 0 ; x < radius ; ++x )
	{
		register CARD32 *xs1 = &s1[x];
		register CARD32 *xs2 = &s2[x];
		register CARD32 v1 = s1[x]*g0;
		register CARD32 v2 = s2[x]*g0;
		for( j = 1 ; j <= x ; ++j )
			MIDDLE_STRETCH_GAUSS;
		for( ; j < radius ; ++j ) 
		{
			GAUSS_COEFF_TYPE g = gauss[j];
			CARD32 m1 = xs1[j]*g;
			CARD32 m2 = xs2[j]*g;
			v1 += m1;
			v2 += m2;
		}
		v1 = v1<<10;
		v2 = v2<<10;
		{
			GAUSS_COEFF_TYPE gs = gauss_sums[x];
			d1[x] = v1/gs;
			d2[x] = v2/gs;
		}
	}	
	while( x <= len-radius )
	{
		register CARD32 *xs1 = &s1[x];
		register CARD32 *xs2 = &s2[x];
		register CARD32 v1 = s1[x]*g0;
		register CARD32 v2 = s2[x]*g0;
		for( j = 1 ; j < radius ; ++j ) 
			MIDDLE_STRETCH_GAUSS;
		d1[x] = v1 ;
		d2[x] = v2 ;
		++x;
	}
	while( --tail > 0 )/*x < len*/
	{
		register CARD32 *xs1 = &s1[x];
		register CARD32 *xs2 = &s2[x];
		register CARD32 v1 = xs1[0]*g0;
		register CARD32 v2 = xs2[0]*g0;
		for( j = 1 ; j < tail ; ++j ) 
			MIDDLE_STRETCH_GAUSS;
		for( ; j <radius ; ++j )
		{
			GAUSS_COEFF_TYPE g = gauss[j];
			CARD32 m1 = xs1[-j]*g;
			CARD32 m2 = xs2[-j]*g;
			v1 += m1;
			v2 += m2;
		}
		v1 = v1<<10;
		v2 = v2<<10;
		{
			GAUSS_COEFF_TYPE gs = gauss_sums[tail];
			d1[x] = v1/gs;
			d2[x] = v2/gs;
		}
		++x;
	}
#undef 	MIDDLE_STRETCH_GAUSS
}
#endif

static inline void
load_gauss_scanline(ASScanline *result, ASImageDecoder *imdec, int horz, GAUSS_COEFF_TYPE *sgauss, GAUSS_COEFF_TYPE *sgauss_sums, ASFlagType filter )
{
    ASFlagType lf; 
	int x, chan;
#ifdef USE_PARALLEL_OPTIMIZATION
	int todo_count = 0;
	int todo[IC_NUM_CHANNELS] = {-1,-1,-1,-1};
#endif
	imdec->decode_image_scanline(imdec);
	lf = imdec->buffer.flags&filter ;
	result->flags = imdec->buffer.flags;
	result->back_color = imdec->buffer.back_color;

	for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
	{
		CARD32 *res_chan = result->channels[chan];
		CARD32 *src_chan = imdec->buffer.channels[chan];
		if( get_flags(lf, 0x01<<chan) )
		{
			if( horz == 1 ) 
			{
				for( x =  0 ; x < result->width ; ++x ) 
					res_chan[x] = src_chan[x]<<10 ;
			}else
			{
#ifdef USE_PARALLEL_OPTIMIZATION
				todo[todo_count++] = chan;
#else			
				gauss_component_int(src_chan, res_chan, horz, sgauss, sgauss_sums, result->width);
#endif
			}
	    }else if( get_flags( result->flags, 0x01<<chan ) )
	        copy_component( src_chan, res_chan, 0, result->width);
		else if( get_flags( filter, 0x01<<chan ) )
		{
			CARD32 fill = (CARD32)ARGB32_RED8(imdec->buffer.back_color)<<10;
			for( x =  0 ; x < result->width ; ++x ) res_chan[x] = fill ;
		}
	}

#ifdef USE_PARALLEL_OPTIMIZATION
	switch( 4 - todo_count )
	{
		case 0 : /* todo_count == 4 */
			gauss_component_int2(imdec->buffer.channels[todo[2]], result->channels[todo[2]],
								 imdec->buffer.channels[todo[3]], result->channels[todo[3]],
								 horz, sgauss, sgauss_sums, result->width);
		case 2 : /* todo_count == 2 */
			gauss_component_int2(imdec->buffer.channels[todo[0]], result->channels[todo[0]], 
								 imdec->buffer.channels[todo[1]], result->channels[todo[1]],
								 horz, sgauss, sgauss_sums, result->width); break ;
		case 1 : /* todo_count == 3 */
			gauss_component_int2(imdec->buffer.channels[todo[1]], result->channels[todo[1]],
								 imdec->buffer.channels[todo[2]], result->channels[todo[2]],
								 horz, sgauss, sgauss_sums, result->width);
		case 3 : /* todo_count == 1 */
			gauss_component_int( imdec->buffer.channels[todo[0]], 
								 result->channels[todo[0]], 
								 horz, sgauss, sgauss_sums, result->width); break ;
	}
#endif
}


ASImage* blur_asimage_gauss(ASVisual* asv, ASImage* src, double dhorz, double dvert,
                            ASFlagType filter,
							ASAltImFormats out_format, unsigned int compression_out, int quality)
{
	ASImage *dst = NULL;
	ASImageOutput *imout;
	ASImageDecoder *imdec;
	int y, x, chan;
	int horz = (int)dhorz;
	int vert = (int)dvert;
	int width, height ; 
#if 0
	struct timeval stv;
	gettimeofday (&stv,NULL);
#define PRINT_BACKGROUND_OP_TIME do{ struct timeval tv;gettimeofday (&tv,NULL); tv.tv_sec-= stv.tv_sec;\
                                     fprintf (stderr,__FILE__ "%d: elapsed  %ld usec\n",__LINE__,\
                                              tv.tv_sec*1000000+tv.tv_usec-stv.tv_usec );}while(0)
#else                                           
#define PRINT_BACKGROUND_OP_TIME do{}while(0)                                          
#endif

	if (!src) return NULL;

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	width = src->width ;
	height = src->height ;
	dst = create_destination_image( width, height, out_format, compression_out, src->back_color);

	imout = start_image_output(asv, dst, out_format, 0, quality);
    if (!imout)
    {
        destroy_asimage( &dst );
		return NULL;
	}

	imdec = start_image_decoding(asv, src, SCL_DO_ALL, 0, 0, src->width, src->height, NULL);
	if (!imdec) 
	{
		stop_image_output(&imout);
        destroy_asimage( &dst );
		return NULL;
	}
	
	if( horz > (width-1)/2  ) horz = (width==1 )?1:(width-1)/2 ;
	if( vert > (height-1)/2 ) vert = (height==1)?1:(height-1)/2 ;
	if (horz > 128) 
		horz = 128;
	else if (horz < 1) 
		horz = 1;
	if( vert > 128 )
		vert = 128 ;
	else if( vert < 1 ) 
		vert = 1 ;

	if( vert == 1 && horz == 1 ) 
	{
	    for (y = 0 ; y < dst->height ; y++)
		{
			imdec->decode_image_scanline(imdec);
	        imout->output_image_scanline(imout, &(imdec->buffer), 1);
		}
	}else
	{
		ASScanline result;
		GAUSS_COEFF_TYPE *horz_gauss = NULL;
		GAUSS_COEFF_TYPE *horz_gauss_sums = NULL;

		if( horz > 1 )
		{
			PRINT_BACKGROUND_OP_TIME;
			horz_gauss = safecalloc(horz+1, sizeof(GAUSS_COEFF_TYPE));
			horz_gauss_sums = safecalloc(horz+1, sizeof(GAUSS_COEFF_TYPE));
			calc_gauss_int(horz, horz_gauss, horz_gauss_sums);
			PRINT_BACKGROUND_OP_TIME;
		}
		prepare_scanline(width, 0, &result, asv->BGR_mode);
		if( vert == 1 ) 
		{
		    for (y = 0 ; y < height ; y++)
		    {
				load_gauss_scanline(&result, imdec, horz, horz_gauss, horz_gauss_sums, filter );
				for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
					if( get_flags( filter, 0x01<<chan ) )
					{
						CARD32 *res_chan = result.channels[chan];
						for( x = 0 ; x < width ; ++x ) 
							res_chan[x] = (res_chan[x]&0x03Fc0000)?255:res_chan[x]>>10;
					}
		        imout->output_image_scanline(imout, &result, 1);
			}
		}else
		{ /* new code : */
			GAUSS_COEFF_TYPE *vert_gauss = safecalloc(vert+1, sizeof(GAUSS_COEFF_TYPE));
			GAUSS_COEFF_TYPE *vert_gauss_sums = safecalloc(vert+1, sizeof(GAUSS_COEFF_TYPE));
			int lines_count = vert*2-1;
			int first_line = 0, last_line = lines_count-1;
			ASScanline *lines_mem = safecalloc( lines_count, sizeof(ASScanline));
			ASScanline **lines = safecalloc( dst->height+1, sizeof(ASScanline*));

			/* init */
			calc_gauss_int(vert, vert_gauss, vert_gauss_sums);
			PRINT_BACKGROUND_OP_TIME;

			for( y = 0 ; y < lines_count ; ++y ) 
			{
				lines[y] = &lines_mem[y] ;
				prepare_scanline(width, 0, lines[y], asv->BGR_mode);
				load_gauss_scanline(lines[y], imdec, horz, horz_gauss, horz_gauss_sums, filter );
			}

			PRINT_BACKGROUND_OP_TIME;
			result.flags = 0xFFFFFFFF;
			/* top band  [0, vert-2] */
    		for (y = 0 ; y < vert-1 ; y++)
    		{
				for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
				{
					CARD32 *res_chan = result.channels[chan];
					if( !get_flags(filter, 0x01<<chan) )
		        		copy_component( lines[y]->channels[chan], res_chan, 0, width);
					else
					{	
						register ASScanline **ysrc = &lines[y];
						int j = 0;
						GAUSS_COEFF_TYPE g = vert_gauss[0];
						CARD32 *src_chan1 = ysrc[0]->channels[chan];
						for( x = 0 ; x < width ; ++x ) 
							res_chan[x] = src_chan1[x]*g;
						while( ++j <= y )
						{
							CARD32 *src_chan2 = ysrc[j]->channels[chan];
							g = vert_gauss[j];
							src_chan1 = ysrc[-j]->channels[chan];
							for( x = 0 ; x < width ; ++x ) 
								res_chan[x] += (src_chan1[x]+src_chan2[x])*g;
						}	
						for( ; j < vert ; ++j ) 
						{
							g = vert_gauss[j];
							src_chan1 = ysrc[j]->channels[chan];
							for( x = 0 ; x < width ; ++x ) 
								res_chan[x] += src_chan1[x]*g;
						}
						g = vert_gauss_sums[y];
						for( x = 0 ; x < width ; ++x ) 
						{
							gauss_var_t v = res_chan[x]/g;
							res_chan[x] = (v&0x03Fc0000)?255:v>>10;
						}
					}
				}
        		imout->output_image_scanline(imout, &result, 1);
			}
			PRINT_BACKGROUND_OP_TIME;
			/* middle band [vert-1, height-vert] */
			for( ; y <= height - vert; ++y)
			{
				for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
				{
					CARD32 *res_chan = result.channels[chan];
					if( !get_flags(filter, 0x01<<chan) )
		        		copy_component( lines[y]->channels[chan], res_chan, 0, result.width);
					else
					{	
						register ASScanline **ysrc = &lines[y];
/* surprisingly, having x loops inside y loop yields 30% to 80% better performance */
						int j = 0;
						CARD32 *src_chan1 = ysrc[0]->channels[chan];
						memset( res_chan, 0x00, width*4 );
/*						for( x = 0 ; x < width ; ++x ) 
							res_chan[x] = src_chan1[x]*vert_gauss[0];
 */							
						while( ++j < vert ) 
						{
							CARD32 *src_chan2 = ysrc[j]->channels[chan];
							GAUSS_COEFF_TYPE g = vert_gauss[j];
							src_chan1 = ysrc[-j]->channels[chan];
							switch( g ) 
							{
								case 1 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += src_chan1[x]+src_chan2[x];
									break;
								case 2 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])<<1;
									break;
#if 1
								case 4 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])<<2;
									break;
								case 8 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])<<3;
									break;
								case 16 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])<<4;
									break;
								case 32 :
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])<<5;
									break;
#endif		
								default : 									
									for( x = 0 ; x < width ; ++x ) 
										res_chan[x] += (src_chan1[x]+src_chan2[x])*g;
							}
						}
 						src_chan1 = ysrc[0]->channels[chan];
						for( x = 0 ; x < width ; ++x ) 
						{
							gauss_var_t v = src_chan1[x]*vert_gauss[0] + res_chan[x];
							res_chan[x] = (v&0xF0000000)?255:v>>20;
						} 
					}
				}

        		imout->output_image_scanline(imout, &result, 1);
				++last_line;
				/* fprintf( stderr, "last_line = %d, first_line = %d, height = %d, vert = %d, y = %d\n", last_line, first_line, dst->height, vert, y ); */
				lines[last_line] = lines[first_line] ; 
				++first_line;
				load_gauss_scanline(lines[last_line], imdec, horz, horz_gauss, horz_gauss_sums, filter );
			}
			PRINT_BACKGROUND_OP_TIME;
			/* bottom band */
			for( ; y < height; ++y)
			{
				int tail = height - y ; 
				for( chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan )
				{
					CARD32 *res_chan = result.channels[chan];
					if( !get_flags(filter, 0x01<<chan) )
		        		copy_component( lines[y]->channels[chan], res_chan, 0, result.width);
					else
					{	
						register ASScanline **ysrc = &lines[y];
						int j = 0;
						GAUSS_COEFF_TYPE g ;
						CARD32 *src_chan1 = ysrc[0]->channels[chan];
						for( x = 0 ; x < width ; ++x ) 
							res_chan[x] = src_chan1[x]*vert_gauss[0];
						for( j = 1 ; j < tail ; ++j ) 
						{
							CARD32 *src_chan2 = ysrc[j]->channels[chan];
							g = vert_gauss[j];
							src_chan1 = ysrc[-j]->channels[chan];
							for( x = 0 ; x < width ; ++x ) 
								res_chan[x] += (src_chan1[x]+src_chan2[x])*g;
						}
						for( ; j < vert ; ++j )
						{
							g = vert_gauss[j];
							src_chan1 = ysrc[-j]->channels[chan];
							for( x = 0 ; x < width ; ++x ) 
								res_chan[x] += src_chan1[x]*g;
						}
						g = vert_gauss_sums[tail];
						for( x = 0 ; x < width ; ++x ) 
						{
							gauss_var_t v = res_chan[x]/g;
							res_chan[x] = (v&0x03Fc0000)?255:v>>10;
						}
					}
				}

        		imout->output_image_scanline(imout, &result, 1);
			}
			/* cleanup */
			for( y = 0 ; y < lines_count ; ++y ) 
				free_scanline(&lines_mem[y], True);
			free( lines_mem );
			free( lines );
			free(vert_gauss_sums);
			free(vert_gauss);

		}
		free_scanline(&result, True);
		if( horz_gauss_sums )
			free(horz_gauss_sums);
		if( horz_gauss )
			free(horz_gauss);
	}
PRINT_BACKGROUND_OP_TIME;

	stop_image_decoding(&imdec);
	stop_image_output(&imout);

	return dst;
}

#if 0 /* unused for the time being */
static void calc_gauss_double(double radius, double* gauss) 
{
	int i, mult;
	double std_dev, sum = 0.0;
	double g0, g_last;
	double n, nn, nPI, nnPI;
	if (radius <= 1.0) 
	{
		gauss[0] = 1.0;
		return;
	}
	/* after radius of 128 - gaussian degrades into something weird, 
	   since our colors are only 8 bit */
	if (radius > 128.0) radius = 128.0; 
	std_dev = (radius - 1) * 0.3003866304;
	do
	{
		sum = 0 ;
		n = std_dev * std_dev;
		nn = 2*n ;
		nPI = n*PI;
		nnPI = nn*PI;
		sum = g0 = 1.0 / nnPI ;
		for (i = 1 ; i < radius-1 ; i++) 
			sum += exp((double)-i * (double)i / nn)/nPI; 
		g_last = exp((double)-i * (double)i / nn)/nnPI; 
		sum += g_last*2.0 ; 
	
		mult = (int)((1024.0+radius*0.94)/sum);
		std_dev += 0.1 ;
	}while( g_last*mult  < 1. );

	gauss[0] = g0/sum ; 
	gauss[(int)radius-1] = g_last/sum;
	
	sum *= nnPI;
	for (i = 1 ; i < radius-1 ; i++)
		gauss[i] = exp((double)-i * (double)i / nn)/sum;
}
#endif

/* even though lookup tables take space - using those speeds kernel calculations tenfold */
static const double standard_deviations[128] = 
{
		 0.0,       0.300387,  0.600773,  0.901160,  1.201547,  1.501933,  1.852320,  2.202706,  2.553093,  2.903480,  3.253866,  3.604253,  3.954640,  4.355026,  4.705413,  5.105799, 
		 5.456186,  5.856573,  6.256959,  6.657346,  7.057733,  7.458119,  7.858506,  8.258892,  8.659279,  9.059666,  9.510052,  9.910439, 10.360826, 10.761212, 11.211599, 11.611986, 
		12.062372, 12.512759, 12.963145, 13.413532, 13.863919, 14.314305, 14.764692, 15.215079, 15.665465, 16.115852, 16.566238, 17.066625, 17.517012, 18.017398, 18.467785, 18.968172, 
		19.418558, 19.918945, 20.419332, 20.869718, 21.370105, 21.870491, 22.370878, 22.871265, 23.371651, 23.872038, 24.372425, 24.872811, 25.373198, 25.923584, 26.423971, 26.924358, 
		27.474744, 27.975131, 28.525518, 29.025904, 29.576291, 30.126677, 30.627064, 31.177451, 31.727837, 32.278224, 32.828611, 33.378997, 33.929384, 34.479771, 35.030157, 35.580544, 
		36.130930, 36.731317, 37.281704, 37.832090, 38.432477, 38.982864, 39.583250, 40.133637, 40.734023, 41.334410, 41.884797, 42.485183, 43.085570, 43.685957, 44.286343, 44.886730, 
		45.487117, 46.087503, 46.687890, 47.288276, 47.938663, 48.539050, 49.139436, 49.789823, 50.390210, 50.990596, 51.640983, 52.291369, 52.891756, 53.542143, 54.192529, 54.842916, 
		55.443303, 56.093689, 56.744076, 57.394462, 58.044849, 58.745236, 59.395622, 60.046009, 60.696396, 61.396782, 62.047169, 62.747556, 63.397942, 64.098329, 64.748715, 65.449102
	
};

static const double descr_approxim_mult[128] = 
{
		 0.0,             576.033927, 1539.585724, 2313.193545, 3084.478025, 3855.885078, 4756.332754, 5657.242476, 6558.536133, 7460.139309, 8361.990613, 9264.041672, 10166.254856, 11199.615571, 12102.233350, 13136.515398, 
		 14039.393687,  15074.393173, 16109.866931, 17145.763345, 18182.036948, 19218.647831, 20255.561010, 21292.745815, 22330.175327, 23367.825876, 24540.507339, 25578.741286, 26752.587529, 27791.291872, 28966.144174, 30005.229117, 
		 31180.955186,  32357.252344, 33534.082488, 34711.410459, 35889.203827, 37067.432679, 38246.069415, 39425.088562, 40604.466591, 41784.181759, 42964.213952, 44284.538859, 45465.382595, 46787.130142, 47968.686878, 49291.724522, 
		 50473.909042,  51798.119528, 53123.060725, 54306.137507, 55632.091099, 56958.688068, 58285.899344, 59613.697438, 60942.056354, 62270.951500, 63600.359608, 64930.258655, 66260.627789, 67737.102560, 69068.620203, 70400.544942, 
		 71879.460632,  73212.395873, 74692.932606, 76026.792904, 77508.839552, 78991.791376, 80327.002820, 81811.308203, 83296.434414, 84782.353155, 86269.037314, 87756.460905, 89244.599028, 90733.427810, 92222.924365, 93713.066749, 
		 95203.833910,  96847.414084, 98339.659244, 99832.465294, 101479.012792, 102973.158567, 104621.682880, 106117.081106, 107767.473327, 109418.953577, 110916.202212, 112569.394820, 114223.592283, 115878.766626, 117534.890826, 119191.938777, 
		120849.885258, 122508.705901, 124168.377156, 125828.876263, 127648.916790, 129311.319925, 130974.481906, 132798.169283, 134463.087846, 136128.703019, 137955.784611, 139784.215161, 141452.370894, 143283.009733, 145114.908442, 146948.037106, 
		148619.357599, 150454.489089, 152290.771330, 154128.177628, 155966.682058, 157971.069246, 159812.039780, 161654.030102, 163497.017214, 165507.250512, 167352.489586, 169365.683736, 171213.071546, 173229.105499, 175078.544507, 177097.303447
	
};

static void calc_gauss_int(int radius, GAUSS_COEFF_TYPE* gauss, GAUSS_COEFF_TYPE* gauss_sums) 
{
	int i = radius;
	double dmult;
	double std_dev;
	if (i <= 1) 
	{
		gauss[0] = 1024;
		gauss_sums[0] = 1024;
		return;
	}
	/* after radius of 128 - gaussian degrades into something weird, 
	   since our colors are only 8 bit */
	if (i > 128) i = 128; 
#if 1
	{
		double nn;
		GAUSS_COEFF_TYPE sum = 1024 ;
		std_dev = standard_deviations[i-1];
		dmult = descr_approxim_mult[i-1];
		nn = 2*std_dev * std_dev ;
		dmult /=nn*PI;
		gauss[0] = (GAUSS_COEFF_TYPE)(dmult + 0.5);
		while( i >= 1 )
		{
			gauss[i] = (GAUSS_COEFF_TYPE)(exp((double)-i * (double)i / nn)*dmult + 0.5);
			gauss_sums[i] = sum;
			sum -= gauss[i];
			--i;
		}
		gauss_sums[0] = sum;
	}
#else 
	double g0, g_last, sum = 0.;
	double n, nn, nPI, nnPI;
	std_dev = (radius - 1) * 0.3003866304;
	do
	{
		sum = 0 ;
		n = std_dev * std_dev;
		nn = 2*n ;
		nPI = n*PI;
		nnPI = nn*PI;
		sum = g0 = 1.0 / nnPI ;
		for (i = 1 ; i < radius-1 ; i++) 
			sum += exp((double)-i * (double)i / nn)/nPI; 
		g_last = exp((double)-i * (double)i / nn)/nnPI; 
		sum += g_last*2.0 ; 
	
		dmult = 1024.0/sum;
		std_dev += 0.05 ;
	}while( g_last*dmult  < 1. );
	gauss[0] = g0 * dmult + 0.5; 
	gauss[(int)radius-1] = g_last * dmult + 0.5;
	dmult /=nnPI;
	for (i = 1 ; i < radius-1 ; i++)
		gauss[i] = exp((double)-i * (double)i / nn)*dmult + 0.5;

#endif	

#if 0
	{
		static int count = 0 ; 
		if( ++count == 16 ) 
		{
			fprintf( stderr, "\n		" );
			count = 0 ;
		}
			
		fprintf(stderr, "%lf, ", dmult*nnPI );			
	}
#endif
#if 0
	{
		int sum_da = 0 ;
		fprintf(stderr, "sum = %f, dmult = %f\n", sum, dmult );
		for (i = 0 ; i < radius ; i++)
		{
//			gauss[i] /= sum;
			sum_da += gauss[i]*2 ;
			fprintf(stderr, "discr_approx(%d) = %d\n", i, gauss[i]);
		}
		sum_da -= gauss[0];
	
		fprintf(stderr, "sum_da = %d\n", sum_da );			
	}
#endif	
}


/***********************************************************************
 * Hue,saturation and lightness adjustments.
 **********************************************************************/
ASImage*
adjust_asimage_hsv( ASVisual *asv, ASImage *src,
				    int offset_x, int offset_y,
	  			    int to_width, int to_height,
					int affected_hue, int affected_radius,
					int hue_offset, int saturation_offset, int value_offset,
					ASAltImFormats out_format,
					unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder *imdec ;
	ASImageOutput  *imout ;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

LOCAL_DEBUG_CALLER_OUT( "offset_x = %d, offset_y = %d, to_width = %d, to_height = %d, hue = %u", offset_x, offset_y, to_width, to_height, affected_hue );
	if( src == NULL ) 
		return NULL;
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, offset_x, offset_y, to_width, 0, NULL)) == NULL )
		return NULL;

	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color);
	set_decoder_shift(imdec,8);
	if((imout = start_image_output( asv, dst, out_format, 8, quality)) == NULL )
	{
        destroy_asimage( &dst );
    }else
	{
	    CARD32 from_hue1 = 0, from_hue2 = 0, to_hue1 = 0, to_hue2 = 0 ;
		int y, max_y = to_height;
		Bool do_greyscale = False ; 

		affected_hue = normalize_degrees_val( affected_hue );
		affected_radius = normalize_degrees_val( affected_radius );
		if( value_offset != 0 )
			do_greyscale = (affected_hue+affected_radius >= 360 || affected_hue-affected_radius <= 0 );
		if( affected_hue > affected_radius )
		{
			from_hue1 = degrees2hue16(affected_hue-affected_radius);
			if( affected_hue+affected_radius >= 360 )
			{
				to_hue1 = MAX_HUE16 ;
				from_hue2 = MIN_HUE16 ;
				to_hue2 = degrees2hue16(affected_hue+affected_radius-360);
			}else
				to_hue1 = degrees2hue16(affected_hue+affected_radius);
		}else
		{
			from_hue1 = degrees2hue16(affected_hue+360-affected_radius);
			to_hue1 = MAX_HUE16 ;
			from_hue2 = MIN_HUE16 ;
			to_hue2 = degrees2hue16(affected_hue+affected_radius);
		}
		hue_offset = degrees2hue16(hue_offset);
		saturation_offset = (saturation_offset<<16) / 100;
		value_offset = (value_offset<<16)/100 ;
LOCAL_DEBUG_OUT("adjusting actually...%s", "");
		if( to_height > src->height )
		{
			imout->tiling_step = src->height ;
			max_y = src->height ;
		}
		for( y = 0 ; y < max_y ; y++  )
		{
			register int x = imdec->buffer.width;
			CARD32 *r = imdec->buffer.red;
			CARD32 *g = imdec->buffer.green;
			CARD32 *b = imdec->buffer.blue ;
			long h, s, v ;
			imdec->decode_image_scanline( imdec );
			while( --x >= 0 )
			{
				if( (h = rgb2hue( r[x], g[x], b[x] )) != 0 )
				{
#ifdef DEBUG_HSV_ADJUSTMENT
					fprintf( stderr, "IN  %d: rgb = #%4.4lX.%4.4lX.%4.4lX hue = %ld(%d)        range is (%ld - %ld, %ld - %ld), dh = %d\n", __LINE__, r[x], g[x], b[x], h, ((h>>8)*360)>>8, from_hue1, to_hue1, from_hue2, to_hue2, hue_offset );
#endif

					if( affected_radius >= 180 ||
						(h >= (int)from_hue1 && h <= (int)to_hue1 ) ||
						(h >= (int)from_hue2 && h <= (int)to_hue2 ) )

					{
						s = rgb2saturation( r[x], g[x], b[x] ) + saturation_offset;
						v = rgb2value( r[x], g[x], b[x] )+value_offset;
						h += hue_offset ;
						if( h > MAX_HUE16 )
							h -= MAX_HUE16 ;
						else if( h == 0 )
							h =  MIN_HUE16 ;
						else if( h < 0 )
							h += MAX_HUE16 ;
						if( v < 0 ) v = 0 ;
						else if( v > 0x00FFFF ) v = 0x00FFFF ;

						if( s < 0 ) s = 0 ;
						else if( s > 0x00FFFF ) s = 0x00FFFF ;

						hsv2rgb ( (CARD32)h, (CARD32)s, (CARD32)v, &r[x], &g[x], &b[x]);

#ifdef DEBUG_HSV_ADJUSTMENT
						fprintf( stderr, "OUT %d: rgb = #%4.4lX.%4.4lX.%4.4lX hue = %ld(%ld)     sat = %ld val = %ld\n", __LINE__, r[x], g[x], b[x], h, ((h>>8)*360)>>8, s, v );
#endif
					}
				}else if( do_greyscale ) 
				{
					int tmp = (int)r[x] + value_offset ; 
					g[x] = b[x] = r[x] = (tmp < 0)?0:((tmp>0x00FFFF)?0x00FFff:tmp);
				}
			}
			imdec->buffer.flags = 0xFFFFFFFF ;
			imout->output_image_scanline( imout, &(imdec->buffer), 1);
		}
		stop_image_output( &imout );
	}
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	return dst;
}

static void 
slice_scanline( ASScanline *dst, ASScanline *src, int start_x, int end_x, ASScanline *middle )
{
	CARD32 *sa = src->alpha, *da = dst->alpha ;
	CARD32 *sr = src->red, *dr = dst->red ;
	CARD32 *sg = src->green, *dg = dst->green ;
	CARD32 *sb = src->blue, *db = dst->blue ;
	int max_x = min( start_x, (int)dst->width);
	int tail = (int)src->width - end_x ; 
	int tiling_step = end_x - start_x ;
	int x1, x2, max_x2 ;

	LOCAL_DEBUG_OUT( "start_x = %d, end_x = %d, tail = %d, tiling_step = %d, max_x = %d", start_x, end_x, tail, tiling_step, max_x );
	for( x1 = 0 ; x1 < max_x ; ++x1 ) 
	{
		da[x1] = sa[x1] ; 
		dr[x1] = sr[x1] ; 
		dg[x1] = sg[x1] ; 
		db[x1] = sb[x1] ;	  
	}
	if( x1 >= dst->width )
		return;
	/* middle portion */
	max_x2 = (int) dst->width - tail ; 
	max_x = min(end_x, max_x2);		
	if( middle ) 
	{	  
		CARD32 *ma = middle->alpha-x1 ;
		CARD32 *mr = middle->red-x1 ;
		CARD32 *mg = middle->green-x1 ;
		CARD32 *mb = middle->blue-x1 ;
		LOCAL_DEBUG_OUT( "middle->width = %d", middle->width );

		for( ; x1 < max_x2 ; ++x1 )
		{
			da[x1] = ma[x1] ; 
			dr[x1] = mr[x1] ; 
			dg[x1] = mg[x1] ; 
			db[x1] = mb[x1] ;	  
		}	 
		LOCAL_DEBUG_OUT( "%d: %8.8lX %8.8lX %8.8lX %8.8lX", x1-1, ma[x1-1], mr[x1-1], mg[x1-1], mb[x1-1] );
	}else
	{	
		for( ; x1 < max_x ; ++x1 )
		{
  			x2 = x1 ;
			for( x2 = x1 ; x2 < max_x2 ; x2 += tiling_step )
			{
				da[x2] = sa[x1] ; 
				dr[x2] = sr[x1] ; 
				dg[x2] = sg[x1] ; 
				db[x2] = sb[x1] ;	  
			}				  
		}	 
	}
	/* tail portion */
	x1 = src->width - tail ;
	x2 = max(max_x2,start_x) ; 
	max_x = src->width ;
	max_x2 = dst->width ;
	for( ; x1 < max_x && x2 < max_x2; ++x1, ++x2 )
	{
		da[x2] = sa[x1] ; 
		dr[x2] = sr[x1] ; 
		dg[x2] = sg[x1] ; 
		db[x2] = sb[x1] ;	  
	}
}	 


ASImage*
slice_asimage2( ASVisual *asv, ASImage *src,
			   int slice_x_start, int slice_x_end,
			   int slice_y_start, int slice_y_end,
			   int to_width,
			   int to_height,
			   Bool scale,
			   ASAltImFormats out_format,
			   unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder *imdec = NULL ;
	ASImageOutput  *imout = NULL ;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

LOCAL_DEBUG_CALLER_OUT( "scale = %d, sx1 = %d, sx2 = %d, sy1 = %d, sy2 = %d, to_width = %d, to_height = %d", scale, slice_x_start, slice_x_end, slice_y_start, slice_y_end, to_width, to_height );
	if( src == NULL )
		return NULL;
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, 0, 0, src->width, 0, NULL)) == NULL )
		return NULL;
	if( slice_x_end == 0 && slice_x_start > 0 ) 
		slice_x_end = slice_x_start + 1 ;
	if( slice_y_end == 0 && slice_y_start > 0 ) 
		slice_y_end = slice_y_start + 1 ;
	if( slice_x_end > src->width ) 
		slice_x_end = src->width ;
	if( slice_y_end > src->height ) 
		slice_y_end = src->height ;
	if( slice_x_start > slice_x_end ) 
		slice_x_start = (slice_x_end > 0 ) ? slice_x_end-1 : 0 ;
	if( slice_y_start > slice_y_end ) 
		slice_y_start = (slice_y_end > 0 ) ? slice_y_end-1 : 0 ;

LOCAL_DEBUG_OUT( "sx1 = %d, sx2 = %d, sy1 = %d, sy2 = %d, to_width = %d, to_height = %d", slice_x_start, slice_x_end, slice_y_start, slice_y_end, to_width, to_height );
	dst = create_destination_image( to_width, to_height, out_format, compression_out, src->back_color);
	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
        destroy_asimage( &dst );
    }else 
	{	
		int y1, y2 ;
		int max_y = min( slice_y_start, (int)dst->height);
		int tail = (int)src->height - slice_y_end ; 
		int max_y2 = (int) dst->height - tail ; 		
		ASScanline *out_buf = prepare_scanline( to_width, 0, NULL, asv->BGR_mode );

		out_buf->flags = 0xFFFFFFFF ;

		if( scale ) 
		{
			ASImageDecoder *imdec_scaled ;
			ASImage *tmp ;
			int x_middle = to_width - slice_x_start ; 
			int x_right = src->width - (slice_x_end+1) ;
			int y_middle = to_height - slice_y_start ;
			int y_bottom = src->height - (slice_y_end+1) ;
			x_middle = ( x_middle <= x_right  )? 0 : x_middle-x_right ;
			y_middle = ( y_middle <= y_bottom )? 0 : y_middle-y_bottom ;
			
			if( x_middle > 0 )
			{	
				tmp = scale_asimage2( asv, src, slice_x_start, 0, 
									   slice_x_end-slice_x_start, max_y, 
									   x_middle, max_y, ASA_ASImage, 0, quality );
				imdec_scaled = start_image_decoding(asv, tmp, SCL_DO_ALL, 0, 0, 0, 0, NULL) ;
				for( y1 = 0 ; y1 < max_y ; ++y1 ) 
				{
					imdec->decode_image_scanline( imdec );
					imdec_scaled->decode_image_scanline( imdec_scaled );
					slice_scanline( out_buf, &(imdec->buffer), slice_x_start, slice_x_end, &(imdec_scaled->buffer) );
					imout->output_image_scanline( imout, out_buf, 1);
				}	 
				stop_image_decoding( &imdec_scaled );
				destroy_asimage( &tmp );
			}else
			{	
				for( y1 = 0 ; y1 < max_y ; ++y1 ) 
				{
					imdec->decode_image_scanline( imdec );
					imout->output_image_scanline( imout, &(imdec->buffer), 1);
				}	 
			}
			/*************************************************************/
			/* middle portion */
			if( y_middle > 0 ) 
			{	
				ASImage *sides ;
				ASImageDecoder *imdec_sides ;
				sides = scale_asimage2( asv, src, 0, slice_y_start, 
								   		src->width, slice_y_end-slice_y_start,
								   		src->width, y_middle, ASA_ASImage, 0, quality );
				imdec_sides = start_image_decoding(asv, sides, SCL_DO_ALL, 0, 0, 0, 0, NULL) ;
/*				print_asimage( sides, 0, __FUNCTION__, __LINE__ ); */
				if( x_middle > 0 ) 
				{
					tmp = scale_asimage2( asv, sides, slice_x_start, 0, 
									   	slice_x_end-slice_x_start, y_middle, 
									   	x_middle, y_middle, ASA_ASImage, 0, quality );
/*					print_asimage( tmp, 0, __FUNCTION__, __LINE__ ); */

					imdec_scaled = start_image_decoding(asv, tmp, SCL_DO_ALL, 0, 0, 0, 0, NULL) ;
					for( y1 = 0 ; y1 < y_middle ; ++y1 ) 
					{
						imdec_sides->decode_image_scanline( imdec_sides );
						imdec_scaled->decode_image_scanline( imdec_scaled );
						slice_scanline( out_buf, &(imdec_sides->buffer), slice_x_start, slice_x_end, &(imdec_scaled->buffer) );
						imout->output_image_scanline( imout, out_buf, 1);
					}	 
					stop_image_decoding( &imdec_scaled );
					destroy_asimage( &tmp );
				
				}else
				{
					for( y1 = 0 ; y1 < y_middle ; ++y1 ) 
					{
						imdec_sides->decode_image_scanline( imdec_sides );
						imout->output_image_scanline( imout, &(imdec->buffer), 1);
					}	 
				}		 
				stop_image_decoding( &imdec_sides );
				destroy_asimage( &sides );
			}			
			/*************************************************************/
			/* bottom portion */

			y2 =  max(max_y2,(int)slice_y_start) ; 
			y1 = src->height - tail ;
			imout->next_line = y2 ; 
			imdec->next_line = y1 ;
			max_y = src->height ;
			if( y2 + max_y - y1 > dst->height ) 
				max_y = dst->height + y1 - y2 ;
			LOCAL_DEBUG_OUT( "y1 = %d, max_y = %d", y1, max_y );		   
			if( x_middle > 0 )
			{	
				tmp = scale_asimage2( asv, src, slice_x_start, y1, 
									   slice_x_end-slice_x_start, src->height-y1, 
									   x_middle, src->height-y1, ASA_ASImage, 0, quality );
				imdec_scaled = start_image_decoding(asv, tmp, SCL_DO_ALL, 0, 0, 0, 0, NULL) ;
				for( ; y1 < max_y ; ++y1 )
				{
					imdec->decode_image_scanline( imdec );
					imdec_scaled->decode_image_scanline( imdec_scaled );
					slice_scanline( out_buf, &(imdec->buffer), slice_x_start, slice_x_end, &(imdec_scaled->buffer) );
					imout->output_image_scanline( imout, out_buf, 1);
				}	 
				stop_image_decoding( &imdec_scaled );
				destroy_asimage( &tmp );
			}else
			{	
				for( ; y1 < max_y ; ++y1 )
				{
					imdec->decode_image_scanline( imdec );
					imout->output_image_scanline( imout, &(imdec->buffer), 1);
				}	 
			}
			
		}else	 /* tile middle portion */
		{                      
			imout->tiling_step = 0;
			LOCAL_DEBUG_OUT( "max_y = %d", max_y );
			for( y1 = 0 ; y1 < max_y ; ++y1 ) 
			{
				imdec->decode_image_scanline( imdec );
				slice_scanline( out_buf, &(imdec->buffer), slice_x_start, slice_x_end, NULL );
				imout->output_image_scanline( imout, out_buf, 1);
			}	 
			/* middle portion */
			imout->tiling_step = (int)slice_y_end - (int)slice_y_start;
			max_y = min(slice_y_end, max_y2);
			LOCAL_DEBUG_OUT( "y1 = %d, max_y = %d, tiling_step = %d", y1, max_y, imout->tiling_step );
			for( ; y1 < max_y ; ++y1 )
			{
				imdec->decode_image_scanline( imdec );
				slice_scanline( out_buf, &(imdec->buffer), slice_x_start, slice_x_end, NULL );
				imout->output_image_scanline( imout, out_buf, 1);
			}

			/* bottom portion */
			imout->tiling_step = 0;
			imout->next_line = y2 =  max(max_y2,(int)slice_y_start) ; 
			imdec->next_line = y1 = src->height - tail ;
			max_y = src->height ;
			if( y2 + max_y - y1 > dst->height ) 
				max_y = dst->height + y1 - y2 ;
			LOCAL_DEBUG_OUT( "y1 = %d, max_y = %d", y1, max_y );		   
			for( ; y1 < max_y ; ++y1 )
			{
				imdec->decode_image_scanline( imdec );
				slice_scanline( out_buf, &(imdec->buffer), slice_x_start, slice_x_end, NULL );
				imout->output_image_scanline( imout, out_buf, 1);
			}
		}
		free_scanline( out_buf, False );
		stop_image_output( &imout );
	}
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	return dst;
}

ASImage*
slice_asimage( ASVisual *asv, ASImage *src,
			   int slice_x_start, int slice_x_end,
			   int slice_y_start, int slice_y_end,
			   int to_width, int to_height,
			   ASAltImFormats out_format,
			   unsigned int compression_out, int quality )
{
	
	return slice_asimage2( asv, src, slice_x_start, slice_x_end,
			   			   slice_y_start, slice_y_end, to_width,  to_height,
			   				False, out_format, compression_out, quality );
}


ASImage *
pixelize_asimage( ASVisual *asv, ASImage *src,
			      int clip_x, int clip_y, int clip_width, int clip_height,
				  int pixel_width, int pixel_height,
				  ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder *imdec ;
	ASImageOutput  *imout ;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if (src== NULL)
		return NULL;
		
	if (clip_width <= 0)
		clip_width = src->width;
	if (clip_height <= 0)
		clip_height = src->height;

	if (pixel_width <= 0)
		pixel_width = 1;
	else if (pixel_width > clip_width)
		pixel_width = clip_width;
		
	if (pixel_height <= 0)
		pixel_height = 1;
	else if (pixel_height > clip_height)
		pixel_height = clip_height;
		
LOCAL_DEBUG_CALLER_OUT( "src = %p, offset_x = %d, offset_y = %d, to_width = %d, to_height = %d, pixel_width = %d, pixel_height = %d", src, clip_x, clip_y, clip_width, clip_height, pixel_width, pixel_height );
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, clip_x, clip_y, clip_width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		return NULL;
	}

	dst = create_destination_image( clip_width, clip_height, out_format, compression_out, src->back_color );

	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image output%s", "");
        destroy_asimage( &dst );
    }else
	{
		int y, max_y = clip_height;
LOCAL_DEBUG_OUT("pixelizing actually...%s", "");

		if( pixel_width > 1 || pixel_height > 1 )
		{
			int pixel_h_count = (clip_width+pixel_width-1)/pixel_width;
			ASScanline *pixels = prepare_scanline( pixel_h_count, 0, NULL, asv->BGR_mode );
			ASScanline *out_buf = prepare_scanline( clip_width, 0, NULL, asv->BGR_mode );
			int lines_count = 0;

			out_buf->flags = SCL_DO_ALL;
			
			for( y = 0 ; y < max_y ; y++  )
			{
				int pixel_x = 0, x ;
				imdec->decode_image_scanline( imdec );
				for (x = 0; x < clip_width; x += pixel_width)
				{
					int xx = x+pixel_width;
					ASScanline *srcsl = &(imdec->buffer);
					
					if (xx > clip_width)
						xx = clip_width;
					
					while ( --xx >= x)
					{
						pixels->red[pixel_x] += srcsl->red[xx];
						pixels->green[pixel_x] += srcsl->green[xx];
						pixels->blue[pixel_x] += srcsl->blue[xx];
						pixels->alpha[pixel_x] += srcsl->alpha[xx];
					}
					++pixel_x;
				}
				if (++lines_count >= pixel_height || y == max_y-1)
				{
					pixel_x = 0;
					
					for (x = 0; x < clip_width; x += pixel_width)
					{
						int xx = (x + pixel_width> clip_width) ? clip_width : x + pixel_width;
						int count = (xx - x) * lines_count;
						CARD32 r = pixels->red [pixel_x] / count;
						CARD32 g = pixels->green [pixel_x] / count;
						CARD32 b = pixels->blue [pixel_x] / count;
						CARD32 a = pixels->alpha [pixel_x] / count;
						
						pixels->red [pixel_x] = 0;
						pixels->green [pixel_x] = 0;
						pixels->blue [pixel_x] = 0;
						pixels->alpha [pixel_x] = 0;

						if (xx > clip_width)
							xx = clip_width;

						while ( --xx >= x)
						{
							out_buf->red[xx] 	= r;
							out_buf->green[xx]  = g;
							out_buf->blue[xx] 	= b;
							out_buf->alpha[xx]  = a;
						}

						++pixel_x;
					}
					while (lines_count--)
						imout->output_image_scanline( imout, out_buf, 1);
					lines_count = 0;
				}
			}
			free_scanline( out_buf, False );
			free_scanline( pixels, False );
		}else
			for( y = 0 ; y < max_y ; y++  )
			{
				imdec->decode_image_scanline( imdec );
				imout->output_image_scanline( imout, &(imdec->buffer), 1);
			}
		stop_image_output( &imout );
	}
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	return dst;
}

ASImage *
color2alpha_asimage( ASVisual *asv, ASImage *src,
			         int clip_x, int clip_y, int clip_width, int clip_height,
				     ARGB32 color,
				     ASAltImFormats out_format, unsigned int compression_out, int quality )
{
	ASImage *dst = NULL ;
	ASImageDecoder *imdec ;
	ASImageOutput  *imout ;
	START_TIME(started);

	if( asv == NULL ) 	asv = &__transform_fake_asv ;

	if (src== NULL)
		return NULL;
		
	if (clip_width <= 0)
		clip_width = src->width;
	if (clip_height <= 0)
		clip_height = src->height;

		
LOCAL_DEBUG_CALLER_OUT( "src = %p, offset_x = %d, offset_y = %d, to_width = %d, to_height = %d, color = #%8.8x", src, clip_x, clip_y, clip_width, clip_height, color );
	if( (imdec = start_image_decoding(asv, src, SCL_DO_ALL, clip_x, clip_y, clip_width, 0, NULL)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image decoding%s", "");
		return NULL;
	}

	dst = create_destination_image( clip_width, clip_height, out_format, compression_out, src->back_color );

	if((imout = start_image_output( asv, dst, out_format, 0, quality)) == NULL )
	{
		LOCAL_DEBUG_OUT( "failed to start image output%s", "");
        destroy_asimage( &dst );
    }else
	{
		int y, max_y = min(clip_height,(int)src->height);
		CARD32 cr = ARGB32_RED8(color);
		CARD32 cg = ARGB32_GREEN8(color);
		CARD32 cb = ARGB32_BLUE8(color);
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)					
fprintf (stderr, "color2alpha():%d: color: red = 0x%8.8X green = 0x%8.8X blue = 0x%8.8X\n", __LINE__, cr, cg, cb);
#endif

		for( y = 0 ; y < max_y ; y++  )
		{
			int x ;
			ASScanline *srcsl = &(imdec->buffer);
			imdec->decode_image_scanline( imdec );
			for (x = 0; x < imdec->buffer.width; ++x)
			{
				CARD32 r = srcsl->red[x];
				CARD32 g = srcsl->green[x];
				CARD32 b = srcsl->blue[x];
				CARD32 a = srcsl->alpha[x];
				/* the following logic is stolen from gimp and altered for our color format and beauty*/
				{
					CARD32 aa = a, ar, ag, ab;
					
#define AS_MIN_CHAN_VAL 	2			/* GIMP uses 0.0001 */
#define AS_MAX_CHAN_VAL 	255			/* GIMP uses 1.0 */
#define MAKE_CHAN_ALPHA_FROM_COL(chan) \
					((c##chan < AS_MIN_CHAN_VAL)? (chan)<<4 : \
						((chan > c##chan)? ((chan - c##chan)<<12) / (AS_MAX_CHAN_VAL - c##chan) : \
							((c##chan - chan)<<12) / c##chan))

					ar = MAKE_CHAN_ALPHA_FROM_COL(r);
					ag = MAKE_CHAN_ALPHA_FROM_COL(g);
					ab = MAKE_CHAN_ALPHA_FROM_COL(b);
#undef 	MAKE_CHAN_ALPHA_FROM_COL
			
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)					
fprintf (stderr, "color2alpha():%d: src(argb): %8.8X %8.8X %8.8X %8.8X; ", __LINE__, a, r, g, b);
#endif
  					a = (ar > ag) ? max(ar, ab) : max(ag,ab);
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)					
fprintf (stderr, "alpha: (%8.8X %8.8X %8.8X)->%8.8X; ", ar, ag, ab, a);
#endif

					if (a == 0) a = 1;
#if defined(USE_STUPID_GIMP_WAY_DESTROYING_COLORS)
#define APPLY_ALPHA_TO_CHAN(chan)  ({int __s = chan; int __c = c##chan; __c += (( __s - __c)*4096)/(int)a;(__c<=0)?0:((__c>=255)?255:__c);})
#else
#define APPLY_ALPHA_TO_CHAN(chan)	chan	
#endif
	  				srcsl->red[x] 	= APPLY_ALPHA_TO_CHAN(r);
					srcsl->green[x] 	= APPLY_ALPHA_TO_CHAN(g);
		  			srcsl->blue[x] 	= APPLY_ALPHA_TO_CHAN(b);
#undef APPLY_ALPHA_TO_CHAN
					a = a*aa>>12;
	  				srcsl->alpha[x] = (a>255)?255:a;

#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)					
fprintf (stderr, "result: %8.8X %8.8X %8.8X %8.8X.\n", src->alpha[x], src->red[x], src->green[x], src->blue[x]);
#endif

				}
				/* end of gimp code */
			}
			imout->output_image_scanline( imout, srcsl, 1);
		}
		stop_image_output( &imout );
	}
	stop_image_decoding( &imdec );

	SHOW_TIME("", started);
	return dst;
}


/* ********************************************************************************/
/* The end !!!! 																 */
/* ********************************************************************************/

