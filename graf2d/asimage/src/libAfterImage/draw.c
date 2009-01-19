/*
 * Copyright (c) 2004 Sasha Vasko <sasha at aftercode.net>
 *
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

#undef  LOCAL_DEBUG

#undef DEBUG_ELLIPS

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

#ifdef _WIN32
#define Long64_t __int64 
#else
#define Long64_t long long
#endif

/*#define LOCAL_DEBUG*/
/*#define DO_CLOCKING*/

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asvisual.h"
#include "asimage.h"
#include "draw.h"


#define CTX_SELECT_CANVAS(ctx)	(get_flags((ctx)->flags, ASDrawCTX_UsingScratch)?(ctx)->scratch_canvas:(ctx)->canvas)

#define CTX_PUT_PIXEL(ctx,x,y,ratio) (ctx)->apply_tool_func(ctx,x,y,ratio)
#define CTX_FILL_HLINE(ctx,x_from,y,x_to,ratio) (ctx)->fill_hline_func(ctx,x_from,y,x_to,ratio)


#define CTX_DEFAULT_FILL_THRESHOLD	126
#define CTX_ELLIPS_FILL_THRESHOLD	140

CARD32 _round1x1[1] =
{  255 };

CARD32 _round3x3[9] =
{ 64,  80,  64,
  80, 255,  80,
  64,  80,  64 };

CARD32 _round3x3_solid[9] =
{ 64,  255,  64,
  255, 255,  255,
  64,  255,  64 };

CARD32 _round5x5[25] =
{  0,  24,  64,  24,   0,
  24,  64,  80,  64,  24,
  64,  80, 255,  80,  64,
  24,  64,  80,  64,  24,
   0,  24,  64,  24,   0,};

CARD32 _round5x5_solid[25] =
{  0,  255,  255,  255,   0,
  255,  255, 255,  255,  255,
  255,  255, 255,  255,  255,
  255,  255, 255,  255,  255,
   0,  255,  255,  255,   0,};

ASDrawTool StandardBrushes[AS_DRAW_BRUSHES] = 
{
	{1, 1, 0, 0, _round1x1},
	{3, 3, 1, 1, _round3x3_solid},
	{5, 5, 2, 2, _round5x5_solid}
};	 

/*********************************************************************************/
/* auxilary functions : 											 */
/*********************************************************************************/

static void
apply_tool_2D( ASDrawContext *ctx, int curr_x, int curr_y, CARD32 ratio )
{

/*fprintf( stderr, "(%d,%d)- ratio = %u (0x%X)\n", curr_x, curr_y, ratio, ratio);		*/
	if( ratio  !=  0 ) 
	{	
		CARD32 *src = ctx->tool->matrix ;
		int corner_x = curr_x - ctx->tool->center_x ; 
		int corner_y = curr_y - ctx->tool->center_y ; 
		int tw = ctx->tool->width ;
		int th = ctx->tool->height ;
		int cw = ctx->canvas_width ;
		int ch = ctx->canvas_height ;
		int aw = tw ; 
		int ah = th ;
		CARD32 *dst = CTX_SELECT_CANVAS(ctx) ; 
		int x, y ;

		if( corner_x+tw <= 0 || corner_x >= cw || corner_y+th <= 0 || corner_y >= ch ) 
			return ;
		 
		if( corner_y > 0 ) 
			dst += corner_y * cw ;
		else if( corner_y < 0 ) 
		{
			ah -= -corner_y ;
			src += -corner_y * tw ;
		}

		if( corner_x  > 0 ) 
			dst += corner_x ;  
		else if( corner_x < 0 )
		{	
			src += -corner_x ; 
			aw -= -corner_x ;
		}
	
		if( corner_x + tw > cw ) 
			aw = cw - corner_x;

		if( corner_y + th > ch ) 
			ah = ch - corner_y;

		if( ratio == 255 ) 
		{
			for( y = 0 ; y < ah ; ++y ) 
			{	
				for( x = 0 ; x < aw ; ++x ) 
				{
					if( dst[x] < src[x] ) 
						dst[x] = src[x] ;
					/*
					register CARD32 t = (CARD32)dst[x] + (CARD32)src[x] ;
					dst[x] = t > 255? 255 : t ;
			 		*/
				}
				src += tw ; 
				dst += cw ; 
			}
		}else
		{		 
		/* ratio, implementing anti-aliasing should be applyed to the outer layer of 
		   the tool only ! */		
			CARD32 *tsrc = src-tw;
			CARD32 *tdst = dst-cw;

			for( y = 0 ; y < ah ; ++y ) 
			{	
				CARD32 v1, v2;
				tsrc += tw ; 
				tdst += cw ; 
				v1 = (tsrc[0]*ratio)/255 ;
				v2 = (tsrc[aw-1]*ratio)/255 ;
				if( tdst[0] < v1 ) tdst[0] = v1 ;
				if( tdst[aw-1] < v2 ) tdst[aw-1] = v2 ;
			}
			
			for (x = 1 ; x < aw-1 ; ++x) 
			{	
				CARD32 v1 = (src[x]*ratio)/255 ;
				CARD32 v2 = (tsrc[x]*ratio)/255 ;
				if( dst[x] < v1 ) dst[x] = v1 ;
				if( tdst[x] < v2 ) tdst[x] = v2 ;
			}

			for( y = 1 ; y < ah-1 ; ++y ) 
			{	
				src += tw ; 
				dst += cw ; 
				for( x = 1 ; x < aw-1 ; ++x ) 
					if( dst[x] < src[x] ) 
						dst[x] = src[x] ;
			}
		}
	}
}	   

static inline void alpha_blend_point_argb32( CARD32 *dst, CARD32 value, CARD32 ratio)
{
	CARD32 ta = (ARGB32_ALPHA8(value)*ratio)/255;

	if (ta >= 255)
		*dst = value|0xFF000000;
	else
	{
		CARD32 aa = 255-ta; /* must be 256 or we ! */
		CARD32 orig = *dst;
		CARD32 res = orig&0xFF000000; //((orig&0xFF000000)>>8)*aa + (ta<<24);
		if (res < (ta<<24))
			res = (ta<<24);
		/* red and blue */
		res |= (((orig&0x00FF00FF)*aa + (value&0x00FF00FF)*ta)>>8)&0x00FF00FF;
		/* green */
		res |= (((orig&0x0000FF00)*aa + (value&0x0000FF00)*ta)>>8)&0x0000FF00;
		*dst = res;
	}
}

static void
apply_tool_2D_colored( ASDrawContext *ctx, int curr_x, int curr_y, CARD32 ratio )
{
	if( ratio  !=  0 ) 
	{	
		CARD32 *src = ctx->tool->matrix ;
		int corner_x = curr_x - ctx->tool->center_x ; 
		int corner_y = curr_y - ctx->tool->center_y ; 
		int tw = ctx->tool->width ;
		int th = ctx->tool->height ;
		int cw = ctx->canvas_width ;
		int ch = ctx->canvas_height ;
		int aw = tw ; 
		int ah = th ;
		CARD32 *dst = CTX_SELECT_CANVAS(ctx) ; 
		int x, y ;

		if( corner_x+tw <= 0 || corner_x >= cw || corner_y+th <= 0 || corner_y >= ch ) 
			return ;
		 
		if( corner_y > 0 ) 
			dst += corner_y * cw ;
		else if( corner_y < 0 ) 
		{
			ah -= -corner_y ;
			src += -corner_y * tw ;
		}

		if( corner_x  > 0 ) 
			dst += corner_x ;  
		else if( corner_x < 0 )
		{	
			src += -corner_x ; 
			aw -= -corner_x ;
		}
	
		if( corner_x + tw > cw ) 
			aw = cw - corner_x;

		if( corner_y + th > ch ) 
			ah = ch - corner_y;

		/* ratio, implementing anti-aliasing should be applyed to the outer layer of 
		   the tool only ! */		
		{
			CARD32 *tsrc = src-tw;
			CARD32 *tdst = dst-cw;
		
			if (get_flags(ctx->flags, ASDrawCTX_UsingScratch))
			{
				for( y = 0 ; y < ah ; ++y ) 
				{	
					CARD32 v1, v2;
					tsrc += tw ; 
					tdst += cw ; 
					v1 = (ARGB32_ALPHA8(tsrc[0])*ratio)/255 ;
					v2 = (ARGB32_ALPHA8(tsrc[aw-1])*ratio)/255 ;
					if( tdst[0] < v1 ) tdst[0] = v1 ;
					if( tdst[aw-1] < v2 ) tdst[aw-1] = v2 ;

				}

				for (x = 1 ; x < aw-1 ; ++x) 
				{	
					CARD32 v1 = (ARGB32_ALPHA8(src[x])*ratio)/255 ;
					CARD32 v2 = (ARGB32_ALPHA8(tsrc[x])*ratio)/255 ;
					if( dst[x] < v1 ) dst[x] = v1 ;
					if( tdst[x] < v2 ) tdst[x] = v2 ;
				}

				for( y = 1 ; y < ah-1 ; ++y ) 
				{	
					src += tw ; 
					dst += cw ; 
					for( x = 1 ; x < aw-1 ; ++x ) 
					{
						CARD32 value = ARGB32_ALPHA8(src[x]);
						if( dst[x] < value ) 
							dst[x] = value ;
					}
				}
			}else
			{
				for( y = 0 ; y < ah ; ++y ) 
				{	
					tsrc += tw ; 
					tdst += cw ; 
					alpha_blend_point_argb32( tdst, tsrc[0], ratio);
					alpha_blend_point_argb32( tdst+aw-1, tsrc[aw-1], ratio);
				}
				for (x = 1 ; x < aw-1 ; ++x) 
				{	
					alpha_blend_point_argb32( dst+x, src[x], ratio);
					alpha_blend_point_argb32( tdst+x, tsrc[x], ratio);
				}

				for( y = 1 ; y < ah-1 ; ++y ) 
				{	
					src += tw ; 
					dst += cw ; 
					for( x = 1 ; x < aw-1 ; ++x ) 
						alpha_blend_point_argb32( dst+x, src[x], 255);
				}
			}
		}
	}
}	   


static void
apply_tool_point( ASDrawContext *ctx, int curr_x, int curr_y, CARD32 ratio )
{
	int cw = ctx->canvas_width ;
	if( ratio != 0 && curr_x >= 0 && curr_x < cw && curr_y >= 0 && curr_y < ctx->canvas_height ) 
	{	
		CARD32 value = (ctx->tool->matrix[0]*ratio)/255 ;
		CARD32 *dst = CTX_SELECT_CANVAS(ctx) ;
		dst += curr_y * cw ; 

		if( dst[curr_x] < value ) 
			dst[curr_x] = value ;
	}
}	   


static void 
apply_tool_point_colored(ASDrawContext *ctx, int curr_x, int curr_y, CARD32 ratio)
{
	/* Apply tool point argb32. - stolen from ROOT by Valery Onuchin, with changes */
	int cw = ctx->canvas_width;
	if (curr_x >= 0 && curr_x < cw && curr_y >= 0 && curr_y < ctx->canvas_height && ratio != 0)		
	{
		CARD32 *dst = CTX_SELECT_CANVAS(ctx);
		dst += curr_y * cw + curr_x;
		if (get_flags(ctx->flags, ASDrawCTX_UsingScratch))
		{
			CARD32 value = (ARGB32_ALPHA8(ctx->tool->matrix[0])*ratio)/255 ;
			if( *dst < value ) 
				*dst = value ;
		}		
		else
			alpha_blend_point_argb32( dst, ctx->tool->matrix[0], ratio);
		
	}
}

static void
fill_hline_notile( ASDrawContext *ctx, int x_from, int y, int x_to, CARD32 ratio )
{
	int cw = ctx->canvas_width ;
	if( ratio != 0 && x_to >= 0 && x_from < cw && y >= 0 && y < ctx->canvas_height ) 
	{	
		CARD32 value = ratio ;
		CARD32 *dst = CTX_SELECT_CANVAS(ctx) + y * cw ; 
		int x1 = x_from, x2 = x_to ; 
		if( x1 < 0 ) 
			x1 = 0 ; 
		if( x2 >= cw ) 
			x2 = cw - 1 ; 

		while( x1 <= x2 ) 
		{
			if( dst[x1] < value ) 
				dst[x1] = value ;
			++x1 ;
		}
	}
}

static void 
fill_hline_notile_colored(ASDrawContext *ctx, int x_from, int y, int x_to, CARD32 ratio)
{
	int cw = ctx->canvas_width ;
	if( ratio != 0 && x_to >= 0 && x_from < cw && y >= 0 && y < ctx->canvas_height ) 
	{	
		CARD32 value = ctx->tool->matrix[0] ;
		CARD32 *dst = CTX_SELECT_CANVAS(ctx) + y * cw ; 
		int x1 = x_from, x2 = x_to ; 
		if( x1 < 0 ) 
			x1 = 0 ; 
		if( x2 >= cw ) 
			x2 = cw - 1 ; 

		if (get_flags(ctx->flags, ASDrawCTX_UsingScratch))
		{
			while( x1 <= x2 ) 
			{
				CARD32 value = (ARGB32_ALPHA8(ctx->tool->matrix[0])*ratio)/255 ;
				if( dst[x1] < value ) 
					dst[x1] = value ;
				++x1 ;
			}
		}else
			while( x1 <= x2 ) 
			{
				alpha_blend_point_argb32( dst+x1, value, ratio);
				++x1 ;
			}
	}
}

/*********************************************************************************/
/* drawing functions : 											 */
/*********************************************************************************/
static void
ctx_draw_line_solid( ASDrawContext *ctx, int from_x, int from_y, int to_x, int to_y )
{
	int x, y, end, dir = 1;
	int dx = to_x - from_x ; 
	int dy = to_y - from_y ; 
	if( dx < 0 ) 
		dx = -dx ;
	if( dy < 0 ) 
		dy = -dy ;
	
	if( dx >= dy ) 
	{	
      	int Dy = -dx + 2*dy;
      	int inct = 2*dy;
		int incf = 2*(dy - dx);
		
		if( to_y > from_y ) 
		{	x = from_x ; y = from_y ; end = to_x ; }
		else
		{	x = to_x ; y = to_y ; end = from_x ; 	}
		
		if( end < x ) 
			dir = -1 ;				 
      	
		CTX_PUT_PIXEL( ctx, x, y, 255 );
      	while(x != end)
		{
        	x += dir;
        	if(Dy > 0)
			{
          		Dy += incf;
          		++y;
        	}else 
				Dy += inct;
        	CTX_PUT_PIXEL( ctx, x, y, 255 );
      	}
	}else
	{
      	int Dx = -dy + 2*dx;
      	int inct = 2*dx;
		int incf = 2*(dx - dy);
		
		if( to_x > from_x ) 
		{	y = from_y ; x = from_x ; end = to_y ; }
		else
		{	y = to_y ; x = to_x ; end = from_y ; 	}
		
		if( end < y ) 
			dir = -1 ;				 
      	
		CTX_PUT_PIXEL( ctx, x, y, 255 );
      	while(y != end)
		{
        	y += dir;
        	if(Dx > 0)
			{
          		Dx += incf;
          		++x;
        	}else 
				Dx += inct;
        	CTX_PUT_PIXEL( ctx, x, y, 255 );
      	}
	}		 
}	 

static void
ctx_draw_line_solid_aa( ASDrawContext *ctx, int from_x, int from_y, int to_x, int to_y )
{
#define RATIO_t	int
	int x, y, end, dir = 1;
	int dx = to_x - from_x ; 
	int dy = to_y - from_y ; 
	if( dx < 0 ) 
		dx = -dx ;
	if( dy < 0 ) 
		dy = -dy ;

	if( dx == 0 || dy == 0 ) 
	{	
		ctx_draw_line_solid( ctx, from_x, from_y, to_x, to_y );
		return ;
	}
	
	if( dx >= dy ) 
	{	
		RATIO_t ratio2 = 0x007FFFFF/dx ; 
		CARD32 value = 0x003FFFFF; 
		CARD32 value_incr = ratio2*dy ; 
		int value_decr = (dx - dy)*ratio2 ;

		
		if( to_y > from_y ) 
		{	x = from_x ; y = from_y ; end = to_x ; }
		else
		{	x = to_x ; y = to_y ; end = from_x ; 	}
		
		if( end < x ) 	dir = -1 ;				 
      	
		CTX_PUT_PIXEL( ctx, x, y, 255 );
/*		LOCAL_DEBUG_OUT( "x = %d, y = %d, dir = %d, value = %d, incr = %d, decr = %d, ratio = %d, value_incr = %d", 
						 x, y, dir, value, incr, decr, ratio, value_incr );
  */			
      	while(x != end)
		{
        	x += dir;
        	if( (int)value > value_decr)
			{
				value = (int)value - value_decr ;
          		++y;
        	}else 
				value += value_incr ; 

			{
				RATIO_t above = (value&0x00FF0000)>>16;
				/* LOCAL_DEBUG_OUT( "x = %d, y = %d, value = %d, above = %d",  x, y, value, above ); */
				switch( (above>>5)&0x03 )
				{
					case 0 :  /* 0 - 32 */ 
						above = 128 - above ;
						CTX_PUT_PIXEL( ctx, x, y-1, above ) ;
						CTX_PUT_PIXEL( ctx, x, y, (~(above>>1))&0x00FF ) ; 
						break ;	  
					case 1 :  /* 32 - 64 */ 
						{
							RATIO_t a1 = (above - 32) ;
							CTX_PUT_PIXEL( ctx, x, y+1, a1 ) ;
							above = (~above)&0x7f ;
							CTX_PUT_PIXEL( ctx, x, y-1, above - a1 ) ;
							CTX_PUT_PIXEL( ctx, x, y, 255 ) ; 
						}
						break ;
					case 2 :  /* 64 - 96 */  
						{
							RATIO_t a1 = (96 - above) ;	  
							CTX_PUT_PIXEL( ctx, x, y-1, a1 ) ;
							CTX_PUT_PIXEL( ctx, x, y, 255 ) ; 
							CTX_PUT_PIXEL( ctx, x, y+1, above - a1 ) ;
						}
						break ;
					case 3 :  /* 96 - 128 */ 
						{
							above -= ((~above)&0x7f)>>1 ;	  
							CTX_PUT_PIXEL( ctx, x, y, (~(above>>1))&0x00FF ) ; 
							CTX_PUT_PIXEL( ctx, x, y+1, above ) ;
						}
						break ;
				}
			}
      	}
	}else
	{
		RATIO_t ratio2 = 0x007FFFFF/dy ; 
		CARD32 value = 0x003FFFFF; 
		CARD32 value_incr = ratio2*dx ; 
		int value_decr = (dy - dx)*ratio2 ;

		
		if( to_x > from_x ) 
		{	y = from_y ; x = from_x ; end = to_y ; }
		else
		{	y = to_y ; x = to_x ; end = from_y ; 	}
		
		if( end < y ) 	dir = -1 ;				 
      	
		CTX_PUT_PIXEL( ctx, x, y, 255 );
/*		LOCAL_DEBUG_OUT( "x = %d, y = %d, dir = %d, value = %d, incr = %d, decr = %d, ratio = %d, value_incr = %d", 
						 x, y, dir, value, incr, decr, ratio, value_incr );
  */			
      	while(y != end)
		{
        	y += dir;
        	if( (int)value > value_decr)
			{
				value = (int)value - value_decr ;
          		++x;
        	}else 
				value += value_incr ; 

			{
				RATIO_t above = (value&0x00FF0000)>>16;
				/* LOCAL_DEBUG_OUT( "x = %d, y = %d, value = %d, above = %d",  x, y, value, above ); */
				switch( (above>>5)&0x03 )
				{
					case 0 :  /* 0 - 32 */ 
						above = 128 - above ;
						CTX_PUT_PIXEL( ctx, x-1, y, above ) ;
						CTX_PUT_PIXEL( ctx, x, y, (~(above>>1))&0x00FF ) ; 
						break ;	  
					case 1 :  /* 32 - 64 */ 
						{
							RATIO_t a1 = (above - 32) ;
							CTX_PUT_PIXEL( ctx, x+1, y, a1 ) ;
							above = (~above)&0x7f ;
							CTX_PUT_PIXEL( ctx, x-1, y, above - a1 ) ;
							CTX_PUT_PIXEL( ctx, x, y, 255 ) ; 
						}
						break ;
					case 2 :  /* 64 - 96 */  
						{
							RATIO_t a1 = (96 - above) ;	  
							CTX_PUT_PIXEL( ctx, x-1, y, a1 ) ;
							CTX_PUT_PIXEL( ctx, x, y, 255 ) ; 
							CTX_PUT_PIXEL( ctx, x+1, y, above - a1 ) ;
						}
						break ;
					case 3 :  /* 96 - 128 */ 
						{
							above -= ((~above)&0x7f)>>1 ;	  
							CTX_PUT_PIXEL( ctx, x, y, (~(above>>1))&0x00FF ) ; 
							CTX_PUT_PIXEL( ctx, x+1, y, above ) ;
						}
						break ;
				}
			}
      	}
	}		 
}	 

#define SUPERSAMPLING_BITS	8
#define SUPERSAMPLING_MASK	0x000000FF

static inline void 
render_supersampled_pixel( ASDrawContext *ctx, int xs, int ys )
{
	if( xs >= 0 && ys >= 0 )
	{	
		unsigned int xe = xs&SUPERSAMPLING_MASK ; 
		unsigned int nxe = (~xs)&SUPERSAMPLING_MASK ; 
		unsigned int x = xs>>SUPERSAMPLING_BITS;
		unsigned int ye = ys&SUPERSAMPLING_MASK ; 
		unsigned int nye = (~ys)&SUPERSAMPLING_MASK ; 
		unsigned int y = ys>>SUPERSAMPLING_BITS; 
		unsigned int v = (nxe*nye)>>8 ;	  
			
		CTX_PUT_PIXEL( ctx, x, y, v ) ; 
/*		LOCAL_DEBUG_OUT( "x = %d, y = %d, xe = %d, ye = %d, v = 0x%x", x, y, xe, ye, v ); */
		v = (xe*(nye))>>8 ;	  
		CTX_PUT_PIXEL( ctx, x+1, y, v ) ; 
		v = ((nxe)*ye)>>8 ;
		CTX_PUT_PIXEL( ctx, x, ++y, v ) ; 
		v = (xe*ye)>>8 ;	  
		CTX_PUT_PIXEL( ctx, x+1, y, v ) ; 
	}
}	

typedef struct ASCubicBezier
{
	int x0, y0;
	int x1, y1;
	int x2, y2;
	int x3, y3;
}ASCubicBezier;

static void
ctx_draw_bezier( ASDrawContext *ctx, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3 )
{
	int ch = ctx->canvas_height<<8 ;
	int cw = ctx->canvas_width<<8 ;
	
	ASCubicBezier *bstack = NULL ;
	int bstack_size = 0, bstack_used = 0 ;
	
#define ADD_CubicBezier(X0,Y0,X1,Y1,X2,Y2,X3,Y3)	\
	do{ \
		if( ((X0)>=0 || (X1)>=0 || (X2)>=0 || (X3)>=0) && ((X0)<cw||(X1)<cw||(X2)<cw||(X3)<cw) && \
			((Y0)>=0 || (Y1)>=0 || (Y2)>=0 || (Y3)>=0) && ((Y0)<ch||(Y1)<ch||(Y2)<ch||(Y3)<ch) ){ \
			while( bstack_used >= bstack_size ) { \
				bstack_size += 2048/sizeof(ASCubicBezier); \
				bstack = (ASCubicBezier *)realloc( bstack, bstack_size*sizeof(ASCubicBezier)); \
			} \
			LOCAL_DEBUG_OUT( "(%d,%d),(%d,%d),(%d,%d),(%d,%d)", X0, Y0, X1, Y1, X2, Y2, X3, Y3 ); \
			bstack[bstack_used].x0=X0; \
			bstack[bstack_used].y0=Y0; \
			bstack[bstack_used].x1=X1; \
			bstack[bstack_used].y1=Y1; \
			bstack[bstack_used].x2=X2; \
			bstack[bstack_used].y2=Y2; \
			bstack[bstack_used].x3=X3; \
			bstack[bstack_used].y3=Y3; \
			++bstack_used ; \
		} \
	}while(0)


	ADD_CubicBezier(x0,y0,x1,y1,x2,y2,x3,y3);

	while( bstack_used > 0 )
	{
		--bstack_used ;
		x0 = bstack[bstack_used].x0 ;
		y0 = bstack[bstack_used].y0 ;
		x1 = bstack[bstack_used].x1 ;
		y1 = bstack[bstack_used].y1 ;
		x2 = bstack[bstack_used].x2 ;
		y2 = bstack[bstack_used].y2 ;
		x3 = bstack[bstack_used].x3 ;
		y3 = bstack[bstack_used].y3 ;
		{
			int x01 = x0 + ((x1-x0)>>1) ;  	
			int y01 = y0 + ((y1-y0)>>1) ; 
			int x11 = x1 + ((x2-x1)>>1) ; 	   
			int y11 = y1 + ((y2-y1)>>1) ; 
			int x31 = x3 + ((x2-x3)>>1) ; 	   
			int y31 = y3 + ((y2-y3)>>1) ; 

			int x011 = x01 + ((x11-x01)>>1) ;  	  
			int y011 = y01 + ((y11-y01)>>1) ; 
			int x111 = x11 + ((x31-x11)>>1) ;  	  
			int y111 = y11 + ((y31-y11)>>1) ; 

			int x0111 = x011 + ((x111-x011)>>1) ;  	  
			int y0111 = y011 + ((y111-y011)>>1) ; 

			if( (x0&0xFFFFFF00) == (x0111&0xFFFFFF00) && (y0&0xFFFFFF00) == (y0111&0xFFFFFF00) ) 
			{
			  	render_supersampled_pixel( ctx, x0, y0 );
			}else if( x01 != x1 || y01 != y1 || x011 != x2 || y011 != y2 || x0111 != x3 || y0111 != y3 )
			  	ADD_CubicBezier( x0, y0, x01, y01, x011, y011, x0111, y0111 );

			if( (x3&0xFFFFFF00) == (x0111&0xFFFFFF00) && (y3&0xFFFFFF00) == (y0111&0xFFFFFF00) ) 
			{	
			  	render_supersampled_pixel( ctx, x3, y3 );
			}else if( x0111 != x0 || y0111 != y0 || x111 != x1 || y111 != y1 || x31 != x2 || y31 != y2 ) 	
				ADD_CubicBezier( x0111, y0111, x111, y111, x31, y31, x3, y3 );	
		}
	}
	if( bstack ) 
		free( bstack );
}

typedef struct ASScanlinePart
{
	int y ;
	int x0, x1;
}ASScanlinePart;


static void 
ctx_flood_fill( ASDrawContext *ctx, int x_from, int y, int x_to, CARD32 min_val, CARD32 max_val  )
{
	int ch = ctx->canvas_height ;
	int cw = ctx->canvas_width ;
	
	ASScanlinePart *sstack = NULL ;
	int sstack_size = 0, sstack_used = 0 ;
	CARD32 *canvas = CTX_SELECT_CANVAS(ctx);
	
	LOCAL_DEBUG_OUT( "(%d,%d,%d)", x_from, y, x_to );
#define ADD_ScanlinePart(X0,Y0,X1)	\
	do{ \
		if( ((X0)>=0 || (X1)>=0 ) && ((X0)<cw||(X1)<cw) && \
			 (Y0)>=0 && (Y0)<ch ){ \
			while( sstack_used >= sstack_size ) { \
				sstack_size += 2048/sizeof(ASScanlinePart); \
				sstack = realloc( sstack, sstack_size*sizeof(ASScanlinePart)); \
			} \
			LOCAL_DEBUG_OUT( "(%d,%d,%d)", X0, Y0, X1 ); \
			sstack[sstack_used].x0=X0; \
			sstack[sstack_used].y=Y0; \
			sstack[sstack_used].x1=X1; \
			++sstack_used ; \
		} \
	}while(0)

	ADD_ScanlinePart(x_from,y,x_to);

	while( sstack_used > 0 )
	{
		--sstack_used ; 
		x_from = sstack[sstack_used].x0 ; 
		x_to = sstack[sstack_used].x1 ; 
		y = sstack[sstack_used].y ; 
		if( x_from <  0 )
			x_from = 0 ; 
		if( x_to >= cw ) 
			x_to = cw - 1 ; 
		if( x_from <= x_to ) 
		{	
			/* here we have to check for lines below and above */
			if( y > 0 ) 
			{
				CARD32 *data = canvas + (y-1)*cw ; 
				int xc = x_from ; 

				while( xc <= x_to ) 
				{	
					if( data[xc] <= max_val && data[xc] >= min_val ) 
					{	
						int x0 = xc, x1 = xc ; 
						while( x0 >= 0 && data[x0] <= max_val && data[x0]  >= min_val ) --x0;
						++x0 ;
						while( x1 < cw && data[x1] <= max_val && data[x1]  >= min_val ) ++x1;
						--x1 ; 
					
						ADD_ScanlinePart(x0,y-1,x1);					
						LOCAL_DEBUG_OUT( "x = %d, y = %d, data[x] = 0x%X, x0 = %d, x1 = %d", xc, y-1, data[xc], x0, x1 );
						while( xc <= x_to && xc <= x1+1 ) ++xc ; 
					}else
						++xc ;
				}
			}	 
			if( y < ch-1 ) 
			{
				CARD32 *data = canvas + (y+1)*cw ; 
				int xc = x_from ; 

				while( xc <= x_to ) 
				{	
					if( data[xc] <= max_val && data[xc] >= min_val ) 
					{	
						int x0 = xc, x1 = xc ; 
						while( x0 >= 0 && data[x0] <= max_val && data[x0]  >= min_val ) --x0;
						++x0 ;
						while( x1 < cw && data[x1] <= max_val && data[x1]  >= min_val ) ++x1;
						--x1 ; 
						LOCAL_DEBUG_OUT( "x = %d, y = %d, data[x] = 0x%X, x0 = %d, x1 = %d", xc, y+1, data[xc], x0, x1 );
						ADD_ScanlinePart(x0,y+1,x1);					
						while( xc <= x_to && xc <= x1+1 ) ++xc ; 
					}else
						++xc ;
				}
			}	 
 			CTX_FILL_HLINE(ctx,x_from,y,x_to,255);		
		}
	}	 
	if( sstack ) 
		free( sstack );
	
	
}

/*************************************************************************
 * Clip functions 
 *************************************************************************/
Bool 
clip_line( int k, int x0, int y0, int cw, int ch, int *x, int*y ) 	
{
	int new_x = *x ; 
	int new_y = *y ;

	if( new_x < 0 ) 
	{
		new_x = 0 ;
		new_y = (-x0 / k) + y0 ;
	}	 
	if( new_y < 0 ) 
	{
		new_y = 0 ;
		new_x = -y0 * k + x0 ; 
	}	 
	if( new_x < 0 ) 
		return False;

	/* here both new_x and new_y are non-negative */
	if( new_x >= cw ) 
	{
		new_x = cw - 1 ;
		if( k != 0 )
		{
			new_y = (new_x - x0)/k  + y0 ; 
			if( new_y < 0 ) 
				return False;	   
		}
	}		 
	if( new_y >= ch ) 
	{
		new_y = ch - 1 ;
		new_x = (new_y - y0) * k  + x0 ; 
		if( new_x < 0 || new_x >= cw ) 
			return False;	   
	}		 
	*x = new_x ; 
	*y = new_y ;
	return True;
}

/*********************************************************************************/
/* context functions : 											 				 */
/*********************************************************************************/
Bool asim_set_brush( ASDrawContext *ctx, int brush );

ASDrawContext *
create_asdraw_context( unsigned int width, unsigned int height )
{
	ASDrawContext *ctx = safecalloc( 1, sizeof(ASDrawContext));
	
	ctx->canvas_width = width == 0 ? 1 : width ; 
	ctx->canvas_height = height == 0 ? 1 : height ;
	ctx->canvas = safecalloc(  ctx->canvas_width*ctx->canvas_height, sizeof(CARD32));

	asim_set_brush( ctx, 0 ); 
	ctx->fill_hline_func = fill_hline_notile ;
				   
	return ctx;
}	   

void
destroy_asdraw_context( ASDrawContext *ctx )
{
	if( ctx )
	{
		if( ctx->canvas ) 
			free( ctx->canvas );	 
		if( ctx->scratch_canvas ) 
			free( ctx->scratch_canvas );	 
		free( ctx );
	}	 
}	   

Bool
asim_set_brush( ASDrawContext *ctx, int brush ) 
{
	if( brush >= 0 && brush < AS_DRAW_BRUSHES && ctx != NULL ) 
	{
		ctx->tool = &(StandardBrushes[brush]) ;  
		if( ctx->tool->width == 1 && ctx->tool->height == 1 ) 
			ctx->apply_tool_func = apply_tool_point ;
		else 
			ctx->apply_tool_func = apply_tool_2D ; 

		ctx->fill_hline_func = fill_hline_notile ;

		clear_flags (ctx->flags, ASDrawCTX_ToolIsARGB);

		return True;
	}	 
	return False;
}

Bool
asim_set_custom_brush( ASDrawContext *ctx, ASDrawTool *brush) 
{
	if( brush !=NULL && ctx != NULL ) 
	{
		ctx->tool = brush ;  
		if( ctx->tool->width == 1 && ctx->tool->height == 1 ) 
			ctx->apply_tool_func = apply_tool_point ;
		else 
			ctx->apply_tool_func = apply_tool_2D ; 
	
		ctx->fill_hline_func = fill_hline_notile ;

		clear_flags (ctx->flags, ASDrawCTX_ToolIsARGB);

		return True;
	}	 
	return False;
}

Bool
asim_set_custom_brush_colored( ASDrawContext *ctx, ASDrawTool *brush) 
{
	if( brush !=NULL && ctx != NULL ) 
	{
		ctx->tool = brush ;  
		if( ctx->tool->width == 1 && ctx->tool->height == 1 ) 
			ctx->apply_tool_func = apply_tool_point_colored ;
		else 
			ctx->apply_tool_func = apply_tool_2D_colored ; 

		ctx->fill_hline_func = fill_hline_notile_colored ;
		set_flags (ctx->flags, ASDrawCTX_ToolIsARGB);

		return True;
	}	 
	return False;
}


Bool
asim_start_path( ASDrawContext *ctx ) 
{
	if( ctx == NULL ) 
		return False;
	LOCAL_DEBUG_OUT( "scratch_canvas = %p, flags = 0x%lx", ctx->scratch_canvas, ctx->flags );
	if( ctx->scratch_canvas ) 
	{
		if( get_flags( ctx->flags, ASDrawCTX_UsingScratch ) )
			return False;	  
		memset( ctx->scratch_canvas, 0x00, ctx->canvas_width*ctx->canvas_height*sizeof(CARD32) );
	}else
		ctx->scratch_canvas	 = safecalloc(  ctx->canvas_width*ctx->canvas_height, sizeof(CARD32));
	set_flags( ctx->flags, ASDrawCTX_UsingScratch );
	return True;
}

void asim_flood_fill( ASDrawContext *ctx, int x, int y, CARD32 min_val, CARD32 max_val );


Bool
asim_apply_path( ASDrawContext *ctx, int start_x, int start_y, Bool fill, int fill_start_x, int fill_start_y, CARD8 fill_threshold ) 
{
    if( ctx == NULL || !get_flags( ctx->flags, ASDrawCTX_UsingScratch )) 	
		return False;
	
	LOCAL_DEBUG_CALLER_OUT( "start_x = %d, start_y = %d, fill = %d, fill_start_x = %d, fill_start_y = %d",
							start_x, start_y, fill, fill_start_x, fill_start_y );

	/* TODO : contour tracing functionality : */
	if( fill ) 
		asim_flood_fill( ctx, fill_start_x, fill_start_y, 0, fill_threshold==0?CTX_DEFAULT_FILL_THRESHOLD:fill_threshold );	

	clear_flags( ctx->flags, ASDrawCTX_UsingScratch );	 

	/* actually applying scratch : */
	{
		int i = ctx->canvas_width*ctx->canvas_height ;
		if (get_flags (ctx->flags, ASDrawCTX_CanvasIsARGB))
		{
			CARD32 argb = ctx->tool->matrix[ctx->tool->center_y*ctx->tool->width + ctx->tool->center_x];
			while (--i >= 0)
				if (ctx->scratch_canvas[i])
					alpha_blend_point_argb32( ctx->canvas + i, argb, ctx->scratch_canvas[i]);
		}else
		{
			while( --i >= 0 ) 
				if( ctx->canvas[i] < ctx->scratch_canvas[i] )
					ctx->canvas[i] = ctx->scratch_canvas[i] ;
		}
	}		
	return True;
}

Bool
apply_asdraw_context( ASImage *im, ASDrawContext *ctx, ASFlagType filter ) 
{
	int chan ;
	int width, height ;
	if( im == NULL || ctx == NULL || filter == 0 )
		return False;
	
	width = im->width ;
	height = im->height ;
	if( width != ctx->canvas_width || height != ctx->canvas_height )
		return False;
	
	for( chan = 0 ; chan < IC_NUM_CHANNELS;  chan++ )
		if( get_flags( filter, 0x01<<chan) )
		{
			int y;
			register ASStorageID *rows = im->channels[chan] ;
			register CARD32 *canvas_row = ctx->canvas ; 
			for( y = 0 ; y < height ; ++y )
			{	
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
				{
					int i;
					fprintf( stderr, "row %d, canvas_row = %p:", y, canvas_row );
					for( i = 0 ; i < width ; ++i)
						fprintf( stderr, "0x%8.8lX ", canvas_row[i] );
					fprintf( stderr, "\n" );	
				}
#endif
				if( rows[y] ) 
					forget_data( NULL, rows[y] ); 
				rows[y] = store_data( NULL, (CARD8*)canvas_row, width*sizeof(CARD32), ASStorage_32Bit|ASStorage_RLEDiffCompress, 0);
				canvas_row += width ; 
			}
		}
	return True;
}	   

/**************************************************************************/
/* generic line drawing - uses calls supplied function to actually render 
 * pixels :
 **************************************************************************/
/**************************************************************************/
void asim_move_to( ASDrawContext *ctx, int dst_x, int dst_y );

void
asim_line_to_generic( ASDrawContext *ctx, int dst_x, int dst_y, void (*func)(ASDrawContext*,int,int,int,int))
{
	if( ctx ) 
	{
		int from_x = ctx->curr_x ; 	
		int from_y = ctx->curr_y ; 
		int to_x = dst_x ; 	
		int to_y = dst_y ;
		int cw = ctx->canvas_width ; 
		int ch = ctx->canvas_height ; 
		
		asim_move_to( ctx, dst_x, dst_y );	 				   

		if( to_y == from_y ) 
		{
			if( to_y < 0 || to_y >= ch ) 
				return ; 
			if( from_x < 0 ) 
				from_x = 0 ; 
			else if( from_x >= cw ) 
				from_x = cw - 1 ; 		  
			if( to_x < 0 ) 
				to_x = 0 ; 
			else if( to_x >= cw ) 
				to_x = cw - 1 ; 		  
		}else if( to_x == from_x ) 
		{
			if( to_x < 0 || to_x >= ch ) 
				return ; 
			if( from_y < 0 ) 
				from_y = 0 ; 
			else if( from_y >= ch ) 
				from_y = ch - 1 ; 		  
			if( to_y < 0 ) 
				to_y = 0 ; 
			else if( to_y >= ch ) 
				to_y = ch - 1 ; 		  
		}else
		{			 
			int k = (to_x - from_x)/(to_y - from_y);
			int x0 = from_x ; 
			int y0 = from_y ;

			if( (from_x < 0 && to_x < 0 ) || (from_y < 0 && to_y < 0 ) ||
				(from_x >= cw && to_x >= cw ) || (from_y >= ch && to_y >= ch ))
				return ;
			
			if( !clip_line( k, x0, y0, cw, ch, &from_x, &from_y ) ) 
				return ; 	 
			if( !clip_line( k, x0, y0, cw, ch, &to_x, &to_y ) ) 
				return ; 	 
		}			
		if( from_x != to_x || from_y != to_y ) 
			func( ctx, from_x, from_y, to_x, to_y );
	}	 
} 

/*************************************************************************/
/* Path primitives : *****************************************************/
/*************************************************************************/
void
asim_move_to( ASDrawContext *ctx, int dst_x, int dst_y )
{
	if( ctx ) 
	{
		ctx->curr_x = dst_x ; 	
		ctx->curr_y = dst_y ; 
	}		 
}
	   
void
asim_line_to( ASDrawContext *ctx, int dst_x, int dst_y ) 
{
	asim_line_to_generic( ctx, dst_x, dst_y, ctx_draw_line_solid);	
}	 


void
asim_line_to_aa( ASDrawContext *ctx, int dst_x, int dst_y ) 
{
	asim_line_to_generic( ctx, dst_x, dst_y, ctx_draw_line_solid_aa);	 		
}	 

void
asim_cube_bezier( ASDrawContext *ctx, int x1, int y1, int x2, int y2, int x3, int y3 ) 
{
	if( ctx ) 
	{	
		int x0 = ctx->curr_x;
		int y0 = ctx->curr_y;
		Bool path_started = False;
			
		if (get_flags (ctx->flags, ASDrawCTX_CanvasIsARGB))
			path_started = asim_start_path (ctx);
	
		asim_move_to( ctx, x3, y3 );
		ctx_draw_bezier( ctx, x0<<8, y0<<8, x1<<8, y1<<8, x2<<8, y2<<8, x3<<8, y3<<8 );
	
		if (path_started)
			asim_apply_path( ctx, 0, 0, False, 0, 0, 0);
	}		
}

/*************************************************************************/
/* misc auxilary stuff : *************************************************/
/*************************************************************************/
/* Sinus lookup table */
const signed int ASIM_SIN[91]=
{
	0x00000000,
	0x00000478,0x000008EF,0x00000D66,0x000011DC,0x00001650,0x00001AC2,0x00001F33,0x000023A1,0x0000280C,0x00002C74,
	0x000030D9,0x0000353A,0x00003996,0x00003DEF,0x00004242,0x00004690,0x00004AD9,0x00004F1C,0x00005358,0x0000578F,
	0x00005BBE,0x00005FE6,0x00006407,0x00006820,0x00006C31,0x00007039,0x00007439,0x0000782F,0x00007C1C,0x00008000,
	
	0x000083DA,0x000087A9,0x00008B6D,0x00008F27,0x000092D6,0x00009679,0x00009A11,0x00009D9C,0x0000A11B,0x0000A48E,
	0x0000A7F3,0x0000AB4C,0x0000AE97,0x0000B1D5,0x0000B505,0x0000B827,0x0000BB3A,0x0000BE3F,0x0000C135,0x0000C41B,
	0x0000C6F3,0x0000C9BB,0x0000CC73,0x0000CF1C,0x0000D1B4,0x0000D43C,0x0000D6B3,0x0000D91A,0x0000DB6F,0x0000DDB4,
	0x0000DFE7,0x0000E209,0x0000E419,0x0000E617,0x0000E804,0x0000E9DE,0x0000EBA6,0x0000ED5C,0x0000EEFF,0x0000F090,
	0x0000F20E,0x0000F378,0x0000F4D0,0x0000F615,0x0000F747,0x0000F865,0x0000F970,0x0000FA68,0x0000FB4C,0x0000FC1C,
	0x0000FCD9,0x0000FD82,0x0000FE18,0x0000FE99,0x0000FF07,0x0000FF60,0x0000FFA6,0x0000FFD8,0x0000FFF6,0x00010000
};

static inline int asim_sin( int angle )
{
	while( angle >= 360 ) 
		angle -= 360 ;
	while( angle < 0 ) 
		angle += 360 ;
	if( angle <= 90 ) 
		return ASIM_SIN[angle];
	if( angle <= 180 ) 
		return ASIM_SIN[180-angle];
	if( angle <= 270 ) 
		return -ASIM_SIN[angle-180];
	return -ASIM_SIN[360-angle];
}	 

int 
asim_sqrt( double sval ) 
{
#ifdef HAVE_LONG_LONG
	Long64_t uval = (sval >= 0) ? (Long64_t)sval:-(Long64_t)sval ;
	Long64_t res = uval ; 
	Long64_t t = res*res ;
#else
   long uval = (sval >= 0) ? (long)sval:-(long)sval ;
   long res = uval ; 
   long t = res*res ;
#endif
	
	while( t > uval ) 
	{
		res = res >> 1 ; 
		t = t >> 2 ;	  
	}	 
	if( t == uval ) 
		return (int)res;
	res = (res << 1) + 1 ;
	t = res*res ;
	while( t > uval ) 
	{
		t -= (res<<1)-1 ;
		--res ;
	}		  
	return (int)res;
}	  

/*************************************************************************/
/* some standard closed paths :                                          */
/*************************************************************************/
void
asim_straight_ellips( ASDrawContext *ctx, int x, int y, int rx, int ry, Bool fill ) 
{
	if( ctx && rx > 0 && ry > 0 &&
		x + rx >= 0 && y+ry >= 0 && 
		x - rx  < ctx->canvas_width && y - ry < ctx->canvas_height ) 
	{	 
		int max_y = ry ; 
		int orig_x = x, orig_y = y, orig_rx = rx ; 
#ifdef HAVE_LONG_LONG						   
		Long64_t rx2 = rx*rx, ry2 = ry * ry, d ; 
#else
		long rx2 = rx*rx, ry2 = ry * ry, d ; 
#endif		

#if 1
		if (y + ry  > ctx->canvas_height && y - ry  < 0) 
			max_y = max (y, ctx->canvas_height - y); 
#endif			
#if 0
		if( fill ) 
		{
			long y1 = 0; 
			long x1 = rx-1;
			long ty = rx*rx ;
			long tx = x1*x1 ;
			
			do
			{
				while( tx > ty && x1 > 0 )
				{
					tx -= (x1<<1)+1 ;
					--x1 ;
				}
				CTX_FILL_HLINE(ctx,x-x1,y+y1,x+x1,255);
				CTX_FILL_HLINE(ctx,x-x1,y-y1,x+x1,255);
				
				d = (y1<<1)+1 ;
				/* this is likely to break for large radii when there are no 64 bit ints */
				if( rx != ry )
					d = (d*rx2)/ry2 ;
				ty -= (long)d;
			
			}while( ++y1 < max_y ); 
		}
#endif

		asim_start_path( ctx );
		asim_move_to( ctx, x+rx, y );
		LOCAL_DEBUG_OUT( "x = %d, y = %d, rx = %d, ry = %d", x, y, rx, ry );
/* if no 64 bit integers - then tough luck - have to resort to beziers */
#ifndef HAVE_LONG_LONG						   
		if( (rx == ry && (rx > 16000 || ry > 16000 || x < -16000 || y < -16000 || x > 16000 || y > 16000)) ||
			(rx != ry && (rx > 1000 || ry > 1000 || x > 1000 || y > 1000 || x < -1000 || y < -1000)) ) 
		{  /* somewhat imprecise approximation using 4 bezier curves */
			int drx = rx*142 ;       /* that gives us precision of about 0.05% which is 
									* pretty good for integer math */
			int dry = ry*142 ;
			LOCAL_DEBUG_OUT( "drx = %d, dry = %d", drx, dry );
			rx  = rx << 8 ; 
			ry  = ry << 8 ; 
			x  = x << 8 ; 
			y  = y << 8 ; 
			ctx_draw_bezier( ctx, x+rx, y, x+rx, y+dry, x+drx, y+ry, x, y+ry );
			ctx_draw_bezier( ctx, x, y+ry, x-drx, y+ry, x-rx, y+dry, x-rx, y );
			ctx_draw_bezier( ctx, x-rx, y, x-rx, y-dry, x-drx, y-ry, x, y-ry );
			ctx_draw_bezier( ctx, x, y-ry, x+drx, y-ry, x+rx, y-dry, x+rx, y );

	 	}else
#endif			
		{		  
			x = x<<4 ; 
			y = y<<4 ; 
			rx = rx<<4 ;
			ry = ry<<4 ;
			max_y = (max_y << 4) + 4;

			{
				long min_r = rx - 1, max_r = rx + 1 ; 
				long y1 = 0; 
				long x1 = max_r;
				long min_ty = min_r * min_r ;
				long max_ty = max_r * max_r ;
				long tx = max_ty ;
			
				do
				{
					long start_tx, start_x1 ;
					while( tx > max_ty && x1 > 0 )
					{
						--x1; 
						tx -= (x1<<1)+1 ;
					}	 
					start_tx = tx ; 
					start_x1 = x1 ;

					
					while( tx > min_ty && x1 >= 0) 
					{
						render_supersampled_pixel( ctx, (x-x1)<<4, (y+y1)<<4 );
						render_supersampled_pixel( ctx, (x-x1)<<4, (y-y1)<<4 );
						render_supersampled_pixel( ctx, (x+x1)<<4, (y+y1)<<4 );
						render_supersampled_pixel( ctx, (x+x1)<<4, (y-y1)<<4 );
						--x1; 
						tx -= (x1<<1)+1 ;
					}
					tx = start_tx ; 
					x1 = start_x1 ;
					d = ((y1<<1)+1);
					if( rx != ry )
						d = (d*rx2)/ry2 ;
					min_ty -= (long)d;
					max_ty -= (long)d;
				}while( ++y1 <= max_y ); 
			}
		}		
		asim_apply_path( ctx, orig_x+orig_rx, orig_y, fill, orig_x, orig_y, CTX_ELLIPS_FILL_THRESHOLD );
	}		
}	 


void
asim_circle( ASDrawContext *ctx, int x, int y, int r, Bool fill ) 
{
	asim_straight_ellips( ctx, x, y, r, r, fill );
}	 


void
asim_ellips( ASDrawContext *ctx, int x, int y, int rx, int ry, int angle, Bool fill ) 
{
	while( angle >= 360 ) 
		angle -= 360 ;
	while( angle < 0 ) 
		angle += 360 ;
	
	if( angle == 0 || angle ==180  || rx == ry ) 
	{	
		asim_straight_ellips( ctx, x, y, rx, ry, False );
		if( angle == 180 ) 
			asim_move_to( ctx, x-rx, y );
		return;
	}
	if( angle == 90 || angle == 270 ) 
	{	
		asim_straight_ellips( ctx, x, y, ry, rx, False );
		asim_move_to( ctx, x, y + (angle == 90?-rx:rx) );
		return;
	}

	if( ctx && rx > 0 && ry > 0 ) 
	{	
		int dx0 = rx, dy0 = ry, dx1 = 0, dy1 = rx*4/3 ;
		int x0, y0, x1down, y1down, x2down, y2down, x3, y3, x2up, y2up, x1up, y1up ; 
		int ry4 = (ry<<2)/3 ;
		int sin_val = asim_sin(angle);
		int cos_val = asim_sin(angle+90);
		if(sin_val < 0) 
			sin_val = -sin_val ;
		if(cos_val < 0) 
			cos_val = -cos_val ;
		dx0 = (rx*cos_val)>>8; 
		dy0 = (rx*sin_val)>>8; 
		dx1 = (ry4*sin_val)>>8;
		dy1 = (ry4*cos_val)>>8; 
		if( angle < 180 )
		{
			dy0 = -dy0 ;
			dx1 = -dx1 ; 
		}	 
		if( angle > 90 && angle < 270 )
		{	
			dx0 = -dx0 ; 
			dy1 = -dy1 ;
		}
		x = x << 8;
		y = y << 8;
		x0 = x + dx0 ;
		y0 = y + dy0 ;
		x3 = x - dx0 ;
		y3 = y - dy0 ; 
		x1down = x0 + dx1 ;
		y1down = y0 - dy1 ;
		x2down = x3 + dx1 ; 
		y2down = y3 - dy1 ;  
		x2up = x3 - dx1 ; 
		y2up = y3 + dy1 ;  
		x1up = x0 - dx1 ;  
		y1up = y0 + dy1 ;  
		

		asim_start_path( ctx );
		asim_move_to( ctx, x0>>8, y0>>8 );
		ctx_draw_bezier( ctx, x0, y0, x1down, y1down, x2down, y2down, x3, y3 );
		ctx_draw_bezier( ctx, x3, y3, x2up, y2up, x1up, y1up, x0, y0 );
		asim_apply_path( ctx, x0>>8, y0>>8, fill, x, y, CTX_ELLIPS_FILL_THRESHOLD );
	}		
}	 

void
asim_ellips2( ASDrawContext *ctx, int x, int y, int rx, int ry, int angle, Bool fill ) 
{
	Bool direction = 1 ;

	while( angle >= 360 ) 
		angle -= 360 ;
	while( angle < 0 ) 
		angle += 360 ;
	
	if( angle == 0 || angle ==180  || rx == ry ) 
	{	
		asim_straight_ellips( ctx, x, y, rx, ry, fill );
		if( angle == 180 ) 
			asim_move_to( ctx, x-rx, y );
		return;
	}
	if( angle == 90 || angle == 270 ) 
	{	
		asim_straight_ellips( ctx, x, y, ry, rx, fill );
		asim_move_to( ctx, x, y + (angle == 90?-rx:rx) );
		return;
	}

	if( angle > 180 ) 
		angle -= 180 ; 
	if( angle > 90 ) 
	{	
		angle = 180-angle ; 
		direction = -1 ;
	}


	if( ctx && rx > 0 && ry > 0 ) 
	{	
		int sin90 = 0x00010000 ;
		double sin_val = (double)asim_sin(angle)/(double)sin90;
		double cos_val = (double)asim_sin(angle+90)/(double)sin90;
		double rx2 = rx*rx ; 
		double ry2 = ry*ry ;
		double c2 = rx2 - ry2 ; 
		double xc2 = (c2 * cos_val)*cos_val ;
		double yc2 = (c2 * sin_val)*sin_val ;
		double A = rx2 - xc2 ;
		double B = rx2 - yc2 ;
		double C = -c2*sin_val*cos_val ;
		double F = -rx2*ry2 ;
		int yt = asim_sqrt(A);
		int xt = (int)-C/yt ;
		int xr = asim_sqrt(B);
		int yr = (int)-C/xr ;
		int x1 ; 
		int x2 ; 
		int line = yt; 
		int last_med_dd1 = 0 ; 
		int last_med_dd2 = 0 ; 
		int y1 ;
		double A2 = 2.*A ;
		double BB; 
		double CC = C*(double)((line<<1)-1);


		xt = (int)((A-CC)/A2)  ;
		y1 = line*direction ;
		BB = B*(double)line*(double)line + F - B*(double)line - (B/4.);
		x1 = xt+1 ;
		x2 = xt-1 ;
		--yr ; 

		while( line >= -1 ) 
		{
			double d ; 
			int dx1 = 0, dx2 = 0 ;
			d = A*(double)x1*(double)x1 + BB +CC*(double)x1;
#ifdef DEBUG_ELLIPS					 						   
			fprintf( stderr, "line = %d, d1 = %f", y-line, d ); 
#endif
			if( d < 0 ) 
			{       
				int aa = (int)((A-A2*(double)x1-CC)*255/A2);
				int dd = (int)(-aa-(double)(d*255/A2)) ; 
				int med_dd = (dd>>1)+1; 
	
				if( last_med_dd1 > 0 ) 
				{ /* this prevents suddent jags on the line and makes it as 
				   * smooth as it gets  */
					med_dd = (2*med_dd+last_med_dd1)/3 ;
				}
				++dx1 ;
#ifdef DEBUG_ELLIPS					 						   
				fprintf( stderr, " -> d1 = %f, dx1 = %d, aa = %d, dd = %d, med_dd = %d\n", d, dx1, aa, dd, med_dd ); 
#endif
				if( med_dd+aa > dd ) 
				{
					int v = (dd-med_dd)*255/(aa+255) ;	
					CTX_PUT_PIXEL( ctx, x+(x1-1), y-y1, 255-v ) ;
					CTX_PUT_PIXEL( ctx, x-(x1-1), y+y1, 255-v ) ;
					CTX_PUT_PIXEL( ctx, x+(x1-2), y-y1, v ) ;
					CTX_PUT_PIXEL( ctx, x-(x1-2), y+y1, v ) ;
					dx1 = 2 ;
#ifdef DEBUG_ELLIPS					 						   
					fprintf( stderr, "\t\t dx1 = %d, v = %d\n", dx1, v ); 
#endif
				}else	 
					while( dd > -(aa>>1) ) 
					{ 	
						int v ; 
						v = ( dd >= med_dd )? dd-med_dd: med_dd-dd ;  
						v = v*255/med_dd ; 
						if( v < 0 || v > 255) 
							v = 250 ;

						CTX_PUT_PIXEL( ctx, x+(x1-dx1), y-y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x-(x1-dx1), y+y1, 255-v ) ;
						if( x1 >= xt && dd > med_dd ) 
						{	
							CTX_PUT_PIXEL( ctx, x+(x1-dx1), y-(y1+direction), v ) ;
							CTX_PUT_PIXEL( ctx, x-(x1-dx1), y+(y1+direction), v ) ;
						}
						++dx1 ; 
#ifdef DEBUG_ELLIPS					 						   
						fprintf( stderr, "\t\t dx1 = %d, aa = %d, dd = %d, med_dd = %d, v = %d, x = %d\n", dx1, aa, dd, med_dd, v, x+(x1-dx1) ); 
#endif
						aa += 255 ;
						dd -= aa ; 
					}
				x1 -= (dx1>>1)-1 ;
				last_med_dd1 = med_dd ; 
			}

			d = A*(double)(x2+1)*(double)(x2+1) + BB +CC*(double)(x2+1) ;
			if( line > yr ) 
			{
#ifdef DEBUG_ELLIPS					 						   
				fprintf( stderr, "line = %d, d2 = %f\n", y-line, d ); 
#endif
			 	if( d < 0 ) 	
				{
					int aa = (int)((A+A2*(double)x2+CC)*255/A2);
					int dd = (int)(aa+(double)(d*255/A2))  ; 
					int med_dd = dd/2 - 1; 

					if( last_med_dd2 > 0 ) 
					{ /* this prevents suddent jags on the line and makes it as 
					   * smooth as it gets  */
						med_dd = (med_dd*2+last_med_dd2)/3 ;
					}
					dx2 = 1;
#ifdef DEBUG_ELLIPS					 						   
					fprintf( stderr, " -> d2 = %f, x2 = %d, dx2 = %d, aa = %d, dd = %d, med_dd = %d\n", d, x2, dx2, aa, dd, med_dd ); 
#endif
					if( med_dd-aa < dd ) 
					{
						int v = (med_dd-dd)*255/(aa+255) ;	
						CTX_PUT_PIXEL( ctx, x+(x2+2), y-y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2+2), y+y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x+(x2+3), y-y1, v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2+3), y+y1, v ) ;
						dx2 = 2 ;
#ifdef DEBUG_ELLIPS					 						   
						fprintf( stderr, "\t\t dx2 = %d, v = %d\n", dx2, v ); 
#endif
					}else while( dd < aa/2 ) 
					{ 
						int v ; 
						v = ( dd >= med_dd )? dd-med_dd: med_dd-dd ;  
						v = v*255/-med_dd ; 
						if( v < 0 || v > 255) 
							v = 250 ;
#ifdef DEBUG_ELLIPS					 						   
						fprintf( stderr, "\t\t dx2 = %d, aa = %d, dd = %d, med_dd = %d, v = %d, x = %d\n", dx2, aa, dd, med_dd, v, x+(x2+dx2) ); 						
#endif
						++dx2; 
						CTX_PUT_PIXEL( ctx, x+(x2+dx2), y-y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2+dx2), y+y1, 255-v ) ;
						if( x2 <= xt && dd < med_dd ) 
						{	
							CTX_PUT_PIXEL( ctx, x+(x2+dx2), y-(y1+direction), v ) ;
							CTX_PUT_PIXEL( ctx, x-(x2+dx2), y+(y1+direction), v ) ;
						}
						aa += 255 ;
						dd += aa ; 
					}
					x2 += (dx2>>1)-1 ;
					last_med_dd2 = med_dd ; 
				}	 
			}else if( line < yr ) 
			{
#ifdef DEBUG_ELLIPS					 						   
				fprintf( stderr, "line = %d, BELOW: d2 = %f\n", y-line, d ); 
#endif
				if( d > 0. )
				{	
					int aa = (int)((A-A2*(double)(x2)-CC)*255./A2);
					int dd = (int)(aa+(double)(d*255./A2))  ; 
					int med_dd = dd/2+1; 
					
					if( last_med_dd2 > 0 ) 
						med_dd = (med_dd*2+last_med_dd2)/3 ;

					dx2 = -1 ;
#ifdef DEBUG_ELLIPS					 						   
					fprintf( stderr, " ->BELOW: d2 = %f, dx2 = %d, aa = %d, dd = %d, med_dd = %d\n", d, dx2, aa, dd, med_dd ); 					
#endif
					if( med_dd-aa > dd ) 
					{
						int v = (dd-med_dd)*255/(-aa+255) ;	
#ifdef DEBUG_ELLIPS					 						   
						fprintf( stderr, "\t\t BELOW: dx2 = %d, v = %d\n", dx2, v ); 
#endif
						CTX_PUT_PIXEL( ctx, x+(x2-1), y-y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2-1), y+y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x+(x2-2), y-y1, v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2-2), y+y1, v ) ;
						dx2 = -2 ;
					}else while( dd > aa/2 )	
					{	
						int v ; 
						v = ( dd >= med_dd )? dd-med_dd: med_dd-dd ;  
						v = v*255/med_dd ; 
						if( v < 0 || v > 255) 
							v = 250 ;
#ifdef DEBUG_ELLIPS					 						
						fprintf( stderr, "\t\tBELOW:  dx2 = %d, aa = %d, dd = %d, med_dd = %d, v = %d, x = %d\n", dx2, aa, dd, med_dd, v, x+(x2+dx2) ); 						   
#endif
						CTX_PUT_PIXEL( ctx, x+(x2+dx2), y-y1, 255-v ) ;
						CTX_PUT_PIXEL( ctx, x-(x2+dx2), y+y1, 255-v ) ;
						--dx2; 
						aa += 255 ;
						dd += aa ; 
					}
					x2 += (dx2/2)+1 ;
					last_med_dd2 = med_dd ; 
				}			
			}else
			{	
    			x2 = xr+2 ; 	
#ifdef DEBUG_ELLIPS					 
				fprintf( stderr, "\t\t BELOW: x2 = %d, xr = %d\n", x2,xr); 
#endif
				CTX_PUT_PIXEL( ctx, x+xr, y-y1, 255 ) ;
				CTX_PUT_PIXEL( ctx, x-xr, y+y1, 255 ) ;
				last_med_dd2 = 0 ;
			}
#ifdef DEBUG_ELLIPS					 
			fprintf( stderr, "dx1 = %d, x1 = %d, dx2 = %d, x2 = %d\n", dx1, x1, dx2, x2 ); 
#endif
			if( fill ) 
			{	
				CTX_FILL_HLINE(ctx,x+(x1-2),y-y1,x+x2-1,255);
				CTX_FILL_HLINE(ctx,x-x2-1,y+y1,x-(x1-2),255);
			}
			
			CC -= 2.*C ;
			BB -= B*(double)((line-1)<<1) ;
			y1 -= direction ;
			--line ; 
		}	 
	}		
}	 

/*************************************************************************/
/* Flood fill functionality **********************************************/
/*************************************************************************/

void 
asim_flood_fill( ASDrawContext *ctx, int x, int y, CARD32 min_val, CARD32 max_val  ) 
{
	if( ctx && x >= 0 && x < ctx->canvas_width && y >= 0 && y < ctx->canvas_height )
	{
		int x0 = x, x1 = x ; 
		int cw = ctx->canvas_width ;
		CARD32 *data = CTX_SELECT_CANVAS(ctx) + y*cw ; 
		while( x0 >= 0 && data[x0] <= max_val && data[x0]  >= min_val ) --x0;
		++x0 ;
		while( x1 < cw && data[x1] <= max_val && data[x1]  >= min_val ) ++x1;
		--x1 ; 
		LOCAL_DEBUG_OUT( "x = %d, y = %d, data[x] = 0x%X, x0 = %d, x1 = %d", x, y, data[x], x0, x1 );
		
		if( x0 <= x1 ) 
			ctx_flood_fill( ctx, x0, y, x1, min_val, max_val );			
	}		   
}	 

void
asim_rectangle( ASDrawContext *ctx, int x, int y, int width, int height ) 
{
	asim_move_to( ctx, x, y );
	asim_line_to_generic( ctx, x+width, y, ctx_draw_line_solid);	 		
	asim_line_to_generic( ctx, x+width, y+height, ctx_draw_line_solid);	 		   
	asim_line_to_generic( ctx, x, y+height, ctx_draw_line_solid);	 		   
	asim_line_to_generic( ctx, x, y, ctx_draw_line_solid);	 		   
}	 




/*********************************************************************************/
/* The end !!!! 																 */
/*********************************************************************************/
/*********************************************************************************/
/* Test container : 																 */
/*********************************************************************************/

#ifdef TEST_ASDRAW
#include "afterimage.h"

/*************************************/
/* testing code for ROOT from CERN : */
typedef int Int_t;
typedef unsigned int UInt_t;
typedef int Bool_t;

static ASDrawContext *create_draw_context_argb32(ASImage *im, ASDrawTool *brush)
{
   // Create draw context

   ASDrawContext *ctx = safecalloc (1, sizeof(ASDrawContext));

   ctx->canvas_width = im->width;
   ctx->canvas_height = im->height;
   ctx->canvas = im->alt.argb32;
   ctx->scratch_canvas = 0;

   ctx->flags = ASDrawCTX_CanvasIsARGB;
   asim_set_custom_brush_colored( ctx, brush);
   return ctx;
}


ASImage *TASImage_DrawCircle(ASVisual *asv, ASImage *fImage, Int_t x, Int_t y, Int_t r, const char *col, Int_t thick)
{
   // Draw circle. If thick < 0 - draw filled circle

   thick = !thick ? 1 : thick;
   Int_t sz = thick*thick;
   CARD32 *matrix;
   int i;
   ASImage *img = 0;
   ASDrawContext *ctx;

   ARGB32 color;
   parse_argb_color(col, &color);
   ASDrawTool brush;

   matrix = malloc (sizeof(CARD32)*sz);

   for (i = 0; i < sz; i++) {
      matrix[i] = (CARD32)color;
   }

   brush.matrix = matrix;
   brush.height = brush.width = thick > 0 ? thick : 1;
   brush.center_y = brush.center_x = thick > 0 ? thick/2 : 0;

   img = tile_asimage(asv, fImage, 0, 0, fImage->width, fImage->height,
                            0, ASA_ARGB32, 0, ASIMAGE_QUALITY_DEFAULT);
   
   ctx = create_draw_context_argb32(img, &brush);

   asim_circle(ctx, x,  y, r, thick < 0);
   asim_cube_bezier(ctx, x+50, y+50, x+100, y+100, x+50, y+100);
   asim_move_to(ctx, x, y+200);
   asim_line_to_aa(ctx, x+200, y+100);
   asim_line_to_aa(ctx, x+200, y+200);
   asim_line_to_aa(ctx, x, y+200);

   free (matrix);
//   destroy_asdraw_context32(ctx);

	return img;
}

// ROOT testing end
/* testing code for ROOT from CERN   */
/*************************************/

int main(int argc, char **argv )
{
	ASVisual *asv ;
	ASImage *back = NULL ;
	ASImage *drawing1 = NULL ;
	ASImage *merged_im = NULL ;
	ASImageLayer *layers ;
	int layers_num = 0, i;
	int screen = 0, depth = 0;

	ASDrawContext *ctx ;

#define DRAW_TEST_SIZE 800	
	
	set_output_threshold( 10 );

#ifndef X_DISPLAY_MISSING
	dpy = XOpenDisplay(NULL);
	screen = DefaultScreen(dpy);
	depth = DefaultDepth( dpy, screen );
#endif

	asv = create_asvisual( dpy, screen, depth, NULL );


	back = create_asimage( DRAW_TEST_SIZE, DRAW_TEST_SIZE, 100 );
	back->back_color = ARGB32_White ;
	drawing1 = create_asimage( DRAW_TEST_SIZE, DRAW_TEST_SIZE, 100 );
	drawing1->back_color = ARGB32_Black ;


	ctx = create_asdraw_context(DRAW_TEST_SIZE, DRAW_TEST_SIZE);
	/* actuall drawing starts here */
#if 1
/*	for( i = 0 ; i < 50000 ; ++i ) */
#if 1
	asim_move_to( ctx, 0, 0 ); 
	asim_line_to_aa( ctx, 200, 200 ); 
	asim_line_to_aa( ctx, 100, 10 );
	asim_line_to_aa( ctx, 10, 300 );
	asim_line_to_aa( ctx, 15, 400 );
	asim_move_to(ctx, 15, 420);  
	asim_line_to_aa( ctx, 200, 410 );
	asim_line_to_aa( ctx, 400, 440 );
		
	asim_move_to(ctx, 10, 0); 	  
	asim_set_brush( ctx, 1 ); 
	asim_line_to_aa( ctx, 210, 200 ); 
	asim_line_to_aa( ctx, 110, 10 );
	asim_line_to_aa( ctx, 20, 300 );
	asim_line_to_aa( ctx, 25, 400 );
	asim_move_to(ctx, 15, 430);  
	asim_line_to_aa( ctx, 200, 420 );
	asim_line_to_aa( ctx, 400, 450 );

	asim_move_to(ctx, 20, 0); 	  
	asim_set_brush( ctx, 2 ); 
	asim_line_to_aa( ctx, 220, 200 ); 
	asim_line_to_aa( ctx, 120, 10 );

#endif
	asim_move_to(ctx, 120, 10); 	  
	asim_set_brush( ctx, 2 ); 

	asim_line_to_aa( ctx, 30, 300 );
#if 1
	asim_line_to_aa( ctx, 35, 400 );
	asim_move_to(ctx, 15, 440);    
	asim_line_to_aa( ctx, 200, 430 );
	asim_line_to_aa( ctx, 400, 460 );

    /****************************************/
	/* non-antialiased : */

	asim_move_to(ctx,  200, 0 ); 

	asim_set_brush( ctx, 0 ); 
	asim_line_to( ctx, 400, 200 ); 
	asim_line_to( ctx, 300, 10 );
	asim_line_to( ctx, 210, 300 );
	asim_line_to( ctx, 215, 400 );
	asim_move_to(ctx, 15, 450);  
	asim_line_to( ctx, 200, 440 );
	asim_line_to( ctx, 400, 470 );
		
	asim_move_to(ctx, 210, 0); 	  
	asim_set_brush( ctx, 1 ); 
	asim_line_to( ctx, 410, 200 ); 
	asim_line_to( ctx, 310, 10 );
	asim_line_to( ctx, 220, 300 );
	asim_line_to( ctx, 225, 400 );
	asim_move_to(ctx, 15, 460);  
	asim_line_to( ctx, 200, 450 );
	asim_line_to( ctx, 400, 480 );

	asim_move_to(ctx, 220, 0); 	  
	asim_set_brush( ctx, 2 ); 
	asim_line_to( ctx, 420, 200 ); 
	asim_line_to( ctx, 320, 10 );
	asim_line_to( ctx, 230, 300 );
	asim_line_to( ctx, 235, 400 );
	asim_move_to(ctx, 15, 470);    
	asim_line_to( ctx, 200, 460 );
	asim_line_to( ctx, 400, 490 );

	asim_set_brush( ctx, 0 ); 
	asim_circle( ctx, 600, 110, 80, False );
	asim_circle( ctx, 600, 110, 50, False );
	asim_circle( ctx, 600, 110, 20, False );
	asim_circle( ctx, 600, 110, 1, False );

	asim_set_brush( ctx, 1 ); 
	asim_circle( ctx, 600, 110, 90, False );
	asim_circle( ctx, 600, 110, 60, False );
	asim_circle( ctx, 600, 110, 30, False );
	asim_circle( ctx, 600, 110, 5, False );
	
	asim_set_brush( ctx, 2 ); 
	asim_circle( ctx, 600, 110, 100, False );
	asim_circle( ctx, 600, 110, 70, False );
	asim_circle( ctx, 600, 110, 40, False );
	asim_circle( ctx, 600, 110, 10, False );

	asim_set_brush( ctx, 0 ); 
/*	asim_circle( ctx, -1000, -1000, 2000, False );
	asim_ellips( ctx, -1000, -1000, 2000, 500, -45, False );
 */
	asim_circle( ctx, 595, 550, 200, False );
	i = 30 ;

	{
/*		int k;
		for( k = 0 ; k < 100 ; ++k )  */
		for( i = 3 ; i < 180 ; i+=5 ) 
		{	
/*			asim_ellips2( ctx, 595, 550, 198, 40, i, False );
			asim_ellips( ctx, 595, 540, 198, 40, i, False );
			asim_ellips2( ctx, 595, 550, 198, 40, i+30, False );
			asim_ellips( ctx, 595, 540, 198, 40, i+30, False );
 */
			asim_ellips2( ctx, 595, 550, 198, 60, i, False );
/*			asim_ellips( ctx, 585, 550, 198, 40, i+50, False );
			asim_ellips( ctx, 595, 550, 198, 40, i, False ); 
 */
		}
	}
	asim_circle( ctx, 705, 275, 90, True );
//	asim_circle( ctx, 100, 100, 90, True );

	asim_circle( ctx, -40000, 500, 40500, False );
	asim_circle( ctx, -10000, 500, 10499, False );

	asim_ellips2( ctx, -1000, 500, 1500, 100, 2, False );

	asim_flood_fill( ctx, 664, 166, 0, 126 ); 
	asim_flood_fill( ctx, 670, 77, 0, 126 ); 
	asim_flood_fill( ctx, 120, 80, 0, 126 ); 
	asim_flood_fill( ctx, 300, 460, 0, 126 ); 

#endif
#endif
#if 1
	/* commit drawing : */
	apply_asdraw_context( drawing1, ctx, SCL_DO_ALPHA ); 
	
	layers_num = 2 ;
	layers = create_image_layers( layers_num );
	
	layers[0].im = back ;
	layers[1].im = drawing1 ;
	layers[0].clip_width = DRAW_TEST_SIZE ;
	layers[0].clip_height = DRAW_TEST_SIZE ;
	layers[1].clip_width = DRAW_TEST_SIZE ;
	layers[1].clip_height = DRAW_TEST_SIZE ;
		
	merged_im = merge_layers( asv, layers, layers_num,
		                      DRAW_TEST_SIZE, DRAW_TEST_SIZE,
							  ASA_ASImage,
							  0, ASIMAGE_QUALITY_DEFAULT );

#if 1
	{
		ASImage *overlayed_im = TASImage_DrawCircle(asv, merged_im, 84, 299, 80, "blue", -1);
		ASImage2file( overlayed_im, NULL, "test_asdraw.overlayed.png", ASIT_Png, NULL );
		destroy_asimage( &overlayed_im );
	}
#endif	
	ASImage2file( merged_im, NULL, "test_asdraw.merged.png", ASIT_Png, NULL );
	destroy_asimage( &merged_im );
#endif

	return 0;
}
	

#endif

