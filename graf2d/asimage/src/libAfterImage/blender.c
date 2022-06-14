/*
 * Copyright (c) 2000,2001 Sasha Vasko <sasha at aftercode.net>
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

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

/*#define LOCAL_DEBUG*/
/*#define DO_CLOCKING*/

#ifdef HAVE_MMX
#include <mmintrin.h>
#endif

#include <ctype.h>
#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asvisual.h"
#include "scanline.h"
#include "blender.h"

/*********************************************************************************/
/* colorspace conversion functions : 											 */
/*********************************************************************************/

CARD32
rgb2value( CARD32 red, CARD32 green, CARD32 blue )
{
	if( red > green )
		return MAX(red,blue);
	return MAX(green, blue);
}

CARD32
rgb2saturation( CARD32 red, CARD32 green, CARD32 blue )
{
	register int max_val, min_val ;
	if( red > green )
	{
		max_val = MAX(red,blue);
		min_val = MIN(green,blue);
	}else
	{
		max_val = MAX(green, blue) ;
		min_val = MIN(red,blue) ;
	}
	return max_val > 1 ? ((max_val - min_val)<<15)/(max_val>>1) : 0;
}


/* Traditionally Hue is represented by 360 degree circle.
  For our needs we use 255 degree circle instead.

  Now the circle is separated into 6 segments :
  Trad:		Us:				Color range:    Red:	Green:	Blue:
  0-60		0    -42.5 		red-yellow      FF-7F   0 -7F   0 -0
  60-120    42.5 -85 		yellow-green    7F-0    7F-FF   0 -0
  120-180   85   -127.5 	green-cyan      0 -0    FF-7F   0 -7F
  180-240   127.5-170   	cyan-blue       0 -0    7F-0    7F-FF
  240-300   170  -212.5     blue-magenta    0-7F    0 -0	FF-7F
  300-360   212.5-255       magenta-red     7F-FF   0 -0    7F-0

  As seen from above in each segment at least one of the RGB values is 0.
  To achieve that we find minimum of R, G and b and substract it from all,
  and then multiply values by ratio to bring it into range :

  new_val = ((val - min_val)*0xFEFF)/(max_val-min_val)
  (note that we use 16bit values, instead of 8 bit as above)

  WE store hue in 16 bits, so instead of above value of 42.5 per segment
  we should use 85<<7 == 0x00002A80 per segment.

  When all the RGB values are the same - then hue is invalid and  = 0;
  To distinguish between hue == 0 when color is Red and invalid hue, we
  make all valid hues to fit in range 0x0001-0xFF00, with 0x0001 being RED.
*/

#define HUE_RED_TO_YELLOW		0
#define HUE_YELLOW_TO_GREEN		1
#define HUE_GREEN_TO_CYAN   	2
#define HUE_CYAN_TO_BLUE    	3
#define HUE_BLUE_TO_MAGENTA   	4
#define HUE_MAGENTA_TO_RED   	5

#define HUE16_RED				(HUE16_RANGE*HUE_RED_TO_YELLOW)
#define HUE16_YELLOW			(HUE16_RANGE*HUE_YELLOW_TO_GREEN)
#define HUE16_GREEN			   	(HUE16_RANGE*HUE_GREEN_TO_CYAN)
#define HUE16_CYAN		    	(HUE16_RANGE*HUE_CYAN_TO_BLUE)
#define HUE16_BLUE			   	(HUE16_RANGE*HUE_BLUE_TO_MAGENTA)
#define HUE16_MAGENTA		 	(HUE16_RANGE*HUE_MAGENTA_TO_RED)

#define MAKE_HUE16(hue,red,green,blue,min_val,max_val,delta) \
	do{	if( (red) == (max_val) ){ /* 300 to 60 degrees segment */ \
			if( (blue) <= (green) ){  /* 0-60 degrees segment*/    \
				(hue) = HUE16_RED    + (((green) - (blue)) * (HUE16_RANGE)) / (delta) ;\
				if( (hue) == 0 ) (hue) = MIN_HUE16 ; \
			}else {	               /* 300-0 degrees segment*/ \
				(hue) = HUE16_MAGENTA+ (((red)   - (blue)) * (HUE16_RANGE)) / (delta) ; \
				if( (hue) == 0 ) (hue) = MAX_HUE16 ;                                 \
			}                                                                   \
		}else if( (green) == (max_val) ){ /* 60 to 180 degrees segment */           \
			if( (blue) >= (red) )    /* 120-180 degrees segment*/                   \
				(hue) = HUE16_GREEN  + (((blue)-(red) ) * (HUE16_RANGE)) / (delta) ;    \
			else                 /* 60-120 degrees segment */                   \
				(hue) = HUE16_YELLOW + (((green)-(red)) * (HUE16_RANGE)) / (delta) ;    \
		}else if( (red) >= (green) )     /* 240 to 300 degrees segment */           \
			(hue)     = HUE16_BLUE   + (((red) -(green))* (HUE16_RANGE)) / (delta) ;    \
		else                        /* 180 to 240 degrees segment */            \
			(hue)     = HUE16_CYAN   + (((blue)-(green))* (HUE16_RANGE)) / (delta) ;    \
	}while(0)
#define INTERPRET_HUE16(hue,delta,max_val,red,green,blue)      \
	do{	int range = (hue)/HUE16_RANGE ;                                  \
		int min_val = (max_val) - (delta);                                \
		int mid_val = ((hue) - HUE16_RANGE*range)*(delta) / HUE16_RANGE ;  \
		switch( range )	{                                              \
			case HUE_RED_TO_YELLOW :    /* red was max, then green  */ \
				(red)   = (max_val); (green)=mid_val+(min_val); (blue) = (min_val); break; \
			case HUE_YELLOW_TO_GREEN :  /* green was max, then red */                      \
				(green) = (max_val); (red) =(max_val)-mid_val;  (blue) = (min_val); break; \
			case HUE_GREEN_TO_CYAN :    /* green was max, then blue*/                      \
				(green) = (max_val); (blue)= mid_val+(min_val);	(red)  = (min_val); break; \
			case HUE_CYAN_TO_BLUE :     /* blue was max, then green  */                    \
				(blue)  = (max_val); (green)=(max_val)-mid_val; (red)  = (min_val); break; \
			case HUE_BLUE_TO_MAGENTA :  /* blue was max, then red   */                     \
				(blue)  = (max_val); (red)  = mid_val+(min_val);(green)= (min_val); break; \
			case HUE_MAGENTA_TO_RED :   /* red was max, then blue */                       \
				(red)   = (max_val); (blue) = (max_val)-mid_val;(green)= (min_val); break; \
		}                                                                                  \
	}while(0)


int normalize_degrees_val( int degrees )
{
	while ( degrees < 0 ) degrees += 360 ;
	while ( degrees >= 360 ) degrees -= 360 ;
	return degrees;
}

CARD32
degrees2hue16( int degrees )
{
	CARD32 hue = 0 ;

	while ( degrees < 0 ) degrees += 360 ;
	while ( degrees >= 360 ) degrees -= 360 ;

	hue = degrees * HUE16_RANGE / 60 ;
	return (hue==0)?MIN_HUE16:hue ;
}

int
hue162degrees( CARD32 hue )
{
	if( hue < MIN_HUE16 || hue > MAX_HUE16 )
		return -1 ;

	return (hue*60)/HUE16_RANGE ;
}

CARD32
rgb2hue( CARD32 red, CARD32 green, CARD32 blue )
{
	int max_val, min_val, hue = 0 ;
	if( red > green )
	{
		max_val = MAX(red,blue);
		min_val = MIN(green,blue);
	}else
	{
		max_val = MAX(green,blue);
		min_val = MIN(red,blue);
	}
	if( max_val != min_val)
	{
		int delta = max_val-min_val ;
		MAKE_HUE16(hue,(int)red,(int)green,(int)blue,min_val,max_val,delta);
	}
	return hue;
}

CARD32
rgb2hsv( CARD32 red, CARD32 green, CARD32 blue, CARD32 *saturation, CARD32 *value )
{
	int max_val, min_val, hue = 0 ;
	if( red > green )
	{
		max_val = MAX(red,blue);
		min_val = MIN(green,blue);
	}else
	{
		max_val = MAX(green,blue);
		min_val = MIN(red,blue);
	}
	*value = max_val ;
	if( max_val != min_val)
	{
		int delta = max_val-min_val ;
		*saturation = (max_val>1)?(delta<<15)/(max_val>>1): 0;
		MAKE_HUE16(hue,(int)red,(int)green,(int)blue,min_val,max_val,delta);
	}else
		*saturation = 0 ;
	return hue;
}

void
hsv2rgb (CARD32 hue, CARD32 saturation, CARD32 value, CARD32 *red, CARD32 *green, CARD32 *blue)
{
	if (saturation == 0 || hue == 0 )
	{
    	*blue = *green = *red = value;
	}else
	{
		int delta = ((saturation*(value>>1))>>15) ;
		INTERPRET_HUE16(hue,delta,value,*red,*green,*blue);
	}
}

CARD32                                         /* returns luminance */
rgb2luminance (CARD32 red, CARD32 green, CARD32 blue )
{
	int max_val, min_val;
	if( red > green )
	{
		max_val = MAX(red,blue);
		min_val = MIN(green,blue);
	}else
	{
		max_val = MAX(green,blue);
		min_val = MIN(red,blue);
	}
	return (max_val+min_val)>>1;
}

CARD32                                         /* returns hue */
rgb2hls (CARD32 red, CARD32 green, CARD32 blue, CARD32 *luminance, CARD32 *saturation )
{
	int max_val, min_val, hue = 0 ;
	if( red > green )
	{
		max_val = MAX(red,blue);
		min_val = MIN(green,blue);
	}else
	{
		max_val = MAX(green,blue);
		min_val = MIN(red,blue);
	}
	*luminance = (max_val+min_val)>>1;

	if( max_val != min_val )
	{
		int delta = max_val-min_val ;
		if( *luminance == 0 ) ++(*luminance);
		else if( *luminance == 0x0000FFFF ) --(*luminance);
		*saturation = (*luminance < 0x00008000 )?
							(delta<<15)/ *luminance :
							(delta<<15)/ (0x0000FFFF - *luminance);
		MAKE_HUE16(hue,(int)red,(int)green,(int)blue,min_val,max_val,delta);
	}else
		*saturation = 0 ;
	return hue;
}

void
hls2rgb (CARD32 hue, CARD32 luminance, CARD32 saturation, CARD32 *red, CARD32 *green, CARD32 *blue)
{
	if (saturation == 0)
	{
    	*blue = *green = *red = luminance;
	}else
	{
		int delta = ( luminance < 0x00008000 )?
						(saturation*luminance)>>15 :
	                    (saturation*(0x0000FFFF-luminance))>>15 ;
		int max_val = delta+(((luminance<<1)-delta)>>1) ;

		INTERPRET_HUE16(hue,delta,max_val,*red,*green,*blue);
	}
}

/*************************************************************************/
/* scanline blending 													 */
/*************************************************************************/

typedef struct merge_scanlines_func_desc {
    char *name ;
	int name_len ;
	merge_scanlines_func func;
	char *short_desc;
}merge_scanlines_func_desc;

merge_scanlines_func_desc std_merge_scanlines_func_list[] =
{
  { "add", 3, add_scanlines, "color addition with saturation" },
  { "alphablend", 10, alphablend_scanlines, "alpha-blending" },
  { "allanon", 7, allanon_scanlines, "color values averaging" },
  { "colorize", 8, colorize_scanlines, "hue and saturate bottom image same as top image" },
  { "darken", 6, darken_scanlines, "use lowest color value from both images" },
  { "diff", 4, diff_scanlines, "use absolute value of the color difference between two images" },
  { "dissipate", 9, dissipate_scanlines, "randomly alpha-blend images"},
  { "hue", 3, hue_scanlines, "hue bottom image same as top image"  },
  { "lighten", 7, lighten_scanlines, "use highest color value from both images" },
  { "overlay", 7, overlay_scanlines, "some weird image overlaying(see GIMP)" },
  { "saturate", 8, saturate_scanlines, "saturate bottom image same as top image"},
  { "screen", 6, screen_scanlines, "another weird image overlaying(see GIMP)" },
  { "sub", 3, sub_scanlines, "color substraction with saturation" },
  { "tint", 4, tint_scanlines, "tinting image with image" },
  { "value", 5, value_scanlines, "value bottom image same as top image" },
  { NULL, 0, NULL }
};

merge_scanlines_func
blend_scanlines_name2func( const char *name )
{
	register int i = 0;

	if( name == NULL )
		return NULL ;
    while( isspace((int)*name) ) ++name;
	do
	{
		if( name[0] == std_merge_scanlines_func_list[i].name[0] )
			if( mystrncasecmp( name, std_merge_scanlines_func_list[i].name,
			                   std_merge_scanlines_func_list[i].name_len ) == 0 )
				return std_merge_scanlines_func_list[i].func ;

	}while( std_merge_scanlines_func_list[++i].name != NULL );

	return NULL ;

}

void
list_scanline_merging(FILE* stream, const char *format)
{
	int i = 0 ;
	do
	{
		fprintf( stream, format,
			     std_merge_scanlines_func_list[i].name,
			     std_merge_scanlines_func_list[i].short_desc  );
	}while( std_merge_scanlines_func_list[++i].name != NULL );
}

#define BLEND_SCANLINES_HEADER \
	register int i = -1, max_i = bottom->width ; \
	register CARD32 *ta = top->alpha, *tr = top->red, *tg = top->green, *tb = top->blue; \
	register CARD32 *ba = bottom->alpha, *br = bottom->red, *bg = bottom->green, *bb = bottom->blue; \
	if( offset < 0 ){ \
		offset = -offset ; \
		ta += offset ;	tr += offset ;	tg += offset ;	tb += offset ; \
		if( (int)top->width-offset < max_i )	max_i = (int)(top->width)-offset ; \
	}else{ \
		if( offset > 0 ){ \
			ba += offset ;	br += offset ;	bg += offset ;	bb += offset ; \
			max_i -= offset ; }	\
		if( (int)(top->width) < max_i )	max_i = top->width ; \
	}



void
alphablend_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
	{
		int a = ta[i] ;
		int ca ;
/*fprintf( stderr, "%4.4x%4.4x%4.4x%4.4x+%4.4x%4.4x%4.4x%4.4x ", ba[i], br[i], bg[i], bb[i], ta[i], tr[i], tg[i], tb[i] );*/
		if( a >= 0x0000FF00 )
		{
			br[i] = tr[i] ;
			bg[i] = tg[i] ;
			bb[i] = tb[i] ;
			ba[i] = 0x0000FF00;
		}else if( a > 0x000000FF )
		{
			a = (a>>8) ;
			ca = 255-a;
#if 0 /*ndef HAVE_MMX*/
/* MMX implementaion of alpha-blending below turns out to be 
   30% slower then the original integer math implementation under it 
   I'm probably stupid or something.  
 */
			__m64	va  = _mm_set_pi16 (ca, a, ca, a);
			__m64	vd  = _mm_set_pi16 (br[i],tr[i],ba[i],ta[i]);

			/* b=(b*ca + t*a)>>8 */
			vd = _mm_srli_pi16( vd, 8 );
			vd = _mm_madd_pi16( va, vd );
			ba[i] = _mm_cvtsi64_si32( vd );
			vd = _mm_srli_si64( vd, 32 );
			br[i] = _mm_cvtsi64_si32( vd );
			
			vd = _mm_set_pi16 (bb[i],tb[i],bg[i],tg[i]);
			vd = _mm_srli_pi16( vd, 8 );
			vd = _mm_madd_pi16( va, vd );
			bg[i] = _mm_cvtsi64_si32( vd );
			vd = _mm_srli_si64( vd, 32 );
			bb[i] = _mm_cvtsi64_si32( vd );
			_mm_empty();
#else
			ba[i] = ((ba[i]*ca)>>8)+ta[i] ;
			br[i] = (br[i]*ca+tr[i]*a)>>8 ;
			bg[i] = (bg[i]*ca+tg[i]*a)>>8 ;
			bb[i] = (bb[i]*ca+tb[i]*a)>>8 ;
#endif	
		}
	}
	
/*	fputc( '\n', stderr );*/
}

void    /* this one was first implemented on XImages by allanon :) - mode 131  */
allanon_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
	{
		if( ta[i] != 0 )
		{
			br[i] = (br[i]+tr[i])>>1 ;
			bg[i] = (bg[i]+tg[i])>>1 ;
			bb[i] = (bb[i]+tb[i])>>1 ;
			ba[i] = (ba[i]+ta[i])>>1 ;
		}
	}
}

void
tint_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
	{
		if( ta[i] != 0 )
		{
			br[i] = (br[i]*(tr[i]>>1))>>15 ;
			bg[i] = (bg[i]*(tg[i]>>1))>>15 ;
			bb[i] = (bb[i]*(tb[i]>>1))>>15 ;
		}
	}
}

void    /* addition with saturation : */
add_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
			br[i] = (br[i]+tr[i]) ;
			if( br[i] > 0x0000FFFF )
				br[i] = 0x0000FFFF ;
			bg[i] = (bg[i]+tg[i]) ;
			if( bg[i] > 0x0000FFFF )
				bg[i] = 0x0000FFFF ;
			bb[i] = (bb[i]+tb[i]) ;
			if( bb[i] > 0x0000FFFF )
				bb[i] = 0x0000FFFF ;
			ba[i] = (ba[i]+ta[i]) ;
			if( ba[i] > 0x0000FFFF )
				ba[i] = 0x0000FFFF ;
		}
}

void    /* substruction with saturation : */
sub_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			int res ;
			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
			res = (int)br[i] - (int)tr[i] ;
			br[i] = res < 0 ? 0: res ;
			res = (int)bg[i] - (int)tg[i] ;
			bg[i] = res < 0 ? 0: res ;
			res = (int)bb[i] - (int)tb[i] ;
			bb[i] = res < 0 ? 0: res ;
		}
}

void    /* absolute pixel value difference : */
diff_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
	{
		if( ta[i] )
		{
			int res = (int)br[i] - (int)tr[i] ;
			br[i] = res < 0 ? -res: res ;
			res = (int)bg[i] - (int)tg[i] ;
			bg[i] = res < 0 ? -res: res ;
			res = (int)bb[i] - (int)tb[i] ;
			bb[i] = res < 0 ? -res: res ;

			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
		}
	}
}

void    /* darkest of the two makes it in : */
darken_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			if( ta[i] < ba[i] )
				ba[i] = ta[i] ;
			if( tr[i] < br[i] )
				br[i] = tr[i];
			if( tg[i] < bg[i] )
				bg[i] = tg[i];
			if( tb[i] < bb[i] )
				bb[i] = tb[i];
		}
}

void    /* lightest of the two makes it in : */
lighten_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
			if( tr[i] > br[i] )
				br[i] = tr[i];
			if( tg[i] > bg[i] )
				bg[i] = tg[i];
			if( tb[i] > bb[i] )
				bb[i] = tb[i];
		}
}

void    /* guess what this one does - I could not :) */
screen_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
#define DO_SCREEN_VALUE(b,t) \
			res1 = 0x0000FFFF - (int)b[i] ; res2 = 0x0000FFFF - (int)t[i] ;\
			res1 = 0x0000FFFF - ((res1*res2)>>16); b[i] = res1 < 0 ? 0 : res1

	while( ++i < max_i )
		if( ta[i] )
		{
			int res1 ;
			int res2 ;

			DO_SCREEN_VALUE(br,tr);
			DO_SCREEN_VALUE(bg,tg);
			DO_SCREEN_VALUE(bb,tb);

			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
		}
}

void    /* somehow overlays bottom with top : */
overlay_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
#define DO_OVERLAY_VALUE(b,t) \
				tmp_screen = 0x0000FFFF - (((0x0000FFFF - (int)b[i]) * (0x0000FFFF - (int)t[i])) >> 16); \
				tmp_mult   = (b[i] * t[i]) >> 16; \
				res = (b[i] * tmp_screen + (0x0000FFFF - (int)b[i]) * tmp_mult) >> 16; \
				b[i] = res < 0 ? 0 : res

	while( ++i < max_i )
		if( ta[i] )
		{
			int tmp_screen, tmp_mult, res ;
			DO_OVERLAY_VALUE(br,tr);
			DO_OVERLAY_VALUE(bg,tg);
			DO_OVERLAY_VALUE(bb,tb);
			if( ta[i] > ba[i] )
				ba[i] = ta[i] ;
		}
}

void
hue_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			CARD32 hue = rgb2hue( tr[i], tg[i], tb[i]);
			if( hue > 0 )
			{
				CARD32 saturation = rgb2saturation( br[i], bg[i], bb[i]);
				CARD32 value = rgb2value( br[i], bg[i], bb[i]);;

				hsv2rgb(hue, saturation, value, &br[i], &bg[i], &bb[i]);
			}
			if( ta[i] < ba[i] )
				ba[i] = ta[i] ;
		}
}

void
saturate_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			CARD32 saturation, value;
			CARD32 hue = rgb2hsv( br[i], bg[i], bb[i], &saturation, &value);

			saturation = rgb2saturation( tr[i], tg[i], tb[i]);
			hsv2rgb(hue, saturation, value, &br[i], &bg[i], &bb[i]);
			if( ta[i] < ba[i] )
				ba[i] = ta[i] ;
		}
}

void
value_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER
	while( ++i < max_i )
		if( ta[i] )
		{
			CARD32 saturation, value;
			CARD32 hue = rgb2hsv( br[i], bg[i], bb[i], &saturation, &value);

			value = rgb2value( tr[i], tg[i], tb[i]);
			hsv2rgb(hue, saturation, value, &br[i], &bg[i], &bb[i]);

			if( ta[i] < ba[i] )
				ba[i] = ta[i] ;
		}
}

void
colorize_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	BLEND_SCANLINES_HEADER

	while( ++i < max_i )
		if( ta[i] )
		{
#if 1
			CARD32 luminance, saturation ;
			CARD32 hue = rgb2hls( tr[i], tg[i], tb[i], &luminance, &saturation );

			luminance = rgb2luminance( br[i], bg[i], bb[i]);
			hls2rgb(hue, luminance, saturation, &br[i], &bg[i], &bb[i]);
#else
			CARD32 h, l, s, r, g, b;
			h = rgb2hls( br[i], bg[i], bb[i], &l, &s );
			hls2rgb( h, l, s, &r, &g, &b );
			if( r > br[i]+10 || r < br[i] - 10 )
			{
				fprintf( stderr, "%X.%X.%X -> %X.%X.%X -> %X.%X.%X\n",  br[i], bg[i], bb[i], h, l, s, r, g, b );
				fprintf( stderr, "%d.%d.%d -> %d.%d.%d -> %d.%d.%d\n",  br[i], bg[i], bb[i], h, l, s, r, g, b );
			}
#endif
			if( ta[i] < ba[i] )
				ba[i] = ta[i] ;
		}
}

void
dissipate_scanlines( ASScanline *bottom, ASScanline *top, int offset )
{
	static   CARD32 rnd32_seed = 345824357;
	BLEND_SCANLINES_HEADER

#define MAX_MY_RND32		0x00ffffffff
#ifdef WORD64
#define MY_RND32() \
(rnd32_seed = ((1664525L*rnd32_seed)&MAX_MY_RND32)+1013904223L)
#else
#define MY_RND32() \
(rnd32_seed = (1664525L*rnd32_seed)+1013904223L)
#endif

	/* add some randomization here  if (rand < alpha) - combine */
	while( ++i < max_i )
	{
		int a = ta[i] ;
		if( a > 0 && (int)MY_RND32() < (a<<15) )
		{
			ba[i] += a ;
			if( ba[i] > 0x0000FFFF )
				ba[i] = 0x0000FFFF ;
			a = (a>>8) ;
			br[i] = (br[i]*(255-a)+tr[i]*a)>>8 ;
			bg[i] = (bg[i]*(255-a)+tg[i]*a)>>8 ;
			bb[i] = (bb[i]*(255-a)+tb[i]*a)>>8 ;
		}
	}
}

/*********************************************************************************/
/* The end !!!! 																 */
/*********************************************************************************/

