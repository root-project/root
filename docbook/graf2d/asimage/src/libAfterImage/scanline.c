/*
 * Copyright (c) 2008,2001,2000,1999 Sasha Vasko <sasha at aftercode.net>
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

#define LOCAL_DEBUG
#include <string.h>
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
#include "scanline.h"


/* ********************* ASScanline ************************************/
ASScanline*
prepare_scanline( unsigned int width, unsigned int shift, ASScanline *reusable_memory, Bool BGR_mode  )
{
	register ASScanline *sl = reusable_memory ;
	size_t aligned_width;
	void *ptr;

	if( sl == NULL )
		sl = safecalloc( 1, sizeof( ASScanline ) );
	else
		memset( sl, 0x00, sizeof(ASScanline));

	if( width == 0 ) width = 1 ;
	sl->width 	= width ;
	sl->shift   = shift ;
	/* we want to align data by 8 byte boundary (double)
	 * to allow for code with less ifs and easier MMX/3Dnow utilization :*/
	aligned_width = width + (width&0x00000001);
	sl->buffer = ptr = safecalloc (1, ((aligned_width*4)+16)*sizeof(CARD32)+8);
	if (ptr == NULL)
	{
		if (sl != reusable_memory)
			free (sl);
		return NULL;
	}

	sl->xc1 = sl->red 	= (CARD32*)((((long)ptr+7)>>3)*8);
	sl->xc2 = sl->green = sl->red   + aligned_width;
	sl->xc3 = sl->blue 	= sl->green + aligned_width;
	sl->alpha 	= sl->blue  + aligned_width;

	sl->channels[IC_RED] = sl->red ;
	sl->channels[IC_GREEN] = sl->green ;
	sl->channels[IC_BLUE] = sl->blue ;
	sl->channels[IC_ALPHA] = sl->alpha ;

	if( BGR_mode )
	{
		sl->xc1 = sl->blue ;
		sl->xc3 = sl->red ;
	}
	/* this way we can be sure that our buffers have size of multiplies of 8s
	 * and thus we can skip unneeded checks in code */
#if 0
	/* initializing padding into 0 to avoid any garbadge carry-over
	 * bugs with diffusion: */
	sl->red[aligned_width-1]   = 0;
	sl->green[aligned_width-1] = 0;
	sl->blue[aligned_width-1]  = 0;
	sl->alpha[aligned_width-1] = 0;
#endif	
	sl->back_color = ARGB32_DEFAULT_BACK_COLOR;

	return sl;
}

void
free_scanline( ASScanline *sl, Bool reusable )
{
	if( sl )
	{
		if( sl->buffer )
			free( sl->buffer );
		if( !reusable )
			free( sl );
	}
}

/* demosaicing */
void
destroy_asim_strip (ASIMStrip **pstrip)
{
	if (pstrip)
	{
		ASIMStrip *strip = *pstrip;
		if (strip)
		{
			int i;
			if (strip->lines)
			{
				for (i = 0; i < strip->size; ++i)
					free_scanline (strip->lines[i], False);
				free (strip->lines);
			}
			if (strip->aux_data)
			{
				for (i = 0; i < strip->size; ++i )
					if (strip->aux_data[i])
						free(strip->aux_data[i]);
				free (strip->aux_data); 
			}
			free (strip);
			*pstrip = NULL;
		}
	}
}

ASIMStrip *
create_asim_strip(unsigned int size, unsigned int width, int shift, int bgr)
{
	ASIMStrip *strip;
	int i;
	
	if (width == 0 || size == 0)
		return NULL;
	
	strip = safecalloc( 1, sizeof(ASIMStrip));
	strip->size = size;
	
	if ((strip->lines = safecalloc (size, sizeof(ASScanline*))) == NULL)
	{
		free (strip);
		return NULL;
	}

	if ((strip->aux_data = safecalloc (size, sizeof(void*))) == NULL)
	{
		destroy_asim_strip (&strip);
		return NULL;
	}
	
	for (i = 0 ; i < (int)size; ++i)
		if ((strip->lines[i] = prepare_scanline (width, shift, NULL, bgr)) == NULL)
		{
			strip->size = i;
			destroy_asim_strip (&strip);
			return NULL;
		}

	strip->width = width;
	strip->start_line = 0;
	
	return strip;
}

void
advance_asim_strip (ASIMStrip *strip)
{
	ASScanline *tmp = strip->lines[0];
	void *aux_tmp = strip->aux_data[0];
	int i;
	
	/* move all scanlines up, shuffling first scanline to the back */
	for (i = 0 ; i < strip->size-1; ++i )
	{
		strip->lines[i] = strip->lines[i+1];
		strip->aux_data[i] = strip->aux_data[i+1];
	}
	strip->lines[strip->size-1] = tmp;
	strip->aux_data[strip->size-1] = aux_tmp;

	/* clear the state of the scanline : */
	tmp->flags = 0;	

	strip->start_line++;
} 

/* returns number of lines processed from the data */
int
load_asim_strip (ASIMStrip *strip, CARD8 *data, int data_size, int data_start_line, int data_row_size, 
				 ASIMStripLoader *line_loaders, int line_loaders_num)
{
	int line = 0;
	int loaded = 0;
	
	if (strip == NULL || data == NULL || data_size <= 0 || data_row_size <= 0 || line_loaders == NULL)
		return 0;
	line = data_start_line - strip->start_line;
	if (line < 0)
	{
		data += data_row_size*(-line);
		data_size -= data_row_size*(-line);
		line = 0;
	}
		
	while (line < strip->size && data_size > 0)
	{
		int loader = (strip->start_line+line)%line_loaders_num;
		if (!ASIM_IsStripLineLoaded(strip,line) && line_loaders[loader])
			line_loaders[loader] (strip->lines[line], data, data_size);

		++line;
		++loaded;
		data_size -= data_row_size;
		data += data_row_size;
	}
	return loaded;
}

static int
decode_12_be (CARD32 *c1, CARD32 *c2, CARD8 *data, int width, int data_size)
{
	int x;
	int max_x = (data_size*2)/3;
	
	if (max_x > width)
		max_x = width;
	
	if (max_x > 0)
	{
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
		fprintf (stderr, "decode_12_be CFA data : ");		
		for (x = 0 ; x < (max_x*3)/2; x += 3)
			fprintf (stderr, " |%2.2X %2.2X %2.2X", data[x], data[x+1], data[x+2]);				
		fprintf (stderr, "\n");		
#endif
		for (x = 0 ; x+1 < max_x; ++x)
		{
			CARD32 tail = ((CARD32)data[1])&0x00F0;
			c1[x] = (((CARD32)data[0]) << 8)|tail|(tail>>4);
			c2[x] = ASIM_SCL_MissingValue;
			
			++x;
			tail = data[2]&0x0F;
			c1[x] = ASIM_SCL_MissingValue;
			c2[x] = (((CARD32)data[1]&0x0f) << 12)| ((CARD32)data[2]<<4) |tail;
			
			data += 3;
		}

		if (x < max_x)
		{
			CARD32 tail = ((CARD32)data[1])&0x00F0;
			c1[x] = (((CARD32)data[0]) << 8)|tail|(tail>>4);
			c2[x] = ASIM_SCL_MissingValue;
		}

#if 0
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)	
fprintf (stderr, "decode_12_be  C1 data : ");		
	for (x = 0 ; x < max_x; ++x)
		fprintf (stderr, " %4.4X", c1[x]);						
fprintf (stderr, "\ndecode_12_be  C2 data : ");		
	for (x = 0 ; x < max_x; ++x)
		fprintf (stderr, " %4.4X", c2[x]);						
fprintf (stderr, "\n");				
#endif
#endif
	}
	return max_x;
} 

void decode_BG_12_be (ASScanline *scl, CARD8 *data, int data_size)
{
	if (decode_12_be (scl->blue, scl->green, data, scl->width, data_size))
		set_flags (scl->flags, SCL_DO_GREEN|SCL_DO_BLUE);
}

void decode_GR_12_be (ASScanline *scl, CARD8 *data, int data_size)
{
	if (decode_12_be (scl->green, scl->red, data, scl->width, data_size))	
		set_flags (scl->flags, SCL_DO_GREEN|SCL_DO_RED);
}

void decode_RG_12_be (ASScanline *scl, CARD8 *data, int data_size)
{
	if (decode_12_be (scl->red, scl->green, data, scl->width, data_size))
		set_flags (scl->flags, SCL_DO_GREEN|SCL_DO_RED);
}

void decode_GB_12_be (ASScanline *scl, CARD8 *data, int data_size)
{
	if (decode_12_be (scl->green, scl->blue, data, scl->width, data_size))
		set_flags (scl->flags, SCL_DO_GREEN|SCL_DO_BLUE);
}

/* min gradient interpolation : */
typedef void (*ASIMDiagInterpolationFunc) (CARD32 *dst, CARD32 **channs, int width, int offset);

#define ASIM_ChooseInterpolationGradient(g1,g2)  ((g1/16)*(g1/16)>(g2/16)*(g2/16)?(g1):(g2))

void
interpolate_channel_v_15x51 (CARD32 *dst, CARD32 **channs, int width, int offset)
{
	/* Assumptions :  channs is array of 5 CARD32 pointers not NULL */
	int x;
	for (x = 0; x < width; ++x)
	{
		int v = (int)channs[1][x]*5+(int)channs[3][x]*5-(int)channs[4][x]-(int)channs[0][x];
		dst[x] = v <= 0 ? 0 : v >>3;
	}
}

void
interpolate_channel_v_checkered_15x51 (CARD32 *dst, CARD32 **channs, int width, int offset)
{
	/* Assumptions :  channs is array of 5 CARD32 pointers not NULL */
	int x = offset;
	for (x = 0; x < width; ++x, ++x)
	{
		int v = (int)channs[1][x]*5+(int)channs[3][x]*5-(int)channs[4][x]-(int)channs[0][x];
		dst[x] = v <= 0 ? 0 : v >>3;
	}
}

/* vert smoothing */
void
smooth_channel_v_15x51 (CARD32 *dst, CARD32 **channs, int width, int offset)
{
	/* Assumptions :  channs is array of 5 CARD32 pointers not NULL */
	int x;
	for (x = 0; x < width; ++x)
	{
		int v = (int)(channs[2][x]<<3) + (int)channs[1][x]*5+(int)channs[3][x]*5-(int)channs[4][x]-(int)channs[0][x];
		dst[x] = v <= 0 ? 0 : v >>4;
	}
}

/* horizontal interpolation */
void
interpolate_channel_h_105x501 (CARD32 *chan, int width)
{
	int v;
	int chan0 = chan[0];
	int x = 1;
	
	/* interpolating every other pixel from its 5 neighbours */
	/* Assumptions :  width > 4 and width%2 == 0 */
	if (ASIM_IsMissingValue(chan0))
	{
		x = 0;
		chan0 = chan[1];
	}
	
	v = (int)chan0*4 + (int)chan[x+1]*5 - (int)chan[x+3];

	chan[x] = v < 0 ? 0: v>>3 ;

	v -= (int)chan0*5;

	if (x == 0)
	{
		x += 2;	
		v += (int)chan[x+1]*6 - (int)chan[x+3];
		chan[x] = v < 0 ? 0: v>>3 ;
		v -= (int)chan[x-1]*6 - (int)chan0;
	}

	for ( x += 2 ; x+3 < width; x += 2)
	{
		v += (int)chan[x+1]*6 - (int)chan[x+3];
		chan[x] = v < 0 ? 0: v>>3 ;
		v -= (int)chan[x-1]*6 - (int)chan[x-3];
	}

	v = (int)chan[x+1]+(int)chan[x-1]*4-(int)chan[x-3];
	chan[x] = v <= 0 ? 0 : v >>2;
	v = (int)chan[x+2-1]*3 - (int)chan[x+2-3];
	chan[x+2] = v <= 0 ? 0 : v >> 1 ;
}

void
interpolate_channel_h_grad3 (CARD32 *c1, CARD32 *c2, int width)
{/* interpolate missing values in c2, minimizing variation of different from c2 */
	int v;
	int chan0 = c1[0];
	int x = 1;

	if (ASIM_IsMissingValue(chan0))
	{
		x = 0;	
	}

	v = (int)c2[x] + (int)c1[x+1] - (int)c2[x+2];
	c1[x] = v <= 0 ? 0 : v;
	
	for( x += 2; x+2 < width ; x += 2)
	{
		v = (int)(c2[x]<<1) + (int)c1[x-1] + (int)c1[x+1] - (int)c2[x+2] - (int)c2[x-2];
		c1[x] = v <= 0 ? 0 : v>>1;
	}
	
	if (x < width)
	{
		v = (int)c2[x] + (int)c1[x-1] - (int)c2[x-2];
		c1[x] = v <= 0 ? 0 : v;
	}
}

void print_16bit_chan (CARD32 *chan, int width)
{
	int x;
	for (x = 0 ; x < width ; ++x ) 
	{ 
		int v = chan[x]; 
		fprintf(stderr, " %5.5d", (v < 0)?99999:v );
	}
	fprintf( stderr, "\n");
}

Bool
interpolate_asim_strip_gradients (ASIMStrip *strip, int line, int chan_from, int chan_to, int offset, ASIMDiagInterpolationFunc func )
{
	CARD32 *chan_lines[5] = {NULL, NULL, NULL, NULL, NULL};
	int above = 2, below = 2;
	int i = line;
	int chan = chan_to;

	while (--i >= 0 && below > 0)
		if (get_flags(strip->lines[i]->flags, 0x01<<chan))
		{
			chan_lines[--below] = strip->lines[i]->channels[chan];
			chan = (chan == chan_to) ? chan_from : chan_to;
		}
	if (below > 0)
		return False;

	chan_lines[2] = strip->lines[line]->channels[chan_from];
	/* chan here should be in proper position if below == 0 */

	i = line ;
	while (++i < strip->size && above < 4)
	{
		if (get_flags(strip->lines[i]->flags, 0x01<<chan))
		{
			chan_lines[++above] = strip->lines[i]->channels[chan];
			chan = (chan == chan_to) ? chan_from : chan_to;
		}
	}

	if (above < 4) /* not enough data for interpolation */
		return False;

#if 0
print_16bit_chan (chan_lines[0], strip->lines[line]->width); 
print_16bit_chan (chan_lines[1], strip->lines[line]->width); 
print_16bit_chan (chan_lines[2], strip->lines[line]->width); 
print_16bit_chan (chan_lines[3], strip->lines[line]->width); 
print_16bit_chan (chan_lines[4], strip->lines[line]->width); 
#endif

fprintf( stderr, "Line %d, start_line = %d, offset = %d, chan_to = %d, chan_from = %d\n", line, strip->start_line, offset, chan_to, chan_from);
	func (strip->lines[line]->channels[chan_to], chan_lines, strip->lines[line]->width, offset);

#if 0
print_16bit_chan (strip->lines[line]->channels[chan_to], strip->lines[line]->width); 
fprintf( stderr, "\n");
#endif
	
	return True;
}

static inline int*
checkalloc_diff_aux_data (ASIMStrip *strip, int line) 
{
	if (strip->aux_data[line] == NULL)
		strip->aux_data[line] = safemalloc(strip->lines[line]->width*2*sizeof(int));
	return strip->aux_data[line];
}

void
interpolate_channel_hv_adaptive_1x1(CARD32 *above, CARD32 *dst, CARD32 *below, int width, int offset)
{
	int x = offset;
	
	if (offset == 0)
	{
		dst[0] = (above[0] + below[0] + dst[1])/3;
		x += 2;
	}
	
	for (; x < width-1; ++x, ++x)
	{
		int v;
		int l = dst[x-1], r = dst[x+1];
		/* we have to operate with 14 bit values in order to avoid overflow */
		int diff_h = (l>>2)-(r>>2);
		int t = above[x], b = below[x];
		int diff_v = (t>>2)-(b>>2);
		if ((diff_h * diff_h) < (diff_v * diff_v))
		{
			v = (l + r) >> 1;
			if ((v < t && v < b) || (v > t && v > b))
				v  = ((v << 1) + b + t) >> 2 ;
		}else
		{
			v = (t + b) >> 1;
			if ((v < l && v < r) || (v > l && v > r))
				v  = ((v << 1) + l + r) >> 2 ;
		}
		dst[x] = v;
		
	}
	if (offset == 1)
		dst[x] = (above[x] + below[x] + dst[x-1])/3;
}

Bool calculate_green_diff(ASIMStrip *strip, int line, int chan, int offset)
{
	int width = strip->lines[line]->width;
	CARD32 *green = strip->lines[line]->green;
	CARD32 *src = strip->lines[line]->channels[chan];
	int *diff = checkalloc_diff_aux_data (strip, line);
	int x = offset;
	int v_last, v;
	
	if (diff == NULL)
		return False;

	if (chan == ARGB32_BLUE_CHAN)
		diff += width;

	v_last = (int)src[x] - (int)green[x];
	diff[x] = v_last;
	/* some loop unrolling for optimization purposes - 
	   we don't want to store diff[x] untill the second pass */
	x +=2;
	v = (int)src[x] - (int)green[x];
	diff[x-1] = (v + v_last)/2;
	diff[x] = v_last = v;
	
	while ((x += 2) < width-2)
	{
		v = (int)src[x] - (int)green[x];

		diff[x-1] = (v + v_last)/2;
		v_last = v;
	}

	v = (int)src[x] - (int)green[x];
	diff[x-1] = (v + v_last)/2;
	diff[x] = v;
	
	/* border condition handling : */
	if (offset)
		diff[0] = diff[1];
	else
		diff[width-1] = diff[width-2];

	/* second pass - further smoothing of the difference at the points 
	   where we are most likely to see artifacts */
	for (x = offset + 2; x < width-2 ; ++x,++x)
		diff[x] = (diff[x-1]+diff[x+1])/2;
	
	return True;	
}

Bool 
interpolate_green_diff(ASIMStrip *strip, int line, int chan, int offset)
{
	if (line > 0 && line < strip->size-1)
	{
		ASScanline *above = strip->lines[line-1];
		ASScanline *below = strip->lines[line+1];
		ASFlagType flag = (chan == ARGB32_RED_CHAN)?ASIM_SCL_RGDiffCalculated:ASIM_SCL_BGDiffCalculated;
		if (get_flags (above->flags, flag) && get_flags (below->flags, flag))
		{
			int *diff_above = strip->aux_data[line-1];
			int *diff_below = strip->aux_data[line+1];
			int *diff = checkalloc_diff_aux_data (strip, line);
			int max_x = above->width;
			int x = 0;

			if (diff == NULL)
				return False;

			if (chan == ARGB32_BLUE_CHAN)
			{
				x = max_x;
				max_x *= 2;
			}
			for (; x < max_x; ++x)
				diff[x] = (diff_above[x] + diff_below[x])/2;
			return True;
		}
	}
	return False;
}

Bool 
interpolate_from_green_diff(ASIMStrip *strip, int line, int chan, int offset)
{
	int width = strip->lines[line]->width;
	CARD32 *green = strip->lines[line]->green;
	CARD32 *dst = strip->lines[line]->channels[chan];
	int *diff = strip->aux_data[line];
	int x;
	
	if (diff == NULL)
		return False;

	if (chan == ARGB32_BLUE_CHAN)
		diff += width;
	
	for (x = 0 ; x < width; ++x)
	{
		int v = (int)green[x];
		v += diff[x];
		dst[x] = (v<0)? 0 : v;
	}

	return True;	
}


void
interpolate_asim_strip_custom_rggb2 (ASIMStrip *strip, ASFlagType filter, Bool force_all)
{
	int line;
#if 0
	int chan;
	for (line = 0; line < 2 ; ++line)
		for (chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan)
			if ( get_flags( filter, 0x01<<chan) )
			{
				if (!get_flags(strip->lines[line]->flags, 0x01<<chan))
				{
					copy_component (strip->lines[line+1]->channels[chan], strip->lines[line]->channels[chan], 0, strip->lines[line]->width);
					set_flags (strip->lines[0]->flags, 0x01<<chan);
				}
#if 1
				if (!get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedV|ASIM_SCL_InterpolatedH)<<chan))
				{
					interpolate_channel_h_105x501 (strip->lines[line]->channels[chan], strip->lines[line]->width);
					set_flags(strip->lines[line]->flags, ASIM_SCL_InterpolatedH<<chan);
				}
#endif				
			}

	if (force_all)
		for (line = strip->size-2; strip->size ; ++line)
			for (chan = 0 ; chan < IC_NUM_CHANNELS ; ++chan)
				if ( get_flags( filter, 0x01<<chan) )
				{
					if (!get_flags(strip->lines[line]->flags, 0x01<<chan))
					{
						copy_component (strip->lines[line-1]->channels[chan], strip->lines[line]->channels[chan], 0, strip->lines[line]->width);
						set_flags (strip->lines[0]->flags, 0x01<<chan);
					}

					if (!get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedV|ASIM_SCL_InterpolatedH)<<chan))
					{
						interpolate_channel_h_105x501 (strip->lines[line]->channels[chan], strip->lines[line]->width);
						set_flags(strip->lines[line]->flags, ASIM_SCL_InterpolatedH<<chan);
					}
				}

#endif

#if 1
	/* interpolation of green */
	if ( get_flags( filter, SCL_DO_GREEN) )
	{
		for (line = 1 ; line < strip->size-1 ; ++line)
			if (get_flags(strip->lines[line]->flags, SCL_DO_GREEN)
				&& !get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedV<<ARGB32_GREEN_CHAN)))
			{
				if (get_flags(strip->lines[line-1]->flags, SCL_DO_GREEN)
					&& get_flags(strip->lines[line+1]->flags, SCL_DO_GREEN))
				{
					interpolate_channel_hv_adaptive_1x1 (strip->lines[line-1]->green, 
														 strip->lines[line]->green, 
														 strip->lines[line+1]->green,
														 strip->lines[line]->width,
														 ASIM_IsMissingValue(strip->lines[line]->green[0])?0:1);
					set_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedH|ASIM_SCL_InterpolatedV)<<ARGB32_GREEN_CHAN);
//					strip->lines[line]->flags =  SCL_DO_GREEN| ((ASIM_SCL_InterpolatedH|ASIM_SCL_InterpolatedV)<<ARGB32_GREEN_CHAN);
				}
			}
	}
#endif

/* now that we have smooth subtrate of green - we can build red/blue channels : 
 *   1) Calculate R-G difference for all RG lines, averaging missing values from 2 neightbours
 *   2) Calculate R-G difference for all GB lines, averaging values from lines above and below it.
 *   3) Calculate ALL RED values by adding calulated difference to Green channel.
 *  Do the same for BLUE.
 */
#if 1
	/* interpolation of red from green + (R-G) */
	if ( get_flags( filter, SCL_DO_RED) )
	{
		/* step 1. Calculating R-G for RG lines */
		for (line = 0 ; line < strip->size ; ++line)
			if (get_flags(strip->lines[line]->flags, SCL_DO_RED)
				&& !get_flags(strip->lines[line]->flags, ASIM_SCL_RGDiffCalculated)
			    && get_flags(strip->lines[line]->flags, SCL_DO_GREEN)
				&& get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedAll<<ARGB32_GREEN_CHAN)))
			{
				if (calculate_green_diff(strip, line, ARGB32_RED_CHAN, 0))
					set_flags(strip->lines[line]->flags, ASIM_SCL_RGDiffCalculated);
			}
		/* step 2. Calculating R-G for GB lines */
		for (line = 0 ; line < strip->size ; ++line)
			if (!get_flags(strip->lines[line]->flags, SCL_DO_RED)
				&& !get_flags(strip->lines[line]->flags, ASIM_SCL_RGDiffCalculated))
			{
				if (interpolate_green_diff(strip, line, ARGB32_RED_CHAN, 0))
					set_flags(strip->lines[line]->flags, ASIM_SCL_RGDiffCalculated);
			}
		/* step 3. Calculating RED from green + R-G */
		for (line = 0 ; line < strip->size ; ++line)
			if (get_flags(strip->lines[line]->flags, ASIM_SCL_RGDiffCalculated)
			    && !get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedAll<<ARGB32_RED_CHAN)))
			{
				if (interpolate_from_green_diff(strip, line, ARGB32_RED_CHAN, 0))
					set_flags(strip->lines[line]->flags, SCL_DO_RED|(ASIM_SCL_InterpolatedAll<<ARGB32_RED_CHAN));
			}
	}
#endif
#if 1
	/* interpolation of blue from green + (B-G) */
	if ( get_flags( filter, SCL_DO_BLUE) )
	{
		/* step 1. Calculating B-G for GB lines */
		for (line = 0 ; line < strip->size ; ++line)
			if (get_flags(strip->lines[line]->flags, SCL_DO_BLUE)
				&& !get_flags(strip->lines[line]->flags, ASIM_SCL_BGDiffCalculated)
			    && get_flags(strip->lines[line]->flags, SCL_DO_GREEN)
				&& get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedAll<<ARGB32_GREEN_CHAN)))
			{
				if (calculate_green_diff(strip, line, ARGB32_BLUE_CHAN, 1))
					set_flags(strip->lines[line]->flags, ASIM_SCL_BGDiffCalculated);
			}
		/* step 2. Calculating B-G for RG lines */
		for (line = 0 ; line < strip->size ; ++line)
			if (!get_flags(strip->lines[line]->flags, SCL_DO_BLUE)
				&& !get_flags(strip->lines[line]->flags, ASIM_SCL_BGDiffCalculated))
			{
				if (interpolate_green_diff(strip, line, ARGB32_BLUE_CHAN, 1))
					set_flags(strip->lines[line]->flags, ASIM_SCL_BGDiffCalculated);
			}
		/* step 3. Calculating BLUE from green + R-G */
		for (line = 0 ; line < strip->size ; ++line)
			if (get_flags(strip->lines[line]->flags, ASIM_SCL_BGDiffCalculated)
			    && !get_flags(strip->lines[line]->flags, (ASIM_SCL_InterpolatedAll<<ARGB32_BLUE_CHAN)))
			{
				if (interpolate_from_green_diff(strip, line, ARGB32_BLUE_CHAN, 1))
					set_flags(strip->lines[line]->flags, SCL_DO_BLUE|(ASIM_SCL_InterpolatedAll<<ARGB32_BLUE_CHAN));
			}
	}
#endif

}
