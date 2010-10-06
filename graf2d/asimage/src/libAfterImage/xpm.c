/* This file contains code for unified image loading from XPM file  */
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

#ifdef _WIN32
#include "win32/config.h"
#include <io.h>
#define read _read
#else
#include "config.h"
#endif

#ifdef HAVE_XPM

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <ctype.h>
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
#include <sys/stat.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif
#include <fcntl.h>
#include <string.h>

#ifdef HAVE_LIBXPM      /* XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM XPM */
#ifdef HAVE_LIBXPM_X11
#include <X11/xpm.h>
#else
#include <xpm.h>
#endif
#endif

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asimage.h"
#include "ascmap.h"
#include "xpm.h"

#define MAXPRINTABLE 92
/* number of printable ascii chars minus \ and " for string compat
 * and ? to avoid ANSI trigraphs. */

static char *printable =
" .XoO+@#$%&*=-;:>,<1234567890qwertyuipasdfghjklzxcvbnmMNBVCZASDFGHJKLPIUYTREWQ!~^/()_`'][{}|";

static struct {
	char 	*name ;
	ARGB32   argb ;
	} XpmRGB_Colors[] =
{/* this entire table is taken from libXpm 	       */
 /* Developed by HeDu 3/94 (hedu@cul-ipn.uni-kiel.de)  */
    {"AliceBlue", MAKE_ARGB32(255, 240, 248, 255)},
    {"AntiqueWhite", MAKE_ARGB32(255, 250, 235, 215)},
    {"Aquamarine", MAKE_ARGB32(255, 50, 191, 193)},
    {"Azure", MAKE_ARGB32(255, 240, 255, 255)},
    {"Beige", MAKE_ARGB32(255, 245, 245, 220)},
    {"Bisque", MAKE_ARGB32(255, 255, 228, 196)},
    {"Black", MAKE_ARGB32(255, 0, 0, 0)},
    {"BlanchedAlmond", MAKE_ARGB32(255, 255, 235, 205)},
    {"Blue", MAKE_ARGB32(255, 0, 0, 255)},
    {"BlueViolet", MAKE_ARGB32(255, 138, 43, 226)},
    {"Brown", MAKE_ARGB32(255, 165, 42, 42)},
    {"burlywood", MAKE_ARGB32(255, 222, 184, 135)},
    {"CadetBlue", MAKE_ARGB32(255, 95, 146, 158)},
    {"chartreuse", MAKE_ARGB32(255, 127, 255, 0)},
    {"chocolate", MAKE_ARGB32(255, 210, 105, 30)},
    {"Coral", MAKE_ARGB32(255, 255, 114, 86)},
    {"CornflowerBlue", MAKE_ARGB32(255, 34, 34, 152)},
    {"cornsilk", MAKE_ARGB32(255, 255, 248, 220)},
    {"Cyan", MAKE_ARGB32(255, 0, 255, 255)},
    {"DarkGoldenrod", MAKE_ARGB32(255, 184, 134, 11)},
    {"DarkGreen", MAKE_ARGB32(255, 0, 86, 45)},
    {"DarkKhaki", MAKE_ARGB32(255, 189, 183, 107)},
    {"DarkOliveGreen", MAKE_ARGB32(255, 85, 86, 47)},
    {"DarkOrange", MAKE_ARGB32(255, 255, 140, 0)},
    {"DarkOrchid", MAKE_ARGB32(255, 139, 32, 139)},
    {"DarkSalmon", MAKE_ARGB32(255, 233, 150, 122)},
    {"DarkSeaGreen", MAKE_ARGB32(255, 143, 188, 143)},
    {"DarkSlateBlue", MAKE_ARGB32(255, 56, 75, 102)},
    {"DarkSlateGray", MAKE_ARGB32(255, 47, 79, 79)},
    {"DarkTurquoise", MAKE_ARGB32(255, 0, 166, 166)},
    {"DarkViolet", MAKE_ARGB32(255, 148, 0, 211)},
    {"DeepPink", MAKE_ARGB32(255, 255, 20, 147)},
    {"DeepSkyBlue", MAKE_ARGB32(255, 0, 191, 255)},
    {"DimGray", MAKE_ARGB32(255, 84, 84, 84)},
    {"DodgerBlue", MAKE_ARGB32(255, 30, 144, 255)},
    {"Firebrick", MAKE_ARGB32(255, 142, 35, 35)},
    {"FloralWhite", MAKE_ARGB32(255, 255, 250, 240)},
    {"ForestGreen", MAKE_ARGB32(255, 80, 159, 105)},
    {"gainsboro", MAKE_ARGB32(255, 220, 220, 220)},
    {"GhostWhite", MAKE_ARGB32(255, 248, 248, 255)},
    {"Gold", MAKE_ARGB32(255, 218, 170, 0)},
    {"Goldenrod", MAKE_ARGB32(255, 239, 223, 132)},
    {"Gray", MAKE_ARGB32(255, 126, 126, 126)},
    {"Gray0", MAKE_ARGB32(255, 0, 0, 0)},
    {"Gray1", MAKE_ARGB32(255, 3, 3, 3)},
    {"Gray10", MAKE_ARGB32(255, 26, 26, 26)},
    {"Gray100", MAKE_ARGB32(255, 255, 255, 255)},
    {"Gray11", MAKE_ARGB32(255, 28, 28, 28)},
    {"Gray12", MAKE_ARGB32(255, 31, 31, 31)},
    {"Gray13", MAKE_ARGB32(255, 33, 33, 33)},
    {"Gray14", MAKE_ARGB32(255, 36, 36, 36)},
    {"Gray15", MAKE_ARGB32(255, 38, 38, 38)},
    {"Gray16", MAKE_ARGB32(255, 41, 41, 41)},
    {"Gray17", MAKE_ARGB32(255, 43, 43, 43)},
    {"Gray18", MAKE_ARGB32(255, 46, 46, 46)},
    {"Gray19", MAKE_ARGB32(255, 48, 48, 48)},
    {"Gray2", MAKE_ARGB32(255, 5, 5, 5)},
    {"Gray20", MAKE_ARGB32(255, 51, 51, 51)},
    {"Gray21", MAKE_ARGB32(255, 54, 54, 54)},
    {"Gray22", MAKE_ARGB32(255, 56, 56, 56)},
    {"Gray23", MAKE_ARGB32(255, 59, 59, 59)},
    {"Gray24", MAKE_ARGB32(255, 61, 61, 61)},
    {"Gray25", MAKE_ARGB32(255, 64, 64, 64)},
    {"Gray26", MAKE_ARGB32(255, 66, 66, 66)},
    {"Gray27", MAKE_ARGB32(255, 69, 69, 69)},
    {"Gray28", MAKE_ARGB32(255, 71, 71, 71)},
    {"Gray29", MAKE_ARGB32(255, 74, 74, 74)},
    {"Gray3", MAKE_ARGB32(255, 8, 8, 8)},
    {"Gray30", MAKE_ARGB32(255, 77, 77, 77)},
    {"Gray31", MAKE_ARGB32(255, 79, 79, 79)},
    {"Gray32", MAKE_ARGB32(255, 82, 82, 82)},
    {"Gray33", MAKE_ARGB32(255, 84, 84, 84)},
    {"Gray34", MAKE_ARGB32(255, 87, 87, 87)},
    {"Gray35", MAKE_ARGB32(255, 89, 89, 89)},
    {"Gray36", MAKE_ARGB32(255, 92, 92, 92)},
    {"Gray37", MAKE_ARGB32(255, 94, 94, 94)},
    {"Gray38", MAKE_ARGB32(255, 97, 97, 97)},
    {"Gray39", MAKE_ARGB32(255, 99, 99, 99)},
    {"Gray4", MAKE_ARGB32(255, 10, 10, 10)},
    {"Gray40", MAKE_ARGB32(255, 102, 102, 102)},
    {"Gray41", MAKE_ARGB32(255, 105, 105, 105)},
    {"Gray42", MAKE_ARGB32(255, 107, 107, 107)},
    {"Gray43", MAKE_ARGB32(255, 110, 110, 110)},
    {"Gray44", MAKE_ARGB32(255, 112, 112, 112)},
    {"Gray45", MAKE_ARGB32(255, 115, 115, 115)},
    {"Gray46", MAKE_ARGB32(255, 117, 117, 117)},
    {"Gray47", MAKE_ARGB32(255, 120, 120, 120)},
    {"Gray48", MAKE_ARGB32(255, 122, 122, 122)},
    {"Gray49", MAKE_ARGB32(255, 125, 125, 125)},
    {"Gray5", MAKE_ARGB32(255, 13, 13, 13)},
    {"Gray50", MAKE_ARGB32(255, 127, 127, 127)},
    {"Gray51", MAKE_ARGB32(255, 130, 130, 130)},
    {"Gray52", MAKE_ARGB32(255, 133, 133, 133)},
    {"Gray53", MAKE_ARGB32(255, 135, 135, 135)},
    {"Gray54", MAKE_ARGB32(255, 138, 138, 138)},
    {"Gray55", MAKE_ARGB32(255, 140, 140, 140)},
    {"Gray56", MAKE_ARGB32(255, 143, 143, 143)},
    {"Gray57", MAKE_ARGB32(255, 145, 145, 145)},
    {"Gray58", MAKE_ARGB32(255, 148, 148, 148)},
    {"Gray59", MAKE_ARGB32(255, 150, 150, 150)},
    {"Gray6", MAKE_ARGB32(255, 15, 15, 15)},
    {"Gray60", MAKE_ARGB32(255, 153, 153, 153)},
    {"Gray61", MAKE_ARGB32(255, 156, 156, 156)},
    {"Gray62", MAKE_ARGB32(255, 158, 158, 158)},
    {"Gray63", MAKE_ARGB32(255, 161, 161, 161)},
    {"Gray64", MAKE_ARGB32(255, 163, 163, 163)},
    {"Gray65", MAKE_ARGB32(255, 166, 166, 166)},
    {"Gray66", MAKE_ARGB32(255, 168, 168, 168)},
    {"Gray67", MAKE_ARGB32(255, 171, 171, 171)},
    {"Gray68", MAKE_ARGB32(255, 173, 173, 173)},
    {"Gray69", MAKE_ARGB32(255, 176, 176, 176)},
    {"Gray7", MAKE_ARGB32(255, 18, 18, 18)},
    {"Gray70", MAKE_ARGB32(255, 179, 179, 179)},
    {"Gray71", MAKE_ARGB32(255, 181, 181, 181)},
    {"Gray72", MAKE_ARGB32(255, 184, 184, 184)},
    {"Gray73", MAKE_ARGB32(255, 186, 186, 186)},
    {"Gray74", MAKE_ARGB32(255, 189, 189, 189)},
    {"Gray75", MAKE_ARGB32(255, 191, 191, 191)},
    {"Gray76", MAKE_ARGB32(255, 194, 194, 194)},
    {"Gray77", MAKE_ARGB32(255, 196, 196, 196)},
    {"Gray78", MAKE_ARGB32(255, 199, 199, 199)},
    {"Gray79", MAKE_ARGB32(255, 201, 201, 201)},
    {"Gray8", MAKE_ARGB32(255, 20, 20, 20)},
    {"Gray80", MAKE_ARGB32(255, 204, 204, 204)},
    {"Gray81", MAKE_ARGB32(255, 207, 207, 207)},
    {"Gray82", MAKE_ARGB32(255, 209, 209, 209)},
    {"Gray83", MAKE_ARGB32(255, 212, 212, 212)},
    {"Gray84", MAKE_ARGB32(255, 214, 214, 214)},
    {"Gray85", MAKE_ARGB32(255, 217, 217, 217)},
    {"Gray86", MAKE_ARGB32(255, 219, 219, 219)},
    {"Gray87", MAKE_ARGB32(255, 222, 222, 222)},
    {"Gray88", MAKE_ARGB32(255, 224, 224, 224)},
    {"Gray89", MAKE_ARGB32(255, 227, 227, 227)},
    {"Gray9", MAKE_ARGB32(255, 23, 23, 23)},
    {"Gray90", MAKE_ARGB32(255, 229, 229, 229)},
    {"Gray91", MAKE_ARGB32(255, 232, 232, 232)},
    {"Gray92", MAKE_ARGB32(255, 235, 235, 235)},
    {"Gray93", MAKE_ARGB32(255, 237, 237, 237)},
    {"Gray94", MAKE_ARGB32(255, 240, 240, 240)},
    {"Gray95", MAKE_ARGB32(255, 242, 242, 242)},
    {"Gray96", MAKE_ARGB32(255, 245, 245, 245)},
    {"Gray97", MAKE_ARGB32(255, 247, 247, 247)},
    {"Gray98", MAKE_ARGB32(255, 250, 250, 250)},
    {"Gray99", MAKE_ARGB32(255, 252, 252, 252)},
    {"Green", MAKE_ARGB32(255, 0, 255, 0)},
    {"GreenYellow", MAKE_ARGB32(255, 173, 255, 47)},
    {"honeydew", MAKE_ARGB32(255, 240, 255, 240)},
    {"HotPink", MAKE_ARGB32(255, 255, 105, 180)},
    {"IndianRed", MAKE_ARGB32(255, 107, 57, 57)},
    {"ivory", MAKE_ARGB32(255, 255, 255, 240)},
    {"Khaki", MAKE_ARGB32(255, 179, 179, 126)},
    {"lavender", MAKE_ARGB32(255, 230, 230, 250)},
    {"LavenderBlush", MAKE_ARGB32(255, 255, 240, 245)},
    {"LawnGreen", MAKE_ARGB32(255, 124, 252, 0)},
    {"LemonChiffon", MAKE_ARGB32(255, 255, 250, 205)},
    {"LightBlue", MAKE_ARGB32(255, 176, 226, 255)},
    {"LightCoral", MAKE_ARGB32(255, 240, 128, 128)},
    {"LightCyan", MAKE_ARGB32(255, 224, 255, 255)},
    {"LightGoldenrod", MAKE_ARGB32(255, 238, 221, 130)},
    {"LightGoldenrodYellow", MAKE_ARGB32(255, 250, 250, 210)},
    {"LightGray", MAKE_ARGB32(255, 168, 168, 168)},
    {"LightPink", MAKE_ARGB32(255, 255, 182, 193)},
    {"LightSalmon", MAKE_ARGB32(255, 255, 160, 122)},
    {"LightSeaGreen", MAKE_ARGB32(255, 32, 178, 170)},
    {"LightSkyBlue", MAKE_ARGB32(255, 135, 206, 250)},
    {"LightSlateBlue", MAKE_ARGB32(255, 132, 112, 255)},
    {"LightSlateGray", MAKE_ARGB32(255, 119, 136, 153)},
    {"LightSteelBlue", MAKE_ARGB32(255, 124, 152, 211)},
    {"LightYellow", MAKE_ARGB32(255, 255, 255, 224)},
    {"LimeGreen", MAKE_ARGB32(255, 0, 175, 20)},
    {"linen", MAKE_ARGB32(255, 250, 240, 230)},
    {"Magenta", MAKE_ARGB32(255, 255, 0, 255)},
    {"Maroon", MAKE_ARGB32(255, 143, 0, 82)},
    {"MediumAquamarine", MAKE_ARGB32(255, 0, 147, 143)},
    {"MediumBlue", MAKE_ARGB32(255, 50, 50, 204)},
    {"MediumForestGreen", MAKE_ARGB32(255, 50, 129, 75)},
    {"MediumGoldenrod", MAKE_ARGB32(255, 209, 193, 102)},
    {"MediumOrchid", MAKE_ARGB32(255, 189, 82, 189)},
    {"MediumPurple", MAKE_ARGB32(255, 147, 112, 219)},
    {"MediumSeaGreen", MAKE_ARGB32(255, 52, 119, 102)},
    {"MediumSlateBlue", MAKE_ARGB32(255, 106, 106, 141)},
    {"MediumSpringGreen", MAKE_ARGB32(255, 35, 142, 35)},
    {"MediumTurquoise", MAKE_ARGB32(255, 0, 210, 210)},
    {"MediumVioletRed", MAKE_ARGB32(255, 213, 32, 121)},
    {"MidnightBlue", MAKE_ARGB32(255, 47, 47, 100)},
    {"MintCream", MAKE_ARGB32(255, 245, 255, 250)},
    {"MistyRose", MAKE_ARGB32(255, 255, 228, 225)},
    {"moccasin", MAKE_ARGB32(255, 255, 228, 181)},
    {"NavajoWhite", MAKE_ARGB32(255, 255, 222, 173)},
    {"Navy", MAKE_ARGB32(255, 35, 35, 117)},
    {"NavyBlue", MAKE_ARGB32(255, 35, 35, 117)},
    {"None", MAKE_ARGB32(0, 0, 0, 1)},
    {"OldLace", MAKE_ARGB32(255, 253, 245, 230)},
    {"OliveDrab", MAKE_ARGB32(255, 107, 142, 35)},
    {"Orange", MAKE_ARGB32(255, 255, 135, 0)},
    {"OrangeRed", MAKE_ARGB32(255, 255, 69, 0)},
    {"Orchid", MAKE_ARGB32(255, 239, 132, 239)},
    {"PaleGoldenrod", MAKE_ARGB32(255, 238, 232, 170)},
    {"PaleGreen", MAKE_ARGB32(255, 115, 222, 120)},
    {"PaleTurquoise", MAKE_ARGB32(255, 175, 238, 238)},
    {"PaleVioletRed", MAKE_ARGB32(255, 219, 112, 147)},
    {"PapayaWhip", MAKE_ARGB32(255, 255, 239, 213)},
    {"PeachPuff", MAKE_ARGB32(255, 255, 218, 185)},
    {"peru", MAKE_ARGB32(255, 205, 133, 63)},
    {"Pink", MAKE_ARGB32(255, 255, 181, 197)},
    {"Plum", MAKE_ARGB32(255, 197, 72, 155)},
    {"PowderBlue", MAKE_ARGB32(255, 176, 224, 230)},
    {"purple", MAKE_ARGB32(255, 160, 32, 240)},
    {"Red", MAKE_ARGB32(255, 255, 0, 0)},
    {"RosyBrown", MAKE_ARGB32(255, 188, 143, 143)},
    {"RoyalBlue", MAKE_ARGB32(255, 65, 105, 225)},
    {"SaddleBrown", MAKE_ARGB32(255, 139, 69, 19)},
    {"Salmon", MAKE_ARGB32(255, 233, 150, 122)},
    {"SandyBrown", MAKE_ARGB32(255, 244, 164, 96)},
    {"SeaGreen", MAKE_ARGB32(255, 82, 149, 132)},
    {"seashell", MAKE_ARGB32(255, 255, 245, 238)},
    {"Sienna", MAKE_ARGB32(255, 150, 82, 45)},
    {"SkyBlue", MAKE_ARGB32(255, 114, 159, 255)},
    {"SlateBlue", MAKE_ARGB32(255, 126, 136, 171)},
    {"SlateGray", MAKE_ARGB32(255, 112, 128, 144)},
    {"snow", MAKE_ARGB32(255, 255, 250, 250)},
    {"SpringGreen", MAKE_ARGB32(255, 65, 172, 65)},
    {"SteelBlue", MAKE_ARGB32(255, 84, 112, 170)},
    {"Tan", MAKE_ARGB32(255, 222, 184, 135)},
    {"Thistle", MAKE_ARGB32(255, 216, 191, 216)},
    {"tomato", MAKE_ARGB32(255, 255, 99, 71)},
    {"Transparent", MAKE_ARGB32(0, 0, 0, 1)},
    {"Turquoise", MAKE_ARGB32(255, 25, 204, 223)},
    {"Violet", MAKE_ARGB32(255, 156, 62, 206)},
    {"VioletRed", MAKE_ARGB32(255, 243, 62, 150)},
    {"Wheat", MAKE_ARGB32(255, 245, 222, 179)},
    {"White", MAKE_ARGB32(255, 255, 255, 255)},
    {"WhiteSmoke", MAKE_ARGB32(255, 245, 245, 245)},
    {"Yellow", MAKE_ARGB32(255, 255, 255, 0)},
    {"YellowGreen", MAKE_ARGB32(255, 50, 216, 56)},
    {NULL,0}
};

/****************************************************************
 * Low level parsing code :
 ****************************************************************/
static inline char
get_xpm_char( ASXpmFile *xpm_file )
{
#ifdef HAVE_LIBXPM
	return '\0';
#else
	char c;
	if( xpm_file->curr_byte >= xpm_file->bytes_in )
	{
		if( xpm_file->bytes_in > AS_XPM_BUFFER_UNDO )
		{
			register char* src = &(xpm_file->buffer[xpm_file->bytes_in-AS_XPM_BUFFER_UNDO]);
			register char* dst = &(xpm_file->buffer[0]);
			register int i;
			for( i = 0 ; i < AS_XPM_BUFFER_UNDO ; i++ )
				dst[i] = src[i];
/*			xpm_file->bytes_in = AS_XPM_BUFFER_UNDO+fread( &(xpm_file->buffer[AS_XPM_BUFFER_UNDO]), 1, AS_XPM_BUFFER_SIZE, xpm_file->fp );*/
			xpm_file->bytes_in = xpm_file->data ?  AS_XPM_BUFFER_UNDO + strlen(*xpm_file->data) : 
                                 AS_XPM_BUFFER_UNDO+read( xpm_file->fd, &(xpm_file->buffer[AS_XPM_BUFFER_UNDO]), AS_XPM_BUFFER_SIZE );
			xpm_file->curr_byte = AS_XPM_BUFFER_UNDO ;
		}
		if( xpm_file->bytes_in <= AS_XPM_BUFFER_UNDO )
		{
			xpm_file->parse_state = XPM_Outside ;
			return '\0';
		}
	}
	c = xpm_file->buffer[xpm_file->curr_byte];
/*	fprintf( stderr, "curr byte = %d ( of %d ), char = %c\n", xpm_file->curr_byte, xpm_file->bytes_in, c ); */

	xpm_file->curr_byte++;
	return c;
#endif
}

static inline void
unget_xpm_char( ASXpmFile *xpm_file, char c )
{
#ifndef HAVE_LIBXPM
	if( xpm_file->curr_byte > 0 )
	{
		xpm_file->curr_byte--;
		xpm_file->buffer[xpm_file->curr_byte] = c;
	}
#endif
}

static inline void
skip_xpm_comments( ASXpmFile *xpm_file )
{
	char c;
	if((c=get_xpm_char(xpm_file)) != '*')
		unget_xpm_char(xpm_file, c);
	else
	{
		xpm_file->parse_state = XPM_InComments ;
		while( xpm_file->parse_state == XPM_InComments )
		{
			c = get_xpm_char(xpm_file);
			if( c == '*' )
				if( (c=get_xpm_char(xpm_file)) == '/' )
					xpm_file->parse_state--;
		}
	}
}

static Bool
seek_next_xpm_string( ASXpmFile *xpm_file )
{
	while( xpm_file->parse_state == XPM_InImage )
	{
		register char c;
		c = get_xpm_char(xpm_file);
		if( c == '/')
			skip_xpm_comments( xpm_file );
		else if( c == '"')
			xpm_file->parse_state = XPM_InString;
	}
	return (xpm_file->parse_state >= XPM_InString);
}

static Bool
seek_next_xpm_image( ASXpmFile *xpm_file )
{
	while( xpm_file->parse_state == XPM_InFile )
	{
		register char c;
		c = get_xpm_char(xpm_file);
		if( c == '/')
			skip_xpm_comments( xpm_file );
		else if( c == '{')
			xpm_file->parse_state = XPM_InImage;
	}
	return (xpm_file->parse_state >= XPM_InImage);
}

static Bool
read_next_xpm_string( ASXpmFile *xpm_file )
{
	char c;
	int i = 0;
	while( xpm_file->parse_state == XPM_InString )
	{
		c=get_xpm_char(xpm_file);
		if( c == '"' )
		{
			xpm_file->parse_state = XPM_InImage ;
			c = '\0';
		}

		if( i >= (int)xpm_file->str_buf_size )
		{
			xpm_file->str_buf = realloc( xpm_file->str_buf, xpm_file->str_buf_size+16+(xpm_file->str_buf_size>>2));
			xpm_file->str_buf_size += 16+(xpm_file->str_buf_size>>2) ;
		}
		xpm_file->str_buf[i++] = c;
	}
   xpm_file->curr_img_line++;

	return True;
}

#ifndef HAVE_LIBXPM
static Bool
parse_xpm_cmap_entry( ASXpmFile *xpm_file, char **colornames )
{
	register char *ptr ;
	int key ;
	Bool success = False ;

	if( xpm_file == NULL || xpm_file->str_buf == NULL )
		return False;
	for( key =0 ; key < 6 ; ++key )
		colornames[key] = NULL ;

	ptr = xpm_file->str_buf+xpm_file->bpp ;
	key = -1;
	do
	{
        while( !isspace((int)*ptr) && *ptr != '\0' ) ++ptr;
        while( isspace((int)*ptr) ) ++ptr;
		if( *ptr )
		{
			if( key >= 0 )
			{
				colornames[key] = ptr ;
				key = -1 ;
				success = True;
			}else
			{
    			if( *ptr == 'c' )				/* key #5: color visual */
					key = 5;
				else
				{
					if( *ptr == 's' ) 				/* key #1: symbol */
						key = 1;
    				else if( *ptr == 'm' )				/* key #2: mono visual */
						key = 2;
    				else if( *ptr == 'g' )				/* key #4: gray visual */
						key = 4;
					else
						key = 0;
				}
			}
		}
	}while( *ptr );
	return success;
}
#endif
/*************************************************************************
 * High level xpm reading interface ;
 *************************************************************************/
void
close_xpm_file( ASXpmFile **xpm_file )
{
	if( xpm_file )
		if( *xpm_file )
		{
			if( (*xpm_file)->fd )
				close( (*xpm_file)->fd );
			if( (*xpm_file)->str_buf && !(*xpm_file)->data)
				free( (*xpm_file)->str_buf );
#ifdef HAVE_LIBXPM
			XpmFreeXpmImage (&((*xpm_file)->xpmImage));
#else
			if( (*xpm_file)->buffer && !(*xpm_file)->data)
				free( (*xpm_file)->buffer );
#endif
			free_scanline(&((*xpm_file)->scl), True);
			if( (*xpm_file)->cmap )
				free( (*xpm_file)->cmap );
			if( (*xpm_file)->cmap2 )
			{
				register int i ;
				for( i = 0 ; i < 256 ; i++ )
					if( (*xpm_file)->cmap2[i] )
						free( (*xpm_file)->cmap2[i] );
				free( (*xpm_file)->cmap2 );
			}
			if( (*xpm_file)->cmap_name_xref )
				destroy_ashash( &((*xpm_file)->cmap_name_xref) );
#if 0
			memset( *xpm_file, 0x00, sizeof(ASXpmFile));
#endif
			free( *xpm_file );
			*xpm_file = NULL ;
		}
}

ASXpmFile*
open_xpm_file( const char *realfilename )
{
	ASXpmFile *xpm_file = NULL;
	if( realfilename )
	{
		Bool success = False ;
		int fd ;
		xpm_file = safecalloc( 1, sizeof(ASXpmFile));
#ifndef HAVE_LIBXPM
		fd = open( realfilename, O_RDONLY );
		if( fd >= 0 )
		{
			xpm_file->fd = fd;
			xpm_file->parse_state = XPM_InFile ;
			xpm_file->buffer = safemalloc(AS_XPM_BUFFER_UNDO+AS_XPM_BUFFER_SIZE+1);
         xpm_file->data = 0;
/*			xpm_file->bytes_in = AS_XPM_BUFFER_UNDO+fread( &(xpm_file->buffer[AS_XPM_BUFFER_UNDO]), 1, AS_XPM_BUFFER_SIZE, fp ); */
			xpm_file->bytes_in = AS_XPM_BUFFER_UNDO+read( fd, &(xpm_file->buffer[AS_XPM_BUFFER_UNDO]),  AS_XPM_BUFFER_SIZE );
			xpm_file->curr_byte = AS_XPM_BUFFER_UNDO ;
			if (get_xpm_string( xpm_file ) == XPM_Success)
				success = parse_xpm_header( xpm_file );
		}
#else                                          /* libXpm interface : */
		if( XpmReadFileToXpmImage ((char *)realfilename, &(xpm_file->xpmImage), NULL) == XpmSuccess)
		{
			fd = NULL ;
			xpm_file->width = xpm_file->xpmImage.width;
			xpm_file->height= xpm_file->xpmImage.height;
			xpm_file->cmap_size = xpm_file->xpmImage.ncolors;
			xpm_file->bpp = xpm_file->xpmImage.cpp;
			success = True;
		}
#endif
		if( !success ) {
			close_xpm_file( &xpm_file );
         return NULL;
		} else
		{
			if( xpm_file->width > MAX_IMPORT_IMAGE_SIZE )
				xpm_file->width = MAX_IMPORT_IMAGE_SIZE ;
			if( xpm_file->height > MAX_IMPORT_IMAGE_SIZE )
				xpm_file->height = MAX_IMPORT_IMAGE_SIZE ;
			if( xpm_file->bpp > MAX_XPM_BPP )
				xpm_file->bpp = MAX_XPM_BPP;
			prepare_scanline( xpm_file->width, 0, &(xpm_file->scl), False );
		}
	}
	return xpm_file ;
}


ASXpmFile*
open_xpm_data( const char **data )
{
        ASXpmFile *xpm_file = NULL;
        if( data )
        {
                Bool success = False ;

                xpm_file = safecalloc( 1, sizeof(ASXpmFile));
                xpm_file->data = (char**)data ;
                xpm_file->parse_state = XPM_InFile ;
                xpm_file->buffer = 0;
                xpm_file->curr_byte = AS_XPM_BUFFER_UNDO ;
                if( get_xpm_string( xpm_file ) == XPM_Success) {
                        success = parse_xpm_header( xpm_file );
               }

                if( !success ) {
                  close_xpm_file( &xpm_file );
                  return NULL;
                } else
                {
                        if( xpm_file->width > MAX_IMPORT_IMAGE_SIZE )
                                xpm_file->width = MAX_IMPORT_IMAGE_SIZE ;
                        if( xpm_file->height > MAX_IMPORT_IMAGE_SIZE )
                                xpm_file->height = MAX_IMPORT_IMAGE_SIZE ;
                        if( xpm_file->bpp > MAX_XPM_BPP )
                                xpm_file->bpp = MAX_XPM_BPP;
                        prepare_scanline( xpm_file->width, 0, &(xpm_file->scl), False );
                }
        }
        return xpm_file ;
}


ASXpmFile*
open_xpm_raw_data( const char *data )
{
	ASXpmFile *xpm_file = NULL;
	if( data )
	{
		Bool success = False ;

		xpm_file = safecalloc( 1, sizeof(ASXpmFile));
		xpm_file->data = (char**)&data ;
		xpm_file->parse_state = XPM_InFile ;
		xpm_file->buffer = (char*)data;
		xpm_file->curr_byte = AS_XPM_BUFFER_UNDO ;
      xpm_file->bytes_in = AS_XPM_BUFFER_UNDO + strlen(data);
		if( get_xpm_string( xpm_file )  == XPM_Success)
			success = parse_xpm_header( xpm_file );

		if( !success ) {
			close_xpm_file( &xpm_file );
         return NULL;
		} else
		{
			if( xpm_file->width > MAX_IMPORT_IMAGE_SIZE )
				xpm_file->width = MAX_IMPORT_IMAGE_SIZE ;
			if( xpm_file->height > MAX_IMPORT_IMAGE_SIZE )
				xpm_file->height = MAX_IMPORT_IMAGE_SIZE ;
			if( xpm_file->bpp > MAX_XPM_BPP )
				xpm_file->bpp = MAX_XPM_BPP;
			prepare_scanline( xpm_file->width, 0, &(xpm_file->scl), False );
		}
      xpm_file->curr_img_line = 0;
	}
	return xpm_file ;
}

ASXpmStatus
get_xpm_string( ASXpmFile *xpm_file )
{

   if( xpm_file == NULL )
      return XPM_Error;
   if( !xpm_file->buffer )
   {
      xpm_file->str_buf = xpm_file->data[xpm_file->curr_img_line];
      xpm_file->str_buf_size = 0 ;
      xpm_file->curr_img_line++;
      if( xpm_file->str_buf == NULL )
         return XPM_EndOfFile;
   }else
   {
      if( xpm_file->parse_state < XPM_InFile )
         return XPM_EndOfFile;
      if( xpm_file->parse_state < XPM_InImage )
      {
         if( !seek_next_xpm_image( xpm_file ) )
            return XPM_EndOfFile;
      }
      if( !seek_next_xpm_string( xpm_file ) )
      {
         xpm_file->curr_img++;
         return XPM_EndOfImage;
      }
      if( !read_next_xpm_string( xpm_file ))
         return XPM_Error;
      xpm_file->curr_img_line++;
   }
   return XPM_Success;
}

Bool
parse_xpm_header( ASXpmFile *xpm_file )
{
	register char *ptr ;
	if( xpm_file == NULL || xpm_file->str_buf == NULL )
		return False;

	ptr = xpm_file->str_buf ;
	while( isspace((int)*ptr) ) ++ptr;
	if( *ptr == '\0' )
		return False;
	xpm_file->width = atoi( ptr );
	while( !isspace((int)*ptr) && *ptr != '\0' ) ++ptr;
	while( isspace((int)*ptr) ) ++ptr;
	if( *ptr == '\0' )
		return False;
	xpm_file->height = atoi( ptr );
	while( !isspace((int)*ptr) && *ptr != '\0' ) ++ptr;
	while( isspace((int)*ptr) ) ++ptr;
	if( *ptr == '\0' )
		return False;
	xpm_file->cmap_size = atoi( ptr );
	while( !isspace((int)*ptr) && *ptr != '\0' ) ++ptr;
	while( isspace((int)*ptr) ) ++ptr;
	if( *ptr == '\0' )
		return False;
	xpm_file->bpp = atoi( ptr );
	return True;
}

ASImage *
create_xpm_image( ASXpmFile *xpm_file, int compression )
{
	ASImage *im = NULL;
	if( xpm_file != NULL && xpm_file->width > 0 && xpm_file->height > 0 )
	{
		im = create_asimage( xpm_file->width, xpm_file->height, compression );
	}
	return im;
}

static ARGB32
lookup_xpm_color( char **colornames, ASHashTable *xpm_color_names )
{
    ARGB32 color = 0;
	register int key = 5 ;
	do
	{
		if( colornames[key] )
		{
			if( *(colornames[key]) != '#' )
			{
				ASHashData hdata ;
                if( get_hash_item( xpm_color_names, AS_HASHABLE(colornames[key]), &hdata.vptr ) == ASH_Success )
				{
                    color = hdata.c32 ;
					LOCAL_DEBUG_OUT(" xpm color \"%s\" matched into 0x%lX", colornames[key], color );
					break;
				}
			}
			if( parse_argb_color( colornames[key], &color ) != colornames[key] )
			{
				LOCAL_DEBUG_OUT(" xpm color \"%s\" parsed into 0x%lX", colornames[key], color );
				break;
			}
			LOCAL_DEBUG_OUT(" xpm color \"%s\" is invalid :(", colornames[key] );
			/* unknown color - leaving it at 0 - that will make it transparent */
		}
	}while ( --key > 0);
	return color;
}

void
string_value_destroy (ASHashableValue value, void *data)
{
	if ((char*)value != NULL)
		free ((char*)value);
}

Bool
build_xpm_colormap( ASXpmFile *xpm_file )
{
	size_t real_cmap_size ;
	size_t i ;
#ifdef HAVE_LIBXPM
	XpmColor *xpm_cmap = (xpm_file)?xpm_file->xpmImage.colorTable: NULL ;
#endif
	static ASHashTable *xpm_color_names = NULL ;

	if( xpm_file == NULL )
	{
		destroy_ashash(&xpm_color_names);
		return False;
	}

	if( xpm_file->cmap_name_xref )
		destroy_ashash( &(xpm_file->cmap_name_xref) );
	if( xpm_file->cmap )
	{
		free( xpm_file->cmap );
		xpm_file->cmap = NULL;
	}
	real_cmap_size = xpm_file->cmap_size;
#ifdef HAVE_LIBXPM
	if( real_cmap_size > 1024 )
	{
		xpm_file->cmap = calloc( real_cmap_size, sizeof(ARGB32));
		if( xpm_file->cmap == NULL ) /* we don't want to bomb out if image is busted */
			real_cmap_size = 1024 ;
	}
	xpm_file->cmap = safecalloc( real_cmap_size, sizeof(ARGB32));
#else
	if( xpm_file->bpp == 1 )
	{
		real_cmap_size = 256 ;
		xpm_file->cmap = safecalloc( real_cmap_size, sizeof(ARGB32));
	}else if( xpm_file->bpp == 2 )
	{
		xpm_file->cmap2 = safecalloc( 256, sizeof(ARGB32*));
	}else
		xpm_file->cmap_name_xref = create_ashash( 0, string_hash_value,
													 string_compare,
													 string_value_destroy );
#endif
	if( xpm_color_names == NULL )
	{
		xpm_color_names = create_ashash( 0, casestring_hash_value, casestring_compare, NULL );
		for( i = 0 ; XpmRGB_Colors[i].name != NULL ; i++ )
			add_hash_item( xpm_color_names, (ASHashableValue)XpmRGB_Colors[i].name, (void*)((long)XpmRGB_Colors[i].argb) );
	}

	for( i = 0 ; i < xpm_file->cmap_size ; ++i )
	{
		ARGB32 color ;
#ifdef HAVE_LIBXPM
		if( i < real_cmap_size )
		{
			color = lookup_xpm_color((char**)&(xpm_cmap[i].string), xpm_color_names);
 LOCAL_DEBUG_OUT( "cmap[%d]: 0x%X\n",  i, color );
			xpm_file->cmap[i] = color;
			if( ARGB32_ALPHA8(color) != 0x00FF )
			{	
				if( ARGB32_ALPHA8(color) != 0 ) 
					xpm_file->full_alpha = True ;
				xpm_file->do_alpha = True ;
			}
		}
#else
		char *colornames[6] ;
		if( get_xpm_string( xpm_file ) != XPM_Success)
			break;
LOCAL_DEBUG_OUT( "cmap[%d]: \"%s\"\n",  i, xpm_file->str_buf );
		if( !parse_xpm_cmap_entry( xpm_file, &(colornames[0])))
			continue;
		color = lookup_xpm_color(&(colornames[0]), xpm_color_names);
LOCAL_DEBUG_OUT( "\t\tcolor = 0x%8.8lX\n",  color );
		if( ARGB32_ALPHA8(color) != 0x00FF )
			xpm_file->do_alpha = True ;
		if( xpm_file->bpp == 1 )
			xpm_file->cmap[(unsigned int)(xpm_file->str_buf[0])] = color ;
		else if( xpm_file->bpp == 2 )
		{
			ARGB32 **slot = &(xpm_file->cmap2[(unsigned int)(xpm_file->str_buf[0])]) ;
			if( *slot == NULL )
				*slot = safecalloc( 256, sizeof(ARGB32));
			(*slot)[(unsigned int)(xpm_file->str_buf[1])] = color ;
		}
		else if( i < real_cmap_size )
		{
			char *name = mystrndup(xpm_file->str_buf, xpm_file->bpp);
LOCAL_DEBUG_OUT( "\t\tname = \"%s\"\n", name );
			add_hash_item( xpm_file->cmap_name_xref, (ASHashableValue)name, (void*)((long)color) );
		}
#endif
	}
	xpm_file->cmap_size = real_cmap_size ;
	return True;
}

Bool
convert_xpm_scanline( ASXpmFile *xpm_file, unsigned int line )
{
	CARD32 *r = xpm_file->scl.red, *g = xpm_file->scl.green,
		   *b = xpm_file->scl.blue,*a = (xpm_file->do_alpha)?xpm_file->scl.alpha:NULL ;
	register int k = xpm_file->width ;
	ARGB32 *cmap = xpm_file->cmap ;
#ifdef HAVE_LIBXPM
	unsigned int *data = xpm_file->xpmImage.data+k*line ;
#else
	unsigned char *data ;
	if( get_xpm_string( xpm_file ) != XPM_Success)
		return False ;
	data = (unsigned char*)xpm_file->str_buf ;
#endif
	if( cmap )
	{
		while( --k >= 0 )
			if( data[k] < xpm_file->cmap_size )
			{
				register CARD32 c = cmap[data[k]] ;
				r[k] = ARGB32_RED8(c);
				g[k] = ARGB32_GREEN8(c);
				b[k] = ARGB32_BLUE8(c);
				if( a )
					a[k]  = ARGB32_ALPHA8(c);
			}
	}else if( xpm_file->cmap2 )
	{
		ARGB32 **cmap2 = xpm_file->cmap2 ;
		while( --k >= 0 )
		{
			ARGB32 *slot = cmap2[data[k<<1]] ;
			if( slot != NULL )
			{
				register CARD32 c = slot[data[(k<<1)+1]] ;
				r[k] = ARGB32_RED8(c);
				g[k] = ARGB32_GREEN8(c);
				b[k] = ARGB32_BLUE8(c);
				if( a )
					a[k]  = ARGB32_ALPHA8(c);
			}
		}
	}else if( xpm_file->cmap_name_xref )
	{
		char *pixel ;
		pixel = safemalloc( xpm_file->bpp+1);
		pixel[xpm_file->bpp] = '\0' ;
		data += (k-1)*xpm_file->bpp ;
		while( --k >= 0 )
		{
			register int i = xpm_file->bpp;
            ASHashData hdata = {0} ;
            CARD32 c = 0;
			while( --i >= 0 )
				pixel[i] = data[i] ;
			data -= xpm_file->bpp ;
            get_hash_item( xpm_file->cmap_name_xref, AS_HASHABLE(pixel), &hdata.vptr );
            /* on 64 bit system we must do that since pointers are 64 bit */
            c = hdata.c32;
			r[k] = ARGB32_RED8(c);
			g[k] = ARGB32_GREEN8(c);
			b[k] = ARGB32_BLUE8(c);
			if( a )
				a[k]  = ARGB32_ALPHA8(c);
		}
		free( pixel );
	}
	return True;
}

/**********************************************************************/
/* XPM writing :                                                      */
/**********************************************************************/

ASXpmCharmap*
build_xpm_charmap( ASColormap *cmap, Bool has_alpha, ASXpmCharmap *reusable_memory )
{
	ASXpmCharmap *xpm_cmap = reusable_memory ;
	char *ptr ;
	int i ;
	int rem ;

	xpm_cmap->count = cmap->count+((has_alpha)?1:0) ;

	xpm_cmap->cpp = 0 ;
	for( rem = xpm_cmap->count ; rem > 0 ; rem = rem/MAXPRINTABLE )
		++(xpm_cmap->cpp) ;
	ptr = xpm_cmap->char_code = safemalloc(xpm_cmap->count*(xpm_cmap->cpp+1)) ;
	for( i = 0 ; i < (int)xpm_cmap->count ; i++ )
	{
		register int k = xpm_cmap->cpp ;
		rem = i ;
		ptr[k] = '\0' ;
		while( --k >= 0 )
		{
			ptr[k] = printable[rem%MAXPRINTABLE] ;
			rem /= MAXPRINTABLE ;
		}
		ptr += xpm_cmap->cpp+1 ;
	}

	return xpm_cmap;
}

void destroy_xpm_charmap( ASXpmCharmap *xpm_cmap, Bool reusable )
{
	if( xpm_cmap )
	{
		if( xpm_cmap->char_code )
			free( xpm_cmap->char_code );
		if( !reusable )
			free( xpm_cmap );
	}
}

#endif /* HAVE_XPM */
