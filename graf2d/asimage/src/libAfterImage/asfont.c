/*
 * Copyright (c) 2001 Sasha Vasko <sasha at aftercode.net>
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
#undef LOCAL_DEBUG
#undef DO_CLOCKING

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif


#define DO_X11_ANTIALIASING
#define DO_2STEP_X11_ANTIALIASING
#define DO_3STEP_X11_ANTIALIASING
#define X11_AA_HEIGHT_THRESHOLD 10
#define X11_2STEP_AA_HEIGHT_THRESHOLD 15
#define X11_3STEP_AA_HEIGHT_THRESHOLD 15


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
#include <string.h>
#include <stdio.h>
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

#ifdef HAVE_FREETYPE
# ifdef HAVE_FT2BUILD_H
#  include <ft2build.h>
#  include FT_FREETYPE_H
# endif
# ifdef HAVE_FREETYPE_FREETYPE
#  include <freetype/freetype.h>
# else
#  include <freetype.h>
# endif
# if (FREETYPE_MAJOR == 2) && ((FREETYPE_MINOR == 0) || ((FREETYPE_MINOR == 1) && (FREETYPE_PATCH < 3)))
#  define FT_KERNING_DEFAULT ft_kerning_default
# endif
#endif

#define INCLUDE_ASFONT_PRIVATE

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "asfont.h"
#include "asimage.h"
#include "asvisual.h"

#ifdef HAVE_XRENDER
#include <X11/extensions/Xrender.h>
#endif

#undef MAX_GLYPHS_PER_FONT


/*********************************************************************************/
/* TrueType and X11 font management functions :   								 */
/*********************************************************************************/

/*********************************************************************************/
/* construction destruction miscelanea:			   								 */
/*********************************************************************************/

void asfont_destroy (ASHashableValue value, void *data);

ASFontManager *
create_font_manager( Display *dpy, const char * font_path, ASFontManager *reusable_memory )
{
	ASFontManager *fontman = reusable_memory;
	if( fontman == NULL )
		fontman = safecalloc( 1, sizeof(ASFontManager));
	else
		memset( fontman, 0x00, sizeof(ASFontManager));

	fontman->dpy = dpy ;
	if( font_path )
		fontman->font_path = mystrdup( font_path );

#ifdef HAVE_FREETYPE
	if( !FT_Init_FreeType( &(fontman->ft_library)) )
		fontman->ft_ok = True ;
	else
		show_error( "Failed to initialize FreeType library - TrueType Fonts support will be disabled!");
LOCAL_DEBUG_OUT( "Freetype library is %p", fontman->ft_library );
#endif

	fontman->fonts_hash = create_ashash( 7, string_hash_value, string_compare, asfont_destroy );

	return fontman;
}

void
destroy_font_manager( ASFontManager *fontman, Bool reusable )
{
	if( fontman )
	{

        destroy_ashash( &(fontman->fonts_hash) );

#ifdef HAVE_FREETYPE
		FT_Done_FreeType( fontman->ft_library);
		fontman->ft_ok = False ;
#endif
		if( fontman->font_path )
			free( fontman->font_path );

		if( !reusable )
			free( fontman );
		else
			memset( fontman, 0x00, sizeof(ASFontManager));
	}
}

#ifdef HAVE_FREETYPE
static int load_freetype_glyphs( ASFont *font );
#endif
#ifndef X_DISPLAY_MISSING
static int load_X11_glyphs( Display *dpy, ASFont *font, XFontStruct *xfs );
#endif


static ASFont*
open_freetype_font_int( ASFontManager *fontman, const char *font_string, int face_no, int size, Bool verbose, ASFlagType flags)
{
	ASFont *font = NULL ;
#ifdef HAVE_FREETYPE
	if( fontman && fontman->ft_ok )
	{
		char *realfilename;
		FT_Face face ;
		LOCAL_DEBUG_OUT( "looking for \"%s\" in \"%s\"", font_string, fontman->font_path );
		if( (realfilename = find_file( font_string, fontman->font_path, R_OK )) == NULL )
		{/* we might have face index specifier at the end of the filename */
			char *tmp = mystrdup( font_string );
			register int i = 0;
			while(tmp[i] != '\0' ) ++i ;
			while( --i >= 0 )
				if( !isdigit( tmp[i] ) )
				{
					if( tmp[i] == '.' )
					{
						face_no = atoi( &tmp[i+1] );
						tmp[i] = '\0' ;
					}
					break;
				}
			if( i >= 0 && font_string[i] != '\0' )
				realfilename = find_file( tmp, fontman->font_path, R_OK );
			free( tmp );
		}

		if( realfilename )
		{
			face = NULL ;
LOCAL_DEBUG_OUT( "font file found : \"%s\", trying to load face #%d, using library %p", realfilename, face_no, fontman->ft_library );
			if( FT_New_Face( fontman->ft_library, realfilename, face_no, &face ) )
			{
LOCAL_DEBUG_OUT( "face load failed.%s", "" );

				if( face_no  > 0  )
				{
					show_warning( "face %d is not available in font \"%s\" - falling back to first available.", face_no, realfilename );
					FT_New_Face( fontman->ft_library, realfilename, 0, &face );
				}
			}
LOCAL_DEBUG_OUT( "face found : %p", face );
			if( face != NULL )
			{
#ifdef MAX_GLYPHS_PER_FONT
				if( face->num_glyphs >  MAX_GLYPHS_PER_FONT )
					show_error( "Font \"%s\" contains too many glyphs - %d. Max allowed is %d", realfilename, face->num_glyphs, MAX_GLYPHS_PER_FONT );
				else
#endif
				{
					font = safecalloc( 1, sizeof(ASFont));
					font->magic = MAGIC_ASFONT ;
					font->fontman = fontman;
					font->type = ASF_Freetype ;
					font->flags = flags ;
					font->ft_face = face ;
					if( FT_HAS_KERNING( face ) )
					{	
						set_flags( font->flags, ASF_HasKerning );
						/* fprintf( stderr, "@@@@@@@font %s has kerning!!!\n", realfilename );*/
					}/*else
						fprintf( stderr, "@@@@@@@font %s don't have kerning!!!\n", realfilename ); */
					/* lets confine the font to square cell */
					FT_Set_Pixel_Sizes( font->ft_face, size, size );
					/* but let make our own cell width smaller then height */
					font->space_size = size*2/3 ;
	   				load_freetype_glyphs( font );
				}
			}else if( verbose )
				show_error( "FreeType library failed to load font \"%s\"", realfilename );

			if( realfilename != font_string )
				free( realfilename );
		}
	}
#endif
	return font;
}

ASFont*
open_freetype_font( ASFontManager *fontman, const char *font_string, int face_no, int size, Bool verbose)
{
	return open_freetype_font_int( fontman, font_string, face_no, size, verbose, 0);	
}

static ASFont*
open_X11_font_int( ASFontManager *fontman, const char *font_string, ASFlagType flags)
{
	ASFont *font = NULL ;
#ifndef X_DISPLAY_MISSING
/* 
    #ifdef I18N
     TODO: we have to use FontSet and loop through fonts instead filling
           up 2 bytes per character table with glyphs 
    #else 
*/
    /* for now assume ISO Latin 1 encoding */
	XFontStruct *xfs ;
	if( fontman->dpy == NULL ) 
		return NULL;
	if( (xfs = XLoadQueryFont( fontman->dpy, font_string )) == NULL )
	{
		show_warning( "failed to load X11 font \"%s\". Sorry about that.", font_string );
		return NULL;
	}
	font = safecalloc( 1, sizeof(ASFont));
	font->magic = MAGIC_ASFONT ;
	font->fontman = fontman;
	font->type = ASF_X11 ;
	font->flags = flags ;
	load_X11_glyphs( fontman->dpy, font, xfs );
	XFreeFont( fontman->dpy, xfs );
#endif /* #ifndef X_DISPLAY_MISSING */
	return font;
}

ASFont*
open_X11_font( ASFontManager *fontman, const char *font_string)
{
	return open_X11_font_int( fontman, font_string, 0);
}

ASFont*
get_asfont( ASFontManager *fontman, const char *font_string, int face_no, int size, ASFontType type_and_flags )
{
	ASFont *font = NULL ;
	Bool freetype = False ;
	int type = type_and_flags&ASF_TypeMask ;
	if( face_no >= 100 )
		face_no = 0 ;
	if( size >= 1000 )
		size = 999 ;

	if( fontman && font_string )
	{
		ASHashData hdata = { 0 };
		if( get_hash_item( fontman->fonts_hash, AS_HASHABLE((char*)font_string), &hdata.vptr) != ASH_Success )
		{
			char *ff_name ;
			int len = strlen( font_string)+1 ;
			len += ((size>=100)?3:2)+1 ;
			len += ((face_no>=10)?2:1)+1 ;
			ff_name = safemalloc( len );
			sprintf( ff_name, "%s$%d$%d", font_string, size, face_no );
			if( get_hash_item( fontman->fonts_hash, AS_HASHABLE((char*)ff_name), &hdata.vptr) != ASH_Success )
			{	/* not loaded just yet - lets do it :*/
				if( type == ASF_Freetype || type == ASF_GuessWho )
					font = open_freetype_font_int( fontman, font_string, face_no, size, (type == ASF_Freetype), get_flags(type_and_flags,~ASF_TypeMask));
				if( font == NULL && type != ASF_Freetype )
				{/* don't want to try and load font as X11 unless requested to do so */
					font = open_X11_font_int( fontman, font_string, get_flags(type_and_flags,~ASF_TypeMask) );
				}else
					freetype = True ;
				if( font != NULL )
				{
					if( freetype )
					{
						font->name = ff_name ;
						ff_name = NULL ;
					}else
						font->name = mystrdup( font_string );
					add_hash_item( fontman->fonts_hash, AS_HASHABLE((char*)font->name), font);
				}
			}
			if( ff_name != NULL )
				free( ff_name );
		}

		if( font == NULL )
			font = hdata.vptr ;

		if( font )
			font->ref_count++ ;
	}
	return font;
}

ASFont*
dup_asfont( ASFont *font )
{
	if( font && font->fontman )
		font->ref_count++ ;
	else
		font = NULL;
	return font;
}

int
release_font( ASFont *font )
{
	int res = -1 ;
	if( font )
	{
		if( font->magic == MAGIC_ASFONT )
		{
			if( --(font->ref_count) < 0 )
			{
				ASFontManager *fontman = font->fontman ;
				if( fontman )
					remove_hash_item(fontman->fonts_hash, (ASHashableValue)(char*)font->name, NULL, True);
			}else
				res = font->ref_count ;
		}
	}
	return res ;
}

static inline void
free_glyph_data( register ASGlyph *asg )
{
    if( asg->pixmap )
        free( asg->pixmap );
/*fprintf( stderr, "\t\t%p\n", asg->pixmap );*/
    asg->pixmap = NULL ;
}

static void
destroy_glyph_range( ASGlyphRange **pgr )
{
	ASGlyphRange *gr = *pgr;
	if( gr )
	{
		*pgr = gr->above ;
        if( gr->below )
			gr->below->above = gr->above ;
        if( gr->above )
			gr->above->below = gr->below ;
        if( gr->glyphs )
		{
            int max_i = ((int)gr->max_char-(int)gr->min_char)+1 ;
            register int i = -1 ;
/*fprintf( stderr, " max_char = %d, min_char = %d, i = %d", gr->max_char, gr->min_char, max_i);*/
            while( ++i < max_i )
            {
/*fprintf( stderr, "%d >", i );*/
				free_glyph_data( &(gr->glyphs[i]) );
            }
            free( gr->glyphs );
			gr->glyphs = NULL ;
		}
		free( gr );
	}
}

static void
destroy_font( ASFont *font )
{
	if( font )
	{
#ifdef HAVE_FREETYPE
        if( font->type == ASF_Freetype && font->ft_face )
			FT_Done_Face(font->ft_face);
#endif
        if( font->name )
			free( font->name );
        while( font->codemap )
			destroy_glyph_range( &(font->codemap) );
        free_glyph_data( &(font->default_glyph) );
        if( font->locale_glyphs )
			destroy_ashash( &(font->locale_glyphs) );
        font->magic = 0 ;
		free( font );
	}
}

void
asglyph_destroy (ASHashableValue value, void *data)
{
	if( data )
	{
		free_glyph_data( (ASGlyph*)data );
		free( data );
	}
}

void
asfont_destroy (ASHashableValue value, void *data)
{
	if( data )
	{
	    char* cval = (char*)value ;
        if( ((ASMagic*)data)->magic == MAGIC_ASFONT )
        {
            if( cval == ((ASFont*)data)->name )
                cval = NULL ;          /* name is freed as part of destroy_font */
/*              fprintf( stderr,"freeing font \"%s\"...", (char*) value ); */
              destroy_font( (ASFont*)data );
/*              fprintf( stderr,"   done.\n"); */
        }
        if( cval )
            free( cval );
    }
}

/*************************************************************************/
/* Low level bitmap handling - compression and antialiasing              */
/*************************************************************************/

static unsigned char *
compress_glyph_pixmap( unsigned char *src, unsigned char *buffer,
                       unsigned int width, unsigned int height,
					   int src_step )
{
	unsigned char *pixmap ;
	register unsigned char *dst = buffer ;
	register int k = 0, i = 0 ;
	int count = -1;
	unsigned char last = src[0];
/* new way: if its FF we encode it as 01rrrrrr where rrrrrr is repitition count
 * if its 0 we encode it as 00rrrrrr. Any other symbol we bitshift right by 1
 * and then store it as 1sssssss where sssssss are topmost sugnificant bits.
 * Note  - single FFs and 00s are encoded as any other character. Its been noted
 * that in 99% of cases 0 or FF repeats, and very seldom anything else repeats
 */
	while ( height)
	{
		if( src[k] != last || (last != 0 && last != 0xFF) || count >= 63 )
		{
			if( count == 0 )
				dst[i++] = (last>>1)|0x80;
			else if( count > 0 )
			{
				if( last == 0xFF )
					count |= 0x40 ;
				dst[i++] = count;
				count = 0 ;
			}else
				count = 0 ;
			last = src[k] ;
		}else
		 	count++ ;
/*fprintf( stderr, "%2.2X ", src[k] ); */
		if( ++k >= (int)width )
		{
/*			fputc( '\n', stderr ); */
			--height ;
			k = 0 ;
			src += src_step ;
		}
	}
	if( count == 0 )
		dst[i++] = (last>>1)|0x80;
	else
	{
		if( last == 0xFF )
			count |= 0x40 ;
		dst[i++] = count;
	}
    pixmap  = safemalloc( i/*+(32-(i&0x01F) )*/);
/*fprintf( stderr, "pixmap alloced %p size %d(%d)", pixmap, i, i+(32-(i&0x01F) )); */
	memcpy( pixmap, buffer, i );

	return pixmap;
}

#ifdef DO_X11_ANTIALIASING
void
antialias_glyph( unsigned char *buffer, unsigned int width, unsigned int height )
{
	unsigned char *row1, *row2 ;
	register unsigned char *row ;
	register int x;
	int y;

	row1 = &(buffer[0]);
	row = &(buffer[width]);
	row2 = &(buffer[width+width]);
	for( x = 1 ; x < (int)width-1 ; x++ )
		if( row1[x] == 0 )
		{/* antialiasing here : */
			unsigned int c = (unsigned int)row[x]+
							(unsigned int)row1[x-1]+
							(unsigned int)row1[x+1];
			if( c >= 0x01FE )  /* we cut off secondary aliases */
				row1[x] = c>>2;
		}
	for( y = 1 ; y < (int)height-1 ; y++ )
	{
		if( row[0] == 0 )
		{/* antialiasing here : */
			unsigned int c = (unsigned int)row1[0]+
							(unsigned int)row[1]+
							(unsigned int)row2[0];
			if( c >= 0x01FE )  /* we cut off secondary aliases */
				row[0] = c>>2;
		}
		for( x = 1 ; x < (int)width-1 ; x++ )
		{
			if( row[x] == 0 )
			{/* antialiasing here : */
				unsigned int c = (unsigned int)row1[x]+
								(unsigned int)row[x-1]+
								(unsigned int)row[x+1]+
								(unsigned int)row2[x];
				if( row1[x] != 0 && row[x-1] != 0 && row[x+1] != 0 && row2[x] != 0 &&
					c >= 0x01FE )
					row[x] = c>>3;
				else if( c >= 0x01FE )  /* we cut off secondary aliases */
					row[x] = c>>2;
			}
		}
		if( row[x] == 0 )
		{/* antialiasing here : */
			unsigned int c = (unsigned int)row1[x]+
							(unsigned int)row[x-1]+
							(unsigned int)row2[x];
			if( c >= 0x01FE )  /* we cut off secondary aliases */
				row[x] = c>>2;
		}
		row  += width ;
		row1 += width ;
		row2 += width ;
	}
	for( x = 1 ; x < (int)width-1 ; x++ )
		if( row[x] == 0 )
		{/* antialiasing here : */
			unsigned int c = (unsigned int)row1[x]+
							(unsigned int)row[x-1]+
							(unsigned int)row[x+1];
			if( c >= 0x01FE )  /* we cut off secondary aliases */
				row[x] = c>>2;
		}
#ifdef DO_2STEP_X11_ANTIALIASING
	if( height  > X11_2STEP_AA_HEIGHT_THRESHOLD )
	{
		row1 = &(buffer[0]);
		row = &(buffer[width]);
		row2 = &(buffer[width+width]);
		for( y = 1 ; y < (int)height-1 ; y++ )
		{
			for( x = 1 ; x < (int)width-1 ; x++ )
			{
				if( row[x] == 0 )
				{/* antialiasing here : */
					unsigned int c = (unsigned int)row1[x]+
									(unsigned int)row[x-1]+
									(unsigned int)row[x+1]+
									(unsigned int)row2[x];
					if( row1[x] != 0 && row[x-1] != 0 && row[x+1] != 0 && row2[x] != 0
						&& c >= 0x00FF+0x007F)
						row[x] = c>>3;
					else if( (c >= 0x00FF+0x007F)|| c == 0x00FE )  /* we cut off secondary aliases */
						row[x] = c>>2;
				}
			}
			row  += width ;
			row1 += width ;
			row2 += width ;
		}
	}
#endif
#ifdef DO_3STEP_X11_ANTIALIASING
	if( height  > X11_3STEP_AA_HEIGHT_THRESHOLD )
	{
		row1 = &(buffer[0]);
		row = &(buffer[width]);
		row2 = &(buffer[width+width]);
		for( y = 1 ; y < (int)height-1 ; y++ )
		{
			for( x = 1 ; x < (int)width-1 ; x++ )
			{
				if( row[x] == 0xFF )
				{/* antialiasing here : */
					if( row1[x] < 0xFE || row2[x] < 0xFE )
						if( row[x+1] < 0xFE || row[x-1] < 0xFE )
							row[x] = 0xFE;
				}
			}
			row  += width ;
			row1 += width ;
			row2 += width ;
		}
		row = &(buffer[width]);
		for( y = 1 ; y < (int)height-1 ; y++ )
		{
			for( x = 1 ; x < (int)width-1 ; x++ )
				if( row[x] == 0xFE )
					row[x] = 0xBF ;
			row  += width ;
		}
	}

#endif
}
#endif

static void 
scale_down_glyph_width( unsigned char *buffer, int from_width, int to_width, int height )
{
    int smaller = to_width;
    int bigger  = from_width;
	register int i = 0, k = 0, l;
	/*fprintf( stderr, "scaling glyph down from %d to %d\n", from_width, to_width );*/
    /* LOCAL_DEBUG_OUT( "smaller %d, bigger %d, eps %d", smaller, bigger, eps ); */
    /* now using Bresengham algoritm to fiill the scales :
	 * since scaling is merely transformation
	 * from 0:bigger space (x) to 0:smaller space(y)*/
	for( l = 0 ; l < height ; ++l )
	{
		unsigned char *ptr = &(buffer[l*from_width]) ;
		CARD32 sum = 0;	  
		int count = 0 ;
		int eps = -bigger/2;
		
		k = 0 ;
		for ( i = 0 ; i < bigger ; i++ )
		{
			/* add next elem here */
			sum += (unsigned int)ptr[i] ;
			++count ;			   
			eps += smaller;
	        
			if( eps+eps >= bigger )
			{
				/* divide here */
				/*fprintf( stderr, "i=%d, k=%d, sum=%d, count=%d\n", i, k, sum, count );*/
				ptr[k] = ( count > 1 )?sum/count:sum ;
				sum = 0 ;
				count = 0 ;
				++k ;
				eps -= bigger ;
			}		
		}	  
	}
	/* now we need to compress the pixmap */
	
	l = to_width ;
	k = from_width ;
	do
	{
		for( i = 0 ; i < to_width; ++i )
			buffer[l+i] = buffer[k+i];			
		l += to_width ; 
		k += from_width ;
	}while( l < to_width*height );
}



/*********************************************************************************/
/* encoding/locale handling						   								 */
/*********************************************************************************/

/* Now, this is the mess, I know :
 * Internally we store everything in current locale;
 * WE then need to convert it into Unicode 4 byte codes
 *
 * TODO: think about incoming data - does it has to be made local friendly ???
 * Definately
 */

#ifndef X_DISPLAY_MISSING
static ASGlyphRange*
split_X11_glyph_range( unsigned int min_char, unsigned int max_char, XCharStruct *chars )
{
	ASGlyphRange *first = NULL, **r = &first;
    int c = 0, delta = (max_char-min_char)+1;
LOCAL_DEBUG_CALLER_OUT( "min_char = %u, max_char = %u, chars = %p", min_char, max_char, chars );
	while( c < delta )
	{
		while( c < delta && chars[c].width == 0 ) ++c;

		if( c < delta )
		{
			*r = safecalloc( 1, sizeof(ASGlyphRange));
			(*r)->min_char = c+min_char ;
			while( c < delta && chars[c].width  != 0 ) ++c ;
			(*r)->max_char = (c-1)+min_char;
LOCAL_DEBUG_OUT( "created glyph range from %lu to %lu", (*r)->min_char, (*r)->max_char );
			r = &((*r)->above);
		}
	}
	return first;
}

void
load_X11_glyph_range( Display *dpy, ASFont *font, XFontStruct *xfs, size_t char_offset,
													  unsigned char byte1,
                                                      unsigned char min_byte2,
													  unsigned char max_byte2, GC *gc )
{
	ASGlyphRange  *all, *r ;
	unsigned long  min_char = (byte1<<8)|min_byte2;
	unsigned char *buffer, *compressed_buf ;
	unsigned int   height = xfs->ascent+xfs->descent ;
	static XGCValues gcv;

	buffer = safemalloc( xfs->max_bounds.width*height*2);
	compressed_buf = safemalloc( xfs->max_bounds.width*height*4);
	all = split_X11_glyph_range( min_char, (byte1<<8)|max_byte2, &(xfs->per_char[char_offset]));
	for( r = all ; r != NULL ; r = r->above )
	{
		XCharStruct *chars = &(xfs->per_char[char_offset+r->min_char-min_char]);
        int len = ((int)r->max_char-(int)r->min_char)+1;
		unsigned char char_base = r->min_char&0x00FF;
		register int i ;
		Pixmap p;
		XImage *xim;
		unsigned int total_width = 0 ;
		int pen_x = 0;
LOCAL_DEBUG_OUT( "loading glyph range of %lu-%lu", r->min_char, r->max_char );
		r->glyphs = safecalloc( len, sizeof(ASGlyph) );
		for( i = 0 ; i < len ; i++ )
		{
			int w = chars[i].rbearing - chars[i].lbearing ;
			r->glyphs[i].lead = chars[i].lbearing ;
			r->glyphs[i].width = MAX(w,(int)chars[i].width) ;
			r->glyphs[i].step = chars[i].width;
			total_width += r->glyphs[i].width ;
			if( chars[i].lbearing > 0 )
				total_width += chars[i].lbearing ;
		}
		p = XCreatePixmap( dpy, DefaultRootWindow(dpy), total_width, height, 1 );
		if( *gc == NULL )
		{
			gcv.font = xfs->fid;
			gcv.foreground = 1;
			*gc = XCreateGC( dpy, p, GCFont|GCForeground, &gcv);
		}else
			XSetForeground( dpy, *gc, 1 );
		XFillRectangle( dpy, p, *gc, 0, 0, total_width, height );
		XSetForeground( dpy, *gc, 0 );

		for( i = 0 ; i < len ; i++ )
		{
			XChar2b test_char ;
			int offset = MIN(0,(int)chars[i].lbearing);

			test_char.byte1 = byte1 ;
			test_char.byte2 = char_base+i ;
			/* we cannot draw string at once since in some fonts charcters may
			 * overlap each other : */
			XDrawImageString16( dpy, p, *gc, pen_x-offset, xfs->ascent, &test_char, 1 );
			pen_x += r->glyphs[i].width ;
			if( chars[i].lbearing > 0 )
				pen_x += chars[i].lbearing ;
		}
		/*XDrawImageString( dpy, p, *gc, 0, xfs->ascent, test_str_char, len );*/
		xim = XGetImage( dpy, p, 0, 0, total_width, height, 0xFFFFFFFF, ZPixmap );
		XFreePixmap( dpy, p );
		pen_x = 0 ;
		for( i = 0 ; i < len ; i++ )
		{
			register int x, y ;
			int width = r->glyphs[i].width;
			unsigned char *row = &(buffer[0]);

			if( chars[i].lbearing > 0 )
				pen_x += chars[i].lbearing ;
			for( y = 0 ; y < height ; y++ )
			{
				for( x = 0 ; x < width ; x++ )
				{
/*					fprintf( stderr, "glyph %d (%c): (%d,%d) 0x%X\n", i, (char)(i+r->min_char), x, y, XGetPixel( xim, pen_x+x, y ));*/
					/* remember default GC colors are black on white - 0 on 1 - and we need
					* quite the opposite - 0xFF on 0x00 */
					row[x] = ( XGetPixel( xim, pen_x+x, y ) != 0 )? 0x00:0xFF;
				}
				row += width;
			}

#ifdef DO_X11_ANTIALIASING
			if( height > X11_AA_HEIGHT_THRESHOLD )
				antialias_glyph( buffer, width, height );
#endif
			if( get_flags( font->flags, ASF_Monospaced ) )
			{
				if( r->glyphs[i].lead > 0 && (int)width + (int)r->glyphs[i].lead > (int)font->space_size )
					if( r->glyphs[i].lead > (int)font->space_size/8 ) 
						r->glyphs[i].lead = (int)font->space_size/8 ;
				if( (int)width + r->glyphs[i].lead > (int)font->space_size )
				{	
					r->glyphs[i].width = (int)font->space_size - r->glyphs[i].lead ;
/*					fprintf(stderr, "lead = %d, space_size = %d, width = %d, to_width = %d\n",
							r->glyphs[i].lead, font->space_size, width, r->glyphs[i].width ); */
					scale_down_glyph_width( buffer, width, r->glyphs[i].width, height );
				}
				/*else
				{
					fprintf(stderr, "lead = %d, space_size = %d, width = %d\n",
							r->glyphs[i].lead, font->space_size, width );
				}	 */
				r->glyphs[i].step = font->space_size ;
			}	 
			r->glyphs[i].pixmap = compress_glyph_pixmap( buffer, compressed_buf, r->glyphs[i].width, height, r->glyphs[i].width );
			r->glyphs[i].height = height ;
			r->glyphs[i].ascend = xfs->ascent ;
			r->glyphs[i].descend = xfs->descent ;
LOCAL_DEBUG_OUT( "glyph %u(range %lu-%lu) (%c) is %dx%d ascend = %d, lead = %d",  i, r->min_char, r->max_char, (char)(i+r->min_char), r->glyphs[i].width, r->glyphs[i].height, r->glyphs[i].ascend, r->glyphs[i].lead );
			pen_x += width ;
		}
		if( xim )
			XDestroyImage( xim );
	}
LOCAL_DEBUG_OUT( "done loading glyphs. Attaching set of glyph ranges to the codemap...%s", "" );
	if( all != NULL )
	{
		if( font->codemap == NULL )
			font->codemap = all ;
		else
		{
			for( r = font->codemap ; r != NULL ; r = r->above )
			{
				if( r->min_char > all->min_char )
				{
					if( r->below )
						r->below->above = all ;
					r->below = all ;
					while ( all->above != NULL )
						all = all->above ;
					all->above = r ;
					r->below = all ;
					break;
				}
				all->below = r ;
			}
			if( r == NULL && all->below->above == NULL )
				all->below->above = all ;
		}
	}
	free( buffer ) ;
	free( compressed_buf ) ;
LOCAL_DEBUG_OUT( "all don%s", "" );
}


void
make_X11_default_glyph( ASFont *font, XFontStruct *xfs )
{
	unsigned char *buf, *compressed_buf ;
	int width, height ;
	int x, y;
	unsigned char *row ;


	height = xfs->ascent+xfs->descent ;
	width = xfs->max_bounds.width ;

	if( height <= 0 ) height = 4;
	if( width <= 0 ) width = 4;
	buf = safecalloc( height*width, sizeof(unsigned char) );
	compressed_buf = safemalloc( height*width*2 );
	row = buf;
	for( x = 0 ; x < width ; ++x )
		row[x] = 0xFF;
	for( y = 1 ; y < height-1 ; ++y )
	{
		row += width ;
		row[0] = 0xFF ; row[width-1] = 0xFF ;
	}
	for( x = 0 ; x < width ; ++x )
		row[x] = 0xFF;
	font->default_glyph.pixmap = compress_glyph_pixmap( buf, compressed_buf, width, height, width );
	font->default_glyph.width = width ;
	font->default_glyph.step = width ;
	font->default_glyph.height = height ;
	font->default_glyph.lead = 0 ;
	font->default_glyph.ascend = xfs->ascent ;
	font->default_glyph.descend = xfs->descent ;

	free( buf ) ;
	free( compressed_buf ) ;
}

static int
load_X11_glyphs( Display *dpy, ASFont *font, XFontStruct *xfs )
{
	GC gc = NULL;
	font->max_height = xfs->ascent+xfs->descent;
	font->max_ascend = xfs->ascent;
	font->max_descend = xfs->descent;
	font->space_size = xfs->max_bounds.width ;
	if( !get_flags( font->flags, ASF_Monospaced) )
		font->space_size = font->space_size*2/3 ;

#if 0 /*I18N*/
	if( xfs->max_byte1 > 0 && xfs->min_byte1 > 0 )
	{

		char_num *= rows ;
	}else
	{
		int i;
		int min_byte1 = (xfs->min_char_or_byte2>>8)&0x00FF;
		int max_byte1 = (xfs->max_char_or_byte2>>8)&0x00FF;
        size_t offset = MAX(0x00FF,(int)xfs->max_char_or_byte2-(int)(min_byte1<<8)) ;

		load_X11_glyph_range( dpy, font, xfs, 0, min_byte1,
											xfs->min_char_or_byte2-(min_byte1<<8),
			                                offset, &gc );
		offset -= xfs->min_char_or_byte2-(min_byte1<<8);
		if( max_byte1 > min_byte1 )
		{
			for( i = min_byte1+1; i < max_byte1 ; i++ )
			{
				load_X11_glyph_range( dpy, font, xfs, offset, i, 0x00, 0xFF, &gc );
				offset += 256 ;
			}
			load_X11_glyph_range( dpy, font, xfs, offset, max_byte1,
				                                     0,
                                                     (int)xfs->max_char_or_byte2-(int)(max_byte1<<8), &gc );
		}
	}
#else
	{
		/* we blame X consortium for the following mess : */
		int min_char, max_char, our_min_char = 0x0021, our_max_char = 0x00FF ;
		int byte1 = xfs->min_byte1;
		if( xfs->min_byte1 > 0 )
		{
			min_char = xfs->min_char_or_byte2 ;
			max_char = xfs->max_char_or_byte2 ;
			if( min_char > 0x00FF )
			{
				byte1 = (min_char>>8)&0x00FF;
				min_char &=  0x00FF;
				if( ((max_char>>8)&0x00FF) > byte1 )
					max_char =  0x00FF;
				else
					max_char &= 0x00FF;
			}
		}else
		{
			min_char = ((xfs->min_byte1<<8)&0x00FF00)|(xfs->min_char_or_byte2&0x00FF);
			max_char = ((xfs->min_byte1<<8)&0x00FF00)|(xfs->max_char_or_byte2&0x00FF);
			our_min_char |= ((xfs->min_byte1<<8)&0x00FF00) ;
			our_max_char |= ((xfs->min_byte1<<8)&0x00FF00) ;
		}
		our_min_char = MAX(our_min_char,min_char);
		our_max_char = MIN(our_max_char,max_char);

        load_X11_glyph_range( dpy, font, xfs, (int)our_min_char-(int)min_char, byte1, our_min_char&0x00FF, our_max_char&0x00FF, &gc );
	}
#endif
	if( font->default_glyph.pixmap == NULL )
		make_X11_default_glyph( font, xfs );
	if( gc )
		XFreeGC( dpy, gc );
	return xfs->ascent+xfs->descent;
}
#endif /* #ifndef X_DISPLAY_MISSING */

#ifdef HAVE_FREETYPE
static void
load_glyph_freetype( ASFont *font, ASGlyph *asg, int glyph, UNICODE_CHAR uc )
{
	register FT_Face face ;
	static CARD8 *glyph_compress_buf = NULL, *glyph_scaling_buf = NULL ;
	static int glyph_compress_buf_size = 0, glyph_scaling_buf_size = 0;

	if( font == NULL ) 
	{	
		if( glyph_compress_buf )
		{	
			free( glyph_compress_buf );
			glyph_compress_buf = NULL ; 
		}
		if( glyph_scaling_buf ) 
		{	
			free( glyph_scaling_buf );
			glyph_scaling_buf = NULL ;
		}
		glyph_compress_buf_size = 0 ;
		glyph_scaling_buf_size = 0 ;
		return;
	}

	face = font->ft_face;
	if( FT_Load_Glyph( face, glyph, FT_LOAD_DEFAULT ) )
		return;

	if( FT_Render_Glyph( face->glyph, ft_render_mode_normal ) )
		return;

	if( face->glyph->bitmap.buffer )
	{
		FT_Bitmap 	*bmap = &(face->glyph->bitmap) ;
		register CARD8 *src = bmap->buffer ;
		int src_step ;
/* 		int hpad = (face->glyph->bitmap_left<0)? -face->glyph->bitmap_left: face->glyph->bitmap_left ;
*/
		asg->font_gid = glyph ; 
		asg->width   = bmap->width ;
		asg->height  = bmap->rows ;
		/* Combining Diacritical Marks : */
		if( uc >= 0x0300 && uc <= 0x0362 ) 
			asg->step = 0 ; 
		else
#if 0			
			asg->step = bmap->width+face->glyph->bitmap_left ;
#else
			asg->step = (short)face->glyph->advance.x>>6 ;
#endif
				
		/* we only want to keep lead if it was negative */
		if( uc >= 0x0300 && uc <= 0x0362 && face->glyph->bitmap_left >= 0 ) 
			asg->lead    = -((int)font->space_size - (int)face->glyph->bitmap_left) ;
		else
			asg->lead    = face->glyph->bitmap_left;

		if( bmap->pitch < 0 )
			src += -bmap->pitch*bmap->rows ;
		src_step = bmap->pitch ;

		/* TODO: insert monospaced adjastments here */
		if( get_flags( font->flags, ASF_Monospaced ) && ( uc < 0x0300 || uc > 0x0362 ) )
		{
			if( asg->lead < 0 ) 
			{		
				if( asg->lead < -(int)font->space_size/8 ) 
					asg->lead = -(int)font->space_size/8 ;
				if( (int)asg->width + asg->lead <= (int)font->space_size )
				{	
					asg->lead = (int)font->space_size - (int)asg->width ;
					if( asg->lead > 0 ) 
						asg->lead /= 2 ;
				}
			}else
			{	
			 	if( (int)asg->width + (int)asg->lead > (int)font->space_size )
				{	
					if( asg->lead > (int)font->space_size/8 ) 
						asg->lead = (int)font->space_size/8 ;
				}else                          /* centering the glyph : */
					asg->lead += ((int)font->space_size - ((int)asg->width+asg->lead))/2 ;
			}	
			if( (int)asg->width + asg->lead > (int)font->space_size )
			{	
				register CARD8 *buf ;
				int i ;
				asg->width = (int)font->space_size - asg->lead ;
				if( glyph_scaling_buf_size  < bmap->width*bmap->rows*2 )
				{
					glyph_scaling_buf_size = bmap->width*bmap->rows*2;
					glyph_scaling_buf = realloc( glyph_scaling_buf, glyph_scaling_buf_size );
				}	 
				buf = &(glyph_scaling_buf[0]);
				for( i = 0 ; i < bmap->rows ; ++i ) 
				{
					int k = bmap->width;
					while( --k >= 0 ) 
						buf[k] = src[k] ;
					buf += bmap->width ;
					src += src_step ;					   
				}						
				src = &(glyph_scaling_buf[0]);
				scale_down_glyph_width( src, bmap->width, asg->width, asg->height );
				src_step = asg->width ;
/*					fprintf(stderr, "lead = %d, space_size = %d, width = %d, to_width = %d\n",
						r->glyphs[i].lead, font->space_size, width, r->glyphs[i].width ); */
			}
			/*else
			{
				fprintf(stderr, "lead = %d, space_size = %d, width = %d\n",
						r->glyphs[i].lead, font->space_size, width );
			}	 */
			asg->step = font->space_size ;
		}	 

		
		if( glyph_compress_buf_size  < asg->width*asg->height*3 )
		{
			glyph_compress_buf_size = asg->width*asg->height*3;
			glyph_compress_buf = realloc( glyph_compress_buf, glyph_compress_buf_size );
		}	 
	
		/* we better do some RLE encoding in attempt to preserv memory */
		asg->pixmap  = compress_glyph_pixmap( src, glyph_compress_buf, asg->width, asg->height, src_step );
		asg->ascend  = face->glyph->bitmap_top;
		asg->descend = bmap->rows - asg->ascend;
		LOCAL_DEBUG_OUT( "glyph %p with FT index %u is %dx%d ascend = %d, lead = %d, bmap_top = %d", 
							asg, glyph, asg->width, asg->height, asg->ascend, asg->lead, 
							face->glyph->bitmap_top );
	}
}

static ASGlyphRange*
split_freetype_glyph_range( unsigned long min_char, unsigned long max_char, FT_Face face )
{
	ASGlyphRange *first = NULL, **r = &first;
LOCAL_DEBUG_CALLER_OUT( "min_char = %lu, max_char = %lu, face = %p", min_char, max_char, face );
	while( min_char <= max_char )
	{
		register unsigned long i = min_char;
		while( i <= max_char && FT_Get_Char_Index( face, CHAR2UNICODE(i)) == 0 ) i++ ;
		if( i <= max_char )
		{
			*r = safecalloc( 1, sizeof(ASGlyphRange));
			(*r)->min_char = i ;
			while( i <= max_char && FT_Get_Char_Index( face, CHAR2UNICODE(i)) != 0 ) i++ ;
			(*r)->max_char = i ;
LOCAL_DEBUG_OUT( "created glyph range from %lu to %lu", (*r)->min_char, (*r)->max_char );
			r = &((*r)->above);
		}
		min_char = i ;
	}
	return first;
}

static ASGlyph*
load_freetype_locale_glyph( ASFont *font, UNICODE_CHAR uc )
{
	ASGlyph *asg = NULL ;
	if( FT_Get_Char_Index( font->ft_face, uc) != 0 )
	{
		asg = safecalloc( 1, sizeof(ASGlyph));
		load_glyph_freetype( font, asg, FT_Get_Char_Index( font->ft_face, uc), uc);
		if( add_hash_item( font->locale_glyphs, AS_HASHABLE(uc), asg ) != ASH_Success )
		{
			LOCAL_DEBUG_OUT( "Failed to add glyph %p for char %ld to hash", asg, uc );
			asglyph_destroy( 0, asg);
			asg = NULL ;
		}else
		{
			LOCAL_DEBUG_OUT( "added glyph %p for char %ld to hash font attr(%d,%d,%d) glyph attr (%d,%d)", asg, uc, font->max_ascend, font->max_descend, font->max_height, asg->ascend, asg->descend );

			if( asg->ascend > font->max_ascend )
				font->max_ascend = asg->ascend ;
			if( asg->descend > font->max_descend )
				font->max_descend = asg->descend ;
			font->max_height = font->max_ascend+font->max_descend ;
			LOCAL_DEBUG_OUT( "font attr(%d,%d,%d) glyph attr (%d,%d)", font->max_ascend, font->max_descend, font->max_height, asg->ascend, asg->descend );
		}
	}else
		add_hash_item( font->locale_glyphs, AS_HASHABLE(uc), NULL );
	return asg;
}

static void
load_freetype_locale_glyphs( unsigned long min_char, unsigned long max_char, ASFont *font )
{
	register unsigned long i = min_char ;
LOCAL_DEBUG_CALLER_OUT( "min_char = %lu, max_char = %lu, font = %p", min_char, max_char, font );
	if( font->locale_glyphs == NULL )
		font->locale_glyphs = create_ashash( 0, NULL, NULL, asglyph_destroy );
	while( i <= max_char )
	{
		load_freetype_locale_glyph( font, CHAR2UNICODE(i));
		++i;
	}
	LOCAL_DEBUG_OUT( "font attr(%d,%d,%d)", font->max_ascend, font->max_descend, font->max_height );
}


static int
load_freetype_glyphs( ASFont *font )
{
	int max_ascend = 0, max_descend = 0;
	ASGlyphRange *r ;

    /* we preload only codes in range 0x21-0xFF in current charset */
	/* if draw_unicode_text is used and we need some other glyphs
	 * we'll just need to add them on demand */
	font->codemap = split_freetype_glyph_range( 0x0021, 0x007F, font->ft_face );

	load_glyph_freetype( font, &(font->default_glyph), 0, 0);/* special no-symbol glyph */
    load_freetype_locale_glyphs( 0x0080, 0x00FF, font );
	if( font->codemap == NULL )
	{
		font->max_height = font->default_glyph.ascend+font->default_glyph.descend;
		if( font->max_height <= 0 )
			font->max_height = 1 ;
		font->max_ascend = MAX((int)font->default_glyph.ascend,1);
		font->max_descend = MAX((int)font->default_glyph.descend,1);
	}else
	{
		for( r = font->codemap ; r != NULL ; r = r->above )
		{
			long min_char = r->min_char ;
			long max_char = r->max_char ;
			long i ;
			if( max_char < min_char ) 
			{
				i = max_char ; 
				max_char = min_char ; 
				min_char = i ;	 
			}	 
            r->glyphs = safecalloc( (max_char - min_char) + 1, sizeof(ASGlyph));
			for( i = min_char ; i < max_char ; ++i )
			{
				if( i != ' ' && i != '\t' && i!= '\n' )
				{
					ASGlyph *asg = &(r->glyphs[i-min_char]);
					UNICODE_CHAR uc = CHAR2UNICODE(i);
					load_glyph_freetype( font, asg, FT_Get_Char_Index( font->ft_face, uc), uc);
/* Not needed ?
 * 					if( asg->lead >= 0 || asg->lead+asg->width > 3 )
 *						font->pen_move_dir = LEFT_TO_RIGHT ;
 */
					if( asg->ascend > max_ascend )
						max_ascend = asg->ascend ;
					if( asg->descend > max_descend )
						max_descend = asg->descend ;
				}
			}
		}
		if( (int)font->max_ascend <= max_ascend )
			font->max_ascend = MAX(max_ascend,1);
		if( (int)font->max_descend <= max_descend )
			font->max_descend = MAX(max_descend,1);
	 	font->max_height = font->max_ascend+font->max_descend;
	}
	/* flushing out compression buffer : */
	load_glyph_freetype(NULL, NULL, 0, 0);
	return max_ascend+max_descend;
}
#endif

static inline ASGlyph *get_unicode_glyph( const UNICODE_CHAR uc, ASFont *font )
{
	register ASGlyphRange *r;
	ASGlyph *asg = NULL ;
	ASHashData hdata = {0} ;
	for( r = font->codemap ; r != NULL ; r = r->above )
	{
LOCAL_DEBUG_OUT( "looking for glyph for char %lu (%p) if range (%ld,%ld)", uc, asg, r->min_char, r->max_char);

		if( r->max_char >= uc )
			if( r->min_char <= uc )
			{
				asg = &(r->glyphs[uc - r->min_char]);
LOCAL_DEBUG_OUT( "Found glyph for char %lu (%p)", uc, asg );
				if( asg->width > 0 && asg->pixmap != NULL )
					return asg;
				break;
			}
	}
	if( get_hash_item( font->locale_glyphs, AS_HASHABLE(uc), &hdata.vptr ) != ASH_Success )
	{
#ifdef HAVE_FREETYPE
		asg = load_freetype_locale_glyph( font, uc );
LOCAL_DEBUG_OUT( "glyph for char %lu  loaded as %p", uc, asg );
#endif
	}else
		asg = hdata.vptr ;
LOCAL_DEBUG_OUT( "%sFound glyph for char %lu ( %p )", asg?"":"Did not ", uc, asg );
	return asg?asg:&(font->default_glyph) ;
}


static inline ASGlyph *get_character_glyph( const unsigned char c, ASFont *font )
{
	return get_unicode_glyph( CHAR2UNICODE(c), font );
}

static UNICODE_CHAR
utf8_to_unicode ( const unsigned char *s )
{
	unsigned char c = s[0];

	if (c < 0x80)
	{
  		return (UNICODE_CHAR)c;
	} else if (c < 0xc2)
	{
  		return 0;
    } else if (c < 0xe0)
	{
	    if (!((s[1] ^ 0x80) < 0x40))
    		return 0;
	    return ((UNICODE_CHAR) (c & 0x1f) << 6)
  		       |(UNICODE_CHAR) (s[1] ^ 0x80);
    } else if (c < 0xf0)
	{
	    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
  		      && (c >= 0xe1 || s[1] >= 0xa0)))
	        return 0;
		return ((UNICODE_CHAR) (c & 0x0f) << 12)
        	 | ((UNICODE_CHAR) (s[1] ^ 0x80) << 6)
          	 |  (UNICODE_CHAR) (s[2] ^ 0x80);
	} else if (c < 0xf8 && sizeof(UNICODE_CHAR)*8 >= 32)
	{
	    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
  	        && (s[3] ^ 0x80) < 0x40
    	    && (c >= 0xf1 || s[1] >= 0x90)))
    		return 0;
	    return ((UNICODE_CHAR) (c & 0x07) << 18)
             | ((UNICODE_CHAR) (s[1] ^ 0x80) << 12)
	         | ((UNICODE_CHAR) (s[2] ^ 0x80) << 6)
  	         |  (UNICODE_CHAR) (s[3] ^ 0x80);
	} else if (c < 0xfc && sizeof(UNICODE_CHAR)*8 >= 32)
	{
	    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
  	        && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40
    	    && (c >= 0xf9 || s[1] >= 0x88)))
	        return 0;
		return ((UNICODE_CHAR) (c & 0x03) << 24)
             | ((UNICODE_CHAR) (s[1] ^ 0x80) << 18)
	         | ((UNICODE_CHAR) (s[2] ^ 0x80) << 12)
  	         | ((UNICODE_CHAR) (s[3] ^ 0x80) << 6)
    	     |  (UNICODE_CHAR) (s[4] ^ 0x80);
	} else if (c < 0xfe && sizeof(UNICODE_CHAR)*8 >= 32)
	{
	    if (!((s[1] ^ 0x80) < 0x40 && (s[2] ^ 0x80) < 0x40
  		    && (s[3] ^ 0x80) < 0x40 && (s[4] ^ 0x80) < 0x40
      	    && (s[5] ^ 0x80) < 0x40
        	&& (c >= 0xfd || s[1] >= 0x84)))
	  		return 0;
		return ((UNICODE_CHAR) (c & 0x01) << 30)
      	     | ((UNICODE_CHAR) (s[1] ^ 0x80) << 24)
        	 | ((UNICODE_CHAR) (s[2] ^ 0x80) << 18)
             | ((UNICODE_CHAR) (s[3] ^ 0x80) << 12)
	         | ((UNICODE_CHAR) (s[4] ^ 0x80) << 6)
  	    	 |  (UNICODE_CHAR) (s[5] ^ 0x80);
    }
    return 0;
}

static inline ASGlyph *get_utf8_glyph( const char *utf8, ASFont *font )
{
	UNICODE_CHAR uc = utf8_to_unicode ( (const unsigned char*)utf8 );
	LOCAL_DEBUG_OUT( "translated to Unicode 0x%lX(%ld), UTF8 size = %d", uc, uc, UTF8_CHAR_SIZE(utf8[0]) );
	return get_unicode_glyph( uc, font );
}

/*********************************************************************************/
/* actuall rendering code :						   								 */
/*********************************************************************************/

typedef struct ASGlyphMap
{
	int  height, width ;
	ASGlyph 	**glyphs;
	int 		  glyphs_num;
	short 		 *x_kerning ;
}ASGlyphMap;


static void
apply_text_3D_type( ASText3DType type,
                    int *width, int *height )
{
	switch( type )
	{
		case AST_Embossed   :
		case AST_Sunken     :
				(*width) += 2; (*height) += 2 ;
				break ;
		case AST_ShadeAbove :
		case AST_ShadeBelow :
				(*width)+=3; (*height)+=3 ;
				break ;
		case AST_SunkenThick :
		case AST_EmbossedThick :
				(*width)+=3; (*height)+=3 ;
				break ;
		case AST_OutlineAbove :
		case AST_OutlineBelow :
				(*width) += 1; (*height) += 1 ;
				break ;
		case AST_OutlineFull :
				(*width) += 2; (*height) += 2 ;
				break ;
		default  :
				break ;
	}
}

static unsigned int
goto_tab_stop( ASTextAttributes *attr, unsigned int space_size, unsigned int line_width )
{
	unsigned int tab_size = attr->tab_size*space_size ;
	unsigned int tab_stop = (((attr->origin + line_width)/tab_size)+1)*tab_size ;
	if( attr->tab_stops != NULL && attr->tab_stops_num > 0 ) 	
	{
		unsigned int i ;
		for( i = 0 ; i < attr->tab_stops_num ; ++i ) 
		{	
			if( attr->tab_stops[i] < line_width )
				continue;
			if( attr->tab_stops[i] < tab_stop ) 
				tab_stop = attr->tab_stops[i] ;
			break;
		}
	}
	return tab_stop;		
}

#ifdef HAVE_FREETYPE
#define GET_KERNING(var,prev_gid,this_gid)   \
	do{ if( (prev_gid) != 0 && font->type == ASF_Freetype && get_flags(font->flags, ASF_Monospaced|ASF_HasKerning) == ASF_HasKerning ) { \
		FT_Vector delta; \
		FT_Get_Kerning( font->ft_face, (prev_gid), (this_gid), FT_KERNING_DEFAULT, &delta );\
		(var) = (short)(delta.x >> 6); \
	}}while(0)
#else
#define GET_KERNING(var,prev_gid,this_gid)	do{(var)=0;}while(0)	  
#endif
/*		fprintf( stderr, "####### pair %d ,%d 	has kerning = %d\n", prev_gid,this_gid, var ); */


#define FILL_TEXT_GLYPH_MAP(name,type,getglyph,incr) \
static unsigned int \
name( const type *text, ASFont *font, ASGlyphMap *map, ASTextAttributes *attr, int space_size, unsigned int offset_3d_x ) \
{ \
	int w = 0, line_count = 0, line_width = 0; \
	int i = -1, g = 0 ; \
	ASGlyph *last_asg = NULL ; unsigned int last_gid = 0 ; \
	do \
	{ \
		++i ; \
		LOCAL_DEBUG_OUT("begin_i=%d, char_code = 0x%2.2X",i,text[i]); \
		if( text[i] == '\n' || g == map->glyphs_num-1 ) \
		{ \
			if( last_asg && last_asg->width+last_asg->lead > last_asg->step ) \
				line_width += last_asg->width+last_asg->lead - last_asg->step ; \
			last_asg = NULL; last_gid = 0 ; \
			if( line_width > w ) \
				w = line_width ; \
			line_width = 0 ; \
			++line_count ; \
			map->glyphs[g] = (g == map->glyphs_num-1)?GLYPH_EOT:GLYPH_EOL; \
			++g; \
		}else \
		{ \
			last_asg = NULL ; \
			if( text[i] == ' ' ) \
			{   last_gid = 0 ; \
				line_width += space_size ; \
				map->glyphs[g++] = GLYPH_SPACE; \
			}else if( text[i] == '\t' ) \
			{   last_gid = 0 ; \
				if( !get_flags(attr->rendition_flags, ASTA_UseTabStops) ) line_width += space_size*attr->tab_size ; \
				else line_width = goto_tab_stop( attr, space_size, line_width ); \
				map->glyphs[g++] = GLYPH_TAB; \
			}else \
			{ \
				last_asg = getglyph; \
				map->glyphs[g] = last_asg; \
				GET_KERNING(map->x_kerning[g],last_gid,last_asg->font_gid); \
				if( line_width < -last_asg->lead ) line_width -= (line_width+last_asg->lead);\
				line_width += last_asg->step+offset_3d_x+map->x_kerning[g]; \
				++g; last_gid = last_asg->font_gid ; \
				LOCAL_DEBUG_OUT("pre_i=%d",i); \
				incr; /* i+=CHAR_SIZE(text[i])-1; */ \
				LOCAL_DEBUG_OUT("post_i=%d",i); \
			} \
		} \
	}while( g < map->glyphs_num );  \
	map->width = MAX( w, 1 ); \
	return line_count ; \
}

#ifdef _MSC_VER
FILL_TEXT_GLYPH_MAP(fill_text_glyph_map_Char,char,get_character_glyph(text[i],font),1)
FILL_TEXT_GLYPH_MAP(fill_text_glyph_map_Unicode,UNICODE_CHAR,get_unicode_glyph(text[i],font),1)
#else
FILL_TEXT_GLYPH_MAP(fill_text_glyph_map_Char,char,get_character_glyph(text[i],font),/* */)
FILL_TEXT_GLYPH_MAP(fill_text_glyph_map_Unicode,UNICODE_CHAR,get_unicode_glyph(text[i],font),/* */)
#endif
FILL_TEXT_GLYPH_MAP(fill_text_glyph_map_UTF8,char,get_utf8_glyph(&text[i],font),i+=(UTF8_CHAR_SIZE(text[i])-1))

void
free_glyph_map( ASGlyphMap *map, Bool reusable )
{
    if( map )
    {
		if( map->glyphs )
	        free( map->glyphs );
		if( map->x_kerning )
	        free( map->x_kerning );
        if( !reusable )
            free( map );
    }
}

static int
get_text_length (ASCharType char_type, const char *text)
{
	register int count = 0;
	if( char_type == ASCT_Char )
	{
		register char *ptr = (char*)text ;
		while( ptr[count] != 0 )++count;
	}else if( char_type == ASCT_UTF8 )
	{
		register char *ptr = (char*)text ;
		while( *ptr != 0 ){	++count; ptr += UTF8_CHAR_SIZE(*ptr); }
	}else if( char_type == ASCT_Unicode )
	{
		register UNICODE_CHAR *uc_ptr = (UNICODE_CHAR*)text ;
		while( uc_ptr[count] != 0 )	++count;
	}
	return count;
}

ASGlyph**
get_text_glyph_list (const char *text, ASFont *font, ASCharType char_type, int length)
{
	ASGlyph** glyphs = NULL;
	int i = 0;
	
	if (text == NULL || font == NULL)
		return NULL;
	if (length <= 0)
		if ((length = get_text_length (char_type, text)) <= 0)
			return NULL;
	
	glyphs = safecalloc( length+1, sizeof(ASGlyph*));
	if (char_type == ASCT_Char)
	{
		register char *ptr = (char*)text;
		for (i = 0 ; i < length ; ++i)
			glyphs[i] = get_character_glyph (ptr[i], font);
	}else if (char_type == ASCT_UTF8)
	{
		register char *ptr = (char*)text;
		for (i = 0 ; i < length ; ++i)
		{
			glyphs[i] = get_utf8_glyph (ptr, font);
			ptr += UTF8_CHAR_SIZE(*ptr);
		}		
	}else if( char_type == ASCT_Unicode )
	{
		register UNICODE_CHAR *uc_ptr = (UNICODE_CHAR*)text ;
		for (i = 0 ; i < length ; ++i)
			glyphs[i] = get_unicode_glyph (uc_ptr[i], font);
	}
	
	return glyphs;			
}

static Bool
get_text_glyph_map (const char *text, ASFont *font, ASGlyphMap *map, ASTextAttributes *attr, int length  )
{
	unsigned int line_count = 0;
	int offset_3d_x = 0, offset_3d_y = 0 ;
	int space_size  = 0 ;

	apply_text_3D_type( attr->type, &offset_3d_x, &offset_3d_y );

	if( text == NULL || font == NULL || map == NULL)
		return False;
	
	offset_3d_x += font->spacing_x ;
	offset_3d_y += font->spacing_y ;
	
	space_size  = font->space_size ;
	if( !get_flags( font->flags, ASF_Monospaced) )
		space_size  = (space_size>>1)+1 ;
	space_size += offset_3d_x;

	map->glyphs_num = 1;
	if( length <= 0 ) 
		length = get_text_length (attr->char_type, text);

	map->glyphs_num = 1 + length ;

	map->glyphs = safecalloc( map->glyphs_num, sizeof(ASGlyph*));
	map->x_kerning = safecalloc( map->glyphs_num, sizeof(short));

	if( attr->char_type == ASCT_UTF8 )
		line_count = fill_text_glyph_map_UTF8( text, font, map, attr, space_size, offset_3d_x );
	else if( attr->char_type == ASCT_Unicode )
		line_count = fill_text_glyph_map_Unicode( (UNICODE_CHAR*)text, font, map, attr, space_size, offset_3d_x );
	else /* assuming attr->char_type == ASCT_Char by default */
		line_count = fill_text_glyph_map_Char( text, font, map, attr, space_size, offset_3d_x );
	
    map->height = line_count * (font->max_height+offset_3d_y) - font->spacing_y;

	if( map->height <= 0 )
		map->height = 1 ;

	return True;
}

#define GET_TEXT_SIZE_LOOP(getglyph,incr,len) \
	do{ Bool terminated = True; ++i ;\
		if( len == 0 || i < len )	\
		{ 	terminated = ( text[i] == '\0' || text[i] == '\n' ); \
			if( x_positions ) x_positions[i] = line_width ; \
		} \
		if( terminated ) { \
			if( last_asg && last_asg->width+last_asg->lead > last_asg->step ) \
				line_width += last_asg->width+last_asg->lead - last_asg->step ; \
			last_asg = NULL; last_gid = 0 ; \
			if( line_width > w ) \
				w = line_width ; \
			line_width = 0 ; \
            ++line_count ; \
		}else { \
			last_asg = NULL ; \
			if( text[i] == ' ' ){ \
				line_width += space_size ; last_gid = 0 ;\
			}else if( text[i] == '\t' ){ last_gid = 0 ; \
				if( !get_flags(attr->rendition_flags, ASTA_UseTabStops) ) line_width += space_size*attr->tab_size ; \
				else line_width = goto_tab_stop( attr, space_size, line_width ); \
			}else{ int kerning = 0 ;\
				last_asg = getglyph; \
				GET_KERNING(kerning,last_gid,last_asg->font_gid); \
				if( line_width < -last_asg->lead ) line_width -= (line_width+last_asg->lead);\
				line_width += last_asg->step+offset_3d_x +kerning ;  \
				last_gid = last_asg->font_gid ; \
				incr ; \
			} \
		} \
	}while( (len <= 0 || len > i) && text[i] != '\0' )

static Bool
get_text_size_internal( const char *src_text, ASFont *font, ASTextAttributes *attr, unsigned int *width, unsigned int *height, int length, int *x_positions )
{
    int w = 0, h = 0, line_count = 0;
	int line_width = 0;
    int i = -1;
	ASGlyph *last_asg = NULL ;
	int space_size = 0;
	int offset_3d_x = 0, offset_3d_y = 0 ;
	int last_gid = 0 ;


	apply_text_3D_type( attr->type, &offset_3d_x, &offset_3d_y );
	if( src_text == NULL || font == NULL )
		return False;
	
	offset_3d_x += font->spacing_x ;
	offset_3d_y += font->spacing_y ;

	space_size  = font->space_size ;
	if( !get_flags( font->flags, ASF_Monospaced) )
		space_size  = (space_size>>1)+1 ;
	space_size += offset_3d_x ;

	LOCAL_DEBUG_OUT( "char_type = %d", attr->char_type );
	if( attr->char_type == ASCT_Char )
	{
		char *text = (char*)&src_text[0] ;
#ifdef _MSC_VER
		GET_TEXT_SIZE_LOOP(get_character_glyph(text[i],font),1,length);
#else		
		GET_TEXT_SIZE_LOOP(get_character_glyph(text[i],font),/* */,length);
#endif
	}else if( attr->char_type == ASCT_UTF8 )
	{
		char *text = (char*)&src_text[0] ;
		int byte_length = 0 ;
		if( length > 0 )
		{
			int k ; 
			for( k = 0 ; k < length ; ++k )
			{
				if( text[byte_length] == '\0' ) 
					break;
				byte_length += UTF8_CHAR_SIZE(text[byte_length]);		   
			}	 
		}	 
		GET_TEXT_SIZE_LOOP(get_utf8_glyph(&text[i],font),i+=UTF8_CHAR_SIZE(text[i])-1,byte_length);
	}else if( attr->char_type == ASCT_Unicode )
	{
		UNICODE_CHAR *text = (UNICODE_CHAR*)&src_text[0] ;
#ifdef _MSC_VER
		GET_TEXT_SIZE_LOOP(get_unicode_glyph(text[i],font),1,length);
#else		   
		GET_TEXT_SIZE_LOOP(get_unicode_glyph(text[i],font),/* */,length);
#endif
	}

    h = line_count * (font->max_height+offset_3d_y) - font->spacing_y;

    if( w < 1 )
		w = 1 ;
	if( h < 1 )
		h = 1 ;
	if( width )
		*width = w;
	if( height )
		*height = h;
	return True ;
}

Bool
get_text_size( const char *src_text, ASFont *font, ASText3DType type, unsigned int *width, unsigned int *height )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Char, 8, 0, NULL, 0 }; 
	attr.type = type ;
	if( IsUTF8Locale() ) 
		attr.char_type = ASCT_UTF8 ;
	return get_text_size_internal( (char*)src_text, font, &attr, width, height, 0/*autodetect length*/, NULL );
}

Bool
get_unicode_text_size( const UNICODE_CHAR *src_text, ASFont *font, ASText3DType type, unsigned int *width, unsigned int *height )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Unicode, 8, 0, NULL, 0 }; 
	attr.type = type ;
	return get_text_size_internal( (char*)src_text, font, &attr, width, height, 0/*autodetect length*/, NULL );
}

Bool
get_utf8_text_size( const char *src_text, ASFont *font, ASText3DType type, unsigned int *width, unsigned int *height )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_UTF8, 8, 0, NULL, 0 }; 
	attr.type = type ;
	return get_text_size_internal( (char*)src_text, font, &attr, width, height, 0/*autodetect length*/, NULL );
}

Bool
get_fancy_text_size( const void *src_text, ASFont *font, ASTextAttributes *attr, unsigned int *width, unsigned int *height, int length, int *x_positions )
{
	ASTextAttributes internal_attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Char, 8, 0, NULL, 0 }; 
	if( attr != NULL ) 
	{	
		internal_attr = *attr;
		if( internal_attr.tab_size == 0 ) 
			internal_attr.tab_size = 8 ;
		internal_attr.version = ASTA_VERSION_INTERNAL ;
	}else
	{
		if( IsUTF8Locale() ) 
			internal_attr.char_type = ASCT_UTF8 ;
	}	 
	return get_text_size_internal( src_text, font, &internal_attr, width, height, length, x_positions );
}

inline static void
render_asglyph( CARD8 **scanlines, CARD8 *row,
                int start_x, int y, int width, int height,
				CARD32 ratio )
{
	int count = -1 ;
	int max_y = y + height ;
	register CARD32 data = 0;
	while( y < max_y )
	{
		register CARD8 *dst = scanlines[y]+start_x;
		register int x = -1;
		while( ++x < width )
		{
/*fprintf( stderr, "data = %X, count = %d, x = %d, y = %d\n", data, count, x, y );*/
			if( count < 0 )
			{
				data = *(row++);
				if( (data&0x80) != 0)
				{
					data = ((data&0x7F)<<1);
					if( data != 0 )
						++data;
				}else
				{
					count = data&0x3F ;
					data = ((data&0x40) != 0 )? 0xFF: 0x00;
				}
				if( ratio != 0xFF && data != 0 )
					data = ((data*ratio)>>8)+1 ;
			}
			if( data > dst[x] ) 
				dst[x] = (data > 255)? 0xFF:data ;
			--count;
		}
		++y;
	}
}

inline static void
render_asglyph_over( CARD8 **scanlines, CARD8 *row,
                int start_x, int y, int width, int height,
				CARD32 value )
{
	int count = -1 ;
	int max_y = y + height ;
	CARD32 anti_data = 0;
	register CARD32 data = 0;
	while( y < max_y )
	{
		register CARD8 *dst = scanlines[y]+start_x;
		register int x = -1;
		while( ++x < width )
		{
/*fprintf( stderr, "data = %X, count = %d, x = %d, y = %d\n", data, count, x, y );*/
			if( count < 0 )
			{
				data = *(row++);
				if( (data&0x80) != 0)
				{
					data = ((data&0x7F)<<1);
					if( data != 0 )
						++data;
				}else
				{
					count = data&0x3F ;
					data = ((data&0x40) != 0 )? 0xFF: 0x00;
				}
				anti_data = 256 - data ;
			}
			if( data >= 254 ) 
				dst[x] = value ;
			else
				dst[x] = ((CARD32)dst[x]*anti_data + value*data)>>8 ;
			--count;
		}
		++y;
	}
}



static ASImage *
draw_text_internal( const char *text, ASFont *font, ASTextAttributes *attr, int compression, int length )
{
	ASGlyphMap map ;
	CARD8 *memory, *rgb_memory = NULL;
	CARD8 **scanlines, **rgb_scanlines = NULL ;
	int i = 0, offset = 0, line_height, space_size, base_line;
	ASImage *im;
	int pen_x = 0, pen_y = 0;
	int offset_3d_x = 0, offset_3d_y = 0  ;
	CARD32 back_color = 0 ;
	CARD32 alpha_7 = 0x007F, alpha_9 = 0x009F, alpha_A = 0x00AF, alpha_C = 0x00CF, alpha_F = 0x00FF, alpha_E = 0x00EF;
	START_TIME(started);	   

	// Perform line breaks if a fixed width is specified
	// TODO: this is a quick and dirty fix and should work for now, but we really should fix it 
	// so we don't have to calculate text size so many times as well as make it UNICODE friendly
	// and remove mangling of the source text (Sasha): 
	if (attr->width)
	{
        unsigned int width, height; // SMA
        get_text_size(  text , font, attr->type, &width, &height ); 
        if ( (width > attr->width)  &&  (strchr(text, ' ')) )
        {
          char *tryPtr = strchr(text,' ');
          char *oldTryPtr = tryPtr;
          while (tryPtr)
            {        
               *tryPtr = 0;
               get_text_size(  text , font, attr->type, &width, &height ); 
               if (width > attr->width)
                   *oldTryPtr = '\n';
               
               *tryPtr = ' ';
               oldTryPtr = tryPtr;
               tryPtr = strchr(tryPtr + 1,' ');
            }
        }
	}	    

LOCAL_DEBUG_CALLER_OUT( "text = \"%s\", font = %p, compression = %d", text, font, compression );
	if( !get_text_glyph_map( text, font, &map, attr, length) )
		return NULL;
	
	if( map.width <= 0 ) 
		return NULL;

	apply_text_3D_type( attr->type, &offset_3d_x, &offset_3d_y );

	offset_3d_x += font->spacing_x ;
	offset_3d_y += font->spacing_y ;
	line_height = font->max_height+offset_3d_y ;

LOCAL_DEBUG_OUT( "text size = %dx%d pixels", map.width, map.height );
	im = create_asimage( map.width, map.height, compression );

	space_size  = font->space_size ;
	if( !get_flags( font->flags, ASF_Monospaced) )
		space_size  = (space_size>>1)+1 ;
	space_size += offset_3d_x;

	base_line = font->max_ascend;
LOCAL_DEBUG_OUT( "line_height is %d, space_size is %d, base_line is %d", line_height, space_size, base_line );
	scanlines = safemalloc( line_height*sizeof(CARD8*));
	memory = safecalloc( 1, line_height*map.width);
	for( i = 0 ; i < line_height ; ++i ) 
	{
		scanlines[i] = memory + offset;
		offset += map.width;
	}
	if( attr->type >= AST_OutlineAbove ) 
	{
		CARD32 fc = attr->fore_color ;
		offset = 0 ;
		rgb_scanlines = safemalloc( line_height*3*sizeof(CARD8*));
		rgb_memory = safecalloc( 1, line_height*map.width*3);
		for( i = 0 ; i < line_height*3 ; ++i ) 
		{
			rgb_scanlines[i] = rgb_memory + offset;
			offset += map.width;
		}
		if( (ARGB32_RED16(fc)*222+ARGB32_GREEN16(fc)*707+ARGB32_BLUE16(fc) *71)/1000 < 0x07FFF ) 
		{	
			back_color = 0xFF ;
			memset( rgb_memory, back_color, line_height*map.width*3 );
		}
	}	 
	if( ARGB32_ALPHA8(attr->fore_color) > 0 ) 
	{
		CARD32 a = ARGB32_ALPHA8(attr->fore_color);
		alpha_7 = (0x007F*a)>>8 ;
		alpha_9 = (0x009F*a)>>8 ;
		alpha_A = (0x00AF*a)>>8 ;
		alpha_C = (0x00CF*a)>>8 ;
		alpha_E	= (0x00EF*a)>>8 ;
		alpha_F = (0x00FF*a)>>8 ;
	}	 

	i = -1 ;
	if(get_flags(font->flags, ASF_RightToLeft))
		pen_x = map.width ;

	do
	{
		++i;
/*fprintf( stderr, "drawing character %d '%c'\n", i, text[i] );*/
		if( map.glyphs[i] == GLYPH_EOL || map.glyphs[i] == GLYPH_EOT )
		{
			int y;
			for( y = 0 ; y < line_height ; ++y )
			{
#if 1
#if defined(LOCAL_DEBUG) && !defined(NO_DEBUG_OUTPUT)
				{				
					int x = 0;
					while( x < map.width )
						fprintf( stderr, "%2.2X ", scanlines[y][x++] );
					fprintf( stderr, "\n" );
				}
#endif
#endif
 				im->channels[IC_ALPHA][pen_y+y] = store_data( NULL, scanlines[y], map.width, ASStorage_RLEDiffCompress, 0);
				if( attr->type >= AST_OutlineAbove ) 
				{
	 				im->channels[IC_RED][pen_y+y] 	= store_data( NULL, rgb_scanlines[y], map.width, ASStorage_RLEDiffCompress, 0);
	 				im->channels[IC_GREEN][pen_y+y] = store_data( NULL, rgb_scanlines[y+line_height], map.width, ASStorage_RLEDiffCompress, 0);
	 				im->channels[IC_BLUE][pen_y+y]  = store_data( NULL, rgb_scanlines[y+line_height+line_height], map.width, ASStorage_RLEDiffCompress, 0);
				}	 
			}
			
			memset( memory, 0x00, line_height*map.width );
			if( attr->type >= AST_OutlineAbove ) 
				memset( rgb_memory, back_color, line_height*map.width*3 );
			
			pen_x = get_flags(font->flags, ASF_RightToLeft)? map.width : 0;
			pen_y += line_height;
			if( pen_y <0 )
				pen_y = 0 ;
		}else
		{
			if( map.glyphs[i] == GLYPH_SPACE || map.glyphs[i] == GLYPH_TAB )
			{
				if(map.glyphs[i] == GLYPH_TAB)
				{
					if( !get_flags(attr->rendition_flags, ASTA_UseTabStops) ) pen_x += space_size*attr->tab_size ;
					else pen_x = goto_tab_stop( attr, space_size, pen_x );
				}else if( get_flags(font->flags, ASF_RightToLeft) )
					pen_x -= space_size ;
				else
					pen_x += space_size ;
			}else
			{
				/* now comes the fun part : */
				ASGlyph *asg = map.glyphs[i] ;
				int y = base_line - asg->ascend;
				int start_x = 0, offset_x = 0;

				if( get_flags(font->flags, ASF_RightToLeft) )
					pen_x  -= asg->step+offset_3d_x +map.x_kerning[i];
				else
				{
					LOCAL_DEBUG_OUT( "char # %d : pen_x = %d, kerning = %d, lead = %d, width = %d, step = %d", i, pen_x, map.x_kerning[i], asg->lead, asg->width, asg->step );
					pen_x += map.x_kerning[i] ;
				}
				if( asg->lead > 0 )
					start_x = pen_x + asg->lead ;
				else
					start_x = pen_x + asg->lead ;
				if( start_x < 0 )
				{
					offset_x = -start_x ; 
					start_x = 0 ;
				}
				if( y < 0 )
					y = 0 ;

				switch( attr->type )
				{
					case AST_Plain :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_F );
					    break ;
					case AST_Embossed :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+2, y+2, asg->width, asg->height, alpha_9 );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_C );
 					    break ;
					case AST_Sunken :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_9 );
						render_asglyph( scanlines, asg->pixmap, start_x+2, y+2, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_C );
					    break ;
					case AST_ShadeAbove :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_7 );
						render_asglyph( scanlines, asg->pixmap, start_x+3, y+3, asg->width, asg->height, alpha_F );
					    break ;
					case AST_ShadeBelow :
						render_asglyph( scanlines, asg->pixmap, start_x+3, y+3, asg->width, asg->height, alpha_7 );
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_F );
					    break ;
					case AST_EmbossedThick :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_E );
						render_asglyph( scanlines, asg->pixmap, start_x+3, y+3, asg->width, asg->height, alpha_7 );
						render_asglyph( scanlines, asg->pixmap, start_x+2, y+2, asg->width, asg->height, alpha_C );
 					    break ;
					case AST_SunkenThick :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_7 );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_A );
						render_asglyph( scanlines, asg->pixmap, start_x+3, y+3, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+2, y+2, asg->width, asg->height, alpha_C );
 					    break ;
					case AST_OutlineAbove :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_A );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_F );
						render_asglyph_over( rgb_scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_RED8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height], asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_GREEN8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height*2], asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_BLUE8(attr->fore_color) );
					    break ;
					case AST_OutlineBelow :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_A );
						render_asglyph_over( rgb_scanlines, asg->pixmap, start_x, y, asg->width, asg->height, ARGB32_RED8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height], asg->pixmap, start_x, y, asg->width, asg->height, ARGB32_GREEN8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height*2], asg->pixmap, start_x, y, asg->width, asg->height, ARGB32_BLUE8(attr->fore_color) );
					    break ;
					case AST_OutlineFull :
						render_asglyph( scanlines, asg->pixmap, start_x, y, asg->width, asg->height, alpha_A );
						render_asglyph( scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, alpha_F );
						render_asglyph( scanlines, asg->pixmap, start_x+2, y+2, asg->width, asg->height, alpha_A );
						render_asglyph_over( rgb_scanlines, asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_RED8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height], asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_GREEN8(attr->fore_color) );
						render_asglyph_over( &rgb_scanlines[line_height*2], asg->pixmap, start_x+1, y+1, asg->width, asg->height, ARGB32_BLUE8(attr->fore_color) );
					    break ;
				  default:
				        break ;
				}

				if( !get_flags(font->flags, ASF_RightToLeft) )
  					pen_x  += offset_x + asg->step+offset_3d_x;
			}
		}
	}while( map.glyphs[i] != GLYPH_EOT );
    free_glyph_map( &map, True );
	free( memory );
	free( scanlines );
	if( rgb_memory ) 
		free( rgb_memory );
	if( rgb_scanlines ) 
		free( rgb_scanlines );
	SHOW_TIME("", started);
	return im;
}

ASImage *
draw_text( const char *text, ASFont *font, ASText3DType type, int compression )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Char, 8, 0, NULL, 0, ARGB32_White }; 
	attr.type = type ;
	if( IsUTF8Locale() ) 
		attr.char_type = ASCT_UTF8 ;
	return draw_text_internal( text, font, &attr, compression, 0/*autodetect length*/ );
}

ASImage *
draw_unicode_text( const UNICODE_CHAR *text, ASFont *font, ASText3DType type, int compression )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Unicode, 8, 0, NULL, 0, ARGB32_White }; 
	attr.type = type ;
	return draw_text_internal( (const char*)text, font, &attr, compression, 0/*autodetect length*/ );
}

ASImage *
draw_utf8_text( const char *text, ASFont *font, ASText3DType type, int compression )
{
	ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_UTF8, 8, 0, NULL, 0, ARGB32_White }; 
	attr.type = type ;
	return draw_text_internal( text, font, &attr, compression, 0/*autodetect length*/ );
}

ASImage *
draw_fancy_text( const void *text, ASFont *font, ASTextAttributes *attr, int compression, int length )
{
	ASTextAttributes internal_attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Char, 8, 0, NULL, 0, ARGB32_White }; 
	if( attr != NULL ) 
	{	
		internal_attr = *attr;
		if( internal_attr.tab_size == 0 ) 
			internal_attr.tab_size = 8 ;
		internal_attr.version = ASTA_VERSION_INTERNAL ;
	}else
	{
		if( IsUTF8Locale() ) 
			internal_attr.char_type = ASCT_UTF8 ;
	}  
	return draw_text_internal( text, font, &internal_attr, compression, length );
}

Bool get_asfont_glyph_spacing( ASFont* font, int *x, int *y )
{
	if( font )
	{
		if( x )
			*x = font->spacing_x ;
		if( y )
			*y = font->spacing_y ;
		return True ;
	}
	return False ;
}

Bool set_asfont_glyph_spacing( ASFont* font, int x, int y )
{
	if( font )
	{
		font->spacing_x = (x < 0 )? 0: x;
		font->spacing_y = (y < 0 )? 0: y;
		return True ;
	}
	return False ;
}

/* Misc functions : */
void print_asfont( FILE* stream, ASFont* font)
{
	if( font )
	{
		fprintf( stream, "font.type = %d\n", font->type       );
		fprintf( stream, "font.flags = 0x%lX\n", font->flags       );
#ifdef HAVE_FREETYPE
		fprintf( stream, "font.ft_face = %p\n", font->ft_face    );              /* free type font handle */
#endif
		fprintf( stream, "font.max_height = %d\n", font->max_height );
		fprintf( stream, "font.space_size = %d\n" , font->space_size );
		fprintf( stream, "font.spacing_x  = %d\n" , font->spacing_x );
		fprintf( stream, "font.spacing_y  = %d\n" , font->spacing_y );
		fprintf( stream, "font.max_ascend = %d\n", font->max_ascend );
		fprintf( stream, "font.max_descend = %d\n", font->max_descend );
	}
}

void print_asglyph( FILE* stream, ASFont* font, unsigned long c)
{
	if( font )
	{
		int i, k ;
		ASGlyph *asg = get_unicode_glyph( c, font );
		if( asg == NULL )
			return;

		fprintf( stream, "glyph[%lu].ASCII = %c\n", c, (char)c );
		fprintf( stream, "glyph[%lu].width = %d\n", c, asg->width  );
		fprintf( stream, "glyph[%lu].height = %d\n", c, asg->height  );
		fprintf( stream, "glyph[%lu].lead = %d\n", c, asg->lead  );
		fprintf( stream, "glyph[%lu].ascend = %d\n", c, asg->ascend);
		fprintf( stream, "glyph[%lu].descend = %d\n", c, asg->descend );
		k = 0 ;
		fprintf( stream, "glyph[%lu].pixmap = {", c);
#if 1
		for( i = 0 ; i < asg->height*asg->width ; i++ )
		{
			if( asg->pixmap[k]&0x80 )
			{
				fprintf( stream, "%2.2X ", ((asg->pixmap[k]&0x7F)<<1));
			}else
			{
				int count = asg->pixmap[k]&0x3F;
				if( asg->pixmap[k]&0x40 )
					fprintf( stream, "FF(%d times) ", count+1 );
				else
					fprintf( stream, "00(%d times) ", count+1 );
				i += count ;
			}
		    k++;
		}
#endif
		fprintf( stream, "}\nglyph[%lu].used_memory = %d\n", c, k );
	}
}


#ifndef HAVE_XRENDER
Bool afterimage_uses_xrender(){ return False;}
	
void
draw_text_xrender(  ASVisual *asv, const void *text, ASFont *font, ASTextAttributes *attr, int length,
					int xrender_op, unsigned long	xrender_src, unsigned long xrender_dst,
					int	xrender_xSrc,  int xrender_ySrc, int xrender_xDst, int xrender_yDst )
{}
#else
Bool afterimage_uses_xrender(){ return True;}

void
draw_text_xrender(  ASVisual *asv, const void *text, ASFont *font, ASTextAttributes *attr, int length,
					Picture	xrender_src, Picture xrender_dst,
					int	xrender_xSrc,  int xrender_ySrc, int xrender_xDst, int xrender_yDst )
{
	ASGlyphMap map;
	int max_gid = 0 ;
	int i ;
	int missing_glyphs = 0 ;
	int glyphs_bmap_size = 0, max_height = 0 ;

	if( !get_text_glyph_map( text, font, &map, attr, length) )
		return;
	
	if( map.width == 0 ) 
		return;
	/* xrender code starts here : */
	/* Step 1: we have to make sure we have a valid GlyphSet */
	if( font->xrender_glyphset == 0 ) 
		font->xrender_glyphset = XRenderCreateGlyphSet (asv->dpy, asv->xrender_mask_format);
	/* Step 2: we have to make sure all the glyphs are in GlyphSet */
	for( i = 0 ; map.glyphs[i] != GLYPH_EOT ; ++i ) 
		if( map.glyphs[i] > MAX_SPECIAL_GLYPH && map.glyphs[i]->xrender_gid == 0 ) 
		{
			glyphs_bmap_size += map.glyphs[i]->width * map.glyphs[i]->height ;
			if( map.glyphs[i]->height > max_height ) 
				max_height = map.glyphs[i]->height ;
			++missing_glyphs;
		}
	
	if( missing_glyphs > 0 ) 
	{
		Glyph		*gids;
		XGlyphInfo	*glyphs;
		char *bitmap, *bmap_ptr ;
		int	 nbytes_bitmap = 0;
		CARD8 **scanlines ;

		scanlines = safecalloc(max_height, sizeof(CARD8*));

		bmap_ptr = bitmap = safemalloc( glyphs_bmap_size );
		glyphs = safecalloc( missing_glyphs, sizeof(XGlyphInfo));
		gids = safecalloc( missing_glyphs, sizeof(Glyph));
		for( i = 0 ; map.glyphs[i] != GLYPH_EOT ; ++i ) 
			if( map.glyphs[i] > MAX_SPECIAL_GLYPH && map.glyphs[i]->xrender_gid == 0 ) 
			{	
				ASGlyph *asg = map.glyphs[i];
				int k = asg->height ;  
				char *ptr = bmap_ptr + asg->width*asg->height ;
				bmap_ptr = ptr ; 				   
				while ( --k >= 0 )
				{
					ptr -= asg->width ;	  
					scanlines[k] = ptr ;
				}		
				render_asglyph( scanlines, asg->pixmap,	0, 0, asg->width, asg->height, 0xFF );
				gids[i] = 
			}
		XRenderAddGlyphs( asv->dpy, font->xrender_glyphset, gids, glyphs, missing_glyphs, bitmap, nbytes_bitmap );
		free( gids );
		free( glyphs );
		free( bitmap );
		free( scanlines );
	}
	/* Step 3: actually rendering text  : */
	if( max_gid <= 255 ) 
	{
		char *string = safemalloc( map->glyphs_num-1 );
		for( i = 0 ; map.glyphs[i] != GLYPH_EOT ; ++i ) 
			string[i] = map.glyphs[i]->xrender_gid ;
		XRenderCompositeString8 ( asv->dpy, PictOpOver, xrender_src, xrender_dst, 
								  asv->xrender_mask_format,
			  					  font->xrender_glyphset, 
								  xrender_xSrc,xrender_ySrc,xrender_xDst,xrender_yDst,
								  string, i);
		free( string );
	}else if( max_gid <= 65000 ) 	
	{
		unsigned short *string = safemalloc( sizeof(unsigned short)*(map->glyphs_num-1) );
		for( i = 0 ; map.glyphs[i] != GLYPH_EOT ; ++i ) 
			string[i] = map.glyphs[i]->xrender_gid ;
		XRenderCompositeString16 (asv->dpy, PictOpOver, xrender_src, xrender_dst, 
								  asv->xrender_mask_format,
			  					  font->xrender_glyphset, 
								  xrender_xSrc,xrender_ySrc,xrender_xDst,xrender_yDst,
								  string, i);
		free( string );
	}else
	{
		unsigned int *string = safemalloc( sizeof(int)*(map->glyphs_num-1) );	
		for( i = 0 ; map.glyphs[i] != GLYPH_EOT ; ++i ) 
			string[i] = map.glyphs[i]->xrender_gid ;
		XRenderCompositeString32 (asv->dpy, PictOpOver, xrender_src, xrender_dst, 
								  asv->xrender_mask_format,
			  					  font->xrender_glyphset, 
								  xrender_xSrc,xrender_ySrc,xrender_xDst,xrender_yDst,
								  string, i);
		free( string );
	}	 
	
	/* xrender code ends here : */
	free_glyph_map( &map, True );	  
}

#endif


/*********************************************************************************/
/* The end !!!! 																 */
/*********************************************************************************/

