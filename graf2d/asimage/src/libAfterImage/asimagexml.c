/*
 * Copyright (c) 2001 Sasha Vasko <sasha@aftercode.net>
 * Copyright (c) 2001 Eric Kowalski <eric@beancrock.net>
 * Copyright (c) 2001 Ethan Fisher <allanon@crystaltokyo.com>
 *
 * This module is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 */

#undef LOCAL_DEBUG
#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

#include <ctype.h>
#include <math.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
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
#ifndef _WIN32
#include <sys/times.h>
#endif

#ifdef _WIN32
# include "win32/afterbase.h"
#else
# include "afterbase.h"
#endif
#include "afterimage.h"
#include "imencdec.h"

static char* cdata_str = XML_CDATA_STR;

/****h* libAfterImage/asimagexml
 * NAME
 * ascompose is a tool to compose image(s) and display/save it based on
 * supplied XML input file.
 *
 * DESCRIPTION
 * ascompose reads supplied XML data, and manipulates image accordingly.
 * It could transform images from files of any supported file format,
 * draw gradients, render antialiased texturized text, perform
 * superimposition of arbitrary number of images, and save images into
 * files of any of supported output file formats.
 *
 * At any point, the result of any operation could be assigned a name,
 * and later on referenced under this name.
 *
 * At any point during the script processing, result of any operation
 * could be saved into a file of any supported file types.
 *
 * Internal image format is 32bit ARGB with 8bit per channel.
 *
 * Last image referenced, will be displayed in X window, unless -n option
 * is specified. If -r option is specified, then this image will be
 * displayed in root window of X display, effectively setting a background
 * for a desktop. If -o option is specified, this image will also be
 * saved into the file or requested type.
 *
 * TAGS
 * 
 * Here is the list and description of possible XML tags to use in the
 * script :
 * 	img       - load image from the file.
 * 	recall    - recall previously loaded/generated image by its name.
 * 	text      - render text string into new image.
 * 	save      - save an image into the file.
 * 	bevel     - draw solid bevel frame around the image.
 * 	gradient  - render multipoint gradient.
 * 	mirror    - create mirror copy of an image.
 * 	blur      - perform gaussian blur on an image.
 * 	rotate    - rotate/flip image in 90 degree increments.
 * 	scale     - scale an image to arbitrary size.
 * 	slice     - enlarge image to arbitrary size leaving corners unchanged.
 * 	crop      - crop an image to arbitrary size.
 * 	tile      - tile an image to arbitrary size.
 * 	hsv       - adjust Hue, Saturation and Value of an image.
 * 	pad       - pad image with solid color from either or all sides.
 * 	solid     - generate new image of requested size, filled with solid
 *              color.
 * 	composite - superimpose arbitrary number of images using one of 15
 *              available methods.
 *  if        - conditional processing based on value of the variables
 *  set       - sets value of the variable
 *  printf    - formated printing of the value of the variable
 *
 * Each tag generates new image as the result of the transformation -
 * existing images are never modified and could be reused as many times
 * as needed. See below for description of each tag.
 *
 * Whenever numerical values are involved, the basic math ops (add,
 * subtract, multiply, divide), unary minus, and parentheses are
 * supported.
 *
 * Operator precedence is NOT supported.  Percentages are allowed, and
 * apply to either width or height of the appropriate image (usually
 * the refid image).
 *
 * Also, variables of the form $image.width and $image.height are
 * supported.  $image.width is the width of the image with refid "image",
 * and $image.height is the height of the same image.  The special
 * $xroot.width and $xroot.height values are defined by the the X root
 * window, if there is one.  This allows images to be scaled to the
 * desktop size: <scale width="$xroot.width" height="$xroot.height">.
 *
 * Each tag is only allowed to return ONE image.
 *
* 
 *****/

static ASImageManager *_as_xml_image_manager = NULL ;
static ASFontManager *_as_xml_font_manager = NULL ;

void set_xml_image_manager( ASImageManager *imman )
{
	_as_xml_image_manager = imman ;
}
void set_xml_font_manager( ASFontManager *fontman )
{
	_as_xml_font_manager = fontman ;
}




ASImageManager *create_generic_imageman(const char *path)		
{
	ASImageManager *my_imman = NULL ;
	char *path2 = copy_replace_envvar( getenv( ASIMAGE_PATH_ENVVAR ) );
	show_progress("image path is \"%s\".", path2?path2:"(null)" );
	if( path != NULL )
		my_imman = create_image_manager( NULL, SCREEN_GAMMA, path, path2, NULL );
	else
		my_imman = create_image_manager( NULL, SCREEN_GAMMA, path2, NULL );
	LOCAL_DEBUG_OUT( "created image manager %p with search path \"%s\"", my_imman, my_imman->search_path[0] );
	if( path2 )
		free( path2 );
	return my_imman;
}

ASFontManager *create_generic_fontman(Display *dpy, const char *path)		   
{
	ASFontManager  *my_fontman ;
	char *path2 = copy_replace_envvar( getenv( ASFONT_PATH_ENVVAR ) );
	if( path != NULL )
	{
		if( path2 != NULL )
		{
			int path_len = strlen(path);
			char *full_path = safemalloc( path_len+1+strlen(path2)+1);
			strcpy( full_path, path );
			full_path[path_len] = ':';
			strcpy( &(full_path[path_len+1]), path2 );
			free( path2 );
			path2 = full_path ;
		}else
			path2 = (char*)path ;
	}
	my_fontman = create_font_manager( dpy, path2, NULL );
	if( path2 && path2 != path )
		free( path2 );

	return my_fontman;
}

ASImage *
compose_asimage_xml_from_doc(ASVisual *asv, ASImageManager *imman, ASFontManager *fontman, xml_elem_t* doc, ASFlagType flags, int verbose, Window display_win, const char *path, int target_width, int target_height)
{
	/* Build the image(s) from the xml document structure. */
	ASImage* im = NULL;
	ASImageManager *my_imman = imman, *old_as_xml_imman = _as_xml_image_manager ;
	ASFontManager  *my_fontman = fontman, *old_as_xml_fontman = _as_xml_font_manager ;
	int my_imman_curr_dir_path_idx = MAX_SEARCH_PATHS ;

	if (doc)
	{
		int old_target_width = -1;
		int old_target_height = -1;
		xml_elem_t* ptr;
		Bool local_dir_included = False ;

	    asxml_var_init();
#if (HAVE_AFTERBASE_FLAG==1)
		if (verbose > 1) 
		{
			xml_print(doc);
			fprintf(stderr, "\n");
		}
#endif

		if( my_imman == NULL )
		{	
			if( _as_xml_image_manager == NULL )
			{
				local_dir_included	  = True ;
				_as_xml_image_manager = create_generic_imageman( path );/* we'll want to reuse it in case of recursion */
			}
			my_imman = _as_xml_image_manager ;
		}

		if( !local_dir_included )
		{
			register int i = 0;
			char **paths = my_imman->search_path ;
			while( i < MAX_SEARCH_PATHS && paths[i] != NULL ) ++i;
			if( i < MAX_SEARCH_PATHS ) 
			{	
				paths[i] = mystrdup(path) ;			
				paths[i+1] = NULL ;
				my_imman_curr_dir_path_idx = i ;
			}
		}	 

		if( my_fontman == NULL )
		{
			if( _as_xml_font_manager == NULL )
				_as_xml_font_manager = create_generic_fontman( asv->dpy, path );
			my_fontman = _as_xml_font_manager ;
		}

		/* save old target size to be restored at the end */		
		old_target_width = asxml_var_get(ASXMLVAR_TargetWidth);
		old_target_height = asxml_var_get(ASXMLVAR_TargetHeight);
		/* set current target size */		
		asxml_var_insert(ASXMLVAR_TargetWidth, target_width);
		asxml_var_insert(ASXMLVAR_TargetHeight, target_height);

		for (ptr = doc->child ; ptr ; ptr = ptr->next) {
			ASImage* tmpim = build_image_from_xml(asv, my_imman, my_fontman, ptr, NULL, flags, verbose, display_win);
			if (tmpim && im) safe_asimage_destroy(im);
			if (tmpim) im = tmpim;
		}
		if (im && (target_width > 0 || target_height > 0) )
		  {
			int scale_width = (target_width>0)?target_width:im->width;
			int scale_height = (target_height>0)?target_height:im->height;
			if (im->width != scale_width || im->height != scale_height)
			  {
			  	ASImage *tmp = scale_asimage( asv, im, scale_width, scale_height, ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT );
				if (tmp != NULL)
				  {
				  	safe_asimage_destroy(im);
					im = tmp;
				  }						
			  }
		  }
		/* restore old target size to be restored at the end */		
		asxml_var_insert(ASXMLVAR_TargetWidth, old_target_width);
		asxml_var_insert(ASXMLVAR_TargetHeight, old_target_height);

LOCAL_DEBUG_OUT( "result im = %p, im->imman	= %p, my_imman = %p, im->magic = %8.8lX", im, im?im->imageman:NULL, my_imman, im?im->magic:0 );
		
		if( my_imman_curr_dir_path_idx < MAX_SEARCH_PATHS && my_imman->search_path[my_imman_curr_dir_path_idx]) 
		{
			free(my_imman->search_path[my_imman_curr_dir_path_idx]);
			my_imman->search_path[my_imman_curr_dir_path_idx] = NULL ;			
		}

		if( my_imman != imman && my_imman != old_as_xml_imman )
		{/* detach created image from imman to be destroyed : */
			if( im && im->imageman == my_imman )
				forget_asimage( im );
			destroy_image_manager(my_imman, False);
		}

		if( my_fontman != fontman && my_fontman != old_as_xml_fontman  )
			destroy_font_manager(my_fontman, False);
		/* must restore managers to its original state */
		_as_xml_image_manager = old_as_xml_imman   ;
		_as_xml_font_manager =  old_as_xml_fontman ;
		
	}
	LOCAL_DEBUG_OUT( "returning im = %p, im->imman	= %p, im->magic = %8.8lX", im, im?im->imageman:NULL, im?im->magic:0 );
	return im;
}

ASImage *
compose_asimage_xml_at_size(ASVisual *asv, ASImageManager *imman, ASFontManager *fontman, char *doc_str, ASFlagType flags, int verbose, Window display_win, const char *path, int target_width, int target_height)
{
	xml_elem_t* doc = xml_parse_doc(doc_str, NULL);
	ASImage *im = compose_asimage_xml_from_doc(asv, imman, fontman, doc, flags, verbose, display_win, path, target_width, target_height);
	if (doc)
		xml_elem_delete(NULL, doc);
	return im;
}

inline ASImage *
compose_asimage_xml(ASVisual *asv, ASImageManager *imman, ASFontManager *fontman, char *doc_str, ASFlagType flags, int verbose, Window display_win, const char *path)
{
	xml_elem_t* doc = xml_parse_doc(doc_str, NULL);
	ASImage *im = compose_asimage_xml_from_doc(asv, imman, fontman, doc, flags, verbose, display_win, path, -1, -1);
	if (doc)
		xml_elem_delete(NULL, doc);
	return im;
}


Bool save_asimage_to_file(const char *file2bsaved, ASImage *im,
	           const char *strtype,
			   const char *compress,
			   const char *opacity,
			   int delay, int replace)
{
	ASImageExportParams params ;

	memset( &params, 0x00, sizeof(params) );
	params.gif.flags = EXPORT_ALPHA ;
	if (strtype == NULL || !mystrcasecmp(strtype, "jpeg") || !mystrcasecmp(strtype, "jpg"))  {
		params.type = ASIT_Jpeg;
		params.jpeg.quality = (compress==NULL)?-1:100-atoi(compress);
		if( params.jpeg.quality > 100 )
			params.jpeg.quality = 100;
	} else if (!mystrcasecmp(strtype, "bitmap") || !mystrcasecmp(strtype, "bmp")) {
		params.type = ASIT_Bmp;
	} else if (!mystrcasecmp(strtype, "png")) {
		params.type = ASIT_Png;
		params.png.compression = (compress==NULL)?-1:atoi(compress);
		if( params.png.compression > 99 )
			params.png.compression = 99;
	} else if (!mystrcasecmp(strtype, "xcf")) {
		params.type = ASIT_Xcf;
	} else if (!mystrcasecmp(strtype, "ppm")) {
		params.type = ASIT_Ppm;
	} else if (!mystrcasecmp(strtype, "pnm")) {
		params.type = ASIT_Pnm;
	} else if (!mystrcasecmp(strtype, "ico")) {
		params.type = ASIT_Ico;
	} else if (!mystrcasecmp(strtype, "cur")) {
		params.type = ASIT_Cur;
	} else if (!mystrcasecmp(strtype, "gif")) {
		params.type = ASIT_Gif;
		params.gif.flags |= EXPORT_APPEND ;
		params.gif.opaque_threshold = (opacity==NULL)?127:atoi(opacity) ;
		params.gif.dither = (compress==NULL)?3:atoi(compress)/17;
		if( params.gif.dither > 6 )
			params.gif.dither = 6;
		params.gif.animate_delay = delay ;
	} else if (!mystrcasecmp(strtype, "xpm")) {
		params.type = ASIT_Xpm;
		params.xpm.opaque_threshold = (opacity==NULL)?127:atoi(opacity) ;
		params.xpm.dither = (compress==NULL)?3:atoi(compress)/17;
		if( params.xpm.dither > 6 )
			params.xpm.dither = 6;
	} else if (!mystrcasecmp(strtype, "xbm")) {
		params.type = ASIT_Xbm;
	} else if (!mystrcasecmp(strtype, "tiff")) {
		params.type = ASIT_Tiff;
		params.tiff.compression_type = TIFF_COMPRESSION_NONE ;
		if( compress )
		{
			if( mystrcasecmp( compress, "deflate" ) == 0 )
				params.tiff.compression_type = TIFF_COMPRESSION_DEFLATE ;
			else if( mystrcasecmp( compress, "jpeg" ) == 0 )
				params.tiff.compression_type = TIFF_COMPRESSION_JPEG ;
			else if( mystrcasecmp( compress, "ojpeg" ) == 0 )
				params.tiff.compression_type = TIFF_COMPRESSION_OJPEG ;
			else if( mystrcasecmp( compress, "packbits" ) == 0 )
				params.tiff.compression_type = TIFF_COMPRESSION_PACKBITS ;
		}
	} else {
		show_error("File type not found.");
		return(0);
	}

	if( replace && file2bsaved )
		unlink( file2bsaved );

	return ASImage2file(im, NULL, file2bsaved, params.type, &params);

}

void show_asimage(ASVisual *asv, ASImage* im, Window w, long delay)
{
#ifndef X_DISPLAY_MISSING
	if ( im && w && asv)
	{
		Pixmap p = asimage2pixmap(asv, w, im, NULL, False);
		struct timeval value;

		XSetWindowBackgroundPixmap( asv->dpy, w, p );
		XClearWindow( asv->dpy, w );
		XFlush( asv->dpy );
		XFreePixmap( asv->dpy, p );
		p = None ;
		value.tv_usec = delay % 10000;
		value.tv_sec = delay / 10000;
		PORTABLE_SELECT (1, 0, 0, 0, &value);
	}
#endif /* X_DISPLAY_MISSING */
}

typedef struct ASImageXMLState
{
	ASFlagType 		flags ;
 	ASVisual 		*asv;
	ASImageManager 	*imman ;
	ASFontManager 	*fontman ;

	int verbose ;
	Window display_win ;
	
}ASImageXMLState;


ASImage *commit_xml_image_built( ASImageXMLState *state, char *id, ASImage *result )
{	
	if (state && id && result) 
	{
    	char* buf = NEW_ARRAY(char, strlen(id) + 1 + 6 + 1);
		if( state->verbose > 1 ) 
			show_progress("Storing image id [%s] with image manager %p .", id, state->imman);
    	sprintf(buf, "%s.width", id);
        asxml_var_insert(buf, result->width);
        sprintf(buf, "%s.height", id);
        asxml_var_insert(buf, result->height);
        free(buf);
		if( result->imageman != NULL )
		{
			ASImage *tmp = clone_asimage(result, SCL_DO_ALL );
			safe_asimage_destroy(result );
			result = tmp ;
		}
		if( result )
		{
			if( !store_asimage( state->imman, result, id ) )
			{
				show_warning("Failed to store image id [%s].", id);
				//safe_asimage_destroy(result );
				//result = fetch_asimage( state->imman, id );
				/*show_warning("Old image with the name fetched as %p.", result);*/
			}else
			{
				/* normally generated image will be destroyed right away, so we need to
			 	* increase ref count, in order to preserve it for future uses : */
				dup_asimage( result );
			}
		}
	}
	return result;
}

static void
translate_tag_size(	const char *width_str, const char *height_str, ASImage *imtmp, ASImage *refimg, int *width_ret, int *height_ret )
{
	int width_ref = 0;
	int height_ref = 0;
	int width = 0, height = 0 ; 
	LOCAL_DEBUG_OUT("width_str = \"%s\", height_str = \"%s\", imtmp = %p, refimg = %p", width_str?width_str:"(null)", height_str?height_str:"(null)", imtmp, refimg ); 
	
	if( imtmp ) 
	{	
		width_ref = width = imtmp->width ;
		height_ref = height = imtmp->height ;
	}
	if (refimg) 
	{
		width_ref = refimg->width;
		height_ref = refimg->height;
	}
	if( width_str ) 
	{	
		if( width_str[0] == '$' || isdigit( (int)width_str[0] ) )
			width = (int)parse_math(width_str, NULL, width);
	}
	if( height_str ) 
	{	
		if( height_str[0] == '$' || isdigit( (int)height_str[0] ) )
			height = (int)parse_math(height_str, NULL, height);
	}
	if( width_str && height_ref > 0 && mystrcasecmp(width_str,"proportional") == 0 )
		width = (width_ref * height) / height_ref ;
	else if( height_str && width_ref > 0 && mystrcasecmp(height_str,"proportional") == 0 )
		height = (height_ref * width) / width_ref ;
	if( width_ret ) 
		*width_ret = (width==0)?(imtmp?imtmp->width:(refimg?refimg->width:0)):width;
	if( height_ret ) 
		*height_ret = (height==0)?(imtmp?imtmp->height:(refimg?refimg->height:0)):height;

	LOCAL_DEBUG_OUT("width = %d, height = %d", *width_ret, *height_ret ); 

}

/****** libAfterImage/asimagexml/text
 * NAME
 * text - render text string into new image, using specific font, size
 *        and texture.
 * SYNOPSIS
 * <text id="new_id" font="font" point="size" fgcolor="color"
 *       bgcolor="color" fgimage="image_id" bgimage="image_id"
 *       spacing="points" type="3dtype">My Text Here</text>
 * ATTRIBUTES
 * id       Optional.  Image will be given this name for future reference.
 * font     Optional.  Default is "fixed".  Font to use for text.
 * point    Optional.  Default is 12.  Size of text in points.
 * fgcolor  Optional.  No default.  The text will be drawn in this color.
 * bgcolor  Optional.  No default.  The area behind the text will be drawn
 *          in this color.
 * fgimage  Optional.  No default.  The text will be textured by this image.
 * bgimage  Optional.  No default.  The area behind the text will be filled
 *          with this image.
 * spacing  Optional.  Default 0.  Extra pixels to place between each glyph.
 * type     Optional.  Default 0.  Valid values are from 0 to 7 and each 
 * 			represeend different 3d type.
 * NOTES
 * <text> without bgcolor, fgcolor, fgimage, or bgimage will NOT
 * produce visible output by itself.  See EXAMPLES below.
 ******/
static ASImage *
handle_asxml_tag_text( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm )
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	const char* text = NULL;
	const char* font_name = "fixed";
	const char* fgimage_str = NULL;
	const char* bgimage_str = NULL;
	const char* fgcolor_str = NULL;
	const char* bgcolor_str = NULL;
	ARGB32 fgcolor = ARGB32_White, bgcolor = ARGB32_Black;
	int point = 12, spacing = 0, type = AST_Plain;
	unsigned int width = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "font")) font_name = ptr->parm;
		else if (!strcmp(ptr->tag, "point")) point = strtol(ptr->parm, NULL, 0);
		else if (!strcmp(ptr->tag, "spacing")) spacing = strtol(ptr->parm, NULL, 0);
		else if (!strcmp(ptr->tag, "fgimage")) fgimage_str = ptr->parm;
		else if (!strcmp(ptr->tag, "bgimage")) bgimage_str = ptr->parm;
		else if (!strcmp(ptr->tag, "fgcolor")) fgcolor_str = ptr->parm;
	   	else if (!strcmp(ptr->tag, "bgcolor")) bgcolor_str = ptr->parm;
		else if (!strcmp(ptr->tag, "type")) type = strtol(ptr->parm, NULL, 0);
		else if (!strcmp(ptr->tag, "width")) width = strtol(ptr->parm, NULL, 0);
	}
	for (ptr = doc->child ; ptr && text == NULL ; ptr = ptr->next)
		if (!strcmp(ptr->tag, cdata_str)) text = ptr->parm;
	
	if (text && point > 0) 
	{
		struct ASFont *font = NULL;
		if( state->verbose > 1 ) 
			show_progress("Rendering text [%s] with font [%s] size [%d].", text, font_name, point);
		if (state->fontman) font = get_asfont(state->fontman, font_name, 0, point, ASF_GuessWho);
		if (font != NULL) 
		{
		  ASTextAttributes attr = {ASTA_VERSION_INTERNAL, 0, 0, ASCT_Char, 8, 0, NULL, 0, ARGB32_White, width};
			attr.type = type ;
			if( IsUTF8Locale() ) 
				attr.char_type = ASCT_UTF8 ;
			set_asfont_glyph_spacing(font, spacing, 0);
			if( fgcolor_str ) 
				parse_argb_color(fgcolor_str, &(attr.fore_color) );
			
			result = draw_fancy_text( text, font, &attr, 0, 0/*autodetect length*/ );
			if (result && fgcolor_str) {
#if 0			   
				result->back_color = attr.fore_color ;
#else
				ASImage* fgimage = create_asimage(result->width, result->height, ASIMAGE_QUALITY_TOP);
				parse_argb_color(fgcolor_str, &fgcolor);
				fill_asimage(state->asv, fgimage, 0, 0, result->width, result->height, fgcolor);
				move_asimage_channel(fgimage, IC_ALPHA, result, IC_ALPHA);
				safe_asimage_destroy(result);
				result = fgimage ;
#endif
			}
			if (result && fgimage_str) {
				ASImage* fgimage = NULL;
				fgimage = get_asimage(state->imman, fgimage_str, 0xFFFFFFFF, 100 );
				if( state->verbose > 1 ) 
					show_progress("Using image [%s](%p) as foreground. Text size is %dx%d", fgimage_str, fgimage, result->width, result->height);
				if (fgimage) {
					ASImage *tmp = tile_asimage(state->asv, fgimage, 0, 0, result->width, result->height, 0, ASA_ASImage, 100, ASIMAGE_QUALITY_TOP);
					if( tmp )
					{
					   	release_asimage( fgimage );
						fgimage = tmp ;
					}
					move_asimage_channel(fgimage, IC_ALPHA, result, IC_ALPHA);
					safe_asimage_destroy(result);
					result = fgimage;
				}
			}
			if (result && (bgcolor_str || bgimage_str)) {
				ASImageLayer layers[2];
				init_image_layers(&(layers[0]), 2);
				if (bgimage_str) layers[0].im = fetch_asimage(state->imman, bgimage_str);
				if (bgcolor_str)
					if( parse_argb_color(bgcolor_str, &bgcolor) != bgcolor_str )
					{
						if( layers[0].im != NULL )
							layers[0].im->back_color = bgcolor ;
						else
							layers[0].solid_color = bgcolor ;
					}
				result->back_color = fgcolor ;
				layers[0].dst_x = 0;
				layers[0].dst_y = 0;
				layers[0].clip_width = result->width;
				layers[0].clip_height = result->height;
				layers[0].bevel = NULL;
				layers[1].im = result;
				layers[1].dst_x = 0;
				layers[1].dst_y = 0;
				layers[1].clip_width = result->width;
				layers[1].clip_height = result->height;
				result = merge_layers(state->asv, layers, 2, result->width, result->height, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
				safe_asimage_destroy( layers[0].im );
			}
		}
	}

	return result;
}	
/****** libAfterImage/asimagexml/composite
 * NAME
 * composite - superimpose arbitrary number of images on top of each
 * other.
 * SYNOPSIS
 * <composite id="new_id" op="op_desc"
 *            keep-transparency="0|1" merge="0|1">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * op       Optional. Default is "alphablend". The compositing operation.
 *          Valid values are the standard AS blending ops: add, alphablend,
 *          allanon, colorize, darken, diff, dissipate, hue, lighten,
 *          overlay, saturate, screen, sub, tint, value.
 * merge    Optional. Default is "expand". Valid values are "clip" and
 *          "expand". Determines whether final image will be expanded to
 *          the maximum size of the layers, or clipped to the bottom
 *          layer.
 * keep-transparency
 *          Optional. Default is "0". Valid values are "0" and "1". If
 *          set to "1", the transparency of the bottom layer will be
 *          kept for the final image.
 * NOTES
 * All images surrounded by this tag will be composited with the given op.
 *
 * ATTRIBUTES
 *  All tags surrounded by this tag may have some of the common attributes
 *  in addition to their normal ones.  Under no circumstances is there a 
 *  conflict with the normal child attributes:
 * 
 * crefid   Optional. An image ID defined with the "id" parameter for
 *          any previously created image. If set, percentages in "x"
 *          and "y" will be derived from the width and height of the
 *          crefid image.
 * x        Optional. Default is 0. Pixel coordinate of left edge.
 * y        Optional. Default is 0. Pixel coordinate of top edge.
 * align    Optional. Alternative to x - allowed values are right, center
 *          and left.
 * valign   Optional. Alternative to y - allowed values are top, middle
 *          and bottom.
 * clip_x   Optional. Default is 0. X Offset on infinite surface tiled
 *          with this image, from which to cut portion of an image to be
 *          used in composition.
 * clip_y   Optional. Default is 0. Y Offset on infinite surface tiled
 *          with this image, from which to cut portion of an image to be
 *          used in composition.
 * clip_width
 *          Optional. Default is image width. Tile image to this width
 *          prior to superimposition.
 * clip_height
 *          Optional. Default is image height. Tile image to this height
 *          prior to superimposition.
 * tile     Optional. Default is 0. If set will cause image to be tiled
 *          across entire composition, unless overridden by clip_width or
 *          clip_height.
 * tint     Optional. Additionally tint an image to specified color.
 *          Tinting can both lighten and darken an image. Tinting color
 *          0 or #7f7f7f7f yields no tinting. Tinting can be performed
 *          on any channel, including alpha channel.
 * SEE ALSO
 * libAfterImage
 ******/
static ASImage *
handle_asxml_tag_composite( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm )
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	const char* pop = "alphablend";
	int keep_trans = 0;
	int merge = 0;
	int num = 0;
	int width = 0, height = 0;
	ASImageLayer *layers;
#define  ASXML_ALIGN_LEFT 	(0x01<<0)
#define  ASXML_ALIGN_RIGHT 	(0x01<<1)
#define  ASXML_ALIGN_TOP    (0x01<<2)
#define  ASXML_ALIGN_BOTTOM (0x01<<3)
	int *align ;
	int i ;
	merge_scanlines_func op_func = NULL ;

	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "op")) { pop = ptr->parm; op_func = blend_scanlines_name2func(pop); }
		else if (!strcmp(ptr->tag, "keep-transparency")) keep_trans = strtol(ptr->parm, NULL, 0);
		else if (!strcmp(ptr->tag, "merge") && !mystrcasecmp(ptr->parm, "clip")) merge = 1;
	}
	/* Find out how many subimages we have. */
	for (ptr = doc->child ; ptr ; ptr = ptr->next) 
		if (strcmp(ptr->tag, cdata_str)) num++;

	if( num == 0 ) 
	{
		show_warning( "composite tag with no subimages to compose from specified!");	  
		return NULL;
	}

	
	if( op_func == NULL ) 
	{	
		LOCAL_DEBUG_OUT( "defaulting to alpha-blending%s","");
		op_func = alphablend_scanlines ;
	}
	/* Build the layers first. */
	layers = create_image_layers( num );
	align = safecalloc( num, sizeof(int));

	for (num = 0, ptr = doc->child ; ptr ; ptr = ptr->next) 
	{
		int x = 0, y = 0;
		int clip_x = 0, clip_y = 0;
		int clip_width = 0, clip_height = 0;
		ARGB32 tint = 0;
		Bool tile = False ;
		xml_elem_t* sparm = NULL;
		if (!strcmp(ptr->tag, cdata_str)) continue;
		if( (layers[num].im = build_image_from_xml(state->asv, state->imman, state->fontman, ptr, &sparm, state->flags, state->verbose, state->display_win)) != NULL )
		{
			clip_width = layers[num].im->width;
			clip_height = layers[num].im->height;
		}
		if (sparm) 
		{
			xml_elem_t* tmp;
			const char* x_str = NULL;
			const char* y_str = NULL;
			const char* clip_x_str = NULL;
			const char* clip_y_str = NULL;
			const char* clip_width_str = NULL;
			const char* clip_height_str = NULL;
			const char* refid = NULL;
			for (tmp = sparm ; tmp ; tmp = tmp->next) {
				if (!strcmp(tmp->tag, "crefid")) refid = tmp->parm;
				else if (!strcmp(tmp->tag, "x")) x_str = tmp->parm;
				else if (!strcmp(tmp->tag, "y")) y_str = tmp->parm;
				else if (!strcmp(tmp->tag, "clip_x")) clip_x_str = tmp->parm;
				else if (!strcmp(tmp->tag, "clip_y")) clip_y_str = tmp->parm;
				else if (!strcmp(tmp->tag, "clip_width")) clip_width_str = tmp->parm;
				else if (!strcmp(tmp->tag, "clip_height")) clip_height_str = tmp->parm;
				else if (!strcmp(tmp->tag, "tint")) parse_argb_color(tmp->parm, &tint);
				else if (!strcmp(tmp->tag, "tile")) tile = True;
				else if (!strcmp(tmp->tag, "align"))
				{
					if (!strcmp(tmp->parm, "left"))set_flags( align[num], ASXML_ALIGN_LEFT);
					else if (!strcmp(tmp->parm, "right"))set_flags( align[num], ASXML_ALIGN_RIGHT);
					else if (!strcmp(tmp->parm, "center"))set_flags( align[num], ASXML_ALIGN_LEFT|ASXML_ALIGN_RIGHT);
				}else if (!strcmp(tmp->tag, "valign"))
				{
					if (!strcmp(tmp->parm, "top"))set_flags( align[num], ASXML_ALIGN_TOP) ;
					else if (!strcmp(tmp->parm, "bottom"))set_flags( align[num], ASXML_ALIGN_BOTTOM);
					else if (!strcmp(tmp->parm, "middle"))set_flags( align[num], ASXML_ALIGN_TOP|ASXML_ALIGN_BOTTOM);
				}
			}
			if (refid) {
				ASImage* refimg = fetch_asimage(state->imman, refid);
				if (refimg) {
					x = refimg->width;
					y = refimg->height;
				}
				safe_asimage_destroy(refimg );
			}
			x = x_str ? (int)parse_math(x_str, NULL, x) : 0;
			y = y_str ? (int)parse_math(y_str, NULL, y) : 0;
			clip_x = clip_x_str ? (int)parse_math(clip_x_str, NULL, x) : 0;
			clip_y = clip_y_str ? (int)parse_math(clip_y_str, NULL, y) : 0;
			if( clip_width_str )
				clip_width = (int)parse_math(clip_width_str, NULL, clip_width);
			else if( tile )
				clip_width = 0 ;
			if( clip_height_str )
				clip_height = (int)parse_math(clip_height_str, NULL, clip_height);
			else if( tile )
				clip_height = 0 ;
		}
		if (layers[num].im) {
			layers[num].dst_x = x;
			layers[num].dst_y = y;
			layers[num].clip_x = clip_x;
			layers[num].clip_y = clip_y;
			layers[num].clip_width = clip_width ;
			layers[num].clip_height = clip_height ;
			layers[num].tint = tint;
			layers[num].bevel = 0;
			layers[num].merge_scanlines = op_func;
			if( clip_width + x > 0 )
			{
				if( width < clip_width + x )
					width = clip_width + x;
			}else
				if (width < (int)(layers[num].im->width)) width = layers[num].im->width;
			if( clip_height + y > 0 )
			{
				if( height < clip_height + y )
					height = clip_height + y ;
			}else
				if (height < (int)(layers[num].im->height)) height = layers[num].im->height;
			num++;
		}
		if (sparm) xml_elem_delete(NULL, sparm);
	}

	if (num && merge && layers[0].im ) {
		width = layers[0].im->width;
		height = layers[0].im->height;
	}


	for (i = 0 ; i < num ; i++)
	{
		if( get_flags(align[i], ASXML_ALIGN_LEFT|ASXML_ALIGN_RIGHT ) )
		{
			int im_width = ( layers[i].clip_width == 0 )? layers[i].im->width : layers[i].clip_width ;
			int x = 0 ;
			if( get_flags( align[i], ASXML_ALIGN_RIGHT ) )
				x = width - im_width ;
			if( get_flags( align[i], ASXML_ALIGN_LEFT ) )
				x /= 2;
			layers[i].dst_x = x;
		}
		if( get_flags(align[i], ASXML_ALIGN_TOP|ASXML_ALIGN_BOTTOM ) )
		{
			int im_height = ( layers[i].clip_height == 0 )? layers[i].im->height : layers[i].clip_height;
			int y = 0 ;
			if( get_flags( align[i], ASXML_ALIGN_BOTTOM ) )
				y = height - im_height ;
			if( get_flags( align[i], ASXML_ALIGN_TOP ) )
				y /= 2;
			layers[i].dst_y = y;
		}
		if( layers[i].clip_width == 0 )
			layers[i].clip_width = width - layers[i].dst_x;
		if( layers[i].clip_height == 0 )
			layers[i].clip_height = height - layers[i].dst_y;
	}

	if (state->verbose > 1) {
		show_progress("Compositing [%d] image(s) with op [%s].  Final geometry [%dx%d].", num, pop, width, height);
		if (keep_trans) show_progress("  Keeping transparency.");
	
		for (i = 0 ; i < num ; i++) {
			show_progress("  Image [%d] geometry [%dx%d+%d+%d]", i, layers[i].clip_width, layers[i].clip_height, layers[i].dst_x, layers[i].dst_y);
			if (layers[i].tint) show_progress(" tint (#%08x)", (unsigned int)layers[i].tint);
		}
	}

	result = merge_layers( state->asv, layers, num, width, height, 
							ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
	if (keep_trans && result && layers[0].im)
		copy_asimage_channel(result, IC_ALPHA, layers[0].im, IC_ALPHA);
	
	while (--num >= 0 )
		safe_asimage_destroy( layers[num].im );
	
	free(align);
	free(layers);

	return result;
}

/****** libAfterImage/asimagexml/img
 * NAME
 * img - load image from the file.
 * SYNOPSIS
 * <img id="new_img_id" src="filename"/>
 * ATTRIBUTES
 * id     Optional.  Image will be given this name for future reference.
 * src    Required.  The filename (NOT URL) of the image file to load.
 * NOTES
 * The special image src "xroot:" will import the background image
 * of the root X window, if any.  No attempt will be made to offset this
 * image to fit the location of the resulting window, if one is displayed.
 ******/
static ASImage *
handle_asxml_tag_img( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, int dst_width, int dst_height)
{
	ASImage *result = NULL ;
	const char* src = NULL;
	xml_elem_t* ptr ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "src")) src = ptr->parm;
	}
	if (src && !strcmp(src, "xroot:")) {
		unsigned int width, height;
		Pixmap rp = GetRootPixmap(None);
		if( state->verbose > 1 ) 
			show_progress("Getting root pixmap.");
		if (rp) {
			get_dpy_drawable_size( state->asv->dpy, rp, &width, &height);
			result = pixmap2asimage(state->asv, rp, 0, 0, width, height, 0xFFFFFFFF, False, 100);
			if( dst_width == 0 ) dst_width = width ; 
			if( dst_height == 0 ) dst_height = height ; 
			if( dst_width != (int)width || dst_height != (int)height ) 
			{
				ASImage *tmp = scale_asimage( NULL, result, dst_width, dst_height, ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT );
				if( tmp ) 
				{
					safe_asimage_destroy( result );
					result = tmp ;
				}  	
			}	
		}
	} else if (src) {
		if( state->verbose > 1 ) 
			show_progress("Loading image [%s] using imman (%p) with search path \"%s\" (dst_size = %dx%d).", src, state->imman, state->imman?state->imman->search_path[0]:"", dst_width, dst_height);
		if( dst_width != 0 || dst_height != 0 ) 
			result = get_thumbnail_asimage( state->imman, src, dst_width, dst_height, (dst_width==0||dst_height==0)?AS_THUMBNAIL_PROPORTIONAL:0 );
		else
			result = get_asimage( state->imman, src, 0xFFFFFFFF, 100 );
	}
	return result;
}	

/****** libAfterImage/asimagexml/recall
 * NAME
 * recall - recall previously generated and named image by its id.
 * SYNOPSIS
 * <recall id="new_id" srcid="image_id" default_src="filename"/>
 * ATTRIBUTES
 * id       Optional.  Image will be given this name for future reference.
 * srcid    Required.  An image ID defined with the "id" parameter for
 *          any previously created image.
 ******/
static ASImage *
handle_asxml_tag_recall( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr = parm ; 
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	while ( ptr && !result ) 
	{	
		if (!strcmp(ptr->tag, "srcid"))
		{ 
			if( state->verbose > 1 ) 
				show_progress("Recalling image id [%s] from imman %p.", ptr->parm, state->imman);
			result = fetch_asimage(state->imman, ptr->parm );
			if (!result)
				show_warning("Image recall failed for id [%s].", ptr->parm);
		}	
		ptr = ptr->next ;
	}
	if( result == NULL ) 
	{
		for( ptr = parm ; ptr && !result ; ptr = ptr->next )
			if (!strcmp(ptr->tag, "default_src"))
			{ 
				if( state->verbose > 1 ) 
					show_progress("loading default image [%s] from imman %p.", ptr->parm, state->imman);
				result = get_asimage( state->imman, ptr->parm, 0xFFFFFFFF, 100 );
			}
	}
	return result;
}	

/****** libAfterImage/asimagexml/release
 * NAME
 * release - release (destroy if possible) previously generated and named image by its id.
 * SYNOPSIS
 * <release srcid="image_id"/>
 * ATTRIBUTES
 * srcid    Required.  An image ID defined with the "id" parameter for
 *          any previously created image.
 ******/
static ASImage *
handle_asxml_tag_release( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	xml_elem_t* ptr ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
		if (!strcmp(ptr->tag, "srcid"))
		{
			if( state->verbose > 1 ) 
				show_progress("Releasing image id [%s] from imman %p.", ptr->parm, state->imman);
			release_asimage_by_name(state->imman, (char*)ptr->parm );
			break;
		}
	return NULL;
}	

/****** libAfterImage/asimagexml/color
 * NAME
 * color - defines symbolic name for a color and set of variables.
 * SYNOPSIS
 * <color name="sym_name" domain="var_domain" argb="colorvalue"/>
 * ATTRIBUTES
 * name   Symbolic name for the color value, to be used to refer to that color.
 * argb   8 characters hex definition of the color or other symbolic color name.
 * domain string to be used to prepend names of defined variables.
 * NOTES
 * In addition to defining symbolic name for the color this tag will define
 * 7 other variables : 	domain.sym_name.red, domain.sym_name.green, 
 * 					   	domain.sym_name.blue, domain.sym_name.alpha, 
 * 					  	domain.sym_name.hue, domain.sym_name.saturation,
 *                     	domain.sym_name.value
 ******/
static ASImage *
handle_asxml_tag_color( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	xml_elem_t* ptr ;
	char* name = NULL;
	const char* argb_text = NULL;
	const char* var_domain = NULL;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "name")) name = ptr->parm;
		else if (!strcmp(ptr->tag, "argb")) argb_text = ptr->parm;
		else if (!strcmp(ptr->tag, "domain")) var_domain = ptr->parm;
	}
	if (name && argb_text) 
	{
		ARGB32 argb = ARGB32_Black;
		if( parse_argb_color( argb_text, &argb ) != argb_text )
		{
			char *tmp;
			CARD32 hue16, sat16, val16 ;
			int vd_len = var_domain?strlen(var_domain):0 ;

			tmp = safemalloc( vd_len + 1+ strlen(name )+32+1 ) ;

			if( var_domain && var_domain[0] != '\0' )
			{
				if( var_domain[vd_len-1] != '.' )
				{
					sprintf( tmp, "%s.", var_domain );
					++vd_len ;
				}else
					strcpy( tmp, var_domain );
			}


#ifdef HAVE_AFTERBASE
	   		if( state->verbose > 1 ) 
				show_progress("defining synonim [%s] for color value #%8.8X.", name, argb);
	   		register_custom_color( name, argb );
#endif
			sprintf( tmp+vd_len, "%s.alpha", name );
			asxml_var_insert( tmp, ARGB32_ALPHA8(argb) );
			sprintf( tmp+vd_len, "%s.red", name );
			asxml_var_insert( tmp, ARGB32_RED8(argb) );
			sprintf( tmp+vd_len, "%s.green", name );
			asxml_var_insert( tmp, ARGB32_GREEN8(argb) );
			sprintf( tmp+vd_len, "%s.blue", name );
			asxml_var_insert( tmp, ARGB32_BLUE8(argb) );

			hue16 = rgb2hsv( ARGB32_RED16(argb), ARGB32_GREEN16(argb), ARGB32_BLUE16(argb), &sat16, &val16 );

			sprintf( tmp+vd_len, "%s.hue", name );
			asxml_var_insert( tmp, hue162degrees( hue16 ) );
			sprintf( tmp+vd_len, "%s.saturation", name );
			asxml_var_insert( tmp, val162percent( sat16 ) );
			sprintf( tmp+vd_len, "%s.value", name );
			asxml_var_insert( tmp, val162percent( val16 ) );
			free( tmp );
		}
	}
	return NULL;
}
/****** libAfterImage/asimagexml/printf
 * NAME
 * printf - prints variable value to standard output.
 * SYNOPSIS
 * <printf format="format_string" var="variable_name" val="expression"/>
 * ATTRIBUTES
 * format_string  Standard C format string with exactly 1 placeholder.
 * var            Name of the variable, which value will be printed.
 * val            math expression to be printed.
 * NOTES
 ******/
static ASImage *
handle_asxml_tag_printf( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	xml_elem_t* ptr ;
	const char* format = NULL;
	const char* var = NULL;
	int val = 0 ;
	Bool use_val = False ;
	int arg_count = 0, i;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "format")) format = ptr->parm;
		else if (!strcmp(ptr->tag, "var")) { var = ptr->parm; use_val = False; }
		else if (!strcmp(ptr->tag, "val")) { val = (int)parse_math(ptr->parm, NULL, 0); use_val = True; }
	}
   		
	if( format != NULL ) 
	{	
		char *interpreted_format = interpret_ctrl_codes( mystrdup(format) );
		
		for( i = 0 ; format[i] != '\0' ; ++i )
			if( format[i] == '%' )
			{
				if( format[i+1] != '%' ) 
			 		++arg_count ; 
				else 
					++i ;
			}
		
		if( use_val && arg_count == 1) 
			printf( interpreted_format, val );
		else if( var != NULL && arg_count == 1 ) 
			printf( interpreted_format, asxml_var_get(var) );				
		else if( arg_count == 0 )
			fputs( interpreted_format, stdout );				   
		free( interpreted_format );
	}
		
	return NULL;
}
/****** libAfterImage/asimagexml/set
 * NAME
 * set - declares variable, assigning it a numeric value of expression.
 * SYNOPSIS
 * <set var="variable_name" domain="var_domain" val="expression"/>
 * ATTRIBUTES
 * var            Name of the variable, which value will be set.
 * val            math expression to be evaluated.
 * domain         (optional) variable's domain to be prepended to its name
 *                using format var_domain.variable_name
 ******/
static ASImage *
handle_asxml_tag_set( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	xml_elem_t* ptr ;
	const char* var_domain = NULL ;
	const char* var = NULL;
	int val = 0 ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "var")) 			var = ptr->parm;
		else if (!strcmp(ptr->tag, "domain")) 	var_domain = ptr->parm;
		else if (!strcmp(ptr->tag, "val"))  	val = (int)parse_math(ptr->parm, NULL, 0);
	}
   		
	if( var != NULL ) 
	{	
		char *tmp = (char*)var ; 
		if( var_domain && var_domain[0] != '\0' )
		{
			int vd_len = strlen(var_domain);
			tmp = safemalloc( vd_len + 1 + strlen(var) + 1 );
			sprintf( tmp, ( var_domain[vd_len-1] != '.' )?"%s.%s":"%s%s", var_domain, var );
		}
		asxml_var_insert( tmp, val );
		if( tmp != var ) 
			free( tmp );
	}
		
	return NULL;
}
/****** libAfterImage/asimagexml/if
 * NAME
 * if - evaluates logical expression and if result evaluates to not true(or false 
 * if <unless> tag is used ), handles tags within.
 * SYNOPSIS
 * <if val1="expression" [op="gt|lt|ge|le|eq|ne" val2="expression"]/>
 * 	[<then>...</then><else>...</else>]
 * </if>
 * <unless val1="expression" [op="gt|lt|ge|le|eq|ne" val2="expression"]/>
 * ATTRIBUTES
 * val1            math expression to be evaluated.
 * val2            math expression to be evaluated.
 * op         	 (optional) comparison op to be applied to values
 * EXAMPLE
 * <if val1="$ascs.Base.value" val2="50" op="gt"><then>...</then><else>...</else></if>
 ******/
static ASImage *
handle_asxml_tag_if( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm)
{
	xml_elem_t* ptr ;
	int val1 = 0, val2 = 0 ;
	const char *op = NULL ;
	int res = 0 ; 
	ASImage *im = NULL, *imtmp = NULL  ; 
	LOCAL_DEBUG_OUT("doc = %p, parm = %p", doc, parm ); 
	
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "op")) 			op = ptr->parm;
		else if (!strcmp(ptr->tag, "val1"))  	val1 = (int)parse_math(ptr->parm, NULL, 0);
		else if (!strcmp(ptr->tag, "val2"))  	val2 = (int)parse_math(ptr->parm, NULL, 0);
	}
   		
	if( op != NULL ) 
	{	
		if     ( strcmp(op, "gt") == 0 ) res = (val1 > val2);
		else if( strcmp(op, "lt") == 0 ) res = (val1 < val2);
		else if( strcmp(op, "ge") == 0 ) res = (val1 >= val2);
		else if( strcmp(op, "le") == 0 ) res = (val1 <= val2);
		else if( strcmp(op, "eq") == 0 ) res = (val1 == val2);
		else if( strcmp(op, "ne") == 0 ) res = (val1 != val2);
	}

	if( doc->tag[0] == 'u' ) /* <unless> */
		res = !res ;

	ptr = NULL ;
	for (ptr = doc->child ; ptr ; ptr = ptr->next) 
	{
		if( strcmp(ptr->tag, res?"then":"else" ) )
		{
			ptr = ptr->child ;
			break;
		}
		if( res && ptr->next == NULL ) 
			ptr = doc->child ;
	}
	
	while( ptr ) 
	{
		imtmp = build_image_from_xml(state->asv, state->imman, state->fontman, ptr, NULL, state->flags, state->verbose, state->display_win);
		if( im && imtmp ) safe_asimage_destroy( im ); 
		if( imtmp ) im = imtmp ;
		 ptr = ptr->next ;
	}
	return im ;
}

/****** libAfterImage/asimagexml/gradient
 * NAME
 * gradient - render multipoint gradient.
 * SYNOPSIS
 * <gradient id="new_id" angle="degrees" 
 *           refid="refid" width="pixels" height="pixels"
 *           colors ="color1 color2 color3 [...]"
 *           offsets="fraction1 fraction2 fraction3 [...]"/>
 * ATTRIBUTES
 * id       Optional.  Image will be given this name for future reference.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Optional.  The result will have this width.
 * height   Optional.  The result will have this height.
 * colors   Required.  Whitespace-separated list of colors.  At least two
 *          colors are required.  Each color in this list will be visited
 *          in turn, at the intervals given by the offsets attribute.
 * offsets  Optional.  Whitespace-separated list of floating point values
 *          ranging from 0.0 to 1.0.  The colors from the colors attribute
 *          are given these offsets, and the final gradient is rendered
 *          from the combination of the two.  If both colors and offsets
 *          are given but the number of colors and offsets do not match,
 *          the minimum of the two will be used, and the other will be
 *          truncated to match.  If offsets are not given, a smooth
 *          stepping from 0.0 to 1.0 will be used.
 * angle    Optional.  Given in degrees.  Default is 0.  This is the
 *          direction of the gradient.  Currently the only supported
 *          values are 0, 45, 90, 135, 180, 225, 270, 315.  0 means left
 *          to right, 90 means top to bottom, etc.
 *****/
static ASImage *
handle_asxml_tag_gradient( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	double angle = 0;
	char* color_str = NULL;
	char* offset_str = NULL;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, width = %d, height = %d", doc, parm, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "angle")) angle = strtod(ptr->parm, NULL);
		else if (!strcmp(ptr->tag, "colors")) color_str = ptr->parm;
		else if (!strcmp(ptr->tag, "offsets")) offset_str = ptr->parm;
	}
	if ( color_str) 
	{
		ASGradient gradient;
		int reverse = 0, npoints1 = 0, npoints2 = 0;
		char* p;
		angle = fmod(angle, 2 * PI);
		if (angle > 2 * PI * 15 / 16 || angle < 2 * PI * 1 / 16) {
			gradient.type = GRADIENT_Left2Right;
		} else if (angle < 2 * PI * 3 / 16) {
			gradient.type = GRADIENT_TopLeft2BottomRight;
		} else if (angle < 2 * PI * 5 / 16) {
			gradient.type = GRADIENT_Top2Bottom;
		} else if (angle < 2 * PI * 7 / 16) {
			gradient.type = GRADIENT_BottomLeft2TopRight; reverse = 1;
		} else if (angle < 2 * PI * 9 / 16) {
			gradient.type = GRADIENT_Left2Right; reverse = 1;
		} else if (angle < 2 * PI * 11 / 16) {
			gradient.type = GRADIENT_TopLeft2BottomRight; reverse = 1;
		} else if (angle < 2 * PI * 13 / 16) {
			gradient.type = GRADIENT_Top2Bottom; reverse = 1;
		} else {
			gradient.type = GRADIENT_BottomLeft2TopRight;
		}
		for (p = color_str ; isspace((int)*p) ; p++);
		for (npoints1 = 0 ; *p ; npoints1++) {
			if (*p) for ( ; *p && !isspace((int)*p) ; p++);
			for ( ; isspace((int)*p) ; p++);
		}
		if (offset_str) {
			for (p = offset_str ; isspace((int)*p) ; p++);
			for (npoints2 = 0 ; *p ; npoints2++) {
				if (*p) for ( ; *p && !isspace((int)*p) ; p++);
				for ( ; isspace((int)*p) ; p++);
			}
		}
		gradient.npoints = max( npoints1, npoints2 );
		if (npoints1 > 1) {
			int i;
			gradient.color = safecalloc(gradient.npoints, sizeof(ARGB32));
			gradient.offset = NEW_ARRAY(double, gradient.npoints);
			for (p = color_str ; isspace((int)*p) ; p++);
			for (npoints1 = 0 ; *p ; ) {
				char* pb = p, ch;
				if (*p) for ( ; *p && !isspace((int)*p) ; p++);
				for ( ; isspace((int)*p) ; p++);
				ch = *p; *p = '\0';
				if (parse_argb_color(pb, gradient.color + npoints1) != pb)
				{
					npoints1++;
				}else
					show_warning( "failed to parse color [%s] - defaulting to black", pb );
				*p = ch;
			}
			if (offset_str) {
				for (p = offset_str ; isspace((int)*p) ; p++);
				for (npoints2 = 0 ; *p ; ) {
					char* pb = p, ch;
					if (*p) for ( ; *p && !isspace((int)*p) ; p++);
					ch = *p; *p = '\0';
					gradient.offset[npoints2] = strtod(pb, &pb);
					if (pb == p) npoints2++;
					*p = ch;
					for ( ; isspace((int)*p) ; p++);
				}
			} else {
				for (npoints2 = 0 ; npoints2 < gradient.npoints ; npoints2++)
					gradient.offset[npoints2] = (double)npoints2 / (gradient.npoints - 1);
			}
			if (reverse) {
				for (i = 0 ; i < gradient.npoints / 2 ; i++) {
					int i2 = gradient.npoints - 1 - i;
					ARGB32 c = gradient.color[i];
					double o = gradient.offset[i];
					gradient.color[i] = gradient.color[i2];
					gradient.color[i2] = c;
					gradient.offset[i] = gradient.offset[i2];
					gradient.offset[i2] = o;
				}
				for (i = 0 ; i < gradient.npoints ; i++) {
					gradient.offset[i] = 1.0 - gradient.offset[i];
				}
			}
			if (state->verbose > 1) {
				show_progress("Generating [%dx%d] gradient with angle [%f] and npoints [%d/%d].", width, height, angle, npoints1, npoints2);
				for (i = 0 ; i < gradient.npoints ; i++) {
					show_progress("  Point [%d] has color [#%08x] and offset [%f].", i, (unsigned int)gradient.color[i], gradient.offset[i]);
				}
			}
			result = make_gradient(state->asv, &gradient, width, height, SCL_DO_ALL, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
			if( gradient.color ) 
				free( gradient.color );
			if( gradient.offset )
				free( gradient.offset );
		}
	}
	return result;
}

/****** libAfterImage/asimagexml/solid
 * NAME
 * solid - generate image of specified size and fill it with solid color.
 * SYNOPSIS
 * <solid id="new_id" color="color" opacity="opacity" 
 * 	width="pixels" height="pixels"
 *	refid="refid" width="pixels" height="pixels"/>
 * 
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * width    Optional.  The result will have this width.
 * height   Optional.  The result will have this height.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * color    Optional.  Default is "#ffffffff".  An image will be created
 *          and filled with this color.
 * width    Required.  The image will have this width.
 * height   Required.  The image will have this height.
 * opacity  Optional. Default is 100. Values from 0 to 100 represent the
 *          opacity of resulting image with 100 being completely opaque.
 * 		    Effectively overrides alpha component of the color setting.
 ******/
static ASImage *
handle_asxml_tag_solid( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	Bool opacity_set = False ;
	int opacity = 100 ;
	ARGB32 color = ARGB32_White;
	CARD32 a, r, g, b ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, width = %d, height = %d", doc, parm, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "color")) parse_argb_color(ptr->parm, &color);
		else if (!strcmp(ptr->tag, "opacity")) { opacity = atol(ptr->parm); opacity_set = True ; }
	}
	if( state->verbose > 1 )
		show_progress("Creating solid color [#%08x] image [%dx%d].", (unsigned int)color, width, height);
	result = create_asimage(width, height, ASIMAGE_QUALITY_TOP);
	if( opacity < 0 ) opacity = 0 ;
	else if( opacity > 100 )  opacity = 100 ;
	a = opacity_set? (0x000000FF * (CARD32)opacity)/100: ARGB32_ALPHA8(color);
	r = ARGB32_RED8(color);
	g = ARGB32_GREEN8(color);
	b = ARGB32_BLUE8(color);
	color = MAKE_ARGB32(a,r,g,b);
	if (result) 
		fill_asimage(state->asv, result, 0, 0, width, height, color);

	return result;
}



/****** libAfterImage/asimagexml/save
 * NAME
 * save - write generated/loaded image into the file of one of the
 *        supported types
 * SYNOPSIS
 * <save id="new_id" dst="filename" format="format" compress="value"
 *       opacity="value" replace="0|1" delay="mlsecs">
 * ATTRIBUTES
 * id       Optional.  Image will be given this name for future reference.
 * dst      Optional.  Name of file image will be saved to. If omitted
 *          image will be dumped into stdout - usefull for CGI apps.
 * format   Optional.  Ouput format of saved image.  Defaults to the
 *          extension of the "dst" parameter.  Valid values are the
 *          standard AS image file formats: xpm, jpg, png, gif, tiff.
 * compress Optional.  Compression level if supported by output file
 *          format. Valid values are in range of 0 - 100 and any of
 *          "deflate", "jpeg", "ojpeg", "packbits" for TIFF files.
 *          Note that JPEG and GIF will produce images with deteriorated
 *          quality when compress is greater then 0. For JPEG default is
 *          25, for PNG default is 6 and for GIF it is 0.
 * opacity  Optional. Level below which pixel is considered to be
 *          transparent, while saving image as XPM or GIF. Valid values
 *          are in range 0-255. Default is 127.
 * replace  Optional. Causes ascompose to delete file if the file with the
 *          same name already exists. Valid values are 0 and 1. Default
 *          is 1 - files are deleted before being saved. Disable this to
 *          get multimage animated gifs.
 * delay    Optional. Delay to be stored in GIF image. This could be used
 *          to create animated gifs. Note that you have to set replace="0"
 *          and then write several images into the GIF file with the same
 *          name.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 *******/
static ASImage *
handle_asxml_tag_save( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	const char* dst = NULL;
	const char* ext = NULL;
	const char* compress = NULL ;
	const char* opacity = NULL ;
	int delay = 0 ;
	int replace = 1;
	/*<save id="" dst="" format="" compression="" delay="" replace="" opacity=""> */
	int autoext = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p", doc, parm, imtmp ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "dst")) dst = ptr->parm;
		else if (!strcmp(ptr->tag, "format")) ext = ptr->parm;
		else if (!strncmp(ptr->tag, "compress", 8)) compress = ptr->parm;
		else if (!strcmp(ptr->tag, "opacity")) opacity = ptr->parm;
		else if (!strcmp(ptr->tag, "delay"))   delay = atoi(ptr->parm);
		else if (!strcmp(ptr->tag, "replace")) replace = atoi(ptr->parm);
	}
	if (dst && !ext) {
		ext = strrchr(dst, '.');
		if (ext) ext++;
		autoext = 1;
	}
	
	result = imtmp;
	
	if ( autoext && ext )
		show_warning("No format given.  File extension [%s] used as format.", ext);
	if( state->verbose > 1 )
		show_progress("reSaving image to file [%s].", dst?dst:"stdout");
	if (result && get_flags( state->flags, ASIM_XML_ENABLE_SAVE) )
	{
		if( state->verbose > 1 )
			show_progress("Saving image to file [%s].", dst?dst:"stdout");
		if( !save_asimage_to_file(dst, result, ext, compress, opacity, delay, replace))
			show_error("Unable to save image into file [%s].", dst?dst:"stdout");
	}

	return result;
}

/****** libAfterImage/asimagexml/background
 * NAME
 * background - set image's background color.
 * SYNOPSIS
 *  <background id="new_id" color="color">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * color    Required. Color to be used for background - fills all the
 *          spaces in image with missing pixels.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_background( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	ARGB32 argb = ARGB32_Black;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p", doc, parm, imtmp ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "color")) parse_argb_color( ptr->parm, &argb );
	}
	if (imtmp) {
		result = clone_asimage( imtmp, SCL_DO_ALL );
		result->back_color = argb ;
	}
	if( state->verbose > 1 )
		show_progress( "Setting back_color for image %p to 0x%8.8X", result, argb );
	return result;
}

/****** libAfterImage/asimagexml/blur
 * NAME
 * blur - perform a gaussian blurr on an image.
 * SYNOPSIS
 * <blur id="new_id" horz="radius" vert="radius" channels="argb">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * horz     Optional. Horizontal radius of the blur in pixels.
 * vert     Optional. Vertical radius of the blur in pixels.
 * channels Optional. Applys blur only on listed color channels:
 *                       a - alpha,
 *                       r - red,
 *                       g - green,
 *                       b - blue
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_blur( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	int horz = 0, vert = 0;
    int filter = SCL_DO_ALL;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p", doc, parm, imtmp ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "horz")) horz = atoi(ptr->parm);
        else if (!strcmp(ptr->tag, "vert")) vert = atoi(ptr->parm);
        else if (!strcmp(ptr->tag, "channels"))
        {
            int i = 0 ;
            char *str = &(ptr->parm[0]) ;
            filter = 0 ;
            while( str[i] != '\0' )
            {
                if( str[i] == 'a' )
                    filter |= SCL_DO_ALPHA ;
                else if( str[i] == 'r' )
                    filter |= SCL_DO_RED ;
                else if( str[i] == 'g' )
                    filter |= SCL_DO_GREEN ;
                else if( str[i] == 'b' )
                    filter |= SCL_DO_BLUE ;
                ++i ;
            }
        }
	}
    result = blur_asimage_gauss(state->asv, imtmp, horz, vert, filter, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
	if( state->verbose > 1 )	
		show_progress("Blurrer image with radii %d, %d.", horz, vert);
	return result;
}



/****** libAfterImage/asimagexml/bevel
 * NAME
 * bevel - draws solid bevel frame around the image.
 * SYNOPSIS
 * <bevel id="new_id" colors="color1 color2" 
 * 		  width="pixels" height="pixels" refid="refid"
 *        border="left top right bottom" solid=0|1 outline=0|1>
 * ATTRIBUTES
 * id       Optional.  Image will be given this name for future reference.
 * colors   Optional.  Whitespace-separated list of colors.  Exactly two
 *          colors are required.  Default is "#ffdddddd #ff555555".  The
 *          first color is the color of the upper and left edges, and the
 *          second is the color of the lower and right edges.
 * borders  Optional.  Whitespace-separated list of integer values.
 *          Default is "10 10 10 10".  The values represent the offsets
 *          toward the center of the image of each border: left, top,
 *          right, bottom.
 * solid    Optional - default is 1. If set to 0 will draw bevel gradually
 *          fading into the image.
 * outline  Optional - default is 0. If set to 1 will draw bevel around the 
 * 			image vs. inside the image.
 * width    Optional. The result will have this width.
 * height   Optional. The result will have this height.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_bevel( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	char* color_str = NULL;
	char* border_str = NULL;
	int solid = 1, outline = 0 ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "colors")) color_str = ptr->parm;
		else if (!strcmp(ptr->tag, "border")) border_str = ptr->parm;
		else if (!strcmp(ptr->tag, "solid")) solid = atoi(ptr->parm);
		else if (!strcmp(ptr->tag, "outline")) outline = atoi(ptr->parm);
	}
	if (imtmp) 
	{
		ASImageBevel bevel;
		ASImageLayer layer;
		memset( &bevel, 0x00, sizeof(ASImageBevel) );
		if( solid )
			bevel.type = BEVEL_SOLID_INLINE;
		bevel.hi_color = 0xffdddddd;
		bevel.lo_color = 0xff555555;
		if( outline ) 
			bevel.top_outline = bevel.left_outline = bevel.right_outline = bevel.bottom_outline = 10;
		else
			bevel.top_inline = bevel.left_inline = bevel.right_inline = bevel.bottom_inline = 10;
		if (color_str) {
			char* p = color_str;
			while (isspace((int)*p)) p++;
			p = (char*)parse_argb_color(p, &bevel.hi_color);
			while (isspace((int)*p)) p++;
			parse_argb_color(p, &bevel.lo_color);
		}
		if (border_str) {
			char* p = (char*)border_str;
			if( outline )
			{
				bevel.left_outline = (unsigned short)parse_math(p, &p, width);
				bevel.top_outline = (unsigned short)parse_math(p, &p, height);
				bevel.right_outline = (unsigned short)parse_math(p, &p, width);
				bevel.bottom_outline = (unsigned short)parse_math(p, &p, height);
			}else
			{			  
				bevel.left_inline = (unsigned short)parse_math(p, &p, width);
				bevel.top_inline = (unsigned short)parse_math(p, &p, height);
				bevel.right_inline = (unsigned short)parse_math(p, &p, width);
				bevel.bottom_inline = (unsigned short)parse_math(p, &p, height);
			}
		}
		bevel.hihi_color = bevel.hi_color;
		bevel.hilo_color = bevel.hi_color;
		bevel.lolo_color = bevel.lo_color;
		if( state->verbose > 1 )
			show_progress("Generating bevel with offsets [%d %d %d %d] and colors [#%08x #%08x].", bevel.left_inline, bevel.top_inline, bevel.right_inline, bevel.bottom_inline, (unsigned int)bevel.hi_color, (unsigned int)bevel.lo_color);
		init_image_layers( &layer, 1 );
		layer.im = imtmp;
		if( width <= bevel.left_outline+bevel.right_outline )
			layer.clip_width = 1;
		else
			layer.clip_width = width-(bevel.left_outline+bevel.right_outline);
		if( height <= bevel.top_outline+bevel.bottom_outline )
			layer.clip_height = 1;
		else
			layer.clip_height = height-(bevel.top_outline+bevel.bottom_outline);
		layer.bevel = &bevel;
		result = merge_layers(state->asv, &layer, 1, 
							  width, height, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
	}
	return result;
}

/****** libAfterImage/asimagexml/mirror
 * NAME
 * mirror - create new image as mirror copy of an old one.
 * SYNOPSIS
 *  <mirror id="new_id" dir="direction" 
 * 			width="pixels" height="pixels" refid="refid">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * dir      Required. Possible values are "vertical" and "horizontal".
 *          The image will be flipped over the x-axis if dir is vertical,
 *          and flipped over the y-axis if dir is horizontal.
 * width    Optional.  The result will have this width.
 * height   Optional.  The result will have this height.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_mirror( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	int dir = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "dir")) dir = !mystrcasecmp(ptr->parm, "vertical");
	}
	result = mirror_asimage(state->asv, imtmp, 0, 0, width, height, dir,
							ASA_ASImage, 
							0, ASIMAGE_QUALITY_DEFAULT);
	if( state->verbose > 1 )
		show_progress("Mirroring image [%sally].", dir ? "horizont" : "vertic");
	return result;
}

/****** libAfterImage/asimagexml/rotate
 * NAME
 * rotate - rotate an image in 90 degree increments (flip).
 * SYNOPSIS
 *  <rotate id="new_id" angle="degrees"
 * 			width="pixels" height="pixels" refid="refid">
  * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * angle    Required.  Given in degrees.  Possible values are currently
 *          "90", "180", and "270".  Rotates the image through the given
 *          angle.
 * width    Optional.  The result will have this width.
 * height   Optional.  The result will have this height.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_rotate( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr ;
	double angle = 0;
	int dir = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
		if (!strcmp(ptr->tag, "angle")) angle = strtod(ptr->parm, NULL);
	
	angle = fmod(angle, 2 * PI);
	if (angle > 2 * PI * 7 / 8 || angle < 2 * PI * 1 / 8)
		dir = 0;
	else if (angle < 2 * PI * 3 / 8)
		dir = FLIP_VERTICAL;
	else if (angle < 2 * PI * 5 / 8)
		dir = FLIP_UPSIDEDOWN;
	else 
		dir = FLIP_VERTICAL | FLIP_UPSIDEDOWN;
	if (dir) 
	{
		if( get_flags(dir, FLIP_VERTICAL))
		{
			int tmp = width ;
			width = height ;
			height = tmp ;	
		}	 
		result = flip_asimage(state->asv, imtmp, 0, 0, width, height, dir, ASA_ASImage, 0, ASIMAGE_QUALITY_DEFAULT);
		if( state->verbose > 1 )
			show_progress("Rotating image [%f degrees].", angle);
	} else 
		result = imtmp;

	return result;
}

/****** libAfterImage/asimagexml/scale
 * NAME
 * scale - scale image to arbitrary size
 * SYNOPSIS
 * <scale id="new_id" refid="other_imag" src_x="pixels"  src_y="pixels" 
 *        src_width="pixels" src_height="pixels" 
 *        width="pixels" height="pixels">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Required.  The image will be scaled to this width.
 * height   Required.  The image will be scaled to this height.
 * src_x   Optional. Default is 0. X Offset on infinite surface tiled
 *          with this image, from which to cut portion of an image to be
 *          used in scaling.
 * src_y   Optional. Default is 0. Y Offset on infinite surface tiled
 *          with this image, from which to cut portion of an image to be
 *          used in scaling.
 * src_width
 *          Optional. Default is image width. Tile image to this width
 *          prior to scaling.
 * src_height
 *          Optional. Default is image height. Tile image to this height
 *          prior to scaling.
  * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 * If you want to keep image proportions while scaling - use "proportional"
 * instead of specific size for particular dimention.
 ******/
static ASImage *
handle_asxml_tag_scale( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int src_x = 0, src_y = 0 ;
	int src_width = 0, src_height = 0 ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "src_x")) 		src_x = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "src_y")) 	src_y = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "src_width")) 	src_width = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "src_height")) 	src_height = (int)parse_math(ptr->parm, NULL, width);
	}	
	if( state->verbose > 1 )
		show_progress("Scaling image to [%dx%d].", width, height);
	result = scale_asimage2( state->asv, imtmp, 
							src_x, src_y, src_width, src_height,
							width, height, 
							ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT);
	return result;
}
/****** libAfterImage/asimagexml/slice
 * NAME
 * slice - slice image to arbitrary size leaving corners unchanged
 * SYNOPSIS
 * <slice id="new_id" ref_id="other_imag" width="pixels" height="pixels"
 *        x_start="slice_x_start" x_end="slice_x_end"
 * 		  y_start="slice_y_start" y_end="slice_y_end"
 * 		  scale="0|1">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Required.  The image will be scaled to this width.
 * height   Required.  The image will be scaled to this height.
 * x_start  Optional. Position at which vertical image slicing begins. 
 * 			Corresponds to the right side of the left corners.
 * x_end    Optional. Position at which vertical image slicing end.
 * 			Corresponds to the left side of the right corners.
 * y_start  Optional. Position at which horisontal image slicing begins. 
 *          Corresponds to the bottom side of the top corners.
 * y_end    Optional. Position at which horisontal image slicing end.
 * 			Corresponds to the top side of the bottom corners.
 * scale    Optional. If set to 1 will cause middle portion of the 
 * 			image to be scaled instead of tiled.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 * Contents of the image between x_start and x_end will be tiled 
 * horizontally. Contents of the image between y_start and y_end will be 
 * tiled vertically. This is usefull to get background images to fit the 
 * size of the text or a widget, while preserving its borders undistorted, 
 * which is the usuall result of simple scaling.
 * If you want to keep image proportions while resizing-use "proportional"
 * instead of specific size for particular dimention.
 ******/
static ASImage *
handle_asxml_tag_slice( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int x_start = 0, x_end = 0 ;
	int y_start = 0, y_end = 0 ;
	Bool scale = False ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "x_start")) 		x_start = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "x_end")) 	x_end = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "y_start")) 	y_start = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "y_end")) 	y_end = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "scale")) 	scale = (ptr->parm[0] == '1');
	}

	if( state->verbose > 1 )
		show_progress("Slicing image to [%dx%d].", width, height);
	result = slice_asimage2( state->asv, imtmp, x_start, x_end, y_start, y_end, width, height, 
							 scale, ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT);
	return result;
}

/****** libAfterImage/asimagexml/pixelize
 * NAME
 * pixelize - pixelize image using arbitrary pixel size
 * SYNOPSIS
 * <pixelize id="new_id" ref_id="other_imag" width="pixels" height="pixels"
 *        clip_x="clip_x" clip_y="clip_y"
 *        pixel_width="pixel_width" pixel_height="pixel_height">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Required.  The image will be scaled to this width.
 * height   Required.  The image will be scaled to this height.
 * clip_x   Optional. Offset into original image.
 * clip_y   Optional. Offset into original image.
 * pixel_width Required. Horizontal pixelization step;
 * pixel_height Required. Vertical pixelization step;
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 * If you want to keep image proportions while resizing-use "proportional"
 * instead of specific size for particular dimention.
 ******/
static ASImage *
handle_asxml_tag_pixelize( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int clip_x = 0, clip_y = 0 ;
	int pixel_width = 1, pixel_height = 1 ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "clip_x")) 		clip_x = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "clip_y")) 	clip_y = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "pixel_width")) 		pixel_width = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "pixel_height")) 	pixel_height = (int)parse_math(ptr->parm, NULL, height);
	}

	if( state->verbose > 1 )
		show_progress("Pixelizing image to [%dx%d] using pixel size %dx%d.", 
						width, height, pixel_width, pixel_height);
	result = pixelize_asimage (state->asv, imtmp, clip_x, clip_y, width, height,
							   pixel_width, pixel_height,
							   ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT);
	return result;
}

/****** libAfterImage/asimagexml/color2alpha
 * NAME
 * color2alpha - set alpha channel based on color closeness to specified color
 * SYNOPSIS
 * <color2alpha id="new_id" ref_id="other_imag" width="pixels" height="pixels"
 *        clip_x="clip_x" clip_y="clip_y"
 *        color="color">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Required.  The image will be scaled to this width.
 * height   Required.  The image will be scaled to this height.
 * clip_x   Optional. Offset into original image.
 * clip_y   Optional. Offset into original image.
 * color    Required. Color to match against.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 * If you want to keep image proportions while resizing-use "proportional"
 * instead of specific size for particular dimention.
 ******/
static ASImage *
handle_asxml_tag_color2alpha( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int clip_x = 0, clip_y = 0 ;
	ARGB32 color;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "clip_x")) 		clip_x = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "clip_y")) 	clip_y = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "color")) 	parse_argb_color(ptr->parm, &color);
	}

	if( state->verbose > 1 )
		show_progress("color2alpha image to [%dx%d] using color #%8.8X.", width, height, color);
	result = color2alpha_asimage (state->asv, imtmp, clip_x, clip_y, width, height,
							   		color,
							   		ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT);
	return result;
}

/****** libAfterImage/asimagexml/crop
 * NAME
 * crop - crop image to arbitrary area within it.
 * SYNOPSIS
 *  <crop id="new_id" refid="other_image" srcx="pixels" srcy="pixels"
 *        width="pixels" height="pixels" tint="color">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional. An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * srcx     Optional. Default is "0". Skip this many pixels from the left.
 * srcy     Optional. Default is "0". Skip this many pixels from the top.
 * width    Optional. Default is "100%".  Keep this many pixels wide.
 * height   Optional. Default is "100%".  Keep this many pixels tall.
 * tint     Optional. Additionally tint an image to specified color.
 *          Tinting can both lighten and darken an image. Tinting color
 *          0 or #7f7f7f7f yeilds no tinting. Tinting can be performed on
 *          any channel, including alpha channel.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_crop( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	ARGB32 tint = 0 ;
	int srcx = 0, srcy = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "srcx")) srcx = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "srcy")) srcy = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "tint")) parse_argb_color(ptr->parm, &tint);
	}
	if( state->verbose > 1 )
		show_progress("Cropping image to [%dx%d].", width, height);
	result = tile_asimage(state->asv, imtmp, srcx, srcy, width, height, tint, ASA_ASImage, 100, ASIMAGE_QUALITY_TOP);
	return result;
}

/****** libAfterImage/asimagexml/tile
 * NAME
 * tile - tile an image to specified area.
 * SYNOPSIS
 *  <tile id="new_id" refid="other_image" width="pixels" height="pixels"
 *        x_origin="pixels" y_origin="pixels" tint="color" complement=0|1>
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional. An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Optional. Default is "100%". The image will be tiled to this
 *          width.
 * height   Optional. Default is "100%". The image will be tiled to this
 *          height.
 * x_origin Optional. Horizontal position on infinite surface, covered
 *          with tiles of the image, from which to cut out resulting
 *          image.
 * y_origin Optional. Vertical position on infinite surface, covered
 *          with tiles of the image, from which to cut out resulting
 *          image.
 * tint     Optional. Additionally tint an image to specified color.
 *          Tinting can both lighten and darken an image. Tinting color
 *          0 or #7f7f7f7f yields no tinting. Tinting can be performed
 *          on any channel, including alpha channel.
 * complement Optional. Will use color that is the complement to tint color
 *          for the tinting, if set to 1. Default is 0.
 *
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_tile( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int xorig = 0, yorig = 0;
	ARGB32 tint = 0 ;
	char *complement_str = NULL ;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "x_origin")) xorig = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "y_origin")) yorig = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "tint")) parse_argb_color(ptr->parm, &tint);
		else if (!strcmp(ptr->tag, "complement")) complement_str = ptr->parm;
	}
	if( complement_str )
	{
		register char *ptr = complement_str ;
		CARD32 a = ARGB32_ALPHA8(tint),
				r = ARGB32_RED8(tint),
				g = ARGB32_GREEN8(tint),
				b = ARGB32_BLUE8(tint) ;
		while( *ptr )
		{
			if( *ptr == 'a' ) 		a = ~a ;
			else if( *ptr == 'r' ) 	r = ~r ;
			else if( *ptr == 'g' ) 	g = ~g ;
			else if( *ptr == 'b' ) 	b = ~b ;
			++ptr ;
		}

		tint = MAKE_ARGB32(a, r, g, b );
	}
	if( state->verbose > 1 )
		show_progress("Tiling image to [%dx%d].", width, height);
	result = tile_asimage(state->asv, imtmp, xorig, yorig, width, height, tint, ASA_ASImage, 100, ASIMAGE_QUALITY_TOP);
	return result;
}

/****** libAfterImage/asimagexml/hsv
 * NAME
 * hsv - adjust Hue, Saturation and/or Value of an image and optionally
 * tile an image to arbitrary area.
 * SYNOPSIS
 * <hsv id="new_id" refid="other_image"
 *      x_origin="pixels" y_origin="pixels" width="pixels" height="pixels"
 *      affected_hue="degrees|color" affected_radius="degrees"
 *      hue_offset="degrees" saturation_offset="value"
 *      value_offset="value">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * refid    Optional. An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * width    Optional. Default is "100%". The image will be tiled to this
 *          width.
 * height   Optional. Default is "100%". The image will be tiled to this
 *          height.
 * x_origin Optional. Horizontal position on infinite surface, covered
 *          with tiles of the image, from which to cut out resulting
 *          image.
 * y_origin Optional. Vertical position on infinite surface, covered
 *          with tiles of the image, from which to cut out resulting
 *          image.
 * affected_hue    Optional. Limits effects to the renage of hues around
 *          this hue. If numeric value is specified - it is treated as
 *          degrees on 360 degree circle, with :
 *              red = 0,
 *              yellow = 60,
 *              green = 120,
 *              cyan = 180,
 *              blue = 240,
 *              magenta = 300.
 *          If colorname or value preceded with # is specified here - it
 *          will be treated as RGB color and converted into hue
 *          automagically.
 * affected_radius
 *          Optional. Value in degrees to be used in order to
 *          calculate the range of affected hues. Range is determined by
 *          substracting and adding this value from/to affected_hue.
 * hue_offset
 *          Optional. Value by which to adjust the hue.
 * saturation_offset
 *          Optional. Value by which to adjust the saturation.
 * value_offset
 *          Optional. Value by which to adjust the value.
 * NOTES
 * One of the Offsets must be not 0, in order for operation to be
 * performed.
 *
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_hsv( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	int affected_hue = 0, affected_radius = 360 ;
	int hue_offset = 0, saturation_offset = 0, value_offset = 0 ;
	int xorig = 0, yorig = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) 
	{
		if (!strcmp(ptr->tag, "x_origin")) xorig = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "y_origin")) yorig = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "affected_hue"))
		{
			if( isdigit( (int)ptr->parm[0] ) )
				affected_hue = (int)parse_math(ptr->parm, NULL, 360);
			else
			{
				ARGB32 color = 0;
				if( parse_argb_color( ptr->parm, &color ) != ptr->parm )
				{
					affected_hue = rgb2hue( ARGB32_RED16(color),
											ARGB32_GREEN16(color),
											ARGB32_BLUE16(color));
  					affected_hue = hue162degrees( affected_hue );
				}
			}
		}
		else if (!strcmp(ptr->tag, "affected_radius")) 	affected_radius = (int)parse_math(ptr->parm, NULL, 360);
		else if (!strcmp(ptr->tag, "hue_offset")) 		hue_offset = (int)parse_math(ptr->parm, NULL, 360);
		else if (!strcmp(ptr->tag, "saturation_offset"))saturation_offset = (int)parse_math(ptr->parm, NULL, 100);
		else if (!strcmp(ptr->tag, "value_offset")) 	value_offset = (int)parse_math(ptr->parm, NULL, 100);
	}
	if( hue_offset == -1 && saturation_offset == -1 ) 
	{
		hue_offset = 0 ; 
		saturation_offset = -99 ;
	}
	if (hue_offset!=0 || saturation_offset != 0 || value_offset != 0 ) 
	{
		result = adjust_asimage_hsv(state->asv, imtmp, xorig, yorig, width, height,
				                    affected_hue, affected_radius,
									hue_offset, saturation_offset, value_offset,
				                    ASA_ASImage, 100, ASIMAGE_QUALITY_TOP);
	}
	if( state->verbose > 1 )
		show_progress("adjusted HSV of the image by [%d,%d,%d] affected hues are %+d-%+d.result = %p", hue_offset, saturation_offset, value_offset, affected_hue-affected_radius, affected_hue+affected_radius, result);
	return result;
}

/****** libAfterImage/asimagexml/pad
 * NAME
 * pad - pad an image with solid color rectangles.
 * SYNOPSIS
 * <pad id="new_id" left="pixels" top="pixels"
 *      right="pixels" bottom="pixels" color="color"
 * 		refid="refid" width="pixels" height="pixels">
 * ATTRIBUTES
 * id       Optional. Image will be given this name for future reference.
 * width    Optional.  The result will have this width.
 * height   Optional.  The result will have this height.
 * refid    Optional.  An image ID defined with the "id" parameter for
 *          any previously created image.  If set, percentages in "width"
 *          and "height" will be derived from the width and height of the
 *          refid image.
 * left     Optional. Size to add to the left of the image.
 * top      Optional. Size to add to the top of the image.
 * right    Optional. Size to add to the right of the image.
 * bottom   Optional. Size to add to the bottom of the image.
 * color    Optional. Color value to fill added areas with. It could be
 *          transparent of course. Default is #FF000000 - totally black.
 * NOTES
 * This tag applies to the first image contained within the tag.  Any
 * further images will be discarded.
 ******/
static ASImage *
handle_asxml_tag_pad( ASImageXMLState *state, xml_elem_t* doc, xml_elem_t* parm, ASImage *imtmp, int width, int height)
{
	ASImage *result = NULL ;
	xml_elem_t* ptr;
	ARGB32 color  = ARGB32_Black;
	int left = 0, top = 0, right = 0, bottom = 0;
	LOCAL_DEBUG_OUT("doc = %p, parm = %p, imtmp = %p, width = %d, height = %d", doc, parm, imtmp, width, height ); 
	for (ptr = parm ; ptr ; ptr = ptr->next) {
		if (!strcmp(ptr->tag, "left"))   left = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "top"))    top = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "right"))  right = (int)parse_math(ptr->parm, NULL, width);
		else if (!strcmp(ptr->tag, "bottom")) bottom = (int)parse_math(ptr->parm, NULL, height);
		else if (!strcmp(ptr->tag, "color"))  parse_argb_color(ptr->parm, &color);
	}
	if( state->verbose > 1 )
		show_progress("Padding image to [%dx%d%+d%+d].", width+left+right, height+top+bottom, left, top);
	if (left > 0 || top > 0 || right > 0 || bottom > 0 )
		result = pad_asimage(state->asv, imtmp, left, top, width+left+right, height+top+bottom,
					            color, ASA_ASImage, 100, ASIMAGE_QUALITY_DEFAULT);
	return result;
}

#define REPLACE_STRING(str,val) do {if(str)free(str);(str) = (val);}while(0)

/* Each tag is only allowed to return ONE image. */
ASImage*
build_image_from_xml( ASVisual *asv, ASImageManager *imman, ASFontManager *fontman, xml_elem_t* doc, xml_elem_t** rparm, ASFlagType flags, int verbose, Window display_win)
{
	xml_elem_t* ptr;
	char* id = NULL;
	ASImage* result = NULL;
	ASImageXMLState state ; 

	if( IsCDATA(doc) )  return NULL ;

	memset( &state, 0x00, sizeof(state));
	state.flags = flags ;
	state.asv = asv ; 
	state.imman = imman ; 
	state.fontman = fontman ; 
	state.verbose = verbose ;
	state.display_win = display_win ;

	if( doc ) 
	{		 
		xml_elem_t* parm = xml_parse_parm(doc->parm, NULL);
		xml_elem_t* ptr ;
		char* refid = NULL;
		char* width_str = NULL;
		char* height_str = NULL;
		ASImage *refimg = NULL ; 
		int width = 0, height = 0 ;
		LOCAL_DEBUG_OUT("parm = %p", parm);

		for (ptr = parm ; ptr ; ptr = ptr->next)
		{	
			if (ptr->tag[0] == 'i' && ptr->tag[1] == 'd' && ptr->tag[2] == '\0')
				REPLACE_STRING(id,mystrdup(ptr->parm));
			else if (strcmp(ptr->tag, "refid") == 0 ) 	refid = ptr->parm ;
			else if (strcmp(ptr->tag, "width") == 0 ) 	width_str = ptr->parm ;
			else if (strcmp(ptr->tag, "height") == 0 ) 	height_str = ptr->parm ;
		}		

		if( id ) 
			if( (result = fetch_asimage( imman, id)) != NULL ) 
			{
				free( id );
				xml_elem_delete(NULL, parm);
				return result ; 
			}

		if( refid ) 
			refimg = fetch_asimage( imman, refid);

		if (!strcmp(doc->tag, "composite")) 
			result = handle_asxml_tag_composite( &state, doc, parm );  	
		else if (!strcmp(doc->tag, "text")) 
			result = handle_asxml_tag_text( &state, doc, parm );  	
		else if (!strcmp(doc->tag, "img")) 
		{
			translate_tag_size(	width_str, height_str, NULL, refimg, &width, &height );  
			result = handle_asxml_tag_img( &state, doc, parm, width, height );
		}else if (!strcmp(doc->tag, "recall")) 
			result = handle_asxml_tag_recall( &state, doc, parm );
		else if (!strcmp(doc->tag, "release"))
			result = handle_asxml_tag_release( &state, doc, parm );
		else if (!strcmp(doc->tag, "color"))
			result = handle_asxml_tag_color( &state, doc, parm ); 
		else if (!strcmp(doc->tag, "printf"))
			result = handle_asxml_tag_printf( &state, doc, parm ); 
		else if (!strcmp(doc->tag, "set"))
			result = handle_asxml_tag_set( &state, doc, parm ); 
		else if (!strcmp(doc->tag, "if") || !strcmp(doc->tag, "unless") )
			result = handle_asxml_tag_if( &state, doc, parm ); 
		else if ( !strcmp(doc->tag, "gradient") )
		{	
			translate_tag_size(	width_str, height_str, NULL, refimg, &width, &height );  
			if( width > 0 && height > 0 )
				result = handle_asxml_tag_gradient( &state, doc, parm, width, height ); 	   
		}else if (!strcmp(doc->tag, "solid"))
		{	
			translate_tag_size(	width_str, height_str, NULL, refimg, &width, &height );  
			if( width > 0 && height > 0 )
				result = handle_asxml_tag_solid( &state, doc, parm, width, height);
		}else
		{	
			ASImage *imtmp = NULL ; 

			for (ptr = doc->child ; ptr && !imtmp ; ptr = ptr->next) 
				imtmp = build_image_from_xml(asv, imman, fontman, ptr, NULL, flags, verbose, display_win);

			if( imtmp ) 
			{	
				if (imtmp && !strcmp(doc->tag, "save")) 
					result = handle_asxml_tag_save( &state, doc, parm, imtmp ); 	
				else if (imtmp && !strcmp(doc->tag, "background")) 
					result = handle_asxml_tag_background( &state, doc, parm, imtmp ); 	
				else if (imtmp && !strcmp(doc->tag, "blur")) 
					result = handle_asxml_tag_blur( &state, doc, parm, imtmp ); 	
				else 
				{	
					translate_tag_size(	width_str, height_str, imtmp, refimg, &width, &height );   
		
					if ( width > 0 && height > 0 )
					{ 
#define HANDLE_SIZED_TAG(ttag) \
		else if( !strcmp(doc->tag, #ttag) )	result = handle_asxml_tag_##ttag( &state, doc, parm, imtmp, width, height )
						if (0){}
						HANDLE_SIZED_TAG(bevel);
						HANDLE_SIZED_TAG(mirror);
						HANDLE_SIZED_TAG(rotate);
						HANDLE_SIZED_TAG(scale);
						HANDLE_SIZED_TAG(slice);
						HANDLE_SIZED_TAG(crop);
						HANDLE_SIZED_TAG(tile);
						HANDLE_SIZED_TAG(hsv);
						HANDLE_SIZED_TAG(pad);
						HANDLE_SIZED_TAG(pixelize);
						HANDLE_SIZED_TAG(color2alpha);
#undef HANDLE_SIZED_TAG						
					}		
				}
				
				if( result != imtmp ) 
					safe_asimage_destroy(imtmp);
			}
		}
		
		if( refimg ) 
			release_asimage( refimg );
		
		if (rparm) *rparm = parm; 
		else xml_elem_delete(NULL, parm);
	}
	LOCAL_DEBUG_OUT("result = %p, id = \"%s\"", result, id?id:"(null)" );



	/* No match so far... see if one of our children can do any better.*/
	if (!result  && doc ) 
	{
		xml_elem_t* tparm = NULL;
		for (ptr = doc->child ; ptr && !result ; ptr = ptr->next) 
		{
			xml_elem_t* sparm = NULL;
			result = build_image_from_xml(asv, imman, fontman, ptr, &sparm, flags, verbose, display_win);
			if (result) 
			{
				if (tparm) xml_elem_delete(NULL, tparm);
				tparm = sparm; 
			}else 
				if (sparm) xml_elem_delete(NULL, sparm);

		}
		if (rparm) 
		{ 
			if( *rparm ) xml_elem_delete(NULL, *rparm); *rparm = tparm; 
		}else 
			xml_elem_delete(NULL, tparm);
	}

	LOCAL_DEBUG_OUT("result = %p", result );
	result = commit_xml_image_built( &state, id, result );
	if( id ) 
		free( id );
	LOCAL_DEBUG_OUT("result = %p", result );
	if( result )
	{
		LOCAL_DEBUG_OUT("result's size = %dx%d", result->width, result->height );	
	}	 
	return result;
}




