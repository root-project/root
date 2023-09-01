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

#undef LOCAL_DEBUG
#define DO_CLOCKING 

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
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif

#ifdef HAVE_GLX
# include <GL/gl.h>
# include <GL/glx.h>
/*# include <GL/glu.h> */
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
#include "ximage.h"


/* ***************************************************************************/
/* ASImage->XImage->pixmap->XImage->ASImage conversion						*/
/* ***************************************************************************/

ASImage      *
picture_ximage2asimage (ASVisual *asv, XImage *xim, XImage *alpha_xim, unsigned int compression)
{
	ASImage      *im = NULL;
	unsigned char *xim_line;
	int           i, height, width, bpl;
	ASScanline    xim_buf;
#ifdef LOCAL_DEBUG
	CARD32       *tmp ;
#endif
#if 0
  struct timeval stv;
	gettimeofday (&stv,NULL);
#define PRINT_BACKGROUND_OP_TIME do{ struct timeval tv;gettimeofday (&tv,NULL); tv.tv_sec-= stv.tv_sec;\
                                     fprintf (stderr,__FILE__ "%d: elapsed  %ld usec\n",__LINE__,\
                                              tv.tv_sec*1000000+tv.tv_usec-stv.tv_usec );}while(0)
#else                                           
#define PRINT_BACKGROUND_OP_TIME do{}while(0)                                          
#endif

	if( xim && alpha_xim )
		if( xim->width != alpha_xim->width ||
		    xim->height != alpha_xim->height )
			return NULL ;
	if( xim == NULL && alpha_xim == NULL )
		return NULL ;

	width = xim?xim->width:alpha_xim->width;
	height = xim?xim->height:alpha_xim->height;

	im = create_asimage( width, height, compression);
	prepare_scanline( width, 0, &xim_buf, asv->BGR_mode );
#ifdef LOCAL_DEBUG
	tmp = safemalloc( width * sizeof(CARD32));
#endif
PRINT_BACKGROUND_OP_TIME;
	if( xim )
	{
		bpl 	 = xim->bytes_per_line;
		xim_line = (unsigned char *)xim->data;

		for (i = 0; i < height; i++)
		{
#if 0
/* unfortunately this method does not seem to yield any better results performancewise: */
			if (asv->true_depth == 32 || asv->true_depth == 24)
			{
				if( asv->msb_first ) 
				{
				}else
					asimage_add_line_bgra (im, xim_line, i);
			}else 
#endif			
			if( xim->depth == (int)asv->true_depth )
			{
			    GET_SCANLINE(asv,xim,&xim_buf,i,xim_line);
    		    asimage_add_line (im, IC_RED,   xim_buf.red, i);
			    asimage_add_line (im, IC_GREEN, xim_buf.green, i);
			    asimage_add_line (im, IC_BLUE,  xim_buf.blue, i);
				if( xim->depth == 32 && alpha_xim == NULL ) 
				    asimage_add_line (im, IC_ALPHA,  xim_buf.alpha, i);
#ifdef LOCAL_DEBUG
			    if( !asimage_compare_line( im, IC_RED,  xim_buf.red, tmp, i, True ) )
				exit(0);
			    if( !asimage_compare_line( im, IC_GREEN,  xim_buf.green, tmp, i, True ) )
				exit(0);
			    if( !asimage_compare_line( im, IC_BLUE,  xim_buf.blue, tmp, i, True ) )
				exit(0);
#endif
			}else if( xim->depth == 8 )
			{
			    register int x = width;
			    while(--x >= 0 )
	    		        xim_buf.blue[x] = (CARD32)(xim_line[x]);
	    		    asimage_add_line (im, IC_RED,   xim_buf.red, i);
			    asimage_add_line (im, IC_GREEN, xim_buf.red, i);
			    asimage_add_line (im, IC_BLUE,  xim_buf.red, i);
			}else if( xim->depth == 1 )
			{
			    register int x = width;
			    while(--x >= 0 )
			    {
#ifndef X_DISPLAY_MISSING
				xim_buf.red[x] = (XGetPixel(xim, x, i) == 0)?0x00:0xFF;
#else
				xim_buf.red[x] = 0xFF ;
#endif
			    }
	    		    asimage_add_line (im, IC_RED,   xim_buf.red, i);
			    asimage_add_line (im, IC_GREEN, xim_buf.red, i);
			    asimage_add_line (im, IC_BLUE,  xim_buf.red, i);
			}

			xim_line += bpl;
		}
	}
PRINT_BACKGROUND_OP_TIME;
	if( alpha_xim )
	{
		CARD32 *dst = xim_buf.alpha ;
		bpl 	 = alpha_xim->bytes_per_line;
		xim_line = (unsigned char *)alpha_xim->data;

		for (i = 0; i < height; i++)
		{
			register int x = width;
			if( alpha_xim->depth == 8 )
			{
				while(--x >= 0 ) dst[x] = (CARD32)(xim_line[x]);
			}else
			{
				while(--x >= 0 )
#ifndef X_DISPLAY_MISSING
					dst[x] = (XGetPixel(alpha_xim, x, i) == 0)?0x00:0xFF;
#else
					dst[x] = 0xFF ;
#endif
			}
			asimage_add_line (im, IC_ALPHA, xim_buf.alpha, i);
#ifdef LOCAL_DEBUG
			if( !asimage_compare_line( im, IC_ALPHA,  xim_buf.alpha, tmp, i, True ) )
				exit(0);
#endif
			xim_line += bpl;
		}
	}
	free_scanline(&xim_buf, True);
PRINT_BACKGROUND_OP_TIME;

	return im;
}

ASImage      *
ximage2asimage (ASVisual *asv, XImage * xim, unsigned int compression)
{
	return picture_ximage2asimage (asv, xim, NULL, compression);
}

static inline int
xim_set_component( register CARD32 *src, register CARD32 value, int offset, int len )
{
	register int i ;
	for( i = offset ; i < len ; ++i )
		src[i] = value;
	return len-offset;
}

Bool
subimage2ximage (ASVisual *asv, ASImage *im, int x, int y, XImage* xim)
{
	int            i, max_i;
	ASScanline     xim_buf;
	ASImageOutput *imout ;
/*#ifdef DO_CLOCKING
	clock_t       started = clock ();
#endif*/
	int width, height ;
	ASImage *scratch_im ;

	if (im == NULL)
	{
LOCAL_DEBUG_OUT( "Attempt to convert NULL ASImage into XImage.", "" );
		return False;
	}
	if( x >= (int)im->width || y >= (int)im->height ) 
		return False;
	width = xim->width ;
	if( width > (int)im->width - x) 
		width = (int)im->width - x;		   
	width = ( x > (int)im->width - width )?im->width - width:im->width - x ;
	height = xim->height ;
	if( height > (int)im->height - y ) 
		height = im->height - y ;		
	scratch_im = create_asimage( width, height, 0);
	scratch_im->alt.ximage = xim ;
LOCAL_DEBUG_OUT( "target width = %d, height = %d", width, height );
	if( (imout = start_image_output( asv, scratch_im, ASA_ScratchXImage, 0, ASIMAGE_QUALITY_DEFAULT )) == NULL )
	{
LOCAL_DEBUG_OUT( "Failed to start ASImageOutput for ASImage %p and ASVisual %p", im, asv );
		return False;
	}

	prepare_scanline( width, 0, &xim_buf, asv->BGR_mode );
/*#ifdef DO_CLOCKING
	started = clock ();
#endif*/
	set_flags( xim_buf.flags, SCL_DO_ALL );
	max_i = y + height ;
	for (i = y; i < max_i; i++)
	{
		int count ;
		if( (count = asimage_decode_line (im, IC_RED,   xim_buf.red, i, x, xim_buf.width)) < (int)xim_buf.width )
			xim_set_component( xim_buf.red, ARGB32_RED8(im->back_color), count, xim_buf.width );
		if( (count = asimage_decode_line (im, IC_GREEN, xim_buf.green, i, x, xim_buf.width))< (int)xim_buf.width )
			xim_set_component( xim_buf.green, ARGB32_GREEN8(im->back_color), count, xim_buf.width );
		if( (count = asimage_decode_line (im, IC_BLUE,  xim_buf.blue, i, x, xim_buf.width)) < (int)xim_buf.width )
			xim_set_component( xim_buf.blue, ARGB32_BLUE8(im->back_color), count, xim_buf.width );
		if( xim->depth == 32 ) 
			if( (count = asimage_decode_line (im, IC_ALPHA,  xim_buf.alpha, i, x, xim_buf.width)) < (int)xim_buf.width )
				xim_set_component( xim_buf.alpha, ARGB32_ALPHA8(im->back_color), count, xim_buf.width );
			
		imout->output_image_scanline( imout, &xim_buf, 1 );
/*		LOCAL_DEBUG_OUT( "line %d, count = %d", i, count ); */
	}
/*#ifdef DO_CLOCKING
	fprintf (stderr, "asimage->ximage time (clocks): %lu\n", clock () - started);
#endif*/
	free_scanline(&xim_buf, True);
	stop_image_output(&imout);

	scratch_im->alt.ximage = NULL ;
	destroy_asimage( &scratch_im );
	return True;
}



static XImage*
asimage2ximage_ext (ASVisual *asv, ASImage *im, Bool scratch)
{
	XImage        *xim = NULL;
	int            i;
	ASImageOutput *imout ;
	ASImageDecoder *imdec;
/*#ifdef DO_CLOCKING
	clock_t       started = clock ();
#endif*/

	if (im == NULL)
	{
LOCAL_DEBUG_OUT( "Attempt to convert NULL ASImage into XImage.", "" );
		return xim;
	}
	if( (imout = start_image_output( asv, im, scratch?ASA_ScratchXImage:ASA_XImage, 0, ASIMAGE_QUALITY_DEFAULT )) == NULL )
	{
LOCAL_DEBUG_OUT( "Failed to start ASImageOutput for ASImage %p and ASVisual %p", im, asv );
		return xim;
	}
	xim = im->alt.ximage ;
	/* no data in ximage yet */
	set_flags( im->flags, ASIM_XIMAGE_NOT_USEFUL);
/*#ifdef DO_CLOCKING
	started = clock ();
#endif*/
#if	1
	if ((imdec = start_image_decoding(  asv, im, (xim->depth >= 24)?SCL_DO_ALL:SCL_DO_COLOR, 
										0, 0, im->width, im->height, NULL)) != NULL )
	{	 
		for (i = 0; i < (int)im->height; i++)
		{	
			imdec->decode_image_scanline( imdec ); 
			imout->output_image_scanline( imout, &(imdec->buffer), 1);
		}
		stop_image_decoding( &imdec );
	}
#else	  
	{	
		ASScanline     xim_buf;
		prepare_scanline( im->width, 0, &xim_buf, asv->BGR_mode );
		set_flags( xim_buf.flags, SCL_DO_ALL );
		for (i = 0; i < (int)im->height; i++)
		{
			int count ;
			if( (count = asimage_decode_line (im, IC_RED,   xim_buf.red, i, 0, xim_buf.width)) < (int)xim_buf.width )
				xim_set_component( xim_buf.red, ARGB32_RED8(im->back_color), count, xim_buf.width );
			if( (count = asimage_decode_line (im, IC_GREEN, xim_buf.green, i, 0, xim_buf.width))< (int)xim_buf.width )
				xim_set_component( xim_buf.green, ARGB32_GREEN8(im->back_color), count, xim_buf.width );
			if( (count = asimage_decode_line (im, IC_BLUE,  xim_buf.blue, i, 0, xim_buf.width)) < (int)xim_buf.width )
				xim_set_component( xim_buf.blue, ARGB32_BLUE8(im->back_color), count, xim_buf.width );
			if( xim->depth == 32 ) 
				if( (count = asimage_decode_line (im, IC_ALPHA,  xim_buf.alpha, i, 0, xim_buf.width)) < (int)xim_buf.width )
					xim_set_component( xim_buf.alpha, ARGB32_ALPHA8(im->back_color), count, xim_buf.width );
			imout->output_image_scanline( imout, &xim_buf, 1 );
		}
		free_scanline(&xim_buf, True);
	}
#endif

/*#ifdef DO_CLOCKING
	fprintf (stderr, "asimage->ximage time (clocks): %lu\n", clock () - started);
#endif*/

	stop_image_output(&imout);
	clear_flags( im->flags, ASIM_XIMAGE_NOT_USEFUL);

	return xim;
}

XImage*
asimage2ximage (ASVisual *asv, ASImage *im)
{
	return asimage2ximage_ext (asv, im, False);	
}

XImage*
asimage2alpha_ximage (ASVisual *asv, ASImage *im, Bool bitmap )
{
	XImage        *xim = NULL;
	int            i;
	ASScanline     xim_buf;
	ASImageOutput *imout ;
	ASFlagType flag = bitmap?0:ASIM_XIMAGE_8BIT_MASK;

	if (im == NULL)
		return xim;

	if( im->alt.mask_ximage )
		if( (im->flags & ASIM_XIMAGE_8BIT_MASK )^flag)
		{
#ifndef X_DISPLAY_MISSING
			XDestroyImage( im->alt.mask_ximage );
#endif
			im->alt.mask_ximage = NULL ;
		}
    clear_flags( im->flags, ASIM_XIMAGE_8BIT_MASK );
	set_flags( im->flags, flag );

	if( (imout = start_image_output( asv, im, ASA_MaskXImage, 0, ASIMAGE_QUALITY_POOR )) == NULL )
		return xim;
	xim = im->alt.mask_ximage ;
	prepare_scanline( xim->width, 0, &xim_buf, asv->BGR_mode );
	xim_buf.flags = SCL_DO_ALPHA ;
	for (i = 0; i < (int)im->height; i++)
	{
		int count = asimage_decode_line (im, IC_ALPHA, xim_buf.alpha, i, 0, xim_buf.width);
		if( count < (int)xim_buf.width )
			xim_set_component( xim_buf.alpha, ARGB32_ALPHA8(im->back_color), count, xim_buf.width );
		imout->output_image_scanline( imout, &xim_buf, 1 );
	}
	free_scanline(&xim_buf, True);

	stop_image_output(&imout);

	return xim;
}

XImage*
asimage2mask_ximage (ASVisual *asv, ASImage *im)
{
	return asimage2alpha_ximage (asv, im, True );
}

ASImage      *
pixmap2ximage(ASVisual *asv, Pixmap p, int x, int y, unsigned int width, unsigned int height, unsigned long plane_mask, unsigned int compression)
{
#ifndef X_DISPLAY_MISSING
	XImage       *xim = ASGetXImage (asv, p, x, y, width, height, plane_mask);
	ASImage      *im = NULL;

	if (xim)
	{
		im = create_asimage( xim->width, xim->height, compression);
		im->flags = ASIM_DATA_NOT_USEFUL ;
		im->alt.ximage = xim ;
	}
	return im;
#else
    return NULL ;
#endif
}

ASImage      *
picture2asimage(ASVisual *asv, Pixmap rgb, Pixmap a , int x, int y, unsigned int width, unsigned int height, unsigned long plane_mask, Bool keep_cache, unsigned int compression)
{
#ifndef X_DISPLAY_MISSING
	XImage       *xim = ASGetXImage (asv, rgb, x, y, width, height, plane_mask);
	XImage       *alpha_xim = (a==None)?NULL:ASGetXImage (asv, a, x, y, width, height, 0xFFFFFFFF);
	ASImage      *im = NULL;

	if (xim)
	{
		im = picture_ximage2asimage (asv, xim, alpha_xim, compression);
		if( keep_cache )
		{
			im->alt.ximage = xim ;
			if( alpha_xim )
			{
				im->alt.mask_ximage = alpha_xim ;
				if( alpha_xim->depth == 8 )
					set_flags( im->flags, ASIM_XIMAGE_8BIT_MASK );
			}
		}else
		{
			XDestroyImage (xim);
			if( alpha_xim )
				XDestroyImage (alpha_xim);
		}
	}
	return im;
#else
    return NULL ;
#endif
}

ASImage      *
pixmap2asimage(ASVisual *asv, Pixmap p, int x, int y, unsigned int width, unsigned int height, unsigned long plane_mask, Bool keep_cache, unsigned int compression)
{
	return picture2asimage(asv, p, None, x, y, width, height, plane_mask, keep_cache, compression);
}

Bool
put_ximage( ASVisual *asv, XImage *xim, Drawable d, GC gc,
            int src_x, int src_y, int dest_x, int dest_y,
  		    unsigned int width, unsigned int height )
{
#ifndef X_DISPLAY_MISSING
	GC 			  my_gc = gc ;

	if( src_x < 0 )
	{
		width += src_x ;
		src_x = 0;
	}else if( src_x > xim->width )
		return False;
	if( xim->width  > src_x+width )
		width = xim->width - src_x ;
	if( src_y < 0 )
	{
		height+= src_y ;
		src_y = 0;
	}else if( src_y > xim->height )
		return False;
	if( xim->height  > src_y+height )
		height = xim->height - src_y ;

	if( my_gc == NULL )
	{
		XGCValues gcv ;
		my_gc = XCreateGC( asv->dpy, d, 0, &gcv );
	}
	ASPutXImage( asv, d, my_gc, xim, src_x, src_y, dest_x, dest_y, width, height );
	if( my_gc != gc )
		XFreeGC( asv->dpy, my_gc );
	return True ;
#else
	return False ;
#endif
}

#define IS_POWER_OF2(i)  (((((i)>>16)&1)+(((i)>>15)&1)+(((i)>>14)&1)+(((i)>>13)&1)+(((i)>>12)&1)+(((i)>>11)&1)+(((i)>>10)&1)+(((i)>>9)&1)+(((i)>>8)&1) \
                         +(((i)>>7)&1)+(((i)>>6)&1)+(((i)>>5)&1)+(((i)>>4)&1)+(((i)>>3)&1)+(((i)>>2)&1))==1)

Bool
asimage2drawable_gl(	ASVisual *asv, Drawable d, ASImage *im,
                  		int src_x, int src_y, int dest_x, int dest_y,
        		  		int width, int height, int d_width, int d_height, 
						Bool force_direct )
{
	if( im != NULL && get_flags( asv->glx_support, ASGLX_Available ) && d != None )
	{
#ifdef HAVE_GLX
		int glbuf_size = (get_flags( asv->glx_support, ASGLX_RGBA )? 4 : 3 ) * width * height;
		CARD8 *glbuf = NULL;
		ASImageDecoder *imdec  = NULL ;
		GLXPixmap glxp = None;
		
		if ((imdec = start_image_decoding( asv, im, 
									   	get_flags( asv->glx_support, ASGLX_RGBA )?SCL_DO_ALL:SCL_DO_COLOR, 
									   	src_x, src_y, width, height, NULL)) != NULL )
		{	 
			int i, l = glbuf_size;
			glbuf = safemalloc( glbuf_size );
			for (i = 0; i < (int)height; i++)
			{	
				int k = width;
				imdec->decode_image_scanline( imdec ); 
				if( get_flags( asv->glx_support, ASGLX_RGBA ) ) 
				{	  
					while( --k >= 0 ) 
					{
						glbuf[--l] = imdec->buffer.alpha[k] ;
						glbuf[--l] = imdec->buffer.blue[k] ;
						glbuf[--l] = imdec->buffer.green[k] ;
						glbuf[--l] = imdec->buffer.red[k] ;
					}	 
				}else
				{	
					while( --k >= 0 ) 
					{
						glbuf[--l] = imdec->buffer.blue[k] ;
						glbuf[--l] = imdec->buffer.green[k] ;
						glbuf[--l] = imdec->buffer.red[k] ;
					}	 
				}
			}
			stop_image_decoding( &imdec );
		}else
			return False;

		if( !force_direct ) 
		{	
			glxp = glXCreateGLXPixmap( asv->dpy, &(asv->visual_info), d);
			/* d is either invalid drawable or is a window */
			if( glxp == None ) 
				force_direct = True ; 
		}                      
		if( glxp == None ) 
		{
			if( asv->glx_scratch_gc_direct	!= NULL )
				glXMakeCurrent (asv->dpy, d, asv->glx_scratch_gc_direct);
			else
				glXMakeCurrent (asv->dpy, d, asv->glx_scratch_gc_indirect);
		}else
			glXMakeCurrent (asv->dpy, glxp, asv->glx_scratch_gc_indirect);
		
		if( glGetError() != 0 ) 
			return False;

  		if ( get_flags( asv->glx_support, ASGLX_DoubleBuffer ) ) 
   			glDrawBuffer (GL_FRONT);

		glDisable(GL_BLEND);		/* optimize pixel transfer rates */
	  	glDisable (GL_DEPTH_TEST);
	  	glDisable (GL_DITHER);
	  	glDisable (GL_FOG);
	  	glDisable (GL_LIGHTING);

		glViewport( 0, 0, d_width, d_height);
		glMatrixMode (GL_PROJECTION);
		glLoadIdentity ();
		/* gluOrtho2D (0, d_width, 0, d_height); */
		glMatrixMode (GL_MODELVIEW);
		glLoadIdentity ();
		glTranslatef (0.375, 0.375, 0.0);


#if 1
		if( !IS_POWER_OF2(width) || !IS_POWER_OF2(height))
		{
			/* now put pixels on */
			glRasterPos2i( dest_x, d_height - (dest_y+height) );
			glDrawPixels(   width, height, 
							get_flags( asv->glx_support, ASGLX_RGBA )?GL_RGBA:GL_RGB, 
							GL_UNSIGNED_BYTE, glbuf );
		}else
#endif		
		{ /* this stuff might be faster : */
			GLuint texture ;

#define TARGET_TEXTURE_ID GL_TEXTURE_2D

#if TARGET_TEXTURE_ID!=GL_TEXTURE_2D
			glEnable(GL_TEXTURE_2D);
#endif			
			glEnable(TARGET_TEXTURE_ID);
			glGenTextures(1, &texture);

			glBindTexture(TARGET_TEXTURE_ID, texture);
	    	glTexParameteri(TARGET_TEXTURE_ID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	     	glTexParameteri(TARGET_TEXTURE_ID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(TARGET_TEXTURE_ID, 0, get_flags( asv->glx_support, ASGLX_RGBA )?GL_RGBA:GL_RGB, 
					      /* width and height must be the power of 2 !!! */
						  width, height, 
						  0, GL_RGBA, GL_UNSIGNED_BYTE, glbuf);

		 	glBegin(GL_QUADS);
			/* bottom-left */
		   	glTexCoord2d(0., 0.); glVertex2i(dest_x, d_height - (dest_y+height) );
			/* bottom-right */
		   	glTexCoord2d(1.0, 0.0); glVertex2i(dest_x+width, d_height - (dest_y+height));
			/* top-right */
		   	glTexCoord2d(1.0, 1.0); glVertex2i(dest_x+width, d_height - dest_y);    
			/* top-left */ 
		   	glTexCoord2d(0.0, 1.0); glVertex2i(dest_x, d_height - dest_y);
		   	glEnd();

			glBindTexture(TARGET_TEXTURE_ID, 0);
			glFinish();
		}							

		free( glbuf );
		glXMakeCurrent (asv->dpy, None, NULL);	  
		if( glxp ) 
			glXDestroyGLXPixmap( asv->dpy, glxp);
		glFinish(); 				   
		return True;
#endif /* #ifdef HAVE_GLX */
	}
	{
		static Bool warning_shown = False ; 
		if( !warning_shown ) 
		{
			warning_shown = True ;
			show_warning( "Support for GLX is unavailable.");
		}
	}
	return False;
}

Bool
asimage2drawable( ASVisual *asv, Drawable d, ASImage *im, GC gc,
                  int src_x, int src_y, int dest_x, int dest_y,
        		  unsigned int width, unsigned int height,
				  Bool use_cached)
{
#ifndef X_DISPLAY_MISSING
	if( im )
	{
		Bool 		  my_xim = False ;
		XImage       *xim ;
		Bool res = False;
		if ( !use_cached || im->alt.ximage == NULL )
		{
            if( (xim = asimage2ximage_ext( asv, im, False )) == NULL )
			{
				show_error("cannot export image into XImage.");
				return None ;
			}
			 my_xim = True ;
		}else
			xim = im->alt.ximage ;
		if (xim != NULL )
		{
            res = put_ximage( asv, xim, d, gc,  src_x, src_y, dest_x, dest_y, width, height );
			if( my_xim && xim == im->alt.ximage ) 
				im->alt.ximage = NULL ;
			if( xim != im->alt.ximage )
				XDestroyImage (xim);
		}
		return res;
	}
#endif
	return False ;
}

Bool
asimage2alpha_drawable( ASVisual *asv, Drawable d, ASImage *im, GC gc,
            		    int src_x, int src_y, int dest_x, int dest_y,
        				unsigned int width, unsigned int height,
						Bool use_cached)
{
#ifndef X_DISPLAY_MISSING
	if( im )
	{
		XImage       *xim ;
		unsigned int  alpha_depth = 1 ;
		int dumm; unsigned int udumm; Window root ;
		Bool res = False ;

		XGetGeometry( asv->dpy, d, &root, &dumm, &dumm, &udumm, &udumm, &udumm, &alpha_depth );

		if ( !use_cached || im->alt.mask_ximage == NULL || im->alt.mask_ximage->depth != alpha_depth )
		{
			if( (xim = asimage2alpha_ximage (asv, im, (alpha_depth == 1) )) == NULL )
			{
				show_error("cannot export image into alpha XImage.");
				return None ;
			}
		}else
			xim = im->alt.mask_ximage ;
		if (xim != NULL )
		{
			res = put_ximage( asv, xim, d, gc,	src_x, src_y, dest_x, dest_y, width, height );
			if( xim != im->alt.mask_ximage )
				XDestroyImage (xim);
		}
		return res;
	}
#endif
	return False ;
}


Pixmap
asimage2pixmap(ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached)
{
#ifndef X_DISPLAY_MISSING
	if( im )
	{
		Pixmap        p = None;

		p = create_visual_pixmap( asv, root, im->width, im->height, 0 );

		if( !asimage2drawable( asv, p, im, gc, 0, 0, 0, 0, im->width, im->height, use_cached) )
		{
			XFreePixmap( asv->dpy, p );
			p = None ;
		}
		return p;
	}
#endif
	return None ;
}

Pixmap
asimage2alpha(ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached, Bool bitmap)
{
#ifndef X_DISPLAY_MISSING
	XImage       *xim ;
	Pixmap        mask = None;
	GC 			  my_gc = gc ;

	int target_depth = bitmap?1:8;

	if ( !use_cached || im->alt.mask_ximage == NULL ||
	     im->alt.mask_ximage->depth != target_depth )
	{
		if( (xim = asimage2alpha_ximage( asv, im, bitmap )) == NULL )
		{
			show_error("cannot export image's mask into XImage.");
			return None ;
		}
	}else
		xim = im->alt.mask_ximage ;
	mask = create_visual_pixmap( asv, root, xim->width, xim->height, target_depth );
	if( my_gc == NULL )
	{
		XGCValues gcv ;
		my_gc = XCreateGC( asv->dpy, mask, 0, &gcv );
	}
	ASPutXImage( asv, mask, my_gc, xim, 0, 0, 0, 0, xim->width, xim->height );
	if( my_gc != gc )
		XFreeGC( asv->dpy, my_gc );
	if( xim != im->alt.mask_ximage )
		XDestroyImage (xim);
	return mask;
#else
	return None ;
#endif
}

Pixmap
asimage2mask(ASVisual *asv, Window root, ASImage *im, GC gc, Bool use_cached)
{
	return asimage2alpha(asv, root, im, gc, use_cached, True);
}
/* ********************************************************************************/
/* The end !!!! 																 */
/* ********************************************************************************/

