/*--------------------------------*-C-*---------------------------------*
 * File:	pixmap.c
 *----------------------------------------------------------------------*
 * Copyright (c) 1999 Ethan Fischer <allanon@crystaltokyo.com>
 * Copyright (c) 1999 Sasha Vasko   <sasha at aftercode.net>
 *
 * This program is free software; you can redistribute it and/or modify
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
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
 * Originally written:
 *    1999	Sasha Vasko <sasha at aftercode.net>
 *----------------------------------------------------------------------*/

#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

/*#define LOCAL_DEBUG */
/* #define DO_CLOCKING */

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
#include "blender.h"
#include "asimage.h"
#include "imencdec.h"
#include "ximage.h"
#include "transform.h"
#include "pixmap.h"


/*#define CREATE_TRG_PIXMAP(asv,w,h) XCreatePixmap(dpy, RootWindow(dpy,DefaultScreen(dpy)), (w), (h), DefaultDepth(dpy,DefaultScreen(dpy)))*/
#define CREATE_TRG_PIXMAP(asv,w,h) create_visual_pixmap(asv,RootWindow(asv->dpy,DefaultScreen(asv->dpy)),(w),(h),0)


/****************************************************************************
 *
 * fill part of a pixmap with the root pixmap, offset properly to look
 * "transparent"
 *
 ***************************************************************************/
int
FillPixmapWithTile (Pixmap pixmap, Pixmap tile, int x, int y, int width, int height, int tile_x, int tile_y)
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;
  if (tile != None && pixmap != None)
    {
      GC gc;
      XGCValues gcv;

      gcv.tile = tile;
      gcv.fill_style = FillTiled;
      gcv.ts_x_origin = -tile_x;
      gcv.ts_y_origin = -tile_y;
      gc = XCreateGC (dpy, tile, GCFillStyle | GCTile | GCTileStipXOrigin | GCTileStipYOrigin, &gcv);
      XFillRectangle (dpy, pixmap, gc, x, y, width, height);
      XFreeGC (dpy, gc);
      return 1;
    }
#endif
  return 0;
}

Pixmap
GetRootPixmap (Atom id)
{
	Pixmap currentRootPixmap = None;
#ifndef X_DISPLAY_MISSING
	static Atom root_pmap_atom = None ; 
	Display *dpy = get_default_asvisual()->dpy;
	if (id == None)
	{
		if( root_pmap_atom == None ) 	  
  			root_pmap_atom = XInternAtom (dpy, "_XROOTPMAP_ID", True);
		id = root_pmap_atom ;
	}

    if (id != None)
    {
  		Atom act_type;
    	int act_format;
    	unsigned long nitems, bytes_after;
    	unsigned char *prop = NULL;

/*fprintf(stderr, "\n aterm GetRootPixmap(): root pixmap is set");                  */
    	if (XGetWindowProperty (  dpy, RootWindow(dpy,DefaultScreen(dpy)), id, 0, 1, False, XA_PIXMAP,
							      &act_type, &act_format, &nitems, &bytes_after,
			    				  &prop) == Success)
		{
			if (prop)
	  		{
	    		currentRootPixmap = *((Pixmap *) prop);
	    		XFree (prop);
/*fprintf(stderr, "\n aterm GetRootPixmap(): root pixmap is [%lu]", currentRootPixmap); */
		    }
		}
    }
#endif
    return currentRootPixmap;
}

#ifndef X_DISPLAY_MISSING
static int
pixmap_error_handler (Display * dpy, XErrorEvent * error)
{
#ifdef DEBUG_IMAGING
	show_error ("XError # %u, in resource %lu, Request: %d.%d",
				 error->error_code, error->resourceid, error->request_code, error->minor_code);
#endif
  return 0;
}
#endif

Pixmap
ValidatePixmap (Pixmap p, int bSetHandler, int bTransparent, unsigned int *pWidth, unsigned int *pHeight)
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;
	int (*oldXErrorHandler) (Display *, XErrorEvent *) = NULL;
    /* we need to check if pixmap is still valid */
	Window root;
    int junk;
	unsigned int ujunk ;
	if (bSetHandler)
		oldXErrorHandler = XSetErrorHandler (pixmap_error_handler);

    if (bTransparent)
	    p = GetRootPixmap (None);
	if (!pWidth)
  		pWidth = &ujunk;
    if (!pHeight)
	    pHeight = &ujunk;

    if (p != None)
	{
  		if (!XGetGeometry (dpy, p, &root, &junk, &junk, pWidth, pHeight, &ujunk, &ujunk))
			p = None;
    }
	if(bSetHandler)
  		XSetErrorHandler (oldXErrorHandler);

	return p;
#else
	return None ;
#endif
}

int
GetRootDimensions (int *width, int *height)
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;
	if( dpy ) 
	{	
		*height = XDisplayHeight(dpy, DefaultScreen(dpy) );
		*width = XDisplayWidth(dpy, DefaultScreen(dpy) );
	}
	return 1 ;
#else
	return 0;
#endif
}

int
GetWinPosition (Window w, int *x, int *y)
{
	return get_dpy_window_position( get_default_asvisual()->dpy, None, w, NULL, NULL, x, y );
}

ARGB32
shading2tint32(ShadingInfo * shading)
{
	if( shading && !NO_NEED_TO_SHADE(*shading))
	{
		CARD16 r16 = ((shading->tintColor.red*shading->shading / 100)>>9)&0x00FF ;
		CARD16 g16 = ((shading->tintColor.green*shading->shading / 100)>>9)&0x00FF ;
		CARD16 b16 = ((shading->tintColor.blue*shading->shading / 100)>>9)&0x00FF ;
		CARD16 a16 = ((0x0000007F*shading->shading / 100))&0x00FF ;
		return MAKE_ARGB32(a16,r16,g16,b16);
	}
	return TINT_LEAVE_SAME ;
}

Pixmap
scale_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint)
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING

	if (src != None)
    {
		ASImage *src_im ;
		src_im = pixmap2ximage(asv, src, 0, 0, src_w, src_h, AllPlanes, 0 );
		if( src_im ) 
		{
			if( src_w != width || src_h != height ) 
			{
				ASImage *tmp = scale_asimage( asv, src_im, width, height, 
				                              (tint != TINT_LEAVE_SAME)?ASA_ASImage:ASA_XImage, 
											  0, ASIMAGE_QUALITY_DEFAULT );
				destroy_asimage( &src_im );
				src_im = tmp;
			}
			if( src_im && tint != TINT_LEAVE_SAME )
			{
				ASImage *tinted = tile_asimage ( asv, src_im, 0, 0,
				  								 width,  height, tint,
												 ASA_XImage,
												 0, ASIMAGE_QUALITY_DEFAULT );
				destroy_asimage( &src_im );
				src_im = tinted ;
			}
			if( src_im ) 
			{
				trg	= asimage2pixmap(asv, None, src_im, gc, True);
				destroy_asimage( &src_im );
			}
		}
	}
#endif
	return trg;
}

Pixmap
ScalePixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading)
{
	Pixmap p = None ;
#ifndef X_DISPLAY_MISSING
	p = scale_pixmap( get_default_asvisual(), src, src_w, src_h, width, height, gc, shading2tint32(shading) );
#endif
	return p;
}


void
copyshade_drawable_area( ASVisual *asv, Drawable src, Pixmap trg,
				  		 int x, int y, int w, int h,
				  		 int trg_x, int trg_y,
				  		 GC gc, ARGB32 tint) 		
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;
	if( tint == TINT_LEAVE_SAME || asv == NULL )
	{
		XCopyArea (dpy, src, trg, gc, x, y, w, h, trg_x, trg_y);
	}else
	{
		ASImage *src_im = pixmap2ximage( asv, src, x, y, w, h, AllPlanes, 0 );
		if( src_im )
		{
			ASImage *tinted = tile_asimage ( asv, src_im, 0, 0,
		  									 w,  h, tint,
											 ASA_XImage,
											 0, ASIMAGE_QUALITY_DEFAULT );
			destroy_asimage( &src_im );
			if( tinted ) 
			{
				asimage2drawable( asv, trg, tinted, gc,
                				  0, 0, trg_x, trg_y,
        						  w, h, True );
				destroy_asimage( &tinted );
			}
		}		
	}
#endif
}

void
CopyAndShadeArea ( Drawable src, Pixmap trg,
				   int x, int y, int w, int h,
				   int trg_x, int trg_y,
				   GC gc, ShadingInfo * shading)
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;
	ARGB32 tint = shading2tint32( shading );

    if (x < 0 || y < 0)
		return;

	if( tint == TINT_LEAVE_SAME )
	{
		XCopyArea (dpy, src, trg, gc, x, y, w, h, trg_x, trg_y);
	}else
	{
		copyshade_drawable_area( get_default_asvisual(), src, trg, x, y, w, h, trg_x, trg_y, gc, tint );
	}
#endif
}

void
tile_pixmap (ASVisual *asv, Pixmap src, Pixmap trg, int src_w, int src_h, int x, int y, int w, int h, GC gc, ARGB32 tint)
{
#ifndef X_DISPLAY_MISSING
	int tile_x, tile_y, left_w, bott_h;
  
    tile_x = x % src_w;
	tile_y = y % src_h;
	left_w = MIN (src_w - tile_x, w);
	bott_h = MIN (src_h - tile_y, h);

/*fprintf( stderr, "\nShadeTiledPixmap(): tile_x = %d, tile_y = %d, left_w = %d, bott_h = %d, SRC = %dx%d TRG=%dx%d", tile_x, tile_y, left_w, bott_h, src_w, src_h, w, h); */

	/* We don't really want to do simple tile_asimage here since if tint is notint ,
	 * then we could get by with simple XCopyArea !!! 
	 */
	copyshade_drawable_area( asv, src, trg, tile_x, tile_y, left_w, bott_h, 0, 0, gc, tint);
    if (bott_h < h)
    {				/* right-top parts */
        copyshade_drawable_area( asv, src, trg, tile_x, 0, left_w, h - bott_h, 0, bott_h, gc, tint);
    }
	if (left_w < w)
    {				/* left-bott parts */
  	    copyshade_drawable_area( asv, src, trg, 0, tile_y, w - left_w, bott_h, left_w, 0, gc, tint);
    	if (bott_h < h)		/* left-top parts */
			copyshade_drawable_area( asv, src, trg, 0, 0, w - left_w, h - bott_h, left_w, bott_h, gc, tint);
    }
#endif
}

void
ShadeTiledPixmap (Pixmap src, Pixmap trg, int src_w, int src_h, int x, int y, int w, int h, GC gc, ShadingInfo * shading)
{
#ifndef X_DISPLAY_MISSING
	ARGB32 tint = shading2tint32( shading );
	tile_pixmap (get_default_asvisual(), src, trg, src_w, src_h, x, y, w, h, gc, tint);
#endif
}

Pixmap
shade_pixmap (ASVisual *asv, Pixmap src, int x, int y, int width, int height, GC gc, ARGB32 tint)
{
#ifndef X_DISPLAY_MISSING

    Pixmap trg = CREATE_TRG_PIXMAP (asv, width, height);
  
	if (trg != None)
    	copyshade_drawable_area (asv, src, trg, x, y, width, height, 0, 0, gc, tint);
	return trg;
#else
	return None ;
#endif
}

Pixmap
ShadePixmap (Pixmap src, int x, int y, int width, int height, GC gc, ShadingInfo * shading)
{
    Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	ARGB32 tint = shading2tint32( shading );
  
	trg = CREATE_TRG_PIXMAP (get_default_asvisual(), width, height);
	if (trg != None)
    {
    	copyshade_drawable_area (get_default_asvisual(), src, trg, x, y, width, height, 0, 0, gc, tint);
    }
#endif
	return trg;
}

Pixmap
center_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint)
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;

    int x, y, w, h, src_x = 0, src_y = 0;
	/* create target pixmap of the size of the window */
	trg = CREATE_TRG_PIXMAP (asv,width, height);
	if (trg != None)
    {
    	/* fill it with background color */
    	XFillRectangle (dpy, trg, gc, 0, 0, width, height);
    	/* place image at the center of it */
    	x = (width - src_w) >> 1;
    	y = (height - src_h) >> 1;
    	if (x < 0)
		{
			src_x -= x;
			w = MIN (width, src_w + x);
			x = 0;
		}else
			w = MIN (width, src_w);
    	if (y < 0)
		{
			src_y -= y;
			h = MIN (height, src_h + y);
			y = 0;
		}else
			h = MIN (height, src_h);

    	copyshade_drawable_area ( asv, src, trg, src_x, src_y, w, h, x, y, gc, tint);
    }
#endif
	return trg;
}

Pixmap
CenterPixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading)
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	ARGB32 tint = shading2tint32( shading );
  
	trg = center_pixmap( get_default_asvisual(), src, src_w, src_h, width, height, gc, tint );
#endif		
	return trg ;	
}

Pixmap
grow_pixmap (ASVisual *asv, Pixmap src, int src_w, int src_h, int width, int height, GC gc, ARGB32 tint )
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;

	int w, h;
	/* create target pixmap of the size of the window */
	trg = CREATE_TRG_PIXMAP (asv,width, height);
	if (trg != None)
    {
    	/* fill it with background color */
    	XFillRectangle (dpy, trg, gc, 0, 0, width, height);
    	/* place image at the center of it */
    	w = MIN (width, src_w);
    	h = MIN (height, src_h);

    	copyshade_drawable_area(asv, src, trg, 0, 0, w, h, 0, 0, gc, tint);
    }
#endif	
	return trg;
}

Pixmap
GrowPixmap (Pixmap src, int src_w, int src_h, int width, int height, GC gc, ShadingInfo * shading)
{
	Pixmap trg = None;
#ifndef X_DISPLAY_MISSING
	ARGB32 tint = shading2tint32( shading );
  
	trg = grow_pixmap( get_default_asvisual(), src, src_w, src_h, width, height, gc, tint );
#endif		
	return trg ;	
}

/****************************************************************************
 * grab a section of the screen and darken it
 ***************************************************************************/
static Pixmap
cut_pixmap ( ASVisual *asv, Pixmap src, Pixmap trg,
             int x, int y,
	  		 unsigned int src_w, unsigned int src_h,
	  		 unsigned int width, unsigned int height,
	  		 GC gc, ARGB32 tint )
{
#ifndef X_DISPLAY_MISSING
	Display *dpy = get_default_asvisual()->dpy;

	Bool my_pixmap = (trg == None )?True:False ;
	int screen_w, screen_h ;
	int w = width, h = height;
	int offset_x = 0, offset_y = 0;
	int screen = DefaultScreen(dpy);

	if (width < 2 || height < 2 )
  		return trg;

	screen_w = DisplayWidth( dpy, screen );
	screen_h = DisplayHeight( dpy, screen );

	while( x+(int)width < 0 )  x+= screen_w ;
	while( x >= screen_w )  x-= screen_w ;
	while( y+(int)height < 0 )  y+= screen_h ;
	while( y >= screen_h )  y-= screen_h ;

	if( x < 0 )
	{
  		offset_x = (-x);
  		w -= offset_x ;
  		x = 0 ;
	}
	if( y < 0 )
	{
  		offset_y = (-y) ;
  		h -= offset_y;
  		y = 0 ;
	}
	if( x+w >= screen_w )
  		w = screen_w - x ;

    if( y+height >= screen_h )
	    h = screen_h - y ;
	if( (src_w == 0 || src_h == 0) && src != None ) 
	{
		Window root;
      	unsigned int dum;
      	int dummy;
		if (!XGetGeometry (dpy, src, &root, &dummy, &dummy, &src_w, &src_h, &dum, &dum))
			src = None ;
	}	 

	if (src == None) /* we don't have root pixmap ID */
    { /* we want to create Overrideredirect window overlapping out window
         with background type of Parent Relative and then grab it */
    	XSetWindowAttributes attr ;
    	XEvent event ;
    	int tick_count = 0 ;
    	Bool grabbed = False ;
        
		attr.background_pixmap = ParentRelative ;
		attr.backing_store = Always ;
		attr.event_mask = ExposureMask ;
		attr.override_redirect = True ;
		src = create_visual_window(asv, RootWindow(dpy,screen), x, y, w, h,
	  		                0, CopyFromParent, 
			  				CWBackPixmap|CWBackingStore|CWOverrideRedirect|CWEventMask,
			  				&attr); 

		if( src == None ) return trg ;
		XGrabServer( dpy );
		grabbed = True ;
		XMapRaised( dpy, src );
		XSync(dpy, False );
		start_ticker(1);
		/* now we have to wait for our window to become mapped - waiting for Expose */
		for( tick_count = 0 ; !XCheckWindowEvent( dpy, src, ExposureMask, &event ) && tick_count < 100 ; tick_count++)
	  		wait_tick();
		if( tick_count < 100 )
		{
		    if( trg == None )    trg = CREATE_TRG_PIXMAP (asv,width, height);
	  		if (trg != None)
		    {	/* custom code to cut area, so to ungrab server ASAP */
		        if (tint != TINT_LEAVE_SAME)
	  		    {
	  				ASImage *src_im = pixmap2ximage( asv, src, 0, 0, w, h, AllPlanes, 0 );
					XDestroyWindow( dpy, src );
				    src = None ;
					XUngrabServer( dpy );
					grabbed = False ;
					if (src_im != NULL)
					{
						ASImage *tinted = tile_asimage ( asv, src_im, 0, 0,
		  												 w,  h, tint,
														 ASA_XImage,
														 0, ASIMAGE_QUALITY_DEFAULT );
						destroy_asimage( &src_im );
						asimage2drawable( asv, trg, tinted, gc,
      			        				  0, 0, offset_x, offset_y,
        								  w, h, True );
						destroy_asimage( &tinted );
					}else if( my_pixmap )
					{
		  				XFreePixmap( dpy, trg );
		  				trg = None ;
					}
				}else
		  			XCopyArea (dpy, src, trg, gc, 0, 0, w, h, offset_x, offset_y);
	  		}
        }
		if( src )
	  		XDestroyWindow( dpy, src );
		if( grabbed )
	  		XUngrabServer( dpy );
		return trg ;
    }
	/* we have root pixmap ID */
	/* find out our coordinates relative to the root window */
	if (x + w > src_w || y + h > src_h)
    {			/* tiled pixmap processing here */
  		Pixmap tmp ;
    	w = MIN (w, (int)src_w);
    	h = MIN (h, (int)src_h);

    	tmp = CREATE_TRG_PIXMAP (asv, w, h);
    	if (tmp != None)
    	{
			tile_pixmap (asv, src, tmp, src_w, src_h, x, y, w, h, gc, tint);
      		if( trg == None )
			{
          		if( (trg = CREATE_TRG_PIXMAP (asv, w+offset_x, h+offset_y)) != None )
					XCopyArea (dpy, tmp, trg, gc, 0, 0, w, h, offset_x, offset_y);
			}else
	  			FillPixmapWithTile( trg, tmp, offset_x, offset_y, width, height, 0, 0 );

			XFreePixmap( dpy, tmp );
      		return trg;
    	}
    }

	/* create target pixmap of the size of the window */
	if( trg == None )    
		trg = CREATE_TRG_PIXMAP (asv, width, height);
	if (trg != None)
    {	/* cut area */
		copyshade_drawable_area( asv, src, trg, x, y, w, h, offset_x, offset_y, gc, tint); 		
    }
	return trg;
#else
	return None ;
#endif	
}

static Pixmap
CutPixmap ( Pixmap src, Pixmap trg,
            int x, int y,
	    unsigned int src_w, unsigned int src_h,
	    unsigned int width, unsigned int height,
	    GC gc, ShadingInfo * shading)
{
	Pixmap res = None;
#ifndef X_DISPLAY_MISSING
	ARGB32 tint = shading2tint32( shading );
	res = cut_pixmap( get_default_asvisual(), src, trg, x, y, src_w, src_h, width, height, gc, tint );
#endif		
	return res ;	
}

Pixmap
cut_win_pixmap (ASVisual *asv, Window win, Drawable src, int src_w, int src_h, int width,
	  		    int height, GC gc, ARGB32 tint)
{
  int x = 0, y = 0;

  if (!get_dpy_window_position( asv->dpy, None, win, NULL, NULL, &x, &y))
	return None;

  return cut_pixmap( asv, src, None, x, y, src_w, src_h, width, height, gc, tint );
}

Pixmap
CutWinPixmap (Window win, Drawable src, int src_w, int src_h, int width,
	      int height, GC gc, ShadingInfo * shading)
{
  int x = 0, y = 0;

  if (!get_dpy_window_position( get_default_asvisual()->dpy, None, win, NULL, NULL, &x, &y))
	return None;

  return CutPixmap( src, None, x, y, src_w, src_h, width, height, gc, shading );
}

/* PROTO */
int
fill_with_darkened_background (ASVisual *asv, Pixmap * pixmap, ARGB32 tint, int x, int y, int width, int height, int root_x, int root_y, int bDiscardOriginal, ASImage *root_im)
{
#ifndef X_DISPLAY_MISSING
	unsigned int root_w, root_h;
	Pixmap root_pixmap;
	Display *dpy = get_default_asvisual()->dpy;
	int screen = DefaultScreen(dpy);

	/* added by Sasha on 02/24/1999 to use transparency&shading provided by
       libasimage 1.1 */
	root_pixmap = ValidatePixmap (None, 1, 1, &root_w, &root_h);

	if (root_pixmap != None)
    {
		if (*pixmap == None)
		{
			*pixmap = create_visual_pixmap(asv, RootWindow (dpy, screen), width, height, 0);
			bDiscardOriginal = 1;
		}
		
    	if ( tint != TINT_LEAVE_SAME)
		{
			ASImage *src_im = (root_im == NULL)?pixmap2ximage( asv, root_pixmap, 0, 0, root_w, root_h, AllPlanes, 0 ):root_im;
			if( root_im )
			{
				ASImage *tinted = tile_asimage ( asv, src_im, -root_x, -root_y,
		  										 width,  height, tint,
												 ASA_XImage,
											     0, ASIMAGE_QUALITY_DEFAULT );
				if( root_im != src_im )
					destroy_asimage( &src_im );
				if( tinted ) 
				{
					asimage2drawable( asv, *pixmap, tinted, NULL,
                					  0, 0, x, y,
        							  width, height, True );
					destroy_asimage( &tinted );
				}
			}		
		}else
	    	FillPixmapWithTile (*pixmap, root_pixmap, x, y, width, height, root_x, root_y);
        return 1;
	}
#endif
	return 0;
}

/****************************************************************************
 * grab a section of the screen and combine it with an XImage
 ***************************************************************************/
int
fill_with_pixmapped_background (ASVisual *asv, Pixmap * pixmap, ASImage *image, int x, int y, int width, int height, int root_x, int root_y, int bDiscardOriginal, ASImage *root_im)
{
#ifndef X_DISPLAY_MISSING
	unsigned int root_w, root_h;
	Pixmap root_pixmap;
	int screen = DefaultScreen(asv->dpy);

	root_pixmap = ValidatePixmap (None, 1, 1, &root_w, &root_h);
	if (root_pixmap != None)
    {
		ASImageLayer layers[2];
		ASImage *merged_im ;
		
		init_image_layers( &layers[0], 2 );
		layers[0].merge_scanlines = allanon_scanlines ;
		layers[0].im = root_im ?root_im:
		                        pixmap2ximage( asv, root_pixmap, 0, 0, root_w, root_h, AllPlanes, 0 );
		layers[0].dst_x = x ;
		layers[0].dst_y = y ;
		layers[0].clip_x = root_x ;
		layers[0].clip_y = root_y ;
		layers[0].clip_width = width ;
		layers[0].clip_height = height ;
		
		layers[1].im = image ; 
		layers[1].dst_x = x ;
		layers[1].dst_y = y ;
		layers[1].clip_x = 0 ;
		layers[1].clip_y = 0 ;
		layers[1].clip_width = width ;
		layers[1].clip_height = height ;

		merged_im = merge_layers( asv, &layers[0], 2,
		                          width, height,
								  ASA_XImage,
							      0, ASIMAGE_QUALITY_DEFAULT );
		if( root_im != layers[0].im )
			destroy_asimage( &(layers[0].im) );

		if( merged_im ) 
		{
  			if (*pixmap == None)
				*pixmap = create_visual_pixmap (asv, RootWindow (asv->dpy, screen), width, height, 0);

			asimage2drawable( asv, *pixmap, merged_im, NULL,
          					  0, 0, x, y,
  							  width, height, True );
			destroy_asimage( &merged_im );
		}			
    	return 1;
    }
#endif
	return 0;
}
/************************************************/
