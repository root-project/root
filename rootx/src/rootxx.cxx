// @(#)root/rootx:$Name:  $:$Id: rootxx.cxx,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Fons Rademakers   19/02/98

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rootxx                                                               //
//                                                                      //
// X11 based routines used to display the splash screen for rootx,      //
// the root front-end program.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <unistd.h>
#include <X11/Xlib.h>

#include "RConfig.h"
#include "Xpm.h"

#if defined(R__AIX) || defined(R__SOLARIS)
#   include <sys/select.h>
#endif
#include <time.h>
#include <sys/time.h>

#ifdef XpmVersion
#ifndef XpmSuccess
#define XpmSuccess       0
#endif
#ifndef XpmColorError
#define XpmColorError    1
#endif
#endif

// X bitmaps
#include "rootlogo_xbm.h"
#ifdef XpmVersion
#include "rootlogo_xpm.h"
#endif


static Display *gDisplay    = 0;
static Window   gLogoWindow = 0;
static Pixmap   gLogoPixmap = 0;

static struct timeval gPopupTime;



static void StayUp(int milliSec)
{
   // Make sure milliSec milliseconds have passed since logo was popped up.

   // get current time
   struct timeval ctv, dtv, tv;

   tv.tv_sec  = milliSec / 1000;
   tv.tv_usec = (milliSec % 1000) * 1000;

   gettimeofday(&ctv, 0);
   if ((dtv.tv_usec = ctv.tv_usec - gPopupTime.tv_usec) < 0) {
      dtv.tv_usec += 1000000;
      gPopupTime.tv_sec++;
   }
   dtv.tv_sec = ctv.tv_sec - gPopupTime.tv_sec;

   if ((ctv.tv_usec = tv.tv_usec - dtv.tv_usec) < 0) {
      tv.tv_usec += 1000000;
      dtv.tv_sec++;
   }
   ctv.tv_sec = tv.tv_sec - dtv.tv_sec;
   if (ctv.tv_sec < 0) return;

   select(0, 0, 0, 0, &ctv);
}

static Pixel Color(const char *name)
{
   // Convert NAME into a color, using PIX as default.

   XColor exact, color;
   Colormap cmap = DefaultColormap(gDisplay, DefaultScreen(gDisplay));

   XAllocNamedColor(gDisplay, cmap, (char*)name, &exact, &color);

   return color.pixel;
}

static Pixmap GetRootLogo()
{
   Pixmap logo = 0;
   int depth = PlanesOfScreen(XDefaultScreenOfDisplay(gDisplay));

#ifdef XpmVersion
   if (depth > 1) {
      XWindowAttributes win_attr;
      XGetWindowAttributes(gDisplay, gLogoWindow, &win_attr);

      XpmAttributes attr;
      attr.valuemask    = XpmVisual | XpmColormap | XpmDepth;
      attr.visual       = win_attr.visual;
      attr.colormap     = win_attr.colormap;
      attr.depth        = win_attr.depth;

#ifdef XpmColorKey              // Not available in XPM 3.2 and earlier
      attr.valuemask |= XpmColorKey;
      if (depth > 4)
         attr.color_key = XPM_COLOR;
      else if (depth > 2)
         attr.color_key = XPM_GRAY4;
      else if (depth > 1)
         attr.color_key = XPM_GRAY;
      else if (depth == 1)
         attr.color_key = XPM_MONO;
      else
         attr.valuemask &= ~XpmColorKey;

#endif // defined(XpmColorKey)

      int ret = XpmCreatePixmapFromData(gDisplay, gLogoWindow,
                                        (char **)rootlogo, &logo,
                                        (Pixmap *)0, &attr);
      XpmFreeAttributes(&attr);

      if (ret == XpmSuccess || ret == XpmColorError)
          return logo;

      if (logo)
         XFreePixmap(gDisplay, logo);
      logo = 0;
   }

#endif // defined(XpmVersion)

   if (depth > 4)
      logo = XCreatePixmapFromBitmapData(gDisplay, gLogoWindow,
                                         rootlogo_bits,
                                         rootlogo_width, rootlogo_height,
                                         Color("brown"),
                                         Color("white"),
                                         depth);
   else {
      int screen = DefaultScreen(gDisplay);
      logo = XCreatePixmapFromBitmapData(gDisplay, gLogoWindow,
                                         rootlogo_bits,
                                         rootlogo_width, rootlogo_height,
                                         BlackPixel(gDisplay, screen),
                                         WhitePixel(gDisplay, screen),
                                         depth);
   }
   return logo;
}

void PopupLogo()
{
   // Popup logo, waiting till ROOT is ready to run.

   gDisplay = XOpenDisplay("");
   if (!gDisplay) return;

   Pixel back, fore;
   int screen = DefaultScreen(gDisplay);

   back = WhitePixel(gDisplay, screen);
   fore = BlackPixel(gDisplay, screen);

   gLogoWindow = XCreateSimpleWindow(gDisplay, DefaultRootWindow(gDisplay),
                                     -100, -100, 50, 50, 0, fore, back);

   gLogoPixmap = GetRootLogo();

   Window root;
   int x, y;
   unsigned int w, h, bw, depth;
   XGetGeometry(gDisplay, gLogoPixmap, &root, &x, &y, &w, &h, &bw, &depth);

   Screen *xscreen = XDefaultScreenOfDisplay(gDisplay);
   x = (WidthOfScreen(xscreen) - w) / 2;
   y = (HeightOfScreen(xscreen) - h) / 2;

   XMoveResizeWindow(gDisplay, gLogoWindow, x, y, w, h);
   XSync(gDisplay, False);   // make sure move & resize is done before mapping

   unsigned long valmask;
   XSetWindowAttributes xswa;
   valmask = CWBackPixmap | CWOverrideRedirect;
   xswa.background_pixmap = gLogoPixmap;
   xswa.override_redirect = True;
   XChangeWindowAttributes(gDisplay, gLogoWindow, valmask, &xswa);

   XMapRaised(gDisplay, gLogoWindow);
   XSync(gDisplay, False);

   gettimeofday(&gPopupTime, 0);
}

void PopdownLogo()
{
   // Pop down the logo. ROOT is ready to run.

   StayUp(4000);

   if (gLogoWindow) {
      XUnmapWindow(gDisplay, gLogoWindow);
      XDestroyWindow(gDisplay, gLogoWindow);
      gLogoWindow = 0;
   }
   if (gLogoPixmap) {
      XFreePixmap(gDisplay, gLogoPixmap);
      gLogoPixmap = 0;
   }
   if (gDisplay) {
      XSync(gDisplay, False);
      XCloseDisplay(gDisplay);
      gDisplay = 0;
   }
}

void CloseDisplay()
{
   // Close connection to X server (called by child).

   if (gDisplay)
      close(ConnectionNumber(gDisplay));
}
