// @(#)root/rootx:$Id$
// Author: Fons Rademakers   19/02/98
// Re-written for ROOT 6 by Timur Pocheptsov 11/03/2014.

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rootxx                                                               //
//                                                                      //
// X11 based routines used to display the splash screen for rootx,      //
// the root front-end program.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pwd.h>
#include <sys/types.h>
#include <X11/Xlib.h>
#include <X11/xpm.h>
#include <X11/extensions/shape.h>

#include "Rtypes.h"

#if defined(R__AIX) || defined(R__SOLARIS)
#   include <sys/select.h>
#endif
#include <time.h>
#include <sys/time.h>

static Display     *gDisplay       = 0;
static Window       gLogoWindow    = 0;
static Pixmap       gLogoPixmap    = 0;
//Non-rect window:
static Pixmap       gShapeMask     = 0;
//
static Pixmap       gCreditsPixmap = 0;
static GC           gGC            = 0;
static XFontStruct *gFont          = 0;
static bool         gDone          = false;
static bool         gMayPopdown    = false;
static bool         gAbout         = false;
//Non-rect window:
static bool         gHasShapeExt   = false;
//
static Pixel        gBackground    = Pixel();
static Pixel        gForeground    = Pixel();


static unsigned int gWidth         = 0;
static unsigned int gHeight        = 0;
static int          gStayUp        = 4000;   // 4 seconds

//gCreditsRect: x and y to be set at runtime (depending on shape extension and
//images).
static XRectangle   gCreditsRect   = {0, 0, 455, 80}; // clip rect in logo

static unsigned int gCreditsWidth  = gCreditsRect.width; // credits pixmap size
static unsigned int gCreditsHeight = 0;

static struct timeval gPopupTime;

static const char *gConception[] = {
   "Rene Brun",
   "Fons Rademakers",
   0
};

static const char *gLeadDevelopers[] = {
   "Rene Brun",
   "Philippe Canal",
   "Fons Rademakers",
   0
};

static const char *gRootDevelopers[] = {
   "Bertrand Bellenot",
   "Olivier Couet",
   "Gerardo Ganis",
   "Andrei Gheata",
   "Lorenzo Moneta",
   "Axel Naumann",
   "Paul Russo",
   "Matevz Tadel",
   0
};

//static const char *gCintDevelopers[] = {
//   "Masaharu Goto",
//   0
//};

static const char *gRootDocumentation[] = {
   "Ilka Antcheva",
   0
};

static char **gContributors = 0;

//__________________________________________________________________
static bool StayUp(int milliSec)
{
   // Returns false if milliSec milliseconds have passed since logo
   // was popped up, true otherwise.

   struct timeval ctv, dtv, tv, ptv = gPopupTime;

   tv.tv_sec  = milliSec / 1000;
   tv.tv_usec = (milliSec % 1000) * 1000;

   gettimeofday(&ctv, 0);
   if ((dtv.tv_usec = ctv.tv_usec - ptv.tv_usec) < 0) {
      dtv.tv_usec += 1000000;
      ptv.tv_sec++;
   }
   dtv.tv_sec = ctv.tv_sec - ptv.tv_sec;

   if ((ctv.tv_usec = tv.tv_usec - dtv.tv_usec) < 0) {
      ctv.tv_usec += 1000000;
      dtv.tv_sec++;
   }
   ctv.tv_sec = tv.tv_sec - dtv.tv_sec;

   if (ctv.tv_sec < 0) return false;

   return true;
}

//__________________________________________________________________
static void Sleep(int milliSec)
{
   // Sleep for specified amount of milli seconds.

   // get current time
   struct timeval tv;

   tv.tv_sec  = milliSec / 1000;
   tv.tv_usec = (milliSec % 1000) * 1000;

   select(0, 0, 0, 0, &tv);
}

//Aux. "Window management" part.

//__________________________________________________________________
static bool CreateSplashscreenWindow()
{
   assert(gDisplay != 0 && "CreateSplashscreenWindow, gDisplay is None");

   //TODO: check the screen???
   const int screen = DefaultScreen(gDisplay);

   gBackground = WhitePixel(gDisplay, screen);
   gForeground = BlackPixel(gDisplay, screen);

   gLogoWindow = XCreateSimpleWindow(gDisplay, DefaultRootWindow(gDisplay),
                                     -100, -100, 50, 50, 0,
                                     gForeground, gBackground);

   return gLogoWindow;
}

//__________________________________________________________________
static void SetSplashscreenPosition()
{
   assert(gDisplay != 0 && "SetSplashscreenPosition, gDisplay is None");
   assert(gLogoWindow != 0 && "SetSplashscreenPosition, gLogoWindow is None");
   
   Window rootWindow = Window();
   int x = 0, y = 0;
   unsigned int borderWidth = 0, depth = 0;
   XGetGeometry(gDisplay, gLogoPixmap, &rootWindow, &x, &y,
                &gWidth, &gHeight, &borderWidth, &depth);

   //TODO: this is wrong with multi-head display setup!
   Screen * const screen = XDefaultScreenOfDisplay(gDisplay);
   if (screen) {
      x = (WidthOfScreen(screen) - gWidth) / 2;
      y = (HeightOfScreen(screen) - gHeight) / 2;
   } else {
      //Some stupid numbers.
      x = 100;
      y = 100;
   }

   XMoveResizeWindow(gDisplay, gLogoWindow, x, y, gWidth, gHeight);
   XSync(gDisplay, False);// make sure move & resize is done before mapping

}

//__________________________________________________________________
static void SetBackgroundPixmapAndMask()
{
   assert(gDisplay != 0 && "SetBackgroundPixmapAndMask, gDisplay is None");
   assert(gLogoWindow != 0 && "SetBackgroundPixmapAndMask, gLogoWindow is None");
   assert(gLogoPixmap != 0 && "SetBackgroundPixmapAndMask, gLogoPixmap is None");

   unsigned long mask = CWBackPixmap | CWOverrideRedirect;
   XSetWindowAttributes winAttr = {};
   winAttr.background_pixmap = gLogoPixmap;
   winAttr.override_redirect = True;
   XChangeWindowAttributes(gDisplay, gLogoWindow, mask, &winAttr);
   
   if (gHasShapeExt) {
      assert(gShapeMask != 0 && "SetBackgroundPixmapAndMask, gShapeMask is None");
      XShapeCombineMask(gDisplay, gLogoWindow, ShapeBounding, 0, 0, gShapeMask, ShapeSet);
   }
}

//__________________________________________________________________
static void SelectFontAndTextColor()
{
   assert(gDisplay != 0 && "SelectFontAndTextColor, gDisplay is None");
   assert(gLogoWindow != 0 && "SelectFontAndTextColor, gLogoWindow is None");

   if (!(gGC = XCreateGC(gDisplay, gLogoWindow, 0, 0))) {
      printf("rootx - XCreateGC failed\n");
      return;
   }

   gFont = XLoadQueryFont(gDisplay, "-adobe-helvetica-medium-r-*-*-10-*-*-*-*-*-iso8859-1");
   if (!gFont) {
      printf("Couldn't find font \"-adobe-helvetica-medium-r-*-*-10-*-*-*-*-*-iso8859-1\",\n"
             "trying \"fixed\". Please fix your system so helvetica can be found, \n"
             "this font typically is in the rpm (or pkg equivalent) package \n"
             "XFree86-[75,100]dpi-fonts or fonts-xorg-[75,100]dpi.\n");
      gFont = XLoadQueryFont(gDisplay, "fixed");
      if (!gFont)
         printf("Also couln't find font \"fixed\", your system is terminally misconfigured.\n");
   }

   if (gFont)
      XSetFont(gDisplay, gGC, gFont->fid);
   
   XSetForeground(gDisplay, gGC, gForeground);
   XSetBackground(gDisplay, gGC, gBackground);
}

//__________________________________________________________________
static bool LoadROOTLogoPixmap(const char *imageFileName, bool needMask)
{
   //Splashscreen background image and (probably) a mask (if we want to use
   //shape combine - non-rect window).

   assert(imageFileName != 0 && "LoadROOTLogoPixmap, parameter 'imageFileName' is null");
   assert(gDisplay != 0 && "LoadROOTLogoPixmap, gDisplay is null");
   assert(gLogoWindow != 0 && "LoadROOTLogoPixmap, gLogoWindow is None");//'None' instead of '0'?

   Screen * const screen = XDefaultScreenOfDisplay(gDisplay);
   if (!screen)
      return false;

   //TODO: Check the result?
   const int depth = PlanesOfScreen(screen);

   XWindowAttributes winAttr = {};
   XGetWindowAttributes(gDisplay, gLogoWindow, &winAttr);

   XpmAttributes xpmAttr = {};
   xpmAttr.valuemask    = XpmVisual | XpmColormap | XpmDepth;
   xpmAttr.visual       = winAttr.visual;
   xpmAttr.colormap     = winAttr.colormap;
   xpmAttr.depth        = winAttr.depth;

#ifdef XpmColorKey              // Not available in XPM 3.2 and earlier
   xpmAttr.valuemask |= XpmColorKey;
   if (depth > 4)
      xpmAttr.color_key = XPM_COLOR;
   else if (depth > 2)
      xpmAttr.color_key = XPM_GRAY4;
   else if (depth > 1)
      xpmAttr.color_key = XPM_GRAY;
   else if (depth == 1)
      xpmAttr.color_key = XPM_MONO;
   else
      xpmAttr.valuemask &= ~XpmColorKey;
#endif // defined(XpmColorKey)

   if (!strlen(imageFileName) || !imageFileName[0])//at least some check.
      return false;

   std::string path(100, ' ');
   
#ifdef ROOTICONPATH
   assert(strlen(ROOTICONPATH) != 0 &&
          "LoadROOTLogoPixmap, invalid 'ROOTICONPATH'");

   path = ROOTICONPATH;
   path += "/";
   path += imageFileName;
#else
   assert(strlen(getenv("ROOTSYS")) != 0 &&
          "LoadROOTLogoPixmap, the $ROOTSYS string is too long");
   path = getenv("ROOTSYS");
   path += "/icons/";
   path += imageFileName;
#endif

   Pixmap logo = None, mask = None;
   const int ret = XpmReadFileToPixmap(gDisplay, gLogoWindow, path.c_str(), &logo,
                                       gHasShapeExt ? &mask : 0, &xpmAttr);
   XpmFreeAttributes(&xpmAttr);

   if ((ret == XpmSuccess || ret == XpmColorError) && logo) {
      if (needMask) {
         if (mask) {
            gLogoPixmap = logo;
            gShapeMask = mask;
            return true;
         }
      } else {
         gLogoPixmap = logo;
         return true;
      }
   }

   printf("rootx xpm error: %s\n", XpmGetErrorString(ret));

   if (logo)
      XFreePixmap(gDisplay, logo);

   if (mask)
      XFreePixmap(gDisplay, mask);

   return false;
}

//__________________________________________________________________
static bool GetRootLogoAndShapeMask()
{
   //1. Test if X11 supports shape combine mask.
   //2.a if no - go to 3.
   //2.b If yes - try to read both background image
   //    and the mask. If any of operations failed - go to 3.
   //    If both succeeded - return true.
   //3. Try to read image without transparency (mask not needed anymore).
   
   assert(gDisplay != 0 && "GetRootLogoAndShapeMask, gDisplay is None");

   gHasShapeExt = false;
   int eventBase = 0, errorBase = 0;

   gHasShapeExt = XShapeQueryExtension(gDisplay, &eventBase, &errorBase);
   
   gLogoPixmap = 0;
   gShapeMask = 0;
   
   if (gHasShapeExt) {
      if (!LoadROOTLogoPixmap("Root6SplashEXT.xpm", true)) {//true - mask is needed.
         gHasShapeExt = false;//We do not have a mask and can not call shape combine.
         LoadROOTLogoPixmap("Root6Splash.xpm", false);
      }
   } else
      LoadROOTLogoPixmap("Root6Splash.xpm", false);

   return gLogoPixmap;
}

//Text-rendering and animation.

//__________________________________________________________________
static void DrawVersion()
{
   // Draw version string.
   char version[80] = {};
#ifndef ROOT_RELEASE
   assert(0 && "DrawVersion, 'ROOT_RELEASE' is not defined");
   return;
#endif

   assert(strlen(ROOT_RELEASE) < sizeof version - 1 &&
          "DrawVersion, the string ROOT_RELEASE is too long");
   
/*   sprintf(version, "Version %s", ROOT_RELEASE);

   XDrawString(gDisplay, gLogoWindow, gGC, 15, gHeight - 15, version,
               strlen(version));*/
}

//Name DrawXXX is bad, actually this DrawXXX instead of drawing XXX can just
//calculate a height for XXX only, without actual drawing.

//__________________________________________________________________
static int DrawCreditItem(const char *creditItem, const char **members, int y, bool draw)
{
   // Draw credit item.
   assert(creditItem != 0 && "DrawCreditItem, parameter 'creditItem' is null");

   assert(gFont != 0 && "DrawCreditItem, gFont is None");
   assert(gDisplay != 0 && "DrawCreditItem, gDisplay is None");
   
   const int lineSpacing = gFont->max_bounds.ascent + gFont->max_bounds.descent;
   assert(lineSpacing > 0 && "DrawCreditItem, lineSpacing must be positive");
   
   std::string credit(creditItem);
   
   for (unsigned i = 0; members && members[i]; ++i) {
      if (i)
         credit += ", ";

      if (XTextWidth(gFont, credit.data(), credit.length()) +
          XTextWidth(gFont, members[i], strlen(members[i])) > (int) gCreditsWidth)
      {
         if (draw) {
            XDrawString(gDisplay, gCreditsPixmap, gGC, 0, y,
                        credit.data(), credit.length());
         }
         
         y += lineSpacing;
         credit = "   ";
      }
      
      credit += members[i];
   }

   if (draw) {
      XDrawString(gDisplay, gCreditsPixmap, gGC, 0, y,
                  credit.data(), credit.length());
   }

   return y;
}

//__________________________________________________________________
static int DrawCredits(bool draw, bool extended)
{
   // Draw credits. If draw is true draw credits,
   // otherwise just return size of all credit text.
   // If extended is true draw or returns size for extended full
   // credits list.

   if (!gFont || !gGC)
      return 150;  // size not important no text will be drawn anyway
   
   XSetForeground(gDisplay, gGC, gBackground);//Text is white!

   assert((draw == false || gCreditsPixmap != 0) &&
          "DrawCredits, parameter 'draw' is true, but destination pixmap is None");
   const int lineSpacing = gFont->max_bounds.ascent + gFont->max_bounds.descent;
   assert(lineSpacing > 0 && "DrawCredits, lineSpacing must be positive");
   
   int y = lineSpacing;
   y = DrawCreditItem("Conception: ", gConception, y, draw);
   y += 2 * lineSpacing;

   y = DrawCreditItem("Lead Developers: ", gLeadDevelopers, y, draw);
   y += 2 * lineSpacing - 1;  // special layout tweak ... WUT????

   y = DrawCreditItem("Core Engineering: ", gRootDevelopers, y, draw);
   y += 2 * lineSpacing;

   y = DrawCreditItem("Documentation: ", gRootDocumentation, y, draw);

   if (extended && gContributors) {
      y += 2 * lineSpacing;
      y = DrawCreditItem("Contributors: ", (const char **)gContributors, y, draw);

      y += 2 * lineSpacing;
      y = DrawCreditItem("Our sincere thanks and apologies to anyone who deserves", 0, y, draw);
      y += lineSpacing;
      y = DrawCreditItem("credit but fails to appear in this list.", 0, y, draw);

      struct passwd *pwd = getpwuid(getuid());
      if (pwd) {
         char *name = new char [strlen(pwd->pw_gecos)+1];
         strcpy(name, pwd->pw_gecos);
         char *s = strchr(name, ',');
         if (s) *s = 0;
         char line[1024];
         if (strlen(name))
            snprintf(line, sizeof(line), "Extra special thanks go to %s,", name);
         else
            snprintf(line, sizeof(line), "Extra special thanks go to %s,", pwd->pw_name);
         delete [] name;
         y += 2*lineSpacing;
         y = DrawCreditItem(line, 0, y, draw);
         y += lineSpacing;
         y = DrawCreditItem("one of our favorite users.", 0, y, draw);
      }
   }
   
   XSetForeground(gDisplay, gGC, gForeground);

   return y;
}

//__________________________________________________________________
static void CreateTextPixmap()
{
   assert(gDisplay != 0 && "CreateTextPixmap, gDisplay is None");
   assert(gLogoWindow != 0 && "CreateTextPixmap, gLogoWindow is None");
   
   if (!gGC)//Something is wrong and we don't need pixmap anymore.
      return;

   Window rootWindow = Window();
   int x = 0, y = 0;
   unsigned int borderWidth = 0, depth = 0;
   unsigned int width = 0, height = 0;
   XGetGeometry(gDisplay, gLogoWindow, &rootWindow, &x, &y, &width, &height, &borderWidth, &depth);

   gCreditsHeight = DrawCredits(false, gAbout) + gCreditsRect.height + 50;
   gCreditsPixmap = XCreatePixmap(gDisplay, gLogoWindow, gCreditsWidth, gCreditsHeight, depth);

   if (gHasShapeExt)
      gCreditsRect.x = 115;
   else
      gCreditsRect.x = 15;
   
   assert(gHeight > 105 && "CreateTextPixmap, internal error - unexpected geometry");
   gCreditsRect.y = gHeight - 105;
}


//__________________________________________________________________
void ScrollCredits(int ypos)
{
   assert(gDisplay != 0 && "ScrollCredits, gDisplay is None");

   if (!gGC || !gLogoPixmap || !gCreditsPixmap)
      return;

   XCopyArea(gDisplay, gLogoPixmap, gCreditsPixmap, gGC, gCreditsRect.x, gCreditsRect.y,
             gCreditsRect.width, gCreditsRect.height, 0, ypos);

   DrawCredits(true, true);

   XRectangle crect[1];
   crect[0] = gCreditsRect;
   XSetClipRectangles(gDisplay, gGC, 0, 0, crect, 1, Unsorted);

   XCopyArea(gDisplay, gCreditsPixmap, gLogoWindow, gGC,
             0, ypos, gCreditsWidth, gCreditsHeight, gCreditsRect.x, gCreditsRect.y);

   XSetClipMask(gDisplay, gGC, None);
}


//Aux. non-GUI function.

//__________________________________________________________________
static void ReadContributors()
{
   // Read the file $ROOTSYS/README/CREDITS for the names of the
   // contributors.

   char buf[2048];
#ifdef ROOTDOCDIR
   snprintf(buf, sizeof(buf), "%s/CREDITS", ROOTDOCDIR);
#else
   snprintf(buf, sizeof(buf), "%s/README/CREDITS", getenv("ROOTSYS"));
#endif

   gContributors = 0;

   FILE *f = fopen(buf, "r");
   if (!f) return;

   int cnt = 0;
   while (fgets(buf, sizeof(buf), f)) {
      if (!strncmp(buf, "N: ", 3)) {
         cnt++;
      }
   }
   gContributors = new char*[cnt+1];

   cnt = 0;
   rewind(f);
   while (fgets(buf, sizeof(buf), f)) {
      if (!strncmp(buf, "N: ", 3)) {
         int len = strlen(buf);
         buf[len-1] = 0;    // remove \n
         len -= 3;          // remove "N: "
         gContributors[cnt] = new char[len];
         strncpy(gContributors[cnt], buf+3, len);
         cnt++;
      }
   }
   gContributors[cnt] = 0;

   fclose(f);
}

//__________________________________________________________________
void PopupLogo(bool about)
{
   // Popup logo, waiting till ROOT is ready to run.

   //Initialize and check what we can do:
   gDisplay = XOpenDisplay("");
   if (!gDisplay) {
      printf("rootx - XOpenDisplay failed\n");
      return;
   }

   //Create a window.
   if (!CreateSplashscreenWindow()) {
      printf("rootx - CreateSplashscreenWindow failed\n");
      XCloseDisplay(gDisplay);
      gDisplay = 0;
      return;
   }

   if (!GetRootLogoAndShapeMask()) {
      printf("rootx - failed to create a background pixmap\n");
      XDestroyWindow(gDisplay, gLogoWindow);
      XCloseDisplay(gDisplay);
      gDisplay = 0;
      return;
   }

   gAbout = about;

   SetBackgroundPixmapAndMask();
   SetSplashscreenPosition();
   SelectFontAndTextColor();

   if (gAbout)
      ReadContributors();


   CreateTextPixmap();
   if (gCreditsPixmap)
      ScrollCredits(0);

   XSelectInput(gDisplay, gLogoWindow, ButtonPressMask | ExposureMask);
   XMapRaised(gDisplay, gLogoWindow);
   
   gettimeofday(&gPopupTime, 0);
}

//__________________________________________________________________
void WaitLogo()
{
   // Main event loop waiting till time arrives to pop down logo
   // or when forced by button press event.

   if (!gDisplay || !gLogoWindow)
      return;

   int ypos = 0;
   bool stopScroll = false;

   ScrollCredits(ypos);
   DrawVersion();
   XFlush(gDisplay);

   while (!gDone) {
      XEvent event = {};
      while (XCheckMaskEvent(gDisplay, ButtonPressMask | ExposureMask, &event));
      switch (event.type) {
         case Expose:
            if (event.xexpose.count == 0) {
               ScrollCredits(ypos);
               DrawVersion();
            }
            break;
         case ButtonPress:
            if (gAbout && event.xbutton.button == 3)
               stopScroll = stopScroll ? false : true;
            else
               gDone = true;
            break;
         default:
            break;
      }

      Sleep(100);

      if (!gAbout && !StayUp(gStayUp) && gMayPopdown)
         gDone = true;

      if (gAbout && !stopScroll) {
         if (ypos == 0) Sleep(2000);
         ypos++;
         if (ypos > (int) (gCreditsHeight - gCreditsRect.height - 50))
            ypos = -int(gCreditsRect.height);
         ScrollCredits(ypos);
         XFlush(gDisplay);
      }
   }

   if (gLogoWindow) {
      XUnmapWindow(gDisplay, gLogoWindow);
      XDestroyWindow(gDisplay, gLogoWindow);
      gLogoWindow = 0;
   }
   if (gLogoPixmap) {
      XFreePixmap(gDisplay, gLogoPixmap);
      gLogoPixmap = 0;
   }
   if (gShapeMask) {
      XFreePixmap(gDisplay, gShapeMask);
      gShapeMask = 0;
   }
   if (gCreditsPixmap) {
      XFreePixmap(gDisplay, gCreditsPixmap);
      gCreditsPixmap = 0;
   }
   if (gFont) {
      XFreeFont(gDisplay, gFont);
      gFont = 0;
   }
   if (gGC) {
      XFreeGC(gDisplay, gGC);
      gGC = 0;
   }
   if (gDisplay) {
      XSync(gDisplay, False);
      XCloseDisplay(gDisplay);
      gDisplay = 0;
   }
}

//__________________________________________________________________
void PopdownLogo()
{
   // ROOT is ready to run, may pop down the logo if stay up time expires.

   gMayPopdown = true;
}

//__________________________________________________________________
void CloseDisplay()
{
   // Close connection to X server (called by child).

   if (gDisplay)
      close(ConnectionNumber(gDisplay));
}
