// @(#)root/rootx:$Id$
// Author: Fons Rademakers   19/02/98

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rootxx                                                               //
//                                                                      //
// X11 based routines used to display the splash screen for rootx,      //
// the root front-end program.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <pwd.h>
#include <sys/types.h>
#include <X11/Xlib.h>
#include <X11/xpm.h>

#include "Rtypes.h"

#if defined(R__AIX) || defined(R__SOLARIS)
#   include <sys/select.h>
#endif
#include <time.h>
#include <sys/time.h>

static Display     *gDisplay       = 0;
static Window       gLogoWindow    = 0;
static Pixmap       gLogoPixmap    = 0;
static Pixmap       gCreditsPixmap = 0;
static GC           gGC            = 0;
static XFontStruct *gFont          = 0;
static bool         gDone          = false;
static bool         gMayPopdown    = false;
static bool         gAbout         = false;
static unsigned int gWidth         = 0;
static unsigned int gHeight        = 0;
static int          gStayUp        = 4000;   // 4 seconds
static XRectangle   gCreditsRect   = { 15, 155, 285, 130 }; // clip rect in logo
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

static const char *gCintDevelopers[] = {
   "Masaharu Goto",
   0
};

static const char *gRootDocumentation[] = {
   "Ilka Antcheva",
   0
};

static char **gContributors = 0;



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

static void Sleep(int milliSec)
{
   // Sleep for specified amount of milli seconds.

   // get current time
   struct timeval tv;

   tv.tv_sec  = milliSec / 1000;
   tv.tv_usec = (milliSec % 1000) * 1000;

   select(0, 0, 0, 0, &tv);
}

static Pixmap GetRootLogo()
{
   // Get logo from xpm file.

   Pixmap logo = 0;
   Screen *xscreen = XDefaultScreenOfDisplay(gDisplay);
   if (!xscreen) return logo;
   int depth = PlanesOfScreen(xscreen);

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

   char file[2048];
#ifdef ROOTICONPATH
   snprintf(file, sizeof(file), "%s/Splash.xpm", ROOTICONPATH);
#else
   snprintf(file, sizeof(file), "%s/icons/Splash.xpm", getenv("ROOTSYS"));
#endif
   int ret = XpmReadFileToPixmap(gDisplay, gLogoWindow,
                                 file, &logo, 0, &attr);
   XpmFreeAttributes(&attr);

   if (ret == XpmSuccess || ret == XpmColorError)
      return logo;

   printf("rootx xpm error: %s\n", XpmGetErrorString(ret));

   if (logo) XFreePixmap(gDisplay, logo);
   logo = 0;

   return logo;
}

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

static void DrawVersion()
{
   // Draw version string.

   char version[80];
   sprintf(version, "Version %s", ROOT_RELEASE);

   XDrawString(gDisplay, gLogoWindow, gGC, 15, gHeight - 15, version,
               strlen(version));
}

static int DrawCreditItem(const char *creditItem, const char **members,
                          int y, bool draw)
{
   // Draw credit item.

   char credit[1024];
   int i;
   int lineSpacing = gFont->max_bounds.ascent + gFont->max_bounds.descent;

   strlcpy(credit, creditItem, sizeof(credit));
   for (i = 0; members && members[i]; i++) {
      if (i) strcat(credit, ", ");
      if (XTextWidth(gFont, credit, strlen(credit)) +
          XTextWidth(gFont, members[i], strlen(members[i])) > (int) gCreditsWidth) {
         if (draw)
            XDrawString(gDisplay, gCreditsPixmap, gGC, 0, y, credit, strlen(credit));
         y += lineSpacing;
         strcpy(credit, "   ");
      }
      strcat(credit, members[i]);
   }
   if (draw)
      XDrawString(gDisplay, gCreditsPixmap, gGC, 0, y, credit, strlen(credit));

   return y;
}

static int DrawCredits(bool draw, bool extended)
{
   // Draw credits. If draw is true draw credits,
   // otherwise just return size of all credit text.
   // If extended is true draw or returns size for extended full
   // credits list.

   if (!gFont) return 150;  // size not important no text will be drawn anyway

   int lineSpacing = gFont->max_bounds.ascent + gFont->max_bounds.descent;
   int y = lineSpacing;

   y = DrawCreditItem("Conception: ", gConception, y, draw);

   y += 2*lineSpacing;

   y = DrawCreditItem("Lead Developers: ", gLeadDevelopers, y, draw);

   y += 2*lineSpacing - 1;  // special layout tweak

   y = DrawCreditItem("Core Engineering: ", gRootDevelopers, y, draw);

   y += 2*lineSpacing - 1;  // to just not cut the bottom of the "p"

   y = DrawCreditItem("CINT C/C++ Intepreter: ", gCintDevelopers, y, draw);

   y += 2*lineSpacing;

   y = DrawCreditItem("Documentation: ", gRootDocumentation, y, draw);

   if (extended && gContributors) {
      y += 2*lineSpacing;
      y = DrawCreditItem("Contributors: ", (const char **)gContributors, y, draw);

      y += 2*lineSpacing;
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

   return y;
}

void ScrollCredits(int ypos)
{
   XRectangle crect[1];
   crect[0] = gCreditsRect;
   XSetClipRectangles(gDisplay, gGC, 0, 0, crect, 1, Unsorted);

   XCopyArea(gDisplay, gCreditsPixmap, gLogoWindow, gGC,
             0, ypos, gCreditsWidth, gCreditsHeight, gCreditsRect.x, gCreditsRect.y);

   XSetClipMask(gDisplay, gGC, None);
}

void PopupLogo(bool about)
{
   // Popup logo, waiting till ROOT is ready to run.

   gDisplay = XOpenDisplay("");
   if (!gDisplay) return;

   gAbout = about;

   Pixel back, fore;
   int screen = DefaultScreen(gDisplay);

   back = WhitePixel(gDisplay, screen);
   fore = BlackPixel(gDisplay, screen);

   gLogoWindow = XCreateSimpleWindow(gDisplay, DefaultRootWindow(gDisplay),
                                     -100, -100, 50, 50, 0, fore, back);

   gLogoPixmap = GetRootLogo();
   if (!gLogoPixmap) {
      XCloseDisplay(gDisplay);
      gDisplay = 0;
      return;
   }

   Window root;
   int x, y;
   unsigned int bw, depth;
   XGetGeometry(gDisplay, gLogoPixmap, &root, &x, &y, &gWidth, &gHeight,
                &bw, &depth);

   Screen *xscreen = XDefaultScreenOfDisplay(gDisplay);
   if (!xscreen) {
      XCloseDisplay(gDisplay);
      gDisplay = 0;
      return;
   }
   x = (WidthOfScreen(xscreen) - gWidth) / 2;
   y = (HeightOfScreen(xscreen) - gHeight) / 2;

   XMoveResizeWindow(gDisplay, gLogoWindow, x, y, gWidth, gHeight);
   XSync(gDisplay, False);   // make sure move & resize is done before mapping

   unsigned long valmask;
   XSetWindowAttributes xswa;
   valmask = CWBackPixmap | CWOverrideRedirect;
   xswa.background_pixmap = gLogoPixmap;
   xswa.override_redirect = True;
   XChangeWindowAttributes(gDisplay, gLogoWindow, valmask, &xswa);

   gGC = XCreateGC(gDisplay, gLogoWindow, 0, 0);
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
   XSetForeground(gDisplay, gGC, fore);
   XSetBackground(gDisplay, gGC, back);

   if (about)
      ReadContributors();

   gCreditsHeight = DrawCredits(false, about) + gCreditsRect.height + 50;
   gCreditsPixmap = XCreatePixmap(gDisplay, gLogoWindow, gCreditsWidth, gCreditsHeight, depth);
   XSetForeground(gDisplay, gGC, back);
   XFillRectangle(gDisplay, gCreditsPixmap, gGC, 0, 0, gCreditsWidth, gCreditsHeight);
   XSetForeground(gDisplay, gGC, fore);
   DrawCredits(true, about);

   XSelectInput(gDisplay, gLogoWindow, ButtonPressMask | ExposureMask);

   XMapRaised(gDisplay, gLogoWindow);

   gettimeofday(&gPopupTime, 0);
}

void WaitLogo()
{
   // Main event loop waiting till time arrives to pop down logo
   // or when forced by button press event.

   if (!gDisplay) return;

   int ypos = 0;
   bool stopScroll = false;

   ScrollCredits(ypos);
   DrawVersion();
   XFlush(gDisplay);

   while (!gDone) {
      XEvent event;
      if (XCheckMaskEvent(gDisplay, ButtonPressMask | ExposureMask, &event)) {
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

void PopdownLogo()
{
   // ROOT is ready to run, may pop down the logo if stay up time expires.

   gMayPopdown = true;
}

void CloseDisplay()
{
   // Close connection to X server (called by child).

   if (gDisplay)
      close(ConnectionNumber(gDisplay));
}
