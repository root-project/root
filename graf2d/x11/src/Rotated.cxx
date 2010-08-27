// @(#)root/x11:$Id$
// Author: O.Couet   17/11/93

#ifndef _XVERTEXT_INCLUDED_
#define _XVERTEXT_INCLUDED_

//______________________________________________________________________________
/* ********************************************************************** *
 *
 * xvertext 5.0, Copyright (c) 1993 Alan Richardson (mppa3@uk.ac.sussex.syma)
 *
 * Alignment definition modified by O.Couet.
 * Mods IBM/VM by O.Couet.
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted, provided
 * that the above copyright notice appear in all copies and that both the
 * copyright notice and this permission notice appear in supporting
 * documentation.  All work developed as a consequence of the use of
 * this program should duly acknowledge such use. No representations are
 * made about the suitability of this software for any purpose.  It is
 * provided "as is" without express or implied warranty.
 *
 * ********************************************************************** */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "TMath.h"

/* ************************************************************************ *
 *
 * Header file for the `xvertext 5.0' routines.
 *
 * Copyright (c) 1993 Alan Richardson (mppa3@uk.ac.sussex.syma)
 *
 * ************************************************************************ */

#define XV_VERSION      5.0
#define XV_COPYRIGHT \
      "xvertext routines Copyright (c) 1993 Alan Richardson"

/* ---------------------------------------------------------------------- */

/* text alignment */

#define NONE             0
#define TLEFT            1
#define TCENTRE          2
#define TRIGHT           3
#define MLEFT            4
#define MCENTRE          5
#define MRIGHT           6
#define BLEFT            7
#define BCENTRE          8
#define BRIGHT           9

#ifdef VAX
#define X11R3
#endif


#endif /* _XVERTEXT_INCLUDED_ */

/* ---------------------------------------------------------------------- */

/* Make sure cache size is set */

#ifndef CACHE_SIZE_LIMIT
#define CACHE_SIZE_LIMIT 0
#endif /*CACHE_SIZE_LIMIT */

/* Make sure a cache method is specified */

#ifndef CACHE_XIMAGES
#ifndef CACHE_BITMAPS
#define CACHE_BITMAPS
#endif /*CACHE_BITMAPS*/
#endif /*CACHE_XIMAGES*/

/* ---------------------------------------------------------------------- */

/* Debugging macros */

#ifdef DEBUG
static int gRotatedDebug=1;
#else
static int gRotatedDebug=0;
#endif /*DEBUG*/

#define DEBUG_PRINT1(a) if (gRotatedDebug) printf (a)
#define DEBUG_PRINT2(a, b) if (gRotatedDebug) printf (a, b)
#define DEBUG_PRINT3(a, b, c) if (gRotatedDebug) printf (a, b, c)
#define DEBUG_PRINT4(a, b, c, d) if (gRotatedDebug) printf (a, b, c, d)
#define DEBUG_PRINT5(a, b, c, d, e) if (gRotatedDebug) printf (a, b, c, d, e)

/* ---------------------------------------------------------------------- */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------- */

/* A structure holding everything needed for a rotated string */

typedef struct RotatedTextItemTemplate_t {
   Pixmap fBitmap;
   XImage *fXimage;

   char *fText;
   char *font_name;
   Font fid;
   float fAngle;
   int fAlign;
   float fMagnify;

   int fColsIn;
   int fRowsIn;
   int fColsOut;
   int fRowsOut;

   int fNl;
   int fMaxWidth;
   float *fCornersX;
   float *fCornersY;

   long int fSize;
   int fCached;

   struct RotatedTextItemTemplate_t *fNext;
} RotatedTextItem_t;

static RotatedTextItem_t *gFirstTextItem=0;

/* ---------------------------------------------------------------------- */

/* A structure holding current magnification and bounding box padding */

static struct StyleTemplate_t {
   float fMagnify;
   int fBbxPadl;
} gRotStyle={
   1.,
   0
   };

/* ---------------------------------------------------------------------- */
static char              *my_strdup(char *);
static char              *my_strtok(char *, char *);

float                     XRotVersion(char*, int);
void                      XRotSetMagnification(float);
void                      XRotSetBoundingBoxPad(int);
int                       XRotDrawString(Display*, XFontStruct*, float,Drawable, GC, int, int, char*);
int                       XRotDrawImageString(Display*, XFontStruct*, float,Drawable, GC, int, int, char*);
int                       XRotDrawAlignedString(Display*, XFontStruct*, float,Drawable, GC, int, int, char*, int);
int                       XRotDrawAlignedImageString(Display*, XFontStruct*, float,Drawable, GC, int, int, char*, int);
XPoint                   *XRotTextExtents(Display*, XFontStruct*, float,int, int, char*, int);

static XImage            *MakeXImage(Display *dpy,int  w, int h);
static int                XRotPaintAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x,int y, char *text,int align, int bg);
static int                XRotDrawHorizontalString(Display *dpy, XFontStruct *font, Drawable drawable, GC gc, int x, int y, char *text,int align, int bg);
static RotatedTextItem_t *XRotRetrieveFromCache(Display *dpy, XFontStruct *font, float angle, char *text, int align);
static RotatedTextItem_t *XRotCreateTextItem(Display *dpy, XFontStruct *font, float angle, char *text, int align);
static void               XRotAddToLinkedList(Display *dpy, RotatedTextItem_t *item);
static void               XRotFreeTextItem(Display *dpy, RotatedTextItem_t *item);
static XImage            *XRotMagnifyImage(Display *dpy, XImage *ximage);


//______________________________________________________________________________
static char *my_strdup(char *str)
{
   // Routine to mimic `strdup()' (some machines don't have it)

   char *s;

   if(str==0) return 0;

   s=(char *)malloc((unsigned)(strlen(str)+1));
   if(s!=0) strcpy(s, str);

   return s;
}

//______________________________________________________________________________
static char *my_strtok(char *str1, char *str2)
{
   // Routine to replace `strtok' : this one returns a zero length string if
   // it encounters two consecutive delimiters

   char *ret;
   int i, j, stop;
   static int start, len;
   static char *stext;

   if(str2==0) return 0;

   /* initialise if str1 not 0 */
   if(str1!=0) {
      start=0;
      stext=str1;
      len=strlen(str1);
   }

   /* run out of tokens ? */
   if(start>=len) return 0;

   /* loop through characters */
   for(i=start; i<len; i++) {
      /* loop through delimiters */
      stop=0;
      for(j=0; j<(int)strlen(str2); j++)
      if(stext[i]==str2[j])
      stop=1;

      if(stop) break;
   }

   stext[i]='\0';

   ret=stext+start;

   start=i+1;

   return ret;
}


//______________________________________________________________________________
float XRotVersion(char *str,int n)
{
   // Return version/copyright information

   if(str!=0) strncpy(str, XV_COPYRIGHT, n);
   return XV_VERSION;
}


//______________________________________________________________________________
void XRotSetMagnification(float m)
{
   // Set the font magnification factor for all subsequent operations

   if(m>0.) gRotStyle.fMagnify=m;
}


//______________________________________________________________________________
void XRotSetBoundingBoxPad(int p)
{
   // Set the padding used when calculating bounding boxes

   if(p>=0) gRotStyle.fBbxPadl=p;
}


//______________________________________________________________________________
static XImage *MakeXImage(Display *dpy,int  w, int h)
{
   // Create an XImage structure and allocate memory for it

   XImage *image;
   char *data;

   /* reserve memory for image */
   data=(char *)calloc((unsigned)(((w-1)/8+1)*h), 1);
   if(data==0) return 0;

   /* create the XImage */
   image=XCreateImage(dpy, DefaultVisual(dpy, DefaultScreen(dpy)), 1, XYBitmap,
                   0, data, w, h, 8, 0);
   if(image==0) return 0;

   image->byte_order=image->bitmap_bit_order=MSBFirst;
   return image;
}


//______________________________________________________________________________
int XRotDrawString(Display *dpy, XFontStruct *font,float angle,Drawable drawable, GC gc, int x, int y, char *str)
{
   // A front end to XRotPaintAlignedString:
   //     -no alignment, no background

   return (XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                  x, y, str, NONE, 0));
}


//______________________________________________________________________________
int XRotDrawImageString(Display *dpy,XFontStruct *font, float angle, Drawable drawable,GC  gc, int x, int y, char *str)
{
   // A front end to XRotPaintAlignedString:
   //     -no alignment, paints background

   return(XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                 x, y, str, NONE, 1));
}


//______________________________________________________________________________
int XRotDrawAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text,int align)
{
   // A front end to XRotPaintAlignedString:
   //     -does alignment, no background

   return(XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                 x, y, text, align, 0));
}


//______________________________________________________________________________
int XRotDrawAlignedImageString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text,
                               int align)
{
   // A front end to XRotPaintAlignedString:
   //     -does alignment, paints background

   return(XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                 x, y, text, align, 1));
}


//______________________________________________________________________________
static int XRotPaintAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text,
                                  int align, int bg)
{
   // Aligns and paints a rotated string

   int i;
   GC my_gc;
   int xp, yp;
   float hot_x, hot_y;
   float hot_xp, hot_yp;
   float sin_angle, cos_angle;
   RotatedTextItem_t *item;
   Pixmap bitmap_to_paint;

   /* return early for 0/empty strings */
   if(text==0) return 0;

   if(strlen(text)==0) return 0;

   /* manipulate angle to 0<=angle<360 degrees */
   while(angle<0) angle+=360;

   while(angle>=360) angle-=360;

   angle*=M_PI/180;

   /* horizontal text made easy */
   if(angle==0. && gRotStyle.fMagnify==1.)
      return(XRotDrawHorizontalString(dpy, font, drawable, gc, x, y,
                                      text, align, bg));

   /* get a rotated bitmap */
   item=XRotRetrieveFromCache(dpy, font, angle, text, align);
   if(item==0) return 0;

   /* this gc has similar properties to the user's gc */
   my_gc=XCreateGC(dpy, drawable, 0, 0);
   XCopyGC(dpy, gc, GCForeground|GCBackground|GCFunction|GCPlaneMask,
           my_gc);

   /* alignment : which point (hot_x, hot_y) relative to bitmap centre
      coincides with user's specified point? */

   /* y position */
   if(align==TLEFT || align==TCENTRE || align==TRIGHT)
      hot_y=(float)item->fRowsIn/2*gRotStyle.fMagnify;
   else if(align==MLEFT || align==MCENTRE || align==MRIGHT)
   {
      /*  Modify by O.Couet to have Bottom alignment without font->descent */
      hot_y=0;
      /*    hot_y=-((float)item->fRowsIn/4-(float)font->descent)*gRotStyle.fMagnify; */
   }
   else if(align==BLEFT || align==BCENTRE || align==BRIGHT)
   {
      /*  Modify by O.Couet to have Bottom alignment without font->descent */
      /*  hot_y=-(float)item->fRowsIn/2*gRotStyle.fMagnify; */
      hot_y=-((float)item->fRowsIn/2-(float)font->descent)*gRotStyle.fMagnify;
   }
   else
      hot_y=-((float)item->fRowsIn/2-(float)font->descent)*gRotStyle.fMagnify;

   /* x position */
   if(align==TLEFT || align==MLEFT || align==BLEFT || align==NONE)
      hot_x=-(float)item->fMaxWidth/2*gRotStyle.fMagnify;
   else if(align==TCENTRE || align==MCENTRE || align==BCENTRE)
      hot_x=0;
   else
      hot_x=(float)item->fMaxWidth/2*gRotStyle.fMagnify;

   /* pre-calculate sin and cos */
   sin_angle=TMath::Sin(angle);
   cos_angle=TMath::Cos(angle);

   /* rotate hot_x and hot_y around bitmap centre */
   hot_xp= hot_x*cos_angle - hot_y*sin_angle;
   hot_yp= hot_x*sin_angle + hot_y*cos_angle;

   /* text background will be drawn using XFillPolygon */
   if(bg) {
      GC depth_one_gc;
      XPoint *xpoints;
      Pixmap empty_stipple;

      /* reserve space for XPoints */
      xpoints=(XPoint *)malloc((unsigned)(4*item->fNl*sizeof(XPoint)));
      if(!xpoints) return 1;

      /* rotate corner positions */
      for(i=0; i<4*item->fNl; i++) {
         xpoints[i].x=int((float)x + ( (item->fCornersX[i]-hot_x)*cos_angle +
                                   (item->fCornersY[i]+hot_y)*sin_angle));
         xpoints[i].y=int((float)y + (-(item->fCornersX[i]-hot_x)*sin_angle +
                                   (item->fCornersY[i]+hot_y)*cos_angle));
      }

      /* we want to swap foreground and background colors here;
         XGetGCValues() is only available in R4+ */

      empty_stipple=XCreatePixmap(dpy, drawable, 1, 1, 1);

      depth_one_gc=XCreateGC(dpy, empty_stipple, 0, 0);
      XSetForeground(dpy, depth_one_gc, 0);
      XFillRectangle(dpy, empty_stipple, depth_one_gc, 0, 0, 2, 2);

      XSetStipple(dpy, my_gc, empty_stipple);
      XSetFillStyle(dpy, my_gc, FillOpaqueStippled);

      XFillPolygon(dpy, drawable, my_gc, xpoints, 4*item->fNl, Nonconvex,
                   CoordModeOrigin);

      /* free our resources */
      free((char *)xpoints);
      XFreeGC(dpy, depth_one_gc);
      XFreePixmap(dpy, empty_stipple);
   }

   /* where should top left corner of bitmap go ? */
   xp=int((float)x-((float)item->fColsOut/2 +hot_xp));
   yp=int((float)y-((float)item->fRowsOut/2 -hot_yp));

   /* by default we draw the rotated bitmap, solid */
   bitmap_to_paint=item->fBitmap;

    /* handle user stippling */
#ifndef X11R3
   {
      GC depth_one_gc;
      XGCValues values;
      Pixmap new_bitmap, inverse;

      /* try and get some GC properties */
      if(XGetGCValues(dpy, gc,
                      GCStipple|GCFillStyle|GCForeground|GCBackground|
                      GCTileStipXOrigin|GCTileStipYOrigin,
                      &values)) {

         /* only do this if stippling requested */
         if((values.fill_style==FillStippled ||
             values.fill_style==FillOpaqueStippled) && !bg) {

            /* opaque stipple: draw rotated text in background colour */
            if(values.fill_style==FillOpaqueStippled) {
               XSetForeground(dpy, my_gc, values.background);
               XSetFillStyle(dpy, my_gc, FillStippled);
               XSetStipple(dpy, my_gc, item->fBitmap);
               XSetTSOrigin(dpy, my_gc, xp, yp);
               XFillRectangle(dpy, drawable, my_gc, xp, yp,
                              item->fColsOut, item->fRowsOut);
               XSetForeground(dpy, my_gc, values.foreground);
            }

            /* this will merge the rotated text and the user's stipple */
            new_bitmap=XCreatePixmap(dpy, drawable,
                                     item->fColsOut, item->fRowsOut, 1);

            /* create a GC */
            depth_one_gc=XCreateGC(dpy, new_bitmap, 0, 0);
            XSetForeground(dpy, depth_one_gc, 1);
            XSetBackground(dpy, depth_one_gc, 0);

            /* set the relative stipple origin */
            XSetTSOrigin(dpy, depth_one_gc,
                         values.ts_x_origin-xp, values.ts_y_origin-yp);

            /* fill the whole bitmap with the user's stipple */
            XSetStipple(dpy, depth_one_gc, values.stipple);
            XSetFillStyle(dpy, depth_one_gc, FillOpaqueStippled);
            XFillRectangle(dpy, new_bitmap, depth_one_gc,
                           0, 0, item->fColsOut, item->fRowsOut);

            /* set stipple origin back to normal */
            XSetTSOrigin(dpy, depth_one_gc, 0, 0);

            /* this will contain an inverse copy of the rotated text */
            inverse=XCreatePixmap(dpy, drawable,
                                  item->fColsOut, item->fRowsOut, 1);

            /* invert text */
            XSetFillStyle(dpy, depth_one_gc, FillSolid);
            XSetFunction(dpy, depth_one_gc, GXcopyInverted);
            XCopyArea(dpy, item->fBitmap, inverse, depth_one_gc,
                      0, 0, item->fColsOut, item->fRowsOut, 0, 0);

            /* now delete user's stipple everywhere EXCEPT on text */
            XSetForeground(dpy, depth_one_gc, 0);
            XSetBackground(dpy, depth_one_gc, 1);
            XSetStipple(dpy, depth_one_gc, inverse);
            XSetFillStyle(dpy, depth_one_gc, FillStippled);
            XSetFunction(dpy, depth_one_gc, GXcopy);
            XFillRectangle(dpy, new_bitmap, depth_one_gc,
                           0, 0, item->fColsOut, item->fRowsOut);

            /* free resources */
            XFreePixmap(dpy, inverse);
            XFreeGC(dpy, depth_one_gc);

            /* this is the new bitmap */
            bitmap_to_paint=new_bitmap;
         }
      }
   }
#endif /*X11R3*/

   /* paint text using stipple technique */
   XSetFillStyle(dpy, my_gc, FillStippled);
   XSetStipple(dpy, my_gc, bitmap_to_paint);
   XSetTSOrigin(dpy, my_gc, xp, yp);
   XFillRectangle(dpy, drawable, my_gc, xp, yp,
                  item->fColsOut, item->fRowsOut);

   /* free our resources */
   XFreeGC(dpy, my_gc);

   /* stippled bitmap no longer needed */
   if(bitmap_to_paint!=item->fBitmap)
      XFreePixmap(dpy, bitmap_to_paint);

#ifdef CACHE_XIMAGES
   XFreePixmap(dpy, item->fBitmap);
#endif /*CACHE_XIMAGES*/

   /* if item isn't cached, destroy it completely */
   if(!item->fCached)
      XRotFreeTextItem(dpy,item);

   /* we got to the end OK! */
   return 0;
}


//______________________________________________________________________________
static int XRotDrawHorizontalString(Display *dpy, XFontStruct *font, Drawable drawable, GC gc, int x, int y, char *text,
                                    int align, int bg)
{
   //  Draw a horizontal string in a quick fashion

   GC my_gc;
   int nl=1, i;
   int height;
   int xp, yp;
   char *str1, *str2, *str3;
   const char *str2_a="\0", *str2_b="\n\0";
   int dir, asc, desc;
   XCharStruct overall;

   DEBUG_PRINT1("**\nHorizontal text.\n");

   /* this gc has similar properties to the user's gc (including stipple) */
   my_gc=XCreateGC(dpy, drawable, 0, 0);
   XCopyGC(dpy, gc,
           GCForeground|GCBackground|GCFunction|GCStipple|GCFillStyle|
           GCTileStipXOrigin|GCTileStipYOrigin|GCPlaneMask, my_gc);
   XSetFont(dpy, my_gc, font->fid);

   /* count number of sections in string */
   if(align!=NONE)
      for(i=0; i<(int)strlen(text)-1; i++)
         if(text[i]=='\n')
            nl++;

   /* ignore newline characters if not doing alignment */
   if(align==NONE)
      str2=(char *)str2_a;
   else
      str2=(char *)str2_b;

   /* overall font height */
   height=font->ascent+font->descent;

   /* y position */
   if(align==TLEFT || align==TCENTRE || align==TRIGHT)
      yp=y+font->ascent;
   else if(align==MLEFT || align==MCENTRE || align==MRIGHT)
   {
      /*  Modify by O.Couet to have Middle alignment without font->descent */
      /*  yp=y-nl*height/2+font->ascent; */
      yp=y-nl*(height-font->descent)/2+font->ascent;
   }
   else if(align==BLEFT || align==BCENTRE || align==BRIGHT)
   {
      /*  Modify by O.Couet to have Bottom alignment without font->descent */
      /*  yp=y-nl*height+font->ascent; */
      yp=y-nl*(height-font->descent)+font->ascent;
   }
   else
      yp=y;

   str1=my_strdup(text);
   if(str1==0) return 1;

   str3=my_strtok(str1, str2);

   /* loop through each section in the string */
   do {
      XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                   &overall);

      /* where to draw section in x ? */
      if(align==TLEFT || align==MLEFT || align==BLEFT || align==NONE)
         xp=x;
      else if(align==TCENTRE || align==MCENTRE || align==BCENTRE)
         xp=x-overall.rbearing/2;
      else
         xp=x-overall.rbearing;

      /* draw string onto bitmap */
      if(!bg)
         XDrawString(dpy, drawable, my_gc, xp, yp, str3, strlen(str3));
      else
         XDrawImageString(dpy, drawable, my_gc, xp, yp, str3, strlen(str3));

      /* move to next line */
      yp+=height;

      str3=my_strtok((char *)0, str2);
   }
   while(str3!=0);

   free(str1);
   XFreeGC(dpy, my_gc);

   return 0;
}


//______________________________________________________________________________
static RotatedTextItem_t *XRotRetrieveFromCache(Display *dpy, XFontStruct *font, float angle, char *text, int align)
{
   // Query cache for a match with this font/text/angle/alignment
   //    request, otherwise arrange for its creation

   Font fid;
   char *font_name;
   unsigned long name_value;
   RotatedTextItem_t *item=0;
   RotatedTextItem_t *i1=gFirstTextItem;

   /* get font name, if it exists */
   if(XGetFontProperty(font, XA_FONT, &name_value)) {
      DEBUG_PRINT1("got font name OK\n");
      font_name=XGetAtomName(dpy, name_value);
      fid=0;
   }
#ifdef CACHE_FID
   /* otherwise rely (unreliably?) on font ID */
   else {
      DEBUG_PRINT1("can't get fontname, caching FID\n");
      font_name=0;
      fid=font->fid;
   }
#else
   /* not allowed to cache font ID's */
   else {
      DEBUG_PRINT1("can't get fontname, can't cache\n");
      font_name=0;
      fid=0;
   }
#endif /*CACHE_FID*/

   /* look for a match in cache */

   /* matching formula:
      identical text;
      identical fontname (if defined, font ID's if not);
      angles close enough (<0.00001 here, could be smaller);
      HORIZONTAL alignment matches, OR it's a one line string;
      magnifications the same */

   while(i1 && !item) {
      /* match everything EXCEPT fontname/ID */
      if(strcmp(text, i1->fText)==0 &&
         TMath::Abs(angle-i1->fAngle)<0.00001 &&
         gRotStyle.fMagnify==i1->fMagnify &&
         (i1->fNl==1 ||
         ((align==0)?9:(align-1))%3==
         ((i1->fAlign==0)?9:(i1->fAlign-1))%3)) {

         /* now match fontname/ID */
         if(font_name!=0 && i1->font_name!=0) {
            if(strcmp(font_name, i1->font_name)==0) {
               item=i1;
               DEBUG_PRINT1("Matched against font names\n");
            }
            else
               i1=i1->fNext;
         }
#ifdef CACHE_FID
         else if(font_name==0 && i1->font_name==0) {
            if(fid==i1->fid) {
               item=i1;
               DEBUG_PRINT1("Matched against FID's\n");
            }
            else
               i1=i1->fNext;
         }
#endif /*CACHE_FID*/
         else
            i1=i1->fNext;
      }
      else
         i1=i1->fNext;
   }

   if(item)
      DEBUG_PRINT1("**\nFound target in cache.\n");
   if(!item)
      DEBUG_PRINT1("**\nNo match in cache.\n");

   /* no match */
   if(!item) {
      /* create new item */
      item=XRotCreateTextItem(dpy, font, angle, text, align);
      if(!item)
         return 0;

      /* record what it shows */
      item->fText=my_strdup(text);

      /* fontname or ID */
      if(font_name!=0) {
         item->font_name=my_strdup(font_name);
         item->fid=0;
      }
      else {
         item->font_name=0;
         item->fid=fid;
      }

      item->fAngle=angle;
      item->fAlign=align;
      item->fMagnify=gRotStyle.fMagnify;

      /* cache it */
      XRotAddToLinkedList(dpy, item);
   }

   if(font_name)
      XFree(font_name);

   /* if XImage is cached, need to recreate the bitmap */

#ifdef CACHE_XIMAGES
   {
      GC depth_one_gc;

      /* create bitmap to hold rotated text */
      item->fBitmap=XCreatePixmap(dpy, DefaultRootWindow(dpy),
                                  item->fColsOut, item->fRowsOut, 1);

      /* depth one gc */
      depth_one_gc=XCreateGC(dpy, item->fBitmap, 0, 0);
      XSetBackground(dpy, depth_one_gc, 0);
      XSetForeground(dpy, depth_one_gc, 1);

      /* make the text bitmap from XImage */
      XPutImage(dpy, item->fBitmap, depth_one_gc, item->fXimage, 0, 0, 0, 0,
                item->fColsOut, item->fRowsOut);

      XFreeGC(dpy, depth_one_gc);
   }
#endif /*CACHE_XIMAGES*/

   return item;
}


//______________________________________________________________________________
static RotatedTextItem_t *XRotCreateTextItem(Display *dpy, XFontStruct *font, float angle, char *text, int align)
{
   //  Create a rotated text item

   RotatedTextItem_t *item;
   Pixmap canvas;
   GC font_gc;
   XImage *imageIn;
   register int i, j;
   char *str1, *str2, *str3;
   const char *str2_a="\0", *str2_b="\n\0";
   int height;
   int byte_w_in, byte_w_out;
   int xp, yp;
   float sin_angle, cos_angle;
   int it, jt;
   float di, dj;
   int ic=0;
   float xl, xr, xinc;
   int byte_out;
   int dir, asc, desc;
   XCharStruct overall;
   int old_cols_in=0, old_rows_in=0;

   /* allocate memory */
   item=(RotatedTextItem_t *)malloc((unsigned)sizeof(RotatedTextItem_t));
   if(!item) return 0;

   /* count number of sections in string */
   item->fNl=1;
   if(align!=NONE)
      for(i=0; i<(int)strlen(text)-1; i++)
         if(text[i]=='\n')
            item->fNl++;

   /* ignore newline characters if not doing alignment */
   if(align==NONE)
      str2=(char *)str2_a;
   else
      str2=(char *)str2_b;

   /* find width of longest section */
   str1=my_strdup(text);
   if(str1==0) {
      free(item);
      return 0;
   }

   str3=my_strtok(str1, str2);

   XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                &overall);

   item->fMaxWidth=overall.rbearing;

   /* loop through each section */
   do {
      str3=my_strtok((char *)0, str2);

      if(str3!=0) {
         XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                      &overall);
         if(overall.rbearing>item->fMaxWidth)
            item->fMaxWidth=overall.rbearing;
      }
   }
   while(str3!=0);

   free(str1);

   /* overall font height */
   height=font->ascent+font->descent;

   /* dimensions horizontal text will have */
   item->fColsIn=item->fMaxWidth;
   item->fRowsIn=item->fNl*height;

   /* bitmap for drawing on */
   canvas=XCreatePixmap(dpy, DefaultRootWindow(dpy),
                         item->fColsIn, item->fRowsIn, 1);

   /* create a GC for the bitmap */
   font_gc=XCreateGC(dpy, canvas, 0, 0);
   XSetBackground(dpy, font_gc, 0);
   XSetFont(dpy, font_gc, font->fid);

   /* make sure the bitmap is blank */
   XSetForeground(dpy, font_gc, 0);
   XFillRectangle(dpy, canvas, font_gc, 0, 0,
                  item->fColsIn+1, item->fRowsIn+1);
   XSetForeground(dpy, font_gc, 1);

   /* pre-calculate sin and cos */
   sin_angle=TMath::Sin(angle);
   cos_angle=TMath::Cos(angle);

   /* text background will be drawn using XFillPolygon */
   item->fCornersX=
       (float *)malloc((unsigned)(4*item->fNl*sizeof(float)));
   if(!item->fCornersX) {
      free(item);
      return 0;
   }

   item->fCornersY=
       (float *)malloc((unsigned)(4*item->fNl*sizeof(float)));
   if(!item->fCornersY) {
      free(item);
      return 0;
   }

   /* draw text horizontally */

   /* start at top of bitmap */
   yp=font->ascent;

   str1=my_strdup(text);
   if(str1==0) {
      free(item);
      return 0;
   }

   str3=my_strtok(str1, str2);

   /* loop through each section in the string */
   do {
      XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                   &overall);

      /* where to draw section in x ? */
      if(align==TLEFT || align==MLEFT || align==BLEFT || align==NONE)
         xp=0;
      else if(align==TCENTRE || align==MCENTRE || align==BCENTRE)
         xp=(item->fMaxWidth-overall.rbearing)/2;
      else
         xp=item->fMaxWidth-overall.rbearing;

      /* draw string onto bitmap */
      XDrawString(dpy, canvas, font_gc, xp, yp, str3, strlen(str3));

      /* keep a note of corner positions of this string */
      item->fCornersX[ic]=((float)xp-(float)item->fColsIn/2)*gRotStyle.fMagnify;
      item->fCornersY[ic]=((float)(yp-font->ascent)-(float)item->fRowsIn/2)
          *gRotStyle.fMagnify;
      item->fCornersX[ic+1]=item->fCornersX[ic];
      item->fCornersY[ic+1]=item->fCornersY[ic]+(float)height*gRotStyle.fMagnify;
      item->fCornersX[item->fNl*4-1-ic]=item->fCornersX[ic]+
          (float)overall.rbearing*gRotStyle.fMagnify;
      item->fCornersY[item->fNl*4-1-ic]=item->fCornersY[ic];
      item->fCornersX[item->fNl*4-2-ic]=
          item->fCornersX[item->fNl*4-1-ic];
      item->fCornersY[item->fNl*4-2-ic]=item->fCornersY[ic+1];

      ic+=2;

      /* move to next line */
      yp+=height;

      str3=my_strtok((char *)0, str2);
   }
   while(str3!=0);

   free(str1);

   /* create image to hold horizontal text */
   imageIn=MakeXImage(dpy, item->fColsIn, item->fRowsIn);
   if(imageIn==0) {
      free(item);
      return 0;
   }

   /* extract horizontal text */
   XGetSubImage(dpy, canvas, 0, 0, item->fColsIn, item->fRowsIn,
                1, XYPixmap, imageIn, 0, 0);
   imageIn->format=XYBitmap;

   /* magnify horizontal text */
   if(gRotStyle.fMagnify!=1.) {
      imageIn=XRotMagnifyImage(dpy, imageIn);

      old_cols_in=item->fColsIn;
      old_rows_in=item->fRowsIn;
      item->fColsIn=int((float)item->fColsIn*gRotStyle.fMagnify);
      item->fRowsIn=int((float)item->fRowsIn*gRotStyle.fMagnify);
   }

   /* how big will rotated text be ? */
   item->fColsOut=int(TMath::Abs((float)item->fRowsIn*sin_angle) +
       TMath::Abs((float)item->fColsIn*cos_angle) +0.99999 +2);

   item->fRowsOut=int(TMath::Abs((float)item->fRowsIn*cos_angle) +
       TMath::Abs((float)item->fColsIn*sin_angle) +0.99999 +2);

   if(item->fColsOut%2==0) item->fColsOut++;

   if(item->fRowsOut%2==0) item->fRowsOut++;

   /* create image to hold rotated text */
   item->fXimage=MakeXImage(dpy, item->fColsOut, item->fRowsOut);
   if(item->fXimage==0) {
      free(item);
      return 0;
   }

   byte_w_in=(item->fColsIn-1)/8+1;
   byte_w_out=(item->fColsOut-1)/8+1;

   /* we try to make this bit as fast as possible - which is why it looks
      a bit over-the-top */

   /* vertical distance from centre */
   dj=0.5-(float)item->fRowsOut/2;

   /* where abouts does text actually lie in rotated image? */
   if(angle==0 || angle==M_PI/2 ||
      angle==M_PI || angle==3*M_PI/2) {
      xl=0;
      xr=(float)item->fColsOut;
      xinc=0;
   }
   else if(angle<M_PI) {
      xl=(float)item->fColsOut/2+
         (dj-(float)item->fRowsIn/(2*cos_angle))/
         TMath::Tan(angle)-2;
      xr=(float)item->fColsOut/2+
         (dj+(float)item->fRowsIn/(2*cos_angle))/
         TMath::Tan(angle)+2;
      xinc=1./TMath::Tan(angle);
   }
   else {
      xl=(float)item->fColsOut/2+
         (dj+(float)item->fRowsIn/(2*cos_angle))/
         TMath::Tan(angle)-2;
      xr=(float)item->fColsOut/2+
         (dj-(float)item->fRowsIn/(2*cos_angle))/
         TMath::Tan(angle)+2;

      xinc=1./TMath::Tan(angle);
   }

   /* loop through all relevent bits in rotated image */
   for(j=0; j<item->fRowsOut; j++) {

      /* no point re-calculating these every pass */
      di=(float)((xl<0)?0:(int)xl)+0.5-(float)item->fColsOut/2;
      byte_out=(item->fRowsOut-j-1)*byte_w_out;

      /* loop through meaningful columns */
      for(i=((xl<0)?0:(int)xl);
         i<((xr>=item->fColsOut)?item->fColsOut:(int)xr); i++) {

         /* rotate coordinates */
         it=int((float)item->fColsIn/2 + ( di*cos_angle + dj*sin_angle));
         jt=int((float)item->fRowsIn/2 - (-di*sin_angle + dj*cos_angle));

         /* set pixel if required */
         if(it>=0 && it<item->fColsIn && jt>=0 && jt<item->fRowsIn)
            if((imageIn->data[jt*byte_w_in+it/8] & 128>>(it%8))>0)
               item->fXimage->data[byte_out+i/8]|=128>>i%8;

         di+=1;
      }
      dj+=1;
      xl+=xinc;
      xr+=xinc;
   }
   XDestroyImage(imageIn);

   if(gRotStyle.fMagnify!=1.) {
      item->fColsIn=old_cols_in;
      item->fRowsIn=old_rows_in;
   }


#ifdef CACHE_BITMAPS

   /* create a bitmap to hold rotated text */
   item->fBitmap=XCreatePixmap(dpy, DefaultRootWindow(dpy),
                               item->fColsOut, item->fRowsOut, 1);

   /* make the text bitmap from XImage */
   XPutImage(dpy, item->fBitmap, font_gc, item->fXimage, 0, 0, 0, 0,
             item->fColsOut, item->fRowsOut);

   XDestroyImage(item->fXimage);

#endif /*CACHE_BITMAPS*/

   XFreeGC(dpy, font_gc);
   XFreePixmap(dpy, canvas);

   return item;
}


//______________________________________________________________________________
static void XRotAddToLinkedList(Display *dpy, RotatedTextItem_t *item)
{
   // Adds a text item to the end of the cache, removing as many items
   //     from the front as required to keep cache size below limit

   static long int current_size=0;
   static RotatedTextItem_t *last=0;
   RotatedTextItem_t *i1=gFirstTextItem, *i2;

#ifdef CACHE_BITMAPS

   /* I don't know how much memory a pixmap takes in the server -
          probably this + a bit more we can't account for */

   item->fSize=((item->fColsOut-1)/8+1)*item->fRowsOut;

#else

   /* this is pretty much the size of a RotatedTextItem_t */

   item->fSize=((item->fColsOut-1)/8+1)*item->fRowsOut +
      sizeof(XImage) + strlen(item->text) +
         item->fNl*8*sizeof(float) + sizeof(RotatedTextItem_t);

   if(item->font_name!=0)
      item->fSize+=strlen(item->font_name);
   else
      item->fSize+=sizeof(Font);

#endif /*CACHE_BITMAPS */

#ifdef DEBUG
   /* count number of items in cache, for debugging */
   {
      int i=0;

      while(i1) {
         i++;
         i1=i1->fNext;
      }
      DEBUG_PRINT2("Cache has %d items.\n", i);
      i1=gFirstTextItem;
   }
#endif

   DEBUG_PRINT4("current cache size=%ld, new item=%ld, limit=%d\n",
                 current_size, item->fSize, CACHE_SIZE_LIMIT*1024);

   /* if this item is bigger than whole cache, forget it */
   if(item->fSize>CACHE_SIZE_LIMIT*1024) {
      DEBUG_PRINT1("Too big to cache\n\n");
      item->fCached=0;
      return;
   }

   /* remove elements from cache as needed */
   while(i1 && current_size+item->fSize>CACHE_SIZE_LIMIT*1024) {

      DEBUG_PRINT2("Removed %ld bytes\n", i1->fSize);

      if(i1->font_name!=0)
         DEBUG_PRINT5("  (`%s'\n   %s\n   angle=%f align=%d)\n",
                      i1->fText, i1->font_name, i1->fAngle, i1->fAlign);

#ifdef CACHE_FID
      if(i1->font_name==0)
         DEBUG_PRINT5("  (`%s'\n  FID=%ld\n   angle=%f align=%d)\n",
                      i1->fText, i1->fid, i1->angle, i1->align);
#endif /*CACHE_FID*/

      current_size-=i1->fSize;

      i2=i1->fNext;

      /* free resources used by the unlucky item */
      XRotFreeTextItem(dpy, i1);

      /* remove it from linked list */
      gFirstTextItem=i2;
      i1=i2;
   }

   /* add new item to end of linked list */
   if(gFirstTextItem==0) {
      item->fNext=0;
      gFirstTextItem=item;
      last=item;
   }
   else {
      item->fNext=0;
      last->fNext=item;
      last=item;
   }

   /* new cache size */
   current_size+=item->fSize;

   item->fCached=1;

   DEBUG_PRINT1("Added item to cache.\n");
}


//______________________________________________________________________________
static void XRotFreeTextItem(Display *dpy, RotatedTextItem_t *item)
{
   //  Free the resources used by a text item

   free(item->fText);

   if(item->font_name!=0)
      free(item->font_name);

   free((char *)item->fCornersX);
   free((char *)item->fCornersY);

#ifdef CACHE_BITMAPS
   XFreePixmap(dpy, item->fBitmap);
#else
   XDestroyImage(item->fXimage);
#endif /* CACHE_BITMAPS */

   free((char *)item);
}


//______________________________________________________________________________
static XImage *XRotMagnifyImage(Display *dpy, XImage *ximage)
{
   // Magnify an XImage using bilinear interpolation

   int i, j;
   float x, y;
   float u,t;
   XImage *imageOut;
   int cols_in, rows_in;
   int cols_out, rows_out;
   register int i2, j2;
   float z1, z2, z3, z4;
   int byte_width_in, byte_width_out;
   float mag_inv;

   /* size of input image */
   cols_in=ximage->width;
   rows_in=ximage->height;

   /* size of final image */
   cols_out=int((float)cols_in*gRotStyle.fMagnify);
   rows_out=int((float)rows_in*gRotStyle.fMagnify);

   /* this will hold final image */
   imageOut=MakeXImage(dpy, cols_out, rows_out);
   if(imageOut==0)
      return 0;

   /* width in bytes of input, output images */
   byte_width_in=(cols_in-1)/8+1;
   byte_width_out=(cols_out-1)/8+1;

   /* for speed */
   mag_inv=1./gRotStyle.fMagnify;

   y=0.;

   /* loop over magnified image */
   for(j2=0; j2<rows_out; j2++) {
      x=0;
      j=int(y);

      for(i2=0; i2<cols_out; i2++) {
         i=int(x);

         /* bilinear interpolation - where are we on bitmap ? */
         /* right edge */
         if(i==cols_in-1 && j!=rows_in-1) {
            t=0;
            u=y-(float)j;

            z1=(ximage->data[j*byte_width_in+i/8] & 128>>(i%8))>0;
            z2=z1;
            z3=(ximage->data[(j+1)*byte_width_in+i/8] & 128>>(i%8))>0;
            z4=z3;
         }
         /* top edge */
         else if(i!=cols_in-1 && j==rows_in-1) {
            t=x-(float)i;
            u=0;

            z1=(ximage->data[j*byte_width_in+i/8] & 128>>(i%8))>0;
            z2=(ximage->data[j*byte_width_in+(i+1)/8] & 128>>((i+1)%8))>0;
            z3=z2;
            z4=z1;
         }
         /* top right corner */
         else if(i==cols_in-1 && j==rows_in-1) {
            u=0;
            t=0;

            z1=(ximage->data[j*byte_width_in+i/8] & 128>>(i%8))>0;
            z2=z1;
            z3=z1;
            z4=z1;
         }
         /* somewhere `safe' */
         else {
            t=x-(float)i;
            u=y-(float)j;

            z1=(ximage->data[j*byte_width_in+i/8] & 128>>(i%8))>0;
            z2=(ximage->data[j*byte_width_in+(i+1)/8] & 128>>((i+1)%8))>0;
            z3=(ximage->data[(j+1)*byte_width_in+(i+1)/8] &
                128>>((i+1)%8))>0;
            z4=(ximage->data[(j+1)*byte_width_in+i/8] & 128>>(i%8))>0;
         }

         /* if interpolated value is greater than 0.5, set bit */
         if(((1-t)*(1-u)*z1 + t*(1-u)*z2 + t*u*z3 + (1-t)*u*z4)>0.5)
            imageOut->data[j2*byte_width_out+i2/8]|=128>>i2%8;

         x+=mag_inv;
      }
      y+=mag_inv;
   }

   /* destroy original */
   XDestroyImage(ximage);

   /* return big image */
   return imageOut;
}


//______________________________________________________________________________
XPoint *XRotTextExtents(Display *, XFontStruct *font, float angle, int x, int y, char *text,int align)
{
   // Calculate the bounding box some text will have when painted

   register int i;
   char *str1, *str2, *str3;
   const char *str2_a="\0", *str2_b="\n\0";
   int height;
   float sin_angle, cos_angle;
   int nl, max_width;
   int cols_in, rows_in;
   float hot_x, hot_y;
   XPoint *xp_in, *xp_out;
   int dir, asc, desc;
   XCharStruct overall;

   /* manipulate angle to 0<=angle<360 degrees */
   while(angle<0) angle+=360;

   while(angle>360) angle-=360;

   angle*=M_PI/180;

   /* count number of sections in string */
   nl=1;
   if(align!=NONE)
      for(i=0; i<(int)strlen(text)-1; i++)
         if(text[i]=='\n')
            nl++;

   /* ignore newline characters if not doing alignment */
   if(align==NONE)
      str2=(char *)str2_a;
   else
      str2=(char *)str2_b;

   /* find width of longest section */
   str1=my_strdup(text);
   if(str1==0) return 0;

   str3=my_strtok(str1, str2);

   if(str3==0) {
      XTextExtents(font, str1, strlen(str1), &dir, &asc, &desc,
                   &overall);
   } else {
      XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                   &overall);
   }

   max_width=overall.rbearing;

   /* loop through each section */
   do {
      str3=my_strtok((char *)0, str2);

      if(str3!=0) {
         XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                      &overall);

         if(overall.rbearing>max_width)
            max_width=overall.rbearing;
      }
   }
   while(str3!=0);

   free(str1);

   /* overall font height */
   height=font->ascent+font->descent;

   /* dimensions horizontal text will have */
   cols_in=max_width;
   rows_in=nl*height;

   /* pre-calculate sin and cos */
   sin_angle=TMath::Sin(angle);
   cos_angle=TMath::Cos(angle);

   /* y position */
   if(align==TLEFT || align==TCENTRE || align==TRIGHT)
      hot_y=(float)rows_in/2*gRotStyle.fMagnify;
   else if(align==MLEFT || align==MCENTRE || align==MRIGHT)
      hot_y=0;
   else if(align==BLEFT || align==BCENTRE || align==BRIGHT)
      hot_y=-(float)rows_in/2*gRotStyle.fMagnify;
   else
      hot_y=-((float)rows_in/2-(float)font->descent)*gRotStyle.fMagnify;

   /* x position */
   if(align==TLEFT || align==MLEFT || align==BLEFT || align==NONE)
      hot_x=-(float)max_width/2*gRotStyle.fMagnify;
   else if(align==TCENTRE || align==MCENTRE || align==BCENTRE)
      hot_x=0;
   else
      hot_x=(float)max_width/2*gRotStyle.fMagnify;

   /* reserve space for XPoints */
   xp_in=(XPoint *)malloc((unsigned)(5*sizeof(XPoint)));
   if(!xp_in) return 0;

   xp_out=(XPoint *)malloc((unsigned)(5*sizeof(XPoint)));
   if(!xp_out) {
      free(xp_in);
      return 0;
   }

   /* bounding box when horizontal, relative to bitmap centre */
   xp_in[0].x=(short int)(-(float)cols_in*gRotStyle.fMagnify/2-gRotStyle.fBbxPadl);
   xp_in[0].y=(short int)( (float)rows_in*gRotStyle.fMagnify/2+gRotStyle.fBbxPadl);
   xp_in[1].x=(short int)( (float)cols_in*gRotStyle.fMagnify/2+gRotStyle.fBbxPadl);
   xp_in[1].y=(short int)( (float)rows_in*gRotStyle.fMagnify/2+gRotStyle.fBbxPadl);
   xp_in[2].x=(short int)( (float)cols_in*gRotStyle.fMagnify/2+gRotStyle.fBbxPadl);
   xp_in[2].y=(short int)(-(float)rows_in*gRotStyle.fMagnify/2-gRotStyle.fBbxPadl);
   xp_in[3].x=(short int)(-(float)cols_in*gRotStyle.fMagnify/2-gRotStyle.fBbxPadl);
   xp_in[3].y=(short int)(-(float)rows_in*gRotStyle.fMagnify/2-gRotStyle.fBbxPadl);
   xp_in[4].x=xp_in[0].x;
   xp_in[4].y=xp_in[0].y;

   /* rotate and translate bounding box */
   for(i=0; i<5; i++) {
      xp_out[i].x=(short int)((float)x + ( ((float)xp_in[i].x-hot_x)*cos_angle +
                                           ((float)xp_in[i].y+hot_y)*sin_angle));
      xp_out[i].y=(short int)((float)y + (-((float)xp_in[i].x-hot_x)*sin_angle +
                                          ((float)xp_in[i].y+hot_y)*cos_angle));
   }

   free((char *)xp_in);

   return xp_out;
}
