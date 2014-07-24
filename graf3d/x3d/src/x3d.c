/* @(#)root/x3d:$Id$ */
/* Author: Mark Spychalla*/
/*
  Copyright 1992 Mark Spychalla

  Permission to use, copy, modify, distribute, and sell this software and
  its documentation for any purpose is hereby granted without fee,
  provided that the above copyright notice appear in all copies and that
  both that copyright notice and this permission notice appear in
  supporting documentation, and that the name of Mark Spychalla not be used
  in advertising or publicity pertaining to distribution of the software
  without specific, written prior permission.  Mark Spychalla makes no
  representations about the suitability of this software for any purpose.
  It is provided "as is" without express or implied warranty.

  Mark Spychalla disclaims all warranties with regard to this software,
  including all implied warranties of merchantability and fitness, in no
  event shall Mark Spychalla be liable for any special, indirect or
  consequential damages or any damages whatsoever resulting from loss of use,
  data or profits, whether in an action of contract, negligence or other
  tortious action, arising out of or in connection with the use or performance
  of this software.
*/



/*

NOTE ON X3D CODING STYLE:

   Don't think I usually code in the gerberized fashion that X3D demonstrates.
X3D was written for speed at any cost.  My goal was to write the fastest 3D
object viewer that I could, period.   Regular programs ought to be written
with different goals in mind such as:

1) A program has excellent documentation that ANYONE can read.
2) A program when released has no strange "features" or bugs.
3) A program is robust and handles ALL extreme and unusual cases.
4) A program is written in phases and modules with hard tests for each one.
5) A program is written for any user who doesn't need special knowledge
   to use the program.
6) A program has well defined user requirements and functional specifications.
7) A program is written with regard to future expansion and integreation
   with other systems (portability).

When programming following these additional principles make programs easier
to maintain.

A) Choose variable names that accurately describes what the variable does/is.
B) Write comments to inform someone faced with the task of modifying your code.
C) Avoid excessive comments.  Write the code so that it says what it does.
D) Follow a strict one-in, one-out flow of control structues except in the
   case of fatal error conditions.
E) Avoid using global variables.
F) Do not cause side effects to variables that were not parameters to a
   function.
G) Have a single function perform a single purpose.
H) Select a single indentation style and stick with it.
I) Use a consistent naming convention.

The following principles help me when I try optimizing code:

a) If optimizing, use a profiler to determine which sections of code most of
   the time is spent in.  Spend most of your effort in the most used sections.
   Don't bother optimizing a procedure using less than 10% of the time.

b) High level optimizations are far more effective than cycle shaving.
   (e.g. use quick sort instead of optimizing a bubble sort.)

c) Be flexible in your approach to solving a problem.  List exactly what you
   need as a result at a minimum.  Get rid of unnecessary assumptions.

d) Become familiar with sets of operations that are equivalent, or nearly so.
   Learn the relative expense of basic operations.

e) If possible, be careful not to needlessly sacrifice significant readability
   of the code for a cycle or two.

-- Spy

*/


#ifndef WIN32
#include "x3d.h"
#include "X3DBuffer.h"
#endif

#ifdef WIN32

unsigned long x3d_main(float *longitude, float *latitude, float *psi,
                       const char *string) { return 0L; }
void x3d_terminate() { }
void x3d_get_position(float *longitude, float *latitude, float *psi) { }
int  x3d_dispatch_event(unsigned long event) { return 0; }
void x3d_set_display(unsigned long display) { }
void x3d_update() { }

#else


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/X.h>


extern Color   *colors;
extern point   *points;
extern segment *segs;
extern polygon *polys;

extern int  currPoint, currSeg, currPoly;

static polygon **list;
static point   *bounds;
static int quitApplication = 0;
static Display *gDisplay = NULL;
static Ginfo *gGInfo = NULL;
static Oinfo *gOInfo = NULL;

static int gRedDiv, gGreenDiv, gBlueDiv, gRedShift, gGreenShift, gBlueShift;



static void sort(list1, numPolys)
polygon **list1;
int numPolys;
/*****************************************************************************
   Specialized quick sort for painter algorithm.
*****************************************************************************/
{
polygon **v0, **v1, **v2, **v3, **v4, **v5, **v6, *poly;
register int stackIndex, stackNotSet, length, start, end, high;
float dist;
int numPoints;
StackElement stack[MAXSTACK];
point **Point, **lastPoint;

   v0 = list1;
   v1 = &(list1[numPolys]);

/* Set the key value to be the average of the vertices' distances */

   while(v0 < v1){
      poly = *v0;
      numPoints = poly->numPoints;
      Point = poly->points;
      lastPoint = Point + numPoints;
      dist = 0.0;

      do{
         dist += (*Point)->dist;
         Point++;
      }while(Point < lastPoint);

      poly->dist = dist / ((float)numPoints);
      v0++;
      }

/* Initialize for the qsort() */

   stackIndex = 1;
   stackNotSet = 0;
   start = 0;
   end = numPolys - 1;

/* Do Qsort */

   while(stackIndex){

       if(stackNotSet){
          start = stack[stackIndex].start;
          end = stack[stackIndex].end;
       }

       stackIndex--;
       stackNotSet = 1;
       length = end - start;

/* Big enough to qsort ? */

      if(length > STOP){
         v1 = &(list1[start]);
         v2 = &(list1[start + (length / 4)]);
         v3 = &(list1[start + (length / 2)]);
         v4 = &(list1[start + ((length * 3) / 4)]);
         v5 = &(list1[end]);
         v6 = v1;

         median5(v1,v2,v3,v4,v5)

         *v0 = *v3;
         *v3 = *v6;
         *v6 = *v0;

         v1 = &(list1[start + 1]);
         v2 = &(list1[end]);

/* Split */

         dist = (*v6)->dist;
         while((*v2)->dist < dist) v2--;
         while((*v1)->dist > dist) v1++;

         v5 = v0;

         while(v1 < v2){

            *v5 = *v2;
            *v2 = *v1;

            v5 = v1;

            do{
               v2--;
            }while(((*v2)->dist < dist) && (v1 < v2));
            if (v2 <= v1) break;

            do{
               v1++;
            }while(((*v1)->dist > dist) && (v1 < v2));
            if (v2 <= v1) break;
         }

         v2 = v1 - 1;

         *v5 = *v2;
         *v2 = *v6;
         *v6 = *v0;

         high = v2 - list1;

/* Put sublists on the stack, smallest on top */

         if((high - start) > (end - high)){
            stack[++stackIndex].start = start;
            stack[stackIndex].end = high - 1;
            ++stackIndex;
            start = high + 1;
            stackNotSet = 0;
         }else{
            stack[++stackIndex].start = high + 1;
            stack[stackIndex].end = end;
            ++stackIndex;
            end = high - 1;
            stackNotSet = 0;
            }
      }
   }

/* insertion sort all the remaining sublists at once */

   v2 = list1;
   v3 = &(list1[numPolys - 1]);
   v4 = v2 + 1;

   while(v4 <= v3){

      *v0 = *v4;
      v1 = v4 - 1;

      while((v1 >= v2) && ((*v1)->dist < (*v0)->dist)){
         *(v1 + 1) = *v1;
         v1--;
         }

      *(v1 + 1) = *v0;
      v4++;
      }
}



static void Rotate(points1, cx, cy, cz, sx, sy, sz)
anglePoint *points1;
double cx, cy, cz, sx, sy, sz;
/******************************************************************************
   Rotate about Z, X, then Y, for two points.
******************************************************************************/
{
int index1;
double x, y, z, t;

    for(index1 = 0; index1 < 2; index1++){
       x = points1[index1].x;
       y = points1[index1].y;
       z = points1[index1].z;

       t = x * cz + y * sz;
       y = y * cz - x * sz;
       x = t;

       points1[index1].y = y * cx + z * sx;

       z = z * cx - y * sx;

       points1[index1].x = x * cy + z * sy;
       points1[index1].z = z * cy - x * sy;
       }
}



static double DotProduct(x1, Y1, x2, y2)
double x1, Y1, x2, y2;
/******************************************************************************
   Dot product (calculate the cosine of the angle between two vectors).
******************************************************************************/
{
double temp;

   if((x1 == 0.0 && Y1 == 0.0)){
      return 1.0;
      }

   temp = sqrt(x1 * x1 + Y1 * Y1);
   x1 = x1 / temp;
   Y1 = Y1 / temp;

   temp = x1 * x2 + Y1 * y2;

   if(temp > 1.0)
      temp = fmod(temp, 1.0);

   if(temp < -1.0)
      temp = -fmod(-temp, 1.0);

   return(temp);
}



static void CalculateAngles(X, Y, Z, X1, Y1, Z1)
double *X, *Y, *Z;
double X1, Y1, Z1;
/******************************************************************************
   Calculate what the result of the angle changes of X1, Y1, and Z1 are
   in my weird coordinate system.
******************************************************************************/
{
anglePoint points1[2];

   points1[0].x = 0.0; points1[0].y = 0.0; points1[0].z = 1.0;
   points1[1].x = 1.0; points1[1].y = 0.0; points1[1].z = 0.0;

   Rotate(points1, cos(*X), cos(*Y), cos(*Z), sin(*X), sin(*Y), sin(*Z));
   Rotate(points1, cos(X1), cos(Y1), cos(Z1), sin(X1), sin(Y1), sin(Z1));

   *Y = acos(DotProduct(points1[0].x, points1[0].z, 0.0, 1.0));

   if(points1[0].x < 0.0)
      *Y = -*Y;

   Rotate(points1, 1.0, cos(-*Y), 1.0, 0.0, sin(-*Y), 0.0);
   *X = acos(DotProduct(points1[0].y, points1[0].z, 0.0, 1.0));

   if(points1[0].y < 0.0)
      *X = -*X;

   Rotate(points1, cos(-*X), 1.0, 1.0, sin(-*X), 0.0, 0.0);
   *Z = acos(DotProduct(points1[1].x, points1[1].y, 1.0, 0.0));

   if(!(points1[1].y < 0.0))
      *Z = -*Z;
}



static void DrawLogo(g, x, y)
Ginfo *g;
int x, y;
/******************************************************************************
   Display the Logo.
******************************************************************************/
{
int hUnit, vUnit;
   _XPoint points1[512];
   void *ptrp = points1;

   hUnit = XTextWidth(g->font, LONGESTSTRING, strlen(LONGESTSTRING)) /
   strlen(LONGESTSTRING);
   vUnit = FONTHEIGHT(g->font);

/* X */

   points1[0].x =  9 * hUnit + x; points1[0].y =  1 * vUnit + y;
   points1[1].x =  9 * hUnit + vUnit + x; points1[1].y =  1 * vUnit + y;
   points1[2].x = 14 * hUnit + vUnit + x; points1[2].y =  6 * vUnit + y;
   points1[3].x = 14 * hUnit + x; points1[3].y =  6 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 14 * hUnit + vUnit + x; points1[0].y =  1 * vUnit + y;
   points1[1].x = 14 * hUnit + x; points1[1].y =  1 * vUnit + y;
   points1[2].x =  9 * hUnit + x; points1[2].y =  6 * vUnit + y;
   points1[3].x =  9 * hUnit + vUnit + x; points1[3].y =  6 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

/* 3 */

   points1[0].x = 18 * hUnit + x; points1[0].y =  1 * vUnit + y;
   points1[1].x = 22 * hUnit + x; points1[1].y =  1 * vUnit + y;
   points1[2].x = 23 * hUnit + x; points1[2].y =  2 * vUnit + y;
   points1[3].x = 18 * hUnit + x; points1[3].y =  2 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 23 * hUnit - vUnit + x; points1[0].y =  2 * vUnit + y;
   points1[1].x = 23 * hUnit + x; points1[1].y =  2 * vUnit + y;
   points1[2].x = 23 * hUnit + x; points1[2].y =  3 * vUnit + y;
   points1[3].x = 23 * hUnit - vUnit + x; points1[3].y =  4 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 23 * hUnit - vUnit + x; points1[0].y =  3 * vUnit + y;
   points1[1].x = 23 * hUnit + x; points1[1].y =  4 * vUnit + y;
   points1[2].x = 23 * hUnit + x; points1[2].y =  5 * vUnit + y;
   points1[3].x = 23 * hUnit - vUnit + x; points1[3].y =  5 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 18 * hUnit + x; points1[0].y =  5 * vUnit + y;
   points1[1].x = 23 * hUnit + x; points1[1].y =  5 * vUnit + y;
   points1[2].x = 22 * hUnit + x; points1[2].y =  6 * vUnit + y;
   points1[3].x = 18 * hUnit + x; points1[3].y =  6 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 19 * hUnit + x; points[0].y =  3 * vUnit + y;
   points1[1].x = 23 * hUnit - vUnit + x; points1[1].y =  3 * vUnit + y;
   points1[2].x = 23 * hUnit - vUnit + x; points1[2].y =  4 * vUnit + y;
   points1[3].x = 19 * hUnit + x; points1[3].y =  4 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

/* D */

   points1[0].x = 26 * hUnit + x; points1[0].y =  1 * vUnit + y;
   points1[1].x = 30 * hUnit + x; points1[1].y =  1 * vUnit + y;
   points1[2].x = 30 * hUnit + vUnit + x; points1[2].y =  2 * vUnit + y;
   points1[3].x = 26 * hUnit + x; points1[3].y =  2 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 26 * hUnit + x; points1[0].y =  5 * vUnit + y;
   points1[1].x = 30 * hUnit + vUnit + x; points1[1].y =  5 * vUnit + y;
   points1[2].x = 30 * hUnit + x; points1[2].y =  6 * vUnit + y;
   points1[3].x = 26 * hUnit + x; points1[3].y =  6 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 26 * hUnit + x; points1[0].y =  1 * vUnit + y;
   points1[1].x = 26 * hUnit + vUnit + x; points1[1].y =  1 * vUnit + y;
   points1[2].x = 26 * hUnit + vUnit + x; points1[2].y =  6 * vUnit + y;
   points1[3].x = 26 * hUnit + x; points1[3].y =  6 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);

   points1[0].x = 30 * hUnit + x; points1[0].y =  2 * vUnit + y;
   points1[1].x = 30 * hUnit + vUnit + x; points1[1].y =  2 * vUnit + y;
   points1[2].x = 30 * hUnit + vUnit + x; points1[2].y =  5 * vUnit + y;
   points1[3].x = 30 * hUnit + x; points1[3].y =  5 * vUnit + y;

   XFillPolygon(g->dpy, g->helpWin, g->helpGc, ptrp, 4, Convex,
   CoordModeOrigin);
}



static void DisplayMenu(g)
Ginfo *g;
/******************************************************************************
   Display the help menu.
******************************************************************************/
{
int x = 5, y = 5;

   XSetFont(g->dpy, g->helpGc, g->font->fid );

   XSetWindowBackground(g->dpy, g->helpWin, g->black);
   XSetForeground(g->dpy, g->helpGc, g->white);
   XSetBackground(g->dpy, g->helpGc, g->black);

   XSetStipple(g->dpy, g->helpGc, g->stipple[NUMSTIPPLES / 3]);
   XSetFillStyle(g->dpy, g->helpGc, FillOpaqueStippled);

   DrawLogo(g, (XTextWidth(g->font, LONGESTSTRING, strlen(LONGESTSTRING)) /
   (int)strlen(LONGESTSTRING)) / 2, FONTHEIGHT(g->font) / 3);

   XSetFillStyle(g->dpy, g->helpGc, FillSolid);

   DrawLogo(g, 0, 0);

   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"              VERSION 2.2");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"  CONTROLS SUMMARY");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"     QUIT                    q Q");
   HelpPrint(g,x,y,"     WIREFRAME MODE          w W");
   HelpPrint(g,x,y,"     HIDDEN LINE MODE        e E");
   HelpPrint(g,x,y,"     HIDDEN SURFACE MODE     r R");
   HelpPrint(g,x,y,"     MOVE OBJECT DOWN        u U");
   HelpPrint(g,x,y,"     MOVE OBJECT UP          i I");
   HelpPrint(g,x,y,"     TOGGLE CONTROLS STYLE   o O");
   HelpPrint(g,x,y,"     TOGGLE STEREO DISPLAY   s S");
   HelpPrint(g,x,y,"     TOGGLE BLUE STEREO VIEW d D");
   HelpPrint(g,x,y,"     TOGGLE DOUBLE BUFFER    f F");
   HelpPrint(g,x,y,"     MOVE OBJECT RIGHT       h H");
   HelpPrint(g,x,y,"     MOVE OBJECT BACKWARD    j J");
   HelpPrint(g,x,y,"     MOVE OBJECT FOREWARD    k K");
   HelpPrint(g,x,y,"     MOVE OBJECT LEFT        l L");
   HelpPrint(g,x,y,"     TOGGLE HELP MENU        m M");
   HelpPrint(g,x,y,"     ROTATE ABOUT X          x X a A");
   HelpPrint(g,x,y,"     ROTATE ABOUT Y          y Y b B");
   HelpPrint(g,x,y,"     ROTATE ABOUT Z          z Z c C");
   HelpPrint(g,x,y,"     AUTOROTATE ABOUT X      1 2 3");
   HelpPrint(g,x,y,"     AUTOROTATE ABOUT Y      4 5 6");
   HelpPrint(g,x,y,"     AUTOROTATE ABOUT Z      7 8 9");
   HelpPrint(g,x,y,"     ADJUST FOCUS            [ ] { }");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"  POINTER MOVEMENT WITH LEFT BUTTON :");
   HelpPrint(g,x,y,"");
   HelpPrint(g,x,y,"     ROTATE OBJECT ABOUT X   Vertical");
   HelpPrint(g,x,y,"     ROTATE OBJECT ABOUT Z   Horizontal");
}



static void ResetPurpleRectangle(XL, YL, XH, YH, g)
int XL, YL, XH, YH;
Ginfo *g;
/******************************************************************************
   Reset the vertices of the purple rectangle.
******************************************************************************/
{
   g->redSegments[3].x1  = (XL + MARGIN);
   g->blueSegments[3].x1 = (XL + MARGIN);
   g->redSegments[3].y1  = (YL + MARGIN);
   g->blueSegments[3].y1 = (YL + MARGIN);
   g->redSegments[3].x2  = (XH - MARGIN);
   g->blueSegments[3].x2 = (XH - MARGIN);
   g->redSegments[3].y2  = (YL + MARGIN);
   g->blueSegments[3].y2 = (YL + MARGIN);
   g->redSegments[2].x1  = (XH - MARGIN);
   g->blueSegments[2].x1 = (XH - MARGIN);
   g->redSegments[2].y1  = (YH - MARGIN);
   g->blueSegments[2].y1 = (YH - MARGIN);
   g->redSegments[2].x2  = (XL + MARGIN);
   g->blueSegments[2].x2 = (XL + MARGIN);
   g->redSegments[2].y2  = (YH - MARGIN);
   g->blueSegments[2].y2 = (YH - MARGIN);
   g->redSegments[1].x1  = (XH - MARGIN);
   g->blueSegments[1].x1 = (XH - MARGIN);
   g->redSegments[1].y1  = (YL + MARGIN);
   g->blueSegments[1].y1 = (YL + MARGIN);
   g->redSegments[1].x2  = (XH - MARGIN);
   g->blueSegments[1].x2 = (XH - MARGIN);
   g->redSegments[1].y2  = (YH - MARGIN);
   g->blueSegments[1].y2 = (YH - MARGIN);
   g->redSegments[0].x1  = (XL + MARGIN);
   g->blueSegments[0].x1 = (XL + MARGIN);
   g->redSegments[0].y1  = (YL + MARGIN);
   g->blueSegments[0].y1 = (YL + MARGIN);
   g->redSegments[0].x2  = (XL + MARGIN);
   g->blueSegments[0].x2 = (XL + MARGIN);
   g->redSegments[0].y2  = (YH - MARGIN);
   g->blueSegments[0].y2 = (YH - MARGIN);
}



static void OneBitSetColors(g)
Ginfo *g;
/******************************************************************************
   Set up color information/stipples for a one bit display.
******************************************************************************/
{
int index1;
Color *color;
int numColors;

   color = g->colors;
   numColors = g->numColors;

/* Set the colors (may not be used) */

   for(index1 = 0; index1 < numColors; index1++){
      color[index1].value = 1;

/* Set the stipples */

      color[index1].stipple =(int)((double)NUMSTIPPLES *
      ((double)sqrt((double)(
      (double)color[index1].red   * (double)color[index1].red +
      (double)color[index1].green * (double)color[index1].green +
      (double)color[index1].blue  * (double)color[index1].blue))
      / MAXCOLORDIST));
      }
}



static void EightBitSetColors(g)
Ginfo *g;
/******************************************************************************
   Set up color information/stipples for a eight bit display.
******************************************************************************/
{
Color *colors1;
int numColors;
int colorIndex = 0;
int index1, index2, redIndex, blueIndex, greenIndex;
XColor c;

   colors1 = g->colors;
   numColors = g->numColors;

/* Put "black" into the place reserved for it in the end */

   colors1[numColors].red   = 0;
   colors1[numColors].green = 0;
   colors1[numColors].blue  = 0;

/* Put "red" into the place reserved for it in the end */

   colors1[numColors + 1].red   = 255;
   colors1[numColors + 1].green = 0;
   colors1[numColors + 1].blue  = 0;

/* Put "blue" into the place reserved for it in the end */

   colors1[numColors + 2].red   = 0;
   colors1[numColors + 2].green = 0;
   colors1[numColors + 2].blue  = 255;

/* Put "purple" into the place reserved for it in the end */

   colors1[numColors + 3].red   = 255;
   colors1[numColors + 3].green = 0;
   colors1[numColors + 3].blue  = 255;

/* Blank out the colormap */

   for(index1 = 0; index1 < 256; index1++){
      c.red    = 0;
      c.green  = 0;
      c.blue   = 0;
      c.flags  = DoRed | DoGreen | DoBlue;
      c.pixel  = 255;
      c.pad    = 0;
      g->cmapColors[0][index1] = c;
      g->cmapColors[1][index1] = c;
      g->cmapColors[2][index1] = c;
      }

   if(numColors <= BUFFER_CMAP){

      colorIndex= numColors + 3;
      index1 = 15;

/* Set stipple, and colormap double buffer colors */

      while((index1 > 0) && (colorIndex >= 0)){
         c.red    = colors1[colorIndex].red   << 8;
         c.green  = colors1[colorIndex].green << 8;
         c.blue   = colors1[colorIndex].blue  << 8;
         c.flags  = DoRed | DoGreen | DoBlue;

         colors1[colorIndex].value = index1 * 16 + index1;

         colors1[colorIndex].stipple =(int)((double)NUMSTIPPLES *
         ((double)sqrt((double)(
         (double)colors1[colorIndex].red   * (double)colors1[colorIndex].red +
         (double)colors1[colorIndex].green * (double)colors1[colorIndex].green +
         (double)colors1[colorIndex].blue  * (double)colors1[colorIndex].blue))
         / MAXCOLORDIST));

         for(index2 = 1; index2 < 16; index2++){
            c.pixel  = index2 * 16 + index1;
            g->cmapColors[0][index2 * 16 + index1] = c;

            c.pixel  = index1 * 16 + index2;
            g->cmapColors[1][index1 * 16 + index2] = c;
            }

         index1--;
         colorIndex--;
         }
   }else{

/* Set permanent black, red, blue, purple for cmap double buffer */

      for(index1 = 0; index1 < 4; index1++){
         c.red    = colors1[numColors + index1].red   << 8;
         c.green  = colors1[numColors + index1].green << 8;
         c.blue   = colors1[numColors + index1].blue  << 8;
         c.flags  = DoRed | DoGreen | DoBlue;
         c.pixel  = 12 + index1;
         g->cmapColors[0][12 + index1] = c;
         g->cmapColors[1][12 + index1] = c;
         colors1[numColors + index1].value = c.pixel;
         }

      if(numColors <= MAX_COLORS){
         colorIndex = 0;
         index1 = 9;
         index2 = 0;

/* Fill in the rest of the colors */

         while(colorIndex < numColors){
            if((index1 < 12) || (index1 > 15)){
               c.red    = colors1[colorIndex].red   << 8;
               c.green  = colors1[colorIndex].green << 8;
               c.blue   = colors1[colorIndex].blue  << 8;
               c.flags  = DoRed | DoGreen | DoBlue;
               c.pixel  = index1;
               g->cmapColors[0][index1] = c;
               g->cmapColors[1][index1] = c;
               colors1[colorIndex].value = index1;

               colors1[colorIndex].stipple =(int)((double)NUMSTIPPLES *
               ((double)sqrt((double)(
               (double)colors1[colorIndex].red *
               (double)colors1[colorIndex].red +
               (double)colors1[colorIndex].green *
               (double)colors1[colorIndex].green +
               (double)colors1[colorIndex].blue  *
               (double)colors1[colorIndex].blue))
               / MAXCOLORDIST));

               colorIndex++;
               }
            index1++;
            }
      }else{
         index1 = 17;
         index2 = 0;
         redIndex   = 0;
         greenIndex = 0;
         blueIndex  = 0;

/* Otherwise use a default lot */

         while(blueIndex < MAXVALUE){
            c.red    = (redIndex * VALUESCALE) << 8;
            c.green  = (greenIndex * VALUESCALE) << 8;
            c.blue   = (blueIndex * VALUESCALE) << 8;
            c.flags  = DoRed | DoGreen | DoBlue;
            c.pixel  = index1;
            g->cmapColors[0][index1] = c;
            g->cmapColors[1][index1] = c;

            redIndex++;

            if(redIndex >= MAXVALUE){
               redIndex = 0;
               greenIndex++;
               }

            if(greenIndex >= MAXVALUE){
               greenIndex = 0;
               blueIndex++;
               }
            index1++;
            }

         for(index1 = 0; index1 < numColors; index1++){
            colors1[index1].value = colors1[index1].red * 36 +
               colors1[index1].green * 6 + colors1[index1].blue + 17;

            colors1[colorIndex].stipple =(int)((double)NUMSTIPPLES *
            ((double)sqrt((double)(
            (double)colors1[colorIndex].red *
            (double)colors1[colorIndex].red +
            (double)colors1[colorIndex].green *
            (double)colors1[colorIndex].green +
            (double)colors1[colorIndex].blue  *
            (double)colors1[colorIndex].blue))
            / MAXCOLORDIST));

            }
         }
      }

/* Set the colors for the special fast colormap double buffer */

   index1 = 0;
   for(redIndex = 0; redIndex < 4; redIndex++){
      for(blueIndex = 0; blueIndex < 4; blueIndex++){
          if(redIndex != blueIndex){
             g->wireframeColors[0][index1] =
                g->cmapColors[0][(redIndex + 12) * 16 + (blueIndex + 12)];
             g->wireframeColors[1][index1] =
                g->cmapColors[1][(redIndex + 12) * 16 + (blueIndex + 12)];
             index1++;
             }
          }
      }

/* Just in case set the rest of the colors */

   for(index1 = 13; index1 < 256; index1++){
      g->wireframeColors[0][index1] = g->wireframeColors[0][3];
      g->wireframeColors[1][index1] = g->wireframeColors[1][3];
      }

/* Set the colors for the pix stereo mode */

   for(redIndex = 0; redIndex < 15; redIndex++){
      for(blueIndex = 0; blueIndex < 15; blueIndex++){
         c.red    = (redIndex * 17) << 8;
         c.green  = 0;
         c.blue   = (blueIndex * 17) << 8;
         c.flags  = DoRed | DoGreen | DoBlue;
         c.pixel  = (redIndex + 1) * 16 + (blueIndex + 1);
         g->cmapColors[2][c.pixel] = c;
         }
      }

/* Set stereoColor to nearest color */

   for(index1 = 0; index1 < numColors; index1++){
      colorIndex = (int)((double)15 *
      ((double)sqrt((double)((double)colors1[index1].red *
      (double)colors1[index1].red + (double)colors1[index1].green *
      (double)colors1[index1].green + (double)colors1[index1].blue *
      (double)colors1[index1].blue)) / MAXCOLORDIST));

      colors1[index1].stereoColor = (colorIndex + 1) * 16 + (colorIndex + 1);
      }

/* Set various important color values */

   g->stereoBlack  = (0 + 1) * 16 + (0 + 1);
   g->redMask  = BUFFER0;
   g->blueMask = BUFFER1;
   g->Black  = colors1[numColors].value;
   g->Red    = colors1[numColors + 1].value;
   g->Blue   = colors1[numColors + 2].value;
   g->Purple = colors1[numColors + 3].value;
}


static void TrueColorSetColors(g)
Ginfo *g;
/******************************************************************************
   Set up color information/stipples for TrueColor displays.
******************************************************************************/
{
int index1, colorValue;
Color *colors1;
int numColors;

   /* On TrueColor displays the color pixel value composed directly of
      r, g and b components. The order of the r, g and b and the number
      of bits used for each color is specified by the X server and decoded
      in the gRedShift, gRedDiv etc. values. Since X3D uses 255 as max
      color the div is the number of bits to left shift (divide) from 255.
   */

   colors1 = g->colors;
   numColors = g->numColors;

   for(index1 = 0; index1 < numColors; index1++){

/* In TrueColor every color is what it is */

      colors1[index1].value =
      (colors1[index1].red >>   gRedDiv)   << gRedShift |
      (colors1[index1].green >> gGreenDiv) << gGreenShift |
      (colors1[index1].blue >>  gBlueDiv)  << gBlueShift;

/* Set stipple */

      colors1[index1].stipple =(int)((double)NUMSTIPPLES *
      ((double)sqrt((double)(
      (double)colors1[index1].red   * (double)colors1[index1].red +
      (double)colors1[index1].green * (double)colors1[index1].green +
      (double)colors1[index1].blue  * (double)colors1[index1].blue))
      / MAXCOLORDIST));

/* Set stereo color */

      colorValue= (int)((double)(255 >> gRedDiv) *
      ((double)sqrt((double)((double)colors1[index1].red *
      (double)colors1[index1].red + (double)colors1[index1].green *
      (double)colors1[index1].green + (double)colors1[index1].blue *
      (double)colors1[index1].blue)) / MAXCOLORDIST));

      colors1[index1].stereoColor = (colorValue << gRedShift) |
                                  (colorValue << gBlueShift);
      }

/* Set various important color values */

   g->stereoBlack  = 0;
   g->redMask  = (255 >> gRedDiv) << gRedShift;
   g->blueMask = (255 >> gBlueDiv) << gBlueShift;
   g->Black  = 0;
   g->Red    = (255 >> gRedDiv) << gRedShift;
   g->Blue   = (255 >> gBlueDiv) << gBlueShift;
   g->Purple = g->Red | g->Blue;
}


char title[80];
Atom wm_protocols[2];

static void InitDisplay(o, g, parent)
Oinfo *o;
Ginfo *g;
Window parent;
/******************************************************************************
   Set up an X window and our colormap.  We rely on X's own error handling and
   reporting for most bad X calls because X buffers requests.
******************************************************************************/
{
static int stipples[NUMSTIPPLES][NUMSTIPPLES * 2 + 1] = {
{0},
{1, 1, 1},
{2, 0, 2, 2, 1},
{3, 1, 0, 1, 2, 3, 1},
{4, 0, 1, 0, 3, 2, 1, 2, 3},
{5, 0, 0, 0, 2, 2, 0, 2, 3, 3, 2},
{6, 0, 2, 1, 1, 1, 3, 2, 0, 3, 1, 3, 3},
{7, 0, 1, 0, 3, 1, 0, 1, 1, 2, 2, 3, 1, 3, 3},
{8, 0, 1, 0, 3, 1, 0, 1, 2, 2, 1, 2, 3, 3, 0, 3, 2},
{9, 0, 0, 0, 2, 1, 2, 1, 3, 2, 0, 2, 1, 2, 3, 3, 1, 3, 2},
{10,0, 0, 0, 1, 0, 3, 1, 0, 1, 2, 2, 1, 2, 2, 2, 3, 3, 0, 3, 2},
{11,0, 1, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 1, 2, 2, 3, 0, 3, 2, 3, 3},
{12,0, 0, 0, 2, 1, 0, 1, 1, 1, 2, 1, 3, 2, 1, 2, 3, 3, 0, 3, 1, 3, 2, 3, 3},
{13,0, 0, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 2, 0, 2, 2, 2, 3, 3, 0, 3, 1, 3, 2,
 3, 3},
{14,0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 3, 2, 1, 2, 2, 2, 3, 3, 0, 3, 1,
 3, 2, 3, 3},
{15,0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 3, 0,
 3, 1, 3, 2, 3, 3},
{16,0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3,
 3, 0, 3, 1, 3, 2, 3, 3}
};

char bits[(STIPPLESIZE * STIPPLESIZE) / BITSPERBYTE];

GC temp_gc;
XColor oldColormap[MAXCOLORS];
XWindowAttributes attributes;
XSetWindowAttributes attribs;
XWMHints wmhint;
int index1, index2, screen;
Visual *vis;
XSizeHints sizehint;
int x, y, NUMCOLORS;
unsigned int width, height, numSegments;
int useroot = 0;

   if (gDisplay)
      useroot = 1;


   numSegments = o->numSegs;

   if((g->redColors = (long *)calloc(1, (numSegments + 4) * (sizeof(long))))
    == NULL){
(void)fprintf(stderr, "Unable to allocate memory for redColors\n"); return;}

   if((g->redSegments = (XSegment *)calloc(1, (numSegments + 4) *
   (sizeof(XSegment)))) == NULL){
(void)fprintf(stderr, "Unable to allocate memory for redSegments\n"); return;}

   if((g->blueSegments = (XSegment *)calloc(1, (numSegments + 4) *
   sizeof(XSegment))) == (XSegment *)NULL){
(void)fprintf(stderr, "Unable to allocate memory for blueSegments\n"); return;}

/* Can we connect with the server? */

   if (!useroot)
      g->dpy = XOpenDisplay(g->DisplayName);
   else
      g->dpy = gDisplay;
   if(g->dpy == NULL){
      fprintf(stderr, "Cannot connect to server\n");
      return;
   }

   screen = DefaultScreen(g->dpy);
   g->black =  (long)BlackPixel(g->dpy, screen);
   g->white =  (long)WhitePixel(g->dpy, screen);

/* Initialize various flags and default values */

   g->requestSize = XMaxRequestSize(g->dpy) / REQUESTFACTOR;
   g->dpyX = DisplayWidth(g->dpy, screen);
   g->dpyY = DisplayHeight(g->dpy, screen);
   g->winX = g->dpyX / 2;
   g->winY = g->dpyY / 2 /* - 25 */ ;
   g->mono = g->ColorSelect = g->oldPointerX = g->oldPointerY = 0;
   g->Block = 1;
   g->Relative = 0;

/* Initialize the fonts */

   if((g->font = XLoadQueryFont(g->dpy, FONT)) == NULL){
      fprintf(stderr, "Unable to load font: %s ... trying fixed\n", FONT);

      if((g->font = XLoadQueryFont(g->dpy, FIXED)) == NULL){
         fprintf(stderr, "Unable to load font: %s\n", FIXED);
         return;
         }
      }

   if((g->titleFont = XLoadQueryFont(g->dpy, TITLEFONT)) == NULL){
      fprintf(stderr, "Unable to load font: %s ... trying fixed\n", TITLEFONT);

      if((g->titleFont = XLoadQueryFont(g->dpy, FIXED)) == NULL){
         fprintf(stderr, "Unable to load font: %s\n", FIXED);
         return;
         }
      }

   if((g->boldFont = XLoadQueryFont(g->dpy, BOLDFONT)) == NULL){
      fprintf(stderr, "Unable to load font: %s ... trying fixed\n", BOLDFONT);

      if((g->boldFont = XLoadQueryFont(g->dpy, FIXED)) == NULL){
         fprintf(stderr, "Unable to load font: %s\n", FIXED);
         return;
         }
      }

/* Which visual do we get? */

   gRedDiv = gGreenDiv = gBlueDiv = gRedShift = gGreenShift = gBlueShift = -1;
   g->depth = DefaultDepth(g->dpy, screen);

   vis = DefaultVisual(g->dpy, screen);
   if (g->depth > EIGHT && vis->class == TrueColor) {
      int i;
      for (i = 0; i < (int)sizeof(vis->blue_mask)*8; i++) {
         if (gBlueShift == -1 && ((vis->blue_mask >> i) & 1))
            gBlueShift = i;
         if ((vis->blue_mask >> i) == 1) {
            gBlueDiv = 8 - i - 1 + gBlueShift;  /* max value is 255, i.e. 8 bits */
            break;
         }
      }
      for (i = 0; i < (int)sizeof(vis->green_mask)*8; i++) {
         if (gGreenShift == -1 && ((vis->green_mask >> i) & 1))
            gGreenShift = i;
         if ((vis->green_mask >> i) == 1) {
            gGreenDiv = 8 - i - 1 + gGreenShift;
            break;
         }
      }
      for (i = 0; i < (int)sizeof(vis->red_mask)*8; i++) {
         if (gRedShift == -1 && ((vis->red_mask >> i) & 1))
            gRedShift = i;
         if ((vis->red_mask >> i) == 1) {
            gRedDiv = 8 - i - 1 + gRedShift;
            break;
         }
      }
      /*
      printf("gRedDiv = %d, gGreenDiv = %d, gBlueDiv = %d, gRedShift = %d, gGreenShift = %d, gBlueShift = %d\n",
             gRedDiv, gGreenDiv, gBlueDiv, gRedShift, gGreenShift, gBlueShift);
      */
   } else if (g->depth > EIGHT)
      g->depth = EIGHT;

   g->pix = XCreatePixmap(g->dpy, RootWindow(g->dpy,screen), g->winX,
   g->winY, g->depth);

/* Everything else we treat as monochrome whether or not
   something better may be supported */

/* Make a vanilla window */

   g->helpWinX =XTextWidth(g->font, LONGESTSTRING, strlen(LONGESTSTRING));
   g->helpWinY = FONTHEIGHT(g->font) * HELPLINES;

   g->helpWin = XCreateSimpleWindow(g->dpy, RootWindow(g->dpy, screen), 0, 0,
      g->helpWinX, g->helpWinY, 0, 0, 0);

   if (parent)
       g->win = XCreateSimpleWindow(g->dpy, parent, 0, 0,
                                    g->winX, g->winY, 0, 0, 0);
   else
       g->win = XCreateSimpleWindow(g->dpy, RootWindow(g->dpy,screen), 0, 0,
                                    g->winX, g->winY, 0, 0, 0);

/* Any user geometry? */

   if(g->Geometry && !useroot){

      x = 0;
      y = 0;
      width = g->winX;
      height = g->winY;
      sizehint.flags = USPosition | USSize;

      XParseGeometry(g->Geometry, &x, &y, &width, &height);

      sizehint.x = x;
      sizehint.y = y;
      sizehint.width  = width;
      sizehint.height = height;
      g->winX = width;
      g->winY = height;

      XResizeWindow(g->dpy, g->win, width, height);
      XSetNormalHints(g->dpy, g->win, &sizehint);
      }

/* Set horizontal and vertical ranges */

   g->winH = (int)(g->winX / 2.0);
   g->winV = (int)(g->winY / 2.0);

/* Make our graphics context */

   g->gc = XCreateGC(g->dpy, g->win, 0x0, NULL);
   g->helpGc = XCreateGC(g->dpy, g->helpWin, 0x0, NULL);

/* Create Tiles for monochrome display */

   for(index1 = 0; index1 < NUMSTIPPLES; index1++){
      g->stipple[index1]= XCreateBitmapFromData(g->dpy, g->win, bits,
      STIPPLESIZE, STIPPLESIZE);
      temp_gc = XCreateGC(g->dpy, g->stipple[index1], 0x0, NULL);
      XSetForeground(g->dpy, temp_gc, 0);
      XFillRectangle(g->dpy, g->stipple[index1], temp_gc, 0, 0, STIPPLESIZE,
      STIPPLESIZE);
      XSetForeground(g->dpy, temp_gc, 1);
      for(index2 = 0; index2 < stipples[index1][0]; index2++){
         XDrawPoint(g->dpy, g->stipple[index1], temp_gc,
         stipples[index1][index2 * 2 + 1], stipples[index1][index2 * 2 + 2]);
         }
      XFreeGC(g->dpy, temp_gc);
      }

   if (!useroot) {
/* We want to have the input focus if we can */

      XSetInputFocus(g->dpy, PointerRoot, RevertToNone, CurrentTime);

/*
   Thanks go to Otmar Lendl for the following bit of code that
   permits the program to work properly with losing window managers
   that don't handle input focus correctly.
*/

      wmhint.input = True;
      wmhint.flags = InputHint;
      XSetWMHints(g->dpy,g->win,&wmhint);
   }

/* Please do not do backing store on the contents of our window */

   attribs.backing_store = NotUseful;
   XChangeWindowAttributes(g->dpy, g->win, CWBackingStore, &attribs);

/* We only want certain kinds of events */

   XSelectInput(g->dpy, g->win, ButtonPressMask | ButtonReleaseMask |
      KeyPressMask | Button1MotionMask | Button2MotionMask |
      StructureNotifyMask | ExposureMask | ColormapChangeMask);

   XSelectInput(g->dpy, g->helpWin, ButtonPressMask | ButtonReleaseMask |
      KeyPressMask | Button1MotionMask | Button2MotionMask |
      StructureNotifyMask | ExposureMask | ColormapChangeMask);

   if (!useroot) {
/* Do not generate expose events */

      XSetGraphicsExposures(g->dpy, g->gc, 0);

/* Name our windows */

      XStoreName(g->dpy, g->win, title);
      XStoreName(g->dpy, g->helpWin, "ROOT://X3D/Help");

   }
/* Some window managers are not friendly, explicitly set the background color */

   XSetWindowBackground(g->dpy, g->win, g->black);

   if(g->depth == ONE){
      OneBitSetColors(g);
      }

   if(g->depth > EIGHT){
      TrueColorSetColors(g);
      }

   if(g->depth == EIGHT){

      NUMCOLORS = 256;

/* Make our colormap */

      g->colormap = XCreateColormap(g->dpy, g->win, DefaultVisual(g->dpy,
      screen), AllocAll);

/* Get the current colormap */

      XGetWindowAttributes(g->dpy, RootWindow(g->dpy,screen), &attributes);

/* Since we only use 16 colors, set all our other entries to the old values.
   Hopefully some other windows might display in true colors */

      for(index1 = 0; index1 < NUMCOLORS; index1++)
         oldColormap[index1].pixel = index1;

      XQueryColors(g->dpy, attributes.colormap, oldColormap, NUMCOLORS);
      XStoreColors(g->dpy, g->colormap, oldColormap, NUMCOLORS);

/* Set up the colormap */

      EightBitSetColors(g);

/* Set our special 12 colors to something */

      XStoreColors(g->dpy, g->colormap, g->cmapColors[0], 256);
      XSetWindowColormap(g->dpy, g->helpWin, g->colormap);
      XSetWindowColormap(g->dpy, g->win, g->colormap);
   }

/* Make the purple rectangle */

   ResetPurpleRectangle(0, 0, g->winX, g->winY, g);


   if (!useroot) {
     /*
      *   Set up some window manager properties (see the ICCCM).
      */

      wm_protocols[0] = XInternAtom (g->dpy, "WM_DELETE_WINDOW", False);
      wm_protocols[1] = XInternAtom (g->dpy, "WM_SAVE_YOURSELF", False);
      XSetWMProtocols (g->dpy, g->win, wm_protocols, 2);
   }

/*
   Make the windows appear.
*/
   XMapWindow(g->dpy, g->win);
   if (!useroot) DisplayMenu(g);
}



static int CheckEvent(Display *display, XEvent *event, char *arg)
/******************************************************************************
   Check an event to see if it is one we are interested in.
   This is used by X to wake up our program once some interesting event
   happens.  Returns: 1 if we are interested, 0 if we are not.
******************************************************************************/
{
   if (display || arg) { } /* use unused arguments */

   if(event == NULL){
      fprintf(stderr, "WARNING: Null event in CheckEvent()!!\n");
      return 0;
   }

   if((event->type == MotionNotify) || (event->type == KeyPress) ||
      (event->type == ConfigureNotify) || (event->type == Expose) ||
      (event->type == ColormapNotify) || (event->type == ClientMessage))
      return 1;

   return 0;
}



static void GetInput(xevent, pointerX, pointerY, command, same, g)
XEvent *xevent;
int *pointerX, *pointerY;
char *command;
int *same;
Ginfo *g;
/******************************************************************************
   Get an interesting event and update the user input information.

   The routine will eventually block waiting for an event if block is 1
   and the no events of interest have shown up.
******************************************************************************/
{
XEvent event;
XSizeHints sizehint;
int  numEvents;
char string[TMPSTRLEN];

/* set command to a meaningless value (hopefully) */

   *command = '\0';

   if (!xevent) {

      do{
          string[0] = '\0';

/* How many events? */

          numEvents = XEventsQueued(g->dpy, QueuedAfterReading);

/* Block to obtain an event yet? */

          if((numEvents == 0) && (g->Block)){

/* If the user falls asleep stop using CPU cycles */

             XIfEvent(g->dpy, &event, CheckEvent, NULL);
             numEvents = 1;
          }else{

/* If we have at least one event , fetch the first event off the queue*/

             if(numEvents)
                XNextEvent(g->dpy,&event);
          }

      }while((numEvents == 0) && (g->Block));

   } else {
      event = *xevent;
      numEvents = 1;
   }

/* Process the events we have obtained (if any) */

   while(numEvents){

      switch(event.type){

         case MotionNotify    :
            if(numEvents == 1){
               *pointerX = (int)event.xmotion.x;
               *pointerY = (int)event.xmotion.y;
            }
            break;

         case KeyPress        :
            if(numEvents == 1){
               XLookupString(&event.xkey,string,TMPSTRLEN,NULL,NULL);
               *command = string[0];
               }
            break;

         case ConfigureNotify :

            if(event.xconfigure.window == g->win){

               g->winX = event.xconfigure.width;
               g->winY = event.xconfigure.height;
               g->winH = (int)(g->winX / 2.0);
               g->winV = (int)(g->winY / 2.0);
               ResetPurpleRectangle(0, 0, g->winX, g->winY, g);
               sizehint.flags  = USSize | USPosition;
               sizehint.width  = g->winX;
               sizehint.height = g->winY;
               XSetNormalHints(g->dpy, g->win, &sizehint);

/* Resize our pix etc... */

               XFreePixmap(g->dpy, g->pix);

               g->pix = XCreatePixmap(g->dpy, g->win, g->winX, g->winY,
               g->depth);

               }
            if(event.xconfigure.window == g->helpWin){
               DisplayMenu(g);
               }
            g->modeChanged = 1;
            *same = 0;
            break;

         case Expose          :
            if(event.xexpose.window == g->helpWin){
               DisplayMenu(g);
               }
            *same = 0;
            break;

         case MapNotify       :

            if(event.xmap.window == g->helpWin){

               sizehint.flags = USPosition | USSize;
               XSetNormalHints(g->dpy, g->helpWin, &sizehint);

               DisplayMenu(g);
               }
            g->modeChanged = 1;
            *same = 0;
            break;

         case ColormapNotify  :

            if(event.xcolormap.colormap == g->colormap){

               if(event.xcolormap.state == ColormapUninstalled){
                  g->mono = 1;
               }else{
                  g->mono = 0;
                  }

              g->modeChanged = 1;
              *same = 0;
              }
            break;

         case ClientMessage:
              if (event.xclient.data.l[0] == (long)wm_protocols[0])
                  /* WM_DELETE_WINDOW */
              {
              /*
               *   XmbufDestroyBuffers( dpy, window );
               *   XDestroyWindow (dpy, window);
               *   XCloseDisplay( dpy );
               */
                  quitApplication = 1;
              }
              else if (event.xclient.data.l[0] == (long)wm_protocols[1])
                  /* WM_SAVE_YOURSELF */
              {

              }
            break;


         default:
            break;
         }

         numEvents--;
         if(numEvents)
            XNextEvent(g->dpy,&event);
      }
}


float deltaMove = 0;

static
int UpdatePosition(event, o, g)
XEvent *event;
Oinfo *o;
Ginfo *g;
/******************************************************************************
   Update the scene position information using user input.

   The routine will eventually block waiting for an event if block is True
   and the no events of interest show up due to the call to GetInput()
******************************************************************************/
{
int same, pointerX, pointerY, dx, dy;
char command;
double X, Y, Z;

   X = Y = Z = 0.0;

   same = 1;

   pointerX = g->oldPointerX;
   pointerY = g->oldPointerY;

   while(same) {

/* dx, dy, dz are the amount to step about each axis every frame
   We want the scene to continue to rotate even if the user does
   not give any new input */

/* Do not forget to put your automatic update variables into this if
   statement.  Be careful somehow you can get MANY bugs with these!  */

      if((o->dX) || (o->dY) || (o->dZ)){
         same = 0;
         g->Block = 0;
      }else
         g->Block = 1;

/* Get the input */

      GetInput(event, &pointerX, &pointerY, &command, &same, g);

/* Fill in code for your favorite keyboard and pointer controls */

/* My default controls */

/* Note: I do not move the origin which the scene is rotated about around.
   You may want to do oX += ???; oY += ???; oZ += ???    */

      switch(command){
         case ' ' : break;

         case 'm' :
         case 'M' : same = 0; g->helpMenu    = !g->helpMenu;
             /*
             if(g->helpMenu == False){
               XUnmapWindow(g->dpy, g->helpWin);
             }else{
               XMapWindow(g->dpy, g->helpWin);
             }
             */
             break;

         case 's' :
         case 'S' : same = 0; g->stereo  = !g->stereo; g->modeChanged = 1;
             break;

         case 'd' :
         case 'D' : same = 0; g->stereoBlue = !g->stereoBlue; g->modeChanged = 1;
             break;

         case 'f' :
         case 'F' : same = 0; g->buffer = !g->buffer; g->modeChanged = 1;
             break;

         case 'o' :
         case 'O' : same = 0; g->Relative = !g->Relative; break;

         case 'w' :
         case 'W' : same = 0; g->renderMode = WIREFRAME; g->modeChanged = 1;
             break;

         case 'e' :
         case 'E' : if(o->numPolys) {
                      same = 0; g->renderMode = HIDDENLINE; g->modeChanged = 1;
                    }
             break;

         case 'r' :
         case 'R' : if(o->numPolys) {
                      same = 0; g->renderMode = SOLID; g->modeChanged = 1;
                    }
             break;

         case 'l' : same = 0; o->tX  -= deltaMove; break;
         case 'j' : same = 0; o->tY  -= deltaMove; break;
         case 'k' : same = 0; o->tY  += deltaMove; break;
         case 'h' : same = 0; o->tX  += deltaMove; break;
         case 'i' : same = 0; o->tZ  += deltaMove; break;
         case 'u' : same = 0; o->tZ  -= deltaMove; break;
         case 'L' : same = 0; o->tX  -= 5*deltaMove; break;
         case 'J' : same = 0; o->tY  -= 5*deltaMove; break;
         case 'K' : same = 0; o->tY  += 5*deltaMove; break;
         case 'H' : same = 0; o->tX  += 5*deltaMove; break;
         case 'I' : same = 0; o->tZ  += 5*deltaMove; break;
         case 'U' : same = 0; o->tZ  -= 5*deltaMove; break;
         case '1' : same = 0; o->dX += 0.02; break;
         case '2' : same = 0; o->dX =  0.0 ; break;
         case '3' : same = 0; o->dX -= 0.02; break;
         case '4' : same = 0; o->dY -= 0.02; break;
         case '5' : same = 0; o->dY =  0.0 ; break;
         case '6' : same = 0; o->dY += 0.02; break;
         case '7' : same = 0; o->dZ += 0.02; break;
         case '8' : same = 0; o->dZ =  0.0 ; break;
         case '9' : same = 0; o->dZ -= 0.02; break;
         case 'x' : same = 0; X -= 0.03; break;
         case 'X' : same = 0; X += 0.03; break;
         case 'y' : same = 0; Y += 0.03; break;
         case 'Y' : same = 0; Y -= 0.03; break;
         case 'z' : same = 0; Z -= 0.03; break;
         case 'Z' : same = 0; Z += 0.03; break;
         case 'a' : same = 0; X -= 0.05; break;
         case 'A' : same = 0; X += 0.05; break;
         case 'b' : same = 0; Y += 0.05; break;
         case 'B' : same = 0; Y -= 0.05; break;
         case 'c' : same = 0; Z -= 0.05; break;
         case 'C' : same = 0; Z += 0.05; break;
         case '[' : same = 0;
                    o->focus += 0.1;
                    if((o->focus > 1.8))
                       o->focus = 1.8;
                    break;
         case ']' : same = 0;
                    o->focus -= 0.1;
                    if((o->focus < -0.8))
                       o->focus = -0.8;
                    break;
         case '{' : same = 0; o->BViewpointX -= 4.0; break;
         case '}' : same = 0; o->BViewpointX += 4.0; break;

         case 'q' :
         case 'Q' : return(1);

         default : {

/* My pointer movement stuff */

/* Only update if the movement was reasonably small */

            dx = pointerX - g->oldPointerX;
            dy = pointerY - g->oldPointerY;

            if((dy * dy <= SMALLMOVEMENT) &&
               (dx * dx <= SMALLMOVEMENT)){

/* Rotate proportionally with the amount the pointer moved */
/* Note: I only control the X and Z axes by the pointer */

               X -= (dy * POINTERRATIO);
               Z -= (dx * POINTERRATIO);
               same = 0;
               }

            g->oldPointerY = pointerY;
            g->oldPointerX = pointerX;
            }
         }
      }

/* Keep angles 0 - 6.28 */
   X = fmod(X + o->dX, TWOPI);
   Y = fmod(Y + o->dY, TWOPI);
   Z = fmod(Z + o->dZ, TWOPI);

/* Fix up the angles */

   if(g->Relative){
      o->X = fmod(X + o->X, TWOPI);
      o->Y = fmod(Y + o->Y, TWOPI);
      o->Z = fmod(Z + o->Z, TWOPI);
   }else{
      CalculateAngles(&(o->X), &(o->Y), &(o->Z), X, Y, Z);
      }

   return quitApplication;
}



static int clipSegment(pX, pY, qX, qY, Pclip, Qclip, H, V)
float *pX, *pY, *qX, *qY;
int Pclip, Qclip;
float H,V;
/******************************************************************************
   Calculate the portion of the projected line segment that is visible.
******************************************************************************/
{
register float PX, PY, QX, QY, dx, dy;

   PX = *pX; QX = *qX;
   PY = *pY; QY = *qY;

   dx = QX - PX;
   dy = QY - PY;

/* Clip P first so it will be somewhere on the screen,
   if we cannot move P on screen the segment is not visible */

/* See x3d.h for the meaning of the clipping flags */

   switch(Pclip){

      case 1 :      /*  00001  */
         clipWithTop(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H))
            return 0;
         break;

      case 2 :      /*  00010  */
         clipWithBottom(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H))
            return 0;
         break;

      case 4 :      /*  00100  */
         clipWithLeftSide(PX, PY, dx, dy, H)
         if((PY < -V) || (PY > V))
            return 0;
         break;

      case 5 :          /*  00101  */
         clipWithTop(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H)){
            clipWithLeftSide(PX, PY, dx, dy, H)
            if((PY < -V) || (PY > V))
               return 0;
            }
         break;

      case 6 :          /*  00110  */
         clipWithBottom(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H)){
            clipWithLeftSide(PX, PY, dx, dy, H)
            if((PY < -V) || (PY > V))
               return 0;
            }
         break;

      case 8 :          /*  01000  */
         clipWithRightSide(PX, PY, dx, dy, H)
         if((PY < -V) || (PY > V))
            return 0;
         break;

      case 9 :          /*  01001  */
         clipWithTop(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H)){
            clipWithRightSide(PX, PY, dx, dy, H)
            if((PY < -V) || (PY > V))
               return 0;
            }
         break;

      case 10 :          /*  01010  */
         clipWithBottom(PX, PY, dx, dy, V)
         if((PX < -H) || (PX > H)){
            clipWithRightSide(PX, PY, dx, dy, H)
            if((PY < -V) || (PY > V))
               return 0;
            }
         break;

   }

/* P is now somewhere on screen, calculate where Q should be */

   switch(Qclip){

      case 1 :        /*  00001  */
         clipWithTop(QX, QY, dx, dy, V)
         break;

      case 2 :        /*  00010  */
         clipWithBottom(QX, QY, dx, dy, V)
         break;

      case 4 :        /*  00100  */
         clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 5 :        /*  00101  */
         clipWithTop(QX, QY, dx, dy, V)
         if(QX < -H)
            clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 6 :        /*  00110  */
         clipWithBottom(QX, QY, dx, dy, V)
         if(QX < -H)
            clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 8 :        /*  01000  */
         clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 9 :        /*  01001  */
         clipWithTop(QX, QY, dx, dy, V)
         if(QX > H)
            clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 10 :        /*  01010  */
         clipWithBottom(QX, QY, dx, dy, V)
         if(QX > H)
            clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 21 :        /*  10101  */
         clipWithTop(QX, QY, dx, dy, V)
         if(QX < -H)
            clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 22 :        /*  10110  */
         clipWithBottom(QX, QY, dx, dy, V)
         if(QX < -H)
            clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 23 :        /*  10111  */
         if(QY < PY)
            clipWithTop(QX, QY, dx, dy, V)
         else
            clipWithBottom(QX, QY, dx, dy, V)
         if(QX < -H)
            clipWithLeftSide(QX, QY, dx, dy, H)
         break;

      case 25 :        /*  11001  */
         clipWithTop(QX, QY, dx, dy, V)
         if(QX > H)
            clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 26 :        /*  11010  */
         clipWithBottom(QX, QY, dx, dy, V)
         if(QX > H)
            clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 27 :        /*  11011  */
         if(QY < PY)
            clipWithTop(QX, QY, dx, dy, V)
         else
            clipWithBottom(QX, QY, dx, dy, V)
         if(QX > H)
            clipWithRightSide(QX, QY, dx, dy, H)
         break;

      case 29 :        /*  11101  */
         if(QX < PX)
            clipWithRightSide(QX, QY, dx, dy, H)
         else
            clipWithLeftSide(QX, QY, dx, dy, H)
         if(QY > V)
            clipWithTop(QX, QY, dx, dy, V)
         break;

      case 30 :        /*  11110  */
         if(QX < PX)
            clipWithRightSide(QX, QY, dx, dy, H)
         else
            clipWithLeftSide(QX, QY, dx, dy, H)
         if(QY < -V)
            clipWithBottom(QX, QY, dx, dy, V)
         break;

      case 31 :        /*  11111  */
         if(QX < PX)
            clipWithRightSide(QX, QY, dx, dy, H)
         else
            clipWithLeftSide(QX, QY, dx, dy, H)

         if((QY < -V) || (QY > V)) {
            if(*qY < PY)
               clipWithTop(QX, QY, dx, dy, V)
            else
               clipWithBottom(QX, QY, dx, dy, V)
         }
         break;

   }

   *pX = PX; *qX = QX;
   *pY = PY; *qY = QY;
   return 1;
}



static void clip(o, g)
Oinfo *o;
Ginfo *g;
/******************************************************************************
   Clip a list of segments.
******************************************************************************/
{
register int PClipFlags, QClipFlags, Pclip, Qclip, Tclip;
register float H, V;
register short pX, pY, qX, qY;
float PX, PY, QX, QY;
segment *seg, *lastSeg;
point *P, *Q, *T;
XSegment *red, *blue;
long *redCol;

   lastSeg = &(o->segs[o->numSegs]);
       seg = o->segs;

   H = (float)g->winH;
   V = (float)g->winV;

   red  = &(g->redSegments[g->numberRed]);
   redCol = &(g->redColors[g->numRedColors]);

   if((g->mono) || ((g->stereo) && (!(g->stereoBlue))) ||
   ((g->renderMode == WIREFRAME) && (!g->stereo))){

      if(o->objClip){

/* For every segment ... */

         while(seg < lastSeg){

            P = seg->P;
            if (P == 0) continue;
            PClipFlags = P->ClipFlags;
            Q = seg->Q;
            if (Q == 0) continue;
            QClipFlags = Q->ClipFlags;

/* Optimization for best case */

            if((PClipFlags | QClipFlags) == 0){
               if (seg->color) *redCol = seg->color->value;
               else            *redCol = 0;
               redCol++;
               ((xsegment *)red)->P = P->R;
               ((xsegment *)red)->Q = Q->R;
               red++;
            }else{

/* Red segments */

/* Shuffle the bits around so we get the right configuration
   for the clipping function */

               Pclip = (PClipFlags & RBmask) | ((PClipFlags & RLeftRight) >> 3);
               Qclip = (QClipFlags & RBmask) | ((QClipFlags & RLeftRight) >> 3);

               if((Qclip | Pclip) == 0){

               if (seg->color) *redCol = seg->color->value;
               else            *redCol = 0;
               redCol++;
               red->x1 = P->R.x;
               red->y1 = P->R.y;
               red->x2 = Q->R.x;
               red->y2 = Q->R.y;
               red++;

               }else{

                  if((Qclip & Pclip) == 0){

/* We make P be a point in front of us if there is one in front,
   (it simplifies the code) */

                     if((Pclip > Qclip) || (Pclip & Behind)){
                        T = P; P = Q; Q = T;
                        Tclip = Pclip; Pclip = Qclip; Qclip = Tclip;
                        }

                     PX = P->RX - H; PY = -(P->Y - V);
                     QX = Q->RX - H; QY = -(Q->Y - V);

                     if(clipSegment(&PX, &PY, &QX, &QY, Pclip, Qclip, H, V)){

                        if (seg->color) *redCol = seg->color->value;
                        else            *redCol = 0;
                        redCol++;
                        red->x1 = (short)(PX + H);
                        red->y1 = (short)(V - PY);
                        red->x2 = (short)(QX + H);
                        red->y2 = (short)(V - QY);
                        red++;

                        }
                     }
                  }
               }
         seg++;
         }
      }else{

/* Optimization for object completely visible */

         if((g->renderMode == WIREFRAME) && (g->stereo)){
            while(seg < lastSeg){
                  ((xsegment *)red)->P = seg->P->R;
                  ((xsegment *)red)->Q = seg->Q->R;
                  red++;
                  seg++;
               }
         }else{
            while(seg < lastSeg){

                  if (seg->color) *redCol = seg->color->value;
                  else            {*redCol = 0; break;}
                  redCol++;
                  ((xsegment *)red)->P = seg->P->R;
                  ((xsegment *)red)->Q = seg->Q->R;
                  red++;
                  seg++;
               }
            }
         }
   }else{
      blue = &(g->blueSegments[g->numberBlue]);

      if(o->objClip){
         while(seg < lastSeg){

            P = seg->P; Q = seg->Q;
            PClipFlags = P->ClipFlags;
            QClipFlags = Q->ClipFlags;

/* Optimization for best case */

            if((PClipFlags | QClipFlags) == 0){

               pX = P->R.x; pY = P->R.y;
               qX = Q->R.x; qY = Q->R.y;
               red->x1 = pX;
               red->y1 = pY;
               red->x2 = qX;
               red->y2 = qY;
               red++;
               pX = P->sBX; qX = Q->sBX;
               blue->x1 = pX;
               blue->y1 = pY;
               blue->x2 = qX;
               blue->y2 = qY;
               blue++;

            }else{

/* Red segments */

/* Shuffle the bits around so we get the right configuration
   for the clipping function */

               Pclip = (PClipFlags & RBmask) | ((PClipFlags & RLeftRight) >> 3);
               Qclip = (QClipFlags & RBmask) | ((QClipFlags & RLeftRight) >> 3);

               if((Qclip | Pclip) == 0){

               red->x1 = P->R.x;
               red->y1 = P->R.y;
               red->x2 = Q->R.x;
               red->y2 = Q->R.y;
               red++;

               }else{

                  if((Qclip & Pclip) == 0){

/* We make P be a point in front of us if there is one in front,
   (it simplifies the code) */

                     if((Pclip > Qclip) || (Pclip & Behind)){
                        T = P; P = Q; Q = T;
                        Tclip = Pclip; Pclip = Qclip; Qclip = Tclip;
                        }

                     PX = P->RX - H; PY = -(P->Y - V);
                     QX = Q->RX - H; QY = -(Q->Y - V);

                     if(clipSegment(&PX, &PY, &QX, &QY, Pclip, Qclip, H, V)){

                        red->x1 = (short)(PX + H);
                        red->y1 = (short)(V - PY);
                        red->x2 = (short)(QX + H);
                        red->y2 = (short)(V - QY);
                        red++;

                        }

                     }
                  }

/* Blue segments */

               PClipFlags = P->ClipFlags;
               QClipFlags = Q->ClipFlags;

/* Shuffle the bits around so we get the right configuration
   for the clipping function */

               Pclip = (PClipFlags & Rmask);
               Qclip = (QClipFlags & Rmask);

               if((Qclip | Pclip) == 0){

                  blue->x1 = P->sBX;
                  blue->y1 = P->R.y;
                  blue->x2 = Q->sBX;
                  blue->y2 = Q->R.y;
                  blue++;

               }else{
                  if((Qclip & Pclip) == 0){

/* We make P be a point in front of us if there is one in front,
   (it simplifies the code) */

                     if((Pclip > Qclip) || (Pclip & Behind)){
                        T = P; P = Q; Q = T;
                        Tclip = Pclip; Pclip = Qclip; Qclip = Tclip;
                        }

/* check the bits, clip if necessary, and add the segment to the
   appropriate buffer if visible */

                     PX = P->BX - H; PY = -(P->Y - V);
                     QX = Q->BX - H; QY = -(Q->Y - V);

                     if(clipSegment(&PX, &PY, &QX, &QY, Pclip, Qclip, H, V)){

                        blue->x1 = (short)(PX + H);
                        blue->y1 = (short)(V - PY);
                        blue->x2 = (short)(QX + H);
                        blue->y2 = (short)(V - QY);
                        blue++;

                        }
                     }
                  }
               }
            seg++;
         }
      }else{

/* Optimization for object completely visible */

         while(seg < lastSeg){
            P = seg->P; Q = seg->Q;
            pX = P->R.x; pY = P->R.y;
            qX = Q->R.x; qY = Q->R.y;
            red->x1 = pX;
            red->y1 = pY;
            red->x2 = qX;
            red->y2 = qY;
            red++;
            pX = P->sBX; qX = Q->sBX;
            blue->x1 = pX;
            blue->y1 = pY;
            blue->x2 = qX;
            blue->y2 = qY;
            blue++;
            seg++;
            }

         }

      g->numberBlue = blue - g->blueSegments;
   }

   g->numRedColors = redCol - g->redColors;
   g->numberRed = red - g->redSegments;
}


#ifdef USE_INTS

/* The REDROTATE macro does 3 rotates, 2 translates, 1 projection */

#define REDROTATE                                                         \
      r8 = p0->x + c16;                                                   \
      r5 = p0->y + c17;                                                   \
      r2 = ((r8 * c5 + r5 * c6) >> SHIFT);                                \
      r7 = ((r5 * c5 - r8 * c6) >> SHIFT);                                \
      r8 = p0->z + c18;                                                   \
      r5 = ((r7 * c3 + r8 * c4) >> SHIFT) + c12;                          \
      r6 = ((r8 * c3 - r7 * c4) >> SHIFT);                                \
      r7 = c7 + ((((r6 * c1 - r2 * c2) >> SHIFT) + c13) * c0) / r5;       \
      r8 = c9 - ((((r2 * c1 + r6 * c2) >> SHIFT) + c11) * c0) / r5;       \
      r5 = (c0 * 8192) / r5

/* Project the blue point too */

#define STEREOROTATE                                                      \
      r8 = p0->x + c16;                                                   \
      r5 = p0->y + c17;                                                   \
      r2 = ((r8 * c5 + r5 * c6) >> SHIFT);                                \
      r7 = ((r5 * c5 - r8 * c6) >> SHIFT);                                \
      r8 = p0->z + c18;                                                   \
      r5 = ((r7 * c3 + r8 * c4) >> SHIFT) + c12;                          \
      r6 = ((r8 * c3 - r7 * c4) >> SHIFT);                                \
      r7 = c7 + ((((r6 * c1 - r2 * c2) >> SHIFT) + c13) * c0) / r5;       \
      r8 = c9 - ((((r2 * c1 + r6 * c2) >> SHIFT) + c11) * c0) / r5;       \
      r9 = r8 - c15 - (c0 * c14) / r5;                                    \
      r5 = (c0 * 8192) / r5

#else

/* The REDROTATE macro does 3 rotates, 2 translates, 1 projection */

#define REDROTATE                                                         \
      r8 = p0->x + c16;                                                   \
      r5 = p0->y + c17;                                                   \
      r2 = r8 * c5 + r5 * c6;                                             \
      r7 = r5 * c5 - r8 * c6;                                             \
      r8 = p0->z + c18;                                                   \
      r5 = c0 / (r7 * c3 + r8 * c4 + c12);                                \
      r6 = r8 *  c3 - r7 * c4;                                            \
      r7 = c7 + (r6 * c1 - r2 * c2 + c13) * r5;                           \
      r8 = c9 - (r2 * c1 + r6 * c2 + c11) * r5

#define STEREOROTATE                                                      \
      REDROTATE;                                                          \
      r9 = r8 - c15 - r5 * c14

#endif


/* Set all the red clipping flags */

#define REDCLIPFLAGS                                                      \
     (r5 > 0) * ALLmask ^ ((r7 < 0) | (r7 > c10) << ClipWithBottom |      \
     (r8 > c8) << RClipWithRight    | (r8 <   0) << RClipWithLeft)

/* Set all the red and blue clipping flags */

#define STEREOCLIPFLAGS                                                   \
     (r5 > 0) * ALLmask ^ ((r7 < 0) | (r7 > c10) << ClipWithBottom |      \
     (r8 > c8) << RClipWithRight    | (r8 <   0) << RClipWithLeft  |      \
     (r9 > c8) << BClipWithRight    | (r9 <   0) << BClipWithLeft)



static void rotate(o, g)
Oinfo *o;
Ginfo *g;
/******************************************************************************
   Rotate, project, and set the clipping flags for a list of points.
******************************************************************************/
{
register number r2,r5,r6,r7,r8,r9;
register point *p0;
register number c1,c2,c3,c4,c5,c6;
register point  *p1;
register number c0,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18;
register int    objClip;
register short  RX, BX;

   p0 = o->bounds; p1 = &(o->bounds[NUMBOUNDS]); c0 = (number)o->viewpointY;
   c1 = (number)(cos(o->Y) * TRIG_ADJ); c2 = (number)(sin(o->Y) * TRIG_ADJ);
   c3 = (number)(cos(o->X) * TRIG_ADJ); c4 = (number)(sin(o->X) * TRIG_ADJ);
   c5 = (number)(cos(o->Z) * TRIG_ADJ); c6 = (number)(sin(o->Z) * TRIG_ADJ);
   c7 = (number)g->winH;              c8 = (number)(2.0 * g->winH);
   c9 = (number)g->winV;             c10 = (number)(2.0 * g->winV);
  c11 = (number)o->tX;               c12 = (number)(o->tY - o->viewpointY);
  c13 = (number)o->tZ;                       c14 = (number)o->BViewpointX;
  c15 = (number)(o->BViewpointX * o->focus); c16 = (number)-o->oX;
  c17 = (number)-o->oY;                      c18 = (number)-o->oZ;

   objClip = 0;
   o->Hmin2 = o->Hmin1; o->Vmin2 = o->Vmin1;
   o->Hmax2 = o->Hmax1; o->Vmax2 = o->Vmax1;
   o->Hmin1 = o->Hmin; o->Vmin1 = o->Vmin;
   o->Hmax1 = o->Hmax; o->Vmax1 = o->Vmax;
   o->Hmin = c8; o->Vmin = c10;
   o->Hmax =  0; o->Vmax =  0;

/* What happens to the bounding box? */

   while(p0 != p1){
      STEREOROTATE;

      if(STEREOCLIPFLAGS)
         objClip = 1;

      if((int)r7 < o->Vmin){ o->Vmin = (int)r7; }
      if((int)r7 > o->Vmax){ o->Vmax = (int)r7; }

      if((int)r8 < o->Hmin){ o->Hmin = (int)r8; }
      if((int)r8 > o->Hmax){ o->Hmax = (int)r8; }

      if((g->stereo) && (g->stereoBlue)){
         if((int)r9 < o->Hmin){ o->Hmin = (int)r9; }
         if((int)r9 > o->Hmax){ o->Hmax = (int)r9; }
         }

      p0++;
      }

/* Update screen clearing, pix copying variables */

   if((g->modeChanged) || (objClip)){
      o->Hmin = o->Hmin1 = o->Hmin2 = 0;
      o->Vmin = o->Vmin1 = o->Vmin2 = 0;
      o->Hmax = o->Hmax1 = o->Hmax2 = g->winX;
      o->Vmax = o->Vmax1 = o->Vmax2 = g->winY;
   }else{
      if(o->Hmin < 0){o->Hmin = 0;}
      if(o->Vmin < 0){o->Vmin = 0;}
      if(o->Hmax > c8){o->Hmax = c8;}
      if(o->Vmax > c10){o->Vmax = c10;}

      if(o->Hmin1 < o->Hmin2){o->Hmin2 = o->Hmin1;}
      if(o->Vmin1 < o->Vmin2){o->Vmin2 = o->Vmin1;}
      if(o->Hmax1 > o->Hmax2){o->Hmax2 = o->Hmax1;}
      if(o->Vmax1 > o->Vmax2){o->Vmax2 = o->Vmax1;}
      }

   o->fillX = o->Hmin2; o->fillY = o->Vmin2;
   o->fillWidth  = o->Hmax2 - o->Hmin2 + 1;
   o->fillHeight = o->Vmax2 - o->Vmin2 + 1;

   if(o->Hmin < o->Hmin2){o->Hmin2 = o->Hmin;}
   if(o->Vmin < o->Vmin2){o->Vmin2 = o->Vmin;}
   if(o->Hmax > o->Hmax2){o->Hmax2 = o->Hmax;}
   if(o->Vmax > o->Vmax2){o->Vmax2 = o->Vmax;}

   o->copyX = o->Hmin2; o->copyY = o->Vmin2;
   o->copyWidth  = o->Hmax2 - o->Hmin2 + 1;
   o->copyHeight = o->Vmax2 - o->Vmin2 + 1;

   o->objClip = objClip;
   p0 = o->points;
   p1 = &(o->points[o->numPoints]);

   if(objClip){

/* The object is not totally visible, do clipping */

      if((g->stereo) && (g->stereoBlue)){
         if(g->renderMode == WIREFRAME){
            while(p0 != p1){
               STEREOROTATE;
               p0->Y  = (float)r7;   p0->R.y = (short)r7;
               p0->RX = (float)r8;   p0->R.x = (short)r8;
               p0->BX = (float)r9;   p0->sBX = (short)r9;
               p0->ClipFlags = STEREOCLIPFLAGS;
               p0++;
               }
         }else{

            while(p0 != p1){
               STEREOROTATE;
               p0->dist = (float)r5;
               p0->Y  = (float)r7;   p0->R.y = (short)r7;
               p0->RX = (float)r8;   p0->R.x = (short)r8;
               p0->BX = (float)r9;   p0->sBX = (short)r9;
               p0->ClipFlags = STEREOCLIPFLAGS;
               p0++;
               }
            }
      }else{
         if(g->renderMode == WIREFRAME){
            while(p0 != p1){
               REDROTATE;
               p0->Y  = (float)r7;   p0->R.y = (short)r7;
               p0->RX = (float)r8;   p0->R.x = (short)r8;
               p0->ClipFlags = REDCLIPFLAGS;
               p0++;
               }
         }else{

            while(p0 != p1){
               REDROTATE;
               p0->dist = (float)r5;
               p0->Y  = (float)r7;   p0->R.y = (short)r7;
               p0->RX = (float)r8;   p0->R.x = (short)r8;
               p0->ClipFlags = REDCLIPFLAGS;
               p0++;
               }
            }
         }
   }else{

/* The object is totally visible, skip clipping */

      if((g->stereo) && (g->stereoBlue)){
         if(g->renderMode == WIREFRAME){
            while(p0 != p1){
               STEREOROTATE;
               p0->R.y = (short)r7;
               p0->R.x = (short)r8;
               p0->sBX = (short)r9;
               p0++;
               }
         }else{

            while(p0 != p1){
               STEREOROTATE;
               p0->dist = (float)r5;
               p0->R.y = (short)r7;

               RX = r8;
               p0->R.x = (float)RX;

               BX = r9;
               p0->sBX = (float)BX;

               p0++;
               }
            }
      }else{
         if(g->renderMode == WIREFRAME){
            while(p0 != p1){
               REDROTATE;
               p0->R.y = (short)r7;
               p0->R.x = (short)r8;
               p0++;
               }
         }else{

            while(p0 != p1){
               REDROTATE;
               p0->dist = (float)r5;

               p0->R.y = (short)r7;
               RX = (short)r8;
               p0->R.x = (float)RX;

               p0++;
               }
            }
         }
      }
}




static void DrawSegments(display, win, gc, segs1, numSegs, g)
Display *display;
Window win;
GC gc;
XSegment segs1[];
int numSegs;
Ginfo *g;
/******************************************************************************
   Thanks to Mark Cook for the suggestion to pay attention the the
   maximum request size of the X server!
******************************************************************************/
{
int requestSize, evenAmount, remainder1, index1;

   requestSize = g->requestSize;
   evenAmount = (numSegs / requestSize) * requestSize;
   remainder1  = numSegs - evenAmount;

   index1 = 0;

   while(index1 < evenAmount){
      XDrawSegments(display, win, gc, &segs1[index1], requestSize);
      index1 += requestSize;
      }

   if(remainder1 > 0)
      XDrawSegments(display, win, gc, &segs1[index1], remainder1);
}



static void DrawLines(o, g, mode)
Oinfo *o;
Ginfo *g;
int mode;
/******************************************************************************
   Draw lines for the three display modes.
******************************************************************************/
{
Drawable dest;
long lastColor;
int lastChange, index1;

   dest = g->dest;

   switch(mode){

      case BW:
         XSetForeground(g->dpy, g->gc, g->black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);
         XSetForeground(g->dpy, g->gc, g->white);
         DrawSegments  (g->dpy, dest,  g->gc, g->redSegments, g->numberRed, g);
         break;

      case STEREO:

         XSetForeground(g->dpy, g->gc,  (long)g->Black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);

         XSetForeground(g->dpy, g->gc, (long)g->Red);
         DrawSegments(g->dpy, dest, g->gc, g->redSegments, g->numberRed, g);

         if(g->stereoBlue){

            XSetFunction  (g->dpy, g->gc,  GXor);

            XSetForeground(g->dpy, g->gc,  (long)g->Blue);
            DrawSegments(g->dpy, dest, g->gc, g->blueSegments, g->numberBlue,
            g);

            XSetFunction (g->dpy, g->gc, GXcopy);
            }

         break;

      case COLOR:

         XSetForeground(g->dpy, g->gc, g->Black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);

         lastChange = 4;
         lastColor = g->redColors[lastChange];

         for(index1 = 5; index1 < g->numberRed; index1++){
            if(g->redColors[index1] != lastColor){
               XSetForeground(g->dpy, g->gc, lastColor);
               DrawSegments(g->dpy, dest,  g->gc, &(g->redSegments[lastChange]),
               index1 - lastChange, g);
               lastChange = index1;
               lastColor = g->redColors[lastChange];
               }
            }

         XSetForeground(g->dpy, g->gc, lastColor);
         DrawSegments(g->dpy, dest,  g->gc, &(g->redSegments[lastChange]),
         g->numberRed - lastChange, g);

         break;

      default:
         break;
      }
}



static void DrawHiddenLines(o, g, mode)
Oinfo *o;
Ginfo *g;
int mode;
/******************************************************************************
   Draw polygon outlines using painter algorithm for the three display modes.
******************************************************************************/
{
register int index1, npoints, numPolys;
register polygon *poly, **list1;
register point  **pointPtr, **lastPointPtr;
Drawable dest;

_XPoint points1[512], *XPointPtr;
   void *ptrp = points1;
   dest = g->dest;
   numPolys = o->numPolys;
   list1     = o->list;

   sort(list1, numPolys);

   switch(mode){

      case BW:

         XSetForeground(g->dpy, g->gc, g->black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);

         for(index1 = 0; index1 < numPolys; index1++){

            poly = list1[index1];
            XPointPtr = points1;
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XSetForeground(g->dpy, g->gc, g->black);
            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);
            points1[npoints] = points1[0];
            XSetForeground(g->dpy, g->gc, g->white);
            XDrawLines(g->dpy, dest, g->gc, ptrp, npoints + 1,
            CoordModeOrigin);
            }

         XSetFillStyle(g->dpy, g->gc, FillSolid);
         break;

      case STEREO:

         XSetForeground(g->dpy, g->gc, g->stereoBlack);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);

         XSetPlaneMask(g->dpy, g->gc, g->redMask);

         for(index1 = 0; index1 < numPolys; index1++){
            poly = list1[index1];
            XPointPtr = points1;
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XSetForeground(g->dpy, g->gc, g->stereoBlack);
            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);
            points1[npoints] = points1[0];
            XSetForeground(g->dpy, g->gc, poly->color->stereoColor);
            XDrawLines(g->dpy, dest, g->gc, ptrp, npoints + 1,
            CoordModeOrigin);
            }

         if(g->stereoBlue){
            XSetPlaneMask(g->dpy, g->gc, g->blueMask);

            for(index1 = 0; index1 < numPolys; index1++){
               poly = list1[index1];
               XPointPtr = points1;
               npoints      =   poly->numPoints;
               pointPtr     =   poly->points;
               lastPointPtr = &(poly->points[npoints]);

               while(pointPtr < lastPointPtr){
                  XPointPtr->x = (*pointPtr)->sBX;
                  XPointPtr->y = (*pointPtr)->R.y;
                  XPointPtr++;
                  pointPtr++;
                  }

               XSetForeground(g->dpy, g->gc, g->stereoBlack);
               XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
               CoordModeOrigin);
               points1[npoints] = points1[0];
               XSetForeground(g->dpy, g->gc, poly->color->stereoColor);
               XDrawLines(g->dpy, dest, g->gc, ptrp, npoints + 1,
               CoordModeOrigin);
               }
            }
            XSetPlaneMask(g->dpy, g->gc, AllPlanes);
         break;

      case COLOR:

         XSetForeground(g->dpy, g->gc, g->Black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);

         for(index1 = 0; index1 < numPolys; index1++){
            poly = list1[index1];
            XPointPtr = points1;
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XSetForeground(g->dpy, g->gc, g->Black);
            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);

            points1[npoints] = points1[0];
            XSetForeground(g->dpy, g->gc, poly->color->value);
            XDrawLines(g->dpy, dest, g->gc, ptrp, npoints + 1,
            CoordModeOrigin);
            }

         break;

      default:
         break;
      }
}



static void DrawPolys(o, g, mode)
Oinfo *o;
Ginfo *g;
int mode;
/******************************************************************************
   Draw polygons using painter algorithm for the three display modes.
******************************************************************************/
{
register int index1, npoints, numPolys;
register polygon *poly, **list1;
register point  **pointPtr, **lastPointPtr;
Drawable dest;
_XPoint points1[512], *XPointPtr;
long lastColor;

   void *ptrp = points1;
   dest = g->dest;
   numPolys = o->numPolys;
   list1     = o->list;

   sort(list1, numPolys);

   switch(mode){

      case BW:

         XSetForeground(g->dpy, g->gc, g->black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);
         XSetForeground(g->dpy, g->gc, g->white);
         XSetBackground(g->dpy, g->gc, g->black);
         XSetFillStyle(g->dpy, g->gc, FillOpaqueStippled);

         for(index1 = 0; index1 < numPolys; index1++){

            poly = list1[index1];
            XPointPtr = points1;
            XSetStipple(g->dpy, g->gc, g->stipple[poly->color->stipple]);
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);
            }

         XSetFillStyle(g->dpy, g->gc, FillSolid);
         break;

      case STEREO:

         XSetForeground(g->dpy, g->gc, g->stereoBlack);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);
         lastColor = g->stereoBlack;

         XSetPlaneMask(g->dpy, g->gc, g->redMask);

         for(index1 = 0; index1 < numPolys; index1++){
            poly = list1[index1];
            XPointPtr = points1;
            if(poly->color->stereoColor != lastColor){
               XSetForeground(g->dpy, g->gc, poly->color->stereoColor);
               lastColor = poly->color->stereoColor;
               }
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);
            }

         if(g->stereoBlue){
            XSetPlaneMask(g->dpy, g->gc, g->blueMask);

            for(index1 = 0; index1 < numPolys; index1++){
               poly = list1[index1];
               XPointPtr = points1;
               if(poly->color->stereoColor != lastColor){
                  XSetForeground(g->dpy, g->gc, poly->color->stereoColor);
                  lastColor = poly->color->stereoColor;
                  }
               npoints      =   poly->numPoints;
               pointPtr     =   poly->points;
               lastPointPtr = &(poly->points[npoints]);

               while(pointPtr < lastPointPtr){
                  XPointPtr->x = (*pointPtr)->sBX;
                  XPointPtr->y = (*pointPtr)->R.y;
                  XPointPtr++;
                  pointPtr++;
                  }

               XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
               CoordModeOrigin);
               }
            }
            XSetPlaneMask(g->dpy, g->gc, AllPlanes);
         break;

      case COLOR:

         XSetForeground(g->dpy, g->gc, g->Black);
         XFillRectangle(g->dpy, dest,  g->gc, o->fillX, o->fillY,
         o->fillWidth, o->fillHeight);
         lastColor = g->Black;

         for(index1 = 0; index1 < numPolys; index1++){
            poly = list1[index1];
            XPointPtr = points1;
            if(poly->color->value != lastColor){
               XSetForeground(g->dpy, g->gc, poly->color->value);
               lastColor = poly->color->value;
               }
            npoints      =   poly->numPoints;
            pointPtr     =   poly->points;
            lastPointPtr = &(poly->points[npoints]);

            while(pointPtr < lastPointPtr){
               *XPointPtr = (*pointPtr)->R;
               XPointPtr++;
               pointPtr++;
               }

            XFillPolygon(g->dpy, dest, g->gc, ptrp, npoints, Convex,
            CoordModeOrigin);
            }

         break;

      default:
         break;
      }
}



static void BeginImage(Oinfo *o, Ginfo *g)
/******************************************************************************
   Prepare to draw some x3d objects.
******************************************************************************/
{

/* Try to get rid of a few of the mode change glitches (several exist yet) */
   if (o) { }  /* use unused argument */
   if(g->modeChanged){
      XSetPlaneMask(g->dpy, g->gc, AllPlanes);
      if((g->mono) || (g->depth == ONE)){
         XSetForeground(g->dpy, g->gc, g->black);
      }else{
         if(g->stereo){
            XSetForeground(g->dpy, g->gc,  (long)g->Black);
         }else{
            XSetForeground(g->dpy, g->gc, g->Black);
            }
         }

      XFillRectangle(g->dpy, g->pix,  g->gc, 0, 0, g->winX, g->winY);
      XFillRectangle(g->dpy, g->win,  g->gc, 0, 0, g->winX, g->winY);
      }

   if(g->buffer){
      if((g->mono) || (g->depth != EIGHT)){
         g->dest = g->pix;
      }else{
         if((g->numColors > BUFFER_CMAP) || (((g->renderMode == SOLID) ||
         (g->renderMode == HIDDENLINE)) && (g->stereo))){
            g->dest = g->pix;
         }else{
            g->dest = g->win;

            if(g->ColorSelect){
               XSetPlaneMask(g->dpy, g->gc, BUFFER1);
            }else{
               XSetPlaneMask(g->dpy, g->gc, BUFFER0);
               }
            }
         }
   }else{
      g->dest = g->win;
      }
}



static void DrawObject(o, g)
Oinfo *o;
Ginfo *g;
/******************************************************************************
   Draw an x3d objext to the screen.  Be sure to draw the objects back to
   front when calling this function.
******************************************************************************/
{

/* rotate the object */

   rotate(o, g);

/* Clip against the screen edges */

   clip(o, g);

/* Draw in the proper render mode */

   switch(g->renderMode){

      case WIREFRAME:
         if((g->mono) || (g->depth == ONE)){
            DrawLines(o, g, BW);
         }else{
            if(g->stereo){
               DrawLines(o, g, STEREO);
            }else{
               DrawLines(o, g, COLOR);
               }
            }
         break;

      case HIDDENLINE:
         if((g->mono) || (g->depth == ONE)){
            DrawHiddenLines(o, g, BW);
         }else{
            if(g->stereo){
               DrawHiddenLines(o, g, STEREO);
            }else{
               DrawHiddenLines(o, g, COLOR);
               }
            }
         break;

      case SOLID:
         if((g->mono) || (g->depth == ONE)){
            DrawPolys(o, g, BW);
         }else{
            if(g->stereo){
               DrawPolys(o, g, STEREO);
            }else{
               DrawPolys(o, g, COLOR);
               }
            }
         break;

      default :
         fprintf(stderr, "Unknown Render Mode!\n");
         return;
         break;
      }

/* Reset the number of lines (We always keep the purple rectangle) */

   g->numberRed  = 4;
   g->numberBlue = 4;
   g->numRedColors = 4;
   g->modeChanged = 0;
}



static void EndImage(o, g)
Oinfo *o;
Ginfo *g;
/******************************************************************************
   Finish drawing x3d objects.
******************************************************************************/
{

/* Colormap double buffer? */

   if((g->depth == EIGHT) && (!(g->mono)) && (g->numColors <= BUFFER_CMAP) &&
   (!((g->stereo) && (g->renderMode == HIDDENLINE))) &&
   (!((g->stereo) && (g->renderMode == SOLID)))){
      g->ColorSelect = !g->ColorSelect;

      if((g->stereo) && (g->renderMode == WIREFRAME) && (!(g->modeChanged))){
         XStoreColors(g->dpy, g->colormap, g->wireframeColors[g->ColorSelect],
         12);
      }else{
         XStoreColors(g->dpy, g->colormap, g->cmapColors[g->ColorSelect], 256);
         }
   }else{

/* Stereo solid, wireframe? */

      if(g->depth == EIGHT){
         if(((g->renderMode == SOLID) || (g->renderMode == HIDDENLINE))
         && (g->stereo)){
            XStoreColors(g->dpy, g->colormap, g->cmapColors[2], 256);
         }else{
            XStoreColors(g->dpy, g->colormap, g->cmapColors[g->ColorSelect],
            256);
            }
         }

/* generic */

      if(g->buffer){
         XCopyArea(g->dpy, g->pix, g->win, g->gc, o->copyX, o->copyY,
         o->copyWidth, o->copyHeight, o->copyX, o->copyY);
         }
      }

   XFlush(g->dpy);
}




void MakePolygonArray()
/******************************************************************************
   Make polygon pointer array
******************************************************************************/
{

    /*
     *  Make polygon pointer array
     */

    int index1, index2, i;
    point *prevPoint;
    segment *tmpSeg;


    if (gSize3D.numPolys) {
         /* sort use the last space as a place holder */
        list = (polygon **) calloc(gSize3D.numPolys+1, sizeof(polygon *));
        if(!list){
            puts("Unable to allocate memory for pointer list !");
            return;
        }
    }
    /* This is only for preventing 'Bus error' */
    else list = (polygon **) calloc(2, sizeof(polygon *));

    for(i = 0; i < gSize3D.numPolys; i++)
        list[i] = &(polys[i]);


    /*
     *  Update more lists
     */

    for(index1 = 0; index1 < gSize3D.numPolys; index1++) {

       index2 = 0;

       if((list[index1]->segs[0]->P == list[index1]->segs[1]->P) ||
           (list[index1]->segs[0]->P == list[index1]->segs[1]->Q)) {
           prevPoint = list[index1]->segs[0]->Q;
       }
       else{
           prevPoint = list[index1]->segs[0]->P;
       }

       while(index2 < list[index1]->numSegs){

           tmpSeg = list[index1]->segs[index2];

           if(tmpSeg->P == prevPoint){
               prevPoint = tmpSeg->Q;
           }
           else{
               prevPoint = tmpSeg->P;
           }

        /*
         *  Update points' polygon lists
         */

           if(prevPoint->numPolys == 0){
               if((prevPoint->polys = (polygon **)calloc(1, sizeof(polygon *)))== NULL){
                   puts("Unable to allocate memory for point polygons !");
                   return;
               }
           }
           else {
               if((prevPoint->polys = (polygon **)realloc(prevPoint->polys,
                   (prevPoint->numPolys + 1) * sizeof(polygon *))) == NULL){
                   puts("Unable to allocate memory for point polygons !");
                   return;
               }
           }
           prevPoint->polys[prevPoint->numPolys] = &(polys[index1]);
           prevPoint->numPolys++;

        /*
         *  Update polygons' point lists
         */

           if(polys[index1].numPoints == 0){
               if((polys[index1].points = (point **)calloc(1, sizeof(point *)))== NULL){
                   puts("Unable to allocate memory for polygon points !");
                   return;
               }
           }
           else {
               if((polys[index1].points = (point **) realloc(
                   polys[index1].points, (polys[index1].numPoints + 1) *
                   sizeof(point *))) == NULL){
                   puts("Unable to allocate memory for point polygons !");
                   return;
               }
           }

           polys[index1].points[polys[index1].numPoints] = prevPoint;
           polys[index1].numPoints++;

           index2++;
       }
    }
}


/*****************************************************************************
*                                                                            *
* main procedure                                                             *
*                                                                            *
******************************************************************************/
unsigned long
x3d_main(longitude, latitude, psi, string, parent)
float   *longitude;
float   *latitude;
float   *psi;
char    *string;
Window   parent;
{
    Ginfo *g = NULL;
    Oinfo *o = NULL;
    int i, j;
    FILE *fp;
    int export = 0;
    char filename[80], *indx;

    float xMin, yMin, zMin, xMax, yMax, zMax, correctionFactor;
    float xCenter, yCenter, zCenter;
    float xRange, zRange;

    quitApplication = 0;

    if((o = (Oinfo *) calloc(1, sizeof(Oinfo))) == NULL){
       (void) fprintf(stderr, "Unable to allocate memory for Object\n");
       return 0L;
    }
    gOInfo = o;

    if((g = (Ginfo *) calloc(1, sizeof(Ginfo))) == NULL){
       (void) fprintf(stderr, "Unable to allocate memory for Ginfo\n");
       return 0L;
    }
    gGInfo = g;

    indx = NULL;
    strcpy(title, "ROOT://X3D");

/*
 *  Print help
 */
    if (!strcmp(string, "help")) {
      puts("**** x3d QUICK HELP **************************************\n");
        puts(" QUIT                    q Q     MOVE OBJECT DOWN      u U" );
        puts(" TOGGLE CONTROLS STYLE   o O     MOVE OBJECT UP        i I" );
        puts(" TOGGLE STEREO DISPLAY   s S     MOVE OBJECT RIGHT     h H" );
        puts(" TOGGLE BLUE STEREO VIEW d D     MOVE OBJECT BACKWARD  j J" );
        puts(" TOGGLE DOUBLE BUFFER    f F     MOVE OBJECT FOREWARD  k K" );
        puts(" TOGGLE HELP MENU        m M     MOVE OBJECT LEFT      l L" );
        puts(" ROTATE ABOUT X      x X a A     AUTOROTATE ABOUT X  1 2 3" );
        puts(" ROTATE ABOUT Y      y Y b B     AUTOROTATE ABOUT Y  4 5 6" );
        puts(" ROTATE ABOUT Z      z Z c C     AUTOROTATE ABOUT Z  7 8 9\n");
        puts(" ADJUST FOCUS        [ ] { }     HIDDEN LINE MODE      e E" );
   puts(" WIREFRAME MODE          w W     HIDDEN SURFACE MODE   r R\n");
        puts(" POINTER MOVEMENT WITH LEFT BUTTON :\n");
        puts(" ROTATE OBJECT ABOUT X   Vertical" );
        puts(" ROTATE OBJECT ABOUT Z   Horizontal\n");

        return 0L;
   }
   else if ((indx = (char *) strstr(string, "hull:")) != NULL) {
       strcpy (filename, indx + 5);
       if (strlen(filename)) export = 1;
    }
   else if ((indx = (char *) strstr(string, "java:")) != NULL) {
       strcpy (filename, indx + 5);
       if (strlen(filename)) export = 2;
    }
   else if (string) {
       strcat(title, "/");
       strcat(title, string);
   }

    switch( export ) {
        case 1: /* hull */

            if (gSize3D.numPolys) {
                fp = fopen (filename, "w");
                if (fp != NULL) {
                    for (i = 0; i < gSize3D.numPolys; i++) {
                        fprintf (fp, "\n# polygon No. %d color ( R G B )\n", i);
                        fprintf (fp, "%5d%5d%5d\n\n", polys[i].color->red, polys[i].color->green, polys[i].color->blue);
                        for (j = 0; j < polys[i].numSegs; j++) {
                            fprintf (fp, "%20.6f%20.6f%20.6f\n", polys[i].segs[j]->P->x, polys[i].segs[j]->P->y, polys[i].segs[j]->P->z);
                        }
                    }
                    fclose (fp);
                }
            }
            else puts ("Can't export (number of polygons=0)");
            break;

        case 2: /* java */

            if( gSize3D.numSegs ) {
                fp = fopen( filename, "w" );
                if( fp != NULL ) {
                    for( i = 0; i < gSize3D.numSegs; i++ ) {
                        fprintf( fp, "v %20.6f%20.6f%20.6f\n", segs[i].P->x, segs[i].P->y, segs[i].P->z);
                        fprintf( fp, "v %20.6f%20.6f%20.6f\n", segs[i].Q->x, segs[i].Q->y, segs[i].Q->z);
                        fprintf( fp, "l %6d%6d\n", i*2+1, i*2+2 );
                    }
                    fclose (fp);
                }
            }
            break;

        default:
            break;
    }

/*
 *  Try to find the boundaries of the object
 */

    xMin = yMin = zMin =  999999;
    xMax = yMax = zMax = -999999;

   for (i = 0; i < gSize3D.numPoints; i++) {
       xMin = xMin <= points[i].x ? xMin : points[i].x;
       xMax = xMax >= points[i].x ? xMax : points[i].x;

       yMin = yMin <= points[i].y ? yMin : points[i].y;
       yMax = yMax >= points[i].y ? yMax : points[i].y;

       zMin = zMin <= points[i].z ? zMin : points[i].z;
       zMax = zMax >= points[i].z ? zMax : points[i].z;
   }

/*
 *  Compute the range & center of the object
 */

    xRange  = fabs(xMax - xMin);
    zRange  = fabs(zMax - zMin);

    xCenter = (xMax + xMin) / 2.0;
    yCenter = (yMax + yMin) / 2.0;
    zCenter = (zMax + zMin) / 2.0;

/*
 *  Compute the correctionFactor, rescale & put the object in the center
 */
   correctionFactor = 6000.0 / (xRange > zRange ? xRange : zRange);

   for (i = 0; i < gSize3D.numPoints; i++) {
       points[i].x = (points[i].x - xCenter) * correctionFactor;
       points[i].y = (points[i].y - yCenter) * correctionFactor;
       points[i].z = (points[i].z - zCenter) * correctionFactor;
   }

   deltaMove = (float) (xRange >= zRange ? xRange : zRange) / 20.0 * correctionFactor;

/*
 *  Calculate the bounding cube
 */

    bounds = NULL;
    bounds = (point *) calloc(8, sizeof(point));

    if (!bounds)
        (void) fprintf(stderr, "Unable to allocate memory for bounding cube.\n");

    xMin = (xMin - xCenter) * correctionFactor;
    xMax = (xMax - xCenter) * correctionFactor;
    yMin = (yMin - yCenter) * correctionFactor;
    yMax = (yMax - yCenter) * correctionFactor;
    zMin = (zMin - zCenter) * correctionFactor;
    zMax = (zMax - zCenter) * correctionFactor;

    if (bounds) {
        bounds[0].x = xMin; bounds[0].y = yMin; bounds[0].z = zMin;
        bounds[1].x = xMin; bounds[1].y = yMin; bounds[1].z = zMax;
        bounds[2].x = xMin; bounds[2].y = yMax; bounds[2].z = zMin;
        bounds[3].x = xMin; bounds[3].y = yMax; bounds[3].z = zMax;
        bounds[4].x = xMax; bounds[4].y = yMin; bounds[4].z = zMin;
        bounds[5].x = xMax; bounds[5].y = yMin; bounds[5].z = zMax;
        bounds[6].x = xMax; bounds[6].y = yMax; bounds[6].z = zMin;
        bounds[7].x = xMax; bounds[7].y = yMax; bounds[7].z = zMax;
    }

/*
 *  Make polygon pointer array
 */

    MakePolygonArray();

    g->Geometry    = "800x600";
    g->DisplayName = NULL;
    g->renderMode  = WIREFRAME;
    g->buffer      = 1;
    g->mono        = 0;
    g->stereo      = 0;
    g->stereoBlue  = 1;
    g->colors      = colors;
    g->numColors   = 28;
    g->win         = 0;
    o->points      = points;
    o->numPoints   = gSize3D.numPoints;
    o->segs        = segs;
    o->numSegs     = gSize3D.numSegs;
    o->polys       = polys;
    o->numPolys    = gSize3D.numPolys;
    o->list        = list;
    o->bounds      = bounds;


    if (!export && bounds) {

    /* Define viewing parameters */

       o->BViewpointX = 100.0;                   /* stereo separation factor */
       o->viewpointY  = -650.0;                  /* view point               */
       o->tX          = 640.0;                   /* observer X coordinate    */
       o->tY          = 6000.0;                  /* observer Y coordinate    */
       o->tZ          = 490.0;                   /* observer Z coordinate    */
       o->oX          = 0.0;                     /* origin X coordinate      */
       o->oY          = 0.0;                     /* origin Y coordinate      */
       o->oZ          = 0.0;                     /* origin Z coordinate      */
       o->X           = (double) (*latitude);    /* rotate angle around X    */
       o->Y           = (double) (*psi);         /* rotate angle around Y    */
       o->Z           = (double) (*longitude);   /* rotate angle around Z    */
       o->dX          = 0.0;                     /* autorotate around X      */
       o->dY          = 0.0;                     /* autorotate around Y      */
       o->dZ          = 0.0;                     /* autorotate around Z      */
       o->focus       = 0.2;


    /* Initialize the display */


       InitDisplay(o, g, parent);

       return g->win;
    }
    return 0;
}

void x3d_update()
{
   Ginfo *g = gGInfo;
   Oinfo *o = gOInfo;

   BeginImage(o, g);
   DrawObject(o, g);
   EndImage(o, g);
}

int x3d_dispatch_event(unsigned long evnt)
{

   XEvent *event = (XEvent *)evnt;
   Ginfo *g = gGInfo;
   Oinfo *o = gOInfo;

   UpdatePosition(event, o, g);

   x3d_update();

   return 1;
}

void x3d_get_position(float *longitude, float *latitude, float *psi)
{
   Oinfo *o = gOInfo;

   /* Update longitude and latitude */
   *latitude  =  (float) (o->X);
   *psi       =  (float) (o->Y);
   *longitude =  (float) (o->Z);
}

void x3d_terminate()
{
   int i;
   Ginfo *g = gGInfo;
   Oinfo *o = gOInfo;

   if (g->win) {
    /* Destroy windows */
       XDestroyWindow(g->dpy, g->win);
       XDestroyWindow(g->dpy, g->helpWin);

    /* Destroy graphics contexts */
       XFreeGC(g->dpy, g->gc);
       XFreeGC(g->dpy, g->helpGc);

    /* Free pixmap */
       XFreePixmap(g->dpy, g->pix);

    /* Close display */

       if (!gDisplay) {
          XSetCloseDownMode(g->dpy, DestroyAll);
          XCloseDisplay(g->dpy);
       }

    /* Free allocated memory */
       if (g->redColors)    free (g->redColors);
       if (g->redSegments)  free (g->redSegments);
       if (g->blueSegments) free (g->blueSegments);
       if (o)               free (o);
       if (g)               free (g);

    }

/*
 *  Free allocated memory & reset counters
 */
    currPoint = currSeg = currPoly = 0;

    for (i = 0; i < gSize3D.numPolys; i++) {
/*
 *       for (j = 0; j < polys[i].numPoints; j++) {
 *           if (polys[i].points[j]->polys)
 *               free (polys[i].points[j]->polys);
 *       }
 */
        if (polys[i].points) free (polys[i].points);
    }

    for (i = 0; i < gSize3D.numSegs; i++)
        if (segs[i].polys) free (segs[i].polys);

    for (i = 0; i < gSize3D.numPoints; i++)
        if (points[i].segs) free (points[i].segs);

    if (points) free (points);
    if (colors) free (colors);
    if (segs)   free (segs);
    if (polys)  free (polys);
    if (list)   free (list);
    if (bounds) free (bounds);
}

void x3d_set_display(unsigned long disp)
{
   gDisplay = (Display*) disp;
}

int x3d_exec_command(int pointerX, int pointerY, char command)
/******************************************************************************
   Update the scene position information using user input.

   The routine will eventually block waiting for an event if block is True
   and the no events of interest show up due to the call to GetInput()
******************************************************************************/
{
int dx, dy;
double X, Y, Z;

   Ginfo *g = gGInfo;
   Oinfo *o = gOInfo;

   X = Y = Z = 0.0;

/* dx, dy, dz are the amount to step about each axis every frame
   We want the scene to continue to rotate even if the user does
   not give any new input */

/* Do not forget to put your automatic update variables into this if
   statement.  Be careful somehow you can get MANY bugs with these!  */

      g->Block = 1;

/* Note: I do not move the origin which the scene is rotated about around.
   You may want to do oX += ???; oY += ???; oZ += ???    */

      switch(command){
         case ' ' : break;

         case 'm' :
         case 'M' : g->helpMenu    = !g->helpMenu;
             /*
             if(g->helpMenu == False){
               XUnmapWindow(g->dpy, g->helpWin);
             }else{
               XMapWindow(g->dpy, g->helpWin);
             }
             */
             break;

         case 's' :
         case 'S' : g->stereo  = !g->stereo; g->modeChanged = 1;
             break;

         case 'd' :
         case 'D' : g->stereoBlue = !g->stereoBlue; g->modeChanged = 1;
             break;

         case 'f' :
         case 'F' : g->buffer = !g->buffer; g->modeChanged = 1;
             break;

         case 'o' :
         case 'O' : g->Relative = !g->Relative; break;

         case 'w' :
         case 'W' : g->renderMode = WIREFRAME; g->modeChanged = 1;
             break;

         case 'e' :
         case 'E' : if(o->numPolys) {
                      g->renderMode = HIDDENLINE; g->modeChanged = 1;
                    }
             break;

         case 'r' :
         case 'R' : if(o->numPolys) {
                      g->renderMode = SOLID; g->modeChanged = 1;
                    }
             break;

         case 'l' : o->tX  -= deltaMove; break;
         case 'j' : o->tY  -= deltaMove; break;
         case 'k' : o->tY  += deltaMove; break;
         case 'h' : o->tX  += deltaMove; break;
         case 'i' : o->tZ  += deltaMove; break;
         case 'u' : o->tZ  -= deltaMove; break;
         case 'L' : o->tX  -= 5*deltaMove; break;
         case 'J' : o->tY  -= 5*deltaMove; break;
         case 'K' : o->tY  += 5*deltaMove; break;
         case 'H' : o->tX  += 5*deltaMove; break;
         case 'I' : o->tZ  += 5*deltaMove; break;
         case 'U' : o->tZ  -= 5*deltaMove; break;
         case '1' : o->dX += 0.02; break;
         case '2' : o->dX =  0.0 ; break;
         case '3' : o->dX -= 0.02; break;
         case '4' : o->dY -= 0.02; break;
         case '5' : o->dY =  0.0 ; break;
         case '6' : o->dY += 0.02; break;
         case '7' : o->dZ += 0.02; break;
         case '8' : o->dZ =  0.0 ; break;
         case '9' : o->dZ -= 0.02; break;
         case 'x' : X -= 0.03; break;
         case 'X' : X += 0.03; break;
         case 'y' : Y += 0.03; break;
         case 'Y' : Y -= 0.03; break;
         case 'z' : Z -= 0.03; break;
         case 'Z' : Z += 0.03; break;
         case 'a' : X -= 0.05; break;
         case 'A' : X += 0.05; break;
         case 'b' : Y += 0.05; break;
         case 'B' : Y -= 0.05; break;
         case 'c' : Z -= 0.05; break;
         case 'C' : Z += 0.05; break;
         case '[' : o->focus += 0.1;
                    if((o->focus > 1.8))
                       o->focus = 1.8;
                    break;
         case ']' : o->focus -= 0.1;
                    if((o->focus < -0.8))
                       o->focus = -0.8;
                    break;
         case '{' : o->BViewpointX -= 4.0; break;
         case '}' : o->BViewpointX += 4.0; break;

         case 'q' :
         case 'Q' : return(1);

         default : {

/* My pointer movement stuff */

/* Only update if the movement was reasonably small */

            dx = pointerX - g->oldPointerX;
            dy = pointerY - g->oldPointerY;

            if((dy * dy <= SMALLMOVEMENT) &&
               (dx * dx <= SMALLMOVEMENT)){

/* Rotate proportionally with the amount the pointer moved */
/* Note: I only control the X and Z axes by the pointer */

               X -= (dy * POINTERRATIO);
               Z -= (dx * POINTERRATIO);
            }
            g->oldPointerY = pointerY;
            g->oldPointerX = pointerX;
         }
      }
/*      } */

/* Keep angles 0 - 6.28 */

   X = fmod(X + o->dX, TWOPI);
   Y = fmod(Y + o->dY, TWOPI);
   Z = fmod(Z + o->dZ, TWOPI);

/* Fix up the angles */

   if(g->Relative){
      o->X = fmod(X + o->X, TWOPI);
      o->Y = fmod(Y + o->Y, TWOPI);
      o->Z = fmod(Z + o->Z, TWOPI);
   }else{
      CalculateAngles(&(o->X), &(o->Y), &(o->Z), X, Y, Z);
   }

   x3d_update();

   return quitApplication;
}

#endif
