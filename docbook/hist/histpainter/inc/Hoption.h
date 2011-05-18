/* @(#)root/histpainter:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Hoption
#define ROOT_Hoption


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THoption                                                             //
//                                                                      //
// Histogram option structure.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


typedef struct Hoption_t {
   // chopt may be the concatenation of the following options:

   int Axis;        // "A"  Axis are not drawn around the graph.
   int Bar;         // "B"  A Bar chart is drawn at each point.
   int Curve;       // "C"  A smooth Curve is drawn.
   int Error;       // "E"  Draw Errors with current marker type and size.
   int Fill;        // "F"  A fill area is drawn ("CF" draw a smooth fill area).
   int Off;         // "][" With H option, the first and last vertical lines are not drawn.
   int Keep;        // "K"  The status of the histogram is kept in memory
   int Line;        // "L"  A simple polyline beetwen every point is drawn.
   int Mark;        // "P"  The current Marker is drawn at each point
   int Same;        // "S"  Histogram is plotted in the current PAD.
   int Update;      // "U"  Update histogram previously plotted with option K
   int Star;        // "*"  A * is plotted at each point
   int Arrow;       // "ARR"   Draw 2D plot with Arrows.
   int Box;         // "BOX"   Draw 2D plot with proportional Boxes.
   int Char;        // "CHAR"  Draw 2D plot with a character set.
   int Color;       // "COL"   Draw 2D plot with Colored boxes.
   int Contour;     // "CONT"  Draw 2D plot as a Contour plot.
   int Func;        // "FUNC"  Draw only the function (for example in case of fit).
   int Hist;        // "HIST"  Draw only the histogram.
   int Lego;        // "LEGO"  Draw as a Lego plot(LEGO,Lego=1, LEGO1,Lego1=11, LEGO2,Lego=12).
   int Scat;        // "SCAT"  Draw 2D plot a Scatter plot.
   int Surf;        // "SURF"  Draw as a Surface (SURF,Surf=1, SURF1,Surf=11, SURF2,Surf=12)
   int Text;        // "TEXT"  Draw 2D plot with the content of each cell.
   int Tri;         // "TRI"   Draw 2D plot with Delaunay triangles.
   int Pie;         // "PIE"   Draw 1D plot as a pie chart.
   int System;      // type of coordinate system(1=car,2=pol,3=cyl,4=sph,5=psr)
   int Zscale;      // "Z"   to display the Z scale (color palette)
   int FrontBox;    //  = 0 to suppress the front box
   int BackBox;     //  = 0 to suppress the back box
   int List;        //  = 1 to generate the TObjArray "contours"
   int HighRes;     //  = 1 to select high resolution
   int Proj;        //  = 1 to get an Aitoff projection, usefull for skymaps or exposure maps..   
                    //  = 2 to get a Mercator ptojection
                    //  = 3 to get a Sinusoidal ptojection
                    //  = 4 to get a Parabolic ptojection
   int AxisPos;     //  Axis position
   int Spec;        // TSpectrum graphics

   int Zero;        // if selected with any LEGO option the empty are not drawn.

   // the following structure members are set to 1 if the corresponding option
   // in the current style is selected.

   int Logx;        // log scale in X. Also set by histogram option
   int Logy;        // log scale in Y. Also set by histogram option
   int Logz;        // log scale in Z. Also set by histogram option

} Hoption_t;

#endif
