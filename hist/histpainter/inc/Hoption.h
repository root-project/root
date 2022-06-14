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


////////////////////////////////////////////////////////////////////////////////
/*! \struct Hoption_t
    \brief Histograms' drawing options structure.

Used internally by THistPainter to manage histograms' drawing options.

*/


typedef struct Hoption_t {

///@{
/// @name Histogram's drawing options
/// The drawing option may be the concatenation of some of the following options:

   int Axis;        ///< <b>"A"</b> Axis are not drawn around the graph.
   int Bar;         ///< <b>"B"</b>, <b>"BAR"</b> and <b>"HBAR"</b> A Bar chart is drawn at each point.
   int Curve;       ///< <b>"C"</b> A smooth Curve is drawn.
   int Error;       ///< <b>"En"</b> Draw Errors with current marker type and size (0 <= n <=6).
   int Fill;        ///< <b>"F"</b> A fill area is drawn ("CF" draw a smooth fill area).
   int Off;         ///< <b>"]["</b> The first and last vertical lines are not drawn.
   int Line;        ///< <b>"L"</b> A simple polyline through every point is drawn.
   int Mark;        ///< <b>"P"</b> The current Marker is drawn at each point.
   int Same;        ///< <b>"SAME"</b>  Histogram is plotted in the current pad.
   int Star;        ///< <b>"*"</b> With option <b>"P"</b>, a * is plotted at each point.
   int Arrow;       ///< <b>"ARR"</b> Draw 2D plot with Arrows.
   int Box;         ///< <b>"BOX"</b> Draw 2D plot with proportional Boxes.
   int Char;        ///< <b>"CHAR"</b> Draw 2D plot with a character set.
   int Color;       ///< <b>"COL"</b> Draw 2D plot with Colored boxes.
   int Contour;     ///< <b>"CONTn"</b> Draw 2D plot as a Contour plot (0 <= n <= 5).
   int Func;        ///< <b>"FUNC"</b> Draw only the function (for example in case of fit).
   int Hist;        ///< <b>"HIST"</b> Draw only the histogram.
   int Lego;        ///< <b>"LEGO"</b> and <b>"LEGOn"</b> Draw as a Lego plot(1 <= n <= 4).
   int Scat;        ///< <b>"SCAT"</b> Draw 2D plot a Scatter plot.
   int Surf;        ///< <b>"SURF"</b> and <b>"SURFn"</b> Draw as a Surface ((1 <= n <= 4).
   int Text;        ///< <b>"TEXT"</b> Draw 2D plot with the content of each cell.
   int Tri;         ///< <b>"TRI"</b> Draw TGraph2D with Delaunay triangles.
   int Pie;         ///< <b>"PIE"</b>  Draw 1D plot as a pie chart.
   long Candle;     ///< <b>"CANDLE"</b> and <b>"VIOLIN"</b> Draw a 2D histogram as candle/box plot or violin plot.
   int System;      ///< <b>"POL"</b>, <b>"CYL"</b>, <b>"SPH"</b> and <b>"PSR"</b> Type of coordinate system for 3D plots.
   int Zscale;      ///< <b>"Z"</b> Display the color palette.
   int FrontBox;    ///< <b>"FB"</b> Suppress the front box for the 3D plots.
   int BackBox;     ///< <b>"BB"</b> Suppress the back box for the 3D plots.
   int List;        ///< <b>"LIST"</b> Generate the TObjArray "contours". To be used with option <b>"CONT"</b>
   int Proj;        ///< <b>"AITOFF"</b>, <b>"MERCATOR"</b>, <b>"SINUSOIDAL"</b> and <b>"PARABOLIC"</b> projections for 2d plots.
   int AxisPos;     ///< <b>"X+"</b> and <b>"Y+"</b> Axis position
   int Spec;        ///< <b>"SPEC"</b> TSpectrum graphics
   int Zero;        ///< <b>"0"</b> if selected with any LEGO option the empty bins are not drawn.
   int MinimumZero; ///< <b>"MIN0"</b> or gStyle->GetHistMinimumZero()
///@}

///@{
/// @name Other options
/// The following structure members are set to 1 if the corresponding setting is selected.

   int Logx;        ///< log scale in X. Also set by histogram option
   int Logy;        ///< log scale in Y. Also set by histogram option
   int Logz;        ///< log scale in Z. Also set by histogram option

///@}

} Hoption_t;

#endif
