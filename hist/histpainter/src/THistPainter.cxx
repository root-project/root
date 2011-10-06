// @(#)root/histpainter:$Id$
// Author: Rene Brun   26/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TSystem.h"
#include "THistPainter.h"
#include "TH2.h"
#include "TH2Poly.h"
#include "TH3.h"
#include "TProfile.h"
#include "THStack.h"
#include "TF2.h"
#include "TF3.h"
#include "TCutG.h"
#include "TMatrixDBase.h"
#include "TMatrixFBase.h"
#include "TVectorD.h"
#include "TVectorF.h"
#include "TPad.h"
#include "TPaveStats.h"
#include "TFrame.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TPoints.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TPie.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TPainter3dAlgorithms.h"
#include "TGraph2DPainter.h"
#include "TGraphDelaunay.h"
#include "TView.h"
#include "TMath.h"
#include "TRandom2.h"
#include "TObjArray.h"
#include "TVectorD.h"
#include "Hoption.h"
#include "Hparam.h"
#include "TPluginManager.h"
#include "TPaletteAxis.h"
#include "TCrown.h"
#include "TVirtualPadEditor.h"
#include "TEnv.h"
#include "TPoint.h"


//______________________________________________________________________________
/* Begin_Html
<center><h2>The histogram painter class</h2></center>

<ul>
<li><a href="#HP00">Introduction</li></a>
<li><a href="#HP01">Histograms' plotting options</li></a>
<ul>
<li><a href="#HP01a">Options supported for 1D and 2D histograms</li></a>
<li><a href="#HP01b">Options supported for 1D histograms</li></a>
<li><a href="#HP01c">Options supported for 2D histograms</li></a>
<li><a href="#HP01d">Options supported for 3D histograms</li></a>
<li><a href="#HP01e">Options supported for histograms' stacks (<tt>THStack</tt>)</li></a>
</ul>
<li><a href="#HP02">Setting the Style</li></a>
<li><a href="#HP03">Setting line, fill, marker, and text attributes</li></a>
<li><a href="#HP04">Setting Tick marks on the histogram axis</li></a>
<li><a href="#HP05">Giving titles to the X, Y and Z axis</li></a>
<li><a href="#HP060">The option "SAME"</li></a>
<ul>
<li><a href="#HP060a">Limitations</li></a>
</ul>
<li><a href="#HP06">Superimposing two histograms with different scales in the same pad</li></a>
<li><a href="#HP07">Statistics Display</li></a>
<li><a href="#HP08">Fit Statistics</li></a>
<li><a href="#HP09">The error bars options</li></a>
<li><a href="#HP100">The bar chart option</li></a>
<li><a href="#HP10">The "BAR" and "HBAR" options</li></a>
<li><a href="#HP11">The SCATter plot option (default for 2D histograms)</li></a>
<li><a href="#HP12">The ARRow option</li></a>
<li><a href="#HP13">The BOX option</li></a>
<li><a href="#HP14">The COLor option</li></a>
<li><a href="#HP15">The TEXT and TEXTnn Option</li></a>
<li><a href="#HP16">The CONTour options</li></a>
<ul>
<li><a href="#HP16a">The LIST option</li></a>
</ul>
<li><a href="#HP17">The LEGO options</li></a>
<li><a href="#HP18">The "SURFace" options</li></a>
<li><a href="#HP19">Cylindrical, Polar, Spherical and PseudoRapidity/Phi options</li></a>
<li><a href="#HP20">Base line for bar-charts and lego plots</li></a>
<li><a href="#HP20a">TH2Poly Drawing</li></a>
<li><a href="#HP21">The SPEC option</li></a>
<li><a href="#HP22">Option "Z" : Adding the color palette on the right side of the pad</li></a>
<li><a href="#HP23">Setting the color palette</li></a>
<li><a href="#HP24">Drawing a sub-range of a 2-D histogram; the [cutg] option</li></a>
<li><a href="#HP25">Drawing options for 3D histograms</li></a>
<li><a href="#HP26">Drawing option for histograms' stacks</li></a>
<li><a href="#HP27">Drawing of 3D implicit functions</li></a>
<li><a href="#HP28">Associated functions drawing</li></a>
<li><a href="#HP29">Drawing using OpenGL</li></a>
<ul>
<li><a href="#HP29a">General information: plot types and supported options</li></a>
<li><a href="#HP29b">TH3 as boxes (spheres)</li></a>
<li><a href="#HP29c">TH3 as iso-surface(s)</li></a>
<li><a href="#HP29d">TF3 (implicit function)</li></a>
<li><a href="#HP29e">Parametric surfaces</li></a>
<li><a href="#HP29f">Interaction with the plots</li></a>
<li><a href="#HP29g">Selectable parts</li></a>
<li><a href="#HP29h">Rotation and zooming</li></a>
<li><a href="#HP29i">Panning</li></a>
<li><a href="#HP29j">Box cut</li></a>
<li><a href="#HP29k">Plot specific interactions (dynamic slicing etc.)</li></a>
<li><a href="#HP29l">Surface with option "GLSURF"</li></a>
<li><a href="#HP29m">TF3</li></a>
<li><a href="#HP29n">Box</li></a>
<li><a href="#HP29o">Iso</li></a>
<li><a href="#HP29p">Parametric plot</li></a>
</ul>
</ul>


<a name="HP00"></a><h3>Introduction</h3>


Histograms are drawn via the <tt>THistPainter</tt> class. Each histogram has a
pointer to its own painter (to be usable in a multithreaded program). When the
canvas has to be redrawn, the <tt>Paint</tt> function of each objects in the
pad is called. In case of histograms, <tt>TH1::Paint</tt> invokes directly
<tt>THistPainter::Paint</tt>.

<p>To draw a histogram "<tt>h</tt>" is enough to do:
<pre>
      h->Draw();
</pre>
"<tt>h</tt>" can be of any kind: 1D, 2D or 3D. To choose how the histogram will
be drawn, the <tt>Draw()</tt> method can be invoked with an option. For instance
to draw a 2D histogram as a lego plot it is enough to do:
<pre>
      h->Draw("lego");
</pre>
<tt>THistPainter</tt> offers many options to paint 1D, 2D and 3D histograms.

<p>When the <tt>Draw()</tt> method of a histogram is called for the first time
(<tt>TH1::Draw</tt>), it creates a <tt>THistPainter</tt> object and saves a
pointer to this "painter" as a data member of the histogram. The
<tt>THistPainter</tt> class specializes in the drawing of histograms. It is
separated from the histogram so that one can have histograms without the
graphics overhead, for example in a batch program. Each histogram have its own
painter rather than a central singleton painter painting all histograms, allows
two histograms to be drawn in two threads without overwriting the painter's
values.

<p>When a displayed histogram is filled again, there is not need to call the
<tt>Draw()</tt> method again; the image will be refreshed the next time the
pad will be updated.

<p>A pad is updated after one of these three actions:
<ol>
<li>  a carriage control on the ROOT command line,
<li>  a click inside the pad,
<li>  a call to <tt>TPad::Update</tt>.
</ol>

<p>By default a call to <tt>TH1::Draw()</tt> clears the pad of all objects
before drawing the new image of the histogram. One can use the <tt>"SAME"</tt>
option to leave the previous display intact and superimpose the new histogram.
The same histogram can be drawn with different graphics options in different
pads.

<p>When a displayed histogram is deleted, its image is automatically removed
from the pad.

<p> To create a copy of the histogram when drawing it, one can use
<tt>TH1::DrawClone()</tt>. This will clone the histogram and allow to change
and delete the original one without affecting the clone.


<a name="HP01"></a><h3>Histograms' plotting options</h3>


Most options can be concatenated with or without spaces or commas, for example:
<pre>
      h->Draw("E1 SAME");
</pre>
The options are not case sensitive:
<pre>
      h->Draw("e1 same");
</pre>

The default drawing option can be set with <tt>TH1::SetOption</tt> and retrieve
using <tt>TH1::GetOption</tt>:
<pre>
      root [0] h->Draw();          // Draw "h" using the standard histogram representation.
      root [1] h->Draw("E");       // Draw "h" using error bars
      root [3] h->SetOption("E");  // Change the default drawing option for "h"
      root [4] h->Draw();          // Draw "h" using error bars
      root [5] h->GetOption();     // Retrieve the default drawing option for "h"
      (const Option_t* 0xa3ff948)"E"
</pre>

<a name="HP01a"></a><h4><u>Options supported for 1D and 2D histograms</u></h4>

<table border=0>

<tr><th valign=top>"AXIS"</th><td>
Draw only axis.
</td></tr>

<tr><th valign=top>"AXIG"</th><td>
Draw only grid (if the grid is requested).
</td></tr>

<tr><th valign=top>"HIST"</th><td>
When an histogram has errors it is visualized by default with error bars. To
visualize it without errors use the option "HIST" together with the required
option (eg "hist same c").  The "HIST" option can also be used to plot only the
histogram and not the associated function(s).
</td></tr>

<tr><th valign=top>"FUNC"</th><td>
When an histogram has a fitted function, this option allows to draw the fit
result only.
</td></tr>

<tr><th valign=top>"SAME"</th><td>
Superimpose on previous picture in the same pad.
</td></tr>

<tr><th valign=top>"LEGO"</th><td>
Draw a lego plot with hidden line removal.
</td></tr>

<tr><th valign=top>"LEGO1"</th><td>
Draw a lego plot with hidden surface removal.
</td></tr>

<tr><th valign=top>"LEGO2"</th><td>
Draw a lego plot using colors to show the cell contents When the option "0" is
used with any LEGO option, the empty bins are not drawn.
</td></tr>

<tr><th valign=top>"TEXT"</th><td>
Draw bin contents as text (format set via <tt>gStyle->SetPaintTextFormat</tt>).
</td></tr>

<tr><th valign=top>"TEXTnn"</th><td>
Draw bin contents as text at angle nn (0 < nn < 90).
</td></tr>

<tr><th valign=top>"X+"</th><td>
The X-axis is drawn on the top side of the plot.
</td></tr>

<tr><th valign=top>"Y+"</th><td>
The Y-axis is drawn on the right side of the plot.
</td></tr>

</table>

<a name="HP01b"></a><h4><u>Options supported for 1D histograms</u></h4>

<table border=0>

<tr><th valign=top>" "</th><td>
Default.
</td></tr>

<tr><th valign=top>"AH"</th><td>
Draw histogram without axis. "A" can be combined with any drawing option. For
instance, "AC" draws the histogram as a smooth Curve without axis.
</td></tr>

<tr><th valign=top>"]["</th><td>
When this option is selected the first and last vertical lines of the histogram
are not drawn.
</td></tr>

<tr><th valign=top>"B"</th><td>
Bar chart option.
</td></tr>

<tr><th valign=top>"BAR"</th><td>
Like option "B", but bars can be drawn with a 3D effect.
</td></tr>

<tr><th valign=top>"HBAR"</th><td>
Like option "BAR", but bars are drawn horizontally.
</td></tr>

<tr><th valign=top>"C"</th><td>
Draw a smooth Curve through the histogram bins.
</td></tr>

<tr><th valign=top>"E"</th><td>
Draw error bars.
</td></tr>

<tr><th valign=top>"E0"</th><td>
Draw error bars. Markers are drawn for bins with 0 contents.
</td></tr>

<tr><th valign=top>"E1"</th><td>
Draw error bars with perpendicular lines at the edges.
</td></tr>

<tr><th valign=top>"E2"</th><td>
Draw error bars with rectangles.
</td></tr>

<tr><th valign=top>"E3"</th><td>
Draw a fill area through the end points of the vertical error bars.
</td></tr>

<tr><th valign=top>"E4"</th><td>
Draw a smoothed filled area through the end points of the error bars.
</td></tr>

<tr><th valign=top>"E5"</th><td>
Like E3 but ignore the bins with 0 contents.
</td></tr>

<tr><th valign=top>"E6"</th><td>
Like E4 but ignore the bins with 0 contents.
</td></tr>

<tr><th valign=top>"X0"</th><td>
When used with one of the "E" option, it suppress the error bar along
X as <tt>gStyle->SetErrorX(0)</tt> would do.
</td></tr>

<tr><th valign=top>"L"</th><td>
Draw a line through the bin contents.
</td></tr>

<tr><th valign=top>"P"</th><td>
Draw current marker at each bin except empty bins.
</td></tr>

<tr><th valign=top>"P0"</th><td>
Draw current marker at each bin including empty bins.
</td></tr>

<tr><th valign=top>"PIE"</th><td>
Draw histogram as a Pie Chart.
</td></tr>

<tr><th valign=top>"*H"</th><td>
Draw histogram with a * at each bin.
</td></tr>

<tr><th valign=top>"LF2"</th><td>
Draw histogram like with option "L" but with a fill area. Note that "L" draws
also a fill area if the hist fill color is set but the fill area corresponds to
the histogram contour.
</td></tr>

<tr><th valign=top>"9"</th><td>
Force histogram to be drawn in high resolution mode. By default, the histogram
is drawn in low resolution in case the number of bins is greater than the number
of pixels in the current pad. This option should be combined with a "drawing
option" like "H" or "L".
</td></tr>

</table>

<a name="HP01c"></a><h4><u>Options supported for 2D histograms</u></h4>

<table border=0>

<tr><th valign=top>" "</th><td>
Default (scatter plot).
</td></tr>

<tr><th valign=top>"ARR"</th><td>
Arrow mode. Shows gradient between adjacent cells.
</td></tr>

<tr><th valign=top>"BOX"</th><td>
A box is drawn for each cell with surface proportional to the content's
absolute value. A negative content is marked with a X.
</td></tr>

<tr><th valign=top>"BOX1"</th><td>
A button is drawn for each cell with surface proportional to content's absolute
value. A sunken button is drawn for negative values a raised one for positive.
</td></tr>

<tr><th valign=top>"COL"</th><td>
A box is drawn for each cell with a color scale varying with contents. All the
none empty bins are painted. Empty bins are not painted unless some bins have
a negative content because in that case the null bins might be not empty.
</td></tr>

<tr><th valign=top>"COLZ"</th><td>
Same as "COL". In addition the color palette is also drawn.
</td></tr>

<tr><th valign=top>"CONT"</th><td>
Draw a contour plot (same as CONT0).
</td></tr>

<tr><th valign=top>"CONT0"</th><td>
Draw a contour plot using surface colors to distinguish contours.
</td></tr>

<tr><th valign=top>"CONT1"</th><td>
Draw a contour plot using line styles to distinguish contours.
</td></tr>

<tr><th valign=top>"CONT2"</th><td>
Draw a contour plot using the same line style for all contours.
</td></tr>

<tr><th valign=top>"CONT3"</th><td>
Draw a contour plot using fill area colors.
</td></tr>

<tr><th valign=top>"CONT4"</th><td>
Draw a contour plot using surface colors (SURF option at theta = 0).
</td></tr>

<tr><th valign=top>"CONT5"</th><td>
(TGraph2D only) Draw a contour plot using Delaunay triangles.
</td></tr>

<tr><th valign=top>"LIST"</th><td>
Generate a list of TGraph objects for each contour.
</td></tr>

<tr><th valign=top>"CYL"</th><td>
Use Cylindrical coordinates. The X coordinate is mapped on the angle and the Y
coordinate on the cylinder length.
</td></tr>

<tr><th valign=top>"POL"</th><td>
Use Polar coordinates. The X coordinate is mapped on the angle and the Y
coordinate on the radius.
</td></tr>

<tr><th valign=top>"SPH"</th><td>
Use Spherical coordinates. The X coordinate is mapped on the latitude and the Y
coordinate on the longitude.
</td></tr>

<tr><th valign=top>"PSR"</th><td>
Use PseudoRapidity/Phi coordinates. The X coordinate is mapped on Phi.
</td></tr>

<tr><th valign=top>"SURF"</th><td>
Draw a surface plot with hidden line removal.
</td></tr>

<tr><th valign=top>"SURF1"</th><td>
Draw a surface plot with hidden surface removal.
</td></tr>

<tr><th valign=top>"SURF2"</th><td>
Draw a surface plot using colors to show the cell contents.
</td></tr>

<tr><th valign=top>"SURF3"</th><td>
Same as SURF with in addition a contour view drawn on the top.
</td></tr>

<tr><th valign=top>"SURF4"</th><td>
Draw a surface using Gouraud shading.
</td></tr>

<tr><th valign=top>"SURF5"</th><td>
Same as SURF3 but only the colored contour is drawn. Used with option CYL, SPH
or PSR it allows to draw colored contours on a sphere, a cylinder or a in
pseudo rapidity space. In cartesian or polar coordinates, option SURF3 is used.
</td></tr>

<tr><th valign=top>"FB"</th><td>
With LEGO or SURFACE, suppress the Front-Box.
</td></tr>

<tr><th valign=top>"BB"</th><td>
With LEGO or SURFACE, suppress the Back-Box.
</td></tr>

<tr><th valign=top>"A"</th><td>
With LEGO or SURFACE, suppress the axis.
</td></tr>

<tr><th valign=top>"SCAT"</th><td>
Draw a scatter-plot (default).
</td></tr>

<tr><th valign=top>"[cutg]"</th><td>
Draw only the sub-range selected by the TCutG named "cutg".
</td></tr>

</table>

<a name="HP01d"></a><h4><u>Options supported for 3D histograms</u></h4>

<table border=0>

<tr><th valign=top>" "</th><td>
Default (scatter plot).
</td></tr>

<tr><th valign=top>"ISO"</th><td>
Draw a Gouraud shaded 3d iso surface through a 3d histogram. It paints one
surface at the value computed as follow:
<tt>SumOfWeights/(NbinsX*NbinsY*NbinsZ)</tt>.
</td></tr>

<tr><th valign=top>"BOX"</th><td>
Draw a for each cell with volume proportional to the content's absolute value.
</td></tr>

<tr><th valign=top>"LEGO"</th><td>
Same as <tt>BOX</tt>.
</td></tr>

</table>

<a name="HP01e"></a><h4><u>Options supported for histograms' stacks (<tt>THStack</tt>)</u></h4>

<table border=0>

<tr><th valign=top>" "</th><td>
Default, the histograms are drawn on top of each other (as lego plots for 2D
histograms).
</td></tr>

<tr><th valign=top>"NOSTACK"</th><td>
Histograms in the stack are all paint in the same pad as if the option
<tt>"SAME"</tt> had been specified.
</td></tr>

<tr><th valign=top>"PADS"</th><td>
The current pad/canvas is subdivided into a number of pads equal to the number
of histograms in the stack and each histogram is paint into a separate pad.
</td></tr>

</table>


<a name="HP02"></a><h3>Setting the Style</h3>


Histograms use the current style (<tt>gStyle</tt>). When one changes the current
style and would like to propagate the changes to the histogram,
<tt>TH1::UseCurrentStyle</tt> should be called. Call <tt>UseCurrentStyle</tt> on
each histogram is needed.
<br>
To force all the histogram to use the current style use:
<pre>
      gROOT->ForceStyle();
</pre>
All the histograms read after this call will use the current style.


<a name="HP03"></a><h3>Setting line, fill, marker, and text attributes</h3>


The histogram classes inherit from the attribute classes:
<tt>TAttLine</tt>, <tt>TAttFill</tt> and <tt>TAttMarker</tt>.
See the description of these classes for the list of options.


<a name="HP04"></a><h3>Setting Tick marks on the histogram axis</h3>


The <tt>TPad::SetTicks</tt> method specifies the type of tick marks on the axis.
If <tt> tx = gPad->GetTickx()</tt> and <tt>ty = gPad->GetTicky()</tt> then:
<pre>
      tx = 1;   tick marks on top side are drawn (inside)
      tx = 2;   tick marks and labels on top side are drawn
      ty = 1;   tick marks on right side are drawn (inside)
      ty = 2;   tick marks and labels on right side are drawn
</pre>
By default only the left Y axis and X bottom axis are drawn
(<tt>tx = ty = 0</tt>)

<p><tt>TPad::SetTicks(tx,ty)</tt> allows to set these options.
See also The <tt>TAxis</tt> functions to set specific axis attributes.

<p>In case multiple color filled histograms are drawn on the same pad, the fill
area may hide the axis tick marks. One can force a redraw of the axis over all
the histograms by calling:
<pre>
      gPad->RedrawAxis();
</pre>


<a name="HP05"></a><h3>Giving titles to the X, Y and Z axis</h3>


<pre>
      h->GetXaxis()->SetTitle("X axis title");
      h->GetYaxis()->SetTitle("Y axis title");
</pre>
The histogram title and the axis titles can be any <tt>TLatex</tt> string.
The titles are part of the persistent histogram.


<a name="HP060"></a><h3>The option "SAME"</h3>


By default, when an histogram is drawn, the current pad is cleared before
drawing. In order to keep the previous drawing and draw on top of it the
option <tt>"SAME"</tt> should be use. The histogram drawn with the option
<tt>"SAME"</tt> uses the coordinates system available in the current pad.
<p>
This option can be used alone or combined with any valid drawing option but
some combinations must be use with care.

<a name="HP060a"></a><h4><u>Limitations</u></h4>
<ul>
<li>It does not work when
combined with the <tt>"LEGO"</tt> and <tt>"SURF"</tt> options unless the
histogram plotted with the option <tt>"SAME"</tt> has <u>exactly</u> the same
ranges on the X, Y and Z axis as the currently drawn histogram. To superimpose
lego plots <a href="#HP26">histograms' stacks</a> should be used.</li>
</ul>

<a name="HP06"></a><h3>Superimposing two histograms with different scales in the same pad</h3>


The following example creates two histograms, the second histogram is the bins
integral of the first one. It shows a procedure to draw the two histograms in
the same pad and it draws the scale of the second histogram using a new vertical
axis on the right side. See also the tutorial <tt>transpad.C</tt> for a variant
of this example.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   // create/fill draw h1
   gStyle->SetOptStat(kFALSE);
   TH1F *h1 = new TH1F("h1","Superimposing two histograms with different scales",100,-3,3);
   Int_t i;
   for (i=0;i<10000;i++) h1->Fill(gRandom->Gaus(0,1));
   h1->Draw();
   c1->Update();

   // create hint1 filled with the bins integral of h1
   TH1F *hint1 = new TH1F("hint1","h1 bins integral",100,-3,3);
   Float_t sum = 0;
   for (i=1;i<=100;i++) {
      sum += h1->GetBinContent(i);
      hint1->SetBinContent(i,sum);
   }

   // scale hint1 to the pad coordinates
   Float_t rightmax = 1.1*hint1->GetMaximum();
   Float_t scale = gPad->GetUymax()/rightmax;
   hint1->SetLineColor(kRed);
   hint1->Scale(scale);
   hint1->Draw("same");

   // draw an axis on the right side
   TGaxis *axis = new TGaxis(gPad->GetUxmax(),gPad->GetUymin(),
   gPad->GetUxmax(), gPad->GetUymax(),0,rightmax,510,"+L");
   axis->SetLineColor(kRed);
   axis->SetTextColor(kRed);
   axis->Draw();
   return c1;
}
End_Macro
Begin_Html


<a name="HP07"></a><h3>Statistics Display</h3>


The type of information shown in the histogram statistics box can be selected
with:
<pre>
      gStyle->SetOptStat(mode);
</pre>
The "<tt>mode</tt>" has up to nine digits that can be set to on(1 or 2), off(0).
<pre>
      mode = iourmen  (default = 000001111)
      k = 1;  kurtosis printed
      k = 2;  kurtosis and kurtosis error printed
      s = 1;  skewness printed
      s = 2;  skewness and skewness error printed
      i = 1;  integral of bins printed
      o = 1;  number of overflows printed
      u = 1;  number of underflows printed
      r = 1;  rms printed
      r = 2;  rms and rms error printed
      m = 1;  mean value printed
      m = 2;  mean and mean error values printed
      e = 1;  number of entries printed
      n = 1;  name of histogram is printed
</pre>
For example:
<pre>
      gStyle->SetOptStat(11);
</pre>
displays only the name of histogram and the number of entries, whereas:
<pre>
      gStyle->SetOptStat(1101);
</pre>
displays the name of histogram, mean value and RMS.

<p><b>WARNING 1:</b> never do:
<pre>
      <s>gStyle->SetOptStat(000111);</s>
</pre>
but instead do:
<pre>
      gStyle->SetOptStat(1111);
</pre>
because <tt>0001111</tt> will be taken as an octal number!

<p><b>WARNING 2:</b> for backward compatibility with older versions
<pre>
      gStyle->SetOptStat(1);
</pre>
is taken as:
<pre>
      gStyle->SetOptStat(1111)
</pre>
To print only the name of the histogram do:
<pre>
      gStyle->SetOptStat(1000000001);
</pre>
<b>NOTE</b> that in case of 2D histograms, when selecting only underflow
(10000) or overflow (100000), the statistics box will show all combinations
of underflow/overflows and not just one single number.

<p>The parameter mode can be any combination of the letters
<tt>kKsSiourRmMen</tt>
<pre>
      k :  kurtosis printed
      K :  kurtosis and kurtosis error printed
      s :  skewness printed
      S :  skewness and skewness error printed
      i :  integral of bins printed
      o :  number of overflows printed
      u :  number of underflows printed
      r :  rms printed
      R :  rms and rms error printed
      m :  mean value printed
      M :  mean value mean error values printed
      e :  number of entries printed
      n :  name of histogram is printed
</pre>
For example, to print only name of histogram and number of entries do:
<pre>
      gStyle->SetOptStat("ne");
</pre>
To print only the name of the histogram do:
<pre>
      gStyle->SetOptStat("n");
</pre>
The default value is:
<pre>
      gStyle->SetOptStat("nemr");
</pre>

<p>When a histogram is painted, a <tt>TPaveStats</tt> object is created and added
to the list of functions of the histogram. If a <tt>TPaveStats</tt> object
already exists in the histogram list of functions, the existing object is just
updated with the current histogram parameters.

<p>Once a histogram is painted, the statistics box can be accessed using
<tt>h->FindObject("stats")</tt>. In the command line it is enough to do:
<pre>
      Root > h->Draw()
      Root > TPaveStats *st = (TPaveStats*)h->FindObject("stats")
</pre>
because after <tt>h->Draw()</tt> the histogram is automatically painted. But
in a script file the painting should be forced using <tt>gPad->Update()</tt>
in order to make sure the statistics box is created:
<pre>
      h->Draw();
      gPad->Update();
      TPaveStats *st = (TPaveStats*)h->FindObject("stats");
</pre>

<p>Without <tt>gPad->Update()</tt> the line <tt>h->FindObject("stats")</tt>
returns a null pointer.

<p>When a histogram is drawn with the option "<tt>SAME</tt>", the statistics box
is not drawn. To force the statistics box drawing with the option
"<tt>SAME</tt>", the option "<tt>SAMES</tt>" must be used.
If the new statistics box hides the previous statistics box, one can change
its position with these lines ("<tt>h</tt>" being the pointer to the histogram):
<pre>
      Root > TPaveStats *st = (TPaveStats*)h->FindObject("stats")
      Root > st->SetX1NDC(newx1); //new x start position
      Root > st->SetX2NDC(newx2); //new x end position
</pre>
To change the type of information for an histogram with an existing
<tt>TPaveStats</tt> one should do:
<pre>
      st->SetOptStat(mode);
</pre>
Where "<tt>mode</tt>" has the same meaning than when calling
<tt>gStyle->SetOptStat(mode)</tt> (see above).

<p>One can delete the statistics box for a histogram <tt>TH1* h</tt> with:
<pre>
      h->SetStats(0)
</pre>
and activate it again with:
<pre>
      h->SetStats(1).
</pre>


<a name="HP08"></a><h3>Fit Statistics</h3>


The type of information about fit parameters printed in the histogram statistics
box can be selected via the parameter mode. The parameter mode can be
<tt>= pcev</tt>  (default <tt>= 0111</tt>)
<pre>
      p = 1;  print Probability
      c = 1;  print Chisquare/Number of degrees of freedom
      e = 1;  print errors (if e=1, v must be 1)
      v = 1;  print name/values of parameters
</pre>
Example:
<pre>
      gStyle->SetOptFit(1011);
</pre>
print fit probability, parameter names/values and errors.
<ol>
<li> When <tt>"v" = 1</tt> is specified, only the non-fixed parameters are
     shown.
<li> When <tt>"v" = 2</tt> all parameters are shown.
</ol>
Note: <tt>gStyle->SetOptFit(1)</tt> means "default value", so it is equivalent
to <tt>gStyle->SetOptFit(111)</tt>


<a name="HP09"></a><h3>The error bars options</h3>


<table border=0>

<tr><th valign=top>"E"</th><td>
Default. Shows only the error bars, not a marker.
</td></tr>

<tr><th valign=top>"E1"</th><td>
Small lines are drawn at the end of the error bars.
</td></tr>

<tr><th valign=top>"E2"</th><td>
Error rectangles are drawn.
</td></tr>

<tr><th valign=top>"E3"</th><td>
A filled area is drawn through the end points of the vertical error bars.
</td></tr>

<tr><th valign=top>"E4"</th><td>
A smoothed filled area is drawn through the end points of the vertical error
bars.
</td></tr>

<tr><th valign=top>"E0"</th><td>
Draw also bins with null contents.
</td></tr>

</table>

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH1F *he = new TH1F("he","Distribution drawn with error bars (option E1)  ",100,-3,3);
   Int_t i;
   for (i=0;i<10000;i++) he->Fill(gRandom->Gaus(0,1));
   gStyle->SetEndErrorSize(3);
   gStyle->SetErrorX(1.);
   he->SetMarkerStyle(20);
   he->Draw("E1");
   return c1;
}
End_Macro
Begin_Html

<p>The options "E3" and "E4" draw an error band through the end points of the
vertical error bars. With "E4" the error band is smoothed. Because of the
smoothing algorithm used some artefacts may appear at the end of the band
like in the following example. In such cases "E3" should be used instead
of "E4".

End_Html
Begin_Macro(source)
{
   TCanvas *ce4 = new TCanvas("ce4","ce4",600,400);
   ce4->Divide(2,1);
   TH1F *he4 = new TH1F("he4","Distribution drawn with option E4",100,-3,3);
   Int_t i;
   for (i=0;i<10000;i++) he4->Fill(gRandom->Gaus(0,1));
   he4->SetFillColor(kRed);
   he4->GetXaxis()->SetRange(40,48);
   ce4->cd(1);
   he4->Draw("E4");
   ce4->cd(2);
   TH1F *he3 = he4->DrawClone("E3");
   he3->SetTitle("Distribution drawn option E3");
   return ce4;
}
End_Macro
Begin_Html


<a name="HP100"></a><h3>The bar chart option</h3>


The option "B" allows to draw simple vertical bar charts.
The bar width is controlled with <tt>TH1::SetBarWidth()</tt>,
and the bar offset wihtin the bin, with <tt>TH1::SetBarOffset()</tt>.
These two settings are useful to draw several histograms on the
same plot as shown in the following example:

End_Html
Begin_Macro(source)
{
   int i;
   const Int_t nx = 8;
   char *os_X[nx]   = {"8","32","128","512","2048","8192","32768","131072"};
   float d_35_0[nx] = {0.75, -3.30, -0.92, 0.10, 0.08, -1.69, -1.29, -2.37};
   float d_35_1[nx] = {1.01, -3.02, -0.65, 0.37, 0.34, -1.42, -1.02, -2.10};

   TCanvas *cb = new TCanvas("cb","cb",600,400);
   cb->SetGrid();

   gStyle->SetHistMinimumZero();

   TH1F *h1b = new TH1F("h1b","Option B example",nx,0,nx);
   h1b->SetFillColor(4);
   h1b->SetBarWidth(0.4);
   h1b->SetBarOffset(0.1);
   h1b->SetStats(0);
   h1b->SetMinimum(-5);
   h1b->SetMaximum(5);

   for (i=1; i<=nx; i++) {
      h1b->Fill(os_X[i-1], d_35_0[i-1]);
      h1b->GetXaxis()->SetBinLabel(i,os_X[i-1]);
   }

   h1b->Draw("b");

   TH1F *h2b = new TH1F("h2b","h2b",nx,0,nx);
   h2b->SetFillColor(38);
   h2b->SetBarWidth(0.4);
   h2b->SetBarOffset(0.5);
   h2b->SetStats(0);
   for (i=1;i<=nx;i++) h2b->Fill(os_X[i-1], d_35_1[i-1]);

   h2b->Draw("b same");

   return cb;
}
End_Macro
Begin_Html


<a name="HP10"></a><h3>The "BAR" and "HBAR" options</h3>


When the option "bar" or "hbar" is specified, a bar chart is drawn. A vertical
bar-chart is drawn with the options <tt>"bar"</tt>, <tt>"bar0"</tt>,
<tt>"bar1"</tt>, <tt>"bar2"</tt>, <tt>"bar3"</tt>, <tt>"bar4"</tt>.
An horizontal bar-chart is drawn with the options <tt>"hbar"</tt>,
<tt>"hbar0"</tt>, <tt>"hbar1"</tt>, <tt>"hbar2"</tt></tt>, <tt>"hbar3"</tt>,
<tt>"hbar4"</tt>.
<ul>
<li> The bar is filled with the histogram fill color.
<li> The left side of the bar is drawn with a light fill color.
<li> The right side of the bar is drawn with a dark fill color.
<li> The percentage of the bar drawn with either the light or dark color is:
<ul>
<li>    0% for option "(h)bar" or "(h)bar0"
<li>   10% for option "(h)bar1"
<li>   20% for option "(h)bar2"
<li>   30% for option "(h)bar3"
<li>   40% for option "(h)bar4"
</ul>
</ul>

End_Html
Begin_Macro(source)
../../../tutorials/hist/hbars.C
End_Macro
Begin_Html

<p>To control the bar width (default is the bin width) <tt>TH1::SetBarWidth()</tt>
should be used.
<br>
To control the bar offset (default is 0) <tt>TH1::SetBarOffset()</tt> should
be used.
<br>
These two parameters are useful when several histograms are plotted using
the option <tt>SAME</tt>. They allow to plot the histograms next to each other.


<a name="HP11"></a><h3>The SCATter plot option (default for 2D histograms)</h3>


For each cell (i,j) a number of points proportional to the cell content is
drawn. A maximum of <tt>kNMAX</tt> points per cell is drawn. If the maximum is above
<tt>kNMAX</tt> contents are normalized to <tt>kNMAX</tt> (<tt>kNMAX=2000</tt>).
If option is of the form <tt>"scat=ff"</tt>, (eg <tt>scat=1.8</tt>,
<tt>scat=1e-3</tt>), then <tt>ff</tt> is used as a scale factor to compute the
number of dots. <tt>"scat=1"</tt> is the default.
<p>
By default the scatter plot is painted with a "dot marker" which not scalable
(see the <a href="http://root.cern.ch/root/html/TAttMarker.html#M3">TAttMarker
documentation</a>). To change the marker size, a scalable marker type should be
used. For instance a circle (marker style 20).


End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hscat = new TH2F("hscat","Option SCATter example (default for 2D histograms)  ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hscat->Fill(px,5*py);
      hscat->Fill(3+0.5*px,2*py-10.);
   }
   hscat->Draw("scat=0.5");
   return c1;
}
End_Macro
Begin_Html


<a name="HP12"></a><h3>The ARRow option</h3>


Shows gradient between adjacent cells. For each cell (i,j) an arrow is drawn
The orientation of the arrow follows the cell gradient.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *harr = new TH2F("harr","Option ARRow example",20,-4,4,20,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      harr->Fill(px,5*py);
      harr->Fill(3+0.5*px,2*py-10.,0.1);
   }
   harr->Draw("ARR");
   return c1;
}
End_Macro
Begin_Html


<a name="HP13"></a><h3>The BOX option</h3>


For each cell (i,j) a box is drawn. The size (surface) of the box is
proportional to the absolute value of the cell content.
The cells with a negative content draw with a <tt>X</tt> on top of the boxes.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   hbox  = new TH2F("hbox","Option BOX example",3,0,3,3,0,3);
   hbox->SetFillColor(42);
   hbox->Fill(0.5, 0.5,  1.);
   hbox->Fill(0.5, 1.5,  4.);
   hbox->Fill(0.5, 2.5,  3.);
   hbox->Fill(1.5, 0.5,  2.);
   hbox->Fill(1.5, 1.5, 12.);
   hbox->Fill(1.5, 2.5, -6.);
   hbox->Fill(2.5, 0.5, -4.);
   hbox->Fill(2.5, 1.5,  6.);
   hbox->Fill(2.5, 2.5,  0.5);
   hbox->Draw("BOX");
   return c1;
}
End_Macro
Begin_Html

<p>With option <tt>"BOX1"</tt> a button is drawn for each cell with surface
proportional to content's absolute value. A sunken button is drawn for
negative values a raised one for positive.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   hbox1  = new TH2F("hbox1","Option BOX1 example",3,0,3,3,0,3);
   hbox1->SetFillColor(42);
   hbox1->Fill(0.5, 0.5,  1.);
   hbox1->Fill(0.5, 1.5,  4.);
   hbox1->Fill(0.5, 2.5,  3.);
   hbox1->Fill(1.5, 0.5,  2.);
   hbox1->Fill(1.5, 1.5, 12.);
   hbox1->Fill(1.5, 2.5, -6.);
   hbox1->Fill(2.5, 0.5, -4.);
   hbox1->Fill(2.5, 1.5,  6.);
   hbox1->Fill(2.5, 2.5,  0.5);
   hbox1->Draw("BOX1");
   return c1;
}
End_Macro
Begin_Html

<p>When the option <tt>"SAME"</tt> (or "SAMES") is used with the option <tt>"BOX"</tt>,
the boxes' sizes are computing taking the previous plots into account. The range
along the Z axis is imposed by the first plot (the one without option
<tt>"SAME"</tt>); therefore the order in which the plots are done is relevant.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hb1 = new TH2F("hb1","Example of BOX plots with option SAME ",40,-3,3,40,-3,3);
   TH2F *hb2 = new TH2F("hb2","hb2",40,-3,3,40,-3,3);
   TH2F *hb3 = new TH2F("hb3","hb3",40,-3,3,40,-3,3);
   TH2F *hb4 = new TH2F("hb4","hb4",40,-3,3,40,-3,3);
   for (Int_t i=0;i<1000;i++) {
      double x,y;
      gRandom->Rannor(x,y);
      if(x>0 && y>0) hb1->Fill(x,y,4);
      if(x<0 && y<0) hb2->Fill(x,y,3);
      if(x>0 && y<0) hb3->Fill(x,y,2);
      if(x<0 && y>0) hb4->Fill(x,y,1);
   }
   hb1->SetFillColor(1);
   hb2->SetFillColor(2);
   hb3->SetFillColor(3);
   hb4->SetFillColor(4);
   hb1->Draw("box");
   hb2->Draw("box same");
   hb3->Draw("box same");
   hb4->Draw("box same");
   return c1;
}
End_Macro
Begin_Html


<a name="HP14"></a><h3>The COLor option</h3>


For each cell (i,j) a box is drawn with a color proportional to the cell
content.

<p>The color table used is defined in the current style.

<p>If the histogram's minimum and maximum are the same (flat histogram), the
mapping on colors is not possible, therefore nothing is painted. To paint a
flat histogram it is enough to set the histogram minimum
(<tt>TH1::SetMinimum()</tt>) different from the bins' content.

<p>The default number of color levels used to paint the cells is 20.
It can be changed with <tt>TH1::SetContour()</tt> or
<tt>TStyle::SetNumberContours()</tt>. The higher this number is, the smoother
is the color change between cells.

<p>The color palette in TStyle can be modified via <tt>gStyle->SetPalette()</tt>.

<p>All the none empty bins are painted. Empty bins are not painted unless
some bins have a negative content because in that case the null bins
might be not empty.

<p>Combined with the option <tt>"COL"</tt>, the option <tt>"Z"</tt> allows to
display the color palette defined by <tt>gStyle->SetPalette()</tt>.

<p>In the following example, the histogram has only positive bins; the empty
bins (containing 0) <u>are not drawn</u>.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcol1 = new TH2F("hcol1","Option COLor example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcol1->Fill(px,5*py);
   }
   gStyle->SetPalette(1);
   hcol1->Draw("COLZ");
   return c1;
}
End_Macro
Begin_Html

<p>In the following example, the histogram has some negative bins; the empty
bins (containing 0) <u>are drawn</u>.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcol2 = new TH2F("hcol2","Option COLor example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcol2->Fill(px,5*py);
   }
   hcol2->Fill(0,0,-200);
   gStyle->SetPalette(1);
   hcol2->Draw("COLZ");
   return c1;
}
End_Macro
Begin_Html


<a name="HP15"></a><h3>The TEXT and TEXTnn Option</h3>


For each bin the content is printed. The text attributes are:
<ul>
<li> text font = current TStyle font (<tt>gStyle->SetTextFont()</tt>).
<li> text size = 0.02*padheight*markersize (if <tt>h</tt> is the histogram drawn
     with the option <tt>"TEXT"</tt> the marker size can be changed with
     <tt>h->SetMarkerSize(markersize)</tt>).
<li> text color = marker color.
</ul>
By default the format <tt>"g"</tt> is used. This format can be redefined
by calling <tt>gStyle->SetPaintTextFormat()</tt>.

<p>It is also possible to use <tt>"TEXTnn"</tt> in order to draw the text with
the angle <tt>nn</tt> (<tt>0 < nn < 90</tt>).

<p>For 2D histograms the text is plotted in the center of each non empty cells.
It is possible to plot empty cells by calling gStyle->SetHistMinimumZero().
For 1D histogram the text is plotted at a y position equal to the bin content.

<p>For 2D histograms when the option "E" (errors) is combined with the option
text ("TEXTE"), the error for each bin is also printed.

End_Html
Begin_Macro(source)
{
   TCanvas *c01 = new TCanvas("c01","c01",700,400);
   c01->Divide(2,1);
   TH1F *htext1 = new TH1F("htext1","Option TEXT on 1D histograms ",10,-4,4);
   TH2F *htext2 = new TH2F("htext2","Option TEXT on 2D histograms ",10,-4,4,10,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
        htext1->Fill(px,0.1);
      htext2->Fill(px,5*py,0.1);
   }
   gStyle->SetPaintTextFormat("4.1f m");
   htext2->SetMarkerSize(1.8);
   c01->cd(1);
   htext2->Draw("TEXT45");
   c01->cd(2);
   htext1->Draw();
   htext1->Draw("TEXT0 SAME");
   return c01;
}
End_Macro
Begin_Html

<p>In the case of profile histograms it is possible to print the number
of entries instead of the bin content. It is enough to combine the
option "E" (for entries) with the option "TEXT".

End_Html
Begin_Macro(source)
{
   TCanvas *c02 = new TCanvas("c02","c02",700,400);
   c02->Divide(2,1);
   gStyle->SetPaintTextFormat("g");

   TProfile *profile = new TProfile("profile","profile",10,0,10);
   profile->SetMarkerSize(2.2);
   profile->Fill(0.5,1);
   profile->Fill(1.5,2);
   profile->Fill(2.5,3);
   profile->Fill(3.5,4);
   profile->Fill(4.5,5);
   profile->Fill(5.5,5);
   profile->Fill(6.5,4);
   profile->Fill(7.5,3);
   profile->Fill(8.5,2);
   profile->Fill(9.5,1);
   c02->cd(1); profile->Draw("HIST TEXT0");
   c02->cd(2); profile->Draw("HIST TEXT0E");

   return c02;
}
End_Macro
Begin_Html

<a name="HP16"></a><h3>The CONTour options</h3>


The following contour options are supported:

<table border=0>

<tr><th valign=top>"CONT"</th><td>
Draw a contour plot (same as CONT0).
</td></tr>

<tr><th valign=top>"CONT0"</th><td>
Draw a contour plot using surface colors to distinguish contours.
</td></tr>

<tr><th valign=top>"CONT1"</th><td>
Draw a contour plot using the line colors to distinguish contours.
</td></tr>

<tr><th valign=top>"CONT2"</th><td>
Draw a contour plot using the line styles to distinguish contours.
</td></tr>

<tr><th valign=top>"CONT3"</th><td>
Draw a contour plot solid lines for all contours.
</td></tr>

<tr><th valign=top>"CONT4"</th><td>
Draw a contour plot using surface colors (<tt>"SURF"</tt> option at theta = 0).
</td></tr>

<tr><th valign=top>"CONT5"</th><td>
Draw a contour plot using Delaunay triangles.
</td></tr>

</table>

The following example shows a 2D histogram plotted with the option
<tt>"CONTZ"</tt>. The option <tt>"CONT"</tt> draws a contour plot using surface
colors to distinguish contours.  Combined with the option <tt>"CONT"</tt> (or
<tt>"CONT0"</tt>), the option <tt>"Z"</tt> allows to display the color palette
defined by <tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcontz = new TH2F("hcontz","Option CONTZ example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcontz->Fill(px-1,5*py);
      hcontz->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   hcontz->Draw("CONTZ");
   return c1;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"CONT1Z"</tt>. The option <tt>"CONT1"</tt> draws a contour plot using the
line colors to distinguish contours. Combined with the option <tt>"CONT1"</tt>,
the option <tt>"Z"</tt> allows to display the color palette defined by
<tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcont1 = new TH2F("hcont1","Option CONT1Z example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcont1->Fill(px-1,5*py);
      hcont1->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   hcont1->Draw("CONT1Z");
   return c1;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"CONT2"</tt>. The option <tt>"CONT2"</tt> draws a contour plot using the
line styles to distinguish contours.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcont2 = new TH2F("hcont2","Option CONT2 example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcont2->Fill(px-1,5*py);
      hcont2->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hcont2->Draw("CONT2");
   return c1;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"CONT3"</tt>. The option <tt>"CONT3"</tt> draws contour plot solid lines for
all contours.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcont3 = new TH2F("hcont3","Option CONT3 example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcont3->Fill(px-1,5*py);
      hcont3->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hcont3->Draw("CONT3");
   return c1;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"CONT4"</tt>. The option <tt>"CONT4"</tt> draws a contour plot using surface
colors to distinguish contours (<tt>"SURF"</tt> option at theta = 0). Combined
with the option <tt>"CONT"</tt> (or <tt>"CONT0"</tt>), the option <tt>"Z"</tt>
allows to display the color palette defined by <tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *hcont4 = new TH2F("hcont4","Option CONT4Z example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hcont4->Fill(px-1,5*py);
      hcont4->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   hcont4->Draw("CONT4Z");
   return c1;
}
End_Macro
Begin_Html

<p>The default number of contour levels is 20 equidistant levels and can be changed
with <tt>TH1::SetContour()</tt> or <tt>TStyle::SetNumberContours()</tt>.

<a name="HP16a"></a><h4><u>The LIST option</u></h4>

<p>When option <tt>"LIST"</tt> is specified together with option
<tt>"CONT"</tt>, the points used to draw the contours are saved in
<tt>TGraph</tt> objects:
<pre>
      h->Draw("CONT LIST");
      gPad->Update();
</pre>
The contour are saved in <tt>TGraph</tt> objects once the pad is painted.
Therefore to use this functionnality in a macro, <tt>gPad->Update()</tt>
should be performed after the histogram drawing. Once the list is
built, the contours are accessible in the following way:
<pre>
      TObjArray *contours = gROOT->GetListOfSpecials()->FindObject("contours")
      Int_t ncontours     = contours->GetSize();
      TList *list         = (TList*)contours->At(i);
</pre>
Where <tt>i</tt> is a contour number, and list contains a list of
<tt>TGraph</tt> objects.
For one given contour, more than one disjoint polyline may be generated.
The number of TGraphs per contour is given by:
<pre>
      list->GetSize();
</pre>
To access the first graph in the list one should do:
<pre>
      TGraph *gr1 = (TGraph*)list->First();
</pre>

The following example shows how to use this functionality.

End_Html
Begin_Macro(source)
../../../tutorials/hist/ContourList.C
End_Macro
Begin_Html

<p>The following options select the <tt>"CONT4"</tt> option and are useful for
sky maps or exposure maps.

<table border=0>

<tr><th valign=top>"AITOFF"</th><td>
Draw a contour via an AITOFF projection.
</td></tr>

<tr><th valign=top>"MERCATOR"</th><td>
Draw a contour via an Mercator projection.
</td></tr>

<tr><th valign=top>"SINUSOIDAL"</th><td>
Draw a contour via an Sinusoidal projection.
</td></tr>

<tr><th valign=top>"PARABOLIC"</th><td>
Draw a contour via an Parabolic projection.
</td></tr>

</table>

End_Html
Begin_Macro(source)
../../../tutorials/graphics/earth.C
End_Macro
Begin_Html


<a name="HP17"></a><h3>The LEGO options</h3>


In a lego plot the cell contents are drawn as 3-d boxes. The height of each box
is proportional to the cell content. The lego aspect is control with the
following options:

<table border=0>

<tr><th valign=top>"LEGO" </th><td>
Draw a lego plot using the hidden lines removal technique.
</td></tr>

<tr><th valign=top>"LEGO1"</th><td>
Draw a lego plot using the hidden surface removal technique.
</td></tr>

<tr><th valign=top>"LEGO2"</th><td>
Draw a lego plot using colors to show the cell contents.
</td></tr>

<tr><th valign=top>"0"</th><td>
When used with any LEGO option, the empty bins are not drawn.
</td></tr>

</table>
See the limitations with <a href="#HP060a">the option "SAME"</a>.

<p>The following example shows a 2D histogram plotted with the option
<tt>"LEGO"</tt>. The option <tt>"LEGO"</tt> draws a lego plot using the hidden
lines removal technique.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hlego = new TH2F("hlego","Option LEGO example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hlego->Fill(px-1,5*py);
      hlego->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hlego->Draw("LEGO");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"LEGO1"</tt>. The option <tt>"LEGO1"</tt> draws a lego plot using the
hidden surface removal technique. Combined with any <tt>"LEGOn"</tt> option, the
option <tt>"0"</tt> allows to not drawn the empty bins.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hlego1 = new TH2F("hlego1","Option LEGO1 example (with option 0)  ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hlego1->Fill(px-1,5*py);
      hlego1->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   hlego1->SetFillColor(kYellow);
   hlego1->Draw("LEGO1 0");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"LEGO2"</tt>. The option <tt>"LEGO2"</tt> draws a lego plot using colors to
show the cell contents.  Combined with the option <tt>"LEGO2"</tt>, the option
<tt>"Z"</tt> allows to display the color palette defined by
<tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hlego2 = new TH2F("hlego2","Option LEGO2Z example ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hlego2->Fill(px-1,5*py);
      hlego2->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   hlego2->Draw("LEGO2Z");
   return c2;
}
End_Macro
Begin_Html



<a name="HP18"></a><h3>The "SURFace" options</h3>


In a surface plot, cell contents are represented as a mesh.
The height of the mesh is proportional to the cell content.

<table border=0>

<tr><th valign=top>"SURF"</th><td>
Draw a surface plot using the hidden line removal technique.
</td></tr>

<tr><th valign=top>"SURF1"</th><td>
Draw a surface plot using the hidden surface removal technique.
</td></tr>

<tr><th valign=top>"SURF2"</th><td>
Draw a surface plot using colors to show the cell contents.
</td></tr>

<tr><th valign=top>"SURF3"</th><td>
Same as <tt>SURF</tt> with an additionial filled contour plot on top.
</td></tr>

<tr><th valign=top>"SURF4"</th><td>
Draw a surface using the Gouraud shading technique.
</td></tr>

<tr><th valign=top>"SURF5"</th><td>
Used with one of the options CYL, PSR and CYL this option allows to draw a
a filled contour plot.
</td></tr>

<tr><th valign=top>"SURF6"</th><td>
This option should not be used directly. It is used internally when the
CONT is used with option the option SAME on a 3D plot.
</td></tr>

<tr><th valign=top>"SURF7"</th><td>
Same as <tt>SURF2</tt> with an additionial line contour plot on top.
</td></tr>

</table>

See the limitations with <a href="#HP060a">the option "SAME"</a>.
<p>
The following example shows a 2D histogram plotted with the option
<tt>"SURF"</tt>. The option <tt>"SURF"</tt> draws a lego plot using the hidden
lines removal technique.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf = new TH2F("hsurf","Option SURF example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf->Fill(px-1,5*py);
      hsurf->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf->Draw("SURF");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF1"</tt>. The option <tt>"SURF1"</tt> draws a surface plot using the
hidden surface removal technique.  Combined with the option <tt>"SURF1"</tt>,
the option <tt>"Z"</tt> allows to display the color palette defined by
<tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf1 = new TH2F("hsurf1","Option SURF1 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf1->Fill(px-1,5*py);
      hsurf1->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf1->Draw("SURF1");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF2"</tt>. The option <tt>"SURF2"</tt> draws a surface plot using colors
to show the cell contents. Combined with the option <tt>"SURF2"</tt>, the option
<tt>"Z"</tt> allows to display the color palette defined by
<tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf2 = new TH2F("hsurf2","Option SURF2 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf2->Fill(px-1,5*py);
      hsurf2->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf2->Draw("SURF2");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF3"</tt>. The option <tt>"SURF3"</tt> draws a surface plot using the
hidden line removal technique with, in addition, a filled contour view drawn on the
top.  Combined with the option <tt>"SURF3"</tt>, the option <tt>"Z"</tt> allows
to display the color palette defined by <tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf3 = new TH2F("hsurf3","Option SURF3 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf3->Fill(px-1,5*py);
      hsurf3->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf3->Draw("SURF3");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF4"</tt>. The option <tt>"SURF4"</tt> draws a surface using the Gouraud
shading technique.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf4 = new TH2F("hsurf4","Option SURF4 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf4->Fill(px-1,5*py);
      hsurf4->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf4->SetFillColor(kOrange);
   hsurf4->Draw("SURF4");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF5 CYL"</tt>.  Combined with the option <tt>"SURF5"</tt>, the option
<tt>"Z"</tt> allows to display the color palette defined by <tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf5 = new TH2F("hsurf4","Option SURF5 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf5->Fill(px-1,5*py);
      hsurf5->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf5->SetFillColor(kOrange);
   hsurf5->Draw("SURF5 CYL");
   return c2;
}
End_Macro
Begin_Html

<p>The following example shows a 2D histogram plotted with the option
<tt>"SURF7"</tt>. The option <tt>"SURF7"</tt> draws a surface plot using the
hidden surfaces removal technique with, in addition, a line contour view drawn on the
top.  Combined with the option <tt>"SURF7"</tt>, the option <tt>"Z"</tt> allows
to display the color palette defined by <tt>gStyle->SetPalette()</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TH2F *hsurf7 = new TH2F("hsurf3","Option SURF7 example ",30,-4,4,30,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hsurf7->Fill(px-1,5*py);
      hsurf7->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hsurf7->Draw("SURF7");
   return c2;
}
End_Macro
Begin_Html

<p>As shown in the following example, when a contour plot is painted on top of a
surface plot using the option <tt>SAME</tt>, the contours appear in 3D on the
surface.

End_Html
Begin_Macro(source)
{
   TCanvas *c1=new TCanvas("c2","c2",600,400);
   int NBins = 50;
   double d = 2;
   TH2F* hsc = new TH2F("hsc", "Surface and contour with option SAME ", NBins, -d, d, NBins, -d, d);
   for (int bx = 1;  bx <= NBins; ++bx) {
      for (int by = 1;  by <= NBins; ++by) {
         double x = hsc->GetXaxis()->GetBinCenter(bx);
         double y = hsc->GetYaxis()->GetBinCenter(by);
         hsc->SetBinContent(bx, by, exp(-x*x)*exp(-y*y));
      }
   }
   gStyle->SetPalette(1);
   hsc->Draw("surf2");
   hsc->Draw("CONT1 SAME");
   return c2;
}
End_Macro
Begin_Html


<a name="HP19"></a><h3>Cylindrical, Polar, Spherical and PseudoRapidity/Phi options</h3>


Legos and surfaces plots are represented by default in Cartesian coordinates.
Combined with any <tt>"LEGOn"</tt> or <tt>"SURFn"</tt> options the following
options allow to draw a lego or a surface in other coordinates systems.

<table border=0>

<tr><th valign=top>"CYL"</th><td>
Use Cylindrical coordinates. The X coordinate is mapped on the angle and the Y
coordinate on the cylinder length.
</td></tr>

<tr><th valign=top>"POL"</th><td>
Use Polar coordinates. The X coordinate is mapped on the angle and the Y
coordinate on the radius.
</td></tr>

<tr><th valign=top>"SPH"</th><td>
Use Spherical coordinates. The X coordinate is mapped on the latitude and the
Y coordinate on the longitude.
</td></tr>

<tr><th valign=top>"PSR"</th><td>
Use PseudoRapidity/Phi coordinates. The X coordinate is mapped on Phi.
</td></tr>

</table>

<b>WARNING:</b> Axis are not drawn with these options.

<p>The following example shows the same histogram as a lego plot is the four
different coordinates systems.

End_Html
Begin_Macro(source)
{
   TCanvas *c3 = new TCanvas("c3","c3",600,400);
   c3->Divide(2,2);
   TH2F *hlcc = new TH2F("hlcc","Cylindrical coordinates",20,-4,4,20,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hlcc->Fill(px-1,5*py);
      hlcc->Fill(2+0.5*px,2*py-10.,0.1);
   }
   hlcc->SetFillColor(kYellow);
   c3->cd(1) ; hlcc->Draw("LEGO1 CYL");
   c3->cd(2) ; TH2F *hlpc = hlcc->DrawClone("LEGO1 POL");
   hlpc->SetTitle("Polar coordinates");
   c3->cd(3) ; TH2F *hlsc = hlcc->DrawClone("LEGO1 SPH");
   hlsc->SetTitle("Spherical coordinates");
   c3->cd(4) ; TH2F *hlprpc = hlcc->DrawClone("LEGO1 PSR");
   hlprpc->SetTitle("PseudoRapidity/Phi coordinates");
   return c3;
}
End_Macro
Begin_Html

<p>The following example shows the same histogram as a surface plot is the four
different coordinates systems.

End_Html
Begin_Macro(source)
{
   TCanvas *c4 = new TCanvas("c4","c4",600,400);
   c4->Divide(2,2);
   TH2F *hscc = new TH2F("hscc","Cylindrical coordinates",20,-4,4,20,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      hscc->Fill(px-1,5*py);
      hscc->Fill(2+0.5*px,2*py-10.,0.1);
   }
   gStyle->SetPalette(1);
   c4->cd(1) ; hscc->Draw("SURF1 CYL");
   c4->cd(2) ; TH2F *hspc = hscc->DrawClone("SURF1 POL");
   hspc->SetTitle("Polar coordinates");
   c4->cd(3) ; TH2F *hssc = hscc->DrawClone("SURF1 SPH");
   hssc->SetTitle("Spherical coordinates");
   c4->cd(4) ; TH2F *hsprpc = hscc->DrawClone("SURF1 PSR");
   hsprpc->SetTitle("PseudoRapidity/Phi coordinates");
   return c4;
}
End_Macro
Begin_Html


<a name="HP20"></a><h3>Base line for bar-charts and lego plots</h3>


By default the base line used to draw the boxes for bar-charts and lego plots is
the histogram minimum. It is possible to force this base line to be 0 with the
command:
<pre>
      gStyle->SetHistMinimumZero();
</pre>

End_Html
Begin_Macro(source)
{
   TCanvas *c5 = new TCanvas("c5","c5",700,400);
   c5->Divide(2,1);
   gStyle->SetHistMinimumZero(1);
   TH1F *hz1 = new TH1F("hz1","Bar-chart drawn from 0",20,-3,3);
   TH2F *hz2 = new TH2F("hz2","Lego plot drawn from 0",20,-3,3,20,-3,3);
   Int_t i;
   Double_t x,y;
   hz1->SetFillColor(kBlue);
   hz2->SetFillColor(kBlue);
   for (i=0;i<10000;i++) {
      x = gRandom->Gaus(0,1);
      y = gRandom->Gaus(0,1);
      if (x>0) {
         hz1->Fill(x,1);
         hz2->Fill(x,y,1);
      } else {
         hz1->Fill(x,-1);
         hz2->Fill(x,y,-2);
      }
   }
   c5->cd(1); hz1->Draw("bar2");
   c5->cd(2); hz2->Draw("lego1");
   return c5;
}
End_Macro
Begin_Html

<p>This option also works for horizontal plots. The example given in the section
<a href="http://root.cern.ch/root/html/THistPainter.html#HP100">
"The bar chart option"</a> appears as follow:

End_Html
Begin_Macro(source)
{
   int i;
   const Int_t nx = 8;
   char *os_X[nx]   = {"8","32","128","512","2048","8192","32768","131072"};
   float d_35_0[nx] = {0.75, -3.30, -0.92, 0.10, 0.08, -1.69, -1.29, -2.37};
   float d_35_1[nx] = {1.01, -3.02, -0.65, 0.37, 0.34, -1.42, -1.02, -2.10};

   TCanvas *cbh = new TCanvas("cbh","cbh",400,600);
   cbh->SetGrid();

   gStyle->SetHistMinimumZero();

   TH1F *h1bh = new TH1F("h1bh","Option HBAR centered on 0",nx,0,nx);
   h1bh->SetFillColor(4);
   h1bh->SetBarWidth(0.4);
   h1bh->SetBarOffset(0.1);
   h1bh->SetStats(0);
   h1bh->SetMinimum(-5);
   h1bh->SetMaximum(5);

   for (i=1; i<=nx; i++) {
      h1bh->Fill(os_X[i-1], d_35_0[i-1]);
      h1bh->GetXaxis()->SetBinLabel(i,os_X[i-1]);
   }

   h1bh->Draw("hbar");

   TH1F *h2bh = new TH1F("h2bh","h2bh",nx,0,nx);
   h2bh->SetFillColor(38);
   h2bh->SetBarWidth(0.4);
   h2bh->SetBarOffset(0.5);
   h2bh->SetStats(0);
   for (i=1;i<=nx;i++) h2bh->Fill(os_X[i-1], d_35_1[i-1]);

   h2bh->Draw("hbar same");

   return cbh;
}
End_Macro
Begin_Html


<a name="HP20a"></a><h3>TH2Poly Drawing</h3>


The following options are supported:

<table border=0>

<tr><th valign=top>"SCAT"</th><td>
Draw a scatter plot (default).
</td></tr>

<tr><th valign=top>"COL"</th><td>
Draw a color plot. All the none empty bins are painted. Empty bins are not
painted.
</td></tr>

<tr><th valign=top>"COLZ"</th><td>
Same as "COL". In addition the color palette is also drawn.
</td></tr>

<tr><th valign=top>"TEXT"</th><td>
Draw bin contents as text (format set via <tt>gStyle->SetPaintTextFormat</tt>).
</td></tr>

<tr><th valign=top>"TEXTN"</th><td>
Draw bin names as text.
</td></tr>

<tr><th valign=top>"TEXTnn"</th><td>
Draw bin contents as text at angle nn (0 < nn < 90).
</td></tr>

<tr><th valign=top>"L"</th><td>
Draw the bins boundaries as lines.
The lines attibutes are the TGraphs ones.
</td></tr>

<tr><th valign=top>"P"</th><td>
Draw the bins boundaries as markers.
The markers attibutes are the TGraphs ones.
</td></tr>

<tr><th valign=top>"F"</th><td>
Draw the bins boundaries as filled polygons.
The filled polygons attibutes are the TGraphs ones.
</td></tr>

</table>

<p><a href="http://root.cern.ch/root/html/TH2Poly.html"><tt>TH2Poly</tt></a> can
be drawn as a color plot (option COL). <tt>TH2Poly</tt> bins can have any
shapes. The bins are defined as graphs. The following macro is a very simple
example showing how to book a TH2Poly and draw it.
End_Html
Begin_Macro(source)
{
   TCanvas *ch2p1 = new TCanvas("ch2p1","ch2p1",600,400);
   TH2Poly *h2p = new TH2Poly();
   h2p->SetName("h2poly_name");
   h2p->SetTitle("h2poly_title");
   Double_t x1[] = {0, 5, 6};
   Double_t y1[] = {0, 0, 5};
   Double_t x2[] = {0, -1, -1, 0};
   Double_t y2[] = {0, 0, -1, 3};
   Double_t x3[] = {4, 3, 0, 1, 2.4};
   Double_t y3[] = {4, 3.7, 1, 4.7, 3.5};
   h2p->AddBin(3, x1, y1);
   h2p->AddBin(4, x2, y2);
   h2p->AddBin(4, x3, y3);
   h2p->Fill(0.1, 0.01, 3);
   h2p->Fill(-0.5, -0.5, 7);
   h2p->Fill(-0.7, -0.5, 1);
   h2p->Fill(1, 3, 1.5);
   Double_t fx[] = {0.1, -0.5, -0.7, 1};
   Double_t fy[] = {0.01, -0.5, -0.5, 3};
   Double_t fw[] = {3, 1, 1, 1.5};
   h2p->FillN(4, fx, fy, fw);
   gStyle->SetPalette(1);
   h2p->Draw("col");
   return ch2p1;
}
End_Macro
Begin_Html

<p>Rectangular bins are a frequent case. The special version of
the <tt>AddBin</tt> method allows to define them more easily like
shown in the following example.

End_Html
Begin_Macro(source)
../../../tutorials/hist/th2polyBoxes.C
End_Macro
Begin_Html

<p>One <tt>TH2Poly</tt> bin can be a list of polygons. Such bins are defined
by calling <tt>AddBin</tt> with a <tt>TMultiGraph</tt>. The following example
shows a such case:
End_Html
Begin_Macro(source)
{
   TCanvas *ch2p2 = new TCanvas("ch2p2","ch2p2",600,400);

   Int_t i, bin;
   const Int_t nx = 48;
   char *states [nx] = {
      "alabama",      "arizona",        "arkansas",       "california",
      "colorado",     "connecticut",    "delaware",       "florida",
      "georgia",      "idaho",          "illinois",       "indiana",
      "iowa",         "kansas",         "kentucky",       "louisiana",
      "maine",        "maryland",       "massachusetts",  "michigan",
      "minnesota",    "mississippi",    "missouri",       "montana",
      "nebraska",     "nevada",         "new_hampshire",  "new_jersey",
      "new_mexico",   "new_york",       "north_carolina", "north_dakota",
      "ohio",         "oklahoma",       "oregon",         "pennsylvania",
      "rhode_island", "south_carolina", "south_dakota",   "tennessee",
      "texas",        "utah",           "vermont",        "virginia",
      "washington",   "west_virginia",  "wisconsin",      "wyoming"
   };
   Float_t pop[nx] = {
    4708708, 6595778,  2889450, 36961664, 5024748,  3518288,  885122, 18537969,
    9829211, 1545801, 12910409,  6423113, 3007856,  2818747, 4314113,  4492076,
    1318301, 5699478,  6593587,  9969727, 5266214,  2951996, 5987580,   974989,
    1796619, 2643085,  1324575,  8707739, 2009671, 19541453, 9380884,   646844,
   11542645, 3687050,  3825657, 12604767, 1053209,  4561242,  812383,  6296254,
   24782302, 2784572,   621760,  7882590, 6664195,  1819777, 5654774,   544270
   };

   Double_t lon1 = -130;
   Double_t lon2 = -65;
   Double_t lat1 = 24;
   Double_t lat2 = 50;
   TH2Poly *p = new TH2Poly("USA","USA Population",lon1,lon2,lat1,lat2);

   TFile *f;
   f = TFile::Open("http://root.cern.ch/files/usa.root");

   TMultiGraph *mg;
   TKey *key;
   TIter nextkey(gDirectory->GetListOfKeys());
   while (key = (TKey*)nextkey()) {
      obj = key->ReadObj();
      if (obj->InheritsFrom("TMultiGraph")) {
         mg = (TMultiGraph*)obj;
         bin = p->AddBin(mg);
      }
   }

   for (i=0; i<nx; i++) p->Fill(states[i], pop[i]);

   gStyle->SetOptStat(11);
   gStyle->SetPalette(1);
   p->Draw("COLZ L");
   return ch2p2;
}
End_Macro
Begin_Html

<p> <tt>TH2Poly</tt> histograms can also be plotted using the GL interface using
the option "GLLEGO".

<a name="HP21"></a><h3>The SPEC option</h3>


This option allows to use the <tt>TSpectrum2Painter</tt> tools. See the full
documentation in <tt>TSpectrum2Painter::PaintSpectrum</tt>.


<a name="HP22"></a><h3>Option "Z" : Adding the color palette on the right side of the pad</h3>


When this option is specified, a color palette with an axis indicating the value
of the corresponding color is drawn on the right side of the picture. In case,
not enough space is left, one can increase the size of the right margin by
calling <tt>TPad::SetRightMargin()</tt>. The attributes used to display the
palette axis values are taken from the Z axis of the object. For example, to
set the labels size on the palette axis do:
<pre>
      hist->GetZaxis()->SetLabelSize().
</pre>
<b>WARNING:</b> The palette axis is always drawn vertically.


<a name="HP23"></a><h3>Setting the color palette</h3>


To change the color palette <tt>TStyle::SetPalette</tt> should be used, eg:
<pre>
      gStyle->SetPalette(ncolors,colors);
</pre>
For example the option <tt>"COL"</tt> draws a 2D histogram with cells
represented by a box filled with a color index which is a function
of the cell content.
If the cell content is N, the color index used will be the color number
in <tt>colors[N]</tt>, etc. If the maximum cell content is greater than
<tt>ncolors</tt>, all cell contents are scaled to <tt>ncolors</tt>.

<p>If <tt> ncolors <= 0</tt>, a default palette (see below) of 50 colors is
defined. This palette is recommended for pads, labels ...

<p>If <tt>ncolors == 1 && colors == 0</tt>, a pretty palette with a violet to
red spectrum is created. It is recommended you use this palette when drawing
legos, surfaces or contours.

<p>If ncolors > 50 and colors=0, the DeepSea palette is used.
(see <tt>TColor::CreateGradientColorTable</tt> for more details)

<p> If <tt>ncolors > 0 && colors == 0</tt>, the default palette is used
with a maximum of ncolors.

<p> The default palette defines:
<ul>
<li> index  0  to  9 : shades of grey
<li> index 10  to 19 : shades of brown
<li> index 20  to 29 : shades of blue
<li> index 30  to 39 : shades of red
<li> index 40  to 49 : basic colors
</ul>
The color numbers specified in the palette can be viewed by selecting
the item <tt>"colors"</tt> in the <tt>"VIEW"</tt> menu of the canvas tool bar.
The red, green, and blue components of a color can be changed thanks to
<tt>TColor::SetRGB()</tt>.


<a name="HP24"></a><h3>Drawing a sub-range of a 2D histogram; the [cutg] option</h3>


Using a <tt>TCutG</tt> object, it is possible to draw a sub-range of a 2D
histogram. One must create a graphical cut (mouse or C++) and specify the name
of the cut between <tt>[]</tt> in the <tt>Draw()</tt> option.
For example, with a <tt>TCutG</tt> named <tt>"cutg"</tt>, one can call:
<pre>
      myhist->Draw("surf1 [cutg]");
</pre>
To invert the cut, it is enough to put a <tt>"-"</tt> in front of its name:
<pre>
      myhist->Draw("surf1 [-cutg]");
</pre>
It is possible to apply several cuts (<tt>","</tt> means logical AND):
<pre>
      myhist->Draw("surf1 [cutg1,cutg2]");
</pre>

End_Html
Begin_Macro(source)
../../../tutorials/fit/fit2a.C
End_Macro
Begin_Html


<a name="HP25"></a><h3>Drawing options for 3D histograms</h3>


<table border=0>

<tr><th valign=top>"ISO"</th><td>
Draw a Gouraud shaded 3d iso surface through a 3d histogram. It paints one
surface at the value computed as follow:
<tt>SumOfWeights/(NbinsX*NbinsY*NbinsZ)</tt>
</td></tr>

<tr><th valign=top>"BOX"</th><td>
Draw a for each cell with volume proportional to the content's absolute value.
</td></tr>

</table>

By default, like 2D histograms, 3D histograms are drawn as scatter plots.

<p>The following example shows a 3D histogram plotted as a scatter plot.

End_Html
Begin_Macro(source)
{
   TCanvas *c06 = new TCanvas("c06","c06",600,400);
   gStyle->SetOptStat(kFALSE);
   TH3F *h3scat = new TH3F("h3scat","Option SCAT (default) ",15,-2,2,15,-2,2,15,0,4);
   Double_t x, y, z;
   for (Int_t i=0;i<10000;i++) {
      gRandom->Rannor(x, y);
      z = x*x + y*y;
      h3scat->Fill(x,y,z);
   }
   h3scat->Draw();
   return c06;
}
End_Macro
Begin_Html

<p>The following example shows a 3D histogram plotted with the option <tt>"BOX"</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c16 = new TCanvas("c16","c16",600,400);
   gStyle->SetOptStat(kFALSE);
   TH3F *h3box = new TH3F("h3box","Option BOX",15,-2,2,15,-2,2,15,0,4);
   Double_t x, y, z;
   for (Int_t i=0;i<10000;i++) {
      gRandom->Rannor(x, y);
      z = x*x + y*y;
      h3box->Fill(x,y,z);
   }
   h3box->Draw("BOX");
   return c16;
}
End_Macro
Begin_Html

<p>The following example shows a 3D histogram plotted with the option <tt>"ISO"</tt>.

End_Html
Begin_Macro(source)
{
   TCanvas *c26 = new TCanvas("c26","c26",600,400);
   gStyle->SetOptStat(kFALSE);
   TH3F *h3iso = new TH3F("h3iso","Option ISO",15,-2,2,15,-2,2,15,0,4);
   Double_t x, y, z;
   for (Int_t i=0;i<10000;i++) {
      gRandom->Rannor(x, y);
      z = x*x + y*y;
      h3iso->Fill(x,y,z);
   }
   h3iso->SetFillColor(kCyan);
   h3iso->Draw("ISO");
   return c26;
}
End_Macro
Begin_Html


<a name="HP26"></a><h3>Drawing option for histograms' stacks</h3>


Stacks of histograms are managed with the <tt>THStack</tt>. A <tt>THStack</tt>
is a collection of <tt>TH1</tt> (or derived) objects. For painting only the
<tt>THStack</tt> containing <tt>TH1</tt> only or
<tt>THStack</tt> containing <tt>TH2</tt> only will be considered.

<p>By default, histograms are shown stacked:
<ol>
<li> The first histogram is paint.
<li> The the sum of the first and second, etc...
</ol>
If the option <tt>"NOSTACK"</tt> is specified, the histograms are all paint in
the same pad as if the option <tt>"SAME"</tt> had been specified. This allows to
compute X and Y scales common to all the histograms, like
<tt>TMultiGraph</tt> does for graphs.

<p>If the option <tt>"PADS"</tt> is specified, the current pad/canvas is
subdivided into a number of pads equal to the number of histograms and each
histogram is paint into a separate pad.

<p>The following example shows various types of stacks.

End_Html
Begin_Macro(source)
../../../tutorials/hist/hstack.C
End_Macro
Begin_Html

<p>If at least one of the histograms in the stack has errors, the whole stack is
visualized by default with error bars. To visualize it without errors the
option <tt>"HIST"</tt> should be used.

End_Html
Begin_Macro(source)
{
   TCanvas *cst1 = new TCanvas("cst1","cst1",700,400);
   cst1->Divide(2,1);

   TH1F * hst11 = new TH1F("hst11", "", 20, -10, 10);
   hst11->Sumw2();
   hst11->FillRandom("gaus", 1000);
   hst11->SetFillColor(kViolet);
   hst11->SetLineColor(kViolet);

   TH1F * hst12 = new TH1F("hst12", "", 20, -10, 10);
   hst12->FillRandom("gaus", 500);
   hst12->SetFillColor(kBlue);
   hst12->SetLineColor(kBlue);

   THStack st1("st1", "st1");
   st1.Add(hst11);
   st1.Add(hst12);

   cst1->cd(1); st1.Draw();
   cst1->cd(2); st1.Draw("hist");

   return cst1;
}
End_Macro
Begin_Html

<a name="HP27"></a><h3>Drawing of 3D implicit functions</h3>


3D implicit functions (<tt>TF3</tt>) can be drawn as iso-surfaces.
The implicit function f(x,y,z) = 0 is drawn in cartesian coordinates.
In the following example the options "FB" and "BB" suppress the
"Front Box" and "Back Box" around the plot.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",600,400);
   TF3 *f3 = new TF3("f3","sin(x*x+y*y+z*z-36)",-2,2,-2,2,-2,2);
   f3->SetClippingBoxOn(0,0,0);
   f3->SetFillColor(30);
   f3->SetLineColor(15);
   f3->Draw("FBBB");
   return c2;
}
End_Macro
Begin_Html


<a name="HP28"></a><h3>Associated functions drawing</h3>


An associated function is created by <tt>TH1::Fit</tt>. More than on fitted
function can be associated with one histogram (see <tt>TH1::Fit</tt>).

<p>A <tt>TF1</tt> object <tt>f1</tt> can be added to the list of associated
functions of an histogram <tt>h</tt> without calling <tt>TH1::Fit</tt>
simply doing:
<pre>
      h->GetListOfFunctions()->Add(f1);
</pre>
or
<pre>
      h->GetListOfFunctions()->Add(f1,someoption);
</pre>
To retrieve a function by name from this list, do:
<pre>
      TF1 *f1 = (TF1*)h->GetListOfFunctions()->FindObject(name);
</pre>
or
<pre>
      TF1 *f1 = h->GetFunction(name);
</pre>
Associated functions are automatically painted when an histogram is drawn.
To avoid the painting of the associated functions the option <tt>HIST</tt>
should be added to the list of the options used to paint the histogram.


<a name="HP29"></a><h3>Drawing using OpenGL</h3>


The class <tt>TGLHistPainter</tt> allows to paint data set using the OpenGL 3D
graphics library. The plotting options start with <tt>GL</tt> keyword.
In addition, in order to inform canvases that OpenGL should be used to render
3D representations, the following option should be set:
<pre>
      gStyle->SetCanvasPreferGL(true);
</pre>

<a name="HP29a"></a><h4><u>General information: plot types and supported options</u></h4>

The following types of plots are provided:

<p>For lego plots the supported options are:

<table border=0>

<tr><th valign=top>"GLLEGO"</th><td>
Draw a lego plot. It works also for <tt>TH2Poly</tt>.
</td></tr>

<tr><th valign=top>"GLLEGO2</th><td>
Bins with color levels.
</td></tr>

<tr><th valign=top>"GLLEGO3</th><td>
Cylindrical bars.
</td></tr>

</table>

<p>Lego painter in cartesian supports logarithmic scales for X, Y, Z.
In polar only Z axis can be logarithmic, in cylindrical only Y.

<p>For surface plots (<tt>TF2</tt> and <tt>TH2</tt>) the supported options are:

<table border=0>

<tr><th valign=top>"GLSURF" </th><td>
Draw a surface.
</td></tr>

<tr><th valign=top>"GLSURF1"</th><td>
Surface with color levels
</td></tr>

<tr><th valign=top>"GLSURF2"</th><td>
The same as "GLSURF1" but without polygon outlines.
</td></tr>

<tr><th valign=top>"GLSURF3"</th><td>
Color level projection on top of plot (works only in cartesian coordinate
system).
</td></tr>

<tr><th valign=top>"GLSURF4"</th><td>
Same as "GLSURF" but without polygon outlines.
</td></tr>

</table>

The surface painting in cartesian coordinates supports logarithmic scales along
X, Y, Z axis. In polar coordinates only the Z axis can be logarithmic,
in cylindrical coordinates only the Y axis.

<p>Additional options to SURF and LEGO - Coordinate systems:

<table border=0>

<tr><th valign=top>" "</th><td>
Default, cartesian coordinates system.
</td></tr>

<tr><th valign=top>"POL"</th><td>
Polar coordinates system.
</td></tr>

<tr><th valign=top>"CYL"</th><td>
Cylindrical coordinates system.
</td></tr>

<tr><th valign=top>"SPH"</th><td>
Spherical coordinates system.
</td></tr>

</table>

<a name="HP29b"></a><h4><u>TH3 as boxes (spheres)</u></h4>

The supported options are:

<table border=0>

<tr><th valign=top>GLBOX" </th><td>
TH3 as a set of boxes, size of box is proportional to bin content.
</td></tr>

<tr><th valign=top>GLBOX1"</th><td>
The same as "glbox", but spheres are drawn instead of boxes.
</td></tr>

</table>

<a name="HP29c"></a><h4><u>TH3 as iso-surface(s)</u></h4>

The supported option is:

<table border=0>

<tr><th valign=top>"GLISO" </th><td>
TH3 is drawn using iso-surfaces.
</td></tr>

</table>

<a name="HP29d"></a><h4><u>TF3 (implicit function)</u></h4>

The supported option is:

<table border=0>

<tr><th valign=top>GLTF3" </th><td>
Draw a TF3.
</td></tr>

</table>

<a name="HP29e"></a><h4><u>Parametric surfaces</u></h4>

<tt>$ROOTSYS/tutorials/gl/glparametric.C</tt> shows how to create parametric
equations and visualize the surface.

<a name="HP29f"></a><h4><u>Interaction with the plots</u></h4>

All the interactions are implemented via standard methods
<tt>DistancetoPrimitive()</tt> and <tt>ExecuteEvent()</tt>. That's why all the
interactions with the OpenGL plots are possible only when the mouse cursor is
in the plot's area (the plot's area is the part of a the pad occupied by
gl-produced picture). If the mouse cursor is not above gl-picture, the standard
pad interaction is performed.

<a name="HP29g"></a><h4><u>Selectable parts</u></h4>

Different parts of the plot can be selected:
<ul>
<li> xoz, yoz, xoy back planes:
When such a plane selected, it's highlighted in green if the
dynamic slicing by this plane is supported, and it's
highlighted in red, if the dynamic slicing is not supported.
<li> The plot itself:
On surfaces, the selected surface is outlined in red. (TF3 and
ISO are not outlined). On lego plots, the selected bin is
highlighted. The bin number and content are displayed in pad's
status bar. In box plots, the box or sphere is highlighted and
the bin info is displayed in pad's status bar.
</ul>

<a name="HP29h"></a><h4><u>Rotation and zooming</u></h4>

<ul>
<li>Rotation:
When the plot is selected, it can be rotated by pressing and
holding the left mouse button and move the cursor.
<li>Zoom/Unzoom:
Mouse wheel or 'j', 'J', 'k', 'K' keys.
</ul>

<a name="HP29i"></a><h4><u>Panning</u></h4>

The selected plot can be moved in a pad's area by pressing and
holding the left mouse button and the shift key.

<a name="HP29j"></a><h4><u>Box cut</u></h4>

Surface, iso, box, TF3 and parametric painters support box cut by
pressing the 'c' or 'C' key when the mouse cursor is in a plot's
area. That will display a transparent box, cutting away part of the
surface (or boxes) in order to show internal part of plot. This box
can be moved inside the plot's area (the full size of the box is
equal to the plot's surrounding box) by selecting one of the box
cut axes and pressing the left mouse button to move it.

<a name="HP29k"></a><h4><u>Plot specific interactions (dynamic slicing etc.)</u></h4>

Currently, all gl-plots support some form of slicing. When back plane
is selected (and if it's highlighted in green) you can press and hold
left mouse button and shift key and move this back plane inside
plot's area, creating the slice. During this "slicing" plot becomes
semi-transparent. To remove all slices (and projected curves for
surfaces) double click with left mouse button in a plot's area.

<a name="HP29l"></a><h4><u>Surface with option "GLSURF"</u></h4>

The surface profile is displayed on the slicing plane.
The profile projection is drawn on the back plane
by pressing <tt>'p'</tt> or <tt>'P'</tt> key.

<a name="HP29m"></a><h4><u>TF3</u></h4>

The contour plot is drawn on the slicing plane. For TF3 the color
scheme can be changed by pressing 's' or 'S'.

<a name="HP29n"></a><h4><u>Box</u></h4>

The contour plot corresponding to slice plane position is drawn in real time.

<a name="HP29o"></a><h4><u>Iso</u></h4>

Slicing is similar to "GLBOX" option.

<a name="HP29p"></a><h4><u>Parametric plot</u></h4>

No slicing. Additional keys: 's' or 'S' to change color scheme -
about 20 color schemes supported ('s' for "scheme"); 'l' or 'L' to
increase number of polygons ('l' for "level" of details), 'w' or 'W'
to show outlines ('w' for "wireframe").

End_Html */

TH1 *gCurrentHist = 0;

Hoption_t Hoption;
Hparam_t  Hparam;

const Int_t kNMAX = 2000;

const Int_t kMAXCONTOUR  = 104;
const UInt_t kCannotRotate = BIT(11);

static TString gStringEntries;
static TString gStringMean;
static TString gStringMeanX;
static TString gStringMeanY;
static TString gStringMeanZ;
static TString gStringRMS;
static TString gStringRMSX;
static TString gStringRMSY;
static TString gStringRMSZ;
static TString gStringUnderflow;
static TString gStringOverflow;
static TString gStringIntegral;
static TString gStringSkewness;
static TString gStringSkewnessX;
static TString gStringSkewnessY;
static TString gStringSkewnessZ;
static TString gStringKurtosis;
static TString gStringKurtosisX;
static TString gStringKurtosisY;
static TString gStringKurtosisZ;

ClassImp(THistPainter)


//______________________________________________________________________________
THistPainter::THistPainter()
{
   /* Begin_html
   Default constructor.
   End_html */

   fH = 0;
   fXaxis = 0;
   fYaxis = 0;
   fZaxis = 0;
   fFunctions = 0;
   fXbuf  = 0;
   fYbuf  = 0;
   fNcuts = 0;
   fStack = 0;
   fLego  = 0;
   fPie   = 0;
   fGraph2DPainter = 0;
   fShowProjection = 0;
   fShowOption = "";
   for (int i=0; i<kMaxCuts; i++) {
      fCuts[i] = 0;
      fCutsOpt[i] = 0;
   }

   gStringEntries   = gEnv->GetValue("Hist.Stats.Entries",   "Entries");
   gStringMean      = gEnv->GetValue("Hist.Stats.Mean",      "Mean");
   gStringMeanX     = gEnv->GetValue("Hist.Stats.MeanX",     "Mean x");
   gStringMeanY     = gEnv->GetValue("Hist.Stats.MeanY",     "Mean y");
   gStringMeanZ     = gEnv->GetValue("Hist.Stats.MeanZ",     "Mean z");
   gStringRMS       = gEnv->GetValue("Hist.Stats.RMS",       "RMS");
   gStringRMSX      = gEnv->GetValue("Hist.Stats.RMSX",      "RMS x");
   gStringRMSY      = gEnv->GetValue("Hist.Stats.RMSY",      "RMS y");
   gStringRMSZ      = gEnv->GetValue("Hist.Stats.RMSZ",      "RMS z");
   gStringUnderflow = gEnv->GetValue("Hist.Stats.Underflow", "Underflow");
   gStringOverflow  = gEnv->GetValue("Hist.Stats.Overflow",  "Overflow");
   gStringIntegral  = gEnv->GetValue("Hist.Stats.Integral",  "Integral");
   gStringSkewness  = gEnv->GetValue("Hist.Stats.Skewness",  "Skewness");
   gStringSkewnessX = gEnv->GetValue("Hist.Stats.SkewnessX", "Skewness x");
   gStringSkewnessY = gEnv->GetValue("Hist.Stats.SkewnessY", "Skewness y");
   gStringSkewnessZ = gEnv->GetValue("Hist.Stats.SkewnessZ", "Skewness z");
   gStringKurtosis  = gEnv->GetValue("Hist.Stats.Kurtosis",  "Kurtosis");
   gStringKurtosisX = gEnv->GetValue("Hist.Stats.KurtosisX", "Kurtosis x");
   gStringKurtosisY = gEnv->GetValue("Hist.Stats.KurtosisY", "Kurtosis y");
   gStringKurtosisZ = gEnv->GetValue("Hist.Stats.KurtosisZ", "Kurtosis z");
}


//______________________________________________________________________________
THistPainter::~THistPainter()
{
   /* Begin_html
   Default destructor.
   End_html */
}


//______________________________________________________________________________
Int_t THistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   /* Begin_html
   Compute the distance from the point px,py to a line.
   <p>
   Compute the closest distance of approach from point px,py to elements of
   an histogram. The distance is computed in pixels units.
   <p>
   Algorithm:<br>
   Currently, this simple model computes the distance from the mouse to the
   histogram contour only.
   End_html */

   const Int_t big = 9999;
   const Int_t kMaxDiff = 7;

   if (fPie) return fPie->DistancetoPrimitive(px, py);

   Double_t x  = gPad->AbsPixeltoX(px);
   Double_t x1 = gPad->AbsPixeltoX(px+1);

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   Int_t curdist = big;
   Int_t yxaxis, dyaxis,xyaxis, dxaxis;
   Bool_t dsame;
   TString doption = gPad->GetPadPointer()->GetDrawOption();
   Double_t factor = 1;
   if (fH->GetNormFactor() != 0) {
      factor = fH->GetNormFactor()/fH->GetSumOfWeights();
   }
   //     return if point is not in the histogram area

   //     If a 3D view exists, check distance to axis
   TView *view = gPad->GetView();
   Int_t d1,d2,d3;
   if (view && Hoption.Contour != 14) {
      Double_t ratio;
      d3 = view->GetDistancetoAxis(3, px, py, ratio);
      if (d3 <= kMaxDiff) {gPad->SetSelected(fZaxis); return 0;}
      d1 = view->GetDistancetoAxis(1, px, py, ratio);
      if (d1 <= kMaxDiff) {gPad->SetSelected(fXaxis); return 0;}
      d2 = view->GetDistancetoAxis(2, px, py, ratio);
      if (d2 <= kMaxDiff) {gPad->SetSelected(fYaxis); return 0;}
      if ( px > puxmin && px < puxmax && py > puymax && py < puymin) curdist = 1;
      goto FUNCTIONS;
   }
   //     check if point is close to an axis
   doption.ToLower();
   dsame = kFALSE;
   if (doption.Contains("same")) dsame = kTRUE;

   dyaxis = Int_t(2*(puymin-puymax)*fYaxis->GetLabelSize());
   if (doption.Contains("y+")) {
      xyaxis = puxmax + Int_t((puxmax-puxmin)*fYaxis->GetLabelOffset());
      if (px <= xyaxis+dyaxis && px >= xyaxis && py >puymax && py < puymin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fYaxis);
            else                    gPad->SetSelected(fXaxis);
            return 0;
         }
      }
   } else {
      xyaxis = puxmin - Int_t((puxmax-puxmin)*fYaxis->GetLabelOffset());
      if (px >= xyaxis-dyaxis && px <= xyaxis && py >puymax && py < puymin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fYaxis);
            else                    gPad->SetSelected(fXaxis);
            return 0;
         }
      }
   }

   dxaxis = Int_t((puymin-puymax)*fXaxis->GetLabelSize());
   if (doption.Contains("x+")) {
      yxaxis = puymax - Int_t((puymin-puymax)*fXaxis->GetLabelOffset());
      if (py >= yxaxis-dxaxis && py <= yxaxis && px <puxmax && px > puxmin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fXaxis);
            else                    gPad->SetSelected(fYaxis);
            return 0;
         }
      }
   } else {
      yxaxis = puymin + Int_t((puymin-puymax)*fXaxis->GetLabelOffset());
      if (yxaxis < puymin) yxaxis = puymin;
      if (py <= yxaxis+dxaxis && py >= yxaxis && px <puxmax && px > puxmin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fXaxis);
            else                    gPad->SetSelected(fYaxis);
            return 0;
         }
      }
   }

   //     if object is 2D or 3D return this object
   if (fH->GetDimension() == 2) {
      if (fH->InheritsFrom(TH2Poly::Class())) {
         TH2Poly *th2 = (TH2Poly*)fH;
         Double_t xmin, ymin, xmax, ymax;
         gPad->GetRangeAxis(xmin, ymin, xmax, ymax);
         Double_t pxu = gPad->AbsPixeltoX(px);
         Double_t pyu = gPad->AbsPixeltoY(py);
         if ((pxu>xmax) || (pxu < xmin) || (pyu>ymax) || (pyu < ymin)) {
            curdist = big;
            goto FUNCTIONS;
         } else {
            Int_t bin = th2->FindBin(pxu, pyu);
            if (bin>0) curdist = 1;
            else       curdist = big;
            goto FUNCTIONS;
         }
      }
      Int_t delta2 = 5; //Give a margin of delta2 pixels to be in the 2-d area
      if ( px > puxmin + delta2
        && px < puxmax - delta2
        && py > puymax + delta2
        && py < puymin - delta2) {curdist =1; goto FUNCTIONS;}
   }

   //     point is inside histogram area. Find channel number
   if (gPad->IsVertical()) {
      Int_t bin      = fXaxis->FindFixBin(gPad->PadtoX(x));
      Int_t binsup   = fXaxis->FindFixBin(gPad->PadtoX(x1));
      Double_t binval = factor*fH->GetBinContent(bin);
      Int_t pybin    = gPad->YtoAbsPixel(gPad->YtoPad(binval));
      if (binval == 0 && pybin < puymin) pybin = 10000;
      // special case if more than one bin for the pixel
      if (binsup-bin>1) {
         Double_t binvalmin, binvalmax;
         binvalmin=binval;
         binvalmax=binval;
         for (Int_t ibin=bin+1; ibin<binsup; ibin++) {
            Double_t binvaltmp = factor*fH->GetBinContent(ibin);
            if (binvalmin>binvaltmp) binvalmin=binvaltmp;
            if (binvalmax<binvaltmp) binvalmax=binvaltmp;
         }
         Int_t pybinmin = gPad->YtoAbsPixel(gPad->YtoPad(binvalmax));
         Int_t pybinmax = gPad->YtoAbsPixel(gPad->YtoPad(binvalmin));
         if (py<pybinmax+kMaxDiff/2 && py>pybinmin-kMaxDiff/2) pybin = py;
      }
      if (TMath::Abs(py - pybin) <= kMaxDiff) return TMath::Abs(py - pybin);
   } else {
      Double_t y  = gPad->AbsPixeltoY(py);
      Double_t y1 = gPad->AbsPixeltoY(py+1);
      Int_t bin      = fXaxis->FindFixBin(gPad->PadtoY(y));
      Int_t binsup   = fXaxis->FindFixBin(gPad->PadtoY(y1));
      Double_t binval = factor*fH->GetBinContent(bin);
      Int_t pxbin    = gPad->XtoAbsPixel(gPad->XtoPad(binval));
      if (binval == 0 && pxbin > puxmin) pxbin = 10000;
      // special case if more than one bin for the pixel
      if (binsup-bin>1) {
         Double_t binvalmin, binvalmax;
         binvalmin=binval;
         binvalmax=binval;
         for (Int_t ibin=bin+1; ibin<binsup; ibin++) {
            Double_t binvaltmp = factor*fH->GetBinContent(ibin);
            if (binvalmin>binvaltmp) binvalmin=binvaltmp;
            if (binvalmax<binvaltmp) binvalmax=binvaltmp;
         }
         Int_t pxbinmin = gPad->XtoAbsPixel(gPad->XtoPad(binvalmax));
         Int_t pxbinmax = gPad->XtoAbsPixel(gPad->XtoPad(binvalmin));
         if (px<pxbinmax+kMaxDiff/2 && px>pxbinmin-kMaxDiff/2) pxbin = px;
      }
      if (TMath::Abs(px - pxbin) <= kMaxDiff) return TMath::Abs(px - pxbin);
   }
   //     Loop on the list of associated functions and user objects
FUNCTIONS:
   TObject *f;
   TIter   next(fFunctions);
   while ((f = (TObject*) next())) {
      Int_t dist;
      if (f->InheritsFrom(TF1::Class())) dist = f->DistancetoPrimitive(-px,py);
      else                               dist = f->DistancetoPrimitive(px,py);
      if (dist < kMaxDiff) {gPad->SetSelected(f); return dist;}
   }
   return curdist;
}


//______________________________________________________________________________
void THistPainter::DrawPanel()
{
   /* Begin_html
   Display a panel with all histogram drawing options.
   End_html */

   gCurrentHist = fH;
   if (!gPad) {
      Error("DrawPanel", "need to draw histogram first");
      return;
   }
   TVirtualPadEditor *editor = TVirtualPadEditor::GetPadEditor();
   editor->Show();
   gROOT->ProcessLine(Form("((TCanvas*)0x%lx)->Selected((TVirtualPad*)0x%lx,(TObject*)0x%lx,1)",
                           (ULong_t)gPad->GetCanvas(), (ULong_t)gPad, (ULong_t)fH));
}


//______________________________________________________________________________
void THistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   /* Begin_html
   Execute the actions corresponding to "event".
   <p>
   This function is called when a histogram is clicked with the locator at
   the pixel position px,py.
   End_html */

   static Int_t bin, px1, py1, px2, py2, pyold;
   Double_t xlow, xup, ylow, binval, x, baroffset, barwidth, binwidth;

   if (!gPad->IsEditable()) return;

   if (fPie) {
      fPie->ExecuteEvent(event, px, py);
      return;
   }
   //     come here if we have a lego/surface in the pad
   TView *view = gPad->GetView();

   if (!fShowProjection && view && view->TestBit(kCannotRotate) == 0) {
      view->ExecuteRotateView(event, px, py);
      return;
   }

   Double_t factor = 1;
   if (fH->GetNormFactor() != 0) {
      factor = fH->GetNormFactor()/fH->GetSumOfWeights();
   }

   switch (event) {

   case kButton1Down:

      gVirtualX->SetLineColor(-1);
      fH->TAttLine::Modify();

      // No break !!!

   case kMouseMotion:

      if (fShowProjection) {ShowProjection3(px,py); break;}

      if (Hoption.Bar) {
         baroffset = fH->GetBarOffset();
         barwidth  = fH->GetBarWidth();
      } else {
         baroffset = 0;
         barwidth  = 1;
      }
      x        = gPad->AbsPixeltoX(px);
      bin      = fXaxis->FindFixBin(gPad->PadtoX(x));
      binwidth = fH->GetBinWidth(bin);
      xlow     = gPad->XtoPad(fH->GetBinLowEdge(bin) + baroffset*binwidth);
      xup      = gPad->XtoPad(xlow + barwidth*binwidth);
      ylow     = gPad->GetUymin();
      px1      = gPad->XtoAbsPixel(xlow);
      px2      = gPad->XtoAbsPixel(xup);
      py1      = gPad->YtoAbsPixel(ylow);
      py2      = py;
      pyold    = py;
      if (gROOT->GetEditHistograms()) gPad->SetCursor(kArrowVer);
      else                            gPad->SetCursor(kPointer);

      break;

   case kButton1Motion:

   if (gROOT->GetEditHistograms()) {
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //    Draw the old box
         py2 += py - pyold;
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //    Draw the new box
         pyold = py;
   }

      break;

   case kButton1Up:

      if (gROOT->GetEditHistograms()) {
         binval = gPad->PadtoY(gPad->AbsPixeltoY(py2))/factor;
         fH->SetBinContent(bin,binval);
         PaintInit();   // recalculate Hparam structure and recalculate range
      }

      // might resize pad pixmap so should be called before any paint routine
      RecalculateRange();

      gPad->Modified(kTRUE);
      gVirtualX->SetLineColor(-1);

      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (1) {
         px = py = 0;
         event = gVirtualX->RequestLocator(1, 1, px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }
   }
}

//______________________________________________________________________________
TList *THistPainter::GetContourList(Double_t contour) const
{
   /* Begin_html
   Get a contour (as a list of TGraphs) using the Delaunay triangulation.
   End_html */

   TGraphDelaunay *dt;

   // Check if fH contains a TGraphDelaunay
   TList *hl = fH->GetListOfFunctions();
   dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
   if (!dt) return 0;

   gCurrentHist = fH;

   if (!fGraph2DPainter) ((THistPainter*)this)->fGraph2DPainter = new TGraph2DPainter(dt);

   return fGraph2DPainter->GetContourList(contour);
}


//______________________________________________________________________________
char *THistPainter::GetObjectInfo(Int_t px, Int_t py) const
{
   /* Begin_html
   Display the histogram info (bin number, contents, integral up to bin
   corresponding to cursor position px,py.
   End_html */

   if (!gPad) return (char*)"";
   static char info[100];
   Double_t x  = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y  = gPad->PadtoY(gPad->AbsPixeltoY(py));
   Double_t x1 = gPad->PadtoX(gPad->AbsPixeltoX(px+1));
   const char *drawOption = fH->GetDrawOption();
   Double_t xmin, xmax, uxmin,uxmax;
   Double_t ymin, ymax, uymin,uymax;
   if (fH->GetDimension() == 2) {
      if (gPad->GetView() || strncmp(drawOption,"cont",4) == 0
                          || strncmp(drawOption,"CONT",4) == 0) {
         uxmin=gPad->GetUxmin();
         uxmax=gPad->GetUxmax();
         xmin = fXaxis->GetBinLowEdge(fXaxis->GetFirst());
         xmax = fXaxis->GetBinUpEdge(fXaxis->GetLast());
         x = xmin +(xmax-xmin)*(x-uxmin)/(uxmax-uxmin);
         uymin=gPad->GetUymin();
         uymax=gPad->GetUymax();
         ymin = fYaxis->GetBinLowEdge(fYaxis->GetFirst());
         ymax = fYaxis->GetBinUpEdge(fYaxis->GetLast());
         y = ymin +(ymax-ymin)*(y-uymin)/(uymax-uymin);
      }
   }
   Int_t binx,biny,binmin,binx1;
   if (gPad->IsVertical()) {
      binx   = fXaxis->FindFixBin(x);
      binmin = fXaxis->GetFirst();
      binx1  = fXaxis->FindFixBin(x1);
      // special case if more than 1 bin in x per pixel
      if (binx1-binx>1 && fH->GetDimension() == 1) {
         Double_t binval=fH->GetBinContent(binx);
         Int_t binnear=binx;
         for (Int_t ibin=binx+1; ibin<binx1; ibin++) {
            Double_t binvaltmp = fH->GetBinContent(ibin);
            if (TMath::Abs(y-binvaltmp) < TMath::Abs(y-binval)) {
               binval=binvaltmp;
               binnear=ibin;
            }
         }
         binx = binnear;
      }
   } else {
      x1 = gPad->PadtoY(gPad->AbsPixeltoY(py+1));
      binx   = fXaxis->FindFixBin(y);
      binmin = fXaxis->GetFirst();
      binx1  = fXaxis->FindFixBin(x1);
      // special case if more than 1 bin in x per pixel
      if (binx1-binx>1 && fH->GetDimension() == 1) {
         Double_t binval=fH->GetBinContent(binx);
         Int_t binnear=binx;
         for (Int_t ibin=binx+1; ibin<binx1; ibin++) {
            Double_t binvaltmp = fH->GetBinContent(ibin);
            if (TMath::Abs(x-binvaltmp) < TMath::Abs(x-binval)) {
               binval=binvaltmp;
               binnear=ibin;
            }
         }
         binx = binnear;
      }
   }
   if (fH->GetDimension() == 1) {
      Double_t integ = 0;
      for (Int_t bin=binmin;bin<=binx;bin++) {integ += fH->GetBinContent(bin);}
      snprintf(info,100,"(x=%g, y=%g, binx=%d, binc=%g, Sum=%g)",x,y,binx,fH->GetBinContent(binx),integ);
   } else {
      if (fH->InheritsFrom(TH2Poly::Class())) {
         TH2Poly *th2 = (TH2Poly*)fH;
         biny = th2->FindBin(x,y);
         snprintf(info,100,"%s (x=%g, y=%g, bin=%d, binc=%g)",th2->GetBinTitle(biny),x,y,biny,th2->GetBinContent(biny));
      } else {
         biny = fYaxis->FindFixBin(y);
         snprintf(info,100,"(x=%g, y=%g, binx=%d, biny=%d, binc=%g)",x,y,binx,biny,fH->GetCellContent(binx,biny));
      }
   }
   return info;
}


//______________________________________________________________________________
Bool_t THistPainter::IsInside(Int_t ix, Int_t iy)
{
   /* Begin_html
   Return kTRUE if the cell ix, iy is inside one of the graphical cuts.
   End_html */

   for (Int_t i=0;i<fNcuts;i++) {
      Double_t x = fXaxis->GetBinCenter(ix);
      Double_t y = fYaxis->GetBinCenter(iy);
      if (fCutsOpt[i] > 0) {
         if (!fCuts[i]->IsInside(x,y)) return kFALSE;
      } else {
         if (fCuts[i]->IsInside(x,y))  return kFALSE;
      }
   }
   return kTRUE;
}


//______________________________________________________________________________
Bool_t THistPainter::IsInside(Double_t x, Double_t y)
{
   /* Begin_html
   Return kTRUE if the point x,y is inside one of the graphical cuts.
   End_html */

   for (Int_t i=0;i<fNcuts;i++) {
      if (fCutsOpt[i] > 0) {
         if (!fCuts[i]->IsInside(x,y)) return kFALSE;
      } else {
         if (fCuts[i]->IsInside(x,y))  return kFALSE;
      }
   }
   return kTRUE;
}


//______________________________________________________________________________
Int_t THistPainter::MakeChopt(Option_t *choptin)
{
   /* Begin_html
   Decode string "choptin" and fill Hoption structure.
   End_html */

   char *l;
   char chopt[128];
   Int_t nch = strlen(choptin);
   strlcpy(chopt,choptin,128);

   Hoption.Axis = Hoption.Bar    = Hoption.Curve   = Hoption.Error = 0;
   Hoption.Hist = Hoption.Line   = Hoption.Mark    = Hoption.Fill  = 0;
   Hoption.Same = Hoption.Func   = Hoption.Scat    = 0;
   Hoption.Star = Hoption.Arrow  = Hoption.Box     = Hoption.Text  = 0;
   Hoption.Char = Hoption.Color  = Hoption.Contour = Hoption.Logx  = 0;
   Hoption.Logy = Hoption.Logz   = Hoption.Lego    = Hoption.Surf  = 0;
   Hoption.Off  = Hoption.Tri    = Hoption.Proj    = Hoption.AxisPos = 0;
   Hoption.Spec = Hoption.Pie    = 0;

   //    special 2D options
   Hoption.List     = 0;
   Hoption.Zscale   = 0;
   Hoption.FrontBox = 1;
   Hoption.BackBox  = 1;
   Hoption.System   = kCARTESIAN;

   Hoption.HighRes  = 0;

   Hoption.Zero     = 0;

   //check for graphical cuts
   MakeCuts(chopt);

   for (Int_t i=0;i<nch;i++) chopt[i] = toupper(chopt[i]);
   if (fH->GetDimension() > 1) Hoption.Scat = 1;
   if (!nch) Hoption.Hist = 1;
   if (fFunctions->First()) Hoption.Func = 2;
   if (fH->GetSumw2N() && fH->GetDimension() == 1) Hoption.Error = 2;

   l = strstr(chopt,"SPEC");
   if (l) {
      Hoption.Scat = 0;
      strncpy(l,"    ",4);
      Int_t bs=0;
      l = strstr(chopt,"BF(");
      if (l) {
         if (sscanf(&l[3],"%d",&bs) > 0) {
            Int_t i=0;
            while (l[i]!=')') {
               l[i] = ' ';
               i++;
            }
            l[i] = ' ';
         }
      }
      Hoption.Spec = TMath::Max(1600,bs);
      return 1;
   }

   l = strstr(chopt,"GL");
   if (l) {
      strncpy(l,"  ",2);
   }
   l = strstr(chopt,"X+");
   if (l) {
      Hoption.AxisPos = 10;
      strncpy(l,"  ",2);
   }
   l = strstr(chopt,"Y+");
   if (l) {
      Hoption.AxisPos += 1;
      strncpy(l,"  ",2);
   }
   if((Hoption.AxisPos == 10 || Hoption.AxisPos == 1) && (nch == 2)) Hoption.Hist = 1;
   if(Hoption.AxisPos == 11 && nch == 4) Hoption.Hist = 1;

   l = strstr(chopt,"SAMES");
   if (l) {
      if (nch == 5) Hoption.Hist = 1;
      Hoption.Same = 2;
      strncpy(l,"     ",5);
   }
   l = strstr(chopt,"SAME");
   if (l) {
      if (nch == 4) Hoption.Hist = 1;
      Hoption.Same = 1;
      strncpy(l,"    ",4);
   }

   l = strstr(chopt,"PIE");
   if (l) {
      Hoption.Pie = 1;
      strncpy(l,"   ",3);
   }

   l = strstr(chopt,"LEGO");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Lego = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Lego = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Lego = 12; l[4] = ' '; }
      l = strstr(chopt,"FB"); if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB"); if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
      l = strstr(chopt,"0");  if (l) { Hoption.Zero = 1;  strncpy(l," ",1); }
   }

   l = strstr(chopt,"SURF");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Surf = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Surf = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Surf = 12; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Surf = 13; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Surf = 14; l[4] = ' '; }
      if (l[4] == '5') { Hoption.Surf = 15; l[4] = ' '; }
      if (l[4] == '6') { Hoption.Surf = 16; l[4] = ' '; }
      if (l[4] == '7') { Hoption.Surf = 17; l[4] = ' '; }
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"TF3");
   if (l) {
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"ISO");
   if (l) {
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"LIST");    if (l) { Hoption.List = 1;  strncpy(l,"    ",4);}

   l = strstr(chopt,"CONT");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Contour = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Contour = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Contour = 12; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Contour = 13; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Contour = 14; l[4] = ' '; }
      if (l[4] == '5') { Hoption.Contour = 15; l[4] = ' '; }
   }
   l = strstr(chopt,"HBAR");
   if (l) {
      Hoption.Hist = 0;
      Hoption.Bar = 20; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Bar = 21; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Bar = 22; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Bar = 23; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Bar = 24; l[4] = ' '; }
   }
   l = strstr(chopt,"BAR");
   if (l) {
      Hoption.Hist = 0;
      Hoption.Bar = 10; strncpy(l,"   ",3);
      if (l[3] == '1') { Hoption.Bar = 11; l[3] = ' '; }
      if (l[3] == '2') { Hoption.Bar = 12; l[3] = ' '; }
      if (l[3] == '3') { Hoption.Bar = 13; l[3] = ' '; }
      if (l[3] == '4') { Hoption.Bar = 14; l[3] = ' '; }
   }

   l = strstr(chopt,"ARR" ); if (l) { Hoption.Arrow  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"BOX" );
   if (l) {
      Hoption.Scat = 0;
      Hoption.Box  = 1; strncpy(l,"   ", 3);
      if (l[3] == '1') { Hoption.Box = 11; l[3] = ' '; }
   }
   l = strstr(chopt,"COLZ"); if (l) { Hoption.Color  = 2; strncpy(l,"    ",4); Hoption.Scat = 0; Hoption.Zscale = 1;}
   l = strstr(chopt,"COL" ); if (l) { Hoption.Color  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"CHAR"); if (l) { Hoption.Char   = 1; strncpy(l,"    ",4); Hoption.Scat = 0; }
   l = strstr(chopt,"FUNC"); if (l) { Hoption.Func   = 2; strncpy(l,"    ",4); Hoption.Hist = 0; }
   l = strstr(chopt,"HIST"); if (l) { Hoption.Hist   = 2; strncpy(l,"    ",4); Hoption.Func = 0; Hoption.Error = 0;}
   l = strstr(chopt,"AXIS"); if (l) { Hoption.Axis   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"AXIG"); if (l) { Hoption.Axis   = 2; strncpy(l,"    ",4); }
   l = strstr(chopt,"SCAT"); if (l) { Hoption.Scat   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"TEXT");
   if (l) {
      Int_t angle;
      if (sscanf(&l[4],"%d",&angle) > 0) {
         if (angle < 0)  angle=0;
         if (angle > 90) angle=90;
         Hoption.Text = 1000+angle;
      } else {
         Hoption.Text = 1;
      }
      strncpy(l,"    ", 4);
      l = strstr(chopt,"N");
      if (l && fH->InheritsFrom(TH2Poly::Class())) Hoption.Text += 3000;
      Hoption.Scat = 0;
   }
   l = strstr(chopt,"POL");  if (l) { Hoption.System = kPOLAR;       strncpy(l,"   ",3); }
   l = strstr(chopt,"CYL");  if (l) { Hoption.System = kCYLINDRICAL; strncpy(l,"   ",3); }
   l = strstr(chopt,"SPH");  if (l) { Hoption.System = kSPHERICAL;   strncpy(l,"   ",3); }
   l = strstr(chopt,"PSR");  if (l) { Hoption.System = kRAPIDITY;    strncpy(l,"   ",3); }

   l = strstr(chopt,"TRI");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Color  = 0;
      Hoption.Tri = 1; strncpy(l,"   ",3);
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
      l = strstr(chopt,"ERR");  if (l) strncpy(l,"   ",3);
   }

   l = strstr(chopt,"AITOFF");
   if (l) {
      Hoption.Proj = 1; strncpy(l,"     ",6);       //Aitoff projection
   }
   l = strstr(chopt,"MERCATOR");
   if (l) {
      Hoption.Proj = 2; strncpy(l,"       ",8);     //Mercator projection
   }
   l = strstr(chopt,"SINUSOIDAL");
   if (l) {
      Hoption.Proj = 3; strncpy(l,"         ",10);  //Sinusoidal projection
   }
   l = strstr(chopt,"PARABOLIC");
   if (l) {
      Hoption.Proj = 4; strncpy(l,"        ",9);    //Parabolic projection
   }
   if (Hoption.Proj > 0) {
      Hoption.Scat = 0;
      Hoption.Contour = 14;
   }

   if (strstr(chopt,"A"))   Hoption.Axis = -1;
   if (strstr(chopt,"B"))   Hoption.Bar  = 1;
   if (strstr(chopt,"C")) { Hoption.Curve =1; Hoption.Hist = -1;}
   if (strstr(chopt,"F"))   Hoption.Fill =1;
   if (strstr(chopt,"][")) {Hoption.Off  =1; Hoption.Hist =1;}
   if (strstr(chopt,"F2"))  Hoption.Fill =2;
   if (strstr(chopt,"L")) { Hoption.Line =1; Hoption.Hist = -1;}
   if (strstr(chopt,"P")) { Hoption.Mark =1; Hoption.Hist = -1;}
   if (strstr(chopt,"Z"))   Hoption.Zscale =1;
   if (strstr(chopt,"*"))   Hoption.Star =1;
   if (strstr(chopt,"H"))   Hoption.Hist =2;
   if (strstr(chopt,"P0"))  Hoption.Mark =10;

   if (fH->InheritsFrom(TH2Poly::Class())) {
      if (Hoption.Fill+Hoption.Line+Hoption.Mark != 0 ) Hoption.Scat = 0;
   }

   if (strstr(chopt,"E")) {
      if (fH->GetDimension() == 1) {
         Hoption.Error = 1;
         if (strstr(chopt,"E0"))  Hoption.Error = 10;
         if (strstr(chopt,"E1"))  Hoption.Error = 11;
         if (strstr(chopt,"E2"))  Hoption.Error = 12;
         if (strstr(chopt,"E3"))  Hoption.Error = 13;
         if (strstr(chopt,"E4"))  Hoption.Error = 14;
         if (strstr(chopt,"E5"))  Hoption.Error = 15;
         if (strstr(chopt,"E6"))  Hoption.Error = 16;
         if (strstr(chopt,"X0")) {
            if (Hoption.Error == 1)  Hoption.Error += 20;
            Hoption.Error += 10;
         }
         if (Hoption.Text && fH->InheritsFrom(TProfile::Class())) {
            Hoption.Text += 2000;
            Hoption.Error = 0;
         }
      } else {
         if (Hoption.Error == 0) {
            Hoption.Error = 100;
            Hoption.Scat  = 0;
         }
         if (Hoption.Text) {
            Hoption.Text += 2000;
            Hoption.Error = 0;
         }
      }
   }

   if (strstr(chopt,"9"))  Hoption.HighRes = 1;

   if (Hoption.Surf == 15) {
      if (Hoption.System == kPOLAR || Hoption.System == kCARTESIAN) {
         Hoption.Surf = 13;
         Warning("MakeChopt","option SURF5 is not supported in Cartesian and Polar modes");
      }
   }

   //      Copy options from current style
   Hoption.Logx = gPad->GetLogx();
   Hoption.Logy = gPad->GetLogy();
   Hoption.Logz = gPad->GetLogz();

   //       Check options incompatibilities
   if (Hoption.Bar  == 1) Hoption.Hist = -1;
   return 1;
}


//______________________________________________________________________________
Int_t THistPainter::MakeCuts(char *choptin)
{
   /* Begin_html
   Decode string "choptin" and fill Graphical cuts structure.
   End_html */

   fNcuts = 0;
   char *left = (char*)strchr(choptin,'[');
   if (!left) return 0;
   char *right = (char*)strchr(choptin,']');
   if (!right) return 0;
   Int_t nch = right-left;
   if (nch < 2) return 0;
   char *cuts = left+1;
   *right = 0;
   char *comma, *minus;
   Int_t i;
   while(1) {
      comma = strchr(cuts,',');
      if (comma) *comma = 0;
      minus = strchr(cuts,'-');
      if (minus) cuts = minus+1;
      while (*cuts == ' ') cuts++;
      Int_t nc = strlen(cuts);
      while (cuts[nc-1] == ' ') {cuts[nc-1] = 0; nc--;}
      TIter next(gROOT->GetListOfSpecials());
      TCutG *cut=0;
      TObject *obj;
      while ((obj = next())) {
         if (!obj->InheritsFrom(TCutG::Class())) continue;
         if (strcmp(obj->GetName(),cuts)) continue;
         cut = (TCutG*)obj;
         break;
      }
      if (cut) {
         fCuts[fNcuts] = cut;
         fCutsOpt[fNcuts] = 1;
         if (minus) fCutsOpt[fNcuts] = -1;
         fNcuts++;
      }
      if (!comma) break;
      cuts = comma+1;
   }
   for (i=0;i<=nch;i++) left[i] = ' ';
   return fNcuts;
}


//______________________________________________________________________________
void THistPainter::Paint(Option_t *option)
{
   /* Begin_Html
   <a href="#HP00">Control routine to paint any kind of histograms.</a>
   End_html */

   if (fH->GetBuffer()) fH->BufferEmpty(-1);

   gPad->SetVertical(kTRUE);

   TH1 *oldhist = gCurrentHist;
   gCurrentHist = fH;
   TH1 *hsave   = fH;
   Double_t minsav = fH->GetMinimumStored();

   if (!MakeChopt(option)) return; //check options and fill Hoption structure

   // Paint using TSpectrum2Painter
   if (Hoption.Spec) {
      if (!TableInit()) return;
      if (!TClass::GetClass("TSpectrum2Painter")) gSystem->Load("libSpectrumPainter");
      gROOT->ProcessLineFast(Form("TSpectrum2Painter::PaintSpectrum((TH2F*)0x%lx,\"%s\",%d)",
                                  (ULong_t)fH, option, Hoption.Spec));
      return;
   }

   if (Hoption.Pie) {
      if (!fPie) fPie = new TPie(fH);
      fPie->Paint(option);
      return;
   } else {
      if (fPie) delete fPie;
      fPie = 0;
   }

   fXbuf  = new Double_t[kNMAX];
   fYbuf  = new Double_t[kNMAX];
   if (fH->GetDimension() > 2) {
      PaintH3(option);
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction(option);
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      delete [] fXbuf; delete [] fYbuf;
      return;
   }
   TView *view = gPad->GetView();
   if (view) {
      if (!Hoption.Lego && !Hoption.Surf && !Hoption.Tri) {
         delete view;
         gPad->SetView(0);
      }
   }
   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf) {
      // In case of 1D histogram, Z axis becomes Y axis.
      Int_t logysav=0, logzsav=0;
      if (fH->GetDimension() == 1) {
         logysav = Hoption.Logy;
         logzsav = Hoption.Logz;
         Hoption.Logz = 0;
         if (Hoption.Logy) {
            Hoption.Logz = 1;
            Hoption.Logy = 0;
         }
      }
      PaintTable(option);
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction(option);
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      delete [] fXbuf; delete [] fYbuf;
      if (fH->GetDimension() == 1) {
         Hoption.Logy = logysav;
         Hoption.Logz = logzsav;
      }
      return;
   }

   if (Hoption.Bar >= 20) {PaintBarH(option);
      delete [] fXbuf; delete [] fYbuf;
      return;
   }

   // fill Hparam structure with histo parameters
   if (!PaintInit()) {
      delete [] fXbuf; delete [] fYbuf;
      return;
   }

   //          Picture surround (if new page) and page number (if requested).
   //          Histogram surround (if not option "Same").
   PaintFrame();

   //          Paint histogram axis only
   Bool_t gridx = gPad->GetGridx();
   Bool_t gridy = gPad->GetGridy();
   if (Hoption.Axis > 0) {
      if (Hoption.Axis > 1) PaintAxis(kTRUE);  //axis with grid
      else {
         if (gridx) gPad->SetGridx(0);
         if (gridy) gPad->SetGridy(0);
         PaintAxis(kFALSE);
         if (gridx) gPad->SetGridx(1);
         if (gridy) gPad->SetGridy(1);
      }
      if (Hoption.Same ==1) Hoption.Same = 2;
      goto paintstat;
   }
   if (gridx || gridy) PaintAxis(kTRUE); //    Draw the grid only

   //          test for options BAR or HBAR
   if (Hoption.Bar >= 10) {
      PaintBar(option);
   }

   //          do not draw histogram if error bars required
   if (!Hoption.Error) {
      if (Hoption.Hist && Hoption.Bar<10) PaintHist(option);
   }

   //         test for error bars or option E
   if (Hoption.Error) {
      PaintErrors(option);
      if (Hoption.Hist == 2) PaintHist(option);
   }

   if (Hoption.Text) PaintText(option);

   //         test for associated function
   if (Hoption.Func) {
      Hoption_t hoptsave = Hoption;
      Hparam_t  hparsave = Hparam;
      PaintFunction(option);
      SetHistogram(hsave);
      Hoption = hoptsave;
      Hparam  = hparsave;
   }

   if (gridx) gPad->SetGridx(0);
   if (gridy) gPad->SetGridy(0);
   PaintAxis(kFALSE);
   if (gridx) gPad->SetGridx(1);
   if (gridy) gPad->SetGridy(1);

   PaintTitle();  // Draw histogram title

   // Draw box with histogram statistics and/or fit parameters
paintstat:
   if (Hoption.Same != 1 && !fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
      TIter next(fFunctions);
      TObject *obj = 0;
      while ((obj = next())) {
         if (obj->InheritsFrom(TF1::Class())) break;
         obj = 0;
      }
      PaintStat(gStyle->GetOptStat(),(TF1*)obj);
   }
   fH->SetMinimum(minsav);
   gCurrentHist = oldhist;
   delete [] fXbuf; fXbuf = 0;
   delete [] fYbuf; fYbuf = 0;

}


//______________________________________________________________________________
void THistPainter::PaintArrows(Option_t *)
{
   /* Begin_html
   <a href="#HP12">Control function to draw a table as an arrow plot.</a>
   End_html */

   Style_t linesav   = fH->GetLineStyle();
   Width_t widthsav  = fH->GetLineWidth();
   fH->SetLineStyle(1);
   fH->SetLineWidth(1);
   fH->TAttLine::Modify();

   Double_t xk, xstep, yk, ystep;
   Double_t dx, dy, si, co, anr, x1, x2, y1, y2, xc, yc, dxn, dyn;
   Int_t   ncx  = Hparam.xlast - Hparam.xfirst + 1;
   Int_t   ncy  = Hparam.ylast - Hparam.yfirst + 1;
   Double_t xrg = gPad->GetUxmin();
   Double_t yrg = gPad->GetUymin();
   Double_t xln = gPad->GetUxmax() - xrg;
   Double_t yln = gPad->GetUymax() - yrg;
   Double_t cx  = (xln/Double_t(ncx) -0.03)/2;
   Double_t cy  = (yln/Double_t(ncy) -0.03)/2;
   Double_t dn  = 1.E-30;

   for (Int_t id=1;id<=2;id++) {
      for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
         yk    = fYaxis->GetBinLowEdge(j);
         ystep = fYaxis->GetBinWidth(j);
         for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
            xk    = fXaxis->GetBinLowEdge(i);
            xstep = fXaxis->GetBinWidth(i);
            if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
            if (i == Hparam.xfirst) {
               dx = fH->GetCellContent(i+1, j) - fH->GetCellContent(i, j);
            } else if (i == Hparam.xlast) {
               dx = fH->GetCellContent(i, j) - fH->GetCellContent(i-1, j);
            } else {
               dx = 0.5*(fH->GetCellContent(i+1, j) - fH->GetCellContent(i-1, j));
            }
            if (j == Hparam.yfirst) {
               dy = fH->GetCellContent(i, j+1) - fH->GetCellContent(i, j);
            } else if (j == Hparam.ylast) {
               dy = fH->GetCellContent(i, j) - fH->GetCellContent(i, j-1);
            } else {
               dy = 0.5*(fH->GetCellContent(i, j+1) - fH->GetCellContent(i, j-1));
            }
            if (id == 1) {
               dn = TMath::Max(dn, TMath::Abs(dx));
               dn = TMath::Max(dn, TMath::Abs(dy));
            } else if (id == 2) {
               xc  = xrg + xln*(Double_t(i - Hparam.xfirst+1)-0.5)/Double_t(ncx);
               dxn = cx*dx/dn;
               x1  = xc - dxn;
               x2  = xc + dxn;
               yc  = yrg + yln*(Double_t(j - Hparam.yfirst+1)-0.5)/Double_t(ncy);
               dyn = cy*dy/dn;
               y1  = yc - dyn;
               y2  = yc + dyn;
               fXbuf[0] = x1;
               fXbuf[1] = x2;
               fYbuf[0] = y1;
               fYbuf[1] = y2;
               if (TMath::Abs(x2-x1) > 0.01 || TMath::Abs(y2-y1) > 0.01) {
                  anr = 0.005*.5*TMath::Sqrt(2/(dxn*dxn + dyn*dyn));
                  si  = anr*(dxn + dyn);
                  co  = anr*(dxn - dyn);
                  fXbuf[2] = x2 - si;
                  fYbuf[2] = y2 + co;
                  gPad->PaintPolyLine(3, fXbuf, fYbuf);
                  fXbuf[0] = x2;
                  fXbuf[1] = x2 - co;
                  fYbuf[0] = y2;
                  fYbuf[1] = y2 - si;
                  gPad->PaintPolyLine(2, fXbuf, fYbuf);
               }
               else {
                  gPad->PaintPolyLine(2, fXbuf, fYbuf);
               }
            }
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();
   fH->SetLineStyle(linesav);
   fH->SetLineWidth(widthsav);
   fH->TAttLine::Modify();
}


//______________________________________________________________________________
void THistPainter::PaintAxis(Bool_t drawGridOnly)
{
   /* Begin_html
   Draw axis (2D case) of an histogram.
   <p>
   If drawGridOnly is TRUE, only the grid is painted (if needed). This allows
   to draw the grid and the axis separately. In THistPainter::Paint this
   feature is used to make sure that the grid is drawn in the background and
   the axis tick marks in the foreground of the pad.
   End_html */

   if (Hoption.Axis == -1) return;
   if (Hoption.Same && Hoption.Axis <= 0) return;

   // Repainting alphanumeric labels axis on a plot done with
   // the option HBAR (horizontal) needs some adjustements.
   TAxis *xaxis = 0;
   TAxis *yaxis = 0;
   if (Hoption.Same && Hoption.Axis) { // Axis repainted (TPad::RedrawAxis)
      if (fXaxis->GetLabels() || fYaxis->GetLabels()) { // One axis has alphanumeric labels
         TIter next(gPad->GetListOfPrimitives());
         TObject *obj;
         // Check if the first TH1 of THStack in the pad is drawn with the option HBAR
         while ((obj = next())) {
            if (!obj->InheritsFrom(TH1::Class()) &&
                !obj->InheritsFrom(THStack::Class())) continue;
            TString opt = obj->GetDrawOption();
            opt.ToLower();
            // if drawn with HBAR, the axis should be inverted and the pad set to horizontal
            if (strstr(opt,"hbar")) {
               gPad->SetVertical(kFALSE);
               xaxis = fXaxis;
               yaxis = fYaxis;
               if (!strcmp(xaxis->GetName(),"xaxis")) {
                  fXaxis = yaxis;
                  fYaxis = xaxis;
               }
            }
            break;
         }
      }
   }

   static char chopt[10] = "";
   Double_t gridl = 0;
   Int_t ndiv, ndivx, ndivy, nx1, nx2, ndivsave;
   Int_t useHparam = 0;
   Double_t umin, umax, uminsave, umaxsave;
   Short_t xAxisPos = Hoption.AxisPos/10;
   Short_t yAxisPos = Hoption.AxisPos - 10*xAxisPos;

   Double_t axmin = gPad->GetUxmin();
   Double_t axmax = gPad->GetUxmax();
   Double_t aymin = gPad->GetUymin();
   Double_t aymax = gPad->GetUymax();
   char *cw = 0;
   TGaxis axis;

   // In case of option 'cont4' or in case of option 'same' over a 'cont4 plot'
   // Hparam must be use for the axis limits.
   if (Hoption.Contour == 14) useHparam = 1;
   if (Hoption.Same) {
      TObject *obj;
      TIter next(gPad->GetListOfPrimitives());
      while ((obj=next())) {
         if (strstr(obj->GetDrawOption(),"cont4")) {
            useHparam = 1;
            break;
         }
      }
   }

   // Paint X axis
   ndivx = fXaxis->GetNdivisions();
   if (ndivx > 1000) {
      nx2   = ndivx/100;
      nx1   = TMath::Max(1, ndivx%100);
      ndivx = 100*nx2 + Int_t(Float_t(nx1)*gPad->GetAbsWNDC());
   }
   axis.SetTextAngle(0);
   axis.ImportAxisAttributes(fXaxis);

   chopt[0] = 0;
   // coverity [Calling risky function]
   strlcat(chopt, "SDH",10);
   // coverity [Calling risky function]
   if (ndivx < 0) strlcat(chopt, "N",10);
   if (gPad->GetGridx()) {
      gridl = (aymax-aymin)/(gPad->GetY2() - gPad->GetY1());
      // coverity [Calling risky function]
      strlcat(chopt, "W",10);
   }

   // Define X-Axis limits
   if (Hoption.Logx) {
      // coverity [Calling risky function]
      strlcat(chopt, "G",10);
      ndiv = TMath::Abs(ndivx);
      if (useHparam) {
         umin = TMath::Power(10,Hparam.xmin);
         umax = TMath::Power(10,Hparam.xmax);
      } else {
         umin = TMath::Power(10,axmin);
         umax = TMath::Power(10,axmax);
      }
   } else {
      ndiv = TMath::Abs(ndivx);
      if (useHparam) {
         umin = Hparam.xmin;
         umax = Hparam.xmax;
      } else {
         umin = axmin;
         umax = axmax;
      }
   }

   // Display axis as time
   if (fXaxis->GetTimeDisplay()) {
      // coverity [Calling risky function]
      strlcat(chopt,"t",10);
      if (strlen(fXaxis->GetTimeFormatOnly()) == 0) {
         axis.SetTimeFormat(fXaxis->ChooseTimeFormat(Hparam.xmax-Hparam.xmin));
      }
   }

   // The main X axis can be on the bottom or on the top of the pad
   Double_t xAxisYPos1, xAxisYPos2;
   if (xAxisPos == 1) {
      // Main X axis top
      xAxisYPos1 = aymax;
      xAxisYPos2 = aymin;
   } else {
      // Main X axis bottom
      xAxisYPos1 = aymin;
      xAxisYPos2 = aymax;
   }

   // Paint the main X axis (always)
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   if (xAxisPos) {
      // coverity [Calling risky function]
      strlcat(chopt, "-",10);
      gridl = -gridl;
   }
   if (Hoption.Same && Hoption.Axis) { // Axis repainted (TPad::RedrawAxis)
      axis.SetLabelSize(0.);
      axis.SetTitle("");
   }
   axis.PaintAxis(axmin, xAxisYPos1,
                  axmax, xAxisYPos1,
                  umin, umax,  ndiv, chopt, gridl, drawGridOnly);

   // Paint additional X axis (if needed)
   if (gPad->GetTickx()) {
      if (xAxisPos) {
         cw=strstr(chopt,"-");
         *cw='z';
      } else {
         // coverity [Calling risky function]
         strlcat(chopt, "-",10);
      }
      // coverity [Calling risky function]
      if (gPad->GetTickx() < 2) strlcat(chopt, "U",10);
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(axmin, xAxisYPos2,
                     axmax, xAxisYPos2,
                     uminsave, umaxsave,  ndivsave, chopt, gridl, drawGridOnly);
   }

   // Paint Y axis
   ndivy = fYaxis->GetNdivisions();
   axis.ImportAxisAttributes(fYaxis);

   chopt[0] = 0;
   // coverity [Calling risky function]
   strlcat(chopt, "SDH",10);
   // coverity [Calling risky function]
   if (ndivy < 0) strlcat(chopt, "N",10);
   if (gPad->GetGridy()) {
      gridl = (axmax-axmin)/(gPad->GetX2() - gPad->GetX1());
      // coverity [Calling risky function]
      strlcat(chopt, "W",10);
   }

   // Define Y-Axis limits
   if (Hoption.Logy) {
      // coverity [Calling risky function]
      strlcat(chopt, "G",10);
      ndiv = TMath::Abs(ndivy);
      if (useHparam) {
         umin = TMath::Power(10,Hparam.ymin);
         umax = TMath::Power(10,Hparam.ymax);
      } else {
         umin = TMath::Power(10,aymin);
         umax = TMath::Power(10,aymax);
      }
   } else {
      ndiv = TMath::Abs(ndivy);
      if (useHparam) {
         umin = Hparam.ymin;
         umax = Hparam.ymax;
      } else {
         umin = aymin;
         umax = aymax;
      }
   }

   // Display axis as time
   if (fYaxis->GetTimeDisplay()) {
      // coverity [Calling risky function]
      strlcat(chopt,"t",10);
      if (strlen(fYaxis->GetTimeFormatOnly()) == 0) {
         axis.SetTimeFormat(fYaxis->ChooseTimeFormat(Hparam.ymax-Hparam.ymin));
      }
   }

   // The main Y axis can be on the left or on the right of the pad
   Double_t yAxisXPos1, yAxisXPos2;
   if (yAxisPos == 1) {
      // Main Y axis left
      yAxisXPos1 = axmax;
      yAxisXPos2 = axmin;
   } else {
      // Main Y axis right
      yAxisXPos1 = axmin;
      yAxisXPos2 = axmax;
   }

   // Paint the main Y axis (always)
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   if (yAxisPos) {
      // coverity [Calling risky function]
      strlcat(chopt, "+L",10);
      gridl = -gridl;
   }
   if (Hoption.Same && Hoption.Axis) { // Axis repainted (TPad::RedrawAxis)
      axis.SetLabelSize(0.);
      axis.SetTitle("");
   }
   axis.PaintAxis(yAxisXPos1, aymin,
                  yAxisXPos1, aymax,
                  umin, umax,  ndiv, chopt, gridl, drawGridOnly);

   // Paint the additional Y axis (if needed)
   if (gPad->GetTicky()) {
      if (gPad->GetTicky() < 2) {
         // coverity [Calling risky function]
         strlcat(chopt, "U",10);
         axis.SetTickSize(-fYaxis->GetTickLength());
      } else {
         // coverity [Calling risky function]
         strlcat(chopt, "+L",10);
      }
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(yAxisXPos2, aymin,
                     yAxisXPos2, aymax,
                     uminsave, umaxsave,  ndivsave, chopt, gridl, drawGridOnly);
   }

   // Reset the axis if they have been inverted in case of option HBAR
   if (xaxis) {
      fXaxis = xaxis;
      fYaxis = yaxis;
   }
}


//______________________________________________________________________________
void THistPainter::PaintBar(Option_t *)
{
   /* Begin_html
   <a href="#HP10">Draw a bar-chart in a normal pad.</a>
   End_html */

   Int_t bar = Hoption.Bar - 10;
   Double_t xmin,xmax,ymin,ymax,umin,umax,w,y;
   Double_t offset = fH->GetBarOffset();
   Double_t width  = fH->GetBarWidth();
   TBox box;
   Int_t hcolor = fH->GetFillColor();
   Int_t hstyle = fH->GetFillStyle();
   box.SetFillColor(hcolor);
   box.SetFillStyle(hstyle);
   for (Int_t bin=fXaxis->GetFirst();bin<=fXaxis->GetLast();bin++) {
      y    = fH->GetBinContent(bin);
      xmin = gPad->XtoPad(fXaxis->GetBinLowEdge(bin));
      xmax = gPad->XtoPad(fXaxis->GetBinUpEdge(bin));
      ymin = gPad->GetUymin();
      ymax = gPad->YtoPad(y);
      if (ymax < gPad->GetUymin()) continue;
      if (ymax > gPad->GetUymax()) ymax = gPad->GetUymax();
      if (ymin < gPad->GetUymin()) ymin = gPad->GetUymin();
      if (gStyle->GetHistMinimumZero() && ymin < 0)
         ymin=TMath::Min(0.,gPad->GetUymax());
      w    = (xmax-xmin)*width;
      xmin += offset*(xmax-xmin);
      xmax = xmin + w;
      if (bar < 1) {
         box.PaintBox(xmin,ymin,xmax,ymax);
      } else {
         umin = xmin + bar*(xmax-xmin)/10.;
         umax = xmax - bar*(xmax-xmin)/10.;
         //box.SetFillColor(hcolor+150); //bright
         box.SetFillColor(TColor::GetColorBright(hcolor)); //bright
         box.PaintBox(xmin,ymin,umin,ymax);
         box.SetFillColor(hcolor);
         box.PaintBox(umin,ymin,umax,ymax);
         box.SetFillColor(TColor::GetColorDark(hcolor)); //dark
         box.PaintBox(umax,ymin,xmax,ymax);
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintBarH(Option_t *)
{
   /* Begin_html
   <a href="#HP10">Draw a bar char in a rotated pad (X vertical, Y horizontal).</a>
   End_html */

   gPad->SetVertical(kFALSE);

   PaintInitH();

   TAxis *xaxis = fXaxis;
   TAxis *yaxis = fYaxis;
   if (!strcmp(xaxis->GetName(),"xaxis")) {
      fXaxis = yaxis;
      fYaxis = xaxis;
   }

   PaintFrame();

   Int_t bar = Hoption.Bar - 20;
   Double_t xmin,xmax,ymin,ymax,umin,umax,w;
   Double_t offset = fH->GetBarOffset();
   Double_t width  = fH->GetBarWidth();
   TBox box;
   Int_t hcolor = fH->GetFillColor();
   Int_t hstyle = fH->GetFillStyle();
   box.SetFillColor(hcolor);
   box.SetFillStyle(hstyle);
   for (Int_t bin=fYaxis->GetFirst();bin<=fYaxis->GetLast();bin++) {
      ymin = gPad->YtoPad(fYaxis->GetBinLowEdge(bin));
      ymax = gPad->YtoPad(fYaxis->GetBinUpEdge(bin));
      xmin = gPad->GetUxmin();
      xmax = gPad->XtoPad(fH->GetBinContent(bin));
      if (xmax < gPad->GetUxmin()) continue;
      if (xmax > gPad->GetUxmax()) xmax = gPad->GetUxmax();
      if (xmin < gPad->GetUxmin()) xmin = gPad->GetUxmin();
      if (gStyle->GetHistMinimumZero() && xmin < 0)
         xmin=TMath::Min(0.,gPad->GetUxmax());
      w    = (ymax-ymin)*width;
      ymin += offset*(ymax-ymin);
      ymax = ymin + w;
      if (bar < 1) {
         box.PaintBox(xmin,ymin,xmax,ymax);
      } else {
         umin = ymin + bar*(ymax-ymin)/10.;
         umax = ymax - bar*(ymax-ymin)/10.;
         box.SetFillColor(TColor::GetColorDark(hcolor)); //dark
         box.PaintBox(xmin,ymin,xmax,umin);
         box.SetFillColor(hcolor);
         box.PaintBox(xmin,umin,xmax,umax);
         box.SetFillColor(TColor::GetColorBright(hcolor)); //bright
         box.PaintBox(xmin,umax,xmax,ymax);
      }
   }

   PaintTitle();
   //    Draw box with histogram statistics and/or fit parameters
   if (Hoption.Same != 1 && !fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
      TIter next(fFunctions);
      TObject *obj = 0;
      while ((obj = next())) {
         if (obj->InheritsFrom(TF1::Class())) break;
         obj = 0;
      }
      PaintStat(gStyle->GetOptStat(),(TF1*)obj);
   }

   PaintAxis(kFALSE);
   fXaxis = xaxis;
   fYaxis = yaxis;
}


//______________________________________________________________________________
void THistPainter::PaintBoxes(Option_t *)
{
   /* Begin_html
   <a href="#HP13">Control function to draw a 2D histogram as a box plot.</a>
   End_html */

   Style_t fillsav   = fH->GetFillStyle();
   Style_t colsav    = fH->GetFillColor();
   if (fH->GetFillColor() == 0)  fH->SetFillStyle(0);
   if (Hoption.Box == 11) fH->SetFillStyle(1001);
   fH->TAttLine::Modify();
   fH->TAttFill::Modify();

   Double_t z, xk,xstep, yk, ystep, xcent, ycent, xlow, xup, ylow, yup;
   Double_t ux1 = gPad->PixeltoX(1);
   Double_t ux0 = gPad->PixeltoX(0);
   Double_t uy1 = gPad->PixeltoY(1);
   Double_t uy0 = gPad->PixeltoY(0);
   Double_t dxmin = 0.51*(gPad->PadtoX(ux1)-gPad->PadtoX(ux0));
   Double_t dymin = 0.51*(gPad->PadtoY(uy0)-gPad->PadtoY(uy1));

   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();

   if (Hoption.Logz) {
      if (zmin > 0) {
         zmin = TMath::Log10(zmin*0.1);
         zmax = TMath::Log10(zmax);
      } else {
         return;
      }
   } else {
      zmax = TMath::Max(TMath::Abs(zmin),TMath::Abs(zmax));
      zmin = 0;
   }

   // In case of option SAME, zmin and zmax values are taken from the
   // first plotted 2D histogram.
   if (Hoption.Same) {
      TH2 *h2;
      TIter next(gPad->GetListOfPrimitives());
      while ((h2 = (TH2 *)next())) {
         if (!h2->InheritsFrom(TH2::Class())) continue;
         zmin = h2->GetMinimum();
         zmax = h2->GetMaximum();
         if (Hoption.Logz) {
            zmax = TMath::Log10(zmax);
            if (zmin <= 0) {
               zmin = TMath::Log10(zmax*0.001);
            } else {
               zmin = TMath::Log10(zmin);
            }
         }
         break;
      }
   }

   Double_t zratio, dz = zmax - zmin;
   Bool_t kZNeg        = kFALSE;

   // Define the dark and light colors the "button style" boxes.
   Color_t color = fH->GetFillColor();
   Color_t light=0, dark=0;
   if (Hoption.Box == 11) {
      light = TColor::GetColorBright(color);
      dark  = TColor::GetColorDark(color);
   }

   // Loop over all the bins and draw the boxes
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      ycent = 0.5*ystep;
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         xcent = 0.5*xstep;
         z     = Hparam.factor*fH->GetBinContent(bin);
         kZNeg = kFALSE;
         if (z < 0) {
            if (Hoption.Logz) continue;
            z = -z;
            kZNeg = kTRUE;
         }
         if (Hoption.Logz) {
            if (z != 0) z = TMath::Log10(z);
            else        z = zmin;
         }

         if (z <  zmin) continue; // Can be the case with
         if (z >  zmax) z = zmax; // option Same

         if (dz == 0) continue;
         zratio = TMath::Sqrt((z-zmin)/dz);
         if (zratio == 0) continue;

         xup  = xcent*zratio + xk + xcent;
         xlow = 2*(xk + xcent) - xup;
         if (xup-xlow < dxmin) xup = xlow+dxmin;
         if (Hoption.Logx) {
            if (xup > 0)  xup  = TMath::Log10(xup);
            else continue;
            if (xlow > 0) xlow = TMath::Log10(xlow);
            else continue;
         }

         yup  = ycent*zratio + yk + ycent;
         ylow = 2*(yk + ycent) - yup;
         if (yup-ylow < dymin) yup = ylow+dymin;
         if (Hoption.Logy) {
            if (yup > 0)  yup  = TMath::Log10(yup);
            else continue;
            if (ylow > 0) ylow = TMath::Log10(ylow);
            else continue;
         }

         xlow = TMath::Max(xlow, gPad->GetUxmin());
         ylow = TMath::Max(ylow, gPad->GetUymin());
         xup  = TMath::Min(xup , gPad->GetUxmax());
         yup  = TMath::Min(yup , gPad->GetUymax());

         if (xlow >= xup) continue;
         if (ylow >= yup) continue;

         if (Hoption.Box == 1) {
            fH->SetFillColor(color);
            fH->TAttFill::Modify();
            gPad->PaintBox(xlow, ylow, xup, yup);
            if (kZNeg) {
               gPad->PaintLine(xlow, ylow, xup, yup);
               gPad->PaintLine(xlow, yup, xup, ylow);
            }
         } else if (Hoption.Box == 11) {
            // Draw the center of the box
            fH->SetFillColor(color);
            fH->TAttFill::Modify();
            gPad->PaintBox(xlow, ylow, xup, yup);

            // Draw top&left part of the box
            Double_t x[7], y[7];
            Double_t bwidth = 0.1;
            x[0] = xlow;                     y[0] = ylow;
            x[1] = xlow + bwidth*(xup-xlow); y[1] = ylow + bwidth*(yup-ylow);
            x[2] = x[1];                     y[2] = yup - bwidth*(yup-ylow);
            x[3] = xup - bwidth*(xup-xlow);  y[3] = y[2];
            x[4] = xup;                      y[4] = yup;
            x[5] = xlow;                     y[5] = yup;
            x[6] = xlow;                     y[6] = ylow;
            if (kZNeg) fH->SetFillColor(dark);
            else       fH->SetFillColor(light);
            fH->TAttFill::Modify();
            gPad->PaintFillArea(7, x, y);

            // Draw bottom&right part of the box
            x[0] = xlow;                     y[0] = ylow;
            x[1] = xlow + bwidth*(xup-xlow); y[1] = ylow + bwidth*(yup-ylow);
            x[2] = xup - bwidth*(xup-xlow);  y[2] = y[1];
            x[3] = x[2];                     y[3] = yup - bwidth*(yup-ylow);
            x[4] = xup;                      y[4] = yup;
            x[5] = xup;                      y[5] = ylow;
            x[6] = xlow;                     y[6] = ylow;
            if (kZNeg) fH->SetFillColor(light);
            else       fH->SetFillColor(dark);
            fH->TAttFill::Modify();
            gPad->PaintFillArea(7, x, y);
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();
   fH->SetFillStyle(fillsav);
   fH->SetFillColor(colsav);
   fH->TAttFill::Modify();
}


//______________________________________________________________________________
void THistPainter::PaintColorLevels(Option_t *)
{
   /* Begin_html
   <a href="#HP14">Control function to draw a 2D histogram as a color plot.</a>
   End_html */

   Double_t z, zc, xk, xstep, yk, ystep, xlow, xup, ylow, yup;

   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();

   Double_t dz = zmax - zmin;
   if (dz <= 0) { // Histogram filled with a constant value
      zmax += 0.1*TMath::Abs(zmax);
      zmin -= 0.1*TMath::Abs(zmin);
      dz = zmax - zmin;
   }

   if (Hoption.Logz) {
      if (zmin > 0) {
         zmin = TMath::Log10(zmin);
         zmax = TMath::Log10(zmax);
         dz = zmax - zmin;
      } else {
         return;
      }
   }

   Style_t fillsav   = fH->GetFillStyle();
   Style_t colsav    = fH->GetFillColor();
   fH->SetFillStyle(1001);
   fH->TAttFill::Modify();

   // Initialize the levels on the Z axis
   Int_t ncolors  = gStyle->GetNumberOfColors();
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);
   Double_t scale = ndivz/dz;

   Int_t color;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (Hoption.System == kPOLAR && xk<0) xk= 2*TMath::Pi()+xk;
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         z     = fH->GetBinContent(bin);
         if (z == 0 && (zmin >= 0 || Hoption.Logz)) continue; // don't draw the empty bins for histograms with positive content
         if (Hoption.Logz) {
            if (z > 0) z = TMath::Log10(z);
            else       z = zmin;
         }
         if (z < zmin) continue;
         xup  = xk + xstep;
         xlow = xk;
         if (Hoption.Logx) {
            if (xup > 0)  xup  = TMath::Log10(xup);
            else continue;
            if (xlow > 0) xlow = TMath::Log10(xlow);
            else continue;
         }
         yup  = yk + ystep;
         ylow = yk;
         if (Hoption.System != kPOLAR) {
            if (Hoption.Logy) {
               if (yup > 0)  yup  = TMath::Log10(yup);
               else continue;
               if (ylow > 0) ylow = TMath::Log10(ylow);
               else continue;
            }
            if (xup  < gPad->GetUxmin()) continue;
            if (yup  < gPad->GetUymin()) continue;
            if (xlow > gPad->GetUxmax()) continue;
            if (ylow > gPad->GetUymax()) continue;
            if (xlow < gPad->GetUxmin()) xlow = gPad->GetUxmin();
            if (ylow < gPad->GetUymin()) ylow = gPad->GetUymin();
            if (xup  > gPad->GetUxmax()) xup  = gPad->GetUxmax();
            if (yup  > gPad->GetUymax()) yup  = gPad->GetUymax();
         }

         if (fH->TestBit(TH1::kUserContour)) {
            zc = fH->GetContourLevelPad(0);
            if (z < zc) continue;
            color = -1;
            for (Int_t k=0; k<ndiv; k++) {
               zc = fH->GetContourLevelPad(k);
               if (z < zc) {
                  continue;
               } else {
                  color++;
               }
            }
         } else {
            color = Int_t(0.01+(z-zmin)*scale);
         }

         Int_t theColor = Int_t((color+0.99)*Float_t(ncolors)/Float_t(ndivz));
         if (theColor > ncolors-1) theColor = ncolors-1;
         fH->SetFillColor(gStyle->GetColorPalette(theColor));
         fH->TAttFill::Modify();
         if (Hoption.System != kPOLAR) {
            gPad->PaintBox(xlow, ylow, xup, yup);
         } else  {
            TCrown crown(0,0,ylow,yup,xlow*TMath::RadToDeg(),xup*TMath::RadToDeg());
            crown.SetFillColor(gStyle->GetColorPalette(theColor));
            crown.Paint();
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();

   fH->SetFillStyle(fillsav);
   fH->SetFillColor(colsav);
   fH->TAttFill::Modify();

}


//______________________________________________________________________________
void THistPainter::PaintContour(Option_t *option)
{
   /* Begin_html
   <a href="#HP16">Control function to draw a 2D histogram as a contour plot.</a>
   End_html */

   Int_t i, j, count, ncontour, icol, n, lj, m, ix, jx, ljfill;
   Int_t itars, mode, ir[4];
   Double_t xsave, ysave, thesave,phisave,x[4], y[4], zc[4];

   if (Hoption.Contour == 14) {
      Hoption.Surf = 12;
      Hoption.Axis = 1;
      thesave = gPad->GetTheta();
      phisave = gPad->GetPhi();
      gPad->SetPhi(0.);
      gPad->SetTheta(90.);
      PaintSurface(option);
      gPad->SetPhi(phisave);
      gPad->SetTheta(thesave);
      gPad->GetView()->SetBit(kCannotRotate); //tested in ExecuteEvent
      PaintAxis();
      return;
   }

   if (Hoption.Same) {
      // If the contour is painted on a 3d plot, the contour lines are
      // paint in 3d too.
      TObject *obj;
      TIter next(gPad->GetListOfPrimitives());
      while ((obj=next())) {
         if (strstr(obj->GetDrawOption(),"surf") ||
             strstr(obj->GetDrawOption(),"lego") ||
             strstr(obj->GetDrawOption(),"tri")) {
               Hoption.Surf = 16;
               PaintSurface(option);
               return;
         }
      }
   }

   if (Hoption.Contour == 15) {
      TGraphDelaunay *dt;
      TList *hl = fH->GetListOfFunctions();
      dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
      if (!dt) return;
      if (!fGraph2DPainter) fGraph2DPainter = new TGraph2DPainter(dt);
      fGraph2DPainter->Paint(option);
      return;
   }

   gPad->SetBit(TGraph::kClipFrame);

   Double_t *levels  = new Double_t[2*kMAXCONTOUR];
   Double_t *xarr    = new Double_t[2*kMAXCONTOUR];
   Double_t *yarr    = new Double_t[2*kMAXCONTOUR];
   Int_t  *itarr     = new Int_t[2*kMAXCONTOUR];

   Int_t npmax = 0;
   for (i=0;i<2*kMAXCONTOUR;i++) itarr[i] = 0;

   ncontour = fH->GetContour();
   if (ncontour == 0) {
      ncontour = gStyle->GetNumberContours();
      fH->SetContour(ncontour);
   }
   if (ncontour > kMAXCONTOUR) {
      Warning("PaintContour", "maximum number of contours is %d, asked for %d",
              kMAXCONTOUR, ncontour);
      ncontour = kMAXCONTOUR-1;
   }
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ncontour);

   for (i=0;i<ncontour;i++) levels[i] = fH->GetContourLevelPad(i);
   //for (i=0;i<ncontour;i++)
   //   levels[i] = Hparam.zmin+(Hparam.zmax-Hparam.zmin)/ncontour*i;
   Int_t linesav   = fH->GetLineStyle();
   Int_t colorsav  = fH->GetLineColor();
   Int_t fillsav  = fH->GetFillColor();
   if (Hoption.Contour == 13) {
      fH->TAttLine::Modify();
   }

   TPolyLine **polys = 0;
   TPolyLine *poly=0;
   TObjArray *contours = 0;
   TList *list = 0;
   TGraph *graph = 0;
   Int_t *np = 0;
   if (Hoption.Contour == 1) {
      np = new Int_t[ncontour];
      for (i=0;i<ncontour;i++) np[i] = 0;
      polys = new TPolyLine*[ncontour];
      for (i=0;i<ncontour;i++) {
         polys[i] = new TPolyLine(100);
      }
      if (Hoption.List == 1) {
         contours = (TObjArray*)gROOT->GetListOfSpecials()->FindObject("contours");
         if (contours) {
            gROOT->GetListOfSpecials()->Remove(contours);
            count = contours->GetSize();
            for (i=0;i<count;i++) {
               list = (TList*)contours->At(i);
               if (list) list->Delete();
            }
         }
         contours = new TObjArray(ncontour);
         contours->SetName("contours");
         gROOT->GetListOfSpecials()->Add(contours);
         for (i=0;i<ncontour;i++) {
            list = new TList();
            contours->Add(list);
         }
      }
   }
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz   = TMath::Abs(ncontour);

   Int_t k,ipoly;
   for (j=Hparam.yfirst; j<Hparam.ylast; j++) {
      y[0] = fYaxis->GetBinCenter(j);
      y[1] = y[0];
      y[2] = fYaxis->GetBinCenter(j+1);
      y[3] = y[2];
      for (i=Hparam.xfirst; i<Hparam.xlast; i++) {
         zc[0] = fH->GetBinContent(i,   j);
         zc[1] = fH->GetBinContent(i+1, j);
         zc[2] = fH->GetBinContent(i+1, j+1);
         zc[3] = fH->GetBinContent(i,   j+1);
         if (!IsInside(fXaxis->GetBinCenter(i),fYaxis->GetBinCenter(j))) continue;
         if (Hoption.Logz) {
            if (zc[0] > 0)   zc[0] = TMath::Log10(zc[0]);
            else             zc[0] = Hparam.zmin;
            if (zc[1] > 0)   zc[1] = TMath::Log10(zc[1]);
            else             zc[1] = Hparam.zmin;
            if (zc[2] > 0)   zc[2] = TMath::Log10(zc[2]);
            else             zc[2] = Hparam.zmin;
            if (zc[3] > 0)   zc[3] = TMath::Log10(zc[3]);
            else             zc[3] = Hparam.zmin;
         }
         for (k=0;k<4;k++) {
            ir[k] = TMath::BinarySearch(ncontour,levels,zc[k]);
         }
         if (ir[0] != ir[1] || ir[1] != ir[2] || ir[2] != ir[3] || ir[3] != ir[0]) {
            x[0] = fXaxis->GetBinCenter(i);
            x[3] = x[0];
            x[1] = fXaxis->GetBinCenter(i+1);
            x[2] = x[1];
            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=1;
            for (ix=1;ix<=4;ix++) {
               m = n%4 + 1;
               ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],zc[m-1],
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1], levels);
               lj += 2*ljfill;
               n = m;
            }

            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=2;
            for (ix=1;ix<=4;ix++) {
               if (n == 1) m = 4;
               else        m = n-1;
               ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],zc[m-1],
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1], levels);
               lj += 2*ljfill;
               n = m;
            }

   //     Re-order endpoints

            count = 0;
            for (ix=1; ix<=lj-5; ix +=2) {
               //count = 0;
               while (itarr[ix-1] != itarr[ix]) {
                  xsave = xarr[ix];
                  ysave = yarr[ix];
                  itars = itarr[ix];
                  for (jx=ix; jx<=lj-5; jx +=2) {
                     xarr[jx]  = xarr[jx+2];
                     yarr[jx]  = yarr[jx+2];
                     itarr[jx] = itarr[jx+2];
                  }
                  xarr[lj-3]  = xsave;
                  yarr[lj-3]  = ysave;
                  itarr[lj-3] = itars;
                  if (count > 100) break;
                  count++;
               }
            }

            if (count > 100) continue;
            for (ix=1; ix<=lj-2; ix +=2) {
               theColor = Int_t((itarr[ix-1]+0.99)*Float_t(ncolors)/Float_t(ndivz));
               icol = gStyle->GetColorPalette(theColor);
               if (Hoption.Contour == 11) {
                  fH->SetLineColor(icol);
               }
               if (Hoption.Contour == 12) {
                  mode = icol%5;
                  if (mode == 0) mode = 5;
                  fH->SetLineStyle(mode);
               }
               if (Hoption.Contour != 1) {
                  fH->TAttLine::Modify();
                  gPad->PaintPolyLine(2,&xarr[ix-1],&yarr[ix-1]);
                  continue;
               }

               ipoly = itarr[ix-1];
               if (ipoly >=0 && ipoly <ncontour) {
                  poly = polys[ipoly];
                  poly->SetPoint(np[ipoly]  ,xarr[ix-1],yarr[ix-1]);
                  poly->SetPoint(np[ipoly]+1,xarr[ix],  yarr[ix]);
                  np[ipoly] += 2;
                  if (npmax < np[ipoly]) npmax = np[ipoly];
               }
            }
         } // end of if (ir[0]
      } //end of for (i
   } //end of for (j

   Double_t xmin,ymin;
   Double_t *xp, *yp;
   Int_t nadd,iminus,iplus;
   Double_t *xx, *yy;
   Int_t istart;
   Int_t first = ncontour;
   Int_t *polysort = 0;
   Int_t contListNb;
   if (Hoption.Contour != 1) goto theEND;

   //The 2 points line generated above are now sorted/merged to generate
   //a list of consecutive points.
   // If the option "List" has been specified, the list of points is saved
   // in the form of TGraph objects in the ROOT list of special objects.
   xmin = gPad->GetUxmin();
   ymin = gPad->GetUymin();
   xp = new Double_t[2*npmax];
   yp = new Double_t[2*npmax];
   polysort = new Int_t[ncontour];
   //find first positive contour
   for (ipoly=0;ipoly<ncontour;ipoly++) {
      if (levels[ipoly] >= 0) {first = ipoly; break;}
   }
   //store negative contours from 0 to minimum, then all positive contours
   k = 0;
   for (ipoly=first-1;ipoly>=0;ipoly--) {polysort[k] = ipoly; k++;}
   for (ipoly=first;ipoly<ncontour;ipoly++) {polysort[k] = ipoly; k++;}
   // we can now draw sorted contours
   contListNb = 0;
   fH->SetFillStyle(1001);
   for (k=0;k<ncontour;k++) {
      ipoly = polysort[k];
      if (np[ipoly] == 0) continue;
      if (Hoption.List) list = (TList*)contours->At(contListNb);
      contListNb++;
      poly = polys[ipoly];
      xx = poly->GetX();
      yy = poly->GetY();
      istart = 0;
      while (1) {
         iminus = npmax;
         iplus  = iminus+1;
         xp[iminus]= xx[istart];   yp[iminus] = yy[istart];
         xp[iplus] = xx[istart+1]; yp[iplus]  = yy[istart+1];
         xx[istart]   = xmin; yy[istart]   = ymin;
         xx[istart+1] = xmin; yy[istart+1] = ymin;
         while (1) {
            nadd = 0;
            for (i=2;i<np[ipoly];i+=2) {
               if (xx[i] == xp[iplus] && yy[i] == yp[iplus]) {
                  iplus++;
                  xp[iplus] = xx[i+1]; yp[iplus]  = yy[i+1];
                  xx[i]   = xmin; yy[i]   = ymin;
                  xx[i+1] = xmin; yy[i+1] = ymin;
                  nadd++;
               }
               if (xx[i+1] == xp[iminus] && yy[i+1] == yp[iminus]) {
                  iminus--;
                  xp[iminus] = xx[i];   yp[iminus]  = yy[i];
                  xx[i]   = xmin; yy[i]   = ymin;
                  xx[i+1] = xmin; yy[i+1] = ymin;
                  nadd++;
               }
            }
            if (nadd == 0) break;
         }
         theColor = Int_t((ipoly+0.99)*Float_t(ncolors)/Float_t(ndivz));
         icol = gStyle->GetColorPalette(theColor);
         if (ndivz > 1) fH->SetFillColor(icol);
         fH->TAttFill::Modify();
         gPad->PaintFillArea(iplus-iminus+1,&xp[iminus],&yp[iminus]);
         if (Hoption.List) {
            graph = new TGraph(iplus-iminus+1,&xp[iminus],&yp[iminus]);
            graph->SetFillColor(icol);
            graph->SetLineWidth(fH->GetLineWidth());
            list->Add(graph);
         }
         //check if more points are left
         istart = 0;
         for (i=2;i<np[ipoly];i+=2) {
            if (xx[i] != xmin && yy[i] != ymin) {
               istart = i;
               break;
            }
         }
         if (istart == 0) break;
      }
   }

   for (i=0;i<ncontour;i++) delete polys[i];
   delete [] polys;
   delete [] xp;
   delete [] yp;
   delete [] polysort;

theEND:
   gPad->ResetBit(TGraph::kClipFrame);
   if (Hoption.Zscale) PaintPalette();
   fH->SetLineStyle(linesav);
   fH->SetLineColor(colorsav);
   fH->SetFillColor(fillsav);
   if (np) delete [] np;
   delete [] xarr;
   delete [] yarr;
   delete [] itarr;
   delete [] levels;
}


//______________________________________________________________________________
Int_t THistPainter::PaintContourLine(Double_t elev1, Int_t icont1, Double_t x1, Double_t y1,
                            Double_t elev2, Int_t icont2, Double_t x2, Double_t y2,
                            Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels)
{
   /* Begin_html
   Fill the matrix XARR YARR for Contour Plot.
   End_html */

   Bool_t vert;
   Double_t tlen, tdif, elev, diff, pdif, xlen;
   Int_t n, i, icount;

   if (x1 == x2) {
      vert = kTRUE;
      tlen = y2 - y1;
   } else {
      vert = kFALSE;
      tlen = x2 - x1;
   }

   n = icont1 +1;
   tdif = elev2 - elev1;
   i = 0;
   icount = 0;
   while (n <= icont2 && i <= kMAXCONTOUR/2 -3) {
      //elev = fH->GetContourLevel(n);
      elev = levels[n];
      diff = elev - elev1;
      pdif = diff/tdif;
      xlen = tlen*pdif;
      if (vert) {
         if (Hoption.Logx)
            xarr[i] = TMath::Log10(x1);
         else
            xarr[i] = x1;
         if (Hoption.Logy)
            yarr[i] = TMath::Log10(y1 + xlen);
         else
            yarr[i] = y1 + xlen;
      } else {
         if (Hoption.Logx)
            xarr[i] = TMath::Log10(x1 + xlen);
         else
            xarr[i] = x1 + xlen;
         if (Hoption.Logy)
            yarr[i] = TMath::Log10(y1);
         else
            yarr[i] = y1;
      }
      itarr[i] = n;
      icount++;
      i +=2;
      n++;
   }
   return icount;
}


//______________________________________________________________________________
void THistPainter::PaintErrors(Option_t *)
{
   /* Begin_html
   <a href="#HP09">Draw 1D histograms error bars.</a>
   End_html */

   const Int_t kBASEMARKER=8;
   Double_t xp, yp, ex1, ex2, ey1, ey2;
   Double_t delta;
   Double_t s2x, s2y, bxsize, bysize, symbolsize, xerror, sbase;
   Double_t xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4;
   Double_t xmin, xmax, ymin, ymax;
   Double_t logxmin = 0;
   Double_t logymin = 0;
   Int_t i, k, npoints, first, last, fixbin;
   Int_t if1 = 0;
   Int_t if2 = 0;
   Int_t drawmarker, errormarker;
   Int_t option0, option1, option2, option3, option4, optionE, optionEX0, optionI0;

   Double_t *xline = 0;
   Double_t *yline = 0;
   option0 = option1 = option2 = option3 = option4 = optionE = optionEX0 = optionI0 = 0;
   if (Int_t(Hoption.Error/10) == 2) {optionEX0 = 1; Hoption.Error -= 10;}
   if (Hoption.Error == 31) {optionEX0 = 1; Hoption.Error = 1;}
   if (Hoption.Error == 10) option0 = 1;
   if (Hoption.Error == 11) option1 = 1;
   if (Hoption.Error == 12) option2 = 1;
   if (Hoption.Error == 13) option3 = 1;
   if (Hoption.Error == 14) {option4 = 1; option3 = 1;}
   if (Hoption.Error == 15) {optionI0 = 1; option3 = 1;}
   if (Hoption.Error == 16) {optionI0 = 1; option4 = 1; option3 = 1;}
   if (option2+option3 == 0) optionE = 1;
   if (Hoption.Error == 0) optionE = 0;
   if (fXaxis->GetXbins()->fN) fixbin = 0;
   else                        fixbin = 1;

   errormarker = fH->GetMarkerStyle();
   if (optionEX0) {
      xerror = 0;
   } else {
      xerror = gStyle->GetErrorX();
   }
   symbolsize  = fH->GetMarkerSize();
   if (errormarker == 1) symbolsize = 0.01;
   sbase       = symbolsize*kBASEMARKER;
   // set the graphics attributes

   fH->TAttLine::Modify();
   fH->TAttFill::Modify();
   fH->TAttMarker::Modify();

   // set the first and last bin

   Double_t factor = Hparam.factor;
   first      = Hparam.xfirst;
   last       = Hparam.xlast;
   npoints    = last - first  +1;
   xmin       = gPad->GetUxmin();
   xmax       = gPad->GetUxmax();
   ymin       = gPad->GetUymin();
   ymax       = gPad->GetUymax();


   if (option3) {
      xline = new Double_t[2*npoints];
      yline = new Double_t[2*npoints];
      if (!xline || !yline) {
         Error("PaintErrors", "too many points, out of memory");
         return;
      }
      if1 = 1;
      if2 = 2*npoints;
   }

   //  compute the offset of the error bars due to the symbol size
   s2x    = gPad->PixeltoX(Int_t(0.5*sbase)) - gPad->PixeltoX(0);
   s2y    =-gPad->PixeltoY(Int_t(0.5*sbase)) + gPad->PixeltoY(0);

   // compute size of the lines at the end of the error bars
   Int_t dxend = Int_t(gStyle->GetEndErrorSize());
   bxsize    = gPad->PixeltoX(dxend) - gPad->PixeltoX(0);
   bysize    =-gPad->PixeltoY(dxend) + gPad->PixeltoY(0);


   if (fixbin) {
      if (Hoption.Logx) xp = TMath::Power(10,Hparam.xmin) + 0.5*Hparam.xbinsize;
      else              xp = Hparam.xmin + 0.5*Hparam.xbinsize;
   }
   else {
      delta = fH->GetBinWidth(first);
      xp    = fH->GetBinLowEdge(first) + 0.5*delta;
   }

   // if errormarker = 0 or symbolsize = 0. no symbol is drawn
   if (Hoption.Logx) logxmin = TMath::Power(10,Hparam.xmin);
   if (Hoption.Logy) logymin = TMath::Power(10,Hparam.ymin);

   //    ---------------------- Loop over the points---------------------
   for (k=first; k<=last; k++) {

      //          get the data
      //     xp      = X position of the current point
      //     yp      = Y position of the current point
      //     ex1   = Low X error
      //     ex2   = Up X error
      //     ey1   = Low Y error
      //     ey2   = Up Y error
      //     (xi,yi) = Error bars coordinates

      if (Hoption.Logx) {
         if (xp <= 0) goto L30;
         if (xp < logxmin) goto L30;
         if (xp > TMath::Power(10,xmax)) break;
      } else {
         if (xp < xmin) goto L30;
         if (xp > xmax) break;
      }
      yp = factor*fH->GetBinContent(k);
      if (optionI0 && yp==0) goto L30;
      if (fixbin) {
         ex1 = xerror*Hparam.xbinsize;
      } else {
         delta = fH->GetBinWidth(k);
         ex1 = xerror*delta;
      }
      ey1 = factor*fH->GetBinError(k);
      ex2 = ex1;
      ey2 = ey1;

      xi4 = xp;
      xi3 = xp;
      xi2 = xp + ex2;
      xi1 = xp - ex1;

      yi1 = yp;
      yi2 = yp;
      yi3 = yp - ey1;
      yi4 = yp + ey2;

      //          take the LOG if necessary
      if (Hoption.Logx) {
         xi1 = TMath::Log10(TMath::Max(xi1,logxmin));
         xi2 = TMath::Log10(TMath::Max(xi2,logxmin));
         xi3 = TMath::Log10(TMath::Max(xi3,logxmin));
         xi4 = TMath::Log10(TMath::Max(xi4,logxmin));
      }
      if (Hoption.Logy) {
         yi1 = TMath::Log10(TMath::Max(yi1,logymin));
         yi2 = TMath::Log10(TMath::Max(yi2,logymin));
         yi3 = TMath::Log10(TMath::Max(yi3,logymin));
         yi4 = TMath::Log10(TMath::Max(yi4,logymin));
      }

      // test if error bars are not outside the limits
      //  otherwise they are truncated

      xi1 = TMath::Max(xi1,xmin);
      xi2 = TMath::Min(xi2,xmax);
      yi3 = TMath::Max(yi3,ymin);
      yi4 = TMath::Min(yi4,ymax);

      //  test if the marker is on the frame limits. If "Yes", the
      //  marker will not be drawn and the error bars will be readjusted.

      drawmarker = kTRUE;
      if (!option0 && !option3) {
         if (Hoption.Logy && yp < logymin) goto L30;
         if (yi1 < ymin || yi1 > ymax) goto L30;
         if (Hoption.Error != 0 && yp == 0 && ey1 <= 0) drawmarker = kFALSE;
      }
      if (!symbolsize || !errormarker) drawmarker = kFALSE;

      //  draw the error rectangles
      if (option2) gPad->PaintBox(xi1,yi3,xi2,yi4);

      //  keep points for fill area drawing
      if (option3) {
         xline[if1-1] = xi3;
         xline[if2-1] = xi3;
         yline[if1-1] = yi4;
         yline[if2-1] = yi3;
         if1++;
         if2--;
      }

      //          draw the error bars
      if (optionE && drawmarker) {
         if ((yi3 < yi1 - s2y) && (yi3 < ymax)) gPad->PaintLine(xi3,yi3,xi4,TMath::Min(yi1 - s2y,ymax));
         if ((yi1 + s2y < yi4) && (yi4 > ymin)) gPad->PaintLine(xi3,TMath::Max(yi1 + s2y, ymin),xi4,yi4);
         // don't duplicate the horizontal line
         if (Hoption.Hist != 2){
            if (yi1<ymax && yi1>ymin) {
              if (xi1 < xi3 - s2x) gPad->PaintLine(xi1,yi1,xi3 - s2x,yi2);
              if (xi3 + s2x < xi2) gPad->PaintLine(xi3 + s2x,yi1,xi2,yi2);
            }
         }
      }
      if (optionE && !drawmarker && ey1 != 0) {
         if ((yi3 < yi1) && (yi3 < ymax)) gPad->PaintLine(xi3,yi3,xi4,TMath::Min(yi1,ymax));
         if ((yi1 < yi4) && (yi4 > ymin)) gPad->PaintLine(xi3,TMath::Max(yi1,ymin),xi4,yi4);
         // don't duplicate the horizontal line
         if (Hoption.Hist != 2){
            if (yi1<ymax && yi1>ymin) {
               if (xi1 < xi3) gPad->PaintLine(xi1,yi1,xi3,yi2);
               if (xi3 < xi2) gPad->PaintLine(xi3,yi1,xi2,yi2);
            }
         }
      }

      //          draw line at the end of the error bars

      if (option1 && drawmarker) {
         if (yi3 < yi1-s2y) gPad->PaintLine(xi3 - bxsize,yi3,xi3 + bxsize,yi3);
         if (yi4 > yi1+s2y) gPad->PaintLine(xi3 - bxsize,yi4,xi3 + bxsize,yi4);
         if (xi1 < xi3-s2x) gPad->PaintLine(xi1,yi1 - bysize,xi1,yi1 + bysize);
         if (xi2 > xi3+s2x) gPad->PaintLine(xi2,yi1 - bysize,xi2,yi1 + bysize);
      }

      //          draw the marker

      if (drawmarker) gPad->PaintPolyMarker(1, &xi3, &yi1);

L30:
      if (fixbin) xp += Hparam.xbinsize;
      else {
         if (k < last) {
            delta = fH->GetBinWidth(k+1);
            xp    = fH->GetBinLowEdge(k+1) + 0.5*delta;
         }
      }
   }  //end of for loop

   //          draw the filled area

   if (option3) {
      TGraph graph;
      graph.SetLineStyle(fH->GetLineStyle());
      graph.SetLineColor(fH->GetLineColor());
      graph.SetLineWidth(fH->GetLineWidth());
      graph.SetFillStyle(fH->GetFillStyle());
      graph.SetFillColor(fH->GetFillColor());
      Int_t logx = gPad->GetLogx();
      Int_t logy = gPad->GetLogy();
      gPad->SetLogx(0);
      gPad->SetLogy(0);

      // In some cases the number of points in the fill area is smaller than
      // 2*npoints. In such cases the array xline and yline must be arranged
      // before being plotted. The next loop does that.
      if (if2 > npoints) {
         for(i=1; i<if1 ;i++) {
            xline[if1-2+i] = xline[if2-1+i];
            yline[if1-2+i] = yline[if2-1+i];
         }
         npoints = if1-1;
      }
      if (option4) graph.PaintGraph(2*npoints,xline,yline,"FC");
      else         graph.PaintGraph(2*npoints,xline,yline,"F");
      gPad->SetLogx(logx);
      gPad->SetLogy(logy);
      delete [] xline;
      delete [] yline;
   }
}


//______________________________________________________________________________
void THistPainter::Paint2DErrors(Option_t *)
{
   /* Begin_html
   Draw 2D histograms errors.
   End_html */

   fH->TAttMarker::Modify();
   fH->TAttLine::Modify();

   // Define the 3D view
   fXbuf[0] = Hparam.xmin;
   fYbuf[0] = Hparam.xmax;
   fXbuf[1] = Hparam.ymin;
   fYbuf[1] = Hparam.ymax;
   fXbuf[2] = Hparam.zmin;
   fYbuf[2] = Hparam.zmax;
   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);
   TView *view = gPad->GetView();
   if (!view) {
      Error("Paint2DErrors", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   Int_t irep;
   view->SetView(phideg, thedeg, psideg, irep);

   // Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();
   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);
   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   // Paint the Back Box if needed
   if (Hoption.BackBox && !Hoption.Same && !Hoption.Lego && !Hoption.Surf) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   // Paint the Errors
   Double_t x, ex, x1, x2;
   Double_t y, ey, y1, y2;
   Double_t z, ez, z1, z2;
   Double_t temp1[3],temp2[3];
   Double_t xyerror;
   if (Hoption.Error == 110) {
      xyerror = 0 ;
   } else {
      xyerror = gStyle->GetErrorX();
   }

   Double_t xk, xstep, yk, ystep;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      y  = fYaxis->GetBinCenter(j);
      ey = fYaxis->GetBinWidth(j)*xyerror;
      y1 = y-ey;
      y2 = y+ey;
      if (Hoption.Logy) {
         if (y > 0)  y = TMath::Log10(y);
         else        continue;
         if (y1 > 0) y1 = TMath::Log10(y1);
         else        y1 = Hparam.ymin;
         if (y2 > 0) y2 = TMath::Log10(y2);
         else        y2 = Hparam.ymin;
      }
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         Int_t bin = fH->GetBin(i,j);
         x  = fXaxis->GetBinCenter(i);
         ex = fXaxis->GetBinWidth(i)*xyerror;
         x1 = x-ex;
         x2 = x+ex;
         if (Hoption.Logx) {
            if (x > 0)  x = TMath::Log10(x);
            else        continue;
            if (x1 > 0) x1 = TMath::Log10(x1);
            else        x1 = Hparam.xmin;
            if (x2 > 0) x2 = TMath::Log10(x2);
            else        x2 = Hparam.xmin;
         }
         z  = fH->GetBinContent(bin);
         ez = fH->GetBinError(bin);
         z1 = z-ez;
         z2 = z+ez;
         if (Hoption.Logz) {
            if (z > 0)   z = TMath::Log10(z);
            else         z = Hparam.zmin;
            if (z1 > 0) z1 = TMath::Log10(z1);
            else        z1 = Hparam.zmin;
            if (z2 > 0) z2 = TMath::Log10(z2);
            else        z2 = Hparam.zmin;

         }
         if (z <= Hparam.zmin) continue;
         if (z >  Hparam.zmax) z = Hparam.zmax;

         temp1[0] = x1;
         temp1[1] = y;
         temp1[2] = z;
         temp2[0] = x2;
         temp2[1] = y;
         temp2[2] = z;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y1;
         temp1[2] = z;
         temp2[0] = x;
         temp2[1] = y2;
         temp2[2] = z;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y;
         temp1[2] = z1;
         temp2[0] = x;
         temp2[1] = y;
         temp2[2] = z2;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y;
         temp1[2] = z;
         view->WCtoNDC(temp1, &temp2[0]);
         gPad->PaintPolyMarker(1, &temp2[0], &temp2[1]);
      }
   }

   // Paint the Front Box if needed
   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }

   // Paint the Axis if needed
   if (!Hoption.Axis && !Hoption.Same && !Hoption.Lego && !Hoption.Surf) {
      TGaxis *axis = new TGaxis();
      PaintLegoAxis(axis, 90);
      delete axis;
   }

   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintFrame()
{
   /* Begin_html
   Calculate range and clear pad (canvas).
   End_html */

   if (Hoption.Same) return;

   RecalculateRange();

   if (Hoption.Lego || Hoption.Surf || Hoption.Tri ||
       Hoption.Contour == 14 || Hoption.Error >= 100) {
      TObject *frame = gPad->FindObject("TFrame");
      if (frame) gPad->GetListOfPrimitives()->Remove(frame);
      return;
   }
   gPad->PaintPadFrame(Hparam.xmin,Hparam.ymin,Hparam.xmax,Hparam.ymax);
}


//______________________________________________________________________________
void THistPainter::PaintFunction(Option_t *)
{
   /* Begin_html
   <a href="#HP28">Paint functions associated to an histogram.</a>
   End_html */

   TObjOptLink *lnk = (TObjOptLink*)fFunctions->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      TVirtualPad *padsave = gPad;
      if (obj->InheritsFrom(TF2::Class())) {
         if (obj->TestBit(TF2::kNotDraw) == 0) {
            if (Hoption.Lego || Hoption.Surf) {
               obj->Paint("surf same");
            } else {
               obj->Paint("cont3 same");
            }
         }
      } else if (obj->InheritsFrom(TF1::Class())) {
         if (obj->TestBit(TF1::kNotDraw) == 0) obj->Paint("lsame");
      } else  {
         obj->Paint(lnk->GetOption());
      }
      lnk = (TObjOptLink*)lnk->Next();
      padsave->cd();
   }
}


//______________________________________________________________________________
void THistPainter::PaintHist(Option_t *)
{
   /* Begin_html
   <a href="#HP01b">Control routine to draw 1D histograms.</a>
   End_html */

   static char chopth[17];

   Int_t htype, oldhtype;
   Int_t i, j, first, last, nbins, fixbin;
   Double_t c1, yb;
   yb = 0;

   strlcpy(chopth, "                ",17);

   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;
   Double_t baroffset = fH->GetBarOffset();
   Double_t barwidth  = fH->GetBarWidth();
   Double_t baroffsetsave = gStyle->GetBarOffset();
   Double_t barwidthsave  = gStyle->GetBarWidth();
   gStyle->SetBarOffset(baroffset);
   gStyle->SetBarWidth(barwidth);

   //       Create "LIFE" structure to keep current histogram status

   first = Hparam.xfirst;
   last  = Hparam.xlast;
   nbins = last - first + 1;

   Double_t *keepx = 0;
   Double_t *keepy = 0;
   if (fXaxis->GetXbins()->fN) fixbin = 0;
   else                        fixbin = 1;
   if (fixbin) keepx = new Double_t[2];
   else        keepx = new Double_t[nbins+1];
   keepy = new Double_t[nbins];
   Double_t logymin = 0;
   if (Hoption.Logy) logymin = TMath::Power(10,ymin);

   //      Loop on histogram bins

   for (j=first; j<=last;j++) {
      c1 = Hparam.factor*fH->GetBinContent(j);
      if (TMath::Abs(ymax-ymin) > 0) {
         if (Hoption.Logy) yb = TMath::Log10(TMath::Max(c1,.1*logymin));
         else              yb = c1;
      }
      yb = TMath::Max(yb, ymin);
      yb = TMath::Min(yb, ymax);
      keepy[j-first] = yb;
   }

   //              Draw histogram according to value of FillStyle and FillColor

   if (fixbin) { keepx[0] = Hparam.xmin; keepx[1] = Hparam.xmax; }
   else {
      for (i=0; i<nbins; i++) keepx[i] = fXaxis->GetBinLowEdge(i+first);
      keepx[nbins] = fXaxis->GetBinUpEdge(nbins-1+first);
   }

   //         Prepare Fill area (systematic with option "Bar").

   oldhtype = fH->GetFillStyle();
   htype    = oldhtype;
   if (Hoption.Bar) {
      if (htype == 0 || htype == 1000) htype = 1001;
   }

   Width_t lw = (Width_t)fH->GetLineWidth();

   //         Code option for GrapHist

   if (Hoption.Line) chopth[0] = 'L';
   if (Hoption.Star) chopth[1] = '*';
   if (Hoption.Mark) chopth[2] = 'P';
   if (Hoption.Mark == 10) chopth[3] = '0';
   if (Hoption.Line || Hoption.Curve || Hoption.Hist || Hoption.Bar) {
      if (Hoption.Curve)    chopth[3] = 'C';
      if (Hoption.Hist > 0) chopth[4] = 'H';
      else if (Hoption.Bar) chopth[5] = 'B';
      if (fH->GetFillColor() && htype) {
         if (Hoption.Logy) {
            chopth[6] = '1';
         }
         if (Hoption.Hist > 0 || Hoption.Curve || Hoption.Line) {
            chopth[7] = 'F';
         }
      }
   }
   if (!fixbin && strlen(chopth)) {
      chopth[8] = 'N';
   }

   // coverity [Calling risky function]
   if (Hoption.Fill == 2)    strlcat(chopth,"2",17);
   // coverity [Calling risky function]
   if (Hoption.HighRes != 0) strlcat(chopth,"9",17);

   //         Option LOGX

   if (Hoption.Logx) {
      chopth[9]  = 'G';
      chopth[10] = 'X';
      if (fixbin) {
         keepx[0] = TMath::Power(10,keepx[0]);
         keepx[1] = TMath::Power(10,keepx[1]);
      }
   }

   if (Hoption.Off) {
      chopth[11] = ']';
      chopth[12] = '[';
   }

   //         Draw the histogram

   TGraph graph;
   graph.SetLineWidth(lw);
   graph.SetLineStyle(fH->GetLineStyle());
   graph.SetLineColor(fH->GetLineColor());
   graph.SetFillStyle(htype);
   graph.SetFillColor(fH->GetFillColor());
   graph.SetMarkerStyle(fH->GetMarkerStyle());
   graph.SetMarkerSize(fH->GetMarkerSize());
   graph.SetMarkerColor(fH->GetMarkerColor());
   if (!Hoption.Same) graph.ResetBit(TGraph::kClipFrame);

   graph.PaintGrapHist(nbins, keepx, keepy ,chopth);

   delete [] keepx;
   delete [] keepy;
   gStyle->SetBarOffset(baroffsetsave);
   gStyle->SetBarWidth(barwidthsave);

   htype=oldhtype;
}


//______________________________________________________________________________
void THistPainter::PaintH3(Option_t *option)
{
   /* Begin_html
   <a href="#HP01d">Control function to draw a 3D histograms.</a>
   End_html */

   char *cmd;
   TString opt = fH->GetDrawOption();
   opt.ToLower();
   Int_t irep;

   if (fH->GetDrawOption() && (strstr(opt,"box") ||  strstr(opt,"lego"))) {
      cmd = Form("TMarker3DBox::PaintH3((TH1 *)0x%lx,\"%s\");",(Long_t)fH,option);
   } else if (fH->GetDrawOption() && strstr(opt,"iso")) {
      PaintH3Iso();
      return;
   } else if (strstr(option,"tf3")) {
      PaintTF3();
      return;
   } else {
      cmd = Form("TPolyMarker3D::PaintH3((TH1 *)0x%lx,\"%s\");",(Long_t)fH,option);
   }

   TView *view = gPad->GetView();
   if (!view) return;
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   // Paint the data
   gROOT->ProcessLine(cmd);

   if (Hoption.Same) return;

   // Draw axis
   view->SetOutlineToCube();
   view->GetOutline()->Paint(option);
   Hoption.System = kCARTESIAN;
   TGaxis *axis = new TGaxis();
   PaintLegoAxis(axis,90);
   delete axis;

   // Draw palette. In case of 4D plot with TTree::Draw() the palette should
   // be painted with the option colz.
   if (fH->GetDrawOption() && strstr(opt,"colz")) {
      Int_t ndiv   = fH->GetContour();
      if (ndiv == 0 ) {
         ndiv = gStyle->GetNumberContours();
         fH->SetContour(ndiv);
      }
      PaintPalette();
   }

   // Draw title
   PaintTitle();

   //Draw stats and fit results
   TF1 *fit  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TF1::Class())) {
         fit = (TF1*)obj;
         break;
      }
   }
   if (Hoption.Same != 1) {
      if (!fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
         PaintStat3(gStyle->GetOptStat(),fit);
      }
   }

}


//______________________________________________________________________________
Int_t THistPainter::PaintInit()
{
   /* Begin_html
   Compute histogram parameters used by the drawing routines.
   End_html */

   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf) return 1;

   static const char *where = "PaintInit";
   Double_t yMARGIN = gStyle->GetHistTopMargin();
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //     Compute X axis parameters

   Int_t last      = fXaxis->GetLast();
   Int_t first     = fXaxis->GetFirst();
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in X, replace xmin,max by the log
   if (Hoption.Logx) {
      if (Hparam.xlowedge <=0 ) {
         if (Hoption.Same) {
            Hparam.xlowedge = TMath::Power(10, gPad->GetUxmin());
         } else {
            Hparam.xlowedge = 0.1*Hparam.xbinsize;
         }
         Hparam.xmin  = Hparam.xlowedge;
      }
      if (Hparam.xmin <=0 || Hparam.xmax <=0) {
         Error(where, "cannot set X axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.xmin);
      Hparam.xlast = fXaxis->FindFixBin(Hparam.xmax);
      Hparam.xmin  = TMath::Log10(Hparam.xmin);
      Hparam.xmax  = TMath::Log10(Hparam.xmax);
      if (Hparam.xlast  > last)  Hparam.xlast  = last;
      if (Hparam.xfirst < first) Hparam.xfirst = first;
   }

   //     Compute Y axis parameters
   Double_t bigp = TMath::Power(10,32);
   Double_t ymax = -bigp;
   Double_t ymin = bigp;
   Double_t c1, e1;
   Double_t xv[1];
   Double_t fval;
   Int_t i;
   TObject *f;
   TF1 *f1;
   Double_t allchan = 0;
   Int_t nonNullErrors = 0;
   TIter   next(fFunctions);
   for (i=first; i<=last;i++) {
      c1 = fH->GetBinContent(i);
      ymax = TMath::Max(ymax,c1);
      if (Hoption.Logy) {
         if (c1 > 0) ymin = TMath::Min(ymin,c1);
      } else {
         ymin = TMath::Min(ymin,c1);
      }
      if (Hoption.Error) {
         e1 = fH->GetBinError(i);
         if (e1 > 0) nonNullErrors++;
         ymax = TMath::Max(ymax,c1+e1);
         if (Hoption.Logy) {
            if (c1-e1>0.01*TMath::Abs(c1)) ymin = TMath::Min(ymin,c1-e1);
         } else {
            ymin = TMath::Min(ymin,c1-e1);
         }
      }
      if (Hoption.Func) {
         xv[0] = fXaxis->GetBinCenter(i);
         while ((f = (TObject*) next())) {
            if (f->IsA() == TF1::Class()) {
               f1 = (TF1*)f;
               if (xv[0] < f1->GetXmin() || xv[0] > f1->GetXmax()) continue;
               fval = f1->Eval(xv[0],0,0);
               ymax = TMath::Max(ymax,fval);
               if (Hoption.Logy) {
                  if (c1 > 0 && fval > 0.3*c1) ymin = TMath::Min(ymin,fval);
               }
            }
         }
         next.Reset();
      }
      allchan += c1;
   }
   if (!nonNullErrors) {
      if (Hoption.Error) {
         if (!Hoption.Mark && !Hoption.Line && !Hoption.Star && !Hoption.Curve) Hoption.Hist = 2;
         Hoption.Error=0;
      }
   }


   //     Take into account maximum , minimum

   if (Hoption.Logy && ymin <= 0) {
      if (ymax >= 1) ymin = TMath::Max(.005,ymax*1e-10);
      else           ymin = 0.001*ymax;
   }

   Double_t xm = ymin;
   if (maximum) ymax = fH->GetMaximumStored();
   if (minimum) xm   = fH->GetMinimumStored();
   if (Hoption.Logy && xm < 0) {
      Error(where, "log scale requested with a negative argument (%f)", xm);
      return 0;
   } else if (Hoption.Logy && xm>=0 && ymax==0) { // empty histogram in log scale
      ymin = 0.01;
      ymax = 10.;
   } else {
      ymin = xm;
   }

   if (ymin >= ymax) {
      if (Hoption.Logy) {
         if (ymax > 0) ymin = 0.001*ymax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", ymax);
            return 0;
         }
      }
      else {
         if (ymin > 0) {
            ymin = 0;
            ymax *= 2;
         } else if (ymin < 0) {
            ymax = 0;
            ymin *= 2;
         } else {
            ymin = 0;
            ymax = 1;
         }
      }
   }

   // In some cases, mainly because of precision issues, ymin and ymax could almost equal.
   if(TMath::AreEqualRel(ymin,ymax,1E-15)) {
      ymin = ymin*(1-1E-14);
      ymax = ymax*(1+1E-14);
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   ymax = factor*ymax;
   ymin = factor*ymin;
   //just in case the norm factor is negative
   // this may happen with a positive norm factor and a negative integral !
   if (ymax < ymin) {
      Double_t temp = ymax;
      ymax = ymin;
      ymin = temp;
   }

   //         For log scales, histogram coordinates are LOG10(ymin) and
   //         LOG10(ymax). Final adjustment (if not option "Same"
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logy) {
      if (ymin <=0 || ymax <=0) {
         Error(where, "Cannot set Y axis to log scale");
         return 0;
      }
      ymin = TMath::Log10(ymin);
      if (!minimum) ymin += TMath::Log10(0.5);
      ymax = TMath::Log10(ymax);
      if (!maximum) ymax += TMath::Log10(2*(0.9/0.95));
      if (!Hoption.Same) {
         Hparam.ymin = ymin;
         Hparam.ymax = ymax;
      }
      return 1;
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - margin if <0.
   if (!minimum) {
      if (gStyle->GetHistMinimumZero()) {
         if (ymin >= 0) ymin = 0;
         else           ymin -= yMARGIN*(ymax-ymin);
      } else {
         Double_t dymin = yMARGIN*(ymax-ymin);
         if (ymin >= 0 && (ymin-dymin <= 0)) ymin  = 0;
         else                                ymin -= dymin;
      }
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   if (!maximum) {
      ymax += yMARGIN*(ymax-ymin);
   }

   Hparam.ymin = ymin;
   Hparam.ymax = ymax;
   return 1;
}


//______________________________________________________________________________
Int_t THistPainter::PaintInitH()
{
   /* Begin_html
   Compute histogram parameters used by the drawing routines for a rotated pad.
   End_html */

   static const char *where = "PaintInitH";
   Double_t yMARGIN = gStyle->GetHistTopMargin();
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //     Compute X axis parameters

   Int_t last      = fXaxis->GetLast();
   Int_t first     = fXaxis->GetFirst();
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.ymin     = Hparam.xlowedge;
   Hparam.ymax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in Y, replace ymin,max by the log
   if (Hoption.Logy) {
      if (Hparam.xlowedge <=0 ) {
         Hparam.xlowedge = 0.1*Hparam.xbinsize;
         Hparam.ymin  = Hparam.xlowedge;
      }
      if (Hparam.ymin <=0 || Hparam.ymax <=0) {
         Error(where, "cannot set Y axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.ymin);
      Hparam.xlast = fXaxis->FindFixBin(Hparam.ymax);
      Hparam.ymin  = TMath::Log10(Hparam.ymin);
      Hparam.ymax  = TMath::Log10(Hparam.ymax);
      if (Hparam.xlast > last) Hparam.xlast = last;
   }

   //     Compute Y axis parameters
   Double_t bigp = TMath::Power(10,32);
   Double_t xmax = -bigp;
   Double_t xmin = bigp;
   Double_t c1, e1;
   Double_t xv[1];
   Double_t fval;
   Int_t i;
   TObject *f;
   TF1 *f1;
   Double_t allchan = 0;
   TIter   next(fFunctions);
   for (i=first; i<=last;i++) {
      c1 = fH->GetBinContent(i);
      xmax = TMath::Max(xmax,c1);
      xmin = TMath::Min(xmin,c1);
      if (Hoption.Error) {
         e1 = fH->GetBinError(i);
         xmax = TMath::Max(xmax,c1+e1);
         xmin = TMath::Min(xmin,c1-e1);
      }
      if (Hoption.Func) {
         xv[0] = fXaxis->GetBinCenter(i);
         while ((f = (TObject*) next())) {
            if (f->IsA() == TF1::Class()) {
               f1 = (TF1*)f;
               if (xv[0] < f1->GetXmin() || xv[0] > f1->GetXmax()) continue;
               fval = f1->Eval(xv[0],0,0);
               xmax = TMath::Max(xmax,fval);
               if (Hoption.Logy) {
                  if (fval > 0.3*c1) xmin = TMath::Min(xmin,fval);
               }
            }
         }
         next.Reset();
      }
      allchan += c1;
   }

   //     Take into account maximum , minimum

   if (Hoption.Logx && xmin <= 0) {
      if (xmax >= 1) xmin = TMath::Max(.5,xmax*1e-10);
      else           xmin = 0.001*xmax;
   }
   Double_t xm = xmin;
   if (maximum) xmax = fH->GetMaximumStored();
   if (minimum) xm   = fH->GetMinimumStored();
   if (Hoption.Logx && xm <= 0) {
      Error(where, "log scale requested with zero or negative argument (%f)", xm);
      return 0;
   }
   else xmin = xm;
   if (xmin >= xmax) {
      if (Hoption.Logx) {
         if (xmax > 0) xmin = 0.001*xmax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", xmax);
            return 0;
         }
      }
      else {
         if (xmin > 0) {
            xmin = 0;
            xmax *= 2;
         } else if (xmin < 0) {
            xmax = 0;
            xmin *= 2;
         } else {
            xmin = -1;
            xmax = 1;
         }
      }
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   xmax = factor*xmax;
   xmin = factor*xmin;

   //         For log scales, histogram coordinates are LOG10(ymin) and
   //         LOG10(ymax). Final adjustment (if not option "Same"
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logx) {
      if (xmin <=0 || xmax <=0) {
         Error(where, "Cannot set Y axis to log scale");
         return 0;
      }
      xmin = TMath::Log10(xmin);
      if (!minimum) xmin += TMath::Log10(0.5);
      xmax = TMath::Log10(xmax);
      if (!maximum) xmax += TMath::Log10(2*(0.9/0.95));
      if (!Hoption.Same) {
         Hparam.xmin = xmin;
         Hparam.xmax = xmax;
      }
      return 1;
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - margin if <0.
   if (!minimum) {
      if (xmin >= 0) xmin = 0;
      else           xmin -= yMARGIN*(xmax-xmin);
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   if (!maximum) {
      xmax += yMARGIN*(xmax-xmin);
   }
   Hparam.xmin = xmin;
   Hparam.xmax = xmax;
   return 1;
}


//______________________________________________________________________________
void THistPainter::PaintH3Iso()
{
   /* Begin_html
   <a href="#HP25">Control function to draw a 3D histogram with Iso Surfaces.</a>
   End_html */

   const Double_t ydiff = 1;
   const Double_t yligh1 = 10;
   const Double_t qa = 0.15;
   const Double_t qd = 0.15;
   const Double_t qs = 0.8;
   Double_t fmin, fmax;
   Int_t i, irep;
   Int_t nbcol = 28;
   Int_t icol1 = 201;
   Int_t ic1 = icol1;
   Int_t ic2 = ic1+nbcol;
   Int_t ic3 = ic2+nbcol;

   TGaxis *axis = new TGaxis();
   TAxis *xaxis = fH->GetXaxis();
   TAxis *yaxis = fH->GetYaxis();
   TAxis *zaxis = fH->GetZaxis();

   Int_t nx = fH->GetNbinsX();
   Int_t ny = fH->GetNbinsY();
   Int_t nz = fH->GetNbinsZ();

   Double_t *x = new Double_t[nx];
   Double_t *y = new Double_t[ny];
   Double_t *z = new Double_t[nz];

   for ( i=0 ; i<nx ; i++) x[i] = xaxis->GetBinCenter(i+1);
   for ( i=0 ; i<ny ; i++) y[i] = yaxis->GetBinCenter(i+1);
   for ( i=0 ; i<nz ; i++) z[i] = zaxis->GetBinCenter(i+1);

   fXbuf[0] = xaxis->GetBinLowEdge(xaxis->GetFirst());
   fYbuf[0] = xaxis->GetBinUpEdge(xaxis->GetLast());
   fXbuf[1] = yaxis->GetBinLowEdge(yaxis->GetFirst());
   fYbuf[1] = yaxis->GetBinUpEdge(yaxis->GetLast());
   fXbuf[2] = zaxis->GetBinLowEdge(zaxis->GetFirst());
   fYbuf[2] = zaxis->GetBinUpEdge(zaxis->GetLast());

   Double_t s[3];
   s[0] = fH->GetSumOfWeights()/(fH->GetNbinsX()*fH->GetNbinsY()*fH->GetNbinsZ());
   s[1] = 0.5*s[0];
   s[2] = 1.5*s[0];

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintH3Iso", "no TView in current pad");
      delete [] x;
      delete [] y;
      delete [] z;
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   Double_t dcol = 0.5/Double_t(nbcol);
   TColor *colref = gROOT->GetColor(fH->GetFillColor());
   Float_t r, g, b, hue, light, satur;
   colref->GetRGB(r,g,b);
   TColor::RGBtoHLS(r,g,b,hue,light,satur);
   TColor *acol;
   for (Int_t col=0;col<nbcol;col++) {
      acol = gROOT->GetColor(col+icol1);
      TColor::HLStoRGB(hue, .4+col*dcol, satur, r, g, b);
      acol->SetRGB(r, g, b);
   }

   fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.BackBox) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   fLego->LightSource(0, ydiff, 0, 0, 0, irep);
   fLego->LightSource(1, yligh1, 1, 1, 1, irep);
   fLego->SurfaceProperty(qa, qd, qs, 1, irep);
   fmin = ydiff*qa;
   fmax = ydiff*qa + (yligh1+0.1)*(qd+qs);
   fLego->SetIsoSurfaceParameters(fmin, fmax, nbcol, ic1, ic2, ic3);

   fLego->IsoSurface(1, s, nx, ny, nz, x, y, z, "BF");

   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   PaintTitle();

   delete axis;
   delete fLego; fLego = 0;
   delete [] x;
   delete [] y;
   delete [] z;
}


//______________________________________________________________________________
void THistPainter::PaintLego(Option_t *)
{
   /* Begin_html
   <a href="#HP17">Control function to draw a 2D histogram as a lego plot.</a>
   End_html */

   Int_t raster = 1;
   if (Hparam.zmin == 0 && Hparam.zmax == 0) {Hparam.zmin = -1; Hparam.zmax = 1;}
   Int_t   nx      = Hparam.xlast - Hparam.xfirst + 1;
   Int_t   ny      = Hparam.ylast - Hparam.yfirst + 1;
   Double_t zmin   = Hparam.zmin;
   Double_t zmax   = Hparam.zmax;
   Double_t xlab1  = Hparam.xmin;
   Double_t xlab2  = Hparam.xmax;
   Double_t ylab1  = Hparam.ymin;
   Double_t ylab2  = Hparam.ymax;
   Double_t dangle = 10*3.141592/180; //Delta angle for Rapidity option
   Double_t deltaz = TMath::Abs(zmin);
   if (deltaz == 0) deltaz = 1;
   if (zmin >= zmax) {
      zmin -= 0.5*deltaz;
      zmax += 0.5*deltaz;
   }
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*(1+gStyle->GetHistTopMargin());

   //     Compute the lego limits and instantiate a lego object
   fXbuf[0] = -1;
   fYbuf[0] =  1;
   fXbuf[1] = -1;
   fYbuf[1] =  1;
   if (Hoption.System == kPOLAR) {
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Logy) {
         if (ylab1 > 0) fXbuf[2] = TMath::Log10(ylab1);
         else           fXbuf[2] = 0;
         if (ylab2 > 0) fYbuf[2] = TMath::Log10(ylab2);
         else           fYbuf[2] = 0;
      } else {
         fXbuf[2] = ylab1;
         fYbuf[2] = ylab2;
      }
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kSPHERICAL) {
      fXbuf[2] = -1;
      fYbuf[2] =  1;
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kRAPIDITY) {
      fXbuf[2] = -1/TMath::Tan(dangle);
      fYbuf[2] =  1/TMath::Tan(dangle);
   } else {
      fXbuf[0] = xlab1;
      fYbuf[0] = xlab2;
      fXbuf[1] = ylab1;
      fYbuf[1] = ylab2;
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
      raster  = 0;
   }

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf, Hoption.System);

   //          Create axis object

   TGaxis *axis = new TGaxis();

   //                  Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

   //     Initialize colors for the lighting model (option Lego1 only)
   if (Hoption.Lego == 1) {
         Color_t colormain = fH->GetLineColor();
         fLego->SetColorMain(colormain,0);
   }
   if (Hoption.Lego == 11) {
      Int_t nids = 1;
      if (fStack) nids = fStack->GetSize();
      TH1 *hid = fH;
      for (Int_t id=0;id<=nids;id++) {
         if (id > 0 && fStack) hid = (TH1*)fStack->At(id-1);
         Color_t colormain = hid->GetFillColor();
         if (colormain == 1) colormain = 17; //avoid drawing with black
         Color_t colordark = TColor::GetColorDark(colormain);
         fLego->SetColorMain(colormain,id);
         fLego->SetColorDark(colordark,id);
         if (id == 0)    fLego->SetColorMain(colormain,-1);  // Set Bottom color
         if (id == nids) fLego->SetColorMain(colormain,99);  // Set Top color
      }
   }

   //     Now ready to draw the lego plot
   Int_t irep = 0;

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintLego", "no TView in current pad");
      return;
   }

   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   fLego->SetLineColor(fH->GetLineColor());
   fLego->SetFillStyle(fH->GetFillStyle());

   //     Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();

   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Lego == 11 || Hoption.Lego == 12) {
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

   if (Hoption.Lego == 12) DefineColorLevels(ndivz);

   fLego->SetLegoFunction(&TPainter3dAlgorithms::LegoFunction);
   if (Hoption.Lego ==  1) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceRaster2);
   if (Hoption.Lego == 11) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode3);
   if (Hoption.Lego == 12) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
   if (Hoption.System == kPOLAR) {
      if (Hoption.Lego ==  1) fLego->LegoPolar(1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoPolar(1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoPolar(1,nx,ny,"BF");
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Lego ==  1) fLego->LegoCylindrical(1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoCylindrical(1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoCylindrical(1,nx,ny,"BF");
   } else if (Hoption.System == kSPHERICAL) {
      if (Hoption.Lego ==  1) fLego->LegoSpherical(0,1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoSpherical(0,1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoSpherical(0,1,nx,ny,"BF");
   } else if (Hoption.System == kRAPIDITY) {
      if (Hoption.Lego ==  1) fLego->LegoSpherical(1,1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoSpherical(1,1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoSpherical(1,1,nx,ny,"BF");
   } else {
      if (Hoption.Lego ==  1) {
                              fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
                              fLego->LegoCartesian(90,nx,ny,"FB");}
      if (Hoption.Lego == 11) fLego->LegoCartesian(90,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoCartesian(90,nx,ny,"BF");
   }

   if (Hoption.Lego == 1 || Hoption.Lego == 11) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);
   if (Hoption.Zscale) PaintPalette();
   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintLegoAxis(TGaxis *axis, Double_t ang)
{
   /* Begin_html
   Draw the axis for legos and surface plots.
   End_html */

   static Double_t epsil = 0.001;

   Double_t cosa, sina;
   Double_t bmin, bmax;
   Double_t r[24]        /* was [3][8] */;
   Int_t ndivx, ndivy, ndivz, i;
   Double_t x1[3], x2[3], y1[3], y2[3], z1[3], z2[3], av[24]  /*  was [3][8] */;
   static char chopax[8], chopay[8], chopaz[8];
   Int_t ix1, ix2, iy1, iy2, iz1, iz2;
   Double_t rad;

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintLegoAxis", "no TView in current pad");
      return;
   }

   // In polar coordinates, draw a short line going from the external circle
   // corresponding to r = 1 up to r = 1.1
   if (Hoption.System == kPOLAR) {
      r[0] = 1;
      r[1] = 0;
      r[2] = 0;
      view->WCtoNDC(r, x1);
      r[0] = 1.1;
      r[1] = 0;
      r[2] = 0;
      view->WCtoNDC(r, x2);
      gPad->PaintLine(x1[0],x1[1],x2[0],x2[1]);
      return ;
   }

   if (Hoption.System != kCARTESIAN) return ;

   rad = TMath::ATan(1.) * 4. /180.;
   cosa = TMath::Cos(ang*rad);
   sina = TMath::Sin(ang*rad);

   view->AxisVertex(ang, av, ix1, ix2, iy1, iy2, iz1, iz2);
   for (i = 1; i <= 8; ++i) {
      r[i*3 - 3] = av[i*3 - 3] + av[i*3 - 2]*cosa;
      r[i*3 - 2] = av[i*3 - 2]*sina;
      r[i*3 - 1] = av[i*3 - 1];
   }

   view->WCtoNDC(&r[ix1*3 - 3], x1);
   view->WCtoNDC(&r[ix2*3 - 3], x2);
   view->WCtoNDC(&r[iy1*3 - 3], y1);
   view->WCtoNDC(&r[iy2*3 - 3], y2);
   view->WCtoNDC(&r[iz1*3 - 3], z1);
   view->WCtoNDC(&r[iz2*3 - 3], z2);

   view->SetAxisNDC(x1, x2, y1, y2, z1, z2);

   Double_t *rmin = view->GetRmin();
   Double_t *rmax = view->GetRmax();

   // Initialize the axis options
   if (x1[0] > x2[0]) strlcpy(chopax, "SDH=+",8);
   else               strlcpy(chopax, "SDH=-",8);
   if (y1[0] > y2[0]) strlcpy(chopay, "SDH=+",8);
   else               strlcpy(chopay, "SDH=-",8);
   strlcpy(chopaz, "SDH+=",8);

   // Option LOG is required ?
   // coverity [Calling risky function]
   if (Hoption.Logx) strlcat(chopax,"G",8);
   // coverity [Calling risky function]
   if (Hoption.Logy) strlcat(chopay,"G",8);
   // coverity [Calling risky function]
   if (Hoption.Logz) strlcat(chopaz,"G",8);

   // Initialize the number of divisions. If the
   // number of divisions is negative, option 'N' is required.
   ndivx = fXaxis->GetNdivisions();
   ndivy = fYaxis->GetNdivisions();
   ndivz = fZaxis->GetNdivisions();
   if (ndivx < 0) {
      ndivx = TMath::Abs(ndivx);
      // coverity [Calling risky function]
      strlcat(chopax, "N",8);
   }
   if (ndivy < 0) {
      ndivy = TMath::Abs(ndivy);
      // coverity [Calling risky function]
      strlcat(chopay, "N",8);
   }
   if (ndivz < 0) {
      ndivz = TMath::Abs(ndivz);
      // coverity [Calling risky function]
      strlcat(chopaz, "N",8);
   }

   // Set Axis attributes.
   // The variable SCALE  rescales the VSIZ
   // in order to have the same label size for all angles.

   axis->SetLineWidth(1);

   // X axis drawing
   if (TMath::Abs(x1[0] - x2[0]) >= epsil || TMath::Abs(x1[1] - x2[1]) > epsil) {
      axis->ImportAxisAttributes(fXaxis);
      axis->SetLabelOffset(fXaxis->GetLabelOffset()+fXaxis->GetTickLength());
      if (Hoption.Logx && !fH->InheritsFrom(TH3::Class())) {
         bmin = TMath::Power(10, rmin[0]);
         bmax = TMath::Power(10, rmax[0]);
      } else {
         bmin = rmin[0];
         bmax = rmax[0];
      }
      // Option time display is required ?
      if (fXaxis->GetTimeDisplay()) {
         // coverity [Calling risky function]
         strlcat(chopax,"t",8);
         if (strlen(fXaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fXaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fXaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopax);
      axis->PaintAxis(x1[0], x1[1], x2[0], x2[1], bmin, bmax, ndivx, chopax);
   }

   // Y axis drawing
   if (TMath::Abs(y1[0] - y2[0]) >= epsil || TMath::Abs(y1[1] - y2[1]) > epsil) {
      axis->ImportAxisAttributes(fYaxis);
      axis->SetLabelOffset(fYaxis->GetLabelOffset()+fYaxis->GetTickLength());

      if (fH->GetDimension() < 2) {
         strlcpy(chopay, "V=+UN",8);
         ndivy = 0;
      }
      if (TMath::Abs(y1[0] - y2[0]) < epsil) {
         y2[0] = y1[0];
      }
      if (Hoption.Logy && !fH->InheritsFrom(TH3::Class())) {
         bmin = TMath::Power(10, rmin[1]);
         bmax = TMath::Power(10, rmax[1]);
      } else {
         bmin = rmin[1];
         bmax = rmax[1];
      }
      // Option time display is required ?
      if (fYaxis->GetTimeDisplay()) {
         // coverity [Calling risky function]
         strlcat(chopay,"t",8);
         if (strlen(fYaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fYaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fYaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopay);
      axis->PaintAxis(y1[0], y1[1], y2[0], y2[1], bmin, bmax, ndivy, chopay);
   }

   // Z axis drawing
   if (TMath::Abs(z1[0] - z2[0]) >= 100*epsil || TMath::Abs(z1[1] - z2[1]) > 100*epsil) {
      axis->ImportAxisAttributes(fZaxis);
      if (Hoption.Logz && !fH->InheritsFrom(TH3::Class())) {
         bmin = TMath::Power(10, rmin[2]);
         bmax = TMath::Power(10, rmax[2]);
      } else {
         bmin = rmin[2];
         bmax = rmax[2];
      }
      // Option time display is required ?
      if (fZaxis->GetTimeDisplay()) {
         // coverity [Calling risky function]
         strlcat(chopaz,"t",8);
         if (strlen(fZaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fZaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fZaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopaz);
      axis->PaintAxis(z1[0], z1[1], z2[0], z2[1], bmin, bmax, ndivz, chopaz);
   }

   fH->SetLineStyle(1);
}


//______________________________________________________________________________
void THistPainter::PaintPalette()
{
   /* Begin_html
   <a href="#HP22">Paint the color palette on the right side of the pad.</a>
   End_html */

   TPaletteAxis *palette = (TPaletteAxis*)fFunctions->FindObject("palette");
   TView *view = gPad->GetView();
   if (palette) {
      if (view) {
         if (!palette->TestBit(TPaletteAxis::kHasView)) {
            delete palette; palette = 0;
         }
      } else {
         if (palette->TestBit(TPaletteAxis::kHasView)) {
            delete palette; palette = 0;
         }
      }
   }

   if (!palette) {
      Double_t xup  = gPad->GetUxmax();
      Double_t x2   = gPad->PadtoX(gPad->GetX2());
      Double_t ymin = gPad->PadtoY(gPad->GetUymin());
      Double_t ymax = gPad->PadtoY(gPad->GetUymax());
      Double_t xr   = 0.05*(gPad->GetX2() - gPad->GetX1());
      Double_t xmin = gPad->PadtoX(xup +0.1*xr);
      Double_t xmax = gPad->PadtoX(xup + xr);
      if (xmax > x2) xmax = gPad->PadtoX(gPad->GetX2()-0.01*xr);
      palette = new TPaletteAxis(xmin,ymin,xmax,ymax,fH);
      fFunctions->Add(palette);
      palette->Paint();
   }
}


//______________________________________________________________________________
void THistPainter::PaintScatterPlot(Option_t *option)
{
   /* Begin_html
   <a href="#HP11">Control function to draw a 2D histogram as a scatter plot.</a>
   End_html */

   fH->TAttMarker::Modify();

   Int_t k, marker;
   Double_t dz, z, xk,xstep, yk, ystep;
   Double_t scale = 1;
   Bool_t ltest  = kFALSE;
   Double_t zmax  = fH->GetMaximum();
   Double_t zmin  = fH->GetMinimum();
   if (zmin == 0 && zmax == 0) return;
   if (zmin == zmax) {
      zmax += 0.1*TMath::Abs(zmax);
      zmin -= 0.1*TMath::Abs(zmin);
   }
   Int_t ncells = (Hparam.ylast-Hparam.yfirst)*(Hparam.xlast-Hparam.xfirst);
   if (Hoption.Logz) {
      if (zmin > 0) zmin = TMath::Log10(zmin);
      else          zmin = 0;
      if (zmax > 0) zmax = TMath::Log10(zmax);
      else          zmax = 0;
      if (zmin == 0 && zmax == 0) return;
      dz = zmax - zmin;
      scale = 100/dz;
      if (ncells > 10000) scale /= 5;
      ltest = kTRUE;
   } else {
      dz = zmax - zmin;
      if (dz >= kNMAX || zmax < 1) {
         scale = (kNMAX-1)/dz;
         if (ncells > 10000) scale /= 5;
         ltest = kTRUE;
      }
   }
   if (fH->GetMinimumStored() == -1111) {
      Double_t yMARGIN = gStyle->GetHistTopMargin();
      if (gStyle->GetHistMinimumZero()) {
         if (zmin >= 0) zmin = 0;
         else           zmin -= yMARGIN*(zmax-zmin);
      } else {
         Double_t dzmin = yMARGIN*(zmax-zmin);
         if (zmin >= 0 && (zmin-dzmin <= 0)) zmin  = 0;
         else                                zmin -= dzmin;
      }
   }

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("scat=")) {
      char optscat[100];
      strlcpy(optscat,opt.Data(),100);
      char *oscat = strstr(optscat,"scat=");
      char *blank = strstr(oscat," "); if (blank) *blank = 0;
      sscanf(oscat+5,"%lg",&scale);
   }
   // use an independent instance of a random generator
   // instead of gRandom to avoid conflicts and
   // to get same random numbers when drawing the same histogram
   TRandom2 random;
   marker=0;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         z     = fH->GetBinContent(bin);
         if (z < zmin) z = zmin;
         if (z > zmax) z = zmax;
         if (Hoption.Logz) {
            if (z > 0) z = TMath::Log10(z) - zmin;
         } else {
            z    -=  zmin;
         }
         if (z <= 0) continue;
         k = Int_t(z*scale);
         if (ltest) k++;
         if (k > 0) {
            for (Int_t loop=0; loop<k; loop++) {
               if (k+marker >= kNMAX) {
                  gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
                  marker=0;
               }
               fXbuf[marker] = (random.Rndm(loop)*xstep) + xk;
               fYbuf[marker] = (random.Rndm(loop)*ystep) + yk;
               if (Hoption.Logx){
                  if (fXbuf[marker] > 0) fXbuf[marker] = TMath::Log10(fXbuf[marker]);
                  else                   break;
               }
               if (Hoption.Logy){
                  if (fYbuf[marker] > 0) fYbuf[marker] = TMath::Log10(fYbuf[marker]);
                  else                  break;
               }
               if (fXbuf[marker] < gPad->GetUxmin()) break;
               if (fYbuf[marker] < gPad->GetUymin()) break;
               if (fXbuf[marker] > gPad->GetUxmax()) break;
               if (fYbuf[marker] > gPad->GetUymax()) break;
               marker++;
            }
         }
      }
   }
   if (marker > 0) gPad->PaintPolyMarker(marker, fXbuf, fYbuf);

   if (Hoption.Zscale) PaintPalette();
}


//______________________________________________________________________________
void THistPainter::PaintSpecialObjects(const TObject *obj, Option_t *option)
{
   /* Begin_html
   Static function to paint special objects like vectors and matrices.
   This function is called via gROOT->ProcessLine to paint these objects
   without having a direct dependency of the graphics or histogramming
   system.
   End_html */

   if (!obj) return;
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);

   if (obj->InheritsFrom(TMatrixFBase::Class())) {
      // case TMatrixF
      TH2F *R__TMatrixFBase = new TH2F((TMatrixFBase &)*obj);
      R__TMatrixFBase->SetBit(kCanDelete);
      R__TMatrixFBase->Draw(option);

   } else if (obj->InheritsFrom(TMatrixDBase::Class())) {
      // case TMatrixD
      TH2D *R__TMatrixDBase = new TH2D((TMatrixDBase &)*obj);
      R__TMatrixDBase->SetBit(kCanDelete);
      R__TMatrixDBase->Draw(option);

   } else if (obj->InheritsFrom(TVectorF::Class())) {
      //case TVectorF
      TH1F *R__TVectorF = new TH1F((TVectorF &)*obj);
      R__TVectorF->SetBit(kCanDelete);
      R__TVectorF->Draw(option);

   } else if (obj->InheritsFrom(TVectorD::Class())) {
      //case TVectorD
      TH1D *R__TVectorD = new TH1D((TVectorD &)*obj);
      R__TVectorD->SetBit(kCanDelete);
      R__TVectorD->Draw(option);
   }

   TH1::AddDirectory(status);
}


//______________________________________________________________________________
void THistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   /* Begin_html
   <a href="#HP07">Draw the statistics box for 1D and profile histograms.</a>
   End_html */

   static char t[100];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }

   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (!dofit) fit = 0;
   if (dofit  == 1) dofit  =  111;
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + print_mean + print_rms +
                  print_under + print_over + print_integral +
                  print_skew + print_kurt;
   Int_t print_fval    = dofit%10;
   Int_t print_ferrors = (dofit/10)%10;
   Int_t print_fchi2   = (dofit/100)%10;
   Int_t print_fprob   = (dofit/1000)%10;
   Int_t nlinesf = print_fval + print_fchi2 + print_fprob;
   if (fit) {
      if (print_fval < 2) nlinesf += fit->GetNumberFreeParameters();
      else                nlinesf += fit->GetNpar();
   }
   if (fH->InheritsFrom(TProfile::Class())) nlinesf += print_mean + print_rms;

   // Pavetext with statistics
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) { fFunctions->Remove(stats); delete stats;}
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = (nlines+nlinesf)*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*(nlines+nlinesf)*gStyle->GetStatH();
   }
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetTextFont(gStyle->GetStatFont());
      if (gStyle->GetStatFont()%10 > 2)
         stats->SetTextSize(gStyle->GetStatFontSize());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(fH->GetName());
   if (print_entries) {
      if (fH->GetEntries() < 1e7) snprintf(t,100,"%s = %-7d",gStringEntries.Data(),Int_t(fH->GetEntries()+0.5));
      else                        snprintf(t,100,"%s = %14.7g",gStringEntries.Data(),Float_t(fH->GetEntries()));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         snprintf(textstats,50,"%s  = %s%s",gStringMean.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetMean(1));
      } else {
         snprintf(textstats,50,"%s  = %s%s #pm %s%s",gStringMean.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetMean(1),fH->GetMeanError(1));
      }
      stats->AddText(t);
      if (fH->InheritsFrom(TProfile::Class())) {
         if (print_mean == 1) {
            snprintf(textstats,50,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
            snprintf(t,100,textstats,fH->GetMean(2));
         } else {
            snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                      ,"%",stats->GetStatFormat());
            snprintf(t,100,textstats,fH->GetMean(2),fH->GetMeanError(2));
         }
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         snprintf(textstats,50,"%s   = %s%s",gStringRMS.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetRMS(1));
      } else {
         snprintf(textstats,50,"%s   = %s%s #pm %s%s",gStringRMS.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetRMS(1),fH->GetRMSError(1));
      }
      stats->AddText(t);
      if(fH->InheritsFrom(TProfile::Class())) {
         if (print_rms == 1) {
            snprintf(textstats,50,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
            snprintf(t,100,textstats,fH->GetRMS(2));
         } else {
            snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
            snprintf(t,100,textstats,fH->GetRMS(2),fH->GetRMSError(2));
         }
         stats->AddText(t);
      }
   }
   if (print_under) {
      snprintf(textstats,50,"%s = %s%s",gStringUnderflow.Data(),"%",stats->GetStatFormat());
      snprintf(t,100,textstats,fH->GetBinContent(0));
      stats->AddText(t);
   }
   if (print_over) {
      snprintf(textstats,50,"%s  = %s%s",gStringOverflow.Data(),"%",stats->GetStatFormat());
      snprintf(t,100,textstats,fH->GetBinContent(fXaxis->GetNbins()+1));
      stats->AddText(t);
   }
   if (print_integral) {
      snprintf(textstats,50,"%s = %s%s",gStringIntegral.Data(),"%",stats->GetStatFormat());
      snprintf(t,100,textstats,fH->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringSkewness.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetSkewness(1));
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewness.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetSkewness(1),fH->GetSkewness(11));
      }
      stats->AddText(t);
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringKurtosis.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetKurtosis(1));
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosis.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,fH->GetKurtosis(1),fH->GetKurtosis(11));
      }
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      snprintf(textstats,50,"#chi^{2} / ndf = %s%s / %d","%",stats->GetFitFormat(),ndf);
      snprintf(t,100,textstats,(Float_t)fit->GetChisquare());
      if (print_fchi2) stats->AddText(t);
      if (print_fprob) {
         snprintf(textstats,50,"Prob  = %s%s","%",stats->GetFitFormat());
         snprintf(t,100,textstats,(Float_t)TMath::Prob(fit->GetChisquare(),ndf));
         stats->AddText(t);
      }
      if (print_fval || print_ferrors) {
         Double_t parmin,parmax;
         for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
            fit->GetParLimits(ipar,parmin,parmax);
            if (print_fval < 2 && parmin*parmax != 0 && parmin >= parmax) continue;
            if (print_ferrors) {
               snprintf(textstats,50,"%-8s = %s%s #pm %s ",fit->GetParName(ipar), "%",stats->GetFitFormat(),
                       GetBestFormat(fit->GetParameter(ipar), fit->GetParError(ipar), stats->GetFitFormat()));
               snprintf(t,100,textstats,(Float_t)fit->GetParameter(ipar)
                               ,(Float_t)fit->GetParError(ipar));
            } else {
               snprintf(textstats,50,"%-8s = %s%s ",fit->GetParName(ipar),"%",stats->GetFitFormat());
               snprintf(t,100,textstats,(Float_t)fit->GetParameter(ipar));
            }
            t[63] = 0;
            stats->AddText(t);
         }
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}


//______________________________________________________________________________
void THistPainter::PaintStat2(Int_t dostat, TF1 *fit)
{
   /* Begin_html
   <a href="#HP07">Draw the statistics box for 2D histograms.</a>
   End_html */

   if (fH->GetDimension() != 2) return;
   TH2 *h2 = (TH2*)fH;

   static char t[100];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }
   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + 2*print_mean + 2*print_rms + print_integral;
   if (print_under || print_over) nlines += 3;

   // Pavetext with statistics
   if (!gStyle->GetOptFit()) fit = 0;
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) delete stats;
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = nlines*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*nlines*gStyle->GetStatH();
   }
   if (fit) stath += gStyle->GetStatH();
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetTextFont(gStyle->GetStatFont());
      if (gStyle->GetStatFont()%10 > 2)
         stats->SetTextSize(gStyle->GetStatFontSize());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(h2->GetName());
   if (print_entries) {
      if (h2->GetEntries() < 1e7) snprintf(t,100,"%s = %-7d",gStringEntries.Data(),Int_t(h2->GetEntries()+0.5));
      else                        snprintf(t,100,"%s = %14.7g",gStringEntries.Data(),Float_t(h2->GetEntries()));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat());
         snprintf(t,50,textstats,h2->GetMean(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetMean(2));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetMean(1),h2->GetMeanError(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetMean(2),h2->GetMeanError(2));
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetRMS(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetRMS(2));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetRMS(1),h2->GetRMSError(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetRMS(2),h2->GetRMSError(2));
         stats->AddText(t);
      }
   }
   if (print_integral) {
      snprintf(t,100,"%s  = %6.4g",gStringIntegral.Data(),h2->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetSkewness(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetSkewness(2));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetSkewness(1),h2->GetSkewness(11));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetSkewness(2),h2->GetSkewness(12));
         stats->AddText(t);
      }
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetKurtosis(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetKurtosis(2));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetKurtosis(1),h2->GetKurtosis(11));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h2->GetKurtosis(2),h2->GetKurtosis(12));
         stats->AddText(t);
      }
   }
   if (print_under || print_over) {
      //get 3*3 under/overflows for 2d hist
      Double_t unov[9];

      unov[0] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[1] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[2] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[3] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[4] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[5] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[6] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,0,h2->GetYaxis()->GetFirst()-1);
      unov[7] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),0,h2->GetYaxis()->GetFirst()-1);
      unov[8] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,0,h2->GetYaxis()->GetFirst()-1);

      snprintf(t, 100," %7d|%7d|%7d\n", (Int_t)unov[0], (Int_t)unov[1], (Int_t)unov[2]);
      stats->AddText(t);
      if (h2->GetEntries() < 1e7)
         snprintf(t, 100," %7d|%7d|%7d\n", (Int_t)unov[3], (Int_t)unov[4], (Int_t)unov[5]);
      else
         snprintf(t, 100," %7d|%14.7g|%7d\n", (Int_t)unov[3], (Float_t)unov[4], (Int_t)unov[5]);
      stats->AddText(t);
      snprintf(t, 100," %7d|%7d|%7d\n", (Int_t)unov[6], (Int_t)unov[7], (Int_t)unov[8]);
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      snprintf(t,100,"#chi^{2} / ndf = %6.4g / %d",(Float_t)fit->GetChisquare(),ndf);
      stats->AddText(t);
      for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
         snprintf(t,100,"%-8s = %5.4g #pm %5.4g ",fit->GetParName(ipar)
                                   ,(Float_t)fit->GetParameter(ipar)
                                   ,(Float_t)fit->GetParError(ipar));
         t[32] = 0;
         stats->AddText(t);
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}


//______________________________________________________________________________
void THistPainter::PaintStat3(Int_t dostat, TF1 *fit)
{
   /* Begin_html
   <a href="#HP07">Draw the statistics box for 3D histograms.</a>
   End_html */

   if (fH->GetDimension() != 3) return;
   TH3 *h3 = (TH3*)fH;

   static char t[100];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }
   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + 3*print_mean + 3*print_rms + print_integral;
   if (print_under || print_over) nlines += 3;

   // Pavetext with statistics
   if (!gStyle->GetOptFit()) fit = 0;
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) delete stats;
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = nlines*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*nlines*gStyle->GetStatH();
   }
   if (fit) stath += gStyle->GetStatH();
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetTextFont(gStyle->GetStatFont());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(h3->GetName());
   if (print_entries) {
      if (h3->GetEntries() < 1e7) snprintf(t,100,"%s = %-7d",gStringEntries.Data(),Int_t(h3->GetEntries()+0.5));
      else                        snprintf(t,100,"%s = %14.7g",gStringEntries.Data(),Float_t(h3->GetEntries()+0.5));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringMeanZ.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(3));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(1),h3->GetMeanError(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(2),h3->GetMeanError(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringMeanZ.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetMean(3),h3->GetMeanError(3));
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringRMSZ.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(3));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(1),h3->GetRMSError(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(2),h3->GetRMSError(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringRMSZ.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetRMS(3),h3->GetRMSError(3));
         stats->AddText(t);
      }
   }
   if (print_integral) {
      snprintf(t,100,"%s  = %6.4g",gStringIntegral.Data(),h3->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringSkewnessZ.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(3));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(1),h3->GetSkewness(11));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(2),h3->GetSkewness(12));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringSkewnessZ.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetSkewness(3),h3->GetSkewness(13));
         stats->AddText(t);
      }
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         snprintf(textstats,50,"%s = %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(1));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(2));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s",gStringKurtosisZ.Data(),"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(3));
         stats->AddText(t);
      } else {
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(1),h3->GetKurtosis(11));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(2),h3->GetKurtosis(12));
         stats->AddText(t);
         snprintf(textstats,50,"%s = %s%s #pm %s%s",gStringKurtosisZ.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         snprintf(t,100,textstats,h3->GetKurtosis(3),h3->GetKurtosis(13));
         stats->AddText(t);
      }
   }
   if (print_under || print_over) {
      // no underflow - overflow printing for a 3D histogram
      // one would need a 3D table
//       //get 3*3 under/overflows for 2d hist
//       Double_t unov[9];

//       unov[0] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
//       unov[1] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
//       unov[2] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
//       unov[3] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
//       unov[4] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
//       unov[5] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
//       unov[6] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,0,h3->GetYaxis()->GetFirst()-1);
//       unov[7] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),0,h3->GetYaxis()->GetFirst()-1);
//       unov[8] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,0,h3->GetYaxis()->GetFirst()-1);

//       sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[0], (Int_t)unov[1], (Int_t)unov[2]);
//       stats->AddText(t);
//       if (h3->GetEntries() < 1e7)
//          sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[3], (Int_t)unov[4], (Int_t)unov[5]);
//       else
//          sprintf(t, " %7d|%14.7g|%7d\n", (Int_t)unov[3], (Float_t)unov[4], (Int_t)unov[5]);
//       stats->AddText(t);
//       sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[6], (Int_t)unov[7], (Int_t)unov[8]);
//       stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      snprintf(t,100,"#chi^{2} / ndf = %6.4g / %d",(Float_t)fit->GetChisquare(),ndf);
      stats->AddText(t);
      for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
         snprintf(t,100,"%-8s = %5.4g #pm %5.4g ",fit->GetParName(ipar)
                                   ,(Float_t)fit->GetParameter(ipar)
                                   ,(Float_t)fit->GetParError(ipar));
         t[32] = 0;
         stats->AddText(t);
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}


//______________________________________________________________________________
void THistPainter::PaintSurface(Option_t *)
{
   /* Begin_html
   <a href="#HP18">Control function to draw a 2D histogram as a surface plot.</a>
   End_html */

   const Double_t ydiff = 1;
   const Double_t yligh1 = 10;
   const Double_t qa = 0.15;
   const Double_t qd = 0.15;
   const Double_t qs = 0.8;
   Double_t fmin, fmax;
   Int_t raster = 0;
   Int_t irep   = 0;

   if (Hparam.zmin == 0 && Hparam.zmax == 0) {Hparam.zmin = -1; Hparam.zmax = 1;}
   Int_t   nx      = Hparam.xlast - Hparam.xfirst;
   Int_t   ny      = Hparam.ylast - Hparam.yfirst;
   Double_t zmin   = Hparam.zmin;
   Double_t zmax   = Hparam.zmax;
   Double_t xlab1  = Hparam.xmin;
   Double_t xlab2  = Hparam.xmax;
   Double_t ylab1  = Hparam.ymin;
   Double_t ylab2  = Hparam.ymax;
   Double_t dangle = 10*3.141592/180; //Delta angle for Rapidity option
   Double_t deltaz = TMath::Abs(zmin);
   if (deltaz == 0) deltaz = 1;
   if (zmin >= zmax) {
      zmin -= 0.5*deltaz;
      zmax += 0.5*deltaz;
   }
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*(1+gStyle->GetHistTopMargin());
   //     Compute the lego limits and instantiate a lego object
   fXbuf[0] = -1;
   fYbuf[0] =  1;
   fXbuf[1] = -1;
   fYbuf[1] =  1;
   if (Hoption.System >= kPOLAR && (Hoption.Surf == 1 || Hoption.Surf == 13)) raster = 1;
   if (Hoption.System == kPOLAR) {
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Logy) {
         if (ylab1 > 0) fXbuf[2] = TMath::Log10(ylab1);
         else           fXbuf[2] = 0;
         if (ylab2 > 0) fYbuf[2] = TMath::Log10(ylab2);
         else           fYbuf[2] = 0;
      } else {
         fXbuf[2] = ylab1;
         fYbuf[2] = ylab2;
      }
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kSPHERICAL) {
      fXbuf[2] = -1;
      fYbuf[2] =  1;
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kRAPIDITY) {
      fXbuf[2] = -1/TMath::Tan(dangle);
      fYbuf[2] =  1/TMath::Tan(dangle);
   } else {
      fXbuf[0] = xlab1;
      fYbuf[0] = xlab2;
      fXbuf[1] = ylab1;
      fYbuf[1] = ylab2;
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   }

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf, Hoption.System);
   fLego->SetLineColor(fH->GetLineColor());
   fLego->SetFillColor(fH->GetFillColor());

   //          Create axis object

   TGaxis *axis = new TGaxis();

   //                  Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

   if (Hoption.Surf == 13 || Hoption.Surf == 15) fLego->SetMesh(3);
   if (Hoption.Surf == 12 || Hoption.Surf == 14 || Hoption.Surf == 17) fLego->SetMesh(0);

   //     Close the surface in case of non cartesian coordinates.

   if (Hoption.System != kCARTESIAN) {nx++; ny++;}

   //     Now ready to draw the surface plot

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintSurface", "no TView in current pad");
      return;
   }

   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   //     Set color/style for back box
   if (Hoption.Same) {
      fLego->SetFillStyle(0);
      fLego->SetFillColor(1);
   } else {
      fLego->SetFillStyle(gPad->GetFrameFillStyle());
      fLego->SetFillColor(gPad->GetFrameFillColor());
   }
   fLego->TAttFill::Modify();

   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   //     Draw the filled contour on top
   Int_t icol1 = fH->GetFillColor();

   Int_t hoption35 = Hoption.Surf;
   if (Hoption.Surf == 13 || Hoption.Surf == 15) {
      DefineColorLevels(ndivz);
      Hoption.Surf = 23;
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::SurfaceFunction);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
      Hoption.Surf = hoption35;
      fLego->SetMesh(1);
   }

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 14 || Hoption.Surf == 17) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

   //     Gouraud Shading surface
   if (Hoption.Surf == 14) {
   //    Set light sources
      fLego->LightSource(0, ydiff, 0,0,0,irep);
      fLego->LightSource(1, yligh1 ,1,1,1,irep);
      fLego->SurfaceProperty(qa, qd, qs, 1, irep);
      fmin = ydiff*qa;
      fmax = fmin + (yligh1+0.1)*(qd+qs);
      Int_t nbcol = 28;
      icol1 = 201;
      Double_t dcol = 0.5/Double_t(nbcol);
      TColor *colref = gROOT->GetColor(fH->GetFillColor());
      if (!colref) return;
      Float_t r,g,b,hue,light,satur;
      colref->GetRGB(r,g,b);
      TColor::RGBtoHLS(r,g,b,hue,light,satur);
      TColor *acol;
      for (Int_t col=0;col<nbcol;col++) {
         acol = gROOT->GetColor(col+icol1);
         TColor::HLStoRGB(hue,.4+col*dcol,satur,r,g,b);
         acol->SetRGB(r,g,b);
      }
      fLego->Spectrum(nbcol, fmin, fmax, icol1, 1, irep);
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::GouraudFunction);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
   } else if (Hoption.Surf == 15) {
   // The surface is not drawn in this case.
   } else {
   //     Draw the surface
      if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 16 || Hoption.Surf == 17) {
         DefineColorLevels(ndivz);
      } else {
         fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      }
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::SurfaceFunction);
      if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceRaster1);
      if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfacePolar(1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SurfacePolar(1,nx,ny,"BF");
      } else if (Hoption.System == kCYLINDRICAL) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceCylindrical(1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      } else if (Hoption.System == kSPHERICAL) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceSpherical(0,1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      } else if (Hoption.System == kRAPIDITY) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceSpherical(1,1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      } else {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         if (Hoption.Surf == 16) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove3);
         if (Hoption.Surf ==  1 || Hoption.Surf == 13 || Hoption.Surf == 16) fLego->SurfaceCartesian(90,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 17) fLego->SurfaceCartesian(90,nx,ny,"BF");
      }
   }

   // Paint the line contour on top for option SURF7
   if (Hoption.Surf == 17) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      Hoption.Surf = 23;
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::SurfaceFunction);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove3);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"FB");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"FB");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"FB");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"FB");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"FB");
   }

   if ((!Hoption.Same) &&
       (Hoption.Surf == 1 || Hoption.Surf == 13 || Hoption.Surf == 16)) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   if (Hoption.Zscale) PaintPalette();

   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintTriangles(Option_t *option)
{
   /* Begin_html
   Control function to draw a table using Delaunay triangles.
   End_html */

   TGraphDelaunay *dt;

   // Check if fH contains a TGraphDelaunay
   TList *hl = fH->GetListOfFunctions();
   dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
   if (!dt) return;

   // If needed, create a TGraph2DPainter
   if (!fGraph2DPainter) fGraph2DPainter = new TGraph2DPainter(dt);

   // Define the 3D view
   if (Hoption.Same) {
      TView *viewsame = gPad->GetView();
      if (!viewsame) {
         Error("PaintTriangles", "no TView in current pad, do not use option SAME");
         return;
      }
      Double_t *rmin = viewsame->GetRmin();
      Double_t *rmax = viewsame->GetRmax();
      fXbuf[0] = rmin[0];
      fYbuf[0] = rmax[0];
      fXbuf[1] = rmin[1];
      fYbuf[1] = rmax[1];
      fXbuf[2] = rmin[2];
      fYbuf[2] = rmax[2];
   } else {
      fXbuf[0] = Hparam.xmin;
      fYbuf[0] = Hparam.xmax;
      fXbuf[1] = Hparam.ymin;
      fYbuf[1] = Hparam.ymax;
      fXbuf[2] = Hparam.zmin;
      fYbuf[2] = Hparam.zmax;
   }

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);
   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTriangles", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   Int_t irep;
   view->SetView(phideg, thedeg, psideg, irep);

   // Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();
   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);
   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   // Paint the Back Box if needed
   if (Hoption.BackBox && !Hoption.Same) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   // Paint the triangles
   fGraph2DPainter->Paint(option);

   // Paint the Front Box if needed
   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }

   // Paint the Axis if needed
   if (!Hoption.Axis && !Hoption.Same) {
      TGaxis *axis = new TGaxis();
      PaintLegoAxis(axis, 90);
      delete axis;
   }

   if (Hoption.Zscale) PaintPalette();

   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::DefineColorLevels(Int_t ndivz)
{
   /* Begin_html
   Define the color levels used to paint legos, surfaces etc..
   End_html */

   Int_t i, irep;

   // Initialize the color levels
   if (ndivz >= 100) {
      Warning("PaintSurface", "too many color levels, %d, reset to 8", ndivz);
      ndivz = 8;
   }
   Double_t *funlevel = new Double_t[ndivz+1];
   Int_t *colorlevel = new Int_t[ndivz+1];
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   for (i = 0; i < ndivz; ++i) {
      funlevel[i] = fH->GetContourLevelPad(i);
      theColor = Int_t((i+0.99)*Float_t(ncolors)/Float_t(ndivz));
      colorlevel[i] = gStyle->GetColorPalette(theColor);
   }
   colorlevel[ndivz] = gStyle->GetColorPalette(ncolors-1);
   fLego->ColorFunction(ndivz, funlevel, colorlevel, irep);
   delete [] colorlevel;
   delete [] funlevel;
}


//______________________________________________________________________________
void THistPainter::PaintTable(Option_t *option)
{
   /* Begin_html
   <a href="#HP01c">Control function to draw 2D/3D histograms (tables).</a>
   End_html */

   if (!TableInit()) return;  //fill Hparam structure with histo parameters

   PaintFrame();

   //if palette option not specified, delete a possible existing palette
   if (!Hoption.Zscale) {
      TObject *palette = fFunctions->FindObject("palette");
      if (palette) delete palette;
   }

   if (fH->InheritsFrom(TH2Poly::Class())) {
      if (Hoption.Fill)    PaintTH2PolyBins("f");
      if (Hoption.Color)   PaintTH2PolyColorLevels(option);
      if (Hoption.Scat)    PaintTH2PolyScatterPlot(option);
      if (Hoption.Text)    PaintTH2PolyText(option);
      if (Hoption.Line)    PaintTH2PolyBins("l");
      if (Hoption.Mark)    PaintTH2PolyBins("P");
   } else if (fH->GetEntries() != 0 && Hoption.Axis<=0) {
      if (Hoption.Scat)    PaintScatterPlot(option);
      if (Hoption.Arrow)   PaintArrows(option);
      if (Hoption.Box)     PaintBoxes(option);
      if (Hoption.Color)   PaintColorLevels(option);
      if (Hoption.Contour) PaintContour(option);
      if (Hoption.Text)    PaintText(option);
      if (Hoption.Error >= 100)   Paint2DErrors(option);
   }

   if (Hoption.Lego) PaintLego(option);
   if (Hoption.Surf && !Hoption.Contour) PaintSurface(option);
   if (Hoption.Tri) PaintTriangles(option);

   if (!Hoption.Lego && !Hoption.Surf &&
       !Hoption.Tri  && !(Hoption.Error >= 100)) PaintAxis(kFALSE); // Draw the axes

   PaintTitle();    //    Draw histogram title

   TF1 *fit  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TF1::Class())) {
         fit = (TF1*)obj;
         break;
      }
   }
   if (Hoption.Same != 1) {
      if (!fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
         PaintStat2(gStyle->GetOptStat(),fit);
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintTH2PolyBins(Option_t *option)
{
   /* Begin_html
    Control function to draw a TH2Poly bins' contours.
    option = "F" draw the bins as filled areas.
    option = "L" draw the bins as line.
    option = "P" draw the bins as markers.
    End_html */

   TString opt = option;
   opt.ToLower();
   Bool_t line = kFALSE;
   Bool_t fill = kFALSE;
   Bool_t mark = kFALSE;
   if (opt.Contains("l")) line = kTRUE;
   if (opt.Contains("f")) fill = kTRUE;
   if (opt.Contains("p")) mark = kTRUE;

   TH2PolyBin  *b;

   TIter next(((TH2Poly*)fH)->GetBins());
   TObject *obj, *poly;

   while ((obj=next())) {
      b     = (TH2PolyBin*)obj;
      poly  = b->GetPolygon();

      // Paint the TGraph bins.
      if (poly->IsA() == TGraph::Class()) {
         TGraph *g  = (TGraph*)poly;
         g->TAttLine::Modify();
         g->TAttMarker::Modify();
         g->TAttFill::Modify();
         if (line) g->Paint("L");
         if (fill) g->Paint("F");
         if (mark) g->Paint("P");
      }

      // Paint the TMultiGraph bins.
      if (poly->IsA() == TMultiGraph::Class()) {
         TMultiGraph *mg = (TMultiGraph*)poly;
         TList *gl = mg->GetListOfGraphs();
         if (!gl) return;
         TGraph *g;
         TIter nextg(gl);
         while ((g = (TGraph*) nextg())) {
            g->TAttLine::Modify();
            g->TAttMarker::Modify();
            g->TAttFill::Modify();
            if (line) g->Paint("L");
            if (fill) g->Paint("F");
            if (mark) g->Paint("P");
         }
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintTH2PolyColorLevels(Option_t *)
{
   /* Begin_html
    <a href="#HP20a">Control function to draw a TH2Poly as a color plot.</a>
    End_html */

   Int_t ncolors, color, theColor;
   Double_t z, zc;
   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();
   if (Hoption.Logz) {
      if (zmax > 0) {
         if (zmin <= 0) zmin = TMath::Min((Double_t)1, (Double_t)0.001*zmax);
         zmin = TMath::Log10(zmin);
         zmax = TMath::Log10(zmax);
      } else {
         return;
      }
   }
   Double_t dz = zmax - zmin;

   // Initialize the levels on the Z axis
   ncolors     = gStyle->GetNumberOfColors();
   Int_t ndiv  = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);
   Double_t scale = ndivz/dz;

   TH2PolyBin  *b;

   TIter next(((TH2Poly*)fH)->GetBins());
   TObject *obj, *poly;

   while ((obj=next())) {
      b     = (TH2PolyBin*)obj;
      poly  = b->GetPolygon();

      z = b->GetContent();
      if (Hoption.Logz) {
         if (z > 0) z = TMath::Log10(z);
         else       z = zmin;
      }
      if (z < zmin) continue;

      // Define the bin color.
      if (fH->TestBit(TH1::kUserContour)) {
         zc = fH->GetContourLevelPad(0);
         if (z < zc) continue;
         color = -1;
         for (Int_t k=0; k<ndiv; k++) {
            zc = fH->GetContourLevelPad(k);
            if (z < zc) {
               continue;
            } else {
               color++;
            }
         }
      } else {
         color = Int_t(0.01+(z-zmin)*scale);
      }
      theColor = Int_t((color+0.99)*Float_t(ncolors)/Float_t(ndivz));
      if (theColor > ncolors-1) theColor = ncolors-1;

      // Paint the TGraph bins.
      if (poly->IsA() == TGraph::Class()) {
         TGraph *g  = (TGraph*)poly;
         g->SetFillColor(gStyle->GetColorPalette(theColor));
         g->TAttFill::Modify();
         g->Paint("F");
      }

      // Paint the TMultiGraph bins.
      if (poly->IsA() == TMultiGraph::Class()) {
         TMultiGraph *mg = (TMultiGraph*)poly;
         TList *gl = mg->GetListOfGraphs();
         if (!gl) return;
         TGraph *g;
         TIter nextg(gl);
         while ((g = (TGraph*) nextg())) {
            g->SetFillColor(gStyle->GetColorPalette(theColor));
            g->TAttFill::Modify();
            g->Paint("F");
         }
      }
   }
   if (Hoption.Zscale) PaintPalette();
}


//______________________________________________________________________________
void THistPainter::PaintTH2PolyScatterPlot(Option_t *)
{
   /* Begin_html
    <a href="#HP20a">Control function to draw a TH2Poly as a scatter plot.</a>
    End_html */

   Int_t k, loop, marker=0;
   Double_t z, xk,xstep, yk, ystep, xp, yp;
   Double_t scale = 1;
   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();
   if (Hoption.Logz) {
      if (zmax > 0) {
         if (zmin <= 0) zmin = TMath::Min((Double_t)1, (Double_t)0.001*zmax);
         zmin = TMath::Log10(zmin);
         zmax = TMath::Log10(zmax);
      } else {
         return;
      }
   }
   Double_t dz = zmax - zmin;
   scale = (kNMAX-1)/dz;


   // use an independent instance of a random generator
   // instead of gRandom to avoid conflicts and
   // to get same random numbers when drawing the same histogram
   TRandom2 random;

   TH2PolyBin  *b;

   TIter next(((TH2Poly*)fH)->GetBins());
   TObject *obj, *poly;

   Double_t maxarea = 0, a;
   while ((obj=next())) {
      b     = (TH2PolyBin*)obj;
      a     = b->GetArea();
      if (a>maxarea) maxarea = a;
   }

   next.Reset();

   while ((obj=next())) {
      b     = (TH2PolyBin*)obj;
      poly  = b->GetPolygon();
      z     = b->GetContent();
      if (z < zmin) z = zmin;
      if (z > zmax) z = zmax;
      if (Hoption.Logz) {
         if (z > 0) z = TMath::Log10(z) - zmin;
      } else {
         z    -=  zmin;
      }
      k     = Int_t((z*scale)*(b->GetArea()/maxarea));
      xk    = b->GetXMin();
      yk    = b->GetYMin();
      xstep = b->GetXMax()-xk;
      ystep = b->GetYMax()-yk;

      // Paint the TGraph bins.
      if (poly->IsA() == TGraph::Class()) {
         TGraph *g  = (TGraph*)poly;
         if (k <= 0 || z <= 0) continue;
         loop = 0;
         while (loop<k) {
            if (k+marker >= kNMAX) {
               gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
               marker=0;
            }
            xp = (random.Rndm(loop)*xstep) + xk;
            yp = (random.Rndm(loop)*ystep) + yk;
            if (g->IsInside(xp,yp)) {
               fXbuf[marker] = xp;
               fYbuf[marker] = yp;
               marker++;
               loop++;
            }
         }
         if (marker > 0) gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
      }

      // Paint the TMultiGraph bins.
      if (poly->IsA() == TMultiGraph::Class()) {
         TMultiGraph *mg = (TMultiGraph*)poly;
         TList *gl = mg->GetListOfGraphs();
         if (!gl) return;
         if (k <= 0 || z <= 0) continue;
         loop = 0;
         while (loop<k) {
            if (k+marker >= kNMAX) {
               gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
               marker=0;
            }
            xp = (random.Rndm(loop)*xstep) + xk;
            yp = (random.Rndm(loop)*ystep) + yk;
            if (mg->IsInside(xp,yp)) {
               fXbuf[marker] = xp;
               fYbuf[marker] = yp;
               marker++;
               loop++;
            }
         }
         if (marker > 0) gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
      }
   }
   PaintTH2PolyBins("l");
}


//______________________________________________________________________________
void THistPainter::PaintTH2PolyText(Option_t *)
{
   /* Begin_html
    <a href="#HP20a">Control function to draw a TH2Poly as a text plot.</a>
    End_html */

   TLatex text;
   text.SetTextFont(gStyle->GetTextFont());
   text.SetTextColor(fH->GetMarkerColor());
   text.SetTextSize(0.02*fH->GetMarkerSize());

   Double_t x, y, z, e, angle = 0;
   char value[50];
   char format[32];
   snprintf(format,32,"%s%s","%",gStyle->GetPaintTextFormat());
   if (Hoption.Text >= 1000) angle = Hoption.Text%1000;
   Int_t opt = (Int_t)Hoption.Text/1000;

   text.SetTextAlign(22);
   if (Hoption.Text ==  1) angle = 0;
   text.SetTextAngle(angle);
   text.TAttText::Modify();

   TH2PolyBin *b;

   TIter next(((TH2Poly*)fH)->GetBins());
   TObject *obj, *p;

   while ((obj=next())) {
      b = (TH2PolyBin*)obj;
      p = b->GetPolygon();
      x = (b->GetXMin()+b->GetXMax())/2;
      if (Hoption.Logx) {
         if (x > 0)  x  = TMath::Log10(x);
         else continue;
      }
      y = (b->GetYMin()+b->GetYMax())/2;
      if (Hoption.Logy) {
         if (y > 0)  y  = TMath::Log10(y);
         else continue;
      }
      z = b->GetContent();
      if (z < Hparam.zmin || (z == 0 && !gStyle->GetHistMinimumZero()) ) continue;
      if (opt==2) {
         e = fH->GetBinError(b->GetBinNumber());
         snprintf(format,32,"#splitline{%s%s}{#pm %s%s}",
                                    "%",gStyle->GetPaintTextFormat(),
                                    "%",gStyle->GetPaintTextFormat());
         snprintf(value,50,format,z,e);
      } else {
         snprintf(value,50,format,z);
      }
      if (opt==3) text.PaintLatex(x,y,angle,0.02*fH->GetMarkerSize(),p->GetName());
      else        text.PaintLatex(x,y,angle,0.02*fH->GetMarkerSize(),value);
   }

   PaintTH2PolyBins("l");
}


//______________________________________________________________________________
void THistPainter::PaintText(Option_t *)
{
   /* Begin_html
   <a href="#HP15">Control function to draw a 1D/2D histograms with the bin values.</a>
   End_html */

   TLatex text;
   text.SetTextFont(gStyle->GetTextFont());
   text.SetTextColor(fH->GetMarkerColor());
   text.SetTextSize(0.02*fH->GetMarkerSize());

   Double_t x, y, z, e, angle = 0;
   char value[50];
   char format[32];
   snprintf(format,32,"%s%s","%",gStyle->GetPaintTextFormat());
   if (Hoption.Text >= 1000) angle = Hoption.Text%1000;

   // 1D histograms
   if (fH->GetDimension() == 1) {
      Bool_t getentries = kFALSE;
      Double_t yt;
      TProfile *hp = (TProfile*)fH;
      if (Hoption.Text>2000 && fH->InheritsFrom(TProfile::Class())) {
         Hoption.Text = Hoption.Text-2000;
         getentries = kTRUE;
      }
      if (Hoption.Text ==  1) angle = 90;
      text.SetTextAlign(11);
      if (angle == 90) text.SetTextAlign(12);
      if (angle ==  0) text.SetTextAlign(21);
      text.TAttText::Modify();
      Double_t dt = 0.02*(gPad->GetY2()-gPad->GetY1());
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         x  = fH->GetXaxis()->GetBinCenter(i);
         y  = fH->GetBinContent(i);
         yt = y;
         if (getentries) yt = hp->GetBinEntries(i);
         snprintf(value,50,format,yt);
         if (Hoption.Logx) {
            if (x > 0)  x  = TMath::Log10(x);
            else continue;
         }
         if (Hoption.Logy) {
            if (y > 0)  y  = TMath::Log10(y);
            else continue;
         }
         if (y >= gPad->GetY2()) continue;
         if (y <= gPad->GetY1()) continue;
         text.PaintLatex(x,y+0.2*dt,angle,0.02*fH->GetMarkerSize(),value);
      }

   // 2D histograms
   } else {
      text.SetTextAlign(22);
      if (Hoption.Text ==  1) angle = 0;
      text.SetTextAngle(angle);
      text.TAttText::Modify();
      for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
         y    = fYaxis->GetBinCenter(j);
         if (Hoption.Logy) {
            if (y > 0)  y  = TMath::Log10(y);
            else continue;
         }
         for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
            Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
            x    = fXaxis->GetBinCenter(i);
            if (Hoption.Logx) {
               if (x > 0)  x  = TMath::Log10(x);
               else continue;
            }
            if (!IsInside(x,y)) continue;
            z = fH->GetBinContent(bin);
            if (z < Hparam.zmin || (z == 0 && !gStyle->GetHistMinimumZero()) ) continue;
            if (Hoption.Text>2000) {
               e = fH->GetBinError(bin);
               snprintf(format,32,"#splitline{%s%s}{#pm %s%s}",
                                          "%",gStyle->GetPaintTextFormat(),
                                          "%",gStyle->GetPaintTextFormat());
               snprintf(value,50,format,z,e);
            } else {
               snprintf(value,50,format,z);
            }
            text.PaintLatex(x,y,angle,0.02*fH->GetMarkerSize(),value);
         }
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintTF3()
{
   /* Begin_html
   <a href="#HP27">Control function to draw a 3D implicit functions.</a>
   End_html */

   Int_t irep;

   TGaxis *axis = new TGaxis();
   TAxis *xaxis = fH->GetXaxis();
   TAxis *yaxis = fH->GetYaxis();
   TAxis *zaxis = fH->GetZaxis();

   fXbuf[0] = xaxis->GetBinLowEdge(xaxis->GetFirst());
   fYbuf[0] = xaxis->GetBinUpEdge(xaxis->GetLast());
   fXbuf[1] = yaxis->GetBinLowEdge(yaxis->GetFirst());
   fYbuf[1] = yaxis->GetBinUpEdge(yaxis->GetLast());
   fXbuf[2] = zaxis->GetBinLowEdge(zaxis->GetFirst());
   fYbuf[2] = zaxis->GetBinUpEdge(zaxis->GetLast());

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTF3", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.BackBox) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode1);

   fLego->ImplicitFunction(fXbuf, fYbuf, fH->GetNbinsX(),
                                         fH->GetNbinsY(),
                                         fH->GetNbinsZ(), "BF");

   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   PaintTitle();

   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintTitle()
{
   /* Begin_html
   Draw the histogram title
   <p>
   The title is drawn according to the title alignment returned by
   <tt>GetTitleAlign()</tt>. It is a 2 digits integer): hv
   <p>
   where <tt>"h"</tt> is the horizontal alignment and <tt>"v"</tt> is the
   vertical alignment.
   <ul>
   <li> <tt>"h"</tt> can get the values 1 2 3 for left, center, and right
   <li> <tt>"v"</tt> can get the values 1 2 3 for bottom, middle and top
   </ul>
   for instance the default alignment is: 13 (left top)
   End_html */

   if (Hoption.Same) return;
   if (fH->TestBit(TH1::kNoTitle)) return;
   Int_t nt = strlen(fH->GetTitle());
   TPaveText *title = 0;
   TObject *obj;
   TIter next(gPad->GetListOfPrimitives());
   while ((obj = next())) {
      if (!obj->InheritsFrom(TPaveText::Class())) continue;
      title = (TPaveText*)obj;
      if (strcmp(title->GetName(),"title")) {title = 0; continue;}
      break;
   }
   if (nt == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht <= 0) ht = 1.1*gStyle->GetTitleFontSize();
   if (ht <= 0) ht = 0.05;
   if (wt <= 0) {
      TLatex l;
      l.SetTextSize(ht);
      l.SetTitle(fH->GetTitle());
      // adjustment in case the title has several lines (#splitline)
      ht = TMath::Max(ht, 1.2*l.GetYsize()/(gPad->GetY2() - gPad->GetY1()));
      Double_t wndc = l.GetXsize()/(gPad->GetX2() - gPad->GetX1());
      wt = TMath::Min(0.7, 0.02+wndc);
   }
   if (title) {
      TText *t0 = (TText*)title->GetLine(0);
      if (t0) {
         if (!strcmp(t0->GetTitle(),fH->GetTitle())) return;
         t0->SetTitle(fH->GetTitle());
         if (wt > 0) title->SetX2NDC(title->GetX1NDC()+wt);
      }
      return;
   }

   Int_t talh = gStyle->GetTitleAlign()/10;
   if (talh < 1) talh = 1; if (talh > 3) talh = 3;
   Int_t talv = gStyle->GetTitleAlign()%10;
   if (talv < 1) talv = 1; if (talv > 3) talv = 3;
   Double_t xpos, ypos;
   xpos = gStyle->GetTitleX();
   ypos = gStyle->GetTitleY();
   if (talh == 2) xpos = xpos-wt/2.;
   if (talh == 3) xpos = xpos-wt;
   if (talv == 2) ypos = ypos+ht/2.;
   if (talv == 1) ypos = ypos+ht;

   TPaveText *ptitle = new TPaveText(xpos, ypos-ht, xpos+wt, ypos,"blNDC");

   //     box with the histogram title
   ptitle->SetFillColor(gStyle->GetTitleFillColor());
   ptitle->SetFillStyle(gStyle->GetTitleStyle());
   ptitle->SetName("title");
   ptitle->SetBorderSize(gStyle->GetTitleBorderSize());
   ptitle->SetTextColor(gStyle->GetTitleTextColor());
   ptitle->SetTextFont(gStyle->GetTitleFont(""));
   if (gStyle->GetTitleFont("")%10 > 2)
      ptitle->SetTextSize(gStyle->GetTitleFontSize());
   ptitle->AddText(fH->GetTitle());
   ptitle->SetBit(kCanDelete);
   ptitle->Draw();
   ptitle->Paint();

}


//______________________________________________________________________________
void THistPainter::ProcessMessage(const char *mess, const TObject *obj)
{
   /* Begin_html
    Process message "mess".
   End_html */

   if (!strcmp(mess,"SetF3")) {
      TPainter3dAlgorithms::SetF3((TF3*)obj);
   } else if (!strcmp(mess,"SetF3ClippingBoxOff")) {
      TPainter3dAlgorithms::SetF3ClippingBoxOff();
   } else if (!strcmp(mess,"SetF3ClippingBoxOn")) {
      TVectorD &v =  (TVectorD&)(*obj);
      Double_t xclip = v(0);
      Double_t yclip = v(1);
      Double_t zclip = v(2);
      TPainter3dAlgorithms::SetF3ClippingBoxOn(xclip,yclip,zclip);
   }
}


//______________________________________________________________________________
Int_t THistPainter::ProjectAitoff2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   /* Begin_html
   Static function.<br>
   Convert Right Ascension, Declination to X,Y using an AITOFF projection.
   This procedure can be used to create an all-sky map in Galactic
   coordinates with an equal-area Aitoff projection.  Output map
   coordinates are zero longitude centered.
   Also called Hammer-Aitoff projection (first presented by Ernst von Hammer in 1892)
   <p>
   source: GMT<br>
   code from  Ernst-Jan Buis
   End_html */

   Double_t x, y;

   Double_t alpha2 = (l/2)*TMath::DegToRad();
   Double_t delta  = b*TMath::DegToRad();
   Double_t r2     = TMath::Sqrt(2.);
   Double_t f      = 2*r2/TMath::Pi();
   Double_t cdec   = TMath::Cos(delta);
   Double_t denom  = TMath::Sqrt(1. + cdec*TMath::Cos(alpha2));
   x      = cdec*TMath::Sin(alpha2)*2.*r2/denom;
   y      = TMath::Sin(delta)*r2/denom;
   x     *= TMath::RadToDeg()/f;
   y     *= TMath::RadToDeg()/f;
   //  x *= -1.; // for a skymap swap left<->right
   Al = x;
   Ab = y;

   return 0;
}


//______________________________________________________________________________
Int_t THistPainter::ProjectMercator2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   /* Begin_html
   Static function <br>
   Probably the most famous of the various map projections, the Mercator projection
   takes its name from Mercator who presented it in 1569. It is a cylindrical, conformal projection
   with no distortion along the equator.
   The Mercator projection has been used extensively for world maps in which the distortion towards
   the polar regions grows rather large, thus incorrectly giving the impression that, for example,
   Greenland is larger than South America. In reality, the latter is about eight times the size of
   Greenland. Also, the Former Soviet Union looks much bigger than Africa or South America. One may wonder
   whether this illusion has had any influence on U.S. foreign policy.' (Source: GMT)
   code from  Ernst-Jan Buis
   End_html */

   Al = l;
   Double_t aid = TMath::Tan((TMath::PiOver2() + b*TMath::DegToRad())/2);
   Ab = TMath::Log(aid);
   return 0;
}


//______________________________________________________________________________
Int_t THistPainter::ProjectSinusoidal2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   /* Begin_html
   Static function code from  Ernst-Jan Buis
   End_html */

   Al = l*cos(b*TMath::DegToRad());
   Ab = b;
   return 0;
}


//______________________________________________________________________________
Int_t THistPainter::ProjectParabolic2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   /* Begin_html
   Static function code from  Ernst-Jan Buis
   End_html */

   Al = l*(2.*TMath::Cos(2*b*TMath::DegToRad()/3) - 1);
   Ab = 180*TMath::Sin(b*TMath::DegToRad()/3);
   return 0;
}


//______________________________________________________________________________
void THistPainter::RecalculateRange()
{
   /* Begin_html
   Recompute the histogram range following graphics operations.
   End_html */

   if (Hoption.Same) return;

   //     Compute x,y range
   Double_t xmin = Hparam.xmin;
   Double_t xmax = Hparam.xmax;
   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;

   Double_t xmin_aid, ymin_aid, xmax_aid, ymax_aid;
   if (Hoption.Proj ==1) {
      // TODO : check x range not lower than -180 and not higher than 180
      THistPainter::ProjectAitoff2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectAitoff2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectAitoff2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectAitoff2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         // there is an  'equator', check its range in the plot..
         THistPainter::ProjectAitoff2xy(Hparam.xmin*0.9999, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectAitoff2xy(Hparam.xmax*0.9999, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectAitoff2xy(0, Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectAitoff2xy(0, Hparam.ymax, xmax_aid, ymax_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   } else if ( Hoption.Proj ==2) {
      if (Hparam.ymin <= -90 || Hparam.ymax >=90) {
         Warning("Mercator Projection", "Latitude out of range %f or %f", Hparam.ymin, Hparam.ymax);
         Hoption.Proj = 0;
      } else {
         THistPainter::ProjectMercator2xy(Hparam.xmin, Hparam.ymin, xmin, ymin);
         THistPainter::ProjectMercator2xy(Hparam.xmax, Hparam.ymax, xmax, ymax);
      }
   } else if (Hoption.Proj == 3) {
      THistPainter::ProjectSinusoidal2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         THistPainter::ProjectSinusoidal2xy(Hparam.xmin, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectSinusoidal2xy(Hparam.xmax, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectSinusoidal2xy(0,Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectSinusoidal2xy(0, Hparam.ymax, xmax_aid, ymin_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   } else if (Hoption.Proj == 4) {
      THistPainter::ProjectParabolic2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectParabolic2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectParabolic2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectParabolic2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         THistPainter::ProjectParabolic2xy(Hparam.xmin, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectParabolic2xy(Hparam.xmax, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectParabolic2xy(0, Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectParabolic2xy(0, Hparam.ymax, xmax_aid, ymin_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   }
   Hparam.xmin= xmin;
   Hparam.xmax= xmax;
   Hparam.ymin= ymin;
   Hparam.ymax= ymax;

   Double_t dx   = xmax-xmin;
   Double_t dy   = ymax-ymin;
   Double_t dxr  = dx/(1 - gPad->GetLeftMargin()   - gPad->GetRightMargin());
   Double_t dyr  = dy/(1 - gPad->GetBottomMargin() - gPad->GetTopMargin());

   // Range() could change the size of the pad pixmap and therefore should
   // be called before the other paint routines
   gPad->Range(xmin - dxr*gPad->GetLeftMargin(),
                      ymin - dyr*gPad->GetBottomMargin(),
                      xmax + dxr*gPad->GetRightMargin(),
                      ymax + dyr*gPad->GetTopMargin());
   gPad->RangeAxis(xmin, ymin, xmax, ymax);
}


//______________________________________________________________________________
void THistPainter::SetHistogram(TH1 *h)
{
   /* Begin_html
   Set current histogram to "h".
   End_html */

   if (h == 0)  return;
   fH = h;
   fXaxis = h->GetXaxis();
   fYaxis = h->GetYaxis();
   fZaxis = h->GetZaxis();
   fFunctions = fH->GetListOfFunctions();
}


//______________________________________________________________________________
Int_t THistPainter::TableInit()
{
   /* Begin_html
   Initialize various options to draw 2D histograms.
   End_html */

   static const char *where = "TableInit";

   Int_t first, last;
   Double_t yMARGIN= gStyle->GetHistTopMargin();
   Double_t zmin, zmax;
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //    -----------------  Compute X axis parameters
   first           = fXaxis->GetFirst();
   last            = fXaxis->GetLast();
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in X, replace xmin,max by the log
   if (Hoption.Logx) {
   //   find the first edge of a bin that is > 0
      if (Hparam.xlowedge <=0 ) {
         Hparam.xlowedge = fXaxis->GetBinUpEdge(fXaxis->FindFixBin(0.01*Hparam.xbinsize));
         Hparam.xmin  = Hparam.xlowedge;
      }
      if (Hparam.xmin <=0 || Hparam.xmax <=0) {
         Error(where, "cannot set X axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.xmin);
      if (Hparam.xfirst < first) Hparam.xfirst = first;
      Hparam.xlast = fXaxis->FindFixBin(Hparam.xmax);
      if (Hparam.xlast > last) Hparam.xlast = last;
      Hparam.xmin  = TMath::Log10(Hparam.xmin);
      Hparam.xmax  = TMath::Log10(Hparam.xmax);
   }

   //    -----------------  Compute Y axis parameters
   first           = fYaxis->GetFirst();
   last            = fYaxis->GetLast();
   Hparam.ylast    = last;
   Hparam.yfirst   = first;
   Hparam.ylowedge = fYaxis->GetBinLowEdge(first);
   Hparam.ybinsize = fYaxis->GetBinWidth(first);
   if (!Hparam.ybinsize) Hparam.ybinsize = 1;
   Hparam.ymin     = Hparam.ylowedge;
   Hparam.ymax     = fYaxis->GetBinLowEdge(last)+fYaxis->GetBinWidth(last);

   //       if log scale in Y, replace ymin,max by the log
   if (Hoption.Logy) {
      if (Hparam.ylowedge <=0 ) {
         Hparam.ylowedge = fYaxis->GetBinUpEdge(fYaxis->FindFixBin(0.01*Hparam.ybinsize));
         Hparam.ymin  = Hparam.ylowedge;
      }
      if (Hparam.ymin <=0 || Hparam.ymax <=0) {
         Error(where, "cannot set Y axis to log scale");
         return 0;
      }
      Hparam.yfirst= fYaxis->FindFixBin(Hparam.ymin);
      if (Hparam.yfirst < first) Hparam.yfirst = first;
      Hparam.ylast = fYaxis->FindFixBin(Hparam.ymax);
      if (Hparam.ylast > last) Hparam.ylast = last;
      Hparam.ymin  = TMath::Log10(Hparam.ymin);
      Hparam.ymax  = TMath::Log10(Hparam.ymax);
   }


   //    -----------------  Compute Z axis parameters
   Double_t bigp = TMath::Power(10,32);
   zmax = -bigp;
   zmin = bigp;
   Double_t c1, e1;
   Double_t allchan = 0;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         c1 = fH->GetCellContent(i,j);
         zmax = TMath::Max(zmax,c1);
         if (Hoption.Error) {
            e1 = fH->GetCellError(i,j);
            zmax = TMath::Max(zmax,c1+e1);
         }
         zmin = TMath::Min(zmin,c1);
         allchan += c1;
      }
   }

   //     Take into account maximum , minimum

   if (maximum) zmax = fH->GetMaximumStored();
   if (minimum) zmin = fH->GetMinimumStored();
   if (Hoption.Logz && zmax < 0) {
      if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
      return 0;
   } else if (Hoption.Logz && zmin>=0 && zmax==0) { // empty histogram in log scale
      zmin = 0.01;
      zmax = 10.;
   }
   if (zmin >= zmax) {
      if (Hoption.Logz) {
         if (zmax > 0) zmin = 0.001*zmax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
            return 0;
         }
      }
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   zmax = factor*zmax;
   zmin = factor*zmin;
   c1 = zmax;
   if (TMath::Abs(zmin) > TMath::Abs(c1)) c1 = zmin;

   //         For log scales, histogram coordinates are log10(ymin) and
   //         log10(ymax). Final adjustment (if not option "Same")
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logz) {
      if (zmin <= 0) {
         zmin = TMath::Min((Double_t)1, (Double_t)0.001*zmax);
         fH->SetMinimum(zmin);
      }
      zmin = TMath::Log10(zmin);
      if (!minimum) zmin += TMath::Log10(0.5);
      zmax = TMath::Log10(zmax);
      if (!maximum) zmax += TMath::Log10(2*(0.9/0.95));
      goto LZMIN;
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   //         MAX% is the value in percent which has been set in HPLSET
   //         (default is 90%).
   if (!maximum) {
      zmax += yMARGIN*(zmax-zmin);
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - yMARGIN if <0.
   if (!minimum) {
      if (gStyle->GetHistMinimumZero()) {
         if (zmin >= 0) zmin = 0;
         else           zmin -= yMARGIN*(zmax-zmin);
      } else {
         Double_t dzmin = yMARGIN*(zmax-zmin);
         if (zmin >= 0 && (zmin-dzmin <= 0)) zmin  = 0;
         else                                zmin -= dzmin;
      }
   }

LZMIN:
   Hparam.zmin = zmin;
   Hparam.zmax = zmax;

   //     Set bar offset and width
   Hparam.baroffset = fH->GetBarOffset();
   Hparam.barwidth  = fH->GetBarWidth();

   return 1;
}


//______________________________________________________________________________
const char * THistPainter::GetBestFormat(Double_t v, Double_t e, const char *f)
{
   /* Begin_html
   This function returns the best format to print the error value (e)
   knowing the parameter value (v) and the format (f) used to print it.
   End_html */

   static char ef[20];
   char tf[20], tv[64];

   // print v with the format f in tv.
   snprintf(tf,20,"%s%s","%",f);
   snprintf(tv,64,tf,v);

   // Analyse tv.
   TString sv = tv;
   int ie = sv.Index("e");
   int iE = sv.Index("E");
   int id = sv.Index(".");

   // v has been printed with the exponent notation.
   // There is 2 cases, the exponent is positive or negative
   if (ie >= 0 || iE >= 0) {
      if (sv.Index("+") >= 0) {
         if (e < 1) {
            snprintf(ef,20,"%s.1f","%");
         } else {
            if (ie >= 0) {
               snprintf(ef,20,"%s.%de","%",ie-id-1);
            } else {
               snprintf(ef,20,"%s.%dE","%",iE-id-1);
            }
         }
      } else {
         if (ie >= 0) {
            snprintf(ef,20,"%s.%de","%",ie-id-1);
         } else {
            snprintf(ef,20,"%s.%dE","%",iE-id-1);
         }
      }

   // There is not '.' in tv. e will be printed with one decimal digit.
   } else if (id < 0) {
      snprintf(ef,20,"%s.1f","%");

   // There is a '.' in tv and no exponent notation. e's decimal part will
   // have the same number of digits as v's one.
   } else {
      snprintf(ef,20,"%s.%df","%",sv.Length()-id-1);
   }

   return ef;
}


//______________________________________________________________________________
void THistPainter::SetShowProjection(const char *option,Int_t nbins)
{
   /* Begin_html
   Set projection.
   End_html */

   if (fShowProjection) return;
   TString opt = option;
   opt.ToLower();
   Int_t projection = 0;
   if (opt.Contains("x"))  projection = 1;
   if (opt.Contains("y"))  projection = 2;
   if (opt.Contains("z"))  projection = 3;
   if (opt.Contains("xy")) projection = 4;
   if (opt.Contains("yx")) projection = 5;
   if (opt.Contains("xz")) projection = 6;
   if (opt.Contains("zx")) projection = 7;
   if (opt.Contains("yz")) projection = 8;
   if (opt.Contains("zy")) projection = 9;
   if (projection < 4) fShowOption = option+1;
   else                fShowOption = option+2;
   fShowProjection = projection+100*nbins;
   gROOT->MakeDefCanvas();
   gPad->SetName(Form("c_%lx_projection_%d", (ULong_t)fH, fShowProjection));
   gPad->SetGrid();
}


//______________________________________________________________________________
void THistPainter::ShowProjectionX(Int_t /*px*/, Int_t py)
{
   /* Begin_html
   Show projection onto X.
   End_html */

   Int_t nbins = (Int_t)fShowProjection/100;
   gPad->SetDoubleBuffer(0); // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert); // set the drawing mode to XOR mode

   // Erase old position and draw a line at current position
   static int pyold1 = 0;
   static int pyold2 = 0;
   float uxmin = gPad->GetUxmin();
   float uxmax = gPad->GetUxmax();
   int pxmin   = gPad->XtoAbsPixel(uxmin);
   int pxmax   = gPad->XtoAbsPixel(uxmax);
   Float_t upy = gPad->AbsPixeltoY(py);
   Float_t y   = gPad->PadtoY(upy);
   Int_t biny1 = fH->GetYaxis()->FindBin(y);
   Int_t biny2 = TMath::Min(biny1+nbins-1, fH->GetYaxis()->GetNbins());
   Int_t py1   = gPad->YtoAbsPixel(fH->GetYaxis()->GetBinLowEdge(biny1));
   Int_t py2   = gPad->YtoAbsPixel(fH->GetYaxis()->GetBinUpEdge(biny2));

   if (pyold1 || pyold2) gVirtualX->DrawBox(pxmin,pyold1,pxmax,pyold2,TVirtualX::kFilled);
   gVirtualX->DrawBox(pxmin,py1,pxmax,py2,TVirtualX::kFilled);
   pyold1 = py1;
   pyold2 = py2;

   // Create or set the new canvas proj x
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_%lx_projection_%d",
                                                                              (ULong_t)fH, fShowProjection));
   if (c) {
      c->Clear();
   } else {
      fShowProjection = 0;
      pyold1 = 0;
      pyold2 = 0;
      return;
   }
   c->cd();
   c->SetLogy(padsav->GetLogz());
   c->SetLogx(padsav->GetLogx());

   // Draw slice corresponding to mouse position
   TH1D *hp = ((TH2*)fH)->ProjectionX("_px", biny1, biny2);
   if (hp) {
      hp->SetFillColor(38);
      if (biny1 == biny2) hp->SetTitle(Form("ProjectionX of biny=%d", biny1));
      else hp->SetTitle(Form("ProjectionX of biny=[%d,%d]", biny1,biny2));
      hp->SetXTitle(fH->GetXaxis()->GetTitle());
      hp->SetYTitle("Number of Entries");
      hp->Draw();
      c->Update();
      padsav->cd();
   }
}


//______________________________________________________________________________
void THistPainter::ShowProjectionY(Int_t px, Int_t /*py*/)
{
   /* Begin_html
   Show projection onto Y.
   End_html */

   Int_t nbins = (Int_t)fShowProjection/100;
   gPad->SetDoubleBuffer(0);             // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode

   // Erase old position and draw a line at current position
   static int pxold1 = 0;
   static int pxold2 = 0;
   float uymin = gPad->GetUymin();
   float uymax = gPad->GetUymax();
   int pymin   = gPad->YtoAbsPixel(uymin);
   int pymax   = gPad->YtoAbsPixel(uymax);
   Float_t upx = gPad->AbsPixeltoX(px);
   Float_t x   = gPad->PadtoX(upx);
   Int_t binx1 = fH->GetXaxis()->FindBin(x);
   Int_t binx2 = TMath::Min(binx1+nbins-1, fH->GetXaxis()->GetNbins());
   Int_t px1   = gPad->XtoAbsPixel(fH->GetXaxis()->GetBinLowEdge(binx1));
   Int_t px2   = gPad->XtoAbsPixel(fH->GetXaxis()->GetBinUpEdge(binx2));

   if (pxold1 || pxold2) gVirtualX->DrawBox(pxold1,pymin,pxold2,pymax,TVirtualX::kFilled);
   gVirtualX->DrawBox(px1,pymin,px2,pymax,TVirtualX::kFilled);
   pxold1 = px1;
   pxold2 = px2;

   // Create or set the new canvas proj y
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_%lx_projection_%d",
                                                                              (ULong_t)fH, fShowProjection));
   if(c) {
      c->Clear();
   } else {
      fShowProjection = 0;
      pxold1 = 0;
      pxold2 = 0;
      return;
   }
   c->cd();
   c->SetLogy(padsav->GetLogz());
   c->SetLogx(padsav->GetLogy());

   // Draw slice corresponding to mouse position
   TH1D *hp = ((TH2*)fH)->ProjectionY("_py", binx1, binx2);
   if (hp) {
      hp->SetFillColor(38);
      if (binx1 == binx2) hp->SetTitle(Form("ProjectionY of binx=%d", binx1));
      else hp->SetTitle(Form("ProjectionY of binx=[%d,%d]", binx1,binx2));
      hp->SetXTitle(fH->GetYaxis()->GetTitle());
      hp->SetYTitle("Number of Entries");
      hp->Draw();
      c->Update();
      padsav->cd();
   }
}


//______________________________________________________________________________
void THistPainter::ShowProjection3(Int_t px, Int_t py)
{
   /* Begin_html
   Show projection (specified by <tt>fShowProjection</tt>) of a <tt>TH3</tt>.
   The drawing option for the projection is in <tt>fShowOption</tt>.
   <p>
   First implementation; R.Brun <br>
   Full implementation: Tim Tran (timtran@jlab.org)  April 2006
   End_html */

   Int_t nbins=(Int_t)fShowProjection/100; //decode nbins
   if (fH->GetDimension() < 3) {
      if (fShowProjection%100 == 1) {ShowProjectionX(px,py); return;}
      if (fShowProjection%100 == 2) {ShowProjectionY(px,py); return;}
   }

   gPad->SetDoubleBuffer(0);             // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode

   // Erase old position and draw a line at current position
   TView *view = gPad->GetView();
   if (!view) return;
   TH3 *h3 = (TH3*)fH;
   TAxis *xaxis = h3->GetXaxis();
   TAxis *yaxis = h3->GetYaxis();
   TAxis *zaxis = h3->GetZaxis();
   Double_t u[3],xx[3];

   static TPoint line1[2];//store end points of a line, initialised 0 by default
   static TPoint line2[2];// second line when slice thickness > 1 bin thickness
   static TPoint line3[2];
   static TPoint line4[2];
   static TPoint endface1[5];
   static TPoint endface2[5];
   static TPoint rect1[5];//store vertices of the polyline (rectangle), initialsed 0 by default
   static TPoint rect2[5];// second rectangle when slice thickness > 1 bin thickness

   Double_t value1=0, value2=0; //bin values cooresponding to the lower and upper bins of the slice
   Double_t uxmin = gPad->GetUxmin();
   Double_t uxmax = gPad->GetUxmax();
   Double_t uymin = gPad->GetUymin();
   Double_t uymax = gPad->GetUymax();

   int pxmin = gPad->XtoAbsPixel(uxmin);
   int pxmax = gPad->XtoAbsPixel(uxmax);
   int pymin = gPad->YtoAbsPixel(uymin);
   int pymax = gPad->YtoAbsPixel(uymax);
   Double_t cx    = (pxmax-pxmin)/(uxmax-uxmin);
   Double_t cy    = (pymax-pymin)/(uymax-uymin);
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_%lx_projection_%d",
                                                                              (ULong_t)fH, fShowProjection));
   if(!c) {
      fShowProjection = 0;
      return;
   }

   switch ((Int_t)fShowProjection%100) {
      case 1:
         // "x"
         {
            Int_t firstY = yaxis->GetFirst();
            Int_t lastY  = yaxis->GetLast();
            Int_t biny = firstY + Int_t((lastY-firstY)*(px-pxmin)/(pxmax-pxmin));
            yaxis->SetRange(biny,biny+nbins-1);
            Int_t firstZ = zaxis->GetFirst();
            Int_t lastZ  = zaxis->GetLast();
            Int_t binz = firstZ + Int_t((lastZ-firstZ)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0] = xaxis->GetXmin();
            xx[2] = zaxis->GetBinCenter(binz);
            xx[1] = yaxis->GetBinCenter(biny);
            view->WCtoNDC(xx,u);
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[0] = xaxis->GetXmax();
            view->WCtoNDC(xx,u);
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               xx[1] = yaxis->GetBinCenter(biny);
               view->WCtoNDC(xx,u);
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               view->WCtoNDC(xx,u);
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz);
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               view->WCtoNDC(xx,u);
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());

               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());

               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            c->Clear();
            c->cd();
            TH1 *hp = h3->Project3D("x");
            yaxis->SetRange(firstY,lastY);
            zaxis->SetRange(firstZ,lastZ);
            if (hp) {
               hp->SetFillColor(38);
               hp->SetTitle(Form("ProjectionX of biny=%d binz=%d", biny, binz));
               hp->SetXTitle(fH->GetXaxis()->GetTitle());
               hp->SetYTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 2:
         // "y"
         {
            Int_t firstX = xaxis->GetFirst();
            Int_t lastX  = xaxis->GetLast();
            Int_t binx = firstX + Int_t((lastX-firstX)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            Int_t firstZ = zaxis->GetFirst();
            Int_t lastZ  = zaxis->GetLast();
            Int_t binz = firstZ + Int_t((lastZ-firstZ)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0]=xaxis->GetBinCenter(binx);
            xx[2] = zaxis->GetBinCenter(binz);
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[1] = yaxis->GetXmax();
            view->WCtoNDC(xx,u);
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1) {
               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               xx[0] = xaxis->GetBinCenter(binx);
               view->WCtoNDC(xx,u);
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               view->WCtoNDC(xx,u);
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz);
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               view->WCtoNDC(xx,u);
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());

               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());

               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            c->Clear();
            c->cd();
            TH1 *hp = h3->Project3D("y");
            xaxis->SetRange(firstX,lastX);
            zaxis->SetRange(firstZ,lastZ);
            if (hp) {
               hp->SetFillColor(38);
               hp->SetTitle(Form("ProjectionY of binx=%d binz=%d", binx, binz));
               hp->SetXTitle(fH->GetYaxis()->GetTitle());
               hp->SetYTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 3:
         // "z"
         {
            Int_t firstX = xaxis->GetFirst();
            Int_t lastX  = xaxis->GetLast();
            Int_t binx = firstX + Int_t((lastX-firstX)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            Int_t firstY = yaxis->GetFirst();
            Int_t lastY  = yaxis->GetLast();
            Int_t biny = firstY + Int_t((lastY-firstY)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0] = xaxis->GetBinCenter(binx);
            xx[1] = yaxis->GetBinCenter(biny);
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[2] = zaxis->GetXmax();
            view->WCtoNDC(xx,u);
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1) {
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               xx[0] = xaxis->GetBinCenter(binx);
               view->WCtoNDC(xx,u);
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               view->WCtoNDC(xx,u);
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny);
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               view->WCtoNDC(xx,u);
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmax();
               view->WCtoNDC(xx,u);
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));

               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());

               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());

               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            c->Clear();
            c->cd();
            TH1 *hp = h3->Project3D("z");
            xaxis->SetRange(firstX,lastX);
            yaxis->SetRange(firstY,lastY);
            if (hp) {
               hp->SetFillColor(38);
               hp->SetTitle(Form("ProjectionZ of binx=%d biny=%d", binx, biny));
               hp->SetXTitle(fH->GetZaxis()->GetTitle());
               hp->SetYTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 4:
         // "xy"
         {
            Int_t first = zaxis->GetFirst();
            Int_t last  = zaxis->GetLast();
            Int_t binz  = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect2[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[1] = yaxis->GetXmax();
            xx[2] = zaxis->GetBinCenter(binz);
            value1=xx[2]; // for screen display
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[0] = xaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[1] = yaxis->GetXmax();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               value2=xx[2];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }

            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("xy");
            zaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionXY of binz=%d (%.1f)", binz,value1));
               else        hp->SetTitle(Form("ProjectionXY, binz range=%d-%d (%.1f-%.1f)", binz,binz+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetYaxis()->GetTitle());
               hp->SetYTitle(fH->GetXaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 5:
         // "yx"
         {
            Int_t first = zaxis->GetFirst();
            Int_t last  = zaxis->GetLast();
            Int_t binz = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect2[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[1] = yaxis->GetXmax();
            xx[2] = zaxis->GetBinCenter(binz);
            value1=xx[2]; // for screen display
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[0] = xaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[1] = yaxis->GetXmax();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               value2=xx[2];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("yx");
            zaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionYX of binz=%d (%.1f)", binz,value1));
               else        hp->SetTitle(Form("ProjectionXY, binz range=%d-%d (%.1f-%.1f)", binz,binz+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetXaxis()->GetTitle());
               hp->SetYTitle(fH->GetYaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 6:
         // "xz"
         {
            Int_t first = yaxis->GetFirst();
            Int_t last  = yaxis->GetLast();
            Int_t biny = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[2] = zaxis->GetXmax();
            xx[1] = yaxis->GetBinCenter(biny);
            value1=xx[1];
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[0] = xaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetXmax();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               value2=xx[1];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("xz");
            yaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionXZ of biny=%d (%.1f)", biny,value1));
               else        hp->SetTitle(Form("ProjectionXZ, biny range=%d-%d (%.1f-%.1f)", biny,biny+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetZaxis()->GetTitle());
               hp->SetYTitle(fH->GetXaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 7:
         // "zx"
         {
            Int_t first = yaxis->GetFirst();
            Int_t last  = yaxis->GetLast();
            Int_t biny = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[2] = zaxis->GetXmax();
            xx[1] = yaxis->GetBinCenter(biny);
            value1=xx[1];
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[0] = xaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetXmax();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               value2=xx[1];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[0] = xaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("zx");
            yaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionZX of biny=%d (%.1f)", biny,value1));
               else        hp->SetTitle(Form("ProjectionZX, binY range=%d-%d (%.1f-%.1f)", biny,biny+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetXaxis()->GetTitle());
               hp->SetYTitle(fH->GetZaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 8:
         // "yz"
         {
            Int_t first = xaxis->GetFirst();
            Int_t last  = xaxis->GetLast();
            Int_t binx = first + Int_t((last-first)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            if(rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[2] = zaxis->GetXmin();
            xx[1] = yaxis->GetXmax();
            xx[0] = xaxis->GetBinCenter(binx);
            value1=xx[0];
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[2] = zaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetXmax();
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               value2=xx[0];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[2] = zaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("yz");
            xaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionYZ of binx=%d (%.1f)", binx,value1));
               else        hp->SetTitle(Form("ProjectionYZ, binx range=%d-%d (%.1f-%.1f)", binx,binx+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetZaxis()->GetTitle());
               hp->SetYTitle(fH->GetYaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

      case 9:
         // "zy"
         {
            Int_t first = xaxis->GetFirst();
            Int_t last  = xaxis->GetLast();
            Int_t binx = first + Int_t((last-first)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            if(rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[2] = zaxis->GetXmin();
            xx[1] = yaxis->GetXmax();
            xx[0] = xaxis->GetBinCenter(binx);
            value1=xx[0];
            view->WCtoNDC(xx,u);
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());
            xx[2] = zaxis->GetXmax();
            view->WCtoNDC(xx,u);
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetXmax();
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               value2=xx[0];
               view->WCtoNDC(xx,u);
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());
               xx[2] = zaxis->GetXmax();
               view->WCtoNDC(xx,u);
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("zy");
            xaxis->SetRange(first,last);
            if (hp) {
               hp->SetFillColor(38);
               if(nbins==1)hp->SetTitle(Form("ProjectionZY of binx=%d (%.1f)", binx,value1));
               else        hp->SetTitle(Form("ProjectionZY, binx range=%d-%d (%.1f-%.1f)", binx,binx+nbins-1,value1,value2));
               hp->SetXTitle(fH->GetYaxis()->GetTitle());
               hp->SetYTitle(fH->GetZaxis()->GetTitle());
               hp->SetZTitle("Number of Entries");
               hp->Draw(fShowOption.Data());
            }
         }
         break;

   }
   c->Update();
   padsav->cd();
}
