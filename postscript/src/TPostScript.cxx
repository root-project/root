// @(#)root/postscript:$Name:  $:$Id: TPostScript.cxx,v 1.10 2001/06/07 08:39:45 brun Exp $
// Author: Rene Brun, Olivier Couet, Pierre Juillot   29/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifdef WIN32
#pragma optimize("",off)
#endif


#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fstream.h>

#include "TROOT.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TPoints.h"
#include "TPostScript.h"
#include "TStyle.h"
#include "TMath.h"

char backslash = '\\';

const Int_t  kMaxBuffer = 250;
const Int_t  kLatex = BIT(10);

ClassImp(TPostScript)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*The  P O S T S C R I P T  class-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//*-*
//*-*   Graphics interface to PostScript.
//*-*
//*-*  This code was initially developped in the context of HIGZ and PAW
//*-*  by Olivier Couet and Pierre Juillot.
//*-*  It has been converted to a C++ class by Rene Brun.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


//______________________________________________________________________________
TPostScript::TPostScript() : TVirtualPS()
{
//*-*-*-*-*-*-*-*-*-*-*Default PostScript constructor*-*-*-*-*-*-*-*-*-*-*-*-*

   fStream = 0;
   fType   = 0;
   gVirtualPS = this;
}

//______________________________________________________________________________
TPostScript::TPostScript(const char *fname, Int_t wtype)
            :TVirtualPS(fname, wtype)
{
//*-*-*-*-*-*-*-*-*-*-*Initialize the PostScript interface*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//*-*
//*-*_Input parameters:
//*-*
//*-*  fname : PostScript file name
//*-*  wtype : PostScript workstation type
//*-*
//*-*
//*-*  The possible workstation types are:
//*-*     111 ps  Portrait
//*-*     112 ps  Landscape
//*-*     113 eps
//*-*
//BEGIN_HTML
/*

<P>To generate a Postscript (or encapsulated ps) file corresponding to
a single image in a canvas, you can: </P>

<UL>
<LI>Select the <B>Print PostScript</B> item in the canvas <B>File</B> menu.
By default, a Postscript file with the name of the canvas.ps is generated.
</LI>
<br>
<LI>Click in the canvas area, near the edges, with the right mouse button
and select the <B>Print</B> item. You can select the name of the Postscript
file. If the file name is xxx.ps, you will generate a Postscript file named
xxx.ps. If the file name is xxx.eps, you generate an encapsulated Postscript
file instead.  </LI>
<br>

<LI>In your program (or macro), you can type:

<PRE> <B>c1-&gt;Print(&quot;xxx.ps&quot;)</B> or <B>c1-&gt;Print(&quot;xxx.eps&quot;)</B></PRE>

<P>This will generate a file corresponding to the picture in the canvas
pointed by <B>c1</B>. </P> </LI>

<PRE> <B>pad1-&gt;Print(&quot;xxx.ps&quot;)</B></PRE>

<P>prints only the picture in the pad pointed by <B>pad1</B>. The size
of the Postcript picture, by default, is computed to keep the aspect ratio
of the picture on the screen, where the size along x is always 20cm. You
can set the size of the PostScript picture before generating the picture
with a command such as: </P>

<PRE>
   <A HREF="html/TPostScript.html">TPostScript</A> myps(&quot;myfile.ps&quot;,111)
   myps.Range(xsize,ysize);
   object-&gt;Draw();
   myps.Close();
</PRE>

<P>You can set the default paper size with:
<PRE>
   <A HREF="html/TStyle.html">gStyle</A>-&gt;<A HREF="html/TStyle.html#TStyle:SetPaperSize">SetPaperSize</A>(xsize,ysize);
</PRE>
<P>You can resume writing again in this file with <B>myps.Open();</B>.
Note that you may have several Postscript files opened simultaneously.
</P>
</UL>

<H2>Special characters</H2>
The following characters have a special action on the Postscript file:
<PRE>
       `   : go to greek
       '   : go to special
       ~   : go to ZapfDingbats
       ?   : go to subscript
       ^   : go to superscript
       !   : go to normal level of script
       &   : backspace one character
       #   : end of greek or of ZapfDingbats
</PRE>
<P>These special characters are printed as such on the screen.
To generate one of these characters on the Postscript file, you must escape it
with the escape character "@".
<P>
The use of these special characters is illustrated in several macros
referenced by the <A HREF="html/TPostScript.html#TPostScript:TPostScript">TPostScript constructor</A>.

<H2>Making several pictures in the same Postscript file: case 1</H2>
<P>The following macro is an example illustrating how to open a Postscript
file and draw several pictures. The generation of a new Postscript page
is automatic when <B>TCanvas::Clear</B> is called by <b>object-&gt;Draw()</b>.
<PRE>
{
   TFile f(&quot;hsimple.root&quot;);
   TCanvas c1(&quot;c1&quot;,&quot;canvas&quot;,800,600);

     //select postscript output type
   Int_t type = 111;   //portrait  ps
// Int_t type = 112;   //landscape ps
// Int_t type = 113;   //eps

    //create a postscript file and set the paper size
   <A HREF="html/TPostScript.html">TPostScript</A> ps(&quot;test.ps&quot;,type);
   ps.Range(16,24);  //set x,y of printed page

    //draw 3 histograms from file hsimple.root on separate pages
   hpx-&gt;Draw();
   c1.Update();      //force drawing in a macro
   hprof-&gt;Draw();
   c1.Update();
   hpx-&gt;Draw(&quot;lego1&quot;);
   c1.Update();
   ps.Close();
}
</PRE>

<H2>Making several pictures in the same Postscript file: case 2</H2>
<P>This example shows 2 pages. The canvas is divided.
<B>TPostScript::NewPage</B> must be called before starting a new picture.
<b>object-&gt;Draw</b> does not clear the canvas in this case
because we clear only the pads and not the main canvas.
Note that <b>c1-&gt;Update</b> must be called at the end of the first picture
<pre>
{
   TFile *f1 = new TFile("hsimple.root");
   TCanvas *c1 = new TCanvas("c1");
   TPostScript *ps = new TPostScript("file.ps",112);
   c1-&gt;Divide(2,1);
<b>// picture 1</b>
   ps-&gt;NewPage();
   c1-&gt;cd(1);
   hpx-&gt;Draw();
   c1-&gt;cd(2);
   hprof-&gt;Draw();
   c1-&gt;Update();

<b>// picture 2</b>
   ps-&gt;NewPage();
   c1-&gt;cd(1);
   hpxpy-&gt;Draw();
   c1-&gt;cd(2);
   ntuple-&gt;Draw("px");
   c1-&gt;Update();
   ps-&gt;Close();

<b>// invoke Postscript viewer</b>
   gSystem-&gt;Exec("gs file.ps");
}   
</pre>
*/
//END_HTML
//*-*  The picture below shows fancy text with national accents or
//*-*  subscripts and superscripts. This picture has been generated by
//*-*  the macro Begin_Html <a href=examples/psexam.C.html>psexam</a>. End_Html
//
//Begin_Html <img src="gif/psexam.gif"> End_Html
//
//     The two following tables list the correspondance between the typed
//     character and its interpretation using the special characters given
//     in TPostScript::Text. These tables are screen copies. True and better
//     resolution PostScript files can be seen at Begin_Html <a href=ps/psexam.ps>psexam</a>, <a href=ps/pstable1.ps>pstable1</a> and <a href=ps/pstable2.ps>pstable2</a>. End_Html
// The macro Begin_Html <a href=examples/pstable.C.html>pstable</a> End_Html has been used to generate the two PostScript tables.
//
//Begin_Html <img src="gif/pstable1.gif"> End_Html
//
//
//Begin_Html <img src="gif/pstable2.gif"> End_Html
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*

   fStream = 0;
   Open(fname, wtype);
}

//______________________________________________________________________________
void TPostScript::Open(const char *fname, Int_t wtype)
{
//*-*-*-*-*-*-*-*-*-*-*Open a PostScript file*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  if (fStream) {
     Warning("Open", "postscript file already open");
     return;
  }

  fMarkerSizeCur = 0;
  fCurrentColor  = 0;
  fRed         = -1;
  fGreen       = -1;
  fBlue        = -1;
  fLenBuffer   = 0;
  fClip        = 0;
  fType        = abs(wtype);
  fClear       = kTRUE;
  fZone        = kFALSE;
  fSave        = 0;
  SetLineScale(gStyle->GetLineScalePS());
  gStyle->GetPaperSize(fXsize, fYsize);
  fMode        = fType%10;
  Float_t xrange, yrange;
  if (gPad) {
     Double_t ww    = gPad->GetWw();
     Double_t wh    = gPad->GetWh();
     if (fType == 113) {
        ww *= gPad->GetWNDC();
        wh *= gPad->GetHNDC();
     }
     Double_t ratio = wh/ww;
     if (fType == 112) {
        xrange = fYsize;
        yrange = xrange*ratio;
        if (yrange > fXsize) { yrange = fXsize; xrange = yrange/ratio;}
     } else {
        xrange = fXsize;
        yrange = fXsize*ratio;
        if (yrange > fYsize) { yrange = fYsize; xrange = yrange/ratio;}
     }
     fXsize = xrange; fYsize = yrange;
  }

//*-*- open OS file
  fStream   = new ofstream(fname,ios::out);
  if (fStream == 0) {
     printf("ERROR in TPostScript::Open: Cannot open file:%s\n",fname);
     return;
  }
  gVirtualPS = this;

  for (Int_t i=0;i<512;i++) fBuffer[i] = ' ';
  if( fType == 113) {
     fBoundingBox = kFALSE;
     PrintStr("%!PS-Adobe-2.0 EPSF-2.0@");
  }
  else {
     fBoundingBox = kTRUE;
     PrintStr("%!PS-Adobe-2.0@");
     Initialize();
  }

  fClipStatus  = kFALSE;
  fRange       = kFALSE;

//*-*- Set a default range
  Range(fXsize, fYsize);

  fPrinted     = kFALSE;
  if (fType == 113) NewPage();
}

//______________________________________________________________________________
TPostScript::~TPostScript()
{
//*-*-*-*-*-*-*-*-*-*-*Default PostScript destructor*-*-*-*-*-*-*-*-*-*-*-*-*

   Close();
}

//______________________________________________________________________________
void TPostScript::Close(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Close a PostScript file*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

   if (!gVirtualPS) return;
   if (gPad) gPad->Update();
   if( fMode != 3) {
     SaveRestore(-1);
     if( fPrinted ) { PrintStr("showpage gr@"); SaveRestore(-1);}
     PrintStr("%%Trailer@");
     PrintStr("%%Pages: ");
     WriteInteger(fNpages);
     PrintStr("@");
     while (fSave > 0) { SaveRestore(-1); }
  }
  else {
     PrintStr("@");
     while (fSave > 0) { SaveRestore(-1); }
     PrintStr("showpage@");
     PrintStr("end@");
  }
  PrintStr("%%EOF@");

//*-*- Close file stream

   if (fStream) { fStream->close(); delete fStream; fStream = 0;}

   gVirtualPS = 0;
}

//______________________________________________________________________________
void TPostScript::On()
{
//*-*-*-*-*-*-*-*-*-*-*Activate an already open PostScript file*-*-*-*-*-*
//*-*                  ========================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*

   if (!fType) {
      Error("On", "no postscript file open");
      Off();
      return;
   }
   gVirtualPS = this;
}

//______________________________________________________________________________
void TPostScript::Off()
{
//*-*-*-*-*-*-*-*-*-*-*DeActivate an already open PostScript file*-*-*-*-*-*
//*-*                  ==========================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*

   gVirtualPS = 0;
}


//______________________________________________________________________________
void TPostScript::DefineMarkers()
{
//*-*-*-*-*-*-*-*-*-*-*Define the markers*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================
//*-*==========> (O.Couet)
  PrintStr("/mp {newpath /y exch def /x exch def} def@");
  PrintStr("/side {[w .77 mul w .23 mul] .385 w mul sd w 0 l currentpoint t -144 r} def@");
  PrintStr("/mr {mp x y w2 0 360 arc} def /m24 {mr s} def /m20 {mr f} def@");
  PrintStr("/mb {mp x y w2 add m w2 neg 0 d 0 w neg d w 0 d 0 w d cl} def@");
  PrintStr("/mt {mp x y w2 add m w2 neg w neg d w 0 d cl} def@");
  PrintStr("/m21 {mb f} def /m25 {mb s} def /m22 {mt f} def /m26{mt s} def@");
  PrintStr("/m23 {mp x y w2 sub m w2 w d w neg 0 d cl f} def@");
  PrintStr("/m27 {mp x y w2 add m w3 neg w2 neg d w3 w2 neg d w3 w2 d cl s} def@");
  PrintStr("/m28 {mp x w2 sub y w2 sub w3 add m w3 0 d ");
  PrintStr(" 0 w3 neg d w3 0 d 0 w3 d w3 0 d ");
  PrintStr(" 0 w3 d w3 neg 0 d 0 w3 d w3 neg 0 d");
  PrintStr(" 0 w3 neg d w3 neg 0 d cl s } def@");
  PrintStr("/m29 {mp gsave x w2 sub y w2 add w3 sub m currentpoint t");
  PrintStr(" 4 {side} repeat cl fill gr} def@");
  PrintStr("/m30 {mp gsave x w2 sub y w2 add w3 sub m currentpoint t");
  PrintStr(" 4 {side} repeat cl s gr} def@");
  PrintStr("/m31 {mp x y w2 sub m 0 w d x w2 sub y m w 0 d");
  PrintStr(" x w2 sub y w2 add m w w neg d x w2 sub y w2");
  PrintStr(" sub m w w d s} def@");
  PrintStr("/m2 {mp x y w2 sub m 0 w d x w2 sub y m w 0 d s} def@");
  PrintStr("/m5 {mp x w2 sub y w2 sub m w w d x w2 sub y w2 add m w w neg d s} def@");

}

//______________________________________________________________________________
void TPostScript::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Draw a Box*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ==========
//*-*
   static Double_t x[4], y[4];
   Int_t ix1 = XtoPS(x1);
   Int_t ix2 = XtoPS(x2);
   Int_t iy1 = YtoPS(y1);
   Int_t iy2 = YtoPS(y2);
   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

   if (fillis == 3 || fillis == 2) {
      if (fillsi > 99) {
         x[0] = x1;   y[0] = y1;
         x[1] = x2;   y[1] = y1;
         x[2] = x2;   y[2] = y2;
         x[3] = x1;   y[3] = y2;
         return;
      }
      if (fillsi > 0 && fillsi < 26) {
         x[0] = x1;   y[0] = y1;
         x[1] = x2;   y[1] = y1;
         x[2] = x2;   y[2] = y2;
         x[3] = x1;   y[3] = y2;
         DrawPS(-4, &x[0], &y[0]);
      }
      if (fillsi == -3) {
         SetColor(5);
         WriteInteger(ix2 - ix1);
         WriteInteger(iy2 - iy1);
         WriteInteger(ix1);
         WriteInteger(iy1);
         PrintFast(3," bf");
      }
   }
   if (fillis == 1) {
      SetColor(fFillColor);
      WriteInteger(ix2 - ix1);
      WriteInteger(iy2 - iy1);
      WriteInteger(ix1);
      WriteInteger(iy1);
      PrintFast(3," bf");
   }
   if (fillis == 0) {
      SetColor(fLineColor);
      WriteInteger(ix2 - ix1);
      WriteInteger(iy2 - iy1);
      WriteInteger(ix1);
      WriteInteger(iy1);
      PrintFast(3," bl");
   }
}

//______________________________________________________________________________
void TPostScript::DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                            Int_t mode, Int_t border, Int_t dark, Int_t light)
{
//*-*-*-*-*-*-*-*-*-*-*Draw a Frame around a box*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  mode = -1  box looks as it is behind the screen
//*-*  mode =  1  box looks as it is in front of the screen
//*-*  border is the bordersize in already precomputed PostScript units
//*-*  dark  is the color for the dark part of the frame
//*-*  light is the color for the light part of the frame
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  static Int_t xps[7], yps[7];
  Int_t i, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;

//*-*- Draw top&left part of the box
  if (mode == -1) SetColor(dark);
  else            SetColor(light);
  xps[0] = XtoPS(xl);          yps[0] = YtoPS(yl);
  xps[1] = xps[0] + border;    yps[1] = yps[0] + border;
  xps[2] = xps[1];             yps[2] = YtoPS(yt) - border;
  xps[3] = XtoPS(xt) - border; yps[3] = yps[2];
  xps[4] = XtoPS(xt);          yps[4] = YtoPS(yt);
  xps[5] = xps[0];             yps[5] = yps[4];
  xps[6] = xps[0];             yps[6] = yps[0];

  ixd0 = xps[0];
  iyd0 = yps[0];
  WriteInteger(ixd0);
  WriteInteger(iyd0);

  PrintFast(2," m");
  idx = 0;
  idy = 0;
  for (i=1;i<7;i++) {
     ixdi = xps[i];
     iydi = yps[i];
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
        continue;
     }
     if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0; }
        if( !idx ) { idx = ix; continue;}
        if( ix*idx > 0 )       idx += ix;
        else { MovePS(idx,0);  idx  = ix; }
        continue;
     }
     if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( !idy) { idy = iy; continue;}
        if( iy*idy > 0 )         idy += iy;
        else { MovePS(0,idy);    idy  = iy; }
     }
  }
  if( idx ) MovePS(idx,0);
  if( idy ) MovePS(0,idy);
  PrintFast(2," f");

//*-*- Draw bottom&right part of the box
  if (mode == -1) SetColor(light);
  else            SetColor(dark);
  xps[0] = XtoPS(xl);          yps[0] = YtoPS(yl);
  xps[1] = xps[0] + border;    yps[1] = yps[0] + border;
  xps[2] = XtoPS(xt) - border; yps[2] = yps[1];
  xps[3] = xps[2];             yps[3] = YtoPS(yt) - border;
  xps[4] = XtoPS(xt);          yps[4] = YtoPS(yt);
  xps[5] = xps[4];             yps[5] = yps[0];
  xps[6] = xps[0];             yps[6] = yps[0];

  ixd0 = xps[0];
  iyd0 = yps[0];
  WriteInteger(ixd0);
  WriteInteger(iyd0);

  PrintFast(2," m");
  idx = 0;
  idy = 0;
  for (i=1;i<7;i++) {
     ixdi = xps[i];
     iydi = yps[i];
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
        continue;
     }
     if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0; }
        if( !idx ) { idx = ix; continue;}
        if( ix*idx > 0 )       idx += ix;
        else { MovePS(idx,0);  idx  = ix; }
        continue;
     }
     if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( !idy) { idy = iy; continue;}
        if( iy*idy > 0 )         idy += iy;
        else { MovePS(0,idy);    idy  = iy; }
     }
  }
  if( idx ) MovePS(idx,0);
  if( idy ) MovePS(0,idy);
  PrintFast(2," f");
}

//______________________________________________________________________________
void TPostScript::DrawPolyLine(Int_t nn, TPoints *xy)
{
//*-*-*-*-*-*-*-*-*-*-*Draw a PolyLine*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================
//*-*
//*-*  Draw a polyline through  the points  xy.
//*-*  If NN=1 moves only to point x,y.
//*-*  If NN=0 the x,y are  written  in the PostScript file
//*-*     according to the current tranformation.
//*-*  If NN>0 the line is clipped as a line.
//*-*  If NN<0 the line is clipped as a fill area.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;
  if (nn > 0) {
     n = nn;
     SetLineStyle(fLineStyle);
     SetLineWidth(fLineWidth);
     SetColor(Int_t(fLineColor));
  } else {
     n = -nn;
     SetLineStyle(1);
     SetLineWidth(1);
     SetColor(Int_t(fLineColor));
  }

  ixd0 = XtoPS(xy[0].GetX());
  iyd0 = YtoPS(xy[0].GetY());
  WriteInteger(ixd0);
  WriteInteger(iyd0);
  if( n <= 1) {
     if( n == 0) return;
     PrintFast(2," m");
     return;
  }

  PrintFast(2," m");
  idx = 0;
  idy = 0;
  for (Int_t i=1;i<n;i++) {
     ixdi = XtoPS(xy[i].GetX());
     iydi = YtoPS(xy[i].GetY());
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
        continue;
     }
     if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0; }
        if( !idx ) { idx = ix; continue;}
        if( ix*idx > 0 )       idx += ix;
        else { MovePS(idx,0);  idx  = ix; }
        continue;
     }
     if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( !idy) { idy = iy; continue;}
        if( iy*idy > 0 )         idy += iy;
        else { MovePS(0,idy);    idy  = iy; }
     }
  }
  if( idx ) MovePS(idx,0);
  if( idy ) MovePS(0,idy);

  if (nn > 0 ) {
     if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
     PrintFast(2," s");
  } else {
     PrintFast(2," f");
  }
}



//______________________________________________________________________________
void TPostScript::DrawPolyLineNDC(Int_t nn, TPoints *xy)
{
//*-*-*-*-*-*-*-*-*-*-*Draw a PolyLine in NDC space*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
//*-*
//*-*  Draw a polyline through  the points  xy.
//*-*  If NN=1 moves only to point x,y.
//*-*  If NN=0 the x,y are  written  in the PostScript file
//*-*     according to the current tranformation.
//*-*  If NN>0 the line is clipped as a line.
//*-*  If NN<0 the line is clipped as a fill area.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;
  if (nn > 0) {
     n = nn;
     SetLineStyle(fLineStyle);
     SetLineWidth(fLineWidth);
     SetColor(Int_t(fLineColor));
  } else {
     n = -nn;
     SetLineStyle(1);
     SetLineWidth(1);
     SetColor(Int_t(fLineColor));
  }

  ixd0 = UtoPS(xy[0].GetX());
  iyd0 = VtoPS(xy[0].GetY());
  WriteInteger(ixd0);
  WriteInteger(iyd0);
  if( n <= 1) {
     if( n == 0) return;
     PrintFast(2," m");
     return;
  }

  PrintFast(2," m");
  idx = 0;
  idy = 0;
  for (Int_t i=1;i<n;i++) {
     ixdi = UtoPS(xy[i].GetX());
     iydi = VtoPS(xy[i].GetY());
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
        continue;
     }
     if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0; }
        if( !idx ) { idx = ix; continue;}
        if( ix*idx > 0 )       idx += ix;
        else { MovePS(idx,0);  idx  = ix; }
        continue;
     }
     if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( !idy) { idy = iy; continue;}
        if( iy*idy > 0 )         idy += iy;
        else { MovePS(0,idy);    idy  = iy; }
     }
  }
  if( idx ) MovePS(idx,0);
  if( idy ) MovePS(0,idy);

  if (nn > 0 ) {
     if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
     PrintFast(2," s");
  } else {
     PrintFast(2," f");
  }
}


//______________________________________________________________________________
void TPostScript::DrawPolyMarker(Int_t n, Float_t *x, Float_t *y)
{
//*-*-*-*-*-*-*-*-*-*-*Draw markers at the n WC points x, y*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  Int_t i, np, markerstyle;
  Float_t markersize;
  static char chtemp[10];
  Style_t linestylesav = fLineStyle;
  Width_t linewidthsav = fLineWidth;
  SetLineStyle(1);
  SetLineWidth(1);
  SetColor(Int_t(fMarkerColor));
  markerstyle = abs(fMarkerStyle);
  if (markerstyle <= 0) strcpy(chtemp, " m20");
  if (markerstyle == 1) strcpy(chtemp, " m20");
  if (markerstyle == 2) strcpy(chtemp, " m2");
  if (markerstyle == 3) strcpy(chtemp, " m31");
  if (markerstyle == 4) strcpy(chtemp, " m24");
  if (markerstyle == 5) strcpy(chtemp, " m5");
  if (markerstyle >= 6 && markerstyle <= 19) strcpy(chtemp, " m20");
  if (markerstyle >= 20 && markerstyle <= 31 ) sprintf(chtemp, " m%d", markerstyle);
  if (markerstyle >= 32) strcpy(chtemp, " m20");

//*-*-              Set the PostScript marker size

  Double_t msize = 0.92*fMarkerSize*fXsize/20;
  markersize     = CMtoPS(msize);
  if (markerstyle == 1) markersize *= 0.1;
  if (markerstyle == 6) markersize *= 0.2;
  if (markerstyle == 7) markersize *= 0.3;
  if (fMarkerSizeCur != markersize) {
     fMarkerSizeCur = markersize;
     PrintFast(3," /w");
     WriteInteger(Int_t(markersize));
     PrintFast(40," def /w2 {w 2 div} def /w3 {w 3 div} def");
  }

  WriteInteger(XtoPS(x[0]));
  WriteInteger(YtoPS(y[0]));
  if (n == 1) {
     PrintStr(chtemp);
     SetLineStyle(linestylesav);
     SetLineWidth(linewidthsav);
     return;
  }
  np = 1;
  for (i=1;i<n;i++) {
     WriteInteger(XtoPS(x[i]));
     WriteInteger(YtoPS(y[i]));
     np++;
     if (np == 100 || i == n-1) {
        WriteInteger(np);
        PrintFast(2," {");
        PrintStr(chtemp);
        PrintFast(3,"} R");
        np = 0;
     }
  }
}



//______________________________________________________________________________
void TPostScript::DrawPolyMarker(Int_t n, Double_t *x, Double_t *y)
{
//*-*-*-*-*-*-*-*-*-*-*Draw markers at the n WC points x, y*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
  Int_t i, np, markerstyle;
  Float_t markersize;
  static char chtemp[10];
  Style_t linestylesav = fLineStyle;
  Width_t linewidthsav = fLineWidth;
  SetLineStyle(1);
  SetLineWidth(1);
  SetColor(Int_t(fMarkerColor));
  markerstyle = abs(fMarkerStyle);
  if (markerstyle <= 0) strcpy(chtemp, " m20");
  if (markerstyle == 1) strcpy(chtemp, " m20");
  if (markerstyle == 2) strcpy(chtemp, " m2");
  if (markerstyle == 3) strcpy(chtemp, " m31");
  if (markerstyle == 4) strcpy(chtemp, " m24");
  if (markerstyle == 5) strcpy(chtemp, " m5");
  if (markerstyle >= 6 && markerstyle <= 19) strcpy(chtemp, " m20");
  if (markerstyle >= 20 && markerstyle <= 31 ) sprintf(chtemp, " m%d", markerstyle);
  if (markerstyle >= 32) strcpy(chtemp, " m20");

//*-*-              Set the PostScript marker size

  Double_t msize = 0.92*fMarkerSize*fXsize/20;
  markersize     = CMtoPS(msize);
  if (markerstyle == 1) markersize *= 0.1;
  if (markerstyle == 6) markersize *= 0.2;
  if (markerstyle == 7) markersize *= 0.3;
  if (fMarkerSizeCur != markersize) {
     fMarkerSizeCur = markersize;
     PrintFast(3," /w");
     WriteInteger(Int_t(markersize));
     PrintFast(40," def /w2 {w 2 div} def /w3 {w 3 div} def");
  }

  WriteInteger(XtoPS(x[0]));
  WriteInteger(YtoPS(y[0]));
  if (n == 1) {
     PrintStr(chtemp);
     SetLineStyle(linestylesav);
     SetLineWidth(linewidthsav);
     return;
  }
  np = 1;
  for (i=1;i<n;i++) {
     WriteInteger(XtoPS(x[i]));
     WriteInteger(YtoPS(y[i]));
     np++;
     if (np == 100 || i == n-1) {
        WriteInteger(np);
        PrintFast(2," {");
        PrintStr(chtemp);
        PrintFast(3,"} R");
        np = 0;
     }
  }
}


//______________________________________________________________________________
void TPostScript::DrawPS(Int_t nn, Float_t *xw, Float_t *yw)
{
//*-*-*-*-*-*-*-*-*-*-*Draw a PolyLine*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================
//*-*
//*-*  Draw a polyline through  the points  XW,YW.
//*-*  If NN=1 moves only to point XW,YW.
//*-*  If NN=0 the XW(1) and YW(1) are  written  in the PostScript file
//*-*     according to the current NT.
//*-*  If NN>0 the line is clipped as a line.
//*-*  If NN<0 the line is clipped as a fill area.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
   static Float_t dyhatch[24] = {.0075,.0075,.0075,.0075,.0075,.0075,.0075,.0075,
                                 .01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,
                                 .015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015};
   static Float_t anglehatch[24] = {180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60};
  Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy, fais, fasi;
  fais = fasi = 0;

  if (nn > 0) {
     n = nn;
     SetLineStyle(fLineStyle);
     SetLineWidth(fLineWidth);
     SetColor(Int_t(fLineColor));
  } else {
     n = -nn;
     SetLineStyle(1);
     SetLineWidth(1);
     SetColor(Int_t(fFillColor));
     fais = fFillStyle/1000;
     fasi = fFillStyle%1000;
     if (fais == 3 || fais == 2) {
        if (fasi > 100 && fasi <125) {
           DrawHatch(dyhatch[fasi-101],anglehatch[fasi-101], n, xw, yw);
           return;
        }
        if (fasi > 0 && fasi < 26) {
           SetFillPatterns(fasi, Int_t(fFillColor));
        }
     }
  }

  Int_t jxd0 = XtoPS(xw[0]);
  Int_t jyd0 = YtoPS(yw[0]);
  ixd0 = jxd0;
  iyd0 = jyd0;
  WriteInteger(ixd0);
  WriteInteger(iyd0);
  if( n <= 1) {
     if( n == 0) return;
     PrintFast(2," m");
     return;
  }

  PrintFast(2," m");
  idx = idy = 0;
  for (Int_t i=1;i<n;i++) {
     ixdi = XtoPS(xw[i]);
     iydi = YtoPS(yw[i]);
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
     } else if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0;}
        if( !idx ) { idx = ix;}
        else if( TMath::Sign(ix,idx) == ix )       idx += ix;
        else { MovePS(idx,0);  idx  = ix;}
     } else if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0;}
        if( !idy) { idy = iy;}
        else if( TMath::Sign(iy,idy) == iy)         idy += iy;
        else { MovePS(0,idy);    idy  = iy;}
     }
  }
  if (idx) MovePS(idx,0);
  if (idy) MovePS(0,idy);

  if (nn > 0 ) {
     if (xw[0] == xw[n-1] && yw[0] == yw[n-1]) PrintFast(3," cl");
     PrintFast(2," s");
  } else {
     if (fais == 0) {PrintFast(5," cl s"); return;}
     if (fais == 3 || fais == 2) {
        if (fasi > 0 && fasi < 26) {
           PrintFast(3," FA");
        }
        return;
     }
     PrintFast(2," f");
  }

}


//______________________________________________________________________________
void TPostScript::DrawPS(Int_t nn, Double_t *xw, Double_t *yw)
{
//*-*-*-*-*-*-*-*-*-*-*Draw a PolyLine*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================
//*-*
//*-*  Draw a polyline through  the points  XW,YW.
//*-*  If NN=1 moves only to point XW,YW.
//*-*  If NN=0 the XW(1) and YW(1) are  written  in the PostScript file
//*-*     according to the current NT.
//*-*  If NN>0 the line is clipped as a line.
//*-*  If NN<0 the line is clipped as a fill area.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
   static Float_t dyhatch[24] = {.0075,.0075,.0075,.0075,.0075,.0075,.0075,.0075,
                                 .01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,
                                 .015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015};
   static Float_t anglehatch[24] = {180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60};
  Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy, fais, fasi;
  fais = fasi = 0;

  if (nn > 0) {
     n = nn;
     SetLineStyle(fLineStyle);
     SetLineWidth(fLineWidth);
     SetColor(Int_t(fLineColor));
  } else {
     n = -nn;
     SetLineStyle(1);
     SetLineWidth(1);
     SetColor(Int_t(fFillColor));
     fais = fFillStyle/1000;
     fasi = fFillStyle%1000;
     if (fais == 3 || fais == 2) {
        if (fasi > 100 && fasi <125) {
           DrawHatch(dyhatch[fasi-101],anglehatch[fasi-101], n, xw, yw);
           return;
        }
        if (fasi > 0 && fasi < 26) {
           SetFillPatterns(fasi, Int_t(fFillColor));
        }
     }
  }

  Int_t jxd0 = XtoPS(xw[0]);
  Int_t jyd0 = YtoPS(yw[0]);
  ixd0 = jxd0;
  iyd0 = jyd0;
  WriteInteger(ixd0);
  WriteInteger(iyd0);
  if( n <= 1) {
     if( n == 0) return;
     PrintFast(2," m");
     return;
  }

  PrintFast(2," m");
  idx = idy = 0;
  for (Int_t i=1;i<n;i++) {
     ixdi = XtoPS(xw[i]);
     iydi = YtoPS(yw[i]);
     ix   = ixdi - ixd0;
     iy   = iydi - iyd0;
     ixd0 = ixdi;
     iyd0 = iydi;
     if( ix && iy) {
        if( idx ) { MovePS(idx,0); idx = 0; }
        if( idy ) { MovePS(0,idy); idy = 0; }
        MovePS(ix,iy);
     } else if ( ix ) {
        if( idy )  { MovePS(0,idy); idy = 0;}
        if( !idx ) { idx = ix;}
        else if( TMath::Sign(ix,idx) == ix )       idx += ix;
        else { MovePS(idx,0);  idx  = ix;}
     } else if( iy ) {
        if( idx ) { MovePS(idx,0); idx = 0;}
        if( !idy) { idy = iy;}
        else if( TMath::Sign(iy,idy) == iy)         idy += iy;
        else { MovePS(0,idy);    idy  = iy;}
     }
  }
  if (idx) MovePS(idx,0);
  if (idy) MovePS(0,idy);

  if (nn > 0 ) {
     if (xw[0] == xw[n-1] && yw[0] == yw[n-1]) PrintFast(3," cl");
     PrintFast(2," s");
  } else {
     if (fais == 0) {PrintFast(5," cl s"); return;}
     if (fais == 3 || fais == 2) {
        if (fasi > 0 && fasi < 26) {
           PrintFast(3," FA");
        }
        return;
     }
     PrintFast(2," f");
  }

}

//______________________________________________________________________________
void TPostScript::DrawHatch(Float_t, Float_t, Int_t, Float_t *, Float_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw Fill area with hacth styles*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   Warning("DrawHatch", "hatch fill style not yet implemented");
}

//______________________________________________________________________________
void TPostScript::DrawHatch(Float_t, Float_t, Int_t, Double_t *, Double_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw Fill area with hacth styles*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   Warning("DrawHatch", "hatch fill style not yet implemented");
}

//______________________________________________________________________________
// @(#)root/postscript:$Name:  $:$Id: TPostScript.cxx,v 1.10 2001/06/07 08:39:45 brun Exp $
// Author: P.Juillot   13/08/92
void TPostScript::FontEncode()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Font Reencoding*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ================
// @(#)root/postscript:$Name:  $:$Id: TPostScript.cxx,v 1.10 2001/06/07 08:39:45 brun Exp $
// Author: P.Juillot   13/08/92

  PrintStr("@/reencdict 24 dict def");
  PrintStr(" /ReEncode");
  PrintStr(" {reencdict begin");
  PrintStr(" /nco&na exch def");
  PrintStr("@/nfnam exch def /basefontname exch");
  PrintStr(" def /basefontdict basefontname");
  PrintStr(" findfont def");
  PrintStr("@/newfont basefontdict maxlength dict def");
  PrintStr(" basefontdict");
  PrintStr(" {exch dup /FID ne");
  PrintStr("@{dup /Encoding eq");
  PrintStr(" {exch dup length array copy");
  PrintStr(" newfont 3 1 roll put} {exch ");
  PrintStr("@newfont 3 1 roll put}");
  PrintStr(" ifelse}");
  PrintStr(" {pop pop}");
  PrintStr(" ifelse");
  PrintStr(" } forall newfont");
  PrintStr("@/FontName nfnam put");
  PrintStr(" nco&na aload pop");
  PrintStr(" nco&na length 2 idiv {newfont");
  PrintStr("@/Encoding get 3 1 roll put}");
  PrintStr(" repeat");
  PrintStr(" nfnam newfont definefont pop");
  PrintStr(" end } def");
  PrintStr("@/accvec [");
  PrintStr(" 176 /agrave");
  PrintStr(" 181 /Agrave");
  PrintStr(" 190 /acircumflex");
  PrintStr(" 192 /Acircumflex");
  PrintStr("@201 /adieresis");
  PrintStr(" 204 /Adieresis");
  PrintStr(" 209 /ccedilla");
  PrintStr(" 210 /Ccedilla");
  PrintStr(" 211 /eacute");
  PrintStr("@212 /Eacute");
  PrintStr(" 213 /egrave");
  PrintStr(" 214 /Egrave");
  PrintStr(" 215 /ecircumflex");
  PrintStr(" 216 /Ecircumflex");
  PrintStr("@217 /edieresis");
  PrintStr(" 218 /Edieresis");
  PrintStr(" 219 /icircumflex");
  PrintStr(" 220 /Icircumflex");
  PrintStr("@221 /idieresis");
  PrintStr(" 222 /Idieresis");
  PrintStr(" 223 /ntilde");
  PrintStr(" 224 /Ntilde");
  PrintStr(" 226 /ocircumflex");
  PrintStr("@228 /Ocircumflex");
  PrintStr(" 229 /odieresis");
  PrintStr(" 230 /Odieresis");
  PrintStr(" 231 /ucircumflex");
  PrintStr(" 236 /Ucircumflex");
  PrintStr("@237 /udieresis");
  PrintStr(" 238 /Udieresis");
  PrintStr(" 239 /aring");
  PrintStr(" 242 /Aring");
  PrintStr(" 243 /ydieresis");
  PrintStr("@244 /Ydieresis");
  PrintStr(" 246 /aacute");
  PrintStr(" 247 /Aacute");
  PrintStr(" 252 /ugrave");
  PrintStr(" 253 /Ugrave");
  PrintStr("] def");
  PrintStr("/Times-Roman /Times-Roman accvec ReEncode@");
  PrintStr("/Times-Italic /Times-Italic accvec ReEncode@");
  PrintStr("/Times-Bold /Times-Bold accvec ReEncode@");
  PrintStr("/Times-BoldItalic /Times-BoldItalic accvec ReEncode@");
  PrintStr("/Helvetica /Helvetica accvec ReEncode@");
  PrintStr("/Helvetica-Oblique /Helvetica-Oblique accvec ReEncode@");
  PrintStr("/Helvetica-Bold /Helvetica-Bold accvec ReEncode@");
  PrintStr("/Helvetica-BoldOblique /Helvetica-BoldOblique  accvec ReEncode@");
  PrintStr("/Courier /Courier accvec ReEncode@");
  PrintStr("/Courier-Oblique /Courier-Oblique accvec ReEncode");
  PrintStr("/Courier-Bold /Courier-Bold accvec ReEncode@");
  PrintStr("/Courier-BoldOblique /Courier-BoldOblique accvec ReEncode@");
//*-*
//*-*-              Initialization of text PostScript procedures
//*-*
  PrintStr("/oshow {gsave [] 0 sd true charpath stroke gr} def@");
  PrintStr("/stwn { /fs exch def /fn exch def /text exch def fn findfont fs sf");
  PrintStr(" text sw pop xs add /xs exch def} def@");
  PrintStr("/stwb { /fs exch def /fn exch def /nbas exch def /textf exch def");
  PrintStr("textf length /tlen exch def nbas tlen gt {/nbas tlendef} if");
  PrintStr("fn findfont fs sf textf dup length nbas sub nbas getinterval sw");
  PrintStr("pop neg xs add /xs exch def} def@");
}

//______________________________________________________________________________
void TPostScript::Initialize()
{
//*-*-*-*-*-*-*-*-*-*-*PostScript Initialisation*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========================
//*-*
//*-* This routine initialize the following PostScript procedures:
//*-*
//*-*+------------+------------------+-----------------------------------+
//*-*| Macro Name | Input parameters |            Explanation            |
//*-*+------------+------------------+-----------------------------------+
//*-*|     l      | x y              | Draw a line to the x y position   |
//*-*+------------+------------------+-----------------------------------+
//*-*|     m      | x y              | Move to the position x y          |
//*-*+------------+------------------+-----------------------------------+
//*-*|     box    | dx dy x y        | Define a box                      |
//*-*+------------+------------------+-----------------------------------+
//*-*|     bl     | dx dy x y        | Draw a line box                   |
//*-*+------------+------------------+-----------------------------------+
//*-*|     bf     | dx dy x y        | Draw a filled box                 |
//*-*+------------+------------------+-----------------------------------+
//*-*|     sw     | text             | Return string width of text       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     t      | x y              | Translate                         |
//*-*+------------+------------------+-----------------------------------+
//*-*|     r      | angle            | Rotate                            |
//*-*+------------+------------------+-----------------------------------+
//*-*|     rl     | i j              | Roll the stack                    |
//*-*+------------+------------------+-----------------------------------+
//*-*|     d      | x y              | Draw a relative line to x y       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     X      | x                | Draw a relative line to x (y=0)   |
//*-*+------------+------------------+-----------------------------------+
//*-*|     Y      | y                | Draw a relative line to y (x=0)   |
//*-*+------------+------------------+-----------------------------------+
//*-*|     rm     | x y              | Move relatively to x y            |
//*-*+------------+------------------+-----------------------------------+
//*-*|     gr     |                  | Restore the graphic context       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     lw     | lwidth           | Set line width to lwidth          |
//*-*+------------+------------------+-----------------------------------+
//*-*|     sd     | [] 0             | Set dash line define by []        |
//*-*+------------+------------------+-----------------------------------+
//*-*|     s      |                  | Stroke mode                       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     c      | r g b            | Set rgb color to r g b            |
//*-*+------------+------------------+-----------------------------------+
//*-*|     cl     |                  | Close path                        |
//*-*+------------+------------------+-----------------------------------+
//*-*|     f      |                  | Fill the last describe path       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     mXX    | x y              | Draw the marker type XX at (x,y)  |
//*-*+------------+------------------+-----------------------------------+
//*-*|     Zone   | ix iy            | Define the current zone           |
//*-*+------------+------------------+-----------------------------------+
//*-*|     black  |                  | The color is black                |
//*-*+------------+------------------+-----------------------------------+
//*-*|     C      | dx dy x y        | Clipping on                       |
//*-*+------------+------------------+-----------------------------------+
//*-*|     NC     |                  | Clipping off                      |
//*-*+------------+------------------+-----------------------------------+
//*-*|     R      |                  | repeat                            |
//*-*+------------+------------------+-----------------------------------+
//*-*
//*-*.==========> (O.Couet)
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*

   Double_t rpxmin, rpymin, width, heigth;
   rpxmin = rpymin = width = heigth = 0;
   Int_t format;
   fNpages=1;
   for (Int_t i=0;i<32;i++) fPatterns[i]=0;
//*-*
//*-*- Mode is last digit of PostScript Workstation type
//*-*-     mode=1,2 for portrait/landscape black and white
//*-*-     mode=3 for Encapsulated PostScript File
//*-*-     mode=4 for portrait colour
//*-*-     mode=5 for lanscape colour
//*-*
   Int_t atype = abs(fType);
   fMode       = atype%10;
   if( fMode <= 0 || fMode > 5) {
      Error("Initialize", "invalid file type %d", fMode);
      return;
   }
//*-*
//*-*- fNXzone (fNYzone) is the total number of windows in x (y)
//*-*
   fNXzone = (atype%1000)/100;
   fNYzone = (atype%100)/10;
   if( fNXzone <= 0 ) fNXzone = 1;
   if( fNYzone <= 0 ) fNYzone = 1;
   fIXzone = 1;
   fIYzone = 1;
//*-*
//*-*- format = 0-99 is the European page format (A4,A3 ...)
//*-*- format = 100 is the US format  8.5x11.0 inch
//*-*- format = 200 is the US format  8.5x14.0 inch
//*-*- format = 300 is the US format 11.0x17.0 inch
//*-*
   format = atype/1000;
   if( format == 0 )  format = 4;
   if( format == 99 ) format = 0;
//*-*
   PrintStr("%%Title: ");
   PrintStr(GetName());
   if( fMode != 3) {;
      if ( fMode == 1 || fMode == 4) PrintFast(10," (Portrait");
      if ( fMode == 2 || fMode == 5) PrintFast(11," (Landscape");
      if ( format <= 99 ) {;
         PrintFast(2," A");
         WriteInteger(format);
         PrintFast(1,")");
      }
      else {
         if ( format == 100 ) PrintFast(8," Letter)");
         if ( format == 200 ) PrintFast(7," Legal)");
         if ( format == 300 ) PrintFast(8," Ledger)");
      }
      PrintStr("@");
      PrintStr("%%Pages: (atend)@");
   }
   else {
      PrintStr("@");
   }

   PrintFast(23,"%%Creator: ROOT Version");
   PrintStr(gROOT->GetVersion());
   PrintStr("@");
   PrintFast(15,"%%CreationDate:");
   TDatime t;
   PrintStr(t.AsString());
   PrintStr("@");
   PrintStr("%%EndComments@");
   PrintStr("%%BeginProlog@");

   if( fMode == 3)PrintStr("80 dict begin@");
//*-*
//*-*-              Initialization of PostScript procedures
//*-*
   PrintStr("/s {stroke} def /l {lineto} def /m {moveto} def /t {translate} def@");
   PrintStr("/sw {stringwidth} def /r {rotate} def /rl {roll}  def /R {repeat} def@");
   PrintStr("/d {rlineto} def /rm {rmoveto} def /gr {grestore} def /f {eofill} def@");
   PrintStr("/c {setrgbcolor} def /lw {setlinewidth} def /sd {setdash} def@");
   PrintStr("/cl {closepath} def /sf {scalefont setfont} def /black {0 setgray} def@");
   PrintStr("/box {m dup 0 exch d exch 0 d 0 exch neg d cl} def@");
   PrintStr("/NC{systemdict begin initclip end}def/C{NC box clip newpath}def@");
   PrintStr("/bl {box s} def /bf {box f} def /Y { 0 exch d} def /X { 0 d} def @");

   DefineMarkers();
   FontEncode();
   MakeGreek();
//*-*
//*-*-     mode=1 for portrait black/white
//*-*
   if (fMode == 1)  {
      rpxmin = 0.7;
      rpymin = TMath::Sqrt(2.)*rpxmin;
      switch (format) {
         case 100 :
              width  = (8.5*2.54)-2.*rpxmin;
              heigth = (11.*2.54)-2.*rpymin;
              break;
         case 200 :
              width  = (8.5*2.54)-2.*rpxmin;
              heigth = (14.*2.54)-2.*rpymin;
              break;
         case 300 :
              width  = (11.*2.54)-2.*rpxmin;
              heigth = (17.*2.54)-2.*rpymin;
              break;
         default  :
              width  = 21.0-2.*rpxmin;
              heigth = 29.7-2.*rpymin;
      };
   }
//*-*
//*-*-     mode=2 for landscape black/white
//*-*
   if (fMode == 2)  {
      rpymin = 0.7;
      rpxmin = TMath::Sqrt(2.)*rpymin;
      switch (format) {
         case 100 :
              width  = (11.*2.54)-2.*rpxmin;
              heigth = (8.5*2.54)-2.*rpymin;
         case 200 :
              width  = (14.*2.54)-2.*rpxmin;
              heigth = (8.5*2.54)-2.*rpymin;
         case 300 :
              width  = (17.*2.54)-2.*rpxmin;
              heigth = (11.*2.54)-2.*rpymin;
         default  :
              width  = 29.7-2.*rpxmin;
              heigth = 21-2.*rpymin;
      };
   }
//*-*
//*-*-     mode=3 encapsulated PostScript
//*-*
   if (fMode == 3)  {
      width   = fX2w;
      heigth  = fY2w;
      format  = 4;
      fNXzone = 1;
      fNYzone = 1;
   }
//*-*
//*-*-     mode=4 for portrait colour
//*-*
   if (fMode == 4)  {
      rpxmin = 0.7;
      rpymin = 3.4;
      switch (format) {
         case 100 :
              width  = (8.5*2.54)-2.*rpxmin;
              heigth = (11.*2.54)-2.*rpymin;
         case 200 :
              width  = (8.5*2.54)-2.*rpxmin;
              heigth = (14.*2.54)-2.*rpymin;
         case 300 :
              width  = (11.*2.54)-2.*rpxmin;
              heigth = (17.*2.54)-2.*rpymin;
         default  :
              width  = (21.0-2*rpxmin);
              heigth = (29.7-2.*rpymin);
      };
   }
//*-*
//*-*-     mode=5 for lanscape colour
//*-*
   if (fMode == 5)  {
      rpxmin = 3.4;
      rpymin = 0.7;
      switch (format) {
         case 100 :
              width  = (11.*2.54)-2.*rpxmin;
              heigth = (8.5*2.54)-2.*rpymin;
         case 200 :
              width  = (14.*2.54)-2.*rpxmin;
              heigth = (8.5*2.54)-2.*rpymin;
         case 300 :
              width  = (17.*2.54)-2.*rpxmin;
              heigth = (11.*2.54)-2.*rpymin;
         default  :
              width  = (29.7-2*rpxmin);
              heigth = (21-2.*rpymin);
      };
   }

   Double_t value = 0;
   if       (format <  100) value = 21*TMath::Power(TMath::Sqrt(2.), 4-format);
   else  if (format == 100) value = 8.5*2.54;
   else  if (format == 200) value = 8.5*2.54;
   else  if (format == 300) value = 11.*2.54;
   if       (format >= 100) format = 4;
//*-*
//*-*- Compute size (in points) of the window for each picture = f(fNXzone,fNYzone)
//*-*
   Double_t sizex = width/Double_t(fNXzone)*TMath::Power(TMath::Sqrt(2.), 4-format);
   Double_t sizey = heigth/Double_t(fNYzone)*TMath::Power(TMath::Sqrt(2.), 4-format);
   Int_t npx      = 4*CMtoPS(sizex);
   Int_t npy      = 4*CMtoPS(sizey);
   if (sizex > sizey) fMaxsize = CMtoPS(sizex);
   else               fMaxsize = CMtoPS(sizey);
//*-*
//*-*- Procedure Zone
//*-*
   if (fMode != 3)  {
      PrintFast(33,"/Zone {/iy exch def /ix exch def ");
      PrintFast(10," ix 1 sub ");
      WriteInteger(npx);
      PrintFast(5," mul ");
      WriteReal(Float_t(fNYzone));
      PrintFast(8," iy sub ");
      WriteInteger(npy);
      PrintStr(" mul t} def@");
   }

   PrintStr("%%EndProlog@");
   PrintFast(8,"newpath ");
   SaveRestore(1);
   if (fMode == 1 || fMode == 4)  {
      WriteInteger(CMtoPS(rpxmin));
      WriteInteger(CMtoPS(rpymin));
      PrintFast(2," t");
   }
   if (fMode == 2 || fMode == 5)  {
      PrintFast(7," 90 r 0");
      WriteInteger(CMtoPS(-value));
      PrintFast(3," t ");
      WriteInteger(CMtoPS(rpxmin));
      WriteInteger(CMtoPS(rpymin));
      PrintFast(2," t");
   }

   PrintFast(15," .25 .25 scale ");
   if (fMode != 3) SaveRestore(1);

   if (fMode != 3) PrintStr("%%Page: number 1@");

   //Check is user has defined a special header in the current style
   Int_t nh = strlen(gStyle->GetHeaderPS());
   if (nh) {
      PrintFast(nh,gStyle->GetHeaderPS());
      if (fMode != 3) SaveRestore(1);
   }
}

//______________________________________________________________________________
void TPostScript::MakeGreek()
{
//*-*-*Reencode the Greek (/Symbol) font into the special font (/Special)*-*-*
//*-*  ===================================================================
//*-*
//*-*.==========> (O.Couet)
  PrintStr("@/accspe [");
  PrintStr(" 65 /plusminus ");
  PrintStr(" 66 /bar ");
  PrintStr(" 67 /existential ");
  PrintStr(" 68 /universal ");
  PrintStr("@69 /exclam ");
  PrintStr(" 70 /numbersign");
  PrintStr(" 71 /greater ");
  PrintStr(" 72 /question ");
  PrintStr(" 73 /integral ");
  PrintStr("@74 /colon ");
  PrintStr(" 75 /semicolon ");
  PrintStr(" 76 /less ");
  PrintStr(" 77 /bracketleft ");
  PrintStr(" 78 /bracketright");
  PrintStr("@79 /greaterequal");
  PrintStr(" 80 /braceleft");
  PrintStr(" 81 /braceright");
  PrintStr(" 82 /radical");
  PrintStr("@83 /spade");
  PrintStr(" 84 /heart");
  PrintStr(" 85 /diamond");
  PrintStr(" 86 /club");
  PrintStr(" 87 /lessequal");
  PrintStr("@88 /multiply");
  PrintStr(" 89 /percent");
  PrintStr(" 90 /infinity");
  PrintStr(" 48 /circlemultiply");
  PrintStr(" 49 /circleplus");
  PrintStr("@50 /emptyset ");
  PrintStr(" 51 /lozenge");
  PrintStr(" 52 /bullet");
  PrintStr(" 53 /arrowright");
  PrintStr(" 54 /arrowup");
  PrintStr("@55 /arrowleft");
  PrintStr(" 56 /arrowdown");
  PrintStr(" 57 /arrowboth");
  PrintStr(" 48 /degree");
  PrintStr(" 44 /comma");
  PrintStr(" 43 /plus");
  PrintStr(" 45 /angle");
  PrintStr(" 42 /angleleft");
  PrintStr(" 47 /divide");
  PrintStr(" 61 /notequal");
  PrintStr(" 40 /equivalence");
  PrintStr(" 41 /second");

  PrintStr(" 97 /approxequal");
  PrintStr(" 98 /congruent");
  PrintStr(" 99 /perpendicular");
  PrintStr(" 100 /partialdiff");
  PrintStr(" 101 /florin");
  PrintStr(" 102 /intersection");
  PrintStr(" 103 /union");
  PrintStr(" 104 /propersuperset");
  PrintStr(" 105 /reflexsuperset");
  PrintStr(" 106 /notsubset");
  PrintStr(" 107 /propersubset");
  PrintStr(" 108 /reflexsubset");
  PrintStr(" 109 /element");
  PrintStr(" 110 /notelement");
  PrintStr(" 111 /gradient");
  PrintStr(" 112 /logicaland");
  PrintStr(" 113 /logicalor");
  PrintStr(" 114 /arrowdblboth");
  PrintStr(" 115 /arrowdblleft");
  PrintStr(" 116 /arrowdblup");
  PrintStr(" 117 /arrowdblright");
  PrintStr(" 118 /arrowdbldown");
  PrintStr(" 119 /ampersand");
  PrintStr(" 120 /omega1");
  PrintStr(" 121 /similar");
  PrintStr(" 122 /aleph");
  PrintStr(" ] def");
  PrintStr("/Symbol /Special accspe ReEncode@");

}

//______________________________________________________________________________
void TPostScript::MovePS(Int_t ix, Int_t iy)
{
//*-*-*-*-*-*-*-*-*-*-*Move to a new position*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
//printf("ix=%d, iy=%d\n",ix,iy);
  if (ix != 0 && iy != 0)  {
      WriteInteger(ix);
      WriteInteger(iy);
      PrintFast(2," d");
   }
   else if (ix != 0)  {
      WriteInteger(ix);
      PrintFast(2," X");
   }
   else if (iy != 0)  {
      WriteInteger(iy);
      PrintFast(2," Y");
   }
}

//______________________________________________________________________________
void TPostScript::NewPage()
{
//*-*-*-*-*-*-*-*-*-*Move to a new PostScript page*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                =============================

//*-*   Compute pad conversion coefficients
  if (gPad) {
//     if (!gPad->GetPadPaint()) gPad->Update();
     Double_t ww   = gPad->GetWw();
     Double_t wh   = gPad->GetWh();
     fYsize       = fXsize*wh/ww;
  } else fYsize = 27;

  if(fType  == 113 && !fBoundingBox) {
     Bool_t psave = fPrinted;
     PrintStr("@%%BoundingBox: ");
     Double_t xlow = gPad->GetAbsXlowNDC();
     Double_t xup  = xlow + gPad->GetAbsWNDC();
     Double_t ylow = gPad->GetAbsYlowNDC();
     Double_t yup  = ylow + gPad->GetAbsHNDC();
     WriteInteger(CMtoPS(fXsize*xlow));
     WriteInteger(CMtoPS(fYsize*ylow));
     WriteInteger(CMtoPS(fXsize*xup));
     WriteInteger(CMtoPS(fYsize*yup));
     PrintStr("@");
     Initialize();
     fBoundingBox  = kTRUE;
     fPrinted      = psave;
  }
  if (fPrinted) {
     if (fSave) SaveRestore(-1);
     fClear    = kTRUE;
     fPrinted  = kFALSE;
  }
  Zone();
}

//______________________________________________________________________________
void TPostScript::PrintStr(const char *str)
{
//*-*-*-*-*-*-*-*Output the string STR in the output buffer*-*-*-*-*-*-*-*-*-*
//*-*            ===========================================
//*-*
  Int_t len = strlen(str);
  if (len == 0) return;
  if( str[0] == '@') {
     if( fLenBuffer ) {
        fStream->write(fBuffer, fLenBuffer);
        fStream->write("\n",1);
     }
     if( len < 2)  fBuffer[0] = ' ';
     else          strcpy(fBuffer, str+1);
     fLenBuffer = len-1;
     fPrinted = kTRUE;
     return;
  }

  if( str[len-1] == '@') {
     if( fLenBuffer ) {
        fStream->write(fBuffer, fLenBuffer);
        fStream->write("\n",1);
     }
     fStream->write(str, len-1);
     fStream->write("\n",1);
     fLenBuffer = 0;
     fPrinted = kTRUE;
     return;
  }

  if( (len + fLenBuffer ) > kMaxBuffer) {
     fStream->write(fBuffer, fLenBuffer);
     fStream->write("\n",1);
     strcpy(fBuffer, str);
     fLenBuffer = len;
  }
  else {
     strcpy(fBuffer + fLenBuffer, str);
     fLenBuffer += len;
  }
  fPrinted = kTRUE;
}

//______________________________________________________________________________
void TPostScript::PrintFast(Int_t len, const char *str)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Fast version of Print*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          =====================
  if( (len + fLenBuffer ) > kMaxBuffer) {
     fStream->write(fBuffer, fLenBuffer);
     fStream->write("\n",1);
     strcpy(fBuffer, str);
     fLenBuffer = len;
  }
  else {
     strcpy(fBuffer + fLenBuffer, str);
     fLenBuffer += len;
  }
  fPrinted = kTRUE;
}

//______________________________________________________________________________
void TPostScript::Range(Float_t xsize, Float_t ysize)
{
//*-*-*-*-*-*-*-*-*-*-*Set the range for the paper in centimeters*-*-*-*-*-*-*
//*-*                  ===========================================
  Float_t xps, yps, xncm, yncm, dxwn, dywn, xwkwn, ywkwn, xymax;

  fXsize = xsize;
  fYsize = ysize;
  if( fType != 113) { xps = fXsize;  yps = fYsize; }
  else {              xps = xsize;   yps = ysize; }

  if( xsize <= xps && ysize < yps) {
     if ( xps > yps ) xymax = xps;
     else             xymax = yps;
     xncm  = xsize/xymax;
     yncm  = ysize/xymax;
     dxwn  = ((xps/xymax)-xncm)/2;
     dywn  = ((yps/xymax)-yncm)/2;
  }
  else {
     if (xps/yps < 1) xwkwn = xps/yps;
     else             xwkwn = 1;
     if (yps/xps < 1) ywkwn = yps/xps;
     else             ywkwn = 1;

     if (xsize < ysize)  {
        xncm = ywkwn*xsize/ysize;
        yncm = ywkwn;
        dxwn = (xwkwn-xncm)/2;
        dywn = 0;
        if( dxwn < 0) {
           xncm = xwkwn;
           dxwn = 0;
           yncm = xwkwn*ysize/xsize;
           dywn = (ywkwn-yncm)/2;
        }
     }
     else {
        xncm = xwkwn;
        yncm = xwkwn*ysize/xsize;
        dxwn = 0;
        dywn = (ywkwn-yncm)/2;
        if( dywn < 0) {
           yncm = ywkwn;
           dywn = 0;
           xncm = ywkwn*xsize/ysize;
           dxwn = (xwkwn-xncm)/2;
        }
     }
  }
  fXVP1  = dxwn;
  fXVP2  = xncm+dxwn;
  fYVP1  = dywn;
  fYVP2  = yncm+dywn;
  fRange = kTRUE;

}

//______________________________________________________________________________
void TPostScript::SaveRestore(Int_t flag)
{
//*-*-*-*-*-*-*-*Compute number of gsaves for restore*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================================
//*-* This allows to write the correct number of grestore at the
//*-* end of the PS file.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

      if (flag == 1) { PrintFast(7," gsave ");  fSave++; }
      else           { PrintFast(4," gr ");     fSave--; }
}


//______________________________________________________________________________
void TPostScript::SetFillColor( Color_t cindex )
{
//*-*-*-*-*-*-*-*-*-*-*Set color index for fill areas*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
//*-*  cindex : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  fFillColor = cindex;
  if (gStyle->GetFillColor() <= 0) cindex = 0;
  SetColor(Int_t(cindex));
}


//______________________________________________________________________________
void TPostScript::SetFillPatterns(Int_t ipat, Int_t color)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Patterns definition*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ===================
// @(#)root/postscript:$Name:  $:$Id: TPostScript.cxx,v 1.10 2001/06/07 08:39:45 brun Exp $
// Author: O.Couet   16/07/99
//*-*
//*-* Define the pattern ipat in the current PS file. ipat can vary from
//*-* 1 to 25. Together with the pattern, the color (color) in which the
//*-* pattern has to be drawn is also required. A pattern is defined in the
//*-* current PS file only the first time it is used. Some level 2
//*-* Postscript functions are used, so on level 1 printers, patterns will
//*-* not work. This is not a big problem because patterns are
//*-* defined only if they are used, so if they are not used a PS level 1
//*-* file will not be poluted by level 2 features, and in any case the old
//*-* patterns used a lot of memory which made them almost unusable on old
//*-* level 1 printers. Finally we should say that level 1 devices are
//*-* becoming very rare. The official PostScript is now level 3 !
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
   char cdef[28];
   char cpat[5];
   sprintf(cpat, " P%2.2d", ipat);
//*-*
//*-*- fPatterns is used as an array of chars. If fPatterns[ipat] != 0 the
//*-*- pattern number ipat as already be defined is this file and it
//*-*- is not necessary to redefine it. fPatterns is set to zero in Initialize.
//*-*- The pattern number 26 allows to know if the macro "cs" has already
//*-*- been defined in the current file (see label 200).
//*-*
   if (fPatterns[ipat] == 0) {
//*-*
//*-*- Define the Patterns.
//*-*
      PrintStr(" << /PatternType 1 /PaintType 2 /TilingType 1");
      switch (ipat) {
         case 1 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 98 /YStep 4");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [1] 0 sd 2 4 m 99 4 l s 1 3 m 98 3 l s");
            PrintStr(" 2 2 m 99 2 l s 1 1 m 98 1 l s");
            PrintStr(" gr end } >> [ 4.0 0 0 4.0 0 0 ]");
            break;
         case 2 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 96 /YStep 4");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [1 3] 0 sd 2 4 m 98 4 l s 0 3 m 96 3 l s");
            PrintStr(" 2 2 m 98 2 l s 0 1 m 96 1 l s");
            PrintStr(" gr end } >> [ 3.0 0 0 3.0 0 0 ]");
            break;
         case 3 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 96 /YStep 16");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [1 3] 0 sd 2 13 m 98 13 l s 0 9 m 96 9 l s");
            PrintStr(" 2 5 m 98 5 l s 0 1 m 96 1 l s");
            PrintStr(" gr end } >> [ 2.0 0 0 2.0 0 0 ]");
            break;
         case 4 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 0 m 100 100 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 5 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 100 m 100 0 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 6 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 50 0 m 50 100 l s");
            PrintStr(" gr end } >> [ 0.12 0 0 0.12 0 0 ]");
            break;
         case 7 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 50 m 100 50 l s");
            PrintStr(" gr end } >> [ 0.12 0 0 0.12 0 0 ]");
            break;
         case 8 :
            PrintStr(" /BBox [ 0 0 101 101 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 0 m 0 30 l 30 0 l f 0 70 m 0 100 l 30 100 l f");
            PrintStr(" 70 100 m 100 100 l 100 70 l f 70 0 m 100 0 l");
            PrintStr(" 100 30 l f 50 20 m 20 50 l 50 80 l 80 50 l f");
            PrintStr(" 50 80 m 30 100 l s 20 50 m 0 30 l s 50 20 m");
            PrintStr(" 70 0 l s 80 50 m 100 70 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 9 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 50 m 50 50 50 180 360 arc");
            PrintStr(" 0 50 m 0 100 50 270 360 arc");
            PrintStr(" 50 100 m 100 100 50 180 270 arc s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 10 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 50 m 100 50 l 1 1 m 100 1 l");
            PrintStr(" 0 0 m 0 50 l 100 0 m 100 50 l");
            PrintStr(" 50 50 m 50 100 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 11 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 0 m 0 20 l 50 0 m 50 20 l");
            PrintStr(" 100 0 m 100 20 l 0 80 m 0 100 l");
            PrintStr(" 50 80 m 50 100 l 100 80 m 100 100 l");
            PrintStr(" 25 30 m 25 70 l 75 30 m 75 70 l");
            PrintStr(" 0 100 m 20 85 l 50 100 m 30 85 l");
            PrintStr(" 50 100 m 70 85 l 100 100 m 80 85 l");
            PrintStr(" 0 0 m 20 15 l 50 0 m 30 15 l");
            PrintStr(" 50 0 m 70 15 l 100 0 m 80 15 l");
            PrintStr(" 5 35 m 45 65 l 5 65 m 45 35 l");
            PrintStr(" 55 35 m 95 65 l 55 65 m 95 35 l s");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
            break;
         case 12 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 80 m 0 100 20 270 360 arc");
            PrintStr(" 30 100 m 50 100 20 180 360 arc");
            PrintStr(" 80 100 m 100 100 20 180 270 arc");
            PrintStr(" 20 0 m 0 0 20 0 90 arc");
            PrintStr(" 70 0 m 50 0 20 0 180 arc");
            PrintStr(" 100 20 m 100 0 20 90 180 arc");
            PrintStr(" 45 50 m 25 50 20 0 360 arc");
            PrintStr(" 95 50 m 75 50 20 0 360 arc s");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
            break;
         case 13 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 0 m 100 100 l 0 100 m 100 0 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 14 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 80 /YStep 80");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 20 m 100 20 l 20 0 m 20 100 l");
            PrintStr(" 0 80 m 100 80 l 80 0 m 80 100 l");
            PrintStr(" 20 40 m 60 40 l 60 20 m 60 60 l");
            PrintStr(" 40 40 m 40 80 l 40 60 m 80 60 l s");
            PrintStr(" gr end } >> [ 0.60 0 0 0.60 0 0 ]");
            break;
         case 15 :
            PrintStr(" /BBox [ 0 0 60 60 ]");
            PrintStr(" /XStep 60 /YStep 60");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 55 m 0 60 5 270 360 arc");
            PrintStr(" 25 60 m 30 60 5 180 360 arc");
            PrintStr(" 55 60 m 60 60 5 180 270 arc");
            PrintStr(" 20 30 m 15 30 5 0 360 arc");
            PrintStr(" 50 30 m 45 30 5 0 360");
            PrintStr(" arc 5 0 m 0 0 5 0 90 arc");
            PrintStr(" 35 0 m 30 0 5 0 180 arc");
            PrintStr(" 60 5 m 60 0 5 90 180 arc s");
            PrintStr(" gr end } >> [ 0.41 0 0 0.41 0 0 ]");
            break;
         case 16 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 50 50 m 25 50 25 0 180 arc s");
            PrintStr(" 50 50 m 75 50 25 180 360 arc s");
            PrintStr(" gr end } >> [ 0.4 0 0 0.2 0 0 ]");
            break;
         case 17 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [24] 0 setdash 0 0 m 100 100 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 18 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [24] 0 setdash 0 100 m 100 0 l s");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 19 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 90 50 m 50 50 40 0 360 arc");
            PrintStr(" 0 50 m 0 100 50 270 360 arc");
            PrintStr(" 50 0 m 0 0 50 0 90 arc");
            PrintStr(" 100 50 m 100 0 50 90 180 arc");
            PrintStr(" 50 100 m 100 100 50 180 270 arc s");
            PrintStr(" gr end } >> [ 0.47 0 0 0.47 0 0 ]");
            break;
         case 20 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 50 50 m 50 75 25 270 450 arc s");
            PrintStr(" 50 50 m 50 25 25 90  270 arc s");
            PrintStr(" gr end } >> [ 0.2 0 0 0.4 0 0 ]");
            break;
         case 21 :
            PrintStr(" /BBox [ 0 0 101 101 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 1 1 m 25 1 l 25 25 l 50 25 l 50 50 l");
            PrintStr(" 75 50 l 75 75 l 100 75 l 100 100 l");
            PrintStr(" 50 1 m 75 1 l 75 25 l 100 25 l 100 50 l");
            PrintStr(" 0 50 m 25 50 l 25 75 l 50 75 l 50 100 l s");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
            break;
         case 22 :
            PrintStr(" /BBox [ 0 0 101 101 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 1 100 m 25 100 l 25 75 l 50 75 l 50 50 l");
            PrintStr(" 75 50 l 75 25 l 100 25 l 100 1 l");
            PrintStr(" 50 100 m 75 100 l 75 75 l 100 75 l 100 50 l");
            PrintStr(" 0 50 m 25 50 l 25 25 l 50 25 l 50 1 l s");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
            break;
         case 23 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" [1 7] 0 sd 0 8 50 { dup dup m 2 mul 0 l s } for");
            PrintStr(" 0 8 50 { dup dup 2 mul 100 m 50 add exch 50");
            PrintStr(" add l s } for 100 0 m 100 100 l 50 50 l f");
            PrintStr(" gr end } >> [ 0.24 0 0 0.24 0 0 ]");
            break;
         case 24 :
            PrintStr(" /BBox [ 0 0 100 100 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 100 100 m 100 36 l 88 36 l 88 88 l f");
            PrintStr(" 100 0 m 100 12 l 56 12 l 50 0 l f");
            PrintStr(" 0 0 m 48 0 l 48 48 l 50 48 l 56 60 l");
            PrintStr(" 36 60 l 36 12 l 0 12 l f [1 7] 0 sd");
            PrintStr(" 61 8 87 { dup dup dup 12 exch m 88 exch l s");
            PrintStr(" 16 exch 4 sub m 88 exch 4 sub l s } for");
            PrintStr(" 13 8 35 { dup dup dup 0 exch m 36 exch l s");
            PrintStr(" 4 exch 4 sub m 36 exch 4 sub l s } for");
            PrintStr(" 37 8 59 { dup dup dup 12 exch m 36 exch l s");
            PrintStr(" 16 exch 4 sub m 36 exch 4 sub l s } for");
            PrintStr(" 13 8 60 { dup dup dup 56 exch m 100 exch l s");
            PrintStr(" 60 exch 4 sub m 100 exch 4 sub l s } for");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
            break;
         case 25 :
            PrintStr(" /BBox [ 0 0 101 101 ]");
            PrintStr(" /XStep 100 /YStep 100");
            PrintStr(" /PaintProc { begin gsave");
            PrintStr(" 0 0 m 30 30 l 70 30 l 70 70 l 100 100 l 100 0 l");
            PrintStr(" f 30 30 m 30 70 l 70 70 l f");
            PrintStr(" gr end } >> [ 0.5 0 0 0.5 0 0 ]");
      };
      sprintf(cdef, " makepattern /%s exch def",&cpat[1]);
      PrintStr(cdef);
      fPatterns[ipat] = 1;
   }
//*-*
//*-*- Define the macro cs and FA if they are not yet defined.
//*-*
  if (fPatterns[26] == 0) {
      PrintStr(" /cs {[/Pattern /DeviceRGB] setcolorspace} def");
      PrintStr(" /FA {f [/DeviceRGB] setcolorspace} def");
      fPatterns[26] = 1;
   }
//*-*
//*-*- Activate the pattern.
//*-*
   PrintFast(3," cs");
   TColor *col = gROOT->GetColor(color);
   if (col) {
      WriteReal(col->GetRed());
      WriteReal(col->GetGreen());
      WriteReal(col->GetBlue());
   }
   PrintFast(4,cpat);
   PrintFast(9," setcolor");
}


//______________________________________________________________________________
void TPostScript::SetLineColor( Color_t cindex )
{
//*-*-*-*-*-*-*-*-*-*-*Set color index for lines*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  cindex    : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  fLineColor = cindex;
  SetColor(Int_t(cindex));
}

//______________________________________________________________________________
void TPostScript::SetLineStyle(Style_t linestyle)
{
//*-*-*-*-*-*-*-*-*-*-*Change the line style*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================
//*-*
//*-*   linestyle = 2 dashed
//*-*             = 3  dotted
//*-*             = 4  dash-dotted
//*-*              else = solid
//*-*
//*-*.==========> See TStyle::SetLineStyleString for style definition
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fLineStyle = linestyle;
   const char *st = gStyle->GetLineStyleString(linestyle);
   Int_t nch = strlen(st);
   PrintFast(nch,st);
}


//______________________________________________________________________________
void TPostScript::SetLineWidth(Width_t linewidth)
{
//*-*-*-*-*-*-*-*-*-*-*Change the line width*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

//   if ( linewidth == fLineWidth) return;
   fLineWidth = linewidth;
//   WriteInteger(Int_t(1.5*fLineWidth));
   WriteInteger(Int_t(fLineScale*fLineWidth));
   PrintFast(3," lw");

}

//______________________________________________________________________________
void TPostScript::SetMarkerColor( Color_t cindex )
{
//*-*-*-*-*-*-*-*-*-*-*Set color index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
//*-*  cindex : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  fMarkerColor = cindex;
  SetColor(Int_t(cindex));
}

//______________________________________________________________________________
void TPostScript::SetColor(Int_t color)
{
//*-*-*-*-*-*-*-*-*-*-*Set the current color*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================
//*-*

//  if (fCurrentColor == color) return;
  if( color < 0) color = 0;
  fCurrentColor = color;
  TColor *col = gROOT->GetColor(color);
  Bool_t white = kTRUE;
  if (col) {
     if( col->GetRed() == fRed && col->GetGreen() == fGreen && col->GetBlue() == fBlue) return;
     fRed   = col->GetRed();
     fGreen = col->GetGreen();
     fBlue  = col->GetBlue();
     white = kFALSE;
  }
  if (white) {fRed = fGreen = fBlue = 1;}
  if( fRed <= 0 && fGreen <= 0 && fBlue <= 0 ) {
     PrintFast(6," black");
  }
  else {
     WriteReal(fRed);
     WriteReal(fGreen);
     WriteReal(fBlue);
     PrintFast(2," c");
  }

}



//______________________________________________________________________________
void TPostScript::SetTextColor( Color_t cindex )
{
//*-*-*-*-*-*-*-*-*-*-*Set color index for text*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
//*-*  cindex    : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  fTextColor = cindex;

  SetColor( Int_t(cindex) );
}

//______________________________________________________________________________
void TPostScript::Text(Double_t xx, Double_t yy, const char *chars)
{
//*-*-*-*-*-*-*-*-*-*-*Write a string of characters*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//*-*
//*-*   This routine writes the string chars into a PostScript file
//*-*    at position xx,yy in world coordinates.
//*-*
//*-*  Note the special action of the following special characters:
//*-*
//*-*       ` : go to greek
//*-*       ' : go to special
//*-*       ~ : go to ZapfDingbats
//*-*       ? : go to subscript
//*-*       ^ : go to superscript
//*-*       ! : go to normal level of script
//*-*       & : backspace one character
//*-*       # : end of greek or of ZapfDingbats
//*-*
//*-*  Note1: This special characters have no effect on the screen.
//*-*  Note2: To print one of the characters above in the Postscript file
//*-*         use the escape character "@" in front of the character.
//*-*
//*-*.==========> (P.Juillot)
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*- npiece= max number of pieces of text ( separated by escape characters)
   const Int_t npiece = 50;
   const Int_t kline  = 80;
   Int_t ifnb[npiece], ifns[npiece], level[npiece], lback[npiece];
//*-*- maximum length of a (PostScript) string

   Int_t ideb, n1, n2, ipiece, fnb;
   ideb = 0;
   const char *psfnb;
   char *pc;
   char *lunps;
   char *scape;
   char *piece[npiece];
   static char newtext[512];
   static char char2[512];
   static char kpiece[kline*npiece];
   static char klunps[80];
   static char pstemp[3];

   Bool_t roman,greek,special,zapf,subscript,superscript,escape;
   Bool_t show;

   const char *cflip = "`'\?^@#!~&";
   static const char *cflipc[] = {"120","47","77","136","100","43","41","176","46"};
   Float_t psrap[15] = {0.920,1.000,0.920,0.910,0.920,0.920,0.925,1.000,1.000,
                        1.000,1.000,0.920,1.000,0.964,1.0};

   static const char *psfont[] = {
    "/Times-Italic", "/Times-Bold", "/Times-BoldItalic",
    "/Helvetica", "/Helvetica-Oblique", "/Helvetica-Bold",
    "/Helvetica-BoldOblique", "/Courier", "/Courier-Oblique",
    "/Courier-Bold", "/Courier-BoldOblique", "/Symbol","/Times-Roman",
    "/ZapfDingbats", "/Times-Italic", "/Times-Bold", "/Times-BoldItalic",
    "/Helvetica", "/Helvetica-Oblique", "/Helvetica-Bold",
    "/Helvetica-BoldOblique", "/Symbol", "/Times-Roman", "/ZapfDingbats",
    "/Special", "/ZapfChancery-MediumItalic", "/AvantGarde-Book",
    "/AvantGarde-BookOblique", "/AvantGarde-Demi",
    "/AvantGarde-DemiOblique", "/Bookman-Demi", "/Bookman-DemiItalic",
    "/Bookman-Light", "/Bookman-LightItalic", "/Palatino-Roman",
    "/Palatino-Italic", "/Palatino-Bold", "/Palatino-BoldItalic",
    "/NewCenturySchlbk-Roman", "/NewCenturySchlbk-Italic",
    "/NewCenturySchlbk-Bold", "/NewCenturySchlbk-BoldItalic"};
//*-*______________________________________


   const Double_t kDEGRAD = TMath::Pi()/180.;
   Double_t x = xx;
   Double_t y = yy;
   lunps   = &klunps[0];
   Int_t nold = strlen(chars);
   if (nold == 0) return;
   if (nold > 512) nold = 512;
//*-*
//*-*- Compute the fonts size. Exit if it is 0
//*-*
   Double_t     wh = (Double_t)gPad->XtoPixel(gPad->GetX2());
   Double_t     hh = (Double_t)gPad->YtoPixel(gPad->GetY1());
   Float_t tsize;
   if (wh < hh)  tsize = fTextSize*wh;
   else          tsize = fTextSize*hh;
   Float_t ftsize;

   Int_t font     = abs(fTextFont)/10;
   if( font > 42 || font < 1) font = 1;
   Float_t fontrap = 1.01;
   if( font <= 14 && font >= 1) fontrap = fontrap*psrap[font-1];
   if (wh < hh) ftsize = fTextSize*fXsize*gPad->GetAbsWNDC();
   else         ftsize = fTextSize*fYsize*gPad->GetAbsHNDC();

   Float_t rchh   = ftsize*fontrap;   //*-* rchh should be the text size in cm
   Int_t fontsize = 4*CMtoPS(rchh);
   if( fontsize <= 0) return;
   Float_t tsizex = gPad->AbsPixeltoX(Int_t(tsize))-gPad->AbsPixeltoX(0);
   Float_t tsizey = gPad->AbsPixeltoY(0)-gPad->AbsPixeltoY(Int_t(tsize));
//*-*
//*-*- Text colour and vertical alignment
//*-*
   SetColor(Int_t(fTextColor));
   Int_t txalh   = fTextAlign/10;
   if (txalh <1) txalh = 1; if (txalh > 3) txalh = 3;
   Int_t txalv   = fTextAlign%10;
   if (txalv <1) txalv = 1; if (txalv > 3) txalv = 3;
   if( txalv == 3) {
      y -= 0.8*tsizey*TMath::Cos(kDEGRAD*fTextAngle);
      x += 0.8*tsizex*TMath::Sin(kDEGRAD*fTextAngle);
   }
   else if( txalv == 2) {
      y -= 0.4*tsizey*TMath::Cos(kDEGRAD*fTextAngle);
      x += 0.4*tsizex*TMath::Sin(kDEGRAD*fTextAngle);
   }
//*-*
//*-*- The hollow fonts are set on by the roman font number
//*-*
   show = kFALSE;
   if( font > 14 && font < 25) show = kTRUE;
//*-*
//*-*- Start a first parsing:
//*-*  - manage the '@' escape character
//*-*  - check if the input string is not too long (J<=505)
//*-*
   Bool_t curlybracket = kFALSE;
   escape  = kFALSE;
   Int_t j = 0;
   Int_t i = 0;
   Int_t flip;

   for (i=0; i<nold; i++) {
      if( j >= 505) {
         Error("Text", "too many characters in input string (%d)", j);
         return;
      }
      if( escape) { escape = kFALSE; continue; }
      if( chars[i] == '{') curlybracket = kTRUE;
      if( chars[i] == '@') {
         for (flip=0; flip<9;flip++) {
            if (chars[i+1] == cflip[flip]) {
               newtext[j] = backslash;
               j++;
               strcpy(&newtext[j], cflipc[flip]);
               j = strlen(&newtext[0]);
               escape = kTRUE;
               break;
            }
         }
         if (escape) continue;
      }
      // in case of ntuple selections, escape ! and &
      if (curlybracket) {
         flip = 0;
         if (chars[i] == cflip[6]) flip = 6;
         if (chars[i] == cflip[8]) flip = 8;
         if (flip) {
            newtext[j] = backslash;
            j++;
            strcpy(&newtext[j], cflipc[flip]);
            j = strlen(&newtext[0]);
            continue;
         }
      }
      newtext[j] = chars[i];
      j++;
      newtext[j] = 0;
   }
   Int_t nchp = strlen(&newtext[0]);
//*-*
//*-*- Now a second parsing to search for the PostScript
//*-*-  characters (following a \) and ( , ), \  *******
//*-*
   Int_t iold = 1;
   Int_t inew = 0;
   Int_t nnew = 0;
   Int_t knew = 0;
//--------------------------------------------------------------------------
while (iold <= nchp ) {  //*-* loop on nchp old characters and look for backslash
//*-*
//*-*-              1. find a backslash
//*-*                  ================
//*-* if this backslash is not the last character of the string, then
//*-* study the character following this backslash
//*-*
   char2[inew] = 0;
   if (newtext[iold-1] == backslash)  {
      if ( iold == nchp) goto L60;
//*-*
//*-*-  1.1  the character following this backslash is also a backslash
//*-*
      if (newtext[iold] == backslash)  {
         char2[inew] = backslash;  //*-*  copy both backslashes
         inew++;
         char2[inew] = backslash;
         inew++;
         iold++;                 //*-* and go to the next character
         goto LOOPEND;
      }
//*-*
//*-*-  1.2  the character following this backslash is a parenthesis: ( or )
//*-*
      if ( newtext[iold] == '('  || newtext[iold] == ')')  {
         char2[inew] = backslash;        //*-* copy the backslash and the parenthesis
         inew++;
         char2[inew] = newtext[iold];
         inew++;
         iold++;                         //*-* and go to the following character
         goto LOOPEND;
      }
//*-*
//*-*-  1.3  the character following this backslash is also a special PostScript character:
//*-*      \n    linefeed (newline)
//*-*      \r    carriage return
//*-*      \t    horizontal tab
//*-*      \f    form feed
//*-*
      if (newtext[iold] == 'n'  ||
          newtext[iold] == 'r'  ||
          newtext[iold] == 't'  ||
          newtext[iold] == 'f')  {
          iold++;   //*-* copy nothing and go to the following character
          goto LOOPEND;
       }
//*-*
//*-*-  1.4  the character following this backslash is the special PostScript character:
//*-*      \b    back space
//*-*
      if ( newtext[iold] == 'b')  {
         char2[inew]='&';     //*-* replace the sequence \b by the & escape character
         inew++;
         iold++;              //*-*  and forget the b
         goto LOOPEND;
      }
//*-*
//*-*-  1.5  the character following this backslash is a digit between 0 and 7,
//*-*-  which means that the input text contains a string like \123 where
//*-*-  123 is an octal number the accepted ranges are 40-176 and 241-376
//*-*-  ( all others are ASCII control characters )
//*-*
//*-*-    =>   first, study the range 40-77 (case of 2 digits after the \)
//*-*
      Int_t k;
      for ( k=40; k<78; k++) {
         sprintf(&pstemp[0],"%d", k);
         Int_t iadd = 0;
         if( strncmp(&pstemp[0], &newtext[iold], 2)) {
            if( newtext[iold] != '0' || strncmp(&pstemp[0], &newtext[iold+1], 2)) continue;
            iadd = 1;
         }
         char2[inew]   = backslash; //*-* OK:copy the backslash and the 2 following digits and add a 0
         char2[inew+1] = '0';
         strncpy(&char2[inew+2], &newtext[iold+iadd], 2);
         inew += 4;
         iold += 2+iadd;           //*-* and go parsing the next following old character
         goto LOOPEND;
      }
//*-*
//*-*-    =>   then, study the ranges  100-177 and 241-377
//*-*-           (case of 3 digits after the backslash)
      for (k=100; k<378; k++) {
         if( k >= 178  &&  k <= 240) continue;
         sprintf(&pstemp[0],"%d", k);
         if( !strncmp(&pstemp[0], &newtext[iold], 3) ) {
            strncpy(&char2[inew], &newtext[iold-1],4);  //*-* OK:  copy the backslash and the 3 following digits
            inew += 4;
            iold += 3;   //*-*  and go parsing the next following old character
            goto LOOPEND;
         }
      }
L60:
//*-*
//*-*-  1.6 this backslash is followed by nothing understandable in PostScript,
//*-*-   it is an "isolated backslash" which will appear as \\.  Copy two backslashes
//*-*
      char2[inew] = backslash;
      inew++;
      char2[inew] = backslash;
      inew++;
      goto LOOPEND;
   }
//*-*
//*-*- 2. find ( or ) not preceeded by a backslash : include one backslash
//*-*
   else if( newtext[iold-1] == '(' ||  newtext[iold-1] == ')')  {
      if( i == 1 || newtext[iold-2] != backslash)  {
         char2[inew] = backslash;
         inew++;
         char2[inew] = newtext[iold-1];
         inew++;
         goto LOOPEND;
      }
      goto LOOPEND;
   }
//*-*- 3. treat normal text
   else {
      char2[inew] = newtext[iold-1];
      inew++;
   }
LOOPEND:
   iold++;
   char2[inew] = 0;
}


//*-*- now a third parsing to cut the text into pieces
//*-*- for each piece of text, I define
//*-*-      a. the string content = PIECE(I)
//*-*-      b. the font # = IFNB(I)= font: roman, 12: greek ,
//*-*-                         14: ZapfdingBats
//*-*-      c. the font size = IFNS(I)
//*-*-      d. a level flag = LEVEL(I) = 1: normal
//*-*-                                   2: superscript
//*-*-                                   3: subscript
//*-*-      e. a "backward" flag = LBACK(I) = 0: normal text ,
//*-*-                                      = 1: superscript and
//*-*-                                           subscript start at
//*-*-                                           the same x
//*-*-                                      = -n: for n backspaces
//*-*
   char * kl = &kpiece[0];
   for (i=0;i<npiece;i++) {
      piece[i] = kl;
      strcpy(kl," ");
      kl += kline;
      ifns[i]  = 0;
      ifnb[i]  = 0;
      level[i] = 0;
      lback[i] = 0;
   }

   roman       = kTRUE;
   greek       = kFALSE;
   special     = kFALSE;
   zapf        = kFALSE;
   superscript = kFALSE;
   subscript   = kFALSE;

   Int_t nt   = 0;       //*-* NT=number for pieces  of text
   Int_t ich  = 0;
   nchp      = strlen(&char2[0]);
while (ich < nchp ) {
   ich++;
   Int_t nbas = 0;
//*-*
//*-*  read character number I and check if it is a escape character
//*-*
   scape = 0;
   if (TestBit(kLatex)) scape = 0;
   else                 scape  = (char*)strchr(cflip, char2[ich-1]);
   if( scape )  {
      if ( *scape == cflip[0] ) {
             greek = kTRUE;       //*-*- find ` : go to greek
      }
      else  if( *scape == cflip[1] ) {
             special = kTRUE;     //*-*- find ' : go to special
      }
      else  if( *scape == cflip[2] ) {
             subscript   = kTRUE;   //*-*- find ? : go to subscript
             superscript = kFALSE;
      }
      else  if( *scape == cflip[3] ) {
             superscript = kTRUE;  //*-*- find ^ : go to superscript
             subscript   = kFALSE;
      }
      else  if( *scape == cflip[5] )  {
         roman   = kTRUE;    //*-*- find # : end of special, greek or of Zapf => go to roman
         zapf    = kFALSE;
         special = kFALSE;
         greek   = kFALSE;
      }
      else  if( *scape == cflip[6] )  {
         superscript = kFALSE;  //*-*- find ! : go to normal level of script
         subscript   = kFALSE;
      }
      else  if( *scape == cflip[7] ) {
             zapf = kTRUE; //*-*- find ~ : go to ZapfDingbats
      }
      else  if( *scape == cflip[8] )  {
         nbas = 1;                             //*-* find & : backspace is required
         for (j=i; j<nchp;j++) {               //*-* check if many backspaces are required:
            if( char2[j] != cflip[8] ) break;  //*-* compute how many consecutive backspaces
            nbas++;                            //*-* in the nchp-I remaining characters
         }
//*-*
//*-*- Since I have to backspace some text, (part of the preceeding piece),
//*-*- I define two pieces:
//*-*
//*-*- a. the string which follows normally the & ( i.e. up to the next
//*-*-    escape character)
//*-*
//*-*- b. the string to be backspaced, i.e. a part of the preceeding piece so
//*-*-    I create a new piece which is a copy the preceeding with LBACK<0
//*-*
         if (nt <= 0) return;
         lback[nt] = -nbas;       //*-* and the other parameters identical
         strcpy(piece[nt], piece[nt-1]);
         ifnb[nt]  = ifnb[nt-1];
         ifns[nt]  = ifns[nt-1];
         level[nt] = 0;  //*-* except the level, since the backspaced piece is not printed
         nt++;
         if( nbas > 1 )  { ich += nbas-1; continue; } //*-* however, in case of multiple backspaces,
      }                                               //*-* take the &&&...&&& as a whole
   }
   else {       //*-* the character is not a control character

//*-*- START of a new text: on the first character, or on the
//*-*-    first non escape char. which follows an escape char.
      if( ich != 1) scape = (char*)strchr(cflip, char2[ich-2]);
      if (TestBit(kLatex)) scape = 0;
      if( scape || ich == 1)  {
         ideb = ich;
         if( roman)       ifnb[nt]   = font;  //*-*- set font # (IFNB)
         if( greek)       ifnb[nt]   = 12;
         if( zapf)        ifnb[nt]   = 14;
         if( special)     ifnb[nt]   = 25;
         ifns[nt] = fontsize;                 //*-*- set font size (IFNS)
         if( superscript) ifns[nt]   = Int_t(0.5 + 0.7*fontsize);
         if( subscript)   ifns[nt]   = Int_t(0.5 + 0.7*fontsize);
         level[nt]   = 0;                     //*-*- set level flag (LEVEL)
         if( superscript) level[nt]  = Int_t(0.5 + fontsize/2);
         if( subscript)   level[nt]  = Int_t(0.5 - fontsize/3);
         nt++;
      }
      if( superscript || subscript) lback[nt-1] = 1;  //*-*set LBACK flag (1 for sub/uperscript)

//*-*- END of this text: on the last character or on
//*-*-      the last non escape which preceeds an escape (but
//*-*-      the terminating escape character itself is not known)
      if( ich != nchp) scape = (char*)strchr(cflip, char2[ich]);
      if (TestBit(kLatex)) scape = 0;
      if( scape || ich == nchp)  {
         Int_t ifin = ich;              //*-* compute text length and make one piece
         Int_t ilen = ifin - ideb + 1;  //*-* if length <74 and not 80, because of
         pc = piece[nt-1];              //*-* () and \040 on the PostScript file
         if( ilen < 74)  {
            strncpy(pc, &char2[ideb-1], ilen);
            pc[ilen] = 0;
            if( char2[ifin-1] == ' ') {  //*-* if the last character is ' '
               pc[ilen-1] = backslash;   //*-* it is replaced with \040
               strcpy(pc+ilen, "040");
            }
         }
         else {                   //*-* make several pieces if length > 74
            Int_t i1  = ideb;
            Int_t i2  = i1+73;
            Int_t nts = nt;
L120:        //*-* check if CHAR2 will not be cut in the middle of an octal code
            Int_t ib = 0;
            if (char2[i2-3] == backslash ) ib = 1;
            if (char2[i2-2] == backslash ) ib = 2;
            if (char2[i2-1] == backslash ) ib = 3;
            if (ib && i2-i1 == 73 && i2 != nchp) i2 += ib - 4;
            pc = piece[nt-1];                 //*-* copy CHAR2 in the piece number NT
            strncpy(pc, &char2[i1-1], i2-i1+1); //*-* with I2 readjusted
            pc[i2-i1+1] = 0;
            if( char2[i2-1] == ' ') {
               Int_t ilp  = strlen(pc);
               pc[ilp]    = backslash;        //*-* if the last character is ' '
               strcpy(pc+ilp+1, "040");       //*-* it is replaced with \040
            }
            if (i2 != ilen) {
               i1        = i2+1;
               if (ilen < i1+73) i2 = ilen;
               else              i2 = i1+73;
               ifnb[nt]  = ifnb[nts-1];
               ifns[nt]  = ifns[nts-1];
               level[nt] = level[nts-1];
               lback[nt] = lback[nts-1];
               nt++;
               goto L120;
            }
         }
      }
   }
}
//---------------------------------------------------------------------------
//*-*
//*-*- Finally, a fourth parsing for 3 reasons:
//*-*
   for (i=0; i<nt; i++) {
//*-*-** 1. LEVEL of sub/superscript after a multiple backsp. text:
//*-*-      one has:
//*-*-      i-2: text normally output lback=0
//*-*-      i-1 : text following in superscript mode
//*-*-      i : part of the preceeding (not printed) in which one
//*-*-          computes the backspace
//*-*-      i+1: text following the backspace
//*-*-      =>  since PIECE(i-1) and PIECE(I+1) are superposed;
//*-*-      I increase the level such that LEVEL(I-1)=IFNS(I-1)

      if( lback[i] < -1)  {
         if ( i > 0 ) {
            if( level[i-1] > 0) level[i-1] =  ifns[i-1];    //*-*   superscript
            if( level[i-1] < 0) level[i-1] = -ifns[i-1];    //*-*   subscript
         }
      }

//*-*-*** 2. LEVEL of sub/ superscript after ONE backspaced text:
//*-*-   put the LEVEL to +(current font size) for superscript
//*-*-    and to - (current font size) for subscript
      if ( i > 0 ) {
         if( lback[i-1] == -1)  {
            if( level[i] > 0) level[i] = ifns[i];  //*-*   superscript
            if( level[i] < 0) level[i] =-ifns[i];  //*-*   subscript
         }
      }

//*-*-** 3. correct in the greek text the 4 characters in the /Symbol font
//*-*- which are not " at their correct place" w.r.t. the HIGZ official table
      if( ifnb[i] == 12)  {
         pc = piece[i];
         for ( j=0;j<(int)strlen(pc);j++) {
            //if(      pc[j] == 'J')  pc[j]='I';
            //else if( pc[j] == 'V')  pc[j]='C';
          //  else if( pc[j] == 'C')  pc[j]='H';
          //  else if( pc[j] == 'H')  pc[j]='C';
            //else if( pc[j] == 'j')  pc[j]='i';
            //else if( pc[j] == 'v')  pc[j]='c';
          //  else if( pc[j] == 'c')  pc[j]='h';
          //  else if( pc[j] == 'h')  pc[j]='c';
         }
      }
   }
//*-*
//*-* write PS
//*-* position of text from arguments
//*-*
   Int_t psangle = Int_t(0.5 + fTextAngle);

   PrintStr("@");

//*-*- 1. text is left aligned
   if( txalh <= 1)  {
      SaveRestore(1);
      WriteInteger(XtoPS(x));
      WriteInteger(YtoPS(y));
      PrintStr("@");
      sprintf(lunps, " t %d r 0 0 m", psangle);
      PrintStr(lunps);
   }
//*-*
//*-*-   2. the text is centered or right-adjusted => compute the whole length
//*-*
   else if( txalh == 2 || txalh == 3)  {

      PrintStr(" /xs 0 def ");//*-* initialize variable containing the string length

      ipiece = 0;
L170:
      ipiece++;             //*-* loop on all pieces and add the length of each piece
      if( ipiece > nt) goto L250;
//*-*
//*-*-   2.1. ONE bakspaced text: forget the piece to be backspaced
//*-*-        and the piece which follows
//*-*
      if( lback[ipiece-1] == -1)  {
         ipiece++;
         goto L170;
      }
//*-*
//*-*-   2.2  backspaced text by more than one backspace
//*-*
      pc    = piece[ipiece-1];
      fnb   = ifnb[ipiece-1];
      psfnb = psfont[fnb-1];
      if( lback[ipiece-1] < -1)  {
         sprintf(lunps, "(%s)", pc);
         PrintStr(lunps);
         sprintf(lunps, " %d %s%d stwb ", abs(lback[ipiece-1]), psfnb, ifns[ipiece-1]);
         PrintStr(lunps);
         goto L170;
      }
//*-*
//*-*-  2.3  many superscript and many subscript at the same x
//*-*
      if( lback[ipiece-1] == 1 && lback[ipiece] == 1)  {
         n1 = 0;
         n2 = 0;
         for (j=ipiece-1;j<nt;j++) {               //*-* loop on pieces,
            if( lback[j] != 1) break;              //*-* computes how many pieces with LBACK=1
            if( level[j] == level[ipiece-1]) n1++; //*-* and check if they are all at the same level
            else           n2++;                   //*-* if yes, this is "standard" text
         }
         if( n1 == 0 || n2 == 0) goto L240;
//*-*
//*-*-      since many fonts are possible in sub/superscript, we output all
//*-*-      the pieces in super/subscript then thoses in sub/superscript
//*-*
         PrintStr(" /s1 0 def ");       //*-*- a) first pieces subscript or superscript
         for (j=ipiece-1;j<nt;j++) {
            if( level[j] != level[ipiece-1]) break;
            Int_t fnbj    = ifnb[j];
            const char * psfnbj = psfont[fnbj-1];
            sprintf(lunps, " %s findfont %d sf",  psfnbj, ifns[ipiece-1]);
            PrintStr(lunps);
            sprintf(lunps," (%s) sw pop s1 add /s1 exch def", piece[j]);
            PrintStr(lunps);
            knew = j+1;
         }

         PrintStr(" /s2 0 def ");     //*-*- b) then superscript or subscript
         knew++;
         for ( j=knew-1;j<nt;j++) {
            if( level[j] != level[knew-1]) break;
            Int_t fnbj    = ifnb[j];
            const char * psfnbj = psfont[fnbj-1];
            sprintf(lunps, " %s findfont %d sf",  psfnbj, ifns[ipiece-1]);
            PrintStr(lunps);
            sprintf(lunps," (%s) sw pop s2 add /s2 exch def", piece[j]);
            PrintStr(lunps);
            nnew = j+1;
          }
//*-*
//*-*- between subscript and superscript, which one is the longest?
         PrintStr(" s1 s2 ge { xs s1 add /xs exch def} { xs s2 add /xs exch def} ifelse ");
//*-*
         ipiece = nnew;       //*-*- since many pieces are treated
         goto L170;           //*-*  :increase piece counter accordingly
      }
//*-*
//*-*- 2.4. "standard" text:
//*-*
L240:
      if( lback[ipiece-1] == 0 || lback[ipiece-1] == 1)  {
         pc    = piece[ipiece-1];
         fnb   = ifnb[ipiece-1];
         psfnb = psfont[fnb-1];
         sprintf(lunps,"(%s)", pc);
         PrintStr(lunps);
         sprintf(lunps," %s %d stwn ", psfnb, ifns[ipiece-1]);
         PrintStr(lunps);
      }
      goto L170;
L250:

      if( txalh == 2)  {          //*-*- Centered text
         SaveRestore(1);
         WriteInteger(XtoPS(x));
         WriteInteger(YtoPS(y));
         PrintStr("@");
         sprintf(lunps," t %d r xs 2 div neg 0 t 0 0 m", psangle);
         PrintStr(lunps);
      }
      else if( txalh == 3)  {    //*-*- Right adjusted text
         SaveRestore(1);
         WriteInteger(XtoPS(x));
         WriteInteger(YtoPS(y));
         PrintStr("@");
         sprintf(lunps," t %d r  xs neg 0 t 0 0 m", psangle);
         PrintStr(lunps);
      }
   }

   ipiece = 0;
L260:                    //*-*-   now output the pieces
   ipiece++;
   pc    = piece[ipiece-1];
   fnb   = ifnb[ipiece-1];
   psfnb = psfont[fnb-1];
   if( ipiece > nt) goto L340;
//*-*
//*-*- make the PostScript file:
//*-*
//*-*- 1. ONE bakspace: output "piece" to be backspaced AND following piece
//*-*-    first save current graphic state, compute backward distance,
//*-*-       and move to that point
//*-*
   if( lback[ipiece-1] == -1)  {
      SaveRestore(1);
      sprintf(lunps,"%s findfont %d sf ", psfnb, ifns[ipiece-1]);
      PrintStr(lunps);
      sprintf(lunps,"(%s)", pc);
      PrintStr(lunps);
      PrintStr(" dup length 1 sub 1 getinterval ");
      sprintf(lunps," stringwidth pop 2 div neg %d rm ", level[ipiece-1]);
      PrintStr(lunps);

//*-*- then, text following one backspace: backspace also 1/2 of text
//*-*-   ( normally one character) print and restore preceeding graphic state
      Int_t fnb1   = ifnb[ipiece];
      const char *psfnb1 = psfont[fnb1-1];
      sprintf(lunps," %s findfont %d sf 0 %d rm ",  psfnb1, ifns[ipiece], level[ipiece]);
      PrintStr(lunps);
      sprintf(lunps,"(%s)", piece[ipiece]);
      PrintStr(lunps);
      PrintStr(" stringwidth pop 2 div neg 0 rm ");
      sprintf(lunps,"(%s)", piece[ipiece]);
      PrintStr(lunps);
      if( show) PrintStr(" oshow");
      else      PrintStr(" show");
      SaveRestore(-1);
      ipiece++;    //*-* since two pieces are treated increase piece counter
      goto L260;
   }
//*-*
//*-*-   2. Many Backspaces
//*-*
   if( lback[ipiece-1] < -1)  {
//*-*       first, protect against a number of bakspaces larger than
//*-*       the total number of characters in the string to be
//*-*       backspaced
      sprintf(lunps," %d /nbas exch def ", abs(lback[ipiece-1]));
      PrintStr(lunps);
      sprintf(lunps," (%s)", pc);
      PrintStr(lunps);
      PrintStr(" length /tlen exch def nbas tlen gt { /nbas tlen def } if ");
      sprintf(lunps, " %s findfont %d sf",  psfnb, ifns[ipiece-1]);
      PrintStr(lunps);
      sprintf(lunps,"(%s)", pc);
      PrintStr(lunps);
      PrintStr(" dup length nbas sub nbas getinterval stringwidth pop neg 0 t ");
      goto L260;
   }
//*-*
//*-*-  3.  superscript and subscript at the same x
   if( lback[ipiece-1] == 1 && lback[ipiece] == 1)  {
      n1=0;
      n2=0;
      for (j=ipiece-1; j<nt; j++) {                //*-* loop on pieces,
         if( lback[j] != 1) break;               //*-* computes how many pieces with LBACK=1
         if( level[j] == level[ipiece-1])  n1++; //*-* if yes, this is "standard" text
         else           n2++;
      }
      if( n1 == 0 || n2 == 0) goto L330;
//*-*
//*-*-      since many fonts are possible in sub/superscript, we output all
//*-*-      the pieces in super/subscript then thoses in sub/superscript
//*-*
//*-*-   a) first pieces subscript or superscript
      SaveRestore(1);
      SaveRestore(1);
      sprintf(lunps," 0 %d rm ", level[ipiece-1]);
      PrintStr(lunps);

      for ( j=ipiece-1; j<nt; j++) {
         if( level[j] != level[ipiece-1]) break;
         Int_t fnb1 = ifnb[j];
         const char *psfnb1 = psfont[fnb1-1];
         sprintf(lunps, " %s findfont %d sf",  psfnb1, ifns[ipiece-1]);
         PrintStr(lunps);
         sprintf(lunps,"(%s)", piece[j]);
         PrintStr(lunps);
         if( show) PrintStr(" dup oshow  true charpath currentpoint pop /s1 exch def");
         else      PrintStr(" show currentpoint pop /s1 exch def");
         knew = j + 1;
       }
      SaveRestore(-1);

//*-*-   b) then superscript or subscript
      knew++;

      sprintf(lunps," 0 %d rm ", level[knew-1]);
      PrintStr(lunps);
      for ( j=knew-1; j<nt; j++) {
         if( level[j] != level[knew-1] ) break;
         Int_t fnb1 = ifnb[j];
         const char *psfnb1 = psfont[fnb1-1];
         sprintf(lunps, " %s findfont %d sf",  psfnb1, ifns[ipiece-1]);
         PrintStr(lunps);
         sprintf(lunps,"(%s)", piece[j]);
         PrintStr(lunps);
         if( show) PrintStr(" dup oshow  true charpath currentpoint pop /s2 exch def");
         else      PrintStr(" show currentpoint pop /s2 exch def");
         nnew = j + 1;
       }

      SaveRestore(-1);
//*-*
//*-*- at which x- value, should one translate the current state?
      PrintStr(" s1 s2 ge {s1 0 t} {s2 0 t} ifelse ");
      ipiece = nnew;        //*-* since many pieces are treated
      goto L260;            //*-* :increase piece counter accordingly
   }
//*-*
//*-*- 4. "standard" text: output current "piece" of text
//*-*
L330:
   if( lback[ipiece-1] == 0 || lback[ipiece-1] == 1)  {
      fnb   = ifnb[ipiece-1];
      psfnb = psfont[fnb-1];
      sprintf(lunps," %s findfont %d sf 0 %d m ", psfnb, ifns[ipiece-1],level[ipiece-1]);
      PrintStr(lunps);
      sprintf(lunps,"(%s)", piece[ipiece-1]);
      PrintStr(lunps);
      if( show)  {
         PrintStr(" dup oshow");
//*-* move currentpoint ( if not last piece of text)
         if( ipiece != nt) PrintStr(" true charpath currentpoint pop 0 t ");
      }
      else {
         PrintStr(" show ");
//*-* move currentpoint ( if not last piece of text)
         if( ipiece != nt) PrintStr(" currentpoint pop 0 t ");
      }
      goto L260;
   }

L340:              //*-*-  end of loop on pieces

   SaveRestore(-1); //*-* save current graphic state after
                    //*-* the last piece of text

}


//______________________________________________________________________________
void TPostScript::TextNDC(Double_t u, Double_t v, const char *chars)
{
//*-*-*-*-*-*-*-*-*-*-*Write a string of characters in NDC*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================================

   Double_t x = gPad->GetX1() + u*(gPad->GetX2() - gPad->GetX1());
   Double_t y = gPad->GetY1() + v*(gPad->GetY2() - gPad->GetY1());
   Text(x, y, chars);
}

//______________________________________________________________________________
Int_t TPostScript::UtoPS(Double_t u)
{
//*-*-*-*-*-*-*-*Convert U from NDC coordinate to PostScript*-*-*-*-*-*-*-*-*
//*-*            ===========================================

   Double_t cm = fXsize*(gPad->GetAbsXlowNDC() + u*gPad->GetAbsWNDC());
   return Int_t(0.5 + 288*cm/2.54);
}


//______________________________________________________________________________
Int_t TPostScript::VtoPS(Double_t v)
{
//*-*-*-*-*-*-*-*Convert V from NDC coordinate to PostScript*-*-*-*-*-*-*-*-*
//*-*            ===========================================

   Double_t cm = fYsize*(gPad->GetAbsYlowNDC() + v*gPad->GetAbsHNDC());
   return Int_t(0.5 + 288*cm/2.54);
}

//______________________________________________________________________________
void TPostScript::WriteInteger(Int_t n)
{
//*-*-*-*-*-*-*-*-*-*-*Write one Integer to the PostScript file*-*-*-*-*-*-*-*
//*-*                  =========================================

   char str[15];
   sprintf(str," %d", n);
   PrintStr(str);
}

//______________________________________________________________________________
void TPostScript::WriteReal(Float_t z)
{
//*-*-*-*-*-*-*-*Write a Real number to the PostScript file*-*-*-*-*-*-*-*-*-*
//*-*            ===========================================

   char str[15];
   sprintf(str," %g", z);
   PrintStr(str);
}

//______________________________________________________________________________
Int_t TPostScript::XtoPS(Double_t x)
{
//*-*-*-*-*-*-*-*Convert X from world coordinate to PostScript*-*-*-*-*-*-*-*-*
//*-*            =============================================

   Double_t u = (x - gPad->GetX1())/(gPad->GetX2() - gPad->GetX1());
   return  UtoPS(u);
}

//______________________________________________________________________________
Int_t TPostScript::YtoPS(Double_t y)
{
//*-*-*-*-*-*-*-*Convert Y from world coordinate to PostScript*-*-*-*-*-*-*-*-*
//*-*            =============================================

   Double_t v = (y - gPad->GetY1())/(gPad->GetY2() - gPad->GetY1());
   return  VtoPS(v);
}

//______________________________________________________________________________
void TPostScript::Zone()
{
//*-*-*-*-*-*-*-*-*-*-*Initialize the PostScript page in zones*-*-*-*-*-*-*-*-*
//*-*                  ========================================

  if( !fClear )return;
  fClear = kFALSE;
//*-* When Zone has been called, fZone is TRUE
  fZone = kTRUE;

  if( fIYzone > fNYzone) {
     fIYzone=1;
     if( fMode != 3) {
        PrintStr("@showpage");
        SaveRestore(-1);
        fNpages++;
        PrintStr("@%%Page: number ");
        WriteInteger(fNpages);
        PrintStr("@");
     }
     else {
        PrintFast(9," showpage");
        SaveRestore(-1);
     }
  }
//*-*
//*-*-              No grestore the first time
//*-*
  if( fMode != 3) {
     if( fIXzone != 1 || fIYzone != 1) SaveRestore(-1);
     SaveRestore(1);
     PrintStr("@");
     WriteInteger(fIXzone);
     WriteInteger(fIYzone);
     PrintFast(5," Zone");
     PrintStr("@");
     fIXzone++;
     if( fIXzone > fNXzone) { fIXzone=1; fIYzone++; }
  }
//*-*
//*-*-              Picture Initialisation
//*-*
  SaveRestore(1);
  PrintFast(5," 0 0 t");
  fRed     = -1;
  fGreen   = -1;
  fBlue    = -1;
  fPrinted = kFALSE;
  fLineColor  = -1;
  fLineStyle  = -1;
  fLineWidth  = -1;
  fFillColor  = -1;
  fFillStyle  = -1;
  fMarkerSizeCur = -1;
}
