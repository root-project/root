// @(#)root/hist:$Name:  $:$Id: TSpectrum.cxx,v 1.16 2003/08/23 00:08:12 rdm Exp $
// Author: Miroslav Morhac   27/05/99

/////////////////////////////////////////////////////////////////////////////
//   THIS CLASS CONTAINS ADVANCED SPECTRA PROCESSING FUNCTIONS.            //
//                                                                         //
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                       //
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                       //
//   ONE-DIMENSIONAL SMOOTHING FUNCTIONS                                   //
//   TWO-DIMENSIONAL SMOOTHING FUNCTIONS                                   //
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTIONS                               //
//   TWO-DIMENSIONAL DECONVOLUTION FUNCTIONS                               //
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTIONS                                 //
//   TWO-DIMENSIONAL PEAK SEARCH FUNCTIONS                                 //
//   ONE-DIMENSIONAL PEAKS FITTING FUNCTIONS                               //
//   TWO-DIMENSIONAL PEAKS FITTING FUNCTIONS                               //
//   ONE-DIMENSIONAL ORTHOGONAL TRANSFORMS FUNCTIONS                       //
//   TWO-DIMENSIONAL ORTHOGONAL TRANSFORMS FUNCTIONS                       //
//                                                                         //
//   These functions were written by:                                      //
//   Miroslav Morhac                                                       //
//   Institute of Physics                                                  //
//   Slovak Academy of Sciences                                            //
//   Dubravska cesta 9, 842 28 BRATISLAVA                                  //
//   SLOVAKIA                                                              //
//                                                                         //
//   email:fyzimiro@savba.sk,    fax:+421 7 54772479                       //
//                                                                         //
//  The original code in C has been repackaged as a C++ class by R.Brun    //                        //
//                                                                         //
//  The algorithms in this class have been published in the following      //
//  references:                                                            //
//   [1]  M.Morhac et al.: Background elimination methods for              //
//   multidimensional coincidence gamma-ray spectra. Nuclear               //
//   Instruments and Methods in Physics Research A 401 (1997) 113-         //
//   132.                                                                  //
//                                                                         //
//   [2]  M.Morhac et al.: Efficient one- and two-dimensional Gold         //
//   deconvolution and its application to gamma-ray spectra                //
//   decomposition. Nuclear Instruments and Methods in Physics             //
//   Research A 401 (1997) 385-408.                                        //
//                                                                         //
//   [3]  M.Morhac et al.: Identification of peaks in multidimensional     //
//   coincidence gamma-ray spectra. Submitted for publication in           //
//   Nuclear Instruments and Methods in Physics Research A.                //
//                                                                         //
//   These NIM papers are also available as Postscript files from:         //
//
/*
   ftp://root.cern.ch/root/SpectrumDec.ps.gz
   ftp://root.cern.ch/root/SpectrumSrc.ps.gz
   ftp://root.cern.ch/root/SpectrumBck.ps.gz
*/ 
//
/////////////////////////////////////////////////////////////////////////////
    
#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TMath.h"
#define PEAK_WINDOW 1024
    ClassImp(TSpectrum)  
//______________________________________________________________________________
TSpectrum::TSpectrum() :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{
   Int_t n = 100;
   fMaxPeaks = n;
   fPosition = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fResolution = 1;
   fHistogram = 0;
   fNPeaks = 0;
}


//______________________________________________________________________________
TSpectrum::TSpectrum(Int_t maxpositions, Float_t resolution) :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{
   
//  maxpositions:  maximum number of peaks
//  resolution:    determines resolution of the neighboring peaks
//                 default value is 1 correspond to 3 sigma distance
//                 between peaks. Higher values allow higher resolution
//                 (smaller distance between peaks.
//                 May be set later through SetResolution.
       Int_t n = TMath::Max(maxpositions, 100);
   fMaxPeaks = n;
   fPosition = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fHistogram = 0;
   fNPeaks = 0;
   SetResolution(resolution);
}


//______________________________________________________________________________
    TSpectrum::~TSpectrum() 
{
   delete[]fPosition;
   delete[]fPositionX;
   delete[]fPositionY;
   delete fHistogram;
}


//______________________________________________________________________________
const char *TSpectrum::Background(TH1 * h, int number_of_iterations,
                                  Option_t * option) 
{
   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   This function calculates background spectrum from source in h.        //
//   The result is placed in the vector pointed by spectrum pointer.       //
//                                                                         //
//   Function parameters:                                                  //
//   spectrum:  pointer to the vector of source spectrum                   //
//   size:      length of spectrum and working space vectors               //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       printf
       ("Background function not yet implemented: h=%s, iter=%d, option=%sn"
        , h->GetName(), number_of_iterations, option);
   return 0;
}


//______________________________________________________________________________
Int_t TSpectrum::Search(TH1 * hin, Double_t sigma, Option_t * option, Double_t threshold) 
{
   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   This function searches for peaks in source spectrum in hin            //
//   The number of found peaks and their positions are written into        //
//   the members fNpeaks and fPositionX.                                   //
//                                                                         //
//   Function parameters:                                                  //
//   hin:       pointer to the histogram of source spectrum                //
//   sigma:   sigma of searched peaks, for details we refer to manual      //
//   threshold: (default=0.05)  peaks with amplitude less than             //
//       threshold*highest_peak are discarded.                             //
//                                                                         //
//   if option is not equal to "goff" (goff is the default), then          //
//   a polymarker object is created and added to the list of functions of  //
//   the histogram. The histogram is drawn with the specified option and   //
//   the polymarker object drawn on top of the histogram.                  //
//   The polymarker coordinates correspond to the npeaks peaks found in    //
//   the histogram.                                                        //
//   A pointer to the polymarker object can be retrieved later via:        //
//    TList *functions = hin->GetListOfFunctions();                        //
//    TPolyMarker *pm = (TPolyMarker*)functions->FindObject("TPolyMarker") //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       
   if (hin == 0) return 0;
   Int_t dimension = hin->GetDimension();
   if (dimension > 2) {
      Error("Search", "Only implemented for 1-d and 2-d histograms");
      return 0;
   }
   if (dimension == 1) {
      Int_t size = hin->GetXaxis()->GetNbins();
      Int_t i, bin, npeaks;
      Float_t * source = new float[size];
      Float_t * dest   = new float[size];
      for (i = 0; i < size; i++) source[i] = hin->GetBinContent(i + 1);

      npeaks = Search1HighRes(source, dest, size, sigma, 100*threshold, kTRUE, 3, kTRUE, 3);

      //TH1 * hnew = (TH1 *) hin->Clone("markov");
      //for (i = 0; i < size; i++)
      //   hnew->SetBinContent(i + 1, source[i]);
      if (strstr(option, "goff"))
         return npeaks;
      for (i = 0; i < npeaks; i++) {
         bin = 1 + Int_t(fPositionX[i] + 0.5);
         fPositionX[i] = hin->GetBinCenter(bin);
         fPositionY[i] = hin->GetBinContent(bin);
      }
      delete [] source;
      delete [] dest;
      
      if (strstr(option, "goff"))
         return npeaks;
      TPolyMarker * pm = (TPolyMarker*)hin->GetListOfFunctions()->FindObject("TPolyMarker");
      if (pm) {
         hin->GetListOfFunctions()->Remove(pm);
         delete pm;
      }
      pm = new TPolyMarker(npeaks, fPositionX, fPositionY);
      hin->GetListOfFunctions()->Add(pm);
      pm->SetMarkerStyle(23);
      pm->SetMarkerColor(kRed);
      pm->SetMarkerSize(1.3);
      hin->Draw(option);
      return npeaks;
   }
   return 0;
}


//______________________________________________________________________________
void TSpectrum::SetResolution(Float_t resolution) 
{
   
//  resolution: determines resolution of the neighboring peaks
//              default value is 1 correspond to 3 sigma distance
//              between peaks. Higher values allow higher resolution
//              (smaller distance between peaks.
//              May be set later through SetResolution.
       if (resolution > 1)
      fResolution = resolution;
   
   else
      fResolution = 1;
}


//_____________________________________________________________________________
//_____________________________________________________________________________
    
/////////////////////NEW FUNCTIONS  APRIL 2003
const char *TSpectrum::Background1(float *spectrum, int size,
                                     int number_of_iterations) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - INCREASING        */ 
/*                                CLIPPING WINDOW			   */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed in the vector pointed by spectrum pointer.    */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the vector of source spectrum		   */ 
/*	size-length of spectrum and working space vectors		   */ 
/*	number_of_iterations, for details we refer to manual		   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
       
//BEGIN_HTML <!--
/* -->
<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta http-equiv=Content-Type content="text/html; charset=windows-1251">
<meta name=ProgId content=Word.Document>
<meta name=Generator content="Microsoft Word 9">
<meta name=Originator content="Microsoft Word 9">
<link rel=File-List href="./Background1_files/filelist.xml">
<title>-BACKGROUND ELIMINATION</title>
<!--[if gte mso 9]><xml>
 <o:DocumentProperties>
  <o:Author>Miroslav Morhac</o:Author>
  <o:Template>Normal</o:Template>
  <o:LastAuthor>fyzimiro</o:LastAuthor>
  <o:Revision>2</o:Revision>
  <o:TotalTime>241</o:TotalTime>
  <o:LastPrinted>2003-03-31T11:06:00Z</o:LastPrinted>
  <o:Created>2003-04-10T09:24:00Z</o:Created>
  <o:LastSaved>2003-04-10T09:24:00Z</o:LastSaved>
  <o:Pages>1</o:Pages>
  <o:Words>11679</o:Words>
  <o:Characters>66571</o:Characters>
  <o:Company>Inst. of Physics, Slovak Academy of Sciences</o:Company>
  <o:Lines>554</o:Lines>
  <o:Paragraphs>133</o:Paragraphs>
  <o:CharactersWithSpaces>81753</o:CharactersWithSpaces>
  <o:Version>9.2720</o:Version>
 </o:DocumentProperties>
</xml><![endif]--><!--[if gte mso 9]><xml>
 <w:WordDocument>
  <w:DisplayHorizontalDrawingGridEvery>0</w:DisplayHorizontalDrawingGridEvery>
  <w:DisplayVerticalDrawingGridEvery>0</w:DisplayVerticalDrawingGridEvery>
  <w:UseMarginsForDrawingGridOrigin/>
  <w:Compatibility>
   <w:FootnoteLayoutLikeWW8/>
   <w:ShapeLayoutLikeWW8/>
   <w:AlignTablesRowByRow/>
   <w:ForgetLastTabAlignment/>
   <w:LayoutRawTableWidth/>
   <w:LayoutTableRowsApart/>
  </w:Compatibility>
 </w:WordDocument>
</xml><![endif]-->
<style>
<!--
// Font Definitions
@font-face
	{font-family:Wingdings;
	panose-1:5 0 0 0 0 0 0 0 0 0;
	mso-font-charset:2;
	mso-generic-font-family:auto;
	mso-font-pitch:variable;
	mso-font-signature:0 268435456 0 0 -2147483648 0;}
 // Style Definitions
p.MsoNormal, li.MsoNormal, div.MsoNormal
	{mso-style-parent:"";
	margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
h1
	{mso-style-next:Normal;
	margin-top:12.0pt;
	margin-right:0in;
	margin-bottom:3.0pt;
	margin-left:0in;
	mso-pagination:widow-orphan;
	page-break-after:avoid;
	mso-outline-level:1;
	font-size:14.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:Arial;
	mso-bidi-font-family:"Times New Roman";
	mso-font-kerning:14.0pt;
	mso-bidi-font-weight:normal;}
h2
	{mso-style-next:Normal;
	margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	page-break-after:avoid;
	mso-outline-level:2;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-bidi-font-weight:normal;}
h3
	{mso-style-next:Normal;
	margin-top:12.0pt;
	margin-right:0in;
	margin-bottom:3.0pt;
	margin-left:0in;
	mso-pagination:widow-orphan;
	page-break-after:avoid;
	mso-outline-level:3;
	font-size:13.0pt;
	font-family:Arial;}
h4
	{mso-style-next:Normal;
	margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	page-break-after:avoid;
	mso-outline-level:4;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	font-weight:normal;}
p.MsoFootnoteText, li.MsoFootnoteText, div.MsoFootnoteText
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
p.MsoHeader, li.MsoHeader, div.MsoHeader
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	tab-stops:center 3.0in right 6.0in;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
p.MsoFooter, li.MsoFooter, div.MsoFooter
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	tab-stops:center 3.0in right 6.0in;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
span.MsoFootnoteReference
	{vertical-align:super;}
p.MsoBodyText, li.MsoBodyText, div.MsoBodyText
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
p.MsoBodyTextIndent, li.MsoBodyTextIndent, div.MsoBodyTextIndent
	{margin-top:0in;
	margin-right:0in;
	margin-bottom:0in;
	margin-left:.25in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
p.MsoBodyText2, li.MsoBodyText2, div.MsoBodyText2
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";
	font-weight:bold;
	mso-bidi-font-weight:normal;}
p.MsoBodyText3, li.MsoBodyText3, div.MsoBodyText3
	{margin:0in;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";
	font-weight:bold;
	mso-bidi-font-weight:normal;}
p.MsoBodyTextIndent2, li.MsoBodyTextIndent2, div.MsoBodyTextIndent2
	{margin:0in;
	margin-bottom:.0001pt;
	text-align:justify;
	text-indent:.5in;
	mso-pagination:widow-orphan;
	font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
p.MsoBodyTextIndent3, li.MsoBodyTextIndent3, div.MsoBodyTextIndent3
	{margin:0in;
	margin-bottom:.0001pt;
	text-indent:.5in;
	mso-pagination:widow-orphan;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";
	font-weight:bold;
	mso-bidi-font-weight:normal;}
p.MTDisplayEquation, li.MTDisplayEquation, div.MTDisplayEquation
	{mso-style-name:MTDisplayEquation;
	mso-style-parent:"Body Text Indent";
	margin:0in;
	margin-bottom:.0001pt;
	text-align:justify;
	line-height:150%;
	mso-pagination:widow-orphan;
	tab-stops:center 215.5pt right 431.0pt;
	font-size:12.0pt;
	mso-bidi-font-size:10.0pt;
	font-family:"Times New Roman";
	mso-fareast-font-family:"Times New Roman";}
@page Section1
	{size:595.35pt 842.0pt;
	margin:1.0in 89.85pt 1.0in 89.85pt;
	mso-header-margin:.5in;
	mso-footer-margin:.5in;
	mso-title-page:yes;
	mso-even-header:url("./Background1_files/header.htm") eh1;
	mso-header:url("./Background1_files/header.htm") h1;
	mso-even-footer:url("./Background1_files/header.htm") ef1;
	mso-footer:url("./Background1_files/header.htm") f1;
	mso-paper-source:0;}
div.Section1
	{page:Section1;}
 // List Definitions
@list l0
	{mso-list-id:24991170;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l0:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l1
	{mso-list-id:127162979;
	mso-list-type:hybrid;
	mso-list-template-ids:-1145955346;}
@list l1:level1
	{mso-level-tab-stop:.5in;
	mso-level-number-position:left;
	text-indent:-.25in;}
@list l2
	{mso-list-id:129250882;
	mso-list-template-ids:-1983590464;}
@list l2:level1
	{mso-level-tab-stop:.5in;
	mso-level-number-position:left;
	text-indent:-.25in;}
@list l2:level2
	{mso-level-number-format:alpha-lower;
	mso-level-tab-stop:1.0in;
	mso-level-number-position:left;
	margin-left:63.0pt;
	text-indent:-9.0pt;}
@list l3
	{mso-list-id:178593267;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l3:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l4
	{mso-list-id:245725304;
	mso-list-type:hybrid;
	mso-list-template-ids:-726902096;}
@list l4:level1
	{mso-level-tab-stop:.5in;
	mso-level-number-position:left;
	text-indent:-.25in;}
@list l5
	{mso-list-id:382556631;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l5:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l6
	{mso-list-id:429618661;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l6:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l7
	{mso-list-id:436873422;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l7:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l8
	{mso-list-id:454328312;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l8:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l9
	{mso-list-id:500900935;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l9:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l10
	{mso-list-id:531573629;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l10:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l11
	{mso-list-id:567233683;
	mso-list-template-ids:-1388549558;}
@list l11:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Symbol;}
@list l11:level2
	{mso-level-number-format:bullet;
	mso-level-text:o;
	mso-level-tab-stop:1.0in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:"Courier New";
	mso-bidi-font-family:"Times New Roman";}
@list l11:level3
	{mso-level-number-format:bullet;
	mso-level-text:\F0A7;
	mso-level-tab-stop:1.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Wingdings;}
@list l11:level4
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:2.0in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Symbol;}
@list l11:level5
	{mso-level-number-format:bullet;
	mso-level-text:o;
	mso-level-tab-stop:2.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:"Courier New";
	mso-bidi-font-family:"Times New Roman";}
@list l11:level6
	{mso-level-number-format:bullet;
	mso-level-text:\F0A7;
	mso-level-tab-stop:3.0in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Wingdings;}
@list l11:level7
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:3.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Symbol;}
@list l11:level8
	{mso-level-number-format:bullet;
	mso-level-text:o;
	mso-level-tab-stop:4.0in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:"Courier New";
	mso-bidi-font-family:"Times New Roman";}
@list l11:level9
	{mso-level-number-format:bullet;
	mso-level-text:\F0A7;
	mso-level-tab-stop:4.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Wingdings;}
@list l12
	{mso-list-id:642778214;
	mso-list-type:hybrid;
	mso-list-template-ids:185498508;}
@list l12:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.75in;
	mso-level-number-position:left;
	margin-left:.75in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l13
	{mso-list-id:661739637;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l13:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l14
	{mso-list-id:668169972;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l14:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l15
	{mso-list-id:696466787;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l15:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l16
	{mso-list-id:697974868;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l16:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l17
	{mso-list-id:738133466;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l17:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l18
	{mso-list-id:757675041;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l18:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l19
	{mso-list-id:819929221;
	mso-list-type:simple;
	mso-list-template-ids:1934940258;}
@list l19:level1
	{mso-level-number-format:alpha-lower;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;}
@list l20
	{mso-list-id:841815699;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l20:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l21
	{mso-list-id:946497998;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l21:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l22
	{mso-list-id:1008798662;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l22:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l23
	{mso-list-id:1059400825;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l23:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l24
	{mso-list-id:1068840417;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l24:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l25
	{mso-list-id:1099788556;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l25:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l26
	{mso-list-id:1106534387;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l26:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l27
	{mso-list-id:1119496346;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l27:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l28
	{mso-list-id:1141120086;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l28:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l29
	{mso-list-id:1151021739;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l29:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l30
	{mso-list-id:1204173880;
	mso-list-type:simple;
	mso-list-template-ids:67698703;}
@list l30:level1
	{mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;}
@list l31
	{mso-list-id:1217397408;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l31:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l32
	{mso-list-id:1262570285;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l32:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l33
	{mso-list-id:1265843418;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l33:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l34
	{mso-list-id:1268653809;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l34:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l35
	{mso-list-id:1279602481;
	mso-list-type:simple;
	mso-list-template-ids:1176016568;}
@list l35:level1
	{mso-level-number-format:alpha-lower;
	mso-level-tab-stop:.75in;
	mso-level-number-position:left;
	margin-left:.75in;
	text-indent:-.25in;}
@list l36
	{mso-list-id:1286422925;
	mso-list-type:hybrid;
	mso-list-template-ids:185498508;}
@list l36:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0A7;
	mso-level-tab-stop:.5in;
	mso-level-number-position:left;
	text-indent:-.25in;
	font-family:Wingdings;}
@list l37
	{mso-list-id:1337805030;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l37:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l38
	{mso-list-id:1411347952;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l38:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l39
	{mso-list-id:1450782188;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l39:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l40
	{mso-list-id:1526749020;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l40:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l41
	{mso-list-id:1561331700;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l41:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l42
	{mso-list-id:1715229423;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l42:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l43
	{mso-list-id:1777142054;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l43:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l44
	{mso-list-id:1870682292;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l44:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l45
	{mso-list-id:1893075106;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l45:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l46
	{mso-list-id:2004579635;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l46:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l47
	{mso-list-id:2060131503;
	mso-list-type:simple;
	mso-list-template-ids:67698689;}
@list l47:level1
	{mso-level-number-format:bullet;
	mso-level-text:\F0B7;
	mso-level-tab-stop:.25in;
	mso-level-number-position:left;
	margin-left:.25in;
	text-indent:-.25in;
	font-family:Symbol;}
@list l48
	{mso-list-id:2102682727;
	mso-list-type:hybrid;
	mso-list-template-ids:192675958;}
@list l48:level1
	{mso-level-tab-stop:1.0in;
	mso-level-number-position:left;
	margin-left:1.0in;
	text-indent:-.25in;}
ol
	{margin-bottom:0in;}
ul
	{margin-bottom:0in;}
-->
</style>
<!--[if gte mso 9]><xml>
 <o:shapedefaults v:ext="edit" spidmax="2050"/>
</xml><![endif]--><!--[if gte mso 9]><xml>
 <o:shapelayout v:ext="edit">
  <o:idmap v:ext="edit" data="1"/>
  <o:regrouptable v:ext="edit">
   <o:entry new="1" old="0"/>
  </o:regrouptable>
 </o:shapelayout></xml><![endif]-->
</head>

<body lang=EN-US style='tab-interval:.5in'>

<div class=Section1>

<p class=MsoNormal><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></p>

<p class=MsoNormal><b style='mso-bidi-font-weight:normal'>-1 DIMENSIONAL
SPECTRA <o:p></o:p></b></p>

<p class=MsoNormal><b style='mso-bidi-font-weight:normal'><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></b></p>

<p class=MsoNormal style='text-align:justify;text-indent:.5in'><span
style='font-size:12.0pt;mso-bidi-font-size:10.0pt'>This function calculates
background spectrum from source spectrum.<span style="mso-spacerun: yes"> 
</span>The result is placed in the vector pointed by spectrum pointer.<span
style="mso-spacerun: yes">  </span>On successful completion it returns 0. On error
it returns pointer to the string describing error. <o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:12.0pt;
mso-bidi-font-size:10.0pt'><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></span></p>

<p class=MsoBodyText><b style='mso-bidi-font-weight:normal'>char
*Background1(float *spectrum, int size, int number_of_iterations);<o:p></o:p></b></p>

<p class=MsoBodyText><b style='mso-bidi-font-weight:normal'><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></b></p>

<p class=MsoBodyTextIndent2 style='text-indent:0in'><span style='font-size:
12.0pt;mso-bidi-font-size:10.0pt'>Function parameters:<span style='mso-tab-count:
1'>     </span><span style='mso-tab-count:2'>                        </span><span
style='mso-tab-count:2'>                        </span><span style='mso-tab-count:
1'>            </span><span style="mso-spacerun: yes">   </span><o:p></o:p></span></p>

<p class=MsoNormal><span style='font-size:12.0pt;mso-bidi-font-size:10.0pt'>-spectrum-pointer
to the vector of source spectrum<span style='mso-tab-count:1'>          </span><span
style='mso-tab-count:1'>            </span><span style="mso-spacerun: yes">  
</span><o:p></o:p></span></p>

<p class=MsoNormal><span style='font-size:12.0pt;mso-bidi-font-size:10.0pt'>-size-length
of spectrum<span style='mso-tab-count:1'>          </span><span
style='mso-tab-count:1'>            </span><span style="mso-spacerun: yes">  
</span><o:p></o:p></span></p>

<p class=MsoBodyText>-number_of_iterations or width of the clipping window<span
style='mso-tab-count:1'>           </span><span style='mso-tab-count:1'>            </span><b
style='mso-bidi-font-weight:normal'><o:p></o:p></b></p>

<p class=MsoBodyText><b style='mso-bidi-font-weight:normal'><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></b></p>

<p class=MTDisplayEquation style='line-height:normal;tab-stops:.5in'>The function
allows to separate useless spectrum information (continuous background) from
peaks, based on Sensitive Nonlinear Iterative Peak Clipping Algorithm. In fact
it represents second order difference filter (-1,2,-1). The basic algorithm is
described in detail in [1], [2].</p>

<p class=MsoBodyText><![if !supportEmptyParas]>&nbsp;<![endif]><o:p></o:p></p>

<p class=MsoBodyText2>References:<span style='font-weight:normal'><o:p></o:p></span></p>

<p class=MsoBodyText2><span style='font-weight:normal'>[1] </span><span
lang=SK style='mso-ansi-language:SK;font-weight:normal'><span
style="mso-spacerun: yes"> </span>M. Morh&aacute;&#269;, J. Kliman, V.
Matou&#353;ek, M. Veselsk&yacute;, I. Turzo</span><span style='font-weight:
normal'>.: Background elimination methods for multidimensional gamma-ray
spectra. NIM, A401 (1997) 113-132.<o:p></o:p></span></p>

<p class=MsoBodyText2><span style='font-weight:normal'>[2]<span
style="mso-spacerun: yes">  </span>C. G Ryan et al.: SNIP, a
statistics-sensitive background treatment for the quantitative analysis of PIXE
spectra in geoscience applications. NIM, B34 (1988), 396-402.<o:p></o:p></span></p>

</div>

</body>

</html>
<img src=*gif/spectrum/Background1.jpg>

                        Fig1. 1D Spectrum with estimated background
<!--*/ 
// -->END_HTML
   int i, j;
   float a, b;
   if (size <= 0)
      return "Wrong Parameters";
   if (number_of_iterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (size < 2 * number_of_iterations + 1)
      return "Too Large Clipping Window";
   float *working_space = new float[size];
   for (i = 1; i <= number_of_iterations; i++) {
      for (j = i; j < size - i; j++) {
         a = spectrum[j];
         b = (spectrum[j - i] + spectrum[j + i]) / 2.0;
         if (b < a)
            a = b;
         working_space[j] = a;
      }
      for (j = i; j < size - i; j++)
         spectrum[j] = working_space[j];
   }
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum::Background1General(float *spectrum, int size,
                                          int number_of_iterations,
                                          int direction, int filter_order,
                                          bool compton) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - GENERAL FUNCTION  */ 
/*                                                                         */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed in the vector pointed by spectrum pointer.    */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the vector of source spectrum		   */ 
/*	size-length of spectrum vector        		                   */ 
/*	number_of_iterations-maximal width of clipping window,             */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK1_INCREASING_WINDOW                 */ 
/*                                 BACK1_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK1_ORDER2                          */ 
/*                                   BACK1_ORDER4                          */ 
/*                                   BACK1_ORDER6                          */ 
/*                                   BACK1_ORDER8	                   */ 
/*	compton- logical variable whether the estimation of Compton edge   */ 
/*               will be incuded                                           */ 
/*             - possible values=BACK1_EXCLUDE_COMPTON                     */ 
/*                               BACK1_INCLUDE_COMPTON                     */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, b1, b2, priz;
   float a, b, c, d, e, yb1, yb2, ai;
   if (size <= 0)
      return "Wrong Parameters";
   if (number_of_iterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (size < 2 * number_of_iterations + 1)
      return "Too Large Clipping Window";
   float *working_space = new float[2 * size];
   for (i = 0; i < size; i++)
      working_space[i + size] = spectrum[i];
   if (direction == BACK1_INCREASING_WINDOW) {
      if (filter_order == BACK1_ORDER2) {
         for (i = 1; i <= number_of_iterations; i++) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER4) {
         for (i = 1; i <= number_of_iterations; i++) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER6) {
         for (i = 1; i <= number_of_iterations; i++) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[size + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[size + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[size + j - (int) ai] / 20;
               d += 15 * working_space[size + j + (int) ai] / 20;
               d -= 6 * working_space[size + j + (int) (2 * ai)] / 20;
               d += working_space[size + j + (int) (3 * ai)] / 20;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER8) {
         for (i = 1; i <= number_of_iterations; i++) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[size + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[size + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[size + j - (int) ai] / 20;
               d += 15 * working_space[size + j + (int) ai] / 20;
               d -= 6 * working_space[size + j + (int) (2 * ai)] / 20;
               d += working_space[size + j + (int) (3 * ai)] / 20;
               e = 0;
               ai = i / 4;
               e -= working_space[size + j - (int) (4 * ai)] / 70;
               e += 8 * working_space[size + j - (int) (3 * ai)] / 70;
               e -= 28 * working_space[size + j - (int) (2 * ai)] / 70;
               e += 56 * working_space[size + j - (int) ai] / 70;
               e += 56 * working_space[size + j + (int) ai] / 70;
               e -= 28 * working_space[size + j + (int) (2 * ai)] / 70;
               e += 8 * working_space[size + j + (int) (3 * ai)] / 70;
               e -= working_space[size + j + (int) (4 * ai)] / 70;
               if (b < e)
                  b = e;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
   }
   
   else if (direction == BACK1_DECREASING_WINDOW) {
      if (filter_order == BACK1_ORDER2) {
         for (i = number_of_iterations; i >= 1; i--) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER4) {
         for (i = number_of_iterations; i >= 1; i--) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER6) {
         for (i = number_of_iterations; i >= 1; i--) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[size + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[size + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[size + j - (int) ai] / 20;
               d += 15 * working_space[size + j + (int) ai] / 20;
               d -= 6 * working_space[size + j + (int) (2 * ai)] / 20;
               d += working_space[size + j + (int) (3 * ai)] / 20;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
      
      else if (filter_order == BACK1_ORDER8) {
         for (i = number_of_iterations; i >= 1; i--) {
            for (j = i; j < size - i; j++) {
               a = working_space[size + j];
               b = (working_space[size + j - i] +
                     working_space[size + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[size + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[size + j - (int) ai] / 6;
               c += 4 * working_space[size + j + (int) ai] / 6;
               c -= working_space[size + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[size + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[size + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[size + j - (int) ai] / 20;
               d += 15 * working_space[size + j + (int) ai] / 20;
               d -= 6 * working_space[size + j + (int) (2 * ai)] / 20;
               d += working_space[size + j + (int) (3 * ai)] / 20;
               e = 0;
               ai = i / 4;
               e -= working_space[size + j - (int) (4 * ai)] / 70;
               e += 8 * working_space[size + j - (int) (3 * ai)] / 70;
               e -= 28 * working_space[size + j - (int) (2 * ai)] / 70;
               e += 56 * working_space[size + j - (int) ai] / 70;
               e += 56 * working_space[size + j + (int) ai] / 70;
               e -= 28 * working_space[size + j + (int) (2 * ai)] / 70;
               e += 8 * working_space[size + j + (int) (3 * ai)] / 70;
               e -= working_space[size + j + (int) (4 * ai)] / 70;
               if (b < e)
                  b = e;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }
            for (j = i; j < size - i; j++)
               working_space[size + j] = working_space[j];
         }
      }
   }
   if (compton == BACK1_INCLUDE_COMPTON) {
      for (i = 0, b2 = 0; i < size; i++) {
         b1 = b2;
         a = working_space[i], b = spectrum[i];
         j = i;
         if (TMath::Abs(a - b) >= 1) {
            b1 = i - 1;
            if (b1 < 0)
               b1 = 0;
            yb1 = spectrum[b1];
            for (b2 = b1 + 1, c = 0, priz = 0; priz == 0 && b2 < size;
                  b2++) {
               a = working_space[b2], b = spectrum[b2];
               c = c + b - yb1;
               if (TMath::Abs(a - b) < 1) {
                  priz = 1;
                  yb2 = b;
               }
            }
            if (b2 == size)
               b2 -= 1;
            yb2 = spectrum[b2];
            if (yb1 <= yb2) {
               for (j = b1, c = 0; j <= b2; j++) {
                  b = spectrum[j];
                  c = c + b - yb1;
               }
               if (c > 1) {
                  c = (yb2 - yb1) / c;
                  for (j = b1, d = 0; j <= b2 && j < size; j++) {
                     b = spectrum[j];
                     d = d + b - yb1;
                     a = c * d + yb1;
                     if (a < spectrum[j])
                        working_space[size + j] = a;
                  }
               }
            }
            
            else {
               for (j = b2, c = 0; j >= b1; j--) {
                  b = spectrum[j];
                  c = c + b - yb2;
               }
               if (c > 1) {
                  c = (yb1 - yb2) / c;
                  for (j = b2, d = 0; j >= b1 && j >= 0; j--) {
                     b = spectrum[j];
                     d = d + b - yb2;
                     a = c * d + yb2;
                     if (a < spectrum[j])
                        working_space[size + j] = a;
                  }
               }
            }
            i = b2;
         }
      }
   }
   for (j = 0; j < size; j++)
      spectrum[j] = working_space[size + j];
   delete[]working_space;
   return 0;
}


//_____________________________________________________________________________
const char* TSpectrum::Smooth1Markov(float *source, int size, int aver_window)
{
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL MARKOV SPECTRUM SMOOTHING FUNCTION                 */
/*                                              			   */
/*	This function calculates smoothed spectrum from source spectrum    */
/*      based on Markov chain method.                                      */
/*	The result is placed in the array pointed by source pointer.       */
/*									   */
/*	Function parameters:						   */
/*	source-pointer to the array of source spectrum   		   */
/*	size length of source array					   */
/*	aver_window-width of averaging smoothing window                	   */
/*									   */
/////////////////////////////////////////////////////////////////////////////
   int xmin, xmax, i, l;
   float a, b, maxch;
   float nom, nip, nim, sp, sm, plocha = 0;
   if(aver_window <= 0)
      return "Averaging Window must be positive";   
   float *working_space = new float[size];      
   xmin = 0,xmax = size - 1;
   for(i = 0, maxch = 0; i < size; i++){
      working_space[i]=0;
      if(maxch < source[i])
         maxch = source[i];
         
      plocha += source[i];
   }
   if(maxch == 0)
      return 0 ;
      
   nom = 1;
   working_space[xmin] = 1;
   for(i = xmin; i < xmax; i++){
      nip = source[i] / maxch;
      nim = source[i + 1] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= aver_window; l++){
         if((i + l) > xmax)
            a = source[xmax] / maxch;
            
         else
            a = source[i + l] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;
            
         else
       	    a = TMath::Sqrt(a + nip);            
	 b = b / a;
	 b = TMath::Exp(b);                        	                                                             
       	 sp = sp + b;
         if((i - l + 1) < xmin)
            a = source[xmin] / maxch;
            
         else
            a = source[i - l + 1] / maxch;
         b = a - nim;
         if(a + nim <= 0)
            a = 1;
            
         else
       	    a = TMath::Sqrt(a + nim);            
	 b = b / a;
	 b = TMath::Exp(b);                        	                                                                      
       	 sm = sm + b;
      }
      a = sp / sm;
      a = working_space[i + 1] = working_space[i] * a;
      nom = nom + a;
   }
   for(i = xmin; i <= xmax; i++){
      working_space[i] = working_space[i] / nom;
   }
   for(i = 0; i < size; i++)
      source[i] = working_space[i] * plocha;
   delete[]working_space;
   return 0;      
}

//_______________________________________________________________________________
const char *TSpectrum::Deconvolution1(float *source, const float *resp,
                                      int size, int number_of_iterations) 
{
   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   This function calculates deconvolution from source spectrum           //
//   according to response spectrum                                        //
//   The result is placed in the vector pointed by source pointer.         //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the vector of source spectrum                     //
//   res:     pointer to the vector of response spectrum                   //
//   size:    length of source and response spectra                        //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       if (size <= 0)
      return "Wrong Parameters";
   
       //   working_space-pointer to the working vector
       //   (its size must be 6*size of source spectrum)
   double *working_space = new double[6 * size];
   int i, j, k, lindex, posit, imin, imax, jmin, jmax, lh_gold;
   double lda, ldb, ldc, area, maximum;
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;
   
//read response vector
       for (i = 0; i < size; i++) {
      lda = resp[i];
      if (lda != 0)
         lh_gold = i + 1;
      working_space[i] = lda;
      area += lda;
      if (lda > maximum) {
         maximum = lda;
         posit = i;
      }
   }
   if (lh_gold == -1)
      return "ZERO RESPONSE VECTOR";
   
//read source vector
       for (i = 0; i < size; i++)
      working_space[2 * size + i] = source[i];
   
//create matrix at*a(vector b)
       i = lh_gold - 1;
   if (i > size)
      i = size;
   imin = -i, imax = i;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      jmin = 0;
      if (i < 0)
         jmin = -i;
      jmax = lh_gold - 1 - i;
      if (jmax > (lh_gold - 1))
         jmax = lh_gold - 1;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j];
         ldc = working_space[i + j];
         lda = lda + ldb * ldc;
      }
      working_space[size + i - imin] = lda;
   }
   
//create vector p
       i = lh_gold - 1;
   imin = -i;
   imax = size + i - 1;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      for (j = 0; j <= (lh_gold - 1); j++) {
         ldb = working_space[j];
         k = i + j;
         if (k >= 0 && k < size) {
            ldc = working_space[2 * size + k];
            lda = lda + ldb * ldc;
         }
      }
      working_space[4 * size + i - imin] = lda;
   }
   
//move vector p
       for (i = imin; i <= imax; i++)
      working_space[2 * size + i - imin] =
          working_space[4 * size + i - imin];
   
//create at*a*at*y (vector ysc)
       for (i = 0; i < size; i++) {
      lda = 0;
      j = lh_gold - 1;
      jmin = -j;
      jmax = j;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j - jmin + size];
         ldc = working_space[2 * size + i + j - jmin];
         lda = lda + ldb * ldc;
      }
      working_space[4 * size + i] = lda;
   }
   
//move ysc
       for (i = 0; i < size; i++)
      working_space[2 * size + i] = working_space[4 * size + i];
   
//create vector c//
       i = 2 * lh_gold - 2;
   if (i > size)
      i = size;
   imin = -i;
   imax = i;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      jmin = -lh_gold + 1 + i;
      if (jmin < (-lh_gold + 1))
         jmin = -lh_gold + 1;
      jmax = lh_gold - 1 + i;
      if (jmax > (lh_gold - 1))
         jmax = lh_gold - 1;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j + lh_gold - 1 + size];
         ldc = working_space[i - j + lh_gold - 1 + size];
         lda = lda + ldb * ldc;
      }
      working_space[i - imin] = lda;
   }
   
//move vector c
       for (i = 0; i < size; i++)
      working_space[i + size] = working_space[i];
   
//initialization of resulting vector
       for (i = 0; i < size; i++)
      working_space[i] = 1;
   
       //**START OF ITERATIONS**
       for (lindex = 0; lindex < number_of_iterations; lindex++) {
      for (i = 0; i < size; i++) {
         if (working_space[2 * size + i] > 0.000001
              && working_space[i] > 0.000001) {
            lda = 0;
            jmin = 2 * lh_gold - 2;
            if (jmin > i)
               jmin = i;
            jmin = -jmin;
            jmax = 2 * lh_gold - 2;
            if (jmax > (size - 1 - i))
               jmax = size - 1 - i;
            for (j = jmin; j <= jmax; j++) {
               ldb = working_space[j + 2 * lh_gold - 2 + size];
               ldc = working_space[i + j];
               lda = lda + ldb * ldc;
            }
            ldb = working_space[2 * size + i];
            if (lda != 0)
               lda = ldb / lda;
            
            else
               lda = 0;
            ldb = working_space[i];
            lda = lda * ldb;
            working_space[3 * size + i] = lda;
         }
      }
      for (i = 0; i < size; i++)
         working_space[i] = working_space[3 * size + i];
   }
   
//shift resulting spectrum
       for (i = 0; i < size; i++) {
      lda = working_space[i];
      j = i + posit;
      j = j % size;
      working_space[size + j] = lda;
   }
   
//write back resulting spectrum
       for (i = 0; i < size; i++)
      source[i] = area * working_space[size + i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
double TSpectrum::Lls(double a) 
{
   
/////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                     //
//                                                                         //
//   LLS operator. It calculates log(log(sqrt(a+1))) value of a.           //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       if (a < 0)
      a = 0;
   a = TMath::Sqrt(a + 1.0);
   a = TMath::Log(a + 1.0);
   a = TMath::Log(a + 1.0);
   return (a);
}
const char *TSpectrum::Deconvolution1HighResolution(float *source,
                                                     const float *resp,
                                                     int size,
                                                     int
                                                     number_of_iterations,
                                                     int
                                                     number_of_repetitions,
                                                     double boost) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL HIGH RESOLUTION DECONVOLUTION FUNCTION		   */ 
/*	This function calculates deconvolution from source spectrum	   */ 
/*	according to response spectrum					   */ 
/*	The result is placed in the vector pointed by source pointer.	   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum			   */ 
/*	resp-pointer to the vector of response spectrum			   */ 
/*	size-length of source and response spectra			   */ 
/*	number_of_iterations, for details we refer to manual		   */ 
/*	number_of_repetitions, for details we refer to manual		   */ 
/*	boost, boosting factor, for details we refer to manual		   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, m, lindex, posit, imin, imax, jmin, jmax, lh_gold, iter,
       repet;
   double lda, ldb, ldc, lday, ldby, area, maximum;
   double a;
   if (size <= 0)
      return "Wrong Parameters";
   if (number_of_iterations <= 0)
      return "Number of iterations must be positive";
   if (number_of_repetitions <= 0)
      return "Number of repetitions must be positive";
   if (boost <= 0)
      return ("Boosting Factor Must be Positive Number");
   
       //   working_space-pointer to the working vector
       //   (its size must be 7*size of source spectrum)
   double *working_space = new double[7 * size];
   for (i = size, iter = 0, j = 1; i > 1;) {
      iter += 1;
      i = i / 2;
      j = j * 2;
   }
   if (j != size)
      return ("SIZE MUST BE POWER OF 2");
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;
   
//read response vector
       for (i = 0; i < size; i++) {
      lda = resp[i];
      if (lda != 0)
         lh_gold = i + 1;
      working_space[i] = lda;
      area = area + lda;
      if (lda > maximum) {
         maximum = lda;
         posit = i;
      }
   }
   if (lh_gold == -1)
      return ("ZERO RESPONSE VECTOR");
   
/////////TOEPLITZ MATRIX INVERSION//////////////////////////
//read source vector
       for (i = 0; i < size; i++) {
      working_space[size + i] = source[i];
   }
   for (i = 0; i < size; i++) {
      lda = 0, lday = 0;
      for (j = i; j < size; j++) {
         ldb = working_space[j];
         ldc = working_space[j - i];
         lda += ldb * ldc;
         ldby = working_space[j + size];
         lday += ldby * ldc;
      }
      working_space[i + 3 * size] = lda;
      working_space[i + 4 * size] = lday;
   }
   for (i = 0; i < 2 * size; i++)
      working_space[i] = working_space[i + 3 * size];
   a = working_space[0];
   if (a == 0)
      return ("SINGULAR MATRIX");
   working_space[3 * size + 2 * size] = working_space[size] / a;
   working_space[3 * size] = working_space[1] / a;
   for (m = 1; m < size; m++) {
      a = working_space[0];
      working_space[3 * size + m + size] = working_space[m + 1];
      working_space[3 * size + m + 3 * size] = working_space[m + size];
      lda = 0, ldb = 0, ldc = 0;
      for (j = 1; j <= m; j++) {
         lda += working_space[j] * working_space[3 * size + j - 1];
         ldb +=
             working_space[m + 1 - j] * working_space[3 * size + j - 1];
         ldc +=
             working_space[m + 1 - j] * working_space[3 * size + j - 1 +
                                                      2 * size];
      }
      a -= lda;
      working_space[3 * size + m + size] -= ldb;
      working_space[3 * size + m + 3 * size] -= ldc;
      if (a == 0)
         return ("SINGULAR MATRIX");
      working_space[3 * size + m + size] /= a;
      working_space[3 * size + m + 3 * size] /= a;
      for (j = 1; j <= m; j++) {
         working_space[3 * size + j - 1 + size] =
             working_space[3 * size + j - 1] - working_space[3 * size + m +
                                                             size] *
             working_space[3 * size + m - j];
         working_space[3 * size + j - 1 + 3 * size] =
             working_space[3 * size + j - 1 + 2 * size] -
             working_space[3 * size + m +
                           3 * size] * working_space[3 * size + m - j];
      }
      for (i = 0; i <= m; i++) {
         working_space[3 * size + i] = working_space[3 * size + i + size];
         working_space[3 * size + i + 2 * size] =
             working_space[3 * size + i + 3 * size];
      }
   }
   for (i = 0; i < size; i++)
      working_space[i] = working_space[3 * size + i + 2 * size];
   
//////////////////////*Fourier deconvolution*///////////////////////////
       for (i = 0; i < size; i++) {
      working_space[6 * size + i] = Lls(working_space[i]);
   }
   
////////////////////End of Fourier deconvolution///////////////////////
//read response vector
       for (i = 0; i < size; i++)
      working_space[i] = resp[i];
   
//read source vector
       for (i = 0; i < size; i++)
      working_space[2 * size + i] = source[i];
   
//create matrix at*a(vector b)
       i = lh_gold - 1;
   if (i > size)
      i = size;
   imin = -i, imax = i;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      jmin = 0;
      if (i < 0)
         jmin = -i;
      jmax = lh_gold - 1 - i;
      if (jmax > (lh_gold - 1))
         jmax = lh_gold - 1;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j];
         ldc = working_space[i + j];
         lda = lda + ldb * ldc;
      }
      working_space[size + i - imin] = lda;
   }
   
//create vector p
       i = lh_gold - 1;
   imin = -i, imax = size + i - 1;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      for (j = 0; j <= (lh_gold - 1); j++) {
         ldb = working_space[j];
         k = i + j;
         if (k >= 0 && k < size) {
            ldc = working_space[2 * size + k];
            lda = lda + ldb * ldc;
         }
      }
      working_space[4 * size + i - imin] = lda;
   }
   
//move vector p
       for (i = imin; i <= imax; i++)
      working_space[2 * size + i - imin] =
          working_space[4 * size + i - imin];
   
//create at*a*at*y (vector ysc)
       for (i = 0; i < size; i++) {
      lda = 0;
      j = lh_gold - 1;
      jmin = -j, jmax = j;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j - jmin + size];
         ldc = working_space[2 * size + i + j - jmin];
         lda = lda + ldb * ldc;
      }
      working_space[4 * size + i] = lda;
   }
   
//move ysc
       for (i = 0; i < size; i++)
      working_space[2 * size + i] = working_space[4 * size + i];
   
//create vector c
       i = 2 * lh_gold - 2;
   if (i > size)
      i = size;
   imin = -i, imax = i;
   for (i = imin; i <= imax; i++) {
      lda = 0;
      jmin = -lh_gold + 1 + i;
      if (jmin < (-lh_gold + 1))
         jmin = -lh_gold + 1;
      jmax = lh_gold - 1 + i;
      if (jmax > (lh_gold - 1))
         jmax = lh_gold - 1;
      for (j = jmin; j <= jmax; j++) {
         ldb = working_space[j + lh_gold - 1 + size];
         ldc = working_space[i - j + lh_gold - 1 + size];
         lda = lda + ldb * ldc;
      }
      working_space[i - imin] = lda;
   }
   
//move vector c
       for (i = 0; i < size; i++)
      working_space[i + size] = working_space[i];
   
//initialization of resulting vector
       for (i = 0, a = 0; i < size; i++) {
      working_space[i] = working_space[6 * size + i];
      a += working_space[6 * size + i];
   }
   for (i = 0; i < size; i++) {
      working_space[i] = working_space[i] / a;
   }
   
       //////START OF ITERATIONS////
       for (repet = 0; repet < number_of_repetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < size; i++)
            working_space[i] = TMath::Power(working_space[i], boost);
      }
      for (lindex = 0; lindex < number_of_iterations; lindex++) {
         for (i = 0; i < size; i++) {
            lda = 0;
            jmin = 2 * lh_gold - 2;
            if (jmin > i)
               jmin = i;
            jmin = -jmin;
            jmax = 2 * lh_gold - 2;
            if (jmax > (size - 1 - i))
               jmax = size - 1 - i;
            for (j = jmin; j <= jmax; j++) {
               ldb = working_space[j + 2 * lh_gold - 2 + size];
               ldc = working_space[i + j];
               lda = lda + ldb * ldc;
            }
            ldb = working_space[2 * size + i];
            if (lda != 0)
               lda = ldb / lda;
            
            else
               lda = 0;
            ldb = working_space[i];
            lda = lda * ldb;
            working_space[3 * size + i] = lda;
         }
         for (i = 0; i < size; i++)
            working_space[i] = working_space[3 * size + i];
      }
   }
   
//shift resulting spectrum
       for (i = 0; i < size; i++) {
      lda = working_space[i];
      j = i + posit;
      j = j % size;
      working_space[size + j] = lda;
   }
   
//write back resulting spectrum
       for (i = 0; i < size; i++)
      source[i] = area * working_space[size + i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum::Deconvolution1Unfolding(float *source,
                                               const float **resp,
                                               int sizex, int sizey,
                                               int number_of_iterations) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL UNFOLDING FUNCTION				   */ 
/*	This function unfolds source spectrum				   */ 
/*	according to response matrix columns.				   */ 
/*	The result is placed in the vector pointed by source pointer.	   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum			   */ 
/*	resp-pointer to the matrix of response spectra			   */ 
/*	sizex-length of source spectrum and # of columns of response matrix*/ 
/*	sizey-length of destination spectrum and # of rows of		   */ 
/*	      response matrix						   */ 
/*	number_of_iterations, for details we refer to manual		   */ 
/*	Note!!! sizex must be >= sizey					   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, lindex, lhx = 0;
   double lda, ldb, ldc, area;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong Parameters";
   if (sizex < sizey)
      return "Sizex must be greater than sizey)";
   if (number_of_iterations <= 0)
      return "Number of iterations must be positive";
   double *working_space =
       new double[sizex * sizey + 2 * sizey * sizey + 4 * sizex];
   
/*read response matrix*/ 
       for (j = 0; j < sizey && lhx != -1; j++) {
      area = 0;
      lhx = -1;
      for (i = 0; i < sizex; i++) {
         lda = resp[j][i];
         if (lda != 0) {
            lhx = i + 1;
         }
         working_space[j * sizex + i] = lda;
         area = area + lda;
      }
      if (lhx != -1) {
         for (i = 0; i < sizex; i++)
            working_space[j * sizex + i] /= area;
      }
   }
   if (lhx == -1)
      return ("ZERO COLUMN IN RESPONSE MATRIX");
   
/*read source vector*/ 
       for (i = 0; i < sizex; i++)
      working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex + i] =
          source[i];
   
/*create matrix at*a + at*y */ 
       for (i = 0; i < sizey; i++) {
      for (j = 0; j < sizey; j++) {
         lda = 0;
         for (k = 0; k < sizex; k++) {
            ldb = working_space[sizex * i + k];
            ldc = working_space[sizex * j + k];
            lda = lda + ldb * ldc;
         }
         working_space[sizex * sizey + sizey * i + j] = lda;
      }
      lda = 0;
      for (k = 0; k < sizex; k++) {
         ldb = working_space[sizex * i + k];
         ldc =
             working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex +
                           k];
         lda = lda + ldb * ldc;
      }
      working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex + i] =
          lda;
   }
   
/*move vector at*y*/ 
       for (i = 0; i < sizey; i++)
      working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex + i] =
          working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex + i];
   
/*create matrix at*a*at*a + vector at*a*at*y */ 
       for (i = 0; i < sizey; i++) {
      for (j = 0; j < sizey; j++) {
         lda = 0;
         for (k = 0; k < sizey; k++) {
            ldb = working_space[sizex * sizey + sizey * i + k];
            ldc = working_space[sizex * sizey + sizey * j + k];
            lda = lda + ldb * ldc;
         }
         working_space[sizex * sizey + sizey * sizey + sizey * i + j] =
             lda;
      }
      lda = 0;
      for (k = 0; k < sizey; k++) {
         ldb = working_space[sizex * sizey + sizey * i + k];
         ldc =
             working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex +
                           k];
         lda = lda + ldb * ldc;
      }
      working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex + i] =
          lda;
   }
   
/*move at*a*at*y*/ 
       for (i = 0; i < sizey; i++)
      working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex + i] =
          working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex + i];
   
/*initialization in resulting vectore */ 
       for (i = 0; i < sizey; i++)
      working_space[sizex * sizey + 2 * sizey * sizey + i] = 1;
   
	/***START OF ITERATIONS***/ 
       for (lindex = 0; lindex < number_of_iterations; lindex++) {
      for (i = 0; i < sizey; i++) {
         lda = 0;
         for (j = 0; j < sizey; j++) {
            ldb =
                working_space[sizex * sizey + sizey * sizey + sizey * i +
                              j];
            ldc = working_space[sizex * sizey + 2 * sizey * sizey + j];
            lda = lda + ldb * ldc;
         }
         ldb =
             working_space[sizex * sizey + 2 * sizey * sizey + 2 * sizex +
                           i];
         if (lda != 0) {
            lda = ldb / lda;
         }
         
         else
            lda = 0;
         ldb = working_space[sizex * sizey + 2 * sizey * sizey + i];
         lda = lda * ldb;
         working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex +
                        i] = lda;
      }
      for (i = 0; i < sizey; i++)
         working_space[sizex * sizey + 2 * sizey * sizey + i] =
             working_space[sizex * sizey + 2 * sizey * sizey + 3 * sizex +
                           i];
   }
   
/*write back resulting spectrum*/ 
       for (i = 0; i < sizex; i++) {
      if (i < sizey)
         source[i] = working_space[sizex * sizey + 2 * sizey * sizey + i];
      
      else
         source[i] = 0;
   }
   delete[]working_space;
   return 0;
}


//_____________________________________________________________________________
    Int_t TSpectrum::Search1HighRes(float *source,float *dest, int size,
                                     float sigma, double threshold,
                                     bool background_remove,int decon_iterations,
                                     bool markov, int aver_window)
{
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL HIGH-RESOLUTION PEAK SEARCH FUNCTION		   */
/*	This function searches for peaks in source spectrum		   */
/*      It is based on deconvolution method. First the background is       */
/*      removed (if desired), then Markov spectrum is calculated           */
/*      (if desired), then the response function is generated              */
/*      according to given sigma and deconvolution is carried out.         */
/*									   */
/*	Function parameters:						   */
/*	source-pointer to the vector of source spectrum			   */
/*	dest-pointer to the vector of resulting deconvolved spectrum	   */
/*	size-length of source spectrum			                   */
/*	sigma-sigma of searched peaks, for details we refer to manual	   */
/*	threshold-threshold value in % for selected peaks, peaks with      */
/*                amplitude less than threshold*highest_peak/100           */
/*                are ignored, see manual                                  */
/*      background_remove-logical variable, set if the removal of          */
/*                background before deconvolution is desired               */
/*      decon_iterations-number of iterations in deconvolution operation   */
/*      markov-logical variable, if it is true, first the source spectrum  */
/*             is replaced by new spectrum calculated using Markov         */
/*             chains method.                                              */
/*	aver_window-averanging window of searched peaks, for details       */
/*                  we refer to manual (applies only for Markov method)    */
/*									   */
/////////////////////////////////////////////////////////////////////////////
   int i, j, number_of_iterations = (int)(7 * sigma + 0.5);
   float a, b;
   int k, lindex, posit, imin, imax, jmin, jmax, lh_gold;
   double lda, ldb, ldc, area,maximum;
   int xmin, xmax, l, peak_index = 0, size_ext = size + 2 * number_of_iterations, shift = number_of_iterations;
   float maxch;
   float nom, nip, nim, sp, sm, plocha = 0;
   if (sigma < 1) {
      Error("Search1HighRes", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }
   	
   if(threshold<=0||threshold>=100){
      Error("Search1HighRes", "Invalid threshold, must be positive and less than 100");
      return 0;
   }
   
   j = (int) (5.0 * sigma + 0.5);
   if (j >= PEAK_WINDOW / 2) {
      Error("Search1HighRes", "Too large sigma");
      return 0;
   }
   
   if (markov == true) {
      if (aver_window <= 0) {
         Error("Search1HighRes", "Averanging window must be positive");
         return 0;
      }
   }
         
   if(background_remove == true){
      if(size < 2 * number_of_iterations + 1){   
         Error("Search1HighRes", "Too large clipping window");
         return 0;
      }
   }
   
   i = (int)(7 * sigma + 0.5);
   i = 2 * i;
   double *working_space = new double [6 * (size + i)];    
   for(i = 0; i < size_ext; i++){
      if(i < shift)
         working_space[i + size_ext] = source[0];
         
      else if(i >= size + shift)
         working_space[i + size_ext] = source[size - 1];
         
      else
         working_space[i + size_ext] = source[i - shift];
   }
   if(background_remove == true){
      for(i = 1; i <= number_of_iterations; i++){
       	 for(j = i; j < size_ext - i; j++){
            a = working_space[size_ext + j];
       	    b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
            if(b < a)
	        a = b;
	           
       	    working_space[j]=a;
         }
       	 for(j = i; j < size_ext - i; j++)
            working_space[size_ext + j] = working_space[j];
      }
      for(j = 0;j < size_ext; j++){
         if(j < shift)
            working_space[size_ext + j] = source[0] - working_space[size_ext + j];
            
         else if(j >= size + shift)
            working_space[size_ext + j] = source[size - 1] - working_space[size_ext + j];
            
         else{
            working_space[size_ext + j] = source[j - shift] - working_space[size_ext + j];
         }
      }
   }
   if(markov == true){
      for(j = 0; j < size_ext; j++)
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      xmin = 0,xmax = size_ext - 1;
      for(i = 0, maxch = 0; i < size_ext; i++){
	 working_space[i] = 0;
         if(maxch < working_space[2 * size_ext + i])
            maxch = working_space[2 * size_ext + i];
         plocha += working_space[2 * size_ext + i];
      }
      if(maxch == 0)
         return 0;
         
      nom = 1;
      working_space[xmin] = 1;
      for(i = xmin; i < xmax; i++){
	 nip = working_space[2 * size_ext + i] / maxch;
         nim = working_space[2 * size_ext + i + 1] / maxch;
         sp = 0,sm = 0;
	 for(l = 1; l <= aver_window; l++){
            if((i + l) > xmax)
               a = working_space[2 * size_ext + xmax] / maxch;
               
            else
               a = working_space[2 * size_ext + i + l] / maxch;
               
	    b = a - nip;
            if(a + nip <= 0)
               a=1;
               
            else
               a = TMath::Sqrt(a + nip);            
               
            b = b / a;
            b = TMath::Exp(b);            
            sp = sp + b;
            if((i - l + 1) < xmin)
               a = working_space[2 * size_ext + xmin] / maxch;
               
            else
               a = working_space[2 * size_ext + i - l + 1] / maxch;
               
            b = a - nim;
            if(a + nim <= 0)
               a = 1;
               
            else
               a = TMath::Sqrt(a + nim);
               
	    b = b / a;
            b = TMath::Exp(b);            	    
       	    sm = sm + b;
       	 }
         a = sp / sm;
         a = working_space[i + 1] = working_space[i] * a;
         nom = nom + a;
      }
      for(i = xmin; i <= xmax; i++){
         working_space[i] = working_space[i] / nom;
      }
      for(j = 0; j < size_ext; j++)
         working_space[size_ext + j] = working_space[j] * plocha;
      for(j = 0; j < size_ext; j++){
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      }
      if(background_remove == true){
         for(i = 1; i <= number_of_iterations; i++){
       	    for(j = i; j < size_ext - i; j++){
               a = working_space[size_ext + j];
               b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
               if(b < a)
	          a = b;
               working_space[j] = a;
            }
       	    for(j = i; j < size_ext - i; j++)
               working_space[size_ext + j] = working_space[j];
         }
	 for(j = 0; j < size_ext; j++){
            working_space[size_ext + j] = working_space[2 * size_ext + j] - working_space[size_ext + j];
         }
      }
   }
//deconvolution starts
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;
//generate response vector
   for(i = 0; i < size_ext; i++){
      lda = (double)i - 3 * sigma;
      lda = lda * lda / (2 * sigma * sigma);    
      j = (int)(1000 * TMath::Exp(-lda));
      lda = j;
      if(lda != 0)
	 lh_gold = i + 1;
	 
      working_space[i] = lda;
      area = area + lda;
      if(lda > maximum){
	 maximum = lda;
	 posit = i;
      }
   }
//read source vector
   for(i = 0; i < size_ext; i++)
      working_space[2 * size_ext + i] = TMath::Abs(working_space[size_ext + i]);
//create matrix at*a(vector b)
   i = lh_gold - 1;
   if(i > size_ext)
     i = size_ext;
     
   imin = -i,imax = i;
   for(i = imin; i <= imax; i++){
      lda = 0;
      jmin = 0;
      if(i < 0)
	 jmin = -i;
      jmax = lh_gold - 1 - i;
      if(jmax > (lh_gold - 1))
	 jmax = lh_gold - 1;
	 
      for(j = jmin;j <= jmax; j++){
	 ldb = working_space[j];
	 ldc = working_space[i + j];
	 lda = lda + ldb * ldc;
      }
      working_space[size_ext + i - imin] = lda;
   }
//create vector p
   i = lh_gold - 1;
   imin = -i,imax = size_ext + i - 1;
   for(i = imin; i <= imax; i++){
      lda = 0;
      for(j = 0; j <= (lh_gold - 1); j++){
	 ldb = working_space[j];
	 k = i + j;
	 if(k >= 0 && k < size_ext){
	    ldc = working_space[2 * size_ext + k];
	    lda = lda + ldb * ldc;
	 }
	 
      }
      working_space[4 * size_ext + i - imin] = lda;
   }
//move vector p
   for(i = imin; i <= imax; i++)
      working_space[2 * size_ext + i - imin] = working_space[4 * size_ext + i - imin];
//initialization of resulting vector
   for(i = 0; i < size_ext; i++)
      working_space[i] = 1;
//START OF ITERATIONS
   for(lindex = 0; lindex < decon_iterations; lindex++){
      for(i = 0; i < size_ext; i++){
         if(TMath::Abs(working_space[2 * size_ext + i]) > 0.00001 && TMath::Abs(working_space[i]) > 0.00001){
	    lda=0;
	    jmin = lh_gold - 1;
       	    if(jmin > i)
               jmin = i;
               
	    jmin = -jmin;
	    jmax = lh_gold - 1;
            if(jmax > (size_ext - 1 - i))
	       jmax=size_ext-1-i;
	       
       	    for(j = jmin; j <= jmax; j++){
               ldb = working_space[j + lh_gold - 1 + size_ext];
	       ldc = working_space[i + j];
	       lda = lda + ldb * ldc;
	    }
       	    ldb = working_space[2 * size_ext + i];
            if(lda != 0)
	       lda = ldb / lda;
	          
	    else
	       lda = 0;
		  
       	    ldb = working_space[i];
            lda = lda * ldb;
	    working_space[3 * size_ext + i] = lda;
	 }
      }
      for(i = 0; i < size_ext; i++){
         working_space[i] = working_space[3 * size_ext + i];
      }
   }
//shift resulting spectrum
   for(i=0;i<size_ext;i++){
      lda = working_space[i];
      j = i + posit;
      j = j % size_ext;
      working_space[size_ext + j] = lda;
   }
//write back resulting spectrum
   maximum = 0;
   j = lh_gold - 1;
   for(i = 0; i < size_ext - j; i++){
      working_space[i] = area * working_space[size_ext + i + j];
      if(maximum < working_space[i])
         maximum = working_space[i];
   }
//searching for peaks in deconvolved spectrum
   for(i = 1; i < size_ext - 1; i++){
      if(working_space[i] > working_space[i - 1] && working_space[i] > working_space[i + 1]){
         if(i >= shift && i < size + shift){
            if(working_space[i] > threshold * maximum / 100.0){
               if(peak_index < fMaxPeaks){
                  for(j = i - 1, a = 0, b = 0; j <= i + 1; j++){
                     a += (double)(j - shift) * working_space[j];
                     b += working_space[j];
                  }
                  a = a / b;
                  if(a < 0)
                     a = 0;
                     
                  if(a >= size)
                     a = size - 1;
                     
                  fPositionX[peak_index] = a;                        
        	  peak_index += 1;
	       }
	       else{
                  Warning("Search1HighRes", "Peak buffer full");
                  return 0;
               }	          
            }
         }
      }
   }
   for(i = 0; i < size; i++)
      dest[i] = working_space[i + shift];      
   delete[]working_space;
   fNPeaks = peak_index;
   return fNPeaks;
}


//_____________________________________________________________________________
/////////////////BEGINNING OF AUXILIARY FUNCTIONS USED BY FITTING FUNCION Fit1//////////////////////////
double TSpectrum::Erfc(double x) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates error function of x.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   double da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap =
       0.47047;
   double a, t, c, w;
   a = TMath::Abs(x);
   w = 1. + dap * a;
   t = 1. / w;
   w = a * a;
   if (w < 700)
      c = exp(-w);
   
   else {
      c = 0;
   }
   c = c * t * (da1 + t * (da2 + t * da3));
   if (x < 0)
      c = 1. - c;
   return (c);
}
double TSpectrum::Derfc(double x) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates derivative of error function of x.             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   double a, t, c, w;
   double da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap =
       0.47047;
   a = TMath::Abs(x);
   w = 1. + dap * a;
   t = 1. / w;
   w = a * a;
   if (w < 700)
      c = exp(-w);
   
   else {
      c = 0;
   }
   c = (-1.) * dap * c * t * t * (da1 + t * (2. * da2 + t * 3. * da3)) -
       2. * a * Erfc(a);
   return (c);
}
double TSpectrum::Deramp(double i, double i0, double sigma, double t,
                           double s, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates derivative of peak shape function (see manual) //
//   according to amplitude of peak.                                        //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -i0-position of peak                                        //
//              -sigma-sigma of peak                                        //
//              -t, s-relative amplitudes                                   //
//              -b-slope                                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   double p, q, r, a;
   p = (i - i0) / sigma;
   if ((p * p) < 700)
      q = exp(-p * p);
   
   else {
      q = 0;
   }
   r = 0;
   if (t != 0) {
      a = p / b;
      if (a > 700)
         a = 700;
      r = t * exp(a) / 2.;
   }
   if (r != 0)
      r = r * Erfc(p + 1. / (2. * b));
   q = q + r;
   if (s != 0)
      q = q + s * Erfc(p) / 2.;
   return (q);
}
double TSpectrum::Deri0(double i, double amp, double i0, double sigma,
                          double t, double s, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates derivative of peak shape function (see manual) //
//   according to peak position.                                            //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -amp-amplitude of peak                                      //
//              -i0-position of peak                                        //
//              -sigma-sigma of peak                                        //
//              -t, s-relative amplitudes                                   //
//              -b-slope                                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   double p, r1, r2, r3, r4, c, d, e;
   p = (i - i0) / sigma;
   d = 2. * sigma;
   if ((p * p) < 700)
      r1 = 2. * p * exp(-p * p) / sigma;
   
   else {
      r1 = 0;
   }
   r2 = 0, r3 = 0;
   if (t != 0) {
      c = p + 1. / (2. * b);
      e = p / b;
      if (e > 700)
         e = 700;
      r2 = -t * exp(e) * Erfc(c) / (d * b);
      r3 = -t * exp(e) * Derfc(c) / d;
   }
   r4 = 0;
   if (s != 0)
      r4 = -s * Derfc(p) / d;
   r1 = amp * (r1 + r2 + r3 + r4);
   return (r1);
}
double TSpectrum::Derderi0(double i, double amp, double i0,
                             double sigma) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates second derivative of peak shape function       //
//   (see manual) according to peak position.                               //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -amp-amplitude of peak                                      //
//              -i0-position of peak                                        //
//              -sigma-width of peak                                        //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   double p, r1, r2, r3, r4;
   p = (i - i0) / sigma;
   if ((p * p) < 700)
      r1 = exp(-p * p);
   
   else {
      r1 = 0;
   }
   r1 = r1 * (4 * p * p - 2) / (sigma * sigma);
   r2 = 0, r3 = 0, r4 = 0;
   r1 = amp * (r1 + r2 + r3 + r4);
   return (r1);
}
double TSpectrum::Dersigma(int num_of_fitted_peaks, double i,
                             const double *parameter, double sigma,
                             double t, double s, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t, s-relative amplitudes                                       //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, r2, r3, r4, c, d, e;
   r = 0;
   d = 2. * sigma;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = 2. * p * p * exp(-p * p) / sigma;
         
         else {
            r1 = 0;
         }
      }
      r2 = 0, r3 = 0;
      if (t != 0) {
         c = p + 1. / (2. * b);
         e = p / b;
         if (e > 700)
            e = 700;
         r2 = -t * p * exp(e) * Erfc(c) / (d * b);
         r3 = -t * p * exp(e) * Derfc(c) / d;
      }
      r4 = 0;
      if (s != 0)
         r4 = -s * p * Derfc(p) / d;
      r = r + parameter[2 * j] * (r1 + r2 + r3 + r4);
   }
   return (r);
}
double TSpectrum::Derdersigma(int num_of_fitted_peaks, double i,
                               const double *parameter, double sigma) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of peaks shape function          //
//   (see manual) according to sigma of peaks.                                  //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, r2, r3, r4;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = exp(-p * p) * p * p * (4. * p * p - 6) / (sigma * sigma);
         
         else {
            r1 = 0;
         }
      }
      r2 = 0, r3 = 0, r4 = 0;
      r = r + parameter[2 * j] * (r1 + r2 + r3 + r4);
   }
   return (r);
}
double TSpectrum::Dert(int num_of_fitted_peaks, double i,
                        const double *parameter, double sigma, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude t.                                         //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      c = p + 1. / (2. * b);
      e = p / b;
      if (e > 700)
         e = 700;
      r1 = exp(e) * Erfc(c);
      r = r + parameter[2 * j] * r1;
   }
   r = r / 2.;
   return (r);
}
double TSpectrum::Ders(int num_of_fitted_peaks, double i,
                        const double *parameter, double sigma) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude s.                                               //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = Erfc(p);
      r = r + parameter[2 * j] * r1;
   }
   r = r / 2.;
   return (r);
}
double TSpectrum::Derb(int num_of_fitted_peaks, double i,
                        const double *parameter, double sigma, double t,
                        double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to slope b.                                                      //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitude                                           //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks && t != 0; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      c = p + 1. / (2. * b);
      e = p / b;
      r1 = p * Erfc(c);
      r1 = r1 + Derfc(c) / 2.;
      if (e > 700)
         e = 700;
      if (e < -700)
         r1 = 0;
      
      else
         r1 = r1 * exp(e);
      r = r + parameter[2 * j] * r1;
   }
   r = -r * t / (2. * b * b);
   return (r);
}
double TSpectrum::Dera1(double i)	//derivative of backgroud according to a1
{
   return (i);
}
double TSpectrum::Dera2(double i)	//derivative of backgroud according to a2
{
   return (i * i);
}
double TSpectrum::Shape(int num_of_fitted_peaks, double i,
                         const double *parameter, double sigma, double t,
                         double s, double b, double a0, double a1,
                         double a2) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates peaks shape function (see manual)                  //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t, s-relative amplitudes                                       //
//              -b-slope                                                        //
//              -a0, a1, a2- background coefficients                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, r2, r3, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      if (sigma > 0.0001)
         p = (i - parameter[2 * j + 1]) / sigma;
      
      else {
         if (i == parameter[2 * j + 1])
            p = 0;
         
         else
            p = 10;
      }
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = exp(-p * p);
         
         else {
            r1 = 0;
         }
      }
      r2 = 0;
      if (t != 0) {
         c = p + 1. / (2. * b);
         e = p / b;
         if (e > 700)
            e = 700;
         r2 = t * exp(e) * Erfc(c) / 2.;
      }
      r3 = 0;
      if (s != 0)
         r3 = s * Erfc(p) / 2.;
      r = r + parameter[2 * j] * (r1 + r2 + r3);
   }
   r = r + a0 + a1 * i + a2 * i * i;
   return (r);
}
double TSpectrum::Area(double a, double sigma, double t, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates area of a peak                                     //
//      Function parameters:                                                    //
//              -a-amplitude of the peak                                        //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitude                                           //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double odm_pi = 1.7724538, r = 0;
   if (b != 0)
      r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * sigma * (odm_pi + t * b * exp(r));
   
   else {
      r = a * sigma * odm_pi;
   }
   return (r);
}
double TSpectrum::Derpa(double sigma, double t, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the area of peak                     //
//   according to its amplitude.                                                //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double odm_pi = 1.7724538, r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = sigma * (odm_pi + t * b * exp(r));
   
   else {
      r = sigma * odm_pi;
   }
   return (r);
}
double TSpectrum::Derpsigma(double a, double t, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the area of peak                     //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double odm_pi = 1.7724538, r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * (odm_pi + t * b * exp(r));
   
   else {
      r = a * odm_pi;
   }
   return (r);
}
double TSpectrum::Derpt(double a, double sigma, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the area of peak                     //
//   according to t parameter.                                                  //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * sigma * b * exp(r);
   
   else {
      r = 0;
   }
   return (r);
}
double TSpectrum::Derpb(double a, double sigma, double t, double b) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the area of peak                     //
//   according to b parameter.                                                  //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double r;
   r = (-1) * 0.25 / (b * b);
   if (TMath::Abs(r) < 700)
      r = a * sigma * t * exp(r) * (1 - 2 * r);
   
   else {
      r = 0;
   }
   return (r);
}
double TSpectrum::Ourpowl(double a, int pw)
{                               //power function
   double c;
   c = 1;
   if (pw > 0)
      c = c * a * a;
   
   else if (pw > 2)
      c = c * a * a;
   
   else if (pw > 4)
      c = c * a * a;
   
   else if (pw > 6)
      c = c * a * a;
   
   else if (pw > 8)
      c = c * a * a;
   
   else if (pw > 10)
      c = c * a * a;
   
   else if (pw > 12)
      c = c * a * a;
   return (c);
}


/////////////////END OF AUXILIARY FUNCTIONS USED BY FITTING FUNCION fit1//////////////////////////
    
/////////////////FITTING FUNCTION WITHOUT MATRIX INVERSION///////////////////////////////////////
const char *TSpectrum::Fit1Awmi(float *source, TSpectrumOneDimFit * p,
                                 int size) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL FIT FUNCTION				           */ 
/*      ALGORITHM WITHOUT MATRIX INVERSION                                 */ 
/*	This function fits the source spectrum. The calling program should */ 
/*      fill in input parameters of the TSpectrumOneDimFit class 	   */ 
/*	The fitted parameters are written into class pointed by 	   */ 
/*	TSpectrumOneDimFit class pointer and fitted data are written into  */ 
/*      source spectrum.                                                   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum			   */ 
/*	p-pointer to the TSpectrumOneDimFit class, see manual              */ 
/*	size-length of source spectrum                                     */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, shift =
       2 * p->number_of_peaks + 7, peak_vel, rozmer, iter, pw, regul_cycle,
       flag;
   double a, b, c, d = 0, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   if (size <= 0)
      return "Wrong Parameters";
   if (p->number_of_peaks <= 0)
      return ("INVALID NUMBER OF PEAKS, MUST BE POSITIVE");
   if (p->number_of_iterations <= 0)
      return ("INVALID NUMBER OF ITERATIONS, MUST BE POSITIVE");
   if (p->alpha <= 0 || p->alpha > 1)
      return ("INVALID COEFFICIENT ALPHA, MUST BE > THAN 0 AND <=1");
   if (p->statistic_type != FIT1_OPTIM_CHI_COUNTS
        && p->statistic_type != FIT1_OPTIM_CHI_FUNC_VALUES
        && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD)
      return ("WRONG TYPE OF STATISTIC");
   if (p->alpha_optim != FIT1_ALPHA_HALVING
        && p->alpha_optim != FIT1_ALPHA_OPTIMAL)
      return ("WRONG OPTIMIZATION ALGORITHM");
   if (p->power != FIT1_FIT_POWER2 && p->power != FIT1_FIT_POWER4
        && p->power != FIT1_FIT_POWER6 && p->power != FIT1_FIT_POWER8
        && p->power != FIT1_FIT_POWER10 && p->power != FIT1_FIT_POWER12)
      return ("WRONG POWER");
   if (p->fit_taylor != FIT1_TAYLOR_ORDER_FIRST
        && p->fit_taylor != FIT1_TAYLOR_ORDER_SECOND)
      return ("WRONG ORDER OF TAYLOR DEVELOPMENT");
   if (p->xmin < 0 || p->xmin > p->xmax)
      return ("INVALID LOW LIMIT OF FITTING REGION");
   if (p->xmax >= size || p->xmax < p->xmin)
      return ("INVALID HIGH LIMIT OF FITTING REGION");
   double *working_space = new double[5 * (2 * p->number_of_peaks + 7)];
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      if (p->amp_init[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[2 * i] = p->amp_init[i];	//vector parameter
      if (p->fix_amp[i] == false) {
         working_space[shift + j] = p->amp_init[i];	//vector xk
         j += 1;
      }
      if (p->position_init[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[2 * i + 1] = p->position_init[i];	//vector parameter
      if (p->fix_position[i] == false) {
         working_space[shift + j] = p->position_init[i];	//vector xk
         j += 1;
      }
   }
   peak_vel = 2 * i;
   if (p->sigma_init < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[2 * i] = p->sigma_init;	//vector parameter
   if (p->fix_sigma == false) {
      working_space[shift + j] = p->sigma_init;	//vector xk
      j += 1;
   }
   if (p->t_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[2 * i + 1] = p->t_init;	//vector parameter
   if (p->fix_t == false) {
      working_space[shift + j] = p->t_init;	//vector xk
      j += 1;
   }
   if (p->b_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[2 * i + 2] = p->b_init;	//vector parameter
   if (p->fix_b == false) {
      working_space[shift + j] = p->b_init;	//vector xk
      j += 1;
   }
   if (p->s_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[2 * i + 3] = p->s_init;	//vector parameter
   if (p->fix_s == false) {
      working_space[shift + j] = p->s_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 4] = p->a0_init;	//vector parameter
   if (p->fix_a0 == false) {
      working_space[shift + j] = p->a0_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 5] = p->a1_init;	//vector parameter
   if (p->fix_a1 == false) {
      working_space[shift + j] = p->a1_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 6] = p->a2_init;	//vector parameter
   if (p->fix_a2 == false) {
      working_space[shift + j] = p->a2_init;	//vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0)
      return ("ALL PARAMETERS ARE FIXED");
   if (rozmer >= p->xmax - p->xmin + 1)
      return
          ("NUMBER OF FITTED PARAMETERS IS LARGER THAN # OF FITTED POINTS");
   for (iter = 0; iter < p->number_of_iterations; iter++) {
      for (j = 0; j < rozmer; j++) {
         working_space[2 * shift + j] = 0, working_space[3 * shift + j] = 0;	//der,temp
      }
      
          //filling vectors
          alpha = p->alpha;
      chi_opt = 0, pw = p->power - 2;
      for (i = p->xmin; i <= p->xmax; i++) {
         yw = source[i];
         ywm = yw;
         f = Shape(p->number_of_peaks, (double) i, working_space,
                    working_space[peak_vel], working_space[peak_vel + 1],
                    working_space[peak_vel + 3],
                    working_space[peak_vel + 2],
                    working_space[peak_vel + 4],
                    working_space[peak_vel + 5],
                    working_space[peak_vel + 6]);
         if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
            if (f > 0.00001)
               chi_opt += yw * TMath::Log(f) - f;
         }
         
         else {
            if (ywm != 0)
               chi_opt += (yw - f) * (yw - f) / ywm;
         }
         if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
            ywm = f;
            if (f < 0.001)
               ywm = 0.001;
         }
         
         else {
            if (ywm == 0)
               ywm = 1;
         }
         
             //calculation of gradient vector
             for (j = 0, k = 0; j < p->number_of_peaks; j++) {
            if (p->fix_amp[j] == false) {
               a = Deramp((double) i, working_space[2 * j + 1],
                           working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 3],
                           working_space[peak_vel + 2]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c;	//der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c;	//temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c;	//der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c;	//temp
                  }
               }
               k += 1;
            }
            if (p->fix_position[j] == false) {
               a = Deri0((double) i, working_space[2 * j],
                          working_space[2 * j + 1],
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
               if (p->fit_taylor == FIT1_TAYLOR_ORDER_SECOND)
                  d = Derderi0((double) i, working_space[2 * j],
                                working_space[2 * j + 1],
                                working_space[peak_vel]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (TMath::Abs(a) > 0.00000001
                       && p->fit_taylor == FIT1_TAYLOR_ORDER_SECOND) {
                     d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                     if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                        d = 0;
                  }
                  
                  else
                     d = 0;
                  a = a + d;
                  if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c;	//der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c;	//temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c;	//der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c;	//temp
                  }
               }
               k += 1;
            }
         }
         if (p->fix_sigma == false) {
            a = Dersigma(p->number_of_peaks, (double) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
            if (p->fit_taylor == FIT1_TAYLOR_ORDER_SECOND)
               d = Derdersigma(p->number_of_peaks, (double) i,
                                working_space, working_space[peak_vel]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (TMath::Abs(a) > 0.00000001
                    && p->fit_taylor == FIT1_TAYLOR_ORDER_SECOND) {
                  d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                  if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                     d = 0;
               }
               
               else
                  d = 0;
               a = a + d;
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_t == false) {
            a = Dert(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel],
                      working_space[peak_vel + 2]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_b == false) {
            a = Derb(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel], working_space[peak_vel + 1],
                      working_space[peak_vel + 2]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_s == false) {
            a = Ders(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_a0 == false) {
            a = 1.;
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_a1 == false) {
            a = Dera1((double) i);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
         if (p->fix_a2 == false) {
            a = Dera2((double) i);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;	//temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;	//der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;	//temp
               }
            }
            k += 1;
         }
      }
      for (j = 0; j < rozmer; j++) {
         if (TMath::Abs(working_space[3 * shift + j]) > 0.000001)
            working_space[2 * shift + j] = working_space[2 * shift + j] / TMath::Abs(working_space[3 * shift + j]);	//der[j]=der[j]/temp[j]
         else
            working_space[2 * shift + j] = 0;	//der[j]
      }
      
          //calculate chi_opt
          chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
          //calculate new parameters
          regul_cycle = 0;
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = working_space[shift + j];	//temp_xk[j]=xk[j]
      }
      
      do {
         if (p->alpha_optim == FIT1_ALPHA_OPTIMAL) {
            if (p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < p->number_of_peaks; i++) {
                  if (p->fix_amp[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_t == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_b == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_s == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_a1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_a2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i = p->xmin; i <= p->xmax; i++) {
                  yw = source[i];
                  ywm = yw;
                  f = Shape(p->number_of_peaks, (double) i, working_space,
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 3],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 4],
                             working_space[peak_vel + 5],
                             working_space[peak_vel + 6]);
                  if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                     if (f > 0.00001)
                        chi2 += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi2 += (yw - f) * (yw - f) / ywm;
                  }
               }
               if (chi2 < chi_min
                    && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD
                    || chi2 > chi_min
                    && p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                  pmin = pi, chi_min = chi2;
               }
               
               else
                  flag = 1;
               if (pi == 0.1)
                  chi_min = chi2;
               chi = chi_min;
            }
            if (pmin != 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < p->number_of_peaks; i++) {
                  if (p->fix_amp[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_t == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_b == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_s == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_a1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_a2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < rozmer; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+pi*alpha*der[j]
            }
            for (i = 0, j = 0; i < p->number_of_peaks; i++) {
               if (p->fix_amp[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                  j += 1;
               }
               if (p->fix_position[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                  j += 1;
               }
            }
            if (p->fix_sigma == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (p->fix_t == false) {
               working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (p->fix_b == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (p->fix_s == false) {
               working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (p->fix_a0 == false) {
               working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (p->fix_a1 == false) {
               working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (p->fix_a2 == false) {
               working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i = p->xmin; i <= p->xmax; i++) {
               yw = source[i];
               ywm = yw;
               f = Shape(p->number_of_peaks, (double) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2],
                          working_space[peak_vel + 4],
                          working_space[peak_vel + 5],
                          working_space[peak_vel + 6]);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  ywm = f;
                  if (f < 0.00001)
                     ywm = 0.00001;
               }
               if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                  if (f > 0.00001)
                     chi += yw * TMath::Log(f) - f;
               }
               
               else {
                  if (ywm != 0)
                     chi += (yw - f) * (yw - f) / ywm;
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (p->alpha_optim == FIT1_ALPHA_HALVING && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (p->alpha_optim == FIT1_ALPHA_OPTIMAL)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD
                 || chi < chi_opt
                 && p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD)
                && regul_cycle < FIT1_NUM_OF_REGUL_CYCLES);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;	//temp_xk[j]
         working_space[2 * shift + j] = 0;	//der[j]
      }
      for (i = p->xmin, chi_cel = 0; i <= p->xmax; i++) {
         yw = source[i];
         if (yw == 0)
            yw = 1;
         f = Shape(p->number_of_peaks, (double) i, working_space,
                    working_space[peak_vel], working_space[peak_vel + 1],
                    working_space[peak_vel + 3],
                    working_space[peak_vel + 2],
                    working_space[peak_vel + 4],
                    working_space[peak_vel + 5],
                    working_space[peak_vel + 6]);
         chi_opt = (yw - f) * (yw - f) / yw;
         chi_cel += (yw - f) * (yw - f) / yw;
         
             //calculate gradient vector
             for (j = 0, k = 0; j < p->number_of_peaks; j++) {
            if (p->fix_amp[j] == false) {
               a = Deramp((double) i, working_space[2 * j + 1],
                           working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 3],
                           working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_position[j] == false) {
               a = Deri0((double) i, working_space[2 * j],
                          working_space[2 * j + 1],
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
         }
         if (p->fix_sigma == false) {
            a = Dersigma(p->number_of_peaks, (double) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_t == false) {
            a = Dert(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel],
                      working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_b == false) {
            a = Derb(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel], working_space[peak_vel + 1],
                      working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_s == false) {
            a = Ders(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//tem_xk[k]
            }
            k += 1;
         }
         if (p->fix_a0 == false) {
            a = 1.0;
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_a1 == false) {
            a = Dera1((double) i);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_a2 == false) {
            a = Dera2((double) i);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;	//temp_xk[k]
            }
            k += 1;
         }
      }
   }
   b = p->xmax - p->xmin + 1 - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      p->area[i] =
          Area(working_space[2 * i], working_space[peak_vel],
               working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (p->fix_amp[i] == false) {
         p->amp_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         if (p->area[i] > 0) {
            a = Derpa(working_space[peak_vel],
                       working_space[peak_vel + 1],
                       working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            p->area_err[i] = TMath::Sqrt(TMath::Abs(a * a * b * chi_er));
         }
         
         else
            p->area_err[i] = 0;
         j += 1;
      }
      
      else {
         p->amp_calc[i] = p->amp_init[i];
         p->amp_err[i] = 0;
         p->area_err[i] = 0;
      }
      if (p->fix_position[i] == false) {
         p->position_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)	//temp[j]
            p->position_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc[i] = p->position_init[i];
         p->position_err[i] = 0;
      }
   }
   if (p->fix_sigma == false) {
      p->sigma_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc = p->sigma_init;
      p->sigma_err = 0;
   }
   if (p->fix_t == false) {
      p->t_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->t_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->t_calc = p->t_init;
      p->t_err = 0;
   }
   if (p->fix_b == false) {
      p->b_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->b_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->b_calc = p->b_init;
      p->b_err = 0;
   }
   if (p->fix_s == false) {
      p->s_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->s_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->s_calc = p->s_init;
      p->s_err = 0;
   }
   if (p->fix_a0 == false) {
      p->a0_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a0_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a0_calc = p->a0_init;
      p->a0_err = 0;
   }
   if (p->fix_a1 == false) {
      p->a1_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a1_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a1_calc = p->a1_init;
      p->a1_err = 0;
   }
   if (p->fix_a2 == false) {
      p->a2_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a2_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a2_calc = p->a2_init;
      p->a2_err = 0;
   }
   b = p->xmax - p->xmin + 1 - rozmer;
   p->chi = chi_cel / b;
   for (i = p->xmin; i <= p->xmax; i++) {
      f = Shape(p->number_of_peaks, (double) i, working_space,
                 working_space[peak_vel], working_space[peak_vel + 1],
                 working_space[peak_vel + 3], working_space[peak_vel + 2],
                 working_space[peak_vel + 4], working_space[peak_vel + 5],
                 working_space[peak_vel + 6]);
      source[i] = f;
   } delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
    
/////////////////FITTING FUNCTION WITH MATRIX INVERSION///////////////////////////////////////
void TSpectrum::StiefelInversion(double **a, int size)
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates solution of the system of linear equations.        //
//   The matrix a should have a dimension size*(size+4)                         //
//   The calling function should fill in the matrix, the column size should     //
//   contain vector y (right side of the system of equations). The result is    //
//   placed into size+1 column of the matrix.                                   //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -a-matrix with dimension size*(size+4)                          //                                            //
//              -size-number of rows of the matrix                              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j, k = 0;
   double sk = 0, b, lambdak, normk, normk_old = 0;
   
   do {
      normk = 0;
      
          //calculation of rk and norm
          for (i = 0; i < size; i++) {
         a[i][size + 2] = -a[i][size];	//rk=-C
         for (j = 0; j < size; j++) {
            a[i][size + 2] += a[i][j] * a[j][size + 1];	//A*xk-C
         }
         normk += a[i][size + 2] * a[i][size + 2];	//calculation normk
      }
      
          //calculation of sk
          if (k != 0) {
         sk = normk / normk_old;
      }
      
          //calculation of uk
          for (i = 0; i < size; i++) {
         a[i][size + 3] = -a[i][size + 2] + sk * a[i][size + 3];	//uk=-rk+sk*uk-1
      }
      
          //calculation of lambdak
          lambdak = 0;
      for (i = 0; i < size; i++) {
         for (j = 0, b = 0; j < size; j++) {
            b += a[i][j] * a[j][size + 3];	//A*uk
         }
         lambdak += b * a[i][size + 3];
      }
      if (TMath::Abs(lambdak) > 1e-50)	//computer zero
         lambdak = normk / lambdak;
      
      else
         lambdak = 0;
      for (i = 0; i < size; i++)
         a[i][size + 1] += lambdak * a[i][size + 3];	//xk+1=xk+lambdak*uk
      normk_old = normk;
      k += 1;
   } while (k < size && TMath::Abs(normk) > 1e-50);	//computer zero
   return;
}
const char *TSpectrum::Fit1Stiefel(float *source, TSpectrumOneDimFit * p,
                                    int size) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL FIT FUNCTION				           */ 
/*      ALGORITHM WITH MATRIX INVERSION (STIEFEL-HESTENS METHOD)           */ 
/*	This function fits the source spectrum. The calling program should */ 
/*      fill in input parameters of the TSpectrumOneDimFit class 	   */ 
/*	The fitted parameters are written into class pointed by 	   */ 
/*	TSpectrumOneDimFit class pointer and fitted data are written into  */ 
/*      source spectrum.                                                   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum			   */ 
/*	p-pointer to the TSpectrumOneDimFit class, see manual       	   */ 
/*	size-length of source spectrum                                     */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, shift =
       2 * p->number_of_peaks + 7, peak_vel, rozmer, iter, regul_cycle,
       flag;
   double a, b, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   if (size <= 0)
      return "Wrong Parameters";
   if (p->number_of_peaks <= 0)
      return ("INVALID NUMBER OF PEAKS, MUST BE POSITIVE");
   if (p->number_of_iterations <= 0)
      return ("INVALID NUMBER OF ITERATIONS, MUST BE POSITIVE");
   if (p->alpha <= 0 || p->alpha > 1)
      return ("INVALID COEFFICIENT ALPHA, MUST BE > THAN 0 AND <=1");
   if (p->statistic_type != FIT1_OPTIM_CHI_COUNTS
        && p->statistic_type != FIT1_OPTIM_CHI_FUNC_VALUES
        && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD)
      return ("WRONG TYPE OF STATISTIC");
   if (p->alpha_optim != FIT1_ALPHA_HALVING
        && p->alpha_optim != FIT1_ALPHA_OPTIMAL)
      return ("WRONG OPTIMIZATION ALGORITHM");
   if (p->xmin < 0 || p->xmin > p->xmax)
      return ("INVALID LOW LIMIT OF FITTING REGION");
   if (p->xmax >= size || p->xmax < p->xmin)
      return ("INVALID HIGH LIMIT OF FITTING REGION");
   double *working_space = new double[5 * (2 * p->number_of_peaks + 7)];
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      if (p->amp_init[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[2 * i] = p->amp_init[i];	//vector parameter
      if (p->fix_amp[i] == false) {
         working_space[shift + j] = p->amp_init[i];	//vector xk
         j += 1;
      }
      if (p->position_init[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[2 * i + 1] = p->position_init[i];	//vector parameter
      if (p->fix_position[i] == false) {
         working_space[shift + j] = p->position_init[i];	//vector xk
         j += 1;
      }
   }
   peak_vel = 2 * i;
   if (p->sigma_init < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[2 * i] = p->sigma_init;	//vector parameter
   if (p->fix_sigma == false) {
      working_space[shift + j] = p->sigma_init;	//vector xk
      j += 1;
   }
   if (p->t_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[2 * i + 1] = p->t_init;	//vector parameter
   if (p->fix_t == false) {
      working_space[shift + j] = p->t_init;	//vector xk
      j += 1;
   }
   if (p->b_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[2 * i + 2] = p->b_init;	//vector parameter
   if (p->fix_b == false) {
      working_space[shift + j] = p->b_init;	//vector xk
      j += 1;
   }
   if (p->s_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[2 * i + 3] = p->s_init;	//vector parameter
   if (p->fix_s == false) {
      working_space[shift + j] = p->s_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 4] = p->a0_init;	//vector parameter
   if (p->fix_a0 == false) {
      working_space[shift + j] = p->a0_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 5] = p->a1_init;	//vector parameter
   if (p->fix_a1 == false) {
      working_space[shift + j] = p->a1_init;	//vector xk
      j += 1;
   }
   working_space[2 * i + 6] = p->a2_init;	//vector parameter
   if (p->fix_a2 == false) {
      working_space[shift + j] = p->a2_init;	//vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0)
      return ("ALL PARAMETERS ARE FIXED");
   if (rozmer >= p->xmax - p->xmin + 1)
      return
          ("NUMBER OF FITTED PARAMETERS IS LARGER THAN # OF FITTED POINTS");
   double **working_matrix = new double *[rozmer];
   for (i = 0; i < rozmer; i++)
      working_matrix[i] = new double[rozmer + 4];
   for (iter = 0; iter < p->number_of_iterations; iter++) {
      for (j = 0; j < rozmer; j++) {
         working_space[3 * shift + j] = 0;	//temp
         for (k = 0; k <= rozmer; k++) {
            working_matrix[j][k] = 0;
         }
      }
      
          //filling working matrix
          alpha = p->alpha;
      chi_opt = 0;
      for (i = p->xmin; i <= p->xmax; i++) {
         
             //calculation of gradient vector
             for (j = 0, k = 0; j < p->number_of_peaks; j++) {
            if (p->fix_amp[j] == false) {
               working_space[2 * shift + k] =
                   Deramp((double) i, working_space[2 * j + 1],
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
               k += 1;
            }
            if (p->fix_position[j] == false) {
               working_space[2 * shift + k] =
                   Deri0((double) i, working_space[2 * j],
                         working_space[2 * j + 1], working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 3],
                         working_space[peak_vel + 2]);
               k += 1;
            }
         } if (p->fix_sigma == false) {
            working_space[2 * shift + k] =
                Dersigma(p->number_of_peaks, (double) i, working_space,
                         working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 3],
                         working_space[peak_vel + 2]);
            k += 1;
         }
         if (p->fix_t == false) {
            working_space[2 * shift + k] =
                Dert(p->number_of_peaks, (double) i, working_space,
                     working_space[peak_vel], working_space[peak_vel + 2]);
            k += 1;
         }
         if (p->fix_b == false) {
            working_space[2 * shift + k] =
                Derb(p->number_of_peaks, (double) i, working_space,
                     working_space[peak_vel], working_space[peak_vel + 1],
                     working_space[peak_vel + 2]);
            k += 1;
         }
         if (p->fix_s == false) {
            working_space[2 * shift + k] =
                Ders(p->number_of_peaks, (double) i, working_space,
                     working_space[peak_vel]);
            k += 1;
         }
         if (p->fix_a0 == false) {
            working_space[2 * shift + k] = 1.;
            k += 1;
         }
         if (p->fix_a1 == false) {
            working_space[2 * shift + k] = Dera1((double) i);
            k += 1;
         }
         if (p->fix_a2 == false) {
            working_space[2 * shift + k] = Dera2((double) i);
            k += 1;
         }
         yw = source[i];
         ywm = yw;
         f = Shape(p->number_of_peaks, (double) i, working_space,
                    working_space[peak_vel], working_space[peak_vel + 1],
                    working_space[peak_vel + 3],
                    working_space[peak_vel + 2],
                    working_space[peak_vel + 4],
                    working_space[peak_vel + 5],
                    working_space[peak_vel + 6]);
         if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
            if (f > 0.00001)
               chi_opt += yw * TMath::Log(f) - f;
         }
         
         else {
            if (ywm != 0)
               chi_opt += (yw - f) * (yw - f) / ywm;
         }
         if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else {
            if (ywm == 0)
               ywm = 1;
         }
         for (j = 0; j < rozmer; j++) {
            for (k = 0; k < rozmer; k++) {
               b = working_space[2 * shift +
                                  j] * working_space[2 * shift + k] / ywm;
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES)
                  b = b * (4 * yw - 2 * f) / ywm;
               working_matrix[j][k] += b;
               if (j == k)
                  working_space[3 * shift + j] += b;
            }
         }
         if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES)
            b = (f * f - yw * yw) / (ywm * ywm);
         
         else
            b = (f - yw) / ywm;
         for (j = 0; j < rozmer; j++) {
            working_matrix[j][rozmer] -= b * working_space[2 * shift + j];
         }
      }
      for (i = 0; i < rozmer; i++) {
         working_matrix[i][rozmer + 1] = 0;	//xk
      }
      StiefelInversion(working_matrix, rozmer);
      for (i = 0; i < rozmer; i++) {
         working_space[2 * shift + i] = working_matrix[i][rozmer + 1];	//der
      }
      
          //calculate chi_opt
          chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
          //calculate new parameters
          regul_cycle = 0;
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = working_space[shift + j];	//temp_xk[j]=xk[j]
      }
      
      do {
         if (p->alpha_optim == FIT1_ALPHA_OPTIMAL) {
            if (p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < p->number_of_peaks; i++) {
                  if (p->fix_amp[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_t == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_b == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_s == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_a1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_a2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i = p->xmin; i <= p->xmax; i++) {
                  yw = source[i];
                  ywm = yw;
                  f = Shape(p->number_of_peaks, (double) i, working_space,
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 3],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 4],
                             working_space[peak_vel + 5],
                             working_space[peak_vel + 6]);
                  if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                     if (f > 0.00001)
                        chi2 += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi2 += (yw - f) * (yw - f) / ywm;
                  }
               }
               if (chi2 < chi_min
                    && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD
                    || chi2 > chi_min
                    && p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                  pmin = pi, chi_min = chi2;
               }
               
               else
                  flag = 1;
               if (pi == 0.1)
                  chi_min = chi2;
               chi = chi_min;
            }
            if (pmin != 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < p->number_of_peaks; i++) {
                  if (p->fix_amp[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_t == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_b == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_s == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_a1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_a2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < rozmer; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j];	//xk[j]=temp_xk[j]+alpha*der[j]
            }
            for (i = 0, j = 0; i < p->number_of_peaks; i++) {
               if (p->fix_amp[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[2 * i] = working_space[shift + j];	//parameter[2*i]=xk[j]
                  j += 1;
               }
               if (p->fix_position[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[2 * i + 1] = working_space[shift + j];	//parameter[2*i+1]=xk[j]
                  j += 1;
               }
            }
            if (p->fix_sigma == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (p->fix_t == false) {
               working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (p->fix_b == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (p->fix_s == false) {
               working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (p->fix_a0 == false) {
               working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (p->fix_a1 == false) {
               working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (p->fix_a2 == false) {
               working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i = p->xmin; i <= p->xmax; i++) {
               yw = source[i];
               ywm = yw;
               f = Shape(p->number_of_peaks, (double) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2],
                          working_space[peak_vel + 4],
                          working_space[peak_vel + 5],
                          working_space[peak_vel + 6]);
               if (p->statistic_type == FIT1_OPTIM_CHI_FUNC_VALUES) {
                  ywm = f;
                  if (f < 0.00001)
                     ywm = 0.00001;
               }
               if (p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD) {
                  if (f > 0.00001)
                     chi += yw * TMath::Log(f) - f;
               }
               
               else {
                  if (ywm != 0)
                     chi += (yw - f) * (yw - f) / ywm;
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (p->alpha_optim == FIT1_ALPHA_HALVING && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (p->alpha_optim == FIT1_ALPHA_OPTIMAL)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && p->statistic_type != FIT1_OPTIM_MAX_LIKELIHOOD
                 || chi < chi_opt
                 && p->statistic_type == FIT1_OPTIM_MAX_LIKELIHOOD)
                && regul_cycle < FIT1_NUM_OF_REGUL_CYCLES);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;	//temp_xk[j]
         working_space[2 * shift + j] = 0;	//der[j]
      }
      for (i = p->xmin, chi_cel = 0; i <= p->xmax; i++) {
         yw = source[i];
         if (yw == 0)
            yw = 1;
         f = Shape(p->number_of_peaks, (double) i, working_space,
                    working_space[peak_vel], working_space[peak_vel + 1],
                    working_space[peak_vel + 3],
                    working_space[peak_vel + 2],
                    working_space[peak_vel + 4],
                    working_space[peak_vel + 5],
                    working_space[peak_vel + 6]);
         chi_opt = (yw - f) * (yw - f) / yw;
         chi_cel += (yw - f) * (yw - f) / yw;
         
             //calculate gradient vector
             for (j = 0, k = 0; j < p->number_of_peaks; j++) {
            if (p->fix_amp[j] == false) {
               a = Deramp((double) i, working_space[2 * j + 1],
                           working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 3],
                           working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_position[j] == false) {
               a = Deri0((double) i, working_space[2 * j],
                          working_space[2 * j + 1],
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
         }
         if (p->fix_sigma == false) {
            a = Dersigma(p->number_of_peaks, (double) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_t == false) {
            a = Dert(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel],
                      working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_b == false) {
            a = Derb(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel], working_space[peak_vel + 1],
                      working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_s == false) {
            a = Ders(p->number_of_peaks, (double) i, working_space,
                      working_space[peak_vel]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//tem_xk[k]
            }
            k += 1;
         }
         if (p->fix_a0 == false) {
            a = 1.0;
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_a1 == false) {
            a = Dera1((double) i);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
         if (p->fix_a2 == false) {
            a = Dera2((double) i);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;	//der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;	//temp_xk[k]
            }
            k += 1;
         }
      }
   }
   b = p->xmax - p->xmin + 1 - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      p->area[i] =
          Area(working_space[2 * i], working_space[peak_vel],
               working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (p->fix_amp[i] == false) {
         p->amp_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         if (p->area[i] > 0) {
            a = Derpa(working_space[peak_vel],
                       working_space[peak_vel + 1],
                       working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            p->area_err[i] = TMath::Sqrt(TMath::Abs(a * a * b * chi_er));
         }
         
         else
            p->area_err[i] = 0;
         j += 1;
      }
      
      else {
         p->amp_calc[i] = p->amp_init[i];
         p->amp_err[i] = 0;
         p->area_err[i] = 0;
      }
      if (p->fix_position[i] == false) {
         p->position_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)	//temp[j]
            p->position_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//Der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc[i] = p->position_init[i];
         p->position_err[i] = 0;
      }
   }
   if (p->fix_sigma == false) {
      p->sigma_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc = p->sigma_init;
      p->sigma_err = 0;
   }
   if (p->fix_t == false) {
      p->t_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->t_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->t_calc = p->t_init;
      p->t_err = 0;
   }
   if (p->fix_b == false) {
      p->b_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->b_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->b_calc = p->b_init;
      p->b_err = 0;
   }
   if (p->fix_s == false) {
      p->s_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->s_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->s_calc = p->s_init;
      p->s_err = 0;
   }
   if (p->fix_a0 == false) {
      p->a0_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a0_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a0_calc = p->a0_init;
      p->a0_err = 0;
   }
   if (p->fix_a1 == false) {
      p->a1_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a1_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a1_calc = p->a1_init;
      p->a1_err = 0;
   }
   if (p->fix_a2 == false) {
      p->a2_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->a2_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->a2_calc = p->a2_init;
      p->a2_err = 0;
   }
   b = p->xmax - p->xmin + 1 - rozmer;
   p->chi = chi_cel / b;
   for (i = p->xmin; i <= p->xmax; i++) {
      f = Shape(p->number_of_peaks, (double) i, working_space,
                 working_space[peak_vel], working_space[peak_vel + 1],
                 working_space[peak_vel + 3], working_space[peak_vel + 2],
                 working_space[peak_vel + 4], working_space[peak_vel + 5],
                 working_space[peak_vel + 6]);
      source[i] = f;
   } for (i = 0; i < rozmer; i++)
      delete[]working_matrix[i];
   delete[]working_matrix;
   delete[]working_space;
   return 0;
}


//____________________________________________________________________________
    
//////////AUXILIARY FUNCTIONS FOR TRANSFORM BASED FUNCTIONS////////////////////////
void TSpectrum::Haar(float *working_space, int num, int direction) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Haar transform of a part of data                   //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, ii, li, l2, l3, j, jj, jj1, lj, iter, m, jmin, jmax;
   double a, b, c, wlk;
   float val;
   for (i = 0; i < num; i++)
      working_space[i + num] = 0;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   if (direction == TRANSFORM1_FORWARD) {
      for (m = 1; m <= iter; m++) {
         li = iter + 1 - m;
         l2 = (int) TMath::Power(2, li - 1);
         for (i = 0; i < (2 * l2); i++) {
            working_space[num + i] = working_space[i];
         }
         for (j = 0; j < l2; j++) {
            l3 = l2 + j;
            jj = 2 * j;
            val = working_space[jj + num] + working_space[jj + 1 + num];
            working_space[j] = val;
            val = working_space[jj + num] - working_space[jj + 1 + num];
            working_space[l3] = val;
         }
      }
   }
   val = working_space[0];
   val = val / TMath::Sqrt(TMath::Power(2, iter));
   working_space[0] = val;
   val = working_space[1];
   val = val / TMath::Sqrt(TMath::Power(2, iter));
   working_space[1] = val;
   for (ii = 2; ii <= iter; ii++) {
      i = ii - 1;
      wlk = 1 / TMath::Sqrt(TMath::Power(2, iter - i));
      jmin = (int) TMath::Power(2, i);
      jmax = (int) TMath::Power(2, ii) - 1;
      for (j = jmin; j <= jmax; j++) {
         val = working_space[j];
         a = val;
         a = a * wlk;
         val = a;
         working_space[j] = val;
      }
   }
   if (direction == TRANSFORM1_INVERSE) {
      for (m = 1; m <= iter; m++) {
         a = 2;
         b = m - 1;
         c = TMath::Power(a, b);
         li = (int) c;
         for (i = 0; i < (2 * li); i++) {
            working_space[i + num] = working_space[i];
         }
         for (j = 0; j < li; j++) {
            lj = li + j;
            jj = 2 * (j + 1) - 1;
            jj1 = jj - 1;
            val = working_space[j + num] - working_space[lj + num];
            working_space[jj] = val;
            val = working_space[j + num] + working_space[lj + num];
            working_space[jj1] = val;
         }
      }
   }
   return;
}
void TSpectrum::Walsh(float *working_space, int num) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Walsh transform of a part of data                  //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, m, nump = 1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter;
   double a;
   float val1, val2;
   for (i = 0; i < num; i++)
      working_space[i + num] = 0;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   for (m = 1; m <= iter; m++) {
      if (m == 1)
         nump = 1;
      
      else
         nump = nump * 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      for (mp = 0; mp < nump; mp++) {
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            working_space[iba + num] = val1 + val2;
            working_space[mnum21 + num] = val1 - val2;
         }
      }
      for (i = 0; i < num; i++) {
         working_space[i] = working_space[i + num];
      }
   }
   a = num;
   a = TMath::Sqrt(a);
   val2 = a;
   for (i = 0; i < num; i++) {
      val1 = working_space[i];
      val1 = val1 / val2;
      working_space[i] = val1;
   }
   return;
}
void TSpectrum::BitReverse(float *working_space, int num) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion carries out bir-reverse reordering of data                    //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of processed data              //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int ipower[26];
   int i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + num] = working_space[i];
   }
   for (i = 1; i <= num; i++) {
      ib = i - 1;
      il = 1;
    lab9:ibd = ib / 2;
      ipower[il - 1] = 1;
      if (ib == (ibd * 2))
         ipower[il - 1] = 0;
      if (ibd == 0)
         goto lab10;
      ib = ibd;
      il = il + 1;
      goto lab9;
    lab10:ip = 1;
      ifac = num;
      for (i1 = 1; i1 <= il; i1++) {
         ifac = ifac / 2;
         ip = ip + ifac * ipower[i1 - 1];
      }
      working_space[ip - 1] = working_space[i - 1 + num];
   }
   return;
}
void TSpectrum::Fourier(float *working_space, int num, int hartley,
                          int direction, int zt_clear) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Fourier based transform of a part of data          //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -hartley-1 if it is Hartley transform, 0 othewise               //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int nxp2, nxp, i, j, k, m, iter, mxp, j1, j2, n1, n2, it;
   double a, b, c, d, sign, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4;
   if (direction == TRANSFORM1_FORWARD && zt_clear == 0) {
      for (i = 0; i < num; i++)
         working_space[i + num] = 0;
   }
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   sign = -1;
   if (direction == TRANSFORM1_INVERSE)
      sign = 1;
   nxp2 = num;
   for (it = 1; it <= iter; it++) {
      nxp = nxp2;
      nxp2 = nxp / 2;
      a = nxp2;
      wpwr = pi / a;
      for (m = 1; m <= nxp2; m++) {
         a = m - 1;
         arg = a * wpwr;
         wr = TMath::Cos(arg);
         wi = sign * TMath::Sin(arg);
         for (mxp = nxp; mxp <= num; mxp += nxp) {
            j1 = mxp - nxp + m;
            j2 = j1 + nxp2;
            val1 = working_space[j1 - 1];
            val2 = working_space[j2 - 1];
            val3 = working_space[j1 - 1 + num];
            val4 = working_space[j2 - 1 + num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a - b;
            ti = c - d;
            a = a + b;
            val1 = a;
            working_space[j1 - 1] = val1;
            c = c + d;
            val1 = c;
            working_space[j1 - 1 + num] = val1;
            a = tr * wr - ti * wi;
            val1 = a;
            working_space[j2 - 1] = val1;
            a = ti * wr + tr * wi;
            val1 = a;
            working_space[j2 - 1 + num] = val1;
         }
      }
   }
   n2 = num / 2;
   n1 = num - 1;
   j = 1;
   for (i = 1; i <= n1; i++) {
      if (i >= j)
         goto lab55;
      val1 = working_space[j - 1];
      val2 = working_space[j - 1 + num];
      val3 = working_space[i - 1];
      working_space[j - 1] = val3;
      working_space[j - 1 + num] = working_space[i - 1 + num];
      working_space[i - 1] = val1;
      working_space[i - 1 + num] = val2;
    lab55:k = n2;
    lab60:if (k >= j)
         goto lab65;
      j = j - k;
      k = k / 2;
      goto lab60;
    lab65:j = j + k;
   }
   a = num;
   a = TMath::Sqrt(a);
   for (i = 0; i < num; i++) {
      if (hartley == 0) {
         val1 = working_space[i];
         b = val1;
         b = b / a;
         val1 = b;
         working_space[i] = val1;
         b = working_space[i + num];
         b = b / a;
         working_space[i + num] = b;
      }
      
      else {
         b = working_space[i];
         c = working_space[i + num];
         b = (b + c) / a;
         working_space[i] = b;
         working_space[i + num] = 0;
      }
   }
   if (hartley == 1 && direction == TRANSFORM1_INVERSE) {
      for (i = 1; i < num; i++)
         working_space[num - i + num] = working_space[i];
      working_space[0 + num] = working_space[0];
      for (i = 0; i < num; i++) {
         working_space[i] = working_space[i + num];
         working_space[i + num] = 0;
      }
   }
   return;
}
void TSpectrum::BitReverseHaar(float *working_space, int shift, int num,
                                 int start) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion carries out bir-reverse reordering for Haar transform         //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of processed data              //
//              -shift-shift of position of processing                          //
//              -start-initial position of processed data                       //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int ipower[26];
   int i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + shift + start] = working_space[i + start];
      working_space[i + shift + start + 2 * shift] =
          working_space[i + start + 2 * shift];
   }
   for (i = 1; i <= num; i++) {
      ib = i - 1;
      il = 1;
    lab9:ibd = ib / 2;
      ipower[il - 1] = 1;
      if (ib == (ibd * 2))
         ipower[il - 1] = 0;
      if (ibd == 0)
         goto lab10;
      ib = ibd;
      il = il + 1;
      goto lab9;
    lab10:ip = 1;
      ifac = num;
      for (i1 = 1; i1 <= il; i1++) {
         ifac = ifac / 2;
         ip = ip + ifac * ipower[i1 - 1];
      }
      working_space[ip - 1 + start] =
          working_space[i - 1 + shift + start];
      working_space[ip - 1 + start + 2 * shift] =
          working_space[i - 1 + shift + start + 2 * shift];
   }
   return;
}
int TSpectrum::GeneralExe(float *working_space, int zt_clear, int num,
                            int degree, int type) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates generalized (mixed) transforms of different degrees//
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -zt_clear-flag to clear imaginary data before staring           //
//              -num-length of processed data                                   //
//              -degree-degree of transform (see manual)                        //
//              -type-type of mixed transform (see manual)                      //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j, k, m, nump, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter,
       mp2step, mppom, ring;
   double a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
   if (zt_clear == 0) {
      for (i = 0; i < num; i++)
         working_space[i + 2 * num] = 0;
   }
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   a = num;
   wpwr = 2.0 * pi / a;
   nump = num;
   mp2step = 1;
   ring = num;
   for (i = 0; i < iter - degree; i++)
      ring = ring / 2;
   for (m = 1; m <= iter; m++) {
      nump = nump / 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      if (m > degree
           && (type == TRANSFORM1_FOURIER_HAAR
               || type == TRANSFORM1_WALSH_HAAR
               || type == TRANSFORM1_COS_HAAR
               || type == TRANSFORM1_SIN_HAAR))
         mp2step *= 2;
      if (ring > 1)
         ring = ring / 2;
      for (mp = 0; mp < nump; mp++) {
         if (type != TRANSFORM1_WALSH_HAAR) {
            mppom = mp;
            mppom = mppom % ring;
            a = 0;
            j = 1;
            k = num / 4;
            for (i = 0; i < (iter - 1); i++) {
               if ((mppom & j) != 0)
                  a = a + k;
               j = j * 2;
               k = k / 2;
            }
            arg = a * wpwr;
            wr = TMath::Cos(arg);
            wi = TMath::Sin(arg);
         }
         
         else {
            wr = 1;
            wi = 0;
         }
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            if (mp2 % mp2step == 0) {
               a0r = a0oldr;
               b0r = b0oldr;
               a0r = 1 / TMath::Sqrt(2.0);
               b0r = 1 / TMath::Sqrt(2.0);
            }
            
            else {
               a0r = 1;
               b0r = 0;
            }
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            val3 = working_space[iba + 2 * num];
            val4 = working_space[mnum21 + 2 * num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a * a0r + b * b0r;
            val1 = tr;
            working_space[num + iba] = val1;
            ti = c * a0r + d * b0r;
            val1 = ti;
            working_space[num + iba + 2 * num] = val1;
            tr =
                a * b0r * wr - c * b0r * wi - b * a0r * wr + d * a0r * wi;
            val1 = tr;
            working_space[num + mnum21] = val1;
            ti =
                c * b0r * wr + a * b0r * wi - d * a0r * wr - b * a0r * wi;
            val1 = ti;
            working_space[num + mnum21 + 2 * num] = val1;
         }
      }
      for (i = 0; i < num; i++) {
         val1 = working_space[num + i];
         working_space[i] = val1;
         val1 = working_space[num + i + 2 * num];
         working_space[i + 2 * num] = val1;
      }
   }
   return (0);
}
int TSpectrum::GeneralInv(float *working_space, int num, int degree,
                            int type) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates inverse generalized (mixed) transforms             //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -degree-degree of transform (see manual)                        //
//              -type-type of mixed transform (see manual)                      //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j, k, m, nump =
       1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter, mp2step, mppom,
       ring;
   double a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   a = num;
   wpwr = 2.0 * pi / a;
   mp2step = 1;
   if (type == TRANSFORM1_FOURIER_HAAR || type == TRANSFORM1_WALSH_HAAR
        || type == TRANSFORM1_COS_HAAR || type == TRANSFORM1_SIN_HAAR) {
      for (i = 0; i < iter - degree; i++)
         mp2step *= 2;
   }
   ring = 1;
   for (m = 1; m <= iter; m++) {
      if (m == 1)
         nump = 1;
      
      else
         nump = nump * 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      if (m > iter - degree + 1)
         ring *= 2;
      for (mp = nump - 1; mp >= 0; mp--) {
         if (type != TRANSFORM1_WALSH_HAAR) {
            mppom = mp;
            mppom = mppom % ring;
            a = 0;
            j = 1;
            k = num / 4;
            for (i = 0; i < (iter - 1); i++) {
               if ((mppom & j) != 0)
                  a = a + k;
               j = j * 2;
               k = k / 2;
            }
            arg = a * wpwr;
            wr = TMath::Cos(arg);
            wi = TMath::Sin(arg);
         }
         
         else {
            wr = 1;
            wi = 0;
         }
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            if (mp2 % mp2step == 0) {
               a0r = a0oldr;
               b0r = b0oldr;
               a0r = 1 / TMath::Sqrt(2.0);
               b0r = 1 / TMath::Sqrt(2.0);
            }
            
            else {
               a0r = 1;
               b0r = 0;
            }
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            val3 = working_space[iba + 2 * num];
            val4 = working_space[mnum21 + 2 * num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a * a0r + b * wr * b0r + d * wi * b0r;
            val1 = tr;
            working_space[num + iba] = val1;
            ti = c * a0r + d * wr * b0r - b * wi * b0r;
            val1 = ti;
            working_space[num + iba + 2 * num] = val1;
            tr = a * b0r - b * wr * a0r - d * wi * a0r;
            val1 = tr;
            working_space[num + mnum21] = val1;
            ti = c * b0r - d * wr * a0r + b * wi * a0r;
            val1 = ti;
            working_space[num + mnum21 + 2 * num] = val1;
         }
      }
      if (m <= iter - degree
           && (type == TRANSFORM1_FOURIER_HAAR
               || type == TRANSFORM1_WALSH_HAAR
               || type == TRANSFORM1_COS_HAAR
               || type == TRANSFORM1_SIN_HAAR))
         mp2step /= 2;
      for (i = 0; i < num; i++) {
         val1 = working_space[num + i];
         working_space[i] = val1;
         val1 = working_space[num + i + 2 * num];
         working_space[i + 2 * num] = val1;
      }
   }
   return (0);
}


//////////END OF AUXILIARY FUNCTIONS FOR TRANSFORM! FUNCTION////////////////////////
    
//////////TRANSFORM1 FUNCTION - CALCULATES DIFFERENT 1-D DIRECT AND INVERSE ORTHOGONAL TRANSFORMS//////
const char *TSpectrum::Transform1(const float *source, float *dest,
                                  int size, int type, int direction,
                                  int degree) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL TRANSFORM FUNCTION				   */ 
/*	This function transforms the source spectrum. The calling program  */ 
/*      should fill in input parameters.                         	   */ 
/*	Transformed data are written into dest spectrum.                   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum, its length should */ 
/*             be size except for inverse FOURIER, FOUR-WALSh, FOUR-HAAR   */ 
/*             transform. These need 2*size length to supply real and      */ 
/*             imaginary coefficients.                                     */ 
/*	dest-pointer to the vector of dest data, its length should be      */ 
/*           size except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These  */ 
/*           need 2*size length to store real and imaginary coefficients   */ 
/*	size-basic length of source and dest spectra                       */ 
/*	type-type of transform                                             */ 
/*      direction-transform direction (forward, inverse)                   */ 
/*      degree-applied only for mixed transforms                           */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, n, k = 1, m, l;
   float val;
   double a, b, pi = 3.14159265358979323846;
   float *working_space = 0;
   if (size <= 0)
      return "Wrong Parameters";
   j = 0;
   n = 1;
   for (; n < size;) {
      j += 1;
      n = n * 2;
   }
   if (n != size)
      return ("LENGTH MUST BE POWER OF 2");
   if (type < TRANSFORM1_HAAR || type > TRANSFORM1_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (direction != TRANSFORM1_FORWARD && direction != TRANSFORM1_INVERSE)
      return ("WRONG TRANSFORM DIRECTION");
   if (type >= TRANSFORM1_FOURIER_WALSH && type <= TRANSFORM1_SIN_HAAR) {
      if (degree > j || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM1_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      j = size / k;
   }
   switch (type) {
   case TRANSFORM1_HAAR:
   case TRANSFORM1_WALSH:
      working_space = new float[2 * size];
      break;
   case TRANSFORM1_COS:
   case TRANSFORM1_SIN:
   case TRANSFORM1_FOURIER:
   case TRANSFORM1_HARTLEY:
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
      working_space = new float[4 * size];
      break;
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      working_space = new float[8 * size];
      break;
   }
   if (direction == TRANSFORM1_FORWARD) {
      switch (type) {
      case TRANSFORM1_HAAR:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Haar(working_space, size, direction);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_WALSH:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Walsh(working_space, size);
         BitReverse(working_space, size);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_COS:
         size = 2 * size;
         for (i = 1; i <= (size / 2); i++) {
            val = source[i - 1];
            working_space[i - 1] = val;
            working_space[size - i] = val;
         }
         Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
         for (i = 0; i < size / 2; i++) {
            a = pi * (double) i / (double) size;
            a = TMath::Cos(a);
            b = working_space[i];
            a = b / a;
            working_space[i] = a;
            working_space[i + size] = 0;
         } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
         for (i = 0; i < size / 2; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_SIN:
         size = 2 * size;
         for (i = 1; i <= (size / 2); i++) {
            val = source[i - 1];
            working_space[i - 1] = val;
            working_space[size - i] = -val;
         }
         Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
         for (i = 0; i < size / 2; i++) {
            a = pi * (double) i / (double) size;
            a = TMath::Sin(a);
            b = working_space[i];
            if (a != 0)
               a = b / a;
            working_space[i - 1] = a;
            working_space[i + size] = 0;
         }
         working_space[size / 2 - 1] =
             working_space[size / 2] / TMath::Sqrt(2.0);
         for (i = 0; i < size / 2; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_FOURIER:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
         for (i = 0; i < 2 * size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_HARTLEY:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, size, 1, TRANSFORM1_FORWARD, 0);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_FOURIER_WALSH:
      case TRANSFORM1_FOURIER_HAAR:
      case TRANSFORM1_WALSH_HAAR:
      case TRANSFORM1_COS_WALSH:
      case TRANSFORM1_COS_HAAR:
      case TRANSFORM1_SIN_WALSH:
      case TRANSFORM1_SIN_HAAR:
         for (i = 0; i < size; i++) {
            val = source[i];
            if (type == TRANSFORM1_COS_WALSH
                 || type == TRANSFORM1_COS_HAAR) {
               j = (int) TMath::Power(2, degree) / 2;
               k = i / j;
               k = 2 * k * j;
               working_space[k + i % j] = val;
               working_space[k + 2 * j - 1 - i % j] = val;
            }
            
            else if (type == TRANSFORM1_SIN_WALSH
                     || type == TRANSFORM1_SIN_HAAR) {
               j = (int) TMath::Power(2, degree) / 2;
               k = i / j;
               k = 2 * k * j;
               working_space[k + i % j] = val;
               working_space[k + 2 * j - 1 - i % j] = -val;
            }
            
            else
               working_space[i] = val;
         }
         if (type == TRANSFORM1_FOURIER_WALSH
              || type == TRANSFORM1_FOURIER_HAAR
              || type == TRANSFORM1_WALSH_HAAR) {
            for (i = 0; i < j; i++)
               BitReverseHaar(working_space, size, k, i * k);
            GeneralExe(working_space, 0, size, degree, type);
         }
         
         else if (type == TRANSFORM1_COS_WALSH
                  || type == TRANSFORM1_COS_HAAR) {
            m = (int) TMath::Power(2, degree);
            l = 2 * size / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * size, m, i * m);
            GeneralExe(working_space, 0, 2 * size, degree, type);
            for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
               a = TMath::Cos(a);
               b = working_space[k + i % j];
               if (i % j == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_space[i] = a;
               working_space[i + 2 * size] = 0;
            }
         }
         
         else if (type == TRANSFORM1_SIN_WALSH
                  || type == TRANSFORM1_SIN_HAAR) {
            m = (int) TMath::Power(2, degree);
            l = 2 * size / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * size, m, i * m);
            GeneralExe(working_space, 0, 2 * size, degree, type);
            for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
               a = TMath::Cos(a);
               b = working_space[j + k + i % j];
               if (i % j == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_space[j + k / 2 - i % j - 1] = a;
               working_space[i + 2 * size] = 0;
            }
         }
         if (type > TRANSFORM1_WALSH_HAAR)
            k = (int) TMath::Power(2, degree - 1);
         
         else
            k = (int) TMath::Power(2, degree);
         j = size / k;
         for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
            working_space[size + i] = working_space[l + i / j];
            working_space[size + i + 2 * size] =
                working_space[l + i / j + 2 * size];
         }
         for (i = 0; i < size; i++) {
            working_space[i] = working_space[size + i];
            working_space[i + 2 * size] =
                working_space[size + i + 2 * size];
         }
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         if (type == TRANSFORM1_FOURIER_WALSH
              || type == TRANSFORM1_FOURIER_HAAR) {
            for (i = 0; i < size; i++) {
               dest[size + i] = working_space[i + 2 * size];
            }
         }
         break;
      }
   }
   
   else if (direction == TRANSFORM1_INVERSE) {
      switch (type) {
      case TRANSFORM1_HAAR:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Haar(working_space, size, direction);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_WALSH:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         BitReverse(working_space, size);
         Walsh(working_space, size);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_COS:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         size = 2 * size;
         working_space[0] = working_space[0] * TMath::Sqrt(2.0);
         for (i = 0; i < size / 2; i++) {
            a = pi * (double) i / (double) size;
            b = TMath::Sin(a);
            a = TMath::Cos(a);
            working_space[i + size] = (double) working_space[i] * b;
            working_space[i] = (double) working_space[i] * a;
         } for (i = 2; i <= (size / 2); i++) {
            working_space[size - i + 1] = working_space[i - 1];
            working_space[size - i + 1 + size] =
                -working_space[i - 1 + size];
         }
         working_space[size / 2] = 0;
         working_space[size / 2 + size] = 0;
         Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 1);
         for (i = 0; i < size / 2; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_SIN:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         size = 2 * size;
         working_space[size / 2] =
             working_space[size / 2 - 1] * TMath::Sqrt(2.0);
         for (i = size / 2 - 1; i > 0; i--) {
            a = pi * (double) i / (double) size;
            working_space[i + size] =
                -(double) working_space[i - 1] * TMath::Cos(a);
            working_space[i] =
                (double) working_space[i - 1] * TMath::Sin(a);
         } for (i = 2; i <= (size / 2); i++) {
            working_space[size - i + 1] = working_space[i - 1];
            working_space[size - i + 1 + size] =
                -working_space[i - 1 + size];
         }
         working_space[0] = 0;
         working_space[size] = 0;
         working_space[size / 2 + size] = 0;
         Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
         for (i = 0; i < size / 2; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_FOURIER:
         for (i = 0; i < 2 * size; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_HARTLEY:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, size, 1, TRANSFORM1_INVERSE, 0);
         for (i = 0; i < size; i++) {
            dest[i] = working_space[i];
         }
         break;
      case TRANSFORM1_FOURIER_WALSH:
      case TRANSFORM1_FOURIER_HAAR:
      case TRANSFORM1_WALSH_HAAR:
      case TRANSFORM1_COS_WALSH:
      case TRANSFORM1_COS_HAAR:
      case TRANSFORM1_SIN_WALSH:
      case TRANSFORM1_SIN_HAAR:
         for (i = 0; i < size; i++) {
            working_space[i] = source[i];
         }
         if (type == TRANSFORM1_FOURIER_WALSH
              || type == TRANSFORM1_FOURIER_HAAR) {
            for (i = 0; i < size; i++) {
               working_space[i + 2 * size] = source[size + i];
            }
         }
         if (type > TRANSFORM1_WALSH_HAAR)
            k = (int) TMath::Power(2, degree - 1);
         
         else
            k = (int) TMath::Power(2, degree);
         j = size / k;
         for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
            working_space[size + l + i / j] = working_space[i];
            working_space[size + l + i / j + 2 * size] =
                working_space[i + 2 * size];
         }
         for (i = 0; i < size; i++) {
            working_space[i] = working_space[size + i];
            working_space[i + 2 * size] =
                working_space[size + i + 2 * size];
         }
         if (type == TRANSFORM1_FOURIER_WALSH
              || type == TRANSFORM1_FOURIER_HAAR
              || type == TRANSFORM1_WALSH_HAAR) {
            GeneralInv(working_space, size, degree, type);
            for (i = 0; i < j; i++)
               BitReverseHaar(working_space, size, k, i * k);
         }
         
         else if (type == TRANSFORM1_COS_WALSH
                  || type == TRANSFORM1_COS_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * size / m;
            for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
               if (i % j == 0) {
                  working_space[2 * size + k + i % j] =
                      working_space[i] * TMath::Sqrt(2.0);
                  working_space[4 * size + 2 * size + k + i % j] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_space[4 * size + 2 * size + k + i % j] =
                      -(double) working_space[i] * b;
                  working_space[2 * size + k + i % j] =
                      (double) working_space[i] * a;
            } } for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               if (i % j == 0) {
                  working_space[2 * size + k + j] = 0;
                  working_space[4 * size + 2 * size + k + j] = 0;
               }
               
               else {
                  working_space[2 * size + k + 2 * j - i % j] =
                      working_space[2 * size + k + i % j];
                  working_space[4 * size + 2 * size + k + 2 * j - i % j] =
                      -working_space[4 * size + 2 * size + k + i % j];
               }
            }
            for (i = 0; i < 2 * size; i++) {
               working_space[i] = working_space[2 * size + i];
               working_space[4 * size + i] =
                   working_space[4 * size + 2 * size + i];
            }
            GeneralInv(working_space, 2 * size, degree, type);
            m = (int) TMath::Power(2, degree);
            l = 2 * size / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * size, m, i * m);
         }
         
         else if (type == TRANSFORM1_SIN_WALSH
                  || type == TRANSFORM1_SIN_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * size / m;
            for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
               if (i % j == 0) {
                  working_space[2 * size + k + j + i % j] =
                      working_space[j + k / 2 - i % j -
                                    1] * TMath::Sqrt(2.0);
                  working_space[4 * size + 2 * size + k + j + i % j] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_space[4 * size + 2 * size + k + j + i % j] =
                      -(double) working_space[j + k / 2 - i % j - 1] * b;
                  working_space[2 * size + k + j + i % j] =
                      (double) working_space[j + k / 2 - i % j - 1] * a;
            } } for (i = 0; i < size; i++) {
               k = i / j;
               k = 2 * k * j;
               if (i % j == 0) {
                  working_space[2 * size + k] = 0;
                  working_space[4 * size + 2 * size + k] = 0;
               }
               
               else {
                  working_space[2 * size + k + i % j] =
                      working_space[2 * size + k + 2 * j - i % j];
                  working_space[4 * size + 2 * size + k + i % j] =
                      -working_space[4 * size + 2 * size + k + 2 * j -
                                     i % j];
               }
            }
            for (i = 0; i < 2 * size; i++) {
               working_space[i] = working_space[2 * size + i];
               working_space[4 * size + i] =
                   working_space[4 * size + 2 * size + i];
            }
            GeneralInv(working_space, 2 * size, degree, type);
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * size, m, i * m);
         }
         for (i = 0; i < size; i++) {
            if (type >= TRANSFORM1_COS_WALSH) {
               k = i / j;
               k = 2 * k * j;
               val = working_space[k + i % j];
            }
            
            else
               val = working_space[i];
            dest[i] = val;
         }
         break;
      }
   }
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
const char *TSpectrum::Filter1Zonal(const float *source, float *dest,
                                    int size, int type, int degree,
                                    int xmin, int xmax,
                                    float filter_coeff) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL FILTER ZONAL FUNCTION   			   */ 
/*	This function transforms the source spectrum. The calling program  */ 
/*      should fill in input parameters. Then it sets transformed          */ 
/*      coefficients in the given region (xmin, xmax) to the given         */ 
/*      filter_coeff and transforms it back                                */ 
/*	Filtered data are written into dest spectrum.                      */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum, its length should */ 
/*             be size except for inverse FOURIER, FOUR-WALSh, FOUR-HAAR   */ 
/*             transform. These need 2*size length to supply real and      */ 
/*             imaginary coefficients.                                     */ 
/*	dest-pointer to the vector of dest data, its length should be      */ 
/*           size except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These  */ 
/*           need 2*size length to store real and imaginary coefficients   */ 
/*	size-basic length of source and dest spectra                       */ 
/*	type-type of transform                                             */ 
/*      degree-applied only for mixed transforms                           */ 
/*	xmin-low limit of filtered region                                  */ 
/*	xmax-high limit of filtered region                                 */ 
/*	filter_coeff-value which is set in filtered region                 */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
       
//////////FILTER1_ZONAL FUNCTION - CALCULATES DIFFERENT 1-D ORTHOGONAL TRANSFORMS, SETS GIVEN REGION TO FILTER COEFFICIENT AND TRANSFORMS IT BACK//////
   int i, j, n, k = 1, m, l;
   float val;
   float *working_space = 0;
   double a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (size <= 0)
      return "Wrong Parameters";
   j = 0;
   n = 1;
   for (; n < size;) {
      j += 1;
      n = n * 2;
   }
   if (n != size)
      return ("LENGTH MUST BE POWER OF 2");
   if (type < TRANSFORM1_HAAR || type > TRANSFORM1_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (type >= TRANSFORM1_FOURIER_WALSH && type <= TRANSFORM1_SIN_HAAR) {
      if (degree > j || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM1_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      j = size / k;
   }
   if (xmin < 0 || xmin > xmax)
      return ("WRONG LOW REGION LIMIT");
   if (xmax < xmin || xmax >= size)
      return ("WRONG HIGH REGION LIMIT");
   switch (type) {
   case TRANSFORM1_HAAR:
   case TRANSFORM1_WALSH:
      working_space = new float[2 * size];
      break;
   case TRANSFORM1_COS:
   case TRANSFORM1_SIN:
   case TRANSFORM1_FOURIER:
   case TRANSFORM1_HARTLEY:
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
      working_space = new float[4 * size];
      break;
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      working_space = new float[8 * size];
      break;
   }
   switch (type) {
   case TRANSFORM1_HAAR:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Haar(working_space, size, TRANSFORM1_FORWARD);
      break;
   case TRANSFORM1_WALSH:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Walsh(working_space, size);
      BitReverse(working_space, size);
      break;
   case TRANSFORM1_COS:
      size = 2 * size;
      for (i = 1; i <= (size / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[size - i] = val;
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         a = TMath::Cos(a);
         b = working_space[i];
         a = b / a;
         working_space[i] = a;
         working_space[i + size] = 0;
      } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
      size = size / 2;
      break;
   case TRANSFORM1_SIN:
      size = 2 * size;
      for (i = 1; i <= (size / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[size - i] = -val;
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         a = TMath::Sin(a);
         b = working_space[i];
         if (a != 0)
            a = b / a;
         working_space[i - 1] = a;
         working_space[i + size] = 0;
      }
      working_space[size / 2 - 1] =
          working_space[size / 2] / TMath::Sqrt(2.0);
      size = size / 2;
      break;
   case TRANSFORM1_FOURIER:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      break;
   case TRANSFORM1_HARTLEY:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, size, 1, TRANSFORM1_FORWARD, 0);
      break;
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      for (i = 0; i < size; i++) {
         val = source[i];
         if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }
         
         else if (type == TRANSFORM1_SIN_WALSH
                  || type == TRANSFORM1_SIN_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = -val;
         }
         
         else
            working_space[i] = val;
      }
      if (type == TRANSFORM1_FOURIER_WALSH
           || type == TRANSFORM1_FOURIER_HAAR
           || type == TRANSFORM1_WALSH_HAAR) {
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, size, k, i * k);
         GeneralExe(working_space, 0, size, degree, type);
      }
      
      else if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
         GeneralExe(working_space, 0, 2 * size, degree, type);
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            a = TMath::Cos(a);
            b = working_space[k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);
            
            else
               a = b / a;
            working_space[i] = a;
            working_space[i + 2 * size] = 0;
         }
      }
      
      else if (type == TRANSFORM1_SIN_WALSH || type == TRANSFORM1_SIN_HAAR) {
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
         GeneralExe(working_space, 0, 2 * size, degree, type);
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            a = TMath::Cos(a);
            b = working_space[j + k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);
            
            else
               a = b / a;
            working_space[j + k / 2 - i % j - 1] = a;
            working_space[i + 2 * size] = 0;
         }
      }
      if (type > TRANSFORM1_WALSH_HAAR)
         k = (int) TMath::Power(2, degree - 1);
      
      else
         k = (int) TMath::Power(2, degree);
      j = size / k;
      for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
         working_space[size + i] = working_space[l + i / j];
         working_space[size + i + 2 * size] =
             working_space[l + i / j + 2 * size];
      }
      for (i = 0; i < size; i++) {
         working_space[i] = working_space[size + i];
         working_space[i + 2 * size] = working_space[size + i + 2 * size];
      }
      break;
   }
   for (i = 0, old_area = 0; i < size; i++) {
      old_area += working_space[i];
   }
   for (i = 0, new_area = 0; i < size; i++) {
      if (i >= xmin && i <= xmax)
         working_space[i] = filter_coeff;
      new_area += working_space[i];
   }
   if (new_area != 0) {
      a = old_area / new_area;
      for (i = 0; i < size; i++) {
         working_space[i] *= a;
      }
   }
   if (type == TRANSFORM1_FOURIER) {
      for (i = 0, old_area = 0; i < size; i++) {
         old_area += working_space[i + size];
      }
      for (i = 0, new_area = 0; i < size; i++) {
         if (i >= xmin && i <= xmax)
            working_space[i + size] = filter_coeff;
         new_area += working_space[i + size];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < size; i++) {
            working_space[i + size] *= a;
         }
      }
   }
   
   else if (type == TRANSFORM1_FOURIER_WALSH
            || type == TRANSFORM1_FOURIER_HAAR) {
      for (i = 0, old_area = 0; i < size; i++) {
         old_area += working_space[i + 2 * size];
      }
      for (i = 0, new_area = 0; i < size; i++) {
         if (i >= xmin && i <= xmax)
            working_space[i + 2 * size] = filter_coeff;
         new_area += working_space[i + 2 * size];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < size; i++) {
            working_space[i + 2 * size] *= a;
         }
      }
   }
   switch (type) {
   case TRANSFORM1_HAAR:
      Haar(working_space, size, TRANSFORM1_INVERSE);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_WALSH:
      BitReverse(working_space, size);
      Walsh(working_space, size);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_COS:
      size = 2 * size;
      working_space[0] = working_space[0] * TMath::Sqrt(2.0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + size] = (double) working_space[i] * b;
         working_space[i] = (double) working_space[i] * a;
      } for (i = 2; i <= (size / 2); i++) {
         working_space[size - i + 1] = working_space[i - 1];
         working_space[size - i + 1 + size] =
             -working_space[i - 1 + size];
      }
      working_space[size / 2] = 0;
      working_space[size / 2 + size] = 0;
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 1);
      for (i = 0; i < size / 2; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_SIN:
      size = 2 * size;
      working_space[size / 2] =
          working_space[size / 2 - 1] * TMath::Sqrt(2.0);
      for (i = size / 2 - 1; i > 0; i--) {
         a = pi * (double) i / (double) size;
         working_space[i + size] =
             -(double) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (double) working_space[i - 1] * TMath::Sin(a);
      } for (i = 2; i <= (size / 2); i++) {
         working_space[size - i + 1] = working_space[i - 1];
         working_space[size - i + 1 + size] =
             -working_space[i - 1 + size];
      }
      working_space[0] = 0;
      working_space[size] = 0;
      working_space[size / 2 + size] = 0;
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size / 2; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_FOURIER:
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_HARTLEY:
      Fourier(working_space, size, 1, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      if (type > TRANSFORM1_WALSH_HAAR)
         k = (int) TMath::Power(2, degree - 1);
      
      else
         k = (int) TMath::Power(2, degree);
      j = size / k;
      for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
         working_space[size + l + i / j] = working_space[i];
         working_space[size + l + i / j + 2 * size] =
             working_space[i + 2 * size];
      }
      for (i = 0; i < size; i++) {
         working_space[i] = working_space[size + i];
         working_space[i + 2 * size] = working_space[size + i + 2 * size];
      }
      if (type == TRANSFORM1_FOURIER_WALSH
           || type == TRANSFORM1_FOURIER_HAAR
           || type == TRANSFORM1_WALSH_HAAR) {
         GeneralInv(working_space, size, degree, type);
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, size, k, i * k);
      }
      
      else if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
         j = (int) TMath::Power(2, degree) / 2;
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * size + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * size + 2 * size + k + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * size + 2 * size + k + i % j] =
                   -(double) working_space[i] * b;
               working_space[2 * size + k + i % j] =
                   (double) working_space[i] * a;
         } } for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * size + k + j] = 0;
               working_space[4 * size + 2 * size + k + j] = 0;
            }
            
            else {
               working_space[2 * size + k + 2 * j - i % j] =
                   working_space[2 * size + k + i % j];
               working_space[4 * size + 2 * size + k + 2 * j - i % j] =
                   -working_space[4 * size + 2 * size + k + i % j];
            }
         }
         for (i = 0; i < 2 * size; i++) {
            working_space[i] = working_space[2 * size + i];
            working_space[4 * size + i] =
                working_space[4 * size + 2 * size + i];
         }
         GeneralInv(working_space, 2 * size, degree, type);
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
      }
      
      else if (type == TRANSFORM1_SIN_WALSH || type == TRANSFORM1_SIN_HAAR) {
         j = (int) TMath::Power(2, degree) / 2;
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * size + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * size + 2 * size + k + j + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * size + 2 * size + k + j + i % j] =
                   -(double) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * size + k + j + i % j] =
                   (double) working_space[j + k / 2 - i % j - 1] * a;
         } } for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * size + k] = 0;
               working_space[4 * size + 2 * size + k] = 0;
            }
            
            else {
               working_space[2 * size + k + i % j] =
                   working_space[2 * size + k + 2 * j - i % j];
               working_space[4 * size + 2 * size + k + i % j] =
                   -working_space[4 * size + 2 * size + k + 2 * j - i % j];
            }
         }
         for (i = 0; i < 2 * size; i++) {
            working_space[i] = working_space[2 * size + i];
            working_space[4 * size + i] =
                working_space[4 * size + 2 * size + i];
         }
         GeneralInv(working_space, 2 * size, degree, type);
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
      }
      for (i = 0; i < size; i++) {
         if (type >= TRANSFORM1_COS_WALSH) {
            k = i / j;
            k = 2 * k * j;
            val = working_space[k + i % j];
         }
         
         else
            val = working_space[i];
         dest[i] = val;
      }
      break;
   }
   delete[]working_space;
   return 0;
}


//___________________________________________________________________________
    
//////////ENHANCE1 FUNCTION - CALCULATES DIFFERENT 1-D ORTHOGONAL TRANSFORMS, MULTIPLIES GIVEN REGION BY ENHANCE COEFFICIENT AND TRANSFORMS IT BACK//////
const char *TSpectrum::Enhance1(const float *source, float *dest, int size,
                                int type, int degree, int xmin, int xmax,
                                float enhance_coeff) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	ONE-DIMENSIONAL ENHANCE ZONAL FUNCTION		        	   */ 
/*	This function transforms the source spectrum. The calling program  */ 
/*      should fill in input parameters. Then it multiplies transformed    */ 
/*      coefficients in the given region (xmin, xmax) by the given         */ 
/*      enhance_coeff and transforms it back                               */ 
/*	Processed data are written into dest spectrum.                     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the vector of source spectrum, its length should */ 
/*             be size except for inverse FOURIER, FOUR-WALSh, FOUR-HAAR   */ 
/*             transform. These need 2*size length to supply real and      */ 
/*             imaginary coefficients.                                     */ 
/*	dest-pointer to the vector of dest data, its length should be      */ 
/*           size except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These  */ 
/*           need 2*size length to store real and imaginary coefficients   */ 
/*	size-basic length of source and dest spectra                       */ 
/*	type-type of transform                                             */ 
/*      degree-applied only for mixed transforms                           */ 
/*	xmin-low limit of filtered region                                  */ 
/*	xmax-high limit of filtered region                                 */ 
/*	enhance_coeff-value by which the filtered region is multiplied     */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, n, k = 1, m, l;
   float val;
   float *working_space = 0;
   double a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (size <= 0)
      return "Wrong Parameters";
   j = 0;
   n = 1;
   for (; n < size;) {
      j += 1;
      n = n * 2;
   }
   if (n != size)
      return ("LENGTH MUST BE POWER OF 2");
   if (type < TRANSFORM1_HAAR || type > TRANSFORM1_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (type >= TRANSFORM1_FOURIER_WALSH && type <= TRANSFORM1_SIN_HAAR) {
      if (degree > j || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM1_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      j = size / k;
   }
   if (xmin < 0 || xmin > xmax)
      return ("WRONG LOW REGION LIMIT");
   if (xmax < xmin || xmax >= size)
      return ("WRONG HIGH REGION LIMIT");
   switch (type) {
   case TRANSFORM1_HAAR:
   case TRANSFORM1_WALSH:
      working_space = new float[2 * size];
      break;
   case TRANSFORM1_COS:
   case TRANSFORM1_SIN:
   case TRANSFORM1_FOURIER:
   case TRANSFORM1_HARTLEY:
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
      working_space = new float[4 * size];
      break;
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      working_space = new float[8 * size];
      break;
   }
   switch (type) {
   case TRANSFORM1_HAAR:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Haar(working_space, size, TRANSFORM1_FORWARD);
      break;
   case TRANSFORM1_WALSH:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Walsh(working_space, size);
      BitReverse(working_space, size);
      break;
   case TRANSFORM1_COS:
      size = 2 * size;
      for (i = 1; i <= (size / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[size - i] = val;
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         a = TMath::Cos(a);
         b = working_space[i];
         a = b / a;
         working_space[i] = a;
         working_space[i + size] = 0;
      } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
      size = size / 2;
      break;
   case TRANSFORM1_SIN:
      size = 2 * size;
      for (i = 1; i <= (size / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[size - i] = -val;
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         a = TMath::Sin(a);
         b = working_space[i];
         if (a != 0)
            a = b / a;
         working_space[i - 1] = a;
         working_space[i + size] = 0;
      }
      working_space[size / 2 - 1] =
          working_space[size / 2] / TMath::Sqrt(2.0);
      size = size / 2;
      break;
   case TRANSFORM1_FOURIER:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, size, 0, TRANSFORM1_FORWARD, 0);
      break;
   case TRANSFORM1_HARTLEY:
      for (i = 0; i < size; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, size, 1, TRANSFORM1_FORWARD, 0);
      break;
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      for (i = 0; i < size; i++) {
         val = source[i];
         if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }
         
         else if (type == TRANSFORM1_SIN_WALSH
                  || type == TRANSFORM1_SIN_HAAR) {
            j = (int) TMath::Power(2, degree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = -val;
         }
         
         else
            working_space[i] = val;
      }
      if (type == TRANSFORM1_FOURIER_WALSH
           || type == TRANSFORM1_FOURIER_HAAR
           || type == TRANSFORM1_WALSH_HAAR) {
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, size, k, i * k);
         GeneralExe(working_space, 0, size, degree, type);
      }
      
      else if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
         GeneralExe(working_space, 0, 2 * size, degree, type);
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            a = TMath::Cos(a);
            b = working_space[k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);
            
            else
               a = b / a;
            working_space[i] = a;
            working_space[i + 2 * size] = 0;
         }
      }
      
      else if (type == TRANSFORM1_SIN_WALSH || type == TRANSFORM1_SIN_HAAR) {
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
         GeneralExe(working_space, 0, 2 * size, degree, type);
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            a = TMath::Cos(a);
            b = working_space[j + k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);
            
            else
               a = b / a;
            working_space[j + k / 2 - i % j - 1] = a;
            working_space[i + 2 * size] = 0;
         }
      }
      if (type > TRANSFORM1_WALSH_HAAR)
         k = (int) TMath::Power(2, degree - 1);
      
      else
         k = (int) TMath::Power(2, degree);
      j = size / k;
      for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
         working_space[size + i] = working_space[l + i / j];
         working_space[size + i + 2 * size] =
             working_space[l + i / j + 2 * size];
      }
      for (i = 0; i < size; i++) {
         working_space[i] = working_space[size + i];
         working_space[i + 2 * size] = working_space[size + i + 2 * size];
      }
      break;
   }
   for (i = 0, old_area = 0; i < size; i++) {
      old_area += working_space[i];
   }
   for (i = 0, new_area = 0; i < size; i++) {
      if (i >= xmin && i <= xmax)
         working_space[i] *= enhance_coeff;
      new_area += working_space[i];
   }
   if (new_area != 0) {
      a = old_area / new_area;
      for (i = 0; i < size; i++) {
         working_space[i] *= a;
      }
   }
   if (type == TRANSFORM1_FOURIER) {
      for (i = 0, old_area = 0; i < size; i++) {
         old_area += working_space[i + size];
      }
      for (i = 0, new_area = 0; i < size; i++) {
         if (i >= xmin && i <= xmax)
            working_space[i + size] *= enhance_coeff;
         new_area += working_space[i + size];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < size; i++) {
            working_space[i + size] *= a;
         }
      }
   }
   
   else if (type == TRANSFORM1_FOURIER_WALSH
            || type == TRANSFORM1_FOURIER_HAAR) {
      for (i = 0, old_area = 0; i < size; i++) {
         old_area += working_space[i + 2 * size];
      }
      for (i = 0, new_area = 0; i < size; i++) {
         if (i >= xmin && i <= xmax)
            working_space[i + 2 * size] *= enhance_coeff;
         new_area += working_space[i + 2 * size];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < size; i++) {
            working_space[i + 2 * size] *= a;
         }
      }
   }
   switch (type) {
   case TRANSFORM1_HAAR:
      Haar(working_space, size, TRANSFORM1_INVERSE);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_WALSH:
      BitReverse(working_space, size);
      Walsh(working_space, size);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_COS:
      size = 2 * size;
      working_space[0] = working_space[0] * TMath::Sqrt(2.0);
      for (i = 0; i < size / 2; i++) {
         a = pi * (double) i / (double) size;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + size] = (double) working_space[i] * b;
         working_space[i] = (double) working_space[i] * a;
      } for (i = 2; i <= (size / 2); i++) {
         working_space[size - i + 1] = working_space[i - 1];
         working_space[size - i + 1 + size] =
             -working_space[i - 1 + size];
      }
      working_space[size / 2] = 0;
      working_space[size / 2 + size] = 0;
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 1);
      for (i = 0; i < size / 2; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_SIN:
      size = 2 * size;
      working_space[size / 2] =
          working_space[size / 2 - 1] * TMath::Sqrt(2.0);
      for (i = size / 2 - 1; i > 0; i--) {
         a = pi * (double) i / (double) size;
         working_space[i + size] =
             -(double) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (double) working_space[i - 1] * TMath::Sin(a);
      } for (i = 2; i <= (size / 2); i++) {
         working_space[size - i + 1] = working_space[i - 1];
         working_space[size - i + 1 + size] =
             -working_space[i - 1 + size];
      }
      working_space[0] = 0;
      working_space[size] = 0;
      working_space[size / 2 + size] = 0;
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size / 2; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_FOURIER:
      Fourier(working_space, size, 0, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_HARTLEY:
      Fourier(working_space, size, 1, TRANSFORM1_INVERSE, 0);
      for (i = 0; i < size; i++) {
         dest[i] = working_space[i];
      }
      break;
   case TRANSFORM1_FOURIER_WALSH:
   case TRANSFORM1_FOURIER_HAAR:
   case TRANSFORM1_WALSH_HAAR:
   case TRANSFORM1_COS_WALSH:
   case TRANSFORM1_COS_HAAR:
   case TRANSFORM1_SIN_WALSH:
   case TRANSFORM1_SIN_HAAR:
      if (type > TRANSFORM1_WALSH_HAAR)
         k = (int) TMath::Power(2, degree - 1);
      
      else
         k = (int) TMath::Power(2, degree);
      j = size / k;
      for (i = 0, l = 0; i < size; i++, l = (l + k) % size) {
         working_space[size + l + i / j] = working_space[i];
         working_space[size + l + i / j + 2 * size] =
             working_space[i + 2 * size];
      }
      for (i = 0; i < size; i++) {
         working_space[i] = working_space[size + i];
         working_space[i + 2 * size] = working_space[size + i + 2 * size];
      }
      if (type == TRANSFORM1_FOURIER_WALSH
           || type == TRANSFORM1_FOURIER_HAAR
           || type == TRANSFORM1_WALSH_HAAR) {
         GeneralInv(working_space, size, degree, type);
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, size, k, i * k);
      }
      
      else if (type == TRANSFORM1_COS_WALSH || type == TRANSFORM1_COS_HAAR) {
         j = (int) TMath::Power(2, degree) / 2;
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * size + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * size + 2 * size + k + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * size + 2 * size + k + i % j] =
                   -(double) working_space[i] * b;
               working_space[2 * size + k + i % j] =
                   (double) working_space[i] * a;
         } } for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * size + k + j] = 0;
               working_space[4 * size + 2 * size + k + j] = 0;
            }
            
            else {
               working_space[2 * size + k + 2 * j - i % j] =
                   working_space[2 * size + k + i % j];
               working_space[4 * size + 2 * size + k + 2 * j - i % j] =
                   -working_space[4 * size + 2 * size + k + i % j];
            }
         }
         for (i = 0; i < 2 * size; i++) {
            working_space[i] = working_space[2 * size + i];
            working_space[4 * size + i] =
                working_space[4 * size + 2 * size + i];
         }
         GeneralInv(working_space, 2 * size, degree, type);
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
      }
      
      else if (type == TRANSFORM1_SIN_WALSH || type == TRANSFORM1_SIN_HAAR) {
         j = (int) TMath::Power(2, degree) / 2;
         m = (int) TMath::Power(2, degree);
         l = 2 * size / m;
         for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * size + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * size + 2 * size + k + j + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * size + 2 * size + k + j + i % j] =
                   -(double) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * size + k + j + i % j] =
                   (double) working_space[j + k / 2 - i % j - 1] * a;
         } } for (i = 0; i < size; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * size + k] = 0;
               working_space[4 * size + 2 * size + k] = 0;
            }
            
            else {
               working_space[2 * size + k + i % j] =
                   working_space[2 * size + k + 2 * j - i % j];
               working_space[4 * size + 2 * size + k + i % j] =
                   -working_space[4 * size + 2 * size + k + 2 * j - i % j];
            }
         }
         for (i = 0; i < 2 * size; i++) {
            working_space[i] = working_space[2 * size + i];
            working_space[4 * size + i] =
                working_space[4 * size + 2 * size + i];
         }
         GeneralInv(working_space, 2 * size, degree, type);
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * size, m, i * m);
      }
      for (i = 0; i < size; i++) {
         if (type >= TRANSFORM1_COS_WALSH) {
            k = i / j;
            k = 2 * k * j;
            val = working_space[k + i % j];
         }
         
         else
            val = working_space[i];
         dest[i] = val;
      }
      break;
   }
   delete[]working_space;
   return 0;
}


