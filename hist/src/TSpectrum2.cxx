// @(#)root/hist:$Name:  $:$Id: TSpectrum2.cxx,v 1.6 2003/05/08 15:00:38 brun Exp $
// Author: Miroslav Morhac   11/04/2003

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
    
#include "TSpectrum2.h"
#include "TPolyMarker.h"
#include "TMath.h"
#define PEAK_WINDOW 1024
    ClassImp(TSpectrum2)  
//______________________________________________________________________________
TSpectrum2::TSpectrum2() :TNamed("Spectrum", "Miroslav Morhac peak finder") 
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
TSpectrum2::TSpectrum2(Int_t maxpositions, Float_t resolution) :TNamed("Spectrum", "Miroslav Morhac peak finder") 
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
    TSpectrum2::~TSpectrum2() 
{
   delete[]fPosition;
   delete[]fPositionX;
   delete[]fPositionY;
   delete fHistogram;
}


//______________________________________________________________________________
const char *TSpectrum2::Background(TH1 * h, int number_of_iterations,
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
    Int_t TSpectrum2::Search(TH1 * hin, Double_t sigma,
                             Option_t * option, Double_t threshold) 
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
//            Note that sigma is in number of bins                         //
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
       if (hin == 0)
      return 0;
   Int_t dimension = hin->GetDimension();
   if (dimension > 2) {
      Error("Search", "Only implemented for 1-d and 2-d histograms");
      return 0;
   }
   if (dimension == 2) {
      Int_t sizex = hin->GetXaxis()->GetNbins();
      Int_t sizey = hin->GetYaxis()->GetNbins();
      Int_t i, j, binx,biny, npeaks;
      Float_t ** source = new float *[sizex];
      Float_t ** dest   = new float *[sizex];
      for (i = 0; i < sizex; i++) {
         source[i] = new float[sizey];
         dest[i]   = new float[sizey];
         for (j = 0; j < sizey; j++) {
            source[i][j] = (Float_t) hin->GetBinContent(i + 1, j + 1);
         }
      }
      npeaks = Search2HighRes(source, dest, sizex, sizey, sigma, 100*threshold, kTRUE, 3, kTRUE, 10);

      //The logic in the loop should be improved to use the fact
      //that fPositionX,Y give a precise position inside a bin.
      //The current algorithm takes the center of the bin only.
      for (i = 0; i < npeaks; i++) {
         binx = 1 + Int_t(fPositionX[i] + 0.5);
         biny = 1 + Int_t(fPositionY[i] + 0.5);
         fPositionX[i] = hin->GetXaxis()->GetBinCenter(binx);
         fPositionY[i] = hin->GetYaxis()->GetBinCenter(biny);
      }
      for (i = 0; i < sizex; i++) {
         delete [] source[i];
         delete [] dest[i];
      }
      delete [] source;
      delete [] dest;
      
      if (strstr(option, "goff"))
         return npeaks;
      TPolyMarker * pm = new TPolyMarker(npeaks, fPositionX, fPositionY);
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
void TSpectrum2::SetResolution(Float_t resolution) 
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
//______________________________________________________________________________
const char *TSpectrum2::Background2(float **spectrum, int sizex, int sizey,
                                    int number_of_iterations) 
{
   
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   This function calculates background spectrum from source spectrum.    //
//   The result is placed to the array pointed by spectrum pointer.        //
//                                                                         //
//   Function parameters:                                                  //
//   spectrum:  pointer to the array of source spectrum                    //
//   sizex:     x length of spectrum and working space arrays              //
//   sizey:     y length of spectrum and working space arrays              //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizex < 2 * number_of_iterations + 1)
      return ("Too Large Clipping Window");
   
       //   working_space-pointer to the working array
   float **working_space = new float *[sizex];
   int i, x, y;
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   float a, b, p1, p2, p3, p4, s1, s2, s3, s4;
   for (i = 1; i <= number_of_iterations; i++) {
      for (y = i; y < sizey - i; y++) {
         for (x = i; x < sizex - i; x++) {
            a = spectrum[x][y];
            p1 = spectrum[x - i][y - i];
            p2 = spectrum[x - i][y + i];
            p3 = spectrum[x + i][y - i];
            p4 = spectrum[x + i][y + i];
            s1 = spectrum[x][y - i];
            s2 = spectrum[x - i][y];
            s3 = spectrum[x + i][y];
            s4 = spectrum[x][y + i];
            b = (p1 + p2) / 2.0;
            if (b > s2)
               s2 = b;
            b = (p1 + p3) / 2.0;
            if (b > s1)
               s1 = b;
            b = (p2 + p4) / 2.0;
            if (b > s4)
               s4 = b;
            b = (p3 + p4) / 2.0;
            if (b > s3)
               s3 = b;
            s1 = s1 - (p1 + p3) / 2.0;
            s2 = s2 - (p1 + p2) / 2.0;
            s3 = s3 - (p3 + p4) / 2.0;
            s4 = s4 - (p2 + p4) / 2.0;
            b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 + p3 +
                                                      p4) / 4.0;
            if (b < a)
               a = b;
            working_space[x][y] = a;
         }
      }
      for (y = i; y < sizey - i; y++) {
         for (x = i; x < sizex - i; x++) {
            spectrum[x][y] = working_space[x][y];
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum2::Background2RectangularRidges(float **spectrum,
                                                     int sizex, int sizey,
                                                     int
                                                     number_of_iterations_x,
                                                     int
                                                     number_of_iterations_y,
                                                     int direction,
                                                     int filter_order,
                                                     int filter_type) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - RECTANGULAR RIDGES*/ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed to the array pointed by spectrum pointer.     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the array of source spectrum		   */ 
/*	sizex-x length of spectrum                      		   */ 
/*	sizey-y length of spectrum 		                           */ 
/*	number_of_iterations_x-maximal x width of clipping window          */ 
/*	number_of_iterations_y-maximal y width of clipping window          */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK2_INCREASING_WINDOW                 */ 
/*                                 BACK2_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK2_ORDER2                          */ 
/*                                   BACK2_ORDER4                          */ 
/*                                   BACK2_ORDER6                          */ 
/*                                   BACK2_ORDER8	                   */ 
/*	filter_type-determines the algorithm of the filtering              */ 
/*                  -possible values=BACK2_SUCCESSIVE_FILTERING            */ 
/*                                   BACK2_ONE_STEP_FILTERING              */ 
/*									   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, x, y, sampling, r1, r2;
   float a, b, c, d, e, p1, p2, p3, p4, s1, s2, s3, s4, ar1, ar2;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations_x < 1 || number_of_iterations_y < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizex < 2 * number_of_iterations_x + 1
        || sizey < 2 * number_of_iterations_y + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   sampling =
       (int) TMath::Max(number_of_iterations_x, number_of_iterations_y);
   if (direction == BACK2_INCREASING_WINDOW) {
      if (filter_type == BACK2_SUCCESSIVE_FILTERING) {
         if (filter_order == BACK2_ORDER2) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER4) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               ar1 = r1 / 2;
               ar2 = r2 / 2;
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER6) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x - r1][y - (int) ar2] +
                           15 * spectrum[x - r1][y + (int) ar2] -
                           6 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           spectrum[x - r1][y + (int) (3 * ar2)]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y - r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           15 * spectrum[x - (int) ar1][y - r2] +
                           15 * spectrum[x + (int) ar1][y - r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           spectrum[x + (int) (3 * ar1)][y - r2]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y + r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           15 * spectrum[x - (int) ar1][y + r2] +
                           15 * spectrum[x + (int) ar1][y + r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           spectrum[x + (int) (3 * ar1)][y + r2]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x + r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x + r1][y - (int) ar2] +
                           15 * spectrum[x + r1][y + (int) ar2] -
                           6 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           spectrum[x + r1][y + (int) (3 * ar2)]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER8) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x - r1][y - (int) ar2] +
                           15 * spectrum[x - r1][y + (int) ar2] -
                           6 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           spectrum[x - r1][y + (int) (3 * ar2)]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - r1][y - (int) (4 * ar2)] +
                           8 * spectrum[x - r1][y - (int) (3 * ar2)] -
                           28 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           56 * spectrum[x - r1][y - (int) ar2] +
                           56 * spectrum[x - r1][y + (int) ar2] -
                           28 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           8 * spectrum[x - r1][y + (int) (3 * ar2)] -
                           spectrum[x - r1][y + (int) (4 * ar2)]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y - r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           15 * spectrum[x - (int) ar1][y - r2] +
                           15 * spectrum[x + (int) ar1][y - r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           spectrum[x + (int) (3 * ar1)][y - r2]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - (int) (4 * ar1)][y - r2] +
                           8 * spectrum[x - (int) (3 * ar1)][y - r2] -
                           28 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           56 * spectrum[x - (int) ar1][y - r2] +
                           56 * spectrum[x + (int) ar1][y - r2] -
                           28 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           8 * spectrum[x + (int) (3 * ar1)][y - r2] -
                           spectrum[x + (int) (4 * ar1)][y - r2]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y + r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           15 * spectrum[x - (int) ar1][y + r2] +
                           15 * spectrum[x + (int) ar1][y + r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           spectrum[x + (int) (3 * ar1)][y + r2]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - (int) (4 * ar1)][y + r2] +
                           8 * spectrum[x - (int) (3 * ar1)][y + r2] -
                           28 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           56 * spectrum[x - (int) ar1][y + r2] +
                           56 * spectrum[x + (int) ar1][y + r2] -
                           28 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           8 * spectrum[x + (int) (3 * ar1)][y + r2] -
                           spectrum[x + (int) (4 * ar1)][y + r2]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x + r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x + r1][y - (int) ar2] +
                           15 * spectrum[x + r1][y + (int) ar2] -
                           6 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           spectrum[x + r1][y + (int) (3 * ar2)]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x + r1][y - (int) (4 * ar2)] +
                           8 * spectrum[x + r1][y - (int) (3 * ar2)] -
                           28 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           56 * spectrum[x + r1][y - (int) ar2] +
                           56 * spectrum[x + r1][y + (int) ar2] -
                           28 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           8 * spectrum[x + r1][y + (int) (3 * ar2)] -
                           spectrum[x + r1][y + (int) (4 * ar2)]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
      }
      
      else if (filter_type == BACK2_ONE_STEP_FILTERING) {
         if (filter_order == BACK2_ORDER2) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = i; y < sizey - i; y++) {
                  for (x = i; x < sizex - i; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER4) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               ar1 = r1 / 2, ar2 = r2 / 2;
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER6) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER8) {
            for (i = 1; i <= sampling; i++) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = -(spectrum[x - (int) (4 * ar1)]
                            [y - (int) (4 * ar2)] + spectrum[x -
                                                             (int) (4 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (4
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (4 * ar1)][y -
                                                            (int) (4 *
                                                                   ar2)] +
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (4 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (4 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y + (int) ar2]);
                     e -=
                         70 * (spectrum[x][y - (int) (4 * ar2)] +
                               spectrum[x - (int) (4 * ar1)][y] +
                               spectrum[x][y + (int) (4 * ar2)] +
                               spectrum[x + (int) (4 * ar1)][y]);
                     e -=
                         64 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x -
                                                           (int) (3 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (3 * ar1)][y -
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     e +=
                         560 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e -=
                         784 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e +=
                         1568 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     e +=
                         1568 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     e -=
                         1960 * (spectrum[x][y - (int) (2 * ar2)] +
                                 spectrum[x - (int) (2 * ar1)][y] +
                                 spectrum[x][y + (int) (2 * ar2)] +
                                 spectrum[x + (int) (2 * ar1)][y]);
                     e -=
                         3136 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y - (int) ar2]);
                     e +=
                         3920 * (spectrum[x][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y] +
                                 spectrum[x][y + (int) ar2] + spectrum[x +
                                                                       (int)
                                                                       ar1]
                                 [y]);
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
      }
   }
   
   else if (direction == BACK2_DECREASING_WINDOW) {
      if (filter_type == BACK2_SUCCESSIVE_FILTERING) {
         if (filter_order == BACK2_ORDER2) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER4) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               ar1 = r1 / 2;
               ar2 = r2 / 2;
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER6) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x - r1][y - (int) ar2] +
                           15 * spectrum[x - r1][y + (int) ar2] -
                           6 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           spectrum[x - r1][y + (int) (3 * ar2)]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y - r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           15 * spectrum[x - (int) ar1][y - r2] +
                           15 * spectrum[x + (int) ar1][y - r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           spectrum[x + (int) (3 * ar1)][y - r2]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y + r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           15 * spectrum[x - (int) ar1][y + r2] +
                           15 * spectrum[x + (int) ar1][y + r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           spectrum[x + (int) (3 * ar1)][y + r2]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x + r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x + r1][y - (int) ar2] +
                           15 * spectrum[x + r1][y + (int) ar2] -
                           6 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           spectrum[x + r1][y + (int) (3 * ar2)]) / 20;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER8) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     p1 = spectrum[x - r1][y - r2];
                     p2 = spectrum[x - r1][y + r2];
                     p3 = spectrum[x + r1][y - r2];
                     p4 = spectrum[x + r1][y + r2];
                     s1 = spectrum[x][y - r2];
                     s2 = spectrum[x - r1][y];
                     s3 = spectrum[x + r1][y];
                     s4 = spectrum[x][y + r2];
                     b = (p1 + p2) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x - r1][y - (int) ar2] +
                           4 * spectrum[x - r1][y + (int) ar2] -
                           spectrum[x - r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x - r1][y - (int) ar2] +
                           15 * spectrum[x - r1][y + (int) ar2] -
                           6 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           spectrum[x - r1][y + (int) (3 * ar2)]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - r1][y - (int) (4 * ar2)] +
                           8 * spectrum[x - r1][y - (int) (3 * ar2)] -
                           28 * spectrum[x - r1][y - (int) (2 * ar2)] +
                           56 * spectrum[x - r1][y - (int) ar2] +
                           56 * spectrum[x - r1][y + (int) ar2] -
                           28 * spectrum[x - r1][y + (int) (2 * ar2)] +
                           8 * spectrum[x - r1][y + (int) (3 * ar2)] -
                           spectrum[x - r1][y + (int) (4 * ar2)]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s2)
                        s2 = b;
                     b = (p1 + p3) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y - r2] +
                           4 * spectrum[x - (int) ar1][y - r2] +
                           4 * spectrum[x + (int) ar1][y - r2] -
                           spectrum[x + (int) (2 * ar1)][y - r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y - r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           15 * spectrum[x - (int) ar1][y - r2] +
                           15 * spectrum[x + (int) ar1][y - r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           spectrum[x + (int) (3 * ar1)][y - r2]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - (int) (4 * ar1)][y - r2] +
                           8 * spectrum[x - (int) (3 * ar1)][y - r2] -
                           28 * spectrum[x - (int) (2 * ar1)][y - r2] +
                           56 * spectrum[x - (int) ar1][y - r2] +
                           56 * spectrum[x + (int) ar1][y - r2] -
                           28 * spectrum[x + (int) (2 * ar1)][y - r2] +
                           8 * spectrum[x + (int) (3 * ar1)][y - r2] -
                           spectrum[x + (int) (4 * ar1)][y - r2]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s1)
                        s1 = b;
                     b = (p2 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x - (int) (2 * ar1)][y + r2] +
                           4 * spectrum[x - (int) ar1][y + r2] +
                           4 * spectrum[x + (int) ar1][y + r2] -
                           spectrum[x + (int) (2 * ar1)][y + r2]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x - (int) (3 * ar1)][y + r2] -
                           6 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           15 * spectrum[x - (int) ar1][y + r2] +
                           15 * spectrum[x + (int) ar1][y + r2] -
                           6 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           spectrum[x + (int) (3 * ar1)][y + r2]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x - (int) (4 * ar1)][y + r2] +
                           8 * spectrum[x - (int) (3 * ar1)][y + r2] -
                           28 * spectrum[x - (int) (2 * ar1)][y + r2] +
                           56 * spectrum[x - (int) ar1][y + r2] +
                           56 * spectrum[x + (int) ar1][y + r2] -
                           28 * spectrum[x + (int) (2 * ar1)][y + r2] +
                           8 * spectrum[x + (int) (3 * ar1)][y + r2] -
                           spectrum[x + (int) (4 * ar1)][y + r2]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s4)
                        s4 = b;
                     b = (p3 + p4) / 2.0;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = (-spectrum[x + r1][y - (int) (2 * ar2)] +
                           4 * spectrum[x + r1][y - (int) ar2] +
                           4 * spectrum[x + r1][y + (int) ar2] -
                           spectrum[x + r1][y + (int) (2 * ar2)]) / 6;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = (spectrum[x + r1][y - (int) (3 * ar2)] -
                           6 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           15 * spectrum[x + r1][y - (int) ar2] +
                           15 * spectrum[x + r1][y + (int) ar2] -
                           6 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           spectrum[x + r1][y + (int) (3 * ar2)]) / 20;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = (-spectrum[x + r1][y - (int) (4 * ar2)] +
                           8 * spectrum[x + r1][y - (int) (3 * ar2)] -
                           28 * spectrum[x + r1][y - (int) (2 * ar2)] +
                           56 * spectrum[x + r1][y - (int) ar2] +
                           56 * spectrum[x + r1][y + (int) ar2] -
                           28 * spectrum[x + r1][y + (int) (2 * ar2)] +
                           8 * spectrum[x + r1][y + (int) (3 * ar2)] -
                           spectrum[x + r1][y + (int) (4 * ar2)]) / 70;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b > s3)
                        s3 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p3 + p4) / 2.0;
                     s4 = s4 - (p2 + p4) / 2.0;
                     b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                               p3 +
                                                               p4) / 4.0;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
      }
      
      else if (filter_type == BACK2_ONE_STEP_FILTERING) {
         if (filter_order == BACK2_ORDER2) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = i; y < sizey - i; y++) {
                  for (x = i; x < sizex - i; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER4) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               ar1 = r1 / 2, ar2 = r2 / 2;
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER6) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
         
         else if (filter_order == BACK2_ORDER8) {
            for (i = sampling; i >= 1; i--) {
               r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                   (int) TMath::Min(i, number_of_iterations_y);
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     a = spectrum[x][y];
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = -(spectrum[x - (int) (4 * ar1)]
                            [y - (int) (4 * ar2)] + spectrum[x -
                                                             (int) (4 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (4
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (4 * ar1)][y -
                                                            (int) (4 *
                                                                   ar2)] +
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (4 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (4 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y + (int) ar2]);
                     e -=
                         70 * (spectrum[x][y - (int) (4 * ar2)] +
                               spectrum[x - (int) (4 * ar1)][y] +
                               spectrum[x][y + (int) (4 * ar2)] +
                               spectrum[x + (int) (4 * ar1)][y]);
                     e -=
                         64 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x -
                                                           (int) (3 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (3 * ar1)][y -
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     e +=
                         560 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e -=
                         784 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e +=
                         1568 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     e +=
                         1568 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     e -=
                         1960 * (spectrum[x][y - (int) (2 * ar2)] +
                                 spectrum[x - (int) (2 * ar1)][y] +
                                 spectrum[x][y + (int) (2 * ar2)] +
                                 spectrum[x + (int) (2 * ar1)][y]);
                     e -=
                         3136 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y - (int) ar2]);
                     e +=
                         3920 * (spectrum[x][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y] +
                                 spectrum[x][y + (int) ar2] + spectrum[x +
                                                                       (int)
                                                                       ar1]
                                 [y]);
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                     working_space[x][y] = a;
                  }
               }
               for (y = r2; y < sizey - r2; y++) {
                  for (x = r1; x < sizex - r1; x++) {
                     spectrum[x][y] = working_space[x][y];
                  }
               }
            }
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum2::Background2RectangularRidgesX(float **spectrum,
                                                      int sizex, int sizey,
                                                      int
                                                      number_of_iterations,
                                                      int direction,
                                                      int filter_order) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION -                   */ 
/*      RECTANGULAR 1D-RIDGES IN X-DIMENSION                               */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed to the array pointed by spectrum pointer.     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the array of source spectrum		   */ 
/*	sizex-x length of spectrum                      		   */ 
/*	sizey-y length of spectrum                      		   */ 
/*	number_of_iterations-maximal x width of clipping window            */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK2_INCREASING_WINDOW                 */ 
/*                                 BACK2_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK2_ORDER2                          */ 
/*                                   BACK2_ORDER4                          */ 
/*                                   BACK2_ORDER6                          */ 
/*                                   BACK2_ORDER8	                   */ 
/*									   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, x, y, sampling;
   float a, b, c, d, e, ai;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizex < 2 * number_of_iterations + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   sampling = number_of_iterations;
   if (direction == BACK2_INCREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = 1; i <= sampling; i++) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = 1; i <= sampling; i++) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = 1; i <= sampling; i++) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x - (int) (3 * ai)][y] / 20;
                  d -= 6 * spectrum[x - (int) (2 * ai)][y] / 20;
                  d += 15 * spectrum[x - (int) ai][y] / 20;
                  d += 15 * spectrum[x + (int) ai][y] / 20;
                  d -= 6 * spectrum[x + (int) (2 * ai)][y] / 20;
                  d += spectrum[x + (int) (3 * ai)][y] / 20;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = 1; i <= sampling; i++) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x - (int) (3 * ai)][y] / 20;
                  d -= 6 * spectrum[x - (int) (2 * ai)][y] / 20;
                  d += 15 * spectrum[x - (int) ai][y] / 20;
                  d += 15 * spectrum[x + (int) ai][y] / 20;
                  d -= 6 * spectrum[x + (int) (2 * ai)][y] / 20;
                  d += spectrum[x + (int) (3 * ai)][y] / 20;
                  e = 0;
                  ai = i / 4;
                  e -= spectrum[x - (int) (4 * ai)][y] / 70;
                  e += 8 * spectrum[x - (int) (3 * ai)][y] / 70;
                  e -= 28 * spectrum[x - (int) (2 * ai)][y] / 70;
                  e += 56 * spectrum[x - (int) ai][y] / 70;
                  e += 56 * spectrum[x + (int) ai][y] / 70;
                  e -= 28 * spectrum[x + (int) (2 * ai)][y] / 70;
                  e += 8 * spectrum[x + (int) (3 * ai)][y] / 70;
                  e -= spectrum[x + (int) (4 * ai)][y] / 70;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   
   else if (direction == BACK2_DECREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = sampling; i >= 1; i--) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = sampling; i >= 1; i--) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = sampling; i >= 1; i--) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x - (int) (3 * ai)][y] / 20;
                  d -= 6 * spectrum[x - (int) (2 * ai)][y] / 20;
                  d += 15 * spectrum[x - (int) ai][y] / 20;
                  d += 15 * spectrum[x + (int) ai][y] / 20;
                  d -= 6 * spectrum[x + (int) (2 * ai)][y] / 20;
                  d += spectrum[x + (int) (3 * ai)][y] / 20;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = sampling; i >= 1; i--) {
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  a = spectrum[x][y];
                  b = (spectrum[x - i][y] + spectrum[x + i][y]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x - (int) (2 * ai)][y] / 6;
                  c += 4 * spectrum[x - (int) ai][y] / 6;
                  c += 4 * spectrum[x + (int) ai][y] / 6;
                  c -= spectrum[x + (int) (2 * ai)][y] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x - (int) (3 * ai)][y] / 20;
                  d -= 6 * spectrum[x - (int) (2 * ai)][y] / 20;
                  d += 15 * spectrum[x - (int) ai][y] / 20;
                  d += 15 * spectrum[x + (int) ai][y] / 20;
                  d -= 6 * spectrum[x + (int) (2 * ai)][y] / 20;
                  d += spectrum[x + (int) (3 * ai)][y] / 20;
                  e = 0;
                  ai = i / 4;
                  e -= spectrum[x - (int) (4 * ai)][y] / 70;
                  e += 8 * spectrum[x - (int) (3 * ai)][y] / 70;
                  e -= 28 * spectrum[x - (int) (2 * ai)][y] / 70;
                  e += 56 * spectrum[x - (int) ai][y] / 70;
                  e += 56 * spectrum[x + (int) ai][y] / 70;
                  e -= 28 * spectrum[x + (int) (2 * ai)][y] / 70;
                  e += 8 * spectrum[x + (int) (3 * ai)][y] / 70;
                  e -= spectrum[x + (int) (4 * ai)][y] / 70;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = 0; y < sizey; y++) {
               for (x = i; x < sizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum2::Background2RectangularRidgesY(float **spectrum,
                                                      int sizex, int sizey,
                                                      int
                                                      number_of_iterations,
                                                      int direction,
                                                      int filter_order) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION -                   */ 
/*      RECTANGULAR 1D-RIDGES IN Y-DIMENSION                               */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed to the array pointed by spectrum pointer.     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the array of source spectrum		   */ 
/*	sizex-x length of spectrum                      		   */ 
/*	sizey-y length of spectrum                      		   */ 
/*	number_of_iterations-maximal y width of clipping window            */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK2_INCREASING_WINDOW                 */ 
/*                                 BACK2_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK2_ORDER2                          */ 
/*                                   BACK2_ORDER4                          */ 
/*                                   BACK2_ORDER6                          */ 
/*                                   BACK2_ORDER8	                   */ 
/*									   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, x, y, sampling;
   float a, b, c, d, e, ai;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizey < 2 * number_of_iterations + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   sampling = number_of_iterations;
   if (direction == BACK2_INCREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = 1; i <= sampling; i++) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = 1; i <= sampling; i++) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = 1; i <= sampling; i++) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x][y - (int) (3 * ai)] / 20;
                  d -= 6 * spectrum[x][y - (int) (2 * ai)] / 20;
                  d += 15 * spectrum[x][y - (int) ai] / 20;
                  d += 15 * spectrum[x][y + (int) ai] / 20;
                  d -= 6 * spectrum[x][y + (int) (2 * ai)] / 20;
                  d += spectrum[x][y + (int) (3 * ai)] / 20;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = 1; i <= sampling; i++) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x][y - (int) (3 * ai)] / 20;
                  d -= 6 * spectrum[x][y - (int) (2 * ai)] / 20;
                  d += 15 * spectrum[x][y - (int) ai] / 20;
                  d += 15 * spectrum[x][y + (int) ai] / 20;
                  d -= 6 * spectrum[x][y + (int) (2 * ai)] / 20;
                  d += spectrum[x][y + (int) (3 * ai)] / 20;
                  e = 0;
                  ai = i / 4;
                  e -= spectrum[x][y - (int) (4 * ai)] / 70;
                  e += 8 * spectrum[x][y - (int) (3 * ai)] / 70;
                  e -= 28 * spectrum[x][y - (int) (2 * ai)] / 70;
                  e += 56 * spectrum[x][y - (int) ai] / 70;
                  e += 56 * spectrum[x][y + (int) ai] / 70;
                  e -= 28 * spectrum[x][y + (int) (2 * ai)] / 70;
                  e += 8 * spectrum[x][y + (int) (3 * ai)] / 70;
                  e -= spectrum[x][y + (int) (4 * ai)] / 70;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   
   else if (direction == BACK2_DECREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = sampling; i >= 1; i--) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = sampling; i >= 1; i--) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = sampling; i >= 1; i--) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x][y - (int) (3 * ai)] / 20;
                  d -= 6 * spectrum[x][y - (int) (2 * ai)] / 20;
                  d += 15 * spectrum[x][y - (int) ai] / 20;
                  d += 15 * spectrum[x][y + (int) ai] / 20;
                  d -= 6 * spectrum[x][y + (int) (2 * ai)] / 20;
                  d += spectrum[x][y + (int) (3 * ai)] / 20;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = sampling; i >= 1; i--) {
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  a = spectrum[x][y];
                  b = (spectrum[x][y - i] + spectrum[x][y + i]) / 2.0;
                  c = 0;
                  ai = i / 2;
                  c -= spectrum[x][y - (int) (2 * ai)] / 6;
                  c += 4 * spectrum[x][y - (int) ai] / 6;
                  c += 4 * spectrum[x][y + (int) ai] / 6;
                  c -= spectrum[x][y + (int) (2 * ai)] / 6;
                  d = 0;
                  ai = i / 3;
                  d += spectrum[x][y - (int) (3 * ai)] / 20;
                  d -= 6 * spectrum[x][y - (int) (2 * ai)] / 20;
                  d += 15 * spectrum[x][y - (int) ai] / 20;
                  d += 15 * spectrum[x][y + (int) ai] / 20;
                  d -= 6 * spectrum[x][y + (int) (2 * ai)] / 20;
                  d += spectrum[x][y + (int) (3 * ai)] / 20;
                  e = 0;
                  ai = i / 4;
                  e -= spectrum[x][y - (int) (4 * ai)] / 70;
                  e += 8 * spectrum[x][y - (int) (3 * ai)] / 70;
                  e -= 28 * spectrum[x][y - (int) (2 * ai)] / 70;
                  e += 56 * spectrum[x][y - (int) ai] / 70;
                  e += 56 * spectrum[x][y + (int) ai] / 70;
                  e -= 28 * spectrum[x][y + (int) (2 * ai)] / 70;
                  e += 8 * spectrum[x][y + (int) (3 * ai)] / 70;
                  e -= spectrum[x][y + (int) (4 * ai)] / 70;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (x = 0; x < sizex; x++) {
               for (y = i; y < sizey - i; y++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum2::Background2SkewRidges(float **spectrum, int sizex,
                                              int sizey,
                                              int number_of_iterations_x,
                                              int number_of_iterations_y,
                                              int direction,
                                              int filter_order) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - SKEW RIDGES       */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed to the array pointed by spectrum pointer.     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the array of source spectrum		   */ 
/*	sizex-x length of spectrum                      		   */ 
/*	sizey-y length of spectrum                      		   */ 
/*	number_of_iterations_x-maximal x width of clipping window          */ 
/*	number_of_iterations_y-maximal y width of clipping window          */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK2_INCREASING_WINDOW                 */ 
/*                                 BACK2_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK2_ORDER2                          */ 
/*                                   BACK2_ORDER4                          */ 
/*                                   BACK2_ORDER6                          */ 
/*                                   BACK2_ORDER8	                   */ 
/*									   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, x, y, sampling, r1, r2;
   float a, b, c, d, e, p, ar1, ar2, array[32];
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations_x < 1 || number_of_iterations_y < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizex < 2 * number_of_iterations_x + 1
        || sizey < 2 * number_of_iterations_y + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   sampling =
       (int) TMath::Max(number_of_iterations_x, number_of_iterations_y);
   if (direction == BACK2_INCREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            ar1 = r1 / 2, ar2 = r2 / 2;
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  ar1 = r1 / 2, ar2 = r2 / 2;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  ar1 = r1 / 3, ar2 = r2 / 3;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = 300 * (array[0] + array[1] + array[2] + array[3]) -
                      225 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d - 120 * (array[0] + array[1] + array[2] +
                                  array[3]) + 90 * (array[4] + array[5] +
                                                    array[6] + array[7] +
                                                    array[8] + array[9] +
                                                    array[10] +
                                                    array[11]) -
                      36 * (array[12] + array[13] + array[14] + array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d + 20 * (array[0] + array[1] + array[2] +
                                 array[3]) - 15 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) +
                      6 * (array[12] + array[13] + array[14] + array[15] +
                           array[16] + array[17] + array[18] + array[19]) -
                      (array[20] + array[21] + array[22] + array[23]);
                  d = d / 400;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  ar1 = r1 / 2, ar2 = r2 / 2;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  ar1 = r1 / 3, ar2 = r2 / 3;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = 300 * (array[0] + array[1] + array[2] + array[3]) -
                      225 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d - 120 * (array[0] + array[1] + array[2] +
                                  array[3]) + 90 * (array[4] + array[5] +
                                                    array[6] + array[7] +
                                                    array[8] + array[9] +
                                                    array[10] +
                                                    array[11]) -
                      36 * (array[12] + array[13] + array[14] + array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d + 20 * (array[0] + array[1] + array[2] +
                                 array[3]) - 15 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) +
                      6 * (array[12] + array[13] + array[14] + array[15] +
                           array[16] + array[17] + array[18] + array[19]) -
                      (array[20] + array[21] + array[22] + array[23]);
                  d = d / 400;
                  ar1 = r1 / 4, ar2 = r2 / 4;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = 3920 * (array[0] + array[1] + array[2] + array[3]) -
                      3136 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e - 1960 * (array[0] + array[1] + array[2] +
                                   array[3]) + 1568 * (array[4] +
                                                       array[5] +
                                                       array[6] +
                                                       array[7] +
                                                       array[8] +
                                                       array[9] +
                                                       array[10] +
                                                       array[11]) -
                      784 * (array[12] + array[13] + array[14] +
                             array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e + 560 * (array[0] + array[1] + array[2] +
                                  array[3]) - 448 * (array[4] + array[5] +
                                                     array[6] + array[7] +
                                                     array[8] + array[9] +
                                                     array[10] +
                                                     array[11]) +
                      224 * (array[12] + array[13] + array[14] +
                             array[15] + array[16] + array[17] +
                             array[18] + array[19]) - 64 * (array[20] +
                                                            array[21] +
                                                            array[22] +
                                                            array[23]);
                  for (j = -4, k = 0; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (4 * ar2)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (4 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (4 * ar2)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (4 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 31; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e - 70 * (array[0] + array[1] + array[2] +
                                 array[3]) + 56 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) -
                      28 * (array[12] + array[13] + array[14] + array[15] +
                            array[16] + array[17] + array[18] +
                            array[19]) + 8 * (array[20] + array[21] +
                                              array[22] + array[23] +
                                              array[24] + array[25] +
                                              array[26] + array[27]) -
                      (array[28] + array[29] + array[30] + array[31]);
                  e = e / 4900;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   
   else if (direction == BACK2_DECREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            ar1 = r1 / 2, ar2 = r2 / 2;
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  ar1 = r1 / 2, ar2 = r2 / 2;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  ar1 = r1 / 3, ar2 = r2 / 3;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = 300 * (array[0] + array[1] + array[2] + array[3]) -
                      225 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d - 120 * (array[0] + array[1] + array[2] +
                                  array[3]) + 90 * (array[4] + array[5] +
                                                    array[6] + array[7] +
                                                    array[8] + array[9] +
                                                    array[10] +
                                                    array[11]) -
                      36 * (array[12] + array[13] + array[14] + array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d + 20 * (array[0] + array[1] + array[2] +
                                 array[3]) - 15 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) +
                      6 * (array[12] + array[13] + array[14] + array[15] +
                           array[16] + array[17] + array[18] + array[19]) -
                      (array[20] + array[21] + array[22] + array[23]);
                  d = d / 400;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y - r2];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + r1][y + j * r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x + j * r1][y + r1];
                  }
                  for (j = -1; j < 1; j++, k++) {
                     array[k] = spectrum[x - r1][y + j * r1];
                  }
                  for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * (array[0] + array[1] + array[2] + array[3]) -
                        (array[4] + array[5] + array[6] + array[7])) / 4;
                  ar1 = r1 / 2, ar2 = r2 / 2;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = 24 * (array[0] + array[1] + array[2] + array[3]) -
                      16 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  c = c - 6 * (array[0] + array[1] + array[2] +
                                array[3]) + 4 * (array[4] + array[5] +
                                                 array[6] + array[7] +
                                                 array[8] + array[9] +
                                                 array[10] + array[11]) -
                      (array[12] + array[13] + array[14] + array[15]);
                  c = c / 36;
                  ar1 = r1 / 3, ar2 = r2 / 3;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = 300 * (array[0] + array[1] + array[2] + array[3]) -
                      225 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d - 120 * (array[0] + array[1] + array[2] +
                                  array[3]) + 90 * (array[4] + array[5] +
                                                    array[6] + array[7] +
                                                    array[8] + array[9] +
                                                    array[10] +
                                                    array[11]) -
                      36 * (array[12] + array[13] + array[14] + array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  d = d + 20 * (array[0] + array[1] + array[2] +
                                 array[3]) - 15 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) +
                      6 * (array[12] + array[13] + array[14] + array[15] +
                           array[16] + array[17] + array[18] + array[19]) -
                      (array[20] + array[21] + array[22] + array[23]);
                  d = d / 400;
                  ar1 = r1 / 4, ar2 = r2 / 4;
                  for (j = -1, k = 0; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y - (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) ar1][y + (int) (j * ar1)];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y + (int) ar2];
                  } for (j = -1; j < 1; j++, k++) {
                     array[k] =
                         spectrum[x - (int) ar1][y + (int) (j * ar1)];
                  } for (j = 7; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = 3920 * (array[0] + array[1] + array[2] + array[3]) -
                      3136 * (array[4] + array[5] + array[6] + array[7]);
                  for (j = -2, k = 0; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (2 * ar2)];
                  } for (j = -2; j < 2; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (2 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 15; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e - 1960 * (array[0] + array[1] + array[2] +
                                   array[3]) + 1568 * (array[4] +
                                                       array[5] +
                                                       array[6] +
                                                       array[7] +
                                                       array[8] +
                                                       array[9] +
                                                       array[10] +
                                                       array[11]) -
                      784 * (array[12] + array[13] + array[14] +
                             array[15]);
                  for (j = -3, k = 0; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (3 * ar2)];
                  } for (j = -3; j < 3; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (3 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 23; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e + 560 * (array[0] + array[1] + array[2] +
                                  array[3]) - 448 * (array[4] + array[5] +
                                                     array[6] + array[7] +
                                                     array[8] + array[9] +
                                                     array[10] +
                                                     array[11]) +
                      224 * (array[12] + array[13] + array[14] +
                             array[15] + array[16] + array[17] +
                             array[18] + array[19]) - 64 * (array[20] +
                                                            array[21] +
                                                            array[22] +
                                                            array[23]);
                  for (j = -4, k = 0; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y -
                                                       (int) (4 * ar2)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (4 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x + (int) (j * ar1)][y +
                                                       (int) (4 * ar2)];
                  } for (j = -4; j < 4; j++, k++) {
                     array[k] =
                         spectrum[x - (int) (4 * ar1)][y +
                                                       (int) (j * ar1)];
                  } for (j = 31; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  e = e - 70 * (array[0] + array[1] + array[2] +
                                 array[3]) + 56 * (array[4] + array[5] +
                                                   array[6] + array[7] +
                                                   array[8] + array[9] +
                                                   array[10] + array[11]) -
                      28 * (array[12] + array[13] + array[14] + array[15] +
                            array[16] + array[17] + array[18] +
                            array[19]) + 8 * (array[20] + array[21] +
                                              array[22] + array[23] +
                                              array[24] + array[25] +
                                              array[26] + array[27]) -
                      (array[28] + array[29] + array[30] + array[31]);
                  e = e / 4900;
                  if (b < e)
                     b = e;
                  if (b < d)
                     b = d;
                  if (b < c)
                     b = c;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
const char *TSpectrum2::Background2NonlinearRidges(float **spectrum,
                                                   int sizex, int sizey,
                                                   int
                                                   number_of_iterations_x,
                                                   int
                                                   number_of_iterations_y,
                                                   int direction,
                                                   int filter_order) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - NONLINEAR RIDGES  */ 
/*	This function calculates background spectrum from source spectrum. */ 
/*	The result is placed to the array pointed by spectrum pointer.     */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	spectrum-pointer to the array of source spectrum		   */ 
/*	sizex-x length of spectrum                      		   */ 
/*	sizey-y length of spectrum                      		   */ 
/*	number_of_iterations_x-maximal x width of clipping window          */ 
/*	number_of_iterations_y-maximal y width of clipping window          */ 
/*                           for details we refer to manual	           */ 
/*	direction- direction of change of clipping window                  */ 
/*               - possible values=BACK2_INCREASING_WINDOW                 */ 
/*                                 BACK2_DECREASING_WINDOW                 */ 
/*	filter_order-order of clipping filter,                             */ 
/*                  -possible values=BACK2_ORDER2                          */ 
/*                                   BACK2_ORDER4                          */ 
/*                                   BACK2_ORDER6                          */ 
/*                                   BACK2_ORDER8	                   */ 
/*									   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, x, y, sampling, r1, r2, x1max, x2max, y1max, y2max;
   float a, b, c, d, e, f, p, ar1, ar2, array[1000], p1, p2, p3, p4, s1,
       s2, s3, s4;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations_x < 1 || number_of_iterations_y < 1)
      return "Width of Clipping Window Must Be Positive";
   if (sizex < 2 * number_of_iterations_x + 1
        || sizey < 2 * number_of_iterations_y + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];
   sampling =
       (int) TMath::Max(number_of_iterations_x, number_of_iterations_y);
   if (direction == BACK2_INCREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            ar1 = r1 / 2, ar2 = r2 / 2;
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = -(spectrum[x - (int) (4 * ar1)]
                            [y - (int) (4 * ar2)] + spectrum[x -
                                                             (int) (4 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (4
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (4 * ar1)][y -
                                                            (int) (4 *
                                                                   ar2)] +
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (4 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (4 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y + (int) ar2]);
                     e -=
                         70 * (spectrum[x][y - (int) (4 * ar2)] +
                               spectrum[x - (int) (4 * ar1)][y] +
                               spectrum[x][y + (int) (4 * ar2)] +
                               spectrum[x + (int) (4 * ar1)][y]);
                     e -=
                         64 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x -
                                                           (int) (3 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (3 * ar1)][y -
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     e +=
                         560 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e -=
                         784 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e +=
                         1568 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     e +=
                         1568 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     e -=
                         1960 * (spectrum[x][y - (int) (2 * ar2)] +
                                 spectrum[x - (int) (2 * ar1)][y] +
                                 spectrum[x][y + (int) (2 * ar2)] +
                                 spectrum[x + (int) (2 * ar1)][y]);
                     e -=
                         3136 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y - (int) ar2]);
                     e +=
                         3920 * (spectrum[x][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y] +
                                 spectrum[x][y + (int) ar2] + spectrum[x +
                                                                       (int)
                                                                       ar1]
                                 [y]);
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = 3920 * array[0] - 3136 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r1 - 1] +
                                                        array[r1]) -
                         784 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r2 - 1] +
                                                        array[r2]) -
                         784 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r1 - 1] +
                                                        array[r1]) -
                         784 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r2 - 1] +
                                                        array[r2]) -
                         784 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r1 / 3 - 1] +
                                array[2 * r1 / 3]) +
                         224 * (array[4 * r1 / 3 - 1] +
                                array[4 * r1 / 3]) - 64 * array[2 * r1 -
                                                                1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r2 / 3 - 1] +
                                array[2 * r2 / 3]) +
                         224 * (array[4 * r2 / 3 - 1] +
                                array[4 * r2 / 3]) - 64 * array[2 * r2 -
                                                                1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r1 / 3 - 1] +
                                array[2 * r1 / 3]) +
                         224 * (array[4 * r1 / 3 - 1] +
                                array[4 * r1 / 3]) - 64 * array[2 * r1 -
                                                                1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r2 / 3 - 1] +
                                array[2 * r2 / 3]) +
                         224 * (array[4 * r2 / 3 - 1] +
                                array[4 * r2 / 3]) - 64 * array[2 * r2 -
                                                                1];
                     for (j = -4 * r1, k = 0; j < 4 * r1; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (4 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r1 / 4 - 1] +
                                                    array[2 * r1 / 4]) -
                         28 * (array[r1 - 1] + array[r1]) +
                         8 * (array[3 * r1 / 4 - 1] + array[3 * r1 / 4]) -
                         array[2 * r1 - 1];
                     for (j = -4 * r2, k = 0; j < 4 * r2; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r2 / 4 - 1] +
                                                    array[2 * r2 / 4]) -
                         28 * (array[r2 - 1] + array[r2]) +
                         8 * (array[3 * r2 / 4 - 1] + array[3 * r2 / 4]) -
                         array[2 * r2 - 1];
                     for (j = -4 * r1, k = 0; j < 4 * r1; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (4 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r1 / 4 - 1] +
                                                    array[2 * r1 / 4]) -
                         28 * (array[r1 - 1] + array[r1]) +
                         8 * (array[3 * r1 / 4 - 1] + array[3 * r1 / 4]) -
                         array[2 * r1 - 1];
                     for (j = -4 * r2, k = 0; j < 4 * r2; j += 4, k++) {
                        array[k] =
                            spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r2 / 4 - 1] +
                                                    array[2 * r2 / 4]) -
                         28 * (array[r2 - 1] + array[r2]) +
                         8 * (array[3 * r2 / 4 - 1] + array[3 * r2 / 4]) -
                         array[2 * r2 - 1];
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   
   else if (direction == BACK2_DECREASING_WINDOW) {
      if (filter_order == BACK2_ORDER2) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER4) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            ar1 = r1 / 2, ar2 = r2 / 2;
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER6) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     d = d / 400;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
      
      else if (filter_order == BACK2_ORDER8) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, number_of_iterations_x), r2 =
                (int) TMath::Min(i, number_of_iterations_y);
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  a = spectrum[x][y];
                  f = 0, x1max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x + j][y - r2];
                     if (array[k] > f) {
                        f = array[k];
                        x1max = k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b = (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y1max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x + r1][y + j];
                     if (array[k] > f) {
                        f = array[k];
                        y1max = k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  f = 0, x2max = 0;
                  for (j = -r1, k = 0; j < r1; j++, k++) {
                     array[k] = spectrum[x - j][y + r2];
                     if (array[k] > f) {
                        f = array[k];
                        x2max = 2 * r1 - k;
                     }
                  }
                  for (j = 2 * r1 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r1 - 1]) / 4;
                  f = 0, y2max = 0;
                  for (j = -r2, k = 0; j < r2; j++, k++) {
                     array[k] = spectrum[x - r1][y - j];
                     if (array[k] > f) {
                        f = array[k];
                        y2max = 2 * r2 - k;
                     }
                  }
                  for (j = 2 * r2 - 1; j > 0; j--) {
                     for (k = 0; k < j; k++) {
                        if (array[k + 1] > array[k]) {
                           p = array[k];
                           array[k] = array[k + 1];
                           array[k + 1] = p;
                        }
                     }
                  }
                  b += (2 * array[0] - array[2 * r2 - 1]) / 4;
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  c = (p1 + p2) / 2.0;
                  if (c > s2)
                     s2 = c;
                  c = (p1 + p3) / 2.0;
                  if (c > s1)
                     s1 = c;
                  c = (p2 + p4) / 2.0;
                  if (c > s4)
                     s4 = c;
                  c = (p3 + p4) / 2.0;
                  if (c > s3)
                     s3 = c;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  if (s1 > 0 && s4 > 0 && x1max == x2max && s2 > 0
                       && s3 > 0 && y1max == y2max) {
                     b = -(spectrum[x - r1][y - r2] +
                            spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                        r2]
                            + spectrum[x + r1][y + r2]) / 4 +
                         (spectrum[x][y - r2] + spectrum[x - r1][y] +
                          spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     c = -(spectrum[x - (int) (2 * ar1)]
                            [y - (int) (2 * ar2)] + spectrum[x -
                                                             (int) (2 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (2
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (2 * ar1)][y -
                                                            (int) (2 *
                                                                   ar2)] +
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     c +=
                         4 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     c -=
                         6 * (spectrum[x][y - (int) (2 * ar2)] +
                              spectrum[x - (int) (2 * ar1)][y] +
                              spectrum[x][y + (int) (2 * ar2)] +
                              spectrum[x + (int) (2 * ar1)][y]);
                     c -=
                         16 * (spectrum[x - (int) ar1][y - (int) ar2] +
                               spectrum[x - (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y + (int) ar2] +
                               spectrum[x + (int) ar1][y - (int) ar2]);
                     c +=
                         24 * (spectrum[x][y - (int) ar2] +
                               spectrum[x - (int) ar1][y] + spectrum[x][y +
                                                                        (int)
                                                                        ar2]
                               + spectrum[x + (int) ar1][y]);
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     d = -(spectrum[x - (int) (3 * ar1)]
                            [y - (int) (3 * ar2)] + spectrum[x -
                                                             (int) (3 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (3
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (3 * ar1)][y -
                                                            (int) (3 *
                                                                   ar2)] +
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     d +=
                         6 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     d -=
                         15 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     d +=
                         20 * (spectrum[x][y - (int) (3 * ar2)] +
                               spectrum[x - (int) (3 * ar1)][y] +
                               spectrum[x][y + (int) (3 * ar2)] +
                               spectrum[x + (int) (3 * ar1)][y]);
                     d -=
                         36 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     d +=
                         90 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     d +=
                         90 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     d -=
                         120 * (spectrum[x][y - (int) (2 * ar2)] +
                                spectrum[x - (int) (2 * ar1)][y] +
                                spectrum[x][y + (int) (2 * ar2)] +
                                spectrum[x + (int) (2 * ar1)][y]);
                     d -=
                         225 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                spectrum[x - (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y + (int) ar2] +
                                spectrum[x + (int) ar1][y - (int) ar2]);
                     d +=
                         300 * (spectrum[x][y - (int) ar2] +
                                spectrum[x - (int) ar1][y] +
                                spectrum[x][y + (int) ar2] + spectrum[x +
                                                                      (int)
                                                                      ar1]
                                [y]);
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     e = -(spectrum[x - (int) (4 * ar1)]
                            [y - (int) (4 * ar2)] + spectrum[x -
                                                             (int) (4 *
                                                                    ar1)][y
                                                                          +
                                                                          (int)
                                                                          (4
                                                                           *
                                                                           ar2)]
                            + spectrum[x + (int) (4 * ar1)][y -
                                                            (int) (4 *
                                                                   ar2)] +
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e +=
                         8 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (4 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (4
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (4 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (4 * ar2)]);
                     e -=
                         28 *
                         (spectrum[x - (int) (4 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (4 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (4 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (4 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (4 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (4 * ar2)]);
                     e +=
                         56 *
                         (spectrum[x - (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (4 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (4 * ar1)][y + (int) ar2]);
                     e -=
                         70 * (spectrum[x][y - (int) (4 * ar2)] +
                               spectrum[x - (int) (4 * ar1)][y] +
                               spectrum[x][y + (int) (4 * ar2)] +
                               spectrum[x + (int) (4 * ar1)][y]);
                     e -=
                         64 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x -
                                                           (int) (3 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (3 * ar1)][y -
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (3 * ar2)] + spectrum[x +
                                                           (int) (2 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (3
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (3 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (3 * ar2)]);
                     e +=
                         224 *
                         (spectrum[x - (int) (3 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x +
                                                           (int) (3 *
                                                                  ar1)][y -
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (3 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (3 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (3 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (3 * ar2)]);
                     e -=
                         448 *
                         (spectrum[x - (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (3 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (3 * ar1)][y + (int) ar2]);
                     e +=
                         560 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e -=
                         784 *
                         (spectrum[x - (int) (2 * ar1)]
                          [y - (int) (2 * ar2)] + spectrum[x -
                                                           (int) (2 *
                                                                  ar1)][y +
                                                                        (int)
                                                                        (2
                                                                         *
                                                                         ar2)]
                          + spectrum[x + (int) (2 * ar1)][y -
                                                          (int) (2 *
                                                                 ar2)] +
                          spectrum[x + (int) (2 * ar1)][y +
                                                        (int) (2 * ar2)]);
                     d += 20 * (spectrum[x][y - (int) (3 * ar2)] +
                                spectrum[x - (int) (3 * ar1)][y] +
                                spectrum[x][y + (int) (3 * ar2)] +
                                spectrum[x + (int) (3 * ar1)][y]);
                     e +=
                         1568 *
                         (spectrum[x - (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y - (int) (2 * ar2)] +
                          spectrum[x - (int) ar1][y + (int) (2 * ar2)] +
                          spectrum[x + (int) ar1][y + (int) (2 * ar2)]);
                     e +=
                         1568 *
                         (spectrum[x - (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y - (int) ar2] +
                          spectrum[x - (int) (2 * ar1)][y + (int) ar2] +
                          spectrum[x + (int) (2 * ar1)][y + (int) ar2]);
                     e -=
                         1960 * (spectrum[x][y - (int) (2 * ar2)] +
                                 spectrum[x - (int) (2 * ar1)][y] +
                                 spectrum[x][y + (int) (2 * ar2)] +
                                 spectrum[x + (int) (2 * ar1)][y]);
                     e -=
                         3136 * (spectrum[x - (int) ar1][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y + (int) ar2] +
                                 spectrum[x + (int) ar1][y - (int) ar2]);
                     e +=
                         3920 * (spectrum[x][y - (int) ar2] +
                                 spectrum[x - (int) ar1][y] +
                                 spectrum[x][y + (int) ar2] + spectrum[x +
                                                                       (int)
                                                                       ar1]
                                 [y]);
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  
                  else {
                     ar1 = r1 / 2, ar2 = r2 / 2;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c + 24 * array[0] - 16 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r1 - 1] +
                                                  array[r1]) -
                         array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     c = c - 6 * array[0] + 4 * (array[r2 - 1] +
                                                  array[r2]) -
                         array[2 * r2 - 1];
                     c = c / 36;
                     ar1 = r1 / 3, ar2 = r2 / 3;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = 300 * array[0] - 255 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d += 300 * array[0] - 225 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r1 - 1] +
                                                     array[r1]) -
                         36 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d - 120 * array[0] + 90 * (array[r2 - 1] +
                                                     array[r2]) -
                         36 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r1 / 3 - 1] +
                                                    array[2 * r1 / 3]) +
                         6 * (array[4 * r1 / 3 - 1] + array[4 * r1 / 3]) -
                         array[2 * r1 - 1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     d = d + 20 * array[0] - 15 * (array[2 * r2 / 3 - 1] +
                                                    array[2 * r2 / 3]) +
                         6 * (array[4 * r2 / 3 - 1] + array[4 * r2 / 3]) -
                         array[2 * r2 - 1];
                     d = d / 400;
                     ar1 = r1 / 4, ar2 = r2 / 4;
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = 3920 * array[0] - 3136 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x + (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r2 - 1];
                     for (j = -r1, k = 0; j < r1; j++, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) ar2];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r1 - 1];
                     for (j = -r2, k = 0; j < r2; j++, k++) {
                        array[k] =
                            spectrum[x - (int) ar1][y +
                                                    (int) (j * ar1 / r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e += 3920 * array[0] - 3136 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r1 - 1] +
                                                        array[r1]) -
                         784 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r2 - 1] +
                                                        array[r2]) -
                         784 * array[2 * r2 - 1];
                     for (j = -2 * r1, k = 0; j < 2 * r1; j += 2, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (2 *
                                                                      ar2)];
                     } for (j = 2 * r1 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r1 - 1] +
                                                        array[r1]) -
                         784 * array[2 * r1 - 1];
                     for (j = -2 * r2, k = 0; j < 2 * r2; j += 2, k++) {
                        array[k] =
                            spectrum[x - (int) (2 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2 - 1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 1960 * array[0] + 1568 * (array[r2 - 1] +
                                                        array[r2]) -
                         784 * array[2 * r2 - 1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r1 / 3 - 1] +
                                array[2 * r1 / 3]) +
                         224 * (array[4 * r1 / 3 - 1] +
                                array[4 * r1 / 3]) - 64 * array[2 * r1 -
                                                                1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r2 / 3 - 1] +
                                array[2 * r2 / 3]) +
                         224 * (array[4 * r2 / 3 - 1] +
                                array[4 * r2 / 3]) - 64 * array[2 * r2 -
                                                                1];
                     for (j = -3 * r1, k = 0; j < 3 * r1; j += 3, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (3 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r1 / 3 - 1] +
                                array[2 * r1 / 3]) +
                         224 * (array[4 * r1 / 3 - 1] +
                                array[4 * r1 / 3]) - 64 * array[2 * r1 -
                                                                1];
                     for (j = -3 * r2, k = 0; j < 3 * r2; j += 3, k++) {
                        array[k] =
                            spectrum[x - (int) (3 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e + 560 * array[0] -
                         448 * (array[2 * r2 / 3 - 1] +
                                array[2 * r2 / 3]) +
                         224 * (array[4 * r2 / 3 - 1] +
                                array[4 * r2 / 3]) - 64 * array[2 * r2 -
                                                                1];
                     for (j = -4 * r1, k = 0; j < 4 * r1; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y -
                                                               (int) (4 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r1 / 4 - 1] +
                                                    array[2 * r1 / 4]) -
                         28 * (array[r1 - 1] + array[r1]) +
                         8 * (array[3 * r1 / 4 - 1] + array[3 * r1 / 4]) -
                         array[2 * r1 - 1];
                     for (j = -4 * r2, k = 0; j < 4 * r2; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (4 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r2 / 4 - 1] +
                                                    array[2 * r2 / 4]) -
                         28 * (array[r2 - 1] + array[r2]) +
                         8 * (array[3 * r2 / 4 - 1] + array[3 * r2 / 4]) -
                         array[2 * r2 - 1];
                     for (j = -4 * r1, k = 0; j < 4 * r1; j += 4, k++) {
                        array[k] =
                            spectrum[x + (int) (j * ar1 / r1)][y +
                                                               (int) (4 *
                                                                      ar2)];
                     } for (j = 2 * r1; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r1 / 4 - 1] +
                                                    array[2 * r1 / 4]) -
                         28 * (array[r1 - 1] + array[r1]) +
                         8 * (array[3 * r1 / 4 - 1] + array[3 * r1 / 4]) -
                         array[2 * r1 - 1];
                     for (j = -4 * r2, k = 0; j < 4 * r2; j += 4, k++) {
                        array[k] =
                            spectrum[x - (int) (4 * ar1)][y +
                                                          (int) (j * ar1 /
                                                                 r2)];
                     } for (j = 2 * r2; j > 0; j--) {
                        for (k = 0; k < j; k++) {
                           if (array[k + 1] > array[k]) {
                              p = array[k];
                              array[k] = array[k + 1];
                              array[k + 1] = p;
                           }
                        }
                     }
                     e = e - 70 * array[0] + 56 * (array[2 * r2 / 4 - 1] +
                                                    array[2 * r2 / 4]) -
                         28 * (array[r2 - 1] + array[r2]) +
                         8 * (array[3 * r2 / 4 - 1] + array[3 * r2 / 4]) -
                         array[2 * r2 - 1];
                     e = e / 4900;
                     if (b < e)
                        b = e;
                     if (b < d)
                        b = d;
                     if (b < c)
                        b = c;
                     if (b < a && b > 0)
                        a = b;
                  }
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < sizey - r2; y++) {
               for (x = r1; x < sizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}

//_____________________________________________________________________________
const char* TSpectrum2::Smooth2Markov(float **source, int sizex, int sizey, int aver_window)
{
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL MARKOV SPECTRUM SMOOTHING FUNCTION                 */
/*                                              			   */
/*	This function calculates smoothed spectrum from source spectrum    */
/*      based on Markov chain method.                                      */
/*	The result is placed in the array pointed by source pointer.       */
/*									   */
/*	Function parameters:						   */
/*	source-pointer to the array of source spectrum   		   */
/*	sizex-x length of source					   */
/*	sizey-y length of source					   */
/*	aver_window-width of averaging smoothing window                	   */
/*									   */
/////////////////////////////////////////////////////////////////////////////
   int xmin, xmax, ymin, ymax, i, j, l;
   double a, b, maxch;
   double nom, nip, nim, sp, sm, spx, spy, smx, smy, plocha = 0;
   if(aver_window <= 0)
      return "Averaging Window must be positive";      
   float **working_space = new float* [sizex];
   for(i = 0; i < sizex; i++)
      working_space[i] = new float[sizey];      
   xmin = 0;
   xmax = sizex - 1;
   ymin = 0;
   ymax = sizey - 1;
   for(i = 0, maxch = 0; i < sizex; i++){
      for(j = 0; j < sizey; j++){
       	 working_space[i][j] = 0;
         if(maxch < source[i][j])
            maxch = source[i][j];
            
         plocha += source[i][j];
      }
   }
   if(maxch == 0)
      return 0;
      
   nom = 0;
   working_space[xmin][ymin] = 1;
   for(i = xmin; i < xmax; i++){
      nip = source[i][ymin] / maxch;
      nim = source[i + 1][ymin] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= aver_window; l++){
         if((i + l) > xmax)
            a = source[xmax][ymin] / maxch;
         
         else
       	    a = source[i + l][ymin] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;
         
         else
       	    a = TMath::Sqrt(a + nip);            
         b = b / a;
         b = TMath::Exp(b);                        	                                                                   
         sp = sp + b;
         if(i - l + 1 < xmin)
            a = source[xmin][ymin] / maxch;
         
         else
       	    a = source[i - l + 1][ymin] / maxch;
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
      a = working_space[i + 1][ymin] = a * working_space[i][ymin];
      nom = nom + a;
   }
   for(i = ymin; i < ymax; i++){
      nip = source[xmin][i] / maxch;
      nim = source[xmin][i + 1] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= aver_window; l++){
         if((i + l) > ymax)
       	    a = source[xmin][ymax] / maxch;
       	    
         else
       	    a = source[xmin][i + l] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;
            
         else
       	    a = TMath::Sqrt(a + nip);            
         b = b / a;
         b = TMath::Exp(b);                        	                                                                            
       	 sp = sp + b;
         if(i - l + 1 < ymin)
            a = source[xmin][ymin] / maxch;
            
         else
            a = source[xmin][i - l + 1] / maxch;
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
      a = working_space[xmin][i + 1] = a * working_space[xmin][i];
      nom = nom + a;
   }
   for(i = xmin; i < xmax; i++){
      for(j = ymin; j < ymax; j++){
       	 nip = source[i][j + 1] / maxch;
         nim = source[i + 1][j + 1] / maxch;
       	 spx = 0,smx = 0;
         for(l = 1; l <= aver_window; l++){
            if(i + l > xmax)
               a = source[xmax][j] / maxch;
               
            else
               a = source[i + l][j] / maxch;
       	    b = a - nip;
            if(a + nip <= 0)
               a = 1;
               
            else
       	       a = TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);                        	                                                                               
            spx = spx + b;
            if(i - l + 1 < xmin)
               a = source[xmin][j] / maxch;
               
            else
               a = source[i - l + 1][j] / maxch;
	    b = a - nim;
            if(a + nim <= 0)
               a = 1;
               
            else
       	       a = TMath::Sqrt(a + nim);            
            b = b / a;
            b = TMath::Exp(b);                        	                                                                                                 
            smx = smx + b;
         }
       	 spy = 0,smy = 0;
         nip = source[i + 1][j] / maxch;
       	 nim = source[i + 1][j + 1] / maxch;
	 for(l = 1; l <= aver_window; l++){
            if(j + l > ymax)
	       a = source[i][ymax]/maxch;
	       
            else
               a = source[i][j + l] / maxch;
            b = a - nip;
            if(a + nip <= 0)
               a = 1;
               
            else
       	       a = TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);                        	                                                                                           
       	    spy = spy + b;
            if(j - l + 1 < ymin)
               a = source[i][ymin] / maxch;
               
            else
               a = source[i][j - l + 1] / maxch;
       	    b = a - nim;
            if(a + nim <= 0)
               a = 1;
               
            else
       	       a = TMath::Sqrt(a + nim);            
            b = b / a;
            b = TMath::Exp(b);                        	                                                                                                             
       	    smy = smy + b;
       	 }
         a = (spx * working_space[i][j + 1] + spy * working_space[i + 1][j]) / (smx +smy);
         working_space[i + 1][j + 1] = a;
         nom = nom + a;
      }
   }
   for(i = xmin; i <= xmax; i++){
      for(j = ymin; j <= ymax; j++){
         working_space[i][j] = working_space[i][j] / nom;
      }
   }
   for(i = 0;i < sizex; i++){
      for(j = 0; j < sizey; j++){
         source[i][j] = plocha * working_space[i][j];
      }
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;   
   return 0;
}

//_______________________________________________________________________________
double TSpectrum2::Lls(double a) 
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


//______________________________________________________________________________________________________________________________
const char *TSpectrum2::Deconvolution2(float **source, const float **resp,
                                       int sizex, int sizey,
                                       int number_of_iterations) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL DECONVOLUTION FUNCTION          		   */ 
/*	This function calculates deconvolution from source spectrum	   */ 
/*	according to response spectrum					   */ 
/*	The result is placed in the matrix pointed by source pointer.	   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the matrix of source spectrum			   */ 
/*	resp-pointer to the matrix of response spectrum			   */ 
/*	sizex-x length of source and response spectra			   */ 
/*	sizey-y length of source and response spectra			   */ 
/*	number_of_iterations, for details we refer to manual		   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, lhx, lhy, i1, i2, j1, j2, k1, k2, lindex, i1min, i1max,
       i2min, i2max, j1min, j1max, j2min, j2max, positx = 0, posity = 0;
   double lda, ldb, ldc, area, maximum = 0;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations <= 0)
      return "Number of iterations must be positive";
   double **working_space = new double *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new double[21 * sizey];
   area = 0;
   lhx = -1, lhy = -1;
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         lda = resp[i][j];
         if (lda != 0) {
            if ((i + 1) > lhx)
               lhx = i + 1;
            if ((j + 1) > lhy)
               lhy = j + 1;
         }
         working_space[i][j] = lda;
         area = area + lda;
         if (lda > maximum) {
            maximum = lda;
            positx = i, posity = j;
         }
      }
   }
   if (lhx == -1 || lhy == -1)
      return ("ZERO RESPONSE DATA");
   
/*calculate at*y and write into p*/ 
       i2min = -lhy + 1, i2max = sizey + lhy - 2;
   i1min = -lhx + 1, i1max = sizex + lhx - 2;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         for (j2 = 0; j2 <= (lhy - 1); j2++) {
            for (j1 = 0; j1 <= (lhx - 1); j1++) {
               k2 = i2 + j2, k1 = i1 + j1;
               if (k2 >= 0 && k2 < sizey && k1 >= 0 && k1 < sizex) {
                  lda = working_space[j1][j2];
                  ldb = source[k1][k2];
                  ldc = ldc + lda * ldb;
               }
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + sizey +
                                              k * 3 * sizey] = ldc;
      }
   }
   
/*calculate matrix b=ht*h*/ 
       i1min = -(lhx - 1), i1max = lhx - 1;
   i2min = -(lhy - 1), i2max = lhy - 1;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         j2min = -i2;
         if (j2min < 0)
            j2min = 0;
         j2max = lhy - 1 - i2;
         if (j2max > lhy - 1)
            j2max = lhy - 1;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = -i1;
            if (j1min < 0)
               j1min = 0;
            j1max = lhx - 1 - i1;
            if (j1max > lhx - 1)
               j1max = lhx - 1;
            for (j1 = j1min; j1 <= j1max; j1++) {
               lda = working_space[j1][j2];
               ldb = working_space[i1 + j1][i2 + j2];
               ldc = ldc + lda * ldb;
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + 10 * sizey +
                                              k * 2 * sizey] = ldc;
      }
   }
   
/*calculate ht*h*ht*y and write into ygold*/ 
       for (i2 = 0; i2 < sizey; i2++) {
      for (i1 = 0; i1 < sizex; i1++) {
         ldc = 0;
         j2min = i2min, j2max = i2max;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = i1min, j1max = i1max;
            for (j1 = j1min; j1 <= j1max; j1++) {
               k = (j1 + sizex) / sizex;
               lda =
                   working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                       10 * sizey +
                                                       k * 2 * sizey];
               k = (i1 + j1 + sizex) / sizex;
               ldb =
                   working_space[(i1 + j1 + sizex) % sizex][i2 + j2 +
                                                            sizey + sizey +
                                                            k * 3 * sizey];
               ldc = ldc + lda * ldb;
            }
         }
         working_space[i1][i2 + 14 * sizey] = ldc;
      }
   }
   
/*calculate matrix cc*/ 
       i2 = 2 * lhy - 2;
   if (i2 > sizey)
      i2 = sizey;
   i2min = -i2, i2max = i2;
   i1 = 2 * lhx - 2;
   if (i1 > sizex)
      i1 = sizex;
   i1min = -i1, i1max = i1;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         j2min = -lhy + i2 + 1;
         if (j2min < -lhy + 1)
            j2min = -lhy + 1;
         j2max = lhy + i2 - 1;
         if (j2max > lhy - 1)
            j2max = lhy - 1;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = -lhx + i1 + 1;
            if (j1min < -lhx + 1)
               j1min = -lhx + 1;
            j1max = lhx + i1 - 1;
            if (j1max > lhx - 1)
               j1max = lhx - 1;
            for (j1 = j1min; j1 <= j1max; j1++) {
               k = (j1 + sizex) / sizex;
               lda =
                   working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                       10 * sizey +
                                                       k * 2 * sizey];
               k = (j1 - i1 + sizex) / sizex;
               ldb =
                   working_space[(j1 - i1 + sizex) % sizex][j2 - i2 +
                                                            sizey +
                                                            10 * sizey +
                                                            k * 2 * sizey];
               ldc = ldc + lda * ldb;
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + 15 * sizey +
                                              k * 2 * sizey] = ldc;
      }
   }
   
/*initialization in x1 matrix*/ 
       for (i2 = 0; i2 < sizey; i2++) {
      for (i1 = 0; i1 < sizex; i1++) {
         working_space[i1][i2 + 19 * sizey] = 1;
         working_space[i1][i2 + 20 * sizey] = 0;
      }
   }
   
	/***START OF ITERATIONS***/ 
       for (lindex = 0; lindex < number_of_iterations; lindex++) {
      for (i2 = 0; i2 < sizey; i2++) {
         for (i1 = 0; i1 < sizex; i1++) {
            lda = working_space[i1][i2 + 19 * sizey];
            ldc = working_space[i1][i2 + 14 * sizey];
            if (lda > 0.000001 && ldc > 0.000001) {
               ldb = 0;
               j2min = i2;
               if (j2min > 2 * lhy - 2)
                  j2min = 2 * lhy - 2;
               j2min = -j2min;
               j2max = sizey - i2 - 1;
               if (j2max > 2 * lhy - 2)
                  j2max = 2 * lhy - 2;
               j1min = i1;
               if (j1min > 2 * lhx - 2)
                  j1min = 2 * lhx - 2;
               j1min = -j1min;
               j1max = sizex - i1 - 1;
               if (j1max > 2 * lhx - 2)
                  j1max = 2 * lhx - 2;
               for (j2 = j2min; j2 <= j2max; j2++) {
                  for (j1 = j1min; j1 <= j1max; j1++) {
                     k = (j1 + sizex) / sizex;
                     ldc =
                         working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                             15 * sizey +
                                                             k * 2 *
                                                             sizey];
                     lda = working_space[i1 + j1][i2 + j2 + 19 * sizey];
                     ldb = ldb + lda * ldc;
                  }
               }
               lda = working_space[i1][i2 + 19 * sizey];
               ldc = working_space[i1][i2 + 14 * sizey];
               if (ldc * lda != 0 && ldb != 0) {
                  lda = lda * ldc / ldb;
               }
               
               else
                  lda = 0;
               working_space[i1][i2 + 20 * sizey] = lda;
            }
         }
      }
      for (i2 = 0; i2 < sizey; i2++) {
         for (i1 = 0; i1 < sizex; i1++)
            working_space[i1][i2 + 19 * sizey] =
                working_space[i1][i2 + 20 * sizey];
      }
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++)
         source[(i + positx) % sizex][(j + posity) % sizey] =
             area * working_space[i][j + 19 * sizey];
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//____________________________________________________________________________
void TSpectrum2::DecFourier2(double *working_space, int num, int iter,
                              int inv) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion calculates Fourier deconvolution using in-place algorithm.//
//   It calculates Fourier transform of the data in working space.          //
//   Function parameters:                                                   //
//      -working_space-pointer of source data. It is replaced by result     //
//      -num-size of vector                                                 //
//      -iter-number of iterations                                          //
//      -inv-1-forward transform, 2-inverse transform                       //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   int nxp2, nxp, i, j, k, m, mxp, j1, j2, n1, n2, it;
   long double a, b, c, d, sign, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   long double val1, val2, val3, val4;
   sign = -1;
   if (inv == 2)
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
            val3 = working_space[num + j1 - 1];
            val4 = working_space[num + j2 - 1];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a - b;
            ti = c - d;
            a = a + b;
            working_space[j1 - 1] = a;
            c = c + d;
            working_space[num + j1 - 1] = c;
            a = tr * wr - ti * wi;
            working_space[j2 - 1] = a;
            a = ti * wr + tr * wi;
            working_space[num + j2 - 1] = a;
         }
      }
   }
   n2 = num / 2;
   n1 = num - 1;
   j = 1;
   for (i = 1; i <= n1; i++) {
      if (i < j) {
         val1 = working_space[j - 1];
         val2 = working_space[num + j - 1];
         val3 = working_space[i - 1];
         val4 = working_space[num + i - 1];
         working_space[i - 1] = val1;
         working_space[num + i - 1] = val2;
         working_space[j - 1] = val3;
         working_space[num + j - 1] = val4;
      }
      k = n2;
      for (; k < j;) {
         j = j - k;
         k = k / 2;
      }
      j = j + k;
   }
   if (inv == 2) {
      a = num;
      for (i = 0; i < num; i++) {
         working_space[i] /= a;
         working_space[i + num] /= a;
      }
   }
   return;
}
const char *TSpectrum2::Deconvolution2HighResolution(float **source,
                                                        const float **resp,
                                                        int sizex,
                                                        int sizey,
                                                        int
                                                        number_of_iterations,
                                                        int
                                                        number_of_repetitions,
                                                        double boost) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL HIGH RESOLUTION DECONVOLUTION FUNCTION		   */ 
/*	This function calculates deconvolution from source spectrum	   */ 
/*	according to response spectrum					   */ 
/*	The result is placed in the matrix pointed by source pointer.	   */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the matrix of source spectrum			   */ 
/*	resp-pointer to the matrix of response spectrum			   */ 
/*	sizex-x length of source and response spectra			   */ 
/*	sizey-y length of source and response spectra			   */ 
/*	number_of_iterations, for details we refer to manual		   */ 
/*	number_of_repetitions, for details we refer to manual		   */ 
/*	boost, boosting factor, for details we refer to manual		   */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, j, k, lhx, lhy, i1, i2, j1, j2, k1, k2, lindex, i1min, i1max,
       i2min, i2max, j1min, j1max, j2min, j2max, positx = 0, posity =
       0, iterx, itery, repet;
   double lda, ldb, ldc, area, maximum = 0, a, b, c, d;
   double ws[8192];
   if (sizex > 4096 || sizey > 4096)
      return ("Maximum Dimension is 4096");
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (number_of_iterations <= 0)
      return "Number of iterations must be positive";
   if (number_of_repetitions <= 0)
      return "Number of repetitions must be positive";
   if (boost <= 0)
      return ("Boosting Factor Must be Positive Number");
   double **working_space = new double *[sizex];
   for (i = 0; i < sizex; i++)
      working_space[i] = new double[22 * sizey];
   
//////////////////////*Fourier deconvolution*///////////////////////////
       for (i = sizex, iterx = 0, j = 1; i > 1;) {
      iterx += 1;
      i = i / 2;
      j = j * 2;
   }
   if (j != sizex)
      return ("SIZE MUST BE POWER OF 2");
   for (i = sizey, itery = 0, j = 1; i > 1;) {
      itery += 1;
      i = i / 2;
      j = j * 2;
   }
   if (j != sizey)
      return ("SIZE MUST BE POWER OF 2");
   
//read response matrix
       area = 0;
   lhx = -1, lhy = -1;
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         lda = resp[i][j];
         if (lda != 0) {
            if ((i + 1) > lhx)
               lhx = i + 1;
            if ((j + 1) > lhy)
               lhy = j + 1;
         }
         working_space[i][j] = lda;
         area = area + lda;
         if (lda > maximum) {
            maximum = lda;
            positx = i, posity = j;
         }
      }
   }
   if (lhx == -1 || lhy == -1)
      return ("ZERO RESPONSE DATA");
   
//forward transform of response
       for (j = 0; j < sizey; j++) {
      for (i = 0; i < sizex; i++) {
         ws[i] = working_space[i][j];
         ws[sizex + i] = 0;
      }
      DecFourier2(ws, sizex, iterx, 1);
      for (i = 0; i < sizex; i++) {
         working_space[i][j] = ws[i];
         working_space[i][j + sizey] = ws[i + sizex];
      }
   }
   for (j = 0; j < sizex; j++) {
      for (i = 0; i < sizey; i++) {
         ws[i] = working_space[j][i];
         ws[i + sizey] = working_space[j][i + sizey];
      }
      DecFourier2(ws, sizey, itery, 1);
      for (i = 0; i < sizey; i++) {
         working_space[j][i] = ws[i];
         working_space[j][i + sizey] = ws[i + sizey];
      }
   }
   
//read source matrix
       for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         working_space[i][j + 2 * sizey] = source[i][j];
      }
   }
   
//forward transform of source
       for (j = 0; j < sizey; j++) {
      for (i = 0; i < sizex; i++) {
         ws[i] = working_space[i][j + 2 * sizey];
         ws[sizex + i] = 0;
      }
      DecFourier2(ws, sizex, iterx, 1);
      for (i = 0; i < sizex; i++) {
         working_space[i][j + 2 * sizey] = ws[i];
         working_space[i][j + 3 * sizey] = ws[i + sizex];
      }
   }
   for (j = 0; j < sizex; j++) {
      for (i = 0; i < sizey; i++) {
         ws[i] = working_space[j][i + 2 * sizey];
         ws[i + sizey] = working_space[j][i + 3 * sizey];
      }
      DecFourier2(ws, sizey, itery, 1);
      for (i = 0; i < sizey; i++) {
         working_space[j][i + 2 * sizey] = ws[i];
         working_space[j][i + 3 * sizey] = ws[i + sizey];
      }
   }
   
//division of complex numbers of transformed source by response
       for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         a = working_space[i][j + 2 * sizey];
         b = working_space[i][j + 3 * sizey];
         c = working_space[i][j];
         d = working_space[i][j + sizey];
         working_space[i][j] = (a * c + b * d) / (c * c + d * d);
         working_space[i][j + sizey] = (b * c - a * d) / (c * c + d * d);
      }
   }
   
//inverse transform of result
       for (j = 0; j < sizex; j++) {
      for (i = 0; i < sizey; i++) {
         ws[i] = working_space[j][i];
         ws[i + sizey] = working_space[j][i + sizey];
      }
      DecFourier2(ws, sizey, itery, 2);
      for (i = 0; i < sizey; i++) {
         working_space[j][i] = ws[i];
         working_space[j][i + sizey] = ws[i + sizey];
      }
   }
   for (j = 0; j < sizey; j++) {
      for (i = 0; i < sizex; i++) {
         ws[i] = working_space[i][j];
         ws[sizex + i] = working_space[i][j + sizey];
      }
      DecFourier2(ws, sizex, iterx, 2);
      for (i = 0; i < sizex; i++) {
         working_space[i][j] = ws[i];
         working_space[i][j + sizey] = ws[i + sizex];
      }
   }
   a = 0;
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         a += working_space[i][j];
      }
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         working_space[i][j] /= a;
         working_space[i][j] *= area;
      }
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         b = working_space[i][j];
         working_space[i][j + 21 * sizey] = Lls(b);
      }
   }
   
////////////////////End of Fourier deconvolution///////////////////////
       
//read response matrix once more
       for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         working_space[i][j] = resp[i][j];
      }
   }
   
/*calculate at*y and write into p*/ 
       i2min = -lhy + 1, i2max = sizey + lhy - 2;
   i1min = -lhx + 1, i1max = sizex + lhx - 2;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         for (j2 = 0; j2 <= (lhy - 1); j2++) {
            for (j1 = 0; j1 <= (lhx - 1); j1++) {
               k2 = i2 + j2, k1 = i1 + j1;
               if (k2 >= 0 && k2 < sizey && k1 >= 0 && k1 < sizex) {
                  lda = working_space[j1][j2];
                  ldb = source[k1][k2];
                  ldc = ldc + lda * ldb;
               }
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + sizey +
                                              k * 3 * sizey] = ldc;
      }
   }
   
/*calculate matrix b=ht*h*/ 
       i1min = -(lhx - 1), i1max = lhx - 1;
   i2min = -(lhy - 1), i2max = lhy - 1;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         j2min = -i2;
         if (j2min < 0)
            j2min = 0;
         j2max = lhy - 1 - i2;
         if (j2max > lhy - 1)
            j2max = lhy - 1;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = -i1;
            if (j1min < 0)
               j1min = 0;
            j1max = lhx - 1 - i1;
            if (j1max > lhx - 1)
               j1max = lhx - 1;
            for (j1 = j1min; j1 <= j1max; j1++) {
               lda = working_space[j1][j2];
               ldb = working_space[i1 + j1][i2 + j2];
               ldc = ldc + lda * ldb;
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + 10 * sizey +
                                              k * 2 * sizey] = ldc;
      }
   }
   
/*calculate ht*h*ht*y and write into ygold*/ 
       for (i2 = 0; i2 < sizey; i2++) {
      for (i1 = 0; i1 < sizex; i1++) {
         ldc = 0;
         j2min = i2min, j2max = i2max;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = i1min, j1max = i1max;
            for (j1 = j1min; j1 <= j1max; j1++) {
               k = (j1 + sizex) / sizex;
               lda =
                   working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                       10 * sizey +
                                                       k * 2 * sizey];
               k = (i1 + j1 + sizex) / sizex;
               ldb =
                   working_space[(i1 + j1 + sizex) % sizex][i2 + j2 +
                                                            sizey + sizey +
                                                            k * 3 * sizey];
               ldc = ldc + lda * ldb;
            }
         }
         working_space[i1][i2 + 14 * sizey] = ldc;
      }
   }
   
/*calculate matrix cc*/ 
       i2 = 2 * lhy - 2;
   if (i2 > sizey)
      i2 = sizey;
   i2min = -i2, i2max = i2;
   i1 = 2 * lhx - 2;
   if (i1 > sizex)
      i1 = sizex;
   i1min = -i1, i1max = i1;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         j2min = -lhy + i2 + 1;
         if (j2min < -lhy + 1)
            j2min = -lhy + 1;
         j2max = lhy + i2 - 1;
         if (j2max > lhy - 1)
            j2max = lhy - 1;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = -lhx + i1 + 1;
            if (j1min < -lhx + 1)
               j1min = -lhx + 1;
            j1max = lhx + i1 - 1;
            if (j1max > lhx - 1)
               j1max = lhx - 1;
            for (j1 = j1min; j1 <= j1max; j1++) {
               k = (j1 + sizex) / sizex;
               lda =
                   working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                       10 * sizey +
                                                       k * 2 * sizey];
               k = (j1 - i1 + sizex) / sizex;
               ldb =
                   working_space[(j1 - i1 + sizex) % sizex][j2 - i2 +
                                                            sizey +
                                                            10 * sizey +
                                                            k * 2 * sizey];
               ldc = ldc + lda * ldb;
            }
         }
         k = (i1 + sizex) / sizex;
         working_space[(i1 + sizex) % sizex][i2 + sizey + 15 * sizey +
                                              k * 2 * sizey] = ldc;
      }
   }
   
/*initialization in x1 matrix*/ 
       for (i = 0, a = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         working_space[i][j + 19 * sizey] =
             working_space[i][j + 21 * sizey];
         a += working_space[i][j + 21 * sizey];
      }
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         working_space[i][j + 19 * sizey] =
             working_space[i][j + 19 * sizey] / a;
         working_space[i][j + 20 * sizey] = 0;
      }
   }
   
	/***START OF ITERATIONS***/ 
       for (repet = 0; repet < number_of_repetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_space[i][j + 19 * sizey] =
                   TMath::Power(working_space[i][j + 19 * sizey], boost);
            }
         }
      }
      for (lindex = 0; lindex < number_of_iterations; lindex++) {
         for (i2 = 0; i2 < sizey; i2++) {
            for (i1 = 0; i1 < sizex; i1++) {
               lda = working_space[i1][i2 + 19 * sizey];
               ldc = working_space[i1][i2 + 14 * sizey];
               ldb = 0;
               j2min = i2;
               if (j2min > 2 * lhy - 2)
                  j2min = 2 * lhy - 2;
               j2min = -j2min;
               j2max = sizey - i2 - 1;
               if (j2max > 2 * lhy - 2)
                  j2max = 2 * lhy - 2;
               j1min = i1;
               if (j1min > 2 * lhx - 2)
                  j1min = 2 * lhx - 2;
               j1min = -j1min;
               j1max = sizex - i1 - 1;
               if (j1max > 2 * lhx - 2)
                  j1max = 2 * lhx - 2;
               for (j2 = j2min; j2 <= j2max; j2++) {
                  for (j1 = j1min; j1 <= j1max; j1++) {
                     k = (j1 + sizex) / sizex;
                     ldc =
                         working_space[(j1 + sizex) % sizex][j2 + sizey +
                                                             15 * sizey +
                                                             k * 2 *
                                                             sizey];
                     lda = working_space[i1 + j1][i2 + j2 + 19 * sizey];
                     ldb = ldb + lda * ldc;
                  }
               }
               lda = working_space[i1][i2 + 19 * sizey];
               ldc = working_space[i1][i2 + 14 * sizey];
               if (ldc * lda != 0 && ldb != 0) {
                  lda = lda * ldc / ldb;
               }
               
               else
                  lda = 0;
               working_space[i1][i2 + 20 * sizey] = lda;
            }
         }
         for (i2 = 0; i2 < sizey; i2++) {
            for (i1 = 0; i1 < sizex; i1++)
               working_space[i1][i2 + 19 * sizey] =
                   working_space[i1][i2 + 20 * sizey];
         }
      }
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++)
         source[(i + positx) % sizex][(j + posity) % sizey] =
             area * working_space[i][j + 19 * sizey];
   }
   for (i = 0; i < sizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}


//_______________________________________________________________________________
    Int_t TSpectrum2::PeakEvaluate(const double *temp, int size, int xmax,
                                   double xmin, bool markov) 
{
   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                      //
//                                                                          //
//   This funcion looks for peaks in the temp vector.                       //
//   It calculates Fourier transform of the data in working space.          //
//   Function parameters:                                                   //
//      -temp-pointer of source data.                                       //
//      -size-size of vector                                                //
//      -xmin-low limit of search                                           //
//      -xmax-upper limit of search                                         //
//      -markov-determines Markov estimate                                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   int i, i1, i2, i3, i4, i5, n1, n2, n3, stav, peak_index;
   double a, b, s, f, si4, fi4, suma, sumai, sold, fold = 0;
   i1 = i2 = i3 = i4 = 0;
   si4 = fi4 = 0;
   stav = 1;
   peak_index = 0;
   sold = 1000000.0;
   suma = 0;
   sumai = 0;
   for (i = 0; i < xmax; i++) {
      s = temp[i], f = temp[i + size];
      if ((s < 0) && (stav >= 2) && (stav <= 5)) {
         a = i + xmin;
         a *= s;
         suma += s;
         sumai += a;
      }
      if ((stav == 1) && (s > f)) {
         stav = 2;
         i1 = i;
      }
      
      else if ((stav == 2) && (s <= f)) {
         stav = 3;
         i2 = i;
      }
      
      else if (stav == 3) {
         if (s <= 0) {
            stav = 4;
            i3 = i;
         }
      }
      
      else if ((stav == 4) && (s >= sold)) {
         si4 = sold;
         fi4 = fold;
         stav = 5;
         i4 = i - 1;
      }
      
      else if ((stav == 5) && (s >= 0)) {
         stav = 6;
         i5 = i;
         if (si4 == 0)
            stav = 0;
         
         else {
            n1 = i5 - i3 + 1;
            a = n1 + 2;
            a = fi4 * a / (2. * si4) + 1 / 2.;
            a = TMath::Abs(a);
            n2 = (int) a;
            a = n1 - 4;
            if (a < 0)
               a = 0;
            a = a * (1 - 2. * (fi4 / si4)) + 1 / 2.;
            a = TMath::Abs(a);
            n3 = (int) (a / fResolution);
            a = TMath::Abs(si4);
            if (markov == false) {
               if (a <= (2.0 * fi4))
                  stav = 0;
               if (n2 >= 1) {
                  if ((i3 - i2 - 1) > n2)
                     stav = 0;
               }
               
               else {
                  if ((i3 - i2 - 1) > 1)
                     stav = 0;
               }
               if ((i2 - i1 + 1) < n3)
                  stav = 0;
               n1 = i5 - i3 + 1;
               a = n1 + 2;
               a = fi4 * a / (2. * si4) + 1 / 2.;
               a = TMath::Abs(a);
               n2 = (int) a;
               a = n1 - 2;
               a = a * (1 - 2. * (fi4 / si4)) + 1 / 2.;
               a = TMath::Abs(a);
               n3 = (int) (a / fResolution);
               a = TMath::Abs(si4);
               if (a <= (2. * fi4))
                  stav = 0;
               if (n2 >= 1) {
                  if ((i3 - i2 - 1) > n2)
                     stav = 0;
               }
               
               else {
                  if ((i3 - i2 - 1) > 1)
                     stav = 0;
               }
               if (temp[0] < temp[size]) {
                  if ((i2 - i1 + 1) < n3)
                     stav = 0;
               }
            }
         }
         if (stav != 0) {
            if (suma != 0)
               b = sumai / suma;
            
            else
               b = i4 + xmin;
            if (peak_index >= fMaxPeaks)
               return (-1);
            
            else {
               fPosition[peak_index] = b;
               peak_index += 1;
            }
         }
         stav = 1;
         suma = 0;
         sumai = 0;
         i = i4;
      }
      sold = s;
      fold = f;
   }
   fNPeaks = peak_index;
   return fNPeaks;
}

//____________________________________________________________________________
Int_t TSpectrum2::Search2HighRes(float **source,float **dest, int sizex, int sizey,
                                 double sigma, double threshold,
                                 bool background_remove,int decon_iterations,
                                 bool markov, int aver_window)
                                     
{
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL HIGH-RESOLUTION PEAK SEARCH FUNCTION		   */
/*	This function searches for peaks in source spectrum		   */
/*      It is based on deconvolution method. First the background is       */
/*      removed (if desired), then Markov spectrum is calculated           */
/*      (if desired), then the response function is generated              */
/*      according to given sigma and deconvolution is carried out.         */
/*									   */
/*	Function parameters:						   */
/*	source-pointer to the matrix of source spectrum			   */
/*	dest-pointer to the matrix of resulting deconvolved spectrum	   */
/*	sizex-x length of source spectrum				   */
/*	sizey-y length of source spectrum				   */
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
   int number_of_iterations = (int)(4 * sigma + 0.5);
   int k, lindex;
   double lda, ldb, ldc, area, maximum;
   int xmin, xmax, l, peak_index = 0, sizex_ext = sizex + 4 * number_of_iterations, sizey_ext = sizey + 4 * number_of_iterations, shift = 2 * number_of_iterations;
   int ymin, ymax, i, j;
   double a, b, maxch, plocha = 0;
   double nom, nip, nim, sp, sm, spx, spy, smx, smy;
   double p1, p2, p3, p4, s1, s2, s3, s4;
   int x, y;
   int lhx, lhy, i1, i2, j1, j2, k1, k2, i1min, i1max, i2min, i2max, j1min, j1max, j2min, j2max, positx, posity;
   if (sigma < 1) {
      Error("Search2HighRes", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }
   	
   if(threshold<=0||threshold>=100){
      Error("Search2HighRes", "Invalid threshold, must be positive and less than 100");
      return 0;
   }
   
   j = (int) (5.0 * sigma + 0.5);
   if (j >= PEAK_WINDOW / 2) {
      Error("Search2HighRes", "Too large sigma");
      return 0;
   }
   
   if (markov == true) {
      if (aver_window <= 0) {
         Error("Search2HighRes", "Averanging window must be positive");
         return 0;
      }
   }
   if(background_remove == true){
      if(sizex < 2 * number_of_iterations + 1 || sizey < 2 * number_of_iterations + 1){
         Error("Search2HighRes", "Too large clipping window");
         return 0;      	
      }
   }   
   i = (int)(4 * sigma + 0.5);
   i = 4 * i;
   double **working_space = new double *[sizex + i];
   for (j = 0; j < sizex + i; j++)
      working_space[j] = new double[15 * (sizey + i)];
   
   for(j = 0; j < sizey_ext; j++){
      for(i = 0; i < sizex_ext; i++){
         if(i < shift){
            if(j < shift)
                  working_space[i][j + sizey_ext] = source[0][0];
                  
            else if(j >= sizey + shift)
                  working_space[i][j + sizey_ext] = source[0][sizey - 1];
                  
            else
                  working_space[i][j + sizey_ext] = source[0][j - shift];
         }
         
         else if(i >= sizex + shift){
            if(j < shift)
               working_space[i][j + sizey_ext] = source[sizex - 1][0];
               
            else if(j >= sizey + shift)
               working_space[i][j + sizey_ext] = source[sizex - 1][sizey - 1];
               
            else
               working_space[i][j + sizey_ext] = source[sizex - 1][j - shift];
         }
         
         else{
            if(j < shift)
               working_space[i][j + sizey_ext] = source[i - shift][0];
               
            else if(j >= sizey + shift)
               working_space[i][j + sizey_ext] = source[i - shift][sizey - 1];
               
            else
               working_space[i][j + sizey_ext] = source[i - shift][j - shift];
         }
      }
   }
   if(background_remove == true){
      for(i = 1; i <= number_of_iterations; i++){
	 for(y = i; y < sizey_ext - i; y++){
            for(x = i; x < sizex_ext - i; x++){
               a = working_space[x][y + sizey_ext];
	       p1 = working_space[x - i][y + sizey_ext - i];
               p2 = working_space[x - i][y + sizey_ext + i];
	       p3 = working_space[x + i][y + sizey_ext - i];
	       p4 = working_space[x + i][y + sizey_ext + i];
	       s1 = working_space[x][y + sizey_ext - i];
	       s2 = working_space[x - i][y + sizey_ext];
               s3 = working_space[x + i][y + sizey_ext];
	       s4 = working_space[x][y + sizey_ext + i];
	       b = (p1 + p2) / 2.0;
	       if(b > s2)
		  s2 = b;
               b = (p1 + p3) / 2.0;
	       if(b > s1)
		  s1 = b;
	       b = (p2 + p4) / 2.0;
	       if(b > s4)
	          s4 = b;
               b = (p3 + p4) / 2.0;
	       if(b > s3)
		  s3 = b;
	       s1 = s1 - (p1 + p3) / 2.0;
	       s2 = s2 - (p1 + p2) / 2.0;
               s3 = s3 - (p3 + p4) / 2.0;
	       s4 = s4 - (p2 + p4) / 2.0;
	       b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
	       if(b < a)
		  a = b;
               working_space[x][y] = a;
	    }
         }
         for(y = i;y < sizey_ext - i; y++){
	    for(x = i; x < sizex_ext - i; x++){
	       working_space[x][y + sizey_ext] = working_space[x][y];
	    }
         }
      }
      for(j = 0;j < sizey_ext; j++){
	 for(i = 0; i < sizex_ext; i++){
            if(i < shift){
               if(j < shift)
                  working_space[i][j + sizey_ext] = source[0][0] - working_space[i][j + sizey_ext];
                  
               else if(j >= sizey + shift)
                  working_space[i][j + sizey_ext] = source[0][sizey - 1] - working_space[i][j + sizey_ext];
                  
               else
                  working_space[i][j + sizey_ext] = source[0][j - shift] - working_space[i][j + sizey_ext];
            }
            
            else if(i >= sizex + shift){
               if(j < shift)
                  working_space[i][j + sizey_ext] = source[sizex - 1][0] - working_space[i][j + sizey_ext];
                  
               else if(j >= sizey + shift)
                  working_space[i][j + sizey_ext] = source[sizex - 1][sizey - 1] - working_space[i][j + sizey_ext];
                  
               else
                  working_space[i][j + sizey_ext] = source[sizex - 1][j - shift] - working_space[i][j + sizey_ext];
            }
            
            else{
               if(j < shift)
                  working_space[i][j + sizey_ext] = source[i - shift][0] - working_space[i][j + sizey_ext];
                  
               else if(j >= sizey + shift)
                  working_space[i][j + sizey_ext] = source[i - shift][sizey - 1] - working_space[i][j + sizey_ext];
                  
               else
                  working_space[i][j + sizey_ext] = source[i - shift][j - shift] - working_space[i][j + sizey_ext];
            }
       	 }
      }
   }
   if(markov == true){
      for(i = 0;i < sizex_ext; i++){
         for(j = 0; j < sizey_ext; j++)
            working_space[i][j + 2 * sizex_ext] = working_space[i][sizey_ext + j];
      }
      xmin = 0;
      xmax = sizex_ext - 1;
      ymin = 0;
      ymax = sizey_ext - 1;
      for(i = 0, maxch = 0; i < sizex_ext; i++){
         for(j = 0; j < sizey_ext; j++){
            working_space[i][j] = 0;
            if(maxch < working_space[i][j + 2 * sizey_ext])
               maxch = working_space[i][j + 2 * sizey_ext];
            plocha += working_space[i][j + 2 * sizey_ext];
         }
      }
      if(maxch == 0)
         return 0;
         
      nom=0;
      working_space[xmin][ymin] = 1;
      for(i = xmin; i < xmax; i++){
	 nip = working_space[i][ymin + 2 * sizey_ext] / maxch;
         nim = working_space[i + 1][ymin + 2 * sizey_ext] / maxch;
         sp = 0,sm = 0;
	 for(l = 1;l <= aver_window; l++){
            if((i + l) > xmax)
	       a = working_space[xmax][ymin + 2 * sizey_ext] / maxch;
	       
            else
               a = working_space[i + l][ymin + 2 * sizey_ext] / maxch;
               
	    b = a - nip;
            if(a + nip <= 0)
               a = 1;
               
            else
        	  a=TMath::Sqrt(a + nip);            
	    b = b / a;
	    b = TMath::Exp(b);                        	    
       	    sp = sp + b;
            if(i - l + 1 < xmin)
               a = working_space[xmin][ymin + 2 * sizey_ext] / maxch;
               
            else
               a = working_space[i - l + 1][ymin + 2 * sizey_ext] / maxch;
	    b = a - nim;
            if(a + nim <= 0)
               a = 1;
               
            else
        	  a=TMath::Sqrt(a + nim);                        
	    b = b / a;
	    b = TMath::Exp(b);                        	    	    
       	    sm = sm + b;
         }
         a = sp / sm;
	 a = working_space[i + 1][ymin] = a * working_space[i][ymin];
	 nom = nom + a;
      }
      for(i = ymin; i < ymax; i++){
	 nip = working_space[xmin][i + 2 * sizey_ext] / maxch;
         nim = working_space[xmin][i + 1 + 2 * sizey_ext] / maxch;
         sp = 0,sm = 0;
	 for(l = 1; l <= aver_window; l++){
            if((i + l) > ymax)
               a = working_space[xmin][ymax + 2 * sizey_ext] / maxch;
               
            else
               a = working_space[xmin][i + l + 2 * sizey_ext] / maxch;
            b = a - nip;
            if(a + nip <= 0)
               a=1;
               
            else
        	  a=TMath::Sqrt(a + nip);            
	    b = b / a;
	    b = TMath::Exp(b);                        	                
       	    sp = sp + b;
            if(i - l + 1 < ymin)
               a = working_space[xmin][ymin + 2 * sizey_ext] / maxch;
               
            else
               a = working_space[xmin][i - l + 1 + 2 * sizey_ext] / maxch;
            b = a - nim;
            if(a + nim <= 0)
               a = 1;
               
            else
        	  a=TMath::Sqrt(a + nim);            
	    b = b / a;
	    b = TMath::Exp(b);                        	                            
       	    sm = sm + b;
         }
         a = sp / sm;
         a = working_space[xmin][i + 1] = a * working_space[xmin][i];
         nom = nom + a;
      }
      for(i = xmin; i < xmax; i++){
         for(j = ymin; j < ymax; j++){
            nip = working_space[i][j + 1 + 2 * sizey_ext] / maxch;
            nim = working_space[i + 1][j + 1 + 2 * sizey_ext] / maxch;
            spx = 0,smx = 0;
	    for(l = 1; l <= aver_window; l++){
               if(i + l > xmax)
                  a = working_space[xmax][j + 2 * sizey_ext] / maxch;
                  
               else
        	  a = working_space[i + l][j + 2 * sizey_ext] / maxch;
               b = a - nip;
               if(a + nip <= 0)
                  a = 1;
                  
               else
        	  a=TMath::Sqrt(a + nip);            
	       b = b / a;
	       b = TMath::Exp(b);                        	                                  
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][j + 2 * sizey_ext] / maxch;
                  
               else
                  a = working_space[i - l + 1][j + 2 * sizey_ext] / maxch;
               b = a - nim;
               if(a + nim <= 0)
                  a=1;
                  
               else
        	  a=TMath::Sqrt(a + nim);            
	       b = b / a;
	       b = TMath::Exp(b);                        	                                              
               smx = smx + b;
            }
            spy = 0,smy = 0;
            nip = working_space[i + 1][j + 2 * sizey_ext] / maxch;
            nim = working_space[i + 1][j + 1 + 2 * sizey_ext] / maxch;
	    for(l = 1; l <= aver_window; l++){
               if(j + l > ymax)
	          a = working_space[i][ymax + 2 * sizey_ext] / maxch;
	          
               else
        	  a = working_space[i][j + l + 2 * sizey_ext] / maxch;
	       b = a - nip;
               if(a + nip <= 0)
                  a = 1;
                  
               else
        	  a=TMath::Sqrt(a + nip);            
	       b = b / a;
	       b = TMath::Exp(b);                        	                                                    
               spy = spy + b;
               if(j - l + 1 < ymin)
                  a = working_space[i][ymin + 2 * sizey_ext] / maxch;
                  
               else
        	  a = working_space[i][j - l + 1 + 2 * sizey_ext] / maxch;
               b=a-nim;
               if(a + nim <= 0)
                  a = 1;
               else
        	  a=TMath::Sqrt(a + nim);            
	       b = b / a;
	       b = TMath::Exp(b);                        	                                                                
               smy = smy + b;
            }
            a = (spx * working_space[i][j + 1] + spy * working_space[i + 1][j]) / (smx + smy);
            working_space[i + 1][j + 1] = a;
       	    nom = nom + a;
         }
      }
      for(i = xmin; i <= xmax; i++){
         for(j = ymin; j <= ymax; j++){
            working_space[i][j] = working_space[i][j] / nom;
         }
      }
      for(i = 0; i < sizex_ext; i++){
         for(j = 0; j < sizey_ext; j++){
            working_space[i][j + sizey_ext] = working_space[i][j] * plocha;
            working_space[i][2 * sizey_ext + j] = working_space[i][sizey_ext + j];
         }
      }
      if(background_remove == true){
         for(i = 1; i <= number_of_iterations; i++){
	    for(y = i; y < sizey_ext - i; y++){
	       for(x = i; x < sizex_ext - i; x++){
	          a = working_space[x][y + sizey_ext];
	          p1 = working_space[x - i][y + sizey_ext - i];
                  p2 = working_space[x - i][y + sizey_ext + i];
        	  p3 = working_space[x + i][y + sizey_ext - i];
	          p4 = working_space[x + i][y + sizey_ext + i];
		  s1 = working_space[x][y + sizey_ext - i];
	          s2 = working_space[x - i][y + sizey_ext];
        	  s3 = working_space[x + i][y + sizey_ext];
	          s4 = working_space[x][y + sizey_ext + i];
                  b = (p1 + p2) / 2.0;
	          if(b > s2)
		     s2 = b;
		     
        	  b = (p1 + p3) / 2.0;
        	  if(b > s1)
		     s1 = b;
		     
        	  b = (p2 + p4) / 2.0;
	          if(b > s4)
		     s4 = b;
		     
        	  b = (p3 + p4) / 2.0;
	          if(b > s3)
		      s3 = b;
        	  s1 = s1 - (p1 + p3) / 2.0;
	          s2 = s2 - (p1 + p2) / 2.0;
 	          s3 = s3 - (p3 + p4) / 2.0;
       		  s4 = s4 - (p2 + p4) / 2.0;
        	  b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
       	          if(b < a)
       	             a = b;
       	             
		   working_space[x][y] = a;
       	       }
	    }
            for(y = i; y < sizey - i; y++){
	       for(x = i; x < sizex - i; x++){
	          working_space[x][y + sizey_ext] = working_space[x][y];
               }
            }
	 }
         for(j = 0; j < sizey_ext; j++){
	    for(i = 0; i < sizex_ext; i++){
               working_space[i][j + sizey_ext] = working_space[i][j + 2 * sizey_ext] - working_space[i][j + sizey_ext];
	    }
         }
      }
   }
//deconvolution starts
   area = 0;
   lhx = -1,lhy = -1;
   positx = 0,posity = 0;
   maximum = 0;
//generate response matrix
   for(i = 0; i < sizex_ext; i++){
      for(j = 0; j < sizey_ext; j++){
         lda = (double)i - 3 * sigma;
         ldb = (double)j - 3 * sigma;
         lda = (lda * lda + ldb * ldb) / (2 * sigma * sigma);
         k=(int)(1000 * TMath::Exp(-lda));
         lda = k;
	 if(lda != 0){
	    if((i + 1) > lhx)
	       lhx = i + 1;
	       
	    if((j + 1) > lhy)
	       lhy = j + 1;
	 }
         working_space[i][j] = lda;
         area = area + lda;
	 if(lda > maximum){
	    maximum = lda;
	    positx = i,posity = j;
         }
      }
   }
//read source matrix
   for(i = 0;i < sizex_ext; i++){
      for(j = 0;j < sizey_ext; j++){
         working_space[i][j + 14 * sizey_ext] = TMath::Abs(working_space[i][j + sizey_ext]);
      }
   }
//calculate matrix b=ht*h
   i = lhx - 1;
   if(i > sizex_ext)
      i = sizex_ext;
      
   j = lhy - 1;
   if(j>sizey_ext)
      j = sizey_ext;
      
   i1min = -i,i1max = i;
   i2min = -j,i2max = j;
   for(i2 = i2min; i2 <= i2max; i2++){
      for(i1 = i1min; i1 <= i1max; i1++){
	 ldc = 0;
	 j2min = -i2;
	 if(j2min < 0)
	    j2min = 0;
	    
	 j2max = lhy - 1 - i2;
	 if(j2max > lhy - 1)
	    j2max = lhy - 1;
	    
	 for(j2 = j2min; j2 <= j2max; j2++){
	    j1min = -i1;
	    if(j1min < 0)
	       j1min = 0;
	       
	    j1max = lhx - 1 - i1;
	    if(j1max > lhx - 1)
	       j1max = lhx - 1;
	       
	    for(j1 = j1min; j1 <= j1max; j1++){
	       lda = working_space[j1][j2];
	       ldb = working_space[i1 + j1][i2 + j2];
	       ldc = ldc + lda * ldb;
	    }
	 }
	 k = (i1 + sizex_ext) / sizex_ext;
	 working_space[(i1 + sizex_ext) % sizex_ext][i2 + sizey_ext + 10 * sizey_ext + k * 2 * sizey_ext] = ldc;
      }
   }
//calculate at*y and write into p
   i = lhx - 1;
   if(i > sizex_ext)
      i = sizex_ext;
	 
   j = lhy - 1;
   if(j > sizey_ext)
      j = sizey_ext;
	 
   i2min = -j,i2max = sizey_ext + j - 1;
   i1min = -i,i1max = sizex_ext + i - 1;
   for(i2 = i2min; i2 <= i2max; i2++){
      for(i1=i1min;i1<=i1max;i1++){
	 ldc=0;
	 for(j2 = 0; j2 <= (lhy - 1); j2++){
	    for(j1 = 0; j1 <= (lhx - 1); j1++){
	       k2 = i2 + j2,k1 = i1 + j1;
	       if(k2 >= 0 && k2 < sizey_ext && k1 >= 0 && k1 < sizex_ext){
		  lda = working_space[j1][j2];
		  ldb = working_space[k1][k2 + 14 * sizey_ext];
		  ldc = ldc + lda * ldb;
	       }
	    }
	 }
	 k = (i1 + sizex_ext) / sizex_ext;
	 working_space[(i1 + sizex_ext) % sizex_ext][i2 + sizey_ext + sizey_ext + k * 3 * sizey_ext] = ldc;
      }
   }
//move matrix p
   for(i2 = 0; i2 < sizey_ext; i2++){
      for(i1 = 0; i1 < sizex_ext; i1++){
	 k = (i1 + sizex_ext) / sizex_ext;
	 ldb = working_space[(i1 + sizex_ext) % sizex_ext][i2 + sizey_ext + sizey_ext + k * 3 * sizey_ext];
         working_space[i1][i2 + 14 * sizey_ext] = ldb;
      }
   }
//initialization in x1 matrix
   for(i2 = 0; i2 < sizey_ext; i2++){
      for(i1 = 0; i1 < sizex_ext; i1++){
	 working_space[i1][i2 + sizey_ext] = 1;
	 working_space[i1][i2 + 2 * sizey_ext] = 0;
      }
   }
//START OF ITERATIONS
   for(lindex = 0; lindex < decon_iterations; lindex++){
      for(i2 = 0; i2 < sizey_ext; i2++){
	 for(i1 = 0; i1 < sizex_ext; i1++){
	    lda = working_space[i1][i2 + sizey_ext];
	    ldc = working_space[i1][i2 + 14 * sizey_ext];
	    if(lda > 0.000001 && ldc > 0.000001){
	       ldb=0;
	       j2min=i2;
	       if(j2min > lhy - 1)
		  j2min = lhy - 1;
		  
	       j2min = -j2min;
	       j2max = sizey_ext - i2 - 1;
	       if(j2max > lhy - 1)
		  j2max = lhy - 1;
		  
	       j1min = i1;
	       if(j1min > lhx - 1)
		  j1min = lhx - 1;
		  
	       j1min = -j1min;
	       j1max = sizex_ext - i1 - 1;
	       if(j1max > lhx - 1)
		  j1max = lhx - 1;
		  
	       for(j2 = j2min; j2 <= j2max; j2++){
		  for(j1 = j1min; j1 <= j1max; j1++){
		     k = (j1 + sizex_ext) / sizex_ext;
		     ldc = working_space[(j1 + sizex_ext) % sizex_ext][j2 + sizey_ext + 10 * sizey_ext + k * 2 * sizey_ext];
		     lda = working_space[i1 + j1][i2 + j2 + sizey_ext];
		     ldb = ldb + lda * ldc;
		  }
	       }
	       lda = working_space[i1][i2 + sizey_ext];
	       ldc = working_space[i1][i2 + 14 * sizey_ext];
	       if(ldc * lda != 0 && ldb != 0){
	          lda =lda * ldc / ldb;
	       }
	       
	       else
	          lda=0;
	       working_space[i1][i2 + 2 * sizey_ext] = lda;
	    }
	 }
      }
      for(i2 = 0; i2 < sizey_ext; i2++){
         for(i1 = 0; i1 < sizex_ext; i1++)
	    working_space[i1][i2 + sizey_ext] = working_space[i1][i2 + 2 * sizey_ext];
      }
   }
//looking for maximum
   maximum=0;
   for(i = 0; i < sizex_ext; i++){
      for(j = 0; j < sizey_ext; j++){
	 working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext] = area * working_space[i][j + sizey_ext];
         if(maximum < working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext])
            maximum = working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext];
            
      }
   }
//searching for peaks in deconvolved spectrum
   for(i = 1; i < sizex_ext - 1; i++){
      for(j = 1; j < sizey_ext - 1; j++){
         if(working_space[i][j] > working_space[i - 1][j] && working_space[i][j] > working_space[i - 1][j - 1] && working_space[i][j] > working_space[i][j - 1] && working_space[i][j] > working_space[i + 1][j - 1] && working_space[i][j] > working_space[i + 1][j] && working_space[i][j] > working_space[i + 1][j + 1] && working_space[i][j] > working_space[i][j + 1] && working_space[i][j] > working_space[i - 1][j + 1]){
            if(i >= shift && i < sizex + shift && j >= shift && j < sizey + shift){
               if(working_space[i][j] > threshold * maximum / 100.0){
               	  if(peak_index < fMaxPeaks){            	               	
                     for(k = i - 1,a = 0,b = 0; k <= i + 1; k++){
                        a += (double)(k - shift) * working_space[k][j];
                        b += working_space[k][j];
                     }
                     a=a/b;
                     if(a < 0)
                        a = 0;
                     	
                     if(a >= sizex)
                        a = sizex - 1;	
                     	
                     fPositionX[peak_index] = a;                        
                     for(k = j - 1,a = 0,b = 0; k <= j + 1; k++){
                        a += (double)(k - shift) * working_space[i][k];
                        b += working_space[i][k];
                     }
                     a=a/b;
                     if(a < 0)
                        a = 0;
                     	
                     if(a >= sizey)
                        a = sizey - 1;	
                     	
                     fPositionY[peak_index] = a;                                             
       		     peak_index += 1;
	          }
                  else{
                     Warning("Search2General","Peak buffer full");
                     return 0;
                  }               	          
               }
            }
         }
      }
   }
//writing back deconvolved spectrum 
   for(i = 0; i < sizex; i++){
      for(j = 0; j < sizey; j++){
         dest[i][j] = working_space[i + shift][j + shift];
      }
   }
   i = (int)(4 * sigma + 0.5);
   i = 4 * i;   
   for (j = 0; j < sizex + i; j++)
      delete[]working_space[j];
   delete[]working_space;
   fNPeaks = peak_index;
   return fNPeaks;
}



//_____________________________________________________________________________
/////////////////BEGINNING OF AUXILIARY FUNCTIONS USED BY FITTING FUNCIONS//////////////////////////
double TSpectrum2::Erfc(double x) 
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
double TSpectrum2::Derfc(double x) 
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
double TSpectrum2::Ourpowl(double a, int pw)
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
void TSpectrum2::StiefelInversion(double **a, int size)
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
double TSpectrum2::Shape2(int num_of_fitted_peaks, double x, double y,
                            const double *parameter, double sigmax,
                            double sigmay, double ro, double a0, double ax,
                            double ay, double txy, double sxy, double tx,
                            double ty, double sx, double sy, double bx,
                            double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates 2D peaks shape function (see manual)               //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -a0,ax,ay-bac kground coefficients                              //
//              -txy,tx,ty, sxy,sy,sx-relative amplitudes                       //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int j;
   double r, p, r1, e, ex, ey, vx, s2, px, py, rx, ry, erx, ery;
   vx = 0;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (x - parameter[7 * j + 1]) / sigmax;
      r = (y - parameter[7 * j + 2]) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (txy != 0) {
            px = 0, py = 0;
            erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
                Erfc(r / s2 + 1 / (2 * by));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            r1 += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            rx = Erfc(p / s2), ry = Erfc(r / s2);
            r1 += 0.5 * sxy * rx * ry;
         }
         vx = vx + parameter[7 * j] * r1;
      }
      p = (x - parameter[7 * j + 5]) / sigmax;
      if (TMath::Abs(p) < 3) {
         e = p * p / 2;
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (tx != 0) {
            px = 0;
            erx = Erfc(p / s2 + 1 / (2 * bx));
            ex = p / (s2 * bx);
            if (TMath::Abs(ex) < 9) {
               px = exp(ex) * erx;
            }
            r1 += 0.5 * tx * px;
         }
         if (sx != 0) {
            rx = Erfc(p / s2);
            r1 += 0.5 * sx * rx;
         }
         vx = vx + parameter[7 * j + 3] * r1;
      }
      r = (y - parameter[7 * j + 6]) / sigmay;
      if (TMath::Abs(r) < 3) {
         e = r * r / 2;
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (ty != 0) {
            py = 0;
            ery = Erfc(r / s2 + 1 / (2 * by));
            ey = r / (s2 * by);
            if (TMath::Abs(ey) < 9) {
               py = exp(ey) * ery;
            }
            r1 += 0.5 * ty * py;
         }
         if (sy != 0) {
            ry = Erfc(r / s2);
            r1 += 0.5 * sy * ry;
         }
         vx = vx + parameter[7 * j + 4] * r1;
      }
   }
   vx = vx + a0 + ax * x + ay * y;
   return (vx);
}
double TSpectrum2::Deramp2(double x, double y, double x0, double y0,
                            double sigmax, double sigmay, double ro,
                            double txy, double sxy, double bx, double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of 2D peaks shape function (see manual) //
//   according to amplitude of 2D peak                                          //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      if (txy != 0) {
         px = 0, py = 0;
         erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         rx = Erfc(p / s2), ry = Erfc(r / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
   }
   return (r1);
}
double TSpectrum2::Derampx(double x, double x0, double sigmax, double tx,
                             double sx, double bx) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of 2D peaks shape function (see manual) //
//   according to amplitude of the ridge                                        //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -tx, sx-relative amplitudes                                     //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r1 = 0, px, erx, rx, ex, s2;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      s2 = TMath::Sqrt(2.0);
      p = p * p / 2;
      if (p < 700)
         r1 = exp(-p);
      
      else {
         r1 = 0;
      }
      if (tx != 0) {
         px = 0;
         erx = Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9) {
            px = exp(ex) * erx;
         }
         r1 += 0.5 * tx * px;
      }
      if (sx != 0) {
         rx = Erfc(p / s2);
         r1 += 0.5 * sx * rx;
      }
   }
   return (r1);
}
double TSpectrum2::Deri02(double x, double y, double a, double x0,
                            double y0, double sigmax, double sigmay,
                            double ro, double txy, double sxy, double bx,
                            double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of 2D peaks shape function (see manual) //
//   according to x position of 2D peak                                         //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * r - p) / sigmax;
      e = e / (1 - ro * ro);
      r1 = r1 * e;
      if (txy != 0) {
         px = 0, py = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * sigmax) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * sigmax)), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         rx = -Derfc(p / s2) / (s2 * sigmax), ry = Erfc(r / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
      r1 = a * r1;
   }
   return (r1);
}
double TSpectrum2::Derderi02(double x, double y, double a, double x0,
                              double y0, double sigmax, double sigmay,
                              double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of 2D peaks shape function       //
//   (see manual) according to x position of 2D peak                            //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * r - p) / sigmax;
      e = e / (1 - ro * ro);
      r1 = r1 * (e * e - 1 / ((1 - ro * ro) * sigmax * sigmax));
      r1 = a * r1;
   }
   return (r1);
}
double TSpectrum2::Derj02(double x, double y, double a, double x0,
                            double y0, double sigmax, double sigmay,
                            double ro, double txy, double sxy, double bx,
                            double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of 2D peaks shape function (see manual) //
//   according to y position of 2D peak                                         //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * p - r) / sigmay;
      e = e / (1 - ro * ro);
      r1 = r1 * e;
      if (txy != 0) {
         px = 0, py = 0;
         ery =
             (-Erfc(r / s2 + 1 / (2 * by)) / (s2 * by * sigmay) -
              Derfc(r / s2 + 1 / (2 * by)) / (s2 * sigmay)), erx =
             Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         ry = -Derfc(r / s2) / (s2 * sigmay), rx = Erfc(p / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
      r1 = a * r1;
   }
   return (r1);
}
double TSpectrum2::Derderj02(double x, double y, double a, double x0,
                               double y0, double sigmax, double sigmay,
                               double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of 2D peaks shape function       //
//   (see manual) according to y position of 2D peak                            //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * p - r) / sigmay;
      e = e / (1 - ro * ro);
      r1 = r1 * (e * e - 1 / ((1 - ro * ro) * sigmay * sigmay));
      r1 = a * r1;
   }
   return (r1);
}
double TSpectrum2::Deri01(double x, double ax, double x0, double sigmax,
                            double tx, double sx, double bx) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of 2D peaks shape function (see manual) //
//   according to x position of 1D ridge                                        //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -ax-amplitude of ridge                                          //
//              -x0-position of peak in x-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -tx, sx-relative amplitudes                                     //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, e, r1 = 0, px, rx, erx, ex, s2;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = p * p / 2;
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      r1 = r1 * p / sigmax;
      if (tx != 0) {
         px = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * sigmax) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * sigmax));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9)
            px = exp(ex) * erx;
         r1 += 0.5 * tx * px;
      }
      if (sx != 0) {
         rx = -Derfc(p / s2) / (s2 * sigmax);
         r1 += 0.5 * sx * rx;
      }
      r1 = ax * r1;
   }
   return (r1);
}
double TSpectrum2::Derderi01(double x, double ax, double x0,
                               double sigmax) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of 2D peaks shape function       //
//   (see manual) according to x position of 1D ridge                           //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -ax-amplitude of ridge                                          //
//              -x0-position of peak in x-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, e, r1 = 0;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      e = p * p / 2;
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      r1 = r1 * (p * p / (sigmax * sigmax) - 1 / (sigmax * sigmax));
      r1 = ax * r1;
   }
   return (r1);
}
double TSpectrum2::Dersigmax(int num_of_fitted_peaks, double x, double y,
                               const double *parameter, double sigmax,
                               double sigmay, double ro, double txy,
                               double sxy, double tx, double sx, double bx,
                               double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to sigmax of peaks.                                              //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy, tx, sx-relative amplitudes                           //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 =
       0, e, a, b, x0, y0, s2, px, py, rx, ry, erx, ery, ex, ey;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - p * p) / sigmax;
         e = e * b / (1 - ro * ro);
         if (txy != 0) {
            px = 0, py = 0;
            erx =
                -Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * sigmax) -
                Derfc(p / s2 + 1 / (2 * bx)) * p / (s2 * sigmax), ery =
                Erfc(r / s2 + 1 / (2 * by));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            e += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            rx = -Derfc(p / s2) * p / (s2 * sigmax), ry = Erfc(r / s2);
            e += 0.5 * sxy * rx * ry;
         }
         r1 = r1 + a * e;
      }
      if (TMath::Abs(p) < 3) {
         x0 = parameter[7 * j + 5];
         p = (x - x0) / sigmax;
         b = p * p / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = 2 * b * e / sigmax;
         if (tx != 0) {
            px = 0;
            erx =
                (-Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * sigmax) -
                 Derfc(p / s2 + 1 / (2 * bx)) * p / (s2 * sigmax));
            ex = p / (s2 * bx);
            if (TMath::Abs(ex) < 9)
               px = exp(ex) * erx;
            e += 0.5 * tx * px;
         }
         if (sx != 0) {
            rx = -Derfc(p / s2) * p / (s2 * sigmax);
            e += 0.5 * sx * rx;
         }
         r1 += parameter[7 * j + 3] * e;
      }
   }
   return (r1);
}
double TSpectrum2::Derdersigmax(int num_of_fitted_peaks, double x,
                                  double y, const double *parameter,
                                  double sigmax, double sigmay,
                                  double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of peaks shape function          //
//   (see manual) according to sigmax of peaks.                                 //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e, a, b, x0, y0;
   int j;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - p * p) / sigmax;
         e = e * (b * b / (1 - ro * ro) -
                   (3 * p * p - 2 * ro * p * r) / (sigmax * sigmax)) / (1 -
                                                                        ro
                                                                        *
                                                                        ro);
         r1 = r1 + a * e;
      }
      if (TMath::Abs(p) < 3) {
         x0 = parameter[7 * j + 5];
         p = (x - x0) / sigmax;
         b = p * p / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = e * (4 * b * b - 6 * b) / (sigmax * sigmax);
         r1 += parameter[7 * j + 3] * e;
      }
   }
   return (r1);
}
double TSpectrum2::Dersigmay(int num_of_fitted_peaks, double x, double y,
                               const double *parameter, double sigmax,
                               double sigmay, double ro, double txy,
                               double sxy, double ty, double sy, double bx,
                               double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to sigmax of peaks.                                              //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy, ty, sy-relative amplitudes                           //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 =
       0, e, a, b, x0, y0, s2, px, py, rx, ry, erx, ery, ex, ey;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - r * r) / sigmay;
         e = e * b / (1 - ro * ro);
         if (txy != 0) {
            px = 0, py = 0;
            ery =
                -Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * sigmay) -
                Derfc(r / s2 + 1 / (2 * by)) * r / (s2 * sigmay), erx =
                Erfc(p / s2 + 1 / (2 * bx));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            e += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            ry = -Derfc(r / s2) * r / (s2 * sigmay), rx = Erfc(p / s2);
            e += 0.5 * sxy * rx * ry;
         }
         r1 = r1 + a * e;
      }
      if (TMath::Abs(r) < 3) {
         y0 = parameter[7 * j + 6];
         r = (y - y0) / sigmay;
         b = r * r / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = 2 * b * e / sigmay;
         if (ty != 0) {
            py = 0;
            ery =
                (-Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * sigmay) -
                 Derfc(r / s2 + 1 / (2 * by)) * r / (s2 * sigmay));
            ey = r / (s2 * by);
            if (TMath::Abs(ey) < 9)
               py = exp(ey) * ery;
            e += 0.5 * ty * py;
         }
         if (sy != 0) {
            ry = -Derfc(r / s2) * r / (s2 * sigmay);
            e += 0.5 * sy * ry;
         }
         r1 += parameter[7 * j + 4] * e;
      }
   }
   return (r1);
}
double TSpectrum2::Derdersigmay(int num_of_fitted_peaks, double x,
                                  double y, const double *parameter,
                                  double sigmax, double sigmay,
                                  double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates second derivative of peaks shape function          //
//   (see manual) according to sigmay of peaks.                                 //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, e, a, b, x0, y0;
   int j;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - r * r) / sigmay;
         e = e * (b * b / (1 - ro * ro) -
                   (3 * r * r - 2 * ro * r * p) / (sigmay * sigmay)) / (1 -
                                                                        ro
                                                                        *
                                                                        ro);
         r1 = r1 + a * e;
      }
      if (TMath::Abs(r) < 3) {
         y0 = parameter[7 * j + 6];
         r = (y - y0) / sigmay;
         b = r * r / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = e * (4 * b * b - 6 * b) / (sigmay * sigmay);
         r1 += parameter[7 * j + 4] * e;
      }
   }
   return (r1);
}
double TSpectrum2::Derro(int num_of_fitted_peaks, double x, double y,
                           const double *parameter, double sx, double sy,
                           double r) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to correlation coefficient ro.                                   //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sx-sigmax of peaks                                             //
//              -sy-sigmay of peaks                                             //
//              -r-correlation coefficient ro                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double px, qx, rx, vx, x0, y0, a, ex, tx;
   int j;
   vx = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      px = (x - x0) / sx;
      qx = (y - y0) / sy;
      if (TMath::Abs(px) < 3 && TMath::Abs(qx) < 3) {
         rx = (px * px - 2 * r * px * qx + qx * qx);
         ex = rx / (2 * (1 - r * r));
         if ((ex) < 700)
            ex = exp(-ex);
         
         else {
            ex = 0;
         }
         tx = px * qx / (1 - r * r);
         tx = tx - r * rx / ((1 - r * r) * (1 - r * r));
         vx = vx + a * ex * tx;
      }
   }
   return (vx);
}
double TSpectrum2::Dertxy(int num_of_fitted_peaks, double x, double y,
                            const double *parameter, double sigmax,
                            double sigmay, double bx, double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude txy.                                       //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, ex, ey, px, py, erx, ery, s2, x0, y0, a;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      px = 0, py = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
          Erfc(r / s2 + 1 / (2 * by));
      ex = p / (s2 * bx), ey = r / (s2 * by);
      if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
         px = exp(ex) * erx, py = exp(ey) * ery;
      }
      r1 += 0.5 * a * px * py;
   }
   return (r1);
}
double TSpectrum2::Dersxy(int num_of_fitted_peaks, double x, double y,
                            const double *parameter, double sigmax,
                            double sigmay) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sxy.                                       //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, rx, ry, x0, y0, a, s2;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      rx = Erfc(p / s2), ry = Erfc(r / s2);
      r1 += 0.5 * a * rx * ry;
   }
   return (r1);
}
double TSpectrum2::Dertx(int num_of_fitted_peaks, double x,
                           const double *parameter, double sigmax,
                           double bx) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude tx.                                        //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r1 = 0, ex, px, erx, s2, ax, x0;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      ax = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      px = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx));
      ex = p / (s2 * bx);
      if (TMath::Abs(ex) < 9) {
         px = exp(ex) * erx;
      }
      r1 += 0.5 * ax * px;
   }
   return (r1);
}
double TSpectrum2::Derty(int num_of_fitted_peaks, double x,
                           const double *parameter, double sigmax,
                           double bx) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude ty.                                        //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r1 = 0, ex, px, erx, s2, ax, x0;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      ax = parameter[7 * j + 4];
      x0 = parameter[7 * j + 6];
      p = (x - x0) / sigmax;
      px = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx));
      ex = p / (s2 * bx);
      if (TMath::Abs(ex) < 9) {
         px = exp(ex) * erx;
      }
      r1 += 0.5 * ax * px;
   }
   return (r1);
}
double TSpectrum2::Dersx(int num_of_fitted_peaks, double x,
                           const double *parameter, double sigmax) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sx.                                        //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r1 = 0, rx, ax, x0, s2;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      ax = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      s2 = TMath::Sqrt(2.0);
      rx = Erfc(p / s2);
      r1 += 0.5 * ax * rx;
   }
   return (r1);
}
double TSpectrum2::Dersy(int num_of_fitted_peaks, double x,
                           const double *parameter, double sigmax) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sy.                                        //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r1 = 0, rx, ax, x0, s2;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      ax = parameter[7 * j + 4];
      x0 = parameter[7 * j + 6];
      p = (x - x0) / sigmax;
      s2 = TMath::Sqrt(2.0);
      rx = Erfc(p / s2);
      r1 += 0.5 * ax * rx;
   }
   return (r1);
}
double TSpectrum2::Derbx(int num_of_fitted_peaks, double x, double y,
                           const double *parameter, double sigmax,
                           double sigmay, double txy, double tx, double bx,
                           double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to slope bx.                                                     //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -txy, tx-relative amplitudes                                    //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, a, x0, y0, s2, px, py, erx, ery, ex, ey;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (txy != 0) {
         px = 0, py = 0;
         erx =
             -Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * bx) -
             Derfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * bx), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * a * txy * px * py;
      }
      a = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      if (tx != 0) {
         px = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * bx) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * bx));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9)
            px = exp(ex) * erx;
         r1 += 0.5 * a * tx * px;
      }
   }
   return (r1);
}
double TSpectrum2::Derby(int num_of_fitted_peaks, double x, double y,
                           const double *parameter, double sigmax,
                           double sigmay, double txy, double ty, double bx,
                           double by) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of peaks shape function (see manual)    //
//   according to slope by.                                                     //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -txy, ty-relative amplitudes                                    //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double p, r, r1 = 0, a, x0, y0, s2, px, py, erx, ery, ex, ey;
   int j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < num_of_fitted_peaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (txy != 0) {
         px = 0, py = 0;
         ery =
             -Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * by) -
             Derfc(r / s2 + 1 / (2 * by)) / (s2 * by * by), erx =
             Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * a * txy * px * py;
      }
      a = parameter[7 * j + 4];
      y0 = parameter[7 * j + 6];
      r = (y - y0) / sigmay;
      if (ty != 0) {
         py = 0;
         ery =
             (-Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * by) -
              Derfc(r / s2 + 1 / (2 * by)) / (s2 * by * by));
         ey = r / (s2 * by);
         if (TMath::Abs(ey) < 9)
            py = exp(ey) * ery;
         r1 += 0.5 * a * ty * py;
      }
   }
   return (r1);
}
double TSpectrum2::Volume(double a, double sx, double sy, double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates volume of a peak                                   //
//      Function parameters:                                                    //
//              -a-amplitude of the peak                                        //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = 2 * a * pi * sx * sy * r;
   return (r);
}
double TSpectrum2::Derpa2(double sx, double sy, double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the volume of a peak                 //
//   according to amplitute                                                     //
//      Function parameters:                                                    //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = 2 * pi * sx * sy * r;
   return (r);
}
double TSpectrum2::Derpsigmax(double a, double sy, double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the volume of a peak                 //
//   according to sigmax                                                        //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sy-sigma of peak                                               //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = a * 2 * pi * sy * r;
   return (r);
}
double TSpectrum2::Derpsigmay(double a, double sx, double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the volume of a peak                 //
//   according to sigmay                                                        //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sx-sigma of peak                                               //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = a * 2 * pi * sx * r;
   return (r);
}
double TSpectrum2::Derpro(double a, double sx, double sy, double ro) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates derivative of the volume of a peak                 //
//   according to ro                                                            //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   double pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = -a * 2 * pi * sx * sy * ro / r;
   return (r);
}


/////////////////END OF AUXILIARY FUNCTIONS USED BY FITTING FUNCION fit2//////////////////////////
    
/////////////////FITTING FUNCTION WITHOUT MATRIX INVERSION///////////////////////////////////////
const char *TSpectrum2::Fit2Awmi(float **source, TSpectrumTwoDimFit * p,
                                  int sizex, int sizey) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL FIT FUNCTION				           */ 
/*      Algorithm without matrix inversion                                 */ 
/*	This function fits the source spectrum. The calling program should */ 
/*      fill in input parameters of the TSpectrumTwoDimFit class 	   */ 
/*	The fitted parameters are written into class pointed by 	   */ 
/*	TSpectrumTwoDimFit structure pointer and fitted data are written   */ 
/*      into source spectrum.                                              */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the matrix of source spectrum			   */ 
/*	p-pointer to the TSpectrumTwoDimFit class, see manual              */ 
/*	sizex-length x of source spectrum                                  */ 
/*	sizey-length y of source spectrum                                  */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, i1, i2, j, k, shift =
       7 * p->number_of_peaks + 14, peak_vel, rozmer, iter, pw,
       regul_cycle, flag;
   double a, b, c, d = 0, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (p->number_of_peaks <= 0)
      return ("INVALID NUMBER OF PEAKS, MUST BE POSITIVE");
   if (p->number_of_iterations <= 0)
      return ("INVALID NUMBER OF ITERATIONS, MUST BE POSITIVE");
   if (p->alpha <= 0 || p->alpha > 1)
      return ("INVALID COEFFICIENT ALPHA, MUST BE > THAN 0 AND <=1");
   if (p->statistic_type != FIT2_OPTIM_CHI_COUNTS
        && p->statistic_type != FIT2_OPTIM_CHI_FUNC_VALUES
        && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD)
      return ("WRONG TYPE OF STATISTIC");
   if (p->alpha_optim != FIT2_ALPHA_HALVING
        && p->alpha_optim != FIT2_ALPHA_OPTIMAL)
      return ("WRONG OPTIMIZATION ALGORITHM");
   if (p->power != FIT2_FIT_POWER2 && p->power != FIT2_FIT_POWER4
        && p->power != FIT2_FIT_POWER6 && p->power != FIT2_FIT_POWER8
        && p->power != FIT2_FIT_POWER10 && p->power != FIT2_FIT_POWER12)
      return ("WRONG POWER");
   if (p->fit_taylor != FIT2_TAYLOR_ORDER_FIRST
        && p->fit_taylor != FIT2_TAYLOR_ORDER_SECOND)
      return ("WRONG ORDER OF TAYLOR DEVELOPMENT");
   if (p->xmin < 0 || p->xmin > p->xmax)
      return ("INVALID LOW LIMIT X OF FITTING REGION");
   if (p->xmax >= sizex || p->xmax < p->xmin)
      return ("INVALID HIGH LIMIT X OF FITTING REGION");
   if (p->ymin < 0 || p->ymin > p->ymax)
      return ("INVALID LOW LIMIT Y OF FITTING REGION");
   if (p->ymax >= sizey || p->ymax < p->ymin)
      return ("INVALID HIGH LIMIT Y OF FITTING REGION");
   double *working_space = new double[5 * (7 * p->number_of_peaks + 14)];
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      if (p->amp_init[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i] = p->amp_init[i];	//vector parameter
      if (p->fix_amp[i] == false) {
         working_space[shift + j] = p->amp_init[i];	//vector xk
         j += 1;
      }
      if (p->position_init_x[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_x[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 1] = p->position_init_x[i];	//vector parameter
      if (p->fix_position_x[i] == false) {
         working_space[shift + j] = p->position_init_x[i];	//vector xk
         j += 1;
      }
      if (p->position_init_y[i] < p->ymin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_y[i] > p->ymax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 2] = p->position_init_y[i];	//vector parameter
      if (p->fix_position_y[i] == false) {
         working_space[shift + j] = p->position_init_y[i];	//vector xk
         j += 1;
      }
      if (p->amp_init_x1[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i + 3] = p->amp_init_x1[i];	//vector parameter
      if (p->fix_amp_x1[i] == false) {
         working_space[shift + j] = p->amp_init_x1[i];	//vector xk
         j += 1;
      }
      if (p->amp_init_y1[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i + 4] = p->amp_init_y1[i];	//vector parameter
      if (p->fix_amp_y1[i] == false) {
         working_space[shift + j] = p->amp_init_y1[i];	//vector xk
         j += 1;
      }
      if (p->position_init_x1[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_x1[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 5] = p->position_init_x1[i];	//vector parameter
      if (p->fix_position_x1[i] == false) {
         working_space[shift + j] = p->position_init_x1[i];	//vector xk
         j += 1;
      }
      if (p->position_init_y1[i] < p->ymin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_y1[i] > p->ymax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 6] = p->position_init_y1[i];	//vector parameter
      if (p->fix_position_y1[i] == false) {
         working_space[shift + j] = p->position_init_y1[i];	//vector xk
         j += 1;
      }
   }
   peak_vel = 7 * i;
   if (p->sigma_init_x < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[7 * i] = p->sigma_init_x;	//vector parameter
   if (p->fix_sigma_x == false) {
      working_space[shift + j] = p->sigma_init_x;	//vector xk
      j += 1;
   }
   if (p->sigma_init_y < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[7 * i + 1] = p->sigma_init_y;	//vector parameter
   if (p->fix_sigma_y == false) {
      working_space[shift + j] = p->sigma_init_y;	//vector xk
      j += 1;
   }
   if (p->ro_init < -1 || p->ro_init > 1)
      return ("INITIAL VALUE OF RO MUST BE FROM REGION <-1,1>");
   working_space[7 * i + 2] = p->ro_init;	//vector parameter
   if (p->fix_ro == false) {
      working_space[shift + j] = p->ro_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 3] = p->a0_init;	//vector parameter
   if (p->fix_a0 == false) {
      working_space[shift + j] = p->a0_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 4] = p->ax_init;	//vector parameter
   if (p->fix_ax == false) {
      working_space[shift + j] = p->ax_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 5] = p->ay_init;	//vector parameter
   if (p->fix_ay == false) {
      working_space[shift + j] = p->ay_init;	//vector xk
      j += 1;
   }
   if (p->txy_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 6] = p->txy_init;	//vector parameter
   if (p->fix_txy == false) {
      working_space[shift + j] = p->txy_init;	//vector xk
      j += 1;
   }
   if (p->sxy_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 7] = p->sxy_init;	//vector parameter
   if (p->fix_sxy == false) {
      working_space[shift + j] = p->sxy_init;	//vector xk
      j += 1;
   }
   if (p->tx_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 8] = p->tx_init;	//vector parameter
   if (p->fix_tx == false) {
      working_space[shift + j] = p->tx_init;	//vector xk
      j += 1;
   }
   if (p->ty_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 9] = p->ty_init;	//vector parameter
   if (p->fix_ty == false) {
      working_space[shift + j] = p->ty_init;	//vector xk
      j += 1;
   }
   if (p->sx_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 10] = p->sxy_init;	//vector parameter
   if (p->fix_sx == false) {
      working_space[shift + j] = p->sx_init;	//vector xk
      j += 1;
   }
   if (p->sy_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 11] = p->sy_init;	//vector parameter
   if (p->fix_sy == false) {
      working_space[shift + j] = p->sy_init;	//vector xk
      j += 1;
   }
   if (p->bx_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[7 * i + 12] = p->bx_init;	//vector parameter
   if (p->fix_bx == false) {
      working_space[shift + j] = p->bx_init;	//vector xk
      j += 1;
   }
   if (p->by_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[7 * i + 13] = p->by_init;	//vector parameter
   if (p->fix_by == false) {
      working_space[shift + j] = p->by_init;	//vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0)
      return ("ALL PARAMETERS ARE FIXED");
   if (rozmer >= (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1))
      return
          ("NUMBER OF FITTED PARAMETERS IS LARGER THAN # OF FITTED POINTS");
   for (iter = 0; iter < p->number_of_iterations; iter++) {
      for (j = 0; j < rozmer; j++) {
         working_space[2 * shift + j] = 0, working_space[3 * shift + j] = 0;	//der,temp
      }
      
          //filling vectors
          alpha = p->alpha;
      chi_opt = 0, pw = p->power - 2;
      for (i1 = p->xmin; i1 <= p->xmax; i1++) {
         for (i2 = p->ymin; i2 <= p->ymax; i2++) {
            yw = source[i1][i2];
            ywm = yw;
            f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
               if (f > 0.00001)
                  chi_opt += yw * TMath::Log(f) - f;
            }
            
            else {
               if (ywm != 0)
                  chi_opt += (yw - f) * (yw - f) / ywm;
            }
            if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
               ywm = f;
               if (f < 0.00001)
                  ywm = 0.00001;
            }
            
            else if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
               ywm = f;
               if (f < 0.00001)
                  ywm = 0.00001;
            }
            
            else {
               if (ywm == 0)
                  ywm = 1;
            }
            
                //calculation of gradient vector
                for (j = 0, k = 0; j < p->number_of_peaks; j++) {
               if (p->fix_amp[j] == false) {
                  a = Deramp2((double) i1, (double) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
               if (p->fix_position_x[j] == false) {
                  a = Deri02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                     d = Derderi02((double) i1, (double) i2,
                                    working_space[7 * j],
                                    working_space[7 * j + 1],
                                    working_space[7 * j + 2],
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if ((a + d) <= 0 && a >= 0 || (a + d) >= 0
                             && a <= 0)
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
               if (p->fix_position_y[j] == false) {
                  a = Derj02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                     d = Derderj02((double) i1, (double) i2,
                                    working_space[7 * j],
                                    working_space[7 * j + 1],
                                    working_space[7 * j + 2],
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if ((a + d) <= 0 && a >= 0 || (a + d) >= 0
                             && a <= 0)
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
               if (p->fix_amp_x1[j] == false) {
                  a = Derampx((double) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
               if (p->fix_amp_y1[j] == false) {
                  a = Derampx((double) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
               if (p->fix_position_x1[j] == false) {
                  a = Deri01((double) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                     d = Derderi01((double) i1, working_space[7 * j + 3],
                                    working_space[7 * j + 5],
                                    working_space[peak_vel]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if ((a + d) <= 0 && a >= 0 || (a + d) >= 0
                             && a <= 0)
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c;	//Der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c;	//temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c;	//Der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c;	//temp
                     }
                  }
                  k += 1;
               }
               if (p->fix_position_y1[j] == false) {
                  a = Deri01((double) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                     d = Derderi01((double) i2, working_space[7 * j + 4],
                                    working_space[7 * j + 6],
                                    working_space[peak_vel + 1]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if ((a + d) <= 0 && a >= 0 || (a + d) >= 0
                             && a <= 0)
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_sigma_x == false) {
               a = Dersigmax(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                  d = Derdersigmax(p->number_of_peaks, (double) i1,
                                    (double) i2, working_space,
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (TMath::Abs(a) > 0.00000001
                       && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                     d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                     if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                        d = 0;
                  }
                  
                  else
                     d = 0;
                  a = a + d;
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_sigma_y == false) {
               a = Dersigmay(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND)
                  d = Derdersigmay(p->number_of_peaks, (double) i1,
                                    (double) i2, working_space,
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (TMath::Abs(a) > 0.00000001
                       && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                     d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                     if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                        d = 0;
                  }
                  
                  else
                     d = 0;
                  a = a + d;
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_ro == false) {
               a = Derro(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (TMath::Abs(a) > 0.00000001
                       && p->fit_taylor == FIT2_TAYLOR_ORDER_SECOND) {
                     d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                     if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                        d = 0;
                  }
                  
                  else
                     d = 0;
                  a = a + d;
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_ax == false) {
               a = i1;
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_ay == false) {
               a = i2;
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_txy == false) {
               a = Dertxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_sxy == false) {
               a = Dersxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_tx == false) {
               a = Dertx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_ty == false) {
               a = Derty(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_sx == false) {
               a = Dersx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_sy == false) {
               a = Dersy(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_bx == false) {
               a = Derbx(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
            if (p->fix_by == false) {
               a = Derby(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
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
         if (p->alpha_optim == FIT2_ALPHA_OPTIMAL) {
            if (p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD)
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
                     working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_x1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_y1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x1[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y1[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma_x == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_sigma_y == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_ro == false) {
                  if (working_space[shift + j] < -1) {	//xk[j]
                     working_space[shift + j] = -1;	//xk[j]
                  }
                  if (working_space[shift + j] > 1) {	//xk[j]
                     working_space[shift + j] = 1;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_ax == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_ay == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_txy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (p->fix_sxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (p->fix_tx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (p->fix_ty == false) {
                  working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (p->fix_sx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (p->fix_sy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (p->fix_bx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (p->fix_by == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i1 = p->xmin; i1 <= p->xmax; i1++) {
                  for (i2 = p->ymin; i2 <= p->ymax; i2++) {
                     yw = source[i1][i2];
                     ywm = yw;
                     f = Shape2(p->number_of_peaks, (double) i1,
                                 (double) i2, working_space,
                                 working_space[peak_vel],
                                 working_space[peak_vel + 1],
                                 working_space[peak_vel + 2],
                                 working_space[peak_vel + 3],
                                 working_space[peak_vel + 4],
                                 working_space[peak_vel + 5],
                                 working_space[peak_vel + 6],
                                 working_space[peak_vel + 7],
                                 working_space[peak_vel + 8],
                                 working_space[peak_vel + 9],
                                 working_space[peak_vel + 10],
                                 working_space[peak_vel + 11],
                                 working_space[peak_vel + 12],
                                 working_space[peak_vel + 13]);
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
                        ywm = f;
                        if (f < 0.00001)
                           ywm = 0.00001;
                     }
                     if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
                        if (f > 0.00001)
                           chi2 += yw * TMath::Log(f) - f;
                     }
                     
                     else {
                        if (ywm != 0)
                           chi2 += (yw - f) * (yw - f) / ywm;
                     }
                  }
               }
               if (chi2 < chi_min
                    && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD
                    || chi2 > chi_min
                    && p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
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
                     working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_x1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_y1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x1[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y1[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma_x == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_sigma_y == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_ro == false) {
                  if (working_space[shift + j] < -1) {	//xk[j]
                     working_space[shift + j] = -1;	//xk[j]
                  }
                  if (working_space[shift + j] > 1) {	//xk[j]
                     working_space[shift + j] = 1;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_ax == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_ay == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_txy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (p->fix_sxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (p->fix_tx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (p->fix_ty == false) {
                  working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (p->fix_sx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (p->fix_sy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (p->fix_bx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (p->fix_by == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
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
                  working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                  j += 1;
               }
               if (p->fix_position_x[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                  j += 1;
               }
               if (p->fix_position_y[i] == false) {
                  if (working_space[shift + j] < p->ymin)	//xk[j]
                     working_space[shift + j] = p->ymin;	//xk[j]
                  if (working_space[shift + j] > p->ymax)	//xk[j]
                     working_space[shift + j] = p->ymax;	//xk[j]
                  working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                  j += 1;
               }
               if (p->fix_amp_x1[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                  j += 1;
               }
               if (p->fix_amp_y1[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                  j += 1;
               }
               if (p->fix_position_x1[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                  j += 1;
               }
               if (p->fix_position_y1[i] == false) {
                  if (working_space[shift + j] < p->ymin)	//xk[j]
                     working_space[shift + j] = p->ymin;	//xk[j]
                  if (working_space[shift + j] > p->ymax)	//xk[j]
                     working_space[shift + j] = p->ymax;	//xk[j]
                  working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                  j += 1;
               }
            }
            if (p->fix_sigma_x == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (p->fix_sigma_y == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (p->fix_ro == false) {
               if (working_space[shift + j] < -1) {	//xk[j]
                  working_space[shift + j] = -1;	//xk[j]
               }
               if (working_space[shift + j] > 1) {	//xk[j]
                  working_space[shift + j] = 1;	//xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (p->fix_a0 == false) {
               working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (p->fix_ax == false) {
               working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (p->fix_ay == false) {
               working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (p->fix_txy == false) {
               working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            if (p->fix_sxy == false) {
               working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
               j += 1;
            }
            if (p->fix_tx == false) {
               working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
               j += 1;
            }
            if (p->fix_ty == false) {
               working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
               j += 1;
            }
            if (p->fix_sx == false) {
               working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
               j += 1;
            }
            if (p->fix_sy == false) {
               working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
               j += 1;
            }
            if (p->fix_bx == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
               j += 1;
            }
            if (p->fix_by == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i1 = p->xmin; i1 <= p->xmax; i1++) {
               for (i2 = p->ymin; i2 <= p->ymax; i2++) {
                  yw = source[i1][i2];
                  ywm = yw;
                  f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 3],
                              working_space[peak_vel + 4],
                              working_space[peak_vel + 5],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
                     if (f > 0.00001)
                        chi += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi += (yw - f) * (yw - f) / ywm;
                  }
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (p->alpha_optim == FIT2_ALPHA_HALVING && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (p->alpha_optim == FIT2_ALPHA_OPTIMAL)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD
                 || chi < chi_opt
                 && p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD)
                && regul_cycle < FIT2_NUM_OF_REGUL_CYCLES);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;	//temp_xk[j]
         working_space[2 * shift + j] = 0;	//der[j]
      }
      for (i1 = p->xmin, chi_cel = 0; i1 <= p->xmax; i1++) {
         for (i2 = p->ymin; i2 <= p->ymax; i2++) {
            yw = source[i1][i2];
            if (yw == 0)
               yw = 1;
            f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            chi_opt = (yw - f) * (yw - f) / yw;
            chi_cel += (yw - f) * (yw - f) / yw;
            
                //calculate gradient vector
                for (j = 0, k = 0; j < p->number_of_peaks; j++) {
               if (p->fix_amp[j] == false) {
                  a = Deramp2((double) i1, (double) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_x[j] == false) {
                  a = Deri02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_y[j] == false) {
                  a = Derj02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_amp_x1[j] == false) {
                  a = Derampx((double) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_amp_y1[j] == false) {
                  a = Derampx((double) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_x1[j] == false) {
                  a = Deri01((double) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_y1[j] == false) {
                  a = Deri01((double) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c;	//temp_xk[k]
                  }
                  k += 1;
               }
            }
            if (p->fix_sigma_x == false) {
               a = Dersigmax(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sigma_y == false) {
               a = Dersigmay(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ro == false) {
               a = Derro(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_a0 == false) {
               a = 1.;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ax == false) {
               a = i1;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ay == false) {
               a = i2;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_txy == false) {
               a = Dertxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sxy == false) {
               a = Dersxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_tx == false) {
               a = Dertx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ty == false) {
               a = Derty(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sx == false) {
               a = Dersx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sy == false) {
               a = Dersy(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_bx == false) {
               a = Derbx(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_by == false) {
               a = Derby(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
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
   }
   b = (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1) - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      p->volume[i] =
          Volume(working_space[7 * i], working_space[peak_vel],
                 working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (p->volume[i] > 0) {
         c = 0;
         if (p->fix_amp[i] == false) {
            a = Derpa2(working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_sigma_x == false) {
            a = Derpsigmax(working_space[shift + j],
                            working_space[peak_vel + 1],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_sigma_y == false) {
            a = Derpsigmay(working_space[shift + j],
                            working_space[peak_vel],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 1];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_ro == false) {
            a = Derpro(working_space[shift + j], working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 2];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         p->volume_err[i] = TMath::Sqrt(TMath::Abs(chi_er * c));
      }
      
      else {
         p->volume_err[i] = 0;
      }
      if (p->fix_amp[i] == false) {
         p->amp_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc[i] = p->amp_init[i];
         p->amp_err[i] = 0;
      }
      if (p->fix_position_x[i] == false) {
         p->position_calc_x[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_x[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_x[i] = p->position_init_x[i];
         p->position_err_x[i] = 0;
      }
      if (p->fix_position_y[i] == false) {
         p->position_calc_y[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_y[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_y[i] = p->position_init_y[i];
         p->position_err_y[i] = 0;
      }
      if (p->fix_amp_x1[i] == false) {
         p->amp_calc_x1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err_x1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc_x1[i] = p->amp_init_x1[i];
         p->amp_err_x1[i] = 0;
      }
      if (p->fix_amp_y1[i] == false) {
         p->amp_calc_y1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err_y1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc_y1[i] = p->amp_init_y1[i];
         p->amp_err_y1[i] = 0;
      }
      if (p->fix_position_x1[i] == false) {
         p->position_calc_x1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_x1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_x1[i] = p->position_init_x1[i];
         p->position_err_x1[i] = 0;
      }
      if (p->fix_position_y1[i] == false) {
         p->position_calc_y1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_y1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_y1[i] = p->position_init_y1[i];
         p->position_err_y1[i] = 0;
      }
   }
   if (p->fix_sigma_x == false) {
      p->sigma_calc_x = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err_x = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc_x = p->sigma_init_x;
      p->sigma_err_x = 0;
   }
   if (p->fix_sigma_y == false) {
      p->sigma_calc_y = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err_y = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc_y = p->sigma_init_y;
      p->sigma_err_y = 0;
   }
   if (p->fix_ro == false) {
      p->ro_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ro_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ro_calc = p->ro_init;
      p->ro_err = 0;
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
   if (p->fix_ax == false) {
      p->ax_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ax_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ax_calc = p->ax_init;
      p->ax_err = 0;
   }
   if (p->fix_ay == false) {
      p->ay_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ay_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ay_calc = p->ay_init;
      p->ay_err = 0;
   }
   if (p->fix_txy == false) {
      p->txy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->txy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->txy_calc = p->txy_init;
      p->txy_err = 0;
   }
   if (p->fix_sxy == false) {
      p->sxy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sxy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sxy_calc = p->sxy_init;
      p->sxy_err = 0;
   }
   if (p->fix_tx == false) {
      p->tx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->tx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->tx_calc = p->tx_init;
      p->tx_err = 0;
   }
   if (p->fix_ty == false) {
      p->ty_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ty_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ty_calc = p->ty_init;
      p->ty_err = 0;
   }
   if (p->fix_sx == false) {
      p->sx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sx_calc = p->sx_init;
      p->sx_err = 0;
   }
   if (p->fix_sy == false) {
      p->sy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sy_calc = p->sy_init;
      p->sy_err = 0;
   }
   if (p->fix_bx == false) {
      p->bx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->bx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->bx_calc = p->bx_init;
      p->bx_err = 0;
   }
   if (p->fix_by == false) {
      p->by_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->by_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->by_calc = p->by_init;
      p->by_err = 0;
   }
   b = (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1) - rozmer;
   p->chi = chi_cel / b;
   for (i1 = p->xmin; i1 <= p->xmax; i1++) {
      for (i2 = p->ymin; i2 <= p->ymax; i2++) {
         f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                     working_space, working_space[peak_vel],
                     working_space[peak_vel + 1],
                     working_space[peak_vel + 2],
                     working_space[peak_vel + 3],
                     working_space[peak_vel + 4],
                     working_space[peak_vel + 5],
                     working_space[peak_vel + 6],
                     working_space[peak_vel + 7],
                     working_space[peak_vel + 8],
                     working_space[peak_vel + 9],
                     working_space[peak_vel + 10],
                     working_space[peak_vel + 11],
                     working_space[peak_vel + 12],
                     working_space[peak_vel + 13]);
         source[i1][i2] = f;
   } } delete[]working_space;
   return 0;
}


// ____________________________________________________________________________________________________________________________
const char *TSpectrum2::Fit2Stiefel(float **source,
                                     TSpectrumTwoDimFit * p, int sizex,
                                     int sizey) 
{
   
/////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL FIT FUNCTION				           */ 
/*      Algorithm with matrix inversion (Stiefel-Hestens inversion)        */ 
/*	This function fits the source spectrum. The calling program should */ 
/*      fill in input parameters of the TSpectrumTwoDimFit class 	   */ 
/*	The fitted parameters are written into class pointed by 	   */ 
/*	TSpectrumTwoDimFit structure pointer and fitted data are written   */ 
/*      back into source spectrum.                                         */ 
/*									   */ 
/*	Function parameters:						   */ 
/*	source-pointer to the matrix of source spectrum			   */ 
/*	p-pointer to the TSpectrumTwoDimFit class, see manual          	   */ 
/*	sizex-length x of source spectrum                                  */ 
/*	sizey-length y of source spectrum                                  */ 
/*									   */ 
/////////////////////////////////////////////////////////////////////////////
   int i, i1, i2, j, k, shift =
       7 * p->number_of_peaks + 14, peak_vel, rozmer, iter, regul_cycle,
       flag;
   double a, b, c, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   if (p->number_of_peaks <= 0)
      return ("INVALID NUMBER OF PEAKS, MUST BE POSITIVE");
   if (p->number_of_iterations <= 0)
      return ("INVALID NUMBER OF ITERATIONS, MUST BE POSITIVE");
   if (p->alpha <= 0 || p->alpha > 1)
      return ("INVALID COEFFICIENT ALPHA, MUST BE > THAN 0 AND <=1");
   if (p->statistic_type != FIT2_OPTIM_CHI_COUNTS
        && p->statistic_type != FIT2_OPTIM_CHI_FUNC_VALUES
        && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD)
      return ("WRONG TYPE OF STATISTIC");
   if (p->alpha_optim != FIT2_ALPHA_HALVING
        && p->alpha_optim != FIT2_ALPHA_OPTIMAL)
      return ("WRONG OPTIMIZATION ALGORITHM");
   if (p->xmin < 0 || p->xmin > p->xmax)
      return ("INVALID LOW LIMIT X OF FITTING REGION");
   if (p->xmax >= sizex || p->xmax < p->xmin)
      return ("INVALID HIGH LIMIT X OF FITTING REGION");
   if (p->ymin < 0 || p->ymin > p->ymax)
      return ("INVALID LOW LIMIT Y OF FITTING REGION");
   if (p->ymax >= sizey || p->ymax < p->ymin)
      return ("INVALID HIGH LIMIT Y OF FITTING REGION");
   double *working_space = new double[5 * (7 * p->number_of_peaks + 14)];
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      if (p->amp_init[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i] = p->amp_init[i];	//vector parameter
      if (p->fix_amp[i] == false) {
         working_space[shift + j] = p->amp_init[i];	//vector xk
         j += 1;
      }
      if (p->position_init_x[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_x[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 1] = p->position_init_x[i];	//vector parameter
      if (p->fix_position_x[i] == false) {
         working_space[shift + j] = p->position_init_x[i];	//vector xk
         j += 1;
      }
      if (p->position_init_y[i] < p->ymin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_y[i] > p->ymax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 2] = p->position_init_y[i];	//vector parameter
      if (p->fix_position_y[i] == false) {
         working_space[shift + j] = p->position_init_y[i];	//vector xk
         j += 1;
      }
      if (p->amp_init_x1[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i + 3] = p->amp_init_x1[i];	//vector parameter
      if (p->fix_amp_x1[i] == false) {
         working_space[shift + j] = p->amp_init_x1[i];	//vector xk
         j += 1;
      }
      if (p->amp_init_y1[i] < 0)
         return ("INITIAL VALUE OF AMPLITUDE MUST BE NONNEGATIVE");
      working_space[7 * i + 4] = p->amp_init_y1[i];	//vector parameter
      if (p->fix_amp_y1[i] == false) {
         working_space[shift + j] = p->amp_init_y1[i];	//vector xk
         j += 1;
      }
      if (p->position_init_x1[i] < p->xmin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_x1[i] > p->xmax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 5] = p->position_init_x1[i];	//vector parameter
      if (p->fix_position_x1[i] == false) {
         working_space[shift + j] = p->position_init_x1[i];	//vector xk
         j += 1;
      }
      if (p->position_init_y1[i] < p->ymin)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      if (p->position_init_y1[i] > p->ymax)
         return
             ("INITIAL VALUE OF POSITION MUST BE WITHIN FITTING REGION");
      working_space[7 * i + 6] = p->position_init_y1[i];	//vector parameter
      if (p->fix_position_y1[i] == false) {
         working_space[shift + j] = p->position_init_y1[i];	//vector xk
         j += 1;
      }
   }
   peak_vel = 7 * i;
   if (p->sigma_init_x < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[7 * i] = p->sigma_init_x;	//vector parameter
   if (p->fix_sigma_x == false) {
      working_space[shift + j] = p->sigma_init_x;	//vector xk
      j += 1;
   }
   if (p->sigma_init_y < 0)
      return ("INITIAL VALUE OF SIGMA MUST BE NONNEGATIVE");
   working_space[7 * i + 1] = p->sigma_init_y;	//vector parameter
   if (p->fix_sigma_y == false) {
      working_space[shift + j] = p->sigma_init_y;	//vector xk
      j += 1;
   }
   if (p->ro_init < -1 || p->ro_init > 1)
      return ("INITIAL VALUE OF RO MUST BE FROM REGION <-1,1>");
   working_space[7 * i + 2] = p->ro_init;	//vector parameter
   if (p->fix_ro == false) {
      working_space[shift + j] = p->ro_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 3] = p->a0_init;	//vector parameter
   if (p->fix_a0 == false) {
      working_space[shift + j] = p->a0_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 4] = p->ax_init;	//vector parameter
   if (p->fix_ax == false) {
      working_space[shift + j] = p->ax_init;	//vector xk
      j += 1;
   }
   working_space[7 * i + 5] = p->ay_init;	//vector parameter
   if (p->fix_ay == false) {
      working_space[shift + j] = p->ay_init;	//vector xk
      j += 1;
   }
   if (p->txy_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 6] = p->txy_init;	//vector parameter
   if (p->fix_txy == false) {
      working_space[shift + j] = p->txy_init;	//vector xk
      j += 1;
   }
   if (p->sxy_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 7] = p->sxy_init;	//vector parameter
   if (p->fix_sxy == false) {
      working_space[shift + j] = p->sxy_init;	//vector xk
      j += 1;
   }
   if (p->tx_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 8] = p->tx_init;	//vector parameter
   if (p->fix_tx == false) {
      working_space[shift + j] = p->tx_init;	//vector xk
      j += 1;
   }
   if (p->ty_init < 0)
      return ("INITIAL VALUE OF T MUST BE NONNEGATIVE");
   working_space[7 * i + 9] = p->ty_init;	//vector parameter
   if (p->fix_ty == false) {
      working_space[shift + j] = p->ty_init;	//vector xk
      j += 1;
   }
   if (p->sx_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 10] = p->sxy_init;	//vector parameter
   if (p->fix_sx == false) {
      working_space[shift + j] = p->sx_init;	//vector xk
      j += 1;
   }
   if (p->sy_init < 0)
      return ("INITIAL VALUE OF S MUST BE NONNEGATIVE");
   working_space[7 * i + 11] = p->sy_init;	//vector parameter
   if (p->fix_sy == false) {
      working_space[shift + j] = p->sy_init;	//vector xk
      j += 1;
   }
   if (p->bx_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[7 * i + 12] = p->bx_init;	//vector parameter
   if (p->fix_bx == false) {
      working_space[shift + j] = p->bx_init;	//vector xk
      j += 1;
   }
   if (p->by_init <= 0)
      return ("INITIAL VALUE OF B MUST BE POSITIVE");
   working_space[7 * i + 13] = p->by_init;	//vector parameter
   if (p->fix_by == false) {
      working_space[shift + j] = p->by_init;	//vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0)
      return ("ALL PARAMETERS ARE FIXED");
   if (rozmer >= (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1))
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
      for (i1 = p->xmin; i1 <= p->xmax; i1++) {
         for (i2 = p->ymin; i2 <= p->ymax; i2++) {
            
                //calculation of gradient vector
                for (j = 0, k = 0; j < p->number_of_peaks; j++) {
               if (p->fix_amp[j] == false) {
                  working_space[2 * shift + k] =
                      Deramp2((double) i1, (double) i2,
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  k += 1;
               }
               if (p->fix_position_x[j] == false) {
                  working_space[2 * shift + k] =
                      Deri02((double) i1, (double) i2,
                             working_space[7 * j],
                             working_space[7 * j + 1],
                             working_space[7 * j + 2],
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
               if (p->fix_position_y[j] == false) {
                  working_space[2 * shift + k] =
                      Derj02((double) i1, (double) i2,
                             working_space[7 * j],
                             working_space[7 * j + 1],
                             working_space[7 * j + 2],
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
               if (p->fix_amp_x1[j] == false) {
                  working_space[2 * shift + k] =
                      Derampx((double) i1, working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  k += 1;
               }
               if (p->fix_amp_y1[j] == false) {
                  working_space[2 * shift + k] =
                      Derampx((double) i2, working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  k += 1;
               }
               if (p->fix_position_x1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01((double) i1, working_space[7 * j + 3],
                             working_space[7 * j + 5],
                             working_space[peak_vel],
                             working_space[peak_vel + 8],
                             working_space[peak_vel + 10],
                             working_space[peak_vel + 12]);
                  k += 1;
               }
               if (p->fix_position_y1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01((double) i2, working_space[7 * j + 4],
                             working_space[7 * j + 6],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 9],
                             working_space[peak_vel + 11],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
            } if (p->fix_sigma_x == false) {
               working_space[2 * shift + k] =
                   Dersigmax(p->number_of_peaks, (double) i1, (double) i2,
                             working_space, working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 8],
                             working_space[peak_vel + 10],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
               k += 1;
            }
            if (p->fix_sigma_y == false) {
               working_space[2 * shift + k] =
                   Dersigmay(p->number_of_peaks, (double) i1, (double) i2,
                             working_space, working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 9],
                             working_space[peak_vel + 11],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
               k += 1;
            }
            if (p->fix_ro == false) {
               working_space[2 * shift + k] =
                   Derro(p->number_of_peaks, (double) i1, (double) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 2]);
               k += 1;
            }
            if (p->fix_a0 == false) {
               working_space[2 * shift + k] = 1.;
               k += 1;
            }
            if (p->fix_ax == false) {
               working_space[2 * shift + k] = i1;
               k += 1;
            }
            if (p->fix_ay == false) {
               working_space[2 * shift + k] = i2;
               k += 1;
            }
            if (p->fix_txy == false) {
               working_space[2 * shift + k] =
                   Dertxy(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               k += 1;
            }
            if (p->fix_sxy == false) {
               working_space[2 * shift + k] =
                   Dersxy(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1]);
               k += 1;
            }
            if (p->fix_tx == false) {
               working_space[2 * shift + k] =
                   Dertx(p->number_of_peaks, (double) i1, working_space,
                         working_space[peak_vel],
                         working_space[peak_vel + 12]);
               k += 1;
            }
            if (p->fix_ty == false) {
               working_space[2 * shift + k] =
                   Derty(p->number_of_peaks, (double) i2, working_space,
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            if (p->fix_sx == false) {
               working_space[2 * shift + k] =
                   Dersx(p->number_of_peaks, (double) i1, working_space,
                         working_space[peak_vel]);
               k += 1;
            }
            if (p->fix_sy == false) {
               working_space[2 * shift + k] =
                   Dersy(p->number_of_peaks, (double) i2, working_space,
                         working_space[peak_vel + 1]);
               k += 1;
            }
            if (p->fix_bx == false) {
               working_space[2 * shift + k] =
                   Derbx(p->number_of_peaks, (double) i1, (double) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 6],
                         working_space[peak_vel + 8],
                         working_space[peak_vel + 12],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            if (p->fix_by == false) {
               working_space[2 * shift + k] =
                   Derby(p->number_of_peaks, (double) i1, (double) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 6],
                         working_space[peak_vel + 8],
                         working_space[peak_vel + 12],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            yw = source[i1][i2];
            ywm = yw;
            f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
               if (f > 0.00001)
                  chi_opt += yw * TMath::Log(f) - f;
            }
            
            else {
               if (ywm != 0)
                  chi_opt += (yw - f) * (yw - f) / ywm;
            }
            if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
               ywm = f;
               if (f < 0.00001)
                  ywm = 0.00001;
            }
            
            else if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
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
                                     j] * working_space[2 * shift +
                                                        k] / ywm;
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES)
                     b = b * (4 * yw - 2 * f) / ywm;
                  working_matrix[j][k] += b;
                  if (j == k)
                     working_space[3 * shift + j] += b;
               }
            }
            if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES)
               b = (f * f - yw * yw) / (ywm * ywm);
            
            else
               b = (f - yw) / ywm;
            for (j = 0; j < rozmer; j++) {
               working_matrix[j][rozmer] -=
                   b * working_space[2 * shift + j];
            }
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
         if (p->alpha_optim == FIT2_ALPHA_OPTIMAL) {
            if (p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD)
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
                     working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_x1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_y1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x1[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y1[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma_x == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_sigma_y == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_ro == false) {
                  if (working_space[shift + j] < -1) {	//xk[j]
                     working_space[shift + j] = -1;	//xk[j]
                  }
                  if (working_space[shift + j] > 1) {	//xk[j]
                     working_space[shift + j] = 1;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_ax == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_ay == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_txy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (p->fix_sxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (p->fix_tx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (p->fix_ty == false) {
                  working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (p->fix_sx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (p->fix_sy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (p->fix_bx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (p->fix_by == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i1 = p->xmin; i1 <= p->xmax; i1++) {
                  for (i2 = p->ymin; i2 <= p->ymax; i2++) {
                     yw = source[i1][i2];
                     ywm = yw;
                     f = Shape2(p->number_of_peaks, (double) i1,
                                 (double) i2, working_space,
                                 working_space[peak_vel],
                                 working_space[peak_vel + 1],
                                 working_space[peak_vel + 2],
                                 working_space[peak_vel + 3],
                                 working_space[peak_vel + 4],
                                 working_space[peak_vel + 5],
                                 working_space[peak_vel + 6],
                                 working_space[peak_vel + 7],
                                 working_space[peak_vel + 8],
                                 working_space[peak_vel + 9],
                                 working_space[peak_vel + 10],
                                 working_space[peak_vel + 11],
                                 working_space[peak_vel + 12],
                                 working_space[peak_vel + 13]);
                     if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
                        ywm = f;
                        if (f < 0.00001)
                           ywm = 0.00001;
                     }
                     if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
                        if (f > 0.00001)
                           chi2 += yw * TMath::Log(f) - f;
                     }
                     
                     else {
                        if (ywm != 0)
                           chi2 += (yw - f) * (yw - f) / ywm;
                     }
                  }
               }
               if (chi2 < chi_min
                    && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD
                    || chi2 > chi_min
                    && p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
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
                     working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_x1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (p->fix_amp_y1[i] == false) {
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = 0;	//xk[j]
                     working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_x1[i] == false) {
                     if (working_space[shift + j] < p->xmin)	//xk[j]
                        working_space[shift + j] = p->xmin;	//xk[j]
                     if (working_space[shift + j] > p->xmax)	//xk[j]
                        working_space[shift + j] = p->xmax;	//xk[j]
                     working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (p->fix_position_y1[i] == false) {
                     if (working_space[shift + j] < p->ymin)	//xk[j]
                        working_space[shift + j] = p->ymin;	//xk[j]
                     if (working_space[shift + j] > p->ymax)	//xk[j]
                        working_space[shift + j] = p->ymax;	//xk[j]
                     working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (p->fix_sigma_x == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (p->fix_sigma_y == false) {
                  if (working_space[shift + j] < 0.001) {	//xk[j]
                     working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (p->fix_ro == false) {
                  if (working_space[shift + j] < -1) {	//xk[j]
                     working_space[shift + j] = -1;	//xk[j]
                  }
                  if (working_space[shift + j] > 1) {	//xk[j]
                     working_space[shift + j] = 1;	//xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (p->fix_a0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (p->fix_ax == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (p->fix_ay == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (p->fix_txy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (p->fix_sxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (p->fix_tx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (p->fix_ty == false) {
                  working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (p->fix_sx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (p->fix_sy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (p->fix_bx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (p->fix_by == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                     if (working_space[shift + j] < 0)	//xk[j]
                        working_space[shift + j] = -0.001;	//xk[j]
                     else
                        working_space[shift + j] = 0.001;	//xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
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
                  working_space[7 * i] = working_space[shift + j];	//parameter[7*i]=xk[j]
                  j += 1;
               }
               if (p->fix_position_x[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[7 * i + 1] = working_space[shift + j];	//parameter[7*i+1]=xk[j]
                  j += 1;
               }
               if (p->fix_position_y[i] == false) {
                  if (working_space[shift + j] < p->ymin)	//xk[j]
                     working_space[shift + j] = p->ymin;	//xk[j]
                  if (working_space[shift + j] > p->ymax)	//xk[j]
                     working_space[shift + j] = p->ymax;	//xk[j]
                  working_space[7 * i + 2] = working_space[shift + j];	//parameter[7*i+2]=xk[j]
                  j += 1;
               }
               if (p->fix_amp_x1[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[7 * i + 3] = working_space[shift + j];	//parameter[7*i+3]=xk[j]
                  j += 1;
               }
               if (p->fix_amp_y1[i] == false) {
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = 0;	//xk[j]
                  working_space[7 * i + 4] = working_space[shift + j];	//parameter[7*i+4]=xk[j]
                  j += 1;
               }
               if (p->fix_position_x1[i] == false) {
                  if (working_space[shift + j] < p->xmin)	//xk[j]
                     working_space[shift + j] = p->xmin;	//xk[j]
                  if (working_space[shift + j] > p->xmax)	//xk[j]
                     working_space[shift + j] = p->xmax;	//xk[j]
                  working_space[7 * i + 5] = working_space[shift + j];	//parameter[7*i+5]=xk[j]
                  j += 1;
               }
               if (p->fix_position_y1[i] == false) {
                  if (working_space[shift + j] < p->ymin)	//xk[j]
                     working_space[shift + j] = p->ymin;	//xk[j]
                  if (working_space[shift + j] > p->ymax)	//xk[j]
                     working_space[shift + j] = p->ymax;	//xk[j]
                  working_space[7 * i + 6] = working_space[shift + j];	//parameter[7*i+6]=xk[j]
                  j += 1;
               }
            }
            if (p->fix_sigma_x == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];	//parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (p->fix_sigma_y == false) {
               if (working_space[shift + j] < 0.001) {	//xk[j]
                  working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 1] = working_space[shift + j];	//parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (p->fix_ro == false) {
               if (working_space[shift + j] < -1) {	//xk[j]
                  working_space[shift + j] = -1;	//xk[j]
               }
               if (working_space[shift + j] > 1) {	//xk[j]
                  working_space[shift + j] = 1;	//xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];	//parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (p->fix_a0 == false) {
               working_space[peak_vel + 3] = working_space[shift + j];	//parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (p->fix_ax == false) {
               working_space[peak_vel + 4] = working_space[shift + j];	//parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (p->fix_ay == false) {
               working_space[peak_vel + 5] = working_space[shift + j];	//parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (p->fix_txy == false) {
               working_space[peak_vel + 6] = working_space[shift + j];	//parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            if (p->fix_sxy == false) {
               working_space[peak_vel + 7] = working_space[shift + j];	//parameter[peak_vel+7]=xk[j]
               j += 1;
            }
            if (p->fix_tx == false) {
               working_space[peak_vel + 8] = working_space[shift + j];	//parameter[peak_vel+8]=xk[j]
               j += 1;
            }
            if (p->fix_ty == false) {
               working_space[peak_vel + 9] = working_space[shift + j];	//parameter[peak_vel+9]=xk[j]
               j += 1;
            }
            if (p->fix_sx == false) {
               working_space[peak_vel + 10] = working_space[shift + j];	//parameter[peak_vel+10]=xk[j]
               j += 1;
            }
            if (p->fix_sy == false) {
               working_space[peak_vel + 11] = working_space[shift + j];	//parameter[peak_vel+11]=xk[j]
               j += 1;
            }
            if (p->fix_bx == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 12] = working_space[shift + j];	//parameter[peak_vel+12]=xk[j]
               j += 1;
            }
            if (p->fix_by == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {	//xk[j]
                  if (working_space[shift + j] < 0)	//xk[j]
                     working_space[shift + j] = -0.001;	//xk[j]
                  else
                     working_space[shift + j] = 0.001;	//xk[j]
               }
               working_space[peak_vel + 13] = working_space[shift + j];	//parameter[peak_vel+13]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i1 = p->xmin; i1 <= p->xmax; i1++) {
               for (i2 = p->ymin; i2 <= p->ymax; i2++) {
                  yw = source[i1][i2];
                  ywm = yw;
                  f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 3],
                              working_space[peak_vel + 4],
                              working_space[peak_vel + 5],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (p->statistic_type == FIT2_OPTIM_CHI_FUNC_VALUES) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD) {
                     if (f > 0.00001)
                        chi += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi += (yw - f) * (yw - f) / ywm;
                  }
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (p->alpha_optim == FIT2_ALPHA_HALVING && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (p->alpha_optim == FIT2_ALPHA_OPTIMAL)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && p->statistic_type != FIT2_OPTIM_MAX_LIKELIHOOD
                 || chi < chi_opt
                 && p->statistic_type == FIT2_OPTIM_MAX_LIKELIHOOD)
                && regul_cycle < FIT2_NUM_OF_REGUL_CYCLES);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;	//temp_xk[j]
         working_space[2 * shift + j] = 0;	//der[j]
      }
      for (i1 = p->xmin, chi_cel = 0; i1 <= p->xmax; i1++) {
         for (i2 = p->ymin; i2 <= p->ymax; i2++) {
            yw = source[i1][i2];
            if (yw == 0)
               yw = 1;
            f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            chi_opt = (yw - f) * (yw - f) / yw;
            chi_cel += (yw - f) * (yw - f) / yw;
            
                //calculate gradient vector
                for (j = 0, k = 0; j < p->number_of_peaks; j++) {
               if (p->fix_amp[j] == false) {
                  a = Deramp2((double) i1, (double) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_x[j] == false) {
                  a = Deri02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_y[j] == false) {
                  a = Derj02((double) i1, (double) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_amp_x1[j] == false) {
                  a = Derampx((double) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_amp_y1[j] == false) {
                  a = Derampx((double) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_x1[j] == false) {
                  a = Deri01((double) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
               if (p->fix_position_y1[j] == false) {
                  a = Deri01((double) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt;	//der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b;	//temp_xk[k]
                  }
                  k += 1;
               }
            }
            if (p->fix_sigma_x == false) {
               a = Dersigmax(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sigma_y == false) {
               a = Dersigmay(p->number_of_peaks, (double) i1, (double) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ro == false) {
               a = Derro(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_a0 == false) {
               a = 1.;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ax == false) {
               a = i1;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ay == false) {
               a = i2;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_txy == false) {
               a = Dertxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sxy == false) {
               a = Dersxy(p->number_of_peaks, (double) i1, (double) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_tx == false) {
               a = Dertx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_ty == false) {
               a = Derty(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sx == false) {
               a = Dersx(p->number_of_peaks, (double) i1, working_space,
                          working_space[peak_vel]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_sy == false) {
               a = Dersy(p->number_of_peaks, (double) i2, working_space,
                          working_space[peak_vel + 1]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_bx == false) {
               a = Derbx(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
            if (p->fix_by == false) {
               a = Derby(p->number_of_peaks, (double) i1, (double) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;	//der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;	//temp_xk[k]
               }
               k += 1;
            }
         }
      }
   }
   b = (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1) - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < p->number_of_peaks; i++) {
      p->volume[i] =
          Volume(working_space[7 * i], working_space[peak_vel],
                 working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (p->volume[i] > 0) {
         c = 0;
         if (p->fix_amp[i] == false) {
            a = Derpa2(working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_sigma_x == false) {
            a = Derpsigmax(working_space[shift + j],
                            working_space[peak_vel + 1],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_sigma_y == false) {
            a = Derpsigmay(working_space[shift + j],
                            working_space[peak_vel],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 1];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (p->fix_ro == false) {
            a = Derpro(working_space[shift + j], working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 2];	//temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         p->volume_err[i] = TMath::Sqrt(TMath::Abs(chi_er * c));
      }
      
      else {
         p->volume_err[i] = 0;
      }
      if (p->fix_amp[i] == false) {
         p->amp_calc[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc[i] = p->amp_init[i];
         p->amp_err[i] = 0;
      }
      if (p->fix_position_x[i] == false) {
         p->position_calc_x[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_x[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_x[i] = p->position_init_x[i];
         p->position_err_x[i] = 0;
      }
      if (p->fix_position_y[i] == false) {
         p->position_calc_y[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_y[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_y[i] = p->position_init_y[i];
         p->position_err_y[i] = 0;
      }
      if (p->fix_amp_x1[i] == false) {
         p->amp_calc_x1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err_x1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc_x1[i] = p->amp_init_x1[i];
         p->amp_err_x1[i] = 0;
      }
      if (p->fix_amp_y1[i] == false) {
         p->amp_calc_y1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->amp_err_y1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->amp_calc_y1[i] = p->amp_init_y1[i];
         p->amp_err_y1[i] = 0;
      }
      if (p->fix_position_x1[i] == false) {
         p->position_calc_x1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_x1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_x1[i] = p->position_init_x1[i];
         p->position_err_x1[i] = 0;
      }
      if (p->fix_position_y1[i] == false) {
         p->position_calc_y1[i] = working_space[shift + j];	//xk[j]
         if (working_space[3 * shift + j] != 0)
            p->position_err_y1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
         j += 1;
      }
      
      else {
         p->position_calc_y1[i] = p->position_init_y1[i];
         p->position_err_y1[i] = 0;
      }
   }
   if (p->fix_sigma_x == false) {
      p->sigma_calc_x = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err_x = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc_x = p->sigma_init_x;
      p->sigma_err_x = 0;
   }
   if (p->fix_sigma_y == false) {
      p->sigma_calc_y = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sigma_err_y = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sigma_calc_y = p->sigma_init_y;
      p->sigma_err_y = 0;
   }
   if (p->fix_ro == false) {
      p->ro_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ro_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ro_calc = p->ro_init;
      p->ro_err = 0;
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
   if (p->fix_ax == false) {
      p->ax_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ax_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ax_calc = p->ax_init;
      p->ax_err = 0;
   }
   if (p->fix_ay == false) {
      p->ay_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ay_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ay_calc = p->ay_init;
      p->ay_err = 0;
   }
   if (p->fix_txy == false) {
      p->txy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->txy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->txy_calc = p->txy_init;
      p->txy_err = 0;
   }
   if (p->fix_sxy == false) {
      p->sxy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sxy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sxy_calc = p->sxy_init;
      p->sxy_err = 0;
   }
   if (p->fix_tx == false) {
      p->tx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->tx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->tx_calc = p->tx_init;
      p->tx_err = 0;
   }
   if (p->fix_ty == false) {
      p->ty_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->ty_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->ty_calc = p->ty_init;
      p->ty_err = 0;
   }
   if (p->fix_sx == false) {
      p->sx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sx_calc = p->sx_init;
      p->sx_err = 0;
   }
   if (p->fix_sy == false) {
      p->sy_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->sy_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->sy_calc = p->sy_init;
      p->sy_err = 0;
   }
   if (p->fix_bx == false) {
      p->bx_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->bx_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->bx_calc = p->bx_init;
      p->bx_err = 0;
   }
   if (p->fix_by == false) {
      p->by_calc = working_space[shift + j];	//xk[j]
      if (working_space[3 * shift + j] != 0)	//temp[j]
         p->by_err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));	//der[j]/temp[j]
      j += 1;
   }
   
   else {
      p->by_calc = p->by_init;
      p->by_err = 0;
   }
   b = (p->xmax - p->xmin + 1) * (p->ymax - p->ymin + 1) - rozmer;
   p->chi = chi_cel / b;
   for (i1 = p->xmin; i1 <= p->xmax; i1++) {
      for (i2 = p->ymin; i2 <= p->ymax; i2++) {
         f = Shape2(p->number_of_peaks, (double) i1, (double) i2,
                     working_space, working_space[peak_vel],
                     working_space[peak_vel + 1],
                     working_space[peak_vel + 2],
                     working_space[peak_vel + 3],
                     working_space[peak_vel + 4],
                     working_space[peak_vel + 5],
                     working_space[peak_vel + 6],
                     working_space[peak_vel + 7],
                     working_space[peak_vel + 8],
                     working_space[peak_vel + 9],
                     working_space[peak_vel + 10],
                     working_space[peak_vel + 11],
                     working_space[peak_vel + 12],
                     working_space[peak_vel + 13]);
         source[i1][i2] = f;
   } } for (i = 0; i < rozmer; i++)
      delete[]working_matrix[i];
   delete[]working_matrix;
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
//////////AUXILIARY FUNCTIONS FOR TRANSFORM BASED FUNCTIONS////////////////////////
void TSpectrum2::Haar(float *working_space, int num, int direction) 
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
   if (direction == TRANSFORM2_FORWARD) {
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
   if (direction == TRANSFORM2_INVERSE) {
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
void TSpectrum2::Walsh(float *working_space, int num) 
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
void TSpectrum2::BitReverse(float *working_space, int num) 
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
void TSpectrum2::Fourier(float *working_space, int num, int hartley,
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
   if (direction == TRANSFORM2_FORWARD && zt_clear == 0) {
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
   if (direction == TRANSFORM2_INVERSE)
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
   if (hartley == 1 && direction == TRANSFORM2_INVERSE) {
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
void TSpectrum2::BitReverseHaar(float *working_space, int shift, int num,
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
int TSpectrum2::GeneralExe(float *working_space, int zt_clear, int num,
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
           && (type == TRANSFORM2_FOURIER_HAAR
               || type == TRANSFORM2_WALSH_HAAR
               || type == TRANSFORM2_COS_HAAR
               || type == TRANSFORM2_SIN_HAAR))
         mp2step *= 2;
      if (ring > 1)
         ring = ring / 2;
      for (mp = 0; mp < nump; mp++) {
         if (type != TRANSFORM2_WALSH_HAAR) {
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
int TSpectrum2::GeneralInv(float *working_space, int num, int degree,
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
   if (type == TRANSFORM2_FOURIER_HAAR || type == TRANSFORM2_WALSH_HAAR
        || type == TRANSFORM2_COS_HAAR || type == TRANSFORM2_SIN_HAAR) {
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
         if (type != TRANSFORM2_WALSH_HAAR) {
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
           && (type == TRANSFORM2_FOURIER_HAAR
               || type == TRANSFORM2_WALSH_HAAR
               || type == TRANSFORM2_COS_HAAR
               || type == TRANSFORM2_SIN_HAAR))
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
void TSpectrum2::HaarWalsh2(float **working_matrix,
                              float *working_vector, int numx, int numy,
                              int direction, int type) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates 2D Haar and Walsh transforms                       //
//      Function parameters:                                                    //
//              -working_matrix-pointer to matrix of transformed data           //
//              -working_vector-pointer to vector where the data are processed  //
//              -numx,numy-lengths of processed data                            //
//              -direction-forward or inverse                                   //
//              -type-type of transform (see manual)                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j;
   if (direction == TRANSFORM2_FORWARD) {
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case TRANSFORM2_HAAR:
            Haar(working_vector, numx, TRANSFORM2_FORWARD);
            break;
         case TRANSFORM2_WALSH:
            Walsh(working_vector, numx);
            BitReverse(working_vector, numx);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
         }
         switch (type) {
         case TRANSFORM2_HAAR:
            Haar(working_vector, numy, TRANSFORM2_FORWARD);
            break;
         case TRANSFORM2_WALSH:
            Walsh(working_vector, numy);
            BitReverse(working_vector, numy);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
         }
      }
   }
   
   else if (direction == TRANSFORM2_INVERSE) {
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
         }
         switch (type) {
         case TRANSFORM2_HAAR:
            Haar(working_vector, numy, TRANSFORM2_INVERSE);
            break;
         case TRANSFORM2_WALSH:
            BitReverse(working_vector, numy);
            Walsh(working_vector, numy);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
         }
      }
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case TRANSFORM2_HAAR:
            Haar(working_vector, numx, TRANSFORM2_INVERSE);
            break;
         case TRANSFORM2_WALSH:
            BitReverse(working_vector, numx);
            Walsh(working_vector, numx);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
   }
   return;
}
void TSpectrum2::FourCos2(float **working_matrix, float *working_vector,
                            int numx, int numy, int direction, int type) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates 2D Fourier based transforms                        //
//      Function parameters:                                                    //
//              -working_matrix-pointer to matrix of transformed data           //
//              -working_vector-pointer to vector where the data are processed  //
//              -numx,numy-lengths of processed data                            //
//              -direction-forward or inverse                                   //
//              -type-type of transform (see manual)                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j, iterx, itery, n, size;
   double pi = 3.14159265358979323846;
   j = 0;
   n = 1;
   for (; n < numx;) {
      j += 1;
      n = n * 2;
   }
   j = 0;
   n = 1;
   for (; n < numy;) {
      j += 1;
      n = n * 2;
   }
   i = numx;
   iterx = 0;
   for (; i > 1;) {
      iterx += 1;
      i = i / 2;
   }
   i = numy;
   itery = 0;
   for (; i > 1;) {
      itery += 1;
      i = i / 2;
   }
   size = numx;
   if (size < numy)
      size = numy;
   if (direction == TRANSFORM2_FORWARD) {
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case TRANSFORM2_COS:
            for (i = 1; i <= numx; i++) {
               working_vector[2 * numx - i] = working_vector[i - 1];
            }
            Fourier(working_vector, 2 * numx, 0, TRANSFORM2_FORWARD, 0);
            for (i = 0; i < numx; i++) {
               working_vector[i] =
                   working_vector[i] / TMath::Cos(pi * i / (2 * numx));
            }
            working_vector[0] = working_vector[0] / TMath::Sqrt(2.);
            break;
         case TRANSFORM2_SIN:
            for (i = 1; i <= numx; i++) {
               working_vector[2 * numx - i] = -working_vector[i - 1];
            }
            Fourier(working_vector, 2 * numx, 0, TRANSFORM2_FORWARD, 0);
            for (i = 1; i < numx; i++) {
               working_vector[i - 1] =
                   working_vector[i] / TMath::Sin(pi * i / (2 * numx));
            }
            working_vector[numx - 1] =
                working_vector[numx] / TMath::Sqrt(2.);
            break;
         case TRANSFORM2_FOURIER:
            Fourier(working_vector, numx, 0, TRANSFORM2_FORWARD, 0);
            break;
         case TRANSFORM2_HARTLEY:
            Fourier(working_vector, numx, 1, TRANSFORM2_FORWARD, 0);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
            if (type == TRANSFORM2_FOURIER)
               working_matrix[i][j + numy] = working_vector[i + numx];
            
            else
               working_matrix[i][j + numy] = working_vector[i + 2 * numx];
         }
      }
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == TRANSFORM2_FOURIER)
               working_vector[j + numy] = working_matrix[i][j + numy];
            
            else
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];
         }
         switch (type) {
         case TRANSFORM2_COS:
            for (j = 1; j <= numy; j++) {
               working_vector[2 * numy - j] = working_vector[j - 1];
            }
            Fourier(working_vector, 2 * numy, 0, TRANSFORM2_FORWARD, 0);
            for (j = 0; j < numy; j++) {
               working_vector[j] =
                   working_vector[j] / TMath::Cos(pi * j / (2 * numy));
               working_vector[j + 2 * numy] = 0;
            }
            working_vector[0] = working_vector[0] / TMath::Sqrt(2.);
            break;
         case TRANSFORM2_SIN:
            for (j = 1; j <= numy; j++) {
               working_vector[2 * numy - j] = -working_vector[j - 1];
            }
            Fourier(working_vector, 2 * numy, 0, TRANSFORM2_FORWARD, 0);
            for (j = 1; j < numy; j++) {
               working_vector[j - 1] =
                   working_vector[j] / TMath::Sin(pi * j / (2 * numy));
               working_vector[j + numy] = 0;
            }
            working_vector[numy - 1] =
                working_vector[numy] / TMath::Sqrt(2.);
            working_vector[numy] = 0;
            break;
         case TRANSFORM2_FOURIER:
            Fourier(working_vector, numy, 0, TRANSFORM2_FORWARD, 1);
            break;
         case TRANSFORM2_HARTLEY:
            Fourier(working_vector, numy, 1, TRANSFORM2_FORWARD, 0);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == TRANSFORM2_FOURIER)
               working_matrix[i][j + numy] = working_vector[j + numy];
            
            else
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];
         }
      }
   }
   
   else if (direction == TRANSFORM2_INVERSE) {
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == TRANSFORM2_FOURIER)
               working_vector[j + numy] = working_matrix[i][j + numy];
            
            else
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];
         }
         switch (type) {
         case TRANSFORM2_COS:
            working_vector[0] = working_vector[0] * TMath::Sqrt(2.);
            for (j = 0; j < numy; j++) {
               working_vector[j + 2 * numy] =
                   working_vector[j] * TMath::Sin(pi * j / (2 * numy));
               working_vector[j] =
                   working_vector[j] * TMath::Cos(pi * j / (2 * numy));
            }
            for (j = 1; j < numy; j++) {
               working_vector[2 * numy - j] = working_vector[j];
               working_vector[2 * numy - j + 2 * numy] =
                   -working_vector[j + 2 * numy];
            }
            working_vector[numy] = 0;
            working_vector[numy + 2 * numy] = 0;
            Fourier(working_vector, 2 * numy, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_SIN:
            working_vector[numy] =
                working_vector[numy - 1] * TMath::Sqrt(2.);
            for (j = numy - 1; j > 0; j--) {
               working_vector[j + 2 * numy] =
                   -working_vector[j -
                                   1] * TMath::Cos(pi * j / (2 * numy));
               working_vector[j] =
                   working_vector[j - 1] * TMath::Sin(pi * j / (2 * numy));
            }
            for (j = 1; j < numy; j++) {
               working_vector[2 * numy - j] = working_vector[j];
               working_vector[2 * numy - j + 2 * numy] =
                   -working_vector[j + 2 * numy];
            }
            working_vector[0] = 0;
            working_vector[0 + 2 * numy] = 0;
            working_vector[numy + 2 * numy] = 0;
            Fourier(working_vector, 2 * numy, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_FOURIER:
            Fourier(working_vector, numy, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_HARTLEY:
            Fourier(working_vector, numy, 1, TRANSFORM2_INVERSE, 1);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == TRANSFORM2_FOURIER)
               working_matrix[i][j + numy] = working_vector[j + numy];
            
            else
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];
         }
      }
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
            if (type == TRANSFORM2_FOURIER)
               working_vector[i + numx] = working_matrix[i][j + numy];
            
            else
               working_vector[i + 2 * numx] = working_matrix[i][j + numy];
         }
         switch (type) {
         case TRANSFORM2_COS:
            working_vector[0] = working_vector[0] * TMath::Sqrt(2.);
            for (i = 0; i < numx; i++) {
               working_vector[i + 2 * numx] =
                   working_vector[i] * TMath::Sin(pi * i / (2 * numx));
               working_vector[i] =
                   working_vector[i] * TMath::Cos(pi * i / (2 * numx));
            }
            for (i = 1; i < numx; i++) {
               working_vector[2 * numx - i] = working_vector[i];
               working_vector[2 * numx - i + 2 * numx] =
                   -working_vector[i + 2 * numx];
            }
            working_vector[numx] = 0;
            working_vector[numx + 2 * numx] = 0;
            Fourier(working_vector, 2 * numx, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_SIN:
            working_vector[numx] =
                working_vector[numx - 1] * TMath::Sqrt(2.);
            for (i = numx - 1; i > 0; i--) {
               working_vector[i + 2 * numx] =
                   -working_vector[i -
                                   1] * TMath::Cos(pi * i / (2 * numx));
               working_vector[i] =
                   working_vector[i - 1] * TMath::Sin(pi * i / (2 * numx));
            }
            for (i = 1; i < numx; i++) {
               working_vector[2 * numx - i] = working_vector[i];
               working_vector[2 * numx - i + 2 * numx] =
                   -working_vector[i + 2 * numx];
            }
            working_vector[0] = 0;
            working_vector[0 + 2 * numx] = 0;
            working_vector[numx + 2 * numx] = 0;
            Fourier(working_vector, 2 * numx, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_FOURIER:
            Fourier(working_vector, numx, 0, TRANSFORM2_INVERSE, 1);
            break;
         case TRANSFORM2_HARTLEY:
            Fourier(working_vector, numx, 1, TRANSFORM2_INVERSE, 1);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
   }
   return;
}
void TSpectrum2::General2(float **working_matrix, float *working_vector,
                            int numx, int numy, int direction, int type,
                            int degree) 
{
   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates generalized (mixed) 2D transforms                  //
//      Function parameters:                                                    //
//              -working_matrix-pointer to matrix of transformed data           //
//              -working_vector-pointer to vector where the data are processed  //
//              -numx,numy-lengths of processed data                            //
//              -direction-forward or inverse                                   //
//              -type-type of transform (see manual)                            //
//              -degree-degree of transform (see manual)                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, j, jstup, kstup, l, m;
   float val, valx, valz;
   double a, b, pi = 3.14159265358979323846;
   if (direction == TRANSFORM2_FORWARD) {
      for (j = 0; j < numy; j++) {
         kstup = (int) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0; i < numx; i++) {
            val = working_matrix[i][j];
            if (type == TRANSFORM2_COS_WALSH
                 || type == TRANSFORM2_COS_HAAR) {
               jstup = (int) TMath::Power(2, degree) / 2;
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + i % jstup] = val;
               working_vector[kstup + 2 * jstup - 1 - i % jstup] = val;
            }
            
            else if (type == TRANSFORM2_SIN_WALSH
                     || type == TRANSFORM2_SIN_HAAR) {
               jstup = (int) TMath::Power(2, degree) / 2;
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + i % jstup] = val;
               working_vector[kstup + 2 * jstup - 1 - i % jstup] = -val;
            }
            
            else
               working_vector[i] = val;
         }
         switch (type) {
         case TRANSFORM2_FOURIER_WALSH:
         case TRANSFORM2_FOURIER_HAAR:
         case TRANSFORM2_WALSH_HAAR:
            GeneralExe(working_vector, 0, numx, degree, type);
            for (i = 0; i < jstup; i++)
               BitReverseHaar(working_vector, numx, kstup, i * kstup);
            break;
         case TRANSFORM2_COS_WALSH:
         case TRANSFORM2_COS_HAAR:
            m = (int) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            GeneralExe(working_vector, 0, 2 * numx, degree, type);
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (i % jstup) / (double) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[kstup + i % jstup];
               if (i % jstup == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_vector[i] = a;
               working_vector[i + 4 * numx] = 0;
            }
            break;
         case TRANSFORM2_SIN_WALSH:
         case TRANSFORM2_SIN_HAAR:
            m = (int) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            GeneralExe(working_vector, 0, 2 * numx, degree, type);
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (i % jstup) / (double) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[jstup + kstup + i % jstup];
               if (i % jstup == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_vector[jstup + kstup / 2 - i % jstup - 1] = a;
               working_vector[i + 4 * numx] = 0;
            }
            break;
         }
         if (type > TRANSFORM2_WALSH_HAAR)
            kstup = (int) TMath::Power(2, degree - 1);
         
         else
            kstup = (int) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0, l = 0; i < numx; i++, l = (l + kstup) % numx) {
            working_vector[numx + i] = working_vector[l + i / jstup];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[numx + i + 2 * numx] =
                   working_vector[l + i / jstup + 2 * numx];
            
            else
               working_vector[numx + i + 4 * numx] =
                   working_vector[l + i / jstup + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_vector[numx + i];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[i + 2 * numx] =
                   working_vector[numx + i + 2 * numx];
            
            else
               working_vector[i + 4 * numx] =
                   working_vector[numx + i + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_matrix[i][j + numy] = working_vector[i + 2 * numx];
            
            else
               working_matrix[i][j + numy] = working_vector[i + 4 * numx];
         }
      }
      for (i = 0; i < numx; i++) {
         kstup = (int) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0; j < numy; j++) {
            valx = working_matrix[i][j];
            valz = working_matrix[i][j + numy];
            if (type == TRANSFORM2_COS_WALSH
                 || type == TRANSFORM2_COS_HAAR) {
               jstup = (int) TMath::Power(2, degree) / 2;
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + j % jstup] = valx;
               working_vector[kstup + 2 * jstup - 1 - j % jstup] = valx;
               working_vector[kstup + j % jstup + 4 * numy] = valz;
               working_vector[kstup + 2 * jstup - 1 - j % jstup +
                               4 * numy] = valz;
            }
            
            else if (type == TRANSFORM2_SIN_WALSH
                     || type == TRANSFORM2_SIN_HAAR) {
               jstup = (int) TMath::Power(2, degree) / 2;
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + j % jstup] = valx;
               working_vector[kstup + 2 * jstup - 1 - j % jstup] = -valx;
               working_vector[kstup + j % jstup + 4 * numy] = valz;
               working_vector[kstup + 2 * jstup - 1 - j % jstup +
                               4 * numy] = -valz;
            }
            
            else {
               working_vector[j] = valx;
               working_vector[j + 2 * numy] = valz;
            }
         }
         switch (type) {
         case TRANSFORM2_FOURIER_WALSH:
         case TRANSFORM2_FOURIER_HAAR:
         case TRANSFORM2_WALSH_HAAR:
            GeneralExe(working_vector, 1, numy, degree, type);
            for (j = 0; j < jstup; j++)
               BitReverseHaar(working_vector, numy, kstup, j * kstup);
            break;
         case TRANSFORM2_COS_WALSH:
         case TRANSFORM2_COS_HAAR:
            m = (int) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            GeneralExe(working_vector, 1, 2 * numy, degree, type);
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (j % jstup) / (double) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[kstup + j % jstup];
               if (j % jstup == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_vector[j] = a;
               working_vector[j + 4 * numy] = 0;
            }
            break;
         case TRANSFORM2_SIN_WALSH:
         case TRANSFORM2_SIN_HAAR:
            m = (int) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            GeneralExe(working_vector, 1, 2 * numy, degree, type);
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (j % jstup) / (double) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[jstup + kstup + j % jstup];
               if (j % jstup == 0)
                  a = b / TMath::Sqrt(2.0);
               
               else
                  a = b / a;
               working_vector[jstup + kstup / 2 - j % jstup - 1] = a;
               working_vector[j + 4 * numy] = 0;
            }
            break;
         }
         if (type > TRANSFORM2_WALSH_HAAR)
            kstup = (int) TMath::Power(2, degree - 1);
         
         else
            kstup = (int) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0, l = 0; j < numy; j++, l = (l + kstup) % numy) {
            working_vector[numy + j] = working_vector[l + j / jstup];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[numy + j + 2 * numy] =
                   working_vector[l + j / jstup + 2 * numy];
            
            else
               working_vector[numy + j + 4 * numy] =
                   working_vector[l + j / jstup + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_vector[numy + j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[j + 2 * numy] =
                   working_vector[numy + j + 2 * numy];
            
            else
               working_vector[j + 4 * numy] =
                   working_vector[numy + j + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];
            
            else
               working_matrix[i][j + numy] = working_vector[j + 4 * numy];
         }
      }
   }
   
   else if (direction == TRANSFORM2_INVERSE) {
      for (i = 0; i < numx; i++) {
         kstup = (int) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];
            
            else
               working_vector[j + 4 * numy] = working_matrix[i][j + numy];
         }
         if (type > TRANSFORM2_WALSH_HAAR)
            kstup = (int) TMath::Power(2, degree - 1);
         
         else
            kstup = (int) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0, l = 0; j < numy; j++, l = (l + kstup) % numy) {
            working_vector[numy + l + j / jstup] = working_vector[j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[numy + l + j / jstup + 2 * numy] =
                   working_vector[j + 2 * numy];
            
            else
               working_vector[numy + l + j / jstup + 4 * numy] =
                   working_vector[j + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_vector[numy + j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[j + 2 * numy] =
                   working_vector[numy + j + 2 * numy];
            
            else
               working_vector[j + 4 * numy] =
                   working_vector[numy + j + 4 * numy];
         }
         switch (type) {
         case TRANSFORM2_FOURIER_WALSH:
         case TRANSFORM2_FOURIER_HAAR:
         case TRANSFORM2_WALSH_HAAR:
            for (j = 0; j < jstup; j++)
               BitReverseHaar(working_vector, numy, kstup, j * kstup);
            GeneralInv(working_vector, numy, degree, type);
            break;
         case TRANSFORM2_COS_WALSH:
         case TRANSFORM2_COS_HAAR:
            jstup = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (j % jstup) / (double) (2 * jstup);
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + j % jstup] =
                      working_vector[j] * TMath::Sqrt(2.0);
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] =
                      -(double) working_vector[j] * b;
                  working_vector[2 * numy + kstup + j % jstup] =
                      (double) working_vector[j] * a;
            } } for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + jstup] = 0;
                  working_vector[2 * numy + kstup + jstup + 4 * numy] = 0;
               }
               
               else {
                  working_vector[2 * numy + kstup + 2 * jstup -
                                  j % jstup] =
                      working_vector[2 * numy + kstup + j % jstup];
                  working_vector[2 * numy + kstup + 2 * jstup -
                                  j % jstup + 4 * numy] =
                      -working_vector[2 * numy + kstup + j % jstup +
                                      4 * numy];
               }
            }
            for (j = 0; j < 2 * numy; j++) {
               working_vector[j] = working_vector[2 * numy + j];
               working_vector[j + 4 * numy] =
                   working_vector[2 * numy + j + 4 * numy];
            }
            GeneralInv(working_vector, 2 * numy, degree, type);
            m = (int) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            break;
         case TRANSFORM2_SIN_WALSH:
         case TRANSFORM2_SIN_HAAR:
            jstup = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (j % jstup) / (double) (2 * jstup);
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + jstup + j % jstup] =
                      working_vector[jstup + kstup / 2 - j % jstup -
                                     1] * TMath::Sqrt(2.0);
                  working_vector[2 * numy + kstup + jstup + j % jstup +
                                  4 * numy] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numy + kstup + jstup + j % jstup +
                                  4 * numy] =
                      -(double) working_vector[jstup + kstup / 2 -
                                               j % jstup - 1] * b;
                  working_vector[2 * numy + kstup + jstup + j % jstup] =
                      (double) working_vector[jstup + kstup / 2 -
                                              j % jstup - 1] * a;
            } } for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup] = 0;
                  working_vector[2 * numy + kstup + 4 * numy] = 0;
               }
               
               else {
                  working_vector[2 * numy + kstup + j % jstup] =
                      working_vector[2 * numy + kstup + 2 * jstup -
                                     j % jstup];
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] =
                      -working_vector[2 * numy + kstup + 2 * jstup -
                                      j % jstup + 4 * numy];
               }
            }
            for (j = 0; j < 2 * numy; j++) {
               working_vector[j] = working_vector[2 * numy + j];
               working_vector[j + 4 * numy] =
                   working_vector[2 * numy + j + 4 * numy];
            }
            GeneralInv(working_vector, 2 * numy, degree, type);
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            break;
         }
         for (j = 0; j < numy; j++) {
            if (type > TRANSFORM2_WALSH_HAAR) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               valx = working_vector[kstup + j % jstup];
               valz = working_vector[kstup + j % jstup + 4 * numy];
            }
            
            else {
               valx = working_vector[j];
               valz = working_vector[j + 2 * numy];
            }
            working_matrix[i][j] = valx;
            working_matrix[i][j + numy] = valz;
         }
      }
      for (j = 0; j < numy; j++) {
         kstup = (int) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[i + 2 * numx] = working_matrix[i][j + numy];
            
            else
               working_vector[i + 4 * numx] = working_matrix[i][j + numy];
         }
         if (type > TRANSFORM2_WALSH_HAAR)
            kstup = (int) TMath::Power(2, degree - 1);
         
         else
            kstup = (int) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0, l = 0; i < numx; i++, l = (l + kstup) % numx) {
            working_vector[numx + l + i / jstup] = working_vector[i];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[numx + l + i / jstup + 2 * numx] =
                   working_vector[i + 2 * numx];
            
            else
               working_vector[numx + l + i / jstup + 4 * numx] =
                   working_vector[i + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_vector[numx + i];
            if (type == TRANSFORM2_FOURIER_WALSH
                 || type == TRANSFORM2_FOURIER_HAAR
                 || type == TRANSFORM2_WALSH_HAAR)
               working_vector[i + 2 * numx] =
                   working_vector[numx + i + 2 * numx];
            
            else
               working_vector[i + 4 * numx] =
                   working_vector[numx + i + 4 * numx];
         }
         switch (type) {
         case TRANSFORM2_FOURIER_WALSH:
         case TRANSFORM2_FOURIER_HAAR:
         case TRANSFORM2_WALSH_HAAR:
            for (i = 0; i < jstup; i++)
               BitReverseHaar(working_vector, numx, kstup, i * kstup);
            GeneralInv(working_vector, numx, degree, type);
            break;
         case TRANSFORM2_COS_WALSH:
         case TRANSFORM2_COS_HAAR:
            jstup = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (i % jstup) / (double) (2 * jstup);
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + i % jstup] =
                      working_vector[i] * TMath::Sqrt(2.0);
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] =
                      -(double) working_vector[i] * b;
                  working_vector[2 * numx + kstup + i % jstup] =
                      (double) working_vector[i] * a;
            } } for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + jstup] = 0;
                  working_vector[2 * numx + kstup + jstup + 4 * numx] = 0;
               }
               
               else {
                  working_vector[2 * numx + kstup + 2 * jstup -
                                  i % jstup] =
                      working_vector[2 * numx + kstup + i % jstup];
                  working_vector[2 * numx + kstup + 2 * jstup -
                                  i % jstup + 4 * numx] =
                      -working_vector[2 * numx + kstup + i % jstup +
                                      4 * numx];
               }
            }
            for (i = 0; i < 2 * numx; i++) {
               working_vector[i] = working_vector[2 * numx + i];
               working_vector[i + 4 * numx] =
                   working_vector[2 * numx + i + 4 * numx];
            }
            GeneralInv(working_vector, 2 * numx, degree, type);
            m = (int) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            break;
         case TRANSFORM2_SIN_WALSH:
         case TRANSFORM2_SIN_HAAR:
            jstup = (int) TMath::Power(2, degree) / 2;
            m = (int) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (double) (i % jstup) / (double) (2 * jstup);
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + jstup + i % jstup] =
                      working_vector[jstup + kstup / 2 - i % jstup -
                                     1] * TMath::Sqrt(2.0);
                  working_vector[2 * numx + kstup + jstup + i % jstup +
                                  4 * numx] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numx + kstup + jstup + i % jstup +
                                  4 * numx] =
                      -(double) working_vector[jstup + kstup / 2 -
                                               i % jstup - 1] * b;
                  working_vector[2 * numx + kstup + jstup + i % jstup] =
                      (double) working_vector[jstup + kstup / 2 -
                                              i % jstup - 1] * a;
            } } for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup] = 0;
                  working_vector[2 * numx + kstup + 4 * numx] = 0;
               }
               
               else {
                  working_vector[2 * numx + kstup + i % jstup] =
                      working_vector[2 * numx + kstup + 2 * jstup -
                                     i % jstup];
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] =
                      -working_vector[2 * numx + kstup + 2 * jstup -
                                      i % jstup + 4 * numx];
               }
            }
            for (i = 0; i < 2 * numx; i++) {
               working_vector[i] = working_vector[2 * numx + i];
               working_vector[i + 4 * numx] =
                   working_vector[2 * numx + i + 4 * numx];
            }
            GeneralInv(working_vector, 2 * numx, degree, type);
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            break;
         }
         for (i = 0; i < numx; i++) {
            if (type > TRANSFORM2_WALSH_HAAR) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               val = working_vector[kstup + i % jstup];
            }
            
            else
               val = working_vector[i];
            working_matrix[i][j] = val;
         }
      }
   }
   return;
}


///////////////////////END OF AUXILIARY TRANSFORM2 FUNCTIONS//////////////////////////////////////////
    
//////////TRANSFORM2 FUNCTION - CALCULATES DIFFERENT 2-D DIRECT AND INVERSE ORTHOGONAL TRANSFORMS//////
const char *TSpectrum2::Transform2(const float **source, float **dest,
                                   int sizex, int sizey, int type,
                                   int direction, int degree) 
{
   
//////////////////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL TRANSFORM FUNCTION				                */ 
/*	This function transforms the source spectrum. The calling program               */ 
/*      should fill in input parameters.                         	                */ 
/*	Transformed data are written into dest spectrum.                                */ 
/*									                */ 
/*	Function parameters:						                */ 
/*	source-pointer to the matrix of source spectrum, its size should                */ 
/*             be sizex*sizey except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR         */ 
/*             transform. These need sizex*2*sizey length to supply real and            */ 
/*             imaginary coefficients.                                                  */ 
/*	dest-pointer to the matrix of destination data, its size should be              */ 
/*           sizex*sizey except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These        */ 
/*           need sizex*2*sizey length to store real and imaginary coefficients         */ 
/*	sizex,sizey-basic dimensions of source and dest spectra                         */ 
/*	type-type of transform                                                          */ 
/*      direction-transform direction (forward, inverse)                                */ 
/*      degree-applied only for mixed transforms                                        */ 
/*									                */ 
//////////////////////////////////////////////////////////////////////////////////////////
   int i, j, nx, ny, k, jx, jy;
   int size;
   float *working_vector = 0, **working_matrix = 0;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   jx = 0;
   nx = 1;
   for (; nx < sizex;) {
      jx += 1;
      nx = nx * 2;
   }
   if (nx != sizex)
      return ("LENGTH X MUST BE POWER OF 2");
   jy = 0;
   ny = 1;
   for (; ny < sizey;) {
      jy += 1;
      ny = ny * 2;
   }
   if (ny != sizey)
      return ("LENGTH Y MUST BE POWER OF 2");
   if (type < TRANSFORM2_HAAR || type > TRANSFORM2_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (direction != TRANSFORM2_FORWARD && direction != TRANSFORM2_INVERSE)
      return ("WRONG TRANSFORM DIRECTION");
   if (type >= TRANSFORM2_FOURIER_WALSH && type <= TRANSFORM2_SIN_HAAR) {
      if (degree > jx || degree > jy || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM2_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      jx = sizex / k;
      jy = sizey / k;
   }
   size = (int) TMath::Max(sizex, sizey);
   switch (type) {
   case TRANSFORM2_HAAR:
   case TRANSFORM2_WALSH:
      working_vector = new float[2 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[sizey];
      break;
   case TRANSFORM2_COS:
   case TRANSFORM2_SIN:
   case TRANSFORM2_FOURIER:
   case TRANSFORM2_HARTLEY:
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
      working_vector = new float[4 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      working_vector = new float[8 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   }
   if (direction == TRANSFORM2_FORWARD) {
      switch (type) {
      case TRANSFORM2_HAAR:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                     direction, TRANSFORM2_HAAR);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_WALSH:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                     direction, TRANSFORM2_WALSH);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_COS:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_COS);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_SIN:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_SIN);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_FOURIER:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_FOURIER);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j + sizey] = working_matrix[i][j + sizey];
            }
         }
         break;
      case TRANSFORM2_HARTLEY:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_HARTLEY);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_FOURIER_WALSH:
      case TRANSFORM2_FOURIER_HAAR:
      case TRANSFORM2_WALSH_HAAR:
      case TRANSFORM2_COS_WALSH:
      case TRANSFORM2_COS_HAAR:
      case TRANSFORM2_SIN_WALSH:
      case TRANSFORM2_SIN_HAAR:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         General2(working_matrix, working_vector, sizex, sizey, direction,
                   type, degree);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         if (type == TRANSFORM2_FOURIER_WALSH
              || type == TRANSFORM2_FOURIER_HAAR) {
            for (i = 0; i < sizex; i++) {
               for (j = 0; j < sizey; j++) {
                  dest[i][j + sizey] = working_matrix[i][j + sizey];
               }
            }
         }
         break;
      }
   }
   
   else if (direction == TRANSFORM2_INVERSE) {
      switch (type) {
      case TRANSFORM2_HAAR:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                     direction, TRANSFORM2_HAAR);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_WALSH:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                     direction, TRANSFORM2_WALSH);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_COS:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_COS);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_SIN:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_SIN);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_FOURIER:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j + sizey] = source[i][j + sizey];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_FOURIER);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_HARTLEY:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, sizex, sizey, direction,
                   TRANSFORM2_HARTLEY);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case TRANSFORM2_FOURIER_WALSH:
      case TRANSFORM2_FOURIER_HAAR:
      case TRANSFORM2_WALSH_HAAR:
      case TRANSFORM2_COS_WALSH:
      case TRANSFORM2_COS_HAAR:
      case TRANSFORM2_SIN_WALSH:
      case TRANSFORM2_SIN_HAAR:
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               working_matrix[i][j] = source[i][j];
            }
         }
         if (type == TRANSFORM2_FOURIER_WALSH
              || type == TRANSFORM2_FOURIER_HAAR) {
            for (i = 0; i < sizex; i++) {
               for (j = 0; j < sizey; j++) {
                  working_matrix[i][j + sizey] = source[i][j + sizey];
               }
            }
         }
         General2(working_matrix, working_vector, sizex, sizey, direction,
                   type, degree);
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j];
            }
         }
         break;
      }
   }
   for (i = 0; i < sizex; i++) {
      delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return 0;
}


//////////END OF TRANSFORM2 FUNCTION/////////////////////////////////
//_______________________________________________________________________________________
    
//////////FILTER2_ZONAL FUNCTION - CALCULATES DIFFERENT 2-D ORTHOGONAL TRANSFORMS, SETS GIVEN REGION TO FILTER COEFFICIENT AND TRANSFORMS IT BACK//////
const char *TSpectrum2::Filter2Zonal(const float **source, float **dest,
                                     int sizex, int sizey, int type,
                                     int degree, int xmin, int xmax,
                                     int ymin, int ymax,
                                     float filter_coeff) 
{
   
//////////////////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL FILTER ZONAL FUNCTION   			                */ 
/*	This function transforms the source spectrum. The calling program               */ 
/*      should fill in input parameters. Then it sets transformed                       */ 
/*      coefficients in the given region to the given                                   */ 
/*      filter_coeff and transforms it back                                             */ 
/*	Filtered data are written into dest spectrum.                                   */ 
/*									                */ 
/*	Function parameters:						                */ 
/*	source-pointer to the matrix of source spectrum, its size should                */ 
/*             be sizex*sizey                                                           */ 
/*	dest-pointer to the matrix of destination data, its size should be              */ 
/*           sizex*sizey                                                                */ 
/*	sizex,sizey-basic dimensions of source and dest spectra                         */ 
/*	type-type of transform                                                          */ 
/*      degree-applied only for mixed transforms                                        */ 
/*	xmin-low limit x of filtered region                                             */ 
/*	xmax-high limit x of filtered region                                            */ 
/*	ymin-low limit y of filtered region                                             */ 
/*	ymax-high limit y of filtered region                                            */ 
/*	filter_coeff-value which is set in filtered region                              */ 
/*									                */ 
//////////////////////////////////////////////////////////////////////////////////////////
   int i, j, nx, ny, k, jx, jy;
   double a, old_area = 0, new_area = 0;
   int size;
   float *working_vector = 0, **working_matrix = 0;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   jx = 0;
   nx = 1;
   for (; nx < sizex;) {
      jx += 1;
      nx = nx * 2;
   }
   if (nx != sizex)
      return ("LENGTH X MUST BE POWER OF 2");
   jy = 0;
   ny = 1;
   for (; ny < sizey;) {
      jy += 1;
      ny = ny * 2;
   }
   if (ny != sizey)
      return ("LENGTH Y MUST BE POWER OF 2");
   if (type < TRANSFORM2_HAAR || type > TRANSFORM2_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (type >= TRANSFORM2_FOURIER_WALSH && type <= TRANSFORM2_SIN_HAAR) {
      if (degree > jx || degree > jy || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM2_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      jx = sizex / k;
      jy = sizey / k;
   }
   if (xmin < 0 || xmin > xmax)
      return ("WRONG LOW REGION X LIMIT");
   if (xmax < xmin || xmax >= sizex)
      return ("WRONG HIGH REGION X LIMIT");
   if (ymin < 0 || ymin > ymax)
      return ("WRONG LOW REGION Y LIMIT");
   if (ymax < ymin || ymax >= sizey)
      return ("WRONG HIGH REGION Y LIMIT");
   size = (int) TMath::Max(sizex, sizey);
   switch (type) {
   case TRANSFORM2_HAAR:
   case TRANSFORM2_WALSH:
      working_vector = new float[2 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[sizey];
      break;
   case TRANSFORM2_COS:
   case TRANSFORM2_SIN:
   case TRANSFORM2_FOURIER:
   case TRANSFORM2_HARTLEY:
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
      working_vector = new float[4 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      working_vector = new float[8 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   }
   switch (type) {
   case TRANSFORM2_HAAR:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_FORWARD, TRANSFORM2_HAAR);
      break;
   case TRANSFORM2_WALSH:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_FORWARD, TRANSFORM2_WALSH);
      break;
   case TRANSFORM2_COS:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_COS);
      break;
   case TRANSFORM2_SIN:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_SIN);
      break;
   case TRANSFORM2_FOURIER:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_FOURIER);
      break;
   case TRANSFORM2_HARTLEY:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_HARTLEY);
      break;
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      General2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, type, degree);
      break;
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         if (i >= xmin && i <= xmax && j >= ymin && j <= ymax)
            working_matrix[i][j] = filter_coeff;
      }
   }
   if (type == TRANSFORM2_FOURIER || type == TRANSFORM2_FOURIER_WALSH
        || type == TRANSFORM2_FOURIER_HAAR) {
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            if (i >= xmin && i <= xmax && j >= ymin && j <= ymax)
               working_matrix[i][j + sizey] = filter_coeff;
         }
      }
   }
   switch (type) {
   case TRANSFORM2_HAAR:
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_INVERSE, TRANSFORM2_HAAR);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_WALSH:
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_INVERSE, TRANSFORM2_WALSH);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_COS:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_COS);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_SIN:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_SIN);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_FOURIER:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_FOURIER);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_HARTLEY:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_HARTLEY);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      General2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, type, degree);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   }
   for (i = 0; i < sizex; i++) {
      delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return 0;
}


//////////  END OF FILTER2_ZONAL FUNCTION/////////////////////////////////
//______________________________________________________________________
    
//////////ENHANCE2 FUNCTION - CALCULATES DIFFERENT 2-D ORTHOGONAL TRANSFORMS, MULTIPLIES GIVEN REGION BY ENHANCE COEFFICIENT AND TRANSFORMS IT BACK//////
const char *TSpectrum2::Enhance2(const float **source, float **dest,
                                 int sizex, int sizey, int type,
                                 int degree, int xmin, int xmax, int ymin,
                                 int ymax, float enhance_coeff) 
{
   
//////////////////////////////////////////////////////////////////////////////////////////
/*	TWO-DIMENSIONAL ENHANCE ZONAL FUNCTION  			                */ 
/*	This function transforms the source spectrum. The calling program               */ 
/*      should fill in input parameters. Then it multiplies transformed                 */ 
/*      coefficients in the given region by the given                                   */ 
/*      enhance_coeff and transforms it back                                            */ 
/*									                */ 
/*	Function parameters:						                */ 
/*	source-pointer to the matrix of source spectrum, its size should                */ 
/*             be sizex*sizey                                                           */ 
/*	dest-pointer to the matrix of destination data, its size should be              */ 
/*           sizex*sizey                                                                */ 
/*	sizex,sizey-basic dimensions of source and dest spectra                         */ 
/*	type-type of transform                                                          */ 
/*      degree-applied only for mixed transforms                                        */ 
/*	xmin-low limit x of filtered region                                             */ 
/*	xmax-high limit x of filtered region                                            */ 
/*	ymin-low limit y of filtered region                                             */ 
/*	ymax-high limit y of filtered region                                            */ 
/*	enhance_coeff-value which is set in filtered region                             */ 
/*									                */ 
//////////////////////////////////////////////////////////////////////////////////////////
   int i, j, nx, ny, k, jx, jy;
   double a, old_area = 0, new_area = 0;
   int size;
   float *working_vector = 0, **working_matrix = 0;
   if (sizex <= 0 || sizey <= 0)
      return "Wrong parameters";
   jx = 0;
   nx = 1;
   for (; nx < sizex;) {
      jx += 1;
      nx = nx * 2;
   }
   if (nx != sizex)
      return ("LENGTH X MUST BE POWER OF 2");
   jy = 0;
   ny = 1;
   for (; ny < sizey;) {
      jy += 1;
      ny = ny * 2;
   }
   if (ny != sizey)
      return ("LENGTH Y MUST BE POWER OF 2");
   if (type < TRANSFORM2_HAAR || type > TRANSFORM2_SIN_HAAR)
      return ("WRONG TRANSFORM TYPE");
   if (type >= TRANSFORM2_FOURIER_WALSH && type <= TRANSFORM2_SIN_HAAR) {
      if (degree > jx || degree > jy || degree < 1)
         return ("WRONG DEGREE");
      if (type >= TRANSFORM2_COS_WALSH)
         degree += 1;
      k = (int) TMath::Power(2, degree);
      jx = sizex / k;
      jy = sizey / k;
   }
   if (xmin < 0 || xmin > xmax)
      return ("WRONG LOW REGION X LIMIT");
   if (xmax < xmin || xmax >= sizex)
      return ("WRONG HIGH REGION X LIMIT");
   if (ymin < 0 || ymin > ymax)
      return ("WRONG LOW REGION Y LIMIT");
   if (ymax < ymin || ymax >= sizey)
      return ("WRONG HIGH REGION Y LIMIT");
   size = (int) TMath::Max(sizex, sizey);
   switch (type) {
   case TRANSFORM2_HAAR:
   case TRANSFORM2_WALSH:
      working_vector = new float[2 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[sizey];
      break;
   case TRANSFORM2_COS:
   case TRANSFORM2_SIN:
   case TRANSFORM2_FOURIER:
   case TRANSFORM2_HARTLEY:
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
      working_vector = new float[4 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      working_vector = new float[8 * size];
      working_matrix = new float *[sizex];
      for (i = 0; i < sizex; i++)
         working_matrix[i] = new float[2 * sizey];
      break;
   }
   switch (type) {
   case TRANSFORM2_HAAR:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_FORWARD, TRANSFORM2_HAAR);
      break;
   case TRANSFORM2_WALSH:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_FORWARD, TRANSFORM2_WALSH);
      break;
   case TRANSFORM2_COS:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_COS);
      break;
   case TRANSFORM2_SIN:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_SIN);
      break;
   case TRANSFORM2_FOURIER:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_FOURIER);
      break;
   case TRANSFORM2_HARTLEY:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, TRANSFORM2_HARTLEY);
      break;
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            working_matrix[i][j] = source[i][j];
            old_area = old_area + source[i][j];
         }
      }
      General2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_FORWARD, type, degree);
      break;
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++) {
         if (i >= xmin && i <= xmax && j >= ymin && j <= ymax)
            working_matrix[i][j] *= enhance_coeff;
      }
   }
   if (type == TRANSFORM2_FOURIER || type == TRANSFORM2_FOURIER_WALSH
        || type == TRANSFORM2_FOURIER_HAAR) {
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            if (i >= xmin && i <= xmax && j >= ymin && j <= ymax)
               working_matrix[i][j + sizey] *= enhance_coeff;
         }
      }
   }
   switch (type) {
   case TRANSFORM2_HAAR:
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_INVERSE, TRANSFORM2_HAAR);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_WALSH:
      HaarWalsh2(working_matrix, working_vector, sizex, sizey,
                  TRANSFORM2_INVERSE, TRANSFORM2_WALSH);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_COS:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_COS);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_SIN:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_SIN);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_FOURIER:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_FOURIER);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_HARTLEY:
      FourCos2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, TRANSFORM2_HARTLEY);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case TRANSFORM2_FOURIER_WALSH:
   case TRANSFORM2_FOURIER_HAAR:
   case TRANSFORM2_WALSH_HAAR:
   case TRANSFORM2_COS_WALSH:
   case TRANSFORM2_COS_HAAR:
   case TRANSFORM2_SIN_WALSH:
   case TRANSFORM2_SIN_HAAR:
      General2(working_matrix, working_vector, sizex, sizey,
                TRANSFORM2_INVERSE, type, degree);
      for (i = 0; i < sizex; i++) {
         for (j = 0; j < sizey; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < sizex; i++) {
            for (j = 0; j < sizey; j++) {
               dest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   }
   for (i = 0; i < sizex; i++) {
      delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return 0;
}


//////////  END OF ENHANCE2 FUNCTION/////////////////////////////////
    
// --------------------------------------------------------------------------------
    
// ROOT page - Class index - Top of the page
    
// --------------------------------------------------------------------------------
    
// This page has been automatically generated. If you have any comments or suggestions about the page layout send a mail to ROOT support, or contact the developers with any questions or problems regarding ROOT.
    
