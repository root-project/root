// @(#)root/hist:$Name:  $:$Id: TSpectrum.cxx,v 1.5 2001/06/01 07:04:18 brun Exp $
// Author: Miroslav Morhac   27/05/99

/////////////////////////////////////////////////////////////////////////////
//   THIS CLASS CONTAINS ADVANCED SPECTRA PROCESSING FUNCTIONS.            //
//                                                                         //
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   TWO-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   TWO-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
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
//  The algorithms in this class have been published at the following      //
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
//Begin_Html
/*
   ftp://root.cern.ch/root/SpectrumDec.ps.gz
   ftp://root.cern.ch/root/SpectrumSrc.ps.gz
   ftp://root.cern.ch/root/SpectrumBck.ps.gz
*/
//End_Html
/////////////////////////////////////////////////////////////////////////////

#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TMath.h"
   #define PEAK_WINDOW 1024

ClassImp(TSpectrum)

//______________________________________________________________________________
TSpectrum::TSpectrum()
   :TNamed("Spectrum","Miroslav Morhac peak finder")
{
   Int_t n = 100;
   fMaxPeaks  = n;
   fPosition  = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fResolution= 1;
   fHistogram = 0;
   fNPeaks    = 0;
}

//______________________________________________________________________________
TSpectrum::TSpectrum(Int_t maxpositions, Float_t resolution)
   :TNamed("Spectrum","Miroslav Morhac peak finder")
{
//  maxpositions:  maximum number of peaks 
//  resolution:    determines resolution of the neighboring peaks 
//                 default value is 1 correspond to 3 sigma distance
//                 between peaks. Higher values allow higher resolution 
//                 (smaller distance between peaks.
//                 May be set later through SetResolution.
   
   Int_t n = TMath::Max(maxpositions,100);
   fMaxPeaks  = n;
   fPosition  = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fHistogram = 0;
   fNPeaks    = 0;
   SetResolution(resolution);
}

//______________________________________________________________________________
TSpectrum::~TSpectrum()
{
   delete [] fPosition;
   delete [] fPositionX;
   delete [] fPositionY;
   delete fHistogram;
}

//______________________________________________________________________________
char *TSpectrum::Background(TH1 *h,int number_of_iterations, Option_t *option)
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

    printf("Background function not yet implemented: h=%s, iter=%d, option=%s\n"
          ,h->GetName(),number_of_iterations,option);
   return 0;
}

//______________________________________________________________________________
char *TSpectrum::Background1(float *spectrum,int size,int number_of_iterations)
{
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   This function calculates background spectrum from source spectrum.    //
//   The result is placed in the vector pointed by spectrum pointer.       //
//                                                                         //
//   Function parameters:                                                  //
//   spectrum:  pointer to the vector of source spectrum                   //
//   size:      length of spectrum and working space vectors               //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
   int i,j;
   float a,b;
   if (size <= 0) return (char*)"Wrong Parameters";
   float *working_space = new float[size];
   for(i=1;i<=number_of_iterations;i++) {
      for(j=i;j<size-i;j++) {
         a = spectrum[j];
         b = (spectrum[j-i]+spectrum[j+i])/2.0;
         if (b<a) a=b;
         working_space[j]=a;
      }
      for(j=i;j<size-i;j++) spectrum[j]=working_space[j];
   }
   delete [] working_space;
   return(0);
}

//______________________________________________________________________________
char *TSpectrum::Background2(float **spectrum,int sizex,int sizey,int number_of_iterations)
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

   if (sizex <=0 || sizey <= 0) return (char*)"Wrong parameters";
       //   working_space-pointer to the working array
   float **working_space = new float* [sizex];
   int i,x,y;
   for(i=0;i<sizex;i++) working_space[i] = new float[sizey];
   float a,b,p1,p2,p3,p4,s1,s2,s3,s4;
   for(i=1;i<=number_of_iterations;i++) {
      for(y=i;y<sizey-i;y++) {
         for(x=i;x<sizex-i;x++) {
            a  = spectrum[x][y];
            p1 = spectrum[x-i][y-i];
            p2 = spectrum[x-i][y+i];
            p3 = spectrum[x+i][y-i];
            p4 = spectrum[x+i][y+i];
            s1 = spectrum[x][y-i];
            s2 = spectrum[x-i][y];
            s3 = spectrum[x+i][y];
            s4 = spectrum[x][y+i];
            b  = (p1+p2)/2.0; if (b>s2) s2 = b;
            b  = (p1+p3)/2.0; if (b>s1) s1 = b;
            b  = (p2+p4)/2.0; if (b>s4) s4 = b;
            b  = (p3+p4)/2.0; if (b>s3) s3 = b;
            s1 = s1-(p1+p3)/2.0;
            s2 = s2-(p1+p2)/2.0;
            s3 = s3-(p3+p4)/2.0;
            s4 = s4-(p2+p4)/2.0;
            b  = (s1+s4)/2.0+(s2+s3)/2.0+(p1+p2+p3+p4)/4.0;
            if (b<a) a = b;
            working_space[x][y] = a;
         }
      }
      for(y=i;y<sizey-i;y++) {
         for(x=i;x<sizex-i;x++) {
            spectrum[x][y] = working_space[x][y];
         }
      }
   }
   for(i=0;i<sizex;i++) delete [] working_space[i];
   delete [] working_space;
   return(0);
}

//______________________________________________________________________________
char *TSpectrum::Deconvolution1(float *source,float *resp,int size,int number_of_iterations)
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

   if (size <= 0) return (char*)"Wrong Parameters";
      //   working_space-pointer to the working vector
      //   (its size must be 6*size of source spectrum)
   double *working_space = new double[6*size];
   int i,j,k,lindex,posit,imin,imax,jmin,jmax,lh_gold;
   double lda,ldb,ldc,area,maximum;
   area    = 0;
   lh_gold = -1;
   posit   = 0;
   maximum = 0;
//read response vector
   for(i=0;i<size;i++){
      lda = resp[i];
      if(lda!=0) lh_gold = i+1;
      working_space[i] = lda;
      area += lda;
      if(lda>maximum) {
         maximum = lda;
         posit   = i;
      }
   }
   if(lh_gold==-1) return("ZERO RESPONSE VECTOR");
//read source vector
   for(i=0;i<size;i++) working_space[2*size+i] = source[i];
//create matrix at*a(vector b)
   i = lh_gold-1;
   if (i>size) i=size;
   imin = -i, imax = i;
   for(i=imin;i<=imax;i++) {
      lda  = 0;
      jmin = 0; if (i<0) jmin=-i;
      jmax = lh_gold-1-i; if (jmax>(lh_gold-1)) jmax=lh_gold-1;
      for(j=jmin;j<=jmax;j++) {
         ldb = working_space[j];
         ldc = working_space[i+j];
         lda = lda+ldb*ldc;
      }
      working_space[size+i-imin] = lda;
   }
//create vector p
   i    = lh_gold-1;
   imin = -i;
   imax = size+i-1;
   for(i=imin;i<=imax;i++) {
      lda = 0;
      for(j=0;j<=(lh_gold-1);j++) {
         ldb = working_space[j];
         k   = i+j;
         if (k>=0&&k<size) {
            ldc = working_space[2*size+k];
            lda = lda+ldb*ldc;
         }
      }
      working_space[4*size+i-imin] = lda;
   }
//move vector p
   for(i=imin;i<=imax;i++)
      working_space[2*size+i-imin]=working_space[4*size+i-imin];
//create at*a*at*y (vector ysc)
   for(i=0;i<size;i++) {
      lda  = 0;
      j    = lh_gold-1;
      jmin = -j;
      jmax = j;
      for(j=jmin;j<=jmax;j++) {
         ldb = working_space[j-jmin+size];
         ldc = working_space[2*size+i+j-jmin];
         lda = lda+ldb*ldc;
      }
      working_space[4*size+i] = lda;
   }
//move ysc
   for(i=0;i<size;i++)
      working_space[2*size+i]=working_space[4*size+i];
//create vector c//
   i    = 2*lh_gold-2; if (i>size) i=size;
   imin = -i;
   imax = i;
   for(i=imin;i<=imax;i++) {
      lda  = 0;
      jmin = -lh_gold+1+i; if (jmin<(-lh_gold+1)) jmin=-lh_gold+1;
      jmax = lh_gold-1+i;  if(jmax>(lh_gold-1))   jmax=lh_gold-1;
      for(j=jmin;j<=jmax;j++) {
         ldb = working_space[j+lh_gold-1+size];
         ldc = working_space[i-j+lh_gold-1+size];
         lda = lda+ldb*ldc;
      }
      working_space[i-imin] = lda;
   }
//move vector c
   for(i=0;i<size;i++)
      working_space[i+size] = working_space[i];
//initialization of resulting vector
   for(i=0;i<size;i++) working_space[i] = 1;
   //**START OF ITERATIONS**
   for(lindex=0;lindex<number_of_iterations;lindex++) {
      for(i=0;i<size;i++) {
         if (working_space[2*size+i]>0.000001&&working_space[i]>0.000001) {
            lda  = 0;
            jmin = 2*lh_gold-2; if(jmin>i) jmin=i;
            jmin = -jmin;
            jmax = 2*lh_gold-2; if(jmax>(size-1-i)) jmax=size-1-i;
            for(j=jmin;j<=jmax;j++) {
               ldb = working_space[j+2*lh_gold-2+size];
               ldc = working_space[i+j];
               lda = lda+ldb*ldc;
            }
            ldb = working_space[2*size+i];
            if (lda!=0) lda = ldb/lda;
            else        lda = 0;
            ldb = working_space[i];
            lda = lda*ldb;
            working_space[3*size+i] = lda;
         }
      }
      for(i=0;i<size;i++) working_space[i]=working_space[3*size+i];
   }
//shift resulting spectrum
   for(i=0;i<size;i++) {
      lda = working_space[i];
      j   = i+posit;
      j   = j%size;
      working_space[size+j] = lda;
   }
//write back resulting spectrum
   for(i=0;i<size;i++) source[i] = area*working_space[size+i];
   delete [] working_space;
   return(0);
}

//______________________________________________________________________________
char *TSpectrum::Deconvolution2(float** source,float** resp,int sizex,int sizey,int number_of_iterations)
{
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   This function calculates deconvolution from source spectrum           //
//   according to response spectrum                                        //
//   The result is placed in the matrix pointed by source pointer.         //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the matrix of source spectrum                     //
//   resp:    pointer to the matrix of response spectrum                   //
//   sizex:   x length of source and response spectra                      //
//   sizey:   y length of source and response spectra                      //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

   if (sizex <=0 || sizey <= 0) return (char*)"Wrong parameters";
      //   working_space-pointer to the working matrix
      //   (its size must be sizex*21*sizey of source spectrum)
   double **working_space = new double* [sizex];
   int i,j,k,lhx,lhy,i1,i2,j1,j2,k1,k2,lindex,i1min,i1max,i2min,i2max,j1min,j1max,j2min,j2max;
   for(i=0;i<sizex;i++) working_space[i] = new double[21*sizey];
   int positx=0, posity=0;
   double lda,ldb,ldc,area,maximum=0;
   area = 0;
   lhx  = - 1; lhy = -1;
   for(i=0;i<sizex;i++) {
      for(j=0;j<sizey;j++) {
         lda=resp[i][j];
         if (lda!=0){
            if ((i+1)>lhx) lhx = i+1;
            if ((j+1)>lhy) lhy = j+1;
         }
         working_space[i][j] = lda;
         area += lda;
         if (lda>maximum) {
            maximum = lda;
            positx  = i;
            posity  = j;
         }
      }
   }
   if (lhx==-1||lhy==-1) return("ZERO RESPONSE DATA");
//calculate at*y and write into p
   i2min = -lhy+1,i2max=sizey+lhy-2;
   i1min = -lhx+1,i1max=sizex+lhx-2;
   for(i2=i2min;i2<=i2max;i2++) {
      for(i1=i1min;i1<=i1max;i1++) {
         ldc = 0;
         for(j2=0;j2<=(lhy-1);j2++) {
            for(j1=0;j1<=(lhx-1);j1++) {
               k2 = i2+j2,k1=i1+j1;
               if (k2>=0&&k2<sizey&&k1>=0&&k1<sizex) {
                  lda = working_space[j1][j2];
                  ldb = source[k1][k2];
                  ldc = ldc+lda*ldb;
               }
            }
         }
         k = (i1+sizex)/sizex;
         working_space[(i1+sizex)%sizex][i2+sizey+sizey+k*3*sizey] = ldc;
      }
   }
//calculate matrix b=ht*h
   i1min = -(lhx-1),i1max=lhx-1;
   i2min = -(lhy-1),i2max=lhy-1;
   for(i2=i2min;i2<=i2max;i2++) {
      for(i1=i1min;i1<=i1max;i1++) {
         ldc   = 0;
         j2min = -i2;      if (j2min<0)     j2min = 0;
         j2max = lhy-1-i2; if (j2max>lhy-1) j2max = lhy-1;
         for(j2=j2min;j2<=j2max;j2++) {
            j1min = -i1;      if (j1min<0)     j1min = 0;
            j1max = lhx-1-i1; if (j1max>lhx-1) j1max = lhx-1;
            for(j1=j1min;j1<=j1max;j1++) {
               lda = working_space[j1][j2];
               ldb = working_space[i1+j1][i2+j2];
               ldc = ldc+lda*ldb;
            }
         }
         k = (i1+sizex)/sizex;
         working_space[(i1+sizex)%sizex][i2+sizey+10*sizey+k*2*sizey] = ldc;
      }
   }
//calculate ht*h*ht*y and write into ygold
   for(i2=0;i2<sizey;i2++) {
      for(i1=0;i1<sizex;i1++) {
         ldc = 0;
         j2min = i2min;
         j2max = i2max;
         for(j2=j2min;j2<=j2max;j2++) {
            j1min = i1min;
            j1max = i1max;
            for(j1=j1min;j1<=j1max;j1++) {
               k   = (j1+sizex)/sizex;
               lda = working_space[(j1+sizex)%sizex][j2+sizey+10*sizey+k*2*sizey];
               k   = (i1+j1+sizex)/sizex;
               ldb = working_space[(i1+j1+sizex)%sizex][i2+j2+sizey+sizey+k*3*sizey];
               ldc = ldc+lda*ldb;
            }
         }
         working_space[i1][i2+14*sizey] = ldc;
      }
   }
//calculate matrix cc
   i2    = 2*lhy-2; if (i2>sizey) i2 = sizey;
   i2min = -i2;
   i2max = i2;
   i1    = 2*lhx-2; if (i1>sizex) i1 = sizex;
   i1min = -i1;
   i1max = i1;
   for(i2=i2min;i2<=i2max;i2++) {
      for(i1=i1min;i1<=i1max;i1++) {
         ldc   = 0;
         j2min = -lhy+i2+1; if (j2min<-lhy+1) j2min = -lhy+1;
         j2max=lhy+i2-1;    if (j2max>lhy-1)  j2max =  lhy-1;
         for(j2=j2min;j2<=j2max;j2++) {
            j1min = -lhx+i1+1; if (j1min<-lhx+1) j1min = -lhx+1;
            j1max = lhx+i1-1;  if (j1max>lhx-1)  j1max =  lhx-1;
            for(j1=j1min;j1<=j1max;j1++) {
               k   = (j1+sizex)/sizex;
               lda = working_space[(j1+sizex)%sizex][j2+sizey+10*sizey+k*2*sizey];
               k   = (j1-i1+sizex)/sizex;
               ldb = working_space[(j1-i1+sizex)%sizex][j2-i2+sizey+10*sizey+k*2*sizey];
               ldc = ldc+lda*ldb;
            }
         }
         k = (i1+sizex)/sizex;
         working_space[(i1+sizex)%sizex][i2+sizey+15*sizey+k*2*sizey] = ldc;
      }
   }
//initialization in x1 matrix
   for(i2=0;i2<sizey;i2++) {
      for(i1=0;i1<sizex;i1++) {
         working_space[i1][i2+19*sizey] = 1;
         working_space[i1][i2+20*sizey] = 0;
      }
   }
   //**START OF ITERATIONS**
   for(lindex=0;lindex<number_of_iterations;lindex++) {
      for(i2=0;i2<sizey;i2++) {
         for(i1=0;i1<sizex;i1++) {
            lda = working_space[i1][i2+19*sizey];
            ldc = working_space[i1][i2+14*sizey];
            if (lda>0.000001&&ldc>0.000001) {
               ldb   = 0;
               j2min = i2; if (j2min>2*lhy-2) j2min = 2*lhy-2;
               j2min = -j2min;
               j2max = sizey-i2-1; if (j2max>2*lhy-2) j2max = 2*lhy-2;
               j1min = i1; if (j1min>2*lhx-2) j1min=2*lhx-2;
               j1min = -j1min;
               j1max = sizex-i1-1; if (j1max>2*lhx-2) j1max=2*lhx-2;
               for(j2=j2min;j2<=j2max;j2++) {
                  for(j1=j1min;j1<=j1max;j1++) {
                     k   = (j1+sizex)/sizex;
                     ldc = working_space[(j1+sizex)%sizex][j2+sizey+15*sizey+k*2*sizey];
                     lda = working_space[i1+j1][i2+j2+19*sizey];
                     ldb = ldb+lda*ldc;
                  }
               }
               lda = working_space[i1][i2+19*sizey];
               ldc = working_space[i1][i2+14*sizey];
               if (ldc*lda!=0&&ldb!=0) lda = lda*ldc/ldb;
               else                    lda = 0;
               working_space[i1][i2+20*sizey] = lda;
            }
         }
      }
      for(i2=0;i2<sizey;i2++) {
         for(i1=0;i1<sizex;i1++)
            working_space[i1][i2+19*sizey]=working_space[i1][i2+20*sizey];
      }
   }
   for(i=0;i<sizex;i++) {
      for(j=0;j<sizey;j++)
         source[(i+positx)%sizex][(j+posity)%sizey]=area*working_space[i][j+19*sizey];
   }
   for(i=0;i<sizex;i++) delete [] working_space[i];
   delete [] working_space;
   return(0);
}

//______________________________________________________________________________
Int_t TSpectrum::Search(TH1 *hin, Double_t sigma, Option_t *option)
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
      Error("Search","Only implemented for 1-d and 2-d histograms");
      return 0;
   }
   if (dimension == 1 ) {
      Int_t size = hin->GetXaxis()->GetNbins();
      Int_t i, bin, npeaks;
      Float_t *source = new float [size];
      for (i=0;i<size;i++) source[i] = hin->GetBinContent(i+1);

      npeaks=Search1(source,size,sigma);
      for(i=0;i<npeaks;i++) {
         bin = 1+Int_t(fPositionX[i] +0.5);
         fPositionX[i] = hin->GetBinCenter(bin);
         fPositionY[i] = hin->GetBinContent(bin);
      }
      if (strstr(option,"goff")) return npeaks;
      TPolyMarker *pm = new TPolyMarker(npeaks,fPositionX, fPositionY);
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
Int_t TSpectrum::Search1(float *spectrum,int size,double sigma)
{
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   This function searches for peaks in source spectrum                   //
//   The number of found peaks and their positions are written into        //
//   the members fNpeaks and fPositionX.                                   //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the vector of source spectrum                     //
//   size:    length of source spectrum                                    //
//   sigma:   sigma of searched peaks, for details we refer to manual      //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
   int xmin,xmax,i,j,l,i1,i2,i3,i5,n1,n2,n3,stav,peak_index,lmin,lmax;
   i1 = i2 = i3 = 0;
   double a,b,s,f,si4,fi4,suma,sumai,sold,fold=0,norma,filter[PEAK_WINDOW];
   si4 = fi4 = 0;
   for(i=0;i<PEAK_WINDOW;i++) filter[i]=0;
   j=(int)(3.0*sigma);
   for(i=-j;i<=j;i++) {
      a = i;
      a = -a*a;
      b = 2.0*sigma*sigma;
      a = a/b;
      a = exp(a);
      s = i;
      s = s*s;
      s = s-sigma*sigma;
      s = s/(sigma*sigma*sigma*sigma);
      s = s*a;
      filter[PEAK_WINDOW/2+i]=s;
   }
   norma = 0;
   for(i=0;i<PEAK_WINDOW;i++) norma=norma+TMath::Abs(filter[i]);
   for(i=0;i<PEAK_WINDOW;i++) filter[i]=filter[i]/norma;
   suma  = 0;
   sumai = 0;
   stav  =1 ;
   peak_index = 0;
   sold = PEAK_WINDOW/2;
   xmin = (int)(3.0*sigma);
   xmax = size-(int)(3.0*sigma);
   lmin = PEAK_WINDOW/2-(int)(3.0*sigma);
   lmax = PEAK_WINDOW/2+(int)(3.0*sigma);
   for(i=xmin;i<=xmax;i++) {
       s = 0;
       f = 0;
      for(l=lmin;l<=lmax;l++) {
         if(i+l-PEAK_WINDOW/2 >= size) break;
         a  = spectrum[i+l-PEAK_WINDOW/2];
         s += a*filter[l];
         f += a*filter[l]*filter[l];
      }
      f = sqrt(f);
      if (s<0) {
         a      = i;
         a     *= s;
         suma  += s;
         sumai += a;
      }
      if ((stav==1)&&(s>f)) {
stav1:
         stav = 2;
         suma = 0;
         sumai= 0;
         i1   = i;
      }
      else if ((stav==2)&&(s<=f)) {
         stav = 3;
         i2   = i;
      }
      else if(stav==3) {
         if (s>f)
            goto stav1;
         if (s<=0) {
            stav = 4;
            i3   = i;
         }
      }
      else if((stav==4)&&(s>=sold)) {
         si4 = sold;
         fi4 = fold;
         stav= 5;
      }
      else if((stav==5)&&(s>=0)) {
         stav = 6;
         i5   = i;
         if (si4==0)
            stav = 0;
         else{
            n1 = i5-i3+1;
            a  = n1+2;
            a  = fi4*a/(2.*si4)+1/2.;
            a  = TMath::Abs(a);
            n2 = (int)a;
            a  = n1-4; if (a<0) a=0;
            a  = a*(1-2.*(fi4/si4))+1/2.;
            a  = TMath::Abs(a);
            n3 = (int)(a/fResolution);
            a  = TMath::Abs(si4); if(a<=(2.*fi4)) stav=0;
            if (n2>=1) {
               if ((i3-i2-1)>n2)
                  stav = 0;
            }
            else {
               if ((i3-i2-1)>1) stav = 0;
            }
            if ((i2-i1+1)<n3) stav=0;
         }
         if (stav!=0) {
            b = sumai/suma;
            if (peak_index < fMaxPeaks) {
               fPositionX[peak_index] = b;
               peak_index += 1;
            } else {
               Warning("Search1","PEAK BUFFER FULL");
               return 0;
            }
         }
         stav  = 1;
         suma  = 0;
         sumai = 0;
      }
      sold = s;
      fold = f;
   }
   fNPeaks = peak_index;
   return fNPeaks;
}

//______________________________________________________________________________
Int_t TSpectrum::PeakEvaluate(double *temp,int size,int xmax,double xmin)
{
   int i,i1,i2,i3,i4,i5,n1,n2,n3,stav,peak_index;
   double a,b,s,f,si4,fi4,suma,sumai,sold,fold=0;
   i1 = i2 = i3 = i4 = 0;
   si4 = fi4 = 0;
   stav=1;
   peak_index=0;
   sold = 1000000.0;
   suma = 0;
   sumai= 0;
   for(i=0;i<xmax;i++) {
      s = temp[i],f=temp[i+size];
      if((s<0)&&(stav>=2)&&(stav<=5)) {
         a     = i+xmin;
         a    *= s;
         suma += s;
         sumai+= a;
      }
      if ((stav==1)&&(s>f)) {
         stav = 2;
         i1   = i;
      }
      else if((stav==2)&&(s<=f)) {
         stav = 3 ;
         i2   = i;
      }
      else if(stav==3) {
         if (s<=0) {
            stav = 4;
            i3   = i;
         }
      }
      else if((stav==4)&&(s>=sold)) {
         si4  = sold;
         fi4  = fold;
         stav = 5;
         i4   = i-1;
      }
      else if((stav==5)&&(s>=0)) {
         stav = 6;
         i5   = i;
         if (si4==0)
            stav = 0;
         else{
            n1 = i5-i3+1;
            a  = n1+2;
            a  = fi4*a/(2.*si4)+1/2.;
            a  = TMath::Abs(a);
            n2 = (int)a;
            a  = n1-4; if(a<0) a=0;
            a  = a*(1-2.*(fi4/si4))+1/2.;
            a  = TMath::Abs(a);
            n3 = (int)(a/fResolution);
            a  = TMath::Abs(si4);
            if (a<=(2.0*fi4))
               stav = 0;
            if (n2>=1) {
               if ((i3-i2-1)>n2) stav=0;
            }
            else{
               if((i3-i2-1)>1) stav=0;
            }
            if((i2-i1+1)<n3) stav=0;
            n1 = i5-i3+1;
            a  = n1+2;
            a  = fi4*a/(2.*si4)+1/2.;
            a  = TMath::Abs(a);
            n2 = (int)a;
            a  = n1-2;
            a  = a*(1-2.*(fi4/si4))+1/2.;
            a  = TMath::Abs(a);
            n3 = (int)(a/fResolution);
            a  = TMath::Abs(si4);
            if (a<=(2.*fi4)) stav=0;
            if (n2>=1) {
               if ((i3-i2-1)>n2) stav=0;
            }
            else {
               if ((i3-i2-1)>1) stav=0;
            }
            if (temp[0]<temp[size]) {
               if ((i2-i1+1)<n3) stav=0;
            }
         }
         if (stav!=0) {
            if (suma!=0) b = sumai/suma;
            else         b = i4+xmin;
            if (peak_index >= fMaxPeaks)
               return(-1);
            else{
               fPosition[peak_index] = b;
               peak_index += 1;
            }
         }
         stav  = 1;
         suma  = 0;
         sumai = 0;
         i     = i4;
      }
      sold = s;
      fold = f;
   }
   fNPeaks = peak_index;
   return fNPeaks;
}

//______________________________________________________________________________
Int_t TSpectrum::Search2(float **source,int sizex,int sizey,double sigma)
{
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   This function searches for peaks in source spectrum                   //
//   The number of found peaks and their positions are written into        //
//   the members fNPeaks, fPositionX and fPositionY.                       //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the vector of source spectrum                     //
//   sizex:   x length of source spectrum                                  //
//   sizey:   y length of source spectrum                                  //
//   sigma:   sigma of searched peaks, for details we refer to manual      //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

   if (sizex <=0 || sizey <= 0) return -1;
      //   working_space-pointer to the working matrix
      //         (its size must be sizex*2*sizey of source spectrum)
      //   working_vector_x-pointer to the working vector x
      //         (its size must be 2*sizex of source spectrum)
      //   working_vector_y-pointer to the working vector y
      //         (its size must be 2*sizey of source spectrum)
   double **working_space   = new double* [sizex];
   double *working_vector_x = new double [2*sizex];
   double *working_vector_y = new double [2*sizey];
   double a,b,s,f,dpeakx,dpeaky,dxmin,dxmax,dymin,dymax,filter[PEAK_WINDOW],norma,val,val1,val2,val3,val4,val5,val6,val7,val8;
   int x,y,n,priz,polx,poly,peak_index=0,i,j,li,lj,lmin,lmax,xmin,xmax,ymin,ymax;
   for(i=0;i<sizex;i++) working_space[i] = new double[2*sizey];
   polx = poly = 0;
   double pocet_sigma = 5;
   for(j=0;j<sizey;j++) {
      for(i=0;i<sizex;i++) {
         working_space[i][j] = 0;
         working_space[i][j+sizey] = 0;
      }
   }
   for(i=0;i<PEAK_WINDOW;i++) filter[i] = 0;
   j = (int)(pocet_sigma*sigma+0.5);
   for(i=-j;i<=j;i++) {
      a = i;
      a = -a*a;
      b = 2.0*sigma*sigma;
      a = a/b;
      a = exp(a);
      s = i;
      s = s*s;
      s = s-sigma*sigma;
      s = s/(sigma*sigma*sigma*sigma);
      s = s*a;
      filter[PEAK_WINDOW/2+i] = s;
   }
   norma = 0;
   for(i=0;i<PEAK_WINDOW;i++) {
      norma = norma+TMath::Abs(filter[i]);
   }
   for(i=0;i<PEAK_WINDOW;i++) {
      filter[i] = filter[i]/norma;
   }
   a = pocet_sigma*sigma+0.5;
   i = (int)a;
   ymin = i;
   ymax = sizey-i;
   xmin = i;
   xmax = sizex-i;
   lmin = PEAK_WINDOW/2-i;
   lmax = PEAK_WINDOW/2+i;
   for(i=xmin;i<xmax;i++) {
      for(j=ymin;j<ymax;j++) {
         s = 0;
         f = 0;
         for(li=lmin;li<=lmax;li++) {
            for(lj=lmin;lj<=lmax;lj++) {
               a  = source[j+lj-PEAK_WINDOW/2][i+li-PEAK_WINDOW/2];
               s += a*filter[li]*filter[lj];
               f += a*filter[li]*filter[li]*filter[lj]*filter[lj];
            }
         }
         f = sqrt(f);
         working_space[i][j] = s;
         working_space[i][j+sizey] = f;
      }
   }
   for(x=xmin;x<xmax;x++) {
      for(y=ymin+1;y<ymax;y++) {
       val  = working_space[x][y];
       val1 = working_space[x-1][y-1];
       if (val>=val1) {
        val2 = working_space[x][y-1];
        if (val>=val2) {
         val3 = working_space[x+1][y-1];
         if (val>=val3) {
          val4 = working_space[x-1][y];
          if (val>=val4) {
           val5 = working_space[x+1][y];
           if (val>=val5) {
            val6 = working_space[x-1][y+1];
            if (val>=val6) {
             val7 = working_space[x][y+1];
             if (val>=val7) {
              val8 = working_space[x+1][y+1];
              if (val>=val8) {
               if (val!=val1||val!=val2||val!=val3||val!=val4||val!=val5||val!=val6||val!=val7||val!=val8) {
                priz = 0;
                for(j=0;(j<peak_index)&&(priz==0);j++) {
                   dxmin = fPositionX[j]-sigma;
                   dxmax = fPositionX[j]+sigma;
                   dymin = fPositionY[j]-sigma;
                   dymax = fPositionY[j]+sigma;
                   if ((x>=dxmin)&&(x<=dxmax)&&(y>=dymin)&&(y<=dymax)) priz=1;
                }
                if (priz==0) {
                   s = 0;
                   f = 0;
                   for(li=lmin;li<=lmax;li++) {
                      a  = source[y][x+li-PEAK_WINDOW/2];
                      s += a*filter[li];
                      f += a*filter[li]*filter[li];
                   }
                   f = sqrt(f);
                   if (s<f) {
                      s = 0;
                      f = 0;
                      for(li=lmin;li<=lmax;li++) {
                         a  = source[y+li-PEAK_WINDOW/2][x];
                         s += a*filter[li];
                         f += a*filter[li]*filter[li];
                      }
                      f = sqrt(f);
                      if (s<f) {
                       for(i=x+lmin-PEAK_WINDOW/2;i<=x+lmax-PEAK_WINDOW/2;i++) {
                          working_vector_x[i-x-lmin+PEAK_WINDOW/2] = -working_space[i][y];
                          working_vector_x[i-x-lmin+PEAK_WINDOW/2+sizex] = working_space[i][y+sizey];
                       }
          //find peaks in y-th column
                       n = PeakEvaluate(working_vector_x,sizex,lmax-lmin+1,x+lmin-PEAK_WINDOW/2);
                       if (n==-1) {
                          Warning("Search2","TOO MANY PEAKS IN ONE COLUMN");
                          return -1;
                       }
                       if (n!=0) {
                          val=sizex;
                          for(i=0;i<n;i++) {
                             a = fPosition[i];
                             a = TMath::Abs(a-x);
                             if (a<val) {
                                val  = a;
                                polx = i;
                             }
                          }
                          dpeakx = fPosition[polx];
                          for(i=y+lmin-PEAK_WINDOW/2;i<=y+lmax-PEAK_WINDOW/2;i++) {
                             working_vector_y[i-y-lmin+PEAK_WINDOW/2]=-working_space[x][i];
                             working_vector_y[i-y-lmin+PEAK_WINDOW/2+sizey]=working_space[x][i+sizey];
                           }
           //find peaks in x-th row
                           n = PeakEvaluate(working_vector_y,sizey,lmax-lmin+1,y+lmin-PEAK_WINDOW/2);
                           if (n==-1) {
                              Warning("Search2","TOO MANY PEAKS IN ONE ROW");
                              return -1;
                           }
                           if (n!=0) {
                              val = sizey;
                              for(i=0;i<n;i++) {
                                 a = fPosition[i];
                                 a = TMath::Abs(a-y);
                                 if (a<val) {
                                    val  = a;
                                    poly = i;
                                 }
                              }
                              dpeaky = fPosition[poly];
                              if (peak_index < fMaxPeaks) {
                                 fPositionX[peak_index] = dpeakx;
                                 fPositionY[peak_index] = dpeaky;
                                 peak_index += 1;
                              } else {
                                 Warning("Search2","PEAK BUFFER FULL");
                                 return 0;
                              }
                           }
                        }
                      }
                    }
                  }
                }
              }
             }
            }
           }
          }
         }
        }
       }
      }
   }
   for(i=0;i<sizex;i++) delete [] working_space[i];
   delete [] working_space;
   delete [] working_vector_x;
   delete [] working_vector_y;

   fNPeaks = peak_index;
   return fNPeaks;
}

//______________________________________________________________________________
void TSpectrum::SetResolution(Float_t resolution)
{
//  resolution: determines resolution of the neighboring peaks 
//              default value is 1 correspond to 3 sigma distance
//              between peaks. Higher values allow higher resolution 
//              (smaller distance between peaks.
//              May be set later through SetResolution.

   if (resolution > 1) fResolution = resolution;
   else                fResolution = 1;
}
