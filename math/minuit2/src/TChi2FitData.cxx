// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include <cassert>

#include "RConfig.h"
#include "TChi2FitData.h"

#include "TVirtualFitter.h" 

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

#include "TList.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"





TChi2FitData::TChi2FitData( const TVirtualFitter & fitter, bool skipEmptyBins) : 
fSize(0), fSkipEmptyBins(skipEmptyBins), fIntegral(false)
{
   // constructor - create fit data from fitter ROOT data object (histogram, graph, etc...)
   // need a pointer to TF1 (model function) for avoid points in regions rejected by TF1 
   // option skipEmptyBins is used to skip the bin with zero content or to use them in the fit 
   // (in that case an error of 1 is set)
   
   TF1 * func = dynamic_cast<TF1 *> ( fitter.GetUserFunc() );  
   assert( func != 0);
   
   TObject * obj = fitter.GetObjectFit(); 
   // downcast to see type of object
   TH1 * hfit = dynamic_cast<TH1*> ( obj );
   if (hfit) { 
      GetFitData(hfit, func, &fitter);    
      return; 
   } 
   // case of TGraph
   TGraph * graph = dynamic_cast<TGraph*> ( obj );
   if (graph) { 
      GetFitData(graph, func, &fitter);    
      return; 
   } 
   // case of TGraph2D
   TGraph2D * graph2D = dynamic_cast<TGraph2D*> ( obj );
   if (graph2D) { 
      GetFitData(graph2D, func, &fitter);    
      return; 
   } 
   // case of TMultiGraph
   TMultiGraph * multigraph = dynamic_cast<TMultiGraph*> ( obj );
   if (multigraph) { 
      GetFitData(graph2D, func, &fitter);    
      return; 
   } 
   // else 
#ifdef DEBUG
   std::cout << "other fit type are not yet supported- assert" << std::endl;
#endif
   assert(hfit != 0); 
   
}


void TChi2FitData::GetFitData(const TH1 * hfit, const TF1 * func, const TVirtualFitter * hFitter) {
   // get Histogram Data
   
   assert(hfit != 0); 
   assert(hFitter != 0);
   assert(func != 0);
   
   //std::cout << "creating Fit Data from histogram " << hfit->GetName() << std::endl; 
   
   //  use TVirtual fitter to get fit range (should have a FitRange class ) 
   
   // first and last bin
   int hxfirst = hFitter->GetXfirst(); 
   int hxlast  = hFitter->GetXlast(); 
   int hyfirst = hFitter->GetYfirst(); 
   int hylast  = hFitter->GetYlast(); 
   int hzfirst = hFitter->GetZfirst(); 
   int hzlast  = hFitter->GetZlast(); 
   TAxis *xaxis  = hfit->GetXaxis();
   TAxis *yaxis  = hfit->GetYaxis();
   TAxis *zaxis  = hfit->GetZaxis();
   
   // get fit option 
   Foption_t fitOption = hFitter->GetFitOption();
   if (fitOption.Integral) fIntegral=true;
   
   int n = (hxlast-hxfirst+1)*(hylast-hyfirst+1)*(hzlast-hzfirst+1); 
   
#ifdef DEBUG
   std::cout << "TChi2FitData: ifirst = " << hxfirst << " ilast =  " << hxlast 
             << "total bins  " << hxlast-hxfirst+1  
             << "skip empty bins "  << fSkipEmptyBins << std::endl; 
#endif
   
   fInvErrors.reserve(n);
   fValues.reserve(n);
   fCoordinates.reserve(n);
   
   int ndim =  hfit->GetDimension();
   assert( ndim > 0 );
   CoordData x = CoordData( hfit->GetDimension() );
   int binx = 0; 
   int biny = 0; 
   int binz = 0; 
   
   for ( binx = hxfirst; binx <= hxlast; ++binx) {
      if (fIntegral) {
         x[0] = xaxis->GetBinLowEdge(binx);       
      }
      else
         x[0] = xaxis->GetBinCenter(binx);
      
      if ( ndim > 1 ) { 
         for ( biny = hyfirst; biny <= hylast; ++biny) {
            if (fIntegral) 
               x[1] = yaxis->GetBinLowEdge(biny);
            else
               x[1] = yaxis->GetBinCenter(biny);
            
            if ( ndim >  2 ) { 
               for ( binz = hzfirst; binz <= hzlast; ++binz) {
                  if (fIntegral) 
                     x[2] = zaxis->GetBinLowEdge(binz);
                  else
                     x[2] = zaxis->GetBinCenter(binz);
                  if (!func->IsInside(&x.front()) ) continue;
                  double error =  hfit->GetBinError(binx, biny, binz); 
                  if (fitOption.W1) error = 1;
                  SetDataPoint( x,  hfit->GetBinContent(binx, biny, binz), error );
               }  // end loop on z bins
            }
            else if (ndim == 2) { 
               // for dim == 2
               if (!func->IsInside(&x.front()) ) continue;
               double error =  hfit->GetBinError(binx, biny); 
               if (fitOption.W1) error = 1;
               SetDataPoint( x,  hfit->GetBinContent(binx, biny), error );
            }   
            
         }  // end loop on y bins
         
      }
      else if (ndim == 1) { 
         // for 1D 
         if (!func->IsInside(&x.front()) ) continue;
         double error =  hfit->GetBinError(binx); 
         if (fitOption.W1) error = 1;
         SetDataPoint( x,  hfit->GetBinContent(binx), error );
      }
      
   }   // end 1D loop 
   
   // in case of integral store additional point with upper x values
   if (fIntegral) { 
      x[0] = xaxis->GetBinLowEdge(hxlast) +  xaxis->GetBinWidth(hxlast); 
      if (ndim > 1) { 
         x[1] = yaxis->GetBinLowEdge(hylast) +  yaxis->GetBinWidth(hylast); 
      }
      if (ndim > 2) { 
         x[2] = zaxis->GetBinLowEdge(hzlast) +  zaxis->GetBinWidth(hzlast); 
      }
      fCoordinates.push_back(x);
   }
   
#ifdef DEBUG
   std::cout << "TChi2FitData: Hist FitData size is " << fCoordinates.size() << std::endl;
#endif
   
}


void TChi2FitData::GetFitData(const TGraph * gr, const TF1 * func, const TVirtualFitter * hFitter ) {
   // get TGraph data (neglect error in x, that is used in the extended method)
   
   assert(gr != 0); 
   assert(hFitter != 0);
   assert(func != 0);
   
   // fit options
   Foption_t fitOption = hFitter->GetFitOption();
   
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();
   
   CoordData x = CoordData( 1 );  // 1D graph
   
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];
      // neglect error in x (it is a different chi2) 
      if (!func->IsInside(&x.front()) ) continue;
      double errorY = gr->GetErrorY(i); 
      // consider error = 0 as 1 
      if (errorY <= 0) errorY = 1;
      if (fitOption.W1) errorY = 1;
      SetDataPoint( x, gy[i], errorY );
      
   }
}


void TChi2FitData::GetFitData(const TGraph2D * gr, const TF1 * func, const TVirtualFitter * hFitter ) {
   // fetch graph 2D data for CHI2 fit. 
   // neglect errors in x and y (one use the ExtendedChi2 method)
   
   assert(gr != 0); 
   assert(hFitter != 0);
   assert(func != 0);
   
   // fit options
   Foption_t fitOption = hFitter->GetFitOption();
   
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();
   double *gz = gr->GetZ();
   
   CoordData x = CoordData( 2 );  // 2D graph
   
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];
      x[1] = gy[i];
      if (!func->IsInside(&x.front()) ) continue;
      // neglect error in x (it is a different chi2) 
      double error = gr->GetErrorZ(i); 
      // consider error = 0 as 1 
      if (error <= 0) error = 1;
      if (fitOption.W1) error = 1;
      SetDataPoint( x, gz[i], error );
      
   }
}



void TChi2FitData::GetFitData(const TMultiGraph * mg, const TF1 * func, const TVirtualFitter * hFitter ) {
   // data from a multigraph
   
   assert(mg != 0); 
   assert(hFitter != 0);
   assert(func != 0);
   
   // fit options
   Foption_t fitOption = hFitter->GetFitOption();
   
   TGraph *gr;
   TIter next(mg->GetListOfGraphs());   
   
   int  nPoints;
   double *gx;
   double *gy;
   
   CoordData x = CoordData( 1 );  // 1D graph
   
   while ((gr = (TGraph*) next())) {
      nPoints = gr->GetN();
      gx      = gr->GetX();
      gy      = gr->GetY();
      for ( int i = 0; i < nPoints; ++i) { 
         
         x[0] = gx[i];
         // neglect error in x (it is a different chi2) 
         if (!func->IsInside(&x.front()) ) continue;
         double errorY = gr->GetErrorY(i); 
         // consider error = 0 as 1 
         if (errorY <= 0) errorY = 1;
         if (fitOption.W1) errorY = 1;
         SetDataPoint( x, gy[i], errorY );
         
      }
   }
   
}

void TChi2FitData::SetDataPoint( const CoordData & x, double y, double error) { 
   // set internally the data in the internal vectors and count them
   // if error is <=0 (like for zero content bin) skip or set to 1, according to the set option
   if (error <= 0) {  
      if (SkipEmptyBins() ) 
         return;
      else   
         // set errors to 1 
         error = 1;
   }
   
   fCoordinates.push_back(x);
   fValues.push_back(y);
   fInvErrors.push_back(1./error);
   fSize++;
}
