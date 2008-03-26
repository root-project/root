// @(#)root/hist:$Id$
// Author: L. Moneta Thu Aug 31 10:40:20 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TH1Interface

#include "THFitInterface.h"

#include "Fit/BinData.h"

//#include "Fit/BinPoint.h"

//#define DEBUG
#ifdef DEBUG
#include <iostream> 
#endif

#include <cassert> 

#include "TH1.h"
#include "TF1.h"
#include "TError.h"
#include "TGraph2D.h"

#include <cmath>

namespace ROOT { 

namespace Fit { 

// add a namespace to distinguish from the Graph functions 
namespace HInterface { 


bool IsPointOutOfRange(const TF1 * func, const double * x) { 
   // function to check if a point is outside range
   if (func ==0) return false; 
   return !func->IsInside(x);       
}

bool AdjustError(const DataOptions & option, double & error) {
   // adjust the given error accoring to the option
   //  if false is returned bin must be skipped 
   if (option.fErrors1) error = 1;
   if (error <= 0 ) { 
      if (option.fUseEmpty) 
         error = 1.; // set error to 1 for empty bins 
      else 
         return false; 
   }
   return true; 
}

} // end namespace HInterface

void FillData(BinData & dv, const TH1 * hfit, TF1 * func) 
{
   // Function to fill the binned Fit data structure from a TH1 
   // The dimension of the data is the same of the histogram dimension
   // The funciton pointer is need in case of integral is used and to reject points 
   // rejected in the function

   // the TF1 pointer cannot be constant since EvalPar and InitArgs are not const methods
   
   // get fit option 
   const DataOptions & fitOpt = dv.Opt();

   
   assert(hfit != 0); 
   
   //std::cout << "creating Fit Data from histogram " << hfit->GetName() << std::endl; 

   int hxfirst = hfit->GetXaxis()->GetFirst();
   int hxlast  = hfit->GetXaxis()->GetLast();

   int hyfirst = hfit->GetYaxis()->GetFirst();
   int hylast  = hfit->GetYaxis()->GetLast();

   int hzfirst = hfit->GetZaxis()->GetFirst();
   int hzlast  = hfit->GetZaxis()->GetLast();

   // function by default has same range (use that one if requested otherwise use data one)

   
   //  get the range (add the function range ??)
   // to check if inclusion/excluion at end/point
   const DataRange & range = dv.Range(); 
   if (range.Size(0) != 0) { 
      double xlow   = range(0).first; 
      double xhigh  = range(0).second; 
#ifdef DEBUG
      std::cout << "xlow " << xlow << " xhigh = " << xhigh << std::endl;
#endif
      hxfirst =  hfit->GetXaxis()->FindBin(xlow);
      hxlast  =  hfit->GetXaxis()->FindBin(xhigh);
      if (range.Size(0) > 1  ) 
         Warning("ROOT::Fit::THFitInterface","support only one range interval for X coordinate"); 
   }

   if (hfit->GetDimension() > 1 && range.Size(1) != 0) { 
      double ylow   = range(1).first; 
      double yhigh  = range(1).second; 
      hyfirst =  hfit->GetYaxis()->FindBin(ylow);
      hylast  =  hfit->GetYaxis()->FindBin(yhigh);
      if (range.Size(0) > 1  ) 
         Warning("ROOT::Fit::THFitInterface","support only one range interval for Y coordinate"); 
   }

   if (hfit->GetDimension() > 2 && range.Size(2) != 0) { 
      double zlow   = range(2).first; 
      double zhigh  = range(2).second; 
      hzfirst =  hfit->GetZaxis()->FindBin(zlow);
      hzlast  =  hfit->GetZaxis()->FindBin(zhigh);
      if (range.Size(0) > 1  ) 
         Warning("ROOT::Fit::THFitInterface","support only one range interval for Z coordinate"); 
   }
   
   
   int n = (hxlast-hxfirst+1)*(hylast-hyfirst+1)*(hzlast-hzfirst+1); 
   if (fitOpt.fIntegral) n += 1;
   
#ifdef DEBUG
   std::cout << "THFitInterface: ifirst = " << hxfirst << " ilast =  " << hxlast 
             << " total bins  " << hxlast-hxfirst+1  
             << std::endl; 
#endif
   
   // reserve n for more efficient usage
   //dv.Data().reserve(n);
   
   int ndim =  hfit->GetDimension();
   assert( ndim > 0 );
   //typedef  BinPoint::CoordData CoordData; 
   //CoordData x = CoordData( hfit->GetDimension() );
   dv.Initialize(n,ndim); 
   std::vector<double> x(ndim); 

   int binx = 0; 
   int biny = 0; 
   int binz = 0; 

   TAxis *xaxis  = hfit->GetXaxis();
   TAxis *yaxis  = hfit->GetYaxis();
   TAxis *zaxis  = hfit->GetZaxis();

   
   for ( binx = hxfirst; binx <= hxlast; ++binx) {
      if (fitOpt.fIntegral) {
         x[0] = xaxis->GetBinLowEdge(binx);       
      }
      else
         x[0] = xaxis->GetBinCenter(binx);
      

      // need to evaluate function to know about rejected points
      // hugly but no other solutions
      if (func != 0) { 
         func->RejectPoint(false);
         func->InitArgs( &x[0],0 ); 
         func->EvalPar( &x[0],0 ); 
         if (func->RejectedPoint() ) continue; 
      }

      if ( ndim > 1 ) { 
         for ( biny = hyfirst; biny <= hylast; ++biny) {
            if (fitOpt.fIntegral) 
               x[1] = yaxis->GetBinLowEdge(biny);
            else
               x[1] = yaxis->GetBinCenter(biny);
            
            if ( ndim >  2 ) { 
               for ( binz = hzfirst; binz <= hzlast; ++binz) {
                  if (fitOpt.fIntegral) 
                     x[2] = zaxis->GetBinLowEdge(binz);
                  else
                     x[2] = zaxis->GetBinCenter(binz);
                  if (fitOpt.fUseRange && HInterface::IsPointOutOfRange(func,&x.front()) ) continue;
                  double error =  hfit->GetBinError(binx, biny, binz); 
                  if (!HInterface::AdjustError(fitOpt,error) ) continue; 
                  //dv.Add(BinPoint(  x,  hfit->GetBinContent(binx, biny, binz), error ) );
                  dv.Add(   &x.front(),  hfit->GetBinContent(binx, biny, binz), error  );
               }  // end loop on z bins
            }
            else if (ndim == 2) { 
               // for dim == 2
               if (fitOpt.fUseRange && HInterface::IsPointOutOfRange(func,&x.front()) ) continue;
               double error =  hfit->GetBinError(binx, biny); 
               if (!HInterface::AdjustError(fitOpt,error) ) continue; 
               dv.Add( &x.front(), hfit->GetBinContent(binx, biny), error  );
            }   
            
         }  // end loop on y bins
         
      }
      else if (ndim == 1) { 
#ifdef DEBUG
         std::cout << "bin " << binx << " add point " << x[0] << "  " << hfit->GetBinContent(binx) << std::endl;
#endif
         // for 1D 
         if (fitOpt.fUseRange && HInterface::IsPointOutOfRange(func,&x.front()) ) continue;
         double error =  hfit->GetBinError(binx); 
         if (!HInterface::AdjustError(fitOpt,error) ) continue; 
         dv.Add( x.front(),  hfit->GetBinContent(binx), error  );
      }
      
   }   // end 1D loop 
   
   // in case of integral store additional point with upper x values 
   if (fitOpt.fIntegral) { 
      x[0] = xaxis->GetBinLowEdge(hxlast) +  xaxis->GetBinWidth(hxlast); 
      if (ndim > 1) { 
         x[1] = yaxis->GetBinLowEdge(hylast) +  yaxis->GetBinWidth(hylast); 
      }
      if (ndim > 2) { 
         x[2] = zaxis->GetBinLowEdge(hzlast) +  zaxis->GetBinWidth(hzlast); 
      }
      //dv.Add(BinPoint( x, 0, 1.) ); // use dummy y= 0  &  err =1  for this extra point needed for integral
      dv.Add( &x.front() , 0, 1. ); // use dummy y= 0  &  err =1  for this extra point needed for integral
   }
   
#ifdef DEBUG
   std::cout << "THFitInterface::FillData: Hist FitData size is " << dv.Size() << std::endl;
#endif
   
}

void FillData ( BinData  & dv, const TGraph2D * gr, TF1 * func ) {  
   //  fill the data vector from a TGraph2D. Pass also the TF1 function which is 
   // needed in case to exclude points rejected by the function
   // in case of a pure TGraph 
   assert(gr != 0); 

   // get fit option 
   DataOptions & fitOpt = dv.Opt();
   
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();
   double *gz = gr->GetZ();
   
   // if all errors are zero set option of using errors to 1
   if ( gr->GetEZ() == 0) fitOpt.fErrors1 = true;
   
   double x[2]; 
   dv.Initialize(nPoints,2); 
   
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];
      x[1] = gy[i];
      if (!func->IsInside( x ) ) continue;
      // neglect error in x and y (it is a different chi2) 
      double error = gr->GetErrorZ(i); 
      if (!HInterface::AdjustError(fitOpt,error) ) continue; 
      dv.Add( x, gz[i], error );      
   }

#ifdef DEBUG
   std::cout << "THFitInterface::FillData Graph FitData size is " << dv.Size() << std::endl;
#endif

}


//______________________________________________________________________________
void InitGaus(const ROOT::Fit::BinData & data, TF1 * f1)
{
   //   -*-*-*-*Compute Initial values of parameters for a gaussian
   //           derivaed from function H1InitGaus defined in TH1.cxx  
   //           ===================================================


   static const double sqrtpi = 2.506628;

   //   - Compute mean value and RMS of the data
   unsigned int n = data.Size();
   if (n == 0) return; 
   double sumx = 0; 
   double sumx2 = 0; 
   double allcha = 0;
   double valmax = 0; 
   for (unsigned int i = 0; i < n; ++ i) { 
      double val; 
      double x = *(data.GetPoint(i,val) );
      sumx  += val*x; 
      sumx2 += val*x*x; 
      allcha += val; 
      if (val > valmax) valmax = val; 
   }

   if (allcha <= 0) return;
   double mean = sumx/allcha;
   double rms  = sumx2/allcha - mean*mean;

   double rangex = *(data.Coords(n-1)) - *(data.Coords(0));

   if (rms > 0) 
      rms  = std::sqrt(rms);
   else
      rms  = rangex/4;


    //if the distribution is really gaussian, the best approximation
   //is binwidx*allcha/(sqrtpi*rms)
   //However, in case of non-gaussian tails, this underestimates
   //the normalisation constant. In this case the maximum value
   //is a better approximation.
   //We take the average of both quantities
   double constant = 0.5*(valmax+ rangex/(sqrtpi*rms));


   //In case the mean value is outside the histo limits and
   //the RMS is bigger than the range, we take
   //  mean = center of bins
   //  rms  = half range
//    Double_t xmin = curHist->GetXaxis()->GetXmin();
//    Double_t xmax = curHist->GetXaxis()->GetXmax();
//    if ((mean < xmin || mean > xmax) && rms > (xmax-xmin)) {
//       mean = 0.5*(xmax+xmin);
//       rms  = 0.5*(xmax-xmin);
//    }

   f1->SetParameter(0,constant);
   f1->SetParameter(1,mean);
   f1->SetParameter(2,rms);
   f1->SetParLimits(2,0,10*rms);


#ifdef DEBUG
   std::cout << "Gaussian initial par values" << constant << "   " << mean << "  " << rms << std::endl;
#endif

}



   } // end namespace Fit

} // end namespace ROOT

