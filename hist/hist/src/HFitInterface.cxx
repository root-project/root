// @(#)root/hist:$Id$
// Author: L. Moneta Thu Aug 31 10:40:20 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TH1Interface

#include "HFitInterface.h"

#include "Fit/BinData.h"
#include "Fit/FitResult.h"
#include "Math/IParamFunction.h"


#include <cassert> 
#include <cmath>

#include "TH1.h"
#include "TF1.h"
#include "TGraph2D.h"
#include "TGraph.h" 
#include "TGraphErrors.h" 
// #include "TGraphErrors.h" 
// #include "TGraphBentErrors.h" 
// #include "TGraphAsymmErrors.h" 
#include "TMultiGraph.h" 
#include "TList.h"
#include "TError.h"


//#define DEBUG
#ifdef DEBUG
#include "TClass.h"
#include <iostream> 
#endif


namespace ROOT { 

namespace Fit { 

// add a namespace to distinguish from the Graph functions 
namespace HFitInterface { 


bool IsPointOutOfRange(const TF1 * func, const double * x) { 
   // function to check if a point is outside range
   if (func ==0) return false; 
   return !func->IsInside(x);       
}

bool AdjustError(const DataOptions & option, double & error, double value = 1) {
   // adjust the given error according to the option
   // return false when point must be skipped.
   // When point error = 0, the point is kept if the option UseEmpty is set or if 
   // fErrors1 is set and the point value is not zero.
   // The value should be used only for points representing counts (histograms), not for the graph. 
   // In the graph points with zero errors are by default skipped indepentently of the value. 
   // If one wants to keep the points, the option fUseEmpty must be set

   if (error <= 0) { 
      if (option.fUseEmpty || (option.fErrors1 && std::abs(value) > 0 ) ) 
         error = 1.; // set error to 1 
      else
         return false;   // skip  bins with zero errors or empty
   } else if (option.fErrors1) 
      error = 1;   // set all error to 1 for non-empty bins
   return true; 
}

void ExamineRange(TAxis * axis, std::pair<double,double> range,int &hxfirst,int &hxlast) {
   // examine the range given with the pair on the given histogram axis
   // correct in case the bin values hxfirst hxlast
   double xlow   = range.first; 
   double xhigh  = range.second; 
#ifdef DEBUG
   std::cout << "xlow " << xlow << " xhigh = " << xhigh << std::endl;
#endif
   // ignore ranges specified outside histogram range
   int ilow = axis->FindBin(xlow);
   int ihigh = axis->FindBin(xhigh);
   if (ilow > hxlast || ihigh < hxfirst) { 
      Warning("ROOT::Fit::FillData","fit range is outside histogram range, no fit data for %s",axis->GetName()); 
   } 
   // consider only range defined with-in histogram not oustide. Always exclude underflow/overflow
   hxfirst =  std::min( std::max( ilow, hxfirst), hxlast+1) ;
   hxlast  =  std::max( std::min( ihigh, hxlast), hxfirst-1) ;
   // exclude bins where range coverage is less than half bin width
   if (hxfirst < hxlast) { 
      if ( axis->GetBinCenter(hxfirst) < xlow)  hxfirst++;
      if ( axis->GetBinCenter(hxlast)  > xhigh) hxlast--;
   }
}


} // end namespace HFitInterface

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
   // to check if inclusion/exclusion at end/point
   const DataRange & range = dv.Range(); 
   if (range.Size(0) != 0) { 
      HFitInterface::ExamineRange( hfit->GetXaxis(), range(0), hxfirst, hxlast); 
      if (range.Size(0) > 1  ) { 
         Warning("ROOT::Fit::FillData","support only one range interval for X coordinate"); 
      }
   }
         
   if (hfit->GetDimension() > 1 && range.Size(1) != 0) { 
      HFitInterface::ExamineRange( hfit->GetYaxis(), range(1), hyfirst, hylast); 
      if (range.Size(1) > 1  ) 
         Warning("ROOT::Fit::FillData","support only one range interval for Y coordinate"); 
   }

   if (hfit->GetDimension() > 2 && range.Size(2) != 0) { 
      HFitInterface::ExamineRange( hfit->GetZaxis(), range(2), hzfirst, hzlast); 
      if (range.Size(2) > 1  ) 
         Warning("ROOT::Fit::FillData","support only one range interval for Z coordinate"); 
   }
   
   
   int n = (hxlast-hxfirst+1)*(hylast-hyfirst+1)*(hzlast-hzfirst+1); 
   if (fitOpt.fIntegral) n += 1;
   
#ifdef DEBUG
   std::cout << "THFitInterface: ifirst = " << hxfirst << " ilast =  " << hxlast 
             << " total bins  " << n  
             << std::endl; 
#endif
   
   // reserve n for more efficient usage
   //dv.Data().reserve(n);
   
   int hdim =  hfit->GetDimension();
   int ndim = hdim; 
   // case of function dimension less than histogram 
   if (func !=0 && func->GetNdim() == hdim-1) ndim = hdim-1;
   assert( ndim > 0 );
   //typedef  BinPoint::CoordData CoordData; 
   //CoordData x = CoordData( hfit->GetDimension() );
   dv.Initialize(n,ndim); 
   std::vector<double> x(hdim); 

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
         (*func)( &x[0] );  // evaluate using stored function parameters
         if (func->RejectedPoint() ) continue; 
      }

      if ( hdim > 1 ) { 
         for ( biny = hyfirst; biny <= hylast; ++biny) {
            if (fitOpt.fIntegral) 
               x[1] = yaxis->GetBinLowEdge(biny);
            else
               x[1] = yaxis->GetBinCenter(biny);
            
            if ( hdim >  2 ) { 
               for ( binz = hzfirst; binz <= hzlast; ++binz) {
                  if (fitOpt.fIntegral) 
                     x[2] = zaxis->GetBinLowEdge(binz);
                  else
                     x[2] = zaxis->GetBinCenter(binz);
//                  if (fitOpt.fUseRange && HFitInterface::IsPointOutOfRange(func,&x.front()) ) continue;
                  double value =  hfit->GetBinContent(binx, biny, binz);
                  double error =  hfit->GetBinError(binx, biny, binz); 
                  if (!HFitInterface::AdjustError(fitOpt,error,value) ) continue; 
                  //dv.Add(BinPoint(  x,  hfit->GetBinContent(binx, biny, binz), error ) );
                  if (ndim < hdim) // case of fitting a function with less dimension
                     dv.Add(   &x.front(),  x[2], error * zaxis->GetBinWidth(binz)  );
                  else 
                     dv.Add(   &x.front(),  value, error  );
               }  // end loop on z bins
            }
            else if (hdim == 2) { 
               // for dim == 2
//               if (fitOpt.fUseRange && HFitInterface::IsPointOutOfRange(func,&x.front()) ) continue;
               double value =  hfit->GetBinContent(binx, biny);
               double error =  hfit->GetBinError(binx, biny); 
               if (!HFitInterface::AdjustError(fitOpt,error,value) ) continue; 
               if (ndim < hdim) // case of fitting a function with less dimension
                  dv.Add(   &x.front(),  x[1], error * yaxis->GetBinWidth(biny)  );
               else 
                  dv.Add( &x.front(), value, error  );
            }   
            
         }  // end loop on y bins
         
      }
      else if (ndim == 1) { 
#ifdef DEBUG
         std::cout << "bin " << binx << " add point " << x[0] << "  " << hfit->GetBinContent(binx) << std::endl;
#endif
         // for 1D 
//         if (fitOpt.fUseRange && HFitInterface::IsPointOutOfRange(func,&x.front()) ) continue;
         double value =  hfit->GetBinContent(binx);
         double error =  hfit->GetBinError(binx); 
         if (!HFitInterface::AdjustError(fitOpt,error,value) ) continue; 
         dv.Add( x.front(),  value, error  );
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
   double rangex = data.Coords(n-1)[0] - data.Coords(0)[0];
   // to avoid binwidth = 0 set arbitrarly to 1
   double binwidth = 1;
   if ( rangex > 0) binwidth = rangex; 
   double x0 = 0;
   for (unsigned int i = 0; i < n; ++ i) { 
      double val; 
      double x = *(data.GetPoint(i,val) );
      sumx  += val*x; 
      sumx2 += val*x*x; 
      allcha += val; 
      if (val > valmax) valmax = val; 
      if (i > 0) { 
         double dx = x - x0; 
         if (dx < binwidth) binwidth = dx; 
      }         
      x0 = x; 
   }

   if (allcha <= 0) return;
   double mean = sumx/allcha;
   double rms  = sumx2/allcha - mean*mean;


   if (rms > 0) 
      rms  = std::sqrt(rms);
   else
      rms  = binwidth*n/4;


    //if the distribution is really gaussian, the best approximation
   //is binwidx*allcha/(sqrtpi*rms)
   //However, in case of non-gaussian tails, this underestimates
   //the normalisation constant. In this case the maximum value
   //is a better approximation.
   //We take the average of both quantities

//   printf("valmax %f other %f bw %f allcha %f rms %f  \n",valmax, binwidth*allcha/(sqrtpi*rms), 
//          binwidth, allcha,rms  );

   double constant = 0.5*(valmax+ binwidth*allcha/(sqrtpi*rms));


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

// filling fit data from TGraph objects

BinData::ErrorType GetDataType(const TGraph * gr, const DataOptions & fitOpt) { 
   // get type of data for TGraph objects
   double *ex = gr->GetEX();
   double *ey = gr->GetEY();
   double * eyl = gr->GetEYlow();
   double * eyh = gr->GetEYhigh();
 
  
   // default case for graphs (when they have errors) 
   BinData::ErrorType type = BinData::kValueError; 
   // if all errors are zero set option of using errors to 1
   if (fitOpt.fErrors1 || ( ey == 0 && ( eyl == 0 || eyh == 0 ) ) ) { 
      type =  BinData::kNoError; 
   }
   // need to treat case when all errors are zero 
   else if ( ex != 0 && fitOpt.fCoordErrors)  { 
      // check that all errors are not zero
      int i = 0; 
      while (i < gr->GetN() && type != BinData::kCoordError) { 
         if (ex[i] > 0) type = BinData::kCoordError; 
         ++i;
      }
   }
   else if ( ( eyl != 0 && eyh != 0)  && fitOpt.fAsymErrors)  { 
      // check also if that all errors are non zero's
      int i = 0; 
      bool zeroError = true;
      while (i < gr->GetN() && zeroError) { 
         double e2X = ( gr->GetErrorXlow(i) + gr->GetErrorXhigh(i) );
         double e2Y = eyl[i] + eyh[i];
         if ( e2X > 0 || e2Y > 0) zeroError = false; 
         ++i;
      }
      if (zeroError) 
         type = BinData::kNoError;
      else 
         type = BinData::kAsymError; 
   }

   // need to look also a case when all errors in y are zero 
   if ( ey != 0 && type != BinData::kCoordError )  { 
      int i = 0; 
      bool zeroError = true;
      while (i < gr->GetN() && zeroError) { 
         if (ey[i] > 0) zeroError = false;; 
         ++i;
      }
      if (zeroError) type = BinData::kNoError;
   }


#ifdef DEBUG
   std::cout << "type is " << type << " graph type is " << gr->IsA()->GetName() << std::endl; 
#endif

   return type; 
}

BinData::ErrorType GetDataType(const TGraph2D * gr, const DataOptions & fitOpt) { 
   // get type of data for TGraph2D object
   double *ex = gr->GetEX();
   double *ey = gr->GetEY();
   double *ez = gr->GetEZ();
  
   // default case for graphs (when they have errors) 
   BinData::ErrorType type = BinData::kValueError; 
   // if all errors are zero set option of using errors to 1
   if (ez == 0 ) { 
      type =  BinData::kNoError; 
   }
   else if ( ex != 0 && ey!=0 && fitOpt.fCoordErrors)  { 
      // check that all errors are not zero
      int i = 0; 
      while (i < gr->GetN() && type != BinData::kCoordError) { 
         if (ex[i] > 0 || ey[i] > 0) type = BinData::kCoordError; 
         ++i;
      }
   }


#ifdef DEBUG
   std::cout << "type is " << type << " graph2D type is " << gr->IsA()->GetName() << std::endl; 
#endif

   return type; 
}



void DoFillData ( BinData  & dv,  const TGraph * gr,  BinData::ErrorType type, TF1 * func ) {  
   // internal method to do the actual filling of the data
   // given a graph and a multigraph

   // get fit option 
   DataOptions & fitOpt = dv.Opt();
      
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();

   const DataRange & range = dv.Range(); 
   bool useRange = ( range.Size(0) > 0);
   double xmin = 0; 
   double xmax = 0; 
   range.GetRange(xmin,xmax); 

   dv.Initialize(nPoints,1, type); 

#ifdef DEBUG
   std::cout << "DoFillData: graph npoints = " << nPoints << " type " << type << std::endl;
   if (func) { 
      double a1,a2; func->GetRange(a1,a2); std::cout << "func range " << a1 << "  " << a2 << std::endl;
   }
#endif

   double x[1]; 
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];

      
      if (useRange && (  x[0] < xmin || x[0] > xmax) ) continue;   

      // need to evaluate function to know about rejected points
      // hugly but no other solutions
      if (func) { 
         func->RejectPoint(false);
         (*func)( x ); // evaluate using stored function parameters 
         if (func->RejectedPoint() ) continue; 
      }


      if (fitOpt.fErrors1)  
         dv.Add( gx[i], gy[i] ); 

      // for the errors use the getters by index to avoid cases when the arrays are zero 
      // (like in a case of a graph)
      else if (type == BinData::kValueError)  { 
         double errorY =  gr->GetErrorY(i);    
         // should consider error = 0 as 1 ? Decide to skip points with zero errors 
         // in case want to keep points with error = 0 as errrors=1 need to set the option UseEmpty
         if (!HFitInterface::AdjustError(fitOpt,errorY) ) continue; 
         dv.Add( gx[i], gy[i], errorY );

#ifdef DEBUG
         std::cout << "Point " << i << "  " << gx[i] <<  "  " << gy[i]  << "  " << errorY << std::endl; 
#endif


      }
      else { // case use error in x or asym errors 
         double errorX = 0; 
         if (fitOpt.fCoordErrors)  
            // shoulkd take combined average (sqrt(0.5(e1^2+e2^2))  or math average ? 
            // gr->GetErrorX(i) returns combined average
            // use math average for same behaviour as before 
            errorX =  std::max( 0.5 * ( gr->GetErrorXlow(i) + gr->GetErrorXhigh(i) ) , 0. ) ;
         

         // adjust error in y according to option 
         double errorY = std::max(gr->GetErrorY(i), 0.); 
         HFitInterface::AdjustError(fitOpt, errorY); 

         // skip points with totla error = 0
         if ( errorX <=0 && errorY <= 0 ) continue; 
         
         if (type == BinData::kAsymError)   { 
            // asymmetric errors 
            dv.Add( gx[i], gy[i], errorX, gr->GetErrorYlow(i), gr->GetErrorYhigh(i) );            
         }
         else {             
            // case symmetric Y errors
            dv.Add( gx[i], gy[i], errorX, errorY );
         }
      }
                        
   }    

#ifdef DEBUG
   std::cout << "TGraphFitInterface::FillData Graph FitData size is " << dv.Size() << std::endl;
#endif
  
}

void FillData ( BinData  & dv, const TGraph * gr,  TF1 * func ) {  
   //  fill the data vector from a TGraph. Pass also the TF1 function which is 
   // needed in case to exclude points rejected by the function
   assert(gr != 0); 

   // get fit option 
   DataOptions & fitOpt = dv.Opt();

   BinData::ErrorType type = GetDataType(gr,fitOpt); 
   // adjust option according to type
   fitOpt.fErrors1 = (type == BinData::kNoError);
   // set this if we want to have error=1 for points with zero errors (by default they are skipped)
   // fitOpt.fUseEmpty = true;
   fitOpt.fCoordErrors = (type ==  BinData::kCoordError);
   fitOpt.fAsymErrors = (type ==  BinData::kAsymError);


   // if sata are filled already check if there are consistent - otherwise do nothing
   if (dv.Size() > 0 && dv.NDim() == 1 ) { 
      // check if size is correct otherwise flag an errors 
      if (dv.PointSize() == 2 && type != BinData::kNoError ) {
         Error("FillData","Inconsistent TGraph with previous data set- skip all graph data"); 
         return;
      }
      if (dv.PointSize() == 3 && type != BinData::kValueError ) {
         Error("FillData","Inconsistent TGraph with previous data set- skip all graph data"); 
         return;
      }
      if (dv.PointSize() == 4 && type != BinData::kCoordError ) {
         Error("FillData","Inconsistent TGraph with previous data set- skip all graph data"); 
         return;
      }
   } 

   DoFillData(dv, gr, type, func); 

}

void FillData ( BinData  & dv, const TMultiGraph * mg, TF1 * func ) {  
   //  fill the data vector from a TMultiGraph. Pass also the TF1 function which is 
   // needed in case to exclude points rejected by the function
   assert(mg != 0);

   TList * grList = mg->GetListOfGraphs(); 
   assert(grList != 0);

#ifdef DEBUG
//   grList->Print();
   TIter itr(grList, kIterBackward);
   TObject *obj;
   std::cout << "multi-graph list of graps: " << std::endl;
   while ((obj = itr())) {
      std::cout << obj->IsA()->GetName() << std::endl; 
   }

#endif

   // get fit option 
   DataOptions & fitOpt = dv.Opt();

   // loop on the graphs to get the data type (use maximum)
   TIter next(grList);   
   
   BinData::ErrorType type = BinData::kNoError; 
   TGraph *gr = 0;
   while ((gr = (TGraph*) next())) {
      BinData::ErrorType t = GetDataType(gr,fitOpt); 
      if (t > type ) type = t; 
   }
   // adjust option according to type
   fitOpt.fErrors1 = (type == BinData::kNoError);
   fitOpt.fCoordErrors = (type ==  BinData::kCoordError);
   fitOpt.fAsymErrors = (type ==  BinData::kAsymError);


#ifdef DEBUG
   std::cout << "Fitting MultiGraph of type  " << type << std::endl; 
#endif

   // fill the data now
   next = grList; 
   while ((gr = (TGraph*) next())) {
      DoFillData( dv, gr, type, func); 
   }

#ifdef DEBUG
   std::cout << "TGraphFitInterface::FillData MultiGraph FitData size is " << dv.Size() << std::endl;
#endif
 
}

void FillData ( BinData  & dv, const TGraph2D * gr, TF1 * func ) {  
   //  fill the data vector from a TGraph2D. Pass also the TF1 function which is 
   // needed in case to exclude points rejected by the function
   // in case of a pure TGraph 
   assert(gr != 0); 

   // get fit option 
   DataOptions & fitOpt = dv.Opt();
   BinData::ErrorType type = GetDataType(gr,fitOpt); 
   // adjust option according to type
   fitOpt.fErrors1 = (type == BinData::kNoError);
   fitOpt.fCoordErrors = (type ==  BinData::kCoordError);
   fitOpt.fAsymErrors = false; // a TGraph2D with asymmetric errors does not exist
   
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();
   double *gz = gr->GetZ();
   
   // if all errors are zero set option of using errors to 1
   if ( gr->GetEZ() == 0) fitOpt.fErrors1 = true;
   
   double x[2]; 
   double ex[2]; 

   // look at data  range
   const DataRange & range = dv.Range(); 
   bool useRangeX = ( range.Size(0) > 0);
   bool useRangeY = ( range.Size(1) > 0);
   double xmin = 0; 
   double xmax = 0; 
   double ymin = 0; 
   double ymax = 0; 
   range.GetRange(xmin,xmax,ymin,ymax); 

   dv.Initialize(nPoints,2, type); 
   
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];
      x[1] = gy[i];

      //if (fitOpt.fUseRange && HFitInterface::IsPointOutOfRange(func, x) ) continue;
      if (useRangeX && (  x[0] < xmin || x[0] > xmax) ) continue;   
      if (useRangeY && (  x[1] < ymin || x[1] > ymax) ) continue;   

      // need to evaluate function to know about rejected points
      // hugly but no other solutions
      if (func) { 
         func->RejectPoint(false);
         (*func)( x ); // evaluate using stored function parameters 
         if (func->RejectedPoint() ) continue; 
      }


      if (type == BinData::kNoError) {   
         dv.Add( x, gz[i] ); 
         continue; 
      }

      double errorZ = gr->GetErrorZ(i); 
      if (!HFitInterface::AdjustError(fitOpt,errorZ) ) continue; 
      
      if (type == BinData::kValueError)  { 
         dv.Add( x, gz[i], errorZ );      
      }
      else if (type == BinData::kCoordError) { // case use error in coordinates (x and y) 
         ex[0] = std::max(gr->GetErrorX(i), 0.);
         ex[1] = std::max(gr->GetErrorY(i), 0.);
         dv.Add( x, gz[i], ex, errorZ );      
      }         
      else 
         assert(0); // should not go here

#ifdef DEBUG
         std::cout << "Point " << i << "  " << gx[i] <<  "  " << gy[i]  << "  " << errorZ << std::endl; 
#endif

   }

#ifdef DEBUG
   std::cout << "THFitInterface::FillData Graph2D FitData size is " << dv.Size() << std::endl;
#endif

}


// confidence intervals
bool GetConfidenceIntervals(const TH1 * h1, const ROOT::Fit::FitResult  & result, TGraphErrors * gr, double cl ) { 
   if (h1->GetDimension() != 1) { 
      Error("GetConfidenceIntervals","Invalid object used for storing confidence intervals"); 
      return false; 
   }
   // fill fit data sets with points to estimate cl. 
   BinData d;
   FillData(d,h1,0);
   gr->Set(d.NPoints() );
   double * ci = gr->GetEY(); // make CL values error of the graph
   result.GetConfidenceIntervals(d,ci,cl);
   // put function value as abscissa of the graph
   for (unsigned int ipoint = 0; ipoint < d.NPoints(); ++ipoint) { 
      const double * x = d.Coords(ipoint);
      const ROOT::Math::IParamMultiFunction * func = result.FittedFunction();
      gr->SetPoint(ipoint, x[0], (*func)(x) );
   }
   return true;
}


} // end namespace Fit

} // end namespace ROOT

