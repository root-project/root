// @(#)root/graf:$Id$
// Author: L. Moneta Thu Nov 15 17:04:20 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TGraphFitInterface

#include "TGraphFitInterface.h"

#include "Fit/BinData.h"

#include "TGraph.h" 
#include "TMultiGraph.h" 
#include "TF1.h" 
#include "TList.h"
#include "TError.h"

//#define DEBUG
#ifdef DEBUG
#include <iostream> 
#endif

#include <cassert> 




namespace ROOT { 

namespace Fit { 



bool IsPointOutOfRange(const TF1 * func, const double * x) { 
   // function to check if a point is outside range
   if (func ==0) return false; 
   return !func->IsInside(x);       
}
bool AdjustError(const DataOptions & option, double & error) {
   // adjust the given error accoring to the option
   //  if false is returned bin must be skipped 
   //if (option.fErrors1) error = 1;
   if (error <= 0 ) { 
      if (option.fUseEmpty) 
         error = 1.; // set error to 1 for empty bins 
      else 
         return false; 
   }
   return true; 
}




void DoFillData ( BinData  & dv,  const TGraph * gr,  BinData::ErrorType type, TF1 * func ) {  
   // internal method to do the actual filling of the data
   // given a graph and a multigraph

   // get fit option 
   DataOptions & fitOpt = dv.Opt();
      
   int  nPoints = gr->GetN();
   double *gx = gr->GetX();
   double *gy = gr->GetY();

   dv.Initialize(nPoints,1, type); 

   double x[1]; 
   for ( int i = 0; i < nPoints; ++i) { 
      
      x[0] = gx[i];
      // neglect error in x (it is a different chi2 function) 

      if (func && !func->IsInside( x )) continue;

      if (fitOpt.fErrors1)  
         dv.Add( gx[i], gy[i] ); 

      // for the errors use the getters by index to avoid cases when the arrays are zero 
      // (like in a case of a graph)
      else if (type == BinData::kValueError)  { 
         double errorY =  gr->GetErrorY(i);    
         // consider error = 0 as 1 
         if (!AdjustError(fitOpt,errorY) ) continue; 
         dv.Add( gx[i], gy[i], errorY );
      }
      else { // case use error in x or asym errors 
         double errorX = 0; 
         if (fitOpt.fCoordErrors)  
            errorX =  std::max( 0.5 * ( gr->GetErrorXlow(i) + gr->GetErrorXhigh(i) ) , 0. ) ;
         
         if (type == BinData::kAsymError)   { 
            // asymmetric errors 
            double erry = gr->GetErrorY(i); 
            if ( !AdjustError(fitOpt, erry)  ) continue; 
            dv.Add( gx[i], gy[i], errorX, gr->GetErrorYlow(i), gr->GetErrorYhigh(i) );            
         }
         // case sym errors
         else {             
            double errorY =  gr->GetErrorY(i);    
            if (errorX <= 0 ) { 
               errorX = 0; 
               if (!AdjustError(fitOpt,errorY) ) continue; 
            }
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

   double *ex = gr->GetEX();
   double *ey = gr->GetEY();
   double * eyl = gr->GetEYlow();
   double * eyh = gr->GetEYhigh();
 
  
   // check for consistency in case of dv has been already filles (case of multi-graph) 
   
   // default case for graphs (when they have errors) 
   BinData::ErrorType type = BinData::kValueError; 
   // if all errors are zero set option of using errors to 1
   if (ey == 0 && ( eyl == 0 || eyh == 0 ) ) { 
      fitOpt.fErrors1 = true;
      type =  BinData::kNoError; 
   }
   else if ( ex != 0 && fitOpt.fCoordErrors)  { 
      type = BinData::kCoordError; 
   }
   else if ( ( eyl != 0 && eyh != 0)  && fitOpt.fAsymErrors)  { 
      type = BinData::kAsymError; 
   }

   // if data are filled already do a re-initialization
   // need to 
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

   TGraph *gr;
   TList * grList = mg->GetListOfGraphs(); 

   // get fit option 
   DataOptions & fitOpt = dv.Opt();

   BinData::ErrorType type = BinData::kNoError; 
   if (!fitOpt.fErrors1 ) { 
      if ( grList->FindObject("TGraphAsymmErrors") != 0  || grList->FindObject("TGraphBentErrors") != 0 ) { 
         if (fitOpt.fAsymErrors)  
            type = BinData::kAsymError; 
         else if (fitOpt.fCoordErrors) 
            type = BinData::kCoordError;
         else 
            type = BinData::kValueError;
      }
      else if (grList->FindObject("TGraphErrors") != 0 ) { 
         if (fitOpt.fCoordErrors) 
            type = BinData::kCoordError;  
         else 
            type = BinData::kValueError;
      }
   }

   TIter next(mg->GetListOfGraphs());   
   
   while ((gr = (TGraph*) next())) {
      DoFillData( dv, gr, type, func); 
   }

#ifdef DEBUG
   std::cout << "TGraphFitInterface::FillData MultiGraph FitData size is " << dv.Size() << std::endl;
#endif
 

}


} // end namespace Fit

} // end namespace ROOT


