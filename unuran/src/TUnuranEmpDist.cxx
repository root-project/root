// @(#)root/unuran:$Name:  $:$Id: TUnuranEmpDist.cxx,v 1.1 2007/03/08 09:31:54 moneta Exp $
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranEmpDist

#include "TUnuranEmpDist.h"

#include "TH1.h"

#include <cassert>


TUnuranEmpDist::TUnuranEmpDist (const TH1 * h1, bool useBuffer) : 
   fMin(0),
   fMax(0)
{
   //Constructor from a TH1 objects. 
   //The buffer of the histo, if available, can be used for 
   // the estimation of the parent distribution using smoothing

   fDim = h1->GetDimension();

   bool unbin = useBuffer &&  h1->GetBufferLength() > 0 ;
   fBinned = !unbin;

   if (fBinned ) { 
      int nbins = h1->GetNbinsX(); 
      fData.reserve(nbins);
      for (int i =0; i < nbins; ++i) 
         fData.push_back( h1->GetBinContent(i+1) );
      
      fMin = h1->GetXaxis()->GetXmin();
      fMax = h1->GetXaxis()->GetXmax();
   }
   else { 
      //std::cout << "use kernel smoothing method" << std::endl;

      int n = h1->GetBufferLength(); 
      const double * bf = h1->GetBuffer(); 
      fData.reserve(n);
      // fill buffer (assume weights are equal to 1)
      // bugger is : [n,w0,x0,y0,..,w1,x1,y1,...wn,xn,yn]
      // buffer contains size
      for (int i = 0; i < n; ++i) {
         int index = (fDim+1)*i + fDim + 1;
         fData.push_back(  bf[index] );         
      }
   }
} 


TUnuranEmpDist::TUnuranEmpDist(const TUnuranEmpDist & rhs) :
   TUnuranBaseDist()
{
   // Implementation of copy ctor using aassignment operator
   operator=(rhs);
}

TUnuranEmpDist & TUnuranEmpDist::operator = (const TUnuranEmpDist &rhs) 
{
   // Implementation of assignment operator (copy only the funciton pointer not the function itself)
   if (this == &rhs) return *this;  // time saving self-test
   fData  = rhs.fData;
   fDim   = rhs.fDim;
   fMin   = rhs.fMin;
   fMax   = rhs.fMax;
   fBinned = rhs.fBinned;
   return *this;
}






