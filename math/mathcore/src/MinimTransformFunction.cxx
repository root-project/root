// @(#)root/mathmore:$Id$
// Author: L. Moneta June  2009

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class MinimTransformFunction

#include "Math/MinimTransformFunction.h"
#include "Math/IFunctionfwd.h"

//#include <iostream>
#include <cmath>
#include <cassert>

namespace ROOT {

   namespace Math {

MinimTransformFunction::MinimTransformFunction ( const IMultiGradFunction * f, const std::vector<EMinimVariableType> & types,
                                                 const std::vector<double> & values,
                                                 const std::map<unsigned int, std::pair<double, double> > & bounds) :
   fX( values ),
   fFunc(f)
{
   // constructor of the class from a pointer to the function (which is managed)
   // vector specifying the variable types (free, bounded or fixed, defined in enum EMinimVariableTypes )
   // variable values (used for the fixed ones) and a map with the bounds (for the bounded variables)

   unsigned int ntot = NTot();   // NTot is fFunc->NDim()
   assert ( types.size() == ntot );
   fVariables.reserve(ntot);
   fIndex.reserve(ntot);
   for (unsigned int i = 0; i < ntot; ++i ) {
      if (types[i] ==  kFix )
         fVariables.push_back( MinimTransformVariable( values[i]) );
      else {
         fIndex.push_back(i);

         if ( types[i] ==  kDefault)
            fVariables.push_back( MinimTransformVariable() );
         else {
            std::map<unsigned int, std::pair<double,double> >::const_iterator itr = bounds.find(i);
            assert ( itr != bounds.end() );
            double low = itr->second.first;
            double up = itr->second.second;
            if (types[i] ==  kBounds )
               fVariables.push_back( MinimTransformVariable( low, up, new SinVariableTransformation()));
            else if (types[i] ==  kLowBound)
               fVariables.push_back( MinimTransformVariable( low, new SqrtLowVariableTransformation()));
            else if (types[i] ==  kUpBound)
               fVariables.push_back( MinimTransformVariable( up, new SqrtUpVariableTransformation()));
         }
      }
   }
}


void MinimTransformFunction::Transformation( const double * x, double * xext) const {
   // transform from internal to external

   unsigned int nfree = fIndex.size();

//       std::cout << "Transform:  internal ";
//       for (int i = 0; i < nfree; ++i) std::cout << x[i] << "  ";
//       std::cout << "\t\t";

   for (unsigned int i = 0; i < nfree; ++i ) {
      unsigned int extIndex = fIndex[i];
      const MinimTransformVariable & var = fVariables[ extIndex ];
      if (var.IsLimited() )
         xext[ extIndex ] = var.InternalToExternal( x[i] );
      else
         xext[ extIndex ] = x[i];
   }

//       std::cout << "Transform:  external ";
//       for (int i = 0; i < fX.size(); ++i) std::cout << fX[i] << "  ";
//       std::cout << "\n";

}

void  MinimTransformFunction::InvTransformation(const double * xExt,  double * xInt) const {
   // inverse function transformation (external -> internal)
   for (unsigned int i = 0; i < NDim(); ++i ) {
      unsigned int extIndex = fIndex[i];
      const MinimTransformVariable & var = fVariables[ extIndex ];
      assert ( !var.IsFixed() );
      if (var.IsLimited() )
         xInt[ i ] = var.ExternalToInternal( xExt[extIndex] );
      else
         xInt[ i ] = xExt[extIndex];
   }
}

void  MinimTransformFunction::InvStepTransformation(const double * x, const double * sExt,  double * sInt) const {
   // inverse function transformation for steps (external -> internal)
   for (unsigned int i = 0; i < NDim(); ++i ) {
      unsigned int extIndex = fIndex[i];
      const MinimTransformVariable & var = fVariables[ extIndex ];
      assert ( !var.IsFixed() );
      if (var.IsLimited() )  {
         // bound variables
         double x2 = x[extIndex] + sExt[extIndex];
         if (var.HasUpperBound() && x2 >= var.UpperBound() )
            x2 = x[extIndex] - sExt[extIndex];
         // transform x and x2
         double xint = var.ExternalToInternal ( x[extIndex] );
         double x2int = var.ExternalToInternal( x2 );
         sInt[i] =  std::abs( x2int - xint);
      }
      else
         sInt[ i ] = sExt[extIndex];
   }
}

void  MinimTransformFunction::GradientTransformation(const double * x, const double *gExt, double * gInt) const {
   //transform gradient vector (external -> internal) at internal point x
   unsigned int nfree = fIndex.size();
   for (unsigned int i = 0; i < nfree; ++i ) {
      unsigned int extIndex = fIndex[i];
      const MinimTransformVariable & var = fVariables[ extIndex ];
      assert (!var.IsFixed() );
      if (var.IsLimited() )
         gInt[i] = gExt[ extIndex ] * var.DerivativeIntToExt( x[i] );
      else
         gInt[i] = gExt[ extIndex ];
   }
}


void  MinimTransformFunction::MatrixTransformation(const double * x, const double *covInt, double * covExt) const {
   //transform covariance matrix (internal -> external) at internal point x
   // use row storages for matrices  m(i,j) = rep[ i * dim + j]
   // ignore fixed points
   unsigned int nfree = fIndex.size();
   unsigned int ntot = NTot();
   for (unsigned int i = 0; i < nfree; ++i ) {
      unsigned int iext = fIndex[i];
      const MinimTransformVariable & ivar = fVariables[ iext ];
      assert (!ivar.IsFixed());
      double ddi = ( ivar.IsLimited() ) ? ivar.DerivativeIntToExt( x[i] ) : 1.0;
      // loop on j  variables  for not fixed i variables (forget that matrix is symmetric) - could be optimized
      for (unsigned int j = 0; j < nfree; ++j ) {
         unsigned int jext = fIndex[j];
         const MinimTransformVariable & jvar = fVariables[ jext ];
         double ddj = ( jvar.IsLimited() ) ? jvar.DerivativeIntToExt( x[j] ) : 1.0;
         assert (!jvar.IsFixed() );
         covExt[ iext * ntot + jext] =  ddi * ddj * covInt[ i * nfree + j];
      }
   }
}


   } // end namespace Math

} // end namespace ROOT

