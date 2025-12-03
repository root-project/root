// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline2                                                              *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Quadratic spline class; uses quadrax function for interpolation           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_TSpline2
#define ROOT_TMVA_TSpline2

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpline2                                                             //
//                                                                      //
// Quadratic interpolation class (using quadrax)                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"

namespace TMVA {

   class TSpline2 : public TSpline {

   public:

      TSpline2( const TString& title, const TGraph *theGraph );
      virtual ~TSpline2( void );

      Double_t Eval( Double_t x ) const override;

      // dummy implementations
      void BuildCoeff( void ) override;
      void GetKnot( Int_t i, Double_t& x, Double_t& y ) const override;

   private:
      std::vector<Double_t> fX;
      std::vector<Double_t> fY;

      Double_t Quadrax( Float_t dm, Float_t dm1,
                        Float_t dm2, Float_t dm3,
                        Float_t cos1, Float_t cos2,
                        Float_t cos3 ) const;

      ClassDefOverride(TSpline2,0); //Quadratic interpolation class (using quadrax)
   };

} // namespace TMVA

#endif


