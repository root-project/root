// @(#)root/tmva $Id: TSpline1.h,v 1.10 2007/04/19 06:53:01 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline1                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Linear interpolation class; derivative of TSpline                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_TSpline1
#define ROOT_TMVA_TSpline1

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpline1                                                             //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"

namespace TMVA {

   class TSpline1 : public TSpline {

   public:
  
      TSpline1( TString title, TGraph* theGraph );
      virtual ~TSpline1( void );

      virtual  Double_t Eval( Double_t x ) const;

      // dummy implementations
      virtual void BuildCoeff( void );
      virtual void GetKnot( Int_t i, Double_t& x, Double_t& y ) const;

   private:

      TGraph *fGraph;  // graph that is splined

      ClassDef(TSpline1,0) //Linear interpolation class
   };

} // namespace TMVA

#endif 


