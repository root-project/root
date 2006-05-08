// @(#)root/tmva $Id: TMVA_TSpline2.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_TSpline2                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      Quadratic spline class; uses quadrax function for interpolation           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_TSpline2.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $        
 **********************************************************************************/

#ifndef ROOT_TMVA_TSpline2
#define ROOT_TMVA_TSpline2

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_TSpline2                                                        //
//                                                                      //
// Quadratic interpolation class (using quadrax)                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"

class TMVA_TSpline2 : public TSpline {

 public:
  
  // dummy implementations:
  virtual void BuildCoeff( void );
  virtual void GetKnot( Int_t i, Double_t& x, Double_t& y ) const;


  TMVA_TSpline2( TString title, TGraph* theGraph );
  virtual ~TMVA_TSpline2( void );
  virtual  Double_t Eval( const Double_t x ) const;

 private:

  TGraph *fGraph;
  Double_t Quadrax(const Float_t dm, const Float_t dm1,
		    const Float_t dm2,const  Float_t dm3,
		    const Float_t cos1, const Float_t cos2, 
		    const Float_t cos3 ) const;
  
  ClassDef(TMVA_TSpline2,0) //Quadratic interpolation class (using quadrax)
};


#endif 


