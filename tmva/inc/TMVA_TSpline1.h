// @(#)root/tmva $Id: TMVA_TSpline1.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_TSpline1                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      Linear interpolation class; derivative of TSpline                         *
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
 * $Id: TMVA_TSpline1.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $        
 **********************************************************************************/

#ifndef ROOT_TMVA_TSpline1
#define ROOT_TMVA_TSpline1

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_TSpline1                                                        //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"

class TMVA_TSpline1 : public TSpline {

 public:
  
  // dummy implementations:
  virtual void BuildCoeff( void );
  virtual void GetKnot( Int_t i, Double_t& x, Double_t& y ) const;


  TMVA_TSpline1( TString title, TGraph* theGraph );
  virtual ~TMVA_TSpline1( void );
  virtual  Double_t Eval( Double_t x ) const;

 private:

  TGraph *fGraph;

  ClassDef(TMVA_TSpline1,0) //Linear interpolation class
};

#endif 


