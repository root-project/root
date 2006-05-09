// @(#)root/tmva $Id: TMVA_Volume.h,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Volume                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Volume for BinarySearchTree                                               *
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_Volume.h,v 1.1 2006/05/08 12:46:31 brun Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_Volume
#define ROOT_TMVA_Volume

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_Volume                                                          //
//                                                                      //
// Volume for BinarySearchTree                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include <vector>
#include "Rtypes.h"
#include "time.h"

class TMVA_BinaryTree;

// volume element, variable space beteen upper and lower bonds of nvar-dimensional
// variable space

class TMVA_Volume {

 public:

  TMVA_Volume( std::vector<Float_t>* l = 0, std::vector<Float_t>* u = 0);
  TMVA_Volume( std::vector<Double_t>* l = 0, std::vector<Double_t>* u = 0);
  TMVA_Volume( TMVA_Volume& );
  TMVA_Volume( Float_t* l , Float_t* u , Int_t nvar );
  TMVA_Volume( Double_t* l , Double_t* u , Int_t nvar );

  // two simple constructors for 1 dimensional volues
  TMVA_Volume( Float_t l , Float_t u );
  TMVA_Volume( Double_t l , Double_t u );

  virtual ~TMVA_Volume( void );

  void Delete       ( void );
  void Scale        ( Double_t f );
  void ScaleInterval( Double_t f );
  void Print        ( void ) const;

  // allow direct access for speed
  std::vector<Double_t> *fLower;
  std::vector<Double_t> *fUpper;

 private:

  Bool_t                fOwnerShip;
};

#endif
