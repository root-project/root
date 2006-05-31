// @(#)root/tmva $Id: Reader.h,v 1.2 2006/05/23 13:03:15 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Reader                                                                *
 *                                                                                *
 * Description:                                                                   *
 *      Reader class to be used in the user application to interpret the trained  *
 *      MVAs in an analysis context                                               *
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
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: Reader.h,v 1.2 2006/05/23 13:03:15 brun Exp $        
 **********************************************************************************/

#ifndef ROOT_TMVA_Reader
#define ROOT_TMVA_Reader

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Reader                                                               //
//                                                                      //
// Reader class to be used in the user application to interpret the     //
// trained MVAs in an analysis context                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif

#include <vector>
#include <string>
#include <math.h>
#include "Riostream.h"

using namespace std;

namespace TMVA {

   class Reader : public TObject {

   public:
      
      // STL types
      Reader ( vector<string>&  varNames, Bool_t verbose = 0 );
      Reader ( const string     varNames, Bool_t verbose = 0 );  // format: "var1:var2:..."

      // Root types
      Reader ( vector<TString>& varNames, Bool_t verbose = 0 );
      Reader ( const TString    varNames, Bool_t verbose = 0 );  // format: "var1:var2:..."

      virtual ~Reader( void );
  
      Bool_t   BookMVA    ( Types::MVA method, TString filename );
      Double_t EvaluateMVA( vector<Double_t>&, Types::MVA method, Double_t aux = 0 );    

      // accessors 
      Bool_t   Verbose( void ) const  { return fVerbose; }
      void     SetVerbose( Bool_t v ) { fVerbose = v; }
  
   private:
  
      // Init Reader class
      void Init( void );

      // vector of input variables
      vector<TString>* fInputVars;

      // vector of methods
      vector<MethodBase*> fMethods;

      // Decode Constructor string (or TString) and fill variable name vector
      void DecodeVarNames( const string varNames );
      void DecodeVarNames( const TString varNames );

      Bool_t fVerbose;

      ClassDef(Reader,0) // Interpret the trained MVAs in an analysis context
         };

}

#endif
