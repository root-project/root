// @(#)root/tmva $Id: Reader.h,v 1.11 2006/10/02 09:10:39 andreas.hoecker Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Reader                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://ttmva.sourceforge.net/LICENSE)                                         *
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

#include "TROOT.h"

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_IMethod
#include "TMVA/IMethod.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

#include <vector>
#include <map>
#include <string>

namespace TMVA {

   class Reader : public TObject {

   public:
      
      // without prior specification of variables
      Reader ( Bool_t verbose = 0 );

      // STL types
      Reader ( std::vector<std::string>&  varNames, Bool_t verbose = 0 );
      Reader ( const std::string     varNames, Bool_t verbose = 0 );  // format: "var1:var2:..."

      // Root types
      Reader ( std::vector<TString>& varNames, Bool_t verbose = 0 );
      Reader ( const TString    varNames, Bool_t verbose = 0 );  // format: "var1:var2:..."

      virtual ~Reader( void );
  
      IMethod* BookMVA( TString methodName, TString weightfile );

      Double_t EvaluateMVA( const std::vector<Float_t>&,  TString methodName, Double_t aux = 0 );    
      Double_t EvaluateMVA( const std::vector<Double_t>&, TString methodName, Double_t aux = 0 );    
      Double_t EvaluateMVA( IMethod* method,              Double_t aux = 0 );    
      Double_t EvaluateMVA( TString methodName,           Double_t aux = 0 );    

      // accessors 
      Bool_t   Verbose( void ) const  { return fVerbose; }
      void     SetVerbose( Bool_t v ) { fVerbose = v; }

      const DataSet& Data() const { return *fDataSet; }
      DataSet& Data() { return *fDataSet; }
      
      void AddVariable( const TString & expression, float * );
      void AddVariable( const TString & expression, int * );
  
   private:

      // this booking method is internal
      IMethod* BookMVA( Types::MVA method,  TString weightfile );

      DataSet * fDataSet;
  
      // Init Reader class
      void Init( void );

      // Decode Constructor string (or TString) and fill variable name std::vector
      void DecodeVarNames( const std::string varNames );
      void DecodeVarNames( const TString varNames );

      Bool_t fVerbose;

      std::map<const TString, IMethod*> fMethodMap; // map of methods

      ClassDef(Reader,0) // Interpret the trained MVAs in an analysis context
   };

}

#endif
