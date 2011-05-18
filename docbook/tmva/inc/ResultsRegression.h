// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ResultsRegression                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base-class for result-vectors                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_ResultsRegression
#define ROOT_TMVA_ResultsRegression

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ResultsRegression                                                    //
//                                                                      //
// Class that is the base-class for a vector of result                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#ifndef ROOT_TH1F
#include "TH1F.h"
#endif
#ifndef ROOT_TH2F
#include "TH2F.h"
#endif

#ifndef ROOT_TMVA_Results
#include "TMVA/Results.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

namespace TMVA {

   class MsgLogger;

   class ResultsRegression : public Results {

   public:

      ResultsRegression( const DataSetInfo* dsi );
      ~ResultsRegression();

      // setters
      void     SetValue( std::vector<Float_t>& value, Int_t ievt );
      void     Resize( Int_t entries )  { fRegValues.resize( entries ); }
      void     Clear()                  { fRegValues.clear(); }

      // getters
      Long64_t GetSize() const        { return fRegValues.size(); }
      virtual const std::vector< Float_t >& operator [] ( Int_t ievt ) const { return fRegValues.at(ievt); }
      std::vector<std::vector< Float_t> >* GetValueVector()  { return &fRegValues; }

      TH2F*  DeviationAsAFunctionOf( UInt_t varNum, UInt_t tgtNum );
      TH1F*  QuadraticDeviation( UInt_t tgtNum, Bool_t truncate = false, Double_t truncvalue = 0. );
      void   CreateDeviationHistograms( TString prefix );

      Types::EAnalysisType  GetAnalysisType() { return Types::kRegression; }


   private:

      mutable std::vector<std::vector< Float_t> >  fRegValues;        //! mva values (Results)
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }
   };
}

#endif
