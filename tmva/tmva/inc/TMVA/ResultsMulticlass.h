// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ResultsMulticlass                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base-class for result-vectors                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_ResultsMulticlass
#define ROOT_TMVA_ResultsMulticlass

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ResultsMulticlass                                                    //
//                                                                      //
// Class which takes the results of a multiclass classification         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH1F.h"
#include "TH2F.h"

#include "TMVA/Results.h"
#include "TMVA/Event.h"
#include "IFitterTarget.h"

#include <vector>

namespace TMVA {

   class MsgLogger;
   
   class ResultsMulticlass : public Results, public IFitterTarget {

   public:

      ResultsMulticlass( const DataSetInfo* dsi, TString resultsName  );
      ~ResultsMulticlass();

      // setters
      void     SetValue( std::vector<Float_t>& value, Int_t ievt );
      void     Resize( Int_t entries )  { fMultiClassValues.resize( entries ); }
      using TObject::Clear;
      virtual void     Clear(Option_t *)  { fMultiClassValues.clear(); }

      // getters
      Long64_t GetSize() const        { return fMultiClassValues.size(); }
      virtual const std::vector< Float_t >&  operator[] ( Int_t ievt ) const { return fMultiClassValues.at(ievt); }
      std::vector<std::vector< Float_t> >* GetValueVector()  { return &fMultiClassValues; }

      Types::EAnalysisType  GetAnalysisType() { return Types::kMulticlass; }
      Float_t GetAchievableEff(UInt_t cls){return fAchievableEff.at(cls);}
      Float_t GetAchievablePur(UInt_t cls){return fAchievablePur.at(cls);}
      std::vector<Float_t>& GetAchievableEff(){return fAchievableEff;}
      std::vector<Float_t>& GetAchievablePur(){return fAchievablePur;}

      TMatrixD GetConfusionMatrix(Double_t effB);

      // histogramming
      void CreateMulticlassPerformanceHistos(TString prefix);
      void     CreateMulticlassHistos( TString prefix, Int_t nbins, Int_t nbins_high);

      Double_t EstimatorFunction( std::vector<Double_t> & );
      std::vector<Double_t> GetBestMultiClassCuts(UInt_t targetClass);

   private:

      mutable std::vector<std::vector< Float_t> >  fMultiClassValues;        // mva values (Results)
      mutable MsgLogger* fLogger;                                            //! message logger
      MsgLogger& Log() const { return *fLogger; }
      UInt_t fClassToOptimize;
      std::vector<Float_t> fAchievableEff;
      std::vector<Float_t> fAchievablePur;
      std::vector<std::vector<Double_t> > fBestCuts;

      // Temporary storage used during GetBestMultiClassCuts
      std::vector<Float_t> fClassSumWeights;
      std::vector<Float_t> fEventWeights;
      std::vector<UInt_t>  fEventClasses;

   protected:
       
       ClassDef(ResultsMulticlass,2);

   }; 

}

#endif
