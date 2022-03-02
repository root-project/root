// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Results                                                               *
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

#ifndef ROOT_TMVA_Results
#define ROOT_TMVA_Results

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Results                                                              //
//                                                                      //
// Class that is the base-class for a vector of result                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <map>

#include "TList.h"

#include "TMVA/Types.h"
#include "TMVA/DataSetInfo.h"

class TH1;
class TH2;
class TGraph;

namespace TMVA {

   class DataSet;
   class MsgLogger;

   class Results:public TObject {

   public:

       Results( const DataSetInfo* dsi, TString resultsName  );
       Results();
       virtual ~Results();

      // setters
      void                Store( TObject* obj, const char* alias=0 );
      void                SetTreeType( Types::ETreeType type ) { fTreeType = type; }

      // getters
      Types::ETreeType    GetTreeType()    const { return fTreeType; }
      const DataSetInfo*  GetDataSetInfo() const { return fDsi; }
      DataSet*            GetDataSet()     const { return fDsi->GetDataSet(); }
      TList*              GetStorage()     const { return fStorage; }
      TObject*            GetObject(const TString & alias) const;
      TH1*                GetHist(const TString & alias) const;
      TH2*                GetHist2D(const TString & alias) const;
      TGraph*             GetGraph(const TString & alias) const;
      virtual Types::EAnalysisType  GetAnalysisType() { return Types::kNoAnalysisType; }
      //test
      Bool_t              DoesExist(const TString & alias) const;

      // delete all stored data
//       using TObject::Delete;
      virtual void Delete(Option_t *option="");

      virtual const std::vector< Float_t >&  operator [] ( Int_t ievt ) const = 0;

   private:
      Types::ETreeType fTreeType;                ///< tree type for this result
      const DataSetInfo *fDsi;                   ///<-> a pointer to the datasetinfo-object
      TList *fStorage;                           ///<-> stores all the result-histograms
      std::map<TString, TObject *> *fHistAlias;  ///<-> internal map for quick access to stored histograms
      mutable MsgLogger*           fLogger;      ///<! message logger
      MsgLogger& Log() const { return *fLogger; }
   public:

       ClassDef(Results,1);

   };
}

#endif
