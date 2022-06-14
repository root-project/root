// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Results                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::Results
\ingroup TMVA
Class that is the base-class for a vector of result
*/

#include "TMVA/Results.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"

#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TList.h"

#include <vector>

namespace TMVA {
   class DataSetInfo;
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::Results::Results( const DataSetInfo* dsi, TString resultsName )
   : fTreeType(Types::kTraining),
     fDsi(dsi),
     fStorage( new TList() ),
     fHistAlias( new std::map<TString, TObject*> ),
     fLogger( new MsgLogger(Form("Results%s",resultsName.Data()), kINFO) )
{
   fStorage->SetOwner();
}

TMVA::Results::Results( )
: fTreeType(Types::kTraining),
fDsi(0),
fStorage( new TList() ),
fHistAlias( new std::map<TString, TObject*> ),
fLogger( new MsgLogger("Results", kINFO))
{
    fStorage->SetOwner();
}


////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::Results::~Results()
{
   // delete result-histograms
   delete fStorage;
   delete fHistAlias;
   delete fLogger;
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::Results::Store( TObject* obj, const char* alias )
{
   TListIter l(fStorage);
   // check if object is already in list
   while (void* p = (void*)l()) {
      if(p==obj)
         *fLogger << kFATAL << "Histogram pointer " << p << " already exists in results storage" << Endl;
   }

   TString as(obj->GetName());
   if (alias!=0) as=TString(alias);
   if (fHistAlias->find(as) != fHistAlias->end()) {
      // alias exists
      *fLogger << kFATAL << "Alias " << as << " already exists in results storage" << Endl;
   }
   if( obj->InheritsFrom(TH1::Class()) ) {
      ((TH1*)obj)->SetDirectory(0);
   }
   fStorage->Add( obj );
   fHistAlias->insert(std::pair<TString, TObject*>(as,obj));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a stored object if it exists. If it does not, a nullptr is returned.
///

TObject* TMVA::Results::GetObject(const TString & alias) const
{
   std::map<TString, TObject*>::iterator it = fHistAlias->find(alias);

   if (it != fHistAlias->end()) return it->second;

   // alias does not exist
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if there is an object stored in the result for a given alias,
/// false otherwise.
///

Bool_t TMVA::Results::DoesExist(const TString & alias) const
{
   TObject* test = GetObject(alias);

   return (test != nullptr);
}

////////////////////////////////////////////////////////////////////////////////

TH1* TMVA::Results::GetHist(const TString & alias) const
{
   TH1* out=dynamic_cast<TH1*>(GetObject(alias));
   if (!out) Log() <<kWARNING << "You have asked for histogram " << alias << " which does not seem to exist in *Results* .. better don't use it " << Endl;
   return out;
}

////////////////////////////////////////////////////////////////////////////////

TH2* TMVA::Results::GetHist2D(const TString & alias) const
{
   TH2* out=dynamic_cast<TH2*>(GetObject(alias));
   if (!out) Log() <<kWARNING << "You have asked for 2D histogram " << alias << " which does not seem to exist in *Results* .. better don't use it " << Endl;
   return out;
}
////////////////////////////////////////////////////////////////////////////////

TGraph* TMVA::Results::GetGraph(const TString & alias) const
{
   return (TGraph*)GetObject(alias);
}


////////////////////////////////////////////////////////////////////////////////
/// delete all stored histograms

void TMVA::Results::Delete(Option_t *)
{
   fStorage->Delete();
   fHistAlias->clear();
}
