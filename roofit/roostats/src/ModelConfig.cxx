// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/ModelConfig.h"

#include "TROOT.h"

#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#include <sstream>

namespace RooStats {

ModelConfig::~ModelConfig() { 
   // destructor.
//    if( fOwnsWorkspace && fWS) { 
//       std::cout << "ModelConfig : delete own workspace " << std::endl;
//       delete fWS;
//    }
}

void ModelConfig::GuessObsAndNuisance(const RooAbsData& data) {
   // Makes sensible guesses of observables, parameters of interest
   // and nuisance parameters.
   //
   // Defaults:
   //  observables: determined from data,
   //  global observables = explicit obs  -  obs from data
   //  parameters of interest: empty,
   //  nuisance parameters: all parameters except parameters of interest
  //
  // We use NULL to mean not set, so we don't want to fill
  // with empty RooArgSets

   // observables
  if (!GetObservables()) {
    SetObservables(*GetPdf()->getObservables(data));
    //const RooArgSet* temp = data.get();
    //     SetObservables(*(RooArgSet*)temp->snapshot());
    //     delete temp;
   }
   if (!GetGlobalObservables()) {
      RooArgSet co(*GetObservables());
      co.remove(*GetPdf()->getObservables(data));
      RemoveConstantParameters(&co);
      if(co.getSize()>0)
	SetGlobalObservables(co);

      // TODO BUG This does not work as observables with the same name are already in the workspace.
      /*
      RooArgSet o(*GetObservables());
      o.remove(co);
      SetObservables(o);
      */
   }

   // parameters
   //   if (!GetParametersOfInterest()) {
   //      SetParametersOfInterest(RooArgSet());
   //   }
   if (!GetNuisanceParameters()) {
      RooArgSet p(*GetPdf()->getParameters(data));
      p.remove(*GetParametersOfInterest());
      RemoveConstantParameters(&p);
      if(p.getSize()>0)
	SetNuisanceParameters(p);
   }

   ostream& oldstream = RooPrintable::defaultPrintStream(&ccoutI(InputArguments));
   ccoutI(InputArguments) << endl << "=== Using the following for " << GetName() << " ===" << endl;
   if(GetObservables()){
     ccoutI(InputArguments) << "Observables:             ";
     GetObservables()->Print("");
   }
   if(GetParametersOfInterest()) {
     ccoutI(InputArguments) << "Parameters of Interest:  ";
     GetParametersOfInterest()->Print("");
   }
   if(GetNuisanceParameters()){
     ccoutI(InputArguments) << "Nuisance Parameters:     ";
     GetNuisanceParameters()->Print("");
   }
   if(GetGlobalObservables()){
     ccoutI(InputArguments) << "Global Observables:      ";
     GetGlobalObservables()->Print("");
   }
   ccoutI(InputArguments) << endl;
   RooPrintable::defaultPrintStream(&oldstream);
}


void ModelConfig::SetWorkspace(RooWorkspace & ws) {
   // set a workspace that owns all the necessary components for the analysis
   if (!fWS) { 
      fWS = &ws;
      fWSName = ws.GetName();
      fRefWS = &ws; 
   }   
   else{
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->merge(ws);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
   
}

RooWorkspace * ModelConfig::GetWS() const {
   // get workspace if pointer is null get from the TRef 
   if (fWS) return fWS; 
   // get from TRef
   fWS = dynamic_cast<RooWorkspace *>(fRefWS.GetObject() );
   if (fWS) return fWS; 
   coutE(ObjectHandling) << "workspace not set" << endl;
   return 0;
}

void ModelConfig::SetSnapshot(const RooArgSet& set) {
   // save snaphot in the workspace 
   // and use values passed with the set
   if (!fWS) {
      coutE(ObjectHandling) << "workspace not set" << endl;
      return;
   }
   fSnapshotName = GetName();
   if (fSnapshotName.size()  > 0) fSnapshotName += "_";
   fSnapshotName += set.GetName();
   if (fSnapshotName.size()  > 0) fSnapshotName += "_";
   fSnapshotName += "snapshot";
   fWS->saveSnapshot(fSnapshotName.c_str(), set, true);  // import also the given parameter values
   DefineSetInWS(fSnapshotName.c_str(), set);
}    

const RooArgSet * ModelConfig::GetSnapshot() const{
   // load the snapshot from ws and return the corresponding set with the snapshot values
   if (!fWS) return 0; 
   if (!(fWS->loadSnapshot(fSnapshotName.c_str())) ) return 0;
   return fWS->set(fSnapshotName.c_str() );
}

void ModelConfig::LoadSnapshot() const{
   // load the snapshot from ws if it exists
   if (!fWS) return;

   // kill output
   RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
   fWS->loadSnapshot(fSnapshotName.c_str());
   RooMsgService::instance().setGlobalKillBelow(level);
}

void ModelConfig::DefineSetInWS(const char* name, const RooArgSet& set) {
   // helper functions to avoid code duplication
   if (!fWS) {
      coutE(ObjectHandling) << "workspace not set" << endl;
      return;
   }
   if (! fWS->set( name )){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      // use option to import missing constituents
      // TODO IS THIS A BUG? if content with same name exist they will not be imported ?
      // See ModelConfig::GuessObsAndNuissance(...) for example of the problem.

      fWS->defineSet(name, set,true);  

      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}
   
void ModelConfig::ImportPdfInWS(const RooAbsPdf & pdf) { 
   // internal function to import Pdf in WS
   if (!fWS) { 
      coutE(ObjectHandling) << "workspace not set" << endl;
      return;
   }
   if (! fWS->pdf( pdf.GetName() ) ){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->import(pdf, RooFit::RecycleConflictNodes());
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}
   
void ModelConfig::ImportDataInWS(RooAbsData & data) { 
   // internal function to import data in WS
   if (!fWS) {
      coutE(ObjectHandling) << "workspace not set" << endl;
      return;
   }
   if (! fWS->data( data.GetName() ) ){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->import(data);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}


} // end namespace RooStats
