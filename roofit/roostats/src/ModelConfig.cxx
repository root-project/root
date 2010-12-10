// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Sven Kreiss
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


ClassImp(RooStats::ModelConfig)


namespace RooStats {

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

   Print();
}

void ModelConfig::Print(Option_t*) const {
   // print contents
   ccoutI(InputArguments) << endl << "=== Using the following for " << GetName() << " ===" << endl;

   // necessary so that GetObservables()->Print("") gets piped to the
   // ccoutI(InputArguments) stream
   ostream& oldstream = RooPrintable::defaultPrintStream(&ccoutI(InputArguments));

   // args
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
   if(GetConstraintParameters()){
      ccoutI(InputArguments) << "Constraint Parameters:   ";
      GetConstraintParameters()->Print("");
   }
   if(GetConditionalObservables()){
      ccoutI(InputArguments) << "Conditional Observables: ";
      GetConditionalObservables()->Print("");
   }
   if(GetProtoData()){
      ccoutI(InputArguments) << "Proto Data:              ";
      GetProtoData()->Print("");
   }

   // pdfs
   if(GetPdf()) {
      ccoutI(InputArguments) << "PDF:                     ";
      GetPdf()->Print("");
   }
   if(GetPriorPdf()) {
      ccoutI(InputArguments) << "Prior PDF:               ";
      GetPriorPdf()->Print("");
   }

   // snapshot
   if(GetSnapshot()) {
      ccoutI(InputArguments) << "Snapshot:                " << endl;
      GetSnapshot()->Print("v");
   }

   ccoutI(InputArguments) << endl;
   RooPrintable::defaultPrintStream(&oldstream);
}


void ModelConfig::SetWS(RooWorkspace & ws) {
   // set a workspace that owns all the necessary components for the analysis
   if( !fRefWS.GetObject() ) {
      fRefWS = &ws; 
      fWSName = ws.GetName();
   }   
   else{
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      GetWS()->merge(ws);
      RooMsgService::instance().setGlobalKillBelow(level) ;
   }
}

RooWorkspace * ModelConfig::GetWS() const {
   // get from TRef
   RooWorkspace *ws = dynamic_cast<RooWorkspace *>(fRefWS.GetObject() );
   if(!ws) {
      coutE(ObjectHandling) << "workspace not set" << endl;
      return NULL;
   }
   return ws;
}

void ModelConfig::SetSnapshot(const RooArgSet& set) {
   // save snaphot in the workspace 
   // and use values passed with the set
   if ( !GetWS() ) return;

   fSnapshotName = GetName();
   if (fSnapshotName.size()  > 0) fSnapshotName += "_";
   fSnapshotName += set.GetName();
   if (fSnapshotName.size()  > 0) fSnapshotName += "_";
   fSnapshotName += "snapshot";
   GetWS()->saveSnapshot(fSnapshotName.c_str(), set, true);  // import also the given parameter values
   DefineSetInWS(fSnapshotName.c_str(), set);
}    

const RooArgSet * ModelConfig::GetSnapshot() const{
   // Load the snapshot from ws and return the corresponding set with the snapshot values.
   // User must delete returned RooArgSet.
   if ( !GetWS() ) return 0;
   if (!fSnapshotName.length()) return 0;
   if (!(GetWS()->loadSnapshot(fSnapshotName.c_str())) ) return 0;

   return dynamic_cast<const RooArgSet*>(GetWS()->set(fSnapshotName.c_str() )->snapshot());
}

void ModelConfig::LoadSnapshot() const{
   // load the snapshot from ws if it exists
   if ( !GetWS() ) return;

   // kill output
   RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
   GetWS()->loadSnapshot(fSnapshotName.c_str());
   RooMsgService::instance().setGlobalKillBelow(level);
}

void ModelConfig::DefineSetInWS(const char* name, const RooArgSet& set) {
   // helper functions to avoid code duplication
   if ( !GetWS() ) return;

   if ( ! GetWS()->set(name) ) {
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      // use option to import missing constituents
      // TODO IS THIS A BUG? if content with same name exist they will not be imported ?
      // See ModelConfig::GuessObsAndNuissance(...) for example of the problem.

      GetWS()->defineSet(name, set,true);

      RooMsgService::instance().setGlobalKillBelow(level) ;
   }
}
   
void ModelConfig::ImportPdfInWS(const RooAbsPdf & pdf) { 
   // internal function to import Pdf in WS 
   if ( !GetWS() ) return;

   if (! GetWS()->pdf( pdf.GetName() ) ){
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      GetWS()->import(pdf, RooFit::RecycleConflictNodes());
      RooMsgService::instance().setGlobalKillBelow(level) ;
   }
}
   
void ModelConfig::ImportDataInWS(RooAbsData & data) { 
   // internal function to import data in WS
   if ( !GetWS() ) return;

   if (! GetWS()->data( data.GetName() ) ){
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      GetWS()->import(data);
      RooMsgService::instance().setGlobalKillBelow(level) ;
   }
}


} // end namespace RooStats
