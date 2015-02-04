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

using namespace std;

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
     const RooArgSet * obs = GetPdf()->getObservables(data);
     SetObservables(*obs);
     delete obs; 
   }
  // global observables 
   if (!GetGlobalObservables()) {
      RooArgSet co(*GetObservables());
      const RooArgSet * obs = GetPdf()->getObservables(data);     
      co.remove(*obs);
      RemoveConstantParameters(&co);
      if(co.getSize()>0)
	SetGlobalObservables(co);

      // TODO BUG This does not work as observables with the same name are already in the workspace.
      /*
      RooArgSet o(*GetObservables());
      o.remove(co);
      SetObservables(o);
      */
      delete obs; 
   }

   // parameters
   //   if (!GetParametersOfInterest()) {
   //      SetParametersOfInterest(RooArgSet());
   //   }
   if (!GetNuisanceParameters()) {
      const RooArgSet * params = GetPdf()->getParameters(data);
      RooArgSet p(*params);
      p.remove(*GetParametersOfInterest());
      RemoveConstantParameters(&p);
      if(p.getSize()>0)
	SetNuisanceParameters(p);
      delete params;
   }
   
   // print Modelconfig as an info message

   std::ostream& oldstream = RooPrintable::defaultPrintStream(&ccoutI(InputArguments));
   Print();
   RooPrintable::defaultPrintStream(&oldstream);
}

void ModelConfig::Print(Option_t*) const {
   // print contents of Model on the default print stream 
   // It can be changed using RooPrintable
   ostream& os = RooPrintable::defaultPrintStream();

   os << endl << "=== Using the following for " << GetName() << " ===" << endl;

 
   // args
   if(GetObservables()){
      os << "Observables:             ";
      GetObservables()->Print("");
   }
   if(GetParametersOfInterest()) {
      os << "Parameters of Interest:  ";
      GetParametersOfInterest()->Print("");
   }
   if(GetNuisanceParameters()){
      os << "Nuisance Parameters:     ";
      GetNuisanceParameters()->Print("");
   }
   if(GetGlobalObservables()){
      os << "Global Observables:      ";
      GetGlobalObservables()->Print("");
   }
   if(GetConstraintParameters()){
      os << "Constraint Parameters:   ";
      GetConstraintParameters()->Print("");
   }
   if(GetConditionalObservables()){
      os << "Conditional Observables: ";
      GetConditionalObservables()->Print("");
   }
   if(GetProtoData()){
      os << "Proto Data:              ";
      GetProtoData()->Print("");
   }

   // pdfs
   if(GetPdf()) {
      os << "PDF:                     ";
      GetPdf()->Print("");
   }
   if(GetPriorPdf()) {
      os << "Prior PDF:               ";
      GetPriorPdf()->Print("");
   }

   // snapshot
   const RooArgSet * snapshot = GetSnapshot();
   if(snapshot) {
      os << "Snapshot:                " << endl;
      snapshot->Print("v");
      delete snapshot;
   }

   os << endl;
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
   // calling loadSnapshot will also copy the current parameter values in the workspaces
   // since we do not want to change the model parameters - we restore the previous ones 
   if (! GetWS()->set(fSnapshotName.c_str() ) )return 0;
   RooArgSet snapshotVars(*GetWS()->set(fSnapshotName.c_str() ) );
   if (snapshotVars.getSize() == 0) return 0;
   // make my snapshot which will contain a copy of the snapshot variables 
   RooArgSet tempSnapshot; 
   snapshotVars.snapshot(tempSnapshot);  
   // load snapshot value from the workspace 
   if (!(GetWS()->loadSnapshot(fSnapshotName.c_str())) ) return 0;
   // by doing this snapshotVars will have the snapshot values - make the snapshot to return
   const RooArgSet * modelSnapshot = dynamic_cast<const RooArgSet*>( snapshotVars.snapshot());
   // restore now the variables of snapshot in ws to their original values
   // need to const cast since assign is not const (but in reality in just assign values and does not change the set)
   // and anyway the set is const 
   snapshotVars.assignFast(tempSnapshot);
   return modelSnapshot;  
}

void ModelConfig::LoadSnapshot() const{
   // load the snapshot from ws if it exists
   if ( !GetWS() ) return;
   GetWS()->loadSnapshot(fSnapshotName.c_str());
}

void ModelConfig::DefineSetInWS(const char* name, const RooArgSet& set) {
   // helper functions to avoid code duplication
   if ( !GetWS() ) return;

   const RooArgSet * prevSet = GetWS()->set(name); 
   if (  prevSet ) {
      //be careful not to remove passed set in case it is the same updated
      if (prevSet != &set) 
         GetWS()->removeSet(name);
   }
   
   // suppress warning when we re-define a previously defined set (when set == prevSet )
   // and set is not removed in that case
   RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;


   GetWS()->defineSet(name, set,true);

   RooMsgService::instance().setGlobalKillBelow(level) ;
  
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


  Bool_t ModelConfig::SetHasOnlyParameters(const RooArgSet& set, const char* errorMsgPrefix) {

    RooArgSet nonparams ;
    RooFIter iter = set.fwdIterator() ;
    RooAbsArg* arg ;
    while ((arg=iter.next())) {
      if (!arg->isFundamental()) {
	nonparams.add(*arg) ;
      }
    }
    
    if (errorMsgPrefix && nonparams.getSize()>0) {
      cout << errorMsgPrefix << " ERROR: specified set contains non-parameters: " << nonparams << endl ;
    }
    return (nonparams.getSize()==0) ;
  }

} // end namespace RooStats
