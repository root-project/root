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

#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif


namespace RooStats {

ModelConfig::~ModelConfig() { 
   // destructor.
   if( fOwnsWorkspace && fWS) { 
      std::cout << "ModelConfig : delete own workspace " << std::endl;
      delete fWS;
   }
}

void ModelConfig::SetWorkspace(RooWorkspace & ws) {
   // set a workspace that owns all the necessary components for the analysis
   if (!fWS)
      fWS = &ws;
   else{
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->merge(ws);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
   
}

void ModelConfig::DefineSetInWS(const char* name, RooArgSet& set) {
   // helper functions to avoid code duplication
   if (!fWS) {
      fWS = new RooWorkspace();
      fOwnsWorkspace = true; 
   }
   if (! fWS->set( name )){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      // use option to import missing constituents
      // if content with same name exist they will not be imported ? 

      fWS->defineSet(name, set,true);  

      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}
   
void ModelConfig::ImportPdfInWS(RooAbsPdf & pdf) { 
   // internal function to import Pdf in WS
   if (!fWS) 
      fWS = new RooWorkspace();
   if (! fWS->pdf( pdf.GetName() ) ){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->import(pdf);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}
   
void ModelConfig::ImportDataInWS(RooAbsData & data) { 
   // internal function to import data in WS
   if (!fWS) {
      fWS = new RooWorkspace();
      fOwnsWorkspace = true; 
   }
   if (! fWS->data( data.GetName() ) ){
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      fWS->import(data);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   }
}


} // end namespace RooStats
