/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooSimPdfBuilder.cc,v 1.1 2001/10/30 07:29:15 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   17-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
 *****************************************************************************/

#define _REENTRANT
#include <string.h>
#include <strings.h>
#include "RooFitCore/RooSimPdfBuilder.hh"

#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooMappedCategory.hh"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooLinearVar.hh"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitCore/RooTruthModel.hh"
#include "RooFitCore/RooAddModel.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooCustomizer.hh"
#include "RooFitCore/RooThresholdCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooSimFitContext.hh"
#include "RooFitCore/RooTrace.hh"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooGenericPdf.hh"


ClassImp(RooSimPdfBuilder)
;



RooSimPdfBuilder::RooSimPdfBuilder(const RooArgSet& protoPdfSet) :
  _protoPdfSet(protoPdfSet)
{
}




RooArgSet* RooSimPdfBuilder::createProtoBuildConfig()
{
  // Make RooArgSet of configuration objects
  RooArgSet* buildConfig = new RooArgSet ;
  buildConfig->addOwned(* new RooStringVar("physModels","List and mapping of physics models to include in build","",1024)) ;
  buildConfig->addOwned(* new RooStringVar("splitCats","List of categories used for splitting","",1024)) ;

  TIterator* iter = _protoPdfSet.createIterator() ;
  RooAbsPdf* proto ;
  while (proto=(RooAbsPdf*)iter->Next()) {
    buildConfig->addOwned(* new RooStringVar(proto->GetName(),proto->GetName(),"",2048)) ;
  }
  delete iter ;

  return buildConfig ;
}




const RooAbsPdf* RooSimPdfBuilder::buildPdf(const char* buildConfigFile, const RooAbsData* dataSet)
{
  // Read configuration from file
  RooArgSet* buildConfig = createProtoBuildConfig() ;
  buildConfig->readFromFile(buildConfigFile) ;

  const RooAbsPdf* ret = buildPdf(*buildConfig,dataSet) ;

  delete buildConfig ;
  return ret ;
}



const RooAbsPdf* RooSimPdfBuilder::buildPdf(const RooArgSet& buildConfig, const RooAbsData* dataSet)
{
  // Initialize needed components
  const RooArgSet* dataVars = dataSet->get() ;

  // Retrieve physics index category
  char buf[1024] ;
  strcpy(buf,((RooStringVar*)buildConfig.find("physModels"))->getVal()) ;
  RooAbsCategoryLValue* physCat(0) ;
  if (strstr(buf," : ")) {
    const char* physCatName = strtok(buf," ") ;
    physCat = dynamic_cast<RooAbsCategoryLValue*>(dataVars->find(physCatName)) ;
    if (!physCat) {
      cout << "RooSimPdfBuilder::buildPdf: ERROR physics index category " << physCatName 
	   << " not found in dataset variables" << endl ;
      return 0 ;      
    }
    cout << "RooSimPdfBuilder::buildPdf: category indexing physics model: " << physCatName << endl ;
  }

  // Create list of physics models to be built
  char *physName ;
  RooArgSet physModelSet ;
  if (physCat) {
    // Absorb colon token
    char* colon = strtok(0," ") ;
    physName = strtok(0," ") ;
  } else {
    physName = strtok(buf," ") ;
  }

  Bool_t first(kTRUE) ;
  RooArgSet stateMap ;
  while(physName) {

    char *stateName(0) ;

    // physName may be <state>=<pdfName> or just <pdfName> is state and pdf have identical names
    if (strchr(physName,'=')) {
      // Must have a physics category for mapping to make sense
      if (!physCat) {
	cout << "RooSimPdfBuilder::buildPdf: WARNING: without physCat specification "
	     << "<physCatState>=<pdfProtoName> association is meaningless" << endl ;
      }
      stateName = physName ;
      physName = strchr(stateName,'=') ;
      *(physName++) = 0 ;      
    } else {
      stateName = physName ;
    }

    RooAbsPdf* physModel = (RooAbsPdf*) _protoPdfSet.find(physName) ;
    if (!physModel) {
      cout << "RooSimPdfBuilder::buildPdf: ERROR requested physics model " 
	   << physName << " is not defined" << endl ;
      return 0 ;
    }    

    // Check if state mapping has already been defined
    if (stateMap.find(stateName)) {
      cout << "RooSimPdfBuilder::buildPdf: WARNING: multiple PDFs specified for state " 
	   << stateName << ", only first will be used" << endl ;
      continue ;
    }

    // Add pdf to list of models to be processed
    physModelSet.add(*physModel,kTRUE) ; // silence duplicate insertion warnings

    // Store state->pdf mapping    
    stateMap.addOwned(* new RooStringVar(stateName,stateName,physName)) ;

    // Continue with next mapping
    physName = strtok(0," ") ;
    if (first) {
      first = kFALSE ;
    } else if (physCat==0) {
      cout << "RooSimPdfBuilder::buildPdf: WARNING: without physCat specification, only the first model will be used" << endl ;
      break ;
    }
  }
  cout << "RooSimPdfBuilder::buildPdf: list of physics models " ; physModelSet.Print("1") ;



  // Create list of dataset categories to be used in splitting
  RooArgSet splitCatSet ;
  strcpy(buf,((RooStringVar*)buildConfig.find("splitCats"))->getVal()) ;
  char *catName = strtok(buf," ") ;
  while(catName) {
    RooCategory* splitCat = dynamic_cast<RooCategory*>(dataVars->find(catName)) ;
    if (!splitCat) {
      cout << "RooSimPdfBuilder::buildPdf: ERROR requested split category " << catName 
	   << " is not a RooCategory in the dataset" << endl ;
      return 0 ;
    }
    splitCatSet.add(*splitCat) ;
    catName = strtok(0," ") ;
  }
  if (physCat) splitCatSet.add(*physCat) ;
  RooSuperCategory masterSplitCat("masterSplitCat","Master splitting category",splitCatSet) ;
  
  cout << "RooSimPdfBuilder::buildPdf: list of splitting categories " ; splitCatSet.Print("1") ;



  TList customizerList ;

  // Loop over requested physics models and build components
  TIterator* physIter = physModelSet.createIterator() ;
  RooAbsPdf* physModel ;
  while(physModel=(RooAbsPdf*)physIter->Next()) {
    cout << "RooSimPdfBuilder::buildPdf: processing physics model " << physModel->GetName() << endl ;

    RooCustomizer* physCustomizer = new RooCustomizer(*physModel,masterSplitCat,_splitLeafList) ;
    customizerList.Add(physCustomizer) ;

    // Parse the splitting rules for this physics model
    RooStringVar* ruleStr = (RooStringVar*) buildConfig.find(physModel->GetName()) ;
    if (ruleStr) {
      strcpy(buf,ruleStr->getVal()) ;

      char *tokenPtr(0) ;
      char* token = strtok_r(buf," ",&tokenPtr) ;
      
      enum Mode { SplitCat, Colon, ParamList } ;
      Mode mode(SplitCat) ;

      char* splitCatName ;
      RooAbsCategoryLValue* splitCat ;

      while(token) {
	switch (mode) {
	case SplitCat:
	  {
	    splitCatName = token ;
	   	    
	    if (strchr(splitCatName,',')) {
	      // Composite splitting category
	      
	      // Check if already instantiated
	      splitCat = (RooAbsCategoryLValue*) _compSplitCatSet.find(splitCatName) ;	      
	      TString origCompCatName(splitCatName) ;
	      if (!splitCat) {
		// Build now
		char *tokptr(0) ;
		char *catName = strtok_r(token,",",&tokptr) ;
		RooArgSet compCatSet ;
		while(catName) {
		  RooAbsArg* cat = splitCatSet.find(catName) ;
		  if (!cat) {
		    cout << "RooSimPdfBuilder::buildPdf: ERROR " << catName
			 << " is not a valid category for splitting" << endl ;
		    customizerList.Delete() ;
		    return 0 ;
		  }
		  compCatSet.add(*cat) ;
		  catName = strtok_r(0,",",&tokptr) ;
		}		
		splitCat = new RooSuperCategory(origCompCatName,origCompCatName,compCatSet) ;
		_compSplitCatSet.addOwned(*splitCat) ;
		//cout << "composite splitcat: " << splitCat->GetName() ;
	      }
	    } else {
	      // Fundamental splitting category
	      splitCat = (RooAbsCategoryLValue*) splitCatSet.find(splitCatName) ;
	      if (!splitCat) {
		cout << "RooSimPdfBuilder::buildPdf: ERROR splitting category " 
		     << splitCatName << " not found in the splitcat list" << endl ;
		customizerList.Delete() ;
		return 0 ;
	      }
	    }
	    
	    mode = Colon ;
	    break ;
	  }
	case Colon:
	  {
	    if (strcmp(token,":")) {
	      cout << "RooSimPdfBuilder::buildPdf: ERROR in parsing, expected ':' after " 
		   << splitCat << ", found " << token << endl ;
	      customizerList.Delete() ;
	      return 0 ;	    
	    }
	    mode = ParamList ;
	    break ;
	  }
	case ParamList:
	  {
	    // Verify the validity of the parameter list and build the corresponding argset
	    RooArgSet splitParamList ;
	    RooArgSet* paramList = physModel->getParameters(dataVars) ;

	    char *tokptr(0) ;
	    char *paramName = strtok_r(token,",",&tokptr) ;
	    while(paramName) {
	      RooAbsArg* param = paramList->find(paramName) ;
	      if (!param) {
		cout << "RooSimPdfBuilder::buildPdf: ERROR " << paramName 
		     << " is not a parameter of physics model " << physModel->GetName() << endl ;
		delete paramList ;
		customizerList.Delete() ;
		return 0 ;
	      }
	      splitParamList.add(*param) ;
	      paramName = strtok_r(0,",",&tokptr) ;
	    }

	    // Add the rule to the appropriate customizer ;
	    physCustomizer->splitArgs(splitParamList,*splitCat) ;

	    mode = SplitCat ;
	    break ;
	  }
	}
	token = strtok_r(0," ",&tokenPtr) ;
      }
      if (mode!=SplitCat) {
	cout << "RooSimPdfBuilder::buildPdf: ERROR in parsing, expected " 
	     << (mode==Colon?":":"parameter list") << " after " << token << endl ;
      }

      RooArgSet* paramSet = physModel->getParameters(dataVars) ;
    } else {
      cout << "RooSimPdfBuilder::buildPdf: no splitting rules for " << physModel->GetName() << endl ;
    }    
  }
  
  cout << "RooSimPdfBuilder::buildPdf: configured customizers for all physics models" << endl ;
  customizerList.Print() ;

  // Create fit category from physCat and splitCatList ;
  RooArgSet fitCatList ;
  if (physCat) fitCatList.add(*physCat) ;
  fitCatList.add(splitCatSet) ;
  RooSuperCategory *fitCat = new RooSuperCategory("fitCat","fitCat",fitCatList) ;

  // Create master PDF 
  RooSimultaneous* simPdf = new RooSimultaneous("simPdf","simPdf",*fitCat) ;

  // Add component PDFs to master PDF
  TIterator* fcIter = fitCat->typeIterator() ;

  RooCatType* fcState ;  
  while(fcState=(RooCatType*)fcIter->Next()) {
    // Select fitCat state
    fitCat->setLabel(fcState->GetName()) ;
    
    // Select appropriate PDF for this physCat state
    RooCustomizer* physCustomizer ;
    if (physCat) {      
      RooStringVar* physNameVar = (RooStringVar*) stateMap.find(physCat->getLabel()) ;
      if (!physNameVar) continue ;
      physCustomizer = (RooCustomizer*) customizerList.FindObject(physNameVar->getVal());  
    } else {
      physCustomizer = (RooCustomizer*) customizerList.First() ;
    }

    cout << "RooSimPdfBuilder::buildPdf: Customizing physics model " << physCustomizer->GetName() 
	 << " for mode " << fcState->GetName() << endl ;    

    // Customizer PDF for current state and add to master simPdf
    RooAbsPdf* fcPdf = (RooAbsPdf*) physCustomizer->build(masterSplitCat.getLabel()) ;
    simPdf->addPdf(*fcPdf,fcState->GetName()) ;
  }

  // Move customizers (owning the cloned branch node components) to the attic
  _retiredCustomizerList.AddAll(&customizerList) ;

  delete physIter ;
  return simPdf ;
}





RooSimPdfBuilder::~RooSimPdfBuilder() 
{
}
 

