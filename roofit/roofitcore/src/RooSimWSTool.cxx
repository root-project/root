/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --


#include "RooFit.h"
#include "RooSimWSTool.h"
#include "RooMsgService.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooStringVar.h"
#include "RooSuperCategory.h"
#include "RooCatType.h"
#include "RooCustomizer.h"
#include "RooMultiCategory.h"
#include "RooSimultaneous.h"
#include "RooGlobalFunc.h"
#include "RooFracRemainder.h"

ClassImp(RooSimWSTool) 
ClassImp(RooSimWSTool::BuildConfig) 
ClassImp(RooSimWSTool::MultiBuildConfig) 
ClassImp(RooSimWSTool::SplitRule) 
ClassImp(RooSimWSTool::ObjBuildConfig) 
ClassImp(RooSimWSTool::ObjSplitRule) 
;

using namespace std ;

RooSimWSTool::RooSimWSTool(RooWorkspace& ws) : _ws(&ws) 
{
}


RooSimWSTool::~RooSimWSTool() 
{
  // Destructor
}


RooSimultaneous* RooSimWSTool::build(const char* simPdfName, const char* protoPdfName, const RooCmdArg& arg1,const RooCmdArg& arg2,
					const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6)
{
  BuildConfig bc(protoPdfName,arg1,arg2,arg3,arg4,arg5,arg6) ;
  return build(simPdfName,bc) ;
}



RooSimultaneous* RooSimWSTool::build(const char* simPdfName,BuildConfig& bc) 
{
  ObjBuildConfig* obc = validateConfig(bc) ;
  if (!obc) return 0 ;
  
  obc->print() ;
  
  RooSimultaneous* ret =  executeBuild(simPdfName,*obc) ;

  delete obc ;
  return ret ;
}



RooSimWSTool::ObjBuildConfig* RooSimWSTool::validateConfig(BuildConfig& bc)
{
  // Create empty object version of build config
  ObjBuildConfig* obc = new ObjBuildConfig ;

  if (bc._masterCatName.length()>0) {
    obc->_masterCat = _ws->cat(bc._masterCatName.c_str()) ;
    if (!obc->_masterCat) {
      coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: associated workspace " << _ws->GetName() 
			    << " does not contain a category named " << bc._masterCatName 
			    << " that was designated as master index category in the build configuration" << endl ;
      delete obc ;
      return 0 ;
    }
  } else {
    obc->_masterCat = 0 ;
  }
  
  map<string,SplitRule>::iterator pdfiter ;
  // Check that we have the p.d.f.s
  for (pdfiter = bc._pdfmap.begin() ; pdfiter != bc._pdfmap.end() ; ++pdfiter) {    

    // Check that p.d.f exists
    RooAbsPdf* pdf = _ws->pdf(pdfiter->second.GetName()) ;
    if (!pdf) {
      coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: associated workspace " << _ws->GetName() 
			    << " does not contain a pdf named " << pdfiter->second.GetName() << endl ;
      delete obc ;
      return 0 ;
    }      

    // Create empty object version of split rule set
    ObjSplitRule osr ;

    // Convert names of parameters and splitting categories to objects in workspace, fill object split rule
    SplitRule& sr = pdfiter->second ;

    map<string, pair<list<string>,string> >::iterator pariter ;
    for (pariter=sr._paramSplitMap.begin() ; pariter!=sr._paramSplitMap.end() ; ++pariter) {

      // Check that variable with given name exists in workspace
      RooAbsArg* farg = _ws->fundArg(pariter->first.c_str()) ;
      if (!farg) {
	coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: associated workspace " << _ws->GetName() 
			      << " does not contain a variable named " << pariter->first.c_str() 
			      << " as specified in splitting rule of parameter " << pariter->first << " of p.d.f " << pdf << endl ;
	delete obc ;
	return 0 ;
      } 

      // Check that given variable is indeed related to given p.d.f
      if (!pdf->dependsOn(*farg)) {
	coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: specified parameter " << pariter->first 
			      << " in split is not function of p.d.f " << pdf->GetName() << endl ;
	delete obc ;
	return 0 ;
      }
      
      
      RooArgSet splitCatSet ;
      list<string>::iterator catiter ;
      for (catiter = pariter->second.first.begin() ; catiter!=pariter->second.first.end() ; ++catiter) {
	RooAbsCategory* cat = _ws->catfunc(catiter->c_str()) ;
	if (!cat) {
	  coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: associated workspace " << _ws->GetName() 
				<< " does not contain a category named " << catiter->c_str() 
				<< " as specified in splitting rule of parameter " << pariter->first << " of p.d.f " << pdf << endl ;
	  delete obc ;
	  return 0 ;
	}
	splitCatSet.add(*cat) ;
      }      

      // Check if composite splitCatSet does not contain category functions that depend on other categories used in the same split
      TIterator* iter = splitCatSet.createIterator() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)iter->Next())) {
	RooArgSet tmp(splitCatSet) ;
	tmp.remove(*arg) ;
	if (arg->dependsOnValue(tmp)) {
	  coutE(InputArguments) << "RooSimWSTool::build(" << GetName() << ") ERROR: Ill defined split: splitting category function " << arg->GetName() 
				<< " used in composite split " << splitCatSet << " of parameter " << farg->GetName() << " of pdf " << pdf->GetName() 
				<< " depends on one or more of the other splitting categories in the composite split" << endl ;
	  delete obc ;
	  delete iter ;
	  return 0 ;
	}
      }
      delete iter ;

      // If a constrained split is specified, check that split parameter is a real-valued type
      if (pariter->second.second.size()>0) {
	if (!dynamic_cast<RooAbsReal*>(farg)) {
	  coutE(InputArguments) << "RooSimWSTool::build(" << GetName() << ") ERROR: Constrained split specified in non real-valued parameter " << farg->GetName() << endl ;
	  delete obc ;
	  return 0 ;
	}
      }

      // Fill object build config with object split rule
      osr._paramSplitMap[farg].first.add(splitCatSet) ;
      osr._paramSplitMap[farg].second = pariter->second.second ;

      // For multi-pdf configurations, check that the master index state name associated with this p.d.f exists as a state in the master category
      if (obc->_masterCat) {
	list<string>::iterator misi ;
	for (misi=sr._miStateNameList.begin() ; misi!=sr._miStateNameList.end() ; ++misi) {
	  const RooCatType* ctype = obc->_masterCat->lookupType(misi->c_str(),kFALSE) ;
	  if (ctype==0) {	  
	    coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: master index category " << obc->_masterCat->GetName() 
				  << " does not have a state named " << *misi << " which was specified as state associated with p.d.f " 
				  << sr.GetName() << endl ;
	    delete obc ;
	    return 0 ;	    
	  }
	  osr._miStateList.push_back(ctype) ;
	}	
      }

      // Add specified split cats to global list of all splitting categories
      obc->_usedSplitCats.add(splitCatSet,kTRUE) ;
      
    }

    obc->_pdfmap[pdf] = osr ;

  }

  // Check validity of build restriction specifications, if any
  map<string,string>::iterator riter ;
  for (riter=bc._restr.begin() ; riter!=bc._restr.end() ; ++riter) {
    RooCategory* cat = _ws->cat(riter->first.c_str()) ;
    if (!cat) {
      coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: associated workspace " << _ws->GetName() 
			    << " does not contain a category named " << riter->first
			    << " for which build was requested to be restricted to states " << riter->second << endl ;
      delete obc ;
      return 0 ;
    }	

    char buf[4096] ;
    list<const RooCatType*> rlist ;
    strcpy(buf,riter->second.c_str()) ;
    
    char* tok = strtok(buf,",") ;
    while(tok) {
      const RooCatType* ctype = cat->lookupType(tok,kFALSE) ;
      if (!ctype) {
	coutE(ObjectHandling) << "RooSimWSTool::build(" << GetName() << ") ERROR: restricted build category " << cat->GetName() 
			      << " does not have state " << tok << " as specified in restriction list" << endl ;	
	delete obc ;
	return 0 ;
      }
      rlist.push_back(ctype) ;
      tok = strtok(0,",") ;
    }
    
    obc->_restr[cat] = rlist ;
  }
   
  return obc ;
}




RooSimultaneous* RooSimWSTool::executeBuild(const char* simPdfName, ObjBuildConfig& obc)
{
  RooArgSet cleanupList ;

  RooAbsCategoryLValue* physCat = obc._masterCat ;

  RooArgSet physModelSet ;
  map<string,RooAbsPdf*> stateMap ;

  map<RooAbsPdf*,ObjSplitRule>::iterator physIter = obc._pdfmap.begin() ;
  while(physIter!=obc._pdfmap.end()) {
    
    RooAbsPdf* physModel = physIter->first ;
    physModelSet.add(*physModel,kTRUE) ; // silence duplicate insertion warnings
    
    list<const RooCatType*>::iterator stiter ;
    for (stiter=physIter->second._miStateList.begin() ; stiter!=physIter->second._miStateList.end() ; ++stiter) {
      stateMap[(*stiter)->GetName()] = physModel ;
    }

    // Continue with next mapping
    ++physIter ;
  }
  coutI(ObjectHandling) << "RooSimWSTool::executeBuild: list of prototype pdfs " << physModelSet << endl ;

  RooArgSet splitCatSet(obc._usedSplitCats) ;
  if (physCat) splitCatSet.add(*physCat) ;
  
  RooSuperCategory masterSplitCat("masterSplitCat","Master splitting category",splitCatSet) ;
    coutI(ObjectHandling) << "RooSimWSTool::executeBuild: list of splitting categories " << splitCatSet << endl ;

  RooArgSet splitNodeListOwned ; // owns all newly created components
  RooArgSet splitNodeListAll ; // all leaf nodes, preload with ws contents to auto-connect existing specializations
  TList* customizerList = new TList ;

  // Loop over requested physics models and build components
  TIterator* physMIter = physModelSet.createIterator() ;
  RooAbsPdf* physModel ;
  while((physModel=(RooAbsPdf*)physMIter->Next())) {
    coutI(ObjectHandling) << "RooSimPdfBuilder::executeBuild: processing prototype pdf " << physModel->GetName() << endl ;

    RooCustomizer* physCustomizer = new RooCustomizer(*physModel,masterSplitCat,splitNodeListOwned,&splitNodeListAll) ;
    customizerList->Add(physCustomizer) ;

    map<RooAbsArg*, pair<RooArgSet,string> >::iterator splitIter ;
    for (splitIter = obc._pdfmap[physModel]._paramSplitMap.begin() ; splitIter != obc._pdfmap[physModel]._paramSplitMap.end() ; ++splitIter) {

      // If split is composite, first make multicategory with name 'A,B,C' and insert in WS
      
      // Construct name of (composite) split category (function)
      RooArgSet& splitCatSet = splitIter->second.first ;
      string splitName = makeSplitName(splitCatSet) ;

      // If composite split object does not exist yet, create it now
      RooAbsCategory* splitCat = _ws->catfunc(splitName.c_str()) ;
      if (!splitCat) {
	splitCat = new RooMultiCategory(splitName.c_str(),splitName.c_str(),splitCatSet) ;
	cleanupList.addOwned(*splitCat) ;
	_ws->import(*splitCat) ;
      }
            
      // If remainder category needs to be made, create RFV of appropriate for that and insert in WS
      if(splitIter->second.second.size()>0) {
	
	// Check that specified split name is in fact valid
	if (!splitCat->lookupType(splitIter->second.second.c_str())) {
	  coutE(InputArguments) << "RooSimWSTool::executeBuild(" << GetName() << ") ERROR: name of remainder state for constrained split, '" 
				<< splitIter->second.second << "' , does not match any state name of (composite) split category " << splitCat->GetName() << endl ;
	  return 0 ;
	}

	// First build manually the specializations of all non-remainder states, as the remainder state depends on these
	RooArgSet fracLeafList ;
	TIterator* sctiter = splitCat->typeIterator() ;
	RooCatType* type ;
	while((type=(RooCatType*)sctiter->Next())) {
	  
	  // Skip remainder state
	  if (splitIter->second.second == type->GetName()) continue ;
	  	  
	  // Construct name of split leaf
	  TString splitLeafName(splitIter->first->GetName()) ;
	  splitLeafName.Append("_") ;
	  splitLeafName.Append(type->GetName()) ;
	  
	  // Check if split leaf already exists	  
	  RooAbsArg* splitLeaf = _ws->fundArg(splitLeafName) ;
	  if (!splitLeaf) {
	    // If not create it now
	    splitLeaf = (RooAbsArg*) splitIter->first->clone(splitLeafName) ;
	    _ws->import(*splitLeaf) ;
	  }
	  fracLeafList.add(*splitLeaf) ;
	}
	delete sctiter ;		
	

	// Build specialization for remainder state and insert in workspace 
	RooFracRemainder* fracRem = new RooFracRemainder(Form("%s_%s",splitIter->first->GetName(),splitIter->second.second.c_str()),"Remainder fraction",fracLeafList) ;
	cleanupList.addOwned(*fracRem) ;
	_ws->import(*fracRem) ;

      }
      

      // Add split definition to customizer
      physCustomizer->splitArgs(*splitIter->first,*splitCat) ;
    }
  }
  delete physMIter ;

  // List all existing workspace components as prebuilt items for the customizers at this point
  splitNodeListAll.add(_ws->components()) ;

  coutI(ObjectHandling)  << "RooSimWSTool::executeBuild: configured customizers for all prototype pdfs" << endl ;

  // Create fit category from physCat and splitCatList ;
  RooArgSet fitCatList ;
  if (physCat) fitCatList.add(*physCat) ;
  fitCatList.add(splitCatSet) ;
  TIterator* fclIter = fitCatList.createIterator() ;
  string mcatname = string(simPdfName) + "_index" ;
  RooSuperCategory *fitCat = new RooSuperCategory(mcatname.c_str(),mcatname.c_str(),fitCatList) ;
  cleanupList.addOwned(*fitCat) ;

  // Create master PDF 
  RooSimultaneous* simPdf = new RooSimultaneous(simPdfName,simPdfName,*fitCat) ;
  cleanupList.addOwned(*simPdf) ;
  
  // Add component PDFs to master PDF
  TIterator* fcIter = fitCat->typeIterator() ;

  RooCatType* fcState ;  
  while((fcState=(RooCatType*)fcIter->Next())) {
    // Select fitCat state
    fitCat->setLabel(fcState->GetName()) ;

    // Check if this fitCat state is selected
    fclIter->Reset() ;
    RooAbsCategory* splitCat ;
    Bool_t select(kFALSE) ;
    if (obc._restr.size()>0) {
      while((splitCat=(RooAbsCategory*)fclIter->Next())) {
	// Find selected state list 
	
	list<const RooCatType*> slist = obc._restr[splitCat] ;    
	if (slist.size()==0) {
	  continue ;
	}
	
	list<const RooCatType*>::iterator sli ;
	for (sli=slist.begin() ; sli!=slist.end() ; ++sli) {
	  if (string(splitCat->getLabel())==(*sli)->GetName()) {
	    select=kTRUE ;
	  }
	}      
      }
      if (!select) continue ;
    } else {
      select = kTRUE ;
    }
    
    // Select appropriate PDF for this physCat state
    RooCustomizer* physCustomizer ;
    if (physCat) {      
      RooAbsPdf* pdf = stateMap[physCat->getLabel()] ;
      if (pdf==0) continue ;
      physCustomizer = (RooCustomizer*) customizerList->FindObject(pdf->GetName());  
    } else {
      physCustomizer = (RooCustomizer*) customizerList->First() ;
    }

    coutI(ObjectHandling) << "RooSimWSTool::executeBuild: Customizing prototype pdf " << physCustomizer->GetName() 
			  << " for mode " << fcState->GetName() << endl ;    

    // Customizer PDF for current state and add to master simPdf
    RooAbsPdf* fcPdf = (RooAbsPdf*) physCustomizer->build(masterSplitCat.getLabel(),kFALSE) ;
    simPdf->addPdf(*fcPdf,fcState->GetName()) ;
  }
  delete fcIter ;

  _ws->import(*simPdf,obc._conflProtocol) ;

  // Delete customizers
  customizerList->Delete() ;
  delete customizerList ;

  return (RooSimultaneous*) _ws->pdf(simPdf->GetName()) ;
}


std::string RooSimWSTool::makeSplitName(const RooArgSet& splitCatSet) 
{
  string name ;

  TIterator* iter = splitCatSet.createIterator() ;
  RooAbsArg* arg ;
  Bool_t first=kTRUE ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (first) {
      first=kFALSE;
    } else {
      name += "," ;
    }
    name += arg->GetName() ;
  }
  delete iter ;

  return name ;
}


//----------------


void RooSimWSTool::SplitRule::splitParameter(const char* paramNameList, const char* categoryNameList) 
{
  char paramBuf[4096] ;
  char catBuf[4096] ;
  strcpy(paramBuf,paramNameList) ;
  strcpy(catBuf,categoryNameList) ;

  // First parse category list
  list<string> catList ;
  char* cat = strtok(catBuf,",") ;
  while(cat) {
    catList.push_back(cat) ;
    cat = strtok(0,",") ;
  }

  // Now parse parameter list
  char* param = strtok(paramBuf,",") ;
  while(param) {
    _paramSplitMap[param] = pair<list<string>,string>(catList,"") ;
    param = strtok(0,",") ;
  }
}

void RooSimWSTool::SplitRule::splitParameterConstrained(const char* paramNameList, const char* categoryNameList, const char* remainderStateName) 
{
  char paramBuf[4096] ;
  char catBuf[4096] ;
  strcpy(paramBuf,paramNameList) ;
  strcpy(catBuf,categoryNameList) ;

  // First parse category list
  list<string> catList ;
  char* cat = strtok(catBuf,",") ;
  while(cat) {
    catList.push_back(cat) ;
    cat = strtok(0,",") ;
  }

  // Now parse parameter list
  char* param = strtok(paramBuf,",") ;
  while(param) {
    _paramSplitMap[param] = pair<list<string>,string>(catList,remainderStateName) ;
    param = strtok(0,",") ;
  }
}

void RooSimWSTool::SplitRule::configure(const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
					   const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6)
{
  list<const RooCmdArg*> cmdList ;  
  cmdList.push_back(&arg1) ;  cmdList.push_back(&arg2) ;
  cmdList.push_back(&arg3) ;  cmdList.push_back(&arg4) ;
  cmdList.push_back(&arg5) ;  cmdList.push_back(&arg6) ;

  list<const RooCmdArg*>::iterator iter ;
  for (iter=cmdList.begin() ; iter!=cmdList.end() ; ++iter) {

    if ((*iter)->opcode()==0) continue ;

    string name = (*iter)->opcode() ;

    if (name=="SplitParam") {
      splitParameter((*iter)->getString(0),(*iter)->getString(1)) ;
    } else if (name=="SplitParamConstrained") {
      splitParameterConstrained((*iter)->getString(0),(*iter)->getString(1),(*iter)->getString(2)) ;
    }
  }
}


//----------------


RooSimWSTool::BuildConfig::BuildConfig(const char* pdfName, SplitRule& sr)
{
  internalAddPdf(pdfName,"",sr) ;
}

RooSimWSTool::BuildConfig::BuildConfig(const char* pdfName, const RooCmdArg& arg1,const RooCmdArg& arg2,
					  const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6)
{
  SplitRule sr(pdfName) ;
  sr.configure(arg1,arg2,arg3,arg4,arg5,arg6) ;
  internalAddPdf(pdfName,"",sr) ;
  _conflProtocol = RooFit::RenameConflictNodes(pdfName) ;

  list<const RooCmdArg*> cmdList ;  
  cmdList.push_back(&arg1) ;  cmdList.push_back(&arg2) ;
  cmdList.push_back(&arg3) ;  cmdList.push_back(&arg4) ;
  cmdList.push_back(&arg5) ;  cmdList.push_back(&arg6) ;

  list<const RooCmdArg*>::iterator iter ;
  for (iter=cmdList.begin() ; iter!=cmdList.end() ; ++iter) {
    if ((*iter)->opcode()==0) continue ;
    string name = (*iter)->opcode() ;
    if (name=="Restrict") {
      restrictBuild((*iter)->getString(0),(*iter)->getString(1)) ;
    }    
    if (name=="RenameConflictNodes") {
      _conflProtocol = *(*iter) ;
    }
  }
}

RooSimWSTool::BuildConfig::BuildConfig(const RooArgSet& /*legacyBuildConfig*/) 
{
  // Constructor to make BuildConfig from legacy RooSimPdfBuilder configuration
  // Empty for now
}

void RooSimWSTool::BuildConfig::internalAddPdf(const char* pdfName, const char* miStateNameList,SplitRule& sr) 
{
  char buf[4096] ;
  strcpy(buf,miStateNameList) ;
  
  char* tok = strtok(buf,",") ;
  while(tok) {
    sr._miStateNameList.push_back(tok) ;
    tok = strtok(0,",") ;
  }
    
  _pdfmap[pdfName] = sr ;  
}

void RooSimWSTool::BuildConfig::restrictBuild(const char* catName, const char* stateList) 
{
  _restr[catName] = stateList ;
}


//----------------


RooSimWSTool::MultiBuildConfig::MultiBuildConfig(const char* masterIndexCat)  
{
  _masterCatName = masterIndexCat ;
}


void RooSimWSTool::MultiBuildConfig::addPdf(const char* miStateList, const char* pdfName, const RooCmdArg& arg1,const RooCmdArg& arg2,
					       const RooCmdArg& arg3,const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6)
{
  SplitRule sr(pdfName) ;
  sr.configure(arg1,arg2,arg3,arg4,arg5,arg6) ;
  internalAddPdf(pdfName,miStateList,sr) ;
}



void RooSimWSTool::MultiBuildConfig::addPdf(const char* miStateList, const char* pdfName, SplitRule& sr) 
{
  internalAddPdf(pdfName,miStateList,sr) ;
}



//--------------------

RooSimWSTool::ObjSplitRule::~ObjSplitRule()
{
}


// -----------------------------------------


void RooSimWSTool::ObjBuildConfig::print()
{
  // --- Dump contents of object build config ---
  map<RooAbsPdf*,ObjSplitRule>::iterator ri ;
  for (ri = _pdfmap.begin() ; ri != _pdfmap.end() ; ++ri ) {    
    cout << "Splitrule for p.d.f " << ri->first->GetName() << endl ;
    map<RooAbsArg*,pair<RooArgSet,string> >::iterator csi ;
    for (csi = ri->second._paramSplitMap.begin() ; csi != ri->second._paramSplitMap.end() ; ++csi ) {    
      if (csi->second.second.length()>0) {
	cout << " parameter " << csi->first->GetName() << " is split with constraint in categories " << csi->second.first 
	     << " with remainder in state " << csi->second.second << endl ;      
      } else {
	cout << " parameter " << csi->first->GetName() << " is split with constraint in categories " << csi->second.first << endl ;      
      }
    }        
  }

  map<RooAbsCategory*,list<const RooCatType*> >::iterator riter ;
  for (riter=_restr.begin() ; riter!=_restr.end() ; ++riter) {
    cout << "Restricting build in category " << riter->first->GetName() << " to states " ;
    list<const RooCatType*>::iterator i ;
    for (i=riter->second.begin() ; i!=riter->second.end() ; i++) {
      if (i!=riter->second.begin()) cout << "," ;
      cout << (*i)->GetName() ;
    }
    cout << endl ;
  }

}


