/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPdfCustomizer.cc,v 1.10 2001/09/28 21:59:28 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jul-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --

#include "TString.h"

#include "RooFitCore/RooAbsCategoryLValue.hh" 
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"

#include "RooFitCore/RooPdfCustomizer.hh"

ClassImp(RooPdfCustomizer) 
;


RooPdfCustomizer::RooPdfCustomizer(const RooAbsArg& pdf, const RooAbsCategoryLValue& masterCat, RooArgSet& splitLeafs) :
  _masterPdf((RooAbsArg*)&pdf), _masterCat((RooAbsCategoryLValue*)&masterCat), _cloneLeafList(&splitLeafs),
  _masterBranchList("masterBranchList"), _masterLeafList("masterLeafList"), _masterUnsplitLeafList("masterUnsplitLeafList"), 
  _cloneBranchList("cloneBranchList"), _sterile(kFALSE)
{
  initialize() ;
}



RooPdfCustomizer::RooPdfCustomizer(const RooAbsArg& pdf, const char* name) :
  _masterPdf((RooAbsArg*)&pdf), _masterCat(0), _cloneLeafList(0),
  _masterBranchList("masterBranchList"), _masterLeafList("masterLeafList"), _masterUnsplitLeafList("masterUnsplitLeafList"), 
  _cloneBranchList("cloneBranchList"), _sterile(kTRUE), _name(name)
{
  initialize() ;
}




void RooPdfCustomizer::initialize() 
{
  _masterPdf->leafNodeServerList(&_masterLeafList) ;
  _masterPdf->leafNodeServerList(&_masterUnsplitLeafList) ;
  _masterPdf->branchNodeServerList(&_masterBranchList) ;

  _masterLeafListIter = _masterLeafList.createIterator() ;
  _masterBranchListIter = _masterBranchList.createIterator() ;
}



RooPdfCustomizer::~RooPdfCustomizer() 
{
  delete _masterLeafListIter ;
  delete _masterBranchListIter ;

  _cloneBranchList.Delete() ;
  //_cloneLeafList.Delete() ;
}


  
void RooPdfCustomizer::splitArgs(const RooArgSet& set, const RooAbsCategory& splitCat) 
{
  if (_sterile) {
    cout << "RooPdfCustomizer::splitArgs(" << _name 
	 << ") ERROR cannot set spitting rules on this sterile customizer" << endl ;
    return ;
  }
  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()){
    splitArg(*arg,splitCat) ;
  }
}


void RooPdfCustomizer::splitArg(const RooAbsArg& arg, const RooAbsCategory& splitCat) 
{
  if (_sterile) {
    cout << "RooPdfCustomizer::splitArg(" << _name 
	 << ") ERROR cannot set spitting rules on this sterile customizer" << endl ;
    return ;
  }

  _splitArgList.Add((RooAbsArg*)&arg) ;
  _splitCatList.Add((RooAbsCategory*)&splitCat) ;
}


void RooPdfCustomizer::replaceArg(const RooAbsArg& orig, const RooAbsArg& subst) 
{
  _replaceArgList.Add((RooAbsArg*)&orig) ;
  _replaceSubList.Add((RooAbsArg*)&subst) ;
}


RooArgSet* RooPdfCustomizer::fullParamList(const RooArgSet* depList) const 
{
  TString listName("Variable list for ") ;
  listName.Append(_masterPdf->GetName()) ;
  listName.Append(" clones") ;
  RooArgList list(listName) ;

  list.add(_masterUnsplitLeafList) ;
  list.add(*_cloneLeafList) ;

  TIterator* iter = depList->createIterator() ;
  RooAbsArg* dep ;
  while (dep=(RooAbsArg*)iter->Next()) {
    RooAbsArg* dep2 = list.find(dep->GetName()) ;
    if (dep2) list.remove(*dep2) ;
  }
  delete iter ;
  
  list.sort() ;
  return new RooArgSet(list) ;
}





RooAbsArg* RooPdfCustomizer::build(Bool_t verbose) 
{
  return doBuild(_name,verbose) ;
}



RooAbsArg* RooPdfCustomizer::build(const char* masterCatState, Bool_t verbose) 
{
  if (_sterile) {
    cout << "RooPdfCustomizer::build(" << _name 
	 << ") ERROR cannot use leaf spitting build() on this sterile customizer" << endl ;
    return 0 ;
  }

  if (_masterCat->setLabel(masterCatState)) {
    cout << "RooPdfCustomizer::build(" << _masterPdf->GetName() << "): ERROR label '" << masterCatState 
	 << "' not defined for master splitting category " << _masterCat->GetName() << endl ;
    return 0 ;
  }

  return doBuild(masterCatState,verbose) ;
}


RooAbsArg* RooPdfCustomizer::doBuild(const char* masterCatState, Bool_t verbose) 
{
  // Set masterCat to given state

  // Find leafs that must be split according to provided description, Clone leafs, change their names
  RooArgSet masterLeafsToBeSplit("masterLeafsToBeSplit") ;
  RooArgSet masterLeafsToBeReplaced("masterLeafsToBeReplaced") ;
  RooArgSet masterReplacementLeafs("masterReplacementLeafs") ;
  RooArgSet clonedMasterLeafs("clonedMasterLeafs") ;
  _masterLeafListIter->Reset() ;
  RooAbsArg* leaf ;
  while(leaf=(RooAbsArg*)_masterLeafListIter->Next()) {
    RooAbsArg* splitArg = !_sterile?(RooAbsArg*) _splitArgList.FindObject(leaf->GetName()):0 ;
    if (splitArg) {
      RooAbsCategory* splitCat = (RooAbsCategory*) _splitCatList.At(_splitArgList.IndexOf(splitArg)) ;
      if (verbose) {
	cout << "RooPdfCustomizer::build(" << _masterPdf->GetName() 
	     << "): PDF parameter " << leaf->GetName() << " is split by category " << splitCat->GetName() << endl ;
      }

      // Take this leaf out of the unsplit list
      _masterUnsplitLeafList.remove(*leaf,kTRUE) ;
      
      TString newName(leaf->GetName()) ;
      newName.Append("_") ;
      newName.Append(splitCat->getLabel()) ;	

      // Check if this leaf instance already exists
      if (_cloneLeafList->find(newName)) {

	// Copy instance to one-time use list for this build
	clonedMasterLeafs.add(*_cloneLeafList->find(newName)) ;

      } else {

	TString newTitle(leaf->GetTitle()) ;
	newTitle.Append(" (") ;
	newTitle.Append(splitCat->getLabel()) ;
	newTitle.Append(")") ;
      
	// Create a new clone
	RooAbsArg* clone = (RooAbsArg*) leaf->Clone(newName.Data()) ;
	clone->SetTitle(newTitle) ;

	// Affix attribute with old name to clone to support name changing server redirect
	TString nameAttrib("ORIGNAME:") ;
	nameAttrib.Append(leaf->GetName()) ;
	clone->setAttribute(nameAttrib) ;

	// Add to one-time use list and life-time use list
	clonedMasterLeafs.add(*clone) ;
	_cloneLeafList->add(*clone) ;	
      }
      masterLeafsToBeSplit.add(*leaf) ;     
    }

    RooAbsArg* replaceArg = (RooAbsArg*) _replaceArgList.FindObject(leaf->GetName()) ;
    if (replaceArg) {
      RooAbsArg* substArg = (RooAbsArg*) _replaceSubList.At(_replaceArgList.IndexOf(replaceArg)) ;
      if (verbose) {
	cout << "RooPdfCustomizer::build(" << _masterPdf->GetName() 
	     << "): PDF leaf " << leaf->GetName() << " will be replaced by " << substArg->GetName() << endl ;
      }

      // Affix attribute with old name to support name changing server redirect
      TString nameAttrib("ORIGNAME:") ;
      nameAttrib.Append(leaf->GetName()) ;
      substArg->setAttribute(nameAttrib) ;

      // Add to list
      masterLeafsToBeReplaced.add(*leaf) ;
      masterReplacementLeafs.add(*substArg) ;
    }
  }
  if (!_sterile) _cloneLeafList->add(clonedMasterLeafs) ;


  // Find branches that are affected by splitting and must be cloned
  RooArgSet masterBranchesToBeCloned("masterBranchesToBeCloned") ;
  _masterBranchListIter->Reset() ;
  RooAbsArg* branch ;
  while(branch=(RooAbsArg*)_masterBranchListIter->Next()) {
    if (branch->dependsOn(masterLeafsToBeSplit)) {
      if (verbose) {
	cout << "RooPdfCustomizer::build(" << _masterPdf->GetName() << ") Component PDF " 
	     << branch->IsA()->GetName() << "::" << branch->GetName() << " cloned: depends on a split parameter" << endl ;
      }
      masterBranchesToBeCloned.add(*branch) ;
    } else if (branch->dependsOn(masterLeafsToBeReplaced)) {
      if (verbose) {
	cout << "RooPdfCustomizer::build(" << _masterPdf->GetName() << ") Component PDF " 
	     << branch->IsA()->GetName() << "::" << branch->GetName() << " cloned: depends on a replaced parameter" << endl ;
      }
      masterBranchesToBeCloned.add(*branch) ;
    }
  }

  // Clone branches, changes their names 
  RooAbsArg* cloneTopPdf(0) ;
  RooArgSet clonedMasterBranches("clonedMasterBranches") ;
  TIterator* iter = masterBranchesToBeCloned.createIterator() ;
  while(branch=(RooAbsArg*)iter->Next()) {
    TString newName(branch->GetName()) ;
    newName.Append("_") ;
    newName.Append(masterCatState) ;

    // Affix attribute with old name to clone to support name changing server redirect
    RooAbsArg* clone = (RooAbsArg*) branch->Clone(newName.Data()) ;
    TString nameAttrib("ORIGNAME:") ;
    nameAttrib.Append(branch->GetName()) ;
    clone->setAttribute(nameAttrib) ;

    clonedMasterBranches.add(*clone) ;      

    // Save pointer to clone of top-level pdf
    if (branch==_masterPdf) cloneTopPdf=(RooAbsArg*)clone ;
  }
  delete iter ;
  _cloneBranchList.add(clonedMasterBranches) ;


  // Reconnect cloned branches to each other and to cloned leafs
  iter = clonedMasterBranches.createIterator() ;
  while(branch=(RooAbsArg*)iter->Next()) {
    branch->redirectServers(clonedMasterBranches,kFALSE,kTRUE) ;
    branch->redirectServers(clonedMasterLeafs,kFALSE,kTRUE) ;
    branch->redirectServers(masterReplacementLeafs,kFALSE,kTRUE) ;
  }
  delete iter ;  


  return cloneTopPdf ;
}
