/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCustomizer.cc,v 1.6 2001/12/13 23:37:31 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jul-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooCustomizer is a factory class to produce clones
// of a prototype composite PDF object with the same structure but
// different leaf servers (parameters or dependents)
//
// RooCustomizer supports two kinds of modifications:
// 
// -> replace(leaf_arg,repl_arg) 
// replaces each occurence of leaf_arg with repl_arg in the composite pdf.
//
// -> split(split_arg)
// is used when building multiple clones of the same prototype. Each
// occurrence of split_arg is replaceed with a clone of split_arg
// named split_arg_[MCstate], where [MCstate] is the name of the
// 'master category state' that indexes the clones to be built.
//
//
// [Example]
//
// Splitting is particularly useful when building simultaneous fits to
// subsets of the data sample with different background properties.
// In such a case, the user builds a single prototype PDF representing
// the structure of the signal and background and splits the dataset
// into categories with different background properties. Using
// RooCustomizer a PDF for each subfit can be constructed from the
// prototype that has same structure and signal parameters, but
// different instances of the background parameters: e.g.
//
//     ...
//     RooExponential bg("bg","background",x,alpha) ;
//     RooGaussian sig("sig","signal",x,mean,sigma) ;
//     RooAddPdf pdf("pdf","pdf",sig,bg,sigfrac) ;
//
//     RooDataSet data("data","dataset",RooArgSet(x,runblock),...)
//
//     RooCategory runblock("runblock","run block") ;
//     runblock.defineType("run1") ;
//     runblock.defineType("run2") ;
//
//     RooArgSet splitLeafs
//     RooCustomizer cust(pdf,runblock,splitLeafs)
//     cust.split(alpha,runblock)
//
//     RooAbsPdf* pdf_run1 = cust.build("run1") ;
//     RooAbsPdf* pdf_run2 = cust.build("run2") ;
//
//     RooSimultaneous simpdf("simpdf","simpdf",RooArgSet(*pdf_run1,*pdf_run2)) 
//
// If the master category state is a super category, leafs may be split
// by any subset of that master category. E.g. if the master category
// is 'A x B', leafs may be split by A, B or AxB.
//
// In addition to replacing leaf nodes, RooCustomizer clones all branch
// nodes that depend directly or indirectly on modified leaf nodes, so
// that the input pdf is untouched by each build operation.
//
// The customizer owns all the branch nodes including the returned top
// level node, so the customizer should live as longs as the cloned
// composites are needed.
//
// Any leaf nodes that are created by the customizer will be put into
// the leaf list that is passed into the customizers constructor (splitLeafs in
// the above example. The list owner is responsible for deleting these leaf
// nodes after the customizer is deleted.
//
//
// [Advanced techniques]
//
// By default the customizer clones the prototype leaf node when splitting a leaf,
// but the user can feed pre-defined split leafs in leaf list. These leafs
// must have the name <split_leaf>_<splitcat_label> to be picked up. The list
// of pre-supplied leafs may be partial, any missing split leafs will be auto
// generated.
//
// Another common construction is to have two prototype PDFs, each to be customized
// by a separate customizer instance, that share parameters. To ensure that
// the customized clones also share their respective split leafs, i.e.
//
//   PDF1(x,y;A) and PDF2(z,A)   ---> PDF1_run1(x,y,A_run1) and PDF2_run1(x,y,A_run1)
//                                    PDF1_run2(x,y,A_run2) and PDF2_run2(x,y,A_run2)
//
// feed the same split leaf list into both customizers. In that case the second customizer
// will pick up the split leafs instantiated by the first customizer and the link between
// the two PDFs is retained
//


#include "TString.h"

#include "RooFitCore/RooAbsCategoryLValue.hh" 
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"

#include "RooFitCore/RooCustomizer.hh"

ClassImp(RooCustomizer) 
;


RooCustomizer::RooCustomizer(const RooAbsArg& pdf, const RooAbsCategoryLValue& masterCat, RooArgSet& splitLeafs) :
  TNamed(pdf.GetName(),pdf.GetTitle()),
  _masterPdf((RooAbsArg*)&pdf), _masterCat((RooAbsCategoryLValue*)&masterCat), _cloneNodeList(&splitLeafs),
  _masterBranchList("masterBranchList"), _masterLeafList("masterLeafList"), 
  _cloneBranchList("cloneBranchList"), _sterile(kFALSE)
{
  // Constructor with masterCat state. Customizers created by this constructor offer the full functionality
  initialize() ;
}



RooCustomizer::RooCustomizer(const RooAbsArg& pdf, const char* name) :
  TNamed(pdf.GetName(),pdf.GetTitle()),
  _masterPdf((RooAbsArg*)&pdf), _masterCat(0), _cloneNodeList(0),
  _masterBranchList("masterBranchList"), _masterLeafList("masterLeafList"), 
  _cloneBranchList("cloneBranchList"), _sterile(kTRUE), _name(name)
{
  // Sterile Constructor. Customizers created by this constructor offer only the replace() method. The supplied
  // 'name' is used as suffix for any cloned branch nodes
  initialize() ;
}




void RooCustomizer::initialize() 
{
  // Initialization function
  _masterPdf->leafNodeServerList(&_masterLeafList) ;
  _masterPdf->branchNodeServerList(&_masterBranchList) ;

  _masterLeafListIter = _masterLeafList.createIterator() ;
  _masterBranchListIter = _masterBranchList.createIterator() ;
}



RooCustomizer::~RooCustomizer() 
{
  // Destructor

  delete _masterLeafListIter ;
  delete _masterBranchListIter ;

//   _cloneBranchList.Delete() ;
}


  
void RooCustomizer::splitArgs(const RooArgSet& set, const RooAbsCategory& splitCat) 
{
  // Split all args in 'set' by 'splitCat' states. 'splitCats' must be subset of
  // or equal to the master category supplied in the customizer constructor.
  //
  // Splitting is only available on customizers created with a master index category

  if (_sterile) {
    cout << "RooCustomizer::splitArgs(" << _name 
	 << ") ERROR cannot set spitting rules on this sterile customizer" << endl ;
    return ;
  }
  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()){
    splitArg(*arg,splitCat) ;
  }
}


void RooCustomizer::splitArg(const RooAbsArg& arg, const RooAbsCategory& splitCat) 
{
  // Split 'arg' by 'splitCat' states. 'splitCats' must be subset of
  // or equal to the master category supplied in the customizer constructor.
  //
  // Splitting is only available on customizers created with a master index category

  if (_splitArgList.FindObject(arg.GetName())) {
    cout << "RooCustomizer(" << GetName() << ") ERROR: multiple splitting rules defined for " 
	 << arg.GetName() << " only using first rule" << endl ;
    return ;
  }

  if (_sterile) {
    cout << "RooCustomizer::splitArg(" << _name 
	 << ") ERROR cannot set spitting rules on this sterile customizer" << endl ;
    return ;
  }

  _splitArgList.Add((RooAbsArg*)&arg) ;
  _splitCatList.Add((RooAbsCategory*)&splitCat) ;
}


void RooCustomizer::replaceArg(const RooAbsArg& orig, const RooAbsArg& subst) 
{
  // Replace any occurence of arg 'orig' with arg 'subst'

  if (_replaceArgList.FindObject(orig.GetName())) {
    cout << "RooCustomizer(" << GetName() << ") ERROR: multiple replacement rules defined for " 
	 << orig.GetName() << " only using first rule" << endl ;
    return ;
  }

  _replaceArgList.Add((RooAbsArg*)&orig) ;
  _replaceSubList.Add((RooAbsArg*)&subst) ;
}



RooAbsArg* RooCustomizer::build(Bool_t verbose) 
{
  // Build a clone of the prototype executing all registered 'replace' rules
  // If verbose is set a message is printed for each leaf or branch node
  // modification. The returned composite arg is owned by the customizer
  return doBuild(_name,verbose) ;
}



RooAbsArg* RooCustomizer::build(const char* masterCatState, Bool_t verbose) 
{
  // Build a clone of the prototype executing all registered 'replace' rules
  // and 'split' rules for the masterCat state named 'masterCatState'.
  // If verbose is set a message is printed for each leaf or branch node
  // modification. The returned composite arg is owned by the customizer.
  // This function cannot be called on customizer build with the sterile constructor.

  if (_sterile) {
    cout << "RooCustomizer::build(" << _name 
	 << ") ERROR cannot use leaf spitting build() on this sterile customizer" << endl ;
    return 0 ;
  }

  // Set masterCat to given state
  if (_masterCat->setLabel(masterCatState)) {
    cout << "RooCustomizer::build(" << _masterPdf->GetName() << "): ERROR label '" << masterCatState 
	 << "' not defined for master splitting category " << _masterCat->GetName() << endl ;
    return 0 ;
  }

  return doBuild(masterCatState,verbose) ;
}


RooAbsArg* RooCustomizer::doBuild(const char* masterCatState, Bool_t verbose) 
{
  // Protected build engine

  // Find nodes that must be split according to provided description, Clone nodes, change their names
  RooArgSet masterNodesToBeSplit("masterNodesToBeSplit") ;
  RooArgSet masterNodesToBeReplaced("masterNodesToBeReplaced") ;
  RooArgSet masterReplacementNodes("masterReplacementNodes") ;
  RooArgSet clonedMasterNodes("clonedMasterNodes") ;
  _masterLeafListIter->Reset() ;
  RooAbsArg* node ;

  RooArgSet nodeList(_masterLeafList) ;
  nodeList.add(_masterBranchList) ;
  TIterator* nIter = nodeList.createIterator() ;

  while(node=(RooAbsArg*)nIter->Next()) {
    RooAbsArg* splitArg = !_sterile?(RooAbsArg*) _splitArgList.FindObject(node->GetName()):0 ;
    if (splitArg) {
      RooAbsCategory* splitCat = (RooAbsCategory*) _splitCatList.At(_splitArgList.IndexOf(splitArg)) ;
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() 
	     << "): tree node " << node->GetName() << " is split by category " << splitCat->GetName() << endl ;
      }
      
      TString newName(node->GetName()) ;
      newName.Append("_") ;
      newName.Append(splitCat->getLabel()) ;	

      // Check if this node instance already exists
      if (_cloneNodeList->find(newName)) {

	// Copy instance to one-time use list for this build
	RooAbsArg* specNode = _cloneNodeList->find(newName) ;
	clonedMasterNodes.add(*specNode) ;
	if (verbose) {
	  cout << "RooCustomizer::build(" << _masterPdf->GetName() 
	       << ") Adding existing node specialization " << newName << " to clonedMasterNodes" << endl ;
	}

	// Affix attribute with old name to clone to support name changing server redirect
	TString nameAttrib("ORIGNAME:") ;
	nameAttrib.Append(node->GetName()) ;
	specNode->setAttribute(nameAttrib) ;

      } else {

	if (node->isDerived()) {
	  cout << "RooCustomizer::build(" << _masterPdf->GetName() 
	       << "): WARNING: branch node " << node->GetName() << " is split but has no pre-defined specializations" << endl ;
	}

	TString newTitle(node->GetTitle()) ;
	newTitle.Append(" (") ;
	newTitle.Append(splitCat->getLabel()) ;
	newTitle.Append(")") ;
      
	// Create a new clone
	RooAbsArg* clone = (RooAbsArg*) node->Clone(newName.Data()) ;
	clone->SetTitle(newTitle) ;

	// Affix attribute with old name to clone to support name changing server redirect
	TString nameAttrib("ORIGNAME:") ;
	nameAttrib.Append(node->GetName()) ;
	clone->setAttribute(nameAttrib) ;

	// Add to one-time use list and life-time use list
	clonedMasterNodes.add(*clone) ;
	_cloneNodeList->addOwned(*clone) ;	
      }
      masterNodesToBeSplit.add(*node) ;     
    }

    RooAbsArg* replaceArg = (RooAbsArg*) _replaceArgList.FindObject(node->GetName()) ;
    if (replaceArg) {
      RooAbsArg* substArg = (RooAbsArg*) _replaceSubList.At(_replaceArgList.IndexOf(replaceArg)) ;
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() 
	     << "): tree node " << node->GetName() << " will be replaced by " << substArg->GetName() << endl ;
      }

      // Affix attribute with old name to support name changing server redirect
      TString nameAttrib("ORIGNAME:") ;
      nameAttrib.Append(node->GetName()) ;
      substArg->setAttribute(nameAttrib) ;

      // Add to list
      masterNodesToBeReplaced.add(*node) ;
      masterReplacementNodes.add(*substArg) ;
    }
  }
  if (!_sterile) _cloneNodeList->addOwned(clonedMasterNodes) ;


  // Find branches that are affected by splitting and must be cloned
  RooArgSet masterBranchesToBeCloned("masterBranchesToBeCloned") ;
  _masterBranchListIter->Reset() ;
  RooAbsArg* branch ;
  while(branch=(RooAbsArg*)_masterBranchListIter->Next()) {
    
    // If branch is split itself, don't handle here
    if (masterNodesToBeSplit.find(branch->GetName())) {
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() << ") Branch node " << branch->GetName() << " is already split" << endl ;
      }
      continue ;
    }
    if (masterNodesToBeReplaced.find(branch->GetName())) {
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() << ") Branch node " << branch->GetName() << " is already replaced" << endl ;
      }
      continue ;
    }

    if (branch->dependsOn(masterNodesToBeSplit)) {
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() << ") Branch node " 
	     << branch->IsA()->GetName() << "::" << branch->GetName() << " cloned: depends on a split parameter" << endl ;
      }
      masterBranchesToBeCloned.add(*branch) ;
    } else if (branch->dependsOn(masterNodesToBeReplaced)) {
      if (verbose) {
	cout << "RooCustomizer::build(" << _masterPdf->GetName() << ") Branch node " 
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
  _cloneBranchList.addOwned(clonedMasterBranches) ;


  // Reconnect cloned branches to each other and to cloned nodess
  iter = clonedMasterBranches.createIterator() ;
  while(branch=(RooAbsArg*)iter->Next()) {
    branch->redirectServers(clonedMasterBranches,kFALSE,kTRUE) ;
    branch->redirectServers(clonedMasterNodes,kFALSE,kTRUE) ;
    branch->redirectServers(masterReplacementNodes,kFALSE,kTRUE) ;
  }
  delete iter ;  


  return cloneTopPdf?cloneTopPdf:_masterPdf ;
}



void RooCustomizer::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  os << "RooCustomizer for " << _masterPdf->GetName() << (_sterile?" (sterile)":"") << endl ;

  Int_t i, nsplit = _splitArgList.GetSize() ;
  if (nsplit>0) {
    os << "  Splitting rules:" << endl ;
    for (i=0 ; i<nsplit ; i++) {
      os << "   " << _splitArgList.At(i)->GetName() << " is split by " << _splitCatList.At(i)->GetName() << endl ;
    }
  }

  Int_t nrepl = _replaceArgList.GetSize() ;
  if (nrepl>0) {
    os << "  Replacement rules:" << endl ;
    for (i=0 ; i<nrepl ; i++) {
      os << "   " << _replaceSubList.At(i)->GetName() << " replaces " << _replaceArgList.At(i)->GetName() << endl ;
    }
  }
  
  return ;
}
