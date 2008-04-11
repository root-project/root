/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// The RooWorkspace is a persistable container for RooFit projects. A workspace
// can contain and own variables, p.d.f.s, functions and datasets. All objects
// that live in the workspace are owned by the workspace. The import() method
// enforces consistency of objects upon insertion into the workspace (e.g. no
// duplicate object with the same name are allowed) and makes sure all objects
// in the workspace are connected to each other. Easy accessor methods like
// pdf(), var() and data() allow to refer to the contents of the workspace by
// object name. The entire RooWorkspace can be saved into a ROOT TFile and organises
// the consistent streaming of its contents without duplication.

#include "RooFit.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooAbsData.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"
// #include "RooModelView.h"
#include <map>
#include <string>
#include <list>
using namespace std ;

#include "TClass.h"
#include "Riostream.h"
#include <string.h>
#include <assert.h>

ClassImp(RooWorkspace)
;

RooWorkspace::RooWorkspace(const char* name, const char* title) : TNamed(name,title?title:name)
// Empty workspace constructor
{
}


RooWorkspace::RooWorkspace(const RooWorkspace& other) : TNamed(other)
{
  // Workspace copy constructor
  other._allOwnedNodes.snapshot(_allOwnedNodes,kTRUE) ;
  TIterator* iter = other._dataList.MakeIterator() ;
  TObject* data ;
  while((data=iter->Next())) {
    _dataList.Add(data->Clone()) ;
  }
  delete iter ;
}


RooWorkspace::~RooWorkspace() 
{
  // Workspace destructor
  _dataList.Delete() ;
}


Bool_t RooWorkspace::import(const RooAbsArg& arg, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3) 
{
  //  Import a RooAbsArg object, e.g. function, p.d.f or variable into the workspace. This import function clones the input argument and will
  //  own the clone. If a composite object is offered for import, e.g. a p.d.f with parameters and observables, the
  //  complete tree of objects is imported. If any of the _variables_ of a composite object (parameters/observables) are already 
  //  in the workspace the imported p.d.f. is connected to the already existing variables. If any of the _function_ objects (p.d.f, formulas) 
  //  to be imported already exists in the workspace an error message is printed and the import of the entire tree of objects is cancelled. 
  //  Several optional arguments can be provided to modify the import procedure.
  //
  //  Accepted arguments
  //  -------------------------------
  //  RenameConflictNodes(const char* suffix) -- Add suffix to branch node name if name conflicts with existing node in workspace
  //  RenameNodes(const char* suffix) -- Add suffix to all branch node names including top level node
  //  RenameVariable(const char* inputName, const char* outputName) -- Rename variable as specified upon import.
  //
  //  The RenameConflictNodes and RenameNodes arguments are mutually exclusive. The RenameVariable argument can be repeated
  //  as often as necessary to rename multiple variables. Alternatively, a single RenameVariable argument can be given with
  //  two comma separated lists.

  RooLinkedList args ;
  args.Add((TObject*)&arg1) ;
  args.Add((TObject*)&arg2) ;
  args.Add((TObject*)&arg3) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooWorkspace::import(%s)",GetName())) ;

  pc.defineString("conflictSuffix","RenameConflictNodes",0) ;
  pc.defineString("allSuffix","RenameAllNodes",0) ;
  pc.defineString("varChangeIn","RenameVar",0,"",kTRUE) ;
  pc.defineString("varChangeOut","RenameVar",1,"",kTRUE) ;
  pc.defineMutex("RenameConflictNodes","RenameAllNodes") ;

  // Process and check varargs 
  pc.process(args) ;
  if (!pc.ok(kTRUE)) {
    return kTRUE ;
  }

  // Decode renaming logic into suffix string and boolean for conflictOnly mode
  const char* suffixC = pc.getString("conflictSuffix") ;
  const char* suffixA = pc.getString("allSuffix") ;
  const char* varChangeIn = pc.getString("varChangeIn") ;
  const char* varChangeOut = pc.getString("varChangeOut") ;

  // Turn zero length strings into null pointers 
  if (suffixC && strlen(suffixC)==0) suffixC = 0 ;
  if (suffixA && strlen(suffixA)==0) suffixA = 0 ;

  Bool_t conflictOnly = suffixA ? kFALSE : kTRUE ;
  const char* suffix = suffixA ? suffixA : suffixC ;
  
  // Scan for overlaps with current contents
  if (!suffix && _allOwnedNodes.find(arg.GetName())) {
    coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR importing object named " << arg.GetName() 
			 << ": already in the workspace and no conflict resolution protocol specified" << endl ;
    return kTRUE ;    
  }

  // Make list of conflicting nodes
  RooArgSet conflictNodes ;
  RooArgSet branchSet ;
  arg.branchNodeServerList(&branchSet) ;
  TIterator* iter = branchSet.createIterator() ;
  RooAbsArg* branch ;
  while ((branch=(RooAbsArg*)iter->Next())) {
    if (_allOwnedNodes.find(branch->GetName())) {
      conflictNodes.add(*branch) ;
    }
  }
  delete iter ;
  
  // Terminate here if there are conflicts and no resolution protocol
  if (conflictNodes.getSize()>0 && !suffix) {
      coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR object named " << arg.GetName() << ": component(s) " 
	   << conflictNodes << " already in the workspace and no conflict resolution protocol specified" << endl ;      
      return kTRUE ;
  }
    
  // Now create a working copy of the incoming object tree
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(arg).snapshot(kTRUE) ;
  RooAbsArg* cloneTop = cloneSet->find(arg.GetName()) ;

  // Mark all nodes for renaming if we are not in conflictOnly mode
  if (!conflictOnly) {
    conflictNodes.removeAll() ;
    conflictNodes.add(branchSet) ;
  }

  // Mark nodes that are to be renamed with special attribute
  TIterator* citer = conflictNodes.createIterator() ;
  string topName2 = cloneTop->GetName() ;
  RooAbsArg* cnode ;
  while ((cnode=(RooAbsArg*)citer->Next())) {
    RooAbsArg* cnode2 = cloneSet->find(cnode->GetName()) ;
    string origName = cnode2->GetName() ;
    cnode2->SetName(Form("%s_%s",cnode2->GetName(),suffix)) ;
    cnode2->SetTitle(Form("%s (%s)",cnode2->GetTitle(),suffix)) ;
    string tag = Form("ORIGNAME:%s",origName.c_str()) ;
    cnode2->setAttribute(tag.c_str()) ;

    // Save name of new top level node for later use
    if (cnode2==cloneTop) {
      topName2 = cnode2->GetName() ;
    }

    coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() 
		       << ") Resolving name conflict in workspace by changing name of imported node  " 
		       << origName << " to " << cnode2->GetName() << endl ;
  }  
  delete citer ;

  // Process any change in variable names 
  if (strlen(varChangeIn)>0) {
    
    // Parse comma separated lists into map<string,string>
    char tmp[1024] ;
    strcpy(tmp,varChangeIn) ;
    list<string> tmpIn,tmpOut ;
    char* ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpIn.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }
    strcpy(tmp,varChangeOut) ;
    ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpOut.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }    
    map<string,string> varMap ;
    list<string>::iterator iin = tmpIn.begin() ;
    list<string>::iterator iout = tmpOut.begin() ;
    for (;iin!=tmpIn.end() ; ++iin,++iout) {
      varMap[*iin]=*iout ;
    }       
    
    // Process all changes in variable names
    TIterator* cliter = cloneSet->createIterator() ;
    while ((cnode=(RooAbsArg*)cliter->Next())) {
      
      if (varMap.find(cnode->GetName())!=varMap.end()) { 	
	string origName = cnode->GetName() ;
	cnode->SetName(varMap[cnode->GetName()].c_str()) ;
	string tag = Form("ORIGNAME:%s",origName.c_str()) ;
	cnode->setAttribute(tag.c_str()) ;
	coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") Changing name of variable " 
			   << origName << " to " << cnode->GetName() << " on request" << endl ;
      }    
    }
    delete cliter ;
  }
  
  // Now clone again with renaming effective
  RooArgSet* cloneSet2 = (RooArgSet*) RooArgSet(*cloneTop).snapshot(kTRUE) ;
  RooAbsArg* cloneTop2 = cloneSet2->find(topName2.c_str()) ;

  // Make final check list of conflicting nodes
  RooArgSet conflictNodes2 ;
  RooArgSet branchSet2 ;
  arg.branchNodeServerList(&branchSet) ;
  TIterator* iter2 = branchSet2.createIterator() ;
  RooAbsArg* branch2 ;
  while ((branch2=(RooAbsArg*)iter2->Next())) {
    if (_allOwnedNodes.find(branch2->GetName())) {
      conflictNodes2.add(*branch2) ;
    }
  }
  delete iter2 ;

  // Terminate here if there are conflicts and no resolution protocol
  if (conflictNodes2.getSize()) {
    coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR object named " << arg.GetName() << ": component(s) " 
			  << conflictNodes2 << " cause naming conflict after conflict resolution protocol was executed" << endl ;      
    return kTRUE ;
  }
    
  // Print a message for each imported node
  iter = cloneSet2->createIterator() ;
  RooAbsArg* node ;
  RooArgSet recycledNodes ;
  while((node=(RooAbsArg*)iter->Next())) {

    // Check if node is already in workspace (can only happen for variables)
    if (_allOwnedNodes.find(node->GetName())) {
      // Do not import node, add not to list of nodes that require reconnection
      coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") using existing copy of variable " << node->IsA()->GetName() 
			 << "::" << node->GetName() << " for import of " << cloneTop2->IsA()->GetName() << "::" 
			 << cloneTop2->GetName() << endl ;      
      recycledNodes.add(*_allOwnedNodes.find(node->GetName())) ;

    } else {
      // Import node
      coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") importing " << node->IsA()->GetName() << "::" 
			 << node->GetName() << endl ;
      _allOwnedNodes.addOwned(*node) ;
    }
  }

  // Release working copy
  delete cloneSet ;


  // Reconnect any nodes that need to be
  if (recycledNodes.getSize()>0) {
    iter->Reset() ;
    while((node=(RooAbsArg*)iter->Next())) {
      node->redirectServers(recycledNodes) ;
    }
  }

  delete iter ;
  

  return kFALSE ;
}


Bool_t RooWorkspace::import(RooAbsData& data, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3) 
{
  //  Import a dataset (RooDataSet or RooDataHist) into the work space. The workspace will contain a copy of the data
  //  The dataset and its variables can be renamed upon insertion with the options below
  //
  //  Accepted arguments
  //  -------------------------------
  //  RenameDataset(const char* suffix) -- Rename dataset upon insertion
  //  RenameVariable(const char* inputName, const char* outputName) -- Change names of observables in dataset upon insertion

  coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") importing dataset " << data.GetName() << endl ;

  RooLinkedList args ;
  args.Add((TObject*)&arg1) ;
  args.Add((TObject*)&arg2) ;
  args.Add((TObject*)&arg3) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooWorkspace::import(%s)",GetName())) ;

  pc.defineString("dsetName","RenameDataset",0,"") ;
  pc.defineString("varChangeIn","RenameVar",0,"",kTRUE) ;
  pc.defineString("varChangeOut","RenameVar",1,"",kTRUE) ;

  // Process and check varargs 
  pc.process(args) ;
  if (!pc.ok(kTRUE)) {
    return kTRUE ;
  }

  // Decode renaming logic into suffix string and boolean for conflictOnly mode
  const char* dsetName = pc.getString("dsetName") ;
  const char* varChangeIn = pc.getString("varChangeIn") ;
  const char* varChangeOut = pc.getString("varChangeOut") ;

  // Transform emtpy string into null pointer
  if (dsetName && strlen(dsetName)==0) {
    dsetName=0 ;
  }

  // Rename dataset if required
  RooAbsData* clone ;
  if (dsetName) {
    coutI(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") changing name of dataset from  " << data.GetName() << " to " << dsetName << endl ;
    clone = (RooAbsData*) data.Clone(dsetName) ;
  } else {
    clone = (RooAbsData*) data.Clone(data.GetName()) ;
  }


  // Process any change in variable names 
  if (strlen(varChangeIn)>0) {
    
    // Parse comma separated lists of variable name changes
    char tmp[1024] ;
    strcpy(tmp,varChangeIn) ;
    list<string> tmpIn,tmpOut ;
    char* ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpIn.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }
    strcpy(tmp,varChangeOut) ;
    ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpOut.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }    
    list<string>::iterator iin = tmpIn.begin() ;
    list<string>::iterator iout = tmpOut.begin() ;

    for (; iin!=tmpIn.end() ; ++iin,++iout) {
      coutI(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") changing name of dataset observable " << *iin << " to " << *iout << endl ;
      clone->changeObservableName(iin->c_str(),iout->c_str()) ;
    }
  }

  // Now import the dataset observables that are not already imported
  TIterator* iter = clone->get()->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!_allOwnedNodes.find(arg->GetName())) {
      import(*arg) ;
    }
  }
  delete iter ;
    
  _dataList.Add(clone) ;
  return kFALSE ;
}


Bool_t RooWorkspace::merge(const RooWorkspace& /*other*/) 
{
  // Stub for merge function with another workspace (not implemented yet)
  return kFALSE ;
}


Bool_t RooWorkspace::join(const RooWorkspace& /*other*/) 
{
  // Stub for join function with another workspace (not implemented yet)
  return kFALSE ;
}

RooAbsPdf* RooWorkspace::pdf(const char* name) 
{ 
  // Retrieve p.d.f (RooAbsPdf) with given name. A null pointer is returned if not found
  return dynamic_cast<RooAbsPdf*>(_allOwnedNodes.find(name)) ; 
}

RooAbsReal* RooWorkspace::function(const char* name) 
{ 
  // Retrieve function (RooAbsReal) with given name. Note that all RooAbsPdfs are also RooAbsReals. A null pointer is returned if not found.
  return dynamic_cast<RooAbsReal*>(_allOwnedNodes.find(name)) ; 
}

RooRealVar* RooWorkspace::var(const char* name) 
{ 
  // Retrieve real-valued variable (RooRealVar) with given name. A null pointer is returned if not found
  return dynamic_cast<RooRealVar*>(_allOwnedNodes.find(name)) ; 
}

RooCategory* RooWorkspace::cat(const char* name) 
{ 
  // Retrieve discrete variable (RooCategory) with given name. A null pointer is returned if not found
  return dynamic_cast<RooCategory*>(_allOwnedNodes.find(name)) ; 
}

RooAbsData* RooWorkspace::data(const char* name) 
{
  // Retrieve dataset (binned or unbinned) with given name. A null pointer is returned if not found
  return (RooAbsData*)_dataList.FindObject(name) ;
}


// RooModelView* RooWorkspace::addView(const char* name, const RooArgSet& observables) 
// {
//   RooModelView* newView = new RooModelView(*this,observables,name,name) ;
//   _views.Add(newView) ;
//   return newView ;
// }


// RooModelView* RooWorkspace::view(const char* name) 
// {
//   return (RooModelView*) _views.FindObject(name) ;
// }


// void RooWorkspace::removeView(const char* /*name*/) 
// {
// }


void RooWorkspace::Print(Option_t* /*opts*/) const 
{
  // Print contents of the workspace 
  cout << endl << "RooWorkspace(" << GetName() << ") " << GetTitle() << " contents" << endl << endl  ;

  RooAbsArg* arg ;

  RooArgSet pdfSet ;
  RooArgSet funcSet ;
  RooArgSet varSet ;

  // Split list of components in pdfs, functions and variables
  TIterator* iter = _allOwnedNodes.createIterator() ;
  while((arg=(RooAbsArg*)iter->Next())) {

    if (arg->IsA()->InheritsFrom(RooAbsPdf::Class())) {
      pdfSet.add(*arg) ;
    }

    if (arg->IsA()->InheritsFrom(RooAbsReal::Class()) && 
	!arg->IsA()->InheritsFrom(RooAbsPdf::Class()) && 
	!arg->IsA()->InheritsFrom(RooRealVar::Class())) {
      funcSet.add(*arg) ;
    }

    if (arg->IsA()->InheritsFrom(RooRealVar::Class())) {
      varSet.add(*arg) ;
    }
    if (arg->IsA()->InheritsFrom(RooCategory::Class())) {
      varSet.add(*arg) ;
    }

  }
  delete iter ;


  if (varSet.getSize()>0) {
    cout << "variables" << endl ;
    cout << "---------" << endl ;
    cout << varSet << endl ;
    cout << endl ;
  }

  if (pdfSet.getSize()>0) {
    cout << "p.d.f.s" << endl ;
    cout << "-------" << endl ;
    iter = pdfSet.createIterator() ;
    while((arg=(RooAbsArg*)iter->Next())) {
      arg->Print() ;
    }
    delete iter ;
    cout << endl ;
  }

  if (funcSet.getSize()>0) {
    cout << "functions" << endl ;
    cout << "--------" << endl ;
    iter = funcSet.createIterator() ;
    while((arg=(RooAbsArg*)iter->Next())) {
      arg->Print() ;
    }
    delete iter ;
    cout << endl ;
  }


  if (_dataList.GetSize()>0) {
    cout << "datasets" << endl ;
    cout << "--------" << endl ;
    iter = _dataList.MakeIterator() ;
    RooAbsData* data ;
    while((data=(RooAbsData*)iter->Next())) {
      cout << data->IsA()->GetName() << "::" << data->GetName() << *data->get() << endl ;
    }
    delete iter ;
    cout << endl ;
  }


//   if (_views.GetSize()>0) {
//     cout << "views" << endl ;
//     cout << "-----" << endl ;
//     iter = _views.MakeIterator() ;
//     RooModelView* view ;
//     while((view=(RooModelView*)iter->Next())) {
//       view->Print() ;
//     }
//     delete iter ;
//   }

  return ;
}

