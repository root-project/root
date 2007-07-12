/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id$
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

// -- CLASS DESCRIPTION [PLOT] --

#include "RooFit.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooAbsData.h"
// #include "RooModelView.h"

#include "TClass.h"
#include "Riostream.h"
#include <string.h>
#include <assert.h>

ClassImp(RooWorkspace)
;

RooWorkspace::RooWorkspace(const char* name, const char* title) : TNamed(name,title?title:name)
{
}


RooWorkspace::RooWorkspace(const RooWorkspace& other) : TNamed(other)
{
}


RooWorkspace::~RooWorkspace() 
{
}


Bool_t RooWorkspace::import(const RooAbsArg& arg) 
{
  // Scan for overlaps with current contents
  if (_allOwnedNodes.find(arg.GetName())) {
    cout << "RooWorkSpace::import(" << GetName() << ") ERROR importing object named " << arg.GetName() << ": already in the workspace" << endl ;
    return kTRUE ;    
  }
  
  // Scan for overlaps of lower nodes with current contents
  RooArgSet branchSet ;
  arg.branchNodeServerList(&branchSet) ;
  TIterator* iter = branchSet.createIterator() ;
  RooAbsArg* branch ;
  while ((branch=(RooAbsArg*)iter->Next())) {
    if (_allOwnedNodes.find(branch->GetName())) {
      cout << "RooWorkSpace::import(" << GetName() << ") ERROR object named " << arg.GetName() << ": component " 
	   << branch->GetName() << " already in the workspace" << endl ;      
      return kTRUE ;
    }
  }
  delete iter ;

  // Print a message for each imported node
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(arg).snapshot(kTRUE) ;
  iter = cloneSet->createIterator() ;  
  RooAbsArg* node ;
  RooArgSet recycledNodes ;
  while((node=(RooAbsArg*)iter->Next())) {

    // Check if node is already in workspace (can only happen for variables)
    if (_allOwnedNodes.find(node->GetName())) {
      // Do not import node, add not to list of nodes that require reconnection
      cout << "RooWorkspace::import(" << GetName() << ") using existing copy of variable " << node->IsA()->GetName() 
	   << "::" << node->GetName() << " for import of " << arg.IsA()->GetName() << "::" << arg.GetName() << endl ;      
      recycledNodes.add(*_allOwnedNodes.find(node->GetName())) ;

    } else {
      // Import node
      cout << "RooWorkspace::import(" << GetName() << ") importing " << node->IsA()->GetName() << "::" << node->GetName() << endl ;
      _allOwnedNodes.addOwned(*node) ;
    }
  }


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


Bool_t RooWorkspace::import(RooAbsData& data) 
{
  _dataList.Add(&data) ;
  return kFALSE ;
}


Bool_t RooWorkspace::merge(const RooWorkspace& /*other*/) 
{
  return kFALSE ;
}


Bool_t RooWorkspace::join(const RooWorkspace& /*other*/) 
{
  return kFALSE ;
}

RooAbsPdf* RooWorkspace::pdf(const char* name) 
{ 
  return dynamic_cast<RooAbsPdf*>(_allOwnedNodes.find(name)) ; 
}

RooAbsReal* RooWorkspace::function(const char* name) 
{ 
  return dynamic_cast<RooAbsReal*>(_allOwnedNodes.find(name)) ; 
}

RooRealVar* RooWorkspace::var(const char* name) 
{ 
  return dynamic_cast<RooRealVar*>(_allOwnedNodes.find(name)) ; 
}

RooCategory* RooWorkspace::cat(const char* name) 
{ 
  return dynamic_cast<RooCategory*>(_allOwnedNodes.find(name)) ; 
}

RooAbsData* RooWorkspace::data(const char* name) 
{
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
    iter = pdfSet.createIterator() ;
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

