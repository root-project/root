/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   21-Nov-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TROOT.h"
#include "TList.h"
#include "TDirectory.h"
#include "TString.h"
#include "RooFitCore/RooDirItem.hh"

ClassImp(RooDirItem) ;

RooDirItem::RooDirItem() : _dir(0) 
{
  // Default constructor
}

RooDirItem::RooDirItem(const RooDirItem& other) : _dir(0) 
{
  // Copy constructor
}


RooDirItem::~RooDirItem() 
{  
  // Destructor
}


void RooDirItem::removeFromDir(TObject* obj) 
{
  // Remove self from directory it was added to
  if (_dir) {
    if (!_dir->TestBit(TDirectory::kCloseDirectory))
      _dir->GetList()->Remove(obj) ;
  }
} 


void RooDirItem::appendToDir(TObject* obj, Bool_t forceMemoryResident) 
{
  if (forceMemoryResident) {
    // Append self forcibly to memory directory
    TString pwd(gDirectory->GetPath()) ;
    TString memDir(gROOT->GetName()) ;
    memDir.Append(":/") ;
    gDirectory->cd(memDir) ;

    _dir = gDirectory ;
    gDirectory->Append(obj) ;
    
    gDirectory->cd(pwd) ;    
  } else {
    // Append self to present gDirectory
    _dir = gDirectory ;
    gDirectory->Append(obj) ;
  }
}

