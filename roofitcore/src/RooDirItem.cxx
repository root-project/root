/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDirItem.cc,v 1.11 2005/06/20 15:44:51 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooDirItem is a utility base class for RooFit objects that are to be attached
// to ROOT directories. Concrete classes inherit the appendToDir and removeToDir
// methods that can be used to safely attach and detach one self from a TDirectory

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "TROOT.h"
#include "TList.h"
#include "TDirectoryFile.h"
#include "TString.h"
#include "RooDirItem.h"

ClassImp(RooDirItem) ;

RooDirItem::RooDirItem() : _dir(0) 
{
  // Default constructor
}

RooDirItem::RooDirItem(const RooDirItem& /*other*/) : _dir(0) 
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
    if (!_dir->TestBit(TDirectoryFile::kCloseDirectory))
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

