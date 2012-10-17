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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooDirItem is a utility base class for RooFit objects that are to be attached
// to ROOT directories. Concrete classes inherit the appendToDir and removeToDir
// methods that can be used to safely attach and detach one self from a TDirectory
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "TROOT.h"
#include "TList.h"
#include "TDirectoryFile.h"
#include "TString.h"
#include "RooDirItem.h"

ClassImp(RooDirItem) ;


//_____________________________________________________________________________
RooDirItem::RooDirItem() : _dir(0) 
{
  // Default constructor
}


//_____________________________________________________________________________
RooDirItem::RooDirItem(const RooDirItem& /*other*/) : _dir(0) 
{
  // Copy constructor
}



//_____________________________________________________________________________
RooDirItem::~RooDirItem() 
{  
  // Destructor
}



//_____________________________________________________________________________
void RooDirItem::removeFromDir(TObject* obj) 
{
  // Remove object from directory it was added to

  if (_dir) {
    if (!_dir->TestBit(TDirectoryFile::kCloseDirectory))
      _dir->GetList()->Remove(obj) ;
  }
} 



//_____________________________________________________________________________
void RooDirItem::appendToDir(TObject* obj, Bool_t forceMemoryResident) 
{
  // Append object to directory. If forceMemoryResident is
  // true, force addition to ROOT memory directory if that
  // is not the current directory

  if (forceMemoryResident) {
    // Append self forcibly to memory directory

    TString pwd(gDirectory->GetPath()) ;
    TString memDir(gROOT->GetName()) ;
    memDir.Append(":/") ;
    Bool_t notInMemNow= (pwd!=memDir) ;

    //cout << "RooDirItem::appendToDir pwd=" << pwd << " memDir=" << memDir << endl ;

    if (notInMemNow) { 
      gDirectory->cd(memDir) ;
    }

    _dir = gDirectory ;
    gDirectory->Append(obj) ;
    
    if (notInMemNow) {
      gDirectory->cd(pwd) ;    
    }

  } else {
    // Append self to present gDirectory
    _dir = gDirectory ;
    gDirectory->Append(obj) ;
  }
}

