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
#ifndef ROO_DIR_ITEM
#define ROO_DIR_ITEM

#include "Rtypes.h"
class TDirectory ;

class RooDirItem {
public:
  RooDirItem() ;
  RooDirItem(const RooDirItem& other) ;
  virtual ~RooDirItem() ;

protected:

  void appendToDir(TObject* obj, Bool_t forceMemoryResident=kFALSE) ;
  void removeFromDir(TObject* obj) ;

  TDirectory* _dir ;     //! Do not persist
  ClassDef(RooDirItem,1) // Base class for RooFit objects that are listed TDirectories
};

#endif
