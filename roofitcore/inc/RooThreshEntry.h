/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_THRESH_ENTRY
#define ROO_THRESH_ENTRY

#include <iostream.h>
#include "TNamed.h"
#include "TRegexp.h"
class RooCatType ;

class RooThreshEntry : public TObject {
public:
  inline RooThreshEntry() : TObject(), _thresh(0), _cat(0) {} 
  RooThreshEntry(Double_t thresh, const RooCatType* cat) ;
  RooThreshEntry(const RooThreshEntry& other) ;
  virtual TObject* Clone(const char*) const { return new RooThreshEntry(*this); }

  virtual Int_t Compare(const TObject *) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

  inline Double_t thresh() const { return _thresh ; }
  inline const RooCatType* cat() const { return _cat ; }

protected:

  Double_t _thresh ;
  RooCatType* _cat ;
	
  ClassDef(RooThreshEntry,1) // Utility class, holding a threshold/category state pair
} ;


#endif
