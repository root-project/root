/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgList.cc,v 1.1 2001/09/17 18:48:12 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   23-Aug-2001 DK Enforce set semantics in the public interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooArgList is a container object that can hold multiple RooAbsArg objects.
// The container has set semantics which means that:

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include "TClass.h"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooTrace.hh"

ClassImp(RooArgList)
  ;

RooArgList::RooArgList() :
  RooAbsCollection()
{
}

RooArgList::RooArgList(const RooArgSet& set) :
  RooAbsCollection(set.GetName())
{
  add(set) ;
}


RooArgList::RooArgList(const char *name) :
  RooAbsCollection(name)
{
}

RooArgList::RooArgList(const RooAbsArg& var1,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1);
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2);
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3);
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4);
}

RooArgList::RooArgList(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5);
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;
}

RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;add(var8) ;
}


RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const RooAbsArg& var9, const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7); add(var8); add(var9);
}



RooArgList::RooArgList(const TCollection& tcoll, const char* name) :
  RooAbsCollection(name)
{
  TIterator* iter = tcoll.MakeIterator() ;
  TObject* obj ;
  while(obj=iter->Next()) {
    if (!dynamic_cast<RooAbsArg*>(obj)) {
      cout << "RooArgList::RooArgList(TCollection) element " << obj->GetName() 
	   << " is not a RooAbsArg, ignored" << endl ;
      continue ;
    }
    add(*(RooAbsArg*)obj) ;
  }
  delete iter ;
}



RooArgList::RooArgList(const RooArgList& other, const char *name) 
  : RooAbsCollection(other,name)
{
}



RooArgList::~RooArgList() 
{
}

