/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgSet.cc,v 1.31 2001/08/22 01:01:32 david Exp $
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
// RooArgSet is a container object that can hold multiple RooAbsArg objects.
// The container has set semantics which means that:
//  - Every object it contains must have a unique name returned by GetName().
//  - Contained objects are not ordered although the set can be traversed
//    using an iterator returned by createIterator(). The iterator does not
//    necessarily follow the object insertion order.
//  - Objects can be retrieved by name only, and not by index.

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include "TClass.h"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooTrace.hh"

ClassImp(RooArgSet)
  ;

RooArgSet::RooArgSet() :
  _name(), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
//   cout << "!!!!! RooArgSet default ctor called !!!!!" << endl ;
//   assert(0) ;
}

RooArgSet::RooArgSet(const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4);
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4); add(var5);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;add(var8) ;
}


RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const RooAbsArg& var9, const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7); add(var8); add(var9);
}


RooArgSet::RooArgSet(const RooArgSet& other, const char *name) :
  _name(name), _list(), _isCopy(kFALSE)
{
  RooTrace::create(this) ;
  if (!name) setName(other.GetName()) ;

  // Transfer contents (not owned)
  TIterator *iterator= other.createIterator();
  RooAbsArg *arg(0);
  while(arg= (RooAbsArg*)iterator->Next()) {
    add(*arg);
  }
  delete iterator;
}

RooArgSet::~RooArgSet() 
{
  // delete all variables in our list if we own them
  if(_isCopy){ 
    Delete();
  }
  RooTrace::destroy(this) ;
}

RooArgSet* RooArgSet::snapshot(Bool_t deepCopy) const
{
  // Take a snap shot: clone current list and recursively add
  // all its external dependents

  // First create empty list
  TString snapName("Snapshot of ") ;
  snapName.Append(GetName()) ;
  RooArgSet* snapshot = new RooArgSet(snapName.Data()) ;

  // Copy contents
  TIterator *iterator= createIterator();
  RooAbsArg *orig(0);
  while(0 != (orig= (RooAbsArg*)iterator->Next())) {
    RooAbsArg *copy= (RooAbsArg*)orig->Clone();
    snapshot->add(*copy);
  }
  delete iterator;

  TIterator* vIter = snapshot->createIterator() ;
  RooAbsArg* var ;

  // Add external dependents
  if (deepCopy) {
    // Recursively add clones of all servers
    while (var=(RooAbsArg*)vIter->Next()) {
      snapshot->addServerClonesToList(*var) ;
    }
  }

  // Redirect all server connections to internal list members
  vIter->Reset() ;
  while (var=(RooAbsArg*)vIter->Next()) {
    var->redirectServers(*snapshot,deepCopy) ;
  }
  delete vIter ;

  // Transfer ownership of contents to list
  snapshot->_isCopy = kTRUE ;
  return snapshot ;
}



void RooArgSet::addServerClonesToList(const RooAbsArg& var)
{
  // Add clones of servers of given argument to list
  TIterator* sIter = var.serverIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)sIter->Next()) {
    if (!find(server->GetName())) {
      RooAbsArg* serverClone = (RooAbsArg*)server->Clone() ;
      serverClone->setAttribute("SnapShot_ExtRefClone") ;
      add(*serverClone) ;      
      addServerClonesToList(*server) ;
    }
  }
  delete sIter ;
}

RooArgSet &RooArgSet::operator=(const RooArgSet& other) {
  // The assignment operator sets the value of any argument in our set
  // that also appears in the other set.

  RooAbsArg *elem, *theirs ;
  Int_t index(getSize());
  while(--index >= 0) {
    elem= (RooAbsArg*)_list.At(index);
    theirs= other.find(elem->GetName());
    if(!theirs) continue;

    theirs->syncCache() ;
    elem->copyCache(theirs) ;

  }
  return *this;
}

RooAbsArg *RooArgSet::addClone(const RooAbsArg& var, Bool_t silent) {
  // Add a clone of the specified argument to list. Returns a pointer to
  // the clone if successful, or else zero if a variable of the same name
  // is already in the list or the list does *not* own its variables (in
  // this case, try add() instead.)

  const char *name= var.GetName();
  // check that this *is* a copy of a list
  if(!_isCopy) {
    cout << "RooArgSet(" << _name << "): can only add clones to a copied list" << endl;
    return 0;
  }

  // is this variable name already in this list?
  RooAbsArg *other(0);
  if(other= find(name)) {
    if(other != &var) {
      if (!silent)
	// print a warning if this variable is not the same one we
	// already have
	cout << "RooArgSet(" << _name << "): cannot add clone of second variable \"" << name
	     << "\"" << endl;
    }
    // don't add duplicates
    return 0;
  }
  // add a pointer to a clone of this variable to our list (we now own it!)
  RooAbsArg *clone= (RooAbsArg*)var.Clone();
  if(0 != clone) add(*clone);

  return clone;
}

Bool_t RooArgSet::add(const RooAbsArg& var, Bool_t silent) {
  // Add the specified argument to list. Returns kTRUE if successful, or
  // else kFALSE if a variable of the same name is already in the list
  // or the list owns its variables (in this case, try addClone() instead).

  const char *name= var.GetName();
  // check that this isn't a copy of a list
  if(_isCopy) {
    cout << "RooArgSet(" << _name << "): cannot add variables to a copied list" << endl;
    return kFALSE;
  }

  // is this variable name already in this list?
  RooAbsArg *other(0);
  if(other= find(name)) {
    if(other != &var) {
      if (!silent)
	// print a warning if this variable is not the same one we
	// already have
	cout << "RooArgSet(" << _name << "): cannot add second variable \"" << name
	     << "\"" << endl;
    }
    // don't add duplicates
    return kFALSE;
  }
  // add a pointer to this variable to our list (we don't own it!)
  add(var);
  return kTRUE;
}




Bool_t RooArgSet::add(const RooArgSet& list)
{
  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= add((RooAbsArg&)*list._list.At(index)) ;
  }

  return result;  
}

Bool_t RooArgSet::replace(const RooArgSet &other) {
  // Replace any args in our set with args of the same name from the other set
  // and return kTRUE for success. Fails if this list is a copy of another.

  // check that this isn't a copy of a list
  if(_isCopy) {
    cout << "RooArgSet: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }
  // loop over elements in the other list
  TIterator *otherArgs= other.createIterator();
  const RooAbsArg *arg(0);
  while(arg= (const RooAbsArg*)otherArgs->Next()) {
    // do we have an arg of the same name in our set?
    RooAbsArg *found= find(arg->GetName());
    if(found) replace(*found,*arg);
  }
  delete otherArgs;
  return kTRUE;
}

Bool_t RooArgSet::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  // Replace var1 with var2 and return kTRUE for success. Fails if
  // this list is a copy of another, if var1 is not already in this set,
  // or if var2 is already in this set. var1 and var2 do not need to have
  // the same name.

  // check that this isn't a copy of a list
  if(_isCopy) {
    cout << "RooArgSet: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }
  // is var1 already in this list?
  const char *name= var1.GetName();
  RooAbsArg *other= find(name);
  if(other != &var1) {
    cout << "RooArgSet: variable \"" << name << "\" is not in the list"
	 << " and cannot be replaced" << endl;
    return kFALSE;
  }
  // is var2's name already in this list?
  other= find(var2.GetName());
  if(other != 0 && other != &var1) {
    cout << "RooArgSet: cannot replace \"" << name
	 << "\" with already existing \"" << var2.GetName() << "\"" << endl;
    return kFALSE;
  }
  // replace var1 with var2
  _list.AddBefore((TObject*)&var1,(TObject*)&var2);
  _list.Remove((TObject*)&var1);
  return kTRUE;
}



Bool_t RooArgSet::remove(const RooAbsArg& var, Bool_t silent) {
  // Remove the specified argument from our list. Return kFALSE if
  // the specified argument is not found in our list. An exact pointer
  // match is required, not just a match by name. A variable can be
  // removed from a copied list and will be deleted at the same time.

  // is var already in this list?
  const char *name= var.GetName();
  RooAbsArg *found= find(name);
  if(found != &var) {    
    if (!silent) cout << "RooArgSet: variable \"" << name << "\" is not in the list"
		      << " and cannot be removed" << endl;
    return kFALSE;
  }
  _list.Remove(found);
  if(_isCopy) delete found;

  return kTRUE;
}

Bool_t RooArgSet::remove(const RooArgSet& list, Bool_t silent) {
  // Remove each argument in the input list from our list using remove(const RooAbsArg&).
  // Return kFALSE in case of problems.

  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= remove((RooAbsArg&)*list._list.At(index),silent) ;
  }

  return result;
}

void RooArgSet::removeAll() {
  // Remove all arguments from our set, deleting them if we own them.
  // This effectively restores our object to the state it would have
  // just after calling the RooArgSet(const char*) constructor.

  if(_isCopy) {
    Delete();
    _isCopy= kFALSE;
  }
  else {
    Clear();
  }
}

void RooArgSet::setAttribAll(const Text_t* name, Bool_t value) 
{
  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    arg->setAttribute(name,value) ;
  }
  delete iter ;
}


RooArgSet* RooArgSet::selectByAttrib(const char* name, Bool_t value) const
{
  // Create output set
  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooArgSet *sel = new RooArgSet(selName.Data()) ;
  
  // Scan set contents for matching attribute
  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    if (arg->getAttribute(name)==value)
      sel->add(*arg) ;
  }
  delete iter ;

  return sel ;
}



RooAbsArg *RooArgSet::find(const char *name) const {
  // Find object with given name in list
  return (RooAbsArg*)FindObject(name);
}


Bool_t RooArgSet::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  if (compact) {
    
    TIterator *iterator= createIterator();
    RooStreamParser parser(is) ;
    RooAbsArg *next(0);
    while(0 != (next= (RooAbsArg*)iterator->Next())) {
      if (!next->getAttribute("Dynamic")) {
	if (next->readFromStream(is,kTRUE,verbose)) {
	  parser.zapToEnd() ;

	  delete iterator ;
	  return kTRUE ;
	}	
      } else {
      }
    }

    if (!parser.atEOL()) {
      TString rest = parser.readLine() ;
      cout << "RooArgSet::readFromStream(" << GetName() 
	   << "): ignoring extra characters at end of line: '" << rest << "'" << endl ;
    }

    delete iterator;    
    return kFALSE ;

  } else {

    RooStreamParser parser(is) ;
    TString token ;
    Bool_t retVal(kFALSE) ;

    // Conditional stack and related state variables
    Bool_t anyCondTrue[100] ;
    Bool_t condStack[100] ;
    Bool_t lastLineWasElse=kFALSE ;
    Int_t condStackLevel=0 ;
    condStack[0]=kTRUE ;

    while (1) {
      // Read next token until end of file
      token = parser.readToken() ;
      if (is.eof() || is.bad()) break ;

      // Skip empty lines 
      if (token.IsNull()) continue ;

      // Process include directives
      if (!token.CompareTo("include")) {
	if (parser.atEOL()) {
	  cout << "RooArgSet::readFromStream(" << GetName() 
	       << "): no filename found after include statement" << endl ;
	  return kTRUE ;
	}
	TString filename = parser.readLine() ;
	ifstream incfs(filename) ;
	if (!incfs.good()) {
	  cout << "RooArgSet::readFromStream(" << GetName() << "): cannot open include file " << filename << endl ;
	  return kTRUE ;
	}
	cout << "RooArgSet::readFromStream(" << GetName() << "): processing include file " 
	     << filename << endl ;
	if (readFromStream(incfs,compact,verbose)) return kTRUE ;
	continue ;
      }

      // Conditional statement evaluation
      if (!token.CompareTo("if")) {

	// Extract conditional expressions and check validity
	TString expr = parser.readLine() ;
	RooFormula form(expr,expr,*this) ;
	if (!form.ok()) return kTRUE ;

	// Evaluate expression
	Bool_t status = form.eval()?kTRUE:kFALSE ;
	if (lastLineWasElse) {
	  anyCondTrue[condStackLevel] |= status ;
	  lastLineWasElse=kFALSE ;
	} else {
	  condStackLevel++ ;
	  anyCondTrue[condStackLevel] = status ;
	}
	condStack[condStackLevel] = status ;

	if (verbose) cout << "RooArgSet::readFromStream(" << GetName() 
			  << "): conditional expression " << expr << " = " 
			  << (condStack[condStackLevel]?"true":"false") << endl ;
	continue ; // go to next line
      }

      if (!token.CompareTo("else")) {
	// Must have seen an if statement before
	if (condStackLevel==0) {
	  cout << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'else'" << endl ;
	}

	if (parser.atEOL()) {
	  // simple else: process if nothing else was true
	  condStack[condStackLevel] = !anyCondTrue[condStackLevel] ; 
	  parser.zapToEnd() ;
	  continue ;
	} else {
	  // if anything follows it should be 'if'
	  token = parser.readToken() ;
	  if (token.CompareTo("if")) {
	    cout << "RooArgSet::readFromStream(" << GetName() << "): syntax error: 'else " << token << "'" << endl ;
	    return kTRUE ;
	  } else {
	    if (anyCondTrue[condStackLevel]) {
	      // No need for further checking, true conditional already processed
	      condStack[condStackLevel] = kFALSE ;
	      parser.zapToEnd() ;
	      continue ;
	    } else {
	      // Process as normal 'if' no true conditional was encountered 
	      parser.putBackToken(token) ;
	      lastLineWasElse=kTRUE ;
	      continue ;
	    }
	  }
	}	
      }

      if (!token.CompareTo("endif")) {
	// Must have seen an if statement before
	if (condStackLevel==0) {
	  cout << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'endif'" << endl ;
	  return kTRUE ;
	}

	// Decrease stack by one
	condStackLevel-- ;
	continue ;
      } 

      // If current conditional is true
      if (condStack[condStackLevel]) {

	// Process echo statements
	if (!token.CompareTo("echo")) {
	  TString message = parser.readLine() ;
	  cout << "RooArgSet::readFromStream(" << GetName() << "): >> " << message << endl ;
	  continue ;
	} 
	
	// Process abort statements
	if (!token.CompareTo("abort")) {
	  TString message = parser.readLine() ;
	  cout << "RooArgSet::readFromStream(" << GetName() << "): USER ABORT" << endl ;
	  return kTRUE ;
	} 
	
	// Interpret the rest as <arg> = <value_expr> 
	RooAbsArg *arg ;
	if ((arg = find(token)) && !arg->getAttribute("Dynamic")) {
	  if (parser.expectToken("=",kTRUE)) {
	    parser.zapToEnd() ;
	    retVal=kTRUE ;
	    cout << "RooArgSet::readFromStream(" << GetName() 
		 << "): missing '=' sign: " << arg << endl ;
	    continue ;
	  }
	  retVal |= arg->readFromStream(is,kFALSE,verbose) ;	
	} else {
	  cout << "RooArgSet::readFromStream(" << GetName() << "): argument " 
	       << token << " not in list, ignored" << endl ;
	  parser.zapToEnd() ;
	}
      } else {
	parser.readLine() ;
      }
    }

    // Did we fully unwind the conditional stack?
    if (condStackLevel!=0) {
      cout << "RooArgSet::readFromStream(" << GetName() << "): missing 'endif'" << endl ;
      return kTRUE ;
    }

    return retVal ;
  }
}



void RooArgSet::writeToStream(ostream& os, Bool_t compact) 
{
  TIterator *iterator= createIterator();
  RooAbsArg *next(0);
  while(0 != (next= (RooAbsArg*)iterator->Next())) {
    if (compact) {
      next->writeToStream(os,kTRUE) ;
      os << " " ;
    } else  {
      os << next->GetName() << " = " ;
      next->writeToStream(os,kFALSE) ;
      os << endl ;
    }
  }
  delete iterator;  
  if (compact) os << endl ;
}

void RooArgSet::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this argument set to the specified stream.
  //
  //  Standard: OneLine description of each argument
  //     Shape: Standard description of each argument
  //   Verbose: Shape description of each argument

  // we cannot use oneLinePrint() since we do not inherit from TNamed
  os << ClassName() << "::" << GetName() << ":" << (_isCopy?" COPY":"") << endl;
  if(opt >= Standard) {
    TIterator *iterator= createIterator();
    int index= 0;
    RooAbsArg *next(0);
    opt= lessVerbose(opt);
    TString deeper(indent);
    deeper.Append("     ");
    while(0 != (next= (RooAbsArg*)iterator->Next())) {
      os << indent << setw(3) << ++index << ") ";
      next->printToStream(os,opt,deeper);
    }
    delete iterator;
  }
}
