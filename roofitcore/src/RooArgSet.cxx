/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgSet.cc,v 1.36 2001/09/06 20:49:15 verkerke Exp $
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
#include "RooFitCore/RooArgList.hh"

ClassImp(RooArgSet)
  ;

RooArgSet::RooArgSet() :
  RooAbsCollection()
{
}

RooArgSet::RooArgSet(const RooArgList& list) :
  RooAbsCollection(list.GetName())
{
  add(list,kTRUE) ; // verbose to catch duplicate errors
}


RooArgSet::RooArgSet(const char *name) :
  RooAbsCollection(name)
{
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4);
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;add(var8) ;
}


RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const RooAbsArg& var9, const char *name) :
  RooAbsCollection(name)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7); add(var8); add(var9);
}



RooArgSet::RooArgSet(const RooArgSet& other, const char *name) 
  : RooAbsCollection(other,name)
{
}



RooArgSet::~RooArgSet() 
{
}



Bool_t RooArgSet::add(const RooAbsArg& var, Bool_t silent) 
{
  return checkForDup(var,silent)? kFALSE : RooAbsCollection::add(var,silent) ;
}


Bool_t RooArgSet::addOwned(RooAbsArg& var, Bool_t silent)
{
  return checkForDup(var,silent)? kFALSE : RooAbsCollection::addOwned(var,silent) ;
}


RooAbsArg* RooArgSet::addClone(const RooAbsArg& var, Bool_t silent) 
{
  return checkForDup(var,silent)? 0 : RooAbsCollection::addClone(var,silent) ;
}



Bool_t RooArgSet::checkForDup(const RooAbsArg& var, Bool_t silent) const 
{
  // is this variable name already in this list?
  RooAbsArg *other(0);
  if(other= find(var.GetName())) {
    if(other != &var) {
      if (!silent)
	// print a warning if this variable is not the same one we
	// already have
	cout << ClassName() << "::" << GetName() << "::addClone: cannot add second copy of argument \""
	     << var.GetName() << "\"" << endl;
    }
    // don't add duplicates
    return kTRUE;
  }

  return kFALSE ;
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



