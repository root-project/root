/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgSet.cc,v 1.17 2001/05/10 18:58:47 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include "TClass.h"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooFormula.hh"

ClassImp(RooArgSet)


RooArgSet::RooArgSet() :
  _name(), THashList(), _isCopy(kFALSE)
{
}

RooArgSet::RooArgSet(const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3); add(var4);
}

RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3); add(var4); add(var5);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;
}

RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const char *name) :
  _name(name), THashList(), _isCopy(kFALSE)
{
  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;add(var8) ;
}


RooArgSet::RooArgSet(const RooArgSet& other, const char *name) :
  _name(name), THashList(), _isCopy(kTRUE)
{
  if (!name) SetName(other.GetName()) ;

  TIterator *iterator= other.MakeIterator();
  RooAbsArg *orig(0);
  while(0 != (orig= (RooAbsArg*)iterator->Next())) {
    RooAbsArg *copy= (RooAbsArg*)orig->Clone();
    Add(copy);
  }
  delete iterator;

  iterator = MakeIterator() ;
  while (orig = (RooAbsArg*)iterator->Next()) {
    orig->redirectServers(*this) ;
  }
  delete iterator ;
}

RooArgSet::~RooArgSet() 
{
  // delete all variables in our list if we own them
  if(_isCopy) Delete();
}

RooArgSet* RooArgSet::snapshot() {
  // Take a snap shot: clone current list and recursively add
  // all its external dependents

  // First clone current list and contents
  RooArgSet* snapshot = new RooArgSet(*this,TString("Snapshot of ").Append(GetName())) ;

  // Add external dependents
  TIterator* vIter = snapshot->MakeIterator() ;
  RooAbsArg* var ;

  // Recursively add clones of all servers
  while (var=(RooAbsArg*)vIter->Next()) {
    snapshot->addServerClonesToList(*var) ;
  }

  // Redirect all server connections to internal list members
  vIter->Reset() ;
  while (var=(RooAbsArg*)vIter->Next()) {
    var->redirectServers(*snapshot,kTRUE) ;
  }
  delete vIter ;
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
      Add(serverClone) ;      
      addServerClonesToList(*server) ;
    }
  }
  delete sIter ;
}

RooArgSet &RooArgSet::operator=(const RooArgSet& other) {
  // Assignment operator
  RooAbsArg *elem, *theirs ;
  Int_t index(GetSize());
  while(--index >= 0) {
    elem= (RooAbsArg*)At(index);
    theirs= other.find(elem->GetName());
    if(!theirs) continue;
    elem->copyCache(theirs) ;
  }
  return *this;
}


Bool_t RooArgSet::add(const RooAbsArg& var, Bool_t silent) {
  // Add argument to list

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
  Add((TObject*)&var);
  return kTRUE;
}




Bool_t RooArgSet::add(const RooArgSet& list)
{
  Bool_t result(false) ;

  Int_t n= list.GetSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= add((RooAbsArg&)*list.At(index)) ;
  }

  return result;  
}



Bool_t RooArgSet::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
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
  Remove((TObject*)&var1);
  Add((TObject*)&var2);
  return kTRUE;
}



Bool_t RooArgSet::remove(const RooAbsArg& var) {
  // Remove argument from list

  // check that this isn't a copy of a list
  if(_isCopy) {
    cout << "RooArgSet: cannot remove variables in a copied list" << endl;
    return kFALSE;
  }
  // is var already in this list?
  const char *name= var.GetName();
  RooAbsArg *other= find(name);
  if(other != &var) {
    cout << "RooArgList: variable \"" << name << "\" is not in the list"
	 << " and cannot be removed" << endl;
    return kFALSE;
  }
  Remove((TObject*)&var);
  return kTRUE;
}



void RooArgSet::setAttribAll(const Text_t* name, Bool_t value) 
{
  TIterator* iter=MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    arg->setAttribute(name,value) ;
  }
}


RooArgSet* RooArgSet::selectByAttrib(const char* name, Bool_t value) 
{
  // Create output set
  RooArgSet *sel = new RooArgSet(TString(GetName()).Append("_selection")) ;
  
  // Scan set contents for matching attribute
  TIterator* iter=MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    if (arg->getAttribute(name)==value)
      sel->add(*arg) ;
  }

  // Return set if not empty
  if (sel->GetSize()) return sel ;

  // Destroy empty set and return null
  delete sel ;
  return 0 ;
}



RooAbsArg *RooArgSet::find(const char *name) const {
  // Find object with given name in list
  return (RooAbsArg*)FindObject(name);
}


Bool_t RooArgSet::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  if (compact) {
    
    TIterator *iterator= MakeIterator();
    RooStreamParser parser(is) ;
    RooAbsArg *next(0);
    while(0 != (next= (RooAbsArg*)iterator->Next())) {
      if (!next->getAttribute("Dynamic")) {
	if (next->readFromStream(is,kTRUE,verbose)) {
	  parser.zapToEnd() ;
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
  TIterator *iterator= MakeIterator();
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
  os << ClassName() << "::" << GetName() << ":" << endl;
  if(opt >= Standard) {
    TIterator *iterator= MakeIterator();
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
