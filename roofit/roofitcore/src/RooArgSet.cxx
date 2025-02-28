/***************************************************************************** * Project: RooFit                                                           *
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
/// \class RooArgSet
/// RooArgSet is a container object that can hold multiple RooAbsArg objects.
/// The container has set semantics which means that:
///
///  - Every object it contains must have a unique name returned by GetName().
///
///  - Contained objects are not ordered, although the set can be traversed
///    just like standard library containers such as `std::vector`
///    (in fact, the RooArgSet uses a `std::vector<RooAbsArg *>` under the hood).
///    The iterator does not necessarily follow the object insertion order.
///
///  - Objects can be retrieved by name only, and not by index.
///
///
/// Ownership of contents
/// -------------------------
/// Unowned objects are inserted with the add() method. Owned objects
/// are added with addOwned() or addClone(). A RooArgSet either owns all
/// of it contents, or none, which is determined by the first `add`
/// call. Once an ownership status is selected, inappropriate `add` calls
/// will return error status. Clearing the list via removeAll() resets the
/// ownership status. Arguments supplied in the constructor are always added
/// as unowned elements.
///
///
/// Uniquely identifying RooArgSet objects
/// ---------------------------------------
///
/// \warning Before v6.28, it was ensured that no RooArgSet objects on the heap
/// were located at an address that had already been used for a RooArgSet before.
/// With v6.28, this is not guaranteed anymore. Hence, if your code uses pointer
/// comparisons to uniquely identify RooArgSet instances, please consider using
/// the new `RooArgSet::uniqueId()`.

#include "RooArgSet.h"

#include "RooStreamParser.h"
#include "RooFormula.h"
#include "RooAbsRealLValue.h"
#include "RooAbsCategoryLValue.h"
#include "RooStringVar.h"
#include "RooTrace.h"
#include "RooArgList.h"
#include "RooSentinel.h"
#include "RooMsgService.h"
#include "RooConstVar.h"
#include "strlcpy.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using std::istream, std::ostream, std::ifstream, std::ofstream, std::endl;


void RooArgSet::cleanup() { }


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooArgSet::RooArgSet()
{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor from a RooArgList. If the list contains multiple
/// objects with the same name, only the first is store in the set.
/// Warning messages will be printed for dropped items.
RooArgSet::RooArgSet(const RooAbsCollection& coll) :
  RooAbsCollection(coll.GetName())
{
  add(coll,true) ; // verbose to catch duplicate errors
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor from a RooArgSet / RooArgList and a pointer to another RooFit object.
///
/// \param[in] collection Collection of RooFit objects to be added. If a list contains multiple
/// objects with the same name, only the first is stored in the set.
/// Warning messages will be printed for dropped items.
/// \param[in] var1 Further object to be added. If it is already in `collection`,
/// nothing happens, and the warning message is suppressed.
RooArgSet::RooArgSet(const RooAbsCollection& collection, const RooAbsArg* var1) :
  RooAbsCollection(collection.GetName())
{
  if (var1 && !collection.contains(*var1)) {
    add(*var1,true) ;
  }
  add(collection,true) ; // verbose to catch duplicate errors
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Empty set constructor.
RooArgSet::RooArgSet(const char *name) :
  RooAbsCollection(name)
{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a set from two existing sets. The new set will not own its
/// contents.
RooArgSet::RooArgSet(const RooArgSet& set1, const RooArgSet& set2, const char *name) : RooAbsCollection(name)
{
  add(set1) ;
  add(set2) ;
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor from a root TCollection. Elements in the collection that
/// do not inherit from RooAbsArg will be skipped. A warning message
/// will be printed for every skipped item.

RooArgSet::RooArgSet(const TCollection& tcoll, const char* name) :
  RooAbsCollection(name)
{
  for(TObject* obj : tcoll) {
    if (!dynamic_cast<RooAbsArg*>(obj)) {
      coutW(InputArguments) << "RooArgSet::RooArgSet(TCollection) element " << obj->GetName()
             << " is not a RooAbsArg, ignored" << std::endl ;
      continue ;
    }
    add(*static_cast<RooAbsArg*>(obj)) ;
  }
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Note that a copy of a set is always non-owning,
/// even if the source set owns its contents. To create an owning copy of
/// a set (owning or not), use the snapshot() method.
RooArgSet::RooArgSet(const RooArgSet& other, const char *name)
  : RooAbsCollection(other,name)
{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooArgSet::~RooArgSet()
{
  TRACE_DESTROY;
}


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Get reference to an element using its name. Named element must exist in set.
/// \throws invalid_argument if an element with the given name is not in the set.
///
/// Note that since most RooFit objects use an assignment operator that copies
/// values, an expression like
/// ```
/// mySet["x"] = y;
/// ```
/// will not replace the element "x", it just assigns the values of y.
RooAbsArg& RooArgSet::operator[](const TString& name) const
{
  RooAbsArg* arg = find(name) ;
  if (!arg) {
    coutE(InputArguments) << "RooArgSet::operator[](" << GetName() << ") ERROR: no element named " << name << " in set" << std::endl ;
    throw std::invalid_argument((TString("No element named '") + name + "' in set " + GetName()).Data());
  }
  return *arg ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if element with var's name is already in set

bool RooArgSet::checkForDup(const RooAbsArg& var, bool silent) const
{
  RooAbsArg *other = find(var);
  if (other) {
    if (other != &var) {
      if (!silent) {
   // print a warning if this variable is not the same one we
   // already have
   coutE(InputArguments) << "RooArgSet::checkForDup: ERROR argument with name " << var.GetName() << " is already in this set" << std::endl;
      }
    }
    // don't add duplicates
    return true;
  }
  return false ;
}







////////////////////////////////////////////////////////////////////////////////
/// Write contents of the argset to specified file.
/// See writeToStream() for details

void RooArgSet::writeToFile(const char* fileName) const
{
  ofstream ofs(fileName) ;
  if (ofs.fail()) {
    coutE(InputArguments) << "RooArgSet::writeToFile(" << GetName() << ") error opening file " << fileName << std::endl ;
    return ;
  }
  writeToStream(ofs,false) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Read contents of the argset from specified file.
/// See readFromStream() for details

bool RooArgSet::readFromFile(const char* fileName, const char* flagReadAtt, const char* section, bool verbose)
{
  ifstream ifs(fileName) ;
  if (ifs.fail()) {
    coutE(InputArguments) << "RooArgSet::readFromFile(" << GetName() << ") error opening file " << fileName << std::endl ;
    return true ;
  }
  return readFromStream(ifs,false,flagReadAtt,section,verbose) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Write the contents of the argset in ASCII form to given stream.
///
/// A line is written for each element contained in the form
/// `<argName> = <argValue>`
///
/// The `<argValue>` part of each element is written by the arguments'
/// writeToStream() function.
/// \param os The stream to write to.
/// \param compact Write only the bare values, separated by ' '.
/// \note In compact mode, the stream cannot be read back into a RooArgSet,
/// but only into a RooArgList, because the variable names are lost.
/// \param section If non-null, add a section header like `[<section>]`.
void RooArgSet::writeToStream(ostream& os, bool compact, const char* section) const
{
  if (section && section[0] != '\0')
    os << '[' << section << ']' << '\n';

  if (compact) {
    for (const auto next : _list) {
      next->writeToStream(os, true);
      os << " ";
    }
    os << std::endl;
  } else {
    for (const auto next : _list) {
      os << next->GetName() << " = " ;
      next->writeToStream(os,false) ;
      os << std::endl ;
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Read the contents of the argset in ASCII form from given stream.
///
/// The stream is read to end-of-file and each line is assumed to be
/// of the form
/// \code
///   <argName> = <argValue>
/// \endcode
/// Lines starting with argNames not matching any element in the list
/// will be ignored with a warning message. In addition limited C++ style
/// preprocessing and flow control is provided. The following constructions
/// are recognized:
/// \code
///   include "include.file"
/// \endcode
/// Include given file, recursive inclusion OK
/// \code
/// if (<boolean_expression>)
///   <name> = <value>
///   ....
/// else if (<boolean_expression>)
///   ....
/// else
///   ....
/// endif
/// \endcode
///
/// All expressions are evaluated by RooFormula, and may involve any of
/// the sets variables.
/// \code
///   echo <Message>
/// \endcode
/// Print console message while reading from stream
/// \code
///   abort
/// \endcode
/// Force termination of read sequence with error status
///
/// The value of each argument is read by the arguments readFromStream
/// function.

bool RooArgSet::readFromStream(istream& is, bool compact, const char* flagReadAtt, const char* section, bool verbose)
{
  if (compact) {
    coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << ") compact mode not supported" << std::endl ;
    return true ;
  }

  RooStreamParser parser(is) ;
  parser.setPunctuation("=") ;
  TString token ;
  bool retVal(false) ;

  // Conditional stack and related state variables
  // coverity[UNINIT]
  bool anyCondTrue[100] ;
  bool condStack[100] ;
  bool lastLineWasElse=false ;
  Int_t condStackLevel=0 ;
  condStack[0]=true ;

  // Prepare section processing
  TString sectionHdr("[") ;
  if (section) sectionHdr.Append(section) ;
  sectionHdr.Append("]") ;
  bool inSection(section?false:true) ;

  bool reprocessToken = false ;
  while (true) {

    if (is.eof() || is.fail() || parser.atEOF()) {
      break ;
    }

    // Read next token until memEnd of file
    if (!reprocessToken) {
      token = parser.readToken() ;
    }
    reprocessToken = false ;

    // Skip empty lines
    if (token.IsNull()) {
      continue ;
    }

    // Process include directives
    if (!token.CompareTo("include")) {
      if (parser.atEOL()) {
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName()
                   << "): no filename found after include statement" << std::endl ;
        return true ;
      }
      TString filename = parser.readLine() ;
      ifstream incfs(filename) ;
      if (!incfs.good()) {
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): cannot open include file " << filename << std::endl ;
        return true ;
      }
      coutI(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): processing include file "
          << filename << std::endl ;
      if (readFromStream(incfs,compact,flagReadAtt,inSection?nullptr:section,verbose)) return true ;
      continue ;
    }

    // Process section headers if requested
    if (*token.Data()=='[') {
      TString hdr(token) ;
      const char* last = token.Data() + token.Length() -1 ;
      if (*last != ']') {
        hdr.Append(" ") ;
        hdr.Append(parser.readLine()) ;
      }
      //       parser.putBackToken(token) ;
      //       token = parser.readLine() ;
      if (section) {
        inSection = !sectionHdr.CompareTo(hdr) ;
      }
      continue ;
    }

    // If section is specified, ignore all data outside specified section
    if (!inSection) {
      parser.zapToEnd(true) ;
      continue ;
    }

    // Conditional statement evaluation
    if (!token.CompareTo("if")) {

      // Extract conditional expressions and check validity
      TString expr = parser.readLine() ;
      RooFormula form(expr,expr,*this) ;
      if (!form.ok()) return true ;

      // Evaluate expression
      bool status = form.eval()?true:false ;
      if (lastLineWasElse) {
        anyCondTrue[condStackLevel] |= status ;
        lastLineWasElse=false ;
      } else {
        condStackLevel++ ;
        anyCondTrue[condStackLevel] = status ;
      }
      condStack[condStackLevel] = status ;

      if (verbose) {
        cxcoutD(Eval) << "RooArgSet::readFromStream(" << GetName() << "): conditional expression " << expr << " = "
                      << (condStack[condStackLevel] ? "true" : "false") << std::endl;
      }
      continue ; // go to next line
    }

    if (!token.CompareTo("else")) {
      // Must have seen an if statement before
      if (condStackLevel==0) {
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'else'" << std::endl ;
      }

      if (parser.atEOL()) {
        // simple else: process if nothing else was true
        condStack[condStackLevel] = !anyCondTrue[condStackLevel] ;
        parser.zapToEnd(false) ;
        continue ;
      } else {
        // if anything follows it should be 'if'
        token = parser.readToken() ;
        if (token.CompareTo("if")) {
          coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): syntax error: 'else " << token << "'" << std::endl ;
          return true ;
        } else {
          if (anyCondTrue[condStackLevel]) {
            // No need for further checking, true conditional already processed
            condStack[condStackLevel] = false ;
            parser.zapToEnd(false) ;
            continue ;
          } else {
            // Process as normal 'if' no true conditional was encountered
            reprocessToken = true ;
            lastLineWasElse=true ;
            continue ;
          }
        }
      }
    }

    if (!token.CompareTo("endif")) {
      // Must have seen an if statement before
      if (condStackLevel==0) {
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'endif'" << std::endl ;
        return true ;
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
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): >> " << message << std::endl ;
        continue ;
      }

      // Process abort statements
      if (!token.CompareTo("abort")) {
        TString message = parser.readLine() ;
        coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): USER ABORT" << std::endl ;
        return true ;
      }

      // Interpret the rest as <arg> = <value_expr>
      RooAbsArg *arg ;

      if ((arg = find(token)) && !arg->getAttribute("Dynamic")) {
        if (parser.expectToken("=",true)) {
          parser.zapToEnd(true) ;
          retVal=true ;
          coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName()
                << "): missing '=' sign: " << arg << std::endl ;
          continue ;
        }
        bool argRet = arg->readFromStream(is,false,verbose) ;
        if (!argRet && flagReadAtt) arg->setAttribute(flagReadAtt,true) ;
        retVal |= argRet ;
      } else {
        if (verbose) {
          coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): argument "
              << token << " not in list, ignored" << std::endl ;
        }
        parser.zapToEnd(true) ;
      }
    } else {
      parser.readLine() ;
    }
  }

  // Did we fully unwind the conditional stack?
  if (condStackLevel!=0) {
    coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): missing 'endif'" << std::endl ;
    return true ;
  }

  return retVal ;
}


bool RooArgSet::isInRange(const char* rangeSpec)
{
  char buf[1024] ;
  strlcpy(buf,rangeSpec,1024) ;
  char* token = strtok(buf,",") ;

  while(token) {

    bool accept=true ;
    for (auto * lvarg : dynamic_range_cast<RooAbsRealLValue*>(*this)) {
      if (lvarg) {
   if (!lvarg->inRange(token)) {
     accept=false ;
     break ;
   }
      }
      // WVE MUST HANDLE RooAbsCategoryLValue ranges as well
    }
    if (accept) {
      return true ;
    }

    token = strtok(nullptr,",") ;
  }

  return false ;
}


void RooArgSet::processArg(double value) { processArg(RooFit::RooConst(value)); }
