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

// -- CLASS DESCRIPTION [CAT] --
// RooMappedCategory provides a category-to-category mapping defined
// by pattern matching on their state labels
//
// The mapping function consists of a series of wild card regular expressions.
// Each expression is matched to the input categories state labels, and an associated
// output state label.

#include <cstdio>
#include <memory>
#include <cstdlib>

#include "RooFit.h"

#include "Riostream.h"
#include "TString.h"
#include "RooMappedCategory.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "TBuffer.h"
#include "TString.h"
#include "RooAbsCache.h"

ClassImp(RooMappedCategory)
ClassImp(RooMappedCategory::Entry)

class RooMappedCategoryCache : public RooAbsCache {
    public:
        RooMappedCategoryCache(RooAbsArg* owner = 0) : RooAbsCache(owner)
        { initialise(); }
        RooMappedCategoryCache(const RooAbsCache& other, RooAbsArg* owner = 0) :
            RooAbsCache(other, owner)
        { initialise(); }

        // look up our parent's output based on our parent's input category index
        const RooCatType* lookup(Int_t idx) const
        { return _map[idx]; }

        virtual void wireCache()
        { _map.clear(); initialise(); }

        virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/)
        { _map.clear(); initialise(); return kFALSE; }

    private:
        mutable std::map<Int_t, const RooCatType*> _map;

        // pre-map categories of input category on something easily searchable
        // like the index (not the name!)
        void initialise()
        {
            const RooMappedCategory& parent = *static_cast<const RooMappedCategory*>(_owner);
            TIterator* tit(static_cast<const RooAbsCategory&>(
                        parent._inputCat.arg()).typeIterator());
            for (const RooCatType* inCat = static_cast<const RooCatType*>(tit->Next());
                    inCat; inCat = static_cast<const RooCatType*>(tit->Next())) {
                const char* inKey = inCat->GetName();
                // Scan array of regexps
                bool found = false;
                for (std::map<std::string, RooMappedCategory::Entry>::const_iterator
                        iter = parent._mapArray.begin(),
                        end = parent._mapArray.end(); end != iter; ++iter) {
                    if (iter->second.match(inKey)) {
                        found = true;
                        _map[inCat->getVal()] = &(iter->second.outCat());
                        break;
                    }
                }
                if (!found) _map[inCat->getVal()] = parent._defCat;
            }
            delete tit;
        }
};

RooMappedCategory::RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defOut, Int_t defOutIdx) :
  RooAbsCategory(name, title), _inputCat("input","Input category",this,inputCat),
  _mapcache(0)
{
  // Constructor with input category and name of default output state, which is assigned
  // to all input category states that do not follow any mapping rule.
  if (defOutIdx==NoCatIdx) {
    _defCat = (RooCatType*) defineType(defOut) ;
  } else {
    _defCat = (RooCatType*) defineType(defOut,defOutIdx) ;
  }
}


RooMappedCategory::RooMappedCategory(const RooMappedCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputCat("input",this,other._inputCat), _mapArray(other._mapArray),
  _mapcache(0)
{
  _defCat = (RooCatType*) lookupType(other._defCat->GetName()) ;
}



RooMappedCategory::~RooMappedCategory()
{
    // Destructor
    delete _mapcache;
}



Bool_t RooMappedCategory::map(const char* inKeyRegExp, const char* outKey, Int_t outIdx)
{
  // Add mapping rule: any input category state label matching the 'inKeyRegExp'
  // wildcard expression will be mapped to an output state with name 'outKey'
  //
  // Rules are evaluated in the order they were added. In case an input state
  // matches more than one rule, the first rules output state will be assigned

  if (!inKeyRegExp || !outKey) return kTRUE ;

  // Check if pattern is already registered
  if (_mapArray.find(inKeyRegExp)!=_mapArray.end()) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName() << "): ERROR expression "
                          << inKeyRegExp << " already mapped" << std::endl ;
    return kTRUE ;
  }

  // Check if output type exists, if not register
  const RooCatType* outType = lookupType(outKey) ;
  if (!outType) {
    if (outIdx==NoCatIdx) {
      outType = defineType(outKey) ;
    } else {
      outType = defineType(outKey,outIdx) ;
    }
  }
  if (!outType) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName()
                          << "): ERROR, unable to output type " << outKey << std::endl ;
    return kTRUE ;
  }

  // Create new map entry ;
  Entry e(inKeyRegExp,outType) ;
  if (!e.ok()) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName()
                          << "): ERROR, expression " << inKeyRegExp << " didn't compile" << std::endl ;
    return kTRUE ;
  }

  _mapArray[inKeyRegExp] = e ;
  return kFALSE ;
}



RooCatType RooMappedCategory::evaluate() const
{
    const RooMappedCategoryCache* cache = getOrCreateCache();
    return *(cache->lookup(Int_t(_inputCat)));
}

const RooMappedCategoryCache* RooMappedCategory::getOrCreateCache() const
{
    if (!_mapcache) _mapcache = new RooMappedCategoryCache(
            const_cast<RooMappedCategory*>(this));
    return _mapcache;
}

void RooMappedCategory::printMultiline(std::ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this mapped category to the specified stream. In addition to the info
  // from RooAbsCategory::printStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of mapping rules

  RooAbsCategory::printMultiline(os,content,verbose,indent);

  if (verbose) {
    os << indent << "--- RooMappedCategory ---" << std::endl
       << indent << "  Maps from " ;
    _inputCat.arg().printStream(os,0,kStandard);

    os << indent << "  Default value is ";
    _defCat->printStream(os,kName|kValue,kSingleLine);

    os << indent << "  Mapping rules:" << std::endl;
    for (std::map<std::string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
      os << indent << "  " << iter->first << " -> " << iter->second.outCat().GetName() << std::endl ;
    }
  }
}


Bool_t RooMappedCategory::readFromStream(std::istream& is, Bool_t compact, Bool_t /*verbose*/)
{
  // Read object contents from given stream
   if (compact) {
     coutE(InputArguments) << "RooMappedCategory::readFromSteam(" << GetName() << "): can't read in compact mode" << std::endl ;
     return kTRUE ;
   } else {

     //Clear existing definitions, but preserve default output
     TString defCatName(_defCat->GetName()) ;
     _mapArray.clear() ;
     delete _mapcache;
     _mapcache = 0;
     clearTypes() ;
     _defCat = (RooCatType*) defineType(defCatName) ;

     TString token,errorPrefix("RooMappedCategory::readFromStream(") ;
     errorPrefix.Append(GetName()) ;
     errorPrefix.Append(")") ;
     RooStreamParser parser(is,errorPrefix) ;
     parser.setPunctuation(":,") ;

     TString destKey,srcKey ;
     Bool_t readToken(kTRUE) ;

    // Loop over definition sequences
     while(1) {
       if (readToken) token=parser.readToken() ;
       if (token.IsNull()) break ;
       readToken=kTRUE ;

       destKey = token ;
       if (parser.expectToken(":",kTRUE)) return kTRUE ;

       // Loop over list of sources for this destination
       while(1) {
         srcKey = parser.readToken() ;
         token = parser.readToken() ;

         // Map a value
         if (map(srcKey,destKey)) return kTRUE ;

         // Unless next token is ',' current token
         // is destination part of next sequence
         if (token.CompareTo(",")) {
           readToken = kFALSE ;
           break ;
         }
       }
     }
     return kFALSE ;
   }
   //return kFALSE ; // statement unreachable (OSF)
}


//_____________________________________________________________________________
void RooMappedCategory::printMetaArgs(std::ostream& os) const
{
  // Customized printing of arguments of a RooMappedCategory to more intuitively reflect the contents of the
  // product operator construction

  // Scan array of regexps
  RooCatType prevOutCat ;
  Bool_t first(kTRUE) ;
  os << "map=(" ;
  for (std::map<std::string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
    if (iter->second.outCat().getVal()!=prevOutCat.getVal()) {
      if (!first) { os << " " ; }
      first=kFALSE ;

      os << iter->second.outCat().GetName() << ":" << iter->first ;
      prevOutCat=iter->second.outCat() ;
    } else {
      os << "," << iter->first ;
    }
  }

  if (!first) { os << " " ; }
  os << _defCat->GetName() << ":*" ;

  os << ") " ;
}




void RooMappedCategory::writeToStream(std::ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  if (compact) {
    // Write value only
    os << getLabel() ;
  } else {
    // Write mapping expression

    // Scan array of regexps
    RooCatType prevOutCat ;
    Bool_t first(kTRUE) ;
    for (std::map<std::string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
      if (iter->second.outCat().getVal()!=prevOutCat.getVal()) {
        if (!first) { os << " " ; }
        first=kFALSE ;

        os << iter->second.outCat().GetName() << "<-" << iter->first ;
        prevOutCat=iter->second.outCat() ;
      } else {
        os << "," << iter->first ;
      }
    }

    if (!first) { os << " " ; }
    os << _defCat->GetName() << ":*" ;
  }
}




//_____________________________________________________________________________
RooMappedCategory::Entry& RooMappedCategory::Entry::operator=(const RooMappedCategory::Entry& other)
{
  if (&other==this) return *this ;

  _expr = other._expr ;
  _cat = other._cat ;

  if (_regexp) {
    delete _regexp ;
  }
  _regexp = new TRegexp(_expr.Data(),kTRUE) ;

  return *this;
}



//_____________________________________________________________________________
TString RooMappedCategory::Entry::mangle(const char* exp) const
{
  // Mangle name : escape regexp character '+'
  TString t ;
  const char *c = exp ;
  while(*c) {
    if (*c=='+') t.Append('\\') ;
    t.Append(*c) ;
    c++ ;
  }
  return t ;
}



//_____________________________________________________________________________
void RooMappedCategory::Entry::Streamer(TBuffer &R__b)
{
  typedef ::RooMappedCategory::Entry ThisClass;

   // Stream an object of class RooWorkspace::CodeRepo.
   if (R__b.IsReading()) {

     UInt_t R__s, R__c;
     R__b.ReadVersion(&R__s, &R__c);

     // Stream contents of ClassFiles map
     R__b >> _expr ;
     _cat.Streamer(R__b) ;
     _regexp = new TRegexp(_expr.Data(),kTRUE) ;
     R__b.CheckByteCount(R__s, R__c, ThisClass::IsA());

   } else {

     UInt_t R__c;
     R__c = R__b.WriteVersion(ThisClass::IsA(), kTRUE);

     // Stream contents of ClassRelInfo map
     R__b << _expr ;
     _cat.Streamer(R__b) ;

     R__b.SetByteCount(R__c, kTRUE);

   }
}
