/******************************************************
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
// RooAbsArg is the common abstract base class for objects that
// represent a value (of arbitrary type) and "shape" that in general
// depends on (is a client of) other RooAbsArg subclasses. The only
// state information about a value that is maintained in this base
// class consists of named attributes and flags that track when either
// the value or the shape of this object changes. The meaning of shape
// depends on the client implementation but could be, for example, the
// allowed range of a value. The base class is also responsible for
// managing client/server links and propagating value/shape changes
// through an expression tree. RooAbsArg implements public interfaces
// for inspecting client/server relationships and
// setting/clearing/testing named attributes.
// END_HTML
//
#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"
#include "TObjString.h"
#include "TVirtualStreamerInfo.h"
// #include "TGraphStruct.h"

#include "RooSecondMoment.h"

#include "RooMsgService.h"
#include "RooAbsArg.h"
#include "RooArgSet.h"
#include "RooArgProxy.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"
#include "RooAbsData.h"
#include "RooAbsCategoryLValue.h"
#include "RooAbsRealLValue.h"
#include "RooTrace.h"
#include "RooStringVar.h"
#include "RooRealIntegral.h"
#include "RooConstVar.h"
#include "RooMsgService.h"
#include "RooExpensiveObjectCache.h"
#include "RooAbsDataStore.h"
#include "RooResolutionModel.h"
#include "RooVectorDataStore.h"
#include "RooTreeDataStore.h"

#include <string.h>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <sstream>

using namespace std ;

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

ClassImp(RooAbsArg)
;

Bool_t RooAbsArg::_verboseDirty(kFALSE) ;
Bool_t RooAbsArg::_inhibitDirty(kFALSE) ;
Bool_t RooAbsArg::inhibitDirty() const { return _inhibitDirty && !_localNoInhibitDirty; }

std::map<RooAbsArg*,TRefArray*> RooAbsArg::_ioEvoList ;
std::stack<RooAbsArg*> RooAbsArg::_ioReadStack ;


//_____________________________________________________________________________
RooAbsArg::RooAbsArg() :
  TNamed(),
  _deleteWatch(kFALSE),
  _operMode(Auto),
  _fast(kFALSE),
  _ownedComponents(0),
  _prohibitServerRedirect(kFALSE),
  _eocache(0),
  _namePtr(0),
  _isConstant(kFALSE),
  _localNoInhibitDirty(kFALSE)
{
  // Default constructor

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

  _namePtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;

}

//_____________________________________________________________________________
RooAbsArg::RooAbsArg(const char *name, const char *title) :
  TNamed(name,title),
  _deleteWatch(kFALSE),
  _valueDirty(kTRUE),
  _shapeDirty(kTRUE),
  _operMode(Auto),
  _fast(kFALSE),
  _ownedComponents(0),
  _prohibitServerRedirect(kFALSE),
  _eocache(0),
  _namePtr(0),
  _isConstant(kFALSE),
  _localNoInhibitDirty(kFALSE)
{
  // Create an object with the specified name and descriptive title.
  // The newly created object has no clients or servers and has its
  // dirty flags set.

  _namePtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

}

//_____________________________________________________________________________
RooAbsArg::RooAbsArg(const RooAbsArg& other, const char* name)
  : TNamed(other.GetName(),other.GetTitle()),
    RooPrintable(other),
    _boolAttrib(other._boolAttrib),
    _stringAttrib(other._stringAttrib),
    _deleteWatch(other._deleteWatch),
    _operMode(Auto),
    _fast(kFALSE),
    _ownedComponents(0),
    _prohibitServerRedirect(kFALSE),
    _eocache(other._eocache),
    _namePtr(other._namePtr),
    _isConstant(other._isConstant),
    _localNoInhibitDirty(other._localNoInhibitDirty)
{
  // Copy constructor transfers all boolean and string properties of the original
  // object. Transient properties and client-server links are not copied

  // Use name in argument, if supplied
  if (name) {
    TNamed::SetName(name) ;
    _namePtr = (TNamed*) RooNameReg::instance().constPtr(name) ;
  } else {
    // Same name, Ddon't recalculate name pointer (expensive)
    TNamed::SetName(other.GetName()) ;
    _namePtr = other._namePtr ;
  }

  // Copy server list by hand
  RooFIter sIter = other._serverList.fwdIterator() ;
  RooAbsArg* server ;
  Bool_t valueProp, shapeProp ;
  while ((server = sIter.next())) {
    valueProp = server->_clientListValue.findArg(&other)?kTRUE:kFALSE ;
    shapeProp = server->_clientListShape.findArg(&other)?kTRUE:kFALSE ;
    addServer(*server,valueProp,shapeProp) ;
  }

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

  setValueDirty() ;
  setShapeDirty() ;

  //setAttribute(Form("CloneOf(%08x)",&other)) ;
  //cout << "RooAbsArg::cctor(" << this << ") #bools = " << _boolAttrib.size() << " #strings = " << _stringAttrib.size() << endl ;

}


//_____________________________________________________________________________
RooAbsArg::~RooAbsArg()
{
  // Destructor.

  // Notify all servers that they no longer need to serve us
  while (_serverList.GetSize() > 0) {
    removeServer(*static_cast<RooAbsArg*>(_serverList.First()), kTRUE) ;
  }

  // Notify all client that they are in limbo
  RooRefCountList tmpclientList(_clientList); // have to copy, as we invalidate iterators
  RooFIter clientIter = tmpclientList.fwdIterator() ;
  RooAbsArg* client = 0;
  Bool_t first(kTRUE) ;
  while ((client=clientIter.next())) {
    client->setAttribute("ServerDied") ;
    TString attr("ServerDied:");
    attr.Append(GetName());
    attr.Append(Form("(%lx)",(ULong_t)this)) ;
    client->setAttribute(attr.Data());
    client->removeServer(*this,kTRUE);

    if (_verboseDirty) {

      if (first) {
	cxcoutD(Tracing) << "RooAbsArg::dtor(" << GetName() << "," << this << ") DeleteWatch: object is being destroyed" << endl ;
	first = kFALSE ;
      }

      cxcoutD(Tracing)  << fName << "::" << ClassName() << ":~RooAbsArg: dependent \""
		       << client->GetName() << "\" should have been deleted first" << endl ;
    }
  }

  delete _clientShapeIter ;
  delete _clientValueIter ;

  if (_ownedComponents) {
    delete _ownedComponents ;
    _ownedComponents = 0 ;
  }

}


//_____________________________________________________________________________
void RooAbsArg::setDirtyInhibit(Bool_t flag)
{
  // Control global dirty inhibit mode. When set to true no value or shape dirty
  // flags are propagated and cache is always considered to be dirty.
  _inhibitDirty = flag ;
}


//_____________________________________________________________________________
void RooAbsArg::verboseDirty(Bool_t flag)
{
  // Activate verbose messaging related to dirty flag propagation
  _verboseDirty = flag ;
}

//_____________________________________________________________________________
Bool_t RooAbsArg::isCloneOf(const RooAbsArg& other) const
{
  // Check if this object was created as a clone of 'other'

  return (getAttribute(Form("CloneOf(%lx)",(ULong_t)&other)) ||
	  other.getAttribute(Form("CloneOf(%lx)",(ULong_t)this))) ;
}


//_____________________________________________________________________________
void RooAbsArg::setAttribute(const Text_t* name, Bool_t value)
{
  // Set (default) or clear a named boolean attribute of this object.

  // Preserve backward compatibility - any strong 
  if(string("Constant")==name) {
    _isConstant = value ;
  }
  
  if (value) {
    _boolAttrib.insert(name) ;
  } else {    
    set<string>::iterator iter = _boolAttrib.find(name) ;
    if (iter != _boolAttrib.end()) {
      _boolAttrib.erase(iter) ;
    }

  }

}


//_____________________________________________________________________________
Bool_t RooAbsArg::getAttribute(const Text_t* name) const
{
  // Check if a named attribute is set. By default, all attributes are unset.

  return (_boolAttrib.find(name) != _boolAttrib.end()) ;
}


//_____________________________________________________________________________
void RooAbsArg::setStringAttribute(const Text_t* key, const Text_t* value)
{
  // Associate string 'value' to this object under key 'key'

  if (value) {
    _stringAttrib[key] = value ;
  } else {
    if (_stringAttrib.find(key)!=_stringAttrib.end()) {
      _stringAttrib.erase(key) ;
    }
  }
}

//_____________________________________________________________________________
const Text_t* RooAbsArg::getStringAttribute(const Text_t* key) const
{
  // Get string attribute mapped under key 'key'. Returns null pointer
  // if no attribute exists under that key

  map<string,string>::const_iterator iter = _stringAttrib.find(key) ;
  if (iter!=_stringAttrib.end()) {
    return iter->second.c_str() ;
  } else {
    return 0 ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::setTransientAttribute(const Text_t* name, Bool_t value)
{
  // Set (default) or clear a named boolean attribute of this object.

  if (value) {

    _boolAttribTransient.insert(name) ;

  } else {

    set<string>::iterator iter = _boolAttribTransient.find(name) ;
    if (iter != _boolAttribTransient.end()) {
      _boolAttribTransient.erase(iter) ;
    }

  }

}


//_____________________________________________________________________________
Bool_t RooAbsArg::getTransientAttribute(const Text_t* name) const
{
  // Check if a named attribute is set. By default, all attributes
  // are unset.

  return (_boolAttribTransient.find(name) != _boolAttribTransient.end()) ;
}




//_____________________________________________________________________________
void RooAbsArg::addServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp)
{
  // Register another RooAbsArg as a server to us, ie, declare that
  // we depend on it. In addition to the basic client-server relationship,
  // we can declare dependence on the server's value and/or shape.

  if (_prohibitServerRedirect) {
    cxcoutF(LinkStateMgmt) << "RooAbsArg::addServer(" << this << "," << GetName()
			   << "): PROHIBITED SERVER ADDITION REQUESTED: adding server " << server.GetName()
			   << "(" << &server << ") for " << (valueProp?"value ":"") << (shapeProp?"shape":"") << endl ;
    assert(0) ;
  }

  cxcoutD(LinkStateMgmt) << "RooAbsArg::addServer(" << this << "," << GetName() << "): adding server " << server.GetName()
			 << "(" << &server << ") for " << (valueProp?"value ":"") << (shapeProp?"shape":"") << endl ;

  if (server.operMode()==ADirty && operMode()!=ADirty && valueProp) {
    setOperMode(ADirty) ;
  }


  // LM: use hash tables for larger lists
  if (_serverList.GetSize() > 999 && _serverList.getHashTableSize() == 0) _serverList.setHashTableSize(1000);
  if (server._clientList.GetSize() > 999 && server._clientList.getHashTableSize() == 0) server._clientList.setHashTableSize(1000);
  if (server._clientListValue.GetSize() >  999 && server._clientListValue.getHashTableSize() == 0) server._clientListValue.setHashTableSize(1000);

  // Add server link to given server
  _serverList.Add(&server) ;

  server._clientList.Add(this) ;
  if (valueProp) server._clientListValue.Add(this) ;
  if (shapeProp) server._clientListShape.Add(this) ;
}



//_____________________________________________________________________________
void RooAbsArg::addServerList(RooAbsCollection& serverList, Bool_t valueProp, Bool_t shapeProp)
{
  // Register a list of RooAbsArg as servers to us by calls
  // addServer() for each arg in the list

  RooAbsArg* arg ;
  RooFIter iter = serverList.fwdIterator() ;
  while ((arg=iter.next())) {
    addServer(*arg,valueProp,shapeProp) ;
  }
}



//_____________________________________________________________________________
void RooAbsArg::removeServer(RooAbsArg& server, Bool_t force)
{
  // Unregister another RooAbsArg as a server to us, ie, declare that
  // we no longer depend on its value and shape.

  if (_prohibitServerRedirect) {
    cxcoutF(LinkStateMgmt) << "RooAbsArg::addServer(" << this << "," << GetName() << "): PROHIBITED SERVER REMOVAL REQUESTED: removing server "
			   << server.GetName() << "(" << &server << ")" << endl ;
    assert(0) ;
  }

  if (_verboseDirty) {
    cxcoutD(LinkStateMgmt) << "RooAbsArg::removeServer(" << GetName() << "): removing server "
			   << server.GetName() << "(" << &server << ")" << endl ;
  }

  // Remove server link to given server
  if (!force) {
    _serverList.Remove(&server) ;

    server._clientList.Remove(this) ;
    server._clientListValue.Remove(this) ;
    server._clientListShape.Remove(this) ;
  } else {
    _serverList.RemoveAll(&server) ;

    server._clientList.RemoveAll(this) ;
    server._clientListValue.RemoveAll(this) ;
    server._clientListShape.RemoveAll(this) ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, Bool_t propValue, Bool_t propShape)
{
  // Replace 'oldServer' with 'newServer'

  Int_t count = _serverList.refCount(&oldServer);
  removeServer(oldServer, kTRUE);
  while (count--) {
    addServer(newServer, propValue, propShape);
  }
}


//_____________________________________________________________________________
void RooAbsArg::changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp)
{
  // Change dirty flag propagation mask for specified server

  if (!_serverList.findArg(&server)) {
    coutE(LinkStateMgmt) << "RooAbsArg::changeServer(" << GetName() << "): Server "
	 << server.GetName() << " not registered" << endl ;
    return ;
  }

  // This condition should not happen, but check anyway
  if (!server._clientList.findArg(this)) {
    coutE(LinkStateMgmt) << "RooAbsArg::changeServer(" << GetName() << "): Server "
			 << server.GetName() << " doesn't have us registered as client" << endl ;
    return ;
  }

  // Remove all propagation links, then reinstall requested ones ;
  Int_t vcount = server._clientListValue.refCount(this) ;
  Int_t scount = server._clientListShape.refCount(this) ;
  server._clientListValue.RemoveAll(this) ;
  server._clientListShape.RemoveAll(this) ;
  if (valueProp) {
    server._clientListValue.Add(this, vcount) ;
  }
  if (shapeProp) {
    server._clientListShape.Add(this, scount) ;
  }
}



//_____________________________________________________________________________
void RooAbsArg::leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, Bool_t recurseNonDerived) const
{
  // Fill supplied list with all leaf nodes of the arg tree, starting with
  // ourself as top node. A leaf node is node that has no servers declared.

  treeNodeServerList(list,arg,kFALSE,kTRUE,kFALSE,recurseNonDerived) ;
}



//_____________________________________________________________________________
void RooAbsArg::branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, Bool_t recurseNonDerived) const
{
  // Fill supplied list with all branch nodes of the arg tree starting with
  // ourself as top node. A branch node is node that has one or more servers declared.

  treeNodeServerList(list,arg,kTRUE,kFALSE,kFALSE,recurseNonDerived) ;
}


//_____________________________________________________________________________
void RooAbsArg::treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, Bool_t doBranch, Bool_t doLeaf, Bool_t valueOnly, Bool_t recurseFundamental) const
{
  // Fill supplied list with nodes of the arg tree, following all server links,
  // starting with ourself as top node.

//   if (arg==0) {
//     cout << "treeNodeServerList(" << GetName() << ") doBranch=" << (doBranch?"T":"F") << " doLeaf = " << (doLeaf?"T":"F") << " valueOnly=" << (valueOnly?"T":"F") << endl ;
//   }

  if (!arg) {
//     if (list->getHashTableSize()==0) {
//       list->setHashTableSize(1000) ;
//     }
    arg=this ;
  }

  // Decide if to add current node
  if ((doBranch&&doLeaf) ||
      (doBranch&&arg->isDerived()) ||
      (doLeaf&&arg->isFundamental()&&(!(recurseFundamental&&arg->isDerived()))) ||
      (doLeaf && !arg->isFundamental() && !arg->isDerived())) {

    list->add(*arg,kTRUE) ;
  }

  // Recurse if current node is derived
  if (arg->isDerived() && (!arg->isFundamental() || recurseFundamental)) {
    RooAbsArg* server ;
    RooFIter sIter = arg->serverMIterator() ;
    while ((server=sIter.next())) {

      // Skip non-value server nodes if requested
      Bool_t isValueSrv = server->_clientListValue.findArg(arg)?kTRUE:kFALSE ;
      if (valueOnly && !isValueSrv) {
	continue ;
      }
      treeNodeServerList(list,server,doBranch,doLeaf,valueOnly,recurseFundamental) ;
    }
  }
}


//_____________________________________________________________________________
RooArgSet* RooAbsArg::getParameters(const RooAbsData* set, Bool_t stripDisconnected) const
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that don't match any of the names of the variable list
  // of the supplied data set (the dependents). The caller of this
  // function is responsible for deleting the returned argset.
  // The complement of this function is getObservables()

  return getParameters(set?set->get():0,stripDisconnected) ;
}


//_____________________________________________________________________________
void RooAbsArg::addParameters(RooArgSet& params, const RooArgSet* nset,Bool_t stripDisconnected) const
{
  // INTERNAL helper function for getParameters()

  RooArgSet parList("parameters") ;

  RooFIter siter = serverMIterator() ;
  RooAbsArg* server ;

  RooArgSet nodeParamServers ;
  RooArgSet nodeBranchServers ;
  while((server=siter.next())) {
    if (server->isValueServer(*this)) {
      if (server->isFundamental()) {
	if (!nset || !server->dependsOn(*nset)) {
	  nodeParamServers.add(*server) ;
	}
      } else {
	nodeBranchServers.add(*server) ;
      }
    }
  }

  // Allow pdf to strip parameters from list before adding it
  getParametersHook(nset,&nodeParamServers,stripDisconnected) ;

  // Add parameters of this node to the combined list
  params.add(nodeParamServers,kTRUE) ;

  // Now recurse into branch servers
  RooFIter biter = nodeBranchServers.fwdIterator() ;
  while((server=biter.next())) {
    server->addParameters(params,nset) ;
  }
}


//_____________________________________________________________________________
RooArgSet* RooAbsArg::getParameters(const RooArgSet* nset, Bool_t stripDisconnected) const
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that don't match any of the names the args in the
  // supplied argset. The caller of this function is responsible
  // for deleting the returned argset. The complement of this function
  // is getObservables()


  RooArgSet parList("parameters") ;

  addParameters(parList,nset,stripDisconnected) ;

  RooArgList tmp(parList) ;
  tmp.sort() ;
  return new RooArgSet(tmp) ;
}



//_____________________________________________________________________________
RooArgSet* RooAbsArg::getObservables(const RooAbsData* set) const
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that match any of the names of the variable list
  // of the supplied data set (the dependents). The caller of this
  // function is responsible for deleting the returned argset.
  // The complement of this function is getObservables()

  if (!set) return new RooArgSet ;

  return getObservables(set->get()) ;
}


//_____________________________________________________________________________
RooArgSet* RooAbsArg::getObservables(const RooArgSet* dataList, Bool_t valueOnly) const
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that match any of the names the args in the
  // supplied argset. The caller of this function is responsible
  // for deleting the returned argset. The complement of this function
  // is getObservables()

  //cout << "RooAbsArg::getObservables(" << GetName() << ")" << endl ;

  RooArgSet* depList = new RooArgSet("dependents") ;
  if (!dataList) return depList ;

  // Make iterator over tree leaf node list
  RooArgSet leafList("leafNodeServerList") ;
  treeNodeServerList(&leafList,0,kFALSE,kTRUE,valueOnly) ;
  //leafNodeServerList(&leafList) ;
  RooFIter sIter = leafList.fwdIterator() ;

  RooAbsArg* arg ;
  if (valueOnly) {
    while ((arg=sIter.next())) {
      if (arg->dependsOnValue(*dataList) && arg->isLValue()) {
	depList->add(*arg) ;
      }
    }
  } else {
    while ((arg=sIter.next())) {
      if (arg->dependsOn(*dataList) && arg->isLValue()) {
	depList->add(*arg) ;
      }
    }
  }
  //delete sIter ;

//   // Call hook function for all branch nodes
//   RooArgSet branchList ;
//   branchNodeServerList(&branchList) ;
//   RooAbsArg* branch ;
//   RooLinkedListIter bIter = branchList.iterator() ;
//   while((branch=(RooAbsArg*)bIter.Next())) {
//     branch->getObservablesHook(dataList, depList) ;
//   }
//   //delete bIter ;
  
  return depList ;
}


RooArgSet* RooAbsArg::getComponents() const
{
  // Return a RooArgSet with all component (branch nodes) of the
  // expression tree headed by this object

  TString name(GetName()) ;
  name.Append("_components") ;

  RooArgSet* set = new RooArgSet(name) ;
  branchNodeServerList(set) ;

  return set ;
}



//_____________________________________________________________________________
Bool_t RooAbsArg::checkObservables(const RooArgSet*) const
{
  // Overloadable function in which derived classes can implement
  // consistency checks of the variables. If this function returns
  // true, indicating an error, the fitter or generator will abort.

  return kFALSE ;
}


//_____________________________________________________________________________
Bool_t RooAbsArg::recursiveCheckObservables(const RooArgSet* nset) const
{
  // Recursively call checkObservables on all nodes in the expression tree

  RooArgSet nodeList ;
  treeNodeServerList(&nodeList) ;
  RooFIter iter = nodeList.fwdIterator() ;

  RooAbsArg* arg ;
  Bool_t ret(kFALSE) ;
  while((arg=iter.next())) {
    if (arg->getAttribute("ServerDied")) {
      coutE(LinkStateMgmt) << "RooAbsArg::recursiveCheckObservables(" << GetName() << "): ERROR: one or more servers of node "
			   << arg->GetName() << " no longer exists!" << endl ;
      arg->Print("v") ;
      ret = kTRUE ;
    }
    ret |= arg->checkObservables(nset) ;
  }

  return ret ;
}


//_____________________________________________________________________________
Bool_t RooAbsArg::dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg, Bool_t valueOnly) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  Bool_t result(kFALSE);
  RooFIter sIter = serverList.fwdIterator();
  RooAbsArg* server ;
  while ((!result && (server=sIter.next()))) {
    if (dependsOn(*server,ignoreArg,valueOnly)) {
      result= kTRUE;
    }
  }
  return result;
}


//_____________________________________________________________________________
Bool_t RooAbsArg::dependsOn(const RooAbsArg& testArg, const RooAbsArg* ignoreArg, Bool_t valueOnly) const
{
  // Test whether we depend on (ie, are served by) the specified object.
  // Note that RooAbsArg objects are considered equivalent if they have
  // the same name.

  if (this==ignoreArg) return kFALSE ;

  // First check if testArg is self    
  //if (!TString(testArg.GetName()).CompareTo(GetName())) return kTRUE ;
  if (testArg.namePtr()==namePtr()) return kTRUE ; 


  // Next test direct dependence
  RooAbsArg* server = findServer(testArg) ;
  if (server!=0) {

    // Return true if valueOnly is FALSE or if server is value server, otherwise keep looking
    if ( !valueOnly || server->isValueServer(*this)) {
      return kTRUE ;
    }
  }

  // If not, recurse
  RooFIter sIter = serverMIterator() ;
  while ((server=sIter.next())) {

    if ( !valueOnly || server->isValueServer(*this)) {
      if (server->dependsOn(testArg,ignoreArg,valueOnly)) {
	return kTRUE ;
      }
    }
  }

  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooAbsArg::overlaps(const RooAbsArg& testArg, Bool_t valueOnly) const
{
  // Test if any of the nodes of tree are shared with that of the given tree

  RooArgSet list("treeNodeList") ;
  treeNodeServerList(&list) ;

  return valueOnly ? testArg.dependsOnValue(list) : testArg.dependsOn(list) ;
}



//_____________________________________________________________________________
Bool_t RooAbsArg::observableOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const
{
  // Test if any of the dependents of the arg tree (as determined by getObservables)
  // overlaps with those of the testArg.

  return observableOverlaps(dset->get(),testArg) ;
}


//_____________________________________________________________________________
Bool_t RooAbsArg::observableOverlaps(const RooArgSet* nset, const RooAbsArg& testArg) const
{
  // Test if any of the dependents of the arg tree (as determined by getObservables)
  // overlaps with those of the testArg.

  RooArgSet* depList = getObservables(nset) ;
  Bool_t ret = testArg.dependsOn(*depList) ;
  delete depList ;
  return ret ;
}



//_____________________________________________________________________________
void RooAbsArg::setValueDirty(const RooAbsArg* source) const
{
  // Mark this object as having changed its value, and propagate this status
  // change to all of our clients. If the object is not in automatic dirty
  // state propagation mode, this call has no effect

  if (_operMode!=Auto || _inhibitDirty) return ;

  // Handle no-propagation scenarios first
  if (_clientListValue.GetSize()==0) {
    _valueDirty = kTRUE ;
    return ;
  }

  // Cyclical dependency interception
  if (source==0) {
    source=this ;
  } else if (source==this) {
    // Cyclical dependency, abort
    coutE(LinkStateMgmt) << "RooAbsArg::setValueDirty(" << GetName()
			 << "): cyclical dependency detected, source = " << source->GetName() << endl ;
    //assert(0) ;
    return ;
  }

  // Propagate dirty flag to all clients if this is a down->up transition
  if (_verboseDirty) {
    cxcoutD(LinkStateMgmt) << "RooAbsArg::setValueDirty(" << (source?source->GetName():"self") << "->" << GetName() << "," << this
			   << "): dirty flag " << (_valueDirty?"already ":"") << "raised" << endl ;
  }

  _valueDirty = kTRUE ;


  RooFIter clientValueIter = _clientListValue.fwdIterator() ;
  RooAbsArg* client ;
  while ((client=clientValueIter.next())) {
    client->setValueDirty(source) ;
  }


}


//_____________________________________________________________________________
void RooAbsArg::setShapeDirty(const RooAbsArg* source) const
{
  // Mark this object as having changed its shape, and propagate this status
  // change to all of our clients.

  if (_verboseDirty) {
    cxcoutD(LinkStateMgmt) << "RooAbsArg::setShapeDirty(" << GetName()
			   << "): dirty flag " << (_shapeDirty?"already ":"") << "raised" << endl ;
  }

  if (_clientListShape.GetSize()==0) {
    _shapeDirty = kTRUE ;
    return ;
  }

  // Set 'dirty' shape state for this object and propagate flag to all its clients
  if (source==0) {
    source=this ;
  } else if (source==this) {
    // Cyclical dependency, abort
    coutE(LinkStateMgmt) << "RooAbsArg::setShapeDirty(" << GetName()
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  // Propagate dirty flag to all clients if this is a down->up transition
  _shapeDirty=kTRUE ;

  RooFIter clientShapeIter = _clientListShape.fwdIterator() ;
  RooAbsArg* client ;
  while ((client=clientShapeIter.next())) {
    client->setShapeDirty(source) ;
    client->setValueDirty(source) ;
  }

}



//_____________________________________________________________________________
Bool_t RooAbsArg::redirectServers(const RooAbsCollection& newSetOrig, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursionStep)
{
  // Substitute our servers with those listed in newSet. If nameChange is false, servers and
  // and substitutes are matched by name. If nameChange is true, servers are matched to args
  // in newSet that have the 'ORIGNAME:<servername>' attribute set. If mustReplaceAll is set,
  // a warning is printed and error status is returned if not all servers could be sucessfully
  // substituted.

  // Trivial case, no servers
  if (!_serverList.First()) return kFALSE ;
  if (newSetOrig.getSize()==0) return kFALSE ;

  // Strip any non-matchin removal nodes from newSetOrig
  RooAbsCollection* newSet ;

  if (nameChange) {

    newSet = new RooArgSet ;
    RooFIter iter = newSetOrig.fwdIterator() ;
    RooAbsArg* arg ;
    while((arg=iter.next())) {

      if (string("REMOVAL_DUMMY")==arg->GetName()) {
	
	if (arg->getAttribute("REMOVE_ALL")) {
// 	  cout << "RooAbsArg::redir including remove_all node " << arg->GetName() << endl ;
	  newSet->add(*arg) ;
	} else if (arg->getAttribute(Form("REMOVE_FROM_%s",getStringAttribute("ORIGNAME")))) {
// 	  cout << "RooAbsArg::redir including remove_from_" << GetName() << " node " << arg->GetName() << endl ;
	  newSet->add(*arg) ;
	}
      } else {
	newSet->add(*arg) ;
      }
    }

//     cout << "RooAbsArg::redirect with name change(" << GetName() << ") newSet = " << newSet << " origSet = " << newSetOrig << endl ;

  } else {
    newSet = (RooAbsCollection*) &newSetOrig ;
  }

  // Replace current servers with new servers with the same name from the given list
  Bool_t ret(kFALSE) ;

  //Copy original server list to not confuse the iterator while deleting
  RooLinkedList origServerList, origServerValue, origServerShape ;
  RooAbsArg *oldServer, *newServer ;
  RooFIter sIter = _serverList.fwdIterator() ;
  while ((oldServer=sIter.next())) {
    origServerList.Add(oldServer) ;

    // Retrieve server side link state information
    if (oldServer->_clientListValue.findArg(this)) {
      origServerValue.Add(oldServer) ;
    }
    if (oldServer->_clientListShape.findArg(this)) {
      origServerShape.Add(oldServer) ;
    }
  }

  // Delete all previously registered servers
  sIter = origServerList.fwdIterator() ;
  Bool_t propValue, propShape ;
  while ((oldServer=sIter.next())) {

    newServer= oldServer->findNewServer(*newSet, nameChange);

    if (newServer && _verboseDirty) {
      cxcoutD(LinkStateMgmt) << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName()
			     << " redirected from " << oldServer << " to " << newServer << endl ;
    }
    
    if (!newServer) {
      if (mustReplaceAll) {
	cxcoutD(LinkStateMgmt) << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName()
			       << " (" << (void*)oldServer << ") not redirected" << (nameChange?"[nameChange]":"") << endl ;
	ret = kTRUE ;
      }
      continue ;
    }
    
    propValue=origServerValue.findArg(oldServer)?kTRUE:kFALSE ;
    propShape=origServerShape.findArg(oldServer)?kTRUE:kFALSE ;
    // cout << "replaceServer with name " << oldServer->GetName() << " old=" << oldServer << " new=" << newServer << endl ;
    if (newServer != this) {
      replaceServer(*oldServer,*newServer,propValue,propShape) ;
    }
  }


  setValueDirty() ;
  setShapeDirty() ;

  // Process the proxies
  Bool_t allReplaced=kTRUE ;
  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!p) continue ;
    Bool_t ret2 = p->changePointer(*newSet,nameChange,kFALSE) ; 
    allReplaced &= ret2 ;
  }
  
  if (mustReplaceAll && !allReplaced) {
    coutE(LinkStateMgmt) << "RooAbsArg::redirectServers(" << GetName()
			 << "): ERROR, some proxies could not be adjusted" << endl ;
    ret = kTRUE ;
  }

  // Optional subclass post-processing
  for (Int_t i=0 ;i<numCaches() ; i++) {
    ret |= getCache(i)->redirectServersHook(*newSet,mustReplaceAll,nameChange,isRecursionStep) ;
  }
  ret |= redirectServersHook(*newSet,mustReplaceAll,nameChange,isRecursionStep) ;

  if (nameChange) {
    delete newSet ;
  }

  return ret ;
}

//_____________________________________________________________________________
RooAbsArg *RooAbsArg::findNewServer(const RooAbsCollection &newSet, Bool_t nameChange) const
{
  // Find the new server in the specified set that matches the old server.
  // Allow a name change if nameChange is kTRUE, in which case the new
  // server is selected by searching for a new server with an attribute
  // of "ORIGNAME:<oldName>". Return zero if there is not a unique match.

  RooAbsArg *newServer = 0;
  if (!nameChange) {
    newServer = newSet.find(*this) ;
  }
  else {
    // Name changing server redirect:
    // use 'ORIGNAME:<oldName>' attribute instead of name of new server
    TString nameAttrib("ORIGNAME:") ;
    nameAttrib.Append(GetName()) ;

    RooArgSet* tmp = (RooArgSet*) newSet.selectByAttrib(nameAttrib,kTRUE) ;
    if(0 != tmp) {

      // Check if any match was found
      if (tmp->getSize()==0) {
	delete tmp ;
	return 0 ;
      }

      // Check if match is unique
      if(tmp->getSize()>1) {
	coutF(LinkStateMgmt) << "RooAbsArg::redirectServers(" << GetName() << "): FATAL Error, " << tmp->getSize() << " servers with "
			     << nameAttrib << " attribute" << endl ;
	tmp->Print("v") ;
	assert(0) ;
      }

      // use the unique element in the set
      newServer= tmp->first();
      delete tmp ;
    }
  }
  return newServer;
}

Bool_t RooAbsArg::recursiveRedirectServers(const RooAbsCollection& newSet, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t recurseInNewSet)
{
  // Recursively redirect all servers with new server in collection 'newSet'.
  // Substitute our servers with those listed in newSet. If nameChange is false, servers and
  // and substitutes are matched by name. If nameChange is true, servers are matched to args
  // in newSet that have the 'ORIGNAME:<servername>' attribute set. If mustReplaceAll is set,
  // a warning is printed and error status is returned if not all servers could be sucessfully
  // substituted. If recurseInNewSet is true, the recursion algorithm also recursion into
  // expression trees under the arguments in the new servers (i.e. those in newset)


  // Cyclic recursion protection
  static RooLinkedList callStack ;
  if (callStack.findArg(this)) {
    return kFALSE ;
  } else {
    callStack.Add(this) ;
  }

  // Do not recurse into newset if not so specified
//   if (!recurseInNewSet && newSet.contains(*this)) {
//     return kFALSE ;
//   }


  // Apply the redirectServers function recursively on all branch nodes in this argument tree.
  Bool_t ret(kFALSE) ;

  cxcoutD(LinkStateMgmt) << "RooAbsArg::recursiveRedirectServers(" << this << "," << GetName() << ") newSet = " << newSet << " mustReplaceAll = "
			 << (mustReplaceAll?"T":"F") << " nameChange = " << (nameChange?"T":"F") << " recurseInNewSet = " << (recurseInNewSet?"T":"F") << endl ;

  // Do redirect on self (identify operation as recursion step)
  ret |= redirectServers(newSet,mustReplaceAll,nameChange,kTRUE) ;

  // Do redirect on servers
  RooFIter sIter = serverMIterator() ;
  RooAbsArg* server ;
  while((server=sIter.next())) {
    ret |= server->recursiveRedirectServers(newSet,mustReplaceAll,nameChange,recurseInNewSet) ;
  }

  callStack.Remove(this) ;
  return ret ;
}



//_____________________________________________________________________________
void RooAbsArg::registerProxy(RooArgProxy& proxy)
{
  // Register an RooArgProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur. The proxied argument is
  // also added as value and/or shape server

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    coutE(LinkStateMgmt) << "RooAbsArg::registerProxy(" << GetName() << "): proxy named "
			 << proxy.GetName() << " for arg " << proxy.absArg()->GetName()
			 << " already registered" << endl ;
    return ;
  }

//   cout << (void*)this << " " << GetName() << ": registering proxy "
//        << (void*)&proxy << " with name " << proxy.name() << " in mode "
//        << (proxy.isValueServer()?"V":"-") << (proxy.isShapeServer()?"S":"-") << endl ;

  // Register proxied object as server
  if (proxy.absArg()) {
    addServer(*proxy.absArg(),proxy.isValueServer(),proxy.isShapeServer()) ;
  }

  // Register proxy itself
  _proxyList.Add(&proxy) ;
}


//_____________________________________________________________________________
void RooAbsArg::unRegisterProxy(RooArgProxy& proxy)
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
}



//_____________________________________________________________________________
void RooAbsArg::registerProxy(RooSetProxy& proxy)
{
  // Register an RooSetProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur.

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    coutE(LinkStateMgmt) << "RooAbsArg::registerProxy(" << GetName() << "): proxy named "
			 << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  _proxyList.Add(&proxy) ;
}



//_____________________________________________________________________________
void RooAbsArg::unRegisterProxy(RooSetProxy& proxy)
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
}



//_____________________________________________________________________________
void RooAbsArg::registerProxy(RooListProxy& proxy)
{
  // Register an RooListProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur.

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    coutE(LinkStateMgmt) << "RooAbsArg::registerProxy(" << GetName() << "): proxy named "
			 << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  Int_t nProxyOld = _proxyList.GetEntries() ;
  _proxyList.Add(&proxy) ;
  if (_proxyList.GetEntries()!=nProxyOld+1) {
    cout << "RooAbsArg::registerProxy(" << GetName() << ") proxy registration failure! nold=" << nProxyOld << " nnew=" << _proxyList.GetEntries() << endl ;
  }
}



//_____________________________________________________________________________
void RooAbsArg::unRegisterProxy(RooListProxy& proxy)
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
}



//_____________________________________________________________________________
RooAbsProxy* RooAbsArg::getProxy(Int_t index) const
{
  // Return the nth proxy from the proxy list.

  // Cross cast: proxy list returns TObject base pointer, we need
  // a RooAbsProxy base pointer. C++ standard requires
  // a dynamic_cast for this.
  return dynamic_cast<RooAbsProxy*> (_proxyList.At(index)) ;
}



//_____________________________________________________________________________
Int_t RooAbsArg::numProxies() const
{
  // Return the number of registered proxies.

  return _proxyList.GetEntries() ;
}



//_____________________________________________________________________________
void RooAbsArg::setProxyNormSet(const RooArgSet* nset)
{
  // Forward a change in the cached normalization argset
  // to all the registered proxies.

  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!p) continue ;
    getProxy(i)->changeNormSet(nset) ;
  }
}



//_____________________________________________________________________________
void RooAbsArg::attachToTree(TTree& ,Int_t)
{
  // Overloadable function for derived classes to implement
  // attachment as branch to a TTree

  coutE(Contents) << "RooAbsArg::attachToTree(" << GetName()
		  << "): Cannot be attached to a TTree" << endl ;
}



//_____________________________________________________________________________
Bool_t RooAbsArg::isValid() const
{
  // WVE (08/21/01) Probably obsolete now
  return kTRUE ;
}




//_____________________________________________________________________________
void RooAbsArg::printName(ostream& os) const
{
  // Print object name

  os << GetName() ;
}



//_____________________________________________________________________________
void RooAbsArg::printTitle(ostream& os) const
{
  // Print object title
  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooAbsArg::printClassName(ostream& os) const
{
  // Print object class name
  os << IsA()->GetName() ;
}


void RooAbsArg::printAddress(ostream& os) const
{
  // Print addrss of this RooAbsArg
  os << this ;
}



//_____________________________________________________________________________
void RooAbsArg::printArgs(ostream& os) const
{
  // Print object arguments, ie its proxies

  // Print nothing if there are no dependencies
  if (numProxies()==0) return ;

  os << "[ " ;
  for (Int_t i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (p==0) continue ;
    if (!TString(p->name()).BeginsWith("!")) {
      p->print(os) ;
      os << " " ;
    }
  }
  printMetaArgs(os) ;
  os << "]" ;
}



//_____________________________________________________________________________
Int_t RooAbsArg::defaultPrintContents(Option_t* /*opt*/) const
{
  // Define default contents to print
  return kName|kClassName|kValue|kArgs ;
}



//_____________________________________________________________________________
void RooAbsArg::printMultiline(ostream& os, Int_t /*contents*/, Bool_t /*verbose*/, TString indent) const
{
  // Implement multi-line detailed printing

  os << indent << "--- RooAbsArg ---" << endl;
  // dirty state flags
  os << indent << "  Value State: " ;
  switch(_operMode) {
  case ADirty: os << "FORCED DIRTY" ; break ;
  case AClean: os << "FORCED clean" ; break ;
  case Auto: os << (isValueDirty() ? "DIRTY":"clean") ; break ;
  }
  os << endl
     << indent << "  Shape State: " << (isShapeDirty() ? "DIRTY":"clean") << endl;
  // attribute list
  os << indent << "  Attributes: " ;
  printAttribList(os) ;
  os << endl ;
  // our memory address (for x-referencing with client addresses of other args)
  os << indent << "  Address: " << (void*)this << endl;
  // client list
  os << indent << "  Clients: " << endl;
  RooFIter clientIter= _clientList.fwdIterator();
  RooAbsArg* client ;
  while ((client=clientIter.next())) {
    os << indent << "    (" << (void*)client  << ","
       << (_clientListValue.findArg(client)?"V":"-")
       << (_clientListShape.findArg(client)?"S":"-")
       << ") " ;
    client->printStream(os,kClassName|kTitle|kName,kSingleLine);
  }

  // server list
  os << indent << "  Servers: " << endl;
  RooFIter serverIter= _serverList.fwdIterator();
  RooAbsArg* server ;
  while ((server=serverIter.next())) {
    os << indent << "    (" << (void*)server << ","
       << (server->_clientListValue.findArg(this)?"V":"-")
       << (server->_clientListShape.findArg(this)?"S":"-")
       << ") " ;
    server->printStream(os,kClassName|kName|kTitle,kSingleLine);
  }

  // proxy list
  os << indent << "  Proxies: " << endl ;
  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* proxy=getProxy(i) ;
    if (!proxy) continue ;
    if (proxy->IsA()->InheritsFrom(RooArgProxy::Class())) {
      os << indent << "    " << proxy->name() << " -> " ;
      RooAbsArg* parg = ((RooArgProxy*)proxy)->absArg() ;
      if (parg) {
	parg->printStream(os,kName,kSingleLine) ;
      } else {
	os << " (empty)" << endl ; ;
      }
    } else {
      os << indent << "    " << proxy->name() << " -> " ;
      os << endl ;
      TString moreIndent(indent) ;
      moreIndent.Append("    ") ;
      ((RooSetProxy*)proxy)->printStream(os,kName,kStandard,moreIndent.Data()) ;
    }
  }
}


//_____________________________________________________________________________
void RooAbsArg::printTree(ostream& os, TString /*indent*/) const
{
  // Print object tree structure
  const_cast<RooAbsArg*>(this)->printCompactTree(os) ;
}


//_____________________________________________________________________________
ostream& operator<<(ostream& os, RooAbsArg &arg)
{
  // Ostream operator
  arg.writeToStream(os,kTRUE) ;
  return os ;
}

//_____________________________________________________________________________
istream& operator>>(istream& is, RooAbsArg &arg)
{
  // Istream operator
  arg.readFromStream(is,kTRUE,kFALSE) ;
  return is ;
}

//_____________________________________________________________________________
void RooAbsArg::printAttribList(ostream& os) const
{
  // Print the attribute list

  set<string>::const_iterator iter = _boolAttrib.begin() ;
  Bool_t first(kTRUE) ;
  while (iter != _boolAttrib.end()) {
    os << (first?" [":",") << *iter ;
    first=kFALSE ;
    ++iter ;
  }
  if (!first) os << "] " ;
}

//_____________________________________________________________________________
void RooAbsArg::attachDataSet(const RooAbsData &data)
{
  // Replace server nodes with names matching the dataset variable names
  // with those data set variables, making this PDF directly dependent on the dataset

  const RooArgSet* set = data.get() ;
  RooArgSet branches ;
  branchNodeServerList(&branches,0,kTRUE) ;

  RooFIter iter = branches.fwdIterator() ;
  RooAbsArg* branch ;
  while((branch=iter.next())) {
    branch->redirectServers(*set,kFALSE,kFALSE) ;
  }
}



//_____________________________________________________________________________
void RooAbsArg::attachDataStore(const RooAbsDataStore &dstore)
{
  // Replace server nodes with names matching the dataset variable names
  // with those data set variables, making this PDF directly dependent on the dataset

  const RooArgSet* set = dstore.get() ;
  RooArgSet branches ;
  branchNodeServerList(&branches,0,kTRUE) ;

  RooFIter iter = branches.fwdIterator() ;
  RooAbsArg* branch ;
  while((branch=iter.next())) {
    branch->redirectServers(*set,kFALSE,kFALSE) ;
  }
}



//_____________________________________________________________________________
Int_t RooAbsArg::Compare(const TObject* other) const
{
  // Utility function used by TCollection::Sort to compare contained TObjects
  // We implement comparison by name, resulting in alphabetical sorting by object name.

  return strcmp(GetName(),other->GetName()) ;
}



//_____________________________________________________________________________
void RooAbsArg::printDirty(Bool_t depth) const
{
  // Print information about current value dirty state information.
  // If depth flag is true, information is recursively printed for
  // all nodes in this arg tree.

  if (depth) {

    RooArgSet branchList ;
    branchNodeServerList(&branchList) ;
    RooFIter bIter = branchList.fwdIterator() ;
    RooAbsArg* branch ;
    while((branch=bIter.next())) {
      branch->printDirty(kFALSE) ;
    }

  } else {
    cout << GetName() << " : " ;
    switch (_operMode) {
    case AClean: cout << "FORCED clean" ; break ;
    case ADirty: cout << "FORCED DIRTY" ; break ;
    case Auto:   cout << "Auto  " << (isValueDirty()?"DIRTY":"clean") ;
    }
    cout << endl ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::optimizeCacheMode(const RooArgSet& observables)
{
  // Activate cache mode optimization with given definition of observables.
  // The cache operation mode of all objects in the expression tree will
  // modified such that all nodes that depend directly or indirectly on
  // any of the listed observables will be set to ADirty, as they are
  // expected to change every time. This save change tracking overhead for
  // nodes that are a priori known to change every time

  RooLinkedList proc;
  RooArgSet opt ;
  optimizeCacheMode(observables,opt,proc) ;

  coutI(Optimization) << "RooAbsArg::optimizeCacheMode(" << GetName() << ") nodes " << opt << " depend on observables, "
			<< "changing cache operation mode from change tracking to unconditional evaluation" << endl ;
}


//_____________________________________________________________________________
void RooAbsArg::optimizeCacheMode(const RooArgSet& observables, RooArgSet& optimizedNodes, RooLinkedList& processedNodes)
{
  // Activate cache mode optimization with given definition of observables.
  // The cache operation mode of all objects in the expression tree will
  // modified such that all nodes that depend directly or indirectly on
  // any of the listed observables will be set to ADirty, as they are
  // expected to change every time. This save change tracking overhead for
  // nodes that are a priori known to change every time

  // Optimization applies only to branch nodes, not to leaf nodes
  if (!isDerived()) {
    return ;
  }


  // Terminate call if this node was already processed (tree structure may be cyclical)
  if (processedNodes.findArg(this)) {
    return ;
  } else {
    processedNodes.Add(this) ;
  }

  // Set cache mode operator to 'AlwaysDirty' if we depend on any of the given observables
  if (dependsOnValue(observables)) {

    if (dynamic_cast<RooRealIntegral*>(this)) {
      cxcoutI(Integration) << "RooAbsArg::optimizeCacheMode(" << GetName() << ") integral depends on value of one or more observables and will be evaluated for every event" << endl ;
    }
    optimizedNodes.add(*this,kTRUE) ;
    if (operMode()==AClean) {
    } else {
      setOperMode(ADirty,kTRUE) ; // WVE propagate flag recursively to top of tree
    }
  } else {
  }
  // Process any RooAbsArgs contained in any of the caches of this object
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->optimizeCacheMode(observables,optimizedNodes,processedNodes) ;
  }

  // Forward calls to all servers
  RooFIter sIter = serverMIterator() ;
  RooAbsArg* server ;
  while((server=sIter.next())) {
    server->optimizeCacheMode(observables,optimizedNodes,processedNodes) ;
  }

}

//_____________________________________________________________________________
Bool_t RooAbsArg::findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList)
{
  // Find branch nodes with all-constant parameters, and add them to the list of
  // nodes that can be cached with a dataset in a test statistic calculation

  RooLinkedList proc ;
  Bool_t ret = findConstantNodes(observables,cacheList,proc) ;

  // If node can be optimized and hasn't been identified yet, add it to the list
  coutI(Optimization) << "RooAbsArg::findConstantNodes(" << GetName() << "): components "
			<< cacheList << " depend exclusively on constant parameters and will be precalculated and cached" << endl ;

  return ret ;
}



//_____________________________________________________________________________
Bool_t RooAbsArg::findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList, RooLinkedList& processedNodes)
{
  // Find branch nodes with all-constant parameters, and add them to the list of
  // nodes that can be cached with a dataset in a test statistic calculation

  // Caching only applies to branch nodes
  if (!isDerived()) {
    return kFALSE;
  }

  // Terminate call if this node was already processed (tree structure may be cyclical)
  if (processedNodes.findArg(this)) {
    return kFALSE ;
  } else {
    processedNodes.Add(this) ;
  }

  // Check if node depends on any non-constant parameter
  Bool_t canOpt(kTRUE) ;
  RooArgSet* paramSet = getParameters(observables) ;
  RooFIter iter = paramSet->fwdIterator() ;
  RooAbsArg* param ;
  while((param = iter.next())) {
    if (!param->isConstant()) {
      canOpt=kFALSE ;
      break ;
    }
  }
  delete paramSet ;


  if (getAttribute("NeverConstant")) {
    canOpt = kFALSE ;
  }

  if (canOpt) {
    setAttribute("ConstantExpression") ;
  }

  // If yes, list node eligible for caching, if not test nodes one level down
  if (canOpt||getAttribute("CacheAndTrack")) {
    
    if (!cacheList.find(*this) && dependsOnValue(observables) && !observables.find(*this) ) {
      
      // Add to cache list
      cxcoutD(Optimization) << "RooAbsArg::findConstantNodes(" << GetName() << ") adding self to list of constant nodes" << endl ;

      if (canOpt) setAttribute("ConstantExpressionCached") ;
      cacheList.add(*this,kFALSE) ;
    }
  }

  if (!canOpt) {
    
    // If not, see if next level down can be cached
    RooFIter sIter = serverMIterator() ;
    RooAbsArg* server ;
    while((server=sIter.next())) {
      if (server->isDerived()) {
	server->findConstantNodes(observables,cacheList,processedNodes) ;
      }
    }
  }
  
  // Forward call to all cached contained in current object
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->findConstantNodes(observables,cacheList,processedNodes) ;
  }

  return kFALSE ;
}




//_____________________________________________________________________________
void RooAbsArg::constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTrackingOpt)
{
  // Interface function signaling a request to perform constant term
  // optimization. This default implementation takes no action other than to
  // forward the calls to all servers

  RooFIter sIter = serverMIterator() ;
  RooAbsArg* server ;
  while((server=sIter.next())) {
    server->constOptimizeTestStatistic(opcode,doAlsoTrackingOpt) ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::setOperMode(OperMode mode, Bool_t recurseADirty)
{
  // Change cache operation mode to given mode. If recurseAdirty
  // is true, then a mode change to AlwaysDirty will automatically
  // be propagated recursively to all client nodes

  // Prevent recursion loops
  if (mode==_operMode) return ;

  _operMode = mode ;
  _fast = ((mode==AClean) || dynamic_cast<RooRealVar*>(this)!=0 || dynamic_cast<RooConstVar*>(this)!=0 ) ;
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->operModeHook() ;
  }
  operModeHook() ;

  // Propagate to all clients
  if (mode==ADirty && recurseADirty) {
    RooFIter iter = valueClientMIterator() ;
    RooAbsArg* client ;
    while((client=iter.next())) {
      client->setOperMode(mode) ;
    }
  }
}


//_____________________________________________________________________________
void RooAbsArg::printCompactTree(const char* indent, const char* filename, const char* namePat, RooAbsArg* client)
{
  // Print tree structure of expression tree on stdout, or to file if filename is specified.
  // If namePat is not "*", only nodes with names matching the pattern will be printed.
  // The client argument is used in recursive calls to properly display the value or shape nature
  // of the client-server links. It should be zero in calls initiated by users.

  if (filename) {
    ofstream ofs(filename) ;
    printCompactTree(ofs,indent,namePat,client) ;
  } else {
    printCompactTree(cout,indent,namePat,client) ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::printCompactTree(ostream& os, const char* indent, const char* namePat, RooAbsArg* client)
{
  // Print tree structure of expression tree on given ostream.
  // If namePat is not "*", only nodes with names matching the pattern will be printed.
  // The client argument is used in recursive calls to properly display the value or shape nature
  // of the client-server links. It should be zero in calls initiated by users.

  if ( !namePat || TString(GetName()).Contains(namePat)) {
    os << indent << this ;
    if (client) {
      os << "/" ;
      if (isValueServer(*client)) os << "V" ; else os << "-" ;
      if (isShapeServer(*client)) os << "S" ; else os << "-" ;
    }
    os << " " ;

    os << IsA()->GetName() << "::" << GetName() <<  " = " ;
    printValue(os) ;

    if (_serverList.GetSize()>0) {
      switch(operMode()) {
      case Auto:   os << " [Auto," << (isValueDirty()?"Dirty":"Clean") << "] "  ; break ;
      case AClean: os << " [ACLEAN] " ; break ;
      case ADirty: os << " [ADIRTY] " ; break ;
      }
    }
    os << endl ;

    for (Int_t i=0 ;i<numCaches() ; i++) {
      getCache(i)->printCompactTreeHook(os,indent) ;
    }
    printCompactTreeHook(os,indent) ;
  }

  TString indent2(indent) ;
  indent2 += "  " ;
  RooFIter iter = serverMIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    arg->printCompactTree(os,indent2,namePat,this) ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::printComponentTree(const char* indent, const char* namePat, Int_t nLevel)
{
  // Print tree structure of expression tree on given ostream, only branch nodes are printed.
  // Lead nodes (variables) will not be shown
  //
  // If namePat is not "*", only nodes with names matching the pattern will be printed.

  if (nLevel==0) return ;
  if (isFundamental()) return ;
  RooResolutionModel* rmodel = dynamic_cast<RooResolutionModel*>(this) ;
  if (rmodel && rmodel->isConvolved()) return ;
  if (InheritsFrom("RooConstVar")) return ;

  if ( !namePat || TString(GetName()).Contains(namePat)) {
    cout << indent ;
    Print() ;
  }

  TString indent2(indent) ;
  indent2 += "  " ;
  RooFIter iter = serverMIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    arg->printComponentTree(indent2.Data(),namePat,nLevel-1) ;
  }
}


//_____________________________________________________________________________
TString RooAbsArg::cleanBranchName() const
{
  // Construct a mangled name from the actual name that
  // is free of any math symbols that might be interpreted by TTree

  // Check for optional alternate name of branch for this argument
  TString rawBranchName = GetName() ;
  if (getStringAttribute("BranchName")) {
    rawBranchName = getStringAttribute("BranchName") ;
  }

  TString cleanName(rawBranchName) ;
  cleanName.ReplaceAll("/","D") ;
  cleanName.ReplaceAll("-","M") ;
  cleanName.ReplaceAll("+","P") ;
  cleanName.ReplaceAll("*","X") ;
  cleanName.ReplaceAll("[","L") ;
  cleanName.ReplaceAll("]","R") ;
  cleanName.ReplaceAll("(","L") ;
  cleanName.ReplaceAll(")","R") ;
  cleanName.ReplaceAll("{","L") ;
  cleanName.ReplaceAll("}","R") ;

  if (cleanName.Length()<=60) return cleanName ;

  // Name is too long, truncate and include CRC32 checksum of full name in clean name
  static char buf[1024] ;
  strlcpy(buf,cleanName.Data(),1024) ;
  snprintf(buf+46,1024-46,"_CRC%08x",crc32(cleanName.Data())) ;

  return TString(buf) ;
}





UInt_t RooAbsArg::crc32(const char* data)
{
  // Calculate crc32 checksum on given string
  unsigned long sz = strlen(data);
  switch (strlen(data)) {
    case 0:
      return 0;
    case 1:
      return data[0];
    case 2:
      return (data[0] << 8) | data[1];
    case 3:
      return (data[0] << 16) | (data[1] << 8) | data[2];
    case 4:
      return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    default:
      return crc32(data + 4, sz - 4, (data[0] << 24) | (data[1] << 16) |
	  (data[2] << 8) | data[3]);
  }
}


UInt_t RooAbsArg::crc32(const char* data, ULong_t sz, UInt_t crc)
{
  // update CRC32 with new data

  // use precomputed table, rather than computing it on the fly
  static const UInt_t crctab[256] = { 0x00000000,
    0x04c11db7, 0x09823b6e, 0x0d4326d9, 0x130476dc, 0x17c56b6b,
    0x1a864db2, 0x1e475005, 0x2608edb8, 0x22c9f00f, 0x2f8ad6d6,
    0x2b4bcb61, 0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd,
    0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9, 0x5f15adac,
    0x5bd4b01b, 0x569796c2, 0x52568b75, 0x6a1936c8, 0x6ed82b7f,
    0x639b0da6, 0x675a1011, 0x791d4014, 0x7ddc5da3, 0x709f7b7a,
    0x745e66cd, 0x9823b6e0, 0x9ce2ab57, 0x91a18d8e, 0x95609039,
    0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5, 0xbe2b5b58,
    0xbaea46ef, 0xb7a96036, 0xb3687d81, 0xad2f2d84, 0xa9ee3033,
    0xa4ad16ea, 0xa06c0b5d, 0xd4326d90, 0xd0f37027, 0xddb056fe,
    0xd9714b49, 0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
    0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1, 0xe13ef6f4,
    0xe5ffeb43, 0xe8bccd9a, 0xec7dd02d, 0x34867077, 0x30476dc0,
    0x3d044b19, 0x39c556ae, 0x278206ab, 0x23431b1c, 0x2e003dc5,
    0x2ac12072, 0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16,
    0x018aeb13, 0x054bf6a4, 0x0808d07d, 0x0cc9cdca, 0x7897ab07,
    0x7c56b6b0, 0x71159069, 0x75d48dde, 0x6b93dddb, 0x6f52c06c,
    0x6211e6b5, 0x66d0fb02, 0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1,
    0x53dc6066, 0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba,
    0xaca5c697, 0xa864db20, 0xa527fdf9, 0xa1e6e04e, 0xbfa1b04b,
    0xbb60adfc, 0xb6238b25, 0xb2e29692, 0x8aad2b2f, 0x8e6c3698,
    0x832f1041, 0x87ee0df6, 0x99a95df3, 0x9d684044, 0x902b669d,
    0x94ea7b2a, 0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e,
    0xf3b06b3b, 0xf771768c, 0xfa325055, 0xfef34de2, 0xc6bcf05f,
    0xc27dede8, 0xcf3ecb31, 0xcbffd686, 0xd5b88683, 0xd1799b34,
    0xdc3abded, 0xd8fba05a, 0x690ce0ee, 0x6dcdfd59, 0x608edb80,
    0x644fc637, 0x7a089632, 0x7ec98b85, 0x738aad5c, 0x774bb0eb,
    0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f, 0x5c007b8a,
    0x58c1663d, 0x558240e4, 0x51435d53, 0x251d3b9e, 0x21dc2629,
    0x2c9f00f0, 0x285e1d47, 0x36194d42, 0x32d850f5, 0x3f9b762c,
    0x3b5a6b9b, 0x0315d626, 0x07d4cb91, 0x0a97ed48, 0x0e56f0ff,
    0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623, 0xf12f560e,
    0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7, 0xe22b20d2, 0xe6ea3d65,
    0xeba91bbc, 0xef68060b, 0xd727bbb6, 0xd3e6a601, 0xdea580d8,
    0xda649d6f, 0xc423cd6a, 0xc0e2d0dd, 0xcda1f604, 0xc960ebb3,
    0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7, 0xae3afba2,
    0xaafbe615, 0xa7b8c0cc, 0xa379dd7b, 0x9b3660c6, 0x9ff77d71,
    0x92b45ba8, 0x9675461f, 0x8832161a, 0x8cf30bad, 0x81b02d74,
    0x857130c3, 0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
    0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c, 0x7b827d21,
    0x7f436096, 0x7200464f, 0x76c15bf8, 0x68860bfd, 0x6c47164a,
    0x61043093, 0x65c52d24, 0x119b4be9, 0x155a565e, 0x18197087,
    0x1cd86d30, 0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec,
    0x3793a651, 0x3352bbe6, 0x3e119d3f, 0x3ad08088, 0x2497d08d,
    0x2056cd3a, 0x2d15ebe3, 0x29d4f654, 0xc5a92679, 0xc1683bce,
    0xcc2b1d17, 0xc8ea00a0, 0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb,
    0xdbee767c, 0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18,
    0xf0a5bd1d, 0xf464a0aa, 0xf9278673, 0xfde69bc4, 0x89b8fd09,
    0x8d79e0be, 0x803ac667, 0x84fbdbd0, 0x9abc8bd5, 0x9e7d9662,
    0x933eb0bb, 0x97ffad0c, 0xafb010b1, 0xab710d06, 0xa6322bdf,
    0xa2f33668, 0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4
  };

  crc = ~crc;
  while (sz--) crc = (crc << 8) ^ UInt_t(*data++) ^ crctab[crc >> 24];

  return ~crc;
}

UInt_t RooAbsArg::fnv1a32(const char* data)
{
  // calculate 32 bit FNV1A hash of string
  return fnv1a32(data, strlen(data));
}

UInt_t RooAbsArg::fnv1a32(const char* data, ULong_t sz, UInt_t hash)
{
  // update 32 bit FNV1A hash
  const UInt_t fnv1a32mult = 16777619u;
  while (sz--) {
    hash ^= *data++;
    hash *= fnv1a32mult;
  }
  return hash;
}

ULong64_t RooAbsArg::fnv1a64(const char* data)
{
  // calculate 64 bit FNV1A hash of string
  return fnv1a64(data, strlen(data));
}

ULong64_t RooAbsArg::fnv1a64(const char* data, ULong_t sz, ULong64_t hash)
{
  // update 64 bit FNV1A hash
  const ULong64_t fnv1a64mult = (ULong64_t(1) << 40) | ULong64_t(435);
  while (sz--) {
    hash ^= *data++;
    hash *= fnv1a64mult;
  }
  return hash;
}

//_____________________________________________________________________________
void RooAbsArg::printCompactTreeHook(ostream&, const char *)
{
  // Hook function interface for object to insert additional information
  // when printed in the context of a tree structure. This default
  // implementation prints nothing
}


//_____________________________________________________________________________
void RooAbsArg::registerCache(RooAbsCache& cache)
{
  // Register RooAbsCache with this object. This function is called
  // by RooAbsCache constructors for objects that are a datamember
  // of this RooAbsArg. By registering itself the RooAbsArg is aware
  // of all its cache data members and will forward server change
  // and cache mode change calls to the cache objects, which in turn
  // can forward them their contents

  _cacheList.push_back(&cache) ;
}


//_____________________________________________________________________________
void RooAbsArg::unRegisterCache(RooAbsCache& cache)
{
  // Unregister a RooAbsCache. Called from the RooAbsCache destructor
  std::remove(_cacheList.begin(), _cacheList.end(), &cache);
}


//_____________________________________________________________________________
Int_t RooAbsArg::numCaches() const
{
  // Return number of registered caches

  return _cacheList.size() ;
}


//_____________________________________________________________________________
RooAbsCache* RooAbsArg::getCache(Int_t index) const
{
  // Return registered cache object by index

  return _cacheList[index] ;
}


//_____________________________________________________________________________
RooArgSet* RooAbsArg::getVariables(Bool_t stripDisconnected) const
{
  // Return RooArgSet with all variables (tree leaf nodes of expresssion tree)

  return getParameters(RooArgSet(),stripDisconnected) ;
}


//_____________________________________________________________________________
RooLinkedList RooAbsArg::getCloningAncestors() const
{
  // Return ancestors in cloning chain of this RooAbsArg. NOTE: Returned pointers
  // are not guaranteed to be 'live', so do not dereference without proper caution

  RooLinkedList retVal ;

  set<string>::const_iterator iter= _boolAttrib.begin() ;
  while(iter != _boolAttrib.end()) {
    if (TString(*iter).BeginsWith("CloneOf(")) {
      char buf[128] ;
      strlcpy(buf,iter->c_str(),128) ;
      strtok(buf,"(") ;
      char* ptrToken = strtok(0,")") ;
      RooAbsArg* ptr = (RooAbsArg*) strtol(ptrToken,0,16) ;
      retVal.Add(ptr) ;
    }
  }

  return retVal ;
}


//_____________________________________________________________________________
void RooAbsArg::graphVizTree(const char* fileName, const char* delimiter, bool useTitle, bool useLatex)
{
  // Create a GraphViz .dot file visualizing the expression tree headed by
  // this RooAbsArg object. Use the GraphViz tool suite to make e.g. a gif
  // or ps file from the .dot file
  //
  // Based on concept developed by Kyle Cranmer

  ofstream ofs(fileName) ;
  if (!ofs) {
    coutE(InputArguments) << "RooAbsArg::graphVizTree() ERROR: Cannot open graphViz output file with name " << fileName << endl ;
    return ;
  }
  graphVizTree(ofs, delimiter, useTitle, useLatex) ;
}

//_____________________________________________________________________________
void RooAbsArg::graphVizTree(ostream& os, const char* delimiter, bool useTitle, bool useLatex)
{
  // Write the GraphViz representation of the expression tree headed by
  // this RooAbsArg object to the given ostream.
  //
  // Based on concept developed by Kyle Cranmer

  if (!os) {
    coutE(InputArguments) << "RooAbsArg::graphVizTree() ERROR: output stream provided as input argument is in invalid state" << endl ;
  }

  // Write header
  os << "digraph " << GetName() << "{" << endl ;

  // First list all the tree nodes
  RooArgSet nodeSet ;
  treeNodeServerList(&nodeSet) ;
  RooFIter iter = nodeSet.fwdIterator() ;
  RooAbsArg* node ;

  // iterate over nodes
  while((node=iter.next())) {
    string nodeName = node->GetName();
    string nodeTitle = node->GetTitle();
    string nodeLabel = (useTitle && !nodeTitle.empty()) ? nodeTitle : nodeName;

    // if using latex, replace ROOT's # with normal latex backslash
    string::size_type position = nodeLabel.find("#") ;
    while(useLatex && position!=nodeLabel.npos){
      nodeLabel.replace(position, 1, "\\");
    }

    string typeFormat = "\\texttt{";
    string nodeType = (useLatex) ? typeFormat+node->IsA()->GetName()+"}" : node->IsA()->GetName();

    os << "\"" << nodeName << "\" [ color=" << (node->isFundamental()?"blue":"red")
       << ", label=\"" << nodeType << delimiter << nodeLabel << "\"];" << endl ;

  }

  // Get set of all server links
  set<pair<RooAbsArg*,RooAbsArg*> > links ;
  graphVizAddConnections(links) ;

  // And write them out
  set<pair<RooAbsArg*,RooAbsArg*> >::iterator liter = links.begin() ;
  for( ; liter != links.end() ; ++liter ) {
    os << "\"" << liter->first->GetName() << "\" -> \"" << liter->second->GetName() << "\";" << endl ;
  }

  // Write trailer
  os << "}" << endl ;

}

//_____________________________________________________________________________
void RooAbsArg::graphVizAddConnections(set<pair<RooAbsArg*,RooAbsArg*> >& linkSet)
{
  // Utility function that inserts all point-to-point client-server connections
  // between any two RooAbsArgs in the expression tree headed by this object
  // in the linkSet argument.

  RooFIter sIter = serverMIterator() ;
  RooAbsArg* server ;
  while((server=sIter.next())) {
    linkSet.insert(make_pair(this,server)) ;
    server->graphVizAddConnections(linkSet) ;
  }
}



// //_____________________________________________________________________________
// TGraphStruct* RooAbsArg::graph(Bool_t useFactoryTag, Double_t textSize)
// {
//   // Return a TGraphStruct object filled with the tree structure of the pdf object

//   TGraphStruct* theGraph = new TGraphStruct() ;

//   // First list all the tree nodes
//   RooArgSet nodeSet ;
//   treeNodeServerList(&nodeSet) ;
//   TIterator* iter = nodeSet.createIterator() ;
//   RooAbsArg* node ;


//   // iterate over nodes
//   while((node=(RooAbsArg*)iter->Next())) {

//     // Skip node that represent numeric constants
//     if (node->IsA()->InheritsFrom(RooConstVar::Class())) continue ;

//     string nodeName ;
//     if (useFactoryTag && node->getStringAttribute("factory_tag")) {
//       nodeName  = node->getStringAttribute("factory_tag") ;
//     } else {
//       if (node->isFundamental()) {
// 	nodeName = node->GetName();
//       } else {
// 	ostringstream oss ;
// 	node->printStream(oss,(node->defaultPrintContents(0)&(~kValue)),node->defaultPrintStyle(0)) ;
// 	nodeName= oss.str() ;
// // 	nodeName = Form("%s::%s",node->IsA()->GetName(),node->GetName());

//       }
//     }
//     if (strncmp(nodeName.c_str(),"Roo",3)==0) {
//       nodeName = string(nodeName.c_str()+3) ;
//     }
//     node->setStringAttribute("graph_name",nodeName.c_str()) ;

//     TGraphNode* gnode = theGraph->AddNode(nodeName.c_str(),nodeName.c_str()) ;
//     gnode->SetLineWidth(2) ;
//     gnode->SetTextColor(node->isFundamental()?kBlue:kRed) ;
//     gnode->SetTextSize(textSize) ;
//   }
//   delete iter ;

//   // Get set of all server links
//   set<pair<RooAbsArg*,RooAbsArg*> > links ;
//   graphVizAddConnections(links) ;

//   // And insert them into the graph
//   set<pair<RooAbsArg*,RooAbsArg*> >::iterator liter = links.begin() ;
//   for( ; liter != links.end() ; ++liter ) {

//     TGraphNode* n1 = (TGraphNode*)theGraph->GetListOfNodes()->FindObject(liter->first->getStringAttribute("graph_name")) ;
//     TGraphNode* n2 = (TGraphNode*)theGraph->GetListOfNodes()->FindObject(liter->second->getStringAttribute("graph_name")) ;
//     if (n1 && n2) {
//       TGraphEdge* edge = theGraph->AddEdge(n1,n2) ;
//       edge->SetLineWidth(2) ;
//     }
//   }

//   return theGraph ;
// }



// //_____________________________________________________________________________
// Bool_t RooAbsArg::inhibitDirty()
// {
//   // Return current status of the inhibitDirty global flag. If true
//   // no dirty state change tracking occurs and all caches are considered
//   // to be always dirty.
//   return _inhibitDirty ;
// }


//_____________________________________________________________________________
Bool_t RooAbsArg::addOwnedComponents(const RooArgSet& comps)
{
  // Take ownership of the contents of 'comps'

  if (!_ownedComponents) {
    _ownedComponents = new RooArgSet("owned components") ;
  }
  return _ownedComponents->addOwned(comps) ;
}



//_____________________________________________________________________________
RooAbsArg* RooAbsArg::cloneTree(const char* newname) const
{
  // Clone tree expression of objects. All tree nodes will be owned by
  // the head node return by cloneTree()

  // Clone tree using snapshot
  RooArgSet* clonedNodes = (RooArgSet*) RooArgSet(*this).snapshot(kTRUE) ;

  // Find the head node in the cloneSet
  RooAbsArg* head = clonedNodes->find(*this) ;

  // Remove the head node from the cloneSet
  // To release it from the set ownership
  clonedNodes->remove(*head) ;

  // Add the set as owned component of the head
  head->addOwnedComponents(*clonedNodes) ;

  // Delete intermediate container
  clonedNodes->releaseOwnership() ;
  delete clonedNodes ;

  // Adjust name of head node if requested
  if (newname) {
    head->TNamed::SetName(newname) ;
    head->_namePtr = (TNamed*) RooNameReg::instance().constPtr(newname) ;
  }

  // Return the head
  return head ;
}



//_____________________________________________________________________________
void RooAbsArg::attachToStore(RooAbsDataStore& store) 
{
  if (dynamic_cast<RooTreeDataStore*>(&store)) {
    attachToTree(((RooTreeDataStore&)store).tree()) ;
  } else if (dynamic_cast<RooVectorDataStore*>(&store)) {
    attachToVStore((RooVectorDataStore&)store) ;
  }
}



//_____________________________________________________________________________
RooExpensiveObjectCache& RooAbsArg::expensiveObjectCache() const
{
  if (_eocache) {
    return *_eocache ;
  } else {
    return RooExpensiveObjectCache::instance() ;
  }
}


//_____________________________________________________________________________
const char* RooAbsArg::aggregateCacheUniqueSuffix() const
{
  string suffix ;

  RooArgSet branches ;
  branchNodeServerList(&branches) ;
  RooFIter iter = branches.fwdIterator();
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    const char* tmp = arg->cacheUniqueSuffix() ;
    if (tmp) suffix += tmp ;
  }
  return Form("%s",suffix.c_str()) ;
}


//_____________________________________________________________________________
void RooAbsArg::wireAllCaches() 
{
  RooArgSet branches ;
  branchNodeServerList(&branches) ;
  RooFIter iter = branches.fwdIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
//     cout << "wiring caches on node " << arg->GetName() << endl ;
    for (deque<RooAbsCache*>::iterator iter2 = arg->_cacheList.begin() ; iter2 != arg->_cacheList.end() ; ++iter2) {
      (*iter2)->wireCache() ;
    }
  }
}



//_____________________________________________________________________________
void RooAbsArg::SetName(const char* name) 
{
  TNamed::SetName(name) ;
  TNamed* newPtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;
  if (newPtr != _namePtr) {
    //cout << "Rename '" << _namePtr->GetName() << "' to '" << name << "' (set flag in new name)" << endl;
    _namePtr = newPtr;
    _namePtr->SetBit(RooNameReg::kRenamedArg);
  }
}




//_____________________________________________________________________________
void RooAbsArg::SetNameTitle(const char *name, const char *title)
{
  TNamed::SetNameTitle(name,title) ;
  TNamed* newPtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;
  if (newPtr != _namePtr) {
    //cout << "Rename '" << _namePtr->GetName() << "' to '" << name << "' (set flag in new name)" << endl;
    _namePtr = newPtr;
    _namePtr->SetBit(RooNameReg::kRenamedArg);
  }
}


//______________________________________________________________________________
void RooAbsArg::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooAbsArg.

   if (R__b.IsReading()) {
     _ioReadStack.push(this) ;
     R__b.ReadClassBuffer(RooAbsArg::Class(),this);
     _ioReadStack.pop() ;
     _namePtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;  
     _isConstant = getAttribute("Constant") ;
   } else {
     R__b.WriteClassBuffer(RooAbsArg::Class(),this);
   }
}

//______________________________________________________________________________
void RooAbsArg::ioStreamerPass2() 
{
  // Method called by workspace container to finalize schema evolution issues
  // that cannot be handled in a single ioStreamer pass.
  //
  // A second pass is typically needed when evolving data member of RooAbsArg-derived
  // classes that are container classes with references to other members, which may
  // not yet be 'live' in the first ioStreamer() evolution pass.
  //
  // Classes may overload this function, but must call the base method in the
  // overloaded call to ensure base evolution is handled properly


  // Handling of v5-v6 migration (TRefArray _proxyList --> RooRefArray _proxyList)
  map<RooAbsArg*,TRefArray*>::iterator iter = _ioEvoList.find(this) ;
  if (iter != _ioEvoList.end()) {

    // Transfer contents of saved TRefArray to RooRefArray now
    for (int i=0 ; i < iter->second->GetEntries() ; i++) {
      _proxyList.Add(iter->second->At(i)) ;
    }
    // Delete TRefArray and remove from list
    delete iter->second ;
    _ioEvoList.erase(iter) ;
  }
}




//______________________________________________________________________________
void RooAbsArg::ioStreamerPass2Finalize() 
{
  // Method called by workspace container to finalize schema evolution issues
  // that cannot be handled in a single ioStreamer pass. This static finalize method
  // is called after ioStreamerPass2() is called on each directly listed object 
  // in the workspace. It's purpose is to complete schema evolution of any
  // objects in the workspace that are not directly listed as content elements
  // (e.g. analytical convolution tokens )
  

  // Handling of v5-v6 migration (TRefArray _proxyList --> RooRefArray _proxyList)
  map<RooAbsArg*,TRefArray*>::iterator iter = _ioEvoList.begin() ;
  while (iter != _ioEvoList.end()) {
    
    // Transfer contents of saved TRefArray to RooRefArray now
    for (int i=0 ; i < iter->second->GetEntries() ; i++) {
      iter->first->_proxyList.Add(iter->second->At(i)) ;
    }

    // Save iterator position for deletion after increment
    map<RooAbsArg*,TRefArray*>::iterator iter_tmp = iter ;
    
    iter++ ;
    
    // Delete TRefArray and remove from list
    delete iter_tmp->second ;
    _ioEvoList.erase(iter_tmp) ;
    
  }

}


//______________________________________________________________________________
void RooRefArray::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooRefArray.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {

      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }      

      // Make temporary refArray and read that from the streamer 
      TRefArray* refArray = new TRefArray ;
      refArray->Streamer(R__b) ;
      R__b.CheckByteCount(R__s, R__c, refArray->IsA());
      
      // Schedule deferred processing of TRefArray into proxy list
      RooAbsArg::_ioEvoList[RooAbsArg::_ioReadStack.top()] = refArray ;
      
   } else {

     R__c = R__b.WriteVersion(RooRefArray::IsA(), kTRUE);

     // Make a temporary refArray and write that to the streamer
     TRefArray refArray ;
     TIterator* iter = MakeIterator() ; 
     TObject* tmpObj ; while ((tmpObj = iter->Next())) { 
       refArray.Add(tmpObj) ; 
     } 
     delete iter ; 

     refArray.Streamer(R__b) ;
     R__b.SetByteCount(R__c, kTRUE) ;
     
   }
}


