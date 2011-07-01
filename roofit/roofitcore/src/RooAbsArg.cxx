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
#include "RooMsgService.h"
#include "RooExpensiveObjectCache.h"
#include "RooAbsDataStore.h"
#include "RooResolutionModel.h"

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
Bool_t RooAbsArg::_flipAClean(kFALSE) ;

//_____________________________________________________________________________
RooAbsArg::RooAbsArg() :
  TNamed(),
  _deleteWatch(kFALSE),
  _operMode(Auto),
  _ownedComponents(0),
  _prohibitServerRedirect(kFALSE),
  _eocache(0)
{
  // Default constructor

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

  RooTrace::create(this) ;
}

//_____________________________________________________________________________
RooAbsArg::RooAbsArg(const char *name, const char *title) :
  TNamed(name,title),
  _deleteWatch(kFALSE),
  _valueDirty(kTRUE),
  _shapeDirty(kTRUE),
  _operMode(Auto),
  _ownedComponents(0),
  _prohibitServerRedirect(kFALSE),
  _eocache(0)
{
  // Create an object with the specified name and descriptive title.
  // The newly created object has no clients or servers and has its
  // dirty flags set.

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;
  RooTrace::create(this) ;

}

//_____________________________________________________________________________
RooAbsArg::RooAbsArg(const RooAbsArg& other, const char* name)
  : TNamed(other.GetName(),other.GetTitle()),
    RooPrintable(other),
    _boolAttrib(other._boolAttrib),
    _stringAttrib(other._stringAttrib),
    _deleteWatch(other._deleteWatch),
    _operMode(Auto),
    _ownedComponents(0),
    _prohibitServerRedirect(kFALSE),
    _eocache(other._eocache)
{
  // Copy constructor transfers all boolean and string properties of the original
  // object. Transient properties and client-server links are not copied

  // Use name in argument, if supplied
  if (name) SetName(name) ;

  // Copy server list by hand
  TIterator* sIter = other._serverList.MakeIterator() ;
  RooAbsArg* server ;
  Bool_t valueProp, shapeProp ;
  while ((server = (RooAbsArg*) sIter->Next())) {
    valueProp = server->_clientListValue.FindObject((TObject*)&other)?kTRUE:kFALSE ;
    shapeProp = server->_clientListShape.FindObject((TObject*)&other)?kTRUE:kFALSE ;
    addServer(*server,valueProp,shapeProp) ;
  }
  delete sIter ;

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

  setValueDirty() ;
  setShapeDirty() ;

  //setAttribute(Form("CloneOf(%08x)",&other)) ;
  //cout << "RooAbsArg::cctor(" << this << ") #bools = " << _boolAttrib.size() << " #strings = " << _stringAttrib.size() << endl ;

  RooTrace::create(this) ;
}


//_____________________________________________________________________________
RooAbsArg::~RooAbsArg()
{
  // Destructor.

  // Notify all servers that they no longer need to serve us
  TIterator* serverIter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while ((server=(RooAbsArg*)serverIter->Next())) {
    removeServer(*server,kTRUE) ;
  }
  delete serverIter ;

  //Notify all client that they are in limbo
  TIterator* clientIter = _clientList.MakeIterator() ;
  RooAbsArg* client = 0;
  Bool_t first(kTRUE) ;
  while ((client=(RooAbsArg*)clientIter->Next())) {
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
  delete clientIter ;

  delete _clientShapeIter ;
  delete _clientValueIter ;

  if (_ownedComponents) {
    delete _ownedComponents ;
    _ownedComponents = 0 ;
  }

  RooTrace::destroy(this) ;
}


//_____________________________________________________________________________
void RooAbsArg::setDirtyInhibit(Bool_t flag)
{
  // Control global dirty inhibit mode. When set to true no value or shape dirty
  // flags are propagated and cache is always considered to be dirty.
  _inhibitDirty = flag ;
}


//_____________________________________________________________________________
void RooAbsArg::setACleanADirty(Bool_t flag)
{
  // This global switch changes the cache mode of all objects marked as 'always clean'
  // to 'always dirty'. For internal use in RooRealIntegral
  _flipAClean = flag ;
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
  TIterator* iter = serverList.createIterator() ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    addServer(*arg,valueProp,shapeProp) ;
  }
  delete iter ;
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

  Int_t count = _serverList.refCount(&oldServer) ;
  removeServer(oldServer,kTRUE) ;
  while(count--) {
    addServer(newServer,propValue,propShape) ;
  }
}


//_____________________________________________________________________________
void RooAbsArg::changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp)
{
  // Change dirty flag propagation mask for specified server

  if (!_serverList.FindObject(&server)) {
    coutE(LinkStateMgmt) << "RooAbsArg::changeServer(" << GetName() << "): Server "
	 << server.GetName() << " not registered" << endl ;
    return ;
  }

  // This condition should not happen, but check anyway
  if (!server._clientList.FindObject(this)) {
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
    while (vcount--) server._clientListValue.Add(this) ;
  }
  if (shapeProp) {
    while(scount--) server._clientListShape.Add(this) ;
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

  if (!arg) {
//     if (list->getHashTableSize()==0) {
//       list->setHashTableSize(1000) ;
//     }
    arg=this ;
  }

  // Decide if to add current node
  if ((doBranch&&doLeaf) ||
      (doBranch&&arg->isDerived()) ||
      (doLeaf&&arg->isFundamental()&&(!(recurseFundamental&&arg->isDerived()))) ) {
    list->add(*arg,kTRUE) ;
  }

  // Recurse if current node is derived
  if (arg->isDerived() && (!arg->isFundamental() || recurseFundamental)) {
    RooAbsArg* server ;
    TIterator* sIter = arg->serverIterator() ;
    while ((server=(RooAbsArg*)sIter->Next())) {

      // Skip non-value server nodes if requested
      Bool_t isValueSrv = server->_clientListValue.FindObject((TObject*)arg)?kTRUE:kFALSE ;
      if (valueOnly && !isValueSrv) {
	continue ;
      }
      treeNodeServerList(list,server,doBranch,doLeaf,valueOnly,recurseFundamental) ;
    }
    delete sIter ;
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

  TIterator* siter = serverIterator() ;
  RooAbsArg* server ;

  RooArgSet nodeParamServers ;
  RooArgSet nodeBranchServers ;
  while((server=(RooAbsArg*)siter->Next())) {
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
  delete siter ;

  // Allow pdf to strip parameters from list before adding it
  getParametersHook(nset,&nodeParamServers,stripDisconnected) ;

  // Add parameters of this node to the combined list
  params.add(nodeParamServers,kTRUE) ;

  // Now recurse into branch servers
  TIterator* biter = nodeBranchServers.createIterator() ;
  while((server=(RooAbsArg*)biter->Next())) {
    server->addParameters(params,nset) ;
  }
  delete biter ;
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
  RooLinkedListIter sIter = leafList.iterator() ;

  RooAbsArg* arg ;
  if (valueOnly) {
    while ((arg=(RooAbsArg*)sIter.Next())) {
      if (arg->dependsOnValue(*dataList) && arg->isLValue()) {
	depList->add(*arg) ;
      }
    }
  } else {
    while ((arg=(RooAbsArg*)sIter.Next())) {
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
  TIterator* iter = nodeList.createIterator() ;

  RooAbsArg* arg ;
  Bool_t ret(kFALSE) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (arg->getAttribute("ServerDied")) {
      coutE(LinkStateMgmt) << "RooAbsArg::recursiveCheckObservables(" << GetName() << "): ERROR: one or more servers of node "
			   << arg->GetName() << " no longer exists!" << endl ;
      arg->Print("v") ;
      ret = kTRUE ;
    }
    ret |= arg->checkObservables(nset) ;
  }
  delete iter ;

  return ret ;
}


//_____________________________________________________________________________
Bool_t RooAbsArg::dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg, Bool_t valueOnly) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  Bool_t result(kFALSE);
  TIterator* sIter = serverList.createIterator();
  RooAbsArg* server ;
  while ((!result && (server=(RooAbsArg*)sIter->Next()))) {
    if (dependsOn(*server,ignoreArg,valueOnly)) {
      result= kTRUE;
    }
  }
  delete sIter;
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
  if (!TString(testArg.GetName()).CompareTo(GetName())) return kTRUE ;


  // Next test direct dependence
  RooAbsArg* server = findServer(testArg.GetName()) ;
  if (server!=0) {

    // Return true if valueOnly is FALSE or if server is value server, otherwise keep looking
    if ( !valueOnly || server->isValueServer(GetName())) {
      return kTRUE ;
    }
  }

  // If not, recurse
  TIterator* sIter = serverIterator() ;
  while ((server=(RooAbsArg*)sIter->Next())) {

    if ( !valueOnly || server->isValueServer(GetName())) {
      if (server->dependsOn(testArg,ignoreArg,valueOnly)) {
	delete sIter ;
	return kTRUE ;
      }
    }
  }

  delete sIter ;
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

  _clientValueIter->Reset() ;
  RooAbsArg* client ;
  while ((client=(RooAbsArg*)_clientValueIter->Next())) {
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

  _clientShapeIter->Reset() ;
  RooAbsArg* client ;
  while ((client=(RooAbsArg*)_clientShapeIter->Next())) {
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
    TIterator* iter = newSetOrig.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {

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
  THashList origServerList, origServerValue, origServerShape ;
  RooAbsArg *oldServer, *newServer ;
  TIterator* sIter = _serverList.MakeIterator() ;
  while ((oldServer=(RooAbsArg*)sIter->Next())) {
    origServerList.Add(oldServer) ;

    // Retrieve server side link state information
    if (oldServer->_clientListValue.FindObject(this)) {
      origServerValue.Add(oldServer) ;
    }
    if (oldServer->_clientListShape.FindObject(this)) {
      origServerShape.Add(oldServer) ;
    }
  }
  delete sIter ;

  // Delete all previously registered servers
  sIter = origServerList.MakeIterator() ;
  Bool_t propValue, propShape ;
  while ((oldServer=(RooAbsArg*)sIter->Next())) {

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
    
    propValue=origServerValue.FindObject(oldServer)?kTRUE:kFALSE ;
    propShape=origServerShape.FindObject(oldServer)?kTRUE:kFALSE ;
    // cout << "replaceServer with name " << oldServer->GetName() << " old=" << oldServer << " new=" << newServer << endl ;
    if (newServer != this) {
      replaceServer(*oldServer,*newServer,propValue,propShape) ;
    }
  }

  delete sIter ;

  setValueDirty() ;
  setShapeDirty() ;

  // Take self out of newset disallowing cyclical dependencies
  RooAbsCollection* newSet2 = (RooAbsCollection*) newSet->clone("newSet2") ;
  newSet2->remove(*this,kTRUE,kTRUE) ;

  // Process the proxies
  Bool_t allReplaced=kTRUE ;
  for (int i=0 ; i<numProxies() ; i++) {
    // WVE: Need to make exception here too for newServer != this
    Bool_t ret2 = getProxy(i)->changePointer(*newSet2,nameChange) ;
    allReplaced &= ret2 ;
  }

  delete newSet2 ;

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
    newServer = newSet.find(GetName()) ;
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
  if (callStack.FindObject(this)) {
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
  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)sIter->Next())) {
    ret |= server->recursiveRedirectServers(newSet,mustReplaceAll,nameChange,recurseInNewSet) ;
  }
  delete sIter ;

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
  _proxyList.Add(&proxy) ;
}



//_____________________________________________________________________________
void RooAbsArg::unRegisterProxy(RooListProxy& proxy)
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
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
void RooAbsArg::copyList(TList& dest, const TList& source)
{
  // WVE (08/21/01) Probably obsolete now
  dest.Clear() ;

  TIterator* sIter = source.MakeIterator() ;
  TObject* obj ;
  while ((obj = sIter->Next())) {
    dest.Add(obj) ;
  }
  delete sIter ;
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
  TIterator *clientIter= _clientList.MakeIterator();
  RooAbsArg* client ;
  while ((client=(RooAbsArg*)clientIter->Next())) {
    os << indent << "    (" << (void*)client  << ","
       << (_clientListValue.FindObject(client)?"V":"-")
       << (_clientListShape.FindObject(client)?"S":"-")
       << ") " ;
    client->printStream(os,kClassName|kTitle|kName,kSingleLine);
  }
  delete clientIter;

  // server list
  os << indent << "  Servers: " << endl;
  TIterator *serverIter= _serverList.MakeIterator();
  RooAbsArg* server ;
  while ((server=(RooAbsArg*)serverIter->Next())) {
    os << indent << "    (" << (void*)server << ","
       << (server->_clientListValue.FindObject((TObject*)this)?"V":"-")
       << (server->_clientListShape.FindObject((TObject*)this)?"S":"-")
       << ") " ;
    server->printStream(os,kClassName|kName|kTitle,kSingleLine);
  }
  delete serverIter;

  // proxy list
  os << indent << "  Proxies: " << endl ;
  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* proxy=getProxy(i) ;

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

  TIterator* iter = branches.createIterator() ;
  RooAbsArg* branch ;
  while((branch=(RooAbsArg*)iter->Next())) {
    branch->redirectServers(*set,kFALSE,kFALSE) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
void RooAbsArg::attachDataStore(const RooAbsDataStore &dstore)
{
  // Replace server nodes with names matching the dataset variable names
  // with those data set variables, making this PDF directly dependent on the dataset

  const RooArgSet* set = dstore.get() ;
  RooArgSet branches ;
  branchNodeServerList(&branches,0,kTRUE) ;

  TIterator* iter = branches.createIterator() ;
  RooAbsArg* branch ;
  while((branch=(RooAbsArg*)iter->Next())) {
    branch->redirectServers(*set,kFALSE,kFALSE) ;
  }
  delete iter ;
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
    TIterator* bIter = branchList.createIterator() ;
    RooAbsArg* branch ;
    while((branch=(RooAbsArg*)bIter->Next())) {
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
  if (processedNodes.FindObject(this)) {
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
  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)sIter->Next())) {
    server->optimizeCacheMode(observables,optimizedNodes,processedNodes) ;
  }
  delete sIter ;

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
  if (processedNodes.FindObject(this)) {
    return kFALSE ;
  } else {
    processedNodes.Add(this) ;
  }

  // Check if node depends on any non-constant parameter
  Bool_t canOpt(kTRUE) ;
  RooArgSet* paramSet = getParameters(observables) ;
  TIterator* iter = paramSet->createIterator() ;
  RooAbsArg* param ;
  while((param = (RooAbsArg*)iter->Next())) {
    if (!param->isConstant()) {
      canOpt=kFALSE ;
      break ;
    }
  }
  delete iter ;
  delete paramSet ;

  // If yes, list node eligible for caching, if not test nodes one level down
  if (canOpt) {

    if (!cacheList.find(GetName()) && dependsOnValue(observables) && !observables.find(GetName()) ) {

      // Add to cache list
      cxcoutD(Optimization) << "RooAbsArg::findConstantNodes(" << GetName() << ") adding self to list of constant nodes" << endl ;

      cacheList.add(*this,kFALSE) ;
    }

  } else {

    // If not, see if next level down can be cached
    TIterator* sIter = serverIterator() ;
    RooAbsArg* server ;
    while((server=(RooAbsArg*)sIter->Next())) {
      if (server->isDerived()) {
	server->findConstantNodes(observables,cacheList,processedNodes) ;
      }
    }
    delete sIter ;
  }

  // Forward call to all cached contained in current object
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->findConstantNodes(observables,cacheList,processedNodes) ;
  }

  return kFALSE ;
}




//_____________________________________________________________________________
void RooAbsArg::constOptimizeTestStatistic(ConstOpCode opcode)
{
  // Interface function signaling a request to perform constant term
  // optimization. This default implementation takes no action other than to
  // forward the calls to all servers

  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)sIter->Next())) {
    server->constOptimizeTestStatistic(opcode) ;
  }
  delete sIter ;
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
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->operModeHook() ;
  }
  operModeHook() ;

  // Propagate to all clients
  if (mode==ADirty && recurseADirty) {
    TIterator* iter = valueClientIterator() ;
    RooAbsArg* client ;
    while((client=(RooAbsArg*)iter->Next())) {
      client->setOperMode(mode) ;
    }
    delete iter ;
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
      case Auto:   os << " [Auto] "  ; break ;
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
  TIterator * iter = serverIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    arg->printCompactTree(os,indent2,namePat,this) ;
  }
  delete iter ;
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
  TIterator * iter = serverIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    arg->printComponentTree(indent2.Data(),namePat,nLevel-1) ;
  }
  delete iter ;
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

  // Calculate and extract length of string
  Int_t len = strlen(data) ;
  if (len<4) {
    oocoutE((RooAbsArg*)0,InputArguments) << "RooAbsReal::crc32 cannot calculate checksum of less than 4 bytes of data" << endl ;
    return 0 ;
  }

  // Initialize CRC table on first use
  static Bool_t init(kFALSE) ;
  static unsigned int crctab[256];
  if (!init) {
    int i, j;
    unsigned int crc;
    for (i = 0; i < 256; i++){
      crc = i << 24;
      for (j = 0; j < 8; j++) {
	if (crc & 0x80000000) {
	  crc = (crc << 1) ^ 0x04c11db7 ;
	} else {
	  crc = crc << 1;
	}
      }
      crctab[i] = crc;
    }
    init = kTRUE ;
  }

  unsigned int        result(0);
  int                 i(0);

  result = *data++ << 24;
  result |= *data++ << 16;
  result |= *data++ << 8;
  result |= *data++;
  result = ~ result;
  len -=4;

  for (i=0; i<len; i++) {
    result = (result << 8 | *data++) ^ crctab[result >> 24];
  }

  return ~result;
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
  TIterator* iter = nodeSet.createIterator() ;
  RooAbsArg* node ;

  // iterate over nodes
  while((node=(RooAbsArg*)iter->Next())) {
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
  delete iter ;

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

  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)sIter->Next())) {
    linkSet.insert(make_pair(this,server)) ;
    server->graphVizAddConnections(linkSet) ;
  }
  delete sIter ;
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



//_____________________________________________________________________________
Bool_t RooAbsArg::inhibitDirty()
{
  // Return current status of the inhibitDirty global flag. If true
  // no dirty state change tracking occurs and all caches are considered
  // to be always dirty.
  return _inhibitDirty ;
}


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
  RooAbsArg* head = clonedNodes->find(GetName()) ;

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
    head->SetName(newname) ;
  }

  // Return the head
  return head ;
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
Bool_t RooAbsArg::flipAClean()
{
  return _flipAClean ;
}



//_____________________________________________________________________________
const char* RooAbsArg::aggregateCacheUniqueSuffix() const
{
  string suffix ;

  RooArgSet branches ;
  branchNodeServerList(&branches) ;
  TIterator* iter = branches.createIterator( );
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    const char* tmp = arg->cacheUniqueSuffix() ;
    if (tmp) suffix += tmp ;
  }
  delete iter ;
  return Form("%s",suffix.c_str()) ;
}


