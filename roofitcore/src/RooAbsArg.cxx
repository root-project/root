/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.cc,v 1.61 2001/10/27 22:28:17 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooAbsArg is the common abstract base class for objects that represent a
// value (of arbitrary type) and "shape" that in general depends on (is a client of)
// other RooAbsArg subclasses. The only state information about a value that
// is maintained in this base class consists of named attributes and flags
// that track when either the value or the shape of this object changes. The
// meaning of shape depends on the client implementation but could be, for
// example, the allowed range of a value. The base class is also responsible
// for managing client/server links and propagating value/shape changes
// through an expression tree.
//
// RooAbsArg implements public interfaces for inspecting client/server
// relationships and setting/clearing/testing named attributes. The class
// also defines a pure virtual public interface for I/O streaming.

#include "TObjString.h"

#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooSetProxy.hh"
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooTrace.hh"
#include "RooFitCore/RooStringVar.hh" 

#include <string.h>
#include <iomanip.h>

ClassImp(RooAbsArg)
;

Bool_t RooAbsArg::_verboseDirty(kFALSE) ;
Int_t  RooAbsArg::_nameLength(0) ;

RooAbsArg::RooAbsArg() : TNamed(), _attribList(),
  _operMode(Auto)
{
  // Default constructor creates an unnamed object. At present this
  // will trigger an assert(0) because may indicate an attempt to
  // recreate a RooAbsArg from a root stream buffer, which is not
  // supported yet.

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;
//   cout << "RooAbsArg::ctor WARNING: default ctor called" << endl ;
//   assert(0) ;

  RooTrace::create(this) ;
}

RooAbsArg::RooAbsArg(const char *name, const char *title)
  : TNamed(name,title), _valueDirty(kTRUE), _shapeDirty(kTRUE),
  _operMode(Auto)
{    
  // Create an object with the specified name and descriptive title.
  // The newly created object has no clients or servers and has its
  // dirty flags set.

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;
  RooTrace::create(this) ;
}

RooAbsArg::RooAbsArg(const RooAbsArg& other, const char* name)
  : TNamed(other.GetName(),other.GetTitle()), _operMode(Auto)
{
  // Copy constructor transfers all properties of the original
  // object, except for its list of clients. The newly created 
  // object has an empty client list and has its dirty
  // flags set.

  // Use name in argument, if supplied
  if (name) SetName(name) ;

  // take attributes from target
  TObject* obj ;
  TIterator* aIter = other._attribList.MakeIterator() ;
  while (obj=aIter->Next()) {
    _attribList.Add(new TObjString(obj->GetName())) ;
  }
  delete aIter ;

  // Copy server list by hand
  TIterator* sIter = other._serverList.MakeIterator() ;
  RooAbsArg* server ;
  Bool_t valueProp, shapeProp ;
  while (server = (RooAbsArg*) sIter->Next()) {
    valueProp = server->_clientListValue.FindObject((TObject*)&other)?kTRUE:kFALSE ;
    shapeProp = server->_clientListShape.FindObject((TObject*)&other)?kTRUE:kFALSE ;
    addServer(*server,valueProp,shapeProp) ;
  }
  delete sIter ;

  _clientShapeIter = _clientListShape.MakeIterator() ;
  _clientValueIter = _clientListValue.MakeIterator() ;

  setValueDirty() ;
  setShapeDirty() ;

  RooTrace::create(this) ;
}


RooAbsArg::~RooAbsArg() 
{
  // Destructor notifies its servers that they no longer need to serve us and
  // notifies its clients that they are now in limbo (!)

  // Notify all servers that they no longer need to serve us
  TIterator* serverIter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)serverIter->Next()) {
    removeServer(*server) ;
  }
  delete serverIter ;

  //Notify all client that they are in limbo
  TIterator* clientIter = _clientList.MakeIterator() ;
  RooAbsArg* client(0);
  while (client=(RooAbsArg*)clientIter->Next()) {
    TString attr("ServerDied:");
    attr.Append(GetName());
    client->setAttribute(attr.Data());
    client->removeServer(*this);
    if (_verboseDirty) {
      cout << fName << "::" << ClassName() << ":~RooAbsArg: dependent \""
	   << client->GetName() << "\" should have been deleted first" << endl ;
    }
  }
  delete clientIter ;

  _attribList.Delete() ;
  

  delete _clientShapeIter ;
  delete _clientValueIter ;

  RooTrace::destroy(this) ;
}



void RooAbsArg::setAttribute(const Text_t* name, Bool_t value) 
{
  // Set (default) or clear a named boolean attribute of this object.

  TObject* oldAttrib = _attribList.FindObject(name) ;

  if (value) {
    // Add string to attribute list, if not already there
    if (!oldAttrib) {
      TObjString* nameObj = new TObjString(name) ;
      _attribList.Add(nameObj) ;
    }
  } else {
    // Remove string from attribute list, if found
    if (oldAttrib) {
      _attribList.Remove(oldAttrib) ;
    }
  }
}


Bool_t RooAbsArg::getAttribute(const Text_t* name) const
{
  // Check if a named attribute is set. By default, all attributes
  // are unset.
  return _attribList.FindObject(name)?kTRUE:kFALSE ;
}


void RooAbsArg::addServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) 
{
  // Register another RooAbsArg as a server to us, ie, declare that
  // we depend on it. In addition to the basic client-server relationship,
  // we can declare dependence on the server's value and/or shape.

  if (_verboseDirty) {
       cout << "RooAbsArg::addServer(" << GetName() << "): adding server " << server.GetName() 
	    << "(" << &server << ") for " << (valueProp?"value ":"") << (shapeProp?"shape":"") << endl ;
  }

  // Add server link to given server
  if (!_serverList.FindObject(&server)) {
    _serverList.Add(&server) ;
  } else {
//     cout << "RooAbsArg::addServer(" << GetName() << "): Server " 
// 	 << server.GetName() << " already registered" << endl ;
    return ;
  }

  if (!server._clientList.FindObject(this)) {    
    server._clientList.Add(this) ;
    if (valueProp) server._clientListValue.Add(this) ;
    if (shapeProp) server._clientListShape.Add(this) ;
  } else {
    cout << "RooAbsArg::addServer(" << GetName() 
	 << "): Already registered as client of " << server.GetName() << endl ;
  }
} 



void RooAbsArg::addServerList(RooAbsCollection& serverList, Bool_t valueProp, Bool_t shapeProp) 
{
  // Register a list of RooAbsArg as servers to us by calls addServer() for each
  // arg in the list
  RooAbsArg* arg ;
  TIterator* iter = serverList.createIterator() ;
  while (arg=(RooAbsArg*)iter->Next()) {
    addServer(*arg,valueProp,shapeProp) ;
  }
  delete iter ;
}



void RooAbsArg::removeServer(RooAbsArg& server) 
{
  // Unregister another RooAbsArg as a server to us, ie, declare that
  // we no longer depend on its value and shape.

  if (_verboseDirty) {
    cout << "RooAbsArg::removeServer(" << GetName() << "): removing server " 
	 << server.GetName() << "(" << &server << ")" << endl ;
  }

  // Remove server link to given server
  if (_serverList.FindObject(&server)) {
    _serverList.Remove(&server) ;
  } else {
    cout << fName << "::" << ClassName() << "(" << this << "):removeServer: Server "
	 << server.GetName() << "(" << &server << ") is not registered" << endl;
    return ;
  }

  if (server._clientList.FindObject(this)) {
    server._clientList.Remove(this) ;
    server._clientListValue.Remove(this) ;
    server._clientListShape.Remove(this) ;
  } else {
    cout << "RooAbsArg::removeServer(" << GetName() 
	 << "): Never registered as client of " << server.GetName() << endl ;
  }
} 



void RooAbsArg::changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp)
{
  // Change dirty flag propagation mask for specified server

  if (!_serverList.FindObject(&server)) {
    cout << "RooAbsArg::changeServer(" << GetName() << "): Server " 
	 << server.GetName() << " not registered" << endl ;
    return ;
  }

  // This condition should not happen, but check anyway
  if (!server._clientList.FindObject(this)) {    
    cout << "RooAbsArg::changeServer(" << GetName() << "): Server " 
	 << server.GetName() << " doesn't have us registered as client" << endl ;
    return ;
  }

  // Remove all propagation links, then reinstall requested ones ;
  server._clientListValue.Remove(this) ;
  server._clientListShape.Remove(this) ;
  if (valueProp) server._clientListValue.Add(this) ;
  if (shapeProp) server._clientListShape.Add(this) ;
}



void RooAbsArg::leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg) const
{
  // Fill supplied list with all leaf nodes of the arg tree, starting with
  // ourself as top node. A leaf node is node that has no servers declared.

  treeNodeServerList(list,arg,kFALSE,kTRUE) ;
}



void RooAbsArg::branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg) const 
{
  // Fill supplied list with all branch nodes of the arg tree starting with 
  // ourself as top node. A branch node is node that has one or more servers declared.

  treeNodeServerList(list,arg,kTRUE,kFALSE) ;
}


void RooAbsArg::treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, Bool_t doBranch, Bool_t doLeaf) const
{
  // Fill supplied list with nodes of the arg tree, following all server links, 
  // starting with ourself as top node.

  if (!arg) arg=this ;

  // Decide if to add current node
  if ((doBranch&&doLeaf) ||
      (doBranch&&arg->isDerived()) ||
      (doLeaf&&!arg->isDerived())) {
    list->add(*arg,kTRUE) ;  
  }

  // Recurse if current node is derived
  if (arg->isDerived()) {
    RooAbsArg* server ;
    TIterator* sIter = arg->serverIterator() ;
    while (server=(RooAbsArg*)sIter->Next()) {
      treeNodeServerList(list,server,doBranch,doLeaf) ;
    }  
    delete sIter ;
  }
}


RooArgSet* RooAbsArg::getParameters(const RooAbsData* set) const 
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that don't match any of the names of the variable list
  // of the supplied data set (the dependents). The caller of this
  // function is responsible for deleting the returned argset.
  // The complement of this function is getDependents()
  return getParameters(set?set->get():0) ;
}



RooArgSet* RooAbsArg::getParameters(const RooArgSet* nset) const 
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that don't match any of the names the args in the 
  // supplied argset. The caller of this function is responsible 
  // for deleting the returned argset. The complement of this function 
  // is getDependents()


  RooArgSet parList("parameters") ;

  // Create and fill deep server list
  RooArgSet leafList("leafNodeServerList") ;
  leafNodeServerList(&leafList) ;

  // Copy non-dependent servers to parameter list
  TIterator* sIter = leafList.createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {

    if (!nset || !arg->dependsOn(*nset)) {
      parList.add(*arg) ;
    }
  }
  delete sIter ;

  // Call hook function for all branch nodes
  RooArgSet branchList ;
  branchNodeServerList(&branchList) ;
  RooAbsArg* branch ;
  TIterator* bIter = branchList.createIterator() ;
  while(branch=(RooAbsArg*)bIter->Next()) {
    branch->getParametersHook(nset, &parList) ; 
  }
  delete bIter ;

  RooArgList tmp(parList) ;
  tmp.sort() ;
  return new RooArgSet(tmp) ;
}



RooArgSet* RooAbsArg::getDependents(const RooAbsData* set) const 
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that match any of the names of the variable list
  // of the supplied data set (the dependents). The caller of this
  // function is responsible for deleting the returned argset.
  // The complement of this function is getDependents()

  if (!set) return new RooArgSet ;

  return getDependents(set->get()) ;
}


RooArgSet* RooAbsArg::getDependents(const RooArgSet* dataList) const 
{
  // Create a list of leaf nodes in the arg tree starting with
  // ourself as top node that match any of the names the args in the 
  // supplied argset. The caller of this function is responsible 
  // for deleting the returned argset. The complement of this function 
  // is getDependents()
  
  RooArgSet* depList = new RooArgSet("dependents") ;
  if (!dataList) return depList ;

  // Make iterator over tree leaf node list
  RooArgSet leafList("leafNodeServerList") ;
  leafNodeServerList(&leafList) ;    
  TIterator *sIter = leafList.createIterator() ;

  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (arg->dependsOn(*dataList)) {
      depList->add(*arg) ;
    }
  }  
  delete sIter ;

  // Call hook function for all branch nodes
  RooArgSet branchList ;
  branchNodeServerList(&branchList) ;
  RooAbsArg* branch ;
  TIterator* bIter = branchList.createIterator() ;
  while(branch=(RooAbsArg*)bIter->Next()) {
    branch->getDependentsHook(dataList, depList) ;
  }
  delete bIter ;

  return depList ;
}



Bool_t RooAbsArg::checkDependents(const RooArgSet* nset) const 
{
  // Overloadable function in which derived classes can implement
  // consistency checks of the variables. If this function returns
  // true, indicating an error, the fitter or generator will abort.
  return kFALSE ;
}


Bool_t RooAbsArg::recursiveCheckDependents(const RooArgSet* nset) const 
{
  RooArgSet nodeList ;
  treeNodeServerList(&nodeList) ;
  TIterator* iter = nodeList.createIterator() ;

  RooAbsArg* arg ;
  Bool_t ret(kFALSE) ;
  while(arg=(RooAbsArg*)iter->Next()) {
    ret |= arg->checkDependents(nset) ;
  }
  delete iter ;

  return ret ;
}


Bool_t RooAbsArg::dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  Bool_t result(kFALSE);
  TIterator* sIter = serverList.createIterator();
  RooAbsArg* server ;
  while (!result && (server=(RooAbsArg*)sIter->Next())) {
    if (dependsOn(*server,ignoreArg)) {
      result= kTRUE;
    }
  }
  delete sIter;
  return result;
}


Bool_t RooAbsArg::dependsOn(const RooAbsArg& testArg, const RooAbsArg* ignoreArg) const
{
  // Test whether we depend on (ie, are served by) the specified object.
  // Note that RooAbsArg objects are considered equivalent if they have
  // the same name.

  if (this==ignoreArg) return kFALSE ;

  // First check if testArg is self 
  if (!TString(testArg.GetName()).CompareTo(GetName())) return kTRUE ;

  // Next test direct dependence
  if (_serverList.FindObject(testArg.GetName())) return kTRUE ;

  // If not, recurse
  TIterator* sIter = serverIterator() ;
  RooAbsArg* server(0) ;
  while (server=(RooAbsArg*)sIter->Next()) {
    if (server->dependsOn(testArg)) {
      delete sIter ;
      return kTRUE ;
    }
  }

  delete sIter ;
  return kFALSE ;
}



Bool_t RooAbsArg::overlaps(const RooAbsArg& testArg) const 
{
  // Test if any of the nodes of tree are shared with that of the given tree

  RooArgSet list("treeNodeList") ;
  treeNodeServerList(&list) ;

  return testArg.dependsOn(list) ;
}



Bool_t RooAbsArg::dependentOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const
{
  // Test if any of the dependents of the arg tree (as determined by getDependents) 
  // overlaps with those of the testArg.

  return dependentOverlaps(dset->get(),testArg) ;
}


Bool_t RooAbsArg::dependentOverlaps(const RooArgSet* nset, const RooAbsArg& testArg) const
{
  // Test if any of the dependents of the arg tree (as determined by getDependents) 
  // overlaps with those of the testArg.

  RooArgSet* depList = getDependents(nset) ;
  Bool_t ret = testArg.dependsOn(*depList) ;
  delete depList ;
  return ret ;
}



void RooAbsArg::setValueDirty(const RooAbsArg* source) const
{ 
  // Mark this object as having changed its value, and propagate this status
  // change to all of our clients. If the object is not in automatic dirty
  // state propagation mode, this call has no effect

  if (_operMode!=Auto) return ;

  if (_verboseDirty) cout << "RooAbsArg::setValueDirty(" << GetName() << "," << this
			  << "): dirty flag " << (_valueDirty?"already ":"") << "raised" << endl ;

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
    cout << "RooAbsArg::setValueDirty(" << GetName() 
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  // Propagate dirty flag to all clients if this is a down->up transition
  _valueDirty = kTRUE ;
  _clientValueIter->Reset() ;
  RooAbsArg* client ;
  while (client=(RooAbsArg*)_clientValueIter->Next()) {
    client->setValueDirty(source) ;
  }

} 


void RooAbsArg::setShapeDirty(const RooAbsArg* source) const
{ 
  // Mark this object as having changed its shape, and propagate this status
  // change to all of our clients.

  if (_verboseDirty) cout << "RooAbsArg::setShapeDirty(" << GetName() 
			  << "): dirty flag " << (_shapeDirty?"already ":"") << "raised" << endl ;

  if (_clientListShape.GetSize()==0) {
    _shapeDirty = kTRUE ;
    return ;
  }

  // Set 'dirty' shape state for this object and propagate flag to all its clients
  if (source==0) {
    source=this ; 
  } else if (source==this) {
    // Cyclical dependency, abort
    cout << "RooAbsArg::setShapeDirty(" << GetName() 
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  // Propagate dirty flag to all clients if this is a down->up transition
  _shapeDirty=kTRUE ; 

  _clientShapeIter->Reset() ;
  RooAbsArg* client ;
  while (client=(RooAbsArg*)_clientShapeIter->Next()) {
    client->setShapeDirty(source) ;
    client->setValueDirty(source) ;
  }

} 



Bool_t RooAbsArg::redirectServers(const RooAbsCollection& newSet, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  // Substitute our servers with those listed in newSet. If nameChange is false, servers and
  // and substitutes are matched by name. If nameChange is true, servers are matched to args
  // in newSet that have the 'ORIGNAME:<servername>' attribute set. If mustReplaceAll is set,
  // a warning is printed and error status is returned if not all servers could be sucessfully
  // substituted.

  // Trivial case, no servers
  if (!_serverList.First()) return kFALSE ;

  // Replace current servers with new servers with the same name from the given list
  Bool_t ret(kFALSE) ;

  //Copy original server list to not confuse the iterator while deleting
  THashList origServerList, origServerValue, origServerShape ;
  RooAbsArg *oldServer, *newServer ;
  TIterator* sIter = _serverList.MakeIterator() ;
  while (oldServer=(RooAbsArg*)sIter->Next()) {
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
  while (oldServer=(RooAbsArg*)sIter->Next()) {

    newServer= oldServer->findNewServer(newSet, nameChange);
    if (newServer && _verboseDirty) {
      cout << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName() 
 	   << " redirected from " << oldServer << " to " << newServer << endl ;
    }

    if (!newServer) {
      if (mustReplaceAll) {
	cout << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName() 
	     << " (" << (void*)oldServer << ") not redirected" << endl ;
	ret = kTRUE ;
      }
      continue ;
    }
    
    propValue=origServerValue.FindObject(oldServer)?kTRUE:kFALSE ;
    propShape=origServerShape.FindObject(oldServer)?kTRUE:kFALSE ;
    removeServer(*oldServer) ;
    addServer(*newServer,propValue,propShape) ;
  }

  delete sIter ;
 
  setValueDirty() ;
  setShapeDirty() ;

  // Process the proxies
  Bool_t allReplaced=kTRUE ;
  for (int i=0 ; i<numProxies() ; i++) {
    Bool_t ret = getProxy(i)->changePointer(newSet,nameChange) ;    
    allReplaced &= ret ;
  }

  if (mustReplaceAll && !allReplaced) {
    cout << "RooAbsArg::redirectServers(" << GetName() 
	 << "): ERROR, some proxies could not be adjusted" << endl ;
    ret = kTRUE ;
  }

  // Optional subclass post-processing
  ret |= redirectServersHook(newSet,mustReplaceAll) ;

  return ret ;
}

RooAbsArg *RooAbsArg::findNewServer(const RooAbsCollection &newSet, Bool_t nameChange) const {
  // Find the new server in the specified set that matches the old server.
  // Allow a name change if nameChange is kTRUE, in which case the new
  // server is selected by searching for a new server with an attribute
  // of "ORIGNAME:<oldName>". Return zero if there is not a unique match.

  RooAbsArg *newServer(0);
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
	cout << "RooAbsArg::redirectServers(" << GetName() << "): FATAL Error, " << tmp->getSize() << " servers with " 
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

Bool_t RooAbsArg::recursiveRedirectServers(const RooAbsCollection& newSet, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  // Apply the redirectServers function recursively on all branch nodes in this argument tree.

  Bool_t ret(kFALSE) ;
  
  // Do redirect on self
  ret |= redirectServers(newSet,mustReplaceAll,nameChange) ;

  // Do redirect on servers
  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while(server=(RooAbsArg*)sIter->Next()) {
    ret |= server->recursiveRedirectServers(newSet,mustReplaceAll,nameChange) ;
  }
  delete sIter ;

  return ret ;
}



void RooAbsArg::registerProxy(RooArgProxy& proxy) 
{
  // Register an RooArgProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur. The proxied argument is
  // also added as value and/or shape server

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    cout << "RooAbsArg::registerProxy(" << GetName() << "): proxy named " 
	 << proxy.GetName() << " for arg " << proxy.absArg()->GetName() 
	 << " already registered" << endl ;
    return ;
  }

//   cout << (void*)this << " " << GetName() << ": registering proxy " 
//        << (void*)&proxy << " with name " << proxy.name() << " in mode " 
//        << (proxy.isValueServer()?"V":"-") << (proxy.isShapeServer()?"S":"-") << endl ;

  // Register proxied object as server
  addServer(*proxy.absArg(),proxy.isValueServer(),proxy.isShapeServer()) ;

  // Register proxy itself
  _proxyList.Add(&proxy) ;  
}


void RooAbsArg::unRegisterProxy(RooArgProxy& proxy)  
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
}



void RooAbsArg::registerProxy(RooSetProxy& proxy) 
{
  // Register an RooSetProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur.

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    cout << "RooAbsArg::registerProxy(" << GetName() << "): proxy named " 
 	 << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  _proxyList.Add(&proxy) ;  
}



void RooAbsArg::unRegisterProxy(RooSetProxy& proxy)  
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
}



void RooAbsArg::registerProxy(RooListProxy& proxy) 
{
  // Register an RooListProxy in the proxy list. This function is called by owned
  // proxies upon creation. After registration, this arg wil forward pointer
  // changes from serverRedirects and updates in cached normalization sets
  // to the proxies immediately after they occur.

  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    cout << "RooAbsArg::registerProxy(" << GetName() << "): proxy named " 
 	 << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  _proxyList.Add(&proxy) ;  
}



void RooAbsArg::unRegisterProxy(RooListProxy& proxy)  
{
  // Remove proxy from proxy list. This functions is called by owned proxies
  // upon their destruction.

  _proxyList.Remove(&proxy) ;
}



RooAbsProxy* RooAbsArg::getProxy(Int_t index) const
{
  // Return the nth proxy from the proxy list.

  // Cross cast: proxy list returns TObject base pointer, we need
  // a RooAbsProxy base pointer. C++ standard requires
  // a dynamic_cast for this. 
  return dynamic_cast<RooAbsProxy*> (_proxyList.At(index)) ;
}



Int_t RooAbsArg::numProxies() const
{
  // Return the number of registered proxies.

  return _proxyList.GetSize() ;
}



void RooAbsArg::setProxyNormSet(const RooArgSet* nset) 
{
  // Forward a change in the cached normalization argset
  // to all the registered proxies.

  for (int i=0 ; i<numProxies() ; i++) {
    getProxy(i)->changeNormSet(nset) ;
  }
}



void RooAbsArg::attachToTree(TTree& t, Int_t bufSize)
{
  // Overloadable function for derived classes to implement
  // attachment as branch to a TTree

  cout << "RooAbsArg::attachToTree(" << GetName() 
       << "): Cannot be attached to a TTree" << endl ;
}



Bool_t RooAbsArg::isValid() const
{
  // WVE (08/21/01) Probably obsolete now
  return kTRUE ;
}



void RooAbsArg::copyList(TList& dest, const TList& source)  
{
  // WVE (08/21/01) Probably obsolete now
  dest.Clear() ;

  TIterator* sIter = source.MakeIterator() ;
  TObject* obj ;
  while (obj = sIter->Next()) {
    dest.Add(obj) ;
  }
  delete sIter ;
}


void RooAbsArg::printToStream(ostream& os, PrintOption opt, TString indent)  const
{
  // Print the state of this object to the specified output stream.
  //
  //  OneLine : use RooPrintable::oneLinePrint()
  // Standard : use virtual writeToStream() method in non-compact mode
  //  Verbose : list dirty flags,attributes, clients, servers, and proxies
  //
  // Subclasses will normally call this method first in their implementation,
  // and then add any additional state of their own with the Shape or Verbose
  // options.

  if(opt == Standard) {
    os << ClassName() << "::" << GetName() ;
    Int_t nfill = _nameLength-strlen(GetName()) ;
    while(nfill-- > 0) os << " " ;
    os << ": " ;
    if (isDerived()) {
      os << "(" ;
      writeToStream(os,kFALSE);
      os << ") -> " ;
      writeToStream(os,kTRUE);
    } else {
      writeToStream(os,kFALSE);
    }
    os << endl;
  }
  else {
    oneLinePrint(os,*this);
    if(opt == Verbose) {
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
      while (client=(RooAbsArg*)clientIter->Next()) {
	os << indent << "    (" << (void*)client  << ","
	   << (_clientListValue.FindObject(client)?"V":"-")
	   << (_clientListShape.FindObject(client)?"S":"-")
	   << ") " ;
	client->printToStream(os,OneLine);
      }
      delete clientIter;
      
      // server list
      os << indent << "  Servers: " << endl;
      TIterator *serverIter= _serverList.MakeIterator();
      RooAbsArg* server ;
      while (server=(RooAbsArg*)serverIter->Next()) {
	os << indent << "    (" << (void*)server << ","
	   << (server->_clientListValue.FindObject((TObject*)this)?"V":"-")
	   << (server->_clientListShape.FindObject((TObject*)this)?"S":"-")
	   << ") " ;
	server->printToStream(os,OneLine);
      }
      delete serverIter;

      // proxy list
      os << indent << "  Proxies: " << endl ;
      for (int i=0 ; i<numProxies() ; i++) {
	RooAbsProxy* proxy=getProxy(i) ;

	if (proxy->IsA()->InheritsFrom(RooArgProxy::Class())) {
	  os << indent << "    " << proxy->name() << " -> " ;
	  ((RooArgProxy*)proxy)->absArg()->printToStream(os,OneLine) ;
	} else {
	  os << indent << "    " << proxy->name() << " -> " ;
	  TString moreIndent(indent) ;
	  moreIndent.Append("    ") ;
	  ((RooSetProxy*)proxy)->printToStream(os,Standard,moreIndent.Data()) ;
	}
      }
    }
  }
}

ostream& operator<<(ostream& os, RooAbsArg &arg)
{
  arg.writeToStream(os,kTRUE) ;
  return os ;
}

istream& operator>>(istream& is, RooAbsArg &arg)
{
  arg.readFromStream(is,kTRUE,kFALSE) ;
  return is ;
}

void RooAbsArg::printAttribList(ostream& os) const
{
  // Print the attribute list 

  TIterator *attribIter= _attribList.MakeIterator();
  TObjString* attrib ;
  Bool_t first(kTRUE) ;
  while (attrib=(TObjString*)attribIter->Next()) {
    os << (first?" [":",") << attrib->String() ;
    first=kFALSE ;
  }
  if (!first) os << "] " ;
  delete attribIter ;
}

void RooAbsArg::attachDataSet(const RooAbsData &set) 
{
  // Replace server nodes with names matching the dataset variable names
  // with those data set variables, making this PDF directly dependent on the dataset
  recursiveRedirectServers(*set.get(),kFALSE);
}



Int_t RooAbsArg::Compare(const TObject* other) const 
{
  // Utility function used by TCollection::Sort to compare contained TObjects
  // We implement comparison by name, resulting in alphabetical sorting by object name.
  return strcmp(GetName(),other->GetName()) ;
}


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
    while(branch=(RooAbsArg*)bIter->Next()) {
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



RooAbsArg& RooAbsArg::operator=(Int_t ival) 
{
  cout << "RooAbsArg::operator=(" << GetName() << ",Int_t) not an lvalue of the appropriate type" << endl ;
  return *this ;
}


RooAbsArg& RooAbsArg::operator=(Double_t fval) 
{
  cout << "RooAbsArg::operator=(" << GetName() << ",Double_t) not an lvalue of the appropriate type" << endl ;
  return *this ;
}


RooAbsArg& RooAbsArg::operator=(const char* cval) 
{
  cout << "RooAbsArg::operator=(" << GetName() << ",const char*) not an lvalue of the appropriate type" << endl ;
  return *this ;
}
