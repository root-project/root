/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.cc,v 1.19 2001/05/02 18:08:59 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
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
#include "RooFitCore/RooDataSet.hh"

#include <string.h>

ClassImp(RooAbsArg)

Bool_t RooAbsArg::_verboseDirty(kFALSE) ;

RooAbsArg::RooAbsArg() : TNamed(), _attribList()
{
  // Default constructor creates an unnamed object.
}

RooAbsArg::RooAbsArg(const char *name, const char *title)
  : TNamed(name,title), _valueDirty(kTRUE), _shapeDirty(kTRUE)
{    
  // Create an object with the specified name and descriptive title.
  // The newly created object has no clients or servers and has its
  // dirty flags set.
}

RooAbsArg::RooAbsArg(const RooAbsArg& other, const char* name)
  : TNamed(other.GetName(),other.GetTitle())
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
    _attribList.Add(obj->Clone()) ;
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

  setValueDirty() ;
  setShapeDirty() ;
}



RooAbsArg::~RooAbsArg() 
{
  // Destructor notifies its servers that they no longer need to serve us and
  // notifies its clients that they are now in limbo (!)

  _attribList.Delete() ;

  //Notify all servers that they no longer need to serve us
  TIterator* serverIter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)serverIter->Next()) {
    removeServer(*server) ;
  }

  //Notify all client that they are in limbo
  TIterator* clientIter = _clientList.MakeIterator() ;
  Bool_t fatalError(kFALSE) ;
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    client->setAttribute("FATAL:ServerDied") ;
    cout << "RooAbsArg::~RooAbsArg(" << GetName() << "): Fatal error: dependent RooAbsArg " 
	 << client->GetName() << " should have been deleted before" << endl ;
    fatalError=kTRUE ;
  }

  //assert(!fatalError) ;
}


RooAbsArg& RooAbsArg::operator=(const RooAbsArg& other) 
{  
  // Assignment operator: copies value contents and server list of 'other' object
  // All other properties are untouched.

  // Base class operator
  TNamed::operator=(other) ;

  // Remove all current servers
  TIterator* iter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server = (RooAbsArg*) iter->Next()) {
    cout << "...removing " << server->GetName() << endl;
    removeServer(*server) ;
  }
  delete iter ;

  // Add all new servers
  iter = other._serverList.MakeIterator() ;
  while (server = (RooAbsArg*) iter->Next()) {
    cout << "...adding " << server->GetName() << endl;
    addServer(*server) ;
  }
  delete iter ;
  
  // proxies?

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
  return *this ;
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
  // we depend on its value and shape.

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



void RooAbsArg::addServerList(RooArgSet& serverList, Bool_t valueProp, Bool_t shapeProp) 
{
  RooAbsArg* arg ;
  TIterator* iter = serverList.MakeIterator() ;
  while (arg=(RooAbsArg*)iter->Next()) {
    addServer(*arg,valueProp,shapeProp) ;
  }
  delete iter ;
}



void RooAbsArg::removeServer(RooAbsArg& server) 
{
  // Unregister another RooAbsArg as a server to us, ie, declare that
  // we no longer depend on its value and shape.

  // Remove server link to given server
  if (_serverList.FindObject(&server)) {
    _serverList.Remove(&server) ;
  } else {
    cout << "RooAbsArg::removeServer(" << GetName() << "): Server " 
	 << server.GetName() << " wasn't registered" << endl ;
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



void RooAbsArg::leafNodeServerList(RooArgSet* list, const RooAbsArg* arg) const
{
  treeNodeServerList(list,arg,kFALSE,kTRUE) ;
}



void RooAbsArg::branchNodeServerList(RooArgSet* list, const RooAbsArg* arg) const 
{
  treeNodeServerList(list,arg,kTRUE,kFALSE) ;
}


void RooAbsArg::treeNodeServerList(RooArgSet* list, const RooAbsArg* arg, Bool_t doBranch, Bool_t doLeaf) const
  // Do recursive deep copy of all 'ultimate' servers 
{
  if (!arg) arg=this ;

  if ((doBranch&&doLeaf) ||
      (doBranch&&arg->isDerived()) ||
      (doLeaf&&!arg->isDerived())) {
    list->add(*arg) ;  
  }

  RooAbsArg* server ;
  TIterator* sIter = arg->serverIterator() ;
  while (server=(RooAbsArg*)sIter->Next()) {
    treeNodeServerList(list,server,doBranch,doLeaf) ;
    }
  
  delete sIter ;
}




RooArgSet* RooAbsArg::getParameters(const RooDataSet* set) const 
{
  RooArgSet* parList = new RooArgSet("parameters") ;
  const RooArgSet* dataList = set->get() ;

  // Create and fill deep server list
  RooArgSet leafList("leafNodeServerList") ;
  leafNodeServerList(&leafList) ;

  // Copy non-dependent servers to parameter list
  TIterator* sIter = leafList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (!dataList->FindObject(arg->GetName())) {
      parList->add(*arg) ;
    }
  }
  delete sIter ;

  return parList ;
}



RooArgSet* RooAbsArg::getDependents(const RooDataSet* set) const 
{
  RooArgSet* depList = new RooArgSet("dependents") ;
  const RooArgSet* dataList = set->get() ;

  // Create and fill deep server list
  RooArgSet leafList("leafNodeServerList") ;
  leafNodeServerList(&leafList) ;

  // Copy dependent servers to dependent list
  TIterator* sIter = leafList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (dataList->FindObject(arg->GetName())) {
      depList->add(*arg) ;
    }
  }
  delete sIter ;

  return depList ;
}


Bool_t RooAbsArg::checkDependents(const RooDataSet* set) const 
{
  // Always OK if we have no servers
  if (!_serverList.First()) {
    return kFALSE ;
  }
  
  Bool_t ret=kFALSE ;

  RooAbsArg *server, *server2 ;
  TIterator *sIter  = serverIterator() ;
  TIterator *sIter2 = serverIterator() ;
  while(server=(RooAbsArg*)sIter->Next()) {

    // First check if argument is OK
    if (server->checkDependents(set)) {
      ret = kTRUE ;
    } else {    
      sIter2->Reset() ;
      while((server2=(RooAbsArg*)sIter2->Next()) && server2!=server) {
	if (server->dependentOverlaps(set,*server2)) {
	  cout << "RooAbsArg::checkDependents(" << GetName() << "): ERROR: components " << server->GetName() 
	       << " and " << server2->GetName() << " have one or more dependents in common" << endl ;
	  ret = kTRUE ;
	}
      }
    }
  }

  delete sIter ;
  delete sIter2 ;

  return ret ;
}


Bool_t RooAbsArg::dependsOn(const RooArgSet& serverList) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  Bool_t result(kFALSE);
  TIterator* sIter = serverList.MakeIterator();
  RooAbsArg* server ;
  while (!result && (server=(RooAbsArg*)sIter->Next())) {
    if (dependsOn(*server)) {
      result= kTRUE;
    }
  }
  delete sIter;
  return result;
}


Bool_t RooAbsArg::dependsOn(const RooAbsArg& testArg) const
{
  // Test whether we depend on (ie, are served by) the specified object.
  // Note that RooAbsArg objects are considered equivalent if they have
  // the same name.

  // First test direct dependence
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
  RooArgSet list("list") ;
  treeNodeServerList(&list) ;

  return testArg.dependsOn(list) ;
}



Bool_t RooAbsArg::dependentOverlaps(const RooDataSet* dset, const RooAbsArg& testArg) const
{
  RooArgSet* depList = getDependents(dset) ;
  Bool_t ret = testArg.dependsOn(*depList) ;
  delete depList ;
  return ret ;
}



void RooAbsArg::setValueDirty(Bool_t flag, const RooAbsArg* source) const
{ 
  // Mark this object as having changed its value, and propagate this status
  // change to all of our clients.

  if (source==0) {
    source=this ; 
  } else if (source==this) {
    // Cyclical dependency, abort
    cout << "RooAbsArg::setValueDirty(" << GetName() 
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  if (_verboseDirty) cout << "RooAbsArg::setValueDirty(" << GetName() 
			  << "," << (void*)this << "): dirty flag " << (flag?"raised":"cleared") << endl ;

  _valueDirty=flag ; 

  if (flag==kTRUE) {
    // Set 'dirty' flag for all clients of this object
    TIterator *clientIter= _clientListValue.MakeIterator();
    RooAbsArg* client ;
    while (client=(RooAbsArg*)clientIter->Next()) {
      client->setValueDirty(kTRUE,source) ;
    }
  }
} 


void RooAbsArg::setShapeDirty(Bool_t flag, const RooAbsArg* source) const
{ 
  // Set 'dirty' shape state for this object and propagate flag to all its clients
  if (source==0) {
    source=this ; 
  } else if (source==this) {
    // Cyclical dependency, abort
    cout << "RooAbsArg::setShapeDirty(" << GetName() 
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  if (_verboseDirty) cout << "RooAbsArg::setShapeDirty(" << GetName() 
			  << "," << (void*)this << "): dirty flag " << (flag?"raised":"cleared") << endl ;
  _shapeDirty=flag ; 

  if (flag==kTRUE) {
    // Set 'dirty' flag for all clients of this object
    TIterator *clientIter= _clientListShape.MakeIterator();
    RooAbsArg* client ;
    while (client=(RooAbsArg*)clientIter->Next()) {
      client->setShapeDirty(kTRUE,source) ;
    }
  }
} 



Bool_t RooAbsArg::redirectServers(const RooArgSet& newSet, Bool_t mustReplaceAll) 
{
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
    newServer = newSet.find(oldServer->GetName()) ;
    if (!newServer) {
      if (mustReplaceAll) {
	cout << "RooAbsArg::redirectServers: server " << oldServer->GetName() 
	     << " not redirected" << endl ;
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
 
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;

  // Process the proxies
  Bool_t allReplaced=kTRUE ;
  for (int i=0 ; i<numProxies() ; i++) {
    allReplaced &= getProxy(i).changePointer(newSet) ;    
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



Bool_t RooAbsArg::recursiveRedirectServers(const RooArgSet& newSet, Bool_t mustReplaceAll) 
{
  Bool_t ret(kFALSE) ;
  
  // Do redirect on self
  ret |= redirectServers(newSet,mustReplaceAll) ;

  // Do redirect on servers
  TIterator* sIter = serverIterator() ;
  RooAbsArg* server ;
  while(server=(RooAbsArg*)sIter->Next()) {
    ret |= server->recursiveRedirectServers(newSet,mustReplaceAll) ;
  }

  return ret ;
}



void RooAbsArg::registerProxy(RooArgProxy& proxy) 
{
  // Every proxy can be registered only once
  if (_proxyArray.FindObject(&proxy)) {
    cout << "RooAbsArg::registerProxy(" << GetName() << "): proxy named " 
	 << proxy.GetName() << " for arg " << proxy.absArg()->GetName() 
	 << " already registered" << endl ;
    return ;
  }

  // Register proxied object as server
  addServer(*proxy.absArg()) ;

  // Register proxy itself
  _proxyArray.Add(&proxy) ;  
}



RooArgProxy& RooAbsArg::getProxy(Int_t index) const
{
  return *((RooArgProxy*)_proxyArray.At(index)) ;
}



Int_t RooAbsArg::numProxies() const
{
  return _proxyArray.GetEntries() ;
}



void RooAbsArg::attachToTree(TTree& t, Int_t bufSize)
{
  cout << "RooAbsArg::attachToTree(" << GetName() 
       << "): Cannot be attached to a TTree" << endl ;
}



Bool_t RooAbsArg::isValid() const
{
  return kTRUE ;
}



void RooAbsArg::copyList(TList& dest, const TList& source)  
{
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
    writeToStream(os,kFALSE);
    os << endl;
  }
  else {
    oneLinePrint(os,*this);
    if(opt == Verbose) {
      os << indent << "--- RooAbsArg ---" << endl;
      // dirty state flags
      os << indent << "  Value State: " << (_valueDirty ? "DIRTY":"clean") << endl
	 << indent << "  Shape State: " << (_shapeDirty ? "DIRTY":"clean") << endl;
      // attribute list
      os << indent << "  Attributes: " ;
      printAttribList(os) ;
      os << endl ;

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
	RooArgProxy& proxy=getProxy(i) ;
	os << indent << "    " << proxy.GetName() << " -> " ;
	proxy.absArg()->printToStream(os,OneLine) ;
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
}
