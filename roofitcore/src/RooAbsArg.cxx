/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.cc,v 1.3 2001/03/16 07:59:11 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "TObjString.h"

#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooAbsArg)

Bool_t RooAbsArg::_verboseDirty(kFALSE) ;

RooAbsArg::RooAbsArg() : TNamed(), _attribList()
{
}

RooAbsArg::RooAbsArg(const char *name, const char *title) : 
  TNamed(name,title), _valueDirty(kTRUE), _shapeDirty(kTRUE)
{    
  // Default constructor
}

RooAbsArg::RooAbsArg(const RooAbsArg& other) : TNamed(other)
{
  // Copy constructor

  // take attributes from target
  TObject* obj ;
  TIterator* aIter = other._attribList.MakeIterator() ;
  while (obj=aIter->Next()) {
    _attribList.Add(obj->Clone()) ;
  }

  // Copy server list by hand
  TIterator* sIter = other._serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server = (RooAbsArg*) sIter->Next()) {
    addServer(*server) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsArg::~RooAbsArg() 
{
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
  assert(!fatalError) ;
}


TObject* RooAbsArg::Clone() {
  // Special clone function that takes care of bidirectional client-server links
  
  // Streamer-based clone()
  RooAbsArg* clone = (RooAbsArg*) TObject::Clone() ;
  
  // Copy server list by hand
  TIterator* iter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server = (RooAbsArg*) iter->Next()) {
    clone->addServer(*server) ;
  }

  return clone ;
}

RooAbsArg& RooAbsArg::operator=(RooAbsArg& other) 
{  
  // Assignment operator 

  // Base class operator
  TNamed::operator=(other) ;

  // Remove all current servers
  TIterator* iter = _serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server = (RooAbsArg*) iter->Next()) {
    removeServer(*server) ;
  }
  delete iter ;

  // Add all new servers
  iter = other._serverList.MakeIterator() ;
  while (server = (RooAbsArg*) iter->Next()) {
    addServer(*server) ;
  }
  delete iter ;

  raiseClientValueDirtyFlags() ;
  raiseClientShapeDirtyFlags() ;
  return *this ;
}


void RooAbsArg::setAttribute(Text_t* name, Bool_t value) 
{
  // Add or remove attribute from hash list

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


Bool_t RooAbsArg::getAttribute(Text_t* name) const
{
  // Check presence of attribute in list
  return _attribList.FindObject(name)?kTRUE:kFALSE ;
}


void RooAbsArg::addServer(RooAbsArg& server) 
{
  // Add server link to given server
  if (!_serverList.FindObject(&server)) {
    _serverList.Add(&server) ;
  } else {
    cout << "RooAbsArg::addServer(" << GetName() << "): Server " 
	 << server.GetName() << " already registered" << endl ;
    return ;
  }

  if (!server._clientList.FindObject(this)) {
    server._clientList.Add(this) ;
  } else {
    cout << "RooAbsArg::addServer(" << GetName() 
	 << "): Already registered as client of " << server.GetName() << endl ;
  }
} 


void RooAbsArg::removeServer(RooAbsArg& server) 
{
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
  } else {
    cout << "RooAbsArg::removeServer(" << GetName() 
	 << "): Never registered as client of " << server.GetName() << endl ;
  }
} 


Bool_t RooAbsArg::dependsOn(RooArgSet& serverList) 
{
  TIterator* sIter = serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)sIter->Next()) {
    if (dependsOn(*server)) return kTRUE  ;
  }
  return kFALSE ;
}


Bool_t RooAbsArg::dependsOn(RooAbsArg& server) 
{
  return _serverList.FindObject(server.GetName())?kTRUE:kFALSE ;
}


void RooAbsArg::setValueDirty(Bool_t flag, RooAbsArg* source) 
{ 
  // Set 'dirty' value state for this object and propagate flag to all its clients
  if (source==0) {
    source=this ; 
  } else if (source==this) {
    // Cyclical dependency, abort
    cout << "RooAbsArg::setValueDirty(" << GetName() 
	 << "): cyclical dependency detected" << endl ;
    return ;
  }

  if (_verboseDirty) cout << "RooAbsArg::setValueDirty(" << GetName() 
			  << "): dirty flag " << (flag?"raised":"cleared") << endl ;
  if (_serverList.First()!=0) {
    _valueDirty=flag ; 
  }
  if (flag==kTRUE) raiseClientValueDirtyFlags(source) ; 
} 


void RooAbsArg::raiseClientValueDirtyFlags(RooAbsArg* source)
{
  // Set 'dirty' flag for all clients of this object
  if (source==0) source=this ;
  TIterator *clientIter= _clientList.MakeIterator();
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    client->setValueDirty(kTRUE,source) ;
  }
}

void RooAbsArg::setShapeDirty(Bool_t flag, RooAbsArg* source) 
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
			  << "): dirty flag " << (flag?"raised":"cleared") << endl ;
  if (_serverList.First()!=0) {
    _shapeDirty=flag ; 
  }
  if (flag==kTRUE) raiseClientShapeDirtyFlags(source) ; 
} 


void RooAbsArg::raiseClientShapeDirtyFlags(RooAbsArg* source)
{  
  // Set 'dirty' flag for all clients of this object
  if (source==0) source=this ;
  TIterator *clientIter= _clientList.MakeIterator();
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    client->setShapeDirty(kTRUE,source) ;
  }
}


Bool_t RooAbsArg::redirectServers(RooArgSet& newSet, Bool_t mustReplaceAll) 
{
  // Replace current servers with new servers with the same name from the given list

  Bool_t ret(kFALSE) ;

  //Copy original server list to not confuse the iterator while deleting
  THashList origServerList ;
  RooAbsArg *oldServer, *newServer ;
  TIterator* sIter = _serverList.MakeIterator() ;
  while (oldServer=(RooAbsArg*)sIter->Next()) {
    origServerList.Add(oldServer) ;
  }
  delete sIter ;
  

  // Delete all previously registered servers 
  sIter = origServerList.MakeIterator() ;
  while (oldServer=(RooAbsArg*)sIter->Next()) {
    newServer = newSet.find(oldServer->GetName()) ;
    if (!newServer) {
      if (mustReplaceAll) {
	cout << "RooAbsArg::redirectServers: server " << oldServer->GetName() << " not redirected" << endl ;
	ret = kTRUE ;
      }
      continue ;
    }

    removeServer(*oldServer) ;
    addServer(*newServer) ;
  }

  delete sIter ;
 
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;

  // Optional subclass post-processing
  ret |= redirectServersHook(newSet,mustReplaceAll) ;

  return ret ;
}


void RooAbsArg::attachToTree(TTree& t, Int_t bufSize=32000)
{
  cout << "RooAbsArg::attachToTree(" << GetName() << "): Cannot be attached to a TTree" << endl ;
}



Bool_t RooAbsArg::isValid() 
{
  return kTRUE ;
}



void RooAbsArg::printToStream(ostream& str, PrintOption opt) 
{
  // Print contents

  // We only have attributes to show
  str << GetName() << ": attributes :" ;
  printAttribList(str) ;
  str << endl ;
  
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


void RooAbsArg::printAttribList(ostream& os) 
{
  // Print the attribute list 
  TIterator *attribIter= _attribList.MakeIterator();
  TObjString* attrib ;
  while (attrib=(TObjString*)attribIter->Next()) {
    os << " " << attrib->String() ;
  }
}


void RooAbsArg::printLinks() 
{
  // DEBUG: Print client and server links of this object 
  cout << GetName() << "(" << (void*)this << "): link map" << endl ;
  cout << "\tClients (depending on this RooAbsArg)" << endl ;
  TIterator *clientIter= _clientList.MakeIterator();
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    cout << "\t" << client->GetName() << "(" << (void*)client << ")" << endl ;
  }
  
  cout << "\tServers (needed by this RooAbsArg)" << endl ;
  TIterator *serverIter= _serverList.MakeIterator();
  RooAbsArg* server ;
  while (server=(RooAbsArg*)serverIter->Next()) {
    cout << "\t" << server->GetName() << "(" << (void*)server << ")" << endl ;
  }
}
