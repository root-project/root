/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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

RooAbsArg::RooAbsArg(const char *name, const char *title) : TNamed(name,title)
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
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    client->setAttribute("FATAL:ServerDied") ;
  }
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

  raiseClientDirtyFlags() ;
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


void RooAbsArg::setDirty(Bool_t flag) 
{ 
  // Set 'dirty' state for this object and propagate flag to all its clients
  if (_verboseDirty) cout << "RooAbsArg::setDirty(" << GetName() << "): dirty flag " << (flag?"raised":"cleared") << endl ;
  Bool_t prevStat = _dirty ;
  if (_serverList.First()!=0) {
    _dirty=flag ; 
  }
  if (flag==kTRUE && prevStat==kFALSE) raiseClientDirtyFlags() ; 
} 


void RooAbsArg::raiseClientDirtyFlags()
{
  // Set 'dirty' flag for all clients of this object
  TIterator *clientIter= _clientList.MakeIterator();
  RooAbsArg* client ;
  while (client=(RooAbsArg*)clientIter->Next()) {
    client->setDirty(kTRUE) ;
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
  return ret ;
}



void RooAbsArg::PrintToStream(ostream& str, Option_t* = 0) 
{
  // Print contents

  // We only have attributes to show
  str << GetName() << ": attributes :" ;
  printAttribList(str) ;
  str << endl ;
  
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
