/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.cc,v 1.10 2001/03/29 01:59:08 verkerke Exp $
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

RooAbsArg::RooAbsArg(const char* name, const RooAbsArg& other)
  : TNamed(name,other.GetTitle())
{
  // Copy constructor transfers attributes and servers from the original
  // object. The newly created object has no clients and has its dirty
  // flags set.
  initCopy(other) ;
}



RooAbsArg::RooAbsArg(const RooAbsArg& other)
  : TNamed(other)
{
  // Copy constructor transfers attributes and servers from the original
  // object. The newly created object has no clients and has its dirty
  // flags set.
  initCopy(other) ;
}



void RooAbsArg::initCopy(const RooAbsArg& other)
{
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
  assert(!fatalError) ;
}


TObject* RooAbsArg::Clone() {
  // Special clone function that takes care of bidirectional client-server links.
  
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

RooAbsArg& RooAbsArg::operator=(const RooAbsArg& other) 
{  
  // Assignment operator.

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

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
  return *this ;
}


void RooAbsArg::setAttribute(Text_t* name, Bool_t value) 
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


Bool_t RooAbsArg::getAttribute(Text_t* name) const
{
  // Check if a named attribute is set. By default, all attributes
  // are unset.
  return _attribList.FindObject(name)?kTRUE:kFALSE ;
}


void RooAbsArg::addServer(RooAbsArg& server) 
{
  // Register another RooAbsArg as a server to us, ie, declare that
  // we depend on its value and shape.

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
  } else {
    cout << "RooAbsArg::removeServer(" << GetName() 
	 << "): Never registered as client of " << server.GetName() << endl ;
  }
} 


Bool_t RooAbsArg::dependsOn(const RooArgSet& serverList) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  TIterator* sIter = serverList.MakeIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)sIter->Next()) {
    if (dependsOn(*server)) return kTRUE  ;
  }
  return kFALSE ;
}


Bool_t RooAbsArg::dependsOn(const RooAbsArg& server) const
{
  // Test whether we depend on (ie, are served by) the specified object.
  // Note that RooAbsArg objects are considered equivalent if they have
  // the same name.

  return _serverList.FindObject(server.GetName())?kTRUE:kFALSE ;
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
			  << "): dirty flag " << (flag?"raised":"cleared") << endl ;

  _valueDirty=flag ; 

  if (flag==kTRUE) {
    // Set 'dirty' flag for all clients of this object
    TIterator *clientIter= _clientList.MakeIterator();
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
			  << "): dirty flag " << (flag?"raised":"cleared") << endl ;
  _shapeDirty=flag ; 

  if (flag==kTRUE) {
    // Set 'dirty' flag for all clients of this object
    TIterator *clientIter= _clientList.MakeIterator();
    RooAbsArg* client ;
    while (client=(RooAbsArg*)clientIter->Next()) {
      client->setShapeDirty(kTRUE,source) ;
    }
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



Bool_t RooAbsArg::isValid() const
{
  return kTRUE ;
}



void RooAbsArg::printToStream(ostream& str, PrintOption opt)  const
{
  // Print the state of this object to the specified output stream.
  // With PrintOption=Verbose, print out lists of attributes, clients,
  // and servers. Otherwise, print our class, name and title only.

  cout << GetName() << ": " << GetTitle() << endl;
  if(opt == Verbose) {
    // attribute list
    str << "  Attributes :" ;
    printAttribList(str) ;
    str << endl ;
    // client list
    str << "  Clients:";
    TIterator *clientIter= _clientList.MakeIterator();
    RooAbsArg* client ;
    while (client=(RooAbsArg*)clientIter->Next()) {
      client->printToStream(str,OneLine);
    }
    // server list
    str << "  Servers:";
    TIterator *serverIter= _serverList.MakeIterator();
    RooAbsArg* server ;
    while (server=(RooAbsArg*)serverIter->Next()) {
      server->printToStream(str,OneLine);
    }
  }
}

void RooAbsArg::Print(Option_t *options) const {
  // Print the state of this object using printToStream() with the
  // following PrintOption mapping:
  //
  //  "1" - OneLine
  //  "S" - Shape
  //  "V" - Verbose
  //
  // The default is Standard.

  TString opts(options);
  opts.ToLower();
  PrintOption popt(Standard);
  if(opts.Contains("1")) { popt= OneLine ; }
  if(opts.Contains("s")) { popt= Shape; }
  if(opts.Contains("v")) { popt= Verbose;}
  printToStream(cout,popt);
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
