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
/**  \class RooAbsArg
     \ingroup Roofitcore

RooAbsArg is the common abstract base class for objects that
represent a value and a "shape" in RooFit. Values or shapes usually depend on values
or shapes of other RooAbsArg instances. Connecting several RooAbsArg in
a computation graph models an expression tree that can be evaluated.

### Building a computation graph of RooFit objects
Therefore, RooAbsArg provides functionality to connect objects of type RooAbsArg into
a computation graph to pass values between those objects.
A value can e.g. be a real-valued number, (instances of RooAbsReal), or an integer, that is,
catgory index (instances of RooAbsCategory). The third subclass of RooAbsArg is RooStringVar,
but it is rarely used.

The "shapes" that a RooAbsArg can possess can e.g. be the definition
range of an observable, or how many states a category object has. In computations,
values are expected to change often, while shapes remain mostly constant
(unless e.g. a new range is set for an observable).

Nodes of a computation graph are connected using instances of RooAbsProxy.
If Node B declares a member `RooTemplateProxy<TypeOfNodeA>`, Node A will be
registered as a server of values to Node B, and Node B will know that it is
a client of node A. Using functions like dependsOn(), or getObservables()
/ getParameters(), the relation of `A --> B` can be queried. Using graphVizTree(),
one can create a visualisation of the expression tree.


An instance of RooAbsArg can have named attributes. It also has flags
to indicate that either its value or its shape were changed (= it is dirty).
RooAbsArg provides functionality to manage client/server relations in
a computation graph (\ref clientServerInterface), and helps propagating
value/shape changes through the graph. RooAbsArg implements interfaces
for inspecting client/server relationships (\ref clientServerInterface) and
setting/clearing/querying named attributes.

### Caching of values
The values of nodes in the computation graph are cached in RooFit. If
a value is used in two nodes of a graph, it doesn't need to be recomputed. If
a node acquires a new value, it notifies its consumers ("clients") that
their cached values are dirty. See the functions in \ref optimisationInterface
for details.
A node uses its isValueDirty() and isShapeDirty() functions to decide if a
computation is necessary. Caching can be vetoed globally by setting a
bit using setDirtyInhibit(). This will make computations slower, but all the
nodes of the computation graph will be evaluated irrespective of whether their
state is clean or dirty. Using setOperMode(), caching can also be enabled/disabled
for single nodes.

*/

#include "TBuffer.h"
#include "TClass.h"
#include "TVirtualStreamerInfo.h"
#include "strlcpy.h"

#include "RooSecondMoment.h"
#include "RooWorkspace.h"

#include "RooMsgService.h"
#include "RooAbsArg.h"
#include "RooArgSet.h"
#include "RooArgProxy.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"
#include "RooAbsData.h"
#include "RooAbsCategoryLValue.h"
#include "RooTrace.h"
#include "RooRealIntegral.h"
#include "RooConstVar.h"
#include "RooExpensiveObjectCache.h"
#include "RooAbsDataStore.h"
#include "RooResolutionModel.h"
#include "RooVectorDataStore.h"
#include "RooTreeDataStore.h"
#include "RooHelpers.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std;

ClassImp(RooAbsArg);
;

bool RooAbsArg::_verboseDirty(false) ;
bool RooAbsArg::_inhibitDirty(false) ;
bool RooAbsArg::inhibitDirty() const { return _inhibitDirty && !_localNoInhibitDirty; }

std::map<RooAbsArg*,std::unique_ptr<TRefArray>> RooAbsArg::_ioEvoList;
std::stack<RooAbsArg*> RooAbsArg::_ioReadStack ;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsArg::RooAbsArg()
   : TNamed(), _deleteWatch(false), _valueDirty(true), _shapeDirty(true), _operMode(Auto), _fast(false), _ownedComponents(nullptr),
     _prohibitServerRedirect(false), _namePtr(0), _isConstant(false), _localNoInhibitDirty(false),
     _myws(0)
{
  _namePtr = RooNameReg::instance().constPtr(GetName()) ;

}

////////////////////////////////////////////////////////////////////////////////
/// Create an object with the specified name and descriptive title.
/// The newly created object has no clients or servers and has its
/// dirty flags set.

RooAbsArg::RooAbsArg(const char *name, const char *title)
   : TNamed(name, title), _deleteWatch(false), _valueDirty(true), _shapeDirty(true), _operMode(Auto), _fast(false),
     _ownedComponents(0), _prohibitServerRedirect(false), _namePtr(0), _isConstant(false),
     _localNoInhibitDirty(false), _myws(0)
{
  if (name == nullptr || strlen(name) == 0) {
    throw std::logic_error("Each RooFit object needs a name. "
        "Objects representing the same entity (e.g. an observable 'x') are identified using their name.");
  }
  _namePtr = RooNameReg::instance().constPtr(GetName()) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor transfers all boolean and string properties of the original
/// object. Transient properties and client-server links are not copied

RooAbsArg::RooAbsArg(const RooAbsArg &other, const char *name)
   : TNamed(name ? name : other.GetName(), other.GetTitle()), RooPrintable(other),
     _boolAttrib(other._boolAttrib),
     _stringAttrib(other._stringAttrib), _deleteWatch(other._deleteWatch), _operMode(Auto), _fast(false),
     _ownedComponents(0), _prohibitServerRedirect(false),
     _namePtr(name ? RooNameReg::instance().constPtr(name) : other._namePtr),
     _isConstant(other._isConstant), _localNoInhibitDirty(other._localNoInhibitDirty), _myws(0)
{

  // Copy server list by hand
  bool valueProp, shapeProp ;
  for (const auto server : other._serverList) {
    valueProp = server->_clientListValue.containsByNamePtr(&other);
    shapeProp = server->_clientListShape.containsByNamePtr(&other);
    addServer(*server,valueProp,shapeProp) ;
  }

  setValueDirty() ;
  setShapeDirty() ;

  //setAttribute(Form("CloneOf(%08x)",&other)) ;
  //cout << "RooAbsArg::cctor(" << this << ") #bools = " << _boolAttrib.size() << " #strings = " << _stringAttrib.size() << endl ;

}


////////////////////////////////////////////////////////////////////////////////
/// Destructor.

RooAbsArg::~RooAbsArg()
{
  // Notify all servers that they no longer need to serve us
  while (!_serverList.empty()) {
    removeServer(*_serverList.containedObjects().back(), true);
  }

  // Notify all clients that they are in limbo
  std::vector<RooAbsArg*> clientListTmp(_clientList.begin(), _clientList.end()); // have to copy, as we invalidate iterators
  bool first(true) ;
  for (auto client : clientListTmp) {
    client->setAttribute("ServerDied") ;
    TString attr("ServerDied:");
    attr.Append(GetName());
    attr.Append(Form("(%zx)",(size_t)this)) ;
    client->setAttribute(attr.Data());
    client->removeServer(*this,true);

    if (_verboseDirty) {

      if (first) {
   cxcoutD(Tracing) << "RooAbsArg::dtor(" << GetName() << "," << this << ") DeleteWatch: object is being destroyed" << endl ;
   first = false ;
      }

      cxcoutD(Tracing)  << fName << "::" << ClassName() << ":~RooAbsArg: dependent \""
             << client->GetName() << "\" should have been deleted first" << endl ;
    }
  }

  if (_ownedComponents) {
    delete _ownedComponents ;
    _ownedComponents = 0 ;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Control global dirty inhibit mode. When set to true no value or shape dirty
/// flags are propagated and cache is always considered to be dirty.

void RooAbsArg::setDirtyInhibit(bool flag)
{
  _inhibitDirty = flag ;
}


////////////////////////////////////////////////////////////////////////////////
/// Activate verbose messaging related to dirty flag propagation

void RooAbsArg::verboseDirty(bool flag)
{
  _verboseDirty = flag ;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if this object was created as a clone of 'other'

bool RooAbsArg::isCloneOf(const RooAbsArg& other) const
{
  return (getAttribute(Form("CloneOf(%zx)",(size_t)&other)) ||
     other.getAttribute(Form("CloneOf(%zx)",(size_t)this))) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Set (default) or clear a named boolean attribute of this object.

void RooAbsArg::setAttribute(const Text_t* name, bool value)
{
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


////////////////////////////////////////////////////////////////////////////////
/// Check if a named attribute is set. By default, all attributes are unset.

bool RooAbsArg::getAttribute(const Text_t* name) const
{
  return (_boolAttrib.find(name) != _boolAttrib.end()) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Associate string 'value' to this object under key 'key'

void RooAbsArg::setStringAttribute(const Text_t* key, const Text_t* value)
{
  if (value) {
    _stringAttrib[key] = value ;
  } else {
    removeStringAttribute(key);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a string attribute with a given key.

void RooAbsArg::removeStringAttribute(const Text_t* key)
{
  _stringAttrib.erase(key) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Get string attribute mapped under key 'key'. Returns null pointer
/// if no attribute exists under that key

const Text_t* RooAbsArg::getStringAttribute(const Text_t* key) const
{
  map<string,string>::const_iterator iter = _stringAttrib.find(key) ;
  if (iter!=_stringAttrib.end()) {
    return iter->second.c_str() ;
  } else {
    return 0 ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Set (default) or clear a named boolean attribute of this object.

void RooAbsArg::setTransientAttribute(const Text_t* name, bool value)
{
  if (value) {

    _boolAttribTransient.insert(name) ;

  } else {

    set<string>::iterator iter = _boolAttribTransient.find(name) ;
    if (iter != _boolAttribTransient.end()) {
      _boolAttribTransient.erase(iter) ;
    }

  }

}


////////////////////////////////////////////////////////////////////////////////
/// Check if a named attribute is set. By default, all attributes
/// are unset.

bool RooAbsArg::getTransientAttribute(const Text_t* name) const
{
  return (_boolAttribTransient.find(name) != _boolAttribTransient.end()) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Register another RooAbsArg as a server to us, ie, declare that
/// we depend on it.
/// \param server The server to be registered.
/// \param valueProp In addition to the basic client-server relationship, declare dependence on the server's value.
/// \param shapeProp In addition to the basic client-server relationship, declare dependence on the server's shape.
/// \param refCount Optionally add with higher reference count (if multiple components depend on it)

void RooAbsArg::addServer(RooAbsArg& server, bool valueProp, bool shapeProp, std::size_t refCount)
{
  if (_prohibitServerRedirect) {
    cxcoutF(LinkStateMgmt) << "RooAbsArg::addServer(" << this << "," << GetName()
            << "): PROHIBITED SERVER ADDITION REQUESTED: adding server " << server.GetName()
            << "(" << &server << ") for " << (valueProp?"value ":"") << (shapeProp?"shape":"") << endl ;
    throw std::logic_error("PROHIBITED SERVER ADDITION REQUESTED in RooAbsArg::addServer");
  }

  cxcoutD(LinkStateMgmt) << "RooAbsArg::addServer(" << this << "," << GetName() << "): adding server " << server.GetName()
          << "(" << &server << ") for " << (valueProp?"value ":"") << (shapeProp?"shape":"") << endl ;

  if (server.operMode()==ADirty && operMode()!=ADirty && valueProp) {
    setOperMode(ADirty) ;
  }


  // LM: use hash tables for larger lists
//  if (_serverList.GetSize() > 999 && _serverList.getHashTableSize() == 0) _serverList.setHashTableSize(1000);
//  if (server._clientList.GetSize() > 999 && server._clientList.getHashTableSize() == 0) server._clientList.setHashTableSize(1000);
//  if (server._clientListValue.GetSize() >  999 && server._clientListValue.getHashTableSize() == 0) server._clientListValue.setHashTableSize(1000);

  // Add server link to given server
  _serverList.Add(&server, refCount) ;

  server._clientList.Add(this, refCount);
  if (valueProp) server._clientListValue.Add(this, refCount);
  if (shapeProp) server._clientListShape.Add(this, refCount);
}



////////////////////////////////////////////////////////////////////////////////
/// Register a list of RooAbsArg as servers to us by calling
/// addServer() for each arg in the list

void RooAbsArg::addServerList(RooAbsCollection& serverList, bool valueProp, bool shapeProp)
{
  _serverList.reserve(_serverList.size() + serverList.size());

  for (const auto arg : serverList) {
    addServer(*arg,valueProp,shapeProp) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Unregister another RooAbsArg as a server to us, ie, declare that
/// we no longer depend on its value and shape.

void RooAbsArg::removeServer(RooAbsArg& server, bool force)
{
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
  _serverList.Remove(&server, force) ;

  server._clientList.Remove(this, force) ;
  server._clientListValue.Remove(this, force) ;
  server._clientListShape.Remove(this, force) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Replace 'oldServer' with 'newServer'

void RooAbsArg::replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, bool propValue, bool propShape)
{
  Int_t count = _serverList.refCount(&oldServer);
  removeServer(oldServer, true);
  addServer(newServer, propValue, propShape, count);
}


////////////////////////////////////////////////////////////////////////////////
/// Change dirty flag propagation mask for specified server

void RooAbsArg::changeServer(RooAbsArg& server, bool valueProp, bool shapeProp)
{
  if (!_serverList.containsByNamePtr(&server)) {
    coutE(LinkStateMgmt) << "RooAbsArg::changeServer(" << GetName() << "): Server "
    << server.GetName() << " not registered" << endl ;
    return ;
  }

  // This condition should not happen, but check anyway
  if (!server._clientList.containsByNamePtr(this)) {
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



////////////////////////////////////////////////////////////////////////////////
/// Fill supplied list with all leaf nodes of the arg tree, starting with
/// ourself as top node. A leaf node is node that has no servers declared.

void RooAbsArg::leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, bool recurseNonDerived) const
{
  treeNodeServerList(list,arg,false,true,false,recurseNonDerived) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fill supplied list with all branch nodes of the arg tree starting with
/// ourself as top node. A branch node is node that has one or more servers declared.

void RooAbsArg::branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, bool recurseNonDerived) const
{
  treeNodeServerList(list,arg,true,false,false,recurseNonDerived) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Fill supplied list with nodes of the arg tree, following all server links,
/// starting with ourself as top node.
/// \param[in] list Output list
/// \param[in] arg Start searching at this element of the tree.
/// \param[in] doBranch Add branch nodes to the list.
/// \param[in] doLeaf Add leaf nodes to the list.
/// \param[in] valueOnly Only check if an element is a value server (no shape server).
/// \param[in] recurseFundamental

void RooAbsArg::treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg, bool doBranch, bool doLeaf, bool valueOnly, bool recurseFundamental) const
{
//   if (arg==0) {
//     cout << "treeNodeServerList(" << GetName() << ") doBranch=" << (doBranch?"T":"F") << " doLeaf = " << (doLeaf?"T":"F") << " valueOnly=" << (valueOnly?"T":"F") << endl ;
//   }

  if (!arg) {
    list->reserve(10);
    arg=this ;
  }

  // Decide if to add current node
  if ((doBranch&&doLeaf) ||
      (doBranch&&arg->isDerived()) ||
      (doLeaf&&arg->isFundamental()&&(!(recurseFundamental&&arg->isDerived()))) ||
      (doLeaf && !arg->isFundamental() && !arg->isDerived())) {

    list->add(*arg,true) ;
  }

  // Recurse if current node is derived
  if (arg->isDerived() && (!arg->isFundamental() || recurseFundamental)) {
    for (const auto server : arg->_serverList) {

      // Skip non-value server nodes if requested
      bool isValueSrv = server->_clientListValue.containsByNamePtr(arg);
      if (valueOnly && !isValueSrv) {
        continue ;
      }
      treeNodeServerList(list,server,doBranch,doLeaf,valueOnly,recurseFundamental) ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Create a list of leaf nodes in the arg tree starting with
/// ourself as top node that don't match any of the names of the variable list
/// of the supplied data set (the dependents). The caller of this
/// function is responsible for deleting the returned argset.
/// The complement of this function is getObservables()

RooArgSet* RooAbsArg::getParameters(const RooAbsData* set, bool stripDisconnected) const
{
  return getParameters(set?set->get():0,stripDisconnected) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Add all parameters of the function and its daughters to `params`.
/// \param[in] params Collection that stores all parameters. Add all new parameters to this.
/// \param[in] nset Normalisation set (optional). If a value depends on this set, it's not a parameter.
/// \param[in] stripDisconnected Passed on to getParametersHook().

void RooAbsArg::addParameters(RooAbsCollection& params, const RooArgSet* nset, bool stripDisconnected) const
{

  RooArgSet nodeParamServers;
  std::vector<RooAbsArg*> branchList;
  for (const auto server : _serverList) {
    if (server->isValueServer(*this)) {
      if (server->isFundamental()) {
        if (!nset || !server->dependsOn(*nset)) {
          nodeParamServers.add(*server);
        }
      } else {
        branchList.push_back(server);
      }
    }
  }

  // Allow pdf to strip parameters from list before adding it
  getParametersHook(nset,&nodeParamServers,stripDisconnected) ;

  // Add parameters of this node to the combined list
  params.add(nodeParamServers,true) ;

  // Now recurse into branch servers
  std::sort(branchList.begin(), branchList.end());
  const auto last = std::unique(branchList.begin(), branchList.end());
  for (auto serverIt = branchList.begin(); serverIt < last; ++serverIt) {
    (*serverIt)->addParameters(params, nset);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Obtain an estimate of the number of parameters of the function and its daughters.
/// Calling `addParamters` for large functions (NLL) can cause many reallocations of
/// `params` due to the recursive behaviour. This utility function aims to pre-compute
/// the total number of parameters, so that enough memory is reserved.
/// The estimate is not fully accurate (overestimate) as there is no equivalent to `getParametersHook`.
/// \param[in] nset Normalisation set (optional). If a value depends on this set, it's not a parameter.

std::size_t RooAbsArg::getParametersSizeEstimate(const RooArgSet* nset) const
{

  std::size_t res = 0;
  std::vector<RooAbsArg*> branchList;
  for (const auto server : _serverList) {
    if (server->isValueServer(*this)) {
      if (server->isFundamental()) {
        if (!nset || !server->dependsOn(*nset)) {
          res++;
        }
      } else {
        branchList.push_back(server);
      }
    }
  }

  // Now recurse into branch servers
  std::sort(branchList.begin(), branchList.end());
  const auto last = std::unique(branchList.begin(), branchList.end());
  for (auto serverIt = branchList.begin(); serverIt < last; ++serverIt) {
    res += (*serverIt)->getParametersSizeEstimate(nset);
  }

  return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a list of leaf nodes in the arg tree starting with
/// ourself as top node that don't match any of the names the args in the
/// supplied argset. The caller of this function is responsible
/// for deleting the returned argset. The complement of this function
/// is getObservables().

RooArgSet* RooAbsArg::getParameters(const RooArgSet* observables, bool stripDisconnected) const {
  auto * outputSet = new RooArgSet;
  getParameters(observables, *outputSet, stripDisconnected);
  return outputSet;
}


////////////////////////////////////////////////////////////////////////////////
/// Fills a list with leaf nodes in the arg tree starting with
/// ourself as top node that don't match any of the names the args in the
/// supplied argset. Returns `true` only if something went wrong.
/// The complement of this function is getObservables().
/// \param[in] observables Set of leafs to ignore because they are observables and not parameters.
/// \param[out] outputSet Output set.
/// \param[in] stripDisconnected Allow pdf to strip parameters from list before adding it.

bool RooAbsArg::getParameters(const RooArgSet* observables, RooArgSet& outputSet, bool stripDisconnected) const
{
   using RooHelpers::getColonSeparatedNameString;

   // Check for cached parameter set
   if (_myws) {
      auto nsetObs = getColonSeparatedNameString(observables ? *observables : RooArgSet());
      const RooArgSet *paramSet = _myws->set(Form("CACHE_PARAMS_OF_PDF_%s_FOR_OBS_%s", GetName(), nsetObs.c_str()));
      if (paramSet) {
         outputSet.add(*paramSet);
         return false;
      }
   }

   outputSet.clear();
   outputSet.setName("parameters");

   RooArgList tempList;
   // reserve all memory needed in one go
   tempList.reserve(getParametersSizeEstimate(observables));

   addParameters(tempList, observables, stripDisconnected);

   // The adding from the list to the set has to be silent to not complain
   // about duplicate parameters. After all, it's normal that parameters can
   // appear in sifferent components of the model.
   outputSet.add(tempList, /*silent=*/true);
   outputSet.sort();

   // Cache parameter set
   if (_myws && outputSet.size() > 10) {
      auto nsetObs = getColonSeparatedNameString(observables ? *observables : RooArgSet());
      _myws->defineSetInternal(Form("CACHE_PARAMS_OF_PDF_%s_FOR_OBS_%s", GetName(), nsetObs.c_str()), outputSet);
   }

   return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a list of leaf nodes in the arg tree starting with
/// ourself as top node that match any of the names of the variable list
/// of the supplied data set (the dependents). The caller of this
/// function is responsible for deleting the returned argset.
/// The complement of this function is getParameters().

RooArgSet* RooAbsArg::getObservables(const RooAbsData* set) const
{
  if (!set) return new RooArgSet ;

  return getObservables(set->get()) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a list of leaf nodes in the arg tree starting with
/// ourself as top node that match any of the names the args in the
/// supplied argset. The caller of this function is responsible
/// for deleting the returned argset. The complement of this function
/// is getParameters().

RooArgSet* RooAbsArg::getObservables(const RooArgSet* dataList, bool valueOnly) const
{
  auto depList = new RooArgSet;
  getObservables(dataList, *depList, valueOnly);
  return depList;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a list of leaf nodes in the arg tree starting with
/// ourself as top node that match any of the names the args in the
/// supplied argset.
/// Returns `true` only if something went wrong.
/// The complement of this function is getParameters().
/// \param[in] dataList Set of leaf nodes to match.
/// \param[out] outputSet Output set.
/// \param[in] valueOnly If this parameter is true, we only match leafs that
///                      depend on the value of any arg in `dataList`.

bool RooAbsArg::getObservables(const RooAbsCollection* dataList, RooArgSet& outputSet, bool valueOnly) const
{
  outputSet.clear();
  outputSet.setName("dependents");

  if (!dataList) return false;

  // Make iterator over tree leaf node list
  RooArgSet leafList("leafNodeServerList") ;
  treeNodeServerList(&leafList,0,false,true,valueOnly) ;

  if (valueOnly) {
    for (const auto arg : leafList) {
      if (arg->dependsOnValue(*dataList) && arg->isLValue()) {
        outputSet.add(*arg) ;
      }
    }
  } else {
    for (const auto arg : leafList) {
      if (arg->dependsOn(*dataList) && arg->isLValue()) {
        outputSet.add(*arg) ;
      }
    }
  }

  return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a RooArgSet with all components (branch nodes) of the
/// expression tree headed by this object.
RooArgSet* RooAbsArg::getComponents() const
{
  TString name(GetName()) ;
  name.Append("_components") ;

  RooArgSet* set = new RooArgSet(name) ;
  branchNodeServerList(set) ;

  return set ;
}



////////////////////////////////////////////////////////////////////////////////
/// Overloadable function in which derived classes can implement
/// consistency checks of the variables. If this function returns
/// true, indicating an error, the fitter or generator will abort.

bool RooAbsArg::checkObservables(const RooArgSet*) const
{
  return false ;
}


////////////////////////////////////////////////////////////////////////////////
/// Recursively call checkObservables on all nodes in the expression tree

bool RooAbsArg::recursiveCheckObservables(const RooArgSet* nset) const
{
  RooArgSet nodeList ;
  treeNodeServerList(&nodeList) ;

  bool ret(false) ;
  for(RooAbsArg * arg : nodeList) {
    if (arg->getAttribute("ServerDied")) {
      coutE(LinkStateMgmt) << "RooAbsArg::recursiveCheckObservables(" << GetName() << "): ERROR: one or more servers of node "
            << arg->GetName() << " no longer exists!" << endl ;
      arg->Print("v") ;
      ret = true ;
    }
    ret |= arg->checkObservables(nset) ;
  }

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Test whether we depend on (ie, are served by) any object in the
/// specified collection. Uses the dependsOn(RooAbsArg&) member function.

bool RooAbsArg::dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg, bool valueOnly) const
{
  // Test whether we depend on (ie, are served by) any object in the
  // specified collection. Uses the dependsOn(RooAbsArg&) member function.

  for (auto server : serverList) {
    if (dependsOn(*server,ignoreArg,valueOnly)) {
      return true;
    }
  }
  return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Test whether we depend on (ie, are served by) the specified object.
/// Note that RooAbsArg objects are considered equivalent if they have
/// the same name.

bool RooAbsArg::dependsOn(const RooAbsArg& testArg, const RooAbsArg* ignoreArg, bool valueOnly) const
{
  if (this==ignoreArg) return false ;

  // First check if testArg is self
  //if (!TString(testArg.GetName()).CompareTo(GetName())) return true ;
  if (testArg.namePtr()==namePtr()) return true ;


  // Next test direct dependence
  RooAbsArg* foundServer = findServer(testArg) ;
  if (foundServer) {

    // Return true if valueOnly is FALSE or if server is value server, otherwise keep looking
    if ( !valueOnly || foundServer->isValueServer(*this)) {
      return true ;
    }
  }

  // If not, recurse
  for (const auto server : _serverList) {
    if ( !valueOnly || server->isValueServer(*this)) {
      if (server->dependsOn(testArg,ignoreArg,valueOnly)) {
        return true ;
      }
    }
  }

  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Test if any of the nodes of tree are shared with that of the given tree

bool RooAbsArg::overlaps(const RooAbsArg& testArg, bool valueOnly) const
{
  RooArgSet list("treeNodeList") ;
  treeNodeServerList(&list) ;

  return valueOnly ? testArg.dependsOnValue(list) : testArg.dependsOn(list) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Test if any of the dependents of the arg tree (as determined by getObservables)
/// overlaps with those of the testArg.

bool RooAbsArg::observableOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const
{
  return observableOverlaps(dset->get(),testArg) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Test if any of the dependents of the arg tree (as determined by getObservables)
/// overlaps with those of the testArg.

bool RooAbsArg::observableOverlaps(const RooArgSet* nset, const RooAbsArg& testArg) const
{
  RooArgSet* depList = getObservables(nset) ;
  bool ret = testArg.dependsOn(*depList) ;
  delete depList ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Mark this object as having changed its value, and propagate this status
/// change to all of our clients. If the object is not in automatic dirty
/// state propagation mode, this call has no effect.

void RooAbsArg::setValueDirty(const RooAbsArg* source)
{
  if (_operMode!=Auto || _inhibitDirty) return ;

  // Handle no-propagation scenarios first
  if (_clientListValue.empty()) {
    _valueDirty = true ;
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

  _valueDirty = true ;


  for (auto client : _clientListValue) {
    client->setValueDirty(source) ;
  }


}


////////////////////////////////////////////////////////////////////////////////
/// Mark this object as having changed its shape, and propagate this status
/// change to all of our clients.

void RooAbsArg::setShapeDirty(const RooAbsArg* source)
{
  if (_verboseDirty) {
    cxcoutD(LinkStateMgmt) << "RooAbsArg::setShapeDirty(" << GetName()
            << "): dirty flag " << (_shapeDirty?"already ":"") << "raised" << endl ;
  }

  if (_clientListShape.empty()) {
    _shapeDirty = true ;
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
  _shapeDirty=true ;

  for (auto client : _clientListShape) {
    client->setShapeDirty(source) ;
    client->setValueDirty(source) ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Replace all direct servers of this object with the new servers in `newServerList`.
/// This substitutes objects that we receive values from with new objects that have the same name.
/// \see recursiveRedirectServers() Use recursive version if servers that are only indirectly serving this object should be replaced as well.
/// \see redirectServers() If only the direct servers of an object need to be replaced.
///
/// Note that changing the types of objects is generally allowed, but can be wrong if the interface of an object changes.
/// For example, one can reparametrise a model by substituting a variable with a function:
/// \f[
///   f(x\, |\, a) = a \cdot x \rightarrow f(x\, |\, b) = (2.1 \cdot b) \cdot x
/// \f]
/// If an object, however, expects a PDF, and this is substituted with a function that isn't normalised, wrong results might be obtained
/// or it might even crash the program. The types of the objects being substituted are not checked.
///
/// \param[in] newSetOrig Set of new servers that should be used instead of the current servers.
/// \param[in] mustReplaceAll A warning is printed and error status is returned if not all servers could be
/// substituted successfully.
/// \param[in] nameChange If false, an object named "x" is only replaced with an object also named "x" in `newSetOrig`.
/// If the object in `newSet` is called differently, set `nameChange` to true and use setAttribute() on the x object:
/// ```
/// objectToReplaceX.setAttribute("ORIGNAME:x")
/// ```
/// Now, the renamed object will be selected based on the attribute "ORIGNAME:<name>".
/// \param[in] isRecursionStep Internal switch used when called from recursiveRedirectServers().
bool RooAbsArg::redirectServers(const RooAbsCollection& newSetOrig, bool mustReplaceAll, bool nameChange, bool isRecursionStep)
{
  // Trivial case, no servers
  if (_serverList.empty()) return false ;
  if (newSetOrig.empty()) return false ;

  // Strip any non-matching removal nodes from newSetOrig
  RooAbsCollection* newSet ;

  if (nameChange) {
    newSet = new RooArgSet ;
    for (auto arg : newSetOrig) {

      if (string("REMOVAL_DUMMY")==arg->GetName()) {

        if (arg->getAttribute("REMOVE_ALL")) {
          newSet->add(*arg) ;
        } else if (arg->getAttribute(Form("REMOVE_FROM_%s",getStringAttribute("ORIGNAME")))) {
          newSet->add(*arg) ;
        }
      } else {
        newSet->add(*arg) ;
      }
    }
  } else {
    newSet = (RooAbsCollection*) &newSetOrig ;
  }

  // Replace current servers with new servers with the same name from the given list
  bool ret(false) ;

  //Copy original server list to not confuse the iterator while deleting
  std::vector<RooAbsArg*> origServerList, origServerValue, origServerShape;
  auto origSize = _serverList.size();
  origServerList.reserve(origSize);
  origServerValue.reserve(origSize);

  for (const auto oldServer : _serverList) {
    origServerList.push_back(oldServer) ;

    // Retrieve server side link state information
    if (oldServer->_clientListValue.containsByNamePtr(this)) {
      origServerValue.push_back(oldServer) ;
    }
    if (oldServer->_clientListShape.containsByNamePtr(this)) {
      origServerShape.push_back(oldServer) ;
    }
  }

  // Delete all previously registered servers
  for (auto oldServer : origServerList) {

    RooAbsArg * newServer= oldServer->findNewServer(*newSet, nameChange);

    if (newServer && _verboseDirty) {
      cxcoutD(LinkStateMgmt) << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName()
                  << " redirected from " << oldServer << " to " << newServer << endl ;
    }

    if (!newServer) {
      if (mustReplaceAll) {
        coutE(LinkStateMgmt) << "RooAbsArg::redirectServers(" << (void*)this << "," << GetName() << "): server " << oldServer->GetName()
                    << " (" << (void*)oldServer << ") not redirected" << (nameChange?"[nameChange]":"") << endl ;
        ret = true ;
      }
      continue ;
    }

    auto findByNamePtr = [&oldServer](const RooAbsArg * item) {
      return oldServer->namePtr() == item->namePtr();
    };
    bool propValue = std::any_of(origServerValue.begin(), origServerValue.end(), findByNamePtr);
    bool propShape = std::any_of(origServerShape.begin(), origServerShape.end(), findByNamePtr);

    if (newServer != this) {
      replaceServer(*oldServer,*newServer,propValue,propShape) ;
    }
  }


  setValueDirty() ;
  setShapeDirty() ;

  // Process the proxies
  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!p) continue ;
    bool ret2 = p->changePointer(*newSet,nameChange,false) ;

    if (mustReplaceAll && !ret2) {
      auto ap = dynamic_cast<const RooArgProxy*>(p);
      coutE(LinkStateMgmt) << "RooAbsArg::redirectServers(" << GetName()
              << "): ERROR, proxy '" << p->name()
              << "' with arg '" << (ap ? ap->absArg()->GetName() : "<could not cast>") << "' could not be adjusted" << endl;
      ret = true ;
    }
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

////////////////////////////////////////////////////////////////////////////////
/// Find the new server in the specified set that matches the old server.
///
/// \param[in] newSet Search this set by name for a new server.
/// \param[in] nameChange If true, search for an item with the bool attribute "ORIGNAME:<oldName>" set.
/// Use `<object>.setAttribute("ORIGNAME:<oldName>")` to set this attribute.
/// \return Pointer to the new server or `nullptr` if there's no unique match.
RooAbsArg *RooAbsArg::findNewServer(const RooAbsCollection &newSet, bool nameChange) const
{
  RooAbsArg *newServer = 0;
  if (!nameChange) {
    newServer = newSet.find(*this) ;
  }
  else {
    // Name changing server redirect:
    // use 'ORIGNAME:<oldName>' attribute instead of name of new server
    TString nameAttrib("ORIGNAME:") ;
    nameAttrib.Append(GetName()) ;

    RooArgSet* tmp = (RooArgSet*) newSet.selectByAttrib(nameAttrib,true) ;
    if(0 != tmp) {

      // Check if any match was found
      if (tmp->empty()) {
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


////////////////////////////////////////////////////////////////////////////////
/// Recursively replace all servers with the new servers in `newSet`.
/// This substitutes objects that we receive values from (also indirectly through other objects) with new objects that have the same name.
///
/// *Copied from redirectServers:*
///
/// \copydetails RooAbsArg::redirectServers
/// \param newSet Roo collection
/// \param recurseInNewSet be recursive
bool RooAbsArg::recursiveRedirectServers(const RooAbsCollection& newSet, bool mustReplaceAll, bool nameChange, bool recurseInNewSet)
{
  // Cyclic recursion protection
  static std::set<const RooAbsArg*> callStack;
  {
    std::set<const RooAbsArg*>::iterator it = callStack.lower_bound(this);
    if (it != callStack.end() && this == *it) {
      return false;
    } else {
      callStack.insert(it, this);
    }
  }

  // Do not recurse into newset if not so specified
//   if (!recurseInNewSet && newSet.contains(*this)) {
//     return false ;
//   }


  // Apply the redirectServers function recursively on all branch nodes in this argument tree.
  bool ret(false) ;

  cxcoutD(LinkStateMgmt) << "RooAbsArg::recursiveRedirectServers(" << this << "," << GetName() << ") newSet = " << newSet << " mustReplaceAll = "
          << (mustReplaceAll?"T":"F") << " nameChange = " << (nameChange?"T":"F") << " recurseInNewSet = " << (recurseInNewSet?"T":"F") << endl ;

  // Do redirect on self (identify operation as recursion step)
  ret |= redirectServers(newSet,mustReplaceAll,nameChange,true) ;

  // Do redirect on servers
  for (const auto server : _serverList){
    ret |= server->recursiveRedirectServers(newSet,mustReplaceAll,nameChange,recurseInNewSet) ;
  }

  callStack.erase(this);
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Register an RooArgProxy in the proxy list. This function is called by owned
/// proxies upon creation. After registration, this arg wil forward pointer
/// changes from serverRedirects and updates in cached normalization sets
/// to the proxies immediately after they occur. The proxied argument is
/// also added as value and/or shape server

void RooAbsArg::registerProxy(RooArgProxy& proxy)
{
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
  _proxyListCache.isDirty = true;
}


////////////////////////////////////////////////////////////////////////////////
/// Remove proxy from proxy list. This functions is called by owned proxies
/// upon their destruction.

void RooAbsArg::unRegisterProxy(RooArgProxy& proxy)
{
  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
  _proxyListCache.isDirty = true;
}



////////////////////////////////////////////////////////////////////////////////
/// Register an RooSetProxy in the proxy list. This function is called by owned
/// proxies upon creation. After registration, this arg wil forward pointer
/// changes from serverRedirects and updates in cached normalization sets
/// to the proxies immediately after they occur.

void RooAbsArg::registerProxy(RooSetProxy& proxy)
{
  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    coutE(LinkStateMgmt) << "RooAbsArg::registerProxy(" << GetName() << "): proxy named "
          << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  _proxyList.Add(&proxy) ;
  _proxyListCache.isDirty = true;
}



////////////////////////////////////////////////////////////////////////////////
/// Remove proxy from proxy list. This functions is called by owned proxies
/// upon their destruction.

void RooAbsArg::unRegisterProxy(RooSetProxy& proxy)
{
  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
  _proxyListCache.isDirty = true;
}



////////////////////////////////////////////////////////////////////////////////
/// Register an RooListProxy in the proxy list. This function is called by owned
/// proxies upon creation. After registration, this arg wil forward pointer
/// changes from serverRedirects and updates in cached normalization sets
/// to the proxies immediately after they occur.

void RooAbsArg::registerProxy(RooListProxy& proxy)
{
  // Every proxy can be registered only once
  if (_proxyList.FindObject(&proxy)) {
    coutE(LinkStateMgmt) << "RooAbsArg::registerProxy(" << GetName() << "): proxy named "
          << proxy.GetName() << " already registered" << endl ;
    return ;
  }

  // Register proxy itself
  Int_t nProxyOld = _proxyList.GetEntries() ;
  _proxyList.Add(&proxy) ;
  _proxyListCache.isDirty = true;
  if (_proxyList.GetEntries()!=nProxyOld+1) {
    cout << "RooAbsArg::registerProxy(" << GetName() << ") proxy registration failure! nold=" << nProxyOld << " nnew=" << _proxyList.GetEntries() << endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Remove proxy from proxy list. This functions is called by owned proxies
/// upon their destruction.

void RooAbsArg::unRegisterProxy(RooListProxy& proxy)
{
  _proxyList.Remove(&proxy) ;
  _proxyList.Compress() ;
  _proxyListCache.isDirty = true;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the nth proxy from the proxy list.

RooAbsProxy* RooAbsArg::getProxy(Int_t index) const
{
  // Cross cast: proxy list returns TObject base pointer, we need
  // a RooAbsProxy base pointer. C++ standard requires
  // a dynamic_cast for this.
  return dynamic_cast<RooAbsProxy*> (_proxyList.At(index)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the number of registered proxies.

Int_t RooAbsArg::numProxies() const
{
   return _proxyList.GetEntriesFast();
}



////////////////////////////////////////////////////////////////////////////////
/// Forward a change in the cached normalization argset
/// to all the registered proxies.

void RooAbsArg::setProxyNormSet(const RooArgSet* nset)
{
  if (_proxyListCache.isDirty) {
    // First time we loop over proxies: cache the results to avoid future
    // costly dynamic_casts
    _proxyListCache.cache.clear();
    for (int i=0 ; i<numProxies() ; i++) {
      RooAbsProxy* p = getProxy(i) ;
      if (!p) continue ;
      _proxyListCache.cache.push_back(p);
    }
    _proxyListCache.isDirty = false;
  }

  for ( auto& p : _proxyListCache.cache ) {
    p->changeNormSet(nset);
  }

  // If the proxy normSet changed, we also have to set our value dirty flag.
  // Otherwise, value for the new normalization set might not get recomputed!
  setValueDirty();
}



////////////////////////////////////////////////////////////////////////////////
/// Overloadable function for derived classes to implement
/// attachment as branch to a TTree

void RooAbsArg::attachToTree(TTree& ,Int_t)
{
  coutE(Contents) << "RooAbsArg::attachToTree(" << GetName()
        << "): Cannot be attached to a TTree" << endl ;
}



////////////////////////////////////////////////////////////////////////////////
/// WVE (08/21/01) Probably obsolete now

bool RooAbsArg::isValid() const
{
  return true ;
}




////////////////////////////////////////////////////////////////////////////////
/// Print object name

void RooAbsArg::printName(ostream& os) const
{
  os << GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print object title

void RooAbsArg::printTitle(ostream& os) const
{
  os << GetTitle() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print object class name

void RooAbsArg::printClassName(ostream& os) const
{
  os << ClassName() ;
}


void RooAbsArg::printAddress(ostream& os) const
{
  // Print addrss of this RooAbsArg
  os << this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print object arguments, ie its proxies

void RooAbsArg::printArgs(ostream& os) const
{
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



////////////////////////////////////////////////////////////////////////////////
/// Define default contents to print

Int_t RooAbsArg::defaultPrintContents(Option_t* /*opt*/) const
{
  return kName|kClassName|kValue|kArgs ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement multi-line detailed printing

void RooAbsArg::printMultiline(ostream& os, Int_t /*contents*/, bool /*verbose*/, TString indent) const
{
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
  for (const auto client : _clientList) {
    os << indent << "    (" << (void*)client  << ","
       << (_clientListValue.containsByNamePtr(client)?"V":"-")
       << (_clientListShape.containsByNamePtr(client)?"S":"-")
       << ") " ;
    client->printStream(os,kClassName|kTitle|kName,kSingleLine);
  }

  // server list
  os << indent << "  Servers: " << endl;
  for (const auto server : _serverList) {
    os << indent << "    (" << (void*)server << ","
       << (server->_clientListValue.containsByNamePtr(this)?"V":"-")
       << (server->_clientListShape.containsByNamePtr(this)?"S":"-")
       << ") " ;
    server->printStream(os,kClassName|kName|kTitle,kSingleLine);
  }

  // proxy list
  os << indent << "  Proxies: " << std::endl;
  for (int i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* proxy=getProxy(i) ;
    if (!proxy) continue ;
    os << indent << "    " << proxy->name() << " -> " ;
    if(auto * argProxy = dynamic_cast<RooArgProxy*>(proxy)) {
      if (RooAbsArg* parg = argProxy->absArg()) {
        parg->printStream(os,kName,kSingleLine) ;
      } else {
        os << " (empty)" << std::endl;
      }
      // If a RooAbsProxy is not a RooArgProxy, it is a RooSetProxy or a
      // RooListProxy. However, they are treated the same in this function, so
      // we try the dynamic cast to their common base class, RooAbsCollection.
    } else if(auto * collProxy = dynamic_cast<RooAbsCollection*>(proxy)) {
      os << std::endl;
      TString moreIndent(indent) ;
      moreIndent.Append("    ") ;
      collProxy->printStream(os,kName,kStandard,moreIndent.Data());
    } else {
      throw std::runtime_error("Unsupported proxy type.");
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Print object tree structure

void RooAbsArg::printTree(ostream& os, TString /*indent*/) const
{
  const_cast<RooAbsArg*>(this)->printCompactTree(os) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Ostream operator

ostream& operator<<(ostream& os, RooAbsArg const& arg)
{
  arg.writeToStream(os,true) ;
  return os ;
}

////////////////////////////////////////////////////////////////////////////////
/// Istream operator

istream& operator>>(istream& is, RooAbsArg &arg)
{
  arg.readFromStream(is,true,false) ;
  return is ;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the attribute list

void RooAbsArg::printAttribList(ostream& os) const
{
  set<string>::const_iterator iter = _boolAttrib.begin() ;
  bool first(true) ;
  while (iter != _boolAttrib.end()) {
    os << (first?" [":",") << *iter ;
    first=false ;
    ++iter ;
  }
  if (!first) os << "] " ;
}


////////////////////////////////////////////////////////////////////////////////
/// Bind this node to objects in `set`.
/// Search the set for objects that have the same name as our servers, and
/// attach ourselves to those. After this operation, this node is computing its
/// values based on the new servers. This can be used to e.g. read values from
// a dataset.


void RooAbsArg::attachArgs(const RooAbsCollection &set)
{
  RooArgSet branches;
  branchNodeServerList(&branches,0,true);

  for(auto const& branch : branches) {
    branch->redirectServers(set,false,false);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Replace server nodes with names matching the dataset variable names
/// with those data set variables, making this PDF directly dependent on the dataset.

void RooAbsArg::attachDataSet(const RooAbsData &data)
{
  attachArgs(*data.get());
}


////////////////////////////////////////////////////////////////////////////////
/// Replace server nodes with names matching the dataset variable names
/// with those data set variables, making this PDF directly dependent on the dataset

void RooAbsArg::attachDataStore(const RooAbsDataStore &dstore)
{
  attachArgs(*dstore.get());
}


////////////////////////////////////////////////////////////////////////////////
/// Utility function used by TCollection::Sort to compare contained TObjects
/// We implement comparison by name, resulting in alphabetical sorting by object name.

Int_t RooAbsArg::Compare(const TObject* other) const
{
  return strcmp(GetName(),other->GetName()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print information about current value dirty state information.
/// If depth flag is true, information is recursively printed for
/// all nodes in this arg tree.

void RooAbsArg::printDirty(bool depth) const
{
  if (depth) {

    RooArgSet branchList ;
    branchNodeServerList(&branchList) ;
    for(RooAbsArg * branch : branchList) {
      branch->printDirty(false) ;
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


////////////////////////////////////////////////////////////////////////////////
/// Activate cache mode optimization with given definition of observables.
/// The cache operation mode of all objects in the expression tree will
/// modified such that all nodes that depend directly or indirectly on
/// any of the listed observables will be set to ADirty, as they are
/// expected to change every time. This save change tracking overhead for
/// nodes that are a priori known to change every time

void RooAbsArg::optimizeCacheMode(const RooArgSet& observables)
{
  RooLinkedList proc;
  RooArgSet opt ;
  optimizeCacheMode(observables,opt,proc) ;

  coutI(Optimization) << "RooAbsArg::optimizeCacheMode(" << GetName() << ") nodes " << opt << " depend on observables, "
         << "changing cache operation mode from change tracking to unconditional evaluation" << endl ;
}


////////////////////////////////////////////////////////////////////////////////
/// Activate cache mode optimization with given definition of observables.
/// The cache operation mode of all objects in the expression tree will
/// modified such that all nodes that depend directly or indirectly on
/// any of the listed observables will be set to ADirty, as they are
/// expected to change every time. This save change tracking overhead for
/// nodes that are a priori known to change every time

void RooAbsArg::optimizeCacheMode(const RooArgSet& observables, RooArgSet& optimizedNodes, RooLinkedList& processedNodes)
{
  // Optimization applies only to branch nodes, not to leaf nodes
  if (!isDerived()) {
    return ;
  }


  // Terminate call if this node was already processed (tree structure may be cyclical)
  // LM : RooLinkedList::findArg looks by name and not but by object pointer,
  //  should one use RooLinkedList::FindObject (look byt pointer) instead of findArg when
  // tree contains nodes with the same name ?
  // Add an info message if the require node does not exist but a different node already exists with same name

  if (processedNodes.FindObject(this))
     return;

  // check if findArgs returns something different (i.e. a different node with same name) when
  // this node has not been processed (FindObject returns a null pointer)
  auto obj = processedNodes.findArg(this);
  assert(obj != this); // obj == this cannot happen
  if (obj)
     // here for nodes with duplicate names
     cxcoutI(Optimization) << "RooAbsArg::optimizeCacheMode(" << GetName()
                           << " node " << this << " exists already as " << obj << " but with the SAME name !" << endl;

  processedNodes.Add(this);

  // Set cache mode operator to 'AlwaysDirty' if we depend on any of the given observables
  if (dependsOnValue(observables)) {

    if (dynamic_cast<RooRealIntegral*>(this)) {
      cxcoutI(Integration) << "RooAbsArg::optimizeCacheMode(" << GetName() << ") integral depends on value of one or more observables and will be evaluated for every event" << endl ;
    }
    optimizedNodes.add(*this,true) ;
    if (operMode()==AClean) {
    } else {
      setOperMode(ADirty,true) ; // WVE propagate flag recursively to top of tree
    }
  } else {
  }
  // Process any RooAbsArgs contained in any of the caches of this object
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->optimizeCacheMode(observables,optimizedNodes,processedNodes) ;
  }

  // Forward calls to all servers
  for (const auto server : _serverList) {
    server->optimizeCacheMode(observables,optimizedNodes,processedNodes) ;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Find branch nodes with all-constant parameters, and add them to the list of
/// nodes that can be cached with a dataset in a test statistic calculation

bool RooAbsArg::findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList)
{
  RooLinkedList proc ;
  bool ret = findConstantNodes(observables,cacheList,proc) ;

  // If node can be optimized and hasn't been identified yet, add it to the list
  coutI(Optimization) << "RooAbsArg::findConstantNodes(" << GetName() << "): components "
         << cacheList << " depend exclusively on constant parameters and will be precalculated and cached" << endl ;

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find branch nodes with all-constant parameters, and add them to the list of
/// nodes that can be cached with a dataset in a test statistic calculation

bool RooAbsArg::findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList, RooLinkedList& processedNodes)
{
  // Caching only applies to branch nodes
  if (!isDerived()) {
    return false;
  }

  // Terminate call if this node was already processed (tree structure may be cyclical)
  if (processedNodes.findArg(this)) {
    return false ;
  } else {
    processedNodes.Add(this) ;
  }

  // Check if node depends on any non-constant parameter
  bool canOpt(true) ;
  RooArgSet* paramSet = getParameters(observables) ;
  for(RooAbsArg * param : *paramSet) {
    if (!param->isConstant()) {
      canOpt=false ;
      break ;
    }
  }
  delete paramSet ;


  if (getAttribute("NeverConstant")) {
    canOpt = false ;
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
      cacheList.add(*this,false) ;
    }
  }

  if (!canOpt) {

    // If not, see if next level down can be cached
    for (const auto server : _serverList) {
      if (server->isDerived()) {
        server->findConstantNodes(observables,cacheList,processedNodes) ;
      }
    }
  }

  // Forward call to all cached contained in current object
  for (Int_t i=0 ;i<numCaches() ; i++) {
    getCache(i)->findConstantNodes(observables,cacheList,processedNodes) ;
  }

  return false ;
}




////////////////////////////////////////////////////////////////////////////////
/// Interface function signaling a request to perform constant term
/// optimization. This default implementation takes no action other than to
/// forward the calls to all servers

void RooAbsArg::constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt)
{
  for (const auto server : _serverList) {
    server->constOptimizeTestStatistic(opcode,doAlsoTrackingOpt) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Change cache operation mode to given mode. If recurseAdirty
/// is true, then a mode change to AlwaysDirty will automatically
/// be propagated recursively to all client nodes

void RooAbsArg::setOperMode(OperMode mode, bool recurseADirty)
{
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
    for (auto clientV : _clientListValue) {
      clientV->setOperMode(mode) ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Print tree structure of expression tree on stdout, or to file if filename is specified.
/// If namePat is not "*", only nodes with names matching the pattern will be printed.
/// The client argument is used in recursive calls to properly display the value or shape nature
/// of the client-server links. It should be zero in calls initiated by users.

void RooAbsArg::printCompactTree(const char* indent, const char* filename, const char* namePat, RooAbsArg* client)
{
  if (filename) {
    ofstream ofs(filename) ;
    printCompactTree(ofs,indent,namePat,client) ;
  } else {
    printCompactTree(cout,indent,namePat,client) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Print tree structure of expression tree on given ostream.
/// If namePat is not "*", only nodes with names matching the pattern will be printed.
/// The client argument is used in recursive calls to properly display the value or shape nature
/// of the client-server links. It should be zero in calls initiated by users.

void RooAbsArg::printCompactTree(ostream& os, const char* indent, const char* namePat, RooAbsArg* client)
{
  if ( !namePat || TString(GetName()).Contains(namePat)) {
    os << indent << this ;
    if (client) {
      os << "/" ;
      if (isValueServer(*client)) os << "V" ; else os << "-" ;
      if (isShapeServer(*client)) os << "S" ; else os << "-" ;
    }
    os << " " ;

    os << ClassName() << "::" << GetName() <<  " = " ;
    printValue(os) ;

    if (!_serverList.empty()) {
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
  for (const auto arg : _serverList) {
    arg->printCompactTree(os,indent2,namePat,this) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Print tree structure of expression tree on given ostream, only branch nodes are printed.
/// Lead nodes (variables) will not be shown
///
/// If namePat is not "*", only nodes with names matching the pattern will be printed.

void RooAbsArg::printComponentTree(const char* indent, const char* namePat, Int_t nLevel)
{
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
  for (const auto arg : _serverList) {
    arg->printComponentTree(indent2.Data(),namePat,nLevel-1) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a mangled name from the actual name that
/// is free of any math symbols that might be interpreted by TTree

TString RooAbsArg::cleanBranchName() const
{
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

  return cleanName;
}


////////////////////////////////////////////////////////////////////////////////
/// Hook function interface for object to insert additional information
/// when printed in the context of a tree structure. This default
/// implementation prints nothing

void RooAbsArg::printCompactTreeHook(ostream&, const char *)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Register RooAbsCache with this object. This function is called
/// by RooAbsCache constructors for objects that are a datamember
/// of this RooAbsArg. By registering itself the RooAbsArg is aware
/// of all its cache data members and will forward server change
/// and cache mode change calls to the cache objects, which in turn
/// can forward them their contents

void RooAbsArg::registerCache(RooAbsCache& cache)
{
  _cacheList.push_back(&cache) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Unregister a RooAbsCache. Called from the RooAbsCache destructor

void RooAbsArg::unRegisterCache(RooAbsCache& cache)
{
  _cacheList.erase(std::remove(_cacheList.begin(), _cacheList.end(), &cache),
     _cacheList.end());
}


////////////////////////////////////////////////////////////////////////////////
/// Return number of registered caches

Int_t RooAbsArg::numCaches() const
{
  return _cacheList.size() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return registered cache object by index

RooAbsCache* RooAbsArg::getCache(Int_t index) const
{
  return _cacheList[index] ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return RooArgSet with all variables (tree leaf nodes of expresssion tree)

RooArgSet* RooAbsArg::getVariables(bool stripDisconnected) const
{
  return getParameters(RooArgSet(),stripDisconnected) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return ancestors in cloning chain of this RooAbsArg. NOTE: Returned pointers
/// are not guaranteed to be 'live', so do not dereference without proper caution

RooLinkedList RooAbsArg::getCloningAncestors() const
{
  RooLinkedList retVal ;

  set<string>::const_iterator iter= _boolAttrib.begin() ;
  while(iter != _boolAttrib.end()) {
    if (TString(*iter).BeginsWith("CloneOf(")) {
      char buf[128] ;
      strlcpy(buf,iter->c_str(),128) ;
      strtok(buf,"(") ;
      char* ptrToken = strtok(0,")") ;
      RooAbsArg* ptr = (RooAbsArg*) strtoll(ptrToken,0,16) ;
      retVal.Add(ptr) ;
    }
  }

  return retVal ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a GraphViz .dot file visualizing the expression tree headed by
/// this RooAbsArg object. Use the GraphViz tool suite to make e.g. a gif
/// or ps file from the .dot file.
/// If a node derives from RooAbsReal, its current (unnormalised) value is
/// printed as well.
///
/// Based on concept developed by Kyle Cranmer.

void RooAbsArg::graphVizTree(const char* fileName, const char* delimiter, bool useTitle, bool useLatex)
{
  ofstream ofs(fileName) ;
  if (!ofs) {
    coutE(InputArguments) << "RooAbsArg::graphVizTree() ERROR: Cannot open graphViz output file with name " << fileName << endl ;
    return ;
  }
  graphVizTree(ofs, delimiter, useTitle, useLatex) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Write the GraphViz representation of the expression tree headed by
/// this RooAbsArg object to the given ostream.
/// If a node derives from RooAbsReal, its current (unnormalised) value is
/// printed as well.
///
/// Based on concept developed by Kyle Cranmer.

void RooAbsArg::graphVizTree(ostream& os, const char* delimiter, bool useTitle, bool useLatex)
{
  if (!os) {
    coutE(InputArguments) << "RooAbsArg::graphVizTree() ERROR: output stream provided as input argument is in invalid state" << endl ;
  }

  // silent warning messages coming when evaluating a RooAddPdf without a normalization set
  RooHelpers::LocalChangeMsgLevel locmsg(RooFit::WARNING, 0u, RooFit::Eval, false);

  // Write header
  os << "digraph \"" << GetName() << "\"{" << endl ;

  // First list all the tree nodes
  RooArgSet nodeSet ;
  treeNodeServerList(&nodeSet) ;

  // iterate over nodes
  for(RooAbsArg * node : nodeSet) {
    string nodeName = node->GetName();
    string nodeTitle = node->GetTitle();
    string nodeLabel = (useTitle && !nodeTitle.empty()) ? nodeTitle : nodeName;

    // if using latex, replace ROOT's # with normal latex backslash
    string::size_type position = nodeLabel.find("#") ;
    while(useLatex && position!=nodeLabel.npos){
      nodeLabel.replace(position, 1, "\\");
    }

    string typeFormat = "\\texttt{";
    string nodeType = (useLatex) ? typeFormat+node->ClassName()+"}" : node->ClassName();

    if (auto realNode = dynamic_cast<RooAbsReal*>(node)) {
      nodeLabel += delimiter + std::to_string(realNode->getVal());
    }

    os << "\"" << nodeName << "\" [ color=" << (node->isFundamental()?"blue":"red")
       << ", label=\"" << nodeType << delimiter << nodeLabel << "\"];" << endl ;

  }

  // Get set of all server links
  set<pair<RooAbsArg*,RooAbsArg*> > links ;
  graphVizAddConnections(links) ;

  // And write them out
  for(auto const& link : links) {
    os << "\"" << link.first->GetName() << "\" -> \"" << link.second->GetName() << "\";" << endl ;
  }

  // Write trailer
  os << "}" << endl ;

}

////////////////////////////////////////////////////////////////////////////////
/// Utility function that inserts all point-to-point client-server connections
/// between any two RooAbsArgs in the expression tree headed by this object
/// in the linkSet argument.

void RooAbsArg::graphVizAddConnections(set<pair<RooAbsArg*,RooAbsArg*> >& linkSet)
{
  for (const auto server : _serverList) {
    linkSet.insert(make_pair(this,server)) ;
    server->graphVizAddConnections(linkSet) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Take ownership of the contents of 'comps'.

bool RooAbsArg::addOwnedComponents(const RooAbsCollection& comps)
{
  if (!_ownedComponents) {
    _ownedComponents = new RooArgSet("owned components") ;
  }
  return _ownedComponents->addOwned(comps) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Take ownership of the contents of 'comps'. Different from the overload that
/// taked the RooArgSet by `const&`, this version can also take an owning
/// RooArgSet without error, because the ownership will not be ambiguous afterwards.

bool RooAbsArg::addOwnedComponents(RooAbsCollection&& comps)
{
  if (!_ownedComponents) {
    _ownedComponents = new RooArgSet("owned components") ;
  }
  return _ownedComponents->addOwned(std::move(comps)) ;
}


////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooAbsArg::addOwnedComponents(RooAbsCollection&& comps)

bool RooAbsArg::addOwnedComponents(RooArgList&& comps) {
  return addOwnedComponents(static_cast<RooAbsCollection&&>(std::move(comps)));
}


////////////////////////////////////////////////////////////////////////////////
/// Clone tree expression of objects. All tree nodes will be owned by
/// the head node return by cloneTree()

RooAbsArg* RooAbsArg::cloneTree(const char* newname) const
{
  // Clone tree using snapshot
  RooArgSet clonedNodes;
  RooArgSet(*this).snapshot(clonedNodes, true);

  // Find the head node in the cloneSet
  RooAbsArg* head = clonedNodes.find(*this) ;
  assert(head);

  // We better to release the ownership before removing the "head". Otherwise,
  // "head" might also be deleted as the clonedNodes collection owns it.
  // (Actually this does not happen because even an owning collection doesn't
  // delete the element when removed by pointer lookup, but it's better not to
  // rely on this unexpected fact).
  clonedNodes.releaseOwnership();

  // Remove the head node from the cloneSet
  // To release it from the set ownership
  clonedNodes.remove(*head) ;

  // Add the set as owned component of the head
  head->addOwnedComponents(std::move(clonedNodes)) ;

  // Adjust name of head node if requested
  if (newname) {
    head->TNamed::SetName(newname) ;
    head->_namePtr = RooNameReg::instance().constPtr(newname) ;
  }

  // Return the head
  return head ;
}



////////////////////////////////////////////////////////////////////////////////

void RooAbsArg::attachToStore(RooAbsDataStore& store)
{
  if (dynamic_cast<RooTreeDataStore*>(&store)) {
    attachToTree(((RooTreeDataStore&)store).tree()) ;
  } else if (dynamic_cast<RooVectorDataStore*>(&store)) {
    attachToVStore((RooVectorDataStore&)store) ;
  }
}



////////////////////////////////////////////////////////////////////////////////

RooExpensiveObjectCache& RooAbsArg::expensiveObjectCache() const
{
  if (_eocache) {
    return *_eocache ;
  } else {
    return RooExpensiveObjectCache::instance() ;
  }
}


////////////////////////////////////////////////////////////////////////////////

const char* RooAbsArg::aggregateCacheUniqueSuffix() const
{
  string suffix ;

  RooArgSet branches ;
  branchNodeServerList(&branches) ;
  for(RooAbsArg * arg : branches) {
    const char* tmp = arg->cacheUniqueSuffix() ;
    if (tmp) suffix += tmp ;
  }
  return Form("%s",suffix.c_str()) ;
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsArg::wireAllCaches()
{
  RooArgSet branches ;
  branchNodeServerList(&branches) ;
  for(auto const& arg : branches) {
    for (auto const& arg2 : arg->_cacheList) {
      arg2->wireCache() ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooAbsArg::SetName(const char* name)
{
  TNamed::SetName(name) ;
  auto newPtr = RooNameReg::instance().constPtr(GetName()) ;
  if (newPtr != _namePtr) {
    //cout << "Rename '" << _namePtr->GetName() << "' to '" << name << "' (set flag in new name)" << endl;
    _namePtr = newPtr;
    const_cast<TNamed*>(_namePtr)->SetBit(RooNameReg::kRenamedArg);
    RooNameReg::incrementRenameCounter();
  }
}




////////////////////////////////////////////////////////////////////////////////

void RooAbsArg::SetNameTitle(const char *name, const char *title)
{
  TNamed::SetTitle(title) ;
  SetName(name);
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooAbsArg.

void RooAbsArg::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
     _ioReadStack.push(this) ;
     R__b.ReadClassBuffer(RooAbsArg::Class(),this);
     _ioReadStack.pop() ;
     _namePtr = RooNameReg::instance().constPtr(GetName()) ;
     _isConstant = getAttribute("Constant") ;
   } else {
     R__b.WriteClassBuffer(RooAbsArg::Class(),this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Method called by workspace container to finalize schema evolution issues
/// that cannot be handled in a single ioStreamer pass.
///
/// A second pass is typically needed when evolving data member of RooAbsArg-derived
/// classes that are container classes with references to other members, which may
/// not yet be 'live' in the first ioStreamer() evolution pass.
///
/// Classes may overload this function, but must call the base method in the
/// overloaded call to ensure base evolution is handled properly

void RooAbsArg::ioStreamerPass2()
{
  // Handling of v5-v6 migration (TRefArray _proxyList --> RooRefArray _proxyList)
  auto iter = _ioEvoList.find(this);
  if (iter != _ioEvoList.end()) {

    // Transfer contents of saved TRefArray to RooRefArray now
    if (!_proxyList.GetEntriesFast())
       _proxyList.Expand(iter->second->GetEntriesFast());
    for (int i = 0; i < iter->second->GetEntriesFast(); i++) {
       _proxyList.Add(iter->second->At(i));
    }
    // Delete TRefArray and remove from list
    _ioEvoList.erase(iter);
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Method called by workspace container to finalize schema evolution issues
/// that cannot be handled in a single ioStreamer pass. This static finalize method
/// is called after ioStreamerPass2() is called on each directly listed object
/// in the workspace. It's purpose is to complete schema evolution of any
/// objects in the workspace that are not directly listed as content elements
/// (e.g. analytical convolution tokens )

void RooAbsArg::ioStreamerPass2Finalize()
{
  // Handling of v5-v6 migration (TRefArray _proxyList --> RooRefArray _proxyList)
  for (const auto& iter : _ioEvoList) {

    // Transfer contents of saved TRefArray to RooRefArray now
    if (!iter.first->_proxyList.GetEntriesFast())
       iter.first->_proxyList.Expand(iter.second->GetEntriesFast());
    for (int i = 0; i < iter.second->GetEntriesFast(); i++) {
       iter.first->_proxyList.Add(iter.second->At(i));
    }
  }

  _ioEvoList.clear();
}


RooAbsArg::RefCountListLegacyIterator_t *
RooAbsArg::makeLegacyIterator(const RooAbsArg::RefCountList_t& list) const {
  return new RefCountListLegacyIterator_t(list.containedObjects());
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooRefArray.

void RooRefArray::Streamer(TBuffer &R__b)
{
   UInt_t R__s, R__c;
   if (R__b.IsReading()) {

      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }

      // Make temporary refArray and read that from the streamer
      auto refArray = std::make_unique<TRefArray>();
      refArray->Streamer(R__b) ;
      R__b.CheckByteCount(R__s, R__c, refArray->IsA());

      // Schedule deferred processing of TRefArray into proxy list
      RooAbsArg::_ioEvoList[RooAbsArg::_ioReadStack.top()] = std::move(refArray);

   } else {

     R__c = R__b.WriteVersion(RooRefArray::IsA(), true);

     // Make a temporary refArray and write that to the streamer
     TRefArray refArray(GetEntriesFast());
     for(TObject * tmpObj : *this) {
       refArray.Add(tmpObj) ;
     }

     refArray.Streamer(R__b) ;
     R__b.SetByteCount(R__c, true) ;

   }
}

/// Print at the prompt
namespace cling {
std::string printValue(RooAbsArg *raa)
{
   std::stringstream s;
   if (0 == *raa->GetName() && 0 == *raa->GetTitle()) {
      s << "An instance of " << raa->ClassName() << ".";
      return s.str();
   }
   raa->printStream(s, raa->defaultPrintContents(""), raa->defaultPrintStyle(""));
   return s.str();
}
} // namespace cling


/// Disables or enables the usage of squared weights. Needs to be overloaded in
/// the likelihood classes for which this is relevant.
void RooAbsArg::applyWeightSquared(bool flag) {
   for(auto * server : servers()) {
      server->applyWeightSquared(flag);
   }
}


/// Fills a RooArgSet to be used as the normalization set for a server, given a
/// normalization set for this RooAbsArg. If the output is a `nullptr`, it
/// means that the normalization set doesn't change.
///
/// \param[in] normSet The normalization set for this RooAbsArg.
/// \param[in] server A server of this RooAbsArg that we determine the
///            normalization set for.
/// \param[out] serverNormSet Output parameter. Normalization set for the
///             server.
std::unique_ptr<RooArgSet> RooAbsArg::fillNormSetForServer(RooArgSet const& /*normSet*/, RooAbsArg const& /*server*/) const {
   return nullptr;
}
