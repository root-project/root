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

/**
\file RooRealIntegral.cxx
\class RooRealIntegral
\ingroup Roofitcore

Performs hybrid numerical/analytical integrals of RooAbsReal objects.
The class performs none of the actual integration, but only manages the logic
of what variables can be integrated analytically, accounts for eventual jacobian
terms and defines what numerical integrations needs to be done to complement the
analytical integral.
The actual analytical integrations (if any) are done in the PDF themselves, the numerical
integration is performed in the various implementations of the RooAbsIntegrator base class.
**/

#include <RooRealIntegral.h>

#include <RooAbsCategoryLValue.h>
#include <RooAbsRealLValue.h>
#include <RooConstVar.h>
#include <RooDouble.h>
#include <RooExpensiveObjectCache.h>
#include <RooInvTransform.h>
#include <RooMsgService.h>
#include <RooNameReg.h>
#include <RooNumIntConfig.h>
#include <RooNumIntFactory.h>
#include <RooRealBinding.h>
#include <RooSuperCategory.h>
#include <RooTrace.h>

#include <iostream>
#include <memory>


namespace {

/// Utility function that returns true if 'object server' is a server
/// to exactly one of the RooAbsArgs in 'exclLVBranches'
bool servesExclusively(const RooAbsArg *server, const RooArgSet &exclLVBranches, const RooArgSet &allBranches)
{
   // Determine if given server serves exclusively exactly one of the given nodes in exclLVBranches

   // Special case, no LV servers available
   if (exclLVBranches.empty())
      return false;

   // If server has no clients and is not an LValue itself, return false
   if (!server->hasClients() && exclLVBranches.find(server->GetName())) {
      return false;
   }

   // WVE must check for value relations only here!!!!

   // Loop over all clients
   Int_t numLVServ(0);
   for (const auto client : server->valueClients()) {
      // If client is not an LValue, recurse
      if (!(exclLVBranches.find(client->GetName()) == client)) {
         if (allBranches.find(client->GetName()) == client) {
            if (!servesExclusively(client, exclLVBranches, allBranches)) {
               // Client is a non-LValue that doesn't have an exclusive LValue server
               return false;
            }
         }
      } else {
         // Client is an LValue
         numLVServ++;
      }
   }

   return (numLVServ == 1);
}

struct ServerToAdd {
   ServerToAdd(RooAbsArg *theArg, bool isShape) : arg{theArg}, isShapeServer{isShape} {}
   RooAbsArg *arg = nullptr;
   bool isShapeServer = false;
};

void addObservableToServers(RooAbsReal const &function, RooAbsArg &leaf, std::vector<ServerToAdd> &serversToAdd,
                            const char *rangeName)
{
   auto leaflv = dynamic_cast<RooAbsRealLValue *>(&leaf);
   if (leaflv && leaflv->getBinning(rangeName).isParameterized()) {
      oocxcoutD(&function, Integration)
         << function.GetName() << " : Observable " << leaf.GetName()
         << " has parameterized binning, add value dependence of boundary objects rather than shape of leaf"
         << std::endl;
      if (leaflv->getBinning(rangeName).lowBoundFunc()) {
         serversToAdd.emplace_back(leaflv->getBinning(rangeName).lowBoundFunc(), false);
      }
      if (leaflv->getBinning(rangeName).highBoundFunc()) {
         serversToAdd.emplace_back(leaflv->getBinning(rangeName).highBoundFunc(), false);
      }
   } else {
      oocxcoutD(&function, Integration) << function.GetName() << ": Adding observable " << leaf.GetName()
                                        << " as shape dependent" << std::endl;
      serversToAdd.emplace_back(&leaf, true);
   }
}

void addParameterToServers(RooAbsReal const &function, RooAbsArg &leaf, std::vector<ServerToAdd> &serversToAdd,
                           bool isShapeServer)
{
   if (!isShapeServer) {
      oocxcoutD(&function, Integration) << function.GetName() << ": Adding parameter " << leaf.GetName()
                                        << " as value dependent" << std::endl;
   } else {
      oocxcoutD(&function, Integration) << function.GetName() << ": Adding parameter " << leaf.GetName()
                                        << " as shape dependent" << std::endl;
   }
   serversToAdd.emplace_back(&leaf, isShapeServer);
}

enum class MarkedState { Dependent, Independent, AlreadyAdded };

/// Mark all args that recursively are value clients of "dep".
void unmarkDepValueClients(RooAbsArg const &dep, RooArgSet const &args, std::vector<MarkedState> &marked)
{
   assert(args.size() == marked.size());
   auto index = args.index(dep);
   if (index >= 0) {
      marked[index] = MarkedState::Dependent;
      for (RooAbsArg *client : dep.valueClients()) {
         unmarkDepValueClients(*client, args, marked);
      }
   }
}

std::vector<ServerToAdd>
getValueAndShapeServers(RooAbsReal const &function, RooArgSet const &depList, const char *rangeName)
{
   std::vector<ServerToAdd> serversToAdd;

   // Get the full computation graph and sort it topologically
   RooArgList allArgsList;
   function.treeNodeServerList(&allArgsList, nullptr, true, true, /*valueOnly=*/false, false);
   RooArgSet allArgs{allArgsList};
   allArgs.sortTopologically();

   // Figure out what are all the value servers only
   RooArgList allValueArgsList;
   function.treeNodeServerList(&allValueArgsList, nullptr, true, true, /*valueOnly=*/true, false);
   RooArgSet allValueArgs{allValueArgsList};

   // All "marked" args will be added as value servers to the integral
   std::vector<MarkedState> marked(allArgs.size(), MarkedState::Independent);
   marked.back() = MarkedState::Dependent; // We don't want to consider the function itself

   // Mark all args that are (indirect) value servers of the integration
   // variable or the integration variable itself. If something was marked,
   // it means the integration variable was in the compute graph and we will
   // add it to the server list.
   for (RooAbsArg *dep : depList) {
      if (RooAbsArg *depInArgs = allArgs.find(dep->GetName())) {
         unmarkDepValueClients(*depInArgs, allArgs, marked);
         addObservableToServers(function, *depInArgs, serversToAdd, rangeName);
      }
   }

   // We are adding all independent direct servers of the args depending on the
   // integration variables
   for (std::size_t i = 0; i < allArgs.size(); ++i) {
      if (marked[i] == MarkedState::Dependent) {
         for (RooAbsArg *server : allArgs[i]->servers()) {
            int index = allArgs.index(server->GetName());
            if (index >= 0 && marked[index] == MarkedState::Independent) {
               addParameterToServers(function, *server, serversToAdd, !allValueArgs.find(*server));
               marked[index] = MarkedState::AlreadyAdded;
            }
         }
      }
   }

   return serversToAdd;
}

void fillAnIntOKDepList(RooAbsReal const &function, RooArgSet const &intDeps, RooArgSet &anIntOKDepList,
                        const RooArgSet &allBranches)
{
   // If any of the branches in the computation graph of the function depend on
   // the integrated variable, we can't do analytical integration. The only
   // case where this would work is if the branch is an l-value with known
   // Jacobian, but this case is already handled in step B) in the constructor
   // by reexpressing the original integration variables in terms of
   // higher-order l-values if possible.
   RooArgSet filteredIntDeps;
   for (RooAbsArg *intDep : intDeps) {
      bool depOK = true;
      for (RooAbsArg *branch : allBranches) {
         // It's ok if the branch is the integration variable itself
         if (intDep->namePtr() != branch->namePtr() && branch->dependsOnValue(*intDep)) {
            depOK = false;
         }
         if (!depOK) break;
      }
      if (depOK) {
         filteredIntDeps.add(*intDep);
      }
   }

   for (const auto arg : function.servers()) {

      // Dependent or parameter?
      if (!arg->dependsOnValue(filteredIntDeps)) {
         continue;
      } else if (!arg->isValueServer(function) && !arg->isShapeServer(function)) {
         // Skip arg if it is neither value or shape server
         continue;
      }

      bool depOK(false);
      // Check for integratable AbsRealLValue

      if (arg->isDerived()) {
         RooAbsRealLValue *realArgLV = dynamic_cast<RooAbsRealLValue *>(arg);
         RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue *>(arg);
         if ((realArgLV && filteredIntDeps.find(realArgLV->GetName()) &&
              (realArgLV->isJacobianOK(filteredIntDeps) != 0)) ||
             catArgLV) {

            // Derived LValue with valid jacobian
            depOK = true;

            // Now, check for overlaps
            bool overlapOK = true;
            for (const auto otherArg : function.servers()) {
               // skip comparison with self
               if (arg == otherArg)
                  continue;
               if (dynamic_cast<RooConstVar const *>(otherArg))
                  continue;
               if (arg->overlaps(*otherArg, true)) {
               }
            }
            // coverity[DEADCODE]
            if (!overlapOK)
               depOK = false;
         }
      } else {
         // Fundamental types are always OK
         depOK = true;
      }

      // Add server to list of dependents that are OK for analytical integration
      if (depOK) {
         anIntOKDepList.add(*arg, true);
         oocxcoutI(&function, Integration)
            << function.GetName() << ": Observable " << arg->GetName()
            << " is suitable for analytical integration (if supported by p.d.f)" << std::endl;
      }
   }
}

} // namespace

Int_t RooRealIntegral::_cacheAllNDim(2) ;

////////////////////////////////////////////////////////////////////////////////

RooRealIntegral::RooRealIntegral()
{
  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct integral of 'function' over observables in 'depList'
/// in range 'rangeName'  with normalization observables 'funcNormSet'
/// (for p.d.f.s). In the integral is performed to the maximum extent
/// possible the internal (analytical) integrals advertised by function.
/// The other integrations are performed numerically. The optional
/// config object prescribes how these numeric integrations are configured.
///
/// \Note If pdf component selection was globally overridden to always include
/// all components (either with RooAbsReal::globalSelectComp(bool) or a
/// RooAbsReal::GlobalSelectComponentRAII), then any created integral will
/// ignore component selections during its lifetime. This is especially useful
/// when creating normalization or projection integrals.
RooRealIntegral::RooRealIntegral(const char *name, const char *title,
             const RooAbsReal& function, const RooArgSet& depList,
             const RooArgSet* funcNormSet, const RooNumIntConfig* config,
             const char* rangeName) :
  RooAbsReal(name,title),
  _valid(true),
  _respectCompSelect{!_globalSelectComp},
  _sumList("!sumList","Categories to be summed numerically",this,false,false),
  _intList("!intList","Variables to be integrated numerically",this,false,false),
  _anaList("!anaList","Variables to be integrated analytically",this,false,false),
  _jacList("!jacList","Jacobian product term",this,false,false),
  _facList("!facList","Variables independent of function",this,false,true),
  _function("!func","Function to be integrated",this,false,false),
  _iconfig(const_cast<RooNumIntConfig*>(config)),
  _sumCat("!sumCat","SuperCategory for summation",this,false,false),
  _rangeName(const_cast<TNamed*>(RooNameReg::ptr(rangeName)))
{
  //   A) Check that all dependents are lvalues
  //
  //   B) Check if list of dependents can be re-expressed in
  //      lvalues that are higher in the expression tree
  //
  //   C) Check for dependents that the PDF insists on integrating
  //      analytically itself
  //
  //   D) Make list of servers that can be integrated analytically
  //      Add all parameters/dependents as value/shape servers
  //
  //   E) Interact with function to make list of objects actually integrated analytically
  //
  //   F) Make list of numerical integration variables consisting of:
  //     - Category dependents of RealLValues in analytical integration
  //     - Leaf nodes server lists of function server that are not analytically integrated
  //     - Make Jacobian list for analytically integrated RealLValues
  //
  //   G) Split numeric list in integration list and summation list
  //

  oocxcoutI(&function,Integration) << "RooRealIntegral::ctor(" << GetName() << ") Constructing integral of function "
                 << function.GetName() << " over observables" << depList << " with normalization "
                 << (funcNormSet?*funcNormSet:RooArgSet()) << " with range identifier "
                 << (rangeName?rangeName:"<none>") << std::endl ;


  // Choose same expensive object cache as integrand
  setExpensiveObjectCache(function.expensiveObjectCache()) ;
//   std::cout << "RRI::ctor(" << GetName() << ") setting expensive object cache to " << &expensiveObjectCache() << " as taken from " << function.GetName() << std::endl ;

  // Use objects integrator configuration if none is specified
  if (!_iconfig) _iconfig = const_cast<RooNumIntConfig*>(function.getIntegratorConfig());

  // Save private copy of funcNormSet, if supplied, excluding factorizing terms
  if (funcNormSet) {
    _funcNormSet = std::make_unique<RooArgSet>();
    for (const auto nArg : *funcNormSet) {
      if (function.dependsOn(*nArg)) {
        _funcNormSet->addClone(*nArg) ;
      }
    }
  }

  //_funcNormSet = funcNormSet ? (RooArgSet*)funcNormSet->snapshot(false) : 0 ;

  // Make internal copy of dependent list
  RooArgSet intDepList(depList) ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * A) Check that all dependents are lvalues and filter out any
  //      dependents that the PDF doesn't explicitly depend on
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  for (auto arg : intDepList) {
    if(!arg->isLValue()) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ": cannot integrate non-lvalue ";
      arg->Print("1");
      _valid= false;
    }
    if (!function.dependsOn(*arg)) {
      std::unique_ptr<RooAbsArg> argClone{static_cast<RooAbsArg*>(arg->Clone())};
      _facList.add(*argClone);
      addOwnedComponents(std::move(argClone));
    }
  }

  if (!_facList.empty()) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Factorizing obserables are " << _facList << std::endl ;
  }


  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * B) Check if list of dependents can be re-expressed in       *
  // *    lvalues that are higher in the expression tree           *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  // Initial fill of list of LValue branches
  RooArgSet exclLVBranches("exclLVBranches") ;
  RooArgSet branchList;
  function.branchNodeServerList(&branchList) ;

  RooArgList branchListVDAll;
  function.treeNodeServerList(&branchListVDAll,nullptr,true,false,/*valueOnly=*/true);
  // The branchListVD is similar to branchList but considers only value
  // dependence, and we want to exclude the function itself
  RooArgSet branchListVD;
  branchListVD.reserve(branchListVDAll.size());
  for (RooAbsArg *branch : branchListVDAll) {
    if (branch != &function) branchListVD.add(*branch);
  }

  for (auto branch: branchList) {
    RooAbsRealLValue    *realArgLV = dynamic_cast<RooAbsRealLValue*>(branch) ;
    RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue*>(branch) ;
    if ((realArgLV && (realArgLV->isJacobianOK(intDepList)!=0)) || catArgLV) {
      exclLVBranches.add(*branch) ;
    }
  }
  exclLVBranches.remove(depList,true,true) ;

  // Initial fill of list of LValue leaf servers (put in intDepList, but the
  // instances that are in the actual computation graph of the function)
  RooArgSet exclLVServers("exclLVServers") ;
  function.getObservables(&intDepList, exclLVServers);

  // Obtain mutual exclusive dependence by iterative reduction
  bool converged(false) ;
  while(!converged) {
    converged=true ;

    // Reduce exclLVServers to only those serving exclusively exclLVBranches
    std::vector<RooAbsArg*> toBeRemoved;
    for (auto server : exclLVServers) {
      if (!servesExclusively(server,exclLVBranches,branchListVD)) {
        toBeRemoved.push_back(server);
        converged=false ;
      }
    }
    exclLVServers.remove(toBeRemoved.begin(), toBeRemoved.end());

    // Reduce exclLVBranches to only those depending exclusively on exclLVservers
    // Attention: counting loop, since erasing from container
    for (std::size_t i=0; i < exclLVBranches.size(); ++i) {
      const RooAbsArg* branch = exclLVBranches[i];
      RooArgSet brDepList;
      branch->getObservables(&intDepList, brDepList);
      RooArgSet bsList(brDepList,"bsList") ;
      bsList.remove(exclLVServers,true,true) ;
      if (!bsList.empty()) {
        exclLVBranches.remove(*branch,true,true) ;
        --i;
        converged=false ;
      }
    }
  }

  // Eliminate exclLVBranches that do not depend on any LVServer
  // Attention: Counting loop, since modifying container
  for (std::size_t i=0; i < exclLVBranches.size(); ++i) {
    const RooAbsArg* branch = exclLVBranches[i];
    if (!branch->dependsOnValue(exclLVServers)) {
      exclLVBranches.remove(*branch,true,true) ;
      --i;
    }
  }

  // Replace exclusive lvalue branch servers with lvalue branches
  // WVE Don't do this for binned distributions - deal with this using numeric integration with transformed bin boundaroes
  if (!exclLVServers.empty() && !function.isBinnedDistribution(exclLVBranches)) {
    intDepList.remove(exclLVServers) ;
    intDepList.add(exclLVBranches) ;
  }


  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * C) Check for dependents that the PDF insists on integrating *
  //      analytically itself                                       *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet anIntOKDepList ;
  for (auto arg : intDepList) {
    if (function.forceAnalyticalInt(*arg)) {
      anIntOKDepList.add(*arg) ;
    }
  }

  if (!anIntOKDepList.empty()) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Observables that function forcibly requires to be integrated internally " << anIntOKDepList << std::endl ;
  }


  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * D) Make list of servers that can be integrated analytically *
  //      Add all parameters/dependents as value/shape servers     *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  auto serversToAdd = getValueAndShapeServers(function, depList, rangeName);
  fillAnIntOKDepList(function, intDepList, anIntOKDepList, branchListVD);
  // We will not add the servers just now, because it makes only sense to add
  // them once we have made sure that this integral is not operating in
  // pass-through mode. It will be done at the end of this constructor.

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * E) interact with function to make list of objects actually integrated analytically  *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  _mode = function.getAnalyticalIntegralWN(anIntOKDepList,_anaList,_funcNormSet.get(),RooNameReg::str(_rangeName)) ;

  // Avoid confusion -- if mode is zero no analytical integral is defined regardless of contents of _anaList
  if (_mode==0) {
    _anaList.removeAll() ;
  }

  if (_mode!=0) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Function integrated observables " << _anaList << " internally with code " << _mode << std::endl ;
  }

  // WVE kludge: synchronize dset for use in analyticalIntegral
  // LM : I think this is needed only if  _funcNormSet is not an empty set
  if (_funcNormSet && !_funcNormSet->empty()) {
    function.getVal(_funcNormSet.get()) ;
  }

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * F) Make list of numerical integration variables consisting of:            *
  // *   - Category dependents of RealLValues in analytical integration          *
  // *   - Expanded server lists of server that are not analytically integrated  *
  // *    Make Jacobian list with analytically integrated RealLValues            *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  // Loop over actually analytically integrated dependents
  for (const auto arg : _anaList) {

    // Process only derived RealLValues
    if (dynamic_cast<RooAbsRealLValue const *>(arg) && arg->isDerived() && !arg->isFundamental()) {

      // Add to list of Jacobians to calculate
      _jacList.add(*arg) ;

      // Add category dependent of LValueReal used in integration
      std::unique_ptr<RooArgSet> argDepList{arg->getObservables(&intDepList)};
      for (const auto argDep : *argDepList) {
        if (dynamic_cast<RooAbsCategoryLValue const *>(argDep) && intDepList.contains(*argDep)) {
          addNumIntDep(*argDep);
        }
      }
    }
  }


  // If nothing was integrated analytically, swap back LVbranches for LVservers for subsequent numeric integration
  if (_anaList.empty()) {
    if (!exclLVServers.empty()) {
      //cout << "NUMINT phase analList is empty. exclLVServers = " << exclLVServers << std::endl ;
      intDepList.remove(exclLVBranches) ;
      intDepList.add(exclLVServers) ;
     }
  }
  //cout << "NUMINT intDepList = " << intDepList << std::endl ;

  // Loop again over function servers to add remaining numeric integrations
  for (const auto arg : function.servers()) {

    // Process only servers that are not treated analytically
    if (!_anaList.find(arg->GetName()) && arg->dependsOn(intDepList)) {

      // Process only derived RealLValues
      if (dynamic_cast<RooAbsLValue*>(arg) && arg->isDerived() && intDepList.contains(*arg)) {
        addNumIntDep(*arg) ;
      } else {

        // WVE this will only get the observables, but not l-value transformations
        // Expand server in final dependents
        auto argDeps = std::unique_ptr<RooArgSet>(arg->getObservables(&intDepList));

          // Add final dependents, that are not forcibly integrated analytically,
          // to numerical integration list
          for (const auto dep : *argDeps) {
            if (!_anaList.find(dep->GetName())) {
              addNumIntDep(*dep);
            }
          }
      }
    }
  }

  if (!_anaList.empty()) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Observables " << _anaList << " are analytically integrated with code " << _mode << std::endl ;
  }
  if (!_intList.empty()) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Observables " << _intList << " are numerically integrated" << std::endl ;
  }
  if (!_sumList.empty()) {
    oocxcoutI(&function,Integration) << function.GetName() << ": Observables " << _sumList << " are numerically summed" << std::endl ;
  }


  // Determine operating mode
  if (!_intList.empty() || !_sumList.empty()) {
    // Numerical and optional Analytical integration
    _intOperMode = Hybrid ;
  } else if (!_anaList.empty()) {
    // Purely analytical integration
    _intOperMode = Analytic ;
  } else {
    // No integration performed, where the function is a direct value server
    _intOperMode = PassThrough ;
    _function._valueServer = true;
  }
  // We are only setting the function proxy now that it's clear if it's a value
  // server or not.
  _function.setArg(const_cast<RooAbsReal&>(function));

  // Determine auto-dirty status
  autoSelectDirtyMode() ;

  // Create value caches for _intList and _sumList
  _intList.snapshot(_saveInt) ;
  _sumList.snapshot(_saveSum) ;


  if (!_sumList.empty()) {
    _sumCat.addOwned(std::make_unique<RooSuperCategory>(Form("%s_sumCat",GetName()),"sumCat",_sumList));
  }

  // Only if we are not in pass-through mode we need to add the shape and value
  // servers separately.
  if(_intOperMode != PassThrough) {
    for(auto const& toAdd : serversToAdd) {
      addServer(*toAdd.arg, !toAdd.isShapeServer, toAdd.isShapeServer);
    }
  }

  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set appropriate cache operation mode for integral depending on cache operation
/// mode of server objects

void RooRealIntegral::autoSelectDirtyMode()
{
  // If any of our servers are is forcedDirty or a projectedDependent, then we need to be ADirty
  for (const auto server : _serverList) {
    if (server->isValueServer(*this)) {
      RooArgSet leafSet ;
      server->leafNodeServerList(&leafSet) ;
      for (const auto leaf : leafSet) {
        if (leaf->operMode()==ADirty && leaf->isValueServer(*this)) {
          setOperMode(ADirty) ;
          break ;
        }
        if (leaf->getAttribute("projectedDependent")) {
          setOperMode(ADirty) ;
          break ;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// (Re)Initialize numerical integration engine if necessary. Return true if
/// successful, or otherwise false.

bool RooRealIntegral::initNumIntegrator() const
{
  // if we already have an engine, check if it still works for the present limits.
  if(_numIntEngine) {
    if(_numIntEngine->isValid() && _numIntEngine->checkLimits() && !_restartNumIntEngine ) return true;
    // otherwise, cleanup the old engine
    _numIntEngine.reset();
    _numIntegrand.reset();
  }

  // All done if there are no arguments to integrate numerically
  if(_intList.empty()) return true;

  // Bind the appropriate analytic integral of our RooRealVar object to
  // those of its arguments that will be integrated out numerically.
  if(_mode != 0) {
    std::unique_ptr<RooAbsReal> analyticalPart{_function->createIntegral(_anaList, *actualFuncNormSet(), RooNameReg::str(_rangeName))};
    _numIntegrand = std::make_unique<RooRealBinding>(*analyticalPart,_intList,nullptr,false,_rangeName);
    const_cast<RooRealIntegral*>(this)->addOwnedComponents(std::move(analyticalPart));
  }
  else {
    _numIntegrand = std::make_unique<RooRealBinding>(*_function,_intList,actualFuncNormSet(),false,_rangeName);
  }
  if(nullptr == _numIntegrand || !_numIntegrand->isValid()) {
    coutE(Integration) << ClassName() << "::" << GetName() << ": failed to create valid integrand." << std::endl;
    return false;
  }

  // Create appropriate numeric integrator using factory
  bool isBinned = _function->isBinnedDistribution(_intList) ;
  std::string integratorName = RooNumIntFactory::instance().getIntegratorName(*_numIntegrand,*_iconfig,0,isBinned);
  _numIntEngine = RooNumIntFactory::instance().createIntegrator(*_numIntegrand,*_iconfig,0,isBinned);

  if(_numIntEngine == nullptr || !_numIntEngine->isValid()) {
    coutE(Integration) << ClassName() << "::" << GetName() << ": failed to create valid integrator." << std::endl;
    return false;
  }

  cxcoutI(NumericIntegration) << "RooRealIntegral::init(" << GetName() << ") using numeric integrator "
           << integratorName << " to calculate Int" << _intList << std::endl ;

  if (_intList.size()>3) {
    cxcoutI(NumericIntegration) << "RooRealIntegral::init(" << GetName() << ") evaluation requires " << _intList.size() << "-D numeric integration step. Evaluation may be slow, sufficient numeric precision for fitting & minimization is not guaranteed" << std::endl ;
  }

  _restartNumIntEngine = false ;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRealIntegral::RooRealIntegral(const RooRealIntegral &other, const char *name)
   : RooAbsReal(other, name),
     _valid(other._valid),
     _respectCompSelect(other._respectCompSelect),
     _sumList("!sumList", this, other._sumList),
     _intList("!intList", this, other._intList),
     _anaList("!anaList", this, other._anaList),
     _jacList("!jacList", this, other._jacList),
     _facList("!facList", this, other._facList),
     _function("!func", this, other._function),
     _iconfig(other._iconfig),
     _sumCat("!sumCat", this, other._sumCat),
     _mode(other._mode),
     _intOperMode(other._intOperMode),
     _rangeName(other._rangeName)
{
 if(other._funcNormSet) {
   _funcNormSet = std::make_unique<RooArgSet>();
   other._funcNormSet->snapshot(*_funcNormSet, false);
 }

 other._intList.snapshot(_saveInt) ;
 other._sumList.snapshot(_saveSum) ;

  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////

RooRealIntegral::~RooRealIntegral()
{
  TRACE_DESTROY;
}

////////////////////////////////////////////////////////////////////////////////

RooFit::OwningPtr<RooAbsReal> RooRealIntegral::createIntegral(const RooArgSet& iset, const RooArgSet* nset, const RooNumIntConfig* cfg, const char* rangeName) const
{
  // Handle special case of no integration with default algorithm
  if (iset.empty()) {
    return RooAbsReal::createIntegral(iset,nset,cfg,rangeName) ;
  }

  // Special handling of integral of integral, return RooRealIntegral that represents integral over all dimensions in one pass
  RooArgSet isetAll(iset) ;
  isetAll.add(_sumList) ;
  isetAll.add(_intList) ;
  isetAll.add(_anaList) ;
  isetAll.add(_facList) ;

  const RooArgSet* newNormSet(nullptr) ;
  std::unique_ptr<RooArgSet> tmp;
  if (nset && !_funcNormSet) {
    newNormSet = nset ;
  } else if (!nset && _funcNormSet) {
    newNormSet = _funcNormSet.get();
  } else if (nset && _funcNormSet) {
    tmp = std::make_unique<RooArgSet>();
    tmp->add(*nset) ;
    tmp->add(*_funcNormSet,true) ;
    newNormSet = tmp.get();
  }
  return  _function->createIntegral(isetAll,newNormSet,cfg,rangeName);
}

////////////////////////////////////////////////////////////////////////////////
/// Return value of object. If the cache is clean, return the
/// cached value, otherwise recalculate on the fly and refill
/// the cache

double RooRealIntegral::getValV(const RooArgSet* nset) const
{
//   // fast-track clean-cache processing
//   if (_operMode==AClean) {
//     return _value ;
//   }

  if (nset && nset->uniqueId().value() != _lastNormSetId) {
    const_cast<RooRealIntegral*>(this)->setProxyNormSet(nset);
    _lastNormSetId = nset->uniqueId().value();
  }

  if (isValueOrShapeDirtyAndClear()) {
    _value = traceEval(nset) ;
  }

  return _value ;
}

////////////////////////////////////////////////////////////////////////////////
/// Perform the integration and return the result

double RooRealIntegral::evaluate() const
{
  GlobalSelectComponentRAII selCompRAII(!_respectCompSelect);

  double retVal(0) ;
  switch (_intOperMode) {

  case Hybrid:
    {
      // Cache numeric integrals in >1d expensive object cache
      RooDouble const* cacheVal(nullptr) ;
      if ((_cacheNum && !_intList.empty()) || int(_intList.size())>=_cacheAllNDim) {
        cacheVal = static_cast<RooDouble const*>(expensiveObjectCache().retrieveObject(GetName(),RooDouble::Class(),parameters()))  ;
      }

      if (cacheVal) {
        retVal = *cacheVal ;
   // std::cout << "using cached value of integral" << GetName() << std::endl ;
      } else {


        // Find any function dependents that are AClean
        // and switch them temporarily to ADirty
        bool origState = inhibitDirty() ;
        setDirtyInhibit(true) ;

        // try to initialize our numerical integration engine
        if(!(_valid= initNumIntegrator())) {
          coutE(Integration) << ClassName() << "::" << GetName()
                             << ":evaluate: cannot initialize numerical integrator" << std::endl;
          return 0;
        }

        // Save current integral dependent values
        _saveInt.assign(_intList) ;
        _saveSum.assign(_sumList) ;

        // Evaluate sum/integral
        retVal = sum() ;


        // This must happen BEFORE restoring dependents, otherwise no dirty state propagation in restore step
        setDirtyInhibit(origState) ;

        // Restore integral dependent values
        _intList.assign(_saveInt) ;
        _sumList.assign(_saveSum) ;

        // Cache numeric integrals in >1d expensive object cache
        if ((_cacheNum && !_intList.empty()) || int(_intList.size())>=_cacheAllNDim) {
          RooDouble* val = new RooDouble(retVal) ;
          expensiveObjectCache().registerObject(_function->GetName(),GetName(),*val,parameters())  ;
          //     std::cout << "### caching value of integral" << GetName() << " in " << &expensiveObjectCache() << std::endl ;
        }

      }
      break ;
    }
  case Analytic:
    {
      retVal = _function->analyticalIntegralWN(_mode,_funcNormSet.get(),RooNameReg::str(_rangeName)) / jacobianProduct() ;
      cxcoutD(Tracing) << "RooRealIntegral::evaluate_analytic(" << GetName()
             << ")func = " << _function->ClassName() << "::" << _function->GetName()
             << " raw = " << retVal << " _funcNormSet = " << (_funcNormSet?*_funcNormSet:RooArgSet()) << std::endl ;


      break ;
    }

  case PassThrough:
    {
      // In pass through mode, the RooRealIntegral should have registered the
      // function as a value server, because we directly depend on its value.
      assert(_function.isValueServer());
      // There should be no other servers besides the actual function and the
      // factorized observables that the function doesn't depend on but are
      // integrated over later.
      assert(servers().size() == _facList.size() + 1);

      //setDirtyInhibit(true) ;
      retVal= _function->getVal(actualFuncNormSet());
      //setDirtyInhibit(false) ;
      break ;
    }
  }


  // Multiply answer with integration ranges of factorized variables
    for (const auto arg : _facList) {
      // Multiply by fit range for 'real' dependents
      if (auto argLV = dynamic_cast<RooAbsRealLValue *>(arg)) {
        retVal *= (argLV->getMax(intRange()) - argLV->getMin(intRange())) ;
      }
      // Multiply by number of states for category dependents
      if (auto argLV = dynamic_cast<RooAbsCategoryLValue *>(arg)) {
        retVal *= argLV->numTypes() ;
      }
    }


  if (dologD(Tracing)) {
    cxcoutD(Tracing) << "RooRealIntegral::evaluate(" << GetName() << ") anaInt = " << _anaList << " numInt = " << _intList << _sumList << " mode = " ;
    switch(_intOperMode) {
    case Hybrid: ccoutD(Tracing) << "Hybrid" ; break ;
    case Analytic: ccoutD(Tracing) << "Analytic" ; break ;
    case PassThrough: ccoutD(Tracing) << "PassThrough" ; break ;
    }

    ccxcoutD(Tracing) << "raw*fact = " << retVal << std::endl ;
  }

  return retVal ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return product of jacobian terms originating from analytical integration

double RooRealIntegral::jacobianProduct() const
{
  if (_jacList.empty()) {
    return 1 ;
  }

  double jacProd(1) ;
  for (const auto elm : _jacList) {
    auto arg = static_cast<const RooAbsRealLValue*>(elm);
    jacProd *= arg->jacobian() ;
  }

  // Take std::abs() here: if jacobian is negative, min and max are swapped and analytical integral
  // will be positive, so must multiply with positive jacobian.
  return std::abs(jacProd) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Perform summation of list of category dependents to be integrated

double RooRealIntegral::sum() const
{
  if (!_sumList.empty()) {
    // Add integrals for all permutations of categories summed over
    double total(0) ;

    RooSuperCategory* sumCat = static_cast<RooSuperCategory*>(_sumCat.first()) ;
    for (const auto& nameIdx : *sumCat) {
      sumCat->setIndex(nameIdx);
      if (!_rangeName || sumCat->inRange(RooNameReg::str(_rangeName))) {
        total += integrate() / jacobianProduct() ;
      }
    }

    return total ;

  } else {
    // Simply return integral
    double ret = integrate() / jacobianProduct() ;
    return ret ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Perform hybrid numerical/analytical integration over all real-valued dependents

double RooRealIntegral::integrate() const
{
  if (!_numIntEngine) {
    // Trivial case, fully analytical integration
    return _function->analyticalIntegralWN(_mode,_funcNormSet.get(),RooNameReg::str(_rangeName)) ;
  } else {
    return _numIntEngine->calculate()  ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Intercept server redirects and reconfigure internal object accordingly

bool RooRealIntegral::redirectServersHook(const RooAbsCollection& newServerList,
                   bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  _restartNumIntEngine = true ;

  autoSelectDirtyMode() ;

  // Update contents value caches for _intList and _sumList
  _saveInt.removeAll() ;
  _saveSum.removeAll() ;
  _intList.snapshot(_saveInt) ;
  _sumList.snapshot(_saveSum) ;

  // Delete parameters cache if we have one
  _params.reset();

  return RooAbsReal::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}

////////////////////////////////////////////////////////////////////////////////

const RooArgSet& RooRealIntegral::parameters() const
{
  if (!_params) {
    _params = std::make_unique<RooArgSet>("params") ;

    RooArgSet params ;
    for (const auto server : _serverList) {
      if (server->isValueServer(*this)) _params->add(*server) ;
    }
  }

  return *_params ;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if current value is valid

bool RooRealIntegral::isValidReal(double /*value*/, bool /*printError*/) const
{
  return true ;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if component selection is allowed

bool RooRealIntegral::getAllowComponentSelection() const {
  return _respectCompSelect;
}

////////////////////////////////////////////////////////////////////////////////
/// Set component selection to be allowed/forbidden

void RooRealIntegral::setAllowComponentSelection(bool allow){
  _respectCompSelect = allow;
}

////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooRealIntegral to more intuitively reflect the contents of the
/// integration operation

void RooRealIntegral::printMetaArgs(std::ostream& os) const
{
  if (!intVars().empty()) {
    os << "Int " ;
  }
  os << _function->GetName() ;
  if (_funcNormSet) {
    os << "_Norm" << *_funcNormSet << " " ;
  }

  // List internally integrated observables and factorizing observables as analytically integrated
  RooArgSet tmp(_anaList) ;
  tmp.add(_facList) ;
  if (!tmp.empty()) {
    os << "d[Ana]" << tmp << " ";
  }

  // List numerically integrated and summed observables as numerically integrated
  RooArgSet tmp2(_intList) ;
  tmp2.add(_sumList) ;
  if (!tmp2.empty()) {
    os << " d[Num]" << tmp2 << " ";
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Print the state of this object to the specified output stream.

void RooRealIntegral::printMultiline(std::ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooAbsReal::printMultiline(os,contents,verbose,indent) ;
  os << indent << "--- RooRealIntegral ---" << std::endl;
  os << indent << "  Integrates ";
  _function->printStream(os,kName|kArgs,kSingleLine,indent);
  TString deeper(indent);
  deeper.Append("  ");
  os << indent << "  operating mode is "
     << (_intOperMode==Hybrid?"Hybrid":(_intOperMode==Analytic?"Analytic":"PassThrough")) << std::endl ;
  os << indent << "  Summed discrete args are " << _sumList << std::endl ;
  os << indent << "  Numerically integrated args are " << _intList << std::endl;
  os << indent << "  Analytically integrated args using mode " << _mode << " are " << _anaList << std::endl ;
  os << indent << "  Arguments included in Jacobian are " << _jacList << std::endl ;
  os << indent << "  Factorized arguments are " << _facList << std::endl ;
  os << indent << "  Function normalization set " ;
  if (_funcNormSet) {
    _funcNormSet->Print("1") ;
  } else {
    os << "<none>";
  }

  os << std::endl ;
}

////////////////////////////////////////////////////////////////////////////////
/// Global switch to cache all integral values that integrate at least ndim dimensions numerically

void RooRealIntegral::setCacheAllNumeric(Int_t ndim)
{
   _cacheAllNDim = ndim;
}

////////////////////////////////////////////////////////////////////////////////
/// Return minimum dimensions of numeric integration for which values are cached.

Int_t RooRealIntegral::getCacheAllNumeric()
{
   return _cacheAllNDim;
}

std::unique_ptr<RooAbsArg>
RooRealIntegral::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   return RooAbsReal::compileForNormSet(_funcNormSet ? *_funcNormSet : normSet, ctx);
}

/// Sort numeric integration variables in summation and integration lists.
/// To be used during construction.
void RooRealIntegral::addNumIntDep(RooAbsArg const &arg)
{
   if (dynamic_cast<RooAbsRealLValue const *>(&arg)) {
      _intList.add(arg, true);
   } else if (dynamic_cast<RooAbsCategoryLValue const *>(&arg)) {
      _sumList.add(arg, true);
   }
}
