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
\file RooDataSet.cxx
\class RooDataSet
\ingroup Roofitcore

Container class to hold unbinned data. The binned equivalent is
RooDataHist. In RooDataSet, each data point in N-dimensional space is represented
by a RooArgSet of RooRealVar, RooCategory or RooStringVar objects, which can be
retrieved using get().

Since RooDataSet saves every event, it allows for fits with highest precision. With a large
amount of data, however, it could be beneficial to represent them in binned form,
i.e., RooDataHist. Binning the data will incur a loss of information, though.
RooDataHist on the other hand may suffer from the curse of dimensionality if a high-dimensional
problem with a lot of bins on each axis is tackled.

### Inspecting a dataset
Inspect a dataset using Print() with the "verbose" option:
```
dataset->Print("V");
dataset->get(0)->Print("V");
dataset->get(1)->Print("V");
...
```

### Plotting data.
See RooAbsData::plotOn().


### Storage strategy
There are two storage backends:
- RooVectorDataStore (default): std::vectors in memory. They are fast, but they
cannot be serialised if the dataset exceeds a size of 1 Gb
- RooTreeDataStore: Uses a TTree under the hood. Note that the TTree is not
attached to any currently-opened TFile in order to avoid double-ownership.
  - Enable tree-backed storage similar to this:
  ```
  TFile outputFile("filename.root", "RECREATE");
  RooAbsData::setDefaultStorageType(RooAbsData::Tree);
  RooDataSet mydata(...);
  ```
  - Or convert an existing memory-backed data storage:
  ```
  RooDataSet mydata(...);

  TFile outputFile("filename.root", "RECREATE");
  mydata.convertToTreeStore();
  ```

For the inverse conversion, see `RooAbsData::convertToVectorStore()`.


### Creating a dataset using RDataFrame
See RooAbsDataHelper, rf408_RDataFrameToRooFit.C

### Uniquely identifying RooDataSet objects

\warning Before v6.28, it was ensured that no RooDataSet objects on the heap
were located at an address that had already been used for a RooDataSet before.
With v6.28, this is not guaranteed anymore. Hence, if your code uses pointer
comparisons to uniquely identify RooDataSet instances, please consider using
the new `RooAbsData::uniqueId()`.


**/

#include "RooDataSet.h"

#include "RooPlot.h"
#include "RooAbsReal.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooFormula.h"
#include "RooFormulaVar.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooMsgService.h"
#include "RooCmdConfig.h"
#include "RooHist.h"
#include "RooTreeDataStore.h"
#include "RooVectorDataStore.h"
#include "RooCompositeDataStore.h"
#include "RooSentinel.h"
#include "RooTrace.h"
#include "RooFitImplHelpers.h"

#include "ROOT/StringUtils.hxx"

#include "Math/Util.h"
#include "TTree.h"
#include "TFile.h"
#include "TBuffer.h"
#include "strlcpy.h"
#include "snprintf.h"

#include <iostream>
#include <memory>
#include <fstream>


using std::endl, std::string, std::map, std::list, std::ifstream, std::ofstream, std::ostream;


void RooDataSet::cleanup() {}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for persistence

RooDataSet::RooDataSet()
{
  TRACE_CREATE;
}

namespace {

struct FinalizeVarsOutput {
   RooArgSet finalVars;
   std::unique_ptr<RooRealVar> weight;
   std::string weightVarName;
   RooArgSet errorSet;
};

FinalizeVarsOutput finalizeVars(RooArgSet const &vars,
                                RooAbsArg * indexCat,
                                const char* wgtVarName,
                                RooAbsData* impData,
                                RooLinkedList const &impSliceData,
                                RooArgSet * errorSet)
{
   FinalizeVarsOutput out;
   out.finalVars.add(vars);

   // Gather all imported weighted datasets to infer the weight variable name
   // and whether we need weight errors
   std::vector<RooAbsData*> weightedImpDatasets;
   if(impData && impData->isWeighted()) weightedImpDatasets.push_back(impData);
   for(auto * data : static_range_cast<RooAbsData*>(impSliceData)) {
      if(data->isWeighted()) {
         weightedImpDatasets.push_back(data);
      }
   }

   bool needsWeightErrors = false;

   // Figure out if the weight needs to store errors
   for(RooAbsData * data : weightedImpDatasets) {
      if(dynamic_cast<RooDataHist const*>(data)) {
         needsWeightErrors = true;
      }
   }

   if (indexCat) {
      out.finalVars.add(*indexCat, true);
   }

   out.weightVarName = wgtVarName ? wgtVarName : "";

   if(out.weightVarName.empty()) {
      // Even if no weight variable is specified, we want to have one if we are
      // importing weighted datasets
      for(RooAbsData * data : weightedImpDatasets) {
         if(auto ds = dynamic_cast<RooDataSet const*>(data)) {
            // If the imported data is a RooDataSet, we take over its weight variable name
            out.weightVarName = ds->weightVar()->GetName();
            break;
         } else {
            out.weightVarName = RooFit::WeightVar().getString(0); // to get the default weight variable name
            // Don't break here! The next imported data might be a RooDataSet,
            // and in that case we want to take over its weight name instead of
            // using the default one.
         }
      }
   }

   // If the weight variable is required but is not in the set, create and add
   // it on the fly
   RooAbsArg * wgtVar = out.finalVars.find(out.weightVarName.c_str());
   if (!out.weightVarName.empty() && !wgtVar) {
      const char* name = out.weightVarName.c_str();
      out.weight = std::make_unique<RooRealVar>(name, name, 1.0);
      wgtVar = out.weight.get();
      out.finalVars.add(*out.weight);
   }

   if(needsWeightErrors) {
      out.errorSet.add(*wgtVar);
   }

   // Combine the error set figured out by finalizeVars and the ones passed by the user
   if(errorSet) out.errorSet.add(*errorSet, /*silent=*/true);

   return out;
}

// generating an unbinned dataset from a binned one
std::unique_ptr<RooDataSet> makeDataSetFromDataHist(RooDataHist const &hist)
{
   using namespace RooFit;

   RooCmdArg const& wgtVarCmdArg = RooFit::WeightVar();
   const char* wgtName = wgtVarCmdArg.getString(0);
   // Instantiate weight variable here such that we can pass it to StoreError()
   RooRealVar wgtVar{wgtName, wgtName, 1.0};

   RooArgSet vars{*hist.get(), wgtVar};

   // We have to explicitly store the errors that are implied by the sum of weights squared.
   auto data = std::make_unique<RooDataSet>(hist.GetName(), hist.GetTitle(), vars, wgtVarCmdArg, StoreError(wgtVar));
   for (int i = 0; i < hist.numEntries(); ++i) {
      data->add(*hist.get(i), hist.weight(i), std::sqrt(hist.weightSquared(i)));
   }

   return data;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Construct an unbinned dataset from a RooArgSet defining the dimensions of the data space. Optionally, data
/// can be imported at the time of construction.
///
/// <table>
/// <tr><th> %RooCmdArg <th> Effect
/// <tr><td> Import(TTree&)   <td> Import contents of given TTree. Only branches of the TTree that have names
///                                corresponding to those of the RooAbsArgs that define the RooDataSet are
///                                imported.
/// <tr><td> ImportFromFile(const char* fileName, const char* treeName) <td> Import tree with given name from file with given name.
/// <tr><td> Import(RooAbsData&)
///     <td> Import contents of given RooDataSet or RooDataHist. Only observables that are common with the definition of this dataset will be imported
/// <tr><td> Index(RooCategory&)         <td> Prepare import of datasets into a N+1 dimensional RooDataSet
///                                where the extra discrete dimension labels the source of the imported histogram.
/// <tr><td> Import(const char*, RooAbsData&)
///     <td> Import a RooDataSet or RooDataHist to be associated with the given state name of the index category
///                    specified in Index(). If the given state name is not yet defined in the index
///                    category it will be added on the fly. The import command can be specified multiple times.
/// <tr><td> Link(const char*, RooDataSet&) <td> Link contents of supplied RooDataSet to this dataset for given index category state name.
///                                   In this mode, no data is copied and the linked dataset must be remain live for the duration
///                                   of this dataset. Note that link is active for both reading and writing, so modifications
///                                   to the aggregate dataset will also modify its components. Link() and Import() are mutually exclusive.
/// <tr><td> OwnLinked()                    <td> Take ownership of all linked datasets
/// <tr><td> Import(std::map<string,RooAbsData*>&) <td> As above, but allows specification of many imports in a single operation
/// <tr><td> Link(std::map<string,RooDataSet*>&)   <td> As above, but allows specification of many links in a single operation
/// <tr><td> Cut(const char*) <br>
///     Cut(RooFormulaVar&)
///     <td> Apply the given cut specification when importing data
/// <tr><td> CutRange(const char*)       <td> Only accept events in the observable range with the given name
/// <tr><td> WeightVar(const char*) <br>
///     WeightVar(const RooAbsArg&)
///     <td> Interpret the given variable as event weight rather than as observable
/// <tr><td> StoreError(const RooArgSet&)     <td> Store symmetric error along with value for given subset of observables
/// <tr><td> StoreAsymError(const RooArgSet&) <td> Store asymmetric error along with value for given subset of observables
/// <tr><td> `GlobalObservables(const RooArgSet&)` <td> Define the set of global observables to be stored in this RooDataSet.
///                                                     A snapshot of the passed RooArgSet is stored, meaning the values wont't change unexpectedly.
/// </table>
///

RooDataSet::RooDataSet(RooStringView name, RooStringView title, const RooArgSet& vars, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3,
             const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8)  :
  RooAbsData(name,title,{})
{
  TRACE_CREATE;

  // Define configuration for this method
  RooCmdConfig pc("RooDataSet::ctor(" + std::string(GetName()) + ")");
  pc.defineInt("ownLinked","OwnLinked",0) ;
  pc.defineObject("impTree","ImportTree",0) ;
  pc.defineObject("impData","ImportData",0) ;
  pc.defineObject("indexCat","IndexCat",0) ;
  pc.defineObject("impSliceData","ImportDataSlice",0,nullptr,true) ; // array
  pc.defineString("impSliceState","ImportDataSlice",0,"",true) ; // array
  pc.defineObject("lnkSliceData","LinkDataSlice",0,nullptr,true) ; // array
  pc.defineString("lnkSliceState","LinkDataSlice",0,"",true) ; // array
  pc.defineString("cutSpec","CutSpec",0,"") ;
  pc.defineObject("cutVar","CutVar",0) ;
  pc.defineString("cutRange","CutRange",0,"") ;
  pc.defineString("wgtVarName","WeightVarName",0,"") ;
  pc.defineInt("newWeight1","WeightVarName",0,0) ;
  pc.defineString("fname","ImportFromFile",0,"") ;
  pc.defineString("tname","ImportFromFile",1,"") ;
  pc.defineObject("wgtVar","WeightVar",0) ;
  pc.defineInt("newWeight2","WeightVar",0,0) ;
  pc.defineObject("dummy1","ImportDataSliceMany",0) ;
  pc.defineObject("dummy2","LinkDataSliceMany",0) ;
  pc.defineSet("errorSet","StoreError",0) ;
  pc.defineSet("asymErrSet","StoreAsymError",0) ;
  pc.defineSet("glObs","GlobalObservables",0,nullptr) ;
  pc.defineMutex("ImportTree","ImportData","ImportDataSlice","LinkDataSlice","ImportFromFile") ;
  pc.defineMutex("CutSpec","CutVar") ;
  pc.defineMutex("WeightVarName","WeightVar") ;
  pc.defineDependency("ImportDataSlice","IndexCat") ;
  pc.defineDependency("LinkDataSlice","IndexCat") ;
  pc.defineDependency("OwnLinked","LinkDataSlice") ;


  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  // Process & check varargs
  pc.process(l) ;
  if (!pc.ok(true)) {
    const std::string errMsg = "Error in RooDataSet constructor: command argument list could not be processed";
    coutE(InputArguments) << errMsg << std::endl;
    throw std::invalid_argument(errMsg);
  }

  if(pc.getSet("glObs")) setGlobalObservables(*pc.getSet("glObs"));

  // Extract relevant objects
  TTree* impTree = static_cast<TTree*>(pc.getObject("impTree")) ;
  auto impData = static_cast<RooAbsData*>(pc.getObject("impData")) ;
  RooFormulaVar* cutVar = static_cast<RooFormulaVar*>(pc.getObject("cutVar")) ;
  const char* cutSpec = pc.getString("cutSpec","",true) ;
  const char* cutRange = pc.getString("cutRange","",true) ;
  const char* wgtVarName = pc.getString("wgtVarName","",true) ;
  RooRealVar* wgtVar = static_cast<RooRealVar*>(pc.getObject("wgtVar")) ;
  const char* impSliceNames = pc.getString("impSliceState","",true) ;
  const RooLinkedList& impSliceData = pc.getObjectList("impSliceData") ;
  const char* lnkSliceNames = pc.getString("lnkSliceState","",true) ;
  const RooLinkedList& lnkSliceData = pc.getObjectList("lnkSliceData") ;
  RooCategory* indexCat = static_cast<RooCategory*>(pc.getObject("indexCat")) ;
  RooArgSet* asymErrorSet = pc.getSet("asymErrSet") ;
  const char* fname = pc.getString("fname") ;
  const char* tname = pc.getString("tname") ;
  Int_t ownLinked = pc.getInt("ownLinked") ;
  Int_t newWeight = pc.getInt("newWeight1") + pc.getInt("newWeight2") ;

  // Lookup name of weight variable if it was specified by object reference
  if(wgtVar) {
    wgtVarName = wgtVar->GetName();
  }

  auto finalVarsInfo = finalizeVars(vars,indexCat,wgtVarName,impData,impSliceData, pc.getSet("errorSet"));
  initializeVars(finalVarsInfo.finalVars);
  if(!finalVarsInfo.weightVarName.empty()) {
    wgtVarName = finalVarsInfo.weightVarName.c_str();
  }

  RooArgSet* errorSet =  finalVarsInfo.errorSet.empty() ? nullptr : &finalVarsInfo.errorSet;

  // Case 1 --- Link multiple dataset as slices
  if (lnkSliceNames) {

    // Make import mapping if index category is specified
    map<string,RooAbsData*> hmap ;
    if (indexCat) {
      char tmp[64000];
      strlcpy(tmp, lnkSliceNames, 64000);
      char *token = strtok(tmp, ",");
      auto hiter = lnkSliceData.begin();
      while (token) {
        hmap[token] = static_cast<RooAbsData *>(*hiter);
        token = strtok(nullptr, ",");
        ++hiter;
      }
    }

    appendToDir(this,true) ;

    // Initialize RooDataSet with optional weight variable
    initialize(nullptr) ;

    map<string,RooAbsDataStore*> storeMap ;
    RooCategory* icat = static_cast<RooCategory*> (indexCat ? _vars.find(indexCat->GetName()) : nullptr ) ;
    if (!icat) {
      throw std::string("RooDataSet::RooDataSet() ERROR in constructor, cannot find index category") ;
    }
    for (map<string,RooAbsData*>::iterator hiter = hmap.begin() ; hiter!=hmap.end() ; ++hiter) {
      // Define state labels in index category (both in provided indexCat and in internal copy in dataset)
      if (indexCat && !indexCat->hasLabel(hiter->first)) {
        indexCat->defineType(hiter->first) ;
        coutI(InputArguments) << "RooDataSet::ctor(" << GetName() << ") defining state \"" << hiter->first << "\" in index category " << indexCat->GetName() << std::endl ;
      }
      if (icat && !icat->hasLabel(hiter->first)) {
        icat->defineType(hiter->first) ;
      }
      icat->setLabel(hiter->first.c_str()) ;
      storeMap[icat->getCurrentLabel()]=hiter->second->store() ;

      // Take ownership of slice if requested
      if (ownLinked) {
        addOwnedComponent(hiter->first.c_str(),*hiter->second) ;
      }
    }

    // Create composite datastore
    _dstore = std::make_unique<RooCompositeDataStore>(name,title,_vars,*icat,storeMap) ;

    return;
  }

    // Create empty datastore
    RooTreeDataStore* tstore = nullptr;
    if (defaultStorageType==Tree) {
      _dstore = std::make_unique<RooTreeDataStore>(name,title,_vars,wgtVarName) ;
      tstore = static_cast<RooTreeDataStore*>(_dstore.get());
    } else if (defaultStorageType==Vector) {
      if (wgtVarName && newWeight) {
        RooAbsArg* wgttmp = _vars.find(wgtVarName) ;
        if (wgttmp) {
          wgttmp->setAttribute("NewWeight") ;
        }
      }
      _dstore = std::make_unique<RooVectorDataStore>(name,title,_vars,wgtVarName) ;
    }


    // Make import mapping if index category is specified
    std::map<string,RooAbsData*> hmap ;
    if (indexCat) {
      auto hiter = impSliceData.begin() ;
      for (const auto& token : ROOT::Split(impSliceNames, ",")) {
        hmap[token] = static_cast<RooDataSet*>(*hiter);
        ++hiter;
      }
    }

    // process StoreError requests
    if (errorSet) {
      std::unique_ptr<RooArgSet> intErrorSet{_vars.selectCommon(*errorSet)};
      intErrorSet->setAttribAll("StoreError") ;
      for(RooAbsArg* arg : *intErrorSet) {
        arg->attachToStore(*_dstore) ;
      }
    }
    if (asymErrorSet) {
      std::unique_ptr<RooArgSet> intAsymErrorSet{_vars.selectCommon(*asymErrorSet)};
      intAsymErrorSet->setAttribAll("StoreAsymError") ;
      for(RooAbsArg* arg : *intAsymErrorSet) {
        arg->attachToStore(*_dstore) ;
      }
    }

    appendToDir(this,true) ;

    // Initialize RooDataSet with optional weight variable
    initialize(wgtVarName);

   // Import one or more datasets
   std::unique_ptr<RooFormulaVar> cutVarTmp;

   if (indexCat) {
      // Case 2 --- Import multiple RooDataSets as slices
      loadValuesFromSlices(*indexCat, hmap, cutRange, cutVar, cutSpec);
   } else if (impData) {
      // Case 3 --- Import RooDataSet
      std::unique_ptr<RooDataSet> impDataSet;

      // If we are importing a RooDataHist, first convert it to a RooDataSet
      if(impData->InheritsFrom(RooDataHist::Class())) {
         impDataSet = makeDataSetFromDataHist(static_cast<RooDataHist const &>(*impData));
         impData = impDataSet.get();
      }
      if (cutSpec) {
         cutVarTmp = std::make_unique<RooFormulaVar>(cutSpec, cutSpec, *impData->get(), /*checkVariables=*/false);
         cutVar = cutVarTmp.get();
      }
      _dstore->loadValues(impData->store(), cutVar, cutRange);

   } else if (impTree || (fname && strlen(fname))) {
      // Case 4 --- Import TTree from memory / file
      std::unique_ptr<TFile> file;

      if (impTree == nullptr) {
         file.reset(TFile::Open(fname));
         if (!file) {
            std::stringstream ss;
            ss << "RooDataSet::ctor(" << GetName() << ") ERROR file '" << fname
               << "' cannot be opened or does not exist";
            const std::string errMsg = ss.str();
            coutE(InputArguments) << errMsg << std::endl;
            throw std::invalid_argument(errMsg);
         }

         file->GetObject(tname, impTree);
         if (!impTree) {
            std::stringstream ss;
            ss << "RooDataSet::ctor(" << GetName() << ") ERROR file '" << fname
               << "' does not contain a TTree named '" << tname << "'";
            const std::string errMsg = ss.str();
            coutE(InputArguments) << errMsg << std::endl;
            throw std::invalid_argument(errMsg);
         }
      }

      if (cutSpec) {
         cutVarTmp = std::make_unique<RooFormulaVar>(cutSpec, cutSpec, _vars, /*checkVariables=*/false);
         cutVar = cutVarTmp.get();
      }

      if (tstore) {
         tstore->loadValues(impTree, cutVar, cutRange);
      } else {
         RooTreeDataStore tmpstore(name, title, _vars, wgtVarName);
         tmpstore.loadValues(impTree, cutVar, cutRange);
         _dstore->append(tmpstore);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooDataSet::RooDataSet(RooDataSet const & other, const char* newname) :
  RooAbsData(other,newname), RooDirItem()
{
  appendToDir(this,true) ;
  initialize(other._wgtVar?other._wgtVar->GetName():nullptr);
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Return an empty clone of this dataset. If vars is not null, only the variables in vars
/// are added to the definition of the empty clone

RooFit::OwningPtr<RooAbsData> RooDataSet::emptyClone(const char* newName, const char* newTitle, const RooArgSet* vars, const char* wgtVarName) const
{
   bool useOldWeight = _wgtVar && (wgtVarName == nullptr || strcmp(wgtVarName, _wgtVar->GetName()) == 0);

   if(newName == nullptr) newName = GetName();
   if(newTitle == nullptr) newTitle = GetTitle();
   if(useOldWeight) wgtVarName = _wgtVar->GetName();

   RooArgSet vars2;
   if(vars == nullptr) {
      vars2.add(_vars);
   } else {
      for(RooAbsArg *var : *vars) {
         // We should take the variables from the original dataset if
         // available, such that we can query the "StoreError" and
         // "StoreAsymError" attributes.
         auto varInData = _vars.find(*var);
         vars2.add(varInData ? *varInData : *var);
      }
      // We also need to add the weight variable of the original dataset if
      // it's not added yet, again to query the error attributes correctly.
      if(useOldWeight && !vars2.find(wgtVarName)) vars2.add(*_wgtVar);
   }

   RooArgSet errorSet;
   RooArgSet asymErrorSet;

   for(RooAbsArg *var : vars2) {
      if(var->getAttribute("StoreError")) errorSet.add(*var);
      if(var->getAttribute("StoreAsymError")) asymErrorSet.add(*var);
   }

   using namespace RooFit;
   return RooFit::makeOwningPtr<RooAbsData>(std::make_unique<RooDataSet>(
      newName, newTitle, vars2, WeightVar(wgtVarName), StoreError(errorSet), StoreAsymError(asymErrorSet)));
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize the dataset. If wgtVarName is not null, interpret the observable
/// with that name as event weight

void RooDataSet::initialize(const char* wgtVarName)
{
  _varsNoWgt.removeAll() ;
  _varsNoWgt.add(_vars) ;
  _wgtVar = nullptr ;
  if (wgtVarName) {
    RooAbsArg* wgt = _varsNoWgt.find(wgtVarName) ;
    if (!wgt) {
      coutE(DataHandling) << "RooDataSet::RooDataSet(" << GetName() << "): designated weight variable "
           << wgtVarName << " not found in set of variables, no weighting will be assigned" << std::endl ;
      throw std::invalid_argument("RooDataSet::initialize() weight variable could not be initialised.");
    } else if (!dynamic_cast<RooRealVar*>(wgt)) {
      coutE(DataHandling) << "RooDataSet::RooDataSet(" << GetName() << "): designated weight variable "
           << wgtVarName << " is not of type RooRealVar, no weighting will be assigned" << std::endl ;
      throw std::invalid_argument("RooDataSet::initialize() weight variable could not be initialised.");
    } else {
      _varsNoWgt.remove(*wgt) ;
      _wgtVar = static_cast<RooRealVar*>(wgt) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods

std::unique_ptr<RooAbsData> RooDataSet::reduceEng(const RooArgSet &varSubset, const RooFormulaVar *cutVar,
                                                  const char *cutRange, std::size_t nStart, std::size_t nStop) const
{
   checkInit();
   RooArgSet tmp(varSubset);
   if (_wgtVar) {
      tmp.add(*_wgtVar);
   }

   auto createEmptyClone = [&]() { return emptyClone(GetName(), GetTitle(), &tmp); };

   std::unique_ptr<RooAbsData> out{createEmptyClone()};

   if (!cutRange || strchr(cutRange, ',') == nullptr) {
      auto &ds = static_cast<RooDataSet &>(*out);
      ds._dstore = _dstore->reduce(ds.GetName(), ds.GetTitle(), ds._vars, cutVar, cutRange, nStart, nStop);
      ds._cachedVars.add(_dstore->cachedVars());
   } else {
      // Composite case: multiple ranges
      auto tokens = ROOT::Split(cutRange, ",");
      if (RooHelpers::checkIfRangesOverlap(tmp, tokens)) {
         std::stringstream errMsg;
         errMsg << "Error in RooAbsData::reduce! The ranges " << cutRange << " are overlapping!";
         throw std::runtime_error(errMsg.str());
      }
      for (const auto &token : tokens) {
         std::unique_ptr<RooAbsData> appendedData{createEmptyClone()};
         auto &ds = static_cast<RooDataSet &>(*appendedData);
         ds._dstore = _dstore->reduce(ds.GetName(), ds.GetTitle(), ds._vars, cutVar, token.c_str(), nStart, nStop);
         ds._cachedVars.add(_dstore->cachedVars());
         static_cast<RooDataSet &>(*out).append(ds);
      }
   }
   return out;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooDataSet::~RooDataSet()
{
  removeFromDir(this) ;
  TRACE_DESTROY;
}



////////////////////////////////////////////////////////////////////////////////
/// Return binned clone of this dataset

RooFit::OwningPtr<RooDataHist> RooDataSet::binnedClone(const char* newName, const char* newTitle) const
{
  std::string title;
  std::string name;
  if (newName) {
    name = newName ;
  } else {
    name = std::string(GetName()) + "_binned" ;
  }
  if (newTitle) {
    title = newTitle ;
  } else {
    title = std::string(GetTitle()) + "_binned" ;
  }

  return RooFit::makeOwningPtr(std::make_unique<RooDataHist>(name,title,*get(),*this));
}



////////////////////////////////////////////////////////////////////////////////
/// Return event weight of current event

double RooDataSet::weight() const
{
  return store()->weight() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return squared event weight of the current event. If this RooDataSet has no
/// weight errors set, this will be the same as `weight() * weight()`, like
/// expected for an unbinned dataset. When weight errors are set, it is assumed
/// that the RooDataSet represents a weighted binned dataset and
/// weightSquared() is the corresponding sum of weight squares for the bin.

double RooDataSet::weightSquared() const
{
  const double w = store()->weight();
  const double e = weightError();
  return e > 0.0 ? e * e : w * w;
}


////////////////////////////////////////////////////////////////////////////////
/// \see RooAbsData::getWeightBatch().
std::span<const double> RooDataSet::getWeightBatch(std::size_t first, std::size_t len, bool sumW2 /*=false*/) const {

  std::size_t nEntries = this->numEntries(); // for the casting to std::size_t

  if(first + len > nEntries) {
    throw std::runtime_error("RooDataSet::getWeightBatch(): requested range not valid for dataset.");
  }

  std::span<const double> allWeights = _dstore->getWeightBatch(0, numEntries());
  if(allWeights.empty()) return {};

  if(!sumW2) return {&*(std::cbegin(allWeights) + first), len};

  // Treat the sumW2 case with a result buffer, first reset buffer if the
  // number of entries doesn't match with the dataset anymore
  if(_sumW2Buffer && _sumW2Buffer->size() != nEntries) _sumW2Buffer.reset();

  if (!_sumW2Buffer) {
    _sumW2Buffer = std::make_unique<std::vector<double>>();
    _sumW2Buffer->reserve(nEntries);

    for (std::size_t i = 0; i < nEntries; ++i) {
      get(i);
      _sumW2Buffer->push_back(weightSquared());
    }
  }

  return std::span<const double>(&*(_sumW2Buffer->begin() + first), len);
}


////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooAbsData::weightError(double&,double&,RooAbsData::ErrorType) const
/// \param etype error type
void RooDataSet::weightError(double& lo, double& hi, ErrorType etype) const
{
  store()->weightError(lo,hi,etype) ;
}


////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooAbsData::weightError(ErrorType)
/// \param etype error type
double RooDataSet::weightError(ErrorType etype) const
{
  return store()->weightError(etype) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return RooArgSet with coordinates of event 'index'

const RooArgSet* RooDataSet::get(Int_t index) const
{
  const RooArgSet* ret  = RooAbsData::get(index) ;
  return ret ? &_varsNoWgt : nullptr ;
}


////////////////////////////////////////////////////////////////////////////////

double RooDataSet::sumEntries() const
{
  return store()->sumEntries() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the sum of weights in all entries matching cutSpec (if specified)
/// and in named range cutRange (if specified)

double RooDataSet::sumEntries(const char* cutSpec, const char* cutRange) const
{
  // Setup RooFormulaVar for cutSpec if it is present
  std::unique_ptr<RooFormula> select = nullptr ;
  if (cutSpec && strlen(cutSpec) > 0) {
    select = std::make_unique<RooFormula>("select",cutSpec,*get()) ;
  }

  // Shortcut for unweighted unselected datasets
  if (!select && !cutRange && !isWeighted()) {
    return numEntries() ;
  }

  // Otherwise sum the weights in the event
  ROOT::Math::KahanSum<double> sumw{0.0};
  for (int i = 0 ; i<numEntries() ; i++) {
    get(i) ;
    if (select && select->eval()==0.) continue ;
    if (cutRange && !_vars.allInRange(cutRange)) continue ;
    sumw += weight();
  }

  return sumw.Sum() ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return true if dataset contains weighted events

bool RooDataSet::isWeighted() const
{
    return store() ? store()->isWeighted() : false;
}



////////////////////////////////////////////////////////////////////////////////
/// Returns true if histogram contains bins with entries with a non-integer weight

bool RooDataSet::isNonPoissonWeighted() const
{
  // Return false if we have no weights
  if (!_wgtVar) return false ;

  // Now examine individual weights
  for (int i=0 ; i<numEntries() ; i++) {
    get(i) ;
    if (std::abs(weight()-Int_t(weight()))>1e-10) return true ;
  }
  // If sum of weights is less than number of events there are negative (integer) weights
  if (sumEntries()<numEntries()) return true ;

  return false ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return a RooArgSet with the coordinates of the current event

const RooArgSet* RooDataSet::get() const
{
  return &_varsNoWgt ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a data point, with its coordinates specified in the 'data' argset, to the data set.
/// Any variables present in 'data' but not in the dataset will be silently ignored.
/// \param[in] data Data point.
/// \param[in] wgt Event weight. Defaults to 1. The current value of the weight variable is
/// ignored.
/// \note To obtain weighted events, a variable must be designated `WeightVar` in the constructor.
/// \param[in] wgtError Optional weight error.
/// \note This requires including the weight variable in the set of `StoreError` variables when constructing
/// the dataset.

void RooDataSet::add(const RooArgSet& data, double wgt, double wgtError)
{
  checkInit() ;

  const double oldW = _wgtVar ? _wgtVar->getVal() : 0.;

  _varsNoWgt.assign(data);

  if (_wgtVar) {
    _wgtVar->setVal(wgt) ;
    if (wgtError!=0.) {
      _wgtVar->setError(wgtError) ;
    }
  } else if ((wgt != 1. || wgtError != 0.) && _errorMsgCount < 5) {
    ccoutE(DataHandling) << "An event weight/error was passed but no weight variable was defined"
        << " in the dataset '" << GetName() << "'. The weight will be ignored." << std::endl;
    ++_errorMsgCount;
  }

  if (_wgtVar && _doWeightErrorCheck
      && wgtError != 0.
      && std::abs(wgt*wgt - wgtError)/wgtError > 1.E-15 //Exception for standard wgt^2 errors, which need not be stored.
      && _errorMsgCount < 5 && !_wgtVar->getAttribute("StoreError")) {
    coutE(DataHandling) << "An event weight error was passed to the RooDataSet '" << GetName()
        << "', but the weight variable '" << _wgtVar->GetName()
        << "' does not store errors. Check `StoreError` in the RooDataSet constructor." << std::endl;
    ++_errorMsgCount;
  }

  fill();

  // Restore weight state
  if (_wgtVar) {
    _wgtVar->setVal(oldW);
    _wgtVar->removeError();
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Add a data point, with its coordinates specified in the 'data' argset, to the data set.
/// Any variables present in 'data' but not in the dataset will be silently ignored.
/// \param[in] indata Data point.
/// \param[in] inweight Event weight. The current value of the weight variable is ignored.
/// \note To obtain weighted events, a variable must be designated `WeightVar` in the constructor.
/// \param[in] weightErrorLo Asymmetric weight error.
/// \param[in] weightErrorHi Asymmetric weight error.
/// \note This requires including the weight variable in the set of `StoreAsymError` variables when constructing
/// the dataset.

void RooDataSet::add(const RooArgSet& indata, double inweight, double weightErrorLo, double weightErrorHi)
{
  checkInit() ;

  const double oldW = _wgtVar ? _wgtVar->getVal() : 0.;

  _varsNoWgt.assign(indata);
  if (_wgtVar) {
    _wgtVar->setVal(inweight) ;
    _wgtVar->setAsymError(weightErrorLo,weightErrorHi) ;
  } else if (inweight != 1. && _errorMsgCount < 5) {
    ccoutE(DataHandling) << "An event weight was given but no weight variable was defined"
        << " in the dataset '" << GetName() << "'. The weight will be ignored." << std::endl;
    ++_errorMsgCount;
  }

  if (_wgtVar && _doWeightErrorCheck
      && _errorMsgCount < 5 && !_wgtVar->getAttribute("StoreAsymError")) {
    coutE(DataHandling) << "An event weight error was passed to the RooDataSet '" << GetName()
        << "', but the weight variable '" << _wgtVar->GetName()
        << "' does not store errors. Check `StoreAsymError` in the RooDataSet constructor." << std::endl;
    ++_errorMsgCount;
  }

  fill();

  // Restore weight state
  if (_wgtVar) {
    _wgtVar->setVal(oldW);
    _wgtVar->removeAsymError();
  }
}





////////////////////////////////////////////////////////////////////////////////
/// Add a data point, with its coordinates specified in the 'data' argset, to the data set.
/// \attention The order and type of the input variables are **assumed** to be the same as
/// for the RooArgSet returned by RooDataSet::get(). Input values will just be written
/// into the internal data columns by ordinal position.
/// \param[in] data Data point.
/// \param[in] wgt Event weight. Defaults to 1. The current value of the weight variable is
/// ignored.
/// \note To obtain weighted events, a variable must be designated `WeightVar` in the constructor.
/// \param[in] wgtError Optional weight error.
/// \note This requires including the weight variable in the set of `StoreError` variables when constructing
/// the dataset.

void RooDataSet::addFast(const RooArgSet& data, double wgt, double wgtError)
{
  checkInit() ;

  const double oldW = _wgtVar ? _wgtVar->getVal() : 0.;

  _varsNoWgt.assignFast(data,_dstore->dirtyProp());
  if (_wgtVar) {
    _wgtVar->setVal(wgt) ;
    if (wgtError!=0.) {
      _wgtVar->setError(wgtError) ;
    }
  } else if (wgt != 1. && _errorMsgCount < 5) {
    ccoutE(DataHandling) << "An event weight was given but no weight variable was defined"
        << " in the dataset '" << GetName() << "'. The weight will be ignored." << std::endl;
    ++_errorMsgCount;
  }

  fill();

  if (_wgtVar && _doWeightErrorCheck
      && wgtError != 0. && wgtError != wgt*wgt //Exception for standard weight error, which need not be stored
      && _errorMsgCount < 5 && !_wgtVar->getAttribute("StoreError")) {
    coutE(DataHandling) << "An event weight error was passed to the RooDataSet '" << GetName()
        << "', but the weight variable '" << _wgtVar->GetName()
        << "' does not store errors. Check `StoreError` in the RooDataSet constructor." << std::endl;
    ++_errorMsgCount;
  }
  if (_wgtVar && _doWeightErrorCheck) {
    _doWeightErrorCheck = false;
  }

  if (_wgtVar) {
    _wgtVar->setVal(oldW);
    _wgtVar->removeError();
  }
}



////////////////////////////////////////////////////////////////////////////////

bool RooDataSet::merge(RooDataSet* data1, RooDataSet* data2, RooDataSet* data3,
          RooDataSet* data4, RooDataSet* data5, RooDataSet* data6)
{
  checkInit() ;
  list<RooDataSet*> dsetList ;
  if (data1) dsetList.push_back(data1) ;
  if (data2) dsetList.push_back(data2) ;
  if (data3) dsetList.push_back(data3) ;
  if (data4) dsetList.push_back(data4) ;
  if (data5) dsetList.push_back(data5) ;
  if (data6) dsetList.push_back(data6) ;
  return merge(dsetList) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Merge columns of supplied data set(s) with this data set.  All
/// data sets must have equal number of entries.  In case of
/// duplicate columns the column of the last dataset in the list
/// prevails

bool RooDataSet::merge(list<RooDataSet*>dsetList)
{

  checkInit() ;
  // Sanity checks: data sets must have the same size
  for (list<RooDataSet*>::iterator iter = dsetList.begin() ; iter != dsetList.end() ; ++iter) {
    if (numEntries()!=(*iter)->numEntries()) {
      coutE(InputArguments) << "RooDataSet::merge(" << GetName() << ") ERROR: datasets have different size" << std::endl ;
      return true ;
    }
  }

  // Extend vars with elements of other dataset
  list<RooAbsDataStore*> dstoreList ;
  for (list<RooDataSet*>::iterator iter = dsetList.begin() ; iter != dsetList.end() ; ++iter) {
    _vars.addClone((*iter)->_vars,true) ;
    dstoreList.push_back((*iter)->store()) ;
  }

  // Merge data stores
  RooAbsDataStore* mergedStore = _dstore->merge(_vars,dstoreList) ;
  mergedStore->SetName(_dstore->GetName()) ;
  mergedStore->SetTitle(_dstore->GetTitle()) ;

  // Replace current data store with merged store
  _dstore.reset(mergedStore);

  initialize(_wgtVar?_wgtVar->GetName():nullptr) ;
  return false ;
}


////////////////////////////////////////////////////////////////////////////////
/// Add all data points of given data set to this data set.
/// Observable in 'data' that are not in this dataset
/// with not be transferred

void RooDataSet::append(RooDataSet& data)
{
  checkInit() ;
  _dstore->append(*data._dstore) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a column with the values of the given (function) argument
/// to this dataset. The function value is calculated for each
/// event using the observable values of each event in case the
/// function depends on variables with names that are identical
/// to the observable names in the dataset

RooAbsArg* RooDataSet::addColumn(RooAbsArg& var, bool adjustRange)
{
  checkInit() ;
  std::unique_ptr<RooAbsArg> ret{_dstore->addColumn(var,adjustRange)};
  RooAbsArg* retPtr = ret.get();
  _vars.addOwned(std::move(ret));
  initialize(_wgtVar?_wgtVar->GetName():nullptr) ;
  return retPtr;
}


////////////////////////////////////////////////////////////////////////////////
/// Special plot method for 'X-Y' datasets used in \f$ \chi^2 \f$ fitting.
/// For general plotting, see RooAbsData::plotOn().
///
/// These datasets
/// have one observable (X) and have weights (Y) and associated errors.
/// <table>
/// <tr><th> Contents options       <th> Effect
/// <tr><td> YVar(RooRealVar& var)  <td> Designate specified observable as 'y' variable
///                                    If not specified, the event weight will be the y variable
/// <tr><th> Histogram drawing options <th> Effect
/// <tr><td> DrawOption(const char* opt)     <td> Select ROOT draw option for resulting TGraph object
/// <tr><td> LineStyle(Int_t style)          <td> Select line style by ROOT line style code, default is solid
/// <tr><td> LineColor(Int_t color)          <td> Select line color by ROOT color code, default is black
/// <tr><td> LineWidth(Int_t width)          <td> Select line with in pixels, default is 3
/// <tr><td> MarkerStyle(Int_t style)        <td> Select the ROOT marker style, default is 21
/// <tr><td> MarkerColor(Int_t color)        <td> Select the ROOT marker color, default is black
/// <tr><td> MarkerSize(double size)       <td> Select the ROOT marker size
/// <tr><td> Rescale(double factor)        <td> Apply global rescaling factor to histogram
/// <tr><th> Misc. other options <th> Effect
/// <tr><td> Name(const chat* name)          <td> Give curve specified name in frame. Useful if curve is to be referenced later
/// <tr><td> Invisible(bool flag)          <td> Add curve to frame, but do not display. Useful in combination AddTo()
/// </table>

RooPlot* RooDataSet::plotOnXY(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
               const RooCmdArg& arg3, const RooCmdArg& arg4,
               const RooCmdArg& arg5, const RooCmdArg& arg6,
               const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  checkInit() ;

  RooLinkedList argList ;
  argList.Add((TObject*)&arg1) ;  argList.Add((TObject*)&arg2) ;
  argList.Add((TObject*)&arg3) ;  argList.Add((TObject*)&arg4) ;
  argList.Add((TObject*)&arg5) ;  argList.Add((TObject*)&arg6) ;
  argList.Add((TObject*)&arg7) ;  argList.Add((TObject*)&arg8) ;

  // Process named arguments
  RooCmdConfig pc("RooDataSet::plotOnXY(" + std::string(GetName()) + ")");
  pc.defineString("drawOption","DrawOption",0,"P") ;
  pc.defineString("histName","Name",0,"") ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,8) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineInt("histInvisible","Invisible",0,0) ;
  pc.defineDouble("scaleFactor","Rescale",0,1.) ;
  pc.defineObject("xvar","XVar",0,nullptr) ;
  pc.defineObject("yvar","YVar",0,nullptr) ;


  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  // Extract values from named arguments
  const char* drawOptions = pc.getString("drawOption") ;
  Int_t histInvisible = pc.getInt("histInvisible") ;
  const char* histName = pc.getString("histName",nullptr,true) ;
  double scaleFactor = pc.getDouble("scaleFactor") ;

  RooRealVar* xvar = static_cast<RooRealVar*>(_vars.find(frame->getPlotVar()->GetName())) ;

  // Determine Y variable (default is weight, if present)
  RooRealVar* yvar = static_cast<RooRealVar*>(pc.getObject("yvar")) ;

  // Sanity check. XY plotting only applies to weighted datasets if no YVar is specified
  if (!_wgtVar && !yvar) {
    coutE(InputArguments) << "RooDataSet::plotOnXY(" << GetName() << ") ERROR: no YVar() argument specified and dataset is not weighted" << std::endl ;
    return nullptr ;
  }

  RooRealVar* dataY = yvar ? static_cast<RooRealVar*>(_vars.find(yvar->GetName())) : nullptr ;
  if (yvar && !dataY) {
    coutE(InputArguments) << "RooDataSet::plotOnXY(" << GetName() << ") ERROR on YVar() argument, dataset does not contain a variable named " << yvar->GetName() << std::endl ;
    return nullptr ;
  }


  // Make RooHist representing XY contents of data
  RooHist* graph = new RooHist ;
  if (histName) {
    graph->SetName(histName) ;
  } else {
    graph->SetName(("hxy_" + std::string(GetName())).c_str());
  }

  for (int i=0 ; i<numEntries() ; i++) {
    get(i) ;
    double x = xvar->getVal() ;
    double exlo = xvar->getErrorLo() ;
    double exhi = xvar->getErrorHi() ;
    double y;
    double eylo;
    double eyhi;
    if (!dataY) {
      y = weight() ;
      weightError(eylo,eyhi) ;
    } else {
      y = dataY->getVal() ;
      eylo = dataY->getErrorLo() ;
      eyhi = dataY->getErrorHi() ;
    }
    graph->addBinWithXYError(x,y,-1*exlo,exhi,-1*eylo,eyhi,scaleFactor) ;
  }

  // Adjust style options according to named arguments
  Int_t lineColor   = pc.getInt("lineColor") ;
  Int_t lineStyle   = pc.getInt("lineStyle") ;
  Int_t lineWidth   = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;

  if (lineColor!=-999) graph->SetLineColor(lineColor) ;
  if (lineStyle!=-999) graph->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) graph->SetLineWidth(lineWidth) ;
  if (markerColor!=-999) graph->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) graph->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) graph->SetMarkerSize(markerSize) ;
  if (fillColor!=-999) graph->SetFillColor(fillColor) ;
  if (fillStyle!=-999) graph->SetFillStyle(fillStyle) ;

  // Add graph to frame
  frame->addPlotable(graph,drawOptions,histInvisible) ;

  return frame ;
}




////////////////////////////////////////////////////////////////////////////////
/// Read given list of ascii files, and construct a data set, using the given
/// ArgList as structure definition.
/// \param fileList Multiple file names, comma separated. Each
/// file is optionally prefixed with 'commonPath' if such a path is
/// provided
///
/// \param varList Specify the dimensions of the dataset to be built.
/// This list describes the order in which these dimensions appear in the
/// ascii files to be read.
/// Each line in the ascii file should contain N white-space separated
/// tokens, with N the number of args in `varList`. Any text beyond
/// N tokens will be ignored with a warning message.
/// (NB: This is the default output of RooArgList::writeToStream())
///
/// \param verbOpt `Q` be quiet, `D` debug mode (verbose)
///
/// \param commonPath All filenames in `fileList` will be prefixed with this optional path.
///
/// \param indexCatName Interpret the data as belonging to category `indexCatName`.
/// When multiple files are read, a RooCategory arg in `varList` can
/// optionally be designated to hold information about the source file
/// of each data point. This feature is enabled by giving the name
/// of the (already existing) category variable in `indexCatName`.
///
/// \attention If the value of any of the variables on a given line exceeds the
/// fit range associated with that dimension, the entire line will be
/// ignored. A warning message is printed in each case, unless the
/// `Q` verbose option is given. The number of events read and skipped
/// is always summarized at the end.
///
/// If no further information is given a label name 'fileNNN' will
/// be assigned to each event, where NNN is the sequential number of
/// the source file in `fileList`.
///
/// Alternatively, it is possible to override the default label names
/// of the index category by specifying them in the fileList string:
/// When instead of `file1.txt,file2.txt` the string
/// `file1.txt:FOO,file2.txt:BAR` is specified, a state named "FOO"
/// is assigned to the index category for each event originating from
/// file1.txt. The labels FOO,BAR may be predefined in the index
/// category via defineType(), but don't have to be.
///
/// Finally, one can also assign the same label to multiple files,
/// either by specifying `file1.txt:FOO,file2,txt:FOO,file3.txt:BAR`
/// or `file1.txt,file2.txt:FOO,file3.txt:BAR`.
///

RooDataSet *RooDataSet::read(const char *fileList, const RooArgList &varList,
              const char *verbOpt, const char* commonPath,
              const char* indexCatName) {
  // Make working copy of variables list
  RooArgList variables(varList) ;

  // Append blinding state category to variable list if not already there
  bool ownIsBlind(true) ;
  RooAbsArg* blindState = variables.find("blindState") ;
  if (!blindState) {
    blindState = new RooCategory("blindState","Blinding State") ;
    variables.add(*blindState) ;
  } else {
    ownIsBlind = false ;
    if (blindState->IsA()!=RooCategory::Class()) {
      oocoutE(nullptr,DataHandling) << "RooDataSet::read: ERROR: variable list already contains"
          << "a non-RooCategory blindState member" << std::endl ;
      return nullptr ;
    }
    oocoutW(nullptr,DataHandling) << "RooDataSet::read: WARNING: recycling existing "
        << "blindState category in variable list" << std::endl ;
  }
  RooCategory* blindCat = static_cast<RooCategory*>(blindState) ;

  // Configure blinding state category
  blindCat->setAttribute("Dynamic") ;
  blindCat->defineType("Normal",0) ;
  blindCat->defineType("Blind",1) ;

  // parse the option string
  TString opts= verbOpt;
  opts.ToLower();
  bool verbose= !opts.Contains("q");
  bool debug= opts.Contains("d");

  auto data = std::make_unique<RooDataSet>("dataset", fileList, variables);
  if (ownIsBlind) { variables.remove(*blindState) ; delete blindState ; }
  if(!data) {
    oocoutE(nullptr,DataHandling) << "RooDataSet::read: unable to create a new dataset"
        << std::endl;
    return nullptr;
  }

  // Redirect blindCat to point to the copy stored in the data set
  blindCat = static_cast<RooCategory*>(data->_vars.find("blindState")) ;

  // Find index category, if requested
  RooCategory *indexCat     = nullptr;
  //RooCategory *indexCatOrig = 0;
  if (indexCatName) {
    RooAbsArg* tmp = nullptr;
    tmp = data->_vars.find(indexCatName) ;
    if (!tmp) {
      oocoutE(data.get(),DataHandling) << "RooDataSet::read: no index category named "
          << indexCatName << " in supplied variable list" << std::endl ;
      return nullptr;
    }
    if (tmp->IsA()!=RooCategory::Class()) {
      oocoutE(data.get(),DataHandling) << "RooDataSet::read: variable " << indexCatName
          << " is not a RooCategory" << std::endl ;
      return nullptr;
    }
    indexCat = static_cast<RooCategory*>(tmp);

    // Prevent RooArgSet from attempting to read in indexCat
    indexCat->setAttribute("Dynamic") ;
  }


  Int_t outOfRange(0) ;

  // Loop over all names in comma separated list
  Int_t fileSeqNum(0);
  for (const auto& filename : ROOT::Split(std::string(fileList), ", ")) {
    // Determine index category number, if this option is active
    if (indexCat) {

      // Find and detach optional file category name
      const char *catname = strchr(filename.c_str(),':');

      if (catname) {
        // Use user category name if provided
        catname++ ;

        if (indexCat->hasLabel(catname)) {
          // Use existing category index
          indexCat->setLabel(catname);
        } else {
          // Register cat name
          indexCat->defineType(catname,fileSeqNum) ;
          indexCat->setIndex(fileSeqNum) ;
        }
      } else {
        // Assign autogenerated name
        char newLabel[128] ;
        snprintf(newLabel,128,"file%03d",fileSeqNum) ;
        if (indexCat->defineType(newLabel,fileSeqNum)) {
          oocoutE(data.get(), DataHandling) << "RooDataSet::read: Error, cannot register automatic type name " << newLabel
              << " in index category " << indexCat->GetName() << std::endl ;
          return nullptr ;
        }
        // Assign new category number
        indexCat->setIndex(fileSeqNum) ;
      }
    }

    oocoutI(data.get(), DataHandling) << "RooDataSet::read: reading file " << filename << std::endl ;

    // Prefix common path
    TString fullName(commonPath) ;
    fullName.Append(filename) ;
    ifstream file(fullName) ;

    if (!file.good()) {
      oocoutE(data.get(), DataHandling) << "RooDataSet::read: unable to open '"
          << filename << "'. Returning nullptr now." << std::endl;
      return nullptr;
    }

    //  double value;
    Int_t line(0) ;
    bool haveBlindString(false) ;

    while(file.good() && !file.eof()) {
      line++;
      if(debug) oocxcoutD(data.get(),DataHandling) << "reading line " << line << std::endl;

      // process comment lines
      if (file.peek() == '#') {
        if(debug) oocxcoutD(data.get(),DataHandling) << "skipping comment on line " << line << std::endl;
      } else {
        // Read single line
        bool readError = variables.readFromStream(file,true,verbose) ;
        data->_vars.assign(variables) ;

        // Stop on read error
        if(!file.good()) {
          oocoutE(data.get(), DataHandling) << "RooDataSet::read(static): read error at line " << line << std::endl ;
          break;
        }

        if (readError) {
          outOfRange++ ;
        } else {
          blindCat->setIndex(haveBlindString) ;
          data->fill(); // store this event
        }
      }

      // Skip all white space (including empty lines).
      while (isspace(file.peek())) {
        char dummy;
        file >> std::noskipws >> dummy >> std::skipws;
      }
    }

    file.close();

    // get next file name
    fileSeqNum++ ;
  }

  if (indexCat) {
    // Copy dynamically defined types from new data set to indexCat in original list
    assert(dynamic_cast<RooCategory*>(variables.find(indexCatName)));
    const auto origIndexCat = static_cast<RooCategory*>(variables.find(indexCatName));
    for (const auto& nameIdx : *indexCat) {
      origIndexCat->defineType(nameIdx.first, nameIdx.second);
    }
  }
  oocoutI(data.get(),DataHandling) << "RooDataSet::read: read " << data->numEntries()
                    << " events (ignored " << outOfRange << " out of range events)" << std::endl;

  return data.release();
}




////////////////////////////////////////////////////////////////////////////////
/// Write the contents of this dataset to an ASCII file with the specified name.
/// Each event will be written as a single line containing the written values
/// of each observable in the order they were declared in the dataset and
/// separated by whitespaces

bool RooDataSet::write(const char* filename) const
{
  // Open file for writing
  ofstream ofs(filename) ;
  if (ofs.fail()) {
    coutE(DataHandling) << "RooDataSet::write(" << GetName() << ") cannot create file " << filename << std::endl ;
    return true ;
  }

  // Write all lines as arglist in compact mode
  coutI(DataHandling) << "RooDataSet::write(" << GetName() << ") writing ASCII file " << filename << std::endl ;
  return write(ofs);
}

////////////////////////////////////////////////////////////////////////////////
/// Write the contents of this dataset to the stream.
/// Each event will be written as a single line containing the written values
/// of each observable in the order they were declared in the dataset and
/// separated by whitespaces

bool RooDataSet::write(ostream & ofs) const {
  checkInit();

  for (Int_t i=0; i<numEntries(); ++i) {
    get(i)->writeToStream(ofs,true);
  }

  if (ofs.fail()) {
    coutW(DataHandling) << "RooDataSet::write(" << GetName() << "): WARNING error(s) have occurred in writing" << std::endl ;
  }

  return ofs.fail() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print info about this dataset to the specified output stream.
///
///   Standard: number of entries
///      Shape: list of variables we define & were generated with

void RooDataSet::printMultiline(ostream& os, Int_t contents, bool verbose, TString indent) const
{
  checkInit() ;
  RooAbsData::printMultiline(os,contents,verbose,indent) ;
  if (_wgtVar) {
    os << indent << "  Dataset variable \"" << _wgtVar->GetName() << "\" is interpreted as the event weight" << std::endl ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Print value of the dataset, i.e. the sum of weights contained in the dataset

void RooDataSet::printValue(ostream& os) const
{
  os << numEntries() << " entries" ;
  if (isWeighted()) {
    os << " (" << sumEntries() << " weighted)" ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print argument of dataset, i.e. the observable names

void RooDataSet::printArgs(ostream& os) const
{
  os << "[" ;
  bool first(true) ;
  for(RooAbsArg* arg : _varsNoWgt) {
    if (first) {
      first=false ;
    } else {
      os << "," ;
    }
    os << arg->GetName() ;
  }
  if (_wgtVar) {
    os << ",weight:" << _wgtVar->GetName() ;
  }
  os << "]" ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change the name of this dataset into the given name

void RooDataSet::SetName(const char *name)
{
  if (_dir) _dir->GetList()->Remove(this);
  // We need to use the function from RooAbsData, because it already overrides TNamed::SetName
  RooAbsData::SetName(name);
  if (_dir) _dir->GetList()->Add(this);
}


////////////////////////////////////////////////////////////////////////////////
/// Change the title of this dataset into the given name

void RooDataSet::SetNameTitle(const char *name, const char* title)
{
  SetName(name);
  SetTitle(title);
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooDataSet.

void RooDataSet::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {

      UInt_t R__s;
      UInt_t R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

      if (R__v > 1) {

         // Use new-style streaming for version >1
         R__b.ReadClassBuffer(RooDataSet::Class(), this, R__v, R__s, R__c);

      } else {

         // Legacy dataset conversion happens here. Legacy RooDataSet inherits from RooTreeData
         // which in turn inherits from RooAbsData. Manually stream RooTreeData contents on
         // file here and convert it into a RooTreeDataStore which is installed in the
         // new-style RooAbsData base class

         // --- This is the contents of the streamer code of RooTreeData version 1 ---
         UInt_t R__s1;
         UInt_t R__c1;
         Version_t R__v1 = R__b.ReadVersion(&R__s1, &R__c1);
         if (R__v1) {
         }

         RooAbsData::Streamer(R__b);
         TTree *X_tree(nullptr);
         R__b >> X_tree;
         RooArgSet X_truth;
         X_truth.Streamer(R__b);
         TString X_blindString;
         X_blindString.Streamer(R__b);
         R__b.CheckByteCount(R__s1, R__c1, TClass::GetClass("RooTreeData"));
         // --- End of RooTreeData-v1 streamer

         // Construct RooTreeDataStore from X_tree and complete initialization of new-style RooAbsData
         _dstore = std::make_unique<RooTreeDataStore>(X_tree, _vars);
         _dstore->SetName(GetName());
         _dstore->SetTitle(GetTitle());
         _dstore->checkInit();

         // This is the contents of the streamer code of RooDataSet version 1
         RooDirItem::Streamer(R__b);
         _varsNoWgt.Streamer(R__b);
         R__b >> _wgtVar;
         R__b.CheckByteCount(R__s, R__c, RooDataSet::IsA());
      }
   } else {
      R__b.WriteClassBuffer(RooDataSet::Class(), this);
   }
}



////////////////////////////////////////////////////////////////////////////////
/// Convert vector-based storage to tree-based storage. This implementation overrides the base class
/// implementation because the latter doesn't transfer weights.
void RooDataSet::convertToTreeStore()
{
   if (storageType != RooAbsData::Tree) {
      _dstore = std::make_unique<RooTreeDataStore>(GetName(), GetTitle(), _vars, *_dstore, nullptr, _wgtVar ? _wgtVar->GetName() : nullptr);
      storageType = RooAbsData::Tree;
   }
}


namespace {

  // Compile-time test if we can still use TStrings for the constructors of
  // RooDataClasses, either for both name and title or for only one of them.
  TString tstr = "tstr";
  const char * cstr = "cstr";
  RooRealVar x{"x", "x", 1.0};
  RooArgSet vars{x};
  RooDataSet d1(tstr, tstr, vars);
  RooDataSet d2(tstr, cstr, vars);
  RooDataSet d3(cstr, tstr, vars);

} // namespace


void RooDataSet::loadValuesFromSlices(RooCategory &indexCat, std::map<std::string, RooAbsData *> const &slices,
                                      const char *rangeName, RooFormulaVar const *cutVar, const char *cutSpec)
{

   if (cutVar && cutSpec) {
      throw std::invalid_argument("Only one of cutVar or cutSpec should be not a nullptr!");
   }

   auto &indexCatInData = *static_cast<RooCategory *>(_vars.find(indexCat.GetName()));

   for (auto const &item : slices) {
      std::unique_ptr<RooDataSet> sliceDataSet;
      RooAbsData* sliceData = item.second;

      // If we are importing a RooDataHist, first convert it to a RooDataSet
      if(sliceData->InheritsFrom(RooDataHist::Class())) {
         sliceDataSet = makeDataSetFromDataHist(static_cast<RooDataHist const &>(*sliceData));
         sliceData = sliceDataSet.get();
      }

      // Define state labels in index category (both in provided indexCat and in internal copy in dataset)
      if (!indexCat.hasLabel(item.first)) {
         indexCat.defineType(item.first);
         coutI(InputArguments) << "RooDataSet::ctor(" << GetName() << ") defining state \"" << item.first
                               << "\" in index category " << indexCat.GetName() << std::endl;
      }
      if (!indexCatInData.hasLabel(item.first)) {
         indexCatInData.defineType(item.first);
      }
      indexCatInData.setLabel(item.first.c_str());
      std::unique_ptr<RooFormulaVar> cutVarTmp;
      if (cutSpec) {
         cutVarTmp = std::make_unique<RooFormulaVar>(cutSpec, cutSpec, *sliceData->get(), /*checkVariables=*/false);
         cutVar = cutVarTmp.get();
      }
      _dstore->loadValues(sliceData->store(), cutVar, rangeName);
   }
}
