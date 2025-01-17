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
\file RooSimultaneous.cxx
\class RooSimultaneous
\ingroup Roofitcore

Facilitates simultaneous fitting of multiple PDFs to subsets of a given dataset.
The class takes an index category, which is used as a selector
for PDFs, and a list of PDFs, each associated
with a state of the index category. RooSimultaneous always returns
the value of the PDF that is associated with the current value
of the index category.

Extended likelihood fitting is supported if all components support
extended likelihood mode. The expected number of events by a RooSimultaneous
is that of the component p.d.f. selected by the index category.

The index category can be accessed using indexCategory().

###Generating events
When generating events from a RooSimultaneous, the index category has to be added to
the dataset. Further, the PDF needs to know the relative probabilities of each category, i.e.,
how many events are in which category. This can be achieved in two ways:
- Generating with proto data that have category entries: An event from the same category as
in the proto data is created for each event in the proto data.
See RooAbsPdf::generate(const RooArgSet&,const RooDataSet&,Int_t,bool,bool,bool) const.
- No proto data: A category is chosen randomly.
\note This requires that the PDFs building the simultaneous are extended. In this way,
the relative probability of each category can be calculated from the number of events
in each category.
**/

#include "RooSimultaneous.h"

#include "Roo1DTable.h"
#include "RooAbsCategoryLValue.h"
#include "RooAbsData.h"
#include "RooAddPdf.h"
#include "RooArgSet.h"
#include "RooBinSamplingPdf.h"
#include "RooCategory.h"
#include "RooCmdConfig.h"
#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include "RooNameReg.h"
#include "RooPlot.h"
#include "RooRandom.h"
#include "RooRealVar.h"
#include "RooSimGenContext.h"
#include "RooSimSplitGenContext.h"
#include "RooSuperCategory.h"

#include "RooFitImplHelpers.h"

#include <ROOT/StringUtils.hxx>

#include <iostream>

namespace {

std::map<std::string, RooAbsPdf *> createPdfMap(const RooArgList &inPdfList, RooAbsCategoryLValue &inIndexCat)
{
   std::map<std::string, RooAbsPdf *> pdfMap;
   auto indexCatIt = inIndexCat.begin();
   for (unsigned int i = 0; i < inPdfList.size(); ++i) {
      auto pdf = static_cast<RooAbsPdf *>(&inPdfList[i]);
      const auto &nameIdx = (*indexCatIt++);
      pdfMap[nameIdx.first] = pdf;
   }
   return pdfMap;
}

} // namespace

RooSimultaneous::InitializationOutput::~InitializationOutput() = default;

void RooSimultaneous::InitializationOutput::addPdf(const RooAbsPdf &pdf, std::string const &catLabel)
{
   finalPdfs.push_back(&pdf);
   finalCatLabels.emplace_back(catLabel);
}

using std::string, std::endl;

ClassImp(RooSimultaneous);



////////////////////////////////////////////////////////////////////////////////
/// Constructor with index category. PDFs associated with indexCat
/// states can be added after construction with the addPdf() function.
///
/// RooSimultaneous can function without having a PDF associated
/// with every single state. The normalization in such cases is taken
/// from the number of registered PDFs, but getVal() will assert if
/// when called for an unregistered index state.

RooSimultaneous::RooSimultaneous(const char *name, const char *title,
             RooAbsCategoryLValue& inIndexCat) :
  RooSimultaneous{name, title, std::map<std::string, RooAbsPdf*>{}, inIndexCat}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor from index category and full list of PDFs.
/// In this constructor form, a PDF must be supplied for each indexCat state
/// to avoid ambiguities. The PDFs are associated with the states of the
/// index category as they appear when iterating through the category states
/// with RooAbsCategory::begin() and RooAbsCategory::end(). This usually means
/// they are associated by ascending index numbers.
///
/// PDFs may not overlap (i.e. share any variables) with the index category (function)

RooSimultaneous::RooSimultaneous(const char *name, const char *title,
             const RooArgList& inPdfList, RooAbsCategoryLValue& inIndexCat) :
  RooSimultaneous{name, title, createPdfMap(inPdfList, inIndexCat), inIndexCat}
{
  if (inPdfList.size() != inIndexCat.size()) {
    std::stringstream errMsg;
    errMsg << "RooSimultaneous::ctor(" << GetName()
           << " ERROR: Number PDF list entries must match number of index category states, no PDFs added";
    coutE(InputArguments) << errMsg.str() << std::endl;
    throw std::invalid_argument(errMsg.str());
  }
}


////////////////////////////////////////////////////////////////////////////////

RooSimultaneous::RooSimultaneous(const char *name, const char *title, std::map<string, RooAbsPdf *> pdfMap,
                                 RooAbsCategoryLValue &inIndexCat)
   : RooSimultaneous(name, title, std::move(*initialize(name ? name : "", inIndexCat, pdfMap)))
{
}

/// For internal use in RooFit.
RooSimultaneous::RooSimultaneous(const char *name, const char *title,
                                 RooFit::Detail::FlatMap<std::string, RooAbsPdf *> const &pdfMap,
                                 RooAbsCategoryLValue &inIndexCat)
   : RooSimultaneous(name, title, RooFit::Detail::flatMapToStdMap(pdfMap), inIndexCat)
{
}

RooSimultaneous::RooSimultaneous(const char *name, const char *title, RooSimultaneous::InitializationOutput &&initInfo)
   : RooAbsPdf(name, title),
     _plotCoefNormSet("!plotCoefNormSet", "plotCoefNormSet", this, false, false),
     _partIntMgr(this, 10),
     _indexCat("indexCat", "Index category", this, *initInfo.indexCat)
{
   for (std::size_t i = 0; i < initInfo.finalPdfs.size(); ++i) {
      addPdf(*initInfo.finalPdfs[i], initInfo.finalCatLabels[i].c_str());
   }

   // Take ownership of eventual super category
   if (initInfo.superIndex) {
      addOwnedComponents(std::move(initInfo.superIndex));
   }
}

/// \cond ROOFIT_INTERNAL

// This class cannot be locally defined in initialize as it cannot be
// used as a template argument in that case
namespace RooSimultaneousAux {
  struct CompInfo {
    RooAbsPdf* pdf ;
    RooSimultaneous* simPdf ;
    const RooAbsCategoryLValue* subIndex ;
    std::unique_ptr<RooArgSet> subIndexComps;
  } ;
}

/// \endcond

std::unique_ptr<RooSimultaneous::InitializationOutput>
RooSimultaneous::initialize(std::string const& name, RooAbsCategoryLValue &inIndexCat,
                            std::map<std::string, RooAbsPdf *> const& pdfMap)

{
  auto out = std::make_unique<RooSimultaneous::InitializationOutput>();
  out->indexCat = &inIndexCat;

  // First see if there are any RooSimultaneous input components
  bool simComps(false) ;
  for (auto const& item : pdfMap) {
    if (dynamic_cast<RooSimultaneous*>(item.second)) {
      simComps = true ;
      break ;
    }
  }

  // If there are no simultaneous component p.d.f. do simple processing through addPdf()
  if (!simComps) {
    for (auto const& item : pdfMap) {
      out->addPdf(*item.second,item.first);
    }
    return out;
  }

  std::string msgPrefix = "RooSimultaneous::initialize(" + name + ") ";

  // Issue info message that we are about to do some rearranging
  oocoutI(nullptr, InputArguments) << msgPrefix << "INFO: one or more input component of simultaneous p.d.f.s are"
         << " simultaneous p.d.f.s themselves, rewriting composite expressions as one-level simultaneous p.d.f. in terms of"
         << " final constituents and extended index category" << std::endl;


  RooArgSet allAuxCats ;
  std::map<string,RooSimultaneousAux::CompInfo> compMap ;
  for (auto const& item : pdfMap) {
    RooSimultaneousAux::CompInfo ci ;
    ci.pdf = item.second ;
    RooSimultaneous* simComp = dynamic_cast<RooSimultaneous*>(item.second) ;
    if (simComp) {
      ci.simPdf = simComp ;
      ci.subIndex = &simComp->indexCat() ;
      ci.subIndexComps = simComp->indexCat().isFundamental()
          ? std::make_unique<RooArgSet>(simComp->indexCat())
          : std::unique_ptr<RooArgSet>(simComp->indexCat().getVariables());
      allAuxCats.add(*ci.subIndexComps,true) ;
    } else {
      ci.simPdf = nullptr;
      ci.subIndex = nullptr;
    }
    compMap[item.first] = std::move(ci);
  }

  // Construct the 'superIndex' from the nominal index category and all auxiliary components
  RooArgSet allCats(inIndexCat) ;
  allCats.add(allAuxCats) ;
  std::string siname = name + "_index";
  out->superIndex = std::make_unique<RooSuperCategory>(siname.c_str(),siname.c_str(),allCats) ;
  auto *superIndex = out->superIndex.get();
  out->indexCat = superIndex;

  // Now process each of original pdf/state map entries
  for (auto const& citem : compMap) {

    RooArgSet repliCats(allAuxCats) ;
    if (citem.second.subIndexComps) {
      repliCats.remove(*citem.second.subIndexComps) ;
    }
    inIndexCat.setLabel(citem.first.c_str()) ;

    if (!citem.second.simPdf) {

      // Entry is a plain p.d.f. assign it to every state permutation of the repliCats set
      RooSuperCategory repliSuperCat("tmp","tmp",repliCats) ;

      // Iterator over all states of repliSuperCat
      for (const auto& nameIdx : repliSuperCat) {
        // Set value
        repliSuperCat.setLabel(nameIdx.first) ;
        // Retrieve corresponding label of superIndex
        string superLabel = superIndex->getCurrentLabel() ;
        out->addPdf(*citem.second.pdf,superLabel);
        oocxcoutD(static_cast<RooAbsArg*>(nullptr), InputArguments) << msgPrefix
                << "assigning pdf " << citem.second.pdf->GetName() << " to super label " << superLabel << std::endl ;
      }
    } else {

      // Entry is a simultaneous p.d.f

      if (repliCats.empty()) {

        // Case 1 -- No replication of components of RooSim component are required

        for (const auto& type : *citem.second.subIndex) {
          const_cast<RooAbsCategoryLValue*>(citem.second.subIndex)->setLabel(type.first.c_str());
          string superLabel = superIndex->getCurrentLabel() ;
          RooAbsPdf* compPdf = citem.second.simPdf->getPdf(type.first);
          if (compPdf) {
            out->addPdf(*compPdf,superLabel);
            oocxcoutD(static_cast<RooAbsArg*>(nullptr), InputArguments) << msgPrefix
                    << "assigning pdf " << compPdf->GetName() << "(member of " << citem.second.pdf->GetName()
                    << ") to super label " << superLabel << std::endl ;
          } else {
            oocoutW(nullptr, InputArguments) << msgPrefix << "WARNING: No p.d.f. associated with label "
                << type.second << " for component RooSimultaneous p.d.f " << citem.second.pdf->GetName()
                << "which is associated with master index label " << citem.first << std::endl ;
          }
        }

      } else {

        // Case 2 -- Replication of components of RooSim component are required

        // Make replication supercat
        RooSuperCategory repliSuperCat("tmp","tmp",repliCats) ;

        for (const auto& stype : *citem.second.subIndex) {
          const_cast<RooAbsCategoryLValue*>(citem.second.subIndex)->setLabel(stype.first.c_str());

          for (const auto& nameIdx : repliSuperCat) {
            repliSuperCat.setLabel(nameIdx.first) ;
            const string superLabel = superIndex->getCurrentLabel() ;
            RooAbsPdf* compPdf = citem.second.simPdf->getPdf(stype.first);
            if (compPdf) {
              out->addPdf(*compPdf,superLabel);
              oocxcoutD(static_cast<RooAbsArg*>(nullptr), InputArguments) << msgPrefix
                      << "assigning pdf " << compPdf->GetName() << "(member of " << citem.second.pdf->GetName()
                      << ") to super label " << superLabel << std::endl ;
            } else {
              oocoutW(nullptr, InputArguments) << msgPrefix << "WARNING: No p.d.f. associated with label "
                  << stype.second << " for component RooSimultaneous p.d.f " << citem.second.pdf->GetName()
                  << "which is associated with master index label " << citem.first << std::endl ;
            }
          }
        }
      }
    }
  }

  return out;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooSimultaneous::RooSimultaneous(const RooSimultaneous& other, const char* name) :
  RooAbsPdf(other,name),
  _plotCoefNormSet("!plotCoefNormSet",this,other._plotCoefNormSet),
  _plotCoefNormRange(other._plotCoefNormRange),
  _partIntMgr(other._partIntMgr,this),
  _indexCat("indexCat",this,other._indexCat),
  _numPdf(other._numPdf)
{
  // Copy proxy list
  for(auto* proxy : static_range_cast<RooRealProxy*>(other._pdfProxyList)) {
    _pdfProxyList.Add(new RooRealProxy(proxy->GetName(),this,*proxy)) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooSimultaneous::~RooSimultaneous()
{
  _pdfProxyList.Delete() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the p.d.f associated with the given index category name

RooAbsPdf* RooSimultaneous::getPdf(RooStringView catName) const
{
  RooRealProxy* proxy = static_cast<RooRealProxy*>(_pdfProxyList.FindObject(catName.c_str()));
  return proxy ? static_cast<RooAbsPdf*>(proxy->absArg()) : nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// Associate given PDF with index category state label 'catLabel'.
/// The name state must be already defined in the index category.
///
/// RooSimultaneous can function without having a PDF associated
/// with every single state. The normalization in such cases is taken
/// from the number of registered PDFs, but getVal() will fail if
/// called for an unregistered index state.
///
/// PDFs may not overlap (i.e. share any variables) with the index category (function).
/// \param[in] pdf PDF to be added.
/// \param[in] catLabel Name of the category state to be associated to the PDF.
/// \return `true` in case of failure.

bool RooSimultaneous::addPdf(const RooAbsPdf& pdf, const char* catLabel)
{
  // PDFs cannot overlap with the index category
  if (pdf.dependsOn(_indexCat.arg())) {
    coutE(InputArguments) << "RooSimultaneous::addPdf(" << GetName() << "): PDF '" << pdf.GetName()
           << "' overlaps with index category '" << _indexCat.arg().GetName() << "'."<< std::endl ;
    return true ;
  }

  // Each index state can only have one PDF associated with it
  if (_pdfProxyList.FindObject(catLabel)) {
    coutE(InputArguments) << "RooSimultaneous::addPdf(" << GetName() << "): index state '"
           << catLabel << "' has already an associated PDF." << std::endl ;
    return true ;
  }

  const RooSimultaneous* simPdf = dynamic_cast<const RooSimultaneous*>(&pdf) ;
  if (simPdf) {

    coutE(InputArguments) << "RooSimultaneous::addPdf(" << GetName()
           << ") ERROR: you cannot add a RooSimultaneous component to a RooSimultaneous using addPdf()."
           << " Use the constructor with RooArgList if input p.d.f.s or the map<string,RooAbsPdf&> instead." << std::endl ;
    return true ;

  } else {

    // Create a proxy named after the associated index state
    TObject* proxy = new RooRealProxy(catLabel,catLabel,this,const_cast<RooAbsPdf&>(pdf));
    _pdfProxyList.Add(proxy) ;
    _numPdf += 1 ;
  }

  return false ;
}

////////////////////////////////////////////////////////////////////////////////
/// Examine the pdf components and check if one of them can be extended or must be extended.
/// It is enough to have one component that can be extended or must be extended to return the flag in
/// the total simultaneous pdf.

RooAbsPdf::ExtendMode RooSimultaneous::extendMode() const
{
   bool anyCanExtend = false;

   for (auto *proxy : static_range_cast<RooRealProxy *>(_pdfProxyList)) {
      auto &pdf = static_cast<RooAbsPdf const&>(proxy->arg());
      if (pdf.mustBeExtended())
         return MustBeExtended;
      anyCanExtend |= pdf.canBeExtended();
   }
   return anyCanExtend ? CanBeExtended : CanNotBeExtended;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the current value:
/// the value of the PDF associated with the current index category state

double RooSimultaneous::evaluate() const
{
   // Retrieve the proxy by index name
   RooRealProxy *proxy = static_cast<RooRealProxy *>(_pdfProxyList.FindObject(_indexCat.label()));

   double nEvtTot = 1.0;
   double nEvtCat = 1.0;

   // Calculate relative weighting factor for sim-pdfs of all extendable components
   if (canBeExtended()) {

      nEvtTot = 0;
      nEvtCat = 0;

      for (auto *proxy2 : static_range_cast<RooRealProxy *>(_pdfProxyList)) {
         auto &pdf2 = static_cast<RooAbsPdf const &>(proxy2->arg());
         if(!pdf2.canBeExtended()) {
            // If one of the pdfs can't be expected, reset the normalization
            // factor to one and break out of the loop.
            nEvtTot = 1.0;
            nEvtCat = 1.0;
            break;
         }
         const double nEvt = pdf2.expectedEvents(_normSet);
         nEvtTot += nEvt;
         if (proxy == proxy2) {
            // Matching by proxy by pointer rather than pdfs, because it's
            // possible to have the same pdf used in different states.
            nEvtCat += nEvt;
         }
      }
   }
   double catFrac = nEvtCat / nEvtTot;

   // Return the selected PDF value, normalized by the relative number of
   // expected events if applicable.
   return *proxy * catFrac;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the number of expected events: If the index is in nset,
/// then return the sum of the expected events of all components,
/// otherwise return the number of expected events of the PDF
/// associated with the current index category state

double RooSimultaneous::expectedEvents(const RooArgSet* nset) const
{
  if (nset->contains(_indexCat.arg())) {

    double sum(0) ;

    for(auto * proxy : static_range_cast<RooRealProxy*>(_pdfProxyList)) {
      sum += (static_cast<RooAbsPdf*>(proxy->absArg()))->expectedEvents(nset) ;
    }

    return sum ;

  } else {

    // Retrieve the proxy by index name
    RooRealProxy* proxy = static_cast<RooRealProxy*>(_pdfProxyList.FindObject(_indexCat.label())) ;

    //assert(proxy!=0) ;
    if (proxy==nullptr) return 0 ;

    // Return the selected PDF value, normalized by the number of index states
    return (static_cast<RooAbsPdf*>(proxy->absArg()))->expectedEvents(nset);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Forward determination of analytical integration capabilities to component p.d.f.s
/// A unique code is assigned to the combined integration capabilities of all associated
/// p.d.f.s

Int_t RooSimultaneous::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                      const RooArgSet* normSet, const char* rangeName) const
{
  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Retrieve (or create) the required partial integral list
  Int_t code ;

  // Check if this configuration was created before
  CacheElem* cache = static_cast<CacheElem*>(_partIntMgr.getObj(normSet,&analVars,nullptr,RooNameReg::ptr(rangeName))) ;
  if (cache) {
    code = _partIntMgr.lastIndex() ;
    return code+1 ;
  }
  cache = new CacheElem ;

  // Create the partial integral set for this request
  for(auto * proxy : static_range_cast<RooRealProxy*>(_pdfProxyList)) {
    cache->_partIntList.addOwned(std::unique_ptr<RooAbsReal>{proxy->arg().createIntegral(analVars,normSet,nullptr,rangeName)});
  }

  // Store the partial integral list and return the assigned code ;
  code = _partIntMgr.setObj(normSet,&analVars,cache,RooNameReg::ptr(rangeName)) ;

  return code+1 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return analytical integration defined by given code

double RooSimultaneous::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* /*rangeName*/) const
{
  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Partial integration scenarios, rangeName already encoded in 'code'
  CacheElem* cache = static_cast<CacheElem*>(_partIntMgr.getObjByIndex(code-1)) ;

  RooRealProxy* proxy = static_cast<RooRealProxy*>(_pdfProxyList.FindObject(_indexCat.label())) ;
  Int_t idx = _pdfProxyList.IndexOf(proxy) ;
  return (static_cast<RooAbsReal*>(cache->_partIntList.at(idx)))->getVal(normSet) ;
}






////////////////////////////////////////////////////////////////////////////////
/// Back-end for plotOn() implementation on RooSimultaneous which
/// needs special handling because a RooSimultaneous PDF cannot
/// project out its index category via integration. plotOn() will
/// abort if this is requested without providing a projection dataset.

RooPlot* RooSimultaneous::plotOn(RooPlot *frame, RooLinkedList& cmdList) const
{
  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Extract projection configuration from command list
  RooCmdConfig pc("RooSimultaneous::plotOn(" + std::string(GetName()) + ")");
  pc.defineString("sliceCatState","SliceCat",0,"",true) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,RooAbsPdf::Relative) ;
  pc.defineObject("sliceCatList","SliceCat",0,nullptr,true) ;
  // This dummy is needed for plotOn to recognize the "SliceCatMany" command.
  // It is not used directly, but the "SliceCat" commands are nested in it.
  // Removing this dummy definition results in "ERROR: unrecognized command: SliceCatMany".
  pc.defineObject("dummy1","SliceCatMany",0) ;
  pc.defineSet("projSet","Project",0) ;
  pc.defineSet("sliceSet","SliceVars",0) ;
  pc.defineSet("projDataSet","ProjData",0) ;
  pc.defineObject("projData","ProjData",1) ;
  pc.defineMutex("Project","SliceVars") ;
  pc.allowUndefined() ; // there may be commands we don't handle here

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  const RooAbsData* projData = static_cast<const RooAbsData*>(pc.getObject("projData")) ;
  const RooArgSet* projDataSet = pc.getSet("projDataSet");
  const RooArgSet* sliceSetTmp = pc.getSet("sliceSet") ;
  std::unique_ptr<RooArgSet> sliceSet( sliceSetTmp ? (static_cast<RooArgSet*>(sliceSetTmp->Clone())) : nullptr );
  const RooArgSet* projSet = pc.getSet("projSet") ;
  double scaleFactor = pc.getDouble("scaleFactor") ;
  ScaleType stype = (ScaleType) pc.getInt("scaleType") ;


  // Look for category slice arguments and add them to the master slice list if found
  const char* sliceCatState = pc.getString("sliceCatState",nullptr,true) ;
  const RooLinkedList& sliceCatList = pc.getObjectList("sliceCatList") ;
  if (sliceCatState) {

    // Make the master slice set if it doesnt exist
    if (!sliceSet) {
      sliceSet = std::make_unique<RooArgSet>();
    }

    // Prepare comma separated label list for parsing
    auto catTokens = ROOT::Split(sliceCatState, ",");

    // Loop over all categories provided by (multiple) Slice() arguments
    unsigned int tokenIndex = 0;
    for(auto * scat : static_range_cast<RooCategory*>(sliceCatList)) {
      const char* slabel = tokenIndex >= catTokens.size() ? nullptr : catTokens[tokenIndex++].c_str();

      if (slabel) {
        // Set the slice position to the value indicated by slabel
        scat->setLabel(slabel) ;
        // Add the slice category to the master slice set
        sliceSet->add(*scat,false) ;
      }
    }
  }

  // Check if we have a projection dataset
  if (!projData) {
    coutE(InputArguments) << "RooSimultaneous::plotOn(" << GetName() << ") ERROR: must have a projection dataset for index category" << std::endl ;
    return frame ;
  }

  // Make list of variables to be projected
  RooArgSet projectedVars ;
  if (sliceSet) {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;

    // Take out the sliced variables
    for (const auto sliceArg : *sliceSet) {
      RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
      if (arg) {
        projectedVars.remove(*arg) ;
      } else {
        coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") slice variable "
            << sliceArg->GetName() << " was not projected anyway" << std::endl ;
      }
    }
  } else if (projSet) {
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,false) ;
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;
  }

  bool projIndex(false) ;

  if (!_indexCat.arg().isDerived()) {
    // *** Error checking for a fundamental index category ***
    //cout << "RooSim::plotOn: index is fundamental" << std::endl ;

    // Check that the provided projection dataset contains our index variable
    if (!projData->get()->find(_indexCat.arg().GetName())) {
      coutE(Plotting) << "RooSimultaneous::plotOn(" << GetName() << ") ERROR: Projection over index category "
            << "requested, but projection data set doesn't contain index category" << std::endl ;
      return frame ;
    }

    if (projectedVars.find(_indexCat.arg().GetName())) {
      projIndex=true ;
    }

  } else {
    // *** Error checking for a composite index category ***

    // Determine if any servers of the index category are in the projectedVars
    RooArgSet projIdxServers ;
    bool anyServers(false) ;
    for (const auto server : flattenedCatList()) {
      if (projectedVars.find(server->GetName())) {
        anyServers=true ;
        projIdxServers.add(*server) ;
      }
    }

    // Check that the projection dataset contains all the
    // index category components we're projecting over

    // Determine if all projected servers of the index category are in the projection dataset
    bool allServers(true) ;
    std::string missing;
    for (const auto server : projIdxServers) {
      if (!projData->get()->find(server->GetName())) {
        allServers=false ;
        missing = server->GetName();
      }
    }

    if (!allServers) {
      coutE(Plotting) << "RooSimultaneous::plotOn(" << GetName()
          << ") ERROR: Projection dataset doesn't contain complete set of index categories to do projection."
          << "\n\tcategory " << missing << " is missing." << std::endl ;
      return frame ;
    }

    if (anyServers) {
      projIndex = true ;
    }
  }

  // Calculate relative weight fractions of components
  std::unique_ptr<Roo1DTable> wTable( projData->table(_indexCat.arg()) );

  // Clone the index category to be able to cycle through the category states for plotting without
  // affecting the category state of our instance
  RooArgSet idxCloneSet;
  RooArgSet(*_indexCat).snapshot(idxCloneSet, true);
  auto idxCatClone = static_cast<RooAbsCategoryLValue*>(idxCloneSet.find(_indexCat->GetName()) );
  assert(idxCatClone);

  // Make list of category columns to exclude from projection data
  std::unique_ptr<RooArgSet> idxCompSliceSet( idxCatClone->getObservables(frame->getNormVars()) );

  // If we don't project over the index, just do the regular plotOn
  if (!projIndex) {

    coutI(Plotting) << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName()
          << " represents a slice in the index category ("  << _indexCat.arg().GetName() << ")" << std::endl ;

    // Reduce projData: take out fitCat (component) columns and entries that don't match selected slice
    // Construct cut string to only select projection data event that match the current slice

    // Make cut string to exclude rows from projection data
    TString cutString ;
    bool first(true) ;
    for (const auto arg : *idxCompSliceSet) {
      auto idxComp = static_cast<RooCategory*>(arg);
      RooAbsArg* slicedComponent = nullptr;
      if (sliceSet && (slicedComponent = sliceSet->find(*idxComp)) != nullptr) {
        auto theCat = static_cast<const RooAbsCategory*>(slicedComponent);
        idxComp->setIndex(theCat->getCurrentIndex(), false);
      }

      if (!first) {
        cutString.Append("&&") ;
      } else {
        first=false ;
      }
      cutString.Append(Form("%s==%d",idxComp->GetName(),idxComp->getCurrentIndex())) ;
    }

    // Make temporary projData without RooSim index category components
    RooArgSet projDataVars(*projData->get()) ;
    projDataVars.remove(*idxCompSliceSet,true,true) ;

    std::unique_ptr<RooAbsData> projDataTmp( const_cast<RooAbsData*>(projData)->reduce(projDataVars,cutString) );

    // Override normalization and projection dataset
    RooCmdArg tmp1 = RooFit::Normalization(scaleFactor*wTable->getFrac(idxCatClone->getCurrentLabel()),stype) ;
    RooCmdArg tmp2 = RooFit::ProjWData(*projDataSet,*projDataTmp) ;

    // WVE -- do not adjust normalization for asymmetry plots
    RooLinkedList cmdList2(cmdList) ;
    if (!cmdList.find("Asymmetry")) {
      cmdList2.Add(&tmp1) ;
    }
    cmdList2.Add(&tmp2) ;

    // Plot single component
    RooPlot* retFrame = getPdf(idxCatClone->getCurrentLabel())->plotOn(frame,cmdList2);
    return retFrame ;
  }

  // If we project over the index, plot using a temporary RooAddPdf
  // using the weights from the data as coefficients

  // Build the list of indexCat components that are sliced
  idxCompSliceSet->remove(projectedVars,true,true) ;

  // Make a new expression that is the weighted sum of requested components
  RooArgList pdfCompList ;
  RooArgList wgtCompList ;
//RooAbsPdf* pdf ;
  double sumWeight(0) ;
  for(auto * proxy : static_range_cast<RooRealProxy*>(_pdfProxyList)) {

    idxCatClone->setLabel(proxy->name()) ;

    // Determine if this component is the current slice (if we slice)
    bool skip(false) ;
    for (const auto idxSliceCompArg : *idxCompSliceSet) {
      const auto idxSliceComp = static_cast<RooAbsCategory*>(idxSliceCompArg);
      RooAbsCategory* idxComp = static_cast<RooAbsCategory*>(idxCloneSet.find(idxSliceComp->GetName())) ;
      if (idxComp->getCurrentIndex()!=idxSliceComp->getCurrentIndex()) {
        skip=true ;
        break ;
      }
    }
    if (skip) continue ;

    // Instantiate a RRV holding this pdfs weight fraction
    wgtCompList.addOwned(std::make_unique<RooRealVar>(proxy->name(),"coef",wTable->getFrac(proxy->name())));
    sumWeight += wTable->getFrac(proxy->name()) ;

    // Add the PDF to list list
    pdfCompList.add(proxy->arg()) ;
  }

  TString plotVarName(GetName()) ;
  RooAddPdf plotVar{plotVarName,"weighted sum of RS components",pdfCompList,wgtCompList};

  // Fix appropriate coefficient normalization in plot function
  if (!_plotCoefNormSet.empty()) {
    plotVar.fixAddCoefNormalization(_plotCoefNormSet) ;
  }

  std::unique_ptr<RooAbsData> projDataTmp;
  RooArgSet projSetTmp ;
  if (projData) {

    // Construct cut string to only select projection data event that match the current slice
    TString cutString ;
    if (!idxCompSliceSet->empty()) {
      bool first(true) ;
      for (const auto idxSliceCompArg : *idxCompSliceSet) {
        const auto idxSliceComp = static_cast<RooAbsCategory*>(idxSliceCompArg);
        if (!first) {
          cutString.Append("&&") ;
        } else {
          first=false ;
        }
        cutString.Append(Form("%s==%d",idxSliceComp->GetName(),idxSliceComp->getCurrentIndex())) ;
      }
    }

    // Make temporary projData without RooSim index category components
    RooArgSet projDataVars(*projData->get()) ;
    RooArgSet idxCatServers;
    _indexCat.arg().getObservables(frame->getNormVars(), idxCatServers) ;

    projDataVars.remove(idxCatServers,true,true) ;

    if (!idxCompSliceSet->empty()) {
      projDataTmp = std::unique_ptr<RooAbsData>{const_cast<RooAbsData*>(projData)->reduce(projDataVars,cutString)};
    } else {
      projDataTmp = std::unique_ptr<RooAbsData>{const_cast<RooAbsData*>(projData)->reduce(projDataVars)};
    }



    if (projSet) {
      projSetTmp.add(*projSet) ;
      projSetTmp.remove(idxCatServers,true,true);
    }
  }


  if (_indexCat.arg().isDerived() && !idxCompSliceSet->empty()) {
    coutI(Plotting) << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName()
          << " represents a slice in index category components " << *idxCompSliceSet << std::endl ;

    RooArgSet idxCompProjSet;
    _indexCat.arg().getObservables(frame->getNormVars(), idxCompProjSet) ;
    idxCompProjSet.remove(*idxCompSliceSet,true,true) ;
    if (!idxCompProjSet.empty()) {
      coutI(Plotting) << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName()
            << " averages with data index category components " << idxCompProjSet << std::endl ;
    }
  } else {
    coutI(Plotting) << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName()
          << " averages with data index category (" << _indexCat.arg().GetName() << ")" << std::endl ;
  }


  // Override normalization and projection dataset
  RooLinkedList cmdList2(cmdList) ;

  RooCmdArg tmp1 = RooFit::Normalization(scaleFactor*sumWeight,stype) ;
  RooCmdArg tmp2 = RooFit::ProjWData(*projDataSet,*projDataTmp) ;
  // WVE -- do not adjust normalization for asymmetry plots
  if (!cmdList.find("Asymmetry")) {
    cmdList2.Add(&tmp1) ;
  }
  cmdList2.Add(&tmp2) ;

  RooPlot* frame2 ;
  if (!projSetTmp.empty()) {
    // Plot temporary function
    RooCmdArg tmp3 = RooFit::Project(projSetTmp) ;
    cmdList2.Add(&tmp3) ;
    frame2 = plotVar.plotOn(frame,cmdList2) ;
  } else {
    // Plot temporary function
    frame2 = plotVar.plotOn(frame,cmdList2) ;
  }

  return frame2 ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of observables
/// for interpretation of fraction coefficients. Needed here because a RooSimultaneous
/// works like a RooAddPdf when plotted

void RooSimultaneous::selectNormalization(const RooArgSet* normSet, bool /*force*/)
{
  _plotCoefNormSet.removeAll() ;
  if (normSet) _plotCoefNormSet.add(*normSet) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of range
/// for interpretation of fraction coefficients. Needed here because a RooSimultaneous
/// works like a RooAddPdf when plotted

void RooSimultaneous::selectNormalizationRange(const char* normRange2, bool /*force*/)
{
  _plotCoefNormRange = RooNameReg::ptr(normRange2) ;
}




////////////////////////////////////////////////////////////////////////////////

RooAbsGenContext* RooSimultaneous::autoGenContext(const RooArgSet &vars, const RooDataSet* prototype,
                    const RooArgSet* auxProto, bool verbose, bool autoBinned, const char* binnedTag) const
{
  const char* idxCatName = _indexCat.arg().GetName() ;

  if (vars.find(idxCatName) && prototype==nullptr
      && (auxProto==nullptr || auxProto->empty())
      && (autoBinned || (binnedTag && strlen(binnedTag)))) {

    // Return special generator config that can also do binned generation for selected states
    return new RooSimSplitGenContext(*this,vars,verbose,autoBinned,binnedTag) ;

  } else {

    // Return regular generator config ;
    return genContext(vars,prototype,auxProto,verbose) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return specialized generator context for simultaneous p.d.f.s

RooAbsGenContext* RooSimultaneous::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                     const RooArgSet* auxProto, bool verbose) const
{
  RooArgSet allVars{vars};
  if(prototype) allVars.add(*prototype->get());

  RooArgSet catsAmongAllVars;
  allVars.selectCommon(flattenedCatList(), catsAmongAllVars);

  // Not generating index cat: return context for pdf associated with present index state
  if(catsAmongAllVars.empty()) {
    auto* proxy = static_cast<RooRealProxy*>(_pdfProxyList.FindObject(_indexCat->getCurrentLabel()));
    if (!proxy) {
      coutE(InputArguments) << "RooSimultaneous::genContext(" << GetName()
             << ") ERROR: no PDF associated with current state ("
             << _indexCat.arg().GetName() << "=" << _indexCat.arg().getCurrentLabel() << ")" << std::endl ;
      return nullptr;
    }
    return static_cast<RooAbsPdf*>(proxy->absArg())->genContext(vars,prototype,auxProto,verbose) ;
  }

  RooArgSet catsAmongProtoVars;
  if(prototype) {
    prototype->get()->selectCommon(flattenedCatList(), catsAmongProtoVars);

    if(!catsAmongProtoVars.empty() && catsAmongProtoVars.size() != flattenedCatList().size()) {
      // Abort if we have only part of the servers
      coutE(Plotting) << "RooSimultaneous::genContext: ERROR: prototype must include either all "
            << " components of the RooSimultaneous index category or none " << std::endl;
      return nullptr;
    }
  }

  return new RooSimGenContext(*this,vars,prototype,auxProto,verbose) ;
}




////////////////////////////////////////////////////////////////////////////////

RooDataHist* RooSimultaneous::fillDataHist(RooDataHist *hist,
                                           const RooArgSet* nset,
                                           double scaleFactor,
                                           bool correctForBinVolume,
                                           bool showProgress) const
{
  if (RooAbsReal::fillDataHist (hist, nset, scaleFactor,
                                correctForBinVolume, showProgress) == nullptr)
    return nullptr;

  const double sum = hist->sumEntries();
  if (sum != 0) {
    for (int i=0 ; i<hist->numEntries() ; i++) {
      hist->set(i, hist->weight(i) / sum, 0.);
    }
  }

  return hist;
}




////////////////////////////////////////////////////////////////////////////////
/// Special generator interface for generation of 'global observables' -- for RooStats tools

RooFit::OwningPtr<RooDataSet> RooSimultaneous::generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents)
{
  // Make set with clone of variables (placeholder for output)
  RooArgSet globClone;
  whatVars.snapshot(globClone);

  auto data = std::make_unique<RooDataSet>("gensimglobal","gensimglobal",whatVars);

  for (Int_t i=0 ; i<nEvents ; i++) {
    for (const auto& nameIdx : indexCat()) {

      // Get pdf associated with state from simpdf
      RooAbsPdf* pdftmp = getPdf(nameIdx.first);

      RooArgSet globtmp;
      pdftmp->getObservables(&whatVars, globtmp) ;

      // If there are any, generate only global variables defined by the pdf
      // associated with this state and transfer values to output placeholder.
      if (!globtmp.empty()) {
        globClone.assign(*std::unique_ptr<RooDataSet>{pdftmp->generate(globtmp,1)}->get(0)) ;
      }
    }
    data->add(globClone) ;
  }

  return RooFit::makeOwningPtr(std::move(data));
}


/// Wraps the components of this RooSimultaneous in RooBinSamplingPdfs.
/// \param[in] data The dataset to be used in the eventual fit, used to figure
///            out the observables and whether the dataset is binned.
/// \param[in] precision Precision argument for all created RooBinSamplingPdfs.
void RooSimultaneous::wrapPdfsInBinSamplingPdfs(RooAbsData const &data, double precision) {

  if (precision < 0.) return;

  RooArgSet newSamplingPdfs;

  for (auto const &item : this->indexCat()) {

    auto const &catName = item.first;
    auto &pdf = *this->getPdf(catName);

    if (auto newSamplingPdf = RooBinSamplingPdf::create(pdf, data, precision)) {
      // Set the "ORIGNAME" attribute the indicate to
      // RooAbsArg::redirectServers() which pdf should be replaced by this
      // RooBinSamplingPdf in the RooSimultaneous.
      newSamplingPdf->setAttribute(
          (std::string("ORIGNAME:") + pdf.GetName()).c_str());
      newSamplingPdfs.addOwned(std::move(newSamplingPdf));
    }
  }

  this->redirectServers(newSamplingPdfs, false, true);
  this->addOwnedComponents(std::move(newSamplingPdfs));
}


/// Wraps the components of this RooSimultaneous in RooBinSamplingPdfs, with a
/// different precision parameter for each component.
/// \param[in] data The dataset to be used in the eventual fit, used to figure
///            out the observables and whether the dataset is binned.
/// \param[in] precisions The map that gives the precision argument for each
///            component in the RooSimultaneous. The keys are the pdf names. If
///            there is no value for a given component, it will not use the bin
///            integration. Otherwise, the value has the same meaning than in
///            the IntegrateBins() command argument for RooAbsPdf::fitTo().
/// \param[in] useCategoryNames If this flag is set, the category names will be
///            used to look up the precision in the precisions map instead of
///            the pdf names.
void RooSimultaneous::wrapPdfsInBinSamplingPdfs(RooAbsData const &data,
                                                std::map<std::string, double> const& precisions,
                                                bool useCategoryNames /*=false*/) {

  constexpr double defaultPrecision = -1.;

  RooArgSet newSamplingPdfs;

  for (auto const &item : this->indexCat()) {

    auto const &catName = item.first;
    auto &pdf = *this->getPdf(catName);
    std::string pdfName = pdf.GetName();

    auto found = precisions.find(useCategoryNames ? catName : pdfName);
    const double precision =
        found != precisions.end() ? found->second : defaultPrecision;
    if (precision < 0.)
      continue;

    if (auto newSamplingPdf = RooBinSamplingPdf::create(pdf, data, precision)) {
      // Set the "ORIGNAME" attribute the indicate to
      // RooAbsArg::redirectServers() which pdf should be replaced by this
      // RooBinSamplingPdf in the RooSimultaneous.
      newSamplingPdf->setAttribute(
          (std::string("ORIGNAME:") + pdf.GetName()).c_str());
      newSamplingPdfs.addOwned(std::move(newSamplingPdf));
    }
  }

  this->redirectServers(newSamplingPdfs, false, true);
  this->addOwnedComponents(std::move(newSamplingPdfs));
}

/// Internal utility function to get a list of all category components for this
/// RooSimultaneous. The output contains only the index category if it is a
/// RooCategory, or the list of all category components if it is a
/// RooSuperCategory.
RooArgSet const& RooSimultaneous::flattenedCatList() const
{
   // Note that the index category of a RooSimultaneous can only be of type
   // RooCategory or RooSuperCategory, because these are the only classes that
   // inherit from RooAbsCategoryLValue.
   if (auto superCat = dynamic_cast<RooSuperCategory const*>(&_indexCat.arg())) {
       return superCat->inputCatList();
   }

   if(!_indexCatSet) {
      _indexCatSet = std::make_unique<RooArgSet>(_indexCat.arg());
   }
   return *_indexCatSet;
}

namespace {

void markObs(RooAbsArg *arg, std::string const &prefix, RooArgSet const &normSet)
{
   for (RooAbsArg *server : arg->servers()) {
      if (server->isFundamental() && normSet.find(*server)) {
         markObs(server, prefix, normSet);
         server->setAttribute("__obs__");
      } else if (!server->isFundamental()) {
         markObs(server, prefix, normSet);
      }
   }
}

void prefixArgs(RooAbsArg *arg, std::string const &prefix, RooArgSet const &normSet)
{
   if (!arg->getStringAttribute("__prefix__")) {
      arg->SetName((prefix + arg->GetName()).c_str());
      arg->setStringAttribute("__prefix__", prefix.c_str());
   }
   for (RooAbsArg *server : arg->servers()) {
      if (server->isFundamental() && normSet.find(*server)) {
         prefixArgs(server, prefix, normSet);
      } else if (!server->isFundamental()) {
         prefixArgs(server, prefix, normSet);
      }
   }
}

} // namespace

std::unique_ptr<RooAbsArg>
RooSimultaneous::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   std::unique_ptr<RooSimultaneous> newSimPdf{static_cast<RooSimultaneous *>(this->Clone())};

   const char *rangeName = this->getStringAttribute("RangeName");
   bool splitRange = this->getAttribute("SplitRange");

   RooArgSet newPdfs;
   std::vector<std::string> catNames;

   for (auto *proxy : static_range_cast<RooRealProxy *>(newSimPdf->_pdfProxyList)) {
      catNames.emplace_back(proxy->GetName());
      std::string const &catName = catNames.back();
      const std::string prefix = "_" + catName + "_";

      const std::string origname = proxy->arg().GetName();

      auto pdfClone = RooHelpers::cloneTreeWithSameParameters(static_cast<RooAbsPdf const &>(proxy->arg()), &normSet);

      markObs(pdfClone.get(), prefix, normSet);

      std::unique_ptr<RooArgSet> pdfNormSet{
         std::unique_ptr<RooArgSet>(pdfClone->getVariables())->selectByAttrib("__obs__", true)};

      if (rangeName) {
         pdfClone->setNormRange(RooHelpers::getRangeNameForSimComponent(rangeName, splitRange, catName).c_str());
      }

      RooFit::Detail::CompileContext pdfContext{*pdfNormSet};
      pdfContext.setLikelihoodMode(ctx.likelihoodMode());
      auto *pdfFinal = pdfContext.compile(*pdfClone, *newSimPdf, *pdfNormSet);

      // We can only prefix the observables after everything related the
      // compiling of the compute graph for the normalization set is done. This
      // is because of a subtlety in conditional RooProdPdfs, which stores the
      // normalization sets for the individual pdfs in RooArgSets that are
      // disconnected from the computation graph, so we have no control over
      // them. An alternative would be to use recursive server re-direction,
      // but this has more performance overhead.
      prefixArgs(pdfFinal, prefix, normSet);

      pdfFinal->fixAddCoefNormalization(*pdfNormSet, false);

      pdfClone->SetName((std::string("_") + pdfClone->GetName()).c_str());
      pdfFinal->addOwnedComponents(std::move(pdfClone));

      pdfFinal->setAttribute(("ORIGNAME:" + origname).c_str());
      newPdfs.add(*pdfFinal);

      // We will remove the old pdf server because we will fill the new ones by
      // hand via the creation of new proxies.
      newSimPdf->removeServer(const_cast<RooAbsReal &>(proxy->arg()), true);
   }

   // Replace pdfs with compiled pdfs. Don't use RooAbsArg::redirectServers()
   // here, because it doesn't support replacing two servers with the same name
   // (it can happen in a RooSimultaneous that two pdfs have the same name).

   // First delete old proxies (we have already removed the servers before).
   newSimPdf->_pdfProxyList.Delete();

   // Recreate the _pdfProxyList with the compiled pdfs
   for (std::size_t i = 0; i < newPdfs.size(); ++i) {
      const char *label = catNames[i].c_str();
      newSimPdf->_pdfProxyList.Add(
         new RooRealProxy(label, label, newSimPdf.get(), *static_cast<RooAbsReal *>(newPdfs[i])));
   }

   ctx.compileServers(*newSimPdf, normSet); // to trigger compiling also the index category

   return newSimPdf;
}
