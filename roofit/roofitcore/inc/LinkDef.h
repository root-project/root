#ifdef __CLING__

//Old LinkDef1.h
#pragma link off all globals;
#pragma link off all classes;
#pragma link C++ class Roo1DTable+ ;
#pragma link C++ class RooAbsArg- ;
#pragma link C++ class stack<RooAbsArg*,deque<RooAbsArg*> > ;
#pragma link C++ class RooRefArray- ;
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-4]" source="TList _proxyList" target="_proxyList" \
    code="{ for (TObject * tmpObj : onfile._proxyList) { _proxyList.Add(tmpObj); } }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[5]" source="TRefArray _proxyList" target="_proxyList" \
  code="{ _proxyList.GetSize() ; if (onfile._proxyList.GetSize()>0) { RooAbsArg::addToIoEvoList(newObj, onfile._proxyList); } }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _serverList" target="_serverList" \
  code="{ _serverList = RooSTLRefCountList<RooAbsArg>::convert(onfile._serverList); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientList" target="_clientList" \
  code="{ _clientList = RooSTLRefCountList<RooAbsArg>::convert(onfile._clientList); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientListValue" target="_clientListValue" \
  code="{ _clientListValue = RooSTLRefCountList<RooAbsArg>::convert(onfile._clientListValue); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientListShape" target="_clientListShape" \
  code="{ _clientListShape = RooSTLRefCountList<RooAbsArg>::convert(onfile._clientListShape); }"
#pragma link C++ class RooSTLRefCountList<RooAbsArg>+;
#pragma link C++ class RooStringVar+ ;
#pragma read sourceClass="RooAbsString" targetClass="RooStringVar" version="[1]" source="Int_t _len; char *_value" target="_string" code="{_string.assign(onfile._value, onfile._len);}"
#pragma link C++ class RooAbsBinning- ;
#pragma link C++ class RooAbsCategory+ ;
#pragma read sourceClass="RooAbsCategory" targetClass="RooAbsCategory" version="[1]" \
  include="RooFitLegacy/RooCatTypeLegacy.h" \
  source="TObjArray _types" target="_stateNames,_insertionOrder" \
  code="{for (const auto* obj : onfile._types) { \
           auto catType = dynamic_cast<const RooCatType*>(obj); assert(catType); \
           _stateNames[catType->GetName()] = catType->getVal(); \
           _insertionOrder.push_back(catType->GetName()); \
         }}";
#pragma read sourceClass="RooAbsCategory" targetClass="RooAbsCategory" version="[2]" \
  include="RooFitLegacy/RooCatTypeLegacy.h" \
  source="std::vector<RooCatType*> _types" target="_stateNames,_insertionOrder" \
  code="{for (const auto catType : onfile._types) { _stateNames[catType->GetName()] = catType->getVal();\
                                                    _insertionOrder.push_back(catType->GetName());\
                                                  } }";
#pragma read sourceClass="RooAbsCategory" targetClass="RooAbsCategory" version="[1-2]" include="RooFitLegacy/RooCatTypeLegacy.h" \
  source="RooCatType _value" target="_currentIndex" code="{ _currentIndex = onfile._value.getVal(); }"
#pragma link C++ class RooAbsCategoryLValue+ ;
#pragma link C++ class RooAbsCollection+ ;
#pragma read sourceClass="RooAbsCollection" targetClass="RooAbsCollection" version="[1]" source="" target="_allRRV" code="{ _allRRV=false ; }"
#pragma read sourceClass="RooAbsCollection" targetClass="RooAbsCollection" version="[2]"\
    source="RooLinkedList _list" target="_list" code="{ \
    for (RooAbsArg * theArg : static_range_cast<RooAbsArg*>(onfile._list)) {_list.push_back(theArg);} }"
#pragma link C++ class RooAbsData- ;
#pragma link C++ class RooAbsFunc+ ;
#pragma link C++ class RooAbsGenContext+ ;
#pragma link C++ class RooAbsHiddenReal+ ;
#pragma link C++ class RooAbsLValue+ ;
#pragma link C++ class RooAbsMCStudyModule+ ;
#pragma link C++ class RooAbsPdf+ ;
#pragma link C++ class RooAbsPdf::GenSpec+ ;
#pragma link C++ class RooAbsProxy+ ;
#pragma link C++ class RooAbsReal+ ;
#pragma link C++ class RooAbsRealLValue+ ;
#pragma link C++ class RooAddGenContext+ ;
#pragma link C++ class RooAddition+ ;
#pragma link C++ class RooAddModel+ ;
#pragma link C++ class RooAICRegistry+ ;
#pragma link C++ class RooArgList+ ;
#pragma link C++ class RooArgProxy+ ;
#pragma link C++ class RooArgSet+ ;
#pragma link C++ class RooBinnedGenContext+ ;
#pragma link C++ class RooBinning-;
#pragma link C++ class RooBrentRootFinder+ ;
#pragma link C++ class RooCategory- ;
#pragma link C++ class RooCategorySharedProperties+ ;
#pragma link C++ class RooCatType+ ;
#pragma link C++ class RooCmdArg+ ;
#pragma link C++ class RooCmdConfig+ ;
#pragma link C++ class RooConstVar+ ;
#pragma read sourceClass="RooConstVar" targetClass="RooConstVar" version="[1]" source="double _value" target="" code="{ newObj->changeVal(onfile._value); }"
#pragma read sourceClass="RooConstraintSum" targetClass="RooConstraintSum" version="[3]"   \
             source="RooSetProxy _paramSet" target="_paramSet" code="{                     \
                 for(RooAbsArg * arg : onfile._paramSet) {                                 \
                    _paramSet.add(*arg);                                                   \
                 }                                                                         \
             }"
#pragma link C++ class RooConvCoefVar+ ;
#pragma link C++ class RooConvGenContext+ ;
#pragma link C++ class RooCurve+ ;
#pragma link C++ class RooDataHist- ;
#pragma link C++ class RooDataProjBinding+ ;
#pragma link C++ class RooDataSet- ;
#pragma link C++ class RooDirItem+ ;
#pragma link C++ class RooDLLSignificanceMCSModule+ ;
#pragma link C++ class RooAbsAnaConvPdf+ ;
#pragma link C++ class RooAddPdf+ ;
#pragma link C++ class RooEfficiency+ ;
#pragma link C++ class RooEffProd+ ;
#pragma link C++ class RooExtendPdf+ ;
#pragma link off class RooErrorHandler+ ;
#pragma link C++ class RooWrapperPdf+;


// Old LinkDef2.h

// "namespace RooFit" is in roofit/roofit/inc/Linkdef1.h
// should not be in the dictionary for two different libraries
// #pragma link C++ namespace RooFit ;
#pragma link C++ namespace RooFitShortHand ;
#pragma link C++ class RooDouble+ ;
#pragma link C++ class RooEffGenContext+ ;
#pragma link C++ class RooEllipse+ ;
#pragma link C++ class RooErrorVar+ ;
#pragma link C++ class RooFitResult- ;
#pragma link C++ class RooFIter+ ;
#pragma link C++ class RooFormulaVar+ ;
#pragma link C++ class RooGenContext+ ;
#pragma link C++ class RooGenericPdf+ ;
#pragma link C++ class RooHist+ ;
#pragma link C++ class RooInvTransform+ ;
#pragma link C++ class RooLinearVar+ ;
#pragma link C++ class RooLinearCombination+ ;
#pragma link C++ class RooLinkedListElem+ ;
#pragma link C++ class RooLinkedList- ;
#pragma link C++ class RooLinTransBinning+ ;
#pragma read sourceClass="RooList" targetClass="TList";
#pragma link C++ class RooListProxy+ ;
#pragma link C++ class RooCollectionProxy<RooArgList>+ ;
#pragma read sourceClass="RooListProxy" targetClass="RooCollectionProxy<RooArgList>";
#pragma link C++ class RooMappedCategory+ ;
#pragma read sourceClass="RooMappedCategory" targetClass="RooMappedCategory" version="[1]" include="RooFitLegacy/RooCatTypeLegacy.h" source="RooCatType* _defCat" target="_defCat" code="{ _defCat = onfile._defCat->getVal(); }"
#pragma link C++ class RooMappedCategory::Entry+;
#pragma read sourceClass="RooMappedCategory::Entry" targetClass="RooMappedCategory::Entry" version="[1]" include="RooFitLegacy/RooCatTypeLegacy.h" \
    source="RooCatType _cat" target="_catIdx" code="{ _catIdx = onfile._cat.getVal(); }"
#pragma link C++ class RooMultiCategory+ ;
#pragma link C++ class RooMultiPdf + ;
#pragma link off class RooNameReg+ ;
#pragma link C++ class RooNumConvolution+ ;
#pragma link C++ class RooNumConvPdf+ ;
#pragma link C++ class RooNumIntConfig+ ;
#pragma link C++ class RooNumIntFactory+ ;
#pragma link C++ class RooPlotable+ ;
#pragma link C++ class RooPlot- ;
#pragma read sourceClass="RooPlot" targetClass="RooPlot" version="[2]"       \
             source="TList _items" target="_items"                           \
             code="{  RooPlot::fillItemsFromTList(_items, onfile._items); }"
#pragma link C++ class RooPolyFunc+ ;
#pragma link C++ class RooPolyVar+ ;
#pragma link C++ class RooPrintable+ ;
#pragma link C++ class RooProdGenContext+ ;
#pragma link C++ class RooProduct+ ;
#pragma read sourceClass="RooProduct" targetClass="RooProduct" version="[1]"                      \
             source="RooSetProxy _compRSet; RooSetProxy _compCSet" target="_compRSet, _compCSet"  \
             code="{                                                                              \
                 _compRSet.initializeAfterIOConstructor(newObj, onfile._compRSet) ;               \
                 _compCSet.initializeAfterIOConstructor(newObj, onfile._compCSet) ;               \
             }"
#pragma link C++ class RooPullVar+ ;
#pragma link C++ class RooQuasiRandomGenerator+ ;
#pragma link C++ class RooRatio+ ;
#pragma link C++ class RooRandom+ ;
#pragma link off class RooErrorHandler+ ;


// Old LinkDef3.h
#pragma link C++ class RooCacheManager<std::vector<double> >+ ;
#pragma link C++ class std::list<RooAbsData*>+ ;
#pragma link C++ class std::map<string,RooAbsData*>+ ;
#pragma link C++ class RooCacheManager<RooAbsCacheElement>+ ;
#pragma link C++ class RooRandomizeParamMCSModule+ ;
#pragma link C++ class RooRangeBinning+ ;
#pragma link C++ class RooRealBinding+ ;
#pragma link C++ class RooRealConstant+ ;
#pragma link C++ class RooRealIntegral+ ;
#pragma link C++ class RooRealProxy+;
#pragma read sourceClass="RooRealProxy" targetClass="RooTemplateProxy<RooAbsReal>";
#pragma link C++ class RooTemplateProxy<RooAbsPdf>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooTemplateProxy<RooAbsPdf>";
#pragma link C++ class RooTemplateProxy<RooAbsRealLValue>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooTemplateProxy<RooAbsRealLValue>";
#pragma link C++ class RooTemplateProxy<RooRealVar>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooTemplateProxy<RooRealVar>";
#pragma link C++ class RooCategoryProxy+ ;
#pragma link C++ class RooTemplateProxy<RooMultiCategory>+;
#pragma read sourceClass="RooCategoryProxy" targetClass="RooTemplateProxy<RooMultiCategory>";
#pragma link C++ class RooTemplateProxy<RooAbsCategoryLValue>+;
#pragma read sourceClass="RooCategoryProxy" targetClass="RooTemplateProxy<RooAbsCategoryLValue>";
#pragma link C++ class RooTemplateProxy<RooHistFunc>+;
#pragma link C++ class RooTemplateProxy<const RooHistFunc>+;
#pragma link C++ class RooRealVar- ;
#pragma link C++ class RooRealVarSharedProperties+ ;
#pragma read sourceClass="RooRealVarSharedProperties" targetClass="RooRealVarSharedProperties" version="[1]" \
  include="RooLinkedList.h" \
  source="RooLinkedList _altBinning" target="_altBinning" \
  code="{ \
    for (TObject * binning : onfile._altBinning) { _altBinning[binning->GetName()] = static_cast<RooAbsBinning*>(binning); } \
  }"
#pragma link C++ class RooRefCountList+ ;
#pragma link C++ class RooSetProxy+ ;
#pragma link C++ class RooCollectionProxy<RooArgSet>+ ;
#pragma read sourceClass="RooSetProxy" targetClass="RooCollectionProxy<RooArgSet>";
#pragma link C++ class RooSharedProperties+ ;
#pragma link C++ class RooSimGenContext+ ;
#pragma link C++ class RooSimSplitGenContext+ ;
#pragma link C++ class RooStreamParser+ ;
#pragma link C++ class RooSuperCategory+ ;
#pragma read sourceClass="RooSuperCategory" targetClass="RooSuperCategory" version="[1]" \
  source="RooSetProxy _catSet" target="_multiCat"\
  code="{auto newArg = new RooMultiCategory((std::string(newObj->GetName()) + \"_internalMultiCat\").c_str(), newObj->GetTitle(), onfile._catSet); \
         _multiCat.setArg(*newArg); }";
#pragma link C++ class RooTable+ ;
#pragma link C++ class RooThresholdCategory+ ;
#pragma read sourceClass="RooThresholdCategory" targetClass="RooThresholdCategory" version="[1]" \
  include="TSortedList.h" \
  source="RooCatType* _defCat; TSortedList _threshList" target="_defIndex,_threshList" \
  code="{const_cast<int&>(_defIndex) = onfile._defCat->getVal(); \
         class RooThreshEntry : public TObject { public: double _thresh; RooCatType _cat;}; \
         for(auto * te : static_range_cast<RooThreshEntry*>(onfile._threshList)) { \
           _threshList.emplace_back(te->_thresh, te->_cat.getVal()); \
         }\
         }";
#pragma read sourceClass="RooThresholdCategory" targetClass="RooThresholdCategory" version="[2]" \
  source="RooCatType* _defCat; std::vector<std::pair<double,RooCatType>> _threshList" target="_defIndex,_threshList" \
  code="{const_cast<int&>(_defIndex) = onfile._defCat->getVal(); \
         for (const auto& threshCatPair : onfile._threshList) { \
           _threshList.emplace_back(threshCatPair.first, threshCatPair.second.getVal()); \
         }\
         }";
#pragma link C++ class RooTObjWrap+ ;
#pragma link C++ class RooTrace+ ;
#pragma link C++ class RooUniformBinning+ ;
#pragma link C++ class RooSimultaneous+ ;
#pragma link C++ class RooRealSumPdf+ ;
#pragma link C++ class RooRealSumFunc + ;
#pragma link C++ class RooResolutionModel+ ;
#pragma link C++ class RooTruthModel+ ;
#pragma link C++ class RooFit::Detail::RooFixedProdPdf+ ;
#pragma link C++ class RooProdPdf+ ;
#pragma read sourceClass="RooProdPdf" targetClass="RooProdPdf" version="[-5]" \
  source="RooLinkedList _pdfNSetList" target="_pdfNSetList"                  \
  code="{for (auto * nset : static_range_cast<RooArgSet*>(onfile._pdfNSetList)) { \
           _pdfNSetList.emplace_back(nset);                                       \
         }                                                                        \
         }";
#pragma link C++ class RooMCStudy+ ;
#pragma link C++ class RooMsgService+ ;
#pragma link C++ class RooMsgService::StreamConfig+ ;
#pragma link C++ class RooProjectedPdf+ ;
#pragma link C++ class RooWorkspace- ;
#pragma link C++ class RooWorkspace::CodeRepo- ;
#pragma link C++ class RooWorkspace::WSDir+ ;
#pragma link C++ class RooWorkspaceHandle+;
#pragma link C++ class std::list<TObject*>+ ;
#pragma link C++ class std::list<RooAbsData*>+ ;
#pragma link C++ class RooProfileLL+ ;
#pragma link C++ class RooAbsCachedPdf+ ;
#pragma link C++ class RooAbsCachedPdf::PdfCacheElem+ ;
#pragma link C++ class RooAbsSelfCachedPdf+ ;
#pragma link C++ class RooHistPdf- ;
#pragma link C++ class RooCachedPdf+ ;
#pragma link C++ class RooFFTConvPdf+ ;
#pragma link C++ class RooDataHistSliceIter+ ;
#pragma link C++ class RooObjCacheManager+ ;
#pragma link C++ class RooAbsCache+ ;
#pragma link C++ class RooAbsCacheElement+ ;
#pragma link C++ class RooExtendedTerm+ ;
#pragma link C++ class RooParamBinning+ ;
#pragma link off class RooErrorHandler+ ;


// Old LinkDef4.h
#pragma link C++ class RooConstraintSum+ ;
#pragma link C++ class RooRecursiveFraction+ ;
#pragma link C++ class RooFracRemainder+ ;
#pragma link C++ class RooAbsCachedReal+ ;
#pragma link C++ class RooAbsSelfCachedReal+ ;
#pragma link C++ class RooCachedReal+ ;
#pragma link C++ class RooNumCdf+ ;
#pragma link C++ class RooChangeTracker+ ;
#pragma link C++ class RooNumRunningInt+ ;
#pragma link C++ class RooHistFunc- ;
#pragma link C++ class RooExpensiveObjectCache+ ;
#pragma link C++ class RooExpensiveObjectCache::ExpensiveObject+ ;
#pragma link C++ class std::map<std::string,RooAbsPdf*>+ ;
// The nomap options excludes the class from the roomap file
#pragma link C++ options=nomap class std::map<std::string,TH1*>+ ;
#pragma link C++ class std::map<std::string,RooAbsDataStore*>+ ;
#pragma link C++ class std::list<RooAbsData*>+ ;
#pragma link C++ class std::list<TObject*>+ ;
#pragma link C++ class RooFactoryWSTool+ ;
#pragma link C++ class RooBinningCategory+ ;
#pragma link C++ class RooDerivative+ ;
#pragma link C++ class RooFunctor+ ;
#pragma link C++ class RooNumGenConfig+ ;
#pragma link C++ class RooMultiVarGaussian+ ;
#pragma link C++ class RooMultiVarGaussian::AnaIntData+ ;
#pragma link C++ class RooMultiVarGaussian::GenData+ ;
#pragma link C++ class RooAbsDataStore+ ;
#pragma link C++ class RooTreeDataStore- ;
#pragma link C++ class RooCompositeDataStore+ ;
#pragma link C++ class RooTreeData+ ;
#pragma link C++ class RooRangeBoolean+ ;
#pragma link C++ class RooVectorDataStore- ;
#pragma link C++ class RooVectorDataStore::RealVector+;
#pragma link C++ class RooVectorDataStore::RealFullVector+ ;
#pragma read sourceClass="RooVectorDataStore::RealFullVector" targetClass="RooVectorDataStore::RealFullVector" version="[1]" \
  source="std::vector<double>* _vecE; std::vector<double>* _vecEL; std::vector<double>* _vecEH" target="_vecE, _vecEL, _vecEH" \
  code="{ if(onfile._vecE) { _vecE = *onfile._vecE; }     \
          if(onfile._vecEL) { _vecEL = *onfile._vecEL; }  \
          if(onfile._vecEH) { _vecEH = *onfile._vecEH; }  }";
#pragma link C++ class RooVectorDataStore::CatVector+;
#pragma read sourceClass="RooVectorDataStore::CatVector" targetClass="RooVectorDataStore::CatVector" version="[1]" \
  source="std::vector<RooCatType> _vec;" target="_vec" \
  include="RooFitLegacy/RooCatTypeLegacy.h" \
  code="{_vec.reserve(onfile._vec.size()); for (const auto& cat : onfile._vec) { _vec.push_back(cat.getVal()); } }";
#pragma link C++ class std::pair<std::string,RooAbsData*>+ ;
#pragma link C++ class std::pair<int,RooLinkedListElem*>+ ;
#pragma link C++ class RooMinimizer+ ;
#pragma link C++ class RooFit::TestStatistics::RooRealL+ ;
#pragma link C++ class RooAbsMoment+ ;
#pragma link C++ class RooMoment+ ;
#pragma link C++ class RooFirstMoment+ ;
#pragma link C++ class RooSecondMoment+ ;
#pragma link C++ class RooStats::ModelConfig+;
#pragma link C++ class RooStudyManager+ ;
#pragma link C++ class RooStudyPackage+ ;
#pragma link C++ class RooAbsStudy+ ;
#pragma link C++ class RooGenFitStudy+ ;
#pragma link C++ class RooExtendedBinding+ ;
#pragma link C++ class std::list<RooAbsStudy*>+ ;
#pragma link C++ class std::map<string,RooDataSet*>+ ;
#pragma link C++ class std::map<string,RooDataHist*>+ ;
// The nomap options excludes the class from the roomap file
#pragma link C++ options=nomap class std::map<string,TH1*>+ ;
#pragma link off class RooErrorHandler+ ;
#pragma link C++ class RooBinSamplingPdf+;
#pragma link C++ class RooBinWidthFunction+;
#pragma link C++ class RooFit::Detail::RooNLLVarNew+;
#pragma link C++ class RooFit::Detail::RooNormalizedPdf+ ;

#endif
