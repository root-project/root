#ifdef __CINT__ 

//Old LinkDef1.h
#pragma link off all globals;
#pragma link off all classes;
#pragma link C++ class Roo1DTable+ ;
#pragma link C++ class RooAbsArg- ;
#pragma link C++ class stack<RooAbsArg*,deque<RooAbsArg*> > ;
#pragma link C++ class RooRefArray- ;
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-4]" source="TList _proxyList" target="_proxyList" \
    code="{ TIterator* iter = onfile._proxyList.MakeIterator() ; TObject* tmpObj ; while ((tmpObj = iter->Next())) { _proxyList.Add(tmpObj) ; } delete iter ; }" 
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[5]" source="TRefArray _proxyList" target="_proxyList" \
  code="{ _proxyList.GetSize() ; if (onfile._proxyList.GetSize()>0) { RooAbsArg::_ioEvoList[newObj] = new TRefArray(onfile._proxyList) ; } }" 
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _serverList" target="_serverList" \
  code="{ _serverList = RooFit::STLRefCountListHelpers::convert(onfile._serverList); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientList" target="_clientList" \
  code="{ _clientList = RooFit::STLRefCountListHelpers::convert(onfile._clientList); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientListValue" target="_clientListValue" \
  code="{ _clientListValue = RooFit::STLRefCountListHelpers::convert(onfile._clientListValue); }"
#pragma read sourceClass="RooAbsArg" targetClass="RooAbsArg" version="[1-6]"\
  source="RooRefCountList _clientListShape" target="_clientListShape" \
  code="{ _clientListShape = RooFit::STLRefCountListHelpers::convert(onfile._clientListShape); }"
#pragma link C++ class RooSTLRefCountList<RooAbsArg>+;
#pragma link C++ class RooStringVar+ ;
#pragma read sourceClass="RooAbsString" targetClass="RooStringVar" version="[1]" source="Int_t _len; char *_value" target="_string" code="{_string.assign(onfile._value, onfile._len);}"
#pragma link C++ class RooAbsBinning- ;
#pragma link C++ class RooAbsCategory+ ;
#pragma read sourceClass="RooAbsCategory" targetClass="RooAbsCategory" version="[1]" \
  source="TObjArray _types" target="_types" \
  code="{TObject* obj; TIterator* it = onfile._types.MakeIterator();\
         while ((obj=it->Next())) {\
           auto cat = dynamic_cast<const RooCatType*>(obj); assert(cat);\
           _types.push_back(new RooCatType(*cat)); }\
         delete it; }";
#pragma link C++ class RooAbsCategoryLValue+ ;
#pragma link C++ class RooAbsCollection+ ;
#pragma read sourceClass="RooAbsCollection" targetClass="RooAbsCollection" version="[1]" source="" target="_allRRV" code="{ _allRRV=kFALSE ; }"
#pragma read sourceClass="RooAbsCollection" targetClass="RooAbsCollection" version="[2]"\
    source="RooLinkedList _list" target="_list" code="{ RooFIter iter = onfile._list.fwdIterator(); RooAbsArg * theArg;\
    while ((theArg = iter.next())) {_list.push_back(theArg);} }"
#pragma link C++ class RooAbsData- ;
#pragma link C++ class RooAbsFunc+ ;
#pragma link C++ class RooAbsGenContext+ ;
#pragma link C++ class RooAbsTestStatistic+ ;
#pragma link C++ class RooAbsHiddenReal+ ;
#pragma link C++ class RooAbsIntegrator+ ;
#pragma link C++ class RooAbsLValue+ ;
#pragma link C++ class RooAbsMCStudyModule+ ;
#pragma link C++ class RooAbsOptTestStatistic+ ;
#pragma link C++ class RooAbsPdf+ ;
#pragma link C++ class RooAbsPdf::GenSpec+ ;
#pragma link C++ class RooAbsProxy+ ;
#pragma link C++ class RooAbsReal+ ;
#pragma link C++ class RooAbsRealLValue+ ;
#pragma link C++ class RooAbsRootFinder+ ;
#pragma link C++ class RooAcceptReject+ ;
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
#pragma link C++ class RooCategoryProxy+ ;
#pragma link C++ class RooCategorySharedProperties+ ;
#pragma link C++ class RooCatType+ ;
#pragma link C++ class RooChi2Var+ ;
#pragma link C++ class RooClassFactory+ ;
#pragma link C++ class RooCmdArg+ ;
#pragma link C++ class RooCmdConfig+ ;
#pragma link C++ class RooConstVar+ ;
#pragma link C++ class RooConvCoefVar+ ;
#pragma link C++ class RooConvGenContext+ ;
#pragma link C++ class RooConvIntegrandBinding+ ;
#pragma link C++ class RooCurve+ ;
#pragma link C++ class RooCustomizer+ ;
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
#pragma link C++ namespace RooFit ;
#pragma link C++ namespace RooFitShortHand ;
#pragma link C++ class RooGlobalFunc;
#pragma link C++ class RooDouble+ ;
#pragma link C++ class RooEffGenContext+ ;
#pragma link C++ class RooEllipse+ ;
#pragma link C++ class RooErrorVar+ ;
#pragma link C++ class RooFitResult- ;
#pragma link C++ class RooFIter+ ;
#pragma link C++ class RooFormula+ ;
#pragma link C++ class RooFormulaVar+ ;
#pragma link C++ class RooGenContext+ ;
#pragma link C++ class RooGenericPdf+ ;
#pragma link C++ class RooGenProdProj+ ;
#pragma link C++ class RooGrid+ ;
#pragma link C++ class RooHashTable+ ;
#pragma link C++ class RooHistError+ ;
#pragma link C++ class RooHist+ ;
#pragma link C++ class RooImproperIntegrator1D+ ;
#pragma link C++ class RooIntegrator1D+ ;
#pragma link C++ class RooBinIntegrator+ ;
#pragma link C++ class RooIntegrator2D+ ;
#pragma link C++ class RooIntegratorBinding+ ;
#pragma link C++ class RooInt+ ;
#pragma link C++ class RooInvTransform+ ;
#pragma link C++ class RooLinearVar+ ;
#pragma link C++ class RooLinkedListElem+ ;
#pragma link C++ class RooLinkedList- ;
#pragma link C++ class RooLinTransBinning+ ;
#pragma link C++ class RooList+ ;
#pragma link C++ class RooListProxy+ ;
#pragma link C++ class RooMapCatEntry+ ;
#pragma link C++ class RooMappedCategory+ ;
#pragma link C++ class RooMappedCategory::Entry- ;
#pragma link C++ class RooMath+ ;
#pragma link C++ class RooMCIntegrator+ ;
#pragma link C++ class RooMinuit+ ;
#pragma link C++ class RooMPSentinel+ ;
#pragma link C++ class RooMultiCategory+ ;
#pragma link C++ class RooMultiCatIter+ ;
#pragma link off class RooNameReg+ ;
#pragma link C++ class RooNameSet+ ;
#pragma link C++ class RooNLLVar+ ;
#pragma link C++ class RooNormSetCache+ ;
#pragma link C++ class RooNumber+ ;
#pragma link C++ class RooNumConvolution+ ;
#pragma link C++ class RooNumConvPdf+ ;
#pragma link C++ class RooNumIntConfig+ ;
#pragma link C++ class RooNumIntFactory+ ;
#pragma link C++ class RooPlotable+ ;
#pragma link C++ class RooPlot- ;
#pragma link C++ class RooPolyVar+ ;
#pragma link C++ class RooPrintable+ ;
#pragma link C++ class RooProdGenContext+ ;
#pragma link C++ class RooProduct+ ;
#pragma read sourceClass="RooProduct" targetClass="RooProduct" version="[1]" source="RooSetProxy _compRSet" target="_compRSet" code="{ _compRSet.add(onfile._compRSet) ; }"
#pragma read sourceClass="RooProduct" targetClass="RooProduct" version="[1]" source="RooSetProxy _compCSet" target="_compCSet" code="{ _compCSet.add(onfile._compCSet) ; }"
#pragma link C++ class RooPullVar+ ;
#pragma link C++ class RooQuasiRandomGenerator+ ;
#pragma link C++ class RooRandom+ ;
#pragma link off class RooErrorHandler+ ;
 

// Old LinkDef3.h
#pragma link C++ class RooCacheManager<std::vector<double> >+ ;
#pragma link C++ class std::list<RooAbsData*>+ ;
#pragma link C++ class std::map<string,RooAbsData*>+ ;
#pragma link C++ class RooCacheManager<RooAbsCacheElement>+ ;
#pragma link C++ class RooRandomizeParamMCSModule+ ;
#pragma link C++ class RooRangeBinning+ ;
#pragma link C++ class RooRealAnalytic+ ;
#pragma link C++ class RooRealBinding+ ;
#pragma link C++ class RooRealConstant+ ;
#pragma link C++ class RooRealIntegral+ ;
#pragma link C++ class RooRealMPFE+ ;
#pragma link C++ class RooRealProxy+ ;
#pragma link C++ class RooProxy<RooAbsPdf>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooProxy<RooAbsPdf>"
#pragma link C++ class RooProxy<RooAbsRealLValue>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooProxy<RooAbsRealLValue>"
#pragma link C++ class RooProxy<RooRealVar>+;
#pragma read sourceClass="RooRealProxy" targetClass="RooProxy<RooRealVar>"
#pragma link C++ class RooRealVar- ;
#pragma link C++ class RooRealVarSharedProperties+ ;
#pragma link C++ class RooRefCountList+ ;
#pragma link C++ class RooScaledFunc+ ;
#pragma link C++ class RooSegmentedIntegrator1D+ ;
#pragma link C++ class RooSegmentedIntegrator2D+ ;
#pragma link C++ class RooSetPair+ ;
#pragma link C++ class RooSetProxy+ ;
#pragma link C++ class RooSharedProperties+ ;
#pragma link C++ class RooSharedPropertiesList+ ;
#pragma link C++ class RooSimGenContext+ ;
#pragma link C++ class RooSimSplitGenContext+ ;
#pragma link C++ class RooStreamParser+ ;
#pragma link C++ class RooSuperCategory+ ;
#pragma link C++ class RooTable+ ;
#pragma link C++ class RooThresholdCategory+ ;
#pragma read sourceClass="RooThresholdCategory" targetClass="RooThresholdCategory" version="[1]" \
  source="TSortedList _threshList" target="_threshList" \
  code="{class RooThreshEntry : public TObject { public: Double_t _thresh; RooCatType _cat;}; \
         RooThreshEntry* te; \
         auto iter = onfile._threshList.MakeIterator();\
         while( (te = (RooThreshEntry*)iter->Next()) ) { \
           _threshList.emplace_back(te->_thresh, te->_cat); \
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
#pragma link C++ class RooProdPdf+ ;
#pragma link C++ class RooSimPdfBuilder+ ;
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
#pragma link C++ class RooSentinel+ ;
#pragma link C++ class RooParamBinning+ ;
#pragma link off class RooErrorHandler+ ;


// Old LinkDef4.h
#pragma link C++ class RooConstraintSum+ ;
#pragma link C++ class RooRecursiveFraction+ ;
#pragma link C++ class RooDataWeightedAverage+ ;
#pragma link C++ class RooSimWSTool+ ;
#pragma link C++ class RooSimWSTool::SplitRule+ ;
#pragma link C++ class RooSimWSTool::BuildConfig+ ;
#pragma link C++ class RooSimWSTool::MultiBuildConfig+ ;
#pragma link C++ class RooSimWSTool::ObjSplitRule+ ;
#pragma link C++ class RooSimWSTool::ObjBuildConfig+ ;
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
#pragma link C++ class RooGenFunction+ ;
#pragma link C++ class RooMultiGenFunction+ ;
#pragma link C++ class RooTFoamBinding+ ;
#pragma link C++ class RooAdaptiveIntegratorND+ ;
#pragma link C++ class RooAbsNumGenerator+ ;
#pragma link C++ class RooFoamGenerator+ ;
#pragma link C++ class RooNumGenConfig+ ;
#pragma link C++ class RooNumGenFactory+ ;
#pragma link C++ class RooMultiVarGaussian+ ;
#pragma link C++ class RooMultiVarGaussian::AnaIntData+ ;
#pragma link C++ class RooMultiVarGaussian::GenData+ ;
#pragma link C++ class RooXYChi2Var+ ;
#pragma link C++ class RooAbsDataStore+ ;
#pragma link C++ class RooTreeDataStore- ;
#pragma link C++ class RooCompositeDataStore+ ;
#pragma link C++ class RooTreeData+ ;
#pragma link C++ class RooRangeBoolean+ ;
#pragma link C++ class RooVectorDataStore- ;
#pragma link C++ class RooVectorDataStore::RealVector- ;
#pragma link C++ class RooVectorDataStore::RealFullVector- ;
#pragma link C++ class RooVectorDataStore::CatVector- ;
#pragma link C++ class std::pair<std::string,RooAbsData*>+ ;
#pragma link C++ class std::pair<int,RooLinkedListElem*>+ ;
#pragma link C++ class RooUnitTest+ ;
#ifndef __ROOFIT_NOROOMINIMIZER
#pragma link C++ class RooMinimizer+ ;
#pragma link C++ class RooMinimizerFcn+ ;
#endif
#pragma link C++ class RooAbsMoment+ ;
#pragma link C++ class RooMoment+ ;
#pragma link C++ class RooFirstMoment+ ;
#pragma link C++ class RooSecondMoment+ ;
#pragma link C++ class RooStudyManager+ ;
#pragma link C++ class RooStudyPackage+ ;
#pragma link C++ class RooAbsStudy+ ;
#pragma link C++ class RooGenFitStudy+ ;
#pragma link C++ class RooProofDriverSelector+ ;
#pragma link C++ class RooExtendedBinding+ ;
#pragma link C++ class std::list<RooAbsStudy*>+ ;
#pragma link C++ class std::map<string,RooDataSet*>+ ;
#pragma link C++ class std::map<string,RooDataHist*>+ ;
// The nomap options excludes the class from the roomap file
#pragma link C++ options=nomap class std::map<string,TH1*>+ ;
#pragma link off class RooErrorHandler+ ;
#endif 
 
