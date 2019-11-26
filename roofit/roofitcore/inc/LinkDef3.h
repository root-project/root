#ifdef __CINT__ 
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
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
#pragma link C++ class RooStringVar+ ;
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
#endif 
 
