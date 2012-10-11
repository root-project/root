#ifdef __CINT__ 
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ function Roo* ;
#ifdef USE_FOR_AUTLOADING
#pragma link C++ class RooFit ;
#else
#pragma link C++ namespace RooFit ;
#endif
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
#pragma link C++ class RooGaussKronrodIntegrator1D+ ;
#pragma link C++ class RooGenCategory+ ;
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
#pragma link C++ class RooLinkedListIter+ ;
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
#pragma link C++ class RooNameReg+ ;
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
#endif 
 
