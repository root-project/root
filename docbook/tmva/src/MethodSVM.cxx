// @(#)root/tmva $Id$
// Author: Marcin Wolter, Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSVM                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <azemla@cern.ch>          - IFJ PAN, Krakow, Poland        *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *
 *                                                                                *
 * Introduction of regression by:                                                 *
 *      Krzysztof Danielowski <danielow@cern.ch> - IFJ PAN & AGH, Krakow, Poland  *
 *      Kamil Kraszewski      <kalq@cern.ch>     - IFJ PAN & UJ, Krakow, Poland   *
 *      Maciej Kruk           <mkruk@cern.ch>    - IFJ PAN & AGH, Krakow, Poland  *
 *                                                                                *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// SMO Platt's SVM classifier with Keerthi & Shavade improvements
//_______________________________________________________________________

#include "Riostream.h"
#include "TMath.h"
#include "TFile.h"

#include "TMVA/ClassifierFactory.h"
#ifndef ROOT_TMVA_MethodSVM
#include "TMVA/MethodSVM.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif

#ifndef ROOT_TMVA_SVWorkingSet
#include "TMVA/SVWorkingSet.h"
#endif

#ifndef ROOT_TMVA_SVEvent
#include "TMVA/SVEvent.h"
#endif

#ifndef ROOT_TMVA_SVKernelFunction
#include "TMVA/SVKernelFunction.h"
#endif

#include <string>

const Int_t basketsize__ = 1280000;
REGISTER_METHOD(SVM)

ClassImp(TMVA::MethodSVM)

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( const TString& jobName, const TString& methodTitle, DataSetInfo& theData,
                            const TString& theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, Types::kSVM, methodTitle, theData, theOption, theTargetDir )
   , fCost(0)
   , fTolerance(0)
   , fMaxIter(0)
   , fNSubSets(0)
   , fBparm(0)
   , fGamma(0)
   , fWgSet(0)
   , fInputData(0)
   , fSupportVectors(0)
   , fSVKernelFunction(0)
   , fMinVars(0)
   , fMaxVars(0)
   , fDoubleSigmaSquared(0)
   , fOrder(0)
   , fTheta(0)
   , fKappa(0)
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( DataSetInfo& theData, const TString& theWeightFile, TDirectory*  theTargetDir )
   : MethodBase( Types::kSVM, theData, theWeightFile, theTargetDir )
   , fCost(0)
   , fTolerance(0)
   , fMaxIter(0)
   , fNSubSets(0)
   , fBparm(0)
   , fGamma(0)
   , fWgSet(0)
   , fInputData(0)
   , fSupportVectors(0)
   , fSVKernelFunction(0)
   , fMinVars(0)
   , fMaxVars(0)
   , fDoubleSigmaSquared(0)
   , fOrder(0)
   , fTheta(0)
   , fKappa(0)
{
   // constructor from weight file
}

//_______________________________________________________________________
TMVA::MethodSVM::~MethodSVM()
{
   // destructor
   if (fInputData !=0)       { delete fInputData; fInputData=0; }
   if (fSupportVectors !=0 ) { delete fSupportVectors; fSupportVectors = 0; }
   if (fWgSet !=0)           { delete fWgSet; fWgSet=0; }
   if (fSVKernelFunction !=0 ) { delete fSVKernelFunction; fSVKernelFunction = 0; }
}

//_______________________________________________________________________
Bool_t TMVA::MethodSVM::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   // SVM can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kRegression     && numberTargets == 1) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodSVM::Init()
{
   // default initialisation

   // SVM always uses normalised input variables
   SetNormalised( kTRUE );

   fInputData = new std::vector<TMVA::SVEvent*>(Data()->GetNEvents());
   fSupportVectors = new std::vector<TMVA::SVEvent*>(0);
}

//_______________________________________________________________________
void TMVA::MethodSVM::DeclareOptions()
{
   // declare options available for this method
   DeclareOptionRef( fCost,   "C",        "Cost parameter" );
   if (DoRegression()) {
      fCost = 0.002;
   }else{
      fCost = 1.;
   }
   DeclareOptionRef( fTolerance = 0.01, "Tol",      "Tolerance parameter" );  //should be fixed
   DeclareOptionRef( fMaxIter   = 1000, "MaxIter",  "Maximum number of training loops" );
   DeclareOptionRef( fNSubSets  = 1,    "NSubSets", "Number of training subsets" );

   // for gaussian kernel parameter(s)
   DeclareOptionRef( fGamma = 1., "Gamma", "RBF kernel parameter: Gamma");
}

//_______________________________________________________________________
void TMVA::MethodSVM::DeclareCompatibilityOptions()
{
   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef( fTheKernel = "Gauss", "Kernel", "Uses kernel function");
   // for gaussian kernel parameter(s)
   DeclareOptionRef( fDoubleSigmaSquared = 2., "Sigma", "Kernel parameter: sigma");
   // for polynomiarl kernel parameter(s)
   DeclareOptionRef( fOrder = 3, "Order", "Polynomial Kernel parameter: polynomial order");
   // for sigmoid kernel parameters
   DeclareOptionRef( fTheta = 1., "Theta", "Sigmoid Kernel parameter: theta");
   DeclareOptionRef( fKappa = 1., "Kappa", "Sigmoid Kernel parameter: kappa");
}

//_______________________________________________________________________
void TMVA::MethodSVM::ProcessOptions()
{
   // option post processing (if necessary)
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kFATAL << "Mechanism to ignore events with negative weights in training not yet available for method: "
            << GetMethodTypeName()
            << " --> please remove \"IgnoreNegWeightsInTraining\" option from booking string."
            << Endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodSVM::Train()
{
   // Train SVM
   Data()->SetCurrentType(Types::kTraining);

   for (Int_t ievt=0; ievt<Data()->GetNEvents(); ievt++){
      Log() << kDEBUG << "Create event vector"<< Endl;
      fInputData->at(ievt) = new SVEvent(GetEvent(ievt), fCost);
   }

   fSVKernelFunction = new SVKernelFunction(fGamma);

   Log()<< kINFO << "Building SVM Working Set..."<< Endl;
   Timer bldwstime( GetName());
   fWgSet = new SVWorkingSet( fInputData, fSVKernelFunction,fTolerance, DoRegression() );
   Log() << kINFO <<"Elapsed time for Working Set build: "<< bldwstime.GetElapsedTime()<<Endl;

   // timing
   Timer timer( GetName() );
   Log() << kINFO << "Sorry, no computing time forecast available for SVM, please wait ..." << Endl;

   fWgSet->Train(fMaxIter);

   Log() << kINFO << "Elapsed time: " << timer.GetElapsedTime()
         << "                                          " << Endl;

   fBparm          = fWgSet->GetBpar();
   fSupportVectors = fWgSet->GetSupportVectors();
}

//_______________________________________________________________________
void TMVA::MethodSVM::AddWeightsXMLTo( void* parent ) const
{
   // write configuration to xml file
   void* wght = gTools().AddChild(parent, "Weights");
   gTools().AddAttr(wght,"fBparm",fBparm);
   gTools().AddAttr(wght,"fGamma",fGamma);
   gTools().AddAttr(wght,"NSupVec",fSupportVectors->size());

   for (std::vector<TMVA::SVEvent*>::iterator veciter=fSupportVectors->begin();
        veciter!=fSupportVectors->end() ; ++veciter ) {
      TVectorD temp(GetNvar()+4);
      temp[0] = (*veciter)->GetNs();
      temp[1] = (*veciter)->GetTypeFlag();
      temp[2] = (*veciter)->GetAlpha();
      temp[3] = (*veciter)->GetAlpha_p();
      for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
         temp[ivar+4] = (*(*veciter)->GetDataVector())[ivar];
      gTools().WriteTVectorDToXML(wght,"SupportVector",&temp);
   }
   // write max/min data values
   void* maxnode = gTools().AddChild(wght, "Maxima");
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      gTools().AddAttr(maxnode, "Var"+gTools().StringFromInt(ivar), GetXmax(ivar));
   void* minnode = gTools().AddChild(wght, "Minima");
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      gTools().AddAttr(minnode, "Var"+gTools().StringFromInt(ivar), GetXmin(ivar));
}

//_______________________________________________________________________
void TMVA::MethodSVM::ReadWeightsFromXML( void* wghtnode )
{
   gTools().ReadAttr( wghtnode, "fBparm",fBparm   );
   gTools().ReadAttr( wghtnode, "fGamma",fGamma);
   UInt_t fNsupv=0;
   gTools().ReadAttr( wghtnode, "NSupVec",fNsupv   );

   Float_t alpha=0.;
   Float_t alpha_p = 0.;

   Int_t typeFlag=-1;
   UInt_t ns = 0;
   std::vector<Float_t>* svector = new std::vector<Float_t>(GetNvar());

   if (fMaxVars!=0) delete fMaxVars;
   fMaxVars = new TVectorD( GetNvar() );
   if (fMinVars!=0) delete fMinVars;
   fMinVars = new TVectorD( GetNvar() );
   if (fSupportVectors!=0) {
      for (vector< SVEvent* >::iterator it = fSupportVectors->begin(); it!=fSupportVectors->end(); ++it)
         delete *it;
      delete fSupportVectors;
   }
   fSupportVectors = new std::vector<TMVA::SVEvent*>(0);
   void* supportvectornode = gTools().GetChild(wghtnode);
   for (UInt_t ievt = 0; ievt < fNsupv; ievt++) {
      TVectorD temp(GetNvar()+4);
      gTools().ReadTVectorDFromXML(supportvectornode,"SupportVector",&temp);
      ns=(UInt_t)temp[0];
      typeFlag=(int)temp[1];
      alpha=temp[2];
      alpha_p=temp[3];
      for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
         (*svector)[ivar]=temp[ivar+4];

      fSupportVectors->push_back(new SVEvent(svector,alpha,alpha_p,typeFlag));
      supportvectornode = gTools().GetNextChild(supportvectornode);
   }

   void* maxminnode = supportvectornode;
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      gTools().ReadAttr( maxminnode,"Var"+gTools().StringFromInt(ivar),(*fMaxVars)[ivar]);
   maxminnode = gTools().GetNextChild(maxminnode);
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      gTools().ReadAttr( maxminnode,"Var"+gTools().StringFromInt(ivar),(*fMinVars)[ivar]);
   if (fSVKernelFunction!=0) delete fSVKernelFunction;
   fSVKernelFunction = new SVKernelFunction(fGamma);
   delete svector;
}

//_______________________________________________________________________
void TMVA::MethodSVM::WriteWeightsToStream( TFile& ) const
{
   //TODO write IT
   // write training sample (TTree) to file
}

//_______________________________________________________________________
void  TMVA::MethodSVM::ReadWeightsFromStream( istream& istr )
{
   if (fSupportVectors !=0) { delete fSupportVectors; fSupportVectors = 0;}
   fSupportVectors = new std::vector<TMVA::SVEvent*>(0);

   // read configuration from input stream
   istr >> fBparm;

   UInt_t fNsupv;
   // coverity[tainted_data_argument]
   istr >> fNsupv;
   fSupportVectors->reserve(fNsupv);

   Float_t typeTalpha=0.;
   Float_t alpha=0.;
   Int_t typeFlag=-1;
   UInt_t ns = 0;
   std::vector<Float_t>* svector = new std::vector<Float_t>(GetNvar());

   fMaxVars = new TVectorD( GetNvar() );
   fMinVars = new TVectorD( GetNvar() );

   for (UInt_t ievt = 0; ievt < fNsupv; ievt++) {
      istr>>ns;
      istr>>typeTalpha;
      typeFlag = typeTalpha<0?-1:1;
      alpha = typeTalpha<0?-typeTalpha:typeTalpha;
      for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
         istr>>svector->at(ivar);

      fSupportVectors->push_back(new SVEvent(svector,alpha,typeFlag,ns));
   }

   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      istr >> (*fMaxVars)[ivar];

   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++)
      istr >> (*fMinVars)[ivar];

   delete fSVKernelFunction;
   if (fTheKernel == "Gauss" ) {
      fSVKernelFunction = new SVKernelFunction(1/fDoubleSigmaSquared);
   } else {
      SVKernelFunction::EKernelType k = SVKernelFunction::kLinear;
      if(fTheKernel == "Linear")           k = SVKernelFunction::kLinear;
      else if (fTheKernel == "Polynomial") k = SVKernelFunction::kPolynomial;
      else if (fTheKernel == "Sigmoid"   ) k = SVKernelFunction::kSigmoidal;
      else {
         Log() << kFATAL <<"Unknown kernel function found in weight file!" << Endl;
      }
      fSVKernelFunction = new SVKernelFunction();
      fSVKernelFunction->setCompatibilityParams(k, fOrder, fTheta, fKappa);
   }
   delete svector;
}

//_______________________________________________________________________
void TMVA::MethodSVM::ReadWeightsFromStream( TFile& /* fFin */ )
{
   // TODO write IT
}

//_______________________________________________________________________
Double_t TMVA::MethodSVM::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // returns MVA value for given event
   Double_t myMVA = 0;

   // TODO: avoid creation of a new SVEvent every time (Joerg)
   SVEvent* ev = new SVEvent( GetEvent(),0. ); //check for specificators

   for (UInt_t ievt = 0; ievt < fSupportVectors->size() ; ievt++) {
      myMVA += ( fSupportVectors->at(ievt)->GetAlpha()
                 * fSupportVectors->at(ievt)->GetTypeFlag()
                 * fSVKernelFunction->Evaluate( fSupportVectors->at(ievt), ev ) );
   }

   delete ev;

   myMVA -= fBparm;

   // cannot determine error
   NoErrorCalc(err, errUpper);

   // 08/12/09: changed sign here to make results agree with convention signal=1
   return 1.0/(1.0 + TMath::Exp(myMVA));
}
//_______________________________________________________________________
const std::vector<Float_t>& TMVA::MethodSVM::GetRegressionValues()
{

   if( fRegressionReturnVal == NULL )
      fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();

   Double_t myMVA = 0;

   const Event *baseev = GetEvent();
   SVEvent* ev = new SVEvent( baseev,0. ); //check for specificators

   for (UInt_t ievt = 0; ievt < fSupportVectors->size() ; ievt++) {
      myMVA += ( fSupportVectors->at(ievt)->GetDeltaAlpha()
                 *fSVKernelFunction->Evaluate( fSupportVectors->at(ievt), ev ) );
   }
   myMVA += fBparm;
   Event * evT = new Event(*baseev);
   evT->SetTarget(0,myMVA);

   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );

   fRegressionReturnVal->push_back(evT2->GetTarget(0));

   delete evT;

   delete ev;

   return *fRegressionReturnVal;
}

//_______________________________________________________________________
void TMVA::MethodSVM::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   const int fNsupv = fSupportVectors->size();
   fout << "   // not implemented for class: \"" << className << "\"" << endl;
   fout << "   float        fBparameter;" << endl;
   fout << "   int          fNOfSuppVec;" << endl;
   fout << "   static float fAllSuppVectors[][" << fNsupv << "];" << endl;
   fout << "   static float fAlphaTypeCoef[" << fNsupv << "];" << endl;
   fout << endl;
   fout << "   // Kernel parameter(s) " << endl;
   fout << "   float fGamma;"  << endl;
   fout << "};" << endl;
   fout << "" << endl;

   //Initialize function definition
   fout << "inline void " << className << "::Initialize() " << endl;
   fout << "{" << endl;
   fout << "   fBparameter = " << fBparm << ";" << endl;
   fout << "   fNOfSuppVec = " << fNsupv << ";" << endl;
   fout << "   fGamma = " << fGamma << ";" <<endl;
   fout << "}" << endl;
   fout << endl;

   // GetMvaValue__ function defninition
   fout << "inline double " << className << "::GetMvaValue__(const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double mvaval = 0; " << endl;
   fout << "   double temp = 0; " << endl;
   fout << endl;
   fout << "   for (int ievt = 0; ievt < fNOfSuppVec; ievt++ ){" << endl;
   fout << "      temp = 0;" << endl;
   fout << "      for ( unsigned int ivar = 0; ivar < GetNvar(); ivar++ ) {" << endl;

   fout << "         temp += (fAllSuppVectors[ivar][ievt] - inputValues[ivar])  " << endl;
   fout << "               * (fAllSuppVectors[ivar][ievt] - inputValues[ivar]); " << endl;
   fout << "      }" << endl;
   fout << "      mvaval += fAlphaTypeCoef[ievt] * exp( -fGamma * temp ); " << endl;

   fout << "   }" << endl;
   fout << "   mvaval -= fBparameter;" << endl;
   fout << "   return 1./(1. + exp(mvaval));" << endl;
   fout << "}" << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   // nothing to clear " << endl;
   fout << "}" << endl;
   fout << "" << endl;

   // define support vectors
   fout << "float " << className << "::fAlphaTypeCoef[] =" << endl;
   fout << "{ ";
   for (Int_t isv = 0; isv < fNsupv; isv++) {
      fout << fSupportVectors->at(isv)->GetDeltaAlpha() * fSupportVectors->at(isv)->GetTypeFlag();
      if (isv < fNsupv-1) fout << ", ";
   }
   fout << " };" << endl << endl;

   fout << "float " << className << "::fAllSuppVectors[][" << fNsupv << "] =" << endl;
   fout << "{";
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++) {
      fout << endl;
      fout << "   { ";
      for (Int_t isv = 0; isv < fNsupv; isv++){
         fout << fSupportVectors->at(isv)->GetDataVector()->at(ivar);
         if (isv < fNsupv-1) fout << ", ";
      }
      fout << " }";
      if (ivar < GetNvar()-1) fout << ", " << endl;
      else                    fout << endl;
   }
   fout << "};" << endl<< endl;
}

//_______________________________________________________________________
void TMVA::MethodSVM::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The Support Vector Machine (SVM) builds a hyperplance separating" << Endl;
   Log() << "signal and background events (vectors) using the minimal subset of " << Endl;
   Log() << "all vectors used for training (support vectors). The extension to" << Endl;
   Log() << "the non-linear case is performed by mapping input vectors into a " << Endl;
   Log() << "higher-dimensional feature space in which linear separation is " << Endl;
   Log() << "possible. The use of the kernel functions thereby eliminates the " << Endl;
   Log() << "explicit transformation to the feature space. The implemented SVM " << Endl;
   Log() << "algorithm performs the classification tasks using linear, polynomial, " << Endl;
   Log() << "Gaussian and sigmoidal kernel functions. The Gaussian kernel allows " << Endl;
   Log() << "to apply any discriminant shape in the input space." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "SVM is a general purpose non-linear classification method, which " << Endl;
   Log() << "does not require data preprocessing like decorrelation or Principal " << Endl;
   Log() << "Component Analysis. It generalises quite well and can handle analyses " << Endl;
   Log() << "with large numbers of input variables." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "Optimal performance requires primarily a proper choice of the kernel " << Endl;
   Log() << "parameters (the width \"Sigma\" in case of Gaussian kernel) and the" << Endl;
   Log() << "cost parameter \"C\". The user must optimise them empirically by running" << Endl;
   Log() << "SVM several times with different parameter sets. The time needed for " << Endl;
   Log() << "each evaluation scales like the square of the number of training " << Endl;
   Log() << "events so that a coarse preliminary tuning should be performed on " << Endl;
   Log() << "reduced data sets." << Endl;
}
