// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Eckhard von Toerne, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TransformationHandler                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>     - U of Bonn, Germany             *  
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <vector>
#include <iomanip>

#include "TMath.h"
#include "TH1.h"
#include "TH2.h"
#include "TAxis.h"
#include "TProfile.h"

#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_Ranking
#include "TMVA/Ranking.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_TransformationHandler
#include "TMVA/TransformationHandler.h"
#endif
#ifndef ROOT_TMVA_VariableTransformBase
#include "TMVA/VariableTransformBase.h"
#endif
#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"
#include "TMVA/VariableRearrangeTransform.h"

//_______________________________________________________________________
TMVA::TransformationHandler::TransformationHandler( DataSetInfo& dsi, const TString& callerName ) 
   : fDataSetInfo(dsi),
     fRootBaseDir(0),
     fCallerName (callerName),
     fLogger     ( new MsgLogger(TString("TFHandler_" + callerName).Data(), kINFO) )
{
   // constructor

   // produce one entry for each class and one entry for all classes. If there is only one class, 
   // produce only one entry
   fNumC = (dsi.GetNClasses()<= 1) ? 1 : dsi.GetNClasses()+1;

   fVariableStats.resize( fNumC );
   for (Int_t i=0; i<fNumC; i++ ) fVariableStats.at(i).resize(dsi.GetNVariables() + dsi.GetNTargets());
}

//_______________________________________________________________________
TMVA::TransformationHandler::~TransformationHandler() 
{
   // destructor
   std::vector<Ranking*>::const_iterator it = fRanking.begin();
   for (; it != fRanking.end(); it++) delete *it;

   fTransformations.SetOwner();
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::SetCallerName( const TString& name ) 
{ 
   fCallerName = name; 
   fLogger->SetSource( TString("TFHandler_" + fCallerName).Data() );
}

//_______________________________________________________________________
TMVA::VariableTransformBase* TMVA::TransformationHandler::AddTransformation( VariableTransformBase *trf, Int_t cls ) 
{
   TString tfname = trf->Log().GetName();
   trf->Log().SetSource(TString(fCallerName+"_"+tfname+"_TF").Data());
   fTransformations.Add(trf);
   fTransformationsReferenceClasses.push_back( cls );
  return trf;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::AddStats( Int_t k, UInt_t ivar, Double_t mean, Double_t rms, Double_t min, Double_t max ) 
{
   if (rms <= 0) {
      Log() << kWARNING << "Variable \"" << Variable(ivar).GetExpression() 
            << "\" has zero or negative RMS^2 " 
            << "==> set to zero. Please check the variable content" << Endl;
      rms = 0;
   }

   VariableStat stat; stat.fMean = mean; stat.fRMS = rms; stat.fMin = min; stat.fMax = max;
   fVariableStats.at(k).at(ivar) = stat;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::SetTransformationReferenceClass( Int_t cls ) 
{
   // overrides the setting for all classes! (this is put in basically for the likelihood-method)
   // be careful with the usage this method
   for (UInt_t i = 0; i < fTransformationsReferenceClasses.size(); i++) {
      fTransformationsReferenceClasses.at( i ) = cls;
   }
}

//_______________________________________________________________________
const TMVA::Event* TMVA::TransformationHandler::Transform( const Event* ev ) const 
{
   // the transformation
   
   TListIter trIt(&fTransformations);
   std::vector<Int_t>::const_iterator rClsIt = fTransformationsReferenceClasses.begin();
   const Event* trEv = ev;
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt()) {
      if (rClsIt == fTransformationsReferenceClasses.end()) Log() << kFATAL<< "invalid read in TransformationHandler::Transform " <<Endl;
      trEv = trf->Transform(trEv, (*rClsIt) );
      rClsIt++;
   }
   return trEv;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::TransformationHandler::InverseTransform( const Event* ev, Bool_t suppressIfNoTargets ) const 
{
   if (fTransformationsReferenceClasses.empty()){
      //Log() << kWARNING << __FILE__ <<":InverseTransform fTransformationsReferenceClasses is empty" << Endl;
      return ev; 
   }
   // the inverse transformation
   TListIter trIt(&fTransformations, kIterBackward);
   std::vector< Int_t >::const_iterator rClsIt = fTransformationsReferenceClasses.end();
   rClsIt--;
   const Event* trEv = ev;
   UInt_t nvars = 0, ntgts = 0, nspcts = 0;
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt() ) { // shouldn't be the transformation called in the inverse order for the inversetransformation?????
      if (trf->IsCreated()) {
	 trf->CountVariableTypes( nvars, ntgts, nspcts );	 
	 if( !(suppressIfNoTargets && ntgts==0) )
	    trEv = trf->InverseTransform(ev, (*rClsIt) );
      }
      else break;
      --rClsIt;
   }
   return trEv;


//    TListIter trIt(&fTransformations);
//    std::vector< Int_t >::const_iterator rClsIt = fTransformationsReferenceClasses.begin();
//    const Event* trEv = ev;
//    UInt_t nvars = 0, ntgts = 0, nspcts = 0;
//    while (VariableTransformBase *trf = (VariableTransformBase*) trIt() ) { // shouldn't be the transformation called in the inverse order for the inversetransformation?????
//       if (trf->IsCreated()) {
// 	 trf->CountVariableTypes( nvars, ntgts, nspcts );	 
// 	 if( !(suppressIfNoTargets && ntgts==0) )
// 	    trEv = trf->InverseTransform(ev, (*rClsIt) );
//       }
//       else break;
//       rClsIt++;
//    }
//    return trEv;

}

//_______________________________________________________________________
std::vector<TMVA::Event*>* TMVA::TransformationHandler::CalcTransformations( const std::vector<Event*>& events, 
                                                                             Bool_t createNewVector ) 
{
   // computation of transformation
   std::vector<Event*>* tmpEvents = const_cast<std::vector<Event*>*>(&events);
   Bool_t replaceColl = kFALSE; // first let TransformCollection create a new vector

   TListIter trIt(&fTransformations);
   std::vector< Int_t >::iterator rClsIt = fTransformationsReferenceClasses.begin();
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt()) {
      if (trf->PrepareTransformation(*tmpEvents)) {
         tmpEvents = TransformCollection(trf, (*rClsIt), tmpEvents, replaceColl);
         // we now created a new vector, so the next transformations replace the 
         // events by their transformed versions
         replaceColl = kTRUE;  
         rClsIt++;
      }
   }

   CalcStats(*tmpEvents);

   // plot the variables once in this transformation
   PlotVariables(*tmpEvents);

   if (!createNewVector) {  // if we don't want that newly created event vector to persist, then delete it
      if (replaceColl) {    
         for ( UInt_t ievt = 0; ievt<tmpEvents->size(); ievt++)
            delete (*tmpEvents)[ievt];
         delete tmpEvents;
      }
      return 0;
   }
   return tmpEvents; // give back the newly created event collection (containing the transformed events)
}

//_______________________________________________________________________
std::vector<TMVA::Event*>* TMVA::TransformationHandler::TransformCollection( VariableTransformBase* trf,
                                                                             Int_t cls,
                                                                             std::vector<TMVA::Event*>* events,
                                                                             Bool_t replace ) const 
{
   // a collection of transformations
   std::vector<TMVA::Event*>* tmpEvents = 0;

   if (replace) {   // the events should be replaced by their transformed versions
      tmpEvents = events;
   } 
   else {           // a new event vector is created
      tmpEvents = new std::vector<TMVA::Event*>(events->size());
   }
   for (UInt_t ievt = 0; ievt<events->size(); ievt++) {  // loop through all events
      if (replace) {  // and replace the event by its transformed version
         *(*tmpEvents)[ievt] = *trf->Transform((*events)[ievt],cls);
      } 
      else {         // and create a new event which is the transformed version of the old event
         (*tmpEvents)[ievt] = new Event(*trf->Transform((*events)[ievt],cls));
      }
   }
   return tmpEvents;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::CalcStats( const std::vector<Event*>& events )
{

   // method to calculate minimum, maximum, mean, and RMS for all
   // variables used in the MVA

   UInt_t nevts = events.size();

   if (nevts==0)
      Log() << kFATAL << "No events available to find min, max, mean and rms" << Endl;

   // if transformation has not been succeeded, the tree may be empty
   const UInt_t nvar = events[0]->GetNVariables();
   const UInt_t ntgt = events[0]->GetNTargets();

   Double_t  *sumOfWeights = new Double_t[fNumC];
   Double_t* *x2           = new Double_t*[fNumC];
   Double_t* *x0           = new Double_t*[fNumC];
   Double_t* *varMin       = new Double_t*[fNumC];
   Double_t* *varMax       = new Double_t*[fNumC];
   
   for (Int_t cls=0; cls<fNumC; cls++) {
      sumOfWeights[cls]=0;
      x2[cls]     = new Double_t[nvar+ntgt];
      x0[cls]     = new Double_t[nvar+ntgt];
      varMin[cls] = new Double_t[nvar+ntgt];
      varMax[cls] = new Double_t[nvar+ntgt];
      for (UInt_t ivar=0; ivar<nvar+ntgt; ivar++) {
         x0[cls][ivar] = x2[cls][ivar] = 0;
         varMin[cls][ivar] = DBL_MAX;
         varMax[cls][ivar] = -DBL_MAX;
      }
   }

   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      Event* ev  = events[ievt];
      Int_t  cls = ev->GetClass();

      Double_t weight = ev->GetWeight();
      sumOfWeights[cls] += weight;
      if (fNumC > 1 ) sumOfWeights[fNumC-1] += weight; // if more than one class, store values for all classes
      for (UInt_t var_tgt = 0; var_tgt < 2; var_tgt++ ){ // first for variables, then for targets
         UInt_t nloop = ( var_tgt==0?nvar:ntgt );
         for (UInt_t ivar=0; ivar<nloop; ivar++) {
            Double_t x = ( var_tgt==0?ev->GetValue(ivar):ev->GetTarget(ivar) );

            if (x < varMin[cls][(var_tgt*nvar)+ivar]) varMin[cls][(var_tgt*nvar)+ivar]= x;
            if (x > varMax[cls][(var_tgt*nvar)+ivar]) varMax[cls][(var_tgt*nvar)+ivar]= x;

            x0[cls][(var_tgt*nvar)+ivar] += x*weight;
            x2[cls][(var_tgt*nvar)+ivar] += x*x*weight;

            if (fNumC > 1) {
               if (x < varMin[fNumC-1][(var_tgt*nvar)+ivar]) varMin[fNumC-1][(var_tgt*nvar)+ivar]= x;
               if (x > varMax[fNumC-1][(var_tgt*nvar)+ivar]) varMax[fNumC-1][(var_tgt*nvar)+ivar]= x;

               x0[fNumC-1][(var_tgt*nvar)+ivar] += x*weight;
               x2[fNumC-1][(var_tgt*nvar)+ivar] += x*x*weight;
            }
         }
      }
   }


   // set Mean and RMS
   for (UInt_t var_tgt = 0; var_tgt < 2; var_tgt++ ){ // first for variables, then for targets
      UInt_t nloop = ( var_tgt==0?nvar:ntgt );
      for (UInt_t ivar=0; ivar<nloop; ivar++) {
         for (Int_t cls = 0; cls < fNumC; cls++) {
            Double_t mean = x0[cls][(var_tgt*nvar)+ivar]/sumOfWeights[cls];
            Double_t rms = TMath::Sqrt( x2[cls][(var_tgt*nvar)+ivar]/sumOfWeights[cls] - mean*mean); 
            AddStats(cls, (var_tgt*nvar)+ivar, mean, rms, varMin[cls][(var_tgt*nvar)+ivar], varMax[cls][(var_tgt*nvar)+ivar]);
         }
      }
   }

   // ------ pretty output of basic statistics -------------------------------
   // find maximum length in V (and column title)
   UInt_t maxL = 8, maxV = 0;
   std::vector<UInt_t> vLengths;
   for (UInt_t ivar=0; ivar<nvar+ntgt; ivar++) {
      if( ivar < nvar )
         maxL = TMath::Max( (UInt_t)Variable(ivar).GetLabel().Length(), maxL );
      else
         maxL = TMath::Max( (UInt_t)Target(ivar-nvar).GetLabel().Length(), maxL );
   }
   maxV = maxL + 2;
   // full column length
   UInt_t clen = maxL + 4*maxV + 11;
   for (UInt_t i=0; i<clen; i++) Log() << "-";
   Log() << Endl;
   // full column length
   Log() << std::setw(maxL) << "Variable";
   Log() << "  " << std::setw(maxV) << "Mean";
   Log() << " " << std::setw(maxV) << "RMS";
   Log() << "   " << std::setw(maxV) << "[        Min ";
   Log() << "  " << std::setw(maxV) << "    Max ]" << Endl;;
   for (UInt_t i=0; i<clen; i++) Log() << "-";
   Log() << Endl;

   // the numbers
   TString format = "%#11.5g";
   for (UInt_t ivar=0; ivar<nvar+ntgt; ivar++) {
      if( ivar < nvar )
         Log() << std::setw(maxL) << Variable(ivar).GetLabel() << ":";
      else
         Log() << std::setw(maxL) << Target(ivar-nvar).GetLabel() << ":";
      Log() << std::setw(maxV) << Form( format.Data(), GetMean(ivar) );
      Log() << std::setw(maxV) << Form( format.Data(), GetRMS(ivar) );
      Log() << "   [" << std::setw(maxV) << Form( format.Data(), GetMin(ivar) );
      Log() << std::setw(maxV) << Form( format.Data(), GetMax(ivar) ) << " ]";
      Log() << Endl;
   }
   for (UInt_t i=0; i<clen; i++) Log() << "-";
   Log() << Endl;
   // ------------------------------------------------------------------------
   
   delete[] sumOfWeights;
   for (Int_t cls=0; cls<fNumC; cls++) {
      delete [] x2[cls];
      delete [] x0[cls];
      delete [] varMin[cls];
      delete [] varMax[cls];
   }
   delete [] x2;
   delete [] x0;
   delete [] varMin;
   delete [] varMax;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::MakeFunction( std::ostream& fout, const TString& fncName, Int_t part ) const 
{
   // create transformation function
   TListIter trIt(&fTransformations);
   std::vector< Int_t >::const_iterator rClsIt = fTransformationsReferenceClasses.begin();
   UInt_t trCounter=1;
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt() ) {
      trf->MakeFunction(fout, fncName, part, trCounter++, (*rClsIt) );
      rClsIt++;
   }
   if (part==1) {
      for (Int_t i=0; i<fTransformations.GetSize(); i++) {
         fout << "   void InitTransform_"<<i+1<<"();" << std::endl;
         fout << "   void Transform_"<<i+1<<"( std::vector<double> & iv, int sigOrBgd ) const;" << std::endl;
      }
   }
   if (part==2) {
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fncName << "::InitTransform()" << std::endl;
      fout << "{" << std::endl;
      for (Int_t i=0; i<fTransformations.GetSize(); i++)
         fout << "   InitTransform_"<<i+1<<"();" << std::endl;
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fncName << "::Transform( std::vector<double>& iv, int sigOrBgd ) const" << std::endl;
      fout << "{" << std::endl;
      for (Int_t i=0; i<fTransformations.GetSize(); i++)
         fout << "   Transform_"<<i+1<<"( iv, sigOrBgd );" << std::endl;

      fout << "}" << std::endl;
   }
}

//_______________________________________________________________________
TString TMVA::TransformationHandler::GetName() const
{
   // return transformation name
   TString name("Id");
   TListIter trIt(&fTransformations);
   VariableTransformBase *trf;
   if ((trf = (VariableTransformBase*) trIt())) {
      name = TString(trf->GetShortName());
      while ((trf = (VariableTransformBase*) trIt())) name += "_" + TString(trf->GetShortName());
   }
   return name;
}

//_______________________________________________________________________
TString TMVA::TransformationHandler::GetVariableAxisTitle( const VariableInfo& info ) const
{
   // incorporates transformation type into title axis (usually for histograms)
   TString xtit = info.GetTitle();
   // indicate transformation, but not in case of single identity transform
   if (fTransformations.GetSize() >= 1) {
      if (fTransformations.GetSize() > 1 ||
          ((VariableTransformBase*)GetTransformationList().Last())->GetVariableTransform() != Types::kIdentity) {
         xtit += " (" + GetName() + ")";
      }
   }
   return xtit;
}

//_______________________________________________________________________
void TMVA::TransformationHandler::PlotVariables( const std::vector<Event*>& events, TDirectory* theDirectory )
{
   // create histograms from the input variables
   // - histograms for all input variables
   // - scatter plots for all pairs of input variables

   if (fRootBaseDir==0 && theDirectory == 0) return;

   Log() << kINFO << "Plot event variables for ";
   if (theDirectory !=0) Log()<< TString(theDirectory->GetName()) << Endl;
   else Log() << GetName() << Endl;

   // extension for transformation type
   TString transfType = "";
   if (theDirectory == 0) {
      transfType += "_";
      transfType += GetName();
   }else{ // you plot for the individual classifiers. Note, here the "statistics" still need to be calculated as you are in the testing phase
      CalcStats(events);
   }

   const UInt_t nvar = fDataSetInfo.GetNVariables();
   const UInt_t ntgt = fDataSetInfo.GetNTargets();
   const Int_t  ncls = fDataSetInfo.GetNClasses();

   // Create all histograms
   // do both, scatter and profile plots
   std::vector<std::vector<TH1*> > hVars( ncls );  // histograms for variables
   std::vector<std::vector<std::vector<TH2F*> > >     mycorr( ncls ); // histograms for correlations
   std::vector<std::vector<std::vector<TProfile*> > > myprof( ncls ); // histograms for profiles

   for (Int_t cls = 0; cls < ncls; cls++) {
      hVars.at(cls).resize ( nvar+ntgt );
      hVars.at(cls).assign ( nvar+ntgt, 0 ); // fill with zeros
      mycorr.at(cls).resize( nvar+ntgt );
      myprof.at(cls).resize( nvar+ntgt );
      for (UInt_t ivar=0; ivar < nvar+ntgt; ivar++) {
         mycorr.at(cls).at(ivar).resize( nvar+ntgt );
         myprof.at(cls).at(ivar).resize( nvar+ntgt );
         mycorr.at(cls).at(ivar).assign( nvar+ntgt, 0 ); // fill with zeros
         myprof.at(cls).at(ivar).assign( nvar+ntgt, 0 ); // fill with zeros
      }
   }

   // if there are too many input variables, the creation of correlations plots blows up
   // memory and basically kills the TMVA execution
   // --> avoid above critical number (which can be user defined)
   if (nvar+ntgt > (UInt_t)gConfig().GetVariablePlotting().fMaxNumOfAllowedVariablesForScatterPlots) {
      Int_t nhists = (nvar+ntgt)*(nvar+ntgt - 1)/2;
      Log() << kINFO << gTools().Color("dgreen") << Endl;
      Log() << kINFO << "<PlotVariables> Will not produce scatter plots ==> " << Endl;
      Log() << kINFO
            << "|  The number of " << nvar << " input variables and " << ntgt << " target values would require " 
            << nhists << " two-dimensional" << Endl;
      Log() << kINFO
            << "|  histograms, which would occupy the computer's memory. Note that this" << Endl;
      Log() << kINFO
            << "|  suppression does not have any consequences for your analysis, other" << Endl;
      Log() << kINFO
            << "|  than not disposing of these scatter plots. You can modify the maximum" << Endl;
      Log() << kINFO
            << "|  number of input variables allowed to generate scatter plots in your" << Endl; 
      Log() << "|  script via the command line:" << Endl;
      Log() << kINFO
            << "|  \"(TMVA::gConfig().GetVariablePlotting()).fMaxNumOfAllowedVariablesForScatterPlots = <some int>;\""
            << gTools().Color("reset") << Endl;
      Log() << Endl;
      Log() << kINFO << "Some more output" << Endl;
   }

   Double_t timesRMS = gConfig().GetVariablePlotting().fTimesRMS;
   UInt_t   nbins1D  = gConfig().GetVariablePlotting().fNbins1D;
   UInt_t   nbins2D  = gConfig().GetVariablePlotting().fNbins2D;

   for (UInt_t var_tgt = 0; var_tgt < 2; var_tgt++) { // create the histos first for the variables, then for the targets
      UInt_t nloops = ( var_tgt == 0? nvar:ntgt );     // number of variables or number of targets
      for (UInt_t ivar=0; ivar<nloops; ivar++) {
         const VariableInfo& info = ( var_tgt == 0 ? Variable( ivar ) : Target(ivar) ); // choose the appropriate one (variable or target)
         TString myVari = info.GetInternalName();  

         Double_t mean = fVariableStats.at(fNumC-1).at( ( var_tgt*nvar )+ivar).fMean;
         Double_t rms  = fVariableStats.at(fNumC-1).at( ( var_tgt*nvar )+ivar).fRMS;

         for (Int_t cls = 0; cls < ncls; cls++) {

            TString className = fDataSetInfo.GetClassInfo(cls)->GetName();

            // add "target" in case of target variable (required for plotting macros)
            className += (ntgt == 1 && var_tgt == 1 ? "_target" : ""); 

            // choose reasonable histogram ranges, by removing outliers
            TH1* h = 0;
            if (info.GetVarType() == 'I') {
               // special treatment for integer variables
               Int_t xmin = TMath::Nint( GetMin( ( var_tgt*nvar )+ivar) );
               Int_t xmax = TMath::Nint( GetMax( ( var_tgt*nvar )+ivar) + 1 );
               Int_t nbins = xmax - xmin;

               h = new TH1F( Form("%s__%s%s", myVari.Data(), className.Data(), transfType.Data()), 
                             info.GetTitle(), nbins, xmin, xmax );
            }
            else {
               Double_t xmin = TMath::Max( GetMin( ( var_tgt*nvar )+ivar), mean - timesRMS*rms );
               Double_t xmax = TMath::Min( GetMax( ( var_tgt*nvar )+ivar), mean + timesRMS*rms );
      
               //std::cout << "Class="<<cls<<" xmin="<<xmin << " xmax="<<xmax<<" mean="<<mean<<" rms="<<rms<<" timesRMS="<<timesRMS<<std::endl;
               // protection
               if (xmin >= xmax) xmax = xmin*1.1; // try first...
               if (xmin >= xmax) xmax = xmin + 1; // this if xmin == xmax == 0
               // safety margin for values equal to the maximum within the histogram
               xmax += (xmax - xmin)/nbins1D;

               h = new TH1F( Form("%s__%s%s", myVari.Data(), className.Data(), transfType.Data()), 
                             info.GetTitle(), nbins1D, xmin, xmax );
            }
            
            h->GetXaxis()->SetTitle( gTools().GetXTitleWithUnit( GetVariableAxisTitle( info ), info.GetUnit() ) );
            h->GetYaxis()->SetTitle( gTools().GetYTitleWithUnit( *h, info.GetUnit(), kFALSE ) );
            hVars.at(cls).at((var_tgt*nvar)+ivar) = h;
   
            // profile and scatter plots
            if (nvar+ntgt <= (UInt_t)gConfig().GetVariablePlotting().fMaxNumOfAllowedVariablesForScatterPlots) {

               for (UInt_t v_t = 0; v_t < 2; v_t++) {
                  UInt_t nl = ( v_t==0?nvar:ntgt );
                  UInt_t start = ( v_t==0? (var_tgt==0?ivar+1:0):(var_tgt==0?nl:ivar+1) );
                  for (UInt_t j=start; j<nl; j++) {
                     // choose the appropriate one (variable or target)
                     const VariableInfo& infoj = ( v_t == 0 ? Variable( j ) : Target(j) ); 
                     TString myVarj = infoj.GetInternalName();  

                     Double_t rxmin = fVariableStats.at(fNumC-1).at( ( v_t*nvar )+ivar).fMin;
                     Double_t rxmax = fVariableStats.at(fNumC-1).at( ( v_t*nvar )+ivar).fMax;
                     Double_t rymin = fVariableStats.at(fNumC-1).at( ( v_t*nvar )+j).fMin;
                     Double_t rymax = fVariableStats.at(fNumC-1).at( ( v_t*nvar )+j).fMax;
                     
                     // scatter plot
                     TH2F* h2 = new TH2F( Form( "scat_%s_vs_%s_%s%s" , myVarj.Data(), myVari.Data(), 
                                                className.Data(), transfType.Data() ), 
                                          Form( "%s versus %s (%s)%s", infoj.GetTitle().Data(), info.GetTitle().Data(), 
                                                className.Data(), transfType.Data() ), 
                                          nbins2D, rxmin , rxmax, 
                                          nbins2D, rymin , rymax );

                     h2->GetXaxis()->SetTitle( gTools().GetXTitleWithUnit( GetVariableAxisTitle( info  ), info .GetUnit() ) );
                     h2->GetYaxis()->SetTitle( gTools().GetXTitleWithUnit( GetVariableAxisTitle( infoj ), infoj.GetUnit() ) );
                     mycorr.at(cls).at((var_tgt*nvar)+ivar).at((v_t*nvar)+j) = h2;
                     
                     // profile plot
                     TProfile* p = new TProfile( Form( "prof_%s_vs_%s_%s%s", myVarj.Data(), 
                                                       myVari.Data(), className.Data(), 
                                                       transfType.Data() ), 
                                                 Form( "profile %s versus %s (%s)%s", 
                                                       infoj.GetTitle().Data(), info.GetTitle().Data(), 
                                                       className.Data(), transfType.Data() ), nbins1D, 
                                                 rxmin, rxmax );
                     //                                                 info.GetMin(), info.GetMax() );

                     p->GetXaxis()->SetTitle( gTools().GetXTitleWithUnit( GetVariableAxisTitle( info  ), info .GetUnit() ) );
                     p->GetYaxis()->SetTitle( gTools().GetXTitleWithUnit( GetVariableAxisTitle( infoj ), infoj.GetUnit() ) );
                     myprof.at(cls).at((var_tgt*nvar)+ivar).at((v_t*nvar)+j) = p;
                  }
               }
            }   
         }
      }
   }

   UInt_t nevts = events.size();

   // compute correlation coefficient between target value and variables (regression only)
   std::vector<Double_t> xregmean ( nvar+1, 0 );
   std::vector<Double_t> x2regmean( nvar+1, 0 );
   std::vector<Double_t> xCregmean( nvar+1, 0 );

   // fill the histograms (this approach should be faster than individual projection
   for (UInt_t ievt=0; ievt<nevts; ievt++) {

      const Event* ev = events[ievt];

      Float_t weight = ev->GetWeight();
      Int_t   cls    = ev->GetClass();

      // average correlation between first target and variables (so far only for single-target regression)
      if (ntgt == 1) {
         Float_t valr = ev->GetTarget(0);
         xregmean[nvar]  += valr;
         x2regmean[nvar] += valr*valr;
         for (UInt_t ivar=0; ivar<nvar; ivar++) {
            Float_t vali = ev->GetValue(ivar);
            xregmean[ivar]  += vali;
            x2regmean[ivar] += vali*vali;
            xCregmean[ivar] += vali*valr;
         }
      }
      
      // fill correlation histograms
      for (UInt_t var_tgt = 0; var_tgt < 2; var_tgt++) { // create the histos first for the variables, then for the targets
         UInt_t nloops = ( var_tgt == 0? nvar:ntgt );    // number of variables or number of targets
         for (UInt_t ivar=0; ivar<nloops; ivar++) {
            Float_t vali = ( var_tgt == 0 ? ev->GetValue(ivar) : ev->GetTarget(ivar) );

            // variable histos
            hVars.at(cls).at( ( var_tgt*nvar )+ivar)->Fill( vali, weight );

            // correlation histos
            if (nvar+ntgt <= (UInt_t)gConfig().GetVariablePlotting().fMaxNumOfAllowedVariablesForScatterPlots) {

               for (UInt_t v_t = 0; v_t < 2; v_t++) {
                  UInt_t nl    = ( v_t==0 ? nvar : ntgt );
                  UInt_t start = ( v_t==0 ? (var_tgt==0?ivar+1:0) : (var_tgt==0?nl:ivar+1) );
                  for (UInt_t j=start; j<nl; j++) {
                     Float_t valj = ( v_t == 0 ? ev->GetValue(j) : ev->GetTarget(j) );
                     mycorr.at(cls).at( ( var_tgt*nvar )+ivar).at( ( v_t*nvar )+j)->Fill( vali, valj, weight );
                     myprof.at(cls).at( ( var_tgt*nvar )+ivar).at( ( v_t*nvar )+j)->Fill( vali, valj, weight );
                  }
               }
            }
         }
      }
   }
      
   // correlation analysis for ranking  (single-target regression only)
   if (ntgt == 1) {
      for (UInt_t ivar=0; ivar<=nvar; ivar++) {
         xregmean[ivar] /= nevts;
         x2regmean[ivar] = x2regmean[ivar]/nevts - xregmean[ivar]*xregmean[ivar];
      }
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         xCregmean[ivar] = xCregmean[ivar]/nevts - xregmean[ivar]*xregmean[nvar];
         xCregmean[ivar] /= TMath::Sqrt( x2regmean[ivar]*x2regmean[nvar] );
      }         
      
      fRanking.push_back( new Ranking( GetName() + "Transformation", "|Correlation with target|" ) );
      for (UInt_t ivar=0; ivar<nvar; ivar++) {   
         Double_t abscor = TMath::Abs( xCregmean[ivar] );
         fRanking.back()->AddRank( Rank( fDataSetInfo.GetVariableInfo(ivar).GetLabel(), abscor ) );
      }

      if (nvar+ntgt <= (UInt_t)gConfig().GetVariablePlotting().fMaxNumOfAllowedVariablesForScatterPlots) {
      
         // compute also mutual information (non-linear correlation measure)
         fRanking.push_back( new Ranking( GetName() + "Transformation", "Mutual information" ) );
         for (UInt_t ivar=0; ivar<nvar; ivar++) {   
            TH2F* h1 = mycorr.at(0).at( nvar ).at( ivar );
            Double_t mi = gTools().GetMutualInformation( *h1 );
            fRanking.back()->AddRank( Rank( fDataSetInfo.GetVariableInfo(ivar).GetLabel(), mi ) );
         }     
         
         // compute correlation ratio (functional correlations measure)
         fRanking.push_back( new Ranking( GetName() + "Transformation", "Correlation Ratio" ) );
         for (UInt_t ivar=0; ivar<nvar; ivar++) {   
            TH2F*    h2 = mycorr.at(0).at( nvar ).at( ivar );
            Double_t cr = gTools().GetCorrelationRatio( *h2 );
            fRanking.back()->AddRank( Rank( fDataSetInfo.GetVariableInfo(ivar).GetLabel(), cr ) );
         } 
         
         // additionally compute correlation ratio from transposed histograms since correlation ratio is asymmetric
         fRanking.push_back( new Ranking( GetName() + "Transformation", "Correlation Ratio (T)" ) );
         for (UInt_t ivar=0; ivar<nvar; ivar++) {   
            TH2F*    h2T = gTools().TransposeHist( *mycorr.at(0).at( nvar ).at( ivar ) );
            Double_t cr  = gTools().GetCorrelationRatio( *h2T  );
            fRanking.back()->AddRank( Rank( fDataSetInfo.GetVariableInfo(ivar).GetLabel(), cr ) );
            delete h2T;
         }      
      }
   }
   // computes ranking of input variables
   // separation for 2-class classification
   else if (fDataSetInfo.GetNClasses() == 2 
            && fDataSetInfo.GetClassInfo("Signal") != NULL 
            && fDataSetInfo.GetClassInfo("Background") != NULL 
      ) { // TODO: ugly hack.. adapt to new framework
      fRanking.push_back( new Ranking( GetName() + "Transformation", "Separation" ) );
      for (UInt_t i=0; i<nvar; i++) {   
         Double_t sep = gTools().GetSeparation( hVars.at(fDataSetInfo.GetClassInfo("Signal")    ->GetNumber()).at(i), 
                                                hVars.at(fDataSetInfo.GetClassInfo("Background")->GetNumber()).at(i) );
         fRanking.back()->AddRank( Rank( hVars.at(fDataSetInfo.GetClassInfo("Signal")->GetNumber()).at(i)->GetTitle(), 
                                         sep ) );
      }
   }

   // for regression compute performance from correlation with target value

   // write histograms

   TDirectory* localDir = theDirectory;
   if (theDirectory == 0) {
      // create directory in root dir
      fRootBaseDir->cd();
      TString outputDir = TString("InputVariables");
      TListIter trIt(&fTransformations);
      while (VariableTransformBase *trf = (VariableTransformBase*) trIt())
         outputDir += "_" + TString(trf->GetShortName());

      TString uniqueOutputDir = outputDir;
      Int_t counter = 0;
      TObject* o = NULL;
      while( (o = fRootBaseDir->FindObject(uniqueOutputDir)) != 0 ){
	 uniqueOutputDir = outputDir+Form("_%d",counter);
         Log() << kINFO << "A " << o->ClassName() << " with name " << o->GetName() << " already exists in " 
               << fRootBaseDir->GetPath() << ", I will try with "<<uniqueOutputDir<<"." << Endl;
	 ++counter;
      }

//       TObject* o = fRootBaseDir->FindObject(outputDir);
//       if (o != 0) {
//          Log() << kFATAL << "A " << o->ClassName() << " with name " << o->GetName() << " already exists in " 
//                << fRootBaseDir->GetPath() << "("<<outputDir<<")" << Endl;
//       }
      localDir = fRootBaseDir->mkdir( uniqueOutputDir );
      localDir->cd();
   
      Log() << kVERBOSE << "Create and switch to directory " << localDir->GetPath() << Endl;
   }
   else {
      theDirectory->cd();
   }

   for (UInt_t i=0; i<nvar+ntgt; i++) {
      for (Int_t cls = 0; cls < ncls; cls++) {
         if (hVars.at(cls).at(i) != 0) {
            hVars.at(cls).at(i)->Write();
            hVars.at(cls).at(i)->SetDirectory(0);
            delete hVars.at(cls).at(i);
         }
      }
   }

   // correlation plots have dedicated directory
   if (nvar+ntgt <= (UInt_t)gConfig().GetVariablePlotting().fMaxNumOfAllowedVariablesForScatterPlots) {

      localDir = localDir->mkdir( "CorrelationPlots" );
      localDir ->cd();
      Log() << kINFO << "Create scatter and profile plots in target-file directory: " << Endl;
      Log() << kINFO << localDir->GetPath() << Endl;
   
      
      for (UInt_t i=0; i<nvar+ntgt; i++) {
         for (UInt_t j=i+1; j<nvar+ntgt; j++) {
            for (Int_t cls = 0; cls < ncls; cls++) {
               if (mycorr.at(cls).at(i).at(j) != 0 ) {
                  mycorr.at(cls).at(i).at(j)->Write();
                  mycorr.at(cls).at(i).at(j)->SetDirectory(0);
                  delete mycorr.at(cls).at(i).at(j);
               }
               if (myprof.at(cls).at(i).at(j) != 0) {
                  myprof.at(cls).at(i).at(j)->Write();
                  myprof.at(cls).at(i).at(j)->SetDirectory(0);
                  delete myprof.at(cls).at(i).at(j);
               }
            }
         }
      }
   }
   if (theDirectory != 0 ) theDirectory->cd();
   else                    fRootBaseDir->cd();
}

//_______________________________________________________________________
std::vector<TString>* TMVA::TransformationHandler::GetTransformationStringsOfLastTransform() const
{
   // returns string for transformation
   VariableTransformBase* trf = ((VariableTransformBase*)GetTransformationList().Last());
   if (!trf) return 0;
   else      return trf->GetTransformationStrings( fTransformationsReferenceClasses.back() );
}

//_______________________________________________________________________
const char* TMVA::TransformationHandler::GetNameOfLastTransform() const
{
   // returns string for transformation
   VariableTransformBase* trf = ((VariableTransformBase*)GetTransformationList().Last());
   if (!trf) return 0;
   else      return trf->GetName();
}

//_______________________________________________________________________
void TMVA::TransformationHandler::WriteToStream( std::ostream& o ) const 
{
   // write transformatino to stream
   TListIter trIt(&fTransformations);
   std::vector< Int_t >::const_iterator rClsIt = fTransformationsReferenceClasses.begin();

   o << "NTransformtations " << fTransformations.GetSize() << std::endl << std::endl;

   ClassInfo* ci;
   UInt_t i = 1;
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt()) {
      o << "#TR -*-*-*-*-*-*-* transformation " << i++ << ": " << trf->GetName() << " -*-*-*-*-*-*-*-" << std::endl;
      trf->WriteTransformationToStream(o);
      ci = fDataSetInfo.GetClassInfo( (*rClsIt) );
      TString clsName;
      if (ci == 0 ) clsName = "AllClasses";
      else clsName = ci->GetName();
      o << "ReferenceClass " << clsName << std::endl; 
      rClsIt++;
   }
}


//_______________________________________________________________________
void TMVA::TransformationHandler::AddXMLTo( void* parent ) const 
{
   // XML node describing the transformation
   //   return;
   if(!parent) return;
   void* trfs = gTools().AddChild(parent, "Transformations");
   gTools().AddAttr( trfs, "NTransformations", fTransformations.GetSize() );
   TListIter trIt(&fTransformations);
   while (VariableTransformBase *trf = (VariableTransformBase*) trIt()) trf->AttachXMLTo(trfs);
}

//_______________________________________________________________________
void TMVA::TransformationHandler::ReadFromStream( std::istream& ) 
{
   //VariableTransformBase* trf = ((VariableTransformBase*)GetTransformationList().Last());
   //trf->ReadTransformationFromStream(fin);
   Log() << kFATAL << "Read transformations not implemented" << Endl;
   // TODO
}

//_______________________________________________________________________
void TMVA::TransformationHandler::ReadFromXML( void* trfsnode )
{
   void* ch = gTools().GetChild( trfsnode );
   while(ch) {
      Int_t idxCls = -1;
      TString trfname;
      gTools().ReadAttr(ch, "Name", trfname);

      VariableTransformBase* newtrf = 0;

      if (trfname == "Decorrelation" ) {
         newtrf = new VariableDecorrTransform(fDataSetInfo);
      }
      else if (trfname == "PCA" ) {
         newtrf = new VariablePCATransform(fDataSetInfo);
      }
      else if (trfname == "Gauss" ) {
         newtrf = new VariableGaussTransform(fDataSetInfo);
      }
      else if (trfname == "Uniform" ) {
         newtrf = new VariableGaussTransform(fDataSetInfo, "Uniform");
      }
      else if (trfname == "Normalize" ) {
         newtrf = new VariableNormalizeTransform(fDataSetInfo);
      }
      else if (trfname == "Rearrange" ) {
         newtrf = new VariableRearrangeTransform(fDataSetInfo);
      } 
      else if (trfname != "None") {
      }
      else {
         Log() << kFATAL << "<ReadFromXML> Variable transform '"
               << trfname << "' unknown." << Endl;
      }
      newtrf->ReadFromXML( ch );
      AddTransformation( newtrf, idxCls );
      ch = gTools().GetNextChild(ch);
   }
}

//_______________________________________________________________________
void TMVA::TransformationHandler::PrintVariableRanking() const
{
   // prints ranking of input variables
   Log() << kINFO << " " << Endl;
   Log() << kINFO << "Ranking input variables (method unspecific)..." << Endl;
   std::vector<Ranking*>::const_iterator it = fRanking.begin();
   for (; it != fRanking.end(); it++) (*it)->Print();
}

//_______________________________________________________________________
Double_t TMVA::TransformationHandler::GetMean( Int_t ivar, Int_t cls ) const
{
   try {
      return fVariableStats.at(cls).at(ivar).fMean;
   }
   catch(...) {
      try {
         return fVariableStats.at(fNumC-1).at(ivar).fMean;
      }
      catch(...) {
         Log() << kWARNING << "Inconsistent variable state when reading the mean value. " << Endl;
      }
   }
   Log() << kWARNING << "Inconsistent variable state when reading the mean value. Value 0 given back" << Endl;
   return 0;
}


//_______________________________________________________________________
Double_t TMVA::TransformationHandler::GetRMS( Int_t ivar, Int_t cls ) const
{
   try {
      return fVariableStats.at(cls).at(ivar).fRMS;
   }
   catch(...) {
      try {
         return fVariableStats.at(fNumC-1).at(ivar).fRMS;
      }
      catch(...) {
         Log() << kWARNING << "Inconsistent variable state when reading the RMS value. " << Endl;
      }
   }
   Log() << kWARNING << "Inconsistent variable state when reading the RMS value. Value 0 given back" << Endl;
   return 0;
}

//_______________________________________________________________________
Double_t TMVA::TransformationHandler::GetMin( Int_t ivar, Int_t cls ) const
{
   try {
      return fVariableStats.at(cls).at(ivar).fMin;
   }
   catch(...) {
      try {
         return fVariableStats.at(fNumC-1).at(ivar).fMin;
      }
      catch(...) {
         Log() << kWARNING << "Inconsistent variable state when reading the minimum value. " << Endl;
      }
   }
   Log() << kWARNING << "Inconsistent variable state when reading the minimum value. Value 0 given back" << Endl;
   return 0;
}

//_______________________________________________________________________
Double_t TMVA::TransformationHandler::GetMax( Int_t ivar, Int_t cls ) const
{
   try {
      return fVariableStats.at(cls).at(ivar).fMax;
   }
   catch(...) {
      try {
         return fVariableStats.at(fNumC-1).at(ivar).fMax;
      }
      catch(...) {
         Log() << kWARNING << "Inconsistent variable state when reading the maximum value. " << Endl;
      }
   }
   Log() << kWARNING << "Inconsistent variable state when reading the maximum value. Value 0 given back" << Endl;
   return 0;
}
