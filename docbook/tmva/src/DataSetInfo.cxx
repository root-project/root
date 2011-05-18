// @(#)root/tmva $Id$
// Author: Joerg Stelzer, Peter Speckmeier

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetInfo                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - DESY, Germany                  *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      DESY Hamburg, Germany                                                     *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <vector>

#include "TEventList.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom3.h"
#include "TMatrixF.h"
#include "TVectorF.h"
#include "TMath.h"
#include "TROOT.h"
#include "TObjString.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_DataSetInfo
#include "TMVA/DataSetInfo.h"
#endif
#ifndef ROOT_TMVA_DataSetManager
#include "TMVA/DataSetManager.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

//_______________________________________________________________________
TMVA::DataSetInfo::DataSetInfo(const TString& name) 
   : TObject(),
     fDataSetManager(NULL),
     fName(name),
     fDataSet( 0 ),
     fNeedsRebuilding( kTRUE ),
     fVariables(),
     fTargets(),
     fSpectators(),
     fClasses( 0 ),
     fNormalization( "NONE" ),
     fSplitOptions(""),
     fOwnRootDir(0),
     fVerbose( kFALSE ),
     fSignalClass(0),
     fTargetsForMulticlass(0),
     fLogger( new MsgLogger("DataSetInfo", kINFO) )
{
   // constructor

}

//_______________________________________________________________________
TMVA::DataSetInfo::~DataSetInfo() 
{
   // destructor
   ClearDataSet();
   
   for(UInt_t i=0, iEnd = fClasses.size(); i<iEnd; ++i) {
      delete fClasses[i];
   }

   delete fTargetsForMulticlass;

   delete fLogger;
}

//_______________________________________________________________________
void TMVA::DataSetInfo::ClearDataSet() const 
{
   if(fDataSet!=0) { delete fDataSet; fDataSet=0; }
}

//_______________________________________________________________________
TMVA::ClassInfo* TMVA::DataSetInfo::AddClass( const TString& className ) 
{

   ClassInfo* theClass = GetClassInfo(className);
   if (theClass) return theClass;

   fClasses.push_back( new ClassInfo(className) );
   fClasses.back()->SetNumber(fClasses.size()-1);

   Log() << kINFO << "Added class \"" << className << "\"\t with internal class number " 
         << fClasses.back()->GetNumber() << Endl;

   if (className == "Signal") fSignalClass = fClasses.size()-1;  // store the signal class index ( for comparison reasons )

   return fClasses.back();
}

//_______________________________________________________________________
void TMVA::DataSetInfo::SetMsgType( EMsgType t ) const 
{  
    fLogger->SetMinType(t);  
} 

//_______________________________________________________________________
TMVA::ClassInfo* TMVA::DataSetInfo::GetClassInfo( const TString& name ) const 
{
   for (std::vector<ClassInfo*>::iterator it = fClasses.begin(); it < fClasses.end(); it++) {
      if ((*it)->GetName() == name) return (*it);
   }
   return 0;
}

//_______________________________________________________________________
TMVA::ClassInfo* TMVA::DataSetInfo::GetClassInfo( Int_t cls ) const 
{
   try {
      return fClasses.at(cls);
   }
   catch(...) {
      return 0;
   }
}

//_______________________________________________________________________
void TMVA::DataSetInfo::PrintClasses() const 
{
   for (UInt_t cls = 0; cls < GetNClasses() ; cls++) {
      Log() << kINFO << "Class index : " << cls << "  name : " << GetClassInfo(cls)->GetName() << Endl;
   }
}

//_______________________________________________________________________
Bool_t TMVA::DataSetInfo::IsSignal( const TMVA::Event* ev ) const 
{
   return (ev->GetClass()  == fSignalClass); 
}

//_______________________________________________________________________
std::vector<Float_t>*  TMVA::DataSetInfo::GetTargetsForMulticlass( const TMVA::Event* ev ) 
{
   if( !fTargetsForMulticlass ) fTargetsForMulticlass = new std::vector<Float_t>( GetNClasses() );
//   fTargetsForMulticlass->resize( GetNClasses() );
   fTargetsForMulticlass->assign( GetNClasses(), 0.0 );
   fTargetsForMulticlass->at( ev->GetClass() ) = 1.0;
   return fTargetsForMulticlass; 
}


//_______________________________________________________________________
Bool_t TMVA::DataSetInfo::HasCuts() const 
{
   Bool_t hasCuts = kFALSE;
   for (std::vector<ClassInfo*>::iterator it = fClasses.begin(); it < fClasses.end(); it++) {
      if( TString((*it)->GetCut()) != TString("") ) hasCuts = kTRUE;
   }
   return hasCuts;
}

//_______________________________________________________________________
const TMatrixD* TMVA::DataSetInfo::CorrelationMatrix( const TString& className ) const 
{ 
   ClassInfo* ptr = GetClassInfo(className);
   return ptr?ptr->GetCorrelationMatrix():0;
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddVariable( const TString& expression, const TString& title, const TString& unit, 
                                                    Double_t min, Double_t max, char varType,
                                                    Bool_t normalized, void* external )
{
   // add a variable (can be a complex expression) to the set of variables used in
   // the MV analysis
   TString regexpr = expression; // remove possible blanks
   regexpr.ReplaceAll(" ", "" );
   fVariables.push_back(VariableInfo( regexpr, title, unit, 
                                      fVariables.size()+1, varType, external, min, max, normalized ));
   fNeedsRebuilding = kTRUE;
   return fVariables.back();
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddVariable( const VariableInfo& varInfo){
   // add variable with given VariableInfo
   fVariables.push_back(VariableInfo( varInfo ));
   fNeedsRebuilding = kTRUE;
   return fVariables.back();
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddTarget( const TString& expression, const TString& title, const TString& unit, 
                                                  Double_t min, Double_t max, 
                                                  Bool_t normalized, void* external )
{
   // add a variable (can be a complex expression) to the set of variables used in
   // the MV analysis
   TString regexpr = expression; // remove possible blanks
   regexpr.ReplaceAll(" ", "" );
   char type='F';
   fTargets.push_back(VariableInfo( regexpr, title, unit, 
                                    fTargets.size()+1, type, external, min, max, normalized ));
   fNeedsRebuilding = kTRUE;
   return fTargets.back();
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddTarget( const VariableInfo& varInfo){
   // add target with given VariableInfo
   fTargets.push_back(VariableInfo( varInfo ));
   fNeedsRebuilding = kTRUE;
   return fTargets.back();
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddSpectator( const TString& expression, const TString& title, const TString& unit, 
                                                     Double_t min, Double_t max, char type,
                                                     Bool_t normalized, void* external )
{
   // add a spectator (can be a complex expression) to the set of spectator variables used in
   // the MV analysis
   TString regexpr = expression; // remove possible blanks
   regexpr.ReplaceAll(" ", "" );
   fSpectators.push_back(VariableInfo( regexpr, title, unit, 
                                       fSpectators.size()+1, type, external, min, max, normalized ));
   fNeedsRebuilding = kTRUE;
   return fSpectators.back();
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::DataSetInfo::AddSpectator( const VariableInfo& varInfo){
   // add spectator with given VariableInfo
   fSpectators.push_back(VariableInfo( varInfo ));
   fNeedsRebuilding = kTRUE;
   return fSpectators.back();
}

//_______________________________________________________________________
Int_t TMVA::DataSetInfo::FindVarIndex(const TString& var) const
{
   // find variable by name
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
      if (var == GetVariableInfo(ivar).GetInternalName()) return ivar;
   
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
      Log() << kINFO  <<  GetVariableInfo(ivar).GetInternalName() << Endl;
   
   Log() << kFATAL << "<FindVarIndex> Variable \'" << var << "\' not found." << Endl;
 
   return -1;
}

//_______________________________________________________________________
void TMVA::DataSetInfo::SetWeightExpression( const TString& expr, const TString& className ) 
{
   // set the weight expressions for the classes
   // if class name is specified, set only for this class
   // if class name is unknown, register new class with this name

   if (className != "") {
      TMVA::ClassInfo* ci = AddClass(className);
      ci->SetWeight( expr );
   } 
   else {
      // no class name specified, set weight for all classes
      if (fClasses.size()==0) {
         Log() << kWARNING << "No classes registered yet, cannot specify weight expression!" << Endl;
      }
      for (std::vector<ClassInfo*>::iterator it = fClasses.begin(); it < fClasses.end(); it++) {
         (*it)->SetWeight( expr );
      }
   }
}

//_______________________________________________________________________
void TMVA::DataSetInfo::SetCorrelationMatrix( const TString& className, TMatrixD* matrix ) 
{
   GetClassInfo(className)->SetCorrelationMatrix(matrix); 
}

//_______________________________________________________________________
void TMVA::DataSetInfo::SetCut( const TCut& cut, const TString& className ) 
{
   // set the cut for the classes
   if (className == "") {  // if no className has been given set the cut for all the classes
      for (std::vector<ClassInfo*>::iterator it = fClasses.begin(); it < fClasses.end(); it++) {
         (*it)->SetCut( cut );
      }
   }
   else {
      TMVA::ClassInfo* ci = AddClass(className);
      ci->SetCut( cut );
   }
}

//_______________________________________________________________________
void TMVA::DataSetInfo::AddCut( const TCut& cut, const TString& className ) 
{
   // set the cut for the classes
   if (className == "") {  // if no className has been given set the cut for all the classes
      for (std::vector<ClassInfo*>::iterator it = fClasses.begin(); it < fClasses.end(); it++) {
         const TCut& oldCut = (*it)->GetCut(); 
         (*it)->SetCut( oldCut+cut );
      }
   }
   else {
      TMVA::ClassInfo* ci = AddClass(className);
      ci->SetCut( ci->GetCut()+cut );
   }
}

//_______________________________________________________________________
std::vector<TString> TMVA::DataSetInfo::GetListOfVariables() const
{
   // returns list of variables
   std::vector<TString> vNames;
   std::vector<TMVA::VariableInfo>::const_iterator viIt = GetVariableInfos().begin();
   for(;viIt != GetVariableInfos().end(); viIt++) vNames.push_back( (*viIt).GetExpression() );

   return vNames;
}

//_______________________________________________________________________
void TMVA::DataSetInfo::PrintCorrelationMatrix( const TString& className )
{ 
   // calculates the correlation matrices for signal and background, 
   // prints them to standard output, and fills 2D histograms
   Log() << kINFO << "Correlation matrix (" << className << "):" << Endl;
   gTools().FormattedOutput( *CorrelationMatrix( className ), GetListOfVariables(), Log() );
}

//_______________________________________________________________________
TH2* TMVA::DataSetInfo::CreateCorrelationMatrixHist( const TMatrixD* m,
                                                     const TString&  hName,
                                                     const TString&  hTitle ) const
{
   if (m==0) return 0;
   
   const UInt_t nvar = GetNVariables();

   // workaround till the TMatrix templates are comonly used
   // this keeps backward compatibility
   TMatrixF* tm = new TMatrixF( nvar, nvar );
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      for (UInt_t jvar=0; jvar<nvar; jvar++) {
         (*tm)(ivar, jvar) = (*m)(ivar,jvar);
      }
   }  

   TH2F* h2 = new TH2F( *tm );
   h2->SetNameTitle( hName, hTitle );

   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      h2->GetXaxis()->SetBinLabel( ivar+1, GetVariableInfo(ivar).GetTitle() );
      h2->GetYaxis()->SetBinLabel( ivar+1, GetVariableInfo(ivar).GetTitle() );
   }
   
   // present in percent, and round off digits
   // also, use absolute value of correlation coefficient (ignore sign)
   h2->Scale( 100.0  ); 
   for (UInt_t ibin=1; ibin<=nvar; ibin++) {
      for (UInt_t jbin=1; jbin<=nvar; jbin++) {
         h2->SetBinContent( ibin, jbin, Int_t(h2->GetBinContent( ibin, jbin )) );
      }
   }
   
   // style settings
   const Float_t labelSize = 0.055;
   h2->SetStats( 0 );
   h2->GetXaxis()->SetLabelSize( labelSize );
   h2->GetYaxis()->SetLabelSize( labelSize );
   h2->SetMarkerSize( 1.5 );
   h2->SetMarkerColor( 0 );
   h2->LabelsOption( "d" ); // diagonal labels on x axis
   h2->SetLabelOffset( 0.011 );// label offset on x axis
   h2->SetMinimum( -100.0 );
   h2->SetMaximum( +100.0 );

   // -------------------------------------------------------------------------------------
   // just in case one wants to change the position of the color palette axis
   // -------------------------------------------------------------------------------------
   //     gROOT->SetStyle("Plain");
   //     TStyle* gStyle = gROOT->GetStyle( "Plain" );
   //     gStyle->SetPalette( 1, 0 );
   //     TPaletteAxis* paletteAxis 
   //                   = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject( "palette" );
   // -------------------------------------------------------------------------------------
   
   Log() << kDEBUG << "Created correlation matrix as 2D histogram: " << h2->GetName() << Endl;
   
   return h2;
}

//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetInfo::GetDataSet() const 
{
   // returns data set
   if (fDataSet==0 || fNeedsRebuilding) {
      if(fDataSet!=0) ClearDataSet();
//      fDataSet = DataSetManager::Instance().CreateDataSet(GetName()); //DSMTEST replaced by following lines
      if( !fDataSetManager )
	 Log() << kFATAL << "DataSetManager has not been set in DataSetInfo (GetDataSet() )." << Endl;
      fDataSet = fDataSetManager->CreateDataSet(GetName());



      fNeedsRebuilding = kFALSE;
   }
   return fDataSet;
}

//_______________________________________________________________________
UInt_t TMVA::DataSetInfo::GetNSpectators(bool all) const
{
   if(all)
      return fSpectators.size();
   UInt_t nsp(0);
   for(std::vector<VariableInfo>::const_iterator spit=fSpectators.begin(); spit!=fSpectators.end(); ++spit) {
      if(spit->GetVarType()!='C') nsp++;
   }
   return nsp;
}

//_______________________________________________________________________
Int_t TMVA::DataSetInfo::GetClassNameMaxLength() const
{
   Int_t maxL = 0;
   for (UInt_t cl = 0; cl < GetNClasses(); cl++) {
      if (TString(GetClassInfo(cl)->GetName()).Length() > maxL) maxL = TString(GetClassInfo(cl)->GetName()).Length();
   }

   return maxL;
}

