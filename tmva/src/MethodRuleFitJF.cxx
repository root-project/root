// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodRuleFitJF                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch>  - Iowa State U., USA     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      Iowa State U.                                                             *
 *      MPI-KP Heidelberg, Germany,                                               * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// J Friedman's RuleFit method
//_______________________________________________________________________

#include "TROOT.h"
#include "TSystem.h"
#include "TMath.h"
#include "TMVA/MethodRuleFitJF.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodRuleFitJF)

TMVA::MethodRuleFitJF::MethodRuleFitJF( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   //
   InitRuleFit();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   if (!HasTrainingTree()) fLogger << kFATAL << "--- " << GetName() << "no training Tree" << Endl;
}

//_______________________________________________________________________
TMVA::MethodRuleFitJF::MethodRuleFitJF( DataSet& theData,
                                    TString theWeightFile,
                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir )
{
   // constructor from weight file
   InitRuleFit();

   DeclareOptions();
}

//_______________________________________________________________________
TMVA::MethodRuleFitJF::~MethodRuleFitJF( void )
{
   // destructor
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::DeclareOptions() 
{
   // declare the options
   DeclareOptionRef( fRFIntParms.max_rules, "max.rules",     "maximum number of rules allowed");
   DeclareOptionRef( fRFIntParms.tree_size, "tree.size",     "maximum tree size");
   DeclareOptionRef( fRFIntParms.path_speed,"path.speed",    "path search speed");
   DeclareOptionRef( fRFIntParms.path_xval, "path.xval",     "number of cross validation iterations");
   DeclareOptionRef( fRFIntParms.path_steps,"path.steps",    "number of path steps");
   DeclareOptionRef( fRFIntParms.path_testfreq,"path.testfreq", "test frequency");
   //
   DeclareOptionRef( fRFRealParms.inter_supp,"inter.supp", "incentive factor");
   //
   DeclareOptionRef(fModelType="both", "model", "model to be used");
   AddPreDefVal(TString("linear"));
   AddPreDefVal(TString("rule"));
   AddPreDefVal(TString("both"));
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::ProcessOptions() 
{
   // process the options
   MethodBase::ProcessOptions();

   if      (fModelType == "linear" ) fRFIntParms.lmode = TMVA::MethodRuleFitJF::kRfLinear;
   else if (fModelType == "rule" )   fRFIntParms.lmode = TMVA::MethodRuleFitJF::kRfRules;
   else                              fRFIntParms.lmode = TMVA::MethodRuleFitJF::kRfBoth;
}


//_______________________________________________________________________
void TMVA::MethodRuleFitJF::InitRuleFit( void )
{
   // default initialisation
   SetMethodName( "RuleFitJF" );
   SetMethodType( TMVA::Types::kRuleFitJF );
   SetTestvarName();

   SetRFWorkDir("./rulefit");
   FillIntParmsDef();
   FillRealParmsDef();
}


//_______________________________________________________________________
void TMVA::MethodRuleFitJF::Train( void )
{
   // training of rules

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "--- " << GetName() << "sanity check failed" << Endl;

   // First train the model
   TMVA::Timer timer( 1, GetName() );

   TrainRuleFit();

   fLogger << kINFO << "train elapsed time: " << timer.GetElapsedTime() << Endl;

   // And do the testing here
   // Result will be stored in an array.
   // When TMVA does the testing it will read the score from the table
   fLogger << kINFO << "--- " << GetName() << " Testing RuleFit..." << Endl;
   TestRuleFit();

   fLogger << kINFO << "--- " << GetName() << " Obtaining variable importances..." << Endl;
   VarImp();
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodRuleFitJF::CreateRanking() 
{
   // computes ranking of input variables
   if (fRFVarImp.size()==0) return 0;

   // create the ranking object
   fRanking = new TMVA::Ranking( GetName(), "Relative variable importance" );

   Int_t indvar;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      indvar = fRFVarImpInd[ivar];
      fRanking->AddRank( *new TMVA::Rank( GetInputExp(indvar), fRFVarImp[ivar] ) );
   }

   return fRanking;
}


//_______________________________________________________________________
void  TMVA::MethodRuleFitJF::WriteWeightsToStream( ostream & ) const
{
   // write model to file
}

//_______________________________________________________________________
void  TMVA::MethodRuleFitJF::ReadWeightsFromStream( istream & )
{
   // read rules from stream
}

//_______________________________________________________________________
Double_t TMVA::MethodRuleFitJF::GetMvaValue()
{
   // returns MVA value for given event
   const UInt_t idx = Data().GetCurrentEvtIdx();
   if (idx<fRFYhat.size()) {
      return fRFYhat[idx];
   } else {
      if ((idx<10) || (idx>=(fRFYhat.size()))) {
         fLogger << kWARNING << "Score function does not match current event index: " << idx << Endl;
         fLogger << kWARNING << "Score size = " << fRFYhat.size() << Endl;
      }
      return 0.0;
   }
}

//
//************ HERE STARTS THE REAL MESS ****************
//

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::SetRFWorkDir(const char * wdir)
{
   // set the directory containing rf_go.exe.
   // it fails (kFATAL) if it does not exist.
   fRFWorkDir = wdir;
   CheckRFWorkDir();
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::CheckRFWorkDir()
{
   // check if the rulefit work dir is properly setup.
   // it aborts (kFATAL) if not.
   //
   // Check existance of directory
   TString oldDir = gSystem->pwd();
   if (!gSystem->cd(fRFWorkDir)) {
      fLogger << kFATAL
              << "Must create a directory named : " << fRFWorkDir << "\n"
              << "In this directory you must put a link to the RuleFit executable (rf_go.exe)." << Endl;
   }
   // check rf_go.exe
   FILE *f = fopen("rf_go.exe","r");
   if (f==0) {
      fLogger << kFATAL
              << "RuleFit workdir (" << fRFWorkDir << ") exists "
              << "but there is no link to the RuleFit executable (rf_go.exe)." << Endl;
   }
   fclose(f);
   gSystem->cd(oldDir.Data());
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::SetTrainParms()
{
   // set the training parameters
   Int_t    n    = Data().GetNEvtTrain();
   Double_t neff = Double_t(n); // When weights are added: should be sum(wt)^2/sum(wt^2)
   fRFIntParms.n = n; // number of data points in tree
   fRFIntParms.p = Data().GetNVariables();
   fRFRealParms.samp_fract  = TMath::Min(0.5,(100.0+6.0*TMath::Sqrt(neff))/neff);
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::SetTestParms()
{
   // set the test params
   Int_t    n    = Data().GetNEvtTest();
   Double_t neff = Double_t(n); // When weights are added: should be sum(wt)^2/sum(wt^2)
   fRFIntParms.n = n; // number of data points in tree
   fRFIntParms.p = Data().GetNVariables();
   fRFRealParms.samp_fract  = TMath::Min(0.5,(100.0+6.0*TMath::Sqrt(neff))/neff);
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::FillRealParmsDef()
{
   // set default real params
   fRFRealParms.xmiss       = 9.0e30;
   fRFRealParms.trim_qntl   = 0.025;
   fRFRealParms.huber       = 0.8;
   fRFRealParms.inter_supp  = 3.0;
   fRFRealParms.memory_par  = 0.01;
   fRFRealParms.samp_fract  = 0.5; // calculated later
   fRFRealParms.path_inc    = 0.01;
   fRFRealParms.conv_fac    = 1.1;
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::FillIntParmsDef()
{
   // set default int params
   fRFIntParms.mode           = (int)kRfClass;
   fRFIntParms.lmode          = (int)kRfBoth;
   //   fRFIntParms.n;
   //   fRFIntParms.p;
   fRFIntParms.max_rules      = 2000;
   fRFIntParms.tree_size      = 4;
   fRFIntParms.path_speed     = 2;
   fRFIntParms.path_xval      = 3;
   fRFIntParms.path_steps     = 50000;
   fRFIntParms.path_testfreq  = 100;
   fRFIntParms.tree_store     = 10000000;
   fRFIntParms.cat_store      = 1000000;

}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteAll()
{
   // write all files read by rf_go.exe
   WriteIntParms();
   WriteRealParms();
   WriteLx();
   WriteProgram();
   WriteVarNames();
   if (fRFProgram==kRfTrain)   WriteTrain();
   if (fRFProgram==kRfPredict) WriteTest();
   if (fRFProgram==kRfVarimp)  WriteRealVarImp();
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteIntParms()
{
   // write int params file
   std::ofstream f;
   if (!OpenRFile("intparms",f)) return kFALSE;
   WriteInt(f,&fRFIntParms.mode,sizeof(fRFIntParms));
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRealParms()
{
   // write int params file
   std::ofstream f;
   if (!OpenRFile("realparms",f)) return kFALSE;
   WriteFloat(f,&fRFRealParms.xmiss,sizeof(fRFRealParms));
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteLx()
{
   // Save input variable mask
   //
   // If the lx vector size is not the same as inputVars,
   // resize it and fill it with 1
   // NOTE: Always set all to 1
   //  if (fRFLx.size() != m_inputVars->size()) {
   fRFLx.clear();
   fRFLx.resize(Data().GetNVariables(),1);
   //  }
   std::ofstream f;
   if (!OpenRFile("lx",f)) return kFALSE;
   WriteInt(f,&fRFLx[0],fRFLx.size());
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteProgram()
{
   // write command to rf_go.exe
   std::ofstream f;
   if (!OpenRFile("program",f)) return kFALSE;
   TString program;
   switch (fRFProgram) {
   case kRfTrain:
      program = "rulefit";
      break;
   case kRfPredict:
      program = "rulefit_pred";
      break;
      // calculate variable importance
   case kRfVarimp:
      program = "varimp";
      break;
   default:
      fRFProgram = kRfTrain;
      program="rulefit";
      break;
   }
   f << program;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRealVarImp()
{
   // write the minimum importance to be considered
   std::ofstream f;
   if (!OpenRFile("realvarimp",f)) return kFALSE;
   Float_t rvp[2];
   rvp[0] = 0.0; // Mode: see varimp() in rulefit.r
   rvp[1] = 0.0; // Minimum importance considered (1 is max)
   WriteFloat(f,&rvp[0],2);
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRfOut()
{
   // written by rf_go.exe; write rulefit output (rfout)
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRfStatus()
{
   // written by rf_go.exe; write rulefit status
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRuleFitMod()
{
   // written by rf_go.exe (NOTE:Format unknown!)
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteRuleFitSum()
{
   // written by rf_go.exe (NOTE: format unknown!)
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteTrain()
{
   // write training data, columnwise
   std::ofstream fx;
   std::ofstream fy;
   std::ofstream fw;
   //
   if (!OpenRFile("train.x",fx)) return kFALSE;
   if (!OpenRFile("train.y",fy)) return kFALSE;
   if (!OpenRFile("train.w",fw)) return kFALSE;
   //
   Float_t x,y,w;
   //
   for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
      for (Int_t ievt=0;ievt<Data().GetNEvtTrain(); ievt++) {
         ReadTrainingEvent(ievt);
         x = GetEventVal(ivar);
         WriteFloat(fx,&x,1);
         if (ivar==0) {
            w = 1.0; // should include weight name
            y = (GetEvent().IsSignal() ? 1.0:-1.0);
            WriteFloat(fy,&y,1);
            WriteFloat(fw,&w,1);
         }
      }
   }
   fLogger << kINFO << "number of training data written : " << Data().GetNEvtTrain() << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteTest()
{
   // Write test data
   std::ofstream f;
   //
   if (!OpenRFile("test.x",f)) return kFALSE;
   //
   Float_t vf;
   Float_t neve;
   //
   neve = static_cast<Float_t>(Data().GetNEvtTest());
   WriteFloat(f,&neve,1);
   // Test data is saved as:
   // 0      : <N> num of events, type float, 4 bytes
   // 1-N    : First variable for all events
   // N+1-2N : Second variable...
   // ...
   for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
      for (Int_t ievt=0;ievt<Data().GetNEvtTest(); ievt++) {
         ReadTestEvent(ievt);
         vf =  GetEventVal(ivar);
         WriteFloat(f,&vf,1);
      }
   }
   fLogger << kINFO << "number of test data written : " << Data().GetNEvtTest() << Endl;
   //
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteVarNames()
{
   // write variable names, ascii
   std::ofstream f;
   if (!OpenRFile("varnames",f)) return kFALSE;
   for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
      f << Data().GetExpression(ivar) << '\n';
   }
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteVarImp()
{
   // written by rf_go.exe
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteYhat()
{
   // written by rf_go.exe
   fLogger << kWARNING << "--- " << GetName() << " is not yet implemented" << Endl;
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::ReadYhat()
{
   // read the score
   fRFYhat.clear();
   //
   std::ifstream f;
   if (!OpenRFile("yhat",f)) return kFALSE;
   Int_t   neve;
   Float_t xval;
   ReadFloat(f,&xval,1);
   neve = static_cast<Int_t>(xval);
   if (neve!=Data().GetNEvtTest()) {
      fLogger << kWARNING << "inconsistent size of yhat file and test tree!" << Endl;
      fLogger << kWARNING << "neve = " << neve << " , tree = " << Data().GetNEvtTest() << Endl;
      return kFALSE;
   }
   for (Int_t ievt=0; ievt<Data().GetNEvtTest(); ievt++) {
      ReadFloat(f,&xval,1);
      fRFYhat.push_back(xval);
   }
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::ReadVarImp()
{
   // read variable importance
   fRFVarImp.clear();
   //
   std::ifstream f;
   if (!OpenRFile("varimp",f)) return kFALSE;
   UInt_t   nvars;
   Float_t xval;
   Float_t xmax=1.0;
   nvars=Data().GetNVariables();
   //
   // First read all importances
   //
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      ReadFloat(f,&xval,1);
      if (ivar==0) {
         xmax=xval;
      } else {
         if (xval>xmax) xmax=xval;
      }
      fRFVarImp.push_back(xval);
   }
   //
   // Read the indices.
   // They are saved as float (!) by rf_go.exe.
   //
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      fRFVarImp[ivar] = fRFVarImp[ivar]/xmax;
      ReadFloat(f,&xval,1);
      fRFVarImpInd.push_back(Int_t(xval)-1);
   }
   return kTRUE;
}

//_______________________________________________________________________
int TMVA::MethodRuleFitJF::RunRuleFit()
{
   // execute rf_go.exe
   TString oldDir = gSystem->pwd();
   TString cmd = "./rf_go.exe"; 
   gSystem->cd(fRFWorkDir.Data());
   int rval = gSystem->Exec(cmd.Data());
   gSystem->cd(oldDir.Data());
   return rval;
}
