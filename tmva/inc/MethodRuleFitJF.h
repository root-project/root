// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Fredrik Tegenfeldt, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRuleFit                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Friedman's RuleFit method                                                 * 
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch>     - CERN, Switzerland       *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *      Kai Voss           <Kai.Voss@cern.ch>           - U. of Victoria, Canada  *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodRuleFitJF
#define ROOT_TMVA_MethodRuleFitJF

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodRuleFit                                                        //
//                                                                      //
// J Friedman's RuleFit method                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif
#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif
#ifndef ROOT_TMVA_GiniIndex
#include "TMVA/GiniIndex.h"
#endif
#ifndef ROOT_TMVA_CrossEntropy
#include "TMVA/CrossEntropy.h"
#endif
#ifndef ROOT_TMVA_MisClassificationError
#include "TMVA/MisClassificationError.h"
#endif
#ifndef ROOT_TMVA_SdivSqrtSplusB
#include "TMVA/SdivSqrtSplusB.h"
#endif

namespace TMVA {

   class MethodRuleFitJF : public MethodBase {

   public:

      MethodRuleFitJF( TString jobName,
                       TString methodTitle, 
                       DataSet& theData,
                       TString theOption = "",
                       TDirectory* theTargetDir = 0 );

      MethodRuleFitJF( DataSet& theData,
                       TString theWeightFile,
                       TDirectory* theTargetDir = NULL );

      virtual ~MethodRuleFitJF( void );

      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      //      virtual Double_t GetMvaValue(Event *e);
      virtual Double_t GetMvaValue();

      // ranking of input variables
      const Ranking* CreateRanking();

      // Set RuleFit working directory
      void SetRFWorkDir(const char * wdir);

      // Check RF work dir - aborts if it fails
      void CheckRFWorkDir();

      // Get working directory
      const TString GetRFWorkDir() const { return fRFWorkDir; }

   protected:

      enum ERFMode    { kRfRegress=1, kRfClass=2 };          // RuleFit modes, default=Class
      enum EModel     { kRfLinear=0, kRfRules=1, kRfBoth=2 }; // models, default=Both (rules+linear)
      enum ERFProgram { kRfTrain=0, kRfPredict, kRfVarimp };    // rf_go.exe running mode
  
      // integer parameters
      typedef struct {
         int mode;
         int lmode;
         int n;
         int p;
         int max_rules;
         int tree_size;
         int path_speed;
         int path_xval;
         int path_steps;
         int path_testfreq;
         int tree_store;
         int cat_store;
      } IntParms;

      // float parameters
      typedef struct {
         float xmiss;
         float trim_qntl;
         float huber;
         float inter_supp;
         float memory_par;
         float samp_fract;
         float path_inc;
         float conv_fac;
      } RealParms;

      // initialize rulefit
      void InitRuleFit( void );

      void GetOptVal( TString & optstr, TString & opt, TString & optval );
      void SetOptValInt( TString & optstr, TString & opt, Int_t & val );
      void SetOptValFloat( TString & optstr, TString & opt, Float_t & val );
      void ProcessRFOptions();
      void SetTrainParms();
      void SetTestParms();
      void FillRealParmsDef();
      void FillIntParmsDef();
  
      int RunRuleFit();

      // set rf_go.exe running mode
      void SetRFTrain()   { fRFProgram = kRfTrain; }
      void SetRFPredict() { fRFProgram = kRfPredict; }
      void SetRFVarimp()  { fRFProgram = kRfVarimp; }

      // run rf_go.exe in various modes
      inline void TrainRuleFit();
      inline void TestRuleFit();
      inline void VarImp();

      // handle rulefit files
      inline TString GetRFName(TString name);
      inline Bool_t OpenRFile(TString name, std::ofstream & f);
      inline Bool_t OpenRFile(TString name, std::ifstream & f);

      // read/write binary files
      inline Bool_t WriteInt(ofstream &   f, const Int_t   *v, Int_t n=1);
      inline Bool_t WriteFloat(ofstream & f, const Float_t *v, Int_t n=1);
      inline Bool_t ReadInt(ifstream & f,   Int_t *v, Int_t n=1);
      inline Bool_t ReadFloat(ifstream & f, Float_t *v, Int_t n=1);
  
      // write rf_go.exe i/o files
      Bool_t WriteAll();
      Bool_t WriteIntParms();
      Bool_t WriteRealParms();
      Bool_t WriteLx();
      Bool_t WriteProgram();
      Bool_t WriteRealVarImp();
      Bool_t WriteRfOut();
      Bool_t WriteRfStatus();
      Bool_t WriteRuleFitMod();
      Bool_t WriteRuleFitSum();
      Bool_t WriteTrain();
      Bool_t WriteVarNames();
      Bool_t WriteVarImp();
      Bool_t WriteYhat();
      Bool_t WriteTest();

      // read rf_go.exe i/o files
      Bool_t ReadYhat();
      Bool_t ReadIntParms();
      Bool_t ReadRealParms();
      Bool_t ReadLx();
      Bool_t ReadProgram();
      Bool_t ReadRealVarImp();
      Bool_t ReadRfOut();
      Bool_t ReadRfStatus();
      Bool_t ReadRuleFitMod();
      Bool_t ReadRuleFitSum();
      Bool_t ReadTrainX();
      Bool_t ReadTrainY();
      Bool_t ReadTrainW();
      Bool_t ReadVarNames();
      Bool_t ReadVarImp();

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      std::vector<Float_t> fRFYhat;      // score results from test sample
      std::vector<Float_t> fRFVarImp;    // variable importances
      std::vector<Int_t>   fRFVarImpInd; // variable index
      TString              fRFWorkDir;   // working directory
      IntParms             fRFIntParms;  // integer parameters
      RealParms            fRFRealParms; // real parameters
      std::vector<int>     fRFLx;        // variable selector
      ERFProgram           fRFProgram;   // what to run
      TString              fModelType;   // model type string

      ClassDef(MethodRuleFitJF,0)        // Friedman's RuleFit method

   };

} // namespace TMVA

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::TrainRuleFit()
{
   // run rf_go.exe to train the model
   SetTrainParms();
   SetRFTrain();
   WriteAll();
   RunRuleFit();
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::TestRuleFit()
{
   // run rf_go.exe with the test data
   SetTestParms();
   SetRFPredict();
   WriteAll();
   RunRuleFit();
   ReadYhat(); // read in the scores
}

//_______________________________________________________________________
void TMVA::MethodRuleFitJF::VarImp()
{
   // run rf_go.exe to get the variable importance
   SetRFVarimp();
   WriteAll();
   RunRuleFit();
   ReadVarImp(); // read in the variable importances
}

//_______________________________________________________________________
TString TMVA::MethodRuleFitJF::GetRFName(TString name)
{
   // get the name inluding the rulefit directory
   return fRFWorkDir+"/"+name;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::OpenRFile(TString name, std::ofstream & f)
{
   // open a file for writing in the rulefit directory
   TString fullName = GetRFName(name);
   f.open(fullName);
   if (!f.is_open()) {
      fLogger << kWARNING << "--- " << GetName() << ": Error opening RuleFit file for output "
              << fullName << Endl;
      return kFALSE;
   }
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::OpenRFile(TString name, std::ifstream & f)
{
   // open a file for reading in the rulefit directory
   TString fullName = GetRFName(name);
   f.open(fullName);
   if (!f.is_open()) {
      fLogger << kWARNING << "--- " << GetName() << ": Error opening RuleFit file for input "
              << fullName << Endl;
      return kFALSE;
   }
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteInt(ofstream &   f, const Int_t   *v, Int_t n)
{
   // write an int
   if (!f.is_open()) return kFALSE;
   return f.write(reinterpret_cast<char const *>(v), n*sizeof(Int_t));
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::WriteFloat(ofstream & f, const Float_t *v, Int_t n)
{
   // write a float
   if (!f.is_open()) return kFALSE;
   return f.write(reinterpret_cast<char const *>(v), n*sizeof(Float_t));
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::ReadInt(ifstream & f,   Int_t *v, Int_t n)
{
   // read an int
   if (!f.is_open()) return kFALSE;
   return f.read(reinterpret_cast<char *>(v), n*sizeof(Int_t));
}

//_______________________________________________________________________
Bool_t TMVA::MethodRuleFitJF::ReadFloat(ifstream & f, Float_t *v, Int_t n)
{
   // read a float
   if (!f.is_open()) return kFALSE;
   return f.read(reinterpret_cast<char *>(v), n*sizeof(Float_t));
}

#endif // MethodRuleFitJF_H
