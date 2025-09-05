#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TString.h>
#include <TArrayD.h>
#include <TArrayI.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TMap.h>
#include <TSystem.h>
#include <Riostream.h>
#endif

class TMyPar : public TObject
{
public:
  Int_t       fNrComm;
  Int_t       fItemSize_5;

  TObjArray  *fSmoothL;    // array of smooth operations
  TMatrixD    fScale;
  Int_t       fLambdaModel;
  TArrayI     fHelpLambda;
  TArrayI     fApplyLambda;
  TMatrixD   *fCyclic;     //[fNrComm]
  TArrayD    *fLambda;     //[fItemSize_5]
  TArrayD     fSaneD;
  TArrayD     fSaneL;
  TArrayD     fSaneH;
  TMatrixD    fDomain;

  Double_t    fClipVar;
  Int_t       fLowCount;
  Int_t       fBayesNickelize;

 public:

  TMyPar();
  virtual ~TMyPar();

  ClassDefOverride(TMyPar,2)
};

ClassImp(TMyPar)
  
//______________________________________________________________________________
TMyPar::TMyPar()
{
  fNrComm         = 0;
  fItemSize_5     = 0;

  fSmoothL        = 0;
  fLambdaModel    = 0;
  fCyclic         = 0;
  fLambda         = 0;

  fClipVar        = 0.;
  fLowCount       = 0;
  fBayesNickelize = 0;
}

//______________________________________________________________________________
TMyPar::~TMyPar()
{
  if (fSmoothL) { delete fSmoothL;   fSmoothL = 0; }
  if (fCyclic)  { delete [] fCyclic; fCyclic  = 0; }
  if (fLambda)  { delete [] fLambda; fLambda  = 0; }
}

class TMyData : public TObject
{
public:

        TString     fModel;
  const TObjArray  *fCommL;
        Long_t      fSDay;
        Long_t      fEDay;
        TString     fParamFile;

        Long_t      fCoeffDate;

	Int_t       fNrSrcs;

        TString     fPeriodGroupStr;
        Double_t    fDotsHalfLife;
        Int_t       fWindow;
        TString     fFitVarType;

        TString     fWorkingDir;
        TString     fSmoothCoeffFile;
        TArrayD     fUse;
        TArrayD     fSsize;
        TArrayD     fNrho;
        TArrayD     fNrhoInv;
        TArrayD    *fPnts;  //[fNrSrcs]

        Int_t       fCalcVarRed;

        TMap       *fSmoothMonoM;
  
 public:

  TMyData();
  virtual ~TMyData();

  ClassDefOverride(TMyData,2)
};

ClassImp(TMyData)
  
//______________________________________________________________________________
TMyData::TMyData()
{
  fSDay         = 0;
  fEDay         = 0;

  fCoeffDate    = 0; 
  fNrSrcs       = 0;

  fDotsHalfLife = 0;
  fWindow       = 0;

  fCommL        = 0;

  fCalcVarRed   = 0;

  fPnts         = 0;

  fSmoothMonoM  = 0;
}

//______________________________________________________________________________
TMyData::~TMyData()
{
   cout << "Executing TMyData::~TMyData()\n"; 
   if (fCommL)       { delete fCommL;       fCommL       = 0; }
   if (fPnts)        { delete [] fPnts;     fPnts        = 0; }
   if (fSmoothMonoM) { fSmoothMonoM->DeleteAll(); delete fSmoothMonoM; fSmoothMonoM = 0; }
}

void runmemleak(bool showdetails = false)
{
  TFile f("memleak.root");
  
  TTree *t = (TTree *)f.Get("MonoData");
  t->Print();
  //t->SetBranchStatus("fSmoothMonoM",0);
  
  TMyData *data = 0;
  
  TBranch *br = t->GetBranch("mono");
  br->SetAddress(&data);
  br->SetAutoDelete(kTRUE);

  Info("memleak","branch has %lld entries",br->GetEntries());

  
  br->GetEntry(0);
  if (data->fSmoothMonoM)
  {
    Info("memleak","fSmoothMonoM->IsOwner(): %d",data->fSmoothMonoM->IsOwner());
    TIter keys(data->fSmoothMonoM);
    while (TObject *key = keys())
    {
      const TString  keyStr = ((TObjString *)key)->GetString();
      const TMyPar  *par    = (TMyPar *)data->fSmoothMonoM->GetValue(key);
      Info("memleak","fSmoothMonoM->par(%s)->fSmoothL->IsOwner(): %d",keyStr.Data(),par->fSmoothL->IsOwner());
    }
  }
  
  ProcInfo_t *info = new ProcInfo_t();
  for (Int_t i = 0; i < 50; i++)
  {
    br->GetEntry(0);
    //delete data; data = 0;
    //data->fSmoothMonoM->DeleteAll();

    gSystem->GetProcInfo(info);
    if (showdetails) Info("memleak","After entry: %d   Memory (Mb) : Res : %ld  Virt : %ld",
                            i,info->fMemResident/1000,info->fMemVirtual/1000);
  }

  delete info;
  delete data; data = 0;

  f.Close();
}
