#ifndef h1analysisTreeReader_h
#define h1analysisTreeReader_h

#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TSelector.h"
#include "TEntryList.h"
#include "TH2.h"
#include "TF1.h"

class h1analysisTreeReader : public TSelector {
public:
   TTreeReader                  myTreeReader;//!

   TTreeReaderValue<Float_t>    fPtds_d; //!
   TTreeReaderValue<Float_t>    fEtads_d; //!
   TTreeReaderValue<Float_t>    fDm_d; //!
   TTreeReaderValue<Int_t>      fIk; //!
   TTreeReaderValue<Int_t>      fIpi; //!
   TTreeReaderValue<Int_t>      fIpis; //!
   TTreeReaderValue<Float_t>    fPtd0_d; //!
   TTreeReaderValue<Float_t>    fMd0_d; //!
   TTreeReaderValue<Float_t>    fRpd0_t; //!
   TTreeReaderArray<Int_t>      fNhitrp; //!
   TTreeReaderArray<Float_t>    fRstart; //!
   TTreeReaderArray<Float_t>    fRend; //!
   TTreeReaderArray<Float_t>    fNlhk; //!
   TTreeReaderArray<Float_t>    fNlhpi; //!
   TTreeReaderValue<Int_t>      fNjets; //!

   TH1F                         *hdmd;//!
   TH2F                         *h2;//!

   Bool_t                        useList;//!
   Bool_t                        fillList;//!
   TEntryList                   *elist;//!
   Long64_t                      fProcessed;//!

   h1analysisTreeReader(TTree* /*tree*/=nullptr) :
      myTreeReader(),
      fPtds_d     (myTreeReader, "ptds_d"  ),
      fEtads_d    (myTreeReader, "etads_d" ),
      fDm_d       (myTreeReader, "dm_d"    ),
      fIk         (myTreeReader, "ik"      ),
      fIpi        (myTreeReader, "ipi"     ),
      fIpis       (myTreeReader, "ipis"    ),
      fPtd0_d     (myTreeReader, "ptd0_d"  ),
      fMd0_d      (myTreeReader, "md0_d"   ),
      fRpd0_t     (myTreeReader, "rpd0_t"  ),
      fNhitrp     (myTreeReader, "nhitrp"  ),
      fRstart     (myTreeReader, "rstart"  ),
      fRend       (myTreeReader, "rend"    ),
      fNlhk       (myTreeReader, "nlhk"    ),
      fNlhpi      (myTreeReader, "nlhpi"   ),
      fNjets      (myTreeReader, "njets"   )
   {
      Reset();
   };

   ~h1analysisTreeReader() override { }
   void    Reset();

   int     Version() const override {return 1;}
   void    Begin(TTree *) override;
   void    SlaveBegin(TTree *) override;
   void    Init(TTree *myTree) override { myTreeReader.SetTree(myTree); }
   Bool_t  Notify() override;
   Bool_t  Process(Long64_t entry) override;
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override {fInput = input;}
   TList  *GetOutputList() const override { return fOutput; }
   void    SlaveTerminate() override;
   void    Terminate() override;

   ClassDefOverride(h1analysisTreeReader,0);
};

//_____________________________________________________________________
void h1analysisTreeReader::Reset()
{
   // Reset the data members to theit initial value

   hdmd = nullptr;
   h2 = nullptr;
   elist = nullptr;
   fillList = kFALSE;
   useList  = kFALSE;
   fProcessed = 0;
}

#endif
