#ifndef h1analysisTreeReader_h
#define h1analysisTreeReader_h

#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TSelector.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TEntryList.h"

class h1analysisTreeReader : public TSelector {
public:
   TTreeReader                  myTreeReader;//!

   TTreeReaderValue<Float_t>    ptds_d; //!
   TTreeReaderValue<Float_t>    etads_d; //!
   TTreeReaderValue<Float_t>    dm_d; //!
   TTreeReaderValue<Int_t>      ik; //!
   TTreeReaderValue<Int_t>      ipi; //!
   TTreeReaderValue<Int_t>      ipis; //!
   TTreeReaderValue<Float_t>    ptd0_d; //!
   TTreeReaderValue<Float_t>    md0_d; //!
   TTreeReaderValue<Float_t>    rpd0_t; //!
   TTreeReaderArray<Int_t>      nhitrp; //!
   TTreeReaderArray<Float_t>    rstart; //!
   TTreeReaderArray<Float_t>    rend; //!
   TTreeReaderArray<Float_t>    nlhk; //!
   TTreeReaderArray<Float_t>    nlhpi; //!
   TTreeReaderValue<Int_t>      njets; //!

   TH1F                         *hdmd;//!
   TH2F                         *h2;//!

   Bool_t                        useList;//!
   Bool_t                        fillList;//!
   TEntryList                   *elist;//!
   Long64_t                      fProcessed;//!

   Long64_t                      fChainOffset;//!

   h1analysisTreeReader(TTree* /*tree*/=0) :
   	myTreeReader(),
	   ptds_d     (myTreeReader, "ptds_d"  ),
		etads_d    (myTreeReader, "etads_d" ),
		dm_d       (myTreeReader, "dm_d"    ),
		ik         (myTreeReader, "ik"      ),
		ipi        (myTreeReader, "ipi"     ),
		ipis       (myTreeReader, "ipis"    ),
		ptd0_d     (myTreeReader, "ptd0_d"  ),
		md0_d      (myTreeReader, "md0_d"   ),
		rpd0_t     (myTreeReader, "rpd0_t"  ),
		nhitrp     (myTreeReader, "nhitrp"  ),
		rstart     (myTreeReader, "rstart"  ),
		rend       (myTreeReader, "rend"    ),
		nlhk       (myTreeReader, "nlhk"    ),
		nlhpi      (myTreeReader, "nlhpi"   ),
		njets      (myTreeReader, "njets"   )
   {
   	Reset();
   };

   virtual ~h1analysisTreeReader() { }
   void    Reset();

   int     Version() const {return 1;}
   void    Begin(TTree *);
   void    SlaveBegin(TTree *);
   void    Init(TTree *) {};
   Bool_t  Notify();
   Bool_t  Process(Long64_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    SlaveTerminate();
   void    Terminate();

   ClassDef(h1analysisTreeReader,0);
};

//_____________________________________________________________________
void h1analysisTreeReader::Reset()
{
   // Reset the data members to theit initial value

   hdmd = 0;
   h2 = 0;
   elist = 0;
   fillList = kFALSE;
   useList  = kFALSE;
   fProcessed = 0;
}

#endif
