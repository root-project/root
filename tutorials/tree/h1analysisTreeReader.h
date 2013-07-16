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
	TTreeReaderValue<Int_t>*      nrun; //!
	TTreeReaderValue<Int_t>*      nevent; //!
	TTreeReaderValue<Int_t>*      nentry; //!
	TTreeReaderArray<UChar_t>*    trelem; //!
	TTreeReaderArray<UChar_t>*    subtr; //!
	TTreeReaderArray<UChar_t>*    rawtr; //!
	TTreeReaderArray<UChar_t>*    L4subtr; //!
	TTreeReaderArray<UChar_t>*    L5class; //!
	TTreeReaderValue<Float_t>*    E33; //!
	TTreeReaderValue<Float_t>*    de33; //!
	TTreeReaderValue<Float_t>*    x33; //!
	TTreeReaderValue<Float_t>*    dx33; //!
	TTreeReaderValue<Float_t>*    y33; //!
	TTreeReaderValue<Float_t>*    dy33; //!
	TTreeReaderValue<Float_t>*    E44; //!
	TTreeReaderValue<Float_t>*    de44; //!
	TTreeReaderValue<Float_t>*    x44; //!
	TTreeReaderValue<Float_t>*    dx44; //!
	TTreeReaderValue<Float_t>*    y44; //!
	TTreeReaderValue<Float_t>*    dy44; //!
	TTreeReaderValue<Float_t>*    Ept; //!
	TTreeReaderValue<Float_t>*    dept; //!
	TTreeReaderValue<Float_t>*    xpt; //!
	TTreeReaderValue<Float_t>*    dxpt; //!
	TTreeReaderValue<Float_t>*    ypt; //!
	TTreeReaderValue<Float_t>*    dypt; //!
	TTreeReaderArray<Float_t>*    pelec; //!
	TTreeReaderValue<Int_t>*      flagelec; //!
	TTreeReaderValue<Float_t>*    xeelec; //!
	TTreeReaderValue<Float_t>*    yeelec; //!
	TTreeReaderValue<Float_t>*    Q2eelec; //!
	TTreeReaderValue<Int_t>*      nelec; //!
	TTreeReaderArray<Float_t>*    Eelec; //!
	TTreeReaderArray<Float_t>*    thetelec; //!
	TTreeReaderArray<Float_t>*    phielec; //!
	TTreeReaderArray<Float_t>*    xelec; //!
	TTreeReaderArray<Float_t>*    Q2elec; //!
	TTreeReaderArray<Float_t>*    xsigma; //!
	TTreeReaderArray<Float_t>*    Q2sigma; //!
	TTreeReaderArray<Float_t>*    sumc; //!
	TTreeReaderValue<Float_t>*    sumetc; //!
	TTreeReaderValue<Float_t>*    yjbc; //!
	TTreeReaderValue<Float_t>*    Q2jbc; //!
	TTreeReaderArray<Float_t>*    sumct; //!
	TTreeReaderValue<Float_t>*    sumetct; //!
	TTreeReaderValue<Float_t>*    yjbct; //!
	TTreeReaderValue<Float_t>*    Q2jbct; //!
	TTreeReaderValue<Float_t>*    Ebeamel; //!
	TTreeReaderValue<Float_t>*    Ebeampr; //!
	TTreeReaderArray<Float_t>*    pvtx_d; //!
	TTreeReaderArray<Float_t>*    cpvtx_d; //!
	TTreeReaderArray<Float_t>*    pvtx_t; //!
	TTreeReaderArray<Float_t>*    cpvtx_t; //!
	TTreeReaderValue<Int_t>*      ntrkxy_t; //!
	TTreeReaderValue<Float_t>*    prbxy_t; //!
	TTreeReaderValue<Int_t>*      ntrkz_t; //!
	TTreeReaderValue<Float_t>*    prbz_t; //!
	TTreeReaderValue<Int_t>*      nds; //!
	TTreeReaderValue<Int_t>*      rankds; //!
	TTreeReaderValue<Int_t>*      qds; //!
	TTreeReaderArray<Float_t>*    pds_d; //!
	TTreeReaderValue<Float_t>*    ptds_d; //!
	TTreeReaderValue<Float_t>*    etads_d; //!
	TTreeReaderValue<Float_t>*    dm_d; //!
	TTreeReaderValue<Float_t>*    ddm_d; //!
	TTreeReaderArray<Float_t>*    pds_t; //!
	TTreeReaderValue<Float_t>*    dm_t; //!
	TTreeReaderValue<Float_t>*    ddm_t; //!
	TTreeReaderValue<Int_t>*      ik; //!
	TTreeReaderValue<Int_t>*      ipi; //!
	TTreeReaderValue<Int_t>*      ipis; //!
	TTreeReaderArray<Float_t>*    pd0_d; //!
	TTreeReaderValue<Float_t>*    ptd0_d; //!
	TTreeReaderValue<Float_t>*    etad0_d; //!
	TTreeReaderValue<Float_t>*    md0_d; //!
	TTreeReaderValue<Float_t>*    dmd0_d; //!
	TTreeReaderArray<Float_t>*    pd0_t; //!
	TTreeReaderValue<Float_t>*    md0_t; //!
	TTreeReaderValue<Float_t>*    dmd0_t; //!
	TTreeReaderArray<Float_t>*    pk_r; //!
	TTreeReaderArray<Float_t>*    ppi_r; //!
	TTreeReaderArray<Float_t>*    pd0_r; //!
	TTreeReaderValue<Float_t>*    md0_r; //!
	TTreeReaderArray<Float_t>*    Vtxd0_r; //!
	TTreeReaderArray<Float_t>*    cvtxd0_r; //!
	TTreeReaderValue<Float_t>*    dxy_r; //!
	TTreeReaderValue<Float_t>*    dz_r; //!
	TTreeReaderValue<Float_t>*    psi_r; //!
	TTreeReaderValue<Float_t>*    rd0_d; //!
	TTreeReaderValue<Float_t>*    drd0_d; //!
	TTreeReaderValue<Float_t>*    rpd0_d; //!
	TTreeReaderValue<Float_t>*    drpd0_d; //!
	TTreeReaderValue<Float_t>*    rd0_t; //!
	TTreeReaderValue<Float_t>*    drd0_t; //!
	TTreeReaderValue<Float_t>*    rpd0_t; //!
	TTreeReaderValue<Float_t>*    drpd0_t; //!
	TTreeReaderValue<Float_t>*    rd0_dt; //!
	TTreeReaderValue<Float_t>*    drd0_dt; //!
	TTreeReaderValue<Float_t>*    prbr_dt; //!
	TTreeReaderValue<Float_t>*    prbz_dt; //!
	TTreeReaderValue<Float_t>*    rd0_tt; //!
	TTreeReaderValue<Float_t>*    drd0_tt; //!
	TTreeReaderValue<Float_t>*    prbr_tt; //!
	TTreeReaderValue<Float_t>*    prbz_tt; //!
	TTreeReaderValue<Int_t>*      ijetd0; //!
	TTreeReaderValue<Float_t>*    ptr3d0_j; //!
	TTreeReaderValue<Float_t>*    ptr2d0_j; //!
	TTreeReaderValue<Float_t>*    ptr3d0_3; //!
	TTreeReaderValue<Float_t>*    ptr2d0_3; //!
	TTreeReaderValue<Float_t>*    ptr2d0_2; //!
	TTreeReaderValue<Float_t>*    Mimpds_r; //!
	TTreeReaderValue<Float_t>*    Mimpbk_r; //!
	TTreeReaderValue<Int_t>*      ntracks; //!
	TTreeReaderArray<Float_t>*    pt; //!
	TTreeReaderArray<Float_t>*    kappa; //!
	TTreeReaderArray<Float_t>*    phi; //!
	TTreeReaderArray<Float_t>*    theta; //!
	TTreeReaderArray<Float_t>*    dca; //!
	TTreeReaderArray<Float_t>*    z0; //!
	TTreeReaderArray<Float_t>*    covar; //!
	TTreeReaderArray<Int_t>*      nhitrp; //!
	TTreeReaderArray<Float_t>*    prbrp; //!
	TTreeReaderArray<Int_t>*      nhitz; //!
	TTreeReaderArray<Float_t>*    prbz; //!
	TTreeReaderArray<Float_t>*    rstart; //!
	TTreeReaderArray<Float_t>*    rend; //!
	TTreeReaderArray<Float_t>*    lhk; //!
	TTreeReaderArray<Float_t>*    lhpi; //!
	TTreeReaderArray<Float_t>*    nlhk; //!
	TTreeReaderArray<Float_t>*    nlhpi; //!
	TTreeReaderArray<Float_t>*    dca_d; //!
	TTreeReaderArray<Float_t>*    ddca_d; //!
	TTreeReaderArray<Float_t>*    dca_t; //!
	TTreeReaderArray<Float_t>*    ddca_t; //!
	TTreeReaderArray<Int_t>*      muqual; //!
	TTreeReaderValue<Int_t>*      imu; //!
	TTreeReaderValue<Int_t>*      imufe; //!
	TTreeReaderValue<Int_t>*      njets; //!
	TTreeReaderArray<Float_t>*    E_j; //!
	TTreeReaderArray<Float_t>*    pt_j; //!
	TTreeReaderArray<Float_t>*    theta_j; //!
	TTreeReaderArray<Float_t>*    eta_j; //!
	TTreeReaderArray<Float_t>*    phi_j; //!
	TTreeReaderArray<Float_t>*    m_j; //!
	TTreeReaderValue<Float_t>*    thrust; //!
	TTreeReaderArray<Float_t>*    pthrust; //!
	TTreeReaderValue<Float_t>*    thrust2; //!
	TTreeReaderArray<Float_t>*    pthrust2; //!
	TTreeReaderValue<Float_t>*    spher; //!
	TTreeReaderValue<Float_t>*    aplan; //!
	TTreeReaderValue<Float_t>*    plan; //!
	TTreeReaderArray<Float_t>*    nnout; //!

	TTreeReader* 				  myTreeReader;//!

	TH1F           *hdmd;//!
	TH2F           *h2;//!

	Bool_t          useList;//!
	Bool_t          fillList;//!
	TEntryList     *elist;//!
	Long64_t        fProcessed;//!

   	TTree          *fChain;//!    //pointer to the analyzed TTree or TChain
   	Long64_t		fChainOffset;//!

	h1analysisTreeReader(TTree* tree=0);

	virtual ~h1analysisTreeReader() { }
	void    Reset();

	int     Version() const {return 1;}
	void    Begin(TTree *);
	void    SlaveBegin(TTree *);
	void    Init(TTree *);
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
h1analysisTreeReader::h1analysisTreeReader(TTree * /*tree*/)
{
  // Constructor

   Reset();
}
//_____________________________________________________________________
void h1analysisTreeReader::Reset()
{
   // Reset the data members to theit initial value

   hdmd = 0;
   h2 = 0;
   fChain = 0;
   elist = 0;
   fillList = kFALSE;
   useList  = kFALSE;
   fProcessed = 0;
}

#endif
