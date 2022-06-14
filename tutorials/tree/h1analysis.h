//////////////////////////////////////////////////////////
//   This class has been automatically generated
//     (Wed Apr 19 21:47:55 2000 by ROOT version 2.24/02)
//   from TTree h42/
//   found on file: Memory Directory
//
//   The example was modfied for the new TSelector version
//   (Thu Sep 25 06:44:10 EDT 2003)
//////////////////////////////////////////////////////////


#ifndef h1analysis_h
#define h1analysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TEntryList.h>
#include <TFile.h>
#include <TSelector.h>

class   TH1F;
class   TH2F;

class h1analysis : public TSelector {
   public :

   TH1F           *hdmd;//!
   TH2F           *h2;//!

   Bool_t          useList;//!
   Bool_t          fillList;//!
   TEntryList     *elist;//!
   Long64_t        fProcessed;//!

   TTree          *fChain;//!    //pointer to the analyzed TTree or TChain
   //Declaration of leaves types
   Int_t           nrun;//!
   Int_t           nevent;//!
   Int_t           nentry;//!
   UChar_t         trelem[192];//!
   UChar_t         subtr[128];//!
   UChar_t         rawtr[128];//!
   UChar_t         L4subtr[128];//!
   UChar_t         L5class[32];//!
   Float_t         E33;//!
   Float_t         de33;//!
   Float_t         x33;//!
   Float_t         dx33;//!
   Float_t         y33;//!
   Float_t         dy33;//!
   Float_t         E44;//!
   Float_t         de44;//!
   Float_t         x44;//!
   Float_t         dx44;//!
   Float_t         y44;//!
   Float_t         dy44;//!
   Float_t         Ept;//!
   Float_t         dept;//!
   Float_t         xpt;//!
   Float_t         dxpt;//!
   Float_t         ypt;//!
   Float_t         dypt;//!
   Float_t         pelec[4];//!
   Int_t           flagelec;//!
   Float_t         xeelec;//!
   Float_t         yeelec;//!
   Float_t         Q2eelec;//!
   Int_t           nelec;//!
   Float_t         Eelec[20];//!
   Float_t         thetelec[20];//!
   Float_t         phielec[20];//!
   Float_t         xelec[20];//!
   Float_t         Q2elec[20];//!
   Float_t         xsigma[20];//!
   Float_t         Q2sigma[20];//!
   Float_t         sumc[4];//!
   Float_t         sumetc;//!
   Float_t         yjbc;//!
   Float_t         Q2jbc;//!
   Float_t         sumct[4];//!
   Float_t         sumetct;//!
   Float_t         yjbct;//!
   Float_t         Q2jbct;//!
   Float_t         Ebeamel;//!
   Float_t         Ebeampr;//!
   Float_t         pvtx_d[3];//!
   Float_t         cpvtx_d[6];//!
   Float_t         pvtx_t[3];//!
   Float_t         cpvtx_t[6];//!
   Int_t           ntrkxy_t;//!
   Float_t         prbxy_t;//!
   Int_t           ntrkz_t;//!
   Float_t         prbz_t;//!
   Int_t           nds;//!
   Int_t           rankds;//!
   Int_t           qds;//!
   Float_t         pds_d[4];//!
   Float_t         ptds_d;//!
   Float_t         etads_d;//!
   Float_t         dm_d;//!
   Float_t         ddm_d;//!
   Float_t         pds_t[4];//!
   Float_t         dm_t;//!
   Float_t         ddm_t;//!
   Int_t           ik;//!
   Int_t           ipi;//!
   Int_t           ipis;//!
   Float_t         pd0_d[4];//!
   Float_t         ptd0_d;//!
   Float_t         etad0_d;//!
   Float_t         md0_d;//!
   Float_t         dmd0_d;//!
   Float_t         pd0_t[4];//!
   Float_t         md0_t;//!
   Float_t         dmd0_t;//!
   Float_t         pk_r[4];//!
   Float_t         ppi_r[4];//!
   Float_t         pd0_r[4];//!
   Float_t         md0_r;//!
   Float_t         Vtxd0_r[3];//!
   Float_t         cvtxd0_r[6];//!
   Float_t         dxy_r;//!
   Float_t         dz_r;//!
   Float_t         psi_r;//!
   Float_t         rd0_d;//!
   Float_t         drd0_d;//!
   Float_t         rpd0_d;//!
   Float_t         drpd0_d;//!
   Float_t         rd0_t;//!
   Float_t         drd0_t;//!
   Float_t         rpd0_t;//!
   Float_t         drpd0_t;//!
   Float_t         rd0_dt;//!
   Float_t         drd0_dt;//!
   Float_t         prbr_dt;//!
   Float_t         prbz_dt;//!
   Float_t         rd0_tt;//!
   Float_t         drd0_tt;//!
   Float_t         prbr_tt;//!
   Float_t         prbz_tt;//!
   Int_t           ijetd0;//!
   Float_t         ptr3d0_j;//!
   Float_t         ptr2d0_j;//!
   Float_t         ptr3d0_3;//!
   Float_t         ptr2d0_3;//!
   Float_t         ptr2d0_2;//!
   Float_t         Mimpds_r;//!
   Float_t         Mimpbk_r;//!
   Int_t           ntracks;//!
   Float_t         pt[200];//!
   Float_t         kappa[200];//!
   Float_t         phi[200];//!
   Float_t         theta[200];//!
   Float_t         dca[200];//!
   Float_t         z0[200];//!
   Float_t         covar[200][15];//!
   Int_t           nhitrp[200];//!
   Float_t         prbrp[200];//!
   Int_t           nhitz[200];//!
   Float_t         prbz[200];//!
   Float_t         rstart[200];//!
   Float_t         rend[200];//!
   Float_t         lhk[200];//!
   Float_t         lhpi[200];//!
   Float_t         nlhk[200];//!
   Float_t         nlhpi[200];//!
   Float_t         dca_d[200];//!
   Float_t         ddca_d[200];//!
   Float_t         dca_t[200];//!
   Float_t         ddca_t[200];//!
   Int_t           muqual[200];//!
   Int_t           imu;//!
   Int_t           imufe;//!
   Int_t           njets;//!
   Float_t         E_j[20];//!
   Float_t         pt_j[20];//!
   Float_t         theta_j[20];//!
   Float_t         eta_j[20];//!
   Float_t         phi_j[20];//!
   Float_t         m_j[20];//!
   Float_t         thrust;//!
   Float_t         pthrust[4];//!
   Float_t         thrust2;//!
   Float_t         pthrust2[4];//!
   Float_t         spher;//!
   Float_t         aplan;//!
   Float_t         plan;//!
   Float_t         nnout[1];//!

//List of branches
   TBranch        *b_nrun;//!
   TBranch        *b_nevent;//!
   TBranch        *b_nentry;//!
   TBranch        *b_trelem;//!
   TBranch        *b_subtr;//!
   TBranch        *b_rawtr;//!
   TBranch        *b_L4subtr;//!
   TBranch        *b_L5class;//!
   TBranch        *b_E33;//!
   TBranch        *b_de33;//!
   TBranch        *b_x33;//!
   TBranch        *b_dx33;//!
   TBranch        *b_y33;//!
   TBranch        *b_dy33;//!
   TBranch        *b_E44;//!
   TBranch        *b_de44;//!
   TBranch        *b_x44;//!
   TBranch        *b_dx44;//!
   TBranch        *b_y44;//!
   TBranch        *b_dy44;//!
   TBranch        *b_Ept;//!
   TBranch        *b_dept;//!
   TBranch        *b_xpt;//!
   TBranch        *b_dxpt;//!
   TBranch        *b_ypt;//!
   TBranch        *b_dypt;//!
   TBranch        *b_pelec;//!
   TBranch        *b_flagelec;//!
   TBranch        *b_xeelec;//!
   TBranch        *b_yeelec;//!
   TBranch        *b_Q2eelec;//!
   TBranch        *b_nelec;//!
   TBranch        *b_Eelec;//!
   TBranch        *b_thetelec;//!
   TBranch        *b_phielec;//!
   TBranch        *b_xelec;//!
   TBranch        *b_Q2elec;//!
   TBranch        *b_xsigma;//!
   TBranch        *b_Q2sigma;//!
   TBranch        *b_sumc;//!
   TBranch        *b_sumetc;//!
   TBranch        *b_yjbc;//!
   TBranch        *b_Q2jbc;//!
   TBranch        *b_sumct;//!
   TBranch        *b_sumetct;//!
   TBranch        *b_yjbct;//!
   TBranch        *b_Q2jbct;//!
   TBranch        *b_Ebeamel;//!
   TBranch        *b_Ebeampr;//!
   TBranch        *b_pvtx_d;//!
   TBranch        *b_cpvtx_d;//!
   TBranch        *b_pvtx_t;//!
   TBranch        *b_cpvtx_t;//!
   TBranch        *b_ntrkxy_t;//!
   TBranch        *b_prbxy_t;//!
   TBranch        *b_ntrkz_t;//!
   TBranch        *b_prbz_t;//!
   TBranch        *b_nds;//!
   TBranch        *b_rankds;//!
   TBranch        *b_qds;//!
   TBranch        *b_pds_d;//!
   TBranch        *b_ptds_d;//!
   TBranch        *b_etads_d;//!
   TBranch        *b_dm_d;//!
   TBranch        *b_ddm_d;//!
   TBranch        *b_pds_t;//!
   TBranch        *b_dm_t;//!
   TBranch        *b_ddm_t;//!
   TBranch        *b_ik;//!
   TBranch        *b_ipi;//!
   TBranch        *b_ipis;//!
   TBranch        *b_pd0_d;//!
   TBranch        *b_ptd0_d;//!
   TBranch        *b_etad0_d;//!
   TBranch        *b_md0_d;//!
   TBranch        *b_dmd0_d;//!
   TBranch        *b_pd0_t;//!
   TBranch        *b_md0_t;//!
   TBranch        *b_dmd0_t;//!
   TBranch        *b_pk_r;//!
   TBranch        *b_ppi_r;//!
   TBranch        *b_pd0_r;//!
   TBranch        *b_md0_r;//!
   TBranch        *b_Vtxd0_r;//!
   TBranch        *b_cvtxd0_r;//!
   TBranch        *b_dxy_r;//!
   TBranch        *b_dz_r;//!
   TBranch        *b_psi_r;//!
   TBranch        *b_rd0_d;//!
   TBranch        *b_drd0_d;//!
   TBranch        *b_rpd0_d;//!
   TBranch        *b_drpd0_d;//!
   TBranch        *b_rd0_t;//!
   TBranch        *b_drd0_t;//!
   TBranch        *b_rpd0_t;//!
   TBranch        *b_drpd0_t;//!
   TBranch        *b_rd0_dt;//!
   TBranch        *b_drd0_dt;//!
   TBranch        *b_prbr_dt;//!
   TBranch        *b_prbz_dt;//!
   TBranch        *b_rd0_tt;//!
   TBranch        *b_drd0_tt;//!
   TBranch        *b_prbr_tt;//!
   TBranch        *b_prbz_tt;//!
   TBranch        *b_ijetd0;//!
   TBranch        *b_ptr3d0_j;//!
   TBranch        *b_ptr2d0_j;//!
   TBranch        *b_ptr3d0_3;//!
   TBranch        *b_ptr2d0_3;//!
   TBranch        *b_ptr2d0_2;//!
   TBranch        *b_Mimpds_r;//!
   TBranch        *b_Mimpbk_r;//!
   TBranch        *b_ntracks;//!
   TBranch        *b_pt;//!
   TBranch        *b_kappa;//!
   TBranch        *b_phi;//!
   TBranch        *b_theta;//!
   TBranch        *b_dca;//!
   TBranch        *b_z0;//!
   TBranch        *b_covar;//!
   TBranch        *b_nhitrp;//!
   TBranch        *b_prbrp;//!
   TBranch        *b_nhitz;//!
   TBranch        *b_prbz;//!
   TBranch        *b_rstart;//!
   TBranch        *b_rend;//!
   TBranch        *b_lhk;//!
   TBranch        *b_lhpi;//!
   TBranch        *b_nlhk;//!
   TBranch        *b_nlhpi;//!
   TBranch        *b_dca_d;//!
   TBranch        *b_ddca_d;//!
   TBranch        *b_dca_t;//!
   TBranch        *b_ddca_t;//!
   TBranch        *b_muqual;//!
   TBranch        *b_imu;//!
   TBranch        *b_imufe;//!
   TBranch        *b_njets;//!
   TBranch        *b_E_j;//!
   TBranch        *b_pt_j;//!
   TBranch        *b_theta_j;//!
   TBranch        *b_eta_j;//!
   TBranch        *b_phi_j;//!
   TBranch        *b_m_j;//!
   TBranch        *b_thrust;//!
   TBranch        *b_pthrust;//!
   TBranch        *b_thrust2;//!
   TBranch        *b_pthrust2;//!
   TBranch        *b_spher;//!
   TBranch        *b_aplan;//!
   TBranch        *b_plan;//!
   TBranch        *b_nnout;//!

   h1analysis(TTree *tree=0);
   virtual ~h1analysis() { }
   void    Reset();

   int     Version() const {return 1;}
   void    Begin(TTree *tree);
   void    SlaveBegin(TTree *tree);
   void    Init(TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Long64_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    SlaveTerminate();
   void    Terminate();

   ClassDef(h1analysis,2);

};

#endif


//_____________________________________________________________________
h1analysis::h1analysis(TTree * /*tree*/)
{
  // Constructor

   Reset();
}
//_____________________________________________________________________
void h1analysis::Reset()
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

//_____________________________________________________________________
void h1analysis::Init(TTree *tree)
{
   //   Set branch addresses

   Info("Init","tree: %p", tree);

   if (tree == 0) return;
   fChain    = tree;

   fChain->SetBranchAddress("nrun",&nrun,&(b_nrun) );
   fChain->SetBranchAddress("nevent",&nevent,&(b_nevent) );
   fChain->SetBranchAddress("nentry",&nentry,&(b_nentry) );
   fChain->SetBranchAddress("trelem",trelem,&(b_trelem) );
   fChain->SetBranchAddress("subtr",subtr,&(b_subtr) );
   fChain->SetBranchAddress("rawtr",rawtr,&(b_rawtr) );
   fChain->SetBranchAddress("L4subtr",L4subtr,&(b_L4subtr) );
   fChain->SetBranchAddress("L5class",L5class,&(b_L5class) );
   fChain->SetBranchAddress("E33",&E33,&(b_E33) );
   fChain->SetBranchAddress("de33",&de33,&(b_de33) );
   fChain->SetBranchAddress("x33",&x33,&(b_x33) );
   fChain->SetBranchAddress("dx33",&dx33,&(b_dx33) );
   fChain->SetBranchAddress("y33",&y33,&(b_y33) );
   fChain->SetBranchAddress("dy33",&dy33,&(b_dy33) );
   fChain->SetBranchAddress("E44",&E44,&(b_E44) );
   fChain->SetBranchAddress("de44",&de44,&(b_de44) );
   fChain->SetBranchAddress("x44",&x44,&(b_x44) );
   fChain->SetBranchAddress("dx44",&dx44,&(b_dx44) );
   fChain->SetBranchAddress("y44",&y44,&(b_y44) );
   fChain->SetBranchAddress("dy44",&dy44,&(b_dy44) );
   fChain->SetBranchAddress("Ept",&Ept,&(b_Ept) );
   fChain->SetBranchAddress("dept",&dept,&(b_dept) );
   fChain->SetBranchAddress("xpt",&xpt,&(b_xpt) );
   fChain->SetBranchAddress("dxpt",&dxpt,&(b_dxpt) );
   fChain->SetBranchAddress("ypt",&ypt,&(b_ypt) );
   fChain->SetBranchAddress("dypt",&dypt,&(b_dypt) );
   fChain->SetBranchAddress("pelec",pelec,&(b_pelec) );
   fChain->SetBranchAddress("flagelec",&flagelec,&(b_flagelec) );
   fChain->SetBranchAddress("xeelec",&xeelec,&(b_xeelec) );
   fChain->SetBranchAddress("yeelec",&yeelec,&(b_yeelec) );
   fChain->SetBranchAddress("Q2eelec",&Q2eelec,&(b_Q2eelec) );
   fChain->SetBranchAddress("nelec",&nelec,&(b_nelec) );
   fChain->SetBranchAddress("Eelec",Eelec,&(b_Eelec) );
   fChain->SetBranchAddress("thetelec",thetelec,&(b_thetelec) );
   fChain->SetBranchAddress("phielec",phielec,&(b_phielec) );
   fChain->SetBranchAddress("xelec",xelec,&(b_xelec) );
   fChain->SetBranchAddress("Q2elec",Q2elec,&(b_Q2elec) );
   fChain->SetBranchAddress("xsigma",xsigma,&(b_xsigma) );
   fChain->SetBranchAddress("Q2sigma",Q2sigma,&(b_Q2sigma) );
   fChain->SetBranchAddress("sumc",sumc,&(b_sumc) );
   fChain->SetBranchAddress("sumetc",&sumetc,&(b_sumetc) );
   fChain->SetBranchAddress("yjbc",&yjbc,&(b_yjbc) );
   fChain->SetBranchAddress("Q2jbc",&Q2jbc,&(b_Q2jbc) );
   fChain->SetBranchAddress("sumct",sumct,&(b_sumct) );
   fChain->SetBranchAddress("sumetct",&sumetct,&(b_sumetct) );
   fChain->SetBranchAddress("yjbct",&yjbct,&(b_yjbct) );
   fChain->SetBranchAddress("Q2jbct",&Q2jbct,&(b_Q2jbct) );
   fChain->SetBranchAddress("Ebeamel",&Ebeamel,&(b_Ebeamel) );
   fChain->SetBranchAddress("Ebeampr",&Ebeampr,&(b_Ebeampr) );
   fChain->SetBranchAddress("pvtx_d",pvtx_d,&(b_pvtx_d) );
   fChain->SetBranchAddress("cpvtx_d",cpvtx_d,&(b_cpvtx_d) );
   fChain->SetBranchAddress("pvtx_t",pvtx_t,&(b_pvtx_t) );
   fChain->SetBranchAddress("cpvtx_t",cpvtx_t,&(b_cpvtx_t) );
   fChain->SetBranchAddress("ntrkxy_t",&ntrkxy_t,&(b_ntrkxy_t) );
   fChain->SetBranchAddress("prbxy_t",&prbxy_t,&(b_prbxy_t) );
   fChain->SetBranchAddress("ntrkz_t",&ntrkz_t,&(b_ntrkz_t) );
   fChain->SetBranchAddress("prbz_t",&prbz_t,&(b_prbz_t) );
   fChain->SetBranchAddress("nds",&nds,&(b_nds) );
   fChain->SetBranchAddress("rankds",&rankds,&(b_rankds) );
   fChain->SetBranchAddress("qds",&qds,&(b_qds) );
   fChain->SetBranchAddress("pds_d",pds_d,&(b_pds_d) );
   fChain->SetBranchAddress("ptds_d",&ptds_d,&(b_ptds_d) );
   fChain->SetBranchAddress("etads_d",&etads_d,&(b_etads_d) );
   fChain->SetBranchAddress("dm_d",&dm_d,&(b_dm_d) );
   fChain->SetBranchAddress("ddm_d",&ddm_d,&(b_ddm_d) );
   fChain->SetBranchAddress("pds_t",pds_t,&(b_pds_t) );
   fChain->SetBranchAddress("dm_t",&dm_t,&(b_dm_t) );
   fChain->SetBranchAddress("ddm_t",&ddm_t,&(b_ddm_t) );
   fChain->SetBranchAddress("ik",&ik,&(b_ik) );
   fChain->SetBranchAddress("ipi",&ipi,&(b_ipi) );
   fChain->SetBranchAddress("ipis",&ipis,&(b_ipis) );
   fChain->SetBranchAddress("pd0_d",pd0_d,&(b_pd0_d) );
   fChain->SetBranchAddress("ptd0_d",&ptd0_d,&(b_ptd0_d) );
   fChain->SetBranchAddress("etad0_d",&etad0_d,&(b_etad0_d) );
   fChain->SetBranchAddress("md0_d",&md0_d,&(b_md0_d) );
   fChain->SetBranchAddress("dmd0_d",&dmd0_d,&(b_dmd0_d) );
   fChain->SetBranchAddress("pd0_t",pd0_t,&(b_pd0_t) );
   fChain->SetBranchAddress("md0_t",&md0_t,&(b_md0_t) );
   fChain->SetBranchAddress("dmd0_t",&dmd0_t,&(b_dmd0_t) );
   fChain->SetBranchAddress("pk_r",pk_r,&(b_pk_r) );
   fChain->SetBranchAddress("ppi_r",ppi_r,&(b_ppi_r) );
   fChain->SetBranchAddress("pd0_r",pd0_r,&(b_pd0_r) );
   fChain->SetBranchAddress("md0_r",&md0_r,&(b_md0_r) );
   fChain->SetBranchAddress("Vtxd0_r",Vtxd0_r,&(b_Vtxd0_r) );
   fChain->SetBranchAddress("cvtxd0_r",cvtxd0_r,&(b_cvtxd0_r) );
   fChain->SetBranchAddress("dxy_r",&dxy_r,&(b_dxy_r) );
   fChain->SetBranchAddress("dz_r",&dz_r,&(b_dz_r) );
   fChain->SetBranchAddress("psi_r",&psi_r,&(b_psi_r) );
   fChain->SetBranchAddress("rd0_d",&rd0_d,&(b_rd0_d) );
   fChain->SetBranchAddress("drd0_d",&drd0_d,&(b_drd0_d) );
   fChain->SetBranchAddress("rpd0_d",&rpd0_d,&(b_rpd0_d) );
   fChain->SetBranchAddress("drpd0_d",&drpd0_d,&(b_drpd0_d) );
   fChain->SetBranchAddress("rd0_t",&rd0_t,&(b_rd0_t) );
   fChain->SetBranchAddress("drd0_t",&drd0_t,&(b_drd0_t) );
   fChain->SetBranchAddress("rpd0_t",&rpd0_t,&(b_rpd0_t) );
   fChain->SetBranchAddress("drpd0_t",&drpd0_t,&(b_drpd0_t) );
   fChain->SetBranchAddress("rd0_dt",&rd0_dt,&(b_rd0_dt) );
   fChain->SetBranchAddress("drd0_dt",&drd0_dt,&(b_drd0_dt) );
   fChain->SetBranchAddress("prbr_dt",&prbr_dt,&(b_prbr_dt) );
   fChain->SetBranchAddress("prbz_dt",&prbz_dt,&(b_prbz_dt) );
   fChain->SetBranchAddress("rd0_tt",&rd0_tt,&(b_rd0_tt) );
   fChain->SetBranchAddress("drd0_tt",&drd0_tt,&(b_drd0_tt) );
   fChain->SetBranchAddress("prbr_tt",&prbr_tt,&(b_prbr_tt) );
   fChain->SetBranchAddress("prbz_tt",&prbz_tt,&(b_prbz_tt) );
   fChain->SetBranchAddress("ijetd0",&ijetd0,&(b_ijetd0) );
   fChain->SetBranchAddress("ptr3d0_j",&ptr3d0_j,&(b_ptr3d0_j) );
   fChain->SetBranchAddress("ptr2d0_j",&ptr2d0_j,&(b_ptr2d0_j) );
   fChain->SetBranchAddress("ptr3d0_3",&ptr3d0_3,&(b_ptr3d0_3) );
   fChain->SetBranchAddress("ptr2d0_3",&ptr2d0_3,&(b_ptr2d0_3) );
   fChain->SetBranchAddress("ptr2d0_2",&ptr2d0_2,&(b_ptr2d0_2) );
   fChain->SetBranchAddress("Mimpds_r",&Mimpds_r,&(b_Mimpds_r) );
   fChain->SetBranchAddress("Mimpbk_r",&Mimpbk_r,&(b_Mimpbk_r) );
   fChain->SetBranchAddress("ntracks",&ntracks,&(b_ntracks) );
   fChain->SetBranchAddress("pt",pt,&(b_pt) );
   fChain->SetBranchAddress("kappa",kappa,&(b_kappa) );
   fChain->SetBranchAddress("phi",phi,&(b_phi) );
   fChain->SetBranchAddress("theta",theta,&(b_theta) );
   fChain->SetBranchAddress("dca",dca,&(b_dca) );
   fChain->SetBranchAddress("z0",z0,&(b_z0) );
   fChain->SetBranchAddress("covar",covar,&(b_covar) );
   fChain->SetBranchAddress("nhitrp",nhitrp,&(b_nhitrp) );
   fChain->SetBranchAddress("prbrp",prbrp,&(b_prbrp) );
   fChain->SetBranchAddress("nhitz",nhitz,&(b_nhitz) );
   fChain->SetBranchAddress("prbz",prbz,&(b_prbz) );
   fChain->SetBranchAddress("rstart",rstart,&(b_rstart) );
   fChain->SetBranchAddress("rend",rend,&(b_rend) );
   fChain->SetBranchAddress("lhk",lhk,&(b_lhk) );
   fChain->SetBranchAddress("lhpi",lhpi,&(b_lhpi) );
   fChain->SetBranchAddress("nlhk",nlhk,&(b_nlhk) );
   fChain->SetBranchAddress("nlhpi",nlhpi,&(b_nlhpi) );
   fChain->SetBranchAddress("dca_d",dca_d,&(b_dca_d) );
   fChain->SetBranchAddress("ddca_d",ddca_d,&(b_ddca_d) );
   fChain->SetBranchAddress("dca_t",dca_t,&(b_dca_t) );
   fChain->SetBranchAddress("ddca_t",ddca_t,&(b_ddca_t) );
   fChain->SetBranchAddress("muqual",muqual,&(b_muqual) );
   fChain->SetBranchAddress("imu",&imu,&(b_imu) );
   fChain->SetBranchAddress("imufe",&imufe,&(b_imufe) );
   fChain->SetBranchAddress("njets",&njets,&(b_njets) );
   fChain->SetBranchAddress("E_j",E_j,&(b_E_j) );
   fChain->SetBranchAddress("pt_j",pt_j,&(b_pt_j) );
   fChain->SetBranchAddress("theta_j",theta_j,&(b_theta_j) );
   fChain->SetBranchAddress("eta_j",eta_j,&(b_eta_j) );
   fChain->SetBranchAddress("phi_j",phi_j,&(b_phi_j) );
   fChain->SetBranchAddress("m_j",m_j,&(b_m_j) );
   fChain->SetBranchAddress("thrust",&thrust,&(b_thrust) );
   fChain->SetBranchAddress("pthrust",pthrust,&(b_pthrust) );
   fChain->SetBranchAddress("thrust2",&thrust2,&(b_thrust2) );
   fChain->SetBranchAddress("pthrust2",pthrust2,&(b_pthrust2) );
   fChain->SetBranchAddress("spher",&spher,&(b_spher) );
   fChain->SetBranchAddress("aplan",&aplan,&(b_aplan) );
   fChain->SetBranchAddress("plan",&plan,&(b_plan) );
   fChain->SetBranchAddress("nnout",&nnout,&(b_nnout) );
}

//_____________________________________________________________________
Bool_t h1analysis::Notify()
{
//   called when loading a new file
//   get branch pointers

   Info("Notify","processing file: %s",fChain->GetCurrentFile()->GetName());

   if (elist && fChain) {
      if (fillList) {
         elist->SetTree(fChain);
      } else if (useList) {
         fChain->SetEntryList(elist);
      }
   }
   return kTRUE;
}
