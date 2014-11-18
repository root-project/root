/////////////////////////////////////////////////////////////////////////
//   This class has been automatically generated 
//   (at Mon Nov  4 15:57:25 2013 by ROOT version 5.34/10)
//   from TChain h42/
/////////////////////////////////////////////////////////////////////////


#ifndef generatedSel_h
#define generatedSel_h

// System Headers needed by the proxy
#if defined(__CINT__) && !defined(__MAKECINT__)
   #define ROOT_Rtypes
   #define ROOT_TError
#endif
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TPad.h>
#include <TH1.h>
#include <TSelector.h>
#include <TBranchProxy.h>
#include <TBranchProxyDirector.h>
#include <TBranchProxyTemplate.h>
#include <TFriendProxy.h>
using namespace ROOT;

// forward declarations needed by this particular proxy


// Header needed by this particular proxy
#include "h1analysisProxy.h"


class h1analysisProxy_Interface {
   // This class defines the list of methods that are directly used by generatedSel,
   // and that can be overloaded in the user's script
public:
   void h1analysisProxy_Begin(TTree*) {}
   void h1analysisProxy_SlaveBegin(TTree*) {}
   Bool_t h1analysisProxy_Notify() { return kTRUE; }
   Bool_t h1analysisProxy_Process(Long64_t) { return kTRUE; }
   void h1analysisProxy_SlaveTerminate() {}
   void h1analysisProxy_Terminate() {}
};


class generatedSel : public TSelector, public h1analysisProxy_Interface {
public :
   TTree          *fChain;         //!pointer to the analyzed TTree or TChain
   TH1            *htemp;          //!pointer to the histogram
   TBranchProxyDirector fDirector; //!Manages the proxys

   // Optional User methods
   TClass         *fClass;    // Pointer to this class's description

   // Proxy for each of the branches, leaves and friends of the tree
   TIntProxy                              nrun;
   TIntProxy                              nevent;
   TIntProxy                              nentry;
   TArrayUCharProxy                       trelem;
   TArrayUCharProxy                       subtr;
   TArrayUCharProxy                       rawtr;
   TArrayUCharProxy                       L4subtr;
   TArrayUCharProxy                       L5class;
   TFloatProxy                            E33;
   TFloatProxy                            de33;
   TFloatProxy                            x33;
   TFloatProxy                            dx33;
   TFloatProxy                            y33;
   TFloatProxy                            dy33;
   TFloatProxy                            E44;
   TFloatProxy                            de44;
   TFloatProxy                            x44;
   TFloatProxy                            dx44;
   TFloatProxy                            y44;
   TFloatProxy                            dy44;
   TFloatProxy                            Ept;
   TFloatProxy                            dept;
   TFloatProxy                            xpt;
   TFloatProxy                            dxpt;
   TFloatProxy                            ypt;
   TFloatProxy                            dypt;
   TArrayFloatProxy                       pelec;
   TIntProxy                              flagelec;
   TFloatProxy                            xeelec;
   TFloatProxy                            yeelec;
   TFloatProxy                            Q2eelec;
   TIntProxy                              nelec;
   TArrayFloatProxy                       Eelec;
   TArrayFloatProxy                       thetelec;
   TArrayFloatProxy                       phielec;
   TArrayFloatProxy                       xelec;
   TArrayFloatProxy                       Q2elec;
   TArrayFloatProxy                       xsigma;
   TArrayFloatProxy                       Q2sigma;
   TArrayFloatProxy                       sumc;
   TFloatProxy                            sumetc;
   TFloatProxy                            yjbc;
   TFloatProxy                            Q2jbc;
   TArrayFloatProxy                       sumct;
   TFloatProxy                            sumetct;
   TFloatProxy                            yjbct;
   TFloatProxy                            Q2jbct;
   TFloatProxy                            Ebeamel;
   TFloatProxy                            Ebeampr;
   TArrayFloatProxy                       pvtx_d;
   TArrayFloatProxy                       cpvtx_d;
   TArrayFloatProxy                       pvtx_t;
   TArrayFloatProxy                       cpvtx_t;
   TIntProxy                              ntrkxy_t;
   TFloatProxy                            prbxy_t;
   TIntProxy                              ntrkz_t;
   TFloatProxy                            prbz_t;
   TIntProxy                              nds;
   TIntProxy                              rankds;
   TIntProxy                              qds;
   TArrayFloatProxy                       pds_d;
   TFloatProxy                            ptds_d;
   TFloatProxy                            etads_d;
   TFloatProxy                            dm_d;
   TFloatProxy                            ddm_d;
   TArrayFloatProxy                       pds_t;
   TFloatProxy                            dm_t;
   TFloatProxy                            ddm_t;
   TIntProxy                              ik;
   TIntProxy                              ipi;
   TIntProxy                              ipis;
   TArrayFloatProxy                       pd0_d;
   TFloatProxy                            ptd0_d;
   TFloatProxy                            etad0_d;
   TFloatProxy                            md0_d;
   TFloatProxy                            dmd0_d;
   TArrayFloatProxy                       pd0_t;
   TFloatProxy                            md0_t;
   TFloatProxy                            dmd0_t;
   TArrayFloatProxy                       pk_r;
   TArrayFloatProxy                       ppi_r;
   TArrayFloatProxy                       pd0_r;
   TFloatProxy                            md0_r;
   TArrayFloatProxy                       Vtxd0_r;
   TArrayFloatProxy                       cvtxd0_r;
   TFloatProxy                            dxy_r;
   TFloatProxy                            dz_r;
   TFloatProxy                            psi_r;
   TFloatProxy                            rd0_d;
   TFloatProxy                            drd0_d;
   TFloatProxy                            rpd0_d;
   TFloatProxy                            drpd0_d;
   TFloatProxy                            rd0_t;
   TFloatProxy                            drd0_t;
   TFloatProxy                            rpd0_t;
   TFloatProxy                            drpd0_t;
   TFloatProxy                            rd0_dt;
   TFloatProxy                            drd0_dt;
   TFloatProxy                            prbr_dt;
   TFloatProxy                            prbz_dt;
   TFloatProxy                            rd0_tt;
   TFloatProxy                            drd0_tt;
   TFloatProxy                            prbr_tt;
   TFloatProxy                            prbz_tt;
   TIntProxy                              ijetd0;
   TFloatProxy                            ptr3d0_j;
   TFloatProxy                            ptr2d0_j;
   TFloatProxy                            ptr3d0_3;
   TFloatProxy                            ptr2d0_3;
   TFloatProxy                            ptr2d0_2;
   TFloatProxy                            Mimpds_r;
   TFloatProxy                            Mimpbk_r;
   TIntProxy                              ntracks;
   TArrayFloatProxy                       pt;
   TArrayFloatProxy                       kappa;
   TArrayFloatProxy                       phi;
   TArrayFloatProxy                       theta;
   TArrayFloatProxy                       dca;
   TArrayFloatProxy                       z0;
   TArrayProxy<TArrayType<Float_t,15> >   covar;
   TArrayIntProxy                         nhitrp;
   TArrayFloatProxy                       prbrp;
   TArrayIntProxy                         nhitz;
   TArrayFloatProxy                       prbz;
   TArrayFloatProxy                       rstart;
   TArrayFloatProxy                       rend;
   TArrayFloatProxy                       lhk;
   TArrayFloatProxy                       lhpi;
   TArrayFloatProxy                       nlhk;
   TArrayFloatProxy                       nlhpi;
   TArrayFloatProxy                       dca_d;
   TArrayFloatProxy                       ddca_d;
   TArrayFloatProxy                       dca_t;
   TArrayFloatProxy                       ddca_t;
   TArrayIntProxy                         muqual;
   TIntProxy                              imu;
   TIntProxy                              imufe;
   TIntProxy                              njets;
   TArrayFloatProxy                       E_j;
   TArrayFloatProxy                       pt_j;
   TArrayFloatProxy                       theta_j;
   TArrayFloatProxy                       eta_j;
   TArrayFloatProxy                       phi_j;
   TArrayFloatProxy                       m_j;
   TFloatProxy                            thrust;
   TArrayFloatProxy                       pthrust;
   TFloatProxy                            thrust2;
   TArrayFloatProxy                       pthrust2;
   TFloatProxy                            spher;
   TFloatProxy                            aplan;
   TFloatProxy                            plan;
   TArrayFloatProxy                       nnout;


   generatedSel(TTree *tree=0) : 
      fChain(0),
      htemp(0),
      fDirector(tree,-1),
      fClass                (TClass::GetClass("generatedSel")),
      nrun                                  (&fDirector,"nrun"),
      nevent                                (&fDirector,"nevent"),
      nentry                                (&fDirector,"nentry"),
      trelem                                (&fDirector,"trelem"),
      subtr                                 (&fDirector,"subtr"),
      rawtr                                 (&fDirector,"rawtr"),
      L4subtr                               (&fDirector,"L4subtr"),
      L5class                               (&fDirector,"L5class"),
      E33                                   (&fDirector,"E33"),
      de33                                  (&fDirector,"de33"),
      x33                                   (&fDirector,"x33"),
      dx33                                  (&fDirector,"dx33"),
      y33                                   (&fDirector,"y33"),
      dy33                                  (&fDirector,"dy33"),
      E44                                   (&fDirector,"E44"),
      de44                                  (&fDirector,"de44"),
      x44                                   (&fDirector,"x44"),
      dx44                                  (&fDirector,"dx44"),
      y44                                   (&fDirector,"y44"),
      dy44                                  (&fDirector,"dy44"),
      Ept                                   (&fDirector,"Ept"),
      dept                                  (&fDirector,"dept"),
      xpt                                   (&fDirector,"xpt"),
      dxpt                                  (&fDirector,"dxpt"),
      ypt                                   (&fDirector,"ypt"),
      dypt                                  (&fDirector,"dypt"),
      pelec                                 (&fDirector,"pelec"),
      flagelec                              (&fDirector,"flagelec"),
      xeelec                                (&fDirector,"xeelec"),
      yeelec                                (&fDirector,"yeelec"),
      Q2eelec                               (&fDirector,"Q2eelec"),
      nelec                                 (&fDirector,"nelec"),
      Eelec                                 (&fDirector,"Eelec"),
      thetelec                              (&fDirector,"thetelec"),
      phielec                               (&fDirector,"phielec"),
      xelec                                 (&fDirector,"xelec"),
      Q2elec                                (&fDirector,"Q2elec"),
      xsigma                                (&fDirector,"xsigma"),
      Q2sigma                               (&fDirector,"Q2sigma"),
      sumc                                  (&fDirector,"sumc"),
      sumetc                                (&fDirector,"sumetc"),
      yjbc                                  (&fDirector,"yjbc"),
      Q2jbc                                 (&fDirector,"Q2jbc"),
      sumct                                 (&fDirector,"sumct"),
      sumetct                               (&fDirector,"sumetct"),
      yjbct                                 (&fDirector,"yjbct"),
      Q2jbct                                (&fDirector,"Q2jbct"),
      Ebeamel                               (&fDirector,"Ebeamel"),
      Ebeampr                               (&fDirector,"Ebeampr"),
      pvtx_d                                (&fDirector,"pvtx_d"),
      cpvtx_d                               (&fDirector,"cpvtx_d"),
      pvtx_t                                (&fDirector,"pvtx_t"),
      cpvtx_t                               (&fDirector,"cpvtx_t"),
      ntrkxy_t                              (&fDirector,"ntrkxy_t"),
      prbxy_t                               (&fDirector,"prbxy_t"),
      ntrkz_t                               (&fDirector,"ntrkz_t"),
      prbz_t                                (&fDirector,"prbz_t"),
      nds                                   (&fDirector,"nds"),
      rankds                                (&fDirector,"rankds"),
      qds                                   (&fDirector,"qds"),
      pds_d                                 (&fDirector,"pds_d"),
      ptds_d                                (&fDirector,"ptds_d"),
      etads_d                               (&fDirector,"etads_d"),
      dm_d                                  (&fDirector,"dm_d"),
      ddm_d                                 (&fDirector,"ddm_d"),
      pds_t                                 (&fDirector,"pds_t"),
      dm_t                                  (&fDirector,"dm_t"),
      ddm_t                                 (&fDirector,"ddm_t"),
      ik                                    (&fDirector,"ik"),
      ipi                                   (&fDirector,"ipi"),
      ipis                                  (&fDirector,"ipis"),
      pd0_d                                 (&fDirector,"pd0_d"),
      ptd0_d                                (&fDirector,"ptd0_d"),
      etad0_d                               (&fDirector,"etad0_d"),
      md0_d                                 (&fDirector,"md0_d"),
      dmd0_d                                (&fDirector,"dmd0_d"),
      pd0_t                                 (&fDirector,"pd0_t"),
      md0_t                                 (&fDirector,"md0_t"),
      dmd0_t                                (&fDirector,"dmd0_t"),
      pk_r                                  (&fDirector,"pk_r"),
      ppi_r                                 (&fDirector,"ppi_r"),
      pd0_r                                 (&fDirector,"pd0_r"),
      md0_r                                 (&fDirector,"md0_r"),
      Vtxd0_r                               (&fDirector,"Vtxd0_r"),
      cvtxd0_r                              (&fDirector,"cvtxd0_r"),
      dxy_r                                 (&fDirector,"dxy_r"),
      dz_r                                  (&fDirector,"dz_r"),
      psi_r                                 (&fDirector,"psi_r"),
      rd0_d                                 (&fDirector,"rd0_d"),
      drd0_d                                (&fDirector,"drd0_d"),
      rpd0_d                                (&fDirector,"rpd0_d"),
      drpd0_d                               (&fDirector,"drpd0_d"),
      rd0_t                                 (&fDirector,"rd0_t"),
      drd0_t                                (&fDirector,"drd0_t"),
      rpd0_t                                (&fDirector,"rpd0_t"),
      drpd0_t                               (&fDirector,"drpd0_t"),
      rd0_dt                                (&fDirector,"rd0_dt"),
      drd0_dt                               (&fDirector,"drd0_dt"),
      prbr_dt                               (&fDirector,"prbr_dt"),
      prbz_dt                               (&fDirector,"prbz_dt"),
      rd0_tt                                (&fDirector,"rd0_tt"),
      drd0_tt                               (&fDirector,"drd0_tt"),
      prbr_tt                               (&fDirector,"prbr_tt"),
      prbz_tt                               (&fDirector,"prbz_tt"),
      ijetd0                                (&fDirector,"ijetd0"),
      ptr3d0_j                              (&fDirector,"ptr3d0_j"),
      ptr2d0_j                              (&fDirector,"ptr2d0_j"),
      ptr3d0_3                              (&fDirector,"ptr3d0_3"),
      ptr2d0_3                              (&fDirector,"ptr2d0_3"),
      ptr2d0_2                              (&fDirector,"ptr2d0_2"),
      Mimpds_r                              (&fDirector,"Mimpds_r"),
      Mimpbk_r                              (&fDirector,"Mimpbk_r"),
      ntracks                               (&fDirector,"ntracks"),
      pt                                    (&fDirector,"pt"),
      kappa                                 (&fDirector,"kappa"),
      phi                                   (&fDirector,"phi"),
      theta                                 (&fDirector,"theta"),
      dca                                   (&fDirector,"dca"),
      z0                                    (&fDirector,"z0"),
      covar                                 (&fDirector,"covar"),
      nhitrp                                (&fDirector,"nhitrp"),
      prbrp                                 (&fDirector,"prbrp"),
      nhitz                                 (&fDirector,"nhitz"),
      prbz                                  (&fDirector,"prbz"),
      rstart                                (&fDirector,"rstart"),
      rend                                  (&fDirector,"rend"),
      lhk                                   (&fDirector,"lhk"),
      lhpi                                  (&fDirector,"lhpi"),
      nlhk                                  (&fDirector,"nlhk"),
      nlhpi                                 (&fDirector,"nlhpi"),
      dca_d                                 (&fDirector,"dca_d"),
      ddca_d                                (&fDirector,"ddca_d"),
      dca_t                                 (&fDirector,"dca_t"),
      ddca_t                                (&fDirector,"ddca_t"),
      muqual                                (&fDirector,"muqual"),
      imu                                   (&fDirector,"imu"),
      imufe                                 (&fDirector,"imufe"),
      njets                                 (&fDirector,"njets"),
      E_j                                   (&fDirector,"E_j"),
      pt_j                                  (&fDirector,"pt_j"),
      theta_j                               (&fDirector,"theta_j"),
      eta_j                                 (&fDirector,"eta_j"),
      phi_j                                 (&fDirector,"phi_j"),
      m_j                                   (&fDirector,"m_j"),
      thrust                                (&fDirector,"thrust"),
      pthrust                               (&fDirector,"pthrust"),
      thrust2                               (&fDirector,"thrust2"),
      pthrust2                              (&fDirector,"pthrust2"),
      spher                                 (&fDirector,"spher"),
      aplan                                 (&fDirector,"aplan"),
      plan                                  (&fDirector,"plan"),
      nnout                                 (&fDirector,"nnout")
      { }
   ~generatedSel();
   Int_t   Version() const {return 1;}
   void    Begin(::TTree *tree);
   void    SlaveBegin(::TTree *tree);
   void    Init(::TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Long64_t entry);
   void    SlaveTerminate();
   void    Terminate();

   ClassDef(generatedSel,0);


//inject the user's code
#include "h1analysisProxy.C"
};

#endif


#ifdef __MAKECINT__
#pragma link C++ class generatedSel;
#endif


inline generatedSel::~generatedSel() {
   // destructor. Clean up helpers.

}

inline void generatedSel::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain = tree;
   fDirector.SetTree(fChain);
   if (htemp == 0) {
      htemp = fDirector.CreateHistogram(GetOption());
      htemp->SetTitle("h1analysisProxy.C");
      fObject = htemp;
   }
}

Bool_t generatedSel::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   fDirector.SetTree(fChain);
   h1analysisProxy_Notify();
   
   return kTRUE;
}
   

inline void generatedSel::Begin(TTree *tree)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   h1analysisProxy_Begin(tree);

}

inline void generatedSel::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   Init(tree);

   h1analysisProxy_SlaveBegin(tree);

}

inline Bool_t generatedSel::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either TTree::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.

   // WARNING when a selector is used with a TChain, you must use
   //  the pointer to the current TTree to call GetEntry(entry).
   //  The entry is always the local entry number in the current tree.
   //  Assuming that fChain is the pointer to the TChain being processed,
   //  use fChain->GetTree()->GetEntry(entry).


   fDirector.SetReadEntry(entry);
   htemp->Fill(h1analysisProxy());
   h1analysisProxy_Process(entry);
   return kTRUE;

}

inline void generatedSel::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.
   h1analysisProxy_SlaveTerminate();
}

inline void generatedSel::Terminate()
{
   // Function called at the end of the event loop.
   htemp = (TH1*)fObject;
   Int_t drawflag = (htemp && htemp->GetEntries()>0);
   
   if (gPad && !drawflag && !fOption.Contains("goff") && !fOption.Contains("same")) {
      gPad->Clear();
   } else {
      if (fOption.Contains("goff")) drawflag = false;
      if (drawflag) htemp->Draw(fOption);
   }
   h1analysisProxy_Terminate();
}
