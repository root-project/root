/////////////////////////////////////////////////////////////////////////
//   This class has been automatically generated 
//   (at Fri Jan 21 16:07:07 2005 by ROOT version 4.03/01)
//   from TTree TopTree/A very simple ROOT tuple tree
//   found on file: toptree.root
/////////////////////////////////////////////////////////////////////////


#ifndef analyzeTop_h
#define analyzeTop_h

// System Headers needed by the proxy
#if defined(__CINT__) && !defined(__MAKECINT__)
   #define ROOT_Rtypes
   #define ROOT_TError
#endif
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelectorDraw.h>
#include <TPad.h>
#include <TH1.h>
#include <TBranchProxy.h>
#include <TBranchProxyDirector.h>
#include <TBranchProxyTemplate.h>
#include <TFriendProxy.h>
#include <TMethodCall.h>

using namespace ROOT;

// forward declarations needed by this particular proxy
class TheEventClass;
class TObject;
class TClonesArray;
class TheObjectClass;
class TheMissingEtClass;
class ThePropertyClass;
class TheEJetsClass;
class TheMuJetsClass;
class TheEMUClass;
class TheDiMuonClass;
class TheDiEMClass;
class TheAllJetsClass;


// Header needed by this particular proxy
#include "TObject.h"
#include "TClonesArray.h"


class analyzeTop : public TSelector {
   public :
   TTree          *fChain;    //!pointer to the analyzed TTree or TChain
   TSelectorDraw  *fHelper;   //!helper class to create the default histogram
   TList          *fInput;    //!input list of the helper
   TH1            *htemp;     //!pointer to the histogram
   TBranchProxyDirector  fDirector; //!Manages the proxys

   // Optional User methods
   TClass         *fClass;    // Pointer to this class's description
   TMethodCall     fBeginMethod;
   TMethodCall     fSlaveBeginMethod;
   TMethodCall     fNotifyMethod;
   TMethodCall     fProcessMethod;
   TMethodCall     fSlaveTerminateMethod;
   TMethodCall     fTerminateMethod;

   // Wrapper class for each unwounded class
   struct TPx_TObject
   {
      TPx_TObject(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix    (top,mid),
         obj         (director, top, mid),
         fUniqueID   (director, "fUniqueID"),
         fBits       (director, "fBits")
      {};
      TPx_TObject(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix    (""),
         obj         (director, parent, membername),
         fUniqueID   (director, "fUniqueID"),
         fBits       (director, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator->() { return obj.ptr(); }
      TObjProxy<TObject > obj;

      TUIntProxy   fUniqueID;
      TUIntProxy   fBits;
   };
   struct TClaPx_TObject
   {
      TClaPx_TObject(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheExecutableVersionClass
      : public TClaPx_TObject
   {
      TClaPx_TheExecutableVersionClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject     (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         executable         (director, ffPrefix, "executable[5]"),
         version            (director, ffPrefix, "version[5]")
      {};
      TClaPx_TheExecutableVersionClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject     (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         executable         (director, ffPrefix, "executable[5]"),
         version            (director, ffPrefix, "version[5]")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayIntProxy   executable;
      TClaArrayIntProxy   version;
   };
   struct TClaPx_TObject_1
   {
      TClaPx_TObject_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheL3NameClass
      : public TClaPx_TObject_1
   {
      TClaPx_TheL3NameClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_1   (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         l3name             (director, ffPrefix, "l3name[10]"),
         unbiased           (director, ffPrefix, "unbiased"),
         luminosity         (director, ffPrefix, "luminosity")
      {};
      TClaPx_TheL3NameClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_1   (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         l3name             (director, ffPrefix, "l3name[10]"),
         unbiased           (director, ffPrefix, "unbiased"),
         luminosity         (director, ffPrefix, "luminosity")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayIntProxy   l3name;
      TClaIntProxy        unbiased;
      TClaFloatProxy      luminosity;
   };
   struct TClaPx_TObject_2
   {
      TClaPx_TObject_2(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_2(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheL2NameClass
      : public TClaPx_TObject_2
   {
      TClaPx_TheL2NameClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_2   (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         l2name             (director, ffPrefix, "l2name[10]")
      {};
      TClaPx_TheL2NameClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_2   (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         l2name             (director, ffPrefix, "l2name[10]")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayIntProxy   l2name;
   };
   struct TClaPx_TObject_3
   {
      TClaPx_TObject_3(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_3(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheL1NameClass
      : public TClaPx_TObject_3
   {
      TClaPx_TheL1NameClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_3   (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         l1name             (director, ffPrefix, "l1name[10]")
      {};
      TClaPx_TheL1NameClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_3   (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         l1name             (director, ffPrefix, "l1name[10]")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayIntProxy   l1name;
   };
   struct TClaPx_TObject_4
   {
      TClaPx_TObject_4(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_4(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheIntClass
      : public TClaPx_TObject_4
   {
      TClaPx_TheIntClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_4(director, top, mid),
         ffPrefix      (top,mid),
         obj           (director, top, mid),
         data          (director, ffPrefix, "data")
      {};
      TClaPx_TheIntClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_4(director, parent, membername),
         ffPrefix      (""),
         obj           (director, parent, membername),
         data          (director, ffPrefix, "data")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy   data;
   };
   struct TClaPx_TObject_5
   {
      TClaPx_TObject_5(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_5(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheIntClass_1
      : public TClaPx_TObject_5
   {
      TClaPx_TheIntClass_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_5(director, top, mid),
         ffPrefix      (top,mid),
         obj           (director, top, mid),
         data          (director, ffPrefix, "data")
      {};
      TClaPx_TheIntClass_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_5(director, parent, membername),
         ffPrefix      (""),
         obj           (director, parent, membername),
         data          (director, ffPrefix, "data")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy   data;
   };
   struct TPx_TheEventClass
      : public TPx_TObject
   {
      TPx_TheEventClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject                       (director, top, mid),
         ffPrefix                          (top,mid),
         obj                               (director, top, mid),
         ExecutableVersionArray            (director, "ExecutableVersionArray"),
         Nexecutables                      (director, "Nexecutables"),
         L3NameArray                       (director, "L3NameArray"),
         l3name_n                          (director, "l3name_n"),
         L2NameArray                       (director, "L2NameArray"),
         l2name_n                          (director, "l2name_n"),
         L1NameArray                       (director, "L1NameArray"),
         l1name_n                          (director, "l1name_n"),
         L1PrescalesArray                  (director, "L1PrescalesArray"),
         l1prescales_n                     (director, "l1prescales_n"),
         AndOrTermArray                    (director, "AndOrTermArray"),
         aoterm_n                          (director, "aoterm_n"),
         runnum                            (director, "runnum"),
         evtnum                            (director, "evtnum"),
         ticknum                           (director, "ticknum"),
         lumblk                            (director, "lumblk"),
         solpol                            (director, "solpol"),
         torpol                            (director, "torpol"),
         emptyCrate                        (director, "emptyCrate"),
         coherentNoise                     (director, "coherentNoise"),
         ringOfFire                        (director, "ringOfFire"),
         nojets                            (director, "nojets"),
         vertex                            (director, "vertex[3]"),
         vertexErr                         (director, "vertexErr[6]"),
         vertexNtrack                      (director, "vertexNtrack"),
         vertexNdof                        (director, "vertexNdof"),
         vertexChi2                        (director, "vertexChi2"),
         ht20                              (director, "ht20"),
         ht25                              (director, "ht25"),
         mc_cross                          (director, "mc_cross"),
         mc_wt                             (director, "mc_wt"),
         NewVtx                            (director, "NewVtx[3]"),
         NewVtxErr                         (director, "NewVtxErr[6]"),
         NewVtxTrks                        (director, "NewVtxTrks"),
         NVmbprob                          (director, "NVmbprob"),
         NVSumPt                           (director, "NVSumPt"),
         NVSumLogPt                        (director, "NVSumLogPt"),
         NVHighestPt                       (director, "NVHighestPt"),
         NewVtxChi2                        (director, "NewVtxChi2"),
         NewVtxNdof                        (director, "NewVtxNdof"),
         tottwr                            (director, "tottwr"),
         twr_l1et_lt_n1_em                 (director, "twr_l1et_lt_n1_em"),
         twr_l1et_gt_2_em                  (director, "twr_l1et_gt_2_em"),
         twr_diff_lt_n1_em                 (director, "twr_diff_lt_n1_em"),
         twr_diff_inbt_em                  (director, "twr_diff_inbt_em"),
         twr_diff_gt_1_em                  (director, "twr_diff_gt_1_em"),
         twr_l1et_lt_n1_had                (director, "twr_l1et_lt_n1_had"),
         twr_l1et_gt_2_had                 (director, "twr_l1et_gt_2_had"),
         twr_diff_lt_n1_had                (director, "twr_diff_lt_n1_had"),
         twr_diff_inbt_had                 (director, "twr_diff_inbt_had"),
         twr_diff_gt_1_had                 (director, "twr_diff_gt_1_had"),
         sumdiff_em                        (director, "sumdiff_em"),
         suml1_em                          (director, "suml1_em"),
         maxdiff_em                        (director, "maxdiff_em"),
         twr_rat_lt_05_em                  (director, "twr_rat_lt_05_em"),
         twr_rat_gt_2_em                   (director, "twr_rat_gt_2_em"),
         sumdiff_had                       (director, "sumdiff_had"),
         suml1_had                         (director, "suml1_had"),
         maxdiff_had                       (director, "maxdiff_had"),
         twr_rat_lt_05_had                 (director, "twr_rat_lt_05_had"),
         twr_rat_gt_2_had                  (director, "twr_rat_gt_2_had")
      {};
      TPx_TheEventClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject                       (director, parent, membername),
         ffPrefix                          (""),
         obj                               (director, parent, membername),
         ExecutableVersionArray            (director, "ExecutableVersionArray"),
         Nexecutables                      (director, "Nexecutables"),
         L3NameArray                       (director, "L3NameArray"),
         l3name_n                          (director, "l3name_n"),
         L2NameArray                       (director, "L2NameArray"),
         l2name_n                          (director, "l2name_n"),
         L1NameArray                       (director, "L1NameArray"),
         l1name_n                          (director, "l1name_n"),
         L1PrescalesArray                  (director, "L1PrescalesArray"),
         l1prescales_n                     (director, "l1prescales_n"),
         AndOrTermArray                    (director, "AndOrTermArray"),
         aoterm_n                          (director, "aoterm_n"),
         runnum                            (director, "runnum"),
         evtnum                            (director, "evtnum"),
         ticknum                           (director, "ticknum"),
         lumblk                            (director, "lumblk"),
         solpol                            (director, "solpol"),
         torpol                            (director, "torpol"),
         emptyCrate                        (director, "emptyCrate"),
         coherentNoise                     (director, "coherentNoise"),
         ringOfFire                        (director, "ringOfFire"),
         nojets                            (director, "nojets"),
         vertex                            (director, "vertex[3]"),
         vertexErr                         (director, "vertexErr[6]"),
         vertexNtrack                      (director, "vertexNtrack"),
         vertexNdof                        (director, "vertexNdof"),
         vertexChi2                        (director, "vertexChi2"),
         ht20                              (director, "ht20"),
         ht25                              (director, "ht25"),
         mc_cross                          (director, "mc_cross"),
         mc_wt                             (director, "mc_wt"),
         NewVtx                            (director, "NewVtx[3]"),
         NewVtxErr                         (director, "NewVtxErr[6]"),
         NewVtxTrks                        (director, "NewVtxTrks"),
         NVmbprob                          (director, "NVmbprob"),
         NVSumPt                           (director, "NVSumPt"),
         NVSumLogPt                        (director, "NVSumLogPt"),
         NVHighestPt                       (director, "NVHighestPt"),
         NewVtxChi2                        (director, "NewVtxChi2"),
         NewVtxNdof                        (director, "NewVtxNdof"),
         tottwr                            (director, "tottwr"),
         twr_l1et_lt_n1_em                 (director, "twr_l1et_lt_n1_em"),
         twr_l1et_gt_2_em                  (director, "twr_l1et_gt_2_em"),
         twr_diff_lt_n1_em                 (director, "twr_diff_lt_n1_em"),
         twr_diff_inbt_em                  (director, "twr_diff_inbt_em"),
         twr_diff_gt_1_em                  (director, "twr_diff_gt_1_em"),
         twr_l1et_lt_n1_had                (director, "twr_l1et_lt_n1_had"),
         twr_l1et_gt_2_had                 (director, "twr_l1et_gt_2_had"),
         twr_diff_lt_n1_had                (director, "twr_diff_lt_n1_had"),
         twr_diff_inbt_had                 (director, "twr_diff_inbt_had"),
         twr_diff_gt_1_had                 (director, "twr_diff_gt_1_had"),
         sumdiff_em                        (director, "sumdiff_em"),
         suml1_em                          (director, "suml1_em"),
         maxdiff_em                        (director, "maxdiff_em"),
         twr_rat_lt_05_em                  (director, "twr_rat_lt_05_em"),
         twr_rat_gt_2_em                   (director, "twr_rat_gt_2_em"),
         sumdiff_had                       (director, "sumdiff_had"),
         suml1_had                         (director, "suml1_had"),
         maxdiff_had                       (director, "maxdiff_had"),
         twr_rat_lt_05_had                 (director, "twr_rat_lt_05_had"),
         twr_rat_gt_2_had                  (director, "twr_rat_gt_2_had")
      {};
      TBranchProxyHelper                 ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TClaPx_TheExecutableVersionClass   ExecutableVersionArray;
      TIntProxy                          Nexecutables;
      TClaPx_TheL3NameClass              L3NameArray;
      TIntProxy                          l3name_n;
      TClaPx_TheL2NameClass              L2NameArray;
      TIntProxy                          l2name_n;
      TClaPx_TheL1NameClass              L1NameArray;
      TIntProxy                          l1name_n;
      TClaPx_TheIntClass                 L1PrescalesArray;
      TIntProxy                          l1prescales_n;
      TClaPx_TheIntClass_1               AndOrTermArray;
      TIntProxy                          aoterm_n;
      TIntProxy                          runnum;
      TIntProxy                          evtnum;
      TIntProxy                          ticknum;
      TIntProxy                          lumblk;
      TIntProxy                          solpol;
      TIntProxy                          torpol;
      TIntProxy                          emptyCrate;
      TIntProxy                          coherentNoise;
      TIntProxy                          ringOfFire;
      TIntProxy                          nojets;
      TArrayFloatProxy                   vertex;
      TArrayFloatProxy                   vertexErr;
      TIntProxy                          vertexNtrack;
      TIntProxy                          vertexNdof;
      TFloatProxy                        vertexChi2;
      TFloatProxy                        ht20;
      TFloatProxy                        ht25;
      TFloatProxy                        mc_cross;
      TFloatProxy                        mc_wt;
      TArrayFloatProxy                   NewVtx;
      TArrayFloatProxy                   NewVtxErr;
      TIntProxy                          NewVtxTrks;
      TFloatProxy                        NVmbprob;
      TFloatProxy                        NVSumPt;
      TFloatProxy                        NVSumLogPt;
      TFloatProxy                        NVHighestPt;
      TFloatProxy                        NewVtxChi2;
      TIntProxy                          NewVtxNdof;
      TIntProxy                          tottwr;
      TIntProxy                          twr_l1et_lt_n1_em;
      TIntProxy                          twr_l1et_gt_2_em;
      TIntProxy                          twr_diff_lt_n1_em;
      TIntProxy                          twr_diff_inbt_em;
      TIntProxy                          twr_diff_gt_1_em;
      TIntProxy                          twr_l1et_lt_n1_had;
      TIntProxy                          twr_l1et_gt_2_had;
      TIntProxy                          twr_diff_lt_n1_had;
      TIntProxy                          twr_diff_inbt_had;
      TIntProxy                          twr_diff_gt_1_had;
      TFloatProxy                        sumdiff_em;
      TFloatProxy                        suml1_em;
      TFloatProxy                        maxdiff_em;
      TIntProxy                          twr_rat_lt_05_em;
      TIntProxy                          twr_rat_gt_2_em;
      TFloatProxy                        sumdiff_had;
      TFloatProxy                        suml1_had;
      TFloatProxy                        maxdiff_had;
      TIntProxy                          twr_rat_lt_05_had;
      TIntProxy                          twr_rat_gt_2_had;
   };
   struct TClaPx_TObject_6
   {
      TClaPx_TObject_6(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_6(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheIntClass_2
      : public TClaPx_TObject_6
   {
      TClaPx_TheIntClass_2(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_6(director, top, mid),
         ffPrefix      (top,mid),
         obj           (director, top, mid),
         data          (director, ffPrefix, "data")
      {};
      TClaPx_TheIntClass_2(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_6(director, parent, membername),
         ffPrefix      (""),
         obj           (director, parent, membername),
         data          (director, ffPrefix, "data")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy   data;
   };
   struct TClaPx_TObject_7
   {
      TClaPx_TObject_7(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_7(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheMuonClass
      : public TClaPx_TObject_7
   {
      TClaPx_TheMuonClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_7     (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         pt                   (director, ffPrefix, "pt"),
         ptcorr               (director, ffPrefix, "ptcorr"),
         ptglobal             (director, ffPrefix, "ptglobal"),
         qptloc               (director, ffPrefix, "qptloc"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         pxloc                (director, ffPrefix, "pxloc"),
         pyloc                (director, ffPrefix, "pyloc"),
         pzloc                (director, ffPrefix, "pzloc"),
         z0                   (director, ffPrefix, "z0"),
         dca                  (director, ffPrefix, "dca[2]"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         etaloc               (director, ffPrefix, "etaloc"),
         philoc               (director, ffPrefix, "philoc"),
         eloss                (director, ffPrefix, "eloss"),
         drjet                (director, ffPrefix, "drjet"),
         dedx                 (director, ffPrefix, "dedx"),
         ring                 (director, ffPrefix, "ring"),
         halo                 (director, ffPrefix, "halo"),
         chi2                 (director, ffPrefix, "chi2"),
         chi2loc              (director, ffPrefix, "chi2loc"),
         sctime_a             (director, ffPrefix, "sctime_a"),
         sctime_b             (director, ffPrefix, "sctime_b"),
         mtcsig               (director, ffPrefix, "mtcsig"),
         mtceta               (director, ffPrefix, "mtceta"),
         mtcphi               (director, ffPrefix, "mtcphi"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         trkdr                (director, ffPrefix, "trkdr"),
         trkcone_pt           (director, ffPrefix, "trkcone_pt"),
         halo_p11             (director, ffPrefix, "halo_p11"),
         apos                 (director, ffPrefix, "apos[3]"),
         charge               (director, ffPrefix, "charge"),
         whits_a              (director, ffPrefix, "whits_a"),
         whits_bc             (director, ffPrefix, "whits_bc"),
         shits_a              (director, ffPrefix, "shits_a"),
         shits_bc             (director, ffPrefix, "shits_bc"),
         cosmic               (director, ffPrefix, "cosmic"),
         cosmictight          (director, ffPrefix, "cosmictight"),
         nseg                 (director, ffPrefix, "nseg"),
         mtcn                 (director, ffPrefix, "mtcn"),
         mtclay               (director, ffPrefix, "mtclay"),
         idxtrk               (director, ffPrefix, "idxtrk"),
         isolated             (director, ffPrefix, "isolated"),
         nsmthit              (director, ffPrefix, "nsmthit"),
         ncfthit              (director, ffPrefix, "ncfthit"),
         match_qual           (director, ffPrefix, "match_qual"),
         haslocal             (director, ffPrefix, "haslocal"),
         hascentral           (director, ffPrefix, "hascentral"),
         hascal               (director, ffPrefix, "hascal"),
         octant               (director, ffPrefix, "octant"),
         region               (director, ffPrefix, "region"),
         istight              (director, ffPrefix, "istight"),
         ismedium             (director, ffPrefix, "ismedium"),
         isloose              (director, ffPrefix, "isloose"),
         tmp_index            (director, ffPrefix, "tmp_index")
      {};
      TClaPx_TheMuonClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_7     (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         pt                   (director, ffPrefix, "pt"),
         ptcorr               (director, ffPrefix, "ptcorr"),
         ptglobal             (director, ffPrefix, "ptglobal"),
         qptloc               (director, ffPrefix, "qptloc"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         pxloc                (director, ffPrefix, "pxloc"),
         pyloc                (director, ffPrefix, "pyloc"),
         pzloc                (director, ffPrefix, "pzloc"),
         z0                   (director, ffPrefix, "z0"),
         dca                  (director, ffPrefix, "dca[2]"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         etaloc               (director, ffPrefix, "etaloc"),
         philoc               (director, ffPrefix, "philoc"),
         eloss                (director, ffPrefix, "eloss"),
         drjet                (director, ffPrefix, "drjet"),
         dedx                 (director, ffPrefix, "dedx"),
         ring                 (director, ffPrefix, "ring"),
         halo                 (director, ffPrefix, "halo"),
         chi2                 (director, ffPrefix, "chi2"),
         chi2loc              (director, ffPrefix, "chi2loc"),
         sctime_a             (director, ffPrefix, "sctime_a"),
         sctime_b             (director, ffPrefix, "sctime_b"),
         mtcsig               (director, ffPrefix, "mtcsig"),
         mtceta               (director, ffPrefix, "mtceta"),
         mtcphi               (director, ffPrefix, "mtcphi"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         trkdr                (director, ffPrefix, "trkdr"),
         trkcone_pt           (director, ffPrefix, "trkcone_pt"),
         halo_p11             (director, ffPrefix, "halo_p11"),
         apos                 (director, ffPrefix, "apos[3]"),
         charge               (director, ffPrefix, "charge"),
         whits_a              (director, ffPrefix, "whits_a"),
         whits_bc             (director, ffPrefix, "whits_bc"),
         shits_a              (director, ffPrefix, "shits_a"),
         shits_bc             (director, ffPrefix, "shits_bc"),
         cosmic               (director, ffPrefix, "cosmic"),
         cosmictight          (director, ffPrefix, "cosmictight"),
         nseg                 (director, ffPrefix, "nseg"),
         mtcn                 (director, ffPrefix, "mtcn"),
         mtclay               (director, ffPrefix, "mtclay"),
         idxtrk               (director, ffPrefix, "idxtrk"),
         isolated             (director, ffPrefix, "isolated"),
         nsmthit              (director, ffPrefix, "nsmthit"),
         ncfthit              (director, ffPrefix, "ncfthit"),
         match_qual           (director, ffPrefix, "match_qual"),
         haslocal             (director, ffPrefix, "haslocal"),
         hascentral           (director, ffPrefix, "hascentral"),
         hascal               (director, ffPrefix, "hascal"),
         octant               (director, ffPrefix, "octant"),
         region               (director, ffPrefix, "region"),
         istight              (director, ffPrefix, "istight"),
         ismedium             (director, ffPrefix, "ismedium"),
         isloose              (director, ffPrefix, "isloose"),
         tmp_index            (director, ffPrefix, "tmp_index")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        pt;
      TClaFloatProxy        ptcorr;
      TClaFloatProxy        ptglobal;
      TClaFloatProxy        qptloc;
      TClaFloatProxy        px;
      TClaFloatProxy        py;
      TClaFloatProxy        pz;
      TClaFloatProxy        pxloc;
      TClaFloatProxy        pyloc;
      TClaFloatProxy        pzloc;
      TClaFloatProxy        z0;
      TClaArrayFloatProxy   dca;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaFloatProxy        etaloc;
      TClaFloatProxy        philoc;
      TClaFloatProxy        eloss;
      TClaFloatProxy        drjet;
      TClaFloatProxy        dedx;
      TClaFloatProxy        ring;
      TClaFloatProxy        halo;
      TClaFloatProxy        chi2;
      TClaFloatProxy        chi2loc;
      TClaFloatProxy        sctime_a;
      TClaFloatProxy        sctime_b;
      TClaFloatProxy        mtcsig;
      TClaFloatProxy        mtceta;
      TClaFloatProxy        mtcphi;
      TClaFloatProxy        tkjdr;
      TClaIntProxy          tkjidx;
      TClaFloatProxy        trkdr;
      TClaFloatProxy        trkcone_pt;
      TClaFloatProxy        halo_p11;
      TClaArrayFloatProxy   apos;
      TClaIntProxy          charge;
      TClaIntProxy          whits_a;
      TClaIntProxy          whits_bc;
      TClaIntProxy          shits_a;
      TClaIntProxy          shits_bc;
      TClaIntProxy          cosmic;
      TClaIntProxy          cosmictight;
      TClaIntProxy          nseg;
      TClaIntProxy          mtcn;
      TClaIntProxy          mtclay;
      TClaIntProxy          idxtrk;
      TClaIntProxy          isolated;
      TClaIntProxy          nsmthit;
      TClaIntProxy          ncfthit;
      TClaIntProxy          match_qual;
      TClaIntProxy          haslocal;
      TClaIntProxy          hascentral;
      TClaIntProxy          hascal;
      TClaIntProxy          octant;
      TClaIntProxy          region;
      TClaIntProxy          istight;
      TClaIntProxy          ismedium;
      TClaIntProxy          isloose;
      TClaIntProxy          tmp_index;
   };
   struct TClaPx_TObject_8
   {
      TClaPx_TObject_8(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_8(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheElectronClass
      : public TClaPx_TObject_8
   {
      TClaPx_TheElectronClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_8     (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         iso                  (director, ffPrefix, "iso"),
         hmx8                 (director, ffPrefix, "hmx8"),
         hmx7                 (director, ffPrefix, "hmx7"),
         eoverp               (director, ffPrefix, "eoverp"),
         trackchi2prob        (director, ffPrefix, "trackchi2prob"),
         spatial_track_c2p    (director, ffPrefix, "spatial_track_c2p"),
         caldetectoreta       (director, ffPrefix, "caldetectoreta"),
         caldetectorphi       (director, ffPrefix, "caldetectorphi"),
         cale                 (director, ffPrefix, "cale"),
         calpx                (director, ffPrefix, "calpx"),
         calpy                (director, ffPrefix, "calpy"),
         calpz                (director, ffPrefix, "calpz"),
         l1et                 (director, ffPrefix, "l1et"),
         l2et                 (director, ffPrefix, "l2et"),
         charge               (director, ffPrefix, "charge"),
         trackmatch           (director, ffPrefix, "trackmatch"),
         istight              (director, ffPrefix, "istight"),
         infiducial           (director, ffPrefix, "infiducial"),
         drmatch              (director, ffPrefix, "drmatch"),
         ndr2                 (director, ffPrefix, "ndr2"),
         sumtrackpt           (director, ffPrefix, "sumtrackpt"),
         track_index          (director, ffPrefix, "track_index"),
         trackpx              (director, ffPrefix, "trackpx"),
         trackpy              (director, ffPrefix, "trackpy"),
         trackpz              (director, ffPrefix, "trackpz"),
         z0                   (director, ffPrefix, "z0"),
         dca                  (director, ffPrefix, "dca[2]"),
         drt2                 (director, ffPrefix, "drt2"),
         lhood                (director, ffPrefix, "lhood"),
         PSe                  (director, ffPrefix, "PSe"),
         EM1e                 (director, ffPrefix, "EM1e"),
         EM2e                 (director, ffPrefix, "EM2e"),
         EM3e                 (director, ffPrefix, "EM3e"),
         EM4e                 (director, ffPrefix, "EM4e"),
         FH1e                 (director, ffPrefix, "FH1e"),
         SigRphi              (director, ffPrefix, "SigRphi"),
         SigZorR              (director, ffPrefix, "SigZorR"),
         CPSstripmin          (director, ffPrefix, "CPSstripmin"),
         CPSchi2min           (director, ffPrefix, "CPSchi2min"),
         CPSstripmax          (director, ffPrefix, "CPSstripmax"),
         CPSchi2max           (director, ffPrefix, "CPSchi2max"),
         CPSstripmax_corr     (director, ffPrefix, "CPSstripmax_corr")
      {};
      TClaPx_TheElectronClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_8     (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         iso                  (director, ffPrefix, "iso"),
         hmx8                 (director, ffPrefix, "hmx8"),
         hmx7                 (director, ffPrefix, "hmx7"),
         eoverp               (director, ffPrefix, "eoverp"),
         trackchi2prob        (director, ffPrefix, "trackchi2prob"),
         spatial_track_c2p    (director, ffPrefix, "spatial_track_c2p"),
         caldetectoreta       (director, ffPrefix, "caldetectoreta"),
         caldetectorphi       (director, ffPrefix, "caldetectorphi"),
         cale                 (director, ffPrefix, "cale"),
         calpx                (director, ffPrefix, "calpx"),
         calpy                (director, ffPrefix, "calpy"),
         calpz                (director, ffPrefix, "calpz"),
         l1et                 (director, ffPrefix, "l1et"),
         l2et                 (director, ffPrefix, "l2et"),
         charge               (director, ffPrefix, "charge"),
         trackmatch           (director, ffPrefix, "trackmatch"),
         istight              (director, ffPrefix, "istight"),
         infiducial           (director, ffPrefix, "infiducial"),
         drmatch              (director, ffPrefix, "drmatch"),
         ndr2                 (director, ffPrefix, "ndr2"),
         sumtrackpt           (director, ffPrefix, "sumtrackpt"),
         track_index          (director, ffPrefix, "track_index"),
         trackpx              (director, ffPrefix, "trackpx"),
         trackpy              (director, ffPrefix, "trackpy"),
         trackpz              (director, ffPrefix, "trackpz"),
         z0                   (director, ffPrefix, "z0"),
         dca                  (director, ffPrefix, "dca[2]"),
         drt2                 (director, ffPrefix, "drt2"),
         lhood                (director, ffPrefix, "lhood"),
         PSe                  (director, ffPrefix, "PSe"),
         EM1e                 (director, ffPrefix, "EM1e"),
         EM2e                 (director, ffPrefix, "EM2e"),
         EM3e                 (director, ffPrefix, "EM3e"),
         EM4e                 (director, ffPrefix, "EM4e"),
         FH1e                 (director, ffPrefix, "FH1e"),
         SigRphi              (director, ffPrefix, "SigRphi"),
         SigZorR              (director, ffPrefix, "SigZorR"),
         CPSstripmin          (director, ffPrefix, "CPSstripmin"),
         CPSchi2min           (director, ffPrefix, "CPSchi2min"),
         CPSstripmax          (director, ffPrefix, "CPSstripmax"),
         CPSchi2max           (director, ffPrefix, "CPSchi2max"),
         CPSstripmax_corr     (director, ffPrefix, "CPSstripmax_corr")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        pt;
      TClaFloatProxy        px;
      TClaFloatProxy        py;
      TClaFloatProxy        pz;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaFloatProxy        e;
      TClaFloatProxy        emf;
      TClaFloatProxy        iso;
      TClaFloatProxy        hmx8;
      TClaFloatProxy        hmx7;
      TClaFloatProxy        eoverp;
      TClaFloatProxy        trackchi2prob;
      TClaFloatProxy        spatial_track_c2p;
      TClaFloatProxy        caldetectoreta;
      TClaFloatProxy        caldetectorphi;
      TClaFloatProxy        cale;
      TClaFloatProxy        calpx;
      TClaFloatProxy        calpy;
      TClaFloatProxy        calpz;
      TClaFloatProxy        l1et;
      TClaFloatProxy        l2et;
      TClaIntProxy          charge;
      TClaIntProxy          trackmatch;
      TClaIntProxy          istight;
      TClaIntProxy          infiducial;
      TClaIntProxy          drmatch;
      TClaIntProxy          ndr2;
      TClaFloatProxy        sumtrackpt;
      TClaIntProxy          track_index;
      TClaFloatProxy        trackpx;
      TClaFloatProxy        trackpy;
      TClaFloatProxy        trackpz;
      TClaFloatProxy        z0;
      TClaArrayFloatProxy   dca;
      TClaFloatProxy        drt2;
      TClaFloatProxy        lhood;
      TClaFloatProxy        PSe;
      TClaFloatProxy        EM1e;
      TClaFloatProxy        EM2e;
      TClaFloatProxy        EM3e;
      TClaFloatProxy        EM4e;
      TClaFloatProxy        FH1e;
      TClaFloatProxy        SigRphi;
      TClaFloatProxy        SigZorR;
      TClaFloatProxy        CPSstripmin;
      TClaFloatProxy        CPSchi2min;
      TClaFloatProxy        CPSstripmax;
      TClaFloatProxy        CPSchi2max;
      TClaFloatProxy        CPSstripmax_corr;
   };
   struct TClaPx_TObject_9
   {
      TClaPx_TObject_9(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_9(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheSoftElectronClass
      : public TClaPx_TObject_9
   {
      TClaPx_TheSoftElectronClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_9(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         eop             (director, ffPrefix, "eop"),
         emf             (director, ffPrefix, "emf"),
         em1f            (director, ffPrefix, "em1f"),
         em2f            (director, ffPrefix, "em2f"),
         em3f            (director, ffPrefix, "em3f"),
         em4f            (director, ffPrefix, "em4f"),
         etot            (director, ffPrefix, "etot"),
         deltar          (director, ffPrefix, "deltar"),
         ptrel           (director, ffPrefix, "ptrel"),
         track           (director, ffPrefix, "track"),
         jet             (director, ffPrefix, "jet"),
         deta            (director, ffPrefix, "deta")
      {};
      TClaPx_TheSoftElectronClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_9(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         eop             (director, ffPrefix, "eop"),
         emf             (director, ffPrefix, "emf"),
         em1f            (director, ffPrefix, "em1f"),
         em2f            (director, ffPrefix, "em2f"),
         em3f            (director, ffPrefix, "em3f"),
         em4f            (director, ffPrefix, "em4f"),
         etot            (director, ffPrefix, "etot"),
         deltar          (director, ffPrefix, "deltar"),
         ptrel           (director, ffPrefix, "ptrel"),
         track           (director, ffPrefix, "track"),
         jet             (director, ffPrefix, "jet"),
         deta            (director, ffPrefix, "deta")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy   eop;
      TClaFloatProxy   emf;
      TClaFloatProxy   em1f;
      TClaFloatProxy   em2f;
      TClaFloatProxy   em3f;
      TClaFloatProxy   em4f;
      TClaFloatProxy   etot;
      TClaFloatProxy   deltar;
      TClaFloatProxy   ptrel;
      TClaIntProxy     track;
      TClaIntProxy     jet;
      TClaFloatProxy   deta;
   };
   struct TClaPx_TObject_10
   {
      TClaPx_TObject_10(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_10(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheJetClass
      : public TClaPx_TObject_10
   {
      TClaPx_TheJetClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_10    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         pt_uncorr_noCH       (director, ffPrefix, "pt_uncorr_noCH"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         chf                  (director, ffPrefix, "chf"),
         hotf                 (director, ffPrefix, "hotf"),
         icdf                 (director, ffPrefix, "icdf"),
         ecmgf                (director, ffPrefix, "ecmgf"),
         ccmgf                (director, ffPrefix, "ccmgf"),
         hadcc                (director, ffPrefix, "hadcc"),
         hadec                (director, ffPrefix, "hadec"),
         seedEt               (director, ffPrefix, "seedEt"),
         etaWidth             (director, ffPrefix, "etaWidth"),
         phiWidth             (director, ffPrefix, "phiWidth"),
         f90                  (director, ffPrefix, "f90"),
         jes                  (director, ffPrefix, "jes[3]"),
         smear_coeff          (director, ffPrefix, "smear_coeff"),
         jes_data_lq          (director, ffPrefix, "jes_data_lq[3]"),
         jes_data_hq          (director, ffPrefix, "jes_data_hq[3]"),
         jes_mc_lq            (director, ffPrefix, "jes_mc_lq[3]"),
         jes_mc_hq            (director, ffPrefix, "jes_mc_hq[3]"),
         met                  (director, ffPrefix, "met[3]"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         tkjphi               (director, ffPrefix, "tkjphi"),
         tkjeta               (director, ffPrefix, "tkjeta"),
         tkjpt                (director, ffPrefix, "tkjpt"),
         muodr                (director, ffPrefix, "muodr"),
         muoptrel             (director, ffPrefix, "muoptrel"),
         muopt                (director, ffPrefix, "muopt"),
         detEta               (director, ffPrefix, "detEta"),
         detPhi               (director, ffPrefix, "detPhi"),
         cps_energy           (director, ffPrefix, "cps_energy"),
         l1et                 (director, ffPrefix, "l1et"),
         l1set                (director, ffPrefix, "l1set"),
         l1dist               (director, ffPrefix, "l1dist"),
         l1trig               (director, ffPrefix, "l1trig"),
         l2et                 (director, ffPrefix, "l2et"),
         l2et05               (director, ffPrefix, "l2et05"),
         l2dist               (director, ffPrefix, "l2dist"),
         l1conf               (director, ffPrefix, "l1conf"),
         tagprob              (director, ffPrefix, "tagprob"),
         sb_iptag             (director, ffPrefix, "sb_iptag"),
         svx_tags             (director, ffPrefix, "svx_tags"),
         svx_indices          (director, ffPrefix, "svx_indices"),
         n90                  (director, ffPrefix, "n90"),
         muoidseg             (director, ffPrefix, "muoidseg"),
         l1nt                 (director, ffPrefix, "l1nt"),
         split                (director, ffPrefix, "split"),
         merge                (director, ffPrefix, "merge"),
         tag                  (director, ffPrefix, "tag"),
         flavor               (director, ffPrefix, "flavor"),
         muotag               (director, ffPrefix, "muotag"),
         quality              (director, ffPrefix, "quality"),
         jlip_prob            (director, ffPrefix, "jlip_prob"),
         jlip_prob_neg        (director, ffPrefix, "jlip_prob_neg"),
         isTaggable           (director, ffPrefix, "isTaggable"),
         muo_index            (director, ffPrefix, "muo_index")
      {};
      TClaPx_TheJetClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_10    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         pt_uncorr_noCH       (director, ffPrefix, "pt_uncorr_noCH"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         chf                  (director, ffPrefix, "chf"),
         hotf                 (director, ffPrefix, "hotf"),
         icdf                 (director, ffPrefix, "icdf"),
         ecmgf                (director, ffPrefix, "ecmgf"),
         ccmgf                (director, ffPrefix, "ccmgf"),
         hadcc                (director, ffPrefix, "hadcc"),
         hadec                (director, ffPrefix, "hadec"),
         seedEt               (director, ffPrefix, "seedEt"),
         etaWidth             (director, ffPrefix, "etaWidth"),
         phiWidth             (director, ffPrefix, "phiWidth"),
         f90                  (director, ffPrefix, "f90"),
         jes                  (director, ffPrefix, "jes[3]"),
         smear_coeff          (director, ffPrefix, "smear_coeff"),
         jes_data_lq          (director, ffPrefix, "jes_data_lq[3]"),
         jes_data_hq          (director, ffPrefix, "jes_data_hq[3]"),
         jes_mc_lq            (director, ffPrefix, "jes_mc_lq[3]"),
         jes_mc_hq            (director, ffPrefix, "jes_mc_hq[3]"),
         met                  (director, ffPrefix, "met[3]"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         tkjphi               (director, ffPrefix, "tkjphi"),
         tkjeta               (director, ffPrefix, "tkjeta"),
         tkjpt                (director, ffPrefix, "tkjpt"),
         muodr                (director, ffPrefix, "muodr"),
         muoptrel             (director, ffPrefix, "muoptrel"),
         muopt                (director, ffPrefix, "muopt"),
         detEta               (director, ffPrefix, "detEta"),
         detPhi               (director, ffPrefix, "detPhi"),
         cps_energy           (director, ffPrefix, "cps_energy"),
         l1et                 (director, ffPrefix, "l1et"),
         l1set                (director, ffPrefix, "l1set"),
         l1dist               (director, ffPrefix, "l1dist"),
         l1trig               (director, ffPrefix, "l1trig"),
         l2et                 (director, ffPrefix, "l2et"),
         l2et05               (director, ffPrefix, "l2et05"),
         l2dist               (director, ffPrefix, "l2dist"),
         l1conf               (director, ffPrefix, "l1conf"),
         tagprob              (director, ffPrefix, "tagprob"),
         sb_iptag             (director, ffPrefix, "sb_iptag"),
         svx_tags             (director, ffPrefix, "svx_tags"),
         svx_indices          (director, ffPrefix, "svx_indices"),
         n90                  (director, ffPrefix, "n90"),
         muoidseg             (director, ffPrefix, "muoidseg"),
         l1nt                 (director, ffPrefix, "l1nt"),
         split                (director, ffPrefix, "split"),
         merge                (director, ffPrefix, "merge"),
         tag                  (director, ffPrefix, "tag"),
         flavor               (director, ffPrefix, "flavor"),
         muotag               (director, ffPrefix, "muotag"),
         quality              (director, ffPrefix, "quality"),
         jlip_prob            (director, ffPrefix, "jlip_prob"),
         jlip_prob_neg        (director, ffPrefix, "jlip_prob_neg"),
         isTaggable           (director, ffPrefix, "isTaggable"),
         muo_index            (director, ffPrefix, "muo_index")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        pt;
      TClaFloatProxy        px;
      TClaFloatProxy        py;
      TClaFloatProxy        pz;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaFloatProxy        pt_uncorr_noCH;
      TClaFloatProxy        e;
      TClaFloatProxy        emf;
      TClaFloatProxy        chf;
      TClaFloatProxy        hotf;
      TClaFloatProxy        icdf;
      TClaFloatProxy        ecmgf;
      TClaFloatProxy        ccmgf;
      TClaFloatProxy        hadcc;
      TClaFloatProxy        hadec;
      TClaFloatProxy        seedEt;
      TClaFloatProxy        etaWidth;
      TClaFloatProxy        phiWidth;
      TClaFloatProxy        f90;
      TClaArrayFloatProxy   jes;
      TClaFloatProxy        smear_coeff;
      TClaArrayFloatProxy   jes_data_lq;
      TClaArrayFloatProxy   jes_data_hq;
      TClaArrayFloatProxy   jes_mc_lq;
      TClaArrayFloatProxy   jes_mc_hq;
      TClaArrayFloatProxy   met;
      TClaFloatProxy        tkjdr;
      TClaFloatProxy        tkjidx;
      TClaFloatProxy        tkjphi;
      TClaFloatProxy        tkjeta;
      TClaFloatProxy        tkjpt;
      TClaFloatProxy        muodr;
      TClaFloatProxy        muoptrel;
      TClaFloatProxy        muopt;
      TClaFloatProxy        detEta;
      TClaFloatProxy        detPhi;
      TClaFloatProxy        cps_energy;
      TClaFloatProxy        l1et;
      TClaFloatProxy        l1set;
      TClaFloatProxy        l1dist;
      TClaFloatProxy        l1trig;
      TClaFloatProxy        l2et;
      TClaFloatProxy        l2et05;
      TClaFloatProxy        l2dist;
      TClaFloatProxy        l1conf;
      TClaFloatProxy        tagprob;
      TClaFloatProxy        sb_iptag;
      TClaIntProxy          svx_tags;
      TClaArrayIntProxy     svx_indices;
      TClaIntProxy          n90;
      TClaIntProxy          muoidseg;
      TClaIntProxy          l1nt;
      TClaIntProxy          split;
      TClaIntProxy          merge;
      TClaIntProxy          tag;
      TClaIntProxy          flavor;
      TClaIntProxy          muotag;
      TClaIntProxy          quality;
      TClaFloatProxy        jlip_prob;
      TClaFloatProxy        jlip_prob_neg;
      TClaIntProxy          isTaggable;
      TClaIntProxy          muo_index;
   };
   struct TClaPx_TObject_11
   {
      TClaPx_TObject_11(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_11(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheJetClass_1
      : public TClaPx_TObject_11
   {
      TClaPx_TheJetClass_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_11    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         pt_uncorr_noCH       (director, ffPrefix, "pt_uncorr_noCH"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         chf                  (director, ffPrefix, "chf"),
         hotf                 (director, ffPrefix, "hotf"),
         icdf                 (director, ffPrefix, "icdf"),
         ecmgf                (director, ffPrefix, "ecmgf"),
         ccmgf                (director, ffPrefix, "ccmgf"),
         hadcc                (director, ffPrefix, "hadcc"),
         hadec                (director, ffPrefix, "hadec"),
         seedEt               (director, ffPrefix, "seedEt"),
         etaWidth             (director, ffPrefix, "etaWidth"),
         phiWidth             (director, ffPrefix, "phiWidth"),
         f90                  (director, ffPrefix, "f90"),
         jes                  (director, ffPrefix, "jes[3]"),
         smear_coeff          (director, ffPrefix, "smear_coeff"),
         jes_data_lq          (director, ffPrefix, "jes_data_lq[3]"),
         jes_data_hq          (director, ffPrefix, "jes_data_hq[3]"),
         jes_mc_lq            (director, ffPrefix, "jes_mc_lq[3]"),
         jes_mc_hq            (director, ffPrefix, "jes_mc_hq[3]"),
         met                  (director, ffPrefix, "met[3]"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         tkjphi               (director, ffPrefix, "tkjphi"),
         tkjeta               (director, ffPrefix, "tkjeta"),
         tkjpt                (director, ffPrefix, "tkjpt"),
         muodr                (director, ffPrefix, "muodr"),
         muoptrel             (director, ffPrefix, "muoptrel"),
         muopt                (director, ffPrefix, "muopt"),
         detEta               (director, ffPrefix, "detEta"),
         detPhi               (director, ffPrefix, "detPhi"),
         cps_energy           (director, ffPrefix, "cps_energy"),
         l1et                 (director, ffPrefix, "l1et"),
         l1set                (director, ffPrefix, "l1set"),
         l1dist               (director, ffPrefix, "l1dist"),
         l1trig               (director, ffPrefix, "l1trig"),
         l2et                 (director, ffPrefix, "l2et"),
         l2et05               (director, ffPrefix, "l2et05"),
         l2dist               (director, ffPrefix, "l2dist"),
         l1conf               (director, ffPrefix, "l1conf"),
         tagprob              (director, ffPrefix, "tagprob"),
         sb_iptag             (director, ffPrefix, "sb_iptag"),
         svx_tags             (director, ffPrefix, "svx_tags"),
         svx_indices          (director, ffPrefix, "svx_indices"),
         n90                  (director, ffPrefix, "n90"),
         muoidseg             (director, ffPrefix, "muoidseg"),
         l1nt                 (director, ffPrefix, "l1nt"),
         split                (director, ffPrefix, "split"),
         merge                (director, ffPrefix, "merge"),
         tag                  (director, ffPrefix, "tag"),
         flavor               (director, ffPrefix, "flavor"),
         muotag               (director, ffPrefix, "muotag"),
         quality              (director, ffPrefix, "quality"),
         jlip_prob            (director, ffPrefix, "jlip_prob"),
         jlip_prob_neg        (director, ffPrefix, "jlip_prob_neg"),
         isTaggable           (director, ffPrefix, "isTaggable"),
         muo_index            (director, ffPrefix, "muo_index")
      {};
      TClaPx_TheJetClass_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_11    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         pt_uncorr_noCH       (director, ffPrefix, "pt_uncorr_noCH"),
         e                    (director, ffPrefix, "e"),
         emf                  (director, ffPrefix, "emf"),
         chf                  (director, ffPrefix, "chf"),
         hotf                 (director, ffPrefix, "hotf"),
         icdf                 (director, ffPrefix, "icdf"),
         ecmgf                (director, ffPrefix, "ecmgf"),
         ccmgf                (director, ffPrefix, "ccmgf"),
         hadcc                (director, ffPrefix, "hadcc"),
         hadec                (director, ffPrefix, "hadec"),
         seedEt               (director, ffPrefix, "seedEt"),
         etaWidth             (director, ffPrefix, "etaWidth"),
         phiWidth             (director, ffPrefix, "phiWidth"),
         f90                  (director, ffPrefix, "f90"),
         jes                  (director, ffPrefix, "jes[3]"),
         smear_coeff          (director, ffPrefix, "smear_coeff"),
         jes_data_lq          (director, ffPrefix, "jes_data_lq[3]"),
         jes_data_hq          (director, ffPrefix, "jes_data_hq[3]"),
         jes_mc_lq            (director, ffPrefix, "jes_mc_lq[3]"),
         jes_mc_hq            (director, ffPrefix, "jes_mc_hq[3]"),
         met                  (director, ffPrefix, "met[3]"),
         tkjdr                (director, ffPrefix, "tkjdr"),
         tkjidx               (director, ffPrefix, "tkjidx"),
         tkjphi               (director, ffPrefix, "tkjphi"),
         tkjeta               (director, ffPrefix, "tkjeta"),
         tkjpt                (director, ffPrefix, "tkjpt"),
         muodr                (director, ffPrefix, "muodr"),
         muoptrel             (director, ffPrefix, "muoptrel"),
         muopt                (director, ffPrefix, "muopt"),
         detEta               (director, ffPrefix, "detEta"),
         detPhi               (director, ffPrefix, "detPhi"),
         cps_energy           (director, ffPrefix, "cps_energy"),
         l1et                 (director, ffPrefix, "l1et"),
         l1set                (director, ffPrefix, "l1set"),
         l1dist               (director, ffPrefix, "l1dist"),
         l1trig               (director, ffPrefix, "l1trig"),
         l2et                 (director, ffPrefix, "l2et"),
         l2et05               (director, ffPrefix, "l2et05"),
         l2dist               (director, ffPrefix, "l2dist"),
         l1conf               (director, ffPrefix, "l1conf"),
         tagprob              (director, ffPrefix, "tagprob"),
         sb_iptag             (director, ffPrefix, "sb_iptag"),
         svx_tags             (director, ffPrefix, "svx_tags"),
         svx_indices          (director, ffPrefix, "svx_indices"),
         n90                  (director, ffPrefix, "n90"),
         muoidseg             (director, ffPrefix, "muoidseg"),
         l1nt                 (director, ffPrefix, "l1nt"),
         split                (director, ffPrefix, "split"),
         merge                (director, ffPrefix, "merge"),
         tag                  (director, ffPrefix, "tag"),
         flavor               (director, ffPrefix, "flavor"),
         muotag               (director, ffPrefix, "muotag"),
         quality              (director, ffPrefix, "quality"),
         jlip_prob            (director, ffPrefix, "jlip_prob"),
         jlip_prob_neg        (director, ffPrefix, "jlip_prob_neg"),
         isTaggable           (director, ffPrefix, "isTaggable"),
         muo_index            (director, ffPrefix, "muo_index")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        pt;
      TClaFloatProxy        px;
      TClaFloatProxy        py;
      TClaFloatProxy        pz;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaFloatProxy        pt_uncorr_noCH;
      TClaFloatProxy        e;
      TClaFloatProxy        emf;
      TClaFloatProxy        chf;
      TClaFloatProxy        hotf;
      TClaFloatProxy        icdf;
      TClaFloatProxy        ecmgf;
      TClaFloatProxy        ccmgf;
      TClaFloatProxy        hadcc;
      TClaFloatProxy        hadec;
      TClaFloatProxy        seedEt;
      TClaFloatProxy        etaWidth;
      TClaFloatProxy        phiWidth;
      TClaFloatProxy        f90;
      TClaArrayFloatProxy   jes;
      TClaFloatProxy        smear_coeff;
      TClaArrayFloatProxy   jes_data_lq;
      TClaArrayFloatProxy   jes_data_hq;
      TClaArrayFloatProxy   jes_mc_lq;
      TClaArrayFloatProxy   jes_mc_hq;
      TClaArrayFloatProxy   met;
      TClaFloatProxy        tkjdr;
      TClaFloatProxy        tkjidx;
      TClaFloatProxy        tkjphi;
      TClaFloatProxy        tkjeta;
      TClaFloatProxy        tkjpt;
      TClaFloatProxy        muodr;
      TClaFloatProxy        muoptrel;
      TClaFloatProxy        muopt;
      TClaFloatProxy        detEta;
      TClaFloatProxy        detPhi;
      TClaFloatProxy        cps_energy;
      TClaFloatProxy        l1et;
      TClaFloatProxy        l1set;
      TClaFloatProxy        l1dist;
      TClaFloatProxy        l1trig;
      TClaFloatProxy        l2et;
      TClaFloatProxy        l2et05;
      TClaFloatProxy        l2dist;
      TClaFloatProxy        l1conf;
      TClaFloatProxy        tagprob;
      TClaFloatProxy        sb_iptag;
      TClaIntProxy          svx_tags;
      TClaArrayIntProxy     svx_indices;
      TClaIntProxy          n90;
      TClaIntProxy          muoidseg;
      TClaIntProxy          l1nt;
      TClaIntProxy          split;
      TClaIntProxy          merge;
      TClaIntProxy          tag;
      TClaIntProxy          flavor;
      TClaIntProxy          muotag;
      TClaIntProxy          quality;
      TClaFloatProxy        jlip_prob;
      TClaFloatProxy        jlip_prob_neg;
      TClaIntProxy          isTaggable;
      TClaIntProxy          muo_index;
   };
   struct TClaPx_TObject_12
   {
      TClaPx_TObject_12(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_12(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheVertexClass
      : public TClaPx_TObject_12
   {
      TClaPx_TheVertexClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_12    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         mult                 (director, ffPrefix, "mult")
      {};
      TClaPx_TheVertexClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_12    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         mult                 (director, ffPrefix, "mult")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayFloatProxy   vertex;
      TClaArrayFloatProxy   vertexerr;
      TClaIntProxy          mult;
   };
   struct TClaPx_TObject_13
   {
      TClaPx_TObject_13(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_13(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheVertexClass_1
      : public TClaPx_TObject_13
   {
      TClaPx_TheVertexClass_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_13    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         mult                 (director, ffPrefix, "mult")
      {};
      TClaPx_TheVertexClass_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_13    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         mult                 (director, ffPrefix, "mult")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaArrayFloatProxy   vertex;
      TClaArrayFloatProxy   vertexerr;
      TClaIntProxy          mult;
   };
   struct TClaPx_TObject_14
   {
      TClaPx_TObject_14(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_14(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheCellClass
      : public TClaPx_TObject_14
   {
      TClaPx_TheCellClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_14(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TClaPx_TheCellClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_14(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy     ieta;
      TClaIntProxy     iphi;
      TClaIntProxy     ilyr;
      TClaFloatProxy   e;
      TClaFloatProxy   phi;
      TClaFloatProxy   eta;
      TClaFloatProxy   pt;
   };
   struct TClaPx_TObject_15
   {
      TClaPx_TObject_15(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_15(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheCellClass_1
      : public TClaPx_TObject_15
   {
      TClaPx_TheCellClass_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_15(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TClaPx_TheCellClass_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_15(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy     ieta;
      TClaIntProxy     iphi;
      TClaIntProxy     ilyr;
      TClaFloatProxy   e;
      TClaFloatProxy   phi;
      TClaFloatProxy   eta;
      TClaFloatProxy   pt;
   };
   struct TClaPx_TObject_16
   {
      TClaPx_TObject_16(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_16(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheTrackClass
      : public TClaPx_TObject_16
   {
      TClaPx_TheTrackClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_16    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         tla                  (director, ffPrefix, "tla"),
         phi                  (director, ffPrefix, "phi"),
         z0                   (director, ffPrefix, "z0"),
         econe20              (director, ffPrefix, "econe20"),
         chsq                 (director, ffPrefix, "chsq"),
         smthits              (director, ffPrefix, "smthits"),
         cfthits              (director, ffPrefix, "cfthits"),
         charge               (director, ffPrefix, "charge"),
         index                (director, ffPrefix, "index"),
         match                (director, ffPrefix, "match"),
         hmask                (director, ffPrefix, "hmask[3]"),
         par                  (director, ffPrefix, "par[5]"),
         err                  (director, ffPrefix, "err[15]"),
         dcaerr               (director, ffPrefix, "dcaerr")
      {};
      TClaPx_TheTrackClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_16    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         pt                   (director, ffPrefix, "pt"),
         px                   (director, ffPrefix, "px"),
         py                   (director, ffPrefix, "py"),
         pz                   (director, ffPrefix, "pz"),
         eta                  (director, ffPrefix, "eta"),
         tla                  (director, ffPrefix, "tla"),
         phi                  (director, ffPrefix, "phi"),
         z0                   (director, ffPrefix, "z0"),
         econe20              (director, ffPrefix, "econe20"),
         chsq                 (director, ffPrefix, "chsq"),
         smthits              (director, ffPrefix, "smthits"),
         cfthits              (director, ffPrefix, "cfthits"),
         charge               (director, ffPrefix, "charge"),
         index                (director, ffPrefix, "index"),
         match                (director, ffPrefix, "match"),
         hmask                (director, ffPrefix, "hmask[3]"),
         par                  (director, ffPrefix, "par[5]"),
         err                  (director, ffPrefix, "err[15]"),
         dcaerr               (director, ffPrefix, "dcaerr")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        pt;
      TClaFloatProxy        px;
      TClaFloatProxy        py;
      TClaFloatProxy        pz;
      TClaFloatProxy        eta;
      TClaFloatProxy        tla;
      TClaFloatProxy        phi;
      TClaFloatProxy        z0;
      TClaFloatProxy        econe20;
      TClaFloatProxy        chsq;
      TClaIntProxy          smthits;
      TClaIntProxy          cfthits;
      TClaIntProxy          charge;
      TClaIntProxy          index;
      TClaIntProxy          match;
      TClaArrayIntProxy     hmask;
      TClaArrayFloatProxy   par;
      TClaArrayFloatProxy   err;
      TClaFloatProxy        dcaerr;
   };
   struct TClaPx_TObject_17
   {
      TClaPx_TObject_17(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_17(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheTrackClusterClass
      : public TClaPx_TObject_17
   {
      TClaPx_TheTrackClusterClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_17(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         phi             (director, ffPrefix, "phi"),
         dphi            (director, ffPrefix, "dphi"),
         z               (director, ffPrefix, "z"),
         dz              (director, ffPrefix, "dz"),
         layer           (director, ffPrefix, "layer"),
         r               (director, ffPrefix, "r"),
         dpdz            (director, ffPrefix, "dpdz"),
         det             (director, ffPrefix, "det")
      {};
      TClaPx_TheTrackClusterClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_17(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         phi             (director, ffPrefix, "phi"),
         dphi            (director, ffPrefix, "dphi"),
         z               (director, ffPrefix, "z"),
         dz              (director, ffPrefix, "dz"),
         layer           (director, ffPrefix, "layer"),
         r               (director, ffPrefix, "r"),
         dpdz            (director, ffPrefix, "dpdz"),
         det             (director, ffPrefix, "det")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy   phi;
      TClaFloatProxy   dphi;
      TClaFloatProxy   z;
      TClaFloatProxy   dz;
      TClaIntProxy     layer;
      TClaFloatProxy   r;
      TClaFloatProxy   dpdz;
      TClaIntProxy     det;
   };
   struct TClaPx_TObject_18
   {
      TClaPx_TObject_18(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_18(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheMCParticleClass
      : public TClaPx_TObject_18
   {
      TClaPx_TheMCParticleClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_18(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         e               (director, ffPrefix, "e"),
         pt              (director, ffPrefix, "pt"),
         eta             (director, ffPrefix, "eta"),
         phi             (director, ffPrefix, "phi"),
         pdgid           (director, ffPrefix, "pdgid"),
         pmask           (director, ffPrefix, "pmask"),
         vertex          (director, ffPrefix, "vertex")
      {};
      TClaPx_TheMCParticleClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_18(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         e               (director, ffPrefix, "e"),
         pt              (director, ffPrefix, "pt"),
         eta             (director, ffPrefix, "eta"),
         phi             (director, ffPrefix, "phi"),
         pdgid           (director, ffPrefix, "pdgid"),
         pmask           (director, ffPrefix, "pmask"),
         vertex          (director, ffPrefix, "vertex")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy   e;
      TClaFloatProxy   pt;
      TClaFloatProxy   eta;
      TClaFloatProxy   phi;
      TClaIntProxy     pdgid;
      TClaIntProxy     pmask;
      TClaIntProxy     vertex;
   };
   struct TClaPx_TObject_19
   {
      TClaPx_TObject_19(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_19(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheMCVertexClass
      : public TClaPx_TObject_19
   {
      TClaPx_TheMCVertexClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_19(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         x               (director, ffPrefix, "x"),
         y               (director, ffPrefix, "y"),
         z               (director, ffPrefix, "z"),
         ct              (director, ffPrefix, "ct"),
         particle        (director, ffPrefix, "particle")
      {};
      TClaPx_TheMCVertexClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_19(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         x               (director, ffPrefix, "x"),
         y               (director, ffPrefix, "y"),
         z               (director, ffPrefix, "z"),
         ct              (director, ffPrefix, "ct"),
         particle        (director, ffPrefix, "particle")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy   x;
      TClaFloatProxy   y;
      TClaFloatProxy   z;
      TClaFloatProxy   ct;
      TClaIntProxy     particle;
   };
   struct TClaPx_TObject_20
   {
      TClaPx_TObject_20(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_20(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheTrackJetClass
      : public TClaPx_TObject_20
   {
      TClaPx_TheTrackJetClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_20(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         z               (director, ffPrefix, "z"),
         px              (director, ffPrefix, "px"),
         py              (director, ffPrefix, "py"),
         pz              (director, ffPrefix, "pz"),
         eta             (director, ffPrefix, "eta"),
         phi             (director, ffPrefix, "phi"),
         mult            (director, ffPrefix, "mult"),
         highestPt       (director, ffPrefix, "highestPt"),
         sumPt           (director, ffPrefix, "sumPt"),
         sigma           (director, ffPrefix, "sigma")
      {};
      TClaPx_TheTrackJetClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_20(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         z               (director, ffPrefix, "z"),
         px              (director, ffPrefix, "px"),
         py              (director, ffPrefix, "py"),
         pz              (director, ffPrefix, "pz"),
         eta             (director, ffPrefix, "eta"),
         phi             (director, ffPrefix, "phi"),
         mult            (director, ffPrefix, "mult"),
         highestPt       (director, ffPrefix, "highestPt"),
         sumPt           (director, ffPrefix, "sumPt"),
         sigma           (director, ffPrefix, "sigma")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy   z;
      TClaFloatProxy   px;
      TClaFloatProxy   py;
      TClaFloatProxy   pz;
      TClaFloatProxy   eta;
      TClaFloatProxy   phi;
      TClaIntProxy     mult;
      TClaFloatProxy   highestPt;
      TClaFloatProxy   sumPt;
      TClaFloatProxy   sigma;
   };
   struct TClaPx_TObject_21
   {
      TClaPx_TObject_21(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_21(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheSecondaryVertexClass
      : public TClaPx_TObject_21
   {
      TClaPx_TheSecondaryVertexClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_21    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         mult                 (director, ffPrefix, "mult"),
         TrackIndices         (director, ffPrefix, "TrackIndices"),
         mem_idx              (director, ffPrefix, "mem_idx"),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         decayLen             (director, ffPrefix, "decayLen"),
         decayLenSig          (director, ffPrefix, "decayLenSig"),
         chi2                 (director, ffPrefix, "chi2"),
         mass                 (director, ffPrefix, "mass"),
         collinearity         (director, ffPrefix, "collinearity"),
         charge               (director, ffPrefix, "charge"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         p                    (director, ffPrefix, "p[3]"),
         pTcorrMass           (director, ffPrefix, "pTcorrMass"),
         isKShort             (director, ffPrefix, "isKShort")
      {};
      TClaPx_TheSecondaryVertexClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_21    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         mult                 (director, ffPrefix, "mult"),
         TrackIndices         (director, ffPrefix, "TrackIndices"),
         mem_idx              (director, ffPrefix, "mem_idx"),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         decayLen             (director, ffPrefix, "decayLen"),
         decayLenSig          (director, ffPrefix, "decayLenSig"),
         chi2                 (director, ffPrefix, "chi2"),
         mass                 (director, ffPrefix, "mass"),
         collinearity         (director, ffPrefix, "collinearity"),
         charge               (director, ffPrefix, "charge"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         p                    (director, ffPrefix, "p[3]"),
         pTcorrMass           (director, ffPrefix, "pTcorrMass"),
         isKShort             (director, ffPrefix, "isKShort")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy          mult;
      TClaArrayIntProxy     TrackIndices;
      TClaIntProxy          mem_idx;
      TClaArrayFloatProxy   vertex;
      TClaArrayFloatProxy   vertexerr;
      TClaFloatProxy        decayLen;
      TClaFloatProxy        decayLenSig;
      TClaFloatProxy        chi2;
      TClaFloatProxy        mass;
      TClaFloatProxy        collinearity;
      TClaFloatProxy        charge;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaArrayFloatProxy   p;
      TClaFloatProxy        pTcorrMass;
      TClaIntProxy          isKShort;
   };
   struct TClaPx_TObject_22
   {
      TClaPx_TObject_22(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_22(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheSVPROBClass
      : public TClaPx_TObject_22
   {
      TClaPx_TheSVPROBClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_22    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         id_jet               (director, ffPrefix, "id_jet"),
         tagword              (director, ffPrefix, "tagword"),
         mult                 (director, ffPrefix, "mult"),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         decayLen             (director, ffPrefix, "decayLen"),
         decayLenSig          (director, ffPrefix, "decayLenSig"),
         chi2                 (director, ffPrefix, "chi2"),
         mass                 (director, ffPrefix, "mass"),
         collinearity         (director, ffPrefix, "collinearity"),
         direction            (director, ffPrefix, "direction"),
         probability          (director, ffPrefix, "probability"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         p                    (director, ffPrefix, "p[3]"),
         isKShort             (director, ffPrefix, "isKShort"),
         pt                   (director, ffPrefix, "pt")
      {};
      TClaPx_TheSVPROBClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_22    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         id_jet               (director, ffPrefix, "id_jet"),
         tagword              (director, ffPrefix, "tagword"),
         mult                 (director, ffPrefix, "mult"),
         vertex               (director, ffPrefix, "vertex[3]"),
         vertexerr            (director, ffPrefix, "vertexerr[6]"),
         decayLen             (director, ffPrefix, "decayLen"),
         decayLenSig          (director, ffPrefix, "decayLenSig"),
         chi2                 (director, ffPrefix, "chi2"),
         mass                 (director, ffPrefix, "mass"),
         collinearity         (director, ffPrefix, "collinearity"),
         direction            (director, ffPrefix, "direction"),
         probability          (director, ffPrefix, "probability"),
         eta                  (director, ffPrefix, "eta"),
         phi                  (director, ffPrefix, "phi"),
         p                    (director, ffPrefix, "p[3]"),
         isKShort             (director, ffPrefix, "isKShort"),
         pt                   (director, ffPrefix, "pt")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy          id_jet;
      TClaIntProxy          tagword;
      TClaIntProxy          mult;
      TClaArrayFloatProxy   vertex;
      TClaArrayFloatProxy   vertexerr;
      TClaFloatProxy        decayLen;
      TClaFloatProxy        decayLenSig;
      TClaFloatProxy        chi2;
      TClaFloatProxy        mass;
      TClaFloatProxy        collinearity;
      TClaFloatProxy        direction;
      TClaFloatProxy        probability;
      TClaFloatProxy        eta;
      TClaFloatProxy        phi;
      TClaArrayFloatProxy   p;
      TClaIntProxy          isKShort;
      TClaFloatProxy        pt;
   };
   struct TClaPx_TObject_23
   {
      TClaPx_TObject_23(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_23(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_ThePreShowerClass
      : public TClaPx_TObject_23
   {
      TClaPx_ThePreShowerClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_23(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         clus_type       (director, ffPrefix, "clus_type"),
         clus_ix         (director, ffPrefix, "clus_ix"),
         clus_iu         (director, ffPrefix, "clus_iu"),
         clus_iv         (director, ffPrefix, "clus_iv"),
         clus_e          (director, ffPrefix, "clus_e"),
         clus_phi        (director, ffPrefix, "clus_phi"),
         clus_z          (director, ffPrefix, "clus_z"),
         clus_dphi       (director, ffPrefix, "clus_dphi"),
         clus_dz         (director, ffPrefix, "clus_dz"),
         clus_ex         (director, ffPrefix, "clus_ex"),
         clus_eu         (director, ffPrefix, "clus_eu"),
         clus_ev         (director, ffPrefix, "clus_ev"),
         clus_r          (director, ffPrefix, "clus_r")
      {};
      TClaPx_ThePreShowerClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_23(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         clus_type       (director, ffPrefix, "clus_type"),
         clus_ix         (director, ffPrefix, "clus_ix"),
         clus_iu         (director, ffPrefix, "clus_iu"),
         clus_iv         (director, ffPrefix, "clus_iv"),
         clus_e          (director, ffPrefix, "clus_e"),
         clus_phi        (director, ffPrefix, "clus_phi"),
         clus_z          (director, ffPrefix, "clus_z"),
         clus_dphi       (director, ffPrefix, "clus_dphi"),
         clus_dz         (director, ffPrefix, "clus_dz"),
         clus_ex         (director, ffPrefix, "clus_ex"),
         clus_eu         (director, ffPrefix, "clus_eu"),
         clus_ev         (director, ffPrefix, "clus_ev"),
         clus_r          (director, ffPrefix, "clus_r")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy     clus_type;
      TClaIntProxy     clus_ix;
      TClaIntProxy     clus_iu;
      TClaIntProxy     clus_iv;
      TClaFloatProxy   clus_e;
      TClaFloatProxy   clus_phi;
      TClaFloatProxy   clus_z;
      TClaFloatProxy   clus_dphi;
      TClaFloatProxy   clus_dz;
      TClaFloatProxy   clus_ex;
      TClaFloatProxy   clus_eu;
      TClaFloatProxy   clus_ev;
      TClaFloatProxy   clus_r;
   };
   struct TClaPx_TObject_24
   {
      TClaPx_TObject_24(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_24(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheTauClass
      : public TClaPx_TObject_24
   {
      TClaPx_TheTauClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_24  (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         pt                 (director, ffPrefix, "pt"),
         px                 (director, ffPrefix, "px"),
         py                 (director, ffPrefix, "py"),
         pz                 (director, ffPrefix, "pz"),
         eta                (director, ffPrefix, "eta"),
         phi                (director, ffPrefix, "phi"),
         e                  (director, ffPrefix, "e"),
         iso                (director, ffPrefix, "iso"),
         profile            (director, ffPrefix, "profile"),
         ettr               (director, ffPrefix, "ettr"),
         rms                (director, ffPrefix, "rms"),
         nn                 (director, ffPrefix, "nn"),
         emf                (director, ffPrefix, "emf"),
         icdf               (director, ffPrefix, "icdf"),
         chf                (director, ffPrefix, "chf"),
         hotf               (director, ffPrefix, "hotf"),
         em12f              (director, ffPrefix, "em12f"),
         em3f               (director, ffPrefix, "em3f"),
         em4f               (director, ffPrefix, "em4f"),
         em12isof           (director, ffPrefix, "em12isof"),
         em3isof            (director, ffPrefix, "em3isof"),
         em4isof            (director, ffPrefix, "em4isof"),
         ntrk               (director, ffPrefix, "ntrk"),
         ntrk_cone_1        (director, ffPrefix, "ntrk_cone_1"),
         ntrk_cone_2        (director, ffPrefix, "ntrk_cone_2"),
         ntrk_cone_3        (director, ffPrefix, "ntrk_cone_3"),
         trkidx             (director, ffPrefix, "trkidx[3]"),
         nem3               (director, ffPrefix, "nem3"),
         empt               (director, ffPrefix, "empt"),
         emphi              (director, ffPrefix, "emphi"),
         emeta              (director, ffPrefix, "emeta"),
         emm                (director, ffPrefix, "emm"),
         emcl_f12           (director, ffPrefix, "emcl_f12"),
         emcl_f3            (director, ffPrefix, "emcl_f3"),
         emcl_f4            (director, ffPrefix, "emcl_f4")
      {};
      TClaPx_TheTauClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_24  (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         pt                 (director, ffPrefix, "pt"),
         px                 (director, ffPrefix, "px"),
         py                 (director, ffPrefix, "py"),
         pz                 (director, ffPrefix, "pz"),
         eta                (director, ffPrefix, "eta"),
         phi                (director, ffPrefix, "phi"),
         e                  (director, ffPrefix, "e"),
         iso                (director, ffPrefix, "iso"),
         profile            (director, ffPrefix, "profile"),
         ettr               (director, ffPrefix, "ettr"),
         rms                (director, ffPrefix, "rms"),
         nn                 (director, ffPrefix, "nn"),
         emf                (director, ffPrefix, "emf"),
         icdf               (director, ffPrefix, "icdf"),
         chf                (director, ffPrefix, "chf"),
         hotf               (director, ffPrefix, "hotf"),
         em12f              (director, ffPrefix, "em12f"),
         em3f               (director, ffPrefix, "em3f"),
         em4f               (director, ffPrefix, "em4f"),
         em12isof           (director, ffPrefix, "em12isof"),
         em3isof            (director, ffPrefix, "em3isof"),
         em4isof            (director, ffPrefix, "em4isof"),
         ntrk               (director, ffPrefix, "ntrk"),
         ntrk_cone_1        (director, ffPrefix, "ntrk_cone_1"),
         ntrk_cone_2        (director, ffPrefix, "ntrk_cone_2"),
         ntrk_cone_3        (director, ffPrefix, "ntrk_cone_3"),
         trkidx             (director, ffPrefix, "trkidx[3]"),
         nem3               (director, ffPrefix, "nem3"),
         empt               (director, ffPrefix, "empt"),
         emphi              (director, ffPrefix, "emphi"),
         emeta              (director, ffPrefix, "emeta"),
         emm                (director, ffPrefix, "emm"),
         emcl_f12           (director, ffPrefix, "emcl_f12"),
         emcl_f3            (director, ffPrefix, "emcl_f3"),
         emcl_f4            (director, ffPrefix, "emcl_f4")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy      pt;
      TClaFloatProxy      px;
      TClaFloatProxy      py;
      TClaFloatProxy      pz;
      TClaFloatProxy      eta;
      TClaFloatProxy      phi;
      TClaFloatProxy      e;
      TClaFloatProxy      iso;
      TClaFloatProxy      profile;
      TClaFloatProxy      ettr;
      TClaFloatProxy      rms;
      TClaFloatProxy      nn;
      TClaFloatProxy      emf;
      TClaFloatProxy      icdf;
      TClaFloatProxy      chf;
      TClaFloatProxy      hotf;
      TClaFloatProxy      em12f;
      TClaFloatProxy      em3f;
      TClaFloatProxy      em4f;
      TClaFloatProxy      em12isof;
      TClaFloatProxy      em3isof;
      TClaFloatProxy      em4isof;
      TClaIntProxy        ntrk;
      TClaIntProxy        ntrk_cone_1;
      TClaIntProxy        ntrk_cone_2;
      TClaIntProxy        ntrk_cone_3;
      TClaArrayIntProxy   trkidx;
      TClaIntProxy        nem3;
      TClaFloatProxy      empt;
      TClaFloatProxy      emphi;
      TClaFloatProxy      emeta;
      TClaFloatProxy      emm;
      TClaFloatProxy      emcl_f12;
      TClaFloatProxy      emcl_f3;
      TClaFloatProxy      emcl_f4;
   };
   struct TPx_TheObjectClass
      : public TPx_TObject
   {
      TPx_TheObjectClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject                     (director, top, mid),
         ffPrefix                        (top,mid),
         obj                             (director, top, mid),
         MemArray                        (director, "MemArray"),
         nobj                            (director, "nobj"),
         MuonArray                       (director, "MuonArray"),
         Nmuons                          (director, "Nmuons"),
         EMArray                         (director, "EMArray"),
         Nems                            (director, "Nems"),
         SEMArray                        (director, "SEMArray"),
         Nsems                           (director, "Nsems"),
         JetArray                        (director, "JetArray"),
         Njets                           (director, "Njets"),
         BadJetArray                     (director, "BadJetArray"),
         Nbadjets                        (director, "Nbadjets"),
         VtxArray                        (director, "VtxArray"),
         Nvtx                            (director, "Nvtx"),
         NewVtxArray                     (director, "NewVtxArray"),
         Nnewvtx                         (director, "Nnewvtx"),
         CellArray                       (director, "CellArray"),
         Ncells                          (director, "Ncells"),
         TowerArray                      (director, "TowerArray"),
         Ntowers                         (director, "Ntowers"),
         TrackArray                      (director, "TrackArray"),
         Ntracks                         (director, "Ntracks"),
         TrackClusterArray               (director, "TrackClusterArray"),
         Ntrackclusters                  (director, "Ntrackclusters"),
         MCParticleArray                 (director, "MCParticleArray"),
         Nmcp                            (director, "Nmcp"),
         MCVertexArray                   (director, "MCVertexArray"),
         Nmcv                            (director, "Nmcv"),
         TrackJetArray                   (director, "TrackJetArray"),
         Ntrackjets                      (director, "Ntrackjets"),
         SecondaryVertexArray            (director, "SecondaryVertexArray"),
         Nsecvertex                      (director, "Nsecvertex"),
         SVPROBArray                     (director, "SVPROBArray"),
         NSVPROB                         (director, "NSVPROB"),
         PreShowerArray                  (director, "PreShowerArray"),
         Npreshower                      (director, "Npreshower"),
         TauArray                        (director, "TauArray"),
         Ntaus                           (director, "Ntaus")
      {};
      TPx_TheObjectClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject                     (director, parent, membername),
         ffPrefix                        (""),
         obj                             (director, parent, membername),
         MemArray                        (director, "MemArray"),
         nobj                            (director, "nobj"),
         MuonArray                       (director, "MuonArray"),
         Nmuons                          (director, "Nmuons"),
         EMArray                         (director, "EMArray"),
         Nems                            (director, "Nems"),
         SEMArray                        (director, "SEMArray"),
         Nsems                           (director, "Nsems"),
         JetArray                        (director, "JetArray"),
         Njets                           (director, "Njets"),
         BadJetArray                     (director, "BadJetArray"),
         Nbadjets                        (director, "Nbadjets"),
         VtxArray                        (director, "VtxArray"),
         Nvtx                            (director, "Nvtx"),
         NewVtxArray                     (director, "NewVtxArray"),
         Nnewvtx                         (director, "Nnewvtx"),
         CellArray                       (director, "CellArray"),
         Ncells                          (director, "Ncells"),
         TowerArray                      (director, "TowerArray"),
         Ntowers                         (director, "Ntowers"),
         TrackArray                      (director, "TrackArray"),
         Ntracks                         (director, "Ntracks"),
         TrackClusterArray               (director, "TrackClusterArray"),
         Ntrackclusters                  (director, "Ntrackclusters"),
         MCParticleArray                 (director, "MCParticleArray"),
         Nmcp                            (director, "Nmcp"),
         MCVertexArray                   (director, "MCVertexArray"),
         Nmcv                            (director, "Nmcv"),
         TrackJetArray                   (director, "TrackJetArray"),
         Ntrackjets                      (director, "Ntrackjets"),
         SecondaryVertexArray            (director, "SecondaryVertexArray"),
         Nsecvertex                      (director, "Nsecvertex"),
         SVPROBArray                     (director, "SVPROBArray"),
         NSVPROB                         (director, "NSVPROB"),
         PreShowerArray                  (director, "PreShowerArray"),
         Npreshower                      (director, "Npreshower"),
         TauArray                        (director, "TauArray"),
         Ntaus                           (director, "Ntaus")
      {};
      TBranchProxyHelper               ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TClaPx_TheIntClass_2             MemArray;
      TIntProxy                        nobj;
      TClaPx_TheMuonClass              MuonArray;
      TIntProxy                        Nmuons;
      TClaPx_TheElectronClass          EMArray;
      TIntProxy                        Nems;
      TClaPx_TheSoftElectronClass      SEMArray;
      TIntProxy                        Nsems;
      TClaPx_TheJetClass               JetArray;
      TIntProxy                        Njets;
      TClaPx_TheJetClass_1             BadJetArray;
      TIntProxy                        Nbadjets;
      TClaPx_TheVertexClass            VtxArray;
      TIntProxy                        Nvtx;
      TClaPx_TheVertexClass_1          NewVtxArray;
      TIntProxy                        Nnewvtx;
      TClaPx_TheCellClass              CellArray;
      TIntProxy                        Ncells;
      TClaPx_TheCellClass_1            TowerArray;
      TIntProxy                        Ntowers;
      TClaPx_TheTrackClass             TrackArray;
      TIntProxy                        Ntracks;
      TClaPx_TheTrackClusterClass      TrackClusterArray;
      TIntProxy                        Ntrackclusters;
      TClaPx_TheMCParticleClass        MCParticleArray;
      TIntProxy                        Nmcp;
      TClaPx_TheMCVertexClass          MCVertexArray;
      TIntProxy                        Nmcv;
      TClaPx_TheTrackJetClass          TrackJetArray;
      TIntProxy                        Ntrackjets;
      TClaPx_TheSecondaryVertexClass   SecondaryVertexArray;
      TIntProxy                        Nsecvertex;
      TClaPx_TheSVPROBClass            SVPROBArray;
      TIntProxy                        NSVPROB;
      TClaPx_ThePreShowerClass         PreShowerArray;
      TIntProxy                        Npreshower;
      TClaPx_TheTauClass               TauArray;
      TIntProxy                        Ntaus;
   };
   struct TClaPx_TObject_25
   {
      TClaPx_TObject_25(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_25(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheCellClass_2
      : public TClaPx_TObject_25
   {
      TClaPx_TheCellClass_2(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_25(director, top, mid),
         ffPrefix        (top,mid),
         obj             (director, top, mid),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TClaPx_TheCellClass_2(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_25(director, parent, membername),
         ffPrefix        (""),
         obj             (director, parent, membername),
         ieta            (director, ffPrefix, "ieta"),
         iphi            (director, ffPrefix, "iphi"),
         ilyr            (director, ffPrefix, "ilyr"),
         e               (director, ffPrefix, "e"),
         phi             (director, ffPrefix, "phi"),
         eta             (director, ffPrefix, "eta"),
         pt              (director, ffPrefix, "pt")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaIntProxy     ieta;
      TClaIntProxy     iphi;
      TClaIntProxy     ilyr;
      TClaFloatProxy   e;
      TClaFloatProxy   phi;
      TClaFloatProxy   eta;
      TClaFloatProxy   pt;
   };
   struct TPx_TheMissingEtClass
      : public TPx_TObject
   {
      TPx_TheMissingEtClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject              (director, top, mid),
         ffPrefix                 (top,mid),
         obj                      (director, top, mid),
         NadaArray                (director, "NadaArray"),
         Nnada                    (director, "Nnada"),
         cal_set                  (director, "cal_set"),
         cal_met                  (director, "cal_met"),
         cal_metx                 (director, "cal_metx"),
         cal_mety                 (director, "cal_mety"),
         cal_phi                  (director, "cal_phi"),
         ch_met                   (director, "ch_met"),
         ch_metx                  (director, "ch_metx"),
         ch_mety                  (director, "ch_mety"),
         jes_met                  (director, "jes_met"),
         jes_metx                 (director, "jes_metx"),
         jes_mety                 (director, "jes_mety"),
         ej_met                   (director, "ej_met"),
         ej_metx                  (director, "ej_metx"),
         ej_mety                  (director, "ej_mety"),
         mj_met                   (director, "mj_met"),
         mj_metx                  (director, "mj_metx"),
         mj_mety                  (director, "mj_mety"),
         mj_phi                   (director, "mj_phi"),
         dphi_mu                  (director, "dphi_mu"),
         dphi_em                  (director, "dphi_em"),
         nada_met                 (director, "nada_met"),
         nada_metx                (director, "nada_metx"),
         nada_mety                (director, "nada_mety"),
         nada_phi                 (director, "nada_phi"),
         ues                      (director, "ues"),
         uex                      (director, "uex"),
         uey                      (director, "uey"),
         fjet_es                  (director, "fjet_es"),
         fjet_ex                  (director, "fjet_ex"),
         fjet_ey                  (director, "fjet_ey"),
         set_all                  (director, "set_all"),
         set_pos                  (director, "set_pos"),
         set_neg                  (director, "set_neg"),
         towerEM_met              (director, "towerEM_met"),
         towerEM_metx             (director, "towerEM_metx"),
         towerEM_mety             (director, "towerEM_mety"),
         towerEM_phi              (director, "towerEM_phi"),
         ncells                   (director, "ncells[2][18]")
      {};
      TPx_TheMissingEtClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject              (director, parent, membername),
         ffPrefix                 (""),
         obj                      (director, parent, membername),
         NadaArray                (director, "NadaArray"),
         Nnada                    (director, "Nnada"),
         cal_set                  (director, "cal_set"),
         cal_met                  (director, "cal_met"),
         cal_metx                 (director, "cal_metx"),
         cal_mety                 (director, "cal_mety"),
         cal_phi                  (director, "cal_phi"),
         ch_met                   (director, "ch_met"),
         ch_metx                  (director, "ch_metx"),
         ch_mety                  (director, "ch_mety"),
         jes_met                  (director, "jes_met"),
         jes_metx                 (director, "jes_metx"),
         jes_mety                 (director, "jes_mety"),
         ej_met                   (director, "ej_met"),
         ej_metx                  (director, "ej_metx"),
         ej_mety                  (director, "ej_mety"),
         mj_met                   (director, "mj_met"),
         mj_metx                  (director, "mj_metx"),
         mj_mety                  (director, "mj_mety"),
         mj_phi                   (director, "mj_phi"),
         dphi_mu                  (director, "dphi_mu"),
         dphi_em                  (director, "dphi_em"),
         nada_met                 (director, "nada_met"),
         nada_metx                (director, "nada_metx"),
         nada_mety                (director, "nada_mety"),
         nada_phi                 (director, "nada_phi"),
         ues                      (director, "ues"),
         uex                      (director, "uex"),
         uey                      (director, "uey"),
         fjet_es                  (director, "fjet_es"),
         fjet_ex                  (director, "fjet_ex"),
         fjet_ey                  (director, "fjet_ey"),
         set_all                  (director, "set_all"),
         set_pos                  (director, "set_pos"),
         set_neg                  (director, "set_neg"),
         towerEM_met              (director, "towerEM_met"),
         towerEM_metx             (director, "towerEM_metx"),
         towerEM_mety             (director, "towerEM_mety"),
         towerEM_phi              (director, "towerEM_phi"),
         ncells                   (director, "ncells[2][18]")
      {};
      TBranchProxyHelper        ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TClaPx_TheCellClass_2     NadaArray;
      TIntProxy                 Nnada;
      TFloatProxy               cal_set;
      TFloatProxy               cal_met;
      TFloatProxy               cal_metx;
      TFloatProxy               cal_mety;
      TFloatProxy               cal_phi;
      TFloatProxy               ch_met;
      TFloatProxy               ch_metx;
      TFloatProxy               ch_mety;
      TFloatProxy               jes_met;
      TFloatProxy               jes_metx;
      TFloatProxy               jes_mety;
      TFloatProxy               ej_met;
      TFloatProxy               ej_metx;
      TFloatProxy               ej_mety;
      TFloatProxy               mj_met;
      TFloatProxy               mj_metx;
      TFloatProxy               mj_mety;
      TFloatProxy               mj_phi;
      TFloatProxy               dphi_mu;
      TFloatProxy               dphi_em;
      TFloatProxy               nada_met;
      TFloatProxy               nada_metx;
      TFloatProxy               nada_mety;
      TFloatProxy               nada_phi;
      TFloatProxy               ues;
      TFloatProxy               uex;
      TFloatProxy               uey;
      TFloatProxy               fjet_es;
      TFloatProxy               fjet_ex;
      TFloatProxy               fjet_ey;
      TFloatProxy               set_all;
      TFloatProxy               set_pos;
      TFloatProxy               set_neg;
      TFloatProxy               towerEM_met;
      TFloatProxy               towerEM_metx;
      TFloatProxy               towerEM_mety;
      TFloatProxy               towerEM_phi;
      TArray2Proxy<Int_t,18 >   ncells;
   };
   struct TClaPx_TObject_26
   {
      TClaPx_TObject_26(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix       (top,mid),
         obj            (director, top, mid),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TClaPx_TObject_26(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix       (""),
         obj            (director, parent, membername),
         fUniqueID      (director, ffPrefix, "fUniqueID"),
         fBits          (director, ffPrefix, "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TObject > obj;

      TClaUIntProxy   fUniqueID;
      TClaUIntProxy   fBits;
   };
   struct TClaPx_TheTopFitClass
      : public TClaPx_TObject_26
   {
      TClaPx_TheTopFitClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TClaPx_TObject_26    (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         chisq                (director, ffPrefix, "chisq"),
         mt                   (director, ffPrefix, "mt"),
         sigmt                (director, ffPrefix, "sigmt"),
         umwhad               (director, ffPrefix, "umwhad"),
         utmass               (director, ffPrefix, "utmass"),
         m_tt                 (director, ffPrefix, "m_tt"),
         pt_t                 (director, ffPrefix, "pt_t"),
         kt                   (director, ffPrefix, "kt"),
         eta_t1               (director, ffPrefix, "eta_t1"),
         eta_t2               (director, ffPrefix, "eta_t2"),
         phi_t1               (director, ffPrefix, "phi_t1"),
         phi_t2               (director, ffPrefix, "phi_t2"),
         pt_t1                (director, ffPrefix, "pt_t1"),
         pt_t2                (director, ffPrefix, "pt_t2"),
         perm                 (director, ffPrefix, "perm"),
         eta_w1               (director, ffPrefix, "eta_w1"),
         eta_w2               (director, ffPrefix, "eta_w2"),
         phi_w1               (director, ffPrefix, "phi_w1"),
         phi_w2               (director, ffPrefix, "phi_w2"),
         pt_w1                (director, ffPrefix, "pt_w1"),
         pt_w2                (director, ffPrefix, "pt_w2"),
         jet_lepb             (director, ffPrefix, "jet_lepb[4]"),
         jet_hadb             (director, ffPrefix, "jet_hadb[4]"),
         jet_hadw1            (director, ffPrefix, "jet_hadw1[4]"),
         jet_hadw2            (director, ffPrefix, "jet_hadw2[4]"),
         lepton               (director, ffPrefix, "lepton[4]"),
         neutrino             (director, ffPrefix, "neutrino[4]"),
         bchisq               (director, ffPrefix, "bchisq"),
         bmt                  (director, ffPrefix, "bmt"),
         bsigmt               (director, ffPrefix, "bsigmt"),
         bumwhad              (director, ffPrefix, "bumwhad"),
         butmass              (director, ffPrefix, "butmass"),
         bm_tt                (director, ffPrefix, "bm_tt"),
         bpt_t                (director, ffPrefix, "bpt_t"),
         bkt                  (director, ffPrefix, "bkt"),
         beta_t1              (director, ffPrefix, "beta_t1"),
         beta_t2              (director, ffPrefix, "beta_t2"),
         bphi_t1              (director, ffPrefix, "bphi_t1"),
         bphi_t2              (director, ffPrefix, "bphi_t2"),
         bpt_t1               (director, ffPrefix, "bpt_t1"),
         bpt_t2               (director, ffPrefix, "bpt_t2"),
         bperm                (director, ffPrefix, "bperm"),
         beta_w1              (director, ffPrefix, "beta_w1"),
         beta_w2              (director, ffPrefix, "beta_w2"),
         bphi_w1              (director, ffPrefix, "bphi_w1"),
         bphi_w2              (director, ffPrefix, "bphi_w2"),
         bpt_w1               (director, ffPrefix, "bpt_w1"),
         bpt_w2               (director, ffPrefix, "bpt_w2"),
         bjet_lepb            (director, ffPrefix, "bjet_lepb[4]"),
         bjet_hadb            (director, ffPrefix, "bjet_hadb[4]"),
         bjet_hadw1           (director, ffPrefix, "bjet_hadw1[4]"),
         bjet_hadw2           (director, ffPrefix, "bjet_hadw2[4]"),
         blepton              (director, ffPrefix, "blepton[4]"),
         bneutrino            (director, ffPrefix, "bneutrino[4]")
      {};
      TClaPx_TheTopFitClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TClaPx_TObject_26    (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         chisq                (director, ffPrefix, "chisq"),
         mt                   (director, ffPrefix, "mt"),
         sigmt                (director, ffPrefix, "sigmt"),
         umwhad               (director, ffPrefix, "umwhad"),
         utmass               (director, ffPrefix, "utmass"),
         m_tt                 (director, ffPrefix, "m_tt"),
         pt_t                 (director, ffPrefix, "pt_t"),
         kt                   (director, ffPrefix, "kt"),
         eta_t1               (director, ffPrefix, "eta_t1"),
         eta_t2               (director, ffPrefix, "eta_t2"),
         phi_t1               (director, ffPrefix, "phi_t1"),
         phi_t2               (director, ffPrefix, "phi_t2"),
         pt_t1                (director, ffPrefix, "pt_t1"),
         pt_t2                (director, ffPrefix, "pt_t2"),
         perm                 (director, ffPrefix, "perm"),
         eta_w1               (director, ffPrefix, "eta_w1"),
         eta_w2               (director, ffPrefix, "eta_w2"),
         phi_w1               (director, ffPrefix, "phi_w1"),
         phi_w2               (director, ffPrefix, "phi_w2"),
         pt_w1                (director, ffPrefix, "pt_w1"),
         pt_w2                (director, ffPrefix, "pt_w2"),
         jet_lepb             (director, ffPrefix, "jet_lepb[4]"),
         jet_hadb             (director, ffPrefix, "jet_hadb[4]"),
         jet_hadw1            (director, ffPrefix, "jet_hadw1[4]"),
         jet_hadw2            (director, ffPrefix, "jet_hadw2[4]"),
         lepton               (director, ffPrefix, "lepton[4]"),
         neutrino             (director, ffPrefix, "neutrino[4]"),
         bchisq               (director, ffPrefix, "bchisq"),
         bmt                  (director, ffPrefix, "bmt"),
         bsigmt               (director, ffPrefix, "bsigmt"),
         bumwhad              (director, ffPrefix, "bumwhad"),
         butmass              (director, ffPrefix, "butmass"),
         bm_tt                (director, ffPrefix, "bm_tt"),
         bpt_t                (director, ffPrefix, "bpt_t"),
         bkt                  (director, ffPrefix, "bkt"),
         beta_t1              (director, ffPrefix, "beta_t1"),
         beta_t2              (director, ffPrefix, "beta_t2"),
         bphi_t1              (director, ffPrefix, "bphi_t1"),
         bphi_t2              (director, ffPrefix, "bphi_t2"),
         bpt_t1               (director, ffPrefix, "bpt_t1"),
         bpt_t2               (director, ffPrefix, "bpt_t2"),
         bperm                (director, ffPrefix, "bperm"),
         beta_w1              (director, ffPrefix, "beta_w1"),
         beta_w2              (director, ffPrefix, "beta_w2"),
         bphi_w1              (director, ffPrefix, "bphi_w1"),
         bphi_w2              (director, ffPrefix, "bphi_w2"),
         bpt_w1               (director, ffPrefix, "bpt_w1"),
         bpt_w2               (director, ffPrefix, "bpt_w2"),
         bjet_lepb            (director, ffPrefix, "bjet_lepb[4]"),
         bjet_hadb            (director, ffPrefix, "bjet_hadb[4]"),
         bjet_hadw1           (director, ffPrefix, "bjet_hadw1[4]"),
         bjet_hadw2           (director, ffPrefix, "bjet_hadw2[4]"),
         blepton              (director, ffPrefix, "blepton[4]"),
         bneutrino            (director, ffPrefix, "bneutrino[4]")
      {};
      TBranchProxyHelper    ffPrefix;
      InjecTBranchProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaFloatProxy        chisq;
      TClaFloatProxy        mt;
      TClaFloatProxy        sigmt;
      TClaFloatProxy        umwhad;
      TClaFloatProxy        utmass;
      TClaFloatProxy        m_tt;
      TClaFloatProxy        pt_t;
      TClaFloatProxy        kt;
      TClaFloatProxy        eta_t1;
      TClaFloatProxy        eta_t2;
      TClaFloatProxy        phi_t1;
      TClaFloatProxy        phi_t2;
      TClaFloatProxy        pt_t1;
      TClaFloatProxy        pt_t2;
      TClaIntProxy          perm;
      TClaFloatProxy        eta_w1;
      TClaFloatProxy        eta_w2;
      TClaFloatProxy        phi_w1;
      TClaFloatProxy        phi_w2;
      TClaFloatProxy        pt_w1;
      TClaFloatProxy        pt_w2;
      TClaArrayFloatProxy   jet_lepb;
      TClaArrayFloatProxy   jet_hadb;
      TClaArrayFloatProxy   jet_hadw1;
      TClaArrayFloatProxy   jet_hadw2;
      TClaArrayFloatProxy   lepton;
      TClaArrayFloatProxy   neutrino;
      TClaFloatProxy        bchisq;
      TClaFloatProxy        bmt;
      TClaFloatProxy        bsigmt;
      TClaFloatProxy        bumwhad;
      TClaFloatProxy        butmass;
      TClaFloatProxy        bm_tt;
      TClaFloatProxy        bpt_t;
      TClaFloatProxy        bkt;
      TClaFloatProxy        beta_t1;
      TClaFloatProxy        beta_t2;
      TClaFloatProxy        bphi_t1;
      TClaFloatProxy        bphi_t2;
      TClaFloatProxy        bpt_t1;
      TClaFloatProxy        bpt_t2;
      TClaIntProxy          bperm;
      TClaFloatProxy        beta_w1;
      TClaFloatProxy        beta_w2;
      TClaFloatProxy        bphi_w1;
      TClaFloatProxy        bphi_w2;
      TClaFloatProxy        bpt_w1;
      TClaFloatProxy        bpt_w2;
      TClaArrayFloatProxy   bjet_lepb;
      TClaArrayFloatProxy   bjet_hadb;
      TClaArrayFloatProxy   bjet_hadw1;
      TClaArrayFloatProxy   bjet_hadw2;
      TClaArrayFloatProxy   blepton;
      TClaArrayFloatProxy   bneutrino;
   };
   struct TPx_ThePropertyClass
      : public TPx_TObject
   {
      TPx_ThePropertyClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject            (director, top, mid),
         ffPrefix               (top,mid),
         obj                    (director, top, mid),
         TopFitArray            (director, "TopFitArray"),
         Nperm                  (director, "Nperm")
      {};
      TPx_ThePropertyClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject            (director, parent, membername),
         ffPrefix               (""),
         obj                    (director, parent, membername),
         TopFitArray            (director, "TopFitArray"),
         Nperm                  (director, "Nperm")
      {};
      TBranchProxyHelper      ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TClaPx_TheTopFitClass   TopFitArray;
      TIntProxy               Nperm;
   };
   struct TPx_TObject_1
   {
      TPx_TObject_1(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix    (top,mid),
         obj         (director, top, mid),
         fUniqueID   (director, obj.proxy(), "fUniqueID"),
         fBits       (director, obj.proxy(), "fBits")
      {};
      TPx_TObject_1(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         ffPrefix    (""),
         obj         (director, parent, membername),
         fUniqueID   (director, obj.proxy(), "fUniqueID"),
         fBits       (director, obj.proxy(), "fBits")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      const TObject* operator->() { return obj.ptr(); }
      TObjProxy<TObject > obj;

      TUIntProxy   fUniqueID;
      TUIntProxy   fBits;
   };
   struct TPx_TheEJetsClass
      : public TPx_TObject_1
   {
      TPx_TheEJetsClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         EJets_mass   (director, ffPrefix, "mass"),
         EJets_mt     (director, ffPrefix, "mt"),
         EJets_ht20   (director, ffPrefix, "ht20"),
         EJets_ht25   (director, ffPrefix, "ht25"),
         EJets_apla   (director, ffPrefix, "apla"),
         EJets_sphe   (director, ffPrefix, "sphe"),
         EJets_plan   (director, ffPrefix, "plan"),
         EJets_dphi_mete(director, ffPrefix, "dphi_mete"),
         EJets_Ht2p   (director, ffPrefix, "Ht2p"),
         EJets_Ktminp (director, ffPrefix, "Ktminp"),
         EJets_ETL    (director, ffPrefix, "ETL"),
         EJets_EtaW   (director, ffPrefix, "EtaW"),
         EJets_Pznu   (director, ffPrefix, "Pznu"),
         EJets_PxW    (director, ffPrefix, "PxW"),
         EJets_PyW    (director, ffPrefix, "PyW"),
         EJets_PzW    (director, ffPrefix, "PzW")
      {};
      TPx_TheEJetsClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix     (""),
         obj          (director, parent, membername),
         EJets_mass   (director, ffPrefix, "mass"),
         EJets_mt     (director, ffPrefix, "mt"),
         EJets_ht20   (director, ffPrefix, "ht20"),
         EJets_ht25   (director, ffPrefix, "ht25"),
         EJets_apla   (director, ffPrefix, "apla"),
         EJets_sphe   (director, ffPrefix, "sphe"),
         EJets_plan   (director, ffPrefix, "plan"),
         EJets_dphi_mete(director, ffPrefix, "dphi_mete"),
         EJets_Ht2p   (director, ffPrefix, "Ht2p"),
         EJets_Ktminp (director, ffPrefix, "Ktminp"),
         EJets_ETL    (director, ffPrefix, "ETL"),
         EJets_EtaW   (director, ffPrefix, "EtaW"),
         EJets_Pznu   (director, ffPrefix, "Pznu"),
         EJets_PxW    (director, ffPrefix, "PxW"),
         EJets_PyW    (director, ffPrefix, "PyW"),
         EJets_PzW    (director, ffPrefix, "PzW")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TFloatProxy   EJets_mass;
      TFloatProxy   EJets_mt;
      TFloatProxy   EJets_ht20;
      TFloatProxy   EJets_ht25;
      TFloatProxy   EJets_apla;
      TFloatProxy   EJets_sphe;
      TFloatProxy   EJets_plan;
      TFloatProxy   EJets_dphi_mete;
      TFloatProxy   EJets_Ht2p;
      TFloatProxy   EJets_Ktminp;
      TFloatProxy   EJets_ETL;
      TFloatProxy   EJets_EtaW;
      TFloatProxy   EJets_Pznu;
      TFloatProxy   EJets_PxW;
      TFloatProxy   EJets_PyW;
      TFloatProxy   EJets_PzW;
   };
   struct TPx_TheMuJetsClass
      : public TPx_TObject_1
   {
      TPx_TheMuJetsClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         MuJets_mass  (director, ffPrefix, "mass"),
         MuJets_mt    (director, ffPrefix, "mt"),
         MuJets_ht20  (director, ffPrefix, "ht20"),
         MuJets_ht25  (director, ffPrefix, "ht25"),
         MuJets_apla  (director, ffPrefix, "apla"),
         MuJets_sphe  (director, ffPrefix, "sphe"),
         MuJets_plan  (director, ffPrefix, "plan"),
         MuJets_dphi_metmu(director, ffPrefix, "dphi_metmu"),
         MuJets_Ht2p  (director, ffPrefix, "Ht2p"),
         MuJets_Ktminp(director, ffPrefix, "Ktminp"),
         MuJets_ETL   (director, ffPrefix, "ETL"),
         MuJets_EtaW  (director, ffPrefix, "EtaW"),
         MuJets_Pznu  (director, ffPrefix, "Pznu"),
         MuJets_PxW   (director, ffPrefix, "PxW"),
         MuJets_PyW   (director, ffPrefix, "PyW"),
         MuJets_PzW   (director, ffPrefix, "PzW"),
         MuJets_MuIdx (director, ffPrefix, "MuIdx"),
         MuJets_Isolated(director, ffPrefix, "Isolated"),
         MuJets_MatchQual(director, ffPrefix, "MatchQual"),
         MuJets_Pt    (director, ffPrefix, "Pt"),
         MuJets_dRJet (director, ffPrefix, "dRJet"),
         MuJets_DcaSignif(director, ffPrefix, "DcaSignif"),
         MuJets_Eta   (director, ffPrefix, "Eta"),
         MuJets_MatchChi2(director, ffPrefix, "MatchChi2"),
         MuJets_JetEt (director, ffPrefix, "JetEt")
      {};
      TPx_TheMuJetsClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix     (""),
         obj          (director, parent, membername),
         MuJets_mass  (director, ffPrefix, "mass"),
         MuJets_mt    (director, ffPrefix, "mt"),
         MuJets_ht20  (director, ffPrefix, "ht20"),
         MuJets_ht25  (director, ffPrefix, "ht25"),
         MuJets_apla  (director, ffPrefix, "apla"),
         MuJets_sphe  (director, ffPrefix, "sphe"),
         MuJets_plan  (director, ffPrefix, "plan"),
         MuJets_dphi_metmu(director, ffPrefix, "dphi_metmu"),
         MuJets_Ht2p  (director, ffPrefix, "Ht2p"),
         MuJets_Ktminp(director, ffPrefix, "Ktminp"),
         MuJets_ETL   (director, ffPrefix, "ETL"),
         MuJets_EtaW  (director, ffPrefix, "EtaW"),
         MuJets_Pznu  (director, ffPrefix, "Pznu"),
         MuJets_PxW   (director, ffPrefix, "PxW"),
         MuJets_PyW   (director, ffPrefix, "PyW"),
         MuJets_PzW   (director, ffPrefix, "PzW"),
         MuJets_MuIdx (director, ffPrefix, "MuIdx"),
         MuJets_Isolated(director, ffPrefix, "Isolated"),
         MuJets_MatchQual(director, ffPrefix, "MatchQual"),
         MuJets_Pt    (director, ffPrefix, "Pt"),
         MuJets_dRJet (director, ffPrefix, "dRJet"),
         MuJets_DcaSignif(director, ffPrefix, "DcaSignif"),
         MuJets_Eta   (director, ffPrefix, "Eta"),
         MuJets_MatchChi2(director, ffPrefix, "MatchChi2"),
         MuJets_JetEt (director, ffPrefix, "JetEt")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TFloatProxy   MuJets_mass;
      TFloatProxy   MuJets_mt;
      TFloatProxy   MuJets_ht20;
      TFloatProxy   MuJets_ht25;
      TFloatProxy   MuJets_apla;
      TFloatProxy   MuJets_sphe;
      TFloatProxy   MuJets_plan;
      TFloatProxy   MuJets_dphi_metmu;
      TFloatProxy   MuJets_Ht2p;
      TFloatProxy   MuJets_Ktminp;
      TFloatProxy   MuJets_ETL;
      TFloatProxy   MuJets_EtaW;
      TFloatProxy   MuJets_Pznu;
      TFloatProxy   MuJets_PxW;
      TFloatProxy   MuJets_PyW;
      TFloatProxy   MuJets_PzW;
      TIntProxy     MuJets_MuIdx;
      TIntProxy     MuJets_Isolated;
      TIntProxy     MuJets_MatchQual;
      TFloatProxy   MuJets_Pt;
      TFloatProxy   MuJets_dRJet;
      TFloatProxy   MuJets_DcaSignif;
      TFloatProxy   MuJets_Eta;
      TFloatProxy   MuJets_MatchChi2;
      TFloatProxy   MuJets_JetEt;
   };
   struct TPx_TheEMUClass
      : public TPx_TObject_1
   {
      TPx_TheEMUClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         EMU_mass     (director, ffPrefix, "mass"),
         EMU_mz       (director, ffPrefix, "mz")
      {};
      TPx_TheEMUClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix     (""),
         obj          (director, parent, membername),
         EMU_mass     (director, ffPrefix, "mass"),
         EMU_mz       (director, ffPrefix, "mz")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TFloatProxy   EMU_mass;
      TFloatProxy   EMU_mz;
   };
   struct TPx_TheDiMuonClass
      : public TPx_TObject_1
   {
      TPx_TheDiMuonClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         DiMuon_mass  (director, ffPrefix, "mass"),
         DiMuon_mz    (director, ffPrefix, "mz"),
         DiMuon_charge(director, ffPrefix, "charge"),
         DiMuon_niso  (director, ffPrefix, "niso"),
         DiMuon_idxm1 (director, ffPrefix, "idxm1"),
         DiMuon_idxm2 (director, ffPrefix, "idxm2")
      {};
      TPx_TheDiMuonClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix     (""),
         obj          (director, parent, membername),
         DiMuon_mass  (director, ffPrefix, "mass"),
         DiMuon_mz    (director, ffPrefix, "mz"),
         DiMuon_charge(director, ffPrefix, "charge"),
         DiMuon_niso  (director, ffPrefix, "niso"),
         DiMuon_idxm1 (director, ffPrefix, "idxm1"),
         DiMuon_idxm2 (director, ffPrefix, "idxm2")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TFloatProxy   DiMuon_mass;
      TFloatProxy   DiMuon_mz;
      TIntProxy     DiMuon_charge;
      TIntProxy     DiMuon_niso;
      TIntProxy     DiMuon_idxm1;
      TIntProxy     DiMuon_idxm2;
   };
   struct TPx_TheDiEMClass
      : public TPx_TObject_1
   {
      TPx_TheDiEMClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         DiEM_mass    (director, ffPrefix, "mass"),
         DiEM_mz      (director, ffPrefix, "mz"),
         DiEM_mtrk    (director, ffPrefix, "mtrk"),
         DiEM_charge  (director, ffPrefix, "charge"),
         DiEM_matches (director, ffPrefix, "matches")
      {};
      TPx_TheDiEMClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix     (""),
         obj          (director, parent, membername),
         DiEM_mass    (director, ffPrefix, "mass"),
         DiEM_mz      (director, ffPrefix, "mz"),
         DiEM_mtrk    (director, ffPrefix, "mtrk"),
         DiEM_charge  (director, ffPrefix, "charge"),
         DiEM_matches (director, ffPrefix, "matches")
      {};
      TBranchProxyHelper ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TFloatProxy   DiEM_mass;
      TFloatProxy   DiEM_mz;
      TFloatProxy   DiEM_mtrk;
      TIntProxy     DiEM_charge;
      TIntProxy     DiEM_matches;
   };
   struct TPx_TheAllJetsClass
      : public TPx_TObject_1
   {
      TPx_TheAllJetsClass(TBranchProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1      (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         AllJets_aplan      (director, ffPrefix, "aplan"),
         AllJets_plan       (director, ffPrefix, "plan"),
         AllJets_spher      (director, ffPrefix, "spher"),
         AllJets_ht         (director, ffPrefix, "ht"),
         AllJets_h          (director, ffPrefix, "h"),
         AllJets_ht3        (director, ffPrefix, "ht3"),
         AllJets_sqrts      (director, ffPrefix, "sqrts"),
         AllJets_centr      (director, ffPrefix, "centr"),
         AllJets_boost      (director, ffPrefix, "boost[3]"),
         AllJets_njets      (director, ffPrefix, "njets"),
         AllJets_njetsweighed(director, ffPrefix, "njetsweighed")
      {};
      TPx_TheAllJetsClass(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :
         TPx_TObject_1      (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         AllJets_aplan      (director, ffPrefix, "aplan"),
         AllJets_plan       (director, ffPrefix, "plan"),
         AllJets_spher      (director, ffPrefix, "spher"),
         AllJets_ht         (director, ffPrefix, "ht"),
         AllJets_h          (director, ffPrefix, "h"),
         AllJets_ht3        (director, ffPrefix, "ht3"),
         AllJets_sqrts      (director, ffPrefix, "sqrts"),
         AllJets_centr      (director, ffPrefix, "centr"),
         AllJets_boost      (director, ffPrefix, "boost[3]"),
         AllJets_njets      (director, ffPrefix, "njets"),
         AllJets_njetsweighed(director, ffPrefix, "njetsweighed")
      {};
      TBranchProxyHelper  ffPrefix;
      InjecTBranchProxyInterface();
      TBranchProxy obj;

      TDoubleProxy        AllJets_aplan;
      TDoubleProxy        AllJets_plan;
      TDoubleProxy        AllJets_spher;
      TDoubleProxy        AllJets_ht;
      TDoubleProxy        AllJets_h;
      TDoubleProxy        AllJets_ht3;
      TDoubleProxy        AllJets_sqrts;
      TDoubleProxy        AllJets_centr;
      TArrayDoubleProxy   AllJets_boost;
      TIntProxy           AllJets_njets;
      TDoubleProxy        AllJets_njetsweighed;
   };

   // Proxy for each of the branches, leaves and friends of the tree
   TPx_TheEventClass                  TopEvent;
   TPx_TObject                        baseTObject;
   TClaPx_TheExecutableVersionClass   ExecutableVersionArray;
   TIntProxy                          Nexecutables;
   TClaPx_TheL3NameClass              L3NameArray;
   TIntProxy                          l3name_n;
   TClaPx_TheL2NameClass              L2NameArray;
   TIntProxy                          l2name_n;
   TClaPx_TheL1NameClass              L1NameArray;
   TIntProxy                          l1name_n;
   TClaPx_TheIntClass                 L1PrescalesArray;
   TIntProxy                          l1prescales_n;
   TClaPx_TheIntClass_1               AndOrTermArray;
   TIntProxy                          aoterm_n;
   TIntProxy                          runnum;
   TIntProxy                          evtnum;
   TIntProxy                          ticknum;
   TIntProxy                          lumblk;
   TIntProxy                          solpol;
   TIntProxy                          torpol;
   TIntProxy                          emptyCrate;
   TIntProxy                          coherentNoise;
   TIntProxy                          ringOfFire;
   TIntProxy                          nojets;
   TArrayFloatProxy                   vertex;
   TArrayFloatProxy                   vertexErr;
   TIntProxy                          vertexNtrack;
   TIntProxy                          vertexNdof;
   TFloatProxy                        vertexChi2;
   TFloatProxy                        ht20;
   TFloatProxy                        ht25;
   TFloatProxy                        mc_cross;
   TFloatProxy                        mc_wt;
   TArrayFloatProxy                   NewVtx;
   TArrayFloatProxy                   NewVtxErr;
   TIntProxy                          NewVtxTrks;
   TFloatProxy                        NVmbprob;
   TFloatProxy                        NVSumPt;
   TFloatProxy                        NVSumLogPt;
   TFloatProxy                        NVHighestPt;
   TFloatProxy                        NewVtxChi2;
   TIntProxy                          NewVtxNdof;
   TIntProxy                          tottwr;
   TIntProxy                          twr_l1et_lt_n1_em;
   TIntProxy                          twr_l1et_gt_2_em;
   TIntProxy                          twr_diff_lt_n1_em;
   TIntProxy                          twr_diff_inbt_em;
   TIntProxy                          twr_diff_gt_1_em;
   TIntProxy                          twr_l1et_lt_n1_had;
   TIntProxy                          twr_l1et_gt_2_had;
   TIntProxy                          twr_diff_lt_n1_had;
   TIntProxy                          twr_diff_inbt_had;
   TIntProxy                          twr_diff_gt_1_had;
   TFloatProxy                        sumdiff_em;
   TFloatProxy                        suml1_em;
   TFloatProxy                        maxdiff_em;
   TIntProxy                          twr_rat_lt_05_em;
   TIntProxy                          twr_rat_gt_2_em;
   TFloatProxy                        sumdiff_had;
   TFloatProxy                        suml1_had;
   TFloatProxy                        maxdiff_had;
   TIntProxy                          twr_rat_lt_05_had;
   TIntProxy                          twr_rat_gt_2_had;
   TPx_TheObjectClass                 Objects;
   TClaPx_TheIntClass_2               MemArray;
   TIntProxy                          nobj;
   TClaPx_TheMuonClass                MuonArray;
   TIntProxy                          Nmuons;
   TClaPx_TheElectronClass            EMArray;
   TIntProxy                          Nems;
   TClaPx_TheSoftElectronClass        SEMArray;
   TIntProxy                          Nsems;
   TClaPx_TheJetClass                 JetArray;
   TIntProxy                          Njets;
   TClaPx_TheJetClass_1               BadJetArray;
   TIntProxy                          Nbadjets;
   TClaPx_TheVertexClass              VtxArray;
   TIntProxy                          Nvtx;
   TClaPx_TheVertexClass_1            NewVtxArray;
   TIntProxy                          Nnewvtx;
   TClaPx_TheCellClass                CellArray;
   TIntProxy                          Ncells;
   TClaPx_TheCellClass_1              TowerArray;
   TIntProxy                          Ntowers;
   TClaPx_TheTrackClass               TrackArray;
   TIntProxy                          Ntracks;
   TClaPx_TheTrackClusterClass        TrackClusterArray;
   TIntProxy                          Ntrackclusters;
   TClaPx_TheMCParticleClass          MCParticleArray;
   TIntProxy                          Nmcp;
   TClaPx_TheMCVertexClass            MCVertexArray;
   TIntProxy                          Nmcv;
   TClaPx_TheTrackJetClass            TrackJetArray;
   TIntProxy                          Ntrackjets;
   TClaPx_TheSecondaryVertexClass     SecondaryVertexArray;
   TIntProxy                          Nsecvertex;
   TClaPx_TheSVPROBClass              SVPROBArray;
   TIntProxy                          NSVPROB;
   TClaPx_ThePreShowerClass           PreShowerArray;
   TIntProxy                          Npreshower;
   TClaPx_TheTauClass                 TauArray;
   TIntProxy                          Ntaus;
   TPx_TheMissingEtClass              MissingEt;
   TClaPx_TheCellClass_2              NadaArray;
   TIntProxy                          Nnada;
   TFloatProxy                        cal_set;
   TFloatProxy                        cal_met;
   TFloatProxy                        cal_metx;
   TFloatProxy                        cal_mety;
   TFloatProxy                        cal_phi;
   TFloatProxy                        ch_met;
   TFloatProxy                        ch_metx;
   TFloatProxy                        ch_mety;
   TFloatProxy                        jes_met;
   TFloatProxy                        jes_metx;
   TFloatProxy                        jes_mety;
   TFloatProxy                        ej_met;
   TFloatProxy                        ej_metx;
   TFloatProxy                        ej_mety;
   TFloatProxy                        mj_met;
   TFloatProxy                        mj_metx;
   TFloatProxy                        mj_mety;
   TFloatProxy                        mj_phi;
   TFloatProxy                        dphi_mu;
   TFloatProxy                        dphi_em;
   TFloatProxy                        nada_met;
   TFloatProxy                        nada_metx;
   TFloatProxy                        nada_mety;
   TFloatProxy                        nada_phi;
   TFloatProxy                        ues;
   TFloatProxy                        uex;
   TFloatProxy                        uey;
   TFloatProxy                        fjet_es;
   TFloatProxy                        fjet_ex;
   TFloatProxy                        fjet_ey;
   TFloatProxy                        set_all;
   TFloatProxy                        set_pos;
   TFloatProxy                        set_neg;
   TFloatProxy                        towerEM_met;
   TFloatProxy                        towerEM_metx;
   TFloatProxy                        towerEM_mety;
   TFloatProxy                        towerEM_phi;
   TArray2Proxy<Int_t,18 >            ncells;
   TPx_ThePropertyClass               Property;
   TClaPx_TheTopFitClass              TopFitArray;
   TIntProxy                          Nperm;
   TPx_TheEJetsClass                  EJets;
   TFloatProxy                        EJets_mass;
   TFloatProxy                        EJets_mt;
   TFloatProxy                        EJets_ht20;
   TFloatProxy                        EJets_ht25;
   TFloatProxy                        EJets_apla;
   TFloatProxy                        EJets_sphe;
   TFloatProxy                        EJets_plan;
   TFloatProxy                        EJets_dphi_mete;
   TFloatProxy                        EJets_Ht2p;
   TFloatProxy                        EJets_Ktminp;
   TFloatProxy                        EJets_ETL;
   TFloatProxy                        EJets_EtaW;
   TFloatProxy                        EJets_Pznu;
   TFloatProxy                        EJets_PxW;
   TFloatProxy                        EJets_PyW;
   TFloatProxy                        EJets_PzW;
   TPx_TheMuJetsClass                 MuJets;
   TFloatProxy                        MuJets_mass;
   TFloatProxy                        MuJets_mt;
   TFloatProxy                        MuJets_ht20;
   TFloatProxy                        MuJets_ht25;
   TFloatProxy                        MuJets_apla;
   TFloatProxy                        MuJets_sphe;
   TFloatProxy                        MuJets_plan;
   TFloatProxy                        MuJets_dphi_metmu;
   TFloatProxy                        MuJets_Ht2p;
   TFloatProxy                        MuJets_Ktminp;
   TFloatProxy                        MuJets_ETL;
   TFloatProxy                        MuJets_EtaW;
   TFloatProxy                        MuJets_Pznu;
   TFloatProxy                        MuJets_PxW;
   TFloatProxy                        MuJets_PyW;
   TFloatProxy                        MuJets_PzW;
   TIntProxy                          MuJets_MuIdx;
   TIntProxy                          MuJets_Isolated;
   TIntProxy                          MuJets_MatchQual;
   TFloatProxy                        MuJets_Pt;
   TFloatProxy                        MuJets_dRJet;
   TFloatProxy                        MuJets_DcaSignif;
   TFloatProxy                        MuJets_Eta;
   TFloatProxy                        MuJets_MatchChi2;
   TFloatProxy                        MuJets_JetEt;
   TPx_TheEMUClass                    EMU;
   TFloatProxy                        EMU_mass;
   TFloatProxy                        EMU_mz;
   TPx_TheDiMuonClass                 DiMuon;
   TFloatProxy                        DiMuon_mass;
   TFloatProxy                        DiMuon_mz;
   TIntProxy                          DiMuon_charge;
   TIntProxy                          DiMuon_niso;
   TIntProxy                          DiMuon_idxm1;
   TIntProxy                          DiMuon_idxm2;
   TPx_TheDiEMClass                   DiEM;
   TFloatProxy                        DiEM_mass;
   TFloatProxy                        DiEM_mz;
   TFloatProxy                        DiEM_mtrk;
   TIntProxy                          DiEM_charge;
   TIntProxy                          DiEM_matches;
   TPx_TheAllJetsClass                AllJets;
   TDoubleProxy                       AllJets_aplan;
   TDoubleProxy                       AllJets_plan;
   TDoubleProxy                       AllJets_spher;
   TDoubleProxy                       AllJets_ht;
   TDoubleProxy                       AllJets_h;
   TDoubleProxy                       AllJets_ht3;
   TDoubleProxy                       AllJets_sqrts;
   TDoubleProxy                       AllJets_centr;
   TArrayDoubleProxy                  AllJets_boost;
   TIntProxy                          AllJets_njets;
   TDoubleProxy                       AllJets_njetsweighed;


   analyzeTop(TTree *tree=0) : 
      fChain(0),
      fHelper(0),
      fInput(0),
      htemp(0),
      fDirector(tree,-1),
      fClass                (gROOT->GetClass("analyzeTop")),
      fBeginMethod          (fClass,"printToptree_Begin","(TTree*)0"),
      fSlaveBeginMethod     (fClass,"printToptree_SlaveBegin","(TTree*)0"),
      fNotifyMethod         (fClass,"printToptree_Notify",""),
      fProcessMethod        (fClass,"printToptree_Process","0"),
      fSlaveTerminateMethod (fClass,"printToptree_SlaveTerminate",""),
      fTerminateMethod      (fClass,"printToptree_Terminate",""),
      TopEvent                          (&fDirector,"TopEvent"),
      baseTObject                       (&fDirector,"TObject"),
      ExecutableVersionArray            (&fDirector,"ExecutableVersionArray"),
      Nexecutables                      (&fDirector,"Nexecutables"),
      L3NameArray                       (&fDirector,"L3NameArray"),
      l3name_n                          (&fDirector,"l3name_n"),
      L2NameArray                       (&fDirector,"L2NameArray"),
      l2name_n                          (&fDirector,"l2name_n"),
      L1NameArray                       (&fDirector,"L1NameArray"),
      l1name_n                          (&fDirector,"l1name_n"),
      L1PrescalesArray                  (&fDirector,"L1PrescalesArray"),
      l1prescales_n                     (&fDirector,"l1prescales_n"),
      AndOrTermArray                    (&fDirector,"AndOrTermArray"),
      aoterm_n                          (&fDirector,"aoterm_n"),
      runnum                            (&fDirector,"runnum"),
      evtnum                            (&fDirector,"evtnum"),
      ticknum                           (&fDirector,"ticknum"),
      lumblk                            (&fDirector,"lumblk"),
      solpol                            (&fDirector,"solpol"),
      torpol                            (&fDirector,"torpol"),
      emptyCrate                        (&fDirector,"emptyCrate"),
      coherentNoise                     (&fDirector,"coherentNoise"),
      ringOfFire                        (&fDirector,"ringOfFire"),
      nojets                            (&fDirector,"nojets"),
      vertex                            (&fDirector,"vertex[3]"),
      vertexErr                         (&fDirector,"vertexErr[6]"),
      vertexNtrack                      (&fDirector,"vertexNtrack"),
      vertexNdof                        (&fDirector,"vertexNdof"),
      vertexChi2                        (&fDirector,"vertexChi2"),
      ht20                              (&fDirector,"ht20"),
      ht25                              (&fDirector,"ht25"),
      mc_cross                          (&fDirector,"mc_cross"),
      mc_wt                             (&fDirector,"mc_wt"),
      NewVtx                            (&fDirector,"NewVtx[3]"),
      NewVtxErr                         (&fDirector,"NewVtxErr[6]"),
      NewVtxTrks                        (&fDirector,"NewVtxTrks"),
      NVmbprob                          (&fDirector,"NVmbprob"),
      NVSumPt                           (&fDirector,"NVSumPt"),
      NVSumLogPt                        (&fDirector,"NVSumLogPt"),
      NVHighestPt                       (&fDirector,"NVHighestPt"),
      NewVtxChi2                        (&fDirector,"NewVtxChi2"),
      NewVtxNdof                        (&fDirector,"NewVtxNdof"),
      tottwr                            (&fDirector,"tottwr"),
      twr_l1et_lt_n1_em                 (&fDirector,"twr_l1et_lt_n1_em"),
      twr_l1et_gt_2_em                  (&fDirector,"twr_l1et_gt_2_em"),
      twr_diff_lt_n1_em                 (&fDirector,"twr_diff_lt_n1_em"),
      twr_diff_inbt_em                  (&fDirector,"twr_diff_inbt_em"),
      twr_diff_gt_1_em                  (&fDirector,"twr_diff_gt_1_em"),
      twr_l1et_lt_n1_had                (&fDirector,"twr_l1et_lt_n1_had"),
      twr_l1et_gt_2_had                 (&fDirector,"twr_l1et_gt_2_had"),
      twr_diff_lt_n1_had                (&fDirector,"twr_diff_lt_n1_had"),
      twr_diff_inbt_had                 (&fDirector,"twr_diff_inbt_had"),
      twr_diff_gt_1_had                 (&fDirector,"twr_diff_gt_1_had"),
      sumdiff_em                        (&fDirector,"sumdiff_em"),
      suml1_em                          (&fDirector,"suml1_em"),
      maxdiff_em                        (&fDirector,"maxdiff_em"),
      twr_rat_lt_05_em                  (&fDirector,"twr_rat_lt_05_em"),
      twr_rat_gt_2_em                   (&fDirector,"twr_rat_gt_2_em"),
      sumdiff_had                       (&fDirector,"sumdiff_had"),
      suml1_had                         (&fDirector,"suml1_had"),
      maxdiff_had                       (&fDirector,"maxdiff_had"),
      twr_rat_lt_05_had                 (&fDirector,"twr_rat_lt_05_had"),
      twr_rat_gt_2_had                  (&fDirector,"twr_rat_gt_2_had"),
      Objects                           (&fDirector,"Objects"),
      MemArray                          (&fDirector,"MemArray"),
      nobj                              (&fDirector,"nobj"),
      MuonArray                         (&fDirector,"MuonArray"),
      Nmuons                            (&fDirector,"Nmuons"),
      EMArray                           (&fDirector,"EMArray"),
      Nems                              (&fDirector,"Nems"),
      SEMArray                          (&fDirector,"SEMArray"),
      Nsems                             (&fDirector,"Nsems"),
      JetArray                          (&fDirector,"JetArray"),
      Njets                             (&fDirector,"Njets"),
      BadJetArray                       (&fDirector,"BadJetArray"),
      Nbadjets                          (&fDirector,"Nbadjets"),
      VtxArray                          (&fDirector,"VtxArray"),
      Nvtx                              (&fDirector,"Nvtx"),
      NewVtxArray                       (&fDirector,"NewVtxArray"),
      Nnewvtx                           (&fDirector,"Nnewvtx"),
      CellArray                         (&fDirector,"CellArray"),
      Ncells                            (&fDirector,"Ncells"),
      TowerArray                        (&fDirector,"TowerArray"),
      Ntowers                           (&fDirector,"Ntowers"),
      TrackArray                        (&fDirector,"TrackArray"),
      Ntracks                           (&fDirector,"Ntracks"),
      TrackClusterArray                 (&fDirector,"TrackClusterArray"),
      Ntrackclusters                    (&fDirector,"Ntrackclusters"),
      MCParticleArray                   (&fDirector,"MCParticleArray"),
      Nmcp                              (&fDirector,"Nmcp"),
      MCVertexArray                     (&fDirector,"MCVertexArray"),
      Nmcv                              (&fDirector,"Nmcv"),
      TrackJetArray                     (&fDirector,"TrackJetArray"),
      Ntrackjets                        (&fDirector,"Ntrackjets"),
      SecondaryVertexArray              (&fDirector,"SecondaryVertexArray"),
      Nsecvertex                        (&fDirector,"Nsecvertex"),
      SVPROBArray                       (&fDirector,"SVPROBArray"),
      NSVPROB                           (&fDirector,"NSVPROB"),
      PreShowerArray                    (&fDirector,"PreShowerArray"),
      Npreshower                        (&fDirector,"Npreshower"),
      TauArray                          (&fDirector,"TauArray"),
      Ntaus                             (&fDirector,"Ntaus"),
      MissingEt                         (&fDirector,"MissingEt"),
      NadaArray                         (&fDirector,"NadaArray"),
      Nnada                             (&fDirector,"Nnada"),
      cal_set                           (&fDirector,"cal_set"),
      cal_met                           (&fDirector,"cal_met"),
      cal_metx                          (&fDirector,"cal_metx"),
      cal_mety                          (&fDirector,"cal_mety"),
      cal_phi                           (&fDirector,"cal_phi"),
      ch_met                            (&fDirector,"ch_met"),
      ch_metx                           (&fDirector,"ch_metx"),
      ch_mety                           (&fDirector,"ch_mety"),
      jes_met                           (&fDirector,"jes_met"),
      jes_metx                          (&fDirector,"jes_metx"),
      jes_mety                          (&fDirector,"jes_mety"),
      ej_met                            (&fDirector,"ej_met"),
      ej_metx                           (&fDirector,"ej_metx"),
      ej_mety                           (&fDirector,"ej_mety"),
      mj_met                            (&fDirector,"mj_met"),
      mj_metx                           (&fDirector,"mj_metx"),
      mj_mety                           (&fDirector,"mj_mety"),
      mj_phi                            (&fDirector,"mj_phi"),
      dphi_mu                           (&fDirector,"dphi_mu"),
      dphi_em                           (&fDirector,"dphi_em"),
      nada_met                          (&fDirector,"nada_met"),
      nada_metx                         (&fDirector,"nada_metx"),
      nada_mety                         (&fDirector,"nada_mety"),
      nada_phi                          (&fDirector,"nada_phi"),
      ues                               (&fDirector,"ues"),
      uex                               (&fDirector,"uex"),
      uey                               (&fDirector,"uey"),
      fjet_es                           (&fDirector,"fjet_es"),
      fjet_ex                           (&fDirector,"fjet_ex"),
      fjet_ey                           (&fDirector,"fjet_ey"),
      set_all                           (&fDirector,"set_all"),
      set_pos                           (&fDirector,"set_pos"),
      set_neg                           (&fDirector,"set_neg"),
      towerEM_met                       (&fDirector,"towerEM_met"),
      towerEM_metx                      (&fDirector,"towerEM_metx"),
      towerEM_mety                      (&fDirector,"towerEM_mety"),
      towerEM_phi                       (&fDirector,"towerEM_phi"),
      ncells                            (&fDirector,"ncells[2][18]"),
      Property                          (&fDirector,"Property"),
      TopFitArray                       (&fDirector,"TopFitArray"),
      Nperm                             (&fDirector,"Nperm"),
      EJets                             (&fDirector,"EJets"),
      EJets_mass                        (&fDirector,"EJets_mass"),
      EJets_mt                          (&fDirector,"EJets_mt"),
      EJets_ht20                        (&fDirector,"EJets_ht20"),
      EJets_ht25                        (&fDirector,"EJets_ht25"),
      EJets_apla                        (&fDirector,"EJets_apla"),
      EJets_sphe                        (&fDirector,"EJets_sphe"),
      EJets_plan                        (&fDirector,"EJets_plan"),
      EJets_dphi_mete                   (&fDirector,"EJets_dphi_mete"),
      EJets_Ht2p                        (&fDirector,"EJets_Ht2p"),
      EJets_Ktminp                      (&fDirector,"EJets_Ktminp"),
      EJets_ETL                         (&fDirector,"EJets_ETL"),
      EJets_EtaW                        (&fDirector,"EJets_EtaW"),
      EJets_Pznu                        (&fDirector,"EJets_Pznu"),
      EJets_PxW                         (&fDirector,"EJets_PxW"),
      EJets_PyW                         (&fDirector,"EJets_PyW"),
      EJets_PzW                         (&fDirector,"EJets_PzW"),
      MuJets                            (&fDirector,"MuJets"),
      MuJets_mass                       (&fDirector,"MuJets_mass"),
      MuJets_mt                         (&fDirector,"MuJets_mt"),
      MuJets_ht20                       (&fDirector,"MuJets_ht20"),
      MuJets_ht25                       (&fDirector,"MuJets_ht25"),
      MuJets_apla                       (&fDirector,"MuJets_apla"),
      MuJets_sphe                       (&fDirector,"MuJets_sphe"),
      MuJets_plan                       (&fDirector,"MuJets_plan"),
      MuJets_dphi_metmu                 (&fDirector,"MuJets_dphi_metmu"),
      MuJets_Ht2p                       (&fDirector,"MuJets_Ht2p"),
      MuJets_Ktminp                     (&fDirector,"MuJets_Ktminp"),
      MuJets_ETL                        (&fDirector,"MuJets_ETL"),
      MuJets_EtaW                       (&fDirector,"MuJets_EtaW"),
      MuJets_Pznu                       (&fDirector,"MuJets_Pznu"),
      MuJets_PxW                        (&fDirector,"MuJets_PxW"),
      MuJets_PyW                        (&fDirector,"MuJets_PyW"),
      MuJets_PzW                        (&fDirector,"MuJets_PzW"),
      MuJets_MuIdx                      (&fDirector,"MuJets_MuIdx"),
      MuJets_Isolated                   (&fDirector,"MuJets_Isolated"),
      MuJets_MatchQual                  (&fDirector,"MuJets_MatchQual"),
      MuJets_Pt                         (&fDirector,"MuJets_Pt"),
      MuJets_dRJet                      (&fDirector,"MuJets_dRJet"),
      MuJets_DcaSignif                  (&fDirector,"MuJets_DcaSignif"),
      MuJets_Eta                        (&fDirector,"MuJets_Eta"),
      MuJets_MatchChi2                  (&fDirector,"MuJets_MatchChi2"),
      MuJets_JetEt                      (&fDirector,"MuJets_JetEt"),
      EMU                               (&fDirector,"EMU"),
      EMU_mass                          (&fDirector,"EMU_mass"),
      EMU_mz                            (&fDirector,"EMU_mz"),
      DiMuon                            (&fDirector,"DiMuon"),
      DiMuon_mass                       (&fDirector,"DiMuon_mass"),
      DiMuon_mz                         (&fDirector,"DiMuon_mz"),
      DiMuon_charge                     (&fDirector,"DiMuon_charge"),
      DiMuon_niso                       (&fDirector,"DiMuon_niso"),
      DiMuon_idxm1                      (&fDirector,"DiMuon_idxm1"),
      DiMuon_idxm2                      (&fDirector,"DiMuon_idxm2"),
      DiEM                              (&fDirector,"DiEM"),
      DiEM_mass                         (&fDirector,"DiEM_mass"),
      DiEM_mz                           (&fDirector,"DiEM_mz"),
      DiEM_mtrk                         (&fDirector,"DiEM_mtrk"),
      DiEM_charge                       (&fDirector,"DiEM_charge"),
      DiEM_matches                      (&fDirector,"DiEM_matches"),
      AllJets                           (&fDirector,"AllJets"),
      AllJets_aplan                     (&fDirector,"AllJets_aplan"),
      AllJets_plan                      (&fDirector,"AllJets_plan"),
      AllJets_spher                     (&fDirector,"AllJets_spher"),
      AllJets_ht                        (&fDirector,"AllJets_ht"),
      AllJets_h                         (&fDirector,"AllJets_h"),
      AllJets_ht3                       (&fDirector,"AllJets_ht3"),
      AllJets_sqrts                     (&fDirector,"AllJets_sqrts"),
      AllJets_centr                     (&fDirector,"AllJets_centr"),
      AllJets_boost                     (&fDirector,"AllJets_boost[3]"),
      AllJets_njets                     (&fDirector,"AllJets_njets"),
      AllJets_njetsweighed              (&fDirector,"AllJets_njetsweighed")
      { }
   ~analyzeTop();
   Int_t   Version() const {return 1;}
   void    Begin(::TTree *tree);
   void    SlaveBegin(::TTree *tree);
   void    Init(::TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Long64_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    SlaveTerminate();
   void    Terminate();

   ClassDef(analyzeTop,0);


//inject the user's code
#include "printToptree.C"
};

#endif


#ifdef __MAKECINT__
#pragma link C++ class analyzeTop::TPx_TObject-;
#pragma link C++ class analyzeTop::TClaPx_TObject-;
#pragma link C++ class analyzeTop::TClaPx_TheExecutableVersionClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_1-;
#pragma link C++ class analyzeTop::TClaPx_TheL3NameClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_2-;
#pragma link C++ class analyzeTop::TClaPx_TheL2NameClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_3-;
#pragma link C++ class analyzeTop::TClaPx_TheL1NameClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_4-;
#pragma link C++ class analyzeTop::TClaPx_TheIntClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_5-;
#pragma link C++ class analyzeTop::TClaPx_TheIntClass_1-;
#pragma link C++ class analyzeTop::TPx_TheEventClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_6-;
#pragma link C++ class analyzeTop::TClaPx_TheIntClass_2-;
#pragma link C++ class analyzeTop::TClaPx_TObject_7-;
#pragma link C++ class analyzeTop::TClaPx_TheMuonClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_8-;
#pragma link C++ class analyzeTop::TClaPx_TheElectronClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_9-;
#pragma link C++ class analyzeTop::TClaPx_TheSoftElectronClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_10-;
#pragma link C++ class analyzeTop::TClaPx_TheJetClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_11-;
#pragma link C++ class analyzeTop::TClaPx_TheJetClass_1-;
#pragma link C++ class analyzeTop::TClaPx_TObject_12-;
#pragma link C++ class analyzeTop::TClaPx_TheVertexClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_13-;
#pragma link C++ class analyzeTop::TClaPx_TheVertexClass_1-;
#pragma link C++ class analyzeTop::TClaPx_TObject_14-;
#pragma link C++ class analyzeTop::TClaPx_TheCellClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_15-;
#pragma link C++ class analyzeTop::TClaPx_TheCellClass_1-;
#pragma link C++ class analyzeTop::TClaPx_TObject_16-;
#pragma link C++ class analyzeTop::TClaPx_TheTrackClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_17-;
#pragma link C++ class analyzeTop::TClaPx_TheTrackClusterClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_18-;
#pragma link C++ class analyzeTop::TClaPx_TheMCParticleClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_19-;
#pragma link C++ class analyzeTop::TClaPx_TheMCVertexClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_20-;
#pragma link C++ class analyzeTop::TClaPx_TheTrackJetClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_21-;
#pragma link C++ class analyzeTop::TClaPx_TheSecondaryVertexClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_22-;
#pragma link C++ class analyzeTop::TClaPx_TheSVPROBClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_23-;
#pragma link C++ class analyzeTop::TClaPx_ThePreShowerClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_24-;
#pragma link C++ class analyzeTop::TClaPx_TheTauClass-;
#pragma link C++ class analyzeTop::TPx_TheObjectClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_25-;
#pragma link C++ class analyzeTop::TClaPx_TheCellClass_2-;
#pragma link C++ class analyzeTop::TPx_TheMissingEtClass-;
#pragma link C++ class analyzeTop::TClaPx_TObject_26-;
#pragma link C++ class analyzeTop::TClaPx_TheTopFitClass-;
#pragma link C++ class analyzeTop::TPx_ThePropertyClass-;
#pragma link C++ class analyzeTop::TPx_TObject_1-;
#pragma link C++ class analyzeTop::TPx_TheEJetsClass-;
#pragma link C++ class analyzeTop::TPx_TheMuJetsClass-;
#pragma link C++ class analyzeTop::TPx_TheEMUClass-;
#pragma link C++ class analyzeTop::TPx_TheDiMuonClass-;
#pragma link C++ class analyzeTop::TPx_TheDiEMClass-;
#pragma link C++ class analyzeTop::TPx_TheAllJetsClass-;
#pragma link C++ class analyzeTop;
#endif


analyzeTop::~analyzeTop() {
   // destructor. Clean up helpers.

   delete fHelper;
   delete fInput;
}

void analyzeTop::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain = tree;
   fDirector.SetTree(fChain);
   fHelper = new TSelectorDraw();
   fInput  = new TList();
   fInput->Add(new TNamed("varexp","0.0")); // Fake a double size histogram
   fInput->Add(new TNamed("selection",""));
   fHelper->SetInputList(fInput);
}

Bool_t analyzeTop::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   fDirector.SetTree(fChain);
   if (fNotifyMethod.IsValid()) fNotifyMethod.Execute(this);
   
   return kTRUE;
}
   

void analyzeTop::Begin(TTree *tree)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   if (fBeginMethod.IsValid()) fBeginMethod.Execute(this,Form("0x%x",tree));

}

void analyzeTop::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   Init(tree);

   TString option = GetOption();
   fHelper->SetOption(option);
   fHelper->Begin(tree);
   htemp = (TH1*)fHelper->GetObject();
   htemp->SetTitle("printToptree.C");
   fObject = htemp;
   if (fSlaveBeginMethod.IsValid()) {
      fSlaveBeginMethod.Execute(this,Form("0x%x",tree));
   }

}

Bool_t analyzeTop::Process(Long64_t entry)
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
   printToptree();
   if (fProcessMethod.IsValid()) fProcessMethod.Execute(this,Form("%d",entry));
   return kTRUE;

}

void analyzeTop::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.
   if (fSlaveTerminateMethod.IsValid()) fSlaveTerminateMethod.Execute(this);
}

void analyzeTop::Terminate()
{
   // Function called at the end of the event loop.
   Int_t drawflag = (htemp && htemp->GetEntries()>0);
   
   if (!drawflag && !fOption.Contains("goff") && !fOption.Contains("same")) {
      gPad->Clear();
   } else {
      if (fOption.Contains("goff")) drawflag = false;
      if (drawflag) htemp->Draw(fOption);
   }
   if (fTerminateMethod.IsValid()) fTerminateMethod.Execute(this);

}
