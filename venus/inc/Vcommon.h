/* @(#)root/venus:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/



#ifndef ROOT_VCommon
#define ROOT_VCommon

extern "C" {

#ifndef __CFORTRAN_LOADED
#include "cfortran.h"
#endif

typedef struct {
  Int_t ifop;
  Int_t ifmt;
  Int_t ifch;
  Int_t ifhi;
  Int_t ifdt;
} FILES_DEF;

#define FILES COMMON_BLOCK(FILES,files)
COMMON_BLOCK_DEF(FILES_DEF,FILES);

typedef struct {
  char   fnch[80];
  Int_t  nfnch;
  char   fnhi[80];
  Int_t  nfnhi;
  char   fndt[80];
  Int_t  nfndt;
} CFNAME_DEF;

#define CFNAME COMMON_BLOCK(CFNAME,cfname)
COMMON_BLOCK_DEF(CFNAME_DEF,CFNAME);


#define VCOMMON_MAXP 1000

typedef struct {
  Int_t   np;
  Float_t tecm;
  Float_t amass[VCOMMON_MAXP];
  Int_t   ident[VCOMMON_MAXP];
  Float_t pcm[VCOMMON_MAXP][5];
  Float_t volu;
  Float_t wtxlog;
  Float_t wtlog;
} CONFIG_DEF;

#define CONFIG COMMON_BLOCK(CONFIG,config)
COMMON_BLOCK_DEF(CONFIG_DEF,CONFIG);

typedef struct {
  Double_t seedi;
  Double_t seedj;
  Double_t seedc;
} CSEED_DEF;

#define CSEED COMMON_BLOCK(CSEED,cseed)
COMMON_BLOCK_DEF(CSEED_DEF,CSEED);

typedef struct {
  Float_t delvol;
  Float_t deleps;
  Float_t dlzeta;
  Float_t etafac;
} PAROH_DEF;

#define PAROH COMMON_BLOCK(PAROH,paroh)
COMMON_BLOCK_DEF(PAROH_DEF,PAROH);

typedef struct {
  Float_t ptmx;
  Float_t gaumx;
  Float_t sigppi;
  Float_t core;
  Float_t fctrmx;
  Int_t   neqmn;
  Int_t   iaqu;
  Float_t cutmsq;
  Float_t taunll;
  Int_t   maxres;
  Float_t ptf;
  Float_t ptq;
  Float_t xcut;
  Int_t   ioptq;
  Int_t   irescl;
  Int_t   ko1ko2;
  Int_t   kentro;
  Int_t   labsys;
  Int_t   ntrymx;
  Float_t delmss;
  Float_t pud;
  Float_t pspinl;
  Float_t pspinh;
  Float_t pispn;
  Int_t   ncolmx;
  Float_t tensn;
  Float_t cutmss;
  Float_t qvapc;
  Float_t qvatc;
  Float_t qsepc;
  Float_t qsetc;
  Float_t rstras;
  Int_t   neqmx;
  Float_t taumx;
  Int_t   nsttau;
  Float_t sigj;
  Float_t pdiqua;
  Float_t parea;
  Float_t delrem;
  Float_t taumin;
  Float_t deltau;
  Float_t factau;
  Int_t   numtau;
  Int_t   iopenu;
  Int_t   iopent;
  Float_t themas;
  Float_t amsiac;
  Float_t wproj;
  Float_t wtarg;
  Int_t   iopbrk;
  Int_t   nclean;
  Int_t   ifrade;
  Float_t amprif;
  Int_t   iojint;
  Float_t pth;
  Float_t pvalen;
  Float_t phard;
  Int_t   ioptf;
  Float_t delrex;
} PARO1_DEF;

#define PARO1 COMMON_BLOCK(PARO1,paro1)
COMMON_BLOCK_DEF(PARO1_DEF,PARO1);

typedef struct {
  Int_t   nevent;
  Int_t   modsho;
  Float_t engy;
  Float_t pnll;
  Float_t pnllx;
  Float_t yhaha;
  Int_t   ish;
  Int_t   iappl;
  Float_t prosea;
  Int_t   laproj;
  Int_t   maproj;
  Int_t   latarg;
  Int_t   matarg;
  Int_t   irandm;
  Int_t   irewch;
  Int_t   istmax;
  Int_t   ipagi;
  Int_t   jpsi;
  Int_t   jpsifi;
  Float_t elepti;
  Float_t elepto;
  Float_t angmue;
  Int_t   ishsub;
  Int_t   idproj;
  Int_t   idtarg;
  Float_t amproj;
  Float_t amtarg;
  Float_t ypjtl;
  Float_t rhophi;
  Int_t   ijphis;
  Int_t   ientro;
  Int_t   kutdiq;
  Int_t   ishevt;
  Int_t   idpm;
  Float_t taurea;
} PARO2_DEF;

#define PARO2 COMMON_BLOCK(PARO2,paro2)
COMMON_BLOCK_DEF(PARO2_DEF,PARO2);

typedef struct {
  Float_t bmaxim;
  Float_t bminim;
  Float_t phimax;
  Float_t phimin;
} PAROI_DEF;

#define PAROI COMMON_BLOCK(PAROI,paroi)
COMMON_BLOCK_DEF(PAROI_DEF,PAROI);

typedef struct {
  Float_t ymximi;
  Int_t   imihis;
  Int_t   iclhis;
  Int_t   iwtime;
  Float_t wtimet;
  Float_t wtimei;
  Float_t wtimea;
} PAROF_DEF;

#define PAROF COMMON_BLOCK(PAROF,parof)
COMMON_BLOCK_DEF(PAROF_DEF,PAROF);

typedef struct {
  Int_t   isphis;
  Int_t   ispall;
  Float_t wtmini;
  Float_t wtstep;
  Int_t   iwcent;
} PAROG_DEF;

#define PAROG COMMON_BLOCK(PAROG,parog)
COMMON_BLOCK_DEF(PAROG_DEF,PAROG);

typedef struct {
  Float_t uentro;
  Float_t sigppe;
  Float_t sigppd;
  Float_t asuhax[7];
  Float_t asuhay[7];
  Float_t omega;
} PARO3_DEF;

#define PARO3 COMMON_BLOCK(PARO3,paro3)
COMMON_BLOCK_DEF(PARO3_DEF,PARO3);

typedef struct {
  Float_t grigam;
  Float_t grirsq;
  Float_t gridel;
  Float_t grislo;
  Float_t gricel;
} PARO4_DEF;

#define PARO4 COMMON_BLOCK(PARO4,paro4)
COMMON_BLOCK_DEF(PARO4_DEF,PARO4);

typedef struct {
  Float_t bag4rt;
  Float_t corlen;
  Float_t dezzer;
  Float_t amuseg;
} PARO5_DEF;

#define PARO5 COMMON_BLOCK(PARO5,paro5)
COMMON_BLOCK_DEF(PARO5_DEF,PARO5);

typedef struct {
  Int_t   iospec;
  Int_t   iocova;
  Int_t   iopair;
  Int_t   iozero;
  Int_t   ioflac;
  Int_t   iomom;
} PARO6_DEF;

#define PARO6 COMMON_BLOCK(PARO6,paro6)
COMMON_BLOCK_DEF(PARO6_DEF,PARO6);

typedef struct {
  Int_t   idensi;
  Int_t   ioclud;
  Int_t   nadd;
  Int_t   icinpu;
  Int_t   iograc;
  Int_t   iocite;
  Int_t   ioceau;
  Int_t   iociau;
} PARO7_DEF;
#define PARO7 COMMON_BLOCK(PARO7,paro7)
COMMON_BLOCK_DEF(PARO7_DEF,PARO7);

typedef struct {
  Float_t smas;
  Float_t uumas;
  Float_t usmas;
  Float_t ssmas;
  Float_t ptq1;
  Float_t ptq2;
  Float_t ptq3;
} PARO8_DEF;

#define PARO8 COMMON_BLOCK(PARO8,paro8)
COMMON_BLOCK_DEF(PARO8_DEF,PARO8);

typedef struct {
  Float_t radmes;
  Float_t radbar;
  Float_t rinmes;
  Float_t rinbar;
  Float_t epscri;
  Float_t sigmes;
  Float_t sigbar;
} PARO9_DEF;

#define PARO9 COMMON_BLOCK(PARO9,paro9)
COMMON_BLOCK_DEF(PARO9_DEF,PARO9);

typedef struct {
  Int_t   iopadi;
  Float_t q2soft;
} PAROA_DEF;

#define PAROA COMMON_BLOCK(PAROA,paroa)
COMMON_BLOCK_DEF(PAROA_DEF,PAROA);

typedef struct {
  Int_t   istore;
  Int_t   iprmpt;
  Int_t   iecho;
} PAROB_DEF;

#define PAROB COMMON_BLOCK(PAROB,parob)
COMMON_BLOCK_DEF(PAROB_DEF,PAROB);

typedef struct {
  Int_t   iostat;
  Int_t   ioinco;
  Int_t   ionlat;
  Int_t   ioobsv;
  Int_t   iosngl;
  Int_t   iorejz;
  Int_t   iompar;
} PAROC_DEF;

#define PAROC COMMON_BLOCK(PAROC,paroc)
COMMON_BLOCK_DEF(PAROC_DEF,PAROC);

typedef struct {
  Int_t   ioinfl;
  Int_t   ioinct;
  Int_t   iowidn;
  Float_t epsgc;
} PAROD_DEF;

#define PAROD COMMON_BLOCK(PAROD,parod)
COMMON_BLOCK_DEF(PAROD_DEF,PAROD);

typedef struct {
  Float_t prob[99];
  Int_t   icbac[2][99];
  Int_t   icfor[2][99];
} PAROE_DEF;

#define PAROE COMMON_BLOCK(PAROE,paroe)
COMMON_BLOCK_DEF(PAROE_DEF,PAROE);

typedef struct {
  Float_t pi;
  Float_t pii;
  Float_t hquer;
  Float_t prom;
  Float_t piom;
  Float_t ainfin;
} CNSTA_DEF;

#define CNSTA COMMON_BLOCK(CNSTA,cnsta)
COMMON_BLOCK_DEF(CNSTA_DEF,CNSTA);

typedef struct {
  Int_t   iversn;
  Int_t   iverso;
} CVSN_DEF;

#define CVSN COMMON_BLOCK(CVSN,cvsn)
COMMON_BLOCK_DEF(CVSN_DEF,CVSN);

typedef struct {
  Int_t   imsg;
  Int_t   jerr;
  Int_t   ntevt;
  Int_t   nrevt;
  Int_t   naevt;
  Int_t   nrstr;
  Int_t   nrptl;
  Float_t amsac;
  Int_t   ipage;
  Int_t   inoiac;
  Int_t   ilamas;
} ACCUM_DEF;

#define ACCUM COMMON_BLOCK(ACCUM,accum)
COMMON_BLOCK_DEF(ACCUM_DEF,ACCUM);

typedef struct {
  Int_t   nptlu;
} CPTLU_DEF;

#define CPTLU COMMON_BLOCK(CPTLU,cptlu)
COMMON_BLOCK_DEF(CPTLU_DEF,CPTLU);

typedef struct {
  Int_t   iter;
  Int_t   itermx;
  Int_t   iterma;
  Int_t   iternc;
  Int_t   iterpr;
  Int_t   iterpl;
  Int_t   iozinc;
  Int_t   iozevt;
} CITER_DEF;

#define CITER COMMON_BLOCK(CITER,citer)
COMMON_BLOCK_DEF(CITER_DEF,CITER);

typedef struct {
  Int_t   keepr;
} CMETRO_DEF;

#define CMETRO COMMON_BLOCK(CMETRO,cmetro)
COMMON_BLOCK_DEF(CMETRO_DEF,CMETRO);

typedef struct {
  Float_t epsr;
  Int_t   nepsr;
} CEPSR_DEF;

#define CEPSR COMMON_BLOCK(CEPSR,cepsr)
COMMON_BLOCK_DEF(CEPSR_DEF,CEPSR);

typedef struct {
  Int_t   keu;
  Int_t   ked;
  Int_t   kes;
  Int_t   kec;
  Int_t   keb;
  Int_t   ket;
} CINFLA_DEF;

#define CINFLA COMMON_BLOCK(CINFLA,cinfla)
COMMON_BLOCK_DEF(CINFLA_DEF,CINFLA);

#define VCOMMON_MXTAU 4
#define VCOMMON_MXVOL 10
#define VCOMMON_MXEPS 16

typedef struct {
  Float_t clust[VCOMMON_MXEPS][VCOMMON_MXVOL][VCOMMON_MXTAU];
} CJINTC_DEF;

#define CJINTC COMMON_BLOCK(CJINTC,cjintc)
COMMON_BLOCK_DEF(CJINTC_DEF,CJINTC);

typedef struct {
  Float_t volsum[VCOMMON_MXTAU];
  Float_t vo2sum[VCOMMON_MXTAU];
  Int_t   nclsum[VCOMMON_MXTAU];
} CJINTD_DEF;

#define CJINTD COMMON_BLOCK(CJINTD,cjintd)
COMMON_BLOCK_DEF(CJINTD_DEF,CJINTD);

typedef struct {
  Int_t   iutotc;
  Int_t   iutote;
} CIUTOT_DEF;

#define CIUTOT COMMON_BLOCK(CIUTOT,ciutot)
COMMON_BLOCK_DEF(CIUTOT_DEF,CIUTOT);

typedef struct {
  Int_t   nopen;
} COPEN_DEF;

#define COPEN COMMON_BLOCK(COPEN,copen)
COMMON_BLOCK_DEF(COPEN_DEF,COPEN);

#define VCOMMON_MXTRIG 99
#define VCOMMON_MXIDCO 99

typedef struct {
  char    xvaria[6];
  char    yvaria[6];
  Int_t   normal;
  Float_t xminim;
  Float_t xmaxim;
  Int_t   nrbins;
  Float_t hisfac;
} P10_DEF;

#define P10 COMMON_BLOCK(P10,p10)
COMMON_BLOCK_DEF(P10_DEF,P10);

typedef struct {
  Int_t   nrtrig;
  Float_t trmin[VCOMMON_MXTRIG];
  Float_t trmax[VCOMMON_MXTRIG];
  char    trvari[VCOMMON_MXTRIG][6];
} P11_DEF;

#define P11 COMMON_BLOCK(P11,p11)
COMMON_BLOCK_DEF(P11_DEF,P11);

typedef struct {
  Int_t   nridco;
  Int_t   idcode[VCOMMON_MXIDCO];
} P12_DEF;

#define P12 COMMON_BLOCK(P12,p12)
COMMON_BLOCK_DEF(P12_DEF,P12);

#define VCOMMON_MXNODY 200

typedef struct {
  Int_t   nrnody;
  Int_t   nody[VCOMMON_MXNODY];
  Int_t   ndecay;
} P13_DEF;

#define P13 COMMON_BLOCK(P13,p13)
COMMON_BLOCK_DEF(P13_DEF,P13);

#define VCOMMON_MXBINS 10000

typedef struct {
  Float_t ar[5][VCOMMON_MXBINS];
} CANAR_DEF;

#define CANAR COMMON_BLOCK(CANAR,canar)
COMMON_BLOCK_DEF(CANAR_DEF,CANAR);

typedef struct {
  Float_t xpar1;
  Float_t xpar2;
  Float_t xpar3;
  Float_t xpar4;
  Float_t xpar5;
  Float_t xpar6;
  Float_t xpar7;
  Float_t xpar8;
} CXPAR_DEF;

#define CXPAR COMMON_BLOCK(CXPAR,cxpar)
COMMON_BLOCK_DEF(CXPAR_DEF,CXPAR);

typedef struct {
  Float_t khisto;
} CKHIST_DEF;

#define CKHIST COMMON_BLOCK(CKHIST,ckhist)
COMMON_BLOCK_DEF(CKHIST_DEF,CKHIST);

#define VCOMMON_MXPTL 65000

typedef struct {
  Int_t   nptl;
  Float_t pptl[VCOMMON_MXPTL][5];
  Int_t   iorptl[VCOMMON_MXPTL];
  Int_t   idptl[VCOMMON_MXPTL];
  Int_t   istptl[VCOMMON_MXPTL];
  Float_t tivptl[VCOMMON_MXPTL][2];
  Int_t   ifrptl[VCOMMON_MXPTL][2];
  Int_t   jorptl[VCOMMON_MXPTL];
  Float_t xorptl[VCOMMON_MXPTL][4];
  Int_t   ibptl[VCOMMON_MXPTL][4];
  Int_t   iclptl[VCOMMON_MXPTL];
} CPTL_DEF;

#define CPTL COMMON_BLOCK(CPTL,cptl)
COMMON_BLOCK_DEF(CPTL_DEF,CPTL);

typedef struct {
  Float_t phievt;
  Int_t   nevt;
  Float_t bimevt;
  Int_t   kolevt;
  Float_t colevt;
  Float_t pmxevt;
  Float_t egyevt;
  Int_t   npjevt;
  Int_t   ntgevt;
  Int_t   npnevt;
  Int_t   nppevt;
  Int_t   ntnevt;
  Int_t   ntpevt;
  Int_t   jpnevt;
  Int_t   jppevt;
  Int_t   jtnevt;
  Int_t   jtpevt;
  Int_t   idiptl[VCOMMON_MXPTL];
  Int_t   idjptl[VCOMMON_MXPTL];
} CEVT_DEF;

#define CEVT COMMON_BLOCK(CEVT,cevt)
COMMON_BLOCK_DEF(CEVT_DEF,CEVT);

typedef struct {
  Float_t quama;
} CQUAMA_DEF;

#define CQUAMA COMMON_BLOCK(CQUAMA,cquama)
COMMON_BLOCK_DEF(CQUAMA_DEF,CQUAMA);

typedef struct {
  Int_t ipio;
} CIPIO_DEF;

#define CIPIO COMMON_BLOCK(CIPIO,cipio)
COMMON_BLOCK_DEF(CIPIO_DEF,CIPIO);

#define VCOMMON_NPTF 129

typedef struct {
  Float_t xptf[VCOMMON_NPTF];
  Float_t wptf[VCOMMON_NPTF];
  Float_t qptfu[VCOMMON_NPTF];
  Float_t qptfs[VCOMMON_NPTF];
  Float_t qptfuu[VCOMMON_NPTF];
  Float_t qptfus[VCOMMON_NPTF];
  Float_t qptfss[VCOMMON_NPTF];
} CPTF_DEF;

#define CPTF COMMON_BLOCK(CPTF,cptf)
COMMON_BLOCK_DEF(CPTF_DEF,CPTF);

#define VCOMMON_NGAU 129

typedef struct {
  Float_t xgau[VCOMMON_NGAU];
  Float_t qgau[VCOMMON_NGAU];
  Float_t wgau[VCOMMON_NGAU];
} CGAU_DEF;

#define CGAU COMMON_BLOCK(CGAU,cgau)
COMMON_BLOCK_DEF(CGAU_DEF,CGAU);

#define VCOMMON_NSPLIT 129

typedef struct {
  Float_t xsplit[VCOMMON_NSPLIT];
  Float_t qsplit[VCOMMON_NSPLIT];
  Float_t qsplix[VCOMMON_NSPLIT];
  Float_t wsplit[VCOMMON_NSPLIT];
} CSPLIT_DEF;

#define CSPLIT COMMON_BLOCK(CSPLIT,csplit)
COMMON_BLOCK_DEF(CSPLIT_DEF,CSPLIT);

#define VCOMMON_NSTRU 2049

typedef struct {
  Float_t xstru[VCOMMON_NSTRU];
  Float_t wstru[VCOMMON_NSTRU];
  Float_t qvap[VCOMMON_NSTRU];
  Float_t qvat[VCOMMON_NSTRU];
  Float_t qsep[VCOMMON_NSTRU];
  Float_t qset[VCOMMON_NSTRU];
} STRU_DEF;

#define STRU COMMON_BLOCK(STRU,stru)
COMMON_BLOCK_DEF(STRU_DEF,STRU);

#define VCOMMON_NPTJ 129

typedef struct {
  Float_t xptj[VCOMMON_NPTJ];
  Float_t qptj[VCOMMON_NPTJ];
  Float_t wptj[VCOMMON_NPTJ];
} CPTJ_DEF;

#define CPTJ COMMON_BLOCK(CPTJ,cptj)
COMMON_BLOCK_DEF(CPTJ_DEF,CPTJ);

#define VCOMMON_NPTQ 129

typedef struct {
  Float_t xptq[VCOMMON_NPTQ];
  Float_t qptq[VCOMMON_NPTQ];
  Float_t qpth[VCOMMON_NPTQ];
  Float_t wptq[VCOMMON_NPTQ];
} CPTQ_DEF;

#define CPTQ COMMON_BLOCK(CPTQ,cptq)
COMMON_BLOCK_DEF(CPTQ_DEF,CPTQ);

#define VCOMMON_NPRBMS 20

typedef struct {
  Float_t prbms[VCOMMON_NPRBMS];
} CPRBMS_DEF;

#define CPRBMS COMMON_BLOCK(CPRBMS,cprbms)
COMMON_BLOCK_DEF(CPRBMS_DEF,CPRBMS);

#define VCOMMON_NDEP 129
#define VCOMMON_NDET 129
#define VCOMMON_KOLLMX 5000

typedef struct {
  Float_t rmproj;
  Float_t rmtarg;
  Float_t bmax;
  Float_t bimp;
  Int_t   koll;
  Int_t   nproj;
  Int_t   ntarg;
  Float_t xdep[VCOMMON_NDEP];
  Float_t qdep[VCOMMON_NDEP];
  Float_t wdep[VCOMMON_NDEP];
  Float_t xdet[VCOMMON_NDET];
  Float_t qdet[VCOMMON_NDET];
  Float_t wdet[VCOMMON_NDET];
  Int_t   nrproj[VCOMMON_KOLLMX];
  Int_t   nrtarg[VCOMMON_KOLLMX];
  Float_t distce[VCOMMON_KOLLMX];
  Int_t   nord[VCOMMON_KOLLMX];
  Float_t coord[VCOMMON_KOLLMX][4];
} COL_DEF;

#define COL COMMON_BLOCK(COL,col)
COMMON_BLOCK_DEF(COL_DEF,COL);

typedef struct {
  Int_t   massnr;
} CDEN_DEF;

#define CDEN COMMON_BLOCK(CDEN,cden)
COMMON_BLOCK_DEF(CDEN_DEF,CDEN);

#define VCOMMON_MSPECS 54

typedef struct {
  Int_t   nspecs;
  Int_t   ispecs[VCOMMON_MSPECS];
  Float_t aspecs[VCOMMON_MSPECS];
  Float_t gspecs[VCOMMON_MSPECS];
} CSPECS_DEF;

#define CSPECS COMMON_BLOCK(CSPECS,cspecs)
COMMON_BLOCK_DEF(CSPECS_DEF,CSPECS);

typedef struct {
  Int_t   nlattc;
  Int_t   npmax;
} CLATT_DEF;

#define CLATT COMMON_BLOCK(CLATT,clatt)
COMMON_BLOCK_DEF(CLATT_DEF,CLATT);

typedef struct {
  Float_t   yield;
} CYIELD_DEF;

#define CYIELD COMMON_BLOCK(CYIELD,cyield)
COMMON_BLOCK_DEF(CYIELD_DEF,CYIELD);


}
#endif

