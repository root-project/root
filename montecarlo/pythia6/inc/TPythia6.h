// @(#)root/pythia6:$Id$
// Author: Rene Brun   19/10/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PYTHIA_TPythia6
#define PYTHIA_TPythia6

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TPythia6                                                                   //
//                                                                            //
// TPythia is an interface class to F77 version of Pythia 6.2                 //
// CERNLIB event generators, written by T.Sjostrand.                          //
// For the details about these generators look at Pythia/Jetset manual:       //
//                                                                            //
// ******************************************************************************
// ******************************************************************************
// **                                                                          **
// **                                                                          **
// **              *......*                  Welcome to the Lund Monte Carlo!  **
// **         *:::!!:::::::::::*                                               **
// **      *::::::!!::::::::::::::*          PPP  Y   Y TTTTT H   H III   A    **
// **    *::::::::!!::::::::::::::::*        P  P  Y Y    T   H   H  I   A A   **
// **   *:::::::::!!:::::::::::::::::*       PPP    Y     T   HHHHH  I  AAAAA  **
// **   *:::::::::!!:::::::::::::::::*       P      Y     T   H   H  I  A   A  **
// **    *::::::::!!::::::::::::::::*!       P      Y     T   H   H III A   A  **
// **      *::::::!!::::::::::::::* !!                                         **
// **      !! *:::!!:::::::::::*    !!       This is PYTHIA version 6.205      **
// **      !!     !* -><- *         !!       Last date of change:  1 Mar 2002  **
// **      !!     !!                !!                                         **
// **      !!     !!                !!       Now is  0 Jan 2000 at  0:00:00    **
// **      !!                       !!                                         **
// **      !!        lh             !!       Disclaimer: this program comes    **
// **      !!                       !!       without any guarantees. Beware    **
// **      !!                 hh    !!       of errors and use common sense    **
// **      !!    ll                 !!       when interpreting results.        **
// **      !!                       !!                                         **
// **      !!                                Copyright T. Sjostrand (2001)     **
// **                                                                          **
// ** An archive of program versions and documentation is found on the web:    **
// ** http://www.thep.lu.se/~torbjorn/Pythia.html                              **
// **                                                                          **
// ** When you cite this program, currently the official reference is          **
// ** T. Sjostrand, P. Eden, C. Friberg, L. Lonnblad, G. Miu, S. Mrenna and    **
// ** E. Norrbin, Computer Physics Commun. 135 (2001) 238.                     **
// ** The large manual is                                                      **
// ** T. Sjostrand, L. Lonnblad and S. Mrenna, LU TP 01-21 [hep-ph/0108264].   **
// ** Also remember that the program, to a large extent, represents original   **
// ** physics research. Other publications of special relevance to your        **
// ** studies may therefore deserve separate mention.                          **
// **                                                                          **
// ** Main author: Torbjorn Sjostrand; Department of Theoretical Physics 2,    **
// **   Lund University, Solvegatan 14A, S-223 62 Lund, Sweden;                **
// **   phone: + 46 - 46 - 222 48 16; e-mail: torbjorn@thep.lu.se              **
// ** Author: Leif Lonnblad; Department of Theoretical Physics 2,              **
// **   Lund University, Solvegatan 14A, S-223 62 Lund, Sweden;                **
// **   phone: + 46 - 46 - 222 77 80; e-mail: leif@thep.lu.se                  **
// ** Author: Stephen Mrenna; Computing Division, Simulations Group,           **
// **   Fermi National Accelerator Laboratory, MS 234, Batavia, IL 60510, USA; **
// **   phone: + 1 - 630 - 840 - 2556; e-mail: mrenna@fnal.gov                 **
// ** Author: Peter Skands; Department of Theoretical Physics 2,               **
// **   Lund University, Solvegatan 14A, S-223 62 Lund, Sweden;                **
// **   phone: + 46 - 46 - 222 31 92; e-mail: zeiler@thep.lu.se                **
// **                                                                          **
// **                                                                          **
// ******************************************************************************
//#ifdef __GNUG__
//#pragma interface
//#endif

#ifndef ROOT_TPythia6Calls
#include "TPythia6Calls.h"
#endif

#ifndef ROOT_TGenerator
#include "TGenerator.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TPythia6 : public TGenerator {

protected:
   static  TPythia6* fgInstance;
   // PYTHIA6 common-blocks
   Pyjets_t*  fPyjets;
   Pydat1_t*  fPydat1;
   Pydat2_t*  fPydat2;
   Pydat3_t*  fPydat3;
   Pydat4_t*  fPydat4;
   Pydatr_t*  fPydatr;
   Pysubs_t*  fPysubs;
   Pypars_t*  fPypars;
   Pyint1_t*  fPyint1;
   Pyint2_t*  fPyint2;
   Pyint3_t*  fPyint3;
   Pyint4_t*  fPyint4;
   Pyint5_t*  fPyint5;
   Pyint6_t*  fPyint6;
   Pyint7_t*  fPyint7;
   Pyint8_t*  fPyint8;
   Pyint9_t*  fPyint9;
   Pymssm_t*  fPymssm;
   Pyssmt_t*  fPyssmt;
   Pyints_t*  fPyints;
   Pybins_t*  fPybins;
   // ****** cleanup stuff (thanks Jim K.)
   class  TPythia6Cleaner {
   public:
      TPythia6Cleaner();
      ~TPythia6Cleaner();
   };
   friend class TPythia6Cleaner;

   TPythia6(const TPythia6&);            // Cannot be copied
   TPythia6& operator=(const TPythia6&); // Cannot be copied

public:
   // ****** constructors and destructor
   TPythia6();
   virtual ~TPythia6();

   static TPythia6 *Instance();

   // ****** accessors
   // FORTRAN indexing in accessing the arrays,
   // indices start from 1 !!!!!

   // ****** access to PYTHIA6 common-blocks

   // ****** /PYJETS/

   Pyjets_t*   GetPyjets        ()           { return fPyjets; }
   int         GetN             ()           { return fPyjets->N; }
   int         GetNPAD          ()           { return fPyjets->NPAD; }
   int         GetK(int ip, int i)           { return fPyjets->K[i-1][ip-1]; }
   double      GetP(int ip, int i)           { return fPyjets->P[i-1][ip-1]; }
   double      GetV(int ip, int i)           { return fPyjets->V[i-1][ip-1]; }

   void        SetN     (int n)              { fPyjets->N = n;    }
   void        SetNPAD  (int n)              { fPyjets->NPAD = n;    }
   void        SetK(int ip, int i, int k)    { fPyjets->K[i-1][ip-1] = k; }
   void        SetP(int ip, int i, double p) { fPyjets->P[i-1][ip-1] = p;    }
   void        SetV(int ip, int i, double v) { fPyjets->V[i-1][ip-1] = v;    }

   // ****** /PYDAT1/

   Pydat1_t*   GetPydat1   () { return fPydat1; }
   int         GetMSTU(int i) { return fPydat1->MSTU[i-1]; }
   double      GetPARU(int i) { return fPydat1->PARU[i-1]; }
   int         GetMSTJ(int i) { return fPydat1->MSTJ[i-1]; }
   double      GetPARJ(int i) { return fPydat1->PARJ[i-1]; }

   void        SetMSTU(int i, int m   ) { fPydat1->MSTU[i-1] = m; }
   void        SetPARU(int i, double p) { fPydat1->PARU[i-1] = p; }
   void        SetMSTJ(int i, int m   ) { fPydat1->MSTJ[i-1] = m; }
   void        SetPARJ(int i, double p) { fPydat1->PARJ[i-1] = p; }

   // ****** /PYDAT2/

   Pydat2_t*   GetPydat2           () { return fPydat2; }
   int         GetKCHG(int ip, int i) { return fPydat2->KCHG[i-1][ip-1]; }
   double      GetPMAS(int ip, int i) { return fPydat2->PMAS[i-1][ip-1]; }
   double      GetPARF        (int i) { return fPydat2->PARF[i-1]; }
   double      GetVCKM(int i,  int j) { return fPydat2->VCKM[j-1][i-1]; }

   void        SetKCHG(int ip, int i, int k   ) { fPydat2->KCHG[i-1][ip-1] = k; }
   void        SetPMAS(int ip, int i, double m) { fPydat2->PMAS[i-1][ip-1] = m; }
   void        SetPARF        (int i, double p) { fPydat2->PARF[i-1]       = p; }
   void        SetVCKM (int i, int j, double v) { fPydat2->VCKM[j-1][i-1]  = v; }

   // ****** /PYDAT3/

   Pydat3_t*   GetPydat3() { return fPydat3; }
   int         GetMDCY(int i, int j) { return fPydat3->MDCY[j-1][i-1]; }
   int         GetMDME(int i, int j) { return fPydat3->MDME[j-1][i-1]; }
   double      GetBRAT       (int i) { return fPydat3->BRAT[i-1]; }
   int         GetKFDP(int i, int j) { return fPydat3->KFDP[j-1][i-1]; }

   void        SetMDCY(int i, int j, int m) { fPydat3->MDCY[j-1][i-1] = m; }
   void        SetMDME(int i, int j, int m) { fPydat3->MDME[j-1][i-1] = m; }
   void        SetBRAT(int i, double b)     { fPydat3->BRAT[i-1]      = b; }
   void        SetKFDP(int i, int j, int k) { fPydat3->KFDP[j-1][i-1] = k; }

   // ****** /PYDAT4/

   Pydat4_t*   GetPydat4() { return fPydat4; }

   // ****** /PYDATR/ - random number generator info

   Pydatr_t*   GetPydatr   () { return fPydatr; }
   int         GetMRPY(int i) { return fPydatr->MRPY[i-1]; }
   double      GetRRPY(int i) { return fPydatr->RRPY[i-1]; }

   void        SetMRPY(int i, int m)    { fPydatr->MRPY[i-1] = m; }
   void        SetRRPY(int i, double r) { fPydatr->RRPY[i-1] = r; }

   // ****** /PYSUBS/

   Pysubs_t*   GetPysubs     () { return fPysubs; }
   int         GetMSEL       () { return fPysubs->MSEL; }
   int         GetMSELPD     () { return fPysubs->MSELPD; }
   int         GetMSUB  (int i) { return fPysubs->MSUB[i-1]; }
   double      GetCKIN  (int i) { return fPysubs->CKIN[i-1]; }
   Int_t       GetKFIN(int i, int j)  {return fPysubs->KFIN[j+40][i-1]; }

   void        SetMSEL   (int m)           { fPysubs->MSEL      = m; }
   void        SetMSELPD (int m)           { fPysubs->MSELPD    = m; }
   void        SetMSUB   (int i, int m)    { fPysubs->MSUB[i-1] = m; }
   void        SetCKIN   (int i, double c) { fPysubs->CKIN[i-1] = c; }
   void        SetKFIN(int i, int j, Int_t kfin=1) { fPysubs->KFIN[j+40][i-1] = kfin; }

   // ****** /PYPARS/

   Pypars_t*   GetPypars() { return fPypars; }
   int         GetMSTP(int i) { return fPypars->MSTP[i-1]; }
   double      GetPARP(int i) { return fPypars->PARP[i-1]; }
   int         GetMSTI(int i) { return fPypars->MSTI[i-1]; }
   double      GetPARI(int i) { return fPypars->PARI[i-1]; }

   void        SetMSTP   (int i, int    m) { fPypars->MSTP[i-1] = m; }
   void        SetPARP   (int i, double p) { fPypars->PARP[i-1] = p; }
   void        SetMSTI   (int i, int    m) { fPypars->MSTI[i-1] = m; }
   void        SetPARI   (int i, double p) { fPypars->PARI[i-1] = p; }

   // ****** /PYINT1/

   Pyint1_t*   GetPyint1() { return fPyint1; }
   int         GetMINT(int i) { return fPyint1->MINT[i-1]; }
   double      GetVINT(int i) { return fPyint1->VINT[i-1]; }

   void        SetMINT(int i, int m   ) { fPyint1->MINT[i-1] = m; }
   void        SetVINT(int i, double v) { fPyint1->VINT[i-1] = v; }

   // ****** /PYINT2/ and /PYINT3/

   Pyint2_t*   GetPyint2() { return fPyint2; }
   Pyint3_t*   GetPyint3() { return fPyint3; }

   // ****** /PYINT4/

   Pyint4_t*   GetPyint4() { return fPyint4; }
   int         GetMWID      (int i) { return fPyint4->MWID[i-1]; }
   double      GetWIDS(int i,int j) { return fPyint4->WIDS[j-1][i-1]; }

   void        SetMWID(int i, int m)           { fPyint4->MWID[i-1]      = m; }
   void        SetWIDS(int i, int j, double w) { fPyint4->WIDS[j-1][i-1] = w; }

   // ****** / PYINT5/

   Pyint5_t*   GetPyint5() { return fPyint5; }
   int         GetNGENPD() { return fPyint5->NGENPD; }
   void        SetNGENPD(int n) { fPyint5->NGENPD = n; }

   // ****** /PYINT6/

   Pyint6_t*   GetPyint6   () { return fPyint6; }
   char*       GetPROC(int i) { fPyint6->PROC[i][27]=0; return fPyint6->PROC[i]; }

   Pyint7_t*   GetPyint7() { return fPyint7; }
   Pyint8_t*   GetPyint8() { return fPyint8; }
   Pyint9_t*   GetPyint9() { return fPyint9; }

   // ****** /PYMSSM/ - indexing in FORTRAN starts
   // from 0!

   Pymssm_t*   GetPymssm()    { return fPymssm; }
   int         GetIMSS(int i) { return fPymssm->IMSS[i]; }
   double      GetRMSS(int i) { return fPymssm->RMSS[i]; }

   void        SetIMSS(int i, int    m) { fPymssm->IMSS[i] = m; }
   void        SetRMSS(int i, double r) { fPymssm->RMSS[i] = r; }

   // ****** /PYSSMT/

   Pyssmt_t*   GetPyssmt()           { return fPyssmt; }
   double      GetZMIX(int i, int j) { return fPyssmt->ZMIX[j-1][i-1]; }
   double      GetUMIX(int i, int j) { return fPyssmt->UMIX[j-1][i-1]; }
   double      GetVMIX(int i, int j) { return fPyssmt->VMIX[j-1][i-1]; }
   double      GetSMZ        (int i) { return fPyssmt->SMZ[i-1]; }
   double      GetSMW        (int i) { return fPyssmt->SMW[i-1]; }

   void        SetZMIX(int i, int j, double z) { fPyssmt->ZMIX[j-1][i-1] = z; }
   void        SetUMIX(int i, int j, double u) { fPyssmt->UMIX[j-1][i-1] = u; }
   void        SetSMZ (int i, double s)        { fPyssmt->SMZ[i-1]       = s; }
   void        SetSMW (int i, double s)        { fPyssmt->SMW[i-1]       = s; }

   Pyints_t*   GetPyints() { return fPyints; }
   Pybins_t*   GetPybins() { return fPybins; }

   // ****** TPYTHIA routines

   void             GenerateEvent();

   void             Initialize(const char *frame, const char *beam, const char *target, float win);

   Int_t            ImportParticles(TClonesArray *particles, Option_t *option="");
   TObjArray       *ImportParticles(Option_t *option="");

   void             OpenFortranFile(int lun, char* name);
   void             CloseFortranFile(int lun);
   int              Pychge(int kf);
   int              Pycomp(int kf);
   void             Pydiff();
   void             Pyedit(int medit);
   void             Pyevnt();
   void             Py1ent(Int_t line, Int_t kf, Double_t pe, Double_t theta, Double_t phi);
   void             Pyexec();
   void             Pyhepc(int mconv);
   void             Pygive(const char *param);
   void             Pyinit(char* frame, char* beam, char* target, double wint);
   void             Pylist(int flag);
   double           Pymass(int kf);
   void             Pyname(int kf, char* name);
   double           Pyr(int idummy);
   void             Pyrget(int lun, int move);
   void             Pyrset(int lun, int move);
   void             Pystat(int flag);
   void             Pytest(int flag);
   //void             Pytune(int itune);   // not (anymore) in libPythia6
   void             Pyupda(int mupda, int lun);
   void             SetupTest();

   ClassDef(TPythia6,0)  //Interface to Pythia6.1 Event Generator
};

#endif
