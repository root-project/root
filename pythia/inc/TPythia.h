// @(#)root/pythia:$Name$:$Id$
// Author: Piotr Golonka   10/09/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPythia
#define ROOT_TPythia


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPythia                                                              //
//                                                                      //
// This class implements an interface to the Pythia event generator.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGenerator
#include "TGenerator.h"
#endif

class TPrimary;

class TPythia : public TGenerator {

private:
   void                Unsupported(const char *name)
                              {printf("TPythia::%s unsupported !\n",name);}

   virtual TPrimary   *GetPrimary(Int_t) {Unsupported("GetPrimary");return 0;};
   virtual void        ShowNeutrons(Bool_t)    {Unsupported("ShowNeutrons");};

public:
                       TPythia();
   virtual            ~TPythia();

   virtual void        Initialize(const char *frame, const char *beam, const char *target, float win);

   virtual Int_t       ImportParticles(TClonesArray *particles, Option_t *option="");
   virtual TObjArray  *ImportParticles(Option_t *option="");


   virtual void        SetupTest();

   // common PYSUBS access routines: (you don't need to care about indexes of
   // arrays - just provide values from PYTHIA documentation!


   //Generic functions:
   virtual void        SetMSEL(Int_t sel);
   virtual Int_t       GetMSEL() const;

   virtual void        SetMSUB(Int_t isub, Bool_t msub=1);
   virtual Bool_t      GetMSUB(Int_t isub) const;

   virtual void        SetKFIN(Int_t i, Int_t j, Bool_t kfin=1);
   virtual Bool_t      GetKFIN(Int_t i, Int_t j) const;

   virtual void        SetCKIN(Int_t key, Float_t value);
   virtual Float_t     GetCKIN(Int_t key) const;

   // common PYPARS access routines:
   virtual void        SetMSTP(Int_t key, Int_t value);
   virtual Int_t       GetMSTP(Int_t key) const;

   virtual void        SetPARP(Int_t key, Float_t value);
   virtual Float_t     GetPARP(Int_t key) const;

   virtual void        SetMSTI(Int_t key, Int_t value);
   virtual Int_t       GetMSTI(Int_t key) const;

   virtual void        SetPARI(Int_t key, Float_t value);
   virtual Float_t     GetPARI(Int_t key) const;

   //common PYINT1 access routines:
   virtual void        SetMINT(Int_t key, Int_t value);
   virtual Int_t       GetMINT(Int_t key) const;

   virtual void        SetVINT(Int_t key, Float_t value);
   virtual Float_t     GetVINT(Int_t key) const;

   //common PYINT2 access routines:
   virtual void        SetISET(Int_t isub,Int_t iset);
   virtual Int_t       GetISET(Int_t isub) const;

   virtual void        SetKFPR(Int_t isub, Int_t j, Int_t kfpr);
   virtual Int_t       GetKFPR(Int_t isub, Int_t j) const;

   virtual void        SetCOEF(Int_t isub, Int_t j, Float_t coef);
   virtual Float_t     GetCOEF(Int_t isub, Int_t j) const;

   virtual void        SetICOL(Int_t kf,Int_t i, Int_t j,Int_t value);
   virtual Int_t       GetICOL(Int_t kf,Int_t i, Int_t j) const;

   // common PYINT3 access routines - this common is "read only" !!!
   virtual Float_t     GetXSFX(Int_t side, Int_t kf) const;
   virtual Int_t       GetISIG(Int_t ichn, Int_t isig) const;
   virtual Float_t     GetSIGH(Int_t ichn) const;

   // common PYINT4 access routines - this common is "read only" !!!
   virtual Float_t     GetWIDP(Int_t kf, Int_t j) const;
   virtual Float_t     GetWIDE(Int_t kf, Int_t j) const;
   virtual Float_t     GetWIDS(Int_t kf, Int_t j) const;

   // common PYINT5 access routines - this common is "read only" !!!
   virtual Int_t       GetNGEN(Int_t isub, Int_t key) const;
   virtual Float_t     GetXSEC(Int_t isub, Int_t key) const;

   //common LUDAT1 access:
   virtual Int_t       GetMSTU(Int_t key) const;
   virtual void        SetMSTU(Int_t key, Int_t value);

   virtual Float_t     GetPARU(Int_t key) const;
   virtual void        SetPARU(Int_t key, Float_t value);

   virtual Int_t       GetMSTJ(Int_t key) const;
   virtual void        SetMSTJ(Int_t key, Int_t value);


   virtual Float_t     GetPARJ(Int_t key) const;
   virtual void        SetPARJ(Int_t key, Float_t value);

   //common LUDAT2 access:
   virtual Int_t       GetKCHG(Int_t kc, Int_t key) const;
   virtual void        SetKCHG(Int_t kc, Int_t key, Int_t value);

   virtual Float_t     GetPMAS(Int_t kc, Int_t key) const;
   virtual void        SetPMAS(Int_t kc, Int_t key, Float_t value);

   virtual Float_t     GetPARF(Int_t key) const;
   virtual void        SetPARF(Int_t key, Float_t value);

   virtual Float_t     GetVCKM(Int_t i, Int_t j) const;
   virtual void        SetVCKM(Int_t i, Int_t j, Float_t value);

   //common LUDAT3 access:
   virtual Int_t       GetMDCY(Int_t kc, Int_t key) const;
   virtual void        SetMDCY(Int_t kc, Int_t key, Int_t value);

   virtual Int_t       GetMDME(Int_t idc, Int_t key) const;
   virtual void        SetMDME(Int_t idc, Int_t key, Int_t value);

   virtual Float_t     GetBRAT(Int_t idc) const;
   virtual void        SetBRAT(Int_t idc, Float_t value);

   virtual Int_t       GetKFDP(Int_t idc, Int_t j) const;
   virtual void        SetKFDP(Int_t idc, Int_t j, Int_t value);

   //common LUDAT4 access:
   virtual char       *GetCHAF(Int_t kc) const;
   virtual void        SetCHAF(Int_t kc,char *value);

   //common LUDATR access:
   virtual void        SetMRLU(Int_t key, Int_t seed);
   virtual Int_t       GetMRLU(Int_t key) const;

   virtual void        SetRRLU(Int_t key, Float_t value);
   virtual Float_t     GetRRLU(Int_t key) const;

   // pythia and lujets subroutines access:
   virtual void        Pystat(Int_t mstat=1);
   virtual void        Pytest(Int_t key=1);
   virtual void        Pyevnt();
   virtual void        Lulist(Int_t mlist=1);
   virtual void        Luexec();
   virtual Int_t       Lucomp(Int_t kf);

   virtual void        Pyinit(char *frame, char *beam, char *target, float win);

   virtual void        GenerateEvent();

   virtual void        Draw(Option_t *option="");

   ClassDef(TPythia,1)  //Interface to Pythia Event Generator
};

#endif
