// @(#)root/proof:$Name:  $:$Id: TProofDraw.h,v 1.1.2.1 2003/11/05 21:58:19 cvsuser Exp $
// Author: Maarten Ballintijn   24/09/2003

#ifndef ROOT_TProofDraw
#define ROOT_TProofDraw


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDraw                                                           //
//                                                                      //
// Implement Tree drawing using PROOF.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TSelector
#include "TSelector.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TTree;
class TTreeFormulaManager;
class TTreeFormula;
class TH1;


class TProofDraw : public TSelector {
private:
   TString        fSelection;
   TString        fVarX;


   TTreeFormulaManager *fManager;
   TTreeFormula        *fSelFormula;
   TTreeFormula        *fVarXFormula;

   TH1           *fHistogram;

   TTree         *fTree;

   void     ClearFormulas();

public:
   TProofDraw();
   virtual            ~TProofDraw();
   virtual int         Version() const { return 1; }
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *);
   virtual void        SlaveBegin(TTree *);
   virtual Bool_t      Notify();
   virtual Bool_t      Process(Int_t /*entry*/);
   virtual void        SlaveTerminate();
   virtual void        Terminate();

   ClassDef(TProofDraw,0)  //Tree drawing selector for PROOF
};

#endif
