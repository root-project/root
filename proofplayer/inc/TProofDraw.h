// @(#)root/proof:$Name:  $:$Id: TProofDraw.h,v 1.3 2004/07/09 01:34:51 rdm Exp $
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
class TStatus;
class TH1;


class TProofDraw : public TSelector {

private:
   TStatus             *fStatus;
   TString              fSelection;
   TString              fVarX;
   TTreeFormulaManager *fManager;
   TTreeFormula        *fSelFormula;
   TTreeFormula        *fVarXFormula;
   TH1                 *fHistogram;
   TTree               *fTree;

   void     ClearFormulas();
   void     SetError(const char *sub, const char *mesg);

public:
   TProofDraw();
   virtual            ~TProofDraw();
   virtual int         Version() const { return 1; }
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *);
   virtual void        SlaveBegin(TTree *);
   virtual Bool_t      Notify();
   virtual Bool_t      Process(Long64_t /*entry*/);
   virtual void        SlaveTerminate();
   virtual void        Terminate();

   ClassDef(TProofDraw,0)  //Tree drawing selector for PROOF
};

#endif
