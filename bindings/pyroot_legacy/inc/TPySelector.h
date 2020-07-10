// Author: Wim Lavrijsen   March 2008

#ifndef ROOT_TPySelector
#define ROOT_TPySelector

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPySelector                                                              //
//                                                                          //
// Python base class equivalent of PROOF TSelector.                         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


//- ROOT
#include "TSelector.h"

#include "TTree.h"

// Python
struct _object;
typedef _object PyObject;


class TPySelector : public TSelector {
public:
   using TSelector::fStatus;
// using TSelector::fAbort;
   using TSelector::fOption;
   using TSelector::fObject;
   using TSelector::fInput;
   using TSelector::fOutput;

public:
   TTree* fChain;

public:
// ctor/dtor ... cctor and assignment are private in base class
   TPySelector( TTree* /* tree */ = 0, PyObject* self = 0 );
   ~TPySelector() override;

   // TSelector set of forwarded (overridden) methods
   Int_t  Version() const override;
   Int_t  GetEntry(Long64_t entry, Int_t getall = 0) override;
   Bool_t Notify() override;

   void   Init(TTree *tree) override;
   void   Begin(TTree *tree = 0 /* not used */) override;
   void   SlaveBegin(TTree *tree) override;
   Bool_t Process(Long64_t entry) override;
   void   SlaveTerminate() override;
   void   Terminate() override;

   void Abort(const char *why, EAbort what = kAbortProcess) override;

   ClassDef( TPySelector, 1 );   //Python equivalent base class for PROOF

private:
// private helpers for forwarding to python
   void SetupPySelf();
   PyObject* CallSelf( const char* method, PyObject* pyobject = 0 );

private:
   PyObject* fPySelf;              //! actual python object
};

#endif
