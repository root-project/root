// Author: Wim Lavrijsen   February 2006

#ifndef ROOT_TPyROOTApplication
#define ROOT_TPyROOTApplication

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPyROOTApplication                                                       //
//                                                                          //
// Setup interactive application for python.                                //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// ROOT
#ifndef ROOT_TApplication
#include "TApplication.h"
#endif


namespace PyROOT {

class TPyROOTApplication : public TApplication {
public:
   static Bool_t CreatePyROOTApplication( Bool_t bLoadLibs = kTRUE );

   static Bool_t InitROOTGlobals();
   static Bool_t InitCINTMessageCallback();
   static Bool_t InitROOTMessageCallback();

public:
   TPyROOTApplication(
      const char* acn, Int_t* argc, char** argv, Bool_t bLoadLibs = kTRUE );

   virtual ~TPyROOTApplication() { }
   ClassDef(TPyROOTApplication,0)   //Setup interactive application
};

} // namespace PyROOT

#endif
