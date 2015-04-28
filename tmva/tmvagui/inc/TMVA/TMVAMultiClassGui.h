#ifndef TMVAMultiClassGui__HH
#define TMVAMultiClassGui__HH
#include <iostream>
#include <vector>

#include "TList.h"
#include "TROOT.h"
#include "TKey.h"
#include "TString.h"
#include "TControlBar.h"
#include "TObjString.h"

#include "tmvaglob.h"
namespace TMVA{


   TList* MultiClassGetKeyList( const TString& pattern );
   // utility function
   void MultiClassActionButton( TControlBar* cbar, 
                      const TString& title, const TString& macro, const TString& comment, 
                      const TString& buttonType, TString requiredKey = "" ); 

   // main GUI
   void TMVAMultiClassGui( const char* fName = "TMVAMulticlass.root" ); 
}
#endif
