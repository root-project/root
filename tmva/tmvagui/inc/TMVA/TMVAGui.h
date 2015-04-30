#ifndef TMVAGui__HH
#define TMVAGui__HH
#include <iostream>
#include <vector>

#include "TList.h"
#include "TROOT.h"
#include "TKey.h"
#include "TString.h"
#include "TControlBar.h"
#include "TObjString.h"
#include "TClass.h"

#include "tmvaglob.h"
namespace TMVA{


   TList* GetKeyList( const TString& pattern );

   // utility function
   void ActionButton( TControlBar* cbar, 
                      const TString& title, const TString& macro, const TString& comment, 
                      const TString& buttonType, TString requiredKey = ""); 

   // main GUI
   void TMVAGui( const char* fName = "TMVA.root" );

   struct  TMVAGUI {
      TMVAGUI(TString name = "TMVA.root" ) {
         TMVA::TMVAGui(name.Data());
      }
   };
   
}


#endif
