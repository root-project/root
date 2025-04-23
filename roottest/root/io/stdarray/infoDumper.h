#ifndef ROOTTEST_INFODUMPER
#define ROOTTEST_INFODUMPER

#include "TClass.h"
#include "TList.h"
#include "TObjArray.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TDataMember.h"
#include <iostream>

/*
From TClass.h

   // Describe the current state of the TClass itself.
   enum EState {
      kNoInfo,          // The state has not yet been initialized, i.e. the TClass
                        // was just created and/or there is no trace of it in the interpreter.
      kForwardDeclared, // The interpreted knows the entity is a class but that's it.
      kEmulated,        // The information about the class only comes from a TStreamerInfo
      kInterpreted,     // The class is described completely/only in the interpreter database.
      kHasTClassInit,   // The class has a TClass proper bootstrap coming from a run
                        // through rootcling/genreflex/TMetaUtils and the library
                        // containing this dictionary has been loaded in memory.
      kLoaded = kHasTClassInit,
      kNamespaceForMeta // Very transient state necessary to bootstrap namespace entries in ROOT Meta w/o interpreter information
   };

 */

const char* EStateNames[] = {
      "kNoInfo",
      "kForwardDeclared",
      "kEmulated",
      "kInterpreted",
      "kHasTClassInit",
      "kLoaded",
      "kNamespaceForMeta"
};


void dumpInfo(TClass* c) {

   if (!c) return;
   auto className = c->GetName();

   cout << "Class " << className << " state is " << EStateNames[c->GetState()] << endl;

   auto dms = c ? c->GetListOfDataMembers() : nullptr;
   if (dms) {
      cout << "List of data members of " << className << endl;
      for (auto dmo : *dms) {
         auto dm = (TDataMember*) dmo;
         cout << "Name: " << dm->GetName() << "\n"
            << "Type Name: " << dm->GetTypeName() << "\n"
            << "Array Dim: " << dm->GetArrayDim() << "\n";
         for (int i=0;i< dm->GetArrayDim();++i) {
            cout << "Max Index [" << i << "]: " << dm->GetMaxIndex(i) << "\n";
         }
         cout << "Array Index: " << dm->GetArrayIndex() << "\n"
            << "Is Pointer: " << dm->IsaPointer() << "\n"
            << "Is Basic: " << dm->IsBasic() << "\n"
            << "---------" << endl;
      }
   }

   auto si = c ? c->GetStreamerInfo() : nullptr;
   if (si){
      cout << "List of streamer elements of " << className << endl;
      for (auto elo : *(si->GetElements())) {
         auto el = (TStreamerElement*)elo;
         cout << "Name: " << el->GetName() << "\n"
            << "Type Name: " << el->GetTypeName() << "\n"
            << "Array Dim: " << el->GetArrayDim() << "\n";
         for (int i=0;i< el->GetArrayDim();++i) {
            cout << "Max Index [" << i << "]: " << el->GetMaxIndex(i) << "\n";
         }
         cout << "Is Pointer: " << el->IsaPointer() << "\n"
            << "Is Basic: " << el->GetType() << "\n"
            << "---------" << endl;
      }
   }
}

void dumpInfo(const char* className) {
   auto c = TClass::GetClass(className);
   dumpInfo(c);
}
#endif
