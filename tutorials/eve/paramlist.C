// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of simple configuration via TEveParamList class.

// Must run in compiled mode for dynamic_cast to work properly.

#if defined(__CINT__) && !defined(__MAKECINT__)
{
   gSystem->CompileMacro("paramlist.C");
   paramlist();
}
#else

#include "TEveManager.h"
#include "TEveParamList.h"
#include "TQObject.h"

class TParamFollower
{
public:
   TParamFollower()
   {
      TQObject::Connect("TEveParamList", "ParamChanged(char*)",
                        "TParamFollower", this, "OnParamChanged(char*)");
   }
   virtual ~TParamFollower()
   {
      TQObject::Disconnect("TParamFollower", "ParamChanged(char*)",
                           this, "OnParamChanged(char*)");
   }

   void OnParamChanged(const char* parameter)
   {
      TEveParamList* pl = dynamic_cast<TEveParamList*>
         (reinterpret_cast<TQObject*>(gTQSender));

      printf("Change in param-list '%s', parameter '%s'.\n",
             pl->GetElementName(), parameter);
   }

   ClassDef(TParamFollower, 0);
};

void paramlist()
{
   TEveManager::Create();

   TEveParamList* x = 0;

   x = new TEveParamList("Top config");
   gEve->AddToListTree(x, 0);

   x->AddParameter(TEveParamList::FloatConfig_t("Pepe", 20, 0, 110));
   x->AddParameter(TEveParamList::IntConfig_t("Dima", 100, 0, 110));
   x->AddParameter(TEveParamList::BoolConfig_t("Chris", 1));

   x = new TEveParamList("Another config");
   gEve->AddToListTree(x, 0);

   x->AddParameter(TEveParamList::FloatConfig_t("MagneticField", 4, -4, 4));
   x->AddParameter(TEveParamList::FloatConfig_t("Temperature", 16, -20, 40));

   new TParamFollower;
}

#endif
