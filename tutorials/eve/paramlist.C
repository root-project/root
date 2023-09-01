/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of simple configuration via TEveParamList class.
///
/// \macro_code
///
/// \author Matevz Tadel

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
      auto pl = dynamic_cast<TEveParamList*> (reinterpret_cast<TQObject*>(gTQSender));

      printf("Change in param-list '%s', parameter '%s'.\n", pl->GetElementName(), parameter);
   }

   ClassDef(TParamFollower, 0);
};

ClassImp(TParamFollower)

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
