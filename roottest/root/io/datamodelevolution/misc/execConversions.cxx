#include "Rtypes.h"

struct A {
   int fValue;
};

struct B {
   float fValue;
};

struct C {
   double fValue;
   ClassDef(C, 3);
};

#ifdef __ROOTCLING__
#pragma read sourceClass="B" targetClass="A";
#pragma read sourceClass="B" targetClass="C";
#pragma read sourceClass="A" targetClass="B";
#pragma read sourceClass="A" targetClass="C";
#pragma read sourceClass="C" targetClass="A";
#pragma read sourceClass="C" targetClass="B";
#endif

#include "TClass.h"
#include "TVirtualStreamerInfo.h"
#include "TError.h"

int testpair(TClass *acl, TClass *bcl)
{
      auto i1 = acl->FindConversionStreamerInfo(bcl->GetName(), bcl->GetCheckSum());
   if (!i1)
      Fatal("test FindConversionStreamerInfo", "(1) Could not find conversion StreamerInfo for %s in %s", bcl->GetName(), acl->GetName());
   
   auto i2 = acl->FindConversionStreamerInfo(bcl->GetName(), bcl->GetCheckSum());
   if (!i2)
      Fatal("test FindConversionStreamerInfo", "(2) Could not find conversion StreamerInfo for %s in %s", bcl->GetName(), acl->GetName());

   if (i1 != i2)
      Fatal("test FindConversionStreamerInfo", "(1) The conversion StreamerInfo from %s to %s was recreated", bcl->GetName(), acl->GetName());

   Int_t version = bcl->IsForeign() ? -1 : bcl->GetClassVersion();

   auto i3 = acl->GetConversionStreamerInfo(bcl->GetName(), version);
   if (!i3)
      Fatal("test GetConversionStreamerInfo", "(3) Could not find conversion StreamerInfo for %s in %s", bcl->GetName(), acl->GetName());
   
   auto i4 = acl->GetConversionStreamerInfo(bcl->GetName(), version);
   if (!i4)
      Fatal("test GetConversionStreamerInfo", "(4) Could not find conversion StreamerInfo for %s in %s", bcl->GetName(), acl->GetName());

   if (i3 != i4)
      Error("test GetConversionStreamerInfo", "(2) The conversion StreamerInfo from %s to %s was recreated", bcl->GetName(), acl->GetName());

   if (i1 != i3 || i1 != i4)
      Error("test GetConversionStreamerInfo", "(3) The conversion StreamerInfo from %s to %s was recreated", bcl->GetName(), acl->GetName());

   return 0;
}

int execConversions() {
   TClass *acl = TClass::GetClass("A");
   if (!acl)
      Fatal("conversions", "Could not find TClass for A");

   TClass *bcl = TClass::GetClass("B");
   if (!bcl)
      Fatal("conversions", "Could not find TClass for B");

   TClass *ccl = TClass::GetClass("C");
   if (!ccl)
      Fatal("conversions", "Could not find TClass for B");

   testpair(acl, bcl);
   testpair(acl, ccl);
   testpair(bcl, acl);
   testpair(bcl, ccl);
   testpair(ccl, acl);
   testpair(ccl, bcl);

   return 0;
}