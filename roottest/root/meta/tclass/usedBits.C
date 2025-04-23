#include "TObject.h"
#include "TClass.h"

class top : public TObject {
public:
   enum { k15 = BIT(15), k16 = BIT(16) };
   static UInt_t LoadUsedBits(TClass *cl) { cl->RegisterUsedBits(k15|k16); return 2; }
   ClassDef(top,1);
};

class bottomOne : public top {
public:
   enum { k17 = BIT(17) };
   static UInt_t LoadUsedBits(TClass *cl) { cl->RegisterUsedBits(k17); return 1; }
   ClassDef(bottomOne,1);
};

class bottomTwo : public top {
public:
   enum { k17 = BIT(16) };
   static UInt_t LoadUsedBits(TClass *cl) { cl->RegisterUsedBits(k17); return 1; }
   ClassDef(bottomTwo,1);
};
   
void usedBits(int level = 2) {

   switch (level) {
   case 0 :
      top::LoadUsedBits( top::Class() );
      bottomOne::LoadUsedBits( bottomOne::Class() );
      bottomTwo::LoadUsedBits( bottomTwo::Class() );
      break;
   case 1:
      top::Class()->LoadUsedBits();
      bottomOne::Class()->LoadUsedBits();
      bottomTwo::Class()->LoadUsedBits();
      break;
   case 2:
      break;
   }

   if (level<2) {
      top::Class()->CheckUsedBits();
      bottomOne::Class()->CheckUsedBits();
      bottomTwo::Class()->CheckUsedBits(); // Should issue an error!
   } else {
      top::Class();
      bottomOne::Class();
      bottomTwo::Class();
   }
}
