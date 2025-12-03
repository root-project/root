#include "TDataMember.h"
#include "TClass.h"
#include "TGWidget.h"
#include "TGTextEntry.h"
#include "TSystem.h"

// ETextJustification fAlignment;        // *OPTION={GetMethod="GetAlignment";SetMethod="SetAlignment";Items=(kTextLeft="Left",kTextCenterX="Center",kTextRight="Right")}*

int check(TDataMember *d, UInt_t i, const char *name, Long_t expected) {

   auto l = (TOptionListItem*)d->GetOptions()->At(i);

   if (l->fValue != expected) {
      fprintf(stdout,"Error: for option %d (%s) got %ld instead of %ld\n",
              i, name, l->fValue, expected);
      return 1;
   } else {
      return 0;
   }
}

int execOptionList() {
   gSystem->Setenv("DISPLAY",""); // Avoid spurrious warning when libGui is loaded.

   auto c = TClass::GetClass("TGTextEntry");
   auto d = (TDataMember*)c->GetListOfDataMembers()->FindObject("fAlignment");

   int result = check(d, 0, "kTextLeft", (Long_t) kTextLeft);
   result += check(d, 1, "kTextCenterX", (Long_t) kTextCenterX);
   result += check(d, 2, "kTextRight", (Long_t) kTextRight);

   return result;

}
