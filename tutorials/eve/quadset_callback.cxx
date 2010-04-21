#include <TEveDigitSet.h>
#include <cstdio>

void quadset_callback(TEveDigitSet* ds, Int_t idx, TObject* obj)
{
   printf("dump_digit_set_hit - 0x%lx, id=%d, obj=0x%lx\n",
          (ULong_t) ds, idx, (ULong_t) obj);
}

TString quadset_tooltip_callback(TEveDigitSet* ds, Int_t idx)
{
   // This gets called for tooltip if the following is set:
   //   q->SetPickable(1);
   //   q->SetAlwaysSecSelect(1);

   return TString::Format("callback tooltip for '%s' - 0x%lx, id=%d\n",
                          ds->GetElementName(), (ULong_t) ds, idx);

}

void quadset_set_callback(TEveDigitSet* ds)
{
   ds->SetCallbackFoo(quadset_callback);
   ds->SetTooltipCBFoo(TEveDigitSet* ds, Int_t idx)
}
