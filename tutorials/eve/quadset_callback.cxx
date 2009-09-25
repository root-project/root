#include <TEveDigitSet.h>
#include <cstdio>

void quadset_callback(TEveDigitSet* ds, Int_t idx, TObject* obj)
{
   printf("dump_digit_set_hit - 0x%lx, id=%d, obj=0x%lx\n",
          (ULong_t) ds, idx, (ULong_t) obj);
}

void quadset_set_callback(TEveDigitSet* ds)
{
   ds->SetCallbackFoo(quadset_callback);
}
