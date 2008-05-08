#include <TEveDigitSet.h>
#include <cstdio>

void quad_test_callback(TEveDigitSet* ds, Int_t idx, TObject* obj)
{
   printf("dump_digit_set_hit - 0x%lx, id=%d, obj=0x%lx\n",
          (ULong_t) ds, idx, (ULong_t) obj);
}

void quad_test_set_callback(TEveDigitSet* ds)
{
   ds->SetCallbackFoo(quad_test_callback);
}
