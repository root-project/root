/*
  Test parsing and calling of conversion operators.
  Calling of conversion operators often fails;
  #define FAIL to see some of them.

  Parsing does fail for some of D's operators which cannot
  be specialized in a Linkdef file; see ConversionOp.h.
*/

#include "TSystem.h"
#include "TClass.h"
#include "ConversionOp.h"

#ifndef CINTFAILURE
# define REPORTCINTFAILURE(TXT) \
  printf("KNOWN FAILURE: %s\n", TXT); \
  if (false)
#else
# define REPORTCINTFAILURE(TXT)
#endif

void ConversionOp() {
   A<B> ab;
   C c;

   // A<T>
   B b;
   B* pb;
   (void)pb; // avoid unused var

   A<B>* pab;
   //printf("KNOWN FAILURE: b = ab\n");
   REPORTCINTFAILURE("b = ab") b = ab;
   REPORTCINTFAILURE("pb = ab") pb = ab;
   pab = ab;
   if (((const A<B>&)ab) < (const B&)b) printf("ab < b\n");
   else printf("ab >= b\n");
   if (ab < *pab) printf("ab < *pab\n");
   else printf("ab >= *pab\n");


   // C
   REPORTCINTFAILURE("ab = c") ab = c;
   A<C> ac;
   ac = c;
   REPORTCINTFAILURE("b = c") b = c;
   REPORTCINTFAILURE("N::B nb = c") N::B nb = c;
   A<C>* pac;
   (void)pac; // avoid unused var
   REPORTCINTFAILURE("pac = c") pac = c;
   const A<B>* cpab;
   (void)cpab; // avoid unused var
   REPORTCINTFAILURE("cpab = c") cpab = c;
   REPORTCINTFAILURE("pb = c") pb = c;
   const A<C> & rac(c);
   (void)rac; // avoid unused var
   REPORTCINTFAILURE("const A<B> & rab(c);") {
      const A<B> & rab(c);
      (void)rab; // avoid unused var
   }
   REPORTCINTFAILURE("const B & rb(c)") {
      const B & rb(c);
      (void)rb; // avoid unused var
   }

   // D
   D d;
   REPORTCINTFAILURE("ab = d + (ab)") ab = d + (ab);
   
   pab = d - (ab);
   /*
     A<int> aint;
     A<int>* paint = d*(aint);
     A<float> afloat;
     A<float>& rafloat(d/afloat);
    */

}

