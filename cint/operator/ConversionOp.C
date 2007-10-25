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

void ConversionOp() {
   A<B> ab;
   C c;

   // A<T>
   B b;
   B* pb;
   if (!&pb) printf("FOO"); // avoid unused var

   A<B>* pab;
   printf("KNOWN FAILURE: b = ab\n");
#ifdef FAIL
   b = ab;
#endif
   printf("KNOWN FAILURE: pb = ab\n");
#ifdef FAIL
   pb = ab;
#endif
   printf("KNOWN FAILURE: pab = ab\n");
#ifdef FAIL
   pab = ab;
#endif
   if (((const A<B>&)ab) < (const B&)b) printf("ab < b\n");
   else printf("ab >= b\n");
   if (ab < *pab) printf("ab < *pab\n");
   else printf("ab >= *pab\n");


   // C
   printf("KNOWN FAILURE: ab = c\n");
#ifdef FAIL
   ab = c;
#endif
   A<C> ac;
   printf("KNOWN FAILURE: ac = c: should call C::op A<C>()\n");
#ifdef FAIL
   ac = c;
#endif
   printf("KNOWN FAILURE: b = c\n");
#ifdef FAIL
   b = c;
#endif
   printf("KNOWN FAILURE: N::B nb = c\n");
#ifdef FAIL
   N::B nb = c;
#endif
   A<C>* pac;
   if (!&pac) printf("FOO"); // avoid unused var
   printf("KNOWN FAILURE: pac = c\n");
#ifdef FAIL
   pac = c;
#endif
   const A<B>* cpab;
   if (!&cpab) printf("FOO"); // avoid unused var
   printf("KNOWN FAILURE: cpab = c\n");
#ifdef FAIL
   cpab = c;
#endif
   printf("KNOWN FAILURE: pb = c\n");
#ifdef FAIL
   pb = c;
#endif
   const A<C> & rac(c);
   if (!&rac) printf("FOO"); // avoid unused var
   printf("KNOWN FAILURE: const A<B> & rab(c);\n");
#ifdef FAIL
   const A<B> & rab(c);
   if (!&rab) printf("FOO"); // avoid unused var
#endif
   printf("KNOWN FAILURE: const B & rb(c)\n");
#ifdef FAIL
   const B & rb(c);
   if (!&rb) printf("FOO"); // avoid unused var
#endif


   // D
   D d;
   printf("KNOWN FAILURE: ab = d + (ab)\n");
#ifdef FAIL
   ab = d + (ab);
#endif
   
   pab = d - (ab);
   /*
     A<int> aint;
     A<int>* paint = d*(aint);
     A<float> afloat;
     A<float>& rafloat(d/afloat);
    */

}

