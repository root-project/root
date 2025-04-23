// autoloading fails! so we need to include <vector>:
#if !defined(__CINT__) || 1
#include <vector> // loads the cintdlls
#include <algorithm>
#include "TROOT.h"
#endif

void dump(std::vector<unsigned int>& v) {
  // dump content, using vui's size():
   for (unsigned int i = 0; i < v.size(); ++i)
      printf("%u ", v[i]);
   printf("\n");

}

int VectorSort() {
   vector<unsigned int> vui;
   vui.resize(8); // do we have any functions?

   // are the elements default initialized?
   printf("elem0: %d\n", vui[0]);
   // "-1" is not a uint - is it properly converted?
   vui[0] = -1;

   // set some elements for sort:
   vui[4] = 4;
   vui[2] = 2;
   // also try op "++":
   ++vui[1];

   dump(vui);

   // sort using algo and iterators
   std::sort(vui.begin(), vui.end());

   dump(vui);

   // check the content of the dictionary
   // a bit too noisy, so we skip it:
   //gROOT->ProcessLine(".class vector<unsigned int>");

   // check the availability and validity of the iterator type
   // (this should be "const_iterator" btw - it's a CINT bug)
   gROOT->ProcessLine(Form("((const vector<unsigned int>*)0x%lx)->end()", (long)&vui));

   // is the iterator dict valid? (not really - it ignores the const_)
   // a bit too noisy, so we skip it:
   //gROOT->ProcessLine(".class vector<unsigned int>::const_iterator");

   // ensure that the iterator is accessible:
   vector<unsigned int,allocator<unsigned int> >::iterator beg = vui.begin();

   // ensure that we can iterate, and that the precedence is correct,
   // and that the op++ doesn't operator on the elements:
   ++beg++;
   static_cast<void>(*(++beg)++);
   ++(*(++beg));
   beg = ++beg++;
   beg = ++beg++;

   dump(vui);

   printf("%u\n", *beg);

#ifdef ClingWorkAroundErracticValuePrinter
   printf("(int)0\n");
#endif
  return 0;
}
