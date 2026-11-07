#include <ROOT/RNTupleReader.hxx>

#include <cstdio>
#include <vector>

#include "NtplHit_New.hxx"

static int gFails = 0;

static void Check(char const *m, double want, double got)
{
   bool ok = want == got;
   if (!ok)
      ++gFails;
   printf("%-4s want %-4g got %-22.17g %s\n", m, want, got, ok ? "ok" : "WRONG");
}

int main()
{
   auto r = ROOT::RNTupleReader::Open("r", "root_test_ntpl_unversioned.root");
   auto v = r->GetModel().GetDefaultEntry().GetPtr<std::vector<Hit>>("hits");

   r->LoadEntry(0);

   Check("fA_r", 1, v->at(0).fA_r);
   Check("fB_r", 2, v->at(0).fB_r);
   Check("fX_r", 1.5, v->at(0).fX_r);
   Check("fY_r", 2.5, v->at(0).fY_r);

   return gFails;
}
