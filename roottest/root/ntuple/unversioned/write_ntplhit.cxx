#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TClass.h>

#include <cstdio>
#include <vector>

#include "NtplHit_Old.hxx"

int main()
{
   Hit h;
   h.fA = 1;
   h.fB = 2;
   h.fX = 1.5;
   h.fY = 2.5;
   auto model = ROOT::RNTupleModel::Create();
   auto pv = model->MakeField<std::vector<Hit>>("hits");
   auto w = ROOT::RNTupleWriter::Recreate(std::move(model), "r", "root_test_ntpl_unversioned.root");
   *pv = {h};
   w->Fill();
   printf("CHECKSUM %u\n", TClass::GetClass("Hit")->GetCheckSum());
   return 0;
}
