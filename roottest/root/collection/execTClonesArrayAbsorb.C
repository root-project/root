{
auto tc = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc->ConstructedAt(i);
auto tc2 = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc2->ConstructedAt(i);
tc->AbsorbObjects(tc2);
if (tc2->GetEntries() != 0) {
  Error("execTClonesArrayAbsorb","Original TClonesArray still contains data: %d\n",tc2->GetEntries());
  return 1;
}
if (tc->GetEntries() != 40) {
  Error("execTClonesArrayAbsorb","Unexpected number of entries in target TClonesArray: %d (vs 40)\n",tc->GetEntries());
  return 2;
}
// Testing if the destructor crashes.
delete tc2;
delete tc;

TObject::SetObjectStat(1);
int before = gObjectTable->Instances();

tc = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc->ConstructedAt(i);
tc2 = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc2->ConstructedAt(i);
tc->Clear();
   //tc->AbsorbObjects(tc2, 0, tc2->GetEntriesFast()-1);
tc->AbsorbObjects(tc2);
if (tc2->GetEntries() != 0) {
  Error("execTClonesArrayAbsorb","Original TClonesArray still contains data: %d\n",tc2->GetEntries());
  return 1;
}
if (tc->GetEntries() != 20) {
  Error("execTClonesArrayAbsorb","Unexpected number of entries in target TClonesArray: %d (vs 20)\n",tc->GetEntries());
  return 2;
}
// Testing if the destructor crashes.
delete tc2;
delete tc;
int after = gObjectTable->Instances();

if (before != after) {
   Error("execTClonesArrayAbsorb","Unexpected number of TNamed left after AbsorbObjects(tc2): %d extra (vs 0)\n",after-before);
   return 3;
}

before = gObjectTable->Instances();

tc = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc->ConstructedAt(i);
tc2 = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc2->ConstructedAt(i);
tc->Clear();
tc->AbsorbObjects(tc2, 0, tc2->GetEntriesFast()-1);
if (tc2->GetEntries() != 0) {
   Error("execTClonesArrayAbsorb","Original TClonesArray still contains data: %d\n",tc2->GetEntries());
   return 1;
}
if (tc->GetEntries() != 20) {
   Error("execTClonesArrayAbsorb","Unexpected number of entries in target TClonesArray: %d (vs 20)\n",tc->GetEntries());
   return 2;
}
// Testing if the destructor crashes.
delete tc2;
delete tc;
after = gObjectTable->Instances();

if (before != after) {
   Error("execTClonesArrayAbsorb","Unexpected number of TNamed left after AbsorbObjects(tc2,0,n): %d extra (vs 0)\n",after-before);
   return 3;
}

return 0;
}
