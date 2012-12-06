{
#ifdef ClingWorkAroundMissingImplicitAuto
   TClonesArray *tc, *tc2;
#endif
tc = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc->ConstructedAt(i);
tc2 = new TClonesArray("TNamed",5); for(int i=0; i<20; ++i) tc2->ConstructedAt(i);
tc->AbsorbObjects(tc2);
delete tc2;
delete tc;
}
