void execSimpleVector() {
   auto c=TClass::GetClass("vector<C>");
   c->GetCheckSum(TClass::kLatestCheckSum);
   // If autoparse succeds, the number of headers parsed is returned. If this is
   // different from 0, it means that the parsing happens only now and did not happen before.
   if( 0 != gInterpreter->AutoParse("vector<C>")) {
      std::cout << "Test successful: autoparse did not happen during checksumming.\n";
   }
}
