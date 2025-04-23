{
   // Avoid loading the library
   gInterpreter->UnloadLibraryMap("selabort_C");
   TChain ch("T");
   ch.Add("Event1.root/T1");
   ch.Add("Event2.root/T2");
   ch.Add("Event3.root/T3");
   ch.Process("selabort.C","thefirstoption");

#ifdef ClingWorkAroundMissingUnloading
   TSelector *sel = static_cast<TSelector*>(TClass::GetClass("selabort")->New());
#else
   TSelector *sel = TSelector::GetSelector("selabort.C");
#endif
   ch.Process(sel,"theoptions");

#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
