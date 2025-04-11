{
#ifdef ClingWorkAroundIncorrectTearDownOrder
   if (1) {
#endif
   TTree t;
   double x = 1;
   t.Branch("x", &x, "x/D");
   t.Fill();
   t.Draw("x", TString::Format(""));
   t.Draw("x", TString::Format(""));
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
#ifdef ClingWorkAroundIncorrectTearDownOrder
   }
#endif
}
