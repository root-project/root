int runstring() 
{
   int result = 0;
   // works but produce a warning
   //TFormula* pf1 = new TFormula("f1","\"abc\" == \"def\"");
   ROOT::v5::TFormula* pf1 = new ROOT::v5::TFormula("f1","\"abc\" == \"def\"");
   int i = pf1->Eval(0,0,0,0);
   if (i!=0) {
      cerr << "== is not working for strings\n";
      result = 1;
   }
   ROOT::v5::TFormula* pf2 = new ROOT::v5::TFormula("f2","strstr(\"abc\",\"def\") != 0");
   i = pf2->Eval(0,0,0,0);
   if (i!=0) {
      cerr << "strstr is not working for strings\n";
      result = 1;
   }
   // same as above
   //TFormula* pf3 = new TFormula("f3","\"abc\" != \"def\"");
   ROOT::v5::TFormula* pf3 = new ROOT::v5::TFormula("f3","\"abc\" != \"def\"");
   i = pf3->Eval(0,0,0,0);
   if (i!=1) {
      cerr << "!= is not working for strings\n";
      result = 1;
   }
#ifdef ClingWorkAroundErracticValuePrinter
  printf("(int)0\n");
#endif
   return result;
}
