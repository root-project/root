int runstring() 
{
   int result = 0;
   TFormula* pf1 = new TFormula("f1","\"abc\" == \"def\"");
   int i = pf1->Eval(0,0,0,0);
   if (i!=0) {
      cerr << "== is not working for strings\n";
      result = 1;
   }
   TFormula* pf2 = new TFormula("f2","strstr(\"abc\",\"def\")");
   i = pf2->Eval(0,0,0,0);
   if (i!=0) {
      cerr << "strstr is not working for strings\n";
      result = 1;
   }
   TFormula* pf3 = new TFormula("f3","\"abc\" != \"def\"");
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
