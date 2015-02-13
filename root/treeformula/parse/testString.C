
int execTestString() {
   TFormulaOld *f = new TFormulaOld("f0","\"a+b\"");
   if (f->GetNdim()!=1) return 0;
   Int_t res = f->Eval(0);
   if (res!=0) return 0;   

   f = new TFormulaOld("f1","strstr(\"a+b\",\"a+b\")");
   if (f->GetNdim()!=1) return 0;
   res = f->Eval(0);
   if (res!=1) return 0;

   f = new TFormulaOld("f2","\"af(d+gd\"+2");
   if (f->GetNdim()!=0) return 0;
   res = f->Eval(0);
   if (res!=2) return 0;

   f = new TFormulaOld("f3","strstr(\"a\",\"a+b\")");
   if (f->GetNdim()!=1) return 0;
   res = f->Eval(0);
   if (res!=0) return 0;

   f = new TFormulaOld("f4","(strstr(\"a\",\"a+b\")+8)");
   if (f->GetNdim()!=1) return 0;
   res = f->Eval(0);
   if (res!=8) return 0;

   f = new TFormulaOld("f5","strstr(\"a + b\",\"a + b\")");
   if (f->GetNdim()!=1) return 0;
   res = f->Eval(0);
   if (res!=1) return 0;

   return 1;
}

void testString() {

   if (!execTestString())  gApplication->Terminate(1);

}
