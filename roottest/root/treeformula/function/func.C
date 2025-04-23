double myfunc(int x,double y,long z) {
   cout << "myfunc with " << x << " " << y << " " << z << endl;
   return x+y+z;
}

bool mfunc() {
   gROOT->ProcessLine(".L testcl.C");

   TFormula *fabs = new TFormula("fabs","abs(x)");
   TFormula *fmabs = new TFormula("fmabs","TMath::Abs(x)");
   TFormula *f = new TFormula("f","myfunc(x,y,z)");
   TFormula *fromcl = new TFormula("f2","testcl::calc(x,y)");

   bool result = true;
   if ( fabs->Eval(-33.3) != 33.3 ) {
      cerr << "fabs fails an calculated : " << fabs->Eval(-33.3) << endl;
      result = false;
   }
   if ( fmabs->Eval(-88.4) != 88.4 ) {
      cerr << "fmabs fails an calculated : " << fmabs->Eval(-88.4) << endl;
      result = false;
   }
   if ( f->Eval(3,4.5,-50000000) != 3+4.5-50000000) {
      cerr << "myfunc fails with " << f->Eval(3,4.5,-50000000) << endl;
      result = false;
   }
   if ( fromcl->Eval(3,5) != 3*5 ) {
      cerr << "testcl::calc failed and returned: " << fromcl->Eval(3,5) << endl;
      result = false;
   }
   if (!result) gApplication->Terminate(1);
   return result;
}

int func() { 
   // this return value is returned to the shell!
   return !mfunc();
}
