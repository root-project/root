void runstaticFunc() {

   gROOT->ProcessLine(".L MyClass.cxx+");
  MyClass * my = new MyClass();
  my->Init();
  my->Integral(0,5);
}
