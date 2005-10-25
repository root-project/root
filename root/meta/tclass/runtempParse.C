{
   gROOT->ProcessLine(".L tempParse.C+");
   TClass *cl1 = gROOT->GetClass("SimpleProperty<string,Verifier<string> >");
   TClass *cl2 = gROOT->GetClass("SimpleProperty<string,Verifier<string>>");
   if (cl1!=cl2) cout << "Error missing a space in the classname was fatal\n";
   return 0;
}