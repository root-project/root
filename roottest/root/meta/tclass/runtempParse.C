{
   gROOT->ProcessLine(".L tempParse.C+");
   TClass *cl1 = gROOT->GetClass("SimpleProperty<string,Verifier<string> >");
   TClass *cl2 = gROOT->GetClass("SimpleProperty<string,Verifier<string>>");
   if (cl1!=cl2) cout << "Error missing a space in the classname was fatal\n";

   TClass *cl3 = TClass::GetClass("std::vector<int>::push_back");
   if (cl3) cout << "Error, TClass thinks that the following is a class: " << cl3->GetName() << endl;
   cl3 = TClass::GetClass("std::vector<int>::empty");
   if (cl3) cout << "Error, TClass thinks that the following is a class: " << cl3->GetName() << endl;
   cl3 = TClass::GetClass("std::vector<int>::pindakaas");
   if (cl3) cout << "Error, TClass thinks that the following is a class: " << cl3->GetName() << endl;

   cl3 = TClass::GetClass("std::vector<int>");
   if (!cl3) cout << "Could not find std::vector<int>\n";

   return 0;
}