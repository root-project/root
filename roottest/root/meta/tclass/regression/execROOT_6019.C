template <typename T1, typename T2, typename T3>
struct B {
};

template <typename T1, typename T2, typename T3>
struct C: public B<T1,T2,T3> {
};

template <class T2,class T3>
struct B<void,T2,T3> {
};

template <class T2,class T3>
struct C<void,T2,T3>: B<void,T2,T3> {
};

template <class T2,class T3>
struct B<T2,void,T3> {
};

template <class T2,class T3>
struct C<T2,void,T3>: B<T2,void,T3> {
};

template <class T2,class T3>
struct B<T2,T3,void> {
};

template <class T2,class T3>
struct C<T2,T3,void>: B<T2,T3,void> {
};

void PrintClass(TString cname)
{
   TClass *cl = TClass::GetClass(cname);
   cl->GetListOfBases()->ls("noaddr");
   cout << "The base of " << cname << " is listed as " << cl->GetListOfBases()->At(0)->GetName();
   cout << '\n';
}

int execROOT_6019() {
  int result = 0;

  TString cname;
  cname.Form("C<void,double,float>");
  PrintClass(cname);
  cname.Form("C<void,Double32_t,Float16_t>");
  PrintClass(cname);
  cname.Form("C<double,void,float>");
  PrintClass(cname);
  cname.Form("C<Double32_t,void,Float16_t>");
  PrintClass(cname);
  cname.Form("C<double,Float16_t,void>");
  PrintClass(cname);
  cname.Form("C<Double32_t,Float16_t,void>");
  PrintClass(cname);

  return result;
}
