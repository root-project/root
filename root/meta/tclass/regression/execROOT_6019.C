template <class T1, class T2>
struct B {
};

template <class T1, class T2>
struct C: public B<T1,T2> {
};

template <class T2>
struct B<void,T2> {
};

template <class T2>
struct C<void,T2>: B<void,T2> {
};

int execROOT_6019() {
  int result = 0;
  
  TClass *cl = TClass::GetClass("C<void,double>");
  cl->GetListOfBases()->ls("noaddr");
  cout << "The base is C<void,double> " << cl->GetListOfBases()->At(0)->GetName();
  cout << '\n';

  return result;
}
