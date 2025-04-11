template<typename T> struct Data { T fVal; };

template <typename T>
int execTest(const char *classname)
{
   T obj;
   TClass *in = TClass::GetClass(classname);
   if (!in) return 1;
   TClass *res = in->GetActualClass(&obj);
   if (!res) { fprintf(stdout,"No result from GetActualClass for %s\n",in->GetName()); return 2; }
   fprintf(stdout, "For %s, got %s\n",in->GetName(),res->GetName());
   return 0;
}

int execInterpActualClass() {
   int result = execTest<pair<const string, Data<int>>>("pair<const string, Data<int>>");
   if (result) return result;

   result = execTest<vector<Data<int const>>>("vector<Data<int const>>");
   if (result) return result;

   return 0;
#if 0
   pair<const string, Data<int>> p;
   TClass *in = TClass::GetClass("pair<const string, Data<int>>");
   if (!in) return 1;
   TClass *res = in->GetActualClass(&p);
   if (!res) { fprintf(stdout,"No result from GetActualClass for %s\n",in->GetName()); return 2; }
   fprintf(stdout, "For %s, got %s\n",in->GetName(),res->GetName());

   vector<const Data<int>> p2
   TClass::GetClass("vector<const Data<int>>")->GetActualClass(&p2)->GetName()
   TClass::GetClass("vector<Data<int const>>")->GetActualClass(&p2)->GetName()
#endif

}