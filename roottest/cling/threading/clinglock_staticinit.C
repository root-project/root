const char* code = R"CODE(
std::atomic<long> res;
auto getCl = [&]{ res = (long)TClass::GetClass("THtml"); };
std::thread tr(getCl);
tr.join();
res.load()
)CODE";
long staticInit = gROOT->ProcessLine(code);

int clinglock_static() {
  const char* name = ((TClass*)(void*)staticInit)->GetName();
   if (!strcmp(name, "THtml"))
      return 0;
   Error("clinglock_static", "Failed to get TClass!");
   return 1;
}
