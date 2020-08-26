using std::string;

std::pair<string,string> SplitPathAndName(const char *path)
{
   char sep = '/';

#ifdef WIN32
   sep = '\\';
#endif

   string s(path);
   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      return std::make_pair(s.substr(0, i), s.substr(i+1));
   }

   return std::make_pair("", path);
}

// Helper function to check if CPPYY_BACKEND_LIBRARY
// points to an existing file.
// If it does not, the variable is unset to let cppyy
// look for the library in its default directories.
void check_cppyy_backend()
{
   auto lcb = gSystem->Getenv("CPPYY_BACKEND_LIBRARY");
   if (lcb) {
      auto pathAndName = SplitPathAndName(lcb);
      auto path = pathAndName.first;
      TString name(pathAndName.second);
      if (!gSystem->FindFile(path.c_str(), name))
         gSystem->Unsetenv("CPPYY_BACKEND_LIBRARY");
   }
}

