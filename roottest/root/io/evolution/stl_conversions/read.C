#ifdef ClingWorkAroundMissingDynamicScope
int readfile_trampoline(const char*filename, Bool_t checkValue = kTRUE)
{
   return gROOT->ProcessLine(TString::Format("readfile(\"%s\",%d);",filename,checkValue));
}
#endif

bool readfiles_stl(char** files, bool cont = false) {

   bool result = true;

   for( int i = 0; files[i]; ++i ) {
      TString filename( Form("%s",files[i] ) );
#ifdef ClingWorkAroundMissingDynamicScope
      if (!readfile_trampoline(filename)) {
#else
      if (!readfile(filename)) {
#endif
         result = false;
      }
      if (!cont && !result) return false;
   }

   return result;

}

void extractfiles(const char* s, char** arr)
{
   int num = 0;
   size_t len = strlen(s);
   TString f;
   while (*s) {
      if (*s == ' ') {
         if (f.Length()) {
            arr[num] = new char[f.Length() + 1];
            strcpy(arr[num++], f.Data());
            f = "";
         }
      } else {
         f += *s;
      }
      ++s;
   }
   if (f.Length()) {
      arr[num] = new char[f.Length() + 1];
      strcpy(arr[num++], f.Data());
      f = "";
   }
}

bool read(const char* filespass, const char *name)
{
   char * pass[256] = {0};

   extractfiles(filespass, pass);

   compile(name);
   return !readfiles_stl(pass, true);
}
