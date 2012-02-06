bool readfiles(const char** files, const char** failing, int type, bool cont = false) {
   bool result = true;
   

   for( int i = 0; files[i]; ++i ) {
      TString filename( Form("%s",files[i] ) );
      if (!readfile(filename)) {
         result = false;
      }
      if (!cont && !result) return false;
   }

   for( int i = 0; failing[i]; ++i ) {
      TString filename( Form("%s",failing[i] ) );
      if (!readfile(filename,false)) {
         result = false;
      }
      if (!cont && !result) return false;
   }

   return result;

}

bool readfiles_stl(const char** files, const char* failing, bool cont = false) {
   //   const char * failing[] = { "" } ; // "float16tooshort.root","double32tooshort.root" };

   bool result = true;
   

   for( int i = 0; files[i]; ++i ) {
      TString filename( Form("%s",files[i] ) );
      if (!readfile(filename)) {
         result = false;
      }
      if (!cont && !result) return false;
   }

//    for( int i = 0; i < sizeof(failing)/sizeof(char*); ++i ) {
//       TString filename( Form("%s",failing[i] ) );
//       if (!readfile(filename,false)) {
//          result = false;
//       }
//       if (!cont && !result) return false;
//    }

   return result;

}

void extractfiles(const char* s, char** arr) {
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

bool read(const char* filespass, const char* filesfail, int type, const char *name) 
{
   const char * pass[256] = {0};
   const char * fail[256] = {0};

   extractfiles(filespass, pass);
   extractfiles(filesfail, fail);

   compile(type, name);
   if (type==0) {
      return !readfiles(pass, fail, true);
   } else {
      return !readfiles_stl(pass, fail, true);
   }
}
