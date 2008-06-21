bool readfiles(int type, bool cont = false) {
   const char * files[] = { "int.root","float16.root","double32.root","regular.root","char.root","short.root","long.root",
      "longlong.root","uchar.root","ushort.root","uint.root","ulong.root","ulonglong.root","float.root","double.root",
      "float16enough.root","float16mantis.root",
      "double32enough.root","double32mantis.root"};

   const char * failing[] = { "float16tooshort.root","double32tooshort.root" };

   bool result = true;
   

   for( int i = 0; i < sizeof(files)/sizeof(char*); ++i ) {
      TString filename( Form("%s",files[i] ) );
      if (!readfile(filename)) {
         result = false;
      }
      if (!cont && !result) return false;
   }

   for( int i = 0; i < sizeof(failing)/sizeof(char*); ++i ) {
      TString filename( Form("%s",failing[i] ) );
      if (!readfile(filename,false)) {
         result = false;
      }
      if (!cont && !result) return false;
   }

   return result;

}

bool readfiles_stl(bool cont = false) {
   const char * files[] = { "map.root", "multimap.root", "vector.root", "list.root" };

   //   const char * failing[] = { "" } ; // "float16tooshort.root","double32tooshort.root" };

   bool result = true;
   

   for( int i = 0; i < sizeof(files)/sizeof(char*); ++i ) {
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

bool read(int type, const char *name) 
{
   compile(type, name);
   if (type==0) {
      return !readfiles(true);
   } else {
      return !readfiles_stl(true);
   }
}
