bool readfiles(bool cont = false) {
   const char * files[] = { "int.root","float16.root","double32.root","regular.root","char.root","short.root","long.root",
      "longlong.root","uchar.root","ushort.root","uint.root","ulong.root","ulonglong.root","float.root","double.root",
      "float16enough.root","float16tooshort.root","float16mantis.root",
      "double32enough.root","double32tooshort.root","double32mantis.root"};

   bool result = true;
   
   for( int i = 0; i < sizeof(files)/sizeof(char*); ++i ) {
      TString filename( Form("%s",files[i] ) );
      if (!readfile(filename)) {
         result = false;
      }
      if (!cont && !result) return false;
   }
   return result;
}

bool read( const char *name) 
{
  compile(name);
  return !readfiles(true);
}
