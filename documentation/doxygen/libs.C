void libs(TString classname)
{
   const char *libname;

   // Find in which library classname sits
   classname.ReplaceAll("_1",":");
   libname = gInterpreter->GetClassSharedLibs(classname.Data());

   if(!libname) return;

   // Print the library name in a external file
   TString mainlib = libname;
   mainlib.ReplaceAll(".so ","");
   FILE *f = fopen("mainlib.dot", "w");
   fprintf(f,"   mainlib [label=%s];\n",mainlib.Data());
   fclose(f);

   // List of libraries used by libname via ldd on linux and otool on Mac

   gSystem->Exec(Form("$DOXYGEN_LDD $ROOTSYS/lib/%s | grep -v %s > libslist.dot",libname,libname));
}
