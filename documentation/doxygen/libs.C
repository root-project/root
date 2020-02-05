void libs(TString classname)
{
   const char *libname;

   // Find in which library classname sits
   classname.ReplaceAll("_1",":");
   classname.ReplaceAll("_01"," ");
   int i = classname.Index("_3");
   if (i>0) classname.Remove(i,classname.Length()-i);

   libname = gInterpreter->GetClassSharedLibs(classname.Data());

   // Library was not found, try to find it from a template in $ROOTSYS/lib/*.rootmap
   if (!libname) {
      gSystem->Exec(Form("grep %s $ROOTSYS/lib/*.rootmap | grep -m 1 map:class | sed -e 's/^.*class //' > classname.txt",classname.Data()));
      FILE *f = fopen("classname.txt", "r");
      char c[160];
      char *str = fgets(c,160,f);
      fclose(f);
      remove("classname.txt");
      if (!str) {
         printf("%s cannot be found in any of the .rootmap files\n", classname.Data());
         remove("libslist.dot");
         return;
      }
      TString cname = c;
      cname.Remove(cname.Length()-1, 1);
      libname = gInterpreter->GetClassSharedLibs(cname.Data());
      if (!libname) {
         printf("Cannot find library for the class %s\n",cname.Data());
         return;
      }
   }

   // Print the library name in a external file
   TString mainlib = libname;
   mainlib.ReplaceAll(" ","");
   mainlib.ReplaceAll(".so","");
   FILE *f = fopen("mainlib.dot", "w");
   fprintf(f,"   mainlib [label=%s];\n",mainlib.Data());
   fclose(f);

   // List of libraries used by libname via ldd on linux and otool on Mac

   gSystem->Exec(Form("$DOXYGEN_LDD $ROOTSYS/lib/%s | grep -v %s > libslist.dot",libname,libname));
}
