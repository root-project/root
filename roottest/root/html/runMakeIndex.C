TString getSrcDir() {
   TString rootsys("$ROOTSYS");
   gSystem->ExpandPathName(rootsys);
   TString cfgLoc(rootsys + "/config/Makefile.config");
   ifstream cfgIn(cfgLoc);
   if (!cfgIn) {
      return rootsys;
   }
   TString line;
   while (line.ReadLine(cfgIn)) {
      if (line.BeginsWith("ROOT_SRCDIR")) {
         Ssiz_t posRP = line.Index("realpath, ");
         if (posRP != -1) {
            line.Remove(0, posRP + 10);
            line.Remove(line.Length() - 1);
         } else {
            line.Remove(0, 19);
         }
         return line;
      }
   }
   return rootsys;
}

void runMakeIndex() {
   THtml h;
   h.SetInputDir(getSrcDir());
   h.LoadAllLibs();
   h.MakeIndex();
}
