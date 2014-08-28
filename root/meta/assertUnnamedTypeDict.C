int assertUnnamedTypeDict() {
   if (gROOT->ProcessLine(".L UnnamedTypes.h+")) {
      Error("assertUnnamedTypeDict()", "Error building library.");
      return 1;
   }
   if (!gROOT->GetListOfGlobals()->FindObject("UnnamedClassInstance")) {
      Error("assertUnnamedTypeDict()", "Error building library.");
      return 1;
   }
   return 0;
}
