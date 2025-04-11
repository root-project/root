int assertUnnamedTypeDictionary() {
   if (gROOT->ProcessLine(".L UnnamedTypes.h+s")) {
      Error("assertUnnamedTypeDict()", "Error building library.");
      return 1;
   }
   if (!gROOT->GetListOfGlobals()->FindObject("UnnamedClassInstance")) {
      Error("assertUnnamedTypeDict()", "Error building library.");
      return 1;
   }
   return 0;
}
