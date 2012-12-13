{
   SetROOTMessageToStdout();
   gROOT->ProcessLine(".x iobug.C+");
   for(int classtype = 0; classtype < 2; ++classtype) {
      for (int clonesmode = 1; clonesmode <4; ++clonesmode) {
         for (int split = 0; split < 4; ++split) {
            int dumpmode = 0;
            int show = 0;
            fprintf(stdout,"Running iobug.C(%d,%d,%d,%d,%d)\n",
                    split,classtype,clonesmode,show,dumpmode);
#ifdef ClingWorkAroundMissingDynamicScope
            gROOT->ProcessLine(TString::Format("iobug(%d,%d,%d,%d,%d);",
                                               split,classtype,clonesmode,show,dumpmode));
#else
            iobug(split,classtype,clonesmode,show,dumpmode);
#endif
         }
      }
   }

}
//iobug.C(1,0,3,0,0)
//iobug.C(1,1,3,0,0)



