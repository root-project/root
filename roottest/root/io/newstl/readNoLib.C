int readNoLib(const char *fname = "vector.root")
{
   TString dirname = gROOT->GetVersion();
   dirname.ReplaceAll(".","-");
   dirname.ReplaceAll("/","-");

   auto _filename = gSystem->ConcatFileName(dirname, fname);
   TString filename = _filename;
   delete [] _filename;

   auto file0 = TFile::Open(filename);
   if (!file0)
      return 1;

   auto stltree = (TTree *) file0->Get("stltree");
   if (!stltree)
      return 1;

   stltree->Scan("split99.fScalar","","colsize=30");
   stltree->Scan("split3.fScalar","","colsize=30");  // intentionally on missing branche
   stltree->Scan("split2.fScalar","","colsize=30");
   stltree->Scan("split1.fScalar","","colsize=30");
   stltree->Scan("split0.fScalar","","colsize=30");
   stltree->Scan("split_1.fScalar","","colsize=30");
   stltree->Scan("split_2.fScalar","","colsize=30");

   stltree->Scan("split99.fPairFlInt.first","","colsize=30");
   stltree->Scan("split2.fPairFlInt.first","","colsize=30");
   stltree->Scan("split1.fPairFlInt.first","","colsize=30");
   stltree->Scan("split0.fPairFlInt.first","","colsize=30");
   stltree->Scan("split_1.fPairFlInt.first","","colsize=30");
   stltree->Scan("split_2.fPairFlInt.first","","colsize=30");

   // print all zeros? -> Problem with strings!
   stltree->Scan("split99.fPairStrDb","","colsize=30");
   stltree->Scan("split2.fPairStrDb","","colsize=30");
   stltree->Scan("split1.fPairStrDb","","colsize=30");
   stltree->Scan("split0.fPairStrDb","","colsize=30");
   stltree->Scan("split_1.fPairStrDb","","colsize=30");
   stltree->Scan("split_2.fPairStrDb","","colsize=30");

   stltree->Scan("split99.fObject.dval");
   stltree->Scan("split2.fObject.dval");
   stltree->Scan("split1.fObject.dval");
   stltree->Scan("split0.fObject.dval");
   stltree->Scan("split_1.fObject.dval");
   stltree->Scan("split_2.fObject.dval");

   stltree->Scan("split99.fPairStrDb.first","","colsize=30");
   stltree->Scan("split2.fPairStrDb.first","","colsize=30");
   stltree->Scan("split1.fPairStrDb.first","","colsize=30");
   stltree->Scan("split0.fPairStrDb.first","","colsize=30");
   stltree->Scan("split_1.fPairStrDb.first","","colsize=30");
   stltree->Scan("split_2.fPairStrDb.first","","colsize=30");

   stltree->Scan("split99.fTemplates.val.val.val","","colsize=30");
   stltree->Scan("split2.fTemplates.val.val.val","","colsize=30");
   stltree->Scan("split1.fTemplates.val.val.val","","colsize=30");
   stltree->Scan("split0.fTemplates.val.val.val","","colsize=30");
   stltree->Scan("split_1.fTemplates.val.val.val","","colsize=30");
   stltree->Scan("split_2.fTemplates.val.val.val","","colsize=30");

   return 0;
}
