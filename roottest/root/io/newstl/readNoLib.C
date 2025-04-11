void readNoLib(const char *filename = 0) {
   TFile *_file0 = TFile::Open(filename ? filename : "4-01-03/vector.root");
   

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

}
