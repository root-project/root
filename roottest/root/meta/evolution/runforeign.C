{
   // gROOT->ProcessLine(".L datatwo.C+");

   gROOT->Reset();

   TFile *f = new TFile("data1.root");
   cout << "\n==> Looking at the StreamerInfo before loading the library\n"; 

   TClass *cl = gROOT->GetClass("data");
   TVirtualStreamerInfo *info = cl->GetStreamerInfo();
   info->ls();
#ifdef ClingWorkAroundJITandInline
   cout.setf(ios_base::hex, ios_base::basefield);
#endif
   cout << cl->GetName() << "'s streamerInfo #" 
      << info->GetClassVersion() << " has a checksum of "; 
#ifdef ClingWorkAroundJITandInline
   cout  << "0x" << info->GetCheckSum() << endl;
#else
   cout  << "0x" << hex << info->GetCheckSum() << endl;
#endif

   cl = gROOT->GetClass("Tdata");
   info = cl->GetStreamerInfo();
   info->ls();
   cout << cl->GetName() << "'s streamerInfo #" 
      << info->GetClassVersion() << " has a checksum of "; 
#ifdef ClingWorkAroundJITandInline
    cout  << "0x" << info->GetCheckSum() << endl;
#else
    cout  << "0x" << hex << info->GetCheckSum() << endl;
#endif

   gROOT->ProcessLine(".L data2.C+");

   cout << "\n==> Looking at the StreamerInfo after loading the library\n"; 

   cl = gROOT->GetClass("data");
   info = cl->GetStreamerInfo();
   info->ls();
   cout << cl->GetName() << "'s streamerInfo #" 
      << info->GetClassVersion() << " has a checksum of " ;
#ifdef ClingWorkAroundJITandInline
   cout << "0x" << info->GetCheckSum() << endl;
#else
   cout << "0x" << hex << info->GetCheckSum() << endl;
#endif

   cout << "\n==> List all the StreamerInfo after loading the library\n"; 
   cl->GetStreamerInfos()->ls();

   cout << "\n==> Looking at the StreamerInfo after loading the library\n"; 

   cl = gROOT->GetClass("Tdata");
   info = cl->GetStreamerInfo();
   info->ls();
   cout << cl->GetName() << "'s streamerInfo #" 
      << info->GetClassVersion() << " has a checksum of " ;
#ifdef ClingWorkAroundJITandInline
   cout << "0x" << info->GetCheckSum() << endl;
#else
   cout << "0x" << hex << info->GetCheckSum() << endl;
#endif

   cout << "\n==> List all the StreamerInfo after loading the library\n"; 
   cl->GetStreamerInfos()->ls();

   TFile *f2 = new TFile("data2.root");
   TFile *f3 = new TFile("data3.root");
   TFile *f4 = new TFile("data4.root");

   cout << "\n==> List all the StreamerInfo after loading all the files\n"; 

   gROOT->GetClass("data")->GetStreamerInfos()->ls();

   f3->Get("myobj");

#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}
