{
#ifndef ClingWorkAroundMissingAutoLoading
   gSystem->Load("libTree");
#endif
   gROOT->ProcessLine(".x make_tuple.C");
   TFile *file = TFile::Open("make_tuple.root");
   TTree *data; file->GetObject("data",data);
#ifndef ClingWorkAroundBrokenUnnamedReturn
   if (data==0) return 1;
   data->MakeProxy("make_tuple_sel","make_tuple_draw.C");
   data->Process("make_tuple_sel.h+");
   return 0;
#else
   int res;
   if (data==0) {
      res = 1;
   } else {
      data->MakeProxy("make_tuple_sel","make_tuple_draw.C");
      data->Process("make_tuple_sel.h+");
      res = 0;
   }
   int ret = res;
#endif
}
