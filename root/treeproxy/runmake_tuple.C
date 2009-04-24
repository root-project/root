{
   gROOT->ProcessLine(".x make_tuple.C");
   TFile *file = TFile::Open("make_tuple.root");
   TTree *data; file->GetObject("data",data);
   if (data==0) return 1;
   data->MakeProxy("make_tuple_sel","make_tuple_draw.C");
   data->Process("make_tuple_sel.h+");
   return 0;
}
