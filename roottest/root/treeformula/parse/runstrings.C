{
   TFile *_file0 = TFile::Open("itsdb.2006-03-17-22-28-46.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *t_its; _file0->GetObject("t_its",t_its);
#endif
   t_its->SetScanField(0);
   t_its->Scan ("its.its_id", "\"qeff_1\"==its.its_name");
   t_its->Scan ("its.its_id", "its.its_name==\"qeff_1\"");
   t_its->Scan ("its.its_id", "its.its_name+2","",1);
   t_its->Scan ("its.its_id", "2+its.its_name","",1);
   gApplication->Terminate(0);
}
