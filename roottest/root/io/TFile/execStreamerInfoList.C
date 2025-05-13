void execStreamerInfoList()
{
   gSystem->CopyFile("geodemo.root","geodemo-update.root", kTRUE);
   auto file = TFile::Open("geodemo-update.root","update");
   if (!file || file->IsZombie()) {
      Error("execStreamerInfoList","Could not open file geodemo-update.root");
      gSystem->Exit(1);
   }
   auto i1 = file->GetStreamerInfoList()->FindObject("TGeoParaboloid");
   if (!i1) {
      Error("execStreamerInfoList","Can not find streamerInfo for TGeoParaboloid.");
      gSystem->Exit(2);
   }

   TGeoTessellated *g = new TGeoTessellated("triaconthaedron", 30);
   file->WriteObject(g,"tess");

   file->Close();
   delete file;


   file = TFile::Open("geodemo-update.root","update");
   if (!file || file->IsZombie()) {
      Error("execStreamerInfoList","Could not open file geodemo-update.root");
      gSystem->Exit(3);
   }
   auto i2 = file->GetStreamerInfoList()->FindObject("TGeoParaboloid");
   if (!i2) {
      Error("execStreamerInfoList","Can not find streamerInfo for TGeoParaboloid after update.");
      gSystem->Exit(4);
   }
   file->Close();
   delete file;
}
