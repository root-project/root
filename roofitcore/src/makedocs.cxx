void makedocs(const char *version, const char *where= "./html") {

  const char *bfarch= gSystem->Getenv("BFARCH");
  TString sourceDir("RELEASE/RooFitCore:RELEASE/tmp/");
  sourceDir.Append(bfarch);
  sourceDir.Append("/RooFitCore");
  gEnv->SetValue("Root.Html.SourceDir",sourceDir.Data());
  gEnv->SetValue("Root.Html.OutputDir",where);
  gEnv->SetValue("Root.Html.Description","// -- CLASS DESCRIPTION --");
  gEnv->SetValue("Root.Html.LastUpdate","File: $#$");

  THtml docMaker;
  docMaker.MakeAll(kTRUE,"Roo*");
}
