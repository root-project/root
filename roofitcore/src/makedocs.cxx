void makedocs(const char *version="Development", const char *where= "./html") 
{
  const char *bfarch= gSystem->Getenv("BFARCH");
  TString sourceDir("RELEASE/RooFitCore:RELEASE/tmp/");
  sourceDir.Append(bfarch);
  sourceDir.Append("/RooFitCore");
  gEnv->SetValue("Root.Html.SourceDir",sourceDir.Data());
  gEnv->SetValue("Root.Html.OutputDir",where);
  gEnv->SetValue("Root.Html.Description","// -- CLASS DESCRIPTION");
  gEnv->SetValue("Root.Html.LastUpdate"," *    File: $Id: ");

  RooHtml docMaker(version);
  docMaker.Convert("demo13.cc","Using variable binning") ;
  docMaker.Convert("demo14.cc","Plotting a PDF projection on a subset of the event sample") ;
  docMaker.Convert("demo15.cc","Plotting with a cut on the likelihood") ;
  docMaker.Convert("demo16.cc","Plotting slices of simultaneous PDFs") ;
  return ;

  docMaker.MakeAll(kTRUE,"Roo*");
  docMaker.MakeIndexNew("Roo*");  

  docMaker.addTopic("PDF","Probability Density functions") ;
  docMaker.addTopic("REAL","Real valued functions") ;
  docMaker.addTopic("CAT","Discrete valued functions") ;
  docMaker.addTopic("DATA","Unbinned and binned data") ;
  docMaker.addTopic("PLOT","Plotting and tabulating") ;
  docMaker.addTopic("CONT","Container classes") ;
  docMaker.addTopic("MISC","Miscellaneous") ;
  docMaker.addTopic("USER","Other user classes") ;
  docMaker.addTopic("AUX","Auxiliary classes for internal use") ;
  docMaker.MakeIndexOfTopics() ;
}
