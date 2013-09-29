// Test the XML parsing of genreflex.
// The testbed is composed by all xml selection files of LHCb and CMS, downloaded
// on September the 25th 2013.

int execXMLReader (){
   
   
   
   TString dirName("experimentsSelectionXMLs");
   
   TSystemDirectory dir(dirName.Data(), dirName.Data());
   TList *files = dir.GetListOfFiles();
   if (!files) return 1;
     
   TSystemFile *file;
   TString fname;
   TIter next(files);
   while ((file=(TSystemFile*)next())) {
      fname = file->GetName();
      if (!file->IsDirectory() && fname.EndsWith(".xml")) {
         TString command = "genreflex emptyHeader.h --quiet  -s "+dirName+"/"+fname;
         std::cout << "Testing: " << fname << std::endl;
         int ret = gSystem->Exec(command);
         if (ret!=0){
            std::cerr << "Failure in testing " << command.Data() << std::endl;
            return ret;
         }
      }
   }
   
   return 0;
   
}
