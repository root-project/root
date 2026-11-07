struct ImagesRAII {
   std::vector<std::string> fFileNames;
   ImagesRAII(const std::vector<std::string> &fileNames) : fFileNames(fileNames) {}
   ~ImagesRAII()
   {
      for (auto &&fileName : fFileNames) {
         gSystem->Unlink(fileName.c_str());
      }
   }
};

int simpleImages()
{
   TCanvas c;
   TH1D h("myHisto", "The Title;the X;the Y", 64, -4, 4);
   h.FillRandom("gaus");
   h.Draw();
   std::vector<std::string> fileNames{"f.jpeg", "f.png", "f.gif", "f.bmp", "f.tiff"};
   ImagesRAII iraii(fileNames);
   for (auto &&fileName : fileNames) {
      c.SaveAs(fileName.c_str());
   }
   return 0;
}
