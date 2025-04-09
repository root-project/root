int countSubstring(const std::string& str, const std::string& sub)
{
   if (sub.length() == 0) return 0;
   int count = 0;
   for (size_t offset = str.find(sub); offset != std::string::npos;
      offset = str.find(sub, offset + sub.length())) { ++count; }
   return count;
}

int countIncludePaths()
{
   for (auto lib : {"libMathCore", "libunordered_mapDict", "libmapDict", "libHist"})
      gSystem->Load(lib);

   std::string includePath(gSystem->GetIncludePath());

   // Exclude from the test the builds with R as external package
   if (std::string::npos != includePath.find("RInside/include")) return 0;

   // At most 10
   auto nPaths = countSubstring(includePath, "-I");
   if (nPaths > 10){
      std::cerr << "The number of include paths is too high (>9) " << nPaths
                << ". The number of \"-I\"s has been counted in the include path of ROOT (gSystem->GetIncludePath()=" << includePath << ")." << std::endl;
      return nPaths;
   }
   return 0;
}
