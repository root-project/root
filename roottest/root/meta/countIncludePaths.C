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

   // count paths coming from the ROOT_INCLUDE_PATH environment variable
   // and exclude them
   int nEnvVarPaths = 0;
   auto *envVarCStr = std::getenv("ROOT_INCLUDE_PATH");
   if (envVarCStr) {
      std::string envVar(envVarCStr);
      nEnvVarPaths = countSubstring(envVar, ":") + 1 - (envVar.back() == ':') - (envVar.front() == ':');
   }

   // At most 10
   auto nPaths = countSubstring(includePath, "-I");
   if ((nPaths - nEnvVarPaths) > 10) {
      std::cerr << "The number of include paths is too high (>9) " << nPaths
                << ". The number of \"-I\"s has been counted in the include path of ROOT (gSystem->GetIncludePath()=" << includePath << ")." << std::endl;
      return nPaths;
   }
   return 0;
}
