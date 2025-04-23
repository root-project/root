int hadd_check_filtered(const char *fname, const char *type)
{
   std::unique_ptr<TFile> file{TFile::Open(fname, "READ")};

   const bool isWhitelist = strcmp(type, "whitelist") == 0;
   auto check = [isWhitelist, &file](const char *name, bool existsInWhitelist) {
      const bool exists = !!file->Get(name);
      bool shouldExist = existsInWhitelist == isWhitelist;
      if (exists != shouldExist) {
         if (exists)
            std::cerr << name << " exists but shouldn't!\n";
         else
            std::cerr << name << "doesn't exist but should!\n";
      }
      return exists == shouldExist;
   };

   // clang-format off
   bool ok = check("tree",     true) &&
             check("dir",      true) &&
             check("form",     true) &&
             check("dir/hist", true) &&
             check("hist",    false) &&
             check("ntpl",    false);
   // clang-format on
   return !ok;
}
