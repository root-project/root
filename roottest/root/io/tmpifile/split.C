void split()
{
   try
   {
      TMPIFile *newfile = new TMPIFile("test_split.root", "RECREATE", 2);
   } catch(const std::exception&)
   {
      return EXIT_FAILURE;
   }
}
