{
   std::string_view sv("asd");
   std::cout << sv << std::endl;
   if (sv == "asd") {
      std::cout << "Comparison with char[4] worked\n";
   } else {
      std::cout << "ERROR: Comparison with char[4] failed\n";
   }
   return 0;
}

