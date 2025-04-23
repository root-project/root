
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RCsvDS.hxx"
#include <iostream>

int test_progressiveCSV()
{
   // Input file: 5000 lines
   auto fileName = "test_progressiveCSV.csv";

   // Create a CSV data source that reads in chunks of 2000 lines
   const auto chunkSize = 2000LL;
   auto rdf = ROOT::RDF::FromCSV(fileName, true, ',', chunkSize);

   auto rdflines = *rdf.Count();
   std::cout << "Total num lines: " << rdflines << std::endl;

   // Now create a CSV data source that reads the entire file into memory at once
   auto rdf2 = ROOT::RDF::FromCSV(fileName);

   rdflines = *rdf2.Count();
   std::cout << "Total num lines: " << rdflines << std::endl;

   return 0; 
}

int main()
{
   return test_progressiveCSV();
}
