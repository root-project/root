
#include "TSystem.h"
#include "ROOT/TDataFrame.hxx"
#include "ROOT/TCsvDS.hxx"
#include <iostream>
#include <cassert>

int test_progressiveCSV()
{
   // Input file: 5000 rows (> 700 KB)
   auto fileName = "test_progressiveCSV.csv";
   const auto sizeInKB = 700.;

   // Create a CSV data source that reads in chunks of 10 rows 
   const auto chunkSize = 10LL;
   auto tdf = ROOT::Experimental::TDF::MakeCsvDataFrame(fileName, true, ',', chunkSize);
   
   ProcInfo_t info;
   gSystem->GetProcInfo(&info);
   auto memBefore = info.fMemResident; // KB

   std::cout << "Number of rows (progressive): " << *tdf.Count() << std::endl;

   gSystem->GetProcInfo(&info);
   auto memAfter = info.fMemResident;

   // Check that indeed we read in chunks (consumed memory should be much less than 700 KB)
   assert(memAfter - memBefore < sizeInKB);


   // Now create a CSV data source that reads the entire file into memory at once
   auto tdf2 = ROOT::Experimental::TDF::MakeCsvDataFrame(fileName);

   gSystem->GetProcInfo(&info);
   memBefore = info.fMemResident;

   std::cout << "Number of rows (at once): " << *tdf2.Count() << std::endl;

   gSystem->GetProcInfo(&info);
   memAfter = info.fMemResident;

   // Check that this time we allocated memory for all the rows 
   assert(memAfter - memBefore > sizeInKB);   

   return 0; 
}

int main()
{
   return test_progressiveCSV();
}
