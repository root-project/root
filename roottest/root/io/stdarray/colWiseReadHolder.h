#ifndef COL_WISE_READ_HOLDER
#define COL_WISE_READ_HOLDER

#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "arrayHolder.h"
#include "checkColWriteOutput_1.h"

#include <sstream>


int checkColWiseReadHolder(const char* filename) {
   auto file_read = TFile::Open(filename);
   TTreeReader myReader("mytree", file_read);
   TTreeReaderValue<ArrayHolder> r(myReader, "mybranch");
   std::stringstream content;
   while(myReader.Next()) {
      content << r->ToString();
   }

   file_read->Close();
   auto isCorrect =  isCorrect_1(content.str());
   return isCorrect ? 0 : 1;
}

int checkColWiseReadMetaHolder(const char* filename) {

   auto file_read = TFile::Open(filename);
   TTreeReader myReader("mytree", file_read);
   TTreeReaderValue<MetaArrayHolder> r(myReader, "mybranch");
   std::stringstream content;
   while(myReader.Next()) {
      content << r->ToString();
   }

   file_read->Close();
   auto isCorrect =  isCorrect_2(content.str());
   return isCorrect ? 0 : 1;
}

int checkColWiseReadMetaHolder2(const char* filename) {

   auto file_read = TFile::Open(filename);
   TTreeReader myReader("mytree", file_read);
   TTreeReaderArray<MetaArrayHolder2> r(myReader, "mybranch");
   std::stringstream content;
   while(myReader.Next()) {
      for (auto&& el : r){
         content << el.ToString();
      }
      content << "---\n";
   }

   file_read->Close();
   auto isCorrect =  isCorrect_3(content.str());
   return isCorrect ? 0 : 1;
}


#endif
