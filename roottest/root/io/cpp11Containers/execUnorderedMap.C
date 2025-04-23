/* Stress the usage of unordered_{multi}map.
 * Dimensons to test
 * o type of streaming (rowwise,columnwise)
 * o "T" (POD,class, nested)
 * o For row wise: XML and binary root format
 */

// The test still has some repetitions which one could wash away with metaprogramming.

#include "commonUtils.h"


template<class T>
void checkObjects(const char* name, const T& a, const T& b){
   if (IsSame(a,b)) return;
   std::cerr << "Objects called " << name << " differ!\n";
}

template <class KeyType, template <typename... T> class Cont>
void check(const char* testName){

   printf("o Checking %s\n",testName);

   TH1::AddDirectory(0); // same name is ok
   gRandom->SetSeed(1); // make all contents identical irrespective of the container

   std::string binFilename(testName); binFilename+="UnorderedMap"; binFilename+=".root";
   std::string xmlFilename(testName); xmlFilename+="UnorderedMap"; xmlFilename+=".xml";

   std::vector<const char*> filenames  {binFilename.c_str()/*, xmlFilename.c_str()*/}; // NEED XML HERE

   for (auto&& filename : filenames){
      createFile(filename);
   }

   Cont<KeyType,double> doubleCont {{1,1.},
                                    {2,2.},
                                    {3,3.},
                                    {4,4.}}; // need double32_t
   Cont<KeyType,TH1F> histoCont {{1,TH1F("h1","ht",100,-2,2)}, {9,TH1F("h2","ht",10,-1.2,1.2)}};
   fillHistoAssoCont(histoCont);

   vector<Cont<KeyType,TH1F>> vecHistoCont {{{1,TH1F("h1","ht",100,-2,2)}, {2,TH1F("h2","ht",10,-1.23,1.23)}},
                                            {{7,TH1F("h3","ht",100,-23,23)},{8,TH1F("h4","ht",10,-1.92,1.92)}}};
   fillHistoNestedAssoCont(vecHistoCont);

   Cont<KeyType,vector<TH1F>> contHistoVec {{1,{TH1F("h1","ht",100,-2,2),TH1F("h2","ht",10,-1.23,1.23)}},
                                            {2,{TH1F("h3","ht",100,-23,23),TH1F("h4","ht",10,-1.92,1.92)}}};
   fillHistoNestedAssoCont(contHistoVec);

   printf("  - RowWise\n");

   // Row wise
   for (auto&& filename : filenames){
      writeReadCheck(doubleCont,"doubleCont",filename);
      writeReadCheck(histoCont,"histoCont",filename);
      writeReadCheck(vecHistoCont,"vecHistoCont",filename);
      writeReadCheck(contHistoVec,"contHistoVec",filename);
   }

   // ColumnWise
   printf("  - ColumnWise\n");
   int NEvts=100;
   // Make a backup of the input
   auto doubleContOrig = doubleCont;
   auto histoContOrig = histoCont;
   auto vecHistoContOrig = vecHistoCont;
   auto contHistoVecOrig = contHistoVec;

   // Write
   gRandom->SetSeed(1);
   {
      printf("    * Write\n");
      TFile f(binFilename.c_str(),"UPDATE");
      TTree t("t","Test Tree");
      t.Branch("doubleCont_split", &doubleCont,16000,99);
      t.Branch("doubleCont", &doubleCont,16000,0);
      t.Branch("histoCont_split", &histoCont,16000,99);
      t.Branch("histoCont", &histoCont,16000,0);
      t.Branch("vecHistoCont_split", &vecHistoCont,16000,99);
      t.Branch("vecHistoCont", &vecHistoCont,16000,0);
      t.Branch("contHistoVec_split", &contHistoVec,16000,99);
      t.Branch("contHistoVec", &contHistoVec,16000,0);

      for (int i=0;i<NEvts;++i){
         randomizeAssoCont(doubleCont);
         fillHistoAssoCont(histoCont,10);
         fillHistoNestedAssoCont(contHistoVec,10);
         t.Fill();
      }
      t.Write();
   }
   // And Read
   gRandom->SetSeed(1);
   {
      printf("    * Read\n");
      TFile f(binFilename.c_str());
      TTreeReader reader("t", &f);
      TTreeReaderValue<decltype(doubleCont)> rdoubleCont_split(reader, "doubleCont_split");
      TTreeReaderValue<decltype(doubleCont)> rdoubleCont(reader, "doubleCont");
      TTreeReaderValue<decltype(histoCont)> rhistoCont_split(reader, "histoCont_split");
      TTreeReaderValue<decltype(histoCont)> rhistoCont(reader, "histoCont");
      TTreeReaderValue<decltype(vecHistoCont)> rvecHistoCont_split(reader, "vecHistoCont_split");
      TTreeReaderValue<decltype(vecHistoCont)> rvecHistoCont(reader, "vecHistoCont");
      TTreeReaderValue<decltype(contHistoVec)> rcontHistoVec_split(reader, "contHistoVec_split");
      TTreeReaderValue<decltype(contHistoVec)> rcontHistoVec(reader, "contHistoVec");
      for (int i=0;i<NEvts;++i){
         // Rebuild original values
         randomizeAssoCont(doubleContOrig);
         fillHistoAssoCont(histoContOrig,10);
         fillHistoNestedAssoCont(contHistoVecOrig,10);
         // Now check them
         reader.Next();
         checkObjects("doubleCont_split",doubleContOrig,*rdoubleCont_split);
         checkObjects("doubleCont",doubleContOrig,*rdoubleCont);
         checkObjects("histoCont_split",histoContOrig,*rhistoCont_split);
         checkObjects("histoCont",histoContOrig,*rhistoCont);
         checkObjects("vecHistoCont_split",vecHistoContOrig,*rvecHistoCont_split);
         checkObjects("vecHistoCont",vecHistoContOrig,*rvecHistoCont);
         checkObjects("contHistoVec_split",contHistoVecOrig,*rcontHistoVec_split);
         checkObjects("contHistoVec",contHistoVecOrig,*rcontHistoVec);
      }
   }



}

template <class T>
void checkMaps(){
   // This as a baselinecheck
   check<T, std::map>("map");
   check<T,std::multimap>("multimap");

   // This is for the actual collection
   check<T,std::unordered_map>("unordered_map");
   check<T,std::unordered_multimap>("unordered_multimap");
}

void execUnorderedMap(){

   checkMaps<int>();
   checkMaps<float>();
   checkMaps<double>();
   checkMaps<Long64_t>();

}
