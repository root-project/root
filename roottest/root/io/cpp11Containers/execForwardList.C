/* Stress the usage of forward_list.
 * Dimensons to test
 * o type of streaming (rowwise,columnwise)
 * o "T" (POD,class, nested)
 * o For row wise: XML and binary root format
 */

// The test still has some repetitions which one could wash away with metaprogramming.

#include "commonUtils.h"

// A shortcut
#define NESTEDCONT {{TH1F("h","ht",100,-2,2),TH1F("h","ht",10,-1.23,1.23)},{TH1F("h","ht",100,-23,23),TH1F("h","ht",10,-1.92,1.92)}}

template<class T>
void checkObjects(const char* name, const T& a, const T& b){
   if (IsSame(a,b)) return;
   std::cerr << "Objects called " << name << " differ!\n";
}

template <template <typename T,typename V=allocator<T>> class Cont>
void check(const char* testName){

   printf("o Checking %s\n",testName);

   TH1::AddDirectory(0); // same name is ok
   gRandom->SetSeed(1); // make all contents identical irrespective of the container

   std::string binFilename(testName); binFilename+="ForwardList"; binFilename+=".root";
   std::string xmlFilename(testName); xmlFilename+="ForwardList"; xmlFilename+=".xml";

   std::vector<const char*> filenames  {binFilename.c_str()/*, xmlFilename.c_str()*/}; // NEED XML HERE

   for (auto&& filename : filenames){
      createFile(filename);
   }

   Cont<double> doubleCont {1.,2.,3.,4.}; // need double32_t
   Cont<complex<double>> complexDCont {{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}};
   Cont<complex<float>> complexFCont {{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}};
   Cont<TH1F> histoCont {TH1F("h","ht",100,-2,2), TH1F("h","ht",10,-1.2,1.2)};
   fillHistoCont(histoCont);


   Cont<Cont<TH1F>> contHistoCont NESTEDCONT;
   fillHistoNestedCont(contHistoCont);

   vector<Cont<TH1F>> vecHistoCont NESTEDCONT;
   fillHistoNestedCont(vecHistoCont);

   Cont<vector<TH1F>> contHistoVec NESTEDCONT;
   fillHistoNestedCont(contHistoVec);

   printf("  - RowWise\n");

   // Row wise
   for (auto&& filename : filenames){
      writeReadCheck(doubleCont,"doubleCont",filename);
      writeReadCheck(complexDCont,"complexDCont",filename);
      writeReadCheck(complexFCont,"complexFCont",filename);
      writeReadCheck(histoCont,"histoCont",filename);
      writeReadCheck(contHistoCont,"contHistoCont",filename);
      writeReadCheck(vecHistoCont,"vecHistoCont",filename);
      writeReadCheck(contHistoVec,"contHistoVec",filename);
   }

   // ColumnWise
   printf("  - ColumnWise\n");
   int NEvts=100;
   // Make a backup of the input
   auto doubleContOrig = doubleCont;
   auto complexDContOrig = complexDCont;
   auto complexFContOrig = complexFCont;
   auto histoContOrig = histoCont;
   auto contHistoContOrig = contHistoCont;
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
      t.Branch("complexDCont_split", &complexDCont,16000,99);
      t.Branch("complexDCont", &complexDCont,16000,0);
      t.Branch("complexFCont_split", &complexFCont,16000,99);
      t.Branch("complexFCont", &complexFCont,16000,0);
      t.Branch("histoCont_split", &histoCont,16000,99);
      t.Branch("histoCont", &histoCont,16000,0);
      t.Branch("contHistoCont_split", &contHistoCont,16000,99);
      t.Branch("contHistoCont", &contHistoCont,16000,0);
      t.Branch("vecHistoCont_split", &vecHistoCont,16000,99);
      t.Branch("vecHistoCont", &vecHistoCont,16000,0);
      t.Branch("contHistoVec_split", &contHistoVec,16000,99);
      t.Branch("contHistoVec", &contHistoVec,16000,0);

      for (int i=0;i<NEvts;++i){
         randomizeCont(doubleCont);
         randomizeCont(complexDCont);
         randomizeCont(complexFCont);
         fillHistoCont(histoCont,10);
         fillHistoNestedCont(contHistoCont,10);
         fillHistoNestedCont(vecHistoCont,10);
         fillHistoNestedCont(contHistoVec,10);
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
      TTreeReaderValue<decltype(complexDCont)> rcomplexDCont_split(reader, "complexDCont_split");
      TTreeReaderValue<decltype(complexDCont)> rcomplexDCont(reader, "complexDCont");
      TTreeReaderValue<decltype(complexFCont)> rcomplexFCont_split(reader, "complexFCont_split");
      TTreeReaderValue<decltype(complexFCont)> rcomplexFCont(reader, "complexFCont");
      TTreeReaderValue<decltype(histoCont)> rhistoCont_split(reader, "histoCont_split");
      TTreeReaderValue<decltype(histoCont)> rhistoCont(reader, "histoCont");
      TTreeReaderValue<decltype(contHistoCont)> rcontHistoCont_split(reader, "contHistoCont_split");
      TTreeReaderValue<decltype(contHistoCont)> rcontHistoCont(reader, "contHistoCont");
      TTreeReaderValue<decltype(vecHistoCont)> rvecHistoCont_split(reader, "vecHistoCont_split");
      TTreeReaderValue<decltype(vecHistoCont)> rvecHistoCont(reader, "vecHistoCont");
      TTreeReaderValue<decltype(contHistoVec)> rcontHistoVec_split(reader, "contHistoVec_split");
      TTreeReaderValue<decltype(contHistoVec)> rcontHistoVec(reader, "contHistoVec");
      for (int i=0;i<NEvts;++i){
         // Rebuild original values
         randomizeCont(doubleContOrig);
         randomizeCont(complexDContOrig);
         randomizeCont(complexFContOrig);
         fillHistoCont(histoContOrig,10);
         fillHistoNestedCont(contHistoContOrig,10);
         fillHistoNestedCont(vecHistoContOrig,10);
         fillHistoNestedCont(contHistoVecOrig,10);
         // Now check them
         reader.Next();
         checkObjects("doubleCont_split",doubleContOrig,*rdoubleCont_split);
         checkObjects("doubleCont",doubleContOrig,*rdoubleCont);
         checkObjects("complexDCont_split",complexDContOrig,*rcomplexDCont_split);
         checkObjects("complexDCont",complexDContOrig,*rcomplexDCont);
         checkObjects("complexFCont_split",complexFContOrig,*rcomplexFCont_split);
         checkObjects("complexFCont",complexFContOrig,*rcomplexFCont);
         checkObjects("histoCont_split",histoContOrig,*rhistoCont_split);
         checkObjects("histoCont",histoContOrig,*rhistoCont);
         checkObjects("contHistoCont_split",contHistoContOrig,*rcontHistoCont_split);
         checkObjects("contHistoCont",contHistoContOrig,*rcontHistoCont);
         checkObjects("vecHistoCont_split",vecHistoContOrig,*rvecHistoCont_split);
         checkObjects("vecHistoCont",vecHistoContOrig,*rvecHistoCont);
         checkObjects("contHistoVec_split",contHistoVecOrig,*rcontHistoVec_split);
         checkObjects("contHistoVec",contHistoVecOrig,*rcontHistoVec);
      }
   }



}


void execForwardList(){

   // This as a baselinecheck
   check<std::list>("list");
   check<std::vector>("vector");
   check<std::deque>("deque");

   // This is for the actual collection
   check<std::forward_list>("forward_list");



}
