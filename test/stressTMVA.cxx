// $Id$
// Author: Joerg Stelzer   11/2007 
///////////////////////////////////////////////////////////////////////////////////
//
//  TMVA functionality test suite
//  =============================
//
//  This program performs tests of TMVA
//
//   To run the program do: 
//   stressTMVA          : run standard test
//
///////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <ostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <sstream>

#include "TString.h"
#include "TFile.h"
#include "TMath.h"
#include "TH1.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TMatrixD.h"


#include "TMVA/Factory.h"
#include "TMVA/Event.h"
#include "TMVA/PDF.h"
#include "TMVA/Reader.h"
#include "TMVA/Types.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableNormalizeTransform.h"


namespace TMVA {

class TMVATest {



   public:
   
      enum tmvatests { CutsGA = 0x001, Likelihood = 0x002, LikelihoodPCA = 0x004, PDERS   = 0x008,
		   KNN    = 0x010, HMatrix    = 0x020, Fisher        = 0x040, FDA_MT  = 0x080,
		   MLP    = 0x100, SVM_Gauss  = 0x200, BDT           = 0x400, RuleFit = 0x800 };

      TMVATest() {}
      ~TMVATest() {}

      void prepareClassificationData();
      void getGaussRnd( TArrayD& v, const TMatrixD& sqrtMat, TRandom& R );
      TMatrixD* produceSqrtMat( const TMatrixD& covMat );


      int test_event();
      int test_createDataSetFromTree();
      int test_pdf_streams();
      int test_factory_basic();
      int test_reader_basic();
      int test_variable_transforms();

   private:

};



void TMVATest::prepareClassificationData(){
   std::cout << "---------- prepare data -----START " << std::endl;
   // create the data
   Int_t N = 20000;
   const Int_t nvar = 4;
   Float_t xvar[nvar];

   // output file
   TFile* dataFile = TFile::Open( "testData.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   std::cout << "signal covariance matrix: " << std::endl;
   covMatS->Print();
   std::cout << "background covariance matrix: " << std::endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; std::cout << "- produce signal" << std::endl; }
      else            { x = xB; m = sqrtMatB; std::cout << "- produce background" << std::endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) std::cout << "... event: " << i << " (" << N << ")" << std::endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   std::cout << "created data file: " << dataFile->GetName() << std::endl;

   std::cout << "---------- prepare data -----END " << std::endl;
}



TMatrixD* TMVATest::produceSqrtMat( const TMatrixD& covMat )
{
   Double_t sum = 0;
   Int_t size = covMat.GetNrows();;
   TMatrixD* sqrtMat = new TMatrixD( size, size );

   for (Int_t i=0; i< size; i++) {
      
      sum = 0;
      for (Int_t j=0;j< i; j++) sum += (*sqrtMat)(i,j) * (*sqrtMat)(i,j);

      (*sqrtMat)(i,i) = TMath::Sqrt(TMath::Abs(covMat(i,i) - sum));

      for (Int_t k=i+1 ;k<size; k++) {

         sum = 0;
         for (Int_t l=0; l<i; l++) sum += (*sqrtMat)(k,l) * (*sqrtMat)(i,l);

         (*sqrtMat)(k,i) = (covMat(k,i) - sum) / (*sqrtMat)(i,i);

      }
   }
   return sqrtMat;
}

void TMVATest::getGaussRnd( TArrayD& v, const TMatrixD& sqrtMat, TRandom& R ) 
{
   // generate "size" correlated Gaussian random numbers

   // sanity check
   const Int_t size = sqrtMat.GetNrows();
   if (size != v.GetSize()) 
      std::cout << "<getGaussRnd> too short input vector: " << size << " " << v.GetSize() << std::endl;

   Double_t* tmpVec = new Double_t[size];

   for (Int_t i=0; i<size; i++) {
      Double_t x, y, z;
      y = R.Rndm();
      z = R.Rndm();
      x = 2*TMath::Pi()*z;
      tmpVec[i] = TMath::Sin(x) * TMath::Sqrt(-2.0*TMath::Log(y));
   }

   for (Int_t i=0; i<size; i++) {
      v[i] = 0;
      for (Int_t j=0; j<=i; j++) v[i] += sqrtMat(i,j) * tmpVec[j];
   }

   delete tmpVec;
}













int TMVATest::test_event(){
   std::cout << "---------- event -----START " << std::endl;
   std::cout << "           constructor: Event() " << std::endl;
   Event* evt0 = new Event();
   delete evt0;

   std::cout << "           constructor: Event( const std::vector<Float_t>&, Bool_t isSignal = kTRUE, Float_t weight = 1.0, Float_t boostweight = 1.0 ) " << std::endl;
   std::vector<Float_t> ev;
   ev.push_back( 1.5 );
   ev.push_back( 2.3 );
   ev.push_back( 8.8 );
   ev.push_back( 2.2 );
   Event *ptrEv = new Event(ev,Types::kSignal );
//   ptrEv->Print( std::cout );
   for( UInt_t iv = 0; iv < ptrEv->GetNVars(); iv++ ){
//      std::cout << "iv : "<< iv  << " val: " << ptrEv->GetVal( iv ) << " evval: " << ev[iv] << std::endl;
      assert( TMath::Abs( ptrEv->GetVal(iv) - ev[iv] ) < 1e-10 );
   }
   delete ptrEv;

   std::cout << "           constructor: Event( const std::vector<Float_t*>*& ) " << std::endl;
   Float_t var1 = 1.2;
   Float_t var2 = 3.3;
   Float_t var3 = 0.0;
   Float_t var4 = -0.12;
   std::vector<Float_t*> *vars = new std::vector<Float_t*>(0);
   vars->push_back( &var1 );
   vars->push_back( &var2 );
   vars->push_back( &var3 );
   vars->push_back( &var4 );

   Event *evt2 = new Event((const std::vector<Float_t*>*&) vars );
   assert( TMath::Abs( var1 - evt2->GetVal( 0 ) ) < 1e-10 );
   delete evt2;

   std::cout << "---------- event -----END " << std::endl;
   return 0;
}







int TMVATest::test_createDataSetFromTree(){
   std::cout << "---------- createDataSetFromTree -----START " << std::endl;
   
   prepareClassificationData();

   TFile *f = TFile::Open( "testData.root" );
   TTree *treeS = (TTree*)f->Get( "TreeS" );
   TTree *treeB = (TTree*)f->Get( "TreeB" );

   DataInputHandler *dih = new DataInputHandler();
   DataSetManager::CreateInstance(*dih);

   TString dsiName = "testDataSet";
   DataSetInfo* dsi = DataSetManager::Instance().GetDataSetInfo(dsiName);
   if(dsi!=0){
      std::cout << "dataset with name " << dsiName << " already present." << std::endl;
      return true;
   }
   std::cout << "no dataset with name " << dsiName << " found. A new one will be created." << std::endl;


   dsi = new DataSetInfo(dsiName);
   DataSetManager::Instance().AddDataSetInfo(*dsi);

   dsi->AddVariable( "var1" );
   dsi->AddVariable( "var2" );
   dsi->AddVariable( "var3" );
   dsi->AddVariable( "var4" );


   dih->AddSignalTree( treeS );
   dih->AddBackgroundTree( treeB );

   std::cout << "     datasets for signal and background created ==> loop through events" << std::endl;

   DataSet* ds = DataSetManager::Instance().CreateDataSet( dsiName );

//    ds = dsi->GetDataSet(); 
   UInt_t n = 0;
   std::cout << "number of events=" << ds->GetNEvents() << std::endl;
   for( UInt_t ievt=0; ievt<ds->GetNEvents(); ievt++ ){
      const Event *ev = ds->GetEvent(ievt);
//       Float_t val = ev->GetVal(ievt);
      if( n%5000== 0 ){
	 std::cout << "<event=" << n << ":vars=" << ev->GetNVariables(); std::cout.flush();
	 for( UInt_t ivar = 0; ivar < ev->GetNVariables(); ivar++ ){
	    std::cout << "|" << ivar << "=" << ev->GetVal(ivar); std::cout.flush();
	 }
	 std::cout << ">"; std::cout.flush();
      }
      n++;
   }
   std::cout << std::endl;


   std::cout << "---------- createDataSetFromTree -----END " << std::endl;
   return 0;
}




int TMVATest::test_pdf_streams(){

   std::cout << "---------- pdf_streams -----START " << std::endl;
   // prepare a histogram for the production of a pdf
   TH1F *hist = new TH1F( "histogram_test_pdf_streams", "histogram_test_pdf_streams", 10, 0.0,10.0 );
   // fill in some values
   for( int i=0; i<100; i++ ){
      hist->Fill( i/10.0 );
   }

   // produce a pdf from the histogram
   PDF *pdf = new PDF( "testPdf" );
   pdf->BuildPDF( hist );

   // create an outstream
   ofstream fout;
   fout.open( "test_pdf_streams.temp" );

   // dump pdf into the outstream (file) and close the file
   fout<<*pdf;
   fout.close();
   
   // delete the pdf
   //delete pdf;
   
   // prepare another pdf
   TMVA::PDF *rec = new TMVA::PDF( "test2Pdf" );
   
   // open the file again
   ifstream fin ( "test_pdf_streams.temp" , ifstream::in );
   fin >> *rec;
   
   TH1 *hrec = rec->GetOriginalHist();

   for( int i=0; i<10; i++ ){
      //std::cout << "check bin " << i << std::endl;
      assert( hist->GetBin( i ) == hrec->GetBin( i ) );
   }

   // delete everything
   delete rec;
   delete pdf;
   delete hist;

   std::cout << "---------- pdf_streams -----END " << std::endl;
   return 0;
}





int TMVATest::test_variable_transforms(){
   std::cout << "---------- variable_transforms -----START " << std::endl;

   test_createDataSetFromTree();

   TString dsiName = "testDataSet";
   TMVA::DataSetInfo* dsi = TMVA::DataSetManager::Instance().GetDataSetInfo(dsiName);

   int returnValue = 0;

//   TMVA::Event::ClearDynamicVariables();

   std::vector<VariableInfo> vars;
   vars.push_back( VariableInfo( "var1", "var1Title", "var1unit", 0, 'F' ) );
   vars.push_back( VariableInfo( "var2", "var2Title", "var2unit", 0, 'F' ) );
   vars.push_back( VariableInfo( "var3", "var3Title", "var3unit", 0, 'F' ) );
   vars.push_back( VariableInfo( "var4", "var4Title", "var4unit", 0, 'F' ) );

   // create some transforms
   std::vector<VariableTransformBase*> vtb;
   //   vtb.push_back( new TMVA::VariableGaussTransform( *dsi ) );
   vtb.push_back( new TMVA::VariableIdentityTransform( *dsi ) ); 
   vtb.push_back( new TMVA::VariableDecorrTransform( *dsi ) );
   vtb.push_back( new TMVA::VariablePCATransform( *dsi ) );
   vtb.push_back( new TMVA::VariableNormalizeTransform( *dsi ) );

   // create some doubles for the transforms of which the values will be set 
   // from an input stream.
   std::vector<VariableTransformBase*> vtbRec;
   //   vtbRec.push_back( new TMVA::VariableGaussTransform( *dsi ) );
   vtbRec.push_back( new TMVA::VariableIdentityTransform( *dsi ) );
   vtbRec.push_back( new TMVA::VariableDecorrTransform( *dsi ) );
   vtbRec.push_back( new TMVA::VariablePCATransform( *dsi ) );
   vtbRec.push_back( new TMVA::VariableNormalizeTransform( *dsi ) );

   TRandom *rnd = new TRandom3();
   std::vector<Event*> evts;
   int nSig = 100;
   int nBack = 100;
   for( int i=0; i<nSig+nBack; i++ ){
      std::vector<Float_t> ev;
      ev.push_back( rnd->Rndm()*(i/3.0) );
      ev.push_back( (i<nSig?10.0:20.0)+rnd->Rndm()*i*i+i );
      ev.push_back( (i<nSig?30.0:25.0)+rnd->Rndm()*i*i );
      ev.push_back( 5.0+rnd->Rndm()*i );
      Event *ptrEv = new Event(ev,(i<nSig?Types::kSignal : Types::kBackground) );
      evts.push_back( ptrEv );
      //      ptrEv->Print( std::cout );
      for( UInt_t iv = 0; iv < ptrEv->GetNVars(); iv++ ){
	 //	 std::cout << "iv : "<< iv  << " val: " << ptrEv->GetVal( iv ) << " evval: " << ev[iv] << std::endl;
	 assert( TMath::Abs( ptrEv->GetVal( iv ) - ev[iv] ) < 1e-10 ); // difference between original value and value stored in event
      }
   }

   bool everythingOK = true;
   std::vector<VariableTransformBase*>::iterator itRec = vtbRec.begin();
   for( std::vector<VariableTransformBase*>::iterator it = vtb.begin(); it < vtb.end(); it++ ){
      std::cout << ">TRANSFORMATION: " << (*it)->GetName() << std::endl;
      (*it)->PrepareTransformation( evts );
      // create an outstream
      std::stringstream st;
      st << "test_transformation_" << (*it)->GetName() << ".temp";

      ofstream fout;
      fout.open( st.str().c_str() );
      (*it)->WriteTransformationToStream( fout );
      fout.close();

      // create an inputstream
      ifstream fin ( st.str().c_str() , ifstream::in );
      (*itRec)->ReadTransformationFromStream( fin );
      fin.close();

      // now loop through all events and compare the transformations of the events with
      // the oiginal and the read in transformation.
      Int_t nCls = dsi->GetNClasses();
      if( nCls > 1 ) nCls++;
      for( Int_t cls = 0; cls < nCls; cls ++ ){
	 for( std::vector<Event*>::iterator itEv = evts.begin(); itEv < evts.end(); itEv++ ){
	    const Event* evOriginal = (*it)->Transform( (*itEv), cls );
	    const Event* evRecieved = (*itRec)->Transform( (*itEv), cls );
	    for( UInt_t iv = 0; iv < (*itEv)->GetNVars(); iv++ ){
	       bool isOK = TMath::Abs( evOriginal->GetVal( iv )-evRecieved->GetVal( iv ) )<1e-10;
	       if( !isOK ) std::cout << "iv " << iv << " orig: " << evOriginal->GetVal( iv ) << "  rec: " << evRecieved->GetVal( iv ) << std::endl;
	       everythingOK &= isOK;
	       //assert( everythingOK ); // transformation of events ARE NOT EQUAL for original and re-read transformation
	    }
	 }
	 if(!everythingOK) {
	    std::cout << "ORIGINAL TRANSFORMATION" << std::endl;
	    (*it)->PrintTransformation( std::cout );
	    std::cout << std::endl;
	    std::cout << "READ TRANSFORMATION" << std::endl;
	    (*itRec)->PrintTransformation( std::cout );
	    std::cout << std::endl;
	 }
	 //      if( everythingOK ){
	 //	 std::cout << "             transformations of events are EQUAL for original and copied transformation" << std::endl;
	 //      }else{
	 //	 std::cout << "             transformations of events ARE NOT EQUAL for original and copied transformation" << std::endl;
	 //      }
	 assert( everythingOK ); // transformation of events ARE NOT EQUAL for original and re-read transformation
	 returnValue &= int(everythingOK);
	 itRec++;  // take as well the next transformation for the "receiver" transformation
      }
   }

   std::cout << "---------- variable_transforms -----END " << std::endl;
   return returnValue;
}





int TMVATest::test_factory_basic(){
	return 1;
}








int TMVATest::test_reader_basic(){
   std::cout << "---------- reader_basic -----START " << std::endl;
   //
   // create the Reader object
   //
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   Float_t var1, var2;
   Float_t var3, var4;
   reader->AddVariable( "var1+var2", &var1 );
   reader->AddVariable( "var1-var2", &var2 );
   reader->AddVariable( "var3",      &var3 );
   reader->AddVariable( "var4",      &var4 );
   //
   // book a MVA method
   //
   std::string dir    = "../macros/weights/";
   std::string prefix = "TMVAnalysis";
   reader->BookMVA( "Fisher method",        dir + prefix + "_Fisher.weights.txt" );
   TFile *input(0);
   TString fname = "../macros/tmva_example.root";   
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find tmva_example.root in the local directory
      std::cout << "--- Accessing data file: " << fname << std::endl;
      input = TFile::Open( fname );
   } 
   else { 
      // second we try accessing the file via the web from
      // http://root.cern.ch/files/tmva_example.root
      std::cout << "--- Accessing tmva_example.root file from http://root.cern.ch/files" << std::endl;
      std::cout << "--- for faster startup you may consider downloading it into you local directory" << std::endl;
      input = TFile::Open("http://root.cern.ch/files/tmva_example.root");
   }
   //
   UInt_t nbin = 100;
   TH1F *histFi;
   histFi = new TH1F( "MVA_Fisher", "MVA_Fisher", nbin, -4, 4 );// prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeS");
   std::cout << "--- Select signal sample" << std::endl;
   Float_t userVar1, userVar2;
   theTree->SetBranchAddress( "var1", &userVar1 );
   theTree->SetBranchAddress( "var2", &userVar2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );
   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
   //TStopwatch sw;
   //sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0){
	 std::cout << "--- ... Processing event: " << ievt << std::endl;
      }

      theTree->GetEntry(ievt);

      var1 = userVar1 + userVar2;
      var2 = userVar1 - userVar2;
      histFi->Fill( reader->EvaluateMVA( "Fisher method" ) );      
   }
   //sw.Stop();

   // delete everything
   delete reader;
   delete histFi;

   std::cout << "---------- reader_basic -----END " << std::endl;
   return true;
}



} // namespace TMVA




int main( int argc, char** argv ) {
   TMVA::TMVATest testsuite;

   std::cout << "******************************************************************" << std::endl;
   std::cout << "* TMVA - S T R E S S suite *" << std::endl;
   std::cout << "******************************************************************" << std::endl;

   int back = 999;

   std::string what = "";

   if( argc>1 ) {
      what = argv[1];
   }

   if( what == "EVENT" ) back = testsuite.test_event();
   if( what == "CREATE_DATASET" ) back = testsuite.test_createDataSetFromTree();
   if( what == "TEST_PDF_STREAMS" ) back = testsuite.test_pdf_streams();
   if( what == "TEST_VARIABLE_TRANSFORMS" ) back = testsuite.test_variable_transforms();
   if( what == "TEST_READER_BASIC" ) back = testsuite.test_reader_basic();

   if( what == "" ){ 
      back = testsuite.test_event();
      back += testsuite.test_createDataSetFromTree();
//      back += testsuite.test_pdf_streams();
//      back += testsuite.test_variable_transforms();
//      back += testsuite.test_reader_basic();
   }


   if( back == 0 ) std::cout << "SUCCESSFUL: " << what << std::endl;
   else            std::cout << "ERROR:      " << what << std::endl;


   return back;
}








/* For the emacs users in the crowd.
Local Variables:
   c-basic-offset: 3
End:
*/
