#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "TTree.h"
#include "TRandom.h"
#include <fstream>
using namespace RooFit ;

TTree* makeTTree() ;


void ASCII_in_out(bool writeFile, const char* filename)
{
   // -----------------------------------------
   // I m p o r t i n g   R O O T  T T r e e s
   // =========================================


   // I m p o r t   T T r e e   i n t o   a   R o o D a t a S e t
   // -----------------------------------------------------------

   TTree* tree = makeTTree() ;

   RooRealVar x("x","x",-10,10) ;
   RooRealVar y("y","y",-10,10) ;

   // Construct unbinned dataset importing tree branches x and y matching between branches and RooRealVars 
   // is done by name of the branch/RRV 
   // 
   // Note that ONLY entries for which x,y have values within their allowed ranges as defined in 
   // RooRealVar x and y are imported. Since the y values in the import tree are in the range [-15,15]
   // and RRV y defines a range [-10,10] this means that the RooDataSet below will have less entries than the TTree 'tree'

   RooDataSet ds("ds","ds",RooArgSet(x,y),Import(*tree)) ;
   
   
   
   // U s e   a s c i i   i m p o r t / e x p o r t   f o r   d a t a s e t s
   // ------------------------------------------------------------------------------------
   if (writeFile) {
      // Write data to output stream
	  std::ofstream outstream(filename);
	  // Optionally, adjust the stream here (e.g. std::setprecision)
	  outstream << std::setprecision(9);
      ds.write(outstream);
      outstream.close();
   } else {
	   //Read data from input stream. The variables of the dataset need to be supplied
	   //to the RooDataSet::read() function.
	   RooDataSet * dataReadBack = RooDataSet::read(filename, RooArgList(x, y), "Q");
	   
	   dataReadBack->Print("V");
	   
	   cout << std::setprecision(9);
	   for (std::size_t i=0; i < ds.numEntries(); ++i) {
		  std::cout << "Generated on the fly: ";
		  ds.get(i)->writeToStream(std::cout, true);
		  std::cout << "Read from reference : ";
		  dataReadBack->get(i)->writeToStream(std::cout, true);
	   }
   }
}



TTree* makeTTree() 
{
  // Create ROOT TTree filled with a Gaussian distribution in x and a uniform distribution in y

  TTree* tree = new TTree("tree","tree") ;
  Double_t* px = new Double_t ;
  Double_t* py = new Double_t ;
  tree->Branch("x",px,"x/D") ;
  tree->Branch("y",py,"y/D") ;
  gRandom->SetSeed(1337);
  for (int i=0 ; i<100 ; i++) {
    *px = gRandom->Gaus(0,3) ;
    *py = gRandom->Uniform()*30 - 15 ;
    tree->Fill() ;
  }
  return tree ;
}


