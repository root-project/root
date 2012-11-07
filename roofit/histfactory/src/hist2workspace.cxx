
#include <string>
#include <exception>

//void topDriver(string input); // in MakeModelAndMeasurements
//void fastDriver(string input); // in MakeModelAndMeasurementsFast

//#include "RooStats/HistFactory/MakeModelAndMeasurements.h"
#include "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc, char** argv) {

  if( !(argc>1) ) {
    std::cerr << "need input file" << std::endl;
    exit(1);
  }
  
  if(argc==2){
    std::string input(argv[1]);
    try {
      fastDriver(input);
    }
    catch (std::string str) {
      std::cerr << "caught exception: " << str << std::endl ;
    }
    catch( const std::exception& e ) {
      std::cerr << "Caught Exception: " << e.what() << std::endl;
    }
  }
  
  if(argc==3){
    std::string flag(argv[1]);
    std::string input(argv[2]);

    if(flag=="-standard_form") {
      try {
	fastDriver(input);
      }
      catch (std::string str) {
	std::cerr << "caught exception: " << str << std::endl ;
      }
      catch( const std::exception& e ) {
	std::cerr << "Caught Exception: " << e.what() << std::endl;
      }
    }
      
    else if(flag=="-number_counting_form") {
      try {
	std::cout << "ERROR: 'number_counting_form' is now depricated." << std::endl;
	//topDriver(input);
      }
      catch (std::string str) {
	std::cerr << "caught exception: " << str << std::endl ;
      }
      catch( const std::exception& e ) {
	std::cerr << "Caught Exception: " << e.what() << std::endl;
      }
    }
    
    else {
      std::cerr <<"unrecognized flag.  Options are -standard_form or -number_counting_form"<<std::endl;
    }
  }
  return 0;
}

#endif
