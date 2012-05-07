
#include "RooStats/HistFactory/PreprocessFunction.h"


RooStats::HistFactory::PreprocessFunction::PreprocessFunction() {;}


void RooStats::HistFactory::PreprocessFunction::Print( std::ostream& stream ) {
  
  stream << "\t \t Name: " << fName
	 << "\t \t Expression: " << fExpression
	 << "\t \t Dependents: " << fDependents
	 << std::endl;  
  
}
