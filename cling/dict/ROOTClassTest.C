// http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=11933

#include <vector>
#include "TVector3.h"

void ROOTClassTest()
{
  std::vector<TVector3> mv;
  
  mv.push_back( TVector3(12., 13., 14.) );
  std::cout << mv.back().Y() << std::endl;
  
  mv.pop_back();  
  if( mv.empty() )std::cout << 0 << std::endl;
  else std::cout << 1 << std::endl;
}
