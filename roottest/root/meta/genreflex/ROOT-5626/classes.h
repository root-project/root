#include <map>
#include <sstream>

#include "WeightContainer.h"

//typedef unsigned long long size_type;  // WORKAROUND for genreflex bug
//size_type should be HepMC::WeightContainer::size_type 
namespace {
	struct dictionary {
	};
}

namespace hepmc_rootio {
  inline void weightcontainer_set_default_names(unsigned int n, std::map<std::string,HepMC::WeightContainer::size_type>& names) {
      std::ostringstream name;
      for ( HepMC::WeightContainer::size_type count = 0; count<n; ++count ) 
      { 
        name.str(std::string());
        name << count;
        names[name.str()] = count;
      }
  }
}

