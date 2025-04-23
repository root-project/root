#ifndef DataFormats_SiStripCluster_Classes_H
#define DataFormats_SiStripCluster_Classes_H

#include "Wrapper.h"
#include "DetSetVector.h"
#include "DetSetVectorNew.h"
#include <vector>

// #include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SiStripCluster.h"
namespace DataFormats_SiStripCluster {
   struct dictionary1 {
      edm::Wrapper<SiStripCluster > zs0;
      edm::Wrapper<std::vector<SiStripCluster> > zs1;
      //     edm::Wrapper<edm::DetSet<SiStripCluster> > zs2;
      // edm::Wrapper<std::vector<edm::DetSet<SiStripCluster> > > zs3;
      //     edm::Wrapper<edm::DetSetVector<SiStripCluster> > zs4;
      edm::Wrapper<edmNew::DetSetVector<SiStripCluster> > zs4_bis;
   };
}

#endif // DataFormats_SiStripCluster_Classes_H

