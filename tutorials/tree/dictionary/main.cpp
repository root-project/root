// ##########################################################################
// # Alvaro Tolosa Delgado @ IFIC(Valencia,Spain)  alvaro.tolosa@ific.uv.es #
// # Copyright (c) 2018 Alvaro Tolosa. All rights reserved.		 #
// ##########################################################################

#include "readTree.cpp"
#include "writeTree.cpp"

int main()
{
   std::cerr << "Starting writeTree()..." << std::endl;
   writeTree();
   std::cerr << "Starting writeTree()... Done! " << std::endl;

   std::cerr << "Starting readTree()..." << std::endl;
   readTree();
   std::cerr << "Starting readTree()... Done! " << std::endl;

   return 0;
}