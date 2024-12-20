// Author: Alvaro Tolosa-Delgado CERN 07/2023
// Author: Jorge Agramunt Ros IFIC(Valencia,Spain) 07/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>

void writeTree();
void readTree();
void readTreeDF();

int main()
{
   std::cout << "Starting writeTree()..." << std::endl;
   writeTree();
   std::cout << "Starting writeTree()... Done! " << std::endl;
   std::cout << "Starting readTree()..." << std::endl;
   readTree();
   std::cout << "Starting readTree()... Done! " << std::endl;
   std::cout << "Starting readTreeDF()..." << std::endl;
   readTreeDF();
   std::cout << "Starting readTreeDF()... Done! " << std::endl;

   return 0;
}
