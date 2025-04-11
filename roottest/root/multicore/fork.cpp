#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "TInterpreter.h"
#include "TSystem.h"

// A simple program that tests forking.

// Suppress the output of roofit
class outputRAII{
public:
   outputRAII(){
     printf("Filtering out RooFit banner\n");
      oldCoutStreamBuf = std::cout.rdbuf();
      std::cout.rdbuf( strCout.rdbuf() );
   }
   ~outputRAII(){
   std::cout.rdbuf( oldCoutStreamBuf );
   std::string line;
   while(std::getline(strCout,line,'\n')){
      if (line.find("Wouter") != std::string::npos &&
          line.find("NIKHEF") != std::string::npos &&
          line.find("sourceforge") != std::string::npos){
         printf("Unexpected output line: %s\n ", line.c_str());
      }
   }
   }
private:
   std::stringstream strCout;
   std::streambuf* oldCoutStreamBuf;
};

void injectInCling(const char* filename){

   std::string line;
   std::ifstream infile;
   infile.open (filename);
   while(!infile.eof()){
      std::getline(infile,line);
//      printf("%s\n", line.c_str());
       gInterpreter->ProcessLine(line.c_str());
   }
}

int main()
{
   printf("Starting\n");

   gInterpreter->ProcessLine("TGraph g;");

   pid_t pid = fork();

   if (pid == 0){
      // child process
      outputRAII out;
      injectInCling("commands1.txt");
   }
   else if (pid > 0){
      // parent process
      outputRAII out;
      injectInCling("commands2.txt");
   }
   else{
      // fork failed
      printf("fork() failed!\n");
      return 1;
   }

   printf("Program with finished\n");


   return 0;
}
