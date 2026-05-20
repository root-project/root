#include "TInterpreter.h"

#include <unistd.h>
#include <sstream>
#include <iostream>
#include <sys/wait.h>

// A simple program that tests forking.

constexpr auto commandsChild = R"(
gDebug=1;
TH1F h1("h","",100,0,1);
std::vector<std::list<TGraph>> v;
h1.GetNbinsX();
gSystem->LoadAllLibraries()
)";

constexpr auto commandsParent = R"(
gDebug=1;
TH1F h1("h","",100,0,1);
std::vector<std::list<TGraph>> v;
std::set<TString> stringset;
stringset.size();
v.size();
)";

bool injectInCling(const char *commandSequence)
{
   bool success = true;

   std::string line;
   std::stringstream commandStream(commandSequence);
   while (commandStream.good()) {
      std::getline(commandStream, line);
      TInterpreter::EErrorCode errorCode = TInterpreter::kNoError;

      gInterpreter->ProcessLine(line.c_str(), &errorCode);

      if (errorCode != TInterpreter::kNoError) {
         std::cerr << "Interpreter returned error " << errorCode << " for line\n\t" << line << "\n";
         success = false;
      }
   }

   return success;
}

int main()
{
   bool success = true;
   std::cout << "Starting\n";

   gInterpreter->ProcessLine("TGraph g;");

   pid_t pid = fork();

   if (pid == 0){
      // child process
      success &= injectInCling(commandsChild);
   } else if (pid > 0) {
      // parent process
      success &= injectInCling(commandsParent);

      int status = 0;
      wait(&status);

      if (WIFEXITED(status)) {
         int exit_status = WEXITSTATUS(status);
         if (exit_status != 0) {
            std::cerr << "Child exited with status " << exit_status << "\n";
            return 2;
         }
      }
   } else {
      // fork failed
      std::cerr << "fork() failed!\n";
      return 3;
   }

   return success ? 0 : 1;
}
