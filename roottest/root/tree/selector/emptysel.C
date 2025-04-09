#define emptysel_cxx

#include "emptysel.h"
#include <TH2.h>
#include <TStyle.h>
#include <iostream>

void emptysel::Begin(TTree * /*tree*/)
{
   printAddress();
   std::cout << "testvar in Begin: " << m_testvar << std::endl;
   TString option = GetOption();

}

void emptysel::SlaveBegin(TTree * /*tree*/)
{

   TString option = GetOption();
   printAddress();

}

Bool_t emptysel::Process(Long64_t /* entry */)
{

   printAddress();
   return kTRUE;
}

void emptysel::SlaveTerminate()
{
   printAddress();

}

void emptysel::Terminate()
{

   printAddress();
}
