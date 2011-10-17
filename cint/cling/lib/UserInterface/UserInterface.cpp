//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include <cling/UserInterface/UserInterface.h>

#include <cling/MetaProcessor/MetaProcessor.h>
#include "textinput/TextInput.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"

#include <iostream>
#include <sys/stat.h>

namespace llvm {
  class Module;
}

//---------------------------------------------------------------------------
// Construct an interface for an interpreter
//---------------------------------------------------------------------------
cling::UserInterface::UserInterface(Interpreter& interp, const char* prompt /*= "[cling] $"*/):
m_MetaProcessor(new MetaProcessor(interp))
{
}


//---------------------------------------------------------------------------
// Destruct the interface
//---------------------------------------------------------------------------
cling::UserInterface::~UserInterface()
{
  delete m_MetaProcessor;
}

//---------------------------------------------------------------------------
// Interact with the user using a prompt
//---------------------------------------------------------------------------
void cling::UserInterface::runInteractively(bool nologo /* = false */)
{
  if (!nologo) {
    std::cerr << std::endl;
    std::cerr << "**** Welcome to the cling prototype! ****" << std::endl;
    std::cerr << "* Type C code and press enter to run it *" << std::endl;
    std::cerr << "* Type .q, exit or ctrl+D to quit       *" << std::endl;
    std::cerr << "*****************************************" << std::endl;
  }
  static const char* histfile = ".cling_history";
  std::string Prompt("[cling]$ ");

  using namespace textinput;
  StreamReader* R = StreamReader::Create();
  TerminalDisplay* D = TerminalDisplay::Create();
  TextInput TI(*R, *D, histfile);
  TI.SetPrompt(Prompt.c_str());
  std::string line;
  MetaProcessorOpts& MPOpts = m_MetaProcessor->getMetaProcessorOpts();
  
  while (!MPOpts.Quitting) {
    TextInput::EReadResult RR = TI.ReadInput();
    TI.TakeInput(line);
    if (RR == TextInput::kRREOF) {
      MPOpts.Quitting = true;
      continue;
    }

    int indent = m_MetaProcessor->process(line.c_str());
    Prompt = "[cling]";
    if (MPOpts.RawInput)
      Prompt.append("! ");
    else
      Prompt.append("$ ");

    if (indent > 0)
      // Continuation requested.
      Prompt.append('?' + std::string(indent * 3, ' '));

    TI.SetPrompt(Prompt.c_str());

  }
}
