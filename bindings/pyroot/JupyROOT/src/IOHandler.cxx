// Author: Danilo Piparo, Omar Zapata   16/12/2015

#include <string>

class JupyROOTExecutorHandler {
private:
   bool fCapturing;
   std::string fStdoutpipe;
   std::string fStderrpipe;
   int fStdout_pipe[2];
   int fStderr_pipe[2];
   int fSaved_stderr;
   int fSaved_stdout;
public:
   JupyROOTExecutorHandler();
   void Poll();
   void InitCapture();
   void EndCapture();
   void Clear();
   std::string& GetStdout();
   std::string& GetStderr();
};

bool JupyROOTExecutorImpl(const char *code);
bool JupyROOTDeclarerImpl(const char *code);


//------------------------------------------------------------------------------

#include <unistd.h>
#include <fcntl.h>
#ifndef F_LINUX_SPECIFIC_BASE
#define F_LINUX_SPECIFIC_BASE       1024
#endif
#ifndef F_SETPIPE_SZ
#define F_SETPIPE_SZ    (F_LINUX_SPECIFIC_BASE + 7)
#endif

constexpr long MAX_PIPE_SIZE = 1048575;

JupyROOTExecutorHandler::JupyROOTExecutorHandler()
{
   fCapturing = false;
}

static void PollImpl(FILE* stdStream, int* pipeHandle, std::string& pipeContent)
{
   int buf_read;
   char ch;
   fflush(stdStream);
   while (true)          {
      buf_read = read(pipeHandle[0], &ch, 1);
      if (buf_read == 1) {
         pipeContent += ch;
      } else break;
   }
}

void JupyROOTExecutorHandler::Poll()
{
   PollImpl(stdout, fStdout_pipe, fStdoutpipe);
   PollImpl(stderr, fStderr_pipe, fStderrpipe);
}

static void InitCaptureImpl(int& savedStdStream, int* pipeHandle)
{
   savedStdStream = dup(STDOUT_FILENO);
   if (pipe(pipeHandle) != 0) {
      return;
   }
   long flags_stdout = fcntl(pipeHandle[0], F_GETFL);
   flags_stdout |= O_NONBLOCK;
   fcntl(pipeHandle[0], F_SETFL, flags_stdout);
   fcntl(pipeHandle[0], F_SETPIPE_SZ, MAX_PIPE_SIZE);
   dup2(pipeHandle[1], STDOUT_FILENO);
   close(pipeHandle[1]);
}

void JupyROOTExecutorHandler::InitCapture()
{
   if (!fCapturing)  {
      InitCaptureImpl(fSaved_stdout, fStdout_pipe);
      InitCaptureImpl(fSaved_stderr, fStderr_pipe);
   }
}

void JupyROOTExecutorHandler::EndCapture()
{
   if (fCapturing)  {
      Poll();
      dup2(fSaved_stdout, STDOUT_FILENO);
      dup2(fSaved_stderr, STDERR_FILENO);
   }
   fCapturing = false;
}

std::string &JupyROOTExecutorHandler::GetStdout()
{
   return fStdoutpipe;
}

std::string &JupyROOTExecutorHandler::GetStderr()
{
   return fStderrpipe;
}

void JupyROOTExecutorHandler::Clear()
{
   fStdoutpipe = "";
   fStderrpipe = "";
}

#include "TInterpreter.h"

bool JupyROOTExecutorImpl(const char *code)
{
   auto status = false;
   try {
      if (gInterpreter->ProcessLine(code))    {
         status = true;
      }
   }
   catch(...) {
      status = true;
   }
   return status;
}

bool JupyROOTDeclarerImpl(const char *code)
{
   bool status = false;
   try {
      if (gInterpreter->Declare(code)) {
         status = true;
      }
   }
   catch(...) {
      status = true;
   }
   return status;
}


JupyROOTExecutorHandler *JupyROOTExecutorHandler_ptr = nullptr;

extern "C" {

   int JupyROOTExecutor(const char *code)
   {
      return JupyROOTExecutorImpl(code);
   }
   int JupyROOTDeclarer(const char *code)
   {
      return JupyROOTDeclarerImpl(code);
   }

   void JupyROOTExecutorHandler_Clear()
   {
      JupyROOTExecutorHandler_ptr->Clear();
   }

   void JupyROOTExecutorHandler_Ctor()
   {
      JupyROOTExecutorHandler_ptr = new JupyROOTExecutorHandler();
   }

   void JupyROOTExecutorHandler_Poll()
   {
      JupyROOTExecutorHandler_ptr->Poll();
   }

   void JupyROOTExecutorHandler_EndCapture()
   {
      JupyROOTExecutorHandler_ptr->EndCapture();
   }

   void JupyROOTExecutorHandler_InitCapture()
   {
      JupyROOTExecutorHandler_ptr->InitCapture();
   }

   const char *JupyROOTExecutorHandler_GetStdout()
   {
      auto out = JupyROOTExecutorHandler_ptr->GetStdout();
      if (out.empty()) return 0;
      return out.c_str();
   }

   const char *JupyROOTExecutorHandler_GetStderr()
   {
      auto out = JupyROOTExecutorHandler_ptr->GetStderr();
      if (out.empty()) return 0;
      return out.c_str();
   }

   void JupyROOTExecutorHandler_Dtor()
   {
      delete JupyROOTExecutorHandler_ptr;
   }

}
