// Author: Danilo Piparo, Omar Zapata, Enric Tejedor   16/12/2015

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <Python.h>

#include <fcntl.h>
#ifdef _MSC_VER // Visual Studio
#include <winsock2.h>
#include <io.h>
#pragma comment(lib, "Ws2_32.lib")
#define pipe(fds) _pipe(fds, 1048575, _O_BINARY)
#define read _read
#define dup _dup
#define dup2 _dup2
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#else
#include <unistd.h>
#endif
#include <string>
#include <iostream>
#include "TInterpreter.h"

//////////////////////////
// MODULE FUNCTIONALITY //
//////////////////////////

class JupyROOTExecutorHandler {
private:
   bool fCapturing = false;
   std::string fStdoutpipe;
   std::string fStderrpipe;
   int fStdout_pipe[2] = {0, 0};
   int fStderr_pipe[2] = {0, 0};
   int fSaved_stderr = 0;
   int fSaved_stdout = 0;

public:
   JupyROOTExecutorHandler();
   void Poll();
   void InitCapture();
   void EndCapture();
   void Clear();
   std::string &GetStdout();
   std::string &GetStderr();
};

#ifndef F_LINUX_SPECIFIC_BASE
#define F_LINUX_SPECIFIC_BASE 1024
#endif
#ifndef F_SETPIPE_SZ
#define F_SETPIPE_SZ (F_LINUX_SPECIFIC_BASE + 7)
#endif

constexpr long MAX_PIPE_SIZE = 1048575;

JupyROOTExecutorHandler::JupyROOTExecutorHandler() {}

static void PollImpl(FILE *stdStream, int *pipeHandle, std::string &pipeContent)
{
   fflush(stdStream);
#ifdef _MSC_VER
   char buffer[60000] = "";
   struct _stat st;
   _fstat(pipeHandle[0], &st);
   if (st.st_size) {
      _read(pipeHandle[0], buffer, 60000);
      pipeContent += buffer;
   }
#else
   int buf_read;
   char ch;
   while (true) {
      buf_read = read(pipeHandle[0], &ch, 1);
      if (buf_read == 1) {
         pipeContent += ch;
      } else
         break;
   }
#endif
}

void JupyROOTExecutorHandler::Poll()
{
   PollImpl(stdout, fStdout_pipe, fStdoutpipe);
   PollImpl(stderr, fStderr_pipe, fStderrpipe);
}

static void InitCaptureImpl(int &savedStdStream, int *pipeHandle, int FILENO)
{
   savedStdStream = dup(FILENO);
   if (pipe(pipeHandle) != 0) {
      return;
   }
#ifndef _MSC_VER
   long flags_stdout = fcntl(pipeHandle[0], F_GETFL);
   if (flags_stdout == -1)
      return;
   flags_stdout |= O_NONBLOCK;
   fcntl(pipeHandle[0], F_SETFL, flags_stdout);
   fcntl(pipeHandle[0], F_SETPIPE_SZ, MAX_PIPE_SIZE);
#endif
   dup2(pipeHandle[1], FILENO);
}

void JupyROOTExecutorHandler::InitCapture()
{
   if (!fCapturing) {
      InitCaptureImpl(fSaved_stdout, fStdout_pipe, STDOUT_FILENO);
      InitCaptureImpl(fSaved_stderr, fStderr_pipe, STDERR_FILENO);
      fCapturing = true;
   }
}

void JupyROOTExecutorHandler::EndCapture()
{
   if (fCapturing) {
      Poll();
      dup2(fSaved_stdout, STDOUT_FILENO);
      dup2(fSaved_stderr, STDERR_FILENO);
      close(fSaved_stdout);
      close(fSaved_stderr);
      close(fStdout_pipe[0]);
      close(fStdout_pipe[1]);
      close(fStderr_pipe[0]);
      close(fStderr_pipe[1]);
      fCapturing = false;
   }
}

void JupyROOTExecutorHandler::Clear()
{
   fStdoutpipe = "";
   fStderrpipe = "";
}

std::string &JupyROOTExecutorHandler::GetStdout()
{
   return fStdoutpipe;
}

std::string &JupyROOTExecutorHandler::GetStderr()
{
   return fStderrpipe;
}

JupyROOTExecutorHandler *JupyROOTExecutorHandler_ptr = nullptr;

bool JupyROOTExecutorImpl(const char *code)
{
   auto status = false;
   try {
      auto err = TInterpreter::kNoError;
      if (gInterpreter->ProcessLine(code, &err)) {
         status = true;
      }

      if (err == TInterpreter::kProcessing) {
         gInterpreter->ProcessLine(".@");
         gInterpreter->ProcessLine("cerr << \"Unbalanced braces. This cell was not processed.\" << endl;");
      }
   } catch (...) {
      status = true;
   }

   return status;
}

bool JupyROOTDeclarerImpl(const char *code)
{
   auto status = false;
   try {
      if (gInterpreter->Declare(code)) {
         status = true;
      }
   } catch (...) {
      status = true;
   }
   return status;
}

PyObject *JupyROOTExecutor(PyObject * /*self*/, PyObject *args)
{
   const char *code;
   if (!PyArg_ParseTuple(args, "s", &code))
      return NULL;

   auto res = JupyROOTExecutorImpl(code);

   return PyLong_FromLong(res);
}

PyObject *JupyROOTDeclarer(PyObject * /*self*/, PyObject *args)
{
   const char *code;
   if (!PyArg_ParseTuple(args, "s", &code))
      return NULL;

   auto res = JupyROOTDeclarerImpl(code);

   return PyLong_FromLong(res);
}

PyObject *JupyROOTExecutorHandler_Clear(PyObject * /*self*/, PyObject * /*args*/)
{
   JupyROOTExecutorHandler_ptr->Clear();
   Py_RETURN_NONE;
}

PyObject *JupyROOTExecutorHandler_Ctor(PyObject * /*self*/, PyObject * /*args*/)
{
   if (!JupyROOTExecutorHandler_ptr) {
      JupyROOTExecutorHandler_ptr = new JupyROOTExecutorHandler();
      // Fixes for ROOT-7999
      gInterpreter->ProcessLine("SetErrorHandler((ErrorHandlerFunc_t)&DefaultErrorHandler);");
   }

   Py_RETURN_NONE;
}

PyObject *JupyROOTExecutorHandler_Poll(PyObject * /*self*/, PyObject * /*args*/)
{
   JupyROOTExecutorHandler_ptr->Poll();
   Py_RETURN_NONE;
}

PyObject *JupyROOTExecutorHandler_EndCapture(PyObject * /*self*/, PyObject * /*args*/)
{
   JupyROOTExecutorHandler_ptr->EndCapture();
   Py_RETURN_NONE;
}

PyObject *JupyROOTExecutorHandler_InitCapture(PyObject * /*self*/, PyObject * /*args*/)
{
   JupyROOTExecutorHandler_ptr->InitCapture();
   Py_RETURN_NONE;
}

PyObject *JupyROOTExecutorHandler_GetStdout(PyObject * /*self*/, PyObject * /*args*/)
{
   auto out = JupyROOTExecutorHandler_ptr->GetStdout().c_str();
   return PyUnicode_FromString(out);
}

PyObject *JupyROOTExecutorHandler_GetStderr(PyObject * /*self*/, PyObject * /*args*/)
{
   auto err = JupyROOTExecutorHandler_ptr->GetStderr().c_str();
   return PyUnicode_FromString(err);
}

PyObject *JupyROOTExecutorHandler_Dtor(PyObject * /*self*/, PyObject * /*args*/)
{
   if (!JupyROOTExecutorHandler_ptr)
      Py_RETURN_NONE;

   delete JupyROOTExecutorHandler_ptr;
   JupyROOTExecutorHandler_ptr = nullptr;

   Py_RETURN_NONE;
}
