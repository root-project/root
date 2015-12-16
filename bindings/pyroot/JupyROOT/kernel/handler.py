# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2015, ROOT Team.
#  Authors: Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#  website: http://oproject.org/ROOT+Jupyter+Kernel (information only for ROOT kernel)
#  Distributed under the terms of the Modified LGPLv3 License.
#
#  The full license is in the file COPYING.rst, distributed with this software.
#-----------------------------------------------------------------------------
from ROOT import gInterpreter

#required class to capture i/o
#NOTE: this class required a system to flush in a fork
#if pipe is 1M 1024*1024 size then hung up

CPPIOClass ='#ifndef F_LINUX_SPECIFIC_BASE\n'
CPPIOClass +='#define F_LINUX_SPECIFIC_BASE       1024\n'
CPPIOClass +='#endif\n'
CPPIOClass +='#ifndef F_SETPIPE_SZ\n'
CPPIOClass +='#define F_SETPIPE_SZ    (F_LINUX_SPECIFIC_BASE + 7)\n'
CPPIOClass +='#endif\n'
CPPIOClass +='class JupyROOTExecutorHandler{'
CPPIOClass +='private:'
CPPIOClass +='  bool capturing;'
CPPIOClass +='  long MAX_PIPE_SIZE=1048575;'#size of the pipi to capture stdout/stderr
#CPPIOClass +='  //this values are to capture stdout, stderr'
CPPIOClass +='  std::string    stdoutpipe;'
CPPIOClass +='  std::string    stderrpipe;'
CPPIOClass +='  int stdout_pipe[2];'
CPPIOClass +='  int stderr_pipe[2];'
CPPIOClass +='  int saved_stderr;'
CPPIOClass +='  int saved_stdout;'
CPPIOClass +='public:'
CPPIOClass +='JupyROOTExecutorHandler(){'
CPPIOClass +='  capturing=false;'
CPPIOClass +='}'
CPPIOClass +='void InitCapture()'
CPPIOClass +='{'
CPPIOClass +='  if(!capturing)'
CPPIOClass +='  {'
CPPIOClass +='          saved_stdout = dup(STDOUT_FILENO);'
CPPIOClass +='          saved_stderr = dup(STDERR_FILENO);'
CPPIOClass +='          if( pipe(stdout_pipe) != 0 ) {    '
CPPIOClass +='                  return;'
CPPIOClass +='          }'
CPPIOClass +='          if( pipe(stderr_pipe) != 0 ) {    '
CPPIOClass +='                  return;'
CPPIOClass +='          }'
CPPIOClass +='          long flags_stdout = fcntl(stdout_pipe[0], F_GETFL);'
CPPIOClass +='          flags_stdout |= O_NONBLOCK;'
CPPIOClass +='          fcntl(stdout_pipe[0], F_SETFL, flags_stdout);'
CPPIOClass +='          fcntl(stdout_pipe[0], F_SETPIPE_SZ, MAX_PIPE_SIZE);'#here PIPE size to capture stdout
CPPIOClass +='          long flags_stderr = fcntl(stderr_pipe[0], F_GETFL);'
CPPIOClass +='          flags_stderr |= O_NONBLOCK;'
CPPIOClass +='          fcntl(stderr_pipe[0], F_SETFL, flags_stderr);'
CPPIOClass +='          fcntl(stderr_pipe[0], F_SETPIPE_SZ, MAX_PIPE_SIZE);'#here PIPE size to capture stderr
CPPIOClass +='          dup2(stdout_pipe[1], STDOUT_FILENO);   '
CPPIOClass +='          close(stdout_pipe[1]);'
CPPIOClass +='          dup2(stderr_pipe[1], STDERR_FILENO);   '
CPPIOClass +='          close(stderr_pipe[1]);'
CPPIOClass +='          capturing = true;'
CPPIOClass +='  }'
CPPIOClass +='}'
CPPIOClass +='void EndCapture()'
CPPIOClass +='{'
CPPIOClass +='  if(capturing)'
CPPIOClass +='  {'
CPPIOClass +='          int buf_readed;'
CPPIOClass +='          char ch;'
CPPIOClass +='          fflush(stdout);'
CPPIOClass +='          while(true)'
CPPIOClass +='          {'
CPPIOClass +='                   buf_readed = read(stdout_pipe[0], &ch, 1);'
CPPIOClass +='                   if(buf_readed==1) stdoutpipe += ch;'
CPPIOClass +='                   else break;'
CPPIOClass +='          }'
CPPIOClass +='          fflush(stderr);'
CPPIOClass +='          while(true)'
CPPIOClass +='          {'
CPPIOClass +='                   buf_readed = read(stderr_pipe[0], &ch, 1);'
CPPIOClass +='                   if(buf_readed==1) stderrpipe += ch;'
CPPIOClass +='                   else break;'
CPPIOClass +='          }'
CPPIOClass +='          dup2(saved_stdout, STDOUT_FILENO);  '
CPPIOClass +='          dup2(saved_stderr, STDERR_FILENO);  '
CPPIOClass +='          capturing = false;'
CPPIOClass +='  }'
CPPIOClass +='}'
CPPIOClass +='std::string getStdout(){'
CPPIOClass +='  return stdoutpipe;'
CPPIOClass +='}'
CPPIOClass +='std::string getStderr(){'
CPPIOClass +='  return stderrpipe;'
CPPIOClass +='}'
CPPIOClass +='void clear(){'
CPPIOClass +='  stdoutpipe="";'
CPPIOClass +='  stderrpipe="";'
CPPIOClass +='}'
CPPIOClass +='};'


CPPIOFunctionTemplate ='Bool_t %s(const char* code)'
CPPIOFunctionTemplate +='{'
CPPIOFunctionTemplate +='  Bool_t status=kFALSE;'
CPPIOFunctionTemplate +='  TRY {'
CPPIOFunctionTemplate +='    if(gInterpreter->%s(code))'
CPPIOFunctionTemplate +='    {'
CPPIOFunctionTemplate +='      status=kTRUE;'
CPPIOFunctionTemplate +='    }'
CPPIOFunctionTemplate +='  } CATCH(excode) {'
CPPIOFunctionTemplate +='    status=kTRUE;'
CPPIOFunctionTemplate +='  } ENDTRY;'
CPPIOFunctionTemplate +='  return status;'
CPPIOFunctionTemplate +='}'

CPPIOFunctions = CPPIOFunctionTemplate %("JupyROOTExecutor", "ProcessLine")
CPPIOFunctions += CPPIOFunctionTemplate %("JupyROOTDeclarer", "Declare")

def _LoadClass():
    gInterpreter.Declare(CPPIOClass)

def _LoadFunctions():
    gInterpreter.Declare(CPPIOFunctions)

def LoadHandlers():
    _LoadClass()
    _LoadFunctions()
