# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2015, ROOT Team.
#  Authors: Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#           Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#           Enric Tejedor enric.tejedor.saavedra@cern.ch> CERN
#  website: http://oproject.org/ROOT+Jupyter+Kernel (information only for ROOT kernel)
#  Distributed under the terms of the Modified LGPLv3 License.
#
#  The full license is in the file COPYING.rst, distributed with this software.
#-----------------------------------------------------------------------------
from metakernel import Magic, option

import sys

#NOTE:actually JupyROOT is not capturing the error on %%cpp -d if the function is wrong 
class CppMagics(Magic):
    def __init__(self, kernel):
        super(CppMagics, self).__init__(kernel)
    @option(
        '-a', '--aclic', action='store', default="default", help='Compile code with ACLiC.'
    )
    @option(
        '-d', '--declare', action='store', default=None, help='Declare functions and/or classes.'
    )
    def cell_cpp(self, args):
        '''Executes the content of the cell as C++ code.'''
        if self.code.strip():
             self.kernel.ioHandler.clear()
             self.kernel.ioHandler.InitCapture()
             
             if args=='-a':
                 self.kernel.ACLiC(self.code)
             elif args=='-d':
                 self.kernel.Declarer(str(self.code))
             else:
                 self.kernel.Executor(str(self.code))
             self.kernel.ioHandler.EndCapture()
             std_out = self.kernel.ioHandler.getStdout()
             std_err = self.kernel.ioHandler.getStderr()
             if std_out != "":
                stream_content_stdout = {'name': 'stdout', 'text': std_out}
                self.kernel.send_response(self.kernel.iopub_socket, 'stream', stream_content_stdout)
             if std_err != "":
                stream_content_stderr = {'name': 'stderr', 'text': std_err}
                self.kernel.send_response(self.kernel.iopub_socket, 'stream', stream_content_stderr)
            
        self.evaluate = False
        
def register_magics(kernel):
    kernel.register_magics(CppMagics)
    
