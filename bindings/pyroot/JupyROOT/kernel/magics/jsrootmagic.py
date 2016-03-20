# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2016, ROOT Team.
#  Authors: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#-----------------------------------------------------------------------------

from JupyROOT.utils import enableJSVis, disableJSVis, enableJSVisDebug

from metakernel import Magic, option

class JSRootMagics(Magic):
    def __init__(self, kernel):
        super(JSRootMagics, self).__init__(kernel)
    @option('arg', default="on", help='Enable or disable JavaScript visualisation. Possible values: on (default), off')

    def cell_jsroot(self, args):
        '''Change the visualisation of plots from images to interactive JavaScript objects.'''
        if args == 'on' or args == '':
           enableJSVis()
        elif args == 'off':
           disableJSVis()
        elif args == 'debug':
           enableJSVisDebug()

def register_magics(kernel):
    kernel.register_magics(JSRootMagics)

