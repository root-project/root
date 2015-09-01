
from IPython.core.magic import (Magics, magics_class, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import utils
from hashlib import sha1


def _codeToFilename(code):
    '''Convert code to a unique file name

    >>> _codeToFilename("int f(i){return i*i;}")
    'dbf7e731.C'
    '''
    fileNameBase = sha1(code).hexdigest()[0:8]
    return fileNameBase + ".C"

def _dumpToUniqueFile(code):
    '''Dump code to file whose name is unique

    >>> _codeToFilename("int f(i){return i*i;}")
    'dbf7e731.C'
    '''
    fileName = _codeToFilename(code)
    ofile = open (fileName,'w')
    ofile.write(code)
    ofile.close()
    return fileName


@magics_class
class CppMagics(Magics):

    @cell_magic
    @magic_arguments()
    @argument('-a', '--aclic', action="store_true", help='Compile code with ACLiC.')
    @argument('-d', '--declare', action="store_true", help='Declare functions and/or classes.')
    def cpp(self, line, cell):
        args = parse_argstring(self.cpp, line)
        if args.aclic:
            fileName = _dumpToUniqueFile(cell)
            utils.invokeAclic(fileName) 
        elif args.declare:
            utils.declareCppCode(cell) 
        else:
            utils.processCppCode(cell)

def load_ipython_extension(ipython):
    ipython.register_magics(CppMagics)
