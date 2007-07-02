# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import sys, os, gendict, selclass, gencapa, genrootmap, string, getopt

class genreflex:
#----------------------------------------------------------------------------------
  def __init__(self):
    self.files           = []
    self.output          = None
    self.outputDir       = None
    self.outputFile      = None
    self.capabilities    = None
    self.rootmap         = None
    self.rootmaplib      = None
    self.select          = None
    self.cppopt          = ''
    self.deep            = False
    self.opts            = {}
    self.gccxmlpath      = None
    self.gccxmlopt       = ''
    self.gccxmlvers      = '0.6.0_patch3'
    self.selector        = None
    self.gccxml          = ''
    self.quiet           = False
#----------------------------------------------------------------------------------
  def usage(self, status = 1) :
    print 'Usage:'
    print '  genreflex headerfile1.h [headerfile2.h] [options] [preprocesor options]'
    print 'Try "genreflex --help" for more information.'
    sys.exit(status)
#----------------------------------------------------------------------------------
  def help(self) :
    print """Generates the LCG dictionary file for each header file\n
    Usage:
      genreflex headerfile1.h [headerfile2.h] [options] [preprocesor options]\n    
    Options:
      -s <file>, --selection_file=<file>
         Class selection file to specify for which classes the dictionary
         will be generated
         Format (XML):
           <lcgdict>
           [<selection>]
             <class [name="classname"] [pattern="wildname"] 
                    [file_name="filename"] [file_pattern="wildname"] 
                    [id="xxxx"] [type="vector"]/>
             <class name="classname" >
               <field name="m_transient" transient="true"/>
               <field name="m_anothertransient" transient="true"/>
               <properties prop1="value1" [prop2="value2"]/>
             </class>
             <function [name="funcname"] [pattern="wildname"] />
             <enum [name="enumname"] [patter="wildname"] />
             <variable [name="varname"] [patter="wildname"] />
           [</selection>]
           <exclusion>
             <class [name="classname"] [pattern="wildname"] />
               <method name="unwanted" />
             </class>
           ...
           </lcgdict>\n
      -o <file>, --output <file>
         Output file name. If an existing directory is specified instead of a file,
         then a filename will be build using the name of the input file and will
         be placed in the given directory. <headerfile>_rflx.cpp \n
      --pool
         Generate minimal dictionary required for POOL persistency\n
      --deep
         Generate dictionary for all dependend classes\n
      --split  (OBSOLETE)
         Generate separate file for stub functions. Option sometimes needed on Windows.\n
      --reflex  (OBSOLETE)
         Generate Reflex dictionaries.\n
      --comments
         Add end-of-line comments in data and functions members as a property called "comment" \n
      --iocomments
         Add end-of-line comments in data and functions members as a property called "comment", but only for comments relevant for ROOT I/O \n
      --no_membertypedefs
         Disable the definition of class member typedefs \n
      --no_templatetypedefs
         Disable resolving of typedefs in template parameters for selection names. E.g. std::vector<MYINT>.\n
      --fail_on_warnings
         The genreflex command fails (retuns the value 1) if any warning message is issued \n
      --gccxmlpath=<path>
         Path path where the gccxml tool is installed.
         If not defined the standard PATH environ variable is used\n
      --gccxmlopt=<gccxmlopt>
         Options to be passed directly to gccxml\n
      -c <file>, --capabilities=<file>
         Generate the capabilities file to be used by the SEAL Plugin Manager. This file
         lists the names of all classes for which the reflection is formation is provided.\n
      --rootmap=<file>
         Generate the rootmap file to be used by ROOT/CINT. This file lists the names of 
         all classes for which the reflection is formation is provided.\n
      --rootmap-lib=<library>
         Library name for the rootmap file.\n
      --debug
         Print extra debug information while processing. Keep intermediate files\n
      --quiet
         No not print informational messages\n
      -h, --help
         Print this help\n
     """ 
    sys.exit()
#----------------------------------------------------------------------------------
  def parse_args(self, argv = sys.argv) :
    options = []
    #----Ontain the list of files to process------------
    for a in argv[1:] :
      if a[0] != '-' :
        self.files.append(a)
      else :
        options = argv[argv.index(a):]
        break
    #----Process options--------------------------------
    try:
      opts, args = getopt.getopt(options, 'ho:s:c:I:U:D:PC', \
      ['help','debug=', 'output=','selection_file=','pool','deep','gccxmlpath=',
       'capabilities=','rootmap=','rootmap-lib=','comments','iocomments','no_membertypedefs',
       'fail_on_warnings', 'quiet', 'gccxmlopt=', 'reflex', 'split','no_templatetypedefs'])
    except getopt.GetoptError, e:
      print "--->> genreflex: ERROR:",e
      self.usage(2)
    self.output = '.'
    self.select = None
    self.gccxmlpath = None
    self.cppopt = ''
    self.pool   = 0
    for o, a in opts:
      if o in ('-h', '--help'):
        self.help()
      if o in ('--no_templatetypedefs',):
        self.opts['resolvettd'] = 0
      if o in ('--debug',):
        self.opts['debug'] = a
      if o in ('-o', '--output'):
        self.output = a
      if o in ('-s', '--selection_file'):
        self.select = a
      if o in ('--pool',):
        self.opts['pool'] = True
      if o in ('--deep',):
        self.deep = True
      if o in ('--split',):
        print '--->> genreflex: WARNING: --split option is obsolete'
      if o in ('--reflex',):
        print '--->> genreflex: WARNING: --reflex option is obsolete'
      if o in ('--comments',):
        self.opts['comments'] = True
      if o in ('--iocomments',):
        self.opts['iocomments'] = True
      if o in ('--no_membertypedefs',):
        self.opts['no_membertypedefs'] = True
      if o in ('--fail_on_warnings',):
        self.opts['fail_on_warnings'] = True
      if o in ('--quiet',):
        self.opts['quiet'] = True
        self.quiet = True
      if o in ('--gccxmlpath',):
        self.gccxmlpath = a
      if o in ('--gccxmlopt',):
        self.gccxmlopt += a +' '
      if o in ('-c', '--capabilities'):
        self.capabilities = a
      if o in ('--rootmap',):
        self.rootmap = a
      if o in ('--rootmap-lib',):
        self.rootmaplib = a
      if o in ('-I', '-U', '-D', '-P', '-C') :
        self.cppopt += o + a +' '
#----------------------------------------------------------------------------------
  def check_files_dirs(self):
    #---Check existance of input files--------------------
    if self.files :
      for f in self.files :
        if not os.path.exists(f) : 
          print '--->> genreflex: ERROR: C++ file "' + f + '" not found'
          self.usage()
    else :
      print '--->> genreflex: ERROR: No input file specified'
      self.usage()
    #---Check existance of output directory----------------
    if os.path.isdir(self.output) :
      self.outputDir  = self.output
      self.outputFile = None
    else :
      self.outputDir, self.outputFile = os.path.split(self.output)
    if self.outputDir and not os.path.isdir(self.outputDir) :
      print '--->> genreflex: ERROR: Output directory ', self.outputDir, ' not found'
      self.usage()
    #---Hande selection class file-------------------------
    classes = []
    if self.select :
      if not os.path.exists(self.select) :
        print '--->> genreflex: ERROR: Class selection file "' + self.select + '" not found'
        self.usage()
      for l in open(self.select).readlines() : classes.append(l[:-1])
    #----------GCCXML command------------------------------
    if not self.gccxmlpath:
      try:
        import gccxmlpath
        self.gccxmlpath = gccxmlpath.gccxmlpath
      except:
        pass
    if self.gccxmlpath :
      if sys.platform == 'win32' :
        self.gccxml = self.gccxmlpath + os.sep + 'gccxml.exe'
      else :
        self.gccxml = self.gccxmlpath + os.sep + 'gccxml'
      if not os.path.isfile(self.gccxml) :
        print '--->> genreflex: ERROR: Path to gccxml given, but no executable found at', self.gccxml
    elif self.which('gccxml') :
      self.gccxml = 'gccxml'
      print '--->> genreflex: INFO: No explicit path to gccxml given. Found gccxml at', self.which('gccxml')
    else :
      if sys.platform == 'win32' :
        self.gccxml = r'\\cern.ch\dfs\Experiments\sw\lcg\external\gccxml\0.6.0_patch3\win32_vc71\bin\gccxml'
      else :
        self.gccxml = '/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc3_ia32_gcc323/bin/gccxml'
      print '--->> genreflex: INFO: No gccxml executable found, using fallback location at', self.gccxml
    #---------------Open selection file-------------------
    try :
      if self.select : self.selector = selclass.selClass(self.select,parse=1)
      else           : self.selector = None
    except :
      sys.exit(1)

#----------------------------------------------------------------------------------
  def genGccxmlInfo(self):
    s = ''
    (inp,out,err) = os.popen3(self.gccxml + ' --print')
    serr = err.read()
    sout = out.read()
    if serr :
      print '--->> genreflex: WARNING: Could not invoke %s --print' % self.gccxml
      print '--->> genreflex: WARNING: %s' % serr
      return s
    gccxmlv = sout.split('\n')[0].split()[-1]
    # For 0.6.0 we can't do much because we have not put in a patch info into the version string 
    if gccxmlv != '0.6.0' and gccxmlv != self.gccxmlvers :
      print '--->> genreflex: WARNING: gccxml versions differ. Used version: %s. Recommended version: %s. ' % ( gccxmlv, self.gccxmlvers)
      print '--->> genreflex: WARNING: gccxml binary used: %s' % ( self.gccxml )
    s += sout    
    compiler = ''
    for l in sout.split('\n'):
      ll = l.split('"')
      if ll[0].find('GCCXML_COMPILER') != -1 :
        compiler = ll[1]
        break
    bcomp = os.path.basename(compiler)
    vopt = ''
    if   bcomp in ('msvc7','msvc71')  : return s
    elif bcomp in ('gcc','g++','c++') : vopt = '--version'
    elif bcomp in ('cl.exe','cl')     : vopt = '' # there is no option to print only the version with cl
    else :
      print '--->> genreflex: WARNING: While trying to retrieve compiler version, found unknown compiler %s' % compiler
      return s
    (inp,out,err) = os.popen3('%s %s'%(compiler,vopt))
    serr = err.read()
    if serr :
      print '--->> genreflex: WARNING: While trying to retrieve compiler information. Cannot invoke %s %s' % (compiler,vopt)
      print '--->> genreflex: WARNING: %s' % serr
      return s
    s += '\nCompiler info:\n' + out.read()
    return s
#----------------------------------------------------------------------------------
  def process_files(self):
    total_warnings = 0
    file_extension = '_rflx.cpp'
    #----------Loop oover all the input files--------------
    for source in self.files :
      path, fullname = os.path.split(source)
      name = fullname[:fullname.find('.')]
      xmlfile = os.path.join(self.outputDir,name+'.xml')
      if( self.outputFile ) :
        dicfile = os.path.join(self.outputDir,self.outputFile)
      else :
        dicfile = os.path.join(self.outputDir,name+file_extension)
      #---------------Parse the header file with GCC_XML
      cmd  = '%s %s %s -fxml=%s %s -D__REFLEX__' %(self.gccxml, self.gccxmlopt, source, xmlfile, self.cppopt)
      if not self.quiet : print '--->> genreflex: INFO: Parsing file %s with GCC_XML' % source,
      status = os.system(cmd)
      if status :
        print '\n--->> genreflex: ERROR: processing file with gccxml. genreflex command failed.'
        sys.exit(1)
      else: 
        if not self.quiet : print 'OK'
      gccxmlinfo = self.genGccxmlInfo()
     #---------------Generate the dictionary---------------
      if not self.quiet : print '--->> genreflex: INFO: Generating Reflex Dictionary'
      dg = gendict.genDictionary(source, self.opts)
      dg.parse(xmlfile)
      classes   = dg.selclasses(self.selector, self.deep)
      functions = dg.selfunctions(self.selector)
      enums     = dg.selenums(self.selector)
      variables = dg.selvariables(self.selector)
      cnames, warnings, errors = dg.generate(dicfile, classes, functions, enums, variables, gccxmlinfo )
      total_warnings += warnings
    #------------Produce Seal Capabilities source file------
      if self.capabilities :
        if os.path.isdir(self.capabilities) :
          capfile = os.path.join(self.capabilities, 'capabilities.cpp')
        else :
          capfile = os.path.join(self.outputDir, self.capabilities)
        gencapa.genCapabilities(capfile, name,  cnames)
    #------------Produce rootmap file-----------------------
      if self.rootmap :
        if os.path.isdir(self.rootmap) :
          mapfile = os.path.join(self.rootmap, 'rootmap')
        else :
          mapfile = os.path.join(self.outputDir, self.rootmap)
        if not self.rootmaplib :  self.rootmaplib = 'lib'+name+'.so'
        genrootmap.genRootMap(mapfile, name,  self.rootmaplib, cnames, classes)
    #------------Report unused class selections in selection
    if self.selector : 
      warnings += self.selector.reportUnusedClasses()
    #------------Delete intermediate files------------------
    if 'debug' not in self.opts :
       os.remove(xmlfile)
    #------------Exit with status if warnings --------------
    if warnings and self.opts.get('fail_on_warnings',False) : 
      print '--->> genreflex: ERROR: Exiting with error due to %d warnings ( --fail_on_warnings enabled )' % warnings
      sys.exit(1)
#---------------------------------------------------------------------
  def which(self, name) :
    if 'PATH' in os.environ :
      if sys.platform == 'win32' : name += '.exe'
      for p in os.environ['PATH'].split(os.pathsep) :
        path = os.path.join(p,name)
        if os.path.exists(path) : return path
    return None
#---------------------------------------------------------------------
if __name__ == "__main__":
  l = genreflex()
  l.parse_args()
  l.check_files_dirs()
  l.process_files()
