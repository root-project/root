# Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import sys, os, gendict, selclass, gencapa, string, getopt

class genreflex:
#----------------------------------------------------------------------------------
  def __init__(self):
    self.files           = []
    self.output          = None
    self.outputDir       = None
    self.outputFile      = None
    self.capabilities    = None
    self.select          = None
    self.cppopt          = ''
    self.deep            = False
    self.opts            = {}
    self.gccxmlpath      = None
    self.selector        = None
    self.gccxml          = ''
    self.quiet           = False
    try:
      import gccxmlpath
      self.gccxmlpath = gccxmlpath.gccxmlpath
    except:
      pass
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
         be placed in the given directory. <headerfile>_dict.cpp \n
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
      --no_membertypedefs
         Disable the definition of class member typedefs \n
      --fail_on_warnings
         The lcgdict command fails (retuns the value 1) if any warning message is issued \n
      --gccxmlpath=<path>
         Path path where the gccxml tool is installed.
         If not defined the standard PATH environ variable is used\n
      -c <file>, --capabilities=<file>
         Generate the capabilities file to be used by the SEAL Plugin Manager. This file
         lists the names of all classes for which the reflection is formation is provided.\n
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
       'capabilities=','comments','no_membertypedefs', 'fail_on_warnings', 'quiet',
       'reflex', 'split'])
    except getopt.GetoptError:
      self.usage(2)
    self.output = '.'
    self.select = None
    self.gccxmlpath = None
    self.cppopt = ''
    self.pool   = 0
    for o, a in opts:
      if o in ('-h', '--help'):
        self.help()
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
        print '-->WARNING: --split option is obsoleted'
      if o in ('--reflex',):
        print '-->WARNING: --reflex option is obsoleted'
      if o in ('--comments',):
        self.opts['comments'] = True
      if o in ('--no_membertypedefs',):
        self.opts['no_membertypedefs'] = True
      if o in ('--fail_on_warnings',):
        self.opts['fail_on_warnings'] = True
      if o in ('--quiet',):
        self.opts['quiet'] = True
        self.quiet = True
      if o in ('--gccxmlpath',):
        self.gccxmlpath = a
      if o in ('-c', '--capabilities'):
        self.capabilities = a
      if o in ('-I', '-U', '-D', '-P', '-C') :
        self.cppopt += o + a +' '
#----------------------------------------------------------------------------------
  def check_files_dirs(self):
    #---Check existance of input files--------------------
    if self.files :
      for f in self.files :
        if not os.path.exists(f) : 
          print '--->>ERROR: C++ file "' + f + '" not found'
          self.usage()
    else :
      print '--->>ERROR: No input file specified'
      self.usage()
    #---Check existance of output directory----------------
    if os.path.isdir(self.output) :
      self.outputDir  = self.output
      self.outputFile = None
    else :
      self.outputDir, self.outputFile = os.path.split(self.output)
    if self.outputDir and not os.path.isdir(self.outputDir) :
      print '--->>ERROR: Output directory ', self.outputDir, ' not found'
      self.usage()
    #---Hande selection class file-------------------------
    classes = []
    if self.select :
      if not os.path.exists(self.select) :
        print '--->>ERROR: Class selection file "' + self.select + '" not found'
        self.usage()
      for l in open(self.select).readlines() : classes.append(l[:-1])
    #----------GCCXML command------------------------------
    if self.gccxmlpath :
      self.gccxml = self.gccxmlpath + os.sep + 'gccxml'
    elif self.which('gccxml') :
      self.gccxml = 'gccxml'
    else :
      if sys.platform == 'win32' :
        self.gccxml = r'\\cern.ch\dfs\Experiments\sw\lcg\external\gccxml\0.6.0_patch3\win32_vc71\gccxml'
      else :
        self.gccxml = '/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc3_ia32_gcc323/bin/gccxml'
    #---------------Open selection file-------------------
    try :
      if self.select : self.selector = selclass.selClass(self.select,parse=1)
      else           : self.selector = None
    except :
      sys.exit(1)
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
      cmd  = '%s %s -fxml=%s %s -D__REFLEX__' %(self.gccxml, source, xmlfile, self.cppopt)
      if not self.quiet : print 'Parsing file %s with GCC_XML' % source,
      status = os.system(cmd)
      if status :
        print 'Error processing file with gccxml. Lcgdict command failed.'
        sys.exit(1)
      else: 
        if not self.quiet : print 'OK'
     #---------------Generate the dictionary---------------
      if not self.quiet : print 'Generating Reflex Dictionary'
      dg = gendict.genDictionary(source, self.opts)
      dg.parse(xmlfile)
      classes   = dg.selclasses(self.selector, self.deep)
      functions = dg.selfunctions(self.selector)
      cnames, warnings, errors = dg.generate(dicfile, classes, functions )
      total_warnings += warnings
    #------------Produce Seal Capabilities source file------
      if self.capabilities :
        if os.path.isdir(self.capabilities) :
          capfile = os.path.join(self.capabilities, 'capabilities.cpp')
        else :
          capfile = os.path.join(self.outputDir, self.capabilities)
        gencapa.genCapabilities(capfile, name,  cnames)
    #------------Report unused class selections in selection
    if self.selector : 
      warnings += self.selector.reportUnusedClasses()
    #------------Delete intermediate files------------------
    if 'debug' not in self.opts :
       os.remove(xmlfile)
    #------------Exit with status if warnings --------------
    if warnings and self.opts.get('fail_on_warnings',False) : 
      print '--->>ERROR: Exiting with error due to %d warnings ( --fail_on_warnings enabled )' % warnings
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
