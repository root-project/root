import sys, os, getopt, genreflex


class genreflex_rootcint:

  def __init__(self):
    self.dict_filename = 'G__genreflex_rootcint.cxx'
    self.dict_header = 'G__genreflex_rootcint.h'
    self.header_files = []
    self.pragmas = []
    self.sel_classes = []
    self.sel_classesT = []
    self.gccxml_ppopts = []
    # this list is quite unfortunate but at the moment gccxml (0.6.0_patch3) will
    # need instances of templated classes to generate information. This will of
    # course not work for abstract ones. And there is no way to retrieve this info
    # from the Linkdef information only, also needed for templated classes without
    # default constructor
    self.abstractTemplateClasses = ['TMatrixTBase','TMatrixTLazy','TMatrixTSymLazy','ROOT::Math::WrappedFunction']
    
  def usage(self):
    pass

  def help(self):
    pass

  def test_gccxml(self):
    gccxmlfound = 1
    if sys.platform == 'win32':
      gccxmlbin = 'gccxml.exe'
    else:
      gccxmlbin = 'gccxml'
    try:
      import gccxmlpath
      if os.path.isfile(gccxmlpath.gccxmlpath+os.sep+gccxmlbin) : gccxmlfound = 0
    except:
      pass
    if sys.platform == 'win32' :
      gccxmlbin = r'\\cern.ch\dfs\Experiments\sw\lcg\external\gccxml\0.6.0_patch3\win32_vc71\bin\gccxml'
    else :
      gccxmlbin = '/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc3_ia32_gcc323/bin/gccxml'
    if os.path.isfile(gccxmlbin) : gccxmlfound = 0
    return gccxmlfound

  def parse_args(self):
    options = sys.argv[1:]
    options2 = []
    # FIXME: removing again TROOT.h and TMemberInspector.h from the options as they are
    # passed in by rootcint, but before the options (gnu style doesn't like it)
    # --> fix is to remove the two headers from the invocation in rootcint.cxx
    for o in options :
      if o.find('TROOT.h') == -1 and o.find('TMemberInspector.h') == -1 : options2.append(o)
    try:
      optlist,args = getopt.getopt(options2,'cv:lf:pg:r:D:I:',['gccxml-available'])
    except getopt.GetoptError, e:
      print sys.argv[0], ': ERROR:', e
      self.usage()
    for o,a in optlist:
      if o in ('--gccxml-available',) : sys.exit(self.test_gccxml())
      if o in ('-c','-v','-l','-p','-g','-r') : pass
      if o in ('-D','-I') :
        self.gccxml_ppopts.append(o+a)
      if o in ('-f',) :
        self.dict_filename = a
    self.header_files = args

  def parse_headers(self):
    for f in self.header_files:
      if os.path.isfile(f):
        fh = open(f)
        for line in fh.readlines():
          if line.find('#pragma') != -1:
            self.pragmas.append(line)
        fh.close()
      else:
        print '%s: WARNING: %s is not a file, skipping' % (sys.argv[0], f)

  def parse_pragmas(self):
    for p in self.pragmas:
      pl = p.split()
      
      if pl[0] == '#pragma':
        if pl[1] == 'link':
          if pl[2] == 'C++':
            if pl[3] == 'class':
              rl = ' '.join(pl[4:])
              rl = rl[:rl.find(';')]
              while rl[-1] in ['+','-','!',' '] : rl = rl[:-1]
              if rl.find('<') != -1 : self.sel_classesT.append(rl)
              else                  : self.sel_classes.append(rl)

  def gen_temp_header(self):
    self.dict_header = self.dict_filename.split('.')[0]+'.h'
    hh = open(self.dict_header,'w')

    hh.write("#include <stddef.h>\n")
    hh.write("#include <stdio.h>\n")
    hh.write("#include <stdlib.h>\n")
    hh.write("#include <math.h>\n")
    hh.write("#include <string.h>\n")

    hh.write('using namespace std;\n\n')

    hh.write('#include "TROOT.h"\n')
    hh.write('#include "TMemberInspector.h"\n')
    hh.write('#include "Rtypes.h"\n')
    
    for f in self.header_files:
      hh.write('#include "%s"\n' % f)

    hh.write('\n\nnamespace ROOT {\n  namespace Reflex {\n    namespace Selection {\n')
    for c in self.sel_classes:
      # let's assume there are no namespaces in ROOT
      # inner classes are automatically selected by genreflex
      if c.find('::') == -1: 
        hh.write('      class %s {};\n' % c)
    hh.write('    }\n  }\n}\n\n')

    if (len(self.sel_classesT)):
      hh.write('namespace {\n')
      i = 0
      for c in self.sel_classesT:
	if c.split('<')[0].strip() not in self.abstractTemplateClasses:
	  hh.write('  %s inst%d;\n' % (c,i))
	  i += 1
      hh.write('}\n\n')
    
if __name__ == "__main__":
  rc = genreflex_rootcint()
  rc.parse_args()
  rc.parse_headers()
  rc.parse_pragmas()
  rc.gen_temp_header()

  gr = genreflex.genreflex()
  # ,'-D__MAKECINT__'
  gr_args = ['',rc.dict_header,'--quiet','-o',rc.dict_filename,'--comments','-I.','-Iinclude','-DTRUE=1','-DFALSE=0','-Dexternalref=extern','-DR__EXTERN=extern']
  gr_args += rc.gccxml_ppopts
  for arg in gr_args :
    if arg in ('-DSYSV',) : gr_args.remove(arg)
  gr.parse_args(gr_args)
  gr.check_files_dirs()
  gr.process_files()
