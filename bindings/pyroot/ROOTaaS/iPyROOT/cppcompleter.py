import utils
import ROOT

# Jit a wrapper for the ttabcom
_TTabComHookCode = """
std::vector<std::string> _TTabComHook(const char* pattern){
  static auto ttc = new TTabCom;
  int pLoc = strlen(pattern);
  std::ostringstream oss;
  ttc->Hook((char* )pattern, &pLoc, oss);
  std::stringstream ss(oss.str());
  std::istream_iterator<std::string> vbegin(ss), vend;
  return std::vector<std::string> (vbegin, vend);
}
"""

class CppCompleter(object):
    '''
    Completer which interfaces to the TTabCom of ROOT. It is activated
    (deactivated) upon the load(unload) of the load of the extension.

    >>> comp = CppCompleter()
    >>> comp.activate()
    >>> for suggestion in comp._completeImpl("TH"):
    ...     print suggestion
    THashList
    THashTable
    THashTableIter
    THelix
    THYPE
    TH1
    TH1C
    TH1S
    TH1I
    TH1F
    TH1D
    TH3
    TH3C
    TH3S
    TH3I
    TH3F
    TH3D
    TH2GL
    TH3GL
    TH1K
    TH2
    TH2C
    TH2S
    TH2I
    TH2F
    TH2D
    TH2PolyBin
    TH2Poly
    THLimitsFinder
    THnBase
    THnIter
    THn
    THnSparseArrayChunk
    THnSparse
    THStack
    THistPainter
    >>> garbage = ROOT.gInterpreter.ProcessLine("TH1F* h;")
    >>> for suggestion in comp._completeImpl("h->GetA"):
    ...     print suggestion
    h->GetAsymmetry
    h->GetAxisColor
    h->GetArray
    h->GetAt
    >>> comp.deactivate()
    >>> for suggestion in comp._completeImpl("TG"):
    ...     print suggestion
    '''

    def __init__(self):
        self.hook = None 
        self.active = True
        self.firstActivation = True
        self.accessors = [".", "->", "::"]

    def activate(self):
        self.active = True
        if self.firstActivation:
            utils.declareCppCode('#include "dlfcn.h"')
            dlOpenRint = 'dlopen("libRint.so",RTLD_NOW);'
            utils.processCppCode(dlOpenRint)
            utils.declareCppCode(_TTabComHookCode)
            self.hook = ROOT._TTabComHook
            self.firstActivation = False

    def deactivate(self):
        self.active = False

    def _getSuggestions(self,line):
        if self.active:
            return self.hook(line)
        return []

    def _getLastAccessorPos(self,line):
        accessorPos = -1
        for accessor in self.accessors:
            tmpAccessorPos = line.rfind(accessor)
            if accessorPos < tmpAccessorPos:
                accessorPos = tmpAccessorPos+len(accessor)
        return accessorPos

    def _completeImpl(self, line):
        suggestions = self._getSuggestions(line)
        print "LINE: %s SUGGESTIONS: %s" %(line,sorted(suggestions))
        if not suggestions: return []
        accessorPos = self._getLastAccessorPos(line)
        if accessorPos > 0:
            suggestions = [line[:accessorPos]+sugg for sugg in suggestions]
        return suggestions

    def complete(self, ip, event) :
        '''
        Autocomplete interfacing to TTabCom. If an accessor of a scope is
        present in the line, the suggestions are prepended with the line.
        That's how completers work. For example:
        myGraph.Set<tab> will return "myGraph.Set+suggestion in the list of
        suggestions.
        '''
        return self._completeImpl(event.line)


_cppCompleter = CppCompleter()

def load_ipython_extension(ipython):
    _cppCompleter.activate()
    ipython.set_hook('complete_command', _cppCompleter.complete, re_key=r"(.*)")

def unload_ipython_extension(ipython):
    _cppCompleter.deactivate()

