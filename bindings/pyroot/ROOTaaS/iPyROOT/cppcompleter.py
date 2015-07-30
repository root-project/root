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
    '''

    def __init__(self):
        ROOT.gInterpreter.Declare(_TTabComHookCode)
        self.active = True
        self.firstActivation = True

    def activate(self):
        self.active = True
        if self.firstActivation:
           utils._loadLibrary("libRint.so")
           self.firstActivation = False

    def deactivate(self):
        self.active = False

    def complete(self, ip, event) :
        '''
        Autocomplete interfacing to TTabCom. If an accessor of a scope is
        present in the line, the suggestions are prepended with the line.
        That's how completers work. For example:
        myGraph.Set<tab> will return "myGraph.Set+suggestion in the list of
        suggestions.
        '''
        if self.active:
           suggestions = ROOT._TTabComHook(event.line)
           accessors = [".", "->", "::"]
           if any(accessor in event.line for accessor in accessors):
               suggestions = [event.line+sugg for sugg in suggestions]
           return suggestions
        else:
           return []


_cppCompleter = CppCompleter()

def load_ipython_extension(ipython):
    _cppCompleter.activate()
    ipython.set_hook('complete_command', _cppCompleter.complete,re_key=r"(.+)")

def unload_ipython_extension(ipython):
    _cppCompleter.deactivate()
