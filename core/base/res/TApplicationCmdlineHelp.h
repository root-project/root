constexpr static const char *kCommandLineOptionsHelp = R"RAW(
usage: root [-b] [-x] [-e] [-n] [-t] [-q] [-l] [-config] [-h|--help] [--version]
            [--notebook] [--web=<type|off>] [dir] [data1.root...dataN.root] [file1.C...fileN.C]
            [file1_C.so...fileN_C.so] [anyfile1..anyfileN] [-- [macro args]]


root is an interactive interpreter of C++ code using Cling and the ROOT framework.
For more information on ROOT, please refer to https://root.cern/
An extensive Users Guide and API Reference are available from that website.


OPTIONS:
  -b, --batch                          Run in batch mode without graphics
  -x, --exit-on-exceptions             Exit on exceptions
  -e, --execute                        Execute the command passed between single quotes
  -n, --no-logon-logoff                Do not execute logon and logoff macros as specified in .rootrc
  -t, --enable-threading               Enable thread-safety and implicit multi-threading (IMT)
  -q, --quit-after-processing          Exit after processing command line macro files
  -l, --no-banner                      Do not show the ROOT banner
  -config                              print ./configure options
  -h, -?, --help                       Show summary of options
  --version                            Show the ROOT version
  --notebook                           Execute ROOT notebook
  --web                                Use web-based display for graphics, browser, geometry [deprecated, use "web=on" instead]
  --web=<type>                         Use the specified web-based display such as chrome, firefox, qt6
                                       For more options see the documentation of TROOT::SetWebDisplay()
  --web=off                            Disable any kind of web-based display
  [dir]                                if dir is a valid directory cd to it before executing
  [data1.root...dataN.root]            Open the given ROOT files; remote protocols (such as http://) are supported
  [file1.C...fileN.C]                  Execute the ROOT macro file1.C ... fileN.C
                                       Compilation flags as well as macro arguments can be passed, see format in https://root.cern/manual/root_macros_and_shared_libraries/
  [file1_C.so...fileN_C.so]            Load and execute file1_C.so ... fileN_C.so (or .dll if on Windows)
                                       They should be already-compiled ROOT macros (shared libraries) or:
                                       regular user shared libraries e.g. userlib.so with a function userlib(args)
  [anyfile1..anyfileN]                 All other arguments pointing to existing files will be checked to see if they are ROOT Files (checking the MIME type inside the file) and if they are not they will be handled as a ROOT macro file
  [-- [macro args]]                    Any arguments after a `--` will be treated as arguments to the last macro passed as a positional argument (e.g. `root myMacro.C -- 1 2 3` will be equivalent to `root "myMacro.C(1,2,3)"`)
)RAW";
