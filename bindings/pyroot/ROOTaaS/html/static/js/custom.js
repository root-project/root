/* ROOTaaS JS */

// Make sure Clike JS lexer is loaded, then configure syntax highlighting for %%cpp and %%dcl magics
require(['codemirror/mode/clike/clike', 'base/js/namespace', 'notebook/js/codecell'], function(Clike, IPython, CodeCell) {
    IPython.CodeCell.config_defaults.highlight_modes['magic_text/x-c++src'] = {'reg':[/^%%cpp|^%%dcl/]};
    console.log("ROOTaaS - C++ magics highlighting configured");
});
