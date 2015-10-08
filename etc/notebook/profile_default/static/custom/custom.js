/* ROOTaaS JS */

highlight_cells = function(IPython, mime) {
    IPython.CodeCell.options_default.cm_config.mode = mime;
    var cells = IPython.notebook.get_cells();
    for (i = 0; i < cells.length; i++) {
        var cell = cells[i];
        if (cell.cell_type == "code") {
            cell.code_mirror.setOption('mode', mime);
            cell.cm_config.mode = mime;
        }
    }
}

// Configure C++ syntax highlighting for magics and C++-only notebooks
require(['base/js/namespace', 'base/js/events', 'codemirror/mode/clike/clike'],
    function(IPython, events, clike) {
        var cppMIME = 'text/x-c++src';
        events.on("kernel_ready.Kernel", function() {
            var kernel_name = IPython.notebook.kernel.name;
            if (kernel_name == "python2") {
                IPython.CodeCell.config_defaults.highlight_modes['magic_' + cppMIME] = {'reg':[/^%%cpp/]};
                console.log("ROOTaaS - C++ magics highlighting configured");
            }
            else if (kernel_name == "root") {
                highlight_cells(IPython, cppMIME);
                $('body').one('keydown.wysiwyg', function() {
                    highlight_cells(IPython, cppMIME);
                });
                events.one("edit_mode.Notebook", function() {
                    highlight_cells(IPython, cppMIME);
                });
                console.log("ROOTaaS - C++ highlighting ON");
            }
        });
    });

// Terminal button
$(document).ready(function() {
    $('div#header-container').append("<a href='terminals/1' class='btn btn-default btn-sm navbar-btn pull-right' style='margin-right: 4px; margin-left: 2px;'>Terminal</a>");
});
