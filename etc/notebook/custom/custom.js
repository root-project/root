/* JupyROOT JS */

// Terminal button
$(document).ready(function() {
    var terminalURL = "/terminals/1";
    var re = new RegExp(".+\/user\/([a-z0-9]+)\/.+", "g");
    if (re.test(document.URL)) {
        // We have been spawned by JupyterHub, add the user name to the URL
        var user = document.URL.replace(re, "$1");
        terminalURL = "/user/" + user + terminalURL;
    }
    $('div#header-container').append("<a href='" + terminalURL + "' class='btn btn-default btn-sm navbar-btn pull-right' style='margin-right: 4px; margin-left: 2px;'>Terminal</a>");
});

// Highlighting for %%cpp cells
require(['notebook/js/codecell'], function(codecell) {
    codecell.CodeCell.options_default.highlight_modes['magic_text/x-c++src'] = {'reg':[/^%%cpp/]};
});
