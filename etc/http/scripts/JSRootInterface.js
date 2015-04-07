// JSRootInterface.js
//
// default user interface for JavaScript ROOT Web Page.
//

var hpainter = null;

function ResetUI() {
   if (hpainter) hpainter.clear(true);
}

function guiLayout() {
   var res = 'collapsible';
   var selects = document.getElementById("layout");
   if (selects)
      res = selects.options[selects.selectedIndex].text;
   return res;
}

function setGuiLayout(value) {
   var selects = document.getElementById("layout");
   if (!selects) return;

   for (var i in selects.options) {
      var s = selects.options[i].text;
      if (typeof s == 'undefined') continue;
      if ((s == value) || (s.replace(/ /g,"") == value)) {
         selects.selectedIndex = i;
         break;
      }
   }
}

function ReadFile() {
   var navigator_version = navigator.appVersion;
   if (typeof ActiveXObject == "function") { // Windows
      // detect obsolete browsers
      if ((navigator_version.indexOf("MSIE 8") != -1) ||
          (navigator_version.indexOf("MSIE 7") != -1))  {
         alert("You need at least MS Internet Explorer version 9.0. Note you can also use any other web browser");
         return;
      }
   }
   else {
      // Safari 5.1.7 on MacOS X doesn't work properly
      if ((navigator_version.indexOf("Windows NT") == -1) &&
          (navigator_version.indexOf("Safari") != -1) &&
          (navigator_version.indexOf("Version/5.1.7") != -1)) {
         alert("There are know issues with Safari 5.1.7 on MacOS X. It may become unresponsive or even hangs. You can use any other web browser");
         return;
      }
   }

   var filename = $("#urlToLoad").val();
   filename.trim();
   if (filename.length == 0) return;

   if (hpainter==null) alert("Hierarchy painter not initialized");
                  else hpainter.OpenRootFile(filename);
}


function BuildSimpleGUI() {

   if (JSROOT.GetUrlOption("nobrowser")!=null)
      return JSROOT.BuildNobrowserGUI();

   var myDiv = $('#simpleGUI');
   var online = false;

   if (myDiv.length==0) {
      myDiv = $('#onlineGUI');
      if (myDiv.length==0) return alert('no div for simple gui found');
      online = true;
   }

   JSROOT.Painter.readStyleFromURL();

   var guiCode = "<div id='left-div' class='column' style='top:1px; bottom:1px'>";

   if (online) {
      guiCode += '<h1><font face="Verdana" size="4">ROOT online server</font></h1>'
         + "<p><font face='Verdana' size='1px'><a href='http://root.cern.ch/js/jsroot.html'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>"
         + '<p> Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format</p>'
         + ' <input type="checkbox" name="monitoring" id="monitoring"/> Monitoring '
         + ' <select style="padding:2px; margin-left:10px; margin-top:5px;" id="layout">'
         + '   <option>simple</option><option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
         + ' </select>';
   } else {

      var files = myDiv.attr("files");
      var path = myDiv.attr("path");

      if (files==null) files = "../files/hsimple.root";
      if (path==null) path = "";
      var arrFiles = files.split(';');

      guiCode += "<h1><font face='Verdana' size='4'>Read a ROOT file</font></h1>"
              + "<p><font face='Verdana' size='1px'><a href='http://root.cern.ch/js/'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>";

      if (JSROOT.GetUrlOption("noselect")==null) {
        guiCode += '<form name="ex">'
           +'<input type="text" name="state" value="" style="width:95%; margin-top:5px;" id="urlToLoad"/>'
           +'<select name="s" style="width:65%; margin-top:5px;" '
           +'onchange="document.ex.state.value = document.ex.s.options[document.ex.s.selectedIndex].value;document.ex.s.selectedIndex=0;document.ex.s.value=\'\'">'
           +'<option value=" " selected="selected">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>';
        for (var i in arrFiles)
           guiCode += '<option value = "' + path + arrFiles[i] + '">' + arrFiles[i] + '</option>';
        guiCode += '</select><br/>'
           +'<p><small>Other file URLs might not work because of <a href="http://en.wikipedia.org/wiki/Same-origin_policy">same-origin security policy</a>, '
           +'see e.g. <a href="https://developer.mozilla.org/en/http_access_control">developer.mozilla.org</a> on how to avoid it.</small></p>'
           +'<input style="padding:2px; margin-top:5px;"'
           +'       onclick="ReadFile()" type="button" title="Read the Selected File" value="Load"/>'
           +'<input style="padding:2px; margin-left:10px;"'
           +'       onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
           +'<select style="padding:2px; margin-left:10px; margin-top:5px;" title="layout kind" id="layout">'
           +'  <option>simple</option><option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
           +'</select><br/>'
           +'</form>';
      }
   }

   guiCode += '<div id="browser"></div>'
           +'</div>'
           +'<div id="separator-div" style="top:1px; bottom:1px"></div>'
           +'<div id="right-div" class="column" style="top:1px; bottom:1px"></div>';

   var drawDivId = 'right-div';

   myDiv.empty().append(guiCode);

   var h0 = null;

   if (online) {
      if (typeof GetCachedHierarchy == 'function') h0 = GetCachedHierarchy();
      if (typeof h0 != 'object') h0 = "";
   }

   hpainter = new JSROOT.HierarchyPainter('root', 'browser');

   hpainter.SetDisplay(guiLayout(), drawDivId);

   JSROOT.Painter.ConfigureVSeparator(hpainter);

   // JSROOT.Painter.ConfigureHSeparator(28, true);

   hpainter.StartGUI(h0, function() {

      setGuiLayout(hpainter.GetLayout());

      // specify display kind every time selection done
      // will be actually used only for first drawing or after reset
      $("#layout").change(function() {
         if (hpainter) hpainter.SetDisplay(guiLayout(), drawDivId);
      });

      if (online) {
         $("#monitoring")
             .prop('checked', hpainter.IsMonitoring())
             .click(function() {
                hpainter.EnableMonitoring(this.checked);
                if (this.checked) hpainter.updateAll();
             });
      } else {
         var fname = "";
         hpainter.ForEachRootFile(function(item) { if (fname=="") fname = item._fullurl; });
         $("#urlToLoad").val(fname);
      }
   });
}
