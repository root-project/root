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

   var myDiv = $('#simpleGUI');
   var online = false;

   if (myDiv.length==0) {
      myDiv = $('#onlineGUI');
      if (myDiv.length==0) return alert('no div for simple gui found');
      online = true;
   }

   JSROOT.Painter.readStyleFromURL();

   var nobrowser = JSROOT.GetUrlOption("nobrowser") != null;

   var guiCode = "<div id='left-div' class='column'>";

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

      var guiCode = "<div id='left-div' class='column'>"
         +"<h1><font face='Verdana' size='4'>Read a ROOT file</font></h1>"
         +"<p><font face='Verdana' size='1px'><a href='http://root.cern.ch/js/jsroot.html'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>";

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
           +'<div id="separator-div"></div>'
           +'<div id="right-div" class="column"></div>';

   var drawDivId = 'right-div';

   if (nobrowser) {
      guiCode = "";
      $('html').css('height','100%');
      $('body').css('min-height','100%').css('margin','0px').css("overflow", "hidden");

      drawDivId = myDiv.attr('id');

      myDiv.css("position", "absolute")
           .css("left", "1px")
           .css("top", "1px")
           .css("bottom", "1px")
           .css("right", "1px");
   }

   myDiv.empty().append(guiCode);

   var filesarr = JSROOT.GetUrlOptionAsArray("file;files");
   var filesdir = JSROOT.GetUrlOption("path");
   if (filesdir!=null)
      for (var i in filesarr) filesarr[i] = filesdir + filesarr[i];

   var itemsarr = JSROOT.GetUrlOptionAsArray("item;items");

   var optionsarr = JSROOT.GetUrlOptionAsArray("opt;opts");

   var monitor = JSROOT.GetUrlOption("monitoring");

   var layout = JSROOT.GetUrlOption("layout");
   if (layout=="") layout = null;

   hpainter = new JSROOT.HierarchyPainter('root', nobrowser ? null : 'browser');

   if (JSROOT.GetUrlOption('files_monitoring')!=null) hpainter.files_monitoring = true;

   JSROOT.RegisterForResize(hpainter);

   if (nobrowser) {
      if (layout==null) layout= "simple";
   } else {
      if (layout==null)
         layout = guiLayout();
      else
         setGuiLayout(layout);

      JSROOT.ConfigureVSeparator(hpainter);

      // specify display kind every time selection done
      // will be actually used only for first drawing or after reset
      $("#layout").change(function() {
         if (hpainter) hpainter.SetDisplay(guiLayout(), drawDivId);
      });
   }

   hpainter.SetDisplay(layout, drawDivId);

   hpainter.EnableMonitoring(monitor!=null);

   var h0 = null;

   if (online) {
      if (!nobrowser)
         $("#monitoring")
          .prop('checked', monitor!=null)
          .click(function() {
             hpainter.EnableMonitoring(this.checked);
             if (this.checked) hpainter.updateAll();
          });

       if (typeof GetCashedHierarchy == 'function') h0 = GetCashedHierarchy();
       if (typeof h0 != 'object') h0 = "";
   } else {
      if ((filesarr.length>0) && !nobrowser)
         $("#urlToLoad").val(filesarr[0]);
   }

   function OpenAllFiles() {
      if (filesarr.length>0)
         hpainter.OpenRootFile(filesarr.shift(), OpenAllFiles);
      else
         hpainter.displayAll(itemsarr, optionsarr, function() { hpainter.RefreshHtml(); });
   }

   function AfterOnlineOpened() {
      // check if server enables monitoring
      if ('_monitoring' in hpainter.h) {
         var v = parseInt(hpainter.h._monitoring);
         if ((v == NaN) || (hpainter.h._monitoring == 'false')) {
            hpainter.EnableMonitoring(false);
         } else {
            hpainter.EnableMonitoring(true);
            hpainter.MonitoringInterval(v);
         }
         if (!nobrowser) $("#monitoring").prop('checked', hpainter.IsMonitoring());
      }

      if ('_loadfile' in hpainter.h)
         filesarr.push(hpainter.h._loadfile);

      if ('_drawitem' in hpainter.h) {
         itemsarr.push(hpainter.h._drawitem);
         optionsarr.push('_drawopt' in hpainter.h ? hpainter.h._drawopt : "");
      }

      OpenAllFiles();
   }

   if (h0!=null) hpainter.OpenOnline(h0, AfterOnlineOpened);
            else OpenAllFiles();

   setInterval(function() { if (hpainter.IsMonitoring()) hpainter.updateAll(); }, hpainter.MonitoringInterval());
}
