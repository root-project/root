// JSRootInterface.js
//
// default user interface for JavaScript ROOT Web Page.
//

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['jquery', 'jquery-ui', 'd3', 'JSRootPainter'], factory );
   } else {

      if (typeof jQuery == 'undefined')
         throw new Error('jQuery not defined', 'JSRootPainter.jquery.js');

      if (typeof jQuery.ui == 'undefined')
         throw new Error('jQuery-ui not defined', 'JSRootPainter.jquery.js');

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRootPainter.jquery.js');

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.jquery.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.jquery.js');

      // Browser globals
      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   var hpainter = null, localfile_read_callback = null;

   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/JSRootInterface.css');

   ResetUI = function() {
      if (hpainter) hpainter.clear(true);
   }

   guiLayout = function() {
      var selects = document.getElementById("gui_layout");
      return selects ? selects.options[selects.selectedIndex].text : 'collapsible';
   }

   setGuiLayout = function(value) {
      var selects = document.getElementById("gui_layout");
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

   LocalFileSelected = function(evnt) {
      var files = evnt.target.files;

      for (var n=0;n<files.length;++n) {
         var f = files[n];
         $("#gui_urlToLoad").val(f.name);
         if (hpainter) hpainter.OpenRootFile(f, localfile_read_callback);
      }

      localfile_read_callback = null;
   }

   ReadSelectedFile = function() {
      var filename = $("#gui_urlToLoad").val().trim();
      if (!filename.length) return;
      if (!hpainter) alert("Hierarchy painter not initialized");

      if ((filename.lastIndexOf(".json") == filename.length-5) ||
            (filename.lastIndexOf(".JSON") == filename.length-5))
         hpainter.OpenJsonFile(filename);
      else
         hpainter.OpenRootFile(filename);
   }

   BuildSimpleGUI = function() {

      var myDiv = $('#simpleGUI'), online = false;

      if (myDiv.length==0) {
         myDiv = $('#onlineGUI');
         if (myDiv.length==0) return alert('no div for simple gui found');
         online = true;
      }

      if (myDiv.attr("ignoreurl") === "true")
         JSROOT.gStyle.IgnoreUrlOptions = true;

      if ((JSROOT.GetUrlOption("nobrowser")!==null) || (myDiv.attr("nobrowser") && myDiv.attr("nobrowser")!=="false"))
         return JSROOT.BuildNobrowserGUI();

      JSROOT.Painter.readStyleFromURL();

      var guiCode = "<div id='left-div' class='column' style='top:1px; bottom:1px'>";

      if (online) {
         guiCode += '<h1><font face="Verdana" size="4"><div id="toptitle">ROOT online server</div></font></h1>'
                 + "<p><font face='Verdana' size='1px'><a href='https://github.com/linev/jsroot'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>"
                 + '<p> Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format</p>'
                 + ' <input type="checkbox" name="monitoring" id="monitoring"/> Monitoring '
                 + ' <select style="padding:2px; margin-left:10px; margin-top:5px;" id="gui_layout">'
                 + ' <option>simple</option><option>collapsible</option><option>flex</option><option>tabs</option><option>grid 1x2</option><option>grid 2x2</option><option>grid 1x3</option><option>grid 2x3</option><option>grid 3x3</option><option>grid 4x4</option>'
                 + ' </select>';
      } else {

         guiCode += "<h1><font face='Verdana' size='4'>Read a ROOT file</font></h1>"
                  + "<p><font face='Verdana' size='1px'><a href='https://root.cern/js/'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>";

         var noselect = JSROOT.GetUrlOption("noselect") || myDiv.attr("noselect");

         if (!noselect) {
            var files = myDiv.attr("files") || "../files/hsimple.root",
                path = JSROOT.GetUrlOption("path") || myDiv.attr("path") || "",
                arrFiles = files.split(';');

            guiCode +=
               '<input type="text" value="" style="width:95%; margin-top:5px;" id="gui_urlToLoad" title="input file name"/>'
               +'<select id="selectFileName" style="width:65%; margin-top:5px;" title="select file name"'
               +'<option value="" selected="selected"></option>';
            for (var i in arrFiles)
               guiCode += '<option value = "' + path + arrFiles[i] + '">' + arrFiles[i] + '</option>';
            guiCode += '</select>'
               +'<input type="file" id="gui_localFile" accept=".root" style="display:none"/><output id="list" style="display:none"></output>'
               +'<input type="button" value="..." id="gui_fileBtn" style="width:15%; margin-top:5px;margin-left:5px;" title="select local file for reading"/><br/>'
               +'<p><small><a href="https://github.com/linev/jsroot/blob/master/docs/JSROOT.md#reading-root-files-from-other-servers">Read docu</a>'
               +' how to open files from other servers.</small></p>'
               +'<input style="padding:2px; margin-top:5px;"'
               +'       onclick="ReadSelectedFile()" type="button" title="Read the Selected File" value="Load"/>'
               +'<input style="padding:2px; margin-left:10px;"'
               +'       onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
               +'<select style="padding:2px; margin-left:10px; margin-top:5px;" title="layout kind" id="gui_layout">'
               +'  <option>simple</option><option>collapsible</option><option>flex</option><option>tabs</option><option>grid 1x2</option><option>grid 2x2</option><option>grid 1x3</option><option>grid 2x3</option><option>grid 3x3</option><option>grid 4x4</option>'
               +'</select><br/>';
         } else
         if (noselect === "file") {
            guiCode += '<select style="padding:2px; margin-left:5px; margin-top:5px;" title="layout kind" id="gui_layout">'
                     + '  <option>simple</option><option>collapsible</option><option>flex</option><option>tabs</option><option>grid 1x2</option><option>grid 2x2</option><option>grid 1x3</option><option>grid 2x3</option><option>grid 3x3</option><option>grid 4x4</option>'
                     + '</select><br/>';
         }
      }

      guiCode += '<div id="browser"></div>'
               + '</div>'
               + '<div id="separator-div" style="top:1px; bottom:1px"></div>'
               + '<div id="right-div" class="column" style="top:1px; bottom:1px"></div>';

      var drawDivId = 'right-div';

      myDiv.empty().append(guiCode);

      var h0 = null;

      if (online) {
         if (typeof GetCachedHierarchy == 'function') h0 = GetCachedHierarchy();
         if (typeof h0 != 'object') h0 = "";
      } else
      if (!noselect) {
         $("#selectFileName").val("").change(function() {
            $("#gui_urlToLoad").val($(this).val());
         });
         $("#gui_fileBtn").click(function() {
            $("#gui_localFile").click();
         });

         $("#gui_localFile").change(LocalFileSelected);
      }

      hpainter = new JSROOT.HierarchyPainter('root', 'browser');

      hpainter.SetDisplay(null, drawDivId);

      hpainter._topname = JSROOT.GetUrlOption("topname") || myDiv.attr("topname");

      JSROOT.Painter.ConfigureVSeparator(hpainter);

      // JSROOT.Painter.ConfigureHSeparator(28, true);

      hpainter.SelectLocalFile = function(read_callback) {
         localfile_read_callback = read_callback;
         $("#gui_localFile").click();
      }

      hpainter.StartGUI(h0, function() {

         setGuiLayout(hpainter.GetLayout());

         // specify display kind every time selection done
         // will be actually used only for first drawing or after reset
         $("#gui_layout").change(function() {
            if (hpainter) hpainter.SetDisplay(guiLayout(), drawDivId);
         });

         if (online) {
            if ((hpainter.h!=null) && ('_toptitle' in hpainter.h))
               $("#toptitle").html(hpainter.h._toptitle);
            $("#monitoring")
              .prop('checked', hpainter.IsMonitoring())
              .click(function() {
                  hpainter.EnableMonitoring(this.checked);
                  hpainter.updateAll(!this.checked);
               });
         } else
         if (!noselect) {
            var fname = "";
            hpainter.ForEachRootFile(function(item) { if (fname=="") fname = item._fullurl; });
            // document.getElementById('gui_localFile').addEventListener('change', LocalFileSelected, false);

            $("#gui_urlToLoad").val(fname).keyup(function(e) {
               if (e.keyCode == 13) ReadSelectedFile();
            });
         }
      }, d3.select('#simpleGUI'));
   }

   return JSROOT;

}));

