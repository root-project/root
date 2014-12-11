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

function BuildNoBrowserGUI(online) {
   var itemsarr = [];
   var optionsarr = [];
   var running_request = {};

   var filename = null;
   if (!online) {
      filename = JSROOT.GetUrlOption("file");
      var filesdir = JSROOT.GetUrlOption("path");
      if (filesdir!=null) filename = filesdir + filename;
   }

   var itemname = JSROOT.GetUrlOption("item");
   if (itemname) itemsarr.push(itemname);
   var opt = JSROOT.GetUrlOption("opt");
   if (opt) optionsarr.push(opt);

   var items = JSROOT.GetUrlOption("items");
   if (items != null) {
      items = JSON.parse(items);
      for (var i in items) itemsarr.push(items[i]);
   }

   var opts = JSROOT.GetUrlOption("opts");
   if (opts!=null) {
      opts = JSON.parse(opts);
      for (var i in opts) optionsarr.push(opts[i]);
   }

   var layout = JSROOT.GetUrlOption("layout");
   if (layout=="") layout = null;

   var monitor = JSROOT.GetUrlOption("monitoring");
   if (monitor == "") monitor = 3000; else
   if (monitor != null) monitor = parseInt(monitor);

   var divid = online ? "onlineGUI" : "simpleGUI";

   $('#'+divid).empty();

   $('html').css('height','100%');
   $('body').css('min-height','100%').css('margin','0px').css("overflow", "hidden");

   $('#'+divid).css("position", "absolute")
               .css("left", "1px")
               .css("top", "1px")
               .css("bottom", "1px")
               .css("right", "1px");

   var objpainter = null;
   var mdi = null;

   function file_error(str) {
      if ((objpainter == null) && (mdi==null))
         $('#'+divid).append("<h4>" + str + "</h4>");
   }

   if ((filename == null) && !online) {
      return file_error('filename not specified');
   }

   if (itemsarr.length == 0) {
      return file_error('itemname not specified');
   }

   var title = online ? "Online"  : ("File: " + filename);
   if (itemsarr.length == 1) title += " item: " + itemsarr[0];
                        else title += " items: " + itemsarr.toString();
   document.title = title;

   function draw_object(indx, obj) {
      document.body.style.cursor = 'wait';
      if (obj==null)  {
         file_error("object " + itemsarr[indx] + " not found");
      } else
      if (mdi) {
         var frame = mdi.FindFrame(itemsarr[indx], true);
         mdi.ActivateFrame(frame);
         JSROOT.redraw($(frame).attr('id'), obj, optionsarr[indx]);
      } else {
         objpainter = JSROOT.redraw(divid, obj, optionsarr[indx]);
      }
      document.body.style.cursor = 'auto';
      running_request[indx] = false;
   }

   function read_object(file, indx) {

      if (itemsarr[indx]=="StreamerInfo")
         draw_object(indx, file.fStreamerInfos);

      file.ReadObject(itemsarr[indx], function(obj) {
         draw_object(indx, obj);
      });
   }

   function request_object(indx) {

      if (running_request[indx]) return;

      running_request[indx] = true;

      var url = itemsarr[indx] + "/root.json.gz?compact=3";

      var itemreq = JSROOT.NewHttpRequest(url, 'object', function(obj) {
         if ((obj != null) &&  (itemsarr[indx] === "StreamerInfo")
               && (obj['_typename'] === 'TList'))
            obj['_typename'] = 'TStreamerInfoList';

         draw_object(indx, obj);
      });

      itemreq.send(null);
   }

   function read_all_objects() {

      if (online) {
         for (var i in itemsarr)
            request_object(i);
         return;
      }

      for (var i in itemsarr)
         if (running_request[i]) {
            console.log("Request for item " + itemsarr[i] + " still running");
            return;
         }

      new JSROOT.TFile(filename, function(file) {
         if (file==null) return file_error("file " + filename + " cannot be opened");

         for (var i in itemsarr) {
            running_request[i] = true;
            read_object(file, i);
         }
      });
   }

   if (itemsarr.length > 1) {
      if ((layout==null) || (layout=='collapsible') || (layout == "")) {
         var divx = 2; divy = 1;
         while (divx*divy < itemsarr.length) {
            if (divy<divx) divy++; else divx++;
         }
         layout = 'grid' + divx + 'x' + divy;
      }

      if (layout=='tabs')
         mdi = new JSROOT.TabsDisplay(divid);
      else
         mdi = new JSROOT.GridDisplay(divid, layout);

      // Than create empty frames for each item
      for (var i in itemsarr)
         mdi.CreateFrame(itemsarr[i]);
   }

   read_all_objects();

   if (monitor>0)
      setInterval(read_all_objects, monitor);

   JSROOT.RegisterForResize(function() { if (objpainter) objpainter.CheckResize(); if (mdi) mdi.CheckResize(); });
}

function ReadFile(filename, checkitem) {
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

   if (filename==null) {
      filename = $("#urlToLoad").val();
      filename.trim();
   } else {
      $("#urlToLoad").val(filename);
   }
   if (filename.length == 0) return;

   var layout = null;
   var itemsarr = [];
   var optionsarr = [];
   if (checkitem) {
      var itemname = JSROOT.GetUrlOption("item");
      if (itemname) itemsarr.push(itemname);
      var items = JSROOT.GetUrlOption("items");
      if (items!=null) {
         items = JSON.parse(items);
         for (var i in items) itemsarr.push(items[i]);
      }

      layout = JSROOT.GetUrlOption("layout");
      if (layout=="") layout = null;

      var opt = JSROOT.GetUrlOption("opt");
      if (opt) optionsarr.push(opt);
      var opts = JSROOT.GetUrlOption("opts");
      if (opts!=null) {
         opts = JSON.parse(opts);
         for (var i in opts) optionsarr.push(opts[i]);
      }
   }

   if (layout==null)
      layout = guiLayout();
   else
      setGuiLayout(layout);

   if (hpainter==null) hpainter = new JSROOT.HierarchyPainter('root', 'browser');
   hpainter.SetDisplay(layout, 'right-div');

   AddInteractions();

   hpainter.OpenRootFile(filename, function() {
      hpainter.displayAll(itemsarr, optionsarr);
   });
}

function ProcessResize(direct)
{
   if (hpainter==null) return;

   if (direct) document.body.style.cursor = 'wait';

   hpainter.CheckResize();

   if (direct) document.body.style.cursor = 'auto';
}

function AddInteractions() {

   JSROOT.ConfigureVSeparator(hpainter);

   JSROOT.RegisterForResize(hpainter);

   // specify display kind every time selection done
   // will be actually used only for first drawing or after reset
   $("#layout").change( function() {
      if (hpainter)
         hpainter.SetDisplay(guiLayout(), "right-div");
   });
}


function BuildOnlineGUI() {
   var myDiv = $('#onlineGUI');
   if (!myDiv) {
      alert("You have to define a div with id='onlineGUI'!");
      return;
   }

   JSROOT.Painter.readStyleFromURL();

   if (JSROOT.GetUrlOption("nobrowser")!=null)
      return BuildNoBrowserGUI(true);

   var guiCode = '<div id="left-div" class="column">'
            + '<h1><font face="Verdana" size="4">ROOT online server</font></h1>'
            + "<p><font face='Verdana' size='1px'><a href='http://root.cern.ch/js/jsroot.html'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>"
            + '<p> Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format</p>'
            + ' <input type="checkbox" name="monitoring" id="monitoring"/> Monitoring '
            + ' <select style="padding:2px; margin-left:10px; margin-top:5px;" id="layout">'
            + '   <option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
            + ' </select>'
            + ' <div id="browser"></div>'
            + '</div>'
            + '<div id="separator-div" class="column"></div>'
            + '<div id="right-div" class="column"></div>';

   $('#onlineGUI').empty().append(guiCode);

   var layout = JSROOT.GetUrlOption("layout");
   if ((layout=="") || (layout==null))
      layout = guiLayout();
   else
      setGuiLayout(layout);

   var monitor = JSROOT.GetUrlOption("monitoring");

   var itemsarr = [], optionsarr = [];
   var itemname = JSROOT.GetUrlOption("item");
   if (itemname) itemsarr.push(itemname);
   var items = JSROOT.GetUrlOption("items");
   if (items!=null) {
      items = JSON.parse(items);
      for (var i in items) itemsarr.push(items[i]);
   }

   var opt = JSROOT.GetUrlOption("opt");
   if (opt) optionsarr.push(opt);
   var opts = JSROOT.GetUrlOption("opts");
   if (opts!=null) {
      opts = JSON.parse(opts);
      for (var i in opts) optionsarr.push(opts[i]);
   }

   if (hpainter==null) hpainter = new JSROOT.HierarchyPainter("root", "browser");
   hpainter.SetDisplay(layout, 'right-div');

   AddInteractions();

   hpainter.EnableMonitoring(monitor!=null);
   $("#monitoring")
      .prop('checked', monitor!=null)
      .click(function() {
         hpainter.EnableMonitoring(this.checked);
         if (this.checked) h.updateAll();
      });

   var h0 = null;
   if (typeof GetCashedHierarchy == 'function') h0 = GetCashedHierarchy();
   if (typeof h0 != 'object') h0 = "";

   hpainter.OpenOnline(h0, function() {
       hpainter.displayAll(itemsarr, optionsarr);
   });

   setInterval(function() { if (hpainter.IsMonitoring()) hpainter.updateAll(); }, hpainter.MonitoringInterval());
}

function BuildSimpleGUI() {

   if (document.getElementById('onlineGUI')) return BuildOnlineGUI();

   var myDiv = $('#simpleGUI');
   if (!myDiv) return;

   JSROOT.Painter.readStyleFromURL();

   if (JSROOT.GetUrlOption("nobrowser")!=null)
      return BuildNoBrowserGUI(false);

   var files = JSROOT.GetUrlOption("files");
   if (files==null) files = myDiv.attr("files");
   var filesdir = JSROOT.GetUrlOption("path");
   if (filesdir==null) filesdir = myDiv.attr("path");

   if (files==null) files = "files/hsimple.root";
   if (filesdir==null) filesdir = "";
   var arrFiles = files.split(';');

   var filename = JSROOT.GetUrlOption("file");

   var guiCode = "<div id='left-div' class='column'>"
      +"<h1><font face='Verdana' size='4'>Read a ROOT file</font></h1>"
      +"<p><font face='Verdana' size='1px'><a href='http://root.cern.ch/js/jsroot.html'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></font></p>";

   if ((JSROOT.GetUrlOption("noselect")==null) || (filename==null)) {
     guiCode += '<form name="ex">'
      +'<input type="text" name="state" value="" style="width:95%; margin-top:5px;" id="urlToLoad"/>'
      +'<select name="s" style="width:65%; margin-top:5px;" '
      +'onchange="document.ex.state.value = document.ex.s.options[document.ex.s.selectedIndex].value;document.ex.s.selectedIndex=0;document.ex.s.value=\'\'">'
      +'<option value=" " selected="selected">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>';
      for (var i=0; i<arrFiles.length; i++) {
         guiCode += '<option value = "' + filesdir + arrFiles[i] + '">' + arrFiles[i] + '</option>';
      }
      guiCode += '</select><br/>'
        +'<p><small>Other URLs might not work because of <a href="http://en.wikipedia.org/wiki/Same-origin_policy">same-origin security policy</a>, '
        +'see e.g. <a href="https://developer.mozilla.org/en/http_access_control">developer.mozilla.org</a> on how to avoid it.</small></p>'
        +'<input style="padding:2px; margin-top:5px;"'
        +'       onclick="ReadFile()" type="button" title="Read the Selected File" value="Load"/>'
        +'<input style="padding:2px; margin-left:10px;"'
        +'       onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
        +'<select style="padding:2px; margin-left:10px; margin-top:5px;" title="layout kind" id="layout">'
        +'  <option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
        +'</select><br/>'
        +'</form>';
   }
   guiCode += '<div id="browser"></div>'
      +'</div>'
      +'<div id="separator-div"></div>'
      +'<div id="right-div" class="column"></div>';

   $('#simpleGUI').empty().append(guiCode);

   // $("#layout").selectmenu();

   AddInteractions();

   if ((typeof filename == 'string') && (filename.length>0))
      ReadFile(filename, true);
}
