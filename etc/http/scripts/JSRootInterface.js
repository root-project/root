// JSRootInterface.js
//
// default user interface for JavaScript ROOT Web Page.
//

function ResetUI() {
   if (JSROOT.H('root') != null) {
      JSROOT.H('root').clear();
      JSROOT.DelHList('root');
   }
   $('#browser').get(0).innerHTML = '';
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
         mdi.Redraw(itemsarr[indx], obj, optionsarr[indx]);
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

   var painter = new JSROOT.HierarchyPainter('root', 'browser');

   painter.SetDisplay(layout, 'right-div');

   painter.OpenRootFile(filename, function() {
      painter.displayAll(itemsarr, optionsarr);
   });
}

function UpdateOnline() {
   var h = JSROOT.H('root');

   if (h['_monitoring_on'] && ('disp' in h))
     h['disp'].ForEach(function(panel, itemname, painter) {
       if (painter==null) return;

       // prevent to update item if previous not completed
       if ('_doing_update' in painter)  return;

       painter['_doing_update'] = true;

       h.get(itemname, function(item, obj) {
         if (painter.UpdateObject(obj)) {
            document.body.style.cursor = 'wait';
            painter.RedrawPad();
            document.body.style.cursor = 'auto';
         }
         delete painter['_doing_update'];
      });
     } , true); // update only visible objects
}

function ProcessResize(direct)
{
   if (direct) document.body.style.cursor = 'wait';

   JSROOT.H('root').CheckResize();

   if (direct) document.body.style.cursor = 'auto';
}

function AddInteractions() {
   var drag_sum = 0;

   var drag_move = d3.behavior.drag()
      .origin(Object)
      .on("dragstart", function() {
          d3.event.sourceEvent.preventDefault();
          // console.log("start drag");
          drag_sum = 0;
       })
      .on("drag", function() {
         d3.event.sourceEvent.preventDefault();
         drag_sum += d3.event.dx;
         // console.log("dx = " + d3.event.dx);
         d3.event.sourceEvent.stopPropagation();
      })
      .on("dragend", function() {
         d3.event.sourceEvent.preventDefault();
         // console.log("stop drag " + drag_sum);

         var width = d3.select("#left-div").style('width');
         width = (parseInt(width.substr(0, width.length - 2)) + Number(drag_sum)).toString() + "px";
         d3.select("#left-div").style('width', width);

         var left = d3.select("#separator-div").style('left');
         left = parseInt(left.substr(0, left.length - 2)) + Number(drag_sum);
         d3.select("#separator-div").style('left',left.toString() + "px");
         d3.select("#right-div").style('left',(left+6).toString() + "px");

         ProcessResize(true);
      });

   d3.select("#separator-div").call(drag_move);

   JSROOT.RegisterForResize(ProcessResize);

   // specify display kind every time selection done
   // will be actually used only for first drawing or after reset
   document.getElementById("layout").onchange = function() {
      if (JSROOT.H('root'))
         JSROOT.H('root').SetDisplay(guiLayout(), "right-div");
   }
}


function BuildOnlineGUI() {
   var myDiv = $('#onlineGUI');
   if (!myDiv) {
      alert("You have to define a div with id='onlineGUI'!");
      return;
   }

   if (JSROOT.GetUrlOption("nobrowser")!=null)
      return BuildNoBrowserGUI(true);

   var guiCode = "<div id='overlay'><font face='Verdana' size='1px'>&nbspJSROOT version " + JSROOT.version + "&nbsp</font></div>"

   guiCode += '<div id="left-div" class="column"><br/>'
            + '  <h1><font face="Verdana" size="4">ROOT online server</font></h1>'
            + '  Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format<br/><br/>'
            + ' <input type="checkbox" name="monitoring" id="monitoring"/> Monitoring '
            + ' <select style="padding:2px; margin-left:10px; margin-top:5px;" id="layout">'
            + '   <option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
            + ' </select>'
            + ' <div id="browser"></div>'
            + '</div>'
            + '<div id="separator-div" class="column"></div>'
            + '<div id="right-div" class="column"></div>';

   $('#onlineGUI').empty();
   $('#onlineGUI').append(guiCode);

   var layout = JSROOT.GetUrlOption("layout");
   if ((layout=="") || (layout==null))
      layout = guiLayout();
   else
      setGuiLayout(layout);

   var interval = 3000;
   var monitor = JSROOT.GetUrlOption("monitoring");
   if (monitor != null) {
      document.getElementById("monitoring").checked = true;
      if (monitor!="") interval = parseInt(monitor);
      if ((interval == NaN) || (interval<=0)) interval = 3000;
   }

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

   var h = new JSROOT.HierarchyPainter("root", "browser");

   h.SetDisplay(layout, 'right-div');

   h['_monitoring_interval'] = interval;
   h['_monitoring_on'] = (monitor!=null);

   h.OpenOnline("", function() {
      h.displayAll(itemsarr, optionsarr);
   });

   setInterval(UpdateOnline, interval);

   AddInteractions();

   $("#monitoring").click(function() {
      h['_monitoring_on'] = this.checked;
      if (h['_monitoring_on']) UpdateOnline();
   });
}

function BuildSimpleGUI() {

   if (document.getElementById('onlineGUI')) return BuildOnlineGUI();

   var myDiv = $('#simpleGUI');
   if (!myDiv) return;

   if (JSROOT.GetUrlOption("nobrowser")!=null)
      return BuildNoBrowserGUI(false);

   var files = JSROOT.GetUrlOption("files");
   if (files==null) files = myDiv.attr("files");
   var filesdir = JSROOT.GetUrlOption("path");
   if (filesdir==null) filesdir = myDiv.attr("path");

   if (files==null) files = "files/hsimple.root";
   if (filesdir==null) filesdir = "";
   var arrFiles = files.split(';');

   var guiCode = "<div id='overlay'><font face='Verdana' size='1px'>&nbspJSROOT version " + JSROOT.version + "&nbsp</font></div>"

   guiCode += "<div id='left-div' class='column'>\n"
      +"<h1><font face='Verdana' size='4'>Read a ROOT file with Javascript</font></h1>\n"
      +"<p><b>Select a ROOT file to read, or enter a url (*): </b><br/>\n"
      +'<small><sub>*: Other URLs might not work because of cross site scripting protection, see e.g. <a href="https://developer.mozilla.org/en/http_access_control">developer.mozilla.org/http_access_control</a> on how to avoid it.</sub></small></p>'
      +'<form name="ex">'
      +'<div style="margin-left:10px;">'
      + '<input type="text" name="state" value="" size="30" id="urlToLoad"/><br/>'
      +'<select name="s" size="1" '
      +'onchange="document.ex.state.value = document.ex.s.options[document.ex.s.selectedIndex].value;document.ex.s.selectedIndex=0;document.ex.s.value=\'\'">'
      +'<option value = " " selected = "selected">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>';
   for (var i=0; i<arrFiles.length; i++) {
      guiCode += '<option value = "' + filesdir + arrFiles[i] + '">' + arrFiles[i] + '</option>';
   }
   guiCode += '</select>'
      +'</div>'
      +'<input style="padding:2px; margin-left:10px; margin-top:5px;"'
      +'       onclick="ReadFile()" type="button" title="Read the Selected File" value="Load"/>'
      +'<input style="padding:2px; margin-left:10px;"'
      +'       onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
      +'<select style="padding:2px; margin-left:10px; margin-top:5px;" id="layout">'
      +'  <option>collapsible</option><option>grid 2x2</option><option>grid 3x3</option><option>grid 4x4</option><option>tabs</option>'
      +'</select>'
      +'</form>'
      +'<br/>'
      +'<div id="browser"></div>'
      +'</div>'
      +'<div id="separator-div" class="column"></div>'
      +'<div id="right-div" class="column"></div>';

   $('#simpleGUI').empty();
   $('#simpleGUI').append(guiCode);
   // $("#layout").selectmenu();

   AddInteractions();

   var filename = JSROOT.GetUrlOption("file");
   if ((typeof filename == 'string') && (filename.length>0))
      ReadFile(filename, true);
}
