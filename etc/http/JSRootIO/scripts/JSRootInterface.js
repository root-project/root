// JSRootInterface.js
//
// interface methods for Javascript ROOT Web Page.
//

// global variables
// source_dir is the variable defining where to take the scripts and the list tree icons
// To use the local ones (e.g. when checking out the files in a web server), just let it
// empty: var source_dir = "";
var source_dir = "http://root.cern.ch/js/";
var gFile;
var obj_list = new Array();
var obj_index = 0;
var last_index = 0;
var function_list = new Array();
var func_list = new Array();
var collections_list = {};

function closeCollapsible(e, el) {
   var sel = $(el)[0].textContent;
   if (typeof(sel) == 'undefined') return;
   sel.replace(' x', '');
   sel.replace(';', '');
   sel.replace(' ', '');
   var i = obj_list.indexOf(sel)
   if (i >= 0) obj_list.splice(i, 1);
   $(el).next().andSelf().remove();
   e.stopPropagation();
};

function addCollapsible(element) {
   $(element)
       .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
       .hover(function() { $(this).toggleClass("ui-state-hover"); })
       .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
       .append('<button type="button" class="closeButton" title="close canvas" onclick="closeCollapsible(event, \''+element+'\')"><img src="'+source_dir+'/img/remove.gif"/></button>')
       .click(function() {
          $(this)
             .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
             .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
             .next().toggleClass("ui-accordion-content-active").slideToggle(0);
          return false;
       })
       .next()
          .addClass("ui-accordion-content  ui-helper-reset ui-widget-content ui-corner-bottom")
             .hide();
   $(element)
      .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
      .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
      .next().toggleClass("ui-accordion-content-active").slideToggle(0);

};

function showElement(element) {
   if ($(element).next().is(":hidden")) {
      $(element)
         .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
         .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
         .next().toggleClass("ui-accordion-content-active").slideDown(0);
   }
   $(element)[0].scrollIntoView();
}

function loadScript(url, callback) {
   // dynamic script loader using callback
   // (as loading scripts may be asynchronous)
   var script = document.createElement("script");
   script.type = "text/javascript";
   if (script.readyState) { // Internet Explorer specific
      script.onreadystatechange = function() {
         if (script.readyState == "loaded" ||
             script.readyState == "complete") {
            script.onreadystatechange = null;
            if (callback!=null) callback();
         }
      }
   } else { // Other browsers
      script.onload = function(){
         if (callback!=null) callback();
      };
   }
   var rnd = Math.floor(Math.random()*80000);
   script.src = url;//+ "?r=" + rnd;
   document.getElementsByTagName("head")[0].appendChild(script);
};

function displayRootStatus(msg) {
   $("#status").append(msg);
};

function displayStreamerInfos(streamerInfo) {
   var findElement = $('#report').find('#treeview');
   if (findElement.length) {
      var element = findElement[0].parentElement.previousSibling.id;
      showElement('#'+element);
   }
   else {
      var uid = "uid_accordion_"+(++last_index);
      var entryInfo = "<h5 id=\""+uid+"\"><a> Streamer Infos </a>&nbsp; </h5><div>\n";
      entryInfo += "<h6>Streamer Infos</h6><span id='treeview' class='dtree'></span></div>\n";
      $("#report").append(entryInfo);
      JSROOTPainter.displayStreamerInfos(streamerInfo, '#treeview');
      addCollapsible('#'+uid);
   }
};

function findObject(obj_name) {
   for (var i in obj_list) {
      if (obj_list[i] == obj_name) {
         var findElement = $('#report').find('#histogram'+i);
         if (findElement.length) {
            var element = findElement[0].previousElementSibling.id;
            showElement('#'+element);
            return true;
         }
      }
   }
   return false;
};

function showObject(obj_name, cycle, dir_id) {
   gFile.ReadObject(obj_name, cycle, dir_id);
};

function showDirectory(dir_name, cycle, dir_id) {
   gFile.ReadDirectory(dir_name, cycle, dir_id);
};

function readTree(tree_name, cycle, node_id) {
   gFile.ReadObject(tree_name, cycle, node_id);
};

function displayTree(tree, cycle, node_id) {
   var url = $("#urlToLoad").val();
   $("#status").html("file: " + url + "<br/>");
   JSROOTPainter.displayTree(tree, '#status', node_id);
};

function displayCollection(name, cycle, c_id, coll) {
   var fullname = name + ";" + cycle;

   collections_list[fullname] = coll;

   JSROOTPainter.addCollectionContents(fullname, c_id, coll, '#status');
};


function displayObject(obj, cycle, idx) {
   if (!obj) return;
   if (!JSROOTPainter.canDrawObject(obj['_typename'])) return;
   var uid = "uid_accordion_"+(++last_index);
   var entryInfo = "<h5 id=\""+uid+"\"><a> " + obj['fName'] + ";" + cycle + "</a>&nbsp; </h5>\n";
   entryInfo += "<div id='histogram" + idx + "'>\n";
   $("#report").append(entryInfo);
   
   var render_to = '#histogram' + idx;
   if (typeof($(render_to)[0]) == 'undefined') return;
   
   $(render_to).empty();

   var vis = JSROOTPainter.createCanvas($(render_to), obj);
   
   if (vis == null) return;

   JSROOTPainter.drawObjectInFrame(vis, obj);
   
   addCollapsible('#'+uid);
};


function showListObject(list_name, obj_name) {

   var fullname = list_name+"/"+obj_name+"1";

   // do not display object twice
   if (obj_list.indexOf(fullname)>=0) return;

   var coll = collections_list[list_name];
   if (!coll) return;

   var obj = null;

   for (var i=0;i<coll.arr.length;i++)
     if (coll.arr[i].fName == obj_name) {
        obj = coll.arr[i];
        break;
     }
   if (!obj) return;

   displayObject(obj, "1", obj_index);
   obj_list.push(fullname);
   obj_index++;
};


function AssertPrerequisites(andThen) {

   if (typeof JSROOTIO == "undefined") {
      // if JSROOTIO is not defined, then dynamically load the required scripts and open the file
      loadScript(source_dir+'scripts/jquery.min.js', function() {
      loadScript(source_dir+'scripts/jquery-ui.min.js', function() {
      loadScript(source_dir+'scripts/d3.v3.min.js', function() {
      loadScript(source_dir+'scripts/jquery.mousewheel.js', function() {
      loadScript(source_dir+'scripts/dtree.js', function() {
      loadScript(source_dir+'scripts/rawinflate.js', function() {
      loadScript(source_dir+'scripts/JSRootCore.js', function() {
      loadScript(source_dir+'scripts/three.min.js', function() {
      loadScript(source_dir+'fonts/helvetiker_regular.typeface.js', function() {
      loadScript(source_dir+'scripts/JSRootIOEvolution.js', function() {
      loadScript(source_dir+'scripts/JSRootPainter.js', function() {

         if (andThen!=null) andThen();

         // if report element exists - this is standard ROOT layout
         if (document.getElementById("report")) {
            var version = "<div id='overlay'><font face='Verdana' size='1px'>&nbspJSROOTIO version:" + JSROOTIO.version + "&nbsp</font></div>";
            $(version).prependTo("body");
            $('#report').addClass("ui-accordion ui-accordion-icons ui-widget ui-helper-reset");
         }

      }) }) }) }) }) }) }) }) }) }) });
   } else {
      if (andThen!=null) andThen();
   }
};

function ReadFile() {
   var navigator_version = navigator.appVersion;
   if (typeof ActiveXObject == "function") { // Windows
      // detect obsolete browsers
      if ((navigator_version.indexOf("MSIE 8") != -1) ||
          (navigator_version.indexOf("MSIE 7") != -1))  {
         alert("You need at least MS Internet Explorer version 9.0. Note you can also use any other web browser (excepted Opera)");
         return;
      }
   }
   else {
      // Safari 5.1.7 on MacOS X doesn't work properly
      if ((navigator_version.indexOf("Windows NT") == -1) &&
          (navigator_version.indexOf("Safari") != -1) &&
          (navigator_version.indexOf("Version/5.1.7") != -1)) {
         alert("There are know issues with Safari 5.1.7 on MacOS X. It may become unresponsive or even hangs. You can use any other web browser (excepted Opera)");
         return;
      }
   }

   var url = $("#urlToLoad").val();
   if (url == "" || url == " ") return;
   $("#status").html("file: " + url + "<br/>");
   if (gFile) {
      gFile.Delete();
      delete gFile;
   }

   gFile = new JSROOTIO.RootFile(url);
}

function ResetUI() {
   obj_list.splice(0, obj_list.length);
   func_list.splice(0, func_list.length);
   collections_list = {};
   obj_index = 0;
   last_index = 0;
   if (gFile) {
      gFile.Delete();
      delete gFile;
   }
   gFile = null;
   $("#report").get(0).innerHTML = '';
   $("#report").innerHTML = '';
   delete $("#report").get(0);
   //window.location.reload(true);
   $('#status').get(0).innerHTML = '';
   $('#report').get(0).innerHTML = '';
   $(window).unbind('resize');
};

function BuildSimpleGUI() {
   AssertPrerequisites(function DisplayGUI() {
   var myDiv = $('#simpleGUI');
   if (!myDiv) {
      alert("You have to define a div with id='simpleGUI'!");
      return;
   }
   var files = myDiv.attr("files");
   if (!files) {
      alert("div id='simpleGUI' must have a files attribute!");
      return;
   }
   var arrFiles = files.split(';');


   var guiCode = "<div id='overlay'><font face='Verdana' size='1px'>&nbspJSROOTIO version:" + JSROOTIO.version + "&nbsp</font></div>"

      guiCode += "<div id='main' class='column'>\n"
      +"<h1><font face='Verdana' size='4'>Read a ROOT file with Javascript</font></h1>\n"
      +"<p><b>Select a ROOT file to read, or enter a url (*): </b><br/>\n"
      +'<small><sub>*: Other URLs might not work because of cross site scripting protection, see e.g. <a href="https://developer.mozilla.org/en/http_access_control">developer.mozilla.org/http_access_control</a> on how to avoid it.</sub></small></p>'
      +'<form name="ex">'
      +'<div style="margin-left:10px;">'
      +'<input type="text" name="state" value="" size="30" id="urlToLoad"/><br/>'
      +'<select name="s" size="1" '
      +'onchange="document.ex.state.value = document.ex.s.options[document.ex.s.selectedIndex].value;document.ex.s.selectedIndex=0;document.ex.s.value=\'\'">'
      +'<option value = " " selected = "selected">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>';
      for (var i=0; i<arrFiles.length; i++) {
         guiCode += '<option value = "' + arrFiles[i] + '">' + arrFiles[i] + '</option>';
      }
      guiCode += '</select>'
      +'</div>'
      +'<input style="padding:2px; margin-left:10px; margin-top:5px;"'
      +'       onclick="ReadFile()" type="button" title="Read the Selected File" value="Load"/>'
      +'<input style="padding:2px; margin-left:10px;"'
      +'       onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
      +'</form>'
      +'<br/>'
      +'<div id="status"></div>'
      +'</div>'
      +'<div id="reportHolder" class="column">'
      +'<div id="report"> </div>'
      +'</div>';
      $('#simpleGUI').append(guiCode);
   });
};
