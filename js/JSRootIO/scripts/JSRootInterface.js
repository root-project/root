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
var frame_id = 0;

function addCollapsible(element) {
   $(element)
       .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
       .hover(function() { $(this).toggleClass("ui-state-hover"); })
       .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
       .click(function() {
          $(this)
             .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
             .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
             .next().toggleClass("ui-accordion-content-active").slideToggle();
          return false;
       })
       .next()
          .addClass("ui-accordion-content  ui-helper-reset ui-widget-content ui-corner-bottom")
             .hide();
   $(element)
      .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
      .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
      .next().toggleClass("ui-accordion-content-active").slideToggle();

};

function showElement(element) {
   if ($(element).next().is(":hidden")) {
      $(element)
         .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
         .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
         .next().toggleClass("ui-accordion-content-active").slideDown("slow");
   }
}

function loadScript(url, callback) {
   // dynamic script loader using callback
   // (as loading scripts may be asynchronous)
   var script = document.createElement("script")
   script.type = "text/javascript";
   if (script.readyState) { // Internet Explorer specific
      script.onreadystatechange = function() {
         if (script.readyState == "loaded" ||
             script.readyState == "complete") {
            script.onreadystatechange = null;
            callback();
         }
      };
   } else { // Other browsers
      script.onload = function(){
         callback();
      };
   }
   var rnd = Math.floor(Math.random()*80000);
   script.src = url;//+ "?r=" + rnd;
   document.getElementsByTagName("head")[0].appendChild(script);
};

function displayRootStatus(msg) {
   $("#status").append(msg);
};

function displayListOfKeys(keys) {
   JSROOTPainter.displayListOfKeys(keys, '#status');
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
         }
         return true;
      }
   }
   return false;
};

function showObject(obj_name, cycle) {
   gFile.ReadObject(obj_name, cycle);
};

function displayDirectory(directory, cycle, dir_id) {
   $("#status").html("<br/>JSROOTIO.RootFile.js version: " + JSROOTIO.version + "<br/>");
   var url = $("#urlToLoad").val();
   $("#status").append("load: " + url + "<br/>");
   JSROOTPainter.addDirectoryKeys(directory.fKeys, '#status', dir_id);
};

function showDirectory(dir_name, cycle, dir_id) {
   gFile.ReadDirectory(dir_name, cycle, dir_id);
};

function displayObject(obj, cycle, idx) {
   if (!obj['_typename'].match(/\bTH1/) &&
       !obj['_typename'].match(/\bTH2/) &&
       obj['_typename'] != 'JSROOTIO.TGraph' &&
       obj['_typename'] != 'JSROOTIO.TCanvas' &&
       obj['_typename'] != 'JSROOTIO.TF1' &&
       obj['_typename'] != 'JSROOTIO.TProfile') {
      return;
   }
   var uid = "uid_accordion_"+(++last_index);
   var entryInfo = "<h5 id=\""+uid+"\"><a> " + obj['fName'] + ";" + cycle + "</a>&nbsp; </h5>\n";
   entryInfo += "<div id='histogram" + idx + "'>\n";
   $("#report").append(entryInfo);
   JSROOTPainter.drawObject(obj, idx);
   addCollapsible('#'+uid);
};

function AssertPrerequisites(andThen) {
   if (typeof JSROOTIO == "undefined") {
      // if JSROOTIO is not defined, then dynamically load the required scripts and open the file
      loadScript('http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js', function() {
      loadScript('http://ajax.googleapis.com/ajax/libs/jqueryui/1.8.18/jquery-ui.min.js', function() {
      loadScript(source_dir+'scripts/dtree.js', function() {
      loadScript(source_dir+'scripts/rawinflate.js', function() {
      loadScript(source_dir+'scripts/JSRootCore.js', function() {
      loadScript('http://d3js.org/d3.v2.min.js', function() {
      loadScript(source_dir+'scripts/JSRootD3Painter.js', function() { 
      loadScript(source_dir+'scripts/JSRootIOEvolution.js', function() {
         if (andThen) {
            andThen();
         }
         else {
            $("#status").html("<br/>JSROOTIO.RootFile.js version: " + JSROOTIO.version + "<br/>");
            var url = $("#urlToLoad").val();
            if (url == "" || url == " ") return;
            $("#status").append("load: " + url + "<br/>");
            ResetUI();
            gFile = new JSROOTIO.RootFile(url);
            $('#report').append("</body></html>");
            $('#report').addClass("ui-accordion ui-accordion-icons ui-widget ui-helper-reset");
         }
      }) }) }) }) }) }) }) });
      return true;
   }
   return false;
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
   if (AssertPrerequisites()) return;
   // else simply open the file
   $("#status").html("<br/>JSROOTIO.RootFile.js version: " + JSROOTIO.version + "<br/>");
   var url = $("#urlToLoad").val();
   if (url == "" || url == " ") return;
   $("#status").append("load: " + url + "<br/>");
   if (gFile) {
      gFile.Delete();
      delete gFile;
   }
   gFile = null;
   gFile = new JSROOTIO.RootFile(url);
   $('#report').append("</body></html>");
};

function ResetUI() {
   obj_list.splice(0, obj_list.length);
   func_list.splice(0, func_list.length);
   obj_index = 0;
   last_index = 0;
   frame_id = 0;
   if (gFile) {
      gFile.Delete();
      delete gFile;
   }
   gFile = null;
   $("#report").get(0).innerHTML = '';
   $("#report").innerHTML = '';
   delete $("#report").get(0);
   //window.location.reload(true);
   $("#status").html("<br/>JSROOTIO.RootFile.js version: " + JSROOTIO.version + "<br/>");
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
   var guiCode = "<div id='main' class='column'>\n"
      +"<h1><font face='Verdana' size='4'>Read a ROOT file with Javascript</font></h1>\n"
      +"<p><b>Select a ROOT file to read, or enter a url (*): </b><br/>\n"
      +'<small><sub>*: Other URLs might not work because of cross site scripting protection, see e.g. <a href="https://developer.mozilla.org/en/http_access_control">http://developer.mozilla.org/en/http_access_control</a> on how to avoid it.</sub></small></p>'
      +'<form name="ex">'
      +'<div style="margin-left:10px;">'
      +'<input type="text" name="state" value="" size="40" id="urlToLoad"/><br/>'
      +'<select name="s" size="1" '
      +'onchange="document.ex.state.value = document.ex.s.options[document.ex.s.selectedIndex].value;document.ex.s.selectedIndex=0;document.ex.s.value=\'\'">'
      +'<option value = " " selected = "selected">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>';
      for (var i=0; i<arrFiles.length; i++) {
         guiCode += '<option value = "' + arrFiles[i] + '">' + arrFiles[i] + '</option>';
      }
      guiCode += '</select>'
      +'</div>'
      +'<input style="padding:2px; margin-left:10px; margin-top:5px;"'
      +' onclick="ReadFile()" type="button" title="Read the Selected File" value="Load"/>'
      +'<input style="padding:2px; margin-left:10px;"'
      +'onclick="ResetUI()" type="button" title="Clear All" value="Reset"/>'
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
