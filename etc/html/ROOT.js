function SetCSSValue(where,what,to){
   var r='cssRules';
   if(document.all && navigator.appName.indexOf('Opera')==-1)
      r='rules';
   var i;
   for(i=0;i<document.styleSheets.length;++i) {
      var cssrules=document.styleSheets[i][r];
      for(j=0;j<cssrules.length;++j)
         if(cssrules[j].selectorText.toUpperCase()==where.toUpperCase()) {
            cssrules[j].style[what]=to;
            return false;
         }
   }
   return false;
}
var elements=new Array('dispoptCBInh.checked','dispoptCBPub.checked');
function SetValuesFromCookie() {
   var i;
   var arrcookie=document.cookie.split(";");
   for(i=0; i<arrcookie.length; ++i) {
      while(arrcookie[i].charAt(0)==' ') 
         arrcookie[i]=arrcookie[i].substring(1,arrcookie[i].length);
      if (arrcookie[i].indexOf("ROOT")==0) {
         var arrval=arrcookie[i].substring(5).split(':');
         for (i=0; i<arrval.length; ++i) {
            var posdelim=elements[i].indexOf(".");
            var what=elements[i].substring(0,posdelim);
            var mem =elements[i].substring(posdelim+1);
            var val=arrval[i];
            if (val=='false') val=false;
            else if (val=='true') val=true;
            var el=document.getElementById(what);
            if (!el) return;
            el[mem]=val;
            CBChanged(el);
         }
         return;
      }
   }
}
function UpdateCookie() {
   var cookietxt="ROOT=";
   var i;
   for (i=0; i<elements.length; ++i) {
      var posdelim=elements[i].indexOf(".");
      var what=elements[i].substring(0,posdelim);
      var mem =elements[i].substring(posdelim+1);
      var val=document.getElementById(what)[mem];
      if (i>0) cookietxt+=':';
      cookietxt+=val;
   }
   var ayear=new Date();
   ayear.setTime(ayear.getTime()+31536000000);
   cookietxt+=";path=/;expires="+ayear.toUTCString();
   document.cookie=cookietxt;
}
function CBChanged(cb){
   if(cb.id=='dispoptCBInh') {
      SetCSSValue('tr.funcinh','display',cb.checked?'':'none');
      SetCSSValue('tr.datainh','display',cb.checked?'':'none');
   } else if(cb.id=='dispoptCBPub') {
      SetCSSValue('#funcprot','display',cb.checked?'':'none');
      SetCSSValue('#funcpriv','display',cb.checked?'':'none');
      SetCSSValue('#dataprot','display',cb.checked?'':'none');
      SetCSSValue('#datapriv','display',cb.checked?'':'none');
      SetCSSValue('#enumprot','display',cb.checked?'':'none');
      SetCSSValue('#enumpriv','display',cb.checked?'':'none');
   }
   UpdateCookie();
}
function SetImg(name, file) {
   var img=document.getElementById(name);
   var src=img.src;
   var posFile=src.lastIndexOf('/');
   var numSlashes=file.split('/').length - 1;
   for (var i=0; i<numSlashes; i++)
     posFile=src.lastIndexOf('/',posFile - 1);
   var oldFile=src.substr(posFile+1);
   src=src.substr(0,posFile+1);
   src+=file;
   img.src=src;
   if (img.useMap) {
      var usemapFile=file;
      var posUsemapExt=usemapFile.lastIndexOf('.');
      if (posUsemapExt!=-1) usemapFile=usemapFile.substr(0,posUsemapExt);
      var posUsemapDir=usemapFile.lastIndexOf('/');
      if (posUsemapDir!=-1) usemapFile=usemapFile.substr(posUsemapDir+1);
      img.useMap="#Map"+usemapFile;
   }
   var posExt=oldFile.lastIndexOf('.');
   oldFile=oldFile.substr(0,posExt);
   var posDir=oldFile.lastIndexOf('/');
   if (posDir!=-1) oldFile=oldFile.substr(posDir + 1);
   document.getElementById("img"+oldFile).className="tab";
   posExt=file.lastIndexOf('.');
   file=file.substr(0,posExt);
   posDir=file.lastIndexOf('/');
   if (posDir!=-1) file=file.substr(posDir + 1);
   document.getElementById("img"+file).className="tabsel";
   return false;
}
function SetDiv(name, id) {
   var i=0;
   var div=document.getElementById(name+'_'+i);
   while(div) {
      if (i==id) { div.className="tabvisible"; }
      else {div.className="tabhidden";}
      ++i;
      div=document.getElementById(name+'_'+i);
   }
   i=0;
   div=document.getElementById(name+'_A'+i);
   while(div) {
      if (i==id) { div.className="tabsel"; }
      else {div.className="tab";}
      ++i;
      div=document.getElementById(name+'_A'+i);
   }
   return false;
}
function WriteFollowPageBox(title, lib, incl) {
   document.writeln('<div id="followpage">');
   document.writeln('<a id="followpageshower" class="followpagedisp" '
      + 'href="#" onclick="javascript:SetCSSValue(\'#followpageshower\',\'display\',\'none\');return SetCSSValue(\'#followpagecontent\',\'display\',\'block\');">+</a>');
   document.writeln('<div id="followpagecontent"><div id="followpagetitle">' + title + '</div>');
   document.writeln('<a class="followpagedisp" id="followpagehider" '
      + 'href="#" onclick="javascript:SetCSSValue(\'#followpageshower\',\'display\',\'inline\');return SetCSSValue(\'#followpagecontent\',\'display\',\'none\');">-</a>');
   if (lib.length || incl.length) {
      document.writeln('<div class="libinfo">');
      if (lib.length)
         document.writeln('library: ' + lib + '<br />');
      if (incl.length)
         document.writeln('#include "' + incl + '"<br />');
      document.writeln('</div>');
   }
   document.writeln('<div id="dispopt">Display options:<br />');
   document.writeln('<form id="formdispopt" action="#">');
   document.writeln('<input id="dispoptCBInh" type="checkbox" '
      + 'onclick="javascript:CBChanged(this);" '
      + 'title="Select to display inherited members" />Show inherited<br />');
   document.writeln('<input id="dispoptCBPub" type="checkbox" checked="checked" '
      + 'onclick="javascript:CBChanged(this);" '
      + 'title="Select to display protected and private members" />Show non-public<br />');
   document.writeln('</form>');
   document.writeln('</div>');
   document.writeln('<div id="followlinks">');
   document.writeln('<a href="#TopOfPage">[ &uarr; Top ]</a> |');
   document.writeln(' <a href="HELP.html">[ ? Help ]</a>'); 
   document.writeln('</div>');
   document.writeln('</div>');
   document.writeln('</div>');
}
