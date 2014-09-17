// JSRootPainter.js
//
// JavaScript ROOT Graphics, using d3.v3.js.
//

(function(){

   if (typeof JSROOT != 'object'){
      var e1 = new Error('JSROOT is not defined');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }

   if (typeof d3 != 'object') {
      var e1 = new Error('This extension requires d3.v3.js');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }

   if (typeof JSROOT.Painter == 'object'){
      var e1 = new Error('JSROOT.Painter already defined');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }
   

   // list of user painters, called with arguments painter(vis, obj, opt)
   JSROOT.fUserPainters = null;
   
   JSROOT.addUserPainter = function(class_name, user_painter) {
      if (JSROOT.fUserPainters == null) JSROOT.fUserPainters = {};
      JSROOT.fUserPainters[class_name] = user_painter;
   }

   JSROOT.gStyle = {
      'Tooltip'       : true,  // tooltip on/off
      'OptimizeDraw'  : false, // if true, drawing of 1-D histogram will be optimized to exclude too-many points
      'AutoStat'      : true,
      'OptStat'       : 1111,
      'StatX'         : 0.78,
      'StatY'         : 0.75,
      'StatW'         : 0.2,
      'StatH'         : 0.16,
      'StatColor'     : 0,
      'StatStyle'     : 1001,
      'StatFont'      : 42,
      'StatFontSize'  : 9,
      'StatTextColor' : 1,
      'TimeOffset'    : 788918400000, // UTC time at 01/01/95
      'StatFormat'    : function(v) { return (Math.abs(v) < 1e5) ? v.toFixed(5) : v.toExponential(7); },
      'StatEntriesFormat'  : function(v) { return (Math.abs(v) < 1e7) ? v.toFixed(0) : v.toExponential(7); }
   };

   JSROOT.Painter = {};

   
   JSROOT.Painter.createmenu = function(event, menuname) {
      
      if (!menuname) menuname = "root_ctx_menu";
      
      var xMousePosition = event.clientX + window.pageXOffset;
      var yMousePosition = event.clientY + window.pageYOffset;

      // console.log("Menu for " + itemname + " pos = " + xMousePosition + "," + yMousePosition);
      
      var x = document.getElementById(menuname);
      if(x) x.parentNode.removeChild(x);
     
      var d = document.createElement('div');
      d.setAttribute('class', 'ctxmenu');
      d.setAttribute('id', menuname);  
      document.body.appendChild(d);
      d.style.left = xMousePosition + "px";
      d.style.top = yMousePosition + "px";
      d.onmouseover = function(e) { this.style.cursor = 'pointer'; }
      d.onclick = function(e) { 
         var x = document.getElementById(menuname);
         if(x) x.parentNode.removeChild(x);
      }
      
      document.body.onclick = function(e) { 
         var x = document.getElementById(menuname);
         if(x) x.parentNode.removeChild(x);
      }
      
      return d;
   }
   
   JSROOT.Painter.menuitem = function(menu,txt,func) {
      var p = document.createElement('p');
      menu.appendChild(p);
      p.onclick = func;
      p.setAttribute('class', 'ctxline');
      p.innerHTML = txt;
   }
   
   JSROOT.Painter.Coord = { kCARTESIAN : 1, kPOLAR : 2, kCYLINDRICAL : 3, kSPHERICAL : 4, kRAPIDITY : 5 };

   /** Function that generates all root colors */
   JSROOT.Painter.root_colors = function () {
      var colorMap = new Array(
            'rgb(255, 255, 255)',
            'rgb(0, 0, 0)',
            'rgb(255, 0, 0)',
            'rgb(0, 255, 0)',
            'rgb(0, 0, 255)',
            'rgb(255, 255, 0)',
            'rgb(255, 0, 255)',
            'rgb(0, 255, 255)',
            'rgb(89, 211, 84)',
            'rgb(89, 84, 216)',
            'rgb(254, 254, 254)',
            'rgb(191, 181, 173)',
            'rgb(76, 76, 76)',
            'rgb(102, 102, 102)',
            'rgb(127, 127, 127)',
            'rgb(153, 153, 153)',
            'rgb(178, 178, 178)',
            'rgb(204, 204, 204)',
            'rgb(229, 229, 229)',
            'rgb(242, 242, 242)',
            'rgb(204, 198, 170)',
            'rgb(204, 198, 170)',
            'rgb(193, 191, 168)',
            'rgb(186, 181, 163)',
            'rgb(178, 165, 150)',
            'rgb(183, 163, 155)',
            'rgb(173, 153, 140)',
            'rgb(155, 142, 130)',
            'rgb(135, 102, 86)',
            'rgb(175, 206, 198)',
            'rgb(132, 193, 163)',
            'rgb(137, 168, 160)',
            'rgb(130, 158, 140)',
            'rgb(173, 188, 198)',
            'rgb(122, 142, 153)',
            'rgb(117, 137, 145)',
            'rgb(104, 130, 150)',
            'rgb(109, 122, 132)',
            'rgb(124, 153, 209)',
            'rgb(127, 127, 155)',
            'rgb(170, 165, 191)',
            'rgb(211, 206, 135)',
            'rgb(221, 186, 135)',
            'rgb(188, 158, 130)',
            'rgb(198, 153, 124)',
            'rgb(191, 130, 119)',
            'rgb(206, 94, 96)',
            'rgb(170, 142, 147)',
            'rgb(165, 119, 122)',
            'rgb(147, 104, 112)',
      'rgb(211, 89, 84)');

      var circleColors = [632, 416, 600, 400, 616, 432];

      var rectangleColors = [800, 820, 840, 860, 880, 900];

      var set1 = [ 255,204,204, 255,153,153, 204,153,153, 255,102,102, 204,102,102
                   ,153,102,102, 255, 51, 51, 204, 51, 51, 153, 51, 51, 102, 51, 51
                   ,255,  0,  0, 204,  0,  0, 153,  0,  0, 102,  0,  0,  51,  0,  0];
      var set2 = [ 204,255,204, 153,255,153, 153,204,153, 102,255,102, 102,204,102
                   ,102,153,102, 51,255, 51,  51,204, 51,  51,153, 51,  51,102, 51
                   ,  0,255,  0,   0,204,  0,   0,153,  0,   0,102,  0,  0, 51,  0];
      var set3 = [ 204,204,255, 153,153,255, 153,153,204, 102,102,255, 102,102,204
                   ,102,102,153,  51, 51,255,  51, 51,204,  51, 51,153,  51, 51,102
                   ,  0,  0,255,   0,  0,204,   0,  0,153,   0,  0,102,   0,  0, 51];
      var set4 = [ 255,255,204, 255,255,153, 204,204,153, 255,255,102, 204,204,102
                   ,153,153,102, 255,255, 51, 204,204, 51, 153,153, 51, 102,102, 51
                   ,255,255,  0, 204,204,  0, 153,153,  0, 102,102,  0,  51, 51,  0];
      var set5 = [ 255,204,255, 255,153,255, 204,153,204, 255,102,255, 204,102,204
                   ,153,102,153, 255, 51,255, 204, 51,204, 153, 51,153, 102, 51,102
                   ,255,  0,255, 204,  0,204, 153,  0,153, 102,  0,102,  51,  0, 51];
      var set6 = [ 204,255,255, 153,255,255, 153,204,204, 102,255,255, 102,204,204
                   ,102,153,153,  51,255,255,  51,204,204,  51,153,153,  51,102,102
                   ,  0,255,255,   0,204,204,   0,153,153,   0,102,102,   0, 51,  51];

      var circleSets = new Array(set1, set2, set3, set4, set5, set6);

      var set7 = [ 255,204,153,  204,153,102,  153,102, 51,  153,102,  0,  204,153, 51
                   ,255,204,102,  255,153,  0,  255,204, 51,  204,153,  0,  255,204,  0
                   ,255,153, 51,  204,102,  0,  102, 51,  0,  153, 51,  0,  204,102, 51
                   ,255,153,102,  255,102,  0,  255,102, 51,  204, 51,  0,  255, 51,  0];
      var set8 = [ 153,255, 51,  102,204,  0,   51,102,  0,   51,153,  0,  102,204, 51
                   ,153,255,102,  102,255,  0,  102,255, 51,   51,204,  0,   51,255, 0
                   ,204,255,153,  153,204,102,  102,153, 51,  102,153,  0,  153,204, 51
                   ,204,255,102,  153,255,  0,  204,255, 51,  153,204,  0,  204,255,  0];
      var set9 = [ 153,255,204,  102,204,153,   51,153,102,    0,153,102,   51,204,153
                   ,102,255,204,    0,255,102,   51,255,204,    0,204,153,    0,255,204
                   , 51,255,153,    0,204,102,    0,102, 51,    0,153, 51,   51,204,102
                   ,102,255,153,    0,255,153,   51,255,102,    0,204, 51,    0,255, 51];
      var set10 = [153,204,255,  102,153,204,   51,102,153,    0, 51,153,   51,102,204
                   ,102,153,255,    0,102,255,   51,102,255,    0, 51,204,    0, 51,255
                   , 51,153,255,    0,102,204,    0, 51,102,    0,102,153,   51,153,204
                   ,102,204,255,    0,153,255,   51,204,255,    0,153,204,    0,204,255];
      var set11 = [204,153,255,  153,102,204,  102, 51,153,  102,  0,153,  153, 51,204
                   ,204,102,255,  153,  0,255,  204, 51,255,  153,  0,204,  204,  0,255
                   ,153, 51,255,  102,  0,204,   51,  0,102,   51,  0,153,  102, 51,204
                   ,153,102,255,  102,  0,255,  102, 51,255,   51,  0,204,   51,  0,255];
      var set12 = [255, 51,153,  204,  0,102,  102,  0, 51,  153,  0, 51,  204, 51,102
                   ,255,102,153,  255,  0,102,  255, 51,102,  204,  0, 51,  255,  0, 51
                   ,255,153,204,  204,102,153,  153, 51,102,  153,  0,102,  204, 51,153
                   ,255,102,204,  255,  0,153,  204,  0,153,  255, 51,204,  255,  0,153];

      var rectSets = new Array(set7, set8, set9, set10, set11, set12);

      /*
       * Define circle colors
       */
      for(var i = 0; i < 6; i++) {
         for(var j = 0; j < 15; j++) {
            var colorn = circleColors[i] + j - 10;
            colorMap[colorn] = 'rgb(' + circleSets[i][3*j] + ', ' + circleSets[i][3*j+1] + ', ' + circleSets[i][3*j+2] + ')';
            colorn = rectangleColors[i] + j - 9;
            colorMap[colorn] = 'rgb('+ rectSets[i][3*j] + ', ' + rectSets[i][3*j+1] + ', ' + rectSets[i][3*j+2] + ')';
         }
      }
      return colorMap;
   }();
   
   JSROOT.Painter.root_line_styles = new Array("", "", "3, 3", "1, 2", "3, 4, 1, 4",
         "5, 3, 1, 3", "5, 3, 1, 3, 1, 3, 1, 3", "5, 5",
         "5, 3, 1, 3, 1, 3", "20, 5", "20, 10, 1, 10", "1, 2");

   //Initialize ROOT markers
   JSROOT.Painter.root_markers = new Array('fcircle','fcircle', 'fcross', 'dcross', 'ocircle',
      'gcross', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
      'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
      'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fsquare', 'ftriangle-up',
      'ftriangle-down', 'ocircle', 'osquare', 'otriangle-up', 'odiamond',
      'ocross', 'fstar', 'ostar', 'dcross', 'otriangle-down', 'fdiamond',
      'fcross');


   /** Function that returns the SVG symbol type identifier for a given root matker
    * The result is an array with 3 elements:
    *    the first is the identifier of the root marker in the SVG symbols
    *    the second is true if the shape is filled and false if it is open
    *    the third is true if the shape should be rotated
    * The identifier will be 6 if the shape is a star or 7 if it is '*'
    */
   JSROOT.Painter.getRootMarker = function(i) {
      var marker = JSROOT.Painter.root_markers[i];
      var shape = 0;
      var toFill = true;
      var toRotate = false;

      if (typeof(marker) != 'undefined') {
         var fst = marker.charAt(0);
         switch (fst) {
         case 'd':
            shape = 7;
            return {'shape' : shape};
         case 'o':
            toFill = false;
            break;
         case 'g':
            toRotate = true;
         }

         var type = marker.substr(1, marker.length);
         switch (type) {
         case "circle":
            shape = 0;
            break;
         case "cross":
            shape = 1;
            break;
         case "diamond":
            shape = 2;
            break;
         case "square":
            shape = 3;
            break;
         case "triangle-up":
            shape = 4;
            break;
         case "triangle-down":
            shape = 5;
            break;
         case "star":
            shape = 6;
            break;
         };
      }
      return {
         'shape'    : shape,
         'toFill'   : toFill,
         'toRotate' : toRotate
      };
   };

   JSROOT.Painter.clearCuts = function(chopt) {
      /* decode string "chopt" and remove graphical cuts */
      var left = chopt.indexOf('[');
      if (left == -1) return chopt;
      var right = chopt.indexOf(']');
      if (right == -1) return chopt;
      var nch = right-left;
      if (nch < 2) return chopt;
      for (var i=0;i<=nch;i++) chopt[left+i] = ' ';
      return chopt;
   };

   JSROOT.Painter.decodeOptions = function(opt, histo, pad) {
      /* decode string 'opt' and fill the option structure */
      var hdim = 1; // histo['fDimension'];
      if (histo['_typename'].match(/\bTH2/)) hdim = 2;
      if (histo['_typename'].match(/\bTH3/)) hdim = 3;
      var nch = opt.length;
      var option = { 'Axis': 0, 'Bar': 0, 'Curve': 0, 'Error': 0, 'Hist': 0,
         'Line': 0, 'Mark': 0, 'Fill': 0, 'Same': 0, 'Func': 0, 'Scat': 0,
         'Star': 0, 'Arrow': 0, 'Box': 0, 'Text': 0, 'Char': 0, 'Color': 0,
         'Contour': 0, 'Logx': 0, 'Logy': 0, 'Logz': 0, 'Lego': 0, 'Surf': 0,
         'Off': 0, 'Tri': 0, 'Proj': 0, 'AxisPos': 0, 'Spec': 0, 'Pie': 0,
         'List': 0, 'Zscale': 0, 'FrontBox': 1, 'BackBox': 1, 'System': JSROOT.Painter.Coord.kCARTESIAN,
         'HighRes': 0, 'Zero': 0
      };
      //check for graphical cuts
      var chopt = opt.toUpperCase();
      chopt = JSROOT.Painter.clearCuts(chopt);
      if (hdim > 1) option.Scat = 1;
      if (!nch) option.Hist = 1;
      if (('fFunctions' in histo) && (histo.fFunctions.arr.length > 0)) option.Func = 2;
      if (('fSumw2' in histo) && (histo.fSumw2.length > 0) && (hdim == 1)) option.Error = 2;

      var l = chopt.indexOf('SPEC');
      if (l != -1) {
         option.Scat = 0;
         chopt = chopt.replace('SPEC', '    ');
         var bs = 0;
         l = chopt.indexOf('BF(');
         if (l != -1) {
            bs = parseInt(chopt)
         }
         option.Spec = Math.max(1600, bs);
         return option;
      }
      l = chopt.indexOf('GL');
      if (l != -1) {
         chopt = chopt.replace('GL', '  ');
      }
      l = chopt.indexOf('X+');
      if (l != -1) {
         option.AxisPos = 10;
         chopt = chopt.replace('X+', '  ');
      }
      l = chopt.indexOf('Y+');
      if (l != -1) {
         option.AxisPos += 1;
         chopt = chopt.replace('Y+', '  ');
      }
      if ((option.AxisPos == 10 || option.AxisPos == 1) && (nch == 2)) option.Hist = 1;
      if (option.AxisPos == 11 && nch == 4) option.Hist = 1;
      l = chopt.indexOf('SAMES');
      if (l != -1) {
         if (nch == 5) option.Hist = 1;
         option.Same = 2;
         chopt = chopt.replace('SAMES', '     ');
      }
      l = chopt.indexOf('SAME');
      if (l != -1) {
         if (nch == 4) option.Hist = 1;
         option.Same = 1;
         chopt = chopt.replace('SAME', '    ');
      }
      l = chopt.indexOf('PIE');
      if (l != -1) {
         option.Pie = 1;
         chopt = chopt.replace('PIE', '   ');
      }
      l = chopt.indexOf('LEGO');
      if (l != -1) {
         option.Scat = 0;
         option.Lego = 1;
         chopt = chopt.replace('LEGO', '    ');
         if (chopt[l+4] == '1') { option.Lego = 11; chopt[l+4] = ' '; }
         if (chopt[l+4] == '2') { option.Lego = 12; chopt[l+4] = ' '; }
         if (chopt[l+4] == '3') { option.Lego = 13; chopt[l+4] = ' '; }
         l = chopt.indexOf('FB'); if (l != -1) { option.FrontBox = 0; chopt = chopt.replace('FB', '  '); }
         l = chopt.indexOf('BB'); if (l != -1) { option.BackBox = 0;  chopt = chopt.replace('BB', '  '); }
         l = chopt.indexOf('0'); if (l != -1) { option.Zero = 1;  chopt = chopt.replace('0', ' '); }
      }
      l = chopt.indexOf('SURF');
      if (l != -1) {
         option.Scat = 0;
         option.Surf = 1; chopt = chopt.replace('SURF', '    ');
         if (chopt[l+4] == '1') { option.Surf = 11; chopt[l+4] = ' '; }
         if (chopt[l+4] == '2') { option.Surf = 12; chopt[l+4] = ' '; }
         if (chopt[l+4] == '3') { option.Surf = 13; chopt[l+4] = ' '; }
         if (chopt[l+4] == '4') { option.Surf = 14; chopt[l+4] = ' '; }
         if (chopt[l+4] == '5') { option.Surf = 15; chopt[l+4] = ' '; }
         if (chopt[l+4] == '6') { option.Surf = 16; chopt[l+4] = ' '; }
         if (chopt[l+4] == '7') { option.Surf = 17; chopt[l+4] = ' '; }
         l = chopt.indexOf('FB'); if (l != -1) { option.FrontBox = 0; chopt = chopt.replace('FB', '  '); }
         l = chopt.indexOf('BB'); if (l != -1) { option.BackBox = 0; chopt = chopt.replace('BB', '  '); }
      }
      l = chopt.indexOf('TF3');
      if (l != -1) {
         l = chopt.indexOf('FB'); if (l != -1) { option.FrontBox = 0; chopt = chopt.replace('FB', '  '); }
         l = chopt.indexOf('BB'); if (l != -1) { option.BackBox = 0; chopt = chopt.replace('BB', '  '); }
      }
      l = chopt.indexOf('ISO');
      if (l != -1) {
         l = chopt.indexOf('FB'); if (l != -1) { option.FrontBox = 0; chopt = chopt.replace('FB', '  '); }
         l = chopt.indexOf('BB'); if (l != -1) { option.BackBox = 0; chopt = chopt.replace('BB', '  '); }
      }
      l = chopt.indexOf('LIST'); if (l != -1) { option.List = 1; chopt = chopt.replace('LIST', '  '); }
      l = chopt.indexOf('CONT');
      if (l != -1) {
         chopt = chopt.replace('CONT', '    ');
         if (hdim > 1) {
            option.Scat = 0;
            option.Contour = 1;
            if (chopt[l+4] == '1') { option.Contour = 11; chopt[l+4] = ' '; }
            if (chopt[l+4] == '2') { option.Contour = 12; chopt[l+4] = ' '; }
            if (chopt[l+4] == '3') { option.Contour = 13; chopt[l+4] = ' '; }
            if (chopt[l+4] == '4') { option.Contour = 14; chopt[l+4] = ' '; }
            if (chopt[l+4] == '5') { option.Contour = 15; chopt[l+4] = ' '; }
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('HBAR');
      if (l != -1) {
         option.Hist = 0;
         option.Bar = 20; chopt = chopt.replace('HBAR', '    ');
         if (chopt[l+4] == '1') { option.Bar = 21; chopt[l+4] = ' '; }
         if (chopt[l+4] == '2') { option.Bar = 22; chopt[l+4] = ' '; }
         if (chopt[l+4] == '3') { option.Bar = 23; chopt[l+4] = ' '; }
         if (chopt[l+4] == '4') { option.Bar = 24; chopt[l+4] = ' '; }
      }
      l = chopt.indexOf('BAR');
      if (l != -1) {
         option.Hist = 0;
         option.Bar = 10; chopt = chopt.replace('BAR', '   ');
         if (chopt[l+3] == '1') { option.Bar = 11; chopt[l+3] = ' '; }
         if (chopt[l+3] == '2') { option.Bar = 12; chopt[l+3] = ' '; }
         if (chopt[l+3] == '3') { option.Bar = 13; chopt[l+3] = ' '; }
         if (chopt[l+3] == '4') { option.Bar = 14; chopt[l+3] = ' '; }
      }
      l = chopt.indexOf('ARR' );
      if (l != -1) {
         chopt = chopt.replace('ARR', '   ');
         if (hdim > 1) {
            option.Arrow  = 1;
            option.Scat = 0;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('BOX' );
      if (l != -1) {
         chopt = chopt.replace('BOX', '   ');
         if (hdim>1) {
            Hoption.Scat = 0;
            Hoption.Box  = 1;
            if (chopt[l+3] == '1') { option.Box = 11; chopt[l+3] = ' '; }
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('COLZ');
      if (l != -1) {
         chopt = chopt.replace('COLZ', '');
         if (hdim>1) {
            option.Color  = 2;
            option.Scat   = 0;
            option.Zscale = 1;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('COL' );
      if (l != -1) {
         chopt = chopt.replace('COL', '   ');
         if (hdim>1) {
            option.Color = 1;
            option.Scat  = 0;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('CHAR'); if (l != -1) { option.Char = 1; chopt = chopt.replace('CHAR', '    '); option.Scat = 0; }
      l = chopt.indexOf('FUNC'); if (l != -1) { option.Func = 2; chopt = chopt.replace('FUNC', '    '); option.Hist = 0; }
      l = chopt.indexOf('HIST'); if (l != -1) { option.Hist = 2; chopt = chopt.replace('HIST', '    '); option.Func = 0; option.Error = 0; }
      l = chopt.indexOf('AXIS'); if (l != -1) { option.Axis = 1; chopt = chopt.replace('AXIS', '    '); }
      l = chopt.indexOf('AXIG'); if (l != -1) { option.Axis = 2; chopt = chopt.replace('AXIG', '    '); }
      l = chopt.indexOf('SCAT'); if (l != -1) { option.Scat = 1; chopt = chopt.replace('SCAT', '    '); }
      l = chopt.indexOf('TEXT');
      if (l != -1) {
         var angle = parseInt(chopt);
         if (!isNaN(angle)) {
            if (angle < 0)  angle = 0;
            if (angle > 90) angle = 90;
            option.Text = 1000 + angle;
         }
         else {
            option.Text = 1;
         }
         chopt = chopt.replace('TEXT', '    ');
         l = chopt.indexOf('N');
         if (l != -1 && histo['_typename'].match(/\bTH2Poly/)) option.Text += 3000;
         option.Scat = 0;
      }
      l = chopt.indexOf('POL');  if (l != -1) { option.System = JSROOT.Painter.Coord.kPOLAR;       chopt = chopt.replace('POL', '   '); }
      l = chopt.indexOf('CYL');  if (l != -1) { option.System = JSROOT.Painter.Coord.kCYLINDRICAL; chopt = chopt.replace('CYL', '   '); }
      l = chopt.indexOf('SPH');  if (l != -1) { option.System = JSROOT.Painter.Coord.kSPHERICAL;   chopt = chopt.replace('SPH', '   '); }
      l = chopt.indexOf('PSR');  if (l != -1) { option.System = JSROOT.Painter.Coord.kRAPIDITY;    chopt = chopt.replace('PSR', '   '); }
      l = chopt.indexOf('TRI');
      if (l != -1) {
         option.Scat = 0;
         option.Color  = 0;
         option.Tri = 1; chopt = chopt.replace('TRI', '   ');
         l = chopt.indexOf('FB');   if (l != -1) { option.FrontBox = 0; chopt = chopt.replace('FB', '  '); }
         l = chopt.indexOf('BB');   if (l != -1) { option.BackBox = 0;  chopt = chopt.replace('BB', '  '); }
         l = chopt.indexOf('ERR');  if (l != -1) chopt = chopt.replace('ERR', '   ');
      }
      l = chopt.indexOf('AITOFF');
      if (l != -1) {
         Hoption.Proj = 1; chopt = chopt.replace('AITOFF', '      ');  // Aitoff projection
      }
      l = chopt.indexOf('MERCATOR');
      if (l != -1) {
         option.Proj = 2; chopt = chopt.replace('MERCATOR', '        ');  // Mercator projection
      }
      l = chopt.indexOf('SINUSOIDAL');
      if (l != -1) {
         option.Proj = 3; chopt = chopt.replace('SINUSOIDAL', '         ');  // Sinusoidal projection
      }
      l = chopt.indexOf('PARABOLIC');
      if (l != -1) {
         option.Proj = 4; chopt = chopt.replace('PARABOLIC', '         ');  // Parabolic projection
      }
      if (option.Proj > 0) {
         option.Scat = 0;
         option.Contour = 14;
      }
      if (chopt.indexOf('A') != -1)    option.Axis = -1;
      if (chopt.indexOf('B') != -1)    option.Bar  = 1;
      if (chopt.indexOf('C') != -1)  { option.Curve =1; option.Hist = -1; }
      if (chopt.indexOf('F') != -1)    option.Fill =1;
      if (chopt.indexOf('][') != -1) { option.Off  =1; option.Hist =1; }
      if (chopt.indexOf('F2') != -1)   option.Fill =2;
      if (chopt.indexOf('L') != -1)  { option.Line =1; option.Hist = -1; }
      if (chopt.indexOf('P') != -1)  { option.Mark =1; option.Hist = -1; }
      if (chopt.indexOf('Z') != -1)    option.Zscale =1;
      if (chopt.indexOf('*') != -1)    option.Star =1;
      if (chopt.indexOf('H') != -1)    option.Hist =2;
      if (chopt.indexOf('P0') != -1)   option.Mark =10;
      if (histo['_typename'].match(/\bTH2Poly/)) {
         if (option.Fill + option.Line + option.Mark != 0 ) option.Scat = 0;
      }

      if (chopt.indexOf('E') != -1) {
         if (hdim == 1) {
            option.Error = 1;
            if (chopt.indexOf('E0') != -1)  option.Error = 10;
            if (chopt.indexOf('E1') != -1)  option.Error = 11;
            if (chopt.indexOf('E2') != -1)  option.Error = 12;
            if (chopt.indexOf('E3') != -1)  option.Error = 13;
            if (chopt.indexOf('E4') != -1)  option.Error = 14;
            if (chopt.indexOf('E5') != -1)  option.Error = 15;
            if (chopt.indexOf('E6') != -1)  option.Error = 16;
            if (chopt.indexOf('X0') != -1) {
               if (option.Error == 1)  option.Error += 20;
               option.Error += 10;
            }
            if (option.Text && histo['_typename'].match(/\bTProfile/)) {
               option.Text += 2000;
               option.Error = 0;
            }
         } else {
            if (option.Error == 0) {
               option.Error = 100;
               option.Scat  = 0;
            }
            if (option.Text) {
               option.Text += 2000;
               option.Error = 0;
            }
         }
      }
      if (chopt.indexOf('9') != -1)  option.HighRes = 1;
      if (option.Surf == 15) {
         if (option.System == JSROOT.Painter.Coord.kPOLAR || 
             option.System == JSROOT.Painter.Coord.kCARTESIAN) {
            option.Surf = 13;
            //Warning('MakeChopt','option SURF5 is not supported in Cartesian and Polar modes');
         }
      }
      if (pad && typeof(pad) != 'undefined') {
         // Copy options from current pad
         option.Logx = pad['fLogx'];
         option.Logy = pad['fLogy'];
         option.Logz = pad['fLogz'];
      } else {
         option.Logx = false;
         option.Logy = false;
         option.Logz = false;
      }
      //  Check options incompatibilities
      if (option.Bar == 1) option.Hist = -1;

      return option;
   };
   
   JSROOT.Painter.root_fonts = new Array('Arial', 'Times New Roman',
      'bold Times New Roman', 'bold italic Times New Roman',
      'Arial', 'oblique Arial', 'bold Arial', 'bold oblique Arial',
      'Courier New', 'oblique Courier New', 'bold Courier New',
      'bold oblique Courier New', 'Symbol', 'Times New Roman',
      'Wingdings', 'Symbol');
   
   JSROOT.Painter.getFontDetails = function(fontIndex) {
   
      var fontName = JSROOT.Painter.root_fonts[Math.floor(fontIndex/10)];
   
      var weight = "";
      var style = "";
      var name = "Arial";

      if (fontName==null) fontName = "";

      if (fontName.indexOf("bold") != -1) {
         weight = "bold";
         //The first 5 characters are removed because "bold " is always first when it occurs
         fontName = fontName.substring(5, fontName.length);
      }
      if (fontName.charAt(0) == 'i') {
         style = "italic";
         fontName = fontName.substring(7, fontName.length);
      }
      else if (fontName.charAt(0) == 'o') {
         style = "oblique";
         fontName = fontName.substring(8, fontName.length);
      }
      if (name == 'Symbol') {
         weight = "";
         style = "";
      }
      return {
         'weight' : weight,
         'style'  : style,
         'name'   : fontName
      };
   };

   JSROOT.Painter.createFillPattern = function (svg, pattern, color) {
      // create fill pattern - only if they don't exists yet
      var id = "pat" + pattern + "_" + color;

      if (svg.attr("id") != null) id = svg.attr("id") + "_" + id;

      if (document.getElementById(id) != null) return id;

      var line_color = JSROOT.Painter.root_colors[color];

      switch (pattern) {
         case 3001:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "3px")
               .attr("height", "2px")
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 0)
               .attr("y", 0)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 2)
               .attr("y", 0)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 1)
               .attr("y", 1)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color);
            break;
         case 3002:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px")
               .attr("height", "2px")
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 1)
               .attr("y", 0)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 3)
               .attr("y", 1)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color);
            break;
         case 3003:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px")
               .attr("height", "4px")
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 2)
               .attr("y", 1)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color)
               .append('svg:rect')
               .attr("x", 0)
               .attr("y", 3)
               .attr("width", 1)
               .attr("height", 1)
               .style("stroke", line_color);
            break;
         case 3004:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "8px")
               .attr("height", "8px")
               .style("stroke", line_color)
               .append("svg:line")
               .attr("x1", 8)
               .attr("y1", 0)
               .attr("x2", 0)
               .attr("y2", 8)
               .style("stroke", line_color)
               .style("stroke-width", 1);
            break;
         case 3005:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "8px")
               .attr("height", "8px")
               .style("stroke", line_color)
               .append("svg:line")
               .attr("x1", 0)
               .attr("y1", 0)
               .attr("x2", 8)
               .attr("y2", 8)
               .style("stroke", line_color)
               .style("stroke-width", 1);
            break;
         case 3006:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px")
               .attr("height", "4px")
               .style("stroke", line_color)
               .append("svg:line")
               .attr("x1", 1)
               .attr("y1", 0)
               .attr("x2", 1)
               .attr("y2", 3)
               .style("stroke", line_color)
               .style("stroke-width", 1);
            break;
         case 3007:
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px")
               .attr("height", "4px")
               .style("stroke", line_color)
               .append("svg:line")
               .attr("x1", 0)
               .attr("y1", 1)
               .attr("x2", 3)
               .attr("y2", 1)
               .style("stroke", line_color)
               .style("stroke-width", 1);
            break;
         default: /* == 3004 */
            svg.append('svg:pattern')
               .attr("id", id)
               .attr("patternUnits", "userSpaceOnUse")
               .attr("width", "8px")
               .attr("height", "8px")
               .style("stroke", line_color)
               .append("svg:line")
               .attr("x1", 8)
               .attr("y1", 0)
               .attr("x2", 0)
               .attr("y2", 8)
               .style("stroke", line_color)
               .style("stroke-width", 1);
            break;
      }
      return id;
   }


   JSROOT.Painter.padtoX = function(pad, x) {
      // Convert x from pad to X.
      if (pad['fLogx'] && x < 50) return Math.exp(2.302585092994 * x);
      return x;
   };

   JSROOT.Painter.padtoY = function(pad, y) {
      // Convert y from pad to Y.
      if (pad['fLogy'] && y < 50) return Math.exp(2.302585092994 * y);
      return y;
   };

   JSROOT.Painter.xtoPad = function(x, pad) {
      if (pad['fLogx']) {
         if (x > 0)
            x = JSROOT.Math.log10(x);
         else
            x = pad['fUxmin'];
      }
      return x;
   };

   JSROOT.Painter.ytoPad = function(y, pad) {
      if (pad['fLogy']) {
         if (y > 0)
            y = JSROOT.Math.log10(y);
         else
            y = pad['fUymin'];
      }
      return y;
   };

   /**
    * Converts an HSL color value to RGB. Conversion formula
    * adapted from http://en.wikipedia.org/wiki/HSL_color_space.
    * Assumes h, s, and l are contained in the set [0, 1] and
    * returns r, g, and b in the set [0, 255].
    *
    * @param   Number  h       The hue
    * @param   Number  s       The saturation
    * @param   Number  l       The lightness
    * @return  Array           The RGB representation
    */
   JSROOT.Painter.HLStoRGB = function(h, l, s) {
      var r, g, b;
      if (s < 1e-300) {
         r = g = b = l; // achromatic
      } else {
         function hue2rgb(p, q, t){
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
         }
         var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
         var p = 2 * l - q;
         r = hue2rgb(p, q, h + 1/3);
         g = hue2rgb(p, q, h);
         b = hue2rgb(p, q, h - 1/3);
      }
      return 'rgb('+Math.round(r * 255)+', '+Math.round(g * 255)+', '+Math.round(b * 255)+')';
   }

   JSROOT.Painter.chooseTimeFormat = function(range, nticks) {
      if (nticks<1) nticks = 1;
      var awidth = range / nticks;
      var reasformat = 0;

      // code from TAxis::ChooseTimeFormat
      // width in seconds ?
      if (awidth>=.5) {
         reasformat = 1;
         //  width in minutes ?
         if (awidth>=30) {
            awidth /= 60; reasformat = 2;
            //  width in hours ?
            if (awidth>=30) {
               awidth /=60; reasformat = 3;
               //  width in days ?
               if (awidth>=12) {
                  awidth /= 24; reasformat = 4;
                  // width in months ?
                  if (awidth>=15.218425) {
                     awidth /= 30.43685; reasformat = 5;
                     //  width in years ?
                     if (awidth>=6) {
                        awidth /= 12; reasformat = 6;
                        if (awidth>=2) {
                           awidth /= 12; reasformat = 7;
                        }
                     }
                  }
               }
            }
         }
      }

      switch (reasformat) {
         case 0: return "%S";
         case 1: return "%Mm%S";
         case 2: return "%Hh%M";
         case 3: return "%d-%Hh";
         case 4: return "%d/%m";
         case 5: return "%d/%m/%y";
         case 6: return "%d/%m/%y";
         case 7: return "%m/%y";
      }

      return "%Y";
   }

   JSROOT.Painter.getTimeFormat = function(axis) {
      var timeFormat = axis['fTimeFormat'];
      var idF = timeFormat.indexOf('%F');
      if (idF >= 0) return timeFormat.substr(0, idF);
      return timeFormat;
   }

   JSROOT.Painter.getTimeOffset = function(axis) {
      var timeFormat = axis['fTimeFormat'];

      var idF = timeFormat.indexOf('%F');

      if (idF >= 0) {
         var lnF = timeFormat.length;
         var stringtimeoffset = timeFormat.substr(idF+2, lnF);
         for (var i=0;i<3;++i) stringtimeoffset = stringtimeoffset.replace('-', '/');
         // special case, used from DABC painters
         if ((stringtimeoffset=="0") || (stringtimeoffset=="")) return 0;

         var stimeoffset = new Date(stringtimeoffset);
         var timeoffset = stimeoffset.getTime();
         var ids = stringtimeoffset.indexOf('s');
         if (ids >= 0) {
            var lns = stringtimeoffset.length;
            var sdp = stringtimeoffset.substr(ids+1, lns);
            var dp = parseFloat(sdp);
            timeoffset += dp;
         }
         return timeoffset;
      }

      return JSROOT.gStyle['TimeOffset'];
   }

   JSROOT.Painter.formatExp = function(label) {
      var str = label;
      if (parseFloat(str) == 1.0) return '1';
      if (parseFloat(str) == 10.0) return '10';
      var str = str.replace('e+', 'x10@');
      var str = str.replace('e-', 'x10@-');
      var _val = str.substring(0, str.indexOf('@'));
      var _exp = str.substr(str.indexOf('@'));
      _val = _val.replace('@', '');
      _exp = _exp.replace('@', '');
      var u, size = _exp.length;
      for (var j=0;j<size;++j) {
         var c = _exp.charAt(j);
         if (c == '+') u = '\u207A';
         else if (c == '-') u = '\u207B'
         else {
            var e = parseInt(c);
            if (e == 1) u = String.fromCharCode(0xB9);
            else if (e > 1 && e < 4) u = String.fromCharCode(0xB0+e);
            else u = String.fromCharCode(0x2070+e);
         }
         _exp = _exp.replace(c, u);
      }
      _val = _val.replace('1x', '');
      return _val+_exp;
   };

   JSROOT.Painter.translateExp = function(str) {
      var lstr = str.match(/\^{[0-9]*}/gi);
      if (lstr != null) {
         var symbol = '';
         for (var i=0;i<lstr.length;++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            var size = symbol.length;
            for (var j=0;j<size;++j) {
               var c = symbol.charAt(j);
               var e = parseInt(c);
               if (e == 1) u = String.fromCharCode(0xB9);
               else if (e > 1 && e < 4) u = String.fromCharCode(0xB0+e);
               else u = String.fromCharCode(0x2070+e);
               symbol = symbol.replace(c, u);
            }
            str = str.replace(lstr[i], symbol);
         }
      }
      return str;
   };

   JSROOT.Painter.symbols_map = {
         // greek letters
         '#alpha' : '\u03B1',
         '#beta' : '\u03B2',
         '#chi' : '\u03C7',
         '#delta' : '\u03B4',
         '#varepsilon' : '\u03B5',
         '#phi' : '\u03C6',
         '#gamma' : '\u03B3',
         '#eta' : '\u03B7',
         '#iota' : '\u03B9',
         '#varphi' : '\u03C6',
         '#kappa' : '\u03BA',
         '#lambda' : '\u03BB',
         '#mu' : '\u03BC',
         '#nu' : '\u03BD',
         '#omicron' : '\u03BF',
         '#pi' : '\u03C0',
         '#theta' : '\u03B8',
         '#rho' : '\u03C1',
         '#sigma' : '\u03C3',
         '#tau' : '\u03C4',
         '#upsilon' : '\u03C5',
         '#varomega' : '\u03D6',
         '#omega' : '\u03C9',
         '#xi' : '\u03BE',
         '#psi' : '\u03C8',
         '#zeta' : '\u03B6',
         '#Alpha' : '\u0391',
         '#Beta' : '\u0392',
         '#Chi' : '\u03A7',
         '#Delta' : '\u0394',
         '#Epsilon' : '\u0395',
         '#Phi' : '\u03A6',
         '#Gamma' : '\u0393',
         '#Eta' : '\u0397',
         '#Iota' : '\u0399',
         '#vartheta' : '\u03D1',
         '#Kappa' : '\u039A',
         '#Lambda' : '\u039B',
         '#Mu' : '\u039C',
         '#Nu' : '\u039D',
         '#Omicron' : '\u039F',
         '#Pi' : '\u03A0',
         '#Theta' : '\u0398',
         '#Rho' : '\u03A1',
         '#Sigma' : '\u03A3',
         '#Tau' : '\u03A4',
         '#Upsilon' : '\u03A5',
         '#varsigma' : '\u03C2',
         '#Omega' : '\u03A9',
         '#Xi' : '\u039E',
         '#Psi' : '\u03A8',
         '#Zeta' : '\u0396',
         '#varUpsilon' : '\u03D2',
         '#epsilon' : '\u03B5',
         // math symbols

         '#sqrt' : '\u221A',

         // from TLatex tables #2 & #3
         '#leq' : '\u2264',
         '#/' : '\u2044',
         '#infty' : '\u221E',
         '#voidb' : '\u0192',
         '#club' : '\u2663',
         '#diamond' : '\u2666',
         '#heart' : '\u2665',
         '#spade' : '\u2660',
         '#leftrightarrow' : '\u2194',
         '#leftarrow' : '\u2190',
         '#uparrow' : '\u2191',
         '#rightarrow' : '\u2192',
         '#downarrow' : '\u2193',
         '#circ' : '\u02C6', // ^
         '#pm' : '\xB1',
         '#doublequote' : '\u2033',
         '#geq' : '\u2265',
         '#times' : '\xD7',
         '#propto' : '\u221D',
         '#partial' : '\u2202',
         '#bullet' : '\u2022',
         '#divide' : '\xF7',
         '#neq' : '\u2260',
         '#equiv' : '\u2261',
         '#approx' : '\u2248', // should be \u2245 ?
         '#3dots' : '\u2026',
         '#cbar' : '\u007C',
         '#topbar' : '\xAF',
         '#downleftarrow' : '\u21B5',
         '#aleph' : '\u2135',
         '#Jgothic' : '\u2111',
         '#Rgothic' : '\u211C',
         '#voidn' : '\u2118',
         '#otimes' : '\u2297',
         '#oplus' : '\u2295',
         '#oslash' : '\u2205',
         '#cap' : '\u2229',
         '#cup' : '\u222A',
         '#supseteq' : '\u2287',
         '#supset' : '\u2283',
         '#notsubset' : '\u2284',
         '#subseteq' : '\u2286',
         '#subset' : '\u2282',
         '#int' : '\u222B',
         '#in' : '\u2208',
         '#notin' : '\u2209',
         '#angle' : '\u2220',
         '#nabla' : '\u2207',
         '#oright' : '\xAE',
         '#ocopyright' : '\xA9',
         '#trademark' : '\u2122',
         '#prod' : '\u220F',
         '#surd' : '\u221A',
         '#upoint' : '\u22C5',
         '#corner' : '\xAC',
         '#wedge' : '\u2227',
         '#vee' : '\u2228',
         '#Leftrightarrow' : '\u21D4',
         '#Leftarrow' : '\u21D0',
         '#Uparrow' : '\u21D1',
         '#Rightarrow' : '\u21D2',
         '#Downarrow' : '\u21D3',
         '#LT' : '\x3C',
         '#void1' : '\xAE',
         '#copyright' : '\xA9',
         '#void3' : '\u2122',
         '#sum' : '\u2211',
         '#arctop' : '',
         '#lbar' : '',
         '#arcbottom' : '',
         '#void8' : '',
         '#bottombar' : '\u230A',
         '#arcbar' : '',
         '#ltbar' : '',
         '#AA' : '\u212B',
         '#aa' : '\u00E5',
         '#void06' : '',
         '#GT' : '\x3E',
         '#forall' : '\u2200',
         '#exists' : '\u2203',
         '#bar' : '',
         '#vec' : '',
         '#dot' : '\u22C5',
         '#hat' : '\xB7',
         '#ddot' : '',
         '#acute' : '\acute',
         '#grave' : '',
         '#check' : '\u2713',
         '#tilde' : '\u02DC',
         '#slash' : '\u2044',
         '#hbar' : '\u0127',
         '#box' : '',
         '#Box' : '',
         '#parallel' : '',
         '#perp' : '\u22A5',
         '#odot' : ''
      };

   JSROOT.Painter.translateLaTeX = function(string) {
      var str = string;
      str = this.translateExp(str);
      while (str.indexOf('^{o}') != -1)
         str = str.replace('^{o}', '\xBA');
      var lstr = str.match(/\#sqrt{(.*?)}/gi);
      if (lstr != null) 
         for (var i=0;i<lstr.length;++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('#sqrt{', '#sqrt');
            symbol = symbol.replace('}', '');
            str = str.replace(lstr[i], symbol);
         }
      lstr = str.match(/\_{(.*?)}/gi);
      if (lstr != null) 
         for (var i=0;i<lstr.length;++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('_{', ''); // &sub
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      lstr = str.match(/\^{(.*?)}/gi);
      if (lstr != null) 
         for (i=0;i<lstr.length;++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      while (str.indexOf('#/') != -1)
         str = str.replace('#/', JSROOT.Painter.symbols_map['#/']);
      for (var x in JSROOT.Painter.symbols_map) {
         while (str.indexOf(x) != -1)
            str = str.replace(x, JSROOT.Painter.symbols_map[x]);
      }
      return str;
   };

   JSROOT.Painter.stringWidth = function(svg, line, font_size, fontDetails) {
      /* compute the bounding box of a string by using temporary svg:text */
      var text = 
         svg.append("svg:text")
            .attr("class", "temp_text")
            .attr("font-family", fontDetails['name'])
            .attr("font-weight", fontDetails['weight'])
            .attr("font-style", fontDetails['style'])
            .attr("font-size", font_size)
            .style("opacity", 0)
            .text(line);
      var w = text.node().getBBox().width;
      text.remove();
      return w;
   }


   // ==============================================================================

   JSROOT.TBasePainter = function()
   {
      this.vis   = null;  // canvas where object is drawn
      this.first = null;  // pointer on first painter
      this.draw_g = null;  // container for all draw objects
      this.original_view_changed = false;  // indicate that original zoom changed and one should recalculate statistic
   }

   JSROOT.TBasePainter.prototype.IsObject = function(obj) {
      return false;
   }
   
   JSROOT.TBasePainter.prototype.CheckResize = function() {
      
      // only first painter should react on resize event
      if ((this.vis==null) || (this.first!=null)) return;
      
      // see in JSROOT.draw
      
      var render_to = this.vis['render_to'];

      if ((this.last_width == $(render_to).width()) &&
         (this.last_height == $(render_to).height())) return;
      
      this.vis.attr("width", $(render_to).width())
              .attr("height", $(render_to).height())
              .attr("viewBox", "0 0 " + $(render_to).width() + " " + $(render_to).height())

      this.RedrawFrame(null, true);
   }

   JSROOT.TBasePainter.prototype.RemoveDraw = function()
   {
      // generic method to delete all graphical elements, associated with painter
      // may not work for all cases

      if (this.draw_g!=null) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   JSROOT.TBasePainter.prototype.SetFrame = function(vis, check_not_first) {

      this.frame = vis['ROOT:frame'];
      this.svg_frame = vis['ROOT:svg_frame'];
      this.pad = vis['ROOT:pad'];

      if (!('painters' in vis) || (vis['painters']==null)) {
         
         vis['painters'] = new Array();

         // only first object in list can have zoom selection
         // create dummy here and not need to check if it exists or not
         this['zoom_xmin'] = 0;
         this['zoom_xmax'] = 0;
         this['zoom_xpad'] = true; // indicate that zooming specified from pad

         this['zoom_ymin'] = 0;
         this['zoom_ymax'] = 0;
         this['zoom_ypad'] = true; // indicate that zooming specified from pad

         if (this.pad && typeof(this.pad) != 'undefined') {
            this['zoom_xmin'] = this.pad.fUxmin;
            this['zoom_xmax'] = this.pad.fUxmax;
            this['zoom_ymin'] = this.pad.fUymin;
            this['zoom_ymax'] = this.pad.fUymax;

            if (this.pad.fLogx > 0) {
               this['zoom_xmin'] = Math.exp(this['zoom_xmin']*Math.log(10));
               this['zoom_xmax'] = Math.exp(this['zoom_xmax']*Math.log(10));
            }

            if (this.pad.fLogy > 0) {
               this['zoom_ymin'] = Math.exp(this['zoom_ymin']*Math.log(10));
               this['zoom_ymax'] = Math.exp(this['zoom_ymax']*Math.log(10));
            }
         }
         
         this.last_width = vis.attr("width");
         this.last_height = vis.attr("height");

      } else {
         this.first = vis['painters'][0];
      }

      vis['painters'].push(this);
      this.vis = vis;    // remember main pad

      if (check_not_first && !this.frame) {
         alert("frame for the " +  (typeof this) + " was not specified");
         return;
      }

      if (check_not_first && !this.first) {
         alert("first painter is not defined - it is a must for " + (typeof this));
         return;
      }
   }

   JSROOT.TBasePainter.prototype.RedrawFrame = function(selobj, resize) {
      // call Redraw methods for each painter in the frame
      // if selobj specified, painter with selected object will be redrawn

      if (!this.vis) return;
      if (! 'painters' in this.vis) return;

      for (var n=0;n<this.vis['painters'].length;n++) {

         var painter = this.vis['painters'][n];

         if ((selobj!=null) && !painter.IsObject(selobj)) continue;

         painter.Redraw(resize);
      }
   }

   JSROOT.TBasePainter.prototype.ClearFrame = function() {

      var vis = this.vis; 

      if (!vis) return;
      
      for (var n=0;n<vis['painters'].length;n++) {

         var painter = vis['painters'][n];
         
         painter.RemoveDraw();
         painter.vis = null;
         painter.frame = null;
         painter.pad = null;
      }
      
      delete vis['painters'];
      vis['painters'] = null;

      while (vis.lastChild) {
         vis.removeChild(vis.lastChild);
     }
      
      vis.selectAll("*").remove();

      vis.id = "cleanup_id";
      
      $("#cleanup_id").empty();

   }


   JSROOT.TBasePainter.prototype.RemoveDrag = function(id)
   {
      var drag_rect_name = id + "_drag_rect";
      var resize_rect_name = id + "_resize_rect";
      if (this[drag_rect_name]) { this[drag_rect_name].remove(); this[drag_rect_name] = null; }
      if (this[resize_rect_name]) { this[resize_rect_name].remove(); this[resize_rect_name] = null; }
   }

   JSROOT.TBasePainter.prototype.AddDrag = function(id, main_rect, callback) {

      var pthis = this;

      var drag_rect_name = id + "_drag_rect";
      var resize_rect_name = id + "_resize_rect";

      var istitle = (main_rect.node().tagName == "text");

      var rect_width = function() {
         if (istitle)
            return main_rect.node().getBBox().width;
         else
            return Number(main_rect.attr("width"));
      }

      var rect_height = function() {
         if (istitle)
            return main_rect.node().getBBox().height;
         else
            return Number(main_rect.attr("height"));
      }

      var rect_x = function() {
         var x = Number(main_rect.attr("x"));
         if (istitle) x -= rect_width()/2;
         return x;
      }

      var rect_y = function() {
         var y = Number(main_rect.attr("y"));
         if (istitle) y -= rect_height();
         return y;
      }

      var drag_move = d3.behavior.drag()
          .origin(Object)
          .on("dragstart", function() {
             d3.event.sourceEvent.preventDefault();

             pthis[drag_rect_name] =
                pthis.vis.append("rect")
                .attr("class", "zoom")
                .attr("id", drag_rect_name)
                .attr("x", rect_x())
                .attr("y", rect_y())
                .attr("width", rect_width())
                .attr("height", rect_height())
                .style("cursor", "move");
          })
          .on("drag", function() {
             d3.event.sourceEvent.preventDefault();
             pthis[drag_rect_name].attr("x", Number(pthis[drag_rect_name].attr("x")) + d3.event.dx);
             pthis[drag_rect_name].attr("y", Number(pthis[drag_rect_name].attr("y")) + d3.event.dy);
             d3.event.sourceEvent.stopPropagation();
          })
          .on("dragend", function() {
             d3.event.sourceEvent.preventDefault();

             pthis[drag_rect_name].style("cursor", "auto");

             var x = Number(pthis[drag_rect_name].attr("x"));
             var y = Number(pthis[drag_rect_name].attr("y"));

             var dx = x - rect_x();
             var dy = y - rect_y();

             pthis[drag_rect_name].remove();
             pthis[drag_rect_name] = null;

             if (istitle) {
                main_rect.attr("x", x + rect_width()/2).attr("y", y + rect_height());
             } else {
                main_rect.attr("x", x).attr("y", y);
                pthis[resize_rect_name]
                   .attr("x", x + rect_width() - 20)
                   .attr("y", y + rect_height() - 20);
             }

             callback.move(x, y, dx, dy);

             // do it after call-back - rectangle has correct coordinates
             if (istitle)
                pthis[resize_rect_name]
                  .attr("x", rect_x() + rect_width() - 20)
                  .attr("y", rect_y() + rect_height() - 20);
          });

      var drag_resize =
         d3.behavior.drag()
          .origin(Object)
          .on("dragstart", function() {
             d3.event.sourceEvent.preventDefault();
             pthis[drag_rect_name] =
                pthis.vis.append("rect")
                .attr("class", "zoom")
                .attr("id", drag_rect_name)
                .attr("x", rect_x())
                .attr("y", rect_y())
                .attr("width", rect_width())
                .attr("height", rect_height())
                .style("cursor", "se-resize");

             // main_rect.style("cursor", "move");
          })
          .on("drag", function() {
             d3.event.sourceEvent.preventDefault();
             pthis[drag_rect_name].attr("width", Number(pthis[drag_rect_name].attr("width")) + d3.event.dx);
             pthis[drag_rect_name].attr("height", Number(pthis[drag_rect_name].attr("height")) + d3.event.dy);
             d3.event.sourceEvent.stopPropagation();
          })
          .on("dragend", function() {
             d3.event.sourceEvent.preventDefault();
             pthis[drag_rect_name].style("cursor", "auto");

             var newwidth = Number(pthis[drag_rect_name].attr("width"));
             var newheight = Number(pthis[drag_rect_name].attr("height"));

             pthis[drag_rect_name].remove();
             pthis[drag_rect_name] = null;

             callback.resize(newwidth, newheight);

             // do it after call-back - rectangle has correct coordinates
             if (istitle)
                pthis[resize_rect_name]
                  .attr("x", rect_x() + rect_width() - 20)
                  .attr("y", rect_y() + rect_height() - 20);

             // console.log(resize_rect_name + " x = " + rect_x() + "  width = " + rect_width());
             // console.log(resize_rect_name + " y = " + rect_y() + "  height = " + rect_height());
          });


      if (this[resize_rect_name] == null) {

         main_rect.call(drag_move);

         this[resize_rect_name] =
            this.vis.append("rect")
              .attr("id", resize_rect_name)
              .style("opacity", "0")
              .style("cursor", "se-resize")
              .call(drag_resize);
      } else {
         // ensure that small resize rect appears after large move rect
         var prnt = this[resize_rect_name].node().parentNode;

         // first move small resize rec before main_rect
         prnt.removeChild(this[resize_rect_name].node());
         prnt.insertBefore(this[resize_rect_name].node(), main_rect.node());
         // than swap them while big rect should be in the front
         prnt.removeChild(main_rect.node());
         prnt.insertBefore(main_rect.node(), this[resize_rect_name].node());
      }

      this[resize_rect_name]
          .attr("x", rect_x() + rect_width() - 20)
          .attr("y", rect_y() + rect_height() - 20)
          .attr("width", 20)
          .attr("height", 20);
   }


   JSROOT.TBasePainter.prototype.FindPainterFor = function(selobj)
   {
      // try to find painter for sepcified object
      // can be used to find painter for some special objects, registered as histogram functions

      if (!this.vis) return null;

      for (var n=0;n<this.vis['painters'].length;n++) {
         var painter = this.vis['painters'][n];

         if (painter.IsObject(selobj)) return painter;
      }

      return null;
   }

   JSROOT.TBasePainter.prototype.PlacePainterAfterMe = function(next) {
      if (!this.vis) return;

      var arr = this.vis['painters'];

      var indx1 = arr.indexOf(this);
      var indx2 = arr.indexOf(next);

      if ((indx1>=0) && (indx2>=0) && (indx2 != indx1+1)) {
         arr.splice(indx2, 1); // remove
         if (indx2<indx1) indx1--;
         arr.splice(indx1+1, 0, next);
      }

   }

   JSROOT.TBasePainter.prototype.Redraw = function() {
      // basic method, should be reimplemented in all derived objects
      // for the case when drawing should be repeated, probably with different options
   }


   JSROOT.TBasePainter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.Painter.menuitem(menu,"Unzoom X", function() { menu['painter'].Unzoom(true, false); } );
      JSROOT.Painter.menuitem(menu,"Unzoom Y", function() { menu['painter'].Unzoom(false, true); } );
      JSROOT.Painter.menuitem(menu,"Unzoom", function() { menu['painter'].Unzoom(true, true); } );
      
      JSROOT.Painter.menuitem(menu, JSROOT.gStyle.Tooltip ? "Disable tooltip" : "Enable tooltip", 
            function() { 
               JSROOT.gStyle.Tooltip = !JSROOT.gStyle.Tooltip;
               menu['painter'].RedrawFrame();
            });
   }

   JSROOT.TBasePainter.prototype.Unzoom = function(dox,doy) {
      var obj = this.first;
      if (!obj) obj = this;

      var changed = false;

      if (dox) {
         if (obj['zoom_xmin'] != obj['zoom_xmax']) changed = true;
         obj['zoom_xmin'] = 0;
         obj['zoom_xmax'] = 0;
      }
      if (doy) {
         if (obj['zoom_ymin'] != obj['zoom_ymax']) changed = true;
         obj['zoom_ymin'] = 0;
         obj['zoom_ymax'] = 0;
      }

      if (changed) {
         obj.original_view_changed = true;
         this.RedrawFrame();
      }
   }

   JSROOT.TBasePainter.prototype.Zoom = function(xmin, xmax, ymin, ymax) {
      var obj = this.first;
      if (!obj) obj = this;
      if (xmin!=xmax) {
         obj['zoom_xmin'] = xmin;
         obj['zoom_xmax'] = xmax;
      }
      if (ymin!=ymax) {
         obj['zoom_ymin'] = ymin;
         obj['zoom_ymax'] = ymax;
      }
      obj.original_view_changed = true;
      this.RedrawFrame();
   }

   JSROOT.TBasePainter.prototype.UpdateObject = function(obj) {
      alert("JSROOT.TBasePainter.UpdateObject not implemented");
      return false;
   }


   /**
    * Now the real drawing functions (using d3.js)
    */

   JSROOT.Painter.add3DInteraction = function(renderer, scene, camera, toplevel, vis) {
      // add 3D mouse interactive functions
      var mouseX, mouseY, mouseDowned = false;
      var mouse = { x: 0, y: 0 }, INTERSECTED;
      
      var tooltip = function() {
         var id = 'tt';
         var top = 3;
         var left = 3;
         var maxw = 150;
         var speed = 10;
         var timer = 20;
         var endalpha = 95;
         var alpha = 0;
         var tt,t,c,b,h;
         var ie = document.all ? true : false;
         return {
            show: function(v, w) {
               if (tt == null) {
                  tt = document.createElement('div');
                  tt.setAttribute('id',id);
                  t = document.createElement('div');
                  t.setAttribute('id',id + 'top');
                  c = document.createElement('div');
                  c.setAttribute('id',id + 'cont');
                  b = document.createElement('div');
                  b.setAttribute('id',id + 'bot');
                  tt.appendChild(t);
                  tt.appendChild(c);
                  tt.appendChild(b);
                  document.body.appendChild(tt);
                  tt.style.opacity = 0;
                  tt.style.filter = 'alpha(opacity=0)';
                  document.onmousemove = this.pos;
               }
               tt.style.display = 'block';
               c.innerHTML = v;
               tt.style.width = w ? w + 'px' : 'auto';
               tt.style.width = 'auto'; // let it be automatically resizing...
               if (!w && ie) {
                  t.style.display = 'none';
                  b.style.display = 'none';
                  tt.style.width = tt.offsetWidth;
                  t.style.display = 'block';
                  b.style.display = 'block';
               }
               //if (tt.offsetWidth > maxw) { tt.style.width = maxw + 'px'; }
               h = parseInt(tt.offsetHeight) + top;
               clearInterval(tt.timer);
               tt.timer = setInterval( function() { tooltip.fade(1) }, timer );
            },
            pos: function(e) {
               var u = ie ? event.clientY + document.documentElement.scrollTop : e.pageY;
               var l = ie ? event.clientX + document.documentElement.scrollLeft : e.pageX;
               tt.style.top = u + 15 + 'px';//(u - h) + 'px';
               tt.style.left = (l + left) + 'px';
            },
            fade: function(d) {
               var a = alpha;
               if ((a != endalpha && d == 1) || (a != 0 && d == -1)) {
                  var i = speed;
                  if (endalpha - a < speed && d == 1) {
                     i = endalpha - a;
                  }
                  else if (alpha < speed && d == -1) {
                     i = a;
                  }
                  alpha = a + (i * d);
                  tt.style.opacity = alpha * .01;
                  tt.style.filter = 'alpha(opacity=' + alpha + ')';
               }
               else {
                  clearInterval(tt.timer);
                  if (d == -1) { tt.style.display = 'none'; }
               }
            },
            hide: function() {
               if (tt == null) return;
               clearInterval(tt.timer);
               tt.timer = setInterval( function() { tooltip.fade(-1) }, timer );
            }
         };
      }();

      var radius = 100;
      var theta = 0;
      var projector = new THREE.Projector();
      function findIntersection() {
         // find intersections
         if ( mouseDowned ) {
            if ( INTERSECTED ) {
               INTERSECTED.material.emissive.setHex( INTERSECTED.currentHex );
               renderer.render( scene, camera );
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip) tooltip.hide();
            return;
         }
         var vector = new THREE.Vector3( mouse.x, mouse.y, 1 );
         projector.unprojectVector( vector, camera );
         var raycaster = new THREE.Raycaster( camera.position, vector.sub( camera.position ).normalize() );
         var intersects = raycaster.intersectObjects( scene.children, true );
         if ( intersects.length > 0 ) {
            var pick = null;
            for (var i=0;i<intersects.length;++i) {
               if ('emissive' in intersects[ i ].object.material) {
                  pick = intersects[ i ];
                  break;
               }
            }
            if (pick && INTERSECTED != pick.object ) {
               if ( INTERSECTED ) INTERSECTED.material.emissive.setHex( INTERSECTED.currentHex );
               INTERSECTED = pick.object;
               INTERSECTED.currentHex = INTERSECTED.material.emissive.getHex();
               INTERSECTED.material.emissive.setHex( 0x5f5f5f );
               renderer.render( scene, camera );
               if (JSROOT.gStyle.Tooltip) tooltip.show(INTERSECTED.name.length > 0 ? INTERSECTED.name : INTERSECTED.parent.name, 200);
            }
         } else {
            if ( INTERSECTED ) {
               INTERSECTED.material.emissive.setHex( INTERSECTED.currentHex );
               renderer.render( scene, camera );
            }
            INTERSECTED = null;
            if (JSROOT.gStyle.Tooltip) tooltip.hide();
         }
      };

      $( renderer.domElement ).on('touchstart mousedown',function (e) {
         //var touch = e.changedTouches[0] || {};
         if (JSROOT.gStyle.Tooltip) tooltip.hide();
         e.preventDefault();
         var touch = e;
         if ('changedTouches' in e) touch = e.changedTouches[0];
         else if ('touches' in e) touch = e.touches[0];
         else if ('originalEvent' in e) {
            if ('changedTouches' in e.originalEvent) touch = e.originalEvent.changedTouches[0];
            else if ('touches' in e.originalEvent) touch = e.originalEvent.touches[0];
         }
         mouseX = touch.pageX;
         mouseY = touch.pageY;
         mouseDowned = true;
      });
      $( renderer.domElement ).on('touchmove mousemove', function(e) {
         if ( mouseDowned ) {
            var touch = e;
            if ('changedTouches' in e) touch = e.changedTouches[0];
            else if ('touches' in e) touch = e.touches[0];
            else if ('originalEvent' in e) {
               if ('changedTouches' in e.originalEvent) touch = e.originalEvent.changedTouches[0];
               else if ('touches' in e.originalEvent) touch = e.originalEvent.touches[0];
            }
            var moveX = touch.pageX - mouseX;
            var moveY = touch.pageY - mouseY;
            // limited X rotate in -45 to 135 deg
            if ( (moveY > 0 && toplevel.rotation.x < Math.PI*3/4) ||
                 (moveY < 0 &&  toplevel.rotation.x > -Math.PI/4) ) {
               toplevel.rotation.x += moveY*0.02;
            }
            toplevel.rotation.y += moveX*0.02;
            renderer.render( scene, camera );
            mouseX = touch.pageX;
            mouseY = touch.pageY;
         }
         else {
            e.preventDefault();
            var mouse_x = 'offsetX' in e.originalEvent ? e.originalEvent.offsetX : e.originalEvent.layerX;
            var mouse_y = 'offsetY' in e.originalEvent ? e.originalEvent.offsetY : e.originalEvent.layerY;
            mouse.x = ( mouse_x / renderer.domElement.width ) * 2 - 1;
            mouse.y = - ( mouse_y / renderer.domElement.height ) * 2 + 1;
            // enable picking once tootips are available...
            findIntersection();
         }
      });
      $( renderer.domElement ).on('touchend mouseup', function(e) {
         mouseDowned = false;
      });

      $( renderer.domElement ).on('mousewheel', function(e, d) {
         e.preventDefault();
         camera.position.z += d * 20;
         renderer.render( scene, camera );
      });
      
      $( renderer.domElement ).on('contextmenu', function(e) {
         
         e.preventDefault();

         if (JSROOT.gStyle.Tooltip) tooltip.hide();
         
         var menu = JSROOT.Painter.createmenu(e.originalEvent);

         
         var item = JSROOT.gStyle.Tooltip ? "Disable tooltip" : "Enable tooltip";
         
         JSROOT.Painter.menuitem(menu, item, function() {
            JSROOT.gStyle.Tooltip = !JSROOT.gStyle.Tooltip;
            tooltip.hide();
         });
         
         if ('painters' in vis)
            JSROOT.Painter.menuitem(menu,"Switch to 2D", function() {
               $( vis[0][0] ).show().parent().find( renderer.domElement ).remove();
            });
         JSROOT.Painter.menuitem(menu,"Close", function() { });
         
      });
   };

   JSROOT.Painter.createFrame = function(vis, frame, shrink_right) {
      var width = Number(vis.attr("width")), height = Number(vis.attr("height"));
      var w = width, h = height;
      
      if (! ('NDC' in vis)) vis['NDC'] = { x1: 0.12, y1: 0.12, x2: 0.95, y2: 0.88 };

      if (shrink_right==null) shrink_right = 0.;

      var lm = width * vis['NDC'].x1;
      var rm = width * (1 - vis['NDC'].x2 + shrink_right);
      var tm = height * vis['NDC'].y1;
      var bm = height * (1-vis['NDC'].y2);
      
      var pad = vis['ROOT:pad'];

      var framecolor = 'white', bordermode = 0,
          bordersize = 0, linecolor = 'black',
          linestyle = 0, linewidth = 1;

      if (frame!=null) {
         bordermode = frame['fBorderMode'];
         bordersize = frame['fBorderSize'];
         linecolor = JSROOT.Painter.root_colors[frame['fLineColor']];
         linestyle = frame['fLineStyle'];
         linewidth = frame['fLineWidth'];
         if (pad) {
            var xspan = width / Math.abs(pad['fX2'] - pad['fX1']);
            var yspan = height / Math.abs(pad['fY2'] - pad['fY1']);
            var px1 = (frame['fX1'] - pad['fX1']) * xspan;
            var py1 = (frame['fY1'] - pad['fY1']) * yspan;
            var px2 = (frame['fX2'] - pad['fX1']) * xspan;
            var py2 = (frame['fY2'] - pad['fY1']) * yspan;
            var pxl, pxt, pyl, pyt;
            if (px1 < px2) { pxl = px1; pxt = px2; }
            else           { pxl = px2; pxt = px1; }
            if (py1 < py2) { pyl = py1; pyt = py2; }
            else           { pyl = py2; pyt = py1; }
            lm = pxl;
            bm = pyl;
            w = pxt - pxl;
            h = pyt - pyl;
            tm = height - pyt;
            rm = width - pxt;
         }
         else {
            lm = frame['fX1'] * width;
            tm = frame['fY1'] * height;
            bm = (1.0 - frame['fY2']) * height;
            rm = (1.0 - frame['fX2'] + shrink_right) * width;
            w -= (lm + rm);
            h -= (tm + bm);
         }
         framecolor = JSROOT.Painter.root_colors[frame['fFillColor']];
         if (frame['fFillStyle'] > 4000 && frame['fFillStyle'] < 4100)
            framecolor = 'none';
      }
      else {
         if (pad) {
            framecolor = JSROOT.Painter.root_colors[pad['fFrameFillColor']];
            if (pad['fFrameFillStyle'] > 4000 && pad['fFrameFillStyle'] < 4100)
               framecolor = 'none';
         }
         w -= (lm + rm);
         h -= (tm + bm);
      }
      if (typeof(framecolor) == 'undefined')
         framecolor = 'white';

      // this is svg:g object - container for every other items belonging to frame
      var hframe = vis['ROOT:frame'];
      if (hframe == null) {
         hframe = vis.append("svg:g");
         vis['ROOT:frame'] = hframe;
      }
      
      // calculate actual NDC coordinates, use them to properly locate PALETTE 
      vis['NDC'] = { x1: lm / width, x2: (lm+w) / width, y1: tm / height, y2: (tm+h) / height }; 

      hframe.attr("x", lm)
            .attr("y", tm)
            .attr("width", w)
            .attr("height", h)
            .attr("transform", "translate(" + lm + "," + tm + ")");
      
      var top_rect = vis['ROOT:top_rect'];
      if (top_rect==null) {
         top_rect = hframe.append("svg:rect");
         vis['ROOT:top_rect'] = top_rect;
      }

      top_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .attr("fill", framecolor)
              .style("stroke", linecolor)
              .style("stroke-width", linewidth);

      var svg_frame = vis['ROOT:svg_frame'];
      if (svg_frame == null) {
         svg_frame = hframe.append("svg");
         vis['ROOT:svg_frame'] = svg_frame;
      }
      
      svg_frame.attr("x", 0)
               .attr("y", 0)
               .attr("width", w)
               .attr("height", h)
               .attr("viewBox", "0 0 "+w+" "+h);
   }

   JSROOT.Painter.shrinkFrame = function(vis, fact)
   {
      if (vis == null) return 0;
      
      JSROOT.Painter.createFrame(vis, null, fact);
   }

   // =========================================================================

   JSROOT.TF1Painter = function(tf1) {
      JSROOT.TBasePainter.call(this);
      this.tf1 = tf1;
   }

   JSROOT.TF1Painter.prototype = Object.create( JSROOT.TBasePainter.prototype );

   JSROOT.TF1Painter.prototype.IsObject = function(obj) {
      return this.tf1 === obj;
   }

   JSROOT.TF1Painter.prototype.Redraw = function()
   {
      this.DrawBins();
   }

   JSROOT.TF1Painter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.TBasePainter.prototype.FillContextMenu.call(this, menu);
   }

   JSROOT.TF1Painter.prototype.Eval = function(x)
   {
      //return (x-5)*(x-5);
      return this.tf1.evalPar(x);
   }

   JSROOT.TF1Painter.prototype.CreateDummyHisto = function()
   {
      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
      if (this.tf1['fNsave'] > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var nb_points = this.tf1['fNpx'];
         for (var i=0;i<nb_points;++i) {
            var h = this.tf1['fSave'][i];
            if ((i==0) || (h > ymax)) ymax = h;
            if ((i==0) || (h < ymin)) ymin = h;
         }
         xmin = this.tf1['fSave'][nb_points+1];
         xmax = this.tf1['fSave'][nb_points+2];
      } else {
         // we don't have the points, so let's try to interpret the function
         // use fNpfits instead of fNpx if possible (to use more points)
         if (this.tf1['fNpfits'] <= 103) this.tf1['fNpfits'] = 103;
         xmin = this.tf1['fXmin'];
         xmax = this.tf1['fXmax'];

         var nb_points = Math.max(this.tf1['fNpx'], this.tf1['fNpfits']);

         var binwidthx = (xmax - xmin) / nb_points;
         var left = -1, right = -1;
         for (var i=0;i<nb_points;++i) {
            var h = this.Eval(xmin + (i * binwidthx));
            if (isNaN(h)) continue;

            if (left<0) { left = i; ymax = h; ymin = h; }
            if ((right<0) || (right == i-1)) right = i;

            if (h > ymax) ymax = h;
            if (h < ymin) ymin = h;
         }

         if (left<right) {
            xmax = xmin + right*binwidthx;
            xmin = xmin + left*binwidthx;
         }
      }

      if (ymax > 0.0) ymax *= 1.05;
      if (ymin < 0.0) ymin *= 1.05;

      var histo = JSROOT.CreateTH1();

      histo['fName'] = this.tf1['fName'] + "_hist";
      histo['fTitle'] = this.tf1['fTitle'];

      histo['fXaxis']['fXmin'] = xmin;
      histo['fXaxis']['fXmax'] = xmax;
      histo['fYaxis']['fXmin'] = ymin;
      histo['fYaxis']['fXmax'] = ymax;

      return histo;
   }


   JSROOT.TF1Painter.prototype.CreateBins = function() {

      var pthis = this;

      if (this.tf1['fNsave'] > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var nb_points = this.tf1['fNpx'];

         var xmin = this.tf1['fSave'][nb_points+1];
         var xmax = this.tf1['fSave'][nb_points+2];
         var binwidthx = (xmax - xmin) / nb_points;

         this['bins'] = d3.range(nb_points).map(function(p) {
            return {
               x: xmin + (p * binwidthx),
               y: pthis.tf1['fSave'][p]
            };
         });
         this['interpolate_method'] = 'monotone';
      }
      else {
         if (this.tf1['fNpfits'] <= 103) this.tf1['fNpfits'] = 333;
         var xmin = this.tf1['fXmin'];
         var xmax = this.tf1['fXmax'];
         var nb_points = Math.max(this.tf1['fNpx'], this.tf1['fNpfits']);
         var binwidthx = (xmax - xmin) / nb_points;
         this['bins'] = d3.range(nb_points).map(function(p) {
            var xx = xmin + (p * binwidthx);
            var yy = pthis.Eval(xx);
            if (isNaN(yy)) yy = 0;
            return {
               x: xx,
               y: yy
            };
         });
         this['interpolate_method'] = 'cardinal-open';
      }
   }

   JSROOT.TF1Painter.prototype.DrawBins = function()
   {
      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));

      this.RemoveDraw();

      this.draw_g = this.first.svg_frame.append("svg:g");

      var pthis = this;
      var x = this.first.x;
      var y = this.first.y;

      var linecolor = JSROOT.Painter.root_colors[this.tf1['fLineColor']];
      if ((this.tf1['fLineColor'] == 0) || (this.tf1['fLineWidth'] == 0)) linecolor = "none";

      var fillcolor = JSROOT.Painter.root_colors[this.tf1['fFillColor']];
      if ((this.tf1['fFillStyle'] == 0) || (this.tf1['fFillColor'] == 0)) fillcolor = "none"; else
      if (this.tf1['fFillStyle'] > 3000 && this.tf1['fFillStyle'] <= 3025) {
         var patternid = JSROOT.Painter.createFillPattern(this.vis, this.tf1['fFillStyle'], this.tf1['fFillColor']);
         fillcolor = "url(#" + patternid + ")";
      }

      var line = d3.svg.line()
         .x(function(d) { return x(d.x);})
         .y(function(d) { return y(d.y);})
         .interpolate(this.interpolate_method);

      var area = d3.svg.area()
         .x(function(d) { return x(d.x); })
         .y1(h)
         .y0(function(d) { return y(d.y); });

      if (linecolor!="none")
         this.draw_g.append("svg:path")
           .attr("class", "line")
           .attr("d", line(pthis.bins))
           .style("stroke", linecolor)
           .style("stroke-width", pthis.tf1['fLineWidth'])
           .style("stroke-dasharray", JSROOT.Painter.root_line_styles[pthis.tf1['fLineStyle']])
           .style("fill", "none");

      if (fillcolor!="none")
         this.draw_g.append("svg:path")
           .attr("class", "area")
           .attr("d", area(pthis.bins))
           .style("stroke", "none")
           .style("fill", fillcolor)
           .style("antialias", "false");


      // add tooltips
      if (JSROOT.gStyle.Tooltip)
         this.draw_g.selectAll("line")
            .data(this.bins)
            .enter()
            .append("svg:circle")
            .attr("cx", function(d) { return x(d.x); })
            .attr("cy", function(d) { return y(d.y); })
            .attr("r", 4)
            .attr("opacity", 0)
            .append("svg:title").text(function(d) {
               return "x = " + d.x.toPrecision(4) + " \ny = " + d.y.toPrecision(4);
            });
   }

   JSROOT.TF1Painter.prototype.UpdateObject = function(obj) {
      if (obj['_typename'] != this.tf1['_typename']) return false;
      // TODO: realy update object content
      this.tf1 = obj;
      this.CreateBins();
      return true;
   }


   JSROOT.Painter.drawFunction = function(vis, tf1)
   {
      var painter = new JSROOT.TF1Painter(tf1);

      if (!('painters' in vis)) {
         var histo = painter.CreateDummyHisto();
         JSROOT.Painter.drawHistogram1D(vis, histo);
      }

      painter.SetFrame(vis, true);

      painter.CreateBins();

      painter.DrawBins();

      return painter;
   }


   // =======================================================================

   JSROOT.TGraphPainter = function(graph) {
      JSROOT.TBasePainter.call(this);
      this.graph = graph;
   }

   JSROOT.TGraphPainter.prototype = Object.create( JSROOT.TBasePainter.prototype );

   JSROOT.TGraphPainter.prototype.IsObject = function(obj) {
      return this.graph === obj;
   }

   JSROOT.TGraphPainter.prototype.Redraw = function()
   {
      this.DrawBins();
   }

   JSROOT.TGraphPainter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.TBasePainter.prototype.FillContextMenu.call(this, menu);
   }


   JSROOT.TGraphPainter.prototype.DecodeOptions = function(opt) {
      this.logx = false;
      this.logy = false;
      this.logz = false;
      this.gridx = false;
      this.gridy = false;
      this.draw_all = true;
      this.optionLine  = 0;
      this.optionAxis  = 0;
      this.optionCurve = 0;
      this.optionStar  = 0;
      this.optionMark  = 0;
      this.optionBar   = 0;
      this.optionR     = 0;
      this.optionOne   = 0;
      this.optionE     = 0;
      this.optionFill  = 0;
      this.optionZ     = 0;
      this.optionCurveFill = 0;
      this.draw_errors = true;
      this.optionNone  = 0; // no any drawing
      this.opt = "LP";

      if ((opt!=null) && (opt!="")) {
         this.opt = opt.toUpperCase();
         this.opt.replace('SAME', '');
      }

      if (this.opt.indexOf('L') != -1) this.optionLine = 1;
      if (this.opt.indexOf('A') != -1) this.optionAxis = 1;
      if (this.opt.indexOf('C') != -1) this.optionCurve= 1;
      if (this.opt.indexOf('*') != -1) this.optionStar = 1;
      if (this.opt.indexOf('P') != -1) this.optionMark = 1;
      if (this.opt.indexOf('B') != -1) this.optionBar  = 1;
      if (this.opt.indexOf('R') != -1) this.optionR    = 1;
      if (this.opt.indexOf('1') != -1) this.optionOne  = 1;
      if (this.opt.indexOf('F') != -1) this.optionFill = 1;
      if (this.opt.indexOf('2') != -1 || this.opt.indexOf('3') != -1 ||
          this.opt.indexOf('4') != -1 || this.opt.indexOf('5') != -1) this.optionE = 1;

      // if no drawing option is selected and if opt<>' ' nothing is done.
      if (this.optionLine + this.optionFill + this.optionCurve + this.optionStar +
          this.optionMark + this.optionBar + this.optionE == 0) {
         if (this.opt.length == 0)
            this.optionLine = 1;
         else {
            this.optionNone = 1;
            return;
         }
      }
      if (this.optionStar) this.graph['fMarkerStyle'] = 3;

      if (this.optionCurve && this.optionFill) {
         this.optionCurveFill = 1;
         this.optionFill      = 0;
      }
      if (this.pad && typeof(this.pad) != 'undefined') {
         this.logx  = this.pad['fLogx'];
         this.logy  = this.pad['fLogy'];
         this.logz  = this.pad['fLogz'];
         this.gridx = this.pad['fGridx'];
         this.gridy = this.pad['fGridy'];
      }

      this.xaxis_type = this.logx ? 'logarithmic' : 'linear';
      this.yaxis_type = this.logy ? 'logarithmic' : 'linear';

      if (this.graph['_typename'] == 'TGraph') {
         // check for axis scale format, and convert if required
         if (this.graph['fHistogram']['fXaxis']['fTimeDisplay']) {
            this.xaxis_type = 'datetime';
         }

         if (this.graph['fHistogram']['fYaxis']['fTimeDisplay']) {
            this.yaxis_type = 'datetime';
         }
      }
      else if (this.graph['_typename'] == 'TGraphErrors') {
         this.maxEX = d3.max(this.graph['fEX']);
         this.maxEY = d3.max(this.graph['fEY']);
         if (this.maxEX < 1.0e-300 && this.maxEY < 1.0e-300)
            this.draw_errors = false;
      }
      this.seriesType = 'scatter';
      if (this.optionBar == 1)
         this.seriesType = 'bar';
      this.showMarker = false;
      if (this.optionMark == 1 || this.optionStar == 1)
         this.showMarker = true;

      if (this.optionLine == 1 || this.optionCurve == 1 || this.optionFill == 1)
         this.seriesType = 'line';

      if (this.optionBar == 1) {
         this.binwidthx = (this.graph['fHistogram']['fXaxis']['fXmax'] -
                           this.graph['fHistogram']['fXaxis']['fXmin']) /
                          (this.graph['fNpoints']-1);
      }
   }

   JSROOT.TGraphPainter.prototype.CreateBins = function() {
      var pthis = this;

      this.bins = d3.range(this.graph['fNpoints']).map(function(p) {
         if (pthis.optionBar == 1) {
            return {
               x: pthis.graph['fX'][p] - (pthis.binwidthx / 2),
               y: pthis.graph['fY'][p], // graph['fHistogram']['fXaxis']['fXmin'],
               bw: pthis.binwidthx,
               bh: pthis.graph['fY'][p]
            }
         }
         else if (pthis.graph['_typename'] == 'TGraphErrors') {
            return {
               x: pthis.graph['fX'][p],
               y: pthis.graph['fY'][p],
               exlow:  pthis.graph['fEX'][p],
               exhigh: pthis.graph['fEX'][p],
               eylow:  pthis.graph['fEY'][p],
               eyhigh: pthis.graph['fEY'][p]
            };
         }
         else if (pthis.graph['_typename'] == 'TGraphAsymmErrors' ||
                  pthis.graph['_typename'].match(/\bRooHist/)) {
            return {
               x: pthis.graph['fX'][p],
               y: pthis.graph['fY'][p],
               exlow:  pthis.graph['fEXlow'][p],
               exhigh: pthis.graph['fEXhigh'][p],
               eylow:  pthis.graph['fEYlow'][p],
               eyhigh: pthis.graph['fEYhigh'][p]
            };
         }
         else {
            return {
               x: pthis.graph['fX'][p],
               y: pthis.graph['fY'][p]
            };
         }
      });


      this.bins_lw = this.graph['fLineWidth']; // line width

      this.exclusionGraph = false;
      if (this.bins_lw <= 99) return;

      // special handling of exclusion graphs

      this.exclusionGraph = true;

      var ec, ff = 1;

      var normx, normy;
      var n = this.graph['fNpoints'];
      var glw = this.graph['fLineWidth'],
          xo = new Array(n+2), yo = new Array(n+2),
          xt = new Array(n+2), yt = new Array(n+2),
          xf = new Array(2*n+2), yf = new Array(2*n+2);
      // negative value means another side of the line...
      if (glw > 32767) {
         glw = 65536 - glw;
      }
      this.bins_lw = glw % 100; // line width
      if (this.bins_lw > 0) this.optionLine = 1;
      ec = JSROOT.Painter.root_colors[this.graph['fFillColor']];
      ec = ec.replace('rgb', 'rgba');
      ec = ec.replace(')', ', 0.20)');

      var a, i, j, nf, wk = (glw/100)*0.005;
      if (this.graph['fLineWidth'] > 32767)
         wk *= -1;

      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));

      var ratio = w / h;

      var xmin = this.first.xmin, xmax = this.first.xmax,
          ymin = this.first.ymin, ymax = this.first.ymax;
      for (i=0; i<n; i++) {
         xo[i] = (this.graph['fX'][i] - xmin) / (xmax - xmin);
         yo[i] = (this.graph['fY'][i] - ymin) / (ymax - ymin);
         if (w > h) yo[i] = yo[i] / ratio;
         else if (h > w) xo[i] = xo[i] / ratio;
      }
      // The first part of the filled area is made of the graph points.
      // Make sure that two adjacent points are different.
      xf[0] = xo[0];
      yf[0] = yo[0];
      nf = 0;
      for (i=1; i<n; i++) {
         if (xo[i] == xo[i-1] && yo[i] == yo[i-1]) continue;
         nf++;
         xf[nf] = xo[i];
         if (xf[i] == xf[i-1]) xf[i] += 0.000001; // add an epsilon to avoid exact vertical lines.
         yf[nf] = yo[i];
      }
      // For each graph points a shifted points is computed to build up
      // the second part of the filled area. First and last points are
      // treated as special cases, outside of the loop.
      if (xf[1] == xf[0]) {
         a = Math.PI / 2.0;
      } else {
         a = Math.atan((yf[1] - yf[0]) / (xf[1] - xf[0]));
      }
      if (xf[0] <= xf[1]) {
         xt[0] = xf[0] - wk * Math.sin(a);
         yt[0] = yf[0] + wk * Math.cos(a);
      } else {
         xt[0] = xf[0] + wk * Math.sin(a);
         yt[0] = yf[0] - wk * Math.cos(a);
      }
      if (xf[nf] == xf[nf-1]) {
         a = Math.PI / 2.0;
      } else {
         a = Math.atan((yf[nf] - yf[nf-1]) / (xf[nf] - xf[nf-1]));
      }
      if (xf[nf] >= xf[nf-1]) {
         xt[nf] = xf[nf] - wk * Math.sin(a);
         yt[nf] = yf[nf] + wk * Math.cos(a);
      } else {
         xt[nf] = xf[nf] + wk * Math.sin(a);
         yt[nf] = yf[nf] - wk * Math.cos(a);
      }

      var a1, a2, a3, xi0, yi0, xi1, yi1, xi2, yi2;
      for (i=1; i<nf; i++) {
         xi0 = xf[i];
         yi0 = yf[i];
         xi1 = xf[i+1];
         yi1 = yf[i+1];
         xi2 = xf[i-1];
         yi2 = yf[i-1];
         if (xi1 == xi0) {
            a1 = Math.PI / 2.0;
         } else {
            a1  = Math.atan((yi1 - yi0) / (xi1 - xi0));
         }
         if (xi1 < xi0) a1 = a1 + Math.PI;
         if (xi2 == xi0) {
            a2 = Math.PI / 2.0;
         } else {
            a2  = Math.atan((yi0 - yi2) / (xi0 - xi2));
         }
         if (xi0 < xi2) a2 = a2 + Math.PI;
         x1 = xi0 - wk * Math.sin(a1);
         y1 = yi0 + wk * Math.cos(a1);
         x2 = xi0 - wk * Math.sin(a2);
         y2 = yi0 + wk * Math.cos(a2);
         xm = (x1 + x2) * 0.5;
         ym = (y1 + y2) * 0.5;
         if (xm == xi0) {
            a3 = Math.PI / 2.0;
         } else {
            a3 = Math.atan((ym - yi0) / (xm - xi0));
         }
         x3 = xi0 - wk * Math.sin(a3 + (Math.PI / 2.0));
         y3 = yi0 + wk * Math.cos(a3 + (Math.PI / 2.0));
         // Rotate (x3,y3) by PI around (xi0,yi0) if it is not on the (xm,ym) side.
         if ((xm - xi0) * (x3 - xi0) < 0 && (ym - yi0) * (y3 - yi0) < 0) {
            x3 = 2 * xi0 - x3;
            y3 = 2 * yi0 - y3;
         }
         if ((xm == x1) && (ym == y1)) {
            x3 = xm;
            y3 = ym;
         }
         xt[i] = x3;
         yt[i] = y3;
      }
      // Close the polygon if the first and last points are the same
      if (xf[nf] == xf[0] && yf[nf] == yf[0]) {
         xm = (xt[nf] + xt[0]) * 0.5;
         ym = (yt[nf] + yt[0]) * 0.5;
         if (xm == xf[0]) {
            a3 = Math.PI / 2.0;
         } else {
            a3 = Math.atan((ym - yf[0]) / (xm - xf[0]));
         }
         x3 = xf[0] + wk * Math.sin(a3 + (Math.PI / 2.0));
         y3 = yf[0] - wk * Math.cos(a3 + (Math.PI / 2.0));
         if ((xm - xf[0]) * (x3 - xf[0]) < 0 && (ym - yf[0]) * (y3 - yf[0]) < 0) {
            x3 = 2 * xf[0] - x3;
            y3 = 2 * yf[0] - y3;
         }
         xt[nf] = x3;
         xt[0]  = x3;
         yt[nf] = y3;
         yt[0]  = y3;
      }
      // Find the crossing segments and remove the useless ones
      var xc, yc, c1, b1, c2, b2;
      var cross = false;
      var nf2 = nf;
      for (i=nf2; i>0; i--) {
         for (j=i-1; j>0; j--) {
            if (xt[i-1] == xt[i] || xt[j-1] == xt[j]) continue;
            c1  = (yt[i-1] - yt[i]) / (xt[i-1] - xt[i]);
            b1  = yt[i] - c1 * xt[i];
            c2  = (yt[j-1] - yt[j]) / (xt[j-1] - xt[j]);
            b2  = yt[j] - c2 * xt[j];
            if (c1 != c2) {
               xc = (b2 - b1) / (c1 - c2);
               yc = c1 * xc + b1;
               if (xc > Math.min(xt[i], xt[i-1]) && xc < Math.max(xt[i], xt[i-1]) &&
                     xc > Math.min(xt[j], xt[j-1]) && xc < Math.max(xt[j], xt[j-1]) &&
                     yc > Math.min(yt[i], yt[i-1]) && yc < Math.max(yt[i], yt[i-1]) &&
                     yc > Math.min(yt[j], yt[j-1]) && yc < Math.max(yt[j], yt[j-1])) {
                  nf++; xf[nf] = xt[i]; yf[nf] = yt[i];
                  nf++; xf[nf] = xc   ; yf[nf] = yc;
                  i = j;
                  cross = true;
                  break;
               } else {
                  continue;
               }
            } else {
               continue;
            }
         }
         if (!cross) {
            nf++;
            xf[nf] = xt[i];
            yf[nf] = yt[i];
         }
         cross = false;
      }
      nf++; xf[nf] = xt[0]; yf[nf] = yt[0]; nf++;
      for (i=0; i<nf; i++) {
         if (w > h) {
            xf[i] = xmin + (xf[i] * (xmax - xmin));
            yf[i] = ymin + (yf[i] * (ymax - ymin)) * ratio;
         }
         else if (h > w) {
            xf[i] = xmin + (xf[i] * (xmax - xmin)) * ratio;
            yf[i] = ymin + (yf[i] * (ymax - ymin));
         }
         else {
            xf[i] = xmin + (xf[i] * (xmax - xmin));
            yf[i] = ymin + (yf[i] * (ymax - ymin));
         }
         if (this.logx && xf[i] <= 0.0) xf[i] = xmin;
         if (this.logy && yf[i] <= 0.0) yf[i] = ymin;
      }

      this.excl = d3.range(nf).map(function(p) {
         return {
            x: xf[p],
            y: yf[p]
         };
      });

      this.excl_ec = ec;
      this.excl_ff = ff;

      /* some clean-up */
      xo.splice(0, xo.length); yo.splice(0, yo.length);
      xo = null; yo = null;
      xt.splice(0, xt.length); yt.splice(0, yt.length);
      xt = null; yt = null;
      xf.splice(0, xf.length); yf.splice(0, yf.length);
      xf = null; yf = null;
   }

   JSROOT.TGraphPainter.prototype.DrawBins = function()
   {
      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));

      this.RemoveDraw();
      this.draw_g = this.first.svg_frame.append("svg:g");

      var pthis = this;

      function TooltipText(d) {
         if (pthis.draw_errors && ('exlow' in d))
            return "x = " + pthis.first.AxisAsText("x", d.x) + " \ny = " + pthis.first.AxisAsText("y", d.y) +
                   "\nerror x = -" + pthis.first.AxisAsText("x", d.exlow) + "/+" + pthis.first.AxisAsText("x", d.exhigh) +
                   "\nerror y = -" + pthis.first.AxisAsText("y", d.eylow) + "/+" + pthis.first.AxisAsText("y", d.eyhigh);
         else
            return "x = " + pthis.first.AxisAsText("x", d.x) + " \ny = " + pthis.first.AxisAsText("y", d.y);
      }

      var x = this.first.x;
      var y = this.first.y;

      if (this.seriesType == 'line') {
         /* contour lines only */
         var line = d3.svg.line()
            .x(function(d) { return x(d.x);})
            .y(function(d) { return y(d.y);});
      }

      if (this.seriesType == 'bar') {
         var fillcolor = JSROOT.Painter.root_colors[this.graph['fFillColor']];
         if (typeof(fillcolor) == 'undefined') fillcolor = "rgb(204, 204, 204)";
         /* filled bar graph */
         var xdom = this.first.x.domain();
         var xfactor = xdom[1]-xdom[0];
         this.draw_errors = false;

         var nodes = this.draw_g
            .selectAll("bar_graph")
            .data(pthis.bins)
            .enter()
            .append("svg:rect")
            .attr("x", function(d) { return x(d.x)} )
            .attr("y", function(d) { return y(d.y)} )
            .attr("width", function(d) { return (w / (xdom[1]-xdom[0]))-1} )
            .attr("height", function(d) { return y(d.y) - y(d.y+d.bh); } )
            .style("fill", fillcolor);

         if (JSROOT.gStyle.Tooltip)
            nodes.append("svg:title").text(function(d) {
                    return "x = " + d.x.toPrecision(4) + " \nentries = " + d.y.toPrecision(4);
                   });
      }
      if (this.exclusionGraph) {
         /* first draw exclusion area, and then the line */
         this.showMarker = false;
         if (this.graph['fFillStyle'] > 3000 && this.graph['fFillStyle'] <= 3025) {
            var patternid = JSROOT.Painter.createFillPattern(this.vis, this.graph['fFillStyle'], this.graph['fFillColor']);
            this.draw_g.append("svg:path")
               .attr("d", line(pthis.excl))
               .style("stroke", "none")
               .style("stroke-width", pthis.excl_ff)
               .style("fill", "url(#" + patternid + ")")
               .style("antialias", "false");
         }
         else {
            this.draw_g.append("svg:path")
               .attr("d", line(pthis.excl))
               .style("stroke", "none")
               .style("stroke-width", pthis.excl_ff)
               .style("fill", pthis.excl_ec);
         }
      }
      if (this.seriesType == 'line') {
         this.draw_g.append("svg:path")
            .attr("d", line(pthis.bins))
            .attr("class", "draw_line")
            .style("stroke", (pthis.optionLine == 1) ? JSROOT.Painter.root_colors[pthis.graph['fLineColor']] : "none")
            .style("stroke-width", pthis.bins_lw)
            .style("stroke-dasharray", JSROOT.Painter.root_line_styles[pthis.graph['fLineStyle']])
            .style("fill", (pthis.optionFill == 1) ? JSROOT.Painter.root_colors[pthis.graph['fFillColor']] : "none");

         if (JSROOT.gStyle.Tooltip)
            this.draw_g.selectAll("draw_line")
              .data(pthis.bins).enter()
              .append("svg:circle")
              .attr("cx", function(d) { return x(d.x); })
              .attr("cy", function(d) { return y(d.y); })
              .attr("r", 3)
              .attr("opacity", 0)
              .append("svg:title").text(TooltipText);

      }
      if ((this.graph['_typename'] == 'TGraphErrors' ||
            this.graph['_typename'] == 'TGraphAsymmErrors' ||
            this.graph['_typename'].match(/\bRooHist/)) && this.draw_errors && !this.optionBar) {

         // here are up to five elements are collected, try to group them

         var nodes = this.draw_g.selectAll("g.node").data(this.bins).enter().append("svg:g");

         // Add x-error indicators

         // than doing filer append error bars
         var xerr = nodes.filter(function(d) { return (d.exlow>0) || (d.exhigh>0); })
              .append("svg:line")
               .attr("x1", function(d) { return x(d.x-d.exlow)} )
               .attr("y1", function(d) { return y(d.y)} )
               .attr("x2", function(d) { return x(d.x+d.exhigh)} )
               .attr("y2", function(d) { return y(d.y)} )
               .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
               .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.exlow>0); })
              .append("svg:line")
               .attr("y1", function(d) { return y(d.y)-3} )
               .attr("x1", function(d) { return x(d.x-d.exlow)})
               .attr("y2", function(d) { return y(d.y)+3})
               .attr("x2", function(d) { return x(d.x-d.exlow)})
               .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
               .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.exhigh>0); })
              .append("svg:line")
               .attr("y1", function(d) { return y(d.y)-3} )
               .attr("x1", function(d) { return x(d.x+d.exhigh) })
               .attr("y2", function(d) { return y(d.y)+3})
               .attr("x2", function(d) { return x(d.x+d.exhigh) })
               .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
               .style("stroke-width", this.graph['fLineWidth']);

         // Add y-error indicators

         var yerr = nodes.filter(function(d) { return (d.eylow>0) || (d.eyhigh>0); })
            .append("svg:line")
            .attr("x1", function(d) { return x(d.x)})
            .attr("y1", function(d) { return y(d.y-d.eylow) })
            .attr("x2", function(d) { return x(d.x)})
            .attr("y2", function(d) { return y(d.y+d.eyhigh) })
            .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
            .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.eylow>0); })
            .append("svg:line")
            .attr("x1", function(d) { return x(d.x)-3})
            .attr("y1", function(d) { return y(d.y-d.eylow) })
            .attr("x2", function(d) { return x(d.x)+3})
            .attr("y2", function(d) { return y(d.y-d.eylow) })
            .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
            .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.eyhigh>0); })
            .append("svg:line")
            .attr("x1", function(d) { return x(d.x)-3})
            .attr("y1", function(d) { return y(d.y+d.eyhigh) })
            .attr("x2", function(d) { return x(d.x)+3})
            .attr("y2", function(d) { return y(d.y+d.eyhigh) })
            .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
            .style("stroke-width", this.graph['fLineWidth']);


         if (JSROOT.gStyle.Tooltip) {
            xerr.append("svg:title").text(TooltipText);
            yerr.append("svg:title").text(TooltipText);
         }
      } else this.draw_errors = false;

      if (this.showMarker) {
         /* Add markers */
         var filled = false;
         if ((this.graph['fMarkerStyle'] == 8) ||
             (this.graph['fMarkerStyle'] > 19 && this.graph['fMarkerStyle'] < 24) ||
             (this.graph['fMarkerStyle'] == 29))
            filled = true;

         var info_marker = JSROOT.Painter.getRootMarker(this.graph['fMarkerStyle']);

         var shape = info_marker['shape'];
         var filled = info_marker['toFill'];
         var toRotate = info_marker['toRotate'];
         var markerSize = this.graph['fMarkerSize'];
         var markerScale = (shape == 0) ? 32 : 64;
         if (this.graph['fMarkerStyle'] == 1) markerScale = 1;

         switch (shape) {
            case 6:
               var marker = "M " + (-4 * markerSize) + " " + (-1 * markerSize)
                           + " L " + 4 * markerSize + " " + (-1 * markerSize)
                           + " L " + (-2.4 * markerSize) + " " + 4 * markerSize
                           + " L 0 " + (-4 * markerSize) + " L " + 2.8 * markerSize
                           + " " + 4 * markerSize + " z";
               break;
            case 7:
               var marker = "M " + (- 4 * markerSize) + " " + (-4 * markerSize)
                           + " L " + 4 * markerSize + " " + 4 * markerSize + " M 0 "
                           + (-4 * markerSize) + " 0 " + 4 * markerSize + " M "
                           + 4 * markerSize + " " + (-4 * markerSize) + " L "
                           + (-4 * markerSize) + " " + 4 * markerSize + " M "
                           + (-4 * markerSize) + " 0 L " + 4 * markerSize + " 0";
               break;
            default:
               var marker = d3.svg.symbol()
                           .type(d3.svg.symbolTypes[shape])
                           .size(markerSize * markerScale);
               break;
         }

         var markers = this.draw_g.selectAll("markers")
            .data(this.bins)
            .enter()
            .append("svg:path")
            .attr("transform", function(d) { return "translate(" + x(d.x) + " , " + y(d.y) + ")"} )
            .style("fill", filled ? JSROOT.Painter.root_colors[this.graph['fMarkerColor']] : "none")
            .style("stroke", JSROOT.Painter.root_colors[this.graph['fMarkerColor']])
            .attr("d", marker);

         if (JSROOT.gStyle.Tooltip)
            markers
            .append("svg:title")
            .text(TooltipText);
      }
   }

   JSROOT.TGraphPainter.prototype.UpdateObject = function(obj) {
      if (obj['_typename']!= this.graph['_typename']) return false;

      // if our own histogram was used as axis drawing, we need update histogram as well
      if (this.ownhisto) this.first.UpdateObject(obj['fHistogram']);

      // TODO: make real update of TGraph object content
      this.graph['fX'] = obj['fX'];
      this.graph['fY'] = obj['fY'];
      this.graph['fNpoints'] = obj['fNpoints'];
      this.CreateBins();
      return true;
   }

   JSROOT.Painter.drawGraph = function(vis, graph, opt)
   {
      var ownhisto = false;
      if (!('painters' in vis)) {
         if (!('fHistogram' in graph)) {
            alert("drawing first graphs without fHistogram field not (yet) supported");
            return -1;
         } else {
            JSROOT.Painter.drawHistogram1D(vis, graph['fHistogram']);
            ownhisto = true;
         }
      }

      var painter = new JSROOT.TGraphPainter(graph);

      painter['ownhisto'] = ownhisto;

      painter.SetFrame(vis, true);

      painter.DecodeOptions(opt);

      painter.CreateBins();

      painter.DrawBins();

      return painter;
   }

   // ============================================================

   JSROOT.TPavePainter = function(pave) {
      JSROOT.TBasePainter.call(this);
      this.pavetext = pave;
      this.Enabled = true;
      this.main_rect = null;
      this.drag_rect = null;
      this.resize_rect = null;
   }

   JSROOT.TPavePainter.prototype = Object.create( JSROOT.TBasePainter.prototype );


   JSROOT.TPavePainter.prototype.IsObject = function(obj) {
      return this.pavetext === obj;
   }

   JSROOT.TPavePainter.prototype.DrawPaveText = function() {
      var pavetext = this.pavetext;
      var vis = this.vis;

      var w = Number(vis.attr("width")), h = Number(vis.attr("height"));

      var pos_x = pavetext['fX1NDC'] * w;
      var pos_y = (1.0 - pavetext['fY1NDC']) * h;
      var width = Math.abs(pavetext['fX2NDC'] - pavetext['fX1NDC']) * w;
      var height = Math.abs(pavetext['fY2NDC'] - pavetext['fY1NDC']) * h;
      pos_y -= height;
      var nlines = pavetext['fLines'].arr.length;
      var font_size = Math.round(height / (nlines * 1.2));
      var fcolor = JSROOT.Painter.root_colors[pavetext['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pavetext['fLineColor']];
      var tcolor = JSROOT.Painter.root_colors[pavetext['fTextColor']];
      var scolor = JSROOT.Painter.root_colors[pavetext['fShadowColor']];
      if (pavetext['fFillStyle'] == 0) fcolor = 'none';
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      // "middle", "start", "end"

      var align = 'start', halign = Math.round(pavetext['fTextAlign']/10);
      var baseline = 'bottom', valign = pavetext['fTextAlign']%10;
      if (halign == 1) align = 'start';  else
      if (halign == 2) align = 'middle';  else
      if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom'; else
      if (valign == 2) baseline = 'middle'; else
      if (valign == 3) baseline = 'top';

      var lmargin = 0;
      switch (halign) {
         case 1:
            lmargin = pavetext['fMargin'] * width;
            break;
         case 2:
            lmargin = width / 2;
            break;
         case 3:
            lmargin = width - (pavetext['fMargin'] * width);
            break;
      }

      // for now ignore all align parameters, draw as is
      if (nlines>1) lmargin = pavetext['fMargin'] * width / 2;

      var fontDetails = JSROOT.Painter.getFontDetails(pavetext['fTextFont']);
      var lwidth = pavetext['fBorderSize'] ? pavetext['fBorderSize'] : 0;

      if (this.main_rect == null) {
         this.main_rect = this.vis.append("rect");
      } else {
         // force main rect of the stat box be last item in the primitives to kept it on the top
         var prnt = this.main_rect.node().parentNode;
         prnt.removeChild(this.main_rect.node());
         prnt.appendChild(this.main_rect.node());
      }

      this.main_rect
         .attr("x", pos_x)
         .attr("y", pos_y)
         .attr("height", height)
         .attr("width", width)
         .attr("fill", fcolor)
         .style("stroke-width", lwidth ? 1 : 0)
         .style("stroke", lcolor);

      var pthis = this;

      this.AddDrag("stat", this.main_rect, {
         move: function(x, y, dx, dy) {
            pthis.draw_g.attr("transform", "translate(" + x + "," + y + ")");

            pthis.pavetext['fX1NDC'] += dx / Number(pthis.vis.attr("width"));
            pthis.pavetext['fX2NDC'] += dx / Number(pthis.vis.attr("width"));
            pthis.pavetext['fY1NDC'] -= dy / Number(pthis.vis.attr("height"));
            pthis.pavetext['fY2NDC'] -= dy / Number(pthis.vis.attr("height"));
         },
         resize: function(width, height) {
            pthis.pavetext['fX2NDC'] = pthis.pavetext['fX1NDC'] + width/Number(pthis.vis.attr("width"));
            pthis.pavetext['fY1NDC'] = pthis.pavetext['fY2NDC'] - height/Number(pthis.vis.attr("height"));

            pthis.RemoveDraw();
            pthis.DrawPaveText();
         }
      });

      // container to just to recalculate coordinates
      this.draw_g =
         vis.append("svg:g").attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      var first_stat = 0, num_cols = 0;
      var maxlw = 0;
      var lines = new Array;

      // adjust font size
      for (var j=0; j<nlines; ++j) {
         var line = JSROOT.Painter.translateLaTeX(pavetext['fLines'].arr[j]['fTitle']);

         lines.push(line);

         var lw = lmargin + JSROOT.Painter.stringWidth(vis, line, font_size, fontDetails);
         if (lw > maxlw) maxlw = lw;

         if ((j==0) || (line.indexOf('|')<0)) continue;
         if (first_stat === 0) first_stat = j;
         var parts = line.split("|");
         if (parts.length>num_cols) num_cols = parts.length;
      }

      if (maxlw > width)
        font_size = font_size * (width / maxlw);

      var stepy = height / nlines;

      if (nlines == 1) {
         this.draw_g.append("text")
            .attr("text-anchor", align)
            .attr("x", lmargin)
            .attr("y", (height/2) + (font_size/3))
            .attr("font-family", fontDetails['name'])
            .attr("font-weight", fontDetails['weight'])
            .attr("font-style", fontDetails['style'])
            .attr("font-size", font_size)
            .attr("fill", tcolor)
            .text(lines[0]);
      }
      else {

         for (var j=0; j<nlines; ++j) {
            var jcolor = JSROOT.Painter.root_colors[pavetext['fLines'].arr[j]['fTextColor']];
            if (pavetext['fLines'].arr[j]['fTextColor'] == 0)  jcolor = tcolor;
            var posy = j * stepy + font_size;

            if (pavetext['_typename'] == 'TPaveStats') {
               if ((first_stat>0) && (j>=first_stat)) {
                  var parts = lines[j].split("|");
                  for (var n=0;n<parts.length;n++)
                     this.draw_g.append("text")
                     .attr("text-anchor", "middle")
                     .attr("x",  width*(n+0.5)/num_cols)
                     .attr("y", posy)
                     .attr("font-family", fontDetails['name'])
                     .attr("font-weight", fontDetails['weight'])
                     .attr("font-style", fontDetails['style'])
                     .attr("font-size", font_size)
                     .attr("fill", jcolor)
                     .text(parts[n]);
               } else
               if ((j==0) || (lines[j].indexOf('=')<0)) {
                  this.draw_g.append("text")
                     .attr("text-anchor", (j == 0) ? "middle" : "start")
                     .attr("x", ((j==0) ? width/2 : pavetext['fMargin'] * width))
                     .attr("y", posy)
                     .attr("font-family", fontDetails['name'])
                     .attr("font-weight", fontDetails['weight'])
                     .attr("font-style", fontDetails['style'])
                     .attr("font-size", font_size)
                     .attr("fill", jcolor)
                     .text(lines[j]);
               } else {
                  var parts = lines[j].split("=");
                  for (var n=0;n<2;n++)
                     this.draw_g.append("text")
                     .attr("text-anchor", (n==0) ? "start" : "end")
                     .attr("x", (n==0) ? pavetext['fMargin'] * width : (1-pavetext['fMargin']) * width)
                     .attr("y", posy)
                     .attr("font-family", fontDetails['name'])
                     .attr("font-weight", fontDetails['weight'])
                     .attr("font-style", fontDetails['style'])
                     .attr("font-size", font_size)
                     .attr("fill", jcolor)
                     .text(parts[n]);
               }
            } else {
               this.draw_g.append("text")
                  .attr("text-anchor", "start")
                  .attr("x", lmargin)
                  .attr("y", posy)
                  .attr("font-family", fontDetails['name'])
                  .attr("font-weight", fontDetails['weight'])
                  .attr("font-style", fontDetails['style'])
                  .attr("font-size", font_size)
                  .attr("fill", jcolor)
                  .text(lines[j]);
            }
         }
      }

      if (pavetext['fBorderSize'] && (pavetext['_typename'] == 'TPaveStats')) {
         this.draw_g.append("svg:line")
            .attr("class", "pavedraw")
            .attr("x1", 0)
            .attr("y1", stepy)
            .attr("x2", width)
            .attr("y2", stepy)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth ? 1 : 'none');
      }

      if ((first_stat > 0) && (num_cols > 1)) {

         for (var nrow = first_stat; nrow < nlines; nrow++)
            this.draw_g.append("svg:line")
              .attr("x1", 0)
              .attr("y1", nrow*stepy)
              .attr("x2", width)
              .attr("y2", nrow*stepy)
              .style("stroke", lcolor)
              .style("stroke-width", lwidth ? 1 : 'none');

         for (var ncol = 0; ncol<num_cols-1; ncol++)
            this.draw_g.append("svg:line")
            .attr("x1", width/num_cols*(ncol+1))
            .attr("y1", first_stat * stepy)
            .attr("x2", width/num_cols*(ncol+1) )
            .attr("y2", height)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth ? 1 : 'none');
      }

      if (lwidth && lwidth > 1) {
         this.draw_g.append("svg:line")
            .attr("x1", width+(lwidth/2))
            .attr("y1", lwidth+1)
            .attr("x2", width+(lwidth/2))
            .attr("y2", height+lwidth-1)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
         this.draw_g.append("svg:line")
            .attr("x1", lwidth+1)
            .attr("y1", height+(lwidth/2))
            .attr("x2", width+lwidth-1)
            .attr("y2", height+(lwidth/2))
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
      }
   }

   JSROOT.TPavePainter.prototype.AddLine = function(txt) {
      this.pavetext['fLines'].arr.push( {'fTitle': txt, "fTextColor": 1} );
   }

   JSROOT.TPavePainter.prototype.IsStats = function() {
      if (!this.pavetext) return false;
      return this.pavetext['fName'] == "stats";
   }


   JSROOT.TPavePainter.prototype.FillStatistic = function()
   {
      if (!this.IsStats()) return false;

      var dostat = new Number(this.pavetext['fOptStat']);
      if (!dostat) dostat = new Number(JSROOT.gStyle.OptStat);

      // we take histogram from first painter
      if (this.first && ('FillStatistic' in this.first)) {

         // make empty at the beginning
         this.pavetext['fLines'].arr.length = 0;

         this.first.FillStatistic(this, dostat);
         return true;
      }
      return true;
   }

   JSROOT.TPavePainter.prototype.Redraw = function() {

      this.RemoveDraw();

      // if pavetext artificially disabled, do not redraw it
      if (!this.Enabled) {
         this.RemoveDrag("stat");
         if (this.main_rect) { this.main_rect.remove(); this.main_rect = null; }
         return;
      }

      // recalculate statistic when manipulation with view were done
      // if (this.first.original_view_changed)
      this.FillStatistic();

      this.DrawPaveText();
   }

   JSROOT.Painter.DrawPaveText = function(vis, pavetext)
   {
      if (pavetext['fX1NDC'] < 0.0 || pavetext['fY1NDC'] < 0.0 ||
          pavetext['fX1NDC'] > 1.0 || pavetext['fY1NDC'] > 1.0) {
         return;
      }

      var painter = new JSROOT.TPavePainter(pavetext);

      painter.SetFrame(vis, true);


      // refill statistic in any case
      // if ('_AutoCreated' in pavetext)
      painter.FillStatistic();

      painter.DrawPaveText();

      return painter;
   }


   // ===========================================================================

   JSROOT.TColzPalettePainter = function(palette) {
      JSROOT.TBasePainter.call(this);
      this.palette = palette;
      this.Enabled = true;
   }

   JSROOT.TColzPalettePainter.prototype = Object.create( JSROOT.TBasePainter.prototype );


   JSROOT.TColzPalettePainter.prototype.IsObject = function(obj) {
      return this.palette === obj;
   }

   JSROOT.TColzPalettePainter.prototype.DrawPalette = function() {
      var palette  = this.palette;
      var vis = this.vis;
      var axis = palette['fAxis'];

      var minbin = this.first.minbin;
      var maxbin = this.first.maxbin;
      var ndiv = axis['fNdiv'];

      var width = Number(vis.attr("width")), height = Number(vis.attr("height"));
      
      var s_height = Math.abs(palette['fY2NDC'] - palette['fY1NDC']) * height;

      var axisOffset = axis['fLabelOffset'] * width;
      var tickSize = axis['fTickSize'] * width;
      
      var nbr1 = Math.max(ndiv % 100, 1);
      // var nbr2 = Math.max(Math.round((ndiv % 10000) / 100), 1);
      // var nbr3 = Math.max(Math.round(ndiv / 10000), 1);

      var z = d3.scale.linear().clamp(true)
               .domain([minbin, maxbin])
               .range([s_height, 0])
               .nice();

      var axisFontDetails = JSROOT.Painter.getFontDetails(axis['fLabelFont']);
      var axisLabelFontSize = axis['fLabelSize'] * height;

      var pos_x = palette['fX1NDC'] * width;
      var pos_y = height - palette['fY1NDC'] * height;
      
      var s_width = Math.abs(palette['fX2NDC'] - palette['fX1NDC']) * width;
      pos_y -= s_height;

      // Draw palette pad
      this.draw_g = vis.append("svg:g")
            .attr("height", s_height)
            .attr("width", s_width)
            .attr("transform", "translate(" + pos_x + ", " + pos_y + ")");


      var paletteColors = this.first.paletteColors;

      // Draw palette
      var rectHeight = s_height / paletteColors.length;
      this.draw_g.selectAll("colorRect")
         .data(paletteColors)
         .enter()
         .append("svg:rect")
         .attr("class", "colorRect")
         .attr("x", 0)
         .attr("y", function(d, i) { return s_height - (i + 1)* rectHeight;})
         .attr("width", s_width)
         .attr("height", rectHeight)
         .attr("fill", function(d) { return d;})
         .attr("stroke", function(d) {return d;});
      /*
       * Build and draw axes
       */

      var z_axis = d3.svg.axis()
                  .scale(z)
                  .orient("right")
                  .tickPadding(axisOffset)
                  .tickSize(-tickSize, -tickSize/2, 0)
                  .ticks(nbr1);

      var zax = this.draw_g.append("svg:g")
               .attr("class", "zaxis")
               .attr("transform", "translate(" + s_width + ", 0)")
               .call(z_axis);

      zax.selectAll("text")
         .attr("font-size", axisLabelFontSize)
         .attr("font-weight", axisFontDetails['weight'])
         .attr("font-style", axisFontDetails['style'])
         .attr("font-family", axisFontDetails['name'])
         .attr("fill", JSROOT.Painter.root_colors[axis['fLabelColor']]);

      /*
       * Add palette axis title
       */
      var title = axis['fTitle'];
      if (title != "" && typeof(axis['fTitleFont']) != 'undefined') {
         axisFontDetails = JSROOT.Painter.getFontDetails(axis['fTitleFont']);
         var axisTitleFontSize = axis['fTitleSize'] * height;
         this.draw_g.append("text")
               .attr("class", "Z axis label")
               .attr("x", s_width + axisLabelFontSize)
               .attr("y", s_height)
               .attr("text-anchor", "end")
               .attr("font-family", axisFontDetails['name'])
               .attr("font-weight", axisFontDetails['weight'])
               .attr("font-style", axisFontDetails['style'])
               .attr("font-size", axisTitleFontSize )
               .text(title);
      }

      if (this.main_rect == null) {
         this.main_rect = this.vis.append("rect")
                              .attr("id","colz_move_rect")
                              .style("opacity", "0");
      } else {
         // ensure that all color drawing inserted before move rect
         var prnt = this.main_rect.node().parentNode;
         prnt.removeChild(this.draw_g.node());
         prnt.insertBefore(this.draw_g.node(), this.main_rect.node());
      }

      this.main_rect
         .attr("x", pos_x)
         .attr("y", pos_y)
         .attr("width", s_width)
         .attr("height", s_height);

      var pthis = this;

      this.AddDrag("colz", this.main_rect, {
         move: function(x, y, dx, dy) {

            pthis.draw_g.attr("transform", "translate(" + x + "," + y + ")");

            pthis.palette['fX1NDC'] += dx / Number(pthis.vis.attr("width"));
            pthis.palette['fX2NDC'] += dx / Number(pthis.vis.attr("width"));
            pthis.palette['fY1NDC'] -= dy / Number(pthis.vis.attr("height"));
            pthis.palette['fY2NDC'] -= dy / Number(pthis.vis.attr("height"));
         },
         resize: function(width, height) {
            pthis.palette['fX2NDC'] = pthis.palette['fX1NDC'] + width/Number(pthis.vis.attr("width"));
            pthis.palette['fY1NDC'] = pthis.palette['fY2NDC'] - height/Number(pthis.vis.attr("height"));

            pthis.RemoveDraw();
            pthis.DrawPalette();
         }
      });


   }

   JSROOT.TColzPalettePainter.prototype.Redraw = function() {

      this.RemoveDraw();

      // if palette artificially disabled, do not redraw it
      if (!this.Enabled) {
         this.RemoveDrag("colz");
         if (this.main_rect) { this.main_rect.remove(); this.main_rect = null; }
         return;
      }

      this.DrawPalette();
   }

   JSROOT.Painter.drawPaletteAxis = function(vis, palette) {
      var painter = new JSROOT.TColzPalettePainter(palette);

      painter.SetFrame(vis, true);

      painter.DrawPalette();

      return painter;
   }

   // =============================================================

   JSROOT.THistPainter = function(histo) {
      JSROOT.TBasePainter.call(this);
      this.histo = histo;
   }

   JSROOT.THistPainter.prototype = Object.create( JSROOT.TBasePainter.prototype );


   JSROOT.THistPainter.prototype.IsObject = function(obj) {
      return this.histo === obj;
   }

   JSROOT.THistPainter.prototype.Dimension = function() {
      if (!this.histo) return 0;
      return this.histo['fDimension'];
   }

   JSROOT.THistPainter.prototype.SetFrame = function(vis, opt) {
      this.draw_content = false;

      if (vis['ROOT:frame']==null)
         JSROOT.Painter.createFrame(vis);

      JSROOT.TBasePainter.prototype.SetFrame.call(this, vis, false)

      if (vis['ROOT:svg_frame']==null) {
         alert("missing svg_frame");
         return;
      }

      if ((opt==null) || (opt=="")) opt = this.histo['fOption'];

      // here we deciding how histogram will look like and how will be shown
      this.options = JSROOT.Painter.decodeOptions(opt, this.histo, this.pad);
//      if (this.histo['_typename'] == "TProfile")
//         this.options.Error = 11;

      this.show_gridx = false;
      this.show_gridy = false;
      if (this.pad && typeof(this.pad) != 'undefined') {
         this.show_gridx = this.pad['fGridx'];
         this.show_gridy = this.pad['fGridy'];
      }
   }

   JSROOT.THistPainter.prototype.ScanContent = function() {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values

      alert("HistPainter.prototype.ScanContent not implemented");
   }



   JSROOT.THistPainter.prototype.UpdateObject = function(obj)
   {
      if (obj['_typename'] != this.histo['_typename']) {
         alert("JSROOT.THistPainter.UpdateObject - wrong class " + obj['_typename']);
         return false;
      }

      // TODO: simple replace of object does not help - one can have different
      // complex relations between histo and stat box, histo and colz axis,
      // on could have THStack or TMultiGraph object
      // The only that could be done is update of content

      // this.histo = obj;

      this.histo['fArray'] = obj['fArray'];
      this.histo['fN'] = obj['fN'];
      this.histo['fTitle'] = obj['fTitle'];
      this.histo['fXaxis']['fNbins'] = obj['fXaxis']['fNbins'];
      this.histo['fXaxis']['fXmin'] = obj['fXaxis']['fXmin'];
      this.histo['fXaxis']['fXmax'] = obj['fXaxis']['fXmax'];
      this.histo['fYaxis']['fXmin'] = obj['fYaxis']['fXmin'];
      this.histo['fYaxis']['fXmax'] = obj['fYaxis']['fXmax'];

      this.ScanContent();

      return true;
   }


   JSROOT.THistPainter.prototype.CreateXY = function()
   {
      // here we create x,y objects which maps our physical coordnates into pixels
      // while only first painter really need such object, all others just reuse it

      if (this.first) {
         this['x'] = this.first['x'];
         this['y'] = this.first['y'];
         return;
      }

      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));

      this['scale_xmin'] = this.xmin;
      this['scale_xmax'] = this.xmax;
      if (this.zoom_xmin != this.zoom_xmax) {
         this['scale_xmin'] = this.zoom_xmin;
         this['scale_xmax'] = this.zoom_xmax;
      }

      if (this.options.Logx) {
         if (this.scale_xmax <= 0) this.scale_xmax = 0;
         if ((this.scale_xmin <= 0) || (this.scale_xmin >= this.scale_xmax)) this.scale_xmin = this.scale_xmax * 0.0001;
         this['x'] = d3.scale.log().domain([this.scale_xmin, this.scale_xmax]).range([0, w]).clamp(true);
      } else {
         this['x'] = d3.scale.linear().domain([this.scale_xmin, this.scale_xmax]).range([0, w]);
      }

      this['scale_ymin'] = this.ymin;
      this['scale_ymax'] = this.ymax;

      if (this.zoom_ypad) {
         if (this.histo.fMinimum != -1111) this.zoom_ymin = this.histo.fMinimum;
         if (this.histo.fMaximum != -1111) this.zoom_ymax = this.histo.fMaximum;
         this['zoom_ypad'] = false;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this['scale_ymin'] = this.zoom_ymin;
         this['scale_ymax'] = this.zoom_ymax;
      }

      if (this.options.Logy) {
         if (this.scale_ymax<=0) this.scale_ymax = 1;
         if ((this.scale_ymin<=0) || (this.scale_ymin>=this.scale_ymax)) this.scale_ymin = 0.0001 * this.scale_ymax;
         this['y'] = d3.scale.log().domain([this.scale_ymin, this.scale_ymax]).range([h, 0]).clamp(true);
      } else {
         this['y'] = d3.scale.linear().domain([this.scale_ymin, this.scale_ymax]).range([h, 0]);
      }
   }

   JSROOT.THistPainter.prototype.CountStat = function()
   {
      alert("CountStat not implemented");
   }

   JSROOT.THistPainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter
      if (this.first) return;

      this.frame.selectAll(".gridLine").remove();
      /* add a grid on x axis, if the option is set */

      // add a grid on x axis, if the option is set
      if (this.show_gridx) {

         this.frame.selectAll("gridLine")
         .data(this.x.ticks(this.x_nticks))
         .enter()
         .append("svg:line")
         .attr("class", "gridLine")
         .attr("x1", this.x)
         .attr("y1", Number(this.frame.attr("height")))
         .attr("x2", this.x)
         .attr("y2", 0)
         .style("stroke", "black")
         .style("stroke-width", this.histo['fLineWidth'])
         .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }

      // add a grid on y axis, if the option is set
      if (this.show_gridy) {

         this.frame.selectAll("gridLine")
         .data(this.y.ticks(this.y_nticks))
         .enter()
         .append("svg:line")
         .attr("class", "gridLine")
         .attr("x1", 0)
         .attr("y1", this.y)
         .attr("x2", Number(this.frame.attr("width")))
         .attr("y2", this.y)
         .style("stroke", "black")
         .style("stroke-width", this.histo['fLineWidth'])
         .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }
   }

   JSROOT.THistPainter.prototype.DrawBins = function() {
      alert("HistPainter.DrawBins not implemented");
   }

   JSROOT.THistPainter.prototype.AxisAsText = function(axis, value)
   {
      if (axis=="x") {
         // this is indication
         if ('dfx' in this) {
            return this.dfx(new Date(this.timeoffsetx + value*1000));
         }

         if (Math.abs(value)<1e-14)
            if (Math.abs(this.xmax - this.xmin)>1e-5) value = 0;
         return value.toPrecision(4);
      }

      if (axis=="y") {
         if ('dfy' in this) {
            return this.dfy(new Date(this.timeoffsety + value*1000));
         }
         if (Math.abs(value)<1e-14)
            if (Math.abs(this.ymax - this.ymin)>1e-5) value = 0;
         return value.toPrecision(4);
      }

      return value.toPrecision(4);
   }

   JSROOT.THistPainter.prototype.DrawAxes = function() {
      // axes can be drawn only for main (first) histogram

      if (this.first) return;

      this['x_axis_sub'] = null;
      this['y_axis_sub'] = null;

      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));
      var noexpx = this.histo['fXaxis'].TestBit(JSROOT.EAxisBits.kNoExponent);
      var noexpy = this.histo['fYaxis'].TestBit(JSROOT.EAxisBits.kNoExponent);
      var moreloglabelsx = this.histo['fXaxis'].TestBit(JSROOT.EAxisBits.kMoreLogLabels);
      var moreloglabelsy = this.histo['fYaxis'].TestBit(JSROOT.EAxisBits.kMoreLogLabels);

      if (this.histo['fXaxis']['fXmax'] < 100 && this.histo['fXaxis']['fXmax']/this.histo['fXaxis']['fXmin'] < 100) noexpx = true;
      if (this.histo['fYaxis']['fXmax'] < 100 && this.histo['fYaxis']['fXmax']/this.histo['fYaxis']['fXmin'] < 100) noexpy = true;

      var ndivx = this.histo['fXaxis']['fNdivisions'];
      this['x_nticks'] = ndivx%100; // used also to draw grids
      var n2ax = (ndivx%10000 - this.x_nticks)/100;
      var n3ax = ndivx/10000;

      var ndivy = this.histo['fYaxis']['fNdivisions'];
      this['y_nticks'] = ndivy%100; // used also to draw grids
      var n2ay = (ndivy%10000 - this.y_nticks)/100;
      var n3ay = ndivy/10000;
      
      /* X-axis label */
      var label = JSROOT.Painter.translateLaTeX(this.histo['fXaxis']['fTitle']);
      var xAxisTitleFontSize = this.histo['fXaxis']['fTitleSize'] * h;
      var xAxisLabelOffset = 3 + (this.histo['fXaxis']['fLabelOffset'] * h);
      var xAxisLabelFontSize = this.histo['fXaxis']['fLabelSize'] * h;
      var xAxisFontDetails = JSROOT.Painter.getFontDetails(this.histo['fXaxis']['fTitleFont']);

      if (label.length > 0) {
         if (!('x_axis_label' in this))
            this['x_axis_label'] =
               this.frame.append("text").attr("class", "x_axis_label");

         this.x_axis_label
         .attr("x", w)
         .attr("y", h)
         .attr("text-anchor", "end")
         .attr("font-family", xAxisFontDetails['name'])
         .attr("font-weight", xAxisFontDetails['weight'])
         .attr("font-style", xAxisFontDetails['style'])
         .attr("font-size", xAxisTitleFontSize)
         .text(label)
         .attr("transform", "translate(0," + (xAxisLabelFontSize + xAxisLabelOffset * this.histo['fXaxis']['fTitleOffset'] + xAxisTitleFontSize) + ")");
      }

      /* Y-axis label */
      label = JSROOT.Painter.translateLaTeX(this.histo['fYaxis']['fTitle']);
      var yAxisTitleFontSize = this.histo['fYaxis']['fTitleSize'] * h;
      var yAxisLabelOffset = 3 + (this.histo['fYaxis']['fLabelOffset'] * w);
      var yAxisLabelFontSize = this.histo['fYaxis']['fLabelSize'] * h;
      var yAxisFontDetails = JSROOT.Painter.getFontDetails(this.histo['fYaxis']['fTitleFont']);

      if (label.length > 0) {
         if (!('y_axis_label' in this))
            this['y_axis_label'] =
               this.frame.append("text").attr("class", "y_axis_label");

         this.y_axis_label
         .attr("x", 0)
         .attr("y", -yAxisLabelFontSize - yAxisTitleFontSize - yAxisLabelOffset * this.histo['fYaxis']['fTitleOffset'])
         .attr("font-family", yAxisFontDetails['name'])
         .attr("font-size", yAxisTitleFontSize)
         .attr("font-weight", yAxisFontDetails['weight'])
         .attr("font-style", yAxisFontDetails['style'])
         .attr("fill", "black")
         .attr("text-anchor", "end")
         .text(label)
         .attr("transform", "rotate(270, 0, 0)");
      }

      var xAxisColor = this.histo['fXaxis']['fAxisColor'];
      var xDivLength = this.histo['fXaxis']['fTickLength'] * h;
      var yAxisColor = this.histo['fYaxis']['fAxisColor'];
      var yDivLength = this.histo['fYaxis']['fTickLength'] * w;

      var pthis = this;

      /*
       * Define the scales, according to the information from the pad
       */
      var xrange = this.xmax - this.xmin;
      if (this.histo['fXaxis']['fTimeDisplay']) {
         if (this.x_nticks > 8) this.x_nticks = 8;
         var timeformatx = JSROOT.Painter.getTimeFormat(this.histo['fXaxis']);

         this['timeoffsetx'] = JSROOT.Painter.getTimeOffset(this.histo['fXaxis']);

         var scale_xrange = this.scale_xmax - this.scale_xmin;

         if ((timeformatx.length == 0) || (scale_xrange < 0.1*xrange))
            timeformatx = JSROOT.Painter.chooseTimeFormat(scale_xrange, this.x_nticks);

         this['dfx'] = d3.time.format(timeformatx);

         this['x_axis'] = d3.svg.axis()
             .scale(this.x)
             .orient("bottom")
             .tickPadding(xAxisLabelOffset)
             .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
             .tickFormat(function(d) { return pthis.dfx(new Date(pthis.timeoffsetx + d*1000)); })
             .ticks(this.x_nticks);
      }
      else if (this.options.Logx) {
         this['x_axis'] = d3.svg.axis()
            .scale(this.x)
            .orient("bottom")
            .tickPadding(xAxisLabelOffset)
            .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
            .tickFormat(function(d) {
                var val = parseFloat(d);
                var vlog = Math.abs(JSROOT.Math.log10(val));
                if (moreloglabelsx) {
                  if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                     if (noexpx) return val.toFixed();
                           else return JSROOT.Painter.formatExp(val.toExponential(0));
                  }
                     else return null;
                }
                else {
                  if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                    if (noexpx) return val.toFixed();
                           else return JSROOT.Painter.formatExp(val.toExponential(0));
                  }
                  else return null;
                }
             });
      }
      else {
         this['x_axis'] = d3.svg.axis()
            .scale(this.x)
            .orient("bottom")
            .tickPadding(xAxisLabelOffset)
            .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
            .tickFormat(function(d) {
                // avoid rounding problems around 0
                if ((Math.abs(d)<1e-14) && (Math.abs(xrange)>1e-5)) d = 0;
                return parseFloat(d.toPrecision(12));
            })
            .ticks(this.x_nticks);
      }

      var yrange = this.ymax - this.ymin;
      if (this.histo['fYaxis']['fTimeDisplay']) {
         if (this.y_nticks > 8) this.y_nticks = 8;
         var timeformaty = JSROOT.Painter.getTimeFormat(this.histo['fYaxis']);

         this['timeoffsety'] = JSROOT.Painter.getTimeOffset(this.histo['fYaxis']);

         var scale_yrange = this.scale_ymax - this.scale_ymin;

         if ((timeformaty.length == 0) || (scale_yrange < 0.1*yrange))
            timeformaty = JSROOT.Painter.chooseTimeFormat(scale_yrange, this.y_nticks);

         this['dfy'] = d3.time.format(timeformaty);

         this['y_axis'] = d3.svg.axis()
               .scale(this.y)
               .orient("left")
               .tickPadding(yAxisLabelOffset)
               .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
               .tickFormat(function(d) { return pthis.dfy(new Date(pthis.timeoffsety + d * 1000)); })
               .ticks(this.y_nticks);
      }
      else if (this.options.Logy) {
         this['y_axis'] = d3.svg.axis()
            .scale(this.y)
            .orient("left")
            .tickPadding(yAxisLabelOffset)
            .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
            .tickFormat(function(d) {
               var val = parseFloat(d);
               var vlog = Math.abs(JSROOT.Math.log10(val));
               if (moreloglabelsy) {
                  if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                     if (noexpy) return val.toFixed();
                     else return JSROOT.Painter.formatExp(val.toExponential(0));
                  }
                  else return null;
               }
               else {
                  if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                     if (noexpy) return val.toFixed();
                     else return JSROOT.Painter.formatExp(val.toExponential(0));
                  }
                  else return null;
               }});
      }
      else {
         if (this.y_nticks >= 10) this.y_nticks -= 2;
         this['y_axis'] = d3.svg.axis()
           .scale(this.y)
           .orient("left")
           .tickPadding(yAxisLabelOffset)
           .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
           .tickFormat(function(d) {
              if ((Math.abs(d)<1e-14) && (Math.abs(yrange)>1e-5)) d = 0;
              return parseFloat(d.toPrecision(12));
           })
           .ticks(this.y_nticks);

      }

      // this is additional ticks, required in d3.v3
      if ((n2ax>0) && !this.options.Logx)
        this['x_axis_sub'] = d3.svg.axis()
         .scale(this.x)
         .orient("bottom")
         .tickPadding(xAxisLabelOffset)
         .innerTickSize(-xDivLength/2)
         .tickFormat(function(d) { return; })
         .ticks(this.x.ticks(this.x_nticks).length*n2ax);

      // this is additional ticks, required in d3.v3
      if ((n2ay>0) && !this.options.Logy) {
         this['y_axis_sub'] = d3.svg.axis()
          .scale(this.y)
          .orient("left")
          .tickPadding(yAxisLabelOffset)
          .innerTickSize(-yDivLength/2)
          .tickFormat(function(d) { return; })
          .ticks(this.y.ticks(this.y_nticks).length*n2ay);
      }


      if ('xax' in this) this['xax'].remove();
      if ('xaxsub' in this) this['xaxsub'].remove();

      this['xax'] =
         this.frame.append("svg:g")
                   .attr("class", "xaxis")
                   .attr("transform", "translate(0," + h + ")")
                   .call(this.x_axis);

      if (this['x_axis_sub'])
         this['xaxsub'] =
            this.frame.append("svg:g")
                      .attr("class", "xaxis")
                      .attr("transform", "translate(0," + h + ")")
                      .call(this.x_axis_sub);

      if ('yax' in this) this['yax'].remove();
      if ('yaxsub' in this) this['yaxsub'].remove();

      this['yax'] = this.frame.append("svg:g").attr("class", "yaxis").call(this.y_axis);
      
      if (this['y_axis_sub'])
         this['yaxsub'] = this.frame.append("svg:g").attr("class", "yaxis").call(this.y_axis_sub);

      var xAxisLabelFontDetails = JSROOT.Painter.getFontDetails(this.histo['fXaxis']['fLabelFont']);
      var yAxisLabelFontDetails = JSROOT.Painter.getFontDetails(this.histo['fYaxis']['fLabelFont']);

      this.xax.selectAll("text")
        .attr("font-family", xAxisLabelFontDetails['name'])
        .attr("font-size", xAxisLabelFontSize)
        .attr("font-weight", xAxisLabelFontDetails['weight'])
        .attr("font-style", xAxisLabelFontDetails['style']);
      this.yax.selectAll("text")
        .attr("font-family", yAxisLabelFontDetails['name'])
        .attr("font-size", yAxisLabelFontSize)
        .attr("font-weight", yAxisLabelFontDetails['weight'])
        .attr("font-style", yAxisLabelFontDetails['style']);

      if ('xax_zoom_rect' in this)
         this.xax_zoom_rect.remove();

      // we will use such rect for zoom selection
      this['xax_zoom_rect'] =
         this.frame.append("svg:rect")
                   .attr("class", "xaxis_zoom")
                   .attr("x", 0)
                   .attr("y", h)
                   .attr("width", w)
                   .attr("height", xAxisLabelFontSize + 3)
                   .style('opacity', 0);

      if ('yax_zoom_rect' in this)
         this.yax_zoom_rect.remove();

      // we will use such rect for zoom selection
      this['yax_zoom_rect'] =
         this.frame.append("svg:rect")
                   .attr("class", "yaxis_zoom")
                   .attr("x", - 2 * yAxisLabelFontSize - 3)
                   .attr("y", 0)
                   .attr("width", 2 * yAxisLabelFontSize + 3)
                   .attr("height", h)
                   .style('opacity', 0);
   }


   JSROOT.THistPainter.prototype.DrawTitle = function() {
      var w = Number(this.vis.attr("width")), h = Number(this.vis.attr("height"));
      var font_size = Math.round(0.050 * h);
      var l_title = JSROOT.Painter.translateLaTeX(this.histo['fTitle']);

      if (!this.pad || typeof(this.pad) == 'undefined') {

         if (!('draw_title' in this))
            this['draw_title'] =
               this.vis.append("text")
                .attr("class", "title")
                .attr("text-anchor", "middle")
                .attr("x", w/2)
                .attr("y", 1 + font_size /* 0.07*h */)
                .attr("font-family", "Arial")
                .attr("font-size", font_size);

         this.draw_title.text(l_title);

         // console.log("title height = " + this.draw_title.node().getBBox().height + "  font size = " + font_size);

         var pthis = this;

         this.AddDrag("title", this.draw_title, {
            move: function(x, y) {
               // pthis.draw_title.attr("x",x).attr("y", y);
            },
            resize: function(width, height) {
               font_size = height*0.8;
               pthis.draw_title.attr("font-size", font_size);
            }
         });

      }
   }

   JSROOT.THistPainter.prototype.ToggleStat = function() {

      var stat = this.FindStat();

      if (stat == null) {

         // when statbox created first time, one need to draw it
         stat = this.CreateStat();

         this.Redraw();

         return;
      }

      var statpainter = this.FindPainterFor(stat);
      if (statpainter == null) {
         alert("Did not found painter for existing stat??");
         return;
      }

      statpainter.Enabled = !statpainter.Enabled;

      // when stat box is drawed, it always can be draw individualy while it should be last
      // for colz RedrawFrame is used
      statpainter.Redraw();
   }

   JSROOT.THistPainter.prototype.GetSelectIndex = function(axis,size,add) {
      // be aware - here index starts from 0
      var indx = 0;
      var obj = this.first;
      if (obj==null) obj = this;
      var nbin = 0;
      if (!add) add = 0;

      if (axis == "x") {
         nbin = this.nbinsx;
         if (obj.zoom_xmin != obj.zoom_xmax) {
            if (size=="left")
               indx = Math.floor((obj.zoom_xmin - this.xmin) / this.binwidthx + add);
            else
               indx = Math.round((obj.zoom_xmax - this.xmin) / this.binwidthx + 0.5 + add);
         } else {
            indx = (size=="left") ? 0 : nbin;
         }

      } else
      if (axis == "y") {
         nbin = this.nbinsy;
         if (obj.zoom_ymin != obj.zoom_ymax) {
            if (size=="left")
               indx = Math.floor((obj.zoom_ymin - this.ymin) / this.binwidthy + add);
            else
               indx = Math.round((obj.zoom_ymax - this.ymin) / this.binwidthy + 0.5 + add);
         } else {
            indx = (size=="left") ? 0 : nbin;
         }
      }

      if (size=="left") {
         if (indx<0) indx = 0;
      } else {
         if (indx>nbin) indx = nbin;
      }

      return indx;
   }

   JSROOT.THistPainter.prototype.FindStat = function() {

      if ('fFunctions' in this.histo)
         for (var i in this.histo.fFunctions.arr) {

            var func = this.histo.fFunctions.arr[i];

            if (func['_typename'] == 'TPaveText' ||
                func['_typename'] == 'TPaveStats') {

               return func;
            }
         }

      return null;
   }

   JSROOT.THistPainter.prototype.CreateStat = function() {

      if (!this.draw_content) return null;
      if (this.FindStat() != null) return null;

      var stats = {};
      stats['_typename'] = 'TPaveStats';
      stats['fName'] = 'stats';

      stats['_AutoCreated'] = true;
      stats['fX1NDC'] = JSROOT.gStyle.StatX;
      stats['fY1NDC'] = JSROOT.gStyle.StatY;
      stats['fX2NDC'] = JSROOT.gStyle.StatX + JSROOT.gStyle.StatW;
      stats['fY2NDC'] = JSROOT.gStyle.StatY + JSROOT.gStyle.StatH;
      if (this.histo['_typename'] && 
          (this.histo['_typename'].match(/\bTProfile/) || this.histo['_typename'].match(/\bTH2/)))
             stats['fY1NDC'] = 0.67;

      stats['fOptFit'] = 0;
      stats['fOptStat'] = JSROOT.gStyle.OptStat;
      stats['fLongest'] = 17;
      stats['fMargin'] = 0.05;

      stats['fBorderSize'] = 1;
      stats['fInit'] = 1;
      stats['fShadowColor'] = 1;
      stats['fCornerRadius'] = 0;

      stats['fX1'] = 1;
      stats['fY1'] = 100;
      stats['fX2'] = 1;
      stats['fY2'] = 100;

      stats['fResizing'] = false;
      stats['fUniqueID'] = 0;
      stats['fBits'] = 0x03000009;
      stats['fLineColor'] = 1;
      stats['fLineStyle'] = 1;
      stats['fLineWidth'] = 1;

      stats['fFillColor'] = JSROOT.gStyle.StatColor;
      stats['fFillStyle'] = JSROOT.gStyle.StatStyle;

      stats['fTextAngle'] = 0;
      stats['fTextSize'] = JSROOT.gStyle.StatFontSize;
      stats['fTextAlign'] = 12;
      stats['fTextColor'] = JSROOT.gStyle.StatTextColor;
      stats['fTextFont'] = JSROOT.gStyle.StatFont;

      stats['fLines'] = JSROOT.CreateTList();

      stats['fLines'].arr.push({'fTitle': "hname", "fTextColor": 1});
//      stats['fLines'].arr.push({'fTitle': "Entries = 4075", "fTextColor": 1});
//      stats['fLines'].arr.push({'fTitle': "Mean = 2000", "fTextColor": 1});
//      stats['fLines'].arr.push({'fTitle': "RMS = 3000", "fTextColor": 1});

      stats['fLines'].arr[0]['fTitle'] = this.histo['fName'];

      if (!'fFunctions' in this.histo)
         this.histo['fFunctions'] = JSROOT.CreateTList();

      this.histo.fFunctions.arr.push(stats);

      return stats;
   }

   JSROOT.THistPainter.prototype.FindPalette = function() {

      if ('fFunctions' in this.histo)
         for (var i in this.histo.fFunctions.arr) {
            var func = this.histo.fFunctions.arr[i];
            if (func['_typename'] == 'TPaletteAxis') return func;
         }

      return null;
   }


   JSROOT.THistPainter.prototype.DrawFunctions = function() {

      // draw statistics box & other TPaveTexts, which are belongs to histogram
      // should be called once to create all painters, which are than updated separately

      if (!('fFunctions' in this.histo)) return;

      var lastpainter = this;
      
      var kNotDraw = JSROOT.BIT(9);  // don't draw the function (TF1) when in a TH1
      
      var EStatusBits = {
            kCanDelete     : JSROOT.BIT(0),   // if object in a list can be deleted
            kMustCleanup   : JSROOT.BIT(3),   // if object destructor must call RecursiveRemove()
            kObjInCanvas   : JSROOT.BIT(3),   // for backward compatibility only, use kMustCleanup
            kIsReferenced  : JSROOT.BIT(4),   // if object is referenced by a TRef or TRefArray
            kHasUUID       : JSROOT.BIT(5),   // if object has a TUUID (its fUniqueID=UUIDNumber)
            kCannotPick    : JSROOT.BIT(6),   // if object in a pad cannot be picked
            kNoContextMenu : JSROOT.BIT(8),   // if object does not want context menu
            kInvalidObject : JSROOT.BIT(13)   // if object ctor succeeded but object should not be used
      };

      for (var i in this.histo.fFunctions.arr) {

         var func = this.histo.fFunctions.arr[i];

         var funcpainter = this.FindPainterFor(func);

         // no need to do something if painter for object was already done
         // object will be redraw automatically
         if (funcpainter==null) {

            if (func['_typename'] == 'TPaveText' ||
                func['_typename'] == 'TPaveStats') {
               funcpainter = JSROOT.Painter.DrawPaveText(this.vis, func);
            }

            if (func['_typename'] == 'TF1') {
               if ((!this.pad && !func.TestBit(kNotDraw)) ||
                   (this.pad && func.TestBit(EStatusBits.kObjInCanvas)))
                  funcpainter = JSROOT.Painter.drawFunction(this.vis, func);
            }

            if (func['_typename'] == 'TPaletteAxis') {
               funcpainter = JSROOT.Painter.drawPaletteAxis(this.vis, func);
            }
         }

         if (func['_typename'] == 'TPaletteAxis' && funcpainter) {
            funcpainter.Enabled = (this.options.Zscale > 0) && (this.options.Color>0);
         }

         // we do it to preserve order in which objects are drawn
         // therefore we need to guarantee that painters are in the same order
         if (funcpainter!=null) {
            lastpainter.PlacePainterAfterMe(funcpainter);
            lastpainter = funcpainter;
         }
      }
   }

   JSROOT.THistPainter.prototype.Redraw = function(resize) {
      
      if (resize) JSROOT.Painter.createFrame(this.vis);
      
      this.CreateXY();
      this.CountStat();
      this.DrawAxes();
      this.DrawGrids();
      this.DrawBins();
      this.DrawTitle();
      this.DrawFunctions();
   }

   JSROOT.THistPainter.prototype.AddInteractive = function() {
      // only first painter in list allowed to add interactive functionality to the main pad
      if (this.first) return;

//      if (!this.draw_content) return;

      var width = Number(this.frame.attr("width")), height = Number(this.frame.attr("height"));
      var e, origin, curr = null, rect = null;
      var lasttouch = new Date(0);

      var zoom_kind = 0;  // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)

      // var zoom = d3.behavior.zoom().x(this.x).y(this.y);

      var pthis = this;

      function closeAllExtras() {
         var x = document.getElementById('root_ctx_menu');
         if(x) x.parentNode.removeChild(x);
         if (rect != null) { rect.remove(); rect = null; }
         zoom_kind = 0;
      }
      
      function showContextMenu() {

         d3.event.preventDefault();

         // ignore context menu when touches zooming is ongoing
         if (zoom_kind>100) return;

         // suppress any running zomming
         closeAllExtras();

         var menu = JSROOT.Painter.createmenu(d3.event, 'root_ctx_menu');
         
         menu['painter'] = pthis;
         
         JSROOT.Painter.menuitem(menu, pthis.histo['fName']);
         JSROOT.Painter.menuitem(menu, "----------------");

         pthis.FillContextMenu(menu);
       }
      
      function startTouchSel() {

         // in case when zooming was started, block any other kind of events
         if (zoom_kind!=0)  {
            d3.event.preventDefault();
            d3.event.stopPropagation();
            return;
         }
         
         e = this;
         // var t = d3.event.changedTouches;
         var arr = d3.touches(e);

         // only double-touch will be handled
         if (arr.length == 1) {

            var now = new Date();
            var diff = now.getTime() - lasttouch.getTime();

            if ((diff < 300) &&
                (curr != null) && (Math.abs(curr[0] - arr[0][0]) < 30) &&
                (Math.abs(curr[1] - arr[0][1]) < 30)) {

               d3.event.preventDefault();
               d3.event.stopPropagation();

               closeAllExtras();
               pthis.Unzoom(true,true);
            } else {
               lasttouch = now;
               curr = arr[0];
            }
         }

         if (arr.length != 2) return;

         d3.event.preventDefault();

         closeAllExtras();

         var pnt1 = arr[0];
         var pnt2 = arr[1];

         curr = new Array; // minimum
         origin = new Array; // maximum

         curr.push(Math.min(pnt1[0], pnt2[0]));
         curr.push(Math.min(pnt1[1], pnt2[1]));
         origin.push(Math.max(pnt1[0], pnt2[0]));
         origin.push(Math.max(pnt1[1], pnt2[1]));

         if (curr[0]<0) {
            zoom_kind = 103; // only y
            curr[0] = 0;
            origin[0] = width;
         } else
         if (origin[1] > height) {
            zoom_kind = 102; // only x
            curr[1] = 0;
            origin[1] = height;
         } else {
            zoom_kind = 101; // x and y
         }

         // d3.select("body").classed("noselect", true);
         // d3.select("body").style("-webkit-user-select", "none");

         rect = pthis.frame
                 .append("rect")
                 .attr("class", "zoom")
                 .attr("id", "zoomRect")
                 .attr("x", curr[0])
                 .attr("y", curr[1])
                 .attr("width", origin[0] - curr[0])
                 .attr("height", origin[1] - curr[1]);

         // pthis.frame.on("dblclick", unZoom);

         d3.select(window)
            .on("touchmove.zoomRect", moveTouchSel)
            .on("touchcancel.zoomRect", endTouchSel)
            .on("touchend.zoomRect", endTouchSel, true);
         d3.event.stopPropagation();
      }


      function moveTouchSel() {

         if (zoom_kind<100) return;

         d3.event.preventDefault();

         //var t = d3.event.changedTouches;
         var arr = d3.touches(e);

         if (arr.length != 2) {
            closeAllExtras();
            zoom_kind = 0;
            return;
         }

         var pnt1 = arr[0];
         var pnt2 = arr[1];

         if (zoom_kind != 103) {
            curr[0] = Math.min(pnt1[0], pnt2[0]);
            origin[0] = Math.max(pnt1[0], pnt2[0]);
         }
         if (zoom_kind != 102) {
            curr[1] = Math.min(pnt1[1], pnt2[1]);
            origin[1] = Math.max(pnt1[1], pnt2[1]);
         }

         rect.attr("x", curr[0])
             .attr("y", curr[1])
             .attr("width", origin[0] - curr[0])
             .attr("height", origin[1] - curr[1]);

         d3.event.stopPropagation();
      };

      function endTouchSel() {

         if (zoom_kind<100) return;

         d3.event.preventDefault();
         d3.select(window).on("touchmove.zoomRect", null).on("touchend.zoomRect", null).on("touchcancel.zoomRect", null);
         d3.select("body").classed("noselect", false);

         var xmin=0, xmax = 0, ymin = 0, ymax = 0;

         var isany = false;

         if ((zoom_kind != 103) && (Math.abs(curr[0] - origin[0]) > 10)) {
            xmin = Math.min(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            xmax = Math.max(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            isany = true;
         }

         if ((zoom_kind != 102) && (Math.abs(curr[1] - origin[1]) > 10)) {
            ymin = Math.min(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            ymax = Math.max(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            isany = true;
         }

         d3.select("body").style("-webkit-user-select", "auto");

         rect.remove();
         rect = null;
         zoom_kind = 0;

         if (isany) pthis.Zoom(xmin, xmax, ymin, ymax);

         d3.event.stopPropagation();
      }

      function startRectSel() {

         // ignore when touch selection is actiavated
         if (zoom_kind>100) return;

         d3.event.preventDefault();

         closeAllExtras();

         e = this;
         origin = d3.mouse(e);

         curr = new Array;
         curr.push(Math.max(0, Math.min(width, origin[0])));
         curr.push(Math.max(0, Math.min(height, origin[1])));

         if (origin[0] < 0) {
            zoom_kind = 3; // only y
            origin[0] = 0;
            origin[1] = curr[1];
            curr[0] = width;
            curr[1] += 1;
         } else
         if (origin[1] > height) {
            zoom_kind = 2; // only x
            origin[0] = curr[0];
            origin[1] = 0;
            curr[0] += 1;
            curr[1] = height;
         } else {
            zoom_kind = 1; // x and y
            origin[0] = curr[0];
            origin[1] = curr[1];
         }

         //d3.select("body").classed("noselect", true);
         //d3.select("body").style("-webkit-user-select", "none");

         rect = pthis.frame
                 .append("rect")
                 .attr("class", "zoom")
                 .attr("id", "zoomRect");

         pthis.frame.on("dblclick", unZoom);

         d3.select(window)
            .on("mousemove.zoomRect", moveRectSel)
            .on("mouseup.zoomRect", endRectSel, true);

         d3.event.stopPropagation();
      };

      function unZoom() {
         d3.event.preventDefault();

         var m = d3.mouse(e);

         closeAllExtras();

         if (m[0] < 0) pthis.Unzoom(false,true); else
         if (m[1] > height) pthis.Unzoom(true,false); else {
            pthis.Unzoom(true,true);
            pthis.frame.on("dblclick", null);
         }
      };

      function moveRectSel() {

         if ((zoom_kind==0) || (zoom_kind>100)) return;

         d3.event.preventDefault();
         var m = d3.mouse(e);

         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));

         switch (zoom_kind) {
            case 1: curr[0] = m[0]; curr[1] = m[1]; break;
            case 2: curr[0] = m[0]; break;
            case 3: curr[1] = m[1]; break;
         }

         rect.attr("x", Math.min(origin[0], curr[0]))
             .attr("y", Math.min(origin[1], curr[1]))
             .attr("width", Math.abs(curr[0] - origin[0]))
             .attr("height", Math.abs(curr[1] - origin[1]));
      };

      function endRectSel() {

         if ((zoom_kind==0) || (zoom_kind>100)) return;

         d3.event.preventDefault();
//         d3.select(window).on("touchmove.zoomRect", null).on("touchend.zoomRect", null);
         d3.select(window).on("mousemove.zoomRect", null).on("mouseup.zoomRect", null);
         d3.select("body").classed("noselect", false);

         var m = d3.mouse(e);

         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));

         switch (zoom_kind) {
            case 1: curr[0] = m[0]; curr[1] = m[1]; break;
            case 2: curr[0] = m[0]; break; // only X
            case 3: curr[1] = m[1]; break; // only Y
         }

         var xmin=0, xmax = 0, ymin = 0, ymax = 0;

         var isany = false;

         if ((zoom_kind != 3) && (Math.abs(curr[0] - origin[0]) > 10)) {
            xmin = Math.min(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            xmax = Math.max(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            isany = true;
         }

         if ((zoom_kind != 2) && (Math.abs(curr[1] - origin[1]) > 10)) {
            ymin = Math.min(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            ymax = Math.max(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            isany = true;
         }

         d3.select("body").style("-webkit-user-select", "auto");

         rect.remove();
         rect = null;
         zoom_kind = 0;

         if (isany) pthis.Zoom(xmin, xmax, ymin, ymax);
      }
      
      this.frame.on("mousedown", startRectSel);
      this.frame.on("touchstart", startTouchSel);
      this.frame.on("contextmenu", showContextMenu);

   }

   JSROOT.THistPainter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.TBasePainter.prototype.FillContextMenu.call(this, menu);
      if (this.options) {
         
         var item =  this.options.Logx > 0 ? "Linear X" : "Log X";  
         
         JSROOT.Painter.menuitem(menu, item, function() { 
            menu['painter'].options.Logx = 1 - menu['painter'].options.Logx;
            menu['painter'].RedrawFrame();
         });
         
         var item =  this.options.Logy > 0 ? "Linear Y" : "Log Y";  
         JSROOT.Painter.menuitem(menu, item, function() { 
            menu['painter'].options.Logy = 1 - menu['painter'].options.Logy;
            menu['painter'].RedrawFrame();
         });
      }
      if (this.draw_content)
         JSROOT.Painter.menuitem(menu, "Toggle stat", function() { 
            menu['painter'].ToggleStat();
         });
   }



   // ======= TH1 painter =======================================================

   JSROOT.TH1Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
      this.draw_bins = null;
   }

   JSROOT.TH1Painter.prototype = Object.create( JSROOT.THistPainter.prototype );

   JSROOT.TH1Painter.prototype.IsTProfile = function() {
      return this.histo && this.histo['_typename'] == 'TProfile';
   }

   JSROOT.TH1Painter.prototype.ScanContent = function() {

      // from here we analyze object content
      // therefore code will be moved
      this.fillcolor = JSROOT.Painter.root_colors[this.histo['fFillColor']];
      this.linecolor = JSROOT.Painter.root_colors[this.histo['fLineColor']];

      if (this.histo['fFillColor'] == 0) this.fillcolor = '#4572A7';
      if (this.histo['fLineColor'] == 0) this.linecolor = '#4572A7';

      var hmin = 0, hmax = 0, hsum = 0;
      // this.stat_entries = d3.sum(this.histo['fArray']);
      
      var profile = this.IsTProfile();

      this.nbinsx = this.histo['fXaxis']['fNbins'];
      
      for (var i=0;i<this.nbinsx;++i) {
         var value = this.histo.getBinContent(i+1);
         hsum += profile ? this.histo.fBinEntries[i+1] : value;
         if (i==0) { hmin = hmax = value; }
         if (value < hmin) hmin = value; else
         if (value > hmax) hmax = value;
      }

      // account overflow/underflow bins
      if (profile)
         hsum += this.histo.fBinEntries[0] + this.histo.fBinEntries[this.nbinsx+1];
      else
         hsum += this.histo.getBinContent(0) + this.histo.getBinContent(this.nbinsx+1);

      this.stat_entries = hsum;

      // if (('fBuffer' in this.histo) && (this.histo['fBuffer'].length>0)) this.stat_entries = this.histo['fBuffer'][0];
      // if ((this.stat_entries == 0) && ('fEntries' in this.histo)) this.stat_entries = this.histo['fEntries'];

      // used in CreateXY and tooltip providing
      this.xmin = this.histo['fXaxis']['fXmin'];
      this.xmax = this.histo['fXaxis']['fXmax'];

      this.binwidthx = (this.xmax - this.xmin);
      if (this.nbinsx>0)
         this.binwidthx = this.binwidthx / this.nbinsx;

      this.ymin = this.histo['fYaxis']['fXmin'];
      this.ymax = this.histo['fYaxis']['fXmax'];

      // console.log("histo fMinimum = " + this.histo['fMinimum'] + "  fMaximum = " + this.histo['fMaximum']);
      // console.log("hmin = " + hmin + "  hmax = " + hmax + "  " + (typeof hmax));
      
      if ((this.nbinsx==0) || ((Math.abs(hmin) < 1e-300 && Math.abs(hmax) < 1e-300))) {
         if (this.histo['fMinimum'] != -1111) this.ymin = this.histo['fMinimum'];
         if (this.histo['fMaximum'] != -1111) this.ymax = this.histo['fMaximum'];
         this.draw_content = false;
      } else {
         if (this.histo['fMinimum'] != -1111) hmin = this.histo['fMinimum'];
         if (this.histo['fMaximum'] != -1111) hmax = this.histo['fMaximum'];
         if (hmin >= hmax) {
            if (hmin == 0) { this.ymax = 0; this.ymax = 1;  }
            else if (hmin<0) { this.ymin = 2*hmin; this.ymax = 0; }
            else { this.ymin = 0; this.ymax = hmin*2; }
         } else {
            var dy = (hmax-hmin) * 0.1;
            this.ymin = hmin - dy;
            if ((this.ymin<0) && (hmin>=0)) this.ymin = 0;
            this.ymax = hmax + dy;
         }
         this.draw_content = true;
      }
       //console.log("xmin = " + this.xmin + "  xmax = " + this.xmax + "  nbins = " + this.nbinsx);
       //console.log("ymin = " + this.ymin + "  ymax = " + this.ymax);

      // If no any draw options specified, do not try draw histogram
      if (this.options.Bar == 0 && this.options.Hist == 0 &&
          this.options.Error == 0 && this.options.Same == 0) {
         this.draw_content = false;
      }
      if (this.options.Axis > 0) // Paint histogram axis only
         this.draw_content = false;
   }

   JSROOT.TH1Painter.prototype.CountStat = function()
   {
      var profile = this.IsTProfile();
      
      if (profile) {
         this.stat_sumw = 0;
         this.stat_sumwx = 0;
         this.stat_sumwx2 = 0;
         this.stat_sumwy = 0;
         this.stat_sumwy2 = 0;
      } else {
         this.stat_sum0 = 0;
         this.stat_sum1 = 0;
         this.stat_sum2 = 0;
      }
      

      var left = this.GetSelectIndex("x","left");
      var right = this.GetSelectIndex("x","right");

      var xx = 0, yy = 0, w = 0;
      // console.log("  xleft = " + left + " xright = " + right);
      
      for (var i=left;i<right;i++) {
         xx = this.xmin + (i+0.5)*this.binwidthx;
         
         if (profile) {
            w  = this.histo.fBinEntries[i+1];
            this.stat_sumw += w;
            this.stat_sumwx += w*xx;
            this.stat_sumwx2 += w*xx*xx;
            this.stat_sumwy += this.histo.fArray[i+1];
            this.stat_sumwy2 += this.histo.fSumw2[i+1];  
         } else {
            yy = this.histo.getBinContent(i+1);
            this.stat_sum0 += yy;
            this.stat_sum1 += xx * yy;
            this.stat_sum2 += xx * xx * yy;
         }
      }
   }

   JSROOT.TH1Painter.prototype.FillStatistic = function(stat, dostat)
   {
      if (!this.histo) return false;

      var print_name    = Math.floor(dostat%10);
      var print_entries = Math.floor(dostat/10)%10;
      var print_mean    = Math.floor(dostat/100)%10;
      var print_rms     = Math.floor(dostat/1000)%10;
      var print_under   = Math.floor(dostat/10000)%10;
      var print_over    = Math.floor(dostat/100000)%10;
      var print_integral= Math.floor(dostat/1000000)%10;
      var print_skew    = Math.floor(dostat/10000000)%10;
      var print_kurt    = Math.floor(dostat/100000000)%10;

      if (print_name > 0)
         stat.AddLine(this.histo['fName']);

      
      if (this.IsTProfile()) {

         if (print_entries > 0)
            stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(this.stat_entries));

         var meanx = 0, meany = 0;
         if (this.stat_sumw > 0) {
            meanx = this.stat_sumwx/this.stat_sumw;
            meany = this.stat_sumwy/this.stat_sumw;
         }

         if (print_mean > 0) {
            stat.AddLine("Mean = " + JSROOT.gStyle.StatFormat(meanx));
            stat.AddLine("Mean y = " + JSROOT.gStyle.StatFormat(meany));
         }

         if (print_rms > 0) {
            var rmsx = 0, rmsy = 0;
            if (this.stat_sumw > 0) {
               rmsx = Math.sqrt(this.stat_sumwx2/this.stat_sumw - meanx*meanx);
               rmsy = Math.sqrt(this.stat_sumwy2/this.stat_sumw - meany*meany);
            }
            stat.AddLine("RMS = " + JSROOT.gStyle.StatFormat(rmsx));
            stat.AddLine("RMS y = " + JSROOT.gStyle.StatFormat(rmsy));
         }
         
      } else {

         if (print_entries > 0)
            stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(this.stat_entries));

         if (print_mean > 0) {
            var res = 0;
            if (this.stat_sum0 > 0) res = this.stat_sum1/this.stat_sum0;
            stat.AddLine("Mean = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_rms > 0) {
            var res = 0;
            if (this.stat_sum0 > 0)
               res = Math.sqrt(this.stat_sum2/this.stat_sum0 - Math.pow(this.stat_sum1/this.stat_sum0, 2));
            stat.AddLine("RMS = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_under > 0) {
            var res = 0;
            if (this.histo['fArray'].length > 0) res = this.histo['fArray'][0];
            stat.AddLine("Underflow = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_over > 0) {
            var res = 0;
            if (this.histo['fArray'].length > 0) res = this.histo['fArray'][this.histo['fArray'].length-1];
            stat.AddLine("Overflow = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_integral > 0) {
            stat.AddLine("Integral = " + JSROOT.gStyle.StatEntriesFormat(this.stat_sum0));
         }

         if (print_skew> 0)
            stat.AddLine("Skew = not avail");

         if (print_kurt> 0)
            stat.AddLine("Kurt = not avail");
      }

      // adjust the size of the stats box with the number of lines
      var nlines = stat.pavetext['fLines'].arr.length;
      var stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont%10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         stat.pavetext['fY1NDC'] = 0.93 - stath;
         stat.pavetext['fY2NDC'] = 0.93;
      }

      return true;
   }

   JSROOT.TH1Painter.prototype.CreateDrawBins = function()
   {
      // method is called directly before bins must be drawn

      var left = this.GetSelectIndex("x","left",-1);
      var right = this.GetSelectIndex("x","right",2);
      var width = Number(this.svg_frame.attr("width"));
      var stepi = 1;

      this.draw_bins = new Array;

      // reduce number of drawn points - we define interval where two points will be selected - max and min
      if ((this.nbinsx > 10000) || (JSROOT.gStyle.OptimizeDraw && (right - left > width)))
         while ((right - left)/stepi > width) stepi++;

      var x1, x2 = this.xmin + left*this.binwidthx;
      var grx1 = -1111, grx2 = -1111, gry;

      // console.log("left " + left + " right " + right + " step " +  stepi);

      var point;

      for (var i = left; i<right; i+=stepi) {
         // if interval wider than specified range, make it shorter
         if ((stepi>1) && (i+stepi>right)) stepi = (right-i);
         x1 = x2; x2 += stepi*this.binwidthx;

         grx1 = grx2; grx2 = this.x(x2);
         if (grx1 < 0) grx1 = this.x(x1);

         var pmax = i, cont = this.histo.getBinContent(i+1);

         for (var ii=1;ii<stepi;ii++) {
            var ccc = this.histo.getBinContent(i+ii+1);
            if (ccc>cont) { cont = ccc; pmax = i + ii; }
         }

         gry = this.y(cont);

         point = { x: grx1, y: gry };

         if (this.options.Error > 0) {
            point['xerr'] = (grx2 - grx1) / 2;
            point['yerr'] =  gry - this.y(cont + this.histo.getBinError(pmax+1));
         }

         if (JSROOT.gStyle.Tooltip) {
            if (this.options.Error > 0) {
               point['x'] = (grx1 + grx2)/2;
               point['tip'] = "x = " + this.AxisAsText("x", x1) + " \ny = " + this.AxisAsText("y",cont) +
                             " \nerror x = " + ((x2-x1)/2).toPrecision(4) +
                             " \nerror y = " + this.histo.getBinError(pmax+1).toPrecision(4);
            }  else {
               point['width'] = grx2-grx1;

               point['tip'] = "bin = " + (pmax+1) + "\n" +
                              "x = [" + this.AxisAsText("x",x1) +", " + this.AxisAsText("x",x2) + "]\n" +
                              "entries = " + cont;
            }
         }

         this.draw_bins.push(point);
      }

      // if we need to draw line or area, we need extra point for correct drawing
      if ((right == this.nbinsx) && (this.options.Error == 0)) {
         var extrapoint = jQuery.extend(true, {}, point);
         extrapoint.x = grx2;
         this.draw_bins.push(extrapoint);
      }
   }


   JSROOT.TH1Painter.prototype.DrawErrors = function()
   {
      var w = Number(this.svg_frame.attr("width")), h = Number(this.svg_frame.attr("height"));
      /* Add a panel for each data point */
      var info_marker = JSROOT.Painter.getRootMarker(this.histo['fMarkerStyle']);
      var shape = info_marker['shape'], filled = info_marker['toFill'],
          toRotate = info_marker['toRotate'], marker_size = this.histo['fMarkerSize'] * 32;

      var line_width = this.histo['fLineWidth'];
      var line_color = JSROOT.Painter.root_colors[this.histo['fLineColor']];
      var marker_color = JSROOT.Painter.root_colors[this.histo['fMarkerColor']];

      if (this.histo['fMarkerStyle'] == 1) marker_size = 1;

      var marker = d3.svg.symbol()
          .type(d3.svg.symbolTypes[shape])
          .size(marker_size);

      var pthis = this;

      function TooltipText(d) { return d.tip; }

      /* Draw x-error indicators */
      var xerr = this.draw_g.selectAll("error_x")
            .data(this.draw_bins)
            .enter()
            .append("svg:line")
            .attr("x1", function(d) { return d.x-d.xerr; })
            .attr("y1", function(d) { return d.y; })
            .attr("x2", function(d) { return d.x+d.xerr; })
            .attr("y2", function(d) { return d.y; })
            .style("stroke", line_color)
            .style("stroke-width", line_width);

      if (this.options.Error == 11) {
           this.draw_g.selectAll("e1_x")
               .data(this.draw_bins)
               .enter()
               .append("svg:line")
               .attr("y1", function(d) { return d.y-3; })
               .attr("x1", function(d) { return d.x-d.xerr; })
               .attr("y2", function(d) { return d.y+3; })
               .attr("x2", function(d) { return d.x-d.xerr; })
               .style("stroke", line_color)
               .style("stroke-width", line_width);
           this.draw_g.selectAll("e1_x")
               .data(this.draw_bins)
               .enter()
               .append("svg:line")
               .attr("y1", function(d) { return d.y-3; })
               .attr("x1", function(d) { return d.x+d.xerr; })
               .attr("y2", function(d) { return d.y+3; })
               .attr("x2", function(d) { return d.x+d.xerr; })
               .style("stroke", line_color)
               .style("stroke-width", line_width);
      }

      /* Draw y-error indicators */
      var yerr = this.draw_g.selectAll("error_y")
            .data(this.draw_bins)
            .enter()
            .append("svg:line")
            .attr("x1", function(d) { return d.x; })
            .attr("y1", function(d) { return d.y-d.yerr; })
            .attr("x2", function(d) { return d.x; })
            .attr("y2", function(d) { return d.y+d.yerr;  })
            .style("stroke", line_color)
            .style("stroke-width", line_width);

      if (this.options.Error == 11) {
         this.draw_g.selectAll("e1_y")
               .data(this.draw_bins)
               .enter()
               .append("svg:line")
               .attr("x1", function(d) { return d.x-3; })
               .attr("y1", function(d) { return d.y-d.yerr; })
               .attr("x2", function(d) { return d.x+3; })
               .attr("y2", function(d) { return d.y-d.yerr; })
               .style("stroke", line_color)
               .style("stroke-width", line_width);
         this.draw_g.selectAll("e1_y")
               .data(this.draw_bins)
               .enter()
               .append("svg:line")
               .attr("x1", function(d) { return d.x-3; })
               .attr("y1", function(d) { return d.y+d.yerr; })
               .attr("x2", function(d) { return d.x+3; })
               .attr("y2", function(d) { return d.y+d.yerr; })
               .style("stroke", line_color)
               .style("stroke-width", line_width);
      }
      var marks = this.draw_g.selectAll("markers")
            .data(this.draw_bins)
            .enter()
            .append("svg:path")
            .attr("class", "marker")
            .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
            .style("fill", marker_color)
            .style("stroke", marker_color)
            .attr("d", marker);

      if (JSROOT.gStyle.Tooltip) {
         marks.append("svg:title").text(TooltipText);
         xerr.append("svg:title").text(TooltipText);
         yerr.append("svg:title").text(TooltipText);
      }
   }


   JSROOT.TH1Painter.prototype.DrawBins = function() {

      // TODO: limit number of drawn by number of visible pixels
      // one could select every second bin, for instance

      this.RemoveDraw();

      delete this.draw_bins; this.draw_bins = null;

      if (!this.draw_content) return;

      this.CreateDrawBins();

      this.draw_g = this.svg_frame.append("svg:g");

      if (this.options.Error > 0) { this.DrawErrors(); return; }

      var width = Number(this.svg_frame.attr("width")), height = Number(this.svg_frame.attr("height"));

      var pthis = this;

      if ((this.histo['fFillStyle'] < 4000 || this.histo['fFillStyle'] > 4100) && this.histo['fFillColor'] != 0) {

         // histogram filling
         var area = d3.svg.area()
            .x(function(d) { return d.x; })
            .y0(function(d) { return d.y; })
            .y1(function(d) { return height; })
            .interpolate("step-after");

         if (this.histo['fFillStyle'] > 3000 && this.histo['fFillStyle'] <= 3025) {
            var patternid = JSROOT.Painter.createFillPattern(this.vis, this.histo['fFillStyle'], this.histo['fFillColor']);

            this.draw_g.append("svg:path")
               .attr("d", area(this.draw_bins))
               .style("stroke", this.linecolor)
               .style("stroke-width", this.histo['fLineWidth'])
               .style("fill", "url(#" + patternid + ")")
               .style("antialias", "false");
         }
         else {
            this.draw_g.append("svg:path")
               .attr("d", area(this.draw_bins))
               .style("stroke", this.linecolor)
               .style("stroke-width", this.histo['fLineWidth'])
               .style("fill", this.fillcolor)
               .style("antialias", "false");
         }
      }
      else {

         var line = d3.svg.line()
            .x(function(d) { return d.x; } )
            .y(function(d) { return d.y; } )
            .interpolate("step-after");

         this.draw_g.append("svg:path")
           .attr("d", line(this.draw_bins)) // to draw one bar, one need two points
           .style("stroke", this.linecolor)
           .style("stroke-width", this.histo['fLineWidth'])
           .style("fill", "none")
           .style("stroke-dasharray", this.histo['fLineStyle'] > 1 ? JSROOT.Painter.root_line_styles[this.histo['fLineStyle']] : null)
           .style("antialias", "false");
      }

      if (JSROOT.gStyle.Tooltip) {
         // TODO: limit number of tooltips by number of visible pixels
         this.draw_g.selectAll("selections")
            .data(this.draw_bins)
            .enter()
            .append("svg:line")
            .attr("x1", function(d) { return d.x + d.width/2; } )
            .attr("y1", function(d) { return Math.max(0, d.y); } )
            .attr("x2", function(d) { return d.x + d.width/2; } )
            .attr("y2", function(d) { return height; } )
            .attr("opacity", 0)
            .style("stroke", "#4572A7")
            .style("stroke-width", function(d) { return d.width; })
            .on('mouseover', function() { d3.select(this).transition().duration(100).style("opacity", 0.3) } )
            .on('mouseout', function() { d3.select(this).transition().duration(100).style("opacity", 0) } )
            .append("svg:title").text(function(d) { return d.tip; });
      }
   }

   JSROOT.TH1Painter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.THistPainter.prototype.FillContextMenu.call(this, menu);
      if (this.draw_content)
         JSROOT.Painter.menuitem(menu, "Auto zoom-in", function() { 
            menu['painter'].AutoZoom();
         });
   }

   JSROOT.TH1Painter.prototype.AutoZoom = function()
   {
      var left = this.GetSelectIndex("x","left",-1);
      var right = this.GetSelectIndex("x","right",1);

      var dist = (right - left);

      if (dist==0) return;

      var min = this.histo.getBinContent(left+1);

      // first find minimum
      for (var indx = left; indx<right; indx++)
         if (this.histo.getBinContent(indx+1) < min)
            min = this.histo.getBinContent(indx+1);

      while ((left<right) && (this.histo.getBinContent(left+1) <= min)) left++;
      while ((left<right) && (this.histo.getBinContent(right) <= min)) right--;

      if ((right - left < dist) && (left < right))
         this.Zoom(this.xmin + left*this.binwidthx, this.xmin + right*this.binwidthx, 0, 0);
   }


   JSROOT.Painter.drawHistogram1D = function(vis, histo, opt) {

      // create painter and add it to canvas
      var painter = new JSROOT.TH1Painter(histo);

      var hadframe = (vis['ROOT:frame'] != null);

      painter.SetFrame(vis, opt);

      painter.ScanContent();

      painter.CreateXY();

      painter.CountStat();

      painter.DrawAxes();

      painter.DrawGrids();

      painter.DrawBins();

      painter.DrawTitle();

      if (JSROOT.gStyle.AutoStat && !hadframe) painter.CreateStat();

      painter.DrawFunctions();

      painter.AddInteractive();

      return painter;
   }


   // ==================== painter for TH2 histograms ==============================

   JSROOT.TH2Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
      this.paletteColors = [];
   }

   JSROOT.TH2Painter.prototype = Object.create( JSROOT.THistPainter.prototype );


   JSROOT.TH2Painter.prototype.FillContextMenu = function(menu)
   {
      JSROOT.THistPainter.prototype.FillContextMenu.call(this, menu);
      JSROOT.Painter.menuitem(menu, "Auto zoom-in", function() { 
         menu['painter'].AutoZoom();
      });
      JSROOT.Painter.menuitem(menu, "Draw in 3D", function() { 
         menu['painter'].Draw3D();
      });
      JSROOT.Painter.menuitem(menu, "Toggle col", function() { 
         menu['painter'].options.Color = 1 - menu['painter'].options.Color;
         menu['painter'].RedrawFrame();
      });
      
      if (this.options.Color > 0)
         JSROOT.Painter.menuitem(menu, "Toggle colz", function() { 
            menu['painter'].ToggleColz();
         });
   }

   JSROOT.TH2Painter.prototype.ToggleColz = function() 
   {
      if (this.FindPalette()==null) {
         var shrink = this.CreatePalette(0.04);
         JSROOT.Painter.shrinkFrame(this.vis, shrink);
         this.options.Zscale = 1;
      } else {
         if (this.options.Zscale>0) this.options.Zscale = 0;
         else this.options.Zscale = 1;
      }

      this.RedrawFrame();
   }

   JSROOT.TH2Painter.prototype.AutoZoom = function()
   {
      var i1 = this.GetSelectIndex("x","left",-1);
      var i2 = this.GetSelectIndex("x","right",1);
      var j1 = this.GetSelectIndex("y","left",-1);
      var j2 = this.GetSelectIndex("y","right",1);

      if ((i1==i2) || (j1==j2)) return;

      var min = this.histo.getBinContent(i1+1, j1+1);

      // first find minimum
      for (var i=i1;i<i2;i++)
         for (var j=j1;j<j2;j++)
            if (this.histo.getBinContent(i+1, j+1)<min)
               min = this.histo.getBinContent(i+1, j+1);

      var ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (var i=i1;i<i2;i++)
         for (var j=j1;j<j2;j++)
            if (this.histo.getBinContent(i+1, j+1)>min) {
               if (i<ileft) ileft = i;
               if (i>=iright) iright = i+1;
               if (j<jleft) jleft = j;
               if (j>=jright) jright = j+1;
            }

      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;

      if ((ileft>i1 || iright<i2) && (ileft < iright-1)) {
         xmin = this.xmin + ileft*this.binwidthx;
         xmax = this.xmin + iright*this.binwidthx;
      }

      if ((jleft>j1 || jright<j2) && (jleft < jright-1)) {
         ymin = this.ymin + jleft*this.binwidthy;
         ymax = this.ymin + jright*this.binwidthy;
      }

      this.Zoom(xmin, xmax, ymin, ymax);
   }


   JSROOT.TH2Painter.prototype.CreatePalette = function(rel_width)
   {
      if (this.FindPalette() != null) return 0.;

      if (!rel_width || rel_width<=0) rel_width = 0.04;

      var pal = {};
      pal['_typename'] = 'TPaletteAxis';
      pal['fName'] = 'palette';

      pal['_AutoCreated'] = true;
      
      pal['fX1NDC'] = this.vis['NDC'].x2 - rel_width;
      pal['fY1NDC'] = this.vis['NDC'].y1;
      pal['fX2NDC'] = this.vis['NDC'].x2; 
      pal['fY2NDC'] = this.vis['NDC'].y2;
      pal['fInit'] = 1;
      pal['fShadowColor'] = 1;
      pal['fCorenerRadius'] = 0;
      pal['fResizing'] = false;
      pal['fBorderSize'] = 4;
      pal['fName'] = "TPave";
      pal['fOption'] = "br";
      pal['fLineColor'] = 1;
      pal['fLineSyle'] = 1;
      pal['fLineWidth'] = 1;
      pal['fFillColor'] = 1;
      pal['fFillSyle'] = 1;

      var axis = {};

      axis['_typename'] = 'TGaxis';
      axis['fTickSize'] =    0.03;
      axis['fLabelOffset'] = 0.005;
      axis['fLabelSize'] =   0.035;
      axis['fTitleOffset'] = 1;
      axis['fTitleSize'] =   0.035;
      axis['fNdiv'] =        8;
      axis['fLabelColor'] =  1;
      axis['fLabelFont'] =   42;
      axis['fChopt'] =       "";
      axis['fName'] =        "";
      axis['fTitle'] =       "";
      axis['fTimeFormat'] =  "";
      axis['fFunctionName'] = "";
      axis['fWmin'] = 0;
      axis['fWmax'] = 100;
      axis['fLineColor'] = 1;
      axis['fLineSyle'] = 1;
      axis['fLineWidth'] = 1;
      axis['fTextAngle'] = 0;
      axis['fTextSize'] = 0.04;
      axis['fTextAlign'] = 11;
      axis['fTextColor'] = 1;
      axis['fTextFont'] = 42;

      pal['fAxis'] = axis;

      if (!'fFunctions' in this.histo)
         this.histo['fFunctions'] = JSROOT.CreateTList();

      // place colz in the beginning, that stat box is always drawn on the top
      this.histo.fFunctions.arr.unshift(pal);
      
      
      // and at the end try to check how much place will be used by the labels in the palette
      
      var width = Number(this.vis.attr("width")), height = Number(this.vis.attr("height"));
      
      var axisOffset = axis['fLabelOffset'] * width;
      var tickSize = axis['fTickSize'] * width;
      var axisFontDetails = JSROOT.Painter.getFontDetails(axis['fLabelFont']);
      var axisLabelFontSize = axis['fLabelSize'] * height;
      
      var ticks = d3.scale.linear().clamp(true)
                   .domain([this.minbin, this.maxbin])
                   .range([height, 0])
                   .nice().ticks(axis['fNdiv'] % 100);

      var maxlen = 0;
      for (var i in ticks) {
         len = JSROOT.Painter.stringWidth(this.vis, ticks[i], axisLabelFontSize, axisFontDetails);
         if (len > maxlen) maxlen = len;
      }

      var rel = (maxlen + axisOffset)/width;
      
      if (pal['fX2NDC'] + rel > 0.98) {
         var shift = pal['fX2NDC'] + rel - 0.98;   

         console.log("Text will overflow pad shift = " + shift);
         pal['fX1NDC']-=shift;
         pal['fX2NDC']-=shift;
         rel_width += shift;
      }
      
      return rel_width + 0.01;
   }


   JSROOT.TH2Painter.prototype.ScanContent = function()
   {
      this.fillcolor = JSROOT.Painter.root_colors[this.histo['fFillColor']];
      this.linecolor = JSROOT.Painter.root_colors[this.histo['fLineColor']];

      //if (this.histo['fFillColor'] == 0) this.fillcolor = '#4572A7'; // why?
      if (this.histo['fLineColor'] == 0) this.linecolor = '#4572A7';

      this.nbinsx = this.histo['fXaxis']['fNbins'];
      this.nbinsy = this.histo['fYaxis']['fNbins'];

      // used in CreateXY method
      this.xmin = this.histo['fXaxis']['fXmin'];
      this.xmax = this.histo['fXaxis']['fXmax'];
      this.ymin = this.histo['fYaxis']['fXmin'];
      this.ymax = this.histo['fYaxis']['fXmax'];

      this.binwidthx = (this.xmax - this.xmin);
      if (this.nbinsx>0) this.binwidthx = this.binwidthx / this.nbinsx;

      this.binwidthy = (this.ymax - this.ymin);
      if (this.nbinsy>0) this.binwidthy = this.binwidthy / this.nbinsy

      this.maxbin = 0;
      this.minbin = 0;
      for (var i=0; i<this.nbinsx; ++i) {
         for (var j=0; j<this.nbinsy; ++j) {
            var bin_content = this.histo.getBinContent(i+1, j+1);
            if (bin_content < this.minbin) this.minbin = bin_content; 
            if (bin_content > this.maxbin) this.maxbin = bin_content;
         }
      }

      // used to enable/disable stat box
      this.draw_content = this.maxbin>0;
      this.maxbin *= 1.05;
   }

   JSROOT.TH2Painter.prototype.CountStat = function()
   {
      this.stat_matrix = new Array();
      for (var n=0;n<9;n++) this.stat_matrix.push(0);
      this.stat_entries = 0;
      this.stat_sum0 = 0;
      this.stat_sumx1 = 0;
      this.stat_sumy1 = 0;
      this.stat_sumx2 = 0;
      this.stat_sumy2 = 0;
      this.stat_sumxy2 = 0;

      var xleft = this.GetSelectIndex("x","left");
      var xright = this.GetSelectIndex("x","right");

      var yleft = this.GetSelectIndex("y","left");
      var yright = this.GetSelectIndex("y","right");

      //console.log("  xleft = " + xleft + " xright = " + xright);
      //console.log("  yleft = " + yleft + " yright = " + yright);

      for (var xi=0;xi<=this.nbinsx+1;xi++) {
         var xside = (xi<=xleft) ? 0 : (xi>xright ? 2 : 1);
         var xx = this.xmin + (xi-0.5)*this.binwidthx;

         for (var yi=0;yi<=this.nbinsx+1;yi++) {
            var yside = (yi<=yleft) ? 0 : (yi>yright ? 2 : 1);
            var yy = this.ymin + (yi-0.5)*this.binwidthy;

            var zz = this.histo.getBinContent(xi,yi);

            this.stat_entries += zz;

            this.stat_matrix[yside*3 + xside]+=zz;

            if ((xside==1) && (yside==1)) {
               this.stat_sum0   += zz;
               this.stat_sumx1  += xx * zz;
               this.stat_sumy1  += yy * zz;
               this.stat_sumx2  += xx * xx * zz;
               this.stat_sumy2  += yy * yy * zz;
               this.stat_sumxy2 += xx * yy * zz;
            }
         }

      }
   }

   JSROOT.TH2Painter.prototype.FillStatistic = function(stat, dostat)
   {
      if (!this.histo) return false;

      var print_name    = Math.floor(dostat%10);
      var print_entries = Math.floor(dostat/10)%10;
      var print_mean    = Math.floor(dostat/100)%10;
      var print_rms     = Math.floor(dostat/1000)%10;
      var print_under   = Math.floor(dostat/10000)%10;
      var print_over    = Math.floor(dostat/100000)%10;
      var print_integral= Math.floor(dostat/1000000)%10;
      var print_skew    = Math.floor(dostat/10000000)%10;
      var print_kurt    = Math.floor(dostat/100000000)%10;

      if (print_name > 0)
         stat.AddLine(this.histo['fName']);

      if (print_entries > 0)
         stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(this.stat_entries));


      var meanx = 0, meany = 0;
      if (this.stat_sum0 > 0) {
         meanx = this.stat_sumx1/this.stat_sum0;
         meany = this.stat_sumy1/this.stat_sum0;
      }

      if (print_mean > 0) {
         stat.AddLine("Mean x = " + JSROOT.gStyle.StatFormat(meanx));
         stat.AddLine("Mean y = " + JSROOT.gStyle.StatFormat(meany));
      }

      var rmsx = 0, rmsy = 0;
      if (this.stat_sum0 > 0) {
         rmsx = Math.sqrt(this.stat_sumx2/this.stat_sum0 - meanx*meanx);
         rmsy = Math.sqrt(this.stat_sumy2/this.stat_sum0 - meany*meany);
      }

      if (print_rms > 0) {
         stat.AddLine("RMS x = " + JSROOT.gStyle.StatFormat(rmsx));
         stat.AddLine("RMS y = " + JSROOT.gStyle.StatFormat(rmsy));
      }

      if (print_integral > 0) {
         stat.AddLine("Integral = " + JSROOT.gStyle.StatEntriesFormat(this.stat_matrix[4]));
      }

      if (print_skew > 0) {
         stat.AddLine("Skewness x = <undef>");
         stat.AddLine("Skewness y = <undef>");
      }

      if (print_kurt > 0)
         stat.AddLine("Kurt = <undef>");

      if ((print_under > 0) || (print_over > 0)) {
         var m = this.stat_matrix;

         stat.AddLine(""+ m[6].toFixed(0)+ " | " + m[7].toFixed(0) +  " | " + m[7].toFixed(0));
         stat.AddLine(""+ m[3].toFixed(0)+ " | " + m[4].toFixed(0) +  " | " + m[5].toFixed(0));
         stat.AddLine(""+ m[0].toFixed(0)+ " | " + m[1].toFixed(0) +  " | " + m[2].toFixed(0));
      }

      // adjust the size of the stats box wrt the number of lines
      var nlines = stat.pavetext['fLines'].arr.length;
      var stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont%10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         stat.pavetext['fY1NDC'] = 0.93 - stath;
         stat.pavetext['fY2NDC'] = 0.93;
      }

      return true;
   }


   JSROOT.TH2Painter.prototype.getValueColor = function(zc) {
      var wmin = this.minbin, wmax = this.maxbin;
      var wlmin = wmin, wlmax = wmax;
      var ndivz = this.histo['fContour'].length;
      if (ndivz < 16) ndivz = 16;
      var scale = ndivz / (wlmax - wlmin);
      if (this.options.Logz) {
         if (wmin <= 0 && wmax > 0) wmin = Math.min(1.0, 0.001 * wmax);
         wlmin = Math.log(wmin)/Math.log(10);
         wlmax = Math.log(wmax)/Math.log(10);
      }

      if (this.paletteColors.length == 0) {
         var saturation = 1,
             lightness = 0.5,
             maxHue = 280,
             minHue = 0,
             maxPretty = 50;
         for (var i=0 ; i<maxPretty ; i++) {
            var hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty))/360.0;
            var rgbval = JSROOT.Painter.HLStoRGB(hue, lightness, saturation);
            this.paletteColors.push(rgbval);
         }
      }
      if (this.options.Logz) zc = Math.log(zc)/Math.log(10);
      if (zc < wlmin) zc = wlmin;
      var ncolors = this.paletteColors.length;
      var color = Math.round(0.01 + (zc - wlmin) * scale);
      var theColor = Math.round((color + 0.99) * ncolors / ndivz) - 1;
      var icol = theColor % ncolors;
      if (icol < 0) icol = 0;

      return this.paletteColors[icol];
   }

   JSROOT.TH2Painter.prototype.CreateDrawBins = function(w,h,coordinates_kind,tipkind)
   {
      var i1 = this.GetSelectIndex("x","left",0);
      var i2 = this.GetSelectIndex("x","right",0);
      var j1 = this.GetSelectIndex("y","left",0);
      var j2 = this.GetSelectIndex("y","right",0);

      var xfactor = 1, yfactor = 1;
      if (coordinates_kind == 1) {
         xfactor = 0.5 * w / (i2-i1) / (this.maxbin - this.minbin);
         yfactor = 0.5 * h / (j2-j1) / (this.maxbin - this.minbin);
      }

      var x1, y1, x2, y2, grx1, gry1, grx2, gry2, fillcol, shrx, shry, binz, point;

      var local_bins = new Array;

      x2 = this.xmin + i1*this.binwidthx;
      grx2 = -11111;
      for (var i=i1;i<i2;i++) {
         x1 = x2; x2 += this.binwidthx;

         if (this.options.Logx && (x1<=0)) continue;

         grx1 = grx2;
         if (grx1 < 0) grx1 = this.x(x1);
         grx2 = this.x(x2);

         y2 = this.ymin + j1*this.binwidthy;
         gry2 = -1111;
         for (var j=j1;j<j2;j++) {
            y1 = y2; y2 += this.binwidthy;
            if (this.options.Logy && (y1<=0)) continue;
            gry1 = gry2;
            if (gry1 < 0) gry1 = this.y(y1);
            gry2 = this.y(y2);
            binz = this.histo.getBinContent(i+1, j+1);
            if (binz <= this.minbin) continue;

            switch (coordinates_kind) {
               case 0:
                  point = {
                     x: grx1,
                     y: gry2,
                     width: grx2 - grx1 + 1,
                     height: gry1 - gry2 + 1,
                     stroke: "none",
                     fill: this.getValueColor(binz)
                  }
                  point['tipcolor'] = (point['fill'] == "black") ? "grey" : "black";
                  break;

               case 1:
                  shrx = xfactor * (this.maxbin - binz);
                  shry = yfactor * (this.maxbin - binz);
                  point = {
                     x: grx1 + shrx,
                     y: gry2 + shry,
                     width: grx2 - grx1 - 2*shrx,
                     height: gry1 - gry2 - 2*shry,
                     stroke: this.linecolor,
                     fill: this.fillcolor
                  }
                  point['tipcolor'] = (point['fill'] == "black") ? "grey" : "black";
                  break;

               case 2:
                  point = {
                     x: (x1+x2)/2,
                     y: (y1+y2)/2,
                     z: binz
                  }
                  break;
            }

            if (tipkind == 1)
               point['tip'] = "x = [" + this.AxisAsText("x", x1) + ", " + this.AxisAsText("x", x2) + "]\n"+
                              "y = [" + this.AxisAsText("y", y1) + ", " + this.AxisAsText("y", y2) + "]\n"+
                              "entries = " + binz;
            else if (tipkind == 2)
               point['tip'] = "x = " + this.AxisAsText("x", x1) + "\n" +
                              "y = " + this.AxisAsText("y", y1) + "\n" +
                              "entries = " + binz;

            local_bins.push(point);
         }
      }

      return local_bins;
   }

   JSROOT.TH2Painter.prototype.DrawBins = function()
   {
      this.RemoveDraw();

      this.draw_g = this.svg_frame.append("svg:g");

      var w = Number(this.frame.attr("width")), h = Number(this.frame.attr("height"));

//    this.options.Scat =1;
//    this.histo['fMarkerStyle'] = 2;

      var draw_markers = (this.options.Scat > 0 && this.histo['fMarkerStyle'] > 1);
      var normal_coordinates = (this.options.Color > 0) || draw_markers;


      var tipkind = 0;
      if (JSROOT.gStyle.Tooltip) tipkind = draw_markers ? 2 : 1;

      var local_bins = this.CreateDrawBins(w,h,normal_coordinates ? 0 : 1, tipkind);

      if (draw_markers) {

         // Add markers
         // TODO: fill is not used
         var filled = false;
         if ((this.histo['fMarkerStyle'] == 8) ||
             (this.histo['fMarkerStyle'] > 19 && this.histo['fMarkerStyle'] < 24) ||
             (this.histo['fMarkerStyle'] == 29))
            filled = true;

         var info_marker = JSROOT.Painter.getRootMarker(this.histo['fMarkerStyle']);

         var shape = info_marker['shape'];
         var filled = info_marker['toFill'];
         var toRotate = info_marker['toRotate'];
         var markerSize = this.histo['fMarkerSize'];
         var markerScale = (shape == 0) ? 32 : 64;
         if (this.histo['fMarkerStyle'] == 1) markerScale = 1;

         var marker = null;

         switch (shape) {
            case 6:
               marker = "M " + (-4 * markerSize) + " " + (-1 * markerSize)
                         + " L " + 4 * markerSize + " " + (-1 * markerSize)
                         + " L " + (-2.4 * markerSize) + " " + 4 * markerSize
                         + " L 0 " + (-4 * markerSize) + " L " + 2.8 * markerSize
                         + " " + 4 * markerSize + " z";
               break;
            case 7:
               marker = "M " + (- 4 * markerSize) + " " + (-4 * markerSize)
                        + " L " + 4 * markerSize + " " + 4 * markerSize + " M 0 "
                        + (-4 * markerSize) + " 0 " + 4 * markerSize + " M "
                        + 4 * markerSize + " " + (-4 * markerSize) + " L "
                        + (-4 * markerSize) + " " + 4 * markerSize + " M "
                        + (-4 * markerSize) + " 0 L " + 4 * markerSize + " 0";
               break;
            default:
               marker = d3.svg.symbol()
                       .type(d3.svg.symbolTypes[shape])
                       .size(markerSize * markerScale);
               break;
         }
         var markers = this.draw_g.selectAll(".marker")
            .data(local_bins)
            .enter()
            .append("svg:path")
            .attr("class", "marker")
            .attr("transform", function(d) {
               return "translate(" + d.x + "," + d.y + ")"
            })
            .style("fill", JSROOT.Painter.root_colors[this.histo['fMarkerColor']])
            .style("stroke", JSROOT.Painter.root_colors[this.histo['fMarkerColor']])
            .attr("d", marker);

         if (JSROOT.gStyle.Tooltip)
            markers.append("svg:title").text(function(d) { return d.tip; });
      }
      else {
         var drawn_bins = this.draw_g.selectAll(".bins")
               .data(local_bins)
               .enter()
               .append("svg:rect")
               .attr("class", "bins")
               .attr("x", function(d) { return d.x; })
               .attr("y", function(d) { return d.y; })
               .attr("width", function(d) { return d.width; })
               .attr("height", function(d) { return d.height; })
               .style("stroke", function(d) { return d.stroke; })
               .style("fill", function(d) { this['f0'] = d.fill; this['f1'] = d.tipcolor; return d.fill; });

         if (JSROOT.gStyle.Tooltip)
            drawn_bins
              .on('mouseover', function() { d3.select(this).transition().duration(100).style("fill", this['f1']); })
              .on('mouseout', function() { d3.select(this).transition().duration(100).style("fill", this['f0']); })
              .append("svg:title") .text(function(d) { return d.tip; });
      }

      delete local_bins;
      local_bins = null;
   }

   JSROOT.TH2Painter.prototype.Draw3D = function()
   {
      var vis = this.vis;
      var frame = this.frame;

      // this.ClearFrame();
      
      var w = Number(vis.attr("width")), h = Number(vis.attr("height")), size = 100;

      var xmin = this.xmin, xmax = this.xmax;
      if (this.zoom_xmin != this.zoom_xmax) { xmin = this.zoom_xmin; xmax = this.zoom_xmax; }
      var ymin = this.ymin, ymax = this.ymax;
      if (this.zoom_ymin != this.zoom_ymax) { ymin = this.zoom_ymin; ymax = this.zoom_ymax; }
      
      var tx, utx, ty, uty, tz, utz;

      if (this.options.Logx) {
         tx = d3.scale.log().domain([xmin, xmax]).range([-size, size]);
         utx = d3.scale.log().domain([-size, size]).range([xmin, xmax]);
      } else {
         tx = d3.scale.linear().domain([xmin, xmax]).range([-size, size]);
         utx = d3.scale.linear().domain([-size, size]).range([xmin, xmax]);
      }
      if (this.options.Logy) {
         ty = d3.scale.log().domain([ymin, ymax]).range([-size, size]);
         uty = d3.scale.log().domain([size, -size]).range([ymin, ymax]);
      } else {
         ty = d3.scale.linear().domain([ymin, ymax]).range([-size, size]);
         uty = d3.scale.linear().domain([size, -size]).range([ymin, ymax]);
      }
      if (this.options.Logz) {
         tz = d3.scale.log().domain([this.minbin, Math.ceil( this.maxbin/10 )*10]).range([0, size*2]);
         utz = d3.scale.log().domain([0, size*2]).range([this.minbin, Math.ceil( this.maxbin/10 )*10]);
      } else {
         tz = d3.scale.linear().domain([this.minbin, Math.ceil( this.maxbin/10 )*10]).range([0, size*2]);
         utz = d3.scale.linear().domain([0, size*2]).range([this.minbin, Math.ceil( this.maxbin/10 )*10]);
      }

      var constx = (size*2 / this.nbinsx) / this.maxbin;
      var consty = (size*2 / this.nbinsy) / this.maxbin;

      var colorFlag = ( this.options.Color > 0);
      var fcolor = d3.rgb(JSROOT.Painter.root_colors[this.histo['fFillColor']]);

      var local_bins = this.CreateDrawBins(100, 100, 2, (JSROOT.gStyle.Tooltip ? 1 : 0));

      var painter = this;

    JSROOT.AssertPrerequisites('3d', function() {
      
      // three.js 3D drawing
      var scene = new THREE.Scene();

      var toplevel = new THREE.Object3D();
      toplevel.rotation.x = 30 * Math.PI / 180;
      toplevel.rotation.y = 30 * Math.PI / 180;
      scene.add( toplevel );

      var wireMaterial = new THREE.MeshBasicMaterial( {
         color: 0x000000,
         wireframe: true,
         wireframeLinewidth: 0.5,
         side: THREE.DoubleSide } );

      // create a new mesh with cube geometry
      var cube = new THREE.Mesh( new THREE.CubeGeometry( size*2, size*2, size*2 ), wireMaterial);
      cube.position.y = size;

      // add the cube to the scene
      toplevel.add( cube );

      var textMaterial = new THREE.MeshBasicMaterial( { color: 0x000000 } );

      // add the calibration vectors and texts
      var geometry = new THREE.Geometry();
      var imax, istep, len = 3, plen, sin45 = Math.sin(45);
      var text3d, text;
      var xmajors = tx.ticks(8);
      var xminors = tx.ticks(50);
      for (var i=-size, j=0, k=0; i<size; ++i ) {
         var is_major = ( utx( i ) <= xmajors[j] && utx( i+1 ) > xmajors[j] ) ? true : false;
         var is_minor = ( utx( i ) <= xminors[k] && utx( i+1 ) > xminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( xmajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( i-centerOffset, -13, size+plen );
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( i+centerOffset, -13, -size-plen );
            text.rotation.y = Math.PI;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( i, 0, size ) );
            geometry.vertices.push( new THREE.Vector3( i, -plen, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( i, 0, -size ) );
            geometry.vertices.push( new THREE.Vector3( i, -plen, -size-plen ) );
         }
      }
      var ymajors = ty.ticks(8);
      var yminors = ty.ticks(50);
      for (var i=size, j=0, k=0; i>-size; --i ) {
         var is_major = ( uty( i ) <= ymajors[j] && uty( i-1 ) > ymajors[j] ) ? true : false;
         var is_minor = ( uty( i ) <= yminors[k] && uty( i-1 ) > yminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( ymajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+plen, -13, i+centerOffset );
            text.rotation.y = Math.PI/2;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-plen, -13, i-centerOffset );
            text.rotation.y = -Math.PI/2;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( size, 0, i ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, -plen, i ) );
            geometry.vertices.push( new THREE.Vector3( -size, 0, i ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, -plen, i ) );
         }
      }
      var zmajors = tz.ticks(8);
      var zminors = tz.ticks(50);
      for (var i=0, j=0, k=0; i<(size*2); ++i ) {
         var is_major = ( utz( i ) <= zmajors[j] && utz( i+1 ) > zmajors[j] ) ? true : false;
         var is_minor = ( utz( i ) <= zminors[k] && utz( i+1 ) > zminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( zmajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var offset = 0.8 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+offset+5, i-2.5, size+offset+5 );
            text.rotation.y = Math.PI*3/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+offset+5, i-2.5, -size-offset-5 );
            text.rotation.y = -Math.PI*3/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-offset-5, i-2.5, size+offset+5 );
            text.rotation.y = Math.PI/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-offset-5, i-2.5, -size-offset-5 );
            text.rotation.y = -Math.PI/4;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( size, i, size ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, i, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( size, i, -size ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, i, -size-plen ) );
            geometry.vertices.push( new THREE.Vector3( -size, i, size ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, i, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( -size, i, -size ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, i, -size-plen ) );
         }
      }

      // add the calibration lines
      var lineMaterial = new THREE.LineBasicMaterial( { color: 0x000000 } );
      var line = new THREE.Line( geometry, lineMaterial );
      line.type = THREE.LinePieces;
      toplevel.add( line );

      // create the bin cubes

      var fillcolor = new THREE.Color( 0xDDDDDD );
      fillcolor.setRGB(fcolor.r/255, fcolor.g/255, fcolor.b/255);
      var bin, wei, hh;

      for (var i = 0; i < local_bins.length; ++i ) {
         hh = local_bins[i];
         wei = tz(hh.z);

         bin = THREE.SceneUtils.createMultiMaterialObject(
            new THREE.CubeGeometry( 2*size/painter.nbinsx, wei, 2*size/painter.nbinsy ),
            [ new THREE.MeshLambertMaterial( { color: fillcolor.getHex(), shading: THREE.NoShading } ),
              wireMaterial ] );
         bin.position.x = tx(hh.x);
         bin.position.y = wei/2;
         bin.position.z = -(ty(hh.y));

         if (JSROOT.gStyle.Tooltip) bin.name = hh.tip;
         toplevel.add( bin );
      }

      delete local_bins;
      local_bins = null;

      // create a point light
      var pointLight = new THREE.PointLight( 0xcfcfcf );
      pointLight.position.set(0, 50, 250);
      scene.add(pointLight);

      //var directionalLight = new THREE.DirectionalLight( 0x7f7f7f );
      //directionalLight.position.set( 0, -70, 100 ).normalize();
      //scene.add( directionalLight );

      var camera = new THREE.PerspectiveCamera( 45, w / h, 1, 1000 );
      camera.position.set( 0, size/2, 500 );
      camera.lookat = cube;

      /**
       * @author alteredq / http://alteredqualia.com/
       * @author mr.doob / http://mrdoob.com/
       */
      var Detector = {
            canvas: !! window.CanvasRenderingContext2D,
            webgl: ( function () { try { return !! window.WebGLRenderingContext && !! document.createElement( 'canvas' ).getContext( 'experimental-webgl' ); } catch( e ) { return false; } } )(),
            workers: !! window.Worker, fileapi: window.File && window.FileReader && window.FileList && window.Blob
      };


      var renderer = Detector.webgl ? new THREE.WebGLRenderer( { antialias: true } ) :
                     new THREE.CanvasRenderer( { antialias: true } );
      renderer.setSize( w, h );
      $( vis[0][0] ).hide().parent().append( renderer.domElement );
      renderer.render( scene, camera );

      JSROOT.Painter.add3DInteraction(renderer, scene, camera, toplevel, vis);
   })}
   
   JSROOT.Painter.drawHistogram2D = function(vis, histo, opt)
   {

      // create painter and add it to canvas
      var painter = new JSROOT.TH2Painter(histo);

      var hadframe = (vis['ROOT:frame'] != null);

      painter.SetFrame(vis, opt);

      painter.ScanContent();

      // check if we need to create palette
      if ((painter.FindPalette() == null) && !hadframe && (painter.options.Zscale>0)) {
         // create pallette
         var shrink = painter.CreatePalette(0.04);
         JSROOT.Painter.shrinkFrame(vis, shrink);
      }

      // check if we need to create statbox
      if (JSROOT.gStyle.AutoStat && !hadframe) painter.CreateStat();

      painter.CreateXY();

      painter.CountStat();

      painter.DrawAxes();

      painter.DrawGrids();

      painter.DrawBins();

      painter.DrawTitle();

      painter.DrawFunctions();

      painter.AddInteractive();

      return painter;
   }

   JSROOT.Painter.drawHistogram3D = function(vis, histo, dopt)
   {
     JSROOT.AssertPrerequisites('3d', function() {
            
      var logx = false, logy = false, logz = false,
          gridx = false, gridy = false, gridz = false;

      var opt = histo['fOption'].toLowerCase();
      // if (opt=="") opt = "colz";

      if (vis['ROOT:pad'] && typeof(vis['ROOT:pad']) != 'undefined') {
         logx = vis['ROOT:pad']['fLogx'];
         logy = vis['ROOT:pad']['fLogy'];
         logz = vis['ROOT:pad']['fLogz'];
         gridx = vis['ROOT:pad']['fGridx'];
         gridy = vis['ROOT:pad']['fGridy'];
         gridz = vis['ROOT:pad']['fGridz'];
      }
      
      var fillcolor = JSROOT.Painter.root_colors[histo['fFillColor']];
      var linecolor = JSROOT.Painter.root_colors[histo['fLineColor']];
      if (histo['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (histo['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      var nbinsx = histo['fXaxis']['fNbins'];
      var nbinsy = histo['fYaxis']['fNbins'];
      var nbinsz = histo['fZaxis']['fNbins'];
      var scalex = (histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin']) /
                    histo['fXaxis']['fNbins'];
      var scaley = (histo['fYaxis']['fXmax'] - histo['fYaxis']['fXmin']) /
                    histo['fYaxis']['fNbins'];
      var scalez = (histo['fZaxis']['fXmax'] - histo['fZaxis']['fXmin']) /
                    histo['fZaxis']['fNbins'];
      var maxbin = -1e32, minbin = 1e32;
      maxbin = d3.max(histo['fArray']);
      minbin = d3.min(histo['fArray']);
      var bins = new Array();
      for (var i=0; i<=nbinsx+2; ++i) {
         for (var j=0; j<nbinsy+2; ++j) {
            for (var k=0; k<nbinsz+2; ++k) {
               var bin_content = histo.getBinContent(i, j, k);
               if (bin_content > minbin) {
                  var point = {
                     x:histo['fXaxis']['fXmin'] + (i*scalex),
                     y:histo['fYaxis']['fXmin'] + (j*scaley),
                     z:histo['fZaxis']['fXmin'] + (k*scalez),
                     n:bin_content
                  };
                  bins.push(point);
               }
            }
         }
      }
      var w = Number(vis.attr("width")), h = Number(vis.attr("height")), size = 100;
      if (logx) {
         var tx = d3.scale.log().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([-size, size]);
         var utx = d3.scale.log().domain([-size, size]).range([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]);
      } else {
         var tx = d3.scale.linear().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([-size, size]);
         var utx = d3.scale.linear().domain([-size, size]).range([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]);
      }
      if (logy) {
         var ty = d3.scale.log().domain([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]).range([-size, size]);
         var uty = d3.scale.log().domain([size, -size]).range([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]);
      } else {
         var ty = d3.scale.linear().domain([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]).range([-size, size]);
         var uty = d3.scale.linear().domain([size, -size]).range([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]);
      }
      if (logz) {
         var tz = d3.scale.log().domain([histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax']]).range([-size, size]);
         var utz = d3.scale.log().domain([-size, size]).range([histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax']]);
      } else {
         var tz = d3.scale.linear().domain([histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax']]).range([-size, size]);
         var utz = d3.scale.linear().domain([-size, size]).range([histo['fZaxis']['fXmin'], histo['fZaxis']['fXmax']]);
      }

      // three.js 3D drawing
      var scene = new THREE.Scene();

      var toplevel = new THREE.Object3D();
      toplevel.rotation.x = 30 * Math.PI / 180;
      toplevel.rotation.y = 30 * Math.PI / 180;
      scene.add( toplevel );

      var wireMaterial = new THREE.MeshBasicMaterial( {
         color: 0x000000,
         wireframe: true,
         wireframeLinewidth: 0.5,
         side: THREE.DoubleSide } );

      // create a new mesh with cube geometry
      var cube = new THREE.Mesh( new THREE.CubeGeometry( size*2, size*2, size*2 ), wireMaterial);

      // add the cube to the scene
      toplevel.add( cube );

      var textMaterial = new THREE.MeshBasicMaterial( { color: 0x000000 } );

      // add the calibration vectors and texts
      var geometry = new THREE.Geometry();
      var imax, istep, len = 3, plen, sin45 = Math.sin(45);
      var text3d, text;
      var xmajors = tx.ticks(5);
      var xminors = tx.ticks(25);
      for (var i=-size, j=0, k=0; i<=size; ++i ) {
         var is_major = ( utx( i ) <= xmajors[j] && utx( i+1 ) > xmajors[j] ) ? true : false;
         var is_minor = ( utx( i ) <= xminors[k] && utx( i+1 ) > xminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( xmajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( i-centerOffset, -size-13, size+plen );
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( i+centerOffset, -size-13, -size-plen );
            text.rotation.y = Math.PI;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( i, -size, size ) );
            geometry.vertices.push( new THREE.Vector3( i, -size-plen, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( i, -size, -size ) );
            geometry.vertices.push( new THREE.Vector3( i, -size-plen, -size-plen ) );
         }
      }
      var ymajors = ty.ticks(5);
      var yminors = ty.ticks(25);
      for (var i=size, j=0, k=0; i>-size; --i ) {
         var is_major = ( uty( i ) <= ymajors[j] && uty( i-1 ) > ymajors[j] ) ? true : false;
         var is_minor = ( uty( i ) <= yminors[k] && uty( i-1 ) > yminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( ymajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var centerOffset = 0.5 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+plen, -size-13, i+centerOffset );
            text.rotation.y = Math.PI/2;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-plen, -size-13, i-centerOffset );
            text.rotation.y = -Math.PI/2;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( size, -size, i ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, -size-plen, i ) );
            geometry.vertices.push( new THREE.Vector3( -size, -size, i ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, -size-plen, i ) );
         }
      }
      var zmajors = tz.ticks(5);
      var zminors = tz.ticks(25);
      for (var i=-size, j=0, k=0; i<=size; ++i ) {
         var is_major = ( utz( i ) <= zmajors[j] && utz( i+1 ) > zmajors[j] ) ? true : false;
         var is_minor = ( utz( i ) <= zminors[k] && utz( i+1 ) > zminors[k] ) ? true : false;
         plen = ( is_major ? len + 2 : len) * sin45;
         if ( is_major ) {
            text3d = new THREE.TextGeometry( zmajors[j], {
               size: 7,
               height: 0,
               curveSegments: 10
            });
            ++j;

            text3d.computeBoundingBox();
            var offset = 0.6 * ( text3d.boundingBox.max.x - text3d.boundingBox.min.x );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+offset+7, i-2.5, size+offset+7 );
            text.rotation.y = Math.PI*3/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( size+offset+7, i-2.5, -size-offset-7 );
            text.rotation.y = -Math.PI*3/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-offset-7, i-2.5, size+offset+7 );
            text.rotation.y = Math.PI/4;
            toplevel.add( text );

            text = new THREE.Mesh( text3d, textMaterial );
            text.position.set( -size-offset-7, i-2.5, -size-offset-7 );
            text.rotation.y = -Math.PI/4;
            toplevel.add( text );
         }
         if ( is_major || is_minor ) {
            ++k;
            geometry.vertices.push( new THREE.Vector3( size, i, size ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, i, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( size, i, -size ) );
            geometry.vertices.push( new THREE.Vector3( size+plen, i, -size-plen ) );
            geometry.vertices.push( new THREE.Vector3( -size, i, size ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, i, size+plen ) );
            geometry.vertices.push( new THREE.Vector3( -size, i, -size ) );
            geometry.vertices.push( new THREE.Vector3( -size-plen, i, -size-plen ) );
         }
      }

      // add the calibration lines
      var lineMaterial = new THREE.LineBasicMaterial( { color: 0x000000 } );
      var line = new THREE.Line( geometry, lineMaterial );
      line.type = THREE.LinePieces;
      toplevel.add( line );

      // create the bin cubes
      var constx = (size*2 / histo['fXaxis']['fNbins']) / maxbin;
      var consty = (size*2 / histo['fYaxis']['fNbins']) / maxbin;
      var constz = (size*2 / histo['fZaxis']['fNbins']) / maxbin;

      var optFlag = ( opt.indexOf('colz') != -1 || opt.indexOf('col') != -1 );
      var fcolor = d3.rgb(JSROOT.Painter.root_colors[histo['fFillColor']]);
      var fillcolor = new THREE.Color( 0xDDDDDD );
      fillcolor.setRGB(fcolor.r/255, fcolor.g/255, fcolor.b/255);
      var bin, wei;
      for (var i = 0; i < bins.length; ++i ) {
         wei = ( optFlag ? maxbin : bins[i].n );
         if (opt.indexOf('box1') != -1) {
            bin = new THREE.Mesh( new THREE.SphereGeometry( 0.5 * wei * constx /*, 16, 16 */ ),
                      new THREE.MeshPhongMaterial( { color: fillcolor.getHex(),
                              specular: 0xbfbfbf /*, shading: THREE.NoShading */ } ) );
         }
         else {
            bin = THREE.SceneUtils.createMultiMaterialObject(
               new THREE.CubeGeometry( wei * constx, wei * constz, wei * consty ),
               [ new THREE.MeshLambertMaterial( { color: fillcolor.getHex(), shading: THREE.NoShading } ),
                 wireMaterial ] );
         }
         bin.position.x = tx( bins[i].x - (scalex/2));
         bin.position.y = tz( bins[i].z - (scalez/2));
         bin.position.z = -(ty( bins[i].y - (scaley/2)));
         bin.name = "x: [" + bins[i].x.toPrecision(4) + ", " + (bins[i].x + scalex).toPrecision(4) + "]<br/>" +
                    "y: [" + bins[i].y.toPrecision(4) + ", " + (bins[i].y + scaley).toPrecision(4) + "]<br/>" +
                    "z: [" + bins[i].z.toPrecision(4) + ", " + (bins[i].z + scalez).toPrecision(4) + "]<br/>" +
                    "entries: " + bins[i].n.toFixed();
         toplevel.add( bin );
      }
      // create a point light
      var pointLight = new THREE.PointLight( 0xcfcfcf );
      pointLight.position.set(0, 50, 250);
      scene.add(pointLight);

      //var directionalLight = new THREE.DirectionalLight( 0x7f7f7f );
      //directionalLight.position.set( 0, -70, 100 ).normalize();
      //scene.add( directionalLight );

      var camera = new THREE.PerspectiveCamera( 45, w / h, 1, 1000 );
      camera.position.set( 0, 0, 500 );
      camera.lookat = cube;

      /**
       * @author alteredq / http://alteredqualia.com/
       * @author mr.doob / http://mrdoob.com/
       */
      var Detector = {
            canvas: !! window.CanvasRenderingContext2D,
            webgl: ( function () { try { return !! window.WebGLRenderingContext && !! document.createElement( 'canvas' ).getContext( 'experimental-webgl' ); } catch( e ) { return false; } } )(),
            workers: !! window.Worker, fileapi: window.File && window.FileReader && window.FileList && window.Blob
      };

      var renderer = Detector.webgl ? new THREE.WebGLRenderer( { antialias: true } ) :
                     new THREE.CanvasRenderer( { antialias: true } );
      renderer.setSize( w, h );
      $( vis[0][0] ).hide().parent().append( renderer.domElement );
      renderer.render( scene, camera );

      JSROOT.Painter.add3DInteraction(renderer, scene, camera, toplevel, vis);
   })}

   JSROOT.Painter.drawHStack = function(vis, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      //    -the first histogram is paint
      //    -then the sum of the first and second, etc

      if (!'fHists' in stack) return;
      if (stack['fHists'].arr.length == 0) return;
      var pad = vis['ROOT:pad'];
      var histos = stack['fHists'];
      var nhists = histos.arr.length;
      if (nhists == 0) return;
      if (opt==null) opt = ""; else opt = opt.toLowerCase();
      var lsame = false;
      if (opt.indexOf("same") != -1) {
         lsame = true;
         opt.replace("same", "");
      }
      // compute the min/max of each axis
      var i, h;
      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
      for (var i=0; i<nhists; ++i) {
         h = histos.arr[i];
         if (i==0 || h['fXaxis']['fXmin'] < xmin) xmin = h['fXaxis']['fXmin'];
         if (i==0 || h['fXaxis']['fXmax'] > xmax) xmax = h['fXaxis']['fXmax'];
         if (i==0 || h['fYaxis']['fXmin'] < ymin) ymin = h['fYaxis']['fXmin'];
         if (i==0 || h['fYaxis']['fXmax'] > ymax) ymax = h['fYaxis']['fXmax'];
      }
      var nostack = opt.indexOf("nostack") == -1 ? false : true;
      if (!nostack) stack.buildStack();

      var themin, themax;
      if (stack['fMaximum'] == -1111) themax = stack.getMaximum(opt);
      else themax = stack['fMaximum'];
      if (stack['fMinimum'] == -1111) {
         themin = stack.getMinimum(opt);
         if (pad && pad['fLogy']) {
            if (themin > 0) themin *= .9;
            else themin = themax * 1.e-3;
         }
         else if (themin > 0)
            themin = 0;
      }
      else themin = stack['fMinimum'];
      if (!('fHistogram' in stack)) {
         h = stack['fHists'].arr[0];
         stack['fHistogram'] = JSROOT.CreateTH1();
         stack['fHistogram']['fName'] = "unnamed";
         stack['fHistogram']['fXaxis'] = JSROOT.clone(h['fXaxis']);
         stack['fHistogram']['fYaxis'] = JSROOT.clone(h['fYaxis']);
         stack['fHistogram']['fXaxis']['fXmin'] = xmin;
         stack['fHistogram']['fXaxis']['fXmax'] = xmax;
         stack['fHistogram']['fYaxis']['fXmin'] = ymin;
         stack['fHistogram']['fYaxis']['fXmax'] = ymax;
      }
      stack['fHistogram']['fTitle'] = stack['fTitle'];
      //var histo = JSROOT.clone(stack['fHistogram']);
      var histo = stack['fHistogram'];
      if (!histo.TestBit(JSROOT.TH1StatusBits.kIsZoomed)) {
         if (nostack && stack['fMaximum'] != -1111) histo['fMaximum'] = stack['fMaximum'];
         else {
            if (pad && pad['fLogy']) histo['fMaximum'] = themax*(1+0.2*JSROOT.Math.log10(themax/themin));
            else histo['fMaximum'] = 1.05 * themax;
         }
         if (nostack && stack['fMinimum'] != -1111) histo['fMinimum'] = stack['fMinimum'];
         else {
            if (pad && pad['fLogy']) histo['fMinimum'] = themin/(1+0.5*JSROOT.Math.log10(themax/themin));
            else histo['fMinimum'] = themin;
         }
      }
      if (!lsame) {

         var hopt = histo['fOption'];
         if ((opt!="") && (hopt.indexOf(opt) == -1)) hopt += opt;

         if (histo['_typename'].match(/\bTH1/))
            JSROOT.Painter.drawHistogram1D(vis, histo, hopt);
         else if (histo['_typename'].match(/\bTH2/))
            JSROOT.Painter.drawHistogram2D(vis, histo, hopt);
      }
      for (var i=0; i<nhists; ++i) {
         if (nostack)
            h = histos.arr[i];
         else
            h = stack['fStack'].arr[nhists-i-1];

         var hopt = h['fOption'];
         if ((opt!="") && (hopt.indexOf(opt) == -1)) hopt += opt;
         hopt += "same";

         if (h['_typename'].match(/\bTH1/))
            JSROOT.Painter.drawHistogram1D(vis, h, hopt);
      }
   };

   JSROOT.Painter.drawLatex = function(vis, string, x, y, attr) {
      var w = Number(vis.attr("width")), h = Number(vis.attr("height"));
      while (string.indexOf('#') != -1)
         string = string.replace('#', '\\');
      string = string.replace(' ', '\\: ');

      var parse = new jsMath.Parser(string, null, null, null);
      parse.Parse();
      if (parse.error) return false;

      // method using jsMath do display formulae and LateX
      // unfortunately it works only on FireFox (Chrome displays it,
      // but at wrong coordinates, and IE doesn't support foreignObject
      // in SVG...)
      string = '\\displaystyle \\rm ' + string;
      var fo = vis.append("foreignObject")
         .attr("x", x)
         .attr("y", y)
         .attr("width", w - x)
         .attr("height", h - y);
      var math = fo.append("xhtml:div")
         .style("display", "inline")
         .style("color", attr['font-color'])
         .style('font-size', (attr['font-size']*0.98)+'px')
         .attr("class", "math")
         .html(string);
      jsMath.ProcessElement(math[0][0]);
      return true;
   };

   JSROOT.Painter.drawLegend = function(vis, pave, opt) {
      var pad = vis['ROOT:pad'];

      var x=0, y=0, w=0, h=0;
      if (pave['fInit'] == 0) {
          x = pave['fX1'] * Number(vis.attr("width"));
          y = Number(vis.attr("height")) - pave['fY1'] * Number(vis.attr("height"));
          w = (pave['fX2'] - pave['fX1']) * Number(vis.attr("width"));
          h = (pave['fY2'] - pave['fY1']) * Number(vis.attr("height"));
      }
      else {
          x = pave['fX1NDC'] * Number(vis.attr("width"));
          y = Number(vis.attr("height")) - pave['fY1NDC'] * Number(vis.attr("height"));
          w = (pave['fX2NDC'] - pave['fX1NDC']) * Number(vis.attr("width"));
          h = (pave['fY2NDC'] - pave['fY1NDC']) * Number(vis.attr("height"));
      }
      y -= h;
      var fillcolor = JSROOT.Painter.root_colors[pave['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pave['fLineColor']];
      var lwidth = pave['fBorderSize'] ? pave['fBorderSize'] : 0;
      if (pave['fFillStyle'] > 4000 && pave['fFillStyle'] < 4100)
         fillcolor = 'none';

      var p = vis.append("svg:g")
         .attr("width", w)
         .attr("height", h)
         .attr("transform", "translate(" + x + "," + y + ")");

      p.append("svg:rect")
         .attr("class", p)
         .attr("x", 0)
         .attr("y", 0)
         .attr("width", w)
         .attr("height", h)
         .attr("fill", fillcolor)
         .style("stroke-width", lwidth ? 1 : 0)
         .style("stroke", lcolor);

      var tcolor = JSROOT.Painter.root_colors[pave['fTextColor']];
      var tpos_x = pave['fMargin'] * w;
      var nlines = pave.fPrimitives.arr.length;
      var font_size = Math.round(h / (nlines * 1.5));
      //var font_size = Math.round(pave['fTextSize'] * vis.height());
      var fontDetails = JSROOT.Painter.getFontDetails(pave['fTextFont']);

      var max_len = 0, mul = 1.4;
      for (var j=0; j<nlines; ++j) {
         line = JSROOT.Painter.translateLaTeX(pave.fPrimitives.arr[j]['fLabel']);
         lw = tpos_x + JSROOT.Painter.stringWidth(vis, line, font_size, fontDetails);
         if (lw > max_len) max_len = lw;
      }
      if (max_len > w) {
         font_size *= 0.85 * (w / max_len);
         mul *= 0.95 * (max_len / w);
      }
      var x1 = pave['fX1NDC'];
      var x2 = pave['fX2NDC'];
      var y1 = pave['fY1NDC'];
      var y2 = pave['fY2NDC'];
      var margin = pave['fMargin']*( x2-x1 )/pave['fNColumns'];
      var yspace = (y2-y1)/nlines;
      var ytext = y2 + 0.5*yspace;  // y-location of 0th entry
      var boxw = margin*0.35;

      for (var i=0; i<nlines; ++i) {
         var leg = pave.fPrimitives.arr[i];
         var lopt = leg['fOption'].toLowerCase();

         //pave.fPrimitives.opt[i];

         //if (((lopt==null) || (lopt=="")) && ('fOption' in leg)) lopt = leg['fOption'];
         //lopt = lopt.toLowerCase();

         var string = leg['fLabel'];

         var pos_y = ((i+1) * (font_size * mul)) - (font_size/3);
         var tpos_y = (i+1) * (font_size * mul);
         if (nlines == 1) {
            var pos_y = (h * 0.75) - (font_size/3);
            var tpos_y = h * 0.75;
         }

         var line_color = JSROOT.Painter.root_colors[leg['fLineColor']];
         var line_width = leg['fLineWidth'];
         var line_style = JSROOT.Painter.root_line_styles[leg['fLineStyle']];

         var fill_color = leg['fFillColor'];
         var fill_style = leg['fFillStyle'];

         var marker_color = JSROOT.Painter.root_colors[leg['fMarkerColor']];
         var marker_size = leg['fMarkerSize'];
         var marker_style = leg['fMarkerStyle'];

         var mo = leg['fObject'];

         if ((mo!=null) && (typeof mo == 'object')) {
            if ('fLineColor' in mo) {
               line_color = JSROOT.Painter.root_colors[mo['fLineColor']];
               line_width = mo['fLineWidth'];
               line_style = JSROOT.Painter.root_line_styles[mo['fLineStyle']];
            }
            if ('fFillColor' in mo) {
               fill_color = mo['fFillColor'];
               fill_style = mo['fFillStyle'];
            }
            if ('fMarkerColor' in mo) {
               marker_color = JSROOT.Painter.root_colors[mo['fMarkerColor']];
               marker_size = mo['fMarkerSize'];
               marker_style = mo['fMarkerStyle'];
            }
         }

         p.append("text")
            .attr("class", "text")
            .attr("text-anchor", "start")
            .attr("x", tpos_x)
            .attr("y", tpos_y)
            .attr("font-weight", fontDetails['weight'])
            .attr("font-style", fontDetails['style'])
            .attr("font-family", fontDetails['name'])
            .attr("font-size", font_size)
            .attr("fill", tcolor)
            .text(string);

         // Draw fill pattern (in a box)
         if (lopt.indexOf('f') != -1) {
            // box total height is yspace*0.7
            // define x,y as the center of the symbol for this entry
            var xsym = margin/2;
            var ysym = ytext;
            var xf = new Array(4), yf = new Array(4);
            xf[0] = xsym - boxw;
            yf[0] = ysym - yspace*0.35;
            xf[1] = xsym + boxw;
            yf[1] = yf[0];
            xf[2] = xf[1];
            yf[2] = ysym + yspace*0.35;
            xf[3] = xf[0];
            yf[3] = yf[2];
            for (var j=0;j<4;j++) {
               xf[j] = xf[j] * Number(vis.attr("width"));
               yf[j] = yf[j] * Number(vis.attr("height"));
            }
            var ww = xf[1] - xf[0];
            var hh = yf[2] - yf[0];
            pos_y = pos_y - (hh/2);
            var pos_x = (tpos_x/2) - (ww/2);

            if (fill_style > 3000) {
               var patternid = JSROOT.Painter.createFillPattern(vis, fill_style, fill_color);
               p.append("svg:rect")
                  .attr("x", pos_x)
                  .attr("y", pos_y)
                  .attr("width", ww)
                  .attr("height", hh)
                  .style("fill", "url(#" + patternid + ")")
                  .style("stroke-width", line_width)
                  .style("stroke", line_color);
            }
            else {
               p.append("svg:rect")
                  .attr("x", pos_x)
                  .attr("y", pos_y)
                  .attr("width", ww)
                  .attr("height", hh)
                  .attr("fill", JSROOT.Painter.root_colors[fill_color])
                  .style("stroke-width", line_width)
                  .style("stroke", line_color);
            }
         }
         // Draw line
         if (lopt.indexOf('l') != -1) {

            // line total length (in x) is margin*0.8
            var line_length = (0.7 * pave['fMargin']) * w;
            var pos_x = (tpos_x - line_length) / 2;
            p.append("svg:line")
               .attr("x1", pos_x)
               .attr("y1", pos_y)
               .attr("x2", pos_x + line_length)
               .attr("y2", pos_y)
               .style("stroke", line_color)
               .style("stroke-width", line_width)
               .style("stroke-dasharray", line_style);

         }
         // Draw error only
         if (lopt.indexOf('e') != -1 && (lopt.indexOf('l') == -1 || lopt.indexOf('f') != -1)) {
         }
         // Draw Polymarker
         if (lopt.indexOf('p') != -1) {

            var line_length = (0.7 * pave['fMargin']) * w;
            var pos_x = tpos_x / 2;

            var filled = false;
            if ((marker_style == 8) ||
                (marker_style > 19 && marker_style < 24) ||
                (marker_style == 29))
               filled = true;

            var info_marker = JSROOT.Painter.getRootMarker(marker_style);

            var shape = info_marker['shape'];
            var filled = info_marker['toFill'];
            var toRotate = info_marker['toRotate'];
            var markerScale = 65;
            if (marker_style == 1) markerScale = 1;

            switch (shape) {
               case 6:
                  var marker = "M " + (-4 * marker_size) + " " + (-1 * marker_size)
                              + " L " + 4 * marker_size + " " + (-1 * marker_size)
                              + " L " + (-2.4 * marker_size) + " " + 4 * marker_size
                              + " L 0 " + (-4 * marker_size) + " L " + 2.8 * marker_size
                              + " " + 4 * marker_size + " z";
                  break;
               case 7:
                  var marker = "M " + (- 4 * marker_size) + " " + (-4 * marker_size)
                              + " L " + 4 * marker_size + " " + 4 * marker_size + " M 0 "
                              + (-4 * marker_size) + " 0 " + 4 * marker_size + " M "
                              + 4 * marker_size + " " + (-4 * marker_size) + " L "
                              + (-4 * marker_size) + " " + 4 * marker_size + " M "
                              + (-4 * marker_size) + " 0 L " + 4 * marker_size + " 0";
                  break;
               default:
                  var marker = d3.svg.symbol()
                              .type(d3.svg.symbolTypes[shape])
                              .size(marker_size * markerScale);
                  break;
            }
            p.append("svg:path")
               .attr("transform", function(d) { return "translate(" + pos_x + "," + pos_y + ")"; })
               .style("fill", filled ? marker_color : "none")
               .style("stroke", marker_color)
               .attr("d", marker);
         }
      }
      if (lwidth && lwidth > 1) {
         p.append("svg:line")
            .attr("x1", w+(lwidth/2))
            .attr("y1", lwidth+1)
            .attr("x2", w+(lwidth/2))
            .attr("y2", h+lwidth-1)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
         p.append("svg:line")
            .attr("x1", lwidth+1)
            .attr("y1", h+(lwidth/2))
            .attr("x2", w+lwidth-1)
            .attr("y2", h+(lwidth/2))
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
      }
      return p;
   };

   JSROOT.Painter.drawMultiGraph = function(vis, mgraph, opt) {
      var maximum, minimum, rwxmin=0, rwxmax=0, rwymin=0, rwymax=0, uxmin=0, uxmax=0, dx, dy;
      var npt = 100;
      var histo = mgraph['fHistogram'];
      var graphs = mgraph['fGraphs'];
      var scalex = 1, scaley = 1;
      var logx = false, logy = false, logz = false, gridx = false, gridy = false;
      var draw_all = true;
      if (vis['ROOT:pad'] && typeof(vis['ROOT:pad']) != 'undefined') {
         rwxmin = vis['ROOT:pad'].fUxmin;
         rwxmax = vis['ROOT:pad'].fUxmax;
         rwymin = vis['ROOT:pad'].fUymin;
         rwymax = vis['ROOT:pad'].fUymax;
         logx = vis['ROOT:pad']['fLogx'];
         logy = vis['ROOT:pad']['fLogy'];
         logz = vis['ROOT:pad']['fLogz'];
         gridx = vis['ROOT:pad']['fGridx'];
         gridy = vis['ROOT:pad']['fGridy'];
      }
      if ('fHistogram' in mgraph && mgraph['fHistogram']) {
         minimum = mgraph['fHistogram']['fYaxis']['fXmin'];
         maximum = mgraph['fHistogram']['fYaxis']['fXmax'];
         if (vis['ROOT:pad'] && typeof(vis['ROOT:pad']) != 'undefined') {
            uxmin   = JSROOT.Painter.padtoX(vis['ROOT:pad'], rwxmin);
            uxmax   = JSROOT.Painter.padtoX(vis['ROOT:pad'], rwxmax);
         }
      } else {
         var g = graphs.arr[0];
         if (g) {
            var r = g.computeRange();
            rwxmin = r['xmin']; rwymin = r['ymin'];
            rwxmax = r['xmax']; rwymax = r['ymax'];
         }
         for (var i=1; i<graphs.arr.length; ++i) {
            var rx1,ry1,rx2,ry2;
            g = graphs.arr[i];
            var r = g.computeRange();
            rx1 = r['xmin']; ry1 = r['ymin'];
            rx2 = r['xmax']; ry2 = r['ymax'];
            if (rx1 < rwxmin) rwxmin = rx1;
            if (ry1 < rwymin) rwymin = ry1;
            if (rx2 > rwxmax) rwxmax = rx2;
            if (ry2 > rwymax) rwymax = ry2;
            if (g['fNpoints'] > npt) npt = g['fNpoints'];
         }
         if (rwxmin == rwxmax) rwxmax += 1.;
         if (rwymin == rwymax) rwymax += 1.;
         dx = 0.05*(rwxmax-rwxmin);
         dy = 0.05*(rwymax-rwymin);
         uxmin = rwxmin - dx;
         uxmax = rwxmax + dx;
         if (logy) {
            if (rwymin <= 0) rwymin = 0.001*rwymax;
            minimum = rwymin/(1+0.5*JSROOT.Math.log10(rwymax/rwymin));
            maximum = rwymax*(1+0.2*JSROOT.Math.log10(rwymax/rwymin));
         } else {
            minimum  = rwymin - dy;
            maximum  = rwymax + dy;
         }
         if (minimum < 0 && rwymin >= 0) minimum = 0;
         if (maximum > 0 && rwymax <= 0) maximum = 0;
      }
      if (mgraph['fMinimum'] != -1111) rwymin = minimum = mgraph['fMinimum'];
      if (mgraph['fMaximum'] != -1111) rwymax = maximum = mgraph['fMaximum'];
      if (uxmin < 0 && rwxmin >= 0) {
         if (logx) uxmin = 0.9*rwxmin;
         //else                 uxmin = 0;
      }
      if (uxmax > 0 && rwxmax <= 0) {
         if (logx) uxmax = 1.1*rwxmax;
         //else                 uxmax = 0;
      }
      if (minimum < 0 && rwymin >= 0) {
         if(logy) minimum = 0.9*rwymin;
         //else                minimum = 0;
      }
      if (maximum > 0 && rwymax <= 0) {
         if(logy) maximum = 1.1*rwymax;
         //else                maximum = 0;
      }
      if (minimum <= 0 && logy) minimum = 0.001*maximum;
      if (uxmin <= 0 && logx) {
         if (uxmax > 1000) uxmin = 1;
         else              uxmin = 0.001*uxmax;
      }
      rwymin = minimum;
      rwymax = maximum;
      if (('fHistogram' in mgraph) && mgraph['fHistogram']) {
         mgraph['fHistogram']['fYaxis']['fXmin'] = rwymin;
         mgraph['fHistogram']['fYaxis']['fXmax'] = rwymax;
      }

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo) {
         histo = JSROOT.CreateTH1();
         histo['fXaxis']['fXmin'] = rwxmin;
         histo['fXaxis']['fXmax'] = rwxmax;
         histo['fYaxis']['fXmin'] = rwymin;
         histo['fYaxis']['fXmax'] = rwymax;
      }

      // histogram painter will be first in the pad, will define axis and interactive actions
      JSROOT.Painter.drawHistogram1D(vis, histo);

      for (var i in graphs.arr)
         JSROOT.Painter.drawGraph(vis, graphs.arr[i]);
   };


   JSROOT.Painter.canDrawObject = function(classname)
   {
      if (!classname) return false;

      if ((this.fUserPainters != null) &&
          (typeof(this.fUserPainters[classname]) === 'function')) return true;

      if (classname.match(/\bTH1/) ||
          classname.match(/\bTH2/) ||
          classname.match(/\bTH3/) ||
          classname.match(/\bTGraph/) ||
          classname.match(/\bRooHist/) ||
          classname.match(/\RooCurve/) ||
          classname == 'TF1' ||
          classname == 'TCanvas' ||
          classname == 'THStack' ||
          classname == 'TProfile') return true;

      return false;
   }

   JSROOT.Painter.drawPad = function(vis, pad) {
      var width = Number(vis.attr("width")), height = Number(vis.attr("height"));
      var x = pad['fAbsXlowNDC'] * width;
      var y = height - pad['fAbsYlowNDC'] * height;
      var w = pad['fAbsWNDC'] * width;
      var h = pad['fAbsHNDC'] * height;
      y -= h;

      var fillcolor = JSROOT.Painter.root_colors[pad['fFillColor']];
      if (pad['fFillStyle'] > 4000 && pad['fFillStyle'] < 4100)
         fillcolor = 'none';

      var border_width = pad['fLineWidth'];
      var border_color = JSROOT.Painter.root_colors[pad['fLineColor']];
      if (pad['fBorderMode'] == 0) {
         border_width = 0;
         border_color = 'none';
      }

      var new_pad = vis.append("svg:g")
         .attr("width", w)
         .attr("height", h)
         .attr("transform", "translate(" + x + "," + y + ")");

      new_pad.append("svg:rect")
         .attr("class", new_pad)
         .attr("x", 0)
         .attr("y", 0)
         .attr("width", w)
         .attr("height", h)
         .attr("fill", fillcolor)
         .style("stroke-width", border_width)
         .style("stroke", border_color);

      new_pad['ROOT:pad'] = pad;

      for (var i in pad.fPrimitives.arr)
         JSROOT.Painter.drawObjectInFrame(new_pad, pad.fPrimitives.arr[i], pad.fPrimitives.opt[i]);

      return new_pad;
   };

   JSROOT.Painter.drawPaveLabel = function(vis, pavelabel) {
      var w = Number(vis.attr("width")), h = Number(vis.attr("height"));
      var pos_x = pavelabel['fX1NDC'] * w;
      var pos_y = (1.0 - pavelabel['fY1NDC']) * h;
      var width = Math.abs(pavelabel['fX2NDC'] - pavelabel['fX1NDC']) * w;
      var height = Math.abs(pavelabel['fY2NDC'] - pavelabel['fY1NDC']) * h;
      pos_y -= height;
      var font_size = Math.round(height / 1.9);
      var fcolor = JSROOT.Painter.root_colors[pavelabel['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pavelabel['fLineColor']];
      var tcolor = JSROOT.Painter.root_colors[pavelabel['fTextColor']];
      var scolor = JSROOT.Painter.root_colors[pavelabel['fShadowColor']];
      if (pavelabel['fFillStyle'] == 0) fcolor = 'none';
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      var align = 'start', halign = Math.round(pavelabel['fTextAlign']/10);
      var baseline = 'bottom', valign = pavelabel['fTextAlign']%10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';
      var lmargin = 0;
      switch (halign) {
         case 1:
            lmargin = pavelabel['fMargin'] * width;
            break;
         case 2:
            lmargin = width/2;
            break;
         case 3:
            lmargin = width - (pavelabel['fMargin'] * width);
            break;
      }
      var lwidth = pavelabel['fBorderSize'] ? pavelabel['fBorderSize'] : 0;
      var fontDetails = JSROOT.Painter.getFontDetails(pavelabel['fTextFont']);

      var pave = vis.append("svg:g")
         .attr("width", width)
         .attr("height", height)
         .attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      pave.append("svg:rect")
         .attr("class", pave)
         .attr("x", 0)
         .attr("y", 0)
         .attr("width", width)
         .attr("height", height)
         .attr("fill", fcolor)
         .style("stroke-width", lwidth ? 1 : 0)
         .style("stroke", lcolor);


      var line = JSROOT.Painter.translateLaTeX(pavelabel['fLabel']);

      var lw = JSROOT.Painter.stringWidth(vis, line, font_size, fontDetails);
      if (lw > width)
         font_size *= 0.98 * (width / lw);

      pave.append("text")
         .attr("class", "text")
         .attr("text-anchor", align)
         .attr("x", lmargin)
         .attr("y", (height/2) + (font_size/3))
         .attr("font-weight", fontDetails['weight'])
         .attr("font-style", fontDetails['style'])
         .attr("font-family", fontDetails['name'])
         .attr("font-size", font_size)
         .attr("fill", tcolor)
         .text(line);

      if (lwidth && lwidth > 1) {
         pave.append("svg:line")
            .attr("x1", width+(lwidth/2))
            .attr("y1", lwidth+1)
            .attr("x2", width+(lwidth/2))
            .attr("y2", height+lwidth-1)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
         pave.append("svg:line")
            .attr("x1", lwidth+1)
            .attr("y1", height+(lwidth/2))
            .attr("x2", width+lwidth-1)
            .attr("y2", height+(lwidth/2))
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
      }
   };

   JSROOT.Painter.drawText = function(vis, text)
   {
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted

      var kTextNDC  = JSROOT.BIT(14);

      var pad = vis['ROOT:pad'];

      var i, w = Number(vis.attr("width")), h = Number(vis.attr("height"));
      var align = 'start', halign = Math.round(text['fTextAlign']/10);
      var baseline = 'bottom', valign = text['fTextAlign']%10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';
      var lmargin = 0;
      switch (halign) {
         case 1:
            lmargin = text['fMargin'] * w;
            break;
         case 2:
            lmargin = w/2;
            break;
         case 3:
            lmargin = w - (text['fMargin'] * w);
            break;
      }
      var font_size = Math.round(text['fTextSize'] * 0.7 * h);
      if (text.TestBit(kTextNDC)) {
         var pos_x = pad['fX1'] + text['fX']*(pad['fX2'] - pad['fX1']);
         var pos_y = pad['fY1'] + text['fY']*(pad['fY2'] - pad['fY1']);
      }
      else {
         font_size = Math.round(text['fTextSize'] * h);
         var pos_x = this.xtoPad(text['fX'], pad);
         var pos_y = this.ytoPad(text['fY'], pad);
      }
      pos_x = ((Math.abs(pad['fX1'])+pos_x)/(pad['fX2'] - pad['fX1']))*w;
      pos_y = (1-((Math.abs(pad['fY1'])+pos_y)/(pad['fY2'] - pad['fY1'])))*h;
      var tcolor = JSROOT.Painter.root_colors[text['fTextColor']];
      var fontDetails = JSROOT.Painter.getFontDetails(text['fTextFont']);

      var string = text['fTitle'];
      // translate the LaTeX symbols
      if (text['_typename'] == 'TLatex')
         string = this.translateLaTeX(text['fTitle']);

      vis.append("text")
         .attr("class", "text")
         .attr("x", pos_x)
         .attr("y", pos_y)
         .attr("font-family", fontDetails['name'])
         .attr("font-weight", fontDetails['weight'])
         .attr("font-style", fontDetails['style'])
         .attr("font-size", font_size)
         .attr("text-anchor", align)
         .attr("fill", tcolor)
         .text(string);
   };


   JSROOT.Painter.drawObjectInFrame = function(vis, obj, opt)
   {
      // ignore objects without type information
      if ((typeof obj != 'object') || (!('_typename' in obj))) return;

      var classname = obj['_typename'];

      if (classname == 'TCanvas') {
         vis['ROOT:canvas'] = obj;
         vis['ROOT:pad'] = obj;
         for (var i=0; i<obj.fPrimitives.arr.length; ++i) {
            // console.log("Draw canvas primitive " + obj.fPrimitives.arr[i]._typename + " opt = " + obj.fPrimitives.opt[i]);
            JSROOT.Painter.drawObjectInFrame(vis, obj.fPrimitives.arr[i], obj.fPrimitives.opt[i]);
         }
         return;
      }

      if (classname == 'TFrame')
         return JSROOT.Painter.createFrame(vis, obj);

      if (classname == 'TPad')
         return JSROOT.Painter.drawPad(vis, obj, opt);

      if (classname == 'TPaveLabel')
         return JSROOT.Painter.drawPaveLabel(vis, obj);

      if (classname == 'TLegend')
         return JSROOT.Painter.drawLegend(vis, obj, opt);

      if (classname == 'TPaveText')
         return JSROOT.Painter.DrawPaveText(vis, obj);

      if ((classname == 'TLatex') || (classname == 'TText'))
         return JSROOT.Painter.drawText(vis, obj);

      if (classname.match(/\bTH1/) || (classname == "TProfile"))
         return JSROOT.Painter.drawHistogram1D(vis, obj, opt);

      if (classname.match(/\bTH2/))
         return JSROOT.Painter.drawHistogram2D(vis, obj, opt);

      if (classname.match(/\bTH3/))
         return JSROOT.Painter.drawHistogram3D(vis, obj, opt);

      if (classname == 'THStack')
         return JSROOT.Painter.drawHStack(vis, obj, opt);

      if (classname == 'TF1')
         return JSROOT.Painter.drawFunction(vis, obj);

      if (classname.match(/\bTGraph/) ||
          classname.match(/\bRooHist/) ||
          classname.match(/\RooCurve/))
         return JSROOT.Painter.drawGraph(vis, obj, opt);

      if (classname == 'TMultiGraph')
         return JSROOT.Painter.drawMultiGraph(vis, obj, opt);

      if ((this.fUserPainters != null) && typeof(this.fUserPainters[classname]) === 'function')
         return this.fUserPainters[classname](vis, obj, opt);
   }

   JSROOT.draw = function(divid, obj, opt)
   {
      if ((typeof obj != 'object') || (!('_typename' in obj))) return;

      var render_to = "#" + divid;

      var fillcolor = 'white';

      d3.select(render_to).style("background-color", fillcolor);
      
      var svg = d3.select(render_to)
                   .append("svg")
//                   .attr({"width": "100%", "height": "100%"})
                   .attr("width", $(render_to).width())
                   .attr("height", $(render_to).height())
                   .style("background-color", fillcolor)
                   .attr("viewBox", "0 0 " + $(render_to).width() + " " + $(render_to).height())
                   .attr("preserveAspectRatio", "xMidYMid meet")
                   .attr("pointer-events", "all");
                   // .call(d3.behavior.zoom().on("zoom", JSROOT.Painter.redraw));
      
      svg['render_to'] = render_to; 

      return JSROOT.Painter.drawObjectInFrame(svg, obj, opt);
   }
   
   // =========== painter of hierarchical structures =================================
   
   JSROOT.HList = [];

   JSROOT.DelHList = function(_name) {
      for (var i in JSROOT.HList)
         if (JSROOT.HList[i].name == _name) {
            var old = JSROOT.HList[i];
            JSROOT.HList.splice(i,1);
            delete old;
            return true;
         }
   } 

   JSROOT.AddHList = function(_name,_h) {
      JSROOT.DelHList(_name);
      JSROOT.HList.push({name:_name, h: _h});
   }
   
   JSROOT.H = function(name) {
      for (var i in JSROOT.HList)
         if (JSROOT.HList[i].name==name) return JSROOT.HList[i].h; 
      return null;
   }
   
   JSROOT.HierarchyPainter = function(name,frameid) {
      JSROOT.AddHList(name, this);
      this.name = name;
      this.frameid = frameid;
      this.h = null;    // hierarchy
      this.url = null;  // url of online server 
   }
   
   JSROOT.HierarchyPainter.prototype.GlobalName = function(suffix) {
      var res = "JSROOT.H(\'" + this.name + "\')";
      if (suffix!=null) res+=suffix;
      return res;
   }
   
   JSROOT.HierarchyPainter.prototype.ListHierarchy = function(folder, lst) {

      folder['_childs'] = [];
      
      for (var i in lst.arr) {
         var obj = lst.arr[i];
         var item = { 
               _name : obj['fName'],  
               _kind : "ROOT." + obj['_typename'],
               _readobj: obj
         };

  
         folder._childs.push(item);
      }
   }

   JSROOT.HierarchyPainter.prototype.StreamerInfoHierarchy = function(folder, lst) {

      folder['_childs'] = [];
      
      for (var i in lst.arr) {
         var entry = lst.arr[i]

         if (typeof(entry['fName']) == 'undefined') {
            console.log("strange element in StreamerInfo with name " + entry['fName']);
            continue;
         }
         
         var item = { 
            _name : entry['fName'],  
            _kind : "",
            _childs : []
         };
  
         folder._childs.push(item);
         
         item._childs.push({ _name : 'Checksum: ' + entry['fCheckSum'] }); 
         item._childs.push({ _name : 'Class version: ' + entry['fClassVersion'] }); 
         if (entry['fTitle'] != '')
            item._childs.push({ _name : 'Title: ' + entry['fTitle'] });
         if (typeof entry['fElements'] == 'undefined') continue;   
         for (var l in entry['fElements']['arr']) {
            var elem = entry['fElements']['arr'][l];
            if ((elem==null) || (typeof(elem['fName']) == 'undefined')) continue;
            var info = elem['fTypeName'] + " " + elem['fName'] + ";";
            if (elem['fTitle'] != '') info += " // " + elem['fTitle']; 
            item._childs.push({ _name :  info });
         }
      }
   }
   
   JSROOT.HierarchyPainter.prototype.TreeHierarchy = function(node, obj)
   {
      node._childs = [];
      
      for (var i in obj['fBranches'].arr) {
         var branch = obj['fBranches'].arr[i]; 
         var nb_leaves = branch['fLeaves'].arr.length;
         
         // display branch with only leaf as leaf
         if (nb_leaves == 1 && branch['fLeaves'].arr[0]['fName'] == branch['fName']) {
            nb_leaves = 0; 
         }
         
         // console.log("name = " + branch['fName'] + " numleavs = " + nb_leaves);
         
         var subitem = {
            _name : branch['fName'],  
            _kind : nb_leaves > 0 ? "ROOT.TBranch" : "ROOT.TLeafF"
         }
         
         node._childs.push(subitem);
         
         if (nb_leaves > 0) {
            subitem._childs = [];
            for (var j=0; j<nb_leaves; ++j) {
               var leafitem = {
                     _name : branch['fLeaves'].arr[j]['fName'],  
                     _kind : "ROOT.TLeafF"
                  }
               subitem._childs.push(leafitem);
            }
         }
      }
   }

   JSROOT.HierarchyPainter.prototype.KeysHierarchy = function(folder, keys, file) {

      folder['_childs'] = [];
      
      var painter = this;
   
      for (var i in keys) {
         var key = keys[i];
         
         var item = { 
               _name : key['fName'] + ";" + key['fCycle'],  
               _kind : "ROOT." + key['fClassName'],
               _title : key['fTitle'],
               _keyname : key['fName'],
               _readobj : null
         };
         
         if ('fRealName' in key) 
            item['_realname'] = key['fRealName'] + ";" + key['fCycle'];
         
         // console.log("key class = " + key['fClassName']);

         if ((key['fClassName'] == 'TTree' || key['fClassName'] == 'TNtuple')) {
            item["_more"] = true;
            
            item['_expand'] = function(node, obj) {
               painter.TreeHierarchy(node, obj);
               return true;
            }
         } else 
         if (key['fClassName'] == 'TDirectory' || key['fClassName'] == 'TDirectoryFile') {
            item["_more"] = true;
            item["_isdir"] = true;
            item['_expand'] = function(node, obj) {
               painter.KeysHierarchy(node, obj.fKeys);
               return true;
            }
         } else
         if ((key['fClassName'] == 'TList') && (key['fName']=='StreamerInfo') && (file!=null)) {
            item['_name'] = 'StreamerInfo';
            item['_kind'] = "ROOT.TStreamerInfoList";
            item['_title'] = "List of streamer infos for binary I/O";
            item['_readobj'] = file.fStreamerInfos;
            item['_expand'] = function(node, obj) {
               painter.StreamerInfoHierarchy(node, obj);
               return true;
            }
            
         } else 
         if (key['fClassName'] == 'TList' || key['fClassName'] == 'TObjArray' || key['fClassName'] == 'TClonesArray') {
            item["_more"] = true;
            item['_expand'] = function(node, obj) {
               painter.ListHierarchy(node, obj);
               return true;
            }
         }
  
         folder._childs.push(item);
      }
   }
  
   
   JSROOT.HierarchyPainter.prototype.FileHierarchy = function(file)
   {
      var painter = this;
      
      var folder = { 
            _name : file.fFileName, 
            _kind : "ROOT.TFile", 
            _file : file,
            _get : function(item, callback) {
               if ((this._file == null) || (item._readobj != null)) {
                  if (typeof callback == 'function') callback(item, item._readobj);
                  return;
               }
               
               var fullname = painter.itemFullName(item, this);
               // var pos = fullname.lastIndexOf(";");
               // if (pos>0) fullname = fullname.slice(0, pos);
               
               this._file.ReadObject(fullname, -1, function(obj) {
                  item._readobj = obj;
                  if ('_expand' in item) item._name = item._keyname; // remove cycle number for objects supporting expand
                  if (typeof callback == 'function') callback(item, obj);
               });
            }
      };
      
      this.KeysHierarchy(folder, file.fKeys, file);
      
      return folder;
   }

   JSROOT.HierarchyPainter.prototype.Find = function(fullname, top, replace) {

      if (!top) top = this.h;
      
      if (fullname.length==0) return top;
      
      var pos = -1;
      
      do {
        // we try to find element with slashes inside       
        pos = fullname.indexOf("/", pos+1);
  
        var localname = (pos < 0) ? fullname : fullname.substr(0, pos);  

        for (var i in top._childs) 
           if (top._childs[i]._name == localname) {
             top._childs[i]['_parent'] = top; // set parent pointer when searching child
             if ((pos+1 == fullname.length) || (pos<0)) {
                if (replace!=null) top._childs[i] = replace;
                return top._childs[i]; 
             }
         
             return this.Find(fullname.substr(pos+1), top._childs[i], replace);
           }
      } while (pos>0);
      return null;
   }

   JSROOT.HierarchyPainter.prototype.itemFullName = function(node, uptoparent)
   {
      var res = ""; 
         
      while ('_parent' in node) {
         if (res.length>0) res = "/" + res;
         res = node._name + res;
         node = node._parent;
         if ((uptoparent!=null) && (node == uptoparent)) break;
      }
      
      return res;  
   }
   
   JSROOT.HierarchyPainter.prototype.CheckCanDo = function(node, cando) 
   {
      var kind = node["_kind"];
      if (kind == null) kind = "";
      
      cando.expand = ('_more' in node);

      if (kind == "ROOT.Session") cando.img1 = JSROOT.source_dir+'img/globe.gif'; else
      if (kind.match(/\bROOT.TH1/)) { cando.img1 = JSROOT.source_dir+'img/histo.png'; cando.scan = false; cando.display = true; } else
      if (kind.match(/\bROOT.TH2/)) { cando.img1 = JSROOT.source_dir+'img/histo2d.png'; cando.scan = false; cando.display = true; } else  
      if (kind.match(/\bROOT.TH3/)) { cando.img1 = JSROOT.source_dir+'img/histo3d.png'; cando.scan = false; cando.display = true; } else
      if (kind == "ROOT.TCanvas") { cando.img1 = JSROOT.source_dir+'img/canvas.png'; cando.display = true; } else
      if (kind == "ROOT.TProfile") { cando.img1 = JSROOT.source_dir+'img/profile.png'; cando.display = true; } else
      if (kind.match(/\bROOT.TGraph/)) { cando.img1 = JSROOT.source_dir+'img/graph.png'; cando.display = true; } else
      if (kind == "ROOT.TF1") { cando.img1 = JSROOT.source_dir+'img/graph.png'; cando.display = true; } else
      if (kind == "ROOT.TTree") cando.img1 = JSROOT.source_dir+'img/tree.png'; else
      if (kind == "ROOT.TFolder") { cando.img1 = JSROOT.source_dir+'img/folder.gif'; cando.img2 = JSROOT.source_dir+'img/folderopen.gif'; }  else
      if (kind == "ROOT.TNtuple") cando.img1 = JSROOT.source_dir+'img/tree.png';   else
      if (kind == "ROOT.TBranch") cando.img1 = JSROOT.source_dir+'img/branch.png';   else
      if (kind.match(/\bROOT.TLeaf/)) cando.img1 = JSROOT.source_dir+'img/leaf.png'; else
      if (kind == "ROOT.TStreamerInfoList") { cando.img1 = JSROOT.source_dir+'img/question.gif'; cando.expand = false; cando.display = true; } else
      if ((kind.indexOf("ROOT.")==0) && JSROOT.Painter.canDrawObject(kind.slice(5))) { cando.img1 = JSROOT.source_dir+'img/histo.png'; cando.scan = false; cando.display = true; }
   }
   
   JSROOT.HierarchyPainter.prototype.createNode = function(node, fullname, parent) 
   {
      var nodename = node._name;
      
      var nodefullname = "";
      
      if (node != this.h) {
         nodefullname = nodename;
         if (fullname.length>0) nodefullname = fullname+ "/" + nodename;
         node['_parent'] = parent;
      }
      
      var cando = {
         expand : false,
         display : false,
         scan : true,
         open : false,
         img1 : "",
         img2 : "",
         html : ""
      };
      
      this.CheckCanDo(node, cando);
      
      // console.log("add kind = " + kind + "  name = " + node._name);  

      if (!node._childs || !cando.scan) {
         if (cando.expand) {   
            cando.html = "javascript: " + this.GlobalName() + ".expand(\'"+nodefullname+"\');";
            if (cando.img1.length == 0) {
               cando.img1 = JSROOT.source_dir+'img/folder.gif'; 
               cando.img2 = JSROOT.source_dir+'img/folderopen.gif';
            }
         } else
         if (cando.display) {
            cando.html = "javascript: " + this.GlobalName() + ".display(\'"+nodefullname+"\');";
         } else
         if (cando.open) 
            cando.html = nodefullname;
      } 
      
      if (cando.img2 == "") cando.img2 = cando.img1;
      
      node['_d'] = {
        name : nodename,
        url : cando.html,
        title : "",
        icon : cando.img1,
        iconOpen :  cando.img2,
        _id : 0,      // id used in html
        _io : false,  // is open
        _is : false,  // is selected 
        _ls : false,  // last sibling
        _hc : false  // has childs
      };

      if ('_realname' in node)
         node['_d']['name'] = node['_realname'];
      
      if ('_title' in node)
         node['_d']['title'] = node['_title'];
      
      if ('_fullname' in node)
         node['_d']['title'] += ("  fullname: " + node['_fullname']);
      
      if (node['_d']['title'].length == 0)
         node['_d']['title'] = node['_d']['name'];
      
      if (parent && parent._childs && (parent._childs[parent._childs.length-1] == node))
         node['_d']._ls = true;
      
      node['_d']._id = this.grid++;

      // allow context menu only for objects which can be displayed
      if (cando.display) 
         node['_d']['ctxt'] = this.GlobalName() + ".contextmenu(this, event, \'"+nodefullname+"\')";

      if (cando.scan && ('_childs' in node)) {
         node['_d']._hc = true;
         for (var i in node._childs) 
            this.createNode(node._childs[i], nodefullname, node);
      }
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(force)
   {
      if (this.frameid == null) return;
      var elem = document.getElementById(this.frameid);
      if (elem==null) return;
      
      if (this.h==null) {
         elem.innerHTML = "<h2>null</h2>";
         return;
      }
      
      if (force && this.h._d!= null) {
         delete this.h._d;
         this.h._d = null;
      }
      
      if (this.h._d == null) {
         this.grid = 0;
         this.createNode(this.h, "", null);
      }

      this['html'] = "<p>";
      this['html'] += "<a href=\"javascript: " + this.GlobalName() + ".toggle(true);\">open all</a>";
      this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".toggle(false);\">close all</a>";
      if (this.url!=null)
         this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".reload();\">reload</a>";                           
      this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".clear();\">clear</a>";                           
                  
      this['html']+="</p>";

      this['html'] += '<div class="dtree">'

      this.addItemHtml(this.h);
      
      this['html'] += '</div>';
      
      elem.innerHTML = this['html'];
   }
   
   JSROOT.HierarchyPainter.prototype.addItemHtml = function(hitem, onlyitem)
   {
      if (this.icon==null)
         this.icon = {
            root           : JSROOT.source_dir+'img/base.gif',
            folder         : JSROOT.source_dir+'img/folder.gif',
            folderOpen     : JSROOT.source_dir+'img/folderopen.gif',
            node           : JSROOT.source_dir+'img/page.gif',
            empty          : JSROOT.source_dir+'img/empty.gif',
            line           : JSROOT.source_dir+'img/line.gif',
            join           : JSROOT.source_dir+'img/join.gif',
            joinBottom     : JSROOT.source_dir+'img/joinbottom.gif',
            plus           : JSROOT.source_dir+'img/plus.gif',
            plusBottom     : JSROOT.source_dir+'img/plusbottom.gif',
            minus          : JSROOT.source_dir+'img/minus.gif',
            minusBottom    : JSROOT.source_dir+'img/minusbottom.gif',
            nlPlus         : JSROOT.source_dir+'img/nolines_plus.gif',
            nlMinus        : JSROOT.source_dir+'img/nolines_minus.gif'
         };
      
      var isroot = (hitem == this.h);

      var node = hitem._d;
      var idname = this.name + "_id_" + node._id; 
      
      if (!onlyitem) this['html'] += '<div class="dTreeNode" id="z' + idname + '">';
      
      // build indent
      var sindent = "";
      var prnt = isroot ? null : hitem._parent;
      while ((prnt!=null) && (prnt!=this.h)) {
         sindent = '<img src="' + (!prnt._d._ls ? this.icon.line : this.icon.empty ) + '" alt="" />' + sindent;
         prnt = prnt._parent;
      }
      this['html'] += sindent;
      
      var opencode = this.GlobalName() + ".open(\'"+this.itemFullName(hitem)+"\')";
      
      if (isroot) {
         // for root node no extra code 
      } else
      if (node._hc) {
         this['html'] += '<a href="javascript: ' + opencode + '"><img src="';
         this['html'] += ( (node._io) ? (node._ls ? this.icon.minusBottom : this.icon.minus) : (node._ls ? this.icon.plusBottom : this.icon.plus ) );
         this['html'] += '" alt="" /></a>';
      } else {
         this['html'] += '<img src="' + ((node._ls ? this.icon.joinBottom : this.icon.join)) + '" alt="" />';
      }
      
      // make node icon
      if (!node.icon) node.icon = isroot ? this.icon.root : ((node._hc) ? this.icon.folder : this.icon.node);
      if (!node.iconOpen) node.iconOpen = (node._hc) ? this.icon.folderOpen : this.icon.node;
      if (isroot) {
         node.icon = this.icon.root;
         node.iconOpen = this.icon.root;
      }
      
      this['html'] += '<img src="' + ((node._io) ? node.iconOpen : node.icon) + '" alt=""/>';

      if (node.url) {
         this['html'] += '<a class="' + (node._is ? 'nodeSel' : 'node') + '" href="' + node.url + '"';
         if (node.title) this['html'] += ' title="' + node.title + '"';
         if (node.ctxt) this['html'] += ' oncontextmenu="' + node.ctxt + '"';
         this['html'] += '>' + node.name + '</a>';
      } else 
      if (node._hc && !isroot) {
         this['html'] += '<a href="javascript: ' + opencode + '" class="node"';
         if (node.title) this['html'] += ' title="' + node.title + '"';
         this['html'] += '>' + node.name + '</a>';
      } else {
         this['html'] += node.name;
      }

      if (onlyitem) return; 
         
      this['html'] += '</div>';
      
      var childs_display = node._hc && (isroot || node._io);   

      // place for childs
      
      this['html'] += '<div id="d' + idname + '" class="clip" style="display:' + (childs_display ? 'block' : 'none') + ';">';
      if (childs_display) 
         for (var i in hitem._childs)
            this.addItemHtml(hitem._childs[i]);
      this['html'] += '</div>';
   }
   
   
   JSROOT.HierarchyPainter.prototype.open = function(itemname)
   {
      var hitem = this.Find(itemname);
      if (hitem==null) return;
      
      this.setDNodeOpenStatus(hitem, !hitem._d._io);
   }
   
   JSROOT.HierarchyPainter.prototype.clear = function()
   {
      if ('disp' in this) this['disp'].Reset();
   }
   
   JSROOT.HierarchyPainter.prototype.setDNodeOpenStatus = function(hitem, status, force)
   {
      if (hitem==null) return;
      
      var node = hitem._d;
      var idname = this.name + "_id_" + node._id;

      node._io = status;

      var zDiv = document.getElementById('z' + idname);
      var dDiv = document.getElementById('d' + idname);
      
      if (zDiv) {
         this['html'] = '';
         this.addItemHtml(hitem, true);
         zDiv.innerHTML = this['html'];
      } 

      if (dDiv) {
         if (node._io && node._hc && (force || (dDiv.childNodes.length==0))) {
            this['html'] = '';
            for (var i in hitem._childs)
               this.addItemHtml(hitem._childs[i]);
            dDiv.innerHTML = this['html']; 
         }
         dDiv.style.display = node._io ? 'block': 'none';
      }
   }
   
   JSROOT.HierarchyPainter.prototype.toggle = function(status)
   {
      var painter = this;
      
      var toggleItem = function(hitem) {

         if (hitem != painter.h)
            hitem._d._io = status;
         
         if ('_childs' in hitem)
            for (var i in hitem._childs)
               toggleItem(hitem._childs[i]);
      }
      
      toggleItem(this.h);

      this.RefreshHtml();
   }
   
   
   JSROOT.HierarchyPainter.prototype.get = function(itemname, callback)
   {
      var item = this.Find(itemname);

      var curr = item;
      while (curr != null) {
         if (('_get' in curr) && (typeof(curr._get)=='function')) {
            curr._get(item, callback);
            return;
         }
         curr = ('_parent' in curr) ? curr['_parent'] : null;
      }
      
      if(('_get' in this) && (typeof this._get == 'function')) {
         // process get in central method - if exists
         this._get(item, callback);
      } else
         if (typeof callback == 'function') callback(item, null);
   }

   JSROOT.HierarchyPainter.prototype.display = function(itemname)
   {
      var pthis = this;
      
      // console.log("try display  " + itemname);
      
      this.get(itemname, function(item, obj) {
         // if (obj!=null) console.log("get object " + itemname + " for the drawing");
         
         if (('disp' in pthis) && (typeof pthis['disp']['Draw'] == 'function'))
            pthis['disp'].Draw(itemname, obj);
      });
   }
   
   JSROOT.HierarchyPainter.prototype.reload = function()
   {
      if (this.url!=null) this.OpenOnline(this.url);
   }
   
   JSROOT.HierarchyPainter.prototype.ExpandDtree = function(node)
   {
      var itemname = this.itemFullName(node);
      
      for (var i in node._childs)
         this.createNode(node._childs[i], itemname, node);
      
      node._d._hc = true;
      node._d._io = true;
      node._d.url = "";
      node._d.name = node._name;
      
      this.setDNodeOpenStatus(node, true, true);
   }
   
   JSROOT.HierarchyPainter.prototype.expand = function(itemname)
   {
      var painter = this;
      
      this.get(itemname, function(item, obj) {
         // if (obj!=null) console.log("get object for the drawing");
         
         if ((item==null) || (obj==null)) return;
         
         var curr = item;
         while (curr != null) {
            if (('_expand' in curr) && (typeof(curr['_expand'])=='function')) {
               if (curr['_expand'](item, obj))
                  painter.ExpandDtree(item);
               return;   
            }
            curr = ('_parent' in curr) ? curr['_parent'] : null;
         }
      });
   }
   
   JSROOT.HierarchyPainter.prototype.OpenRootFile = function(url)
   {
      var pthis = this;
      
      var f = new JSROOT.TFile(url, function(file) {
         if (file==null) return;
         // for the moment file is the only entry
         pthis.h = pthis.FileHierarchy(file);
         
         pthis.RefreshHtml();
      });
   }
   
   JSROOT.HierarchyPainter.prototype.AddOnlineMethods = function(h)
   {
      if (typeof h != 'object') return;
      
      var painter = this;
      
      h['_get'] = function (item, callback) {
         var itemname = painter.itemFullName(item);
         
         if ((itemname.length>0) && (itemname.lastIndexOf("/") != itemname.length-1)) itemname+="/";
         
         itemname += ('_more' in item) ? "h.json?compact=3" : "root.json.gz?compact=3";
         
         var itemreq = JSROOT.NewHttpRequest(itemname, 'text', function(itemres) {

            var obj = JSROOT.parse(itemres);
             
            if (typeof callback == 'function') callback(item, obj);
         });
         
         itemreq.send(null);
      }
      
      h['_expand'] = function(node, obj) {
         // central function for all expand
      
         if ((obj!=null) && (node!=null) && ('_childs' in obj)) {
            node._childs = obj._childs;
            obj._childs = null;
            return true;
         }
         return false;
      }
   }
   

   JSROOT.HierarchyPainter.prototype.OpenOnline = function(url)
   {
      this.url = url; // remember url to be able reload id 
      
      var painter = this;
      
      var req = JSROOT.NewHttpRequest(url, 'text', function(arg) {
          if (arg==null) return;
          
          painter.h = JSON.parse(arg);
          
          painter.AddOnlineMethods(painter.h);
          
          if (painter.h!=null) painter.RefreshHtml(true);
      
      });
      
      req.send(null);
   }

   
   JSROOT.HierarchyPainter.prototype.ShowStreamerInfo = function(sinfo)
   {
      this.h = { _name : "StreamerInfo" };
      this.StreamerInfoHierarchy(this.h, sinfo);
      this.RefreshHtml();
   }
   
   JSROOT.HierarchyPainter.prototype.Adopt = function(h)
   {
      this.h = h;
      this.RefreshHtml();
   }
   
   JSROOT.HierarchyPainter.prototype.CreateSingleOnlineElement = function()
   {
      this.h = {
         _name : ""   
      }
      this.AddOnlineMethods(this.h);
   }

   
   JSROOT.HierarchyPainter.prototype.contextmenu = function(element, event, itemname)
   {
      event.preventDefault();

      var menu = JSROOT.Painter.createmenu(event);

      var painter = this;
      
      JSROOT.Painter.menuitem(menu,"Draw", function() { painter.display(itemname); });
      JSROOT.Painter.menuitem(menu,"Draw in new window", function() { window.open(itemname+"/draw.htm"); });
      JSROOT.Painter.menuitem(menu,"Draw as png", function() { window.open(itemname+"/root.png?w=400&h=300&opt=colz"); });
      JSROOT.Painter.menuitem(menu,"Close", function() { });
      
      return false;
   }
   
   JSROOT.HierarchyPainter.prototype.SetDisplay = function(kind, frameid)
   {
      if (kind == "collapsible") kind = new JSROOT.CollapsibleDisplay(frameid); else
      if (kind == "tabs") kind = new JSROOT.TabsDisplay(frameid);
      
      this['disp'] = kind;
   }

   
   // JSROOT.MDIDisplay - class to manage multiple document interface for frawings 
   
   JSROOT.MDIDisplay = function(frameid) {
      this.frameid = frameid;
   } 
   
   JSROOT.MDIDisplay.prototype.ForEach = function(userfunc, only_visible) {
      alert("ForEach not implemented");
   }

   JSROOT.MDIDisplay.prototype.FindFrame = function(searchitemname) {
      var found_frame = null;
      
      this.ForEach(function(frame, itemname) {
         if (itemname == searchitemname) found_frame = frame; 
      });
      
      return found_frame;
   }

   JSROOT.MDIDisplay.prototype.Draw = function(itemname, obj) {
      if (!obj) return;
      
      var frame = this.FindFrame(itemname);
      if (frame!=null) {
         this.ActivateFrame(frame);
         return;
      }
      
      var issinfo = itemname.lastIndexOf("StreamerInfo") >= 0;
      if (!issinfo && !JSROOT.Painter.canDrawObject(obj['_typename'])) return;
      
      frame = this.CreateFrame(itemname);
      
      var hid = $(frame).attr('id');

      if (issinfo) {
         var painter = new JSROOT.HierarchyPainter('sinfo', hid);
         painter.ShowStreamerInfo(obj);
      } else {
         
         // set aspect ratio for the place, where object will be drawn
         var height = $(frame).width() * 0.66;
         
         document.getElementById(hid).setAttribute("style", "height:"+height+"px");
         document.getElementById(hid).style.height=""+height+'px';
         
         document.getElementById(hid)['painter'] = JSROOT.draw(hid, obj);
      }
      
      this.ActivateFrame(frame);
   }
   
   JSROOT.MDIDisplay.prototype.CheckResize = function() {
      this.ForEach(function(panel, itemname, painter) {
         if ((painter!=null) && (typeof painter['CheckResize'] == 'function')) painter.CheckResize();
      });   
   }
   
   JSROOT.MDIDisplay.prototype.Reset = function() {
      document.getElementById(this.frameid).innerHTML = '';
      JSROOT.DelHList('sinfo');
   }

   // ==================================================
   
   JSROOT.CloseCollapsible = function(e, el) {
      var sel = $(el)[0].textContent;
      if (typeof(sel) == 'undefined') return;
      sel.replace(' x', '');
      sel.replace(';', '');
      sel.replace(' ', '');
      $(el).next().andSelf().remove();
      e.stopPropagation();
   };

   JSROOT.CollapsibleDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }
   
   JSROOT.CollapsibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);
   
   JSROOT.CollapsibleDisplay.prototype.ForEach = function(userfunc, only_visible) {
      var topid  = this.frameid + '_collapsible';
      
      if (document.getElementById(topid)==null) return;
      
      if (typeof userfunc != 'function') return;
      
      $('#' + topid).children().each(function() {

         if (!('itemname' in this)) return;

         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;
         
         userfunc(this, this['itemname'], this['painter']);
      });
   }
   
   JSROOT.CollapsibleDisplay.prototype.ActivateFrame = function(frame) {
      if ($(frame).is(":hidden")) {
         $(frame).prev()
            .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
            .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
            .next().toggleClass("ui-accordion-content-active").slideDown(0);
      }
      $(frame).prev()[0].scrollIntoView();
   }
   
   JSROOT.CollapsibleDisplay.prototype.CreateFrame = function(itemname) {
      
      var topid = this.frameid + '_collapsible';
      
      if (document.getElementById(topid) == null)
         $("#right-div").append('<div id="'+topid+'" class="ui-accordion ui-accordion-icons ui-widget ui-helper-reset" style="overflow:auto; overflow-y:scroll; height:100%"></div>');
      
      var hid = topid + "_sub" + this.cnt++;
      var uid = hid + "h";
      
      var entryInfo = "<h5 id=\""+uid+"\"><a> " + itemname + "</a>&nbsp; </h5>\n";
      entryInfo += "<div id='" + hid + "'></div>\n";
      $("#"+topid).append(entryInfo);
      
      document.getElementById(hid)['itemname'] = itemname;
      
      $('#'+uid)
        .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
        .hover(function() { $(this).toggleClass("ui-state-hover"); })
        .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
        .append('<button type="button" class="closeButton" title="close canvas" onclick="JSROOT.CloseCollapsible(event, \'#'+uid+'\')"><img src="'+JSROOT.source_dir+'/img/remove.gif"/></button>')
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
   
      $('#'+uid)
        .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
        .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
        .next().toggleClass("ui-accordion-content-active").slideToggle(0);

      // $('#'+uid)[0].scrollIntoView();

      return $("#"+hid);
   }
   

   // ================================================
   
   JSROOT.TabsDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0;
   }
   
   JSROOT.TabsDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);
   
   JSROOT.TabsDisplay.prototype.ForEach = function(userfunc, only_visible) {
      var topid  = this.frameid + '_tabs';
      
      if (document.getElementById(topid)==null) return;
      
      if (typeof userfunc != 'function') return;
      
      var cnt = -1;
      var active = $('#' + topid).tabs( "option", "active" );
      
      $('#' + topid).children().each(function() {
         // check if only_visible specified
         if (only_visible && (cnt++!=active)) return;
         
         if (! ('itemname' in this)) return;
         
         userfunc(this, this['itemname'], this['painter']);
      });
   }
   
   JSROOT.TabsDisplay.prototype.ActivateFrame = function(frame) {
      var cnt = 0, id = -1;
      this.ForEach(function(fr) {
         if (fr===frame) id = cnt;
         cnt++;
      });
      
      $('#' + this.frameid + "_tabs").tabs("option", "active", id);
   }

   JSROOT.TabsDisplay.prototype.CreateFrame = function(itemname) {

      var topid = this.frameid + '_tabs';

      var hid = topid + "_sub" + this.cnt++;
      
      var li = '<li><a href="#' + hid + '">' + itemname + '</a><span class="ui-icon ui-icon-close" role="presentation">Remove Tab</span></li>';
      var cont = '<div id="' + hid + '"></div>';

      if (document.getElementById(topid) == null) {
         $("#right-div").append(
           '<div id="'+topid+'">' +
           ' <ul>' + li +
           ' </ul>'+ cont +
           '</div>'
          );
         
         var tabs = $("#" + topid).tabs({ heightStyle: "fill" });
         
         tabs.delegate( "span.ui-icon-close", "click", function() {
            var panelId = $( this ).closest( "li" ).remove().attr( "aria-controls" );
            $( "#" + panelId ).remove();
            tabs.tabs( "refresh" );
         });
      } else {
         
         // var tabs = $("#tabs").tabs();
         
         $("#" + topid).find( ".ui-tabs-nav" ).append( li );
         $("#" + topid).append(cont);
         $("#" + topid).tabs("refresh");
         $("#" + topid).tabs( "option", "active", -1 );
      }
      
      document.getElementById(hid)['itemname'] = itemname;
      
      return $('#'+hid);
   }

})();


// JSRootPainter.js ends


// example of user code for streamer and painter

/*
(function(){

   Amore_String_Streamer = function(buf, obj, prop, streamer) {

      console.log("read property " + prop + " of typename " + streamer[prop]['typename']);

      obj[prop] = buf.ReadTString();
   }

   Amore_Painter = function(vis, obj, opt) {
      // custom draw function.

      console.log("Draw user type " + obj['_typename']);

      JSROOT.Painter.drawObjectInFrame(vis, obj['fVal'], opt);
   }

   JSROOT.addUserStreamer("amore::core::String_t", Amore_String_Streamer);

   JSROOT.addUserPainter("amore::core::MonitorObjectHisto<TH1F>", Amore_Painter);

})();
*/


