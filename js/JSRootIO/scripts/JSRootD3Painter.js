// JSROOTD3Painter.js
//
// core methods for Javascript ROOT Graphics, using d3.js.
//

// The "source_dir" variable is defined in JSRootInterface.js

var d_tree, key_tree;

var kWhite = 0, kBlack = 1, kGray = 920, kRed = 632, kGreen = 416, kBlue = 600,
    kYellow = 400, kMagenta = 616, kCyan = 432, kOrange = 800, kSpring = 820,
    kTeal = 840, kAzure = 860, kViolet = 880, kPink = 900;

var symbols_map = {
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

/*
 * Function that generates all root colors
 */
function generateAllColors () {
   var colorMap = new Array(
      'rgb(255,255,255)',
      'rgb(0,0,0)',
      'rgb(255,0,0)',
      'rgb(0,255,0)',
      'rgb(0,0,255)',
      'rgb(255,255,0)',
      'rgb(255,0,255)',
      'rgb(0,255,255)',
      'rgb(89,211,84)',
      'rgb(89,84,216)',
      'rgb(254,254,254)',
      'rgb(191,181,173)',
      'rgb(76,76,76)',
      'rgb(102,102,102)',
      'rgb(127,127,127)',
      'rgb(153,153,153)',
      'rgb(178,178,178)',
      'rgb(204,204,204)',
      'rgb(229,229,229)',
      'rgb(242,242,242)',
      'rgb(204,198,170)',
      'rgb(204,198,170)',
      'rgb(193,191,168)',
      'rgb(186,181,163)',
      'rgb(178,165,150)',
      'rgb(183,163,155)',
      'rgb(173,153,140)',
      'rgb(155,142,130)',
      'rgb(135,102,86)',
      'rgb(175,206,198)',
      'rgb(132,193,163)',
      'rgb(137,168,160)',
      'rgb(130,158,140)',
      'rgb(173,188,198)',
      'rgb(122,142,153)',
      'rgb(117,137,145)',
      'rgb(104,130,150)',
      'rgb(109,122,132)',
      'rgb(124,153,209)',
      'rgb(127,127,155)',
      'rgb(170,165,191)',
      'rgb(211,206,135)',
      'rgb(221,186,135)',
      'rgb(188,158,130)',
      'rgb(198,153,124)',
      'rgb(191,130,119)',
      'rgb(206,94,96)',
      'rgb(170,142,147)',
      'rgb(165,119,122)',
      'rgb(147,104,112)',
      'rgb(211,89,84)');

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
         colorMap[colorn] = 'rgb(' + circleSets[i][3*j] + ',' + circleSets[i][3*j+1] + ',' + circleSets[i][3*j+2] + ')';
         colorn = rectangleColors[i] + j - 9;
         colorMap[colorn] = 'rgb('+ rectSets[i][3*j] + ',' + rectSets[i][3*j+1] + ',' + rectSets[i][3*j+2] + ')';
      }
    }
    return colorMap;
};

function getFontDetails(fontName) {
   var weight = "";
   var style = "";
   var name = "Arial";

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

/*
 * Function that returns the SVG symbol type identifier for a given root matker
 * The result is an array with 3 elements:
 *    the first is the identifier of the root marker in the SVG symbols
 *    the second is true if the shape is filled and false if it is open
 *    the third is true if the shape should be rotated
 * The identifier will be 6 if the shape is a star or 7 if it is '*'
 */
function getRootMarker(markers, i) {
   var marker = markers[i];
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

function stringWidth(svg, line, font_name, font_weight, font_size, font_style) {
   /* compute the bounding box of a string by using temporary svg:text */
   var text = svg.append("svg:text")
       .attr("class", "temp_text")
       .attr("font-family", font_name)
       .attr("font-weight", font_weight)
       .attr("font-style", font_style)
       .attr("font-size", font_size)
       .style("opacity", 0)
       .text(line);
   w = text.node().getBBox().width;
   text.remove();
   return w;
}

function format_id(id) {
   /* format the string id to remove specials characters
      (that cannot be used in id strings) */
   var g_id = id;
   while (g_id.indexOf(' ') != -1)
      g_id = g_id.replace(' ', '_');
   while (g_id.indexOf(':') != -1)
      g_id = g_id.replace(':', '_');
   while (g_id.indexOf('.') != -1)
      g_id = g_id.replace('.', '_');
   while (g_id.indexOf('>') != -1)
      g_id = g_id.replace('>', 'gt');
   while (g_id.indexOf('<') != -1)
      g_id = g_id.replace('<', 'lt');
   return g_id;
};

/*
    Polyfill for touch dblclick
    http://mckamey.mit-license.org
*/
function doubleTap(elem, speed, distance) {
   if (!('ontouchstart' in elem)) {
      // non-touch has native dblclick and no need for polyfill
      return;
   }
   // default dblclick speed to half sec
   speed = Math.abs(+speed) || 500;//ms
   // default dblclick distance to within 40x40 area
   distance = Math.abs(+distance) || 40;//px

   var taps, x, y;
   var reset = function() {
      // reset state
      taps = 0;
      x = NaN;
      y = NaN;
   };
   reset();

   elem.addEventListener('touchstart', function(e) {
      var touch = e.changedTouches[0] || {}, oldX = x, oldY = y;

      taps++;
      x = +touch.pageX || +touch.clientX || +touch.screenX;
      y = +touch.pageY || +touch.clientY || +touch.screenY;

      // NaN will always be false
      if (Math.abs(oldX-x) < distance && Math.abs(oldY-y) < distance) {
         // fire dblclick event
         var e2 = document.createEvent('MouseEvents');
         if (e2.initMouseEvent) {
             e2.initMouseEvent(
                 'dblclick',
                 true,                   // dblclick bubbles
                 true,                   // dblclick cancelable
                 e.view,                 // copy view
                 taps,                   // click count
                 touch.screenX,          // copy coordinates
                 touch.screenY,
                 touch.clientX,
                 touch.clientY,
                 e.ctrlKey,              // copy key modifiers
                 e.altKey,
                 e.shiftKey,
                 e.metaKey,
                 e.button,               // copy button 0: left, 1: middle, 2: right
                 touch.target);          // copy target
         }
         elem.dispatchEvent(e2);
      }
      setTimeout(reset, speed);
   }, false);

   elem.addEventListener('touchmove', function(e) {
      reset();
   }, false);
};

(function(){

   if (typeof JSROOTPainter == 'object'){
      var e1 = new Error('JSROOTPainter is already defined');
      e1.source = 'JSROOTPainter.js';
      throw e1;
   }

   if (typeof d3 != 'object') {
      var e1 = new Error('This extension requires d3.js.js');
      e1.source = 'JSROOTPainter.js';
      throw e1;
   }

   // Initialize Custom colors
   var root_colors = generateAllColors();

   // Initialize colors of the default palette
   var default_palette = new Array();

   //Initialize ROOT markers
   var root_markers = new Array('fcircle','fcircle', 'fcross', 'dcross', 'ocircle',
      'gcross', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
      'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
      'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fsquare', 'ftriangle-up',
      'ftriangle-down', 'ocircle', 'osquare', 'otriangle-up', 'odiamond',
      'ocross', 'fstar', 'ostar', 'dcross', 'otriangle-down', 'fdiamond',
      'fcross');

   var root_fonts = new Array('Arial', 'Times New Roman',
      'bold Times New Roman', 'bold italic Times New Roman',
      'Arial', 'oblique Arial', 'bold Arial', 'bold oblique Arial',
      'Courier New', 'oblique Courier New', 'bold Courier New',
      'bold oblique Courier New', 'Symbol', 'Times New Roman',
      'Wingdings', 'Symbol');

   var root_line_styles = new Array("", "", "3, 3", "1, 2", "3, 4, 1, 4",
         "5, 3, 1, 3", "5, 3, 1, 3, 1, 3, 1, 3", "5, 5",
         "5, 3, 1, 3, 1, 3", "20, 5", "20, 10, 1, 10", "1, 1");

   JSROOTPainter = {};

   JSROOTPainter.version = '1.4 2012/02/24';

   /*
    * Helper functions
    */

   JSROOTPainter.padtoX = function(pad, x) {
      // Convert x from pad to X.
      if (pad['fLogx'] && x < 50) return Math.exp(2.302585092994 * x);
      return x;
   };

   JSROOTPainter.padtoY = function(pad, y) {
      // Convert y from pad to Y.
      if (pad['fLogy'] && y < 50) return Math.exp(2.302585092994 * y);
      return y;
   };

   JSROOTPainter.xtoPad = function(x, pad) {
      if (pad['fLogx']) {
         if (x > 0)
            x = JSROOTMath.log10(x);
         else
            x = pad['fUxmin'];
      }
      return x;
   };

   JSROOTPainter.ytoPad = function(y, pad) {
      if (pad['fLogy']) {
         if (y > 0)
            y = JSROOTMath.log10(y);
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
   JSROOTPainter.HLStoRGB = function(h, l, s) {
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
      return 'rgb('+Math.round(r * 255)+','+Math.round(g * 255)+','+Math.round(b * 255)+')';
   };

   JSROOTPainter.getMinMax = function(hist, what) {
      if (what == 'max' && hist['fMaximum'] != -1111) return hist['fMaximum'];
      if (what == 'min' && hist['fMinimum'] != -1111) return hist['fMinimum'];
      var bin, binx, biny, binz;
      var xfirst  = 1;;
      var xlast   = hist['fXaxis']['fNbins'];
      var yfirst  = 1;
      var ylast   = hist['fYaxis']['fNbins'];
      var zfirst  = 1;
      var zlast   = hist['fZaxis']['fNbins'];
      var maximum = Number.NEGATIVE_INFINITY;
      var minimum = Number.POSITIVE_INFINITY;
      var tmp_value;
      for (binz=zfirst;binz<=zlast;binz++) {
         for (biny=yfirst;biny<=ylast;biny++) {
            for (binx=xfirst;binx<=xlast;binx++) {
               //bin = hist.getBin(binx,biny,binz);
               //tmp_value = hist.getBinContent(bin);
               tmp_value = hist.getBinContent(binx, biny);
               if (tmp_value > maximum) maximum = tmp_value;
               if (tmp_value < minimum) minimum = tmp_value;
            }
         }
      }
      hist['fMaximum'] = maximum;
      hist['fMinimum'] = minimum;
      if (what == 'max') return maximum;
      if (what == 'min') return minimum;
   }

   JSROOTPainter.getValueColor = function(hist, zc, options) {
      var wmin = this.getMinMax(hist, 'min'),
          wmax = this.getMinMax(hist, 'max'),
          wlmin = wmin,
          wlmax = wmax,
          ndivz = hist['fContour'].length,
          scale = ndivz / (wlmax - wlmin);
      if (options && options['logz']) {
         if (wmin <= 0 && wmax > 0) wmin = Math.min(1.0, 0.001 * wmax);
         wlmin = Math.log(wmin)/Math.log(10);
         wlmax = Math.log(wmax)/Math.log(10);
      }
      if (default_palette.length == 0) {
         var saturation = 1,
             lightness = 0.5,
             maxHue = 280,
             minHue = 0,
             maxPretty = 50,
             hue;
         for (var i=0 ; i<maxPretty ; i++) {
            hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty))/360.0;
            var rgbval = this.HLStoRGB(hue, lightness, saturation);
            default_palette.push(rgbval);
         }
      }
      if (options && options['logz']) zc = Math.log(zc)/Math.log(10);
      if (zc < wlmin) zc = wlmin;
      var ncolors = default_palette.length
      var color = Math.round(0.01 + (zc - wlmin) * scale);
      var theColor = Math.round((color + 0.99) * ncolors / ndivz);
      var icol = theColor % ncolors;
      if (icol < 0) icol = 0;
      return default_palette[icol];
   };

   JSROOTPainter.getTimeOffset = function(axis) {

      var timeFormat = axis['fTimeFormat'];
      var i, timeoffset = 0;
      var idF = timeFormat.indexOf('%F');

      if (idF >= 0) {
         timeformat = timeFormat.substr(0, idF);
      } else {
         timeformat = timeFormat;
      }

      if (idF >= 0) {
         var lnF = timeFormat.length;
         var stringtimeoffset = timeFormat.substr(idF+2, lnF);
         for (i=0;i<3;++i) stringtimeoffset = stringtimeoffset.replace('-', '/');
         var stimeoffset = new Date(stringtimeoffset);
         timeoffset = stimeoffset.getTime();
         var ids = stringtimeoffset.indexOf('s');
         if (ids >= 0) {
            var lns = stringtimeoffset.length;
            var sdp = stringtimeoffset.substr(ids+1, lns);
            var dp = parseFloat(sdp);
            timeoffset += dp;
         }
      } else {
         timeoffset = 788918400000; // UTC time at 01/01/95
      }
      return timeoffset;
   };

   JSROOTPainter.formatExp = function(label) {
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
      for (j=0;j<size;++j) {
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

   JSROOTPainter.translateExp = function(str) {
      var i, j, lstr = str.match(/\^{[0-9]*}/gi);
      if (lstr != null) {
         var symbol = '';
         for (i=0;i<lstr.length;++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            var size = symbol.length;
            for (j=0;j<size;++j) {
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

   JSROOTPainter.translateLaTeX = function(string) {
      var str = string;
      str = this.translateExp(str);
      while (str.indexOf('^{o}') != -1)
         str = str.replace('^{o}', '\xBA');
      var lstr = str.match(/\#sqrt{(.*?)}/gi);
      if (lstr != null) {
         var symbol;
         for (i=0;i<lstr.length;++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('#sqrt{', '#sqrt');
            symbol = symbol.replace('}', '');
            str = str.replace(lstr[i], symbol);
         }
      }
      var lstr = str.match(/\_{(.*?)}/gi);
      if (lstr != null) {
         var symbol;
         for (i=0;i<lstr.length;++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('_{', ''); // &sub
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      }
      var lstr = str.match(/\^{(.*?)}/gi);
      if (lstr != null) {
         var symbol;
         for (i=0;i<lstr.length;++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      }
      while (str.indexOf('#/') != -1)
         str = str.replace('#/', symbols_map['#/']);
      for (x in symbols_map) {
         while (str.indexOf(x) != -1)
            str = str.replace(x, symbols_map[x]);
      }
      return str;
   };

   /**
    * Now the real drawing functions (using d3.js)
    */

   JSROOTPainter.addInteraction = function(vis, obj) {
      var width = vis.attr("width"), height = vis.attr("height");
      var e, origin, rect;
      doubleTap(vis[0][0]);

      if (typeof(vis['objects']) === 'undefined')
         vis['objects'] = new Array();
      vis['objects'].push(obj);

      function refresh() {
         if (vis.x_axis && vis.y_axis) {
            vis.select(".xaxis").call(vis.x_axis);
            vis.select(".yaxis").call(vis.y_axis);
         }
         vis.select(".xaxis").selectAll("text")
            .attr("font-size", vis['x_fsize'])
            .attr("font-family", vis['x_font']['name'])
            .attr("font-weight", vis['x_font']['weight'])
            .attr("font-style", vis['x_font']['style']);
         vis.select(".yaxis").selectAll("text")
            .attr("font-size", vis['y_fsize'])
            .attr("font-family", vis['y_font']['name'])
            .attr("font-weight", vis['y_font']['weight'])
            .attr("font-style", vis['y_font']['style']);
         for (var i=0;i<vis['objects'].length;++i) {
            vis['objects'][i].redraw();
         }
      };
      var zoom = d3.behavior.zoom().x(obj.x).y(obj.y).on("zoom", refresh());
      vis.on("touchstart", startRectSel);
      vis.on("mousedown", startRectSel);

      function startRectSel() {
         d3.event.preventDefault();
         vis.select("#zoom_rect").remove();
         e = this;
         var t = d3.event.changedTouches;
         origin = t ? d3.touches(e, t)[0] : d3.mouse(e);
         rect = vis.append("rect").attr("class", "zoom").attr("id", "zoom_rect");
         d3.select("body").classed("noselect", true);
         d3.select("body").style("-webkit-user-select", "none");
         origin[0] = Math.max(0, Math.min(width, origin[0]));
         origin[1] = Math.max(0, Math.min(height, origin[1]));
         vis.on("dblclick", unZoom);
         d3.select(window)
            .on("mousemove.zoomRect", moveRectSel)
            .on("mouseup.zoomRect", endRectSel, true);
         d3.select(window)
            .on("touchmove.zoomRect", moveRectSel)
            .on("touchend.zoomRect", endRectSel, true);
         d3.event.stopPropagation();
      };

      function unZoom() {
         d3.event.preventDefault();
         var xmin = vis['objects'][0]['x_min'],
             xmax = vis['objects'][0]['x_max'],
             ymin = vis['objects'][0]['y_min'],
             ymax = vis['objects'][0]['y_max'];
         for (var i=0;i<vis['objects'].length;++i) {
            zoom.x(vis['objects'][i].x.domain([xmin, xmax]))
                .y(vis['objects'][i].y.domain([ymin, ymax]));
         }
         refresh();
      };

      function moveRectSel() {
         d3.event.preventDefault();
         var t = d3.event.changedTouches;
         var m = t ? d3.touches(e, t)[0] : d3.mouse(e);
         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));
         rect.attr("x", Math.min(origin[0], m[0]))
             .attr("y", Math.min(origin[1], m[1]))
             .attr("width", Math.abs(m[0] - origin[0]))
             .attr("height", Math.abs(m[1] - origin[1]));
      };

      function endRectSel() {
         d3.event.preventDefault();
         d3.select(window).on("touchmove.zoomRect", null).on("touchend.zoomRect", null);
         d3.select(window).on("mousemove.zoomRect", null).on("mouseup.zoomRect", null);
         d3.select("body").classed("noselect", false);
         var t = d3.event.changedTouches;
         var m = t ? d3.touches(e, t)[0] : d3.mouse(e);
         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));
         if (Math.abs(m[0] - origin[0]) > 10 && Math.abs(m[1] - origin[1]) > 10) {
            var xmin = Math.min(vis['objects'][0].x.invert(origin[0]),
                                vis['objects'][0].x.invert(m[0])),
                xmax = Math.max(vis['objects'][0].x.invert(origin[0]),
                                vis['objects'][0].x.invert(m[0])),
                ymin = Math.min(vis['objects'][0].y.invert(origin[1]),
                                vis['objects'][0].y.invert(m[1])),
                ymax = Math.max(vis['objects'][0].y.invert(origin[1]),
                                vis['objects'][0].y.invert(m[1]));
            for (var i=0;i<vis['objects'].length;++i) {
               zoom.x(vis['objects'][i].x.domain([xmin, xmax]))
                   .y(vis['objects'][i].y.domain([ymin, ymax]));
            }
         }
         rect.remove();
         refresh();
         d3.select("body").style("-webkit-user-select", "auto");
      };
   };

   JSROOTPainter.createCanvas = function(element, idx) {
      var w = element.width(), h = w * 0.6666666;
      var render_to = '#histogram' + idx;
      d3.select(render_to).style("background-color", 'white');
      d3.select(render_to).style("width", "100%");

      var svg = d3.select(render_to).append("svg")
                  .attr("width", w)
                  .attr("height", h)
                  .style("background-color", 'white');
      return svg;
   };

   JSROOTPainter.createFrame = function(vis, pad, histo, frame) {
      var w = vis.attr("width"), h = vis.attr("height");
      var width = w, height = h;
      var lm = w*0.12, rm = w*0.05, tm = h*0.12, bm = h*0.12;
      if (histo && histo['fOption'] && histo['fOption'].toLowerCase() == 'colz')
         rm = w * 0.13;
      var framecolor = 'white', bordermode = 0,
          bordersize = 0, linecolor = 'black',//root_colors[0],
          linestyle = 0, linewidth = 1;
      if (histo && histo['_typename'] == 'JSROOTIO.TF1') {
         linecolor = 'black';
         linewidth = 1;
      }
      if (frame) {
         bordermode = frame['fBorderMode'];
         bordersize = frame['fBorderSize'];
         linecolor = root_colors[frame['fLineColor']];
         linestyle = frame['fLineStyle'];
         linewidth = frame['fLineWidth'];
         if (pad) {
            var xspan = width / Math.abs(pad['fX2'] - pad['fX1']);
            var yspan = height / Math.abs(pad['fY2'] - pad['fY1']);
            px1 = (frame['fX1'] - pad['fX1']) * xspan;
            py1 = (frame['fY1'] - pad['fY1']) * yspan;
            px2 = (frame['fX2'] - pad['fX1']) * xspan;
            py2 = (frame['fY2'] - pad['fY1']) * yspan;
            if (px1 < px2) {pxl = px1; pxt = px2;}
            else           {pxl = px2; pxt = px1;}
            if (py1 < py2) {pyl = py1; pyt = py2;}
            else           {pyl = py2; pyt = py1;}
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
            rm = (1.0 - frame['fX2']) * width;
            w -= (lm + rm);
            h -= (tm + bm);
         }
         framecolor = root_colors[frame['fFillColor']];
         if (frame['fFillStyle'] > 4000 && frame['fFillStyle'] < 4100)
            framecolor = 'none';
      }
      else {
         if (pad) {
            framecolor = root_colors[pad['fFrameFillColor']];
            if (pad['fFrameFillStyle'] > 4000 && pad['fFrameFillStyle'] < 4100)
               framecolor = 'none';
         }
         w -= (lm + rm);
         h -= (tm + bm);
      }
      if (typeof(framecolor) == 'undefined')
         framecolor = 'white';

      var hframe = vis.append("svg:g")
            .attr("x", lm)
            .attr("y", tm)
            .attr("width", w)
            .attr("height", h)
            .attr("transform", "translate(" + lm + "," + tm + ")");

      hframe.append("svg:rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", w)
            .attr("height", h)
            .attr("fill", framecolor)
            .style("stroke", linecolor)
            .style("stroke-width", linewidth);

      var svg_frame = hframe.append("svg")
            .attr("id", "svg_frame_" + (++frame_id))
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", w)
            .attr("height", h)
            .attr("viewBox", "0 0 "+w+" "+h);

      return {
         id: "#svg_frame_" + frame_id,
         frame: hframe,
         xmin: 0,
         xmax: 0,
         ymin: 0,
         ymax: 0
      };
   };

   JSROOTPainter.drawAxes = function(vis, histo, pad, xx, yy) {
      var w = vis.attr("width"), h = vis.attr("height"),
          logx = false, logy = false, logz = false;
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
      }
      var noexpx = histo['fXaxis'].TestBit(EAxisBits.kNoExponent);
      var noexpy = histo['fYaxis'].TestBit(EAxisBits.kNoExponent);
      var moreloglabelsx = histo['fXaxis'].TestBit(EAxisBits.kMoreLogLabels);
      var moreloglabelsy = histo['fYaxis'].TestBit(EAxisBits.kMoreLogLabels);

      if (histo['fXaxis']['fXmax'] < 100 && histo['fXaxis']['fXmax']/histo['fXaxis']['fXmin'] < 100) noexpx = true;
      if (histo['fYaxis']['fXmax'] < 100 && histo['fYaxis']['fXmax']/histo['fYaxis']['fXmin'] < 100) noexpy = true;

      var ndivx = histo['fXaxis']['fNdivisions'];
      var n1ax = ndivx%100;
      var n2ax = (ndivx%10000 - n1ax)/100;
      var n3ax = ndivx/10000;

      var ndivy = histo['fYaxis']['fNdivisions'];
      var n1ay = ndivy%100;
      var n2ay = (ndivy%10000 - n1ay)/100;
      var n3ay = ndivy/10000;

      /* X-axis label */
      var label = this.translateLaTeX(histo['fXaxis']['fTitle']);
      var xAxisTitleFontSize = histo['fXaxis']['fTitleSize'] * h;
      var xAxisLabelOffset = 3 + (histo['fXaxis']['fLabelOffset'] * h);
      var xAxisLabelFontSize = histo['fXaxis']['fLabelSize'] * h;
      var xAxisFontDetails = getFontDetails(root_fonts[Math.floor(histo['fXaxis']['fTitleFont']/10)]);

      vis.append("text")
         .attr("class", "X axis label")
         .attr("x", w)
         .attr("y", h)
         .attr("text-anchor", "end")
         .attr("font-family", xAxisFontDetails['name'])
         .attr("font-weight", xAxisFontDetails['weight'])
         .attr("font-style", xAxisFontDetails['style'])
         .attr("font-size", xAxisTitleFontSize)
         .text(label)
         .attr("transform", "translate(0," + (xAxisLabelFontSize + xAxisLabelOffset * histo['fXaxis']['fTitleOffset'] + xAxisTitleFontSize) + ")");

      /* Y-axis label */
      label = this.translateLaTeX(histo['fYaxis']['fTitle']);
      var yAxisTitleFontSize = histo['fYaxis']['fTitleSize'] * h;
      var yAxisLabelOffset = 3 + (histo['fYaxis']['fLabelOffset'] * w);
      var yAxisLabelFontSize = histo['fYaxis']['fLabelSize'] * h;
      var yAxisFontDetails = getFontDetails(root_fonts[Math.floor(histo['fYaxis']['fTitleFont'] /10)]);

      vis.append("text")
         .attr("class", "Y axis label")
         .attr("x", 0)
         .attr("y", -yAxisLabelFontSize - yAxisTitleFontSize - yAxisLabelOffset * histo['fYaxis']['fTitleOffset'])
         .attr("font-family", yAxisFontDetails['name'])
         .attr("font-size", yAxisTitleFontSize)
         .attr("font-weight", yAxisFontDetails['weight'])
         .attr("font-style", yAxisFontDetails['style'])
         .attr("fill", "black")
         .attr("text-anchor", "end")
         .text(label)
         .attr("transform", "rotate(270, 0, 0)");

      var xAxisColor = histo['fXaxis']['fAxisColor'];
      var xDivLength = histo['fXaxis']['fTickLength'] * h;
      var yAxisColor = histo['fYaxis']['fAxisColor'];
      var yDivLength = histo['fYaxis']['fTickLength'] * w;

      /*
       * Define the scales, according to the information from the pad
       */
      var dfx = d3.format(",.f"), dfy = d3.format(",.f");
      if (histo['fXaxis']['fTimeDisplay']) {
         if (n1ax > 8) n1ax = 8;
         var timeoffset = this.getTimeOffset(histo['fXaxis']);
         var range = histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin'];
         dfx = d3.time.format("%Mm%S");
         if (range>31536000)
            dfx = d3.time.format("%Y");
         else if (range>2419200)
            dfx = d3.time.format("%Y/%m");
         else if (range>86400)
            dfx = d3.time.format("%Y/%m/%d");
         else if (range>3600)
            dfx = d3.time.format("%Hh%Mm%S");
         else if (range>60)
            dfx = d3.time.format("%Hh%M");

         var x_axis = d3.svg.axis()
            .scale(xx)
            .orient("bottom")
            .tickPadding(xAxisLabelOffset)
            .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
            .tickFormat(function(d, i) {
               var datime = new Date(timeoffset + (d * 1000));
               return dfx(datime); })
            .ticks(n1ax);
      }
      else if (logx) {
         var x_axis = d3.svg.axis()
            .scale(xx)
            .orient("bottom")
            .tickPadding(xAxisLabelOffset)
            .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
            .tickFormat(function(d, i) { var val = parseFloat(d);
               var vlog = Math.abs(JSROOTMath.log10(val));
               if (moreloglabelsx) {
                  if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                     if (noexpx) return val.toFixed();
                     else return JSROOTPainter.formatExp(val.toExponential(0));
                  }
                  else return null;
               }
               else {
                  if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                     if (noexpx) return val.toFixed();
                     else return JSROOTPainter.formatExp(val.toExponential(0));
                  }
                  else return null;
               }
            });
      }
      else {
         var x_axis = d3.svg.axis()
            .scale(xx)
            .orient("bottom")
            .tickPadding(xAxisLabelOffset)
            .tickSubdivide(n2ax-1)
            .tickSize(-xDivLength, -xDivLength/2, -xDivLength/4)
            .tickFormat(function(d,i) {
               if (histo['fXaxis']['fTimeDisplay']) return dfx;
               return parseFloat(d.toPrecision(12));
            })
            .ticks(n1ax);
      }
      if (histo['fYaxis']['fTimeDisplay']) {
         if (n1ay > 8) n1ay = 8;
         var timeoffset = this.getTimeOffset(histo['fYaxis']);
         var range = histo['fYaxis']['fXmax'] - histo['fYaxis']['fXmin'];
         dfy = d3.time.format("%Mm%S");

         if (range>31536000)
            dfy = d3.time.format("%Y");
         else if (range>2419200)
            dfy = d3.time.format("%Y/%m");
         else if (range>86400)
            dfy = d3.time.format("%Y/%m/%d");
         else if (range>3600)
            dfy = d3.time.format("%Hh%Mm%S");
         else if (range>60)
            dfy = d3.time.format("%Hh%M");

         var y_axis = d3.svg.axis()
            .scale(yy)
            .orient("left")
            .tickPadding(yAxisLabelOffset)
            .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
            .tickFormat(function(d, i) {
               var datime = new Date(timeoffset + (d * 1000));
               return dfy(datime); })
            .ticks(n1ay);
      }
      else if (logy) {
         var y_axis = d3.svg.axis()
            .scale(yy)
            .orient("left")
            .tickPadding(yAxisLabelOffset)
            .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
            .tickFormat(function(d, i) { var val = parseFloat(d);
               var vlog = Math.abs(JSROOTMath.log10(val));
               if (moreloglabelsy) {
                  if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                     if (noexpy) return val.toFixed();
                     else return JSROOTPainter.formatExp(val.toExponential(0));
                  }
                  else return null;
               }
               else {
                  if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                     if (noexpy) return val.toFixed();
                     else return JSROOTPainter.formatExp(val.toExponential(0));
                  }
                  else return null;
            }});
      }
      else {
         var y_axis = d3.svg.axis()
            .scale(yy)
            .orient("left")
            .tickPadding(yAxisLabelOffset)
            .tickSubdivide(n2ay-1)
            .tickSize(-yDivLength, -yDivLength/2, -yDivLength/4)
            .tickFormat(function(d,i) {
               if (histo['fYaxis']['fTimeDisplay']) return dfy;
               return parseFloat(d.toPrecision(12));
            })
            .ticks(n1ay);
      }
      var xax = vis.append("svg:g")
         .attr("class", "xaxis")
         .attr("transform", "translate(0," + h + ")")
         .call(x_axis);

      var yax = vis.append("svg:g")
         .attr("class", "yaxis")
         .call(y_axis);


      var xAxisLabelFontDetails = getFontDetails(root_fonts[Math.floor(histo['fXaxis']['fLabelFont']/10)]);
      var yAxisLabelFontDetails = getFontDetails(root_fonts[Math.floor(histo['fXaxis']['fLabelFont']/10)]);

      xax.selectAll("text")
         .attr("font-family", xAxisLabelFontDetails['name'])
         .attr("font-size", xAxisLabelFontSize)
         .attr("font-weight", xAxisLabelFontDetails['weight'])
         .attr("font-style", xAxisLabelFontDetails['style']);
      yax.selectAll("text")
         .attr("font-family", yAxisLabelFontDetails['name'])
         .attr("font-size", yAxisLabelFontSize)
         .attr("font-weight", yAxisLabelFontDetails['weight'])
         .attr("font-style", yAxisLabelFontDetails['style']);

      vis['x_axis']  = x_axis;
      vis['y_axis']  = y_axis;
      vis['x_fsize'] = xAxisLabelFontSize;
      vis['y_fsize'] = yAxisLabelFontSize;
      vis['x_font'] = xAxisLabelFontDetails;
      vis['y_font'] = yAxisLabelFontDetails;
   };

   JSROOTPainter.drawCanvas = function(canvas, idx) {
      var render_to = '#histogram' + idx,
          w = $(render_to).width(),
          factor = w / Math.abs(canvas['fUtoPixel']),
          h = Math.abs(canvas['fVtoPixel']) * factor,
          fillcolor = root_colors[canvas['fFillColor']];
      if (canvas['fFillStyle'] > 4000 && canvas['fFillStyle'] < 4100)
         fillcolor = 'none';

      d3.select(render_to).style("background-color", fillcolor);
      d3.select(render_to).style("width", "100%");

      var svg = d3.select(render_to).append("svg")
          .attr("width", w)
          .attr("height", h)
          .style("background-color", fillcolor);

      JSROOTPainter.drawPrimitives(svg, canvas);
      return svg;
   };

   JSROOTPainter.drawErrors = function(svg, bins, histo, pad, x, y) {
      var w = svg.attr("width"), h = svg.attr("height");
      /* Add a panel for each data point */
      var opt = histo['fOption'].toLowerCase();
      var info_marker = getRootMarker(root_markers, histo['fMarkerStyle']);
      var shape = info_marker['shape'], filled = info_marker['toFill'],
          toRotate = info_marker['toRotate'], marker_size = histo['fMarkerSize'] * 32;

      if (histo['fMarkerStyle'] == 1) marker_size = 1;

      var marker = d3.svg.symbol()
          .type(d3.svg.symbolTypes[shape])
          .size(marker_size);

      function do_redraw() {

         JSROOTPainter.drawGrid(svg, histo, pad, x, y);

         var g_id = format_id(histo['fName']);
         svg.selectAll("#e_"+g_id).remove();
         var g = svg.append("svg:g")
            .attr("id", "e_"+g_id);

         var points = g.selectAll("markers")
            .data(histo.bins)
            .enter()
            .append("svg:path")
            .attr("class", "marker")
            .attr("transform", function(d) {
               return "translate(" + histo.x(d.x) + "," + histo.y(d.y) + ")"
            })
            .style("fill", root_colors[histo['fMarkerColor']])
            .style("stroke", root_colors[histo['fMarkerColor']])
            .attr("d", marker)
            .append("svg:title")
            .text(function(d) {
               return "x = " + d.x.toPrecision(5) + " \ny = " + d.y.toPrecision(5) +
                      " \nerror x = " + d.xerr.toPrecision(5) +
                      " \nerror y = " + d.yerr.toPrecision(5);
            });

         /* Add x-error indicators */
         g.selectAll("error_x")
            .data(histo.bins)
            .enter()
            .append("svg:line")
            .attr("x1", function(d) { return histo.x(d.x-d.xerr)} )
            .attr("y1", function(d) { return histo.y(d.y)} )
            .attr("x2", function(d) { return histo.x(d.x+d.xerr)} )
            .attr("y2", function(d) { return histo.y(d.y)} )
            .style("stroke", root_colors[histo['fLineColor']])
            .style("stroke-width", histo['fLineWidth']);

         if (opt.indexOf('e1') != -1) {
            g.selectAll("e1_x")
               .data(histo.bins)
               .enter()
               .append("svg:line")
               .attr("y1", function(d) { return histo.y(d.y)-3} )
               .attr("x1", function(d) { return histo.x(d.x-d.xerr)})
               .attr("y2", function(d) { return histo.y(d.y)+3})
               .attr("x2", function(d) { return histo.x(d.x-d.xerr)})
               .style("stroke", root_colors[histo['fLineColor']])
               .style("stroke-width", histo['fLineWidth']);
            g.selectAll("e1_x")
               .data(histo.bins)
               .enter()
               .append("svg:line")
               .attr("y1", function(d) { return histo.y(d.y)-3} )
               .attr("x1", function(d) { return histo.x(d.x+d.xerr) })
               .attr("y2", function(d) { return histo.y(d.y)+3})
               .attr("x2", function(d) { return histo.x(d.x+d.xerr) })
               .style("stroke", root_colors[histo['fLineColor']])
               .style("stroke-width", histo['fLineWidth']);
         }
         g.selectAll("error_y")
            .data(histo.bins)
            .enter()
            .append("svg:line")
            .attr("x1", function(d) { return histo.x(d.x)})
            .attr("y1", function(d) { return histo.y(d.y-d.yerr) })
            .attr("x2", function(d) { return histo.x(d.x)})
            .attr("y2", function(d) { return histo.y(d.y+d.yerr) })
            .style("stroke", root_colors[histo['fLineColor']])
            .style("stroke-width", histo['fLineWidth']);

         if (opt.indexOf('e1') != -1) {
            g.selectAll("e1_y")
               .data(histo.bins)
               .enter()
               .append("svg:line")
               .attr("x1", function(d) { return histo.x(d.x)-3})
               .attr("y1", function(d) { return histo.y(d.y-d.yerr) })
               .attr("x2", function(d) { return histo.x(d.x)+3})
               .attr("y2", function(d) { return histo.y(d.y-d.yerr) })
               .style("stroke", root_colors[histo['fLineColor']])
               .style("stroke-width", histo['fLineWidth']);
            g.selectAll("e1_y")
               .data(histo.bins)
               .enter()
               .append("svg:line")
               .attr("x1", function(d) { return histo.x(d.x)-3})
               .attr("y1", function(d) { return histo.y(d.y+d.yerr) })
               .attr("x2", function(d) { return histo.x(d.x)+3})
               .attr("y2", function(d) { return histo.y(d.y+d.yerr) })
               .style("stroke", root_colors[histo['fLineColor']])
               .style("stroke-width", histo['fLineWidth']);
         }
         g.selectAll("line")
            .append("svg:title")
            .text(function(d) {
               return "x = " + d.x.toPrecision(5) + " \ny = " + d.y.toPrecision(5) +
                      " \nerror x = " + d.xerr.toPrecision(5) +
                      " \nerror y = " + d.yerr.toPrecision(5);
            });
      };
      histo['redraw'] = do_redraw;
   };

   JSROOTPainter.drawFunction = function(vis, pad, func, hframe) {
      var i, logx = false, logy = false, logz = false,
          gridx = false, gridy = false, draw_all = true;
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      var fillcolor = root_colors[func['fFillColor']];
      var linecolor = root_colors[func['fLineColor']];
      if (func['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (func['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      var h, hmin = 1.0e32, hmax = -1.0e32;
      // use fNpfits instead of fNpx if possible (to use more points)
      if (func['fNpfits'] <= 103) func['fNpfits'] = 333;
      var nb_points = Math.max(func['fNpx'], func['fNpfits']);
      var binwidth = ((func['fXmax'] - func['fXmin']) / nb_points);
      for (var i=0;i<nb_points;++i) {
         h = func.evalPar(func['fXmin'] + (i * binwidth));
         if (h > hmax) hmax = h;
         if (h < hmin) hmin = h;
      }
      if (hmax > 0.0) hmax *= 1.1;
      if (hmin < 0.0) hmin *= 1.1;
      func['fYmin'] = hmin;
      func['fYmax'] = hmax;
      func['x_min'] = func['fXmin'];
      func['x_max'] = func['fXmax'];
      func['y_min'] = func['fYmin'];
      func['y_max'] = func['fYmax'];
      var bins = d3.range(nb_points).map(function(p) {
         return {
            x: func['fXmin'] + (p * binwidth),
            y: func.evalPar(func['fXmin'] + (p * binwidth))
         };
      });
      var ret = hframe != null ? hframe : this.createFrame(vis, pad, func, null);
      var frame = ret['frame'];
      var svg_frame = d3.select(ret['id']);
      var w = frame.attr("width"), h = frame.attr("height");
      if (hframe == null || (hframe['xmin'] < 1e-300 && hframe['xmax'] < 1e-300 &&
          hframe['ymin'] < 1e-300 && hframe['ymax'] < 1e-300)) {
         if (logx)
            var x = d3.scale.log().domain([func['fXmin'], func['fXmax']]).range([0, w]);
         else
            var x = d3.scale.linear().domain([func['fXmin'], func['fXmax']]).range([0, w]);
         if (logy)
            var y = d3.scale.log().domain([hmin, hmax]).range([h, 0]);
         else
            var y = d3.scale.linear().domain([hmin, hmax]).range([h, 0]);
      }
      else {
         draw_all = false;
         if (logx)
            var x = d3.scale.log().domain([hframe['xmin'], hframe['xmax']]).range([0, w]);
         else
            var x = d3.scale.linear().domain([hframe['xmin'], hframe['xmax']]).range([0, w]);
         if (logy)
            var y = d3.scale.log().domain([hframe['ymin'], hframe['ymax']]).range([h, 0]);
         else
            var y = d3.scale.linear().domain([hframe['ymin'], hframe['ymax']]).range([h, 0]);
      }
      func['x'] = x;
      func['y'] = y;
      func['bins'] = bins;

      function do_redraw() {

         var g_id = format_id(func['fName']);
         svg_frame.selectAll("#"+g_id).remove();

         var g = svg_frame.append("svg:g")
            .attr("id", g_id);

         var line = d3.svg.line()
            .x(function(d) { return func.x(d.x);})
            .y(function(d) { return func.y(d.y);})
            .interpolate('cardinal-open');

         g.append("svg:path")
            .attr("class", "line")
            .attr("d", line(bins))
            .style("stroke", linecolor)
            .style("stroke-width", func['fLineWidth'])
            .style("stroke-dasharray", root_line_styles[func['fLineStyle']])
            .style("fill", "none");
      };
      func['redraw'] = do_redraw;

      if (draw_all) {

         var mul = (w > h) ? h : w;
         var label_font_size = Math.round(0.035 * mul);

         /* X-axis */
         var x_axis = d3.svg.axis()
            .scale(x)
            .orient("bottom")
            .tickPadding(5)
            .tickSubdivide(5)
            .tickSize(-0.03 * h, -0.03 * h / 2, null)
            .tickFormat(function(d,i) { return parseFloat(d.toPrecision(12)); })
            .ticks(10);

         /* Y-axis minor ticks */
         var y_axis = d3.svg.axis()
            .scale(y)
            .orient("left")
            .tickPadding(5)
            .tickSubdivide(5)
            .tickSize(-0.03 * w, -0.03 * w / 2, null)
            .tickFormat(function(d,i) { return parseFloat(d.toPrecision(12)); })
            .ticks(10);

         var xax = frame.append("svg:g")
            .attr("class", "xaxis")
            .attr("transform", "translate(0," + h + ")")
            .call(x_axis);

         var yax = frame.append("svg:g")
            .attr("class", "yaxis")
            .call(y_axis);

         var font_size = Math.round(0.050 * h);

         if (!pad || typeof(pad) == 'undefined') {
            vis.append("text")
               .attr("class", "title")
               .attr("text-anchor", "middle")
               .attr("x", vis.attr("width")/2)
               .attr("y", 0.07 * vis.attr("height"))
               .attr("font-family", "Arial")
               .attr("font-size", font_size)
               .text(func['fTitle']);
/*
            // foreign html objects don't work on IE, and not properly on FF... :-(
            vis.append("foreignObject")
               .attr("y", 0.05 * vis.attr("height"))
               .attr("width", vis.attr("width"))
               .attr("height", 100)
            .append("xhtml:body")
               .style("font", "14px 'Helvetica'")
               .html("<h3><center>An HTML Foreign Object in SVG <font face='Symbol' color='green'>abCDEFG!</font></center></h3>");
*/
         }

         xax.selectAll("text").attr("font-size", label_font_size);
         yax.selectAll("text").attr("font-size", label_font_size);

         frame['x_axis']  = x_axis;
         frame['y_axis']  = y_axis;
         frame['x_fsize'] = label_font_size;
         frame['y_fsize'] = label_font_size;
         frame['x_font']  = {'weight' : "",'style' : "", 'name' : "arial" };
         frame['y_font']  = {'weight' : "",'style' : "", 'name' : "arial" };
      }
      this.addInteraction(frame, func);
      func_list.push(func);
   };

   JSROOTPainter.drawFunctions = function(vis, histo, pad, frame) {
      /* draw statistics box & other TPaveTexts */
      if (typeof(histo['fFunctions']) != 'undefined') {
         for (i=0; i<histo['fFunctions'].length; ++i) {
            if (histo['fFunctions'][i]['_typename'] == 'JSROOTIO.TPaveText' ||
                histo['fFunctions'][i]['_typename'] == 'JSROOTIO.TPaveStats') {
               if (histo['fFunctions'][i]['fX1NDC'] < 1.0 && histo['fFunctions'][i]['fY1NDC'] < 1.0 &&
                   histo['fFunctions'][i]['fX1NDC'] > 0.0 && histo['fFunctions'][i]['fY1NDC'] > 0.0) {
                  this.drawPaveText(vis, histo['fFunctions'][i]);
               }
            }
            if (histo['fFunctions'][i]['_typename'] == 'JSROOTIO.TF1') {
               if (!pad && !histo['fFunctions'][i].TestBit(kNotDraw)) {
                  if (histo['fFunctions'][i].TestBit(EStatusBits.kObjInCanvas)) {
                     if (typeof(histo['fFunctions'][i]['isDrawn']) == 'undefined' ||
                         histo['fFunctions'][i]['isDrawn'] == false)
                        this.drawFunction(vis, pad, histo['fFunctions'][i], frame);
                     histo['fFunctions'][i]['isDrawn'] = true;
                  }
               }
               else if (pad && histo['fFunctions'][i].TestBit(EStatusBits.kObjInCanvas)) {
                  if (typeof(histo['fFunctions'][i]['isDrawn']) == 'undefined' ||
                      histo['fFunctions'][i]['isDrawn'] == false)
                     this.drawFunction(vis, pad, histo['fFunctions'][i], frame);
                  histo['fFunctions'][i]['isDrawn'] = true;
               }
            }
         }
      }
   };

   JSROOTPainter.drawGraph = function(vis, pad, graph, hframe) {
      var scalex = 1, scaley = 1, logx = false, logy = false, logz = false,
          gridx = false, gridy = false, draw_all = true;
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      // check for axis scale format, and convert if required
      var xaxis_type = logx ? 'logarithmic' : 'linear';
      if (graph['fHistogram']['fXaxis']['fTimeDisplay']) {
         xaxis_type = 'datetime';
      }
      var yaxis_type = logy ? 'logarithmic' : 'linear';
      if (graph['fHistogram']['fYaxis']['fTimeDisplay']) {
         yaxis_type = 'datetime';
      }
      var seriesType = 'line';
      var showMarker = true;
      if (graph['fOption'] && typeof(graph['fOption']) != 'undefined') {
         var opt = graph['fOption'].toLowerCase();
         if (opt.indexOf('p') == -1 && opt.indexOf('*') == -1)
            showMarker = false;
         if (opt.indexOf('l') == -1 && opt.indexOf('c') == -1)
            seriesType = 'scatter';
      }
      var bins = d3.range(graph['fNpoints']).map(function(p) {
         return {
            x: graph['fX'][p] * scalex,
            y: graph['fY'][p] * scaley
         };
      });
      var ret = hframe != null ? hframe : this.createFrame(vis, pad, graph['fHistogram'], null);
      var frame = ret['frame'];
      var svg_frame = d3.select(ret['id']);
      var w = frame.attr("width"), h = frame.attr("height");
      if (hframe == null || (hframe['xmin'] < 1e-300 && hframe['xmax'] < 1e-300 &&
          hframe['ymin'] < 1e-300 && hframe['ymax'] < 1e-300)) {
         if (logx)
            var x = d3.scale.log().domain([graph['fHistogram']['fXaxis']['fXmin'],
                                 graph['fHistogram']['fXaxis']['fXmax']]).range([0, w]);
         else
            var x = d3.scale.linear().domain([graph['fHistogram']['fXaxis']['fXmin'],
                                    graph['fHistogram']['fXaxis']['fXmax']]).range([0, w]);
         if (logy)
            var y = d3.scale.log().domain([graph['fHistogram']['fYaxis']['fXmin'],
                                 graph['fHistogram']['fYaxis']['fXmax']]).range([h, 0]);
         else
            var y = d3.scale.linear().domain([graph['fHistogram']['fYaxis']['fXmin'],
                                    graph['fHistogram']['fYaxis']['fXmax']]).range([h, 0]);
         graph['x_min'] = graph['fHistogram']['fXaxis']['fXmin'];
         graph['x_max'] = graph['fHistogram']['fXaxis']['fXmax'];
         graph['y_min'] = graph['fHistogram']['fYaxis']['fXmin'];
         graph['y_max'] = graph['fHistogram']['fYaxis']['fXmax'];
      }
      else {
         draw_all = false;
         if (logx)
            var x = d3.scale.log().domain([hframe['xmin'], hframe['xmax']]).range([0, w]);
         else
            var x = d3.scale.linear().domain([hframe['xmin'], hframe['xmax']]).range([0, w]);
         if (logy)
            var y = d3.scale.log().domain([hframe['ymin'], hframe['ymax']]).range([h, 0]);
         else
            var y = d3.scale.linear().domain([hframe['ymin'], hframe['ymax']]).range([h, 0]);

         graph['x_min'] = hframe['xmin'];
         graph['x_max'] = hframe['xmax'];
         graph['y_min'] = hframe['ymin'];
         graph['y_max'] = hframe['ymax'];
      }
      graph['x'] = x;
      graph['y'] = y;
      graph['bins'] = bins;

      // exclusion graphs
      var lw = graph['fLineWidth'];
      var ec, ff, exclusionGraph = false;
      if (graph['fLineWidth'] > 99) {
         var glw = graph['fLineWidth'];
         // negative value means another side of the line...
         if (glw > 32767) {
            glw = 65536 - glw;
         }
         exclusionGraph = true;
         //lw = (glw/100)*0.005; // ? from TGraphPainter src...
         lw = glw % 100; // line width
         //ff = (glw - lw) / 100; // filled width
         ff = (glw - lw) * 0.05; // filled width
         ec = root_colors[graph['fFillColor']];
         ec = ec.replace('rgb', 'rgba');
         ec = ec.replace(')', ',0.30)');
      }

      function do_redraw() {

         if (draw_all)
            JSROOTPainter.drawGrid(frame, graph['fHistogram'], pad, x, y);

         var g_id = format_id(graph['fName']);
         svg_frame.selectAll("#"+g_id).remove();
         var g = svg_frame.append("svg:g")
            .attr("id", g_id);

         if (seriesType == 'line') {
            /* contour lines only */
            var line = d3.svg.line()
               .x(function(d) { return graph.x(d.x);})
               .y(function(d) { return graph.y(d.y);});

            g.append("svg:path")
               .attr("class", "line")
               .attr("d", line(bins))
               .style("stroke", root_colors[graph['fLineColor']])
               .style("stroke-width", lw)
               .style("stroke-dasharray", root_line_styles[graph['fLineStyle']])
               .style("fill", "none");
         }
         if (exclusionGraph) {
            showMarker = false;
            g.append("svg:path")
               .attr("class", "line")
               .attr("d", line(bins))
               .style("stroke", ec)
               .style("stroke-width", ff)
               .style("fill", "none");
         }
         if (showMarker) {
            var filled = false;
            if ((graph['fMarkerStyle'] == 8) ||
                (graph['fMarkerStyle'] > 19 && graph['fMarkerStyle'] < 24) ||
                (graph['fMarkerStyle'] == 29))
               filled = true;

            var info_marker = getRootMarker(root_markers, graph['fMarkerStyle']);

            var shape = info_marker['shape'];
            var filled = info_marker['toFill'];
            var toRotate = info_marker['toRotate'];
            var markerSize = graph['fMarkerSize'];

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
                              .size(markerSize * 65);
                  break;
            }
            g.selectAll("markers")
               .data(bins)
               .enter()
               .append("svg:path")
               .attr("class", "marker")
               .attr("transform", function(d) {return "translate(" + graph.x(d.x) + "," + graph.y(d.y) + ")"})
               .style("fill", filled ? root_colors[graph['fMarkerColor']] : "none")
               .style("stroke", root_colors[graph['fMarkerColor']])
               .attr("d", marker)
               .append("svg:title")
               .text(function(d) { return "x = " + d.x.toPrecision(5) + " \ny = " + d.y.toPrecision(5); });
         }
      };
      graph['redraw'] = do_redraw;

      if (draw_all) {
         this.drawAxes(frame, graph['fHistogram'], pad, x, y);
         this.drawTitle(vis, graph['fHistogram'], pad);
      }
      this.addInteraction(frame, graph);
      this.drawFunctions(vis, graph['fHistogram'], pad, ret);
   };

   JSROOTPainter.drawGrid = function(vis, histo, pad, x, y) {
      var gridx = false, gridy = false;
      if (pad && typeof(pad) != 'undefined') {
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      var ndivx = histo['fXaxis']['fNdivisions'];
      var n1ax = ndivx%100;
      var n2ax = (ndivx%10000 - n1ax)/100;
      var n3ax = ndivx/10000;
      var nn3x = Math.max(n3ax,1);
      var nn2x = Math.max(n2ax,1)*nn3x;
      var nn1x = Math.max(n1ax,1)*nn2x;

      var ndivy = histo['fYaxis']['fNdivisions'];
      var n1ay = ndivy%100;
      var n2ay = (ndivy%10000 - n1ay)/100;
      var n3ay = ndivy/10000;
      var nn3y = Math.max(n3ay,1);
      var nn2y = Math.max(n2ay,1)*nn3y;
      var nn1y = Math.max(n1ay,1)*nn2y;

      vis.selectAll(".gridLine").remove();

      /* add a grid on x axis, if the option is set */
      if (gridx) {
         vis.selectAll("gridLine")
            .data(x.ticks(n1ax))
            .enter()
            .append("svg:line")
            .attr("class", "gridLine")
            .attr("x1", x)
            .attr("y1", vis.attr("height"))
            .attr("x2", x)
            .attr("y2", 0)
            .style("stroke", "black")
            .style("stroke-width", histo['fLineWidth'])
            .style("stroke-dasharray", root_line_styles[11]);
      }

      /* add a grid on y axis, if the option is set */
      if (gridy) {
         vis.selectAll("gridLine")
            .data(y.ticks(n1ay))
            .enter()
            .append("svg:line")
            .attr("class", "gridLine")
            .attr("x1", 0)
            .attr("y1", y)
            .attr("x2", vis.attr("width"))
            .attr("y2", y)
            .style("stroke", "black")
            .style("stroke-width", histo['fLineWidth'])
            .style("stroke-dasharray", root_line_styles[11]);
      }
   };

   JSROOTPainter.drawHistogram1D = function(vis, pad, histo, hframe) {
      var i, logx = false, logy = false, logz = false, gridx = false, gridy = false;
      var opt = histo['fOption'].toLowerCase();
      var draw_all = false;
      if (hframe == null || (hframe['xmin'] < 1e-300 && hframe['xmax'] < 1e-300 &&
          hframe['ymin'] < 1e-300 && hframe['ymax'] < 1e-300)) {
         draw_all = true;
      }
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      var fillcolor = root_colors[histo['fFillColor']];
      var linecolor = root_colors[histo['fLineColor']];
      if (histo['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (histo['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      var hmin = 1.0e32, hmax = -1.0e32;
      for (i=0;i<histo['fXaxis']['fNbins'];++i) {
         if (histo['fArray'][i+1] < hmin) hmin = histo['fArray'][i+1];
         if (histo['fArray'][i+1] > hmax) hmax = histo['fArray'][i+1];
      }
      var mul = (hmin < 0) ? 1.1 : 1.0;
      if (hmin < 1e-300 && hmax < 1e-300) {
         var ymin = histo['fYaxis']['fXmin'], ymax = histo['fYaxis']['fXmax'];
         if (histo['fMinimum'] != -1111) ymin = histo['fMinimum'];
         if (histo['fMaximum'] != -1111) ymax = histo['fMaximum'];

         // special case used for drawing multiple graphs in the same frame
         var ret = hframe != null ? hframe : this.createFrame(vis, pad, histo, null);
         var frame = ret['frame'];
         var svg_frame = d3.select(ret['id']);
         var w = frame.attr("width"), h = frame.attr("height");
         if (logx)
            var x = d3.scale.log().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
         else
            var x = d3.scale.linear().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
         if (logy)
            var y = d3.scale.log().domain([ymin, ymax]).range([h, 0]);
         else
            var y = d3.scale.linear().domain([ymin, ymax]).range([h, 0]);

//         frame['x'] = x;
//         frame['y'] = y;
         // avoid this!
         histo['fYaxis']['fXmin'] = ymin;
         histo['fYaxis']['fXmax'] = ymax;

         histo['x'] = x;
         histo['y'] = y;
         histo['x_min'] = histo['fXaxis']['fXmin'];
         histo['x_max'] = histo['fXaxis']['fXmax'];
         histo['y_min'] = ymin;
         histo['y_max'] = ymax;
         histo['redraw'] = function() {
            JSROOTPainter.drawGrid(frame, histo, pad, x, y);
         };
         this.drawAxes(frame, histo, pad, x, y);
         this.drawTitle(vis, histo, pad);
         this.addInteraction(frame, histo);
         this.drawFunctions(vis, histo, pad, ret);
         return {
            frame: frame,
            xmin: histo['fXaxis']['fXmin'],
            xmax: histo['fXaxis']['fXmax'],
            ymin: ymin,
            ymax: ymax
         };
      }
      if (histo['fMinimum'] != -1111) hmin = histo['fMinimum'];
      if (histo['fMaximum'] != -1111) hmax = histo['fMaximum'];
      histo['fYaxis']['fXmin'] = hmin;
      histo['fYaxis']['fXmax'] = hmax;
      var binwidth = ((histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin']) / histo['fXaxis']['fNbins']);
      var bins = d3.range(histo['fXaxis']['fNbins']).map(function(p) {
         var offset = (opt.indexOf('e') != -1) ? (p * binwidth) - (binwidth / 2.0) : (p * binwidth);
         return {
            x:  histo['fXaxis']['fXmin'] + offset,
            y:  histo['fArray'][p],
            xerr: binwidth / 2.0,
            yerr: histo.getBinError(p)
         };
      });
      var ret = hframe != null ? hframe : this.createFrame(vis, pad, histo, null);
      var frame = ret['frame'];
      var svg_frame = d3.select(ret['id']);
      var w = frame.attr("width"), h = frame.attr("height");
      if (logx)
         var x = d3.scale.log().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      else
         var x = d3.scale.linear().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      if (logy)
         var y = d3.scale.log().domain([mul * d3.min(bins, function(d) { return d.y; }),
                      1.1 * d3.max(bins, function(d) { return d.y; })]).range([h, 0]);
      else
         var y = d3.scale.linear().domain([mul * d3.min(bins, function(d) { return d.y; }),
                      1.1 * d3.max(bins, function(d) { return d.y; })]).range([h, 0]);

      ret['ymin'] = hmin;
      ret['ymax'] = hmax;
      if (histo['fXaxis'].TestBit(EAxisBits.kAxisRange)) {
         ret['xmin'] = histo.getBinLowEdge(histo['fXaxis']['fFirst']);
         ret['xmax'] = histo.getBinUpEdge(histo['fXaxis']['fLast']);
         x.domain([ret['xmin'],ret['xmax']]);
      }
      histo['x_min'] = histo['fXaxis']['fXmin'];
      histo['x_max'] = histo['fXaxis']['fXmax'];
      histo['y_min'] = hmin * mul;
      histo['y_max'] = hmax * 1.1;

      histo['x'] = x;
      histo['y'] = y;
      histo['bins'] = bins;

      if (opt.indexOf('e') != -1) {
         this.drawErrors(svg_frame, bins, histo, pad, x, y);
      }
      else {
         function do_redraw() {

            if (draw_all)
               JSROOTPainter.drawGrid(frame, histo, pad, x, y);

            var g_id = format_id(histo['fName']);
            svg_frame.selectAll("#"+g_id).remove();
            var g = svg_frame.append("svg:g")
               .attr("id", g_id);

            if ((histo['fFillStyle'] < 4000 || histo['fFillStyle'] > 4100) && histo['fFillColor'] != 0) {
               /* histogram filling */
               var area = d3.svg.area()
                  .x(function(d) { return histo.x(d.x);})
                  .y0(function(d) { return histo.y(0);})
                  .y1(function(d) { return histo.y(d.y);})
                  .interpolate("step-before")

               g.append("svg:path")
                  .attr("class", "area")
                  .attr("d", area(bins))
                  .style("stroke", linecolor)
                  .style("stroke-width", histo['fLineWidth'])
                  .style("fill", fillcolor)
                  .style("antialias", "false");
            }
            else {
               /* histogram contour lines only */
               var line = d3.svg.line()
                  .x(function(d) { return histo.x(d.x);})
                  .y(function(d) { return histo.y(d.y);})
                  .interpolate("step-before");

               g.append("svg:path")
                  .attr("class", "line")
                  .attr("d", line(bins))
                  .style("stroke", linecolor)
                  .style("stroke-width", histo['fLineWidth'])
                  .style("fill", "none")
                  .style("antialias", "false");
            }
         };
         histo['redraw'] = do_redraw;
      }
      if (draw_all)
         this.drawAxes(frame, histo, pad, x, y);
      this.drawTitle(vis, histo, pad);
      this.addInteraction(frame, histo);
      this.drawFunctions(vis, histo, pad, ret);
      return null;
   };

   JSROOTPainter.drawHistogram2D = function(vis, pad, histo, hframe) {
      var i, logx = false, logy = false, logz = false, gridx = false, gridy = false;
      var opt = histo['fOption'].toLowerCase();
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      var fillcolor = root_colors[histo['fFillColor']];
      var linecolor = root_colors[histo['fLineColor']];
      if (histo['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (histo['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      var nbinsx = histo['fXaxis']['fNbins'];
      var nbinsy = histo['fYaxis']['fNbins'];
      var scalex = (histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin']) /
                    histo['fXaxis']['fNbins'];
      var scaley = (histo['fYaxis']['fXmax'] - histo['fYaxis']['fXmin']) /
                    histo['fYaxis']['fNbins'];
      var maxbin = -1e32, minbin = 1e32;
      maxbin = d3.max(histo['fArray']);
      minbin = d3.min(histo['fArray']);
      var bins = new Array();
      for (i=0; i<nbinsx; ++i) {
         for (var j=0; j<nbinsy; ++j) {
            var bin_content = histo.getBinContent(i, j);
            if (bin_content > minbin) {
               var point = {
                  x:histo['fXaxis']['fXmin'] + (i*scalex),
                  y:histo['fYaxis']['fXmin'] + (j*scaley),
                  z:bin_content
               };
               bins.push(point);
            }
         }
      }
      var ret = hframe != null ? hframe : this.createFrame(vis, pad, histo, null);
      var frame = ret['frame'];
      var svg_frame = d3.select(ret['id']);
      var w = frame.attr("width"), h = frame.attr("height");
      if (logx)
         var x = d3.scale.log().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      else
         var x = d3.scale.linear().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      if (logy)
         var y = d3.scale.log().domain([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]).range([h, 0]);
      else
         var y = d3.scale.linear().domain([histo['fYaxis']['fXmin'], histo['fYaxis']['fXmax']]).range([h, 0]);

      var c = d3.scale.linear().domain([minbin, maxbin]).range(['red', 'blue']);

      histo['x_min'] = histo['fXaxis']['fXmin'];
      histo['x_max'] = histo['fXaxis']['fXmax'];
      histo['y_min'] = histo['fYaxis']['fXmin'];
      histo['y_max'] = histo['fYaxis']['fXmax'];

      histo['x'] = x;
      histo['y'] = y;
      histo['bins'] = bins;

      function do_redraw() {

         JSROOTPainter.drawGrid(frame, histo, pad, x, y);

         var g_id = format_id(histo['fName']);
         svg_frame.selectAll("#"+g_id).remove();
         var g = svg_frame.append("svg:g")
            .attr("id", g_id);

         var constx = (w / histo['fXaxis']['fNbins']) / maxbin;
         var consty = (h / histo['fYaxis']['fNbins']) / maxbin;
         var xdom = histo.x.domain();
         var ydom = histo.y.domain();
         var xfactor = Math.abs(histo['fXaxis']['fXmax']-histo['fXaxis']['fXmin']) / Math.abs(xdom[1]-xdom[0]);
         var yfactor = Math.abs(histo['fYaxis']['fXmax']-histo['fYaxis']['fXmin']) / Math.abs(ydom[1]-ydom[0]);

         g.selectAll("bins")
            .data(histo.bins)
            .enter()
            .append("svg:rect")
            .attr("class", "bins")
            .attr("x", function(d) { return histo.x(d.x) + (scalex/2) - (d.z * constx/2);})
            .attr("y", function(d) { return histo.y(d.y) + (scaley/2) - (d.z * consty/2);})
            .attr("width", function(d) {
               switch (opt) {
                  case 'colz':
                  case 'col':
                     return (w / histo['fXaxis']['fNbins']) * xfactor;
                  default:
                     return d.z * ((w / histo['fXaxis']['fNbins']) / maxbin) * xfactor;
               }
            })
            .attr("height", function(d) {
               switch (opt) {
                  case 'colz':
                  case 'col':
                     return (h / histo['fYaxis']['fNbins']) * yfactor;
                  default:
                     return d.z * ((h / histo['fYaxis']['fNbins']) / maxbin) * yfactor;
               }
            })
            .style("stroke", function(d) {
               switch (opt) {
                  case 'colz':
                  case 'col':
                     return JSROOTPainter.getValueColor(histo, d.z, pad);
                  default:
                     return "black";
               }
            })
            .style("fill", function(d) {
               switch (opt) {
                  case 'colz':
                  case 'col':
                     return JSROOTPainter.getValueColor(histo, d.z, pad);
                  default:
                     return "none";
               }
            });
         g.selectAll("rect")
            .append("svg:title")
            .text(function(d) { return "x = " + d.x.toPrecision(5) + " \ny = " + d.y.toPrecision(5) + " \nentries = " + d.z; });
      };
      histo['redraw'] = do_redraw;

      if (opt.indexOf('colz') != -1) {
         // just to initialize the default palette
         this.getValueColor(histo, 0, pad);
         for (i=0; i<histo['fFunctions'].length; ++i) {
            if (histo['fFunctions'][i]['_typename'] == 'JSROOTIO.TPaletteAxis')
               this.drawPaletteAxis(vis, histo['fFunctions'][i], minbin, maxbin);
         }
      }

      this.drawAxes(frame, histo, pad, x, y);
      this.drawTitle(vis, histo, pad);
      this.addInteraction(frame, histo);
      this.drawFunctions(vis, histo, pad, ret);
      if (!pad || typeof(pad) == 'undefined')
         this.drawStat(vis, histo);
   };

   JSROOTPainter.drawLegend = function(vis, pad, pave) {
      var x = pave['fX1NDC'] * vis.attr("width")
      var y = vis.attr("height") - pave['fY1NDC'] * vis.attr("height");
      var w = (pave['fX2NDC'] - pave['fX1NDC']) * vis.attr("width");
      var h = (pave['fY2NDC'] - pave['fY1NDC']) * vis.attr("height");
      y -= h;
      var fillcolor = root_colors[pave['fFillColor']];
      var lcolor = root_colors[pave['fLineColor']];
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

      var tcolor = root_colors[pave['fTextColor']];
      var tpos_x = pave['fMargin'] * w;
      var line_length = (0.8 * pave['fMargin']) * w;
      var pos_x = (tpos_x - line_length) / 2;
      var nlines = pave['fPrimitives'].length;
      var font_size = Math.round(h / (nlines * 1.5));
      //var font_size = Math.round(pave['fTextSize'] * vis.height());
      var fontDetails = getFontDetails(root_fonts[Math.floor(pave['fTextFont']/10)]);

      for (var i=0; i<nlines; ++i) {
         var leg = pave['fPrimitives'][i];
         var string = leg['fLabel'];
         var pos_y = ((i+1) * (font_size * 1.4)) - (font_size/3);
         var tpos_y = (i+1) * (font_size * 1.4);
         var line_color = root_colors[leg['fLineColor']];
         var line_style = leg['fLineStyle'];
         var line_width = leg['fLineWidth'];
         var line_style = root_line_styles[leg['fLineStyle']];
         var opt = leg['fOption'].toLowerCase();
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
         if (opt.indexOf('f') != -1) {
         }
         // Draw line
         if (opt.indexOf('l') != -1 || opt.indexOf('f') != -1) {
            // line total length (in x) is margin*0.8
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
         if (opt.indexOf('e') != -1 && (opt.indexOf('l') == -1 || opt.indexOf('f') != -1)) {
         }
         // Draw Polymarker
         if (opt.indexOf('p') != -1) {
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

   JSROOTPainter.drawMultiGraph = function(vis, pad, mgraph, hframe) {
      var histo = mgraph['fHistogram'];
      var graphs = mgraph['fGraphs'];
      var scalex = 1, scaley = 1;
      var logx = false, logy = false, logz = false, gridx = false, gridy = false;
      var draw_all = true;
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx']; logy = pad['fLogy']; logz = pad['fLogz'];
         gridx = pad['fGridx']; gridy = pad['fGridy'];
      }
      // check for axis scale format, and convert if required
      var xaxis_type = logx ? 'logarithmic' : 'linear';
      if (histo['fXaxis']['fTimeDisplay']) {
         xaxis_type = 'datetime';
      }
      var yaxis_type = logy ? 'logarithmic' : 'linear';
      if (histo['fYaxis']['fTimeDisplay']) {
         yaxis_type = 'datetime';
      }
      var frame;
      if (hframe) frame = hframe['frame'];
      else {
         hframe = this.createFrame(vis, pad, histo, null);
         frame = hframe['frame'];
      }
      this.drawHistogram1D(vis, pad, histo, hframe);
      hframe['xmin'] = histo['fXaxis']['fXmin'];
      hframe['xmax'] = histo['fXaxis']['fXmax'];
      hframe['ymin'] = histo['fYaxis']['fXmin'];
      hframe['ymax'] = histo['fYaxis']['fXmax'];
      for (var i=0; i<graphs.length; ++i) {
         graphs[i]['fName'] += i;
         this.drawGraph(vis, pad, graphs[i], hframe);
      }
   };

   JSROOTPainter.drawObject = function(obj, idx) {
      var i, svg = null;
      function draw(init) {

         var render_to = '#histogram' + idx;
         $(render_to).empty();

         for (i=0; i<func_list.length; ++i) {
            func_list[i]['isDrawn'] = false;
         }
         if (obj['_typename'].match(/\bTCanvas/)) {
            svg = JSROOTPainter.drawCanvas(obj, idx);
            if (init == true)
               window.setTimeout(function() { $(render_to)[0].scrollIntoView(); }, 50);
            return;
         }
         svg = JSROOTPainter.createCanvas($(render_to), idx);
         if (svg == null) return false;
         if (obj['_typename'].match(/\bTH1/)) {
            JSROOTPainter.drawHistogram1D(svg, null, obj, null);
         }
         else if (obj['_typename'].match(/\bTH2/)) {
            JSROOTPainter.drawHistogram2D(svg, null, obj, null);
         }
         else if (obj['_typename'].match(/\bTProfile/)) {
            JSROOTPainter.drawProfile(svg, null, obj, null);
         }
         else if (obj['_typename'] == 'JSROOTIO.TF1') {
            JSROOTPainter.drawFunction(svg, null, obj, null);
         }
         else if (obj['_typename'] == 'JSROOTIO.TGraph') {
            JSROOTPainter.drawGraph(svg, null, obj, null);
         }
         else if (obj['_typename'] == 'JSROOTIO.TMultiGraph') {
            JSROOTPainter.drawMultiGraph(svg, null, obj, null);
         }
         if (init == true)
            window.setTimeout(function() { $(render_to)[0].scrollIntoView(); }, 50);
      };
      //A better idom for binding with resize is to debounce
      var debounce = function(fn, timeout) {
         var timeoutID = -1;
         return function() {
            if (timeoutID > -1) {
               window.clearTimeout(timeoutID);
            }
            timeoutID = window.setTimeout(fn, timeout);
         }
      };
      var debounced_draw = debounce(function() { draw(false); }, 100);
      $(window).resize(debounced_draw);
      draw(true);
   };

   JSROOTPainter.drawPad = function(vis, pad) {
      var width = vis.attr("width"), height = vis.attr("height");
      var x = pad['fAbsXlowNDC'] * width;
      var y = height - pad['fAbsYlowNDC'] * height;
      var w = pad['fAbsWNDC'] * width;
      var h = pad['fAbsHNDC'] * height;
      y -= h;

      var fillcolor = root_colors[pad['fFillColor']];
      if (pad['fFillStyle'] > 4000 && pad['fFillStyle'] < 4100)
         fillcolor = 'none';

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
         .style("stroke-width", pad['fLineWidth'])
         .style("stroke", root_colors[pad['fLineColor']]);

      this.drawPrimitives(new_pad, pad);
      return new_pad;
   };

   JSROOTPainter.drawPaletteAxis = function(vis, palette, minbin, maxbin) {
      var width = vis.attr("width");
      var height = vis.attr("height");

      var pos_x = palette['fX1NDC'] * width;
      var pos_y = height - palette['fY1NDC'] * height;
      var s_width = Math.abs(palette['fX2NDC'] - palette['fX1NDC']) * width;
      var s_height = Math.abs(palette['fY2NDC'] - palette['fY1NDC']) * height;
      pos_y -= s_height;

      /*
       * Draw palette pad
       */
      var pal = vis.append("svg:g")
          .attr("height", s_height)
          .attr("width", s_width)
          .attr("transform", "translate(" + pos_x + ", " + pos_y + ")");

      var axis = palette['fAxis'];

      /*
       * Draw the default palette
       */
      var rectHeight = s_height / default_palette.length;
      pal.selectAll("colorRect")
         .data(default_palette)
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
      var nbr1 = 8;//Math.max((ndiv % 10000) % 100, 1);
      var nbr2 = 0;//Math.max(Math.round((ndiv % 10000) / 100), 1);
      var nbr3 = 0;//Math.max(Math.round(ndiv / 10000), 1);

      var z = d3.scale.linear().clamp(true)
               .domain([minbin, maxbin])
               .range([s_height, 0])
               .nice();

      var axisOffset = axis['fLabelOffset'] * width;
      var tickSize = axis['fTickSize'] * width;
      var z_axis = d3.svg.axis()
                  .scale(z)
                  .orient("right")
                  .tickSubdivide(nbr2)
                  .tickPadding(axisOffset)
                  .tickSize(-tickSize, -tickSize/2, 0)
                  .ticks(nbr1);

       var zax = pal.append("svg:g")
               .attr("class", "zaxis")
               .attr("transform", "translate(" + s_width + ", 0)")
               .call(z_axis);

      var axisFontDetails = getFontDetails(root_fonts[Math.floor(axis['fLabelFont'] /10)]);
      var axisLabelFontSize = axis['fLabelSize'] * height;
      zax.selectAll("text")
         .attr("font-size", axisLabelFontSize)
         .attr("font-weight", axisFontDetails['weight'])
         .attr("font-style", axisFontDetails['style'])
         .attr("font-family", axisFontDetails['name'])
         .attr("fill", root_colors[axis['fLabelColor']]);

      /*
       * Add palette axis title
       */
      var title = axis['fTitle'];
      if (title != "") {
         axisFontDetails = getFontDetails(root_fonts[Math.floor(axis['fTitleFont'] /10)]);
         var axisTitleFontSize = axis['fTitleSize'] * height;
         pal.append("text")
               .attr("class", "Z axis label")
               .attr("x", s_width + axisLabelFontSize + axisLabelOffset)
               .attr("y", s_height)
               .attr("text-anchor", "end")
               .attr("font-family", axisFontDetails['name'])
               .attr("font-weight", axisFontDetails['weight'])
               .attr("font-style", axisFontDetails['style'])
               .attr("font-size", axisTitleFontSize )
               .text(title);
      }
   };

   JSROOTPainter.drawPaveLabel = function(vis, pavelabel) {
      var w = vis.attr("width"), h = vis.attr("height");
      var pos_x = pavelabel['fX1NDC'] * w;
      var pos_y = (1.0 - pavelabel['fY1NDC']) * h;
      var width = Math.abs(pavelabel['fX2NDC'] - pavelabel['fX1NDC']) * w;
      var height = Math.abs(pavelabel['fY2NDC'] - pavelabel['fY1NDC']) * h;
      pos_y -= height;
      var font_size = Math.round(height / 1.9);
      var fcolor = root_colors[pavelabel['fFillColor']];
      var lcolor = root_colors[pavelabel['fLineColor']];
      var tcolor = root_colors[pavelabel['fTextColor']];
      var scolor = root_colors[pavelabel['fShadowColor']];
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
      var fontDetails = getFontDetails(root_fonts[Math.floor(pavelabel['fTextFont']/10)]);

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

      var line = this.translateLaTeX(pavelabel['fLabel']);

      var lw = stringWidth(vis, line, fontDetails['name'], fontDetails['weight'],
                           font_size, fontDetails['style']);
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

   JSROOTPainter.drawPaveText = function(vis, pavetext) {
      var i, j, lw, w = vis.attr("width"), h = vis.attr("height");
      var pos_x = pavetext['fX1NDC'] * w;
      var pos_y = (1.0 - pavetext['fY1NDC']) * h;
      var width = Math.abs(pavetext['fX2NDC'] - pavetext['fX1NDC']) * w;
      var height = Math.abs(pavetext['fY2NDC'] - pavetext['fY1NDC']) * h;
      pos_y -= height;
      var line, nlines = pavetext['fLines'].length;
      var font_size = Math.round(height / (nlines * 1.5));
      var fcolor = root_colors[pavetext['fFillColor']];
      var lcolor = root_colors[pavetext['fLineColor']];
      var tcolor = root_colors[pavetext['fTextColor']];
      var scolor = root_colors[pavetext['fShadowColor']];
      if (pavetext['fFillStyle'] == 0) fcolor = 'none';
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      // "middle", "start", "end"
      var align = 'start', halign = Math.round(pavetext['fTextAlign']/10);
      var baseline = 'bottom', valign = pavetext['fTextAlign']%10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';
      var lmargin = 0;
      switch (halign) {
         case 1:
            lmargin = pavetext['fMargin'] * width;
            break;
         case 2:
            lmargin = width/2;
            break;
         case 3:
            lmargin = width - (pavetext['fMargin'] * width);
            break;
      }
      var fontDetails = getFontDetails(root_fonts[Math.floor(pavetext['fTextFont']/10)]);
      var lwidth = pavetext['fBorderSize'] ? pavetext['fBorderSize'] : 0;

      var pave = vis.append("svg:g")
         .attr("x", pos_x)
         .attr("y", pos_y)
         .attr("width", width)
         .attr("height", height)
         .attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      pave.append("svg:rect")
         .attr("class", "frame")
         .attr("x", 0)
         .attr("y", 0)
         .attr("height", height)
         .attr("width", width)
         .attr("fill", fcolor)
         .style("stroke-width", lwidth ? 1 : 0)
         .style("stroke", lcolor);

      if (nlines == 1) {
         line = this.translateLaTeX(pavetext['fLines'][0]['fTitle']);

         lw = stringWidth(vis, line, fontDetails['name'], fontDetails['weight'],
                          font_size, fontDetails['style']);
         if (lw > width)
            font_size *= 0.98 * (width / lw);

         pave.append("text")
            .attr("class", "stat text")
            .attr("text-anchor", align)
            .attr("x", lmargin)
            .attr("y", (height/2) + (font_size/3))
            .attr("font-family", fontDetails['name'])
            .attr("font-weight", fontDetails['weight'])
            .attr("font-style", fontDetails['style'])
            .attr("font-size", font_size)
            .attr("fill", tcolor)
            .text(line);
      }
      else {
         var max_len = 0;
         for (j=0; j<nlines; ++j) {
            line = this.translateLaTeX(pavetext['fLines'][j]['fTitle']);
            lw = lmargin + stringWidth(vis, line, fontDetails['name'], fontDetails['weight'],
                                       font_size, fontDetails['style']);
            if (lw > max_len) max_len = lw;
         }
         if (max_len > width)
            font_size *= 0.98 * (width / max_len);

         for (j=0; j<nlines; ++j) {
            var jcolor = root_colors[pavetext['fLines'][j]['fTextColor']];
            if (pavetext['fLines'][j]['fTextColor'] == 0)
               jcolor = tcolor;
            line = this.translateLaTeX(pavetext['fLines'][j]['fTitle']);
            if (pavetext['_typename'] == 'JSROOTIO.TPaveStats') {
               var off_y = (j == 0) ? 0 : (font_size * 0.05);
               pave.append("text")
                  .attr("class", "stat text")
                  .attr("text-anchor", (j == 0) ? "middle" : "start")
                  .attr("x", ((j==0) ? width/2 : pavetext['fMargin'] * width))
                  .attr("y", ((j == 0) ? off_y + (font_size * 1.2) : off_y + ((j+1) * (font_size * 1.4))))
                  .attr("font-family", fontDetails['name'])
                  .attr("font-weight", fontDetails['weight'])
                  .attr("font-style", fontDetails['style'])
                  .attr("font-size", font_size)
                  .attr("fill", jcolor)
                  .text(line);
            }
            else {
               pave.append("text")
                  .attr("class", "stat text")
                  .attr("text-anchor", "start")
                  .attr("x", lmargin)
                  .attr("y", (j+1) * (font_size * 1.4))
                  .attr("font-family", fontDetails['name'])
                  .attr("font-weight", fontDetails['weight'])
                  .attr("font-style", fontDetails['style'])
                  .attr("font-size", font_size)
                  .attr("fill", jcolor)
                  .text(line);
            }
         }
      }

      if (pavetext['fBorderSize'] && pavetext['_typename'] == 'JSROOTIO.TPaveStats')
         pave.append("svg:line")
            .attr("x1", 0)
            .attr("y1", font_size * 1.6)
            .attr("x2", width)
            .attr("y2", font_size * 1.6)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth ? 1 : 'none');

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

   JSROOTPainter.drawPrimitives = function(vis, pad) {
      var i, j, fframe = null, frame = null;
      var primitives = pad['fPrimitives'];
      for (i=0; i<primitives.length; ++i) {
         var classname = primitives[i]['_typename'];
         if (classname == 'JSROOTIO.TFrame') {
            fframe = frame = this.createFrame(vis, pad, null, primitives[i]);
         }
         if (classname == 'JSROOTIO.TPad') {
            this.drawPad(vis, primitives[i])
         }
         if (classname == 'JSROOTIO.TPaveLabel') {
            this.drawPaveLabel(vis, primitives[i]);
         }
         if (classname == 'JSROOTIO.TLegend') {
            this.drawLegend(vis, pad, primitives[i]);
         }
         if (classname == 'JSROOTIO.TPaveText') {
            this.drawPaveText(vis, primitives[i]);
         }
         if (classname == 'JSROOTIO.TLatex') {
            this.drawText(vis, pad, primitives[i]);
         }
         if (classname == 'JSROOTIO.TText') {
            this.drawText(vis, pad, primitives[i]);
         }
         if (classname.match(/\bTH1/)) {
            this.drawHistogram1D(vis, pad, primitives[i], frame);
            if (fframe) {
               fframe['xmin'] = primitives[i]['fXaxis']['fXmin'];
               fframe['xmax'] = primitives[i]['fXaxis']['fXmax'];
               fframe['ymin'] = primitives[i]['fYaxis']['fXmin'];
               fframe['ymax'] = primitives[i]['fYaxis']['fXmax'];
               if (primitives[i]['fXaxis'].TestBit(EAxisBits.kAxisRange)) {
                  fframe['xmin'] = primitives[i].getBinLowEdge(primitives[i]['fXaxis']['fFirst']);
                  fframe['xmax'] = primitives[i].getBinUpEdge(primitives[i]['fXaxis']['fLast']);
               }
            }
         }
         if (classname.match(/\bTH2/)) {
            this.drawHistogram2D(vis, pad, primitives[i], frame);
         }
         if (classname.match(/\bTProfile/)) {
            this.drawProfile(vis, pad, primitives[i], frame);
         }
         if (classname == 'JSROOTIO.TF1') {
            if (typeof(primitives[i]['isDrawn']) == 'undefined' || primitives[i]['isDrawn'] == false)
               this.drawFunction(vis, pad, primitives[i], fframe);
            primitives[i]['isDrawn'] = true;
         }
         if (classname == 'JSROOTIO.TGraph') {
            primitives[i]['fName'] += i;
            this.drawGraph(vis, pad, primitives[i], frame);
         }
         if (classname == 'JSROOTIO.TMultiGraph') {
            this.drawMultiGraph(vis, pad, primitives[i], frame);
         }
      }
   };

   JSROOTPainter.drawProfile = function(vis, pad, histo, hframe) {
      var i, logx = false, logy = false, logz = false, gridx = false, gridy = false;
      if (pad && typeof(pad) != 'undefined') {
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      var fillcolor = root_colors[histo['fFillColor']];
      var linecolor = root_colors[histo['fLineColor']];
      if (histo['fFillColor'] == 0) {
         fillcolor = '#4572A7';
      }
      if (histo['fLineColor'] == 0) {
         linecolor = '#4572A7';
      }
      //histo['fgApproximate'] = true;
      var binwidth = ((histo['fXaxis']['fXmax'] - histo['fXaxis']['fXmin']) / histo['fXaxis']['fNbins']);
      var bins = d3.range(histo['fXaxis']['fNbins']).map(function(p) {
         return {
            x:  histo['fXaxis']['fXmin'] + (p * binwidth) + (binwidth / 2.0),
            y:  histo.getBinContent(p+1),
            xerr: binwidth / 2.0,
            yerr: histo.getBinError(p+1)
         };
      });
      var ret = hframe != null ? hframe : this.createFrame(vis, pad, histo, null);
      var frame = ret['frame'];
      var svg_frame = d3.select(ret['id']);
      var w = frame.attr("width"), h = frame.attr("height");
      if (logx)
         var x = d3.scale.log().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      else
         var x = d3.scale.linear().domain([histo['fXaxis']['fXmin'], histo['fXaxis']['fXmax']]).range([0, w]);
      if (logy)
         var y = d3.scale.log().domain([histo['fYmin'], histo['fYmax']]).range([h, 0]);
      else
         var y = d3.scale.linear().domain([histo['fYmin'], histo['fYmax']]).range([h, 0]);

      histo['x_min'] = histo['fXaxis']['fXmin'];
      histo['x_max'] = histo['fXaxis']['fXmax'];
      histo['y_min'] = histo['fYmin'];
      histo['y_max'] = histo['fYmax'];

      histo['x'] = x;
      histo['y'] = y;
      histo['bins'] = bins;

      this.drawGrid(frame, histo, pad, x, y);
      this.drawErrors(svg_frame, bins, histo, pad, x, y);
      this.drawAxes(frame, histo, pad, x, y);
      this.drawTitle(vis, histo, pad);
      this.addInteraction(frame, histo);
      this.drawFunctions(vis, histo, pad, ret);
      if (!pad || typeof(pad) == 'undefined')
         this.drawStat(vis, histo);
      return null;
   };

   JSROOTPainter.drawStat = function(vis, histo) {
      var w = vis.attr("width"), h = vis.attr("height");
      var lcount=0, nlines = 4; // name, entries, mean, rms
      if (histo['_typename'] && histo['_typename'].match(/\bTProfile/))
         nlines += 2; // mean y, rms y
      if (histo['_typename'] && histo['_typename'].match(/\bTH2/))
         nlines += 2; // mean y, rms y

      var statx = 0.980;
      var staty = 0.935;
      var statw = 0.2;
      var stath = 0.25 * nlines * 0.16;

      var pos_x = (statx - statw) * w;
      var pos_y = (1.0 - staty) * h;
      var width = statw * w;
      var height = stath * h;
      var font_size = Math.round(height / (nlines * 1.7));
      var fontDetails = getFontDetails(root_fonts[4]);

      var pave = vis.append("svg:g")
         .attr("width", width)
         .attr("height", height)
         .attr("transform", "translate(" + pos_x + ", " + pos_y + ")");

      var stat_rect = pave.append("svg:rect")
         .attr("class", "frame")
         .attr("x", 0)
         .attr("y", 0)
         .attr("width", width)
         .attr("height", height)
         .attr("fill", 'white')
         .style("stroke-width", 1)
         .style("stroke", 'black');

      var  line = histo['fName'];
      pave.append("svg:text")
            .attr("x", width/2)
            .attr("y", ++lcount * (font_size * 1.4))
            .attr("font-size", font_size)
            .attr("text-anchor", "middle")
            .attr("vertical-anchor", "bottom")
            .text(line);

      line = 'Entries = ' + histo['fEntries'];
      pave.append("svg:text")
            .attr("x", 5)
            .attr("y", ((font_size * 0.06)) + (++lcount * (font_size * 1.6)))
            .attr("font-size", font_size)
            .attr("text-anchor", "start")
            .attr("vertical-anchor", "bottom")
            .text(line);

      if (histo['_typename'] && histo['_typename'].match(/\bTH2/))
         line = 'Mean x = ' + histo.getMean(1).toFixed(6.4);
      else
         line = 'Mean = ' + histo.getMean(1).toFixed(6.4);
      pave.append("svg:text")
            .attr("x", 5)
            .attr("y", ((font_size * 0.06)) + (++lcount * (font_size * 1.6)))
            .attr("font-size", font_size)
            .attr("text-anchor", "start")
            .attr("vertical-anchor", "bottom")
            .text(line);
      if ((histo['_typename'] && histo['_typename'].match(/\bTProfile/)) ||
           (histo['_typename'] && histo['_typename'].match(/\bTH2/))) {
         line = 'Mean y = ' + histo.getMean(2).toFixed(6.4);
         pave.append("svg:text")
               .attr("x", 5)
               .attr("y", ((font_size * 0.06)) + (++lcount * (font_size * 1.6)))
               .attr("font-size", font_size)
               .attr("text-anchor", "start")
               .attr("vertical-anchor", "bottom")
               .text(line);
      }
      if (histo['_typename'] && histo['_typename'].match(/\bTH2/))
         line = 'RMS x = ' + histo.getRMS(1).toFixed(6.4);
      else
         line = 'RMS = ' + histo.getRMS(1).toFixed(6.4);
      pave.append("svg:text")
            .attr("x", 5)
            .attr("y", ((font_size * 0.06)) + (++lcount * (font_size * 1.6)))
            .attr("font-size", font_size)
            .attr("text-anchor", "start")
            .attr("vertical-anchor", "bottom")
            .text(line);
      if ((histo['_typename'] && histo['_typename'].match(/\bTProfile/)) ||
           (histo['_typename'] && histo['_typename'].match(/\bTH2/))){
         line = 'RMS y = ' + histo.getRMS(2).toFixed(6.4);
         pave.append("svg:text")
               .attr("x", 5)
               .attr("y", ((font_size * 0.06)) + (++lcount * (font_size * 1.6)))
               .attr("font-size", font_size)
               .attr("text-anchor", "start")
               .attr("vertical-anchor", "bottom")
               .text(line);
      }
      pave.append("svg:line")
          .attr("x1", 0)
          .attr("y1", font_size * 2)
          .attr("x2", width)
          .attr("y2", font_size * 2)
          .style("stroke", 'black')
          .style("stroke-width", 1);
   };

   JSROOTPainter.drawText = function(vis, pad, text) {
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      var i, w = vis.attr("width"), h = vis.attr("height");
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
      var tcolor = root_colors[text['fTextColor']];
      var fontDetails = getFontDetails(root_fonts[Math.floor(text['fTextFont']/10)]);

      var string = text['fTitle'];
      // translate the LaTeX symbols
      if (text['_typename'] == 'JSROOTIO.TLatex')
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

   JSROOTPainter.drawTitle = function(vis, histo, pad) {
      /* draw the title only if we don't draw from a pad (see Olivier for details) */
      var w = vis.attr("width"), h = vis.attr("height");
      var font_size = Math.round(0.050 * h);
      if (!pad || typeof(pad) == 'undefined') {
         vis.append("text")
            .attr("class", "title")
            .attr("text-anchor", "middle")
            .attr("x", w/2)
            .attr("y", 0.07 * vis.attr("height"))
            .attr("font-family", "Arial")
            .attr("font-size", font_size)
            .text(histo['fTitle']);
      }
   };

   /**
    * List tree (dtree) related functions
    */

   JSROOTPainter.displayListOfKeys = function(keys, container) {
      delete key_tree;
      var content = "<p><a href='javascript: key_tree.openAll();'>open all</a> | <a href='javascript: key_tree.closeAll();'>close all</a></p>";
      key_tree = new dTree('key_tree');
      key_tree.config.useCookies = false;
      key_tree.add(0, -1, 'File Content');
      var k = 1;
      var tree_link = '';
      for (var i=0; i<keys.length; ++i) {
         var message = keys[i]['className']+' is not yet implemented.';
         tree_link = "javascript:  alert('" + message + "')";
         var node_img = source_dir+'img/page.gif';
         if (keys[i]['className'].match(/\bTH1/)  ||
             keys[i]['className'].match(/\bTH2/)  ||
             keys[i]['className'] == 'TGraph') {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_img = source_dir+'img/graphical.png';
         }
         else if (keys[i]['className'] ==  'TF1') {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_img = source_dir+'img/graphical.png';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['className'] ==  'TProfile') {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_img = source_dir+'img/graphical.png';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['name'] == 'StreamerInfo') {
            tree_link = "javascript: displayStreamerInfos(gFile.fStreamerInfo.fStreamerInfos);";
            node_img = source_dir+'img/question.gif';
         }
         else if (keys[i]['className'] == 'TDirectory') {
            tree_link = "javascript: showDirectory('"+keys[i]['name']+"',"+keys[i]['cycle']+","+(i+1)+");";
            node_img = source_dir+'img/folder.gif';
         }
         else if (keys[i]['className'].match('TTree') ||
                  keys[i]['className'].match('TNtuple')) {
            node_img = source_dir+'img/tree_t.png';
         }
         else if (keys[i]['className'].match('TGeoManager') ||
                  keys[i]['className'].match('TGeometry')) {
            node_img = source_dir+'img/folder.gif';
         }
         else if (keys[i]['className'].match('TCanvas')) {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_title = keys[i]['name'];
         }
         if (keys[i]['name'] != '' && keys[i]['className'] != 'TFile')
            if (keys[i]['className'] == 'TDirectory')
               key_tree.add(k, 0, keys[i]['name']+';'+keys[i]['cycle'], tree_link, keys[i]['name'], '', node_img,
                            source_dir+'img/folderopen.gif');
            else
               key_tree.add(k, 0, keys[i]['name']+';'+keys[i]['cycle'], tree_link, keys[i]['name'], '', node_img);
            k++;
      }
      content += key_tree;
      $(container).append(content);
   };

   JSROOTPainter.addDirectoryKeys = function(keys, container, dir_id) {
      var pattern_th1 = /TH1/g;
      var pattern_th2 = /TH2/g;
      var tree_link = '';
      var content = "<p><a href='javascript: key_tree.openAll();'>open all</a> | <a href='javascript: key_tree.closeAll();'>close all</a></p>";
      var k = key_tree.aNodes.length;
      var dir_name = key_tree.aNodes[dir_id]['title'];
      for (var i=0; i<keys.length; ++i) {
         var disp_name = keys[i]['name'];
         keys[i]['name'] = dir_name + '/' + keys[i]['name'];
         var message = keys[i]['className']+' is not yet implemented.';
         tree_link = "javascript:  alert('" + message + "')";
         var node_img = source_dir+'img/page.gif';
         var node_title = keys[i]['className'];
         if (keys[i]['className'].match(/\bTH1/) ||
             keys[i]['className'].match(/\bTH2/) ||
             keys[i]['className'] == 'TGraph') {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_img = source_dir+'img/graphical.png';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['className'] ==  'TProfile') {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_img = source_dir+'img/graphical.png';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['name'] == 'StreamerInfo') {
            tree_link = "javascript: displayStreamerInfos(gFile.fStreamerInfo.fStreamerInfos);";
            node_img = source_dir+'img/question.gif';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['className'] == 'TDirectory') {
            tree_link = "javascript: showDirectory('"+keys[i]['name']+"',"+keys[i]['cycle']+","+k+");";
            node_img = source_dir+'img/folder.gif';
            node_title = keys[i]['name'];
         }
         else if (keys[i]['className'].match('TTree') ||
                  keys[i]['className'].match('TNtuple')) {
            node_img = source_dir+'img/tree_t.png';
         }
         else if (keys[i]['className'].match('TCanvas')) {
            tree_link = "javascript: showObject('"+keys[i]['name']+"',"+keys[i]['cycle']+");";
            node_title = keys[i]['name'];
         }
         if (keys[i]['name'] != '' && keys[i]['className'] != 'TFile') {
            if (keys[i]['className'] == 'TDirectory')
               key_tree.add(k, dir_id, disp_name+';'+keys[i]['cycle'], tree_link, node_title, '', node_img,
                            source_dir+'img/folderopen.gif');
            else
               key_tree.add(k, dir_id, disp_name+';'+keys[i]['cycle'], tree_link, node_title, '', node_img);
            k++;
         }
      }
      content += key_tree;
      $(container).append(content);
      key_tree.openTo(dir_id, true);
   }

   JSROOTPainter.displayStreamerInfos = function(streamerInfo, container) {

      delete d;
      var content = "<p><a href='javascript: d_tree.openAll();'>open all</a> | <a href='javascript: d_tree.closeAll();'>close all</a></p>";
      d_tree = new dTree('d_tree');
      d_tree.config.useCookies = false;
      d_tree.add(0, -1, 'Streamer Infos');

      var k = 1;
      var pid = 0;
      var cid = 0;
      var key;
      for (key in streamerInfo) {
         var entry = streamerInfo[key]['name'];
         d_tree.add(k, 0, entry); k++;
      }
      var j=0;
      for (key in streamerInfo) {
         if (typeof(streamerInfo[key]['checksum']) != 'undefined')
            d_tree.add(k, j+1, 'Checksum: ' + streamerInfo[key]['checksum']); ++k;
         if (typeof(streamerInfo[key]['classversion']) != 'undefined')
            d_tree.add(k, j+1, 'Class Version: ' + streamerInfo[key]['classversion']); ++k;
         if (typeof(streamerInfo[key]['title']) != 'undefined')
            d_tree.add(k, j+1, 'Title: ' + streamerInfo[key]['title']); ++k;
         if (typeof(streamerInfo[key]['elements']) != 'undefined') {
            d_tree.add(k, j+1, 'Elements'); pid=k; ++k;
            for (var l=0; l<streamerInfo[key]['elements']['array'].length; ++l) {
               if (typeof(streamerInfo[key]['elements']['array'][l]['element']) != 'undefined') {
                  d_tree.add(k, pid, streamerInfo[key]['elements']['array'][l]['element']['name']); cid=k; ++k;
                  d_tree.add(k, cid, streamerInfo[key]['elements']['array'][l]['element']['title']); ++k;
                  d_tree.add(k, cid, streamerInfo[key]['elements']['array'][l]['element']['typename']); ++k;
               }
               else if (typeof(streamerInfo[key]['elements']['array'][l]['name']) != 'undefined') {
                  d_tree.add(k, pid, streamerInfo[key]['elements']['array'][l]['name']); cid=k; ++k;
                  d_tree.add(k, cid, streamerInfo[key]['elements']['array'][l]['title']); ++k;
                  d_tree.add(k, cid, streamerInfo[key]['elements']['array'][l]['typename']); ++k;
               }
            }
         }
         else if (typeof(streamerInfo[key]['array']) != 'undefined') {
            for (var l=0; l<streamerInfo[key]['array'].length; ++l) {
               d_tree.add(k, j+1, streamerInfo[key]['array'][l]['str']); ++k;
            }
         }
         ++j;
      }
      content += d_tree;
      $(container).html(content);
   };

   var style = "<style>\n"
      +".xaxis path, .xaxis line, .yaxis path, .yaxis line, .zaxis path, .zaxis line {\n"
      +"   fill: none;\n"
      +"   stroke: #000;\n"
      +"   shape-rendering: crispEdges;\n"
      +"}\n"
      +".brush .extent {\n"
      +"  stroke: #fff;\n"
      +"  fill-opacity: .125;\n"
      +"  shape-rendering: crispEdges;\n"
      +"}\n"
      +"rect.zoom {\n"
      +"  stroke: steelblue;\n"
      +"  fill-opacity: 0.1;\n"
      +"}\n"
      /* Correct overflow not hidden in IE9 */
      +"svg:not(:root) { overflow: hidden; }\n"
      +"</style>\n";
   $(style).prependTo("body");

})();


// JSROOTD3Painter.js ends


