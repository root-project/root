/** @license
 *
 * jsPDF - PDF Document creation from JavaScript
 * Version 3.0.3
 *
 * Copyright (c) 2010-2025 James Hall <james@parall.ax>, https://github.com/MrRio/jsPDF
 *               2015-2025 yWorks GmbH, http://www.yworks.com
 *               2015-2025 Lukas Holländer <lukas.hollaender@yworks.com>, https://github.com/HackbrettXXX
 *               2016-2018 Aras Abbasi <aras.abbasi@gmail.com>
 *               2010 Aaron Spike, https://github.com/acspike
 *               2012 Willow Systems Corporation, https://github.com/willowsystems
 *               2012 Pablo Hess, https://github.com/pablohess
 *               2012 Florian Jenett, https://github.com/fjenett
 *               2013 Warren Weckesser, https://github.com/warrenweckesser
 *               2013 Youssef Beddad, https://github.com/lifof
 *               2013 Lee Driscoll, https://github.com/lsdriscoll
 *               2013 Stefan Slonevskiy, https://github.com/stefslon
 *               2013 Jeremy Morel, https://github.com/jmorel
 *               2013 Christoph Hartmann, https://github.com/chris-rock
 *               2014 Juan Pablo Gaviria, https://github.com/juanpgaviria
 *               2014 James Makes, https://github.com/dollaruw
 *               2014 Diego Casorran, https://github.com/diegocr
 *               2014 Steven Spungin, https://github.com/Flamenco
 *               2014 Kenneth Glassey, https://github.com/Gavvers
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Contributor(s):
 *    siefkenj, ahwolf, rickygu, Midnith, saintclair, eaparango,
 *    kim3er, mfo, alnorth, Flamenco
 */

const globalObject = globalThis;

/**
 * A class to parse color values
 * @author Stoyan Stefanov <sstoo@gmail.com>
 * {@link   http://www.phpied.com/rgb-color-parser-in-javascript/}
 * @license Use it if you like it
 */

function RGBColor(color_string) {
  color_string = color_string || "";
  this.ok = false;

  // strip any leading #
  if (color_string.charAt(0) == "#") {
    // remove # if any
    color_string = color_string.substr(1, 6);
  }

  color_string = color_string.replace(/ /g, "");
  color_string = color_string.toLowerCase();

  var channels;

  // before getting into regexps, try simple matches
  // and overwrite the input
  var simple_colors = {
    aliceblue: "f0f8ff",
    antiquewhite: "faebd7",
    aqua: "00ffff",
    aquamarine: "7fffd4",
    azure: "f0ffff",
    beige: "f5f5dc",
    bisque: "ffe4c4",
    black: "000000",
    blanchedalmond: "ffebcd",
    blue: "0000ff",
    blueviolet: "8a2be2",
    brown: "a52a2a",
    burlywood: "deb887",
    cadetblue: "5f9ea0",
    chartreuse: "7fff00",
    chocolate: "d2691e",
    coral: "ff7f50",
    cornflowerblue: "6495ed",
    cornsilk: "fff8dc",
    crimson: "dc143c",
    cyan: "00ffff",
    darkblue: "00008b",
    darkcyan: "008b8b",
    darkgoldenrod: "b8860b",
    darkgray: "a9a9a9",
    darkgreen: "006400",
    darkkhaki: "bdb76b",
    darkmagenta: "8b008b",
    darkolivegreen: "556b2f",
    darkorange: "ff8c00",
    darkorchid: "9932cc",
    darkred: "8b0000",
    darksalmon: "e9967a",
    darkseagreen: "8fbc8f",
    darkslateblue: "483d8b",
    darkslategray: "2f4f4f",
    darkturquoise: "00ced1",
    darkviolet: "9400d3",
    deeppink: "ff1493",
    deepskyblue: "00bfff",
    dimgray: "696969",
    dodgerblue: "1e90ff",
    feldspar: "d19275",
    firebrick: "b22222",
    floralwhite: "fffaf0",
    forestgreen: "228b22",
    fuchsia: "ff00ff",
    gainsboro: "dcdcdc",
    ghostwhite: "f8f8ff",
    gold: "ffd700",
    goldenrod: "daa520",
    gray: "808080",
    green: "008000",
    greenyellow: "adff2f",
    honeydew: "f0fff0",
    hotpink: "ff69b4",
    indianred: "cd5c5c",
    indigo: "4b0082",
    ivory: "fffff0",
    khaki: "f0e68c",
    lavender: "e6e6fa",
    lavenderblush: "fff0f5",
    lawngreen: "7cfc00",
    lemonchiffon: "fffacd",
    lightblue: "add8e6",
    lightcoral: "f08080",
    lightcyan: "e0ffff",
    lightgoldenrodyellow: "fafad2",
    lightgrey: "d3d3d3",
    lightgreen: "90ee90",
    lightpink: "ffb6c1",
    lightsalmon: "ffa07a",
    lightseagreen: "20b2aa",
    lightskyblue: "87cefa",
    lightslateblue: "8470ff",
    lightslategray: "778899",
    lightsteelblue: "b0c4de",
    lightyellow: "ffffe0",
    lime: "00ff00",
    limegreen: "32cd32",
    linen: "faf0e6",
    magenta: "ff00ff",
    maroon: "800000",
    mediumaquamarine: "66cdaa",
    mediumblue: "0000cd",
    mediumorchid: "ba55d3",
    mediumpurple: "9370d8",
    mediumseagreen: "3cb371",
    mediumslateblue: "7b68ee",
    mediumspringgreen: "00fa9a",
    mediumturquoise: "48d1cc",
    mediumvioletred: "c71585",
    midnightblue: "191970",
    mintcream: "f5fffa",
    mistyrose: "ffe4e1",
    moccasin: "ffe4b5",
    navajowhite: "ffdead",
    navy: "000080",
    oldlace: "fdf5e6",
    olive: "808000",
    olivedrab: "6b8e23",
    orange: "ffa500",
    orangered: "ff4500",
    orchid: "da70d6",
    palegoldenrod: "eee8aa",
    palegreen: "98fb98",
    paleturquoise: "afeeee",
    palevioletred: "d87093",
    papayawhip: "ffefd5",
    peachpuff: "ffdab9",
    peru: "cd853f",
    pink: "ffc0cb",
    plum: "dda0dd",
    powderblue: "b0e0e6",
    purple: "800080",
    red: "ff0000",
    rosybrown: "bc8f8f",
    royalblue: "4169e1",
    saddlebrown: "8b4513",
    salmon: "fa8072",
    sandybrown: "f4a460",
    seagreen: "2e8b57",
    seashell: "fff5ee",
    sienna: "a0522d",
    silver: "c0c0c0",
    skyblue: "87ceeb",
    slateblue: "6a5acd",
    slategray: "708090",
    snow: "fffafa",
    springgreen: "00ff7f",
    steelblue: "4682b4",
    tan: "d2b48c",
    teal: "008080",
    thistle: "d8bfd8",
    tomato: "ff6347",
    turquoise: "40e0d0",
    violet: "ee82ee",
    violetred: "d02090",
    wheat: "f5deb3",
    white: "ffffff",
    whitesmoke: "f5f5f5",
    yellow: "ffff00",
    yellowgreen: "9acd32"
  };
  color_string = simple_colors[color_string] || color_string;

  // array of color definition objects
  var color_defs = [
    {
      re: /^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$/,
      example: ["rgb(123, 234, 45)", "rgb(255,234,245)"],
      process: function(bits) {
        return [parseInt(bits[1]), parseInt(bits[2]), parseInt(bits[3])];
      }
    },
    {
      re: /^(\w{2})(\w{2})(\w{2})$/,
      example: ["#00ff00", "336699"],
      process: function(bits) {
        return [
          parseInt(bits[1], 16),
          parseInt(bits[2], 16),
          parseInt(bits[3], 16)
        ];
      }
    },
    {
      re: /^(\w{1})(\w{1})(\w{1})$/,
      example: ["#fb0", "f0f"],
      process: function(bits) {
        return [
          parseInt(bits[1] + bits[1], 16),
          parseInt(bits[2] + bits[2], 16),
          parseInt(bits[3] + bits[3], 16)
        ];
      }
    }
  ];

  // search through the definitions to find a match
  for (var i = 0; i < color_defs.length; i++) {
    var re = color_defs[i].re;
    var processor = color_defs[i].process;
    var bits = re.exec(color_string);
    if (bits) {
      channels = processor(bits);
      this.r = channels[0];
      this.g = channels[1];
      this.b = channels[2];
      this.ok = true;
    }
  }

  // validate/cleanup values
  this.r = this.r < 0 || isNaN(this.r) ? 0 : this.r > 255 ? 255 : this.r;
  this.g = this.g < 0 || isNaN(this.g) ? 0 : this.g > 255 ? 255 : this.g;
  this.b = this.b < 0 || isNaN(this.b) ? 0 : this.b > 255 ? 255 : this.b;

  // some getters
  this.toRGB = function() {
    return "rgb(" + this.r + ", " + this.g + ", " + this.b + ")";
  };
  this.toHex = function() {
    var r = this.r.toString(16);
    var g = this.g.toString(16);
    var b = this.b.toString(16);
    if (r.length == 1) r = "0" + r;
    if (g.length == 1) g = "0" + g;
    if (b.length == 1) b = "0" + b;
    return "#" + r + g + b;
  };
}

let atob, btoa;

if ((typeof process === 'object') && (typeof process.versions === 'object') && process.versions.node && process.versions.v8) {
   atob = str => Buffer.from(str, 'base64').toString('latin1');
   btoa = str => Buffer.from(str, 'latin1').toString('base64');
} else {
   atob = globalThis.atob;
   btoa = globalThis.btoa;
}

function consoleLog() {
  if (globalObject.console && typeof globalObject.console.log === "function") {
    globalObject.console.log.apply(globalObject.console, arguments);
  }
}

function consoleWarn(str) {
  if (globalObject.console) {
    if (typeof globalObject.console.warn === "function") {
      globalObject.console.warn.apply(globalObject.console, arguments);
    } else {
      consoleLog.call(null, arguments);
    }
  }
}

function consoleError(str) {
  if (globalObject.console) {
    if (typeof globalObject.console.error === "function") {
      globalObject.console.error.apply(globalObject.console, arguments);
    } else {
      consoleLog(str);
    }
  }
}
var console = {
  log: consoleLog,
  warn: consoleWarn,
  error: consoleError
};

/**
 * @license
 * Joseph Myers does not specify a particular license for his work.
 *
 * Author: Joseph Myers
 * Accessed from: http://www.myersdaily.org/joseph/javascript/md5.js
 *
 * Modified by: Owen Leong
 */

function md5cycle(x, k) {
  var a = x[0],
    b = x[1],
    c = x[2],
    d = x[3];

  a = ff(a, b, c, d, k[0], 7, -680876936);
  d = ff(d, a, b, c, k[1], 12, -389564586);
  c = ff(c, d, a, b, k[2], 17, 606105819);
  b = ff(b, c, d, a, k[3], 22, -1044525330);
  a = ff(a, b, c, d, k[4], 7, -176418897);
  d = ff(d, a, b, c, k[5], 12, 1200080426);
  c = ff(c, d, a, b, k[6], 17, -1473231341);
  b = ff(b, c, d, a, k[7], 22, -45705983);
  a = ff(a, b, c, d, k[8], 7, 1770035416);
  d = ff(d, a, b, c, k[9], 12, -1958414417);
  c = ff(c, d, a, b, k[10], 17, -42063);
  b = ff(b, c, d, a, k[11], 22, -1990404162);
  a = ff(a, b, c, d, k[12], 7, 1804603682);
  d = ff(d, a, b, c, k[13], 12, -40341101);
  c = ff(c, d, a, b, k[14], 17, -1502002290);
  b = ff(b, c, d, a, k[15], 22, 1236535329);

  a = gg(a, b, c, d, k[1], 5, -165796510);
  d = gg(d, a, b, c, k[6], 9, -1069501632);
  c = gg(c, d, a, b, k[11], 14, 643717713);
  b = gg(b, c, d, a, k[0], 20, -373897302);
  a = gg(a, b, c, d, k[5], 5, -701558691);
  d = gg(d, a, b, c, k[10], 9, 38016083);
  c = gg(c, d, a, b, k[15], 14, -660478335);
  b = gg(b, c, d, a, k[4], 20, -405537848);
  a = gg(a, b, c, d, k[9], 5, 568446438);
  d = gg(d, a, b, c, k[14], 9, -1019803690);
  c = gg(c, d, a, b, k[3], 14, -187363961);
  b = gg(b, c, d, a, k[8], 20, 1163531501);
  a = gg(a, b, c, d, k[13], 5, -1444681467);
  d = gg(d, a, b, c, k[2], 9, -51403784);
  c = gg(c, d, a, b, k[7], 14, 1735328473);
  b = gg(b, c, d, a, k[12], 20, -1926607734);

  a = hh(a, b, c, d, k[5], 4, -378558);
  d = hh(d, a, b, c, k[8], 11, -2022574463);
  c = hh(c, d, a, b, k[11], 16, 1839030562);
  b = hh(b, c, d, a, k[14], 23, -35309556);
  a = hh(a, b, c, d, k[1], 4, -1530992060);
  d = hh(d, a, b, c, k[4], 11, 1272893353);
  c = hh(c, d, a, b, k[7], 16, -155497632);
  b = hh(b, c, d, a, k[10], 23, -1094730640);
  a = hh(a, b, c, d, k[13], 4, 681279174);
  d = hh(d, a, b, c, k[0], 11, -358537222);
  c = hh(c, d, a, b, k[3], 16, -722521979);
  b = hh(b, c, d, a, k[6], 23, 76029189);
  a = hh(a, b, c, d, k[9], 4, -640364487);
  d = hh(d, a, b, c, k[12], 11, -421815835);
  c = hh(c, d, a, b, k[15], 16, 530742520);
  b = hh(b, c, d, a, k[2], 23, -995338651);

  a = ii(a, b, c, d, k[0], 6, -198630844);
  d = ii(d, a, b, c, k[7], 10, 1126891415);
  c = ii(c, d, a, b, k[14], 15, -1416354905);
  b = ii(b, c, d, a, k[5], 21, -57434055);
  a = ii(a, b, c, d, k[12], 6, 1700485571);
  d = ii(d, a, b, c, k[3], 10, -1894986606);
  c = ii(c, d, a, b, k[10], 15, -1051523);
  b = ii(b, c, d, a, k[1], 21, -2054922799);
  a = ii(a, b, c, d, k[8], 6, 1873313359);
  d = ii(d, a, b, c, k[15], 10, -30611744);
  c = ii(c, d, a, b, k[6], 15, -1560198380);
  b = ii(b, c, d, a, k[13], 21, 1309151649);
  a = ii(a, b, c, d, k[4], 6, -145523070);
  d = ii(d, a, b, c, k[11], 10, -1120210379);
  c = ii(c, d, a, b, k[2], 15, 718787259);
  b = ii(b, c, d, a, k[9], 21, -343485551);

  x[0] = add32(a, x[0]);
  x[1] = add32(b, x[1]);
  x[2] = add32(c, x[2]);
  x[3] = add32(d, x[3]);
}

function cmn(q, a, b, x, s, t) {
  a = add32(add32(a, q), add32(x, t));
  return add32((a << s) | (a >>> (32 - s)), b);
}

function ff(a, b, c, d, x, s, t) {
  return cmn((b & c) | (~b & d), a, b, x, s, t);
}

function gg(a, b, c, d, x, s, t) {
  return cmn((b & d) | (c & ~d), a, b, x, s, t);
}

function hh(a, b, c, d, x, s, t) {
  return cmn(b ^ c ^ d, a, b, x, s, t);
}

function ii(a, b, c, d, x, s, t) {
  return cmn(c ^ (b | ~d), a, b, x, s, t);
}

function md51(s) {
  // txt = '';
  var n = s.length,
    state = [1732584193, -271733879, -1732584194, 271733878],
    i;
  for (i = 64; i <= s.length; i += 64) {
    md5cycle(state, md5blk(s.substring(i - 64, i)));
  }
  s = s.substring(i - 64);
  var tail = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (i = 0; i < s.length; i++)
    tail[i >> 2] |= s.charCodeAt(i) << (i % 4 << 3);
  tail[i >> 2] |= 0x80 << (i % 4 << 3);
  if (i > 55) {
    md5cycle(state, tail);
    for (i = 0; i < 16; i++) tail[i] = 0;
  }
  tail[14] = n * 8;
  md5cycle(state, tail);
  return state;
}

/* there needs to be support for Unicode here,
 * unless we pretend that we can redefine the MD-5
 * algorithm for multi-byte characters (perhaps
 * by adding every four 16-bit characters and
 * shortening the sum to 32 bits). Otherwise
 * I suggest performing MD-5 as if every character
 * was two bytes--e.g., 0040 0025 = @%--but then
 * how will an ordinary MD-5 sum be matched?
 * There is no way to standardize text to something
 * like UTF-8 before transformation; speed cost is
 * utterly prohibitive. The JavaScript standard
 * itself needs to look at this: it should start
 * providing access to strings as preformed UTF-8
 * 8-bit unsigned value arrays.
 */
function md5blk(s) {
  /* I figured global was faster.   */
  var md5blks = [],
    i; /* Andy King said do it this way. */
  for (i = 0; i < 64; i += 4) {
    md5blks[i >> 2] =
      s.charCodeAt(i) +
      (s.charCodeAt(i + 1) << 8) +
      (s.charCodeAt(i + 2) << 16) +
      (s.charCodeAt(i + 3) << 24);
  }
  return md5blks;
}

var hex_chr = "0123456789abcdef".split("");

function rhex(n) {
  var s = "",
    j = 0;
  for (; j < 4; j++)
    s += hex_chr[(n >> (j * 8 + 4)) & 0x0f] + hex_chr[(n >> (j * 8)) & 0x0f];
  return s;
}

function hex(x) {
  for (var i = 0; i < x.length; i++) x[i] = rhex(x[i]);
  return x.join("");
}

// Converts a 4-byte number to byte string
function singleToByteString(n) {
  return String.fromCharCode(
    (n & 0xff) >> 0,
    (n & 0xff00) >> 8,
    (n & 0xff0000) >> 16,
    (n & 0xff000000) >> 24
  );
}

// Converts an array of numbers to a byte string
function toByteString(x) {
  return x.map(singleToByteString).join("");
}

// Returns the MD5 hash as a byte string
function md5Bin(s) {
  return toByteString(md51(s));
}

// Returns MD5 hash as a hex string
function md5(s) {
  return hex(md51(s));
}

var md5Check = md5("hello") != "5d41402abc4b2a76b9719d911017c592";

function add32(a, b) {
  if (md5Check) {
    /* if the md5Check does not match
     the expected value, we're dealing
     with an old browser and need
     this function. */
    var lsw = (a & 0xffff) + (b & 0xffff),
      msw = (a >> 16) + (b >> 16) + (lsw >> 16);
    return (msw << 16) | (lsw & 0xffff);
  } else {
    /* this function is much faster,
    so if possible we use it. Some IEs
    are the only ones I know of that
    need the idiotic second function,
    generated by an if clause.  */
    return (a + b) & 0xffffffff;
  }
}

/**
 * @license
 * FPDF is released under a permissive license: there is no usage restriction.
 * You may embed it freely in your application (commercial or not), with or
 * without modifications.
 *
 * Reference: http://www.fpdf.org/en/script/script37.php
 */

function repeat(str, num) {
  return new Array(num + 1).join(str);
}

/**
 * Converts a byte string to a hex string
 *
 * @name rc4
 * @function
 * @param {string} key Byte string of encryption key
 * @param {string} data Byte string of data to be encrypted
 * @returns {string} Encrypted string
 */
function rc4(key, data) {
  var lastKey, lastState;
  if (key !== lastKey) {
    var k = repeat(key, ((256 / key.length) >> 0) + 1);
    var state = [];
    for (var i = 0; i < 256; i++) {
      state[i] = i;
    }
    var j = 0;
    for (var i = 0; i < 256; i++) {
      var t = state[i];
      j = (j + t + k.charCodeAt(i)) % 256;
      state[i] = state[j];
      state[j] = t;
    }
    lastKey = key;
    lastState = state;
  } else {
    state = lastState;
  }
  var length = data.length;
  var a = 0;
  var b = 0;
  var out = "";
  for (var i = 0; i < length; i++) {
    a = (a + 1) % 256;
    t = state[a];
    b = (b + t) % 256;
    state[a] = state[b];
    state[b] = t;
    k = state[(state[a] + state[b]) % 256];
    out += String.fromCharCode(data.charCodeAt(i) ^ k);
  }
  return out;
}

/**
 * @license
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 * Author: Owen Leong (@owenl131)
 * Date: 15 Oct 2020
 * References:
 * https://www.cs.cmu.edu/~dst/Adobe/Gallery/anon21jul01-pdf-encryption.txt
 * https://github.com/foliojs/pdfkit/blob/master/lib/security.js
 * http://www.fpdf.org/en/script/script37.php
 */

var permissionOptions = {
  print: 4,
  modify: 8,
  copy: 16,
  "annot-forms": 32
};

/**
 * Initializes encryption settings
 *
 * @name constructor
 * @function
 * @param {Array} permissions Permissions allowed for user, "print", "modify", "copy" and "annot-forms".
 * @param {String} userPassword Permissions apply to this user. Leaving this empty means the document
 *                              is not password protected but viewer has the above permissions.
 * @param {String} ownerPassword Owner has full functionalities to the file.
 * @param {String} fileId As hex string, should be same as the file ID in the trailer.
 * @example
 * var security = new PDFSecurity(["print"])
 */
function PDFSecurity(permissions, userPassword, ownerPassword, fileId) {
  this.v = 1; // algorithm 1, future work can add in more recent encryption schemes
  this.r = 2; // revision 2

  // set flags for what functionalities the user can access
  let protection = 192;
  permissions.forEach(function(perm) {
    if (typeof permissionOptions.perm !== "undefined") {
      throw new Error("Invalid permission: " + perm);
    }
    protection += permissionOptions[perm];
  });

  // padding is used to pad the passwords to 32 bytes, also is hashed and stored in the final PDF
  this.padding =
    "\x28\xBF\x4E\x5E\x4E\x75\x8A\x41\x64\x00\x4E\x56\xFF\xFA\x01\x08" +
    "\x2E\x2E\x00\xB6\xD0\x68\x3E\x80\x2F\x0C\xA9\xFE\x64\x53\x69\x7A";
  let paddedUserPassword = (userPassword + this.padding).substr(0, 32);
  let paddedOwnerPassword = (ownerPassword + this.padding).substr(0, 32);

  this.O = this.processOwnerPassword(paddedUserPassword, paddedOwnerPassword);
  this.P = -((protection ^ 255) + 1);
  this.encryptionKey = md5Bin(
    paddedUserPassword +
      this.O +
      this.lsbFirstWord(this.P) +
      this.hexToBytes(fileId)
  ).substr(0, 5);
  this.U = rc4(this.encryptionKey, this.padding);
}

/**
 * Breaks down a 4-byte number into its individual bytes, with the least significant bit first
 *
 * @name lsbFirstWord
 * @function
 * @param {number} data 32-bit number
 * @returns {Array}
 */
PDFSecurity.prototype.lsbFirstWord = function(data) {
  return String.fromCharCode(
    (data >> 0) & 0xff,
    (data >> 8) & 0xff,
    (data >> 16) & 0xff,
    (data >> 24) & 0xff
  );
};

/**
 * Converts a byte string to a hex string
 *
 * @name toHexString
 * @function
 * @param {String} byteString Byte string
 * @returns {String}
 */
PDFSecurity.prototype.toHexString = function(byteString) {
  return byteString
    .split("")
    .map(function(byte) {
      return ("0" + (byte.charCodeAt(0) & 0xff).toString(16)).slice(-2);
    })
    .join("");
};

/**
 * Converts a hex string to a byte string
 *
 * @name hexToBytes
 * @function
 * @param {String} hex Hex string
 * @returns {String}
 */
PDFSecurity.prototype.hexToBytes = function(hex) {
  for (var bytes = [], c = 0; c < hex.length; c += 2)
    bytes.push(String.fromCharCode(parseInt(hex.substr(c, 2), 16)));
  return bytes.join("");
};

/**
 * Computes the 'O' field in the encryption dictionary
 *
 * @name processOwnerPassword
 * @function
 * @param {String} paddedUserPassword Byte string of padded user password
 * @param {String} paddedOwnerPassword Byte string of padded owner password
 * @returns {String}
 */
PDFSecurity.prototype.processOwnerPassword = function(
  paddedUserPassword,
  paddedOwnerPassword
) {
  let key = md5Bin(paddedOwnerPassword).substr(0, 5);
  return rc4(key, paddedUserPassword);
};

/**
 * Returns an encryptor function which can take in a byte string and returns the encrypted version
 *
 * @name encryptor
 * @function
 * @param {number} objectId
 * @param {number} generation Not sure what this is for, you can set it to 0
 * @returns {Function}
 * @example
 * out("stream");
 * encryptor = security.encryptor(object.id, 0);
 * out(encryptor(data));
 * out("endstream");
 */
PDFSecurity.prototype.encryptor = function(objectId, generation) {
  let key = md5Bin(
    this.encryptionKey +
      String.fromCharCode(
        objectId & 0xff,
        (objectId >> 8) & 0xff,
        (objectId >> 16) & 0xff,
        generation & 0xff,
        (generation >> 8) & 0xff
      )
  ).substr(0, 10);
  return function(data) {
    return rc4(key, data);
  };
};

/**
 * Convert string to `PDF Name Object`.
 * Detail: PDF Reference 1.3 - Chapter 3.2.4 Name Object
 * @param str
 */
function toPDFName(str) {
  // eslint-disable-next-line no-control-regex
  if (/[^\u0000-\u00ff]/.test(str)) {
    // non ascii string
    throw new Error(
      "Invalid PDF Name Object: " + str + ", Only accept ASCII characters."
    );
  }
  var result = "",
    strLength = str.length;
  for (var i = 0; i < strLength; i++) {
    var charCode = str.charCodeAt(i);
    if (
      charCode < 0x21 ||
      charCode === 0x23 /* # */ ||
      charCode === 0x25 /* % */ ||
      charCode === 0x28 /* ( */ ||
      charCode === 0x29 /* ) */ ||
      charCode === 0x2f /* / */ ||
      charCode === 0x3c /* < */ ||
      charCode === 0x3e /* > */ ||
      charCode === 0x5b /* [ */ ||
      charCode === 0x5d /* ] */ ||
      charCode === 0x7b /* { */ ||
      charCode === 0x7d /* } */ ||
      charCode > 0x7e
    ) {
      // Char    CharCode    hexStr   paddingHexStr    Result
      // "\t"    9           9        09               #09
      // " "     32          20       20               #20
      // "©"     169         a9       a9               #a9
      var hexStr = charCode.toString(16),
        paddingHexStr = ("0" + hexStr).slice(-2);

      result += "#" + paddingHexStr;
    } else {
      // Other ASCII printable characters between 0x21 <= X <= 0x7e
      result += str[i];
    }
  }
  return result;
}

/* eslint-disable no-console */
/**
 * jsPDF's Internal PubSub Implementation.
 * Backward compatible rewritten on 2014 by
 * Diego Casorran, https://github.com/diegocr
 *
 * @class
 * @name PubSub
 * @ignore
 */
function PubSub(context) {
  if (typeof context !== "object") {
    throw new Error(
      "Invalid Context passed to initialize PubSub (jsPDF-module)"
    );
  }
  var topics = {};

  this.subscribe = function(topic, callback, once) {
    once = once || false;
    if (
      typeof topic !== "string" ||
      typeof callback !== "function" ||
      typeof once !== "boolean"
    ) {
      throw new Error(
        "Invalid arguments passed to PubSub.subscribe (jsPDF-module)"
      );
    }

    if (!topics.hasOwnProperty(topic)) {
      topics[topic] = {};
    }

    var token = Math.random().toString(35);
    topics[topic][token] = [callback, !!once];

    return token;
  };

  this.unsubscribe = function(token) {
    for (var topic in topics) {
      if (topics[topic][token]) {
        delete topics[topic][token];
        if (Object.keys(topics[topic]).length === 0) {
          delete topics[topic];
        }
        return true;
      }
    }
    return false;
  };

  this.publish = function(topic) {
    if (topics.hasOwnProperty(topic)) {
      var args = Array.prototype.slice.call(arguments, 1),
        tokens = [];

      for (var token in topics[topic]) {
        var sub = topics[topic][token];
        try {
          sub[0].apply(context, args);
        } catch (ex) {
          if (globalObject.console) {
            console.error("jsPDF PubSub Error", ex.message, ex);
          }
        }
        if (sub[1]) tokens.push(token);
      }
      if (tokens.length) tokens.forEach(this.unsubscribe);
    }
  };

  this.getTopics = function() {
    return topics;
  };
}

function GState(parameters) {
  if (!(this instanceof GState)) {
    return new GState(parameters);
  }

  /**
   * @name GState#opacity
   * @type {any}
   */
  /**
   * @name GState#stroke-opacity
   * @type {any}
   */
  var supported = "opacity,stroke-opacity".split(",");
  for (var p in parameters) {
    if (parameters.hasOwnProperty(p) && supported.indexOf(p) >= 0) {
      this[p] = parameters[p];
    }
  }
  /**
   * @name GState#id
   * @type {string}
   */
  this.id = ""; // set by addGState()
  /**
   * @name GState#objectNumber
   * @type {number}
   */
  this.objectNumber = -1; // will be set by putGState()
}

GState.prototype.equals = function equals(other) {
  var ignore = "id,objectNumber,equals";
  var p;
  if (!other || typeof other !== typeof this) return false;
  var count = 0;
  for (p in this) {
    if (ignore.indexOf(p) >= 0) continue;
    if (this.hasOwnProperty(p) && !other.hasOwnProperty(p)) return false;
    if (this[p] !== other[p]) return false;
    count++;
  }
  for (p in other) {
    if (other.hasOwnProperty(p) && ignore.indexOf(p) < 0) count--;
  }
  return count === 0;
};

function Pattern(gState, matrix) {
  this.gState = gState;
  this.matrix = matrix;

  this.id = ""; // set by addPattern()
  this.objectNumber = -1; // will be set by putPattern()
}

function ShadingPattern(type, coords, colors, gState, matrix) {
  if (!(this instanceof ShadingPattern)) {
    return new ShadingPattern(type, coords, colors, gState, matrix);
  }

  // see putPattern() for information how they are realized
  this.type = type === "axial" ? 2 : 3;
  this.coords = coords;
  this.colors = colors;

  Pattern.call(this, gState, matrix);
}

function TilingPattern(boundingBox, xStep, yStep, gState, matrix) {
  if (!(this instanceof TilingPattern)) {
    return new TilingPattern(boundingBox, xStep, yStep, gState, matrix);
  }

  this.boundingBox = boundingBox;
  this.xStep = xStep;
  this.yStep = yStep;

  this.stream = ""; // set by endTilingPattern();

  this.cloneIndex = 0;

  Pattern.call(this, gState, matrix);
}

/**
 * Creates new jsPDF document object instance.
 * @name jsPDF
 * @class
 * @param {Object} [options] - Collection of settings initializing the jsPDF-instance
 * @param {string} [options.orientation=portrait] - Orientation of the first page. Possible values are "portrait" or "landscape" (or shortcuts "p" or "l").<br />
 * @param {string} [options.unit=mm] Measurement unit (base unit) to be used when coordinates are specified.<br />
 * Possible values are "pt" (points), "mm", "cm", "in", "px", "pc", "em" or "ex". Note that in order to get the correct scaling for "px"
 * units, you need to enable the hotfix "px_scaling" by setting options.hotfixes = ["px_scaling"].
 * @param {string/Array} [options.format=a4] The format of the first page. Can be:<ul><li>a0 - a10</li><li>b0 - b10</li><li>c0 - c10</li><li>dl</li><li>letter</li><li>government-letter</li><li>legal</li><li>junior-legal</li><li>ledger</li><li>tabloid</li><li>credit-card</li></ul><br />
 * Default is "a4". If you want to use your own format just pass instead of one of the above predefined formats the size as an number-array, e.g. [595.28, 841.89]
 * @param {boolean} [options.putOnlyUsedFonts=false] Only put fonts into the PDF, which were used.
 * @param {boolean} [options.compress=false] Compress the generated PDF.
 * @param {number} [options.precision=16] Precision of the element-positions.
 * @param {number} [options.userUnit=1.0] Not to be confused with the base unit. Please inform yourself before you use it.
 * @param {string[]} [options.hotfixes] An array of strings to enable hotfixes such as correct pixel scaling.
 * @param {Object} [options.encryption]
 * @param {string} [options.encryption.userPassword] Password for the user bound by the given permissions list.
 * @param {string} [options.encryption.ownerPassword] Both userPassword and ownerPassword should be set for proper authentication.
 * @param {string[]} [options.encryption.userPermissions] Array of permissions "print", "modify", "copy", "annot-forms", accessible by the user.
 * @param {number|"smart"} [options.floatPrecision=16]
 * @returns {jsPDF} jsPDF-instance
 * @description
 * ```
 * {
 *  orientation: 'p',
 *  unit: 'mm',
 *  format: 'a4',
 *  putOnlyUsedFonts:true,
 *  floatPrecision: 16 // or "smart", default is 16
 * }
 * ```
 *
 * @constructor
 */
function jsPDF(options) {
  var orientation = typeof arguments[0] === "string" ? arguments[0] : "p";
  var unit = arguments[1];
  var format = arguments[2];
  var compressPdf = arguments[3];
  var filters = [];
  var userUnit = 1.0;
  var precision;
  var floatPrecision = 16;
  var defaultPathOperation = "S";
  var encryptionOptions = null;

  options = options || {};

  if (typeof options === "object") {
    orientation = options.orientation;
    unit = options.unit || unit;
    format = options.format || format;
    compressPdf = options.compress || options.compressPdf || compressPdf;
    encryptionOptions = options.encryption || null;
    if (encryptionOptions !== null) {
      encryptionOptions.userPassword = encryptionOptions.userPassword || "";
      encryptionOptions.ownerPassword = encryptionOptions.ownerPassword || "";
      encryptionOptions.userPermissions =
        encryptionOptions.userPermissions || [];
    }
    userUnit =
      typeof options.userUnit === "number" ? Math.abs(options.userUnit) : 1.0;
    if (typeof options.precision !== "undefined") {
      precision = options.precision;
    }
    if (typeof options.floatPrecision !== "undefined") {
      floatPrecision = options.floatPrecision;
    }
    defaultPathOperation = options.defaultPathOperation || "S";
  }

  filters =
    options.filters || (compressPdf === true ? ["FlateEncode"] : filters);

  unit = unit || "mm";
  orientation = ("" + (orientation || "P")).toLowerCase();
  var putOnlyUsedFonts = options.putOnlyUsedFonts || false;
  var usedFonts = {};

  var API = {
    internal: {},
    __private__: {}
  };

  API.__private__.PubSub = PubSub;

  var pdfVersion = "1.3";
  var getPdfVersion = (API.__private__.getPdfVersion = function() {
    return pdfVersion;
  });

  API.__private__.setPdfVersion = function(value) {
    pdfVersion = value;
  };

  // Size in pt of various paper formats
  var pageFormats = {
    a0: [2383.94, 3370.39],
    a1: [1683.78, 2383.94],
    a2: [1190.55, 1683.78],
    a3: [841.89, 1190.55],
    a4: [595.28, 841.89],
    a5: [419.53, 595.28],
    a6: [297.64, 419.53],
    a7: [209.76, 297.64],
    a8: [147.4, 209.76],
    a9: [104.88, 147.4],
    a10: [73.7, 104.88],
    b0: [2834.65, 4008.19],
    b1: [2004.09, 2834.65],
    b2: [1417.32, 2004.09],
    b3: [1000.63, 1417.32],
    b4: [708.66, 1000.63],
    b5: [498.9, 708.66],
    b6: [354.33, 498.9],
    b7: [249.45, 354.33],
    b8: [175.75, 249.45],
    b9: [124.72, 175.75],
    b10: [87.87, 124.72],
    c0: [2599.37, 3676.54],
    c1: [1836.85, 2599.37],
    c2: [1298.27, 1836.85],
    c3: [918.43, 1298.27],
    c4: [649.13, 918.43],
    c5: [459.21, 649.13],
    c6: [323.15, 459.21],
    c7: [229.61, 323.15],
    c8: [161.57, 229.61],
    c9: [113.39, 161.57],
    c10: [79.37, 113.39],
    dl: [311.81, 623.62],
    letter: [612, 792],
    "government-letter": [576, 756],
    legal: [612, 1008],
    "junior-legal": [576, 360],
    ledger: [1224, 792],
    tabloid: [792, 1224],
    "credit-card": [153, 243]
  };

  API.__private__.getPageFormats = function() {
    return pageFormats;
  };

  var getPageFormat = (API.__private__.getPageFormat = function(value) {
    return pageFormats[value];
  });

  format = format || "a4";

  var ApiMode = {
    COMPAT: "compat",
    ADVANCED: "advanced"
  };
  var apiMode = ApiMode.COMPAT;

  function advancedAPI() {
    // prepend global change of basis matrix
    // (Now, instead of converting every coordinate to the pdf coordinate system, we apply a matrix
    // that does this job for us (however, texts, images and similar objects must be drawn bottom up))
    this.saveGraphicsState();
    out(
      new Matrix(
        scaleFactor,
        0,
        0,
        -scaleFactor,
        0,
        getPageHeight() * scaleFactor
      ).toString() + " cm"
    );
    this.setFontSize(this.getFontSize() / scaleFactor);

    // The default in MrRio's implementation is "S" (stroke), whereas the default in the yWorks implementation
    // was "n" (none). Although this has nothing to do with transforms, we should use the API switch here.
    defaultPathOperation = "n";

    apiMode = ApiMode.ADVANCED;
  }

  function compatAPI() {
    this.restoreGraphicsState();
    defaultPathOperation = "S";
    apiMode = ApiMode.COMPAT;
  }

  /**
   * @function combineFontStyleAndFontWeight
   * @param {string} fontStyle Fontstyle or variant. Example: "italic".
   * @param {number | string} fontWeight Weight of the Font. Example: "normal" | 400
   * @returns {string}
   * @private
   */
  var combineFontStyleAndFontWeight = (API.__private__.combineFontStyleAndFontWeight = function(
    fontStyle,
    fontWeight
  ) {
    if (
      (fontStyle == "bold" && fontWeight == "normal") ||
      (fontStyle == "bold" && fontWeight == 400) ||
      (fontStyle == "normal" && fontWeight == "italic") ||
      (fontStyle == "bold" && fontWeight == "italic")
    ) {
      throw new Error("Invalid Combination of fontweight and fontstyle");
    }
    if (fontWeight) {
      fontStyle =
        fontWeight == 400 || fontWeight === "normal"
          ? fontStyle === "italic"
            ? "italic"
            : "normal"
          : (fontWeight == 700 || fontWeight === "bold") &&
            fontStyle === "normal"
          ? "bold"
          : (fontWeight == 700 ? "bold" : fontWeight) + "" + fontStyle;
    }
    return fontStyle;
  });

  /**
   * @callback ApiSwitchBody
   * @param {jsPDF} pdf
   */

  /**
   * For compatibility reasons jsPDF offers two API modes which differ in the way they convert between the the usual
   * screen coordinates and the PDF coordinate system.
   *   - "compat": Offers full compatibility across all plugins but does not allow arbitrary transforms
   *   - "advanced": Allows arbitrary transforms and more advanced features like pattern fills. Some plugins might
   *     not support this mode, though.
   * Initial mode is "compat".
   *
   * You can either provide a callback to the body argument, which means that jsPDF will automatically switch back to
   * the original API mode afterwards; or you can omit the callback and switch back manually using {@link compatAPI}.
   *
   * Note, that the calls to {@link saveGraphicsState} and {@link restoreGraphicsState} need to be balanced within the
   * callback or between calls of this method and its counterpart {@link compatAPI}. Calls to {@link beginFormObject}
   * or {@link beginTilingPattern} need to be closed by their counterparts before switching back to "compat" API mode.
   *
   * @param {ApiSwitchBody=} body When provided, this callback will be called after the API mode has been switched.
   * The API mode will be switched back automatically afterwards.
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name advancedAPI
   */
  API.advancedAPI = function(body) {
    var doSwitch = apiMode === ApiMode.COMPAT;

    if (doSwitch) {
      advancedAPI.call(this);
    }

    if (typeof body !== "function") {
      return this;
    }

    body(this);

    if (doSwitch) {
      compatAPI.call(this);
    }

    return this;
  };

  /**
   * Switches to "compat" API mode. See {@link advancedAPI} for more details.
   *
   * @param {ApiSwitchBody=} body When provided, this callback will be called after the API mode has been switched.
   * The API mode will be switched back automatically afterwards.
   * @return {jsPDF}
   * @memberof jsPDF#
   * @name compatApi
   */
  API.compatAPI = function(body) {
    var doSwitch = apiMode === ApiMode.ADVANCED;

    if (doSwitch) {
      compatAPI.call(this);
    }

    if (typeof body !== "function") {
      return this;
    }

    body(this);

    if (doSwitch) {
      advancedAPI.call(this);
    }

    return this;
  };

  /**
   * @return {boolean} True iff the current API mode is "advanced". See {@link advancedAPI}.
   * @memberof jsPDF#
   * @name isAdvancedAPI
   */
  API.isAdvancedAPI = function() {
    return apiMode === ApiMode.ADVANCED;
  };

  var advancedApiModeTrap = function(methodName) {
    if (apiMode !== ApiMode.ADVANCED) {
      throw new Error(
        methodName +
          " is only available in 'advanced' API mode. " +
          "You need to call advancedAPI() first."
      );
    }
  };

  var roundToPrecision = (API.roundToPrecision = API.__private__.roundToPrecision = function(
    number,
    parmPrecision
  ) {
    var tmpPrecision = precision || parmPrecision;
    if (isNaN(number) || isNaN(tmpPrecision)) {
      throw new Error("Invalid argument passed to jsPDF.roundToPrecision");
    }
    return number.toFixed(tmpPrecision).replace(/0+$/, "");
  });

  // high precision float
  var hpf;
  if (typeof floatPrecision === "number") {
    hpf = API.hpf = API.__private__.hpf = function(number) {
      if (isNaN(number)) {
        throw new Error("Invalid argument passed to jsPDF.hpf");
      }
      return roundToPrecision(number, floatPrecision);
    };
  } else if (floatPrecision === "smart") {
    hpf = API.hpf = API.__private__.hpf = function(number) {
      if (isNaN(number)) {
        throw new Error("Invalid argument passed to jsPDF.hpf");
      }
      if (number > -1 && number < 1) {
        return roundToPrecision(number, 16);
      } else {
        return roundToPrecision(number, 5);
      }
    };
  } else {
    hpf = API.hpf = API.__private__.hpf = function(number) {
      if (isNaN(number)) {
        throw new Error("Invalid argument passed to jsPDF.hpf");
      }
      return roundToPrecision(number, 16);
    };
  }
  var f2 = (API.f2 = API.__private__.f2 = function(number) {
    if (isNaN(number)) {
      throw new Error("Invalid argument passed to jsPDF.f2");
    }
    return roundToPrecision(number, 2);
  });

  var f3 = (API.__private__.f3 = function(number) {
    if (isNaN(number)) {
      throw new Error("Invalid argument passed to jsPDF.f3");
    }
    return roundToPrecision(number, 3);
  });

  var scale = (API.scale = API.__private__.scale = function(number) {
    if (isNaN(number)) {
      throw new Error("Invalid argument passed to jsPDF.scale");
    }
    if (apiMode === ApiMode.COMPAT) {
      return number * scaleFactor;
    } else if (apiMode === ApiMode.ADVANCED) {
      return number;
    }
  });

  var transformY = function(y) {
    if (apiMode === ApiMode.COMPAT) {
      return getPageHeight() - y;
    } else if (apiMode === ApiMode.ADVANCED) {
      return y;
    }
  };

  var transformScaleY = function(y) {
    return scale(transformY(y));
  };

  /**
   * @name setPrecision
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {string} precision
   * @returns {jsPDF}
   */
  API.__private__.setPrecision = API.setPrecision = function(value) {
    if (typeof parseInt(value, 10) === "number") {
      precision = parseInt(value, 10);
    }
  };

  var fileId = "00000000000000000000000000000000";

  var getFileId = (API.__private__.getFileId = function() {
    return fileId;
  });

  var setFileId = (API.__private__.setFileId = function(value) {
    if (typeof value !== "undefined" && /^[a-fA-F0-9]{32}$/.test(value)) {
      fileId = value.toUpperCase();
    } else {
      fileId = fileId
        .split("")
        .map(function() {
          return "ABCDEF0123456789".charAt(Math.floor(Math.random() * 16));
        })
        .join("");
    }

    if (encryptionOptions !== null) {
      encryption = new PDFSecurity(
        encryptionOptions.userPermissions,
        encryptionOptions.userPassword,
        encryptionOptions.ownerPassword,
        fileId
      );
    }
    return fileId;
  });

  /**
   * @name setFileId
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {string} value GUID.
   * @returns {jsPDF}
   */
  API.setFileId = function(value) {
    setFileId(value);
    return this;
  };

  /**
   * @name getFileId
   * @memberof jsPDF#
   * @function
   * @instance
   *
   * @returns {string} GUID.
   */
  API.getFileId = function() {
    return getFileId();
  };

  var creationDate;

  var convertDateToPDFDate = (API.__private__.convertDateToPDFDate = function(
    parmDate
  ) {
    var result = "";
    var tzoffset = parmDate.getTimezoneOffset(),
      tzsign = tzoffset < 0 ? "+" : "-",
      tzhour = Math.floor(Math.abs(tzoffset / 60)),
      tzmin = Math.abs(tzoffset % 60),
      timeZoneString = [tzsign, padd2(tzhour), "'", padd2(tzmin), "'"].join("");

    result = [
      "D:",
      parmDate.getFullYear(),
      padd2(parmDate.getMonth() + 1),
      padd2(parmDate.getDate()),
      padd2(parmDate.getHours()),
      padd2(parmDate.getMinutes()),
      padd2(parmDate.getSeconds()),
      timeZoneString
    ].join("");
    return result;
  });

  var convertPDFDateToDate = (API.__private__.convertPDFDateToDate = function(
    parmPDFDate
  ) {
    var year = parseInt(parmPDFDate.substr(2, 4), 10);
    var month = parseInt(parmPDFDate.substr(6, 2), 10) - 1;
    var date = parseInt(parmPDFDate.substr(8, 2), 10);
    var hour = parseInt(parmPDFDate.substr(10, 2), 10);
    var minutes = parseInt(parmPDFDate.substr(12, 2), 10);
    var seconds = parseInt(parmPDFDate.substr(14, 2), 10);
    // var timeZoneHour = parseInt(parmPDFDate.substr(16, 2), 10);
    // var timeZoneMinutes = parseInt(parmPDFDate.substr(20, 2), 10);

    var resultingDate = new Date(year, month, date, hour, minutes, seconds, 0);
    return resultingDate;
  });

  var setCreationDate = (API.__private__.setCreationDate = function(date) {
    var tmpCreationDateString;
    var regexPDFCreationDate = /^D:(20[0-2][0-9]|203[0-7]|19[7-9][0-9])(0[0-9]|1[0-2])([0-2][0-9]|3[0-1])(0[0-9]|1[0-9]|2[0-3])(0[0-9]|[1-5][0-9])(0[0-9]|[1-5][0-9])(\+0[0-9]|\+1[0-4]|-0[0-9]|-1[0-1])'(0[0-9]|[1-5][0-9])'?$/;
    if (typeof date === "undefined") {
      date = new Date();
    }

    if (date instanceof Date) {
      tmpCreationDateString = convertDateToPDFDate(date);
    } else if (regexPDFCreationDate.test(date)) {
      tmpCreationDateString = date;
    } else {
      throw new Error("Invalid argument passed to jsPDF.setCreationDate");
    }
    creationDate = tmpCreationDateString;
    return creationDate;
  });

  var getCreationDate = (API.__private__.getCreationDate = function(type) {
    var result = creationDate;
    if (type === "jsDate") {
      result = convertPDFDateToDate(creationDate);
    }
    return result;
  });

  /**
   * @name setCreationDate
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {Object} date
   * @returns {jsPDF}
   */
  API.setCreationDate = function(date) {
    setCreationDate(date);
    return this;
  };

  /**
   * @name getCreationDate
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {Object} type
   * @returns {Object}
   */
  API.getCreationDate = function(type) {
    return getCreationDate(type);
  };

  var padd2 = (API.__private__.padd2 = function(number) {
    return ("0" + parseInt(number)).slice(-2);
  });

  var padd2Hex = (API.__private__.padd2Hex = function(hexString) {
    hexString = hexString.toString();
    return ("00" + hexString).substr(hexString.length);
  });

  var objectNumber = 0; // 'n' Current object number
  var offsets = []; // List of offsets. Activated and reset by buildDocument(). Pupulated by various calls buildDocument makes.
  var content = [];
  var contentLength = 0;
  var additionalObjects = [];

  var pages = [];
  var currentPage;
  var hasCustomDestination = false;
  var outputDestination = content;

  var resetDocument = function() {
    //reset fields relevant for objectNumber generation and xref.
    objectNumber = 0;
    contentLength = 0;
    content = [];
    offsets = [];
    additionalObjects = [];

    rootDictionaryObjId = newObjectDeferred();
    resourceDictionaryObjId = newObjectDeferred();
  };

  API.__private__.setCustomOutputDestination = function(destination) {
    hasCustomDestination = true;
    outputDestination = destination;
  };
  var setOutputDestination = function(destination) {
    if (!hasCustomDestination) {
      outputDestination = destination;
    }
  };

  API.__private__.resetCustomOutputDestination = function() {
    hasCustomDestination = false;
    outputDestination = content;
  };

  var out = (API.__private__.out = function(string) {
    string = string.toString();
    contentLength += string.length + 1;
    outputDestination.push(string);

    return outputDestination;
  });

  var write = (API.__private__.write = function(value) {
    return out(
      arguments.length === 1
        ? value.toString()
        : Array.prototype.join.call(arguments, " ")
    );
  });

  var getArrayBuffer = (API.__private__.getArrayBuffer = function(data) {
    var len = data.length,
      ab = new ArrayBuffer(len),
      u8 = new Uint8Array(ab);

    while (len--) u8[len] = data.charCodeAt(len);
    return ab;
  });

  var standardFonts = [
    ["Helvetica", "helvetica", "normal", "WinAnsiEncoding"],
    ["Helvetica-Bold", "helvetica", "bold", "WinAnsiEncoding"],
    ["Helvetica-Oblique", "helvetica", "italic", "WinAnsiEncoding"],
    ["Helvetica-BoldOblique", "helvetica", "bolditalic", "WinAnsiEncoding"],
    ["Courier", "courier", "normal", "WinAnsiEncoding"],
    ["Courier-Bold", "courier", "bold", "WinAnsiEncoding"],
    ["Courier-Oblique", "courier", "italic", "WinAnsiEncoding"],
    ["Courier-BoldOblique", "courier", "bolditalic", "WinAnsiEncoding"],
    ["Times-Roman", "times", "normal", "WinAnsiEncoding"],
    ["Times-Bold", "times", "bold", "WinAnsiEncoding"],
    ["Times-Italic", "times", "italic", "WinAnsiEncoding"],
    ["Times-BoldItalic", "times", "bolditalic", "WinAnsiEncoding"],
    ["ZapfDingbats", "zapfdingbats", "normal", null],
    ["Symbol", "symbol", "normal", null]
  ];

  API.__private__.getStandardFonts = function() {
    return standardFonts;
  };

  var activeFontSize = options.fontSize || 16;

  /**
   * Sets font size for upcoming text elements.
   *
   * @param {number} size Font size in points.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setFontSize
   */
  API.__private__.setFontSize = API.setFontSize = function(size) {
    if (apiMode === ApiMode.ADVANCED) {
      activeFontSize = size / scaleFactor;
    } else {
      activeFontSize = size;
    }
    return this;
  };

  /**
   * Gets the fontsize for upcoming text elements.
   *
   * @function
   * @instance
   * @returns {number}
   * @memberof jsPDF#
   * @name getFontSize
   */
  var getFontSize = (API.__private__.getFontSize = API.getFontSize = function() {
    if (apiMode === ApiMode.COMPAT) {
      return activeFontSize;
    } else {
      return activeFontSize * scaleFactor;
    }
  });

  var R2L = options.R2L || false;

  /**
   * Set value of R2L functionality.
   *
   * @param {boolean} value
   * @function
   * @instance
   * @returns {jsPDF} jsPDF-instance
   * @memberof jsPDF#
   * @name setR2L
   */
  API.__private__.setR2L = API.setR2L = function(value) {
    R2L = value;
    return this;
  };

  /**
   * Get value of R2L functionality.
   *
   * @function
   * @instance
   * @returns {boolean} jsPDF-instance
   * @memberof jsPDF#
   * @name getR2L
   */
  API.__private__.getR2L = API.getR2L = function() {
    return R2L;
  };

  var zoomMode; // default: 1;

  var setZoomMode = (API.__private__.setZoomMode = function(zoom) {
    var validZoomModes = [
      undefined,
      null,
      "fullwidth",
      "fullheight",
      "fullpage",
      "original"
    ];

    if (/^(?:\d+\.\d*|\d*\.\d+|\d+)%$/.test(zoom)) {
      zoomMode = zoom;
    } else if (!isNaN(zoom)) {
      zoomMode = parseInt(zoom, 10);
    } else if (validZoomModes.indexOf(zoom) !== -1) {
      zoomMode = zoom;
    } else {
      throw new Error(
        'zoom must be Integer (e.g. 2), a percentage Value (e.g. 300%) or fullwidth, fullheight, fullpage, original. "' +
          zoom +
          '" is not recognized.'
      );
    }
  });

  API.__private__.getZoomMode = function() {
    return zoomMode;
  };

  var pageMode; // default: 'UseOutlines';
  var setPageMode = (API.__private__.setPageMode = function(pmode) {
    var validPageModes = [
      undefined,
      null,
      "UseNone",
      "UseOutlines",
      "UseThumbs",
      "FullScreen"
    ];

    if (validPageModes.indexOf(pmode) == -1) {
      throw new Error(
        'Page mode must be one of UseNone, UseOutlines, UseThumbs, or FullScreen. "' +
          pmode +
          '" is not recognized.'
      );
    }
    pageMode = pmode;
  });

  API.__private__.getPageMode = function() {
    return pageMode;
  };

  var layoutMode; // default: 'continuous';
  var setLayoutMode = (API.__private__.setLayoutMode = function(layout) {
    var validLayoutModes = [
      undefined,
      null,
      "continuous",
      "single",
      "twoleft",
      "tworight",
      "two"
    ];

    if (validLayoutModes.indexOf(layout) == -1) {
      throw new Error(
        'Layout mode must be one of continuous, single, twoleft, tworight. "' +
          layout +
          '" is not recognized.'
      );
    }
    layoutMode = layout;
  });

  API.__private__.getLayoutMode = function() {
    return layoutMode;
  };

  /**
   * Set the display mode options of the page like zoom and layout.
   *
   * @name setDisplayMode
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {integer|String} zoom   You can pass an integer or percentage as
   * a string. 2 will scale the document up 2x, '200%' will scale up by the
   * same amount. You can also set it to 'fullwidth', 'fullheight',
   * 'fullpage', or 'original'.
   *
   * Only certain PDF readers support this, such as Adobe Acrobat.
   *
   * @param {string} layout Layout mode can be: 'continuous' - this is the
   * default continuous scroll. 'single' - the single page mode only shows one
   * page at a time. 'twoleft' - two column left mode, first page starts on
   * the left, and 'tworight' - pages are laid out in two columns, with the
   * first page on the right. This would be used for books.
   * @param {string} pmode 'UseOutlines' - it shows the
   * outline of the document on the left. 'UseThumbs' - shows thumbnails along
   * the left. 'FullScreen' - prompts the user to enter fullscreen mode.
   *
   * @returns {jsPDF}
   */
  API.__private__.setDisplayMode = API.setDisplayMode = function(
    zoom,
    layout,
    pmode
  ) {
    setZoomMode(zoom);
    setLayoutMode(layout);
    setPageMode(pmode);
    return this;
  };

  var documentProperties = {
    title: "",
    subject: "",
    author: "",
    keywords: "",
    creator: ""
  };

  API.__private__.getDocumentProperty = function(key) {
    if (Object.keys(documentProperties).indexOf(key) === -1) {
      throw new Error("Invalid argument passed to jsPDF.getDocumentProperty");
    }
    return documentProperties[key];
  };

  API.__private__.getDocumentProperties = function() {
    return documentProperties;
  };

  /**
   * Adds a properties to the PDF document.
   *
   * @param {Object} A property_name-to-property_value object structure.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setDocumentProperties
   */
  API.__private__.setDocumentProperties = API.setProperties = API.setDocumentProperties = function(
    properties
  ) {
    // copying only those properties we can render.
    for (var property in documentProperties) {
      if (documentProperties.hasOwnProperty(property) && properties[property]) {
        documentProperties[property] = properties[property];
      }
    }
    return this;
  };

  API.__private__.setDocumentProperty = function(key, value) {
    if (Object.keys(documentProperties).indexOf(key) === -1) {
      throw new Error("Invalid arguments passed to jsPDF.setDocumentProperty");
    }
    return (documentProperties[key] = value);
  };

  var fonts = {}; // collection of font objects, where key is fontKey - a dynamically created label for a given font.
  var fontmap = {}; // mapping structure fontName > fontStyle > font key - performance layer. See addFont()
  var activeFontKey; // will be string representing the KEY of the font as combination of fontName + fontStyle
  var fontStateStack = []; //
  var patterns = {}; // collection of pattern objects
  var patternMap = {}; // see fonts
  var gStates = {}; // collection of graphic state objects
  var gStatesMap = {}; // see fonts
  var activeGState = null;
  var scaleFactor; // Scale factor
  var page = 0;
  var pagesContext = [];
  var events = new PubSub(API);
  var hotfixes = options.hotfixes || [];

  var renderTargets = {};
  var renderTargetMap = {};
  var renderTargetStack = [];
  var pageX;
  var pageY;
  var pageMatrix; // only used for FormObjects

  /**
   * A matrix object for 2D homogenous transformations: <br>
   * | a b 0 | <br>
   * | c d 0 | <br>
   * | e f 1 | <br>
   * pdf multiplies matrices righthand: v' = v x m1 x m2 x ...
   *
   * @class
   * @name Matrix
   * @param {number} sx
   * @param {number} shy
   * @param {number} shx
   * @param {number} sy
   * @param {number} tx
   * @param {number} ty
   * @constructor
   */
  var Matrix = function(sx, shy, shx, sy, tx, ty) {
    if (!(this instanceof Matrix)) {
      return new Matrix(sx, shy, shx, sy, tx, ty);
    }

    if (isNaN(sx)) sx = 1;
    if (isNaN(shy)) shy = 0;
    if (isNaN(shx)) shx = 0;
    if (isNaN(sy)) sy = 1;
    if (isNaN(tx)) tx = 0;
    if (isNaN(ty)) ty = 0;

    this._matrix = [sx, shy, shx, sy, tx, ty];
  };

  /**
   * @name sx
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "sx", {
    get: function() {
      return this._matrix[0];
    },
    set: function(value) {
      this._matrix[0] = value;
    }
  });

  /**
   * @name shy
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "shy", {
    get: function() {
      return this._matrix[1];
    },
    set: function(value) {
      this._matrix[1] = value;
    }
  });

  /**
   * @name shx
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "shx", {
    get: function() {
      return this._matrix[2];
    },
    set: function(value) {
      this._matrix[2] = value;
    }
  });

  /**
   * @name sy
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "sy", {
    get: function() {
      return this._matrix[3];
    },
    set: function(value) {
      this._matrix[3] = value;
    }
  });

  /**
   * @name tx
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "tx", {
    get: function() {
      return this._matrix[4];
    },
    set: function(value) {
      this._matrix[4] = value;
    }
  });

  /**
   * @name ty
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "ty", {
    get: function() {
      return this._matrix[5];
    },
    set: function(value) {
      this._matrix[5] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "a", {
    get: function() {
      return this._matrix[0];
    },
    set: function(value) {
      this._matrix[0] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "b", {
    get: function() {
      return this._matrix[1];
    },
    set: function(value) {
      this._matrix[1] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "c", {
    get: function() {
      return this._matrix[2];
    },
    set: function(value) {
      this._matrix[2] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "d", {
    get: function() {
      return this._matrix[3];
    },
    set: function(value) {
      this._matrix[3] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "e", {
    get: function() {
      return this._matrix[4];
    },
    set: function(value) {
      this._matrix[4] = value;
    }
  });

  Object.defineProperty(Matrix.prototype, "f", {
    get: function() {
      return this._matrix[5];
    },
    set: function(value) {
      this._matrix[5] = value;
    }
  });

  /**
   * @name rotation
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "rotation", {
    get: function() {
      return Math.atan2(this.shx, this.sx);
    }
  });

  /**
   * @name scaleX
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "scaleX", {
    get: function() {
      return this.decompose().scale.sx;
    }
  });

  /**
   * @name scaleY
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "scaleY", {
    get: function() {
      return this.decompose().scale.sy;
    }
  });

  /**
   * @name isIdentity
   * @memberof Matrix#
   */
  Object.defineProperty(Matrix.prototype, "isIdentity", {
    get: function() {
      if (this.sx !== 1) {
        return false;
      }
      if (this.shy !== 0) {
        return false;
      }
      if (this.shx !== 0) {
        return false;
      }
      if (this.sy !== 1) {
        return false;
      }
      if (this.tx !== 0) {
        return false;
      }
      if (this.ty !== 0) {
        return false;
      }
      return true;
    }
  });

  /**
   * Join the Matrix Values to a String
   *
   * @function join
   * @param {string} separator Specifies a string to separate each pair of adjacent elements of the array. The separator is converted to a string if necessary. If omitted, the array elements are separated with a comma (","). If separator is an empty string, all elements are joined without any characters in between them.
   * @returns {string} A string with all array elements joined.
   * @memberof Matrix#
   */
  Matrix.prototype.join = function(separator) {
    return [this.sx, this.shy, this.shx, this.sy, this.tx, this.ty]
      .map(hpf)
      .join(separator);
  };

  /**
   * Multiply the matrix with given Matrix
   *
   * @function multiply
   * @param matrix
   * @returns {Matrix}
   * @memberof Matrix#
   */
  Matrix.prototype.multiply = function(matrix) {
    var sx = matrix.sx * this.sx + matrix.shy * this.shx;
    var shy = matrix.sx * this.shy + matrix.shy * this.sy;
    var shx = matrix.shx * this.sx + matrix.sy * this.shx;
    var sy = matrix.shx * this.shy + matrix.sy * this.sy;
    var tx = matrix.tx * this.sx + matrix.ty * this.shx + this.tx;
    var ty = matrix.tx * this.shy + matrix.ty * this.sy + this.ty;

    return new Matrix(sx, shy, shx, sy, tx, ty);
  };

  /**
   * @function decompose
   * @memberof Matrix#
   */
  Matrix.prototype.decompose = function() {
    var a = this.sx;
    var b = this.shy;
    var c = this.shx;
    var d = this.sy;
    var e = this.tx;
    var f = this.ty;

    var scaleX = Math.sqrt(a * a + b * b);
    a /= scaleX;
    b /= scaleX;

    var shear = a * c + b * d;
    c -= a * shear;
    d -= b * shear;

    var scaleY = Math.sqrt(c * c + d * d);
    c /= scaleY;
    d /= scaleY;
    shear /= scaleY;

    if (a * d < b * c) {
      a = -a;
      b = -b;
      shear = -shear;
      scaleX = -scaleX;
    }

    return {
      scale: new Matrix(scaleX, 0, 0, scaleY, 0, 0),
      translate: new Matrix(1, 0, 0, 1, e, f),
      rotate: new Matrix(a, b, -b, a, 0, 0),
      skew: new Matrix(1, 0, shear, 1, 0, 0)
    };
  };

  /**
   * @function toString
   * @memberof Matrix#
   */
  Matrix.prototype.toString = function(parmPrecision) {
    return this.join(" ");
  };

  /**
   * @function inversed
   * @memberof Matrix#
   */
  Matrix.prototype.inversed = function() {
    var a = this.sx,
      b = this.shy,
      c = this.shx,
      d = this.sy,
      e = this.tx,
      f = this.ty;

    var quot = 1 / (a * d - b * c);

    var aInv = d * quot;
    var bInv = -b * quot;
    var cInv = -c * quot;
    var dInv = a * quot;
    var eInv = -aInv * e - cInv * f;
    var fInv = -bInv * e - dInv * f;

    return new Matrix(aInv, bInv, cInv, dInv, eInv, fInv);
  };

  /**
   * @function applyToPoint
   * @memberof Matrix#
   */
  Matrix.prototype.applyToPoint = function(pt) {
    var x = pt.x * this.sx + pt.y * this.shx + this.tx;
    var y = pt.x * this.shy + pt.y * this.sy + this.ty;
    return new Point(x, y);
  };

  /**
   * @function applyToRectangle
   * @memberof Matrix#
   */
  Matrix.prototype.applyToRectangle = function(rect) {
    var pt1 = this.applyToPoint(rect);
    var pt2 = this.applyToPoint(new Point(rect.x + rect.w, rect.y + rect.h));
    return new Rectangle(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
  };

  /**
   * Clone the Matrix
   *
   * @function clone
   * @memberof Matrix#
   * @name clone
   * @instance
   */
  Matrix.prototype.clone = function() {
    var sx = this.sx;
    var shy = this.shy;
    var shx = this.shx;
    var sy = this.sy;
    var tx = this.tx;
    var ty = this.ty;

    return new Matrix(sx, shy, shx, sy, tx, ty);
  };

  API.Matrix = Matrix;

  /**
   * Multiplies two matrices. (see {@link Matrix})
   * @param {Matrix} m1
   * @param {Matrix} m2
   * @memberof jsPDF#
   * @name matrixMult
   */
  var matrixMult = (API.matrixMult = function(m1, m2) {
    return m2.multiply(m1);
  });

  /**
   * The identity matrix (equivalent to new Matrix(1, 0, 0, 1, 0, 0)).
   * @type {Matrix}
   * @memberof! jsPDF#
   * @name identityMatrix
   */
  var identityMatrix = new Matrix(1, 0, 0, 1, 0, 0);
  API.unitMatrix = API.identityMatrix = identityMatrix;

  /**
   * Adds a new pattern for later use.
   * @param {String} key The key by it can be referenced later. The keys must be unique!
   * @param {API.Pattern} pattern The pattern
   */
  var addPattern = function(key, pattern) {
    // only add it if it is not already present (the keys provided by the user must be unique!)
    if (patternMap[key]) return;

    var prefix = pattern instanceof ShadingPattern ? "Sh" : "P";
    var patternKey = prefix + (Object.keys(patterns).length + 1).toString(10);
    pattern.id = patternKey;

    patternMap[key] = patternKey;
    patterns[patternKey] = pattern;

    events.publish("addPattern", pattern);
  };

  /**
   * A pattern describing a shading pattern.
   *
   * Only available in "advanced" API mode.
   *
   * @param {String} type One of "axial" or "radial"
   * @param {Array<Number>} coords Either [x1, y1, x2, y2] for "axial" type describing the two interpolation points
   * or [x1, y1, r, x2, y2, r2] for "radial" describing inner and the outer circle.
   * @param {Array<Object>} colors An array of objects with the fields "offset" and "color". "offset" describes
   * the offset in parameter space [0, 1]. "color" is an array of length 3 describing RGB values in [0, 255].
   * @param {GState=} gState An additional graphics state that gets applied to the pattern (optional).
   * @param {Matrix=} matrix A matrix that describes the transformation between the pattern coordinate system
   * and the use coordinate system (optional).
   * @constructor
   * @extends API.Pattern
   */
  API.ShadingPattern = ShadingPattern;

  /**
   * A PDF Tiling pattern.
   *
   * Only available in "advanced" API mode.
   *
   * @param {Array.<Number>} boundingBox The bounding box at which one pattern cell gets clipped.
   * @param {Number} xStep Horizontal spacing between pattern cells.
   * @param {Number} yStep Vertical spacing between pattern cells.
   * @param {API.GState=} gState An additional graphics state that gets applied to the pattern (optional).
   * @param {Matrix=} matrix A matrix that describes the transformation between the pattern coordinate system
   * and the use coordinate system (optional).
   * @constructor
   * @extends API.Pattern
   */
  API.TilingPattern = TilingPattern;

  /**
   * Adds a new {@link API.ShadingPattern} for later use. Only available in "advanced" API mode.
   * @param {String} key
   * @param {Pattern} pattern
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name addPattern
   */
  API.addShadingPattern = function(key, pattern) {
    advancedApiModeTrap("addShadingPattern()");

    addPattern(key, pattern);
    return this;
  };

  /**
   * Begins a new tiling pattern. All subsequent render calls are drawn to this pattern until {@link API.endTilingPattern}
   * gets called. Only available in "advanced" API mode.
   * @param {API.Pattern} pattern
   * @memberof jsPDF#
   * @name beginTilingPattern
   */
  API.beginTilingPattern = function(pattern) {
    advancedApiModeTrap("beginTilingPattern()");

    beginNewRenderTarget(
      pattern.boundingBox[0],
      pattern.boundingBox[1],
      pattern.boundingBox[2] - pattern.boundingBox[0],
      pattern.boundingBox[3] - pattern.boundingBox[1],
      pattern.matrix
    );
  };

  /**
   * Ends a tiling pattern and sets the render target to the one active before {@link API.beginTilingPattern} has been called.
   *
   * Only available in "advanced" API mode.
   *
   * @param {string} key A unique key that is used to reference this pattern at later use.
   * @param {API.Pattern} pattern The pattern to end.
   * @memberof jsPDF#
   * @name endTilingPattern
   */
  API.endTilingPattern = function(key, pattern) {
    advancedApiModeTrap("endTilingPattern()");

    // retrieve the stream
    pattern.stream = pages[currentPage].join("\n");

    addPattern(key, pattern);

    events.publish("endTilingPattern", pattern);

    // restore state from stack
    renderTargetStack.pop().restore();
  };

  var newObject = (API.__private__.newObject = function() {
    var oid = newObjectDeferred();
    newObjectDeferredBegin(oid, true);
    return oid;
  });

  // Does not output the object.  The caller must call newObjectDeferredBegin(oid) before outputing any data
  var newObjectDeferred = (API.__private__.newObjectDeferred = function() {
    objectNumber++;
    offsets[objectNumber] = function() {
      return contentLength;
    };
    return objectNumber;
  });

  var newObjectDeferredBegin = function(oid, doOutput) {
    doOutput = typeof doOutput === "boolean" ? doOutput : false;
    offsets[oid] = contentLength;
    if (doOutput) {
      out(oid + " 0 obj");
    }
    return oid;
  };
  // Does not output the object until after the pages have been output.
  // Returns an object containing the objectId and content.
  // All pages have been added so the object ID can be estimated to start right after.
  // This does not modify the current objectNumber;  It must be updated after the newObjects are output.
  var newAdditionalObject = (API.__private__.newAdditionalObject = function() {
    var objId = newObjectDeferred();
    var obj = {
      objId: objId,
      content: ""
    };
    additionalObjects.push(obj);
    return obj;
  });

  var rootDictionaryObjId = newObjectDeferred();
  var resourceDictionaryObjId = newObjectDeferred();

  /////////////////////
  // Private functions
  /////////////////////

  var decodeColorString = (API.__private__.decodeColorString = function(color) {
    var colorEncoded = color.split(" ");
    if (
      colorEncoded.length === 2 &&
      (colorEncoded[1] === "g" || colorEncoded[1] === "G")
    ) {
      // convert grayscale value to rgb so that it can be converted to hex for consistency
      var floatVal = parseFloat(colorEncoded[0]);
      colorEncoded = [floatVal, floatVal, floatVal, "r"];
    } else if (
      colorEncoded.length === 5 &&
      (colorEncoded[4] === "k" || colorEncoded[4] === "K")
    ) {
      // convert CMYK values to rbg so that it can be converted to hex for consistency
      var red = (1.0 - colorEncoded[0]) * (1.0 - colorEncoded[3]);
      var green = (1.0 - colorEncoded[1]) * (1.0 - colorEncoded[3]);
      var blue = (1.0 - colorEncoded[2]) * (1.0 - colorEncoded[3]);

      colorEncoded = [red, green, blue, "r"];
    }
    var colorAsRGB = "#";
    for (var i = 0; i < 3; i++) {
      colorAsRGB += (
        "0" + Math.floor(parseFloat(colorEncoded[i]) * 255).toString(16)
      ).slice(-2);
    }
    return colorAsRGB;
  });

  var encodeColorString = (API.__private__.encodeColorString = function(
    options
  ) {
    var color;

    if (typeof options === "string") {
      options = {
        ch1: options
      };
    }
    var ch1 = options.ch1;
    var ch2 = options.ch2;
    var ch3 = options.ch3;
    var ch4 = options.ch4;
    var letterArray =
      options.pdfColorType === "draw" ? ["G", "RG", "K"] : ["g", "rg", "k"];

    if (typeof ch1 === "string" && ch1.charAt(0) !== "#") {
      var rgbColor = new RGBColor(ch1);
      if (rgbColor.ok) {
        ch1 = rgbColor.toHex();
      } else if (!/^\d*\.?\d*$/.test(ch1)) {
        throw new Error(
          'Invalid color "' + ch1 + '" passed to jsPDF.encodeColorString.'
        );
      }
    }
    //convert short rgb to long form
    if (typeof ch1 === "string" && /^#[0-9A-Fa-f]{3}$/.test(ch1)) {
      ch1 = "#" + ch1[1] + ch1[1] + ch1[2] + ch1[2] + ch1[3] + ch1[3];
    }

    if (typeof ch1 === "string" && /^#[0-9A-Fa-f]{6}$/.test(ch1)) {
      var hex = parseInt(ch1.substr(1), 16);
      ch1 = (hex >> 16) & 255;
      ch2 = (hex >> 8) & 255;
      ch3 = hex & 255;
    }

    if (
      typeof ch2 === "undefined" ||
      (typeof ch4 === "undefined" && ch1 === ch2 && ch2 === ch3)
    ) {
      // Gray color space.
      if (typeof ch1 === "string") {
        color = ch1 + " " + letterArray[0];
      } else {
        switch (options.precision) {
          case 2:
            color = f2(ch1 / 255) + " " + letterArray[0];
            break;
          case 3:
          default:
            color = f3(ch1 / 255) + " " + letterArray[0];
        }
      }
    } else if (typeof ch4 === "undefined" || typeof ch4 === "object") {
      // assume RGBA
      if (ch4 && !isNaN(ch4.a)) {
        //TODO Implement transparency.
        //WORKAROUND use white for now, if transparent, otherwise handle as rgb
        if (ch4.a === 0) {
          color = ["1.", "1.", "1.", letterArray[1]].join(" ");
          return color;
        }
      }
      // assume RGB
      if (typeof ch1 === "string") {
        color = [ch1, ch2, ch3, letterArray[1]].join(" ");
      } else {
        switch (options.precision) {
          case 2:
            color = [
              f2(ch1 / 255),
              f2(ch2 / 255),
              f2(ch3 / 255),
              letterArray[1]
            ].join(" ");
            break;
          default:
          case 3:
            color = [
              f3(ch1 / 255),
              f3(ch2 / 255),
              f3(ch3 / 255),
              letterArray[1]
            ].join(" ");
        }
      }
    } else {
      // assume CMYK
      if (typeof ch1 === "string") {
        color = [ch1, ch2, ch3, ch4, letterArray[2]].join(" ");
      } else {
        switch (options.precision) {
          case 2:
            color = [f2(ch1), f2(ch2), f2(ch3), f2(ch4), letterArray[2]].join(
              " "
            );
            break;
          case 3:
          default:
            color = [f3(ch1), f3(ch2), f3(ch3), f3(ch4), letterArray[2]].join(
              " "
            );
        }
      }
    }
    return color;
  });

  var getFilters = (API.__private__.getFilters = function() {
    return filters;
  });

  var putStream = (API.__private__.putStream = function(options) {
    options = options || {};
    var data = options.data || "";
    var filters = options.filters || getFilters();
    var alreadyAppliedFilters = options.alreadyAppliedFilters || [];
    var addLength1 = options.addLength1 || false;
    var valueOfLength1 = data.length;
    var objectId = options.objectId;
    var encryptor = function(data) {
      return data;
    };
    if (encryptionOptions !== null && typeof objectId == "undefined") {
      throw new Error(
        "ObjectId must be passed to putStream for file encryption"
      );
    }
    if (encryptionOptions !== null) {
      encryptor = encryption.encryptor(objectId, 0);
    }

    var processedData = {};
    if (filters === true) {
      filters = ["FlateEncode"];
    }
    var keyValues = options.additionalKeyValues || [];
    if (typeof jsPDF.API.processDataByFilters !== "undefined") {
      processedData = jsPDF.API.processDataByFilters(data, filters);
    } else {
      processedData = { data: data, reverseChain: [] };
    }
    var filterAsString =
      processedData.reverseChain +
      (Array.isArray(alreadyAppliedFilters)
        ? alreadyAppliedFilters.join(" ")
        : alreadyAppliedFilters.toString());

    if (processedData.data.length !== 0) {
      keyValues.push({
        key: "Length",
        value: processedData.data.length
      });
      if (addLength1 === true) {
        keyValues.push({
          key: "Length1",
          value: valueOfLength1
        });
      }
    }

    if (filterAsString.length != 0) {
      if (filterAsString.split("/").length - 1 === 1) {
        keyValues.push({
          key: "Filter",
          value: filterAsString
        });
      } else {
        keyValues.push({
          key: "Filter",
          value: "[" + filterAsString + "]"
        });

        for (var j = 0; j < keyValues.length; j += 1) {
          if (keyValues[j].key === "DecodeParms") {
            var decodeParmsArray = [];

            for (
              var i = 0;
              i < processedData.reverseChain.split("/").length - 1;
              i += 1
            ) {
              decodeParmsArray.push("null");
            }

            decodeParmsArray.push(keyValues[j].value);
            keyValues[j].value = "[" + decodeParmsArray.join(" ") + "]";
          }
        }
      }
    }

    out("<<");
    for (var k = 0; k < keyValues.length; k++) {
      out("/" + keyValues[k].key + " " + keyValues[k].value);
    }
    out(">>");
    if (processedData.data.length !== 0) {
      out("stream");
      out(encryptor(processedData.data));
      out("endstream");
    }
  });

  var putPage = (API.__private__.putPage = function(page) {
    var pageNumber = page.number;
    var data = page.data;
    var pageObjectNumber = page.objId;
    var pageContentsObjId = page.contentsObjId;

    newObjectDeferredBegin(pageObjectNumber, true);
    out("<</Type /Page");
    out("/Parent " + page.rootDictionaryObjId + " 0 R");
    out("/Resources " + page.resourceDictionaryObjId + " 0 R");
    out(
      "/MediaBox [" +
        parseFloat(hpf(page.mediaBox.bottomLeftX)) +
        " " +
        parseFloat(hpf(page.mediaBox.bottomLeftY)) +
        " " +
        hpf(page.mediaBox.topRightX) +
        " " +
        hpf(page.mediaBox.topRightY) +
        "]"
    );
    if (page.cropBox !== null) {
      out(
        "/CropBox [" +
          hpf(page.cropBox.bottomLeftX) +
          " " +
          hpf(page.cropBox.bottomLeftY) +
          " " +
          hpf(page.cropBox.topRightX) +
          " " +
          hpf(page.cropBox.topRightY) +
          "]"
      );
    }

    if (page.bleedBox !== null) {
      out(
        "/BleedBox [" +
          hpf(page.bleedBox.bottomLeftX) +
          " " +
          hpf(page.bleedBox.bottomLeftY) +
          " " +
          hpf(page.bleedBox.topRightX) +
          " " +
          hpf(page.bleedBox.topRightY) +
          "]"
      );
    }

    if (page.trimBox !== null) {
      out(
        "/TrimBox [" +
          hpf(page.trimBox.bottomLeftX) +
          " " +
          hpf(page.trimBox.bottomLeftY) +
          " " +
          hpf(page.trimBox.topRightX) +
          " " +
          hpf(page.trimBox.topRightY) +
          "]"
      );
    }

    if (page.artBox !== null) {
      out(
        "/ArtBox [" +
          hpf(page.artBox.bottomLeftX) +
          " " +
          hpf(page.artBox.bottomLeftY) +
          " " +
          hpf(page.artBox.topRightX) +
          " " +
          hpf(page.artBox.topRightY) +
          "]"
      );
    }

    if (typeof page.userUnit === "number" && page.userUnit !== 1.0) {
      out("/UserUnit " + page.userUnit);
    }

    events.publish("putPage", {
      objId: pageObjectNumber,
      pageContext: pagesContext[pageNumber],
      pageNumber: pageNumber,
      page: data
    });
    out("/Contents " + pageContentsObjId + " 0 R");
    out(">>");
    out("endobj");
    // Page content
    var pageContent = data.join("\n");

    if (apiMode === ApiMode.ADVANCED) {
      // if the user forgot to switch back to COMPAT mode, we must balance the graphics stack again
      pageContent += "\nQ";
    }

    newObjectDeferredBegin(pageContentsObjId, true);
    putStream({
      data: pageContent,
      filters: getFilters(),
      objectId: pageContentsObjId
    });
    out("endobj");
    return pageObjectNumber;
  });

  var putPages = (API.__private__.putPages = function() {
    var n,
      i,
      pageObjectNumbers = [];

    for (n = 1; n <= page; n++) {
      pagesContext[n].objId = newObjectDeferred();
      pagesContext[n].contentsObjId = newObjectDeferred();
    }

    for (n = 1; n <= page; n++) {
      pageObjectNumbers.push(
        putPage({
          number: n,
          data: pages[n],
          objId: pagesContext[n].objId,
          contentsObjId: pagesContext[n].contentsObjId,
          mediaBox: pagesContext[n].mediaBox,
          cropBox: pagesContext[n].cropBox,
          bleedBox: pagesContext[n].bleedBox,
          trimBox: pagesContext[n].trimBox,
          artBox: pagesContext[n].artBox,
          userUnit: pagesContext[n].userUnit,
          rootDictionaryObjId: rootDictionaryObjId,
          resourceDictionaryObjId: resourceDictionaryObjId
        })
      );
    }
    newObjectDeferredBegin(rootDictionaryObjId, true);
    out("<</Type /Pages");
    var kids = "/Kids [";
    for (i = 0; i < page; i++) {
      kids += pageObjectNumbers[i] + " 0 R ";
    }
    out(kids + "]");
    out("/Count " + page);
    out(">>");
    out("endobj");
    events.publish("postPutPages");
  });

  var putFont = function(font) {
    events.publish("putFont", {
      font: font,
      out: out,
      newObject: newObject,
      putStream: putStream
    });

    if (font.isAlreadyPutted !== true) {
      font.objectNumber = newObject();
      out("<<");
      out("/Type /Font");
      out("/BaseFont /" + toPDFName(font.postScriptName));
      out("/Subtype /Type1");
      if (typeof font.encoding === "string") {
        out("/Encoding /" + font.encoding);
      }
      out("/FirstChar 32");
      out("/LastChar 255");
      out(">>");
      out("endobj");
    }
  };

  var putFonts = function() {
    for (var fontKey in fonts) {
      if (fonts.hasOwnProperty(fontKey)) {
        if (
          putOnlyUsedFonts === false ||
          (putOnlyUsedFonts === true && usedFonts.hasOwnProperty(fontKey))
        ) {
          putFont(fonts[fontKey]);
        }
      }
    }
  };

  var putXObject = function(xObject) {
    xObject.objectNumber = newObject();

    var options = [];
    options.push({ key: "Type", value: "/XObject" });
    options.push({ key: "Subtype", value: "/Form" });
    options.push({
      key: "BBox",
      value:
        "[" +
        [
          hpf(xObject.x),
          hpf(xObject.y),
          hpf(xObject.x + xObject.width),
          hpf(xObject.y + xObject.height)
        ].join(" ") +
        "]"
    });
    options.push({
      key: "Matrix",
      value: "[" + xObject.matrix.toString() + "]"
    });
    // TODO: /Resources

    var stream = xObject.pages[1].join("\n");
    putStream({
      data: stream,
      additionalKeyValues: options,
      objectId: xObject.objectNumber
    });
    out("endobj");
  };

  var putXObjects = function() {
    for (var xObjectKey in renderTargets) {
      if (renderTargets.hasOwnProperty(xObjectKey)) {
        putXObject(renderTargets[xObjectKey]);
      }
    }
  };

  var interpolateAndEncodeRGBStream = function(colors, numberSamples) {
    var tValues = [];
    var t;
    var dT = 1.0 / (numberSamples - 1);
    for (t = 0.0; t < 1.0; t += dT) {
      tValues.push(t);
    }
    tValues.push(1.0);
    // add first and last control point if not present
    if (colors[0].offset != 0.0) {
      var c0 = {
        offset: 0.0,
        color: colors[0].color
      };
      colors.unshift(c0);
    }
    if (colors[colors.length - 1].offset != 1.0) {
      var c1 = {
        offset: 1.0,
        color: colors[colors.length - 1].color
      };
      colors.push(c1);
    }
    var out = "";
    var index = 0;

    for (var i = 0; i < tValues.length; i++) {
      t = tValues[i];
      while (t > colors[index + 1].offset) index++;
      var a = colors[index].offset;
      var b = colors[index + 1].offset;
      var d = (t - a) / (b - a);

      var aColor = colors[index].color;
      var bColor = colors[index + 1].color;

      out +=
        padd2Hex(Math.round((1 - d) * aColor[0] + d * bColor[0]).toString(16)) +
        padd2Hex(Math.round((1 - d) * aColor[1] + d * bColor[1]).toString(16)) +
        padd2Hex(Math.round((1 - d) * aColor[2] + d * bColor[2]).toString(16));
    }
    return out.trim();
  };

  var putShadingPattern = function(pattern, numberSamples) {
    /*
       Axial patterns shade between the two points specified in coords, radial patterns between the inner
       and outer circle.
       The user can specify an array (colors) that maps t-Values in [0, 1] to RGB colors. These are now
       interpolated to equidistant samples and written to pdf as a sample (type 0) function.
       */
    // The number of color samples that should be used to describe the shading.
    // The higher, the more accurate the gradient will be.
    numberSamples || (numberSamples = 21);
    var funcObjectNumber = newObject();
    var stream = interpolateAndEncodeRGBStream(pattern.colors, numberSamples);

    var options = [];
    options.push({ key: "FunctionType", value: "0" });
    options.push({ key: "Domain", value: "[0.0 1.0]" });
    options.push({ key: "Size", value: "[" + numberSamples + "]" });
    options.push({ key: "BitsPerSample", value: "8" });
    options.push({ key: "Range", value: "[0.0 1.0 0.0 1.0 0.0 1.0]" });
    options.push({ key: "Decode", value: "[0.0 1.0 0.0 1.0 0.0 1.0]" });

    putStream({
      data: stream,
      additionalKeyValues: options,
      alreadyAppliedFilters: ["/ASCIIHexDecode"],
      objectId: funcObjectNumber
    });
    out("endobj");

    pattern.objectNumber = newObject();
    out("<< /ShadingType " + pattern.type);
    out("/ColorSpace /DeviceRGB");
    var coords =
      "/Coords [" +
      hpf(parseFloat(pattern.coords[0])) +
      " " + // x1
      hpf(parseFloat(pattern.coords[1])) +
      " "; // y1
    if (pattern.type === 2) {
      // axial
      coords +=
        hpf(parseFloat(pattern.coords[2])) +
        " " + // x2
        hpf(parseFloat(pattern.coords[3])); // y2
    } else {
      // radial
      coords +=
        hpf(parseFloat(pattern.coords[2])) +
        " " + // r1
        hpf(parseFloat(pattern.coords[3])) +
        " " + // x2
        hpf(parseFloat(pattern.coords[4])) +
        " " + // y2
        hpf(parseFloat(pattern.coords[5])); // r2
    }
    coords += "]";
    out(coords);

    if (pattern.matrix) {
      out("/Matrix [" + pattern.matrix.toString() + "]");
    }
    out("/Function " + funcObjectNumber + " 0 R");
    out("/Extend [true true]");
    out(">>");
    out("endobj");
  };

  var putTilingPattern = function(pattern, deferredResourceDictionaryIds) {
    var resourcesObjectId = newObjectDeferred();
    var patternObjectId = newObject();

    deferredResourceDictionaryIds.push({
      resourcesOid: resourcesObjectId,
      objectOid: patternObjectId
    });

    pattern.objectNumber = patternObjectId;
    var options = [];
    options.push({ key: "Type", value: "/Pattern" });
    options.push({ key: "PatternType", value: "1" }); // tiling pattern
    options.push({ key: "PaintType", value: "1" }); // colored tiling pattern
    options.push({ key: "TilingType", value: "1" }); // constant spacing
    options.push({
      key: "BBox",
      value: "[" + pattern.boundingBox.map(hpf).join(" ") + "]"
    });
    options.push({ key: "XStep", value: hpf(pattern.xStep) });
    options.push({ key: "YStep", value: hpf(pattern.yStep) });
    options.push({ key: "Resources", value: resourcesObjectId + " 0 R" });
    if (pattern.matrix) {
      options.push({
        key: "Matrix",
        value: "[" + pattern.matrix.toString() + "]"
      });
    }

    putStream({
      data: pattern.stream,
      additionalKeyValues: options,
      objectId: pattern.objectNumber
    });
    out("endobj");
  };

  var putPatterns = function(deferredResourceDictionaryIds) {
    var patternKey;
    for (patternKey in patterns) {
      if (patterns.hasOwnProperty(patternKey)) {
        if (patterns[patternKey] instanceof ShadingPattern) {
          putShadingPattern(patterns[patternKey]);
        } else if (patterns[patternKey] instanceof TilingPattern) {
          putTilingPattern(patterns[patternKey], deferredResourceDictionaryIds);
        }
      }
    }
  };

  var putGState = function(gState) {
    gState.objectNumber = newObject();
    out("<<");
    for (var p in gState) {
      switch (p) {
        case "opacity":
          out("/ca " + f2(gState[p]));
          break;
        case "stroke-opacity":
          out("/CA " + f2(gState[p]));
          break;
      }
    }
    out(">>");
    out("endobj");
  };

  var putGStates = function() {
    var gStateKey;
    for (gStateKey in gStates) {
      if (gStates.hasOwnProperty(gStateKey)) {
        putGState(gStates[gStateKey]);
      }
    }
  };

  var putXobjectDict = function() {
    out("/XObject <<");
    for (var xObjectKey in renderTargets) {
      if (
        renderTargets.hasOwnProperty(xObjectKey) &&
        renderTargets[xObjectKey].objectNumber >= 0
      ) {
        out(
          "/" +
            xObjectKey +
            " " +
            renderTargets[xObjectKey].objectNumber +
            " 0 R"
        );
      }
    }

    // Loop through images, or other data objects
    events.publish("putXobjectDict");
    out(">>");
  };

  var putEncryptionDict = function() {
    encryption.oid = newObject();
    out("<<");
    out("/Filter /Standard");
    out("/V " + encryption.v);
    out("/R " + encryption.r);
    out("/U <" + encryption.toHexString(encryption.U) + ">");
    out("/O <" + encryption.toHexString(encryption.O) + ">");
    out("/P " + encryption.P);
    out(">>");
    out("endobj");
  };

  var putFontDict = function() {
    out("/Font <<");

    for (var fontKey in fonts) {
      if (fonts.hasOwnProperty(fontKey)) {
        if (
          putOnlyUsedFonts === false ||
          (putOnlyUsedFonts === true && usedFonts.hasOwnProperty(fontKey))
        ) {
          out("/" + fontKey + " " + fonts[fontKey].objectNumber + " 0 R");
        }
      }
    }
    out(">>");
  };

  var putShadingPatternDict = function() {
    if (Object.keys(patterns).length > 0) {
      out("/Shading <<");
      for (var patternKey in patterns) {
        if (
          patterns.hasOwnProperty(patternKey) &&
          patterns[patternKey] instanceof ShadingPattern &&
          patterns[patternKey].objectNumber >= 0
        ) {
          out(
            "/" + patternKey + " " + patterns[patternKey].objectNumber + " 0 R"
          );
        }
      }

      events.publish("putShadingPatternDict");
      out(">>");
    }
  };

  var putTilingPatternDict = function(objectOid) {
    if (Object.keys(patterns).length > 0) {
      out("/Pattern <<");
      for (var patternKey in patterns) {
        if (
          patterns.hasOwnProperty(patternKey) &&
          patterns[patternKey] instanceof API.TilingPattern &&
          patterns[patternKey].objectNumber >= 0 &&
          patterns[patternKey].objectNumber < objectOid // prevent cyclic dependencies
        ) {
          out(
            "/" + patternKey + " " + patterns[patternKey].objectNumber + " 0 R"
          );
        }
      }
      events.publish("putTilingPatternDict");
      out(">>");
    }
  };

  var putGStatesDict = function() {
    if (Object.keys(gStates).length > 0) {
      var gStateKey;
      out("/ExtGState <<");
      for (gStateKey in gStates) {
        if (
          gStates.hasOwnProperty(gStateKey) &&
          gStates[gStateKey].objectNumber >= 0
        ) {
          out("/" + gStateKey + " " + gStates[gStateKey].objectNumber + " 0 R");
        }
      }

      events.publish("putGStateDict");
      out(">>");
    }
  };

  var putResourceDictionary = function(objectIds) {
    newObjectDeferredBegin(objectIds.resourcesOid, true);
    out("<<");
    out("/ProcSet [/PDF /Text /ImageB /ImageC /ImageI]");
    putFontDict();
    putShadingPatternDict();
    putTilingPatternDict(objectIds.objectOid);
    putGStatesDict();
    putXobjectDict();
    out(">>");
    out("endobj");
  };

  var putResources = function() {
    // FormObjects, Patterns etc. might use other FormObjects/Patterns/Images
    // which means their resource dictionaries must contain the already resolved
    // object ids. For this reason we defer the serialization of the resource
    // dicts until all objects have been serialized and have object ids.
    //
    // In order to prevent cyclic dependencies (which Adobe Reader doesn't like),
    // we only put all oids that are smaller than the oid of the object the
    // resource dict belongs to. This is correct behavior, since the streams
    // may only use other objects that have already been defined and thus appear
    // earlier in their respective collection.
    // Currently, this only affects tiling patterns, but a (more) correct
    // implementation of FormObjects would also define their own resource dicts.
    var deferredResourceDictionaryIds = [];

    putFonts();
    putGStates();
    putXObjects();
    putPatterns(deferredResourceDictionaryIds);

    events.publish("putResources");
    deferredResourceDictionaryIds.forEach(putResourceDictionary);
    putResourceDictionary({
      resourcesOid: resourceDictionaryObjId,
      objectOid: Number.MAX_SAFE_INTEGER // output all objects
    });
    events.publish("postPutResources");
  };

  var putAdditionalObjects = function() {
    events.publish("putAdditionalObjects");
    for (var i = 0; i < additionalObjects.length; i++) {
      var obj = additionalObjects[i];
      newObjectDeferredBegin(obj.objId, true);
      out(obj.content);
      out("endobj");
    }
    events.publish("postPutAdditionalObjects");
  };

  var addFontToFontDictionary = function(font) {
    fontmap[font.fontName] = fontmap[font.fontName] || {};
    fontmap[font.fontName][font.fontStyle] = font.id;
  };

  var addFont = function(
    postScriptName,
    fontName,
    fontStyle,
    encoding,
    isStandardFont
  ) {
    var font = {
      id: "F" + (Object.keys(fonts).length + 1).toString(10),
      postScriptName: postScriptName,
      fontName: fontName,
      fontStyle: fontStyle,
      encoding: encoding,
      isStandardFont: isStandardFont || false,
      metadata: {}
    };

    events.publish("addFont", {
      font: font,
      instance: this
    });

    fonts[font.id] = font;
    addFontToFontDictionary(font);
    return font.id;
  };

  var addFonts = function(arrayOfFonts) {
    for (var i = 0, l = standardFonts.length; i < l; i++) {
      var fontKey = addFont.call(
        this,
        arrayOfFonts[i][0],
        arrayOfFonts[i][1],
        arrayOfFonts[i][2],
        standardFonts[i][3],
        true
      );

      if (putOnlyUsedFonts === false) {
        usedFonts[fontKey] = true;
      }
      // adding aliases for standard fonts, this time matching the capitalization
      var parts = arrayOfFonts[i][0].split("-");
      addFontToFontDictionary({
        id: fontKey,
        fontName: parts[0],
        fontStyle: parts[1] || ""
      });
    }
    events.publish("addFonts", {
      fonts: fonts,
      dictionary: fontmap
    });
  };

  var SAFE = function __safeCall(fn) {
    fn.foo = function __safeCallWrapper() {
      try {
        return fn.apply(this, arguments);
      } catch (e) {
        var stack = e.stack || "";
        if (~stack.indexOf(" at ")) stack = stack.split(" at ")[1];
        var m =
          "Error in function " +
          stack.split("\n")[0].split("<")[0] +
          ": " +
          e.message;
        if (globalObject.console) {
          globalObject.console.error(m, e);
          if (globalObject.alert) alert(m);
        } else {
          throw new Error(m);
        }
      }
    };
    fn.foo.bar = fn;
    return fn.foo;
  };

  var to8bitStream = function(text, flags) {
    /**
     * PDF 1.3 spec:
     * "For text strings encoded in Unicode, the first two bytes must be 254 followed by
     * 255, representing the Unicode byte order marker, U+FEFF. (This sequence conflicts
     * with the PDFDocEncoding character sequence thorn ydieresis, which is unlikely
     * to be a meaningful beginning of a word or phrase.) The remainder of the
     * string consists of Unicode character codes, according to the UTF-16 encoding
     * specified in the Unicode standard, version 2.0. Commonly used Unicode values
     * are represented as 2 bytes per character, with the high-order byte appearing first
     * in the string."
     *
     * In other words, if there are chars in a string with char code above 255, we
     * recode the string to UCS2 BE - string doubles in length and BOM is prepended.
     *
     * HOWEVER!
     * Actual *content* (body) text (as opposed to strings used in document properties etc)
     * does NOT expect BOM. There, it is treated as a literal GID (Glyph ID)
     *
     * Because of Adobe's focus on "you subset your fonts!" you are not supposed to have
     * a font that maps directly Unicode (UCS2 / UTF16BE) code to font GID, but you could
     * fudge it with "Identity-H" encoding and custom CIDtoGID map that mimics Unicode
     * code page. There, however, all characters in the stream are treated as GIDs,
     * including BOM, which is the reason we need to skip BOM in content text (i.e. that
     * that is tied to a font).
     *
     * To signal this "special" PDFEscape / to8bitStream handling mode,
     * API.text() function sets (unless you overwrite it with manual values
     * given to API.text(.., flags) )
     * flags.autoencode = true
     * flags.noBOM = true
     *
     * ===================================================================================
     * `flags` properties relied upon:
     *   .sourceEncoding = string with encoding label.
     *                     "Unicode" by default. = encoding of the incoming text.
     *                     pass some non-existing encoding name
     *                     (ex: 'Do not touch my strings! I know what I am doing.')
     *                     to make encoding code skip the encoding step.
     *   .outputEncoding = Either valid PDF encoding name
     *                     (must be supported by jsPDF font metrics, otherwise no encoding)
     *                     or a JS object, where key = sourceCharCode, value = outputCharCode
     *                     missing keys will be treated as: sourceCharCode === outputCharCode
     *   .noBOM
     *       See comment higher above for explanation for why this is important
     *   .autoencode
     *       See comment higher above for explanation for why this is important
     */

    var i,
      l,
      sourceEncoding,
      encodingBlock,
      outputEncoding,
      newtext,
      isUnicode,
      ch,
      bch;

    flags = flags || {};
    sourceEncoding = flags.sourceEncoding || "Unicode";
    outputEncoding = flags.outputEncoding;

    // This 'encoding' section relies on font metrics format
    // attached to font objects by, among others,
    // "Willow Systems' standard_font_metrics plugin"
    // see jspdf.plugin.standard_font_metrics.js for format
    // of the font.metadata.encoding Object.
    // It should be something like
    //   .encoding = {'codePages':['WinANSI....'], 'WinANSI...':{code:code, ...}}
    //   .widths = {0:width, code:width, ..., 'fof':divisor}
    //   .kerning = {code:{previous_char_code:shift, ..., 'fof':-divisor},...}
    if (
      (flags.autoencode || outputEncoding) &&
      fonts[activeFontKey].metadata &&
      fonts[activeFontKey].metadata[sourceEncoding] &&
      fonts[activeFontKey].metadata[sourceEncoding].encoding
    ) {
      encodingBlock = fonts[activeFontKey].metadata[sourceEncoding].encoding;

      // each font has default encoding. Some have it clearly defined.
      if (!outputEncoding && fonts[activeFontKey].encoding) {
        outputEncoding = fonts[activeFontKey].encoding;
      }

      // Hmmm, the above did not work? Let's try again, in different place.
      if (!outputEncoding && encodingBlock.codePages) {
        outputEncoding = encodingBlock.codePages[0]; // let's say, first one is the default
      }

      if (typeof outputEncoding === "string") {
        outputEncoding = encodingBlock[outputEncoding];
      }
      // we want output encoding to be a JS Object, where
      // key = sourceEncoding's character code and
      // value = outputEncoding's character code.
      if (outputEncoding) {
        isUnicode = false;
        newtext = [];
        for (i = 0, l = text.length; i < l; i++) {
          ch = outputEncoding[text.charCodeAt(i)];
          if (ch) {
            newtext.push(String.fromCharCode(ch));
          } else {
            newtext.push(text[i]);
          }

          // since we are looping over chars anyway, might as well
          // check for residual unicodeness
          if (newtext[i].charCodeAt(0) >> 8) {
            /* more than 255 */
            isUnicode = true;
          }
        }
        text = newtext.join("");
      }
    }

    i = text.length;
    // isUnicode may be set to false above. Hence the triple-equal to undefined
    while (isUnicode === undefined && i !== 0) {
      if (text.charCodeAt(i - 1) >> 8) {
        /* more than 255 */
        isUnicode = true;
      }
      i--;
    }
    if (!isUnicode) {
      return text;
    }

    newtext = flags.noBOM ? [] : [254, 255];
    for (i = 0, l = text.length; i < l; i++) {
      ch = text.charCodeAt(i);
      bch = ch >> 8; // divide by 256
      if (bch >> 8) {
        /* something left after dividing by 256 second time */
        throw new Error(
          "Character at position " +
            i +
            " of string '" +
            text +
            "' exceeds 16bits. Cannot be encoded into UCS-2 BE"
        );
      }
      newtext.push(bch);
      newtext.push(ch - (bch << 8));
    }
    return String.fromCharCode.apply(undefined, newtext);
  };

  var pdfEscape = (API.__private__.pdfEscape = API.pdfEscape = function(
    text,
    flags
  ) {
    /**
     * Replace '/', '(', and ')' with pdf-safe versions
     *
     * Doing to8bitStream does NOT make this PDF display unicode text. For that
     * we also need to reference a unicode font and embed it - royal pain in the rear.
     *
     * There is still a benefit to to8bitStream - PDF simply cannot handle 16bit chars,
     * which JavaScript Strings are happy to provide. So, while we still cannot display
     * 2-byte characters property, at least CONDITIONALLY converting (entire string containing)
     * 16bit chars to (USC-2-BE) 2-bytes per char + BOM streams we ensure that entire PDF
     * is still parseable.
     * This will allow immediate support for unicode in document properties strings.
     */
    return to8bitStream(text, flags)
      .replace(/\\/g, "\\\\")
      .replace(/\(/g, "\\(")
      .replace(/\)/g, "\\)");
  });

  var beginPage = (API.__private__.beginPage = function(format) {
    pages[++page] = [];
    pagesContext[page] = {
      objId: 0,
      contentsObjId: 0,
      userUnit: Number(userUnit),
      artBox: null,
      bleedBox: null,
      cropBox: null,
      trimBox: null,
      mediaBox: {
        bottomLeftX: 0,
        bottomLeftY: 0,
        topRightX: Number(format[0]),
        topRightY: Number(format[1])
      }
    };
    _setPage(page);
    setOutputDestination(pages[currentPage]);
  });

  var _addPage = function(parmFormat, parmOrientation) {
    var dimensions, width, height;

    orientation = parmOrientation || orientation;

    if (typeof parmFormat === "string") {
      dimensions = getPageFormat(parmFormat.toLowerCase());
      if (Array.isArray(dimensions)) {
        width = dimensions[0];
        height = dimensions[1];
      }
    }

    if (Array.isArray(parmFormat)) {
      width = parmFormat[0] * scaleFactor;
      height = parmFormat[1] * scaleFactor;
    }

    if (isNaN(width)) {
      width = format[0];
      height = format[1];
    }

    if (width > 14400 || height > 14400) {
      console.warn(
        "A page in a PDF can not be wider or taller than 14400 userUnit. jsPDF limits the width/height to 14400"
      );
      width = Math.min(14400, width);
      height = Math.min(14400, height);
    }

    format = [width, height];

    switch (orientation.substr(0, 1)) {
      case "l":
        if (height > width) {
          format = [height, width];
        }
        break;
      case "p":
        if (width > height) {
          format = [height, width];
        }
        break;
    }

    beginPage(format);

    // Set line width
    setLineWidth(lineWidth);
    // Set draw color
    out(strokeColor);
    // resurrecting non-default line caps, joins
    if (lineCapID !== 0) {
      out(lineCapID + " J");
    }
    if (lineJoinID !== 0) {
      out(lineJoinID + " j");
    }
    events.publish("addPage", {
      pageNumber: page
    });
  };

  var _deletePage = function(n) {
    if (n > 0 && n <= page) {
      pages.splice(n, 1);
      pagesContext.splice(n, 1);
      page--;
      if (currentPage > page) {
        currentPage = page;
      }
      this.setPage(currentPage);
    }
  };

  var _setPage = function(n) {
    if (n > 0 && n <= page) {
      currentPage = n;
    }
  };

  var getNumberOfPages = (API.__private__.getNumberOfPages = API.getNumberOfPages = function() {
    return pages.length - 1;
  });

  /**
   * Returns a document-specific font key - a label assigned to a
   * font name + font type combination at the time the font was added
   * to the font inventory.
   *
   * Font key is used as label for the desired font for a block of text
   * to be added to the PDF document stream.
   * @private
   * @function
   * @param fontName {string} can be undefined on "falthy" to indicate "use current"
   * @param fontStyle {string} can be undefined on "falthy" to indicate "use current"
   * @returns {string} Font key.
   * @ignore
   */
  var getFont = function(fontName, fontStyle, options) {
    var key = undefined,
      fontNameLowerCase;
    options = options || {};

    fontName =
      fontName !== undefined ? fontName : fonts[activeFontKey].fontName;
    fontStyle =
      fontStyle !== undefined ? fontStyle : fonts[activeFontKey].fontStyle;
    fontNameLowerCase = fontName.toLowerCase();

    if (
      fontmap[fontNameLowerCase] !== undefined &&
      fontmap[fontNameLowerCase][fontStyle] !== undefined
    ) {
      key = fontmap[fontNameLowerCase][fontStyle];
    } else if (
      fontmap[fontName] !== undefined &&
      fontmap[fontName][fontStyle] !== undefined
    ) {
      key = fontmap[fontName][fontStyle];
    } else {
      if (options.disableWarning === false) {
        console.warn(
          "Unable to look up font label for font '" +
            fontName +
            "', '" +
            fontStyle +
            "'. Refer to getFontList() for available fonts."
        );
      }
    }

    if (!key && !options.noFallback) {
      key = fontmap["times"][fontStyle];
      if (key == null) {
        key = fontmap["times"]["normal"];
      }
    }
    return key;
  };

  var putInfo = (API.__private__.putInfo = function() {
    var objectId = newObject();
    var encryptor = function(data) {
      return data;
    };
    if (encryptionOptions !== null) {
      encryptor = encryption.encryptor(objectId, 0);
    }
    out("<<");
    out("/Producer (" + pdfEscape(encryptor("jsPDF " + jsPDF.version)) + ")");
    for (var key in documentProperties) {
      if (documentProperties.hasOwnProperty(key) && documentProperties[key]) {
        out(
          "/" +
            key.substr(0, 1).toUpperCase() +
            key.substr(1) +
            " (" +
            pdfEscape(encryptor(documentProperties[key])) +
            ")"
        );
      }
    }
    out("/CreationDate (" + pdfEscape(encryptor(creationDate)) + ")");
    out(">>");
    out("endobj");
  });

  var putCatalog = (API.__private__.putCatalog = function(options) {
    options = options || {};
    var tmpRootDictionaryObjId =
      options.rootDictionaryObjId || rootDictionaryObjId;
    newObject();
    out("<<");
    out("/Type /Catalog");
    out("/Pages " + tmpRootDictionaryObjId + " 0 R");
    // PDF13ref Section 7.2.1
    if (!zoomMode) zoomMode = "fullwidth";
    switch (zoomMode) {
      case "fullwidth":
        out("/OpenAction [3 0 R /FitH null]");
        break;
      case "fullheight":
        out("/OpenAction [3 0 R /FitV null]");
        break;
      case "fullpage":
        out("/OpenAction [3 0 R /Fit]");
        break;
      case "original":
        out("/OpenAction [3 0 R /XYZ null null 1]");
        break;
      default:
        var pcn = "" + zoomMode;
        if (pcn.substr(pcn.length - 1) === "%")
          zoomMode = parseInt(zoomMode) / 100;
        if (typeof zoomMode === "number") {
          out("/OpenAction [3 0 R /XYZ null null " + f2(zoomMode) + "]");
        }
    }
    if (!layoutMode) layoutMode = "continuous";
    switch (layoutMode) {
      case "continuous":
        out("/PageLayout /OneColumn");
        break;
      case "single":
        out("/PageLayout /SinglePage");
        break;
      case "two":
      case "twoleft":
        out("/PageLayout /TwoColumnLeft");
        break;
      case "tworight":
        out("/PageLayout /TwoColumnRight");
        break;
    }
    if (pageMode) {
      /**
       * A name object specifying how the document should be displayed when opened:
       * UseNone      : Neither document outline nor thumbnail images visible -- DEFAULT
       * UseOutlines  : Document outline visible
       * UseThumbs    : Thumbnail images visible
       * FullScreen   : Full-screen mode, with no menu bar, window controls, or any other window visible
       */
      out("/PageMode /" + pageMode);
    }
    events.publish("putCatalog");
    out(">>");
    out("endobj");
  });

  var putTrailer = (API.__private__.putTrailer = function() {
    out("trailer");
    out("<<");
    out("/Size " + (objectNumber + 1));
    // Root and Info must be the last and second last objects written respectively
    out("/Root " + objectNumber + " 0 R");
    out("/Info " + (objectNumber - 1) + " 0 R");
    if (encryptionOptions !== null) {
      out("/Encrypt " + encryption.oid + " 0 R");
    }
    out("/ID [ <" + fileId + "> <" + fileId + "> ]");
    out(">>");
  });

  var putHeader = (API.__private__.putHeader = function() {
    out("%PDF-" + pdfVersion);
    out("%\xBA\xDF\xAC\xE0");
  });

  var putXRef = (API.__private__.putXRef = function() {
    var p = "0000000000";

    out("xref");
    out("0 " + (objectNumber + 1));
    out("0000000000 65535 f ");
    for (var i = 1; i <= objectNumber; i++) {
      var offset = offsets[i];
      if (typeof offset === "function") {
        out((p + offsets[i]()).slice(-10) + " 00000 n ");
      } else {
        if (typeof offsets[i] !== "undefined") {
          out((p + offsets[i]).slice(-10) + " 00000 n ");
        } else {
          out("0000000000 00000 n ");
        }
      }
    }
  });

  var buildDocument = (API.__private__.buildDocument = function() {
    resetDocument();
    setOutputDestination(content);

    events.publish("buildDocument");

    putHeader();
    putPages();
    putAdditionalObjects();
    putResources();
    if (encryptionOptions !== null) putEncryptionDict();
    putInfo();
    putCatalog();

    var offsetOfXRef = contentLength;
    putXRef();
    putTrailer();
    out("startxref");
    out("" + offsetOfXRef);
    out("%%EOF");

    setOutputDestination(pages[currentPage]);

    return content.join("\n");
  });

  var getBlob = (API.__private__.getBlob = function(data) {
    return new Blob([getArrayBuffer(data)], {
      type: "application/pdf"
    });
  });

  /**
   * Generates the PDF document.
   *
   * If `type` argument is undefined, output is raw body of resulting PDF returned as a string.
   *
   * @param {string} type A string identifying one of the possible output types.<br/>
   *                      Possible values are: <br/>
   *                          'arraybuffer' -> (ArrayBuffer)<br/>
   *                          'blob' -> (Blob)<br/>
   *                          'bloburi'/'bloburl' -> (string)<br/>
   *                          'datauristring'/'dataurlstring' -> (string)<br/>
   *                          'datauri'/'dataurl' -> (undefined) -> change location to generated datauristring/dataurlstring<br/>
   * @param {Object|string} options An object providing some additional signalling to PDF generator.<br/>
   *                                Possible options are 'filename'.<br/>
   *                                A string can be passed instead of {filename:string} and defaults to 'generated.pdf'
   * @function
   * @instance
   * @returns {string|window|ArrayBuffer|Blob|jsPDF|null|undefined}
   * @memberof jsPDF#
   * @name output
   */
  var output = (API.output = API.__private__.output = SAFE(function output(
    type,
    options
  ) {
    options = options || {};

    if (typeof options === "string") {
      options = {
        filename: options
      };
    } else {
      options.filename = options.filename || "generated.pdf";
    }

    switch (type) {
      case undefined:
        return buildDocument();
      case "save":
        API.save(options.filename);
        break;
      case "arraybuffer":
        return getArrayBuffer(buildDocument());
      case "blob":
        return getBlob(buildDocument());
      case "bloburi":
      case "bloburl":
        // Developer is responsible of calling revokeObjectURL
        if (
          typeof globalObject.URL !== "undefined" &&
          typeof globalObject.URL.createObjectURL === "function"
        ) {
          return (
            (globalObject.URL &&
              globalObject.URL.createObjectURL(getBlob(buildDocument()))) ||
            void 0
          );
        } else {
          console.warn(
            "bloburl is not supported by your system, because URL.createObjectURL is not supported by your browser."
          );
        }
        break;
      case "datauristring":
      case "dataurlstring":
        var dataURI = "";
        var pdfDocument = buildDocument();
        try {
          dataURI = btoa(pdfDocument);
        } catch (e) {
          dataURI = btoa(unescape(encodeURIComponent(pdfDocument)));
        }
        return (
          "data:application/pdf;filename=" +
          options.filename +
          ";base64," +
          dataURI
        );
      case "datauri":
      case "dataurl":
        return (globalObject.document.location.href = this.output(
          "datauristring",
          options
        ));
      default:
        return null;
    }
  }));

  /**
   * Used to see if a supplied hotfix was requested when the pdf instance was created.
   * @param {string} hotfixName - The name of the hotfix to check.
   * @returns {boolean}
   */
  var hasHotfix = function(hotfixName) {
    return (
      Array.isArray(hotfixes) === true && hotfixes.indexOf(hotfixName) > -1
    );
  };

  switch (unit) {
    case "pt":
      scaleFactor = 1;
      break;
    case "mm":
      scaleFactor = 72 / 25.4;
      break;
    case "cm":
      scaleFactor = 72 / 2.54;
      break;
    case "in":
      scaleFactor = 72;
      break;
    case "px":
      if (hasHotfix("px_scaling") == true) {
        scaleFactor = 72 / 96;
      } else {
        scaleFactor = 96 / 72;
      }
      break;
    case "pc":
      scaleFactor = 12;
      break;
    case "em":
      scaleFactor = 12;
      break;
    case "ex":
      scaleFactor = 6;
      break;
    default:
      if (typeof unit === "number") {
        scaleFactor = unit;
      } else {
        throw new Error("Invalid unit: " + unit);
      }
  }

  var encryption = null;
  setCreationDate();
  setFileId();

  var getEncryptor = function(objectId) {
    if (encryptionOptions !== null) {
      return encryption.encryptor(objectId, 0);
    }
    return function(data) {
      return data;
    };
  };

  //---------------------------------------
  // Public API

  var getPageInfo = (API.__private__.getPageInfo = API.getPageInfo = function(
    pageNumberOneBased
  ) {
    if (isNaN(pageNumberOneBased) || pageNumberOneBased % 1 !== 0) {
      throw new Error("Invalid argument passed to jsPDF.getPageInfo");
    }
    var objId = pagesContext[pageNumberOneBased].objId;
    return {
      objId: objId,
      pageNumber: pageNumberOneBased,
      pageContext: pagesContext[pageNumberOneBased]
    };
  });

  var getPageInfoByObjId = (API.__private__.getPageInfoByObjId = function(
    objId
  ) {
    if (isNaN(objId) || objId % 1 !== 0) {
      throw new Error("Invalid argument passed to jsPDF.getPageInfoByObjId");
    }
    for (var pageNumber in pagesContext) {
      if (pagesContext[pageNumber].objId === objId) {
        break;
      }
    }
    return getPageInfo(pageNumber);
  });

  var getCurrentPageInfo = (API.__private__.getCurrentPageInfo = API.getCurrentPageInfo = function() {
    return {
      objId: pagesContext[currentPage].objId,
      pageNumber: currentPage,
      pageContext: pagesContext[currentPage]
    };
  });

  /**
   * Adds (and transfers the focus to) new page to the PDF document.
   * @param format {String/Array} The format of the new page. Can be: <ul><li>a0 - a10</li><li>b0 - b10</li><li>c0 - c10</li><li>dl</li><li>letter</li><li>government-letter</li><li>legal</li><li>junior-legal</li><li>ledger</li><li>tabloid</li><li>credit-card</li></ul><br />
   * Default is "a4". If you want to use your own format just pass instead of one of the above predefined formats the size as an number-array, e.g. [595.28, 841.89]
   * @param orientation {string} Orientation of the new page. Possible values are "portrait" or "landscape" (or shortcuts "p" (Default), "l").
   * @function
   * @instance
   * @returns {jsPDF}
   *
   * @memberof jsPDF#
   * @name addPage
   */
  API.addPage = function() {
    _addPage.apply(this, arguments);
    return this;
  };
  /**
   * Adds (and transfers the focus to) new page to the PDF document.
   * @function
   * @instance
   * @returns {jsPDF}
   *
   * @memberof jsPDF#
   * @name setPage
   * @param {number} page Switch the active page to the page number specified (indexed starting at 1).
   * @example
   * doc = jsPDF()
   * doc.addPage()
   * doc.addPage()
   * doc.text('I am on page 3', 10, 10)
   * doc.setPage(1)
   * doc.text('I am on page 1', 10, 10)
   */
  API.setPage = function() {
    _setPage.apply(this, arguments);
    setOutputDestination.call(this, pages[currentPage]);
    return this;
  };

  /**
   * @name insertPage
   * @memberof jsPDF#
   *
   * @function
   * @instance
   * @param {Object} beforePage
   * @returns {jsPDF}
   */
  API.insertPage = function(beforePage) {
    this.addPage();
    this.movePage(currentPage, beforePage);
    return this;
  };

  /**
   * @name movePage
   * @memberof jsPDF#
   * @function
   * @instance
   * @param {number} targetPage
   * @param {number} beforePage
   * @returns {jsPDF}
   */
  API.movePage = function(targetPage, beforePage) {
    var tmpPages, tmpPagesContext;
    if (targetPage > beforePage) {
      tmpPages = pages[targetPage];
      tmpPagesContext = pagesContext[targetPage];
      for (var i = targetPage; i > beforePage; i--) {
        pages[i] = pages[i - 1];
        pagesContext[i] = pagesContext[i - 1];
      }
      pages[beforePage] = tmpPages;
      pagesContext[beforePage] = tmpPagesContext;
      this.setPage(beforePage);
    } else if (targetPage < beforePage) {
      tmpPages = pages[targetPage];
      tmpPagesContext = pagesContext[targetPage];
      for (var j = targetPage; j < beforePage; j++) {
        pages[j] = pages[j + 1];
        pagesContext[j] = pagesContext[j + 1];
      }
      pages[beforePage] = tmpPages;
      pagesContext[beforePage] = tmpPagesContext;
      this.setPage(beforePage);
    }
    return this;
  };

  /**
   * Deletes a page from the PDF.
   * @name deletePage
   * @memberof jsPDF#
   * @function
   * @param {number} targetPage
   * @instance
   * @returns {jsPDF}
   */
  API.deletePage = function() {
    _deletePage.apply(this, arguments);
    return this;
  };

  /**
   * Adds text to page. Supports adding multiline text when 'text' argument is an Array of Strings.
   *
   * @function
   * @instance
   * @param {String|Array} text String or array of strings to be added to the page. Each line is shifted one line down per font, spacing settings declared before this call.
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page.
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page.
   * @param {Object} [options] - Collection of settings signaling how the text must be encoded.
   * @param {string} [options.align=left] - The alignment of the text, possible values: left, center, right, justify.
   * @param {string} [options.baseline=alphabetic] - Sets text baseline used when drawing the text, possible values: alphabetic, ideographic, bottom, top, middle, hanging
   * @param {number|Matrix} [options.angle=0] - Rotate the text clockwise or counterclockwise. Expects the angle in degree.
   * @param {number} [options.rotationDirection=1] - Direction of the rotation. 0 = clockwise, 1 = counterclockwise.
   * @param {number} [options.charSpace=0] - The space between each letter.
   * @param {number} [options.horizontalScale=1] - Horizontal scale of the text as a factor of the regular size.
   * @param {number} [options.lineHeightFactor=1.15] - The lineheight of each line.
   * @param {Object} [options.flags] - Flags for to8bitStream.
   * @param {boolean} [options.flags.noBOM=true] - Don't add BOM to Unicode-text.
   * @param {boolean} [options.flags.autoencode=true] - Autoencode the Text.
   * @param {number} [options.maxWidth=0] - Split the text by given width, 0 = no split.
   * @param {string} [options.renderingMode=fill] - Set how the text should be rendered, possible values: fill, stroke, fillThenStroke, invisible, fillAndAddForClipping, strokeAndAddPathForClipping, fillThenStrokeAndAddToPathForClipping, addToPathForClipping.
   * @param {boolean} [options.isInputVisual] - Option for the BidiEngine
   * @param {boolean} [options.isOutputVisual] - Option for the BidiEngine
   * @param {boolean} [options.isInputRtl] - Option for the BidiEngine
   * @param {boolean} [options.isOutputRtl] - Option for the BidiEngine
   * @param {boolean} [options.isSymmetricSwapping] - Option for the BidiEngine
   * @param {number|Matrix} transform If transform is a number the text will be rotated by this value around the anchor set by x and y.
   *
   * If it is a Matrix, this matrix gets directly applied to the text, which allows shearing
   * effects etc.; the x and y offsets are then applied AFTER the coordinate system has been established by this
   * matrix. This means passing a rotation matrix that is equivalent to some rotation angle will in general yield a
   * DIFFERENT result. A matrix is only allowed in "advanced" API mode.
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name text
   */
  API.__private__.text = API.text = function(text, x, y, options, transform) {
    /*
     * Inserts something like this into PDF
     *   BT
     *    /F1 16 Tf  % Font name + size
     *    16 TL % How many units down for next line in multiline text
     *    0 g % color
     *    28.35 813.54 Td % position
     *    (line one) Tj
     *    T* (line two) Tj
     *    T* (line three) Tj
     *   ET
     */
    options = options || {};
    var scope = options.scope || this;
    var payload, da, angle, align, charSpace, maxWidth, flags, horizontalScale;

    // Pre-August-2012 the order of arguments was function(x, y, text, flags)
    // in effort to make all calls have similar signature like
    //   function(data, coordinates... , miscellaneous)
    // this method had its args flipped.
    // code below allows backward compatibility with old arg order.
    if (
      typeof text === "number" &&
      typeof x === "number" &&
      (typeof y === "string" || Array.isArray(y))
    ) {
      var tmp = y;
      y = x;
      x = text;
      text = tmp;
    }

    var transformationMatrix;

    if (arguments[3] instanceof Matrix === false) {
      flags = arguments[3];
      angle = arguments[4];
      align = arguments[5];

      if (typeof flags !== "object" || flags === null) {
        if (typeof angle === "string") {
          align = angle;
          angle = null;
        }
        if (typeof flags === "string") {
          align = flags;
          flags = null;
        }
        if (typeof flags === "number") {
          angle = flags;
          flags = null;
        }
        options = {
          flags: flags,
          angle: angle,
          align: align
        };
      }
    } else {
      advancedApiModeTrap(
        "The transform parameter of text() with a Matrix value"
      );
      transformationMatrix = transform;
    }

    if (isNaN(x) || isNaN(y) || typeof text === "undefined" || text === null) {
      throw new Error("Invalid arguments passed to jsPDF.text");
    }

    if (text.length === 0) {
      return scope;
    }

    var xtra = "";
    var isHex = false;
    var lineHeight =
      typeof options.lineHeightFactor === "number"
        ? options.lineHeightFactor
        : lineHeightFactor;
    var scaleFactor = scope.internal.scaleFactor;

    function ESC(s) {
      s = s.split("\t").join(Array(options.TabLen || 9).join(" "));
      return pdfEscape(s, flags);
    }

    function transformTextToSpecialArray(text) {
      //we don't want to destroy original text array, so cloning it
      var sa = text.concat();
      var da = [];
      var len = sa.length;
      var curDa;
      //we do array.join('text that must not be PDFescaped")
      //thus, pdfEscape each component separately
      while (len--) {
        curDa = sa.shift();
        if (typeof curDa === "string") {
          da.push(curDa);
        } else {
          if (
            Array.isArray(text) &&
            (curDa.length === 1 ||
              (curDa[1] === undefined && curDa[2] === undefined))
          ) {
            da.push(curDa[0]);
          } else {
            da.push([curDa[0], curDa[1], curDa[2]]);
          }
        }
      }
      return da;
    }

    function processTextByFunction(text, processingFunction) {
      var result;
      if (typeof text === "string") {
        result = processingFunction(text)[0];
      } else if (Array.isArray(text)) {
        //we don't want to destroy original text array, so cloning it
        var sa = text.concat();
        var da = [];
        var len = sa.length;
        var curDa;
        var tmpResult;
        //we do array.join('text that must not be PDFescaped")
        //thus, pdfEscape each component separately
        while (len--) {
          curDa = sa.shift();
          if (typeof curDa === "string") {
            da.push(processingFunction(curDa)[0]);
          } else if (Array.isArray(curDa) && typeof curDa[0] === "string") {
            tmpResult = processingFunction(curDa[0], curDa[1], curDa[2]);
            da.push([tmpResult[0], tmpResult[1], tmpResult[2]]);
          }
        }
        result = da;
      }
      return result;
    }

    //Check if text is of type String
    var textIsOfTypeString = false;
    var tmpTextIsOfTypeString = true;

    if (typeof text === "string") {
      textIsOfTypeString = true;
    } else if (Array.isArray(text)) {
      //we don't want to destroy original text array, so cloning it
      var sa = text.concat();
      da = [];
      var len = sa.length;
      var curDa;
      //we do array.join('text that must not be PDFescaped")
      //thus, pdfEscape each component separately
      while (len--) {
        curDa = sa.shift();
        if (
          typeof curDa !== "string" ||
          (Array.isArray(curDa) && typeof curDa[0] !== "string")
        ) {
          tmpTextIsOfTypeString = false;
        }
      }
      textIsOfTypeString = tmpTextIsOfTypeString;
    }
    if (textIsOfTypeString === false) {
      throw new Error(
        'Type of text must be string or Array. "' +
          text +
          '" is not recognized.'
      );
    }

    //If there are any newlines in text, we assume
    //the user wanted to print multiple lines, so break the
    //text up into an array. If the text is already an array,
    //we assume the user knows what they are doing.
    //Convert text into an array anyway to simplify
    //later code.

    if (typeof text === "string") {
      if (text.match(/[\r?\n]/)) {
        text = text.split(/\r\n|\r|\n/g);
      } else {
        text = [text];
      }
    }

    //baseline
    var height = activeFontSize / scope.internal.scaleFactor;
    var descent = height * (lineHeight - 1);

    switch (options.baseline) {
      case "bottom":
        y -= descent;
        break;
      case "top":
        y += height - descent;
        break;
      case "hanging":
        y += height - 2 * descent;
        break;
      case "middle":
        y += height / 2 - descent;
        break;
    }

    //multiline
    maxWidth = options.maxWidth || 0;

    if (maxWidth > 0) {
      if (typeof text === "string") {
        text = scope.splitTextToSize(text, maxWidth);
      } else if (Object.prototype.toString.call(text) === "[object Array]") {
        text = text.reduce(function(acc, textLine) {
          return acc.concat(scope.splitTextToSize(textLine, maxWidth));
        }, []);
      }
    }

    //creating Payload-Object to make text byRef
    payload = {
      text: text,
      x: x,
      y: y,
      options: options,
      mutex: {
        pdfEscape: pdfEscape,
        activeFontKey: activeFontKey,
        fonts: fonts,
        activeFontSize: activeFontSize
      }
    };
    events.publish("preProcessText", payload);

    text = payload.text;
    options = payload.options;

    //angle
    angle = options.angle;

    if (
      transformationMatrix instanceof Matrix === false &&
      angle &&
      typeof angle === "number"
    ) {
      angle *= Math.PI / 180;

      if (options.rotationDirection === 0) {
        angle = -angle;
      }

      if (apiMode === ApiMode.ADVANCED) {
        angle = -angle;
      }

      var c = Math.cos(angle);
      var s = Math.sin(angle);
      transformationMatrix = new Matrix(c, s, -s, c, 0, 0);
    } else if (angle && angle instanceof Matrix) {
      transformationMatrix = angle;
    }

    if (apiMode === ApiMode.ADVANCED && !transformationMatrix) {
      transformationMatrix = identityMatrix;
    }

    //charSpace

    charSpace = options.charSpace || activeCharSpace;

    if (typeof charSpace !== "undefined") {
      xtra += hpf(scale(charSpace)) + " Tc\n";
      this.setCharSpace(this.getCharSpace() || 0);
    }

    horizontalScale = options.horizontalScale;
    if (typeof horizontalScale !== "undefined") {
      xtra += hpf(horizontalScale * 100) + " Tz\n";
    }

    //lang

    options.lang;

    //renderingMode
    var renderingMode = -1;
    var parmRenderingMode =
      typeof options.renderingMode !== "undefined"
        ? options.renderingMode
        : options.stroke;
    var pageContext = scope.internal.getCurrentPageInfo().pageContext;

    switch (parmRenderingMode) {
      case 0:
      case false:
      case "fill":
        renderingMode = 0;
        break;
      case 1:
      case true:
      case "stroke":
        renderingMode = 1;
        break;
      case 2:
      case "fillThenStroke":
        renderingMode = 2;
        break;
      case 3:
      case "invisible":
        renderingMode = 3;
        break;
      case 4:
      case "fillAndAddForClipping":
        renderingMode = 4;
        break;
      case 5:
      case "strokeAndAddPathForClipping":
        renderingMode = 5;
        break;
      case 6:
      case "fillThenStrokeAndAddToPathForClipping":
        renderingMode = 6;
        break;
      case 7:
      case "addToPathForClipping":
        renderingMode = 7;
        break;
    }

    var usedRenderingMode =
      typeof pageContext.usedRenderingMode !== "undefined"
        ? pageContext.usedRenderingMode
        : -1;

    //if the coder wrote it explicitly to use a specific
    //renderingMode, then use it
    if (renderingMode !== -1) {
      xtra += renderingMode + " Tr\n";
      //otherwise check if we used the rendering Mode already
      //if so then set the rendering Mode...
    } else if (usedRenderingMode !== -1) {
      xtra += "0 Tr\n";
    }

    if (renderingMode !== -1) {
      pageContext.usedRenderingMode = renderingMode;
    }

    //align
    align = options.align || "left";
    var leading = activeFontSize * lineHeight;
    var pageWidth = scope.internal.pageSize.getWidth();
    var activeFont = fonts[activeFontKey];
    charSpace = options.charSpace || activeCharSpace;
    maxWidth = options.maxWidth || 0;

    var lineWidths;
    flags = Object.assign({ autoencode: true, noBOM: true }, options.flags);

    var wordSpacingPerLine = [];
    var findWidth = function(v) {
      return (
        (scope.getStringUnitWidth(v, {
          font: activeFont,
          charSpace: charSpace,
          fontSize: activeFontSize,
          doKerning: false
        }) *
          activeFontSize) /
        scaleFactor
      );
    };
    if (Object.prototype.toString.call(text) === "[object Array]") {
      da = transformTextToSpecialArray(text);
      var newY;
      if (align !== "left") {
        lineWidths = da.map(findWidth);
      }
      //The first line uses the "main" Td setting,
      //and the subsequent lines are offset by the
      //previous line's x coordinate.
      var prevWidth = 0;
      var newX;
      if (align === "right") {
        //The passed in x coordinate defines the
        //rightmost point of the text.
        x -= lineWidths[0];
        text = [];
        len = da.length;
        for (var i = 0; i < len; i++) {
          if (i === 0) {
            newX = getHorizontalCoordinate(x);
            newY = getVerticalCoordinate(y);
          } else {
            newX = scale(prevWidth - lineWidths[i]);
            newY = -leading;
          }
          text.push([da[i], newX, newY]);
          prevWidth = lineWidths[i];
        }
      } else if (align === "center") {
        //The passed in x coordinate defines
        //the center point.
        x -= lineWidths[0] / 2;
        text = [];
        len = da.length;
        for (var j = 0; j < len; j++) {
          if (j === 0) {
            newX = getHorizontalCoordinate(x);
            newY = getVerticalCoordinate(y);
          } else {
            newX = scale((prevWidth - lineWidths[j]) / 2);
            newY = -leading;
          }
          text.push([da[j], newX, newY]);
          prevWidth = lineWidths[j];
        }
      } else if (align === "left") {
        text = [];
        len = da.length;
        for (var h = 0; h < len; h++) {
          text.push(da[h]);
        }
      } else if (align === "justify" && activeFont.encoding === "Identity-H") {
        // when using unicode fonts, wordSpacePerLine does not apply
        text = [];
        len = da.length;
        maxWidth = maxWidth !== 0 ? maxWidth : pageWidth;
        let backToStartX = 0;
        for (var l = 0; l < len; l++) {
          newY = l === 0 ? getVerticalCoordinate(y) : -leading;
          newX = l === 0 ? getHorizontalCoordinate(x) : backToStartX;
          if (l < len - 1) {
            let spacing = scale(
              (maxWidth - lineWidths[l]) / (da[l].split(" ").length - 1)
            );
            let words = da[l].split(" ");
            text.push([words[0] + " ", newX, newY]);
            backToStartX = 0; // distance to reset back to the left
            for (let i = 1; i < words.length; i++) {
              let shiftAmount =
                (findWidth(words[i - 1] + " " + words[i]) -
                  findWidth(words[i])) *
                  scaleFactor +
                spacing;
              if (i == words.length - 1) text.push([words[i], shiftAmount, 0]);
              else text.push([words[i] + " ", shiftAmount, 0]);
              backToStartX -= shiftAmount;
            }
          } else {
            text.push([da[l], newX, newY]);
          }
        }
        text.push(["", backToStartX, 0]);
      } else if (align === "justify") {
        text = [];
        len = da.length;
        maxWidth = maxWidth !== 0 ? maxWidth : pageWidth;
        for (var l = 0; l < len; l++) {
          newY = l === 0 ? getVerticalCoordinate(y) : -leading;
          newX = l === 0 ? getHorizontalCoordinate(x) : 0;

          const numSpaces = da[l].split(" ").length - 1;
          const spacing =
            numSpaces > 0 ? (maxWidth - lineWidths[l]) / numSpaces : 0;

          if (l < len - 1) {
            wordSpacingPerLine.push(hpf(scale(spacing)));
          } else {
            wordSpacingPerLine.push(0);
          }
          text.push([da[l], newX, newY]);
        }
      } else {
        throw new Error(
          'Unrecognized alignment option, use "left", "center", "right" or "justify".'
        );
      }
    }

    //R2L
    var doReversing = typeof options.R2L === "boolean" ? options.R2L : R2L;
    if (doReversing === true) {
      text = processTextByFunction(text, function(text, posX, posY) {
        return [
          text
            .split("")
            .reverse()
            .join(""),
          posX,
          posY
        ];
      });
    }

    //creating Payload-Object to make text byRef
    payload = {
      text: text,
      x: x,
      y: y,
      options: options,
      mutex: {
        pdfEscape: pdfEscape,
        activeFontKey: activeFontKey,
        fonts: fonts,
        activeFontSize: activeFontSize
      }
    };
    events.publish("postProcessText", payload);

    text = payload.text;
    isHex = payload.mutex.isHex || false;

    //Escaping
    var activeFontEncoding = fonts[activeFontKey].encoding;

    if (
      activeFontEncoding === "WinAnsiEncoding" ||
      activeFontEncoding === "StandardEncoding"
    ) {
      text = processTextByFunction(text, function(text, posX, posY) {
        return [ESC(text), posX, posY];
      });
    }

    da = transformTextToSpecialArray(text);

    text = [];
    var STRING = 0;
    var ARRAY = 1;
    var variant = Array.isArray(da[0]) ? ARRAY : STRING;
    var posX;
    var posY;
    var content;
    var wordSpacing = "";

    var generatePosition = function(
      parmPosX,
      parmPosY,
      parmTransformationMatrix
    ) {
      var position = "";
      if (parmTransformationMatrix instanceof Matrix) {
        // It is kind of more intuitive to apply a plain rotation around the text anchor set by x and y
        // but when the user supplies an arbitrary transformation matrix, the x and y offsets should be applied
        // in the coordinate system established by this matrix
        if (typeof options.angle === "number") {
          parmTransformationMatrix = matrixMult(
            parmTransformationMatrix,
            new Matrix(1, 0, 0, 1, parmPosX, parmPosY)
          );
        } else {
          parmTransformationMatrix = matrixMult(
            new Matrix(1, 0, 0, 1, parmPosX, parmPosY),
            parmTransformationMatrix
          );
        }

        if (apiMode === ApiMode.ADVANCED) {
          parmTransformationMatrix = matrixMult(
            new Matrix(1, 0, 0, -1, 0, 0),
            parmTransformationMatrix
          );
        }

        position = parmTransformationMatrix.join(" ") + " Tm\n";
      } else {
        position = hpf(parmPosX) + " " + hpf(parmPosY) + " Td\n";
      }
      return position;
    };

    for (var lineIndex = 0; lineIndex < da.length; lineIndex++) {
      wordSpacing = "";

      switch (variant) {
        case ARRAY:
          content =
            (isHex ? "<" : "(") + da[lineIndex][0] + (isHex ? ">" : ")");
          posX = parseFloat(da[lineIndex][1]);
          posY = parseFloat(da[lineIndex][2]);
          break;
        case STRING:
          content = (isHex ? "<" : "(") + da[lineIndex] + (isHex ? ">" : ")");
          posX = getHorizontalCoordinate(x);
          posY = getVerticalCoordinate(y);
          break;
      }

      if (
        typeof wordSpacingPerLine !== "undefined" &&
        typeof wordSpacingPerLine[lineIndex] !== "undefined"
      ) {
        wordSpacing = wordSpacingPerLine[lineIndex] + " Tw\n";
      }

      if (lineIndex === 0) {
        text.push(
          wordSpacing +
            generatePosition(posX, posY, transformationMatrix) +
            content
        );
      } else if (variant === STRING) {
        text.push(wordSpacing + content);
      } else if (variant === ARRAY) {
        text.push(
          wordSpacing +
            generatePosition(posX, posY, transformationMatrix) +
            content
        );
      }
    }

    text = variant === STRING ? text.join(" Tj\nT* ") : text.join(" Tj\n");
    text += " Tj\n";

    var result = "BT\n/";
    result += activeFontKey + " " + activeFontSize + " Tf\n"; // font face, style, size
    result += hpf(activeFontSize * lineHeight) + " TL\n"; // line spacing
    result += textColor + "\n";
    result += xtra;
    result += text;
    result += "ET";

    out(result);
    usedFonts[activeFontKey] = true;
    return scope;
  };

  // PDF supports these path painting and clip path operators:
  //
  // S - stroke
  // s - close/stroke
  // f (F) - fill non-zero
  // f* - fill evenodd
  // B - fill stroke nonzero
  // B* - fill stroke evenodd
  // b - close fill stroke nonzero
  // b* - close fill stroke evenodd
  // n - nothing (consume path)
  // W - clip nonzero
  // W* - clip evenodd
  //
  // In order to keep the API small, we omit the close-and-fill/stroke operators and provide a separate close()
  // method.
  /**
   *
   * @name clip
   * @function
   * @instance
   * @param {string} rule Only possible value is 'evenodd'
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @description All .clip() after calling drawing ops with a style argument of null.
   */
  var clip = (API.__private__.clip = API.clip = function(rule) {
    // Call .clip() after calling drawing ops with a style argument of null
    // W is the PDF clipping op
    if ("evenodd" === rule) {
      out("W*");
    } else {
      out("W");
    }
    return this;
  });

  /**
   * @name clipEvenOdd
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @description Modify the current clip path by intersecting it with the current path using the even-odd rule. Note
   * that this will NOT consume the current path. In order to only use this path for clipping call
   * {@link API.discardPath} afterwards.
   */
  API.clipEvenOdd = function() {
    return clip("evenodd");
  };

  /**
   * Consumes the current path without any effect. Mainly used in combination with {@link clip} or
   * {@link clipEvenOdd}. The PDF "n" operator.
   * @name discardPath
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.__private__.discardPath = API.discardPath = function() {
    out("n");
    return this;
  };

  var isValidStyle = (API.__private__.isValidStyle = function(style) {
    var validStyleVariants = [
      undefined,
      null,
      "S",
      "D",
      "F",
      "DF",
      "FD",
      "f",
      "f*",
      "B",
      "B*",
      "n"
    ];
    var result = false;
    if (validStyleVariants.indexOf(style) !== -1) {
      result = true;
    }
    return result;
  });

  API.__private__.setDefaultPathOperation = API.setDefaultPathOperation = function(
    operator
  ) {
    if (isValidStyle(operator)) {
      defaultPathOperation = operator;
    }
    return this;
  };

  var getStyle = (API.__private__.getStyle = API.getStyle = function(style) {
    // see path-painting operators in PDF spec
    var op = defaultPathOperation; // stroke

    switch (style) {
      case "D":
      case "S":
        op = "S"; // stroke
        break;
      case "F":
        op = "f"; // fill
        break;
      case "FD":
      case "DF":
        op = "B";
        break;
      case "f":
      case "f*":
      case "B":
      case "B*":
        /*
               Allow direct use of these PDF path-painting operators:
               - f    fill using nonzero winding number rule
               - f*    fill using even-odd rule
               - B    fill then stroke with fill using non-zero winding number rule
               - B*    fill then stroke with fill using even-odd rule
               */
        op = style;
        break;
    }
    return op;
  });

  /**
   * Close the current path. The PDF "h" operator.
   * @name close
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  var close = (API.close = function() {
    out("h");
    return this;
  });

  /**
   * Stroke the path. The PDF "S" operator.
   * @name stroke
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.stroke = function() {
    out("S");
    return this;
  };

  /**
   * Fill the current path using the nonzero winding number rule. If a pattern is provided, the path will be filled
   * with this pattern, otherwise with the current fill color. Equivalent to the PDF "f" operator.
   * @name fill
   * @function
   * @instance
   * @param {PatternData=} pattern If provided the path will be filled with this pattern
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.fill = function(pattern) {
    fillWithOptionalPattern("f", pattern);
    return this;
  };

  /**
   * Fill the current path using the even-odd rule. The PDF f* operator.
   * @see API.fill
   * @name fillEvenOdd
   * @function
   * @instance
   * @param {PatternData=} pattern If provided the path will be filled with this pattern
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.fillEvenOdd = function(pattern) {
    fillWithOptionalPattern("f*", pattern);
    return this;
  };

  /**
   * Fill using the nonzero winding number rule and then stroke the current Path. The PDF "B" operator.
   * @see API.fill
   * @name fillStroke
   * @function
   * @instance
   * @param {PatternData=} pattern If provided the path will be stroked with this pattern
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.fillStroke = function(pattern) {
    fillWithOptionalPattern("B", pattern);
    return this;
  };

  /**
   * Fill using the even-odd rule and then stroke the current Path. The PDF "B" operator.
   * @see API.fill
   * @name fillStrokeEvenOdd
   * @function
   * @instance
   * @param {PatternData=} pattern If provided the path will be fill-stroked with this pattern
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.fillStrokeEvenOdd = function(pattern) {
    fillWithOptionalPattern("B*", pattern);
    return this;
  };

  var fillWithOptionalPattern = function(style, pattern) {
    if (typeof pattern === "object") {
      fillWithPattern(pattern, style);
    } else {
      out(style);
    }
  };

  var putStyle = function(style) {
    if (
      style === null ||
      (apiMode === ApiMode.ADVANCED && style === undefined)
    ) {
      return;
    }

    style = getStyle(style);

    // stroking / filling / both the path
    out(style);
  };

  function cloneTilingPattern(patternKey, boundingBox, xStep, yStep, matrix) {
    var clone = new TilingPattern(
      boundingBox || this.boundingBox,
      xStep || this.xStep,
      yStep || this.yStep,
      this.gState,
      matrix || this.matrix
    );
    clone.stream = this.stream;
    var key = patternKey + "$$" + this.cloneIndex++ + "$$";
    addPattern(key, clone);
    return clone;
  }

  var fillWithPattern = function(patternData, style) {
    var patternId = patternMap[patternData.key];
    var pattern = patterns[patternId];

    if (pattern instanceof ShadingPattern) {
      out("q");

      out(clipRuleFromStyle(style));

      if (pattern.gState) {
        API.setGState(pattern.gState);
      }
      out(patternData.matrix.toString() + " cm");
      out("/" + patternId + " sh");
      out("Q");
    } else if (pattern instanceof TilingPattern) {
      // pdf draws patterns starting at the bottom left corner and they are not affected by the global transformation,
      // so we must flip them
      var matrix = new Matrix(1, 0, 0, -1, 0, getPageHeight());

      if (patternData.matrix) {
        matrix = matrix.multiply(patternData.matrix || identityMatrix);
        // we cannot apply a matrix to the pattern on use so we must abuse the pattern matrix and create new instances
        // for each use
        patternId = cloneTilingPattern.call(
          pattern,
          patternData.key,
          patternData.boundingBox,
          patternData.xStep,
          patternData.yStep,
          matrix
        ).id;
      }

      out("q");
      out("/Pattern cs");
      out("/" + patternId + " scn");

      if (pattern.gState) {
        API.setGState(pattern.gState);
      }

      out(style);
      out("Q");
    }
  };

  var clipRuleFromStyle = function(style) {
    switch (style) {
      case "f":
      case "F":
        return "W n";
      case "f*":
        return "W* n";
      case "B":
        return "W S";
      case "B*":
        return "W* S";

      // these two are for compatibility reasons (in the past, calling any primitive method with a shading pattern
      // and "n"/"S" as style would still fill/fill and stroke the path)
      case "S":
        return "W S";
      case "n":
        return "W n";
    }
  };

  /**
   * Begin a new subpath by moving the current point to coordinates (x, y). The PDF "m" operator.
   * @param {number} x
   * @param {number} y
   * @name moveTo
   * @function
   * @instance
   * @memberof jsPDF#
   * @returns {jsPDF}
   */
  var moveTo = (API.moveTo = function(x, y) {
    out(hpf(scale(x)) + " " + hpf(transformScaleY(y)) + " m");
    return this;
  });

  /**
   * Append a straight line segment from the current point to the point (x, y). The PDF "l" operator.
   * @param {number} x
   * @param {number} y
   * @memberof jsPDF#
   * @name lineTo
   * @function
   * @instance
   * @memberof jsPDF#
   * @returns {jsPDF}
   */
  var lineTo = (API.lineTo = function(x, y) {
    out(hpf(scale(x)) + " " + hpf(transformScaleY(y)) + " l");
    return this;
  });

  /**
   * Append a cubic Bézier curve to the current path. The curve shall extend from the current point to the point
   * (x3, y3), using (x1, y1) and (x2, y2) as Bézier control points. The new current point shall be (x3, x3).
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {number} x3
   * @param {number} y3
   * @memberof jsPDF#
   * @name curveTo
   * @function
   * @instance
   * @memberof jsPDF#
   * @returns {jsPDF}
   */
  var curveTo = (API.curveTo = function(x1, y1, x2, y2, x3, y3) {
    out(
      [
        hpf(scale(x1)),
        hpf(transformScaleY(y1)),
        hpf(scale(x2)),
        hpf(transformScaleY(y2)),
        hpf(scale(x3)),
        hpf(transformScaleY(y3)),
        "c"
      ].join(" ")
    );
    return this;
  });

  /**
   * Draw a line on the current page.
   *
   * @name line
   * @function
   * @instance
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {string} style A string specifying the painting style or null.  Valid styles include: 'S' [default] - stroke, 'F' - fill,  and 'DF' (or 'FD') -  fill then stroke. A null value postpones setting the style so that a shape may be composed using multiple method calls. The last drawing method call used to define the shape should not have a null style argument. default: 'S'
   * @returns {jsPDF}
   * @memberof jsPDF#
   */
  API.__private__.line = API.line = function(x1, y1, x2, y2, style) {
    if (
      isNaN(x1) ||
      isNaN(y1) ||
      isNaN(x2) ||
      isNaN(y2) ||
      !isValidStyle(style)
    ) {
      throw new Error("Invalid arguments passed to jsPDF.line");
    }
    if (apiMode === ApiMode.COMPAT) {
      return this.lines([[x2 - x1, y2 - y1]], x1, y1, [1, 1], style || "S");
    } else {
      return this.lines([[x2 - x1, y2 - y1]], x1, y1, [1, 1]).stroke();
    }
  };

  /**
   * @typedef {Object} PatternData
   * {Matrix|undefined} matrix
   * {Number|undefined} xStep
   * {Number|undefined} yStep
   * {Array.<Number>|undefined} boundingBox
   */

  /**
   * Adds series of curves (straight lines or cubic bezier curves) to canvas, starting at `x`, `y` coordinates.
   * All data points in `lines` are relative to last line origin.
   * `x`, `y` become x1,y1 for first line / curve in the set.
   * For lines you only need to specify [x2, y2] - (ending point) vector against x1, y1 starting point.
   * For bezier curves you need to specify [x2,y2,x3,y3,x4,y4] - vectors to control points 1, 2, ending point. All vectors are against the start of the curve - x1,y1.
   *
   * @example .lines([[2,2],[-2,2],[1,1,2,2,3,3],[2,1]], 212,110, [1,1], 'F', false) // line, line, bezier curve, line
   * @param {Array} lines Array of *vector* shifts as pairs (lines) or sextets (cubic bezier curves).
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} scale (Defaults to [1.0,1.0]) x,y Scaling factor for all vectors. Elements can be any floating number Sub-one makes drawing smaller. Over-one grows the drawing. Negative flips the direction.
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @param {Boolean=} closed If true, the path is closed with a straight line from the end of the last curve to the starting point.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name lines
   */
  API.__private__.lines = API.lines = function(
    lines,
    x,
    y,
    scale,
    style,
    closed
  ) {
    var scalex, scaley, i, l, leg, x2, y2, x3, y3, x4, y4, tmp;

    // Pre-August-2012 the order of arguments was function(x, y, lines, scale, style)
    // in effort to make all calls have similar signature like
    //   function(content, coordinateX, coordinateY , miscellaneous)
    // this method had its args flipped.
    // code below allows backward compatibility with old arg order.
    if (typeof lines === "number") {
      tmp = y;
      y = x;
      x = lines;
      lines = tmp;
    }

    scale = scale || [1, 1];
    closed = closed || false;

    if (
      isNaN(x) ||
      isNaN(y) ||
      !Array.isArray(lines) ||
      !Array.isArray(scale) ||
      !isValidStyle(style) ||
      typeof closed !== "boolean"
    ) {
      throw new Error("Invalid arguments passed to jsPDF.lines");
    }

    // starting point
    moveTo(x, y);

    scalex = scale[0];
    scaley = scale[1];
    l = lines.length;
    //, x2, y2 // bezier only. In page default measurement "units", *after* scaling
    //, x3, y3 // bezier only. In page default measurement "units", *after* scaling
    // ending point for all, lines and bezier. . In page default measurement "units", *after* scaling
    x4 = x; // last / ending point = starting point for first item.
    y4 = y; // last / ending point = starting point for first item.

    for (i = 0; i < l; i++) {
      leg = lines[i];
      if (leg.length === 2) {
        // simple line
        x4 = leg[0] * scalex + x4; // here last x4 was prior ending point
        y4 = leg[1] * scaley + y4; // here last y4 was prior ending point
        lineTo(x4, y4);
      } else {
        // bezier curve
        x2 = leg[0] * scalex + x4; // here last x4 is prior ending point
        y2 = leg[1] * scaley + y4; // here last y4 is prior ending point
        x3 = leg[2] * scalex + x4; // here last x4 is prior ending point
        y3 = leg[3] * scaley + y4; // here last y4 is prior ending point
        x4 = leg[4] * scalex + x4; // here last x4 was prior ending point
        y4 = leg[5] * scaley + y4; // here last y4 was prior ending point
        curveTo(x2, y2, x3, y3, x4, y4);
      }
    }

    if (closed) {
      close();
    }

    putStyle(style);
    return this;
  };

  /**
   * Similar to {@link API.lines} but all coordinates are interpreted as absolute coordinates instead of relative.
   * @param {Array<Object>} lines An array of {op: operator, c: coordinates} object, where op is one of "m" (move to), "l" (line to)
   * "c" (cubic bezier curve) and "h" (close (sub)path)). c is an array of coordinates. "m" and "l" expect two, "c"
   * six and "h" an empty array (or undefined).
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name path
   */
  API.path = function(lines) {
    for (var i = 0; i < lines.length; i++) {
      var leg = lines[i];
      var coords = leg.c;
      switch (leg.op) {
        case "m":
          moveTo(coords[0], coords[1]);
          break;
        case "l":
          lineTo(coords[0], coords[1]);
          break;
        case "c":
          curveTo.apply(this, coords);
          break;
        case "h":
          close();
          break;
      }
    }

    return this;
  };

  /**
   * Adds a rectangle to PDF.
   *
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} w Width (in units declared at inception of PDF document)
   * @param {number} h Height (in units declared at inception of PDF document)
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name rect
   */
  API.__private__.rect = API.rect = function(x, y, w, h, style) {
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h) || !isValidStyle(style)) {
      throw new Error("Invalid arguments passed to jsPDF.rect");
    }
    if (apiMode === ApiMode.COMPAT) {
      h = -h;
    }

    out(
      [
        hpf(scale(x)),
        hpf(transformScaleY(y)),
        hpf(scale(w)),
        hpf(scale(h)),
        "re"
      ].join(" ")
    );

    putStyle(style);
    return this;
  };

  /**
   * Adds a triangle to PDF.
   *
   * @param {number} x1 Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y1 Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} x2 Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y2 Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} x3 Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y3 Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name triangle
   */
  API.__private__.triangle = API.triangle = function(
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    style
  ) {
    if (
      isNaN(x1) ||
      isNaN(y1) ||
      isNaN(x2) ||
      isNaN(y2) ||
      isNaN(x3) ||
      isNaN(y3) ||
      !isValidStyle(style)
    ) {
      throw new Error("Invalid arguments passed to jsPDF.triangle");
    }
    this.lines(
      [
        [x2 - x1, y2 - y1], // vector to point 2
        [x3 - x2, y3 - y2], // vector to point 3
        [x1 - x3, y1 - y3] // closing vector back to point 1
      ],
      x1,
      y1, // start of path
      [1, 1],
      style,
      true
    );
    return this;
  };

  /**
   * Adds a rectangle with rounded corners to PDF.
   *
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} w Width (in units declared at inception of PDF document)
   * @param {number} h Height (in units declared at inception of PDF document)
   * @param {number} rx Radius along x axis (in units declared at inception of PDF document)
   * @param {number} ry Radius along y axis (in units declared at inception of PDF document)
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name roundedRect
   */
  API.__private__.roundedRect = API.roundedRect = function(
    x,
    y,
    w,
    h,
    rx,
    ry,
    style
  ) {
    if (
      isNaN(x) ||
      isNaN(y) ||
      isNaN(w) ||
      isNaN(h) ||
      isNaN(rx) ||
      isNaN(ry) ||
      !isValidStyle(style)
    ) {
      throw new Error("Invalid arguments passed to jsPDF.roundedRect");
    }
    var MyArc = (4 / 3) * (Math.SQRT2 - 1);

    rx = Math.min(rx, w * 0.5);
    ry = Math.min(ry, h * 0.5);

    this.lines(
      [
        [w - 2 * rx, 0],
        [rx * MyArc, 0, rx, ry - ry * MyArc, rx, ry],
        [0, h - 2 * ry],
        [0, ry * MyArc, -(rx * MyArc), ry, -rx, ry],
        [-w + 2 * rx, 0],
        [-(rx * MyArc), 0, -rx, -(ry * MyArc), -rx, -ry],
        [0, -h + 2 * ry],
        [0, -(ry * MyArc), rx * MyArc, -ry, rx, -ry]
      ],
      x + rx,
      y, // start of path
      [1, 1],
      style,
      true
    );
    return this;
  };

  /**
   * Adds an ellipse to PDF.
   *
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} rx Radius along x axis (in units declared at inception of PDF document)
   * @param {number} ry Radius along y axis (in units declared at inception of PDF document)
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name ellipse
   */
  API.__private__.ellipse = API.ellipse = function(x, y, rx, ry, style) {
    if (
      isNaN(x) ||
      isNaN(y) ||
      isNaN(rx) ||
      isNaN(ry) ||
      !isValidStyle(style)
    ) {
      throw new Error("Invalid arguments passed to jsPDF.ellipse");
    }
    var lx = (4 / 3) * (Math.SQRT2 - 1) * rx,
      ly = (4 / 3) * (Math.SQRT2 - 1) * ry;

    moveTo(x + rx, y);
    curveTo(x + rx, y - ly, x + lx, y - ry, x, y - ry);
    curveTo(x - lx, y - ry, x - rx, y - ly, x - rx, y);
    curveTo(x - rx, y + ly, x - lx, y + ry, x, y + ry);
    curveTo(x + lx, y + ry, x + rx, y + ly, x + rx, y);

    putStyle(style);
    return this;
  };

  /**
   * Adds an circle to PDF.
   *
   * @param {number} x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} r Radius (in units declared at inception of PDF document)
   * @param {string=} style A string specifying the painting style or null. Valid styles include:
   * 'S' [default] - stroke,
   * 'F' - fill,
   * and 'DF' (or 'FD') -  fill then stroke.
   * In "compat" API mode, a null value postpones setting the style so that a shape may be composed using multiple
   * method calls. The last drawing method call used to define the shape should not have a null style argument.
   *
   * In "advanced" API mode this parameter is deprecated.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name circle
   */
  API.__private__.circle = API.circle = function(x, y, r, style) {
    if (isNaN(x) || isNaN(y) || isNaN(r) || !isValidStyle(style)) {
      throw new Error("Invalid arguments passed to jsPDF.circle");
    }
    return this.ellipse(x, y, r, r, style);
  };

  /**
   * Sets text font face, variant for upcoming text elements.
   * See output of jsPDF.getFontList() for possible font names, styles.
   *
   * @param {string} fontName Font name or family. Example: "times".
   * @param {string} fontStyle Font style or variant. Example: "italic".
   * @param {number | string} fontWeight Weight of the Font. Example: "normal" | 400
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setFont
   */
  API.setFont = function(fontName, fontStyle, fontWeight) {
    if (fontWeight) {
      fontStyle = combineFontStyleAndFontWeight(fontStyle, fontWeight);
    }
    activeFontKey = getFont(fontName, fontStyle, {
      disableWarning: false
    });
    return this;
  };

  /**
   * Gets text font face, variant for upcoming text elements.
   *
   * @function
   * @instance
   * @returns {Object}
   * @memberof jsPDF#
   * @name getFont
   */
  var getFontEntry = (API.__private__.getFont = API.getFont = function() {
    return fonts[getFont.apply(API, arguments)];
  });

  /**
   * Returns an object - a tree of fontName to fontStyle relationships available to
   * active PDF document.
   *
   * @public
   * @function
   * @instance
   * @returns {Object} Like {'times':['normal', 'italic', ... ], 'arial':['normal', 'bold', ... ], ... }
   * @memberof jsPDF#
   * @name getFontList
   */
  API.__private__.getFontList = API.getFontList = function() {
    var list = {},
      fontName,
      fontStyle;

    for (fontName in fontmap) {
      if (fontmap.hasOwnProperty(fontName)) {
        list[fontName] = [];
        for (fontStyle in fontmap[fontName]) {
          if (fontmap[fontName].hasOwnProperty(fontStyle)) {
            list[fontName].push(fontStyle);
          }
        }
      }
    }
    return list;
  };

  /**
   * Add a custom font to the current instance.
   *
   * @param {string} postScriptName PDF specification full name for the font.
   * @param {string} id PDF-document-instance-specific label assinged to the font.
   * @param {string} fontStyle Style of the Font.
   * @param {number | string} fontWeight Weight of the Font.
   * @param {Object} encoding Encoding_name-to-Font_metrics_object mapping.
   * @function
   * @instance
   * @memberof jsPDF#
   * @name addFont
   * @returns {string} fontId
   */
  API.addFont = function(
    postScriptName,
    fontName,
    fontStyle,
    fontWeight,
    encoding
  ) {
    var encodingOptions = [
      "StandardEncoding",
      "MacRomanEncoding",
      "Identity-H",
      "WinAnsiEncoding"
    ];
    if (arguments[3] && encodingOptions.indexOf(arguments[3]) !== -1) {
      //IE 11 fix
      encoding = arguments[3];
    } else if (arguments[3] && encodingOptions.indexOf(arguments[3]) == -1) {
      fontStyle = combineFontStyleAndFontWeight(fontStyle, fontWeight);
    }
    encoding = encoding || "Identity-H";
    return addFont.call(this, postScriptName, fontName, fontStyle, encoding);
  };

  var lineWidth = options.lineWidth || 0.200025; // 2mm
  /**
   * Gets the line width, default: 0.200025.
   *
   * @function
   * @instance
   * @returns {number} lineWidth
   * @memberof jsPDF#
   * @name getLineWidth
   */
  var getLineWidth = (API.__private__.getLineWidth = API.getLineWidth = function() {
    return lineWidth;
  });

  /**
   * Sets line width for upcoming lines.
   *
   * @param {number} width Line width (in units declared at inception of PDF document).
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineWidth
   */
  var setLineWidth = (API.__private__.setLineWidth = API.setLineWidth = function(
    width
  ) {
    lineWidth = width;
    out(hpf(scale(width)) + " w");
    return this;
  });

  /**
   * Sets the dash pattern for upcoming lines.
   *
   * To reset the settings simply call the method without any parameters.
   * @param {Array<number>} dashArray An array containing 0-2 numbers. The first number sets the length of the
   * dashes, the second number the length of the gaps. If the second number is missing, the gaps are considered
   * to be as long as the dashes. An empty array means solid, unbroken lines.
   * @param {number} dashPhase The phase lines start with.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineDashPattern
   */
  API.__private__.setLineDash = jsPDF.API.setLineDash = jsPDF.API.setLineDashPattern = function(
    dashArray,
    dashPhase
  ) {
    dashArray = dashArray || [];
    dashPhase = dashPhase || 0;

    if (isNaN(dashPhase) || !Array.isArray(dashArray)) {
      throw new Error("Invalid arguments passed to jsPDF.setLineDash");
    }

    dashArray = dashArray
      .map(function(x) {
        return hpf(scale(x));
      })
      .join(" ");
    dashPhase = hpf(scale(dashPhase));

    out("[" + dashArray + "] " + dashPhase + " d");
    return this;
  };

  var lineHeightFactor;

  var getLineHeight = (API.__private__.getLineHeight = API.getLineHeight = function() {
    return activeFontSize * lineHeightFactor;
  });

  API.__private__.getLineHeight = API.getLineHeight = function() {
    return activeFontSize * lineHeightFactor;
  };

  /**
   * Sets the LineHeightFactor of proportion.
   *
   * @param {number} value LineHeightFactor value. Default: 1.15.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineHeightFactor
   */
  var setLineHeightFactor = (API.__private__.setLineHeightFactor = API.setLineHeightFactor = function(
    value
  ) {
    value = value || 1.15;
    if (typeof value === "number") {
      lineHeightFactor = value;
    }
    return this;
  });

  /**
   * Gets the LineHeightFactor, default: 1.15.
   *
   * @function
   * @instance
   * @returns {number} lineHeightFactor
   * @memberof jsPDF#
   * @name getLineHeightFactor
   */
  var getLineHeightFactor = (API.__private__.getLineHeightFactor = API.getLineHeightFactor = function() {
    return lineHeightFactor;
  });

  setLineHeightFactor(options.lineHeight);

  var getHorizontalCoordinate = (API.__private__.getHorizontalCoordinate = function(
    value
  ) {
    return scale(value);
  });

  var getVerticalCoordinate = (API.__private__.getVerticalCoordinate = function(
    value
  ) {
    if (apiMode === ApiMode.ADVANCED) {
      return value;
    } else {
      var pageHeight =
        pagesContext[currentPage].mediaBox.topRightY -
        pagesContext[currentPage].mediaBox.bottomLeftY;
      return pageHeight - scale(value);
    }
  });

  var getHorizontalCoordinateString = (API.__private__.getHorizontalCoordinateString = API.getHorizontalCoordinateString = function(
    value
  ) {
    return hpf(getHorizontalCoordinate(value));
  });

  var getVerticalCoordinateString = (API.__private__.getVerticalCoordinateString = API.getVerticalCoordinateString = function(
    value
  ) {
    return hpf(getVerticalCoordinate(value));
  });

  var strokeColor = options.strokeColor || "0 G";

  /**
   *  Gets the stroke color for upcoming elements.
   *
   * @function
   * @instance
   * @returns {string} colorAsHex
   * @memberof jsPDF#
   * @name getDrawColor
   */
  API.__private__.getStrokeColor = API.getDrawColor = function() {
    return decodeColorString(strokeColor);
  };

  /**
   * Sets the stroke color for upcoming elements.
   *
   * Depending on the number of arguments given, Gray, RGB, or CMYK
   * color space is implied.
   *
   * When only ch1 is given, "Gray" color space is implied and it
   * must be a value in the range from 0.00 (solid black) to to 1.00 (white)
   * if values are communicated as String types, or in range from 0 (black)
   * to 255 (white) if communicated as Number type.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When only ch1,ch2,ch3 are given, "RGB" color space is implied and each
   * value must be in the range from 0.00 (minimum intensity) to to 1.00
   * (max intensity) if values are communicated as String types, or
   * from 0 (min intensity) to to 255 (max intensity) if values are communicated
   * as Number types.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When ch1,ch2,ch3,ch4 are given, "CMYK" color space is implied and each
   * value must be a in the range from 0.00 (0% concentration) to to
   * 1.00 (100% concentration)
   *
   * Because JavaScript treats fixed point numbers badly (rounds to
   * floating point nearest to binary representation) it is highly advised to
   * communicate the fractional numbers as String types, not JavaScript Number type.
   *
   * @param {Number|String} ch1 Color channel value or {string} ch1 color value in hexadecimal, example: '#FFFFFF'.
   * @param {Number} ch2 Color channel value.
   * @param {Number} ch3 Color channel value.
   * @param {Number} ch4 Color channel value.
   *
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setDrawColor
   */
  API.__private__.setStrokeColor = API.setDrawColor = function(
    ch1,
    ch2,
    ch3,
    ch4
  ) {
    var options = {
      ch1: ch1,
      ch2: ch2,
      ch3: ch3,
      ch4: ch4,
      pdfColorType: "draw",
      precision: 2
    };

    strokeColor = encodeColorString(options);
    out(strokeColor);
    return this;
  };

  var fillColor = options.fillColor || "0 g";

  /**
   * Gets the fill color for upcoming elements.
   *
   * @function
   * @instance
   * @returns {string} colorAsHex
   * @memberof jsPDF#
   * @name getFillColor
   */
  API.__private__.getFillColor = API.getFillColor = function() {
    return decodeColorString(fillColor);
  };

  /**
   * Sets the fill color for upcoming elements.
   *
   * Depending on the number of arguments given, Gray, RGB, or CMYK
   * color space is implied.
   *
   * When only ch1 is given, "Gray" color space is implied and it
   * must be a value in the range from 0.00 (solid black) to to 1.00 (white)
   * if values are communicated as String types, or in range from 0 (black)
   * to 255 (white) if communicated as Number type.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When only ch1,ch2,ch3 are given, "RGB" color space is implied and each
   * value must be in the range from 0.00 (minimum intensity) to to 1.00
   * (max intensity) if values are communicated as String types, or
   * from 0 (min intensity) to to 255 (max intensity) if values are communicated
   * as Number types.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When ch1,ch2,ch3,ch4 are given, "CMYK" color space is implied and each
   * value must be a in the range from 0.00 (0% concentration) to to
   * 1.00 (100% concentration)
   *
   * Because JavaScript treats fixed point numbers badly (rounds to
   * floating point nearest to binary representation) it is highly advised to
   * communicate the fractional numbers as String types, not JavaScript Number type.
   *
   * @param {Number|String} ch1 Color channel value or {string} ch1 color value in hexadecimal, example: '#FFFFFF'.
   * @param {Number} ch2 Color channel value.
   * @param {Number} ch3 Color channel value.
   * @param {Number} ch4 Color channel value.
   *
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setFillColor
   */
  API.__private__.setFillColor = API.setFillColor = function(
    ch1,
    ch2,
    ch3,
    ch4
  ) {
    var options = {
      ch1: ch1,
      ch2: ch2,
      ch3: ch3,
      ch4: ch4,
      pdfColorType: "fill",
      precision: 2
    };

    fillColor = encodeColorString(options);
    out(fillColor);
    return this;
  };

  var textColor = options.textColor || "0 g";
  /**
   * Gets the text color for upcoming elements.
   *
   * @function
   * @instance
   * @returns {string} colorAsHex
   * @memberof jsPDF#
   * @name getTextColor
   */
  var getTextColor = (API.__private__.getTextColor = API.getTextColor = function() {
    return decodeColorString(textColor);
  });
  /**
   * Sets the text color for upcoming elements.
   *
   * Depending on the number of arguments given, Gray, RGB, or CMYK
   * color space is implied.
   *
   * When only ch1 is given, "Gray" color space is implied and it
   * must be a value in the range from 0.00 (solid black) to to 1.00 (white)
   * if values are communicated as String types, or in range from 0 (black)
   * to 255 (white) if communicated as Number type.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When only ch1,ch2,ch3 are given, "RGB" color space is implied and each
   * value must be in the range from 0.00 (minimum intensity) to to 1.00
   * (max intensity) if values are communicated as String types, or
   * from 0 (min intensity) to to 255 (max intensity) if values are communicated
   * as Number types.
   * The RGB-like 0-255 range is provided for backward compatibility.
   *
   * When ch1,ch2,ch3,ch4 are given, "CMYK" color space is implied and each
   * value must be a in the range from 0.00 (0% concentration) to to
   * 1.00 (100% concentration)
   *
   * Because JavaScript treats fixed point numbers badly (rounds to
   * floating point nearest to binary representation) it is highly advised to
   * communicate the fractional numbers as String types, not JavaScript Number type.
   *
   * @param {Number|String} ch1 Color channel value or {string} ch1 color value in hexadecimal, example: '#FFFFFF'.
   * @param {Number} ch2 Color channel value.
   * @param {Number} ch3 Color channel value.
   * @param {Number} ch4 Color channel value.
   *
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setTextColor
   */
  API.__private__.setTextColor = API.setTextColor = function(
    ch1,
    ch2,
    ch3,
    ch4
  ) {
    var options = {
      ch1: ch1,
      ch2: ch2,
      ch3: ch3,
      ch4: ch4,
      pdfColorType: "text",
      precision: 3
    };
    textColor = encodeColorString(options);

    return this;
  };

  var activeCharSpace = options.charSpace;

  /**
   * Get global value of CharSpace.
   *
   * @function
   * @instance
   * @returns {number} charSpace
   * @memberof jsPDF#
   * @name getCharSpace
   */
  var getCharSpace = (API.__private__.getCharSpace = API.getCharSpace = function() {
    return parseFloat(activeCharSpace || 0);
  });

  /**
   * Set global value of CharSpace.
   *
   * @param {number} charSpace
   * @function
   * @instance
   * @returns {jsPDF} jsPDF-instance
   * @memberof jsPDF#
   * @name setCharSpace
   */
  API.__private__.setCharSpace = API.setCharSpace = function(charSpace) {
    if (isNaN(charSpace)) {
      throw new Error("Invalid argument passed to jsPDF.setCharSpace");
    }
    activeCharSpace = charSpace;
    return this;
  };

  var lineCapID = 0;
  /**
   * Is an Object providing a mapping from human-readable to
   * integer flag values designating the varieties of line cap
   * and join styles.
   *
   * @memberof jsPDF#
   * @name CapJoinStyles
   */
  API.CapJoinStyles = {
    0: 0,
    butt: 0,
    but: 0,
    miter: 0,
    1: 1,
    round: 1,
    rounded: 1,
    circle: 1,
    2: 2,
    projecting: 2,
    project: 2,
    square: 2,
    bevel: 2
  };

  /**
   * Sets the line cap styles.
   * See {jsPDF.CapJoinStyles} for variants.
   *
   * @param {String|Number} style A string or number identifying the type of line cap.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineCap
   */
  API.__private__.setLineCap = API.setLineCap = function(style) {
    var id = API.CapJoinStyles[style];
    if (id === undefined) {
      throw new Error(
        "Line cap style of '" +
          style +
          "' is not recognized. See or extend .CapJoinStyles property for valid styles"
      );
    }
    lineCapID = id;
    out(id + " J");

    return this;
  };

  var lineJoinID = 0;
  /**
   * Sets the line join styles.
   * See {jsPDF.CapJoinStyles} for variants.
   *
   * @param {String|Number} style A string or number identifying the type of line join.
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineJoin
   */
  API.__private__.setLineJoin = API.setLineJoin = function(style) {
    var id = API.CapJoinStyles[style];
    if (id === undefined) {
      throw new Error(
        "Line join style of '" +
          style +
          "' is not recognized. See or extend .CapJoinStyles property for valid styles"
      );
    }
    lineJoinID = id;
    out(id + " j");

    return this;
  };
  /**
   * Sets the miterLimit property, which effects the maximum miter length.
   *
   * @param {number} length The length of the miter
   * @function
   * @instance
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setLineMiterLimit
   */
  API.__private__.setLineMiterLimit = API.__private__.setMiterLimit = API.setLineMiterLimit = API.setMiterLimit = function(
    length
  ) {
    length = length || 0;
    if (isNaN(length)) {
      throw new Error("Invalid argument passed to jsPDF.setLineMiterLimit");
    }
    out(hpf(scale(length)) + " M");

    return this;
  };

  /**
   * An object representing a pdf graphics state.
   * @class GState
   */

  /**
   *
   * @param parameters A parameter object that contains all properties this graphics state wants to set.
   * Supported are: opacity, stroke-opacity
   * @constructor
   */
  API.GState = GState;

  /**
   * Sets a either previously added {@link GState} (via {@link addGState}) or a new {@link GState}.
   * @param {String|GState} gState If type is string, a previously added GState is used, if type is GState
   * it will be added before use.
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setGState
   */
  API.setGState = function(gState) {
    if (typeof gState === "string") {
      gState = gStates[gStatesMap[gState]];
    } else {
      gState = addGState(null, gState);
    }

    if (!gState.equals(activeGState)) {
      out("/" + gState.id + " gs");
      activeGState = gState;
    }
  };

  /**
   * Adds a new Graphics State. Duplicates are automatically eliminated.
   * @param {String} key Might also be null, if no later reference to this gState is needed
   * @param {Object} gState The gState object
   */
  var addGState = function(key, gState) {
    // only add it if it is not already present (the keys provided by the user must be unique!)
    if (key && gStatesMap[key]) return;
    var duplicate = false;
    for (var s in gStates) {
      if (gStates.hasOwnProperty(s)) {
        if (gStates[s].equals(gState)) {
          duplicate = true;
          break;
        }
      }
    }

    if (duplicate) {
      gState = gStates[s];
    } else {
      var gStateKey = "GS" + (Object.keys(gStates).length + 1).toString(10);
      gStates[gStateKey] = gState;
      gState.id = gStateKey;
    }

    // several user keys may point to the same GState object
    key && (gStatesMap[key] = gState.id);

    events.publish("addGState", gState);

    return gState;
  };

  /**
   * Adds a new {@link GState} for later use. See {@link setGState}.
   * @param {String} key
   * @param {GState} gState
   * @function
   * @instance
   * @returns {jsPDF}
   *
   * @memberof jsPDF#
   * @name addGState
   */
  API.addGState = function(key, gState) {
    addGState(key, gState);
    return this;
  };

  /**
   * Saves the current graphics state ("pushes it on the stack"). It can be restored by {@link restoreGraphicsState}
   * later. Here, the general pdf graphics state is meant, also including the current transformation matrix,
   * fill and stroke colors etc.
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name saveGraphicsState
   */
  API.saveGraphicsState = function() {
    out("q");
    // as we cannot set font key and size independently we must keep track of both
    fontStateStack.push({
      key: activeFontKey,
      size: activeFontSize,
      color: textColor
    });
    return this;
  };

  /**
   * Restores a previously saved graphics state saved by {@link saveGraphicsState} ("pops the stack").
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name restoreGraphicsState
   */
  API.restoreGraphicsState = function() {
    out("Q");

    // restore previous font state
    var fontState = fontStateStack.pop();
    activeFontKey = fontState.key;
    activeFontSize = fontState.size;
    textColor = fontState.color;

    activeGState = null;

    return this;
  };

  /**
   * Appends this matrix to the left of all previously applied matrices.
   *
   * @param {Matrix} matrix
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name setCurrentTransformationMatrix
   */
  API.setCurrentTransformationMatrix = function(matrix) {
    out(matrix.toString() + " cm");
    return this;
  };

  /**
   * Inserts a debug comment into the generated pdf.
   * @function
   * @instance
   * @param {String} text
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name comment
   */
  API.comment = function(text) {
    out("#" + text);
    return this;
  };

  /**
   * Point
   */
  var Point = function(x, y) {
    var _x = x || 0;
    Object.defineProperty(this, "x", {
      enumerable: true,
      get: function() {
        return _x;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _x = parseFloat(value);
        }
      }
    });

    var _y = y || 0;
    Object.defineProperty(this, "y", {
      enumerable: true,
      get: function() {
        return _y;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _y = parseFloat(value);
        }
      }
    });

    var _type = "pt";
    Object.defineProperty(this, "type", {
      enumerable: true,
      get: function() {
        return _type;
      },
      set: function(value) {
        _type = value.toString();
      }
    });
    return this;
  };

  /**
   * Rectangle
   */
  var Rectangle = function(x, y, w, h) {
    Point.call(this, x, y);
    this.type = "rect";

    var _w = w || 0;
    Object.defineProperty(this, "w", {
      enumerable: true,
      get: function() {
        return _w;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _w = parseFloat(value);
        }
      }
    });

    var _h = h || 0;
    Object.defineProperty(this, "h", {
      enumerable: true,
      get: function() {
        return _h;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _h = parseFloat(value);
        }
      }
    });

    return this;
  };

  /**
   * FormObject/RenderTarget
   */

  var RenderTarget = function() {
    this.page = page;
    this.currentPage = currentPage;
    this.pages = pages.slice(0);
    this.pagesContext = pagesContext.slice(0);
    this.x = pageX;
    this.y = pageY;
    this.matrix = pageMatrix;
    this.width = getUnscaledPageWidth(currentPage);
    this.height = getUnscaledPageHeight(currentPage);
    this.outputDestination = outputDestination;

    this.id = ""; // set by endFormObject()
    this.objectNumber = -1; // will be set by putXObject()
  };

  RenderTarget.prototype.restore = function() {
    page = this.page;
    currentPage = this.currentPage;
    pagesContext = this.pagesContext;
    pages = this.pages;
    pageX = this.x;
    pageY = this.y;
    pageMatrix = this.matrix;
    setPageWidthWithoutScaling(currentPage, this.width);
    setPageHeightWithoutScaling(currentPage, this.height);
    outputDestination = this.outputDestination;
  };

  var beginNewRenderTarget = function(x, y, width, height, matrix) {
    // save current state
    renderTargetStack.push(new RenderTarget());

    // clear pages
    page = currentPage = 0;
    pages = [];
    pageX = x;
    pageY = y;

    pageMatrix = matrix;

    beginPage([width, height]);
  };

  var endFormObject = function(key) {
    // only add it if it is not already present (the keys provided by the user must be unique!)
    if (renderTargetMap[key]) {
      renderTargetStack.pop().restore();
      return;
    }

    // save the created xObject
    var newXObject = new RenderTarget();

    var xObjectId = "Xo" + (Object.keys(renderTargets).length + 1).toString(10);
    newXObject.id = xObjectId;

    renderTargetMap[key] = xObjectId;
    renderTargets[xObjectId] = newXObject;

    events.publish("addFormObject", newXObject);

    // restore state from stack
    renderTargetStack.pop().restore();
  };

  /**
   * Starts a new pdf form object, which means that all consequent draw calls target a new independent object
   * until {@link endFormObject} is called. The created object can be referenced and drawn later using
   * {@link doFormObject}. Nested form objects are possible.
   * x, y, width, height set the bounding box that is used to clip the content.
   *
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {Matrix} matrix The matrix that will be applied to convert the form objects coordinate system to
   * the parent's.
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name beginFormObject
   */
  API.beginFormObject = function(x, y, width, height, matrix) {
    // The user can set the output target to a new form object. Nested form objects are possible.
    // Currently, they use the resource dictionary of the surrounding stream. This should be changed, as
    // the PDF-Spec states:
    // "In PDF 1.2 and later versions, form XObjects may be independent of the content streams in which
    // they appear, and this is strongly recommended although not requiredIn PDF 1.2 and later versions,
    // form XObjects may be independent of the content streams in which they appear, and this is strongly
    // recommended although not required"
    beginNewRenderTarget(x, y, width, height, matrix);
    return this;
  };

  /**
   * Completes and saves the form object.
   * @param {String} key The key by which this form object can be referenced.
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name endFormObject
   */
  API.endFormObject = function(key) {
    endFormObject(key);
    return this;
  };

  /**
   * Draws the specified form object by referencing to the respective pdf XObject created with
   * {@link API.beginFormObject} and {@link endFormObject}.
   * The location is determined by matrix.
   *
   * @param {String} key The key to the form object.
   * @param {Matrix} matrix The matrix applied before drawing the form object.
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name doFormObject
   */
  API.doFormObject = function(key, matrix) {
    var xObject = renderTargets[renderTargetMap[key]];
    out("q");
    out(matrix.toString() + " cm");
    out("/" + xObject.id + " Do");
    out("Q");
    return this;
  };

  /**
   * Returns the form object specified by key.
   * @param key {String}
   * @returns {{x: number, y: number, width: number, height: number, matrix: Matrix}}
   * @function
   * @returns {jsPDF}
   * @memberof jsPDF#
   * @name getFormObject
   */
  API.getFormObject = function(key) {
    var xObject = renderTargets[renderTargetMap[key]];
    return {
      x: xObject.x,
      y: xObject.y,
      width: xObject.width,
      height: xObject.height,
      matrix: xObject.matrix
    };
  };

  /**
   * Saves as PDF document. An alias of jsPDF.output('save', 'filename.pdf').
   * Uses FileSaver.js-method saveAs.
   *
   * @memberof jsPDF#
   * @name save
   * @function
   * @instance
   * @param  {string} filename The filename including extension.
   * @param  {Object} options An Object with additional options, possible options: 'returnPromise'.
   * @returns {jsPDF|Promise} jsPDF-instance     */
  API.save = function(filename, options) {

    options = options || {};
    options.returnPromise = options.returnPromise || false;

    return Promise.reject(Error('save function removed'));
  };

  // applying plugins (more methods) ON TOP of built-in API.
  // this is intentional as we allow plugins to override
  // built-ins
  for (var plugin in jsPDF.API) {
    if (jsPDF.API.hasOwnProperty(plugin)) {
      if (plugin === "events" && jsPDF.API.events.length) {
        (function(events, newEvents) {
          // jsPDF.API.events is a JS Array of Arrays
          // where each Array is a pair of event name, handler
          // Events were added by plugins to the jsPDF instantiator.
          // These are always added to the new instance and some ran
          // during instantiation.
          var eventname, handler_and_args, i;

          for (i = newEvents.length - 1; i !== -1; i--) {
            // subscribe takes 3 args: 'topic', function, runonce_flag
            // if undefined, runonce is false.
            // users can attach callback directly,
            // or they can attach an array with [callback, runonce_flag]
            // that's what the "apply" magic is for below.
            eventname = newEvents[i][0];
            handler_and_args = newEvents[i][1];
            events.subscribe.apply(
              events,
              [eventname].concat(
                typeof handler_and_args === "function"
                  ? [handler_and_args]
                  : handler_and_args
              )
            );
          }
        })(events, jsPDF.API.events);
      } else {
        API[plugin] = jsPDF.API[plugin];
      }
    }
  }

  function getUnscaledPageWidth(pageNumber) {
    return (
      pagesContext[pageNumber].mediaBox.topRightX -
      pagesContext[pageNumber].mediaBox.bottomLeftX
    );
  }

  function setPageWidthWithoutScaling(pageNumber, value) {
    pagesContext[pageNumber].mediaBox.topRightX =
      value + pagesContext[pageNumber].mediaBox.bottomLeftX;
  }

  function getUnscaledPageHeight(pageNumber) {
    return (
      pagesContext[pageNumber].mediaBox.topRightY -
      pagesContext[pageNumber].mediaBox.bottomLeftY
    );
  }

  function setPageHeightWithoutScaling(pageNumber, value) {
    pagesContext[pageNumber].mediaBox.topRightY =
      value + pagesContext[pageNumber].mediaBox.bottomLeftY;
  }

  var getPageWidth = (API.getPageWidth = function(pageNumber) {
    pageNumber = pageNumber || currentPage;
    return getUnscaledPageWidth(pageNumber) / scaleFactor;
  });

  var setPageWidth = (API.setPageWidth = function(pageNumber, value) {
    setPageWidthWithoutScaling(pageNumber, value * scaleFactor);
  });

  var getPageHeight = (API.getPageHeight = function(pageNumber) {
    pageNumber = pageNumber || currentPage;
    return getUnscaledPageHeight(pageNumber) / scaleFactor;
  });

  var setPageHeight = (API.setPageHeight = function(pageNumber, value) {
    setPageHeightWithoutScaling(pageNumber, value * scaleFactor);
  });

  /**
   * Object exposing internal API to plugins
   * @public
   * @ignore
   */
  API.internal = {
    pdfEscape: pdfEscape,
    getStyle: getStyle,
    getFont: getFontEntry,
    getFontSize: getFontSize,
    getCharSpace: getCharSpace,
    getTextColor: getTextColor,
    getLineHeight: getLineHeight,
    getLineHeightFactor: getLineHeightFactor,
    getLineWidth: getLineWidth,
    write: write,
    getHorizontalCoordinate: getHorizontalCoordinate,
    getVerticalCoordinate: getVerticalCoordinate,
    getCoordinateString: getHorizontalCoordinateString,
    getVerticalCoordinateString: getVerticalCoordinateString,
    collections: {},
    newObject: newObject,
    newAdditionalObject: newAdditionalObject,
    newObjectDeferred: newObjectDeferred,
    newObjectDeferredBegin: newObjectDeferredBegin,
    getFilters: getFilters,
    putStream: putStream,
    events: events,
    scaleFactor: scaleFactor,
    pageSize: {
      getWidth: function() {
        return getPageWidth(currentPage);
      },
      setWidth: function(value) {
        setPageWidth(currentPage, value);
      },
      getHeight: function() {
        return getPageHeight(currentPage);
      },
      setHeight: function(value) {
        setPageHeight(currentPage, value);
      }
    },
    encryptionOptions: encryptionOptions,
    encryption: encryption,
    getEncryptor: getEncryptor,
    output: output,
    getNumberOfPages: getNumberOfPages,
    pages: pages,
    out: out,
    f2: f2,
    f3: f3,
    getPageInfo: getPageInfo,
    getPageInfoByObjId: getPageInfoByObjId,
    getCurrentPageInfo: getCurrentPageInfo,
    getPDFVersion: getPdfVersion,
    Point: Point,
    Rectangle: Rectangle,
    Matrix: Matrix,
    hasHotfix: hasHotfix //Expose the hasHotfix check so plugins can also check them.
  };

  Object.defineProperty(API.internal.pageSize, "width", {
    get: function() {
      return getPageWidth(currentPage);
    },
    set: function(value) {
      setPageWidth(currentPage, value);
    },
    enumerable: true,
    configurable: true
  });
  Object.defineProperty(API.internal.pageSize, "height", {
    get: function() {
      return getPageHeight(currentPage);
    },
    set: function(value) {
      setPageHeight(currentPage, value);
    },
    enumerable: true,
    configurable: true
  });

  //////////////////////////////////////////////////////
  // continuing initialization of jsPDF Document object
  //////////////////////////////////////////////////////
  // Add the first page automatically
  addFonts.call(API, standardFonts);
  activeFontKey = "F1";
  _addPage(format, orientation);

  events.publish("initialized");
  return API;
}

/**
 * jsPDF.API is a STATIC property of jsPDF class.
 * jsPDF.API is an object you can add methods and properties to.
 * The methods / properties you add will show up in new jsPDF objects.
 *
 * One property is prepopulated. It is the 'events' Object. Plugin authors can add topics,
 * callbacks to this object. These will be reassigned to all new instances of jsPDF.
 *
 * @static
 * @public
 * @memberof jsPDF#
 * @name API
 *
 * @example
 * jsPDF.API.mymethod = function(){
 *   // 'this' will be ref to internal API object. see jsPDF source
 *   // , so you can refer to built-in methods like so:
 *   //     this.line(....)
 *   //     this.text(....)
 * }
 * var pdfdoc = new jsPDF()
 * pdfdoc.mymethod() // <- !!!!!!
 */
jsPDF.API = {
  events: []
};
/**
 * The version of jsPDF.
 * @name version
 * @type {string}
 * @memberof jsPDF#
 */
jsPDF.version = "3.0.3";

/* global jsPDF */

var jsPDFAPI = jsPDF.API;
var scaleFactor = 1;

var pdfEscape = function(value) {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
};
var pdfUnescape = function(value) {
  return value
    .replace(/\\\\/g, "\\")
    .replace(/\\\(/g, "(")
    .replace(/\\\)/g, ")");
};

var f2 = function(number) {
  return number.toFixed(2); // Ie, %.2f
};

var f5 = function(number) {
  return number.toFixed(5); // Ie, %.2f
};

jsPDFAPI.__acroform__ = {};
var inherit = function(child, parent) {
  child.prototype = Object.create(parent.prototype);
  child.prototype.constructor = child;
};

var scale = function(x) {
  return x * scaleFactor;
};

var createFormXObject = function(formObject) {
  var xobj = new AcroFormXObject();
  var height = AcroFormAppearance.internal.getHeight(formObject) || 0;
  var width = AcroFormAppearance.internal.getWidth(formObject) || 0;
  xobj.BBox = [0, 0, Number(f2(width)), Number(f2(height))];
  return xobj;
};

/**
 * Bit-Operations
 */
var setBit = (jsPDFAPI.__acroform__.setBit = function(number, bitPosition) {
  number = number || 0;
  bitPosition = bitPosition || 0;

  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.setBit"
    );
  }
  var bitMask = 1 << bitPosition;

  number |= bitMask;

  return number;
});

var clearBit = (jsPDFAPI.__acroform__.clearBit = function(number, bitPosition) {
  number = number || 0;
  bitPosition = bitPosition || 0;

  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.clearBit"
    );
  }
  var bitMask = 1 << bitPosition;

  number &= ~bitMask;

  return number;
});

var getBit = (jsPDFAPI.__acroform__.getBit = function(number, bitPosition) {
  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.getBit"
    );
  }
  return (number & (1 << bitPosition)) === 0 ? 0 : 1;
});

/*
 * Ff starts counting the bit position at 1 and not like javascript at 0
 */
var getBitForPdf = (jsPDFAPI.__acroform__.getBitForPdf = function(
  number,
  bitPosition
) {
  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.getBitForPdf"
    );
  }
  return getBit(number, bitPosition - 1);
});

var setBitForPdf = (jsPDFAPI.__acroform__.setBitForPdf = function(
  number,
  bitPosition
) {
  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.setBitForPdf"
    );
  }
  return setBit(number, bitPosition - 1);
});

var clearBitForPdf = (jsPDFAPI.__acroform__.clearBitForPdf = function(
  number,
  bitPosition
) {
  if (isNaN(number) || isNaN(bitPosition)) {
    throw new Error(
      "Invalid arguments passed to jsPDF.API.__acroform__.clearBitForPdf"
    );
  }
  return clearBit(number, bitPosition - 1);
});

var calculateCoordinates = (jsPDFAPI.__acroform__.calculateCoordinates = function(
  args,
  scope
) {
  var getHorizontalCoordinate = scope.internal.getHorizontalCoordinate;
  var getVerticalCoordinate = scope.internal.getVerticalCoordinate;
  var x = args[0];
  var y = args[1];
  var w = args[2];
  var h = args[3];

  var coordinates = {};

  coordinates.lowerLeft_X = getHorizontalCoordinate(x) || 0;
  coordinates.lowerLeft_Y = getVerticalCoordinate(y + h) || 0;
  coordinates.upperRight_X = getHorizontalCoordinate(x + w) || 0;
  coordinates.upperRight_Y = getVerticalCoordinate(y) || 0;

  return [
    Number(f2(coordinates.lowerLeft_X)),
    Number(f2(coordinates.lowerLeft_Y)),
    Number(f2(coordinates.upperRight_X)),
    Number(f2(coordinates.upperRight_Y))
  ];
});

var calculateAppearanceStream = function(formObject) {
  if (formObject.appearanceStreamContent) {
    return formObject.appearanceStreamContent;
  }

  if (!formObject.V && !formObject.DV) {
    return;
  }

  // else calculate it

  var stream = [];
  var text = formObject._V || formObject.DV;
  var calcRes = calculateX(formObject, text);
  var fontKey = formObject.scope.internal.getFont(
    formObject.fontName,
    formObject.fontStyle
  ).id;

  //PDF 32000-1:2008, page 444
  stream.push("/Tx BMC");
  stream.push("q");
  stream.push("BT"); // Begin Text
  stream.push(formObject.scope.__private__.encodeColorString(formObject.color));
  stream.push("/" + fontKey + " " + f2(calcRes.fontSize) + " Tf");
  stream.push("1 0 0 1 0 0 Tm"); // Transformation Matrix
  stream.push(calcRes.text);
  stream.push("ET"); // End Text
  stream.push("Q");
  stream.push("EMC");

  var appearanceStreamContent = createFormXObject(formObject);
  appearanceStreamContent.scope = formObject.scope;
  appearanceStreamContent.stream = stream.join("\n");
  return appearanceStreamContent;
};

var calculateX = function(formObject, text) {
  var maxFontSize =
    formObject.fontSize === 0 ? formObject.maxFontSize : formObject.fontSize;
  var returnValue = {
    text: "",
    fontSize: ""
  };
  // Remove Brackets
  text = text.substr(0, 1) == "(" ? text.substr(1) : text;
  text =
    text.substr(text.length - 1) == ")"
      ? text.substr(0, text.length - 1)
      : text;
  // split into array of words
  var textSplit = text.split(" ");
  if (formObject.multiline) {
    textSplit = textSplit.map(word => word.split("\n"));
  } else {
    textSplit = textSplit.map(word => [word]);
  }

  var fontSize = maxFontSize; // The Starting fontSize (The Maximum)
  var lineSpacing = 2;
  var borderPadding = 2;

  var height = AcroFormAppearance.internal.getHeight(formObject) || 0;
  height = height < 0 ? -height : height;
  var width = AcroFormAppearance.internal.getWidth(formObject) || 0;
  width = width < 0 ? -width : width;

  var isSmallerThanWidth = function(i, lastLine, fontSize) {
    if (i + 1 < textSplit.length) {
      var tmp = lastLine + " " + textSplit[i + 1][0];
      var TextWidth = calculateFontSpace(tmp, formObject, fontSize).width;
      var FieldWidth = width - 2 * borderPadding;
      return TextWidth <= FieldWidth;
    } else {
      return false;
    }
  };

  fontSize++;
  FontSize: while (fontSize > 0) {
    text = "";
    fontSize--;
    var textHeight = calculateFontSpace("3", formObject, fontSize).height;
    var startY = formObject.multiline
      ? height - fontSize
      : (height - textHeight) / 2;
    startY += lineSpacing;
    var startX;

    var lastY = startY;
    var firstWordInLine = 0,
      lastWordInLine = 0;
    var lastLength;
    var currWord = 0;

    if (fontSize <= 0) {
      // In case, the Text doesn't fit at all
      fontSize = 12;
      text = "(...) Tj\n";
      text +=
        "% Width of Text: " +
        calculateFontSpace(text, formObject, fontSize).width +
        ", FieldWidth:" +
        width +
        "\n";
      break;
    }

    var lastLine = "";
    var lineCount = 0;
    Line: for (var i = 0; i < textSplit.length; i++) {
      if (textSplit.hasOwnProperty(i)) {
        let isWithNewLine = false;
        if (textSplit[i].length !== 1 && currWord !== textSplit[i].length - 1) {
          if (
            (textHeight + lineSpacing) * (lineCount + 2) + lineSpacing >
            height
          ) {
            continue FontSize;
          }

          lastLine += textSplit[i][currWord];
          isWithNewLine = true;
          lastWordInLine = i;
          i--;
        } else {
          lastLine += textSplit[i][currWord] + " ";
          lastLine =
            lastLine.substr(lastLine.length - 1) == " "
              ? lastLine.substr(0, lastLine.length - 1)
              : lastLine;
          var key = parseInt(i);
          var nextLineIsSmaller = isSmallerThanWidth(key, lastLine, fontSize);
          var isLastWord = i >= textSplit.length - 1;

          if (nextLineIsSmaller && !isLastWord) {
            lastLine += " ";
            currWord = 0;
            continue; // Line
          } else if (!nextLineIsSmaller && !isLastWord) {
            if (!formObject.multiline) {
              continue FontSize;
            } else {
              if (
                (textHeight + lineSpacing) * (lineCount + 2) + lineSpacing >
                height
              ) {
                // If the Text is higher than the
                // FieldObject
                continue FontSize;
              }
              lastWordInLine = key;
              // go on
            }
          } else if (isLastWord) {
            lastWordInLine = key;
          } else {
            if (
              formObject.multiline &&
              (textHeight + lineSpacing) * (lineCount + 2) + lineSpacing >
                height
            ) {
              // If the Text is higher than the FieldObject
              continue FontSize;
            }
          }
        }
        // Remove last blank

        var line = "";

        for (var x = firstWordInLine; x <= lastWordInLine; x++) {
          var currLine = textSplit[x];
          if (formObject.multiline) {
            if (x === lastWordInLine) {
              line += currLine[currWord] + " ";
              currWord = (currWord + 1) % currLine.length;
              continue;
            }
            if (x === firstWordInLine) {
              line += currLine[currLine.length - 1] + " ";
              continue;
            }
          }
          line += currLine[0] + " ";
        }

        // Remove last blank
        line =
          line.substr(line.length - 1) == " "
            ? line.substr(0, line.length - 1)
            : line;
        // lastLength -= blankSpace.width;
        lastLength = calculateFontSpace(line, formObject, fontSize).width;

        // Calculate startX
        switch (formObject.textAlign) {
          case "right":
            startX = width - lastLength - borderPadding;
            break;
          case "center":
            startX = (width - lastLength) / 2;
            break;
          case "left":
          default:
            startX = borderPadding;
            break;
        }
        text += f2(startX) + " " + f2(lastY) + " Td\n";
        text += "(" + pdfEscape(line) + ") Tj\n";
        // reset X in PDF
        text += -f2(startX) + " 0 Td\n";

        // After a Line, adjust y position
        lastY = -(fontSize + lineSpacing);

        // Reset for next iteration step
        lastLength = 0;
        firstWordInLine = isWithNewLine ? lastWordInLine : lastWordInLine + 1;
        lineCount++;

        lastLine = "";
        continue Line;
      }
    }
    break;
  }

  returnValue.text = text;
  returnValue.fontSize = fontSize;

  return returnValue;
};

/**
 * Small workaround for calculating the TextMetric approximately.
 *
 * @param text
 * @param fontsize
 * @returns {TextMetrics} (Has Height and Width)
 */
var calculateFontSpace = function(text, formObject, fontSize) {
  var font = formObject.scope.internal.getFont(
    formObject.fontName,
    formObject.fontStyle
  );
  var width =
    formObject.scope.getStringUnitWidth(text, {
      font: font,
      fontSize: parseFloat(fontSize),
      charSpace: 0
    }) * parseFloat(fontSize);
  var height =
    formObject.scope.getStringUnitWidth("3", {
      font: font,
      fontSize: parseFloat(fontSize),
      charSpace: 0
    }) *
    parseFloat(fontSize) *
    1.5;
  return { height: height, width: width };
};

var acroformPluginTemplate = {
  fields: [],
  xForms: [],
  /**
   * acroFormDictionaryRoot contains information about the AcroForm
   * Dictionary 0: The Event-Token, the AcroFormDictionaryCallback has
   * 1: The Object ID of the Root
   */
  acroFormDictionaryRoot: null,
  /**
   * After the PDF gets evaluated, the reference to the root has to be
   * reset, this indicates, whether the root has already been printed
   * out
   */
  printedOut: false,
  internal: null,
  isInitialized: false
};

var annotReferenceCallback = function(scope) {
  //set objId to undefined and force it to get a new objId on buildDocument
  scope.internal.acroformPlugin.acroFormDictionaryRoot.objId = undefined;
  var fields = scope.internal.acroformPlugin.acroFormDictionaryRoot.Fields;
  for (var i in fields) {
    if (fields.hasOwnProperty(i)) {
      var formObject = fields[i];
      //set objId to undefined and force it to get a new objId on buildDocument
      formObject.objId = undefined;
      // add Annot Reference!
      if (formObject.hasAnnotation) {
        // If theres an Annotation Widget in the Form Object, put the
        // Reference in the /Annot array
        createAnnotationReference(formObject, scope);
      }
    }
  }
};

var putForm = function(formObject) {
  if (formObject.scope.internal.acroformPlugin.printedOut) {
    formObject.scope.internal.acroformPlugin.printedOut = false;
    formObject.scope.internal.acroformPlugin.acroFormDictionaryRoot = null;
  }
  formObject.scope.internal.acroformPlugin.acroFormDictionaryRoot.Fields.push(
    formObject
  );
};
/**
 * Create the Reference to the widgetAnnotation, so that it gets referenced
 * in the Annot[] int the+ (Requires the Annotation Plugin)
 */
var createAnnotationReference = function(object, scope) {
  var options = {
    type: "reference",
    object: object
  };
  var findEntry = function(entry) {
    return entry.type === options.type && entry.object === options.object;
  };
  if (
    scope.internal
      .getPageInfo(object.page)
      .pageContext.annotations.find(findEntry) === undefined
  ) {
    scope.internal
      .getPageInfo(object.page)
      .pageContext.annotations.push(options);
  }
};

// Callbacks

var putCatalogCallback = function(scope) {
  // Put reference to AcroForm to DocumentCatalog
  if (
    typeof scope.internal.acroformPlugin.acroFormDictionaryRoot !== "undefined"
  ) {
    // for safety, shouldn't normally be the case
    scope.internal.write(
      "/AcroForm " +
        scope.internal.acroformPlugin.acroFormDictionaryRoot.objId +
        " " +
        0 +
        " R"
    );
  } else {
    throw new Error("putCatalogCallback: Root missing.");
  }
};

/**
 * Adds /Acroform X 0 R to Document Catalog, and creates the AcroForm
 * Dictionary
 */
var AcroFormDictionaryCallback = function(scope) {
  // Remove event
  scope.internal.events.unsubscribe(
    scope.internal.acroformPlugin.acroFormDictionaryRoot._eventID
  );
  delete scope.internal.acroformPlugin.acroFormDictionaryRoot._eventID;
  scope.internal.acroformPlugin.printedOut = true;
};

/**
 * Creates the single Fields and writes them into the Document
 *
 * If fieldArray is set, use the fields that are inside it instead of the
 * fields from the AcroRoot (for the FormXObjects...)
 */
var createFieldCallback = function(fieldArray, scope) {
  var standardFields = !fieldArray;

  if (!fieldArray) {
    // in case there is no fieldArray specified, we want to print out
    // the Fields of the AcroForm
    // Print out Root
    scope.internal.newObjectDeferredBegin(
      scope.internal.acroformPlugin.acroFormDictionaryRoot.objId,
      true
    );
    scope.internal.acroformPlugin.acroFormDictionaryRoot.putStream();
  }

  fieldArray =
    fieldArray || scope.internal.acroformPlugin.acroFormDictionaryRoot.Kids;

  for (var i in fieldArray) {
    if (fieldArray.hasOwnProperty(i)) {
      var fieldObject = fieldArray[i];
      var keyValueList = [];
      var oldRect = fieldObject.Rect;

      if (fieldObject.Rect) {
        fieldObject.Rect = calculateCoordinates(fieldObject.Rect, scope);
      }

      // Start Writing the Object
      scope.internal.newObjectDeferredBegin(fieldObject.objId, true);

      fieldObject.DA = AcroFormAppearance.createDefaultAppearanceStream(
        fieldObject
      );

      if (
        typeof fieldObject === "object" &&
        typeof fieldObject.getKeyValueListForStream === "function"
      ) {
        keyValueList = fieldObject.getKeyValueListForStream();
      }

      fieldObject.Rect = oldRect;

      if (
        fieldObject.hasAppearanceStream &&
        !fieldObject.appearanceStreamContent
      ) {
        // Calculate Appearance
        var appearance = calculateAppearanceStream(fieldObject);
        keyValueList.push({ key: "AP", value: "<</N " + appearance + ">>" });

        scope.internal.acroformPlugin.xForms.push(appearance);
      }

      // Assume AppearanceStreamContent is a Array with N,R,D (at least
      // one of them!)
      if (fieldObject.appearanceStreamContent) {
        var appearanceStreamString = "";
        // Iterate over N,R and D
        for (var k in fieldObject.appearanceStreamContent) {
          if (fieldObject.appearanceStreamContent.hasOwnProperty(k)) {
            var value = fieldObject.appearanceStreamContent[k];
            appearanceStreamString += "/" + k + " ";
            appearanceStreamString += "<<";
            if (Object.keys(value).length >= 1 || Array.isArray(value)) {
              // appearanceStream is an Array or Object!
              for (var i in value) {
                if (value.hasOwnProperty(i)) {
                  var obj = value[i];
                  if (typeof obj === "function") {
                    // if Function is referenced, call it in order
                    // to get the FormXObject
                    obj = obj.call(scope, fieldObject);
                  }
                  appearanceStreamString += "/" + i + " " + obj + " ";

                  // In case the XForm is already used, e.g. OffState
                  // of CheckBoxes, don't add it
                  if (!(scope.internal.acroformPlugin.xForms.indexOf(obj) >= 0))
                    scope.internal.acroformPlugin.xForms.push(obj);
                }
              }
            } else {
              obj = value;
              if (typeof obj === "function") {
                // if Function is referenced, call it in order to
                // get the FormXObject
                obj = obj.call(scope, fieldObject);
              }
              appearanceStreamString += "/" + i + " " + obj;
              if (!(scope.internal.acroformPlugin.xForms.indexOf(obj) >= 0))
                scope.internal.acroformPlugin.xForms.push(obj);
            }
            appearanceStreamString += ">>";
          }
        }

        // appearance stream is a normal Object..
        keyValueList.push({
          key: "AP",
          value: "<<\n" + appearanceStreamString + ">>"
        });
      }

      scope.internal.putStream({
        additionalKeyValues: keyValueList,
        objectId: fieldObject.objId
      });

      scope.internal.out("endobj");
    }
  }
  if (standardFields) {
    createXFormObjectCallback(scope.internal.acroformPlugin.xForms, scope);
  }
};

var createXFormObjectCallback = function(fieldArray, scope) {
  for (var i in fieldArray) {
    if (fieldArray.hasOwnProperty(i)) {
      var key = i;
      var fieldObject = fieldArray[i];
      // Start Writing the Object
      scope.internal.newObjectDeferredBegin(fieldObject.objId, true);

      if (
        typeof fieldObject === "object" &&
        typeof fieldObject.putStream === "function"
      ) {
        fieldObject.putStream();
      }
      delete fieldArray[key];
    }
  }
};

var initializeAcroForm = function(scope, formObject) {
  formObject.scope = scope;
  if (
    scope.internal !== undefined &&
    (scope.internal.acroformPlugin === undefined ||
      scope.internal.acroformPlugin.isInitialized === false)
  ) {
    AcroFormField.FieldNum = 0;
    scope.internal.acroformPlugin = JSON.parse(
      JSON.stringify(acroformPluginTemplate)
    );
    if (scope.internal.acroformPlugin.acroFormDictionaryRoot) {
      throw new Error("Exception while creating AcroformDictionary");
    }
    scaleFactor = scope.internal.scaleFactor;
    // The Object Number of the AcroForm Dictionary
    scope.internal.acroformPlugin.acroFormDictionaryRoot = new AcroFormDictionary();
    scope.internal.acroformPlugin.acroFormDictionaryRoot.scope = scope;

    // add Callback for creating the AcroForm Dictionary
    scope.internal.acroformPlugin.acroFormDictionaryRoot._eventID = scope.internal.events.subscribe(
      "postPutResources",
      function() {
        AcroFormDictionaryCallback(scope);
      }
    );

    scope.internal.events.subscribe("buildDocument", function() {
      annotReferenceCallback(scope);
    }); // buildDocument

    // Register event, that is triggered when the DocumentCatalog is
    // written, in order to add /AcroForm

    scope.internal.events.subscribe("putCatalog", function() {
      putCatalogCallback(scope);
    });

    // Register event, that creates all Fields
    scope.internal.events.subscribe("postPutPages", function(fieldArray) {
      createFieldCallback(fieldArray, scope);
    });

    scope.internal.acroformPlugin.isInitialized = true;
  }
};

//PDF 32000-1:2008, page 26, 7.3.6
var arrayToPdfArray = (jsPDFAPI.__acroform__.arrayToPdfArray = function(
  array,
  objId,
  scope
) {
  var encryptor = function(data) {
    return data;
  };
  if (Array.isArray(array)) {
    var content = "[";
    for (var i = 0; i < array.length; i++) {
      if (i !== 0) {
        content += " ";
      }
      switch (typeof array[i]) {
        case "boolean":
        case "number":
        case "object":
          content += array[i].toString();
          break;
        case "string":
          if (array[i].substr(0, 1) !== "/") {
            if (typeof objId !== "undefined" && scope)
              encryptor = scope.internal.getEncryptor(objId);
            content += "(" + pdfEscape(encryptor(array[i].toString())) + ")";
          } else {
            content += array[i].toString();
          }
          break;
      }
    }
    content += "]";
    return content;
  }
  throw new Error(
    "Invalid argument passed to jsPDF.__acroform__.arrayToPdfArray"
  );
});
function getMatches(string, regex, index) {
  index || (index = 1); // default to the first capturing group
  var matches = [];
  var match;
  while ((match = regex.exec(string))) {
    matches.push(match[index]);
  }
  return matches;
}
var pdfArrayToStringArray = function(array) {
  var result = [];
  if (typeof array === "string") {
    result = getMatches(array, /\((.*?)\)/g);
  }
  return result;
};

var toPdfString = function(string, objId, scope) {
  var encryptor = function(data) {
    return data;
  };
  if (typeof objId !== "undefined" && scope)
    encryptor = scope.internal.getEncryptor(objId);
  string = string || "";
  string.toString();
  string = "(" + pdfEscape(encryptor(string)) + ")";
  return string;
};

// ##########################
// Classes
// ##########################

/**
 * @class AcroFormPDFObject
 * @classdesc A AcroFormPDFObject
 */
var AcroFormPDFObject = function() {
  this._objId = undefined;
  this._scope = undefined;

  /**
   * @name AcroFormPDFObject#objId
   * @type {any}
   */
  Object.defineProperty(this, "objId", {
    get: function() {
      if (typeof this._objId === "undefined") {
        if (typeof this.scope === "undefined") {
          return undefined;
        }
        this._objId = this.scope.internal.newObjectDeferred();
      }
      return this._objId;
    },
    set: function(value) {
      this._objId = value;
    }
  });
  Object.defineProperty(this, "scope", {
    value: this._scope,
    writable: true
  });
};

/**
 * @function AcroFormPDFObject.toString
 */
AcroFormPDFObject.prototype.toString = function() {
  return this.objId + " 0 R";
};

AcroFormPDFObject.prototype.putStream = function() {
  var keyValueList = this.getKeyValueListForStream();
  this.scope.internal.putStream({
    data: this.stream,
    additionalKeyValues: keyValueList,
    objectId: this.objId
  });
  this.scope.internal.out("endobj");
};

/**
 * Returns an key-value-List of all non-configurable Variables from the Object
 *
 * @name getKeyValueListForStream
 * @returns {string}
 */
AcroFormPDFObject.prototype.getKeyValueListForStream = function() {
  var keyValueList = [];
  var keys = Object.getOwnPropertyNames(this).filter(function(key) {
    return (
      key != "content" &&
      key != "appearanceStreamContent" &&
      key != "scope" &&
      key != "objId" &&
      key.substring(0, 1) != "_"
    );
  });

  for (var i in keys) {
    if (Object.getOwnPropertyDescriptor(this, keys[i]).configurable === false) {
      var key = keys[i];
      var value = this[key];

      if (value) {
        if (Array.isArray(value)) {
          keyValueList.push({
            key: key,
            value: arrayToPdfArray(value, this.objId, this.scope)
          });
        } else if (value instanceof AcroFormPDFObject) {
          // In case it is a reference to another PDFObject,
          // take the reference number
          value.scope = this.scope;
          keyValueList.push({ key: key, value: value.objId + " 0 R" });
        } else if (typeof value !== "function") {
          keyValueList.push({ key: key, value: value });
        }
      }
    }
  }
  return keyValueList;
};

var AcroFormXObject = function() {
  AcroFormPDFObject.call(this);

  Object.defineProperty(this, "Type", {
    value: "/XObject",
    configurable: false,
    writable: true
  });

  Object.defineProperty(this, "Subtype", {
    value: "/Form",
    configurable: false,
    writable: true
  });

  Object.defineProperty(this, "FormType", {
    value: 1,
    configurable: false,
    writable: true
  });

  var _BBox = [];
  Object.defineProperty(this, "BBox", {
    configurable: false,
    get: function() {
      return _BBox;
    },
    set: function(value) {
      _BBox = value;
    }
  });

  Object.defineProperty(this, "Resources", {
    value: "2 0 R",
    configurable: false,
    writable: true
  });

  var _stream;
  Object.defineProperty(this, "stream", {
    enumerable: false,
    configurable: true,
    set: function(value) {
      _stream = value.trim();
    },
    get: function() {
      if (_stream) {
        return _stream;
      } else {
        return null;
      }
    }
  });
};

inherit(AcroFormXObject, AcroFormPDFObject);

var AcroFormDictionary = function() {
  AcroFormPDFObject.call(this);

  var _Kids = [];

  Object.defineProperty(this, "Kids", {
    enumerable: false,
    configurable: true,
    get: function() {
      if (_Kids.length > 0) {
        return _Kids;
      } else {
        return undefined;
      }
    }
  });
  Object.defineProperty(this, "Fields", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _Kids;
    }
  });

  // Default Appearance
  var _DA;
  Object.defineProperty(this, "DA", {
    enumerable: false,
    configurable: false,
    get: function() {
      if (!_DA) {
        return undefined;
      }
      var encryptor = function(data) {
        return data;
      };
      if (this.scope) encryptor = this.scope.internal.getEncryptor(this.objId);
      return "(" + pdfEscape(encryptor(_DA)) + ")";
    },
    set: function(value) {
      _DA = value;
    }
  });
};

inherit(AcroFormDictionary, AcroFormPDFObject);

/**
 * The Field Object contains the Variables, that every Field needs
 *
 * @class AcroFormField
 * @classdesc An AcroForm FieldObject
 */
var AcroFormField = function() {
  AcroFormPDFObject.call(this);

  //Annotation-Flag See Table 165
  var _F = 4;
  Object.defineProperty(this, "F", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _F;
    },
    set: function(value) {
      if (!isNaN(value)) {
        _F = value;
      } else {
        throw new Error(
          'Invalid value "' + value + '" for attribute F supplied.'
        );
      }
    }
  });

  /**
   * (PDF 1.2) If set, print the annotation when the page is printed. If clear, never print the annotation, regardless of wether is is displayed on the screen.
   * NOTE 2 This can be useful for annotations representing interactive pushbuttons, which would serve no meaningful purpose on the printed page.
   *
   * @name AcroFormField#showWhenPrinted
   * @default true
   * @type {boolean}
   */
  Object.defineProperty(this, "showWhenPrinted", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(_F, 3));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.F = setBitForPdf(_F, 3);
      } else {
        this.F = clearBitForPdf(_F, 3);
      }
    }
  });

  var _Ff = 0;
  Object.defineProperty(this, "Ff", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _Ff;
    },
    set: function(value) {
      if (!isNaN(value)) {
        _Ff = value;
      } else {
        throw new Error(
          'Invalid value "' + value + '" for attribute Ff supplied.'
        );
      }
    }
  });

  var _Rect = [];
  Object.defineProperty(this, "Rect", {
    enumerable: false,
    configurable: false,
    get: function() {
      if (_Rect.length === 0) {
        return undefined;
      }
      return _Rect;
    },
    set: function(value) {
      if (typeof value !== "undefined") {
        _Rect = value;
      } else {
        _Rect = [];
      }
    }
  });

  /**
   * The x-position of the field.
   *
   * @name AcroFormField#x
   * @default null
   * @type {number}
   */
  Object.defineProperty(this, "x", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (!_Rect || isNaN(_Rect[0])) {
        return 0;
      }
      return _Rect[0];
    },
    set: function(value) {
      _Rect[0] = value;
    }
  });

  /**
   * The y-position of the field.
   *
   * @name AcroFormField#y
   * @default null
   * @type {number}
   */
  Object.defineProperty(this, "y", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (!_Rect || isNaN(_Rect[1])) {
        return 0;
      }
      return _Rect[1];
    },
    set: function(value) {
      _Rect[1] = value;
    }
  });

  /**
   * The width of the field.
   *
   * @name AcroFormField#width
   * @default null
   * @type {number}
   */
  Object.defineProperty(this, "width", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (!_Rect || isNaN(_Rect[2])) {
        return 0;
      }
      return _Rect[2];
    },
    set: function(value) {
      _Rect[2] = value;
    }
  });

  /**
   * The height of the field.
   *
   * @name AcroFormField#height
   * @default null
   * @type {number}
   */
  Object.defineProperty(this, "height", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (!_Rect || isNaN(_Rect[3])) {
        return 0;
      }
      return _Rect[3];
    },
    set: function(value) {
      _Rect[3] = value;
    }
  });

  var _FT = "";
  Object.defineProperty(this, "FT", {
    enumerable: true,
    configurable: false,
    get: function() {
      return _FT;
    },
    set: function(value) {
      switch (value) {
        case "/Btn":
        case "/Tx":
        case "/Ch":
        case "/Sig":
          _FT = value;
          break;
        default:
          throw new Error(
            'Invalid value "' + value + '" for attribute FT supplied.'
          );
      }
    }
  });

  var _T = null;

  Object.defineProperty(this, "T", {
    enumerable: true,
    configurable: false,
    get: function() {
      if (!_T || _T.length < 1) {
        // In case of a Child from a Radio´Group, you don't need a FieldName
        if (this instanceof AcroFormChildClass) {
          return undefined;
        }
        _T = "FieldObject" + AcroFormField.FieldNum++;
      }
      var encryptor = function(data) {
        return data;
      };
      if (this.scope) encryptor = this.scope.internal.getEncryptor(this.objId);
      return "(" + pdfEscape(encryptor(_T)) + ")";
    },
    set: function(value) {
      _T = value.toString();
    }
  });

  /**
   * (Optional) The partial field name (see 12.7.3.2, “Field Names”).
   *
   * @name AcroFormField#fieldName
   * @default null
   * @type {string}
   */
  Object.defineProperty(this, "fieldName", {
    configurable: true,
    enumerable: true,
    get: function() {
      return _T;
    },
    set: function(value) {
      _T = value;
    }
  });

  var _fontName = "helvetica";
  /**
   * The fontName of the font to be used.
   *
   * @name AcroFormField#fontName
   * @default 'helvetica'
   * @type {string}
   */
  Object.defineProperty(this, "fontName", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _fontName;
    },
    set: function(value) {
      _fontName = value;
    }
  });

  var _fontStyle = "normal";
  /**
   * The fontStyle of the font to be used.
   *
   * @name AcroFormField#fontStyle
   * @default 'normal'
   * @type {string}
   */
  Object.defineProperty(this, "fontStyle", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _fontStyle;
    },
    set: function(value) {
      _fontStyle = value;
    }
  });

  var _fontSize = 0;
  /**
   * The fontSize of the font to be used.
   *
   * @name AcroFormField#fontSize
   * @default 0 (for auto)
   * @type {number}
   */
  Object.defineProperty(this, "fontSize", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _fontSize;
    },
    set: function(value) {
      _fontSize = value;
    }
  });

  var _maxFontSize = undefined;
  /**
   * The maximum fontSize of the font to be used.
   *
   * @name AcroFormField#maxFontSize
   * @default 0 (for auto)
   * @type {number}
   */
  Object.defineProperty(this, "maxFontSize", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (_maxFontSize === undefined) {
        // use the old default value here - the value is some kind of random as it depends on the scaleFactor (user unit)
        // ("50" is transformed to the "user space" but then used in "pdf space")
        return 50 / scaleFactor;
      } else {
        return _maxFontSize;
      }
    },
    set: function(value) {
      _maxFontSize = value;
    }
  });

  var _color = "black";
  /**
   * The color of the text
   *
   * @name AcroFormField#color
   * @default 'black'
   * @type {string|rgba}
   */
  Object.defineProperty(this, "color", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _color;
    },
    set: function(value) {
      _color = value;
    }
  });

  var _DA = "/F1 0 Tf 0 g";
  // Defines the default appearance (Needed for variable Text)
  Object.defineProperty(this, "DA", {
    enumerable: true,
    configurable: false,
    get: function() {
      if (
        !_DA ||
        this instanceof AcroFormChildClass ||
        this instanceof AcroFormTextField
      ) {
        return undefined;
      }
      return toPdfString(_DA, this.objId, this.scope);
    },
    set: function(value) {
      value = value.toString();
      _DA = value;
    }
  });

  var _DV = null;
  Object.defineProperty(this, "DV", {
    enumerable: false,
    configurable: false,
    get: function() {
      if (!_DV) {
        return undefined;
      }
      if (this instanceof AcroFormButton === false) {
        return toPdfString(_DV, this.objId, this.scope);
      }
      return _DV;
    },
    set: function(value) {
      value = value.toString();
      if (this instanceof AcroFormButton === false) {
        if (value.substr(0, 1) === "(") {
          _DV = pdfUnescape(value.substr(1, value.length - 2));
        } else {
          _DV = pdfUnescape(value);
        }
      } else {
        _DV = value;
      }
    }
  });

  /**
   * (Optional; inheritable) The default value to which the field reverts when a reset-form action is executed (see 12.7.5.3, “Reset-Form Action”). The format of this value is the same as that of value.
   *
   * @name AcroFormField#defaultValue
   * @default null
   * @type {any}
   */
  Object.defineProperty(this, "defaultValue", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (this instanceof AcroFormButton === true) {
        return pdfUnescape(_DV.substr(1, _DV.length - 1));
      } else {
        return _DV;
      }
    },
    set: function(value) {
      value = value.toString();
      if (this instanceof AcroFormButton === true) {
        _DV = "/" + value;
      } else {
        _DV = value;
      }
    }
  });

  var _V = null;
  Object.defineProperty(this, "_V", {
    enumerable: false,
    configurable: false,
    get: function() {
      if (!_V) {
        return undefined;
      }
      return _V;
    },
    set: function(value) {
      this.V = value;
    }
  });
  Object.defineProperty(this, "V", {
    enumerable: false,
    configurable: false,
    get: function() {
      if (!_V) {
        return undefined;
      }
      if (this instanceof AcroFormButton === false) {
        return toPdfString(_V, this.objId, this.scope);
      }
      return _V;
    },
    set: function(value) {
      value = value.toString();
      if (this instanceof AcroFormButton === false) {
        if (value.substr(0, 1) === "(") {
          _V = pdfUnescape(value.substr(1, value.length - 2));
        } else {
          _V = pdfUnescape(value);
        }
      } else {
        _V = value;
      }
    }
  });

  /**
   * (Optional; inheritable) The field’s value, whose format varies depending on the field type. See the descriptions of individual field types for further information.
   *
   * @name AcroFormField#value
   * @default null
   * @type {any}
   */
  Object.defineProperty(this, "value", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (this instanceof AcroFormButton === true) {
        return pdfUnescape(_V.substr(1, _V.length - 1));
      } else {
        return _V;
      }
    },
    set: function(value) {
      value = value.toString();
      if (this instanceof AcroFormButton === true) {
        _V = "/" + value;
      } else {
        _V = value;
      }
    }
  });

  /**
   * Check if field has annotations
   *
   * @name AcroFormField#hasAnnotation
   * @readonly
   * @type {boolean}
   */
  Object.defineProperty(this, "hasAnnotation", {
    enumerable: true,
    configurable: true,
    get: function() {
      return this.Rect;
    }
  });

  Object.defineProperty(this, "Type", {
    enumerable: true,
    configurable: false,
    get: function() {
      return this.hasAnnotation ? "/Annot" : null;
    }
  });

  Object.defineProperty(this, "Subtype", {
    enumerable: true,
    configurable: false,
    get: function() {
      return this.hasAnnotation ? "/Widget" : null;
    }
  });

  var _hasAppearanceStream = false;
  /**
   * true if field has an appearanceStream
   *
   * @name AcroFormField#hasAppearanceStream
   * @readonly
   * @type {boolean}
   */
  Object.defineProperty(this, "hasAppearanceStream", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _hasAppearanceStream;
    },
    set: function(value) {
      value = Boolean(value);
      _hasAppearanceStream = value;
    }
  });

  /**
   * The page on which the AcroFormField is placed
   *
   * @name AcroFormField#page
   * @type {number}
   */
  var _page;
  Object.defineProperty(this, "page", {
    enumerable: true,
    configurable: true,
    get: function() {
      if (!_page) {
        return undefined;
      }
      return _page;
    },
    set: function(value) {
      _page = value;
    }
  });

  /**
   * If set, the user may not change the value of the field. Any associated widget annotations will not interact with the user; that is, they will not respond to mouse clicks or change their appearance in response to mouse motions. This flag is useful for fields whose values are computed or imported from a database.
   *
   * @name AcroFormField#readOnly
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "readOnly", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 1));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 1);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 1);
      }
    }
  });

  /**
   * If set, the field shall have a value at the time it is exported by a submitform action (see 12.7.5.2, “Submit-Form Action”).
   *
   * @name AcroFormField#required
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "required", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 2));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 2);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 2);
      }
    }
  });

  /**
   * If set, the field shall not be exported by a submit-form action (see 12.7.5.2, “Submit-Form Action”)
   *
   * @name AcroFormField#noExport
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "noExport", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 3));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 3);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 3);
      }
    }
  });

  var _Q = null;
  Object.defineProperty(this, "Q", {
    enumerable: true,
    configurable: false,
    get: function() {
      if (_Q === null) {
        return undefined;
      }
      return _Q;
    },
    set: function(value) {
      if ([0, 1, 2].indexOf(value) !== -1) {
        _Q = value;
      } else {
        throw new Error(
          'Invalid value "' + value + '" for attribute Q supplied.'
        );
      }
    }
  });

  /**
   * (Optional; inheritable) A code specifying the form of quadding (justification) that shall be used in displaying the text:
   * 'left', 'center', 'right'
   *
   * @name AcroFormField#textAlign
   * @default 'left'
   * @type {string}
   */
  Object.defineProperty(this, "textAlign", {
    get: function() {
      var result;
      switch (_Q) {
        case 0:
        default:
          result = "left";
          break;
        case 1:
          result = "center";
          break;
        case 2:
          result = "right";
          break;
      }
      return result;
    },
    configurable: true,
    enumerable: true,
    set: function(value) {
      switch (value) {
        case "right":
        case 2:
          _Q = 2;
          break;
        case "center":
        case 1:
          _Q = 1;
          break;
        case "left":
        case 0:
        default:
          _Q = 0;
      }
    }
  });
};

inherit(AcroFormField, AcroFormPDFObject);

/**
 * @class AcroFormChoiceField
 * @extends AcroFormField
 */
var AcroFormChoiceField = function() {
  AcroFormField.call(this);
  // Field Type = Choice Field
  this.FT = "/Ch";
  // options
  this.V = "()";

  this.fontName = "zapfdingbats";
  // Top Index
  var _TI = 0;

  Object.defineProperty(this, "TI", {
    enumerable: true,
    configurable: false,
    get: function() {
      return _TI;
    },
    set: function(value) {
      _TI = value;
    }
  });

  /**
   * (Optional) For scrollable list boxes, the top index (the index in the Opt array of the first option visible in the list). Default value: 0.
   *
   * @name AcroFormChoiceField#topIndex
   * @default 0
   * @type {number}
   */
  Object.defineProperty(this, "topIndex", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _TI;
    },
    set: function(value) {
      _TI = value;
    }
  });

  var _Opt = [];
  Object.defineProperty(this, "Opt", {
    enumerable: true,
    configurable: false,
    get: function() {
      return arrayToPdfArray(_Opt, this.objId, this.scope);
    },
    set: function(value) {
      _Opt = pdfArrayToStringArray(value);
    }
  });

  /**
   * @memberof AcroFormChoiceField
   * @name getOptions
   * @function
   * @instance
   * @returns {array} array of Options
   */
  this.getOptions = function() {
    return _Opt;
  };

  /**
   * @memberof AcroFormChoiceField
   * @name setOptions
   * @function
   * @instance
   * @param {array} value
   */
  this.setOptions = function(value) {
    _Opt = value;
    if (this.sort) {
      _Opt.sort();
    }
  };

  /**
   * @memberof AcroFormChoiceField
   * @name addOption
   * @function
   * @instance
   * @param {string} value
   */
  this.addOption = function(value) {
    value = value || "";
    value = value.toString();
    _Opt.push(value);
    if (this.sort) {
      _Opt.sort();
    }
  };

  /**
   * @memberof AcroFormChoiceField
   * @name removeOption
   * @function
   * @instance
   * @param {string} value
   * @param {boolean} allEntries (default: false)
   */
  this.removeOption = function(value, allEntries) {
    allEntries = allEntries || false;
    value = value || "";
    value = value.toString();

    while (_Opt.indexOf(value) !== -1) {
      _Opt.splice(_Opt.indexOf(value), 1);
      if (allEntries === false) {
        break;
      }
    }
  };

  /**
   * If set, the field is a combo box; if clear, the field is a list box.
   *
   * @name AcroFormChoiceField#combo
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "combo", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 18));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 18);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 18);
      }
    }
  });

  /**
   * If set, the combo box shall include an editable text box as well as a drop-down list; if clear, it shall include only a drop-down list. This flag shall be used only if the Combo flag is set.
   *
   * @name AcroFormChoiceField#edit
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "edit", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 19));
    },
    set: function(value) {
      //PDF 32000-1:2008, page 444
      if (this.combo === true) {
        if (Boolean(value) === true) {
          this.Ff = setBitForPdf(this.Ff, 19);
        } else {
          this.Ff = clearBitForPdf(this.Ff, 19);
        }
      }
    }
  });

  /**
   * If set, the field’s option items shall be sorted alphabetically. This flag is intended for use by writers, not by readers. Conforming readers shall display the options in the order in which they occur in the Opt array (see Table 231).
   *
   * @name AcroFormChoiceField#sort
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "sort", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 20));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 20);
        _Opt.sort();
      } else {
        this.Ff = clearBitForPdf(this.Ff, 20);
      }
    }
  });

  /**
   * (PDF 1.4) If set, more than one of the field’s option items may be selected simultaneously; if clear, at most one item shall be selected
   *
   * @name AcroFormChoiceField#multiSelect
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "multiSelect", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 22));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 22);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 22);
      }
    }
  });

  /**
   * (PDF 1.4) If set, text entered in the field shall not be spellchecked. This flag shall not be used unless the Combo and Edit flags are both set.
   *
   * @name AcroFormChoiceField#doNotSpellCheck
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "doNotSpellCheck", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 23));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 23);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 23);
      }
    }
  });

  /**
   * (PDF 1.5) If set, the new value shall be committed as soon as a selection is made (commonly with the pointing device). In this case, supplying a value for a field involves three actions: selecting the field for fill-in, selecting a choice for the fill-in value, and leaving that field, which finalizes or “commits” the data choice and triggers any actions associated with the entry or changing of this data. If this flag is on, then processing does not wait for leaving the field action to occur, but immediately proceeds to the third step.
   * This option enables applications to perform an action once a selection is made, without requiring the user to exit the field. If clear, the new value is not committed until the user exits the field.
   *
   * @name AcroFormChoiceField#commitOnSelChange
   * @default false
   * @type {boolean}
   */
  Object.defineProperty(this, "commitOnSelChange", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 27));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 27);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 27);
      }
    }
  });

  this.hasAppearanceStream = false;
};
inherit(AcroFormChoiceField, AcroFormField);

/**
 * @class AcroFormListBox
 * @extends AcroFormChoiceField
 * @extends AcroFormField
 */
var AcroFormListBox = function() {
  AcroFormChoiceField.call(this);
  this.fontName = "helvetica";

  //PDF 32000-1:2008, page 444
  this.combo = false;
};
inherit(AcroFormListBox, AcroFormChoiceField);

/**
 * @class AcroFormComboBox
 * @extends AcroFormListBox
 * @extends AcroFormChoiceField
 * @extends AcroFormField
 */
var AcroFormComboBox = function() {
  AcroFormListBox.call(this);
  this.combo = true;
};
inherit(AcroFormComboBox, AcroFormListBox);

/**
 * @class AcroFormEditBox
 * @extends AcroFormComboBox
 * @extends AcroFormListBox
 * @extends AcroFormChoiceField
 * @extends AcroFormField
 */
var AcroFormEditBox = function() {
  AcroFormComboBox.call(this);
  this.edit = true;
};
inherit(AcroFormEditBox, AcroFormComboBox);

/**
 * @class AcroFormButton
 * @extends AcroFormField
 */
var AcroFormButton = function() {
  AcroFormField.call(this);
  this.FT = "/Btn";

  /**
   * (Radio buttons only) If set, exactly one radio button shall be selected at all times; selecting the currently selected button has no effect. If clear, clicking the selected button deselects it, leaving no button selected.
   *
   * @name AcroFormButton#noToggleToOff
   * @type {boolean}
   */
  Object.defineProperty(this, "noToggleToOff", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 15));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 15);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 15);
      }
    }
  });

  /**
   * If set, the field is a set of radio buttons; if clear, the field is a checkbox. This flag may be set only if the Pushbutton flag is clear.
   *
   * @name AcroFormButton#radio
   * @type {boolean}
   */
  Object.defineProperty(this, "radio", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 16));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 16);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 16);
      }
    }
  });

  /**
   * If set, the field is a pushbutton that does not retain a permanent value.
   *
   * @name AcroFormButton#pushButton
   * @type {boolean}
   */
  Object.defineProperty(this, "pushButton", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 17));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 17);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 17);
      }
    }
  });

  /**
   * (PDF 1.5) If set, a group of radio buttons within a radio button field that use the same value for the on state will turn on and off in unison; that is if one is checked, they are all checked. If clear, the buttons are mutually exclusive (the same behavior as HTML radio buttons).
   *
   * @name AcroFormButton#radioIsUnison
   * @type {boolean}
   */
  Object.defineProperty(this, "radioIsUnison", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 26));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 26);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 26);
      }
    }
  });

  var _MK = {};
  Object.defineProperty(this, "MK", {
    enumerable: false,
    configurable: false,
    get: function() {
      var encryptor = function(data) {
        return data;
      };
      if (this.scope) encryptor = this.scope.internal.getEncryptor(this.objId);
      if (Object.keys(_MK).length !== 0) {
        var result = [];
        result.push("<<");
        var key;
        for (key in _MK) {
          result.push("/" + key + " (" + pdfEscape(encryptor(_MK[key])) + ")");
        }
        result.push(">>");
        return result.join("\n");
      }
      return undefined;
    },
    set: function(value) {
      if (typeof value === "object") {
        _MK = value;
      }
    }
  });

  /**
   * From the PDF reference:
   * (Optional, button fields only) The widget annotation's normal caption which shall be displayed when it is not interacting with the user.
   * Unlike the remaining entries listed in this Table which apply only to widget annotations associated with pushbutton fields (see Pushbuttons in 12.7.4.2, "Button Fields"), the CA entry may be used with any type of button field, including check boxes (see Check Boxes in 12.7.4.2, "Button Fields") and radio buttons (Radio Buttons in 12.7.4.2, "Button Fields").
   *
   * - '8' = Cross,
   * - 'l' =  Circle,
   * - '' = nothing
   * @name AcroFormButton#caption
   * @type {string}
   */
  Object.defineProperty(this, "caption", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _MK.CA || "";
    },
    set: function(value) {
      if (typeof value === "string") {
        _MK.CA = value;
      }
    }
  });

  var _AS;
  Object.defineProperty(this, "AS", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _AS;
    },
    set: function(value) {
      _AS = value;
    }
  });

  /**
   * (Required if the appearance dictionary AP contains one or more subdictionaries; PDF 1.2) The annotation's appearance state, which selects the applicable appearance stream from an appearance subdictionary (see Section 12.5.5, "Appearance Streams")
   *
   * @name AcroFormButton#appearanceState
   * @type {any}
   */
  Object.defineProperty(this, "appearanceState", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _AS.substr(1, _AS.length - 1);
    },
    set: function(value) {
      _AS = "/" + value;
    }
  });
};
inherit(AcroFormButton, AcroFormField);

/**
 * @class AcroFormPushButton
 * @extends AcroFormButton
 * @extends AcroFormField
 */
var AcroFormPushButton = function() {
  AcroFormButton.call(this);
  this.pushButton = true;
};
inherit(AcroFormPushButton, AcroFormButton);

/**
 * @class AcroFormRadioButton
 * @extends AcroFormButton
 * @extends AcroFormField
 */
var AcroFormRadioButton = function() {
  AcroFormButton.call(this);
  this.radio = true;
  this.pushButton = false;

  var _Kids = [];
  Object.defineProperty(this, "Kids", {
    enumerable: true,
    configurable: false,
    get: function() {
      return _Kids;
    },
    set: function(value) {
      if (typeof value !== "undefined") {
        _Kids = value;
      } else {
        _Kids = [];
      }
    }
  });
};
inherit(AcroFormRadioButton, AcroFormButton);

/**
 * The Child class of a RadioButton (the radioGroup) -> The single Buttons
 *
 * @class AcroFormChildClass
 * @extends AcroFormField
 * @ignore
 */
var AcroFormChildClass = function() {
  AcroFormField.call(this);

  var _parent;
  Object.defineProperty(this, "Parent", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _parent;
    },
    set: function(value) {
      _parent = value;
    }
  });

  var _optionName;
  Object.defineProperty(this, "optionName", {
    enumerable: false,
    configurable: true,
    get: function() {
      return _optionName;
    },
    set: function(value) {
      _optionName = value;
    }
  });

  var _MK = {};
  Object.defineProperty(this, "MK", {
    enumerable: false,
    configurable: false,
    get: function() {
      var encryptor = function(data) {
        return data;
      };
      if (this.scope) encryptor = this.scope.internal.getEncryptor(this.objId);
      var result = [];
      result.push("<<");
      var key;
      for (key in _MK) {
        result.push("/" + key + " (" + pdfEscape(encryptor(_MK[key])) + ")");
      }
      result.push(">>");
      return result.join("\n");
    },
    set: function(value) {
      if (typeof value === "object") {
        _MK = value;
      }
    }
  });

  /**
   * From the PDF reference:
   * (Optional, button fields only) The widget annotation's normal caption which shall be displayed when it is not interacting with the user.
   * Unlike the remaining entries listed in this Table which apply only to widget annotations associated with pushbutton fields (see Pushbuttons in 12.7.4.2, "Button Fields"), the CA entry may be used with any type of button field, including check boxes (see Check Boxes in 12.7.4.2, "Button Fields") and radio buttons (Radio Buttons in 12.7.4.2, "Button Fields").
   *
   * - '8' = Cross,
   * - 'l' =  Circle,
   * - '' = nothing
   * @name AcroFormButton#caption
   * @type {string}
   */
  Object.defineProperty(this, "caption", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _MK.CA || "";
    },
    set: function(value) {
      if (typeof value === "string") {
        _MK.CA = value;
      }
    }
  });

  var _AS;
  Object.defineProperty(this, "AS", {
    enumerable: false,
    configurable: false,
    get: function() {
      return _AS;
    },
    set: function(value) {
      _AS = value;
    }
  });

  /**
   * (Required if the appearance dictionary AP contains one or more subdictionaries; PDF 1.2) The annotation's appearance state, which selects the applicable appearance stream from an appearance subdictionary (see Section 12.5.5, "Appearance Streams")
   *
   * @name AcroFormButton#appearanceState
   * @type {any}
   */
  Object.defineProperty(this, "appearanceState", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _AS.substr(1, _AS.length - 1);
    },
    set: function(value) {
      _AS = "/" + value;
    }
  });
  this.caption = "l";
  this.appearanceState = "Off";
  // todo: set AppearanceType as variable that can be set from the
  // outside...
  this._AppearanceType = AcroFormAppearance.RadioButton.Circle;
  // The Default appearanceType is the Circle
  this.appearanceStreamContent = this._AppearanceType.createAppearanceStream(
    this.optionName
  );
};
inherit(AcroFormChildClass, AcroFormField);

AcroFormRadioButton.prototype.setAppearance = function(appearance) {
  if (!("createAppearanceStream" in appearance && "getCA" in appearance)) {
    throw new Error(
      "Couldn't assign Appearance to RadioButton. Appearance was Invalid!"
    );
  }
  for (var objId in this.Kids) {
    if (this.Kids.hasOwnProperty(objId)) {
      var child = this.Kids[objId];
      child.appearanceStreamContent = appearance.createAppearanceStream(
        child.optionName
      );
      child.caption = appearance.getCA();
    }
  }
};

AcroFormRadioButton.prototype.createOption = function(name) {
  // Create new Child for RadioGroup
  var child = new AcroFormChildClass();
  child.Parent = this;
  child.optionName = name;
  // Add to Parent
  this.Kids.push(child);

  addField.call(this.scope, child);

  return child;
};

/**
 * @class AcroFormCheckBox
 * @extends AcroFormButton
 * @extends AcroFormField
 */
var AcroFormCheckBox = function() {
  AcroFormButton.call(this);

  this.fontName = "zapfdingbats";
  this.caption = "3";
  this.appearanceState = "On";
  this.value = "On";
  this.textAlign = "center";
  this.appearanceStreamContent = AcroFormAppearance.CheckBox.createAppearanceStream();
};
inherit(AcroFormCheckBox, AcroFormButton);

/**
 * @class AcroFormTextField
 * @extends AcroFormField
 */
var AcroFormTextField = function() {
  AcroFormField.call(this);
  this.FT = "/Tx";

  /**
   * If set, the field may contain multiple lines of text; if clear, the field’s text shall be restricted to a single line.
   *
   * @name AcroFormTextField#multiline
   * @type {boolean}
   */
  Object.defineProperty(this, "multiline", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 13));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 13);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 13);
      }
    }
  });

  /**
   * (PDF 1.4) If set, the text entered in the field represents the pathname of a file whose contents shall be submitted as the value of the field.
   *
   * @name AcroFormTextField#fileSelect
   * @type {boolean}
   */
  Object.defineProperty(this, "fileSelect", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 21));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 21);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 21);
      }
    }
  });

  /**
   * (PDF 1.4) If set, text entered in the field shall not be spell-checked.
   *
   * @name AcroFormTextField#doNotSpellCheck
   * @type {boolean}
   */
  Object.defineProperty(this, "doNotSpellCheck", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 23));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 23);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 23);
      }
    }
  });

  /**
   * (PDF 1.4) If set, the field shall not scroll (horizontally for single-line fields, vertically for multiple-line fields) to accommodate more text than fits within its annotation rectangle. Once the field is full, no further text shall be accepted for interactive form filling; for noninteractive form filling, the filler should take care not to add more character than will visibly fit in the defined area.
   *
   * @name AcroFormTextField#doNotScroll
   * @type {boolean}
   */
  Object.defineProperty(this, "doNotScroll", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 24));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 24);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 24);
      }
    }
  });

  /**
   * (PDF 1.5) May be set only if the MaxLen entry is present in the text field dictionary (see Table 229) and if the Multiline, Password, and FileSelect flags are clear. If set, the field shall be automatically divided into as many equally spaced positions, or combs, as the value of MaxLen, and the text is laid out into those combs.
   *
   * @name AcroFormTextField#comb
   * @type {boolean}
   */
  Object.defineProperty(this, "comb", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 25));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 25);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 25);
      }
    }
  });

  /**
   * (PDF 1.5) If set, the value of this field shall be a rich text string (see 12.7.3.4, “Rich Text Strings”). If the field has a value, the RV entry of the field dictionary (Table 222) shall specify the rich text string.
   *
   * @name AcroFormTextField#richText
   * @type {boolean}
   */
  Object.defineProperty(this, "richText", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 26));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 26);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 26);
      }
    }
  });

  var _MaxLen = null;
  Object.defineProperty(this, "MaxLen", {
    enumerable: true,
    configurable: false,
    get: function() {
      return _MaxLen;
    },
    set: function(value) {
      _MaxLen = value;
    }
  });

  /**
   * (Optional; inheritable) The maximum length of the field’s text, in characters.
   *
   * @name AcroFormTextField#maxLength
   * @type {number}
   */
  Object.defineProperty(this, "maxLength", {
    enumerable: true,
    configurable: true,
    get: function() {
      return _MaxLen;
    },
    set: function(value) {
      if (Number.isInteger(value)) {
        _MaxLen = value;
      }
    }
  });

  Object.defineProperty(this, "hasAppearanceStream", {
    enumerable: true,
    configurable: true,
    get: function() {
      return this.V || this.DV;
    }
  });
};
inherit(AcroFormTextField, AcroFormField);

/**
 * @class AcroFormPasswordField
 * @extends AcroFormTextField
 * @extends AcroFormField
 */
var AcroFormPasswordField = function() {
  AcroFormTextField.call(this);

  /**
   * If set, the field is intended for entering a secure password that should not be echoed visibly to the screen. Characters typed from the keyboard shall instead be echoed in some unreadable form, such as asterisks or bullet characters.
   * NOTE To protect password confidentiality, readers should never store the value of the text field in the PDF file if this flag is set.
   *
   * @name AcroFormTextField#password
   * @type {boolean}
   */
  Object.defineProperty(this, "password", {
    enumerable: true,
    configurable: true,
    get: function() {
      return Boolean(getBitForPdf(this.Ff, 14));
    },
    set: function(value) {
      if (Boolean(value) === true) {
        this.Ff = setBitForPdf(this.Ff, 14);
      } else {
        this.Ff = clearBitForPdf(this.Ff, 14);
      }
    }
  });
  this.password = true;
};
inherit(AcroFormPasswordField, AcroFormTextField);

// Contains Methods for creating standard appearances
var AcroFormAppearance = {
  CheckBox: {
    createAppearanceStream: function() {
      var appearance = {
        N: {
          On: AcroFormAppearance.CheckBox.YesNormal
        },
        D: {
          On: AcroFormAppearance.CheckBox.YesPushDown,
          Off: AcroFormAppearance.CheckBox.OffPushDown
        }
      };

      return appearance;
    },
    /**
     * Returns the standard On Appearance for a CheckBox
     *
     * @returns {AcroFormXObject}
     */
    YesPushDown: function(formObject) {
      var xobj = createFormXObject(formObject);
      xobj.scope = formObject.scope;
      var stream = [];
      var fontKey = formObject.scope.internal.getFont(
        formObject.fontName,
        formObject.fontStyle
      ).id;
      var encodedColor = formObject.scope.__private__.encodeColorString(
        formObject.color
      );
      var calcRes = calculateX(formObject, formObject.caption);
      stream.push("0.749023 g");
      stream.push(
        "0 0 " +
          f2(AcroFormAppearance.internal.getWidth(formObject)) +
          " " +
          f2(AcroFormAppearance.internal.getHeight(formObject)) +
          " re"
      );
      stream.push("f");
      stream.push("BMC");
      stream.push("q");
      stream.push("0 0 1 rg");
      stream.push(
        "/" + fontKey + " " + f2(calcRes.fontSize) + " Tf " + encodedColor
      );
      stream.push("BT");
      stream.push(calcRes.text);
      stream.push("ET");
      stream.push("Q");
      stream.push("EMC");
      xobj.stream = stream.join("\n");
      return xobj;
    },

    YesNormal: function(formObject) {
      var xobj = createFormXObject(formObject);
      xobj.scope = formObject.scope;
      var fontKey = formObject.scope.internal.getFont(
        formObject.fontName,
        formObject.fontStyle
      ).id;
      var encodedColor = formObject.scope.__private__.encodeColorString(
        formObject.color
      );
      var stream = [];
      var height = AcroFormAppearance.internal.getHeight(formObject);
      var width = AcroFormAppearance.internal.getWidth(formObject);
      var calcRes = calculateX(formObject, formObject.caption);
      stream.push("1 g");
      stream.push("0 0 " + f2(width) + " " + f2(height) + " re");
      stream.push("f");
      stream.push("q");
      stream.push("0 0 1 rg");
      stream.push("0 0 " + f2(width - 1) + " " + f2(height - 1) + " re");
      stream.push("W");
      stream.push("n");
      stream.push("0 g");
      stream.push("BT");
      stream.push(
        "/" + fontKey + " " + f2(calcRes.fontSize) + " Tf " + encodedColor
      );
      stream.push(calcRes.text);
      stream.push("ET");
      stream.push("Q");
      xobj.stream = stream.join("\n");
      return xobj;
    },

    /**
     * Returns the standard Off Appearance for a CheckBox
     *
     * @returns {AcroFormXObject}
     */
    OffPushDown: function(formObject) {
      var xobj = createFormXObject(formObject);
      xobj.scope = formObject.scope;
      var stream = [];
      stream.push("0.749023 g");
      stream.push(
        "0 0 " +
          f2(AcroFormAppearance.internal.getWidth(formObject)) +
          " " +
          f2(AcroFormAppearance.internal.getHeight(formObject)) +
          " re"
      );
      stream.push("f");
      xobj.stream = stream.join("\n");
      return xobj;
    }
  },

  RadioButton: {
    Circle: {
      createAppearanceStream: function(name) {
        var appearanceStreamContent = {
          D: {
            Off: AcroFormAppearance.RadioButton.Circle.OffPushDown
          },
          N: {}
        };
        appearanceStreamContent.N[name] =
          AcroFormAppearance.RadioButton.Circle.YesNormal;
        appearanceStreamContent.D[name] =
          AcroFormAppearance.RadioButton.Circle.YesPushDown;
        return appearanceStreamContent;
      },
      getCA: function() {
        return "l";
      },

      YesNormal: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var stream = [];
        // Make the Radius of the Circle relative to min(height, width) of formObject
        var DotRadius =
          AcroFormAppearance.internal.getWidth(formObject) <=
          AcroFormAppearance.internal.getHeight(formObject)
            ? AcroFormAppearance.internal.getWidth(formObject) / 4
            : AcroFormAppearance.internal.getHeight(formObject) / 4;
        // The Borderpadding...
        DotRadius = Number((DotRadius * 0.9).toFixed(5));
        var c = AcroFormAppearance.internal.Bezier_C;
        var DotRadiusBezier = Number((DotRadius * c).toFixed(5));
        /*
         * The Following is a Circle created with Bezier-Curves.
         */
        stream.push("q");
        stream.push(
          "1 0 0 1 " +
            f5(AcroFormAppearance.internal.getWidth(formObject) / 2) +
            " " +
            f5(AcroFormAppearance.internal.getHeight(formObject) / 2) +
            " cm"
        );
        stream.push(DotRadius + " 0 m");
        stream.push(
          DotRadius +
            " " +
            DotRadiusBezier +
            " " +
            DotRadiusBezier +
            " " +
            DotRadius +
            " 0 " +
            DotRadius +
            " c"
        );
        stream.push(
          "-" +
            DotRadiusBezier +
            " " +
            DotRadius +
            " -" +
            DotRadius +
            " " +
            DotRadiusBezier +
            " -" +
            DotRadius +
            " 0 c"
        );
        stream.push(
          "-" +
            DotRadius +
            " -" +
            DotRadiusBezier +
            " -" +
            DotRadiusBezier +
            " -" +
            DotRadius +
            " 0 -" +
            DotRadius +
            " c"
        );
        stream.push(
          DotRadiusBezier +
            " -" +
            DotRadius +
            " " +
            DotRadius +
            " -" +
            DotRadiusBezier +
            " " +
            DotRadius +
            " 0 c"
        );
        stream.push("f");
        stream.push("Q");
        xobj.stream = stream.join("\n");
        return xobj;
      },
      YesPushDown: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var stream = [];
        var DotRadius =
          AcroFormAppearance.internal.getWidth(formObject) <=
          AcroFormAppearance.internal.getHeight(formObject)
            ? AcroFormAppearance.internal.getWidth(formObject) / 4
            : AcroFormAppearance.internal.getHeight(formObject) / 4;
        // The Borderpadding...
        DotRadius = Number((DotRadius * 0.9).toFixed(5));
        // Save results for later use; no need to waste
        // processor ticks on doing math
        var k = Number((DotRadius * 2).toFixed(5));
        var kc = Number((k * AcroFormAppearance.internal.Bezier_C).toFixed(5));
        var dc = Number(
          (DotRadius * AcroFormAppearance.internal.Bezier_C).toFixed(5)
        );

        stream.push("0.749023 g");
        stream.push("q");
        stream.push(
          "1 0 0 1 " +
            f5(AcroFormAppearance.internal.getWidth(formObject) / 2) +
            " " +
            f5(AcroFormAppearance.internal.getHeight(formObject) / 2) +
            " cm"
        );
        stream.push(k + " 0 m");
        stream.push(k + " " + kc + " " + kc + " " + k + " 0 " + k + " c");
        stream.push(
          "-" + kc + " " + k + " -" + k + " " + kc + " -" + k + " 0 c"
        );
        stream.push(
          "-" + k + " -" + kc + " -" + kc + " -" + k + " 0 -" + k + " c"
        );
        stream.push(kc + " -" + k + " " + k + " -" + kc + " " + k + " 0 c");
        stream.push("f");
        stream.push("Q");
        stream.push("0 g");
        stream.push("q");
        stream.push(
          "1 0 0 1 " +
            f5(AcroFormAppearance.internal.getWidth(formObject) / 2) +
            " " +
            f5(AcroFormAppearance.internal.getHeight(formObject) / 2) +
            " cm"
        );
        stream.push(DotRadius + " 0 m");
        stream.push(
          "" +
            DotRadius +
            " " +
            dc +
            " " +
            dc +
            " " +
            DotRadius +
            " 0 " +
            DotRadius +
            " c"
        );
        stream.push(
          "-" +
            dc +
            " " +
            DotRadius +
            " -" +
            DotRadius +
            " " +
            dc +
            " -" +
            DotRadius +
            " 0 c"
        );
        stream.push(
          "-" +
            DotRadius +
            " -" +
            dc +
            " -" +
            dc +
            " -" +
            DotRadius +
            " 0 -" +
            DotRadius +
            " c"
        );
        stream.push(
          dc +
            " -" +
            DotRadius +
            " " +
            DotRadius +
            " -" +
            dc +
            " " +
            DotRadius +
            " 0 c"
        );
        stream.push("f");
        stream.push("Q");
        xobj.stream = stream.join("\n");
        return xobj;
      },
      OffPushDown: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var stream = [];
        var DotRadius =
          AcroFormAppearance.internal.getWidth(formObject) <=
          AcroFormAppearance.internal.getHeight(formObject)
            ? AcroFormAppearance.internal.getWidth(formObject) / 4
            : AcroFormAppearance.internal.getHeight(formObject) / 4;
        // The Borderpadding...
        DotRadius = Number((DotRadius * 0.9).toFixed(5));
        // Save results for later use; no need to waste
        // processor ticks on doing math
        var k = Number((DotRadius * 2).toFixed(5));
        var kc = Number((k * AcroFormAppearance.internal.Bezier_C).toFixed(5));

        stream.push("0.749023 g");
        stream.push("q");
        stream.push(
          "1 0 0 1 " +
            f5(AcroFormAppearance.internal.getWidth(formObject) / 2) +
            " " +
            f5(AcroFormAppearance.internal.getHeight(formObject) / 2) +
            " cm"
        );
        stream.push(k + " 0 m");
        stream.push(k + " " + kc + " " + kc + " " + k + " 0 " + k + " c");
        stream.push(
          "-" + kc + " " + k + " -" + k + " " + kc + " -" + k + " 0 c"
        );
        stream.push(
          "-" + k + " -" + kc + " -" + kc + " -" + k + " 0 -" + k + " c"
        );
        stream.push(kc + " -" + k + " " + k + " -" + kc + " " + k + " 0 c");
        stream.push("f");
        stream.push("Q");
        xobj.stream = stream.join("\n");
        return xobj;
      }
    },

    Cross: {
      /**
       * Creates the Actual AppearanceDictionary-References
       *
       * @param {string} name
       * @returns {Object}
       * @ignore
       */
      createAppearanceStream: function(name) {
        var appearanceStreamContent = {
          D: {
            Off: AcroFormAppearance.RadioButton.Cross.OffPushDown
          },
          N: {}
        };
        appearanceStreamContent.N[name] =
          AcroFormAppearance.RadioButton.Cross.YesNormal;
        appearanceStreamContent.D[name] =
          AcroFormAppearance.RadioButton.Cross.YesPushDown;
        return appearanceStreamContent;
      },
      getCA: function() {
        return "8";
      },

      YesNormal: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var stream = [];
        var cross = AcroFormAppearance.internal.calculateCross(formObject);
        stream.push("q");
        stream.push(
          "1 1 " +
            f2(AcroFormAppearance.internal.getWidth(formObject) - 2) +
            " " +
            f2(AcroFormAppearance.internal.getHeight(formObject) - 2) +
            " re"
        );
        stream.push("W");
        stream.push("n");
        stream.push(f2(cross.x1.x) + " " + f2(cross.x1.y) + " m");
        stream.push(f2(cross.x2.x) + " " + f2(cross.x2.y) + " l");
        stream.push(f2(cross.x4.x) + " " + f2(cross.x4.y) + " m");
        stream.push(f2(cross.x3.x) + " " + f2(cross.x3.y) + " l");
        stream.push("s");
        stream.push("Q");
        xobj.stream = stream.join("\n");
        return xobj;
      },
      YesPushDown: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var cross = AcroFormAppearance.internal.calculateCross(formObject);
        var stream = [];
        stream.push("0.749023 g");
        stream.push(
          "0 0 " +
            f2(AcroFormAppearance.internal.getWidth(formObject)) +
            " " +
            f2(AcroFormAppearance.internal.getHeight(formObject)) +
            " re"
        );
        stream.push("f");
        stream.push("q");
        stream.push(
          "1 1 " +
            f2(AcroFormAppearance.internal.getWidth(formObject) - 2) +
            " " +
            f2(AcroFormAppearance.internal.getHeight(formObject) - 2) +
            " re"
        );
        stream.push("W");
        stream.push("n");
        stream.push(f2(cross.x1.x) + " " + f2(cross.x1.y) + " m");
        stream.push(f2(cross.x2.x) + " " + f2(cross.x2.y) + " l");
        stream.push(f2(cross.x4.x) + " " + f2(cross.x4.y) + " m");
        stream.push(f2(cross.x3.x) + " " + f2(cross.x3.y) + " l");
        stream.push("s");
        stream.push("Q");
        xobj.stream = stream.join("\n");
        return xobj;
      },
      OffPushDown: function(formObject) {
        var xobj = createFormXObject(formObject);
        xobj.scope = formObject.scope;
        var stream = [];
        stream.push("0.749023 g");
        stream.push(
          "0 0 " +
            f2(AcroFormAppearance.internal.getWidth(formObject)) +
            " " +
            f2(AcroFormAppearance.internal.getHeight(formObject)) +
            " re"
        );
        stream.push("f");
        xobj.stream = stream.join("\n");
        return xobj;
      }
    }
  },

  /**
   * Returns the standard Appearance
   *
   * @returns {AcroFormXObject}
   */
  createDefaultAppearanceStream: function(formObject) {
    // Set Helvetica to Standard Font (size: auto)
    // Color: Black
    var fontKey = formObject.scope.internal.getFont(
      formObject.fontName,
      formObject.fontStyle
    ).id;
    var encodedColor = formObject.scope.__private__.encodeColorString(
      formObject.color
    );
    var fontSize = formObject.fontSize;
    var result = "/" + fontKey + " " + fontSize + " Tf " + encodedColor;
    return result;
  }
};

AcroFormAppearance.internal = {
  Bezier_C: 0.551915024494,

  calculateCross: function(formObject) {
    var width = AcroFormAppearance.internal.getWidth(formObject);
    var height = AcroFormAppearance.internal.getHeight(formObject);
    var a = Math.min(width, height);

    var cross = {
      x1: {
        // upperLeft
        x: (width - a) / 2,
        y: (height - a) / 2 + a // height - borderPadding
      },
      x2: {
        // lowerRight
        x: (width - a) / 2 + a,
        y: (height - a) / 2 // borderPadding
      },
      x3: {
        // lowerLeft
        x: (width - a) / 2,
        y: (height - a) / 2 // borderPadding
      },
      x4: {
        // upperRight
        x: (width - a) / 2 + a,
        y: (height - a) / 2 + a // height - borderPadding
      }
    };

    return cross;
  }
};
AcroFormAppearance.internal.getWidth = function(formObject) {
  var result = 0;
  if (typeof formObject === "object") {
    result = scale(formObject.Rect[2]);
  }
  return result;
};
AcroFormAppearance.internal.getHeight = function(formObject) {
  var result = 0;
  if (typeof formObject === "object") {
    result = scale(formObject.Rect[3]);
  }
  return result;
};

// Public:

/**
 * Add an AcroForm-Field to the jsPDF-instance
 *
 * @name addField
 * @function
 * @instance
 * @param {Object} fieldObject
 * @returns {jsPDF}
 */
var addField = (jsPDFAPI.addField = function(fieldObject) {
  initializeAcroForm(this, fieldObject);

  if (fieldObject instanceof AcroFormField) {
    putForm(fieldObject);
  } else {
    throw new Error("Invalid argument passed to jsPDF.addField.");
  }
  fieldObject.page = fieldObject.scope.internal.getCurrentPageInfo().pageNumber;
  return this;
});

jsPDFAPI.AcroFormChoiceField = AcroFormChoiceField;
jsPDFAPI.AcroFormListBox = AcroFormListBox;
jsPDFAPI.AcroFormComboBox = AcroFormComboBox;
jsPDFAPI.AcroFormEditBox = AcroFormEditBox;
jsPDFAPI.AcroFormButton = AcroFormButton;
jsPDFAPI.AcroFormPushButton = AcroFormPushButton;
jsPDFAPI.AcroFormRadioButton = AcroFormRadioButton;
jsPDFAPI.AcroFormCheckBox = AcroFormCheckBox;
jsPDFAPI.AcroFormTextField = AcroFormTextField;
jsPDFAPI.AcroFormPasswordField = AcroFormPasswordField;
jsPDFAPI.AcroFormAppearance = AcroFormAppearance;

jsPDFAPI.AcroForm = {
  ChoiceField: AcroFormChoiceField,
  ListBox: AcroFormListBox,
  ComboBox: AcroFormComboBox,
  EditBox: AcroFormEditBox,
  Button: AcroFormButton,
  PushButton: AcroFormPushButton,
  RadioButton: AcroFormRadioButton,
  CheckBox: AcroFormCheckBox,
  TextField: AcroFormTextField,
  PasswordField: AcroFormPasswordField,
  Appearance: AcroFormAppearance
};

jsPDF.AcroForm = {
  ChoiceField: AcroFormChoiceField,
  ListBox: AcroFormListBox,
  ComboBox: AcroFormComboBox,
  EditBox: AcroFormEditBox,
  Button: AcroFormButton,
  PushButton: AcroFormPushButton,
  RadioButton: AcroFormRadioButton,
  CheckBox: AcroFormCheckBox,
  TextField: AcroFormTextField,
  PasswordField: AcroFormPasswordField,
  Appearance: AcroFormAppearance
};

var AcroForm = jsPDF.AcroForm;

/** @license
 * jsPDF addImage plugin
 * Copyright (c) 2012 Jason Siefken, https://github.com/siefkenj/
 *               2013 Chris Dowling, https://github.com/gingerchris
 *               2013 Trinh Ho, https://github.com/ineedfat
 *               2013 Edwin Alejandro Perez, https://github.com/eaparango
 *               2013 Norah Smith, https://github.com/burnburnrocket
 *               2014 Diego Casorran, https://github.com/diegocr
 *               2014 James Robb, https://github.com/jamesbrobb
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

(function(jsPDFAPI) {

  var namespace = "addImage_";
  jsPDFAPI.__addimage__ = {};

  var UNKNOWN = "UNKNOWN";

  // Heuristic selection of a good batch for large array .apply. Not limiting make the call overflow.
  // With too small batch iteration will be slow as more calls are made,
  // higher values cause larger and slower garbage collection.
  var ARRAY_APPLY_BATCH = 8192;

  var imageFileTypeHeaders = {
    PNG: [[0x89, 0x50, 0x4e, 0x47]],
    TIFF: [
      [0x4d, 0x4d, 0x00, 0x2a], //Motorola
      [0x49, 0x49, 0x2a, 0x00] //Intel
    ],
    JPEG: [
      [
        0xff,
        0xd8,
        0xff,
        0xe0,
        undefined,
        undefined,
        0x4a,
        0x46,
        0x49,
        0x46,
        0x00
      ], //JFIF
      [
        0xff,
        0xd8,
        0xff,
        0xe1,
        undefined,
        undefined,
        0x45,
        0x78,
        0x69,
        0x66,
        0x00,
        0x00
      ], //Exif
      [0xff, 0xd8, 0xff, 0xdb], //JPEG RAW
      [0xff, 0xd8, 0xff, 0xee] //EXIF RAW
    ],
    JPEG2000: [[0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20]],
    GIF87a: [[0x47, 0x49, 0x46, 0x38, 0x37, 0x61]],
    GIF89a: [[0x47, 0x49, 0x46, 0x38, 0x39, 0x61]],
    WEBP: [
      [
        0x52,
        0x49,
        0x46,
        0x46,
        undefined,
        undefined,
        undefined,
        undefined,
        0x57,
        0x45,
        0x42,
        0x50
      ]
    ],
    BMP: [
      [0x42, 0x4d], //BM - Windows 3.1x, 95, NT, ... etc.
      [0x42, 0x41], //BA - OS/2 struct bitmap array
      [0x43, 0x49], //CI - OS/2 struct color icon
      [0x43, 0x50], //CP - OS/2 const color pointer
      [0x49, 0x43], //IC - OS/2 struct icon
      [0x50, 0x54] //PT - OS/2 pointer
    ]
  };

  /**
   * Recognize filetype of Image by magic-bytes
   *
   * https://en.wikipedia.org/wiki/List_of_file_signatures
   *
   * @name getImageFileTypeByImageData
   * @public
   * @function
   * @param {string|arraybuffer} imageData imageData as binary String or arraybuffer
   * @param {string} format format of file if filetype-recognition fails, e.g. 'JPEG'
   *
   * @returns {string} filetype of Image
   */
  var getImageFileTypeByImageData = (jsPDFAPI.__addimage__.getImageFileTypeByImageData = function(
    imageData,
    fallbackFormat
  ) {
    fallbackFormat = fallbackFormat || UNKNOWN;
    var i;
    var j;
    var result = UNKNOWN;
    var headerSchemata;
    var compareResult;
    var fileType;

    if (
      fallbackFormat === "RGBA" ||
      (imageData.data !== undefined &&
        imageData.data instanceof Uint8ClampedArray &&
        "height" in imageData &&
        "width" in imageData)
    ) {
      return "RGBA";
    }

    if (isArrayBufferView(imageData)) {
      for (fileType in imageFileTypeHeaders) {
        headerSchemata = imageFileTypeHeaders[fileType];
        for (i = 0; i < headerSchemata.length; i += 1) {
          compareResult = true;
          for (j = 0; j < headerSchemata[i].length; j += 1) {
            if (headerSchemata[i][j] === undefined) {
              continue;
            }
            if (headerSchemata[i][j] !== imageData[j]) {
              compareResult = false;
              break;
            }
          }
          if (compareResult === true) {
            result = fileType;
            break;
          }
        }
      }
    } else {
      for (fileType in imageFileTypeHeaders) {
        headerSchemata = imageFileTypeHeaders[fileType];
        for (i = 0; i < headerSchemata.length; i += 1) {
          compareResult = true;
          for (j = 0; j < headerSchemata[i].length; j += 1) {
            if (headerSchemata[i][j] === undefined) {
              continue;
            }
            if (headerSchemata[i][j] !== imageData.charCodeAt(j)) {
              compareResult = false;
              break;
            }
          }
          if (compareResult === true) {
            result = fileType;
            break;
          }
        }
      }
    }

    if (result === UNKNOWN && fallbackFormat !== UNKNOWN) {
      result = fallbackFormat;
    }
    return result;
  });

  // Image functionality ported from pdf.js
  var putImage = function(image) {
    var out = this.internal.write;
    var putStream = this.internal.putStream;
    var getFilters = this.internal.getFilters;

    var filter = getFilters();
    while (filter.indexOf("FlateEncode") !== -1) {
      filter.splice(filter.indexOf("FlateEncode"), 1);
    }

    image.objectId = this.internal.newObject();

    var additionalKeyValues = [];
    additionalKeyValues.push({ key: "Type", value: "/XObject" });
    additionalKeyValues.push({ key: "Subtype", value: "/Image" });
    additionalKeyValues.push({ key: "Width", value: image.width });
    additionalKeyValues.push({ key: "Height", value: image.height });

    if (image.colorSpace === color_spaces.INDEXED) {
      additionalKeyValues.push({
        key: "ColorSpace",
        value:
          "[/Indexed /DeviceRGB " +
          // if an indexed png defines more than one colour with transparency, we've created a sMask
          (image.palette.length / 3 - 1) +
          " " +
          ("sMask" in image && typeof image.sMask !== "undefined"
            ? image.objectId + 2
            : image.objectId + 1) +
          " 0 R]"
      });
    } else {
      additionalKeyValues.push({
        key: "ColorSpace",
        value: "/" + image.colorSpace
      });
      if (image.colorSpace === color_spaces.DEVICE_CMYK) {
        additionalKeyValues.push({ key: "Decode", value: "[1 0 1 0 1 0 1 0]" });
      }
    }
    additionalKeyValues.push({
      key: "BitsPerComponent",
      value: image.bitsPerComponent
    });
    if (
      "decodeParameters" in image &&
      typeof image.decodeParameters !== "undefined"
    ) {
      additionalKeyValues.push({
        key: "DecodeParms",
        value: "<<" + image.decodeParameters + ">>"
      });
    }
    if (
      "transparency" in image &&
      Array.isArray(image.transparency) &&
      image.transparency.length > 0
    ) {
      var transparency = "",
        i = 0,
        len = image.transparency.length;
      for (; i < len; i++)
        transparency +=
          image.transparency[i] + " " + image.transparency[i] + " ";

      additionalKeyValues.push({
        key: "Mask",
        value: "[" + transparency + "]"
      });
    }
    if (typeof image.sMask !== "undefined") {
      additionalKeyValues.push({
        key: "SMask",
        value: image.objectId + 1 + " 0 R"
      });
    }

    var alreadyAppliedFilters =
      typeof image.filter !== "undefined" ? ["/" + image.filter] : undefined;

    putStream({
      data: image.data,
      additionalKeyValues: additionalKeyValues,
      alreadyAppliedFilters: alreadyAppliedFilters,
      objectId: image.objectId
    });

    out("endobj");

    // Soft mask
    if ("sMask" in image && typeof image.sMask !== "undefined") {
      const sMaskBitsPerComponent =
        image.sMaskBitsPerComponent ?? image.bitsPerComponent;
      const sMask = {
        width: image.width,
        height: image.height,
        colorSpace: "DeviceGray",
        bitsPerComponent: sMaskBitsPerComponent,
        data: image.sMask
      };
      if ("filter" in image) {
        sMask.decodeParameters = `/Predictor ${image.predictor} /Colors 1 /BitsPerComponent ${sMaskBitsPerComponent} /Columns ${image.width}`;
        sMask.filter = image.filter;
      }
      putImage.call(this, sMask);
    }

    //Palette
    if (image.colorSpace === color_spaces.INDEXED) {
      var objId = this.internal.newObject();
      //out('<< /Filter / ' + img['f'] +' /Length ' + img['pal'].length + '>>');
      //putStream(zlib.compress(img['pal']));
      putStream({
        data: arrayBufferToBinaryString(new Uint8Array(image.palette)),
        objectId: objId
      });
      out("endobj");
    }
  };
  var putResourcesCallback = function() {
    var images = this.internal.collections[namespace + "images"];
    for (var i in images) {
      putImage.call(this, images[i]);
    }
  };
  var putXObjectsDictCallback = function() {
    var images = this.internal.collections[namespace + "images"],
      out = this.internal.write,
      image;
    for (var i in images) {
      image = images[i];
      out("/I" + image.index, image.objectId, "0", "R");
    }
  };

  var checkCompressValue = function(value) {
    if (value && typeof value === "string") value = value.toUpperCase();
    return value in jsPDFAPI.image_compression ? value : image_compression.NONE;
  };

  var initialize = function() {
    if (!this.internal.collections[namespace + "images"]) {
      this.internal.collections[namespace + "images"] = {};
      this.internal.events.subscribe("putResources", putResourcesCallback);
      this.internal.events.subscribe("putXobjectDict", putXObjectsDictCallback);
    }
  };

  var getImages = function() {
    var images = this.internal.collections[namespace + "images"];
    initialize.call(this);
    return images;
  };
  var getImageIndex = function() {
    return Object.keys(this.internal.collections[namespace + "images"]).length;
  };
  var notDefined = function(value) {
    return typeof value === "undefined" || value === null || value.length === 0;
  };
  var generateAliasFromImageData = function(imageData) {
    if (typeof imageData === "string" || isArrayBufferView(imageData)) {
      return sHashCode(imageData);
    } else if (isArrayBufferView(imageData.data)) {
      return sHashCode(imageData.data);
    }

    return null;
  };

  var isImageTypeSupported = function(type) {
    return typeof jsPDFAPI["process" + type.toUpperCase()] === "function";
  };

  var isDOMElement = function(object) {
    return typeof object === "object" && object.nodeType === 1;
  };

  var getImageDataFromElement = function(element, format) {
    //if element is an image which uses data url definition, just return the dataurl
    if (element.nodeName === "IMG" && element.hasAttribute("src")) {
      var src = "" + element.getAttribute("src");

      //is base64 encoded dataUrl, directly process it
      if (src.indexOf("data:image/") === 0) {
        return atob(
          unescape(src)
            .split("base64,")
            .pop()
        );
      }

      //it is probably an url, try to load it
      var tmpImageData = jsPDFAPI.loadFile(src, true);
      if (tmpImageData !== undefined) {
        return tmpImageData;
      }
    }

    if (element.nodeName === "CANVAS") {
      if (element.width === 0 || element.height === 0) {
        throw new Error(
          "Given canvas must have data. Canvas width: " +
            element.width +
            ", height: " +
            element.height
        );
      }
      var mimeType;
      switch (format) {
        case "PNG":
          mimeType = "image/png";
          break;
        case "WEBP":
          mimeType = "image/webp";
          break;
        case "JPEG":
        case "JPG":
        default:
          mimeType = "image/jpeg";
          break;
      }
      return atob(
        element
          .toDataURL(mimeType, 1.0)
          .split("base64,")
          .pop()
      );
    }
  };

  var checkImagesForAlias = function(alias) {
    var images = this.internal.collections[namespace + "images"];
    if (images) {
      for (var e in images) {
        if (alias === images[e].alias) {
          return images[e];
        }
      }
    }
  };

  var determineWidthAndHeight = function(width, height, image) {
    if (!width && !height) {
      width = -96;
      height = -96;
    }
    if (width < 0) {
      width = (-1 * image.width * 72) / width / this.internal.scaleFactor;
    }
    if (height < 0) {
      height = (-1 * image.height * 72) / height / this.internal.scaleFactor;
    }
    if (width === 0) {
      width = (height * image.width) / image.height;
    }
    if (height === 0) {
      height = (width * image.height) / image.width;
    }

    return [width, height];
  };

  var writeImageToPDF = function(x, y, width, height, image, rotation) {
    var dims = determineWidthAndHeight.call(this, width, height, image),
      coord = this.internal.getCoordinateString,
      vcoord = this.internal.getVerticalCoordinateString;

    var images = getImages.call(this);

    width = dims[0];
    height = dims[1];
    images[image.index] = image;

    if (rotation) {
      rotation *= Math.PI / 180;
      var c = Math.cos(rotation);
      var s = Math.sin(rotation);
      //like in pdf Reference do it 4 digits instead of 2
      var f4 = function(number) {
        return number.toFixed(4);
      };
      var rotationTransformationMatrix = [
        f4(c),
        f4(s),
        f4(s * -1),
        f4(c),
        0,
        0,
        "cm"
      ];
    }
    this.internal.write("q"); //Save graphics state
    if (rotation) {
      this.internal.write(
        [1, "0", "0", 1, coord(x), vcoord(y + height), "cm"].join(" ")
      ); //Translate
      this.internal.write(rotationTransformationMatrix.join(" ")); //Rotate
      this.internal.write(
        [coord(width), "0", "0", coord(height), "0", "0", "cm"].join(" ")
      ); //Scale
    } else {
      this.internal.write(
        [
          coord(width),
          "0",
          "0",
          coord(height),
          coord(x),
          vcoord(y + height),
          "cm"
        ].join(" ")
      ); //Translate and Scale
    }

    if (this.isAdvancedAPI()) {
      // draw image bottom up when in "advanced" API mode
      this.internal.write([1, 0, 0, -1, 0, 0, "cm"].join(" "));
    }

    this.internal.write("/I" + image.index + " Do"); //Paint Image
    this.internal.write("Q"); //Restore graphics state
  };

  /**
   * COLOR SPACES
   */
  var color_spaces = (jsPDFAPI.color_spaces = {
    DEVICE_RGB: "DeviceRGB",
    DEVICE_GRAY: "DeviceGray",
    DEVICE_CMYK: "DeviceCMYK",
    CAL_GREY: "CalGray",
    CAL_RGB: "CalRGB",
    LAB: "Lab",
    ICC_BASED: "ICCBased",
    INDEXED: "Indexed",
    PATTERN: "Pattern",
    SEPARATION: "Separation",
    DEVICE_N: "DeviceN"
  });

  /**
   * DECODE METHODS
   */
  jsPDFAPI.decode = {
    DCT_DECODE: "DCTDecode",
    FLATE_DECODE: "FlateDecode",
    LZW_DECODE: "LZWDecode",
    JPX_DECODE: "JPXDecode",
    JBIG2_DECODE: "JBIG2Decode",
    ASCII85_DECODE: "ASCII85Decode",
    ASCII_HEX_DECODE: "ASCIIHexDecode",
    RUN_LENGTH_DECODE: "RunLengthDecode",
    CCITT_FAX_DECODE: "CCITTFaxDecode"
  };

  /**
   * IMAGE COMPRESSION TYPES
   */
  var image_compression = (jsPDFAPI.image_compression = {
    NONE: "NONE",
    FAST: "FAST",
    MEDIUM: "MEDIUM",
    SLOW: "SLOW"
  });

  /**
   * @name sHashCode
   * @function
   * @param {string} data
   * @returns {string}
   */
  var sHashCode = (jsPDFAPI.__addimage__.sHashCode = function(data) {
    var hash = 0,
      i,
      len;

    if (typeof data === "string") {
      len = data.length;
      for (i = 0; i < len; i++) {
        hash = (hash << 5) - hash + data.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
      }
    } else if (isArrayBufferView(data)) {
      len = data.byteLength / 2;
      for (i = 0; i < len; i++) {
        hash = (hash << 5) - hash + data[i];
        hash |= 0; // Convert to 32bit integer
      }
    }
    return hash;
  });

  /**
   * Validates if given String is a valid Base64-String
   *
   * @name validateStringAsBase64
   * @public
   * @function
   * @param {String} possible Base64-String
   *
   * @returns {boolean}
   */
  var validateStringAsBase64 = (jsPDFAPI.__addimage__.validateStringAsBase64 = function(
    possibleBase64String
  ) {
    possibleBase64String = possibleBase64String || "";
    possibleBase64String.toString().trim();

    var result = true;

    if (possibleBase64String.length === 0) {
      result = false;
    }

    if (possibleBase64String.length % 4 !== 0) {
      result = false;
    }

    if (
      /^[A-Za-z0-9+/]+$/.test(
        possibleBase64String.substr(0, possibleBase64String.length - 2)
      ) === false
    ) {
      result = false;
    }

    if (
      /^[A-Za-z0-9/][A-Za-z0-9+/]|[A-Za-z0-9+/]=|==$/.test(
        possibleBase64String.substr(-2)
      ) === false
    ) {
      result = false;
    }
    return result;
  });

  /**
   * Strips out and returns info from a valid base64 data URI
   *
   * @name extractImageFromDataUrl
   * @function
   * @param {string} dataUrl a valid data URI of format 'data:[<MIME-type>][;base64],<data>'
   * @returns {string} The raw Base64-encoded data.
   */
  var extractImageFromDataUrl = (jsPDFAPI.__addimage__.extractImageFromDataUrl = function(
    dataUrl
  ) {
    if (dataUrl == null) {
      return null;
    }

    // avoid using a regexp for parsing because it might be vulnerable against ReDoS attacks

    dataUrl = dataUrl.trim();

    if (!dataUrl.startsWith("data:")) {
      return null;
    }

    const commaIndex = dataUrl.indexOf(",");
    if (commaIndex < 0) {
      return null;
    }

    const dataScheme = dataUrl.substring(0, commaIndex).trim();
    if (!dataScheme.endsWith("base64")) {
      return null;
    }

    return dataUrl.substring(commaIndex + 1);
  });

  /**
   * Tests supplied object to determine if ArrayBuffer
   *
   * @name isArrayBuffer
   * @function
   * @param {Object} object an Object
   *
   * @returns {boolean}
   */
  jsPDFAPI.__addimage__.isArrayBuffer = function(object) {
    return object instanceof ArrayBuffer;
  };

  /**
   * Tests supplied object to determine if it implements the ArrayBufferView (TypedArray) interface
   *
   * @name isArrayBufferView
   * @function
   * @param {Object} object an Object
   * @returns {boolean}
   */
  var isArrayBufferView = (jsPDFAPI.__addimage__.isArrayBufferView = function(
    object
  ) {
    return (
      object instanceof Int8Array ||
      object instanceof Uint8Array ||
      object instanceof Uint8ClampedArray ||
      object instanceof Int16Array ||
      object instanceof Uint16Array ||
      object instanceof Int32Array ||
      object instanceof Uint32Array ||
      object instanceof Float32Array ||
      object instanceof Float64Array
    );
  });

  /**
   * Convert Binary String to ArrayBuffer
   *
   * @name binaryStringToUint8Array
   * @public
   * @function
   * @param {string} BinaryString with ImageData
   * @returns {Uint8Array}
   */
  var binaryStringToUint8Array = (jsPDFAPI.__addimage__.binaryStringToUint8Array = function(
    binary_string
  ) {
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
      bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes;
  });

  /**
   * Convert the Buffer to a Binary String
   *
   * @name arrayBufferToBinaryString
   * @public
   * @function
   * @param {ArrayBuffer|ArrayBufferView} ArrayBuffer buffer or bufferView with ImageData
   *
   * @returns {String}
   */
  var arrayBufferToBinaryString = (jsPDFAPI.__addimage__.arrayBufferToBinaryString = function(
    buffer
  ) {
    var out = "";
    // There are calls with both ArrayBuffer and already converted Uint8Array or other BufferView.
    // Do not copy the array if input is already an array.
    var buf = isArrayBufferView(buffer) ? buffer : new Uint8Array(buffer);
    for (var i = 0; i < buf.length; i += ARRAY_APPLY_BATCH) {
      // Limit the amount of characters being parsed to prevent overflow.
      // Note that while TextDecoder would be faster, it does not have the same
      // functionality as fromCharCode with any provided encodings as of 3/2021.
      out += String.fromCharCode.apply(
        null,
        buf.subarray(i, i + ARRAY_APPLY_BATCH)
      );
    }
    return out;
  });

  /**
   * Possible parameter for addImage, an RGBA buffer with size.
   *
   * @typedef {Object} RGBAData
   * @property {Uint8ClampedArray} data - Single dimensional array of RGBA values. For example from canvas getImageData.
   * @property {number} width - Image width as the data does not carry this information in itself.
   * @property {number} height - Image height as the data does not carry this information in itself.
   */

  /**
   * Adds an Image to the PDF.
   *
   * @name addImage
   * @public
   * @function
   * @param {string|HTMLImageElement|HTMLCanvasElement|Uint8Array|RGBAData} imageData imageData as base64 encoded DataUrl or Image-HTMLElement or Canvas-HTMLElement or object containing RGBA array (like output from canvas.getImageData).
   * @param {string} format format of file if filetype-recognition fails or in case of a Canvas-Element needs to be specified (default for Canvas is JPEG), e.g. 'JPEG', 'PNG', 'WEBP'
   * @param {number} x x Coordinate (in units declared at inception of PDF document) against left edge of the page
   * @param {number} y y Coordinate (in units declared at inception of PDF document) against upper edge of the page
   * @param {number} width width of the image (in units declared at inception of PDF document)
   * @param {number} height height of the Image (in units declared at inception of PDF document)
   * @param {string} alias alias of the image (if used multiple times)
   * @param {string} compression compression of the generated JPEG, can have the values 'NONE', 'FAST', 'MEDIUM' and 'SLOW'
   * @param {number} rotation rotation of the image in degrees (0-359)
   *
   * @returns jsPDF
   */
  jsPDFAPI.addImage = function() {
    var imageData, format, x, y, w, h, alias, compression, rotation;

    imageData = arguments[0];
    if (typeof arguments[1] === "number") {
      format = UNKNOWN;
      x = arguments[1];
      y = arguments[2];
      w = arguments[3];
      h = arguments[4];
      alias = arguments[5];
      compression = arguments[6];
      rotation = arguments[7];
    } else {
      format = arguments[1];
      x = arguments[2];
      y = arguments[3];
      w = arguments[4];
      h = arguments[5];
      alias = arguments[6];
      compression = arguments[7];
      rotation = arguments[8];
    }

    if (
      typeof imageData === "object" &&
      !isDOMElement(imageData) &&
      "imageData" in imageData
    ) {
      var options = imageData;

      imageData = options.imageData;
      format = options.format || format || UNKNOWN;
      x = options.x || x || 0;
      y = options.y || y || 0;
      w = options.w || options.width || w;
      h = options.h || options.height || h;
      alias = options.alias || alias;
      compression = options.compression || compression;
      rotation = options.rotation || options.angle || rotation;
    }

    //If compression is not explicitly set, determine if we should use compression
    var filter = this.internal.getFilters();
    if (compression === undefined && filter.indexOf("FlateEncode") !== -1) {
      compression = "SLOW";
    }

    if (isNaN(x) || isNaN(y)) {
      throw new Error("Invalid coordinates passed to jsPDF.addImage");
    }

    initialize.call(this);

    var image = processImageData.call(
      this,
      imageData,
      format,
      alias,
      compression
    );

    writeImageToPDF.call(this, x, y, w, h, image, rotation);

    return this;
  };

  var processImageData = function(imageData, format, alias, compression) {
    var result, dataAsBinaryString;

    if (
      typeof imageData === "string" &&
      getImageFileTypeByImageData(imageData) === UNKNOWN
    ) {
      imageData = unescape(imageData);
      var tmpImageData = convertBase64ToBinaryString(imageData, false);

      if (tmpImageData !== "") {
        imageData = tmpImageData;
      } else {
        tmpImageData = jsPDFAPI.loadFile(imageData, true);
        if (tmpImageData !== undefined) {
          imageData = tmpImageData;
        }
      }
    }

    if (isDOMElement(imageData)) {
      imageData = getImageDataFromElement(imageData, format);
    }

    format = getImageFileTypeByImageData(imageData, format);
    if (!isImageTypeSupported(format)) {
      throw new Error(
        "addImage does not support files of type '" +
          format +
          "', please ensure that a plugin for '" +
          format +
          "' support is added."
      );
    }

    // now do the heavy lifting

    if (notDefined(alias)) {
      alias = generateAliasFromImageData(imageData);
    }
    result = checkImagesForAlias.call(this, alias);

    if (!result) {
      // no need to convert if imageData is already uint8array
      if (!(imageData instanceof Uint8Array) && format !== "RGBA") {
        dataAsBinaryString = imageData;
        imageData = binaryStringToUint8Array(imageData);
      }

      result = this["process" + format.toUpperCase()](
        imageData,
        getImageIndex.call(this),
        alias,
        checkCompressValue(compression),
        dataAsBinaryString
      );
    }

    if (!result) {
      throw new Error("An unknown error occurred whilst processing the image.");
    }
    return result;
  };

  /**
   * @name convertBase64ToBinaryString
   * @function
   * @param {string} stringData
   * @returns {string} binary string
   */
  var convertBase64ToBinaryString = (jsPDFAPI.__addimage__.convertBase64ToBinaryString = function(
    stringData,
    throwError
  ) {
    throwError = typeof throwError === "boolean" ? throwError : true;
    var imageData = "";
    var rawData;

    if (typeof stringData === "string") {
      rawData = extractImageFromDataUrl(stringData) ?? stringData;

      try {
        imageData = atob(rawData);
      } catch (e) {
        if (throwError) {
          if (!validateStringAsBase64(rawData)) {
            throw new Error(
              "Supplied Data is not a valid base64-String jsPDF.convertBase64ToBinaryString "
            );
          } else {
            throw new Error(
              "atob-Error in jsPDF.convertBase64ToBinaryString " + e.message
            );
          }
        }
      }
    }
    return imageData;
  });

  /**
   * @name getImageProperties
   * @function
   * @param {Object} imageData
   * @returns {Object}
   */
  jsPDFAPI.getImageProperties = function(imageData) {
    var image;
    var tmpImageData = "";
    var format;

    if (isDOMElement(imageData)) {
      imageData = getImageDataFromElement(imageData);
    }

    if (
      typeof imageData === "string" &&
      getImageFileTypeByImageData(imageData) === UNKNOWN
    ) {
      tmpImageData = convertBase64ToBinaryString(imageData, false);

      if (tmpImageData === "") {
        tmpImageData = jsPDFAPI.loadFile(imageData) || "";
      }
      imageData = tmpImageData;
    }

    format = getImageFileTypeByImageData(imageData);
    if (!isImageTypeSupported(format)) {
      throw new Error(
        "addImage does not support files of type '" +
          format +
          "', please ensure that a plugin for '" +
          format +
          "' support is added."
      );
    }

    if (!(imageData instanceof Uint8Array)) {
      imageData = binaryStringToUint8Array(imageData);
    }

    image = this["process" + format.toUpperCase()](imageData);

    if (!image) {
      throw new Error("An unknown error occurred whilst processing the image");
    }

    image.fileType = format;

    return image;
  };
})(jsPDF.API);

/**
 * @license
 * Copyright (c) 2014 Steven Spungin (TwelveTone LLC)  steven@twelvetone.tv
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

(function(jsPDFAPI) {

  var notEmpty = function(obj) {
    if (typeof obj != "undefined") {
      if (obj != "") {
        return true;
      }
    }
  };

  jsPDF.API.events.push([
    "addPage",
    function(addPageData) {
      var pageInfo = this.internal.getPageInfo(addPageData.pageNumber);
      pageInfo.pageContext.annotations = [];
    }
  ]);

  jsPDFAPI.events.push([
    "putPage",
    function(putPageData) {
      var getHorizontalCoordinateString = this.internal.getCoordinateString;
      var getVerticalCoordinateString = this.internal
        .getVerticalCoordinateString;
      var pageInfo = this.internal.getPageInfoByObjId(putPageData.objId);
      var pageAnnos = putPageData.pageContext.annotations;

      var anno, rect, line;
      var found = false;
      for (var a = 0; a < pageAnnos.length && !found; a++) {
        anno = pageAnnos[a];
        switch (anno.type) {
          case "link":
            if (
              notEmpty(anno.options.url) ||
              notEmpty(anno.options.pageNumber)
            ) {
              found = true;
            }
            break;
          case "reference":
          case "text":
          case "freetext":
            found = true;
            break;
        }
      }
      if (found == false) {
        return;
      }

      this.internal.write("/Annots [");
      for (var i = 0; i < pageAnnos.length; i++) {
        anno = pageAnnos[i];
        var escape = this.internal.pdfEscape;
        var encryptor = this.internal.getEncryptor(putPageData.objId);

        switch (anno.type) {
          case "reference":
            // References to Widget Annotations (for AcroForm Fields)
            this.internal.write(" " + anno.object.objId + " 0 R ");
            break;
          case "text":
            // Create a an object for both the text and the popup
            var objText = this.internal.newAdditionalObject();
            var objPopup = this.internal.newAdditionalObject();
            var encryptorText = this.internal.getEncryptor(objText.objId);

            var title = anno.title || "Note";
            rect =
              "/Rect [" +
              getHorizontalCoordinateString(anno.bounds.x) +
              " " +
              getVerticalCoordinateString(anno.bounds.y + anno.bounds.h) +
              " " +
              getHorizontalCoordinateString(anno.bounds.x + anno.bounds.w) +
              " " +
              getVerticalCoordinateString(anno.bounds.y) +
              "] ";

            line =
              "<</Type /Annot /Subtype /" +
              "Text" +
              " " +
              rect +
              "/Contents (" +
              escape(encryptorText(anno.contents)) +
              ")";
            line += " /Popup " + objPopup.objId + " 0 R";
            line += " /P " + pageInfo.objId + " 0 R";
            line += " /T (" + escape(encryptorText(title)) + ") >>";
            objText.content = line;

            var parent = objText.objId + " 0 R";
            var popoff = 30;
            rect =
              "/Rect [" +
              getHorizontalCoordinateString(anno.bounds.x + popoff) +
              " " +
              getVerticalCoordinateString(anno.bounds.y + anno.bounds.h) +
              " " +
              getHorizontalCoordinateString(
                anno.bounds.x + anno.bounds.w + popoff
              ) +
              " " +
              getVerticalCoordinateString(anno.bounds.y) +
              "] ";
            line =
              "<</Type /Annot /Subtype /" +
              "Popup" +
              " " +
              rect +
              " /Parent " +
              parent;
            if (anno.open) {
              line += " /Open true";
            }
            line += " >>";
            objPopup.content = line;

            this.internal.write(objText.objId, "0 R", objPopup.objId, "0 R");

            break;
          case "freetext":
            rect =
              "/Rect [" +
              getHorizontalCoordinateString(anno.bounds.x) +
              " " +
              getVerticalCoordinateString(anno.bounds.y) +
              " " +
              getHorizontalCoordinateString(anno.bounds.x + anno.bounds.w) +
              " " +
              getVerticalCoordinateString(anno.bounds.y + anno.bounds.h) +
              "] ";
            var color = anno.color || "#000000";
            line =
              "<</Type /Annot /Subtype /" +
              "FreeText" +
              " " +
              rect +
              "/Contents (" +
              escape(encryptor(anno.contents)) +
              ")";
            line +=
              " /DS(font: Helvetica,sans-serif 12.0pt; text-align:left; color:#" +
              color +
              ")";
            line += " /Border [0 0 0]";
            line += " >>";
            this.internal.write(line);
            break;
          case "link":
            if (anno.options.name) {
              var loc = this.annotations._nameMap[anno.options.name];
              anno.options.pageNumber = loc.page;
              anno.options.top = loc.y;
            } else {
              if (!anno.options.top) {
                anno.options.top = 0;
              }
            }

            rect =
              "/Rect [" +
              anno.finalBounds.x +
              " " +
              anno.finalBounds.y +
              " " +
              anno.finalBounds.w +
              " " +
              anno.finalBounds.h +
              "] ";

            line = "";
            if (anno.options.url) {
              line =
                "<</Type /Annot /Subtype /Link " +
                rect +
                "/Border [0 0 0] /A <</S /URI /URI (" +
                escape(encryptor(anno.options.url)) +
                ") >>";
            } else if (anno.options.pageNumber) {
              // first page is 0
              var info = this.internal.getPageInfo(anno.options.pageNumber);
              line =
                "<</Type /Annot /Subtype /Link " +
                rect +
                "/Border [0 0 0] /Dest [" +
                info.objId +
                " 0 R";
              anno.options.magFactor = anno.options.magFactor || "XYZ";
              switch (anno.options.magFactor) {
                case "Fit":
                  line += " /Fit]";
                  break;
                case "FitH":
                  line += " /FitH " + anno.options.top + "]";
                  break;
                case "FitV":
                  anno.options.left = anno.options.left || 0;
                  line += " /FitV " + anno.options.left + "]";
                  break;
                case "XYZ":
                default:
                  var top = getVerticalCoordinateString(anno.options.top);
                  anno.options.left = anno.options.left || 0;
                  // 0 or null zoom will not change zoom factor
                  if (typeof anno.options.zoom === "undefined") {
                    anno.options.zoom = 0;
                  }
                  line +=
                    " /XYZ " +
                    anno.options.left +
                    " " +
                    top +
                    " " +
                    anno.options.zoom +
                    "]";
                  break;
              }
            }

            if (line != "") {
              line += " >>";
              this.internal.write(line);
            }
            break;
        }
      }
      this.internal.write("]");
    }
  ]);

  /**
   * @name createAnnotation
   * @function
   * @param {Object} options
   */
  jsPDFAPI.createAnnotation = function(options) {
    var pageInfo = this.internal.getCurrentPageInfo();
    switch (options.type) {
      case "link":
        this.link(
          options.bounds.x,
          options.bounds.y,
          options.bounds.w,
          options.bounds.h,
          options
        );
        break;
      case "text":
      case "freetext":
        pageInfo.pageContext.annotations.push(options);
        break;
    }
  };

  /**
   * Create a link
   *
   * valid options
   * <li> pageNumber or url [required]
   * <p>If pageNumber is specified, top and zoom may also be specified</p>
   * @name link
   * @function
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {Object} options
   */
  jsPDFAPI.link = function(x, y, w, h, options) {
    var pageInfo = this.internal.getCurrentPageInfo();
    var getHorizontalCoordinateString = this.internal.getCoordinateString;
    var getVerticalCoordinateString = this.internal.getVerticalCoordinateString;

    pageInfo.pageContext.annotations.push({
      finalBounds: {
        x: getHorizontalCoordinateString(x),
        y: getVerticalCoordinateString(y),
        w: getHorizontalCoordinateString(x + w),
        h: getVerticalCoordinateString(y + h)
      },
      options: options,
      type: "link"
    });
  };

  /**
   * Currently only supports single line text.
   * Returns the width of the text/link
   *
   * @name textWithLink
   * @function
   * @param {string} text
   * @param {number} x
   * @param {number} y
   * @param {Object} options
   * @returns {number} width the width of the text/link
   */
  jsPDFAPI.textWithLink = function(text, x, y, options) {
    var totalLineWidth = this.getTextWidth(text);
    var lineHeight = this.internal.getLineHeight() / this.internal.scaleFactor;
    var linkHeight, linkWidth;

    // Checking if maxWidth option is passed to determine lineWidth and number of lines for each line
    if (options.maxWidth !== undefined) {
      var { maxWidth } = options;
      linkWidth = maxWidth;
      var numOfLines = this.splitTextToSize(text, linkWidth).length;
      linkHeight = Math.ceil(lineHeight * numOfLines);
    } else {
      linkWidth = totalLineWidth;
      linkHeight = lineHeight;
    }

    this.text(text, x, y, options);

    //TODO We really need the text baseline height to do this correctly.
    // Or ability to draw text on top, bottom, center, or baseline.
    y += lineHeight * 0.2;
    //handle x position based on the align option
    if (options.align === "center") {
      x = x - totalLineWidth / 2; //since starting from center move the x position by half of text width
    }
    if (options.align === "right") {
      x = x - totalLineWidth;
    }
    this.link(x, y - lineHeight, linkWidth, linkHeight, options);
    return totalLineWidth;
  };

  //TODO move into external library
  /**
   * @name getTextWidth
   * @function
   * @param {string} text
   * @returns {number} txtWidth
   */
  jsPDFAPI.getTextWidth = function(text) {
    var fontSize = this.internal.getFontSize();
    var txtWidth =
      (this.getStringUnitWidth(text) * fontSize) / this.internal.scaleFactor;
    return txtWidth;
  };

  return this;
})(jsPDF.API);

/**
 * @license
 * Copyright (c) 2017 Aras Abbasi
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * jsPDF arabic parser PlugIn
 *
 * @name arabic
 * @module
 */
(function(jsPDFAPI) {

  /**
   * Arabic shape substitutions: char code => (isolated, final, initial, medial).
   * Arabic Substition A
   */
  var arabicSubstitionA = {
    0x0621: [0xfe80], // ARABIC LETTER HAMZA
    0x0622: [0xfe81, 0xfe82], // ARABIC LETTER ALEF WITH MADDA ABOVE
    0x0623: [0xfe83, 0xfe84], // ARABIC LETTER ALEF WITH HAMZA ABOVE
    0x0624: [0xfe85, 0xfe86], // ARABIC LETTER WAW WITH HAMZA ABOVE
    0x0625: [0xfe87, 0xfe88], // ARABIC LETTER ALEF WITH HAMZA BELOW
    0x0626: [0xfe89, 0xfe8a, 0xfe8b, 0xfe8c], // ARABIC LETTER YEH WITH HAMZA ABOVE
    0x0627: [0xfe8d, 0xfe8e], // ARABIC LETTER ALEF
    0x0628: [0xfe8f, 0xfe90, 0xfe91, 0xfe92], // ARABIC LETTER BEH
    0x0629: [0xfe93, 0xfe94], // ARABIC LETTER TEH MARBUTA
    0x062a: [0xfe95, 0xfe96, 0xfe97, 0xfe98], // ARABIC LETTER TEH
    0x062b: [0xfe99, 0xfe9a, 0xfe9b, 0xfe9c], // ARABIC LETTER THEH
    0x062c: [0xfe9d, 0xfe9e, 0xfe9f, 0xfea0], // ARABIC LETTER JEEM
    0x062d: [0xfea1, 0xfea2, 0xfea3, 0xfea4], // ARABIC LETTER HAH
    0x062e: [0xfea5, 0xfea6, 0xfea7, 0xfea8], // ARABIC LETTER KHAH
    0x062f: [0xfea9, 0xfeaa], // ARABIC LETTER DAL
    0x0630: [0xfeab, 0xfeac], // ARABIC LETTER THAL
    0x0631: [0xfead, 0xfeae], // ARABIC LETTER REH
    0x0632: [0xfeaf, 0xfeb0], // ARABIC LETTER ZAIN
    0x0633: [0xfeb1, 0xfeb2, 0xfeb3, 0xfeb4], // ARABIC LETTER SEEN
    0x0634: [0xfeb5, 0xfeb6, 0xfeb7, 0xfeb8], // ARABIC LETTER SHEEN
    0x0635: [0xfeb9, 0xfeba, 0xfebb, 0xfebc], // ARABIC LETTER SAD
    0x0636: [0xfebd, 0xfebe, 0xfebf, 0xfec0], // ARABIC LETTER DAD
    0x0637: [0xfec1, 0xfec2, 0xfec3, 0xfec4], // ARABIC LETTER TAH
    0x0638: [0xfec5, 0xfec6, 0xfec7, 0xfec8], // ARABIC LETTER ZAH
    0x0639: [0xfec9, 0xfeca, 0xfecb, 0xfecc], // ARABIC LETTER AIN
    0x063a: [0xfecd, 0xfece, 0xfecf, 0xfed0], // ARABIC LETTER GHAIN
    0x0641: [0xfed1, 0xfed2, 0xfed3, 0xfed4], // ARABIC LETTER FEH
    0x0642: [0xfed5, 0xfed6, 0xfed7, 0xfed8], // ARABIC LETTER QAF
    0x0643: [0xfed9, 0xfeda, 0xfedb, 0xfedc], // ARABIC LETTER KAF
    0x0644: [0xfedd, 0xfede, 0xfedf, 0xfee0], // ARABIC LETTER LAM
    0x0645: [0xfee1, 0xfee2, 0xfee3, 0xfee4], // ARABIC LETTER MEEM
    0x0646: [0xfee5, 0xfee6, 0xfee7, 0xfee8], // ARABIC LETTER NOON
    0x0647: [0xfee9, 0xfeea, 0xfeeb, 0xfeec], // ARABIC LETTER HEH
    0x0648: [0xfeed, 0xfeee], // ARABIC LETTER WAW
    0x0649: [0xfeef, 0xfef0, 64488, 64489], // ARABIC LETTER ALEF MAKSURA
    0x064a: [0xfef1, 0xfef2, 0xfef3, 0xfef4], // ARABIC LETTER YEH
    0x0671: [0xfb50, 0xfb51], // ARABIC LETTER ALEF WASLA
    0x0677: [0xfbdd], // ARABIC LETTER U WITH HAMZA ABOVE
    0x0679: [0xfb66, 0xfb67, 0xfb68, 0xfb69], // ARABIC LETTER TTEH
    0x067a: [0xfb5e, 0xfb5f, 0xfb60, 0xfb61], // ARABIC LETTER TTEHEH
    0x067b: [0xfb52, 0xfb53, 0xfb54, 0xfb55], // ARABIC LETTER BEEH
    0x067e: [0xfb56, 0xfb57, 0xfb58, 0xfb59], // ARABIC LETTER PEH
    0x067f: [0xfb62, 0xfb63, 0xfb64, 0xfb65], // ARABIC LETTER TEHEH
    0x0680: [0xfb5a, 0xfb5b, 0xfb5c, 0xfb5d], // ARABIC LETTER BEHEH
    0x0683: [0xfb76, 0xfb77, 0xfb78, 0xfb79], // ARABIC LETTER NYEH
    0x0684: [0xfb72, 0xfb73, 0xfb74, 0xfb75], // ARABIC LETTER DYEH
    0x0686: [0xfb7a, 0xfb7b, 0xfb7c, 0xfb7d], // ARABIC LETTER TCHEH
    0x0687: [0xfb7e, 0xfb7f, 0xfb80, 0xfb81], // ARABIC LETTER TCHEHEH
    0x0688: [0xfb88, 0xfb89], // ARABIC LETTER DDAL
    0x068c: [0xfb84, 0xfb85], // ARABIC LETTER DAHAL
    0x068d: [0xfb82, 0xfb83], // ARABIC LETTER DDAHAL
    0x068e: [0xfb86, 0xfb87], // ARABIC LETTER DUL
    0x0691: [0xfb8c, 0xfb8d], // ARABIC LETTER RREH
    0x0698: [0xfb8a, 0xfb8b], // ARABIC LETTER JEH
    0x06a4: [0xfb6a, 0xfb6b, 0xfb6c, 0xfb6d], // ARABIC LETTER VEH
    0x06a6: [0xfb6e, 0xfb6f, 0xfb70, 0xfb71], // ARABIC LETTER PEHEH
    0x06a9: [0xfb8e, 0xfb8f, 0xfb90, 0xfb91], // ARABIC LETTER KEHEH
    0x06ad: [0xfbd3, 0xfbd4, 0xfbd5, 0xfbd6], // ARABIC LETTER NG
    0x06af: [0xfb92, 0xfb93, 0xfb94, 0xfb95], // ARABIC LETTER GAF
    0x06b1: [0xfb9a, 0xfb9b, 0xfb9c, 0xfb9d], // ARABIC LETTER NGOEH
    0x06b3: [0xfb96, 0xfb97, 0xfb98, 0xfb99], // ARABIC LETTER GUEH
    0x06ba: [0xfb9e, 0xfb9f], // ARABIC LETTER NOON GHUNNA
    0x06bb: [0xfba0, 0xfba1, 0xfba2, 0xfba3], // ARABIC LETTER RNOON
    0x06be: [0xfbaa, 0xfbab, 0xfbac, 0xfbad], // ARABIC LETTER HEH DOACHASHMEE
    0x06c0: [0xfba4, 0xfba5], // ARABIC LETTER HEH WITH YEH ABOVE
    0x06c1: [0xfba6, 0xfba7, 0xfba8, 0xfba9], // ARABIC LETTER HEH GOAL
    0x06c5: [0xfbe0, 0xfbe1], // ARABIC LETTER KIRGHIZ OE
    0x06c6: [0xfbd9, 0xfbda], // ARABIC LETTER OE
    0x06c7: [0xfbd7, 0xfbd8], // ARABIC LETTER U
    0x06c8: [0xfbdb, 0xfbdc], // ARABIC LETTER YU
    0x06c9: [0xfbe2, 0xfbe3], // ARABIC LETTER KIRGHIZ YU
    0x06cb: [0xfbde, 0xfbdf], // ARABIC LETTER VE
    0x06cc: [0xfbfc, 0xfbfd, 0xfbfe, 0xfbff], // ARABIC LETTER FARSI YEH
    0x06d0: [0xfbe4, 0xfbe5, 0xfbe6, 0xfbe7], //ARABIC LETTER E
    0x06d2: [0xfbae, 0xfbaf], // ARABIC LETTER YEH BARREE
    0x06d3: [0xfbb0, 0xfbb1] // ARABIC LETTER YEH BARREE WITH HAMZA ABOVE
  };

  /*
    var ligaturesSubstitutionA = {
        0xFBEA: []// ARABIC LIGATURE YEH WITH HAMZA ABOVE WITH ALEF ISOLATED FORM
    };
    */

  var ligatures = {
    0xfedf: {
      0xfe82: 0xfef5, // ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE ISOLATED FORM
      0xfe84: 0xfef7, // ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE ISOLATED FORM
      0xfe88: 0xfef9, // ARABIC LIGATURE LAM WITH ALEF WITH HAMZA BELOW ISOLATED FORM
      0xfe8e: 0xfefb // ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
    },
    0xfee0: {
      0xfe82: 0xfef6, // ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE FINAL FORM
      0xfe84: 0xfef8, // ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE FINAL FORM
      0xfe88: 0xfefa, // ARABIC LIGATURE LAM WITH ALEF WITH HAMZA BELOW FINAL FORM
      0xfe8e: 0xfefc // ARABIC LIGATURE LAM WITH ALEF FINAL FORM
    },
    0xfe8d: { 0xfedf: { 0xfee0: { 0xfeea: 0xfdf2 } } }, // ALLAH
    0x0651: {
      0x064c: 0xfc5e, // Shadda + Dammatan
      0x064d: 0xfc5f, // Shadda + Kasratan
      0x064e: 0xfc60, // Shadda + Fatha
      0x064f: 0xfc61, // Shadda + Damma
      0x0650: 0xfc62 // Shadda + Kasra
    }
  };

  var arabic_diacritics = {
    1612: 64606, // Shadda + Dammatan
    1613: 64607, // Shadda + Kasratan
    1614: 64608, // Shadda + Fatha
    1615: 64609, // Shadda + Damma
    1616: 64610 // Shadda + Kasra
  };

  var alfletter = [1570, 1571, 1573, 1575];

  var noChangeInForm = -1;
  var isolatedForm = 0;
  var finalForm = 1;
  var initialForm = 2;
  var medialForm = 3;

  jsPDFAPI.__arabicParser__ = {};

  //private
  var isInArabicSubstitutionA = (jsPDFAPI.__arabicParser__.isInArabicSubstitutionA = function(
    letter
  ) {
    return typeof arabicSubstitionA[letter.charCodeAt(0)] !== "undefined";
  });

  var isArabicLetter = (jsPDFAPI.__arabicParser__.isArabicLetter = function(
    letter
  ) {
    return (
      typeof letter === "string" &&
      /^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+$/.test(
        letter
      )
    );
  });

  var isArabicEndLetter = (jsPDFAPI.__arabicParser__.isArabicEndLetter = function(
    letter
  ) {
    return (
      isArabicLetter(letter) &&
      isInArabicSubstitutionA(letter) &&
      arabicSubstitionA[letter.charCodeAt(0)].length <= 2
    );
  });

  var isArabicAlfLetter = (jsPDFAPI.__arabicParser__.isArabicAlfLetter = function(
    letter
  ) {
    return (
      isArabicLetter(letter) && alfletter.indexOf(letter.charCodeAt(0)) >= 0
    );
  });

  jsPDFAPI.__arabicParser__.arabicLetterHasIsolatedForm = function(letter) {
    return (
      isArabicLetter(letter) &&
      isInArabicSubstitutionA(letter) &&
      arabicSubstitionA[letter.charCodeAt(0)].length >= 1
    );
  };

  var arabicLetterHasFinalForm = (jsPDFAPI.__arabicParser__.arabicLetterHasFinalForm = function(
    letter
  ) {
    return (
      isArabicLetter(letter) &&
      isInArabicSubstitutionA(letter) &&
      arabicSubstitionA[letter.charCodeAt(0)].length >= 2
    );
  });

  jsPDFAPI.__arabicParser__.arabicLetterHasInitialForm = function(letter) {
    return (
      isArabicLetter(letter) &&
      isInArabicSubstitutionA(letter) &&
      arabicSubstitionA[letter.charCodeAt(0)].length >= 3
    );
  };

  var arabicLetterHasMedialForm = (jsPDFAPI.__arabicParser__.arabicLetterHasMedialForm = function(
    letter
  ) {
    return (
      isArabicLetter(letter) &&
      isInArabicSubstitutionA(letter) &&
      arabicSubstitionA[letter.charCodeAt(0)].length == 4
    );
  });

  var resolveLigatures = (jsPDFAPI.__arabicParser__.resolveLigatures = function(
    letters
  ) {
    var i = 0;
    var tmpLigatures = ligatures;
    var result = "";
    var effectedLetters = 0;

    for (i = 0; i < letters.length; i += 1) {
      if (typeof tmpLigatures[letters.charCodeAt(i)] !== "undefined") {
        effectedLetters++;
        tmpLigatures = tmpLigatures[letters.charCodeAt(i)];

        if (typeof tmpLigatures === "number") {
          result += String.fromCharCode(tmpLigatures);
          tmpLigatures = ligatures;
          effectedLetters = 0;
        }
        if (i === letters.length - 1) {
          tmpLigatures = ligatures;
          result += letters.charAt(i - (effectedLetters - 1));
          i = i - (effectedLetters - 1);
          effectedLetters = 0;
        }
      } else {
        tmpLigatures = ligatures;
        result += letters.charAt(i - effectedLetters);
        i = i - effectedLetters;
        effectedLetters = 0;
      }
    }

    return result;
  });

  jsPDFAPI.__arabicParser__.isArabicDiacritic = function(letter) {
    return (
      letter !== undefined &&
      arabic_diacritics[letter.charCodeAt(0)] !== undefined
    );
  };

  var getCorrectForm = (jsPDFAPI.__arabicParser__.getCorrectForm = function(
    currentChar,
    beforeChar,
    nextChar
  ) {
    if (!isArabicLetter(currentChar)) {
      return -1;
    }

    if (isInArabicSubstitutionA(currentChar) === false) {
      return noChangeInForm;
    }
    if (
      !arabicLetterHasFinalForm(currentChar) ||
      (!isArabicLetter(beforeChar) && !isArabicLetter(nextChar)) ||
      (!isArabicLetter(nextChar) && isArabicEndLetter(beforeChar)) ||
      (isArabicEndLetter(currentChar) && !isArabicLetter(beforeChar)) ||
      (isArabicEndLetter(currentChar) && isArabicAlfLetter(beforeChar)) ||
      (isArabicEndLetter(currentChar) && isArabicEndLetter(beforeChar))
    ) {
      return isolatedForm;
    }

    if (
      arabicLetterHasMedialForm(currentChar) &&
      isArabicLetter(beforeChar) &&
      !isArabicEndLetter(beforeChar) &&
      isArabicLetter(nextChar) &&
      arabicLetterHasFinalForm(nextChar)
    ) {
      return medialForm;
    }

    if (isArabicEndLetter(currentChar) || !isArabicLetter(nextChar)) {
      return finalForm;
    }
    return initialForm;
  });

  /**
   * @name processArabic
   * @function
   * @param {string} text
   * @returns {string}
   */
  var parseArabic = function(text) {
    text = text || "";

    var result = "";
    var i = 0;
    var j = 0;
    var position = 0;
    var currentLetter = "";
    var prevLetter = "";
    var nextLetter = "";

    var words = text.split("\\s+");
    var newWords = [];
    for (i = 0; i < words.length; i += 1) {
      newWords.push("");
      for (j = 0; j < words[i].length; j += 1) {
        currentLetter = words[i][j];
        prevLetter = words[i][j - 1];
        nextLetter = words[i][j + 1];
        if (isArabicLetter(currentLetter)) {
          position = getCorrectForm(currentLetter, prevLetter, nextLetter);
          if (position !== -1) {
            newWords[i] += String.fromCharCode(
              arabicSubstitionA[currentLetter.charCodeAt(0)][position]
            );
          } else {
            newWords[i] += currentLetter;
          }
        } else {
          newWords[i] += currentLetter;
        }
      }

      newWords[i] = resolveLigatures(newWords[i]);
    }
    result = newWords.join(" ");

    return result;
  };

  var processArabic = (jsPDFAPI.__arabicParser__.processArabic = jsPDFAPI.processArabic = function() {
    var text =
      typeof arguments[0] === "string" ? arguments[0] : arguments[0].text;
    var tmpText = [];
    var result;

    if (Array.isArray(text)) {
      var i = 0;
      tmpText = [];
      for (i = 0; i < text.length; i += 1) {
        if (Array.isArray(text[i])) {
          tmpText.push([parseArabic(text[i][0]), text[i][1], text[i][2]]);
        } else {
          tmpText.push([parseArabic(text[i])]);
        }
      }
      result = tmpText;
    } else {
      result = parseArabic(text);
    }
    if (typeof arguments[0] === "string") {
      return result;
    } else {
      arguments[0].text = result;
      return arguments[0];
    }
  });

  jsPDFAPI.events.push(["preProcessText", processArabic]);
})(jsPDF.API);

/** @license
 * jsPDF Autoprint Plugin
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * @name autoprint
 * @module
 */
(function(jsPDFAPI) {

  /**
   * Makes the PDF automatically open the print-Dialog when opened in a PDF-viewer.
   *
   * @name autoPrint
   * @function
   * @param {Object} options (optional) Set the attribute variant to 'non-conform' (default) or 'javascript' to activate different methods of automatic printing when opening in a PDF-viewer .
   * @returns {jsPDF}
   * @example
   * var doc = new jsPDF();
   * doc.text(10, 10, 'This is a test');
   * doc.autoPrint({variant: 'non-conform'});
   * doc.save('autoprint.pdf');
   */
  jsPDFAPI.autoPrint = function(options) {
    var refAutoPrintTag;
    options = options || {};
    options.variant = options.variant || "non-conform";

    switch (options.variant) {
      case "javascript":
        //https://github.com/Rob--W/pdf.js/commit/c676ecb5a0f54677b9f3340c3ef2cf42225453bb
        this.addJS("print({});");
        break;
      case "non-conform":
      default:
        this.internal.events.subscribe("postPutResources", function() {
          refAutoPrintTag = this.internal.newObject();
          this.internal.out("<<");
          this.internal.out("/S /Named");
          this.internal.out("/Type /Action");
          this.internal.out("/N /Print");
          this.internal.out(">>");
          this.internal.out("endobj");
        });

        this.internal.events.subscribe("putCatalog", function() {
          this.internal.out("/OpenAction " + refAutoPrintTag + " 0 R");
        });
        break;
    }
    return this;
  };
})(jsPDF.API);

/**
 * @license
 * Copyright (c) 2014 Steven Spungin (TwelveTone LLC)  steven@twelvetone.tv
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * jsPDF Canvas PlugIn
 * This plugin mimics the HTML5 Canvas
 *
 * The goal is to provide a way for current canvas users to print directly to a PDF.
 * @name canvas
 * @module
 */
(function(jsPDFAPI) {

  /**
   * @class Canvas
   * @classdesc A Canvas Wrapper for jsPDF
   */
  var Canvas = function() {
    var jsPdfInstance = undefined;
    Object.defineProperty(this, "pdf", {
      get: function() {
        return jsPdfInstance;
      },
      set: function(value) {
        jsPdfInstance = value;
      }
    });

    var _width = 150;
    /**
     * The height property is a positive integer reflecting the height HTML attribute of the <canvas> element interpreted in CSS pixels. When the attribute is not specified, or if it is set to an invalid value, like a negative, the default value of 150 is used.
     * This is one of the two properties, the other being width, that controls the size of the canvas.
     *
     * @name width
     */
    Object.defineProperty(this, "width", {
      get: function() {
        return _width;
      },
      set: function(value) {
        if (isNaN(value) || Number.isInteger(value) === false || value < 0) {
          _width = 150;
        } else {
          _width = value;
        }
        if (this.getContext("2d").pageWrapXEnabled) {
          this.getContext("2d").pageWrapX = _width + 1;
        }
      }
    });

    var _height = 300;
    /**
     * The width property is a positive integer reflecting the width HTML attribute of the <canvas> element interpreted in CSS pixels. When the attribute is not specified, or if it is set to an invalid value, like a negative, the default value of 300 is used.
     * This is one of the two properties, the other being height, that controls the size of the canvas.
     *
     * @name height
     */
    Object.defineProperty(this, "height", {
      get: function() {
        return _height;
      },
      set: function(value) {
        if (isNaN(value) || Number.isInteger(value) === false || value < 0) {
          _height = 300;
        } else {
          _height = value;
        }
        if (this.getContext("2d").pageWrapYEnabled) {
          this.getContext("2d").pageWrapY = _height + 1;
        }
      }
    });

    var _childNodes = [];
    Object.defineProperty(this, "childNodes", {
      get: function() {
        return _childNodes;
      },
      set: function(value) {
        _childNodes = value;
      }
    });

    var _style = {};
    Object.defineProperty(this, "style", {
      get: function() {
        return _style;
      },
      set: function(value) {
        _style = value;
      }
    });

    Object.defineProperty(this, "parentNode", {});
  };

  /**
   * The getContext() method returns a drawing context on the canvas, or null if the context identifier is not supported.
   *
   * @name getContext
   * @function
   * @param {string} contextType Is a String containing the context identifier defining the drawing context associated to the canvas. Possible value is "2d", leading to the creation of a Context2D object representing a two-dimensional rendering context.
   * @param {object} contextAttributes
   */
  Canvas.prototype.getContext = function(contextType, contextAttributes) {
    contextType = contextType || "2d";
    var key;

    if (contextType !== "2d") {
      return null;
    }
    for (key in contextAttributes) {
      if (this.pdf.context2d.hasOwnProperty(key)) {
        this.pdf.context2d[key] = contextAttributes[key];
      }
    }
    this.pdf.context2d._canvas = this;
    return this.pdf.context2d;
  };

  /**
   * The toDataURL() method is just a stub to throw an error if accidently called.
   *
   * @name toDataURL
   * @function
   */
  Canvas.prototype.toDataURL = function() {
    throw new Error("toDataURL is not implemented.");
  };

  jsPDFAPI.events.push([
    "initialized",
    function() {
      this.canvas = new Canvas();
      this.canvas.pdf = this;
    }
  ]);

  return this;
})(jsPDF.API);

/**
 * @license
 * ====================================================================
 * Copyright (c) 2013 Youssef Beddad, youssef.beddad@gmail.com
 *               2013 Eduardo Menezes de Morais, eduardo.morais@usp.br
 *               2013 Lee Driscoll, https://github.com/lsdriscoll
 *               2014 Juan Pablo Gaviria, https://github.com/juanpgaviria
 *               2014 James Hall, james@parall.ax
 *               2014 Diego Casorran, https://github.com/diegocr
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * @name cell
 * @module
 */
(function(jsPDFAPI) {

  var NO_MARGINS = { left: 0, top: 0, bottom: 0, right: 0 };

  var px2pt = (0.264583 * 72) / 25.4;
  var printingHeaderRow = false;

  var _initialize = function() {
    if (typeof this.internal.__cell__ === "undefined") {
      this.internal.__cell__ = {};
      this.internal.__cell__.padding = 3;
      this.internal.__cell__.headerFunction = undefined;
      this.internal.__cell__.margins = Object.assign({}, NO_MARGINS);
      this.internal.__cell__.margins.width = this.getPageWidth();
      _reset.call(this);
    }
  };

  var _reset = function() {
    this.internal.__cell__.lastCell = new Cell();
    this.internal.__cell__.pages = 1;
  };

  var Cell = function() {
    var _x = arguments[0];
    Object.defineProperty(this, "x", {
      enumerable: true,
      get: function() {
        return _x;
      },
      set: function(value) {
        _x = value;
      }
    });
    var _y = arguments[1];
    Object.defineProperty(this, "y", {
      enumerable: true,
      get: function() {
        return _y;
      },
      set: function(value) {
        _y = value;
      }
    });
    var _width = arguments[2];
    Object.defineProperty(this, "width", {
      enumerable: true,
      get: function() {
        return _width;
      },
      set: function(value) {
        _width = value;
      }
    });
    var _height = arguments[3];
    Object.defineProperty(this, "height", {
      enumerable: true,
      get: function() {
        return _height;
      },
      set: function(value) {
        _height = value;
      }
    });
    var _text = arguments[4];
    Object.defineProperty(this, "text", {
      enumerable: true,
      get: function() {
        return _text;
      },
      set: function(value) {
        _text = value;
      }
    });
    var _lineNumber = arguments[5];
    Object.defineProperty(this, "lineNumber", {
      enumerable: true,
      get: function() {
        return _lineNumber;
      },
      set: function(value) {
        _lineNumber = value;
      }
    });
    var _align = arguments[6];
    Object.defineProperty(this, "align", {
      enumerable: true,
      get: function() {
        return _align;
      },
      set: function(value) {
        _align = value;
      }
    });

    return this;
  };

  Cell.prototype.clone = function() {
    return new Cell(
      this.x,
      this.y,
      this.width,
      this.height,
      this.text,
      this.lineNumber,
      this.align
    );
  };

  Cell.prototype.toArray = function() {
    return [
      this.x,
      this.y,
      this.width,
      this.height,
      this.text,
      this.lineNumber,
      this.align
    ];
  };

  /**
   * @name setHeaderFunction
   * @function
   * @param {function} func
   */
  jsPDFAPI.setHeaderFunction = function(func) {
    _initialize.call(this);
    this.internal.__cell__.headerFunction =
      typeof func === "function" ? func : undefined;
    return this;
  };

  /**
   * @name getTextDimensions
   * @function
   * @param {string} txt
   * @returns {Object} dimensions
   */
  jsPDFAPI.getTextDimensions = function(text, options) {
    _initialize.call(this);
    options = options || {};
    var fontSize = options.fontSize || this.getFontSize();
    var font = options.font || this.getFont();
    var scaleFactor = options.scaleFactor || this.internal.scaleFactor;
    var width = 0;
    var amountOfLines = 0;
    var height = 0;
    var tempWidth = 0;
    var scope = this;

    if (!Array.isArray(text) && typeof text !== "string") {
      if (typeof text === "number") {
        text = String(text);
      } else {
        throw new Error(
          "getTextDimensions expects text-parameter to be of type String or type Number or an Array of Strings."
        );
      }
    }

    const maxWidth = options.maxWidth;
    if (maxWidth > 0) {
      if (typeof text === "string") {
        text = this.splitTextToSize(text, maxWidth);
      } else if (Object.prototype.toString.call(text) === "[object Array]") {
        text = text.reduce(function(acc, textLine) {
          return acc.concat(scope.splitTextToSize(textLine, maxWidth));
        }, []);
      }
    } else {
      // Without the else clause, it will not work if you do not pass along maxWidth
      text = Array.isArray(text) ? text : [text];
    }

    for (var i = 0; i < text.length; i++) {
      tempWidth = this.getStringUnitWidth(text[i], { font: font }) * fontSize;
      if (width < tempWidth) {
        width = tempWidth;
      }
    }

    if (width !== 0) {
      amountOfLines = text.length;
    }

    width = width / scaleFactor;
    height = Math.max(
      (amountOfLines * fontSize * this.getLineHeightFactor() -
        fontSize * (this.getLineHeightFactor() - 1)) /
        scaleFactor,
      0
    );
    return { w: width, h: height };
  };

  /**
   * @name cellAddPage
   * @function
   */
  jsPDFAPI.cellAddPage = function() {
    _initialize.call(this);

    this.addPage();

    var margins = this.internal.__cell__.margins || NO_MARGINS;
    this.internal.__cell__.lastCell = new Cell(
      margins.left,
      margins.top,
      undefined,
      undefined
    );
    this.internal.__cell__.pages += 1;

    return this;
  };

  /**
   * @name cell
   * @function
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {string} text
   * @param {number} lineNumber lineNumber
   * @param {string} align
   * @return {jsPDF} jsPDF-instance
   */
  var cell = (jsPDFAPI.cell = function() {
    var currentCell;

    if (arguments[0] instanceof Cell) {
      currentCell = arguments[0];
    } else {
      currentCell = new Cell(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5]
      );
    }
    _initialize.call(this);
    var lastCell = this.internal.__cell__.lastCell;
    var padding = this.internal.__cell__.padding;
    var margins = this.internal.__cell__.margins || NO_MARGINS;
    var tableHeaderRow = this.internal.__cell__.tableHeaderRow;
    var printHeaders = this.internal.__cell__.printHeaders;
    // If this is not the first cell, we must change its position
    if (typeof lastCell.lineNumber !== "undefined") {
      if (lastCell.lineNumber === currentCell.lineNumber) {
        //Same line
        currentCell.x = (lastCell.x || 0) + (lastCell.width || 0);
        currentCell.y = lastCell.y || 0;
      } else {
        //New line
        if (
          lastCell.y + lastCell.height + currentCell.height + margins.bottom >
          this.getPageHeight()
        ) {
          this.cellAddPage();
          currentCell.y = margins.top;
          if (printHeaders && tableHeaderRow) {
            this.printHeaderRow(currentCell.lineNumber, true);
            currentCell.y += tableHeaderRow[0].height;
          }
        } else {
          currentCell.y = lastCell.y + lastCell.height || currentCell.y;
        }
      }
    }

    if (typeof currentCell.text[0] !== "undefined") {
      this.rect(
        currentCell.x,
        currentCell.y,
        currentCell.width,
        currentCell.height,
        printingHeaderRow === true ? "FD" : undefined
      );
      if (currentCell.align === "right") {
        this.text(
          currentCell.text,
          currentCell.x + currentCell.width - padding,
          currentCell.y + padding,
          { align: "right", baseline: "top" }
        );
      } else if (currentCell.align === "center") {
        this.text(
          currentCell.text,
          currentCell.x + currentCell.width / 2,
          currentCell.y + padding,
          {
            align: "center",
            baseline: "top",
            maxWidth: currentCell.width - padding - padding
          }
        );
      } else {
        this.text(
          currentCell.text,
          currentCell.x + padding,
          currentCell.y + padding,
          {
            align: "left",
            baseline: "top",
            maxWidth: currentCell.width - padding - padding
          }
        );
      }
    }
    this.internal.__cell__.lastCell = currentCell;
    return this;
  });

  /**
     * Create a table from a set of data.
     * @name table
     * @function
     * @param {Integer} [x] : left-position for top-left corner of table
     * @param {Integer} [y] top-position for top-left corner of table
     * @param {Object[]} [data] An array of objects containing key-value pairs corresponding to a row of data.
     * @param {String[]} [headers] Omit or null to auto-generate headers at a performance cost

     * @param {Object} [config.printHeaders] True to print column headers at the top of every page
     * @param {Object} [config.autoSize] True to dynamically set the column widths to match the widest cell value
     * @param {Object} [config.margins] margin values for left, top, bottom, and width
     * @param {Object} [config.fontSize] Integer fontSize to use (optional)
     * @param {Object} [config.padding] cell-padding in pt to use (optional)
     * @param {Object} [config.headerBackgroundColor] default is #c8c8c8 (optional)
     * @param {Object} [config.headerTextColor] default is #000 (optional)
     * @param {Object} [config.rowStart] callback to handle before print each row (optional)
     * @param {Object} [config.cellStart] callback to handle before print each cell (optional)
     * @returns {jsPDF} jsPDF-instance
     */

  jsPDFAPI.table = function(x, y, data, headers, config) {
    _initialize.call(this);
    if (!data) {
      throw new Error("No data for PDF table.");
    }

    config = config || {};

    var headerNames = [],
      headerLabels = [],
      headerAligns = [],
      i,
      columnMatrix = {},
      columnWidths = {},
      column,
      columnMinWidths = [],
      j,
      tableHeaderConfigs = [],
      //set up defaults. If a value is provided in config, defaults will be overwritten:
      autoSize = config.autoSize || false,
      printHeaders = config.printHeaders === false ? false : true,
      fontSize =
        config.css && typeof config.css["font-size"] !== "undefined"
          ? config.css["font-size"] * 16
          : config.fontSize || 12,
      margins =
        config.margins ||
        Object.assign({ width: this.getPageWidth() }, NO_MARGINS),
      padding = typeof config.padding === "number" ? config.padding : 3,
      headerBackgroundColor = config.headerBackgroundColor || "#c8c8c8",
      headerTextColor = config.headerTextColor || "#000";

    _reset.call(this);

    this.internal.__cell__.printHeaders = printHeaders;
    this.internal.__cell__.margins = margins;
    this.internal.__cell__.table_font_size = fontSize;
    this.internal.__cell__.padding = padding;
    this.internal.__cell__.headerBackgroundColor = headerBackgroundColor;
    this.internal.__cell__.headerTextColor = headerTextColor;
    this.setFontSize(fontSize);

    // Set header values
    if (headers === undefined || headers === null) {
      // No headers defined so we derive from data
      headerNames = Object.keys(data[0]);
      headerLabels = headerNames;
      headerAligns = headerNames.map(function() {
        return "left";
      });
    } else if (Array.isArray(headers) && typeof headers[0] === "object") {
      headerNames = headers.map(function(header) {
        return header.name;
      });
      headerLabels = headers.map(function(header) {
        return header.prompt || header.name || "";
      });
      headerAligns = headers.map(function(header) {
        return header.align || "left";
      });
      // Split header configs into names and prompts
      for (i = 0; i < headers.length; i += 1) {
        columnWidths[headers[i].name] = headers[i].width * px2pt;
      }
    } else if (Array.isArray(headers) && typeof headers[0] === "string") {
      headerNames = headers;
      headerLabels = headerNames;
      headerAligns = headerNames.map(function() {
        return "left";
      });
    }

    if (
      autoSize ||
      (Array.isArray(headers) && typeof headers[0] === "string")
    ) {
      var headerName;
      for (i = 0; i < headerNames.length; i += 1) {
        headerName = headerNames[i];

        // Create a matrix of columns e.g., {column_title: [row1_Record, row2_Record]}

        columnMatrix[headerName] = data.map(function(rec) {
          return rec[headerName];
        });

        // get header width
        this.setFont(undefined, "bold");
        columnMinWidths.push(
          this.getTextDimensions(headerLabels[i], {
            fontSize: this.internal.__cell__.table_font_size,
            scaleFactor: this.internal.scaleFactor
          }).w
        );
        column = columnMatrix[headerName];

        // get cell widths
        this.setFont(undefined, "normal");
        for (j = 0; j < column.length; j += 1) {
          columnMinWidths.push(
            this.getTextDimensions(column[j], {
              fontSize: this.internal.__cell__.table_font_size,
              scaleFactor: this.internal.scaleFactor
            }).w
          );
        }

        // get final column width
        columnWidths[headerName] =
          Math.max.apply(null, columnMinWidths) + padding + padding;

        //have to reset
        columnMinWidths = [];
      }
    }

    // -- Construct the table

    if (printHeaders) {
      var row = {};
      for (i = 0; i < headerNames.length; i += 1) {
        row[headerNames[i]] = {};
        row[headerNames[i]].text = headerLabels[i];
        row[headerNames[i]].align = headerAligns[i];
      }

      var rowHeight = calculateLineHeight.call(this, row, columnWidths);

      // Construct the header row
      tableHeaderConfigs = headerNames.map(function(value) {
        return new Cell(
          x,
          y,
          columnWidths[value],
          rowHeight,
          row[value].text,
          undefined,
          row[value].align
        );
      });

      // Store the table header config
      this.setTableHeaderRow(tableHeaderConfigs);

      // Print the header for the start of the table
      this.printHeaderRow(1, false);
    }

    // Construct the data rows

    var align = headers.reduce(function(pv, cv) {
      pv[cv.name] = cv.align;
      return pv;
    }, {});
    for (i = 0; i < data.length; i += 1) {
      if ("rowStart" in config && config.rowStart instanceof Function) {
        config.rowStart(
          {
            row: i,
            data: data[i]
          },
          this
        );
      }
      var lineHeight = calculateLineHeight.call(this, data[i], columnWidths);

      for (j = 0; j < headerNames.length; j += 1) {
        var cellData = data[i][headerNames[j]];
        if ("cellStart" in config && config.cellStart instanceof Function) {
          config.cellStart(
            {
              row: i,
              col: j,
              data: cellData
            },
            this
          );
        }
        cell.call(
          this,
          new Cell(
            x,
            y,
            columnWidths[headerNames[j]],
            lineHeight,
            cellData,
            i + 2,
            align[headerNames[j]]
          )
        );
      }
    }
    this.internal.__cell__.table_x = x;
    this.internal.__cell__.table_y = y;
    return this;
  };

  /**
   * Calculate the height for containing the highest column
   *
   * @name calculateLineHeight
   * @function
   * @param {Object[]} model is the line of data we want to calculate the height of
   * @param {Integer[]} columnWidths is size of each column
   * @returns {number} lineHeight
   * @private
   */
  var calculateLineHeight = function calculateLineHeight(model, columnWidths) {
    var padding = this.internal.__cell__.padding;
    var fontSize = this.internal.__cell__.table_font_size;
    var scaleFactor = this.internal.scaleFactor;

    return Object.keys(model)
      .map(function(key) {
        var value = model[key];
        return this.splitTextToSize(
          value.hasOwnProperty("text") ? value.text : value,
          columnWidths[key] - padding - padding
        );
      }, this)
      .map(function(value) {
        return (
          (this.getLineHeightFactor() * value.length * fontSize) / scaleFactor +
          padding +
          padding
        );
      }, this)
      .reduce(function(pv, cv) {
        return Math.max(pv, cv);
      }, 0);
  };

  /**
   * Store the config for outputting a table header
   *
   * @name setTableHeaderRow
   * @function
   * @param {Object[]} config
   * An array of cell configs that would define a header row: Each config matches the config used by jsPDFAPI.cell
   * except the lineNumber parameter is excluded
   */
  jsPDFAPI.setTableHeaderRow = function(config) {
    _initialize.call(this);
    this.internal.__cell__.tableHeaderRow = config;
  };

  /**
   * Output the store header row
   *
   * @name printHeaderRow
   * @function
   * @param {number} lineNumber The line number to output the header at
   * @param {boolean} new_page
   */
  jsPDFAPI.printHeaderRow = function(lineNumber, new_page) {
    _initialize.call(this);
    if (!this.internal.__cell__.tableHeaderRow) {
      throw new Error("Property tableHeaderRow does not exist.");
    }

    var tableHeaderCell;

    printingHeaderRow = true;
    if (typeof this.internal.__cell__.headerFunction === "function") {
      var position = this.internal.__cell__.headerFunction(
        this,
        this.internal.__cell__.pages
      );
      this.internal.__cell__.lastCell = new Cell(
        position[0],
        position[1],
        position[2],
        position[3],
        undefined,
        -1
      );
    }
    this.setFont(undefined, "bold");

    var tempHeaderConf = [];
    for (var i = 0; i < this.internal.__cell__.tableHeaderRow.length; i += 1) {
      tableHeaderCell = this.internal.__cell__.tableHeaderRow[i].clone();
      if (new_page) {
        tableHeaderCell.y = this.internal.__cell__.margins.top || 0;
        tempHeaderConf.push(tableHeaderCell);
      }
      tableHeaderCell.lineNumber = lineNumber;
      var currentTextColor = this.getTextColor();
      this.setTextColor(this.internal.__cell__.headerTextColor);
      this.setFillColor(this.internal.__cell__.headerBackgroundColor);
      cell.call(this, tableHeaderCell);
      this.setTextColor(currentTextColor);
    }
    if (tempHeaderConf.length > 0) {
      this.setTableHeaderRow(tempHeaderConf);
    }
    this.setFont(undefined, "normal");
    printingHeaderRow = false;
  };
})(jsPDF.API);

function toLookup(arr) {
  return arr.reduce(function(lookup, name, index) {
    lookup[name] = index;

    return lookup;
  }, {});
}

var fontStyleOrder = {
  italic: ["italic", "oblique", "normal"],
  oblique: ["oblique", "italic", "normal"],
  normal: ["normal", "oblique", "italic"]
};

var fontStretchOrder = [
  "ultra-condensed",
  "extra-condensed",
  "condensed",
  "semi-condensed",
  "normal",
  "semi-expanded",
  "expanded",
  "extra-expanded",
  "ultra-expanded"
];

// For a given font-stretch value, we need to know where to start our search
// from in the fontStretchOrder list.
var fontStretchLookup = toLookup(fontStretchOrder);

var fontWeights = [100, 200, 300, 400, 500, 600, 700, 800, 900];
var fontWeightsLookup = toLookup(fontWeights);

function normalizeFontStretch(stretch) {
  stretch = stretch || "normal";

  return typeof fontStretchLookup[stretch] === "number" ? stretch : "normal";
}

function normalizeFontStyle(style) {
  style = style || "normal";

  return fontStyleOrder[style] ? style : "normal";
}

function normalizeFontWeight(weight) {
  if (!weight) {
    return 400;
  }

  if (typeof weight === "number") {
    // Ignore values which aren't valid font-weights.
    return weight >= 100 && weight <= 900 && weight % 100 === 0 ? weight : 400;
  }

  if (/^\d00$/.test(weight)) {
    return parseInt(weight);
  }

  switch (weight) {
    case "bold":
      return 700;

    case "normal":
    default:
      return 400;
  }
}

function normalizeFontFace(fontFace) {
  var family = fontFace.family.replace(/"|'/g, "").toLowerCase();

  var style = normalizeFontStyle(fontFace.style);
  var weight = normalizeFontWeight(fontFace.weight);
  var stretch = normalizeFontStretch(fontFace.stretch);

  return {
    family: family,
    style: style,
    weight: weight,
    stretch: stretch,
    src: fontFace.src || [],

    // The ref property maps this font-face to the font
    // added by the .addFont() method.
    ref: fontFace.ref || {
      name: family,
      style: [stretch, style, weight].join(" ")
    }
  };
}

/**
 * Turns a list of font-faces into a map, for easier lookup when resolving
 * fonts.
 * @private
 */
function buildFontFaceMap(fontFaces) {
  var map = {};

  for (var i = 0; i < fontFaces.length; ++i) {
    var normalized = normalizeFontFace(fontFaces[i]);

    var name = normalized.family;
    var stretch = normalized.stretch;
    var style = normalized.style;
    var weight = normalized.weight;

    map[name] = map[name] || {};

    map[name][stretch] = map[name][stretch] || {};
    map[name][stretch][style] = map[name][stretch][style] || {};
    map[name][stretch][style][weight] = normalized;
  }

  return map;
}

/**
 * Searches a map of stretches, weights, etc. in the given direction and
 * then, if no match has been found, in the opposite directions.
 *
 * @param {Object.<string, any>} matchingSet A map of the various font variations.
 * @param {any[]} order The order of the different variations
 * @param {number} pivot The starting point of the search in the order list.
 * @param {number} dir The initial direction of the search (desc = -1, asc = 1)
 * @private
 */

function searchFromPivot(matchingSet, order, pivot, dir) {
  var i;

  for (i = pivot; i >= 0 && i < order.length; i += dir) {
    if (matchingSet[order[i]]) {
      return matchingSet[order[i]];
    }
  }

  for (i = pivot; i >= 0 && i < order.length; i -= dir) {
    if (matchingSet[order[i]]) {
      return matchingSet[order[i]];
    }
  }
}

function resolveFontStretch(stretch, matchingSet) {
  if (matchingSet[stretch]) {
    return matchingSet[stretch];
  }

  var pivot = fontStretchLookup[stretch];

  // If the font-stretch value is normal or more condensed, we want to
  // start with a descending search, otherwise we should do ascending.
  var dir = pivot <= fontStretchLookup["normal"] ? -1 : 1;
  var match = searchFromPivot(matchingSet, fontStretchOrder, pivot, dir);

  if (!match) {
    // Since a font-family cannot exist without having at least one stretch value
    // we should never reach this point.
    throw new Error(
      "Could not find a matching font-stretch value for " + stretch
    );
  }

  return match;
}

function resolveFontStyle(fontStyle, matchingSet) {
  if (matchingSet[fontStyle]) {
    return matchingSet[fontStyle];
  }

  var ordering = fontStyleOrder[fontStyle];

  for (var i = 0; i < ordering.length; ++i) {
    if (matchingSet[ordering[i]]) {
      return matchingSet[ordering[i]];
    }
  }

  // Since a font-family cannot exist without having at least one style value
  // we should never reach this point.
  throw new Error("Could not find a matching font-style for " + fontStyle);
}

function resolveFontWeight(weight, matchingSet) {
  if (matchingSet[weight]) {
    return matchingSet[weight];
  }

  if (weight === 400 && matchingSet[500]) {
    return matchingSet[500];
  }

  if (weight === 500 && matchingSet[400]) {
    return matchingSet[400];
  }

  var pivot = fontWeightsLookup[weight];

  // If the font-stretch value is normal or more condensed, we want to
  // start with a descending search, otherwise we should do ascending.
  var dir = weight < 400 ? -1 : 1;
  var match = searchFromPivot(matchingSet, fontWeights, pivot, dir);

  if (!match) {
    // Since a font-family cannot exist without having at least one stretch value
    // we should never reach this point.
    throw new Error(
      "Could not find a matching font-weight for value " + weight
    );
  }

  return match;
}

var defaultGenericFontFamilies = {
  "sans-serif": "helvetica",
  fixed: "courier",
  monospace: "courier",
  terminal: "courier",
  cursive: "times",
  fantasy: "times",
  serif: "times"
};

var systemFonts = {
  caption: "times",
  icon: "times",
  menu: "times",
  "message-box": "times",
  "small-caption": "times",
  "status-bar": "times"
};

function ruleToString(rule) {
  return [rule.stretch, rule.style, rule.weight, rule.family].join(" ");
}

function resolveFontFace(fontFaceMap, rules, opts) {
  opts = opts || {};

  var defaultFontFamily = opts.defaultFontFamily || "times";
  var genericFontFamilies = Object.assign(
    {},
    defaultGenericFontFamilies,
    opts.genericFontFamilies || {}
  );

  var rule = null;
  var matches = null;

  for (var i = 0; i < rules.length; ++i) {
    rule = normalizeFontFace(rules[i]);

    if (genericFontFamilies[rule.family]) {
      rule.family = genericFontFamilies[rule.family];
    }

    if (fontFaceMap.hasOwnProperty(rule.family)) {
      matches = fontFaceMap[rule.family];

      break;
    }
  }

  // Always fallback to a known font family.
  matches = matches || fontFaceMap[defaultFontFamily];

  if (!matches) {
    // At this point we should definitiely have a font family, but if we
    // don't there is something wrong with our configuration
    throw new Error(
      "Could not find a font-family for the rule '" +
        ruleToString(rule) +
        "' and default family '" +
        defaultFontFamily +
        "'."
    );
  }

  matches = resolveFontStretch(rule.stretch, matches);
  matches = resolveFontStyle(rule.style, matches);
  matches = resolveFontWeight(rule.weight, matches);

  if (!matches) {
    // We should've fount
    throw new Error(
      "Failed to resolve a font for the rule '" + ruleToString(rule) + "'."
    );
  }

  return matches;
}

function eatWhiteSpace(input) {
  return input.trimLeft();
}

function parseQuotedFontFamily(input, quote) {
  var index = 0;

  while (index < input.length) {
    var current = input.charAt(index);

    if (current === quote) {
      return [input.substring(0, index), input.substring(index + 1)];
    }

    index += 1;
  }

  // Unexpected end of input
  return null;
}

function parseNonQuotedFontFamily(input) {
  // It implements part of the identifier parser here: https://www.w3.org/TR/CSS21/syndata.html#value-def-identifier
  //
  // NOTE: This parser pretty much ignores escaped identifiers and that there is a thing called unicode.
  //
  // Breakdown of regexp:
  // -[a-z_]     - when identifier starts with a hyphen, you're not allowed to have another hyphen or a digit
  // [a-z_]      - allow a-z and underscore at beginning of input
  // [a-z0-9_-]* - after that, anything goes
  var match = input.match(/^(-[a-z_]|[a-z_])[a-z0-9_-]*/i);

  // non quoted value contains illegal characters
  if (match === null) {
    return null;
  }

  return [match[0], input.substring(match[0].length)];
}

var defaultFont = ["times"];

function parseFontFamily(input) {
  var result = [];
  var ch, parsed;
  var remaining = input.trim();

  if (remaining === "") {
    return defaultFont;
  }

  if (remaining in systemFonts) {
    return [systemFonts[remaining]];
  }

  while (remaining !== "") {
    parsed = null;
    remaining = eatWhiteSpace(remaining);
    ch = remaining.charAt(0);

    switch (ch) {
      case '"':
      case "'":
        parsed = parseQuotedFontFamily(remaining.substring(1), ch);
        break;

      default:
        parsed = parseNonQuotedFontFamily(remaining);
        break;
    }

    if (parsed === null) {
      return defaultFont;
    }

    result.push(parsed[0]);

    remaining = eatWhiteSpace(parsed[1]);

    // We expect end of input or a comma separator here
    if (remaining !== "" && remaining.charAt(0) !== ",") {
      return defaultFont;
    }

    remaining = remaining.replace(/^,/, "");
  }

  return result;
}

/* eslint-disable no-fallthrough */

/**
 * This plugin mimics the HTML5 CanvasRenderingContext2D.
 *
 * The goal is to provide a way for current canvas implementations to print directly to a PDF.
 *
 * @name context2d
 * @module
 */
(function(jsPDFAPI) {
  var ContextLayer = function(ctx) {
    ctx = ctx || {};
    this.isStrokeTransparent = ctx.isStrokeTransparent || false;
    this.strokeOpacity = ctx.strokeOpacity || 1;
    this.strokeStyle = ctx.strokeStyle || "#000000";
    this.fillStyle = ctx.fillStyle || "#000000";
    this.isFillTransparent = ctx.isFillTransparent || false;
    this.fillOpacity = ctx.fillOpacity || 1;
    this.font = ctx.font || "10px sans-serif";
    this.textBaseline = ctx.textBaseline || "alphabetic";
    this.textAlign = ctx.textAlign || "left";
    this.lineWidth = ctx.lineWidth || 1;
    this.lineJoin = ctx.lineJoin || "miter";
    this.lineCap = ctx.lineCap || "butt";
    this.path = ctx.path || [];
    this.transform =
      typeof ctx.transform !== "undefined"
        ? ctx.transform.clone()
        : new Matrix();
    this.globalCompositeOperation = ctx.globalCompositeOperation || "normal";
    this.globalAlpha = ctx.globalAlpha || 1.0;
    this.clip_path = ctx.clip_path || [];
    this.currentPoint = ctx.currentPoint || new Point();
    this.miterLimit = ctx.miterLimit || 10.0;
    this.lastPoint = ctx.lastPoint || new Point();
    this.lineDashOffset = ctx.lineDashOffset || 0.0;
    this.lineDash = ctx.lineDash || [];
    this.margin = ctx.margin || [0, 0, 0, 0];
    this.prevPageLastElemOffset = ctx.prevPageLastElemOffset || 0;

    this.ignoreClearRect =
      typeof ctx.ignoreClearRect === "boolean" ? ctx.ignoreClearRect : true;
    return this;
  };

  //stub
  var f2,
    getHorizontalCoordinateString,
    getVerticalCoordinateString,
    getHorizontalCoordinate,
    getVerticalCoordinate,
    Point,
    Rectangle,
    Matrix,
    _ctx;
  jsPDFAPI.events.push([
    "initialized",
    function() {
      this.context2d = new Context2D(this);

      f2 = this.internal.f2;
      getHorizontalCoordinateString = this.internal.getCoordinateString;
      getVerticalCoordinateString = this.internal.getVerticalCoordinateString;
      getHorizontalCoordinate = this.internal.getHorizontalCoordinate;
      getVerticalCoordinate = this.internal.getVerticalCoordinate;
      Point = this.internal.Point;
      Rectangle = this.internal.Rectangle;
      Matrix = this.internal.Matrix;
      _ctx = new ContextLayer();
    }
  ]);

  var Context2D = function(pdf) {
    Object.defineProperty(this, "canvas", {
      get: function() {
        return { parentNode: false, style: false };
      }
    });

    var _pdf = pdf;
    Object.defineProperty(this, "pdf", {
      get: function() {
        return _pdf;
      }
    });

    var _pageWrapXEnabled = false;
    /**
     * @name pageWrapXEnabled
     * @type {boolean}
     * @default false
     */
    Object.defineProperty(this, "pageWrapXEnabled", {
      get: function() {
        return _pageWrapXEnabled;
      },
      set: function(value) {
        _pageWrapXEnabled = Boolean(value);
      }
    });

    var _pageWrapYEnabled = false;
    /**
     * @name pageWrapYEnabled
     * @type {boolean}
     * @default true
     */
    Object.defineProperty(this, "pageWrapYEnabled", {
      get: function() {
        return _pageWrapYEnabled;
      },
      set: function(value) {
        _pageWrapYEnabled = Boolean(value);
      }
    });

    var _posX = 0;
    /**
     * @name posX
     * @type {number}
     * @default 0
     */
    Object.defineProperty(this, "posX", {
      get: function() {
        return _posX;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _posX = value;
        }
      }
    });

    var _posY = 0;
    /**
     * @name posY
     * @type {number}
     * @default 0
     */
    Object.defineProperty(this, "posY", {
      get: function() {
        return _posY;
      },
      set: function(value) {
        if (!isNaN(value)) {
          _posY = value;
        }
      }
    });

    /**
     * Gets or sets the page margin when using auto paging. Has no effect when {@link autoPaging} is off.
     * @name margin
     * @type {number|number[]}
     * @default [0, 0, 0, 0]
     */
    Object.defineProperty(this, "margin", {
      get: function() {
        return _ctx.margin;
      },
      set: function(value) {
        var margin;
        if (typeof value === "number") {
          margin = [value, value, value, value];
        } else {
          margin = new Array(4);
          margin[0] = value[0];
          margin[1] = value.length >= 2 ? value[1] : margin[0];
          margin[2] = value.length >= 3 ? value[2] : margin[0];
          margin[3] = value.length >= 4 ? value[3] : margin[1];
        }
        _ctx.margin = margin;
      }
    });

    var _autoPaging = false;
    /**
     * Gets or sets the auto paging mode. When auto paging is enabled, the context2d will automatically draw on the
     * next page if a shape or text chunk doesn't fit entirely on the current page. The context2d will create new
     * pages if required.
     *
     * Context2d supports different modes:
     * <ul>
     * <li>
     *   <code>false</code>: Auto paging is disabled.
     * </li>
     * <li>
     *   <code>true</code> or <code>'slice'</code>: Will cut shapes or text chunks across page breaks. Will possibly
     *   slice text in half, making it difficult to read.
     * </li>
     * <li>
     *   <code>'text'</code>: Trys not to cut text in half across page breaks. Works best for documents consisting
     *   mostly of a single column of text.
     * </li>
     * </ul>
     * @name Context2D#autoPaging
     * @type {boolean|"slice"|"text"}
     * @default false
     */
    Object.defineProperty(this, "autoPaging", {
      get: function() {
        return _autoPaging;
      },
      set: function(value) {
        _autoPaging = value;
      }
    });

    var lastBreak = 0;
    /**
     * @name lastBreak
     * @type {number}
     * @default 0
     */
    Object.defineProperty(this, "lastBreak", {
      get: function() {
        return lastBreak;
      },
      set: function(value) {
        lastBreak = value;
      }
    });

    var pageBreaks = [];
    /**
     * Y Position of page breaks.
     * @name pageBreaks
     * @type {number}
     * @default 0
     */
    Object.defineProperty(this, "pageBreaks", {
      get: function() {
        return pageBreaks;
      },
      set: function(value) {
        pageBreaks = value;
      }
    });

    /**
     * @name ctx
     * @type {object}
     * @default {}
     */
    Object.defineProperty(this, "ctx", {
      get: function() {
        return _ctx;
      },
      set: function(value) {
        if (value instanceof ContextLayer) {
          _ctx = value;
        }
      }
    });

    /**
     * @name path
     * @type {array}
     * @default []
     */
    Object.defineProperty(this, "path", {
      get: function() {
        return _ctx.path;
      },
      set: function(value) {
        _ctx.path = value;
      }
    });

    /**
     * @name ctxStack
     * @type {array}
     * @default []
     */
    var _ctxStack = [];
    Object.defineProperty(this, "ctxStack", {
      get: function() {
        return _ctxStack;
      },
      set: function(value) {
        _ctxStack = value;
      }
    });

    /**
     * Sets or returns the color, gradient, or pattern used to fill the drawing
     *
     * @name fillStyle
     * @default #000000
     * @property {(color|gradient|pattern)} value The color of the drawing. Default value is #000000<br />
     * A gradient object (linear or radial) used to fill the drawing (not supported by context2d)<br />
     * A pattern object to use to fill the drawing (not supported by context2d)
     */
    Object.defineProperty(this, "fillStyle", {
      get: function() {
        return this.ctx.fillStyle;
      },
      set: function(value) {
        var rgba;
        rgba = getRGBA(value);

        this.ctx.fillStyle = rgba.style;
        this.ctx.isFillTransparent = rgba.a === 0;
        this.ctx.fillOpacity = rgba.a;

        this.pdf.setFillColor(rgba.r, rgba.g, rgba.b, { a: rgba.a });
        this.pdf.setTextColor(rgba.r, rgba.g, rgba.b, { a: rgba.a });
      }
    });

    /**
     * Sets or returns the color, gradient, or pattern used for strokes
     *
     * @name strokeStyle
     * @default #000000
     * @property {color} color A CSS color value that indicates the stroke color of the drawing. Default value is #000000 (not supported by context2d)
     * @property {gradient} gradient A gradient object (linear or radial) used to create a gradient stroke (not supported by context2d)
     * @property {pattern} pattern A pattern object used to create a pattern stroke (not supported by context2d)
     */
    Object.defineProperty(this, "strokeStyle", {
      get: function() {
        return this.ctx.strokeStyle;
      },
      set: function(value) {
        var rgba = getRGBA(value);

        this.ctx.strokeStyle = rgba.style;
        this.ctx.isStrokeTransparent = rgba.a === 0;
        this.ctx.strokeOpacity = rgba.a;

        if (rgba.a === 0) {
          this.pdf.setDrawColor(255, 255, 255);
        } else if (rgba.a === 1) {
          this.pdf.setDrawColor(rgba.r, rgba.g, rgba.b);
        } else {
          this.pdf.setDrawColor(rgba.r, rgba.g, rgba.b);
        }
      }
    });

    /**
     * Sets or returns the style of the end caps for a line
     *
     * @name lineCap
     * @default butt
     * @property {(butt|round|square)} lineCap butt A flat edge is added to each end of the line <br/>
     * round A rounded end cap is added to each end of the line<br/>
     * square A square end cap is added to each end of the line<br/>
     */
    Object.defineProperty(this, "lineCap", {
      get: function() {
        return this.ctx.lineCap;
      },
      set: function(value) {
        if (["butt", "round", "square"].indexOf(value) !== -1) {
          this.ctx.lineCap = value;
          this.pdf.setLineCap(value);
        }
      }
    });

    /**
     * Sets or returns the current line width
     *
     * @name lineWidth
     * @default 1
     * @property {number} lineWidth The current line width, in pixels
     */
    Object.defineProperty(this, "lineWidth", {
      get: function() {
        return this.ctx.lineWidth;
      },
      set: function(value) {
        if (!isNaN(value)) {
          this.ctx.lineWidth = value;
          this.pdf.setLineWidth(value);
        }
      }
    });

    /**
     * Sets or returns the type of corner created, when two lines meet
     */
    Object.defineProperty(this, "lineJoin", {
      get: function() {
        return this.ctx.lineJoin;
      },
      set: function(value) {
        if (["bevel", "round", "miter"].indexOf(value) !== -1) {
          this.ctx.lineJoin = value;
          this.pdf.setLineJoin(value);
        }
      }
    });

    /**
     * A number specifying the miter limit ratio in coordinate space units. Zero, negative, Infinity, and NaN values are ignored. The default value is 10.0.
     *
     * @name miterLimit
     * @default 10
     */
    Object.defineProperty(this, "miterLimit", {
      get: function() {
        return this.ctx.miterLimit;
      },
      set: function(value) {
        if (!isNaN(value)) {
          this.ctx.miterLimit = value;
          this.pdf.setMiterLimit(value);
        }
      }
    });

    Object.defineProperty(this, "textBaseline", {
      get: function() {
        return this.ctx.textBaseline;
      },
      set: function(value) {
        this.ctx.textBaseline = value;
      }
    });

    Object.defineProperty(this, "textAlign", {
      get: function() {
        return this.ctx.textAlign;
      },
      set: function(value) {
        if (["right", "end", "center", "left", "start"].indexOf(value) !== -1) {
          this.ctx.textAlign = value;
        }
      }
    });

    var _fontFaceMap = null;

    function getFontFaceMap(pdf, fontFaces) {
      if (_fontFaceMap === null) {
        var fontMap = pdf.getFontList();

        var convertedFontFaces = convertToFontFaces(fontMap);

        _fontFaceMap = buildFontFaceMap(convertedFontFaces.concat(fontFaces));
      }

      return _fontFaceMap;
    }

    function convertToFontFaces(fontMap) {
      var fontFaces = [];

      Object.keys(fontMap).forEach(function(family) {
        var styles = fontMap[family];

        styles.forEach(function(style) {
          var fontFace = null;

          switch (style) {
            case "bold":
              fontFace = {
                family: family,
                weight: "bold"
              };
              break;

            case "italic":
              fontFace = {
                family: family,
                style: "italic"
              };
              break;

            case "bolditalic":
              fontFace = {
                family: family,
                weight: "bold",
                style: "italic"
              };
              break;

            case "":
            case "normal":
              fontFace = {
                family: family
              };
              break;
          }

          // If font-face is still null here, it is a font with some styling we don't recognize and
          // cannot map or it is a font added via the fontFaces option of .html().
          if (fontFace !== null) {
            fontFace.ref = {
              name: family,
              style: style
            };

            fontFaces.push(fontFace);
          }
        });
      });

      return fontFaces;
    }

    var _fontFaces = null;
    /**
     * A map of available font-faces, as passed in the options of
     * .html(). If set a limited implementation of the font style matching
     * algorithm defined by https://www.w3.org/TR/css-fonts-3/#font-matching-algorithm
     * will be used. If not set it will fallback to previous behavior.
     */

    Object.defineProperty(this, "fontFaces", {
      get: function() {
        return _fontFaces;
      },
      set: function(value) {
        _fontFaceMap = null;
        _fontFaces = value;
      }
    });

    Object.defineProperty(this, "font", {
      get: function() {
        return this.ctx.font;
      },
      set: function(value) {
        this.ctx.font = value;
        var rx, matches;

        //source: https://stackoverflow.com/a/10136041
        // eslint-disable-next-line no-useless-escape
        rx = /^\s*(?=(?:(?:[-a-z]+\s*){0,2}(italic|oblique))?)(?=(?:(?:[-a-z]+\s*){0,2}(small-caps))?)(?=(?:(?:[-a-z]+\s*){0,2}(bold(?:er)?|lighter|[1-9]00))?)(?:(?:normal|\1|\2|\3)\s*){0,3}((?:xx?-)?(?:small|large)|medium|smaller|larger|[.\d]+(?:\%|in|[cem]m|ex|p[ctx]))(?:\s*\/\s*(normal|[.\d]+(?:\%|in|[cem]m|ex|p[ctx])))?\s*([-_,\"\'\sa-z]+?)\s*$/i;
        matches = rx.exec(value);
        if (matches !== null) {
          var fontStyle = matches[1];
          matches[2];
          var fontWeight = matches[3];
          var fontSize = matches[4];
          matches[5];
          var fontFamily = matches[6];
        } else {
          return;
        }
        var rxFontSize = /^([.\d]+)((?:%|in|[cem]m|ex|p[ctx]))$/i;
        var fontSizeUnit = rxFontSize.exec(fontSize)[2];

        if ("px" === fontSizeUnit) {
          fontSize = Math.floor(
            parseFloat(fontSize) * this.pdf.internal.scaleFactor
          );
        } else if ("em" === fontSizeUnit) {
          fontSize = Math.floor(parseFloat(fontSize) * this.pdf.getFontSize());
        } else {
          fontSize = Math.floor(
            parseFloat(fontSize) * this.pdf.internal.scaleFactor
          );
        }

        this.pdf.setFontSize(fontSize);
        var parts = parseFontFamily(fontFamily);

        if (this.fontFaces) {
          var fontFaceMap = getFontFaceMap(this.pdf, this.fontFaces);

          var rules = parts.map(function(ff) {
            return {
              family: ff,
              stretch: "normal", // TODO: Extract font-stretch from font rule (perhaps write proper parser for it?)
              weight: fontWeight,
              style: fontStyle
            };
          });

          var font = resolveFontFace(fontFaceMap, rules);
          this.pdf.setFont(font.ref.name, font.ref.style);
          return;
        }

        var style = "";
        if (
          fontWeight === "bold" ||
          parseInt(fontWeight, 10) >= 700 ||
          fontStyle === "bold"
        ) {
          style = "bold";
        }

        if (fontStyle === "italic") {
          style += "italic";
        }

        if (style.length === 0) {
          style = "normal";
        }
        var jsPdfFontName = "";

        var fallbackFonts = {
          arial: "Helvetica",
          Arial: "Helvetica",
          verdana: "Helvetica",
          Verdana: "Helvetica",
          helvetica: "Helvetica",
          Helvetica: "Helvetica",
          "sans-serif": "Helvetica",
          fixed: "Courier",
          monospace: "Courier",
          terminal: "Courier",
          cursive: "Times",
          fantasy: "Times",
          serif: "Times"
        };

        for (var i = 0; i < parts.length; i++) {
          if (
            this.pdf.internal.getFont(parts[i], style, {
              noFallback: true,
              disableWarning: true
            }) !== undefined
          ) {
            jsPdfFontName = parts[i];
            break;
          } else if (
            style === "bolditalic" &&
            this.pdf.internal.getFont(parts[i], "bold", {
              noFallback: true,
              disableWarning: true
            }) !== undefined
          ) {
            jsPdfFontName = parts[i];
            style = "bold";
          } else if (
            this.pdf.internal.getFont(parts[i], "normal", {
              noFallback: true,
              disableWarning: true
            }) !== undefined
          ) {
            jsPdfFontName = parts[i];
            style = "normal";
            break;
          }
        }
        if (jsPdfFontName === "") {
          for (var j = 0; j < parts.length; j++) {
            if (fallbackFonts[parts[j]]) {
              jsPdfFontName = fallbackFonts[parts[j]];
              break;
            }
          }
        }
        jsPdfFontName = jsPdfFontName === "" ? "Times" : jsPdfFontName;
        this.pdf.setFont(jsPdfFontName, style);
      }
    });

    Object.defineProperty(this, "globalCompositeOperation", {
      get: function() {
        return this.ctx.globalCompositeOperation;
      },
      set: function(value) {
        this.ctx.globalCompositeOperation = value;
      }
    });

    Object.defineProperty(this, "globalAlpha", {
      get: function() {
        return this.ctx.globalAlpha;
      },
      set: function(value) {
        this.ctx.globalAlpha = value;
      }
    });

    /**
     * A float specifying the amount of the line dash offset. The default value is 0.0.
     *
     * @name lineDashOffset
     * @default 0.0
     */
    Object.defineProperty(this, "lineDashOffset", {
      get: function() {
        return this.ctx.lineDashOffset;
      },
      set: function(value) {
        this.ctx.lineDashOffset = value;
        setLineDash.call(this);
      }
    });

    // Not HTML API
    Object.defineProperty(this, "lineDash", {
      get: function() {
        return this.ctx.lineDash;
      },
      set: function(value) {
        this.ctx.lineDash = value;
        setLineDash.call(this);
      }
    });

    // Not HTML API
    Object.defineProperty(this, "ignoreClearRect", {
      get: function() {
        return this.ctx.ignoreClearRect;
      },
      set: function(value) {
        this.ctx.ignoreClearRect = Boolean(value);
      }
    });
  };

  /**
   * Sets the line dash pattern used when stroking lines.
   * @name setLineDash
   * @function
   * @description It uses an array of values that specify alternating lengths of lines and gaps which describe the pattern.
   */
  Context2D.prototype.setLineDash = function(dashArray) {
    this.lineDash = dashArray;
  };

  /**
   * gets the current line dash pattern.
   * @name getLineDash
   * @function
   * @returns {Array} An Array of numbers that specify distances to alternately draw a line and a gap (in coordinate space units). If the number, when setting the elements, is odd, the elements of the array get copied and concatenated. For example, setting the line dash to [5, 15, 25] will result in getting back [5, 15, 25, 5, 15, 25].
   */
  Context2D.prototype.getLineDash = function() {
    if (this.lineDash.length % 2) {
      // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getLineDash#return_value
      return this.lineDash.concat(this.lineDash);
    } else {
      // The copied value is returned to prevent contamination from outside.
      return this.lineDash.slice();
    }
  };

  Context2D.prototype.fill = function() {
    pathPreProcess.call(this, "fill", false);
  };

  /**
   * Actually draws the path you have defined
   *
   * @name stroke
   * @function
   * @description The stroke() method actually draws the path you have defined with all those moveTo() and lineTo() methods. The default color is black.
   */
  Context2D.prototype.stroke = function() {
    pathPreProcess.call(this, "stroke", false);
  };

  /**
   * Begins a path, or resets the current
   *
   * @name beginPath
   * @function
   * @description The beginPath() method begins a path, or resets the current path.
   */
  Context2D.prototype.beginPath = function() {
    this.path = [
      {
        type: "begin"
      }
    ];
  };

  /**
   * Moves the path to the specified point in the canvas, without creating a line
   *
   * @name moveTo
   * @function
   * @param x {Number} The x-coordinate of where to move the path to
   * @param y {Number} The y-coordinate of where to move the path to
   */
  Context2D.prototype.moveTo = function(x, y) {
    if (isNaN(x) || isNaN(y)) {
      console.error("jsPDF.context2d.moveTo: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.moveTo");
    }

    var pt = this.ctx.transform.applyToPoint(new Point(x, y));

    this.path.push({
      type: "mt",
      x: pt.x,
      y: pt.y
    });
    this.ctx.lastPoint = new Point(x, y);
  };

  /**
   * Creates a path from the current point back to the starting point
   *
   * @name closePath
   * @function
   * @description The closePath() method creates a path from the current point back to the starting point.
   */
  Context2D.prototype.closePath = function() {
    var pathBegin = new Point(0, 0);
    var i = 0;
    for (i = this.path.length - 1; i !== -1; i--) {
      if (this.path[i].type === "begin") {
        if (
          typeof this.path[i + 1] === "object" &&
          typeof this.path[i + 1].x === "number"
        ) {
          pathBegin = new Point(this.path[i + 1].x, this.path[i + 1].y);
          break;
        }
      }
    }
    this.path.push({
      type: "close"
    });
    this.ctx.lastPoint = new Point(pathBegin.x, pathBegin.y);
  };

  /**
   * Adds a new point and creates a line to that point from the last specified point in the canvas
   *
   * @name lineTo
   * @function
   * @param x The x-coordinate of where to create the line to
   * @param y The y-coordinate of where to create the line to
   * @description The lineTo() method adds a new point and creates a line TO that point FROM the last specified point in the canvas (this method does not draw the line).
   */
  Context2D.prototype.lineTo = function(x, y) {
    if (isNaN(x) || isNaN(y)) {
      console.error("jsPDF.context2d.lineTo: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.lineTo");
    }

    var pt = this.ctx.transform.applyToPoint(new Point(x, y));

    this.path.push({
      type: "lt",
      x: pt.x,
      y: pt.y
    });
    this.ctx.lastPoint = new Point(pt.x, pt.y);
  };

  /**
   * Clips a region of any shape and size from the original canvas
   *
   * @name clip
   * @function
   * @description The clip() method clips a region of any shape and size from the original canvas.
   */
  Context2D.prototype.clip = function() {
    this.ctx.clip_path = JSON.parse(JSON.stringify(this.path));
    pathPreProcess.call(this, null, true);
  };

  /**
   * Creates a cubic Bézier curve
   *
   * @name quadraticCurveTo
   * @function
   * @param cpx {Number} The x-coordinate of the Bézier control point
   * @param cpy {Number} The y-coordinate of the Bézier control point
   * @param x {Number} The x-coordinate of the ending point
   * @param y {Number} The y-coordinate of the ending point
   * @description The quadraticCurveTo() method adds a point to the current path by using the specified control points that represent a quadratic Bézier curve.<br /><br /> A quadratic Bézier curve requires two points. The first point is a control point that is used in the quadratic Bézier calculation and the second point is the ending point for the curve. The starting point for the curve is the last point in the current path. If a path does not exist, use the beginPath() and moveTo() methods to define a starting point.
   */
  Context2D.prototype.quadraticCurveTo = function(cpx, cpy, x, y) {
    if (isNaN(x) || isNaN(y) || isNaN(cpx) || isNaN(cpy)) {
      console.error(
        "jsPDF.context2d.quadraticCurveTo: Invalid arguments",
        arguments
      );
      throw new Error(
        "Invalid arguments passed to jsPDF.context2d.quadraticCurveTo"
      );
    }

    var pt0 = this.ctx.transform.applyToPoint(new Point(x, y));
    var pt1 = this.ctx.transform.applyToPoint(new Point(cpx, cpy));

    this.path.push({
      type: "qct",
      x1: pt1.x,
      y1: pt1.y,
      x: pt0.x,
      y: pt0.y
    });
    this.ctx.lastPoint = new Point(pt0.x, pt0.y);
  };

  /**
   * Creates a cubic Bézier curve
   *
   * @name bezierCurveTo
   * @function
   * @param cp1x {Number} The x-coordinate of the first Bézier control point
   * @param cp1y {Number} The y-coordinate of the first Bézier control point
   * @param cp2x {Number} The x-coordinate of the second Bézier control point
   * @param cp2y {Number} The y-coordinate of the second Bézier control point
   * @param x {Number} The x-coordinate of the ending point
   * @param y {Number} The y-coordinate of the ending point
   * @description The bezierCurveTo() method adds a point to the current path by using the specified control points that represent a cubic Bézier curve. <br /><br />A cubic bezier curve requires three points. The first two points are control points that are used in the cubic Bézier calculation and the last point is the ending point for the curve.  The starting point for the curve is the last point in the current path. If a path does not exist, use the beginPath() and moveTo() methods to define a starting point.
   */
  Context2D.prototype.bezierCurveTo = function(cp1x, cp1y, cp2x, cp2y, x, y) {
    if (
      isNaN(x) ||
      isNaN(y) ||
      isNaN(cp1x) ||
      isNaN(cp1y) ||
      isNaN(cp2x) ||
      isNaN(cp2y)
    ) {
      console.error(
        "jsPDF.context2d.bezierCurveTo: Invalid arguments",
        arguments
      );
      throw new Error(
        "Invalid arguments passed to jsPDF.context2d.bezierCurveTo"
      );
    }
    var pt0 = this.ctx.transform.applyToPoint(new Point(x, y));
    var pt1 = this.ctx.transform.applyToPoint(new Point(cp1x, cp1y));
    var pt2 = this.ctx.transform.applyToPoint(new Point(cp2x, cp2y));

    this.path.push({
      type: "bct",
      x1: pt1.x,
      y1: pt1.y,
      x2: pt2.x,
      y2: pt2.y,
      x: pt0.x,
      y: pt0.y
    });
    this.ctx.lastPoint = new Point(pt0.x, pt0.y);
  };

  /**
   * Creates an arc/curve (used to create circles, or parts of circles)
   *
   * @name arc
   * @function
   * @param x {Number} The x-coordinate of the center of the circle
   * @param y {Number} The y-coordinate of the center of the circle
   * @param radius {Number} The radius of the circle
   * @param startAngle {Number} The starting angle, in radians (0 is at the 3 o'clock position of the arc's circle)
   * @param endAngle {Number} The ending angle, in radians
   * @param counterclockwise {Boolean} Optional. Specifies whether the drawing should be counterclockwise or clockwise. False is default, and indicates clockwise, while true indicates counter-clockwise.
   * @description The arc() method creates an arc/curve (used to create circles, or parts of circles).
   */
  Context2D.prototype.arc = function(
    x,
    y,
    radius,
    startAngle,
    endAngle,
    counterclockwise
  ) {
    if (
      isNaN(x) ||
      isNaN(y) ||
      isNaN(radius) ||
      isNaN(startAngle) ||
      isNaN(endAngle)
    ) {
      console.error("jsPDF.context2d.arc: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.arc");
    }
    counterclockwise = Boolean(counterclockwise);

    if (!this.ctx.transform.isIdentity) {
      var xpt = this.ctx.transform.applyToPoint(new Point(x, y));
      x = xpt.x;
      y = xpt.y;

      var x_radPt = this.ctx.transform.applyToPoint(new Point(0, radius));
      var x_radPt0 = this.ctx.transform.applyToPoint(new Point(0, 0));
      radius = Math.sqrt(
        Math.pow(x_radPt.x - x_radPt0.x, 2) +
          Math.pow(x_radPt.y - x_radPt0.y, 2)
      );
    }
    if (Math.abs(endAngle - startAngle) >= 2 * Math.PI) {
      startAngle = 0;
      endAngle = 2 * Math.PI;
    }

    this.path.push({
      type: "arc",
      x: x,
      y: y,
      radius: radius,
      startAngle: startAngle,
      endAngle: endAngle,
      counterclockwise: counterclockwise
    });
    // this.ctx.lastPoint(new Point(pt.x,pt.y));
  };

  /**
   * Creates an arc/curve between two tangents
   *
   * @name arcTo
   * @function
   * @param x1 {Number} The x-coordinate of the first tangent
   * @param y1 {Number} The y-coordinate of the first tangent
   * @param x2 {Number} The x-coordinate of the second tangent
   * @param y2 {Number} The y-coordinate of the second tangent
   * @param radius The radius of the arc
   * @description The arcTo() method creates an arc/curve between two tangents on the canvas.
   */
  // eslint-disable-next-line no-unused-vars
  Context2D.prototype.arcTo = function(x1, y1, x2, y2, radius) {
    throw new Error("arcTo not implemented.");
  };

  /**
   * Creates a rectangle
   *
   * @name rect
   * @function
   * @param x {Number} The x-coordinate of the upper-left corner of the rectangle
   * @param y {Number} The y-coordinate of the upper-left corner of the rectangle
   * @param w {Number} The width of the rectangle, in pixels
   * @param h {Number} The height of the rectangle, in pixels
   * @description The rect() method creates a rectangle.
   */
  Context2D.prototype.rect = function(x, y, w, h) {
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h)) {
      console.error("jsPDF.context2d.rect: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.rect");
    }
    this.moveTo(x, y);
    this.lineTo(x + w, y);
    this.lineTo(x + w, y + h);
    this.lineTo(x, y + h);
    this.lineTo(x, y);
    this.lineTo(x + w, y);
    this.lineTo(x, y);
  };

  /**
   * Draws a "filled" rectangle
   *
   * @name fillRect
   * @function
   * @param x {Number} The x-coordinate of the upper-left corner of the rectangle
   * @param y {Number} The y-coordinate of the upper-left corner of the rectangle
   * @param w {Number} The width of the rectangle, in pixels
   * @param h {Number} The height of the rectangle, in pixels
   * @description The fillRect() method draws a "filled" rectangle. The default color of the fill is black.
   */
  Context2D.prototype.fillRect = function(x, y, w, h) {
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h)) {
      console.error("jsPDF.context2d.fillRect: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.fillRect");
    }
    if (isFillTransparent.call(this)) {
      return;
    }
    var tmp = {};
    if (this.lineCap !== "butt") {
      tmp.lineCap = this.lineCap;
      this.lineCap = "butt";
    }
    if (this.lineJoin !== "miter") {
      tmp.lineJoin = this.lineJoin;
      this.lineJoin = "miter";
    }

    this.beginPath();
    this.rect(x, y, w, h);
    this.fill();

    if (tmp.hasOwnProperty("lineCap")) {
      this.lineCap = tmp.lineCap;
    }
    if (tmp.hasOwnProperty("lineJoin")) {
      this.lineJoin = tmp.lineJoin;
    }
  };

  /**
   *     Draws a rectangle (no fill)
   *
   * @name strokeRect
   * @function
   * @param x {Number} The x-coordinate of the upper-left corner of the rectangle
   * @param y {Number} The y-coordinate of the upper-left corner of the rectangle
   * @param w {Number} The width of the rectangle, in pixels
   * @param h {Number} The height of the rectangle, in pixels
   * @description The strokeRect() method draws a rectangle (no fill). The default color of the stroke is black.
   */
  Context2D.prototype.strokeRect = function strokeRect(x, y, w, h) {
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h)) {
      console.error("jsPDF.context2d.strokeRect: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.strokeRect");
    }
    if (isStrokeTransparent.call(this)) {
      return;
    }
    this.beginPath();
    this.rect(x, y, w, h);
    this.stroke();
  };

  /**
   * Clears the specified pixels within a given rectangle
   *
   * @name clearRect
   * @function
   * @param x {Number} The x-coordinate of the upper-left corner of the rectangle
   * @param y {Number} The y-coordinate of the upper-left corner of the rectangle
   * @param w {Number} The width of the rectangle to clear, in pixels
   * @param h {Number} The height of the rectangle to clear, in pixels
   * @description We cannot clear PDF commands that were already written to PDF, so we use white instead. <br />
   * As a special case, read a special flag (ignoreClearRect) and do nothing if it is set.
   * This results in all calls to clearRect() to do nothing, and keep the canvas transparent.
   * This flag is stored in the save/restore context and is managed the same way as other drawing states.
   *
   */
  Context2D.prototype.clearRect = function(x, y, w, h) {
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h)) {
      console.error("jsPDF.context2d.clearRect: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.clearRect");
    }
    if (this.ignoreClearRect) {
      return;
    }

    this.fillStyle = "#ffffff";
    this.fillRect(x, y, w, h);
  };

  /**
   * Saves the state of the current context
   *
   * @name save
   * @function
   */
  Context2D.prototype.save = function(doStackPush) {
    doStackPush = typeof doStackPush === "boolean" ? doStackPush : true;
    var tmpPageNumber = this.pdf.internal.getCurrentPageInfo().pageNumber;
    for (var i = 0; i < this.pdf.internal.getNumberOfPages(); i++) {
      this.pdf.setPage(i + 1);
      this.pdf.internal.out("q");
    }
    this.pdf.setPage(tmpPageNumber);

    if (doStackPush) {
      this.ctx.fontSize = this.pdf.internal.getFontSize();
      var ctx = new ContextLayer(this.ctx);
      this.ctxStack.push(this.ctx);
      this.ctx = ctx;
    }
  };

  /**
   * Returns previously saved path state and attributes
   *
   * @name restore
   * @function
   */
  Context2D.prototype.restore = function(doStackPop) {
    doStackPop = typeof doStackPop === "boolean" ? doStackPop : true;
    var tmpPageNumber = this.pdf.internal.getCurrentPageInfo().pageNumber;
    for (var i = 0; i < this.pdf.internal.getNumberOfPages(); i++) {
      this.pdf.setPage(i + 1);
      this.pdf.internal.out("Q");
    }
    this.pdf.setPage(tmpPageNumber);

    if (doStackPop && this.ctxStack.length !== 0) {
      this.ctx = this.ctxStack.pop();
      this.fillStyle = this.ctx.fillStyle;
      this.strokeStyle = this.ctx.strokeStyle;
      this.font = this.ctx.font;
      this.lineCap = this.ctx.lineCap;
      this.lineWidth = this.ctx.lineWidth;
      this.lineJoin = this.ctx.lineJoin;
      this.lineDash = this.ctx.lineDash;
      this.lineDashOffset = this.ctx.lineDashOffset;
    }
  };

  /**
   * @name toDataURL
   * @function
   */
  Context2D.prototype.toDataURL = function() {
    throw new Error("toDataUrl not implemented.");
  };

  //helper functions

  /**
   * Get the decimal values of r, g, b and a
   *
   * @name getRGBA
   * @function
   * @private
   * @ignore
   */
  var getRGBA = function(style) {
    var rxRgb = /rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/;
    var rxRgba = /rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)/;
    var rxTransparent = /transparent|rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*0+\s*\)/;

    var r, g, b, a;

    if (style.isCanvasGradient === true) {
      style = style.getColor();
    }

    if (!style) {
      return { r: 0, g: 0, b: 0, a: 0, style: style };
    }

    if (rxTransparent.test(style)) {
      r = 0;
      g = 0;
      b = 0;
      a = 0;
    } else {
      var matches = rxRgb.exec(style);
      if (matches !== null) {
        r = parseInt(matches[1]);
        g = parseInt(matches[2]);
        b = parseInt(matches[3]);
        a = 1;
      } else {
        matches = rxRgba.exec(style);
        if (matches !== null) {
          r = parseInt(matches[1]);
          g = parseInt(matches[2]);
          b = parseInt(matches[3]);
          a = parseFloat(matches[4]);
        } else {
          a = 1;

          if (typeof style === "string" && style.charAt(0) !== "#") {
            var rgbColor = new RGBColor(style);
            if (rgbColor.ok) {
              style = rgbColor.toHex();
            } else {
              style = "#000000";
            }
          }

          if (style.length === 4) {
            r = style.substring(1, 2);
            r += r;
            g = style.substring(2, 3);
            g += g;
            b = style.substring(3, 4);
            b += b;
          } else {
            r = style.substring(1, 3);
            g = style.substring(3, 5);
            b = style.substring(5, 7);
          }
          r = parseInt(r, 16);
          g = parseInt(g, 16);
          b = parseInt(b, 16);
        }
      }
    }
    return { r: r, g: g, b: b, a: a, style: style };
  };

  /**
   * @name isFillTransparent
   * @function
   * @private
   * @ignore
   * @returns {Boolean}
   */
  var isFillTransparent = function() {
    return this.ctx.isFillTransparent || this.globalAlpha == 0;
  };

  /**
   * @name isStrokeTransparent
   * @function
   * @private
   * @ignore
   * @returns {Boolean}
   */
  var isStrokeTransparent = function() {
    return Boolean(this.ctx.isStrokeTransparent || this.globalAlpha == 0);
  };

  /**
   * Draws "filled" text on the canvas
   *
   * @name fillText
   * @function
   * @param text {String} Specifies the text that will be written on the canvas
   * @param x {Number} The x coordinate where to start painting the text (relative to the canvas)
   * @param y {Number} The y coordinate where to start painting the text (relative to the canvas)
   * @param maxWidth {Number} Optional. The maximum allowed width of the text, in pixels
   * @description The fillText() method draws filled text on the canvas. The default color of the text is black.
   */
  Context2D.prototype.fillText = function(text, x, y, maxWidth) {
    if (isNaN(x) || isNaN(y) || typeof text !== "string") {
      console.error("jsPDF.context2d.fillText: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.fillText");
    }
    maxWidth = isNaN(maxWidth) ? undefined : maxWidth;
    if (isFillTransparent.call(this)) {
      return;
    }

    var degs = rad2deg(this.ctx.transform.rotation);

    // We only use X axis as scale hint
    var scale = this.ctx.transform.scaleX;

    putText.call(this, {
      text: text,
      x: x,
      y: y,
      scale: scale,
      angle: degs,
      align: this.textAlign,
      maxWidth: maxWidth
    });
  };

  /**
   * Draws text on the canvas (no fill)
   *
   * @name strokeText
   * @function
   * @param text {String} Specifies the text that will be written on the canvas
   * @param x {Number} The x coordinate where to start painting the text (relative to the canvas)
   * @param y {Number} The y coordinate where to start painting the text (relative to the canvas)
   * @param maxWidth {Number} Optional. The maximum allowed width of the text, in pixels
   * @description The strokeText() method draws text (with no fill) on the canvas. The default color of the text is black.
   */
  Context2D.prototype.strokeText = function(text, x, y, maxWidth) {
    if (isNaN(x) || isNaN(y) || typeof text !== "string") {
      console.error("jsPDF.context2d.strokeText: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.strokeText");
    }
    if (isStrokeTransparent.call(this)) {
      return;
    }

    maxWidth = isNaN(maxWidth) ? undefined : maxWidth;

    var degs = rad2deg(this.ctx.transform.rotation);
    var scale = this.ctx.transform.scaleX;

    putText.call(this, {
      text: text,
      x: x,
      y: y,
      scale: scale,
      renderingMode: "stroke",
      angle: degs,
      align: this.textAlign,
      maxWidth: maxWidth
    });
  };

  /**
   * Returns an object that contains the width of the specified text
   *
   * @name measureText
   * @function
   * @param text {String} The text to be measured
   * @description The measureText() method returns an object that contains the width of the specified text, in pixels.
   * @returns {Number}
   */
  Context2D.prototype.measureText = function(text) {
    if (typeof text !== "string") {
      console.error(
        "jsPDF.context2d.measureText: Invalid arguments",
        arguments
      );
      throw new Error(
        "Invalid arguments passed to jsPDF.context2d.measureText"
      );
    }
    var pdf = this.pdf;
    var k = this.pdf.internal.scaleFactor;

    var fontSize = pdf.internal.getFontSize();
    var txtWidth =
      (pdf.getStringUnitWidth(text) * fontSize) / pdf.internal.scaleFactor;
    txtWidth *= Math.round(((k * 96) / 72) * 10000) / 10000;

    var TextMetrics = function(options) {
      options = options || {};
      var _width = options.width || 0;
      Object.defineProperty(this, "width", {
        get: function() {
          return _width;
        }
      });
      return this;
    };
    return new TextMetrics({ width: txtWidth });
  };

  //Transformations

  /**
   * Scales the current drawing bigger or smaller
   *
   * @name scale
   * @function
   * @param scalewidth {Number} Scales the width of the current drawing (1=100%, 0.5=50%, 2=200%, etc.)
   * @param scaleheight {Number} Scales the height of the current drawing (1=100%, 0.5=50%, 2=200%, etc.)
   * @description The scale() method scales the current drawing, bigger or smaller.
   */
  Context2D.prototype.scale = function(scalewidth, scaleheight) {
    if (isNaN(scalewidth) || isNaN(scaleheight)) {
      console.error("jsPDF.context2d.scale: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.scale");
    }
    var matrix = new Matrix(scalewidth, 0.0, 0.0, scaleheight, 0.0, 0.0);
    this.ctx.transform = this.ctx.transform.multiply(matrix);
  };

  /**
   * Rotates the current drawing
   *
   * @name rotate
   * @function
   * @param angle {Number} The rotation angle, in radians.
   * @description To calculate from degrees to radians: degrees*Math.PI/180. <br />
   * Example: to rotate 5 degrees, specify the following: 5*Math.PI/180
   */
  Context2D.prototype.rotate = function(angle) {
    if (isNaN(angle)) {
      console.error("jsPDF.context2d.rotate: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.rotate");
    }
    var matrix = new Matrix(
      Math.cos(angle),
      Math.sin(angle),
      -Math.sin(angle),
      Math.cos(angle),
      0.0,
      0.0
    );
    this.ctx.transform = this.ctx.transform.multiply(matrix);
  };

  /**
   * Remaps the (0,0) position on the canvas
   *
   * @name translate
   * @function
   * @param x {Number} The value to add to horizontal (x) coordinates
   * @param y {Number} The value to add to vertical (y) coordinates
   * @description The translate() method remaps the (0,0) position on the canvas.
   */
  Context2D.prototype.translate = function(x, y) {
    if (isNaN(x) || isNaN(y)) {
      console.error("jsPDF.context2d.translate: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.translate");
    }
    var matrix = new Matrix(1.0, 0.0, 0.0, 1.0, x, y);
    this.ctx.transform = this.ctx.transform.multiply(matrix);
  };

  /**
   * Replaces the current transformation matrix for the drawing
   *
   * @name transform
   * @function
   * @param a {Number} Horizontal scaling
   * @param b {Number} Horizontal skewing
   * @param c {Number} Vertical skewing
   * @param d {Number} Vertical scaling
   * @param e {Number} Horizontal moving
   * @param f {Number} Vertical moving
   * @description Each object on the canvas has a current transformation matrix.<br /><br />The transform() method replaces the current transformation matrix. It multiplies the current transformation matrix with the matrix described by:<br /><br /><br /><br />a    c    e<br /><br />b    d    f<br /><br />0    0    1<br /><br />In other words, the transform() method lets you scale, rotate, move, and skew the current context.
   */
  Context2D.prototype.transform = function(a, b, c, d, e, f) {
    if (isNaN(a) || isNaN(b) || isNaN(c) || isNaN(d) || isNaN(e) || isNaN(f)) {
      console.error("jsPDF.context2d.transform: Invalid arguments", arguments);
      throw new Error("Invalid arguments passed to jsPDF.context2d.transform");
    }
    var matrix = new Matrix(a, b, c, d, e, f);
    this.ctx.transform = this.ctx.transform.multiply(matrix);
  };

  /**
   * Resets the current transform to the identity matrix. Then runs transform()
   *
   * @name setTransform
   * @function
   * @param a {Number} Horizontal scaling
   * @param b {Number} Horizontal skewing
   * @param c {Number} Vertical skewing
   * @param d {Number} Vertical scaling
   * @param e {Number} Horizontal moving
   * @param f {Number} Vertical moving
   * @description Each object on the canvas has a current transformation matrix. <br /><br />The setTransform() method resets the current transform to the identity matrix, and then runs transform() with the same arguments.<br /><br />In other words, the setTransform() method lets you scale, rotate, move, and skew the current context.
   */
  Context2D.prototype.setTransform = function(a, b, c, d, e, f) {
    a = isNaN(a) ? 1 : a;
    b = isNaN(b) ? 0 : b;
    c = isNaN(c) ? 0 : c;
    d = isNaN(d) ? 1 : d;
    e = isNaN(e) ? 0 : e;
    f = isNaN(f) ? 0 : f;
    this.ctx.transform = new Matrix(a, b, c, d, e, f);
  };

  var hasMargins = function() {
    return (
      this.margin[0] > 0 ||
      this.margin[1] > 0 ||
      this.margin[2] > 0 ||
      this.margin[3] > 0
    );
  };

  /**
   * Draws an image, canvas, or video onto the canvas
   *
   * @function
   * @param img {} Specifies the image, canvas, or video element to use
   * @param sx {Number} Optional. The x coordinate where to start clipping
   * @param sy {Number} Optional. The y coordinate where to start clipping
   * @param swidth {Number} Optional. The width of the clipped image
   * @param sheight {Number} Optional. The height of the clipped image
   * @param x {Number} The x coordinate where to place the image on the canvas
   * @param y {Number} The y coordinate where to place the image on the canvas
   * @param width {Number} Optional. The width of the image to use (stretch or reduce the image)
   * @param height {Number} Optional. The height of the image to use (stretch or reduce the image)
   */
  Context2D.prototype.drawImage = function(
    img,
    sx,
    sy,
    swidth,
    sheight,
    x,
    y,
    width,
    height
  ) {
    var imageProperties = this.pdf.getImageProperties(img);
    var factorX = 1;
    var factorY = 1;

    var clipFactorX = 1;
    var clipFactorY = 1;

    if (typeof swidth !== "undefined" && typeof width !== "undefined") {
      clipFactorX = width / swidth;
      clipFactorY = height / sheight;
      factorX = ((imageProperties.width / swidth) * width) / swidth;
      factorY = ((imageProperties.height / sheight) * height) / sheight;
    }

    //is sx and sy are set and x and y not, set x and y with values of sx and sy
    if (typeof x === "undefined") {
      x = sx;
      y = sy;
      sx = 0;
      sy = 0;
    }

    if (typeof swidth !== "undefined" && typeof width === "undefined") {
      width = swidth;
      height = sheight;
    }
    if (typeof swidth === "undefined" && typeof width === "undefined") {
      width = imageProperties.width;
      height = imageProperties.height;
    }

    var decomposedTransformationMatrix = this.ctx.transform.decompose();
    var angle = rad2deg(decomposedTransformationMatrix.rotate.shx);
    var matrix = new Matrix();
    matrix = matrix.multiply(decomposedTransformationMatrix.translate);
    matrix = matrix.multiply(decomposedTransformationMatrix.skew);
    matrix = matrix.multiply(decomposedTransformationMatrix.scale);
    var xRect = matrix.applyToRectangle(
      new Rectangle(
        x - sx * clipFactorX,
        y - sy * clipFactorY,
        swidth * factorX,
        sheight * factorY
      )
    );
    var pageArray = getPagesByPath.call(this, xRect);
    var pages = [];
    for (var ii = 0; ii < pageArray.length; ii += 1) {
      if (pages.indexOf(pageArray[ii]) === -1) {
        pages.push(pageArray[ii]);
      }
    }

    sortPages(pages);

    var clipPath;
    if (this.autoPaging) {
      var min = pages[0];
      var max = pages[pages.length - 1];
      for (var i = min; i < max + 1; i++) {
        this.pdf.setPage(i);

        var pageWidthMinusMargins =
          this.pdf.internal.pageSize.width - this.margin[3] - this.margin[1];
        var topMargin = i === 1 ? this.posY + this.margin[0] : this.margin[0];
        var firstPageHeight =
          this.pdf.internal.pageSize.height -
          this.posY -
          this.margin[0] -
          this.margin[2];
        var pageHeightMinusMargins =
          this.pdf.internal.pageSize.height - this.margin[0] - this.margin[2];
        var previousPageHeightSum =
          i === 1 ? 0 : firstPageHeight + (i - 2) * pageHeightMinusMargins;

        if (this.ctx.clip_path.length !== 0) {
          var tmpPaths = this.path;
          clipPath = JSON.parse(JSON.stringify(this.ctx.clip_path));
          this.path = pathPositionRedo(
            clipPath,
            this.posX + this.margin[3],
            -previousPageHeightSum + topMargin + this.ctx.prevPageLastElemOffset
          );
          drawPaths.call(this, "fill", true);
          this.path = tmpPaths;
        }
        var tmpRect = JSON.parse(JSON.stringify(xRect));
        tmpRect = pathPositionRedo(
          [tmpRect],
          this.posX + this.margin[3],
          -previousPageHeightSum + topMargin + this.ctx.prevPageLastElemOffset
        )[0];

        const needsClipping = (i > min || i < max) && hasMargins.call(this);

        if (needsClipping) {
          this.pdf.saveGraphicsState();
          this.pdf
            .rect(
              this.margin[3],
              this.margin[0],
              pageWidthMinusMargins,
              pageHeightMinusMargins,
              null
            )
            .clip()
            .discardPath();
        }
        this.pdf.addImage(
          img,
          "JPEG",
          tmpRect.x,
          tmpRect.y,
          tmpRect.w,
          tmpRect.h,
          null,
          null,
          angle
        );
        if (needsClipping) {
          this.pdf.restoreGraphicsState();
        }
      }
    } else {
      this.pdf.addImage(
        img,
        "JPEG",
        xRect.x,
        xRect.y,
        xRect.w,
        xRect.h,
        null,
        null,
        angle
      );
    }
  };

  var getPagesByPath = function(path, pageWrapX, pageWrapY) {
    var result = [];
    pageWrapX = pageWrapX || this.pdf.internal.pageSize.width;
    pageWrapY =
      pageWrapY ||
      this.pdf.internal.pageSize.height - this.margin[0] - this.margin[2];
    var yOffset = this.posY + this.ctx.prevPageLastElemOffset;

    switch (path.type) {
      default:
      case "mt":
      case "lt":
        result.push(Math.floor((path.y + yOffset) / pageWrapY) + 1);
        break;
      case "arc":
        result.push(
          Math.floor((path.y + yOffset - path.radius) / pageWrapY) + 1
        );
        result.push(
          Math.floor((path.y + yOffset + path.radius) / pageWrapY) + 1
        );
        break;
      case "qct":
        var rectOfQuadraticCurve = getQuadraticCurveBoundary(
          this.ctx.lastPoint.x,
          this.ctx.lastPoint.y,
          path.x1,
          path.y1,
          path.x,
          path.y
        );
        result.push(
          Math.floor((rectOfQuadraticCurve.y + yOffset) / pageWrapY) + 1
        );
        result.push(
          Math.floor(
            (rectOfQuadraticCurve.y + rectOfQuadraticCurve.h + yOffset) /
              pageWrapY
          ) + 1
        );
        break;
      case "bct":
        var rectOfBezierCurve = getBezierCurveBoundary(
          this.ctx.lastPoint.x,
          this.ctx.lastPoint.y,
          path.x1,
          path.y1,
          path.x2,
          path.y2,
          path.x,
          path.y
        );
        result.push(
          Math.floor((rectOfBezierCurve.y + yOffset) / pageWrapY) + 1
        );
        result.push(
          Math.floor(
            (rectOfBezierCurve.y + rectOfBezierCurve.h + yOffset) / pageWrapY
          ) + 1
        );
        break;
      case "rect":
        result.push(Math.floor((path.y + yOffset) / pageWrapY) + 1);
        result.push(Math.floor((path.y + path.h + yOffset) / pageWrapY) + 1);
    }

    for (var i = 0; i < result.length; i += 1) {
      while (this.pdf.internal.getNumberOfPages() < result[i]) {
        addPage.call(this);
      }
    }
    return result;
  };

  var addPage = function() {
    var fillStyle = this.fillStyle;
    var strokeStyle = this.strokeStyle;
    var font = this.font;
    var lineCap = this.lineCap;
    var lineWidth = this.lineWidth;
    var lineJoin = this.lineJoin;
    this.pdf.addPage();
    this.fillStyle = fillStyle;
    this.strokeStyle = strokeStyle;
    this.font = font;
    this.lineCap = lineCap;
    this.lineWidth = lineWidth;
    this.lineJoin = lineJoin;
  };

  var pathPositionRedo = function(paths, x, y) {
    for (var i = 0; i < paths.length; i++) {
      switch (paths[i].type) {
        case "bct":
          paths[i].x2 += x;
          paths[i].y2 += y;
        case "qct":
          paths[i].x1 += x;
          paths[i].y1 += y;
        case "mt":
        case "lt":
        case "arc":
        default:
          paths[i].x += x;
          paths[i].y += y;
      }
    }
    return paths;
  };

  var sortPages = function(pages) {
    return pages.sort(function(a, b) {
      return a - b;
    });
  };

  var pathPreProcess = function(rule, isClip) {
    var fillStyle = this.fillStyle;
    var strokeStyle = this.strokeStyle;
    var lineCap = this.lineCap;
    var oldLineWidth = this.lineWidth;
    var lineWidth = Math.abs(oldLineWidth * this.ctx.transform.scaleX);
    var lineJoin = this.lineJoin;

    var origPath = JSON.parse(JSON.stringify(this.path));
    var xPath = JSON.parse(JSON.stringify(this.path));
    var clipPath;
    var tmpPath;
    var pages = [];

    for (var i = 0; i < xPath.length; i++) {
      if (typeof xPath[i].x !== "undefined") {
        var page = getPagesByPath.call(this, xPath[i]);

        for (var ii = 0; ii < page.length; ii += 1) {
          if (pages.indexOf(page[ii]) === -1) {
            pages.push(page[ii]);
          }
        }
      }
    }

    for (var j = 0; j < pages.length; j++) {
      while (this.pdf.internal.getNumberOfPages() < pages[j]) {
        addPage.call(this);
      }
    }
    sortPages(pages);

    if (this.autoPaging) {
      var min = pages[0];
      var max = pages[pages.length - 1];
      for (var k = min; k < max + 1; k++) {
        this.pdf.setPage(k);

        this.fillStyle = fillStyle;
        this.strokeStyle = strokeStyle;
        this.lineCap = lineCap;
        this.lineWidth = lineWidth;
        this.lineJoin = lineJoin;

        var pageWidthMinusMargins =
          this.pdf.internal.pageSize.width - this.margin[3] - this.margin[1];
        var topMargin = k === 1 ? this.posY + this.margin[0] : this.margin[0];
        var firstPageHeight =
          this.pdf.internal.pageSize.height -
          this.posY -
          this.margin[0] -
          this.margin[2];
        var pageHeightMinusMargins =
          this.pdf.internal.pageSize.height - this.margin[0] - this.margin[2];
        var previousPageHeightSum =
          k === 1 ? 0 : firstPageHeight + (k - 2) * pageHeightMinusMargins;

        if (this.ctx.clip_path.length !== 0) {
          var tmpPaths = this.path;
          clipPath = JSON.parse(JSON.stringify(this.ctx.clip_path));
          this.path = pathPositionRedo(
            clipPath,
            this.posX + this.margin[3],
            -previousPageHeightSum + topMargin + this.ctx.prevPageLastElemOffset
          );
          drawPaths.call(this, rule, true);
          this.path = tmpPaths;
        }
        tmpPath = JSON.parse(JSON.stringify(origPath));
        this.path = pathPositionRedo(
          tmpPath,
          this.posX + this.margin[3],
          -previousPageHeightSum + topMargin + this.ctx.prevPageLastElemOffset
        );
        if (isClip === false || k === 0) {
          const needsClipping = (k > min || k < max) && hasMargins.call(this);
          if (needsClipping) {
            this.pdf.saveGraphicsState();
            this.pdf
              .rect(
                this.margin[3],
                this.margin[0],
                pageWidthMinusMargins,
                pageHeightMinusMargins,
                null
              )
              .clip()
              .discardPath();
          }
          drawPaths.call(this, rule, isClip);
          if (needsClipping) {
            this.pdf.restoreGraphicsState();
          }
        }
        this.lineWidth = oldLineWidth;
      }
    } else {
      this.lineWidth = lineWidth;
      drawPaths.call(this, rule, isClip);
      this.lineWidth = oldLineWidth;
    }
    this.path = origPath;
  };

  /**
   * Processes the paths
   *
   * @function
   * @param rule {String}
   * @param isClip {Boolean}
   * @private
   * @ignore
   */
  var drawPaths = function(rule, isClip) {
    if (rule === "stroke" && !isClip && isStrokeTransparent.call(this)) {
      return;
    }

    if (rule !== "stroke" && !isClip && isFillTransparent.call(this)) {
      return;
    }

    var moves = [];

    //var alpha = (this.ctx.fillOpacity < 1) ? this.ctx.fillOpacity : this.ctx.globalAlpha;
    var delta;
    var xPath = this.path;
    for (var i = 0; i < xPath.length; i++) {
      var pt = xPath[i];

      switch (pt.type) {
        case "begin":
          moves.push({
            begin: true
          });
          break;

        case "close":
          moves.push({
            close: true
          });
          break;

        case "mt":
          moves.push({
            start: pt,
            deltas: [],
            abs: []
          });
          break;

        case "lt":
          var iii = moves.length;
          if (xPath[i - 1] && !isNaN(xPath[i - 1].x)) {
            delta = [pt.x - xPath[i - 1].x, pt.y - xPath[i - 1].y];
            if (iii > 0) {
              for (iii; iii >= 0; iii--) {
                if (
                  moves[iii - 1].close !== true &&
                  moves[iii - 1].begin !== true
                ) {
                  moves[iii - 1].deltas.push(delta);
                  moves[iii - 1].abs.push(pt);
                  break;
                }
              }
            }
          }
          break;

        case "bct":
          delta = [
            pt.x1 - xPath[i - 1].x,
            pt.y1 - xPath[i - 1].y,
            pt.x2 - xPath[i - 1].x,
            pt.y2 - xPath[i - 1].y,
            pt.x - xPath[i - 1].x,
            pt.y - xPath[i - 1].y
          ];
          moves[moves.length - 1].deltas.push(delta);
          break;

        case "qct":
          var x1 = xPath[i - 1].x + (2.0 / 3.0) * (pt.x1 - xPath[i - 1].x);
          var y1 = xPath[i - 1].y + (2.0 / 3.0) * (pt.y1 - xPath[i - 1].y);
          var x2 = pt.x + (2.0 / 3.0) * (pt.x1 - pt.x);
          var y2 = pt.y + (2.0 / 3.0) * (pt.y1 - pt.y);
          var x3 = pt.x;
          var y3 = pt.y;
          delta = [
            x1 - xPath[i - 1].x,
            y1 - xPath[i - 1].y,
            x2 - xPath[i - 1].x,
            y2 - xPath[i - 1].y,
            x3 - xPath[i - 1].x,
            y3 - xPath[i - 1].y
          ];
          moves[moves.length - 1].deltas.push(delta);
          break;

        case "arc":
          moves.push({
            deltas: [],
            abs: [],
            arc: true
          });

          if (Array.isArray(moves[moves.length - 1].abs)) {
            moves[moves.length - 1].abs.push(pt);
          }
          break;
      }
    }
    var style;
    if (!isClip) {
      if (rule === "stroke") {
        style = "stroke";
      } else {
        style = "fill";
      }
    } else {
      style = null;
    }

    var began = false;
    for (var k = 0; k < moves.length; k++) {
      if (moves[k].arc) {
        var arcs = moves[k].abs;

        for (var ii = 0; ii < arcs.length; ii++) {
          var arc = arcs[ii];

          if (arc.type === "arc") {
            drawArc.call(
              this,
              arc.x,
              arc.y,
              arc.radius,
              arc.startAngle,
              arc.endAngle,
              arc.counterclockwise,
              undefined,
              isClip,
              !began
            );
          } else {
            drawLine.call(this, arc.x, arc.y);
          }
          began = true;
        }
      } else if (moves[k].close === true) {
        this.pdf.internal.out("h");
        began = false;
      } else if (moves[k].begin !== true) {
        var x = moves[k].start.x;
        var y = moves[k].start.y;
        drawLines.call(this, moves[k].deltas, x, y);
        began = true;
      }
    }

    if (style) {
      putStyle.call(this, style);
    }
    if (isClip) {
      doClip.call(this);
    }
  };

  var getBaseline = function(y) {
    var height =
      this.pdf.internal.getFontSize() / this.pdf.internal.scaleFactor;
    var descent = height * (this.pdf.internal.getLineHeightFactor() - 1);
    switch (this.ctx.textBaseline) {
      case "bottom":
        return y - descent;
      case "top":
        return y + height - descent;
      case "hanging":
        return y + height - 2 * descent;
      case "middle":
        return y + height / 2 - descent;
      case "ideographic":
        // TODO not implemented
        return y;
      case "alphabetic":
      default:
        return y;
    }
  };

  var getTextBottom = function(yBaseLine) {
    var height =
      this.pdf.internal.getFontSize() / this.pdf.internal.scaleFactor;
    var descent = height * (this.pdf.internal.getLineHeightFactor() - 1);
    return yBaseLine + descent;
  };

  Context2D.prototype.createLinearGradient = function createLinearGradient() {
    var canvasGradient = function canvasGradient() {};

    canvasGradient.colorStops = [];
    canvasGradient.addColorStop = function(offset, color) {
      this.colorStops.push([offset, color]);
    };

    canvasGradient.getColor = function() {
      if (this.colorStops.length === 0) {
        return "#000000";
      }

      return this.colorStops[0][1];
    };

    canvasGradient.isCanvasGradient = true;
    return canvasGradient;
  };
  Context2D.prototype.createPattern = function createPattern() {
    return this.createLinearGradient();
  };
  Context2D.prototype.createRadialGradient = function createRadialGradient() {
    return this.createLinearGradient();
  };

  /**
   *
   * @param x Edge point X
   * @param y Edge point Y
   * @param r Radius
   * @param a1 start angle
   * @param a2 end angle
   * @param counterclockwise
   * @param style
   * @param isClip
   */
  var drawArc = function(
    x,
    y,
    r,
    a1,
    a2,
    counterclockwise,
    style,
    isClip,
    includeMove
  ) {
    // http://hansmuller-flex.blogspot.com/2011/10/more-about-approximating-circular-arcs.html
    var curves = createArc.call(this, r, a1, a2, counterclockwise);

    for (var i = 0; i < curves.length; i++) {
      var curve = curves[i];
      if (i === 0) {
        if (includeMove) {
          doMove.call(this, curve.x1 + x, curve.y1 + y);
        } else {
          drawLine.call(this, curve.x1 + x, curve.y1 + y);
        }
      }
      drawCurve.call(
        this,
        x,
        y,
        curve.x2,
        curve.y2,
        curve.x3,
        curve.y3,
        curve.x4,
        curve.y4
      );
    }

    if (!isClip) {
      putStyle.call(this, style);
    } else {
      doClip.call(this);
    }
  };

  var putStyle = function(style) {
    switch (style) {
      case "stroke":
        this.pdf.internal.out("S");
        break;
      case "fill":
        this.pdf.internal.out("f");
        break;
    }
  };

  var doClip = function() {
    this.pdf.clip();
    this.pdf.discardPath();
  };

  var doMove = function(x, y) {
    this.pdf.internal.out(
      getHorizontalCoordinateString(x) +
        " " +
        getVerticalCoordinateString(y) +
        " m"
    );
  };

  var putText = function(options) {
    var textAlign;
    switch (options.align) {
      case "right":
      case "end":
        textAlign = "right";
        break;
      case "center":
        textAlign = "center";
        break;
      case "left":
      case "start":
      default:
        textAlign = "left";
        break;
    }

    var textDimensions = this.pdf.getTextDimensions(options.text);
    var yBaseLine = getBaseline.call(this, options.y);
    var yBottom = getTextBottom.call(this, yBaseLine);
    var yTop = yBottom - textDimensions.h;

    var pt = this.ctx.transform.applyToPoint(new Point(options.x, yBaseLine));
    var decomposedTransformationMatrix = this.ctx.transform.decompose();
    var matrix = new Matrix();
    matrix = matrix.multiply(decomposedTransformationMatrix.translate);
    matrix = matrix.multiply(decomposedTransformationMatrix.skew);
    matrix = matrix.multiply(decomposedTransformationMatrix.scale);

    var baselineRect = this.ctx.transform.applyToRectangle(
      new Rectangle(options.x, yBaseLine, textDimensions.w, textDimensions.h)
    );
    var textBounds = matrix.applyToRectangle(
      new Rectangle(options.x, yTop, textDimensions.w, textDimensions.h)
    );
    var pageArray = getPagesByPath.call(this, textBounds);
    var pages = [];
    for (var ii = 0; ii < pageArray.length; ii += 1) {
      if (pages.indexOf(pageArray[ii]) === -1) {
        pages.push(pageArray[ii]);
      }
    }

    sortPages(pages);

    var clipPath, oldSize, oldLineWidth;
    if (this.autoPaging) {
      var min = pages[0];
      var max = pages[pages.length - 1];
      for (var i = min; i < max + 1; i++) {
        this.pdf.setPage(i);

        var topMargin = i === 1 ? this.posY + this.margin[0] : this.margin[0];
        var firstPageHeight =
          this.pdf.internal.pageSize.height -
          this.posY -
          this.margin[0] -
          this.margin[2];
        var pageHeightMinusBottomMargin =
          this.pdf.internal.pageSize.height - this.margin[2];
        var pageHeightMinusMargins =
          pageHeightMinusBottomMargin - this.margin[0];
        var pageWidthMinusRightMargin =
          this.pdf.internal.pageSize.width - this.margin[1];
        var pageWidthMinusMargins = pageWidthMinusRightMargin - this.margin[3];
        var previousPageHeightSum =
          i === 1 ? 0 : firstPageHeight + (i - 2) * pageHeightMinusMargins;

        if (this.ctx.clip_path.length !== 0) {
          var tmpPaths = this.path;
          clipPath = JSON.parse(JSON.stringify(this.ctx.clip_path));
          this.path = pathPositionRedo(
            clipPath,
            this.posX + this.margin[3],
            -1 * previousPageHeightSum + topMargin
          );
          drawPaths.call(this, "fill", true);
          this.path = tmpPaths;
        }
        var textBoundsOnPage = pathPositionRedo(
          [JSON.parse(JSON.stringify(textBounds))],
          this.posX + this.margin[3],
          -previousPageHeightSum + topMargin + this.ctx.prevPageLastElemOffset
        )[0];

        if (options.scale >= 0.01) {
          oldSize = this.pdf.internal.getFontSize();
          this.pdf.setFontSize(oldSize * options.scale);
          oldLineWidth = this.lineWidth;
          this.lineWidth = oldLineWidth * options.scale;
        }

        var doSlice = this.autoPaging !== "text";

        if (
          doSlice ||
          textBoundsOnPage.y + textBoundsOnPage.h <= pageHeightMinusBottomMargin
        ) {
          if (
            doSlice ||
            (textBoundsOnPage.y >= topMargin &&
              textBoundsOnPage.x <= pageWidthMinusRightMargin)
          ) {
            var croppedText = doSlice
              ? options.text
              : this.pdf.splitTextToSize(
                  options.text,
                  options.maxWidth ||
                    pageWidthMinusRightMargin - textBoundsOnPage.x
                )[0];
            var baseLineRectOnPage = pathPositionRedo(
              [JSON.parse(JSON.stringify(baselineRect))],
              this.posX + this.margin[3],
              -previousPageHeightSum +
                topMargin +
                this.ctx.prevPageLastElemOffset
            )[0];

            const needsClipping =
              doSlice && (i > min || i < max) && hasMargins.call(this);

            if (needsClipping) {
              this.pdf.saveGraphicsState();
              this.pdf
                .rect(
                  this.margin[3],
                  this.margin[0],
                  pageWidthMinusMargins,
                  pageHeightMinusMargins,
                  null
                )
                .clip()
                .discardPath();
            }

            this.pdf.text(
              croppedText,
              baseLineRectOnPage.x,
              baseLineRectOnPage.y,
              {
                angle: options.angle,
                align: textAlign,
                renderingMode: options.renderingMode
              }
            );

            if (needsClipping) {
              this.pdf.restoreGraphicsState();
            }
          }
        } else {
          // This text is the last element of the page, but it got cut off due to the margin
          // so we render it in the next page

          if (textBoundsOnPage.y < pageHeightMinusBottomMargin) {
            // As a result, all other elements have their y offset increased
            this.ctx.prevPageLastElemOffset +=
              pageHeightMinusBottomMargin - textBoundsOnPage.y;
          }
        }

        if (options.scale >= 0.01) {
          this.pdf.setFontSize(oldSize);
          this.lineWidth = oldLineWidth;
        }
      }
    } else {
      if (options.scale >= 0.01) {
        oldSize = this.pdf.internal.getFontSize();
        this.pdf.setFontSize(oldSize * options.scale);
        oldLineWidth = this.lineWidth;
        this.lineWidth = oldLineWidth * options.scale;
      }
      this.pdf.text(options.text, pt.x + this.posX, pt.y + this.posY, {
        angle: options.angle,
        align: textAlign,
        renderingMode: options.renderingMode,
        maxWidth: options.maxWidth
      });

      if (options.scale >= 0.01) {
        this.pdf.setFontSize(oldSize);
        this.lineWidth = oldLineWidth;
      }
    }
  };

  var drawLine = function(x, y, prevX, prevY) {
    prevX = prevX || 0;
    prevY = prevY || 0;

    this.pdf.internal.out(
      getHorizontalCoordinateString(x + prevX) +
        " " +
        getVerticalCoordinateString(y + prevY) +
        " l"
    );
  };

  var drawLines = function(lines, x, y) {
    return this.pdf.lines(lines, x, y, null, null);
  };

  var drawCurve = function(x, y, x1, y1, x2, y2, x3, y3) {
    this.pdf.internal.out(
      [
        f2(getHorizontalCoordinate(x1 + x)),
        f2(getVerticalCoordinate(y1 + y)),
        f2(getHorizontalCoordinate(x2 + x)),
        f2(getVerticalCoordinate(y2 + y)),
        f2(getHorizontalCoordinate(x3 + x)),
        f2(getVerticalCoordinate(y3 + y)),
        "c"
      ].join(" ")
    );
  };

  /**
   * Return a array of objects that represent bezier curves which approximate the circular arc centered at the origin, from startAngle to endAngle (radians) with the specified radius.
   *
   * Each bezier curve is an object with four points, where x1,y1 and x4,y4 are the arc's end points and x2,y2 and x3,y3 are the cubic bezier's control points.
   * @function createArc
   */
  var createArc = function(radius, startAngle, endAngle, anticlockwise) {
    var EPSILON = 0.00001; // Roughly 1/1000th of a degree, see below
    var twoPi = Math.PI * 2;
    var halfPi = Math.PI / 2.0;

    while (startAngle > endAngle) {
      startAngle = startAngle - twoPi;
    }
    var totalAngle = Math.abs(endAngle - startAngle);
    if (totalAngle < twoPi) {
      if (anticlockwise) {
        totalAngle = twoPi - totalAngle;
      }
    }

    // Compute the sequence of arc curves, up to PI/2 at a time.
    var curves = [];

    // clockwise or counterclockwise
    var sgn = anticlockwise ? -1 : +1;

    var a1 = startAngle;
    for (; totalAngle > EPSILON; ) {
      var remain = sgn * Math.min(totalAngle, halfPi);
      var a2 = a1 + remain;
      curves.push(createSmallArc.call(this, radius, a1, a2));
      totalAngle -= Math.abs(a2 - a1);
      a1 = a2;
    }

    return curves;
  };

  /**
   * Cubic bezier approximation of a circular arc centered at the origin, from (radians) a1 to a2, where a2-a1 < pi/2. The arc's radius is r.
   *
   * Returns an object with four points, where x1,y1 and x4,y4 are the arc's end points and x2,y2 and x3,y3 are the cubic bezier's control points.
   *
   * This algorithm is based on the approach described in: A. Riškus, "Approximation of a Cubic Bezier Curve by Circular Arcs and Vice Versa," Information Technology and Control, 35(4), 2006 pp. 371-378.
   */
  var createSmallArc = function(r, a1, a2) {
    var a = (a2 - a1) / 2.0;

    var x4 = r * Math.cos(a);
    var y4 = r * Math.sin(a);
    var x1 = x4;
    var y1 = -y4;

    var q1 = x1 * x1 + y1 * y1;
    var q2 = q1 + x1 * x4 + y1 * y4;
    var k2 = ((4 / 3) * (Math.sqrt(2 * q1 * q2) - q2)) / (x1 * y4 - y1 * x4);

    var x2 = x1 - k2 * y1;
    var y2 = y1 + k2 * x1;
    var x3 = x2;
    var y3 = -y2;

    var ar = a + a1;
    var cos_ar = Math.cos(ar);
    var sin_ar = Math.sin(ar);

    return {
      x1: r * Math.cos(a1),
      y1: r * Math.sin(a1),
      x2: x2 * cos_ar - y2 * sin_ar,
      y2: x2 * sin_ar + y2 * cos_ar,
      x3: x3 * cos_ar - y3 * sin_ar,
      y3: x3 * sin_ar + y3 * cos_ar,
      x4: r * Math.cos(a2),
      y4: r * Math.sin(a2)
    };
  };

  var rad2deg = function(value) {
    return (value * 180) / Math.PI;
  };

  var getQuadraticCurveBoundary = function(sx, sy, cpx, cpy, ex, ey) {
    var midX1 = sx + (cpx - sx) * 0.5;
    var midY1 = sy + (cpy - sy) * 0.5;
    var midX2 = ex + (cpx - ex) * 0.5;
    var midY2 = ey + (cpy - ey) * 0.5;
    var resultX1 = Math.min(sx, ex, midX1, midX2);
    var resultX2 = Math.max(sx, ex, midX1, midX2);
    var resultY1 = Math.min(sy, ey, midY1, midY2);
    var resultY2 = Math.max(sy, ey, midY1, midY2);
    return new Rectangle(
      resultX1,
      resultY1,
      resultX2 - resultX1,
      resultY2 - resultY1
    );
  };

  //De Casteljau algorithm
  var getBezierCurveBoundary = function(ax, ay, bx, by, cx, cy, dx, dy) {
    var tobx = bx - ax;
    var toby = by - ay;
    var tocx = cx - bx;
    var tocy = cy - by;
    var todx = dx - cx;
    var tody = dy - cy;
    var precision = 40;
    var d,
      i,
      px,
      py,
      qx,
      qy,
      rx,
      ry,
      tx,
      ty,
      sx,
      sy,
      x,
      y,
      minx,
      miny,
      maxx,
      maxy,
      toqx,
      toqy,
      torx,
      tory,
      totx,
      toty;
    for (i = 0; i < precision + 1; i++) {
      d = i / precision;
      px = ax + d * tobx;
      py = ay + d * toby;
      qx = bx + d * tocx;
      qy = by + d * tocy;
      rx = cx + d * todx;
      ry = cy + d * tody;
      toqx = qx - px;
      toqy = qy - py;
      torx = rx - qx;
      tory = ry - qy;

      sx = px + d * toqx;
      sy = py + d * toqy;
      tx = qx + d * torx;
      ty = qy + d * tory;
      totx = tx - sx;
      toty = ty - sy;

      x = sx + d * totx;
      y = sy + d * toty;
      if (i == 0) {
        minx = x;
        miny = y;
        maxx = x;
        maxy = y;
      } else {
        minx = Math.min(minx, x);
        miny = Math.min(miny, y);
        maxx = Math.max(maxx, x);
        maxy = Math.max(maxy, y);
      }
    }
    return new Rectangle(
      Math.round(minx),
      Math.round(miny),
      Math.round(maxx - minx),
      Math.round(maxy - miny)
    );
  };

  var getPrevLineDashValue = function(lineDash, lineDashOffset) {
    return JSON.stringify({
      lineDash: lineDash,
      lineDashOffset: lineDashOffset
    });
  };

  var setLineDash = function() {
    // Avoid unnecessary line dash declarations.
    if (
      !this.prevLineDash &&
      !this.ctx.lineDash.length &&
      !this.ctx.lineDashOffset
    ) {
      return;
    }

    // Avoid unnecessary line dash declarations.
    const nextLineDash = getPrevLineDashValue(
      this.ctx.lineDash,
      this.ctx.lineDashOffset
    );
    if (this.prevLineDash !== nextLineDash) {
      this.pdf.setLineDash(this.ctx.lineDash, this.ctx.lineDashOffset);
      this.prevLineDash = nextLineDash;
    }
  };
})(jsPDF.API);

// DEFLATE is a complex format; to read this code, you should probably check the RFC first:

// aliases for shorter compressed code (most minifers don't do this)
var u8 = Uint8Array, u16 = Uint16Array, i32 = Int32Array;
// fixed length extra bits
var fleb = new u8([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, /* unused */ 0, 0, /* impossible */ 0]);
// fixed distance extra bits
var fdeb = new u8([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, /* unused */ 0, 0]);
// code length index map
var clim = new u8([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]);
// get base, reverse index map from extra bits
var freb = function (eb, start) {
    var b = new u16(31);
    for (var i = 0; i < 31; ++i) {
        b[i] = start += 1 << eb[i - 1];
    }
    // numbers here are at max 18 bits
    var r = new i32(b[30]);
    for (var i = 1; i < 30; ++i) {
        for (var j = b[i]; j < b[i + 1]; ++j) {
            r[j] = ((j - b[i]) << 5) | i;
        }
    }
    return { b: b, r: r };
};
var _a = freb(fleb, 2), fl = _a.b, revfl = _a.r;
// we can ignore the fact that the other numbers are wrong; they never happen anyway
fl[28] = 258, revfl[258] = 28;
var _b = freb(fdeb, 0), revfd = _b.r;
// map of value to reverse (assuming 16 bits)
var rev = new u16(32768);
for (var i = 0; i < 32768; ++i) {
    // reverse table algorithm from SO
    var x = ((i & 0xAAAA) >> 1) | ((i & 0x5555) << 1);
    x = ((x & 0xCCCC) >> 2) | ((x & 0x3333) << 2);
    x = ((x & 0xF0F0) >> 4) | ((x & 0x0F0F) << 4);
    rev[i] = (((x & 0xFF00) >> 8) | ((x & 0x00FF) << 8)) >> 1;
}
// create huffman tree from u8 "map": index -> code length for code index
// mb (max bits) must be at most 15
// TODO: optimize/split up?
var hMap = (function (cd, mb, r) {
    var s = cd.length;
    // index
    var i = 0;
    // u16 "map": index -> # of codes with bit length = index
    var l = new u16(mb);
    // length of cd must be 288 (total # of codes)
    for (; i < s; ++i) {
        if (cd[i])
            ++l[cd[i] - 1];
    }
    // u16 "map": index -> minimum code for bit length = index
    var le = new u16(mb);
    for (i = 1; i < mb; ++i) {
        le[i] = (le[i - 1] + l[i - 1]) << 1;
    }
    var co;
    if (r) {
        // u16 "map": index -> number of actual bits, symbol for code
        co = new u16(1 << mb);
        // bits to remove for reverser
        var rvb = 15 - mb;
        for (i = 0; i < s; ++i) {
            // ignore 0 lengths
            if (cd[i]) {
                // num encoding both symbol and bits read
                var sv = (i << 4) | cd[i];
                // free bits
                var r_1 = mb - cd[i];
                // start value
                var v = le[cd[i] - 1]++ << r_1;
                // m is end value
                for (var m = v | ((1 << r_1) - 1); v <= m; ++v) {
                    // every 16 bit value starting with the code yields the same result
                    co[rev[v] >> rvb] = sv;
                }
            }
        }
    }
    else {
        co = new u16(s);
        for (i = 0; i < s; ++i) {
            if (cd[i]) {
                co[i] = rev[le[cd[i] - 1]++] >> (15 - cd[i]);
            }
        }
    }
    return co;
});
// fixed length tree
var flt = new u8(288);
for (var i = 0; i < 144; ++i)
    flt[i] = 8;
for (var i = 144; i < 256; ++i)
    flt[i] = 9;
for (var i = 256; i < 280; ++i)
    flt[i] = 7;
for (var i = 280; i < 288; ++i)
    flt[i] = 8;
// fixed distance tree
var fdt = new u8(32);
for (var i = 0; i < 32; ++i)
    fdt[i] = 5;
// fixed length map
var flm = /*#__PURE__*/ hMap(flt, 9, 0);
// fixed distance map
var fdm = /*#__PURE__*/ hMap(fdt, 5, 0);
// get end of byte
var shft = function (p) { return ((p + 7) / 8) | 0; };
// typed array slice - allows garbage collector to free original reference,
// while being more compatible than .slice
var slc = function (v, s, e) {
    if (s == null || s < 0)
        s = 0;
    if (e == null || e > v.length)
        e = v.length;
    // can't use .constructor in case user-supplied
    return new u8(v.subarray(s, e));
};
// starting at p, write the minimum number of bits that can hold v to d
var wbits = function (d, p, v) {
    v <<= p & 7;
    var o = (p / 8) | 0;
    d[o] |= v;
    d[o + 1] |= v >> 8;
};
// starting at p, write the minimum number of bits (>8) that can hold v to d
var wbits16 = function (d, p, v) {
    v <<= p & 7;
    var o = (p / 8) | 0;
    d[o] |= v;
    d[o + 1] |= v >> 8;
    d[o + 2] |= v >> 16;
};
// creates code lengths from a frequency table
var hTree = function (d, mb) {
    // Need extra info to make a tree
    var t = [];
    for (var i = 0; i < d.length; ++i) {
        if (d[i])
            t.push({ s: i, f: d[i] });
    }
    var s = t.length;
    var t2 = t.slice();
    if (!s)
        return { t: et, l: 0 };
    if (s == 1) {
        var v = new u8(t[0].s + 1);
        v[t[0].s] = 1;
        return { t: v, l: 1 };
    }
    t.sort(function (a, b) { return a.f - b.f; });
    // after i2 reaches last ind, will be stopped
    // freq must be greater than largest possible number of symbols
    t.push({ s: -1, f: 25001 });
    var l = t[0], r = t[1], i0 = 0, i1 = 1, i2 = 2;
    t[0] = { s: -1, f: l.f + r.f, l: l, r: r };
    // efficient algorithm from UZIP.js
    // i0 is lookbehind, i2 is lookahead - after processing two low-freq
    // symbols that combined have high freq, will start processing i2 (high-freq,
    // non-composite) symbols instead
    // see https://reddit.com/r/photopea/comments/ikekht/uzipjs_questions/
    while (i1 != s - 1) {
        l = t[t[i0].f < t[i2].f ? i0++ : i2++];
        r = t[i0 != i1 && t[i0].f < t[i2].f ? i0++ : i2++];
        t[i1++] = { s: -1, f: l.f + r.f, l: l, r: r };
    }
    var maxSym = t2[0].s;
    for (var i = 1; i < s; ++i) {
        if (t2[i].s > maxSym)
            maxSym = t2[i].s;
    }
    // code lengths
    var tr = new u16(maxSym + 1);
    // max bits in tree
    var mbt = ln(t[i1 - 1], tr, 0);
    if (mbt > mb) {
        // more algorithms from UZIP.js
        // TODO: find out how this code works (debt)
        //  ind    debt
        var i = 0, dt = 0;
        //    left            cost
        var lft = mbt - mb, cst = 1 << lft;
        t2.sort(function (a, b) { return tr[b.s] - tr[a.s] || a.f - b.f; });
        for (; i < s; ++i) {
            var i2_1 = t2[i].s;
            if (tr[i2_1] > mb) {
                dt += cst - (1 << (mbt - tr[i2_1]));
                tr[i2_1] = mb;
            }
            else
                break;
        }
        dt >>= lft;
        while (dt > 0) {
            var i2_2 = t2[i].s;
            if (tr[i2_2] < mb)
                dt -= 1 << (mb - tr[i2_2]++ - 1);
            else
                ++i;
        }
        for (; i >= 0 && dt; --i) {
            var i2_3 = t2[i].s;
            if (tr[i2_3] == mb) {
                --tr[i2_3];
                ++dt;
            }
        }
        mbt = mb;
    }
    return { t: new u8(tr), l: mbt };
};
// get the max length and assign length codes
var ln = function (n, l, d) {
    return n.s == -1
        ? Math.max(ln(n.l, l, d + 1), ln(n.r, l, d + 1))
        : (l[n.s] = d);
};
// length codes generation
var lc = function (c) {
    var s = c.length;
    // Note that the semicolon was intentional
    while (s && !c[--s])
        ;
    var cl = new u16(++s);
    //  ind      num         streak
    var cli = 0, cln = c[0], cls = 1;
    var w = function (v) { cl[cli++] = v; };
    for (var i = 1; i <= s; ++i) {
        if (c[i] == cln && i != s)
            ++cls;
        else {
            if (!cln && cls > 2) {
                for (; cls > 138; cls -= 138)
                    w(32754);
                if (cls > 2) {
                    w(cls > 10 ? ((cls - 11) << 5) | 28690 : ((cls - 3) << 5) | 12305);
                    cls = 0;
                }
            }
            else if (cls > 3) {
                w(cln), --cls;
                for (; cls > 6; cls -= 6)
                    w(8304);
                if (cls > 2)
                    w(((cls - 3) << 5) | 8208), cls = 0;
            }
            while (cls--)
                w(cln);
            cls = 1;
            cln = c[i];
        }
    }
    return { c: cl.subarray(0, cli), n: s };
};
// calculate the length of output from tree, code lengths
var clen = function (cf, cl) {
    var l = 0;
    for (var i = 0; i < cl.length; ++i)
        l += cf[i] * cl[i];
    return l;
};
// writes a fixed block
// returns the new bit pos
var wfblk = function (out, pos, dat) {
    // no need to write 00 as type: TypedArray defaults to 0
    var s = dat.length;
    var o = shft(pos + 2);
    out[o] = s & 255;
    out[o + 1] = s >> 8;
    out[o + 2] = out[o] ^ 255;
    out[o + 3] = out[o + 1] ^ 255;
    for (var i = 0; i < s; ++i)
        out[o + i + 4] = dat[i];
    return (o + 4 + s) * 8;
};
// writes a block
var wblk = function (dat, out, final, syms, lf, df, eb, li, bs, bl, p) {
    wbits(out, p++, final);
    ++lf[256];
    var _a = hTree(lf, 15), dlt = _a.t, mlb = _a.l;
    var _b = hTree(df, 15), ddt = _b.t, mdb = _b.l;
    var _c = lc(dlt), lclt = _c.c, nlc = _c.n;
    var _d = lc(ddt), lcdt = _d.c, ndc = _d.n;
    var lcfreq = new u16(19);
    for (var i = 0; i < lclt.length; ++i)
        ++lcfreq[lclt[i] & 31];
    for (var i = 0; i < lcdt.length; ++i)
        ++lcfreq[lcdt[i] & 31];
    var _e = hTree(lcfreq, 7), lct = _e.t, mlcb = _e.l;
    var nlcc = 19;
    for (; nlcc > 4 && !lct[clim[nlcc - 1]]; --nlcc)
        ;
    var flen = (bl + 5) << 3;
    var ftlen = clen(lf, flt) + clen(df, fdt) + eb;
    var dtlen = clen(lf, dlt) + clen(df, ddt) + eb + 14 + 3 * nlcc + clen(lcfreq, lct) + 2 * lcfreq[16] + 3 * lcfreq[17] + 7 * lcfreq[18];
    if (bs >= 0 && flen <= ftlen && flen <= dtlen)
        return wfblk(out, p, dat.subarray(bs, bs + bl));
    var lm, ll, dm, dl;
    wbits(out, p, 1 + (dtlen < ftlen)), p += 2;
    if (dtlen < ftlen) {
        lm = hMap(dlt, mlb, 0), ll = dlt, dm = hMap(ddt, mdb, 0), dl = ddt;
        var llm = hMap(lct, mlcb, 0);
        wbits(out, p, nlc - 257);
        wbits(out, p + 5, ndc - 1);
        wbits(out, p + 10, nlcc - 4);
        p += 14;
        for (var i = 0; i < nlcc; ++i)
            wbits(out, p + 3 * i, lct[clim[i]]);
        p += 3 * nlcc;
        var lcts = [lclt, lcdt];
        for (var it = 0; it < 2; ++it) {
            var clct = lcts[it];
            for (var i = 0; i < clct.length; ++i) {
                var len = clct[i] & 31;
                wbits(out, p, llm[len]), p += lct[len];
                if (len > 15)
                    wbits(out, p, (clct[i] >> 5) & 127), p += clct[i] >> 12;
            }
        }
    }
    else {
        lm = flm, ll = flt, dm = fdm, dl = fdt;
    }
    for (var i = 0; i < li; ++i) {
        var sym = syms[i];
        if (sym > 255) {
            var len = (sym >> 18) & 31;
            wbits16(out, p, lm[len + 257]), p += ll[len + 257];
            if (len > 7)
                wbits(out, p, (sym >> 23) & 31), p += fleb[len];
            var dst = sym & 31;
            wbits16(out, p, dm[dst]), p += dl[dst];
            if (dst > 3)
                wbits16(out, p, (sym >> 5) & 8191), p += fdeb[dst];
        }
        else {
            wbits16(out, p, lm[sym]), p += ll[sym];
        }
    }
    wbits16(out, p, lm[256]);
    return p + ll[256];
};
// deflate options (nice << 13) | chain
var deo = /*#__PURE__*/ new i32([65540, 131080, 131088, 131104, 262176, 1048704, 1048832, 2114560, 2117632]);
// empty
var et = /*#__PURE__*/ new u8(0);
// compresses data into a raw DEFLATE buffer
var dflt = function (dat, lvl, plvl, pre, post, st) {
    var s = st.z || dat.length;
    var o = new u8(pre + s + 5 * (1 + Math.ceil(s / 7000)) + post);
    // writing to this writes to the output buffer
    var w = o.subarray(pre, o.length - post);
    var lst = st.l;
    var pos = (st.r || 0) & 7;
    if (lvl) {
        if (pos)
            w[0] = st.r >> 3;
        var opt = deo[lvl - 1];
        var n = opt >> 13, c = opt & 8191;
        var msk_1 = (1 << plvl) - 1;
        //    prev 2-byte val map    curr 2-byte val map
        var prev = st.p || new u16(32768), head = st.h || new u16(msk_1 + 1);
        var bs1_1 = Math.ceil(plvl / 3), bs2_1 = 2 * bs1_1;
        var hsh = function (i) { return (dat[i] ^ (dat[i + 1] << bs1_1) ^ (dat[i + 2] << bs2_1)) & msk_1; };
        // 24576 is an arbitrary number of maximum symbols per block
        // 424 buffer for last block
        var syms = new i32(25000);
        // length/literal freq   distance freq
        var lf = new u16(288), df = new u16(32);
        //  l/lcnt  exbits  index          l/lind  waitdx          blkpos
        var lc_1 = 0, eb = 0, i = st.i || 0, li = 0, wi = st.w || 0, bs = 0;
        for (; i + 2 < s; ++i) {
            // hash value
            var hv = hsh(i);
            // index mod 32768    previous index mod
            var imod = i & 32767, pimod = head[hv];
            prev[imod] = pimod;
            head[hv] = imod;
            // We always should modify head and prev, but only add symbols if
            // this data is not yet processed ("wait" for wait index)
            if (wi <= i) {
                // bytes remaining
                var rem = s - i;
                if ((lc_1 > 7000 || li > 24576) && (rem > 423 || !lst)) {
                    pos = wblk(dat, w, 0, syms, lf, df, eb, li, bs, i - bs, pos);
                    li = lc_1 = eb = 0, bs = i;
                    for (var j = 0; j < 286; ++j)
                        lf[j] = 0;
                    for (var j = 0; j < 30; ++j)
                        df[j] = 0;
                }
                //  len    dist   chain
                var l = 2, d = 0, ch_1 = c, dif = imod - pimod & 32767;
                if (rem > 2 && hv == hsh(i - dif)) {
                    var maxn = Math.min(n, rem) - 1;
                    var maxd = Math.min(32767, i);
                    // max possible length
                    // not capped at dif because decompressors implement "rolling" index population
                    var ml = Math.min(258, rem);
                    while (dif <= maxd && --ch_1 && imod != pimod) {
                        if (dat[i + l] == dat[i + l - dif]) {
                            var nl = 0;
                            for (; nl < ml && dat[i + nl] == dat[i + nl - dif]; ++nl)
                                ;
                            if (nl > l) {
                                l = nl, d = dif;
                                // break out early when we reach "nice" (we are satisfied enough)
                                if (nl > maxn)
                                    break;
                                // now, find the rarest 2-byte sequence within this
                                // length of literals and search for that instead.
                                // Much faster than just using the start
                                var mmd = Math.min(dif, nl - 2);
                                var md = 0;
                                for (var j = 0; j < mmd; ++j) {
                                    var ti = i - dif + j & 32767;
                                    var pti = prev[ti];
                                    var cd = ti - pti & 32767;
                                    if (cd > md)
                                        md = cd, pimod = ti;
                                }
                            }
                        }
                        // check the previous match
                        imod = pimod, pimod = prev[imod];
                        dif += imod - pimod & 32767;
                    }
                }
                // d will be nonzero only when a match was found
                if (d) {
                    // store both dist and len data in one int32
                    // Make sure this is recognized as a len/dist with 28th bit (2^28)
                    syms[li++] = 268435456 | (revfl[l] << 18) | revfd[d];
                    var lin = revfl[l] & 31, din = revfd[d] & 31;
                    eb += fleb[lin] + fdeb[din];
                    ++lf[257 + lin];
                    ++df[din];
                    wi = i + l;
                    ++lc_1;
                }
                else {
                    syms[li++] = dat[i];
                    ++lf[dat[i]];
                }
            }
        }
        for (i = Math.max(i, wi); i < s; ++i) {
            syms[li++] = dat[i];
            ++lf[dat[i]];
        }
        pos = wblk(dat, w, lst, syms, lf, df, eb, li, bs, i - bs, pos);
        if (!lst) {
            st.r = (pos & 7) | w[(pos / 8) | 0] << 3;
            // shft(pos) now 1 less if pos & 7 != 0
            pos -= 7;
            st.h = head, st.p = prev, st.i = i, st.w = wi;
        }
    }
    else {
        for (var i = st.w || 0; i < s + lst; i += 65535) {
            // end
            var e = i + 65535;
            if (e >= s) {
                // write final block
                w[(pos / 8) | 0] = lst;
                e = s;
            }
            pos = wfblk(w, pos + 1, dat.subarray(i, e));
        }
        st.i = s;
    }
    return slc(o, 0, pre + shft(pos) + post);
};
// Adler32
var adler = function () {
    var a = 1, b = 0;
    return {
        p: function (d) {
            // closures have awful performance
            var n = a, m = b;
            var l = d.length | 0;
            for (var i = 0; i != l;) {
                var e = Math.min(i + 2655, l);
                for (; i < e; ++i)
                    m += n += d[i];
                n = (n & 65535) + 15 * (n >> 16), m = (m & 65535) + 15 * (m >> 16);
            }
            a = n, b = m;
        },
        d: function () {
            a %= 65521, b %= 65521;
            return (a & 255) << 24 | (a & 0xFF00) << 8 | (b & 255) << 8 | (b >> 8);
        }
    };
};
// deflate with opts
var dopt = function (dat, opt, pre, post, st) {
    if (!st) {
        st = { l: 1 };
        if (opt.dictionary) {
            var dict = opt.dictionary.subarray(-32768);
            var newDat = new u8(dict.length + dat.length);
            newDat.set(dict);
            newDat.set(dat, dict.length);
            dat = newDat;
            st.w = dict.length;
        }
    }
    return dflt(dat, opt.level == null ? 6 : opt.level, opt.mem == null ? Math.ceil(Math.max(8, Math.min(13, Math.log(dat.length))) * 1.5) : (12 + opt.mem), pre, post, st);
};
// write bytes
var wbytes = function (d, b, v) {
    for (; v; ++b)
        d[b] = v, v >>>= 8;
};
// zlib header
var zlh = function (c, o) {
    var lv = o.level, fl = lv == 0 ? 0 : lv < 6 ? 1 : lv == 9 ? 3 : 2;
    c[0] = 120, c[1] = (fl << 6) | (o.dictionary && 32);
    c[1] |= 31 - ((c[0] << 8) | c[1]) % 31;
    if (o.dictionary) {
        var h = adler();
        h.p(o.dictionary);
        wbytes(c, 2, h.d());
    }
};
/**
 * Compress data with Zlib
 * @param data The data to compress
 * @param opts The compression options
 * @returns The zlib-compressed version of the data
 */
function zlibSync(data, opts) {
    if (!opts)
        opts = {};
    var a = adler();
    a.p(data);
    var d = dopt(data, opts, opts.dictionary ? 6 : 2, 4);
    return zlh(d, opts), wbytes(d, d.length - 4, a.d()), d;
}
// text decoder
var td = typeof TextDecoder != 'undefined' && /*#__PURE__*/ new TextDecoder();
// text decoder stream
var tds = 0;
try {
    td.decode(et, { stream: true });
    tds = 1;
}
catch (e) { }

/**
 * @license
 * jsPDF filters PlugIn
 * Copyright (c) 2014 Aras Abbasi
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

(function(jsPDFAPI) {

  var ASCII85Encode = function(a) {
    var b, c, d, e, f, g, h, i, j, k;
    // eslint-disable-next-line no-control-regex
    for (
      !/[^\x00-\xFF]/.test(a),
        b = "\x00\x00\x00\x00".slice(a.length % 4 || 4),
        a += b,
        c = [],
        d = 0,
        e = a.length;
      e > d;
      d += 4
    )
      (f =
        (a.charCodeAt(d) << 24) +
        (a.charCodeAt(d + 1) << 16) +
        (a.charCodeAt(d + 2) << 8) +
        a.charCodeAt(d + 3)),
        0 !== f
          ? ((k = f % 85),
            (f = (f - k) / 85),
            (j = f % 85),
            (f = (f - j) / 85),
            (i = f % 85),
            (f = (f - i) / 85),
            (h = f % 85),
            (f = (f - h) / 85),
            (g = f % 85),
            c.push(g + 33, h + 33, i + 33, j + 33, k + 33))
          : c.push(122);
    return (
      (function(a, b) {
        for (var c = b; c > 0; c--) a.pop();
      })(c, b.length),
      String.fromCharCode.apply(String, c) + "~>"
    );
  };

  var ASCII85Decode = function(a) {
    var c,
      d,
      e,
      f,
      g,
      h = String,
      l = "length",
      w = 255,
      x = "charCodeAt",
      y = "slice",
      z = "replace";
    for (
      "~>" === a[y](-2),
        a = a[y](0, -2)
          [z](/\s/g, "")
          [z]("z", "!!!!!"),
        c = "uuuuu"[y](a[l] % 5 || 5),
        a += c,
        e = [],
        f = 0,
        g = a[l];
      g > f;
      f += 5
    )
      (d =
        52200625 * (a[x](f) - 33) +
        614125 * (a[x](f + 1) - 33) +
        7225 * (a[x](f + 2) - 33) +
        85 * (a[x](f + 3) - 33) +
        (a[x](f + 4) - 33)),
        e.push(w & (d >> 24), w & (d >> 16), w & (d >> 8), w & d);
    return (
      (function(a, b) {
        for (var c = b; c > 0; c--) a.pop();
      })(e, c[l]),
      h.fromCharCode.apply(h, e)
    );
  };

  var ASCIIHexEncode = function(value) {
    return (
      value
        .split("")
        .map(function(value) {
          return ("0" + value.charCodeAt().toString(16)).slice(-2);
        })
        .join("") + ">"
    );
  };

  var ASCIIHexDecode = function(value) {
    var regexCheckIfHex = new RegExp(/^([0-9A-Fa-f]{2})+$/);
    value = value.replace(/\s/g, "");
    if (value.indexOf(">") !== -1) {
      value = value.substr(0, value.indexOf(">"));
    }
    if (value.length % 2) {
      value += "0";
    }
    if (regexCheckIfHex.test(value) === false) {
      return "";
    }
    var result = "";
    for (var i = 0; i < value.length; i += 2) {
      result += String.fromCharCode("0x" + (value[i] + value[i + 1]));
    }
    return result;
  };
  /*
  var FlatePredictors = {
      None: 1,
      TIFF: 2,
      PNG_None: 10,
      PNG_Sub: 11,
      PNG_Up: 12,
      PNG_Average: 13,
      PNG_Paeth: 14,
      PNG_Optimum: 15
  };
  */

  var FlateEncode = function(data) {
    var arr = new Uint8Array(data.length);
    var i = data.length;
    while (i--) {
      arr[i] = data.charCodeAt(i);
    }
    arr = zlibSync(arr);
    data = arr.reduce(function(data, byte) {
      return data + String.fromCharCode(byte);
    }, "");
    return data;
  };

  jsPDFAPI.processDataByFilters = function(origData, filterChain) {
    var i = 0;
    var data = origData || "";
    var reverseChain = [];
    filterChain = filterChain || [];

    if (typeof filterChain === "string") {
      filterChain = [filterChain];
    }

    for (i = 0; i < filterChain.length; i += 1) {
      switch (filterChain[i]) {
        case "ASCII85Decode":
        case "/ASCII85Decode":
          data = ASCII85Decode(data);
          reverseChain.push("/ASCII85Encode");
          break;
        case "ASCII85Encode":
        case "/ASCII85Encode":
          data = ASCII85Encode(data);
          reverseChain.push("/ASCII85Decode");
          break;
        case "ASCIIHexDecode":
        case "/ASCIIHexDecode":
          data = ASCIIHexDecode(data);
          reverseChain.push("/ASCIIHexEncode");
          break;
        case "ASCIIHexEncode":
        case "/ASCIIHexEncode":
          data = ASCIIHexEncode(data);
          reverseChain.push("/ASCIIHexDecode");
          break;
        case "FlateEncode":
        case "/FlateEncode":
          data = FlateEncode(data);
          reverseChain.push("/FlateDecode");
          break;
        default:
          throw new Error(
            'The filter: "' + filterChain[i] + '" is not implemented'
          );
      }
    }

    return { data: data, reverseChain: reverseChain.reverse().join(" ") };
  };
})(jsPDF.API);

/**
 * @license
 * ====================================================================
 * Copyright (c) 2013 Youssef Beddad, youssef.beddad@gmail.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * jsPDF JavaScript plugin
 *
 * @name javascript
 * @module
 */
(function(jsPDFAPI) {
  var jsNamesObj, jsJsObj, text;
  /**
   * @name addJS
   * @function
   * @param {string} javascript The javascript to be embedded into the PDF-file.
   * @returns {jsPDF}
   */
  jsPDFAPI.addJS = function(javascript) {
    text = javascript;
    this.internal.events.subscribe("postPutResources", function() {
      jsNamesObj = this.internal.newObject();
      this.internal.out("<<");
      this.internal.out("/Names [(EmbeddedJS) " + (jsNamesObj + 1) + " 0 R]");
      this.internal.out(">>");
      this.internal.out("endobj");

      jsJsObj = this.internal.newObject();
      this.internal.out("<<");
      this.internal.out("/S /JavaScript");
      this.internal.out("/JS (" + text + ")");
      this.internal.out(">>");
      this.internal.out("endobj");
    });
    this.internal.events.subscribe("putCatalog", function() {
      if (jsNamesObj !== undefined && jsJsObj !== undefined) {
        this.internal.out("/Names <</JavaScript " + jsNamesObj + " 0 R>>");
      }
    });
    return this;
  };
})(jsPDF.API);

/**
 * @license
 * Copyright (c) 2014 Steven Spungin (TwelveTone LLC)  steven@twelvetone.tv
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * jsPDF Outline PlugIn
 *
 * Generates a PDF Outline
 * @name outline
 * @module
 */
(function(jsPDFAPI) {

  var namesOid;
  //var destsGoto = [];

  jsPDFAPI.events.push([
    "postPutResources",
    function() {
      var pdf = this;
      var rx = /^(\d+) 0 obj$/;

      // Write action goto objects for each page
      // this.outline.destsGoto = [];
      // for (var i = 0; i < totalPages; i++) {
      // var id = pdf.internal.newObject();
      // this.outline.destsGoto.push(id);
      // pdf.internal.write("<</D[" + (i * 2 + 3) + " 0 R /XYZ null
      // null null]/S/GoTo>> endobj");
      // }
      //
      // for (var i = 0; i < dests.length; i++) {
      // pdf.internal.write("(page_" + (i + 1) + ")" + dests[i] + " 0
      // R");
      // }
      //
      if (this.outline.root.children.length > 0) {
        var lines = pdf.outline.render().split(/\r\n/);
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          var m = rx.exec(line);
          if (m != null) {
            var oid = m[1];
            pdf.internal.newObjectDeferredBegin(oid, false);
          }
          pdf.internal.write(line);
        }
      }

      // This code will write named destination for each page reference
      // (page_1, etc)
      if (this.outline.createNamedDestinations) {
        var totalPages = this.internal.pages.length;
        // WARNING: this assumes jsPDF starts on page 3 and pageIDs
        // follow 5, 7, 9, etc
        // Write destination objects for each page
        var dests = [];
        for (var i = 0; i < totalPages; i++) {
          var id = pdf.internal.newObject();
          dests.push(id);
          var info = pdf.internal.getPageInfo(i + 1);
          pdf.internal.write(
            "<< /D[" + info.objId + " 0 R /XYZ null null null]>> endobj"
          );
        }

        // assign a name for each destination
        var names2Oid = pdf.internal.newObject();
        pdf.internal.write("<< /Names [ ");
        for (var i = 0; i < dests.length; i++) {
          pdf.internal.write("(page_" + (i + 1) + ")" + dests[i] + " 0 R");
        }
        pdf.internal.write(" ] >>", "endobj");

        // var kids = pdf.internal.newObject();
        // pdf.internal.write('<< /Kids [ ' + names2Oid + ' 0 R');
        // pdf.internal.write(' ] >>', 'endobj');

        namesOid = pdf.internal.newObject();
        pdf.internal.write("<< /Dests " + names2Oid + " 0 R");
        pdf.internal.write(">>", "endobj");
      }
    }
  ]);

  jsPDFAPI.events.push([
    "putCatalog",
    function() {
      var pdf = this;
      if (pdf.outline.root.children.length > 0) {
        pdf.internal.write(
          "/Outlines",
          this.outline.makeRef(this.outline.root)
        );
        if (this.outline.createNamedDestinations) {
          pdf.internal.write("/Names " + namesOid + " 0 R");
        }
        // Open with Bookmarks showing
        // pdf.internal.write("/PageMode /UseOutlines");
      }
    }
  ]);

  jsPDFAPI.events.push([
    "initialized",
    function() {
      var pdf = this;

      pdf.outline = {
        createNamedDestinations: false,
        root: {
          children: []
        }
      };

      /**
       * Options: pageNumber
       */
      pdf.outline.add = function(parent, title, options) {
        var item = {
          title: title,
          options: options,
          children: []
        };
        if (parent == null) {
          parent = this.root;
        }
        parent.children.push(item);
        return item;
      };

      pdf.outline.render = function() {
        this.ctx = {};
        this.ctx.val = "";
        this.ctx.pdf = pdf;

        this.genIds_r(this.root);
        this.renderRoot(this.root);
        this.renderItems(this.root);

        return this.ctx.val;
      };

      pdf.outline.genIds_r = function(node) {
        node.id = pdf.internal.newObjectDeferred();
        for (var i = 0; i < node.children.length; i++) {
          this.genIds_r(node.children[i]);
        }
      };

      pdf.outline.renderRoot = function(node) {
        this.objStart(node);
        this.line("/Type /Outlines");
        if (node.children.length > 0) {
          this.line("/First " + this.makeRef(node.children[0]));
          this.line(
            "/Last " + this.makeRef(node.children[node.children.length - 1])
          );
        }
        this.line(
          "/Count " +
            this.count_r(
              {
                count: 0
              },
              node
            )
        );
        this.objEnd();
      };

      pdf.outline.renderItems = function(node) {
        var getVerticalCoordinateString = this.ctx.pdf.internal
          .getVerticalCoordinateString;
        for (var i = 0; i < node.children.length; i++) {
          var item = node.children[i];
          this.objStart(item);

          this.line("/Title " + this.makeString(item.title));

          this.line("/Parent " + this.makeRef(node));
          if (i > 0) {
            this.line("/Prev " + this.makeRef(node.children[i - 1]));
          }
          if (i < node.children.length - 1) {
            this.line("/Next " + this.makeRef(node.children[i + 1]));
          }
          if (item.children.length > 0) {
            this.line("/First " + this.makeRef(item.children[0]));
            this.line(
              "/Last " + this.makeRef(item.children[item.children.length - 1])
            );
          }

          var count = (this.count = this.count_r(
            {
              count: 0
            },
            item
          ));
          if (count > 0) {
            this.line("/Count " + count);
          }

          if (item.options) {
            if (item.options.pageNumber) {
              // Explicit Destination
              //WARNING this assumes page ids are 3,5,7, etc.
              var info = pdf.internal.getPageInfo(item.options.pageNumber);
              this.line(
                "/Dest " +
                  "[" +
                  info.objId +
                  " 0 R /XYZ 0 " +
                  getVerticalCoordinateString(0) +
                  " 0]"
              );
              // this line does not work on all clients (pageNumber instead of page ref)
              //this.line('/Dest ' + '[' + (item.options.pageNumber - 1) + ' /XYZ 0 ' + this.ctx.pdf.internal.pageSize.getHeight() + ' 0]');

              // Named Destination
              // this.line('/Dest (page_' + (item.options.pageNumber) + ')');

              // Action Destination
              // var id = pdf.internal.newObject();
              // pdf.internal.write('<</D[' + (item.options.pageNumber - 1) + ' /XYZ null null null]/S/GoTo>> endobj');
              // this.line('/A ' + id + ' 0 R' );
            }
          }
          this.objEnd();
        }
        for (var z = 0; z < node.children.length; z++) {
          this.renderItems(node.children[z]);
        }
      };

      pdf.outline.line = function(text) {
        this.ctx.val += text + "\r\n";
      };

      pdf.outline.makeRef = function(node) {
        return node.id + " 0 R";
      };

      pdf.outline.makeString = function(val) {
        return "(" + pdf.internal.pdfEscape(val) + ")";
      };

      pdf.outline.objStart = function(node) {
        this.ctx.val += "\r\n" + node.id + " 0 obj" + "\r\n<<\r\n";
      };

      pdf.outline.objEnd = function() {
        this.ctx.val += ">> \r\n" + "endobj" + "\r\n";
      };

      pdf.outline.count_r = function(ctx, node) {
        for (var i = 0; i < node.children.length; i++) {
          ctx.count++;
          this.count_r(ctx, node.children[i]);
        }
        return ctx.count;
      };
    }
  ]);

  return this;
})(jsPDF.API);

function decode(bytes, encoding = 'utf8') {
    const decoder = new TextDecoder(encoding);
    return decoder.decode(bytes);
}
const encoder = new TextEncoder();
function encode(str) {
    return encoder.encode(str);
}

const defaultByteLength = 1024 * 8;
const hostBigEndian = (() => {
    const array = new Uint8Array(4);
    const view = new Uint32Array(array.buffer);
    return !((view[0] = 1) & array[0]);
})();
const typedArrays = {
    int8: globalThis.Int8Array,
    uint8: globalThis.Uint8Array,
    int16: globalThis.Int16Array,
    uint16: globalThis.Uint16Array,
    int32: globalThis.Int32Array,
    uint32: globalThis.Uint32Array,
    uint64: globalThis.BigUint64Array,
    int64: globalThis.BigInt64Array,
    float32: globalThis.Float32Array,
    float64: globalThis.Float64Array,
};
class IOBuffer {
    /**
     * Reference to the internal ArrayBuffer object.
     */
    buffer;
    /**
     * Byte length of the internal ArrayBuffer.
     */
    byteLength;
    /**
     * Byte offset of the internal ArrayBuffer.
     */
    byteOffset;
    /**
     * Byte length of the internal ArrayBuffer.
     */
    length;
    /**
     * The current offset of the buffer's pointer.
     */
    offset;
    lastWrittenByte;
    littleEndian;
    _data;
    _mark;
    _marks;
    /**
     * Create a new IOBuffer.
     * @param data - The data to construct the IOBuffer with.
     * If data is a number, it will be the new buffer's length<br>
     * If data is `undefined`, the buffer will be initialized with a default length of 8Kb<br>
     * If data is an ArrayBuffer, SharedArrayBuffer, an ArrayBufferView (Typed Array), an IOBuffer instance,
     * or a Node.js Buffer, a view will be created over the underlying ArrayBuffer.
     * @param options - An object for the options.
     * @returns A new IOBuffer instance.
     */
    constructor(data = defaultByteLength, options = {}) {
        let dataIsGiven = false;
        if (typeof data === 'number') {
            data = new ArrayBuffer(data);
        }
        else {
            dataIsGiven = true;
            this.lastWrittenByte = data.byteLength;
        }
        const offset = options.offset ? options.offset >>> 0 : 0;
        const byteLength = data.byteLength - offset;
        let dvOffset = offset;
        if (ArrayBuffer.isView(data) || data instanceof IOBuffer) {
            if (data.byteLength !== data.buffer.byteLength) {
                dvOffset = data.byteOffset + offset;
            }
            data = data.buffer;
        }
        if (dataIsGiven) {
            this.lastWrittenByte = byteLength;
        }
        else {
            this.lastWrittenByte = 0;
        }
        this.buffer = data;
        this.length = byteLength;
        this.byteLength = byteLength;
        this.byteOffset = dvOffset;
        this.offset = 0;
        this.littleEndian = true;
        this._data = new DataView(this.buffer, dvOffset, byteLength);
        this._mark = 0;
        this._marks = [];
    }
    /**
     * Checks if the memory allocated to the buffer is sufficient to store more
     * bytes after the offset.
     * @param byteLength - The needed memory in bytes.
     * @returns `true` if there is sufficient space and `false` otherwise.
     */
    available(byteLength = 1) {
        return this.offset + byteLength <= this.length;
    }
    /**
     * Check if little-endian mode is used for reading and writing multi-byte
     * values.
     * @returns `true` if little-endian mode is used, `false` otherwise.
     */
    isLittleEndian() {
        return this.littleEndian;
    }
    /**
     * Set little-endian mode for reading and writing multi-byte values.
     * @returns This.
     */
    setLittleEndian() {
        this.littleEndian = true;
        return this;
    }
    /**
     * Check if big-endian mode is used for reading and writing multi-byte values.
     * @returns `true` if big-endian mode is used, `false` otherwise.
     */
    isBigEndian() {
        return !this.littleEndian;
    }
    /**
     * Switches to big-endian mode for reading and writing multi-byte values.
     * @returns This.
     */
    setBigEndian() {
        this.littleEndian = false;
        return this;
    }
    /**
     * Move the pointer n bytes forward.
     * @param n - Number of bytes to skip.
     * @returns This.
     */
    skip(n = 1) {
        this.offset += n;
        return this;
    }
    /**
     * Move the pointer n bytes backward.
     * @param n - Number of bytes to move back.
     * @returns This.
     */
    back(n = 1) {
        this.offset -= n;
        return this;
    }
    /**
     * Move the pointer to the given offset.
     * @param offset - The offset to move to.
     * @returns This.
     */
    seek(offset) {
        this.offset = offset;
        return this;
    }
    /**
     * Store the current pointer offset.
     * @see {@link IOBuffer#reset}
     * @returns This.
     */
    mark() {
        this._mark = this.offset;
        return this;
    }
    /**
     * Move the pointer back to the last pointer offset set by mark.
     * @see {@link IOBuffer#mark}
     * @returns This.
     */
    reset() {
        this.offset = this._mark;
        return this;
    }
    /**
     * Push the current pointer offset to the mark stack.
     * @see {@link IOBuffer#popMark}
     * @returns This.
     */
    pushMark() {
        this._marks.push(this.offset);
        return this;
    }
    /**
     * Pop the last pointer offset from the mark stack, and set the current
     * pointer offset to the popped value.
     * @see {@link IOBuffer#pushMark}
     * @returns This.
     */
    popMark() {
        const offset = this._marks.pop();
        if (offset === undefined) {
            throw new Error('Mark stack empty');
        }
        this.seek(offset);
        return this;
    }
    /**
     * Move the pointer offset back to 0.
     * @returns This.
     */
    rewind() {
        this.offset = 0;
        return this;
    }
    /**
     * Make sure the buffer has sufficient memory to write a given byteLength at
     * the current pointer offset.
     * If the buffer's memory is insufficient, this method will create a new
     * buffer (a copy) with a length that is twice (byteLength + current offset).
     * @param byteLength - The needed memory in bytes.
     * @returns This.
     */
    ensureAvailable(byteLength = 1) {
        if (!this.available(byteLength)) {
            const lengthNeeded = this.offset + byteLength;
            const newLength = lengthNeeded * 2;
            const newArray = new Uint8Array(newLength);
            newArray.set(new Uint8Array(this.buffer));
            this.buffer = newArray.buffer;
            this.length = newLength;
            this.byteLength = newLength;
            this._data = new DataView(this.buffer);
        }
        return this;
    }
    /**
     * Read a byte and return false if the byte's value is 0, or true otherwise.
     * Moves pointer forward by one byte.
     * @returns The read boolean.
     */
    readBoolean() {
        return this.readUint8() !== 0;
    }
    /**
     * Read a signed 8-bit integer and move pointer forward by 1 byte.
     * @returns The read byte.
     */
    readInt8() {
        return this._data.getInt8(this.offset++);
    }
    /**
     * Read an unsigned 8-bit integer and move pointer forward by 1 byte.
     * @returns The read byte.
     */
    readUint8() {
        return this._data.getUint8(this.offset++);
    }
    /**
     * Alias for {@link IOBuffer#readUint8}.
     * @returns The read byte.
     */
    readByte() {
        return this.readUint8();
    }
    /**
     * Read `n` bytes and move pointer forward by `n` bytes.
     * @param n - Number of bytes to read.
     * @returns The read bytes.
     */
    readBytes(n = 1) {
        return this.readArray(n, 'uint8');
    }
    /**
     * Creates an array of corresponding to the type `type` and size `size`.
     * For example type `uint8` will create a `Uint8Array`.
     * @param size - size of the resulting array
     * @param type - number type of elements to read
     * @returns The read array.
     */
    readArray(size, type) {
        const bytes = typedArrays[type].BYTES_PER_ELEMENT * size;
        const offset = this.byteOffset + this.offset;
        const slice = this.buffer.slice(offset, offset + bytes);
        if (this.littleEndian === hostBigEndian &&
            type !== 'uint8' &&
            type !== 'int8') {
            const slice = new Uint8Array(this.buffer.slice(offset, offset + bytes));
            slice.reverse();
            const returnArray = new typedArrays[type](slice.buffer);
            this.offset += bytes;
            returnArray.reverse();
            return returnArray;
        }
        const returnArray = new typedArrays[type](slice);
        this.offset += bytes;
        return returnArray;
    }
    /**
     * Read a 16-bit signed integer and move pointer forward by 2 bytes.
     * @returns The read value.
     */
    readInt16() {
        const value = this._data.getInt16(this.offset, this.littleEndian);
        this.offset += 2;
        return value;
    }
    /**
     * Read a 16-bit unsigned integer and move pointer forward by 2 bytes.
     * @returns The read value.
     */
    readUint16() {
        const value = this._data.getUint16(this.offset, this.littleEndian);
        this.offset += 2;
        return value;
    }
    /**
     * Read a 32-bit signed integer and move pointer forward by 4 bytes.
     * @returns The read value.
     */
    readInt32() {
        const value = this._data.getInt32(this.offset, this.littleEndian);
        this.offset += 4;
        return value;
    }
    /**
     * Read a 32-bit unsigned integer and move pointer forward by 4 bytes.
     * @returns The read value.
     */
    readUint32() {
        const value = this._data.getUint32(this.offset, this.littleEndian);
        this.offset += 4;
        return value;
    }
    /**
     * Read a 32-bit floating number and move pointer forward by 4 bytes.
     * @returns The read value.
     */
    readFloat32() {
        const value = this._data.getFloat32(this.offset, this.littleEndian);
        this.offset += 4;
        return value;
    }
    /**
     * Read a 64-bit floating number and move pointer forward by 8 bytes.
     * @returns The read value.
     */
    readFloat64() {
        const value = this._data.getFloat64(this.offset, this.littleEndian);
        this.offset += 8;
        return value;
    }
    /**
     * Read a 64-bit signed integer number and move pointer forward by 8 bytes.
     * @returns The read value.
     */
    readBigInt64() {
        const value = this._data.getBigInt64(this.offset, this.littleEndian);
        this.offset += 8;
        return value;
    }
    /**
     * Read a 64-bit unsigned integer number and move pointer forward by 8 bytes.
     * @returns The read value.
     */
    readBigUint64() {
        const value = this._data.getBigUint64(this.offset, this.littleEndian);
        this.offset += 8;
        return value;
    }
    /**
     * Read a 1-byte ASCII character and move pointer forward by 1 byte.
     * @returns The read character.
     */
    readChar() {
        // eslint-disable-next-line unicorn/prefer-code-point
        return String.fromCharCode(this.readInt8());
    }
    /**
     * Read `n` 1-byte ASCII characters and move pointer forward by `n` bytes.
     * @param n - Number of characters to read.
     * @returns The read characters.
     */
    readChars(n = 1) {
        let result = '';
        for (let i = 0; i < n; i++) {
            result += this.readChar();
        }
        return result;
    }
    /**
     * Read the next `n` bytes, return a UTF-8 decoded string and move pointer
     * forward by `n` bytes.
     * @param n - Number of bytes to read.
     * @returns The decoded string.
     */
    readUtf8(n = 1) {
        return decode(this.readBytes(n));
    }
    /**
     * Read the next `n` bytes, return a string decoded with `encoding` and move pointer
     * forward by `n` bytes.
     * If no encoding is passed, the function is equivalent to @see {@link IOBuffer#readUtf8}
     * @param n - Number of bytes to read.
     * @param encoding - The encoding to use. Default is 'utf8'.
     * @returns The decoded string.
     */
    decodeText(n = 1, encoding = 'utf8') {
        return decode(this.readBytes(n), encoding);
    }
    /**
     * Write 0xff if the passed value is truthy, 0x00 otherwise and move pointer
     * forward by 1 byte.
     * @param value - The value to write.
     * @returns This.
     */
    writeBoolean(value) {
        this.writeUint8(value ? 0xff : 0x00);
        return this;
    }
    /**
     * Write `value` as an 8-bit signed integer and move pointer forward by 1 byte.
     * @param value - The value to write.
     * @returns This.
     */
    writeInt8(value) {
        this.ensureAvailable(1);
        this._data.setInt8(this.offset++, value);
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as an 8-bit unsigned integer and move pointer forward by 1
     * byte.
     * @param value - The value to write.
     * @returns This.
     */
    writeUint8(value) {
        this.ensureAvailable(1);
        this._data.setUint8(this.offset++, value);
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * An alias for {@link IOBuffer#writeUint8}.
     * @param value - The value to write.
     * @returns This.
     */
    writeByte(value) {
        return this.writeUint8(value);
    }
    /**
     * Write all elements of `bytes` as uint8 values and move pointer forward by
     * `bytes.length` bytes.
     * @param bytes - The array of bytes to write.
     * @returns This.
     */
    writeBytes(bytes) {
        this.ensureAvailable(bytes.length);
        // eslint-disable-next-line @typescript-eslint/prefer-for-of
        for (let i = 0; i < bytes.length; i++) {
            this._data.setUint8(this.offset++, bytes[i]);
        }
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 16-bit signed integer and move pointer forward by 2
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeInt16(value) {
        this.ensureAvailable(2);
        this._data.setInt16(this.offset, value, this.littleEndian);
        this.offset += 2;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 16-bit unsigned integer and move pointer forward by 2
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeUint16(value) {
        this.ensureAvailable(2);
        this._data.setUint16(this.offset, value, this.littleEndian);
        this.offset += 2;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 32-bit signed integer and move pointer forward by 4
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeInt32(value) {
        this.ensureAvailable(4);
        this._data.setInt32(this.offset, value, this.littleEndian);
        this.offset += 4;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 32-bit unsigned integer and move pointer forward by 4
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeUint32(value) {
        this.ensureAvailable(4);
        this._data.setUint32(this.offset, value, this.littleEndian);
        this.offset += 4;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 32-bit floating number and move pointer forward by 4
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeFloat32(value) {
        this.ensureAvailable(4);
        this._data.setFloat32(this.offset, value, this.littleEndian);
        this.offset += 4;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 64-bit floating number and move pointer forward by 8
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeFloat64(value) {
        this.ensureAvailable(8);
        this._data.setFloat64(this.offset, value, this.littleEndian);
        this.offset += 8;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 64-bit signed bigint and move pointer forward by 8
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeBigInt64(value) {
        this.ensureAvailable(8);
        this._data.setBigInt64(this.offset, value, this.littleEndian);
        this.offset += 8;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write `value` as a 64-bit unsigned bigint and move pointer forward by 8
     * bytes.
     * @param value - The value to write.
     * @returns This.
     */
    writeBigUint64(value) {
        this.ensureAvailable(8);
        this._data.setBigUint64(this.offset, value, this.littleEndian);
        this.offset += 8;
        this._updateLastWrittenByte();
        return this;
    }
    /**
     * Write the charCode of `str`'s first character as an 8-bit unsigned integer
     * and move pointer forward by 1 byte.
     * @param str - The character to write.
     * @returns This.
     */
    writeChar(str) {
        // eslint-disable-next-line unicorn/prefer-code-point
        return this.writeUint8(str.charCodeAt(0));
    }
    /**
     * Write the charCodes of all `str`'s characters as 8-bit unsigned integers
     * and move pointer forward by `str.length` bytes.
     * @param str - The characters to write.
     * @returns This.
     */
    writeChars(str) {
        for (let i = 0; i < str.length; i++) {
            // eslint-disable-next-line unicorn/prefer-code-point
            this.writeUint8(str.charCodeAt(i));
        }
        return this;
    }
    /**
     * UTF-8 encode and write `str` to the current pointer offset and move pointer
     * forward according to the encoded length.
     * @param str - The string to write.
     * @returns This.
     */
    writeUtf8(str) {
        return this.writeBytes(encode(str));
    }
    /**
     * Export a Uint8Array view of the internal buffer.
     * The view starts at the byte offset and its length
     * is calculated to stop at the last written byte or the original length.
     * @returns A new Uint8Array view.
     */
    toArray() {
        return new Uint8Array(this.buffer, this.byteOffset, this.lastWrittenByte);
    }
    /**
     *  Get the total number of bytes written so far, regardless of the current offset.
     * @returns - Total number of bytes.
     */
    getWrittenByteLength() {
        return this.lastWrittenByte - this.byteOffset;
    }
    /**
     * Update the last written byte offset
     * @private
     */
    _updateLastWrittenByte() {
        if (this.offset > this.lastWrittenByte) {
            this.lastWrittenByte = this.offset;
        }
    }
}

/*! pako 2.1.0 https://github.com/nodeca/pako @license (MIT AND Zlib) */
// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

/* eslint-disable space-unary-ops */

/* Public constants ==========================================================*/
/* ===========================================================================*/


//const Z_FILTERED          = 1;
//const Z_HUFFMAN_ONLY      = 2;
//const Z_RLE               = 3;
const Z_FIXED$1               = 4;
//const Z_DEFAULT_STRATEGY  = 0;

/* Possible values of the data_type field (though see inflate()) */
const Z_BINARY              = 0;
const Z_TEXT                = 1;
//const Z_ASCII             = 1; // = Z_TEXT
const Z_UNKNOWN$1             = 2;

/*============================================================================*/


function zero$1(buf) { let len = buf.length; while (--len >= 0) { buf[len] = 0; } }

// From zutil.h

const STORED_BLOCK = 0;
const STATIC_TREES = 1;
const DYN_TREES    = 2;
/* The three kinds of block type */

const MIN_MATCH$1    = 3;
const MAX_MATCH$1    = 258;
/* The minimum and maximum match lengths */

// From deflate.h
/* ===========================================================================
 * Internal compression state.
 */

const LENGTH_CODES$1  = 29;
/* number of length codes, not counting the special END_BLOCK code */

const LITERALS$1      = 256;
/* number of literal bytes 0..255 */

const L_CODES$1       = LITERALS$1 + 1 + LENGTH_CODES$1;
/* number of Literal or Length codes, including the END_BLOCK code */

const D_CODES$1       = 30;
/* number of distance codes */

const BL_CODES$1      = 19;
/* number of codes used to transfer the bit lengths */

const HEAP_SIZE$1     = 2 * L_CODES$1 + 1;
/* maximum heap size */

const MAX_BITS$1      = 15;
/* All codes must not exceed MAX_BITS bits */

const Buf_size      = 16;
/* size of bit buffer in bi_buf */


/* ===========================================================================
 * Constants
 */

const MAX_BL_BITS = 7;
/* Bit length codes must not exceed MAX_BL_BITS bits */

const END_BLOCK   = 256;
/* end of block literal code */

const REP_3_6     = 16;
/* repeat previous bit length 3-6 times (2 bits of repeat count) */

const REPZ_3_10   = 17;
/* repeat a zero length 3-10 times  (3 bits of repeat count) */

const REPZ_11_138 = 18;
/* repeat a zero length 11-138 times  (7 bits of repeat count) */

/* eslint-disable comma-spacing,array-bracket-spacing */
const extra_lbits =   /* extra bits for each length code */
  new Uint8Array([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]);

const extra_dbits =   /* extra bits for each distance code */
  new Uint8Array([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]);

const extra_blbits =  /* extra bits for each bit length code */
  new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7]);

const bl_order =
  new Uint8Array([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]);
/* eslint-enable comma-spacing,array-bracket-spacing */

/* The lengths of the bit length codes are sent in order of decreasing
 * probability, to avoid transmitting the lengths for unused bit length codes.
 */

/* ===========================================================================
 * Local data. These are initialized only once.
 */

// We pre-fill arrays with 0 to avoid uninitialized gaps

const DIST_CODE_LEN = 512; /* see definition of array dist_code below */

// !!!! Use flat array instead of structure, Freq = i*2, Len = i*2+1
const static_ltree  = new Array((L_CODES$1 + 2) * 2);
zero$1(static_ltree);
/* The static literal tree. Since the bit lengths are imposed, there is no
 * need for the L_CODES extra codes used during heap construction. However
 * The codes 286 and 287 are needed to build a canonical tree (see _tr_init
 * below).
 */

const static_dtree  = new Array(D_CODES$1 * 2);
zero$1(static_dtree);
/* The static distance tree. (Actually a trivial tree since all codes use
 * 5 bits.)
 */

const _dist_code    = new Array(DIST_CODE_LEN);
zero$1(_dist_code);
/* Distance codes. The first 256 values correspond to the distances
 * 3 .. 258, the last 256 values correspond to the top 8 bits of
 * the 15 bit distances.
 */

const _length_code  = new Array(MAX_MATCH$1 - MIN_MATCH$1 + 1);
zero$1(_length_code);
/* length code for each normalized match length (0 == MIN_MATCH) */

const base_length   = new Array(LENGTH_CODES$1);
zero$1(base_length);
/* First normalized length for each code (0 = MIN_MATCH) */

const base_dist     = new Array(D_CODES$1);
zero$1(base_dist);
/* First normalized distance for each code (0 = distance of 1) */


function StaticTreeDesc(static_tree, extra_bits, extra_base, elems, max_length) {

  this.static_tree  = static_tree;  /* static tree or NULL */
  this.extra_bits   = extra_bits;   /* extra bits for each code or NULL */
  this.extra_base   = extra_base;   /* base index for extra_bits */
  this.elems        = elems;        /* max number of elements in the tree */
  this.max_length   = max_length;   /* max bit length for the codes */

  // show if `static_tree` has data or dummy - needed for monomorphic objects
  this.has_stree    = static_tree && static_tree.length;
}


let static_l_desc;
let static_d_desc;
let static_bl_desc;


function TreeDesc(dyn_tree, stat_desc) {
  this.dyn_tree = dyn_tree;     /* the dynamic tree */
  this.max_code = 0;            /* largest code with non zero frequency */
  this.stat_desc = stat_desc;   /* the corresponding static tree */
}



const d_code = (dist) => {

  return dist < 256 ? _dist_code[dist] : _dist_code[256 + (dist >>> 7)];
};


/* ===========================================================================
 * Output a short LSB first on the stream.
 * IN assertion: there is enough room in pendingBuf.
 */
const put_short = (s, w) => {
//    put_byte(s, (uch)((w) & 0xff));
//    put_byte(s, (uch)((ush)(w) >> 8));
  s.pending_buf[s.pending++] = (w) & 0xff;
  s.pending_buf[s.pending++] = (w >>> 8) & 0xff;
};


/* ===========================================================================
 * Send a value on a given number of bits.
 * IN assertion: length <= 16 and value fits in length bits.
 */
const send_bits = (s, value, length) => {

  if (s.bi_valid > (Buf_size - length)) {
    s.bi_buf |= (value << s.bi_valid) & 0xffff;
    put_short(s, s.bi_buf);
    s.bi_buf = value >> (Buf_size - s.bi_valid);
    s.bi_valid += length - Buf_size;
  } else {
    s.bi_buf |= (value << s.bi_valid) & 0xffff;
    s.bi_valid += length;
  }
};


const send_code = (s, c, tree) => {

  send_bits(s, tree[c * 2]/*.Code*/, tree[c * 2 + 1]/*.Len*/);
};


/* ===========================================================================
 * Reverse the first len bits of a code, using straightforward code (a faster
 * method would use a table)
 * IN assertion: 1 <= len <= 15
 */
const bi_reverse = (code, len) => {

  let res = 0;
  do {
    res |= code & 1;
    code >>>= 1;
    res <<= 1;
  } while (--len > 0);
  return res >>> 1;
};


/* ===========================================================================
 * Flush the bit buffer, keeping at most 7 bits in it.
 */
const bi_flush = (s) => {

  if (s.bi_valid === 16) {
    put_short(s, s.bi_buf);
    s.bi_buf = 0;
    s.bi_valid = 0;

  } else if (s.bi_valid >= 8) {
    s.pending_buf[s.pending++] = s.bi_buf & 0xff;
    s.bi_buf >>= 8;
    s.bi_valid -= 8;
  }
};


/* ===========================================================================
 * Compute the optimal bit lengths for a tree and update the total bit length
 * for the current block.
 * IN assertion: the fields freq and dad are set, heap[heap_max] and
 *    above are the tree nodes sorted by increasing frequency.
 * OUT assertions: the field len is set to the optimal bit length, the
 *     array bl_count contains the frequencies for each bit length.
 *     The length opt_len is updated; static_len is also updated if stree is
 *     not null.
 */
const gen_bitlen = (s, desc) => {
//    deflate_state *s;
//    tree_desc *desc;    /* the tree descriptor */

  const tree            = desc.dyn_tree;
  const max_code        = desc.max_code;
  const stree           = desc.stat_desc.static_tree;
  const has_stree       = desc.stat_desc.has_stree;
  const extra           = desc.stat_desc.extra_bits;
  const base            = desc.stat_desc.extra_base;
  const max_length      = desc.stat_desc.max_length;
  let h;              /* heap index */
  let n, m;           /* iterate over the tree elements */
  let bits;           /* bit length */
  let xbits;          /* extra bits */
  let f;              /* frequency */
  let overflow = 0;   /* number of elements with bit length too large */

  for (bits = 0; bits <= MAX_BITS$1; bits++) {
    s.bl_count[bits] = 0;
  }

  /* In a first pass, compute the optimal bit lengths (which may
   * overflow in the case of the bit length tree).
   */
  tree[s.heap[s.heap_max] * 2 + 1]/*.Len*/ = 0; /* root of the heap */

  for (h = s.heap_max + 1; h < HEAP_SIZE$1; h++) {
    n = s.heap[h];
    bits = tree[tree[n * 2 + 1]/*.Dad*/ * 2 + 1]/*.Len*/ + 1;
    if (bits > max_length) {
      bits = max_length;
      overflow++;
    }
    tree[n * 2 + 1]/*.Len*/ = bits;
    /* We overwrite tree[n].Dad which is no longer needed */

    if (n > max_code) { continue; } /* not a leaf node */

    s.bl_count[bits]++;
    xbits = 0;
    if (n >= base) {
      xbits = extra[n - base];
    }
    f = tree[n * 2]/*.Freq*/;
    s.opt_len += f * (bits + xbits);
    if (has_stree) {
      s.static_len += f * (stree[n * 2 + 1]/*.Len*/ + xbits);
    }
  }
  if (overflow === 0) { return; }

  // Tracev((stderr,"\nbit length overflow\n"));
  /* This happens for example on obj2 and pic of the Calgary corpus */

  /* Find the first bit length which could increase: */
  do {
    bits = max_length - 1;
    while (s.bl_count[bits] === 0) { bits--; }
    s.bl_count[bits]--;      /* move one leaf down the tree */
    s.bl_count[bits + 1] += 2; /* move one overflow item as its brother */
    s.bl_count[max_length]--;
    /* The brother of the overflow item also moves one step up,
     * but this does not affect bl_count[max_length]
     */
    overflow -= 2;
  } while (overflow > 0);

  /* Now recompute all bit lengths, scanning in increasing frequency.
   * h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
   * lengths instead of fixing only the wrong ones. This idea is taken
   * from 'ar' written by Haruhiko Okumura.)
   */
  for (bits = max_length; bits !== 0; bits--) {
    n = s.bl_count[bits];
    while (n !== 0) {
      m = s.heap[--h];
      if (m > max_code) { continue; }
      if (tree[m * 2 + 1]/*.Len*/ !== bits) {
        // Tracev((stderr,"code %d bits %d->%d\n", m, tree[m].Len, bits));
        s.opt_len += (bits - tree[m * 2 + 1]/*.Len*/) * tree[m * 2]/*.Freq*/;
        tree[m * 2 + 1]/*.Len*/ = bits;
      }
      n--;
    }
  }
};


/* ===========================================================================
 * Generate the codes for a given tree and bit counts (which need not be
 * optimal).
 * IN assertion: the array bl_count contains the bit length statistics for
 * the given tree and the field len is set for all tree elements.
 * OUT assertion: the field code is set for all tree elements of non
 *     zero code length.
 */
const gen_codes = (tree, max_code, bl_count) => {
//    ct_data *tree;             /* the tree to decorate */
//    int max_code;              /* largest code with non zero frequency */
//    ushf *bl_count;            /* number of codes at each bit length */

  const next_code = new Array(MAX_BITS$1 + 1); /* next code value for each bit length */
  let code = 0;              /* running code value */
  let bits;                  /* bit index */
  let n;                     /* code index */

  /* The distribution counts are first used to generate the code values
   * without bit reversal.
   */
  for (bits = 1; bits <= MAX_BITS$1; bits++) {
    code = (code + bl_count[bits - 1]) << 1;
    next_code[bits] = code;
  }
  /* Check that the bit counts in bl_count are consistent. The last code
   * must be all ones.
   */
  //Assert (code + bl_count[MAX_BITS]-1 == (1<<MAX_BITS)-1,
  //        "inconsistent bit counts");
  //Tracev((stderr,"\ngen_codes: max_code %d ", max_code));

  for (n = 0;  n <= max_code; n++) {
    let len = tree[n * 2 + 1]/*.Len*/;
    if (len === 0) { continue; }
    /* Now reverse the bits */
    tree[n * 2]/*.Code*/ = bi_reverse(next_code[len]++, len);

    //Tracecv(tree != static_ltree, (stderr,"\nn %3d %c l %2d c %4x (%x) ",
    //     n, (isgraph(n) ? n : ' '), len, tree[n].Code, next_code[len]-1));
  }
};


/* ===========================================================================
 * Initialize the various 'constant' tables.
 */
const tr_static_init = () => {

  let n;        /* iterates over tree elements */
  let bits;     /* bit counter */
  let length;   /* length value */
  let code;     /* code value */
  let dist;     /* distance index */
  const bl_count = new Array(MAX_BITS$1 + 1);
  /* number of codes at each bit length for an optimal tree */

  // do check in _tr_init()
  //if (static_init_done) return;

  /* For some embedded targets, global variables are not initialized: */
/*#ifdef NO_INIT_GLOBAL_POINTERS
  static_l_desc.static_tree = static_ltree;
  static_l_desc.extra_bits = extra_lbits;
  static_d_desc.static_tree = static_dtree;
  static_d_desc.extra_bits = extra_dbits;
  static_bl_desc.extra_bits = extra_blbits;
#endif*/

  /* Initialize the mapping length (0..255) -> length code (0..28) */
  length = 0;
  for (code = 0; code < LENGTH_CODES$1 - 1; code++) {
    base_length[code] = length;
    for (n = 0; n < (1 << extra_lbits[code]); n++) {
      _length_code[length++] = code;
    }
  }
  //Assert (length == 256, "tr_static_init: length != 256");
  /* Note that the length 255 (match length 258) can be represented
   * in two different ways: code 284 + 5 bits or code 285, so we
   * overwrite length_code[255] to use the best encoding:
   */
  _length_code[length - 1] = code;

  /* Initialize the mapping dist (0..32K) -> dist code (0..29) */
  dist = 0;
  for (code = 0; code < 16; code++) {
    base_dist[code] = dist;
    for (n = 0; n < (1 << extra_dbits[code]); n++) {
      _dist_code[dist++] = code;
    }
  }
  //Assert (dist == 256, "tr_static_init: dist != 256");
  dist >>= 7; /* from now on, all distances are divided by 128 */
  for (; code < D_CODES$1; code++) {
    base_dist[code] = dist << 7;
    for (n = 0; n < (1 << (extra_dbits[code] - 7)); n++) {
      _dist_code[256 + dist++] = code;
    }
  }
  //Assert (dist == 256, "tr_static_init: 256+dist != 512");

  /* Construct the codes of the static literal tree */
  for (bits = 0; bits <= MAX_BITS$1; bits++) {
    bl_count[bits] = 0;
  }

  n = 0;
  while (n <= 143) {
    static_ltree[n * 2 + 1]/*.Len*/ = 8;
    n++;
    bl_count[8]++;
  }
  while (n <= 255) {
    static_ltree[n * 2 + 1]/*.Len*/ = 9;
    n++;
    bl_count[9]++;
  }
  while (n <= 279) {
    static_ltree[n * 2 + 1]/*.Len*/ = 7;
    n++;
    bl_count[7]++;
  }
  while (n <= 287) {
    static_ltree[n * 2 + 1]/*.Len*/ = 8;
    n++;
    bl_count[8]++;
  }
  /* Codes 286 and 287 do not exist, but we must include them in the
   * tree construction to get a canonical Huffman tree (longest code
   * all ones)
   */
  gen_codes(static_ltree, L_CODES$1 + 1, bl_count);

  /* The static distance tree is trivial: */
  for (n = 0; n < D_CODES$1; n++) {
    static_dtree[n * 2 + 1]/*.Len*/ = 5;
    static_dtree[n * 2]/*.Code*/ = bi_reverse(n, 5);
  }

  // Now data ready and we can init static trees
  static_l_desc = new StaticTreeDesc(static_ltree, extra_lbits, LITERALS$1 + 1, L_CODES$1, MAX_BITS$1);
  static_d_desc = new StaticTreeDesc(static_dtree, extra_dbits, 0,          D_CODES$1, MAX_BITS$1);
  static_bl_desc = new StaticTreeDesc(new Array(0), extra_blbits, 0,         BL_CODES$1, MAX_BL_BITS);

  //static_init_done = true;
};


/* ===========================================================================
 * Initialize a new block.
 */
const init_block = (s) => {

  let n; /* iterates over tree elements */

  /* Initialize the trees. */
  for (n = 0; n < L_CODES$1;  n++) { s.dyn_ltree[n * 2]/*.Freq*/ = 0; }
  for (n = 0; n < D_CODES$1;  n++) { s.dyn_dtree[n * 2]/*.Freq*/ = 0; }
  for (n = 0; n < BL_CODES$1; n++) { s.bl_tree[n * 2]/*.Freq*/ = 0; }

  s.dyn_ltree[END_BLOCK * 2]/*.Freq*/ = 1;
  s.opt_len = s.static_len = 0;
  s.sym_next = s.matches = 0;
};


/* ===========================================================================
 * Flush the bit buffer and align the output on a byte boundary
 */
const bi_windup = (s) =>
{
  if (s.bi_valid > 8) {
    put_short(s, s.bi_buf);
  } else if (s.bi_valid > 0) {
    //put_byte(s, (Byte)s->bi_buf);
    s.pending_buf[s.pending++] = s.bi_buf;
  }
  s.bi_buf = 0;
  s.bi_valid = 0;
};

/* ===========================================================================
 * Compares to subtrees, using the tree depth as tie breaker when
 * the subtrees have equal frequency. This minimizes the worst case length.
 */
const smaller = (tree, n, m, depth) => {

  const _n2 = n * 2;
  const _m2 = m * 2;
  return (tree[_n2]/*.Freq*/ < tree[_m2]/*.Freq*/ ||
         (tree[_n2]/*.Freq*/ === tree[_m2]/*.Freq*/ && depth[n] <= depth[m]));
};

/* ===========================================================================
 * Restore the heap property by moving down the tree starting at node k,
 * exchanging a node with the smallest of its two sons if necessary, stopping
 * when the heap property is re-established (each father smaller than its
 * two sons).
 */
const pqdownheap = (s, tree, k) => {
//    deflate_state *s;
//    ct_data *tree;  /* the tree to restore */
//    int k;               /* node to move down */

  const v = s.heap[k];
  let j = k << 1;  /* left son of k */
  while (j <= s.heap_len) {
    /* Set j to the smallest of the two sons: */
    if (j < s.heap_len &&
      smaller(tree, s.heap[j + 1], s.heap[j], s.depth)) {
      j++;
    }
    /* Exit if v is smaller than both sons */
    if (smaller(tree, v, s.heap[j], s.depth)) { break; }

    /* Exchange v with the smallest son */
    s.heap[k] = s.heap[j];
    k = j;

    /* And continue down the tree, setting j to the left son of k */
    j <<= 1;
  }
  s.heap[k] = v;
};


// inlined manually
// const SMALLEST = 1;

/* ===========================================================================
 * Send the block data compressed using the given Huffman trees
 */
const compress_block = (s, ltree, dtree) => {
//    deflate_state *s;
//    const ct_data *ltree; /* literal tree */
//    const ct_data *dtree; /* distance tree */

  let dist;           /* distance of matched string */
  let lc;             /* match length or unmatched char (if dist == 0) */
  let sx = 0;         /* running index in sym_buf */
  let code;           /* the code to send */
  let extra;          /* number of extra bits to send */

  if (s.sym_next !== 0) {
    do {
      dist = s.pending_buf[s.sym_buf + sx++] & 0xff;
      dist += (s.pending_buf[s.sym_buf + sx++] & 0xff) << 8;
      lc = s.pending_buf[s.sym_buf + sx++];
      if (dist === 0) {
        send_code(s, lc, ltree); /* send a literal byte */
        //Tracecv(isgraph(lc), (stderr," '%c' ", lc));
      } else {
        /* Here, lc is the match length - MIN_MATCH */
        code = _length_code[lc];
        send_code(s, code + LITERALS$1 + 1, ltree); /* send the length code */
        extra = extra_lbits[code];
        if (extra !== 0) {
          lc -= base_length[code];
          send_bits(s, lc, extra);       /* send the extra length bits */
        }
        dist--; /* dist is now the match distance - 1 */
        code = d_code(dist);
        //Assert (code < D_CODES, "bad d_code");

        send_code(s, code, dtree);       /* send the distance code */
        extra = extra_dbits[code];
        if (extra !== 0) {
          dist -= base_dist[code];
          send_bits(s, dist, extra);   /* send the extra distance bits */
        }
      } /* literal or match pair ? */

      /* Check that the overlay between pending_buf and sym_buf is ok: */
      //Assert(s->pending < s->lit_bufsize + sx, "pendingBuf overflow");

    } while (sx < s.sym_next);
  }

  send_code(s, END_BLOCK, ltree);
};


/* ===========================================================================
 * Construct one Huffman tree and assigns the code bit strings and lengths.
 * Update the total bit length for the current block.
 * IN assertion: the field freq is set for all tree elements.
 * OUT assertions: the fields len and code are set to the optimal bit length
 *     and corresponding code. The length opt_len is updated; static_len is
 *     also updated if stree is not null. The field max_code is set.
 */
const build_tree = (s, desc) => {
//    deflate_state *s;
//    tree_desc *desc; /* the tree descriptor */

  const tree     = desc.dyn_tree;
  const stree    = desc.stat_desc.static_tree;
  const has_stree = desc.stat_desc.has_stree;
  const elems    = desc.stat_desc.elems;
  let n, m;          /* iterate over heap elements */
  let max_code = -1; /* largest code with non zero frequency */
  let node;          /* new node being created */

  /* Construct the initial heap, with least frequent element in
   * heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1].
   * heap[0] is not used.
   */
  s.heap_len = 0;
  s.heap_max = HEAP_SIZE$1;

  for (n = 0; n < elems; n++) {
    if (tree[n * 2]/*.Freq*/ !== 0) {
      s.heap[++s.heap_len] = max_code = n;
      s.depth[n] = 0;

    } else {
      tree[n * 2 + 1]/*.Len*/ = 0;
    }
  }

  /* The pkzip format requires that at least one distance code exists,
   * and that at least one bit should be sent even if there is only one
   * possible code. So to avoid special checks later on we force at least
   * two codes of non zero frequency.
   */
  while (s.heap_len < 2) {
    node = s.heap[++s.heap_len] = (max_code < 2 ? ++max_code : 0);
    tree[node * 2]/*.Freq*/ = 1;
    s.depth[node] = 0;
    s.opt_len--;

    if (has_stree) {
      s.static_len -= stree[node * 2 + 1]/*.Len*/;
    }
    /* node is 0 or 1 so it does not have extra bits */
  }
  desc.max_code = max_code;

  /* The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
   * establish sub-heaps of increasing lengths:
   */
  for (n = (s.heap_len >> 1/*int /2*/); n >= 1; n--) { pqdownheap(s, tree, n); }

  /* Construct the Huffman tree by repeatedly combining the least two
   * frequent nodes.
   */
  node = elems;              /* next internal node of the tree */
  do {
    //pqremove(s, tree, n);  /* n = node of least frequency */
    /*** pqremove ***/
    n = s.heap[1/*SMALLEST*/];
    s.heap[1/*SMALLEST*/] = s.heap[s.heap_len--];
    pqdownheap(s, tree, 1/*SMALLEST*/);
    /***/

    m = s.heap[1/*SMALLEST*/]; /* m = node of next least frequency */

    s.heap[--s.heap_max] = n; /* keep the nodes sorted by frequency */
    s.heap[--s.heap_max] = m;

    /* Create a new node father of n and m */
    tree[node * 2]/*.Freq*/ = tree[n * 2]/*.Freq*/ + tree[m * 2]/*.Freq*/;
    s.depth[node] = (s.depth[n] >= s.depth[m] ? s.depth[n] : s.depth[m]) + 1;
    tree[n * 2 + 1]/*.Dad*/ = tree[m * 2 + 1]/*.Dad*/ = node;

    /* and insert the new node in the heap */
    s.heap[1/*SMALLEST*/] = node++;
    pqdownheap(s, tree, 1/*SMALLEST*/);

  } while (s.heap_len >= 2);

  s.heap[--s.heap_max] = s.heap[1/*SMALLEST*/];

  /* At this point, the fields freq and dad are set. We can now
   * generate the bit lengths.
   */
  gen_bitlen(s, desc);

  /* The field len is now set, we can generate the bit codes */
  gen_codes(tree, max_code, s.bl_count);
};


/* ===========================================================================
 * Scan a literal or distance tree to determine the frequencies of the codes
 * in the bit length tree.
 */
const scan_tree = (s, tree, max_code) => {
//    deflate_state *s;
//    ct_data *tree;   /* the tree to be scanned */
//    int max_code;    /* and its largest code of non zero frequency */

  let n;                     /* iterates over all tree elements */
  let prevlen = -1;          /* last emitted length */
  let curlen;                /* length of current code */

  let nextlen = tree[0 * 2 + 1]/*.Len*/; /* length of next code */

  let count = 0;             /* repeat count of the current code */
  let max_count = 7;         /* max repeat count */
  let min_count = 4;         /* min repeat count */

  if (nextlen === 0) {
    max_count = 138;
    min_count = 3;
  }
  tree[(max_code + 1) * 2 + 1]/*.Len*/ = 0xffff; /* guard */

  for (n = 0; n <= max_code; n++) {
    curlen = nextlen;
    nextlen = tree[(n + 1) * 2 + 1]/*.Len*/;

    if (++count < max_count && curlen === nextlen) {
      continue;

    } else if (count < min_count) {
      s.bl_tree[curlen * 2]/*.Freq*/ += count;

    } else if (curlen !== 0) {

      if (curlen !== prevlen) { s.bl_tree[curlen * 2]/*.Freq*/++; }
      s.bl_tree[REP_3_6 * 2]/*.Freq*/++;

    } else if (count <= 10) {
      s.bl_tree[REPZ_3_10 * 2]/*.Freq*/++;

    } else {
      s.bl_tree[REPZ_11_138 * 2]/*.Freq*/++;
    }

    count = 0;
    prevlen = curlen;

    if (nextlen === 0) {
      max_count = 138;
      min_count = 3;

    } else if (curlen === nextlen) {
      max_count = 6;
      min_count = 3;

    } else {
      max_count = 7;
      min_count = 4;
    }
  }
};


/* ===========================================================================
 * Send a literal or distance tree in compressed form, using the codes in
 * bl_tree.
 */
const send_tree = (s, tree, max_code) => {
//    deflate_state *s;
//    ct_data *tree; /* the tree to be scanned */
//    int max_code;       /* and its largest code of non zero frequency */

  let n;                     /* iterates over all tree elements */
  let prevlen = -1;          /* last emitted length */
  let curlen;                /* length of current code */

  let nextlen = tree[0 * 2 + 1]/*.Len*/; /* length of next code */

  let count = 0;             /* repeat count of the current code */
  let max_count = 7;         /* max repeat count */
  let min_count = 4;         /* min repeat count */

  /* tree[max_code+1].Len = -1; */  /* guard already set */
  if (nextlen === 0) {
    max_count = 138;
    min_count = 3;
  }

  for (n = 0; n <= max_code; n++) {
    curlen = nextlen;
    nextlen = tree[(n + 1) * 2 + 1]/*.Len*/;

    if (++count < max_count && curlen === nextlen) {
      continue;

    } else if (count < min_count) {
      do { send_code(s, curlen, s.bl_tree); } while (--count !== 0);

    } else if (curlen !== 0) {
      if (curlen !== prevlen) {
        send_code(s, curlen, s.bl_tree);
        count--;
      }
      //Assert(count >= 3 && count <= 6, " 3_6?");
      send_code(s, REP_3_6, s.bl_tree);
      send_bits(s, count - 3, 2);

    } else if (count <= 10) {
      send_code(s, REPZ_3_10, s.bl_tree);
      send_bits(s, count - 3, 3);

    } else {
      send_code(s, REPZ_11_138, s.bl_tree);
      send_bits(s, count - 11, 7);
    }

    count = 0;
    prevlen = curlen;
    if (nextlen === 0) {
      max_count = 138;
      min_count = 3;

    } else if (curlen === nextlen) {
      max_count = 6;
      min_count = 3;

    } else {
      max_count = 7;
      min_count = 4;
    }
  }
};


/* ===========================================================================
 * Construct the Huffman tree for the bit lengths and return the index in
 * bl_order of the last bit length code to send.
 */
const build_bl_tree = (s) => {

  let max_blindex;  /* index of last bit length code of non zero freq */

  /* Determine the bit length frequencies for literal and distance trees */
  scan_tree(s, s.dyn_ltree, s.l_desc.max_code);
  scan_tree(s, s.dyn_dtree, s.d_desc.max_code);

  /* Build the bit length tree: */
  build_tree(s, s.bl_desc);
  /* opt_len now includes the length of the tree representations, except
   * the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
   */

  /* Determine the number of bit length codes to send. The pkzip format
   * requires that at least 4 bit length codes be sent. (appnote.txt says
   * 3 but the actual value used is 4.)
   */
  for (max_blindex = BL_CODES$1 - 1; max_blindex >= 3; max_blindex--) {
    if (s.bl_tree[bl_order[max_blindex] * 2 + 1]/*.Len*/ !== 0) {
      break;
    }
  }
  /* Update opt_len to include the bit length tree and counts */
  s.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;
  //Tracev((stderr, "\ndyn trees: dyn %ld, stat %ld",
  //        s->opt_len, s->static_len));

  return max_blindex;
};


/* ===========================================================================
 * Send the header for a block using dynamic Huffman trees: the counts, the
 * lengths of the bit length codes, the literal tree and the distance tree.
 * IN assertion: lcodes >= 257, dcodes >= 1, blcodes >= 4.
 */
const send_all_trees = (s, lcodes, dcodes, blcodes) => {
//    deflate_state *s;
//    int lcodes, dcodes, blcodes; /* number of codes for each tree */

  let rank;                    /* index in bl_order */

  //Assert (lcodes >= 257 && dcodes >= 1 && blcodes >= 4, "not enough codes");
  //Assert (lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES,
  //        "too many codes");
  //Tracev((stderr, "\nbl counts: "));
  send_bits(s, lcodes - 257, 5); /* not +255 as stated in appnote.txt */
  send_bits(s, dcodes - 1,   5);
  send_bits(s, blcodes - 4,  4); /* not -3 as stated in appnote.txt */
  for (rank = 0; rank < blcodes; rank++) {
    //Tracev((stderr, "\nbl code %2d ", bl_order[rank]));
    send_bits(s, s.bl_tree[bl_order[rank] * 2 + 1]/*.Len*/, 3);
  }
  //Tracev((stderr, "\nbl tree: sent %ld", s->bits_sent));

  send_tree(s, s.dyn_ltree, lcodes - 1); /* literal tree */
  //Tracev((stderr, "\nlit tree: sent %ld", s->bits_sent));

  send_tree(s, s.dyn_dtree, dcodes - 1); /* distance tree */
  //Tracev((stderr, "\ndist tree: sent %ld", s->bits_sent));
};


/* ===========================================================================
 * Check if the data type is TEXT or BINARY, using the following algorithm:
 * - TEXT if the two conditions below are satisfied:
 *    a) There are no non-portable control characters belonging to the
 *       "block list" (0..6, 14..25, 28..31).
 *    b) There is at least one printable character belonging to the
 *       "allow list" (9 {TAB}, 10 {LF}, 13 {CR}, 32..255).
 * - BINARY otherwise.
 * - The following partially-portable control characters form a
 *   "gray list" that is ignored in this detection algorithm:
 *   (7 {BEL}, 8 {BS}, 11 {VT}, 12 {FF}, 26 {SUB}, 27 {ESC}).
 * IN assertion: the fields Freq of dyn_ltree are set.
 */
const detect_data_type = (s) => {
  /* block_mask is the bit mask of block-listed bytes
   * set bits 0..6, 14..25, and 28..31
   * 0xf3ffc07f = binary 11110011111111111100000001111111
   */
  let block_mask = 0xf3ffc07f;
  let n;

  /* Check for non-textual ("block-listed") bytes. */
  for (n = 0; n <= 31; n++, block_mask >>>= 1) {
    if ((block_mask & 1) && (s.dyn_ltree[n * 2]/*.Freq*/ !== 0)) {
      return Z_BINARY;
    }
  }

  /* Check for textual ("allow-listed") bytes. */
  if (s.dyn_ltree[9 * 2]/*.Freq*/ !== 0 || s.dyn_ltree[10 * 2]/*.Freq*/ !== 0 ||
      s.dyn_ltree[13 * 2]/*.Freq*/ !== 0) {
    return Z_TEXT;
  }
  for (n = 32; n < LITERALS$1; n++) {
    if (s.dyn_ltree[n * 2]/*.Freq*/ !== 0) {
      return Z_TEXT;
    }
  }

  /* There are no "block-listed" or "allow-listed" bytes:
   * this stream either is empty or has tolerated ("gray-listed") bytes only.
   */
  return Z_BINARY;
};


let static_init_done = false;

/* ===========================================================================
 * Initialize the tree data structures for a new zlib stream.
 */
const _tr_init$1 = (s) =>
{

  if (!static_init_done) {
    tr_static_init();
    static_init_done = true;
  }

  s.l_desc  = new TreeDesc(s.dyn_ltree, static_l_desc);
  s.d_desc  = new TreeDesc(s.dyn_dtree, static_d_desc);
  s.bl_desc = new TreeDesc(s.bl_tree, static_bl_desc);

  s.bi_buf = 0;
  s.bi_valid = 0;

  /* Initialize the first block of the first file: */
  init_block(s);
};


/* ===========================================================================
 * Send a stored block
 */
const _tr_stored_block$1 = (s, buf, stored_len, last) => {
//DeflateState *s;
//charf *buf;       /* input block */
//ulg stored_len;   /* length of input block */
//int last;         /* one if this is the last block for a file */

  send_bits(s, (STORED_BLOCK << 1) + (last ? 1 : 0), 3);    /* send block type */
  bi_windup(s);        /* align on byte boundary */
  put_short(s, stored_len);
  put_short(s, ~stored_len);
  if (stored_len) {
    s.pending_buf.set(s.window.subarray(buf, buf + stored_len), s.pending);
  }
  s.pending += stored_len;
};


/* ===========================================================================
 * Send one empty static block to give enough lookahead for inflate.
 * This takes 10 bits, of which 7 may remain in the bit buffer.
 */
const _tr_align$1 = (s) => {
  send_bits(s, STATIC_TREES << 1, 3);
  send_code(s, END_BLOCK, static_ltree);
  bi_flush(s);
};


/* ===========================================================================
 * Determine the best encoding for the current block: dynamic trees, static
 * trees or store, and write out the encoded block.
 */
const _tr_flush_block$1 = (s, buf, stored_len, last) => {
//DeflateState *s;
//charf *buf;       /* input block, or NULL if too old */
//ulg stored_len;   /* length of input block */
//int last;         /* one if this is the last block for a file */

  let opt_lenb, static_lenb;  /* opt_len and static_len in bytes */
  let max_blindex = 0;        /* index of last bit length code of non zero freq */

  /* Build the Huffman trees unless a stored block is forced */
  if (s.level > 0) {

    /* Check if the file is binary or text */
    if (s.strm.data_type === Z_UNKNOWN$1) {
      s.strm.data_type = detect_data_type(s);
    }

    /* Construct the literal and distance trees */
    build_tree(s, s.l_desc);
    // Tracev((stderr, "\nlit data: dyn %ld, stat %ld", s->opt_len,
    //        s->static_len));

    build_tree(s, s.d_desc);
    // Tracev((stderr, "\ndist data: dyn %ld, stat %ld", s->opt_len,
    //        s->static_len));
    /* At this point, opt_len and static_len are the total bit lengths of
     * the compressed block data, excluding the tree representations.
     */

    /* Build the bit length tree for the above two trees, and get the index
     * in bl_order of the last bit length code to send.
     */
    max_blindex = build_bl_tree(s);

    /* Determine the best encoding. Compute the block lengths in bytes. */
    opt_lenb = (s.opt_len + 3 + 7) >>> 3;
    static_lenb = (s.static_len + 3 + 7) >>> 3;

    // Tracev((stderr, "\nopt %lu(%lu) stat %lu(%lu) stored %lu lit %u ",
    //        opt_lenb, s->opt_len, static_lenb, s->static_len, stored_len,
    //        s->sym_next / 3));

    if (static_lenb <= opt_lenb) { opt_lenb = static_lenb; }

  } else {
    // Assert(buf != (char*)0, "lost buf");
    opt_lenb = static_lenb = stored_len + 5; /* force a stored block */
  }

  if ((stored_len + 4 <= opt_lenb) && (buf !== -1)) {
    /* 4: two words for the lengths */

    /* The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
     * Otherwise we can't have processed more than WSIZE input bytes since
     * the last block flush, because compression would have been
     * successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
     * transform a block into a stored block.
     */
    _tr_stored_block$1(s, buf, stored_len, last);

  } else if (s.strategy === Z_FIXED$1 || static_lenb === opt_lenb) {

    send_bits(s, (STATIC_TREES << 1) + (last ? 1 : 0), 3);
    compress_block(s, static_ltree, static_dtree);

  } else {
    send_bits(s, (DYN_TREES << 1) + (last ? 1 : 0), 3);
    send_all_trees(s, s.l_desc.max_code + 1, s.d_desc.max_code + 1, max_blindex + 1);
    compress_block(s, s.dyn_ltree, s.dyn_dtree);
  }
  // Assert (s->compressed_len == s->bits_sent, "bad compressed size");
  /* The above check is made mod 2^32, for files larger than 512 MB
   * and uLong implemented on 32 bits.
   */
  init_block(s);

  if (last) {
    bi_windup(s);
  }
  // Tracev((stderr,"\ncomprlen %lu(%lu) ", s->compressed_len>>3,
  //       s->compressed_len-7*last));
};

/* ===========================================================================
 * Save the match info and tally the frequency counts. Return true if
 * the current block must be flushed.
 */
const _tr_tally$1 = (s, dist, lc) => {
//    deflate_state *s;
//    unsigned dist;  /* distance of matched string */
//    unsigned lc;    /* match length-MIN_MATCH or unmatched char (if dist==0) */

  s.pending_buf[s.sym_buf + s.sym_next++] = dist;
  s.pending_buf[s.sym_buf + s.sym_next++] = dist >> 8;
  s.pending_buf[s.sym_buf + s.sym_next++] = lc;
  if (dist === 0) {
    /* lc is the unmatched char */
    s.dyn_ltree[lc * 2]/*.Freq*/++;
  } else {
    s.matches++;
    /* Here, lc is the match length - MIN_MATCH */
    dist--;             /* dist = match distance - 1 */
    //Assert((ush)dist < (ush)MAX_DIST(s) &&
    //       (ush)lc <= (ush)(MAX_MATCH-MIN_MATCH) &&
    //       (ush)d_code(dist) < (ush)D_CODES,  "_tr_tally: bad match");

    s.dyn_ltree[(_length_code[lc] + LITERALS$1 + 1) * 2]/*.Freq*/++;
    s.dyn_dtree[d_code(dist) * 2]/*.Freq*/++;
  }

  return (s.sym_next === s.sym_end);
};

var _tr_init_1  = _tr_init$1;
var _tr_stored_block_1 = _tr_stored_block$1;
var _tr_flush_block_1  = _tr_flush_block$1;
var _tr_tally_1 = _tr_tally$1;
var _tr_align_1 = _tr_align$1;

var trees = {
	_tr_init: _tr_init_1,
	_tr_stored_block: _tr_stored_block_1,
	_tr_flush_block: _tr_flush_block_1,
	_tr_tally: _tr_tally_1,
	_tr_align: _tr_align_1
};

// Note: adler32 takes 12% for level 0 and 2% for level 6.
// It isn't worth it to make additional optimizations as in original.
// Small size is preferable.

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

const adler32 = (adler, buf, len, pos) => {
  let s1 = (adler & 0xffff) |0,
      s2 = ((adler >>> 16) & 0xffff) |0,
      n = 0;

  while (len !== 0) {
    // Set limit ~ twice less than 5552, to keep
    // s2 in 31-bits, because we force signed ints.
    // in other case %= will fail.
    n = len > 2000 ? 2000 : len;
    len -= n;

    do {
      s1 = (s1 + buf[pos++]) |0;
      s2 = (s2 + s1) |0;
    } while (--n);

    s1 %= 65521;
    s2 %= 65521;
  }

  return (s1 | (s2 << 16)) |0;
};


var adler32_1 = adler32;

// Note: we can't get significant speed boost here.
// So write code to minimize size - no pregenerated tables
// and array tools dependencies.

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

// Use ordinary array, since untyped makes no boost here
const makeTable = () => {
  let c, table = [];

  for (var n = 0; n < 256; n++) {
    c = n;
    for (var k = 0; k < 8; k++) {
      c = ((c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1));
    }
    table[n] = c;
  }

  return table;
};

// Create table on load. Just 255 signed longs. Not a problem.
const crcTable$1 = new Uint32Array(makeTable());


const crc32 = (crc, buf, len, pos) => {
  const t = crcTable$1;
  const end = pos + len;

  crc ^= -1;

  for (let i = pos; i < end; i++) {
    crc = (crc >>> 8) ^ t[(crc ^ buf[i]) & 0xFF];
  }

  return (crc ^ (-1)); // >>> 0;
};


var crc32_1 = crc32;

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

var messages = {
  2:      'need dictionary',     /* Z_NEED_DICT       2  */
  1:      'stream end',          /* Z_STREAM_END      1  */
  0:      '',                    /* Z_OK              0  */
  '-1':   'file error',          /* Z_ERRNO         (-1) */
  '-2':   'stream error',        /* Z_STREAM_ERROR  (-2) */
  '-3':   'data error',          /* Z_DATA_ERROR    (-3) */
  '-4':   'insufficient memory', /* Z_MEM_ERROR     (-4) */
  '-5':   'buffer error',        /* Z_BUF_ERROR     (-5) */
  '-6':   'incompatible version' /* Z_VERSION_ERROR (-6) */
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

var constants$2 = {

  /* Allowed flush values; see deflate() and inflate() below for details */
  Z_NO_FLUSH:         0,
  Z_PARTIAL_FLUSH:    1,
  Z_SYNC_FLUSH:       2,
  Z_FULL_FLUSH:       3,
  Z_FINISH:           4,
  Z_BLOCK:            5,
  Z_TREES:            6,

  /* Return codes for the compression/decompression functions. Negative values
  * are errors, positive values are used for special but normal events.
  */
  Z_OK:               0,
  Z_STREAM_END:       1,
  Z_NEED_DICT:        2,
  Z_ERRNO:           -1,
  Z_STREAM_ERROR:    -2,
  Z_DATA_ERROR:      -3,
  Z_MEM_ERROR:       -4,
  Z_BUF_ERROR:       -5,
  //Z_VERSION_ERROR: -6,

  /* compression levels */
  Z_NO_COMPRESSION:         0,
  Z_BEST_SPEED:             1,
  Z_BEST_COMPRESSION:       9,
  Z_DEFAULT_COMPRESSION:   -1,


  Z_FILTERED:               1,
  Z_HUFFMAN_ONLY:           2,
  Z_RLE:                    3,
  Z_FIXED:                  4,
  Z_DEFAULT_STRATEGY:       0,

  /* Possible values of the data_type field (though see inflate()) */
  Z_BINARY:                 0,
  Z_TEXT:                   1,
  //Z_ASCII:                1, // = Z_TEXT (deprecated)
  Z_UNKNOWN:                2,

  /* The deflate compression method */
  Z_DEFLATED:               8
  //Z_NULL:                 null // Use -1 or null inline, depending on var type
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

const { _tr_init, _tr_stored_block, _tr_flush_block, _tr_tally, _tr_align } = trees;




/* Public constants ==========================================================*/
/* ===========================================================================*/

const {
  Z_NO_FLUSH: Z_NO_FLUSH$2, Z_PARTIAL_FLUSH, Z_FULL_FLUSH: Z_FULL_FLUSH$1, Z_FINISH: Z_FINISH$3, Z_BLOCK: Z_BLOCK$1,
  Z_OK: Z_OK$3, Z_STREAM_END: Z_STREAM_END$3, Z_STREAM_ERROR: Z_STREAM_ERROR$2, Z_DATA_ERROR: Z_DATA_ERROR$2, Z_BUF_ERROR: Z_BUF_ERROR$1,
  Z_DEFAULT_COMPRESSION: Z_DEFAULT_COMPRESSION$1,
  Z_FILTERED, Z_HUFFMAN_ONLY, Z_RLE, Z_FIXED, Z_DEFAULT_STRATEGY: Z_DEFAULT_STRATEGY$1,
  Z_UNKNOWN,
  Z_DEFLATED: Z_DEFLATED$2
} = constants$2;

/*============================================================================*/


const MAX_MEM_LEVEL = 9;
/* Maximum value for memLevel in deflateInit2 */
const MAX_WBITS$1 = 15;
/* 32K LZ77 window */
const DEF_MEM_LEVEL = 8;


const LENGTH_CODES  = 29;
/* number of length codes, not counting the special END_BLOCK code */
const LITERALS      = 256;
/* number of literal bytes 0..255 */
const L_CODES       = LITERALS + 1 + LENGTH_CODES;
/* number of Literal or Length codes, including the END_BLOCK code */
const D_CODES       = 30;
/* number of distance codes */
const BL_CODES      = 19;
/* number of codes used to transfer the bit lengths */
const HEAP_SIZE     = 2 * L_CODES + 1;
/* maximum heap size */
const MAX_BITS  = 15;
/* All codes must not exceed MAX_BITS bits */

const MIN_MATCH = 3;
const MAX_MATCH = 258;
const MIN_LOOKAHEAD = (MAX_MATCH + MIN_MATCH + 1);

const PRESET_DICT = 0x20;

const INIT_STATE    =  42;    /* zlib header -> BUSY_STATE */
//#ifdef GZIP
const GZIP_STATE    =  57;    /* gzip header -> BUSY_STATE | EXTRA_STATE */
//#endif
const EXTRA_STATE   =  69;    /* gzip extra block -> NAME_STATE */
const NAME_STATE    =  73;    /* gzip file name -> COMMENT_STATE */
const COMMENT_STATE =  91;    /* gzip comment -> HCRC_STATE */
const HCRC_STATE    = 103;    /* gzip header CRC -> BUSY_STATE */
const BUSY_STATE    = 113;    /* deflate -> FINISH_STATE */
const FINISH_STATE  = 666;    /* stream complete */

const BS_NEED_MORE      = 1; /* block not completed, need more input or more output */
const BS_BLOCK_DONE     = 2; /* block flush performed */
const BS_FINISH_STARTED = 3; /* finish started, need only more output at next deflate */
const BS_FINISH_DONE    = 4; /* finish done, accept no more input or output */

const OS_CODE = 0x03; // Unix :) . Don't detect, use this default.

const err = (strm, errorCode) => {
  strm.msg = messages[errorCode];
  return errorCode;
};

const rank = (f) => {
  return ((f) * 2) - ((f) > 4 ? 9 : 0);
};

const zero = (buf) => {
  let len = buf.length; while (--len >= 0) { buf[len] = 0; }
};

/* ===========================================================================
 * Slide the hash table when sliding the window down (could be avoided with 32
 * bit values at the expense of memory usage). We slide even when level == 0 to
 * keep the hash table consistent if we switch back to level > 0 later.
 */
const slide_hash = (s) => {
  let n, m;
  let p;
  let wsize = s.w_size;

  n = s.hash_size;
  p = n;
  do {
    m = s.head[--p];
    s.head[p] = (m >= wsize ? m - wsize : 0);
  } while (--n);
  n = wsize;
//#ifndef FASTEST
  p = n;
  do {
    m = s.prev[--p];
    s.prev[p] = (m >= wsize ? m - wsize : 0);
    /* If n is not on any hash chain, prev[n] is garbage but
     * its value will never be used.
     */
  } while (--n);
//#endif
};

/* eslint-disable new-cap */
let HASH_ZLIB = (s, prev, data) => ((prev << s.hash_shift) ^ data) & s.hash_mask;
// This hash causes less collisions, https://github.com/nodeca/pako/issues/135
// But breaks binary compatibility
//let HASH_FAST = (s, prev, data) => ((prev << 8) + (prev >> 8) + (data << 4)) & s.hash_mask;
let HASH = HASH_ZLIB;


/* =========================================================================
 * Flush as much pending output as possible. All deflate() output, except for
 * some deflate_stored() output, goes through this function so some
 * applications may wish to modify it to avoid allocating a large
 * strm->next_out buffer and copying into it. (See also read_buf()).
 */
const flush_pending = (strm) => {
  const s = strm.state;

  //_tr_flush_bits(s);
  let len = s.pending;
  if (len > strm.avail_out) {
    len = strm.avail_out;
  }
  if (len === 0) { return; }

  strm.output.set(s.pending_buf.subarray(s.pending_out, s.pending_out + len), strm.next_out);
  strm.next_out  += len;
  s.pending_out  += len;
  strm.total_out += len;
  strm.avail_out -= len;
  s.pending      -= len;
  if (s.pending === 0) {
    s.pending_out = 0;
  }
};


const flush_block_only = (s, last) => {
  _tr_flush_block(s, (s.block_start >= 0 ? s.block_start : -1), s.strstart - s.block_start, last);
  s.block_start = s.strstart;
  flush_pending(s.strm);
};


const put_byte = (s, b) => {
  s.pending_buf[s.pending++] = b;
};


/* =========================================================================
 * Put a short in the pending buffer. The 16-bit value is put in MSB order.
 * IN assertion: the stream state is correct and there is enough room in
 * pending_buf.
 */
const putShortMSB = (s, b) => {

  //  put_byte(s, (Byte)(b >> 8));
//  put_byte(s, (Byte)(b & 0xff));
  s.pending_buf[s.pending++] = (b >>> 8) & 0xff;
  s.pending_buf[s.pending++] = b & 0xff;
};


/* ===========================================================================
 * Read a new buffer from the current input stream, update the adler32
 * and total number of bytes read.  All deflate() input goes through
 * this function so some applications may wish to modify it to avoid
 * allocating a large strm->input buffer and copying from it.
 * (See also flush_pending()).
 */
const read_buf = (strm, buf, start, size) => {

  let len = strm.avail_in;

  if (len > size) { len = size; }
  if (len === 0) { return 0; }

  strm.avail_in -= len;

  // zmemcpy(buf, strm->next_in, len);
  buf.set(strm.input.subarray(strm.next_in, strm.next_in + len), start);
  if (strm.state.wrap === 1) {
    strm.adler = adler32_1(strm.adler, buf, len, start);
  }

  else if (strm.state.wrap === 2) {
    strm.adler = crc32_1(strm.adler, buf, len, start);
  }

  strm.next_in += len;
  strm.total_in += len;

  return len;
};


/* ===========================================================================
 * Set match_start to the longest match starting at the given string and
 * return its length. Matches shorter or equal to prev_length are discarded,
 * in which case the result is equal to prev_length and match_start is
 * garbage.
 * IN assertions: cur_match is the head of the hash chain for the current
 *   string (strstart) and its distance is <= MAX_DIST, and prev_length >= 1
 * OUT assertion: the match length is not greater than s->lookahead.
 */
const longest_match = (s, cur_match) => {

  let chain_length = s.max_chain_length;      /* max hash chain length */
  let scan = s.strstart; /* current string */
  let match;                       /* matched string */
  let len;                           /* length of current match */
  let best_len = s.prev_length;              /* best match length so far */
  let nice_match = s.nice_match;             /* stop if match long enough */
  const limit = (s.strstart > (s.w_size - MIN_LOOKAHEAD)) ?
      s.strstart - (s.w_size - MIN_LOOKAHEAD) : 0/*NIL*/;

  const _win = s.window; // shortcut

  const wmask = s.w_mask;
  const prev  = s.prev;

  /* Stop when cur_match becomes <= limit. To simplify the code,
   * we prevent matches with the string of window index 0.
   */

  const strend = s.strstart + MAX_MATCH;
  let scan_end1  = _win[scan + best_len - 1];
  let scan_end   = _win[scan + best_len];

  /* The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
   * It is easy to get rid of this optimization if necessary.
   */
  // Assert(s->hash_bits >= 8 && MAX_MATCH == 258, "Code too clever");

  /* Do not waste too much time if we already have a good match: */
  if (s.prev_length >= s.good_match) {
    chain_length >>= 2;
  }
  /* Do not look for matches beyond the end of the input. This is necessary
   * to make deflate deterministic.
   */
  if (nice_match > s.lookahead) { nice_match = s.lookahead; }

  // Assert((ulg)s->strstart <= s->window_size-MIN_LOOKAHEAD, "need lookahead");

  do {
    // Assert(cur_match < s->strstart, "no future");
    match = cur_match;

    /* Skip to next match if the match length cannot increase
     * or if the match length is less than 2.  Note that the checks below
     * for insufficient lookahead only occur occasionally for performance
     * reasons.  Therefore uninitialized memory will be accessed, and
     * conditional jumps will be made that depend on those values.
     * However the length of the match is limited to the lookahead, so
     * the output of deflate is not affected by the uninitialized values.
     */

    if (_win[match + best_len]     !== scan_end  ||
        _win[match + best_len - 1] !== scan_end1 ||
        _win[match]                !== _win[scan] ||
        _win[++match]              !== _win[scan + 1]) {
      continue;
    }

    /* The check at best_len-1 can be removed because it will be made
     * again later. (This heuristic is not always a win.)
     * It is not necessary to compare scan[2] and match[2] since they
     * are always equal when the other bytes match, given that
     * the hash keys are equal and that HASH_BITS >= 8.
     */
    scan += 2;
    match++;
    // Assert(*scan == *match, "match[2]?");

    /* We check for insufficient lookahead only every 8th comparison;
     * the 256th check will be made at strstart+258.
     */
    do {
      /*jshint noempty:false*/
    } while (_win[++scan] === _win[++match] && _win[++scan] === _win[++match] &&
             _win[++scan] === _win[++match] && _win[++scan] === _win[++match] &&
             _win[++scan] === _win[++match] && _win[++scan] === _win[++match] &&
             _win[++scan] === _win[++match] && _win[++scan] === _win[++match] &&
             scan < strend);

    // Assert(scan <= s->window+(unsigned)(s->window_size-1), "wild scan");

    len = MAX_MATCH - (strend - scan);
    scan = strend - MAX_MATCH;

    if (len > best_len) {
      s.match_start = cur_match;
      best_len = len;
      if (len >= nice_match) {
        break;
      }
      scan_end1  = _win[scan + best_len - 1];
      scan_end   = _win[scan + best_len];
    }
  } while ((cur_match = prev[cur_match & wmask]) > limit && --chain_length !== 0);

  if (best_len <= s.lookahead) {
    return best_len;
  }
  return s.lookahead;
};


/* ===========================================================================
 * Fill the window when the lookahead becomes insufficient.
 * Updates strstart and lookahead.
 *
 * IN assertion: lookahead < MIN_LOOKAHEAD
 * OUT assertions: strstart <= window_size-MIN_LOOKAHEAD
 *    At least one byte has been read, or avail_in == 0; reads are
 *    performed for at least two bytes (required for the zip translate_eol
 *    option -- not supported here).
 */
const fill_window = (s) => {

  const _w_size = s.w_size;
  let n, more, str;

  //Assert(s->lookahead < MIN_LOOKAHEAD, "already enough lookahead");

  do {
    more = s.window_size - s.lookahead - s.strstart;

    // JS ints have 32 bit, block below not needed
    /* Deal with !@#$% 64K limit: */
    //if (sizeof(int) <= 2) {
    //    if (more == 0 && s->strstart == 0 && s->lookahead == 0) {
    //        more = wsize;
    //
    //  } else if (more == (unsigned)(-1)) {
    //        /* Very unlikely, but possible on 16 bit machine if
    //         * strstart == 0 && lookahead == 1 (input done a byte at time)
    //         */
    //        more--;
    //    }
    //}


    /* If the window is almost full and there is insufficient lookahead,
     * move the upper half to the lower one to make room in the upper half.
     */
    if (s.strstart >= _w_size + (_w_size - MIN_LOOKAHEAD)) {

      s.window.set(s.window.subarray(_w_size, _w_size + _w_size - more), 0);
      s.match_start -= _w_size;
      s.strstart -= _w_size;
      /* we now have strstart >= MAX_DIST */
      s.block_start -= _w_size;
      if (s.insert > s.strstart) {
        s.insert = s.strstart;
      }
      slide_hash(s);
      more += _w_size;
    }
    if (s.strm.avail_in === 0) {
      break;
    }

    /* If there was no sliding:
     *    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
     *    more == window_size - lookahead - strstart
     * => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
     * => more >= window_size - 2*WSIZE + 2
     * In the BIG_MEM or MMAP case (not yet supported),
     *   window_size == input_size + MIN_LOOKAHEAD  &&
     *   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
     * Otherwise, window_size == 2*WSIZE so more >= 2.
     * If there was sliding, more >= WSIZE. So in all cases, more >= 2.
     */
    //Assert(more >= 2, "more < 2");
    n = read_buf(s.strm, s.window, s.strstart + s.lookahead, more);
    s.lookahead += n;

    /* Initialize the hash value now that we have some input: */
    if (s.lookahead + s.insert >= MIN_MATCH) {
      str = s.strstart - s.insert;
      s.ins_h = s.window[str];

      /* UPDATE_HASH(s, s->ins_h, s->window[str + 1]); */
      s.ins_h = HASH(s, s.ins_h, s.window[str + 1]);
//#if MIN_MATCH != 3
//        Call update_hash() MIN_MATCH-3 more times
//#endif
      while (s.insert) {
        /* UPDATE_HASH(s, s->ins_h, s->window[str + MIN_MATCH-1]); */
        s.ins_h = HASH(s, s.ins_h, s.window[str + MIN_MATCH - 1]);

        s.prev[str & s.w_mask] = s.head[s.ins_h];
        s.head[s.ins_h] = str;
        str++;
        s.insert--;
        if (s.lookahead + s.insert < MIN_MATCH) {
          break;
        }
      }
    }
    /* If the whole input has less than MIN_MATCH bytes, ins_h is garbage,
     * but this is not important since only literal bytes will be emitted.
     */

  } while (s.lookahead < MIN_LOOKAHEAD && s.strm.avail_in !== 0);

  /* If the WIN_INIT bytes after the end of the current data have never been
   * written, then zero those bytes in order to avoid memory check reports of
   * the use of uninitialized (or uninitialised as Julian writes) bytes by
   * the longest match routines.  Update the high water mark for the next
   * time through here.  WIN_INIT is set to MAX_MATCH since the longest match
   * routines allow scanning to strstart + MAX_MATCH, ignoring lookahead.
   */
//  if (s.high_water < s.window_size) {
//    const curr = s.strstart + s.lookahead;
//    let init = 0;
//
//    if (s.high_water < curr) {
//      /* Previous high water mark below current data -- zero WIN_INIT
//       * bytes or up to end of window, whichever is less.
//       */
//      init = s.window_size - curr;
//      if (init > WIN_INIT)
//        init = WIN_INIT;
//      zmemzero(s->window + curr, (unsigned)init);
//      s->high_water = curr + init;
//    }
//    else if (s->high_water < (ulg)curr + WIN_INIT) {
//      /* High water mark at or above current data, but below current data
//       * plus WIN_INIT -- zero out to current data plus WIN_INIT, or up
//       * to end of window, whichever is less.
//       */
//      init = (ulg)curr + WIN_INIT - s->high_water;
//      if (init > s->window_size - s->high_water)
//        init = s->window_size - s->high_water;
//      zmemzero(s->window + s->high_water, (unsigned)init);
//      s->high_water += init;
//    }
//  }
//
//  Assert((ulg)s->strstart <= s->window_size - MIN_LOOKAHEAD,
//    "not enough room for search");
};

/* ===========================================================================
 * Copy without compression as much as possible from the input stream, return
 * the current block state.
 *
 * In case deflateParams() is used to later switch to a non-zero compression
 * level, s->matches (otherwise unused when storing) keeps track of the number
 * of hash table slides to perform. If s->matches is 1, then one hash table
 * slide will be done when switching. If s->matches is 2, the maximum value
 * allowed here, then the hash table will be cleared, since two or more slides
 * is the same as a clear.
 *
 * deflate_stored() is written to minimize the number of times an input byte is
 * copied. It is most efficient with large input and output buffers, which
 * maximizes the opportunites to have a single copy from next_in to next_out.
 */
const deflate_stored = (s, flush) => {

  /* Smallest worthy block size when not flushing or finishing. By default
   * this is 32K. This can be as small as 507 bytes for memLevel == 1. For
   * large input and output buffers, the stored block size will be larger.
   */
  let min_block = s.pending_buf_size - 5 > s.w_size ? s.w_size : s.pending_buf_size - 5;

  /* Copy as many min_block or larger stored blocks directly to next_out as
   * possible. If flushing, copy the remaining available input to next_out as
   * stored blocks, if there is enough space.
   */
  let len, left, have, last = 0;
  let used = s.strm.avail_in;
  do {
    /* Set len to the maximum size block that we can copy directly with the
     * available input data and output space. Set left to how much of that
     * would be copied from what's left in the window.
     */
    len = 65535/* MAX_STORED */;     /* maximum deflate stored block length */
    have = (s.bi_valid + 42) >> 3;     /* number of header bytes */
    if (s.strm.avail_out < have) {         /* need room for header */
      break;
    }
      /* maximum stored block length that will fit in avail_out: */
    have = s.strm.avail_out - have;
    left = s.strstart - s.block_start;  /* bytes left in window */
    if (len > left + s.strm.avail_in) {
      len = left + s.strm.avail_in;   /* limit len to the input */
    }
    if (len > have) {
      len = have;             /* limit len to the output */
    }

    /* If the stored block would be less than min_block in length, or if
     * unable to copy all of the available input when flushing, then try
     * copying to the window and the pending buffer instead. Also don't
     * write an empty block when flushing -- deflate() does that.
     */
    if (len < min_block && ((len === 0 && flush !== Z_FINISH$3) ||
                        flush === Z_NO_FLUSH$2 ||
                        len !== left + s.strm.avail_in)) {
      break;
    }

    /* Make a dummy stored block in pending to get the header bytes,
     * including any pending bits. This also updates the debugging counts.
     */
    last = flush === Z_FINISH$3 && len === left + s.strm.avail_in ? 1 : 0;
    _tr_stored_block(s, 0, 0, last);

    /* Replace the lengths in the dummy stored block with len. */
    s.pending_buf[s.pending - 4] = len;
    s.pending_buf[s.pending - 3] = len >> 8;
    s.pending_buf[s.pending - 2] = ~len;
    s.pending_buf[s.pending - 1] = ~len >> 8;

    /* Write the stored block header bytes. */
    flush_pending(s.strm);

//#ifdef ZLIB_DEBUG
//    /* Update debugging counts for the data about to be copied. */
//    s->compressed_len += len << 3;
//    s->bits_sent += len << 3;
//#endif

    /* Copy uncompressed bytes from the window to next_out. */
    if (left) {
      if (left > len) {
        left = len;
      }
      //zmemcpy(s->strm->next_out, s->window + s->block_start, left);
      s.strm.output.set(s.window.subarray(s.block_start, s.block_start + left), s.strm.next_out);
      s.strm.next_out += left;
      s.strm.avail_out -= left;
      s.strm.total_out += left;
      s.block_start += left;
      len -= left;
    }

    /* Copy uncompressed bytes directly from next_in to next_out, updating
     * the check value.
     */
    if (len) {
      read_buf(s.strm, s.strm.output, s.strm.next_out, len);
      s.strm.next_out += len;
      s.strm.avail_out -= len;
      s.strm.total_out += len;
    }
  } while (last === 0);

  /* Update the sliding window with the last s->w_size bytes of the copied
   * data, or append all of the copied data to the existing window if less
   * than s->w_size bytes were copied. Also update the number of bytes to
   * insert in the hash tables, in the event that deflateParams() switches to
   * a non-zero compression level.
   */
  used -= s.strm.avail_in;    /* number of input bytes directly copied */
  if (used) {
    /* If any input was used, then no unused input remains in the window,
     * therefore s->block_start == s->strstart.
     */
    if (used >= s.w_size) {  /* supplant the previous history */
      s.matches = 2;     /* clear hash */
      //zmemcpy(s->window, s->strm->next_in - s->w_size, s->w_size);
      s.window.set(s.strm.input.subarray(s.strm.next_in - s.w_size, s.strm.next_in), 0);
      s.strstart = s.w_size;
      s.insert = s.strstart;
    }
    else {
      if (s.window_size - s.strstart <= used) {
        /* Slide the window down. */
        s.strstart -= s.w_size;
        //zmemcpy(s->window, s->window + s->w_size, s->strstart);
        s.window.set(s.window.subarray(s.w_size, s.w_size + s.strstart), 0);
        if (s.matches < 2) {
          s.matches++;   /* add a pending slide_hash() */
        }
        if (s.insert > s.strstart) {
          s.insert = s.strstart;
        }
      }
      //zmemcpy(s->window + s->strstart, s->strm->next_in - used, used);
      s.window.set(s.strm.input.subarray(s.strm.next_in - used, s.strm.next_in), s.strstart);
      s.strstart += used;
      s.insert += used > s.w_size - s.insert ? s.w_size - s.insert : used;
    }
    s.block_start = s.strstart;
  }
  if (s.high_water < s.strstart) {
    s.high_water = s.strstart;
  }

  /* If the last block was written to next_out, then done. */
  if (last) {
    return BS_FINISH_DONE;
  }

  /* If flushing and all input has been consumed, then done. */
  if (flush !== Z_NO_FLUSH$2 && flush !== Z_FINISH$3 &&
    s.strm.avail_in === 0 && s.strstart === s.block_start) {
    return BS_BLOCK_DONE;
  }

  /* Fill the window with any remaining input. */
  have = s.window_size - s.strstart;
  if (s.strm.avail_in > have && s.block_start >= s.w_size) {
    /* Slide the window down. */
    s.block_start -= s.w_size;
    s.strstart -= s.w_size;
    //zmemcpy(s->window, s->window + s->w_size, s->strstart);
    s.window.set(s.window.subarray(s.w_size, s.w_size + s.strstart), 0);
    if (s.matches < 2) {
      s.matches++;       /* add a pending slide_hash() */
    }
    have += s.w_size;      /* more space now */
    if (s.insert > s.strstart) {
      s.insert = s.strstart;
    }
  }
  if (have > s.strm.avail_in) {
    have = s.strm.avail_in;
  }
  if (have) {
    read_buf(s.strm, s.window, s.strstart, have);
    s.strstart += have;
    s.insert += have > s.w_size - s.insert ? s.w_size - s.insert : have;
  }
  if (s.high_water < s.strstart) {
    s.high_water = s.strstart;
  }

  /* There was not enough avail_out to write a complete worthy or flushed
   * stored block to next_out. Write a stored block to pending instead, if we
   * have enough input for a worthy block, or if flushing and there is enough
   * room for the remaining input as a stored block in the pending buffer.
   */
  have = (s.bi_valid + 42) >> 3;     /* number of header bytes */
    /* maximum stored block length that will fit in pending: */
  have = s.pending_buf_size - have > 65535/* MAX_STORED */ ? 65535/* MAX_STORED */ : s.pending_buf_size - have;
  min_block = have > s.w_size ? s.w_size : have;
  left = s.strstart - s.block_start;
  if (left >= min_block ||
     ((left || flush === Z_FINISH$3) && flush !== Z_NO_FLUSH$2 &&
     s.strm.avail_in === 0 && left <= have)) {
    len = left > have ? have : left;
    last = flush === Z_FINISH$3 && s.strm.avail_in === 0 &&
         len === left ? 1 : 0;
    _tr_stored_block(s, s.block_start, len, last);
    s.block_start += len;
    flush_pending(s.strm);
  }

  /* We've done all we can with the available input and output. */
  return last ? BS_FINISH_STARTED : BS_NEED_MORE;
};


/* ===========================================================================
 * Compress as much as possible from the input stream, return the current
 * block state.
 * This function does not perform lazy evaluation of matches and inserts
 * new strings in the dictionary only for unmatched strings or for short
 * matches. It is used only for the fast compression options.
 */
const deflate_fast = (s, flush) => {

  let hash_head;        /* head of the hash chain */
  let bflush;           /* set if current block must be flushed */

  for (;;) {
    /* Make sure that we always have enough lookahead, except
     * at the end of the input file. We need MAX_MATCH bytes
     * for the next match, plus MIN_MATCH bytes to insert the
     * string following the next match.
     */
    if (s.lookahead < MIN_LOOKAHEAD) {
      fill_window(s);
      if (s.lookahead < MIN_LOOKAHEAD && flush === Z_NO_FLUSH$2) {
        return BS_NEED_MORE;
      }
      if (s.lookahead === 0) {
        break; /* flush the current block */
      }
    }

    /* Insert the string window[strstart .. strstart+2] in the
     * dictionary, and set hash_head to the head of the hash chain:
     */
    hash_head = 0/*NIL*/;
    if (s.lookahead >= MIN_MATCH) {
      /*** INSERT_STRING(s, s.strstart, hash_head); ***/
      s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
      hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
      s.head[s.ins_h] = s.strstart;
      /***/
    }

    /* Find the longest match, discarding those <= prev_length.
     * At this point we have always match_length < MIN_MATCH
     */
    if (hash_head !== 0/*NIL*/ && ((s.strstart - hash_head) <= (s.w_size - MIN_LOOKAHEAD))) {
      /* To simplify the code, we prevent matches with the string
       * of window index 0 (in particular we have to avoid a match
       * of the string with itself at the start of the input file).
       */
      s.match_length = longest_match(s, hash_head);
      /* longest_match() sets match_start */
    }
    if (s.match_length >= MIN_MATCH) {
      // check_match(s, s.strstart, s.match_start, s.match_length); // for debug only

      /*** _tr_tally_dist(s, s.strstart - s.match_start,
                     s.match_length - MIN_MATCH, bflush); ***/
      bflush = _tr_tally(s, s.strstart - s.match_start, s.match_length - MIN_MATCH);

      s.lookahead -= s.match_length;

      /* Insert new strings in the hash table only if the match length
       * is not too large. This saves time but degrades compression.
       */
      if (s.match_length <= s.max_lazy_match/*max_insert_length*/ && s.lookahead >= MIN_MATCH) {
        s.match_length--; /* string at strstart already in table */
        do {
          s.strstart++;
          /*** INSERT_STRING(s, s.strstart, hash_head); ***/
          s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
          hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
          s.head[s.ins_h] = s.strstart;
          /***/
          /* strstart never exceeds WSIZE-MAX_MATCH, so there are
           * always MIN_MATCH bytes ahead.
           */
        } while (--s.match_length !== 0);
        s.strstart++;
      } else
      {
        s.strstart += s.match_length;
        s.match_length = 0;
        s.ins_h = s.window[s.strstart];
        /* UPDATE_HASH(s, s.ins_h, s.window[s.strstart+1]); */
        s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + 1]);

//#if MIN_MATCH != 3
//                Call UPDATE_HASH() MIN_MATCH-3 more times
//#endif
        /* If lookahead < MIN_MATCH, ins_h is garbage, but it does not
         * matter since it will be recomputed at next deflate call.
         */
      }
    } else {
      /* No match, output a literal byte */
      //Tracevv((stderr,"%c", s.window[s.strstart]));
      /*** _tr_tally_lit(s, s.window[s.strstart], bflush); ***/
      bflush = _tr_tally(s, 0, s.window[s.strstart]);

      s.lookahead--;
      s.strstart++;
    }
    if (bflush) {
      /*** FLUSH_BLOCK(s, 0); ***/
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
      /***/
    }
  }
  s.insert = ((s.strstart < (MIN_MATCH - 1)) ? s.strstart : MIN_MATCH - 1);
  if (flush === Z_FINISH$3) {
    /*** FLUSH_BLOCK(s, 1); ***/
    flush_block_only(s, true);
    if (s.strm.avail_out === 0) {
      return BS_FINISH_STARTED;
    }
    /***/
    return BS_FINISH_DONE;
  }
  if (s.sym_next) {
    /*** FLUSH_BLOCK(s, 0); ***/
    flush_block_only(s, false);
    if (s.strm.avail_out === 0) {
      return BS_NEED_MORE;
    }
    /***/
  }
  return BS_BLOCK_DONE;
};

/* ===========================================================================
 * Same as above, but achieves better compression. We use a lazy
 * evaluation for matches: a match is finally adopted only if there is
 * no better match at the next window position.
 */
const deflate_slow = (s, flush) => {

  let hash_head;          /* head of hash chain */
  let bflush;              /* set if current block must be flushed */

  let max_insert;

  /* Process the input block. */
  for (;;) {
    /* Make sure that we always have enough lookahead, except
     * at the end of the input file. We need MAX_MATCH bytes
     * for the next match, plus MIN_MATCH bytes to insert the
     * string following the next match.
     */
    if (s.lookahead < MIN_LOOKAHEAD) {
      fill_window(s);
      if (s.lookahead < MIN_LOOKAHEAD && flush === Z_NO_FLUSH$2) {
        return BS_NEED_MORE;
      }
      if (s.lookahead === 0) { break; } /* flush the current block */
    }

    /* Insert the string window[strstart .. strstart+2] in the
     * dictionary, and set hash_head to the head of the hash chain:
     */
    hash_head = 0/*NIL*/;
    if (s.lookahead >= MIN_MATCH) {
      /*** INSERT_STRING(s, s.strstart, hash_head); ***/
      s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
      hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
      s.head[s.ins_h] = s.strstart;
      /***/
    }

    /* Find the longest match, discarding those <= prev_length.
     */
    s.prev_length = s.match_length;
    s.prev_match = s.match_start;
    s.match_length = MIN_MATCH - 1;

    if (hash_head !== 0/*NIL*/ && s.prev_length < s.max_lazy_match &&
        s.strstart - hash_head <= (s.w_size - MIN_LOOKAHEAD)/*MAX_DIST(s)*/) {
      /* To simplify the code, we prevent matches with the string
       * of window index 0 (in particular we have to avoid a match
       * of the string with itself at the start of the input file).
       */
      s.match_length = longest_match(s, hash_head);
      /* longest_match() sets match_start */

      if (s.match_length <= 5 &&
         (s.strategy === Z_FILTERED || (s.match_length === MIN_MATCH && s.strstart - s.match_start > 4096/*TOO_FAR*/))) {

        /* If prev_match is also MIN_MATCH, match_start is garbage
         * but we will ignore the current match anyway.
         */
        s.match_length = MIN_MATCH - 1;
      }
    }
    /* If there was a match at the previous step and the current
     * match is not better, output the previous match:
     */
    if (s.prev_length >= MIN_MATCH && s.match_length <= s.prev_length) {
      max_insert = s.strstart + s.lookahead - MIN_MATCH;
      /* Do not insert strings in hash table beyond this. */

      //check_match(s, s.strstart-1, s.prev_match, s.prev_length);

      /***_tr_tally_dist(s, s.strstart - 1 - s.prev_match,
                     s.prev_length - MIN_MATCH, bflush);***/
      bflush = _tr_tally(s, s.strstart - 1 - s.prev_match, s.prev_length - MIN_MATCH);
      /* Insert in hash table all strings up to the end of the match.
       * strstart-1 and strstart are already inserted. If there is not
       * enough lookahead, the last two strings are not inserted in
       * the hash table.
       */
      s.lookahead -= s.prev_length - 1;
      s.prev_length -= 2;
      do {
        if (++s.strstart <= max_insert) {
          /*** INSERT_STRING(s, s.strstart, hash_head); ***/
          s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
          hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
          s.head[s.ins_h] = s.strstart;
          /***/
        }
      } while (--s.prev_length !== 0);
      s.match_available = 0;
      s.match_length = MIN_MATCH - 1;
      s.strstart++;

      if (bflush) {
        /*** FLUSH_BLOCK(s, 0); ***/
        flush_block_only(s, false);
        if (s.strm.avail_out === 0) {
          return BS_NEED_MORE;
        }
        /***/
      }

    } else if (s.match_available) {
      /* If there was no match at the previous position, output a
       * single literal. If there was a match but the current match
       * is longer, truncate the previous match to a single literal.
       */
      //Tracevv((stderr,"%c", s->window[s->strstart-1]));
      /*** _tr_tally_lit(s, s.window[s.strstart-1], bflush); ***/
      bflush = _tr_tally(s, 0, s.window[s.strstart - 1]);

      if (bflush) {
        /*** FLUSH_BLOCK_ONLY(s, 0) ***/
        flush_block_only(s, false);
        /***/
      }
      s.strstart++;
      s.lookahead--;
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
    } else {
      /* There is no previous match to compare with, wait for
       * the next step to decide.
       */
      s.match_available = 1;
      s.strstart++;
      s.lookahead--;
    }
  }
  //Assert (flush != Z_NO_FLUSH, "no flush?");
  if (s.match_available) {
    //Tracevv((stderr,"%c", s->window[s->strstart-1]));
    /*** _tr_tally_lit(s, s.window[s.strstart-1], bflush); ***/
    bflush = _tr_tally(s, 0, s.window[s.strstart - 1]);

    s.match_available = 0;
  }
  s.insert = s.strstart < MIN_MATCH - 1 ? s.strstart : MIN_MATCH - 1;
  if (flush === Z_FINISH$3) {
    /*** FLUSH_BLOCK(s, 1); ***/
    flush_block_only(s, true);
    if (s.strm.avail_out === 0) {
      return BS_FINISH_STARTED;
    }
    /***/
    return BS_FINISH_DONE;
  }
  if (s.sym_next) {
    /*** FLUSH_BLOCK(s, 0); ***/
    flush_block_only(s, false);
    if (s.strm.avail_out === 0) {
      return BS_NEED_MORE;
    }
    /***/
  }

  return BS_BLOCK_DONE;
};


/* ===========================================================================
 * For Z_RLE, simply look for runs of bytes, generate matches only of distance
 * one.  Do not maintain a hash table.  (It will be regenerated if this run of
 * deflate switches away from Z_RLE.)
 */
const deflate_rle = (s, flush) => {

  let bflush;            /* set if current block must be flushed */
  let prev;              /* byte at distance one to match */
  let scan, strend;      /* scan goes up to strend for length of run */

  const _win = s.window;

  for (;;) {
    /* Make sure that we always have enough lookahead, except
     * at the end of the input file. We need MAX_MATCH bytes
     * for the longest run, plus one for the unrolled loop.
     */
    if (s.lookahead <= MAX_MATCH) {
      fill_window(s);
      if (s.lookahead <= MAX_MATCH && flush === Z_NO_FLUSH$2) {
        return BS_NEED_MORE;
      }
      if (s.lookahead === 0) { break; } /* flush the current block */
    }

    /* See how many times the previous byte repeats */
    s.match_length = 0;
    if (s.lookahead >= MIN_MATCH && s.strstart > 0) {
      scan = s.strstart - 1;
      prev = _win[scan];
      if (prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan]) {
        strend = s.strstart + MAX_MATCH;
        do {
          /*jshint noempty:false*/
        } while (prev === _win[++scan] && prev === _win[++scan] &&
                 prev === _win[++scan] && prev === _win[++scan] &&
                 prev === _win[++scan] && prev === _win[++scan] &&
                 prev === _win[++scan] && prev === _win[++scan] &&
                 scan < strend);
        s.match_length = MAX_MATCH - (strend - scan);
        if (s.match_length > s.lookahead) {
          s.match_length = s.lookahead;
        }
      }
      //Assert(scan <= s->window+(uInt)(s->window_size-1), "wild scan");
    }

    /* Emit match if have run of MIN_MATCH or longer, else emit literal */
    if (s.match_length >= MIN_MATCH) {
      //check_match(s, s.strstart, s.strstart - 1, s.match_length);

      /*** _tr_tally_dist(s, 1, s.match_length - MIN_MATCH, bflush); ***/
      bflush = _tr_tally(s, 1, s.match_length - MIN_MATCH);

      s.lookahead -= s.match_length;
      s.strstart += s.match_length;
      s.match_length = 0;
    } else {
      /* No match, output a literal byte */
      //Tracevv((stderr,"%c", s->window[s->strstart]));
      /*** _tr_tally_lit(s, s.window[s.strstart], bflush); ***/
      bflush = _tr_tally(s, 0, s.window[s.strstart]);

      s.lookahead--;
      s.strstart++;
    }
    if (bflush) {
      /*** FLUSH_BLOCK(s, 0); ***/
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
      /***/
    }
  }
  s.insert = 0;
  if (flush === Z_FINISH$3) {
    /*** FLUSH_BLOCK(s, 1); ***/
    flush_block_only(s, true);
    if (s.strm.avail_out === 0) {
      return BS_FINISH_STARTED;
    }
    /***/
    return BS_FINISH_DONE;
  }
  if (s.sym_next) {
    /*** FLUSH_BLOCK(s, 0); ***/
    flush_block_only(s, false);
    if (s.strm.avail_out === 0) {
      return BS_NEED_MORE;
    }
    /***/
  }
  return BS_BLOCK_DONE;
};

/* ===========================================================================
 * For Z_HUFFMAN_ONLY, do not look for matches.  Do not maintain a hash table.
 * (It will be regenerated if this run of deflate switches away from Huffman.)
 */
const deflate_huff = (s, flush) => {

  let bflush;             /* set if current block must be flushed */

  for (;;) {
    /* Make sure that we have a literal to write. */
    if (s.lookahead === 0) {
      fill_window(s);
      if (s.lookahead === 0) {
        if (flush === Z_NO_FLUSH$2) {
          return BS_NEED_MORE;
        }
        break;      /* flush the current block */
      }
    }

    /* Output a literal byte */
    s.match_length = 0;
    //Tracevv((stderr,"%c", s->window[s->strstart]));
    /*** _tr_tally_lit(s, s.window[s.strstart], bflush); ***/
    bflush = _tr_tally(s, 0, s.window[s.strstart]);
    s.lookahead--;
    s.strstart++;
    if (bflush) {
      /*** FLUSH_BLOCK(s, 0); ***/
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
      /***/
    }
  }
  s.insert = 0;
  if (flush === Z_FINISH$3) {
    /*** FLUSH_BLOCK(s, 1); ***/
    flush_block_only(s, true);
    if (s.strm.avail_out === 0) {
      return BS_FINISH_STARTED;
    }
    /***/
    return BS_FINISH_DONE;
  }
  if (s.sym_next) {
    /*** FLUSH_BLOCK(s, 0); ***/
    flush_block_only(s, false);
    if (s.strm.avail_out === 0) {
      return BS_NEED_MORE;
    }
    /***/
  }
  return BS_BLOCK_DONE;
};

/* Values for max_lazy_match, good_match and max_chain_length, depending on
 * the desired pack level (0..9). The values given below have been tuned to
 * exclude worst case performance for pathological files. Better values may be
 * found for specific files.
 */
function Config(good_length, max_lazy, nice_length, max_chain, func) {

  this.good_length = good_length;
  this.max_lazy = max_lazy;
  this.nice_length = nice_length;
  this.max_chain = max_chain;
  this.func = func;
}

const configuration_table = [
  /*      good lazy nice chain */
  new Config(0, 0, 0, 0, deflate_stored),          /* 0 store only */
  new Config(4, 4, 8, 4, deflate_fast),            /* 1 max speed, no lazy matches */
  new Config(4, 5, 16, 8, deflate_fast),           /* 2 */
  new Config(4, 6, 32, 32, deflate_fast),          /* 3 */

  new Config(4, 4, 16, 16, deflate_slow),          /* 4 lazy matches */
  new Config(8, 16, 32, 32, deflate_slow),         /* 5 */
  new Config(8, 16, 128, 128, deflate_slow),       /* 6 */
  new Config(8, 32, 128, 256, deflate_slow),       /* 7 */
  new Config(32, 128, 258, 1024, deflate_slow),    /* 8 */
  new Config(32, 258, 258, 4096, deflate_slow)     /* 9 max compression */
];


/* ===========================================================================
 * Initialize the "longest match" routines for a new zlib stream
 */
const lm_init = (s) => {

  s.window_size = 2 * s.w_size;

  /*** CLEAR_HASH(s); ***/
  zero(s.head); // Fill with NIL (= 0);

  /* Set the default configuration parameters:
   */
  s.max_lazy_match = configuration_table[s.level].max_lazy;
  s.good_match = configuration_table[s.level].good_length;
  s.nice_match = configuration_table[s.level].nice_length;
  s.max_chain_length = configuration_table[s.level].max_chain;

  s.strstart = 0;
  s.block_start = 0;
  s.lookahead = 0;
  s.insert = 0;
  s.match_length = s.prev_length = MIN_MATCH - 1;
  s.match_available = 0;
  s.ins_h = 0;
};


function DeflateState() {
  this.strm = null;            /* pointer back to this zlib stream */
  this.status = 0;            /* as the name implies */
  this.pending_buf = null;      /* output still pending */
  this.pending_buf_size = 0;  /* size of pending_buf */
  this.pending_out = 0;       /* next pending byte to output to the stream */
  this.pending = 0;           /* nb of bytes in the pending buffer */
  this.wrap = 0;              /* bit 0 true for zlib, bit 1 true for gzip */
  this.gzhead = null;         /* gzip header information to write */
  this.gzindex = 0;           /* where in extra, name, or comment */
  this.method = Z_DEFLATED$2; /* can only be DEFLATED */
  this.last_flush = -1;   /* value of flush param for previous deflate call */

  this.w_size = 0;  /* LZ77 window size (32K by default) */
  this.w_bits = 0;  /* log2(w_size)  (8..16) */
  this.w_mask = 0;  /* w_size - 1 */

  this.window = null;
  /* Sliding window. Input bytes are read into the second half of the window,
   * and move to the first half later to keep a dictionary of at least wSize
   * bytes. With this organization, matches are limited to a distance of
   * wSize-MAX_MATCH bytes, but this ensures that IO is always
   * performed with a length multiple of the block size.
   */

  this.window_size = 0;
  /* Actual size of window: 2*wSize, except when the user input buffer
   * is directly used as sliding window.
   */

  this.prev = null;
  /* Link to older string with same hash index. To limit the size of this
   * array to 64K, this link is maintained only for the last 32K strings.
   * An index in this array is thus a window index modulo 32K.
   */

  this.head = null;   /* Heads of the hash chains or NIL. */

  this.ins_h = 0;       /* hash index of string to be inserted */
  this.hash_size = 0;   /* number of elements in hash table */
  this.hash_bits = 0;   /* log2(hash_size) */
  this.hash_mask = 0;   /* hash_size-1 */

  this.hash_shift = 0;
  /* Number of bits by which ins_h must be shifted at each input
   * step. It must be such that after MIN_MATCH steps, the oldest
   * byte no longer takes part in the hash key, that is:
   *   hash_shift * MIN_MATCH >= hash_bits
   */

  this.block_start = 0;
  /* Window position at the beginning of the current output block. Gets
   * negative when the window is moved backwards.
   */

  this.match_length = 0;      /* length of best match */
  this.prev_match = 0;        /* previous match */
  this.match_available = 0;   /* set if previous match exists */
  this.strstart = 0;          /* start of string to insert */
  this.match_start = 0;       /* start of matching string */
  this.lookahead = 0;         /* number of valid bytes ahead in window */

  this.prev_length = 0;
  /* Length of the best match at previous step. Matches not greater than this
   * are discarded. This is used in the lazy match evaluation.
   */

  this.max_chain_length = 0;
  /* To speed up deflation, hash chains are never searched beyond this
   * length.  A higher limit improves compression ratio but degrades the
   * speed.
   */

  this.max_lazy_match = 0;
  /* Attempt to find a better match only when the current match is strictly
   * smaller than this value. This mechanism is used only for compression
   * levels >= 4.
   */
  // That's alias to max_lazy_match, don't use directly
  //this.max_insert_length = 0;
  /* Insert new strings in the hash table only if the match length is not
   * greater than this length. This saves time but degrades compression.
   * max_insert_length is used only for compression levels <= 3.
   */

  this.level = 0;     /* compression level (1..9) */
  this.strategy = 0;  /* favor or force Huffman coding*/

  this.good_match = 0;
  /* Use a faster search when the previous match is longer than this */

  this.nice_match = 0; /* Stop searching when current match exceeds this */

              /* used by trees.c: */

  /* Didn't use ct_data typedef below to suppress compiler warning */

  // struct ct_data_s dyn_ltree[HEAP_SIZE];   /* literal and length tree */
  // struct ct_data_s dyn_dtree[2*D_CODES+1]; /* distance tree */
  // struct ct_data_s bl_tree[2*BL_CODES+1];  /* Huffman tree for bit lengths */

  // Use flat array of DOUBLE size, with interleaved fata,
  // because JS does not support effective
  this.dyn_ltree  = new Uint16Array(HEAP_SIZE * 2);
  this.dyn_dtree  = new Uint16Array((2 * D_CODES + 1) * 2);
  this.bl_tree    = new Uint16Array((2 * BL_CODES + 1) * 2);
  zero(this.dyn_ltree);
  zero(this.dyn_dtree);
  zero(this.bl_tree);

  this.l_desc   = null;         /* desc. for literal tree */
  this.d_desc   = null;         /* desc. for distance tree */
  this.bl_desc  = null;         /* desc. for bit length tree */

  //ush bl_count[MAX_BITS+1];
  this.bl_count = new Uint16Array(MAX_BITS + 1);
  /* number of codes at each bit length for an optimal tree */

  //int heap[2*L_CODES+1];      /* heap used to build the Huffman trees */
  this.heap = new Uint16Array(2 * L_CODES + 1);  /* heap used to build the Huffman trees */
  zero(this.heap);

  this.heap_len = 0;               /* number of elements in the heap */
  this.heap_max = 0;               /* element of largest frequency */
  /* The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
   * The same heap array is used to build all trees.
   */

  this.depth = new Uint16Array(2 * L_CODES + 1); //uch depth[2*L_CODES+1];
  zero(this.depth);
  /* Depth of each subtree used as tie breaker for trees of equal frequency
   */

  this.sym_buf = 0;        /* buffer for distances and literals/lengths */

  this.lit_bufsize = 0;
  /* Size of match buffer for literals/lengths.  There are 4 reasons for
   * limiting lit_bufsize to 64K:
   *   - frequencies can be kept in 16 bit counters
   *   - if compression is not successful for the first block, all input
   *     data is still in the window so we can still emit a stored block even
   *     when input comes from standard input.  (This can also be done for
   *     all blocks if lit_bufsize is not greater than 32K.)
   *   - if compression is not successful for a file smaller than 64K, we can
   *     even emit a stored file instead of a stored block (saving 5 bytes).
   *     This is applicable only for zip (not gzip or zlib).
   *   - creating new Huffman trees less frequently may not provide fast
   *     adaptation to changes in the input data statistics. (Take for
   *     example a binary file with poorly compressible code followed by
   *     a highly compressible string table.) Smaller buffer sizes give
   *     fast adaptation but have of course the overhead of transmitting
   *     trees more frequently.
   *   - I can't count above 4
   */

  this.sym_next = 0;      /* running index in sym_buf */
  this.sym_end = 0;       /* symbol table full when sym_next reaches this */

  this.opt_len = 0;       /* bit length of current block with optimal trees */
  this.static_len = 0;    /* bit length of current block with static trees */
  this.matches = 0;       /* number of string matches in current block */
  this.insert = 0;        /* bytes at end of window left to insert */


  this.bi_buf = 0;
  /* Output buffer. bits are inserted starting at the bottom (least
   * significant bits).
   */
  this.bi_valid = 0;
  /* Number of valid bits in bi_buf.  All bits above the last valid bit
   * are always zero.
   */

  // Used for window memory init. We safely ignore it for JS. That makes
  // sense only for pointers and memory check tools.
  //this.high_water = 0;
  /* High water mark offset in window for initialized bytes -- bytes above
   * this are set to zero in order to avoid memory check warnings when
   * longest match routines access bytes past the input.  This is then
   * updated to the new high water mark.
   */
}


/* =========================================================================
 * Check for a valid deflate stream state. Return 0 if ok, 1 if not.
 */
const deflateStateCheck = (strm) => {

  if (!strm) {
    return 1;
  }
  const s = strm.state;
  if (!s || s.strm !== strm || (s.status !== INIT_STATE &&
//#ifdef GZIP
                                s.status !== GZIP_STATE &&
//#endif
                                s.status !== EXTRA_STATE &&
                                s.status !== NAME_STATE &&
                                s.status !== COMMENT_STATE &&
                                s.status !== HCRC_STATE &&
                                s.status !== BUSY_STATE &&
                                s.status !== FINISH_STATE)) {
    return 1;
  }
  return 0;
};


const deflateResetKeep = (strm) => {

  if (deflateStateCheck(strm)) {
    return err(strm, Z_STREAM_ERROR$2);
  }

  strm.total_in = strm.total_out = 0;
  strm.data_type = Z_UNKNOWN;

  const s = strm.state;
  s.pending = 0;
  s.pending_out = 0;

  if (s.wrap < 0) {
    s.wrap = -s.wrap;
    /* was made negative by deflate(..., Z_FINISH); */
  }
  s.status =
//#ifdef GZIP
    s.wrap === 2 ? GZIP_STATE :
//#endif
    s.wrap ? INIT_STATE : BUSY_STATE;
  strm.adler = (s.wrap === 2) ?
    0  // crc32(0, Z_NULL, 0)
  :
    1; // adler32(0, Z_NULL, 0)
  s.last_flush = -2;
  _tr_init(s);
  return Z_OK$3;
};


const deflateReset = (strm) => {

  const ret = deflateResetKeep(strm);
  if (ret === Z_OK$3) {
    lm_init(strm.state);
  }
  return ret;
};


const deflateSetHeader = (strm, head) => {

  if (deflateStateCheck(strm) || strm.state.wrap !== 2) {
    return Z_STREAM_ERROR$2;
  }
  strm.state.gzhead = head;
  return Z_OK$3;
};


const deflateInit2 = (strm, level, method, windowBits, memLevel, strategy) => {

  if (!strm) { // === Z_NULL
    return Z_STREAM_ERROR$2;
  }
  let wrap = 1;

  if (level === Z_DEFAULT_COMPRESSION$1) {
    level = 6;
  }

  if (windowBits < 0) { /* suppress zlib wrapper */
    wrap = 0;
    windowBits = -windowBits;
  }

  else if (windowBits > 15) {
    wrap = 2;           /* write gzip wrapper instead */
    windowBits -= 16;
  }


  if (memLevel < 1 || memLevel > MAX_MEM_LEVEL || method !== Z_DEFLATED$2 ||
    windowBits < 8 || windowBits > 15 || level < 0 || level > 9 ||
    strategy < 0 || strategy > Z_FIXED || (windowBits === 8 && wrap !== 1)) {
    return err(strm, Z_STREAM_ERROR$2);
  }


  if (windowBits === 8) {
    windowBits = 9;
  }
  /* until 256-byte window bug fixed */

  const s = new DeflateState();

  strm.state = s;
  s.strm = strm;
  s.status = INIT_STATE;     /* to pass state test in deflateReset() */

  s.wrap = wrap;
  s.gzhead = null;
  s.w_bits = windowBits;
  s.w_size = 1 << s.w_bits;
  s.w_mask = s.w_size - 1;

  s.hash_bits = memLevel + 7;
  s.hash_size = 1 << s.hash_bits;
  s.hash_mask = s.hash_size - 1;
  s.hash_shift = ~~((s.hash_bits + MIN_MATCH - 1) / MIN_MATCH);

  s.window = new Uint8Array(s.w_size * 2);
  s.head = new Uint16Array(s.hash_size);
  s.prev = new Uint16Array(s.w_size);

  // Don't need mem init magic for JS.
  //s.high_water = 0;  /* nothing written to s->window yet */

  s.lit_bufsize = 1 << (memLevel + 6); /* 16K elements by default */

  /* We overlay pending_buf and sym_buf. This works since the average size
   * for length/distance pairs over any compressed block is assured to be 31
   * bits or less.
   *
   * Analysis: The longest fixed codes are a length code of 8 bits plus 5
   * extra bits, for lengths 131 to 257. The longest fixed distance codes are
   * 5 bits plus 13 extra bits, for distances 16385 to 32768. The longest
   * possible fixed-codes length/distance pair is then 31 bits total.
   *
   * sym_buf starts one-fourth of the way into pending_buf. So there are
   * three bytes in sym_buf for every four bytes in pending_buf. Each symbol
   * in sym_buf is three bytes -- two for the distance and one for the
   * literal/length. As each symbol is consumed, the pointer to the next
   * sym_buf value to read moves forward three bytes. From that symbol, up to
   * 31 bits are written to pending_buf. The closest the written pending_buf
   * bits gets to the next sym_buf symbol to read is just before the last
   * code is written. At that time, 31*(n-2) bits have been written, just
   * after 24*(n-2) bits have been consumed from sym_buf. sym_buf starts at
   * 8*n bits into pending_buf. (Note that the symbol buffer fills when n-1
   * symbols are written.) The closest the writing gets to what is unread is
   * then n+14 bits. Here n is lit_bufsize, which is 16384 by default, and
   * can range from 128 to 32768.
   *
   * Therefore, at a minimum, there are 142 bits of space between what is
   * written and what is read in the overlain buffers, so the symbols cannot
   * be overwritten by the compressed data. That space is actually 139 bits,
   * due to the three-bit fixed-code block header.
   *
   * That covers the case where either Z_FIXED is specified, forcing fixed
   * codes, or when the use of fixed codes is chosen, because that choice
   * results in a smaller compressed block than dynamic codes. That latter
   * condition then assures that the above analysis also covers all dynamic
   * blocks. A dynamic-code block will only be chosen to be emitted if it has
   * fewer bits than a fixed-code block would for the same set of symbols.
   * Therefore its average symbol length is assured to be less than 31. So
   * the compressed data for a dynamic block also cannot overwrite the
   * symbols from which it is being constructed.
   */

  s.pending_buf_size = s.lit_bufsize * 4;
  s.pending_buf = new Uint8Array(s.pending_buf_size);

  // It is offset from `s.pending_buf` (size is `s.lit_bufsize * 2`)
  //s->sym_buf = s->pending_buf + s->lit_bufsize;
  s.sym_buf = s.lit_bufsize;

  //s->sym_end = (s->lit_bufsize - 1) * 3;
  s.sym_end = (s.lit_bufsize - 1) * 3;
  /* We avoid equality with lit_bufsize*3 because of wraparound at 64K
   * on 16 bit machines and because stored blocks are restricted to
   * 64K-1 bytes.
   */

  s.level = level;
  s.strategy = strategy;
  s.method = method;

  return deflateReset(strm);
};

const deflateInit = (strm, level) => {

  return deflateInit2(strm, level, Z_DEFLATED$2, MAX_WBITS$1, DEF_MEM_LEVEL, Z_DEFAULT_STRATEGY$1);
};


/* ========================================================================= */
const deflate$2 = (strm, flush) => {

  if (deflateStateCheck(strm) || flush > Z_BLOCK$1 || flush < 0) {
    return strm ? err(strm, Z_STREAM_ERROR$2) : Z_STREAM_ERROR$2;
  }

  const s = strm.state;

  if (!strm.output ||
      (strm.avail_in !== 0 && !strm.input) ||
      (s.status === FINISH_STATE && flush !== Z_FINISH$3)) {
    return err(strm, (strm.avail_out === 0) ? Z_BUF_ERROR$1 : Z_STREAM_ERROR$2);
  }

  const old_flush = s.last_flush;
  s.last_flush = flush;

  /* Flush as much pending output as possible */
  if (s.pending !== 0) {
    flush_pending(strm);
    if (strm.avail_out === 0) {
      /* Since avail_out is 0, deflate will be called again with
       * more output space, but possibly with both pending and
       * avail_in equal to zero. There won't be anything to do,
       * but this is not an error situation so make sure we
       * return OK instead of BUF_ERROR at next call of deflate:
       */
      s.last_flush = -1;
      return Z_OK$3;
    }

    /* Make sure there is something to do and avoid duplicate consecutive
     * flushes. For repeated and useless calls with Z_FINISH, we keep
     * returning Z_STREAM_END instead of Z_BUF_ERROR.
     */
  } else if (strm.avail_in === 0 && rank(flush) <= rank(old_flush) &&
    flush !== Z_FINISH$3) {
    return err(strm, Z_BUF_ERROR$1);
  }

  /* User must not provide more input after the first FINISH: */
  if (s.status === FINISH_STATE && strm.avail_in !== 0) {
    return err(strm, Z_BUF_ERROR$1);
  }

  /* Write the header */
  if (s.status === INIT_STATE && s.wrap === 0) {
    s.status = BUSY_STATE;
  }
  if (s.status === INIT_STATE) {
    /* zlib header */
    let header = (Z_DEFLATED$2 + ((s.w_bits - 8) << 4)) << 8;
    let level_flags = -1;

    if (s.strategy >= Z_HUFFMAN_ONLY || s.level < 2) {
      level_flags = 0;
    } else if (s.level < 6) {
      level_flags = 1;
    } else if (s.level === 6) {
      level_flags = 2;
    } else {
      level_flags = 3;
    }
    header |= (level_flags << 6);
    if (s.strstart !== 0) { header |= PRESET_DICT; }
    header += 31 - (header % 31);

    putShortMSB(s, header);

    /* Save the adler32 of the preset dictionary: */
    if (s.strstart !== 0) {
      putShortMSB(s, strm.adler >>> 16);
      putShortMSB(s, strm.adler & 0xffff);
    }
    strm.adler = 1; // adler32(0L, Z_NULL, 0);
    s.status = BUSY_STATE;

    /* Compression must start with an empty pending buffer */
    flush_pending(strm);
    if (s.pending !== 0) {
      s.last_flush = -1;
      return Z_OK$3;
    }
  }
//#ifdef GZIP
  if (s.status === GZIP_STATE) {
    /* gzip header */
    strm.adler = 0;  //crc32(0L, Z_NULL, 0);
    put_byte(s, 31);
    put_byte(s, 139);
    put_byte(s, 8);
    if (!s.gzhead) { // s->gzhead == Z_NULL
      put_byte(s, 0);
      put_byte(s, 0);
      put_byte(s, 0);
      put_byte(s, 0);
      put_byte(s, 0);
      put_byte(s, s.level === 9 ? 2 :
                  (s.strategy >= Z_HUFFMAN_ONLY || s.level < 2 ?
                   4 : 0));
      put_byte(s, OS_CODE);
      s.status = BUSY_STATE;

      /* Compression must start with an empty pending buffer */
      flush_pending(strm);
      if (s.pending !== 0) {
        s.last_flush = -1;
        return Z_OK$3;
      }
    }
    else {
      put_byte(s, (s.gzhead.text ? 1 : 0) +
                  (s.gzhead.hcrc ? 2 : 0) +
                  (!s.gzhead.extra ? 0 : 4) +
                  (!s.gzhead.name ? 0 : 8) +
                  (!s.gzhead.comment ? 0 : 16)
      );
      put_byte(s, s.gzhead.time & 0xff);
      put_byte(s, (s.gzhead.time >> 8) & 0xff);
      put_byte(s, (s.gzhead.time >> 16) & 0xff);
      put_byte(s, (s.gzhead.time >> 24) & 0xff);
      put_byte(s, s.level === 9 ? 2 :
                  (s.strategy >= Z_HUFFMAN_ONLY || s.level < 2 ?
                   4 : 0));
      put_byte(s, s.gzhead.os & 0xff);
      if (s.gzhead.extra && s.gzhead.extra.length) {
        put_byte(s, s.gzhead.extra.length & 0xff);
        put_byte(s, (s.gzhead.extra.length >> 8) & 0xff);
      }
      if (s.gzhead.hcrc) {
        strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending, 0);
      }
      s.gzindex = 0;
      s.status = EXTRA_STATE;
    }
  }
  if (s.status === EXTRA_STATE) {
    if (s.gzhead.extra/* != Z_NULL*/) {
      let beg = s.pending;   /* start of bytes to update crc */
      let left = (s.gzhead.extra.length & 0xffff) - s.gzindex;
      while (s.pending + left > s.pending_buf_size) {
        let copy = s.pending_buf_size - s.pending;
        // zmemcpy(s.pending_buf + s.pending,
        //    s.gzhead.extra + s.gzindex, copy);
        s.pending_buf.set(s.gzhead.extra.subarray(s.gzindex, s.gzindex + copy), s.pending);
        s.pending = s.pending_buf_size;
        //--- HCRC_UPDATE(beg) ---//
        if (s.gzhead.hcrc && s.pending > beg) {
          strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
        }
        //---//
        s.gzindex += copy;
        flush_pending(strm);
        if (s.pending !== 0) {
          s.last_flush = -1;
          return Z_OK$3;
        }
        beg = 0;
        left -= copy;
      }
      // JS specific: s.gzhead.extra may be TypedArray or Array for backward compatibility
      //              TypedArray.slice and TypedArray.from don't exist in IE10-IE11
      let gzhead_extra = new Uint8Array(s.gzhead.extra);
      // zmemcpy(s->pending_buf + s->pending,
      //     s->gzhead->extra + s->gzindex, left);
      s.pending_buf.set(gzhead_extra.subarray(s.gzindex, s.gzindex + left), s.pending);
      s.pending += left;
      //--- HCRC_UPDATE(beg) ---//
      if (s.gzhead.hcrc && s.pending > beg) {
        strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
      }
      //---//
      s.gzindex = 0;
    }
    s.status = NAME_STATE;
  }
  if (s.status === NAME_STATE) {
    if (s.gzhead.name/* != Z_NULL*/) {
      let beg = s.pending;   /* start of bytes to update crc */
      let val;
      do {
        if (s.pending === s.pending_buf_size) {
          //--- HCRC_UPDATE(beg) ---//
          if (s.gzhead.hcrc && s.pending > beg) {
            strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
          }
          //---//
          flush_pending(strm);
          if (s.pending !== 0) {
            s.last_flush = -1;
            return Z_OK$3;
          }
          beg = 0;
        }
        // JS specific: little magic to add zero terminator to end of string
        if (s.gzindex < s.gzhead.name.length) {
          val = s.gzhead.name.charCodeAt(s.gzindex++) & 0xff;
        } else {
          val = 0;
        }
        put_byte(s, val);
      } while (val !== 0);
      //--- HCRC_UPDATE(beg) ---//
      if (s.gzhead.hcrc && s.pending > beg) {
        strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
      }
      //---//
      s.gzindex = 0;
    }
    s.status = COMMENT_STATE;
  }
  if (s.status === COMMENT_STATE) {
    if (s.gzhead.comment/* != Z_NULL*/) {
      let beg = s.pending;   /* start of bytes to update crc */
      let val;
      do {
        if (s.pending === s.pending_buf_size) {
          //--- HCRC_UPDATE(beg) ---//
          if (s.gzhead.hcrc && s.pending > beg) {
            strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
          }
          //---//
          flush_pending(strm);
          if (s.pending !== 0) {
            s.last_flush = -1;
            return Z_OK$3;
          }
          beg = 0;
        }
        // JS specific: little magic to add zero terminator to end of string
        if (s.gzindex < s.gzhead.comment.length) {
          val = s.gzhead.comment.charCodeAt(s.gzindex++) & 0xff;
        } else {
          val = 0;
        }
        put_byte(s, val);
      } while (val !== 0);
      //--- HCRC_UPDATE(beg) ---//
      if (s.gzhead.hcrc && s.pending > beg) {
        strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
      }
      //---//
    }
    s.status = HCRC_STATE;
  }
  if (s.status === HCRC_STATE) {
    if (s.gzhead.hcrc) {
      if (s.pending + 2 > s.pending_buf_size) {
        flush_pending(strm);
        if (s.pending !== 0) {
          s.last_flush = -1;
          return Z_OK$3;
        }
      }
      put_byte(s, strm.adler & 0xff);
      put_byte(s, (strm.adler >> 8) & 0xff);
      strm.adler = 0; //crc32(0L, Z_NULL, 0);
    }
    s.status = BUSY_STATE;

    /* Compression must start with an empty pending buffer */
    flush_pending(strm);
    if (s.pending !== 0) {
      s.last_flush = -1;
      return Z_OK$3;
    }
  }
//#endif

  /* Start a new block or continue the current one.
   */
  if (strm.avail_in !== 0 || s.lookahead !== 0 ||
    (flush !== Z_NO_FLUSH$2 && s.status !== FINISH_STATE)) {
    let bstate = s.level === 0 ? deflate_stored(s, flush) :
                 s.strategy === Z_HUFFMAN_ONLY ? deflate_huff(s, flush) :
                 s.strategy === Z_RLE ? deflate_rle(s, flush) :
                 configuration_table[s.level].func(s, flush);

    if (bstate === BS_FINISH_STARTED || bstate === BS_FINISH_DONE) {
      s.status = FINISH_STATE;
    }
    if (bstate === BS_NEED_MORE || bstate === BS_FINISH_STARTED) {
      if (strm.avail_out === 0) {
        s.last_flush = -1;
        /* avoid BUF_ERROR next call, see above */
      }
      return Z_OK$3;
      /* If flush != Z_NO_FLUSH && avail_out == 0, the next call
       * of deflate should use the same flush parameter to make sure
       * that the flush is complete. So we don't have to output an
       * empty block here, this will be done at next call. This also
       * ensures that for a very small output buffer, we emit at most
       * one empty block.
       */
    }
    if (bstate === BS_BLOCK_DONE) {
      if (flush === Z_PARTIAL_FLUSH) {
        _tr_align(s);
      }
      else if (flush !== Z_BLOCK$1) { /* FULL_FLUSH or SYNC_FLUSH */

        _tr_stored_block(s, 0, 0, false);
        /* For a full flush, this empty block will be recognized
         * as a special marker by inflate_sync().
         */
        if (flush === Z_FULL_FLUSH$1) {
          /*** CLEAR_HASH(s); ***/             /* forget history */
          zero(s.head); // Fill with NIL (= 0);

          if (s.lookahead === 0) {
            s.strstart = 0;
            s.block_start = 0;
            s.insert = 0;
          }
        }
      }
      flush_pending(strm);
      if (strm.avail_out === 0) {
        s.last_flush = -1; /* avoid BUF_ERROR at next call, see above */
        return Z_OK$3;
      }
    }
  }

  if (flush !== Z_FINISH$3) { return Z_OK$3; }
  if (s.wrap <= 0) { return Z_STREAM_END$3; }

  /* Write the trailer */
  if (s.wrap === 2) {
    put_byte(s, strm.adler & 0xff);
    put_byte(s, (strm.adler >> 8) & 0xff);
    put_byte(s, (strm.adler >> 16) & 0xff);
    put_byte(s, (strm.adler >> 24) & 0xff);
    put_byte(s, strm.total_in & 0xff);
    put_byte(s, (strm.total_in >> 8) & 0xff);
    put_byte(s, (strm.total_in >> 16) & 0xff);
    put_byte(s, (strm.total_in >> 24) & 0xff);
  }
  else
  {
    putShortMSB(s, strm.adler >>> 16);
    putShortMSB(s, strm.adler & 0xffff);
  }

  flush_pending(strm);
  /* If avail_out is zero, the application will call deflate again
   * to flush the rest.
   */
  if (s.wrap > 0) { s.wrap = -s.wrap; }
  /* write the trailer only once! */
  return s.pending !== 0 ? Z_OK$3 : Z_STREAM_END$3;
};


const deflateEnd = (strm) => {

  if (deflateStateCheck(strm)) {
    return Z_STREAM_ERROR$2;
  }

  const status = strm.state.status;

  strm.state = null;

  return status === BUSY_STATE ? err(strm, Z_DATA_ERROR$2) : Z_OK$3;
};


/* =========================================================================
 * Initializes the compression dictionary from the given byte
 * sequence without producing any compressed output.
 */
const deflateSetDictionary = (strm, dictionary) => {

  let dictLength = dictionary.length;

  if (deflateStateCheck(strm)) {
    return Z_STREAM_ERROR$2;
  }

  const s = strm.state;
  const wrap = s.wrap;

  if (wrap === 2 || (wrap === 1 && s.status !== INIT_STATE) || s.lookahead) {
    return Z_STREAM_ERROR$2;
  }

  /* when using zlib wrappers, compute Adler-32 for provided dictionary */
  if (wrap === 1) {
    /* adler32(strm->adler, dictionary, dictLength); */
    strm.adler = adler32_1(strm.adler, dictionary, dictLength, 0);
  }

  s.wrap = 0;   /* avoid computing Adler-32 in read_buf */

  /* if dictionary would fill window, just replace the history */
  if (dictLength >= s.w_size) {
    if (wrap === 0) {            /* already empty otherwise */
      /*** CLEAR_HASH(s); ***/
      zero(s.head); // Fill with NIL (= 0);
      s.strstart = 0;
      s.block_start = 0;
      s.insert = 0;
    }
    /* use the tail */
    // dictionary = dictionary.slice(dictLength - s.w_size);
    let tmpDict = new Uint8Array(s.w_size);
    tmpDict.set(dictionary.subarray(dictLength - s.w_size, dictLength), 0);
    dictionary = tmpDict;
    dictLength = s.w_size;
  }
  /* insert dictionary into window and hash */
  const avail = strm.avail_in;
  const next = strm.next_in;
  const input = strm.input;
  strm.avail_in = dictLength;
  strm.next_in = 0;
  strm.input = dictionary;
  fill_window(s);
  while (s.lookahead >= MIN_MATCH) {
    let str = s.strstart;
    let n = s.lookahead - (MIN_MATCH - 1);
    do {
      /* UPDATE_HASH(s, s->ins_h, s->window[str + MIN_MATCH-1]); */
      s.ins_h = HASH(s, s.ins_h, s.window[str + MIN_MATCH - 1]);

      s.prev[str & s.w_mask] = s.head[s.ins_h];

      s.head[s.ins_h] = str;
      str++;
    } while (--n);
    s.strstart = str;
    s.lookahead = MIN_MATCH - 1;
    fill_window(s);
  }
  s.strstart += s.lookahead;
  s.block_start = s.strstart;
  s.insert = s.lookahead;
  s.lookahead = 0;
  s.match_length = s.prev_length = MIN_MATCH - 1;
  s.match_available = 0;
  strm.next_in = next;
  strm.input = input;
  strm.avail_in = avail;
  s.wrap = wrap;
  return Z_OK$3;
};


var deflateInit_1 = deflateInit;
var deflateInit2_1 = deflateInit2;
var deflateReset_1 = deflateReset;
var deflateResetKeep_1 = deflateResetKeep;
var deflateSetHeader_1 = deflateSetHeader;
var deflate_2$1 = deflate$2;
var deflateEnd_1 = deflateEnd;
var deflateSetDictionary_1 = deflateSetDictionary;
var deflateInfo = 'pako deflate (from Nodeca project)';

/* Not implemented
module.exports.deflateBound = deflateBound;
module.exports.deflateCopy = deflateCopy;
module.exports.deflateGetDictionary = deflateGetDictionary;
module.exports.deflateParams = deflateParams;
module.exports.deflatePending = deflatePending;
module.exports.deflatePrime = deflatePrime;
module.exports.deflateTune = deflateTune;
*/

var deflate_1$2 = {
	deflateInit: deflateInit_1,
	deflateInit2: deflateInit2_1,
	deflateReset: deflateReset_1,
	deflateResetKeep: deflateResetKeep_1,
	deflateSetHeader: deflateSetHeader_1,
	deflate: deflate_2$1,
	deflateEnd: deflateEnd_1,
	deflateSetDictionary: deflateSetDictionary_1,
	deflateInfo: deflateInfo
};

const _has = (obj, key) => {
  return Object.prototype.hasOwnProperty.call(obj, key);
};

var assign = function (obj /*from1, from2, from3, ...*/) {
  const sources = Array.prototype.slice.call(arguments, 1);
  while (sources.length) {
    const source = sources.shift();
    if (!source) { continue; }

    if (typeof source !== 'object') {
      throw new TypeError(source + 'must be non-object');
    }

    for (const p in source) {
      if (_has(source, p)) {
        obj[p] = source[p];
      }
    }
  }

  return obj;
};


// Join array of chunks to single array.
var flattenChunks = (chunks) => {
  // calculate data length
  let len = 0;

  for (let i = 0, l = chunks.length; i < l; i++) {
    len += chunks[i].length;
  }

  // join chunks
  const result = new Uint8Array(len);

  for (let i = 0, pos = 0, l = chunks.length; i < l; i++) {
    let chunk = chunks[i];
    result.set(chunk, pos);
    pos += chunk.length;
  }

  return result;
};

var common = {
	assign: assign,
	flattenChunks: flattenChunks
};

// String encode/decode helpers


// Quick check if we can use fast array to bin string conversion
//
// - apply(Array) can fail on Android 2.2
// - apply(Uint8Array) can fail on iOS 5.1 Safari
//
let STR_APPLY_UIA_OK = true;

try { String.fromCharCode.apply(null, new Uint8Array(1)); } catch (__) { STR_APPLY_UIA_OK = false; }


// Table with utf8 lengths (calculated by first byte of sequence)
// Note, that 5 & 6-byte values and some 4-byte values can not be represented in JS,
// because max possible codepoint is 0x10ffff
const _utf8len = new Uint8Array(256);
for (let q = 0; q < 256; q++) {
  _utf8len[q] = (q >= 252 ? 6 : q >= 248 ? 5 : q >= 240 ? 4 : q >= 224 ? 3 : q >= 192 ? 2 : 1);
}
_utf8len[254] = _utf8len[254] = 1; // Invalid sequence start


// convert string to array (typed, when possible)
var string2buf = (str) => {
  if (typeof TextEncoder === 'function' && TextEncoder.prototype.encode) {
    return new TextEncoder().encode(str);
  }

  let buf, c, c2, m_pos, i, str_len = str.length, buf_len = 0;

  // count binary size
  for (m_pos = 0; m_pos < str_len; m_pos++) {
    c = str.charCodeAt(m_pos);
    if ((c & 0xfc00) === 0xd800 && (m_pos + 1 < str_len)) {
      c2 = str.charCodeAt(m_pos + 1);
      if ((c2 & 0xfc00) === 0xdc00) {
        c = 0x10000 + ((c - 0xd800) << 10) + (c2 - 0xdc00);
        m_pos++;
      }
    }
    buf_len += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  }

  // allocate buffer
  buf = new Uint8Array(buf_len);

  // convert
  for (i = 0, m_pos = 0; i < buf_len; m_pos++) {
    c = str.charCodeAt(m_pos);
    if ((c & 0xfc00) === 0xd800 && (m_pos + 1 < str_len)) {
      c2 = str.charCodeAt(m_pos + 1);
      if ((c2 & 0xfc00) === 0xdc00) {
        c = 0x10000 + ((c - 0xd800) << 10) + (c2 - 0xdc00);
        m_pos++;
      }
    }
    if (c < 0x80) {
      /* one byte */
      buf[i++] = c;
    } else if (c < 0x800) {
      /* two bytes */
      buf[i++] = 0xC0 | (c >>> 6);
      buf[i++] = 0x80 | (c & 0x3f);
    } else if (c < 0x10000) {
      /* three bytes */
      buf[i++] = 0xE0 | (c >>> 12);
      buf[i++] = 0x80 | (c >>> 6 & 0x3f);
      buf[i++] = 0x80 | (c & 0x3f);
    } else {
      /* four bytes */
      buf[i++] = 0xf0 | (c >>> 18);
      buf[i++] = 0x80 | (c >>> 12 & 0x3f);
      buf[i++] = 0x80 | (c >>> 6 & 0x3f);
      buf[i++] = 0x80 | (c & 0x3f);
    }
  }

  return buf;
};

// Helper
const buf2binstring = (buf, len) => {
  // On Chrome, the arguments in a function call that are allowed is `65534`.
  // If the length of the buffer is smaller than that, we can use this optimization,
  // otherwise we will take a slower path.
  if (len < 65534) {
    if (buf.subarray && STR_APPLY_UIA_OK) {
      return String.fromCharCode.apply(null, buf.length === len ? buf : buf.subarray(0, len));
    }
  }

  let result = '';
  for (let i = 0; i < len; i++) {
    result += String.fromCharCode(buf[i]);
  }
  return result;
};


// convert array to string
var buf2string = (buf, max) => {
  const len = max || buf.length;

  if (typeof TextDecoder === 'function' && TextDecoder.prototype.decode) {
    return new TextDecoder().decode(buf.subarray(0, max));
  }

  let i, out;

  // Reserve max possible length (2 words per char)
  // NB: by unknown reasons, Array is significantly faster for
  //     String.fromCharCode.apply than Uint16Array.
  const utf16buf = new Array(len * 2);

  for (out = 0, i = 0; i < len;) {
    let c = buf[i++];
    // quick process ascii
    if (c < 0x80) { utf16buf[out++] = c; continue; }

    let c_len = _utf8len[c];
    // skip 5 & 6 byte codes
    if (c_len > 4) { utf16buf[out++] = 0xfffd; i += c_len - 1; continue; }

    // apply mask on first byte
    c &= c_len === 2 ? 0x1f : c_len === 3 ? 0x0f : 0x07;
    // join the rest
    while (c_len > 1 && i < len) {
      c = (c << 6) | (buf[i++] & 0x3f);
      c_len--;
    }

    // terminated by end of string?
    if (c_len > 1) { utf16buf[out++] = 0xfffd; continue; }

    if (c < 0x10000) {
      utf16buf[out++] = c;
    } else {
      c -= 0x10000;
      utf16buf[out++] = 0xd800 | ((c >> 10) & 0x3ff);
      utf16buf[out++] = 0xdc00 | (c & 0x3ff);
    }
  }

  return buf2binstring(utf16buf, out);
};


// Calculate max possible position in utf8 buffer,
// that will not break sequence. If that's not possible
// - (very small limits) return max size as is.
//
// buf[] - utf8 bytes array
// max   - length limit (mandatory);
var utf8border = (buf, max) => {

  max = max || buf.length;
  if (max > buf.length) { max = buf.length; }

  // go back from last position, until start of sequence found
  let pos = max - 1;
  while (pos >= 0 && (buf[pos] & 0xC0) === 0x80) { pos--; }

  // Very small and broken sequence,
  // return max, because we should return something anyway.
  if (pos < 0) { return max; }

  // If we came to start of buffer - that means buffer is too small,
  // return max too.
  if (pos === 0) { return max; }

  return (pos + _utf8len[buf[pos]] > max) ? pos : max;
};

var strings = {
	string2buf: string2buf,
	buf2string: buf2string,
	utf8border: utf8border
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

function ZStream() {
  /* next input byte */
  this.input = null; // JS specific, because we have no pointers
  this.next_in = 0;
  /* number of bytes available at input */
  this.avail_in = 0;
  /* total number of input bytes read so far */
  this.total_in = 0;
  /* next output byte should be put there */
  this.output = null; // JS specific, because we have no pointers
  this.next_out = 0;
  /* remaining free space at output */
  this.avail_out = 0;
  /* total number of bytes output so far */
  this.total_out = 0;
  /* last error message, NULL if no error */
  this.msg = ''/*Z_NULL*/;
  /* not visible by applications */
  this.state = null;
  /* best guess about the data type: binary or text */
  this.data_type = 2/*Z_UNKNOWN*/;
  /* adler32 value of the uncompressed data */
  this.adler = 0;
}

var zstream = ZStream;

const toString$1 = Object.prototype.toString;

/* Public constants ==========================================================*/
/* ===========================================================================*/

const {
  Z_NO_FLUSH: Z_NO_FLUSH$1, Z_SYNC_FLUSH, Z_FULL_FLUSH, Z_FINISH: Z_FINISH$2,
  Z_OK: Z_OK$2, Z_STREAM_END: Z_STREAM_END$2,
  Z_DEFAULT_COMPRESSION,
  Z_DEFAULT_STRATEGY,
  Z_DEFLATED: Z_DEFLATED$1
} = constants$2;

/* ===========================================================================*/


/**
 * class Deflate
 *
 * Generic JS-style wrapper for zlib calls. If you don't need
 * streaming behaviour - use more simple functions: [[deflate]],
 * [[deflateRaw]] and [[gzip]].
 **/

/* internal
 * Deflate.chunks -> Array
 *
 * Chunks of output data, if [[Deflate#onData]] not overridden.
 **/

/**
 * Deflate.result -> Uint8Array
 *
 * Compressed result, generated by default [[Deflate#onData]]
 * and [[Deflate#onEnd]] handlers. Filled after you push last chunk
 * (call [[Deflate#push]] with `Z_FINISH` / `true` param).
 **/

/**
 * Deflate.err -> Number
 *
 * Error code after deflate finished. 0 (Z_OK) on success.
 * You will not need it in real life, because deflate errors
 * are possible only on wrong options or bad `onData` / `onEnd`
 * custom handlers.
 **/

/**
 * Deflate.msg -> String
 *
 * Error message, if [[Deflate.err]] != 0
 **/


/**
 * new Deflate(options)
 * - options (Object): zlib deflate options.
 *
 * Creates new deflator instance with specified params. Throws exception
 * on bad params. Supported options:
 *
 * - `level`
 * - `windowBits`
 * - `memLevel`
 * - `strategy`
 * - `dictionary`
 *
 * [http://zlib.net/manual.html#Advanced](http://zlib.net/manual.html#Advanced)
 * for more information on these.
 *
 * Additional options, for internal needs:
 *
 * - `chunkSize` - size of generated data chunks (16K by default)
 * - `raw` (Boolean) - do raw deflate
 * - `gzip` (Boolean) - create gzip wrapper
 * - `header` (Object) - custom header for gzip
 *   - `text` (Boolean) - true if compressed data believed to be text
 *   - `time` (Number) - modification time, unix timestamp
 *   - `os` (Number) - operation system code
 *   - `extra` (Array) - array of bytes with extra data (max 65536)
 *   - `name` (String) - file name (binary string)
 *   - `comment` (String) - comment (binary string)
 *   - `hcrc` (Boolean) - true if header crc should be added
 *
 * ##### Example:
 *
 * ```javascript
 * const pako = require('pako')
 *   , chunk1 = new Uint8Array([1,2,3,4,5,6,7,8,9])
 *   , chunk2 = new Uint8Array([10,11,12,13,14,15,16,17,18,19]);
 *
 * const deflate = new pako.Deflate({ level: 3});
 *
 * deflate.push(chunk1, false);
 * deflate.push(chunk2, true);  // true -> last chunk
 *
 * if (deflate.err) { throw new Error(deflate.err); }
 *
 * console.log(deflate.result);
 * ```
 **/
function Deflate$1(options) {
  this.options = common.assign({
    level: Z_DEFAULT_COMPRESSION,
    method: Z_DEFLATED$1,
    chunkSize: 16384,
    windowBits: 15,
    memLevel: 8,
    strategy: Z_DEFAULT_STRATEGY
  }, options || {});

  let opt = this.options;

  if (opt.raw && (opt.windowBits > 0)) {
    opt.windowBits = -opt.windowBits;
  }

  else if (opt.gzip && (opt.windowBits > 0) && (opt.windowBits < 16)) {
    opt.windowBits += 16;
  }

  this.err    = 0;      // error code, if happens (0 = Z_OK)
  this.msg    = '';     // error message
  this.ended  = false;  // used to avoid multiple onEnd() calls
  this.chunks = [];     // chunks of compressed data

  this.strm = new zstream();
  this.strm.avail_out = 0;

  let status = deflate_1$2.deflateInit2(
    this.strm,
    opt.level,
    opt.method,
    opt.windowBits,
    opt.memLevel,
    opt.strategy
  );

  if (status !== Z_OK$2) {
    throw new Error(messages[status]);
  }

  if (opt.header) {
    deflate_1$2.deflateSetHeader(this.strm, opt.header);
  }

  if (opt.dictionary) {
    let dict;
    // Convert data if needed
    if (typeof opt.dictionary === 'string') {
      // If we need to compress text, change encoding to utf8.
      dict = strings.string2buf(opt.dictionary);
    } else if (toString$1.call(opt.dictionary) === '[object ArrayBuffer]') {
      dict = new Uint8Array(opt.dictionary);
    } else {
      dict = opt.dictionary;
    }

    status = deflate_1$2.deflateSetDictionary(this.strm, dict);

    if (status !== Z_OK$2) {
      throw new Error(messages[status]);
    }

    this._dict_set = true;
  }
}

/**
 * Deflate#push(data[, flush_mode]) -> Boolean
 * - data (Uint8Array|ArrayBuffer|String): input data. Strings will be
 *   converted to utf8 byte sequence.
 * - flush_mode (Number|Boolean): 0..6 for corresponding Z_NO_FLUSH..Z_TREE modes.
 *   See constants. Skipped or `false` means Z_NO_FLUSH, `true` means Z_FINISH.
 *
 * Sends input data to deflate pipe, generating [[Deflate#onData]] calls with
 * new compressed chunks. Returns `true` on success. The last data block must
 * have `flush_mode` Z_FINISH (or `true`). That will flush internal pending
 * buffers and call [[Deflate#onEnd]].
 *
 * On fail call [[Deflate#onEnd]] with error code and return false.
 *
 * ##### Example
 *
 * ```javascript
 * push(chunk, false); // push one of data chunks
 * ...
 * push(chunk, true);  // push last chunk
 * ```
 **/
Deflate$1.prototype.push = function (data, flush_mode) {
  const strm = this.strm;
  const chunkSize = this.options.chunkSize;
  let status, _flush_mode;

  if (this.ended) { return false; }

  if (flush_mode === ~~flush_mode) _flush_mode = flush_mode;
  else _flush_mode = flush_mode === true ? Z_FINISH$2 : Z_NO_FLUSH$1;

  // Convert data if needed
  if (typeof data === 'string') {
    // If we need to compress text, change encoding to utf8.
    strm.input = strings.string2buf(data);
  } else if (toString$1.call(data) === '[object ArrayBuffer]') {
    strm.input = new Uint8Array(data);
  } else {
    strm.input = data;
  }

  strm.next_in = 0;
  strm.avail_in = strm.input.length;

  for (;;) {
    if (strm.avail_out === 0) {
      strm.output = new Uint8Array(chunkSize);
      strm.next_out = 0;
      strm.avail_out = chunkSize;
    }

    // Make sure avail_out > 6 to avoid repeating markers
    if ((_flush_mode === Z_SYNC_FLUSH || _flush_mode === Z_FULL_FLUSH) && strm.avail_out <= 6) {
      this.onData(strm.output.subarray(0, strm.next_out));
      strm.avail_out = 0;
      continue;
    }

    status = deflate_1$2.deflate(strm, _flush_mode);

    // Ended => flush and finish
    if (status === Z_STREAM_END$2) {
      if (strm.next_out > 0) {
        this.onData(strm.output.subarray(0, strm.next_out));
      }
      status = deflate_1$2.deflateEnd(this.strm);
      this.onEnd(status);
      this.ended = true;
      return status === Z_OK$2;
    }

    // Flush if out buffer full
    if (strm.avail_out === 0) {
      this.onData(strm.output);
      continue;
    }

    // Flush if requested and has data
    if (_flush_mode > 0 && strm.next_out > 0) {
      this.onData(strm.output.subarray(0, strm.next_out));
      strm.avail_out = 0;
      continue;
    }

    if (strm.avail_in === 0) break;
  }

  return true;
};


/**
 * Deflate#onData(chunk) -> Void
 * - chunk (Uint8Array): output data.
 *
 * By default, stores data blocks in `chunks[]` property and glue
 * those in `onEnd`. Override this handler, if you need another behaviour.
 **/
Deflate$1.prototype.onData = function (chunk) {
  this.chunks.push(chunk);
};


/**
 * Deflate#onEnd(status) -> Void
 * - status (Number): deflate status. 0 (Z_OK) on success,
 *   other if not.
 *
 * Called once after you tell deflate that the input stream is
 * complete (Z_FINISH). By default - join collected chunks,
 * free memory and fill `results` / `err` properties.
 **/
Deflate$1.prototype.onEnd = function (status) {
  // On success - join
  if (status === Z_OK$2) {
    this.result = common.flattenChunks(this.chunks);
  }
  this.chunks = [];
  this.err = status;
  this.msg = this.strm.msg;
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

// See state defs from inflate.js
const BAD$1 = 16209;       /* got a data error -- remain here until reset */
const TYPE$1 = 16191;      /* i: waiting for type bits, including last-flag bit */

/*
   Decode literal, length, and distance codes and write out the resulting
   literal and match bytes until either not enough input or output is
   available, an end-of-block is encountered, or a data error is encountered.
   When large enough input and output buffers are supplied to inflate(), for
   example, a 16K input buffer and a 64K output buffer, more than 95% of the
   inflate execution time is spent in this routine.

   Entry assumptions:

        state.mode === LEN
        strm.avail_in >= 6
        strm.avail_out >= 258
        start >= strm.avail_out
        state.bits < 8

   On return, state.mode is one of:

        LEN -- ran out of enough output space or enough available input
        TYPE -- reached end of block code, inflate() to interpret next block
        BAD -- error in block data

   Notes:

    - The maximum input bits used by a length/distance pair is 15 bits for the
      length code, 5 bits for the length extra, 15 bits for the distance code,
      and 13 bits for the distance extra.  This totals 48 bits, or six bytes.
      Therefore if strm.avail_in >= 6, then there is enough input to avoid
      checking for available input while decoding.

    - The maximum bytes that a single length/distance pair can output is 258
      bytes, which is the maximum length that can be coded.  inflate_fast()
      requires strm.avail_out >= 258 for each loop to avoid checking for
      output space.
 */
var inffast = function inflate_fast(strm, start) {
  let _in;                    /* local strm.input */
  let last;                   /* have enough input while in < last */
  let _out;                   /* local strm.output */
  let beg;                    /* inflate()'s initial strm.output */
  let end;                    /* while out < end, enough space available */
//#ifdef INFLATE_STRICT
  let dmax;                   /* maximum distance from zlib header */
//#endif
  let wsize;                  /* window size or zero if not using window */
  let whave;                  /* valid bytes in the window */
  let wnext;                  /* window write index */
  // Use `s_window` instead `window`, avoid conflict with instrumentation tools
  let s_window;               /* allocated sliding window, if wsize != 0 */
  let hold;                   /* local strm.hold */
  let bits;                   /* local strm.bits */
  let lcode;                  /* local strm.lencode */
  let dcode;                  /* local strm.distcode */
  let lmask;                  /* mask for first level of length codes */
  let dmask;                  /* mask for first level of distance codes */
  let here;                   /* retrieved table entry */
  let op;                     /* code bits, operation, extra bits, or */
                              /*  window position, window bytes to copy */
  let len;                    /* match length, unused bytes */
  let dist;                   /* match distance */
  let from;                   /* where to copy match from */
  let from_source;


  let input, output; // JS specific, because we have no pointers

  /* copy state to local variables */
  const state = strm.state;
  //here = state.here;
  _in = strm.next_in;
  input = strm.input;
  last = _in + (strm.avail_in - 5);
  _out = strm.next_out;
  output = strm.output;
  beg = _out - (start - strm.avail_out);
  end = _out + (strm.avail_out - 257);
//#ifdef INFLATE_STRICT
  dmax = state.dmax;
//#endif
  wsize = state.wsize;
  whave = state.whave;
  wnext = state.wnext;
  s_window = state.window;
  hold = state.hold;
  bits = state.bits;
  lcode = state.lencode;
  dcode = state.distcode;
  lmask = (1 << state.lenbits) - 1;
  dmask = (1 << state.distbits) - 1;


  /* decode literals and length/distances until end-of-block or not enough
     input data or output space */

  top:
  do {
    if (bits < 15) {
      hold += input[_in++] << bits;
      bits += 8;
      hold += input[_in++] << bits;
      bits += 8;
    }

    here = lcode[hold & lmask];

    dolen:
    for (;;) { // Goto emulation
      op = here >>> 24/*here.bits*/;
      hold >>>= op;
      bits -= op;
      op = (here >>> 16) & 0xff/*here.op*/;
      if (op === 0) {                          /* literal */
        //Tracevv((stderr, here.val >= 0x20 && here.val < 0x7f ?
        //        "inflate:         literal '%c'\n" :
        //        "inflate:         literal 0x%02x\n", here.val));
        output[_out++] = here & 0xffff/*here.val*/;
      }
      else if (op & 16) {                     /* length base */
        len = here & 0xffff/*here.val*/;
        op &= 15;                           /* number of extra bits */
        if (op) {
          if (bits < op) {
            hold += input[_in++] << bits;
            bits += 8;
          }
          len += hold & ((1 << op) - 1);
          hold >>>= op;
          bits -= op;
        }
        //Tracevv((stderr, "inflate:         length %u\n", len));
        if (bits < 15) {
          hold += input[_in++] << bits;
          bits += 8;
          hold += input[_in++] << bits;
          bits += 8;
        }
        here = dcode[hold & dmask];

        dodist:
        for (;;) { // goto emulation
          op = here >>> 24/*here.bits*/;
          hold >>>= op;
          bits -= op;
          op = (here >>> 16) & 0xff/*here.op*/;

          if (op & 16) {                      /* distance base */
            dist = here & 0xffff/*here.val*/;
            op &= 15;                       /* number of extra bits */
            if (bits < op) {
              hold += input[_in++] << bits;
              bits += 8;
              if (bits < op) {
                hold += input[_in++] << bits;
                bits += 8;
              }
            }
            dist += hold & ((1 << op) - 1);
//#ifdef INFLATE_STRICT
            if (dist > dmax) {
              strm.msg = 'invalid distance too far back';
              state.mode = BAD$1;
              break top;
            }
//#endif
            hold >>>= op;
            bits -= op;
            //Tracevv((stderr, "inflate:         distance %u\n", dist));
            op = _out - beg;                /* max distance in output */
            if (dist > op) {                /* see if copy from window */
              op = dist - op;               /* distance back in window */
              if (op > whave) {
                if (state.sane) {
                  strm.msg = 'invalid distance too far back';
                  state.mode = BAD$1;
                  break top;
                }

// (!) This block is disabled in zlib defaults,
// don't enable it for binary compatibility
//#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
//                if (len <= op - whave) {
//                  do {
//                    output[_out++] = 0;
//                  } while (--len);
//                  continue top;
//                }
//                len -= op - whave;
//                do {
//                  output[_out++] = 0;
//                } while (--op > whave);
//                if (op === 0) {
//                  from = _out - dist;
//                  do {
//                    output[_out++] = output[from++];
//                  } while (--len);
//                  continue top;
//                }
//#endif
              }
              from = 0; // window index
              from_source = s_window;
              if (wnext === 0) {           /* very common case */
                from += wsize - op;
                if (op < len) {         /* some from window */
                  len -= op;
                  do {
                    output[_out++] = s_window[from++];
                  } while (--op);
                  from = _out - dist;  /* rest from output */
                  from_source = output;
                }
              }
              else if (wnext < op) {      /* wrap around window */
                from += wsize + wnext - op;
                op -= wnext;
                if (op < len) {         /* some from end of window */
                  len -= op;
                  do {
                    output[_out++] = s_window[from++];
                  } while (--op);
                  from = 0;
                  if (wnext < len) {  /* some from start of window */
                    op = wnext;
                    len -= op;
                    do {
                      output[_out++] = s_window[from++];
                    } while (--op);
                    from = _out - dist;      /* rest from output */
                    from_source = output;
                  }
                }
              }
              else {                      /* contiguous in window */
                from += wnext - op;
                if (op < len) {         /* some from window */
                  len -= op;
                  do {
                    output[_out++] = s_window[from++];
                  } while (--op);
                  from = _out - dist;  /* rest from output */
                  from_source = output;
                }
              }
              while (len > 2) {
                output[_out++] = from_source[from++];
                output[_out++] = from_source[from++];
                output[_out++] = from_source[from++];
                len -= 3;
              }
              if (len) {
                output[_out++] = from_source[from++];
                if (len > 1) {
                  output[_out++] = from_source[from++];
                }
              }
            }
            else {
              from = _out - dist;          /* copy direct from output */
              do {                        /* minimum length is three */
                output[_out++] = output[from++];
                output[_out++] = output[from++];
                output[_out++] = output[from++];
                len -= 3;
              } while (len > 2);
              if (len) {
                output[_out++] = output[from++];
                if (len > 1) {
                  output[_out++] = output[from++];
                }
              }
            }
          }
          else if ((op & 64) === 0) {          /* 2nd level distance code */
            here = dcode[(here & 0xffff)/*here.val*/ + (hold & ((1 << op) - 1))];
            continue dodist;
          }
          else {
            strm.msg = 'invalid distance code';
            state.mode = BAD$1;
            break top;
          }

          break; // need to emulate goto via "continue"
        }
      }
      else if ((op & 64) === 0) {              /* 2nd level length code */
        here = lcode[(here & 0xffff)/*here.val*/ + (hold & ((1 << op) - 1))];
        continue dolen;
      }
      else if (op & 32) {                     /* end-of-block */
        //Tracevv((stderr, "inflate:         end of block\n"));
        state.mode = TYPE$1;
        break top;
      }
      else {
        strm.msg = 'invalid literal/length code';
        state.mode = BAD$1;
        break top;
      }

      break; // need to emulate goto via "continue"
    }
  } while (_in < last && _out < end);

  /* return unused bytes (on entry, bits < 8, so in won't go too far back) */
  len = bits >> 3;
  _in -= len;
  bits -= len << 3;
  hold &= (1 << bits) - 1;

  /* update state and return */
  strm.next_in = _in;
  strm.next_out = _out;
  strm.avail_in = (_in < last ? 5 + (last - _in) : 5 - (_in - last));
  strm.avail_out = (_out < end ? 257 + (end - _out) : 257 - (_out - end));
  state.hold = hold;
  state.bits = bits;
  return;
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

const MAXBITS = 15;
const ENOUGH_LENS$1 = 852;
const ENOUGH_DISTS$1 = 592;
//const ENOUGH = (ENOUGH_LENS+ENOUGH_DISTS);

const CODES$1 = 0;
const LENS$1 = 1;
const DISTS$1 = 2;

const lbase = new Uint16Array([ /* Length codes 257..285 base */
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
  35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0
]);

const lext = new Uint8Array([ /* Length codes 257..285 extra */
  16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18,
  19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 16, 72, 78
]);

const dbase = new Uint16Array([ /* Distance codes 0..29 base */
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
  257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
  8193, 12289, 16385, 24577, 0, 0
]);

const dext = new Uint8Array([ /* Distance codes 0..29 extra */
  16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
  23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
  28, 28, 29, 29, 64, 64
]);

const inflate_table = (type, lens, lens_index, codes, table, table_index, work, opts) =>
{
  const bits = opts.bits;
      //here = opts.here; /* table entry for duplication */

  let len = 0;               /* a code's length in bits */
  let sym = 0;               /* index of code symbols */
  let min = 0, max = 0;          /* minimum and maximum code lengths */
  let root = 0;              /* number of index bits for root table */
  let curr = 0;              /* number of index bits for current table */
  let drop = 0;              /* code bits to drop for sub-table */
  let left = 0;                   /* number of prefix codes available */
  let used = 0;              /* code entries in table used */
  let huff = 0;              /* Huffman code */
  let incr;              /* for incrementing code, index */
  let fill;              /* index for replicating entries */
  let low;               /* low bits for current root entry */
  let mask;              /* mask for low root bits */
  let next;             /* next available space in table */
  let base = null;     /* base value table to use */
//  let shoextra;    /* extra bits table to use */
  let match;                  /* use base and extra for symbol >= match */
  const count = new Uint16Array(MAXBITS + 1); //[MAXBITS+1];    /* number of codes of each length */
  const offs = new Uint16Array(MAXBITS + 1); //[MAXBITS+1];     /* offsets in table for each length */
  let extra = null;

  let here_bits, here_op, here_val;

  /*
   Process a set of code lengths to create a canonical Huffman code.  The
   code lengths are lens[0..codes-1].  Each length corresponds to the
   symbols 0..codes-1.  The Huffman code is generated by first sorting the
   symbols by length from short to long, and retaining the symbol order
   for codes with equal lengths.  Then the code starts with all zero bits
   for the first code of the shortest length, and the codes are integer
   increments for the same length, and zeros are appended as the length
   increases.  For the deflate format, these bits are stored backwards
   from their more natural integer increment ordering, and so when the
   decoding tables are built in the large loop below, the integer codes
   are incremented backwards.

   This routine assumes, but does not check, that all of the entries in
   lens[] are in the range 0..MAXBITS.  The caller must assure this.
   1..MAXBITS is interpreted as that code length.  zero means that that
   symbol does not occur in this code.

   The codes are sorted by computing a count of codes for each length,
   creating from that a table of starting indices for each length in the
   sorted table, and then entering the symbols in order in the sorted
   table.  The sorted table is work[], with that space being provided by
   the caller.

   The length counts are used for other purposes as well, i.e. finding
   the minimum and maximum length codes, determining if there are any
   codes at all, checking for a valid set of lengths, and looking ahead
   at length counts to determine sub-table sizes when building the
   decoding tables.
   */

  /* accumulate lengths for codes (assumes lens[] all in 0..MAXBITS) */
  for (len = 0; len <= MAXBITS; len++) {
    count[len] = 0;
  }
  for (sym = 0; sym < codes; sym++) {
    count[lens[lens_index + sym]]++;
  }

  /* bound code lengths, force root to be within code lengths */
  root = bits;
  for (max = MAXBITS; max >= 1; max--) {
    if (count[max] !== 0) { break; }
  }
  if (root > max) {
    root = max;
  }
  if (max === 0) {                     /* no symbols to code at all */
    //table.op[opts.table_index] = 64;  //here.op = (var char)64;    /* invalid code marker */
    //table.bits[opts.table_index] = 1;   //here.bits = (var char)1;
    //table.val[opts.table_index++] = 0;   //here.val = (var short)0;
    table[table_index++] = (1 << 24) | (64 << 16) | 0;


    //table.op[opts.table_index] = 64;
    //table.bits[opts.table_index] = 1;
    //table.val[opts.table_index++] = 0;
    table[table_index++] = (1 << 24) | (64 << 16) | 0;

    opts.bits = 1;
    return 0;     /* no symbols, but wait for decoding to report error */
  }
  for (min = 1; min < max; min++) {
    if (count[min] !== 0) { break; }
  }
  if (root < min) {
    root = min;
  }

  /* check for an over-subscribed or incomplete set of lengths */
  left = 1;
  for (len = 1; len <= MAXBITS; len++) {
    left <<= 1;
    left -= count[len];
    if (left < 0) {
      return -1;
    }        /* over-subscribed */
  }
  if (left > 0 && (type === CODES$1 || max !== 1)) {
    return -1;                      /* incomplete set */
  }

  /* generate offsets into symbol table for each length for sorting */
  offs[1] = 0;
  for (len = 1; len < MAXBITS; len++) {
    offs[len + 1] = offs[len] + count[len];
  }

  /* sort symbols by length, by symbol order within each length */
  for (sym = 0; sym < codes; sym++) {
    if (lens[lens_index + sym] !== 0) {
      work[offs[lens[lens_index + sym]]++] = sym;
    }
  }

  /*
   Create and fill in decoding tables.  In this loop, the table being
   filled is at next and has curr index bits.  The code being used is huff
   with length len.  That code is converted to an index by dropping drop
   bits off of the bottom.  For codes where len is less than drop + curr,
   those top drop + curr - len bits are incremented through all values to
   fill the table with replicated entries.

   root is the number of index bits for the root table.  When len exceeds
   root, sub-tables are created pointed to by the root entry with an index
   of the low root bits of huff.  This is saved in low to check for when a
   new sub-table should be started.  drop is zero when the root table is
   being filled, and drop is root when sub-tables are being filled.

   When a new sub-table is needed, it is necessary to look ahead in the
   code lengths to determine what size sub-table is needed.  The length
   counts are used for this, and so count[] is decremented as codes are
   entered in the tables.

   used keeps track of how many table entries have been allocated from the
   provided *table space.  It is checked for LENS and DIST tables against
   the constants ENOUGH_LENS and ENOUGH_DISTS to guard against changes in
   the initial root table size constants.  See the comments in inftrees.h
   for more information.

   sym increments through all symbols, and the loop terminates when
   all codes of length max, i.e. all codes, have been processed.  This
   routine permits incomplete codes, so another loop after this one fills
   in the rest of the decoding tables with invalid code markers.
   */

  /* set up for code type */
  // poor man optimization - use if-else instead of switch,
  // to avoid deopts in old v8
  if (type === CODES$1) {
    base = extra = work;    /* dummy value--not used */
    match = 20;

  } else if (type === LENS$1) {
    base = lbase;
    extra = lext;
    match = 257;

  } else {                    /* DISTS */
    base = dbase;
    extra = dext;
    match = 0;
  }

  /* initialize opts for loop */
  huff = 0;                   /* starting code */
  sym = 0;                    /* starting code symbol */
  len = min;                  /* starting code length */
  next = table_index;              /* current table to fill in */
  curr = root;                /* current table index bits */
  drop = 0;                   /* current bits to drop from code for index */
  low = -1;                   /* trigger new sub-table when len > root */
  used = 1 << root;          /* use root table entries */
  mask = used - 1;            /* mask for comparing low */

  /* check available table space */
  if ((type === LENS$1 && used > ENOUGH_LENS$1) ||
    (type === DISTS$1 && used > ENOUGH_DISTS$1)) {
    return 1;
  }

  /* process all codes and make table entries */
  for (;;) {
    /* create table entry */
    here_bits = len - drop;
    if (work[sym] + 1 < match) {
      here_op = 0;
      here_val = work[sym];
    }
    else if (work[sym] >= match) {
      here_op = extra[work[sym] - match];
      here_val = base[work[sym] - match];
    }
    else {
      here_op = 32 + 64;         /* end of block */
      here_val = 0;
    }

    /* replicate for those indices with low len bits equal to huff */
    incr = 1 << (len - drop);
    fill = 1 << curr;
    min = fill;                 /* save offset to next table */
    do {
      fill -= incr;
      table[next + (huff >> drop) + fill] = (here_bits << 24) | (here_op << 16) | here_val |0;
    } while (fill !== 0);

    /* backwards increment the len-bit code huff */
    incr = 1 << (len - 1);
    while (huff & incr) {
      incr >>= 1;
    }
    if (incr !== 0) {
      huff &= incr - 1;
      huff += incr;
    } else {
      huff = 0;
    }

    /* go to next symbol, update count, len */
    sym++;
    if (--count[len] === 0) {
      if (len === max) { break; }
      len = lens[lens_index + work[sym]];
    }

    /* create new sub-table if needed */
    if (len > root && (huff & mask) !== low) {
      /* if first time, transition to sub-tables */
      if (drop === 0) {
        drop = root;
      }

      /* increment past last table */
      next += min;            /* here min is 1 << curr */

      /* determine length of next table */
      curr = len - drop;
      left = 1 << curr;
      while (curr + drop < max) {
        left -= count[curr + drop];
        if (left <= 0) { break; }
        curr++;
        left <<= 1;
      }

      /* check for enough space */
      used += 1 << curr;
      if ((type === LENS$1 && used > ENOUGH_LENS$1) ||
        (type === DISTS$1 && used > ENOUGH_DISTS$1)) {
        return 1;
      }

      /* point entry in root table to sub-table */
      low = huff & mask;
      /*table.op[low] = curr;
      table.bits[low] = root;
      table.val[low] = next - opts.table_index;*/
      table[low] = (root << 24) | (curr << 16) | (next - table_index) |0;
    }
  }

  /* fill in remaining table entry if code is incomplete (guaranteed to have
   at most one remaining entry, since if the code is incomplete, the
   maximum code length that was allowed to get this far is one bit) */
  if (huff !== 0) {
    //table.op[next + huff] = 64;            /* invalid code marker */
    //table.bits[next + huff] = len - drop;
    //table.val[next + huff] = 0;
    table[next + huff] = ((len - drop) << 24) | (64 << 16) |0;
  }

  /* set return parameters */
  //opts.table_index += used;
  opts.bits = root;
  return 0;
};


var inftrees = inflate_table;

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.






const CODES = 0;
const LENS = 1;
const DISTS = 2;

/* Public constants ==========================================================*/
/* ===========================================================================*/

const {
  Z_FINISH: Z_FINISH$1, Z_BLOCK, Z_TREES,
  Z_OK: Z_OK$1, Z_STREAM_END: Z_STREAM_END$1, Z_NEED_DICT: Z_NEED_DICT$1, Z_STREAM_ERROR: Z_STREAM_ERROR$1, Z_DATA_ERROR: Z_DATA_ERROR$1, Z_MEM_ERROR: Z_MEM_ERROR$1, Z_BUF_ERROR,
  Z_DEFLATED
} = constants$2;


/* STATES ====================================================================*/
/* ===========================================================================*/


const    HEAD = 16180;       /* i: waiting for magic header */
const    FLAGS = 16181;      /* i: waiting for method and flags (gzip) */
const    TIME = 16182;       /* i: waiting for modification time (gzip) */
const    OS = 16183;         /* i: waiting for extra flags and operating system (gzip) */
const    EXLEN = 16184;      /* i: waiting for extra length (gzip) */
const    EXTRA = 16185;      /* i: waiting for extra bytes (gzip) */
const    NAME = 16186;       /* i: waiting for end of file name (gzip) */
const    COMMENT = 16187;    /* i: waiting for end of comment (gzip) */
const    HCRC = 16188;       /* i: waiting for header crc (gzip) */
const    DICTID = 16189;    /* i: waiting for dictionary check value */
const    DICT = 16190;      /* waiting for inflateSetDictionary() call */
const        TYPE = 16191;      /* i: waiting for type bits, including last-flag bit */
const        TYPEDO = 16192;    /* i: same, but skip check to exit inflate on new block */
const        STORED = 16193;    /* i: waiting for stored size (length and complement) */
const        COPY_ = 16194;     /* i/o: same as COPY below, but only first time in */
const        COPY = 16195;      /* i/o: waiting for input or output to copy stored block */
const        TABLE = 16196;     /* i: waiting for dynamic block table lengths */
const        LENLENS = 16197;   /* i: waiting for code length code lengths */
const        CODELENS = 16198;  /* i: waiting for length/lit and distance code lengths */
const            LEN_ = 16199;      /* i: same as LEN below, but only first time in */
const            LEN = 16200;       /* i: waiting for length/lit/eob code */
const            LENEXT = 16201;    /* i: waiting for length extra bits */
const            DIST = 16202;      /* i: waiting for distance code */
const            DISTEXT = 16203;   /* i: waiting for distance extra bits */
const            MATCH = 16204;     /* o: waiting for output space to copy string */
const            LIT = 16205;       /* o: waiting for output space to write literal */
const    CHECK = 16206;     /* i: waiting for 32-bit check value */
const    LENGTH = 16207;    /* i: waiting for 32-bit length (gzip) */
const    DONE = 16208;      /* finished check, done -- remain here until reset */
const    BAD = 16209;       /* got a data error -- remain here until reset */
const    MEM = 16210;       /* got an inflate() memory error -- remain here until reset */
const    SYNC = 16211;      /* looking for synchronization bytes to restart inflate() */

/* ===========================================================================*/



const ENOUGH_LENS = 852;
const ENOUGH_DISTS = 592;
//const ENOUGH =  (ENOUGH_LENS+ENOUGH_DISTS);

const MAX_WBITS = 15;
/* 32K LZ77 window */
const DEF_WBITS = MAX_WBITS;


const zswap32 = (q) => {

  return  (((q >>> 24) & 0xff) +
          ((q >>> 8) & 0xff00) +
          ((q & 0xff00) << 8) +
          ((q & 0xff) << 24));
};


function InflateState() {
  this.strm = null;           /* pointer back to this zlib stream */
  this.mode = 0;              /* current inflate mode */
  this.last = false;          /* true if processing last block */
  this.wrap = 0;              /* bit 0 true for zlib, bit 1 true for gzip,
                                 bit 2 true to validate check value */
  this.havedict = false;      /* true if dictionary provided */
  this.flags = 0;             /* gzip header method and flags (0 if zlib), or
                                 -1 if raw or no header yet */
  this.dmax = 0;              /* zlib header max distance (INFLATE_STRICT) */
  this.check = 0;             /* protected copy of check value */
  this.total = 0;             /* protected copy of output count */
  // TODO: may be {}
  this.head = null;           /* where to save gzip header information */

  /* sliding window */
  this.wbits = 0;             /* log base 2 of requested window size */
  this.wsize = 0;             /* window size or zero if not using window */
  this.whave = 0;             /* valid bytes in the window */
  this.wnext = 0;             /* window write index */
  this.window = null;         /* allocated sliding window, if needed */

  /* bit accumulator */
  this.hold = 0;              /* input bit accumulator */
  this.bits = 0;              /* number of bits in "in" */

  /* for string and stored block copying */
  this.length = 0;            /* literal or length of data to copy */
  this.offset = 0;            /* distance back to copy string from */

  /* for table and code decoding */
  this.extra = 0;             /* extra bits needed */

  /* fixed and dynamic code tables */
  this.lencode = null;          /* starting table for length/literal codes */
  this.distcode = null;         /* starting table for distance codes */
  this.lenbits = 0;           /* index bits for lencode */
  this.distbits = 0;          /* index bits for distcode */

  /* dynamic table building */
  this.ncode = 0;             /* number of code length code lengths */
  this.nlen = 0;              /* number of length code lengths */
  this.ndist = 0;             /* number of distance code lengths */
  this.have = 0;              /* number of code lengths in lens[] */
  this.next = null;              /* next available space in codes[] */

  this.lens = new Uint16Array(320); /* temporary storage for code lengths */
  this.work = new Uint16Array(288); /* work area for code table building */

  /*
   because we don't have pointers in js, we use lencode and distcode directly
   as buffers so we don't need codes
  */
  //this.codes = new Int32Array(ENOUGH);       /* space for code tables */
  this.lendyn = null;              /* dynamic table for length/literal codes (JS specific) */
  this.distdyn = null;             /* dynamic table for distance codes (JS specific) */
  this.sane = 0;                   /* if false, allow invalid distance too far */
  this.back = 0;                   /* bits back of last unprocessed length/lit */
  this.was = 0;                    /* initial length of match */
}


const inflateStateCheck = (strm) => {

  if (!strm) {
    return 1;
  }
  const state = strm.state;
  if (!state || state.strm !== strm ||
    state.mode < HEAD || state.mode > SYNC) {
    return 1;
  }
  return 0;
};


const inflateResetKeep = (strm) => {

  if (inflateStateCheck(strm)) { return Z_STREAM_ERROR$1; }
  const state = strm.state;
  strm.total_in = strm.total_out = state.total = 0;
  strm.msg = ''; /*Z_NULL*/
  if (state.wrap) {       /* to support ill-conceived Java test suite */
    strm.adler = state.wrap & 1;
  }
  state.mode = HEAD;
  state.last = 0;
  state.havedict = 0;
  state.flags = -1;
  state.dmax = 32768;
  state.head = null/*Z_NULL*/;
  state.hold = 0;
  state.bits = 0;
  //state.lencode = state.distcode = state.next = state.codes;
  state.lencode = state.lendyn = new Int32Array(ENOUGH_LENS);
  state.distcode = state.distdyn = new Int32Array(ENOUGH_DISTS);

  state.sane = 1;
  state.back = -1;
  //Tracev((stderr, "inflate: reset\n"));
  return Z_OK$1;
};


const inflateReset = (strm) => {

  if (inflateStateCheck(strm)) { return Z_STREAM_ERROR$1; }
  const state = strm.state;
  state.wsize = 0;
  state.whave = 0;
  state.wnext = 0;
  return inflateResetKeep(strm);

};


const inflateReset2 = (strm, windowBits) => {
  let wrap;

  /* get the state */
  if (inflateStateCheck(strm)) { return Z_STREAM_ERROR$1; }
  const state = strm.state;

  /* extract wrap request from windowBits parameter */
  if (windowBits < 0) {
    wrap = 0;
    windowBits = -windowBits;
  }
  else {
    wrap = (windowBits >> 4) + 5;
    if (windowBits < 48) {
      windowBits &= 15;
    }
  }

  /* set number of window bits, free window if different */
  if (windowBits && (windowBits < 8 || windowBits > 15)) {
    return Z_STREAM_ERROR$1;
  }
  if (state.window !== null && state.wbits !== windowBits) {
    state.window = null;
  }

  /* update state and reset the rest of it */
  state.wrap = wrap;
  state.wbits = windowBits;
  return inflateReset(strm);
};


const inflateInit2 = (strm, windowBits) => {

  if (!strm) { return Z_STREAM_ERROR$1; }
  //strm.msg = Z_NULL;                 /* in case we return an error */

  const state = new InflateState();

  //if (state === Z_NULL) return Z_MEM_ERROR;
  //Tracev((stderr, "inflate: allocated\n"));
  strm.state = state;
  state.strm = strm;
  state.window = null/*Z_NULL*/;
  state.mode = HEAD;     /* to pass state test in inflateReset2() */
  const ret = inflateReset2(strm, windowBits);
  if (ret !== Z_OK$1) {
    strm.state = null/*Z_NULL*/;
  }
  return ret;
};


const inflateInit = (strm) => {

  return inflateInit2(strm, DEF_WBITS);
};


/*
 Return state with length and distance decoding tables and index sizes set to
 fixed code decoding.  Normally this returns fixed tables from inffixed.h.
 If BUILDFIXED is defined, then instead this routine builds the tables the
 first time it's called, and returns those tables the first time and
 thereafter.  This reduces the size of the code by about 2K bytes, in
 exchange for a little execution time.  However, BUILDFIXED should not be
 used for threaded applications, since the rewriting of the tables and virgin
 may not be thread-safe.
 */
let virgin = true;

let lenfix, distfix; // We have no pointers in JS, so keep tables separate


const fixedtables = (state) => {

  /* build fixed huffman tables if first call (may not be thread safe) */
  if (virgin) {
    lenfix = new Int32Array(512);
    distfix = new Int32Array(32);

    /* literal/length table */
    let sym = 0;
    while (sym < 144) { state.lens[sym++] = 8; }
    while (sym < 256) { state.lens[sym++] = 9; }
    while (sym < 280) { state.lens[sym++] = 7; }
    while (sym < 288) { state.lens[sym++] = 8; }

    inftrees(LENS,  state.lens, 0, 288, lenfix,   0, state.work, { bits: 9 });

    /* distance table */
    sym = 0;
    while (sym < 32) { state.lens[sym++] = 5; }

    inftrees(DISTS, state.lens, 0, 32,   distfix, 0, state.work, { bits: 5 });

    /* do this just once */
    virgin = false;
  }

  state.lencode = lenfix;
  state.lenbits = 9;
  state.distcode = distfix;
  state.distbits = 5;
};


/*
 Update the window with the last wsize (normally 32K) bytes written before
 returning.  If window does not exist yet, create it.  This is only called
 when a window is already in use, or when output has been written during this
 inflate call, but the end of the deflate stream has not been reached yet.
 It is also called to create a window for dictionary data when a dictionary
 is loaded.

 Providing output buffers larger than 32K to inflate() should provide a speed
 advantage, since only the last 32K of output is copied to the sliding window
 upon return from inflate(), and since all distances after the first 32K of
 output will fall in the output data, making match copies simpler and faster.
 The advantage may be dependent on the size of the processor's data caches.
 */
const updatewindow = (strm, src, end, copy) => {

  let dist;
  const state = strm.state;

  /* if it hasn't been done already, allocate space for the window */
  if (state.window === null) {
    state.wsize = 1 << state.wbits;
    state.wnext = 0;
    state.whave = 0;

    state.window = new Uint8Array(state.wsize);
  }

  /* copy state->wsize or less output bytes into the circular window */
  if (copy >= state.wsize) {
    state.window.set(src.subarray(end - state.wsize, end), 0);
    state.wnext = 0;
    state.whave = state.wsize;
  }
  else {
    dist = state.wsize - state.wnext;
    if (dist > copy) {
      dist = copy;
    }
    //zmemcpy(state->window + state->wnext, end - copy, dist);
    state.window.set(src.subarray(end - copy, end - copy + dist), state.wnext);
    copy -= dist;
    if (copy) {
      //zmemcpy(state->window, end - copy, copy);
      state.window.set(src.subarray(end - copy, end), 0);
      state.wnext = copy;
      state.whave = state.wsize;
    }
    else {
      state.wnext += dist;
      if (state.wnext === state.wsize) { state.wnext = 0; }
      if (state.whave < state.wsize) { state.whave += dist; }
    }
  }
  return 0;
};


const inflate$2 = (strm, flush) => {

  let state;
  let input, output;          // input/output buffers
  let next;                   /* next input INDEX */
  let put;                    /* next output INDEX */
  let have, left;             /* available input and output */
  let hold;                   /* bit buffer */
  let bits;                   /* bits in bit buffer */
  let _in, _out;              /* save starting available input and output */
  let copy;                   /* number of stored or match bytes to copy */
  let from;                   /* where to copy match bytes from */
  let from_source;
  let here = 0;               /* current decoding table entry */
  let here_bits, here_op, here_val; // paked "here" denormalized (JS specific)
  //let last;                   /* parent table entry */
  let last_bits, last_op, last_val; // paked "last" denormalized (JS specific)
  let len;                    /* length to copy for repeats, bits to drop */
  let ret;                    /* return code */
  const hbuf = new Uint8Array(4);    /* buffer for gzip header crc calculation */
  let opts;

  let n; // temporary variable for NEED_BITS

  const order = /* permutation of code lengths */
    new Uint8Array([ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 ]);


  if (inflateStateCheck(strm) || !strm.output ||
      (!strm.input && strm.avail_in !== 0)) {
    return Z_STREAM_ERROR$1;
  }

  state = strm.state;
  if (state.mode === TYPE) { state.mode = TYPEDO; }    /* skip check */


  //--- LOAD() ---
  put = strm.next_out;
  output = strm.output;
  left = strm.avail_out;
  next = strm.next_in;
  input = strm.input;
  have = strm.avail_in;
  hold = state.hold;
  bits = state.bits;
  //---

  _in = have;
  _out = left;
  ret = Z_OK$1;

  inf_leave: // goto emulation
  for (;;) {
    switch (state.mode) {
      case HEAD:
        if (state.wrap === 0) {
          state.mode = TYPEDO;
          break;
        }
        //=== NEEDBITS(16);
        while (bits < 16) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        if ((state.wrap & 2) && hold === 0x8b1f) {  /* gzip header */
          if (state.wbits === 0) {
            state.wbits = 15;
          }
          state.check = 0/*crc32(0L, Z_NULL, 0)*/;
          //=== CRC2(state.check, hold);
          hbuf[0] = hold & 0xff;
          hbuf[1] = (hold >>> 8) & 0xff;
          state.check = crc32_1(state.check, hbuf, 2, 0);
          //===//

          //=== INITBITS();
          hold = 0;
          bits = 0;
          //===//
          state.mode = FLAGS;
          break;
        }
        if (state.head) {
          state.head.done = false;
        }
        if (!(state.wrap & 1) ||   /* check if zlib header allowed */
          (((hold & 0xff)/*BITS(8)*/ << 8) + (hold >> 8)) % 31) {
          strm.msg = 'incorrect header check';
          state.mode = BAD;
          break;
        }
        if ((hold & 0x0f)/*BITS(4)*/ !== Z_DEFLATED) {
          strm.msg = 'unknown compression method';
          state.mode = BAD;
          break;
        }
        //--- DROPBITS(4) ---//
        hold >>>= 4;
        bits -= 4;
        //---//
        len = (hold & 0x0f)/*BITS(4)*/ + 8;
        if (state.wbits === 0) {
          state.wbits = len;
        }
        if (len > 15 || len > state.wbits) {
          strm.msg = 'invalid window size';
          state.mode = BAD;
          break;
        }

        // !!! pako patch. Force use `options.windowBits` if passed.
        // Required to always use max window size by default.
        state.dmax = 1 << state.wbits;
        //state.dmax = 1 << len;

        state.flags = 0;               /* indicate zlib header */
        //Tracev((stderr, "inflate:   zlib header ok\n"));
        strm.adler = state.check = 1/*adler32(0L, Z_NULL, 0)*/;
        state.mode = hold & 0x200 ? DICTID : TYPE;
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        break;
      case FLAGS:
        //=== NEEDBITS(16); */
        while (bits < 16) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        state.flags = hold;
        if ((state.flags & 0xff) !== Z_DEFLATED) {
          strm.msg = 'unknown compression method';
          state.mode = BAD;
          break;
        }
        if (state.flags & 0xe000) {
          strm.msg = 'unknown header flags set';
          state.mode = BAD;
          break;
        }
        if (state.head) {
          state.head.text = ((hold >> 8) & 1);
        }
        if ((state.flags & 0x0200) && (state.wrap & 4)) {
          //=== CRC2(state.check, hold);
          hbuf[0] = hold & 0xff;
          hbuf[1] = (hold >>> 8) & 0xff;
          state.check = crc32_1(state.check, hbuf, 2, 0);
          //===//
        }
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        state.mode = TIME;
        /* falls through */
      case TIME:
        //=== NEEDBITS(32); */
        while (bits < 32) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        if (state.head) {
          state.head.time = hold;
        }
        if ((state.flags & 0x0200) && (state.wrap & 4)) {
          //=== CRC4(state.check, hold)
          hbuf[0] = hold & 0xff;
          hbuf[1] = (hold >>> 8) & 0xff;
          hbuf[2] = (hold >>> 16) & 0xff;
          hbuf[3] = (hold >>> 24) & 0xff;
          state.check = crc32_1(state.check, hbuf, 4, 0);
          //===
        }
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        state.mode = OS;
        /* falls through */
      case OS:
        //=== NEEDBITS(16); */
        while (bits < 16) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        if (state.head) {
          state.head.xflags = (hold & 0xff);
          state.head.os = (hold >> 8);
        }
        if ((state.flags & 0x0200) && (state.wrap & 4)) {
          //=== CRC2(state.check, hold);
          hbuf[0] = hold & 0xff;
          hbuf[1] = (hold >>> 8) & 0xff;
          state.check = crc32_1(state.check, hbuf, 2, 0);
          //===//
        }
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        state.mode = EXLEN;
        /* falls through */
      case EXLEN:
        if (state.flags & 0x0400) {
          //=== NEEDBITS(16); */
          while (bits < 16) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          state.length = hold;
          if (state.head) {
            state.head.extra_len = hold;
          }
          if ((state.flags & 0x0200) && (state.wrap & 4)) {
            //=== CRC2(state.check, hold);
            hbuf[0] = hold & 0xff;
            hbuf[1] = (hold >>> 8) & 0xff;
            state.check = crc32_1(state.check, hbuf, 2, 0);
            //===//
          }
          //=== INITBITS();
          hold = 0;
          bits = 0;
          //===//
        }
        else if (state.head) {
          state.head.extra = null/*Z_NULL*/;
        }
        state.mode = EXTRA;
        /* falls through */
      case EXTRA:
        if (state.flags & 0x0400) {
          copy = state.length;
          if (copy > have) { copy = have; }
          if (copy) {
            if (state.head) {
              len = state.head.extra_len - state.length;
              if (!state.head.extra) {
                // Use untyped array for more convenient processing later
                state.head.extra = new Uint8Array(state.head.extra_len);
              }
              state.head.extra.set(
                input.subarray(
                  next,
                  // extra field is limited to 65536 bytes
                  // - no need for additional size check
                  next + copy
                ),
                /*len + copy > state.head.extra_max - len ? state.head.extra_max : copy,*/
                len
              );
              //zmemcpy(state.head.extra + len, next,
              //        len + copy > state.head.extra_max ?
              //        state.head.extra_max - len : copy);
            }
            if ((state.flags & 0x0200) && (state.wrap & 4)) {
              state.check = crc32_1(state.check, input, copy, next);
            }
            have -= copy;
            next += copy;
            state.length -= copy;
          }
          if (state.length) { break inf_leave; }
        }
        state.length = 0;
        state.mode = NAME;
        /* falls through */
      case NAME:
        if (state.flags & 0x0800) {
          if (have === 0) { break inf_leave; }
          copy = 0;
          do {
            // TODO: 2 or 1 bytes?
            len = input[next + copy++];
            /* use constant limit because in js we should not preallocate memory */
            if (state.head && len &&
                (state.length < 65536 /*state.head.name_max*/)) {
              state.head.name += String.fromCharCode(len);
            }
          } while (len && copy < have);

          if ((state.flags & 0x0200) && (state.wrap & 4)) {
            state.check = crc32_1(state.check, input, copy, next);
          }
          have -= copy;
          next += copy;
          if (len) { break inf_leave; }
        }
        else if (state.head) {
          state.head.name = null;
        }
        state.length = 0;
        state.mode = COMMENT;
        /* falls through */
      case COMMENT:
        if (state.flags & 0x1000) {
          if (have === 0) { break inf_leave; }
          copy = 0;
          do {
            len = input[next + copy++];
            /* use constant limit because in js we should not preallocate memory */
            if (state.head && len &&
                (state.length < 65536 /*state.head.comm_max*/)) {
              state.head.comment += String.fromCharCode(len);
            }
          } while (len && copy < have);
          if ((state.flags & 0x0200) && (state.wrap & 4)) {
            state.check = crc32_1(state.check, input, copy, next);
          }
          have -= copy;
          next += copy;
          if (len) { break inf_leave; }
        }
        else if (state.head) {
          state.head.comment = null;
        }
        state.mode = HCRC;
        /* falls through */
      case HCRC:
        if (state.flags & 0x0200) {
          //=== NEEDBITS(16); */
          while (bits < 16) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          if ((state.wrap & 4) && hold !== (state.check & 0xffff)) {
            strm.msg = 'header crc mismatch';
            state.mode = BAD;
            break;
          }
          //=== INITBITS();
          hold = 0;
          bits = 0;
          //===//
        }
        if (state.head) {
          state.head.hcrc = ((state.flags >> 9) & 1);
          state.head.done = true;
        }
        strm.adler = state.check = 0;
        state.mode = TYPE;
        break;
      case DICTID:
        //=== NEEDBITS(32); */
        while (bits < 32) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        strm.adler = state.check = zswap32(hold);
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        state.mode = DICT;
        /* falls through */
      case DICT:
        if (state.havedict === 0) {
          //--- RESTORE() ---
          strm.next_out = put;
          strm.avail_out = left;
          strm.next_in = next;
          strm.avail_in = have;
          state.hold = hold;
          state.bits = bits;
          //---
          return Z_NEED_DICT$1;
        }
        strm.adler = state.check = 1/*adler32(0L, Z_NULL, 0)*/;
        state.mode = TYPE;
        /* falls through */
      case TYPE:
        if (flush === Z_BLOCK || flush === Z_TREES) { break inf_leave; }
        /* falls through */
      case TYPEDO:
        if (state.last) {
          //--- BYTEBITS() ---//
          hold >>>= bits & 7;
          bits -= bits & 7;
          //---//
          state.mode = CHECK;
          break;
        }
        //=== NEEDBITS(3); */
        while (bits < 3) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        state.last = (hold & 0x01)/*BITS(1)*/;
        //--- DROPBITS(1) ---//
        hold >>>= 1;
        bits -= 1;
        //---//

        switch ((hold & 0x03)/*BITS(2)*/) {
          case 0:                             /* stored block */
            //Tracev((stderr, "inflate:     stored block%s\n",
            //        state.last ? " (last)" : ""));
            state.mode = STORED;
            break;
          case 1:                             /* fixed block */
            fixedtables(state);
            //Tracev((stderr, "inflate:     fixed codes block%s\n",
            //        state.last ? " (last)" : ""));
            state.mode = LEN_;             /* decode codes */
            if (flush === Z_TREES) {
              //--- DROPBITS(2) ---//
              hold >>>= 2;
              bits -= 2;
              //---//
              break inf_leave;
            }
            break;
          case 2:                             /* dynamic block */
            //Tracev((stderr, "inflate:     dynamic codes block%s\n",
            //        state.last ? " (last)" : ""));
            state.mode = TABLE;
            break;
          case 3:
            strm.msg = 'invalid block type';
            state.mode = BAD;
        }
        //--- DROPBITS(2) ---//
        hold >>>= 2;
        bits -= 2;
        //---//
        break;
      case STORED:
        //--- BYTEBITS() ---// /* go to byte boundary */
        hold >>>= bits & 7;
        bits -= bits & 7;
        //---//
        //=== NEEDBITS(32); */
        while (bits < 32) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        if ((hold & 0xffff) !== ((hold >>> 16) ^ 0xffff)) {
          strm.msg = 'invalid stored block lengths';
          state.mode = BAD;
          break;
        }
        state.length = hold & 0xffff;
        //Tracev((stderr, "inflate:       stored length %u\n",
        //        state.length));
        //=== INITBITS();
        hold = 0;
        bits = 0;
        //===//
        state.mode = COPY_;
        if (flush === Z_TREES) { break inf_leave; }
        /* falls through */
      case COPY_:
        state.mode = COPY;
        /* falls through */
      case COPY:
        copy = state.length;
        if (copy) {
          if (copy > have) { copy = have; }
          if (copy > left) { copy = left; }
          if (copy === 0) { break inf_leave; }
          //--- zmemcpy(put, next, copy); ---
          output.set(input.subarray(next, next + copy), put);
          //---//
          have -= copy;
          next += copy;
          left -= copy;
          put += copy;
          state.length -= copy;
          break;
        }
        //Tracev((stderr, "inflate:       stored end\n"));
        state.mode = TYPE;
        break;
      case TABLE:
        //=== NEEDBITS(14); */
        while (bits < 14) {
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
        }
        //===//
        state.nlen = (hold & 0x1f)/*BITS(5)*/ + 257;
        //--- DROPBITS(5) ---//
        hold >>>= 5;
        bits -= 5;
        //---//
        state.ndist = (hold & 0x1f)/*BITS(5)*/ + 1;
        //--- DROPBITS(5) ---//
        hold >>>= 5;
        bits -= 5;
        //---//
        state.ncode = (hold & 0x0f)/*BITS(4)*/ + 4;
        //--- DROPBITS(4) ---//
        hold >>>= 4;
        bits -= 4;
        //---//
//#ifndef PKZIP_BUG_WORKAROUND
        if (state.nlen > 286 || state.ndist > 30) {
          strm.msg = 'too many length or distance symbols';
          state.mode = BAD;
          break;
        }
//#endif
        //Tracev((stderr, "inflate:       table sizes ok\n"));
        state.have = 0;
        state.mode = LENLENS;
        /* falls through */
      case LENLENS:
        while (state.have < state.ncode) {
          //=== NEEDBITS(3);
          while (bits < 3) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          state.lens[order[state.have++]] = (hold & 0x07);//BITS(3);
          //--- DROPBITS(3) ---//
          hold >>>= 3;
          bits -= 3;
          //---//
        }
        while (state.have < 19) {
          state.lens[order[state.have++]] = 0;
        }
        // We have separate tables & no pointers. 2 commented lines below not needed.
        //state.next = state.codes;
        //state.lencode = state.next;
        // Switch to use dynamic table
        state.lencode = state.lendyn;
        state.lenbits = 7;

        opts = { bits: state.lenbits };
        ret = inftrees(CODES, state.lens, 0, 19, state.lencode, 0, state.work, opts);
        state.lenbits = opts.bits;

        if (ret) {
          strm.msg = 'invalid code lengths set';
          state.mode = BAD;
          break;
        }
        //Tracev((stderr, "inflate:       code lengths ok\n"));
        state.have = 0;
        state.mode = CODELENS;
        /* falls through */
      case CODELENS:
        while (state.have < state.nlen + state.ndist) {
          for (;;) {
            here = state.lencode[hold & ((1 << state.lenbits) - 1)];/*BITS(state.lenbits)*/
            here_bits = here >>> 24;
            here_op = (here >>> 16) & 0xff;
            here_val = here & 0xffff;

            if ((here_bits) <= bits) { break; }
            //--- PULLBYTE() ---//
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
            //---//
          }
          if (here_val < 16) {
            //--- DROPBITS(here.bits) ---//
            hold >>>= here_bits;
            bits -= here_bits;
            //---//
            state.lens[state.have++] = here_val;
          }
          else {
            if (here_val === 16) {
              //=== NEEDBITS(here.bits + 2);
              n = here_bits + 2;
              while (bits < n) {
                if (have === 0) { break inf_leave; }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              //===//
              //--- DROPBITS(here.bits) ---//
              hold >>>= here_bits;
              bits -= here_bits;
              //---//
              if (state.have === 0) {
                strm.msg = 'invalid bit length repeat';
                state.mode = BAD;
                break;
              }
              len = state.lens[state.have - 1];
              copy = 3 + (hold & 0x03);//BITS(2);
              //--- DROPBITS(2) ---//
              hold >>>= 2;
              bits -= 2;
              //---//
            }
            else if (here_val === 17) {
              //=== NEEDBITS(here.bits + 3);
              n = here_bits + 3;
              while (bits < n) {
                if (have === 0) { break inf_leave; }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              //===//
              //--- DROPBITS(here.bits) ---//
              hold >>>= here_bits;
              bits -= here_bits;
              //---//
              len = 0;
              copy = 3 + (hold & 0x07);//BITS(3);
              //--- DROPBITS(3) ---//
              hold >>>= 3;
              bits -= 3;
              //---//
            }
            else {
              //=== NEEDBITS(here.bits + 7);
              n = here_bits + 7;
              while (bits < n) {
                if (have === 0) { break inf_leave; }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              //===//
              //--- DROPBITS(here.bits) ---//
              hold >>>= here_bits;
              bits -= here_bits;
              //---//
              len = 0;
              copy = 11 + (hold & 0x7f);//BITS(7);
              //--- DROPBITS(7) ---//
              hold >>>= 7;
              bits -= 7;
              //---//
            }
            if (state.have + copy > state.nlen + state.ndist) {
              strm.msg = 'invalid bit length repeat';
              state.mode = BAD;
              break;
            }
            while (copy--) {
              state.lens[state.have++] = len;
            }
          }
        }

        /* handle error breaks in while */
        if (state.mode === BAD) { break; }

        /* check for end-of-block code (better have one) */
        if (state.lens[256] === 0) {
          strm.msg = 'invalid code -- missing end-of-block';
          state.mode = BAD;
          break;
        }

        /* build code tables -- note: do not change the lenbits or distbits
           values here (9 and 6) without reading the comments in inftrees.h
           concerning the ENOUGH constants, which depend on those values */
        state.lenbits = 9;

        opts = { bits: state.lenbits };
        ret = inftrees(LENS, state.lens, 0, state.nlen, state.lencode, 0, state.work, opts);
        // We have separate tables & no pointers. 2 commented lines below not needed.
        // state.next_index = opts.table_index;
        state.lenbits = opts.bits;
        // state.lencode = state.next;

        if (ret) {
          strm.msg = 'invalid literal/lengths set';
          state.mode = BAD;
          break;
        }

        state.distbits = 6;
        //state.distcode.copy(state.codes);
        // Switch to use dynamic table
        state.distcode = state.distdyn;
        opts = { bits: state.distbits };
        ret = inftrees(DISTS, state.lens, state.nlen, state.ndist, state.distcode, 0, state.work, opts);
        // We have separate tables & no pointers. 2 commented lines below not needed.
        // state.next_index = opts.table_index;
        state.distbits = opts.bits;
        // state.distcode = state.next;

        if (ret) {
          strm.msg = 'invalid distances set';
          state.mode = BAD;
          break;
        }
        //Tracev((stderr, 'inflate:       codes ok\n'));
        state.mode = LEN_;
        if (flush === Z_TREES) { break inf_leave; }
        /* falls through */
      case LEN_:
        state.mode = LEN;
        /* falls through */
      case LEN:
        if (have >= 6 && left >= 258) {
          //--- RESTORE() ---
          strm.next_out = put;
          strm.avail_out = left;
          strm.next_in = next;
          strm.avail_in = have;
          state.hold = hold;
          state.bits = bits;
          //---
          inffast(strm, _out);
          //--- LOAD() ---
          put = strm.next_out;
          output = strm.output;
          left = strm.avail_out;
          next = strm.next_in;
          input = strm.input;
          have = strm.avail_in;
          hold = state.hold;
          bits = state.bits;
          //---

          if (state.mode === TYPE) {
            state.back = -1;
          }
          break;
        }
        state.back = 0;
        for (;;) {
          here = state.lencode[hold & ((1 << state.lenbits) - 1)];  /*BITS(state.lenbits)*/
          here_bits = here >>> 24;
          here_op = (here >>> 16) & 0xff;
          here_val = here & 0xffff;

          if (here_bits <= bits) { break; }
          //--- PULLBYTE() ---//
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
          //---//
        }
        if (here_op && (here_op & 0xf0) === 0) {
          last_bits = here_bits;
          last_op = here_op;
          last_val = here_val;
          for (;;) {
            here = state.lencode[last_val +
                    ((hold & ((1 << (last_bits + last_op)) - 1))/*BITS(last.bits + last.op)*/ >> last_bits)];
            here_bits = here >>> 24;
            here_op = (here >>> 16) & 0xff;
            here_val = here & 0xffff;

            if ((last_bits + here_bits) <= bits) { break; }
            //--- PULLBYTE() ---//
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
            //---//
          }
          //--- DROPBITS(last.bits) ---//
          hold >>>= last_bits;
          bits -= last_bits;
          //---//
          state.back += last_bits;
        }
        //--- DROPBITS(here.bits) ---//
        hold >>>= here_bits;
        bits -= here_bits;
        //---//
        state.back += here_bits;
        state.length = here_val;
        if (here_op === 0) {
          //Tracevv((stderr, here.val >= 0x20 && here.val < 0x7f ?
          //        "inflate:         literal '%c'\n" :
          //        "inflate:         literal 0x%02x\n", here.val));
          state.mode = LIT;
          break;
        }
        if (here_op & 32) {
          //Tracevv((stderr, "inflate:         end of block\n"));
          state.back = -1;
          state.mode = TYPE;
          break;
        }
        if (here_op & 64) {
          strm.msg = 'invalid literal/length code';
          state.mode = BAD;
          break;
        }
        state.extra = here_op & 15;
        state.mode = LENEXT;
        /* falls through */
      case LENEXT:
        if (state.extra) {
          //=== NEEDBITS(state.extra);
          n = state.extra;
          while (bits < n) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          state.length += hold & ((1 << state.extra) - 1)/*BITS(state.extra)*/;
          //--- DROPBITS(state.extra) ---//
          hold >>>= state.extra;
          bits -= state.extra;
          //---//
          state.back += state.extra;
        }
        //Tracevv((stderr, "inflate:         length %u\n", state.length));
        state.was = state.length;
        state.mode = DIST;
        /* falls through */
      case DIST:
        for (;;) {
          here = state.distcode[hold & ((1 << state.distbits) - 1)];/*BITS(state.distbits)*/
          here_bits = here >>> 24;
          here_op = (here >>> 16) & 0xff;
          here_val = here & 0xffff;

          if ((here_bits) <= bits) { break; }
          //--- PULLBYTE() ---//
          if (have === 0) { break inf_leave; }
          have--;
          hold += input[next++] << bits;
          bits += 8;
          //---//
        }
        if ((here_op & 0xf0) === 0) {
          last_bits = here_bits;
          last_op = here_op;
          last_val = here_val;
          for (;;) {
            here = state.distcode[last_val +
                    ((hold & ((1 << (last_bits + last_op)) - 1))/*BITS(last.bits + last.op)*/ >> last_bits)];
            here_bits = here >>> 24;
            here_op = (here >>> 16) & 0xff;
            here_val = here & 0xffff;

            if ((last_bits + here_bits) <= bits) { break; }
            //--- PULLBYTE() ---//
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
            //---//
          }
          //--- DROPBITS(last.bits) ---//
          hold >>>= last_bits;
          bits -= last_bits;
          //---//
          state.back += last_bits;
        }
        //--- DROPBITS(here.bits) ---//
        hold >>>= here_bits;
        bits -= here_bits;
        //---//
        state.back += here_bits;
        if (here_op & 64) {
          strm.msg = 'invalid distance code';
          state.mode = BAD;
          break;
        }
        state.offset = here_val;
        state.extra = (here_op) & 15;
        state.mode = DISTEXT;
        /* falls through */
      case DISTEXT:
        if (state.extra) {
          //=== NEEDBITS(state.extra);
          n = state.extra;
          while (bits < n) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          state.offset += hold & ((1 << state.extra) - 1)/*BITS(state.extra)*/;
          //--- DROPBITS(state.extra) ---//
          hold >>>= state.extra;
          bits -= state.extra;
          //---//
          state.back += state.extra;
        }
//#ifdef INFLATE_STRICT
        if (state.offset > state.dmax) {
          strm.msg = 'invalid distance too far back';
          state.mode = BAD;
          break;
        }
//#endif
        //Tracevv((stderr, "inflate:         distance %u\n", state.offset));
        state.mode = MATCH;
        /* falls through */
      case MATCH:
        if (left === 0) { break inf_leave; }
        copy = _out - left;
        if (state.offset > copy) {         /* copy from window */
          copy = state.offset - copy;
          if (copy > state.whave) {
            if (state.sane) {
              strm.msg = 'invalid distance too far back';
              state.mode = BAD;
              break;
            }
// (!) This block is disabled in zlib defaults,
// don't enable it for binary compatibility
//#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
//          Trace((stderr, "inflate.c too far\n"));
//          copy -= state.whave;
//          if (copy > state.length) { copy = state.length; }
//          if (copy > left) { copy = left; }
//          left -= copy;
//          state.length -= copy;
//          do {
//            output[put++] = 0;
//          } while (--copy);
//          if (state.length === 0) { state.mode = LEN; }
//          break;
//#endif
          }
          if (copy > state.wnext) {
            copy -= state.wnext;
            from = state.wsize - copy;
          }
          else {
            from = state.wnext - copy;
          }
          if (copy > state.length) { copy = state.length; }
          from_source = state.window;
        }
        else {                              /* copy from output */
          from_source = output;
          from = put - state.offset;
          copy = state.length;
        }
        if (copy > left) { copy = left; }
        left -= copy;
        state.length -= copy;
        do {
          output[put++] = from_source[from++];
        } while (--copy);
        if (state.length === 0) { state.mode = LEN; }
        break;
      case LIT:
        if (left === 0) { break inf_leave; }
        output[put++] = state.length;
        left--;
        state.mode = LEN;
        break;
      case CHECK:
        if (state.wrap) {
          //=== NEEDBITS(32);
          while (bits < 32) {
            if (have === 0) { break inf_leave; }
            have--;
            // Use '|' instead of '+' to make sure that result is signed
            hold |= input[next++] << bits;
            bits += 8;
          }
          //===//
          _out -= left;
          strm.total_out += _out;
          state.total += _out;
          if ((state.wrap & 4) && _out) {
            strm.adler = state.check =
                /*UPDATE_CHECK(state.check, put - _out, _out);*/
                (state.flags ? crc32_1(state.check, output, _out, put - _out) : adler32_1(state.check, output, _out, put - _out));

          }
          _out = left;
          // NB: crc32 stored as signed 32-bit int, zswap32 returns signed too
          if ((state.wrap & 4) && (state.flags ? hold : zswap32(hold)) !== state.check) {
            strm.msg = 'incorrect data check';
            state.mode = BAD;
            break;
          }
          //=== INITBITS();
          hold = 0;
          bits = 0;
          //===//
          //Tracev((stderr, "inflate:   check matches trailer\n"));
        }
        state.mode = LENGTH;
        /* falls through */
      case LENGTH:
        if (state.wrap && state.flags) {
          //=== NEEDBITS(32);
          while (bits < 32) {
            if (have === 0) { break inf_leave; }
            have--;
            hold += input[next++] << bits;
            bits += 8;
          }
          //===//
          if ((state.wrap & 4) && hold !== (state.total & 0xffffffff)) {
            strm.msg = 'incorrect length check';
            state.mode = BAD;
            break;
          }
          //=== INITBITS();
          hold = 0;
          bits = 0;
          //===//
          //Tracev((stderr, "inflate:   length matches trailer\n"));
        }
        state.mode = DONE;
        /* falls through */
      case DONE:
        ret = Z_STREAM_END$1;
        break inf_leave;
      case BAD:
        ret = Z_DATA_ERROR$1;
        break inf_leave;
      case MEM:
        return Z_MEM_ERROR$1;
      case SYNC:
        /* falls through */
      default:
        return Z_STREAM_ERROR$1;
    }
  }

  // inf_leave <- here is real place for "goto inf_leave", emulated via "break inf_leave"

  /*
     Return from inflate(), updating the total counts and the check value.
     If there was no progress during the inflate() call, return a buffer
     error.  Call updatewindow() to create and/or update the window state.
     Note: a memory error from inflate() is non-recoverable.
   */

  //--- RESTORE() ---
  strm.next_out = put;
  strm.avail_out = left;
  strm.next_in = next;
  strm.avail_in = have;
  state.hold = hold;
  state.bits = bits;
  //---

  if (state.wsize || (_out !== strm.avail_out && state.mode < BAD &&
                      (state.mode < CHECK || flush !== Z_FINISH$1))) {
    if (updatewindow(strm, strm.output, strm.next_out, _out - strm.avail_out)) ;
  }
  _in -= strm.avail_in;
  _out -= strm.avail_out;
  strm.total_in += _in;
  strm.total_out += _out;
  state.total += _out;
  if ((state.wrap & 4) && _out) {
    strm.adler = state.check = /*UPDATE_CHECK(state.check, strm.next_out - _out, _out);*/
      (state.flags ? crc32_1(state.check, output, _out, strm.next_out - _out) : adler32_1(state.check, output, _out, strm.next_out - _out));
  }
  strm.data_type = state.bits + (state.last ? 64 : 0) +
                    (state.mode === TYPE ? 128 : 0) +
                    (state.mode === LEN_ || state.mode === COPY_ ? 256 : 0);
  if (((_in === 0 && _out === 0) || flush === Z_FINISH$1) && ret === Z_OK$1) {
    ret = Z_BUF_ERROR;
  }
  return ret;
};


const inflateEnd = (strm) => {

  if (inflateStateCheck(strm)) {
    return Z_STREAM_ERROR$1;
  }

  let state = strm.state;
  if (state.window) {
    state.window = null;
  }
  strm.state = null;
  return Z_OK$1;
};


const inflateGetHeader = (strm, head) => {

  /* check state */
  if (inflateStateCheck(strm)) { return Z_STREAM_ERROR$1; }
  const state = strm.state;
  if ((state.wrap & 2) === 0) { return Z_STREAM_ERROR$1; }

  /* save header structure */
  state.head = head;
  head.done = false;
  return Z_OK$1;
};


const inflateSetDictionary = (strm, dictionary) => {
  const dictLength = dictionary.length;

  let state;
  let dictid;
  let ret;

  /* check state */
  if (inflateStateCheck(strm)) { return Z_STREAM_ERROR$1; }
  state = strm.state;

  if (state.wrap !== 0 && state.mode !== DICT) {
    return Z_STREAM_ERROR$1;
  }

  /* check for correct dictionary identifier */
  if (state.mode === DICT) {
    dictid = 1; /* adler32(0, null, 0)*/
    /* dictid = adler32(dictid, dictionary, dictLength); */
    dictid = adler32_1(dictid, dictionary, dictLength, 0);
    if (dictid !== state.check) {
      return Z_DATA_ERROR$1;
    }
  }
  /* copy dictionary to window using updatewindow(), which will amend the
   existing dictionary if appropriate */
  ret = updatewindow(strm, dictionary, dictLength, dictLength);
  if (ret) {
    state.mode = MEM;
    return Z_MEM_ERROR$1;
  }
  state.havedict = 1;
  // Tracev((stderr, "inflate:   dictionary set\n"));
  return Z_OK$1;
};


var inflateReset_1 = inflateReset;
var inflateReset2_1 = inflateReset2;
var inflateResetKeep_1 = inflateResetKeep;
var inflateInit_1 = inflateInit;
var inflateInit2_1 = inflateInit2;
var inflate_2$1 = inflate$2;
var inflateEnd_1 = inflateEnd;
var inflateGetHeader_1 = inflateGetHeader;
var inflateSetDictionary_1 = inflateSetDictionary;
var inflateInfo = 'pako inflate (from Nodeca project)';

/* Not implemented
module.exports.inflateCodesUsed = inflateCodesUsed;
module.exports.inflateCopy = inflateCopy;
module.exports.inflateGetDictionary = inflateGetDictionary;
module.exports.inflateMark = inflateMark;
module.exports.inflatePrime = inflatePrime;
module.exports.inflateSync = inflateSync;
module.exports.inflateSyncPoint = inflateSyncPoint;
module.exports.inflateUndermine = inflateUndermine;
module.exports.inflateValidate = inflateValidate;
*/

var inflate_1$2 = {
	inflateReset: inflateReset_1,
	inflateReset2: inflateReset2_1,
	inflateResetKeep: inflateResetKeep_1,
	inflateInit: inflateInit_1,
	inflateInit2: inflateInit2_1,
	inflate: inflate_2$1,
	inflateEnd: inflateEnd_1,
	inflateGetHeader: inflateGetHeader_1,
	inflateSetDictionary: inflateSetDictionary_1,
	inflateInfo: inflateInfo
};

// (C) 1995-2013 Jean-loup Gailly and Mark Adler
// (C) 2014-2017 Vitaly Puzrin and Andrey Tupitsin
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

function GZheader() {
  /* true if compressed data believed to be text */
  this.text       = 0;
  /* modification time */
  this.time       = 0;
  /* extra flags (not used when writing a gzip file) */
  this.xflags     = 0;
  /* operating system */
  this.os         = 0;
  /* pointer to extra field or Z_NULL if none */
  this.extra      = null;
  /* extra field length (valid if extra != Z_NULL) */
  this.extra_len  = 0; // Actually, we don't need it in JS,
                       // but leave for few code modifications

  //
  // Setup limits is not necessary because in js we should not preallocate memory
  // for inflate use constant limit in 65536 bytes
  //

  /* space at extra (only when reading header) */
  // this.extra_max  = 0;
  /* pointer to zero-terminated file name or Z_NULL */
  this.name       = '';
  /* space at name (only when reading header) */
  // this.name_max   = 0;
  /* pointer to zero-terminated comment or Z_NULL */
  this.comment    = '';
  /* space at comment (only when reading header) */
  // this.comm_max   = 0;
  /* true if there was or will be a header crc */
  this.hcrc       = 0;
  /* true when done reading gzip header (not used when writing a gzip file) */
  this.done       = false;
}

var gzheader = GZheader;

const toString = Object.prototype.toString;

/* Public constants ==========================================================*/
/* ===========================================================================*/

const {
  Z_NO_FLUSH, Z_FINISH,
  Z_OK, Z_STREAM_END, Z_NEED_DICT, Z_STREAM_ERROR, Z_DATA_ERROR, Z_MEM_ERROR
} = constants$2;

/* ===========================================================================*/


/**
 * class Inflate
 *
 * Generic JS-style wrapper for zlib calls. If you don't need
 * streaming behaviour - use more simple functions: [[inflate]]
 * and [[inflateRaw]].
 **/

/* internal
 * inflate.chunks -> Array
 *
 * Chunks of output data, if [[Inflate#onData]] not overridden.
 **/

/**
 * Inflate.result -> Uint8Array|String
 *
 * Uncompressed result, generated by default [[Inflate#onData]]
 * and [[Inflate#onEnd]] handlers. Filled after you push last chunk
 * (call [[Inflate#push]] with `Z_FINISH` / `true` param).
 **/

/**
 * Inflate.err -> Number
 *
 * Error code after inflate finished. 0 (Z_OK) on success.
 * Should be checked if broken data possible.
 **/

/**
 * Inflate.msg -> String
 *
 * Error message, if [[Inflate.err]] != 0
 **/


/**
 * new Inflate(options)
 * - options (Object): zlib inflate options.
 *
 * Creates new inflator instance with specified params. Throws exception
 * on bad params. Supported options:
 *
 * - `windowBits`
 * - `dictionary`
 *
 * [http://zlib.net/manual.html#Advanced](http://zlib.net/manual.html#Advanced)
 * for more information on these.
 *
 * Additional options, for internal needs:
 *
 * - `chunkSize` - size of generated data chunks (16K by default)
 * - `raw` (Boolean) - do raw inflate
 * - `to` (String) - if equal to 'string', then result will be converted
 *   from utf8 to utf16 (javascript) string. When string output requested,
 *   chunk length can differ from `chunkSize`, depending on content.
 *
 * By default, when no options set, autodetect deflate/gzip data format via
 * wrapper header.
 *
 * ##### Example:
 *
 * ```javascript
 * const pako = require('pako')
 * const chunk1 = new Uint8Array([1,2,3,4,5,6,7,8,9])
 * const chunk2 = new Uint8Array([10,11,12,13,14,15,16,17,18,19]);
 *
 * const inflate = new pako.Inflate({ level: 3});
 *
 * inflate.push(chunk1, false);
 * inflate.push(chunk2, true);  // true -> last chunk
 *
 * if (inflate.err) { throw new Error(inflate.err); }
 *
 * console.log(inflate.result);
 * ```
 **/
function Inflate$1(options) {
  this.options = common.assign({
    chunkSize: 1024 * 64,
    windowBits: 15,
    to: ''
  }, options || {});

  const opt = this.options;

  // Force window size for `raw` data, if not set directly,
  // because we have no header for autodetect.
  if (opt.raw && (opt.windowBits >= 0) && (opt.windowBits < 16)) {
    opt.windowBits = -opt.windowBits;
    if (opt.windowBits === 0) { opt.windowBits = -15; }
  }

  // If `windowBits` not defined (and mode not raw) - set autodetect flag for gzip/deflate
  if ((opt.windowBits >= 0) && (opt.windowBits < 16) &&
      !(options && options.windowBits)) {
    opt.windowBits += 32;
  }

  // Gzip header has no info about windows size, we can do autodetect only
  // for deflate. So, if window size not set, force it to max when gzip possible
  if ((opt.windowBits > 15) && (opt.windowBits < 48)) {
    // bit 3 (16) -> gzipped data
    // bit 4 (32) -> autodetect gzip/deflate
    if ((opt.windowBits & 15) === 0) {
      opt.windowBits |= 15;
    }
  }

  this.err    = 0;      // error code, if happens (0 = Z_OK)
  this.msg    = '';     // error message
  this.ended  = false;  // used to avoid multiple onEnd() calls
  this.chunks = [];     // chunks of compressed data

  this.strm   = new zstream();
  this.strm.avail_out = 0;

  let status  = inflate_1$2.inflateInit2(
    this.strm,
    opt.windowBits
  );

  if (status !== Z_OK) {
    throw new Error(messages[status]);
  }

  this.header = new gzheader();

  inflate_1$2.inflateGetHeader(this.strm, this.header);

  // Setup dictionary
  if (opt.dictionary) {
    // Convert data if needed
    if (typeof opt.dictionary === 'string') {
      opt.dictionary = strings.string2buf(opt.dictionary);
    } else if (toString.call(opt.dictionary) === '[object ArrayBuffer]') {
      opt.dictionary = new Uint8Array(opt.dictionary);
    }
    if (opt.raw) { //In raw mode we need to set the dictionary early
      status = inflate_1$2.inflateSetDictionary(this.strm, opt.dictionary);
      if (status !== Z_OK) {
        throw new Error(messages[status]);
      }
    }
  }
}

/**
 * Inflate#push(data[, flush_mode]) -> Boolean
 * - data (Uint8Array|ArrayBuffer): input data
 * - flush_mode (Number|Boolean): 0..6 for corresponding Z_NO_FLUSH..Z_TREE
 *   flush modes. See constants. Skipped or `false` means Z_NO_FLUSH,
 *   `true` means Z_FINISH.
 *
 * Sends input data to inflate pipe, generating [[Inflate#onData]] calls with
 * new output chunks. Returns `true` on success. If end of stream detected,
 * [[Inflate#onEnd]] will be called.
 *
 * `flush_mode` is not needed for normal operation, because end of stream
 * detected automatically. You may try to use it for advanced things, but
 * this functionality was not tested.
 *
 * On fail call [[Inflate#onEnd]] with error code and return false.
 *
 * ##### Example
 *
 * ```javascript
 * push(chunk, false); // push one of data chunks
 * ...
 * push(chunk, true);  // push last chunk
 * ```
 **/
Inflate$1.prototype.push = function (data, flush_mode) {
  const strm = this.strm;
  const chunkSize = this.options.chunkSize;
  const dictionary = this.options.dictionary;
  let status, _flush_mode, last_avail_out;

  if (this.ended) return false;

  if (flush_mode === ~~flush_mode) _flush_mode = flush_mode;
  else _flush_mode = flush_mode === true ? Z_FINISH : Z_NO_FLUSH;

  // Convert data if needed
  if (toString.call(data) === '[object ArrayBuffer]') {
    strm.input = new Uint8Array(data);
  } else {
    strm.input = data;
  }

  strm.next_in = 0;
  strm.avail_in = strm.input.length;

  for (;;) {
    if (strm.avail_out === 0) {
      strm.output = new Uint8Array(chunkSize);
      strm.next_out = 0;
      strm.avail_out = chunkSize;
    }

    status = inflate_1$2.inflate(strm, _flush_mode);

    if (status === Z_NEED_DICT && dictionary) {
      status = inflate_1$2.inflateSetDictionary(strm, dictionary);

      if (status === Z_OK) {
        status = inflate_1$2.inflate(strm, _flush_mode);
      } else if (status === Z_DATA_ERROR) {
        // Replace code with more verbose
        status = Z_NEED_DICT;
      }
    }

    // Skip snyc markers if more data follows and not raw mode
    while (strm.avail_in > 0 &&
           status === Z_STREAM_END &&
           strm.state.wrap > 0 &&
           data[strm.next_in] !== 0)
    {
      inflate_1$2.inflateReset(strm);
      status = inflate_1$2.inflate(strm, _flush_mode);
    }

    switch (status) {
      case Z_STREAM_ERROR:
      case Z_DATA_ERROR:
      case Z_NEED_DICT:
      case Z_MEM_ERROR:
        this.onEnd(status);
        this.ended = true;
        return false;
    }

    // Remember real `avail_out` value, because we may patch out buffer content
    // to align utf8 strings boundaries.
    last_avail_out = strm.avail_out;

    if (strm.next_out) {
      if (strm.avail_out === 0 || status === Z_STREAM_END) {

        if (this.options.to === 'string') {

          let next_out_utf8 = strings.utf8border(strm.output, strm.next_out);

          let tail = strm.next_out - next_out_utf8;
          let utf8str = strings.buf2string(strm.output, next_out_utf8);

          // move tail & realign counters
          strm.next_out = tail;
          strm.avail_out = chunkSize - tail;
          if (tail) strm.output.set(strm.output.subarray(next_out_utf8, next_out_utf8 + tail), 0);

          this.onData(utf8str);

        } else {
          this.onData(strm.output.length === strm.next_out ? strm.output : strm.output.subarray(0, strm.next_out));
        }
      }
    }

    // Must repeat iteration if out buffer is full
    if (status === Z_OK && last_avail_out === 0) continue;

    // Finalize if end of stream reached.
    if (status === Z_STREAM_END) {
      status = inflate_1$2.inflateEnd(this.strm);
      this.onEnd(status);
      this.ended = true;
      return true;
    }

    if (strm.avail_in === 0) break;
  }

  return true;
};


/**
 * Inflate#onData(chunk) -> Void
 * - chunk (Uint8Array|String): output data. When string output requested,
 *   each chunk will be string.
 *
 * By default, stores data blocks in `chunks[]` property and glue
 * those in `onEnd`. Override this handler, if you need another behaviour.
 **/
Inflate$1.prototype.onData = function (chunk) {
  this.chunks.push(chunk);
};


/**
 * Inflate#onEnd(status) -> Void
 * - status (Number): inflate status. 0 (Z_OK) on success,
 *   other if not.
 *
 * Called either after you tell inflate that the input stream is
 * complete (Z_FINISH). By default - join collected chunks,
 * free memory and fill `results` / `err` properties.
 **/
Inflate$1.prototype.onEnd = function (status) {
  // On success - join
  if (status === Z_OK) {
    if (this.options.to === 'string') {
      this.result = this.chunks.join('');
    } else {
      this.result = common.flattenChunks(this.chunks);
    }
  }
  this.chunks = [];
  this.err = status;
  this.msg = this.strm.msg;
};


/**
 * inflate(data[, options]) -> Uint8Array|String
 * - data (Uint8Array|ArrayBuffer): input data to decompress.
 * - options (Object): zlib inflate options.
 *
 * Decompress `data` with inflate/ungzip and `options`. Autodetect
 * format via wrapper header by default. That's why we don't provide
 * separate `ungzip` method.
 *
 * Supported options are:
 *
 * - windowBits
 *
 * [http://zlib.net/manual.html#Advanced](http://zlib.net/manual.html#Advanced)
 * for more information.
 *
 * Sugar (options):
 *
 * - `raw` (Boolean) - say that we work with raw stream, if you don't wish to specify
 *   negative windowBits implicitly.
 * - `to` (String) - if equal to 'string', then result will be converted
 *   from utf8 to utf16 (javascript) string. When string output requested,
 *   chunk length can differ from `chunkSize`, depending on content.
 *
 *
 * ##### Example:
 *
 * ```javascript
 * const pako = require('pako');
 * const input = pako.deflate(new Uint8Array([1,2,3,4,5,6,7,8,9]));
 * let output;
 *
 * try {
 *   output = pako.inflate(input);
 * } catch (err) {
 *   console.log(err);
 * }
 * ```
 **/
function inflate$1(input, options) {
  const inflator = new Inflate$1(options);

  inflator.push(input);

  // That will never happens, if you don't cheat with options :)
  if (inflator.err) throw inflator.msg || messages[inflator.err];

  return inflator.result;
}


/**
 * inflateRaw(data[, options]) -> Uint8Array|String
 * - data (Uint8Array|ArrayBuffer): input data to decompress.
 * - options (Object): zlib inflate options.
 *
 * The same as [[inflate]], but creates raw data, without wrapper
 * (header and adler32 crc).
 **/
function inflateRaw$1(input, options) {
  options = options || {};
  options.raw = true;
  return inflate$1(input, options);
}


/**
 * ungzip(data[, options]) -> Uint8Array|String
 * - data (Uint8Array|ArrayBuffer): input data to decompress.
 * - options (Object): zlib inflate options.
 *
 * Just shortcut to [[inflate]], because it autodetects format
 * by header.content. Done for convenience.
 **/


var Inflate_1$1 = Inflate$1;
var inflate_2 = inflate$1;
var inflateRaw_1$1 = inflateRaw$1;
var ungzip$1 = inflate$1;
var constants = constants$2;

var inflate_1$1 = {
	Inflate: Inflate_1$1,
	inflate: inflate_2,
	inflateRaw: inflateRaw_1$1,
	ungzip: ungzip$1,
	constants: constants
};

const { Inflate, inflate, inflateRaw, ungzip } = inflate_1$1;
var Inflate_1 = Inflate;
var inflate_1 = inflate;

const crcTable = [];
for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
        if (c & 1) {
            c = 0xedb88320 ^ (c >>> 1);
        }
        else {
            c = c >>> 1;
        }
    }
    crcTable[n] = c;
}
const initialCrc = 0xffffffff;
function updateCrc(currentCrc, data, length) {
    let c = currentCrc;
    for (let n = 0; n < length; n++) {
        c = crcTable[(c ^ data[n]) & 0xff] ^ (c >>> 8);
    }
    return c;
}
function crc(data, length) {
    return (updateCrc(initialCrc, data, length) ^ initialCrc) >>> 0;
}
function checkCrc(buffer, crcLength, chunkName) {
    const expectedCrc = buffer.readUint32();
    const actualCrc = crc(new Uint8Array(buffer.buffer, buffer.byteOffset + buffer.offset - crcLength - 4, crcLength), crcLength); // "- 4" because we already advanced by reading the CRC
    if (actualCrc !== expectedCrc) {
        throw new Error(`CRC mismatch for chunk ${chunkName}. Expected ${expectedCrc}, found ${actualCrc}`);
    }
}

function unfilterNone(currentLine, newLine, bytesPerLine) {
    for (let i = 0; i < bytesPerLine; i++) {
        newLine[i] = currentLine[i];
    }
}
function unfilterSub(currentLine, newLine, bytesPerLine, bytesPerPixel) {
    let i = 0;
    for (; i < bytesPerPixel; i++) {
        // just copy first bytes
        newLine[i] = currentLine[i];
    }
    for (; i < bytesPerLine; i++) {
        newLine[i] = (currentLine[i] + newLine[i - bytesPerPixel]) & 0xff;
    }
}
function unfilterUp(currentLine, newLine, prevLine, bytesPerLine) {
    let i = 0;
    if (prevLine.length === 0) {
        // just copy bytes for first line
        for (; i < bytesPerLine; i++) {
            newLine[i] = currentLine[i];
        }
    }
    else {
        for (; i < bytesPerLine; i++) {
            newLine[i] = (currentLine[i] + prevLine[i]) & 0xff;
        }
    }
}
function unfilterAverage(currentLine, newLine, prevLine, bytesPerLine, bytesPerPixel) {
    let i = 0;
    if (prevLine.length === 0) {
        for (; i < bytesPerPixel; i++) {
            newLine[i] = currentLine[i];
        }
        for (; i < bytesPerLine; i++) {
            newLine[i] = (currentLine[i] + (newLine[i - bytesPerPixel] >> 1)) & 0xff;
        }
    }
    else {
        for (; i < bytesPerPixel; i++) {
            newLine[i] = (currentLine[i] + (prevLine[i] >> 1)) & 0xff;
        }
        for (; i < bytesPerLine; i++) {
            newLine[i] =
                (currentLine[i] + ((newLine[i - bytesPerPixel] + prevLine[i]) >> 1)) &
                    0xff;
        }
    }
}
function unfilterPaeth(currentLine, newLine, prevLine, bytesPerLine, bytesPerPixel) {
    let i = 0;
    if (prevLine.length === 0) {
        for (; i < bytesPerPixel; i++) {
            newLine[i] = currentLine[i];
        }
        for (; i < bytesPerLine; i++) {
            newLine[i] = (currentLine[i] + newLine[i - bytesPerPixel]) & 0xff;
        }
    }
    else {
        for (; i < bytesPerPixel; i++) {
            newLine[i] = (currentLine[i] + prevLine[i]) & 0xff;
        }
        for (; i < bytesPerLine; i++) {
            newLine[i] =
                (currentLine[i] +
                    paethPredictor$1(newLine[i - bytesPerPixel], prevLine[i], prevLine[i - bytesPerPixel])) &
                    0xff;
        }
    }
}
function paethPredictor$1(a, b, c) {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    if (pa <= pb && pa <= pc)
        return a;
    else if (pb <= pc)
        return b;
    else
        return c;
}

/**
 * Apllies filter on scanline based on the filter type.
 * @param filterType - The filter type to apply.
 * @param currentLine - The current line of pixel data.
 * @param newLine - The new line of pixel data.
 * @param prevLine - The previous line of pixel data.
 * @param passLineBytes - The number of bytes in the pass line.
 * @param bytesPerPixel - The number of bytes per pixel.
 */
function applyUnfilter(filterType, currentLine, newLine, prevLine, passLineBytes, bytesPerPixel) {
    switch (filterType) {
        case 0:
            unfilterNone(currentLine, newLine, passLineBytes);
            break;
        case 1:
            unfilterSub(currentLine, newLine, passLineBytes, bytesPerPixel);
            break;
        case 2:
            unfilterUp(currentLine, newLine, prevLine, passLineBytes);
            break;
        case 3:
            unfilterAverage(currentLine, newLine, prevLine, passLineBytes, bytesPerPixel);
            break;
        case 4:
            unfilterPaeth(currentLine, newLine, prevLine, passLineBytes, bytesPerPixel);
            break;
        default:
            throw new Error(`Unsupported filter: ${filterType}`);
    }
}

const uint16$1 = new Uint16Array([0x00ff]);
const uint8$1 = new Uint8Array(uint16$1.buffer);
const osIsLittleEndian$1 = uint8$1[0] === 0xff;
/**
 * Decodes the Adam7 interlaced PNG data.
 *
 * @param params - DecodeInterlaceNullParams
 * @returns - array of pixel data.
 */
function decodeInterlaceAdam7(params) {
    const { data, width, height, channels, depth } = params;
    // Adam7 interlacing pattern
    const passes = [
        { x: 0, y: 0, xStep: 8, yStep: 8 }, // Pass 1
        { x: 4, y: 0, xStep: 8, yStep: 8 }, // Pass 2
        { x: 0, y: 4, xStep: 4, yStep: 8 }, // Pass 3
        { x: 2, y: 0, xStep: 4, yStep: 4 }, // Pass 4
        { x: 0, y: 2, xStep: 2, yStep: 4 }, // Pass 5
        { x: 1, y: 0, xStep: 2, yStep: 2 }, // Pass 6
        { x: 0, y: 1, xStep: 1, yStep: 2 }, // Pass 7
    ];
    const bytesPerPixel = Math.ceil(depth / 8) * channels;
    const resultData = new Uint8Array(height * width * bytesPerPixel);
    let offset = 0;
    // Process each pass
    for (let passIndex = 0; passIndex < 7; passIndex++) {
        const pass = passes[passIndex];
        // Calculate pass dimensions
        const passWidth = Math.ceil((width - pass.x) / pass.xStep);
        const passHeight = Math.ceil((height - pass.y) / pass.yStep);
        if (passWidth <= 0 || passHeight <= 0)
            continue;
        const passLineBytes = passWidth * bytesPerPixel;
        const prevLine = new Uint8Array(passLineBytes);
        // Process each scanline in this pass
        for (let y = 0; y < passHeight; y++) {
            // First byte is the filter type
            const filterType = data[offset++];
            const currentLine = data.subarray(offset, offset + passLineBytes);
            offset += passLineBytes;
            // Create a new line for the unfiltered data
            const newLine = new Uint8Array(passLineBytes);
            // Apply the appropriate unfilter
            applyUnfilter(filterType, currentLine, newLine, prevLine, passLineBytes, bytesPerPixel);
            prevLine.set(newLine);
            for (let x = 0; x < passWidth; x++) {
                const outputX = pass.x + x * pass.xStep;
                const outputY = pass.y + y * pass.yStep;
                if (outputX >= width || outputY >= height)
                    continue;
                for (let i = 0; i < bytesPerPixel; i++) {
                    resultData[(outputY * width + outputX) * bytesPerPixel + i] =
                        newLine[x * bytesPerPixel + i];
                }
            }
        }
    }
    if (depth === 16) {
        const uint16Data = new Uint16Array(resultData.buffer);
        if (osIsLittleEndian$1) {
            for (let k = 0; k < uint16Data.length; k++) {
                // PNG is always big endian. Swap the bytes.
                uint16Data[k] = swap16$1(uint16Data[k]);
            }
        }
        return uint16Data;
    }
    else {
        return resultData;
    }
}
function swap16$1(val) {
    return ((val & 0xff) << 8) | ((val >> 8) & 0xff);
}

const uint16 = new Uint16Array([0x00ff]);
const uint8 = new Uint8Array(uint16.buffer);
const osIsLittleEndian = uint8[0] === 0xff;
const empty = new Uint8Array(0);
function decodeInterlaceNull(params) {
    const { data, width, height, channels, depth } = params;
    const bytesPerPixel = Math.ceil(depth / 8) * channels;
    const bytesPerLine = Math.ceil((depth / 8) * channels * width);
    const newData = new Uint8Array(height * bytesPerLine);
    let prevLine = empty;
    let offset = 0;
    let currentLine;
    let newLine;
    for (let i = 0; i < height; i++) {
        currentLine = data.subarray(offset + 1, offset + 1 + bytesPerLine);
        newLine = newData.subarray(i * bytesPerLine, (i + 1) * bytesPerLine);
        switch (data[offset]) {
            case 0:
                unfilterNone(currentLine, newLine, bytesPerLine);
                break;
            case 1:
                unfilterSub(currentLine, newLine, bytesPerLine, bytesPerPixel);
                break;
            case 2:
                unfilterUp(currentLine, newLine, prevLine, bytesPerLine);
                break;
            case 3:
                unfilterAverage(currentLine, newLine, prevLine, bytesPerLine, bytesPerPixel);
                break;
            case 4:
                unfilterPaeth(currentLine, newLine, prevLine, bytesPerLine, bytesPerPixel);
                break;
            default:
                throw new Error(`Unsupported filter: ${data[offset]}`);
        }
        prevLine = newLine;
        offset += bytesPerLine + 1;
    }
    if (depth === 16) {
        const uint16Data = new Uint16Array(newData.buffer);
        if (osIsLittleEndian) {
            for (let k = 0; k < uint16Data.length; k++) {
                // PNG is always big endian. Swap the bytes.
                uint16Data[k] = swap16(uint16Data[k]);
            }
        }
        return uint16Data;
    }
    else {
        return newData;
    }
}
function swap16(val) {
    return ((val & 0xff) << 8) | ((val >> 8) & 0xff);
}

// https://www.w3.org/TR/PNG/#5PNG-file-signature
const pngSignature = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
function checkSignature(buffer) {
    if (!hasPngSignature(buffer.readBytes(pngSignature.length))) {
        throw new Error('wrong PNG signature');
    }
}
function hasPngSignature(array) {
    if (array.length < pngSignature.length) {
        return false;
    }
    for (let i = 0; i < pngSignature.length; i++) {
        if (array[i] !== pngSignature[i]) {
            return false;
        }
    }
    return true;
}

// https://www.w3.org/TR/png/#11tEXt
const textChunkName = 'tEXt';
const NULL = 0;
const latin1Decoder = new TextDecoder('latin1');
function validateKeyword(keyword) {
    validateLatin1(keyword);
    if (keyword.length === 0 || keyword.length > 79) {
        throw new Error('keyword length must be between 1 and 79');
    }
}
// eslint-disable-next-line no-control-regex
const latin1Regex = /^[\u0000-\u00FF]*$/;
function validateLatin1(text) {
    if (!latin1Regex.test(text)) {
        throw new Error('invalid latin1 text');
    }
}
function decodetEXt(text, buffer, length) {
    const keyword = readKeyword(buffer);
    text[keyword] = readLatin1(buffer, length - keyword.length - 1);
}
// https://www.w3.org/TR/png/#11keywords
function readKeyword(buffer) {
    buffer.mark();
    while (buffer.readByte() !== NULL) {
        /* advance */
    }
    const end = buffer.offset;
    buffer.reset();
    const keyword = latin1Decoder.decode(buffer.readBytes(end - buffer.offset - 1));
    // NULL
    buffer.skip(1);
    validateKeyword(keyword);
    return keyword;
}
function readLatin1(buffer, length) {
    return latin1Decoder.decode(buffer.readBytes(length));
}

const ColorType = {
    UNKNOWN: -1,
    GREYSCALE: 0,
    TRUECOLOUR: 2,
    INDEXED_COLOUR: 3,
    GREYSCALE_ALPHA: 4,
    TRUECOLOUR_ALPHA: 6,
};
const CompressionMethod = {
    UNKNOWN: -1,
    DEFLATE: 0,
};
const FilterMethod = {
    UNKNOWN: -1,
    ADAPTIVE: 0,
};
const InterlaceMethod = {
    UNKNOWN: -1,
    NO_INTERLACE: 0,
    ADAM7: 1,
};
const DisposeOpType = {
    NONE: 0,
    BACKGROUND: 1,
    PREVIOUS: 2,
};
const BlendOpType = {
    SOURCE: 0,
    OVER: 1,
};

class PngDecoder extends IOBuffer {
    _checkCrc;
    _inflator;
    _png;
    _apng;
    _end;
    _hasPalette;
    _palette;
    _hasTransparency;
    _transparency;
    _compressionMethod;
    _filterMethod;
    _interlaceMethod;
    _colorType;
    _isAnimated;
    _numberOfFrames;
    _numberOfPlays;
    _frames;
    _writingDataChunks;
    constructor(data, options = {}) {
        super(data);
        const { checkCrc = false } = options;
        this._checkCrc = checkCrc;
        this._inflator = new Inflate_1();
        this._png = {
            width: -1,
            height: -1,
            channels: -1,
            data: new Uint8Array(0),
            depth: 1,
            text: {},
        };
        this._apng = {
            width: -1,
            height: -1,
            channels: -1,
            depth: 1,
            numberOfFrames: 1,
            numberOfPlays: 0,
            text: {},
            frames: [],
        };
        this._end = false;
        this._hasPalette = false;
        this._palette = [];
        this._hasTransparency = false;
        this._transparency = new Uint16Array(0);
        this._compressionMethod = CompressionMethod.UNKNOWN;
        this._filterMethod = FilterMethod.UNKNOWN;
        this._interlaceMethod = InterlaceMethod.UNKNOWN;
        this._colorType = ColorType.UNKNOWN;
        this._isAnimated = false;
        this._numberOfFrames = 1;
        this._numberOfPlays = 0;
        this._frames = [];
        this._writingDataChunks = false;
        // PNG is always big endian
        // https://www.w3.org/TR/PNG/#7Integers-and-byte-order
        this.setBigEndian();
    }
    decode() {
        checkSignature(this);
        while (!this._end) {
            const length = this.readUint32();
            const type = this.readChars(4);
            this.decodeChunk(length, type);
        }
        this.decodeImage();
        return this._png;
    }
    decodeApng() {
        checkSignature(this);
        while (!this._end) {
            const length = this.readUint32();
            const type = this.readChars(4);
            this.decodeApngChunk(length, type);
        }
        this.decodeApngImage();
        return this._apng;
    }
    // https://www.w3.org/TR/PNG/#5Chunk-layout
    decodeChunk(length, type) {
        const offset = this.offset;
        switch (type) {
            // 11.2 Critical chunks
            case 'IHDR': // 11.2.2 IHDR Image header
                this.decodeIHDR();
                break;
            case 'PLTE': // 11.2.3 PLTE Palette
                this.decodePLTE(length);
                break;
            case 'IDAT': // 11.2.4 IDAT Image data
                this.decodeIDAT(length);
                break;
            case 'IEND': // 11.2.5 IEND Image trailer
                this._end = true;
                break;
            // 11.3 Ancillary chunks
            case 'tRNS': // 11.3.2.1 tRNS Transparency
                this.decodetRNS(length);
                break;
            case 'iCCP': // 11.3.3.3 iCCP Embedded ICC profile
                this.decodeiCCP(length);
                break;
            case textChunkName: // 11.3.4.3 tEXt Textual data
                decodetEXt(this._png.text, this, length);
                break;
            case 'pHYs': // 11.3.5.3 pHYs Physical pixel dimensions
                this.decodepHYs();
                break;
            default:
                this.skip(length);
                break;
        }
        if (this.offset - offset !== length) {
            throw new Error(`Length mismatch while decoding chunk ${type}`);
        }
        if (this._checkCrc) {
            checkCrc(this, length + 4, type);
        }
        else {
            this.skip(4);
        }
    }
    decodeApngChunk(length, type) {
        const offset = this.offset;
        if (type !== 'fdAT' && type !== 'IDAT' && this._writingDataChunks) {
            this.pushDataToFrame();
        }
        switch (type) {
            case 'acTL':
                this.decodeACTL();
                break;
            case 'fcTL':
                this.decodeFCTL();
                break;
            case 'fdAT':
                this.decodeFDAT(length);
                break;
            default:
                this.decodeChunk(length, type);
                this.offset = offset + length;
                break;
        }
        if (this.offset - offset !== length) {
            throw new Error(`Length mismatch while decoding chunk ${type}`);
        }
        if (this._checkCrc) {
            checkCrc(this, length + 4, type);
        }
        else {
            this.skip(4);
        }
    }
    // https://www.w3.org/TR/PNG/#11IHDR
    decodeIHDR() {
        const image = this._png;
        image.width = this.readUint32();
        image.height = this.readUint32();
        image.depth = checkBitDepth(this.readUint8());
        const colorType = this.readUint8();
        this._colorType = colorType;
        let channels;
        switch (colorType) {
            case ColorType.GREYSCALE:
                channels = 1;
                break;
            case ColorType.TRUECOLOUR:
                channels = 3;
                break;
            case ColorType.INDEXED_COLOUR:
                channels = 1;
                break;
            case ColorType.GREYSCALE_ALPHA:
                channels = 2;
                break;
            case ColorType.TRUECOLOUR_ALPHA:
                channels = 4;
                break;
            // Kept for exhaustiveness.
            // eslint-disable-next-line unicorn/no-useless-switch-case
            case ColorType.UNKNOWN:
            default:
                throw new Error(`Unknown color type: ${colorType}`);
        }
        this._png.channels = channels;
        this._compressionMethod = this.readUint8();
        if (this._compressionMethod !== CompressionMethod.DEFLATE) {
            throw new Error(`Unsupported compression method: ${this._compressionMethod}`);
        }
        this._filterMethod = this.readUint8();
        this._interlaceMethod = this.readUint8();
    }
    decodeACTL() {
        this._numberOfFrames = this.readUint32();
        this._numberOfPlays = this.readUint32();
        this._isAnimated = true;
    }
    decodeFCTL() {
        const image = {
            sequenceNumber: this.readUint32(),
            width: this.readUint32(),
            height: this.readUint32(),
            xOffset: this.readUint32(),
            yOffset: this.readUint32(),
            delayNumber: this.readUint16(),
            delayDenominator: this.readUint16(),
            disposeOp: this.readUint8(),
            blendOp: this.readUint8(),
            data: new Uint8Array(0),
        };
        this._frames.push(image);
    }
    // https://www.w3.org/TR/PNG/#11PLTE
    decodePLTE(length) {
        if (length % 3 !== 0) {
            throw new RangeError(`PLTE field length must be a multiple of 3. Got ${length}`);
        }
        const l = length / 3;
        this._hasPalette = true;
        const palette = [];
        this._palette = palette;
        for (let i = 0; i < l; i++) {
            palette.push([this.readUint8(), this.readUint8(), this.readUint8()]);
        }
    }
    // https://www.w3.org/TR/PNG/#11IDAT
    decodeIDAT(length) {
        this._writingDataChunks = true;
        const dataLength = length;
        const dataOffset = this.offset + this.byteOffset;
        this._inflator.push(new Uint8Array(this.buffer, dataOffset, dataLength));
        if (this._inflator.err) {
            throw new Error(`Error while decompressing the data: ${this._inflator.err}`);
        }
        this.skip(length);
    }
    decodeFDAT(length) {
        this._writingDataChunks = true;
        let dataLength = length;
        let dataOffset = this.offset + this.byteOffset;
        dataOffset += 4;
        dataLength -= 4;
        this._inflator.push(new Uint8Array(this.buffer, dataOffset, dataLength));
        if (this._inflator.err) {
            throw new Error(`Error while decompressing the data: ${this._inflator.err}`);
        }
        this.skip(length);
    }
    // https://www.w3.org/TR/PNG/#11tRNS
    decodetRNS(length) {
        switch (this._colorType) {
            case ColorType.GREYSCALE:
            case ColorType.TRUECOLOUR: {
                if (length % 2 !== 0) {
                    throw new RangeError(`tRNS chunk length must be a multiple of 2. Got ${length}`);
                }
                if (length / 2 > this._png.width * this._png.height) {
                    throw new Error(`tRNS chunk contains more alpha values than there are pixels (${length / 2} vs ${this._png.width * this._png.height})`);
                }
                this._hasTransparency = true;
                this._transparency = new Uint16Array(length / 2);
                for (let i = 0; i < length / 2; i++) {
                    this._transparency[i] = this.readUint16();
                }
                break;
            }
            case ColorType.INDEXED_COLOUR: {
                if (length > this._palette.length) {
                    throw new Error(`tRNS chunk contains more alpha values than there are palette colors (${length} vs ${this._palette.length})`);
                }
                let i = 0;
                for (; i < length; i++) {
                    const alpha = this.readByte();
                    this._palette[i].push(alpha);
                }
                for (; i < this._palette.length; i++) {
                    this._palette[i].push(255);
                }
                break;
            }
            // Kept for exhaustiveness.
            /* eslint-disable unicorn/no-useless-switch-case */
            case ColorType.UNKNOWN:
            case ColorType.GREYSCALE_ALPHA:
            case ColorType.TRUECOLOUR_ALPHA:
            default: {
                throw new Error(`tRNS chunk is not supported for color type ${this._colorType}`);
            }
            /* eslint-enable unicorn/no-useless-switch-case */
        }
    }
    // https://www.w3.org/TR/PNG/#11iCCP
    decodeiCCP(length) {
        const name = readKeyword(this);
        const compressionMethod = this.readUint8();
        if (compressionMethod !== CompressionMethod.DEFLATE) {
            throw new Error(`Unsupported iCCP compression method: ${compressionMethod}`);
        }
        const compressedProfile = this.readBytes(length - name.length - 2);
        this._png.iccEmbeddedProfile = {
            name,
            profile: inflate_1(compressedProfile),
        };
    }
    // https://www.w3.org/TR/PNG/#11pHYs
    decodepHYs() {
        const ppuX = this.readUint32();
        const ppuY = this.readUint32();
        const unitSpecifier = this.readByte();
        this._png.resolution = { x: ppuX, y: ppuY, unit: unitSpecifier };
    }
    decodeApngImage() {
        this._apng.width = this._png.width;
        this._apng.height = this._png.height;
        this._apng.channels = this._png.channels;
        this._apng.depth = this._png.depth;
        this._apng.numberOfFrames = this._numberOfFrames;
        this._apng.numberOfPlays = this._numberOfPlays;
        this._apng.text = this._png.text;
        this._apng.resolution = this._png.resolution;
        for (let i = 0; i < this._numberOfFrames; i++) {
            const newFrame = {
                sequenceNumber: this._frames[i].sequenceNumber,
                delayNumber: this._frames[i].delayNumber,
                delayDenominator: this._frames[i].delayDenominator,
                data: this._apng.depth === 8
                    ? new Uint8Array(this._apng.width * this._apng.height * this._apng.channels)
                    : new Uint16Array(this._apng.width * this._apng.height * this._apng.channels),
            };
            const frame = this._frames.at(i);
            if (frame) {
                frame.data = decodeInterlaceNull({
                    data: frame.data,
                    width: frame.width,
                    height: frame.height,
                    channels: this._apng.channels,
                    depth: this._apng.depth,
                });
                if (this._hasPalette) {
                    this._apng.palette = this._palette;
                }
                if (this._hasTransparency) {
                    this._apng.transparency = this._transparency;
                }
                if (i === 0 ||
                    (frame.xOffset === 0 &&
                        frame.yOffset === 0 &&
                        frame.width === this._png.width &&
                        frame.height === this._png.height)) {
                    newFrame.data = frame.data;
                }
                else {
                    const prevFrame = this._apng.frames.at(i - 1);
                    this.disposeFrame(frame, prevFrame, newFrame);
                    this.addFrameDataToCanvas(newFrame, frame);
                }
                this._apng.frames.push(newFrame);
            }
        }
        return this._apng;
    }
    disposeFrame(frame, prevFrame, imageFrame) {
        switch (frame.disposeOp) {
            case DisposeOpType.NONE:
                break;
            case DisposeOpType.BACKGROUND:
                for (let row = 0; row < this._png.height; row++) {
                    for (let col = 0; col < this._png.width; col++) {
                        const index = (row * frame.width + col) * this._png.channels;
                        for (let channel = 0; channel < this._png.channels; channel++) {
                            imageFrame.data[index + channel] = 0;
                        }
                    }
                }
                break;
            case DisposeOpType.PREVIOUS:
                imageFrame.data.set(prevFrame.data);
                break;
            default:
                throw new Error('Unknown disposeOp');
        }
    }
    addFrameDataToCanvas(imageFrame, frame) {
        const maxValue = 1 << this._png.depth;
        const calculatePixelIndices = (row, col) => {
            const index = ((row + frame.yOffset) * this._png.width + frame.xOffset + col) *
                this._png.channels;
            const frameIndex = (row * frame.width + col) * this._png.channels;
            return { index, frameIndex };
        };
        switch (frame.blendOp) {
            case BlendOpType.SOURCE:
                for (let row = 0; row < frame.height; row++) {
                    for (let col = 0; col < frame.width; col++) {
                        const { index, frameIndex } = calculatePixelIndices(row, col);
                        for (let channel = 0; channel < this._png.channels; channel++) {
                            imageFrame.data[index + channel] =
                                frame.data[frameIndex + channel];
                        }
                    }
                }
                break;
            // https://www.w3.org/TR/png-3/#13Alpha-channel-processing
            case BlendOpType.OVER:
                for (let row = 0; row < frame.height; row++) {
                    for (let col = 0; col < frame.width; col++) {
                        const { index, frameIndex } = calculatePixelIndices(row, col);
                        for (let channel = 0; channel < this._png.channels; channel++) {
                            const sourceAlpha = frame.data[frameIndex + this._png.channels - 1] / maxValue;
                            const foregroundValue = channel % (this._png.channels - 1) === 0
                                ? 1
                                : frame.data[frameIndex + channel];
                            const value = Math.floor(sourceAlpha * foregroundValue +
                                (1 - sourceAlpha) * imageFrame.data[index + channel]);
                            imageFrame.data[index + channel] += value;
                        }
                    }
                }
                break;
            default:
                throw new Error('Unknown blendOp');
        }
    }
    decodeImage() {
        if (this._inflator.err) {
            throw new Error(`Error while decompressing the data: ${this._inflator.err}`);
        }
        const data = this._isAnimated
            ? (this._frames?.at(0)).data
            : this._inflator.result;
        if (this._filterMethod !== FilterMethod.ADAPTIVE) {
            throw new Error(`Filter method ${this._filterMethod} not supported`);
        }
        if (this._interlaceMethod === InterlaceMethod.NO_INTERLACE) {
            this._png.data = decodeInterlaceNull({
                data: data,
                width: this._png.width,
                height: this._png.height,
                channels: this._png.channels,
                depth: this._png.depth,
            });
        }
        else if (this._interlaceMethod === InterlaceMethod.ADAM7) {
            this._png.data = decodeInterlaceAdam7({
                data: data,
                width: this._png.width,
                height: this._png.height,
                channels: this._png.channels,
                depth: this._png.depth,
            });
        }
        else {
            throw new Error(`Interlace method ${this._interlaceMethod} not supported`);
        }
        if (this._hasPalette) {
            this._png.palette = this._palette;
        }
        if (this._hasTransparency) {
            this._png.transparency = this._transparency;
        }
    }
    pushDataToFrame() {
        const result = this._inflator.result;
        const lastFrame = this._frames.at(-1);
        if (lastFrame) {
            lastFrame.data = result;
        }
        else {
            this._frames.push({
                sequenceNumber: 0,
                width: this._png.width,
                height: this._png.height,
                xOffset: 0,
                yOffset: 0,
                delayNumber: 0,
                delayDenominator: 0,
                disposeOp: DisposeOpType.NONE,
                blendOp: BlendOpType.SOURCE,
                data: result,
            });
        }
        this._inflator = new Inflate_1();
        this._writingDataChunks = false;
    }
}
function checkBitDepth(value) {
    if (value !== 1 &&
        value !== 2 &&
        value !== 4 &&
        value !== 8 &&
        value !== 16) {
        throw new Error(`invalid bit depth: ${value}`);
    }
    return value;
}

var ResolutionUnitSpecifier;
(function (ResolutionUnitSpecifier) {
    /**
     * Unit is unknown
     */
    ResolutionUnitSpecifier[ResolutionUnitSpecifier["UNKNOWN"] = 0] = "UNKNOWN";
    /**
     * Unit is the metre
     */
    ResolutionUnitSpecifier[ResolutionUnitSpecifier["METRE"] = 1] = "METRE";
})(ResolutionUnitSpecifier || (ResolutionUnitSpecifier = {}));

function decodePng(data, options) {
    const decoder = new PngDecoder(data, options);
    return decoder.decode();
}

/**
 * @license
 *
 * Copyright (c) 2014 James Robb, https://github.com/jamesbrobb
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/*
 * @see http://www.w3.org/TR/PNG-Chunks.html
 *
 Color    Allowed      Interpretation
 Type     Bit Depths

   0       1,2,4,8,16  Each pixel is a grayscale sample.

   2       8,16        Each pixel is an R,G,B triple.

   3       1,2,4,8     Each pixel is a palette index;
                       a PLTE chunk must appear.

   4       8,16        Each pixel is a grayscale sample,
                       followed by an alpha sample.

   6       8,16        Each pixel is an R,G,B triple,
                       followed by an alpha sample.
*/

/*
 * @name processPNG
 * Entry point: process a PNG and return image dict and metadata for jsPDF
 */
jsPDF.API.processPNG = function(imageData, index, alias, compression) {
  if (this.__addimage__.isArrayBuffer(imageData)) {
    imageData = new Uint8Array(imageData);
  }
  if (!this.__addimage__.isArrayBufferView(imageData)) {
    return;
  }

  const decodedPng = decodePng(imageData, { checkCrc: true });
  const {
    width,
    height,
    channels,
    palette: decodedPalette,
    depth: bitsPerComponent
  } = decodedPng;

  let result;
  if (decodedPalette && channels === 1) {
    result = processIndexedPNG(decodedPng);
  } else if (channels === 2 || channels === 4) {
    result = processAlphaPNG(decodedPng);
  } else {
    result = processOpaquePNG(decodedPng);
  }

  const {
    colorSpace,
    colorsPerPixel,
    sMaskBitsPerComponent,
    colorBytes,
    alphaBytes,
    needSMask,
    palette,
    mask
  } = result;

  let predictor = null;

  let filter, decodeParameters, sMask;
  if (canCompress(compression)) {
    predictor = getPredictorFromCompression(compression);
    filter = this.decode.FLATE_DECODE;
    decodeParameters = `/Predictor ${predictor} /Colors ${colorsPerPixel} /BitsPerComponent ${bitsPerComponent} /Columns ${width}`;

    const rowByteLength = Math.ceil(
      (width * colorsPerPixel * bitsPerComponent) / 8
    );

    imageData = compressBytes(
      colorBytes,
      rowByteLength,
      colorsPerPixel,
      bitsPerComponent,
      compression
    );
    if (needSMask) {
      const sMaskRowByteLength = Math.ceil((width * sMaskBitsPerComponent) / 8);
      sMask = compressBytes(
        alphaBytes,
        sMaskRowByteLength,
        1,
        sMaskBitsPerComponent,
        compression
      );
    }
  } else {
    filter = undefined;
    decodeParameters = undefined;
    imageData = colorBytes;
    if (needSMask) sMask = alphaBytes;
  }

  if (
    this.__addimage__.isArrayBuffer(imageData) ||
    this.__addimage__.isArrayBufferView(imageData)
  ) {
    imageData = this.__addimage__.arrayBufferToBinaryString(imageData);
  }

  if (
    (sMask && this.__addimage__.isArrayBuffer(sMask)) ||
    this.__addimage__.isArrayBufferView(sMask)
  ) {
    sMask = this.__addimage__.arrayBufferToBinaryString(sMask);
  }

  return {
    alias,
    data: imageData,
    index,
    filter,
    decodeParameters,
    transparency: mask,
    palette,
    sMask,
    predictor,
    width,
    height,
    bitsPerComponent,
    sMaskBitsPerComponent,
    colorSpace
  };
};

/*
   * PNG filter method types
   *
   * @see http://www.w3.org/TR/PNG-Filters.html
   * @see http://www.libpng.org/pub/png/book/chapter09.html
   *
   * This is what the value 'Predictor' in decode params relates to
   *
   * 15 is "optimal prediction", which means the prediction algorithm can change from line to line.
   * In that case, you actually have to read the first byte off each line for the prediction algorthim (which should be 0-4, corresponding to PDF 10-14) and select the appropriate unprediction algorithm based on that byte.
   *
     0       None
     1       Sub
     2       Up
     3       Average
     4       Paeth
   */

function canCompress(value) {
  return value !== jsPDF.API.image_compression.NONE && hasCompressionJS();
}

function hasCompressionJS() {
  return typeof zlibSync === "function";
}
function compressBytes(
  bytes,
  lineByteLength,
  channels,
  bitsPerComponent,
  compression
) {
  let level = 4;
  let filter_method = filterUp;

  switch (compression) {
    case jsPDF.API.image_compression.FAST:
      level = 1;
      filter_method = filterSub;
      break;

    case jsPDF.API.image_compression.MEDIUM:
      level = 6;
      filter_method = filterAverage;
      break;

    case jsPDF.API.image_compression.SLOW:
      level = 9;
      filter_method = filterPaeth;
      break;
  }

  const bytesPerPixel = Math.ceil((channels * bitsPerComponent) / 8);
  bytes = applyPngFilterMethod(
    bytes,
    lineByteLength,
    bytesPerPixel,
    filter_method
  );
  const dat = zlibSync(bytes, { level: level });
  return jsPDF.API.__addimage__.arrayBufferToBinaryString(dat);
}

function applyPngFilterMethod(
  bytes,
  lineByteLength,
  bytesPerPixel,
  filter_method
) {
  const lines = bytes.length / lineByteLength;
  const result = new Uint8Array(bytes.length + lines);
  const filter_methods = getFilterMethods();
  let prevLine;

  for (let i = 0; i < lines; i += 1) {
    const offset = i * lineByteLength;
    const line = bytes.subarray(offset, offset + lineByteLength);

    if (filter_method) {
      result.set(filter_method(line, bytesPerPixel, prevLine), offset + i);
    } else {
      const len = filter_methods.length;
      const results = [];

      for (let j = 0; j < len; j += 1) {
        results[j] = filter_methods[j](line, bytesPerPixel, prevLine);
      }

      const ind = getIndexOfSmallestSum(results.concat());

      result.set(results[ind], offset + i);
    }

    prevLine = line;
  }

  return result;
}

function filterNone(line) {
  /*const result = new Uint8Array(line.length + 1);
    result[0] = 0;
    result.set(line, 1);*/

  const result = Array.apply([], line);
  result.unshift(0);

  return result;
}

function filterSub(line, colorsPerPixel) {
  const len = line.length;
  const result = [];

  result[0] = 1;

  for (let i = 0; i < len; i += 1) {
    const left = line[i - colorsPerPixel] || 0;
    result[i + 1] = (line[i] - left + 0x0100) & 0xff;
  }

  return result;
}

function filterUp(line, colorsPerPixel, prevLine) {
  const len = line.length;
  const result = [];

  result[0] = 2;

  for (let i = 0; i < len; i += 1) {
    const up = (prevLine && prevLine[i]) || 0;
    result[i + 1] = (line[i] - up + 0x0100) & 0xff;
  }

  return result;
}

function filterAverage(line, colorsPerPixel, prevLine) {
  const len = line.length;
  const result = [];

  result[0] = 3;

  for (let i = 0; i < len; i += 1) {
    const left = line[i - colorsPerPixel] || 0;
    const up = (prevLine && prevLine[i]) || 0;
    result[i + 1] = (line[i] + 0x0100 - ((left + up) >>> 1)) & 0xff;
  }

  return result;
}

function filterPaeth(line, colorsPerPixel, prevLine) {
  const len = line.length;
  const result = [];

  result[0] = 4;

  for (let i = 0; i < len; i += 1) {
    const left = line[i - colorsPerPixel] || 0;
    const up = (prevLine && prevLine[i]) || 0;
    const upLeft = (prevLine && prevLine[i - colorsPerPixel]) || 0;
    const paeth = paethPredictor(left, up, upLeft);
    result[i + 1] = (line[i] - paeth + 0x0100) & 0xff;
  }

  return result;
}

function paethPredictor(left, up, upLeft) {
  if (left === up && up === upLeft) {
    return left;
  }
  const pLeft = Math.abs(up - upLeft),
    pUp = Math.abs(left - upLeft),
    pUpLeft = Math.abs(left + up - upLeft - upLeft);
  return pLeft <= pUp && pLeft <= pUpLeft ? left : pUp <= pUpLeft ? up : upLeft;
}

function getFilterMethods() {
  return [filterNone, filterSub, filterUp, filterAverage, filterPaeth];
}

function getIndexOfSmallestSum(arrays) {
  const sum = arrays.map(function(value) {
    return value.reduce(function(pv, cv) {
      return pv + Math.abs(cv);
    }, 0);
  });
  return sum.indexOf(Math.min.apply(null, sum));
}

function getPredictorFromCompression(compression) {
  let predictor;
  switch (compression) {
    case jsPDF.API.image_compression.FAST:
      predictor = 11;
      break;

    case jsPDF.API.image_compression.MEDIUM:
      predictor = 13;
      break;

    case jsPDF.API.image_compression.SLOW:
      predictor = 14;
      break;

    default:
      predictor = 12;
      break;
  }
  return predictor;
}

// Extracted helper for Indexed PNGs (palette-based)
function processIndexedPNG(decodedPng) {
  const { width, height, data, palette: decodedPalette, depth } = decodedPng;
  let needSMask = false;
  let palette = [];
  let mask = [];
  let alphaBytes = undefined;
  let hasSemiTransparency = false;

  const maxMaskLength = 1;
  let maskLength = 0;

  for (let i = 0; i < decodedPalette.length; i++) {
    const [r, g, b, a] = decodedPalette[i];
    palette.push(r, g, b);
    if (a != null) {
      if (a === 0) {
        maskLength++;
        if (mask.length < maxMaskLength) {
          mask.push(i);
        }
      } else if (a < 255) {
        hasSemiTransparency = true;
      }
    }
  }

  if (hasSemiTransparency || maskLength > maxMaskLength) {
    needSMask = true;
    mask = undefined;

    const totalPixels = width * height;
    // per PNG spec, palettes always use 8 bits per component
    alphaBytes = new Uint8Array(totalPixels);
    const dataView = new DataView(data.buffer);
    for (let p = 0; p < totalPixels; p++) {
      const paletteIndex = readSample(dataView, p, depth);
      const [, , , alpha] = decodedPalette[paletteIndex];
      alphaBytes[p] = alpha;
    }
  } else if (maskLength === 0) {
    mask = undefined;
  }

  return {
    colorSpace: "Indexed",
    colorsPerPixel: 1,
    sMaskBitsPerComponent: needSMask ? 8 : undefined,
    colorBytes: data,
    alphaBytes,
    needSMask,
    palette,
    mask
  };
}

/*
 * Splits color and alpha values into separate buffers
 */
function processAlphaPNG(decodedPng) {
  const { data, width, height, channels, depth } = decodedPng;

  const colorSpace = channels === 2 ? "DeviceGray" : "DeviceRGB";
  const colorsPerPixel = channels - 1;

  const totalPixels = width * height;
  const colorChannels = colorsPerPixel; // 1 for Gray, 3 for RGB
  const alphaChannels = 1;
  const totalColorSamples = totalPixels * colorChannels;
  const totalAlphaSamples = totalPixels * alphaChannels;

  const colorByteLen = Math.ceil((totalColorSamples * depth) / 8);
  const alphaByteLen = Math.ceil((totalAlphaSamples * depth) / 8);
  const colorBytes = new Uint8Array(colorByteLen);
  const alphaBytes = new Uint8Array(alphaByteLen);

  const dataView = new DataView(data.buffer);
  const colorView = new DataView(colorBytes.buffer);
  const alphaView = new DataView(alphaBytes.buffer);

  let needSMask = false;
  for (let p = 0; p < totalPixels; p++) {
    const pixelStartIndex = p * channels;
    for (let s = 0; s < colorChannels; s++) {
      const sampleIndex = pixelStartIndex + s;
      const colorValue = readSample(dataView, sampleIndex, depth);
      writeSample(colorView, colorValue, p * colorChannels + s, depth);
    }
    const sampleIndex = pixelStartIndex + colorChannels;
    const alphaValue = readSample(dataView, sampleIndex, depth);
    if (alphaValue < (1 << depth) - 1) {
      needSMask = true;
    }
    writeSample(alphaView, alphaValue, p * alphaChannels, depth);
  }

  return {
    colorSpace,
    colorsPerPixel,
    sMaskBitsPerComponent: needSMask ? depth : undefined,
    colorBytes,
    alphaBytes,
    needSMask
  };
}

function processOpaquePNG(decodedPng) {
  const { data, channels } = decodedPng;
  const colorSpace = channels === 1 ? "DeviceGray" : "DeviceRGB";
  const colorsPerPixel = colorSpace === "DeviceGray" ? 1 : 3;

  let colorBytes;
  if (data instanceof Uint16Array) {
    colorBytes = convertUint16ArrayToUint8Array(data);
  } else {
    colorBytes = data;
  }

  return { colorSpace, colorsPerPixel, colorBytes, needSMask: false };
}

function convertUint16ArrayToUint8Array(data) {
  // PNG/PDF expect MSB-first byte order. Since EcmaScript does not specify
  // the byte order of Uint16Array, we need to use a DataView to ensure the
  // correct byte order.
  const sampleCount = data.length;
  const out = new Uint8Array(sampleCount * 2);
  const outView = new DataView(out.buffer, out.byteOffset, out.byteLength);

  for (let i = 0; i < sampleCount; i++) {
    outView.setUint16(i * 2, data[i], false);
  }
  return out;
}

function readSample(view, sampleIndex, depth) {
  const bitIndex = sampleIndex * depth;
  const byteIndex = Math.floor(bitIndex / 8);
  const bitOffset = 16 - (bitIndex - byteIndex * 8 + depth);
  const bitMask = (1 << depth) - 1;
  const word = safeGetUint16(view, byteIndex);
  return (word >> bitOffset) & bitMask;
}

function writeSample(view, value, sampleIndex, depth) {
  const bitIndex = sampleIndex * depth;
  const byteIndex = Math.floor(bitIndex / 8);
  const bitOffset = 16 - (bitIndex - byteIndex * 8 + depth);
  const bitMask = (1 << depth) - 1;
  const writeValue = (value & bitMask) << bitOffset;
  const word =
    safeGetUint16(view, byteIndex) & ~(bitMask << bitOffset) & 0xffff;
  safeSetUint16(view, byteIndex, word | writeValue);
}

function safeGetUint16(view, byteIndex) {
  if (byteIndex + 1 < view.byteLength) {
    return view.getUint16(byteIndex, false);
  }
  const b0 = view.getUint8(byteIndex);
  return b0 << 8;
}

function safeSetUint16(view, byteIndex, value) {
  if (byteIndex + 1 < view.byteLength) {
    view.setUint16(byteIndex, value, false);
    return;
  }
  const byteToWrite = (value >> 8) & 0xff;
  view.setUint8(byteIndex, byteToWrite);
}

/**
 * @license
 *
 * Copyright (c) 2021 Antti Palola, https://github.com/Pantura
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * jsPDF RGBA array PlugIn
 * @name rgba_support
 * @module
 */
(function(jsPDFAPI) {

  /**
   * @name processRGBA
   * @function
   *
   * Process RGBA Array. This is a one-dimension array with pixel data [red, green, blue, alpha, red, green, ...].
   * RGBA array data can be obtained from DOM canvas getImageData.
   * @ignore
   */
  jsPDFAPI.processRGBA = function(imageData, index, alias) {

    var imagePixels = imageData.data;
    var length = imagePixels.length;
    // jsPDF takes alpha data separately so extract that.
    var rgbOut = new Uint8Array((length / 4) * 3);
    var alphaOut = new Uint8Array(length / 4);
    var outIndex = 0;
    var alphaIndex = 0;

    for (var i = 0; i < length; i += 4) {
      var r = imagePixels[i];
      var g = imagePixels[i + 1];
      var b = imagePixels[i + 2];
      var alpha = imagePixels[i + 3];
      rgbOut[outIndex++] = r;
      rgbOut[outIndex++] = g;
      rgbOut[outIndex++] = b;
      alphaOut[alphaIndex++] = alpha;
    }

    var rgbData = this.__addimage__.arrayBufferToBinaryString(rgbOut);
    var alphaData = this.__addimage__.arrayBufferToBinaryString(alphaOut);

    return {
      alpha: alphaData,
      data: rgbData,
      index: index,
      alias: alias,
      colorSpace: "DeviceRGB",
      bitsPerComponent: 8,
      width: imageData.width,
      height: imageData.height
    };
  };
})(jsPDF.API);

/**
 * @license
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * jsPDF setLanguage Plugin
 *
 * @name setLanguage
 * @module
 */
(function(jsPDFAPI) {

  /**
   * Add Language Tag to the generated PDF
   *
   * @name setLanguage
   * @function
   * @param {string} langCode The Language code as ISO-639-1 (e.g. 'en') or as country language code (e.g. 'en-GB').
   * @returns {jsPDF}
   * @example
   * var doc = new jsPDF()
   * doc.text(10, 10, 'This is a test')
   * doc.setLanguage("en-US")
   * doc.save('english.pdf')
   */
  jsPDFAPI.setLanguage = function(langCode) {

    var langCodes = {
      af: "Afrikaans",
      sq: "Albanian",
      ar: "Arabic (Standard)",
      "ar-DZ": "Arabic (Algeria)",
      "ar-BH": "Arabic (Bahrain)",
      "ar-EG": "Arabic (Egypt)",
      "ar-IQ": "Arabic (Iraq)",
      "ar-JO": "Arabic (Jordan)",
      "ar-KW": "Arabic (Kuwait)",
      "ar-LB": "Arabic (Lebanon)",
      "ar-LY": "Arabic (Libya)",
      "ar-MA": "Arabic (Morocco)",
      "ar-OM": "Arabic (Oman)",
      "ar-QA": "Arabic (Qatar)",
      "ar-SA": "Arabic (Saudi Arabia)",
      "ar-SY": "Arabic (Syria)",
      "ar-TN": "Arabic (Tunisia)",
      "ar-AE": "Arabic (U.A.E.)",
      "ar-YE": "Arabic (Yemen)",
      an: "Aragonese",
      hy: "Armenian",
      as: "Assamese",
      ast: "Asturian",
      az: "Azerbaijani",
      eu: "Basque",
      be: "Belarusian",
      bn: "Bengali",
      bs: "Bosnian",
      br: "Breton",
      bg: "Bulgarian",
      my: "Burmese",
      ca: "Catalan",
      ch: "Chamorro",
      ce: "Chechen",
      zh: "Chinese",
      "zh-HK": "Chinese (Hong Kong)",
      "zh-CN": "Chinese (PRC)",
      "zh-SG": "Chinese (Singapore)",
      "zh-TW": "Chinese (Taiwan)",
      cv: "Chuvash",
      co: "Corsican",
      cr: "Cree",
      hr: "Croatian",
      cs: "Czech",
      da: "Danish",
      nl: "Dutch (Standard)",
      "nl-BE": "Dutch (Belgian)",
      en: "English",
      "en-AU": "English (Australia)",
      "en-BZ": "English (Belize)",
      "en-CA": "English (Canada)",
      "en-IE": "English (Ireland)",
      "en-JM": "English (Jamaica)",
      "en-NZ": "English (New Zealand)",
      "en-PH": "English (Philippines)",
      "en-ZA": "English (South Africa)",
      "en-TT": "English (Trinidad & Tobago)",
      "en-GB": "English (United Kingdom)",
      "en-US": "English (United States)",
      "en-ZW": "English (Zimbabwe)",
      eo: "Esperanto",
      et: "Estonian",
      fo: "Faeroese",
      fj: "Fijian",
      fi: "Finnish",
      fr: "French (Standard)",
      "fr-BE": "French (Belgium)",
      "fr-CA": "French (Canada)",
      "fr-FR": "French (France)",
      "fr-LU": "French (Luxembourg)",
      "fr-MC": "French (Monaco)",
      "fr-CH": "French (Switzerland)",
      fy: "Frisian",
      fur: "Friulian",
      gd: "Gaelic (Scots)",
      "gd-IE": "Gaelic (Irish)",
      gl: "Galacian",
      ka: "Georgian",
      de: "German (Standard)",
      "de-AT": "German (Austria)",
      "de-DE": "German (Germany)",
      "de-LI": "German (Liechtenstein)",
      "de-LU": "German (Luxembourg)",
      "de-CH": "German (Switzerland)",
      el: "Greek",
      gu: "Gujurati",
      ht: "Haitian",
      he: "Hebrew",
      hi: "Hindi",
      hu: "Hungarian",
      is: "Icelandic",
      id: "Indonesian",
      iu: "Inuktitut",
      ga: "Irish",
      it: "Italian (Standard)",
      "it-CH": "Italian (Switzerland)",
      ja: "Japanese",
      kn: "Kannada",
      ks: "Kashmiri",
      kk: "Kazakh",
      km: "Khmer",
      ky: "Kirghiz",
      tlh: "Klingon",
      ko: "Korean",
      "ko-KP": "Korean (North Korea)",
      "ko-KR": "Korean (South Korea)",
      la: "Latin",
      lv: "Latvian",
      lt: "Lithuanian",
      lb: "Luxembourgish",
      mk: "North Macedonia",
      ms: "Malay",
      ml: "Malayalam",
      mt: "Maltese",
      mi: "Maori",
      mr: "Marathi",
      mo: "Moldavian",
      nv: "Navajo",
      ng: "Ndonga",
      ne: "Nepali",
      no: "Norwegian",
      nb: "Norwegian (Bokmal)",
      nn: "Norwegian (Nynorsk)",
      oc: "Occitan",
      or: "Oriya",
      om: "Oromo",
      fa: "Persian",
      "fa-IR": "Persian/Iran",
      pl: "Polish",
      pt: "Portuguese",
      "pt-BR": "Portuguese (Brazil)",
      pa: "Punjabi",
      "pa-IN": "Punjabi (India)",
      "pa-PK": "Punjabi (Pakistan)",
      qu: "Quechua",
      rm: "Rhaeto-Romanic",
      ro: "Romanian",
      "ro-MO": "Romanian (Moldavia)",
      ru: "Russian",
      "ru-MO": "Russian (Moldavia)",
      sz: "Sami (Lappish)",
      sg: "Sango",
      sa: "Sanskrit",
      sc: "Sardinian",
      sd: "Sindhi",
      si: "Singhalese",
      sr: "Serbian",
      sk: "Slovak",
      sl: "Slovenian",
      so: "Somani",
      sb: "Sorbian",
      es: "Spanish",
      "es-AR": "Spanish (Argentina)",
      "es-BO": "Spanish (Bolivia)",
      "es-CL": "Spanish (Chile)",
      "es-CO": "Spanish (Colombia)",
      "es-CR": "Spanish (Costa Rica)",
      "es-DO": "Spanish (Dominican Republic)",
      "es-EC": "Spanish (Ecuador)",
      "es-SV": "Spanish (El Salvador)",
      "es-GT": "Spanish (Guatemala)",
      "es-HN": "Spanish (Honduras)",
      "es-MX": "Spanish (Mexico)",
      "es-NI": "Spanish (Nicaragua)",
      "es-PA": "Spanish (Panama)",
      "es-PY": "Spanish (Paraguay)",
      "es-PE": "Spanish (Peru)",
      "es-PR": "Spanish (Puerto Rico)",
      "es-ES": "Spanish (Spain)",
      "es-UY": "Spanish (Uruguay)",
      "es-VE": "Spanish (Venezuela)",
      sx: "Sutu",
      sw: "Swahili",
      sv: "Swedish",
      "sv-FI": "Swedish (Finland)",
      "sv-SV": "Swedish (Sweden)",
      ta: "Tamil",
      tt: "Tatar",
      te: "Teluga",
      th: "Thai",
      tig: "Tigre",
      ts: "Tsonga",
      tn: "Tswana",
      tr: "Turkish",
      tk: "Turkmen",
      uk: "Ukrainian",
      hsb: "Upper Sorbian",
      ur: "Urdu",
      ve: "Venda",
      vi: "Vietnamese",
      vo: "Volapuk",
      wa: "Walloon",
      cy: "Welsh",
      xh: "Xhosa",
      ji: "Yiddish",
      zu: "Zulu"
    };

    if (this.internal.languageSettings === undefined) {
      this.internal.languageSettings = {};
      this.internal.languageSettings.isSubscribed = false;
    }

    if (langCodes[langCode] !== undefined) {
      this.internal.languageSettings.languageCode = langCode;
      if (this.internal.languageSettings.isSubscribed === false) {
        this.internal.events.subscribe("putCatalog", function() {
          this.internal.write(
            "/Lang (" + this.internal.languageSettings.languageCode + ")"
          );
        });
        this.internal.languageSettings.isSubscribed = true;
      }
    }
    return this;
  };
})(jsPDF.API);

/** @license
 * MIT license.
 * Copyright (c) 2012 Willow Systems Corporation, https://github.com/willowsystems
 *               2014 Diego Casorran, https://github.com/diegocr
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * jsPDF split_text_to_size plugin
 *
 * @name split_text_to_size
 * @module
 */
(function(API) {
  /**
   * Returns an array of length matching length of the 'word' string, with each
   * cell occupied by the width of the char in that position.
   *
   * @name getCharWidthsArray
   * @function
   * @param {string} text
   * @param {Object} options
   * @returns {Array}
   */
  var getCharWidthsArray = (API.getCharWidthsArray = function(text, options) {
    options = options || {};

    var activeFont = options.font || this.internal.getFont();
    var fontSize = options.fontSize || this.internal.getFontSize();
    var charSpace = options.charSpace || this.internal.getCharSpace();

    var widths = options.widths
      ? options.widths
      : activeFont.metadata.Unicode.widths;
    var widthsFractionOf = widths.fof ? widths.fof : 1;
    var kerning = options.kerning
      ? options.kerning
      : activeFont.metadata.Unicode.kerning;
    var kerningFractionOf = kerning.fof ? kerning.fof : 1;
    var doKerning = options.doKerning === false ? false : true;
    var kerningValue = 0;

    var i;
    var length = text.length;
    var char_code;
    var prior_char_code = 0; //for kerning
    var default_char_width = widths[0] || widthsFractionOf;
    var output = [];

    for (i = 0; i < length; i++) {
      char_code = text.charCodeAt(i);

      if (typeof activeFont.metadata.widthOfString === "function") {
        output.push(
          (activeFont.metadata.widthOfGlyph(
            activeFont.metadata.characterToGlyph(char_code)
          ) +
            charSpace * (1000 / fontSize) || 0) / 1000
        );
      } else {
        if (
          doKerning &&
          typeof kerning[char_code] === "object" &&
          !isNaN(parseInt(kerning[char_code][prior_char_code], 10))
        ) {
          kerningValue =
            kerning[char_code][prior_char_code] / kerningFractionOf;
        } else {
          kerningValue = 0;
        }
        output.push(
          (widths[char_code] || default_char_width) / widthsFractionOf +
            kerningValue
        );
      }
      prior_char_code = char_code;
    }

    return output;
  });

  /**
   * Returns a widths of string in a given font, if the font size is set as 1 point.
   *
   * In other words, this is "proportional" value. For 1 unit of font size, the length
   * of the string will be that much.
   *
   * Multiply by font size to get actual width in *points*
   * Then divide by 72 to get inches or divide by (72/25.4) to get 'mm' etc.
   *
   * @name getStringUnitWidth
   * @public
   * @function
   * @param {string} text
   * @param {string} options
   * @returns {number} result
   */
  var getStringUnitWidth = (API.getStringUnitWidth = function(text, options) {
    options = options || {};

    var fontSize = options.fontSize || this.internal.getFontSize();
    var font = options.font || this.internal.getFont();
    var charSpace = options.charSpace || this.internal.getCharSpace();
    var result = 0;

    if (API.processArabic) {
      text = API.processArabic(text);
    }

    if (typeof font.metadata.widthOfString === "function") {
      result =
        font.metadata.widthOfString(text, fontSize, charSpace) / fontSize;
    } else {
      result = getCharWidthsArray
        .apply(this, arguments)
        .reduce(function(pv, cv) {
          return pv + cv;
        }, 0);
    }
    return result;
  });

  /**
  returns array of lines
  */
  var splitLongWord = function(word, widths_array, firstLineMaxLen, maxLen) {
    var answer = [];

    // 1st, chop off the piece that can fit on the hanging line.
    var i = 0,
      l = word.length,
      workingLen = 0;
    while (i !== l && workingLen + widths_array[i] < firstLineMaxLen) {
      workingLen += widths_array[i];
      i++;
    }
    // this is first line.
    answer.push(word.slice(0, i));

    // 2nd. Split the rest into maxLen pieces.
    var startOfLine = i;
    workingLen = 0;
    while (i !== l) {
      if (workingLen + widths_array[i] > maxLen) {
        answer.push(word.slice(startOfLine, i));
        workingLen = 0;
        startOfLine = i;
      }
      workingLen += widths_array[i];
      i++;
    }
    if (startOfLine !== i) {
      answer.push(word.slice(startOfLine, i));
    }

    return answer;
  };

  // Note, all sizing inputs for this function must be in "font measurement units"
  // By default, for PDF, it's "point".
  var splitParagraphIntoLines = function(text, maxlen, options) {
    // at this time works only on Western scripts, ones with space char
    // separating the words. Feel free to expand.

    if (!options) {
      options = {};
    }

    var line = [],
      lines = [line],
      line_length = options.textIndent || 0,
      separator_length = 0,
      current_word_length = 0,
      word,
      widths_array,
      words = text.split(" "),
      spaceCharWidth = getCharWidthsArray.apply(this, [" ", options])[0],
      i,
      l,
      tmp,
      lineIndent;

    if (options.lineIndent === -1) {
      lineIndent = words[0].length + 2;
    } else {
      lineIndent = options.lineIndent || 0;
    }
    if (lineIndent) {
      var pad = Array(lineIndent).join(" "),
        wrds = [];
      words.map(function(wrd) {
        wrd = wrd.split(/\s*\n/);
        if (wrd.length > 1) {
          wrds = wrds.concat(
            wrd.map(function(wrd, idx) {
              return (idx && wrd.length ? "\n" : "") + wrd;
            })
          );
        } else {
          wrds.push(wrd[0]);
        }
      });
      words = wrds;
      lineIndent = getStringUnitWidth.apply(this, [pad, options]);
    }

    for (i = 0, l = words.length; i < l; i++) {
      var force = 0;

      word = words[i];
      if (lineIndent && word[0] == "\n") {
        word = word.substr(1);
        force = 1;
      }
      widths_array = getCharWidthsArray.apply(this, [word, options]);
      current_word_length = widths_array.reduce(function(pv, cv) {
        return pv + cv;
      }, 0);

      if (
        line_length + separator_length + current_word_length > maxlen ||
        force
      ) {
        if (current_word_length > maxlen) {
          // this happens when you have space-less long URLs for example.
          // we just chop these to size. We do NOT insert hiphens
          tmp = splitLongWord.apply(this, [
            word,
            widths_array,
            maxlen - (line_length + separator_length),
            maxlen
          ]);
          // first line we add to existing line object
          line.push(tmp.shift()); // it's ok to have extra space indicator there
          // last line we make into new line object
          line = [tmp.pop()];
          // lines in the middle we apped to lines object as whole lines
          while (tmp.length) {
            lines.push([tmp.shift()]); // single fragment occupies whole line
          }
          current_word_length = widths_array
            .slice(word.length - (line[0] ? line[0].length : 0))
            .reduce(function(pv, cv) {
              return pv + cv;
            }, 0);
        } else {
          // just put it on a new line
          line = [word];
        }

        // now we attach new line to lines
        lines.push(line);
        line_length = current_word_length + lineIndent;
        separator_length = spaceCharWidth;
      } else {
        line.push(word);

        line_length += separator_length + current_word_length;
        separator_length = spaceCharWidth;
      }
    }

    var postProcess;
    if (lineIndent) {
      postProcess = function(ln, idx) {
        return (idx ? pad : "") + ln.join(" ");
      };
    } else {
      postProcess = function(ln) {
        return ln.join(" ");
      };
    }

    return lines.map(postProcess);
  };

  /**
   * Splits a given string into an array of strings. Uses 'size' value
   * (in measurement units declared as default for the jsPDF instance)
   * and the font's "widths" and "Kerning" tables, where available, to
   * determine display length of a given string for a given font.
   *
   * We use character's 100% of unit size (height) as width when Width
   * table or other default width is not available.
   *
   * @name splitTextToSize
   * @public
   * @function
   * @param {string} text Unencoded, regular JavaScript (Unicode, UTF-16 / UCS-2) string.
   * @param {number} size Nominal number, measured in units default to this instance of jsPDF.
   * @param {Object} options Optional flags needed for chopper to do the right thing.
   * @returns {Array} array Array with strings chopped to size.
   */
  API.splitTextToSize = function(text, maxlen, options) {

    options = options || {};

    var fsize = options.fontSize || this.internal.getFontSize(),
      newOptions = function(options) {
        var widths = {
            0: 1
          },
          kerning = {};

        if (!options.widths || !options.kerning) {
          var f = this.internal.getFont(options.fontName, options.fontStyle),
            encoding = "Unicode";
          // NOT UTF8, NOT UTF16BE/LE, NOT UCS2BE/LE
          // Actual JavaScript-native String's 16bit char codes used.
          // no multi-byte logic here

          if (f.metadata[encoding]) {
            return {
              widths: f.metadata[encoding].widths || widths,
              kerning: f.metadata[encoding].kerning || kerning
            };
          } else {
            return {
              font: f.metadata,
              fontSize: this.internal.getFontSize(),
              charSpace: this.internal.getCharSpace()
            };
          }
        } else {
          return {
            widths: options.widths,
            kerning: options.kerning
          };
        }
      }.call(this, options);

    // first we split on end-of-line chars
    var paragraphs;
    if (Array.isArray(text)) {
      paragraphs = text;
    } else {
      paragraphs = String(text).split(/\r?\n/);
    }

    // now we convert size (max length of line) into "font size units"
    // at present time, the "font size unit" is always 'point'
    // 'proportional' means, "in proportion to font size"
    var fontUnit_maxLen = (1.0 * this.internal.scaleFactor * maxlen) / fsize;
    // at this time, fsize is always in "points" regardless of the default measurement unit of the doc.
    // this may change in the future?
    // until then, proportional_maxlen is likely to be in 'points'

    // If first line is to be indented (shorter or longer) than maxLen
    // we indicate that by using CSS-style "text-indent" option.
    // here it's in font units too (which is likely 'points')
    // it can be negative (which makes the first line longer than maxLen)
    newOptions.textIndent = options.textIndent
      ? (options.textIndent * 1.0 * this.internal.scaleFactor) / fsize
      : 0;
    newOptions.lineIndent = options.lineIndent;

    var i,
      l,
      output = [];
    for (i = 0, l = paragraphs.length; i < l; i++) {
      output = output.concat(
        splitParagraphIntoLines.apply(this, [
          paragraphs[i],
          fontUnit_maxLen,
          newOptions
        ])
      );
    }

    return output;
  };
})(jsPDF.API);

/** @license
 jsPDF standard_fonts_metrics plugin
 * Copyright (c) 2012 Willow Systems Corporation, https://github.com/willowsystems
 * MIT license.
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * This file adds the standard font metrics to jsPDF.
 *
 * Font metrics data is reprocessed derivative of contents of
 * "Font Metrics for PDF Core 14 Fonts" package, which exhibits the following copyright and license:
 *
 * Copyright (c) 1989, 1990, 1991, 1992, 1993, 1997 Adobe Systems Incorporated. All Rights Reserved.
 *
 * This file and the 14 PostScript(R) AFM files it accompanies may be used,
 * copied, and distributed for any purpose and without charge, with or without
 * modification, provided that all copyright notices are retained; that the AFM
 * files are not distributed without this file; that all modifications to this
 * file or any of the AFM files are prominently noted in the modified file(s);
 * and that this paragraph is not modified. Adobe Systems has no responsibility
 * or obligation to support the use of the AFM files.
 *
 * @name standard_fonts_metrics
 * @module
 */

(function(API) {
  API.__fontmetrics__ = API.__fontmetrics__ || {};

  var decoded = "0123456789abcdef",
    encoded = "klmnopqrstuvwxyz",
    mappingUncompress = {},
    mappingCompress = {};

  for (var i = 0; i < encoded.length; i++) {
    mappingUncompress[encoded[i]] = decoded[i];
    mappingCompress[decoded[i]] = encoded[i];
  }

  var hex = function(value) {
    return "0x" + parseInt(value, 10).toString(16);
  };

  var compress = (API.__fontmetrics__.compress = function(data) {
    var vals = ["{"];
    var value, keystring, valuestring, numberprefix;

    for (var key in data) {
      value = data[key];

      if (!isNaN(parseInt(key, 10))) {
        key = parseInt(key, 10);
        keystring = hex(key).slice(2);
        keystring =
          keystring.slice(0, -1) + mappingCompress[keystring.slice(-1)];
      } else {
        keystring = "'" + key + "'";
      }

      if (typeof value == "number") {
        if (value < 0) {
          valuestring = hex(value).slice(3);
          numberprefix = "-";
        } else {
          valuestring = hex(value).slice(2);
          numberprefix = "";
        }
        valuestring =
          numberprefix +
          valuestring.slice(0, -1) +
          mappingCompress[valuestring.slice(-1)];
      } else {
        if (typeof value === "object") {
          valuestring = compress(value);
        } else {
          throw new Error(
            "Don't know what to do with value type " + typeof value + "."
          );
        }
      }
      vals.push(keystring + valuestring);
    }
    vals.push("}");
    return vals.join("");
  });

  /**
   * Uncompresses data compressed into custom, base16-like format.
   *
   * @public
   * @function
   * @param
   * @returns {Type}
   */
  var uncompress = (API.__fontmetrics__.uncompress = function(data) {
    if (typeof data !== "string") {
      throw new Error("Invalid argument passed to uncompress.");
    }

    var output = {},
      sign = 1,
      stringparts, // undef. will be [] in string mode
      activeobject = output,
      parentchain = [],
      parent_key_pair,
      keyparts = "",
      valueparts = "",
      key, // undef. will be Truthy when Key is resolved.
      datalen = data.length - 1, // stripping ending }
      ch;

    for (var i = 1; i < datalen; i += 1) {
      // - { } ' are special.

      ch = data[i];

      if (ch == "'") {
        if (stringparts) {
          // end of string mode
          key = stringparts.join("");
          stringparts = undefined;
        } else {
          // start of string mode
          stringparts = [];
        }
      } else if (stringparts) {
        stringparts.push(ch);
      } else if (ch == "{") {
        // start of object
        parentchain.push([activeobject, key]);
        activeobject = {};
        key = undefined;
      } else if (ch == "}") {
        // end of object
        parent_key_pair = parentchain.pop();
        parent_key_pair[0][parent_key_pair[1]] = activeobject;
        key = undefined;
        activeobject = parent_key_pair[0];
      } else if (ch == "-") {
        sign = -1;
      } else {
        // must be number
        if (key === undefined) {
          if (mappingUncompress.hasOwnProperty(ch)) {
            keyparts += mappingUncompress[ch];
            key = parseInt(keyparts, 16) * sign;
            sign = +1;
            keyparts = "";
          } else {
            keyparts += ch;
          }
        } else {
          if (mappingUncompress.hasOwnProperty(ch)) {
            valueparts += mappingUncompress[ch];
            activeobject[key] = parseInt(valueparts, 16) * sign;
            sign = +1;
            key = undefined;
            valueparts = "";
          } else {
            valueparts += ch;
          }
        }
      }
    }
    return output;
  });

  // encoding = 'Unicode'
  // NOT UTF8, NOT UTF16BE/LE, NOT UCS2BE/LE. NO clever BOM behavior
  // Actual 16bit char codes used.
  // no multi-byte logic here

  // Unicode characters to WinAnsiEncoding:
  // {402: 131, 8211: 150, 8212: 151, 8216: 145, 8217: 146, 8218: 130, 8220: 147, 8221: 148, 8222: 132, 8224: 134, 8225: 135, 8226: 149, 8230: 133, 8364: 128, 8240:137, 8249: 139, 8250: 155, 710: 136, 8482: 153, 338: 140, 339: 156, 732: 152, 352: 138, 353: 154, 376: 159, 381: 142, 382: 158}
  // as you can see, all Unicode chars are outside of 0-255 range. No char code conflicts.
  // this means that you can give Win cp1252 encoded strings to jsPDF for rendering directly
  // as well as give strings with some (supported by these fonts) Unicode characters and
  // these will be mapped to win cp1252
  // for example, you can send char code (cp1252) 0x80 or (unicode) 0x20AC, getting "Euro" glyph displayed in both cases.

  var encodingBlock = {
    codePages: ["WinAnsiEncoding"],
    WinAnsiEncoding: uncompress(
      "{19m8n201n9q201o9r201s9l201t9m201u8m201w9n201x9o201y8o202k8q202l8r202m9p202q8p20aw8k203k8t203t8v203u9v2cq8s212m9t15m8w15n9w2dw9s16k8u16l9u17s9z17x8y17y9y}"
    )
  };
  var encodings = {
    Unicode: {
      Courier: encodingBlock,
      "Courier-Bold": encodingBlock,
      "Courier-BoldOblique": encodingBlock,
      "Courier-Oblique": encodingBlock,
      Helvetica: encodingBlock,
      "Helvetica-Bold": encodingBlock,
      "Helvetica-BoldOblique": encodingBlock,
      "Helvetica-Oblique": encodingBlock,
      "Times-Roman": encodingBlock,
      "Times-Bold": encodingBlock,
      "Times-BoldItalic": encodingBlock,
      "Times-Italic": encodingBlock
      //	, 'Symbol'
      //	, 'ZapfDingbats'
    }
  };

  var fontMetrics = {
    Unicode: {
      // all sizing numbers are n/fontMetricsFractionOf = one font size unit
      // this means that if fontMetricsFractionOf = 1000, and letter A's width is 476, it's
      // width is 476/1000 or 47.6% of its height (regardless of font size)
      // At this time this value applies to "widths" and "kerning" numbers.

      // char code 0 represents "default" (average) width - use it for chars missing in this table.
      // key 'fof' represents the "fontMetricsFractionOf" value

      "Courier-Oblique": uncompress(
        "{'widths'{k3w'fof'6o}'kerning'{'fof'-6o}}"
      ),
      "Times-BoldItalic": uncompress(
        "{'widths'{k3o2q4ycx2r201n3m201o6o201s2l201t2l201u2l201w3m201x3m201y3m2k1t2l2r202m2n2n3m2o3m2p5n202q6o2r1w2s2l2t2l2u3m2v3t2w1t2x2l2y1t2z1w3k3m3l3m3m3m3n3m3o3m3p3m3q3m3r3m3s3m203t2l203u2l3v2l3w3t3x3t3y3t3z3m4k5n4l4m4m4m4n4m4o4s4p4m4q4m4r4s4s4y4t2r4u3m4v4m4w3x4x5t4y4s4z4s5k3x5l4s5m4m5n3r5o3x5p4s5q4m5r5t5s4m5t3x5u3x5v2l5w1w5x2l5y3t5z3m6k2l6l3m6m3m6n2w6o3m6p2w6q2l6r3m6s3r6t1w6u1w6v3m6w1w6x4y6y3r6z3m7k3m7l3m7m2r7n2r7o1w7p3r7q2w7r4m7s3m7t2w7u2r7v2n7w1q7x2n7y3t202l3mcl4mal2ram3man3mao3map3mar3mas2lat4uau1uav3maw3way4uaz2lbk2sbl3t'fof'6obo2lbp3tbq3mbr1tbs2lbu1ybv3mbz3mck4m202k3mcm4mcn4mco4mcp4mcq5ycr4mcs4mct4mcu4mcv4mcw2r2m3rcy2rcz2rdl4sdm4sdn4sdo4sdp4sdq4sds4sdt4sdu4sdv4sdw4sdz3mek3mel3mem3men3meo3mep3meq4ser2wes2wet2weu2wev2wew1wex1wey1wez1wfl3rfm3mfn3mfo3mfp3mfq3mfr3tfs3mft3rfu3rfv3rfw3rfz2w203k6o212m6o2dw2l2cq2l3t3m3u2l17s3x19m3m}'kerning'{cl{4qu5kt5qt5rs17ss5ts}201s{201ss}201t{cks4lscmscnscoscpscls2wu2yu201ts}201x{2wu2yu}2k{201ts}2w{4qx5kx5ou5qx5rs17su5tu}2x{17su5tu5ou}2y{4qx5kx5ou5qx5rs17ss5ts}'fof'-6ofn{17sw5tw5ou5qw5rs}7t{cksclscmscnscoscps4ls}3u{17su5tu5os5qs}3v{17su5tu5os5qs}7p{17su5tu}ck{4qu5kt5qt5rs17ss5ts}4l{4qu5kt5qt5rs17ss5ts}cm{4qu5kt5qt5rs17ss5ts}cn{4qu5kt5qt5rs17ss5ts}co{4qu5kt5qt5rs17ss5ts}cp{4qu5kt5qt5rs17ss5ts}6l{4qu5ou5qw5rt17su5tu}5q{ckuclucmucnucoucpu4lu}5r{ckuclucmucnucoucpu4lu}7q{cksclscmscnscoscps4ls}6p{4qu5ou5qw5rt17sw5tw}ek{4qu5ou5qw5rt17su5tu}el{4qu5ou5qw5rt17su5tu}em{4qu5ou5qw5rt17su5tu}en{4qu5ou5qw5rt17su5tu}eo{4qu5ou5qw5rt17su5tu}ep{4qu5ou5qw5rt17su5tu}es{17ss5ts5qs4qu}et{4qu5ou5qw5rt17sw5tw}eu{4qu5ou5qw5rt17ss5ts}ev{17ss5ts5qs4qu}6z{17sw5tw5ou5qw5rs}fm{17sw5tw5ou5qw5rs}7n{201ts}fo{17sw5tw5ou5qw5rs}fp{17sw5tw5ou5qw5rs}fq{17sw5tw5ou5qw5rs}7r{cksclscmscnscoscps4ls}fs{17sw5tw5ou5qw5rs}ft{17su5tu}fu{17su5tu}fv{17su5tu}fw{17su5tu}fz{cksclscmscnscoscps4ls}}}"
      ),
      "Helvetica-Bold": uncompress(
        "{'widths'{k3s2q4scx1w201n3r201o6o201s1w201t1w201u1w201w3m201x3m201y3m2k1w2l2l202m2n2n3r2o3r2p5t202q6o2r1s2s2l2t2l2u2r2v3u2w1w2x2l2y1w2z1w3k3r3l3r3m3r3n3r3o3r3p3r3q3r3r3r3s3r203t2l203u2l3v2l3w3u3x3u3y3u3z3x4k6l4l4s4m4s4n4s4o4s4p4m4q3x4r4y4s4s4t1w4u3r4v4s4w3x4x5n4y4s4z4y5k4m5l4y5m4s5n4m5o3x5p4s5q4m5r5y5s4m5t4m5u3x5v2l5w1w5x2l5y3u5z3r6k2l6l3r6m3x6n3r6o3x6p3r6q2l6r3x6s3x6t1w6u1w6v3r6w1w6x5t6y3x6z3x7k3x7l3x7m2r7n3r7o2l7p3x7q3r7r4y7s3r7t3r7u3m7v2r7w1w7x2r7y3u202l3rcl4sal2lam3ran3rao3rap3rar3ras2lat4tau2pav3raw3uay4taz2lbk2sbl3u'fof'6obo2lbp3xbq3rbr1wbs2lbu2obv3rbz3xck4s202k3rcm4scn4sco4scp4scq6ocr4scs4mct4mcu4mcv4mcw1w2m2zcy1wcz1wdl4sdm4ydn4ydo4ydp4ydq4yds4ydt4sdu4sdv4sdw4sdz3xek3rel3rem3ren3reo3rep3req5ter3res3ret3reu3rev3rew1wex1wey1wez1wfl3xfm3xfn3xfo3xfp3xfq3xfr3ufs3xft3xfu3xfv3xfw3xfz3r203k6o212m6o2dw2l2cq2l3t3r3u2l17s4m19m3r}'kerning'{cl{4qs5ku5ot5qs17sv5tv}201t{2ww4wy2yw}201w{2ks}201x{2ww4wy2yw}2k{201ts201xs}2w{7qs4qu5kw5os5qw5rs17su5tu7tsfzs}2x{5ow5qs}2y{7qs4qu5kw5os5qw5rs17su5tu7tsfzs}'fof'-6o7p{17su5tu5ot}ck{4qs5ku5ot5qs17sv5tv}4l{4qs5ku5ot5qs17sv5tv}cm{4qs5ku5ot5qs17sv5tv}cn{4qs5ku5ot5qs17sv5tv}co{4qs5ku5ot5qs17sv5tv}cp{4qs5ku5ot5qs17sv5tv}6l{17st5tt5os}17s{2kwclvcmvcnvcovcpv4lv4wwckv}5o{2kucltcmtcntcotcpt4lt4wtckt}5q{2ksclscmscnscoscps4ls4wvcks}5r{2ks4ws}5t{2kwclvcmvcnvcovcpv4lv4wwckv}eo{17st5tt5os}fu{17su5tu5ot}6p{17ss5ts}ek{17st5tt5os}el{17st5tt5os}em{17st5tt5os}en{17st5tt5os}6o{201ts}ep{17st5tt5os}es{17ss5ts}et{17ss5ts}eu{17ss5ts}ev{17ss5ts}6z{17su5tu5os5qt}fm{17su5tu5os5qt}fn{17su5tu5os5qt}fo{17su5tu5os5qt}fp{17su5tu5os5qt}fq{17su5tu5os5qt}fs{17su5tu5os5qt}ft{17su5tu5ot}7m{5os}fv{17su5tu5ot}fw{17su5tu5ot}}}"
      ),
      Courier: uncompress("{'widths'{k3w'fof'6o}'kerning'{'fof'-6o}}"),
      "Courier-BoldOblique": uncompress(
        "{'widths'{k3w'fof'6o}'kerning'{'fof'-6o}}"
      ),
      "Times-Bold": uncompress(
        "{'widths'{k3q2q5ncx2r201n3m201o6o201s2l201t2l201u2l201w3m201x3m201y3m2k1t2l2l202m2n2n3m2o3m2p6o202q6o2r1w2s2l2t2l2u3m2v3t2w1t2x2l2y1t2z1w3k3m3l3m3m3m3n3m3o3m3p3m3q3m3r3m3s3m203t2l203u2l3v2l3w3t3x3t3y3t3z3m4k5x4l4s4m4m4n4s4o4s4p4m4q3x4r4y4s4y4t2r4u3m4v4y4w4m4x5y4y4s4z4y5k3x5l4y5m4s5n3r5o4m5p4s5q4s5r6o5s4s5t4s5u4m5v2l5w1w5x2l5y3u5z3m6k2l6l3m6m3r6n2w6o3r6p2w6q2l6r3m6s3r6t1w6u2l6v3r6w1w6x5n6y3r6z3m7k3r7l3r7m2w7n2r7o2l7p3r7q3m7r4s7s3m7t3m7u2w7v2r7w1q7x2r7y3o202l3mcl4sal2lam3man3mao3map3mar3mas2lat4uau1yav3maw3tay4uaz2lbk2sbl3t'fof'6obo2lbp3rbr1tbs2lbu2lbv3mbz3mck4s202k3mcm4scn4sco4scp4scq6ocr4scs4mct4mcu4mcv4mcw2r2m3rcy2rcz2rdl4sdm4ydn4ydo4ydp4ydq4yds4ydt4sdu4sdv4sdw4sdz3rek3mel3mem3men3meo3mep3meq4ser2wes2wet2weu2wev2wew1wex1wey1wez1wfl3rfm3mfn3mfo3mfp3mfq3mfr3tfs3mft3rfu3rfv3rfw3rfz3m203k6o212m6o2dw2l2cq2l3t3m3u2l17s4s19m3m}'kerning'{cl{4qt5ks5ot5qy5rw17sv5tv}201t{cks4lscmscnscoscpscls4wv}2k{201ts}2w{4qu5ku7mu5os5qx5ru17su5tu}2x{17su5tu5ou5qs}2y{4qv5kv7mu5ot5qz5ru17su5tu}'fof'-6o7t{cksclscmscnscoscps4ls}3u{17su5tu5os5qu}3v{17su5tu5os5qu}fu{17su5tu5ou5qu}7p{17su5tu5ou5qu}ck{4qt5ks5ot5qy5rw17sv5tv}4l{4qt5ks5ot5qy5rw17sv5tv}cm{4qt5ks5ot5qy5rw17sv5tv}cn{4qt5ks5ot5qy5rw17sv5tv}co{4qt5ks5ot5qy5rw17sv5tv}cp{4qt5ks5ot5qy5rw17sv5tv}6l{17st5tt5ou5qu}17s{ckuclucmucnucoucpu4lu4wu}5o{ckuclucmucnucoucpu4lu4wu}5q{ckzclzcmzcnzcozcpz4lz4wu}5r{ckxclxcmxcnxcoxcpx4lx4wu}5t{ckuclucmucnucoucpu4lu4wu}7q{ckuclucmucnucoucpu4lu}6p{17sw5tw5ou5qu}ek{17st5tt5qu}el{17st5tt5ou5qu}em{17st5tt5qu}en{17st5tt5qu}eo{17st5tt5qu}ep{17st5tt5ou5qu}es{17ss5ts5qu}et{17sw5tw5ou5qu}eu{17sw5tw5ou5qu}ev{17ss5ts5qu}6z{17sw5tw5ou5qu5rs}fm{17sw5tw5ou5qu5rs}fn{17sw5tw5ou5qu5rs}fo{17sw5tw5ou5qu5rs}fp{17sw5tw5ou5qu5rs}fq{17sw5tw5ou5qu5rs}7r{cktcltcmtcntcotcpt4lt5os}fs{17sw5tw5ou5qu5rs}ft{17su5tu5ou5qu}7m{5os}fv{17su5tu5ou5qu}fw{17su5tu5ou5qu}fz{cksclscmscnscoscps4ls}}}"
      ),
      Symbol: uncompress(
        "{'widths'{k3uaw4r19m3m2k1t2l2l202m2y2n3m2p5n202q6o3k3m2s2l2t2l2v3r2w1t3m3m2y1t2z1wbk2sbl3r'fof'6o3n3m3o3m3p3m3q3m3r3m3s3m3t3m3u1w3v1w3w3r3x3r3y3r3z2wbp3t3l3m5v2l5x2l5z3m2q4yfr3r7v3k7w1o7x3k}'kerning'{'fof'-6o}}"
      ),
      Helvetica: uncompress(
        "{'widths'{k3p2q4mcx1w201n3r201o6o201s1q201t1q201u1q201w2l201x2l201y2l2k1w2l1w202m2n2n3r2o3r2p5t202q6o2r1n2s2l2t2l2u2r2v3u2w1w2x2l2y1w2z1w3k3r3l3r3m3r3n3r3o3r3p3r3q3r3r3r3s3r203t2l203u2l3v1w3w3u3x3u3y3u3z3r4k6p4l4m4m4m4n4s4o4s4p4m4q3x4r4y4s4s4t1w4u3m4v4m4w3r4x5n4y4s4z4y5k4m5l4y5m4s5n4m5o3x5p4s5q4m5r5y5s4m5t4m5u3x5v1w5w1w5x1w5y2z5z3r6k2l6l3r6m3r6n3m6o3r6p3r6q1w6r3r6s3r6t1q6u1q6v3m6w1q6x5n6y3r6z3r7k3r7l3r7m2l7n3m7o1w7p3r7q3m7r4s7s3m7t3m7u3m7v2l7w1u7x2l7y3u202l3rcl4mal2lam3ran3rao3rap3rar3ras2lat4tau2pav3raw3uay4taz2lbk2sbl3u'fof'6obo2lbp3rbr1wbs2lbu2obv3rbz3xck4m202k3rcm4mcn4mco4mcp4mcq6ocr4scs4mct4mcu4mcv4mcw1w2m2ncy1wcz1wdl4sdm4ydn4ydo4ydp4ydq4yds4ydt4sdu4sdv4sdw4sdz3xek3rel3rem3ren3reo3rep3req5ter3mes3ret3reu3rev3rew1wex1wey1wez1wfl3rfm3rfn3rfo3rfp3rfq3rfr3ufs3xft3rfu3rfv3rfw3rfz3m203k6o212m6o2dw2l2cq2l3t3r3u1w17s4m19m3r}'kerning'{5q{4wv}cl{4qs5kw5ow5qs17sv5tv}201t{2wu4w1k2yu}201x{2wu4wy2yu}17s{2ktclucmucnu4otcpu4lu4wycoucku}2w{7qs4qz5k1m17sy5ow5qx5rsfsu5ty7tufzu}2x{17sy5ty5oy5qs}2y{7qs4qz5k1m17sy5ow5qx5rsfsu5ty7tufzu}'fof'-6o7p{17sv5tv5ow}ck{4qs5kw5ow5qs17sv5tv}4l{4qs5kw5ow5qs17sv5tv}cm{4qs5kw5ow5qs17sv5tv}cn{4qs5kw5ow5qs17sv5tv}co{4qs5kw5ow5qs17sv5tv}cp{4qs5kw5ow5qs17sv5tv}6l{17sy5ty5ow}do{17st5tt}4z{17st5tt}7s{fst}dm{17st5tt}dn{17st5tt}5o{ckwclwcmwcnwcowcpw4lw4wv}dp{17st5tt}dq{17st5tt}7t{5ow}ds{17st5tt}5t{2ktclucmucnu4otcpu4lu4wycoucku}fu{17sv5tv5ow}6p{17sy5ty5ow5qs}ek{17sy5ty5ow}el{17sy5ty5ow}em{17sy5ty5ow}en{5ty}eo{17sy5ty5ow}ep{17sy5ty5ow}es{17sy5ty5qs}et{17sy5ty5ow5qs}eu{17sy5ty5ow5qs}ev{17sy5ty5ow5qs}6z{17sy5ty5ow5qs}fm{17sy5ty5ow5qs}fn{17sy5ty5ow5qs}fo{17sy5ty5ow5qs}fp{17sy5ty5qs}fq{17sy5ty5ow5qs}7r{5ow}fs{17sy5ty5ow5qs}ft{17sv5tv5ow}7m{5ow}fv{17sv5tv5ow}fw{17sv5tv5ow}}}"
      ),
      "Helvetica-BoldOblique": uncompress(
        "{'widths'{k3s2q4scx1w201n3r201o6o201s1w201t1w201u1w201w3m201x3m201y3m2k1w2l2l202m2n2n3r2o3r2p5t202q6o2r1s2s2l2t2l2u2r2v3u2w1w2x2l2y1w2z1w3k3r3l3r3m3r3n3r3o3r3p3r3q3r3r3r3s3r203t2l203u2l3v2l3w3u3x3u3y3u3z3x4k6l4l4s4m4s4n4s4o4s4p4m4q3x4r4y4s4s4t1w4u3r4v4s4w3x4x5n4y4s4z4y5k4m5l4y5m4s5n4m5o3x5p4s5q4m5r5y5s4m5t4m5u3x5v2l5w1w5x2l5y3u5z3r6k2l6l3r6m3x6n3r6o3x6p3r6q2l6r3x6s3x6t1w6u1w6v3r6w1w6x5t6y3x6z3x7k3x7l3x7m2r7n3r7o2l7p3x7q3r7r4y7s3r7t3r7u3m7v2r7w1w7x2r7y3u202l3rcl4sal2lam3ran3rao3rap3rar3ras2lat4tau2pav3raw3uay4taz2lbk2sbl3u'fof'6obo2lbp3xbq3rbr1wbs2lbu2obv3rbz3xck4s202k3rcm4scn4sco4scp4scq6ocr4scs4mct4mcu4mcv4mcw1w2m2zcy1wcz1wdl4sdm4ydn4ydo4ydp4ydq4yds4ydt4sdu4sdv4sdw4sdz3xek3rel3rem3ren3reo3rep3req5ter3res3ret3reu3rev3rew1wex1wey1wez1wfl3xfm3xfn3xfo3xfp3xfq3xfr3ufs3xft3xfu3xfv3xfw3xfz3r203k6o212m6o2dw2l2cq2l3t3r3u2l17s4m19m3r}'kerning'{cl{4qs5ku5ot5qs17sv5tv}201t{2ww4wy2yw}201w{2ks}201x{2ww4wy2yw}2k{201ts201xs}2w{7qs4qu5kw5os5qw5rs17su5tu7tsfzs}2x{5ow5qs}2y{7qs4qu5kw5os5qw5rs17su5tu7tsfzs}'fof'-6o7p{17su5tu5ot}ck{4qs5ku5ot5qs17sv5tv}4l{4qs5ku5ot5qs17sv5tv}cm{4qs5ku5ot5qs17sv5tv}cn{4qs5ku5ot5qs17sv5tv}co{4qs5ku5ot5qs17sv5tv}cp{4qs5ku5ot5qs17sv5tv}6l{17st5tt5os}17s{2kwclvcmvcnvcovcpv4lv4wwckv}5o{2kucltcmtcntcotcpt4lt4wtckt}5q{2ksclscmscnscoscps4ls4wvcks}5r{2ks4ws}5t{2kwclvcmvcnvcovcpv4lv4wwckv}eo{17st5tt5os}fu{17su5tu5ot}6p{17ss5ts}ek{17st5tt5os}el{17st5tt5os}em{17st5tt5os}en{17st5tt5os}6o{201ts}ep{17st5tt5os}es{17ss5ts}et{17ss5ts}eu{17ss5ts}ev{17ss5ts}6z{17su5tu5os5qt}fm{17su5tu5os5qt}fn{17su5tu5os5qt}fo{17su5tu5os5qt}fp{17su5tu5os5qt}fq{17su5tu5os5qt}fs{17su5tu5os5qt}ft{17su5tu5ot}7m{5os}fv{17su5tu5ot}fw{17su5tu5ot}}}"
      ),
      ZapfDingbats: uncompress("{'widths'{k4u2k1w'fof'6o}'kerning'{'fof'-6o}}"),
      "Courier-Bold": uncompress("{'widths'{k3w'fof'6o}'kerning'{'fof'-6o}}"),
      "Times-Italic": uncompress(
        "{'widths'{k3n2q4ycx2l201n3m201o5t201s2l201t2l201u2l201w3r201x3r201y3r2k1t2l2l202m2n2n3m2o3m2p5n202q5t2r1p2s2l2t2l2u3m2v4n2w1t2x2l2y1t2z1w3k3m3l3m3m3m3n3m3o3m3p3m3q3m3r3m3s3m203t2l203u2l3v2l3w4n3x4n3y4n3z3m4k5w4l3x4m3x4n4m4o4s4p3x4q3x4r4s4s4s4t2l4u2w4v4m4w3r4x5n4y4m4z4s5k3x5l4s5m3x5n3m5o3r5p4s5q3x5r5n5s3x5t3r5u3r5v2r5w1w5x2r5y2u5z3m6k2l6l3m6m3m6n2w6o3m6p2w6q1w6r3m6s3m6t1w6u1w6v2w6w1w6x4s6y3m6z3m7k3m7l3m7m2r7n2r7o1w7p3m7q2w7r4m7s2w7t2w7u2r7v2s7w1v7x2s7y3q202l3mcl3xal2ram3man3mao3map3mar3mas2lat4wau1vav3maw4nay4waz2lbk2sbl4n'fof'6obo2lbp3mbq3obr1tbs2lbu1zbv3mbz3mck3x202k3mcm3xcn3xco3xcp3xcq5tcr4mcs3xct3xcu3xcv3xcw2l2m2ucy2lcz2ldl4mdm4sdn4sdo4sdp4sdq4sds4sdt4sdu4sdv4sdw4sdz3mek3mel3mem3men3meo3mep3meq4mer2wes2wet2weu2wev2wew1wex1wey1wez1wfl3mfm3mfn3mfo3mfp3mfq3mfr4nfs3mft3mfu3mfv3mfw3mfz2w203k6o212m6m2dw2l2cq2l3t3m3u2l17s3r19m3m}'kerning'{cl{5kt4qw}201s{201sw}201t{201tw2wy2yy6q-t}201x{2wy2yy}2k{201tw}2w{7qs4qy7rs5ky7mw5os5qx5ru17su5tu}2x{17ss5ts5os}2y{7qs4qy7rs5ky7mw5os5qx5ru17su5tu}'fof'-6o6t{17ss5ts5qs}7t{5os}3v{5qs}7p{17su5tu5qs}ck{5kt4qw}4l{5kt4qw}cm{5kt4qw}cn{5kt4qw}co{5kt4qw}cp{5kt4qw}6l{4qs5ks5ou5qw5ru17su5tu}17s{2ks}5q{ckvclvcmvcnvcovcpv4lv}5r{ckuclucmucnucoucpu4lu}5t{2ks}6p{4qs5ks5ou5qw5ru17su5tu}ek{4qs5ks5ou5qw5ru17su5tu}el{4qs5ks5ou5qw5ru17su5tu}em{4qs5ks5ou5qw5ru17su5tu}en{4qs5ks5ou5qw5ru17su5tu}eo{4qs5ks5ou5qw5ru17su5tu}ep{4qs5ks5ou5qw5ru17su5tu}es{5ks5qs4qs}et{4qs5ks5ou5qw5ru17su5tu}eu{4qs5ks5qw5ru17su5tu}ev{5ks5qs4qs}ex{17ss5ts5qs}6z{4qv5ks5ou5qw5ru17su5tu}fm{4qv5ks5ou5qw5ru17su5tu}fn{4qv5ks5ou5qw5ru17su5tu}fo{4qv5ks5ou5qw5ru17su5tu}fp{4qv5ks5ou5qw5ru17su5tu}fq{4qv5ks5ou5qw5ru17su5tu}7r{5os}fs{4qv5ks5ou5qw5ru17su5tu}ft{17su5tu5qs}fu{17su5tu5qs}fv{17su5tu5qs}fw{17su5tu5qs}}}"
      ),
      "Times-Roman": uncompress(
        "{'widths'{k3n2q4ycx2l201n3m201o6o201s2l201t2l201u2l201w2w201x2w201y2w2k1t2l2l202m2n2n3m2o3m2p5n202q6o2r1m2s2l2t2l2u3m2v3s2w1t2x2l2y1t2z1w3k3m3l3m3m3m3n3m3o3m3p3m3q3m3r3m3s3m203t2l203u2l3v1w3w3s3x3s3y3s3z2w4k5w4l4s4m4m4n4m4o4s4p3x4q3r4r4s4s4s4t2l4u2r4v4s4w3x4x5t4y4s4z4s5k3r5l4s5m4m5n3r5o3x5p4s5q4s5r5y5s4s5t4s5u3x5v2l5w1w5x2l5y2z5z3m6k2l6l2w6m3m6n2w6o3m6p2w6q2l6r3m6s3m6t1w6u1w6v3m6w1w6x4y6y3m6z3m7k3m7l3m7m2l7n2r7o1w7p3m7q3m7r4s7s3m7t3m7u2w7v3k7w1o7x3k7y3q202l3mcl4sal2lam3man3mao3map3mar3mas2lat4wau1vav3maw3say4waz2lbk2sbl3s'fof'6obo2lbp3mbq2xbr1tbs2lbu1zbv3mbz2wck4s202k3mcm4scn4sco4scp4scq5tcr4mcs3xct3xcu3xcv3xcw2l2m2tcy2lcz2ldl4sdm4sdn4sdo4sdp4sdq4sds4sdt4sdu4sdv4sdw4sdz3mek2wel2wem2wen2weo2wep2weq4mer2wes2wet2weu2wev2wew1wex1wey1wez1wfl3mfm3mfn3mfo3mfp3mfq3mfr3sfs3mft3mfu3mfv3mfw3mfz3m203k6o212m6m2dw2l2cq2l3t3m3u1w17s4s19m3m}'kerning'{cl{4qs5ku17sw5ou5qy5rw201ss5tw201ws}201s{201ss}201t{ckw4lwcmwcnwcowcpwclw4wu201ts}2k{201ts}2w{4qs5kw5os5qx5ru17sx5tx}2x{17sw5tw5ou5qu}2y{4qs5kw5os5qx5ru17sx5tx}'fof'-6o7t{ckuclucmucnucoucpu4lu5os5rs}3u{17su5tu5qs}3v{17su5tu5qs}7p{17sw5tw5qs}ck{4qs5ku17sw5ou5qy5rw201ss5tw201ws}4l{4qs5ku17sw5ou5qy5rw201ss5tw201ws}cm{4qs5ku17sw5ou5qy5rw201ss5tw201ws}cn{4qs5ku17sw5ou5qy5rw201ss5tw201ws}co{4qs5ku17sw5ou5qy5rw201ss5tw201ws}cp{4qs5ku17sw5ou5qy5rw201ss5tw201ws}6l{17su5tu5os5qw5rs}17s{2ktclvcmvcnvcovcpv4lv4wuckv}5o{ckwclwcmwcnwcowcpw4lw4wu}5q{ckyclycmycnycoycpy4ly4wu5ms}5r{cktcltcmtcntcotcpt4lt4ws}5t{2ktclvcmvcnvcovcpv4lv4wuckv}7q{cksclscmscnscoscps4ls}6p{17su5tu5qw5rs}ek{5qs5rs}el{17su5tu5os5qw5rs}em{17su5tu5os5qs5rs}en{17su5qs5rs}eo{5qs5rs}ep{17su5tu5os5qw5rs}es{5qs}et{17su5tu5qw5rs}eu{17su5tu5qs5rs}ev{5qs}6z{17sv5tv5os5qx5rs}fm{5os5qt5rs}fn{17sv5tv5os5qx5rs}fo{17sv5tv5os5qx5rs}fp{5os5qt5rs}fq{5os5qt5rs}7r{ckuclucmucnucoucpu4lu5os}fs{17sv5tv5os5qx5rs}ft{17ss5ts5qs}fu{17sw5tw5qs}fv{17sw5tw5qs}fw{17ss5ts5qs}fz{ckuclucmucnucoucpu4lu5os5rs}}}"
      ),
      "Helvetica-Oblique": uncompress(
        "{'widths'{k3p2q4mcx1w201n3r201o6o201s1q201t1q201u1q201w2l201x2l201y2l2k1w2l1w202m2n2n3r2o3r2p5t202q6o2r1n2s2l2t2l2u2r2v3u2w1w2x2l2y1w2z1w3k3r3l3r3m3r3n3r3o3r3p3r3q3r3r3r3s3r203t2l203u2l3v1w3w3u3x3u3y3u3z3r4k6p4l4m4m4m4n4s4o4s4p4m4q3x4r4y4s4s4t1w4u3m4v4m4w3r4x5n4y4s4z4y5k4m5l4y5m4s5n4m5o3x5p4s5q4m5r5y5s4m5t4m5u3x5v1w5w1w5x1w5y2z5z3r6k2l6l3r6m3r6n3m6o3r6p3r6q1w6r3r6s3r6t1q6u1q6v3m6w1q6x5n6y3r6z3r7k3r7l3r7m2l7n3m7o1w7p3r7q3m7r4s7s3m7t3m7u3m7v2l7w1u7x2l7y3u202l3rcl4mal2lam3ran3rao3rap3rar3ras2lat4tau2pav3raw3uay4taz2lbk2sbl3u'fof'6obo2lbp3rbr1wbs2lbu2obv3rbz3xck4m202k3rcm4mcn4mco4mcp4mcq6ocr4scs4mct4mcu4mcv4mcw1w2m2ncy1wcz1wdl4sdm4ydn4ydo4ydp4ydq4yds4ydt4sdu4sdv4sdw4sdz3xek3rel3rem3ren3reo3rep3req5ter3mes3ret3reu3rev3rew1wex1wey1wez1wfl3rfm3rfn3rfo3rfp3rfq3rfr3ufs3xft3rfu3rfv3rfw3rfz3m203k6o212m6o2dw2l2cq2l3t3r3u1w17s4m19m3r}'kerning'{5q{4wv}cl{4qs5kw5ow5qs17sv5tv}201t{2wu4w1k2yu}201x{2wu4wy2yu}17s{2ktclucmucnu4otcpu4lu4wycoucku}2w{7qs4qz5k1m17sy5ow5qx5rsfsu5ty7tufzu}2x{17sy5ty5oy5qs}2y{7qs4qz5k1m17sy5ow5qx5rsfsu5ty7tufzu}'fof'-6o7p{17sv5tv5ow}ck{4qs5kw5ow5qs17sv5tv}4l{4qs5kw5ow5qs17sv5tv}cm{4qs5kw5ow5qs17sv5tv}cn{4qs5kw5ow5qs17sv5tv}co{4qs5kw5ow5qs17sv5tv}cp{4qs5kw5ow5qs17sv5tv}6l{17sy5ty5ow}do{17st5tt}4z{17st5tt}7s{fst}dm{17st5tt}dn{17st5tt}5o{ckwclwcmwcnwcowcpw4lw4wv}dp{17st5tt}dq{17st5tt}7t{5ow}ds{17st5tt}5t{2ktclucmucnu4otcpu4lu4wycoucku}fu{17sv5tv5ow}6p{17sy5ty5ow5qs}ek{17sy5ty5ow}el{17sy5ty5ow}em{17sy5ty5ow}en{5ty}eo{17sy5ty5ow}ep{17sy5ty5ow}es{17sy5ty5qs}et{17sy5ty5ow5qs}eu{17sy5ty5ow5qs}ev{17sy5ty5ow5qs}6z{17sy5ty5ow5qs}fm{17sy5ty5ow5qs}fn{17sy5ty5ow5qs}fo{17sy5ty5ow5qs}fp{17sy5ty5qs}fq{17sy5ty5ow5qs}7r{5ow}fs{17sy5ty5ow5qs}ft{17sv5tv5ow}7m{5ow}fv{17sv5tv5ow}fw{17sv5tv5ow}}}"
      )
    }
  };

  /*
	This event handler is fired when a new jsPDF object is initialized
	This event handler appends metrics data to standard fonts within
	that jsPDF instance. The metrics are mapped over Unicode character
	codes, NOT CIDs or other codes matching the StandardEncoding table of the
	standard PDF fonts.
	Future:
	Also included is the encoding maping table, converting Unicode (UCS-2, UTF-16)
	char codes to StandardEncoding character codes. The encoding table is to be used
	somewhere around "pdfEscape" call.
	*/
  API.events.push([
    "addFont",
    function(data) {
      var font = data.font;

      var metrics = fontMetrics["Unicode"][font.postScriptName];
      if (metrics) {
        font.metadata["Unicode"] = {};
        font.metadata["Unicode"].widths = metrics.widths;
        font.metadata["Unicode"].kerning = metrics.kerning;
      }

      var encodingBlock = encodings["Unicode"][font.postScriptName];
      if (encodingBlock) {
        font.metadata["Unicode"].encoding = encodingBlock;
        font.encoding = encodingBlock.codePages[0];
      }
    }
  ]); // end of adding event handler
})(jsPDF.API);

/**
 * @license
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * @name ttfsupport
 * @module
 */
(function(jsPDF) {

  var binaryStringToUint8Array = function(binary_string) {
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
      bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes;
  };

  var addFont = function(font, file) {
    // eslint-disable-next-line no-control-regex
    if (/^\x00\x01\x00\x00/.test(file)) {
      file = binaryStringToUint8Array(file);
    } else {
      file = binaryStringToUint8Array(atob(file));
    }
    font.metadata = jsPDF.API.TTFFont.open(file);
    font.metadata.Unicode = font.metadata.Unicode || {
      encoding: {},
      kerning: {},
      widths: []
    };
    font.metadata.glyIdsUsed = [0];
  };

  jsPDF.API.events.push([
    "addFont",
    function(data) {
      var file = undefined;
      var font = data.font;
      var instance = data.instance;
      if (font.isStandardFont) {
        return;
      }
      if (typeof instance !== "undefined") {
        if (instance.existsFileInVFS(font.postScriptName) === false) {
          file = instance.loadFile(font.postScriptName);
        } else {
          file = instance.getFileFromVFS(font.postScriptName);
        }
        if (typeof file !== "string") {
          throw new Error(
            "Font is not stored as string-data in vFS, import fonts or remove declaration doc.addFont('" +
              font.postScriptName +
              "')."
          );
        }
        addFont(font, file);
      } else {
        throw new Error(
          "Font does not exist in vFS, import fonts or remove declaration doc.addFont('" +
            font.postScriptName +
            "')."
        );
      }
    }
  ]); // end of adding event handler
})(jsPDF);

/**
 * @license
 * ====================================================================
 * Copyright (c) 2013 Eduardo Menezes de Morais, eduardo.morais@usp.br
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * jsPDF total_pages plugin
 * @name total_pages
 * @module
 */
(function(jsPDFAPI) {
  /**
   * @name putTotalPages
   * @function
   * @param {string} pageExpression Regular Expression
   * @returns {jsPDF} jsPDF-instance
   */

  jsPDFAPI.putTotalPages = function(pageExpression) {

    var replaceExpression;
    var totalNumberOfPages = 0;
    if (parseInt(this.internal.getFont().id.substr(1), 10) < 15) {
      replaceExpression = new RegExp(pageExpression, "g");
      totalNumberOfPages = this.internal.getNumberOfPages();
    } else {
      replaceExpression = new RegExp(
        this.pdfEscape16(pageExpression, this.internal.getFont()),
        "g"
      );
      totalNumberOfPages = this.pdfEscape16(
        this.internal.getNumberOfPages() + "",
        this.internal.getFont()
      );
    }

    for (var n = 1; n <= this.internal.getNumberOfPages(); n++) {
      for (var i = 0; i < this.internal.pages[n].length; i++) {
        this.internal.pages[n][i] = this.internal.pages[n][i].replace(
          replaceExpression,
          totalNumberOfPages
        );
      }
    }

    return this;
  };
})(jsPDF.API);

/**
 * @license
 * jsPDF viewerPreferences Plugin
 * @author Aras Abbasi (github.com/arasabbasi)
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * Adds the ability to set ViewerPreferences and by thus
 * controlling the way the document is to be presented on the
 * screen or in print.
 * @name viewerpreferences
 * @module
 */
(function(jsPDFAPI) {
  /**
   * Set the ViewerPreferences of the generated PDF
   *
   * @name viewerPreferences
   * @function
   * @public
   * @param {Object} options Array with the ViewerPreferences<br />
   * Example: doc.viewerPreferences({"FitWindow":true});<br />
   * <br />
   * You can set following preferences:<br />
   * <br/>
   * <b>HideToolbar</b> <i>(boolean)</i><br />
   * Default value: false<br />
   * <br />
   * <b>HideMenubar</b> <i>(boolean)</i><br />
   * Default value: false.<br />
   * <br />
   * <b>HideWindowUI</b> <i>(boolean)</i><br />
   * Default value: false.<br />
   * <br />
   * <b>FitWindow</b> <i>(boolean)</i><br />
   * Default value: false.<br />
   * <br />
   * <b>CenterWindow</b> <i>(boolean)</i><br />
   * Default value: false<br />
   * <br />
   * <b>DisplayDocTitle</b> <i>(boolean)</i><br />
   * Default value: false.<br />
   * <br />
   * <b>NonFullScreenPageMode</b> <i>(string)</i><br />
   * Possible values: UseNone, UseOutlines, UseThumbs, UseOC<br />
   * Default value: UseNone<br/>
   * <br />
   * <b>Direction</b> <i>(string)</i><br />
   * Possible values: L2R, R2L<br />
   * Default value: L2R.<br />
   * <br />
   * <b>ViewArea</b> <i>(string)</i><br />
   * Possible values: MediaBox, CropBox, TrimBox, BleedBox, ArtBox<br />
   * Default value: CropBox.<br />
   * <br />
   * <b>ViewClip</b> <i>(string)</i><br />
   * Possible values: MediaBox, CropBox, TrimBox, BleedBox, ArtBox<br />
   * Default value: CropBox<br />
   * <br />
   * <b>PrintArea</b> <i>(string)</i><br />
   * Possible values: MediaBox, CropBox, TrimBox, BleedBox, ArtBox<br />
   * Default value: CropBox<br />
   * <br />
   * <b>PrintClip</b> <i>(string)</i><br />
   * Possible values: MediaBox, CropBox, TrimBox, BleedBox, ArtBox<br />
   * Default value: CropBox.<br />
   * <br />
   * <b>PrintScaling</b> <i>(string)</i><br />
   * Possible values: AppDefault, None<br />
   * Default value: AppDefault.<br />
   * <br />
   * <b>Duplex</b> <i>(string)</i><br />
   * Possible values: Simplex, DuplexFlipLongEdge, DuplexFlipShortEdge
   * Default value: none<br />
   * <br />
   * <b>PickTrayByPDFSize</b> <i>(boolean)</i><br />
   * Default value: false<br />
   * <br />
   * <b>PrintPageRange</b> <i>(Array)</i><br />
   * Example: [[1,5], [7,9]]<br />
   * Default value: as defined by PDF viewer application<br />
   * <br />
   * <b>NumCopies</b> <i>(Number)</i><br />
   * Possible values: 1, 2, 3, 4, 5<br />
   * Default value: 1<br />
   * <br />
   * For more information see the PDF Reference, sixth edition on Page 577
   * @param {boolean} doReset True to reset the settings
   * @function
   * @returns jsPDF jsPDF-instance
   * @example
   * var doc = new jsPDF()
   * doc.text('This is a test', 10, 10)
   * doc.viewerPreferences({'FitWindow': true}, true)
   * doc.save("viewerPreferences.pdf")
   *
   * // Example printing 10 copies, using cropbox, and hiding UI.
   * doc.viewerPreferences({
   *   'HideWindowUI': true,
   *   'PrintArea': 'CropBox',
   *   'NumCopies': 10
   * })
   */
  jsPDFAPI.viewerPreferences = function(options, doReset) {
    options = options || {};
    doReset = doReset || false;

    var configuration;
    var configurationTemplate = {
      HideToolbar: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.3
      },
      HideMenubar: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.3
      },
      HideWindowUI: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.3
      },
      FitWindow: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.3
      },
      CenterWindow: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.3
      },
      DisplayDocTitle: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.4
      },
      NonFullScreenPageMode: {
        defaultValue: "UseNone",
        value: "UseNone",
        type: "name",
        explicitSet: false,
        valueSet: ["UseNone", "UseOutlines", "UseThumbs", "UseOC"],
        pdfVersion: 1.3
      },
      Direction: {
        defaultValue: "L2R",
        value: "L2R",
        type: "name",
        explicitSet: false,
        valueSet: ["L2R", "R2L"],
        pdfVersion: 1.3
      },
      ViewArea: {
        defaultValue: "CropBox",
        value: "CropBox",
        type: "name",
        explicitSet: false,
        valueSet: ["MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"],
        pdfVersion: 1.4
      },
      ViewClip: {
        defaultValue: "CropBox",
        value: "CropBox",
        type: "name",
        explicitSet: false,
        valueSet: ["MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"],
        pdfVersion: 1.4
      },
      PrintArea: {
        defaultValue: "CropBox",
        value: "CropBox",
        type: "name",
        explicitSet: false,
        valueSet: ["MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"],
        pdfVersion: 1.4
      },
      PrintClip: {
        defaultValue: "CropBox",
        value: "CropBox",
        type: "name",
        explicitSet: false,
        valueSet: ["MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"],
        pdfVersion: 1.4
      },
      PrintScaling: {
        defaultValue: "AppDefault",
        value: "AppDefault",
        type: "name",
        explicitSet: false,
        valueSet: ["AppDefault", "None"],
        pdfVersion: 1.6
      },
      Duplex: {
        defaultValue: "",
        value: "none",
        type: "name",
        explicitSet: false,
        valueSet: [
          "Simplex",
          "DuplexFlipShortEdge",
          "DuplexFlipLongEdge",
          "none"
        ],
        pdfVersion: 1.7
      },
      PickTrayByPDFSize: {
        defaultValue: false,
        value: false,
        type: "boolean",
        explicitSet: false,
        valueSet: [true, false],
        pdfVersion: 1.7
      },
      PrintPageRange: {
        defaultValue: "",
        value: "",
        type: "array",
        explicitSet: false,
        valueSet: null,
        pdfVersion: 1.7
      },
      NumCopies: {
        defaultValue: 1,
        value: 1,
        type: "integer",
        explicitSet: false,
        valueSet: null,
        pdfVersion: 1.7
      }
    };

    var configurationKeys = Object.keys(configurationTemplate);

    var rangeArray = [];
    var i = 0;
    var j = 0;
    var k = 0;
    var isValid;

    var method;
    var value;

    function arrayContainsElement(array, element) {
      var iterator;
      var result = false;

      for (iterator = 0; iterator < array.length; iterator += 1) {
        if (array[iterator] === element) {
          result = true;
        }
      }
      return result;
    }

    if (this.internal.viewerpreferences === undefined) {
      this.internal.viewerpreferences = {};
      this.internal.viewerpreferences.configuration = JSON.parse(
        JSON.stringify(configurationTemplate)
      );
      this.internal.viewerpreferences.isSubscribed = false;
    }
    configuration = this.internal.viewerpreferences.configuration;

    if (options === "reset" || doReset === true) {
      var len = configurationKeys.length;

      for (k = 0; k < len; k += 1) {
        configuration[configurationKeys[k]].value =
          configuration[configurationKeys[k]].defaultValue;
        configuration[configurationKeys[k]].explicitSet = false;
      }
    }

    if (typeof options === "object") {
      for (method in options) {
        value = options[method];
        if (
          arrayContainsElement(configurationKeys, method) &&
          value !== undefined
        ) {
          if (
            configuration[method].type === "boolean" &&
            typeof value === "boolean"
          ) {
            configuration[method].value = value;
          } else if (
            configuration[method].type === "name" &&
            arrayContainsElement(configuration[method].valueSet, value)
          ) {
            configuration[method].value = value;
          } else if (
            configuration[method].type === "integer" &&
            Number.isInteger(value)
          ) {
            configuration[method].value = value;
          } else if (configuration[method].type === "array") {
            for (i = 0; i < value.length; i += 1) {
              isValid = true;
              if (value[i].length === 1 && typeof value[i][0] === "number") {
                rangeArray.push(String(value[i] - 1));
              } else if (value[i].length > 1) {
                for (j = 0; j < value[i].length; j += 1) {
                  if (typeof value[i][j] !== "number") {
                    isValid = false;
                  }
                }
                if (isValid === true) {
                  rangeArray.push([value[i][0] - 1, value[i][1] - 1].join(" "));
                }
              }
            }
            configuration[method].value = "[" + rangeArray.join(" ") + "]";
          } else {
            configuration[method].value = configuration[method].defaultValue;
          }

          configuration[method].explicitSet = true;
        }
      }
    }

    if (this.internal.viewerpreferences.isSubscribed === false) {
      this.internal.events.subscribe("putCatalog", function() {
        var pdfDict = [];
        var vPref;
        for (vPref in configuration) {
          if (configuration[vPref].explicitSet === true) {
            if (configuration[vPref].type === "name") {
              pdfDict.push("/" + vPref + " /" + configuration[vPref].value);
            } else {
              pdfDict.push("/" + vPref + " " + configuration[vPref].value);
            }
          }
        }
        if (pdfDict.length !== 0) {
          this.internal.write(
            "/ViewerPreferences\n<<\n" + pdfDict.join("\n") + "\n>>"
          );
        }
      });
      this.internal.viewerpreferences.isSubscribed = true;
    }

    this.internal.viewerpreferences.configuration = configuration;
    return this;
  };
})(jsPDF.API);

/** ====================================================================
 * @license
 * jsPDF XMP metadata plugin
 * Copyright (c) 2016 Jussi Utunen, u-jussi@suomi24.fi
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ====================================================================
 */

/**
 * @name xmp_metadata
 * @module
 */
(function(jsPDFAPI) {

  var postPutResources = function() {
    var xmpmeta_beginning = '<x:xmpmeta xmlns:x="adobe:ns:meta/">';
    var rdf_beginning =
      '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:jspdf="' +
      this.internal.__metadata__.namespaceuri +
      '"><jspdf:metadata>';
    var rdf_ending = "</jspdf:metadata></rdf:Description></rdf:RDF>";
    var xmpmeta_ending = "</x:xmpmeta>";
    var utf8_xmpmeta_beginning = unescape(
      encodeURIComponent(xmpmeta_beginning)
    );
    var utf8_rdf_beginning = unescape(encodeURIComponent(rdf_beginning));
    var utf8_metadata = unescape(
      encodeURIComponent(this.internal.__metadata__.metadata)
    );
    var utf8_rdf_ending = unescape(encodeURIComponent(rdf_ending));
    var utf8_xmpmeta_ending = unescape(encodeURIComponent(xmpmeta_ending));

    var total_len =
      utf8_rdf_beginning.length +
      utf8_metadata.length +
      utf8_rdf_ending.length +
      utf8_xmpmeta_beginning.length +
      utf8_xmpmeta_ending.length;

    this.internal.__metadata__.metadata_object_number = this.internal.newObject();
    this.internal.write(
      "<< /Type /Metadata /Subtype /XML /Length " + total_len + " >>"
    );
    this.internal.write("stream");
    this.internal.write(
      utf8_xmpmeta_beginning +
        utf8_rdf_beginning +
        utf8_metadata +
        utf8_rdf_ending +
        utf8_xmpmeta_ending
    );
    this.internal.write("endstream");
    this.internal.write("endobj");
  };

  var putCatalog = function() {
    if (this.internal.__metadata__.metadata_object_number) {
      this.internal.write(
        "/Metadata " +
          this.internal.__metadata__.metadata_object_number +
          " 0 R"
      );
    }
  };

  /**
   * Adds XMP formatted metadata to PDF
   *
   * @name addMetadata
   * @function
   * @param {String} metadata The actual metadata to be added. The metadata shall be stored as XMP simple value. Note that if the metadata string contains XML markup characters "<", ">" or "&", those characters should be written using XML entities.
   * @param {String} namespaceuri Sets the namespace URI for the metadata. Last character should be slash or hash.
   * @returns {jsPDF} jsPDF-instance
   */
  jsPDFAPI.addMetadata = function(metadata, namespaceuri) {
    if (typeof this.internal.__metadata__ === "undefined") {
      this.internal.__metadata__ = {
        metadata: metadata,
        namespaceuri: namespaceuri || "http://jspdf.default.namespaceuri/"
      };
      this.internal.events.subscribe("putCatalog", putCatalog);

      this.internal.events.subscribe("postPutResources", postPutResources);
    }
    return this;
  };
})(jsPDF.API);

/**
 * @name utf8
 * @module
 */
(function(jsPDF) {
  var jsPDFAPI = jsPDF.API;

  /***************************************************************************************************/
  /* function : pdfEscape16                                                                          */
  /* comment : The character id of a 2-byte string is converted to a hexadecimal number by obtaining */
  /*   the corresponding glyph id and width, and then adding padding to the string.                  */
  /***************************************************************************************************/
  var pdfEscape16 = (jsPDFAPI.pdfEscape16 = function(text, font) {
    var widths = font.metadata.Unicode.widths;
    var padz = ["", "0", "00", "000", "0000"];
    var ar = [""];
    for (var i = 0, l = text.length, t; i < l; ++i) {
      t = font.metadata.characterToGlyph(text.charCodeAt(i));
      font.metadata.glyIdsUsed.push(t);
      font.metadata.toUnicode[t] = text.charCodeAt(i);
      if (widths.indexOf(t) == -1) {
        widths.push(t);
        widths.push([parseInt(font.metadata.widthOfGlyph(t), 10)]);
      }
      if (t == "0") {
        //Spaces are not allowed in cmap.
        return ar.join("");
      } else {
        t = t.toString(16);
        ar.push(padz[4 - t.length], t);
      }
    }
    return ar.join("");
  });

  var toUnicodeCmap = function(map) {
    var code, codes, range, unicode, unicodeMap, _i, _len;
    unicodeMap =
      "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo <<\n  /Registry (Adobe)\n  /Ordering (UCS)\n  /Supplement 0\n>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000><ffff>\nendcodespacerange";
    codes = Object.keys(map).sort(function(a, b) {
      return a - b;
    });

    range = [];
    for (_i = 0, _len = codes.length; _i < _len; _i++) {
      code = codes[_i];
      if (range.length >= 100) {
        unicodeMap +=
          "\n" +
          range.length +
          " beginbfchar\n" +
          range.join("\n") +
          "\nendbfchar";
        range = [];
      }

      if (
        map[code] !== undefined &&
        map[code] !== null &&
        typeof map[code].toString === "function"
      ) {
        unicode = ("0000" + map[code].toString(16)).slice(-4);
        code = ("0000" + (+code).toString(16)).slice(-4);
        range.push("<" + code + "><" + unicode + ">");
      }
    }

    if (range.length) {
      unicodeMap +=
        "\n" +
        range.length +
        " beginbfchar\n" +
        range.join("\n") +
        "\nendbfchar\n";
    }
    unicodeMap +=
      "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";
    return unicodeMap;
  };

  var identityHFunction = function(options) {
    var font = options.font;
    var out = options.out;
    var newObject = options.newObject;
    var putStream = options.putStream;

    if (
      font.metadata instanceof jsPDF.API.TTFFont &&
      font.encoding === "Identity-H"
    ) {
      //Tag with Identity-H
      var widths = font.metadata.Unicode.widths;
      var data = font.metadata.subset.encode(font.metadata.glyIdsUsed, 1);
      var pdfOutput = data;
      var pdfOutput2 = "";
      for (var i = 0; i < pdfOutput.length; i++) {
        pdfOutput2 += String.fromCharCode(pdfOutput[i]);
      }
      var fontTable = newObject();
      putStream({ data: pdfOutput2, addLength1: true, objectId: fontTable });
      out("endobj");

      var cmap = newObject();
      var cmapData = toUnicodeCmap(font.metadata.toUnicode);
      putStream({ data: cmapData, addLength1: true, objectId: cmap });
      out("endobj");

      var fontDescriptor = newObject();
      out("<<");
      out("/Type /FontDescriptor");
      out("/FontName /" + toPDFName(font.fontName));
      out("/FontFile2 " + fontTable + " 0 R");
      out("/FontBBox " + jsPDF.API.PDFObject.convert(font.metadata.bbox));
      out("/Flags " + font.metadata.flags);
      out("/StemV " + font.metadata.stemV);
      out("/ItalicAngle " + font.metadata.italicAngle);
      out("/Ascent " + font.metadata.ascender);
      out("/Descent " + font.metadata.decender);
      out("/CapHeight " + font.metadata.capHeight);
      out(">>");
      out("endobj");

      var DescendantFont = newObject();
      out("<<");
      out("/Type /Font");
      out("/BaseFont /" + toPDFName(font.fontName));
      out("/FontDescriptor " + fontDescriptor + " 0 R");
      out("/W " + jsPDF.API.PDFObject.convert(widths));
      out("/CIDToGIDMap /Identity");
      out("/DW 1000");
      out("/Subtype /CIDFontType2");
      out("/CIDSystemInfo");
      out("<<");
      out("/Supplement 0");
      out("/Registry (Adobe)");
      out("/Ordering (" + font.encoding + ")");
      out(">>");
      out(">>");
      out("endobj");

      font.objectNumber = newObject();
      out("<<");
      out("/Type /Font");
      out("/Subtype /Type0");
      out("/ToUnicode " + cmap + " 0 R");
      out("/BaseFont /" + toPDFName(font.fontName));
      out("/Encoding /" + font.encoding);
      out("/DescendantFonts [" + DescendantFont + " 0 R]");
      out(">>");
      out("endobj");

      font.isAlreadyPutted = true;
    }
  };

  jsPDFAPI.events.push([
    "putFont",
    function(args) {
      identityHFunction(args);
    }
  ]);

  var winAnsiEncodingFunction = function(options) {
    var font = options.font;
    var out = options.out;
    var newObject = options.newObject;
    var putStream = options.putStream;

    if (
      font.metadata instanceof jsPDF.API.TTFFont &&
      font.encoding === "WinAnsiEncoding"
    ) {
      //Tag with WinAnsi encoding
      var data = font.metadata.rawData;
      var pdfOutput = data;
      var pdfOutput2 = "";
      for (var i = 0; i < pdfOutput.length; i++) {
        pdfOutput2 += String.fromCharCode(pdfOutput[i]);
      }
      var fontTable = newObject();
      putStream({ data: pdfOutput2, addLength1: true, objectId: fontTable });
      out("endobj");

      var cmap = newObject();
      var cmapData = toUnicodeCmap(font.metadata.toUnicode);
      putStream({ data: cmapData, addLength1: true, objectId: cmap });
      out("endobj");

      var fontDescriptor = newObject();
      out("<<");
      out("/Descent " + font.metadata.decender);
      out("/CapHeight " + font.metadata.capHeight);
      out("/StemV " + font.metadata.stemV);
      out("/Type /FontDescriptor");
      out("/FontFile2 " + fontTable + " 0 R");
      out("/Flags 96");
      out("/FontBBox " + jsPDF.API.PDFObject.convert(font.metadata.bbox));
      out("/FontName /" + toPDFName(font.fontName));
      out("/ItalicAngle " + font.metadata.italicAngle);
      out("/Ascent " + font.metadata.ascender);
      out(">>");
      out("endobj");
      font.objectNumber = newObject();
      for (var j = 0; j < font.metadata.hmtx.widths.length; j++) {
        font.metadata.hmtx.widths[j] = parseInt(
          font.metadata.hmtx.widths[j] * (1000 / font.metadata.head.unitsPerEm)
        ); //Change the width of Em units to Point units.
      }
      out(
        "<</Subtype/TrueType/Type/Font/ToUnicode " +
          cmap +
          " 0 R/BaseFont/" +
          toPDFName(font.fontName) +
          "/FontDescriptor " +
          fontDescriptor +
          " 0 R" +
          "/Encoding/" +
          font.encoding +
          " /FirstChar 29 /LastChar 255 /Widths " +
          jsPDF.API.PDFObject.convert(font.metadata.hmtx.widths) +
          ">>"
      );
      out("endobj");
      font.isAlreadyPutted = true;
    }
  };

  jsPDFAPI.events.push([
    "putFont",
    function(args) {
      winAnsiEncodingFunction(args);
    }
  ]);

  var utf8TextFunction = function(args) {
    var text = args.text || "";
    var x = args.x;
    var y = args.y;
    var options = args.options || {};
    var mutex = args.mutex || {};

    var pdfEscape = mutex.pdfEscape;
    var activeFontKey = mutex.activeFontKey;
    var fonts = mutex.fonts;
    var key = activeFontKey;

    var str = "",
      s = 0,
      cmapConfirm;
    var strText = "";
    var encoding = fonts[key].encoding;

    if (fonts[key].encoding !== "Identity-H") {
      return {
        text: text,
        x: x,
        y: y,
        options: options,
        mutex: mutex
      };
    }
    strText = text;

    key = activeFontKey;
    if (Array.isArray(text)) {
      strText = text[0];
    }
    for (s = 0; s < strText.length; s += 1) {
      if (fonts[key].metadata.hasOwnProperty("cmap")) {
        cmapConfirm =
          fonts[key].metadata.cmap.unicode.codeMap[strText[s].charCodeAt(0)];
        /*
             if (Object.prototype.toString.call(text) === '[object Array]') {
                var i = 0;
               // for (i = 0; i < text.length; i += 1) {
                    if (Object.prototype.toString.call(text[s]) === '[object Array]') {
                        cmapConfirm = fonts[key].metadata.cmap.unicode.codeMap[strText[s][0].charCodeAt(0)]; //Make sure the cmap has the corresponding glyph id
                    } else {

                    }
                //}

            } else {
                cmapConfirm = fonts[key].metadata.cmap.unicode.codeMap[strText[s].charCodeAt(0)]; //Make sure the cmap has the corresponding glyph id
            }*/
      }
      if (!cmapConfirm) {
        if (
          strText[s].charCodeAt(0) < 256 &&
          fonts[key].metadata.hasOwnProperty("Unicode")
        ) {
          str += strText[s];
        } else {
          str += "";
        }
      } else {
        str += strText[s];
      }
    }
    var result = "";
    if (parseInt(key.slice(1)) < 14 || encoding === "WinAnsiEncoding") {
      //For the default 13 font
      result = pdfEscape(str, key)
        .split("")
        .map(function(cv) {
          return cv.charCodeAt(0).toString(16);
        })
        .join("");
    } else if (encoding === "Identity-H") {
      result = pdfEscape16(str, fonts[key]);
    }
    mutex.isHex = true;

    return {
      text: result,
      x: x,
      y: y,
      options: options,
      mutex: mutex
    };
  };

  var utf8EscapeFunction = function(parms) {
    var text = parms.text || "",
      x = parms.x,
      y = parms.y,
      options = parms.options,
      mutex = parms.mutex;
    var tmpText = [];
    var args = {
      text: text,
      x: x,
      y: y,
      options: options,
      mutex: mutex
    };

    if (Array.isArray(text)) {
      var i = 0;
      for (i = 0; i < text.length; i += 1) {
        if (Array.isArray(text[i])) {
          if (text[i].length === 3) {
            tmpText.push([
              utf8TextFunction(Object.assign({}, args, { text: text[i][0] }))
                .text,
              text[i][1],
              text[i][2]
            ]);
          } else {
            tmpText.push(
              utf8TextFunction(Object.assign({}, args, { text: text[i] })).text
            );
          }
        } else {
          tmpText.push(
            utf8TextFunction(Object.assign({}, args, { text: text[i] })).text
          );
        }
      }
      parms.text = tmpText;
    } else {
      parms.text = utf8TextFunction(
        Object.assign({}, args, { text: text })
      ).text;
    }
  };

  jsPDFAPI.events.push(["postProcessText", utf8EscapeFunction]);
})(jsPDF);

/**
 * @license
 * jsPDF virtual FileSystem functionality
 *
 * Licensed under the MIT License.
 * http://opensource.org/licenses/mit-license
 */

/**
 * Use the vFS to handle files
 *
 * @name vFS
 * @module
 */
(function(jsPDFAPI) {

  var _initializeVFS = function() {
    if (typeof this.internal.vFS === "undefined") {
      this.internal.vFS = {};
    }
    return true;
  };

  /**
   * Check if the file exists in the vFS
   *
   * @name existsFileInVFS
   * @function
   * @param {string} Possible filename in the vFS.
   * @returns {boolean}
   * @example
   * doc.existsFileInVFS("someFile.txt");
   */
  jsPDFAPI.existsFileInVFS = function(filename) {
    _initializeVFS.call(this);
    return typeof this.internal.vFS[filename] !== "undefined";
  };

  /**
   * Add a file to the vFS
   *
   * @name addFileToVFS
   * @function
   * @param {string} filename The name of the file which should be added.
   * @param {string} filecontent The content of the file.
   * @returns {jsPDF}
   * @example
   * doc.addFileToVFS("someFile.txt", "BADFACE1");
   */
  jsPDFAPI.addFileToVFS = function(filename, filecontent) {
    _initializeVFS.call(this);
    this.internal.vFS[filename] = filecontent;
    return this;
  };

  /**
   * Get the file from the vFS
   *
   * @name getFileFromVFS
   * @function
   * @param {string} The name of the file which gets requested.
   * @returns {string}
   * @example
   * doc.getFileFromVFS("someFile.txt");
   */
  jsPDFAPI.getFileFromVFS = function(filename) {
    _initializeVFS.call(this);

    if (typeof this.internal.vFS[filename] !== "undefined") {
      return this.internal.vFS[filename];
    }
    return null;
  };
})(jsPDF.API);

/**
 * @license
 * Unicode Bidi Engine based on the work of Alex Shensis (@asthensis)
 * MIT License
 */

(function(jsPDF) {
  /**
   * Table of Unicode types.
   *
   * Generated by:
   *
   * var bidi = require("./bidi/index");
   * var bidi_accumulate = bidi.slice(0, 256).concat(bidi.slice(0x0500, 0x0500 + 256 * 3)).
   * concat(bidi.slice(0x2000, 0x2000 + 256)).concat(bidi.slice(0xFB00, 0xFB00 + 256)).
   * concat(bidi.slice(0xFE00, 0xFE00 + 2 * 256));
   *
   * for( var i = 0; i < bidi_accumulate.length; i++) {
   * 	if(bidi_accumulate[i] === undefined || bidi_accumulate[i] === 'ON')
   * 		bidi_accumulate[i] = 'N'; //mark as neutral to conserve space and substitute undefined
   * }
   * var bidiAccumulateStr = 'return [ "' + bidi_accumulate.toString().replace(/,/g, '", "') + '" ];';
   * require("fs").writeFile('unicode-types.js', bidiAccumulateStr);
   *
   * Based on:
   * https://github.com/mathiasbynens/unicode-8.0.0
   */
  var bidiUnicodeTypes = [
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "S",
    "B",
    "S",
    "WS",
    "B",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "B",
    "B",
    "B",
    "S",
    "WS",
    "N",
    "N",
    "ET",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "ES",
    "CS",
    "ES",
    "CS",
    "CS",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "CS",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "B",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "CS",
    "N",
    "ET",
    "ET",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "L",
    "N",
    "N",
    "BN",
    "N",
    "N",
    "ET",
    "ET",
    "EN",
    "EN",
    "N",
    "L",
    "N",
    "N",
    "N",
    "EN",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "ET",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "R",
    "NSM",
    "R",
    "NSM",
    "NSM",
    "R",
    "NSM",
    "NSM",
    "R",
    "NSM",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "N",
    "N",
    "N",
    "N",
    "N",
    "R",
    "R",
    "R",
    "R",
    "R",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "N",
    "N",
    "AL",
    "ET",
    "ET",
    "AL",
    "CS",
    "AL",
    "N",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AL",
    "AL",
    "N",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "AN",
    "ET",
    "AN",
    "AN",
    "AL",
    "AL",
    "AL",
    "NSM",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AN",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AL",
    "AL",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "N",
    "AL",
    "AL",
    "NSM",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "N",
    "N",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "AL",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "R",
    "R",
    "N",
    "N",
    "N",
    "N",
    "R",
    "N",
    "N",
    "N",
    "N",
    "N",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "WS",
    "BN",
    "BN",
    "BN",
    "L",
    "R",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "WS",
    "B",
    "LRE",
    "RLE",
    "PDF",
    "LRO",
    "RLO",
    "CS",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "CS",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "WS",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "N",
    "LRI",
    "RLI",
    "FSI",
    "PDI",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "BN",
    "EN",
    "L",
    "N",
    "N",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "ES",
    "ES",
    "N",
    "N",
    "N",
    "L",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "ES",
    "ES",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "R",
    "NSM",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "ES",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "N",
    "R",
    "R",
    "R",
    "R",
    "R",
    "N",
    "R",
    "N",
    "R",
    "R",
    "N",
    "R",
    "R",
    "N",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "NSM",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "CS",
    "N",
    "CS",
    "N",
    "N",
    "CS",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "ET",
    "N",
    "N",
    "ES",
    "ES",
    "N",
    "N",
    "N",
    "N",
    "N",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "N",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "AL",
    "N",
    "N",
    "BN",
    "N",
    "N",
    "N",
    "ET",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "ES",
    "CS",
    "ES",
    "CS",
    "CS",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "EN",
    "CS",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "N",
    "N",
    "L",
    "L",
    "L",
    "N",
    "N",
    "N",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "ET",
    "ET",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N",
    "N"
  ];

  /**
   * Unicode Bidi algorithm compliant Bidi engine.
   * For reference see http://unicode.org/reports/tr9/
   */

  /**
   * constructor ( options )
   *
   * Initializes Bidi engine
   *
   * @param {Object} See 'setOptions' below for detailed description.
   * options are cashed between invocation of 'doBidiReorder' method
   *
   * sample usage pattern of BidiEngine:
   * var opt = {
   * 	isInputVisual: true,
   * 	isInputRtl: false,
   * 	isOutputVisual: false,
   * 	isOutputRtl: false,
   * 	isSymmetricSwapping: true
   * }
   * var sourceToTarget = [], levels = [];
   * var bidiEng = Globalize.bidiEngine(opt);
   * var src = "text string to be reordered";
   * var ret = bidiEng.doBidiReorder(src, sourceToTarget, levels);
   */

  jsPDF.__bidiEngine__ = jsPDF.prototype.__bidiEngine__ = function(options) {
    var _UNICODE_TYPES = _bidiUnicodeTypes;

    var _STATE_TABLE_LTR = [
      [0, 3, 0, 1, 0, 0, 0],
      [0, 3, 0, 1, 2, 2, 0],
      [0, 3, 0, 0x11, 2, 0, 1],
      [0, 3, 5, 5, 4, 1, 0],
      [0, 3, 0x15, 0x15, 4, 0, 1],
      [0, 3, 5, 5, 4, 2, 0]
    ];

    var _STATE_TABLE_RTL = [
      [2, 0, 1, 1, 0, 1, 0],
      [2, 0, 1, 1, 0, 2, 0],
      [2, 0, 2, 1, 3, 2, 0],
      [2, 0, 2, 0x21, 3, 1, 1]
    ];

    var _TYPE_NAMES_MAP = { L: 0, R: 1, EN: 2, AN: 3, N: 4, B: 5, S: 6 };

    var _UNICODE_RANGES_MAP = {
      0: 0,
      5: 1,
      6: 2,
      7: 3,
      0x20: 4,
      0xfb: 5,
      0xfe: 6,
      0xff: 7
    };

    var _SWAP_TABLE = [
      "\u0028",
      "\u0029",
      "\u0028",
      "\u003C",
      "\u003E",
      "\u003C",
      "\u005B",
      "\u005D",
      "\u005B",
      "\u007B",
      "\u007D",
      "\u007B",
      "\u00AB",
      "\u00BB",
      "\u00AB",
      "\u2039",
      "\u203A",
      "\u2039",
      "\u2045",
      "\u2046",
      "\u2045",
      "\u207D",
      "\u207E",
      "\u207D",
      "\u208D",
      "\u208E",
      "\u208D",
      "\u2264",
      "\u2265",
      "\u2264",
      "\u2329",
      "\u232A",
      "\u2329",
      "\uFE59",
      "\uFE5A",
      "\uFE59",
      "\uFE5B",
      "\uFE5C",
      "\uFE5B",
      "\uFE5D",
      "\uFE5E",
      "\uFE5D",
      "\uFE64",
      "\uFE65",
      "\uFE64"
    ];

    var _LTR_RANGES_REG_EXPR = new RegExp(
      /^([1-4|9]|1[0-9]|2[0-9]|3[0168]|4[04589]|5[012]|7[78]|159|16[0-9]|17[0-2]|21[569]|22[03489]|250)$/
    );

    var _lastArabic = false,
      _hasUbatB,
      _hasUbatS,
      DIR_LTR = 0,
      DIR_RTL = 1,
      _isInVisual,
      _isInRtl,
      _isOutVisual,
      _isOutRtl,
      _isSymmetricSwapping,
      _dir = DIR_LTR;

    this.__bidiEngine__ = {};

    var _init = function(text, sourceToTargetMap) {
      if (sourceToTargetMap) {
        for (var i = 0; i < text.length; i++) {
          sourceToTargetMap[i] = i;
        }
      }
      if (_isInRtl === undefined) {
        _isInRtl = _isContextualDirRtl(text);
      }
      if (_isOutRtl === undefined) {
        _isOutRtl = _isContextualDirRtl(text);
      }
    };

    // for reference see 3.2 in http://unicode.org/reports/tr9/
    //
    var _getCharType = function(ch) {
      var charCode = ch.charCodeAt(),
        range = charCode >> 8,
        rangeIdx = _UNICODE_RANGES_MAP[range];

      if (rangeIdx !== undefined) {
        return _UNICODE_TYPES[rangeIdx * 256 + (charCode & 0xff)];
      } else if (range === 0xfc || range === 0xfd) {
        return "AL";
      } else if (_LTR_RANGES_REG_EXPR.test(range)) {
        //unlikely case
        return "L";
      } else if (range === 8) {
        // even less likely
        return "R";
      }
      return "N"; //undefined type, mark as neutral
    };

    var _isContextualDirRtl = function(text) {
      for (var i = 0, charType; i < text.length; i++) {
        charType = _getCharType(text.charAt(i));
        if (charType === "L") {
          return false;
        } else if (charType === "R") {
          return true;
        }
      }
      return false;
    };

    // for reference see 3.3.4 & 3.3.5 in http://unicode.org/reports/tr9/
    //
    var _resolveCharType = function(chars, types, resolvedTypes, index) {
      var cType = types[index],
        wType,
        nType,
        i,
        len;
      switch (cType) {
        case "L":
        case "R":
          _lastArabic = false;
          break;
        case "N":
        case "AN":
          break;

        case "EN":
          if (_lastArabic) {
            cType = "AN";
          }
          break;

        case "AL":
          _lastArabic = true;
          cType = "R";
          break;

        case "WS":
          cType = "N";
          break;

        case "CS":
          if (
            index < 1 ||
            index + 1 >= types.length ||
            ((wType = resolvedTypes[index - 1]) !== "EN" && wType !== "AN") ||
            ((nType = types[index + 1]) !== "EN" && nType !== "AN")
          ) {
            cType = "N";
          } else if (_lastArabic) {
            nType = "AN";
          }
          cType = nType === wType ? nType : "N";
          break;

        case "ES":
          wType = index > 0 ? resolvedTypes[index - 1] : "B";
          cType =
            wType === "EN" &&
            index + 1 < types.length &&
            types[index + 1] === "EN"
              ? "EN"
              : "N";
          break;

        case "ET":
          if (index > 0 && resolvedTypes[index - 1] === "EN") {
            cType = "EN";
            break;
          } else if (_lastArabic) {
            cType = "N";
            break;
          }
          i = index + 1;
          len = types.length;
          while (i < len && types[i] === "ET") {
            i++;
          }
          if (i < len && types[i] === "EN") {
            cType = "EN";
          } else {
            cType = "N";
          }
          break;

        case "NSM":
          if (_isInVisual && !_isInRtl) {
            //V->L
            len = types.length;
            i = index + 1;
            while (i < len && types[i] === "NSM") {
              i++;
            }
            if (i < len) {
              var c = chars[index];
              var rtlCandidate = (c >= 0x0591 && c <= 0x08ff) || c === 0xfb1e;
              wType = types[i];
              if (rtlCandidate && (wType === "R" || wType === "AL")) {
                cType = "R";
                break;
              }
            }
          }
          if (index < 1 || (wType = types[index - 1]) === "B") {
            cType = "N";
          } else {
            cType = resolvedTypes[index - 1];
          }
          break;

        case "B":
          _lastArabic = false;
          _hasUbatB = true;
          cType = _dir;
          break;

        case "S":
          _hasUbatS = true;
          cType = "N";
          break;

        case "LRE":
        case "RLE":
        case "LRO":
        case "RLO":
        case "PDF":
          _lastArabic = false;
          break;
        case "BN":
          cType = "N";
          break;
      }
      return cType;
    };

    var _handleUbatS = function(types, levels, length) {
      for (var i = 0; i < length; i++) {
        if (types[i] === "S") {
          levels[i] = _dir;
          for (var j = i - 1; j >= 0; j--) {
            if (types[j] === "WS") {
              levels[j] = _dir;
            } else {
              break;
            }
          }
        }
      }
    };

    var _invertString = function(text, sourceToTargetMap, levels) {
      var charArray = text.split("");
      if (levels) {
        _computeLevels(charArray, levels, { hiLevel: _dir });
      }
      charArray.reverse();
      sourceToTargetMap && sourceToTargetMap.reverse();
      return charArray.join("");
    };

    // For reference see 3.3 in http://unicode.org/reports/tr9/
    //
    var _computeLevels = function(chars, levels, params) {
      var action,
        condition,
        i,
        index,
        newLevel,
        prevState,
        condPos = -1,
        len = chars.length,
        newState = 0,
        resolvedTypes = [],
        stateTable = _dir ? _STATE_TABLE_RTL : _STATE_TABLE_LTR,
        types = [];

      _lastArabic = false;
      _hasUbatB = false;
      _hasUbatS = false;
      for (i = 0; i < len; i++) {
        types[i] = _getCharType(chars[i]);
      }
      for (index = 0; index < len; index++) {
        prevState = newState;
        resolvedTypes[index] = _resolveCharType(
          chars,
          types,
          resolvedTypes,
          index
        );
        newState = stateTable[prevState][_TYPE_NAMES_MAP[resolvedTypes[index]]];
        action = newState & 0xf0;
        newState &= 0x0f;
        levels[index] = newLevel = stateTable[newState][5];
        if (action > 0) {
          if (action === 0x10) {
            for (i = condPos; i < index; i++) {
              levels[i] = 1;
            }
            condPos = -1;
          } else {
            condPos = -1;
          }
        }
        condition = stateTable[newState][6];
        if (condition) {
          if (condPos === -1) {
            condPos = index;
          }
        } else {
          if (condPos > -1) {
            for (i = condPos; i < index; i++) {
              levels[i] = newLevel;
            }
            condPos = -1;
          }
        }
        if (types[index] === "B") {
          levels[index] = 0;
        }
        params.hiLevel |= newLevel;
      }
      if (_hasUbatS) {
        _handleUbatS(types, levels, len);
      }
    };

    // for reference see 3.4 in http://unicode.org/reports/tr9/
    //
    var _invertByLevel = function(
      level,
      charArray,
      sourceToTargetMap,
      levels,
      params
    ) {
      if (params.hiLevel < level) {
        return;
      }
      if (level === 1 && _dir === DIR_RTL && !_hasUbatB) {
        charArray.reverse();
        sourceToTargetMap && sourceToTargetMap.reverse();
        return;
      }
      var ch,
        high,
        end,
        low,
        len = charArray.length,
        start = 0;

      while (start < len) {
        if (levels[start] >= level) {
          end = start + 1;
          while (end < len && levels[end] >= level) {
            end++;
          }
          for (low = start, high = end - 1; low < high; low++, high--) {
            ch = charArray[low];
            charArray[low] = charArray[high];
            charArray[high] = ch;
            if (sourceToTargetMap) {
              ch = sourceToTargetMap[low];
              sourceToTargetMap[low] = sourceToTargetMap[high];
              sourceToTargetMap[high] = ch;
            }
          }
          start = end;
        }
        start++;
      }
    };

    // for reference see 7 & BD16 in http://unicode.org/reports/tr9/
    //
    var _symmetricSwap = function(charArray, levels, params) {
      if (params.hiLevel !== 0 && _isSymmetricSwapping) {
        for (var i = 0, index; i < charArray.length; i++) {
          if (levels[i] === 1) {
            index = _SWAP_TABLE.indexOf(charArray[i]);
            if (index >= 0) {
              charArray[i] = _SWAP_TABLE[index + 1];
            }
          }
        }
      }
    };

    var _reorder = function(text, sourceToTargetMap, levels) {
      var charArray = text.split(""),
        params = { hiLevel: _dir };

      if (!levels) {
        levels = [];
      }
      _computeLevels(charArray, levels, params);
      _symmetricSwap(charArray, levels, params);
      _invertByLevel(DIR_RTL + 1, charArray, sourceToTargetMap, levels, params);
      _invertByLevel(DIR_RTL, charArray, sourceToTargetMap, levels, params);
      return charArray.join("");
    };

    // doBidiReorder( text, sourceToTargetMap, levels )
    // Performs Bidi reordering by implementing Unicode Bidi algorithm.
    // Returns reordered string
    // @text [String]:
    // - input string to be reordered, this is input parameter
    // $sourceToTargetMap [Array] (optional)
    // - resultant mapping between input and output strings, this is output parameter
    // $levels [Array] (optional)
    // - array of calculated Bidi levels, , this is output parameter
    this.__bidiEngine__.doBidiReorder = function(
      text,
      sourceToTargetMap,
      levels
    ) {
      _init(text, sourceToTargetMap);
      if (!_isInVisual && _isOutVisual && !_isOutRtl) {
        // LLTR->VLTR, LRTL->VLTR
        _dir = _isInRtl ? DIR_RTL : DIR_LTR;
        text = _reorder(text, sourceToTargetMap, levels);
      } else if (_isInVisual && _isOutVisual && _isInRtl ^ _isOutRtl) {
        // VRTL->VLTR, VLTR->VRTL
        _dir = _isInRtl ? DIR_RTL : DIR_LTR;
        text = _invertString(text, sourceToTargetMap, levels);
      } else if (!_isInVisual && _isOutVisual && _isOutRtl) {
        // LLTR->VRTL, LRTL->VRTL
        _dir = _isInRtl ? DIR_RTL : DIR_LTR;
        text = _reorder(text, sourceToTargetMap, levels);
        text = _invertString(text, sourceToTargetMap);
      } else if (_isInVisual && !_isInRtl && !_isOutVisual && !_isOutRtl) {
        // VLTR->LLTR
        _dir = DIR_LTR;
        text = _reorder(text, sourceToTargetMap, levels);
      } else if (_isInVisual && !_isOutVisual && _isInRtl ^ _isOutRtl) {
        // VLTR->LRTL, VRTL->LLTR
        text = _invertString(text, sourceToTargetMap);
        if (_isInRtl) {
          //LLTR -> VLTR
          _dir = DIR_LTR;
          text = _reorder(text, sourceToTargetMap, levels);
        } else {
          //LRTL -> VRTL
          _dir = DIR_RTL;
          text = _reorder(text, sourceToTargetMap, levels);
          text = _invertString(text, sourceToTargetMap);
        }
      } else if (_isInVisual && _isInRtl && !_isOutVisual && _isOutRtl) {
        //  VRTL->LRTL
        _dir = DIR_RTL;
        text = _reorder(text, sourceToTargetMap, levels);
        text = _invertString(text, sourceToTargetMap);
      } else if (!_isInVisual && !_isOutVisual && _isInRtl ^ _isOutRtl) {
        // LRTL->LLTR, LLTR->LRTL
        var isSymmetricSwappingOrig = _isSymmetricSwapping;
        if (_isInRtl) {
          //LRTL->LLTR
          _dir = DIR_RTL;
          text = _reorder(text, sourceToTargetMap, levels);
          _dir = DIR_LTR;
          _isSymmetricSwapping = false;
          text = _reorder(text, sourceToTargetMap, levels);
          _isSymmetricSwapping = isSymmetricSwappingOrig;
        } else {
          //LLTR->LRTL
          _dir = DIR_LTR;
          text = _reorder(text, sourceToTargetMap, levels);
          text = _invertString(text, sourceToTargetMap);
          _dir = DIR_RTL;
          _isSymmetricSwapping = false;
          text = _reorder(text, sourceToTargetMap, levels);
          _isSymmetricSwapping = isSymmetricSwappingOrig;
          text = _invertString(text, sourceToTargetMap);
        }
      }
      return text;
    };

    /**
     * @name setOptions( options )
     * @function
     * Sets options for Bidi conversion
     * @param {Object}:
     * - isInputVisual {boolean} (defaults to false): allowed values: true(Visual mode), false(Logical mode)
     * - isInputRtl {boolean}: allowed values true(Right-to-left direction), false (Left-to-right directiion), undefined(Contectual direction, i.e.direction defined by first strong character of input string)
     * - isOutputVisual {boolean} (defaults to false): allowed values: true(Visual mode), false(Logical mode)
     * - isOutputRtl {boolean}: allowed values true(Right-to-left direction), false (Left-to-right directiion), undefined(Contectual direction, i.e.direction defined by first strong characterof input string)
     * - isSymmetricSwapping {boolean} (defaults to false): allowed values true(needs symmetric swapping), false (no need in symmetric swapping),
     */
    this.__bidiEngine__.setOptions = function(options) {
      if (options) {
        _isInVisual = options.isInputVisual;
        _isOutVisual = options.isOutputVisual;
        _isInRtl = options.isInputRtl;
        _isOutRtl = options.isOutputRtl;
        _isSymmetricSwapping = options.isSymmetricSwapping;
      }
    };

    this.__bidiEngine__.setOptions(options);
    return this.__bidiEngine__;
  };

  var _bidiUnicodeTypes = bidiUnicodeTypes;

  var bidiEngine = new jsPDF.__bidiEngine__({ isInputVisual: true });

  var bidiEngineFunction = function(args) {
    var text = args.text;
    args.x;
    args.y;
    var options = args.options || {};
    args.mutex || {};
    options.lang;
    var tmpText = [];

    options.isInputVisual =
      typeof options.isInputVisual === "boolean" ? options.isInputVisual : true;
    bidiEngine.setOptions(options);

    if (Object.prototype.toString.call(text) === "[object Array]") {
      var i = 0;
      tmpText = [];
      for (i = 0; i < text.length; i += 1) {
        if (Object.prototype.toString.call(text[i]) === "[object Array]") {
          tmpText.push([
            bidiEngine.doBidiReorder(text[i][0]),
            text[i][1],
            text[i][2]
          ]);
        } else {
          tmpText.push([bidiEngine.doBidiReorder(text[i])]);
        }
      }
      args.text = tmpText;
    } else {
      args.text = bidiEngine.doBidiReorder(text);
    }
    bidiEngine.setOptions({ isInputVisual: true });
  };

  jsPDF.API.events.push(["postProcessText", bidiEngineFunction]);
})(jsPDF);

/* eslint-disable no-control-regex */

jsPDF.API.TTFFont = (function() {
  /************************************************************************/
  /* function : open                                                       */
  /* comment : Decode the encoded ttf content and create a TTFFont object. */
  /************************************************************************/
  TTFFont.open = function(file) {
    return new TTFFont(file);
  };
  /***************************************************************/
  /* function : TTFFont gernerator                               */
  /* comment : Decode TTF contents are parsed, Data,             */
  /* Subset object is created, and registerTTF function is called.*/
  /***************************************************************/
  function TTFFont(rawData) {
    var data;
    this.rawData = rawData;
    data = this.contents = new Data(rawData);
    this.contents.pos = 4;
    if (data.readString(4) === "ttcf") {
      throw new Error("TTCF not supported.");
    } else {
      data.pos = 0;
      this.parse();
      this.subset = new Subset(this);
      this.registerTTF();
    }
  }
  /********************************************************/
  /* function : parse                                     */
  /* comment : TTF Parses the file contents by each table.*/
  /********************************************************/
  TTFFont.prototype.parse = function() {
    this.directory = new Directory(this.contents);
    this.head = new HeadTable(this);
    this.name = new NameTable(this);
    this.cmap = new CmapTable(this);
    this.toUnicode = {};
    this.hhea = new HheaTable(this);
    this.maxp = new MaxpTable(this);
    this.hmtx = new HmtxTable(this);
    this.post = new PostTable(this);
    this.os2 = new OS2Table(this);
    this.loca = new LocaTable(this);
    this.glyf = new GlyfTable(this);
    this.ascender =
      (this.os2.exists && this.os2.ascender) || this.hhea.ascender;
    this.decender =
      (this.os2.exists && this.os2.decender) || this.hhea.decender;
    this.lineGap = (this.os2.exists && this.os2.lineGap) || this.hhea.lineGap;
    return (this.bbox = [
      this.head.xMin,
      this.head.yMin,
      this.head.xMax,
      this.head.yMax
    ]);
  };
  /***************************************************************/
  /* function : registerTTF                                      */
  /* comment : Get the value to assign pdf font descriptors.     */
  /***************************************************************/
  TTFFont.prototype.registerTTF = function() {
    var e, hi, low, raw, _ref;
    this.scaleFactor = 1000.0 / this.head.unitsPerEm;
    this.bbox = function() {
      var _i, _len, _ref, _results;
      _ref = this.bbox;
      _results = [];
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        e = _ref[_i];
        _results.push(Math.round(e * this.scaleFactor));
      }
      return _results;
    }.call(this);
    this.stemV = 0;
    if (this.post.exists) {
      raw = this.post.italic_angle;
      hi = raw >> 16;
      low = raw & 0xff;
      if ((hi & 0x8000) !== 0) {
        hi = -((hi ^ 0xffff) + 1);
      }
      this.italicAngle = +("" + hi + "." + low);
    } else {
      this.italicAngle = 0;
    }
    this.ascender = Math.round(this.ascender * this.scaleFactor);
    this.decender = Math.round(this.decender * this.scaleFactor);
    this.lineGap = Math.round(this.lineGap * this.scaleFactor);
    this.capHeight = (this.os2.exists && this.os2.capHeight) || this.ascender;
    this.xHeight = (this.os2.exists && this.os2.xHeight) || 0;
    this.familyClass = ((this.os2.exists && this.os2.familyClass) || 0) >> 8;
    this.isSerif =
      (_ref = this.familyClass) === 1 ||
      _ref === 2 ||
      _ref === 3 ||
      _ref === 4 ||
      _ref === 5 ||
      _ref === 7;
    this.isScript = this.familyClass === 10;
    this.flags = 0;
    if (this.post.isFixedPitch) {
      this.flags |= 1 << 0;
    }
    if (this.isSerif) {
      this.flags |= 1 << 1;
    }
    if (this.isScript) {
      this.flags |= 1 << 3;
    }
    if (this.italicAngle !== 0) {
      this.flags |= 1 << 6;
    }
    this.flags |= 1 << 5;
    if (!this.cmap.unicode) {
      throw new Error("No unicode cmap for font");
    }
  };
  TTFFont.prototype.characterToGlyph = function(character) {
    var _ref;
    return (
      ((_ref = this.cmap.unicode) != null ? _ref.codeMap[character] : void 0) ||
      0
    );
  };
  TTFFont.prototype.widthOfGlyph = function(glyph) {
    var scale;
    scale = 1000.0 / this.head.unitsPerEm;
    return this.hmtx.forGlyph(glyph).advance * scale;
  };
  TTFFont.prototype.widthOfString = function(string, size, charSpace) {
    var charCode, i, scale, width, _ref;
    string = "" + string;
    width = 0;
    for (
      i = 0, _ref = string.length;
      0 <= _ref ? i < _ref : i > _ref;
      i = 0 <= _ref ? ++i : --i
    ) {
      charCode = string.charCodeAt(i);
      width +=
        this.widthOfGlyph(this.characterToGlyph(charCode)) +
          charSpace * (1000 / size) || 0;
    }
    scale = size / 1000;
    return width * scale;
  };
  TTFFont.prototype.lineHeight = function(size, includeGap) {
    var gap;
    if (includeGap == null) {
      includeGap = false;
    }
    gap = includeGap ? this.lineGap : 0;
    return ((this.ascender + gap - this.decender) / 1000) * size;
  };
  return TTFFont;
})();

/************************************************************************************************/
/* function : Data                                                                              */
/* comment : The ttf data decoded and stored in an array is read and written to the Data object.*/
/************************************************************************************************/
var Data = (function() {
  function Data(data) {
    this.data = data != null ? data : [];
    this.pos = 0;
    this.length = this.data.length;
  }
  Data.prototype.readByte = function() {
    return this.data[this.pos++];
  };
  Data.prototype.writeByte = function(byte) {
    return (this.data[this.pos++] = byte);
  };
  Data.prototype.readUInt32 = function() {
    var b1, b2, b3, b4;
    b1 = this.readByte() * 0x1000000;
    b2 = this.readByte() << 16;
    b3 = this.readByte() << 8;
    b4 = this.readByte();
    return b1 + b2 + b3 + b4;
  };
  Data.prototype.writeUInt32 = function(val) {
    this.writeByte((val >>> 24) & 0xff);
    this.writeByte((val >> 16) & 0xff);
    this.writeByte((val >> 8) & 0xff);
    return this.writeByte(val & 0xff);
  };
  Data.prototype.readInt32 = function() {
    var int;
    int = this.readUInt32();
    if (int >= 0x80000000) {
      return int - 0x100000000;
    } else {
      return int;
    }
  };
  Data.prototype.writeInt32 = function(val) {
    if (val < 0) {
      val += 0x100000000;
    }
    return this.writeUInt32(val);
  };
  Data.prototype.readUInt16 = function() {
    var b1, b2;
    b1 = this.readByte() << 8;
    b2 = this.readByte();
    return b1 | b2;
  };
  Data.prototype.writeUInt16 = function(val) {
    this.writeByte((val >> 8) & 0xff);
    return this.writeByte(val & 0xff);
  };
  Data.prototype.readInt16 = function() {
    var int;
    int = this.readUInt16();
    if (int >= 0x8000) {
      return int - 0x10000;
    } else {
      return int;
    }
  };
  Data.prototype.writeInt16 = function(val) {
    if (val < 0) {
      val += 0x10000;
    }
    return this.writeUInt16(val);
  };
  Data.prototype.readString = function(length) {
    var i, ret;
    ret = [];
    for (
      i = 0;
      0 <= length ? i < length : i > length;
      i = 0 <= length ? ++i : --i
    ) {
      ret[i] = String.fromCharCode(this.readByte());
    }
    return ret.join("");
  };
  Data.prototype.writeString = function(val) {
    var i, _ref, _results;
    _results = [];
    for (
      i = 0, _ref = val.length;
      0 <= _ref ? i < _ref : i > _ref;
      i = 0 <= _ref ? ++i : --i
    ) {
      _results.push(this.writeByte(val.charCodeAt(i)));
    }
    return _results;
  };
  /*Data.prototype.stringAt = function (pos, length) {
            this.pos = pos;
            return this.readString(length);
        };*/
  Data.prototype.readShort = function() {
    return this.readInt16();
  };
  Data.prototype.writeShort = function(val) {
    return this.writeInt16(val);
  };
  Data.prototype.readLongLong = function() {
    var b1, b2, b3, b4, b5, b6, b7, b8;
    b1 = this.readByte();
    b2 = this.readByte();
    b3 = this.readByte();
    b4 = this.readByte();
    b5 = this.readByte();
    b6 = this.readByte();
    b7 = this.readByte();
    b8 = this.readByte();
    if (b1 & 0x80) {
      return (
        ((b1 ^ 0xff) * 0x100000000000000 +
          (b2 ^ 0xff) * 0x1000000000000 +
          (b3 ^ 0xff) * 0x10000000000 +
          (b4 ^ 0xff) * 0x100000000 +
          (b5 ^ 0xff) * 0x1000000 +
          (b6 ^ 0xff) * 0x10000 +
          (b7 ^ 0xff) * 0x100 +
          (b8 ^ 0xff) +
          1) *
        -1
      );
    }
    return (
      b1 * 0x100000000000000 +
      b2 * 0x1000000000000 +
      b3 * 0x10000000000 +
      b4 * 0x100000000 +
      b5 * 0x1000000 +
      b6 * 0x10000 +
      b7 * 0x100 +
      b8
    );
  };
  Data.prototype.writeLongLong = function(val) {
    var high, low;
    high = Math.floor(val / 0x100000000);
    low = val & 0xffffffff;
    this.writeByte((high >> 24) & 0xff);
    this.writeByte((high >> 16) & 0xff);
    this.writeByte((high >> 8) & 0xff);
    this.writeByte(high & 0xff);
    this.writeByte((low >> 24) & 0xff);
    this.writeByte((low >> 16) & 0xff);
    this.writeByte((low >> 8) & 0xff);
    return this.writeByte(low & 0xff);
  };
  Data.prototype.readInt = function() {
    return this.readInt32();
  };
  Data.prototype.writeInt = function(val) {
    return this.writeInt32(val);
  };
  /*Data.prototype.slice = function (start, end) {
            return this.data.slice(start, end);
        };*/
  Data.prototype.read = function(bytes) {
    var buf, i;
    buf = [];
    for (
      i = 0;
      0 <= bytes ? i < bytes : i > bytes;
      i = 0 <= bytes ? ++i : --i
    ) {
      buf.push(this.readByte());
    }
    return buf;
  };
  Data.prototype.write = function(bytes) {
    var byte, i, _len, _results;
    _results = [];
    for (i = 0, _len = bytes.length; i < _len; i++) {
      byte = bytes[i];
      _results.push(this.writeByte(byte));
    }
    return _results;
  };
  return Data;
})();

var Directory = (function() {
  var checksum;

  /*****************************************************************************************************/
  /* function : Directory generator                                                                    */
  /* comment : Initialize the offset, tag, length, and checksum for each table for the font to be used.*/
  /*****************************************************************************************************/
  function Directory(data) {
    var entry, i, _ref;
    this.scalarType = data.readInt();
    this.tableCount = data.readShort();
    this.searchRange = data.readShort();
    this.entrySelector = data.readShort();
    this.rangeShift = data.readShort();
    this.tables = {};
    for (
      i = 0, _ref = this.tableCount;
      0 <= _ref ? i < _ref : i > _ref;
      i = 0 <= _ref ? ++i : --i
    ) {
      entry = {
        tag: data.readString(4),
        checksum: data.readInt(),
        offset: data.readInt(),
        length: data.readInt()
      };
      this.tables[entry.tag] = entry;
    }
  }
  /********************************************************************************************************/
  /* function : encode                                                                                    */
  /* comment : It encodes and stores the font table object and information used for the directory object. */
  /********************************************************************************************************/
  Directory.prototype.encode = function(tables) {
    var adjustment,
      directory,
      directoryLength,
      entrySelector,
      headOffset,
      log2,
      offset,
      rangeShift,
      searchRange,
      sum,
      table,
      tableCount,
      tableData,
      tag;
    tableCount = Object.keys(tables).length;
    log2 = Math.log(2);
    searchRange = Math.floor(Math.log(tableCount) / log2) * 16;
    entrySelector = Math.floor(searchRange / log2);
    rangeShift = tableCount * 16 - searchRange;
    directory = new Data();
    directory.writeInt(this.scalarType);
    directory.writeShort(tableCount);
    directory.writeShort(searchRange);
    directory.writeShort(entrySelector);
    directory.writeShort(rangeShift);
    directoryLength = tableCount * 16;
    offset = directory.pos + directoryLength;
    headOffset = null;
    tableData = [];
    for (tag in tables) {
      table = tables[tag];
      directory.writeString(tag);
      directory.writeInt(checksum(table));
      directory.writeInt(offset);
      directory.writeInt(table.length);
      tableData = tableData.concat(table);
      if (tag === "head") {
        headOffset = offset;
      }
      offset += table.length;
      while (offset % 4) {
        tableData.push(0);
        offset++;
      }
    }
    directory.write(tableData);
    sum = checksum(directory.data);
    adjustment = 0xb1b0afba - sum;
    directory.pos = headOffset + 8;
    directory.writeUInt32(adjustment);
    return directory.data;
  };
  /***************************************************************/
  /* function : checksum                                         */
  /* comment : Duplicate the table for the tag.                  */
  /***************************************************************/
  checksum = function(data) {
    var i, sum, tmp, _ref;
    data = __slice.call(data);
    while (data.length % 4) {
      data.push(0);
    }
    tmp = new Data(data);
    sum = 0;
    for (i = 0, _ref = data.length; i < _ref; i = i += 4) {
      sum += tmp.readUInt32();
    }
    return sum & 0xffffffff;
  };
  return Directory;
})();

var Table,
  __hasProp = {}.hasOwnProperty,
  __extends = function(child, parent) {
    for (var key in parent) {
      if (__hasProp.call(parent, key)) child[key] = parent[key];
    }

    function ctor() {
      this.constructor = child;
    }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor();
    child.__super__ = parent.prototype;
    return child;
  };

/***************************************************************/
/* function : Table                                            */
/* comment : Save info for each table, and parse the table.    */
/***************************************************************/
Table = (function() {
  function Table(file) {
    var info;
    this.file = file;
    info = this.file.directory.tables[this.tag];
    this.exists = !!info;
    if (info) {
      (this.offset = info.offset), (this.length = info.length);
      this.parse(this.file.contents);
    }
  }
  Table.prototype.parse = function() {};
  Table.prototype.encode = function() {};
  Table.prototype.raw = function() {
    if (!this.exists) {
      return null;
    }
    this.file.contents.pos = this.offset;
    return this.file.contents.read(this.length);
  };
  return Table;
})();

var HeadTable = (function(_super) {
  __extends(HeadTable, _super);

  function HeadTable() {
    return HeadTable.__super__.constructor.apply(this, arguments);
  }
  HeadTable.prototype.tag = "head";
  HeadTable.prototype.parse = function(data) {
    data.pos = this.offset;
    this.version = data.readInt();
    this.revision = data.readInt();
    this.checkSumAdjustment = data.readInt();
    this.magicNumber = data.readInt();
    this.flags = data.readShort();
    this.unitsPerEm = data.readShort();
    this.created = data.readLongLong();
    this.modified = data.readLongLong();
    this.xMin = data.readShort();
    this.yMin = data.readShort();
    this.xMax = data.readShort();
    this.yMax = data.readShort();
    this.macStyle = data.readShort();
    this.lowestRecPPEM = data.readShort();
    this.fontDirectionHint = data.readShort();
    this.indexToLocFormat = data.readShort();
    return (this.glyphDataFormat = data.readShort());
  };
  HeadTable.prototype.encode = function(indexToLocFormat) {
    var table;
    table = new Data();
    table.writeInt(this.version);
    table.writeInt(this.revision);
    table.writeInt(this.checkSumAdjustment);
    table.writeInt(this.magicNumber);
    table.writeShort(this.flags);
    table.writeShort(this.unitsPerEm);
    table.writeLongLong(this.created);
    table.writeLongLong(this.modified);
    table.writeShort(this.xMin);
    table.writeShort(this.yMin);
    table.writeShort(this.xMax);
    table.writeShort(this.yMax);
    table.writeShort(this.macStyle);
    table.writeShort(this.lowestRecPPEM);
    table.writeShort(this.fontDirectionHint);
    table.writeShort(indexToLocFormat);
    table.writeShort(this.glyphDataFormat);
    return table.data;
  };
  return HeadTable;
})(Table);

/************************************************************************************/
/* function : CmapEntry                                                             */
/* comment : Cmap Initializes and encodes object information (required by pdf spec).*/
/************************************************************************************/
var CmapEntry = (function() {
  function CmapEntry(data, offset) {
    var code,
      count,
      endCode,
      glyphId,
      glyphIds,
      i,
      idDelta,
      idRangeOffset,
      index,
      saveOffset,
      segCount,
      segCountX2,
      start,
      startCode,
      tail,
      _j,
      _k,
      _len;
    this.platformID = data.readUInt16();
    this.encodingID = data.readShort();
    this.offset = offset + data.readInt();
    saveOffset = data.pos;
    data.pos = this.offset;
    this.format = data.readUInt16();
    this.length = data.readUInt16();
    this.language = data.readUInt16();
    this.isUnicode =
      (this.platformID === 3 && this.encodingID === 1 && this.format === 4) ||
      (this.platformID === 0 && this.format === 4) ||
      (this.platformID === 1 && this.encodingID === 0 &&  this.format === 0);
    this.codeMap = {};
    switch (this.format) {
      case 0:
        for (i = 0; i < 256; ++i) {
          this.codeMap[i] = data.readByte();
        }
        break;
      case 4:
        segCountX2 = data.readUInt16();
        segCount = segCountX2 / 2;
        data.pos += 6;
        endCode = (function() {
          var _j, _results;
          _results = [];
          for (
            i = _j = 0;
            0 <= segCount ? _j < segCount : _j > segCount;
            i = 0 <= segCount ? ++_j : --_j
          ) {
            _results.push(data.readUInt16());
          }
          return _results;
        })();
        data.pos += 2;
        startCode = (function() {
          var _j, _results;
          _results = [];
          for (
            i = _j = 0;
            0 <= segCount ? _j < segCount : _j > segCount;
            i = 0 <= segCount ? ++_j : --_j
          ) {
            _results.push(data.readUInt16());
          }
          return _results;
        })();
        idDelta = (function() {
          var _j, _results;
          _results = [];
          for (
            i = _j = 0;
            0 <= segCount ? _j < segCount : _j > segCount;
            i = 0 <= segCount ? ++_j : --_j
          ) {
            _results.push(data.readUInt16());
          }
          return _results;
        })();
        idRangeOffset = (function() {
          var _j, _results;
          _results = [];
          for (
            i = _j = 0;
            0 <= segCount ? _j < segCount : _j > segCount;
            i = 0 <= segCount ? ++_j : --_j
          ) {
            _results.push(data.readUInt16());
          }
          return _results;
        })();
        count = (this.length - data.pos + this.offset) / 2;
        glyphIds = (function() {
          var _j, _results;
          _results = [];
          for (
            i = _j = 0;
            0 <= count ? _j < count : _j > count;
            i = 0 <= count ? ++_j : --_j
          ) {
            _results.push(data.readUInt16());
          }
          return _results;
        })();
        for (i = _j = 0, _len = endCode.length; _j < _len; i = ++_j) {
          tail = endCode[i];
          start = startCode[i];
          for (
            code = _k = start;
            start <= tail ? _k <= tail : _k >= tail;
            code = start <= tail ? ++_k : --_k
          ) {
            if (idRangeOffset[i] === 0) {
              glyphId = code + idDelta[i];
            } else {
              index = idRangeOffset[i] / 2 + (code - start) - (segCount - i);
              glyphId = glyphIds[index] || 0;
              if (glyphId !== 0) {
                glyphId += idDelta[i];
              }
            }
            this.codeMap[code] = glyphId & 0xffff;
          }
        }
    }
    data.pos = saveOffset;
  }
  CmapEntry.encode = function(charmap, encoding) {
    var charMap,
      code,
      codeMap,
      codes,
      delta,
      deltas,
      diff,
      endCode,
      endCodes,
      entrySelector,
      glyphIDs,
      i,
      id,
      indexes,
      last,
      map,
      nextID,
      offset,
      old,
      rangeOffsets,
      rangeShift,
      searchRange,
      segCount,
      segCountX2,
      startCode,
      startCodes,
      startGlyph,
      subtable,
      _i,
      _j,
      _k,
      _l,
      _len,
      _len1,
      _len2,
      _len3,
      _len4,
      _len5,
      _len6,
      _len7,
      _m,
      _n,
      _name,
      _o,
      _p,
      _q;
    subtable = new Data();
    codes = Object.keys(charmap).sort(function(a, b) {
      return a - b;
    });
    switch (encoding) {
      case "macroman":
        id = 0;
        indexes = (function() {
          var _results = [];
          for (i = 0; i < 256; ++i) {
            _results.push(0);
          }
          return _results;
        })();
        map = {
          0: 0
        };
        codeMap = {};
        for (_i = 0, _len = codes.length; _i < _len; _i++) {
          code = codes[_i];
          if (map[(_name = charmap[code])] == null) {
            map[_name] = ++id;
          }
          codeMap[code] = {
            old: charmap[code],
            new: map[charmap[code]]
          };
          indexes[code] = map[charmap[code]];
        }
        subtable.writeUInt16(1);
        subtable.writeUInt16(0);
        subtable.writeUInt32(12);
        subtable.writeUInt16(0);
        subtable.writeUInt16(262);
        subtable.writeUInt16(0);
        subtable.write(indexes);
        return {
          charMap: codeMap,
          subtable: subtable.data,
          maxGlyphID: id + 1
        };
      case "unicode":
        startCodes = [];
        endCodes = [];
        nextID = 0;
        map = {};
        charMap = {};
        last = diff = null;
        for (_j = 0, _len1 = codes.length; _j < _len1; _j++) {
          code = codes[_j];
          old = charmap[code];
          if (map[old] == null) {
            map[old] = ++nextID;
          }
          charMap[code] = {
            old: old,
            new: map[old]
          };
          delta = map[old] - code;
          if (last == null || delta !== diff) {
            if (last) {
              endCodes.push(last);
            }
            startCodes.push(code);
            diff = delta;
          }
          last = code;
        }
        if (last) {
          endCodes.push(last);
        }
        endCodes.push(0xffff);
        startCodes.push(0xffff);
        segCount = startCodes.length;
        segCountX2 = segCount * 2;
        searchRange = 2 * Math.pow(Math.log(segCount) / Math.LN2, 2);
        entrySelector = Math.log(searchRange / 2) / Math.LN2;
        rangeShift = 2 * segCount - searchRange;
        deltas = [];
        rangeOffsets = [];
        glyphIDs = [];
        for (i = _k = 0, _len2 = startCodes.length; _k < _len2; i = ++_k) {
          startCode = startCodes[i];
          endCode = endCodes[i];
          if (startCode === 0xffff) {
            deltas.push(0);
            rangeOffsets.push(0);
            break;
          }
          startGlyph = charMap[startCode]["new"];
          if (startCode - startGlyph >= 0x8000) {
            deltas.push(0);
            rangeOffsets.push(2 * (glyphIDs.length + segCount - i));
            for (
              code = _l = startCode;
              startCode <= endCode ? _l <= endCode : _l >= endCode;
              code = startCode <= endCode ? ++_l : --_l
            ) {
              glyphIDs.push(charMap[code]["new"]);
            }
          } else {
            deltas.push(startGlyph - startCode);
            rangeOffsets.push(0);
          }
        }
        subtable.writeUInt16(3);
        subtable.writeUInt16(1);
        subtable.writeUInt32(12);
        subtable.writeUInt16(4);
        subtable.writeUInt16(16 + segCount * 8 + glyphIDs.length * 2);
        subtable.writeUInt16(0);
        subtable.writeUInt16(segCountX2);
        subtable.writeUInt16(searchRange);
        subtable.writeUInt16(entrySelector);
        subtable.writeUInt16(rangeShift);
        for (_m = 0, _len3 = endCodes.length; _m < _len3; _m++) {
          code = endCodes[_m];
          subtable.writeUInt16(code);
        }
        subtable.writeUInt16(0);
        for (_n = 0, _len4 = startCodes.length; _n < _len4; _n++) {
          code = startCodes[_n];
          subtable.writeUInt16(code);
        }
        for (_o = 0, _len5 = deltas.length; _o < _len5; _o++) {
          delta = deltas[_o];
          subtable.writeUInt16(delta);
        }
        for (_p = 0, _len6 = rangeOffsets.length; _p < _len6; _p++) {
          offset = rangeOffsets[_p];
          subtable.writeUInt16(offset);
        }
        for (_q = 0, _len7 = glyphIDs.length; _q < _len7; _q++) {
          id = glyphIDs[_q];
          subtable.writeUInt16(id);
        }
        return {
          charMap: charMap,
          subtable: subtable.data,
          maxGlyphID: nextID + 1
        };
    }
  };
  return CmapEntry;
})();

var CmapTable = (function(_super) {
  __extends(CmapTable, _super);

  function CmapTable() {
    return CmapTable.__super__.constructor.apply(this, arguments);
  }
  CmapTable.prototype.tag = "cmap";
  CmapTable.prototype.parse = function(data) {
    var entry, i, tableCount;
    data.pos = this.offset;
    this.version = data.readUInt16();
    tableCount = data.readUInt16();
    this.tables = [];
    this.unicode = null;
    for (
      i = 0;
      0 <= tableCount ? i < tableCount : i > tableCount;
      i = 0 <= tableCount ? ++i : --i
    ) {
      entry = new CmapEntry(data, this.offset);
      this.tables.push(entry);
      if (entry.isUnicode) {
        if (this.unicode == null) {
          this.unicode = entry;
        }
      }
    }
    return true;
  };
  /*************************************************************************/
  /* function : encode                                                     */
  /* comment : Encode the cmap table corresponding to the input character. */
  /*************************************************************************/
  CmapTable.encode = function(charmap, encoding) {
    var result, table;
    if (encoding == null) {
      encoding = "macroman";
    }
    result = CmapEntry.encode(charmap, encoding);
    table = new Data();
    table.writeUInt16(0);
    table.writeUInt16(1);
    result.table = table.data.concat(result.subtable);
    return result;
  };
  return CmapTable;
})(Table);

var HheaTable = (function(_super) {
  __extends(HheaTable, _super);

  function HheaTable() {
    return HheaTable.__super__.constructor.apply(this, arguments);
  }
  HheaTable.prototype.tag = "hhea";
  HheaTable.prototype.parse = function(data) {
    data.pos = this.offset;
    this.version = data.readInt();
    this.ascender = data.readShort();
    this.decender = data.readShort();
    this.lineGap = data.readShort();
    this.advanceWidthMax = data.readShort();
    this.minLeftSideBearing = data.readShort();
    this.minRightSideBearing = data.readShort();
    this.xMaxExtent = data.readShort();
    this.caretSlopeRise = data.readShort();
    this.caretSlopeRun = data.readShort();
    this.caretOffset = data.readShort();
    data.pos += 4 * 2;
    this.metricDataFormat = data.readShort();
    return (this.numberOfMetrics = data.readUInt16());
  };
  /*HheaTable.prototype.encode = function (ids) {
            var i, table, _i, _ref;
            table = new Data;
            table.writeInt(this.version);
            table.writeShort(this.ascender);
            table.writeShort(this.decender);
            table.writeShort(this.lineGap);
            table.writeShort(this.advanceWidthMax);
            table.writeShort(this.minLeftSideBearing);
            table.writeShort(this.minRightSideBearing);
            table.writeShort(this.xMaxExtent);
            table.writeShort(this.caretSlopeRise);
            table.writeShort(this.caretSlopeRun);
            table.writeShort(this.caretOffset);
            for (i = _i = 0, _ref = 4 * 2; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
                table.writeByte(0);
            }
            table.writeShort(this.metricDataFormat);
            table.writeUInt16(ids.length);
            return table.data;
        };*/
  return HheaTable;
})(Table);

var OS2Table = (function(_super) {
  __extends(OS2Table, _super);

  function OS2Table() {
    return OS2Table.__super__.constructor.apply(this, arguments);
  }
  OS2Table.prototype.tag = "OS/2";
  OS2Table.prototype.parse = function(data) {
    data.pos = this.offset;
    this.version = data.readUInt16();
    this.averageCharWidth = data.readShort();
    this.weightClass = data.readUInt16();
    this.widthClass = data.readUInt16();
    this.type = data.readShort();
    this.ySubscriptXSize = data.readShort();
    this.ySubscriptYSize = data.readShort();
    this.ySubscriptXOffset = data.readShort();
    this.ySubscriptYOffset = data.readShort();
    this.ySuperscriptXSize = data.readShort();
    this.ySuperscriptYSize = data.readShort();
    this.ySuperscriptXOffset = data.readShort();
    this.ySuperscriptYOffset = data.readShort();
    this.yStrikeoutSize = data.readShort();
    this.yStrikeoutPosition = data.readShort();
    this.familyClass = data.readShort();
    this.panose = (function() {
      var i, _results;
      _results = [];
      for (i = 0; i < 10; ++i) {
        _results.push(data.readByte());
      }
      return _results;
    })();
    this.charRange = (function() {
      var i, _results;
      _results = [];
      for (i = 0; i < 4; ++i) {
        _results.push(data.readInt());
      }
      return _results;
    })();
    this.vendorID = data.readString(4);
    this.selection = data.readShort();
    this.firstCharIndex = data.readShort();
    this.lastCharIndex = data.readShort();
    if (this.version > 0) {
      this.ascent = data.readShort();
      this.descent = data.readShort();
      this.lineGap = data.readShort();
      this.winAscent = data.readShort();
      this.winDescent = data.readShort();
      this.codePageRange = (function() {
        var i, _results;
        _results = [];
        for (i = 0; i < 2; i = ++i) {
          _results.push(data.readInt());
        }
        return _results;
      })();
      if (this.version > 1) {
        this.xHeight = data.readShort();
        this.capHeight = data.readShort();
        this.defaultChar = data.readShort();
        this.breakChar = data.readShort();
        return (this.maxContext = data.readShort());
      }
    }
  };
  /*OS2Table.prototype.encode = function () {
            return this.raw();
        };*/
  return OS2Table;
})(Table);

var PostTable = (function(_super) {
  __extends(PostTable, _super);

  function PostTable() {
    return PostTable.__super__.constructor.apply(this, arguments);
  }
  PostTable.prototype.tag = "post";
  PostTable.prototype.parse = function(data) {
    var length, numberOfGlyphs, _results;
    data.pos = this.offset;
    this.format = data.readInt();
    this.italicAngle = data.readInt();
    this.underlinePosition = data.readShort();
    this.underlineThickness = data.readShort();
    this.isFixedPitch = data.readInt();
    this.minMemType42 = data.readInt();
    this.maxMemType42 = data.readInt();
    this.minMemType1 = data.readInt();
    this.maxMemType1 = data.readInt();
    switch (this.format) {
      case 0x00010000:
        break;
      case 0x00020000:
        numberOfGlyphs = data.readUInt16();
        this.glyphNameIndex = [];
        var i;
        for (
          i = 0;
          0 <= numberOfGlyphs ? i < numberOfGlyphs : i > numberOfGlyphs;
          i = 0 <= numberOfGlyphs ? ++i : --i
        ) {
          this.glyphNameIndex.push(data.readUInt16());
        }
        this.names = [];
        _results = [];
        while (data.pos < this.offset + this.length) {
          length = data.readByte();
          _results.push(this.names.push(data.readString(length)));
        }
        return _results;
      case 0x00025000:
        numberOfGlyphs = data.readUInt16();
        return (this.offsets = data.read(numberOfGlyphs));
      case 0x00030000:
        break;
      case 0x00040000:
        return (this.map = function() {
          var _j, _ref, _results1;
          _results1 = [];
          for (
            i = _j = 0, _ref = this.file.maxp.numGlyphs;
            0 <= _ref ? _j < _ref : _j > _ref;
            i = 0 <= _ref ? ++_j : --_j
          ) {
            _results1.push(data.readUInt32());
          }
          return _results1;
        }.call(this));
    }
  };
  return PostTable;
})(Table);

/*********************************************************************************************************/
/* function : NameEntry                                                                                  */
/* comment : Store copyright information, platformID, encodingID, and languageID in the NameEntry object.*/
/*********************************************************************************************************/
var NameEntry = (function() {
  function NameEntry(raw, entry) {
    this.raw = raw;
    this.length = raw.length;
    this.platformID = entry.platformID;
    this.encodingID = entry.encodingID;
    this.languageID = entry.languageID;
  }
  return NameEntry;
})();

var NameTable = (function(_super) {
  __extends(NameTable, _super);

  function NameTable() {
    return NameTable.__super__.constructor.apply(this, arguments);
  }
  NameTable.prototype.tag = "name";
  NameTable.prototype.parse = function(data) {
    var count,
      entries,
      entry,
      i,
      name,
      stringOffset,
      strings,
      text,
      _j,
      _len,
      _name;
    data.pos = this.offset;
    data.readShort(); //format
    count = data.readShort();
    stringOffset = data.readShort();
    entries = [];
    for (
      i = 0;
      0 <= count ? i < count : i > count;
      i = 0 <= count ? ++i : --i
    ) {
      entries.push({
        platformID: data.readShort(),
        encodingID: data.readShort(),
        languageID: data.readShort(),
        nameID: data.readShort(),
        length: data.readShort(),
        offset: this.offset + stringOffset + data.readShort()
      });
    }
    strings = {};
    for (i = _j = 0, _len = entries.length; _j < _len; i = ++_j) {
      entry = entries[i];
      data.pos = entry.offset;
      text = data.readString(entry.length);
      name = new NameEntry(text, entry);
      if (strings[(_name = entry.nameID)] == null) {
        strings[_name] = [];
      }
      strings[entry.nameID].push(name);
    }
    this.strings = strings;
    this.copyright = strings[0];
    this.fontFamily = strings[1];
    this.fontSubfamily = strings[2];
    this.uniqueSubfamily = strings[3];
    this.fontName = strings[4];
    this.version = strings[5];
    try {
      this.postscriptName = strings[6][0].raw.replace(
        /[\x00-\x19\x80-\xff]/g,
        ""
      );
    } catch (e) {
      this.postscriptName = strings[4][0].raw.replace(
        /[\x00-\x19\x80-\xff]/g,
        ""
      );
    }
    this.trademark = strings[7];
    this.manufacturer = strings[8];
    this.designer = strings[9];
    this.description = strings[10];
    this.vendorUrl = strings[11];
    this.designerUrl = strings[12];
    this.license = strings[13];
    this.licenseUrl = strings[14];
    this.preferredFamily = strings[15];
    this.preferredSubfamily = strings[17];
    this.compatibleFull = strings[18];
    return (this.sampleText = strings[19]);
  };
  /*NameTable.prototype.encode = function () {
            var id, list, nameID, nameTable, postscriptName, strCount, strTable, string, strings, table, val, _i, _len, _ref;
            strings = {};
            _ref = this.strings;
            for (id in _ref) {
                val = _ref[id];
                strings[id] = val;
            }
            postscriptName = new NameEntry("" + subsetTag + "+" + this.postscriptName, {
                platformID: 1
                , encodingID: 0
                , languageID: 0
            });
            strings[6] = [postscriptName];
            subsetTag = successorOf(subsetTag);
            strCount = 0;
            for (id in strings) {
                list = strings[id];
                if (list != null) {
                    strCount += list.length;
                }
            }
            table = new Data;
            strTable = new Data;
            table.writeShort(0);
            table.writeShort(strCount);
            table.writeShort(6 + 12 * strCount);
            for (nameID in strings) {
                list = strings[nameID];
                if (list != null) {
                    for (_i = 0, _len = list.length; _i < _len; _i++) {
                        string = list[_i];
                        table.writeShort(string.platformID);
                        table.writeShort(string.encodingID);
                        table.writeShort(string.languageID);
                        table.writeShort(nameID);
                        table.writeShort(string.length);
                        table.writeShort(strTable.pos);
                        strTable.writeString(string.raw);
                    }
                }
            }
            return nameTable = {
                postscriptName: postscriptName.raw
                , table: table.data.concat(strTable.data)
            };
        };*/
  return NameTable;
})(Table);

var MaxpTable = (function(_super) {
  __extends(MaxpTable, _super);

  function MaxpTable() {
    return MaxpTable.__super__.constructor.apply(this, arguments);
  }
  MaxpTable.prototype.tag = "maxp";
  MaxpTable.prototype.parse = function(data) {
    data.pos = this.offset;
    this.version = data.readInt();
    this.numGlyphs = data.readUInt16();
    this.maxPoints = data.readUInt16();
    this.maxContours = data.readUInt16();
    this.maxCompositePoints = data.readUInt16();
    this.maxComponentContours = data.readUInt16();
    this.maxZones = data.readUInt16();
    this.maxTwilightPoints = data.readUInt16();
    this.maxStorage = data.readUInt16();
    this.maxFunctionDefs = data.readUInt16();
    this.maxInstructionDefs = data.readUInt16();
    this.maxStackElements = data.readUInt16();
    this.maxSizeOfInstructions = data.readUInt16();
    this.maxComponentElements = data.readUInt16();
    return (this.maxComponentDepth = data.readUInt16());
  };
  /*MaxpTable.prototype.encode = function (ids) {
            var table;
            table = new Data;
            table.writeInt(this.version);
            table.writeUInt16(ids.length);
            table.writeUInt16(this.maxPoints);
            table.writeUInt16(this.maxContours);
            table.writeUInt16(this.maxCompositePoints);
            table.writeUInt16(this.maxComponentContours);
            table.writeUInt16(this.maxZones);
            table.writeUInt16(this.maxTwilightPoints);
            table.writeUInt16(this.maxStorage);
            table.writeUInt16(this.maxFunctionDefs);
            table.writeUInt16(this.maxInstructionDefs);
            table.writeUInt16(this.maxStackElements);
            table.writeUInt16(this.maxSizeOfInstructions);
            table.writeUInt16(this.maxComponentElements);
            table.writeUInt16(this.maxComponentDepth);
            return table.data;
        };*/
  return MaxpTable;
})(Table);

var HmtxTable = (function(_super) {
  __extends(HmtxTable, _super);

  function HmtxTable() {
    return HmtxTable.__super__.constructor.apply(this, arguments);
  }
  HmtxTable.prototype.tag = "hmtx";
  HmtxTable.prototype.parse = function(data) {
    var i, last, lsbCount, m, _j, _ref, _results;
    data.pos = this.offset;
    this.metrics = [];
    for (
      i = 0, _ref = this.file.hhea.numberOfMetrics;
      0 <= _ref ? i < _ref : i > _ref;
      i = 0 <= _ref ? ++i : --i
    ) {
      this.metrics.push({
        advance: data.readUInt16(),
        lsb: data.readInt16()
      });
    }
    lsbCount = this.file.maxp.numGlyphs - this.file.hhea.numberOfMetrics;
    this.leftSideBearings = (function() {
      var _j, _results;
      _results = [];
      for (
        i = _j = 0;
        0 <= lsbCount ? _j < lsbCount : _j > lsbCount;
        i = 0 <= lsbCount ? ++_j : --_j
      ) {
        _results.push(data.readInt16());
      }
      return _results;
    })();
    this.widths = function() {
      var _j, _len, _ref1, _results;
      _ref1 = this.metrics;
      _results = [];
      for (_j = 0, _len = _ref1.length; _j < _len; _j++) {
        m = _ref1[_j];
        _results.push(m.advance);
      }
      return _results;
    }.call(this);
    last = this.widths[this.widths.length - 1];
    _results = [];
    for (
      i = _j = 0;
      0 <= lsbCount ? _j < lsbCount : _j > lsbCount;
      i = 0 <= lsbCount ? ++_j : --_j
    ) {
      _results.push(this.widths.push(last));
    }
    return _results;
  };
  /***************************************************************/
  /* function : forGlyph                                         */
  /* comment : Returns the advance width and lsb for this glyph. */
  /***************************************************************/
  HmtxTable.prototype.forGlyph = function(id) {
    if (id in this.metrics) {
      return this.metrics[id];
    }
    return {
      advance: this.metrics[this.metrics.length - 1].advance,
      lsb: this.leftSideBearings[id - this.metrics.length]
    };
  };
  /*HmtxTable.prototype.encode = function (mapping) {
            var id, metric, table, _i, _len;
            table = new Data;
            for (_i = 0, _len = mapping.length; _i < _len; _i++) {
                id = mapping[_i];
                metric = this.forGlyph(id);
                table.writeUInt16(metric.advance);
                table.writeUInt16(metric.lsb);
            }
            return table.data;
        };*/
  return HmtxTable;
})(Table);

var __slice = [].slice;

var GlyfTable = (function(_super) {
  __extends(GlyfTable, _super);

  function GlyfTable() {
    return GlyfTable.__super__.constructor.apply(this, arguments);
  }
  GlyfTable.prototype.tag = "glyf";
  GlyfTable.prototype.parse = function() {
    return (this.cache = {});
  };
  GlyfTable.prototype.glyphFor = function(id) {
    var data,
      index,
      length,
      loca,
      numberOfContours,
      raw,
      xMax,
      xMin,
      yMax,
      yMin;
    if (id in this.cache) {
      return this.cache[id];
    }
    loca = this.file.loca;
    data = this.file.contents;
    index = loca.indexOf(id);
    length = loca.lengthOf(id);
    if (length === 0) {
      return (this.cache[id] = null);
    }
    data.pos = this.offset + index;
    raw = new Data(data.read(length));
    numberOfContours = raw.readShort();
    xMin = raw.readShort();
    yMin = raw.readShort();
    xMax = raw.readShort();
    yMax = raw.readShort();
    if (numberOfContours === -1) {
      this.cache[id] = new CompoundGlyph(raw, xMin, yMin, xMax, yMax);
    } else {
      this.cache[id] = new SimpleGlyph(
        raw,
        numberOfContours,
        xMin,
        yMin,
        xMax,
        yMax
      );
    }
    return this.cache[id];
  };
  GlyfTable.prototype.encode = function(glyphs, mapping, old2new) {
    var glyph, id, offsets, table, _i, _len;
    table = [];
    offsets = [];
    for (_i = 0, _len = mapping.length; _i < _len; _i++) {
      id = mapping[_i];
      glyph = glyphs[id];
      offsets.push(table.length);
      if (glyph) {
        table = table.concat(glyph.encode(old2new));
      }
    }
    offsets.push(table.length);
    return {
      table: table,
      offsets: offsets
    };
  };
  return GlyfTable;
})(Table);

var SimpleGlyph = (function() {
  /**************************************************************************/
  /* function : SimpleGlyph                                                 */
  /* comment : Stores raw, xMin, yMin, xMax, and yMax values for this glyph.*/
  /**************************************************************************/
  function SimpleGlyph(raw, numberOfContours, xMin, yMin, xMax, yMax) {
    this.raw = raw;
    this.numberOfContours = numberOfContours;
    this.xMin = xMin;
    this.yMin = yMin;
    this.xMax = xMax;
    this.yMax = yMax;
    this.compound = false;
  }
  SimpleGlyph.prototype.encode = function() {
    return this.raw.data;
  };
  return SimpleGlyph;
})();

var CompoundGlyph = (function() {
  var ARG_1_AND_2_ARE_WORDS,
    MORE_COMPONENTS,
    WE_HAVE_AN_X_AND_Y_SCALE,
    WE_HAVE_A_SCALE,
    WE_HAVE_A_TWO_BY_TWO;
  ARG_1_AND_2_ARE_WORDS = 0x0001;
  WE_HAVE_A_SCALE = 0x0008;
  MORE_COMPONENTS = 0x0020;
  WE_HAVE_AN_X_AND_Y_SCALE = 0x0040;
  WE_HAVE_A_TWO_BY_TWO = 0x0080;

  /********************************************************************************************************************/
  /* function : CompoundGlypg generator                                                                               */
  /* comment : It stores raw, xMin, yMin, xMax, yMax, glyph id, and glyph offset for the corresponding compound glyph.*/
  /********************************************************************************************************************/
  function CompoundGlyph(raw, xMin, yMin, xMax, yMax) {
    var data, flags;
    this.raw = raw;
    this.xMin = xMin;
    this.yMin = yMin;
    this.xMax = xMax;
    this.yMax = yMax;
    this.compound = true;
    this.glyphIDs = [];
    this.glyphOffsets = [];
    data = this.raw;
    while (true) {
      flags = data.readShort();
      this.glyphOffsets.push(data.pos);
      this.glyphIDs.push(data.readUInt16());
      if (!(flags & MORE_COMPONENTS)) {
        break;
      }
      if (flags & ARG_1_AND_2_ARE_WORDS) {
        data.pos += 4;
      } else {
        data.pos += 2;
      }
      if (flags & WE_HAVE_A_TWO_BY_TWO) {
        data.pos += 8;
      } else if (flags & WE_HAVE_AN_X_AND_Y_SCALE) {
        data.pos += 4;
      } else if (flags & WE_HAVE_A_SCALE) {
        data.pos += 2;
      }
    }
  }
  /****************************************************************************************************************/
  /* function : CompoundGlypg encode                                                                              */
  /* comment : After creating a table for the characters you typed, you call directory.encode to encode the table.*/
  /****************************************************************************************************************/
  CompoundGlyph.prototype.encode = function() {
    var i, result, _len, _ref;
    result = new Data(__slice.call(this.raw.data));
    _ref = this.glyphIDs;
    for (i = 0, _len = _ref.length; i < _len; ++i) {
      result.pos = this.glyphOffsets[i];
    }
    return result.data;
  };
  return CompoundGlyph;
})();

var LocaTable = (function(_super) {
  __extends(LocaTable, _super);

  function LocaTable() {
    return LocaTable.__super__.constructor.apply(this, arguments);
  }
  LocaTable.prototype.tag = "loca";
  LocaTable.prototype.parse = function(data) {
    var format, i;
    data.pos = this.offset;
    format = this.file.head.indexToLocFormat;
    if (format === 0) {
      return (this.offsets = function() {
        var _ref, _results;
        _results = [];
        for (i = 0, _ref = this.length; i < _ref; i += 2) {
          _results.push(data.readUInt16() * 2);
        }
        return _results;
      }.call(this));
    } else {
      return (this.offsets = function() {
        var _ref, _results;
        _results = [];
        for (i = 0, _ref = this.length; i < _ref; i += 4) {
          _results.push(data.readUInt32());
        }
        return _results;
      }.call(this));
    }
  };
  LocaTable.prototype.indexOf = function(id) {
    return this.offsets[id];
  };
  LocaTable.prototype.lengthOf = function(id) {
    return this.offsets[id + 1] - this.offsets[id];
  };
  LocaTable.prototype.encode = function(offsets, activeGlyphs) {
    var LocaTable = new Uint32Array(this.offsets.length);
    var glyfPtr = 0;
    var listGlyf = 0;
    for (var k = 0; k < LocaTable.length; ++k) {
      LocaTable[k] = glyfPtr;
      if (listGlyf < activeGlyphs.length && activeGlyphs[listGlyf] == k) {
        ++listGlyf;
        LocaTable[k] = glyfPtr;
        var start = this.offsets[k];
        var len = this.offsets[k + 1] - start;
        if (len > 0) {
          glyfPtr += len;
        }
      }
    }
    var newLocaTable = new Array(LocaTable.length * 4);
    for (var j = 0; j < LocaTable.length; ++j) {
      newLocaTable[4 * j + 3] = LocaTable[j] & 0x000000ff;
      newLocaTable[4 * j + 2] = (LocaTable[j] & 0x0000ff00) >> 8;
      newLocaTable[4 * j + 1] = (LocaTable[j] & 0x00ff0000) >> 16;
      newLocaTable[4 * j] = (LocaTable[j] & 0xff000000) >> 24;
    }
    return newLocaTable;
  };
  return LocaTable;
})(Table);

/************************************************************************************/
/* function : invert                                                                */
/* comment : Change the object's (key: value) to create an object with (value: key).*/
/************************************************************************************/
var invert = function(object) {
  var key, ret, val;
  ret = {};
  for (key in object) {
    val = object[key];
    ret[val] = key;
  }
  return ret;
};

/*var successorOf = function (input) {
        var added, alphabet, carry, i, index, isUpperCase, last, length, next, result;
        alphabet = 'abcdefghijklmnopqrstuvwxyz';
        length = alphabet.length;
        result = input;
        i = input.length;
        while (i >= 0) {
            last = input.charAt(--i);
            if (isNaN(last)) {
                index = alphabet.indexOf(last.toLowerCase());
                if (index === -1) {
                    next = last;
                    carry = true;
                }
                else {
                    next = alphabet.charAt((index + 1) % length);
                    isUpperCase = last === last.toUpperCase();
                    if (isUpperCase) {
                        next = next.toUpperCase();
                    }
                    carry = index + 1 >= length;
                    if (carry && i === 0) {
                        added = isUpperCase ? 'A' : 'a';
                        result = added + next + result.slice(1);
                        break;
                    }
                }
            }
            else {
                next = +last + 1;
                carry = next > 9;
                if (carry) {
                    next = 0;
                }
                if (carry && i === 0) {
                    result = '1' + next + result.slice(1);
                    break;
                }
            }
            result = result.slice(0, i) + next + result.slice(i + 1);
            if (!carry) {
                break;
            }
        }
        return result;
    };*/

var Subset = (function() {
  function Subset(font) {
    this.font = font;
    this.subset = {};
    this.unicodes = {};
    this.next = 33;
  }
  /*Subset.prototype.use = function (character) {
            var i, _i, _ref;
            if (typeof character === 'string') {
                for (i = _i = 0, _ref = character.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
                    this.use(character.charCodeAt(i));
                }
                return;
            }
            if (!this.unicodes[character]) {
                this.subset[this.next] = character;
                return this.unicodes[character] = this.next++;
            }
        };*/
  /*Subset.prototype.encodeText = function (text) {
            var char, i, string, _i, _ref;
            string = '';
            for (i = _i = 0, _ref = text.length; 0 <= _ref ? _i < _ref : _i > _ref; i = 0 <= _ref ? ++_i : --_i) {
                char = this.unicodes[text.charCodeAt(i)];
                string += String.fromCharCode(char);
            }
            return string;
        };*/
  /***************************************************************/
  /* function : generateCmap                                     */
  /* comment : Returns the unicode cmap for this font.         */
  /***************************************************************/
  Subset.prototype.generateCmap = function() {
    var mapping, roman, unicode, unicodeCmap, _ref;
    unicodeCmap = this.font.cmap.tables[0].codeMap;
    mapping = {};
    _ref = this.subset;
    for (roman in _ref) {
      unicode = _ref[roman];
      mapping[roman] = unicodeCmap[unicode];
    }
    return mapping;
  };
  /*Subset.prototype.glyphIDs = function () {
            var ret, roman, unicode, unicodeCmap, val, _ref;
            unicodeCmap = this.font.cmap.tables[0].codeMap;
            ret = [0];
            _ref = this.subset;
            for (roman in _ref) {
                unicode = _ref[roman];
                val = unicodeCmap[unicode];
                if ((val != null) && __indexOf.call(ret, val) < 0) {
                    ret.push(val);
                }
            }
            return ret.sort();
        };*/
  /******************************************************************/
  /* function : glyphsFor                                           */
  /* comment : Returns simple glyph objects for the input character.*/
  /******************************************************************/
  Subset.prototype.glyphsFor = function(glyphIDs) {
    var additionalIDs, glyph, glyphs, id, _i, _len, _ref;
    glyphs = {};
    for (_i = 0, _len = glyphIDs.length; _i < _len; _i++) {
      id = glyphIDs[_i];
      glyphs[id] = this.font.glyf.glyphFor(id);
    }
    additionalIDs = [];
    for (id in glyphs) {
      glyph = glyphs[id];
      if (glyph != null ? glyph.compound : void 0) {
        additionalIDs.push.apply(additionalIDs, glyph.glyphIDs);
      }
    }
    if (additionalIDs.length > 0) {
      _ref = this.glyphsFor(additionalIDs);
      for (id in _ref) {
        glyph = _ref[id];
        glyphs[id] = glyph;
      }
    }
    return glyphs;
  };
  /***************************************************************/
  /* function : encode                                           */
  /* comment : Encode various tables for the characters you use. */
  /***************************************************************/
  Subset.prototype.encode = function(glyID, indexToLocFormat) {
    var cmap,
      code,
      glyf,
      glyphs,
      id,
      ids,
      loca,
      new2old,
      newIDs,
      nextGlyphID,
      old2new,
      oldID,
      oldIDs,
      tables,
      _ref;
    cmap = CmapTable.encode(this.generateCmap(), "unicode");
    glyphs = this.glyphsFor(glyID);
    old2new = {
      0: 0
    };
    _ref = cmap.charMap;
    for (code in _ref) {
      ids = _ref[code];
      old2new[ids.old] = ids["new"];
    }
    nextGlyphID = cmap.maxGlyphID;
    for (oldID in glyphs) {
      if (!(oldID in old2new)) {
        old2new[oldID] = nextGlyphID++;
      }
    }
    new2old = invert(old2new);
    newIDs = Object.keys(new2old).sort(function(a, b) {
      return a - b;
    });
    oldIDs = (function() {
      var _i, _len, _results;
      _results = [];
      for (_i = 0, _len = newIDs.length; _i < _len; _i++) {
        id = newIDs[_i];
        _results.push(new2old[id]);
      }
      return _results;
    })();
    glyf = this.font.glyf.encode(glyphs, oldIDs, old2new);
    loca = this.font.loca.encode(glyf.offsets, oldIDs);
    tables = {
      cmap: this.font.cmap.raw(),
      glyf: glyf.table,
      loca: loca,
      hmtx: this.font.hmtx.raw(),
      hhea: this.font.hhea.raw(),
      maxp: this.font.maxp.raw(),
      post: this.font.post.raw(),
      name: this.font.name.raw(),
      head: this.font.head.encode(indexToLocFormat)
    };
    if (this.font.os2.exists) {
      tables["OS/2"] = this.font.os2.raw();
    }
    return this.font.directory.encode(tables);
  };
  return Subset;
})();

jsPDF.API.PDFObject = (function() {
  var pad;

  function PDFObject() {}
  pad = function(str, length) {
    return (Array(length + 1).join("0") + str).slice(-length);
  };
  /*****************************************************************************/
  /* function : convert                                                        */
  /* comment :Converts pdf tag's / FontBBox and array values in / W to strings */
  /*****************************************************************************/
  PDFObject.convert = function(object) {
    var e, items, key, out, val;
    if (Array.isArray(object)) {
      items = (function() {
        var _i, _len, _results;
        _results = [];
        for (_i = 0, _len = object.length; _i < _len; _i++) {
          e = object[_i];
          _results.push(PDFObject.convert(e));
        }
        return _results;
      })().join(" ");
      return "[" + items + "]";
    } else if (typeof object === "string") {
      return "/" + object;
    } else if (object != null ? object.isString : void 0) {
      return "(" + object + ")";
    } else if (object instanceof Date) {
      return (
        "(D:" +
        pad(object.getUTCFullYear(), 4) +
        pad(object.getUTCMonth(), 2) +
        pad(object.getUTCDate(), 2) +
        pad(object.getUTCHours(), 2) +
        pad(object.getUTCMinutes(), 2) +
        pad(object.getUTCSeconds(), 2) +
        "Z)"
      );
    } else if ({}.toString.call(object) === "[object Object]") {
      out = ["<<"];
      for (key in object) {
        val = object[key];
        out.push("/" + key + " " + PDFObject.convert(val));
      }
      out.push(">>");
      return out.join("\n");
    } else {
      return "" + object;
    }
  };
  return PDFObject;
})();

export { AcroForm, AcroFormAppearance, AcroFormButton, AcroFormCheckBox, AcroFormChoiceField, AcroFormComboBox, AcroFormEditBox, AcroFormListBox, AcroFormPasswordField, AcroFormPushButton, AcroFormRadioButton, AcroFormTextField, GState, ShadingPattern, TilingPattern, jsPDF as default, jsPDF };
