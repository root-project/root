// JSIO.core.js
//
// core methods for Javascript IO.
//
// by Dino Chiesa
//
// Tue, 19 Jan 2010  17:44
//
// Licensed under the Ms-PL, see
// the accompanying License.txt file
//


(function(){
    if (typeof JSIO == "object"){
        var e1 = new Error("JSIO is already defined");
        e1.source = "JSIO.core.js";
        throw e1;
    }

    JSIO = {};

    JSIO.version = "1.3 2011May10";

    // Format a number as hex.  Quantities over 7ffffff will be displayed properly.
    JSIO.decimalToHexString = function(number, digits) {
        if (number < 0) {
            number = 0xFFFFFFFF + number + 1;
        }
        var r1 = number.toString(16).toUpperCase();
        if (digits) {
            r1 = "00000000" + r1;
            r1 = r1.substring(r1.length - digits);
        }
        return r1;
    };

    JSIO.stringOfLength = function (charCode, length) {
        var s3 = "";
        for (var i = 0; i < length; i++) {
            s3 += String.fromCharCode(charCode);
        }
        return s3;
    };

    JSIO.formatByteArray = function(b) {
        var s1 = "0000  ";
        var s2 = "";
        for (var i = 0; i < b.length; i++) {
            if (i !== 0 && i % 16 === 0) {
                s1 += "    " + s2 +"\n" + JSIO.decimalToHexString(i, 4) + "  ";
                s2 = "";
            }
            s1 += JSIO.decimalToHexString(b[i], 2) + " ";
            if (b[i] >=32 && b[i] <= 126) {
                s2 += String.fromCharCode(b[i]);
            } else {
                s2 += ".";
            }
        }
        if (s2.length > 0) {
            s1 += JSIO.stringOfLength(32, ((i%16>0)? ((16 - i%16) * 3) : 0) + 4) + s2;
        }
        return s1;
    };

    JSIO.htmlEscape = function(str) {
        return str
            .replace(new RegExp( "&", "g" ), "&amp;")
            .replace(new RegExp( "<", "g" ), "&lt;")
            .replace(new RegExp( ">", "g" ), "&gt;")
            .replace(new RegExp( "\x13", "g" ), "<br/>")
            .replace(new RegExp( "\x10", "g" ), "<br/>");
    };


})();

/// JSIO.core.js ends

