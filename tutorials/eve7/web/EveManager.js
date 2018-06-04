/// @file JSRootPainter.more.js
/// Part of JavaScript ROOT graphics with more classes like TEllipse, TLine, ...
/// Such classes are rarely used and therefore loaded only on demand

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootCore'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"));
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.more.js');

      factory(JSROOT);
   }
} (function(JSROOT) {

   "use strict";

   // JSROOT.sources.push("evemgr");
  

   function EveManager() {
       this.map = [];
       this.childs = [];
   }

    EveManager.prototype.Update = function(arr) {
        for (var n=1; n<arr.length;++n) {
            var elem = arr[n];

            var obj = this.map[elem.fElementId];

            if (!obj) {
                // element was not exists up to now
                var parent = null;
                if (elem.fMotherId !== 0) {
                    parent = this.map[elem.fMotherId];
                } else {
                    parent = this;
                }
                if (!parent) {
                    console.error('Parent object ' + elem.fMotherId + ' does not exists - why?');
                    return;
                }

                if (parent.childs === undefined)
                    parent.childs = [];

                parent.childs.push(elem);

                this.map[elem.fElementId] = elem;
                    
            } else {
                // update existing element

                // just copy all properties from new object info to existing one 
                JSROOT.extend(obj, elem);

                //obj.fMotherId = elem.fMotherId;
              
            }
            
        }
       
    }


   JSROOT.EveManager = EveManager;

   return JSROOT;

}));
