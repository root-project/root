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

                obj = this.map[elem.fElementId] = elem;
                    
            } else {
                // update existing element

                // just copy all properties from new object info to existing one 
                JSROOT.extend(obj, elem);

                //obj.fMotherId = elem.fMotherId;
              
            }

            // obj.fVisible = !!obj.fName;

            // obj.fType = "Detail";
            
        }
       
    }

    EveManager.prototype.CanEdit = function(elem) {
        if (elem._typename=="ROOT::Experimental::TEvePointSet") return true;
	if (elem._typename=="ROOT::Experimental::TEveJetCone") return true;
        if (elem._typename=="ROOT::Experimental::TEveTrack") return true;
        return false;
    }

    EveManager.prototype.AnyVisible = function(arr) {
        if (!arr) return false;
        for (var k=0;k<arr.length;++k) {
           if (arr[k].fName) return true;
        }
        return false;
    }

    /** Returns element with given ID */
    EveManager.prototype.GetElement = function(id) {
        return this.map[id];
    }

    EveManager.prototype.CreateModel = function(tgt, src) {
       
        
        if (tgt === undefined) {
            tgt = [];
            src = this.childs;
            // console.log('original model', src);
        }

        for (var n=0;n<src.length;++n) {
            var elem = src[n];
            
            var newelem = { fName: elem.fName, id: elem.fElementId };

            if (this.CanEdit(elem))
                newelem.fType = "DetailAndActive";
              else
                newelem.fType = "Active";
            
            tgt.push(newelem);
            if ((elem.childs !== undefined) && this.AnyVisible(elem.childs))
                newelem.childs = this.CreateModel([], elem.childs);

        }

        return tgt;
    }


   JSROOT.EveManager = EveManager;

   return JSROOT;

}));
