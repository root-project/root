/// @file EveManager.js

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
       this.last_json = [];
   }

    EveManager.prototype.Update = function(arr) {

        // this.last_arr = arr;

        if (arr[0].fTotalBinarySize)
            this.last_json.push(arr);
        
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

    EveManager.prototype.UpdateBinary = function(rawdata, offset) {
        if (!this.last_json) return;


        if (!rawdata.byteLength) return;

        var arr = this.last_json.shift();


       var lastoff = 0;
        
        for (var n=1; n<arr.length;++n)
        {
            var elem = arr[n];

            // console.log('elem', elem.fName, elem.rnr_offset);
             
            if (!elem.render_data) continue;

            var rd = elem.render_data;
            var off = offset + rd.rnr_offset;

            var obj = this.GetElement(elem.fElementId);

            console.log('elem', elem.fName, off, rawdata.byteLength);

            if (off !== lastoff)
                console.error('Element', elem.fName, 'offset mismatch', off, lastoff);

            if (rd.vert_size) {
                obj.fVertexBuffer = new Float32Array(rawdata, off, rd.vert_size);
                off += rd.vert_size*4;
                // console.log('elems', elem.fName, elem.fVertexBuffer);
            }

            if (rd.norm_size) {
                obj.fNormalBuffer = new Float32Array(rawdata, off, rd.norm_size);
                off += rd.norm_size*4;
            }

            if (rd.index_size) {
                obj.fIndexBuffer = new Int32Array(rawdata, off, rd.index_size);
                off += rd.index_size*4;
            }

            lastoff = off;
        }
        
        if (lastoff !== rawdata.byteLength)
            console.error('Raw data decoding error - length mismatch', lastoff, rawdata.byteLength);
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

    /** Create model, which can be used in TreeView */
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
