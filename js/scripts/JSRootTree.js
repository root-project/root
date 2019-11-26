/// @file JSRootTree.js
/// Collect all TTree-relevant methods like reading and processing

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootCore', 'JSRootIOEvolution', 'JSRootMath'], factory );
   } else if (typeof exports === 'object' && typeof module !== 'undefined') {
      factory(require("./JSRootCore.js"), require("./JSRootIOEvolution.js"), require("./JSRootMath.js"));
   } else {
      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootTree.js');
      if (typeof JSROOT.IO != 'object')
         throw new Error('JSROOT.IO not defined', 'JSRootTree.js');
      factory(JSROOT);
   }
} (function(JSROOT) {

   "use strict";

   JSROOT.sources.push("tree");

   JSROOT.BranchType = { kLeafNode: 0, kBaseClassNode: 1, kObjectNode: 2, kClonesNode: 3,
                         kSTLNode: 4, kClonesMemberNode: 31, kSTLMemberNode: 41 };

   JSROOT.IO.BranchBits = {
      kDoNotProcess: JSROOT.BIT(10), // Active bit for branches
      kIsClone: JSROOT.BIT(11), // to indicate a TBranchClones
      kBranchObject: JSROOT.BIT(12), // branch is a TObject*
      kBranchAny: JSROOT.BIT(17), // branch is an object*
      kAutoDelete: JSROOT.BIT(15),
      kDoNotUseBufferMap: JSROOT.BIT(22) // If set, at least one of the entry in the branch will use the buffer's map of classname and objects.
   }


   /**
    * @summary class to read data from TTree
    *
    * @constructor
    * @memberof JSROOT
    */
   function TSelector() {
      this.branches = []; // list of branches to read
      this.names = []; // list of member names for each branch in tgtobj
      this.directs = []; // indication if only branch without any children should be read
      this.break_execution = 0;
      this.tgtobj = {};
   }

   /** @summary Add branch to the selector
    * @desc Either branch name or branch itself should be specified
    * Second parameter defines member name in the tgtobj
    * If selector.AddBranch("px", "read_px") is called,
    * branch will be read into selector.tgtobj.read_px member
    * @param {string} branch - name of branch (or branch object itself}
    * @param {string} name - member name in tgtobj where data will be read
    */

   TSelector.prototype.AddBranch = function(branch, name, direct) {
      if (!name)
         name = (typeof branch === 'string') ? branch : ("br" + this.branches.length);
      this.branches.push(branch);
      this.names.push(name);
      this.directs.push(direct);
      return this.branches.length-1;
   }

   TSelector.prototype.indexOfBranch = function(branch) {
      return this.branches.indexOf(branch);
   }

   TSelector.prototype.nameOfBranch = function(indx) {
      return this.names[indx];
   }

   TSelector.prototype.ShowProgress = function(value) {
      // this function can be used to check current TTree progress
   }

   /** @summary call this function to abort processing */
   TSelector.prototype.Abort = function() {
      this.break_execution = -1111;
   }

   /** @summary function called before start processing
    * @abstract
    * @param {object} tree - tree object
    */
   TSelector.prototype.Begin = function(tree) {
   }

   /** @summary function called when next entry extracted from the tree
    * @abstract
    * @param {number} entry - read entry number
    */
   TSelector.prototype.Process = function(entry) {
   }

   /** @summary function called at the very end of processing
    * @abstract
    * @param {boolean} res - true if all data were correctly processed
    */
   TSelector.prototype.Terminate = function(res) {
   }

   // =================================================================

   JSROOT.CheckArrayPrototype = function(arr, check_content) {
      // return 0 when not array
      // 1 - when arbitrary array
      // 2 - when plain (1-dim) array with same-type content
      if (typeof arr !== 'object') return 0;
      var proto = Object.prototype.toString.apply(arr);
      if (proto.indexOf('[object')!==0) return 0;
      var pos = proto.indexOf('Array]');
      if (pos < 0) return 0;
      if (pos > 8) return 2; // this is typed array like Int32Array

      if (!check_content) return 1; //
      var typ, plain = true;
      for (var k=0;k<arr.length;++k) {
         var sub = typeof arr[k];
         if (!typ) typ = sub;
         if (sub!==typ) { plain = false; break; }
         if ((sub=="object") && JSROOT.CheckArrayPrototype(arr[k])) { plain = false; break; }
      }

      return plain ? 2 : 1;
   }

   function ArrayIterator(arr, select, tgtobj) {
      // class used to iterate over all array indexes until number value

      this.object = arr;
      this.value = 0; // value always used in iterator
      this.arr = []; // all arrays
      this.indx = []; // all indexes
      this.cnt = -1; // current index counter
      this.tgtobj = tgtobj;

      if (typeof select === 'object')
         this.select = select; // remember indexes for selection
      else
         this.select = []; // empty array, undefined for each dimension means iterate over all indexes
   }

   ArrayIterator.prototype.next = function() {
      var obj, typ, cnt = this.cnt, seltyp;

      if (cnt >= 0) {

         if (++this.fastindx < this.fastlimit) {
            this.value = this.fastarr[this.fastindx];
            return true;
         }

         while (--cnt >= 0) {
            if ((this.select[cnt]===undefined) && (++this.indx[cnt] < this.arr[cnt].length)) break;
         }
         if (cnt < 0) return false;
      }

      while (true) {

         obj = (cnt < 0) ? this.object : (this.arr[cnt])[this.indx[cnt]];

         typ = obj ? typeof obj : "any";

         if (typ === "object") {
            if (obj._typename !== undefined) {
               if (JSROOT.IsRootCollection(obj)) { obj = obj.arr; typ = "array"; }
                                            else typ = "any";
            } else if (!isNaN(obj.length) && (JSROOT.CheckArrayPrototype(obj)>0)) {
               typ = "array";
            } else {
               typ = "any";
            }
         }

         if (this.select[cnt+1]=="$self$") {
            this.value = obj;
            this.fastindx = this.fastlimit = 0;
            this.cnt = cnt+1;
            return true;
         }

         if ((typ=="any") && (typeof this.select[cnt+1] === "string")) {
            // this is extraction of the member from arbitrary class
            this.arr[++cnt] = obj;
            this.indx[cnt] = this.select[cnt]; // use member name as index
            continue;
         }

         if ((typ === "array") && ((obj.length > 0) || (this.select[cnt+1]==="$size$"))) {
            this.arr[++cnt] = obj;
            switch (this.select[cnt]) {
               case undefined: this.indx[cnt] = 0; break;
               case "$last$": this.indx[cnt] = obj.length-1; break;
               case "$size$":
                  this.value = obj.length;
                  this.fastindx = this.fastlimit = 0;
                  this.cnt = cnt;
                  return true;
                  break;
               default:
                  if (!isNaN(this.select[cnt])) {
                     this.indx[cnt] = this.select[cnt];
                     if (this.indx[cnt] < 0) this.indx[cnt] = obj.length-1;
                  } else {
                     // this is compile variable as array index - can be any expression
                     this.select[cnt].Produce(this.tgtobj);
                     this.indx[cnt] = Math.round(this.select[cnt].get(0));
                  }
            }
         } else {

            if (cnt<0) return false;

            this.value = obj;
            if (this.select[cnt]===undefined) {
               this.fastarr = this.arr[cnt];
               this.fastindx = this.indx[cnt];
               this.fastlimit = this.fastarr.length;
            } else {
               this.fastindx = this.fastlimit = 0; // no any iteration on that level
            }

            this.cnt = cnt;
            return true;
         }
      }

      // unreachable code
      // return false;
   }

   ArrayIterator.prototype.reset = function() {
      this.arr = [];
      this.indx = [];
      delete this.fastarr;
      this.cnt = -1;
      this.value = 0;
   }

   // ============================================================================

   function TDrawVariable(globals) {
      // object with single variable in TTree::Draw expression
      this.globals = globals;

      this.code = "";
      this.brindex = []; // index of used branches from selector
      this.branches = []; // names of branches in target object
      this.brarray = []; // array specifier for each branch
      this.func = null; // generic function for variable calculation

      this.kind = undefined;
      this.buf = []; // buffer accumulates temporary values
   }

   TDrawVariable.prototype.Parse = function(tree,selector,code,only_branch,branch_mode) {
      // when only_branch specified, its placed in the front of the expression

      function is_start_symbol(symb) {
         if ((symb >= "A") && (symb <= "Z")) return true;
         if ((symb >= "a") && (symb <= "z")) return true;
         return (symb === "_");
      }

      function is_next_symbol(symb) {
         if (is_start_symbol(symb)) return true;
         if ((symb >= "0") && (symb <= "9")) return true;
         return false;
      }

      if (!code) code = ""; // should be empty string at least

      this.code = (only_branch ? only_branch.fName : "") + code;

      var pos = 0, pos2 = 0, br = null;
      while ((pos < code.length) || only_branch) {

         var arriter = [];

         if (only_branch) {
            br = only_branch;
            only_branch = undefined;
         } else {
            // first try to find branch
            while ((pos < code.length) && !is_start_symbol(code[pos])) pos++;
            pos2 = pos;
            while ((pos2 < code.length) && (is_next_symbol(code[pos2]) || code[pos2]===".")) pos2++;
            if (code[pos2]=="$") {
               var repl = "";
               switch (code.substr(pos, pos2-pos)) {
                  case "LocalEntry":
                  case "Entry": repl = "arg.$globals.entry"; break;
                  case "Entries": repl = "arg.$globals.entries"; break;
               }
               if (repl) {
                  code = code.substr(0, pos) + repl + code.substr(pos2+1);
                  pos = pos + repl.length;
                  continue;
               }
            }

            br = tree.FindBranch(code.substr(pos, pos2-pos), true);
            if (!br) { pos = pos2+1; continue; }

            // when full id includes branch name, replace only part of extracted expression
            if (br.branch && (br.rest!==undefined)) {
               pos2 -= br.rest.length;
               branch_mode = br.read_mode; // maybe selection of the sub-object done
               br = br.branch;
            }

            // when code ends with the point - means object itself will be accessed
            // sometime branch name itself ends with the point
            if ((pos2>=code.length-1) && (code[code.length-1]===".")) {
               arriter.push("$self$");
               pos2 = code.length;
            }
         }

         // now extract all levels of iterators
         while (pos2 < code.length) {

            if ((code[pos2]==="@") && (code.substr(pos2,5)=="@size") && (arriter.length==0)) {
               pos2+=5;
               branch_mode = true;
               break;
            }

            if (code[pos2] === ".") {
               // this is object member
               var prev = ++pos2;

               if ((code[prev]==="@") && (code.substr(prev,5)==="@size")) {
                  arriter.push("$size$");
                  pos2+=5;
                  break;
               }

               if (!is_start_symbol(code[prev])) {
                  arriter.push("$self$"); // last point means extraction of object itself
                  break;
               }

               while ((pos2 < code.length) && is_next_symbol(code[pos2])) pos2++;

               // this is looks like function call - do not need to extract member with
               if (code[pos2]=="(") { pos2 = prev-1; break; }

               // this is selection of member, but probably we need to activate iterator for ROOT collection
               if ((arriter.length===0) && br) {
                  // TODO: if selected member is simple data type - no need to make other checks - just break here
                  if ((br.fType === JSROOT.BranchType.kClonesNode) || (br.fType === JSROOT.BranchType.kSTLNode)) {
                     arriter.push(undefined);
                  } else {
                     var objclass = JSROOT.IO.GetBranchObjectClass(br, tree, false, true);
                     if (objclass && JSROOT.IsRootCollection(null, objclass)) arriter.push(undefined);
                  }
               }
               arriter.push(code.substr(prev, pos2-prev));
               continue;
            }

            if (code[pos2]!=="[") break;

            // simple []
            if (code[pos2+1]=="]") { arriter.push(undefined); pos2+=2; continue; }

            var prev = pos2++, cnt = 0;
            while ((pos2 < code.length) && ((code[pos2]!="]") || (cnt>0))) {
               if (code[pos2]=='[') cnt++; else if (code[pos2]==']') cnt--;
               pos2++;
            }
            var sub = code.substr(prev+1, pos2-prev-1);
            switch(sub) {
               case "":
               case "$all$": arriter.push(undefined); break;
               case "$last$": arriter.push("$last$"); break;
               case "$size$": arriter.push("$size$"); break;
               case "$first$": arriter.push(0); break;
               default:
                  if (!isNaN(parseInt(sub))) {
                     arriter.push(parseInt(sub));
                  } else {
                     // try to compile code as draw variable
                     var subvar = new TDrawVariable(this.globals);
                     if (!subvar.Parse(tree,selector, sub)) return false;
                     arriter.push(subvar);
                  }
            }
            pos2++;
         }

         if (arriter.length===0) arriter = undefined; else
         if ((arriter.length===1) && (arriter[0]===undefined)) arriter = true;

         var indx = selector.indexOfBranch(br);
         if (indx<0) indx = selector.AddBranch(br, undefined, branch_mode);

         branch_mode = undefined;

         this.brindex.push(indx);
         this.branches.push(selector.nameOfBranch(indx));
         this.brarray.push(arriter);

         // this is simple case of direct usage of the branch
         if ((pos===0) && (pos2 === code.length) && (this.branches.length===1)) {
            this.direct_branch = true; // remember that branch read as is
            return true;
         }

         var replace = "arg.var" + (this.branches.length-1);

         code = code.substr(0, pos) + replace + code.substr(pos2);

         pos = pos + replace.length;
      }

      // support usage of some standard TMath functions
      code = code.replace(/TMath::Exp\(/g, 'Math.exp(')
                 .replace(/TMath::Abs\(/g, 'Math.abs(')
                 .replace(/TMath::Prob\(/g, 'arg.$math.Prob(')
                 .replace(/TMath::Gaus\(/g, 'arg.$math.Gaus(');

      this.func = new Function("arg", "return (" + code + ")");

      return true;
   }

   TDrawVariable.prototype.is_dummy = function() {
      return (this.branches.length === 0) && !this.func;
   }

   TDrawVariable.prototype.Produce = function(obj) {
      // after reading tree braches into the object, calculate variable value

      this.length = 1;
      this.isarray = false;

      if (this.is_dummy()) {
         this.value = 1.; // used as dummy weight variable
         this.kind = "number";
         return;
      }

      var arg = { $globals: this.globals, $math: JSROOT.Math }, usearrlen = -1, arrs = [];
      for (var n=0;n<this.branches.length;++n) {
         var name = "var" + n;
         arg[name] = obj[this.branches[n]];

         // try to check if branch is array and need to be iterated
         if (this.brarray[n]===undefined)
            this.brarray[n] = (JSROOT.CheckArrayPrototype(arg[name]) > 0) || JSROOT.IsRootCollection(arg[name]);

         // no array - no pain
         if (this.brarray[n]===false) continue;

         // check if array can be used as is - one dimension and normal values
         if ((this.brarray[n]===true) && (JSROOT.CheckArrayPrototype(arg[name], true) === 2)) {
            // plain array, can be used as is
            arrs[n] = arg[name];
         } else {
            var iter = new ArrayIterator(arg[name], this.brarray[n], obj);
            arrs[n] = [];
            while (iter.next()) arrs[n].push(iter.value);
         }
         if ((usearrlen < 0) || (usearrlen < arrs[n].length)) usearrlen = arrs[n].length;
      }

      if (usearrlen < 0) {
         this.value = this.direct_branch ? arg.var0 : this.func(arg);
         if (!this.kind) this.kind = typeof this.value;
         return;
      }

      if (usearrlen == 0) {
         // empty array - no any histogram should be filled
         this.length = 0;
         this.value = 0;
         return;
      }

      this.length = usearrlen;
      this.isarray = true;

      if (this.direct_branch) {
         this.value = arrs[0]; // just use array
      } else {
         this.value = new Array(usearrlen);

         for (var k=0;k<usearrlen;++k) {
            for (var n=0;n<this.branches.length;++n) {
               if (arrs[n]) arg["var"+n] = arrs[n][k];
            }
            this.value[k] = this.func(arg);
         }
      }

      if (!this.kind) this.kind = typeof this.value[0];
   }

   TDrawVariable.prototype.get = function(indx) {
      return this.isarray ? this.value[indx] : this.value;
   }

   TDrawVariable.prototype.AppendArray = function(tgtarr) {
      // append array to the buffer

      this.buf = this.buf.concat(tgtarr[this.branches[0]]);
   }

   // =============================================================================

   /**
    * @summary Selector class for TTree::Draw function
    *
    * @constructor
    * @memberof JSROOT
    * @augments JSROOT.TSelector
    */
   function TDrawSelector(callback) {
      TSelector.call(this);

      this.ndim = 0;
      this.vars = []; // array of expression variables
      this.cut = null; // cut variable
      this.hist = null;
      this.histo_callback = callback;
      this.histo_drawopt = "";
      this.hist_name = "$htemp";
      this.hist_title = "Result of TTree::Draw";
      this.graph = false;
      this.hist_args = []; // arguments for histogram creation
      this.arr_limit = 1000;  // number of accumulated items before create histogram
      this.htype = "F";
      this.monitoring = 0;
      this.globals = {}; // object with global parameters, which could be used in any draw expression
      this.last_progress = 0;
      this.aver_diff = 0;
   }

   TDrawSelector.prototype = Object.create(TSelector.prototype);

   TDrawSelector.prototype.ParseParameters = function(tree, args, expr) {

      if (!expr || (typeof expr !== "string")) return "";

      // parse parameters which defined at the end as expression;par1name:par1value;par2name:par2value
      var pos = expr.lastIndexOf(";");
      while (pos>=0) {
         var parname = expr.substr(pos+1), parvalue = undefined;
         expr = expr.substr(0,pos);
         pos = expr.lastIndexOf(";");

         var separ = parname.indexOf(":");
         if (separ>0) { parvalue = parname.substr(separ+1); parname = parname.substr(0, separ);  }

         var intvalue = parseInt(parvalue);
         if (!parvalue || isNaN(intvalue)) intvalue = undefined;

         switch (parname) {
            case "num":
            case "entries":
            case "numentries":
               if (parvalue==="all") args.numentries = tree.fEntries; else
               if (parvalue==="half") args.numentries = Math.round(tree.fEntries/2); else
               if (intvalue !== undefined) args.numentries = intvalue;
               break;
            case "first":
               if (intvalue !== undefined) args.firstentry = intvalue;
               break;
            case "mon":
            case "monitor":
               args.monitoring = (intvalue !== undefined) ? intvalue : 5000;
               break;
            case "player":
               args.player = true;
               break;
            case "dump":
               args.dump = true;
               break;
            case "maxseg":
            case "maxrange":
               if (intvalue) tree.$file.fMaxRanges = intvalue;
               break;
            case "accum":
               if (intvalue) this.arr_limit = intvalue;
               break;
            case "htype":
               if (parvalue && (parvalue.length===1)) {
                  this.htype = parvalue.toUpperCase();
                  if ((this.htype!=="C") && (this.htype!=="S") && (this.htype!=="I")
                       && (this.htype!=="F") && (this.htype!=="L") && (this.htype!=="D")) this.htype = "F";
               }
               break;
            case "hbins" :
               this.hist_nbins = parseInt(parvalue);
               if (isNaN(this.hist_nbins) || (this.hist_nbins<=3)) delete this.hist_nbins;
               break;
            case "drawopt":
               args.drawopt = parvalue;
               break;
            case "graph":
               args.graph = intvalue || true;
               break;
         }
      }

      pos = expr.lastIndexOf(">>");
      if (pos>=0) {
         var harg = expr.substr(pos+2).trim();
         expr = expr.substr(0,pos).trim();
         pos = harg.indexOf("(");
         if (pos>0) {
            this.hist_name = harg.substr(0, pos);
            harg = harg.substr(pos);
         }
         if (harg === "dump") {
            args.dump = true;
         } else if (harg.indexOf("Graph") == 0) {
           args.graph = true;
         } else if (pos<0) {
            this.hist_name = harg;
         } else  if ((harg[0]=="(") && (harg[harg.length-1]==")"))  {
            harg = harg.substr(1,harg.length-2).split(",");
            var isok = true;
            for (var n=0;n<harg.length;++n) {
               harg[n] = (n%3===0) ? parseInt(harg[n]) : parseFloat(harg[n]);
               if (isNaN(harg[n])) isok = false;
            }
            if (isok) this.hist_args = harg;
         }
      }

      if (args.dump) {
         this.dump_values = true;
         args.reallocate_objects = true;
         if (args.numentries===undefined) args.numentries = 10;
      }

      return expr;
   }

   TDrawSelector.prototype.ParseDrawExpression = function(tree, args) {

      // parse complete expression
      var expr = this.ParseParameters(tree, args, args.expr), cut = "";

      // parse option for histogram creation
      this.hist_title = "drawing '" + expr + "' from " + tree.fName;

      var pos = 0;
      if (args.cut) {
         cut = args.cut;
      } else {
         pos = expr.replace(/TMath::/g, 'TMath__').lastIndexOf("::"); // avoid confusion due-to :: in the namespace
         if (pos>0) {
            cut = expr.substr(pos+2).trim();
            expr = expr.substr(0,pos).trim();
         }
      }

      args.parse_expr = expr;
      args.parse_cut = cut;

      // var names = expr.split(":"); // to allow usage of ? operator, we need to handle : as well
      var names = [], nbr1 = 0, nbr2 = 0, prev = 0;
      for (pos=0; pos < expr.length; ++pos) {
         switch (expr[pos]) {
            case "(" : nbr1++; break;
            case ")" : nbr1--; break;
            case "[" : nbr2++; break;
            case "]" : nbr2--; break;
            case ":" :
               if (expr[pos+1]==":") { pos++; continue; }
               if (!nbr1 && !nbr2 && (pos>prev)) names.push(expr.substr(prev,pos-prev));
               prev = pos+1;
               break;
         }
      }
      if (!nbr1 && !nbr2 && (pos>prev)) names.push(expr.substr(prev,pos-prev));

      if ((names.length < 1) || (names.length > 3)) return false;

      this.ndim = names.length;

      var is_direct = !cut;

      for (var n=0;n<this.ndim;++n) {
         this.vars[n] = new TDrawVariable(this.globals);
         if (!this.vars[n].Parse(tree, this, names[n])) return false;
         if (!this.vars[n].direct_branch) is_direct = false;
      }

      this.cut = new TDrawVariable(this.globals);
      if (cut)
         if (!this.cut.Parse(tree, this, cut)) return false;

      if (!this.branches.length) {
         console.warn('no any branch is selected');
         return false;
      }

      if (is_direct) this.ProcessArrays = this.ProcessArraysFunc;

      this.monitoring = args.monitoring;

      this.graph = args.graph;

      if (args.drawopt !== undefined)
         this.histo_drawopt = args.drawopt;
      else
         this.histo_drawopt = (this.ndim===2) ? "col" : "";

      return true;
   }

   TDrawSelector.prototype.DrawOnlyBranch = function(tree, branch, expr, args) {
      this.ndim = 1;

      if (expr.indexOf("dump")==0) expr = ";" + expr;

      expr = this.ParseParameters(tree, args, expr);

      this.monitoring = args.monitoring;

      if (args.dump) {
         this.dump_values = true;
         args.reallocate_objects = true;
      }

      if (this.dump_values) {

         this.hist = []; // array of dump objects

         this.leaf = args.leaf;

         // branch object remains, therefore we need to copy fields to see them all
         this.copy_fields = ((args.branch.fLeaves && (args.branch.fLeaves.arr.length > 1)) ||
                              (args.branch.fBranches && (args.branch.fBranches.arr.length > 0))) && !args.leaf;

         this.AddBranch(branch, "br0", args.direct_branch); // add branch

         this.Process = this.ProcessDump;

         return true;
      }

      this.vars[0] = new TDrawVariable(this.globals);
      if (!this.vars[0].Parse(tree, this, expr, branch, args.direct_branch)) return false;
      this.hist_title = "drawing branch '" + branch.fName + (expr ? "' expr:'" + expr : "") + "'  from " + tree.fName;

      this.cut = new TDrawVariable(this.globals);

      if (this.vars[0].direct_branch) this.ProcessArrays = this.ProcessArraysFunc;

      return true;
   }

   TDrawSelector.prototype.Begin = function(tree) {
      this.globals.entries = tree.fEntries;

      if (this.monitoring)
         this.lasttm = new Date().getTime();
   }

   TDrawSelector.prototype.ShowProgress = function(value) {
      // this function should be defined not here

      if (typeof document == 'undefined' || !JSROOT.progress) return;

      if ((value===undefined) || isNaN(value)) return JSROOT.progress();

      if (this.last_progress !== value) {
         var diff = value - this.last_progress;
         if (!this.aver_diff) this.aver_diff = diff;
         this.aver_diff = diff*0.3 + this.aver_diff*0.7;
      }

      var ndig = 0;
      if (this.aver_diff <= 0) ndig = 0; else
      if (this.aver_diff < 0.0001) ndig = 3; else
      if (this.aver_diff < 0.001) ndig = 2; else
      if (this.aver_diff < 0.01) ndig = 1;

      var main_box = document.createElement("p"),
          text_node = document.createTextNode("TTree draw " + (value*100).toFixed(ndig) + " %  "),
          selector = this;

      main_box.appendChild(text_node);
      main_box.title = "Click on element to break drawing";

      main_box.onclick = function() {
         if (++selector.break_execution<3) {
            main_box.title = "Tree draw will break after next I/O operation";
            return text_node.nodeValue = "Breaking ... ";
         }
         selector.Abort();
         JSROOT.progress();
      }

      JSROOT.progress(main_box);
      this.last_progress = value;
   }

   TDrawSelector.prototype.GetBitsBins = function(nbits, res) {
      res.nbins = res.max = nbits;
      res.fLabels = JSROOT.Create("THashList");
      for (var k=0;k<nbits;++k) {
         var s = JSROOT.Create("TObjString");
         s.fString = k.toString();
         s.fUniqueID = k+1;
         res.fLabels.Add(s);
      }
      return res;
   }

   TDrawSelector.prototype.GetMinMaxBins = function(axisid, nbins) {

      var res = { min: 0, max: 0, nbins: nbins, k: 1., fLabels: null, title: "" };

      if (axisid >= this.ndim) return res;

      var arr = this.vars[axisid].buf;

      res.title = this.vars[axisid].code || "";

      if (this.vars[axisid].kind === "object") {
         // this is any object type
         var typename, similar = true, maxbits = 8;
         for (var k=0;k<arr.length;++k) {
            if (!arr[k]) continue;
            if (!typename) typename = arr[k]._typename;
            if (typename !== arr[k]._typename) similar = false; // check all object types
            if (arr[k].fNbits) maxbits = Math.max(maxbits, arr[k].fNbits+1);
         }

         if (typename && similar) {
            if ((typename==="TBits") && (axisid===0)) {
               this.Fill1DHistogram = this.FillTBitsHistogram;
               if (maxbits % 8) maxbits = (maxbits & 0xfff0) + 8;

               if ((this.hist_name === "bits") && (this.hist_args.length == 1) && this.hist_args[0])
                  maxbits = this.hist_args[0];

               return this.GetBitsBins(maxbits, res);
            }
         }
      }

      if (this.vars[axisid].kind === "string") {
         res.lbls = []; // all labels

         for (var k=0;k<arr.length;++k)
            if (res.lbls.indexOf(arr[k])<0)
               res.lbls.push(arr[k]);

         res.lbls.sort();
         res.max = res.nbins = res.lbls.length;

         res.fLabels = JSROOT.Create("THashList");
         for (var k=0;k<res.lbls.length;++k) {
            var s = JSROOT.Create("TObjString");
            s.fString = res.lbls[k];
            s.fUniqueID = k+1;
            if (s.fString === "") s.fString = "<empty>";
            res.fLabels.Add(s);
         }
      } else if ((axisid === 0) && (this.hist_name === "bits") && (this.hist_args.length <= 1)) {
         this.Fill1DHistogram = this.FillBitsHistogram;
         return this.GetBitsBins(this.hist_args[0] || 32, res);
      } else if (axisid*3 + 2 < this.hist_args.length) {
         res.nbins = this.hist_args[axisid*3];
         res.min = this.hist_args[axisid*3+1];
         res.max = this.hist_args[axisid*3+2];
      } else {



         res.min = Math.min.apply(null, arr);
         res.max = Math.max.apply(null, arr);

         if (this.hist_nbins)
            nbins = res.nbins = this.hist_nbins;

         res.isinteger = (Math.round(res.min)===res.min) && (Math.round(res.max)===res.max);
         if (res.isinteger)
            for (var k=0;k<arr.length;++k)
               if (arr[k]!==Math.round(arr[k])) { res.isinteger = false; break; }

         if (res.isinteger) {
            res.min = Math.round(res.min);
            res.max = Math.round(res.max);
            if (res.max-res.min < nbins*5) {
               res.min -= 1;
               res.max += 2;
               res.nbins = Math.round(res.max - res.min);
            } else {
               var range = (res.max - res.min + 2), step = Math.floor(range / nbins);
               while (step*nbins < range) step++;
               res.max = res.min + nbins*step;
            }
         } else
         if (res.min >= res.max) {
            res.max = res.min;
            if (Math.abs(res.min)<100) { res.min-=1; res.max+=1; } else
            if (res.min>0) { res.min*=0.9; res.max*=1.1; } else { res.min*=1.1; res.max*=0.9; }
         } else {
            res.max += (res.max-res.min)/res.nbins;
         }
      }

      res.k = res.nbins/(res.max-res.min);

      res.GetBin = function(value) {
         var bin = this.lbls ? this.lbls.indexOf(value) : Math.floor((value-this.min)*this.k);
         return (bin<0) ? 0 : ((bin>this.nbins) ? this.nbins+1 : bin+1);
      }

      return res;
   }

   TDrawSelector.prototype.CreateHistogram = function() {
      if (this.hist || !this.vars[0].buf) return;

      if (this.dump_values) {
         // just create array where dumped valus will be collected
         this.hist = [];

         // reassign fill method
         this.Fill1DHistogram = this.Fill2DHistogram = this.Fill3DHistogram = this.DumpValue;
      } else if (this.graph) {
        var N = this.vars[0].buf.length;

        if(this.ndim == 1) {
          // A 1-dimensional graph will just have the x axis as an index
          this.hist = JSROOT.CreateTGraph(N, Array.from(Array(N).keys()), this.vars[0].buf);
        } else if(this.ndim == 2) {
           this.hist = JSROOT.CreateTGraph(N,this.vars[0].buf, this.vars[1].buf);
           delete this.vars[1].buf;
        }

        this.hist.fTitle = this.hist_title;
        this.hist.fName = "Graph";

      } else {

         this.x = this.GetMinMaxBins(0, (this.ndim > 1) ? 50 : 200);

         this.y = this.GetMinMaxBins(1, 50);

         this.z = this.GetMinMaxBins(2, 50);

         switch (this.ndim) {
            case 1: this.hist = JSROOT.CreateHistogram("TH1"+this.htype, this.x.nbins); break;
            case 2: this.hist = JSROOT.CreateHistogram("TH2"+this.htype, this.x.nbins, this.y.nbins); break;
            case 3: this.hist = JSROOT.CreateHistogram("TH3"+this.htype, this.x.nbins, this.y.nbins, this.z.nbins); break;
         }

         this.hist.fXaxis.fTitle = this.x.title;
         this.hist.fXaxis.fXmin = this.x.min;
         this.hist.fXaxis.fXmax = this.x.max;
         this.hist.fXaxis.fLabels = this.x.fLabels;

         if (this.ndim > 1) this.hist.fYaxis.fTitle = this.y.title;
         this.hist.fYaxis.fXmin = this.y.min;
         this.hist.fYaxis.fXmax = this.y.max;
         this.hist.fYaxis.fLabels = this.y.fLabels;

         if (this.ndim > 2) this.hist.fZaxis.fTitle = this.z.title;
         this.hist.fZaxis.fXmin = this.z.min;
         this.hist.fZaxis.fXmax = this.z.max;
         this.hist.fZaxis.fLabels = this.z.fLabels;

         this.hist.fName = this.hist_name;
         this.hist.fTitle = this.hist_title;
         this.hist.$custom_stat = (this.hist_name == "$htemp") ? 111110 : 111111;
      }

      var var0 = this.vars[0].buf, cut = this.cut.buf, len = var0.length;

      if (!this.graph) {
        switch (this.ndim) {
           case 1:
              for (var n=0;n<len;++n)
                 this.Fill1DHistogram(var0[n], cut ? cut[n] : 1.);
              break;
           case 2:
              var var1 = this.vars[1].buf;
              for (var n=0;n<len;++n)
                 this.Fill2DHistogram(var0[n], var1[n], cut ? cut[n] : 1.);
              delete this.vars[1].buf;
              break;
           case 3:
              var var1 = this.vars[1].buf, var2 = this.vars[2].buf;
              for (var n=0;n<len;++n)
                 this.Fill3DHistogram(var0[n], var1[n], var2[n], cut ? cut[n] : 1.);
              delete this.vars[1].buf;
              delete this.vars[2].buf;
              break;
        }
      }

      delete this.vars[0].buf;
      delete this.cut.buf;
   }

   TDrawSelector.prototype.FillTBitsHistogram = function(xvalue, weight) {
      if (!weight || !xvalue || !xvalue.fNbits || !xvalue.fAllBits) return;

      var sz = Math.min(xvalue.fNbits+1, xvalue.fNbytes*8);

      for (var bit=0,mask=1,b=0;bit<sz;++bit) {
         if (xvalue.fAllBits[b] && mask) {
            if (bit <= this.x.nbins)
               this.hist.fArray[bit+1] += weight;
            else
               this.hist.fArray[this.x.nbins+1] += weight;
         }

         mask*=2;
         if (mask>=0x100) { mask = 1; ++b; }
      }
   }

   TDrawSelector.prototype.FillBitsHistogram = function(xvalue, weight) {
      if (!weight) return;

      for (var bit=0,mask=1;bit<this.x.nbins;++bit) {
         if (xvalue & mask) this.hist.fArray[bit+1] += weight;
         mask*=2;
      }
   }

   TDrawSelector.prototype.Fill1DHistogram = function(xvalue, weight) {
      var bin = this.x.GetBin(xvalue);
      this.hist.fArray[bin] += weight;

      if (!this.x.lbls) {
         this.hist.fTsumw += weight;
         this.hist.fTsumwx += weight*xvalue;
         this.hist.fTsumwx2 += weight*xvalue*xvalue;
      }
   }

   TDrawSelector.prototype.Fill2DHistogram = function(xvalue, yvalue, weight) {
      var xbin = this.x.GetBin(xvalue),
          ybin = this.y.GetBin(yvalue);

      this.hist.fArray[xbin+(this.x.nbins+2)*ybin] += weight;
      if (!this.x.lbls && !this.y.lbls) {
         this.hist.fTsumw += weight;
         this.hist.fTsumwx += weight*xvalue;
         this.hist.fTsumwy += weight*yvalue;
         this.hist.fTsumwx2 += weight*xvalue*xvalue;
         this.hist.fTsumwxy += weight*xvalue*yvalue;
         this.hist.fTsumwy2 += weight*yvalue*yvalue;
      }
   }

   TDrawSelector.prototype.Fill3DHistogram = function(xvalue, yvalue, zvalue, weight) {
      var xbin = this.x.GetBin(xvalue),
          ybin = this.y.GetBin(yvalue),
          zbin = this.z.GetBin(zvalue);

      this.hist.fArray[xbin + (this.x.nbins+2) * (ybin + (this.y.nbins+2)*zbin) ] += weight;
      if (!this.x.lbls && !this.y.lbls && !this.z.lbls) {
         this.hist.fTsumw += weight;
         this.hist.fTsumwx += weight*xvalue;
         this.hist.fTsumwy += weight*yvalue;
         this.hist.fTsumwz += weight*zvalue;
         this.hist.fTsumwx2 += weight*xvalue*xvalue;
         this.hist.fTsumwy2 += weight*yvalue*yvalue;
         this.hist.fTsumwz2 += weight*zvalue*zvalue;
         this.hist.fTsumwxy += weight*xvalue*yvalue;
         this.hist.fTsumwxz += weight*xvalue*zvalue;
         this.hist.fTsumwyz += weight*yvalue*zvalue;
      }
   }

   TDrawSelector.prototype.DumpValue = function(v1, v2, v3, v4) {
      var obj;
      switch (this.ndim) {
         case 1: obj = { x: v1, weight: v2 }; break;
         case 2: obj = { x: v1, y: v2, weight: v3 }; break;
         case 3: obj = { x: v1, y: v2, z: v3, weight: v4 }; break;
      }

      if (this.cut.is_dummy()) {
         if (this.ndim===1) obj = v1; else delete obj.weight;
      }

      this.hist.push(obj);
   }

   TDrawSelector.prototype.ProcessArraysFunc = function(entry) {
      // function used when all branches can be read as array
      // most typical usage - histogramming of single branch

      if (this.arr_limit || this.graph) {
         var var0 = this.vars[0], len = this.tgtarr.br0.length,
             var1 = this.vars[1], var2 = this.vars[2];
         if ((var0.buf.length===0) && (len>=this.arr_limit) && !this.graph) {
            // special use case - first array large enough to create histogram directly base on it
            var0.buf = this.tgtarr.br0;
            if (var1) var1.buf = this.tgtarr.br1;
            if (var2) var2.buf = this.tgtarr.br2;
         } else
         for (var k=0;k<len;++k) {
            var0.buf.push(this.tgtarr.br0[k]);
            if (var1) var1.buf.push(this.tgtarr.br1[k]);
            if (var2) var2.buf.push(this.tgtarr.br2[k]);
         }
         var0.kind = "number";
         if (var1) var1.kind = "number";
         if (var2) var2.kind = "number";
         this.cut.buf = null; // do not create buffer for cuts
         if (!this.graph && (var0.buf.length >= this.arr_limit)) {
            this.CreateHistogram();
            this.arr_limit = 0;
         }
      } else {
         var br0 = this.tgtarr.br0, len = br0.length;
         switch(this.ndim) {
            case 1:
               for (var k=0;k<len;++k)
                  this.Fill1DHistogram(br0[k], 1.);
               break;
            case 2:
               var br1 = this.tgtarr.br1;
               for (var k=0;k<len;++k)
                  this.Fill2DHistogram(br0[k], br1[k], 1.);
               break;
            case 3:
               var br1 = this.tgtarr.br1, br2 = this.tgtarr.br2;
               for (var k=0;k<len;++k)
                  this.Fill3DHistogram(br0[k], br1[k], br2[k], 1.);
               break;
         }
      }
   }

   TDrawSelector.prototype.ProcessDump = function(entry) {
      // simple dump of the branch - no need to analyze something

      var res = this.leaf ? this.tgtobj.br0[this.leaf] : this.tgtobj.br0;

      if (res && this.copy_fields) {
         if (JSROOT.CheckArrayPrototype(res)===0) {
            this.hist.push(JSROOT.extend({}, res));
         } else {
            this.hist.push(res);
         }
      } else {
         this.hist.push(res);
      }
   }

   TDrawSelector.prototype.Process = function(entry) {

      this.globals.entry = entry; // can be used in any expression

      this.cut.Produce(this.tgtobj);
      if (!this.dump_values && !this.cut.value) return;

      for (var n=0;n<this.ndim;++n)
         this.vars[n].Produce(this.tgtobj);

      var var0 = this.vars[0], var1 = this.vars[1], var2 = this.vars[2], cut = this.cut;

      if (this.graph || this.arr_limit) {
         switch(this.ndim) {
            case 1:
              for (var n0=0;n0<var0.length;++n0) {
                 var0.buf.push(var0.get(n0));
                 cut.buf.push(cut.value);
              }
              break;
            case 2:
              for (var n0=0;n0<var0.length;++n0)
                 for (var n1=0;n1<var1.length;++n1) {
                    var0.buf.push(var0.get(n0));
                    var1.buf.push(var1.get(n1));
                    cut.buf.push(cut.value);
                 }
              break;
            case 3:
               for (var n0=0;n0<var0.length;++n0)
                  for (var n1=0;n1<var1.length;++n1)
                     for (var n2=0;n2<var2.length;++n2) {
                        var0.buf.push(var0.get(n0));
                        var1.buf.push(var1.get(n1));
                        var2.buf.push(var2.get(n2));
                        cut.buf.push(cut.value);
                     }
               break;
         }
         if (!this.graph && var0.buf.length >= this.arr_limit) {
            this.CreateHistogram();
            this.arr_limit = 0;
         }
      } else if (this.hist) {
         switch(this.ndim) {
            case 1:
               for (var n0=0;n0<var0.length;++n0)
                  this.Fill1DHistogram(var0.get(n0), cut.value);
               break;
            case 2:
               for (var n0=0;n0<var0.length;++n0)
                  for (var n1=0;n1<var1.length;++n1)
                     this.Fill2DHistogram(var0.get(n0), var1.get(n1), cut.value);
               break;
            case 3:
               for (var n0=0;n0<var0.length;++n0)
                  for (var n1=0;n1<var1.length;++n1)
                     for (var n2=0;n2<var2.length;++n2)
                        this.Fill3DHistogram(var0.get(n0), var1.get(n1), var2.get(n2), cut.value);
               break;
         }
      }

      if (this.monitoring && this.hist && !this.dump_values) {
         var now = new Date().getTime();
         if (now - this.lasttm > this.monitoring) {
            this.lasttm = now;
            JSROOT.CallBack(this.histo_callback, this.hist, this.histo_drawopt, true);
         }
      }
   }

   TDrawSelector.prototype.Terminate = function(res) {
      if (res && !this.hist) this.CreateHistogram();

      this.ShowProgress();

      return JSROOT.CallBack(this.histo_callback, this.hist, this.dump_values ? "inspect" : this.histo_drawopt);
   }

   // ======================================================================

   JSROOT.IO.FindBrachStreamerElement = function(branch, file) {
      // return TStreamerElement associated with the branch - if any
      // unfortunately, branch.fID is not number of element in streamer info

      if (!branch || !file || (branch._typename!=="TBranchElement") || (branch.fID<0) || (branch.fStreamerType<0)) return null;

      var s_i = file.FindStreamerInfo(branch.fClassName, branch.fClassVersion, branch.fCheckSum),
          arr = (s_i && s_i.fElements) ? s_i.fElements.arr : null;
      if (!arr) return null;

      var match_name = branch.fName,
          pos = match_name.indexOf("[");
      if (pos>0) match_name = match_name.substr(0, pos);
      pos = match_name.lastIndexOf(".");
      if (pos>0) match_name = match_name.substr(pos+1);

      function match_elem(elem) {
         if (!elem) return false;
         if (elem.fName !== match_name) return false;
         if (elem.fType === branch.fStreamerType) return true;
         if ((elem.fType === JSROOT.IO.kBool) && (branch.fStreamerType === JSROOT.IO.kUChar)) return true;
         if (((branch.fStreamerType===JSROOT.IO.kSTL) || (branch.fStreamerType===JSROOT.IO.kSTL + JSROOT.IO.kOffsetL) ||
              (branch.fStreamerType===JSROOT.IO.kSTLp) || (branch.fStreamerType===JSROOT.IO.kSTLp + JSROOT.IO.kOffsetL))
                 && (elem.fType === JSROOT.IO.kStreamer)) return true;
         console.warn('Should match element', elem.fType, 'with branch', branch.fStreamerType);
         return false;
      }

      // first check branch fID - in many cases gut guess
      if (match_elem(arr[branch.fID])) return arr[branch.fID];

      // console.warn('Missmatch with branch name and extracted element', branch.fName, match_name, (s_elem ? s_elem.fName : "---"));

      for (var k=0;k<arr.length;++k)
         if ((k!==branch.fID) && match_elem(arr[k])) return arr[k];

      console.error('Did not found/match element for branch', branch.fName, 'class', branch.fClassName);

      return null;
   }


   JSROOT.IO.DefineMemberTypeName = function(file, parent_class, member_name) {
      // return type name of given member in the class

      var s_i = file.FindStreamerInfo(parent_class),
          arr = (s_i && s_i.fElements) ? s_i.fElements.arr : null,
          elem = null;
      if (!arr) return "";

      for (var k=0;k<arr.length;++k) {
         if (arr[k].fTypeName === "BASE") {
            var res = JSROOT.IO.DefineMemberTypeName(file, arr[k].fName, member_name);
            if (res) return res;
         } else
         if (arr[k].fName === member_name) { elem = arr[k]; break; }
      }

      if (!elem) return "";

      var clname = elem.fTypeName;
      if (clname[clname.length-1]==="*") clname = clname.substr(0, clname.length-1);

      return clname;
   }

   JSROOT.IO.GetBranchObjectClass = function(branch, tree, with_clones, with_leafs) {
      // return class name of the object, stored in the branch

      if (!branch || (branch._typename!=="TBranchElement")) return "";

      if ((branch.fType === JSROOT.BranchType.kLeafNode) && (branch.fID===-2) && (branch.fStreamerType===-1)) {
         // object where all sub-branches will be collected
         return branch.fClassName;
      }

      if (with_clones && branch.fClonesName && ((branch.fType === JSROOT.BranchType.kClonesNode) || (branch.fType === JSROOT.BranchType.kSTLNode)))
         return branch.fClonesName;

      var s_elem = JSROOT.IO.FindBrachStreamerElement(branch, tree.$file);

      if ((branch.fType === JSROOT.BranchType.kBaseClassNode) && s_elem && (s_elem.fTypeName==="BASE"))
          return s_elem.fName;

      if (branch.fType === JSROOT.BranchType.kObjectNode) {
         if (s_elem && ((s_elem.fType === JSROOT.IO.kObject) || (s_elem.fType === JSROOT.IO.kAny)))
            return s_elem.fTypeName;
         return "TObject";
      }

      if ((branch.fType === JSROOT.BranchType.kLeafNode) && s_elem && with_leafs) {
         if ((s_elem.fType === JSROOT.IO.kObject) || (s_elem.fType === JSROOT.IO.kAny)) return s_elem.fTypeName;
         if (s_elem.fType === JSROOT.IO.kObjectp) return s_elem.fTypeName.substr(0, s_elem.fTypeName.length-1);
      }

      return "";
   }

   JSROOT.IO.MakeMethodsList = function(typename) {
      // create fast list to assign all methods to the object

      var methods = JSROOT.getMethods(typename);

      var res = {
         names : [],
         values : [],
         Create : function() {
            var obj = {};
            for (var n=0;n<this.names.length;++n)
               obj[this.names[n]] = this.values[n];
            return obj;
         }
      }

      res.names.push("_typename"); res.values.push(typename);
      for (var key in methods) {
         res.names.push(key);
         res.values.push(methods[key]);
      }
      return res;
   }

   JSROOT.IO.DetectBranchMemberClass = function(brlst, prefix, start) {
      // try to define classname for the branch member, scanning list of branches
      var clname = "";
      for (var kk=(start || 0); kk<brlst.arr.length; ++kk)
         if ((brlst.arr[kk].fName.indexOf(prefix)===0) && brlst.arr[kk].fClassName) clname = brlst.arr[kk].fClassName;
      return clname;
   }

   /** @namespace JSROOT.TreeMethods
    * @summary these are TTree methods, which are automatically assigned to every TTree */
   JSROOT.TreeMethods = {};

   /** @summary Process selector
    * @param {object} selector - instance of JSROOT.TSelector class
    * @param {object} args - different arguments */
   JSROOT.TreeMethods.Process = function(selector, args) {
      // function similar to the TTree::Process

      if (!args) args = {};

      if (!selector || !this.$file || !selector.branches) {
         console.error('required parameter missing for TTree::Process');
         if (selector) selector.Terminate(false);
         return false;
      }

      // central handle with all information required for reading
      var handle = {
          tree: this, // keep tree reference
          file: this.$file, // keep file reference
          selector: selector, // reference on selector
          arr: [], // list of branches
          curr: -1,  // current entry ID
          current_entry: -1, // current processed entry
          simple_read: true, // all baskets in all used branches are in sync,
          process_arrays: true // one can process all branches as arrays
      };

      var namecnt = 0;

      function CreateLeafElem(leaf, name) {
         // function creates TStreamerElement which corresponds to the elementary leaf
         var datakind = 0;
         switch (leaf._typename) {
            case 'TLeafF': datakind = JSROOT.IO.kFloat; break;
            case 'TLeafD': datakind = JSROOT.IO.kDouble; break;
            case 'TLeafO': datakind = JSROOT.IO.kBool; break;
            case 'TLeafB': datakind = leaf.fIsUnsigned ? JSROOT.IO.kUChar : JSROOT.IO.kChar; break;
            case 'TLeafS': datakind = leaf.fIsUnsigned ? JSROOT.IO.kUShort : JSROOT.IO.kShort; break;
            case 'TLeafI': datakind = leaf.fIsUnsigned ? JSROOT.IO.kUInt : JSROOT.IO.kInt; break;
            case 'TLeafL': datakind = leaf.fIsUnsigned ? JSROOT.IO.kULong64 : JSROOT.IO.kLong64; break;
            case 'TLeafC': datakind = JSROOT.IO.kTString; break; // datakind = leaf.fIsUnsigned ? JSROOT.IO.kUChar : JSROOT.IO.kChar; break;
            default: return null;
         }
         return JSROOT.IO.CreateStreamerElement(name || leaf.fName, datakind);
      }

      function FindInHandle(branch) {
         for (var k=0;k<handle.arr.length;++k)
            if (handle.arr[k].branch === branch) return handle.arr[k];
         return null;
      }

      function AddBranchForReading(branch, target_object, target_name, read_mode) {
         // central method to add branch for reading
         // read_mode == true - read only this branch
         // read_mode == '$child$' is just member of object from for STL or clonesarray
         // read_mode == '<any class name>' is sub-object from STL or clonesarray, happens when such new object need to be created
         // read_mode == '.member_name' select only reading of member_name instead of complete object

         if (typeof branch === 'string')
            branch = handle.tree.FindBranch(branch);

         if (!branch) { console.error('Did not found branch'); return null; }

         var item = FindInHandle(branch);

         if (item) {
            console.error('Branch already configured for reading', branch.fName);
            if (item.tgt !== target_object) console.error('Target object differs');
            return elem;
         }

         if (!branch.fEntries) {
            console.warn('Branch ', branch.fName, ' does not have entries');
            return null;
         }

         // console.log('Add branch', branch.fName);

         item = {
               branch: branch,
               tgt: target_object, // used target object - can be differ for object members
               name: target_name,
               index: -1, // index in the list of read branches
               member: null, // member to read branch
               type: 0, // keep type identifier
               curr_entry: -1, // last processed entry
               raw : null, // raw buffer for reading
               basket : null, // current basket object
               curr_basket: 0,  // number of basket used for processing
               read_entry: -1,  // last entry which is already read
               staged_entry: -1, // entry which is staged for reading
               first_readentry: -1, // first entry to read
               staged_basket: 0,  // last basket staged for reading
               numentries: branch.fEntries,
               numbaskets: branch.fWriteBasket, // number of baskets which can be read from the file
               counters: null, // branch indexes used as counters
               ascounter: [], // list of other branches using that branch as counter
               baskets: [], // array for read baskets,
               staged_prev: 0, // entry limit of previous I/O request
               staged_now: 0, // entry limit of current I/O request
               progress_showtm: 0, // last time when progress was showed
               GetBasketEntry: function(k) {
                  if (!this.branch || (k > this.branch.fMaxBaskets)) return 0;
                  var res = (k < this.branch.fMaxBaskets) ? this.branch.fBasketEntry[k] : 0;
                  if (res) return res;
                  var bskt = (k>0) ? this.branch.fBaskets.arr[k-1] : null;
                  return bskt ? (this.branch.fBasketEntry[k-1] + bskt.fNevBuf) : 0;
               },
               GetTarget: function(tgtobj) {
                  // returns target object which should be used for the branch reading
                  if (!this.tgt) return tgtobj;
                  for (var k=0;k<this.tgt.length;++k) {
                     var sub = this.tgt[k];
                     if (!tgtobj[sub.name]) tgtobj[sub.name] = sub.lst.Create();
                     tgtobj = tgtobj[sub.name];
                  }
                  return tgtobj;
               },
               GetEntry: function(entry) {
                 // This should be equivalent to TBranch::GetEntry() method
                  var shift = entry - this.first_entry, off;
                  if (!this.branch.TestBit(JSROOT.IO.BranchBits.kDoNotUseBufferMap))
                     this.raw.ClearObjectMap();
                  if (this.basket.fEntryOffset) {
                     off = this.basket.fEntryOffset[shift];
                     if (this.basket.fDisplacement)
                        this.raw.fDisplacement = this.basket.fDisplacement[shift];
                  } else {
                     off = this.basket.fKeylen + this.basket.fNevBufSize * shift;
                  }
                  this.raw.locate(off - this.raw.raw_shift);

                  // this.member.func(this.raw, this.GetTarget(tgtobj));
               }
         };

         // last basket can be stored directly with the branch
         while (item.GetBasketEntry(item.numbaskets+1)) item.numbaskets++;

         // check all counters if we
         var nb_branches = branch.fBranches ? branch.fBranches.arr.length : 0,
             nb_leaves = branch.fLeaves ? branch.fLeaves.arr.length : 0,
             leaf = (nb_leaves>0) ? branch.fLeaves.arr[0] : null,
             elem = null, // TStreamerElement used to create reader
             member = null, // member for actual reading of the branch
             is_brelem = (branch._typename==="TBranchElement"),
             child_scan = 0, // scan child branches after main branch is appended
             item_cnt = null, item_cnt2 = null, object_class = "";

         if (branch.fBranchCount) {

            item_cnt = FindInHandle(branch.fBranchCount);

            if (!item_cnt)
               item_cnt = AddBranchForReading(branch.fBranchCount, target_object, "$counter" + namecnt++, true);

            if (!item_cnt) { console.error('Cannot add counter branch', branch.fBranchCount.fName); return null; }

            var BranchCount2 = branch.fBranchCount2;

            if (!BranchCount2 && (branch.fBranchCount.fStreamerType===JSROOT.IO.kSTL) &&
                ((branch.fStreamerType === JSROOT.IO.kStreamLoop) || (branch.fStreamerType === JSROOT.IO.kOffsetL+JSROOT.IO.kStreamLoop))) {
                 // special case when count member from kStreamLoop not assigned as fBranchCount2
                 var elemd = JSROOT.IO.FindBrachStreamerElement(branch, handle.file),
                     arrd = branch.fBranchCount.fBranches.arr;

                 if (elemd && elemd.fCountName && arrd)
                    for(var k=0;k<arrd.length;++k)
                       if (arrd[k].fName === branch.fBranchCount.fName + "." + elemd.fCountName) {
                          BranchCount2 = arrd[k];
                          break;
                       }

                 if (!BranchCount2) console.error('Did not found branch for second counter of kStreamLoop element');
              }

            if (BranchCount2) {
               item_cnt2 = FindInHandle(BranchCount2);

               if (!item_cnt2) item_cnt2 = AddBranchForReading(BranchCount2, target_object, "$counter" + namecnt++, true);

               if (!item_cnt2) { console.error('Cannot add counter branch2', BranchCount2.fName); return null; }
            }
         } else
         if (nb_leaves===1 && leaf && leaf.fLeafCount) {
            var br_cnt = handle.tree.FindBranch(leaf.fLeafCount.fName);

            if (br_cnt) {
               item_cnt = FindInHandle(br_cnt);

               if (!item_cnt) item_cnt = AddBranchForReading(br_cnt, target_object, "$counter" + namecnt++, true);

               if (!item_cnt) { console.error('Cannot add counter branch', br_cnt.fName); return null; }
            }
         }

         function ScanBranches(lst, master_target, chld_kind) {
            if (!lst || !lst.arr.length) return true;

            var match_prefix = branch.fName;
            if (match_prefix[match_prefix.length-1] === ".") match_prefix = match_prefix.substr(0,match_prefix.length-1);
            if ((typeof read_mode=== "string") && (read_mode[0]==".")) match_prefix += read_mode;
            match_prefix+=".";

            for (var k=0;k<lst.arr.length;++k) {
               var br = lst.arr[k];
               if ((chld_kind>0) && (br.fType!==chld_kind)) continue;

               if (br.fType === JSROOT.BranchType.kBaseClassNode) {
                  if (!ScanBranches(br.fBranches, master_target, chld_kind)) return false;
                  continue;
               }

               var elem = JSROOT.IO.FindBrachStreamerElement(br, handle.file);
               if (elem && (elem.fTypeName==="BASE")) {
                  // if branch is data of base class, map it to original target
                  if (br.fTotBytes && !AddBranchForReading(br, target_object, target_name, read_mode)) return false;
                  if (!ScanBranches(br.fBranches, master_target, chld_kind)) return false;
                  continue;
               }

               var subname = br.fName, chld_direct = 1;

               if (br.fName.indexOf(match_prefix)===0) {
                  subname = subname.substr(match_prefix.length);
               } else {
                  if (chld_kind>0) continue; // for defined children names prefix must be present
               }

               var p = subname.indexOf('[');
               if (p>0) subname = subname.substr(0,p);
               p = subname.indexOf('<');
               if (p>0) subname = subname.substr(0,p);

               if (chld_kind > 0) {
                  chld_direct = "$child$";
                  var pp = subname.indexOf(".");
                  if (pp>0) chld_direct = JSROOT.IO.DetectBranchMemberClass(lst, branch.fName + "." + subname.substr(0,pp+1), k) || "TObject";
               }

               if (!AddBranchForReading(br, master_target, subname, chld_direct)) return false;
            }

            return true;
         }

         if (branch._typename === "TBranchObject") {
            member = {
               name: target_name,
               typename: branch.fClassName,
               virtual: leaf.fVirtual,
               func: function(buf,obj) {
                  var clname = this.typename;
                  if (this.virtual) clname = buf.ReadFastString(buf.ntou1()+1);
                  obj[this.name] = buf.ClassStreamer({}, clname);
               }
            };
         } else

         if ((branch.fType === JSROOT.BranchType.kClonesNode) || (branch.fType === JSROOT.BranchType.kSTLNode)) {

            elem = JSROOT.IO.CreateStreamerElement(target_name, JSROOT.IO.kInt);

            if (!read_mode || ((typeof read_mode==="string") && (read_mode[0]===".")) || (read_mode===1)) {
               handle.process_arrays = false;

               member = {
                  name: target_name,
                  conttype: branch.fClonesName || "TObject",
                  reallocate: args.reallocate_objects,
                  func: function(buf,obj) {
                     var size = buf.ntoi4(), n = 0;
                     if (!obj[this.name] || this.reallocate) {
                        obj[this.name] = new Array(size);
                     } else {
                        n = obj[this.name].length;
                        obj[this.name].length = size; // reallocate array
                     }

                     while (n<size) obj[this.name][n++] = this.methods.Create(); // create new objects
                  }
               }

               if ((typeof read_mode==="string") && (read_mode[0]===".")) {
                  member.conttype = JSROOT.IO.DetectBranchMemberClass(branch.fBranches, branch.fName+read_mode);
                  if (!member.conttype) {
                     console.error('Cannot select object', read_mode, "in the branch", branch.fName);
                     return null;
                  }
               }

               member.methods = JSROOT.IO.MakeMethodsList(member.conttype);

               child_scan = (branch.fType === JSROOT.BranchType.kClonesNode) ? JSROOT.BranchType.kClonesMemberNode : JSROOT.BranchType.kSTLMemberNode;
            }
          } else

          if ((object_class = JSROOT.IO.GetBranchObjectClass(branch, handle.tree))) {

             if (read_mode === true) {
                console.warn('Object branch ' + object_class + ' can not have data to be read directly');
                return null;
             }

             handle.process_arrays = false;

             var newtgt = new Array(target_object ? (target_object.length + 1) : 1);
             for (var l=0;l<newtgt.length-1;++l) newtgt[l] = target_object[l];
             newtgt[newtgt.length-1] = { name: target_name, lst: JSROOT.IO.MakeMethodsList(object_class) };

             if (!ScanBranches(branch.fBranches, newtgt,  0)) return null;

             return item; // this kind of branch does not have baskets and not need to be read

          } else if (is_brelem && (nb_leaves === 1) && (leaf.fName === branch.fName) && (branch.fID==-1)) {

             elem = JSROOT.IO.CreateStreamerElement(target_name, branch.fClassName);

             if (elem.fType === JSROOT.IO.kAny) {

                var streamer = handle.file.GetStreamer(branch.fClassName, { val: branch.fClassVersion, checksum: branch.fCheckSum });
                if (!streamer) { elem = null; console.warn('not found streamer!'); } else
                   member = {
                         name: target_name,
                         typename: branch.fClassName,
                         streamer: streamer,
                         func: function(buf,obj) {
                            var res = { _typename: this.typename };
                            for (var n = 0; n < this.streamer.length; ++n)
                               this.streamer[n].func(buf, res);
                            obj[this.name] = res;
                         }
                   };
             }

             // elem.fType = JSROOT.IO.kAnyP;

             // only STL containers here
             // if (!elem.fSTLtype) elem = null;
          } else if (is_brelem && (nb_leaves <= 1)) {

             elem = JSROOT.IO.FindBrachStreamerElement(branch, handle.file);

             // this is basic type - can try to solve problem differently
             if (!elem && branch.fStreamerType && (branch.fStreamerType < 20))
                elem = JSROOT.IO.CreateStreamerElement(target_name, branch.fStreamerType);

          } else if (nb_leaves === 1) {
              // no special constrains for the leaf names

             elem = CreateLeafElem(leaf, target_name);

          } else if ((branch._typename === "TBranch") && (nb_leaves > 1)) {
             // branch with many elementary leaves

             var arr = new Array(nb_leaves), isok = true;
             for (var l=0;l<nb_leaves;++l) {
                arr[l] = CreateLeafElem(branch.fLeaves.arr[l]);
                arr[l] = JSROOT.IO.CreateMember(arr[l], handle.file);
                if (!arr[l]) isok = false;
             }

             if (isok)
                member = {
                   name: target_name,
                   leaves: arr,
                   func: function(buf, obj) {
                      var tgt = obj[this.name], l = 0;
                      if (!tgt) obj[this.name] = tgt = {};
                      while (l<this.leaves.length)
                         this.leaves[l++].func(buf,tgt);
                   }
               }
          }

          if (!elem && !member) {
             console.warn('Not supported branch kind', branch.fName, branch._typename);
             return null;
          }

          if (!member) {
             member = JSROOT.IO.CreateMember(elem, handle.file);

             if ((member.base !== undefined) && member.basename) {
                // when element represent base class, we need handling which differ from normal IO
                member.func = function(buf, obj) {
                   if (!obj[this.name]) obj[this.name] = { _typename: this.basename };
                   buf.ClassStreamer(obj[this.name], this.basename);
                };
             }
          }

          if (item_cnt && (typeof read_mode === "string")) {

             member.name0 = item_cnt.name;

             var snames = target_name.split(".");

             if (snames.length === 1) {
                // no point in the name - just plain array of objects
                member.get = function(arr,n) { return arr[n]; }
             } else if (read_mode === "$child$") {
                console.error('target name contains point, but suppose to be direct child', target_name);
                return null;
             } else if (snames.length === 2) {
                target_name = member.name = snames[1];
                member.name1 = snames[0];
                member.subtype1 = read_mode;
                member.methods1 = JSROOT.IO.MakeMethodsList(member.subtype1);
                member.get = function(arr,n) {
                   var obj1 = arr[n][this.name1];
                   if (!obj1) obj1 = arr[n][this.name1] = this.methods1.Create();
                   return obj1;
                }
             } else {
                // very complex task - we need to reconstruct several embedded members with their types
                // try our best - but not all data types can be reconstructed correctly
                // while classname is not enough - there can be different versions

                if (!branch.fParentName) {
                   console.error('Not possible to provide more than 2 parts in the target name', target_name);
                   return null;
                }

                target_name = member.name = snames.pop(); // use last element
                member.snames = snames; // remember all sub-names
                member.smethods = []; // and special handles to create missing objects

                var parent_class = branch.fParentName; // unfortunately, without version

                for (var k=0;k<snames.length;++k) {
                   var chld_class = JSROOT.IO.DefineMemberTypeName(handle.file, parent_class, snames[k]);
                   member.smethods[k] = JSROOT.IO.MakeMethodsList(chld_class || "AbstractClass");
                   parent_class = chld_class;
                }
                member.get = function(arr,n) {
                   var obj1 = arr[n][this.snames[0]];
                   if (!obj1) obj1 = arr[n][this.snames[0]] = this.smethods[0].Create();
                   for (var k=1;k<this.snames.length;++k) {
                      var obj2 = obj1[this.snames[k]];
                      if (!obj2) obj2 = obj1[this.snames[k]] = this.smethods[k].Create();
                      obj1 = obj2;
                   }
                   return obj1;
                }
             }

             // case when target is sub-object and need to be created before


             if (member.objs_branch_func) {
                // STL branch provides special function for the reading
                member.func = member.objs_branch_func;
             } else {
                member.func0 = member.func;

                member.func = function(buf,obj) {
                   var arr = obj[this.name0], n = 0; // objects array where reading is done
                   while(n<arr.length)
                      this.func0(buf,this.get(arr,n++)); // read all individual object with standard functions
                }
             }

          } else if (item_cnt) {

             handle.process_arrays = false;

             if ((elem.fType === JSROOT.IO.kDouble32) || (elem.fType === JSROOT.IO.kFloat16)) {
                // special handling for compressed floats

                member.stl_size = item_cnt.name;
                member.func = function(buf, obj) {
                   obj[this.name] = this.readarr(buf, obj[this.stl_size]);
                }

             } else
             if (((elem.fType === JSROOT.IO.kOffsetP+JSROOT.IO.kDouble32) || (elem.fType === JSROOT.IO.kOffsetP+JSROOT.IO.kFloat16)) && branch.fBranchCount2) {
                // special handling for variable arrays of compressed floats in branch - not tested

                member.stl_size = item_cnt.name;
                member.arr_size = item_cnt2.name;
                member.func = function(buf, obj) {
                   var sz0 = obj[this.stl_size], sz1 = obj[this.arr_size], arr = new Array(sz0);
                   for (var n=0;n<sz0;++n)
                      arr[n] = (buf.ntou1() === 1) ? this.readarr(buf, sz1[n]) : [];
                      obj[this.name] = arr;
                }

             } else
             // special handling of simple arrays
             if (((elem.fType > 0) && (elem.fType < JSROOT.IO.kOffsetL)) || (elem.fType === JSROOT.IO.kTString) ||
                 (((elem.fType > JSROOT.IO.kOffsetP) && (elem.fType < JSROOT.IO.kOffsetP + JSROOT.IO.kOffsetL)) && branch.fBranchCount2)) {

                member = {
                      name: target_name,
                      stl_size: item_cnt.name,
                      type: elem.fType,
                      func: function(buf, obj) {
                         obj[this.name] = buf.ReadFastArray(obj[this.stl_size], this.type);
                      }
                };

                if (branch.fBranchCount2) {
                   member.type -= JSROOT.IO.kOffsetP;
                   member.arr_size = item_cnt2.name;
                   member.func = function(buf, obj) {
                      var sz0 = obj[this.stl_size], sz1 = obj[this.arr_size], arr = new Array(sz0);
                      for (var n=0;n<sz0;++n)
                         arr[n] = (buf.ntou1() === 1) ? buf.ReadFastArray(sz1[n], this.type) : [];
                         obj[this.name] = arr;
                   }
                }

             } else
             if ((elem.fType > JSROOT.IO.kOffsetP) && (elem.fType < JSROOT.IO.kOffsetP + JSROOT.IO.kOffsetL) && member.cntname) {

                member.cntname = item_cnt.name;
             } else
             if (elem.fType == JSROOT.IO.kStreamer) {
                // with streamers one need to extend existing array

                if (item_cnt2)
                   throw new Error('Second branch counter not supported yet with JSROOT.IO.kStreamer');

                // function provided by normal I/O
                member.func = member.branch_func;
                member.stl_size = item_cnt.name;
             } else
             if ((elem.fType === JSROOT.IO.kStreamLoop) || (elem.fType === JSROOT.IO.kOffsetL+JSROOT.IO.kStreamLoop)) {
                if (item_cnt2) {
                   // special solution for kStreamLoop
                   member.stl_size = item_cnt.name;
                   member.cntname = item_cnt2.name;
                   member.func = member.branch_func; // this is special function, provided by base I/O
                } else {
                   member.cntname = item_cnt.name;
                }
             } else  {

                member.name = "$stl_member";

                var loop_size_name;

                if (item_cnt2) {
                   if (member.cntname) {
                      loop_size_name = item_cnt2.name;
                      member.cntname = "$loop_size";
                   } else {
                      throw new Error('Second branch counter not used - very BAD');
                   }
                }

                var stlmember = {
                      name: target_name,
                      stl_size: item_cnt.name,
                      loop_size: loop_size_name,
                      member0: member,
                      func: function(buf, obj) {
                         var cnt = obj[this.stl_size], arr = new Array(cnt), n = 0;
                         for (var n=0;n<cnt;++n) {
                            if (this.loop_size) obj.$loop_size = obj[this.loop_size][n];
                            this.member0.func(buf, obj);
                            arr[n] = obj.$stl_member;
                         }
                         delete obj.$stl_member;
                         delete obj.$loop_size;
                         obj[this.name] = arr;
                      }
                };

                member = stlmember;
             }
          }

          // set name used to store result
          member.name = target_name;

         item.member = member; // member for reading
         if (elem) item.type = elem.fType;
         item.index = handle.arr.length; // index in the global list of branches

         if (item_cnt) {
            item.counters = [ item_cnt.index ];
            item_cnt.ascounter.push(item.index);

            if (item_cnt2) {
               item.counters.push(item_cnt2.index);
               item_cnt2.ascounter.push(item.index);
            }
         }

         handle.arr.push(item);

         // now one should add all other child branches
         if (child_scan)
            if (!ScanBranches(branch.fBranches, target_object, child_scan)) return null;

         return item;
      }

      // main loop to add all branches from selector for reading
      for (var nn=0; nn<selector.branches.length; ++nn) {

         var item = AddBranchForReading(selector.branches[nn], undefined, selector.names[nn], selector.directs[nn]);

         if (!item) {
            selector.Terminate(false);
            return false;
         }
      }

      // check if simple reading can be performed and there are direct data in branch

      for (var h=1; (h < handle.arr.length) && handle.simple_read; ++h) {

         var item = handle.arr[h], item0 = handle.arr[0];

         if ((item.numentries !== item0.numentries) || (item.numbaskets !== item0.numbaskets)) handle.simple_read = false;
         for (var n=0;n<item.numbaskets;++n)
            if (item.GetBasketEntry(n) !== item0.GetBasketEntry(n)) handle.simple_read = false;
      }

      // now calculate entries range

      handle.firstentry = handle.lastentry = 0;
      for (var nn = 0; nn < handle.arr.length; ++nn) {
         var branch = handle.arr[nn].branch, e1 = branch.fFirstEntry;
         if (e1 === undefined) e1 = (branch.fBasketBytes[0] ? branch.fBasketEntry[0] : 0);
         handle.firstentry = Math.max(handle.firstentry, e1);
         handle.lastentry = (nn===0) ? (e1 + branch.fEntries) : Math.min(handle.lastentry, e1 + branch.fEntries);
      }

      if (handle.firstentry >= handle.lastentry) {
         console.warn('No any common events for selected branches');
         selector.Terminate(false);
         return false;
      }

      handle.process_min = handle.firstentry;
      handle.process_max = handle.lastentry;

      if (!isNaN(args.firstentry) && (args.firstentry>handle.firstentry) && (args.firstentry < handle.lastentry))
         handle.process_min = args.firstentry;

      handle.current_entry = handle.staged_now = handle.process_min;

      if (!isNaN(args.numentries) && (args.numentries>0)) {
         var max = handle.process_min + args.numentries;
         if (max<handle.process_max) handle.process_max = max;
      }

      if ((typeof selector.ProcessArrays === 'function') && handle.simple_read) {
         // this is indication that selector can process arrays of values
         // only strictly-matched tree structure can be used for that

         for (var k=0;k<handle.arr.length;++k) {
            var elem = handle.arr[k];
            if ((elem.type<=0) || (elem.type >= JSROOT.IO.kOffsetL) || (elem.type === JSROOT.IO.kCharStar)) handle.process_arrays = false;
         }

         if (handle.process_arrays) {
            // create other members for fast processing

            selector.tgtarr = {}; // object with arrays

            for(var nn=0;nn<handle.arr.length;++nn) {
               var item = handle.arr[nn],
                   elem = JSROOT.IO.CreateStreamerElement(item.name, item.type);

               elem.fType = item.type + JSROOT.IO.kOffsetL;
               elem.fArrayLength = 10;
               elem.fArrayDim = 1;
               elem.fMaxIndex[0] = 10; // 10 if artificial number, will be replaced during reading

               item.arrmember = JSROOT.IO.CreateMember(elem, handle.file);
            }
         }
      } else {
         handle.process_arrays = false;
      }

      function ReadBaskets(bitems, baskets_call_back) {
         // read basket with tree data, selecting different files

         var places = [], filename = "";

         function ExtractPlaces() {
            // extract places to read and define file name

            places = []; filename = "";

            for (var n=0;n<bitems.length;++n) {
               if (bitems[n].done) continue;

               var branch = bitems[n].branch;

               if (places.length===0)
                  filename = branch.fFileName;
               else
                  if (filename !== branch.fFileName) continue;

               bitems[n].selected = true; // mark which item was selected for reading

               places.push(branch.fBasketSeek[bitems[n].basket], branch.fBasketBytes[bitems[n].basket]);
            }

            return places.length > 0;
         }

         function ReadProgress(value) {

            if ((handle.staged_prev === handle.staged_now) ||
               (handle.process_max <= handle.process_min)) return;

            var tm = new Date().getTime();

            if (tm - handle.progress_showtm < 500) return; // no need to show very often

            handle.progress_showtm = tm;

            var portion = (handle.staged_prev + value * (handle.staged_now - handle.staged_prev)) /
                          (handle.process_max - handle.process_min);

            handle.selector.ShowProgress(portion);
         }

         function ProcessBlobs(blobs) {
            if (!blobs || ((places.length>2) && (blobs.length*2 !== places.length)))
               return JSROOT.CallBack(baskets_call_back, null);

            var baskets = [], n = 0;

            for (var k=0;k<bitems.length;++k) {
               if (!bitems[k].selected) continue;

               bitems[k].selected = false;
               bitems[k].done = true;

               var blob = (places.length > 2) ? blobs[n++] : blobs,
                   buf = JSROOT.CreateTBuffer(blob, 0, handle.file),
                   basket = buf.ClassStreamer({}, "TBasket");

               if (basket.fNbytes !== bitems[k].branch.fBasketBytes[bitems[k].basket])
                  console.error('mismatch in read basket sizes', bitems[k].branch.fBasketBytes[bitems[k].basket]);

               // items[k].obj = basket; // keep basket object itself if necessary

               bitems[k].bskt_obj = basket; // only number of entries in the basket are relevant for the moment

               if (basket.fKeylen + basket.fObjlen === basket.fNbytes) {
                  // use data from original blob
                  buf.raw_shift = 0;
               } else {
                  // unpack data and create new blob
                  var objblob = JSROOT.R__unzip(blob, basket.fObjlen, false, buf.o);

                  if (objblob) {
                     buf = JSROOT.CreateTBuffer(objblob, 0, handle.file);
                     buf.raw_shift = basket.fKeylen;
                     buf.fTagOffset = basket.fKeylen;
                  } else {
                     throw new Error('FAIL TO UNPACK');
                  }
               }

               bitems[k].raw = buf; // here already unpacked buffer

               if (bitems[k].branch.fEntryOffsetLen > 0)
                  buf.ReadBasketEntryOffset(basket, buf.raw_shift);
            }

            if (ExtractPlaces())
               handle.file.ReadBuffer(places, ProcessBlobs, filename, ReadProgress);
            else
               JSROOT.CallBack(baskets_call_back, bitems);
         }

         // extract places where to read
         if (ExtractPlaces())
            handle.file.ReadBuffer(places, ProcessBlobs, filename, ReadProgress);
         else
            JSROOT.CallBack(baskets_call_back, null);
      }

      function ReadNextBaskets() {

         var totalsz = 0, bitems = [], isany = true, is_direct = false, min_staged = handle.process_max;

         while ((totalsz < 1e6) && isany) {
            isany = false;
            // very important, loop over branches in reverse order
            // let check counter branch after reading of normal branch is prepared
            for (var n=handle.arr.length-1; n>=0; --n) {
               var elem = handle.arr[n];

               while (elem.staged_basket < elem.numbaskets) {

                  var k = elem.staged_basket++;

                  // no need to read more baskets, process_max is not included
                  if (elem.GetBasketEntry(k) >= handle.process_max) break;

                  // check which baskets need to be read
                  if (elem.first_readentry < 0) {
                     var lmt = elem.GetBasketEntry(k+1),
                         not_needed = (lmt <= handle.process_min);

                     //for (var d=0;d<elem.ascounter.length;++d) {
                     //   var dep = handle.arr[elem.ascounter[d]]; // dependent element
                     //   if (dep.first_readentry < lmt) not_needed = false; // check that counter provide required data
                     //}

                     if (not_needed) continue; // if that basket not required, check next

                     elem.curr_basket = k; // basket where reading will start

                     elem.first_readentry = elem.GetBasketEntry(k); // remember which entry will be read first
                  }

                  // check if basket already loaded in the branch

                  var bitem = {
                        id: n, // to find which element we are reading
                        branch: elem.branch,
                        basket: k,
                        raw: null // here should be result
                     };

                  var bskt = elem.branch.fBaskets.arr[k];
                  if (bskt) {
                     bitem.raw = bskt.fBufferRef;
                     if (bitem.raw)
                        bitem.raw.locate(0); // reset pointer - same branch may be read several times
                     else
                        bitem.raw = JSROOT.CreateTBuffer(null, 0, handle.file); // create dummy buffer - basket has no data
                     bitem.raw.raw_shift = bskt.fKeylen;

                     if (bskt.fBufferRef && (elem.branch.fEntryOffsetLen > 0))
                        bitem.raw.ReadBasketEntryOffset(bskt, bitem.raw.raw_shift);

                     bitem.bskt_obj = bskt;
                     is_direct = true;
                     elem.baskets[k] = bitem;
                  } else {
                     bitems.push(bitem);
                     totalsz += elem.branch.fBasketBytes[k];
                     isany = true;
                  }

                  elem.staged_entry = elem.GetBasketEntry(k+1);

                  min_staged = Math.min(min_staged, elem.staged_entry);

                  break;
               }
            }
         }

         if ((totalsz === 0) && !is_direct)
            return handle.selector.Terminate(true);

         handle.staged_prev = handle.staged_now;
         handle.staged_now = min_staged;

         var portion = 0;
         if (handle.process_max > handle.process_min)
            portion = (handle.staged_prev - handle.process_min)/ (handle.process_max - handle.process_min);

         handle.selector.ShowProgress(portion);

         handle.progress_showtm = new Date().getTime();

         if (totalsz > 0) return ReadBaskets(bitems, ProcessBaskets);

         if (is_direct) return ProcessBaskets([]); // directly process baskets

         throw new Error("No any data is requested - never come here");
      }

      function ProcessBaskets(bitems) {
         // this is call-back when next baskets are read

         if ((handle.selector.break_execution !== 0) || (bitems===null))
            return handle.selector.Terminate(false);

         // redistribute read baskets over branches
         for(var n=0;n<bitems.length;++n)
            handle.arr[bitems[n].id].baskets[bitems[n].basket] = bitems[n];

         // now process baskets

         var isanyprocessed = false;

         while(true) {

            var loopentries = 100000000, min_curr = handle.process_max, n, elem;

            // first loop used to check if all required data exists
            for (n=0;n<handle.arr.length;++n) {

               elem = handle.arr[n];

               if (!elem.raw || !elem.basket || (elem.first_entry + elem.basket.fNevBuf <= handle.current_entry)) {
                  delete elem.raw;
                  delete elem.basket;

                  if ((elem.curr_basket >= elem.numbaskets)) {
                     if (n==0) return handle.selector.Terminate(true);
                     continue; // ignore non-master branch
                  }

                  // this is single response from the tree, includes branch, bakset number, raw data
                  var bitem = elem.baskets[elem.curr_basket];

                  // basket not read
                  if (!bitem) {
                     // no data, but no any event processed - problem
                     if (!isanyprocessed) {
                        console.warn('no data?', elem.branch.fName, elem.curr_basket);
                        return handle.selector.Terminate(false);
                     }

                     // try to read next portion of tree data
                     return ReadNextBaskets();
                  }

                  elem.raw = bitem.raw;
                  elem.basket = bitem.bskt_obj;
                  // elem.nev = bitem.fNevBuf; // number of entries in raw buffer
                  elem.first_entry = elem.GetBasketEntry(bitem.basket);

                  bitem.raw = null; // remove reference on raw buffer
                  bitem.branch = null; // remove reference on the branch
                  bitem.bskt_obj = null; // remove reference on the branch
                  elem.baskets[elem.curr_basket++] = undefined; // remove from array
               }

                // define how much entries can be processed before next raw buffer will be finished
               loopentries = Math.min(loopentries, elem.first_entry + elem.basket.fNevBuf - handle.current_entry);
            }

            // second loop extracts all required data

            // do not read too much
            if (handle.current_entry + loopentries > handle.process_max)
               loopentries = handle.process_max - handle.current_entry;

            if (handle.process_arrays && (loopentries>1)) {
               // special case - read all data from baskets as arrays

               for (n=0;n<handle.arr.length;++n) {
                  elem = handle.arr[n];

                  elem.GetEntry(handle.current_entry);

                  elem.arrmember.arrlength = loopentries;
                  elem.arrmember.func(elem.raw, handle.selector.tgtarr);

                  elem.raw = null;
               }

               handle.selector.ProcessArrays(handle.current_entry);

               handle.current_entry += loopentries;

               isanyprocessed = true;
            } else

            // main processing loop
            while(loopentries--) {

               for (n=0;n<handle.arr.length;++n) {
                  elem = handle.arr[n];

                  // locate buffer offset at proper place
                  elem.GetEntry(handle.current_entry);

                  elem.member.func(elem.raw, elem.GetTarget(handle.selector.tgtobj));
               }

               handle.selector.Process(handle.current_entry);

               handle.current_entry++;

               isanyprocessed = true;
            }

            if (handle.current_entry >= handle.process_max)
                return handle.selector.Terminate(true);
         }
      }

      // call begin before first entry is read
      handle.selector.Begin(this);

      ReadNextBaskets();

      return true; // indicate that reading of tree will be performed
   }

   /** @summary Search branch with specified name
    * @desc if complex enabled, search branch and rest part
    * @private */
   JSROOT.TreeMethods.FindBranch = function(name, complex, lst) {

      var top_search = false, search = name, res = null;

      if (lst===undefined) {
         top_search = true;
         lst = this.fBranches;
         var pos = search.indexOf("[");
         if (pos>0) search = search.substr(0,pos);
      }

      if (!lst || (lst.arr.length===0)) return null;

      for (var n=0;n<lst.arr.length;++n) {
         var brname = lst.arr[n].fName;
         if (brname[brname.length-1] == "]")
            brname = brname.substr(0, brname.indexOf("["));

         // special case when branch name includes STL map name
         if ((search.indexOf(brname)!==0) && (brname.indexOf("<")>0)) {
            var p1 = brname.indexOf("<"), p2 = brname.lastIndexOf(">");
            brname = brname.substr(0, p1) + brname.substr(p2+1);
         }

         if (brname === search) { res = { branch: lst.arr[n], rest:"" }; break; }

         if (search.indexOf(brname)!==0) continue;

         // this is a case when branch name is in the begin of the search string

         // check where point is
         var pnt = brname.length;
         if (brname[pnt-1] === '.') pnt--;
         if (search[pnt] !== '.') continue;

         res = this.FindBranch(search, complex, lst.arr[n].fBranches);
         if (!res) res = this.FindBranch(search.substr(pnt+1), complex, lst.arr[n].fBranches);

         if (!res) res = { branch: lst.arr[n], rest: search.substr(pnt) };

         break;
      }

      if (!top_search || !res) return res;

      if (name.length > search.length) res.rest += name.substr(search.length);

      if (!complex && (res.rest.length>0)) return null;

      return complex ? res : res.branch;
   }

   /** @summary  this is JSROOT implementation of TTree::Draw
     * @disc in callback returns histogram and draw options
     * @param {object} args - different setting or simply draw expression
     * @param {string} args.expr - draw expression
     * @param {string} [args.cut=undefined]   - cut expression (also can be part of 'expr' after '::')
     * @param {string} [args.drawopt=undefined] - draw options for result histogram
     * @param {number} [args.firstentry=0] - first entry to process
     * @param {number} [args.numentries=undefined] - number of entries to process, all by default
     * @param {object} [args.branch=undefined] - TBranch object from TTree itself for the direct drawing
     * @param {function} result_callback - function called when draw is completed
     */
   JSROOT.TreeMethods.Draw = function(args, result_callback) {

      if (typeof args === 'string') args = { expr: args };

      if (!args.expr) args.expr = "";

      // special debugging code
      if (args.expr === "testio")
         return this.IOTest(args, result_callback);

      var selector = new TDrawSelector(result_callback);

      if (args.branch) {
         if (!selector.DrawOnlyBranch(this, args.branch, args.expr, args)) selector = null;
      } else {
         if (!selector.ParseDrawExpression(this, args)) selector = null;
      }

      if (!selector)
         return JSROOT.CallBack(result_callback, null);

      return this.Process(selector, args);
   }

   JSROOT.TreeMethods.IOTest = function(args, result_callback) {
      // generic I/O test for all branches in the tree

      if (!args.names && !args.bracnhes) {

         args.branches = [];
         args.names = [];
         args.nchilds = [];
         args.nbr = 0;

         function CollectBranches(obj, prntname) {
            if (!obj || !obj.fBranches) return 0;

            var cnt = 0;

            for (var n=0;n<obj.fBranches.arr.length;++n) {
               var br = obj.fBranches.arr[n],
               name = (prntname ? prntname + "/" : "") + br.fName;
               args.branches.push(br);
               args.names.push(name);
               args.nchilds.push(0);
               var pos = args.nchilds.length-1;
               cnt += br.fLeaves ? br.fLeaves.arr.length : 0;
               var nchld = CollectBranches(br, name);

               cnt += nchld;
               args.nchilds[pos] = nchld;

            }
            return cnt;
         }

         var numleaves = CollectBranches(this);

         args.names.push("Total are " + args.branches.length + " branches with " + numleaves + " leaves");
      }

      args.lasttm = new Date().getTime();
      args.lastnbr = args.nbr;

      var tree = this;

      function TestNextBranch() {

         var selector = new TSelector;

         selector.AddBranch(args.branches[args.nbr], "br0");

         selector.Process = function() {
            if (this.tgtobj.br0 === undefined)
               this.fail = true;
         }

         selector.Terminate = function(res) {
            if (typeof res !== 'string')
               res = (!res || this.fails) ? "FAIL" : "ok";

            args.names[args.nbr] = res + " " + args.names[args.nbr];
            args.nbr++;

            if (args.nbr >= args.branches.length) {
               JSROOT.progress();
               return JSROOT.CallBack(result_callback, args.names, "inspect");
            }

            var now = new Date().getTime();

            if ((now - args.lasttm > 5000) || (args.nbr - args.lastnbr > 50))
               setTimeout(tree.IOTest.bind(tree,args,result_callback), 100); // use timeout to avoid deep recursion
            else
               TestNextBranch();
         }

         JSROOT.progress("br " + args.nbr + "/" + args.branches.length + " " + args.names[args.nbr]);

         var br = args.branches[args.nbr],
             object_class = JSROOT.IO.GetBranchObjectClass(br, tree),
             num = br.fEntries,
             skip_branch = (!br.fLeaves || (br.fLeaves.arr.length === 0));

         if (object_class) skip_branch = (args.nchilds[args.nbr]>100);

         // skip_branch = args.nchilds[args.nbr]>1;

         if (skip_branch || (num<=0)) {
            // ignore empty branches or objects with too-many subbranch
            // if (object_class) console.log('Ignore branch', br.fName, 'class', object_class, 'with', args.nchilds[args.nbr],'subbranches');
            selector.Terminate("ignore");
         } else {

            var drawargs = { numentries: 10 },
                first = br.fFirstEntry || 0,
                last = br.fEntryNumber || (first+num);

            if (num < drawargs.numentries) {
               drawargs.numentries = num;
            } else {
               // select randomly first entry to test I/O
               drawargs.firstentry = first + Math.round((last-first-drawargs.numentries)*Math.random());
            }

            // keep console output for debug purposes
            console.log('test branch', br.fName, 'first', (drawargs.firstentry || 0), "num", drawargs.numentries);

            tree.Process(selector, drawargs);
         }
      }

      TestNextBranch();
   }


   JSROOT.TSelector = TSelector;
   JSROOT.TDrawVariable = TDrawVariable;
   JSROOT.TDrawSelector = TDrawSelector;

   return JSROOT;

}));
