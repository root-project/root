(function(){

   if (typeof JSROOT != "object") {
      var e1 = new Error("httptextlog.js requires JSROOT to be already loaded");
      e1.source = "go4.js";
      throw e1;
   }
   
   console.log("here 1");
   
   MakeMsgListRequest = function(hitem, item) {
      var arg = "&max=1000";
      if ('last-id' in item) arg+= "&id="+item['last-id'];
      return 'exe.json.gz?method=Select' + arg;      
   }
   
   AfterMsgListRequest = function(hitem, item, obj) {
      if (item==null) return;
      
      if (obj==null) {
         delete item['last-id'];
         return;
      } 
      // ignore all other classes   
      if (obj['_typename'] != 'TList') return;
       
      // change class name - it is only important for drawing 
      obj['_typename'] = "TMsgList";
      
      if (obj.arr.length>0) {
         item['last-id'] = obj.arr[0].fString;

         // add clear function for item
         if (!('clear' in item)) 
            item['clear'] = function() { delete this['last-id']; }
      }
   }
   
   
   TMsgListPainter = function(lst) {
      JSROOT.TBasePainter.call(this);
      
      this.lst = lst;
         
      return this;
   }

   TMsgListPainter.prototype = Object.create( JSROOT.TBasePainter.prototype );

   TMsgListPainter.prototype.RedrawObject = function(obj) {
      this.lst = obj;
      this.Draw();
      return true;
   }

   TMsgListPainter.prototype.Draw = function() {
      
      if (this.lst == null) return;
      
      var frame = d3.select("#" + this.divid);
      
      var main = frame.select("div");
      if (main.empty()) 
         main = frame.append("div")
                     .style('max-width','100%')
                     .style('max-height','100%')
                     .style('overflow','auto');
      
      var old = main.selectAll("pre");
      var newsize = old.size() + this.lst.arr.length - 1; 

      // in the browser keep maximum 1000 entries
      if (newsize > 1000) 
         old.select(function(d,i) { return i < newsize - 1000 ? this : null; }).remove();
      
      for (var i = this.lst.arr.length-1;i>0;i--)
         main.append("pre").html(this.lst.arr[i].fString);
      
      // (re) set painter to first child element
      this.SetDivId(this.divid);
   }
   
   var DrawTMsgList = function(divid, lst, opt) {
      var painter = new TMsgListPainter(lst);
      painter.SetDivId(divid);
      painter.Draw();
      return painter.DrawingReady();
   }
   
   JSROOT.addDrawFunc("TMsgList", DrawTMsgList, "");
   
   console.log("here 10");


})();
