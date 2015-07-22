(function(){

   if (typeof JSROOT != "object") {
      var e1 = new Error("httptextlog.js requires JSROOT to be already loaded");
      e1.source = "httptextlog.js";
      throw e1;
   }
   
   function MakeMsgListRequest(hitem, item) {
      // this function produces url for http request
      // here one provides id of last string received with previous request 
      
      var arg = "&max=1000";
      if ('last-id' in item) arg+= "&id="+item['last-id'];
      return 'exe.json.gz?method=Select' + arg;      
   }
   
   function AfterMsgListRequest(hitem, item, obj) {
      // after data received, one replaces typename for produced object  
      
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
   

   function DrawMsgList(divid, lst, opt) {
      
      var painter = new JSROOT.TBasePainter();
      painter.SetDivId(divid);
      
      painter.Draw = function(lst) {
         if (lst == null) return;
         
         var frame = d3.select("#" + this.divid);
         
         var main = frame.select("div");
         if (main.empty()) 
            main = frame.append("div")
                        .style('max-width','100%')
                        .style('max-height','100%')
                        .style('overflow','auto');
         
         var old = main.selectAll("pre");
         var newsize = old.size() + lst.arr.length - 1; 

         // in the browser keep maximum 1000 entries
         if (newsize > 1000) 
            old.select(function(d,i) { return i < newsize - 1000 ? this : null; }).remove();
         
         for (var i=lst.arr.length-1;i>0;i--)
            main.append("pre").html(lst.arr[i].fString);
         
         // (re) set painter to first child element
         this.SetDivId(this.divid);
      }

      painter.RedrawObject = function(obj) {
         this.Draw(obj);
         return true;
      }
      
      painter.Draw(lst);
      return painter.DrawingReady();
   }
   
   JSROOT.addDrawFunc({name:"TMsgList", icon:"img_text", make_request:MakeMsgListRequest, after_request:AfterMsgListRequest, func:DrawMsgList, opt:"list"});

})();
