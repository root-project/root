/// \file
/// \ingroup tutorial_http
/// JavaScript code for drawing TMsgList class from httptextlog.C macro
///
/// \macro_code
///
/// \author  Sergey Linev

import { BasePainter, addDrawFunc } from 'jsroot';

/** @summary this function produces url for http request
  * @desc here one provides id of last string received with previous request */
function makeMsgListRequest(hitem, item) {
   let arg = 'max=1000';
   if ('last-id' in item)
      arg += `&id=${item['last-id']}`;
   return `exe.json.gz?method=Select&${arg}`;
}

/** @summary after data received, one replaces typename for produced object */
function afterMsgListRequest(hitem, item, obj) {
   if (!item)
      return;

   if (!obj) {
      delete item['last-id'];
      return;
   }
   // ignore all other classes
   if (obj._typename != 'TList') return;

   // change class name - it is only important for drawing
   obj._typename = 'TMsgList';

   if (obj.arr.length > 0) {
      item['last-id'] = obj.arr[0].fString;

      // add clear function for item
      if (!item.clear)
         item.clear = function() { delete this['last-id']; }
   }
}

class TMsgListPainter extends BasePainter {

   /** @summary draw list entries */
   drawList(lst) {
      if (!lst) return;

      let frame = this.selectDom(),
          main = frame.select('div');

      if (main.empty()) {
         main = frame.append('div')
                     .style('max-width', '100%')
                     .style('max-height', '100%')
                     .style('overflow', 'auto');
         // (re) set painter to first child element
         this.setTopPainter();
      }

      let old = main.selectAll('pre'),
          newsize = old.size() + lst.arr.length - 1;

      // in the browser keep maximum 1000 entries
      if (newsize > 1000)
         old.select(function(d, i) { return i < newsize - 1000 ? this : null; }).remove();

      for (let i = lst.arr.length - 1; i > 0; i--)
         main.append('pre').style('margin', '2px').html(lst.arr[i].fString);
   }

   /** @summary redraw list */
   redrawObject(obj) {
      this.drawList(obj);
      return true;
   }

   /** @summary Draw TMsgList object */
   static async draw(dom, obj /*, opt */) {
      const painter = new TMsgListPainter(dom);
      painter.drawList(obj);
      return painter;
   }

} // class TMsgListPainter

// register draw function to JSROOT
addDrawFunc({
   name: 'TMsgList',
   icon: 'img_text',
   make_request: makeMsgListRequest,
   after_request: afterMsgListRequest,
   func: TMsgListPainter.draw,
   opt: 'list'
});
