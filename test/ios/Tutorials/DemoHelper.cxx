#include <stdexcept>
#include <memory>

#include "ExclusionGraphDemo.h"
#include "PolarGraphDemo.h"
#include "HsimpleDemo.h"
#include "SurfaceDemo.h"
#include "H2PolyDemo.h"
#include "DemoHelper.h"
#include "LegoDemo.h"

namespace ROOT {
namespace iOS {
namespace Demos {

bool CreateTutorials(DemoBase **demos, const char *filePath)
{
   try {
      std::auto_ptr<HsimpleDemo> demo0(new HsimpleDemo);
      std::auto_ptr<SurfaceDemo> demo1(new SurfaceDemo);
      std::auto_ptr<PolarGraphDemo> demo2(new PolarGraphDemo);
      std::auto_ptr<LegoDemo> demo3(new LegoDemo);
      std::auto_ptr<ExclusionGraphDemo> demo4(new ExclusionGraphDemo);
      std::auto_ptr<H2PolyDemo> demo5(new H2PolyDemo(filePath));
      
      demos[0] = demo0.release();
      demos[1] = demo1.release();
      demos[2] = demo2.release();
      demos[3] = demo3.release();
      demos[4] = demo4.release();
      demos[5] = demo5.release();
   } catch (const std::exception &e) {
      return false;
   }

   return false;//true;
}

}
}
}
