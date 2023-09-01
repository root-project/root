#ifndef ROOT7_RNTupleUtil_Test_CustomStructUtil
#define ROOT7_RNTupleUtil_Test_CustomStructUtil

#include <string>
#include <vector>

struct BaseUtil {
   int base;
};

struct CustomStructUtil : BaseUtil {
   float a = 0.0;
   std::vector<float> v1;
   std::vector<std::vector<float>> nnlo;
   std::string s;
};

struct HitUtil : BaseUtil {
   float x;
   float y;

   bool operator==(const HitUtil &other) const { return base == other.base && x == other.x && y == other.y; }
};

struct TrackUtil : BaseUtil {
   float E;
   std::vector<HitUtil> hits;

   bool operator==(const TrackUtil &other) const { return base == other.base && E == other.E && hits == other.hits; }
};

struct ComplexStructUtil : BaseUtil {
   float pt;
   std::vector<TrackUtil> tracks;

   void Init1()
   {
      tracks.clear();
      base = 1;
      pt = 2.0;
      HitUtil hit;
      hit.base = 3;
      hit.x = 4.0;
      hit.y = 5.0;
      TrackUtil track;
      track.base = 6;
      track.E = 7.0;
      track.hits.emplace_back(hit);
      tracks.emplace_back(track);
   }

   void Init2()
   {
      tracks.clear();
      base = 100;
      pt = 101.0;
   }

   void Init3()
   {
      tracks.clear();
      base = 1000;
      pt = 1001.0;
      HitUtil hit1;
      hit1.base = 1002;
      hit1.x = 1003.0;
      hit1.y = 1004.0;
      HitUtil hit2;
      hit2.base = 1005;
      hit2.x = 1006.0;
      hit2.y = 1007.0;
      TrackUtil track1;
      track1.base = 1008;
      track1.E = 1009.0;
      tracks.emplace_back(track1);
      TrackUtil track2;
      track2.base = 1010;
      track2.E = 1011.0;
      track2.hits.emplace_back(hit1);
      track2.hits.emplace_back(hit2);
      tracks.emplace_back(track2);
   }

   bool operator==(const ComplexStructUtil &other) const
   {
      return base == other.base && pt == other.pt && tracks == other.tracks;
   }
};

#endif
