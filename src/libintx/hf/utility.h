#ifndef LIBINTX_HF_UTILITY_H
#define LIBINTX_HF_UTILITY_H

#include <map>

namespace libintx::hf {

  auto make_multimap(auto &&f, const auto &range, size_t Max = 0) {
    using KV = decltype(f(0,range[0]));
    using K = typename KV::first_type;
    using V = typename KV::second_type;
    std::map< K,std::vector<V> > map;
    for (size_t i = 0; i < range.size(); ++i) {
      auto [k,v] = f(i,range[i]);
      map[k].push_back(v);
    }
    std::vector< std::pair<const K,std::vector<V> > > vmap;
    vmap.reserve(map.size());
    for (auto &[k,v] : map) {
      size_t batches = v.size();
      if (Max) batches = (v.size() + Max - 1)/Max;
      size_t Batch = (v.size() + batches - 1)/batches;
      for (size_t i = 0; i < v.size(); i += Batch) {
        auto it = v.begin()+i;
        size_t N = std::min(Batch,v.size()-i);
        vmap.push_back({k,std::vector(it,it+N)});
      }
    }
    return vmap;
  }

}

#endif /* LIBINTX_HF_UTILITY_H */
