#include "libintx/hf/guess.h"
#include "libintx/hf/engine.h"
#include "libintx/hf/utility.h"
#include "libintx/ao/screening.h"
#include "libintx/ao/md/engine.h"
#include "libintx/shell.h"
#include "libintx/tensor.h"
#include "libintx/utility.h"

namespace libintx::hf {

  using Key = std::tuple<int,int>;

  auto nbf(const Key &k) {
    return npure(std::get<0>(k));
  }

  template<typename V>
  auto nbf(const std::pair<const Key,V> &kv) {
    return npure(std::get<0>(kv.first));
  }

  template<typename V>
  auto& indices(const std::pair<const Key,V> &kv) {
    return kv.second;
  }

  template<typename ... Args>
  auto& indices(const std::pair<const Key, std::tuple<Args...> > &kv) {
    return std::get<0>(kv.second);
  }

  template<typename V>
  auto& norms(const std::pair<const Key,V> &kv) {
    return std::get<1>(kv.second);
  }

  auto compute_norms(
    const auto &Norm,
    const Basis<Gaussian> &basis1,
    const std::vector<Index1> &idx1,
    const Basis<Gaussian> &basis2)
  {
    std::vector< std::vector<float> > norms(idx1.size());
    for (size_t i = 0; i < idx1.size(); ++i) {
      auto &norms_i = norms[i];
      norms_i.resize(basis2.size());
      for (size_t j = 0; j < basis2.size(); ++j) {
        float q = ao::screening::compute(Norm, basis1[idx1[i]], basis2[j]);
        norms_i[j] = std::sqrt(q);
      }
    }
    return norms;
  }

  static void fock2(
    const std::shared_ptr< Basis<Gaussian> >& basis,
    const std::shared_ptr< Basis<Gaussian> >& minbs,
    const std::vector<double> D,
    std::function<void(Index2, const double*)> F,
    const std::vector<Index1>& Is,
    const std::vector<Index1>& Js,
    libintx::num_threads num_threads,
    ao::screening::Norm<double> norm,
    double precision)
  {

    //printf("hf::guess::fock2: precision=%e\n", precision);

    bool symmetric = (Is == Js);

    auto key = [](auto &shell) {
      return Key{ shell.L, shell.K };
    };

    std::vector< std::tuple<float,Index1> > x_norms(minbs->size());
    std::vector<float> d_norms(minbs->size());
    for (size_t i = 0; i < minbs->size(); ++i) {
      auto &x = (*minbs)[i];
      float q = ao::screening::compute(norm, x, x);
      x_norms[i] = { std::sqrt(q), i };
      d_norms[i] = norm(nbf(x), D.data() + minbs->range(i).begin(), 1);
    }
    std::sort(x_norms.begin(), x_norms.end());

    std::map<Key, std::vector< std::tuple<Index1,float> > > Xs;
    for (auto &[norm,idx] : x_norms) {
      auto &shell = (*minbs)[idx];
      Xs[key(shell)].push_back({idx,norm});
    }

    auto As = make_multimap(
      [&](Index1 idx0, Index1 idx) {
        auto &shell = (*basis)[idx];
        return std::pair(key(shell), idx0); // NB idx0 NOT idx
      },
      Is,
      128
    );
    auto ax_norms = compute_norms(norm, *basis, Is, *minbs);

    auto Bs = make_multimap(
      [&](Index1 idx0, Index1 idx) {
        auto &shell = (*basis)[idx];
        return std::pair(key(shell), idx0); // NB idx0 NOT idx
      },
      Js,
      128
    );
    auto bx_norms = compute_norms(norm, *basis, Js, *minbs);

#pragma omp parallel num_threads(num_threads.value)
    {

      md::IntegralEngine<4> exchange({basis, minbs, basis, minbs});
      md::IntegralEngine<4> coulomb({basis, basis, minbs, minbs});
      coulomb.precision = precision;
      exchange.precision = precision;

      std::vector<double> F_buffer, K_buffer;
      struct {
        std::vector< std::tuple<float,Index2> > pairs;
        std::vector<Index2> indices;
        std::vector<double> norms;
      } ijs;

#pragma omp for collapse(2) schedule(dynamic,1)
      for (auto& [A,is] : As) {
        for (auto& [B,js] : Bs) {

          ijs.pairs.clear();
          ijs.indices.clear();
          ijs.norms.clear();

          ijs.pairs.reserve(is.size()*js.size());
          ijs.indices.reserve(is.size()*js.size());
          ijs.norms.reserve(is.size()*js.size());

          for (auto &j : js) {
            for (auto &i : is) {
              if (symmetric && j > i) continue;
              float q = ao::screening::compute(norm, (*basis)[Is[i]], (*basis)[Js[j]]);
              ijs.pairs.push_back({ std::sqrt(q), { Is.at(i), Js.at(j) } });
            }
          }
          if (ijs.pairs.empty()) continue;

          std::sort(ijs.pairs.begin(), ijs.pairs.end());
          for (auto &[q,ij] : ijs.pairs) {
            ijs.indices.push_back(ij);
            ijs.norms.push_back(q);
          }

          size_t nab = ijs.pairs.size();

          int na = nbf(A);
          int nb = nbf(B);

          F_buffer.assign(na*nb*nab, 0.0);

          auto Fab = TensorRef<double,2>{
            F_buffer.data(),
            { nab, (size_t)na*nb }
          };

          // J matrix

          auto bra = coulomb.make_bra(ijs.indices, ijs.norms.data());

          for (auto &[X,xs] : Xs) {
            std::vector<Index2> xpairs;
            std::vector<double> xnorms;
            for (auto [ix,nx] : xs) {
              if (ijs.norms.back()*nx*d_norms[ix] < precision) continue;
              xpairs.push_back({ix,ix});
              xnorms.push_back(nx);
            }
            if (xpairs.empty()) continue;
            int nx = nbf(X);
            auto contract = [&](BraKet<Index1> idx, BraKet<size_t> dims, const auto &V) {
              for (size_t kl = 0; kl < dims.ket; ++kl) {
                auto *Dx = D.data() + minbs->range(xpairs.at(kl+idx.ket).first).begin();
                for (int ix = 0; ix < nx; ++ix) {
                  auto dx = Dx[ix];
                  //if (!dx) continue;
                  for (int ib = 0, iab = 0; ib < nb; ++ib) {
                    for (int ia = 0; ia < na; ++ia, ++iab) {
                      auto *f = &Fab(idx.bra,iab);
                      auto *v = &V(0,ia,ib,ix,ix,kl);
                      for (size_t ij = 0; ij < dims.bra; ++ij) {
                        (*f++) += 2*dx*(*v++);
                      }
                    }
                  }
                }
              }
            }; // contract
            auto ket = coulomb.make_ket(xpairs, xnorms.data());
            coulomb.compute(Coulomb, *bra, *ket, contract);
          } // Xs

          for (size_t ij = 0; auto &[i,j] : ijs.indices) {

            std::vector<Index2> bra = { { Is.at(i), 0 } };
            std::vector<Index2> ket = { { Js.at(j), 0 } };

            const auto &a_norms = ax_norms[i];
            const auto &b_norms = bx_norms[j];

            auto &K = K_buffer;
            K.assign(na*nb, 0.0);

            for (size_t ix = 0; ix < minbs->size(); ++ix) {
              if (a_norms[ix]*b_norms[ix]*d_norms[ix] < precision) continue;
              int nx = nbf((*minbs)[ix]);
              bra[0].second = ix;
              ket[0].second = ix;
              const auto *Dx = D.data() + minbs->range(ix).begin();
              auto contract = [&](BraKet<Index1> idx, BraKet<size_t> dims, const auto &V) {
                for (int ib = 0; ib < nb; ++ib) {
                  for (int ia = 0; ia < na; ++ia) {
                    double f = 0;
                    for (int ix = 0; ix < nx; ++ix) {
                      f += Dx[ix]*V(0,ia,ix,ib,ix,0);
                    }
                    K[ia+ib*na] += f;
                  }
                }
              };
              exchange.compute(
                Coulomb,
                *exchange.make_bra(bra, nullptr),
                *exchange.make_ket(ket, nullptr),
                contract
              );
            }

            for (int iab = 0; iab < na*nb; ++iab) {
              Fab(ij,iab) -= K[iab];
            }

            ++ij;

          }

          std::vector<double> f1(na*nb);
          std::vector<double> f2(na*nb);
          for (size_t ij = 0; ij < ijs.indices.size(); ++ij) {
            auto *Fij = F_buffer.data() + ij;
            for (int ib = 0; ib < nb; ++ib) {
              for (int ia = 0; ia < na; ++ia) {
                f1[ia+ib*na] = *Fij;
                f2[ib+ia*nb] = *Fij;
                Fij += ijs.indices.size();
              }
            }
            auto [i,j] = ijs.indices[ij];
            F({i,j}, f1.data());
            if (!symmetric || i == j) continue;
            F({j,i}, f2.data());
          }

        }
      }

    } // omp parallel

  }

  void SOAD::fock(
    const Wfn<Gaussian> &wfn,
    std::function<void(Index2, const double*)> F,
    const std::vector<Index1> &is,
    const std::vector<Index1> &js)
  {
    using S = std::tuple<int, std::vector<Gaussian::Primitive> >;
    static const std::map<int, std::vector<S> > basis_set = {
#include "libintx/hf/sto-3g.h"
    };
    auto minbs = std::make_shared< Basis<Gaussian> >();
    std::vector<double> mind;
    for (auto [Z,r] : wfn.atoms()) {
      int ne = Z;
      auto basis = libintx::make_basis(std::vector{ std::tuple{Z,r} }, basis_set);
      // sto-3g should be: 0=1s, 1=2s, 2=2p, 3=3s, 4=3p, 5=4s, 6=4p, 7=3d
      for (int idx = 0; auto &g : *basis) {
        int nbf = libintx::nbf(g);
        int ne2 = std::min(ne,2*nbf);
        if (Z <= 36) {
          // singly occupied 4s
          if (idx == 5 && (Z == 19 || Z == 24 || Z == 29)) {
            ne2 = std::min(ne2,1);
          }
          // 3d filled before 4p
          if (idx == 6) {
            ne2 = std::clamp<int>(Z-30, 0, 6);
          }
        }
        double d = double(ne2)/nbf;
        d /= 2; // normalise
        // partition ne2 electrons over all orbital
        for (int i = 0; i < nbf; ++i) {
          mind.push_back(d);
        }
        minbs->push_back(g);
        ne -= ne2;
        ++idx;
      }
    }
    libintx_assert(mind.size() == minbs->nbf());
    fock2(wfn.basis(), minbs, mind, F, is, js, this->num_threads, this->norm, this->precision);
  }

  void SOAD::fock(const Wfn<Gaussian> &wfn, MatrixRef<double> F) {
    auto Fij = [&](auto idx, auto *src) {
      auto ri = wfn.basis()->range(idx.first);
      auto rj = wfn.basis()->range(idx.second);
      for (int j = rj.begin(); j != rj.end(); ++j) {
        for (int i = ri.begin(); i != ri.end(); ++i) {
          F(i,j) += *src++;
        }
      }
    };
    std::vector<Index1> index(wfn.basis()->size());
    for (size_t i = 0; i < index.size(); ++i) {
      index[i] = i;
    }
    SOAD::fock(wfn, Fij, index, index);
  }

}
