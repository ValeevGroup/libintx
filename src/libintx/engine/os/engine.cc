#include "libintx/engine/os/engine.h"
#include "libintx/engine/os/os3.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/config.h"

namespace libintx::os {

  struct Kernel3 : Kernel<3> {
    Kernel3(const Gaussian& a, const Gaussian& b, const Gaussian& x) {
      static boys::Chebyshev<7,40,117,117*7> boys;
      this->kernel_ = ObaraSaika3::kernel(a,b,x,boys);
      assert(kernel_);
      buffer_.reset(new double[ncart(a)*ncart(b)*ncart(x)]);
    }
    const double* compute(const Double<3> &a, const Double<3> &b, const Double<3> &x) override {
      kernel_->compute(a, b, x, buffer_.get());
      return this->buffer();
    }
    const double* buffer() override {
      return buffer_.get();
    }
  private:
    std::unique_ptr<ObaraSaika3> kernel_;
    std::unique_ptr<double[]> buffer_;
  };

  std::unique_ptr< Kernel<3> > eri(const Gaussian& a, const Gaussian& b, const Gaussian& x) {
    return std::make_unique<Kernel3>(
      Kernel3(a,b,x)
    );
  }

}
