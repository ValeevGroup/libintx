/**
 * @file current_device.hpp
 *
 * @brief Wrappers for getting and setting CUDA's choice of
 * which device is 'current'
 *
 * CUDA has one device set as 'current'; and much of the Runtime API
 * implicitly refers to that device only. This file contains wrappers
 * for getting and setting it - as standalone functions - and
 * a RAII class which can be used for setting it for the duration of
 * a scope, popping back the old setting as the scope is exited.
 *
 * @note that code for getting the current device as a CUDA device
 * proxy class is found in @ref device.hpp
 *
 * @note the scoped device setter is used extensively throughout
 * this CUDA API wrapper library.
 *
 */
#ifndef CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
#define CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime_api.h>

namespace cuda {
namespace device {
namespace current {

/**
 * Obtains the numeric id of the device set as current for the CUDA Runtime API
 */
inline device::id_t get_id()
{
	device::id_t  device;
	status_t result = cudaGetDevice(&device);
	throw_if_error(result, "Failure obtaining current device index");
	return device;
}

/**
 * Set a device as the current one for the CUDA Runtime API (so that API calls
 * not specifying a device apply to it.)
 *
 * @param[in] device Numeric ID of the device to make current
 */
inline void set(device::id_t  device)
{
	status_t result = cudaSetDevice(device);
	throw_if_error(result, "Failure setting current device to " + ::std::to_string(device));
}

/**
 * Reset the CUDA Runtime API's current device to its default value - the default device
 */
inline void set_to_default() { return set(device::default_device_id); }


/**
 * A RAII-based mechanism for setting the CUDA Runtime API's current device for
 * what remains of the current scope, and changing it back to its previous value
 * when exiting the scope.
 *
 * @tparam AssumedCurrent the current device override is also used in code which
 * can be instantiated when the current device has already been set, or when it
 * has not been set; for this reason, the scoped current device override also
 * has this feature (which when set to `true` makes it into a do-nothing
 * object).
 */
template <bool AssumedCurrent = false> class scoped_override_t;

template <>
class scoped_override_t<detail::do_not_assume_device_is_current> {
protected:
	// Note the previous device and the current one might be one and the same;
	// in that case, the push is idempotent (but who guarantees this? Hmm.)
	static inline device::id_t  push(device::id_t new_device)
	{
		device::id_t  previous_device = device::current::get_id();
		device::current::set(new_device);
		return previous_device;
	}
	static inline void pop(device::id_t  old_device) { device::current::set(old_device); }

public:
	scoped_override_t(device::id_t  device) { previous_device = push(device); }
	~scoped_override_t() { pop(previous_device); }
private:
	device::id_t  previous_device;
};

template <>
class scoped_override_t<detail::assume_device_is_current> {
public:
	scoped_override_t(device::id_t) { }
	~scoped_override_t() { }
};


/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_DEVICE_FOR_THIS_SCOPE(_device_id) \
	::cuda::device::current::scoped_override_t<::cuda::detail::do_not_assume_device_is_current> scoped_device_override(_device_id)


} // namespace current
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
