/**
 * @file peer_to_peer.hpp
 *
 * @brief Settings and actions related to the interaction of multiple devices (adding
 * on those already in @ref device.hpp)
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
#define CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_

#include <cuda/api/device.hpp>

namespace cuda {
namespace device {
namespace peer_to_peer {

/**
 * @brief The value of type for all CUDA device "attributes"; see also @ref attribute_t.
 */
using attribute_value_t = int;

/**
 * @brief An identifier of a integral-numeric-value attribute of a CUDA device.
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings (see
 * @ref cuda::device::properties_t), and flags are either binary or
 * small-finite-domain type fitting into an overall flagss value (see
 * @ref cuda::device_t::flags_t). Flags and properties are obtained all at once,
 * attributes are more one-at-a-time.
 */
using attribute_t = cudaDeviceP2PAttr;

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 * @return true iff acess is possible
 */
inline bool can_access(device_t<> accessor,	device_t<> peer)
{
	return accessor.can_access(peer);
}

/**
 * @brief Enable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
template<bool AccessorIsAssumedCurrent>
inline void enable_access(
	device_t<AccessorIsAssumedCurrent>                accessor,
	device_t<> peer)
{
	return accessor.enable_access_to(peer);
}

/**
 * @brief Disable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
template<bool AccessorIsAssumedCurrent>
inline void disable_access(
	device_t<AccessorIsAssumedCurrent>&               accessor,
	device_t<> peer)
{
	return accessor.disable_access_to(peer);
}


template<bool FirstAccessorIsAssumedCurrent, bool SecondAccessorIsAssumedCurrent>
inline void can_access_each_other(device_t<FirstAccessorIsAssumedCurrent> first, device_t<SecondAccessorIsAssumedCurrent> second)
{
	return first.can_access(second) and second.can_access(first);
}


template<bool FirstAccessorIsAssumedCurrent, bool SecondAccessorIsAssumedCurrent>
inline void enable_bidirectional_access(device_t<FirstAccessorIsAssumedCurrent> first, device_t<SecondAccessorIsAssumedCurrent> second)
{
	enable_access<FirstAccessorIsAssumedCurrent >(first,  second);
	enable_access<SecondAccessorIsAssumedCurrent>(second, first );
}

template<bool FirstAccessorIsAssumedCurrent, bool SecondAccessorIsAssumedCurrent>
inline void disable_bidirectional_access(device_t<FirstAccessorIsAssumedCurrent> first, device_t<SecondAccessorIsAssumedCurrent> second)
{
	// Note: What happens when first and second have the same id?
	disable_access<FirstAccessorIsAssumedCurrent >(first,  second);
	disable_access<SecondAccessorIsAssumedCurrent>(second, first );
}

/**
 * @brief Get one of the numeric attributes for a(n ordered) pair of devices,
 * relating to their interaction
 *
 * @note This is the device-pair equivalent of @ref device_t::get_attribute()
 *
 * @param attribute identifier of the attribute of interest
 * @param source source device
 * @param destination destination device
 * @return the numeric attribute value
 */
inline attribute_value_t get_attribute(
	attribute_t       attribute,
	const device_t<>  accessor,
	const device_t<>  peer)
{
	attribute_value_t value;
	auto status = cudaDeviceGetP2PAttribute(&value, attribute, accessor.id(), peer.id());
	throw_if_error(status,
		"Failed obtaining peer-to-peer device attribute for device pair (" + ::std::to_string(accessor.id()) + ", "
			+ ::std::to_string(peer.id()) + ')');
	return value;
}

} // namespace peer_to_peer
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
