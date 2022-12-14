/**
 * @file profiling.hpp
 *
 * @brief wrappers for the CUDA profiler API functions,
 * and convenience RAII classes for profiler-output-marked
 * time range and points.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PROFILING_HPP_
#define CUDA_API_WRAPPERS_PROFILING_HPP_

#include <cuda/api/types.hpp>
#include <cstdint>
#include <string>

namespace cuda {

namespace profiling {

struct color_t {
	using underlying_type = uint32_t;
	unsigned char alpha, red, green, blue;

	static constexpr color_t from_hex(underlying_type raw_argb) noexcept {
		return {
			(unsigned char) ((raw_argb >> 24) & 0xFF),
			(unsigned char) ((raw_argb >> 16) & 0xFF),
			(unsigned char) ((raw_argb >>  8) & 0xFF),
			(unsigned char) ((raw_argb >>  0) & 0xFF),
		};
	}
	operator underlying_type() const noexcept { return as_hex(); }
	underlying_type as_hex() const noexcept
	{
		return
			((underlying_type) alpha)  << 24 |
			((underlying_type) red  )  << 16 |
			((underlying_type) green)  <<  8 |
			((underlying_type) blue )  <<  0;
	}
	static constexpr color_t Black()       noexcept { return from_hex(0x00000000); }
	static constexpr color_t White()       noexcept { return from_hex(0x00FFFFFF); }
	static constexpr color_t FullRed()     noexcept { return from_hex(0x00FF0000); }
	static constexpr color_t FullGreen()   noexcept { return from_hex(0x0000FF00); }
	static constexpr color_t FullBlue()    noexcept { return from_hex(0x000000FF); }
	static constexpr color_t FullYellow()  noexcept { return from_hex(0x00FFFF00); }
	static constexpr color_t LightRed()    noexcept { return from_hex(0x00FFDDDD); }
	static constexpr color_t LightGreen()  noexcept { return from_hex(0x00DDFFDD); }
	static constexpr color_t LightBlue()   noexcept { return from_hex(0x00DDDDFF); }
	static constexpr color_t LightYellow() noexcept { return from_hex(0x00FFFFDD); }
	static constexpr color_t DarkRed()     noexcept { return from_hex(0x00880000); }
	static constexpr color_t DarkGreen()   noexcept { return from_hex(0x00008800); }
	static constexpr color_t DarkBlue()    noexcept { return from_hex(0x00000088); }
	static constexpr color_t DarkYellow()  noexcept { return from_hex(0x00888800); }
};

namespace range {
enum class Type { unspecified, kernel, pci_express_transfer	};
/**
 * The range handle is actually `nvtxRangeId_t`; but - other than this typedef,
 * we don't need to include the nVIDIA Toolkit Extensions headers at all here,
 * and can leave them within the implementation only.
 */
using handle_t = uint64_t;
} // namespace range

namespace mark {

void point (const ::std::string& message, color_t color);

inline void point (const ::std::string& message)
{
	point(message, color_t::Black());
}

range::handle_t range_start (
	const ::std::string&  description,
	range::Type         type,
	color_t             color);

inline range::handle_t range_start (
	const ::std::string&  description,
	range::Type         type)
{
	return range_start(description, type, color_t::LightRed());
}

inline range::handle_t range_start (
	const ::std::string&  description)
{
	return range_start(description, range::Type::unspecified);
}

void range_end (range::handle_t range);

} // namespace mark


/**
 * A RAII class whose scope of existence is reflected as a range in the profiler.
 * Use it in the scope in which you perform some interesting operation, e.g.
 * perform a synchronous I/O operation (and have it conclude of course), or
 * launch and synch several related kernels.
 */
class scoped_range_marker {
public:
	scoped_range_marker(
		const ::std::string& description,
		profiling::range::Type type = profiling::range::Type::unspecified);
	~scoped_range_marker();
protected:
	profiling::range::handle_t range;
};

/**
 * Start CUDA profiling for the current process
 */
void start();

/**
 * Sttop CUDA profiling for the current process
 */
void stop();

/**
 * A class to instantiate in the part of your application
 * which does any work you intend to use the CUDA profiler
 * to profile. This could well be your main() function.
 */
class scope {
public:
	scope() { start(); }
	~scope() { stop(); }
};

namespace naming {

/**
 * @brief Have the profiler refer to a thread using a specified string
 * identifier (rather than its numeric ID).
 *
 * @param[in] thread_id  A native numeric ID of the thread; on Linux systems
 * this would be a `pthread_t`, and on Windows - a DWORD (as is returned,
 * for example, by `GetCurrentThreadId()`)
 * @param[in] name The string identifier to use for the specified thread
 */
template <typename CharT>
void name_host_thread(uint32_t thread_id, const ::std::basic_string<CharT>& name);
#if defined(__unix__) || defined(_WIN32)
template <typename CharT>
void name_this_thread(const ::std::basic_string<CharT>& name);
#endif

//void name_device_stream(device::id_t  device, stream::id_t stream);

} // namespace naming

} // namespace profiling

} // namespace cuda

#endif // CUDA_API_WRAPPERS_PROFILING_HPP_
