#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace visioncore {

/**
 * @brief STL-compatible allocator for aligned memory allocation.
 * @tparam T The element type.
 * @tparam Alignment Alignment boundary in bytes (must be a power of 2, default is 32 for AVX).
 */
template <typename T, size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    AlignedAllocator() noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_t n) {
        if (n == 0) {
            return nullptr;
        }
        
        size_t size = n * sizeof(T);
        void* ptr = nullptr;
        
#if defined(_MSC_VER)
        ptr = _aligned_malloc(size, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, size) != 0) {
            throw std::bad_alloc();
        }
#endif
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t /*n*/) noexcept {
        if (p == nullptr) {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

} // namespace visioncore
