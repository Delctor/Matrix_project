#pragma once
#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <quickSort.h>

#define True 0b11111111
#define False 0b00000000

#define uint32_to_float(v) __m256i v2 = _mm256_srli_epi32(v, 1); \
    __m256i v1 = _mm256_sub_epi32(v, v2); \
    __m256 v2f = _mm256_cvtepi32_ps(v2); \
    __m256 v1f = _mm256_cvtepi32_ps(v1); \
    __m256 uint32ToFloat = _mm256_add_ps(v2f, v1f);

#define uint64_to_double(v) __m256d uint64ToDouble = _mm256_add_pd(_mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(_mm256_srli_epi64(v, 32), mask1)), _mm256_castsi256_pd(mask3)), _mm256_castsi256_pd(_mm256_blend_epi16(v, mask2, 204)));

#define masks_uint64_to_double __m256i mask1 = _mm256_castpd_si256(_mm256_set1_pd(19342813113834066795298816.0)); \
	__m256i mask2 = _mm256_castpd_si256(_mm256_set1_pd(4503599627370496.0)); \
	__m256i mask3 = _mm256_castpd_si256(_mm256_set1_pd(19342813118337666422669312.0));

#ifdef __cplusplus
#define INITIALIZER(f) \
        static void f(void); \
        struct f##_t_ { f##_t_(void) { f(); } }; static f##_t_ f##_; \
        static void f(void)
#elif defined(_MSC_VER)
#pragma section(".CRT$XCU",read)
#define INITIALIZER2_(f,p) \
        static void f(void); \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker,"/include:" p #f "_")) \
        static void f(void)
#ifdef _WIN64
#define INITIALIZER(f) INITIALIZER2_(f,"")
#else
#define INITIALIZER(f) INITIALIZER2_(f,"_")
#endif
#else
#define INITIALIZER(f) \
        static void f(void) __attribute__((constructor)); \
        static void f(void)
#endif

namespace alge
{
    __m256i __seeds__;

    INITIALIZER(initialize)
    {
        uint64_t seeds[4];
        _rdrand64_step(seeds);
        _rdrand64_step(seeds + 1);
        _rdrand64_step(seeds + 2);
        _rdrand64_step(seeds + 3);
        __seeds__ = _mm256_loadu_epi64(seeds);
    }

    template <typename T>
    class vector
    {
        static_assert(std::is_same<T, double>::value ||
            std::is_same<T, float>::value ||
            std::is_same<T, int>::value ||
            std::is_same<T, uint64_t>::value ||
            std::is_same<T, int64_t>::value ||
            std::is_same<T, uint8_t>::value
            ,
            "The data type can only be double, float, int, uint64_t, int64_t or uint8_t");
    };

    template <typename T, bool tranposed = false, bool contiguous = true>
    class matrix
    {
        static_assert(std::is_same<T, double>::value ||
            std::is_same<T, float>::value ||
            std::is_same<T, uint8_t>::value
            ,
            "The data type can only be double, float or uint8_t");
    };
}
