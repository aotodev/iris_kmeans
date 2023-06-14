#pragma once

#include <cstdint>
#include <assert.h>
#include <limits>
#include <cmath>
#include <iostream>

#include <immintrin.h>
#include <ostream>

struct vec4
{
	float& x = m_array[0];
	float& y = m_array[1];
	float& z = m_array[2];
	float& w = m_array[3];

	void reset()
	{
		x = 0.0f; y = 0.0f; z = 0.0f; w = 0.0f;
	}

	vec4() = default;

	vec4(float v)
		: m_array{ v, v, v, v }
	{}

	vec4(float xv, float yv, float zv, float wv)
		: m_array{ xv, yv, zv, wv }
	{}

	~vec4() = default;

	vec4(const vec4& other)
	{
		_mm_store_ps(m_array, _mm_load_ps(other.m_array));
	}

	vec4& operator=(const vec4& other)
	{
		_mm_store_ps(m_array, _mm_load_ps(other.m_array));
		return *this;
	}

	vec4(const vec4&& other)
	{
		_mm_store_ps(m_array, _mm_load_ps(other.m_array));
	}

	vec4& operator=(vec4&& other)
	{
		_mm_store_ps(m_array, _mm_load_ps(other.m_array));
		return *this;
	}

	vec4(__m128 mv)
	{
		_mm_store_ps(m_array, mv);
	}

	vec4& operator=(const __m128& mv)
	{
		_mm_store_ps(m_array, mv);
		return *this;
	}

	bool operator==(const vec4& other) const
	{
		return std::abs(sum() - other.sum()) < std::numeric_limits<float>::epsilon();
	}

	bool operator!=(const vec4& other) const
	{
		return !(*this == other);
	}

	inline float& operator[](uint32_t index)
	{
		assert(index < 4);

		return m_array[index];
	}

	inline vec4& operator+=(const vec4& other)
	{
		_mm_store_ps(this->m_array, _mm_add_ps(_mm_load_ps(this->m_array), _mm_load_ps(other.m_array)));
		return *this;
	}

	inline vec4& operator-=(const vec4& other)
	{
		_mm_store_ps(this->m_array, _mm_sub_ps(_mm_load_ps(this->m_array), _mm_load_ps(other.m_array)));
		return *this;
	}

	inline vec4& operator*=(const vec4& other)
	{
		_mm_store_ps(this->m_array, _mm_mul_ps(_mm_load_ps(this->m_array), _mm_load_ps(other.m_array)));
		return *this;
	}

	inline vec4& operator/=(const vec4& other)
	{
		_mm_store_ps(this->m_array, _mm_div_ps(_mm_load_ps(this->m_array), _mm_load_ps(other.m_array)));
		return *this;
	}

	inline vec4 operator+(const vec4& other) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_add_ps(_mm_load_ps(this->m_array),_mm_load_ps(other.m_array)));
		return out;
	}

	inline vec4 operator-(const vec4& other) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_sub_ps(_mm_load_ps(this->m_array),_mm_load_ps(other.m_array)));
		return out;
	}

	inline vec4 operator*(const vec4& other) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_mul_ps(_mm_load_ps(this->m_array),_mm_load_ps(other.m_array)));
		return out;
	}

	inline vec4 operator/(const vec4& other) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_div_ps(_mm_load_ps(this->m_array),_mm_load_ps(other.m_array)));
		return out;
	}

	inline vec4 operator+(float constant) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_add_ps(_mm_load_ps(this->m_array),_mm_set_ps1(constant)));
		return out;
	}

	inline vec4 operator-(float constant) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_sub_ps(_mm_load_ps(this->m_array),_mm_set_ps1(constant)));
		return out;
	}

	inline vec4 operator*(float constant) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_mul_ps(_mm_load_ps(this->m_array),_mm_set_ps1(constant)));
		return out;
	}

	inline vec4 operator/(float constant) const
	{
		vec4 out;
		_mm_store_ps(out.m_array, _mm_div_ps(_mm_load_ps(this->m_array),_mm_set_ps1(constant)));
		return out;
	}

	friend std::ostream& operator<<(std::ostream& os, const vec4 vec)
	{
		os << vec.x << ", " << vec.y << ", "  << vec.z << ", "  << vec.w;
		return os;
	}

	float* ptr() { return m_array; }
	const float* ptr() const { return m_array; }

	// returns the sum of all 4 components
	inline float sum() const
	{
		return m_array[0] + m_array[1] + m_array[2] + m_array[3];
	}

	inline float distance(const vec4& v2)
	{
		__m128 dif = _mm_load_ps((v2 - *this).m_array);
		dif = _mm_mul_ps(dif, dif);

		vec4 temp;
		_mm_store_ps(temp.m_array, dif);

		return std::sqrt(temp.sum());
	}

private:
	alignas(16) float m_array[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
};

static inline float distance_vec4(const vec4& v1, const vec4& v2)
{
	__m128 dif = _mm_load_ps((v2 - v1).ptr());
	dif = _mm_mul_ps(dif, dif);

	vec4 temp;
	_mm_store_ps(temp.ptr(), dif);

	return std::sqrt(temp.sum());
}
