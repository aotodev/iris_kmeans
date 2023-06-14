#pragma once

#include <cstdint>
#include <stdio.h>
#include <limits>
#include <random>
#include <vector>

#include "vec4.hpp"


static std::random_device s_random_device;
static std::default_random_engine s_random_engine(s_random_device());


template<size_t centroidsCount, size_t it>
[[nodiscard]] static std::vector<uint32_t> kmeans_vec4(const vec4* pDataPoints, const size_t dataCount)
{
	std::uniform_int_distribution<size_t> uniform_distribution(0ULL, dataCount - 1);

	struct centroid { vec4 pos; float count = 0.0f; };

	centroid centroids[centroidsCount];
	centroid centroidsSum[centroidsCount];

	/* allocate memory for the labels */
	alignas(64) std::vector<uint32_t> labels(dataCount, 0u); 

	/* init centroids */
	for (size_t i = 0; i < centroidsCount; i++)
	{
		centroids[i].pos = pDataPoints[uniform_distribution(s_random_engine)];
		centroids[i].count = 0.0f;
	}

	size_t iterations = it;

	while (iterations--)
	{
		/* classify data points according to the centroid's new position */
		for (size_t i = 0; i < dataCount; i++)
		{
			float label = 0.0f;
			float previous = std::numeric_limits<float>::max();

			for (size_t j = 0; j < centroidsCount; j++)
			{
				float current = distance_vec4(pDataPoints[i], centroids[j].pos);
				label = current < previous ? float(j) : label;

				previous = current;
			}

			labels[i] = (uint32_t)label;

			centroidsSum[(size_t)label].pos += pDataPoints[i];
			centroidsSum[(size_t)label].count++;
		}

		/* update centroids' new position according to the new classification */
		for (size_t i = 0; i < centroidsCount; i++)
		{
			centroids[i].pos = centroidsSum[i].pos / (centroidsSum[i].count + std::numeric_limits<float>::min());

			/* reset sum for the next iteration */
			centroidsSum[i].pos = vec4();
			centroidsSum[i].count = 0.0f;
		}
		
	}

	for(uint32_t i = 0; i < centroidsCount; i++)
		printf("centroid[%u] == [ %.3f, %.3f, %.3f, %.3f ]\n", i,
			centroids[i].pos.x,
			centroids[i].pos.y,
			centroids[i].pos.z,
			centroids[i].pos.w);

	return labels;
}
