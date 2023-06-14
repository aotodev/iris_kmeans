#include "kmeans.hpp"
#include "iris_data.hpp"

#include <stdio.h>

int main()
{
    auto dataPoints = load_iris_data_vec4("data/iris_data.csv", true, 150);

    auto labels = kmeans_vec4<3, 10000>(dataPoints.data(), dataPoints.size());

    printf("\n------------------------------------\n");
    for(uint32_t i = 0; i < 150; i++)
        printf("labels[%u] == %u\n", i, labels[i]);
}
