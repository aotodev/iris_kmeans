#pragma once

#include "../vec4.hpp"

#include <stdint.h>
#include <string>

#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>


/* just a helper to load the iris data */
inline std::vector<vec4> load_iris_data_vec4(const std::string& path, bool hasHeader, uint32_t obervationCount)
{
    /* RAII, no need to close it */
    std::ifstream csv_stream(path);

    if (csv_stream.fail())
        std::cout << "failed to load file iris data\n";

    std::string line, temp;
    std::stringstream stream;
    
    // ignore label line
    if(hasHeader)
        std::getline(csv_stream, line);

    std::vector<vec4> outData;
    outData.reserve(obervationCount ? obervationCount : 256);

    while (std::getline(csv_stream, line))
    {
        stream << line;
        auto& point = outData.emplace_back(0);

        for(int i = 0; i < 4; i++)
        {
            std::getline(stream, temp, ',');
            point[i] = std::stof(temp);
        }

        stream.clear();
    }

    /* release any extra memory if needed */
    outData.shrink_to_fit();

    return outData;
}
