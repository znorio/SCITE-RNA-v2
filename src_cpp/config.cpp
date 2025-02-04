#include "config.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <stack>

std::map<std::string, std::string> config_variables;

std::string rtrim(const std::string& str) {
    size_t last = str.find_last_not_of(" \t");
    return (last == std::string::npos) ? "" : str.substr(0, last + 1);
}

void load_config(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::string line;
    std::string current_section; // To track the current section for nested keys

    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = rtrim(line); // Only trim trailing whitespaces

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        size_t leading_spaces = line.find_first_not_of(" \t");

        if (line.back() == ':') {
            // New section header
            current_section = rtrim(line.substr(0, line.size() - 1)); // Trim trailing ':'
        } else if (line.find(":") != std::string::npos) {
            // Key-value pair
            size_t colon_pos = line.find(":");
            std::string key = rtrim(line.substr(0, colon_pos));
            std::string value = rtrim(line.substr(colon_pos + 1));

            if (leading_spaces > 0 && !current_section.empty()) {
                // Nested key
                config_variables[current_section + "." + key] = value;
            } else {
                // Reset to a top-level key
                current_section.clear();
                config_variables[key] = value;
            }
        }
    }

    // Debug: Print all key-value pairs
//    std::cout << "Loaded configuration variables:" << std::endl;
//    for (const auto& [key, value] : config_variables) {
//        std::cout << key << ": " << value << std::endl;
//    }
}
