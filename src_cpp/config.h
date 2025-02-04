#ifndef SCITE_RNA_CONFIG_H
#define SCITE_RNA_CONFIG_H

#include <map>
#include <string>

extern std::map<std::string, std::string> config_variables;
void load_config(const std::string& file_path);

#endif //SCITE_RNA_CONFIG_H
