#include <vector>
#include <iostream>
#include <fstream>  
#include <dirent.h>
#include <stdio.h>
#include <string>
#include <algorithm>

using namespace std;

int loadDirectory(string dir_path, std::vector<std::string>& file_list, string pattern);
