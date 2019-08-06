#include "utils.hpp"

int loadDirectory(std::string dir_path, std::vector<std::string>& file_list, string pattern) {
    int pattern_len = pattern.size();
    DIR * dir;
    struct dirent * ptr;
    string x, dirPath;
    dir = opendir(dir_path.c_str()); //打开一个目录
    while((ptr = readdir(dir)) != NULL) {
        x = ptr->d_name;
        if (x == "." or x == "..")
            continue;
        if (x.find(pattern) == string::npos) 
            continue;
        dirPath = dir_path + "/" + x;
        // printf(">>> file_path : %s\n", dirPath.c_str()); //输出文件绝对路径
        file_list.emplace_back(x.substr(0, x.size()-pattern_len));
    }
    closedir(dir);

    sort(file_list.begin(), file_list.end(), [](const string& a, const string& b) {
        return atof(a.c_str()) < atof(b.c_str());
    });

    int N = file_list.size();
    file_list.resize(N);
    return N;
}
