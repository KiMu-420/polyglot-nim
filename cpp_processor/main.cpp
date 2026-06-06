#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "json.hpp"

using json = nlohmann::json;

std::vector<std::string> chunk_text(const std::string& text, int max_words = 500) {
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) words.push_back(word);
    
    std::vector<std::string> chunks;
    for (size_t i = 0; i < words.size(); i += max_words) {
        std::string chunk;
        for (size_t j = i; j < std::min(i + max_words, words.size()); ++j)
            chunk += words[j] + " ";
        chunks.push_back(chunk);
    }
    return chunks;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: processor <input.txt> <output.json>" << std::endl;
        return 1;
    }
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Cannot open input file." << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string content = buffer.str();
    
    auto chunks = chunk_text(content);
    json j;
    j["chunks"] = chunks;
    std::ofstream out(argv[2]);
    out << j.dump(2);
    std::cout << "Created " << chunks.size() << " chunks." << std::endl;
    return 0;
}
