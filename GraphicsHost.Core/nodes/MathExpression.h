#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace gh::nodes {

class MathError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Evaluates an arithmetic expression: numbers, the variables in `vars` (single letters), + - * / % ^ (also written × ÷ −),
// unary minus, parentheses, and the functions min(x, y), max(x, y), round(x), floor(x), ceil(x), abs(x), sqrt(x).
// Throws MathError with a message for people (unknown name, unbalanced parenthesis, division by zero ...).
double evaluateMath(const std::string& expression, const std::map<char, double>& vars);

}  // namespace gh::nodes
