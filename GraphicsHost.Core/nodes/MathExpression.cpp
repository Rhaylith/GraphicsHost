#include "nodes/MathExpression.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>

namespace gh::nodes {

namespace {

class Parser {
public:
    Parser(const std::string& text, const std::map<char, double>& vars) : s_(normalise(text)), vars_(vars) {}

    double parse() {
        const double v = sum();
        skip();
        if (pos_ < s_.size()) throw MathError("unexpected '" + std::string(1, s_[pos_]) + "' in the expression");
        return v;
    }

private:
    // × ÷ − (UTF-8) become * / -
    static std::string normalise(const std::string& in) {
        std::string out;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in.compare(i, 2, "\xC3\x97") == 0) { out += '*'; ++i; }
            else if (in.compare(i, 2, "\xC3\xB7") == 0) { out += '/'; ++i; }
            else if (in.compare(i, 3, "\xE2\x88\x92") == 0) { out += '-'; i += 2; }
            else out += in[i];
        }
        return out;
    }

    void skip() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    bool take(char c) {
        skip();
        if (pos_ < s_.size() && s_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    double sum() {
        double v = product();
        for (;;) {
            if (take('+')) v += product();
            else if (take('-')) v -= product();
            else return v;
        }
    }
    double product() {
        double v = unary();
        for (;;) {
            if (take('*')) v *= unary();
            else if (take('/')) {
                const double d = unary();
                if (d == 0) throw MathError("division by zero");
                v /= d;
            } else if (take('%')) {
                const double d = unary();
                if (d == 0) throw MathError("division by zero");
                v = std::fmod(v, d);
            } else return v;
        }
    }
    double unary() {
        if (take('-')) return -unary();
        if (take('+')) return unary();
        return power();
    }
    double power() {
        const double base = atom();
        if (take('^')) return std::pow(base, unary());   // right-associative: 2^3^2 = 2^9
        return base;
    }
    double atom() {
        skip();
        if (pos_ >= s_.size()) throw MathError("the expression ends too early");
        if (take('(')) {
            const double v = sum();
            if (!take(')')) throw MathError("a parenthesis is not closed");
            return v;
        }
        const char c = s_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            char* end = nullptr;
            const double v = std::strtod(s_.c_str() + pos_, &end);
            pos_ = static_cast<size_t>(end - s_.c_str());
            return v;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t end = pos_;
            while (end < s_.size() && std::isalnum(static_cast<unsigned char>(s_[end]))) ++end;
            const std::string name = s_.substr(pos_, end - pos_);
            pos_ = end;
            if (take('(')) return call(name);
            if (name.size() == 1) {
                const auto it = vars_.find(name[0]);
                if (it != vars_.end()) return it->second;
            }
            throw MathError("unknown name '" + name + "'");
        }
        throw MathError("unexpected '" + std::string(1, c) + "' in the expression");
    }
    double call(const std::string& name) {
        std::vector<double> args;
        if (!take(')')) {
            do args.push_back(sum());
            while (take(','));
            if (!take(')')) throw MathError("a parenthesis is not closed");
        }
        const auto need = [&](size_t n) {
            if (args.size() != n) throw MathError(name + " takes " + std::to_string(n) + (n == 1 ? " value" : " values"));
        };
        if (name == "min") { need(2); return std::min(args[0], args[1]); }
        if (name == "max") { need(2); return std::max(args[0], args[1]); }
        if (name == "round") { need(1); return std::round(args[0]); }
        if (name == "floor") { need(1); return std::floor(args[0]); }
        if (name == "ceil") { need(1); return std::ceil(args[0]); }
        if (name == "abs") { need(1); return std::fabs(args[0]); }
        if (name == "sqrt") {
            need(1);
            if (args[0] < 0) throw MathError("sqrt of a negative number");
            return std::sqrt(args[0]);
        }
        throw MathError("unknown function '" + name + "'");
    }

    std::string s_;
    const std::map<char, double>& vars_;
    size_t pos_ = 0;
};

}  // namespace

double evaluateMath(const std::string& expression, const std::map<char, double>& vars) {
    return Parser(expression, vars).parse();
}

}  // namespace gh::nodes
