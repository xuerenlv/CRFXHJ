//
// Created by 薛洪见 on 2018/3/5.
//

#include "utils.h"


double logsumexp(double x, double y, bool is_x_zero) {
	if (is_x_zero) return y;  // init mode
	const double vmin = std::min(x, y);
	const double vmax = std::max(x, y);
	if (vmax > vmin + MINUS_LOG_EPSILON) {
		return vmax;
	} else {
		return vmax + std::log(std::exp(vmin - vmax) + 1.0);
	}
}

ParseResult parseArgs(string cof_file_name) {
	std::ifstream ifs(cof_file_name);
	ParseResult pr = toml::parse(ifs);
	if (!pr.valid()) {
		cout<<pr.errorReason<<endl;
		exit(0);
	}
	return pr;
}


vector<string> string_split(const string& s, const string &c) {
	vector<string> v;
	string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = s.find(c);
	while(string::npos != pos2){
		v.push_back(s.substr(pos1, pos2-pos1));
		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if(pos1 != s.length())
		v.push_back(s.substr(pos1));
	return v;
}
























