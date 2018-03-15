//
// Created by 薛洪见 on 2018/3/6.
//

#include "path.h"





void Path::calcCost(const lbfgsfloatval_t* alpha, int label_size, float cost_factor) {
	float c = 0;
	for (auto f : *fvector) {
		c += alpha[f + lnode->y*label_size + rnode->y];
	}
	cost = cost_factor * c;
}


void Path::calcExpectation(lbfgsfloatval_t* alpha_gradient, double Z, size_t label_size) const {
	const double c = std::exp(lnode->alpha + cost + rnode->beta - Z);
	for (auto f : *fvector) {
		alpha_gradient[f + lnode->y * label_size + rnode->y] += c;
	}
}


void Path::add(Node *_lnode, Node *_rnode) {
	lnode = _lnode;
	rnode = _rnode;
	lnode->rpath.push_back(this);
	rnode->lpath.push_back(this);
}










