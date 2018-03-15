#include <iostream>
#include "utils.h"
#include "dataset.h"
#include "model.h"
#include <lbfgs.h>


void test_model(ParseResult *pr){
	CRFModel crfmodel(pr);
	crfmodel.init_test();
	crfmodel.run_test();
}


void train_model(ParseResult *pr){
	CRFModel crfmodel(pr);
	crfmodel.init_train();
	if (pr->value.get<int>("crf.algorithm") == 0){ // L1 L2
		pp("---- L1 - L2 ----");
		crfmodel.run_train_regul();
	}else if (pr->value.get<int>("crf.algorithm") == 1){ // MIRA
		pp("---- MIRA ----");
		crfmodel.run_train_mira();
	}
	crfmodel.save_model();
}


int main(int argc, char *argv[]) {
	if(string(argv[1])=="train") {
		ParseResult pr = parseArgs("crf_train_conf.toml");
		cout<<"TRAIN START !"<<endl;
		train_model(&pr);
	}else if(string(argv[1])=="test") {
		ParseResult pr = parseArgs("crf_test_conf.toml");
		cout<<"TEST START !"<<endl;
		test_model(&pr);
	}else{
		pp("no train or test !");
	}
}


















