//
// Created by 薛洪见 on 2018/3/6.
//

#ifndef CRF_XHJ_MODEL_H
#define CRF_XHJ_MODEL_H

#include "dataset.h"
#include "feature.h"
#include "utils.h"
#include "toml.h"
#include "parallel.h"
#include <lbfgs.h>


class CRFModel{
public:
	CRFModel(ParseResult* p):pr(p), l1_param(0.0), l2_param(0.0){}
	
	void save_model();
	
	
	// ********************** test start**********************
	void init_test();
	
	void run_test(){
		dataSet->buildFeatures();
		dataSet->buildGraph();
		dataSet->calcCost(model_alpha, stringToNum<float>(pr->value.get<string>("crf.cost_factor")));
		dataSet->viterbi();
		struct eval_metric mm = dataSet->evalMetric();
		pp("Evaluation result: sacc="<<1.0-mm.serr<<" tacc="<<1.0-mm.terr);
		
		dataSet->saveForTest(pr->value.get<string>("crf.location_test_result"));
	};
	
	// ********************** test  end **********************
	
	
	// ********************** train start**********************
	void init_train();
	
	void run_train_mira() {
		pp("MIRA");
		int shrinking_size = pr->value.get<int>("crf.mira_shrinking_size");
		lbfgsfloatval_t C = pr->value.get<int>("crf.mira_C");
		
		float cost_factor = stringToNum<float>(pr->value.get<string>("crf.cost_factor"));
		unsigned int max_iterations = pr->value.get<int>("crf.maxiter");
		unsigned int sentence_num = (*(dataSet->getSentences())).size();
		auto sentences = dataSet->getSentences();
		vector<unsigned int> shrink(sentence_num);
		vector <lbfgsfloatval_t> upper_bound(sentence_num);
		
		std::fill(shrink.begin(), shrink.end(), 0);
		std::fill(upper_bound.begin(), upper_bound.end(), 0.0);
		
		int converge = 0;
		int all = 0;
		for (auto sent : *sentences) {
			all += sent->size();
		}
		
		for (unsigned int iter = 0; iter < max_iterations; ++iter) {
			int zeroone = 0;
			int err = 0;
			int active_set = 0;
			int upper_active_set = 0;
			double max_kkt_violation = 0.0;
			
			for (unsigned int i = 0; i < sentence_num; ++i) {
				if (shrink[i] >= shrinking_size) {
					continue;
				}
				
				++active_set;
				std::fill(gradient, gradient + feature_size, 0.0);
				dataSet->calcCostBysenindex(model_alpha, cost_factor, i);
				dataSet->viterbiBysenindex(i);
				int error_num = dataSet->eval_metricBysenindex(i);
				double cost_diff = dataSet->collins(model_alpha, cost_factor, gradient, i);
				
				err += error_num;
				if (error_num) {
					++zeroone;
				}
				
				if (error_num == 0) {
					++shrink[i];
				} else {
					shrink[i] = 0;
					double s = 0.0;
					for (unsigned int k = 0; k < feature_size; ++k) {
						s += gradient[k] * gradient[k];
					}
					double mu = std::max(0.0, (error_num - cost_diff) / s);
					if (upper_bound[i] + mu > C) {
						mu = C - upper_bound[i];
						++upper_active_set;
					} else {
						max_kkt_violation = std::max(error_num - cost_diff, max_kkt_violation);
					}

					if (mu > 1e-10) {
						upper_bound[i] += mu;
						upper_bound[i] = std::min(C, upper_bound[i]);
						for (unsigned int k = 0; k < feature_size; ++k) {
							model_alpha[k] += mu * gradient[k];
						}
					}
				}
			}
			
			
			double obj = 0.0;
			for (unsigned int i = 0; i < feature_size; ++i) {
				obj += model_alpha[i] * model_alpha[i];
			}
			pp("Iteration="<<iter<<" terr="<<1.0 * err / all<<" serr="<<1.0 * zeroone / sentence_num<<" act="<<active_set<<" uact="<<upper_active_set<<" obj="<<obj<<" kkt="<<max_kkt_violation);

			if (max_kkt_violation <= 0.0) {
				std::fill(shrink.begin(), shrink.end(), 0);
				converge++;
			} else {
				converge = 0;
			}
			if (converge == 2) {
				break;  // 2 is ad-hoc
			}
		}
	}
	
	
	void run_train_regul(){
		thread_num = pr->value.get<int>("crf.thread");
		thread_pool = new Parallel::Parallel(thread_num);
		thread_pool_input = new vector<vector<size_t>>(thread_num);
		for (size_t i = 0; i < thread_num; ++i) {
			vector<size_t> v;
			(*thread_pool_input).push_back(v);
		}
		for (size_t i = 0; i < (*(dataSet->getSentences())).size(); ++i) {
			(*thread_pool_input)[i%thread_num].push_back(i);
		}
		
		l1_param = stringToNum<float>(pr->value.get<string>("crf.regu_param_l1"));
		l2_param = stringToNum<float>(pr->value.get<string>("crf.regu_param_l2"));
		cost_factor = stringToNum<float>(pr->value.get<string>("crf.cost_factor"));
		
		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);
		param.max_iterations = pr->value.get<int>("crf.maxiter");
		param.past = 3;
		param.delta = stringToNum<float>(pr->value.get<string>("crf.eta"));
		if(l1_param > 0.0){
			param.orthantwise_c = l1_param;
			param.orthantwise_start = 0;
			param.orthantwise_end = feature_size;
			param.linesearch = 2; // 2
		}
		cout<<"l1 param:\t"<<l1_param<<endl;
		cout<<"l2 param:\t"<<l2_param<<endl;
		lbfgsfloatval_t fx;
		int ret = lbfgs(feature_size, model_alpha, &fx, _evaluate, _progress, this, &param);
		cout<<"L-BFGS optimization terminated with status code = "<<ret<<endl;
	};
	
	static lbfgsfloatval_t _evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {
		return reinterpret_cast<CRFModel*>(instance)->evaluate(x, g, n, step);
	}

	static int _progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
		return reinterpret_cast<CRFModel*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
	}

	lbfgsfloatval_t evaluate(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {
		std::fill(g, g+n, 0.0);
		lbfgsfloatval_t fx = 0.0;
		
		thread_pool->foreach(thread_pool_input->begin(), thread_pool_input->end(), [&](vector<size_t> v_vec){
			for(size_t i : v_vec){
				dataSet->calcCostBysenindex(x,cost_factor,i);
				dataSet->forwardbackwardBysenindex(i);
				fx += dataSet->calcGradientBysenindex(g,i);
			}
		});
		
		if(l2_param>0.0) {
			for (int i = 0; i < n; ++i) {
				fx += (l2_param * x[i] * x[i]) / 2.0;
				g[i] += l2_param * x[i] ;
			}
		}
		return fx;
	}

	int progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
		dataSet->calcCost(x, stringToNum<float>(pr->value.get<string>("crf.cost_factor")));
		dataSet->viterbi();
		struct eval_metric mm = dataSet->evalMetric();
		int zero_num = 0;
		for (int i = 0; i < n; ++i) {
			if(std::abs(x[i])<1e-5){
				zero_num++;
			}
		}
		pp("Iteration="<<k<<" obj="<<fx<<" zero_rate="<<(1.0*zero_num/n)<<" serr="<<mm.serr<<" terr="<<mm.terr);
		return 0;
	}
	// ********************** train end**********************

private:
	ParseResult* pr;
	DataSet* dataSet;
	unsigned int feature_size;
	lbfgsfloatval_t*   model_alpha;
	lbfgsfloatval_t*   gradient;
	float l1_param, l2_param;
	float cost_factor;
	
	Parallel::Parallel* thread_pool;
	vector<vector<size_t>>* thread_pool_input;
	unsigned int thread_num;
};





#endif //CRF_XHJ_MODEL_H
