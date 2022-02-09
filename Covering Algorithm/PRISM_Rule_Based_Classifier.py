# -*- coding: utf-8 -*-
"""
    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: PRISM, A Rule-Based Classifier
    Description: A rule-based classfier, featuring a non-ordered technique with a non-incremental
                 and selector-based learning style, comprising a general-especific learning approach
                 assuring a 100% precison of classification.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: November 2, 2017.
"""

import pandas as pd;
import numpy as np;


class RuleBasedClassifiers():

    X_train = [];
    X_test = [];
    y_train = [];
    y_test = [];

    n_samples = 0;
    n_features = 0;

    csv_datasets = ["contactLenses.xlsx"];

    csv_datasets_col_names = [['age','spectacle prescription','astigmatism','tear production rate','recommeded lenses']];


    #处理连续属性，
    def repair_continuous_attributes(self, dataset, features):

        self.n_samples = dataset.shape[0];
        self.n_features = dataset.shape[1] - 1;

        for feat in features:
            if dataset[feat].dtype == np.float64:
                dataset[feat] = dataset[feat].astype(int);


    #从csv文件中读入数据，并将文件对应的特征名赋给features
    def csv_processor(self, csv_path, feature_names):

        dataset = pd.read_excel(csv_path);
        dataset.columns = feature_names;
        return dataset;

    #修复缺失项
    def fix_dataset_missing_values(self,dataset):

        for column in dataset.columns:

            dataset[column] = dataset[column].replace('?',np.NaN);
            dataset[column] = dataset[column].fillna(dataset[column].value_counts().index[0]);


   
    def build_learning_sets(self,dataset,class_attr,train_size):

        #对数据集进行抽样，如果数据过大，可以通过对数据抽样降低计算规模，这里抽样比例设为1，相当于选择了所有数据
        #index指数据的行标，抽样数据后需要把数据重新编号，drop设为True避免将原来的行标作为一列放入数据
        dataset = dataset.sample(frac=1).reset_index(drop=True);
        
        #计算训练数据集大小及测试数据集大小
        n_train = int(self.n_samples*train_size);
#        n_test = self.n_samples - n_train;

        #预处理
        dataset_ = dataset.copy(deep=True);
        self.fix_dataset_missing_values(dataset_);
   
        #获取训练数据集和测试数据集
        self.y_train = dataset_.ix[0:n_train,class_attr].copy(deep=True);
        self.y_test = dataset_.ix[n_train+1:self.n_samples,class_attr].copy(deep=True);

        dataset_ = dataset_.drop(class_attr,1);

        self.X_train = dataset_.ix[0:n_train].copy(deep=True);
        self.X_test = dataset_.ix[n_train+1:self.n_samples].copy(deep=True);


    #显示数据集信息
    def display_data_info(self, dataset):

        print("\n1. Number of samples: " + str(self.n_samples));
        print("\n2. Number of features: " + str(self.n_features));
        print("\n3. Feature types:");
        print(dataset.dtypes);
        print("\n4. Data:");
        print(dataset);
        print("\n5. Training sets:");
        print(self.X_train);
        print(self.y_train);
        print("\n6. Testing sets:");
        print(self.X_test);
        print(self.y_test);


    def data_preprocessing(self):
        
        #默认读入第一个数据集
        print('A) ::Processing CSV files::');
        dataset = self.csv_processor(self.csv_datasets[0],self.csv_datasets_col_names[0]);

        print('B) ::Repairing continuous attributes in Dataset::');
        self.repair_continuous_attributes(dataset,dataset.columns);

        print('C) ::Building train/test sets::');
        self.build_learning_sets(dataset,dataset.columns[-1],1.0);

        #print('D) ::Dataset Information::');
        #self.display_data_info(dataset);


    def PRISM(self):

        print("\n:::PRISM Algorithm:::");

        prism_rule_set = [];
        
        #对每个类标label
        for label in set(self.y_train):

            print("<<<<<<<<< CURRENT LABEL: "+str(label)+">>>>>>>>>>");

            #获得类标为label的所有实例
            instances = [i for i, val in enumerate(self.y_train) if val == label];

            #创建规则rule使之能够覆盖instances中的所有实例
            while instances:
                rule = [];
                X_train_ = self.X_train.copy(deep=True);
                instances_covered = [];
                perfect_rule = False;

                print(" ******** WHILE PERFECT RULE? ********* ");
                print("\n");

                rule_precision = 0.0;
                rule_coverage = 0.0;
                
                #当前规则不是一个完美规则且还有特征可用于测试
                while perfect_rule == False and len(rule) <= self.n_features:
                    #记录当前最好的规则
                    optimal_selector = [("","")];
                    
                    #记录三个数值
                    #optimal_selector_prec[0]:当前最好的规则覆盖的正例的个数
                    #optimal_selector_prec[1]:当前最好的规则覆盖的所有实例的个数
                    #optimal_selector_prec[2]：当前最好的规则的精度
                    optimal_selector_prec = [0.0,0.0,0.0];
                    instances_covered = [];

                    print("^^^^^^^^ INSTANCES TO FIT ^^^^^^^^^");
                    print(instances);
                    print("\n");


                    print(" %%%%%%%% PREVIOUS OPT SELECTOR %%%%%%% ");
                    print(optimal_selector);
                    print(optimal_selector_prec);
                    print("\n");

                    #对所有的属性获取该属性值的atrribute
                    for attribute in X_train_.columns:
                        #获取该属性所在列
                        attr_column = X_train_.loc[:,attribute];

                        #对该属性的所有值attr_value
                        for attr_value in set(attr_column):
    
                            #获取attribute==attr_value的所有实例
                            total_attr_values_instances = attr_column[(attr_column == attr_value)].index.get_values();
                            
                            #计算覆盖的实例的个数
                            total_matches = len(total_attr_values_instances);
                            print("::::TOTALS::: size = "+str(total_matches));
                            
    
                            #计算attribute==attr_value覆盖的所有类标为label的实例的个数
                            positive_attr_values_instances = list(set(total_attr_values_instances) & set(instances));
                            positive_matches = len(positive_attr_values_instances);
                            print("::::POSITIVES::: size = "+str(positive_matches));
                           
    
                            #计算规则if attribute==attr_value then label的准确率
                            precision = (1.0 * positive_matches) / total_matches;
    
                            #计算规则if attribute==attr_value then label的覆盖率
                            coverage = (1.0 * positive_matches) / self.n_samples;
    
    
                          #保存当前最好的属性测试
                            if coverage > optimal_selector_prec[2]:
                                optimal_selector = (attribute,attr_value);
                                optimal_selector_prec[0] = positive_matches;
                                optimal_selector_prec[1] = total_matches;
                                optimal_selector_prec[2] = coverage;
                                rule_precision = precision;
                                rule_coverage = coverage;
                                instances_covered = positive_attr_values_instances;
    
                            elif coverage == optimal_selector_prec[2] and positive_matches > optimal_selector_prec[0]:
                                optimal_selector = (attribute, attr_value);
                                optimal_selector_prec[0] = positive_matches;
                                optimal_selector_prec[1] = total_matches;
                                optimal_selector_prec[2] = coverage;
                                instances_covered = positive_attr_values_instances;
                                rule_precision = precision;
                                rule_coverage = coverage;

                    print(" %%%%%%%% UPDATED OPT SELECTOR ? %%%%%%% ");
                    print(optimal_selector);
                    print(optimal_selector_prec);
                    print("\n");



                    if optimal_selector_prec[2] > 0.0 and optimal_selector_prec[2] < 1.0:

                        print(" ***** AFTER CHECK ALL ATTR-VALS PAIRS MY RULE IS NOT PERFECT BUT (PREC > 0) ***** ");

                        print(X_train_);
                        print(X_train_.index.get_values());
 
                        #在规则中添加该轮选择的属性测试
                        rule.append(optimal_selector);
                        selector = rule[-1]

                        print("FILTER SELECTOR::");
                        print(selector);

                        #删除没被覆盖的实例，删除已经被测试的属性
                        print("ACCESSING TO SELECTOR ATTR TO OBTAIN FILTER INDEXES:::");
                        filtered_rows = X_train_[(X_train_[selector[0]] != selector[1])].index.get_values();
                        print(filtered_rows);

                        print("FILTERING DATASET BY CUMULATIVE RULE OF SELECTORS::");

                        X_train_ = X_train_.drop(filtered_rows).copy(deep=True);
                        X_train_ = X_train_.drop(selector[0], 1);

                        print("IF THERE ARE NO MORE ATTRIBUTES TO COMPOSE THE RULE:::");

                        #若没有属性可用，则该规则构建完成，尝试构建新的规则
                        if len(X_train_.columns) == 0:
                            perfect_rule = True;
                            continue;

                        print(" %%%%%%%%%% X_train_ FILTERED BY CURRENT COMPOSED RULE %%%%%%%%%%%");
                        print(X_train_);
                        print("\n");


                    elif optimal_selector_prec[2] == 1.0: 

                        print(" ***** AFTER CHECK ALL ATTR-VALS PAIRS MY RULE IS PERFECT!!! ***** ");
                        rule.append(optimal_selector);
                        perfect_rule = True;
                        continue;

                    elif optimal_selector_prec[2] == 0.0:
                        input("....... UNSUAL CASE .......");




                print("^^^^^^^^ INSTANCES COVERED ^^^^^^^^^");
                print(instances_covered);
                print("\n");

                instances = list(set(instances) - set(instances_covered));

                print("^^^^^^^^ INSTANCES REMAINING ^^^^^^^^^");
                print(instances);

                rule.append(label);
                rule.append([rule_precision,rule_coverage]);

                print("++++++++ RULE FOUND +++++++++");
                metrics = rule[-1];

                print("Rule:");
                print(rule);
                print("Rule-Precision: "+str(metrics[0]));
                print("Rule-Coverage: "+str(metrics[1]));
                print("\n");

                prism_rule_set.append(rule);

        return prism_rule_set;




RBC = RuleBasedClassifiers();

RBC.data_preprocessing();
rule_set = RBC.PRISM();

print("%%%%%%%%%%%%%%%%% FINAL PRISM RULE SET %%%%%%%%%%%%%%%%%");
print("\n");
for prism_rule in rule_set:
    print(prism_rule);
