package org.mif;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.PartitionGenerator;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.*;
import java.util.*;

public class TreeEvaluator {
  public static void evaluateTrees(String input, int randomSeed) throws Exception {
    // create randomness source
    var rand = new Random(randomSeed);

    // read data from arff file
    var data = new Instances(new BufferedReader(new FileReader(input)));
    data.setClassIndex(data.numAttributes() - 1);

    /// create randomized copy of the data
    var randomizedData = new Instances(data);
    randomizedData.randomize(rand);

    // create training and test data with randomized percentage split (mimics weka)
    int trainSize = (int) Math.round(randomizedData.numInstances() * 0.66);
    int testSize = randomizedData.numInstances() - trainSize;
    Instances splitTrainingData = new Instances(randomizedData, 0, trainSize);
    Instances splitTestData = new Instances(randomizedData, trainSize, testSize);

    // create training data with stratified split
    var stratifiedTrainingDataFilter = new StratifiedRemoveFolds();
    var trainingDataFilterOptions = Utils.splitOptions("-V -N 3 -F 1 -S " + randomSeed);
    stratifiedTrainingDataFilter.setOptions(trainingDataFilterOptions);
    stratifiedTrainingDataFilter.setInputFormat(randomizedData);
    Instances stratifiedTrainingData = Filter.useFilter(data, stratifiedTrainingDataFilter);

    // create test data with stratified split
    var stratifiedTestDataFilter = new StratifiedRemoveFolds();
    var testFilterDataOptions = Utils.splitOptions("-N 3 -F 1 -S " + randomSeed);
    stratifiedTestDataFilter.setOptions(testFilterDataOptions);
    stratifiedTestDataFilter.setInputFormat(randomizedData);
    Instances stratifiedTestData = Filter.useFilter(data, stratifiedTestDataFilter);

    // create ArrayLists to collect results
    var resultsJ48 = new ArrayList<TreeEvaluation>();
    var resultsRandomTree = new ArrayList<TreeEvaluation>();
    var resultsRepTree = new ArrayList<TreeEvaluation>();

    for (var confidence = 0.01; confidence <= 0.5; confidence += 0.01) {
      for (var minObjects = 1; minObjects <= 200; minObjects += 1) {
        // create J48 options
        var optionsString = "-C " + confidence + " -M " + minObjects;
        var treeOptions = Utils.splitOptions(optionsString);

        // create base J48
        var baseTree = new J48();
        baseTree.setOptions(treeOptions);

        TreeEvaluation treeEvaluation = getTreeEvaluation(rand, data, splitTrainingData, splitTestData, stratifiedTrainingData, stratifiedTestData, optionsString, baseTree, "J48");

        resultsJ48.add(treeEvaluation);
      }
    }

    for (var numFolds = 2; numFolds <= 50; numFolds += 1) {
      for (var minNum = 2; minNum <= 200; minNum += 1) {
        // create RepTree options
        var optionsString = "-N " + numFolds + " -M " + minNum + " -S " + randomSeed + " -V 0.001 -L -1 -I 0.0";
        var treeOptions = Utils.splitOptions(optionsString);

        // create base RepTree
        var baseTree = new REPTree();
        baseTree.setOptions(treeOptions);

        TreeEvaluation treeEvaluation = getTreeEvaluation(rand, data, splitTrainingData, splitTestData, stratifiedTrainingData, stratifiedTestData, optionsString, baseTree, "REPTree");

        resultsRepTree.add(treeEvaluation);
      }
    }


    for (var kValue = 0; kValue <= 50; kValue += 1) {
      for (var minNum = 1; minNum <= 200; minNum += 1) {
        // create RandomTree options
        var optionsString = "-K " + kValue + " -M " + minNum + " -S " + randomSeed + " -V 0.001";
        var treeOptions = Utils.splitOptions(optionsString);

        // create base RandomTree
        var baseTree = new RandomTree();
        baseTree.setOptions(treeOptions);

        TreeEvaluation treeEvaluation = getTreeEvaluation(rand, data, splitTrainingData, splitTestData, stratifiedTrainingData, stratifiedTestData, optionsString, baseTree, "RandomTree");

        resultsRandomTree.add(treeEvaluation);
      }
    }

    // sort the results by score
    resultsJ48.sort(Comparator.comparing(TreeEvaluation::unifiedScore));
    resultsRepTree.sort(Comparator.comparing(TreeEvaluation::unifiedScore));
    resultsRandomTree.sort(Comparator.comparing(TreeEvaluation::unifiedScore));

    // the last element has the best score
    var bestJ48 = resultsJ48.get(resultsJ48.size() - 1);
    var bestRepTree = resultsRepTree.get(resultsRepTree.size() - 1);
    var bestRandomTree = resultsRandomTree.get(resultsRandomTree.size() - 1);

    System.out.println("random seed: " + randomSeed);
    System.out.println("best J48/ score: " + bestJ48.unifiedScore() + " options: " + bestJ48.treeOptions);
    System.out.println("best REPTree/ score: " + bestRepTree.unifiedScore() + " options: " + bestRepTree.treeOptions);
    System.out.println("best RandomTree/ score: " + bestRandomTree.unifiedScore() + " options: " + bestRandomTree.treeOptions);
  }

  private static TreeEvaluation getTreeEvaluation(Random rand, Instances data, Instances splitTrainingData, Instances splitTestData, Instances stratifiedTrainingData, Instances stratifiedTestData, String optionsString, Classifier baseTree, String treeType) throws Exception {
    // create tree for training set
    var trainingSetTree = AbstractClassifier.makeCopy(baseTree);
    trainingSetTree.buildClassifier(data);
    // evaluate tree using training set
    var trainingSetEval = new Evaluation(data);
    trainingSetEval.evaluateModel(trainingSetTree, data);
    var trainingSetScore = trainingSetEval.pctCorrect() / 100;

    // create tree for cross-validation
    var crossValidationTree = AbstractClassifier.makeCopy(baseTree);
    // evaluate tree using cross-validation
    var crossValidationEval = new Evaluation(data);
    crossValidationEval.crossValidateModel(crossValidationTree, data, 10, rand);
    var crossValidationScore = crossValidationEval.pctCorrect() / 100;

    // create tree for percentage split
    var percentageSplitTree = AbstractClassifier.makeCopy(baseTree);
    percentageSplitTree.buildClassifier(splitTrainingData);
    // evaluate tree for percentage split
    var percentageSplitEval = new Evaluation(splitTrainingData);
    percentageSplitEval.evaluateModel(percentageSplitTree, splitTestData);
    var percentageSplitScore = percentageSplitEval.pctCorrect() / 100;

    // create tree for stratified split
    var stratifiedSplitTree = AbstractClassifier.makeCopy(baseTree);
    stratifiedSplitTree.buildClassifier(stratifiedTrainingData);
    // evaluate tree for percentage split
    var stratifiedSplitEval = new Evaluation(stratifiedTrainingData);
    stratifiedSplitEval.evaluateModel(stratifiedSplitTree, stratifiedTestData);
    var stratifiedSplitScore = stratifiedSplitEval.pctCorrect() / 100;

    var evaluation =  new TreeEvaluation(
        trainingSetScore, crossValidationScore, stratifiedSplitScore, percentageSplitScore, ((PartitionGenerator) trainingSetTree).numElements(), optionsString
    );

//    System.out.println(treeType + " " + optionsString);
//    System.out.println("unified score " + evaluation.unifiedScore());

    return evaluation;
  }
}
