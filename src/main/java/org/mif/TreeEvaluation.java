package org.mif;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

public class TreeEvaluation {
  public double trainingSet;
  public double crossValidation;
  public double stratifiedSplit;
  public double percentageSplit;
  public double treeSize;
  public String treeOptions;

  public TreeEvaluation(
    double trainingSet,
    double crossValidation,
    double stratifiedSplit,
    double percentageSplit,
    double size,
    String treeOptions
  ) {
    this.trainingSet = trainingSet;
    this.crossValidation = crossValidation;
    this.stratifiedSplit = stratifiedSplit;
    this.percentageSplit = percentageSplit;
    this.treeSize = size;
    this.treeOptions = treeOptions;
  }

  // initial score is used to collect the initial set of good solutions
  public double initialScore() {
    // the initial accuracy is defined as the worst metric from all measurement types
    var accuracy = DoubleStream.of(this.trainingSet, this.crossValidation, this.stratifiedSplit, this.percentageSplit).min().orElseThrow();
    // a penalty of 0.01 is applied for each additional node in the tree
    // tress with 3 nodes or fewer are not penalized
    // the penalty should not exceed `accuracy`, so the final result is always in the range (0, 1)
    var sizePenalty = Math.max(0, Math.min(accuracy, (this.treeSize - 3) / 100));

    return accuracy - sizePenalty;
  }

  // final score is used to pick the best solutions from the set of initially picked solutions
  public double finalScore() {
    // the final accuracy is defined as the average accuracy based on all measurement types
    var accuracy = DoubleStream.of(this.trainingSet, this.crossValidation, this.stratifiedSplit, this.percentageSplit).average().orElseThrow();

    return accuracy;
  }
}
