package wela

import weka.classifiers.bayes.NaiveBayes
import wela.core.Attribute
import wela.core.NominalAttr
import weka.classifiers.trees.RandomForest
import wela.core.NumericAttr
import weka.classifiers.functions.LeastMedSq

package object classifiers {
  implicit val canTrainBayes = new CanTrain[NaiveBayes] {
    def canTrain(labelAttr: Attribute): Boolean = labelAttr match {
      case a: NominalAttr => true
      case _ => false
    }
  }
  implicit val canTrainRF = new CanTrain[RandomForest] {
    def canTrain(labelAttr: Attribute): Boolean = labelAttr match {
      case a: NominalAttr => true
      case _ => false
    }
  }
  implicit val canTrainLeastMedSq = new CanTrain[LeastMedSq] {
    def canTrain(labelAttr: Attribute): Boolean = labelAttr match {
      case a: NumericAttr => true
      case _ => false
    }
  }
}