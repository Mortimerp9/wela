package wela

import weka.classifiers.{Classifier => WekaClassifier}
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.LeastMedSq
import weka.classifiers.trees.RandomForest
import wela.core.AbstractDataset
import wela.core.Attribute
import wela.core.NominalAttr
import wela.core.NumericAttr

package object classifiers {

  trait CanTrain[+T <: WekaClassifier, +L <: Attribute, +AS <: List[Attribute]] {
    def canTrain(cl: WekaClassifier, att: List[Attribute]): Boolean = att.foldLeft(true)((bool, a) => bool && cl.getCapabilities().test(a.toWekaAttribute))
    def canTrain(cl: WekaClassifier, data: AbstractDataset[Attribute, List[Attribute]]): Boolean =
      canTrain(cl, data.problem.label) &&
        canTrain(cl, data.problem.label :: data.problem.attrs.toList) &&
        cl.getCapabilities().test(data.wekaInstances)
    def canTrain(cl: WekaClassifier, label: Attribute): Boolean =cl.getCapabilities().test(label)
  }

  trait CanTrainBayes[T <: NominalAttr, AS <: List[Attribute]] extends CanTrain[NaiveBayes, T, AS]
  implicit def canTrainBayes[T <: NominalAttr, AS <: List[Attribute]] = new CanTrainBayes[T, AS] {}

  trait CanTrainRF[T <: NominalAttr, AS <: List[NumericAttr]] extends CanTrain[RandomForest, T, AS]
  implicit def canTrainRF[T <: NominalAttr, AS <: List[NumericAttr]] = new CanTrainRF[T, AS] {}

  trait CanTrainLSMSQ[T <: Attribute, AS <: List[Attribute]] extends CanTrain[LeastMedSq, T, AS]
  implicit def canTrainLeastMedSq[T <: Attribute, AS <: List[Attribute]] = new CanTrainLSMSQ[T, AS] {}
}