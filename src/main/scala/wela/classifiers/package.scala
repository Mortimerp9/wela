package wela

import weka.classifiers.{ Classifier => WekaClassifier }
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.LeastMedSq
import weka.classifiers.trees.RandomForest
import wela.core.AbstractDataset
import wela.core.Attribute
import wela.core.NominalAttr
import wela.core.NumericAttribute
import scalaz._
import Scalaz._
import wela.core.MappedDataset
import wela.core.NumericAttribute

package object classifiers {

  private implicit class BoolToValidation(test: Boolean) {
    def toValidation[T](cl: T, fail: String): Validation[String, T] = if (test) {
      cl.success
    } else {
      fail.fail[T]
    }
  }

  trait CanTrain[T <: WekaClassifier, L <: Attribute, AS <: List[Attribute]] {
    def canTrain(cl: T, att: List[Attribute]): ValidationNel[String, T] = att.foldLeft(cl.success[String].toValidationNel) { (valid, a) =>
      cl.getCapabilities().test(a.toWekaAttribute).toValidation(cl, s"${cl.getClass} does not support attributes of type ${a}").toValidationNel
    }
    def canTrain(cl: T, data: AbstractDataset[L, AS]): ValidationNel[String, T] = {
      val canTrainLabel = canTrain(cl, data.problem.label).toValidationNel
      val canTrainAttr = data match {
        case m: MappedDataset[L, AS] => canTrain(cl, m.mappedAttributes.values.toList)
        case _ =>
          canTrain(cl, data.problem.attrs)
      }
      val canTrainInstances =
        cl.getCapabilities().test(data.wekaInstances).toValidation(cl, s"${cl.getClass} does not support these instances").toValidationNel
      (canTrainLabel |@| canTrainAttr |@| canTrainInstances) {
        case (_, _, _) => cl
      }
    }
    def canTrain(cl: T, label: L): Validation[String, T] = cl.getCapabilities().test(label).toValidation(cl, s"${cl.getClass} does not support ${label} as a classification label")
  }

  trait CanTrainBayes[T <: NominalAttr, AS <: List[Attribute]] extends CanTrain[NaiveBayes, T, AS]
  implicit def canTrainBayes[T <: NominalAttr, AS <: List[Attribute]] = new CanTrainBayes[T, AS] {}

  trait CanTrainRF[T <: NominalAttr, AS <: List[NumericAttribute]] extends CanTrain[RandomForest, T, AS]
  implicit def canTrainRF[T <: NominalAttr, AS <: List[NumericAttribute]] = new CanTrainRF[T, AS] {}

  trait CanTrainLSMSQ[T <: Attribute, AS <: List[Attribute]] extends CanTrain[LeastMedSq, T, AS]
  implicit def canTrainLeastMedSq[T <: Attribute, AS <: List[Attribute]] = new CanTrainLSMSQ[T, AS] {}
}