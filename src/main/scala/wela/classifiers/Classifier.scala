package wela.classifiers

import weka.classifiers.{Classifier => WekaClassifier}
import wela.core._
import scalaz._
import Scalaz._
import java.io._
import wela.core.NumericAttribute
import wela.core.NominalAttribute
import wela.core.StringAttribute

object Classifier {
  def apply[C <: WekaClassifier](cl: => C) = new Classifier(cl)
}

class Classifier[C <: WekaClassifier](cl: => C) {
  def train[L <: Attribute, AS <: List[Attribute]](dataset: AbstractDataset[L, AS])(implicit can: CanTrain[C, L, AS]): ValidationNel[String, TrainedClassifier[C, L, AS]] = {
    def mkCl[L1 <: Attribute](tc: C => TrainedClassifier[C, L1, AS]): ValidationNel[String, TrainedClassifier[C, L, AS]] = {
      val classifierInstance = cl;
      can.canTrain(classifierInstance, dataset).map {
        whatever =>
          classifierInstance.buildClassifier(dataset.wekaInstances)
          tc(classifierInstance).asInstanceOf[TrainedClassifier[C, L, AS]]
      }
    }
    dataset.problem.label match {
      case a: NominalAttribute => mkCl[NominalAttribute](cl => NominalTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a: StringAttribute => mkCl[StringAttribute](cl => StringTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a: NumericAttribute => mkCl[NumericAttribute](cl => NumericTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a => s"attribute ${a} is not supported".fail.toValidationNel
    }
  }
}

trait TrainedClassifier[C <: WekaClassifier, +L <: Attribute, +AS <: List[Attribute]] {
  type DistType

  def cl: C

  def dataset: AbstractDataset[L, AS]

  def distributionForInstance(inst: Instance): DistType

  def classifyInstance(inst: Instance): Validation[String, L#ValType] = {
    val i = dataset.makeInstance(inst)
    val idx = cl.classifyInstance(i)
    dataset.problem.label.resolve(idx) match {
      case Some(v) => v.success[String]
      case _ => s"erroneous prediction ${idx}".fail[L#ValType]
    }
  }

}

sealed trait NominalTC[C <: WekaClassifier, +L <: NominalAttr, +AS <: List[Attribute]] extends TrainedClassifier[C, L, AS] {
  type DistType <: Seq[(L#ValType, Double)]

  override def distributionForInstance(inst: Instance): DistType = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    val levels = dataset.problem.label.levels
    dist.take(levels.size).zipWithIndex.map {
      case (d, idx) =>
        levels(idx) -> d
    }.toSeq.asInstanceOf[DistType]
  }
}

case class NumericTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers](override val cl: C, override val dataset: AbstractDataset[NumericAttribute, AS]) extends TrainedClassifier[C, NumericAttribute, AS] {
  override type DistType = Double

  override def distributionForInstance(inst: Instance): Double = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    dist(0)
  }
}

case class StringTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers](override val cl: C, override val dataset: AbstractDataset[StringAttribute, AS]) extends NominalTC[C, StringAttribute, AS] {
  override type DistType = Seq[(StringValue, Double)]
}

case class NominalTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers](override val cl: C, override val dataset: AbstractDataset[NominalAttribute, AS]) extends NominalTC[C, NominalAttribute, AS] {
  override type DistType = Seq[(SymbolValue, Double)]
}
 