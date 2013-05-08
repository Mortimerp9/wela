package wela.classifiers

import weka.classifiers.{ Classifier => WekaClassifier }
import wela.core._

object Classifier {
  def apply[C <: WekaClassifier](cl: => C) = new Classifier(cl)
}

class Classifier[C <: WekaClassifier](cl: => C) {
  def train[L <: Attribute, AS <: List[Attribute]](dataset: AbstractDataset[L, AS])(implicit can: CanTrain[C, L, AS]): Option[TrainedClassifier[C, L, AS]] = {
    def mkCl[L1 <: Attribute](tc: C => TrainedClassifier[C, L1, AS]): Option[TrainedClassifier[C, L, AS]] = {
      val classifierInstance = cl;
      if (can.canTrain(classifierInstance, dataset)) {
        classifierInstance.buildClassifier(dataset.wekaInstances)
        Some(tc(classifierInstance).asInstanceOf[TrainedClassifier[C, L, AS]])
      } else None
    }
    dataset.problem.label match {
      case a: NominalAttribute => mkCl[NominalAttribute](cl => NominalTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a: StringAttribute => mkCl[StringAttribute](cl => StringTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a: NumericAttribute => mkCl[NumericAttribute](cl => NumericTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a => None
    }
  }
}

trait TrainedClassifier[C <: WekaClassifier, +L <: Attribute, +AS <: List[Attribute]] {
  type LV
  type DistType
  def cl: C
  def dataset: AbstractDataset[L, AS]
  def distributionForInstance(inst: Instance): DistType
  def classifyInstance(inst: Instance): Option[LV]
}

case class NumericTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers] (override val cl: C, override val dataset: AbstractDataset[NumericAttribute, AS]) extends TrainedClassifier[C, NumericAttribute, AS] {
  override type DistType = Double
  override type LV = NumericValue

  override def classifyInstance(inst: Instance): Option[NumericValue] = {
    val i = dataset.makeInstance(inst)
    val idx = cl.classifyInstance(i)
    Some(idx)
  }

  override def distributionForInstance(inst: Instance): Double = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    dist(0)
  }
}

case class StringTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers] (override val cl: C, override val dataset: AbstractDataset[StringAttribute, AS]) extends TrainedClassifier[C, StringAttribute, AS] {
  override type LV = StringValue
  override type DistType = Seq[(StringValue, Double)]

  override def classifyInstance(inst: Instance): Option[StringValue] = {
    val i = dataset.makeInstance(inst)
    val idx = cl.classifyInstance(i)
    val levels = dataset.problem.label.levels
    if (levels.size > idx) {
      Some(levels(idx toInt))
    } else None
  }

  override def distributionForInstance(inst: Instance): Seq[(StringValue, Double)] = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    val levels = dataset.problem.label.levels
    dist.take(levels.size).zipWithIndex.map {
      case (d, idx) =>
        levels(idx) -> d
    } toSeq
  }
}

case class NominalTrainedClassifier[C <: WekaClassifier, AS <: List[Attribute]] protected[classifiers] (override val cl: C, override val dataset: AbstractDataset[NominalAttribute, AS]) extends TrainedClassifier[C, NominalAttribute, AS] {
  override type LV = SymbolValue
  override type DistType = Seq[(SymbolValue, Double)]

  override def classifyInstance(inst: Instance): Option[SymbolValue] = {
    val i = dataset.makeInstance(inst)
    val idx = cl.classifyInstance(i)
    val levels = dataset.problem.label.levels
    if (levels.size > idx) {
      Some(levels(idx toInt))
    } else None
  }

  override def distributionForInstance(inst: Instance): Seq[(SymbolValue, Double)] = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    val levels = dataset.problem.label.levels
    dist.take(levels.size).zipWithIndex.map {
      case (d, idx) =>
        levels(idx) -> d
    } toSeq
  }
}
 