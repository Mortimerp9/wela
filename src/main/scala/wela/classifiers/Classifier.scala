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
      case a: NominalAttr => mkCl[a.type](cl => NominalTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a: NumericAttr => mkCl[a.type](cl => NumericTrainedClassifier(cl, dataset.asInstanceOf[AbstractDataset[a.type, AS]]))
      case a => None
    }
  }
}

trait TrainedClassifier[C <: WekaClassifier, +L <: Attribute, +AS <: List[Attribute]] {
  type ProbType
  def cl: C
  def dataset: AbstractDataset[L, AS]
  def classifyInstance(inst: Instance): Option[AttributeValue] = {
    val i = dataset.makeInstance(inst)
    val idx = cl.classifyInstance(i)
    dataset.problem.label match {
      case a: NominalAttr =>
        if (a.levels.size > idx) {
          Some(a.levels(idx.toInt))
        } else None
      case a: NumericAttr => Some(idx)
      case _ =>
        None
    }
  }
}

case class NumericTrainedClassifier[C <: WekaClassifier, L <: NumericAttr, AS <: List[Attribute]] protected[classifiers] (override val cl: C, override val dataset: AbstractDataset[L, AS]) extends TrainedClassifier[C, L, AS] {
  type ProbType = Double

  def distributionForInstance(inst: Instance): Option[Double] = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    dataset.problem.label match {
      case a: NumericAttr if dist.size > 0 => Some(dist(0))
      case _ => None
    }

  }
}

case class NominalTrainedClassifier[C <: WekaClassifier, L <: NominalAttr, AS <: List[Attribute]] protected[classifiers] (override val cl: C, override val dataset: AbstractDataset[L, AS]) extends TrainedClassifier[C, L, AS] {
  type ProbType = Seq[(Symbol, Double)]

  def distributionForInstance(inst: Instance): Seq[(Symbol, Double)] = {
    val i = dataset.makeInstance(inst)
    val dist = cl.distributionForInstance(i)
    dataset.problem.label match {
      case a: NominalAttr =>
        dist.take(a.levels.size).zipWithIndex.map {
          case (d, idx) =>
            Symbol(a.value(idx)) -> d
        } toSeq
      case _ => Nil
    }
  }
}
 