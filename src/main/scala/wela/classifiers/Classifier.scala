package wela.classifiers

import weka.classifiers.{ Classifier => WekaClassifier }
import wela.core._

trait CanTrain[T] {
  def canTrain(labelAttr: Attribute): Boolean
}

object Classifier {
  def apply[C <: WekaClassifier](cl: => C)(implicit can: CanTrain[C]) = new Classifier(cl)
}

class Classifier[C <: WekaClassifier](cl: => C)(implicit can: CanTrain[C]) {
  def train(dataset: Dataset): Option[TrainedClassifier] = {
    def mkCl = {
      val classifierInstance = cl;
      classifierInstance.buildClassifier(dataset.instances)
      classifierInstance
    }
    dataset.problem.labelAttribute.flatMap {
      case a: NominalAttr if can.canTrain(a) => Some(NominalTrainedClassifier(mkCl, dataset))
      case a: NumericAttr if can.canTrain(a) => Some(NumericTrainedClassifier(mkCl, dataset))
      case _ => None
    }
  }
}

trait TrainedClassifier {
  type ProbType
  def cl: WekaClassifier
  def dataset: Dataset
  def classifyInstance(inst: Instance): Option[AttributeValue] = {
    dataset.problem.labelAttribute.flatMap { attr =>
      val i = dataset.makeInstance(inst)
      val idx = cl.classifyInstance(i)
      attr match {
        case a: NominalAttr =>
          if (a.values.size > idx) {
            Some(a.values(idx.toInt))
          } else None
        case a: NumericAttr => Some(idx)
        case _ =>
          None
      }
    }
  }
}

case class NumericTrainedClassifier protected[classifiers] (override val cl: WekaClassifier, override val dataset: Dataset) extends TrainedClassifier {
  type ProbType = Double

  def distributionForInstance(inst: Instance): Option[Double] = {
    dataset.problem.labelAttribute.flatMap { attr =>
      val i = dataset.makeInstance(inst)
      val dist = cl.distributionForInstance(i)
      attr match {
        case a: NumericAttr if dist.size > 0 => Some(dist(0))
        case _ => None
      }
    }
  }
}

case class NominalTrainedClassifier protected[classifiers] (override val cl: WekaClassifier, override val dataset: Dataset) extends TrainedClassifier {
  type ProbType = Seq[(Symbol, Double)]

  def distributionForInstance(inst: Instance): Seq[(Symbol, Double)] = {
    dataset.problem.labelAttribute.map { attr =>
      val i = dataset.makeInstance(inst)
      val dist = cl.distributionForInstance(i)
      attr match {
        case a: NominalAttr =>
          dist.take(a.values.size).zipWithIndex.map {
            case (d, idx) =>
              Symbol(a.value(idx)) -> d
          } toSeq
        case _ => Nil
      }
    } getOrElse (Nil)
  }
}
 