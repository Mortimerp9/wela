package wela.core

import weka.core.{ Attribute => WekaAttribute, Instance => WekaInstance, Instances => WekaInstances }
import scala.collection.mutable.HashMap

case class Problem[+L <: Attribute](name: String, label: L) {
  def withAttributes[A <: Attribute](attr: A*): ProblemWithAttributes[L, List[A]] = {
    new ProblemWithAttributes(name, attr.toList, label)
  }
}

class ProblemWithAttributes[+L <: Attribute, +AS <: List[Attribute]] protected[core] (val name: String, val attrs: AS, val label: L) {
  val attrDefinitions = (label :: attrs).map(a => a.name -> a).toMap
  def withInstances(inst: Instance*): AbstractDataset[L, AS] = {
    new Dataset(this, inst)
  }
}

trait AbstractDataset[+L <: Attribute, +AS <: List[Attribute]] {

  protected def instances: Seq[Instance]
  protected[wela] def problem: ProblemWithAttributes[L, AS]
  protected def wekaInstanceCol: WekaInstances

  /**
   * create an instance within this problem definition. This doesn't add anything to the set of wrapped instances
   */
  protected[wela] def makeInstance(inst: Instance): WekaInstance

  protected def makeInstance(inst: Instance, attrDefinitions: Map[Symbol, Attribute]): WekaInstance = {
    val wInstance = new WekaInstance(attrDefinitions.size)
    inst.foreach {
      case (attr, value) =>
        val k = attrDefinitions.get(attr)
        if (k.isDefined) {
          val attrDef = k.get
          (attrDef, value) match {
            case (a: NumericAttribute, v: NumericValue) => wInstance.setValue(a.toWekaAttribute, v)
            case (a: StringAttribute, v: StringValue) => wInstance.setValue(a.toWekaAttribute, v)
            case (a: NominalAttribute, v: SymbolValue) => wInstance.setValue(a.toWekaAttribute, v.name)
            case _ => throw new IllegalArgumentException("instance doesn't correspond to attribute definition")
          }
        }
    }
    wInstance
  }

  def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedDataset[L, List[Attribute]]
  def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedDataset[L, List[Attribute]]
  def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedDataset[L, List[Attribute]]

  /**
   * get the Weka Instances
   */
  protected[wela] def wekaInstances: WekaInstances = {
    instances.foreach { i => wekaInstanceCol.add(makeInstance(i)) }
    wekaInstanceCol
  }
}

class Dataset[+L <: Attribute, +AS <: List[Attribute]] protected[core] (override val problem: ProblemWithAttributes[L, AS], override val instances: Seq[Instance])
  extends AbstractDataset[L, AS] {

  protected val wekaInstanceCol: WekaInstances = {
    val attrs: FastVector[WekaAttribute] = problem.attrDefinitions.values.map(_.toWekaAttribute).to[FastVector]
    val in = new WekaInstances(problem.name, attrs, instances.size)
    in.setClass(problem.label)
    in
  }

  /**
   * create an instance within this problem definition. This doesn't add anything to the set of wrapped instances
   */
  protected[wela] def makeInstance(inst: Instance): WekaInstance = {
    val wInstance = makeInstance(inst, problem.attrDefinitions)
    wInstance.setDataset(wekaInstanceCol)
    wInstance
  }

  override def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetMapping(attr, newAttr, f)(comp)
    new MappedDataset(problem, instances, List(mapper))
  }

  override def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedDataset[L, List[Attribute]] = {
    if (!newLabel.isDefined) require(attr != problem.label.name, "you can't replace the training label/class unless you specify a replacement")
    val mapper = new DatasetMultiMapper(attr, newAttributes, f, newLabel)
    new MappedDataset(problem, instances, List(mapper))
  }

  def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetDynMapper(attr, newAttrPrefix, f)
    new MappedDataset(problem, instances, List(mapper))
  }

}

private trait Mapper {
  def attr: Symbol
  def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute]
  def mapInstance(instance: Instance): Instance
}

private trait MultiMapper extends Mapper {
  protected type ValType <: AttributeValue
  def mapValue(value: AttributeValue): Seq[(Symbol, ValType)]

  def mapInstance(instance: Instance): Instance = {
    val instOpt = instance.get(attr)
    if (instOpt.isDefined) {
      val map = instOpt.get
      val newVals = mapValue(map)
      val keepVal = instance.filter {
        case (k, v) => k != attr
      }
      keepVal ++ newVals
    } else {
      //the attribute doesn't exist, just ignore the map
      instance
    }
  }

}

private class DatasetDynMapper(override val attr: Symbol, val newAttrPrefix: String, f: AttributeValue => Seq[String])
  extends MultiMapper {
  override protected type ValType = NumericValue

  val newAttributes = scala.collection.mutable.HashMap[Symbol, Attribute]()

  override def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute] = {
    if (newAttributes.size == 0) {
      throw new IllegalStateException("you can only get the new attributes after mapping on all the instances")
      attrDefinitions
    } else {
      newAttributes.toMap
    }
  }

  override def mapValue(value: AttributeValue): Seq[(Symbol, NumericValue)] = {
    (f(value).groupBy(x => x).map {
      case (name, occurence) =>
        val attrName = Symbol(newAttrPrefix + "_" + name)
        val attr = newAttributes.get(attrName)
        if (!attr.isDefined) {
          val newAttr = NumericAttribute(attrName)
          newAttributes += attrName -> newAttr
        }
        attrName -> dblToAV(occurence.size)
    }) toSeq
  }

}

private class DatasetMultiMapper[+VT <: AttributeValue, +AD <: Attribute](override val attr: Symbol, val newAttributes: Seq[AD], f: AttributeValue => Seq[(Symbol, VT)], val newLabel: Option[Symbol] = None)
  extends MultiMapper {

  override protected type ValType = AttributeValue

  def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute] = {
    val keep = attrDefinitions.filter {
      case (k, v) => k != attr
    }
    keep ++ newAttributes.map(a => a.name -> a)
  }

  override def mapValue(value: AttributeValue): Seq[(Symbol, ValType)] = f(value)

}

private class DatasetMapping[+AV <: AttributeValue, +AD <: Attribute](override val attr: Symbol, val newAttr: AD, f: AttributeValue => AV)(implicit comp: Compatible[AD, AV])
  extends Mapper {

  def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute] = {
    val keep = attrDefinitions.filter {
      case (k, v) => k != attr
    }
    keep + (newAttr.name -> newAttr)
  }

  def mapInstance(instance: Instance): Instance = {
    val instOpt = instance.get(attr)
    if (instOpt.isDefined) {
      val map = instOpt.get
      val newVal = f(map)
      val keepVal = instance.filter {
        case (k, v) => k != attr
      }
      keepVal + (newAttr.name -> newVal)
    } else {
      instance
    }
  }

}

class MappedDataset[+L <: Attribute, +AS <: List[Attribute]] protected[core] (override val problem: ProblemWithAttributes[L, AS], override val instances: Seq[Instance], mappings: List[Mapper])
  extends AbstractDataset[L, AS] {

  lazy val (mappedAttributes, mappedInstances, mappedClass) = mappings.foldRight[(Map[Symbol, Attribute], Seq[Instance], Attribute)]((problem.attrDefinitions, instances, problem.label)) {
    case (mapper, (prAttr, prInstances, prClass)) =>
      val mapInst = prInstances.map(mapper.mapInstance)
      val mapAttr = mapper.mapProblem(prAttr)
      val mapClass = if (mapper.attr == prClass.name) {
        mapper match {
          case d: DatasetMapping[_, _] => d.newAttr
          case d: DatasetMultiMapper[_, _] =>
            require(d.newLabel.isDefined, "if you are mapping the instance label, then you need to provide a replacement")
            if (d.newLabel.isDefined) {
              val mapClass = mapAttr.get(d.newLabel.get)
              require(mapClass.isDefined, "the provided new label is not produced by the mapper")
              mapClass.get
            } else {
              prClass
            }
          case _ => prClass
        }
      } else {
        prClass
      }
      (mapAttr, mapInst, mapClass)
  }

  
  
  override protected val wekaInstanceCol: WekaInstances = {
    val attrs: FastVector[WekaAttribute] = mappedAttributes.values.map(_.toWekaAttribute).to[FastVector]
    val in = new WekaInstances(problem.name, attrs, mappedInstances.size)
    in.setClass(mappedClass)
    in
  }

  override protected[wela] def makeInstance(inst: Instance): WekaInstance = {
    val (mappedAttr, mappedInstances) = mappings.foldLeft[(Map[Symbol, Attribute], Seq[Instance])]((problem.attrDefinitions, Seq(inst))) {
      case ((prAttr, prInstances), mapper) =>
        val mapInst = prInstances.map(mapper.mapInstance)
        (mapper.mapProblem(prAttr), mapInst)
    }
    val wInstance = makeInstance(mappedInstances(0), mappedAttr)
    wInstance.setDataset(wekaInstanceCol)
    wInstance
  }

  override def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetMapping(attr, newAttr, f)(comp)
    new MappedDataset(problem, instances, mapper :: mappings)
  }

  override def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedDataset[L, List[Attribute]] = {
    if (!newLabel.isDefined) require(attr != problem.label.name)
    val mapper = new DatasetMultiMapper(attr, newAttributes, f, newLabel)
    new MappedDataset(problem, instances, mapper :: mappings)
  }

  override def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetDynMapper(attr, newAttrPrefix, f)
    new MappedDataset(problem, instances, mapper :: mappings)
  }

}