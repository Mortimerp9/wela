package wela.core

import weka.core.{ Attribute => WekaAttribute, Instance => WekaInstance, Instances => WekaInstances }

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

  /**
   * just to help set values of instances
   */
  protected implicit class RichInstance(ist: WekaInstance) {
    def setValue[AV <: AttributeValue, A <: Attribute](attr: A, value: AV)(implicit compatible: ConformType[AV, A]) {
      value match {
        case NumericValue(dbl) => ist.setValue(attr.toWekaAttribute, dbl)
        case NominalValue(str) => ist.setValue(attr.toWekaAttribute, str.name)
      }
    }
  }

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
          require(ConformType(value, attrDef), s"instance not conform to the definitions of the dataset ${problem.name}; ${value}; ${attrDef}")
          //we do not know if the value is compatible with the attribute definition, so we need to do a runtime check 
          val conformAll = new ConformType[AttributeValue, Attribute] {}
          wInstance.setValue(k.get, value)(conformAll)
        }
    }
    wInstance
  }
  
  def withMapping[VT <: AttributeValue, AD <: Attribute](attr: Symbol, a: AD)(f: AttributeValue => VT)(implicit conform: ConformType[VT, AD]): MappedDataset[L, List[Attribute]]
  
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
  


  override def withMapping[VT <: AttributeValue, AD <: Attribute](attr: Symbol, a: AD)(f: AttributeValue => VT)(implicit conform: ConformType[VT, AD]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetMapping(attr, a, f)(conform)
    new MappedDataset(problem, instances, List(mapper))
  }

}

private class DatasetMapping[+VT <: AttributeValue, +AD <: Attribute](val attr: Symbol, val a: AD, f: AttributeValue => VT)(implicit conform: ConformType[VT, AD]) {

  def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute] = {
    val keep = attrDefinitions.filter {
      case (k, v) => k != attr
    }
    keep + (a.name -> a)
  }

  def mapInstance(instance: Instance): Instance = {
    val instOpt = instance.get(attr)
    if (instOpt.isDefined) {
      val map = instOpt.get
      instance.updated(attr, f(map))
    } else {
      instance
    }
  }

}

class MappedDataset[+L <: Attribute, +AS <: List[Attribute]] protected[core] (override val problem: ProblemWithAttributes[L, AS], override val instances: Seq[Instance], mappings: List[DatasetMapping[AttributeValue, Attribute]])
  extends AbstractDataset[L, AS] {

  override protected val wekaInstanceCol: WekaInstances = {
    val (mappedAttr, mappedInstances, mappedClass) = mappings.foldLeft[(Map[Symbol, Attribute], Seq[Instance], Option[Attribute])]((problem.attrDefinitions, instances, None)) {
      case ((prAttr, prInstances, prClass), mapper) =>
        val mapClass = if (prClass.isDefined) {
          prClass
        } else if (mapper.attr == problem.label.name) {
          Some(mapper.a)
        } else {
          None
        }
        (mapper.mapProblem(prAttr), prInstances.map(mapper.mapInstance), mapClass)
    }
    val attrs: FastVector[WekaAttribute] = mappedAttr.values.map(_.toWekaAttribute).to[FastVector]
    val in = new WekaInstances(problem.name, attrs, mappedInstances.size)
    val newLabel = mappedClass.getOrElse(problem.label)
    in.setClass(newLabel)
    in
  }

  override protected[wela] def makeInstance(inst: Instance): WekaInstance = {
    val (mappedAttr, mappedInstances) = mappings.foldLeft[(Map[Symbol, Attribute], Seq[Instance])]((problem.attrDefinitions, Seq(inst))) {
      case ((prAttr, prInstances), mapper) =>
        (mapper.mapProblem(prAttr), prInstances.map(mapper.mapInstance))
    }
    val wInstance = makeInstance(mappedInstances(0), mappedAttr)
    wInstance.setDataset(wekaInstanceCol)
    wInstance
  }

  override def withMapping[VT <: AttributeValue, AD <: Attribute](attr: Symbol, a: AD)(f: AttributeValue => VT)(implicit conform: ConformType[VT, AD]): MappedDataset[L, List[Attribute]] = {
    val mapper = new DatasetMapping(attr, a, f)(conform)
    new MappedDataset(problem, instances, mapper :: mappings)
  }

}