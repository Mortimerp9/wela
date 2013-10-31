package wela.core

import weka.core.{Attribute => WekaAttribute, Instance => WekaInstance, Instances => WekaInstances}

trait AbstractDataset[+L <: Attribute, +AS <: List[Attribute]] {

  protected def instances: Seq[Instance]

  protected[wela] def problem: AbstractProblemWithAttributes[L, AS]

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

  /**
   * get the Weka Instances
   */
  protected[wela] def wekaInstances: WekaInstances = {
    instances.foreach {
      i => wekaInstanceCol.add(makeInstance(i))
    }
    wekaInstanceCol
  }
}

class Dataset[+L <: Attribute, +AS <: List[Attribute]] protected[core](override val problem: ProblemWithAttributes[L, AS], override val instances: Seq[Instance])
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

}

class MappedDataset[+L <: Attribute, +AS <: List[Attribute]] protected[core](override val problem: MappedProblemWithAttributes[L, AS], override val instances: Seq[Instance])
  extends AbstractDataset[L, AS] {

  lazy val (mappedAttributes, mappedInstances, mappedClass) = problem.mappings.foldRight[(Map[Symbol, Attribute], Seq[Instance], Attribute)]((problem.attrDefinitions, instances, problem.label)) {
    case (mapper, (prAttr, prInstances, prClass)) =>
      val mapInst = prInstances.map(mapper.mapInstance)
      val mapAttr = mapper.mapProblem(prAttr)
      val mapClass = if (mapper.attr == prClass.name) {
        mapper match {
          case d: ProblemMapping[_, _] => d.newAttr
          case d: ProblemMultiMapper[_, _] =>
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
    val (mappedAttr, mappedInstances) = problem.mappings.foldLeft[(Map[Symbol, Attribute], Seq[Instance])]((problem.attrDefinitions, Seq(inst))) {
      case ((prAttr, prInstances), mapper) =>
        val mapInst = prInstances.map(mapper.mapInstance)
        (mapper.mapProblem(prAttr), mapInst)
    }
    val wInstance = makeInstance(mappedInstances(0), mappedAttr)
    wInstance.setDataset(wekaInstanceCol)
    wInstance
  }
}
