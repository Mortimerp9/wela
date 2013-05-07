package wela.core

import weka.core.{ Attribute => WekaAttribute, Instance => WekaInstance, Instances }

case class Problem[+L <: Attribute](name: String, label: L) {
  def withAttributes[A <: Attribute](attr: A*): ProblemWithAttributes[L, List[A]] = {
    new ProblemWithAttributes(name, attr.toList, label)
  }
}

class ProblemWithAttributes[+L <: Attribute, +AS <: List[Attribute]] protected[core] (val name: String, val attrs: AS, val label: L) {
  val attrDefinitions = (label :: attrs).map(a => a.name -> a).toMap
  def withInstances(inst: Instance*): Dataset[L, AS] = {
    new Dataset(this, inst)
  }
}

class Dataset[+L <: Attribute, +AS <: List[Attribute]] protected[core] (val problem: ProblemWithAttributes[L, AS], val inst: Seq[Instance]) {

  /**
   * get the Weka Instances
   */
  protected[wela] val instances: Instances = {
    val attrs: FastVector[WekaAttribute] = problem.attrDefinitions.values.map(_.toWekaAttribute).to[FastVector]
    new Instances(problem.name, attrs, inst.size)
  }

  /**
   * just to help set values of instances
   */
  private implicit class RichInstance(ist: WekaInstance) {
    def setValue[AV <: AttributeValue, A <: Attribute](attr: A, value: AV)(implicit compatible: ConformType[AV, A]) {
      value match {
        case NumericValue(dbl) => ist.setValue(attr.toWekaAttribute, dbl)
        case NominalValue(str) => ist.setValue(attr.toWekaAttribute, str.name)
      }
    }
  }

  /**
   * create an instance within this problem definition. This doesn't add anything to the set of wrapped instances
   */
  protected[wela] def makeInstance(inst: Instance): WekaInstance = {
    val wInstance = new WekaInstance(problem.attrDefinitions.size)
    inst.foreach {
      case (attr, value) =>
        val k = problem.attrDefinitions.get(attr)
        if (k.isDefined) {
          val attrDef = k.get
          require(ConformType(value, attrDef), s"instance not conform to the definitions of the dataset ${problem.name}")
          //we do not know if the value is compatible with the attribute definition, so we need to do a runtime check 
          val conformAll = new ConformType[AttributeValue, Attribute] {}
          wInstance.setValue(k.get, value)(conformAll)
        }
    }
    wInstance.setDataset(instances)
    wInstance
  }



  /////////////// init
  instances.setClass(problem.label)
  inst.foreach { i => instances.add(makeInstance(i)) }
  /////////////

}