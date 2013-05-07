package wela.core

import weka.core.{ Attribute => WekaAttribute, Instance => WekaInstance, Instances }

case class Problem(name: String, label: Option[Symbol] = None) {
  def withAttributes(attr: Attribute*): ProblemWithAttributes = {
    new ProblemWithAttributes(name, attr, label)
  }
}

class ProblemWithAttributes protected[core] (val name: String, val attrDefinitions: AttributeSet, val label: Option[Symbol] = None) {
  def withInstances(inst: Instance*): Dataset = {
    new Dataset(this, inst)
  }
  
  def labelAttribute: Option[Attribute] = label.flatMap(attrDefinitions.get(_))


}

class Dataset protected[core] (val problem: ProblemWithAttributes, val inst: Seq[Instance]) {
  protected[wela] val instances: Instances = {
    val attrs: FastVector[WekaAttribute] = problem.attrDefinitions.values.map(_.toWekaAttribute).to[FastVector]
    new Instances(problem.name, attrs, inst.size)
  }
  
  implicit class RichInstance(ist: WekaInstance) {
    def setValue[T](attr: Attribute, value: AttributeValue) {
      value match {
        case DoubleValue(dbl) => ist.setValue(attr.toWekaAttribute, dbl)
        case StringValue(str) => ist.setValue(attr.toWekaAttribute, str)
      }
    }
  }

  protected[wela] def makeInstance(inst: Instance): WekaInstance = {
    val wInstance = new WekaInstance(problem.attrDefinitions.size)
    inst.foreach {
      case (attr, value) =>
        val k = problem.attrDefinitions.get(attr)
        if (k.isDefined) {
          wInstance.setValue(k.get, value)
        }
    }
    wInstance.setDataset(instances)
    wInstance
  }
  
  {
    val classAttr = problem.labelAttribute
    if (classAttr.isDefined) instances.setClass(classAttr.get)
  }

  inst.foreach { i => instances.add(makeInstance(i)) }
}