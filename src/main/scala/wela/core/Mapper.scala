package wela.core

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

private class ProblemDynMapper(override val attr: Symbol, val newAttrPrefix: String, f: AttributeValue => Seq[String])
  extends MultiMapper {
  override protected type ValType = NumericValue

  val newAttributes = scala.collection.mutable.HashMap[Symbol, Attribute]()

  override def mapProblem(attrDefinitions: Map[Symbol, Attribute]): Map[Symbol, Attribute] = {
    if (newAttributes.size == 0) {
      throw new IllegalStateException("you can only get the new attributes after mapping on all the instances")
      attrDefinitions
    } else {
      val keep = attrDefinitions.filter {
        case (k, v) => k != attr
      }
      keep ++ newAttributes.toMap
    }
  }

  override def mapValue(value: AttributeValue): Seq[(Symbol, NumericValue)] = {
    f(value).groupBy(x => x).map {
      case (name, occurence) =>
        val attrName = Symbol(newAttrPrefix + "_" + name)
        val attr = newAttributes.get(attrName)
        if (!attr.isDefined) {
          val newAttr = NumericAttribute(attrName)
          newAttributes += attrName -> newAttr
        }
        attrName -> dblToAV(occurence.size)
    }.toSeq
  }

}

private class ProblemMultiMapper[+VT <: AttributeValue, +AD <: Attribute](override val attr: Symbol, val newAttributes: Seq[AD], f: AttributeValue => Seq[(Symbol, VT)], val newLabel: Option[Symbol] = None)
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

private class ProblemMapping[+AV <: AttributeValue, +AD <: Attribute](override val attr: Symbol, val newAttr: AD, f: AttributeValue => AV)(implicit comp: Compatible[AD, AV])
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

